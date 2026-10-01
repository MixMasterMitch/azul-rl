"""Matched training arms for extra Dirichlet noise in Gumbel tree targets."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from agent.eval.arena import checkpoint_hash, paired_interval, write_report
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


def paired_difference(candidate: dict, control: dict) -> dict:
    """Compare independent seed pairs under the same evaluation protocol."""
    for key in (
        "num_games",
        "seed",
        "max_turns",
        "game_batch_size",
        "inference_device",
        "search",
        "greedy",
        "opponent_greedy",
        "split",
        "bot_workers",
    ):
        if candidate["config"][key] != control["config"][key]:
            raise ValueError(f"Unmatched evaluation setting: {key}")
    for key in ("opponent", "opponent_sha256", "opponent_identity", "opponent_search"):
        if candidate[key] != control[key]:
            raise ValueError(f"Unmatched evaluation opponent: {key}")
    if candidate["summary"]["unfinished"] or control["summary"]["unfinished"]:
        raise ValueError("Noise comparison requires finished games")
    if candidate["config"]["num_games"] < 2 or candidate["config"]["num_games"] % 2:
        raise ValueError("Noise comparison requires complete evaluation seed pairs")

    def identities(report: dict) -> list[tuple[int, int]]:
        return [(r["pair_seed"], r["candidate_seat"]) for r in report["records"]]

    expected = [
        (candidate["config"]["seed"] + i // 2, i % 2)
        for i in range(candidate["config"]["num_games"])
    ]
    if identities(candidate) != expected or identities(control) != expected:
        raise ValueError("Unmatched or incomplete evaluation seed pairs")
    if any(
        r["outcome"] == "unfinished" or r["match_score"] not in (0.0, 0.5, 1.0)
        for report in (candidate, control)
        for r in report["records"]
    ):
        raise ValueError("Noise comparison requires finished games with valid scores")
    differences = []
    for offset in range(0, len(expected), 2):
        differences.append(
            sum(
                candidate["records"][i]["match_score"]
                - control["records"][i]["match_score"]
                for i in (offset, offset + 1)
            )
            / 2
        )
    return {
        "match_score_difference": sum(differences) / len(differences),
        "paired_ci95": paired_interval(differences),
        "seed_pairs": len(differences),
    }


def run_noise_study(
    campaign: Campaign, initializer: str, minutes: float = 60.0
) -> dict:
    """Train both arms, save all eight screens, then record a development recommendation."""
    from agent.scripts.competitive import tree_distill_config

    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Training minutes must be finite and positive")
    source = Path(initializer).resolve()
    model, payload = load_net_from_checkpoint(source, "cpu")
    if (
        model.arch != "source_attn"
        or model.hidden != 256
        or payload.get("trained_player_counts") != [2]
    ):
        raise ValueError(
            "Noise study requires the width-256, two-player source_attn finalist"
        )
    del model, payload
    frozen = campaign.root / "initializers" / "frozen_finalist.pt"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        if checkpoint_hash(temporary) != checkpoint_hash(source):
            raise ValueError("Initializer changed while freezing the checkpoint")
        temporary.replace(frozen)
    if checkpoint_hash(frozen) != checkpoint_hash(source):
        raise ValueError("Frozen initializer changed; use a new campaign root")

    control = tree_distill_config(
        campaign.root,
        f"noise_control_seed{campaign.seed}",
        str(frozen),
        seed=campaign.seed,
        minutes=minutes,
        device=campaign.device,
    )
    candidate_id = f"noise_no_dirichlet_seed{campaign.seed}"
    candidate = replace(
        control,
        run_id=candidate_id,
        dirichlet_mix=0.0,
        league_root=str(campaign.root / "experiments" / candidate_id / "league"),
    )
    configs = {"control": control, "no_dirichlet": candidate}
    search = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=64,
        cpu_workers=1,
        temperature=0.25,
        q_scale=28.0,
        root_noise_scale=1.0,
    )
    evaluation_seed = campaign.seed + 30_000_000
    plan = {
        "initializer": str(frozen),
        "initializer_sha256": checkpoint_hash(frozen),
        "minutes_per_arm": minutes,
        "training_seed": campaign.seed,
        "arms": {name: asdict(cfg) for name, cfg in configs.items()},
        "evaluation_seed": evaluation_seed,
        "games_per_screen": 256,
        "evaluation_search": asdict(search),
        "opponents": ["astra", str(frozen)],
        "modes": ["tree64", "greedy"],
        "frozen_opponent_uses_matching_mode": True,
        "primary_metric": "tree64 match score against frozen finalist",
        "astra_regression_tolerance": 0.03,
        "initialization": "same weights; fresh optimizer, replay and isolated league per arm",
        "automatic_promotion": False,
    }
    plan_path = campaign.root / "noise_ablation_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Noise study protocol changed; use a new campaign root")
    write_report(plan_path, plan)

    checkpoints = {name: campaign._train(cfg) for name, cfg in configs.items()}
    reports: dict[str, dict] = {name: {} for name in configs}
    # Interleave arms within each screen so results survive independently and
    # both candidates face identical seeds, settings, and opponent identities.
    for mode in ("tree64", "greedy"):
        for opponent_name, opponent in (("frozen", str(frozen)), ("astra", "astra")):
            for name, checkpoint in checkpoints.items():
                arena = replace(
                    campaign.arena(256, search=search, greedy=mode == "greedy"),
                    seed=evaluation_seed,
                    opponent_greedy=mode == "greedy" and opponent_name == "frozen",
                )
                label = f"{mode}_{opponent_name}"
                report = campaign._match(
                    f"noise_{name}_{label}", checkpoint, opponent, arena, search
                )
                if report["summary"]["unfinished"]:
                    raise RuntimeError(f"Unfinished games in {name}/{label}")
                reports[name][label] = report
                write_report(
                    campaign.root / "noise_ablation_progress.json",
                    {
                        "checkpoints": checkpoints,
                        "screens": {
                            arm: {key: r["summary"] for key, r in screens.items()}
                            for arm, screens in reports.items()
                        },
                    },
                )
    differences = {
        label: paired_difference(
            reports["no_dirichlet"][label], reports["control"][label]
        )
        for label in reports["control"]
    }
    primary = differences["tree64_frozen"]
    astra = differences["tree64_astra"]
    candidate_leads = (
        primary["match_score_difference"] > 0
        and astra["match_score_difference"] >= -0.03
    )
    result = {
        "plan": plan,
        "checkpoints": checkpoints,
        "screens": {
            arm: {key: r["summary"] for key, r in screens.items()}
            for arm, screens in reports.items()
        },
        "candidate_minus_control": differences,
        "recommended_arm": "no_dirichlet" if candidate_leads else "control",
        "primary_difference_resolved": primary["paired_ci95"][0] > 0
        or primary["paired_ci95"][1] < 0,
        "decision_scope": "Development screen; recommend the primary-metric leader subject to the Astra guard.",
        "automatic_promotion": False,
        "provenance": campaign.code,
    }
    write_report(campaign.root / "noise_ablation.json", result)
    campaign.status(
        "noise_ablation_complete",
        recommended_arm=result["recommended_arm"],
        differences=differences,
    )
    return result
