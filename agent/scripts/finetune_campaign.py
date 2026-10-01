"""A bounded 24-hour comparison of learning rate and self-play teacher depth."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

from agent.env.outcomes import REWARD_SEMANTICS_VERSION
from agent.eval.arena import checkpoint_hash, write_report
from agent.eval.builtin_opponents import builtin_identity
from agent.scripts.enhancement_campaign import _verify
from agent.scripts.league_campaign import (
    PANEL,
    SEARCH,
    choose_checkpoint,
    fork_training_state,
    freeze_file,
    panel_difference,
    panel_score,
)
from agent.scripts.lr_campaign import (
    TrainingWindowExhausted,
    retire_replay,
    train_segment,
)
from agent.scripts.noise_study import paired_difference
from agent.train.checkpointing import (
    load_checkpoint_payload,
    load_net_from_checkpoint,
    save_checkpoint,
    save_checkpoint_payload,
)
from agent.train.loop import LoopConfig

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign

ARMS = ("current", "lower_lr", "deeper_teacher")
VARIATIONS = {
    "current": {},
    "lower_lr": {"lr": 0.0003},
    "deeper_teacher": {"selfplay_full_fraction": 1.0},
}
COMMON_MITIGATION = {"search_inference_cache_size": 0}
DIAGNOSTIC_ENVIRONMENT = {
    "PYTHONMALLOC": "debug",
    "PYTHONFAULTHANDLER": "1",
    "MALLOC_PERTURB_": "165",
}


def fork_variant(
    source: Path, league: Path, directory: Path, config: LoopConfig, changes: dict
) -> dict:
    """Fork all state, changing only explicit hyperparameters and run locations.

    Build in a staging directory so a crash cannot leave an unmarked usable fork.
    Adam's moments/steps and all RNG/replay contents survive a learning-rate change.
    """
    if any(key not in {"lr", "selfplay_full_fraction"} for key in changes):
        raise ValueError("Unsupported experiment override")
    identity = {
        "source_sha256": checkpoint_hash(source),
        "changes": changes,
        "common_mitigation": COMMON_MITIGATION,
        "config": asdict(config),
    }
    marker = directory / "variant.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["identity"] != identity:
            raise ValueError("Experiment fork configuration changed")
        return record
    if directory.exists():
        raise ValueError("Cannot replace an unmarked experiment directory")
    staging = directory.with_name(directory.name + ".variant-staging")
    if staging.exists():
        shutil.rmtree(staging)
    fork_training_state(source, league, staging)
    payload = load_checkpoint_payload(source)
    old = payload["config"]
    allowed = {
        "run_id",
        "runs_root",
        "league_root",
        "init_from",
        "provenance",
        "device",
        "max_wall_minutes",
        *changes,
        *COMMON_MITIGATION,
    }
    unexpected = [
        key
        for key, value in asdict(config).items()
        if key in old and old[key] != value and key not in allowed
    ]
    if unexpected or any(
        getattr(config, k) != v for k, v in {**COMMON_MITIGATION, **changes}.items()
    ):
        raise ValueError(f"Undeclared experiment changes: {unexpected}")
    payload["config"] = asdict(config)
    if "lr" in changes:
        for group in payload["optimizer_state_dict"]["param_groups"]:
            group["lr"] = changes["lr"]
            if "initial_lr" in group:
                group["initial_lr"] = changes["lr"]
    if any(
        group["lr"] != config.lr
        for group in payload["optimizer_state_dict"]["param_groups"]
    ):
        raise ValueError("Optimizer learning rate does not match the experiment")
    target = staging / "checkpoints/latest_resume.pt"
    save_checkpoint_payload(target, payload)
    record = {
        "identity": identity,
        "fork_sha256": checkpoint_hash(target),
        "iteration": payload["iteration"],
        "training_wall_s": payload["progress"]["training_wall_s"],
        "optimizer_learning_rates": [
            g["lr"] for g in payload["optimizer_state_dict"]["param_groups"]
        ],
    }
    del payload
    write_report(staging / "variant.json", record)
    staging.rename(directory)
    return record


def prepare(campaign: Campaign, initializer: str) -> dict:
    source = Path(initializer).resolve()
    marker = campaign.root / "finetune_prepared.json"
    if marker.exists():
        prepared = json.loads(marker.read_text())
        if (
            str(source) != prepared["source_resume"]
            or checkpoint_hash(source) != prepared["source_resume_sha256"]
        ):
            raise ValueError("Source resume checkpoint changed")
        for path, digest in prepared["input_hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return prepared
    net, payload = load_net_from_checkpoint(source)
    if (
        net.arch != "source_attn"
        or net.hidden != 256
        or payload.get("trained_player_counts") != [2]
        or payload.get("reward_semantics_version") != REWARD_SEMANTICS_VERSION
        or not all(
            k in payload
            for k in ("optimizer_state_dict", "buffer", "rng_state", "config")
        )
    ):
        raise ValueError(
            "Fine-tuning requires a complete current two-player width-256 checkpoint"
        )
    original = LoopConfig(**payload["config"])
    if (
        original.selfplay_sims != 64
        or original.selfplay_full_sims != 256
        or original.selfplay_full_fraction != 0.25
    ):
        raise ValueError(
            "Expected the champion mixed 64/256-simulation training recipe"
        )
    base_wall = payload["progress"]["training_wall_s"]
    iteration = payload["iteration"]
    frozen = campaign.root / "initializers"
    freeze_file(source, frozen / "source_resume.pt")
    if not (frozen / "start.pt").exists():
        save_checkpoint(
            frozen / "start.pt",
            net,
            iteration=iteration,
            config=payload["config"],
            progress=payload["progress"],
        )
    freeze_file(Path(original.init_from).resolve(), frozen / "previous.pt")
    del net, payload
    league_snapshot = frozen / "league"
    source_league = Path(original.league_root)
    manifest = json.loads((source_league / "league.json").read_text())
    for entry in manifest["entries"]:
        if entry.get("active", True):
            freeze_file(
                source_league / entry["path"],
                league_snapshot / Path(entry["path"]).name,
            )
        entry["path"] = Path(entry["path"]).name
    write_report(league_snapshot / "league.json", manifest)
    configs, forks = {}, {}
    for arm in ARMS:
        name = f"{arm}_seed{original.seed}"
        directory = campaign.root / "experiments" / name
        config = replace(
            original,
            **COMMON_MITIGATION,
            **VARIATIONS[arm],
            run_id=name,
            runs_root=str(campaign.root / "experiments"),
            league_root=str(directory / "league"),
            init_from=str(frozen / "start.pt"),
            provenance=None,
            device=campaign.device,
            max_wall_minutes=base_wall / 60 + 780,
        )
        forks[arm] = fork_variant(
            frozen / "source_resume.pt",
            league_snapshot,
            directory,
            config,
            VARIATIONS[arm],
        )
        configs[arm] = asdict(config)
    prepared = {
        "source_resume": str(source),
        "source_resume_sha256": checkpoint_hash(source),
        "base_training_wall_s": base_wall,
        "initial_iteration": iteration,
        "arms": configs,
        "forks": forks,
        "frozen": {name: str(frozen / f"{name}.pt") for name in ("start", "previous")},
        "input_hashes": {str(p): checkpoint_hash(p) for p in frozen.rglob("*.pt")},
    }
    write_report(marker, prepared)
    return prepared


def choose_continuation(reports: dict[str, dict], baseline: dict) -> dict:
    """Allocate research time on development scores; confidence gates promotion only."""
    baseline_astra = baseline["astra"]["summary"]["match_score"]
    eligible = [
        arm
        for arm in ARMS
        if reports[arm]["astra"]["summary"]["match_score"] >= baseline_astra - 0.03
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda arm: (
                panel_score(reports[arm]),
                reports[arm]["astra"]["summary"]["match_score"],
                -ARMS.index(arm),
            ),
        )
        reason = "Highest development panel score among arms within 3pp of starting Astra score."
    else:
        selected = max(
            ARMS,
            key=lambda arm: (
                reports[arm]["astra"]["summary"]["match_score"],
                panel_score(reports[arm]),
                -ARMS.index(arm),
            ),
        )
        reason = "All pilots missed the Astra guard; investigate the least Astra regression. No promotion implied."
    return {
        "selected_arm": selected,
        "eligible_arms": eligible,
        "reason": reason,
        "panel_scores": {arm: panel_score(reports[arm]) for arm in ARMS},
        "versus_control": {
            arm: panel_difference(reports[arm], reports["current"]) for arm in ARMS[1:]
        },
    }


def _budget(campaign: Campaign) -> dict:
    path = campaign.root / "finetune_campaign_budget.json"
    if not path.exists():
        now = time.time()
        budget = {
            "started_at": now,
            "deadline": now + min(24 * 3600, campaign.deadline - time.monotonic()),
        }
        for name in ("preflight_budget.json", "supervisor.json"):
            other_path = campaign.root / name
            if other_path.exists():
                other = json.loads(other_path.read_text())
                budget = {k: min(budget[k], other[k]) for k in budget}
        write_report(path, budget)
    budget = json.loads(path.read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    return budget


def _report(root: Path, result: dict, budget: dict) -> None:
    finished = datetime.fromtimestamp(
        budget["started_at"] + result["elapsed_s"], timezone.utc
    ).isoformat()
    rows = [
        "# 24-hour two-player fine-tuning campaign",
        "",
        f"Finished {finished}, after {result['elapsed_s'] / 3600:.2f} hours including preparation.",
        "",
        f"Continuation arm: **{result['selected_arm']}**. {result['arm_decision']['reason']}",
        "",
        f"New model passed independent confirmation: **{result['improvement_resolved']}**.",
        f"Selected weights: [{Path(result['selected_checkpoint']).name}]({result['selected_checkpoint']}).",
        "",
        "| Development checkpoint | Panel score | Astra score |",
        "|---|---:|---:|",
    ]
    for label, values in result["development_scores"].items():
        rows.append(
            f"| {label} | {values['panel_score']:.1%} | {values['astra_score']:.1%} |"
        )
    if result["confirmation_reused_start"]:
        rows += [
            "",
            "The starting checkpoint won development selection. It was evaluated once; "
            "duplicate candidate games were skipped. There is no independent candidate/start difference.",
        ]
    rows += [
        "",
        "| Held-out opponent | Evaluated candidate | Starting model |",
        "|---|---:|---:|",
    ]
    for opponent in PANEL:
        rows.append(
            f"| {opponent} | {result['screens'][opponent]['match_score']:.1%} | "
            f"{result['start_screens'][opponent]['match_score']:.1%} |"
        )
    if result["panel_difference"] is not None:
        delta = result["panel_difference"]
        lo, hi = delta["paired_ci95"]
        rows += [
            "",
            f"Panel gain: {delta['match_score_difference'] * 100:+.1f}pp "
            f"(paired 95% interval {lo * 100:+.1f} to {hi * 100:+.1f}pp).",
        ]
    rows += [
        "",
        f"Greedy Astra score: candidate {result['greedy']['candidate']['match_score']:.1%}; "
        f"start {result['greedy']['start']['match_score']:.1%}.",
        "",
        "Match score is win=1, shared victory=0.5, loss=0; training rewards remain +1/0/-1. "
        "Primary matches use identical 64-simulation Rust search. "
        "Pilots start from the same weights, optimizer moments, replay, RNG, and league; "
        "The optional inference cache is disabled in every arm as a common crash mitigation; "
        "the experimental difference is learning rate or teacher fraction. This is one training seed.",
        "",
        f"Resumable continuation state: `{result['resume_checkpoint']}`. It may differ from selected weights. "
        "Source state and historical models were preserved. No deployment occurred.",
        "",
        f"Recovered child failures: {len(result['recovered_failures'])}.",
        "",
        "[Results](finetune_campaign.json) · [Verification](verification.json) · [Plan](finetune_campaign_plan.json)",
    ]
    (root / "REPORT.md").write_text("\n".join(rows) + "\n")


def run_finetune_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = _budget(campaign)
    reliability = json.loads((campaign.root / "reliability.json").read_text())
    if not reliability.get("cleared_for_training"):
        raise RuntimeError(
            "The memory-corruption investigation has not cleared this campaign"
        )
    if reliability.get("require_debug_allocator") and any(
        os.environ.get(key) != value for key, value in DIAGNOSTIC_ENVIRONMENT.items()
    ):
        raise RuntimeError(
            "Required native-memory diagnostics are not enabled in this process"
        )
    prepared = prepare(campaign, initializer)
    configs = {name: LoopConfig(**cfg) for name, cfg in prepared["arms"].items()}
    plan = {
        "study": "finetune",
        "campaign_hours": 24,
        **prepared,
        "initializer": prepared["frozen"]["start"],
        "initializer_sha256": checkpoint_hash(prepared["frozen"]["start"]),
        "pilot_minutes_per_arm": 180,
        "continuation_cumulative_minutes": [330, 480, 630, 780],
        "continuation_window_minutes": 600,
        "continuation_includes_progress_screens": True,
        "development_games_per_opponent": 512,
        "confirmation_games_per_opponent": 2048,
        "greedy_games": 512,
        "evaluation_panel": list(PANEL),
        "evaluation_search": asdict(SEARCH),
        "development_seed": campaign.seed + 130_000_000,
        "confirmation_seed": campaign.seed + 140_000_000,
        "greedy_seed": campaign.seed + 150_000_000,
        "final_evaluation_reserve_minutes": 180,
        "last_screen_reserve_minutes": 15,
        "selection": "Highest development panel score within Astra guard, without requiring pilot significance.",
        "checkpoint_selection": "Highest panel score with Astra within 3pp of start; all pilots and start eligible.",
        "confirmation": "Promote only if paired panel CI lower bound > 0 and Astra delta >= -0.03.",
        "astra_identity": builtin_identity("astra"),
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "bot_workers": campaign.bot_workers,
        "automatic_promotion": False,
        "provenance": campaign.code,
        "common_mitigation": COMMON_MITIGATION,
        "reliability": reliability,
        "diagnostic_environment": {
            key: os.environ.get(key) for key in DIAGNOSTIC_ENVIRONMENT
        },
    }
    plan_path = campaign.root / "finetune_campaign_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Fine-tuning campaign protocol changed")
    write_report(plan_path, plan)
    final_path = campaign.root / "finetune_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        write_report(
            campaign.root / "verification.json", _verify(campaign.root, result)
        )
        _report(campaign.root, result, budget)
        campaign.status(
            "finetune_campaign_complete",
            selected_checkpoint=result["selected_checkpoint"],
        )
        return result

    def panel(label: str, checkpoint: str, confirmation: bool = False) -> dict:
        reports = {}
        for name in PANEL:
            opponent = "astra" if name == "astra" else prepared["frozen"][name]
            cfg = replace(
                campaign.arena(
                    2048 if confirmation else 512,
                    search=SEARCH,
                    split="confirmation" if confirmation else "development",
                ),
                seed=plan["confirmation_seed" if confirmation else "development_seed"],
            )
            reports[name] = campaign._match(
                f"{label}_{name}", checkpoint, opponent, cfg, SEARCH
            )
            if reports[name]["summary"]["unfinished"]:
                raise RuntimeError("Unfinished evaluation games")
        return reports

    def train(
        arm: str, minutes: float, label: str, continuation_deadline: float | None = None
    ) -> dict:
        original_deadline = campaign.deadline
        if continuation_deadline is not None:
            # train_segment subtracts its reserve; reserve the final progress screen
            # inside the ten-hour continuation window as well.
            window_end = time.monotonic() + continuation_deadline - time.time()
            campaign.deadline = min(original_deadline, window_end + 180 * 60)
        try:
            milestone = train_segment(
                campaign,
                configs[arm],
                prepared["base_training_wall_s"] / 60 + minutes,
                label,
                evaluation_reserve_minutes=195,
            )
        finally:
            campaign.deadline = original_deadline
        return {
            **milestone,
            "additional_training_minutes": (
                milestone["training_wall_s"] - prepared["base_training_wall_s"]
            )
            / 60,
        }

    pilots = {arm: train(arm, 180, "pilot") for arm in ARMS}
    baseline = panel("development_start", prepared["frozen"]["start"])
    reports = {arm: panel(f"pilot_{arm}", m["checkpoint"]) for arm, m in pilots.items()}
    decision = choose_continuation(reports, baseline)
    selection_path = campaign.root / "arm_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != decision:
        raise ValueError("Continuation selection changed")
    write_report(selection_path, decision)
    selected = decision["selected_arm"]
    for arm in ARMS:
        if arm != selected:
            retire_replay(configs[arm], pilots[arm])
            pilots[arm]["replay_retained"] = False
    campaign.status("finetune_campaign_selected", **decision)
    candidates = {
        "initializer": baseline,
        **{f"pilot_{arm}": r for arm, r in reports.items()},
    }
    paths = {
        "initializer": prepared["frozen"]["start"],
        **{f"pilot_{arm}": m["checkpoint"] for arm, m in pilots.items()},
    }
    last = dict(pilots)
    window_path = campaign.root / "continuation_budget.json"
    if not window_path.exists():
        write_report(
            window_path,
            {
                "started_at": time.time(),
                "deadline": min(time.time() + 600 * 60, budget["deadline"] - 180 * 60),
            },
        )
    window = json.loads(window_path.read_text())
    limited = False
    for minutes in plan["continuation_cumulative_minutes"]:
        label = f"minutes_{minutes:04d}"
        try:
            milestone = train(selected, minutes, label, window["deadline"])
        except TrainingWindowExhausted:
            limited = True
            break
        last[selected] = milestone
        paths[label] = milestone["checkpoint"]
        candidates[label] = panel(label, paths[label])
        write_report(
            campaign.root / "finetune_campaign_progress.json",
            {
                "selected_arm": selected,
                "last_milestones": last,
                "candidates": {
                    k: {
                        "checkpoint": paths[k],
                        "panel_score": panel_score(v),
                        "screens": {n: r["summary"] for n, r in v.items()},
                    }
                    for k, v in candidates.items()
                },
            },
        )
        if milestone["executed_target_minutes"] < milestone["target_minutes"]:
            limited = True
            break
    best = choose_checkpoint(candidates, baseline["astra"]["summary"]["match_score"])
    checkpoint = paths[best]
    frozen_selection = {
        "selected_arm": selected,
        "candidate_milestone": best,
        "candidate_checkpoint": checkpoint,
        "candidate_sha256": checkpoint_hash(checkpoint),
    }
    selection_path = campaign.root / "final_selection.json"
    if (
        selection_path.exists()
        and json.loads(selection_path.read_text()) != frozen_selection
    ):
        raise ValueError("Candidate changed after confirmation began")
    write_report(selection_path, frozen_selection)
    same = best == "initializer"
    reference = panel("confirmation_start", prepared["frozen"]["start"], True)
    final = reference if same else panel("confirmation_candidate", checkpoint, True)
    delta = None if same else panel_difference(final, reference)
    improved = (
        delta is not None
        and delta["paired_ci95"][0] > 0
        and delta["opponents"]["astra"]["match_score_difference"] >= -0.03
    )
    greedy_cfg = replace(
        campaign.arena(512, search=SEARCH, greedy=True, split="confirmation"),
        seed=plan["greedy_seed"],
    )
    greedy_start = campaign._match(
        "greedy_start_astra", prepared["frozen"]["start"], "astra", greedy_cfg, SEARCH
    )
    greedy_final = (
        greedy_start
        if same
        else campaign._match(
            "greedy_candidate_astra", checkpoint, "astra", greedy_cfg, SEARCH
        )
    )
    if any(r["summary"]["unfinished"] for r in (greedy_start, greedy_final)):
        raise RuntimeError("Unfinished greedy evaluation games")
    retained = checkpoint if improved else prepared["frozen"]["start"]
    resume = (
        Path(configs[selected].runs_root)
        / configs[selected].run_id
        / "checkpoints/latest_resume.pt"
    )
    failures = []
    event_path = campaign.root / "supervisor_events.jsonl"
    if event_path.exists():
        failures = [
            e
            for line in event_path.read_text().splitlines()
            if (e := json.loads(line)).get("event") == "child_exited"
            and e.get("returncode") != 0
        ]
    result = {
        "plan": plan,
        **frozen_selection,
        "selected_milestone": best if improved else "initializer",
        "selected_checkpoint": retained,
        "checkpoint_sha256": checkpoint_hash(retained),
        "arm_decision": decision,
        "confirmation_reused_start": same,
        "screens": {k: v["summary"] for k, v in final.items()},
        "start_screens": {k: v["summary"] for k, v in reference.items()},
        "panel_difference": delta,
        "last_milestones": last,
        "development_scores": {
            k: {
                "panel_score": panel_score(v),
                "astra_score": v["astra"]["summary"]["match_score"],
            }
            for k, v in candidates.items()
        },
        "greedy": {
            "candidate": greedy_final["summary"],
            "start": greedy_start["summary"],
            "difference": None
            if same
            else paired_difference(greedy_final, greedy_start),
        },
        "improvement_resolved": improved,
        "recovered_failures": failures,
        "resume_checkpoint": str(resume),
        "resume_checkpoint_sha256": checkpoint_hash(resume),
        "training_budget_limited": limited,
        "automatic_promotion": False,
        "elapsed_s": time.time() - budget["started_at"],
    }
    verification = _verify(campaign.root, result)
    verification["source_resume_hash_matches"] = (
        checkpoint_hash(initializer) == prepared["source_resume_sha256"]
    )
    verification["recovered_failures"] = failures
    if not verification["source_resume_hash_matches"]:
        raise RuntimeError("Original resume checkpoint changed")
    write_report(campaign.root / "verification.json", verification)
    write_report(final_path, result)
    _report(campaign.root, result, budget)
    campaign.status(
        "finetune_campaign_complete",
        selected_arm=selected,
        selected_checkpoint=retained,
        improvement_resolved=improved,
    )
    return result
