"""Bounded learning-rate comparison followed by resumable two-player training."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

from agent.eval.arena import checkpoint_hash, write_report
from agent.eval.builtin_opponents import builtin_identity
from agent.obs.run import Run
from agent.scripts.noise_study import paired_difference
from agent.search.config import SearchConfig
from agent.train.checkpointing import (
    load_checkpoint_payload,
    load_net_from_checkpoint,
    save_checkpoint,
)
from agent.train.league import League
from agent.train.loop import LoopConfig, run_loop

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


class TrainingWindowExhausted(TimeoutError):
    """The remaining wall budget is reserved for final evaluation."""


def train_segment(
    campaign: Campaign,
    config: LoopConfig,
    minutes: float,
    label: str,
    *,
    evaluation_reserve_minutes: float = 65.0,
) -> dict:
    """Reach a cumulative training budget, retaining the full state between stages."""
    directory = Path(config.runs_root) / config.run_id
    identity = {
        "config": asdict(config),
        "initializer_sha256": checkpoint_hash(config.init_from),
    }
    budget = directory / "segmented_training.json"
    if budget.exists() and json.loads(budget.read_text()) != identity:
        raise ValueError("Segmented training configuration changed; use a new run ID")
    write_report(budget, identity)
    marker = directory / "milestones" / f"{label}.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["target_minutes"] != minutes:
            raise ValueError("Training milestone budget changed")
        if checkpoint_hash(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError("Training milestone checkpoint changed")
        return record
    resume = directory / "checkpoints/latest_resume.pt"
    if any((directory / "milestones").glob("*.json")) and not resume.exists():
        raise ValueError(
            "Continuation requires the retained optimizer and replay checkpoint"
        )
    prior_wall_s = 0.0
    if resume.exists():
        payload = load_checkpoint_payload(resume)
        prior_wall_s = payload["progress"]["training_wall_s"]
        del payload
    if evaluation_reserve_minutes < 0:
        raise ValueError("Evaluation reserve must be nonnegative")
    available_s = campaign.deadline - time.monotonic() - evaluation_reserve_minutes * 60
    if available_s <= 0:
        raise TrainingWindowExhausted(
            "Remaining campaign time is reserved for final evaluation"
        )
    target_minutes = min(minutes, (prior_wall_s + available_s) / 60)
    if label == "pilot" and target_minutes < minutes:
        raise TrainingWindowExhausted("Insufficient time for a full matched pilot")
    league = League(config.league_root)
    if not league.list_entries():
        net, _ = load_net_from_checkpoint(config.init_from)
        net.trained_player_counts = [2]
        league.add_checkpoint(net, "frozen_baseline", metadata={"pinned": True})
        del net
    segment = replace(config, max_wall_minutes=target_minutes)
    campaign.status(
        "training",
        run_id=config.run_id,
        milestone=label,
        target_minutes=target_minutes,
        config=asdict(segment),
        heartbeat=str(directory / "heartbeat.json"),
    )
    run = Run(config.run_id, runs_root=config.runs_root)
    try:
        result = run_loop(run, segment, explicit_fields=set(asdict(segment)))
    finally:
        run.close()
    if result.get("stopped"):
        raise KeyboardInterrupt
    if result["training_wall_s"] < target_minutes * 60 - 0.01:
        raise RuntimeError("Training ended before its cumulative milestone budget")
    net, payload = load_net_from_checkpoint(resume)
    if not all(
        key in payload for key in ("optimizer_state_dict", "buffer", "rng_state")
    ):
        raise RuntimeError("Training checkpoint lacks optimizer, replay, or RNG state")
    snapshot = directory / "milestones" / f"{label}.pt"
    save_checkpoint(
        snapshot,
        net,
        iteration=payload["iteration"],
        config=payload["config"],
        progress=payload["progress"],
    )
    record = {
        "checkpoint": str(snapshot),
        "checkpoint_sha256": checkpoint_hash(snapshot),
        "iteration": payload["iteration"],
        "training_wall_s": payload["progress"]["training_wall_s"],
        "target_minutes": minutes,
        "executed_target_minutes": target_minutes,
        "replay_retained": True,
    }
    del net, payload
    write_report(marker, record)
    return record


def choose_learning_rate(reports: dict[str, dict]) -> dict:
    """Keep the current rate unless the lower rate has a resolved primary advantage."""
    differences = {
        label: paired_difference(reports["lower"][label], reports["current"][label])
        for label in reports["current"]
    }
    primary = differences["tree64_frozen"]
    astra = differences["tree64_astra"]
    use_lower = (
        primary["paired_ci95"][0] > 0 and astra["match_score_difference"] >= -0.03
    )
    return {
        "selected_arm": "lower" if use_lower else "current",
        "lower_minus_current": differences,
        "reason": (
            "Resolved advantage against the frozen initializer without an Astra regression."
            if use_lower
            else "Lower rate did not establish an advantage; retain current rate."
        ),
    }


def choose_checkpoint(candidates: dict[str, dict], baseline_astra: float) -> str:
    """Select on development games; keep the initializer available as a fallback."""
    eligible = [
        name
        for name, candidate in candidates.items()
        if candidate["screens"]["tree64_astra"]["match_score"] >= baseline_astra - 0.03
    ]
    return max(
        eligible,
        key=lambda name: (
            candidates[name]["screens"]["tree64_frozen"]["match_score"],
            candidates[name]["screens"]["tree64_astra"]["match_score"],
        ),
    )


def retire_replay(config: LoopConfig, milestone: dict) -> None:
    """Retire only this campaign's no-longer-needed replay after freezing its weights."""
    if checkpoint_hash(milestone["checkpoint"]) != milestone["checkpoint_sha256"]:
        raise ValueError("Cannot retire replay without a verified weight snapshot")
    directory = Path(config.runs_root) / config.run_id
    (directory / "checkpoints/latest_resume.pt").unlink(missing_ok=True)
    write_report(
        directory / "state.json",
        {
            "iter": milestone["iteration"],
            "last_checkpoint": milestone["checkpoint"],
            "replay_retained": False,
            "experiment_complete": True,
        },
    )


def run_lr_campaign(campaign: Campaign, initializer: str) -> dict:
    """Compare two one-hour arms, continue one for 6.5 hours, and evaluate held out."""
    from agent.scripts.competitive import tree_distill_config

    source = Path(initializer).resolve()
    model, payload = load_net_from_checkpoint(source)
    if (
        model.arch != "source_attn"
        or model.hidden != 256
        or payload.get("trained_player_counts") != [2]
    ):
        raise ValueError(
            "LR campaign requires a width-256, two-player source_attn initializer"
        )
    del model, payload
    frozen = campaign.root / "initializers/frozen_start.pt"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        if checkpoint_hash(temporary) != checkpoint_hash(source):
            raise ValueError("Initializer changed while freezing it")
        temporary.replace(frozen)
    if checkpoint_hash(frozen) != checkpoint_hash(source):
        raise ValueError("Frozen initializer differs from the requested checkpoint")
    current = replace(
        tree_distill_config(
            campaign.root,
            f"lr_current_seed{campaign.seed}",
            str(frozen),
            seed=campaign.seed,
            minutes=450,
            device=campaign.device,
        ),
        dirichlet_mix=0.0,
    )
    lower_id = f"lr_lower_seed{campaign.seed}"
    configs = {
        "current": current,
        "lower": replace(
            current,
            run_id=lower_id,
            lr=0.0003,
            league_root=str(campaign.root / "experiments" / lower_id / "league"),
        ),
    }
    search = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        cpu_workers=1,
        num_simulations=64,
        temperature=0.25,
        q_scale=28.0,
        root_noise_scale=1.0,
    )
    development_seed = campaign.seed + 40_000_000
    confirmation_seed = campaign.seed + 50_000_000
    plan = {
        "initializer": str(frozen),
        "initializer_sha256": checkpoint_hash(frozen),
        "arms": {name: asdict(cfg) for name, cfg in configs.items()},
        "pilot_minutes_per_arm": 60,
        "continuation_cumulative_minutes": [180, 300, 450],
        "evaluation_search": asdict(search),
        "development_seed": development_seed,
        "confirmation_seed": confirmation_seed,
        "development_games": 256,
        "confirmation_tree_games": 1024,
        "confirmation_greedy_games": 256,
        "bot_workers": campaign.bot_workers,
        "astra_regression_tolerance": 0.03,
        "selection": "Lower LR only if its paired frozen-opponent CI is above zero and Astra delta >= -0.03.",
        "checkpoint_selection": "Best development score against frozen start, within 3pp of starting Astra score.",
        "automatic_promotion": False,
        "astra_identity": builtin_identity("astra"),
        "provenance": campaign.code,
    }
    plan_path = campaign.root / "lr_campaign_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("LR campaign protocol changed; use a new campaign root")
    write_report(plan_path, plan)

    # Restarts inherit the original absolute deadline, including time offline.
    budget_path = campaign.root / "lr_campaign_budget.json"
    if not budget_path.exists():
        supervisor = campaign.root / "supervisor.json"
        deadline = time.time() + max(0, campaign.deadline - time.monotonic())
        if supervisor.exists():
            deadline = min(deadline, json.loads(supervisor.read_text())["deadline"])
        write_report(budget_path, {"started_at": time.time(), "deadline": deadline})
    budget = json.loads(budget_path.read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    final_path = campaign.root / "lr_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        for name, milestone in result["last_milestones"].items():
            retire_replay(configs[name], milestone)
        campaign.status(
            "lr_campaign_complete", selected_checkpoint=result["selected_checkpoint"]
        )
        return result

    def match(
        label: str,
        checkpoint: str,
        opponent: str,
        *,
        greedy: bool = False,
        confirmation: bool = False,
        games: int = 256,
    ) -> dict:
        arena = replace(
            campaign.arena(
                games,
                search=search,
                greedy=greedy,
                split="confirmation" if confirmation else "development",
            ),
            seed=confirmation_seed if confirmation else development_seed,
            opponent_greedy=greedy and opponent != "astra",
        )
        report = campaign._match(label, checkpoint, opponent, arena, search)
        if report["summary"]["unfinished"]:
            raise RuntimeError(f"Unfinished evaluation games in {label}")
        return report

    pilots = {
        name: train_segment(campaign, cfg, 60, "pilot") for name, cfg in configs.items()
    }
    reports: dict[str, dict] = {name: {} for name in configs}
    for mode in ("tree64", "greedy"):
        for opponent_name, opponent in (("frozen", str(frozen)), ("astra", "astra")):
            for name, pilot in pilots.items():
                label = f"{mode}_{opponent_name}"
                reports[name][label] = match(
                    f"pilot_{name}_{label}",
                    pilot["checkpoint"],
                    opponent,
                    greedy=mode == "greedy",
                )
    decision = choose_learning_rate(reports)
    write_report(
        campaign.root / "lr_selection.json",
        {
            **decision,
            "screens": {
                name: {key: value["summary"] for key, value in screens.items()}
                for name, screens in reports.items()
            },
        },
    )
    selected_arm = decision["selected_arm"]
    for name in configs:
        if name != selected_arm:
            retire_replay(configs[name], pilots[name])
    campaign.status("lr_selected", **decision)
    baseline = match("development_start_tree64_astra", str(frozen), "astra")
    candidates = {
        "initializer": {
            "checkpoint": str(frozen),
            "screens": {
                "tree64_frozen": {"match_score": 0.5},
                "tree64_astra": baseline["summary"],
            },
        }
    }
    candidates["pilot"] = {
        "checkpoint": pilots[selected_arm]["checkpoint"],
        "screens": {
            key: value["summary"] for key, value in reports[selected_arm].items()
        },
    }
    last_milestones = dict(pilots)
    training_budget_limited = False
    for minutes in plan["continuation_cumulative_minutes"]:
        label = f"minutes_{minutes:04d}"
        try:
            milestone = train_segment(campaign, configs[selected_arm], minutes, label)
        except TrainingWindowExhausted:
            training_budget_limited = True
            break
        last_milestones[selected_arm] = milestone
        screens = {
            name: match(f"{label}_tree64_{name}", milestone["checkpoint"], opponent)[
                "summary"
            ]
            for name, opponent in (("frozen", str(frozen)), ("astra", "astra"))
        }
        candidates[label] = {
            "checkpoint": milestone["checkpoint"],
            "screens": {f"tree64_{name}": summary for name, summary in screens.items()},
        }
        best = choose_checkpoint(candidates, baseline["summary"]["match_score"])
        write_report(
            campaign.root / "lr_campaign_progress.json",
            {
                "selected_arm": selected_arm,
                "candidates": candidates,
                "best_development_checkpoint": best,
                "last_milestones": last_milestones,
            },
        )
        if milestone.get("executed_target_minutes", minutes) < minutes:
            training_budget_limited = True
            break

    best = choose_checkpoint(candidates, baseline["summary"]["match_score"])
    checkpoint = candidates[best]["checkpoint"]
    # Selection is frozen before any held-out outcomes become available.
    selection = {
        "selected_arm": selected_arm,
        "selected_milestone": best,
        "selected_checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash(checkpoint),
    }
    selection_path = campaign.root / "final_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Selected checkpoint changed after held-out evaluation began")
    write_report(selection_path, selection)
    final = {}
    for label, candidate, opponent, greedy, games in [
        ("tree64_frozen", checkpoint, str(frozen), False, 1024),
        ("tree64_astra", checkpoint, "astra", False, 1024),
        ("start_tree64_astra", str(frozen), "astra", False, 1024),
        ("greedy_astra", checkpoint, "astra", True, 256),
        ("start_greedy_astra", str(frozen), "astra", True, 256),
    ]:
        final[label] = match(
            f"confirmation_{label}",
            candidate,
            opponent,
            greedy=greedy,
            games=games,
            confirmation=True,
        )
        write_report(
            campaign.root / "lr_confirmation_progress.json",
            {
                "selection": selection,
                "screens": {key: value["summary"] for key, value in final.items()},
            },
        )
    differences = {
        mode: paired_difference(final[f"{mode}_astra"], final[f"start_{mode}_astra"])
        for mode in ("tree64", "greedy")
    }
    frozen_score = final["tree64_frozen"]["summary"]
    result = {
        "plan": plan,
        **selection,
        "lr_decision": decision,
        "candidates": candidates,
        "screens": {key: value["summary"] for key, value in final.items()},
        "candidate_minus_start_astra": differences,
        "last_milestones": last_milestones,
        "improvement_resolved": frozen_score["match_score_ci95"][0] > 0.5,
        "automatic_promotion": False,
        "provenance": campaign.code,
        "training_budget_limited": training_budget_limited,
        "elapsed_s": time.time() - budget["started_at"],
    }
    write_report(final_path, result)
    for name, milestone in last_milestones.items():
        retire_replay(configs[name], milestone)
    campaign.status(
        "lr_campaign_complete",
        selected_checkpoint=checkpoint,
        selected_arm=selected_arm,
        improvement_resolved=result["improvement_resolved"],
    )
    return result
