"""Matched enhancement/reanalysis pilots with bounded continuation."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

from agent.env.outcomes import REWARD_SEMANTICS_VERSION
from agent.eval.arena import checkpoint_hash, write_report
from agent.eval.builtin_opponents import builtin_identity
from agent.scripts.lr_campaign import (
    TrainingWindowExhausted,
    choose_checkpoint,
    retire_replay,
    train_segment,
)
from agent.scripts.noise_study import paired_difference
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint
from agent.train.presets import enhanced_2p_config

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign
    from agent.train.loop import LoopConfig


def campaign_configs(campaign: Campaign, initializer: str) -> dict[str, LoopConfig]:
    from agent.scripts.competitive import tree_distill_config

    control = replace(
        tree_distill_config(
            campaign.root,
            f"control_seed{campaign.seed}",
            initializer,
            seed=campaign.seed,
            minutes=290,
            device=campaign.device,
        ),
        dirichlet_mix=0.0,
    )
    name = f"enhanced_seed{campaign.seed}"
    enhanced = replace(
        enhanced_2p_config(),
        run_id=name,
        runs_root=control.runs_root,
        init_from=initializer,
        seed=campaign.seed,
        device=campaign.device,
        max_wall_minutes=290,
        keep_recent_checkpoints=2,
        league_root=str(campaign.root / "experiments" / name / "league"),
    )
    return {"control": control, "enhanced": enhanced}


def reanalysis_configs(campaign: Campaign, initializer: str) -> dict[str, LoopConfig]:
    configs = {}
    for arm, positions in (("current", 256), ("reanalysis4x", 1024)):
        name = f"{arm}_seed{campaign.seed}"
        configs[arm] = replace(
            enhanced_2p_config(),
            run_id=name,
            runs_root=str(campaign.root / "experiments"),
            init_from=initializer,
            seed=campaign.seed,
            device=campaign.device,
            max_wall_minutes=495,
            keep_recent_checkpoints=2,
            reanalysis_positions=positions,
            league_root=str(campaign.root / "experiments" / name / "league"),
        )
    return configs


def choose_arm(
    reports: dict[str, dict], control: str = "control", enhanced: str = "enhanced"
) -> dict:
    differences = {
        name: paired_difference(reports[enhanced][name], reports[control][name])
        for name in reports[control]
    }
    primary = differences["tree64_frozen"]
    astra = differences["tree64_astra"]
    use_enhanced = (
        primary["paired_ci95"][0] > 0 and astra["match_score_difference"] >= -0.03
    )
    return {
        "selected_arm": enhanced if use_enhanced else control,
        f"{enhanced}_minus_{control}": differences,
        "reason": (
            f"{enhanced} has a resolved advantage against the frozen start without an Astra drop over 3pp."
            if use_enhanced
            else f"{enhanced} did not pass both pilot gates; retain {control}."
        ),
    }


def _write_report(root: Path, result: dict, budget: dict) -> None:
    def stamp(epoch: float) -> str:
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")

    plan = result["plan"]
    study = plan.get("study", "enhancement")
    rows = [
        f"# {plan.get('campaign_hours', 8)}-hour two-player {study} campaign",
        "",
        f"Completed in {result['elapsed_s'] / 3600:.2f} hours including setup. "
        f"Started {stamp(budget['started_at'])}; finished {stamp(budget['started_at'] + result['elapsed_s'])}.",
        "",
        f"Selected training arm: **{result['selected_arm']}**. {result['arm_decision']['reason']}",
        "",
        f"Selected checkpoint: [{Path(result['selected_checkpoint']).name}]({result['selected_checkpoint']}). "
        f"Development milestone: `{result['selected_milestone']}`.",
        "",
        "| Held-out evaluation | Games | Match score | 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in result["screens"].items():
        lo, hi = summary["match_score_ci95"]
        rows.append(
            f"| {name} | {summary['games']} | {summary['match_score']:.1%} | {lo:.1%}–{hi:.1%} |"
        )
    rows += [
        "",
        "Match score is win=1, shared victory=0.5, loss=0. "
        "Both training arms use binary values +1/0/−1 and fresh replay/optimizer state. "
        "All primary evaluations use identical 64-simulation Rust search settings.",
        "",
        f"Improvement over the frozen start statistically resolved: **{result['improvement_resolved']}**. "
        + (
            "This is one training seed; the pilot varies only the number of reanalysed positions."
            if study == "reanalysis"
            else "This is one training seed; the pilot compares the enhancement bundle, not individual features."
        ),
        "",
    ]
    for mode, delta in result["candidate_minus_start_astra"].items():
        lo, hi = delta["paired_ci95"]
        rows.append(
            f"Astra {mode} change versus the start: {delta['match_score_difference'] * 100:+.1f}pp "
            f"(paired 95% interval {lo * 100:+.1f} to {hi * 100:+.1f}pp)."
        )
    retention = (
        f"The selected training arm's latest full resume state is retained at `{result['resume_checkpoint']}`. "
        "It corresponds to the last training milestone; the best evaluated weights may be earlier."
        if result.get("resume_checkpoint")
        else "Completed replay files were retired."
    )
    rows += [
        "",
        "Milestone weights, game records, frozen settings, and source provenance are retained. "
        + retention
        + " No model was automatically promoted or deployed.",
        "",
        f"[Machine-readable results]({study}_campaign.json) · [Verification](verification.json)",
    ]
    (root / "REPORT.md").write_text("\n".join(rows) + "\n")


def _verify(root: Path, result: dict) -> dict:
    evaluations = [
        json.loads(p.read_text()) for p in (root / "evaluations").glob("*.json")
    ]
    checkpoints = {result["selected_checkpoint"]: result["checkpoint_sha256"]}
    checkpoints[result["plan"]["initializer"]] = result["plan"]["initializer_sha256"]
    for path in (root / "experiments").glob("*/milestones/*.json"):
        milestone = json.loads(path.read_text())
        checkpoints[milestone["checkpoint"]] = milestone["checkpoint_sha256"]
    for milestone in result["last_milestones"].values():
        checkpoints[milestone["checkpoint"]] = milestone["checkpoint_sha256"]
    hashes = {
        path: checkpoint_hash(path) == digest for path, digest in checkpoints.items()
    }
    if not all(hashes.values()) or any(r["summary"]["unfinished"] for r in evaluations):
        raise RuntimeError("Campaign artifact verification failed")
    learner_steps, skipped, errors = 0, 0, []
    for path in (root / "experiments").glob("*/events.log"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "learner_done":
                learner_steps += event["fields"].get("learner_steps_ok", 0)
                skipped += event["fields"].get("learner_steps_skipped", 0)
            if event.get("lvl") == "ERROR":
                errors.append(event)
    hours = result["plan"].get("campaign_hours", 8)
    resume = result.get("resume_checkpoint")
    if resume and checkpoint_hash(resume) != result["resume_checkpoint_sha256"]:
        raise RuntimeError("Retained resume checkpoint changed")
    return {
        "checkpoint_hashes_match": hashes,
        "evaluation_reports": len(evaluations),
        "evaluation_games": sum(r["summary"]["games"] for r in evaluations),
        "unfinished_evaluations": 0,
        "learner_updates": learner_steps,
        "skipped_updates": skipped,
        "errors": errors,
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "within_budget": result["elapsed_s"] <= hours * 3600,
        "campaign_hours": hours,
        "resume_checkpoint": resume,
        **(
            {"within_eight_hours": result["elapsed_s"] <= 8 * 3600}
            if hours == 8
            else {}
        ),
    }


def run_enhancement_campaign(
    campaign: Campaign, initializer: str, *, reanalysis: bool = False
) -> dict:
    study = "reanalysis" if reanalysis else "enhancement"
    stage = "reanalysis" if reanalysis else "enhancements"
    hours, pilot_minutes = (12, 90) if reanalysis else (8, 60)
    source = Path(initializer).resolve()
    net, payload = load_net_from_checkpoint(source)
    if (
        net.arch != "source_attn"
        or net.hidden != 256
        or payload.get("trained_player_counts") != [2]
    ):
        raise ValueError(
            "Enhancement campaign requires a width-256 two-player source_attn initializer"
        )
    del net, payload
    frozen = campaign.root / "initializers/frozen_start.pt"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        if checkpoint_hash(temporary) != checkpoint_hash(source):
            raise ValueError("Initializer changed during freezing")
        temporary.replace(frozen)
    if checkpoint_hash(frozen) != checkpoint_hash(source):
        raise ValueError("Frozen initializer changed; use a new campaign root")

    configs = (reanalysis_configs if reanalysis else campaign_configs)(
        campaign, str(frozen)
    )
    search = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=64,
        cpu_workers=1,
        temperature=0.25,
        q_scale=28.0,
        root_noise_scale=1.0,
    )
    development_seed = campaign.seed + (80_000_000 if reanalysis else 60_000_000)
    confirmation_seed = campaign.seed + (90_000_000 if reanalysis else 70_000_000)
    plan = {
        "study": study,
        "campaign_hours": hours,
        "initializer": str(frozen),
        "initializer_sha256": checkpoint_hash(frozen),
        "arms": {name: asdict(cfg) for name, cfg in configs.items()},
        "pilot_minutes_per_arm": pilot_minutes,
        "continuation_cumulative_minutes": [210, 330, 450, 495]
        if reanalysis
        else [175, 290],
        "final_evaluation_reserve_minutes": 90,
        "progress_screen_reserve_minutes": 10,
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "evaluation_search": asdict(search),
        "development_seed": development_seed,
        "confirmation_seed": confirmation_seed,
        "development_games": 512 if reanalysis else 256,
        "progress_games": 512 if reanalysis else 256,
        "retain_selected_arm_resume": reanalysis,
        "confirmation_tree_games": 1024,
        "confirmation_greedy_games": 256,
        "bot_workers": campaign.bot_workers,
        "astra_identity": builtin_identity("astra"),
        "initialization": "Same frozen weights, fresh optimizer/replay, isolated seeded league per arm.",
        "selection": (
            "Reanalysis4x only if paired frozen-opponent difference CI is above zero and Astra delta >= -0.03."
            if reanalysis
            else "Enhanced only if paired frozen-opponent difference CI is above zero and Astra delta >= -0.03."
        ),
        "checkpoint_selection": "Highest frozen-start development score within 3pp of starting Astra score; initializer eligible.",
        "automatic_promotion": False,
        "provenance": campaign.code,
    }
    plan_path = campaign.root / f"{study}_campaign_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Enhancement protocol changed; use a new campaign root")
    write_report(plan_path, plan)
    budget_path = campaign.root / f"{study}_campaign_budget.json"
    if not budget_path.exists():
        now = time.time()
        budget = {
            "started_at": now,
            "deadline": now
            + min(hours * 3600, max(0, campaign.deadline - time.monotonic())),
        }
        for name in ("preflight_budget.json", "supervisor.json"):
            path = campaign.root / name
            if path.exists():
                existing = json.loads(path.read_text())
                budget["deadline"] = min(budget["deadline"], existing["deadline"])
                budget["started_at"] = min(budget["started_at"], existing["started_at"])
        write_report(budget_path, budget)
    budget = json.loads(budget_path.read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    final_path = campaign.root / f"{study}_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        for name, milestone in result["last_milestones"].items():
            if not reanalysis or name != result["selected_arm"]:
                retire_replay(configs[name], milestone)
        write_report(
            campaign.root / "verification.json", _verify(campaign.root, result)
        )
        _write_report(campaign.root, result, budget)
        campaign.status(
            f"{stage}_complete", selected_checkpoint=result["selected_checkpoint"]
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
        name: train_segment(
            campaign, cfg, pilot_minutes, "pilot", evaluation_reserve_minutes=100
        )
        for name, cfg in configs.items()
    }
    reports: dict[str, dict] = {name: {} for name in configs}
    for mode in ("tree64",) if reanalysis else ("tree64", "greedy"):
        for opponent_name, opponent in (("frozen", str(frozen)), ("astra", "astra")):
            for name, pilot in pilots.items():
                label = f"{mode}_{opponent_name}"
                reports[name][label] = match(
                    f"pilot_{name}_{label}",
                    pilot["checkpoint"],
                    opponent,
                    greedy=mode == "greedy",
                    games=plan["development_games"],
                )
    decision = (
        choose_arm(reports, "current", "reanalysis4x")
        if reanalysis
        else choose_arm(reports)
    )
    write_report(
        campaign.root / "arm_selection.json",
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
    campaign.status(f"{stage}_selected", **decision)
    baseline = match(
        "development_start_tree64_astra",
        str(frozen),
        "astra",
        games=plan["development_games"],
    )
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
            milestone = train_segment(
                campaign,
                configs[selected_arm],
                minutes,
                label,
                evaluation_reserve_minutes=100,
            )
        except TrainingWindowExhausted:
            training_budget_limited = True
            break
        last_milestones[selected_arm] = milestone
        screens = {
            f"tree64_{name}": match(
                f"{label}_tree64_{name}",
                milestone["checkpoint"],
                opponent,
                games=plan["progress_games"],
            )["summary"]
            for name, opponent in (("frozen", str(frozen)), ("astra", "astra"))
        }
        candidates[label] = {"checkpoint": milestone["checkpoint"], "screens": screens}
        write_report(
            campaign.root / f"{study}_campaign_progress.json",
            {
                "selected_arm": selected_arm,
                "candidates": candidates,
                "last_milestones": last_milestones,
                "best_development_checkpoint": choose_checkpoint(
                    candidates, baseline["summary"]["match_score"]
                ),
            },
        )
        if milestone.get("executed_target_minutes", minutes) < minutes:
            training_budget_limited = True
            break

    best = choose_checkpoint(candidates, baseline["summary"]["match_score"])
    checkpoint = candidates[best]["checkpoint"]
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
            campaign.root / f"{study}_confirmation_progress.json",
            {
                "selection": selection,
                "screens": {key: value["summary"] for key, value in final.items()},
            },
        )
    differences = {
        mode: paired_difference(final[f"{mode}_astra"], final[f"start_{mode}_astra"])
        for mode in ("tree64", "greedy")
    }
    resume = (
        Path(configs[selected_arm].runs_root)
        / configs[selected_arm].run_id
        / "checkpoints/latest_resume.pt"
    )
    result = {
        "plan": plan,
        **selection,
        "arm_decision": decision,
        "candidates": candidates,
        "screens": {key: value["summary"] for key, value in final.items()},
        "candidate_minus_start_astra": differences,
        "last_milestones": last_milestones,
        "improvement_resolved": final["tree64_frozen"]["summary"]["match_score_ci95"][0]
        > 0.5,
        "automatic_promotion": False,
        "provenance": campaign.code,
        "training_budget_limited": training_budget_limited,
        "elapsed_s": time.time() - budget["started_at"],
        **(
            {
                "resume_checkpoint": str(resume),
                "resume_checkpoint_sha256": checkpoint_hash(resume),
            }
            if reanalysis
            else {}
        ),
    }
    write_report(campaign.root / "verification.json", _verify(campaign.root, result))
    write_report(final_path, result)
    for name, milestone in last_milestones.items():
        if not reanalysis or name != selected_arm:
            retire_replay(configs[name], milestone)
    _write_report(campaign.root, result, budget)
    campaign.status(
        f"{stage}_complete",
        selected_checkpoint=checkpoint,
        selected_arm=selected_arm,
        improvement_resolved=result["improvement_resolved"],
    )
    return result
