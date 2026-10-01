"""A fresh twelve-hour window for an interrupted distillation comparison."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from agent.eval.arena import checkpoint_hash, write_report
from agent.scripts.capacity_campaign import paired_serving_difference, retire_arm
from agent.scripts.distillation_campaign import (
    arm_config,
    confirmation_games,
    fork_arm,
    freeze_record,
    serving_profile,
)
from agent.scripts.finetune_campaign import DIAGNOSTIC_ENVIRONMENT
from agent.scripts.league_campaign import SEARCH, freeze_file
from agent.scripts.lr_campaign import TrainingWindowExhausted, train_segment
from agent.scripts.weight_refine_campaign import (
    candidate_rank,
    freeze_league,
    retain_best,
)
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_checkpoint_payload, save_checkpoint_payload
from agent.train.loop import LoopConfig

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


def relocate_resume(source: dict, config: LoopConfig) -> dict:
    """Relocate a full state without changing RNG, optimizer, replay or elapsed training."""
    directory = Path(config.runs_root) / config.run_id
    identity = {"source": source, "config": asdict(config)}
    marker = directory / "relocated_resume.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Resume relocation changed")
        return identity
    if directory.exists():
        raise ValueError("Cannot overwrite an unmarked resume directory")
    if checkpoint_hash(source["resume"]) != source["resume_sha256"]:
        raise ValueError("Saved source checkpoint changed")
    staging = directory.with_name(directory.name + ".preparing")
    staging.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_payload(source["resume"])
    original = LoopConfig(**payload["config"])
    allowed = {
        "run_id",
        "runs_root",
        "league_root",
        "max_wall_minutes",
        "provenance",
        "device",
    }
    changed = [
        k
        for k, v in asdict(original).items()
        if k not in allowed and v != getattr(config, k)
    ]
    if changed:
        raise ValueError(
            f"Exact resume cannot change training configuration: {changed}"
        )
    for key in ("optimizer_state_dict", "buffer", "rng_state", "distillation"):
        if key not in payload:
            raise ValueError(f"Interrupted distillation state lacks {key}")
    freeze_league(Path(source["league"]), staging / "league")
    payload["config"] = asdict(config)
    save_checkpoint_payload(staging / "checkpoints/latest_resume.pt", payload)
    write_report(staging / "relocated_resume.json", identity)
    staging.rename(directory)
    return identity


def prepare(campaign: Campaign, initializer: str) -> dict:
    inputs = json.loads(Path(initializer).read_text())
    marker = campaign.root / "prepared.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["inputs"] != inputs:
            raise ValueError("Resume inputs changed")
        for path, digest in result["hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen resume input changed: {path}")
        return result
    for path, digest in inputs["source_hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError(f"Historical input changed: {path}")
    previous = Path(inputs["previous_root"])
    prior = json.loads((previous / "prepared.json").read_text())
    baseline = json.loads((previous / "baseline_selection.json").read_text())
    stopped = json.loads((previous / "STOPPED_FOR_SHUTDOWN.json").read_text())
    states = {}
    for name, state in prior["states"].items():
        destination = campaign.root / "initializers" / name
        freeze_file(Path(state["resume"]), destination / "resume.pt")
        freeze_file(Path(state["checkpoint"]), destination / "weights.pt")
        freeze_league(Path(state["league"]), destination / "league")
        states[name] = {
            **state,
            "resume": str(destination / "resume.pt"),
            "checkpoint": str(destination / "weights.pt"),
            "league": str(destination / "league"),
        }
    source = Path(stopped["checkpoint"])
    freeze_file(source, campaign.root / "initializers/interrupted/resume.pt")
    freeze_league(
        source.parent.parent / "league",
        campaign.root / "initializers/interrupted/league",
    )
    payload = load_checkpoint_payload(source)
    if (
        payload["iteration"] != stopped["iteration"]
        or payload["buffer"]["size"] != stopped["replay_positions"]
        or payload["distillation"]["bank"]["size"] != stopped["teacher_positions"]
    ):
        raise ValueError("Shutdown receipt does not match saved training state")
    run = campaign.root / "experiments/student_resumed"
    student_config = replace(
        LoopConfig(**payload["config"]),
        run_id=run.name,
        runs_root=str(run.parent),
        league_root=str(run / "league"),
        max_wall_minutes=150.0,
        device=campaign.device,
        provenance=None,
    )
    del payload
    historical = {}
    for name in (
        "initial_small_astra",
        "initial_wide_astra",
        "initial_serving_head_to_head",
    ):
        matches = list((previous / "evaluations").glob(name + "_*.json"))
        if len(matches) != 1:
            raise ValueError("Expected one immutable historical baseline report")
        freeze_file(matches[0], campaign.root / "historical" / (name + ".json"))
        historical[name] = json.loads(matches[0].read_text())
    base_report = historical["initial_" + baseline["name"] + "_astra"]
    if base_report["candidate_sha256"] != checkpoint_hash(
        states[baseline["name"]]["checkpoint"]
    ):
        raise ValueError("Historical baseline weights mismatch")
    result = {
        "inputs": inputs,
        "states": states,
        "baseline": baseline,
        "historical": historical,
        "student_config": asdict(student_config),
        "student_source": {
            "resume": str(campaign.root / "initializers/interrupted/resume.pt"),
            "resume_sha256": checkpoint_hash(source),
            "league": str(campaign.root / "initializers/interrupted/league"),
        },
        "previous_best_student": json.loads(
            (previous / "best_student.json").read_text()
        ),
        "hashes": {
            str(p): checkpoint_hash(p)
            for d in ("initializers", "historical")
            for p in (campaign.root / d).rglob("*")
            if p.is_file()
        },
    }
    write_report(marker, result)
    return result


def student_has_clear_regression(head: dict) -> bool:
    return head["summary"]["match_score_ci95"][1] < 0.48


def continuation_targets(
    remaining_minutes: float, reserve: float, screen_minutes: float
) -> list[float]:
    available = max(0.0, remaining_minutes - reserve)
    count = int(available // (60 + screen_minutes))
    targets = [60.0 * i for i in range(1, count + 1)]
    tail = available - count * (60 + screen_minutes) - screen_minutes
    if tail >= 20:
        targets.append(60 * count + tail)
    return targets


def run_resume_distillation(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    ready = json.loads((campaign.root / "ready.json").read_text())
    if not ready.get("passed") or ready["code"] != campaign.code:
        raise ValueError("Resume controller was not validated with current source")
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise ValueError("Native diagnostics must remain enabled")
    result_path = campaign.root / "distillation_resume.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        campaign.status(
            "distillation_resume_complete",
            selected_checkpoint=result["selected_checkpoint"],
        )
        return result
    prepared = prepare(campaign, initializer)
    states = prepared["states"]
    baseline_name = prepared["baseline"]["name"]
    baseline = states[baseline_name]
    profiles = prepared["baseline"]["profiles"]
    searches = {n: SearchConfig(**p["search"]) for n, p in profiles.items()}
    historical = prepared["historical"]
    baseline_astra = historical["initial_" + baseline_name + "_astra"]["summary"][
        "match_score"
    ]
    development_seed = historical["initial_small_astra"]["config"]["seed"]
    plan = freeze_record(
        campaign.root,
        "plan.json",
        {
            "budget": budget,
            "code": campaign.code,
            "source": prepared["student_source"],
            "student": "Resume exact state to cumulative 90 minutes; stop if head-to-head CI upper <48%, else continue to 150 minutes.",
            "wide": "Continue best width-512 checkpoint at LR 0.0003; 60/150 minute milestones.",
            "continuation": "Spend remaining training allocation on best pilot, with hourly serving-budget screens.",
            "replication": "One hour from original initializer with fresh seed only if pilot point score >50% and Astra within 3pp.",
            "development_seed": development_seed,
            "confirmation_seed": campaign.seed + 300_000_000,
            "final_budget": "Measured reserve for approximately 1024 primary games; choose largest count fitting before final outcomes.",
            "automatic_deployment": False,
            "historical_evaluations": "Reused as immutable baseline data; inference and training implementation unchanged.",
        },
    )

    def match(
        label: str,
        path: str,
        opponent: str,
        search: SearchConfig,
        opponent_search: SearchConfig = SEARCH,
        *,
        games: int = 256,
        confirmation: bool = False,
        cpu: bool = False,
    ) -> dict:
        cfg = replace(
            campaign.arena(
                games,
                search=search,
                split="confirmation" if confirmation else "development",
            ),
            seed=plan["confirmation_seed"] + (1_000_000 if cpu else 0)
            if confirmation
            else development_seed,
            inference_device="cpu" if cpu else campaign.device,
            game_batch_size=2 if cpu else 128,
        )
        result = campaign._match(label, path, opponent, cfg, opponent_search)
        if result["summary"]["unfinished"]:
            raise RuntimeError("Unfinished evaluation games")
        return result

    prior_best = prepared["previous_best_student"]
    candidates = {
        baseline_name: {
            "checkpoint": baseline["checkpoint"],
            "hidden": baseline["hidden"],
            "scores": {"start": 0.5, "astra": baseline_astra},
            "arm": None,
        },
        "prior_student_30m": {
            "checkpoint": prior_best["milestone"]["checkpoint"],
            "hidden": 256,
            "scores": prior_best["scores"],
            "arm": "student",
        },
    }
    best_arms = {}
    configs = {
        "student": LoopConfig(**prepared["student_config"]),
        "wide": arm_config(
            campaign, prepared, "wide", distilled=False, seed=campaign.seed
        ),
    }
    seconds_per_game = historical["initial_serving_head_to_head"]["wall_s"] / 256 + sum(
        historical["initial_" + n + "_astra"]["wall_s"] / 128 for n in ("small", "wide")
    )
    # Set aside a powered confirmation, leaving most of the shorter renewal for learning.
    reserve_minutes = max(
        180.0, min(240.0, (1.3 * 1024 * seconds_per_game + 2400) / 60)
    )
    screen_minutes = (
        1.3
        * (
            historical["initial_serving_head_to_head"]["wall_s"]
            + historical["initial_small_astra"]["wall_s"]
        )
        / 60
    )

    def screen(arm: str, milestone: dict, config: LoopConfig, label: str) -> dict:
        search = searches["small" if config.hidden == 256 else "wide"]
        head = match(
            "development_" + label + "_baseline",
            milestone["checkpoint"],
            baseline["checkpoint"],
            search,
            searches[baseline_name],
        )
        astra = match(
            "development_" + label + "_astra",
            milestone["checkpoint"],
            "astra",
            search,
            games=128,
        )
        scores = {
            "start": head["summary"]["match_score"],
            "astra": astra["summary"]["match_score"],
        }
        candidates[label] = {
            "checkpoint": milestone["checkpoint"],
            "hidden": config.hidden,
            "scores": scores,
            "arm": arm,
        }
        best = retain_best(campaign, arm, milestone, scores, baseline_astra, config)
        if arm == "student" and candidate_rank(
            prior_best["scores"], baseline_astra
        ) > candidate_rank(best["scores"], baseline_astra):
            best = prior_best
        best_arms[arm] = best
        write_report(
            campaign.root / "development.json",
            {"candidates": candidates, "best_arms": best_arms},
        )
        return head

    campaign.status("resuming", iteration=108)
    relocate_resume(prepared["student_source"], configs["student"])
    for minutes in (90, 150):
        label = f"student_{minutes}m"
        if (
            minutes == 150
            and json.loads((campaign.root / "student_90m_decision.json").read_text())[
                "stop_student"
            ]
        ):
            break
        milestone = train_segment(
            campaign,
            configs["student"],
            minutes,
            label,
            evaluation_reserve_minutes=reserve_minutes + screen_minutes,
        )
        head = screen("student", milestone, configs["student"], label)
        if minutes == 90:
            freeze_record(
                campaign.root,
                "student_90m_decision.json",
                {
                    "stop_student": student_has_clear_regression(head),
                    "head_to_head": head["summary"],
                },
            )
    retire_arm(campaign, configs["student"], best_arms["student"])
    campaign.status("forking", arm="wide")
    fork_arm(configs["wide"], states["wide"])
    for minutes in (60, 150):
        label = f"wide_{minutes}m"
        milestone = train_segment(
            campaign,
            configs["wide"],
            minutes,
            label,
            evaluation_reserve_minutes=reserve_minutes + screen_minutes,
        )
        screen("wide", milestone, configs["wide"], label)
    retire_arm(campaign, configs["wide"], best_arms["wide"])
    chosen = max(
        best_arms,
        key=lambda arm: candidate_rank(best_arms[arm]["scores"], baseline_astra),
    )
    promising = (
        best_arms[chosen]["scores"]["start"] > 0.5
        and best_arms[chosen]["scores"]["astra"] >= baseline_astra - 0.03
    )
    freeze_record(
        campaign.root,
        "approach_selection.json",
        {"arm": chosen, "state": best_arms[chosen], "replicate": promising},
    )
    for arm in ["replicate", "continuation"] if promising else ["continuation"]:
        source = (
            states["small" if chosen == "student" else "wide"]
            if arm == "replicate"
            else best_arms[chosen]
        )
        init = source.get("checkpoint") or source["milestone"]["checkpoint"]
        config = arm_config(
            campaign,
            prepared,
            arm,
            distilled=chosen == "student",
            seed=campaign.seed + (1009 if arm == "replicate" else 2017),
            initializer=init,
        )
        campaign.status("forking", arm=arm)
        fork_arm(config, source)
        allocation = campaign.root / (arm + "_allocation.json")
        if not allocation.exists():
            targets = (
                [60.0]
                if arm == "replicate"
                else continuation_targets(
                    (campaign.deadline - time.monotonic()) / 60,
                    reserve_minutes,
                    screen_minutes,
                )
            )
            write_report(
                allocation, {"targets": targets, "reserve_minutes": reserve_minutes}
            )
        for minutes in json.loads(allocation.read_text())["targets"]:
            label = f"{arm}_{minutes:g}m"
            marker = (
                Path(config.runs_root)
                / config.run_id
                / "milestones"
                / (label + ".json")
            )
            stop = campaign.root / (arm + "_stopped.json")
            if stop.exists() and not marker.exists():
                break
            try:
                milestone = train_segment(
                    campaign,
                    config,
                    minutes,
                    label,
                    evaluation_reserve_minutes=reserve_minutes + screen_minutes,
                )
            except TrainingWindowExhausted:
                write_report(stop, {"reason": "Final evaluation reserve reached"})
                break
            screen(arm, milestone, config, label)
        retire_arm(campaign, config, best_arms.get(arm, best_arms[chosen]))

    challenger_label = max(
        (n for n, c in candidates.items() if c["arm"]),
        key=lambda n: candidate_rank(candidates[n]["scores"], baseline_astra),
    )
    challenger = candidates[challenger_label]
    profile = serving_profile(
        campaign,
        "final_challenger",
        challenger["checkpoint"],
        challenger["hidden"],
        maximum_simulations=searches[
            "small" if challenger["hidden"] == 256 else "wide"
        ].num_simulations,
    )
    # Recheck the baseline on the rebooted machine too.
    base_profile = serving_profile(
        campaign,
        "final_baseline",
        baseline["checkpoint"],
        baseline["hidden"],
        maximum_simulations=searches[baseline_name].num_simulations,
    )
    search, base_search = (
        SearchConfig(**profile["search"]),
        SearchConfig(**base_profile["search"]),
    )
    probes = [
        match(
            "final_probe_head",
            challenger["checkpoint"],
            baseline["checkpoint"],
            search,
            base_search,
            games=128,
        ),
        match(
            "final_probe_astra", challenger["checkpoint"], "astra", search, games=128
        ),
        match(
            "final_probe_baseline_astra",
            baseline["checkpoint"],
            "astra",
            base_search,
            games=128,
        ),
    ]
    selection_path = campaign.root / "final_selection.json"
    identity = {
        "challenger_label": challenger_label,
        "challenger": challenger,
        "profile": profile,
        "baseline": baseline,
        "baseline_profile": base_profile,
    }
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if any(selection[k] != v for k, v in identity.items()):
            raise ValueError("Frozen final selection changed")
    else:
        games = confirmation_games(
            campaign.deadline - time.monotonic(), sum(p["wall_s"] for p in probes) / 128
        )
        if not games:
            raise TrainingWindowExhausted("Final confirmation no longer fits")
        selection = {**identity, "games": games}
        write_report(selection_path, selection)
    final = {}
    for name, path, opponent, candidate_search, opponent_search in [
        (
            "head_to_head",
            challenger["checkpoint"],
            baseline["checkpoint"],
            search,
            base_search,
        ),
        ("challenger_astra", challenger["checkpoint"], "astra", search, SEARCH),
        ("baseline_astra", baseline["checkpoint"], "astra", base_search, SEARCH),
    ]:
        final[name] = match(
            "confirmation_" + name,
            path,
            opponent,
            candidate_search,
            opponent_search,
            games=selection["games"],
            confirmation=True,
        )
        write_report(
            campaign.root / "confirmation_progress.json",
            {k: v["summary"] for k, v in final.items()},
        )
    final["cpu_audit"] = match(
        "confirmation_cpu_audit",
        challenger["checkpoint"],
        baseline["checkpoint"],
        SearchConfig(**profile["serving_search"]),
        SearchConfig(**base_profile["serving_search"]),
        games=8,
        confirmation=True,
        cpu=True,
    )
    head = final["head_to_head"]["summary"]
    delta = paired_serving_difference(
        final["challenger_astra"], final["baseline_astra"]
    )
    timing = final["cpu_audit"]["timed_move_latency"].get("candidate", {})
    timing_passed = bool(timing.get("count")) and timing["over_2s"] == 0
    provisional = (
        head["match_score"] > 0.5
        and delta["match_score_difference"] >= -0.03
        and timing_passed
    )
    result = {
        "selected_checkpoint": challenger["checkpoint"]
        if provisional
        else baseline["checkpoint"],
        "selected_label": challenger_label if provisional else baseline_name,
        "provisional_improvement": provisional,
        "resolved_improvement": provisional and head["match_score_ci95"][0] > 0.5,
        "cpu_timing_passed": timing_passed,
        "astra_difference": delta,
        "final_selection": selection,
        "best_arms": best_arms,
        "elapsed_s": time.time() - budget["started_at"],
        "automatic_deployment": False,
        "final": {
            k: {
                "summary": r["summary"],
                "wall_s": r["wall_s"],
                "timed_move_latency": r.get("timed_move_latency", {}),
            }
            for k, r in final.items()
        },
    }
    for path, digest in prepared["hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError("Frozen source changed during resumed campaign")
    write_report(result_path, result)
    (campaign.root / "REPORT.md").write_text(
        "# Resumed twelve-hour training results\n\n"
        f"Selected: `{result['selected_checkpoint']}`\n\n"
        f"Challenger match score: {head['match_score']:.2%}; 95% CI {head['match_score_ci95']}. "
        f"Provisional improvement: {provisional}; resolved: {result['resolved_improvement']}.\n\n"
        f"Astra difference: {delta['match_score_difference']:+.2%}; local CPU timing passed: {timing_passed}. "
        "No deployment. Full details are in distillation_resume.json.\n"
    )
    campaign.status(
        "distillation_resume_complete",
        selected_checkpoint=result["selected_checkpoint"],
    )
    return result
