"""24-hour serving-strength study: width-512 continuation and width-256 distillation."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

import torch

from agent.eval.arena import checkpoint_hash, write_report
from agent.eval.latency import benchmark_latency
from agent.scripts.capacity_campaign import paired_serving_difference, retire_arm
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
from agent.train.distillation import make_bank
from agent.train.loop import LoopConfig
from agent.train.reproducibility import capture_rng_state, seed_all

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


def freeze_record(root: Path, name: str, value: dict) -> dict:
    value = json.loads(json.dumps(value))
    path = root / name
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Frozen campaign record changed: {name}")
    write_report(path, value)
    return value


def prepare(campaign: Campaign, initializer: str) -> dict:
    source = json.loads(Path(initializer).read_text())
    marker = campaign.root / "prepared.json"
    if marker.exists():
        prepared = json.loads(marker.read_text())
        if prepared["inputs"] != source:
            raise ValueError("Campaign inputs changed")
        for path, digest in prepared["hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError("Frozen campaign input changed")
        return prepared
    states = {}
    for name, hidden in [("small", 256), ("wide", 512)]:
        record = source[name]
        if (
            checkpoint_hash(record["resume"]) != record["resume_sha256"]
            or checkpoint_hash(record["milestone"]["checkpoint"])
            != record["milestone"]["checkpoint_sha256"]
        ):
            raise ValueError("Source model hash mismatch")
        directory = campaign.root / "initializers" / name
        freeze_file(Path(record["resume"]), directory / "resume.pt")
        freeze_file(Path(record["milestone"]["checkpoint"]), directory / "weights.pt")
        freeze_league(Path(record["league"]), directory / "league")
        payload = load_checkpoint_payload(directory / "resume.pt")
        weights = load_checkpoint_payload(directory / "weights.pt")
        if (
            payload["hidden"] != hidden
            or payload["trained_player_counts"] != [2]
            or any(
                not torch.equal(v, weights["model_state_dict"][k])
                for k, v in payload["model_state_dict"].items()
            )
            or not payload["buffer"]["snapshots"]
        ):
            raise ValueError("Source resume/model/replay mismatch")
        states[name] = {
            "resume": str(directory / "resume.pt"),
            "resume_sha256": record["resume_sha256"],
            "league": str(directory / "league"),
            "checkpoint": str(directory / "weights.pt"),
            "config": payload["config"],
            "hidden": hidden,
        }
        del payload, weights
    prepared = {
        "inputs": source,
        "states": states,
        "hashes": {
            str(p): checkpoint_hash(p)
            for p in (campaign.root / "initializers").rglob("*")
            if p.is_file()
        },
    }
    write_report(marker, prepared)
    return prepared


def arm_config(
    campaign: Campaign,
    prepared: dict,
    name: str,
    *,
    distilled: bool,
    seed: int,
    initializer: str | None = None,
) -> LoopConfig:
    source = prepared["states"]["small" if distilled else "wide"]
    directory = campaign.root / "experiments" / f"{name}_seed{seed}"
    teacher = prepared["states"]["wide"]["checkpoint"]
    return replace(
        LoopConfig(**source["config"]),
        run_id=directory.name,
        runs_root=str(directory.parent),
        league_root=str(directory / "league"),
        init_from=initializer or source["checkpoint"],
        seed=seed,
        device=campaign.device,
        provenance=None,
        max_wall_minutes=1440.0,
        lr=source["config"]["lr"] if distilled else 0.0003,
        keep_recent_checkpoints=1,
        bounded_checkpoint_storage=True,
        search_inference_cache_size=0,
        distillation_teacher=teacher if distilled else "",
        distillation_teacher_sha256=checkpoint_hash(teacher) if distilled else "",
        distillation_positions=512,
        distillation_sims=1024,
        distillation_capacity=32768,
        distillation_fraction=0.5,
    )


def fork_arm(config: LoopConfig, source: dict) -> dict:
    """Explicit new seed fork, retaining optimizer/replay and any teacher bank."""
    directory = Path(config.runs_root) / config.run_id
    identity = {"config": asdict(config), "source": source}
    marker = directory / "distillation_fork.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["identity"] != identity:
            raise ValueError("Training fork changed")
        return result
    if directory.exists():
        raise ValueError("Refusing to overwrite unmarked training directory")
    if checkpoint_hash(source["resume"]) != source["resume_sha256"]:
        raise ValueError("Fork source changed")
    staging = directory.with_name(directory.name + ".preparing")
    if staging.exists():
        shutil.rmtree(staging)
    payload = load_checkpoint_payload(source["resume"])
    freeze_league(Path(source["league"]), staging / "league")
    seed_all(config.seed)
    for group in payload["optimizer_state_dict"]["param_groups"]:
        group["lr"] = config.lr
    payload.update(
        config=asdict(config),
        rng_state=capture_rng_state(),
        progress={"training_wall_s": 0.0},
    )
    if config.distillation_teacher:
        if "distillation" not in payload:
            payload["distillation"] = {
                "teacher_sha256": config.distillation_teacher_sha256,
                "bank": make_bank(config.distillation_capacity, "cpu").state_dict(),
            }
        elif (
            payload["distillation"]["teacher_sha256"]
            != config.distillation_teacher_sha256
        ):
            raise ValueError("Continuation teacher changed")
    elif "distillation" in payload:
        raise ValueError("Cannot silently drop a teacher bank")
    save_checkpoint_payload(staging / "checkpoints/latest_resume.pt", payload)
    result = {
        "identity": identity,
        "initial_iteration": payload["iteration"],
        "replay_size": payload["buffer"]["size"],
        "optimizer_retained": True,
        "fresh_seed": config.seed,
    }
    write_report(staging / "distillation_fork.json", result)
    staging.rename(directory)
    return result


def serving_profile(
    campaign: Campaign,
    label: str,
    checkpoint: str,
    hidden: int,
    maximum_simulations: int | None = None,
) -> dict:
    marker = campaign.root / f"serving_{label}.json"
    digest = checkpoint_hash(checkpoint)
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["checkpoint_sha256"] != digest:
            raise ValueError("Serving candidate changed")
        return record
    # Predeclared near-budget profiles; no noisy win-rate maximization over many
    # search settings. The largest fully completed CPU budget is the primary.
    budgets = (256, 384, 512) if hidden == 512 else (768, 1024, 1280)
    records, selected = [], None
    for simulations in budgets:
        if maximum_simulations is not None and simulations > maximum_simulations:
            continue
        campaign.status("latency", candidate=label, simulations=simulations)
        record = benchmark_latency(
            checkpoint,
            replace(SEARCH, num_simulations=simulations),
            seed=campaign.seed + 270_000_000,
            games=8,
            deadline_s=1.8,
            wall_budget_s=2.0,
        )
        write_report(campaign.root / "latency" / f"{label}_{simulations}.json", record)
        records.append(record)
        if record["qualified"]:
            selected = simulations
    if selected is None:
        raise ValueError("No near-budget profile passes the CPU qualification")
    record = {
        "checkpoint_sha256": digest,
        "search": asdict(replace(SEARCH, num_simulations=selected)),
        "serving_search": asdict(
            replace(SEARCH, num_simulations=selected, move_deadline_s=1.8)
        ),
        "records": records,
        "local_cpu_only": True,
        "move_budget_s": 2.0,
    }
    write_report(marker, record)
    return record


def confirmation_games(remaining_s: float, seconds_per_game: float) -> int:
    """Freeze power before inspecting outcomes, with 30 minutes for CPU audit."""
    if seconds_per_game <= 0:
        raise ValueError("Missing measured evaluation throughput")
    return next(
        (
            n
            for n in (2048, 1536, 1024, 768, 512, 256)
            if 1.3 * n * seconds_per_game + 1800 <= remaining_s
        ),
        0,
    )


def run_distillation_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    ready = json.loads((campaign.root / "ready.json").read_text())
    if ready.get("passed") is not True or ready["code"] != campaign.code:
        raise ValueError("Validation missing or source changed since validation")
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise ValueError("Native diagnostics must be enabled")
    result_path = campaign.root / "distillation_campaign.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        campaign.status(
            "distillation_campaign_complete",
            selected_checkpoint=result["selected_checkpoint"],
        )
        return result
    prepared = prepare(campaign, initializer)
    states = prepared["states"]
    plan = freeze_record(
        campaign.root,
        "plan.json",
        {
            "budget": budget,
            "code": campaign.code,
            "pilot_training_minutes": [30, 90, 180],
            "continuation": "Best full pilot state; hourly checkpoints; measured evaluation reserve.",
            "replication": "Two training hours from the original initializer, independent seed.",
            "teacher": "Frozen width 512, 1024 simulations, no root noise; 50% teacher minibatch and policy-loss share.",
            "value_targets": "Observed outcomes only: +1 sole winner, 0 shared winner, -1 loser.",
            "development_games": {"primary": 256, "astra": 128},
            "confirmation": "Freeze candidates/profiles/counts before fresh paired seeds; up to 2048 games.",
            "selection": "Highest serving-budget score vs frozen baseline within 3pp of baseline Astra.",
            "promotion": "Provisional: held-out point advantage and Astra delta >= -3pp; resolved: CI lower >50%.",
            "automatic_deployment": False,
            "retention": "All milestone weights, best full state and matching league per arm.",
        },
    )
    profiles = {
        name: serving_profile(
            campaign, "initial_" + name, state["checkpoint"], state["hidden"]
        )
        for name, state in states.items()
    }
    searches = {name: SearchConfig(**p["search"]) for name, p in profiles.items()}

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
            seed=campaign.seed
            + (290_000_000 if confirmation else 280_000_000)
            + (1_000_000 if cpu else 0),
            inference_device="cpu" if cpu else campaign.device,
            game_batch_size=2 if cpu else 128,
        )
        result = campaign._match(label, path, opponent, cfg, opponent_search)
        if result["summary"]["unfinished"]:
            raise RuntimeError("Unfinished evaluation games")
        return result

    # More simulations are only adopted if a direct paired comparison resolves
    # an advantage over the already-tested 1024 profile. A faster runtime alone
    # is not evidence of better play.
    if searches["small"].num_simulations > 1024:
        probe = match(
            "small_extra_budget_probe",
            states["small"]["checkpoint"],
            states["small"]["checkpoint"],
            searches["small"],
            replace(SEARCH, num_simulations=1024),
            games=256,
        )
        adopt = probe["summary"]["match_score_ci95"][0] > 0.5
        freeze_record(
            campaign.root,
            "small_budget_selection.json",
            {"adopt_extra_simulations": adopt, "summary": probe["summary"]},
        )
        if not adopt:
            searches["small"] = replace(SEARCH, num_simulations=1024)
            profiles["small"] = {
                **profiles["small"],
                "search": asdict(searches["small"]),
                "serving_search": asdict(
                    replace(searches["small"], move_deadline_s=1.8)
                ),
            }

    initial_astra = {
        n: match(
            "initial_" + n + "_astra", s["checkpoint"], "astra", searches[n], games=128
        )
        for n, s in states.items()
    }
    missing = match(
        "missing_wide512_vs_small1024",
        states["wide"]["checkpoint"],
        states["small"]["checkpoint"],
        replace(SEARCH, num_simulations=512),
        replace(SEARCH, num_simulations=1024),
    )
    if (
        searches["wide"].num_simulations == 512
        and searches["small"].num_simulations == 1024
    ):
        serving = missing
    else:
        serving = match(
            "initial_serving_head_to_head",
            states["wide"]["checkpoint"],
            states["small"]["checkpoint"],
            searches["wide"],
            searches["small"],
        )
    baseline_name = (
        "wide"
        if (
            serving["summary"]["match_score"] > 0.5
            and initial_astra["wide"]["summary"]["match_score"]
            >= initial_astra["small"]["summary"]["match_score"] - 0.03
        )
        else "small"
    )
    freeze_record(
        campaign.root,
        "baseline_selection.json",
        {"name": baseline_name, "profiles": profiles},
    )
    baseline = states[baseline_name]
    baseline_search = searches[baseline_name]
    baseline_astra = initial_astra[baseline_name]["summary"]["match_score"]
    # Use observed near-budget timing, including both Astra controls, and margin.
    seconds_per_suite_game = serving["wall_s"] / 256 + sum(
        r["wall_s"] / 128 for r in initial_astra.values()
    )
    final_reserve_minutes = max(
        240.0, min(360.0, (1.3 * 2048 * seconds_per_suite_game + 1800) / 60)
    )
    development_screen_minutes = (
        (serving["wall_s"] + max(r["wall_s"] for r in initial_astra.values()))
        / 60
        * 1.3
    )
    candidates = {
        baseline_name: {
            "checkpoint": baseline["checkpoint"],
            "hidden": baseline["hidden"],
            "scores": {"start": 0.5, "astra": baseline_astra},
            "arm": None,
        }
    }
    best_arms = {}
    configs = {
        name: arm_config(
            campaign, prepared, name, distilled=name == "student", seed=campaign.seed
        )
        for name in ("student", "wide")
    }

    def screen(arm: str, milestone: dict, config: LoopConfig, label: str) -> None:
        search = searches["small" if config.hidden == 256 else "wide"]
        scores = {
            "start": match(
                "development_" + label + "_baseline",
                milestone["checkpoint"],
                baseline["checkpoint"],
                search,
                baseline_search,
            )["summary"]["match_score"],
            "astra": match(
                "development_" + label + "_astra",
                milestone["checkpoint"],
                "astra",
                search,
                games=128,
            )["summary"]["match_score"],
        }
        candidates[label] = {
            "checkpoint": milestone["checkpoint"],
            "hidden": config.hidden,
            "scores": scores,
            "arm": arm,
        }
        best_arms[arm] = retain_best(
            campaign, arm, milestone, scores, baseline_astra, config
        )
        write_report(
            campaign.root / "development.json",
            {"candidates": candidates, "best_arms": best_arms},
        )

    # Distillation first so its learning path is observed before unattended work.
    for arm in ("student", "wide"):
        config = configs[arm]
        campaign.status("forking", arm=arm)
        fork_arm(config, states["small" if arm == "student" else "wide"])
        for minutes in plan["pilot_training_minutes"]:
            label = f"{arm}_{minutes}m"
            milestone = train_segment(
                campaign,
                config,
                minutes,
                label,
                evaluation_reserve_minutes=final_reserve_minutes
                + development_screen_minutes,
            )
            screen(arm, milestone, config, label)
        retire_arm(campaign, config, best_arms[arm])

    chosen = max(
        best_arms,
        key=lambda arm: candidate_rank(best_arms[arm]["scores"], baseline_astra),
    )
    freeze_record(
        campaign.root,
        "approach_selection.json",
        {"arm": chosen, "state": best_arms[chosen]},
    )
    # Replicate the approach independently, preserving a separate best full state.
    for arm, source, seed, targets in [
        (
            "replicate",
            states["small" if chosen == "student" else "wide"],
            campaign.seed + 1009,
            [60, 120],
        ),
        ("continuation", best_arms[chosen], campaign.seed + 2017, None),
    ]:
        initializer_path = source.get("checkpoint") or source["milestone"]["checkpoint"]
        config = arm_config(
            campaign,
            prepared,
            arm,
            distilled=chosen == "student",
            seed=seed,
            initializer=initializer_path,
        )
        configs[arm] = config
        campaign.status("forking", arm=arm)
        fork_arm(config, source)
        if targets is None:
            allocation_path = campaign.root / "continuation_allocation.json"
            if not allocation_path.exists():
                available = max(
                    0.0,
                    (campaign.deadline - time.monotonic()) / 60 - final_reserve_minutes,
                )
                # Spend remaining training window with hourly serving-budget
                # screens, reserving their measured cost rather than a fixed gap.
                hours = int(available // (60 + development_screen_minutes))
                tail = max(
                    0.0,
                    available
                    - hours * (60 + development_screen_minutes)
                    - development_screen_minutes,
                )
                targets = [60 * i for i in range(1, hours + 1)]
                if tail >= 20:
                    targets.append(60 * hours + tail)
                write_report(
                    allocation_path,
                    {
                        "targets": targets,
                        "reserve_minutes": final_reserve_minutes,
                        "screen_estimate_minutes": development_screen_minutes,
                    },
                )
            targets = json.loads(allocation_path.read_text())["targets"]
        stop_path = campaign.root / f"{arm}_stop.json"
        for minutes in targets:
            label = f"{arm}_{minutes:g}m"
            marker = (
                Path(config.runs_root) / config.run_id / "milestones" / f"{label}.json"
            )
            if stop_path.exists() and not marker.exists():
                break
            if (
                not marker.exists()
                and shutil.disk_usage(campaign.root).free < 0.9 * 1024**3
            ):
                write_report(
                    stop_path, {"reason": "Preserve atomic checkpoint headroom"}
                )
                break
            try:
                milestone = train_segment(
                    campaign,
                    config,
                    minutes,
                    label,
                    evaluation_reserve_minutes=final_reserve_minutes
                    + development_screen_minutes,
                )
            except TrainingWindowExhausted:
                write_report(stop_path, {"reason": "Final evaluation reserve reached"})
                break
            screen(arm, milestone, config, label)
        retire_arm(campaign, config, best_arms.get(arm, best_arms[chosen]))

    selected_label = max(
        candidates,
        key=lambda n: candidate_rank(candidates[n]["scores"], baseline_astra),
    )
    # Baseline remains eligible. If it wins development, independently evaluate
    # the best trained challenger but require held-out evidence before selection.
    challenger_label = (
        selected_label
        if candidates[selected_label]["arm"]
        else max(
            (n for n in candidates if candidates[n]["arm"]),
            key=lambda n: candidate_rank(candidates[n]["scores"], baseline_astra),
        )
    )
    challenger = candidates[challenger_label]
    final_profile = serving_profile(
        campaign,
        "final_challenger",
        challenger["checkpoint"],
        challenger["hidden"],
        maximum_simulations=searches[
            "small" if challenger["hidden"] == 256 else "wide"
        ].num_simulations,
    )
    challenger_search = SearchConfig(**final_profile["search"])
    # Throughput probes use development seeds; counts stay independent of final outcomes.
    probe_head = match(
        "final_probe_head",
        challenger["checkpoint"],
        baseline["checkpoint"],
        challenger_search,
        baseline_search,
        games=128,
    )
    probe_astra = match(
        "final_probe_astra",
        challenger["checkpoint"],
        "astra",
        challenger_search,
        games=128,
    )
    final_per_game = (
        probe_head["wall_s"]
        + probe_astra["wall_s"]
        + initial_astra[baseline_name]["wall_s"]
    ) / 128
    final_path = campaign.root / "final_selection.json"
    if final_path.exists():
        selection = json.loads(final_path.read_text())
        if (
            selection["challenger"] != challenger
            or selection["profile"] != final_profile
        ):
            raise ValueError("Final selection changed")
    else:
        games = confirmation_games(campaign.deadline - time.monotonic(), final_per_game)
        if not games:
            raise TrainingWindowExhausted(
                "Too little time for a meaningful final comparison"
            )
        selection = {
            "challenger_label": challenger_label,
            "challenger": challenger,
            "profile": final_profile,
            "baseline": baseline,
            "baseline_profile": profiles[baseline_name],
            "games": games,
            "measured_suite_seconds_per_game": final_per_game,
        }
        write_report(final_path, selection)
    games = selection["games"]
    final = {}
    for label, path, opponent, search, opponent_search in [
        (
            "head_to_head",
            challenger["checkpoint"],
            baseline["checkpoint"],
            challenger_search,
            baseline_search,
        ),
        (
            "challenger_astra",
            challenger["checkpoint"],
            "astra",
            challenger_search,
            SEARCH,
        ),
        ("baseline_astra", baseline["checkpoint"], "astra", baseline_search, SEARCH),
    ]:
        final[label] = match(
            "confirmation_" + label,
            path,
            opponent,
            search,
            opponent_search,
            games=games,
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
        SearchConfig(**final_profile["serving_search"]),
        SearchConfig(**profiles[baseline_name]["serving_search"]),
        games=8,
        confirmation=True,
        cpu=True,
    )
    delta = paired_serving_difference(
        final["challenger_astra"], final["baseline_astra"]
    )
    head = final["head_to_head"]["summary"]
    audit = final["cpu_audit"]["timed_move_latency"].get("candidate", {})
    timing_passed = bool(audit.get("count")) and audit["over_2s"] == 0
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
        "astra_difference": delta,
        "cpu_timing_passed": timing_passed,
        "final_selection": selection,
        "final": {
            k: {
                "summary": v["summary"],
                "wall_s": v["wall_s"],
                "timed_move_latency": v.get("timed_move_latency", {}),
            }
            for k, v in final.items()
        },
        "best_arms": best_arms,
        "elapsed_s": time.time() - budget["started_at"],
        "automatic_deployment": False,
    }
    for path, digest in prepared["hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError("Frozen inputs changed during campaign")
    write_report(result_path, result)
    report = (
        f"# 24-hour serving-strength and distillation results\n\n"
        f"Selected: `{result['selected_checkpoint']}`\n\n"
        f"Challenger vs baseline: {head['match_score']:.2%}, 95% CI {head['match_score_ci95']}. "
        f"Provisional improvement: {provisional}; resolved improvement: {result['resolved_improvement']}.\n\n"
        f"Astra difference: {delta['match_score_difference']:+.2%}; CPU timing passed: {timing_passed}. "
        f"Local CPU audit only; no deployment. See distillation_campaign.json for full results.\n"
    )
    (campaign.root / "REPORT.md").write_text(report)
    campaign.status(
        "distillation_campaign_complete",
        selected_checkpoint=result["selected_checkpoint"],
    )
    return result
