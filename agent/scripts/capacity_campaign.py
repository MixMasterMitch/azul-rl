"""Twenty-hour capacity study, with CPU-qualified serving-budget comparisons."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

import torch

from agent.eval.arena import checkpoint_hash, paired_interval, write_report
from agent.eval.latency import benchmark_latency
from agent.scripts.finetune_campaign import DIAGNOSTIC_ENVIRONMENT
from agent.scripts.league_campaign import SEARCH, freeze_file
from agent.scripts.lr_campaign import TrainingWindowExhausted, train_segment
from agent.scripts.weight_refine_campaign import (
    candidate_rank,
    freeze_league,
    retain_best,
    retire_completed_arm,
)
from agent.search.config import SearchConfig
from agent.train.checkpointing import (
    load_checkpoint_payload,
    load_net_from_checkpoint,
    save_checkpoint_payload,
)
from agent.train.learner import make_optimizer
from agent.train.loop import LoopConfig
from agent.train.reproducibility import capture_rng_state, seed_all

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


def paired_serving_difference(candidate: dict, control: dict) -> dict:
    """Compare CPU-qualified policies on their common, complete paired-seed prefix.

    Candidate search budgets may differ deliberately. Opponent settings and all
    other evaluation settings must match. Counts can differ for auxiliary screens.
    """
    for key in candidate["config"]:
        if (
            key not in {"search", "num_games"}
            and candidate["config"][key] != control["config"][key]
        ):
            raise ValueError(f"Unmatched serving evaluation setting: {key}")
    for key in ("opponent", "opponent_sha256", "opponent_identity", "opponent_search"):
        if candidate[key] != control[key]:
            raise ValueError(f"Unmatched serving opponent: {key}")
    for report in (candidate, control):
        count, seed = report["config"]["num_games"], report["config"]["seed"]
        if count < 2 or count % 2 or report["summary"]["unfinished"]:
            raise ValueError("Serving comparison requires finished seed pairs")
        expected = [(seed + i // 2, i % 2) for i in range(count)]
        actual = [(r["pair_seed"], r["candidate_seat"]) for r in report["records"]]
        if actual != expected or any(
            r["match_score"] not in (0.0, 0.5, 1.0) or r["outcome"] == "unfinished"
            for r in report["records"]
        ):
            raise ValueError("Unmatched or invalid serving seed pairs")
    count = min(candidate["config"]["num_games"], control["config"]["num_games"])
    differences = [
        sum(
            candidate["records"][j]["match_score"]
            - control["records"][j]["match_score"]
            for j in (i, i + 1)
        )
        / 2
        for i in range(0, count, 2)
    ]
    return {
        "match_score_difference": sum(differences) / len(differences),
        "paired_ci95": paired_interval(differences),
        "seed_pairs": count // 2,
        "candidate_search": candidate["config"]["search"],
        "control_search": control["config"]["search"],
    }


def prepare(campaign: Campaign, inputs: Path) -> dict:
    marker = campaign.root / "prepared.json"
    source = json.loads(inputs.read_text())
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["inputs"] != source:
            raise ValueError("Capacity inputs changed")
        for path, digest in result["input_hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return result
    frozen = campaign.root / "initializers"
    for name in ("teacher", "source_resume", "previous"):
        freeze_file(Path(source[name]), frozen / f"{name}.pt")
    freeze_league(Path(source["league"]), frozen / "league")
    wide, weights = load_net_from_checkpoint(frozen / "wide512.pt")
    teacher, teacher_weights = load_net_from_checkpoint(frozen / "teacher.pt")
    if wide.hidden != 512 or teacher.hidden != 256 or wide.arch != "source_attn":
        raise ValueError(
            "Expected explicitly transferred source-attention 256 and 512 models"
        )
    if weights.get("trained_player_counts") != [2] or teacher_weights.get(
        "trained_player_counts"
    ) != [2]:
        raise ValueError("Expected two-player checkpoints")
    donor = load_checkpoint_payload(frozen / "source_resume.pt")
    if any(
        not torch.equal(v, donor["model_state_dict"][k])
        for k, v in teacher_weights["model_state_dict"].items()
    ):
        raise ValueError("Replay donor must match the selected teacher")
    buffer = donor["buffer"]
    budgets = buffer["policy_sims"][: buffer["size"]]
    if not buffer["size"] or not bool((budgets > 0).all()):
        raise ValueError("Initial replay must have known search budgets")
    if donor["reward_semantics_version"] != weights["reward_semantics_version"]:
        raise ValueError("Reward semantics differ")
    transfer = json.loads((campaign.root / "transfer_validation.json").read_text())
    if any(
        r["argmax_agreement"] < 0.999
        or r["policy_kl_mean"] > 1e-5
        or r["value_mse"] > 1e-6
        for r in transfer.values()
    ):
        raise ValueError("Width transfer failed real-replay validation")
    config = LoopConfig(**teacher_weights["config"])
    arms = {}
    for name, hidden, lr in [
        ("control", 256, config.lr),
        ("wide_current", 512, config.lr),
        ("wide_lower", 512, 0.0003),
    ]:
        directory = campaign.root / "experiments" / f"{name}_seed{campaign.seed}"
        arms[name] = asdict(
            replace(
                config,
                hidden=hidden,
                lr=lr,
                seed=campaign.seed,
                run_id=directory.name,
                runs_root=str(directory.parent),
                league_root=str(directory / "league"),
                init_from=str(
                    frozen / ("teacher.pt" if hidden == 256 else "wide512.pt")
                ),
                device=campaign.device,
                provenance=None,
                max_wall_minutes=150.0,
                search_inference_cache_size=0,
                keep_recent_checkpoints=1,
                bounded_checkpoint_storage=True,
            )
        )
    result = {
        "inputs": source,
        "arms": arms,
        "replay_size": buffer["size"],
        "frozen": {
            name: str(frozen / f"{name}.pt")
            for name in ("teacher", "wide512", "source_resume", "previous")
        },
        "input_hashes": {
            str(p): checkpoint_hash(p) for p in frozen.rglob("*") if p.is_file()
        },
        "initialization": "Integer width expansion; identical tagged replay and fresh optimizers for all pilot arms. Continuation retains the best pilot optimizer/replay/RNG.",
    }
    write_report(marker, result)
    return result


def fork_arm(
    prepared: dict, config: LoopConfig, *, continuation: dict | None = None
) -> dict:
    directory = Path(config.runs_root) / config.run_id
    identity = {
        "config": asdict(config),
        "inputs": prepared["input_hashes"],
        "continuation": continuation,
    }
    marker = directory / "capacity_fork.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["identity"] != identity:
            raise ValueError("Capacity fork changed")
        return result
    if directory.exists():
        raise ValueError("Cannot overwrite an unmarked capacity fork")
    staging = directory.with_name(directory.name + ".preparing")
    if staging.exists():
        shutil.rmtree(staging)
    if continuation:
        if checkpoint_hash(continuation["resume"]) != continuation["resume_sha256"]:
            raise ValueError("Continuation checkpoint changed")
        payload = load_checkpoint_payload(continuation["resume"])
        freeze_league(Path(continuation["league"]), staging / "league")
    else:
        freeze_league(
            Path(prepared["frozen"]["teacher"]).parent / "league", staging / "league"
        )
        seed_all(config.seed)
        net, payload = load_net_from_checkpoint(config.init_from)
        donor = load_checkpoint_payload(prepared["frozen"]["source_resume"])
        buffer = donor["buffer"]
        buffer["inserted_at"] = buffer["inserted_at"] - buffer["iteration"]
        buffer.update(iteration=0, total_sampled=0)
        payload.update(
            buffer=buffer,
            optimizer_state_dict=make_optimizer(
                net, config.lr, config.weight_decay
            ).state_dict(),
            iteration=0,
            rng_state=capture_rng_state(),
        )
    payload.update(
        config=asdict(config),
        progress={"training_wall_s": 0.0},
        checkpoint_compression="deflate",
    )
    save_checkpoint_payload(staging / "checkpoints/latest_resume.pt", payload)
    result = {
        "identity": identity,
        "fresh_optimizer": not payload["optimizer_state_dict"]["state"],
        "replay_size": payload["buffer"]["size"],
        "initial_iteration": payload["iteration"],
        "sha256": checkpoint_hash(staging / "checkpoints/latest_resume.pt"),
    }
    write_report(staging / "capacity_fork.json", result)
    staging.rename(directory)
    return result


def serving_profile(campaign: Campaign, label: str, checkpoint: str) -> dict:
    """Use the largest completed tested budget, with 0.2s application headroom."""
    marker = campaign.root / f"serving_{label}.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if checkpoint_hash(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Serving candidate changed")
        return result
    records, selected = [], None
    for simulations in (64, 128, 256, 512, 1024, 1536):
        campaign.status("latency", candidate=label, simulations=simulations)
        report = benchmark_latency(
            checkpoint,
            replace(SEARCH, num_simulations=simulations),
            seed=campaign.seed + 230_000_000,
            games=8,
            deadline_s=1.8,
        )
        records.append(report)
        write_report(campaign.root / "latency" / f"{label}_{simulations}.json", report)
        if not report["qualified"]:
            break
        selected = simulations
    if selected is None:
        raise ValueError(
            "Candidate fails even 64 simulations within the CPU move budget"
        )
    result = {
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "search": asdict(replace(SEARCH, num_simulations=selected)),
        "serving_search": asdict(
            replace(SEARCH, num_simulations=selected, move_deadline_s=1.8)
        ),
        "records": records,
        "local_cpu_only": True,
        "move_budget_s": 2.0,
    }
    write_report(marker, result)
    return result


def retire_arm(campaign: Campaign, config: LoopConfig, best: dict) -> None:
    """The frozen best state owns its league; discard completed experiment scratch."""
    retire_completed_arm(campaign, config, best)
    league = Path(config.league_root)
    if league.parent.parent != campaign.root / "experiments":
        raise ValueError("Cannot retire a foreign league")
    archived = Path(best["league"])
    manifest = json.loads((archived / "league.json").read_text())
    if any(
        not (archived / entry["path"]).is_file()
        for entry in manifest["entries"]
        if entry.get("active", True)
    ):
        raise ValueError("Archived league is incomplete")
    for path in league.glob("*.pt"):
        path.unlink()
    write_report(league / "retired.json", {"retained_matching_league": str(archived)})


def run_capacity_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise RuntimeError("Native-memory diagnostics must be enabled")
    validation = json.loads((campaign.root / "ready.json").read_text())
    if not validation.get("passed") or validation["code"] != campaign.code:
        raise ValueError("Validation missing or source changed since validation")
    final_path = campaign.root / "capacity_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        campaign.status(
            "capacity_campaign_complete",
            selected_checkpoint=result["selected_checkpoint"],
        )
        return result
    prepared = prepare(campaign, Path(initializer))
    frozen = prepared["frozen"]
    plan = {
        "budget": budget,
        "prepared": prepared,
        "code": campaign.code,
        "pilot_minutes": {
            "control": [30, 60],
            "wide_current": [30, 90, 150],
            "wide_lower": [30, 90, 150],
        },
        "continuation_minutes": [60, 180, 300, 420],
        "final_reserve_minutes": 360,
        "development_games": 256,
        "development_seed": campaign.seed + 240_000_000,
        "confirmation_seed": campaign.seed + 250_000_000,
        "serving_profile_tuning_games": 128,
        "confirmation_games": "Largest of 2048/1024/512/256 fitting measured development throughput; frozen before confirmation.",
        "selection": "Best development score vs teacher within 3pp of baseline Astra, using equal 64 simulations. Final comparisons also use separately CPU-qualified serving budgets.",
        "provisional_use": "Serving-budget held-out head-to-head point score >50%, Astra delta >=-3pp; confidence reported separately.",
        "automatic_deployment": False,
        "move_budget_s": 2.0,
        "cpu_search_deadline_s": 1.8,
        "retention": "All weights and evaluation records; best full state and league per arm; retire only owned scratch.",
    }
    marker = campaign.root / "capacity_plan.json"
    if marker.exists() and json.loads(marker.read_text()) != plan:
        raise ValueError("Capacity protocol changed")
    write_report(marker, plan)

    def match(
        label: str,
        path: str,
        opponent: str,
        *,
        games: int = 256,
        confirmation: bool = False,
        search: SearchConfig = SEARCH,
        opponent_search: SearchConfig = SEARCH,
        cpu: bool = False,
        greedy: bool = False,
    ) -> dict:
        cfg = replace(
            campaign.arena(
                games,
                search=search,
                greedy=greedy,
                split="confirmation" if confirmation else "development",
            ),
            seed=plan["confirmation_seed" if confirmation else "development_seed"]
            + (1_000_000 if greedy else 0)
            + (2_000_000 if cpu else 0),
            inference_device="cpu" if cpu else campaign.device,
            game_batch_size=2 if cpu else 128,
        )
        result = campaign._match(label, path, opponent, cfg, opponent_search)
        if result["summary"]["unfinished"]:
            raise RuntimeError(f"Unfinished evaluation games: {label}")
        return result

    baseline = match("development_teacher_astra", frozen["teacher"], "astra")
    baseline_astra = baseline["summary"]["match_score"]
    candidates = {
        "teacher": {
            "checkpoint": frozen["teacher"],
            "scores": {"start": 0.5, "astra": baseline_astra},
            "hidden": 256,
        }
    }
    best_arms = {}

    def screen(arm: str, milestone: dict, config: LoopConfig, label: str) -> None:
        scores = {
            name: match(
                f"development_{label}_{name}", milestone["checkpoint"], opponent
            )["summary"]["match_score"]
            for name, opponent in [("start", frozen["teacher"]), ("astra", "astra")]
        }
        candidates[label] = {
            "checkpoint": milestone["checkpoint"],
            "scores": scores,
            "arm": arm,
            "hidden": config.hidden,
        }
        best_arms[arm] = retain_best(
            campaign, arm, milestone, scores, baseline_astra, config
        )
        write_report(
            campaign.root / "development.json",
            {"candidates": candidates, "best_arms": best_arms},
        )

    # Give the requested larger model first access to the training window.
    for arm in ("wide_current", "wide_lower", "control"):
        config = LoopConfig(**prepared["arms"][arm])
        campaign.status("forking", arm=arm)
        fork_arm(prepared, config)
        for minutes in plan["pilot_minutes"][arm]:
            label = f"{arm}_{minutes}m"
            milestone = train_segment(
                campaign, config, minutes, label, evaluation_reserve_minutes=360
            )
            screen(arm, milestone, config, label)
        retire_arm(campaign, config, best_arms[arm])

    # Fork the strongest larger-model pilot's full state, including optimizer.
    continuation_path = campaign.root / "continuation_selection.json"
    chosen = max(
        ("wide_current", "wide_lower"),
        key=lambda a: candidate_rank(best_arms[a]["scores"], baseline_astra),
    )
    continuation = {"arm": chosen, "state": best_arms[chosen]}
    if (
        continuation_path.exists()
        and json.loads(continuation_path.read_text()) != continuation
    ):
        raise ValueError("Continuation selection changed")
    write_report(continuation_path, continuation)
    directory = campaign.root / "experiments" / f"continuation_seed{campaign.seed}"
    config = replace(
        LoopConfig(**prepared["arms"][chosen]),
        run_id=directory.name,
        league_root=str(directory / "league"),
        init_from=best_arms[chosen]["milestone"]["checkpoint"],
        max_wall_minutes=420.0,
    )
    allocation_path = campaign.root / "continuation_allocation.json"
    if not allocation_path.exists():
        available = max(0.0, (campaign.deadline - time.monotonic()) / 60 - 420)
        minutes = min(420.0, available)
        write_report(
            allocation_path,
            {
                "minutes": minutes,
                "milestones": sorted(
                    set(
                        [m for m in plan["continuation_minutes"] if m < minutes]
                        + ([minutes] if minutes >= 30 else [])
                    )
                ),
            },
        )
    allocation = json.loads(allocation_path.read_text())
    if allocation["milestones"]:
        campaign.status("forking", arm="continuation")
        fork_arm(prepared, config, continuation=best_arms[chosen])
        stop_path = campaign.root / "continuation_stop.json"
        last_target = 0.0
        for minutes in allocation["milestones"]:
            label = f"continuation_{minutes:g}m"
            if (
                stop_path.exists()
                and minutes > json.loads(stop_path.read_text())["last_target_minutes"]
            ):
                break
            complete = directory / "milestones" / f"{label}.json"
            if (
                not complete.exists()
                and shutil.disk_usage(campaign.root).free < 0.9 * 1024**3
            ):
                write_report(
                    stop_path,
                    {
                        "last_target_minutes": last_target,
                        "reason": "Reserve disk for atomic checkpoints and final evaluation artifacts.",
                    },
                )
                break
            try:
                milestone = train_segment(
                    campaign, config, minutes, label, evaluation_reserve_minutes=360
                )
            except TrainingWindowExhausted:
                write_report(
                    stop_path,
                    {
                        "last_target_minutes": last_target,
                        "reason": "Remaining time reserved for final evaluation.",
                    },
                )
                break
            screen("continuation", milestone, config, label)
            last_target = minutes
        if "continuation" in best_arms:
            retire_arm(campaign, config, best_arms["continuation"])
        else:
            retire_arm(campaign, config, best_arms[chosen])

    # Freeze finalists before any confirmation games. Keep the teacher eligible.
    wide_label = max(
        (k for k, v in candidates.items() if v["hidden"] == 512),
        key=lambda k: candidate_rank(candidates[k]["scores"], baseline_astra),
    )
    small_label = max(
        (k for k, v in candidates.items() if v["hidden"] == 256),
        key=lambda k: candidate_rank(candidates[k]["scores"], baseline_astra),
    )
    finalist = {
        "wide": candidates[wide_label],
        "small": candidates[small_label],
        "wide_label": wide_label,
        "small_label": small_label,
    }
    selection_path = campaign.root / "final_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != finalist:
        raise ValueError("Finalists changed after confirmation began")
    write_report(selection_path, finalist)
    wide, small = finalist["wide"]["checkpoint"], finalist["small"]["checkpoint"]
    profile_paths = {"wide": wide, "small": small}
    if small_label != "teacher":
        profile_paths["teacher"] = frozen["teacher"]
    profiles = {
        name: serving_profile(campaign, name, path)
        for name, path in profile_paths.items()
    }
    tuning_path = campaign.root / "serving_profile_selection.json"
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text())
    else:
        tuning = {}
        for name, path in profile_paths.items():
            maximum = profiles[name]["search"]["num_simulations"]
            reports = {}
            for simulations in sorted({64, max(64, maximum // 2), maximum}):
                report = match(
                    f"profile_{name}_{simulations}_astra",
                    path,
                    "astra",
                    games=128,
                    search=replace(SEARCH, num_simulations=simulations),
                )
                reports[str(simulations)] = {
                    **report["summary"],
                    "wall_s": report.get("wall_s", 0.0),
                }
            selected = max(reports, key=lambda n: (reports[n]["match_score"], int(n)))
            tuning[name] = {
                "selected_simulations": int(selected),
                "development": reports,
                "checkpoint_sha256": checkpoint_hash(path),
            }
        write_report(tuning_path, tuning)
    for name, path in profile_paths.items():
        if tuning[name]["checkpoint_sha256"] != checkpoint_hash(path):
            raise ValueError("Serving profile tuning checkpoint changed")
        profiles[name]["search"]["num_simulations"] = tuning[name][
            "selected_simulations"
        ]
        profiles[name]["serving_search"]["num_simulations"] = tuning[name][
            "selected_simulations"
        ]
    searches = {
        name: SearchConfig(**profile["search"]) for name, profile in profiles.items()
    }
    confirmation_path = campaign.root / "confirmation_allocation.json"
    if not confirmation_path.exists():
        # Two Astra matches plus head-to-head cost no more than approximately
        # twice the summed Astra probes. Add a 25% margin and separate CPU time.
        per_game_s = (
            2.5
            * sum(
                tuning[n]["development"][str(tuning[n]["selected_simulations"])][
                    "wall_s"
                ]
                for n in ("wide", "small")
            )
            / 128
        )
        fixed_reserve_s = (90 if small_label == "teacher" else 150) * 60
        available_s = campaign.deadline - time.monotonic() - fixed_reserve_s
        games = next(
            (n for n in (2048, 1024, 512, 256) if n * per_game_s <= available_s), 256
        )
        write_report(
            confirmation_path,
            {
                "games_per_primary_match": games,
                "estimated_primary_seconds_per_game": per_game_s,
                "fixed_reserve_s": fixed_reserve_s,
                "basis": "Development timing only; no confirmation outcomes inspected.",
            },
        )
    confirmation_allocation = json.loads(confirmation_path.read_text())
    primary_games = confirmation_allocation["games_per_primary_match"]
    final = {}

    def confirm(name: str, path: str, opponent: str, **kwargs: object) -> dict:
        report = match(
            "confirmation_" + name, path, opponent, confirmation=True, **kwargs
        )
        final[name] = report
        write_report(
            campaign.root / "confirmation_progress.json",
            {k: v["summary"] for k, v in final.items()},
        )
        return report

    # Main serving comparison gets fresh paired seeds and the same 2s ceiling.
    confirm(
        "serving_head_to_head",
        wide,
        small,
        games=primary_games,
        search=searches["wide"],
        opponent_search=searches["small"],
    )
    confirm(
        "serving_wide_astra",
        wide,
        "astra",
        games=primary_games,
        search=searches["wide"],
    )
    confirm(
        "serving_small_astra",
        small,
        "astra",
        games=primary_games,
        search=searches["small"],
    )
    confirm("equal64_head_to_head", wide, small, games=2048)
    confirm("equal64_wide_astra", wide, "astra", games=512)
    confirm("equal64_teacher_astra", frozen["teacher"], "astra", games=512)
    # Actual CPU games audit the profiles with their serving deadlines enabled.
    confirm(
        "cpu_timed_head_to_head",
        wide,
        small,
        games=16,
        cpu=True,
        search=SearchConfig(**profiles["wide"]["serving_search"]),
        opponent_search=SearchConfig(**profiles["small"]["serving_search"]),
    )
    delta = paired_serving_difference(
        final["serving_wide_astra"], final["serving_small_astra"]
    )
    head = final["serving_head_to_head"]["summary"]
    provisional = head["match_score"] > 0.5 and delta["match_score_difference"] >= -0.03
    confirmed = provisional and head["match_score_ci95"][0] > 0.5
    # The smaller control must independently beat the frozen teacher before use.
    small_eligible = small_label == "teacher"
    if small_label != "teacher":
        report = confirm(
            "small_control_vs_teacher",
            small,
            frozen["teacher"],
            games=512,
            search=searches["small"],
            opponent_search=searches["teacher"],
        )
        astra = confirm(
            "serving_teacher_astra",
            frozen["teacher"],
            "astra",
            games=512,
            search=searches["teacher"],
        )
        small_eligible = (
            report["summary"]["match_score"] > 0.5
            and paired_serving_difference(final["serving_small_astra"], astra)[
                "match_score_difference"
            ]
            >= -0.03
        )
        against_teacher = confirm(
            "wide_vs_teacher",
            wide,
            frozen["teacher"],
            games=512,
            search=searches["wide"],
            opponent_search=searches["teacher"],
        )
        teacher_delta = paired_serving_difference(final["serving_wide_astra"], astra)[
            "match_score_difference"
        ]
        provisional = (
            provisional
            and against_teacher["summary"]["match_score"] > 0.5
            and teacher_delta >= -0.03
        )
        confirmed = (
            provisional
            and confirmed
            and against_teacher["summary"]["match_score_ci95"][0] > 0.5
        )
    selected_label = (
        wide_label if provisional else small_label if small_eligible else "teacher"
    )
    selected = candidates[selected_label]
    for path, digest in prepared["input_hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError("Frozen input changed during campaign")
    result = {
        "selected_label": selected_label,
        "selected_checkpoint": selected["checkpoint"],
        "selected_sha256": checkpoint_hash(selected["checkpoint"]),
        "finalists": finalist,
        "provisional_larger_improvement": provisional,
        "resolved_larger_improvement": confirmed,
        "serving_profiles": profiles,
        "serving_profile_selection": tuning,
        "serving_astra_difference": delta,
        "screens": {k: v["summary"] for k, v in final.items()},
        "development": candidates,
        "best_full_states": best_arms,
        "allocation": allocation,
        "confirmation_allocation": confirmation_allocation,
        "completed_at": time.time(),
        "elapsed_s": time.time() - budget["started_at"],
        "automatic_deployment": False,
        "limitations": "Serving profiles measured on local one-thread CPU, not AWS Lambda. Primary games use fixed simulation budgets on GPU after CPU qualification; 16 actual timed CPU games are a small audit.",
    }
    write_report(final_path, result)
    rows = [
        "# Larger-model capacity study",
        "",
        f"Recommended checkpoint: **{selected_label}**.",
        "",
        f"Larger model serving-budget head-to-head: {head['match_score']:.2%}; 95% interval {head['match_score_ci95']}.",
        f"Astra change: {delta['match_score_difference'] * 100:+.2f}pp; paired interval {delta['paired_ci95']}.",
        f"Provisional larger-model improvement: {provisional}. Resolved: {confirmed}.",
        "",
        "| Checkpoint | Equal-64 vs teacher | Astra |",
        "|---|---:|---:|",
    ]
    rows += [
        f"| {name} | {v['scores']['start']:.1%} | {v['scores']['astra']:.1%} |"
        for name, v in candidates.items()
    ]
    rows += [
        "",
        result["limitations"],
        "",
        f"Checkpoint: {selected['checkpoint']}",
        "",
        "[Full results](capacity_campaign.json) · [Plan](capacity_plan.json)",
    ]
    (campaign.root / "REPORT.md").write_text("\n".join(rows) + "\n")
    campaign.status(
        "capacity_campaign_complete", selected_checkpoint=selected["checkpoint"]
    )
    return result
