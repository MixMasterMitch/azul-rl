"""Eight-hour matched continuation from provisional weights and tagged replay."""

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
from agent.scripts.finetune_campaign import DIAGNOSTIC_ENVIRONMENT
from agent.scripts.league_campaign import SEARCH, freeze_file
from agent.scripts.lr_campaign import train_segment
from agent.scripts.noise_study import paired_difference
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

ARMS = {"quarter": 0.25, "half": 0.5}


def freeze_league(source: Path, destination: Path) -> None:
    manifest = json.loads((source / "league.json").read_text())
    for entry in manifest["entries"]:
        path = Path(entry["path"])
        if entry.get("active", True):
            freeze_file(
                path if path.is_absolute() else source / path, destination / path.name
            )
        entry["path"] = path.name
    write_report(destination / "league.json", manifest)


def prepare(campaign: Campaign, inputs: Path) -> dict:
    source = json.loads(inputs.read_text())
    marker = campaign.root / "prepared.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["inputs"] != source:
            raise ValueError("Refinement inputs changed")
        for path, digest in result["input_hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return result
    frozen = campaign.root / "initializers"
    for name in ("start", "previous", "replay"):
        freeze_file(Path(source[name]), frozen / f"{name}.pt")
    freeze_league(Path(source["league"]), frozen / "league")
    net, weights = load_net_from_checkpoint(frozen / "start.pt")
    if (
        net.hidden != 256
        or net.arch != "source_attn"
        or weights.get("trained_player_counts") != [2]
    ):
        raise ValueError("Expected current two-player width-256 weights")
    donor = load_checkpoint_payload(frozen / "replay.pt")
    buffer = donor["buffer"]
    budgets = buffer["policy_sims"][: buffer["size"]]
    if not buffer["size"] or not bool((budgets > 0).all()):
        raise ValueError(
            "Every initial replay position must have a known search budget"
        )
    if donor["reward_semantics_version"] != weights["reward_semantics_version"]:
        raise ValueError("Replay and weights have different reward semantics")
    config = LoopConfig(**weights["config"])
    if (
        buffer["capacity"] != config.replay_capacity
        or donor.get("encoder_version") != weights.get("encoder_version")
        or donor.get("trained_player_counts") != [2]
    ):
        raise ValueError("Replay donor is incompatible with selected weights")
    arms = {}
    for name, weight in ARMS.items():
        directory = campaign.root / "experiments" / f"{name}_seed{campaign.seed}"
        arms[name] = asdict(
            replace(
                config,
                policy_fast_weight=weight,
                seed=campaign.seed,
                run_id=directory.name,
                runs_root=str(directory.parent),
                league_root=str(directory / "league"),
                init_from=str(frozen / "start.pt"),
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
        "initialization": "New forks: selected one-hour weights, identical fully tagged donor replay, fresh optimizer and RNG; not an exact resume.",
        "replay_size": buffer["size"],
        "replay_donor_iteration": donor["iteration"],
        "budget_counts": {
            str(int(k)): int((budgets == k).sum()) for k in budgets.unique()
        },
        "frozen": {
            name: str(frozen / f"{name}.pt") for name in ("start", "previous", "replay")
        },
        "input_hashes": {
            str(p): checkpoint_hash(p) for p in frozen.rglob("*") if p.is_file()
        },
    }
    write_report(marker, result)
    return result


def fork_arm(prepared: dict, arm: str) -> dict:
    """Create arms lazily to avoid keeping two unused replay copies on disk."""
    config = LoopConfig(**prepared["arms"][arm])
    directory = Path(config.runs_root) / config.run_id
    identity = {"config": asdict(config), "inputs": prepared["input_hashes"]}
    marker = directory / "refine_fork.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["identity"] != identity:
            raise ValueError("Refinement fork changed")
        return result
    if directory.exists():
        raise ValueError("Cannot overwrite an unmarked refinement fork")
    staging = directory.with_name(directory.name + ".preparing")
    if staging.exists():
        shutil.rmtree(staging)
    freeze_league(
        Path(prepared["frozen"]["start"]).parent / "league", staging / "league"
    )
    seed_all(config.seed)
    net, payload = load_net_from_checkpoint(prepared["frozen"]["start"])
    donor = load_checkpoint_payload(prepared["frozen"]["replay"])
    buffer = donor["buffer"]
    # Preserve actual sample ages while resetting the new fork's iteration clock.
    buffer["inserted_at"] = buffer["inserted_at"] - buffer["iteration"]
    buffer.update(iteration=0, total_sampled=0)
    payload.update(
        buffer=buffer,
        optimizer_state_dict=make_optimizer(
            net, config.lr, config.weight_decay
        ).state_dict(),
        config=asdict(config),
        iteration=0,
        progress={"training_wall_s": 0.0},
        rng_state=capture_rng_state(),
        checkpoint_compression="deflate",
    )
    save_checkpoint_payload(staging / "checkpoints/latest_resume.pt", payload)
    result = {
        "identity": identity,
        "fresh_optimizer": not payload["optimizer_state_dict"]["state"],
        "replay_size": buffer["size"],
        "initial_iteration": 0,
        "sha256": checkpoint_hash(staging / "checkpoints/latest_resume.pt"),
    }
    write_report(staging / "refine_fork.json", result)
    staging.rename(directory)
    return result


def candidate_rank(scores: dict, baseline_astra: float) -> tuple:
    return (scores["astra"] >= baseline_astra - 0.03, scores["start"], scores["astra"])


def retain_best(
    campaign: Campaign,
    arm: str,
    milestone: dict,
    scores: dict,
    baseline_astra: float,
    config: LoopConfig,
) -> dict:
    marker = campaign.root / f"best_{arm}.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous and candidate_rank(scores, baseline_astra) <= candidate_rank(
        previous["scores"], baseline_astra
    ):
        if checkpoint_hash(previous["resume"]) != previous["resume_sha256"]:
            raise ValueError("Retained full checkpoint changed")
        return previous
    directory = Path(config.runs_root) / config.run_id
    resume = directory / "checkpoints/latest_resume.pt"
    payload = load_checkpoint_payload(resume)
    if payload["iteration"] != milestone["iteration"]:
        raise ValueError("Milestone does not match the available full training state")
    weights = load_checkpoint_payload(milestone["checkpoint"])
    if any(
        not torch.equal(v, payload["model_state_dict"][k])
        for k, v in weights["model_state_dict"].items()
    ):
        raise ValueError("Full state and milestone weights differ")
    del payload, weights
    archive = campaign.root / "retained" / arm / Path(milestone["checkpoint"]).stem
    campaign.status("archiving", arm=arm, checkpoint=milestone["checkpoint"])
    # The bounded-storage writer already uses lossless DEFLATE level 6.
    # Atomic future saves replace the source inode, preserving this archive.
    freeze_file(resume, archive / "resume.pt")
    freeze_league(directory / "league", archive / "league")
    record = {
        "milestone": milestone,
        "scores": scores,
        "resume": str(archive / "resume.pt"),
        "resume_sha256": checkpoint_hash(archive / "resume.pt"),
        "league": str(archive / "league"),
    }
    write_report(marker, record)
    if previous:
        old = Path(previous["resume"]).parent
        if old.parent != archive.parent or old == archive:
            raise ValueError("Unsafe archive retirement path")
        shutil.rmtree(old)
    return record


def retire_completed_arm(campaign: Campaign, config: LoopConfig, best: dict) -> None:
    """Retain the best full state and all milestone weights, retire owned scratch."""
    directory = Path(config.runs_root) / config.run_id
    if directory.parent != campaign.root / "experiments":
        raise ValueError("Cannot retire a foreign experiment")
    if checkpoint_hash(best["resume"]) != best["resume_sha256"]:
        raise ValueError("Cannot retire scratch without verified full state")
    for path in (directory / "checkpoints").glob("*.pt"):
        path.unlink()
    write_report(
        directory / "retired.json",
        {
            "best_full_state": best["resume"],
            "reason": "Completed arm; all milestone weights and best matching optimizer/replay/RNG/league retained.",
        },
    )


def run_weight_refine_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise RuntimeError("Native-memory diagnostics must be enabled")
    validation = json.loads((campaign.root / "ready.json").read_text())
    if not validation.get("passed") or validation["code"] != campaign.code:
        raise ValueError("Validation missing or source changed since validation")
    final_path = campaign.root / "weight_refine_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        campaign.status(
            "weight_refine_complete", selected_checkpoint=result["selected_checkpoint"]
        )
        return result
    prepared = prepare(campaign, Path(initializer))
    plan = {
        "budget": budget,
        "prepared": prepared,
        "arms": ARMS,
        "max_minutes_per_arm": 150,
        "milestones_minutes": [30, 60, 150],
        "development_games": 512,
        "confirmation_games": 2048,
        "development_seed": campaign.seed + 210_000_000,
        "confirmation_seed": campaign.seed + 220_000_000,
        "search": asdict(SEARCH),
        "code": campaign.code,
        "selection": "Highest development score versus starting weighted candidate, within 3pp of starting Astra score. Initializer eligible.",
        "provisional_use": "Frozen candidate has held-out head-to-head point score >0.5 and Astra point delta >=-0.03.",
        "confirmed_improvement": "Held-out head-to-head 95% interval lower bound >0.5 and Astra point delta >=-0.03.",
        "retention": "All milestone weights/reports; best full optimizer/replay/RNG and matching league per arm. Discard only this run's superseded full states.",
        "automatic_deployment": False,
    }
    plan_path = campaign.root / "weight_refine_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Refinement protocol changed")
    write_report(plan_path, plan)
    allocation_path = campaign.root / "allocation.json"
    if not allocation_path.exists():
        minutes = min(150.0, (campaign.deadline - time.monotonic() - 150 * 60) / 120)
        if minutes < 60:
            raise TimeoutError("Insufficient time for matched training and evaluation")
        write_report(
            allocation_path,
            {
                "minutes_per_arm": minutes,
                "milestones": sorted(set([30.0, 60.0, minutes])),
                "evaluation_and_archival_reserve_minutes": 150,
            },
        )
    allocation = json.loads(allocation_path.read_text())

    def match(
        label: str,
        path: str,
        opponent: str,
        *,
        confirmation: bool = False,
        games: int | None = None,
        greedy: bool = False,
    ) -> dict:
        cfg = replace(
            campaign.arena(
                games or (2048 if confirmation else 512),
                search=SEARCH,
                greedy=greedy,
                split="confirmation" if confirmation else "development",
            ),
            seed=plan["confirmation_seed" if confirmation else "development_seed"]
            + (1_000_000 if greedy else 0),
        )
        report = campaign._match(label, path, opponent, cfg, SEARCH)
        if report["summary"]["unfinished"]:
            raise RuntimeError(f"Unfinished evaluation games: {label}")
        return report

    frozen = prepared["frozen"]
    baseline = match("development_start_astra", frozen["start"], "astra")
    baseline_astra = baseline["summary"]["match_score"]
    candidates = {
        "initializer": {
            "checkpoint": frozen["start"],
            "scores": {"start": 0.5, "astra": baseline_astra},
        }
    }
    best_arms = {}
    for index, arm in enumerate(ARMS):
        config = LoopConfig(**prepared["arms"][arm])
        fork_arm(prepared, arm)
        for minutes in allocation["milestones"]:
            label = f"{arm}_{minutes:g}m"
            milestone = train_segment(
                campaign,
                config,
                minutes,
                label,
                evaluation_reserve_minutes=95
                + (allocation["minutes_per_arm"] if index == 0 else 0),
            )
            if milestone["executed_target_minutes"] < minutes - 0.001:
                raise RuntimeError("Matched training allocation truncated")
            reports = {
                name: match(
                    f"development_{label}_{name}", milestone["checkpoint"], opponent
                )
                for name, opponent in [("start", frozen["start"]), ("astra", "astra")]
            }
            scores = {
                name: report["summary"]["match_score"]
                for name, report in reports.items()
            }
            candidates[label] = {
                "checkpoint": milestone["checkpoint"],
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
        retire_completed_arm(campaign, config, best_arms[arm])
    best = max(
        candidates,
        key=lambda name: candidate_rank(candidates[name]["scores"], baseline_astra),
    )
    selected = candidates[best]
    selection = {
        "candidate_label": best,
        "candidate_checkpoint": selected["checkpoint"],
        "candidate_sha256": checkpoint_hash(selected["checkpoint"]),
    }
    selection_path = campaign.root / "final_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Candidate changed after confirmation began")
    write_report(selection_path, selection)
    final = {}
    for label, path, opponent, games, greedy in [
        ("head_to_head", selected["checkpoint"], frozen["start"], 2048, False),
        ("candidate_astra", selected["checkpoint"], "astra", 2048, False),
        ("start_astra", frozen["start"], "astra", 2048, False),
        ("previous_champion", selected["checkpoint"], frozen["previous"], 1024, False),
        ("candidate_greedy_astra", selected["checkpoint"], "astra", 512, True),
        ("start_greedy_astra", frozen["start"], "astra", 512, True),
    ]:
        final[label] = match(
            "confirmation_" + label,
            path,
            opponent,
            confirmation=True,
            games=games,
            greedy=greedy,
        )
        write_report(
            campaign.root / "confirmation_progress.json",
            {name: report["summary"] for name, report in final.items()},
        )
    delta = paired_difference(final["candidate_astra"], final["start_astra"])
    head = final["head_to_head"]["summary"]
    acceptable = best != "initializer" and delta["match_score_difference"] >= -0.03
    provisional = acceptable and head["match_score"] > 0.5
    confirmed = acceptable and head["match_score_ci95"][0] > 0.5
    retained = selected["checkpoint"] if provisional else frozen["start"]
    resume = best_arms[selected["arm"]] if provisional else None
    for path, digest in prepared["input_hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError("Frozen input changed during training")
    result = {
        **selection,
        "selected_checkpoint": retained,
        "selected_sha256": checkpoint_hash(retained),
        "provisional_improvement": provisional,
        "improvement_resolved": confirmed,
        "selected_full_state": resume,
        "best_arms": best_arms,
        "development": candidates,
        "screens": {name: report["summary"] for name, report in final.items()},
        "astra_difference": delta,
        "allocation": allocation,
        "completed_at": time.time(),
        "elapsed_s": time.time() - budget["started_at"],
        "automatic_deployment": False,
        "source_hashes_verified": True,
    }
    write_report(final_path, result)
    rows = [
        "# Policy-weight refinement",
        "",
        f"Candidate: **{best}**. Provisional improvement: **{provisional}**. Confirmed head-to-head improvement: **{confirmed}**.",
        "",
        "| Checkpoint | Against starting candidate | Astra |",
        "|---|---:|---:|",
    ]
    rows += [
        f"| {name} | {r['scores']['start']:.1%} | {r['scores']['astra']:.1%} |"
        for name, r in candidates.items()
    ]
    rows += [
        "",
        f"Held-out head-to-head: {head['match_score']:.2%}; 95% interval {head['match_score_ci95']}.",
        f"Astra change: {delta['match_score_difference'] * 100:+.2f}pp; paired interval {delta['paired_ci95']}.",
        "",
        f"Recommended checkpoint: {retained}",
        "",
        prepared["initialization"],
        "",
        "One matched training seed; 64-simulation evaluation. An inconclusive interval does not imply no likely benefit. No deployment.",
        "",
        "[Full results](weight_refine_campaign.json) · [Protocol](weight_refine_plan.json)",
    ]
    (campaign.root / "REPORT.md").write_text("\n".join(rows) + "\n")
    campaign.status(
        "weight_refine_complete",
        selected_checkpoint=retained,
        improvement_resolved=confirmed,
    )
    return result
