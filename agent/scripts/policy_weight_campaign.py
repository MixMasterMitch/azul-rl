"""Matched policy-target weighting pilots after independent campaign closeout."""

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
from agent.scripts.league_campaign import (
    PANEL,
    SEARCH,
    choose_checkpoint,
    freeze_file,
    fork_training_state,
    panel_difference,
    panel_score,
)
from agent.scripts.lr_campaign import train_segment
from agent.train.checkpointing import (
    load_checkpoint_payload,
    load_net_from_checkpoint,
    save_checkpoint,
    save_checkpoint_payload,
)
from agent.train.learner import make_optimizer
from agent.train.loop import LoopConfig
from agent.train.replay_buffer import ReplayBuffer
from agent.train.reproducibility import seed_all
from agent.net import encoder as E

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign

ARMS = {"control": 1.0, "weighted": 0.25}


def fork_weight(
    source: Path, league: Path, directory: Path, config: LoopConfig
) -> dict:
    identity = {"source_sha256": checkpoint_hash(source), "config": asdict(config)}
    marker = directory / "weight_fork.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["identity"] != identity:
            raise ValueError("Policy-weight fork configuration changed")
        return previous
    if directory.exists():
        raise ValueError("Cannot replace an unmarked policy-weight experiment")
    stage = directory.with_name(directory.name + ".weight-staging")
    if stage.exists():
        shutil.rmtree(stage)
    fork_training_state(source, league, stage)
    payload = load_checkpoint_payload(source)
    allowed = {
        "policy_fast_weight",
        "run_id",
        "runs_root",
        "league_root",
        "init_from",
        "provenance",
        "device",
        "max_wall_minutes",
        "search_inference_cache_size",
    }
    unexpected = [
        k
        for k, v in asdict(config).items()
        if k in payload["config"] and payload["config"][k] != v and k not in allowed
    ]
    if unexpected or config.policy_fast_weight not in ARMS.values():
        raise ValueError(f"Undeclared policy-weight fork changes: {unexpected}")
    payload["config"] = asdict(config)
    if any(
        g["lr"] != config.lr for g in payload["optimizer_state_dict"]["param_groups"]
    ):
        raise ValueError("Fork optimizer learning rate mismatch")
    save_checkpoint_payload(stage / "checkpoints/latest_resume.pt", payload)
    record = {
        "identity": identity,
        "initial_iteration": payload["iteration"],
        "fork_sha256": checkpoint_hash(stage / "checkpoints/latest_resume.pt"),
    }
    write_report(stage / "weight_fork.json", record)
    stage.rename(directory)
    return record


def prepare(campaign: Campaign, closeout_path: Path) -> dict:
    marker = campaign.root / "prepared.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["closeout_sha256"] != checkpoint_hash(closeout_path):
            raise ValueError("Closeout result changed")
        for path, digest in record["input_hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return record
    closeout = json.loads(closeout_path.read_text())
    selected = Path(closeout["selected_checkpoint"])
    if checkpoint_hash(selected) != closeout["selected_sha256"]:
        raise ValueError("Confirmed initializer changed")
    net, weights = load_net_from_checkpoint(selected)
    if (
        net.hidden != 256
        or net.arch != "source_attn"
        or weights.get("trained_player_counts") != [2]
    ):
        raise ValueError("Expected a width-256 two-player source-attention initializer")
    config = LoopConfig(**weights["config"])
    label = (
        closeout["candidate_label"]
        if closeout["improvement_resolved"]
        else "initializer"
    )
    old_plan = json.loads((closeout_path.parent / "plan.json").read_text())
    full = (
        Path(closeout["source_resume"])
        if label == "initializer"
        else Path(old_plan["source_pause"]["checkpoint"])
        if label == "paused"
        else None
    )
    frozen = campaign.root / "initializers"
    frozen.mkdir(exist_ok=True)
    freeze_file(selected, frozen / "start.pt")
    prior = json.loads(
        (Path(old_plan["old_root"]) / "finetune_prepared.json").read_text()
    )
    freeze_file(Path(prior["frozen"]["previous"]), frozen / "previous.pt")
    # Use the same original opponent pool in both arms, including when the
    # selected weights have no corresponding optimizer/replay archive.
    league_source = Path(closeout["original_league_snapshot"])
    manifest = json.loads((league_source / "league.json").read_text())
    for entry in manifest["entries"]:
        if entry.get("active", True):
            freeze_file(
                league_source / Path(entry["path"]).name,
                frozen / "league" / Path(entry["path"]).name,
            )
        entry["path"] = Path(entry["path"]).name
    write_report(frozen / "league/league.json", manifest)
    common = frozen / "common_resume.pt"
    if full is not None:
        payload = load_checkpoint_payload(full)
        if not all(
            k in payload
            for k in ("optimizer_state_dict", "buffer", "rng_state", "progress")
        ):
            raise ValueError("Incomplete source continuation state")
        if any(
            not torch.equal(v, payload["model_state_dict"][k])
            for k, v in weights["model_state_dict"].items()
        ):
            raise ValueError("Full-state archive does not match selected weights")
        base_s = payload["progress"]["training_wall_s"]
        del payload
        freeze_file(full, common)
        initialization = (
            "full_state; legacy policy budgets remain unknown with neutral weight"
        )
    else:
        # A milestone stores weights only. Reset both arms identically rather
        # than attaching optimizer moments or replay from a later checkpoint.
        if not common.exists():
            seed_all(config.seed)
            replay = ReplayBuffer(
                config.replay_capacity,
                E.D_GLOBAL,
                E.NUM_SOURCES,
                E.D_SOURCE,
                300,
                4,
                snapshot_capacity=config.reanalysis_snapshot_capacity,
            )
            save_checkpoint(
                common,
                net,
                make_optimizer(net, config.lr, config.weight_decay),
                0,
                asdict(config),
                replay,
                {"training_wall_s": 0.0},
            )
            del replay
        else:
            payload = load_checkpoint_payload(common)
            if (
                payload["iteration"] != 0
                or payload["buffer"]["size"] != 0
                or payload["config"] != asdict(config)
                or any(
                    not torch.equal(v, payload["model_state_dict"][k])
                    for k, v in weights["model_state_dict"].items()
                )
            ):
                raise ValueError("Partially prepared common initializer changed")
            del payload
        base_s = 0.0
        initialization = (
            "selected_weights; identical fresh optimizer, replay and RNG in both arms"
        )
    del net, weights
    configs, forks = {}, {}
    for arm, weight in ARMS.items():
        directory = campaign.root / "experiments" / f"{arm}_seed{config.seed}"
        cfg = replace(
            config,
            policy_fast_weight=weight,
            search_inference_cache_size=0,
            run_id=directory.name,
            runs_root=str(directory.parent),
            league_root=str(directory / "league"),
            init_from=str(frozen / "start.pt"),
            device=campaign.device,
            provenance=None,
            max_wall_minutes=base_s / 60 + 120,
        )
        forks[arm] = fork_weight(common, frozen / "league", directory, cfg)
        configs[arm] = asdict(cfg)
    result = {
        "closeout_sha256": checkpoint_hash(closeout_path),
        "initialization": initialization,
        "base_training_wall_s": base_s,
        "arms": configs,
        "forks": forks,
        "frozen": {n: str(frozen / f"{n}.pt") for n in ("start", "previous")},
        "input_hashes": {str(p): checkpoint_hash(p) for p in frozen.rglob("*.pt")},
    }
    write_report(marker, result)
    return result


def run_policy_weight_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise RuntimeError("Native-memory diagnostics must be enabled")
    validation = json.loads((campaign.root / "ready.json").read_text())
    if not validation.get("passed") or validation["code"] != campaign.code:
        raise ValueError(
            "Tests/smoke validation missing or code changed since validation"
        )
    closeout_path = Path(initializer).resolve()
    closeout_deadline = budget["started_at"] + 2 * 3600
    while not closeout_path.exists():
        if time.time() >= closeout_deadline:
            raise TimeoutError("Closeout did not complete in its two-hour allocation")
        campaign.status("waiting_for_closeout", closeout=str(closeout_path))
        time.sleep(min(30, max(0.01, closeout_deadline - time.time())))
    final_path = campaign.root / "policy_weight_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        campaign.status(
            "policy_weight_complete", selected_checkpoint=result["selected_checkpoint"]
        )
        return result
    prepared = prepare(campaign, closeout_path)
    configs = {name: LoopConfig(**cfg) for name, cfg in prepared["arms"].items()}
    plan = {
        "study": "policy_weight",
        "budget": budget,
        "prepared": prepared,
        "arms": ARMS,
        "pilot_minutes": 120,
        "training_milestones": [60, 120],
        "development_games_per_opponent": 512,
        "confirmation_games_per_opponent": 2048,
        "greedy_games": 512,
        "development_seed": campaign.seed + 180_000_000,
        "confirmation_seed": campaign.seed + 190_000_000,
        "greedy_seed": campaign.seed + 200_000_000,
        "search": asdict(SEARCH),
        "selection": "Best panel score within 3pp of starting Astra score; initializer remains eligible.",
        "confirmation": "Paired panel CI lower bound >0 and Astra point delta >= -0.03.",
        "policy_weighting": "64-sim targets weight .25; >=256 weight 1; unknown weight 1; normalize within valid minibatch.",
        "value_weighting": "All finished-game outcomes retain unit weight; shared victory reward 0.",
        "code": campaign.code,
        "automatic_promotion": False,
    }
    plan_path = campaign.root / "policy_weight_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Policy-weight campaign protocol changed")
    write_report(plan_path, plan)
    allocation_path = campaign.root / "pilot_allocation.json"
    if allocation_path.exists():
        allocation = json.loads(allocation_path.read_text())
    else:
        # Preserve an hour for selection and two hours for confirmation. Both
        # pilots receive the same budget if setup consumed more than planned.
        available = campaign.deadline - time.monotonic() - 190 * 60
        minutes = min(120.0, available / 120)
        if minutes < 30:
            raise TimeoutError(
                "Insufficient time for meaningful matched pilots and confirmation"
            )
        allocation = {
            "minutes_per_arm": minutes,
            "milestones": [minutes / 2, minutes],
            "allocated_at": time.time(),
            "final_evaluation_reserve_minutes": 180,
        }
        write_report(allocation_path, allocation)
    milestones = {}
    for arm, cfg in configs.items():
        for index, minutes in enumerate(allocation["milestones"]):
            label = f"{arm}_{index + 1}"
            reserve = 180 + (allocation["minutes_per_arm"] if arm == "control" else 0)
            milestones[label] = train_segment(
                campaign,
                cfg,
                prepared["base_training_wall_s"] / 60 + minutes,
                label,
                evaluation_reserve_minutes=reserve,
            )
            if (
                milestones[label]["executed_target_minutes"]
                < milestones[label]["target_minutes"] - 0.001
            ):
                raise RuntimeError(
                    "Matched training allocation was truncated unexpectedly"
                )

    def panel(label: str, path: str, confirmation: bool = False) -> dict:
        cfg = replace(
            campaign.arena(
                2048 if confirmation else 512,
                search=SEARCH,
                split="confirmation" if confirmation else "development",
            ),
            seed=plan["confirmation_seed" if confirmation else "development_seed"],
        )
        reports = {
            n: campaign._match(
                f"{label}_{n}",
                path,
                "astra" if n == "astra" else prepared["frozen"][n],
                cfg,
                SEARCH,
            )
            for n in PANEL
        }
        if any(r["summary"]["unfinished"] for r in reports.values()):
            raise RuntimeError("Unfinished evaluation games")
        return reports

    baseline = panel("development_start", prepared["frozen"]["start"])
    candidates = {
        "initializer": baseline,
        **{k: panel(k, v["checkpoint"]) for k, v in milestones.items()},
    }
    paths = {
        "initializer": prepared["frozen"]["start"],
        **{k: v["checkpoint"] for k, v in milestones.items()},
    }
    best = choose_checkpoint(candidates, baseline["astra"]["summary"]["match_score"])
    selection = {
        "candidate_label": best,
        "candidate_checkpoint": paths[best],
        "candidate_sha256": checkpoint_hash(paths[best]),
    }
    selection_path = campaign.root / "final_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Candidate changed after independent confirmation started")
    write_report(selection_path, selection)
    start = panel("confirmation_start", paths["initializer"], True)
    final = (
        start
        if best == "initializer"
        else panel("confirmation_candidate", paths[best], True)
    )
    delta = None if best == "initializer" else panel_difference(final, start)
    passed = (
        delta is not None
        and delta["paired_ci95"][0] > 0
        and delta["opponents"]["astra"]["match_score_difference"] >= -0.03
    )
    greedy_cfg = replace(
        campaign.arena(512, search=SEARCH, greedy=True, split="confirmation"),
        seed=plan["greedy_seed"],
    )
    greedy_start = campaign._match(
        "greedy_start", paths["initializer"], "astra", greedy_cfg, SEARCH
    )
    greedy_final = (
        greedy_start
        if best == "initializer"
        else campaign._match(
            "greedy_candidate", paths[best], "astra", greedy_cfg, SEARCH
        )
    )
    if any(r["summary"]["unfinished"] for r in (greedy_start, greedy_final)):
        raise RuntimeError("Unfinished greedy evaluation")
    retained = paths[best] if passed else paths["initializer"]
    for path, digest in prepared["input_hashes"].items():
        if checkpoint_hash(path) != digest:
            raise RuntimeError(f"Frozen initializer changed: {path}")
    result = {
        **selection,
        "selected_checkpoint": retained,
        "selected_sha256": checkpoint_hash(retained),
        "improvement_resolved": passed,
        "panel_difference": delta,
        "allocation": allocation,
        "development": {
            n: {
                "panel_score": panel_score(r),
                "astra_score": r["astra"]["summary"]["match_score"],
            }
            for n, r in candidates.items()
        },
        "weighted_minus_control": panel_difference(
            candidates["weighted_2"], candidates["control_2"]
        ),
        "screens": {n: r["summary"] for n, r in final.items()},
        "start_screens": {n: r["summary"] for n, r in start.items()},
        "greedy": {
            "candidate": greedy_final["summary"],
            "start": greedy_start["summary"],
        },
        "full_resume_checkpoints": {
            n: str(Path(c.runs_root) / c.run_id / "checkpoints/latest_resume.pt")
            for n, c in configs.items()
        },
        "completed_at": time.time(),
        "elapsed_s": time.time() - budget["started_at"],
        "source_hashes_verified": True,
        "automatic_promotion": False,
    }
    write_report(final_path, result)
    rows = [
        "# Policy-target weighting campaign",
        "",
        f"Confirmed improvement: **{passed}**. Candidate: {best}.",
        "",
        "| Checkpoint | Development panel | Astra |",
        "|---|---:|---:|",
    ]
    for name, r in result["development"].items():
        rows.append(f"| {name} | {r['panel_score']:.1%} | {r['astra_score']:.1%} |")
    if delta:
        rows += [
            "",
            f"Held-out panel gain {delta['match_score_difference'] * 100:+.2f}pp; paired 95% interval {delta['paired_ci95']}.",
        ]
    rows += [
        "",
        f"Retained checkpoint: {retained}",
        "",
        prepared["initialization"],
        "",
        "Value targets and outcome objective unchanged. One training seed; fixed 64-simulation evaluation. No deployment.",
        "",
        "[Results](policy_weight_campaign.json) · [Previous campaign closeout](closeout/REPORT.md)",
    ]
    (campaign.root / "REPORT.md").write_text("\n".join(rows) + "\n")
    campaign.status(
        "policy_weight_complete",
        selected_checkpoint=retained,
        improvement_resolved=passed,
    )
    return result
