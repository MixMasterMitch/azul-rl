"""Sixteen-hour matched auxiliary-score study, including independent validation."""

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
from agent.net.model import AzulNet
from agent.scripts.capacity_campaign import paired_serving_difference
from agent.scripts.distillation_campaign import (
    confirmation_games,
    fork_arm as fork_continuation,
    freeze_record,
    serving_profile,
)
from agent.scripts.finetune_campaign import DIAGNOSTIC_ENVIRONMENT
from agent.scripts.league_campaign import SEARCH, freeze_file
from agent.scripts.lr_campaign import TrainingWindowExhausted, train_segment
from agent.scripts.weight_refine_campaign import freeze_league, retire_completed_arm
from agent.train.checkpointing import (
    load_checkpoint_payload,
    save_checkpoint_payload,
    warm_start_net,
)
from agent.train.learner import make_optimizer
from agent.train.loop import LoopConfig
from agent.train.reproducibility import capture_rng_state, seed_all

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


def augment_training_state(payload: dict, config: LoopConfig) -> dict:
    """Add a dormant head while retaining all existing optimizer moments/replay.

    Both arms get identical added parameters and new random streams. The original
    replay has no observed score labels; their availability remains explicitly false.
    """
    if (
        payload.get("aux_score")
        or payload["hidden"] != config.hidden
        or "distillation" in payload
    ):
        raise ValueError("Expected an ordinary matching-width champion resume")
    seed_all(config.seed)
    net = AzulNet(hidden=config.hidden, arch=config.arch, aux_score=True)
    warm_start_net(net, payload)
    old = payload["optimizer_state_dict"]
    fresh = make_optimizer(net, config.lr, config.weight_decay).state_dict()
    if len(old["param_groups"]) != 1 or len(fresh["param_groups"]) != 1:
        raise ValueError("Unexpected optimizer groups")
    count = len(old["param_groups"][0]["params"])
    if old["param_groups"][0]["params"] != fresh["param_groups"][0]["params"][
        :count
    ] or count != len(
        [p for name, p in net.named_parameters() if not name.startswith("score_head.")]
    ):
        raise ValueError("Optimizer parameter order changed")
    optimizer = {
        "state": old["state"],
        "param_groups": [
            {
                **old["param_groups"][0],
                "params": fresh["param_groups"][0]["params"],
                "lr": config.lr,
            }
        ],
    }
    payload.update(
        model_state_dict=net.state_dict(),
        aux_score=True,
        optimizer_state_dict=optimizer,
        config=asdict(config),
        progress={"training_wall_s": 0.0},
        rng_state=capture_rng_state(),
    )
    return payload


def prepare(campaign: Campaign, initializer: str) -> dict:
    source = json.loads(Path(initializer).read_text())
    marker = campaign.root / "prepared.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["inputs"] != source:
            raise ValueError("Campaign inputs changed")
        for path, digest in result["hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return result
    for path, digest in source["hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError(f"Input changed: {path}")
    frozen = campaign.root / "initializers"
    for name in ("baseline", "resume", "wide", "historical"):
        freeze_file(Path(source[name]), frozen / f"{name}.pt")
    freeze_league(Path(source["league"]), frozen / "league")
    payload = load_checkpoint_payload(frozen / "resume.pt")
    weights = load_checkpoint_payload(frozen / "baseline.pt")
    if (
        payload["hidden"] != 256
        or payload.get("trained_player_counts") != [2]
        or any(
            not torch.equal(v, payload["model_state_dict"][k])
            for k, v in weights["model_state_dict"].items()
        )
    ):
        raise ValueError("Champion weights and full state do not match")
    original = LoopConfig(**payload["config"])
    arms = {}
    for name, weight in [("control", 0.0), ("score", 0.1)]:
        directory = campaign.root / "experiments" / f"{name}_seed{campaign.seed}"
        arms[name] = asdict(
            replace(
                original,
                run_id=directory.name,
                runs_root=str(directory.parent),
                league_root=str(directory / "league"),
                init_from=str(frozen / "baseline.pt"),
                seed=campaign.seed,
                device=campaign.device,
                provenance=None,
                max_wall_minutes=360.0,
                aux_score_head=True,
                aux_score_weight=weight,
                aux_score_scale=50.0,
                bounded_checkpoint_storage=True,
                keep_recent_checkpoints=1,
                search_inference_cache_size=0,
                eval_games=0,
                max_iters=1_000_000,
            )
        )
    result = {
        "inputs": source,
        "arms": arms,
        "paths": {
            n: str(frozen / f"{n}.pt")
            for n in ("baseline", "resume", "wide", "historical")
        },
        "league": str(frozen / "league"),
        "hashes": {
            str(p): checkpoint_hash(p) for p in frozen.rglob("*") if p.is_file()
        },
    }
    write_report(marker, result)
    return result


def fork_pilot(prepared: dict, config: LoopConfig) -> dict:
    directory = Path(config.runs_root) / config.run_id
    marker = directory / "aux_score_fork.json"
    identity = {"config": asdict(config), "hashes": prepared["hashes"]}
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["identity"] != identity:
            raise ValueError("Pilot initialization changed")
        return result
    if directory.exists():
        raise ValueError("Refusing to overwrite an unmarked pilot")
    staging = directory.with_name(directory.name + ".preparing")
    if staging.exists():
        shutil.rmtree(staging)
    payload = augment_training_state(
        load_checkpoint_payload(prepared["paths"]["resume"]), config
    )
    freeze_league(Path(prepared["league"]), staging / "league")
    save_checkpoint_payload(staging / "checkpoints/latest_resume.pt", payload)
    result = {
        "identity": identity,
        "iteration": payload["iteration"],
        "replay_size": payload["buffer"]["size"],
        "optimizer_retained": True,
        "head_initialized_identically": True,
        "fresh_seed": config.seed,
        "legacy_scores": "Unknown; excluded from auxiliary loss",
    }
    write_report(staging / "aux_score_fork.json", result)
    staging.rename(directory)
    return result


def candidate_rank(candidate: dict) -> tuple[bool, float, float]:
    """Only a resolved >3pp auxiliary-opponent regression vetoes a screen.

    Within eligible candidates, head-to-head strength is primary. A noisy Astra
    point estimate alone cannot discard a promising checkpoint again.
    """
    eligible = all(
        d["paired_ci95"][1] >= -0.03 for d in candidate["differences"].values()
    )
    auxiliary = sum(
        d["match_score_difference"] for d in candidate["differences"].values()
    )
    return eligible, candidate["head"]["match_score"], auxiliary


def clear_regression(head: dict) -> bool:
    return head["match_score_ci95"][1] < 0.48


def retain(campaign: Campaign, arm: str, config: LoopConfig, candidate: dict) -> dict:
    marker = campaign.root / f"best_{arm}.json"
    old = json.loads(marker.read_text()) if marker.exists() else None
    if old and candidate_rank(candidate) <= candidate_rank(old["candidate"]):
        if checkpoint_hash(old["resume"]) != old["resume_sha256"]:
            raise ValueError("Retained full state changed")
        return old
    milestone = candidate["milestone"]
    directory = Path(config.runs_root) / config.run_id
    source = directory / "checkpoints/latest_resume.pt"
    payload = load_checkpoint_payload(source)
    weights = load_checkpoint_payload(candidate["checkpoint"])
    if payload["iteration"] != milestone["iteration"] or any(
        not torch.equal(v, payload["model_state_dict"][k])
        for k, v in weights["model_state_dict"].items()
    ):
        raise ValueError("Milestone and full state differ")
    del payload, weights
    destination = campaign.root / "retained" / arm / Path(candidate["checkpoint"]).stem
    campaign.status("archiving", arm=arm)
    freeze_file(source, destination / "resume.pt")
    freeze_league(directory / "league", destination / "league")
    result = {
        "candidate": candidate,
        "milestone": milestone,
        "resume": str(destination / "resume.pt"),
        "resume_sha256": checkpoint_hash(destination / "resume.pt"),
        "league": str(destination / "league"),
    }
    write_report(marker, result)
    if old:
        old_path = Path(old["resume"]).parent
        if old_path.parent != destination.parent or old_path == destination:
            raise ValueError("Unsafe retirement path")
        shutil.rmtree(old_path)
    return result


def retire_scratch(campaign: Campaign, config: LoopConfig, best: dict) -> None:
    """Retire only owned scratch after the complete best state/league is archived."""
    retire_completed_arm(campaign, config, best)
    directory = Path(config.runs_root) / config.run_id
    league = Path(config.league_root)
    if not league.exists():
        return
    if (
        league.resolve() != (directory / "league").resolve()
        or directory.parent != campaign.root / "experiments"
    ):
        raise ValueError("Refusing to retire a foreign league")
    archive = Path(best["league"])
    if (
        not archive.is_relative_to(campaign.root / "retained")
        or archive.resolve() == league.resolve()
    ):
        raise ValueError("Missing independent retained league")
    manifest = json.loads((archive / "league.json").read_text())
    if any(
        not (archive / e["path"]).is_file()
        for e in manifest["entries"]
        if e.get("active", True)
    ):
        raise ValueError("Incomplete retained league")
    shutil.rmtree(league)


def run_aux_score_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    ready = json.loads((campaign.root / "ready.json").read_text())
    if ready.get("passed") is not True or ready["code"] != campaign.code:
        raise ValueError("Source differs from validated auxiliary-score campaign")
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise ValueError("Native diagnostics must remain enabled")
    result_path = campaign.root / "aux_score_campaign.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        campaign.status(
            "aux_score_complete", selected_checkpoint=result["selected_checkpoint"]
        )
        return result
    prepared = prepare(campaign, initializer)
    paths = prepared["paths"]
    configs = {k: LoopConfig(**v) for k, v in prepared["arms"].items()}
    plan = freeze_record(
        campaign.root,
        "plan.json",
        {
            "budget": budget,
            "code": campaign.code,
            "arms": prepared["arms"],
            "pilot_minutes": [30, 60, 120],
            "continuation_minutes": [60, 120, 180, 240],
            "score_target": "Undiscounted final score difference from acting-player perspective; includes bonuses.",
            "score_loss": "0.1 * Huber(prediction, margin/50), averaged over full batch, unknown labels excluded.",
            "control": "Identical dormant head, original optimizer moments and replay; score-loss weight zero.",
            "selection": "Maximize head-to-head point score. Veto only resolved >3pp Astra/historical regression.",
            "early_stop": "Head-to-head 95% upper bound below 48%; otherwise inconclusive pilots remain eligible.",
            "development_seed": campaign.seed + 410_000_000,
            "confirmation_seed": campaign.seed + 420_000_000,
            "final_reserve_minutes": 180,
            "automatic_deployment": False,
            "retention": "All milestone weights and best full optimizer/replay/RNG/league state per arm.",
            "final_selection": "Provisional head point advantage, Astra point loss <=3pp and local CPU <=2s; resolved additionally head CI lower >50%.",
        },
    )
    searches = {
        "small": replace(SEARCH, num_simulations=1024),
        "wide": replace(SEARCH, num_simulations=384),
    }

    def match(
        label: str,
        checkpoint: str,
        opponent: str,
        *,
        wide: bool = False,
        games: int = 256,
        final: bool = False,
        cpu: bool = False,
    ) -> dict:
        search = searches["wide" if wide else "small"]
        opponent_search = SEARCH if opponent == "astra" else searches["small"]
        if cpu:
            search = replace(search, move_deadline_s=1.8)
            opponent_search = replace(opponent_search, move_deadline_s=1.8)
        cfg = replace(
            campaign.arena(
                games, search=search, split="confirmation" if final else "development"
            ),
            seed=plan["confirmation_seed" if final else "development_seed"]
            + (1_000_000 if cpu else 0),
            inference_device="cpu" if cpu else campaign.device,
            game_batch_size=2 if cpu else 128,
        )
        report = campaign._match(label, checkpoint, opponent, cfg, opponent_search)
        if report["summary"]["unfinished"]:
            raise RuntimeError(f"Unfinished evaluation in {label}")
        return report

    # Fresh seeds and CPU qualifications, not the noisy historical development panel.
    for label, path, width in [
        ("baseline", paths["baseline"], 256),
        ("saved_wide", paths["wide"], 512),
    ]:
        profile = serving_profile(
            campaign,
            label,
            path,
            width,
            maximum_simulations=1024 if width == 256 else 384,
        )
        expected = searches["wide" if width == 512 else "small"].num_simulations
        if profile["search"]["num_simulations"] != expected:
            raise RuntimeError(
                "Predeclared serving profile no longer meets the CPU budget"
            )
    baseline = {
        name: match(
            "baseline_" + name,
            paths["baseline"],
            opponent,
            games=1024 if name == "astra" else 256,
        )
        for name, opponent in [("astra", "astra"), ("historical", paths["historical"])]
    }
    candidates: dict[str, dict] = {}
    best: dict[str, dict] = {}

    def screen(
        label: str,
        checkpoint: str,
        *,
        wide: bool = False,
        milestone: dict | None = None,
        arm: str | None = None,
        games: int = 256,
    ) -> dict:
        head = match(
            label + "_head", checkpoint, paths["baseline"], wide=wide, games=games
        )
        opponents = {
            name: match(
                label + "_" + name,
                checkpoint,
                opponent,
                wide=wide,
                games=games if name == "astra" else 128,
            )
            for name, opponent in [
                ("astra", "astra"),
                ("historical", paths["historical"]),
            ]
        }
        candidate = {
            "label": label,
            "checkpoint": checkpoint,
            "hidden": 512 if wide else 256,
            "arm": arm,
            "milestone": milestone,
            "head": head["summary"],
            "opponents": {n: r["summary"] for n, r in opponents.items()},
            "differences": {
                n: paired_serving_difference(r, baseline[n])
                for n, r in opponents.items()
            },
            "seconds_per_game": (head["wall_s"] + opponents["astra"]["wall_s"]) / games
            + baseline["astra"]["wall_s"] / 1024,
        }
        candidates[label] = candidate
        if arm:
            best[arm] = retain(campaign, arm, configs[arm], candidate)
        write_report(
            campaign.root / "development.json",
            {"candidates": candidates, "best_arms": best},
        )
        return candidate

    wide = screen("saved_wide_180m", paths["wide"], wide=True, games=1024)
    write_report(campaign.root / "saved_wide_result.json", wide)
    for arm in ("control", "score"):
        config = configs[arm]
        done = campaign.root / f"{arm}_pilot_complete.json"
        if done.exists():
            completed = json.loads(done.read_text())
            candidates.update(completed["candidates"])
            best[arm] = json.loads((campaign.root / f"best_{arm}.json").read_text())
            retire_scratch(campaign, config, best[arm])
            continue
        campaign.status("forking", arm=arm)
        fork_pilot(prepared, config)
        for minutes in plan["pilot_minutes"]:
            label = f"{arm}_{minutes}m"
            milestone = train_segment(
                campaign, config, minutes, label, evaluation_reserve_minutes=210
            )
            if minutes != 60:
                candidate = screen(
                    label,
                    milestone["checkpoint"],
                    milestone=milestone,
                    arm=arm,
                    games=128 if minutes == 30 else 256,
                )
                if clear_regression(candidate["head"]):
                    write_report(
                        campaign.root / f"{arm}_early_stop.json",
                        {
                            "candidate": candidate,
                            "reason": "Head-to-head upper 95% bound below 48%",
                        },
                    )
                    break
        write_report(
            done,
            {"candidates": {n: c for n, c in candidates.items() if c["arm"] == arm}},
        )
        retire_scratch(campaign, config, best[arm])

    choice_path = campaign.root / "continuation_selection.json"
    if choice_path.exists():
        choice = json.loads(choice_path.read_text())
    else:
        viable = [
            a
            for a in ("control", "score")
            if not clear_regression(best[a]["candidate"]["head"])
            and candidate_rank(best[a]["candidate"])[0]
        ]
        chosen = (
            max(viable, key=lambda a: candidate_rank(best[a]["candidate"]))
            if viable
            else None
        )
        choice = {
            "arm": chosen,
            "source": best[chosen] if chosen else None,
            "reason": "Best eligible pilot point estimate; inconclusive gains remain eligible."
            if chosen
            else "Both pilots clearly regressed; preserve budget and evaluate existing candidates.",
        }
        write_report(choice_path, choice)
    if choice["arm"]:
        source = choice["source"]
        directory = campaign.root / "experiments" / "continuation"
        config = replace(
            configs[choice["arm"]],
            run_id=directory.name,
            runs_root=str(directory.parent),
            league_root=str(directory / "league"),
            seed=campaign.seed + 1009,
            init_from=source["milestone"]["checkpoint"],
            max_wall_minutes=240.0,
        )
        configs["continuation"] = config
        done = campaign.root / "continuation_complete.json"
        if done.exists():
            candidates.update(json.loads(done.read_text())["candidates"])
            if (campaign.root / "best_continuation.json").exists():
                best["continuation"] = json.loads(
                    (campaign.root / "best_continuation.json").read_text()
                )
            retire_scratch(campaign, config, best.get("continuation", source))
        else:
            campaign.status("forking", arm="continuation")
            fork_continuation(config, source)
            for minutes in plan["continuation_minutes"]:
                label = f"continuation_{minutes}m"
                stop = campaign.root / "continuation_stopped.json"
                marker = directory / "milestones" / f"{label}.json"
                if stop.exists() and not marker.exists():
                    break
                try:
                    milestone = train_segment(
                        campaign, config, minutes, label, evaluation_reserve_minutes=210
                    )
                except TrainingWindowExhausted:
                    write_report(stop, {"reason": "Final evaluation reserve"})
                    break
                candidate = screen(
                    label,
                    milestone["checkpoint"],
                    milestone=milestone,
                    arm="continuation",
                )
                if clear_regression(candidate["head"]):
                    write_report(
                        stop,
                        {
                            "reason": "Clear head-to-head regression",
                            "candidate": candidate,
                        },
                    )
                    break
            write_report(
                done,
                {
                    "candidates": {
                        n: c
                        for n, c in candidates.items()
                        if c["arm"] == "continuation"
                    }
                },
            )
            retire_scratch(campaign, config, best.get("continuation", source))

    challenger = max(candidates.values(), key=candidate_rank)
    is_wide = challenger["hidden"] == 512
    profile = serving_profile(
        campaign,
        "final_challenger",
        challenger["checkpoint"],
        challenger["hidden"],
        maximum_simulations=384 if is_wide else 1024,
    )
    if profile["search"]["num_simulations"] != (384 if is_wide else 1024):
        raise RuntimeError(
            "Final candidate cannot complete its predeclared CPU search budget"
        )
    selection_path = campaign.root / "final_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["challenger"] != challenger:
            raise ValueError("Final candidate changed")
    else:
        games = confirmation_games(
            campaign.deadline - time.monotonic(), challenger["seconds_per_game"]
        )
        if not games:
            raise TrainingWindowExhausted("No time for independent confirmation")
        selection = {
            "challenger": challenger,
            "games": min(2048, games),
            "profile": profile,
        }
        write_report(selection_path, selection)
    final = {}
    for label, path, opponent, wide_flag in [
        ("head_to_head", challenger["checkpoint"], paths["baseline"], is_wide),
        ("challenger_astra", challenger["checkpoint"], "astra", is_wide),
        ("baseline_astra", paths["baseline"], "astra", False),
    ]:
        final[label] = match(
            "confirmation_" + label,
            path,
            opponent,
            wide=wide_flag,
            games=selection["games"],
            final=True,
        )
        write_report(
            campaign.root / "confirmation_progress.json",
            {k: r["summary"] for k, r in final.items()},
        )
    final["cpu_audit"] = match(
        "confirmation_cpu",
        challenger["checkpoint"],
        paths["baseline"],
        wide=is_wide,
        games=8,
        final=True,
        cpu=True,
    )
    head = final["head_to_head"]["summary"]
    delta = paired_serving_difference(
        final["challenger_astra"], final["baseline_astra"]
    )
    timing = final["cpu_audit"]["timed_move_latency"]
    passed = all(
        timing.get(side, {}).get("count", 0) > 0 and timing[side]["over_2s"] == 0
        for side in ("candidate", "opponent")
    )
    provisional = (
        candidate_rank(challenger)[0]
        and head["match_score"] > 0.5
        and delta["match_score_difference"] >= -0.03
        and passed
    )
    result = {
        "selected_checkpoint": challenger["checkpoint"]
        if provisional
        else paths["baseline"],
        "provisional_improvement": provisional,
        "resolved_improvement": provisional and head["match_score_ci95"][0] > 0.5,
        "cpu_timing_passed": passed,
        "astra_difference": delta,
        "selection": selection,
        "best_arms": best,
        "elapsed_s": time.time() - budget["started_at"],
        "automatic_deployment": False,
        "final": {
            n: {
                "summary": r["summary"],
                "wall_s": r["wall_s"],
                "timed_move_latency": r.get("timed_move_latency", {}),
            }
            for n, r in final.items()
        },
    }
    for path, digest in prepared["hashes"].items():
        if checkpoint_hash(path) != digest:
            raise ValueError("Frozen input changed during campaign")
    write_report(result_path, result)
    rows = "\n".join(
        f"| {n} | {c['head']['match_score']:.1%} | {c['opponents']['astra']['match_score']:.1%} |"
        for n, c in candidates.items()
    )
    (campaign.root / "REPORT.md").write_text(
        "# Auxiliary score campaign\n\n"
        f"Selected: `{result['selected_checkpoint']}`\n\n"
        f"Final head-to-head: {head['match_score']:.2%}; 95% interval {head['match_score_ci95']}. "
        f"Provisional improvement: {provisional}; resolved: {result['resolved_improvement']}.\n\n"
        f"Astra change: {delta['match_score_difference']:+.2%}; CPU timing passed: {passed}.\n\n"
        "| Development candidate | Champion match score | Astra |\n|---|---:|---:|\n"
        + rows
        + "\n\n"
        "Binary win/tie rewards unchanged. One matched training seed. Local CPU timing only. No deployment.\n"
    )
    campaign.status(
        "aux_score_complete", selected_checkpoint=result["selected_checkpoint"]
    )
    return result
