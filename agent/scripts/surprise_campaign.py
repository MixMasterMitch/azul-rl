"""Twenty-hour reliability-gated saved-candidate closeout and surprise study."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

import torch

from agent.eval.arena import checkpoint_hash, write_report
from agent.scripts.aux_score_campaign import (
    candidate_rank,
    clear_regression,
    retain,
    retire_scratch,
)
from agent.scripts.capacity_campaign import paired_serving_difference
from agent.scripts.distillation_campaign import (
    confirmation_games,
    fork_arm,
    freeze_record,
    serving_profile,
)
from agent.scripts.finetune_campaign import DIAGNOSTIC_ENVIRONMENT
from agent.scripts.league_campaign import SEARCH, freeze_file
from agent.scripts.lr_campaign import TrainingWindowExhausted, train_segment
from agent.scripts.weight_refine_campaign import freeze_league
from agent.train.checkpointing import load_checkpoint_payload
from agent.train.loop import LoopConfig

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign


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
    names = (
        "baseline",
        "resume",
        "wide",
        "historical",
        "score",
        "score_resume",
        "old_control",
    )
    paths = {}
    for name in names:
        path = frozen / f"{name}.pt"
        freeze_file(Path(source[name]), path)
        paths[name] = str(path)
    states = {}
    for name, resume, league_key in [
        ("baseline", "resume", "league"),
        ("score", "score_resume", "score_league"),
    ]:
        league = frozen / league_key
        freeze_league(Path(source[league_key]), league)
        payload = load_checkpoint_payload(paths[resume])
        weights = load_checkpoint_payload(paths[name])
        if (
            payload["hidden"] != 256
            or payload.get("trained_player_counts") != [2]
            or any(
                not torch.equal(v, payload["model_state_dict"][k])
                for k, v in weights["model_state_dict"].items()
            )
            or any(
                k not in payload
                for k in ("optimizer_state_dict", "buffer", "rng_state")
            )
        ):
            raise ValueError(f"{name} weights and full state do not match")
        states[name] = {
            "resume": paths[resume],
            "resume_sha256": checkpoint_hash(paths[resume]),
            "league": str(league),
            "config": payload["config"],
            "checkpoint": paths[name],
        }
        del payload, weights
    result = {
        "inputs": source,
        "paths": paths,
        "states": states,
        "hashes": {
            str(p): checkpoint_hash(p) for p in frozen.rglob("*") if p.is_file()
        },
    }
    write_report(marker, result)
    return result


def supports_score(candidate: dict, versus_control: dict) -> bool:
    """Require two positive point estimates; do not mistake wide-model success for score-head evidence."""
    return (
        candidate_rank(candidate)[0]
        and candidate["head"]["match_score"] > 0.5
        and versus_control["summary"]["match_score"] > 0.5
        and candidate["differences"]["astra"]["match_score_difference"] >= -0.03
    )


def pilot_configs(campaign: Campaign, source: dict) -> dict[str, LoopConfig]:
    original = LoopConfig(**source["config"])
    result = {}
    for arm, fraction in [("control", 0.0), ("surprise", 0.5)]:
        directory = campaign.root / "experiments" / f"{arm}_seed{campaign.seed}"
        result[arm] = replace(
            original,
            run_id=directory.name,
            runs_root=str(directory.parent),
            league_root=str(directory / "league"),
            init_from=source["checkpoint"],
            seed=campaign.seed,
            device=campaign.device,
            provenance=None,
            max_wall_minutes=120.0,
            max_iters=1_000_000,
            policy_surprise_record=True,
            policy_surprise_fraction=fraction,
            policy_surprise_min_sims=256,
            policy_surprise_max_weight=4.0,
            bounded_checkpoint_storage=True,
            keep_recent_checkpoints=1,
            search_inference_cache_size=0,
            eval_games=0,
            checkpoint_every=10,
        )
    return result


def run_surprise_campaign(campaign: Campaign, initializer: str) -> dict:
    budget = json.loads((campaign.root / "preflight_budget.json").read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    ready = json.loads((campaign.root / "ready.json").read_text())
    if ready.get("passed") is not True or ready["code"] != campaign.code:
        raise ValueError("Source differs from validated surprise campaign")
    if not ready.get("reliability_passed"):
        raise ValueError("Native reliability validation missing")
    if any(checkpoint_hash(p) != h for p, h in ready["native_hashes"].items()):
        raise ValueError("Validated native library changed")
    if any(os.environ.get(k) != v for k, v in DIAGNOSTIC_ENVIRONMENT.items()):
        raise ValueError("Native diagnostics must remain enabled")
    result_path = campaign.root / "surprise_campaign.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        campaign.status(
            "surprise_complete", selected_checkpoint=result["selected_checkpoint"]
        )
        return result
    prepared = prepare(campaign, initializer)
    paths = prepared["paths"]
    plan = freeze_record(
        campaign.root,
        "plan.json",
        {
            "budget": budget,
            "code": campaign.code,
            "pilot_minutes": [30, 60, 120],
            "continuation_minutes": [60, 120, 180, 240, 300, 360],
            "sampling": "Full targets: min(4, .5 + .5*KL(target||raw prior)/mean_full_KL); fast and unknown weight 1. Both policy and value see biased samples. No importance correction.",
            "control": "Identical full-state initializer, recording and settings; uniform replay sampling.",
            "score_condition": "Saved score beats champion and matched historical control by point estimate; Astra loss <=3pp; no resolved historical regression.",
            "selection": "Head-to-head point score, resolved >3pp auxiliary regression veto; stop clear regressions with head CI upper below 48%.",
            "closeout_seed": campaign.seed + 510_000_000,
            "development_seed": campaign.seed + 520_000_000,
            "confirmation_seed": campaign.seed + 530_000_000,
            "closeout_games": 1024,
            "score_control_games": 512,
            "final_reserve_minutes": 180,
            "continuation": "Three hours planned; use setup savings for up to six, preserving final reserve plus next screen.",
            "automatic_deployment": False,
            "retention": "All milestone weights and best complete state per arm.",
            "final_rule": "Provisional if head point >50%, Astra delta >=-3pp, eligible, CPU <=2s; resolved also requires head CI lower >50%.",
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
        phase: str = "development",
        cpu: bool = False,
    ) -> dict:
        search = searches["wide" if wide else "small"]
        opponent_search = SEARCH if opponent == "astra" else searches["small"]
        if cpu:
            search = replace(search, move_deadline_s=1.8)
            opponent_search = replace(opponent_search, move_deadline_s=1.8)
        cfg = replace(
            campaign.arena(
                games,
                search=search,
                split="confirmation" if phase == "confirmation" else "development",
            ),
            seed=plan[phase + "_seed"] + (1_000_000 if cpu else 0),
            inference_device="cpu" if cpu else campaign.device,
            game_batch_size=2 if cpu else 128,
        )
        result = campaign._match(label, checkpoint, opponent, cfg, opponent_search)
        if result["summary"]["unfinished"]:
            raise RuntimeError(f"Unfinished evaluation in {label}")
        return result

    for label, path, width in [
        ("baseline", paths["baseline"], 256),
        ("saved_wide", paths["wide"], 512),
        ("saved_score", paths["score"], 256),
    ]:
        profile = serving_profile(
            campaign,
            label,
            path,
            width,
            maximum_simulations=384 if width == 512 else 1024,
        )
        if profile["search"]["num_simulations"] != (384 if width == 512 else 1024):
            raise RuntimeError("Predeclared CPU serving budget no longer qualifies")
    baselines: dict[str, dict] = {}
    candidates, best = {}, {}
    configs: dict[str, LoopConfig] = {}

    def baseline(phase: str) -> dict:
        if phase not in baselines:
            baselines[phase] = {
                name: match(
                    phase + "_baseline_" + name,
                    paths["baseline"],
                    opponent,
                    games=1024 if name == "astra" else 256,
                    phase=phase,
                )
                for name, opponent in [
                    ("astra", "astra"),
                    ("historical", paths["historical"]),
                ]
            }
        return baselines[phase]

    def screen(
        label: str,
        checkpoint: str,
        *,
        wide: bool = False,
        phase: str = "development",
        games: int = 256,
        milestone: dict | None = None,
        arm: str | None = None,
    ) -> dict:
        reference = baseline(phase)
        head = match(
            label + "_head",
            checkpoint,
            paths["baseline"],
            wide=wide,
            games=games,
            phase=phase,
        )
        opponents = {
            n: match(
                label + "_" + n,
                checkpoint,
                opponent,
                wide=wide,
                games=games if n == "astra" else 128,
                phase=phase,
            )
            for n, opponent in [("astra", "astra"), ("historical", paths["historical"])]
        }
        result = {
            "label": label,
            "checkpoint": checkpoint,
            "hidden": 512 if wide else 256,
            "arm": arm,
            "milestone": milestone,
            "phase": phase,
            "head": head["summary"],
            "opponents": {n: r["summary"] for n, r in opponents.items()},
            "differences": {
                n: paired_serving_difference(r, reference[n])
                for n, r in opponents.items()
            },
            "seconds_per_game": (head["wall_s"] + opponents["astra"]["wall_s"]) / games
            + reference["astra"]["wall_s"] / 1024,
        }
        candidates[label] = result
        if arm:
            best[arm] = retain(campaign, arm, configs[arm], result)
        write_report(
            campaign.root / "development.json",
            {"candidates": candidates, "best_arms": best},
        )
        return result

    wide = screen(
        "saved_wide",
        paths["wide"],
        wide=True,
        phase="closeout",
        games=plan["closeout_games"],
    )
    score = screen(
        "saved_score", paths["score"], phase="closeout", games=plan["closeout_games"]
    )
    score_control = match(
        "saved_score_vs_old_control",
        paths["score"],
        paths["old_control"],
        games=plan["score_control_games"],
        phase="closeout",
    )
    source_name = "score" if supports_score(score, score_control) else "baseline"
    source = prepared["states"][source_name]
    selection = freeze_record(
        campaign.root,
        "training_selection.json",
        {
            "source_name": source_name,
            "source": source,
            "saved_wide": wide,
            "saved_score": score,
            "score_vs_control": score_control["summary"],
        },
    )
    configs = pilot_configs(campaign, source)
    freeze_record(
        campaign.root, "pilot_configs.json", {a: asdict(c) for a, c in configs.items()}
    )

    # Matching the complete pilot window matters more than launching a truncated arm.
    pilot_ready = campaign.root / "pilot_budget.json"
    if not pilot_ready.exists():
        available = campaign.deadline - time.monotonic()
        write_report(
            pilot_ready,
            {
                "run": available >= 9 * 3600,
                "available_s": available,
                "required_s": 9 * 3600,
                "reason": "Four training hours + screens/development baseline + final reserve",
            },
        )
    if json.loads(pilot_ready.read_text())["run"]:
        baseline("development")
        for arm in ("control", "surprise"):
            config = configs[arm]
            done = campaign.root / f"{arm}_pilot_complete.json"
            if done.exists():
                candidates.update(json.loads(done.read_text())["candidates"])
                best[arm] = json.loads((campaign.root / f"best_{arm}.json").read_text())
                retire_scratch(campaign, config, best[arm])
                continue
            campaign.status("forking", arm=arm)
            fork_arm(config, source)
            for minutes in plan["pilot_minutes"]:
                label = f"{arm}_{minutes}m"
                milestone = train_segment(
                    campaign, config, minutes, label, evaluation_reserve_minutes=210
                )
                if minutes != 60:
                    c = screen(
                        label,
                        milestone["checkpoint"],
                        games=128 if minutes == 30 else 256,
                        milestone=milestone,
                        arm=arm,
                    )
                    if clear_regression(c["head"]):
                        write_report(
                            campaign.root / f"{arm}_early_stop.json", {"candidate": c}
                        )
                        break
            write_report(
                done,
                {
                    "candidates": {
                        n: c for n, c in candidates.items() if c["arm"] == arm
                    }
                },
            )
            retire_scratch(campaign, config, best[arm])

        marker = campaign.root / "continuation_selection.json"
        if marker.exists():
            choice = json.loads(marker.read_text())
        else:
            viable = [
                a
                for a in ("control", "surprise")
                if candidate_rank(best[a]["candidate"])[0]
                and not clear_regression(best[a]["candidate"]["head"])
            ]
            chosen = (
                max(viable, key=lambda a: candidate_rank(best[a]["candidate"]))
                if viable
                else None
            )
            choice = {"arm": chosen, "source": best[chosen] if chosen else None}
            write_report(marker, choice)
        if choice["arm"]:
            retained = choice["source"]
            directory = campaign.root / "experiments" / "continuation"
            config = replace(
                configs[choice["arm"]],
                run_id=directory.name,
                runs_root=str(directory.parent),
                league_root=str(directory / "league"),
                seed=campaign.seed + 1009,
                init_from=retained["milestone"]["checkpoint"],
                max_wall_minutes=360.0,
            )
            configs["continuation"] = config
            done = campaign.root / "continuation_complete.json"
            if done.exists():
                candidates.update(json.loads(done.read_text())["candidates"])
                if (campaign.root / "best_continuation.json").exists():
                    best["continuation"] = json.loads(
                        (campaign.root / "best_continuation.json").read_text()
                    )
                retire_scratch(campaign, config, best.get("continuation", retained))
            else:
                campaign.status("forking", arm="continuation")
                fork_arm(config, retained)
                for minutes in plan["continuation_minutes"]:
                    label = f"continuation_{minutes}m"
                    stop = campaign.root / "continuation_stopped.json"
                    if (
                        stop.exists()
                        and not (directory / "milestones" / f"{label}.json").exists()
                    ):
                        break
                    try:
                        milestone = train_segment(
                            campaign,
                            config,
                            minutes,
                            label,
                            evaluation_reserve_minutes=220,
                        )
                    except TrainingWindowExhausted:
                        write_report(stop, {"reason": "Final evaluation reserve"})
                        break
                    c = screen(
                        label,
                        milestone["checkpoint"],
                        milestone=milestone,
                        arm="continuation",
                    )
                    if clear_regression(c["head"]):
                        write_report(
                            stop,
                            {"reason": "Clear head-to-head regression", "candidate": c},
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
                retire_scratch(campaign, config, best.get("continuation", retained))

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
        raise RuntimeError("Final candidate fails predeclared CPU search budget")
    marker = campaign.root / "final_selection.json"
    if marker.exists():
        final_selection = json.loads(marker.read_text())
        if final_selection["challenger"] != challenger:
            raise ValueError("Final candidate changed")
    else:
        games = confirmation_games(
            campaign.deadline - time.monotonic(), challenger["seconds_per_game"]
        )
        if not games:
            raise TrainingWindowExhausted("Insufficient independent confirmation time")
        final_selection = {"challenger": challenger, "games": games, "profile": profile}
        write_report(marker, final_selection)
    final = {}
    for label, checkpoint, opponent, wide_flag in [
        ("head_to_head", challenger["checkpoint"], paths["baseline"], is_wide),
        ("challenger_astra", challenger["checkpoint"], "astra", is_wide),
        ("baseline_astra", paths["baseline"], "astra", False),
    ]:
        final[label] = match(
            "confirmation_" + label,
            checkpoint,
            opponent,
            wide=wide_flag,
            games=final_selection["games"],
            phase="confirmation",
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
        phase="confirmation",
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
        "selection": final_selection,
        "training_selection": selection,
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
            raise ValueError("Frozen campaign input changed")
    write_report(result_path, result)
    rows = "\n".join(
        f"| {n} | {c['head']['match_score']:.1%} | {c['opponents']['astra']['match_score']:.1%} |"
        for n, c in candidates.items()
    )
    (campaign.root / "REPORT.md").write_text(
        "# Policy surprise campaign\n\n"
        f"Selected: `{result['selected_checkpoint']}`\n\n"
        f"Final head-to-head: {head['match_score']:.2%}; 95% interval {head['match_score_ci95']}. "
        f"Provisional: {provisional}; resolved: {result['resolved_improvement']}.\n\n"
        f"Astra change: {delta['match_score_difference']:+.2%}; CPU timing passed: {passed}.\n\n"
        "| Candidate | Champion match score | Astra |\n|---|---:|---:|\n"
        + rows
        + "\n\n"
        "One matched seed. Binary rewards unchanged. Local two-second CPU audit. No automatic deployment.\n"
    )
    campaign.status(
        "surprise_complete", selected_checkpoint=result["selected_checkpoint"]
    )
    return result
