"""Ten-hour matched continuation with measured, diverse league opponents."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

from agent.env.outcomes import REWARD_SEMANTICS_VERSION
from agent.eval.arena import checkpoint_hash, paired_interval, write_report
from agent.eval.builtin_opponents import builtin_identity
from agent.scripts.enhancement_campaign import _verify
from agent.scripts.lr_campaign import (
    TrainingWindowExhausted,
    retire_replay,
    train_segment,
)
from agent.scripts.noise_study import paired_difference
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint, save_checkpoint
from agent.train.league import League
from agent.train.loop import LoopConfig

if TYPE_CHECKING:
    from agent.scripts.competitive import Campaign

PANEL = ("start", "previous", "astra")
SEARCH = SearchConfig(
    backend="gumbel_tree",
    tree_core="rust",
    num_simulations=64,
    cpu_workers=1,
    temperature=0.25,
    q_scale=28.0,
    root_noise_scale=1.0,
)


def freeze_file(source: Path, target: Path) -> None:
    """Hard-link immutable archives; future checkpoint saves replace their inode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        temporary.replace(target)
    if checkpoint_hash(source) != checkpoint_hash(target):
        raise ValueError(f"Frozen input differs: {target}")


def fork_training_state(source: Path, league_root: Path, directory: Path) -> dict:
    """Clone complete state while keeping league manifests and later writes isolated."""
    manifest = json.loads((league_root / "league.json").read_text())
    identity = {"resume_sha256": checkpoint_hash(source), "league": manifest}
    marker = directory / "fork.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Fork inputs changed")
        return identity
    if directory.exists():
        raise ValueError("Unmarked training directory cannot be replaced")
    staging = directory.with_name(directory.name + ".preparing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    freeze_file(source, staging / "checkpoints/latest_resume.pt")
    source_league = League(league_root)
    for entry in manifest["entries"]:
        path = source_league._resolve_path(entry["path"])
        if entry.get("active", True):
            freeze_file(path, staging / "league" / Path(entry["path"]).name)
    # JSON is a separate file, never a link into the source league.
    copied = json.loads(json.dumps(manifest))
    for entry in copied["entries"]:
        entry["path"] = Path(entry["path"]).name
    write_report(staging / "league/league.json", copied)
    write_report(staging / "fork.json", identity)
    staging.rename(directory)
    return identity


def prepare(campaign: Campaign, initializer: str) -> dict:
    source = Path(initializer).resolve()
    marker = campaign.root / "prepared.json"
    if marker.exists():
        prepared = json.loads(marker.read_text())
        if (
            str(source) != prepared["source_resume"]
            or checkpoint_hash(source) != prepared["source_resume_sha256"]
        ):
            raise ValueError("Source resume checkpoint changed")
        if any(
            checkpoint_hash(p) != digest
            for p, digest in prepared["input_hashes"].items()
        ):
            raise ValueError("Frozen campaign input changed")
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
            "League campaign requires a complete current two-player width-256 resume checkpoint"
        )
    source_config = LoopConfig(**payload["config"])
    base_wall_s = payload["progress"]["training_wall_s"]
    initial_iteration = payload["iteration"]
    frozen = campaign.root / "initializers"
    frozen.mkdir(exist_ok=True)
    freeze_file(source, frozen / "source_resume.pt")
    if not (frozen / "start.pt").exists():
        save_checkpoint(
            frozen / "start.pt",
            net,
            iteration=initial_iteration,
            config=payload["config"],
            progress=payload["progress"],
        )
    previous_source = Path(source_config.init_from).resolve()
    freeze_file(previous_source, frozen / "previous.pt")
    previous_net, previous_payload = load_net_from_checkpoint(previous_source)
    historical_source = Path(previous_payload["config"]["init_from"]).resolve()
    freeze_file(historical_source, frozen / "historical.pt")
    alternative = (
        source.parent.parent.parent
        / f"reanalysis4x_seed{source_config.seed}/milestones/pilot.pt"
    )
    freeze_file(alternative, frozen / "alternative.pt")
    del net, payload, previous_net, previous_payload
    source_league = Path(source_config.league_root)
    snapshot_league = frozen / "league"
    manifest = json.loads((source_league / "league.json").read_text())
    for entry in manifest["entries"]:
        if entry.get("active", True):
            freeze_file(
                source_league / entry["path"],
                snapshot_league / Path(entry["path"]).name,
            )
    write_report(snapshot_league / "league.json", manifest)
    configs = {}
    for arm in ("current", "rated"):
        name = f"{arm}_seed{source_config.seed}"
        directory = campaign.root / "experiments" / name
        fork_training_state(frozen / "source_resume.pt", snapshot_league, directory)
        configs[arm] = asdict(
            replace(
                source_config,
                run_id=name,
                runs_root=str(campaign.root / "experiments"),
                league_root=str(directory / "league"),
                init_from=str(frozen / "start.pt"),
                provenance=None,
                device=campaign.device,
                max_wall_minutes=base_wall_s / 60 + 330,
            )
        )
    prepared = {
        "source_resume": str(source),
        "source_resume_sha256": checkpoint_hash(source),
        "base_training_wall_s": base_wall_s,
        "initial_iteration": initial_iteration,
        "arms": configs,
        "frozen": {
            name: str(frozen / f"{name}.pt")
            for name in ("start", "previous", "historical", "alternative")
        },
        "input_hashes": {str(p): checkpoint_hash(p) for p in frozen.glob("*.pt")},
    }
    write_report(marker, prepared)
    return prepared


def import_opponent(league: League, source: Path, tag: str) -> dict:
    existing = next((e for e in league.list_entries() if e.get("tag") == tag), None)
    if existing:
        if checkpoint_hash(league._resolve_path(existing["path"])) != checkpoint_hash(
            source
        ):
            raise ValueError("Imported opponent changed")
        return existing
    idx = max((e["idx"] for e in league.list_entries()), default=-1) + 1
    target = league.root / f"ckpt_{idx:05d}_{tag}.pt"
    freeze_file(source, target)
    entry = {
        "idx": idx,
        "path": target.name,
        "tag": tag,
        "iteration": 0,
        "rating": 1500.0,
        "games": 0,
        "hidden": 256,
        "arch": "source_attn",
        "active": True,
        "pinned": True,
    }
    league.manifest["entries"].append(entry)
    league._save_manifest()
    return entry


def panel_score(reports: dict[str, dict]) -> float:
    return sum(reports[name]["summary"]["match_score"] for name in PANEL) / len(PANEL)


def panel_difference(candidate: dict[str, dict], control: dict[str, dict]) -> dict:
    deltas = {name: paired_difference(candidate[name], control[name]) for name in PANEL}
    lengths = {len(candidate[name]["summary"]["pair_scores"]) for name in PANEL}
    seeds = {candidate[name]["config"]["seed"] for name in PANEL}
    if len(lengths) != 1 or len(seeds) != 1:
        raise ValueError(
            "Panel comparisons require matching seed blocks across opponents"
        )
    differences = [
        sum(
            candidate[name]["summary"]["pair_scores"][i]
            - control[name]["summary"]["pair_scores"][i]
            for name in PANEL
        )
        / len(PANEL)
        for i in range(next(iter(lengths)))
    ]
    return {
        "match_score_difference": sum(differences) / len(differences),
        "paired_ci95": paired_interval(differences),
        "opponents": deltas,
        "seed_pairs": len(differences),
        "equal_opponent_weights": True,
    }


def choose_arm(reports: dict[str, dict]) -> dict:
    difference = panel_difference(reports["rated"], reports["current"])
    use_rated = (
        difference["paired_ci95"][0] > 0
        and difference["opponents"]["astra"]["match_score_difference"] >= -0.03
    )
    return {
        "selected_arm": "rated" if use_rated else "current",
        "rated_minus_current": difference,
        "reason": (
            "Measured league passed the combined-score and Astra gates."
            if use_rated
            else "Measured league did not pass both gates; continue current settings."
        ),
    }


def choose_checkpoint(candidates: dict[str, dict], baseline_astra: float) -> str:
    eligible = [
        name
        for name, screens in candidates.items()
        if screens["astra"]["summary"]["match_score"] >= baseline_astra - 0.03
    ]
    return max(
        eligible,
        key=lambda name: (
            panel_score(candidates[name]),
            candidates[name]["astra"]["summary"]["match_score"],
        ),
    )


def rate_league(
    campaign: Campaign, config: LoopConfig, prepared: dict, stage: str
) -> dict:
    """Cache each match and atomically apply it once to the league's rating history."""
    directory = campaign.root / "league_ratings"
    marker = directory / f"{stage}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    league = League(config.league_root)
    league.manifest["measured_strong_min_games"] = 128
    imported = {
        name: import_opponent(league, Path(prepared["frozen"][name]), f"pool_{name}")
        for name in ("start", "historical", "alternative")
    }
    plan_path = directory / f"{stage}_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        entries = [
            e
            for e in league.list_entries()
            if e.get("active", True) and league._entry_available(e)
        ]
        previous = next(e for e in entries if e.get("tag") == "frozen_baseline")
        anchors = [previous, *imported.values()]
        if stage == "initial":
            inherited = [e for e in entries if not e.get("tag", "").startswith("pool_")]
            pool = {
                e["idx"]: e
                for e in [*anchors, inherited[len(inherited) // 2], inherited[-1]]
            }
            pairs = [(a, b) for a, b in combinations(pool, 2)]
            pairs.append((imported["start"]["idx"], "random"))
        else:
            recent = [
                e
                for e in entries
                if e.get("iteration", 0) > prepared["initial_iteration"]
            ][-2:]
            pairs = [
                (e["idx"], a["idx"])
                for e in recent
                for a in anchors
                if e["idx"] != a["idx"]
            ]
        involved = {idx for pair in pairs for idx in pair if isinstance(idx, int)}
        paths = {
            str(idx): str(league._resolve_path(league.entry_by_idx(idx)["path"]))
            for idx in involved
        }
        plan = {
            "pairs": pairs,
            "paths": paths,
            "started_at": time.time(),
            "hashes": {key: checkpoint_hash(path) for key, path in paths.items()},
        }
        write_report(plan_path, plan)
    for offset, (a, b) in enumerate(plan["pairs"]):
        label = f"league_{stage}_{a}_{b}"
        if label in league.manifest.get("campaign_rating_reports", []):
            continue
        candidate, opponent = (
            plan["paths"][str(a)],
            plan["paths"][str(b)] if isinstance(b, int) else b,
        )
        for idx in (a, b):
            if (
                isinstance(idx, int)
                and checkpoint_hash(plan["paths"][str(idx)]) != plan["hashes"][str(idx)]
            ):
                raise ValueError("Rating opponent changed")
        cfg = replace(
            campaign.arena(128, search=SEARCH),
            seed=campaign.seed
            + 120_000_000
            + offset * 1000
            + (0 if stage == "initial" else int(stage.split("_")[-1]) * 100_000),
        )
        report = campaign._match(label, candidate, opponent, cfg, SEARCH)
        if report["summary"]["unfinished"]:
            raise RuntimeError("Unfinished league rating games")
        s = report["summary"]
        league.record_result(
            f"ckpt:{a}",
            f"ckpt:{b}" if isinstance(b, int) else b,
            s["wins"],
            s["losses"],
            s["shared_victories"],
            num_players=2,
        )
        league.manifest.setdefault("campaign_rating_reports", []).append(label)
        league._save_manifest()
    league.recompute_ratings()
    result = {
        "stage": stage,
        "elapsed_s": time.time() - plan["started_at"],
        "games": len(plan["pairs"]) * 128,
        "ratings": [
            {k: e.get(k) for k in ("idx", "tag", "rating_2p", "games")}
            for e in league.list_entries()
            if e.get("games", 0) >= 128
        ],
    }
    write_report(marker, result)
    return result


def _report(root: Path, result: dict, budget: dict) -> None:
    end = datetime.fromtimestamp(
        budget["started_at"] + result["elapsed_s"], timezone.utc
    )
    rows = [
        "# Ten-hour measured-league campaign",
        "",
        f"Completed in {result['elapsed_s'] / 3600:.2f} hours including setup; finished {end.isoformat()}.",
        "",
        f"Selected arm: **{result['selected_arm']}**. {result['arm_decision']['reason']}",
        "",
        f"Selected weights: [{Path(result['selected_checkpoint']).name}]({result['selected_checkpoint']}).",
        "",
        "| Held-out opponent | Candidate score | Starting score |",
        "|---|---:|---:|",
    ]
    for name in PANEL:
        rows.append(
            f"| {name} | {result['screens'][name]['match_score']:.1%} | "
            f"{result['start_screens'][name]['match_score']:.1%} |"
        )
    d = result["panel_difference"]
    rows += [
        "",
        f"Equal-weight panel gain: {d['match_score_difference'] * 100:+.1f}pp "
        f"(paired 95% interval {d['paired_ci95'][0] * 100:+.1f} to {d['paired_ci95'][1] * 100:+.1f}pp).",
        f"Combined improvement gate passed: **{result['improvement_resolved']}**. "
        "The gate requires a resolved panel gain and Astra point-score loss no worse than 3pp.",
        "",
        "Match score counts wins as 1, shared victories as 0.5, and losses as 0. "
        "Primary matches use identical 64-simulation Rust search. Both arms inherit the same "
        "weights, optimizer, replay, and RNG; only the league treatment differs. This is one training trajectory.",
        "",
        f"Latest selected-arm full resume state: `{result['resume_checkpoint']}`. "
        "Its training milestone may be later than the best evaluated weights. "
        "Source resume state and historical checkpoints were preserved. No automatic deployment occurred.",
        "",
        "[Results](league_campaign.json) · [Verification](verification.json) · [Plan](league_campaign_plan.json)",
    ]
    (root / "REPORT.md").write_text("\n".join(rows) + "\n")


def run_league_campaign(campaign: Campaign, initializer: str) -> dict:
    prepared = prepare(campaign, initializer)
    configs = {name: LoopConfig(**cfg) for name, cfg in prepared["arms"].items()}
    budget_path = campaign.root / "league_campaign_budget.json"
    if not budget_path.exists():
        now = time.time()
        budget = {
            "started_at": now,
            "deadline": now + min(10 * 3600, campaign.deadline - time.monotonic()),
        }
        for name in ("preflight_budget.json", "supervisor.json"):
            path = campaign.root / name
            if path.exists():
                other = json.loads(path.read_text())
                budget = {k: min(budget[k], other[k]) for k in budget}
        write_report(budget_path, budget)
    budget = json.loads(budget_path.read_text())
    campaign.deadline = min(
        campaign.deadline, time.monotonic() + budget["deadline"] - time.time()
    )
    plan = {
        "campaign_hours": 10,
        "study": "measured_league",
        **prepared,
        "initializer": prepared["frozen"]["start"],
        "initializer_sha256": checkpoint_hash(prepared["frozen"]["start"]),
        "pilot_minutes_per_arm": 120,
        "continuation_arm_budget_minutes": [225, 330],
        "rating_refresh_arm_budget_minutes": [60, 120, 225],
        "pilot_games_per_opponent": 512,
        "confirmation_games_per_opponent": 1024,
        "evaluation_panel": list(PANEL),
        "evaluation_search": asdict(SEARCH),
        "development_seed": campaign.seed + 100_000_000,
        "confirmation_seed": campaign.seed + 110_000_000,
        "rating_seed_base": campaign.seed + 120_000_000,
        "rating_games_per_pair": 128,
        "rating_cost_charged_to_rated_arm": True,
        "final_evaluation_reserve_minutes": 90,
        "last_screen_reserve_minutes": 15,
        "selection": "Equal-weight paired panel CI above zero and Astra delta >= -0.03; otherwise current.",
        "checkpoint_selection": "Highest equal-weight panel score with Astra within 3pp of start; start eligible.",
        "astra_identity": builtin_identity("astra"),
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "bot_workers": campaign.bot_workers,
        "automatic_promotion": False,
        "provenance": campaign.code,
    }
    plan_path = campaign.root / "league_campaign_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("League campaign protocol changed")
    write_report(plan_path, plan)
    final_path = campaign.root / "league_campaign.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        write_report(
            campaign.root / "verification.json", _verify(campaign.root, result)
        )
        _report(campaign.root, result, budget)
        campaign.status(
            "league_campaign_complete",
            selected_checkpoint=result["selected_checkpoint"],
        )
        return result

    def panel(label: str, checkpoint: str, confirmation: bool = False) -> dict:
        reports = {}
        for name in PANEL:
            opponent = "astra" if name == "astra" else prepared["frozen"][name]
            cfg = replace(
                campaign.arena(
                    1024 if confirmation else 512,
                    search=SEARCH,
                    split="confirmation" if confirmation else "development",
                ),
                seed=plan["confirmation_seed" if confirmation else "development_seed"],
            )
            reports[name] = campaign._match(
                f"{label}_{name}", checkpoint, opponent, cfg, SEARCH
            )
            if reports[name]["summary"]["unfinished"]:
                raise RuntimeError("Unfinished panel games")
        return reports

    def train(arm: str, minutes: float, label: str) -> dict:
        target_path = campaign.root / "arm_budgets" / f"{arm}_{label}.json"
        if target_path.exists():
            target = json.loads(target_path.read_text())
        else:
            rating_s = (
                sum(
                    json.loads(p.read_text())["elapsed_s"]
                    for p in (campaign.root / "league_ratings").glob("*.json")
                    if not p.name.endswith("_plan.json")
                )
                if arm == "rated"
                else 0.0
            )
            target = {
                "arm_budget_minutes": minutes,
                "rating_s": rating_s,
                "cumulative_training_minutes": prepared["base_training_wall_s"] / 60
                + minutes
                - rating_s / 60,
            }
            write_report(target_path, target)
        milestone = train_segment(
            campaign,
            configs[arm],
            target["cumulative_training_minutes"],
            label,
            evaluation_reserve_minutes=105,
        )
        return {
            **milestone,
            **target,
            "additional_training_minutes": (
                milestone["training_wall_s"] - prepared["base_training_wall_s"]
            )
            / 60,
        }

    # Initial rankings and periodic refreshes are charged to the experimental arm.
    rate_league(campaign, configs["rated"], prepared, "initial")
    pilots = {}
    for arm in configs:
        train(arm, 60, "pilot_half")
        if arm == "rated":
            rate_league(campaign, configs[arm], prepared, "refresh_0060")
        pilots[arm] = train(arm, 120, "pilot")
    reports = {arm: panel(f"pilot_{arm}", m["checkpoint"]) for arm, m in pilots.items()}
    decision = choose_arm(reports)
    write_report(campaign.root / "arm_selection.json", decision)
    selected = decision["selected_arm"]
    for arm in configs:
        if arm != selected:
            retire_replay(configs[arm], pilots[arm])
            pilots[arm]["replay_retained"] = False
    campaign.status("league_campaign_selected", **decision)
    baseline = panel("development_start", prepared["frozen"]["start"])
    candidates = {"initializer": baseline, "pilot": reports[selected]}
    paths = {
        "initializer": prepared["frozen"]["start"],
        "pilot": pilots[selected]["checkpoint"],
    }
    last = dict(pilots)
    limited = False
    for previous_minutes, minutes in ((120, 225), (225, 330)):
        if selected == "rated":
            rate_league(
                campaign, configs[selected], prepared, f"refresh_{previous_minutes:04d}"
            )
        label = f"minutes_{minutes:04d}"
        try:
            milestone = train(selected, minutes, label)
        except TrainingWindowExhausted:
            limited = True
            break
        last[selected] = milestone
        paths[label] = milestone["checkpoint"]
        candidates[label] = panel(label, paths[label])
        write_report(
            campaign.root / "league_campaign_progress.json",
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
    selection = {
        "selected_arm": selected,
        "selected_milestone": best,
        "selected_checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash(checkpoint),
    }
    selection_path = campaign.root / "final_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Selection changed after confirmation began")
    write_report(selection_path, selection)
    final = panel("confirmation_candidate", checkpoint, True)
    reference = panel("confirmation_start", prepared["frozen"]["start"], True)
    delta = panel_difference(final, reference)
    resume = (
        Path(configs[selected].runs_root)
        / configs[selected].run_id
        / "checkpoints/latest_resume.pt"
    )
    result = {
        "plan": plan,
        **selection,
        "arm_decision": decision,
        "screens": {k: v["summary"] for k, v in final.items()},
        "start_screens": {k: v["summary"] for k, v in reference.items()},
        "panel_difference": delta,
        "last_milestones": last,
        "development_scores": {k: panel_score(v) for k, v in candidates.items()},
        "improvement_resolved": delta["paired_ci95"][0] > 0
        and delta["opponents"]["astra"]["match_score_difference"] >= -0.03,
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
    if not verification["source_resume_hash_matches"]:
        raise RuntimeError("Source resume changed during campaign")
    write_report(campaign.root / "verification.json", verification)
    write_report(final_path, result)
    _report(campaign.root, result, budget)
    campaign.status(
        "league_campaign_complete",
        **selection,
        improvement_resolved=result["improvement_resolved"],
    )
    return result
