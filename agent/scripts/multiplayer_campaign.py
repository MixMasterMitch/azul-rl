"""Bounded 3p/4p specialist training with an immutable two-player route."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import signal
import time
from zoneinfo import ZoneInfo

import torch

from agent.eval.arena import checkpoint_hash, paired_interval, write_report
from agent.eval.latency import benchmark_latency
from agent.eval.multiplayer import MultiplayerArenaConfig, evaluate_multiplayer
from agent.obs.run import Run
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint, save_checkpoint
from agent.train.loop import LoopConfig, run_loop
from agent.train.multiplayer import initialize_specialist, load_specialist
from agent.train.presets import enhanced_2p_config
from agent.train.reproducibility import provenance


SEARCH = SearchConfig(
    backend="gumbel_tree",
    tree_core="rust",
    num_simulations=256,
    root_noise_scale=1.0,
    reward_mode="binary",
    inference_cache_size=0,
)


def specialist_config(
    root: Path, players: int, initializer: str, seed: int
) -> LoopConfig:
    model, _ = load_net_from_checkpoint(initializer)
    return replace(
        enhanced_2p_config(),
        num_players=players,
        hidden=model.hidden,
        device="cuda",
        init_from=initializer,
        run_id=f"{players}p_seed{seed}",
        runs_root=str(root / "experiments"),
        seed=seed,
        league_root=str(root / "experiments" / f"{players}p_seed{seed}" / "league"),
        replay_capacity=500_000,
        bounded_checkpoint_storage=True,
        keep_recent_checkpoints=1,
        league_max_entries=12,
        league_keep_recent=4,
        search_inference_cache_size=0,
        policy_fast_weight=0.25,
        policy_surprise_fraction=0,
        policy_surprise_record=False,
        selfplay_turns_per_player=90,
        eval_games=0,
    )


def prepare(
    root: Path, baseline: str, expected_hash: str, hours: float, seed: int
) -> dict:
    """Freeze all inputs before training. Re-entry never resets the wall budget."""
    path = root / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if (
            plan["source"],
            plan["frozen_2p"]["checkpoint_sha256"],
            plan["hours"],
            plan["seed"],
        ) != (str(Path(baseline).resolve()), expected_hash, hours, seed):
            raise ValueError("Frozen campaign inputs changed")
        return plan
    if not 2 <= hours <= 48:
        raise ValueError("Campaign budget must be between two and 48 hours")
    if checkpoint_hash(baseline) != expected_hash:
        raise ValueError("Baseline identity differs from the approved reference")
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "initializers/frozen_2p.pt"
    frozen.parent.mkdir(exist_ok=True)
    if frozen.exists():
        raise ValueError(
            "Partial preparation found; inspect it before starting a new run"
        )
    shutil.copy2(baseline, frozen)
    if checkpoint_hash(frozen) != expected_hash:
        raise ValueError("Baseline copy changed")
    initializers = {}
    configs = {}
    for players in (3, 4):
        torch.manual_seed(seed + players)
        initializers[str(players)] = initialize_specialist(
            str(frozen), str(root / f"initializers/transfer_{players}p.pt"), players
        )
        configs[str(players)] = asdict(
            specialist_config(
                root, players, initializers[str(players)]["checkpoint"], seed + players
            )
        )
    entry = {"checkpoint": str(frozen), "checkpoint_sha256": expected_hash}
    started = time.time()
    import azul_astra

    plan = {
        "source": str(Path(baseline).resolve()),
        "frozen_2p": entry,
        "hours": hours,
        "started_at": started,
        "deadline": started + hours * 3600,
        "seed": seed,
        "initializers": initializers,
        "configs": configs,
        "search": asdict(SEARCH),
        "disk_floor_gib": 6,
        "provenance": provenance(),
        "native_hashes": {
            str(p): checkpoint_hash(p)
            for p in Path(azul_astra.__file__).parent.glob("*.so")
        },
        "initialization": "2p policy transfer; fresh multiplayer value readout, replay and optimizer",
        "two_player_protection": "separate frozen checkpoint and verified per-player routing",
        "no_automatic_restart": True,
        "deployment": False,
        "evaluation": "complete seat rotations; win share 1/k among actual k tied winners",
    }
    write_report(path, plan)
    write_report(
        root / "bundle.json",
        {"frozen_2p": entry, "models": {"2": entry}, "deployed": False},
    )
    return plan


class MultiplayerCampaign:
    def __init__(self, root: Path, plan: dict) -> None:
        self.root, self.plan = root, plan

    def status(self, stage: str, **fields: object) -> None:
        value = {"stage": stage, "updated_at": time.time(), **fields}
        write_report(self.root / "status.json", value)
        print(json.dumps(value), flush=True)

    def guard(self, reserve_minutes: float = 0) -> None:
        if time.time() + reserve_minutes * 60 >= self.plan["deadline"]:
            raise TimeoutError("Campaign budget reserved for closeout")
        if shutil.disk_usage(self.root).free < self.plan["disk_floor_gib"] * 1024**3:
            raise RuntimeError("Campaign disk floor reached")
        entry = self.plan["frozen_2p"]
        for path in (entry["checkpoint"], self.plan["source"]):
            if checkpoint_hash(path) != entry["checkpoint_sha256"]:
                raise RuntimeError("Frozen 2p baseline changed")
        for path, digest in self.plan["native_hashes"].items():
            if checkpoint_hash(path) != digest:
                raise RuntimeError("Native binary changed during the campaign")

    def smoke(self) -> None:
        marker = self.root / "smoke_passed.json"
        if marker.exists():
            return
        for players in (3, 4):
            self.guard()
            self.status("smoke", num_players=players)
            cfg = replace(
                LoopConfig(**self.plan["configs"][str(players)]),
                run_id=f"smoke_{players}p",
                runs_root=str(self.root),
                league_root=str(self.root / f"smoke_{players}p/league"),
                max_iters=4,
                max_wall_minutes=15,
                selfplay_games=12,
                selfplay_sims=8,
                selfplay_full_sims=16,
                league_opponent_sims=8,
                replay_capacity=4096,
                learner_batch=32,
                learner_steps_per_iter=2,
                checkpoint_every=4,
                reanalysis_positions=8,
                reanalysis_sims=16,
                reanalysis_snapshot_capacity=64,
            )
            run = Run(cfg.run_id, runs_root=cfg.runs_root)
            try:
                result = run_loop(run, cfg, explicit_fields=set(asdict(cfg)))
            finally:
                run.close()
            if result["stopped"]:
                raise KeyboardInterrupt
            if result["iter"] != 4:
                raise RuntimeError(
                    "Multiplayer smoke did not complete all training phases"
                )
            gc.collect()
            torch.cuda.empty_cache()
        write_report(
            marker, {"passed": True, "players": [3, 4], "updated_at": time.time()}
        )

    def train(self, players: int, minutes: float, label: str) -> dict:
        self.guard(reserve_minutes=90)
        path = self.root / f"milestones/{players}p_{label}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved["target_minutes"] != minutes
                or checkpoint_hash(saved["checkpoint"]) != saved["checkpoint_sha256"]
            ):
                raise RuntimeError("Milestone changed")
            return saved
        cfg = replace(
            LoopConfig(**self.plan["configs"][str(players)]), max_wall_minutes=minutes
        )
        self.status(
            "training",
            num_players=players,
            target_minutes=minutes,
            heartbeat=str(Path(cfg.runs_root) / cfg.run_id / "heartbeat.json"),
        )
        run = Run(cfg.run_id, runs_root=cfg.runs_root)
        try:
            result = run_loop(run, cfg, explicit_fields=set(asdict(cfg)))
        finally:
            run.close()
        if result["stopped"]:
            raise KeyboardInterrupt
        if result["training_wall_s"] < minutes * 60:
            raise RuntimeError(
                "Training stopped before its requested cumulative budget"
            )
        resume = Path(cfg.runs_root) / cfg.run_id / "checkpoints/latest_resume.pt"
        net, payload = load_net_from_checkpoint(resume)
        checkpoint = path.with_suffix(".pt")
        save_checkpoint(
            checkpoint,
            net,
            iteration=payload["iteration"],
            config=payload["config"],
            progress=payload["progress"],
        )
        record = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "iteration": payload["iteration"],
            "training_wall_s": result["training_wall_s"],
            "target_minutes": minutes,
        }
        write_report(path, record)
        del net, payload
        gc.collect()
        torch.cuda.empty_cache()
        self.guard()
        return record

    def evaluate(
        self,
        players: int,
        label: str,
        checkpoint: str,
        opponent: str,
        *,
        confirmation: bool = False,
    ) -> dict:
        self.guard()
        path = self.root / f"evaluations/{players}p_{label}_{opponent}.json"
        games = 384 if confirmation else 96
        cfg = MultiplayerArenaConfig(
            num_players=players,
            num_games=games,
            inference_device="cuda",
            search=SEARCH,
            seed=self.plan["seed"]
            + players * 10_000
            + (20_000_000 if confirmation else 10_000_000),
            split="confirmation" if confirmation else "development",
        )
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["candidate_sha256"] != checkpoint_hash(checkpoint) or saved[
                "config"
            ] != asdict(cfg):
                raise ValueError("Cached multiplayer evaluation inputs changed")
            return saved
        self.status(
            "evaluation",
            num_players=players,
            label=label,
            opponent=opponent,
            games=games,
        )
        report = evaluate_multiplayer(
            checkpoint,
            opponent,
            cfg,
            progress=lambda fields: self.status(
                "evaluation",
                num_players=players,
                label=label,
                opponent=opponent,
                games=games,
                **fields,
            ),
        )
        write_report(path, report)
        if report["summary"]["unfinished"]:
            raise RuntimeError("Unfinished multiplayer evaluation games")
        gc.collect()
        torch.cuda.empty_cache()
        return report

    def run(self) -> dict:
        self.guard()
        torch.set_num_threads(1)
        self.smoke()
        budget_path = self.root / "allocation.json"
        if budget_path.exists():
            allocation = json.loads(budget_path.read_text())
        else:
            # At least 35% plus the remaining setup time is available for all
            # screens, held-out evaluations, checkpoint writes, and CPU audits.
            minutes = int((self.plan["deadline"] - time.time()) / 60 * 0.60 / 2)
            if minutes < 30:
                raise TimeoutError("Insufficient time for both multiplayer arms")
            allocation = {
                "training_minutes_per_player_count": minutes,
                "milestones": sorted(set([30, min(120, minutes), minutes])),
            }
            write_report(budget_path, allocation)
        for players in (3, 4):
            init = self.plan["initializers"][str(players)]["checkpoint"]
            for opponent in ("heuristic", "opus", "astra"):
                self.evaluate(players, "initializer", init, opponent)
        histories: dict[int, list[dict]] = {3: [], 4: []}
        for minutes in allocation["milestones"]:
            for players in (3, 4):
                milestone = self.train(players, minutes, f"{minutes}m")
                scores = {
                    op: self.evaluate(
                        players, f"{minutes}m", milestone["checkpoint"], op
                    )["summary"]
                    for op in ("heuristic", "opus", "astra")
                }
                histories[players].append({"milestone": milestone, "scores": scores})
        write_report(self.root / "development.json", histories)
        bundle = json.loads((self.root / "bundle.json").read_text())
        results = {}
        for players in (3, 4):
            # Development chooses once; confirmation never chooses a runner-up.
            chosen = max(
                histories[players],
                key=lambda item: sum(
                    item["scores"][op]["win_share"]
                    for op in ("heuristic", "opus", "astra")
                ),
            )
            checkpoint = chosen["milestone"]["checkpoint"]
            write_report(self.root / f"selection_{players}p.json", chosen)
            comparisons = {}
            for opponent in ("opus", "astra"):
                candidate = self.evaluate(
                    players, "confirmation", checkpoint, opponent, confirmation=True
                )
                baseline = self.evaluate(
                    players,
                    "initializer_confirmation",
                    self.plan["initializers"][str(players)]["checkpoint"],
                    opponent,
                    confirmation=True,
                )
                differences = [
                    a - b
                    for a, b in zip(
                        candidate["summary"]["group_scores"],
                        baseline["summary"]["group_scores"],
                    )
                ]
                comparisons[opponent] = {
                    "candidate": candidate["summary"],
                    "initializer": baseline["summary"],
                    "win_share_difference": sum(differences) / len(differences),
                    "difference_ci95": paired_interval(differences),
                }
            self.status("cpu_latency", num_players=players)
            latency = benchmark_latency(
                checkpoint,
                SEARCH,
                num_players=players,
                games=8,
                deadline_s=1.8,
                wall_budget_s=2.0,
            )
            write_report(self.root / f"latency_{players}p.json", latency)
            results[str(players)] = {
                "selected": chosen["milestone"],
                "comparisons": comparisons,
                "latency_qualified": latency["qualified"],
                "deployed": False,
            }
            bundle["models"][str(players)] = {
                key: chosen["milestone"][key]
                for key in ("checkpoint", "checkpoint_sha256")
            }
        self.guard()
        write_report(self.root / "bundle.json", bundle)
        # Verify the actual resolver still selects the immutable 2p model.
        frozen = load_specialist(self.root / "bundle.json", 2)
        del frozen
        result = {
            "players": results,
            "two_player_checkpoint_unchanged": True,
            "elapsed_hours": (time.time() - self.plan["started_at"]) / 3600,
            "completed_at": time.time(),
            "deployed": False,
        }
        write_report(self.root / "result.json", result)
        self.status("completed", result=result)
        self.journal("completed", result)
        return result

    def journal(self, outcome: str, result: dict) -> None:
        """Append closeout without rewriting earlier or concurrent journal entries."""
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
            timespec="seconds"
        )
        relative = self.root.relative_to(Path(__file__).resolve().parents[2])
        text = f"\n### Multiplayer campaign closeout — {stamp}\n\nRun `{relative.name}` **{outcome}**. "
        if outcome == "completed":
            text += f"Elapsed {result['elapsed_hours']:.2f}h. Frozen 2p checkpoint and route verified unchanged; no deployment.\n\n"
            for players, row in result["players"].items():
                text += f"- {players}p: selected iteration {row['selected']['iteration']}; CPU latency qualified: {row['latency_qualified']}. "
                for opponent, comp in row["comparisons"].items():
                    lo, hi = comp["difference_ci95"]
                    text += (
                        f"Against {opponent}: win share {comp['candidate']['win_share']:.2%} versus initializer "
                        f"{comp['initializer']['win_share']:.2%}; paired change {100 * comp['win_share_difference']:+.2f}pp "
                        f"[{100 * lo:+.2f}, {100 * hi:+.2f}] (384 games per model). "
                    )
                text += "\n"
        else:
            text += f"No completion or strength gain is inferred. Detail: `{result.get('error', outcome)}`.\n"
        text += f"\nEvidence: [run status](../{relative}/status.json), [frozen plan](../{relative}/plan.json).\n"
        journal = Path(__file__).resolve().parents[2] / "docs/training-journal.md"
        with journal.open("a") as handle:
            handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--hours", required=True, type=float)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = prepare(root, args.baseline, args.baseline_sha256, args.hours, args.seed)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    if (root / "result.json").exists():
        raise ValueError("Campaign is already complete")
    campaign = MultiplayerCampaign(root, plan)

    def stop(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        campaign.run()
    except (Exception, KeyboardInterrupt) as exc:
        stage = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
        detail = {"error": str(exc) or type(exc).__name__}
        campaign.status(stage, **detail)
        campaign.journal(stage, detail)
        raise


if __name__ == "__main__":
    main()
