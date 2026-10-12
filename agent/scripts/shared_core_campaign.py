"""Bounded shared-core consolidation, mixed-table RL, and paired confirmation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import gc
import json
import random
import shutil
import signal
import time
import traceback

import torch

from agent.eval.arena import checkpoint_hash, write_report
from agent.eval.builtin_opponents import builtin_identity
from agent.eval.inference import InferenceModel
from agent.eval.latency import benchmark_latency
from agent.eval.tables import (
    Table,
    default_tables,
    play_table,
    seat_assignments,
    weighted_summary,
)
from agent.train.checkpointing import load_net_from_checkpoint, save_checkpoint
from agent.train.learner import make_optimizer
from agent.train.reanalysis import reanalyse_policy_targets
from agent.train.reproducibility import provenance, seed_all
from agent.train.selfplay import run_selfplay
from agent.train.shared_core import (
    COUNTS,
    initialize_shared,
    donor_corpus,
    new_buffer,
    joint_step,
    validate_imitation,
    label_corpus,
    save_shared,
    restore_shared,
)
from agent.search.config import SearchConfig

EVAL_SEARCH = SearchConfig(
    backend="gumbel_tree",
    tree_core="rust",
    num_simulations=1024,
    reward_mode="binary",
    root_noise_scale=1.0,
    inference_cache_size=0,
)
TRAIN_SEARCH = replace(
    EVAL_SEARCH,
    num_simulations=64,
    max_root_candidates=32,
    q_scale=26.747517709474458,
    dirichlet_alpha=0.25668356196635866,
)
STOP = False


def stop_requested(signum: int, frame: object) -> None:
    global STOP
    STOP = True


def prepare(root: Path, foundation: Path) -> dict:
    path = root / "plan.json"
    if path.exists():
        return json.loads(path.read_text())
    budget = json.loads((root / "budget.json").read_text())
    bundle = json.loads((foundation / "bundle.json").read_text())
    teachers = {}
    for n in COUNTS:
        entry = bundle["models"][str(n)]
        source = Path(entry["checkpoint"])
        if checkpoint_hash(source) != entry["checkpoint_sha256"]:
            raise ValueError("Specialist identity changed")
        timing = json.loads((root / f"latency_specialist_{n}p_1024.json").read_text())
        if (
            not timing["qualified"]
            or timing["checkpoint_sha256"] != entry["checkpoint_sha256"]
        ):
            raise ValueError("Specialist serving profile is not qualified")
        target = root / "teachers" / f"{n}p.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        teachers[str(n)] = {
            "checkpoint": str(target.resolve()),
            "sha256": checkpoint_hash(target),
            "source": str(source),
        }
    donor_paths = {
        "2": Path(
            "agent/runs/competitive_surprise_20h_20261001/smoke_fixed/checkpoints/latest_resume.pt"
        ),
        "3": foundation / "experiments/3p_seed20261011/checkpoints/latest_resume.pt",
        "4": foundation / "experiments/4p_seed20261012/checkpoints/latest_resume.pt",
    }
    import azul_astra

    plan = {
        **budget,
        "seed": 20261008,
        "teachers": teachers,
        "donors": {
            n: {"checkpoint": str(p.resolve()), "sha256": checkpoint_hash(p)}
            for n, p in donor_paths.items()
        },
        "initialization": "2p shared core plus specialist heads/embeddings; fresh optimizer; donated observations relabelled with selected teachers; fresh RL buffers",
        "config": {
            "bounded_checkpoint_storage": True,
            "replay_capacity_per_count": 200000,
            "corpus_train_per_count": 100000,
            "corpus_validation_per_count": 4096,
            "distill_lr": 0.0001,
            "online_lr": 0.00015,
            "anchor_weight": 0.15,
            "batch_per_count": 256,
            "updates_per_round": 72,
            "games_per_count": 96,
        },
        "tables": {str(n): [asdict(t) for t in default_tables(n)] for n in COUNTS},
        "evaluation_search": asdict(EVAL_SEARCH),
        "training_search": asdict(TRAIN_SEARCH),
        "development_groups": 12,
        "confirmation_groups": {"2": 128, "3": 64, "4": 64},
        "direct_2p_confirmation_groups": 1024,
        "development_seed": 700000,
        "confirmation_seed": 900000,
        "practical_noninferiority_margin": 0.02,
        "selection": "Require every development point delta >= -0.02; maximize mean per-count delta, distillation remains fallback; held-out results never select another checkpoint",
        "acceptance": "All three weighted lower 95% difference limits > -0.02 AND direct 2p lower match-score limit > 0.48 AND all CPU profiles qualified; otherwise specialists retained. No deployment.",
        "distill_max_seconds": 3600,
        "distill_max_steps": 20000,
        "online_end": budget["deadline"] - 6 * 3600,
        "native_hashes": {
            str(p): checkpoint_hash(p)
            for p in Path(azul_astra.__file__).parent.glob("*.so")
        },
        "source_hashes": {
            str(p.resolve()): checkpoint_hash(p)
            for parent in (
                "agent/train",
                "agent/net",
                "agent/search",
                "agent/env",
                "agent/eval",
            )
            for p in Path(parent).glob("*.py")
        }
        | {str(Path(__file__).resolve()): checkpoint_hash(__file__)},
        "provenance": provenance(),
    }
    write_report(path, plan)
    return plan


class Campaign:
    def __init__(self, root: Path, plan: dict) -> None:
        self.root, self.plan = root, plan
        self.phase = "initializing"
        self.iteration = 0
        self.net = None
        self.optimizer = None
        self.buffers = {}
        self.progress = {
            "phase": "consolidation",
            "iteration": 0,
            "updates": 0,
            "games": {str(n): 0 for n in COUNTS},
        }
        self.last_guard = 0.0

    def event(self, phase: str, **details: object) -> None:
        self.phase = phase
        row = {"phase": phase, "at": time.time(), **details}
        write_report(self.root / "status.json", row)
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    def guard(self) -> None:
        if STOP:
            raise InterruptedError("Graceful stop requested")
        if time.time() >= self.plan["deadline"] - 300:
            raise InterruptedError("Campaign wall budget exhausted")
        if time.time() - self.last_guard < 60:
            return
        if shutil.disk_usage(self.root).free < 4 * 1024**3:
            raise OSError("Campaign reached the 4 GiB disk floor")
        for entry in self.plan["teachers"].values():
            if checkpoint_hash(entry["checkpoint"]) != entry["sha256"]:
                raise ValueError("Frozen teacher changed")
        for path, digest in self.plan["native_hashes"].items():
            if checkpoint_hash(path) != digest:
                raise ValueError("Native binary changed")
        for path, digest in self.plan.get("source_hashes", {}).items():
            if checkpoint_hash(path) != digest:
                raise ValueError("Campaign source changed")
        self.last_guard = time.time()

    def weights(self, name: str) -> Path:
        path = self.root / "milestones" / f"{name}.pt"
        save_checkpoint(
            path,
            self.net,
            iteration=self.progress["updates"],
            config={"shared_core": True},
            progress=self.progress.copy(),
        )
        return path

    def save(self) -> None:
        if self.net is not None and self.optimizer is not None:
            save_shared(
                self.root / "latest_resume.pt",
                self.net,
                self.optimizer,
                self.buffers,
                self.progress,
                self.plan["config"],
            )

    def evaluate(
        self,
        candidate: str,
        n: int,
        label: str,
        *,
        split: str = "development",
        groups: int | None = None,
        tables: list[Table] | None = None,
        seed: int | None = None,
    ) -> dict:
        self.guard()
        tables = tables or default_tables(n)
        groups = groups or (
            self.plan["development_groups"]
            if split == "development"
            else self.plan["confirmation_groups"][str(n)]
        )
        seed = seed if seed is not None else self.plan[split + "_seed"]
        candidate_hash = checkpoint_hash(candidate)
        model, payload = load_net_from_checkpoint(candidate)
        if n not in payload.get("trained_player_counts", []):
            raise ValueError("Evaluation candidate lacks trained player count")
        wrapped = InferenceModel(model, "cuda")
        other_net, _ = load_net_from_checkpoint(
            self.plan["teachers"][str(n)]["checkpoint"]
        )
        opponents = {"specialist": InferenceModel(other_net, "cuda")}
        identities = {
            "specialist": self.plan["teachers"][str(n)]["sha256"],
            "astra": builtin_identity("astra", n),
            "opus": "builtin_opus",
        }
        reports = {}
        for table in tables:
            self.guard()
            path = self.root / "evaluations" / f"{split}_{label}_{n}p_{table.name}.json"
            if path.exists():
                report = json.loads(path.read_text())
                if (
                    report["candidate_sha256"],
                    report["seed"],
                    report["groups"],
                    report["identities"],
                ) != (candidate_hash, seed, groups, identities):
                    raise ValueError("Cached evaluation inputs changed")
                if report["search"] != asdict(EVAL_SEARCH) or (
                    report["table"]["name"],
                    tuple(report["table"]["opponents"]),
                    report["table"]["weight"],
                ) != (table.name, table.opponents, table.weight):
                    raise ValueError("Cached evaluation table/search changed")
            else:
                self.event(
                    "evaluation", label=label, players=n, table=table.name, split=split
                )
                report = play_table(
                    wrapped,
                    opponents,
                    table,
                    EVAL_SEARCH,
                    seed=seed,
                    groups=groups,
                    progress=lambda p: self.eval_progress(label, n, split, p),
                )
                report.update(
                    candidate=candidate,
                    candidate_sha256=candidate_hash,
                    seed=seed,
                    groups=groups,
                    split=split,
                    identities=identities,
                )
                write_report(path, report)
            reports[table.name] = report
        del wrapped, model, opponents, other_net
        gc.collect()
        return reports

    def eval_progress(self, label: str, n: int, split: str, progress: dict) -> None:
        self.event("evaluation", label=label, players=n, split=split, **progress)
        self.guard()

    def screen(self, path: Path, label: str, references: dict) -> dict:
        result = {}
        for n in COUNTS:
            reports = self.evaluate(str(path), n, label)
            result[str(n)] = {
                "weighted": weighted_summary(reports, default_tables(n)),
                "difference": weighted_summary(
                    reports, default_tables(n), references[str(n)]
                ),
            }
        write_report(self.root / f"screen_{label}.json", result)
        return result

    def run(self) -> dict:
        seed_all(self.plan["seed"])
        torch.set_num_threads(1)
        self.guard()
        paths = {int(n): x["checkpoint"] for n, x in self.plan["teachers"].items()}
        self.net, teachers = initialize_shared(paths)
        self.optimizer = make_optimizer(self.net, lr=self.plan["config"]["distill_lr"])
        self.buffers = {
            n: new_buffer(self.plan["config"]["replay_capacity_per_count"])
            for n in COUNTS
        }
        corpus_path = self.root / "corpora.pt"
        if corpus_path.exists():
            data = torch.load(corpus_path, weights_only=False)
            corpora, validation = data["training"], data["validation"]
        else:
            corpora, validation = {}, {}
            for n in COUNTS:
                self.guard()
                self.event("corpus", players=n)
                donor = self.plan["donors"][str(n)]
                if checkpoint_hash(donor["checkpoint"]) != donor["sha256"]:
                    raise ValueError("Donor changed")
                corpora[n], validation[n] = donor_corpus(
                    donor["checkpoint"],
                    teachers[n],
                    n,
                    seed=self.plan["seed"] + n,
                    train_size=self.plan["config"]["corpus_train_per_count"],
                    validation_size=self.plan["config"]["corpus_validation_per_count"],
                )
                # Add fresh teacher decisions from mixed tables to inherited states.
                mixed = new_buffer(30000)
                for table in default_tables(n)[:2]:
                    play_table(
                        InferenceModel(teachers[n], "cuda"),
                        {"specialist": InferenceModel(teachers[n], "cuda")},
                        table,
                        TRAIN_SEARCH,
                        seed=200000 + n * 1000,
                        groups=max(1, 96 // len(seat_assignments(table))),
                        buffer=mixed,
                        training=True,
                    )
                batch = tuple(
                    getattr(mixed, name)[: mixed.size].clone()
                    for name in (
                        "global_feat",
                        "source_feat",
                        "legal_mask",
                        "policy_target",
                        "value_target",
                    )
                )
                labels = label_corpus(teachers[n], batch, n)
                corpora[n] = tuple(
                    torch.cat((a, b)) for a, b in zip(corpora[n], labels)
                )
            tmp = corpus_path.with_suffix(".tmp")
            torch.save({"training": corpora, "validation": validation}, tmp)
            tmp.replace(corpus_path)
        if (self.root / "latest_resume.pt").exists():
            self.progress = restore_shared(
                self.root / "latest_resume.pt",
                self.net,
                self.optimizer,
                self.buffers,
                self.plan["config"],
            )
        if self.progress["phase"] == "consolidation":
            self.event(
                "consolidation",
                initial_validation=validate_imitation(self.net, validation),
            )
            started = time.monotonic()
            best_score, stale = float("inf"), 0
            for step in range(self.plan["distill_max_steps"]):
                self.guard()
                metrics = joint_step(
                    self.net,
                    self.optimizer,
                    corpora,
                    batch_size=self.plan["config"]["batch_per_count"],
                )
                self.progress["updates"] += 1
                if (step + 1) % 250 == 0:
                    scores = validate_imitation(self.net, validation)
                    score = max(
                        v["policy_kl"] + v["value_mse"] for v in scores.values()
                    )
                    self.event(
                        "consolidation",
                        updates=self.progress["updates"],
                        validation=scores,
                        learner=metrics,
                    )
                    if score < best_score:
                        best_score, stale = score, 0
                        self.weights("distilled")
                        self.save()
                    else:
                        stale += 1
                    if (
                        all(
                            v["policy_kl"] < 0.025 and v["value_mse"] < 0.006
                            for v in scores.values()
                        )
                        or stale >= 8
                        or time.monotonic() - started
                        >= self.plan["distill_max_seconds"]
                    ):
                        break
            distilled = self.root / "milestones/distilled.pt"
            if not distilled.exists():
                self.weights("distilled")
            selected_net, _ = load_net_from_checkpoint(distilled)
            self.net.load_state_dict(selected_net.state_dict())
            self.net.trained_player_counts = list(COUNTS)
            self.optimizer = make_optimizer(
                self.net, lr=self.plan["config"]["online_lr"]
            )
            self.progress["phase"] = "online"
            self.save()
        references = {str(n): self.evaluate(paths[n], n, "specialist") for n in COUNTS}
        distilled = self.root / "milestones/distilled.pt"
        first = self.screen(distilled, "distilled", references)
        candidates = [{"path": str(distilled), "label": "distilled", "screen": first}]
        selection_path = self.root / "candidates.json"
        if selection_path.exists():
            candidates = json.loads(selection_path.read_text())["candidates"]
        if self.progress["phase"] == "online" and not any(
            x["difference"]["ci95"][1] < -0.05 for x in first.values()
        ):
            self.online(corpora, teachers, references, candidates)
        elif self.progress["phase"] == "online":
            self.event(
                "consolidation_regression",
                reason="At least one count has a resolved loss greater than five points; online RL skipped",
            )
        eligible = [
            c
            for c in candidates
            if all(v["difference"]["score"] >= -0.02 for v in c["screen"].values())
        ]
        pool = eligible or [candidates[0]]
        selected = max(
            pool,
            key=lambda c: sum(v["difference"]["score"] for v in c["screen"].values())
            / 3,
        )
        write_report(
            self.root / "selection.json",
            {
                "selected": selected,
                "eligible": bool(eligible),
                "candidate_sha256": checkpoint_hash(selected["path"]),
            },
        )
        self.progress["phase"] = "confirmation"
        self.save()
        result = {
            "selected": selected["path"],
            "selected_sha256": checkpoint_hash(selected["path"]),
            "counts": {},
            "deployed": False,
            "development_eligible": bool(eligible),
        }
        for n in COUNTS:
            self.guard()
            self.event("latency", players=n)
            timing = benchmark_latency(
                selected["path"],
                EVAL_SEARCH,
                num_players=n,
                games=16,
                seed=304913,
                deadline_s=1.8,
                wall_budget_s=2.0,
            )
            write_report(self.root / f"latency_shared_{n}p.json", timing)
            candidate = self.evaluate(
                selected["path"], n, "selected", split="confirmation"
            )
            baseline = self.evaluate(paths[n], n, "specialist", split="confirmation")
            result["counts"][str(n)] = {
                "weighted": weighted_summary(candidate, default_tables(n)),
                "reference_weighted": weighted_summary(baseline, default_tables(n)),
                "difference": weighted_summary(candidate, default_tables(n), baseline),
                "latency_qualified": timing["qualified"],
            }
            write_report(self.root / "confirmation_progress.json", result)
        direct = self.evaluate(
            selected["path"],
            2,
            "selected_direct",
            split="confirmation",
            groups=self.plan["direct_2p_confirmation_groups"],
            tables=[Table("direct", ("specialist",), 1.0)],
            seed=1100000,
        )
        result["direct_2p"] = direct["direct"]["summary"]
        result["eligible_shared_candidate"] = (
            bool(eligible)
            and all(
                v["difference"]["ci95"][0] > -0.02 and v["latency_qualified"]
                for v in result["counts"].values()
            )
            and result["direct_2p"]["win_share_ci95"][0] > 0.48
        )
        result["decision"] = (
            "shared candidate meets the predefined experimental retention gates; no deployment"
            if result["eligible_shared_candidate"]
            else "specialists retained; shared model remains experimental"
        )
        result["training"] = self.progress
        result["elapsed_hours"] = (time.time() - self.plan["started_at"]) / 3600
        write_report(self.root / "result.json", result)
        return result

    def online(
        self, corpora: dict, teachers: dict, references: dict, candidates: list
    ) -> None:
        history_net, _ = load_net_from_checkpoint(
            self.progress.get(
                "history_checkpoint", str(self.root / "milestones/distilled.pt")
            )
        )
        history = InferenceModel(history_net, "cuda")
        next_screen = time.time() + 3 * 3600
        last_save = time.monotonic()
        # Every round covers all counts before any joint learner step.
        while (
            time.time() < self.plan["online_end"] and self.progress["phase"] == "online"
        ):
            self.guard()
            iteration = self.progress["iteration"] + 1
            self.net.eval()
            sims = 256 if iteration % 4 == 0 else 64
            cfg = replace(TRAIN_SEARCH, num_simulations=sims)
            for n in COUNTS:
                if n in self.progress.get("round_completed_counts", []):
                    continue
                self.guard()
                buffer = self.buffers[n]
                buffer.iteration = iteration
                seed = self.plan["seed"] + iteration * 10000 + n * 1000
                self.event(
                    "training_games", iteration=iteration, players=n, simulations=sims
                )
                if iteration % 2:
                    metrics = run_selfplay(
                        self.net,
                        buffer,
                        num_games=self.plan["config"]["games_per_count"],
                        num_players=n,
                        num_sims=sims,
                        max_turns=100 * n,
                        dirichlet_mix=0.0,
                        dirichlet_alpha=cfg.dirichlet_alpha,
                        q_scale=cfg.q_scale,
                        time_discount=1.0,
                        reward_mode="binary",
                        device="cuda",
                        seed=seed,
                        search_backend="gumbel_tree",
                        search_tree_core="rust",
                        search_template=cfg,
                    )
                else:
                    rng = random.Random(seed)
                    tables = default_tables(n)
                    table = rng.choices(tables, weights=[t.weight for t in tables])[0]
                    # Historical learned opponents add diversity without replacing the frozen eval panel.
                    if iteration % 4 == 0:
                        table = Table(
                            table.name + "_history",
                            tuple(
                                "history" if x == "specialist" else x
                                for x in table.opponents
                            ),
                            table.weight,
                        )
                    metrics = play_table(
                        InferenceModel(self.net, "cuda"),
                        {
                            "specialist": InferenceModel(teachers[n], "cuda"),
                            "history": history,
                        },
                        table,
                        cfg,
                        seed=seed,
                        groups=max(
                            1,
                            self.plan["config"]["games_per_count"]
                            // len(seat_assignments(table)),
                        ),
                        buffer=buffer,
                        training=True,
                        opponent_search=TRAIN_SEARCH,
                    )
                self.progress["games"][str(n)] += metrics["games_total"]
                if metrics["finished"] != metrics["games_total"]:
                    raise RuntimeError(
                        "Unfinished training games excluded; stopping for inspection"
                    )
                self.event(
                    "training_games_done",
                    iteration=iteration,
                    players=n,
                    games=metrics["games_total"],
                    samples=metrics["samples_added"],
                )
                if iteration % 4 == 0:
                    stats = reanalyse_policy_targets(
                        InferenceModel(self.net, "cuda"),
                        buffer,
                        n,
                        replace(cfg, num_simulations=256),
                        positions=256,
                        seed=seed,
                    )
                    self.event("reanalysis", iteration=iteration, players=n, **stats)
                self.progress.setdefault("round_completed_counts", []).append(n)
            metrics = {}
            for _ in range(
                self.progress.get("round_updates", 0),
                self.plan["config"]["updates_per_round"],
            ):
                self.guard()
                metrics = joint_step(
                    self.net,
                    self.optimizer,
                    corpora,
                    buffers=self.buffers,
                    batch_size=self.plan["config"]["batch_per_count"],
                    anchor_weight=self.plan["config"]["anchor_weight"],
                )
                self.progress["updates"] += 1
                self.progress["round_updates"] = (
                    self.progress.get("round_updates", 0) + 1
                )
            self.progress["iteration"] = iteration
            self.progress["round_completed_counts"] = []
            self.progress["round_updates"] = 0
            self.event(
                "training_update",
                **{k: v for k, v in self.progress.items() if k != "phase"},
                learner=metrics,
            )
            if time.monotonic() - last_save >= 900:
                self.save()
                last_save = time.monotonic()
            if time.time() >= next_screen or time.time() >= self.plan["online_end"]:
                path = self.weights(f"online_{iteration}")
                self.save()
                screen = self.screen(path, f"online_{iteration}", references)
                candidates.append(
                    {
                        "path": str(path),
                        "label": f"online_{iteration}",
                        "screen": screen,
                    }
                )
                write_report(self.root / "candidates.json", {"candidates": candidates})
                history.net.load_state_dict(self.net.state_dict())
                self.progress["history_checkpoint"] = str(path)
                self.save()
                next_screen = time.time() + 3 * 3600
        # Retain the final boundary even when a screen consumes the remaining training window.
        self.save()
        if (
            candidates
            and candidates[-1]["label"] == f"online_{self.progress['iteration']}"
        ):
            return
        path = self.weights("online_final")
        if self.progress["iteration"] > 0:
            screen = self.screen(path, "online_final", references)
            candidates.append(
                {"path": str(path), "label": "online_final", "screen": screen}
            )
            write_report(self.root / "candidates.json", {"candidates": candidates})

    def closeout(self, status: str, result: dict) -> None:
        elapsed = (time.time() - self.plan["started_at"]) / 3600
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
            timespec="seconds"
        )
        lines = [
            f"\n### Shared-core campaign closeout — {stamp}\n",
            f"Run `{self.root.name}` **{status}** after {elapsed:.2f}h. No deployment.\n",
        ]
        if "counts" in result:
            for n, row in result["counts"].items():
                score, diff = row["weighted"], row["difference"]
                lines.append(
                    f"- {n}p weighted win share {score['score']:.2%}; paired change {diff['score'] * 100:+.2f}pp [{diff['ci95'][0] * 100:+.2f}, {diff['ci95'][1] * 100:+.2f}]; CPU qualified {row['latency_qualified']}.\n"
                )
            lines.append(
                f"\nDecision: {result.get('decision', 'confirmation incomplete')}.\n"
            )
        if "error" in result:
            lines.append(
                f"\nFailure/stop: `{result['error']}`. Last phase: {result.get('last_phase')}.\n"
            )
        lines.append(
            f"\nEvidence: [status](../agent/runs/{self.root.name}/status.json), [plan](../agent/runs/{self.root.name}/plan.json), [results](../agent/runs/{self.root.name}/result.json).\n"
        )
        with Path("docs/training-journal.md").open("a") as f:
            f.write("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--foundation",
        type=Path,
        default=Path("agent/runs/multiplayer_foundation_12h_20261008"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.root.resolve(), args.foundation.resolve())
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    campaign = Campaign(args.root.resolve(), plan)
    try:
        result = campaign.run()
    except BaseException as exc:
        status = (
            "stopped"
            if isinstance(exc, (InterruptedError, KeyboardInterrupt))
            else "failed"
        )
        result = {
            "error": str(exc),
            "last_phase": campaign.phase,
            "traceback": traceback.format_exc(),
            "deployed": False,
        }
        try:
            campaign.save()
        except BaseException as save_exc:
            result["save_error"] = str(save_exc)
        write_report(args.root / "failure.json", result)
        campaign.event(status, **result)
        campaign.closeout(status, result)
        raise
    campaign.event("completed", result=result)
    campaign.closeout("completed", result)


if __name__ == "__main__":
    main()
