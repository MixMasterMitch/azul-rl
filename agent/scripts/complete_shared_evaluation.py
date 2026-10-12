"""Finish a frozen shared-model confirmation without entering a training loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
from zoneinfo import ZoneInfo

import torch

from agent.eval.arena import checkpoint_hash
from agent.eval.builtin_opponents import builtin_identity
from agent.eval.inference import InferenceModel
from agent.eval.tables import (
    Table,
    play_table,
    seat_assignments,
    summarize_table,
    validate_table_prefix,
    weighted_summary,
)
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint
from agent.train.reproducibility import provenance

STOP = False


def normalized(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def durable_json(path: Path, value: dict) -> None:
    """Flush file and containing directory before acknowledging a finished batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BatchStore:
    """Immutable completed batches, bound to all inputs that affect game results."""

    def __init__(self, directory: Path, request: dict) -> None:
        self.directory = directory
        self.request = normalized(request)
        canonical = json.dumps(self.request, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode()).hexdigest()
        metadata = directory / "request.json"
        if metadata.exists():
            if json.loads(metadata.read_text()) != self.request:
                raise ValueError("Saved evaluation request differs")
        else:
            durable_json(metadata, self.request)
        self.records: list[dict] = []
        self.wall_s = 0.0
        for path in sorted(directory.glob("batch_*.json")):
            batch = json.loads(path.read_text())
            self._validate_batch(batch)
            if path.name != f"batch_{batch['start']:06d}.json":
                raise ValueError("Batch filename differs from its offset")
            self.records.extend(batch["records"])
            self.wall_s += batch["wall_s"]

    def _validate_batch(self, batch: dict) -> None:
        if batch.get("request_sha256") != self.digest or batch["start"] != len(
            self.records
        ):
            raise ValueError("Saved batches have a gap or different request identity")
        table = Table(
            **{
                **self.request["table"],
                "opponents": tuple(self.request["table"]["opponents"]),
            }
        )
        total = self.request["groups"] * len(seat_assignments(table))
        if (
            len(batch["records"])
            != min(self.request["batch_size"], total - len(self.records))
            or not batch["records"]
        ):
            raise ValueError("Saved batch has the wrong size")
        validate_table_prefix(
            self.records + batch["records"],
            table,
            seed=self.request["seed"],
            groups=self.request["groups"],
            batch_size=self.request["batch_size"],
            max_turns=self.request["max_turns"],
        )
        if not 0 <= batch["wall_s"] < float("inf"):
            raise ValueError("Invalid batch duration")

    def save(self, batch: dict) -> None:
        batch = {**batch, "request_sha256": self.digest}
        self._validate_batch(batch)
        path = self.directory / f"batch_{batch['start']:06d}.json"
        if path.exists():
            raise ValueError("Refusing to overwrite a completed batch")
        durable_json(path, batch)
        self.records.extend(batch["records"])
        self.wall_s += batch["wall_s"]


def table_from_dict(value: dict) -> Table:
    return Table(value["name"], tuple(value["opponents"]), value["weight"])


def job_request(
    plan: dict,
    table: Table,
    n: int,
    label: str,
    candidate: str,
    candidate_hash: str,
    groups: int,
    seed: int,
) -> dict:
    return normalized(
        {
            "candidate": candidate,
            "candidate_sha256": candidate_hash,
            "label": label,
            "num_players": n,
            "table": asdict(table),
            "search": plan["evaluation_search"],
            "opponent_search": plan["evaluation_search"],
            "seed": seed,
            "groups": groups,
            "split": "confirmation",
            "batch_size": 96,
            "max_turns": 100 * n,
            "bot_workers": 8,
            "inference_device": "cuda",
            "identities": {
                "specialist": plan["teachers"][str(n)]["sha256"],
                "astra": builtin_identity("astra", n),
                "opus": "builtin_opus",
            },
        }
    )


def prepare(root: Path, completion: Path) -> dict:
    manifest_path = completion / "plan.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    original = json.loads((root / "plan.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    selected, digest = selection["selected"]["path"], selection["candidate_sha256"]
    if checkpoint_hash(selected) != digest:
        raise ValueError("Selected checkpoint changed")
    for entry in original["teachers"].values():
        if checkpoint_hash(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("Frozen specialist changed")
    for path, expected in original["native_hashes"].items():
        if checkpoint_hash(path) != expected:
            raise ValueError("Native extension changed")
    # Only the table persistence change is allowed against the historical code.
    allowed = str(Path("agent/eval/tables.py").resolve())
    changes = {}
    for path, expected in original["source_hashes"].items():
        actual = checkpoint_hash(path)
        if actual != expected:
            if path != allowed:
                raise ValueError(
                    f"Unexpected source change since original evaluation: {path}"
                )
            changes[path] = {"before": expected, "after": actual}
    jobs = [
        job_request(
            original,
            Table("direct", ("specialist",), 1.0),
            2,
            "selected_direct",
            selected,
            digest,
            original["direct_2p_confirmation_groups"],
            1100000,
        )
    ]
    for value in original["tables"]["4"]:
        table = table_from_dict(value)
        for label, candidate, sha in [
            ("selected", selected, digest),
            (
                "specialist",
                original["teachers"]["4"]["checkpoint"],
                original["teachers"]["4"]["sha256"],
            ),
        ]:
            jobs.append(
                job_request(
                    original,
                    table,
                    4,
                    label,
                    candidate,
                    sha,
                    original["confirmation_groups"]["4"],
                    original["confirmation_seed"],
                )
            )
    frozen = [
        root / "plan.json",
        root / "selection.json",
        root / "confirmation_progress.json",
        root / "latest_resume.pt",
        root / "corpora.pt",
        root / "review_20261010.json",
        *sorted((root / "evaluations").glob("confirmation_*.json")),
        *sorted(root.glob("latency_shared_*.json")),
    ]
    current_sources = {
        path: checkpoint_hash(path) for path in original["source_hashes"]
    }
    current_sources[str(Path(__file__).resolve())] = checkpoint_hash(__file__)
    started = time.time()
    manifest = {
        "schema_version": 1,
        "root": str(root),
        "started_at": started,
        "deadline": started + 18 * 3600,
        "expected_hours": 12,
        "maximum_hours": 18,
        "purpose": "Complete the originally specified confirmation; no training or reselection",
        "selected": selected,
        "selected_sha256": digest,
        "development_eligible": selection["eligible"],
        "teachers": original["teachers"],
        "jobs": jobs,
        "original_plan_sha256": checkpoint_hash(root / "plan.json"),
        "frozen_inputs": {str(path): checkpoint_hash(path) for path in frozen},
        "source_changes": changes,
        "source_hashes": current_sources,
        "native_hashes": original["native_hashes"],
        "provenance": provenance(),
        "margin": original["practical_noninferiority_margin"],
        "deployed": False,
        "training_steps": 0,
        "validation": json.loads((completion / "validation.json").read_text()),
    }
    durable_json(manifest_path, manifest)
    return manifest


def validate_report(report: dict, request: dict) -> None:
    for key in (
        "candidate_sha256",
        "num_players",
        "table",
        "search",
        "opponent_search",
        "seed",
        "groups",
        "split",
        "identities",
    ):
        if normalized(report[key]) != request[key]:
            raise ValueError(f"Completed evaluation differs: {key}")
    table = table_from_dict(request["table"])
    validate_table_prefix(
        report["records"],
        table,
        seed=request["seed"],
        groups=request["groups"],
        batch_size=request["batch_size"],
        max_turns=request["max_turns"],
    )
    expected = request["groups"] * len(seat_assignments(table))
    if len(report["records"]) != expected:
        raise ValueError("Completed report is incomplete")
    if normalized(summarize_table(report["records"], table)) != normalized(
        report["summary"]
    ):
        raise ValueError("Stored summary differs from its game records")


class Completion:
    def __init__(self, root: Path, directory: Path, plan: dict) -> None:
        self.root, self.directory, self.plan = root, directory, plan
        self.original = json.loads((root / "plan.json").read_text())
        self.last_check = 0.0

    def guard(self, full: bool = False) -> None:
        if STOP or time.time() >= self.plan["deadline"] - 900:
            raise InterruptedError(
                "Evaluation completion stop requested or runtime limit reached"
            )
        if not full and time.time() - self.last_check < 60:
            return
        if shutil.disk_usage(self.directory).free < 2 * 1024**3:
            raise OSError("Evaluation disk floor reached")
        hashes = {
            **self.plan["source_hashes"],
            **self.plan["native_hashes"],
            self.plan["selected"]: self.plan["selected_sha256"],
            **{x["checkpoint"]: x["sha256"] for x in self.plan["teachers"].values()},
        }
        if full:
            hashes.update(self.plan["frozen_inputs"])
        for path, digest in hashes.items():
            if checkpoint_hash(path) != digest:
                raise ValueError(f"Frozen evaluation input changed: {path}")
        self.last_check = time.time()

    def event(self, phase: str, **details: object) -> None:
        event = {"phase": phase, "at": time.time(), **details}
        durable_json(self.directory / "status.json", event)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event), flush=True)

    def report_path(self, request: dict) -> Path:
        return (
            self.root
            / "evaluations"
            / f"confirmation_{request['label']}_{request['num_players']}p_{request['table']['name']}.json"
        )

    def evaluate(self, request: dict) -> dict:
        self.guard()
        destination = self.report_path(request)
        if destination.exists():
            report = json.loads(destination.read_text())
            validate_report(report, request)
            return report
        table = table_from_dict(request["table"])
        key = f"{request['label']}_{request['num_players']}p_{table.name}"
        store = BatchStore(
            self.directory / "batches" / key,
            {
                **request,
                "source_hashes": self.plan["source_hashes"],
                "native_hashes": self.plan["native_hashes"],
            },
        )
        total = request["groups"] * len(seat_assignments(table))
        self.event("evaluation", job=key, completed=len(store.records), total=total)
        if len(store.records) == total:
            report = {
                "table": asdict(table),
                "num_players": request["num_players"],
                "search": request["search"],
                "opponent_search": request["opponent_search"],
                "records": store.records,
                "samples_added": 0,
                "finished": total,
                "games_total": total,
                "summary": summarize_table(store.records, table),
            }
        else:
            net, payload = load_net_from_checkpoint(request["candidate"])
            n = request["num_players"]
            if n not in payload.get("trained_player_counts", []):
                raise ValueError("Candidate player-count metadata differs")
            specialist, other_payload = load_net_from_checkpoint(
                self.plan["teachers"][str(n)]["checkpoint"]
            )
            if n not in other_payload.get("trained_player_counts", []):
                raise ValueError("Specialist player-count metadata differs")
            wrapped = InferenceModel(net, "cuda")
            opponents = {"specialist": InferenceModel(specialist, "cuda")}

            def progress(row: dict) -> None:
                self.event(
                    "evaluation",
                    job=key,
                    **row,
                    saved_batches=len(list(store.directory.glob("batch_*.json"))),
                )
                self.guard()

            report = play_table(
                wrapped,
                opponents,
                table,
                SearchConfig(**request["search"]),
                seed=request["seed"],
                groups=request["groups"],
                batch_size=request["batch_size"],
                max_turns=request["max_turns"],
                bot_workers=request["bot_workers"],
                resume_records=store.records,
                on_batch=store.save,
                progress=progress,
            )
            del wrapped, opponents, net, specialist, payload, other_payload
            gc.collect()
        report.update(
            candidate=request["candidate"],
            candidate_sha256=request["candidate_sha256"],
            seed=request["seed"],
            groups=request["groups"],
            split=request["split"],
            identities=request["identities"],
            wall_s=store.wall_s,
            completion_manifest=str(self.directory / "plan.json"),
            evaluation_request=request,
            batch_progress=str(store.directory),
        )
        validate_report(report, request)
        durable_json(destination, report)
        self.event("table_completed", job=key, games=total, summary=report["summary"])
        return report

    def combine(self, include_missing: bool = False) -> dict:
        result = {
            "selected": self.plan["selected"],
            "selected_sha256": self.plan["selected_sha256"],
            "counts": {},
            "deployed": False,
            "development_eligible": self.plan["development_eligible"],
            "evaluation_completion_manifest": str(self.directory / "plan.json"),
        }
        for n in (2, 3, 4):
            tables = [table_from_dict(x) for x in self.original["tables"][str(n)]]
            panels = {label: {} for label in ("selected", "specialist")}
            for label in panels:
                candidate = (
                    self.plan["selected"]
                    if label == "selected"
                    else self.plan["teachers"][str(n)]["checkpoint"]
                )
                sha = (
                    self.plan["selected_sha256"]
                    if label == "selected"
                    else self.plan["teachers"][str(n)]["sha256"]
                )
                for table in tables:
                    request = job_request(
                        self.original,
                        table,
                        n,
                        label,
                        candidate,
                        sha,
                        self.original["confirmation_groups"][str(n)],
                        self.original["confirmation_seed"],
                    )
                    path = self.report_path(request)
                    if path.exists():
                        report = json.loads(path.read_text())
                        validate_report(report, request)
                        panels[label][table.name] = report
            if any(len(p) != len(tables) for p in panels.values()):
                if include_missing:
                    raise ValueError("Final confirmation panel is incomplete")
                continue
            timing = json.loads((self.root / f"latency_shared_{n}p.json").read_text())
            if (
                timing["checkpoint_sha256"] != self.plan["selected_sha256"]
                or timing["search"] != self.original["evaluation_search"]
            ):
                raise ValueError("Latency qualification identity differs")
            result["counts"][str(n)] = {
                "weighted": weighted_summary(panels["selected"], tables),
                "reference_weighted": weighted_summary(panels["specialist"], tables),
                "difference": weighted_summary(
                    panels["selected"], tables, panels["specialist"]
                ),
                "latency_qualified": timing["qualified"],
            }
        direct_path = self.report_path(self.plan["jobs"][0])
        if direct_path.exists():
            direct = json.loads(direct_path.read_text())
            validate_report(direct, self.plan["jobs"][0])
            result["direct_2p"] = direct["summary"]
        elif include_missing:
            raise ValueError("Final direct confirmation is incomplete")
        return result

    def run(self) -> dict:
        torch.set_num_threads(1)
        self.guard(full=True)
        self.event(
            "starting",
            remaining_games=sum(
                j["groups"] * len(seat_assignments(table_from_dict(j["table"])))
                for j in self.plan["jobs"]
            ),
            training_steps=0,
        )
        for job in self.plan["jobs"]:
            self.evaluate(job)
            durable_json(self.directory / "confirmation_progress.json", self.combine())
        self.guard(full=True)
        result = self.combine(include_missing=True)
        margin = self.plan["margin"]
        result["eligible_shared_candidate"] = (
            result["development_eligible"]
            and all(
                v["difference"]["ci95"][0] > -margin and v["latency_qualified"]
                for v in result["counts"].values()
            )
            and result["direct_2p"]["win_share_ci95"][0] > 0.5 - margin
        )
        result["decision"] = (
            "shared candidate meets the predefined experimental retention gates; no deployment"
            if result["eligible_shared_candidate"]
            else "specialists retained; shared model remains experimental"
        )
        result["training"] = json.loads(
            (self.root / "review_20261010.json").read_text()
        )["training"]
        result["evaluation_completion_wall_hours"] = (
            time.time() - self.plan["started_at"]
        ) / 3600
        result["additional_training_steps"] = 0
        if (self.root / "result.json").exists():
            raise ValueError("Refusing to overwrite an existing final result")
        durable_json(self.directory / "result.json", result)
        durable_json(self.root / "result.json", result)
        self.closeout("completed", result)
        self.event(
            "completed",
            decision=result["decision"],
            result_path=str(self.root / "result.json"),
        )
        return result

    def closeout(self, status: str, result: dict) -> None:
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
            timespec="seconds"
        )
        lines = [
            f"\n### Shared-core evaluation completion — {stamp}\n\n",
            f"Authorized evaluation-only continuation **{status}**. No additional training or deployment.\n\n",
        ]
        if status == "completed":
            lines += [
                "| Count | Shared weighted win share | Specialist | Paired change, 95% CI |\n",
                "|---|---:|---:|---:|\n",
            ]
            for n, row in result["counts"].items():
                d = row["difference"]
                lines.append(
                    f"| {n}p | {row['weighted']['score']:.2%} | {row['reference_weighted']['score']:.2%} | {100 * d['score']:+.2f}pp [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] |\n"
                )
            direct = result["direct_2p"]
            lines.append(
                f"\nThe separate {direct['games']:,}-game 2p direct match scored **{direct['win_share']:.2%}** "
                f"[{direct['win_share_ci95'][0]:.2%}, {direct['win_share_ci95'][1]:.2%}].\n\n"
            )
            lines.append(
                f"Decision: **{result['decision']}**. The original two-point tolerance and all-count gates were unchanged. "
                "These intervals reflect games on the fixed opponent panel for one training seed.\n\n"
            )
            lines.append(
                "The existing 48-position CPU audits for each count remain qualified at 1,024 simulations. "
                "All inherited checkpoint/native hashes and completed reports were verified. "
                "The selected checkpoint remains iteration 338; the final training resume remains iteration 408.\n\n"
            )
        else:
            lines.append(
                f"Stop/failure: `{result.get('error')}`. Completed batches are durable; only an in-flight batch may need repeating.\n\n"
            )
        lines.append(
            f"Evidence: [completion status](../agent/runs/{self.root.name}/{self.directory.name}/status.json), "
            f"[completion manifest](../agent/runs/{self.root.name}/{self.directory.name}/plan.json), "
            f"[final results](../agent/runs/{self.root.name}/result.json).\n"
        )
        with Path("docs/training-journal.md").open("a") as stream:
            stream.write("".join(lines))
        if status == "completed":
            report = "# Shared-core completed confirmation\n" + "".join(lines)
            # Use valid report-relative links rather than the journal's ../agent/runs prefix.
            report = report.replace(f"../agent/runs/{self.root.name}/", "")
            (self.root / "FINAL_REPORT.md").write_text(report)
            journal_path = Path("docs/training-journal.md")
            journal = journal_path.read_text()
            start = journal.index("Last reviewed:")
            end = journal.index("\nThis is the durable record", start)
            journal = (
                journal[:start]
                + (
                    f"Last reviewed: **{stamp[:10]}**. Shared-core confirmation is complete, "
                    "including the 4p mixed-table panel and separate 2p direct match. "
                    f"Decision: **{result['decision']}**. The evaluation service has finished.\n"
                )
                + journal[end:]
            )
            journal_path.write_text(journal)
            earlier = self.root / "REPORT.md"
            earlier.write_text(
                "> Confirmation is now complete. See [the final report](FINAL_REPORT.md) "
                "for all results; the review below records the earlier interrupted state.\n\n"
                + earlier.read_text()
            )


def stop_requested(signum: int, frame: object) -> None:
    global STOP
    STOP = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--completion", default="eval_completion_20261010")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / args.completion
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "runner.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = prepare(root, directory)
    if args.prepare_only:
        print(
            json.dumps(
                {"plan": str(directory / "plan.json"), "jobs": len(plan["jobs"])}
            )
        )
        return
    if (directory / "result.json").exists():
        print("Evaluations already completed; no additional work started.")
        return
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    completion = Completion(root, directory, plan)
    try:
        completion.run()
    except BaseException as exc:
        status = (
            "stopped"
            if isinstance(exc, (InterruptedError, KeyboardInterrupt))
            else "failed"
        )
        failure = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "additional_training_steps": 0,
            "deployed": False,
        }
        durable_json(directory / "failure.json", failure)
        completion.event(status, **failure)
        completion.closeout(status, failure)
        raise


if __name__ == "__main__":
    main()
