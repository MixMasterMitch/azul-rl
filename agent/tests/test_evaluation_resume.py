from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

import pytest
import torch

from agent.eval.inference import InferenceModel
from agent.eval.tables import Table, play_table, seat_assignments, validate_table_prefix
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.scripts.complete_shared_evaluation import BatchStore, validate_report


@pytest.mark.parametrize("players", [2, 3, 4])
def test_saved_batch_interruption_matches_uninterrupted_games(
    tmp_path, players: int
) -> None:
    torch.manual_seed(555)
    wrapped = InferenceModel(AzulNet(hidden=16, arch="source_attn"))
    roles = {
        2: ("specialist",),
        3: ("specialist", "random"),
        4: ("specialist", "random", "heuristic"),
    }[players]
    table = Table("mixed", roles, 1.0)
    batch_size = len(seat_assignments(table))
    cfg = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=2,
        max_root_candidates=2,
    )
    request = {
        "table": asdict(table),
        "seed": 155,
        "groups": 3,
        "batch_size": batch_size,
        "max_turns": 100 * players,
        "candidate_sha256": "frozen",
        "search": asdict(cfg),
    }
    expected = play_table(
        wrapped,
        {"specialist": wrapped},
        table,
        cfg,
        seed=155,
        groups=3,
        batch_size=batch_size,
    )
    store = BatchStore(tmp_path / "batches", request)

    def interrupt_after_saved_batch(progress: dict) -> None:
        assert len(BatchStore(tmp_path / "batches", request).records) == batch_size
        raise InterruptedError("test pause")

    with pytest.raises(InterruptedError, match="test pause"):
        play_table(
            wrapped,
            {"specialist": wrapped},
            table,
            cfg,
            seed=155,
            groups=3,
            batch_size=batch_size,
            on_batch=store.save,
            progress=interrupt_after_saved_batch,
        )
    # New store reconstructs records from disk, as in a new process.
    resumed = BatchStore(tmp_path / "batches", request)
    actual = play_table(
        wrapped,
        {"specialist": wrapped},
        table,
        cfg,
        seed=155,
        groups=3,
        batch_size=batch_size,
        resume_records=resumed.records,
        on_batch=resumed.save,
    )
    assert actual["records"] == expected["records"]
    assert actual["summary"] == expected["summary"]
    assert len(resumed.records) == 3 * batch_size
    assert len(list((tmp_path / "batches").glob("batch_*.json"))) == 3

    def unexpected_batch(batch: dict) -> None:
        raise AssertionError("A completed table must not replay games")

    complete = play_table(
        wrapped,
        {"specialist": wrapped},
        table,
        cfg,
        seed=155,
        groups=3,
        batch_size=batch_size,
        resume_records=resumed.records,
        on_batch=unexpected_batch,
    )
    assert complete["records"] == expected["records"]
    changed = deepcopy(request)
    changed["candidate_sha256"] = "changed"
    with pytest.raises(ValueError, match="request differs"):
        BatchStore(tmp_path / "batches", changed)
    wrong = deepcopy(actual["records"])
    wrong[0]["group_seed"] += 1
    with pytest.raises(ValueError, match="schedule"):
        validate_table_prefix(
            wrong,
            table,
            seed=155,
            groups=3,
            batch_size=batch_size,
            max_turns=100 * players,
        )
    wrong = deepcopy(actual["records"])
    wrong[0]["outcome"] = "unfinished"
    with pytest.raises(ValueError, match="outcome"):
        validate_table_prefix(
            wrong,
            table,
            seed=155,
            groups=3,
            batch_size=batch_size,
            max_turns=100 * players,
        )
    with pytest.raises(ValueError, match="batches"):
        validate_table_prefix(
            actual["records"][:1],
            table,
            seed=155,
            groups=3,
            batch_size=batch_size,
            max_turns=100 * players,
        )
    # Missing a middle batch is an error, never an instruction to drop its games.
    (tmp_path / "batches" / f"batch_{batch_size:06d}.json").unlink()
    with pytest.raises(ValueError, match="gap"):
        BatchStore(tmp_path / "batches", request)


def test_finished_report_rejects_metadata_or_summary_changes() -> None:
    table = Table("direct", ("random",), 1.0)
    net = InferenceModel(AzulNet(hidden=16, arch="source_attn"))
    cfg = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=2,
        max_root_candidates=2,
    )
    report = play_table(net, {}, table, cfg, seed=8, groups=2, batch_size=2)
    report.update(
        candidate_sha256="frozen", seed=8, groups=2, split="confirmation", identities={}
    )
    request = {
        **report,
        "table": {"name": "direct", "opponents": ["random"], "weight": 1.0},
        "batch_size": 2,
        "max_turns": 200,
    }
    validate_report(report, request)
    changed = deepcopy(report)
    changed["candidate_sha256"] = "changed"
    with pytest.raises(ValueError, match="candidate_sha256"):
        validate_report(changed, request)
    changed = deepcopy(report)
    changed["summary"]["win_share"] = 1.5
    with pytest.raises(ValueError, match="summary"):
        validate_report(changed, request)


def test_evaluation_only_completion_reuses_panels_and_writes_final_result(
    tmp_path, monkeypatch
) -> None:
    import json
    import time
    from agent.eval.arena import checkpoint_hash
    from agent.train.checkpointing import save_checkpoint
    from agent.scripts import complete_shared_evaluation as module

    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/training-journal.md").write_text(
        "# Journal\n\nLast reviewed: pending\n\nThis is the durable record.\n"
    )
    root = tmp_path / "run"
    root.mkdir()
    directory = root / "completion"
    directory.mkdir()
    (root / "REPORT.md").write_text("Earlier interrupted review\n")
    net = AzulNet(hidden=16, arch="source_attn").eval()
    net.trained_player_counts = [2, 3, 4]
    checkpoint = root / "selected.pt"
    save_checkpoint(checkpoint, net)
    digest = checkpoint_hash(checkpoint)
    search = asdict(
        SearchConfig(
            backend="gumbel_tree",
            tree_core="rust",
            num_simulations=2,
            max_root_candidates=2,
        )
    )
    teachers = {
        str(n): {"checkpoint": str(checkpoint), "sha256": digest} for n in (2, 3, 4)
    }
    original = {
        "teachers": teachers,
        "evaluation_search": search,
        "confirmation_groups": {"2": 2, "3": 2, "4": 2},
        "confirmation_seed": 345,
        "tables": {
            str(n): [asdict(Table("small", tuple(["random"] * (n - 1)), 1.0))]
            for n in (2, 3, 4)
        },
    }
    module.durable_json(root / "plan.json", original)
    module.durable_json(root / "review_20261010.json", {"training": {"updates": 123}})
    for n in (2, 3, 4):
        module.durable_json(
            root / f"latency_shared_{n}p.json",
            {"checkpoint_sha256": digest, "search": search, "qualified": True},
        )
    wrapped = InferenceModel(net)
    old_hashes = {}
    for n in (2, 3):
        table = module.table_from_dict(original["tables"][str(n)][0])
        for label in ("selected", "specialist"):
            request = module.job_request(
                original, table, n, label, str(checkpoint), digest, 2, 345
            )
            report = play_table(
                wrapped,
                {"specialist": wrapped},
                table,
                SearchConfig(**search),
                seed=345,
                groups=2,
            )
            report.update(
                candidate=str(checkpoint),
                candidate_sha256=digest,
                seed=345,
                groups=2,
                split="confirmation",
                identities=request["identities"],
            )
            path = root / "evaluations" / f"confirmation_{label}_{n}p_small.json"
            module.durable_json(path, report)
            old_hashes[str(path)] = checkpoint_hash(path)
    direct = module.job_request(
        original,
        Table("direct", ("specialist",), 1.0),
        2,
        "selected_direct",
        str(checkpoint),
        digest,
        2,
        1100000,
    )
    table = module.table_from_dict(original["tables"]["4"][0])
    jobs = [direct] + [
        module.job_request(original, table, 4, label, str(checkpoint), digest, 2, 345)
        for label in ("selected", "specialist")
    ]
    plan = {
        "deadline": time.time() + 3600,
        "started_at": time.time(),
        "source_hashes": {},
        "native_hashes": {},
        "frozen_inputs": old_hashes,
        "selected": str(checkpoint),
        "selected_sha256": digest,
        "teachers": teachers,
        "jobs": jobs,
        "development_eligible": True,
        "margin": 0.02,
    }
    monkeypatch.setattr(
        module, "InferenceModel", lambda net, device: InferenceModel(net, "cpu")
    )
    result = module.Completion(root, directory, plan).run()
    assert set(result["counts"]) == {"2", "3", "4"}
    assert result["direct_2p"]["games"] == 4
    assert result["additional_training_steps"] == 0 and not result["deployed"]
    assert json.loads((root / "result.json").read_text()) == json.loads(
        (directory / "result.json").read_text()
    )
    assert (root / "FINAL_REPORT.md").exists()
    assert checkpoint_hash(checkpoint) == digest
    assert all(checkpoint_hash(path) == sha for path, sha in old_hashes.items())
    assert json.loads((directory / "status.json").read_text())["phase"] == "completed"
