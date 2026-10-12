from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

import pytest
import torch

from agent.env.engine import GameEngine
from agent.eval.inference import InferenceModel
from agent.eval.tables import (
    Table,
    default_tables,
    seat_assignments,
    summarize_table,
    weighted_summary,
    play_table,
)
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.train.checkpointing import save_checkpoint, load_net_from_checkpoint
from agent.train.learner import make_optimizer
from agent.train.reanalysis import reanalyse_policy_targets
from agent.train.reproducibility import seed_all
from agent.train.shared_core import (
    initialize_shared,
    label_corpus,
    joint_step,
    new_buffer,
    save_shared,
    restore_shared,
)


def test_seat_permutations_cover_every_opponent_order() -> None:
    assert (
        len(seat_assignments(Table("mixed", ("astra", "opus", "specialist"), 1))) == 24
    )
    assert (
        len(seat_assignments(Table("repeat", ("astra", "astra", "specialist"), 1)))
        == 12
    )
    for n in (2, 3, 4):
        assert sum(t.weight for t in default_tables(n)) == pytest.approx(1)
        for t in default_tables(n):
            assignments = seat_assignments(t)
            for role in set(("candidate", *t.opponents)):
                counts = [
                    sum(row[seat] == role for row in assignments) for seat in range(n)
                ]
                assert len(set(counts)) == 1


def report_for(table: Table, scores: list[float]) -> dict:
    return {
        "table": asdict(table),
        "num_players": len(table.opponents) + 1,
        "search": {"num_simulations": 1024},
        "identities": {"specialist": "fixed"},
        "summary": {
            "group_scores": {str(i): x for i, x in enumerate(scores)},
            "unfinished": 0,
        },
    }


def test_weighted_paired_scores_are_not_pooled_by_games() -> None:
    tables = [Table("a", ("astra",), 0.75), Table("b", ("opus",), 0.25)]
    a = {
        t.name: report_for(t, [0.8, 0.6] if t.name == "a" else [0.2, 0.4])
        for t in tables
    }
    b = {t.name: report_for(t, [0.5, 0.5]) for t in tables}
    summary = weighted_summary(a, tables)
    assert summary["score"] == pytest.approx(0.6)
    assert summary["ci95"] == pytest.approx((0.55, 0.65))
    assert weighted_summary(a, tables, b)["score"] == pytest.approx(0.1)
    b["a"]["identities"]["specialist"] = "changed"
    with pytest.raises(ValueError, match="identities"):
        weighted_summary(a, tables, b)
    a["b"]["summary"]["group_scores"].pop("0")
    with pytest.raises(ValueError, match="Seed blocks"):
        weighted_summary(a, tables)


def test_table_summaries_reject_incomplete_or_duplicate_assignments() -> None:
    table = Table("mixed", ("astra", "opus"), 1)
    records = [
        {"group_seed": 1, "seats": list(row), "outcome": "shared", "win_share": 0.5}
        for row in seat_assignments(table)
    ]
    assert summarize_table(records, table)["win_share"] == 0.5
    with pytest.raises(ValueError, match="complete"):
        summarize_table(records[:-1], table)
    records[-1] = records[0]
    with pytest.raises(ValueError, match="complete"):
        summarize_table(records, table)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_actual_mixed_games_replay_and_reanalysis(n: int) -> None:
    net = AzulNet(hidden=16, arch="source_attn").eval()
    wrapped = InferenceModel(net)
    opponents = (
        ("heuristic",)
        if n == 2
        else ("heuristic", "random")
        if n == 3
        else ("heuristic", "random", "heuristic")
    )
    table = Table("mixed", opponents, 1)
    cfg = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=2,
        max_root_candidates=2,
    )
    buffer = new_buffer(20000, 1024)
    r = play_table(
        wrapped, {}, table, cfg, seed=221, groups=1, buffer=buffer, training=True
    )
    assert r["finished"] == r["games_total"] == len(seat_assignments(table))
    assert buffer.size > 0
    assert torch.isfinite(buffer.policy_target[: buffer.size]).all()
    assert set(buffer.value_target[: buffer.size, :n].unique().tolist()) <= {-1, 0, 1}
    assert torch.allclose(
        buffer.policy_target[: buffer.size].sum(-1), torch.ones(buffer.size)
    )
    assert len(buffer.snapshots) > 0
    stats = reanalyse_policy_targets(wrapped, buffer, n, cfg, positions=4, seed=3)
    assert stats["positions"] == 4
    # Evaluation is reproducible and every score is attributed to the candidate seat.
    a = play_table(wrapped, {}, table, cfg, seed=224, groups=1)
    b = play_table(wrapped, {}, table, cfg, seed=224, groups=1)
    assert a["records"] == b["records"]
    for row in a["records"]:
        expected = (
            1 / len(row["winning_seats"])
            if row["candidate_seat"] in row["winning_seats"]
            else 0
        )
        assert row["win_share"] == expected


def test_shared_update_and_full_resume_preserve_teachers_and_rng(tmp_path) -> None:
    seed_all(321)
    paths = {}
    for n in (2, 3, 4):
        path = tmp_path / f"{n}.pt"
        save_checkpoint(
            path, AzulNet(hidden=16, arch="source_attn"), config={"num_players": n}
        )
        paths[n] = str(path)
    net, teachers = initialize_shared(paths, "cpu")
    teachers_before = {n: deepcopy(t.state_dict()) for n, t in teachers.items()}
    corpora = {}
    buffers = {n: new_buffer(200, 32) for n in (2, 3, 4)}
    for n in (2, 3, 4):
        engine = GameEngine(8, n, seed=4)
        g, s = encode_state(engine)
        corpora[n] = label_corpus(teachers[n], (g, s, engine.legal_action_mask()), n)
        buffers[n].add(*corpora[n], policy_sims=64)
    optimizer = make_optimizer(net, lr=0.001)
    before = net.g_in[0].weight.clone()
    joint_step(net, optimizer, corpora, batch_size=8)
    assert not torch.equal(before, net.g_in[0].weight)
    assert all(p.grad is not None for p in net.pc_embed.parameters())
    for n, t in teachers.items():
        assert all(
            torch.equal(v, teachers_before[n][k]) for k, v in t.state_dict().items()
        )
    config = {"bounded_checkpoint_storage": True}
    progress = {"phase": "online", "iteration": 1}
    path = tmp_path / "resume.pt"
    save_shared(path, net, optimizer, buffers, progress, config)
    loaded, payload = load_net_from_checkpoint(path)
    assert payload["trained_player_counts"] == [2, 3, 4]
    expected = joint_step(net, optimizer, corpora, batch_size=8, buffers=buffers)
    other_optimizer = make_optimizer(loaded)
    other_buffers = {n: new_buffer(200, 32) for n in (2, 3, 4)}
    assert (
        restore_shared(path, loaded, other_optimizer, other_buffers, config) == progress
    )
    actual = joint_step(
        loaded, other_optimizer, corpora, batch_size=8, buffers=other_buffers
    )
    assert expected == actual
    assert all(
        torch.equal(a, b)
        for a, b in zip(net.state_dict().values(), loaded.state_dict().values())
    )


def test_campaign_evaluation_cache_and_paired_mixed_panel(
    tmp_path, monkeypatch
) -> None:
    import time
    from agent.scripts import shared_core_campaign as campaign_module
    from agent.eval.arena import checkpoint_hash

    path = tmp_path / "specialist.pt"
    net = AzulNet(hidden=16, arch="source_attn")
    save_checkpoint(path, net, config={"num_players": 3})
    plan = {
        "deadline": time.time() + 3600,
        "native_hashes": {},
        "development_groups": 2,
        "development_seed": 56,
        "teachers": {"3": {"checkpoint": str(path), "sha256": checkpoint_hash(path)}},
    }
    campaign = campaign_module.Campaign(tmp_path, plan)
    monkeypatch.setattr(
        campaign_module, "InferenceModel", lambda n, device: InferenceModel(n, "cpu")
    )
    monkeypatch.setattr(
        campaign_module,
        "EVAL_SEARCH",
        SearchConfig(
            backend="gumbel_tree",
            tree_core="rust",
            num_simulations=2,
            max_root_candidates=2,
        ),
    )
    tables = [Table("mixed", ("specialist", "random"), 1)]
    a = campaign.evaluate(str(path), 3, "same", tables=tables)
    b = campaign.evaluate(str(path), 3, "same", tables=tables)
    assert weighted_summary(a, tables, b)["score"] == 0
    assert a["mixed"]["records"] == b["mixed"]["records"]
    with pytest.raises(ValueError, match="table/search"):
        campaign.evaluate(
            str(path), 3, "same", tables=[Table("mixed", ("random", "random"), 1)]
        )
