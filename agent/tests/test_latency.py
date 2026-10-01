import pytest

from agent.eval import latency
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.train.checkpointing import save_checkpoint


@pytest.mark.parametrize("completed", [None, 32, 64])
def test_latency_requires_actual_native_completed_budget(
    tmp_path, monkeypatch, completed: int | None
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, AzulNet(hidden=32, arch="source_attn"), config={"num_players": 2}
    )

    def search(engine, model, *, search_config, perf):
        assert search_config.move_deadline_s == 1.8
        if completed is not None:
            perf.add_count("native_tree_simulations_per_root", completed)
        return engine.legal_action_mask().int().argmax(-1), None

    monkeypatch.setattr(latency, "gumbel_root_act", search)
    result = latency.benchmark_latency(
        str(path),
        SearchConfig(backend="gumbel_tree", tree_core="rust"),
        games=1,
        deadline_s=1.8,
    )
    assert result["qualified"] == (completed == 64)
    assert all(
        r["completed_simulations"] == (completed or 0) for r in result["records"]
    )


@pytest.mark.parametrize(
    "elapsed,completed,qualified",
    [(1.8023, 64, True), (2.001, 64, False), (1.79, 63, False)],
)
def test_outer_wall_budget_is_separate_from_search_stop(
    tmp_path, monkeypatch, elapsed, completed, qualified
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, AzulNet(hidden=32, arch="source_attn"), config={"num_players": 2}
    )

    def search(engine, model, *, search_config, perf):
        assert search_config.move_deadline_s == 1.8
        perf.add_count("native_tree_simulations_per_root", completed)
        return engine.legal_action_mask().int().argmax(-1), None

    ticks = iter([0.0, elapsed] * 3)
    monkeypatch.setattr(latency, "gumbel_root_act", search)
    monkeypatch.setattr(latency.time, "monotonic", lambda: next(ticks))
    result = latency.benchmark_latency(
        str(path),
        SearchConfig(backend="gumbel_tree", tree_core="rust"),
        games=1,
        deadline_s=1.8,
        wall_budget_s=2.0,
    )
    assert result["qualified"] is qualified
