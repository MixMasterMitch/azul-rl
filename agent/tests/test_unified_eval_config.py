"""Unified eval policy configuration."""

from __future__ import annotations

import torch

from agent.eval.tournament import EvalConfig
from agent.net.model import AzulNet
from agent.train import unified_eval as UE


def test_unified_eval_default_temperature_matches_tournament() -> None:
    assert UE.UnifiedEvalConfig().temperature == EvalConfig().temperature


def test_unified_eval_passes_configured_temperature(monkeypatch) -> None:
    seen: list[float] = []

    def fake_gumbel_root_act(engine, net, num_sims, temperature, q_scale, **kwargs):
        del net, num_sims, q_scale
        seen.append(float(temperature))
        legal = engine.legal_action_mask()
        actions = legal.to(torch.int64).argmax(dim=-1)
        improved = torch.zeros_like(legal, dtype=torch.float32)
        return actions, improved

    monkeypatch.setattr(UE.G, "gumbel_root_act", fake_gumbel_root_act)

    net = AzulNet(hidden=32, arch="flat")
    config = UE.UnifiedEvalConfig(
        total_games=1,
        num_sims=2,
        max_turns=1,
        turns_per_player=0,
        temperature=0.125,
        weight_2p=1.0,
        weight_3p=0.0,
        weight_4p=0.0,
    )
    UE.run_unified_eval(net.state_dict(), [], config, seed=1, hidden=32, arch="flat")

    assert seen == [0.125]


def test_merge_eval_rates_uses_finished_counts() -> None:
    result = UE._merge_eval_results([
        {"metrics": {"games_2p": 10, "finished_2p": 10, "wins_2p": 8, "losses_2p": 2, "shared_2p": 0, "eval_winrate_2p": .8}},
        {"metrics": {"games_2p": 4, "finished_2p": 4, "wins_2p": 1, "losses_2p": 1, "shared_2p": 2, "eval_winrate_2p": .25}},
    ])["metrics"]
    assert result["eval_winrate_2p"] == 9 / 14
    assert result["eval_match_score_2p"] == 10 / 14


def test_shared_two_player_result_counts_half_each() -> None:
    from agent.env.batched_engine import SHARED_VICTORY
    result = UE.extract_pairwise_results([["a", "b"]], torch.tensor([SHARED_VICTORY]), torch.tensor([True]), 2)
    assert [(r.winner, r.loser, r.weight) for r in result] == [("a", "b", .5), ("b", "a", .5)]


def test_handle_preserves_partial_worker_results() -> None:
    import queue
    class Worker:
        def is_alive(self): return True
        def join(self, timeout=None): pass
        def terminate(self): pass
    class Queue(queue.Queue):
        def close(self): pass
        def join_thread(self): pass
    handle = UE.UnifiedEvalHandle(UE.UnifiedEvalConfig())
    handle._queue = Queue()
    handle._processes = [Worker(), Worker()]
    handle._iteration_tag = 50
    handle._expected_workers = 2
    import time
    handle._started = time.monotonic()
    handle._context = {'entity': 'ckpt:4'}
    handle._queue.put((50, 0, {'pairwise': [], 'metrics': {'games_2p': 1, 'finished_2p': 1, 'wins_2p': 1}}))
    assert handle.try_collect() is None
    assert len(handle._pending) == 1
    handle._queue.put((50, 1, {'pairwise': [], 'metrics': {'games_2p': 1, 'finished_2p': 1, 'wins_2p': 0, 'losses_2p': 1}}))
    iteration, result = handle.try_collect()
    assert iteration == 50 and result['metrics']['eval_winrate_2p'] == .5
    assert result['job_context']['entity'] == 'ckpt:4'
    assert not handle.is_active()
