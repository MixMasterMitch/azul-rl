"""Smoke test for bot-augmented self-play."""

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.net import encoder as ENC
from agent.net.model import AzulNet
from agent.train.bot_selfplay import run_bot_selfplay
from agent.train.replay_buffer import ReplayBuffer


def _buffer(capacity: int = 10_000) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        d_global=ENC.D_GLOBAL,
        n_sources=ENC.NUM_SOURCES,
        d_source=ENC.D_SOURCE,
        num_actions=A.NUM_ACTIONS,
        max_players=BE.MAX_PLAYERS,
        device="cpu",
    )


def test_bot_selfplay_adds_samples() -> None:
    net = AzulNet(hidden=64, arch="flat")
    buffer = _buffer()
    metrics = run_bot_selfplay(
        net,
        buffer,
        num_games=4,
        num_players=2,
        num_sims=4,
        max_turns=80,
        device="cpu",
        seed=0,
        reward_mode="binary",
    )
    assert metrics["games_total"] == 4
    assert metrics["samples_added"] > 0
    assert metrics["bot_heuristic_games"] + metrics["bot_opus_games"] == 4


def test_batched_bot_selfplay_uses_opus_bot_for_opus_games(monkeypatch) -> None:
    import agent.train.bot_selfplay as BS

    class SpyOpusBot:
        calls = 0

        def __init__(self, seed: int | None = None):
            del seed

        def select_action(self, engine: BE.BatchedEngine, game_idx: int) -> int:
            SpyOpusBot.calls += 1
            legal = engine.legal_action_mask()[game_idx].nonzero(as_tuple=True)[0].tolist()
            return int(legal[0])

    def fail_if_heuristic_used(engine: BE.BatchedEngine):
        del engine
        raise AssertionError("opus-only bot self-play should not call batched heuristic")

    monkeypatch.setattr(BS, "HeuristicOpusBot", SpyOpusBot)
    monkeypatch.setattr(BS, "batched_heuristic_actions", fail_if_heuristic_used)

    net = AzulNet(hidden=64, arch="flat")
    metrics = run_bot_selfplay(
        net,
        _buffer(capacity=100),
        num_games=1,
        num_players=2,
        num_sims=1,
        max_turns=1,
        device="cpu",
        seed=0,
        opus_prob=1.0,
        bot_policy="batched",
    )

    assert metrics["bot_opus_games"] == 1
    assert SpyOpusBot.calls > 0
