from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from agent.env.engine import GameEngine
from agent.eval.arena import checkpoint_hash
from agent.eval.multiplayer import (
    MultiplayerArenaConfig,
    evaluate_multiplayer,
    outcome_record,
    summarize_multiplayer,
)
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.scripts.multiplayer_campaign import prepare, specialist_config
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint, save_checkpoint
from agent.train.learner import make_optimizer, step
from agent.train.multiplayer import initialize_specialist, load_specialist


@pytest.mark.parametrize("players", [3, 4])
def test_transfer_and_multiplayer_learning_leave_frozen_2p_identical(
    tmp_path, players: int
) -> None:
    source = tmp_path / "baseline.pt"
    net = AzulNet(hidden=16, arch="source_attn").eval()
    save_checkpoint(source, net, config={"num_players": 2})
    before_hash = checkpoint_hash(source)
    engine2 = GameEngine(4, 2, seed=81)
    g2, s2 = encode_state(engine2)
    legal2 = engine2.legal_action_mask()
    with torch.no_grad():
        before = net(g2, s2, legal2, 2)
    target = tmp_path / f"{players}p.pt"
    initialize_specialist(str(source), str(target), players)
    specialist, payload = load_net_from_checkpoint(target)
    key = str(players - 2)
    assert payload["trained_player_counts"] == []
    assert "buffer" not in payload and "optimizer_state_dict" not in payload
    assert torch.equal(specialist.pc_embed.weight[players - 2], net.pc_embed.weight[0])
    for a, b in zip(
        specialist.policy_heads[key].parameters(), net.policy_heads["0"].parameters()
    ):
        assert torch.equal(a, b)
    engine = GameEngine(4, players, seed=55)
    g, s = encode_state(engine)
    legal = engine.legal_action_mask()
    specialist.eval()
    with torch.no_grad():
        values = specialist(g, s, legal, players)[1][:, :players]
    torch.testing.assert_close(values, torch.full_like(values, 2 / players - 1))
    trunk_before = specialist.g_in[0].weight.detach().clone()
    metrics = step(
        specialist,
        make_optimizer(specialist),
        g,
        s,
        legal,
        legal.float() / legal.sum(1, keepdim=True),
        torch.ones(4, 4),
        players,
    )
    assert metrics["skipped"] == 0
    assert not torch.equal(trunk_before, specialist.g_in[0].weight)
    assert checkpoint_hash(source) == before_hash
    entry = {"checkpoint": str(source), "checkpoint_sha256": before_hash}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"frozen_2p": entry, "models": {"2": entry}}))
    routed = load_specialist(bundle, 2)
    with torch.no_grad():
        after = routed(g2, s2, legal2, 2)
    for a, b in zip(before, after):
        assert torch.equal(a, b)


def test_specialist_route_rejects_substitution_and_checkpoint_tampering(
    tmp_path,
) -> None:
    source = tmp_path / "baseline.pt"
    save_checkpoint(
        source, AzulNet(hidden=16, arch="source_attn"), config={"num_players": 2}
    )
    entry = {"checkpoint": str(source), "checkpoint_sha256": checkpoint_hash(source)}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {"frozen_2p": entry, "models": {"2": {**entry, "checkpoint": "another.pt"}}}
        )
    )
    with pytest.raises(ValueError, match="frozen baseline"):
        load_specialist(bundle, 2)
    bundle.write_text(
        json.dumps({"frozen_2p": entry, "models": {"2": entry, "3": entry}})
    )
    with pytest.raises(ValueError, match="not marked trained"):
        load_specialist(bundle, 3)
    with source.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        load_specialist(bundle, 2)


@pytest.mark.parametrize("players", [2, 5])
def test_transfer_rejects_invalid_counts(tmp_path, players: int) -> None:
    with pytest.raises(ValueError):
        initialize_specialist("missing.pt", str(tmp_path / "new.pt"), players)


def test_transfer_never_overwrites_source(tmp_path) -> None:
    source = tmp_path / "baseline.pt"
    source.write_bytes(b"protected")
    with pytest.raises(ValueError, match="new destination"):
        initialize_specialist(str(source), str(source), 3)
    assert source.read_bytes() == b"protected"


@pytest.mark.parametrize(
    "utilities,seat,expected,share",
    [
        ([0, 0, -1], 2, "loss", 0),
        ([0, 0, -1], 1, "shared", 0.5),
        ([0, -1, 0, 0], 2, "shared", 1 / 3),
        ([-1, -1, 1, -1], 2, "win", 1),
        ([-1, -1, 1, -1], 0, "loss", 0),
    ],
)
def test_multiplayer_tied_winners_do_not_reward_losing_seats(
    utilities, seat, expected, share
) -> None:
    row = outcome_record(utilities, seat, ended=True)
    assert row["outcome"] == expected and row["win_share"] == share


def test_summary_clusters_complete_seat_rotations() -> None:
    records = [
        {
            "group_seed": 1,
            "candidate_seat": seat,
            "outcome": "shared",
            "win_share": 1 / 3,
        }
        for seat in range(3)
    ]
    records += [
        {
            "group_seed": 2,
            "candidate_seat": 0,
            "outcome": "unfinished",
            "win_share": None,
        }
    ]
    result = summarize_multiplayer(records, 3)
    assert result["complete_rotations"] == 1
    assert result["unfinished"] == 1
    assert result["group_scores"] == [1 / 3]
    assert result["win_share"] == 1 / 3
    assert result["sole_wins"] == 0


@pytest.mark.parametrize("players", [3, 4])
def test_multiplayer_arena_native_search_and_reproducible_rotations(
    tmp_path, players: int
) -> None:
    source = tmp_path / "model.pt"
    save_checkpoint(
        source, AzulNet(hidden=16, arch="source_attn"), config={"num_players": players}
    )
    cfg = MultiplayerArenaConfig(
        num_players=players,
        num_games=players,
        game_batch_size=players,
        seed=71,
        bot_workers=1,
        search=SearchConfig(
            backend="gumbel_tree",
            tree_core="rust",
            num_simulations=4,
            max_root_candidates=4,
        ),
    )
    a = evaluate_multiplayer(str(source), "heuristic", cfg)
    b = evaluate_multiplayer(str(source), "heuristic", cfg)
    assert a["records"] == b["records"]
    assert a["summary"]["unfinished"] == 0
    assert a["summary"]["complete_rotations"] == 1
    assert {r["candidate_seat"] for r in a["records"]} == set(range(players))
    assert {r["group_seed"] for r in a["records"]} == {71}
    unfinished = evaluate_multiplayer(
        str(source), "heuristic", replace(cfg, max_turns=1)
    )
    assert unfinished["summary"]["unfinished"] == players
    assert unfinished["summary"]["complete_rotations"] == 0


def test_arena_rejects_incomplete_rotations() -> None:
    with pytest.raises(ValueError, match="complete seat rotations"):
        MultiplayerArenaConfig(num_players=3, num_games=128)


def test_campaign_preparation_is_immutable_and_separates_replay(tmp_path) -> None:
    source = tmp_path / "baseline.pt"
    save_checkpoint(
        source, AzulNet(hidden=16, arch="source_attn"), config={"num_players": 2}
    )
    digest = checkpoint_hash(source)
    root = tmp_path / "campaign"
    plan = prepare(root, str(source), digest, 12, 20261008)
    assert prepare(root, str(source), digest, 12, 20261008) == plan
    with pytest.raises(ValueError, match="Frozen campaign inputs"):
        prepare(root, str(source), digest, 24, 20261008)
    for players in (3, 4):
        config = specialist_config(
            root, players, plan["initializers"][str(players)]["checkpoint"], 7
        )
        assert config.num_players == players
        assert config.eval_games == 0 and config.policy_surprise_fraction == 0
        assert config.search_inference_cache_size == 0
        assert config.league_root.endswith(f"{players}p_seed7/league")
    assert checkpoint_hash(source) == digest
    assert load_specialist(root / "bundle.json", 2).hidden == 16
