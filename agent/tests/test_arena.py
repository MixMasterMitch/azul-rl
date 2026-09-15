from __future__ import annotations
from dataclasses import replace
import json
import pytest
import torch
from agent.env.batched_engine import BatchedEngine
from agent.eval.arena import ArenaConfig, evaluate_match, summarize_games, promotion_decision
from agent.net.model import AzulNet
from agent.train.checkpointing import save_checkpoint
from agent.search.config import SearchConfig


def test_per_game_draw_streams_do_not_depend_on_batch_order() -> None:
    a = BatchedEngine(4, 2, 'cpu', game_seeds=[11, 11, 19, 19])
    b = BatchedEngine(2, 2, 'cpu', game_seeds=[19, 11])
    assert torch.equal(a.factory_tiles[0], a.factory_tiles[1])
    assert torch.equal(a.factory_tiles[0], b.factory_tiles[1])
    assert torch.equal(a.factory_tiles[2], b.factory_tiles[0])
    for e in (a, b):
        e.factory_tiles.zero_()
        e._fill_factories()
    assert torch.equal(a.factory_tiles[0], b.factory_tiles[1])
    assert torch.equal(a.factory_tiles[2], b.factory_tiles[0])


def test_paired_arena_reproducibility_and_records(tmp_path) -> None:
    net = AzulNet(hidden=32, arch='flat')
    path = tmp_path/'net.pt'; save_checkpoint(path, net, config={'num_players': 2})
    cfg = ArenaConfig(num_games=4, game_batch_size=4, seed=133, search=SearchConfig(num_simulations=2))
    a = evaluate_match(str(path), 'random', cfg)
    b = evaluate_match(str(path), 'random', cfg)
    assert a['records'] == b['records']
    assert [g['candidate_seat'] for g in a['records']] == [0, 1, 0, 1]
    assert a['records'][0]['pair_seed'] == a['records'][1]['pair_seed']
    assert a['candidate_sha256'] == b['candidate_sha256']
    assert a['summary']['unfinished'] == 0


def test_summary_preserves_shared_and_unfinished() -> None:
    summary = summarize_games([{'pair_seed': 1, 'outcome': 'win', 'match_score': 1},
        {'pair_seed': 1, 'outcome': 'shared', 'match_score': .5},
        {'pair_seed': 2, 'outcome': 'unfinished', 'match_score': None}])
    assert summary['match_score'] == .75
    assert summary['unfinished'] == 1
    assert summary['pair_scores'] == [.75]


def test_promotion_requires_confirmation_and_no_unfinished() -> None:
    cfg = {'num_games': 1024, 'seed': 1, 'max_turns': 300, 'game_batch_size': 128,
           'inference_device': 'cpu', 'split': 'confirmation', 'search': {}, 'greedy': False}
    records = [{'pair_seed': 1+i//2, 'candidate_seat': i%2,
                'outcome': 'loss' if i % 3 == 0 else 'win',
                'match_score': 0 if i % 3 == 0 else 1} for i in range(1024)]
    champ = {'config': cfg, 'records': records, 'candidate_sha256': 'candidate',
             'opponent_sha256': 'champion', 'opponent_search': {}}
    opus = {'config': cfg, 'records': list(records), 'candidate_sha256': 'candidate', 'opponent': 'opus'}
    reference = {**opus, 'candidate_sha256': 'champion'}
    assert promotion_decision(champ, opus, reference)['promote']
    with pytest.raises(ValueError, match='different candidates'):
        promotion_decision(champ, opus, {**reference, 'candidate_sha256': 'wrong'})
    with pytest.raises(ValueError, match='every ordered'):
        promotion_decision(champ, {**opus, 'records': records[:-1]}, reference)
    champ['records'] = [{**records[0], 'outcome': 'unfinished', 'match_score': None}] + records[1:]
    assert not promotion_decision(champ, opus, reference)['promote']
    cfg['split'] = 'development'
    with pytest.raises(ValueError): promotion_decision(champ, opus, reference)


def test_timed_arena_gives_each_game_its_own_deadline(monkeypatch) -> None:
    from agent.eval import arena
    batches = []
    def search(engine, model, search_config):
        batches.append(engine.batch_size)
        return engine.legal_action_mask().long().argmax(1), None
    monkeypatch.setattr(arena, 'gumbel_root_act', search)
    e = BatchedEngine(4, 2, 'cpu', seed=1)
    arena._net_policy(None, e, SearchConfig(move_deadline_s=5), False)
    assert batches == [1, 1, 1, 1]
