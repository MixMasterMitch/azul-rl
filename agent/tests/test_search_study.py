from __future__ import annotations

import json
from pathlib import Path
import time

import pytest
import torch

from agent.eval.arena import ArenaConfig
from agent.net.model import AzulNet
from agent.scripts import search_study
from agent.scripts.search_study import BudgetExpired, SearchStudy, combine_shards, paired_difference
from agent.train.checkpointing import save_checkpoint


def test_interrupted_match_resumes_without_replaying_completed_seed_pairs(tmp_path: Path, monkeypatch) -> None:
    torch.set_num_threads(1)
    checkpoint = tmp_path/'net.pt'
    save_checkpoint(checkpoint, AzulNet(hidden=32, arch='flat'), config={'num_players': 2})
    deadline = time.time()+1000
    options = dict(device='cpu', seed=491, bot_workers=1, deadline=deadline, batch_size=2)
    study = SearchStudy(tmp_path/'resumed', str(checkpoint), str(checkpoint), **options)
    real_match = search_study.evaluate_match
    calls = []
    def interrupt(candidate, opponent, config, *args, **kwargs):
        calls.append(config.seed)
        if len(calls) == 2:
            raise RuntimeError('simulated interruption')
        return real_match(candidate, opponent, config, *args, **kwargs)
    monkeypatch.setattr(search_study, 'evaluate_match', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        study.match('test', 'one_ply', 'random', 4)
    saved = tmp_path/'resumed/evaluations/test/batch_0000.json'
    stamp = saved.stat().st_mtime_ns
    monkeypatch.setattr(search_study, 'evaluate_match', real_match)
    resumed = SearchStudy(tmp_path/'resumed', str(checkpoint), str(checkpoint), **options)
    result = resumed.match('test', 'one_ply', 'random', 4)
    truth = SearchStudy(tmp_path/'truth', str(checkpoint), str(checkpoint), **options).match('test', 'one_ply', 'random', 4)
    assert saved.stat().st_mtime_ns == stamp
    assert result['records'] == truth['records']
    assert result['value_calibration_by_shard'] == truth['value_calibration_by_shard']
    assert result['complete'] and result['summary']['unfinished'] == 0
    assert resumed.completed_games == 4
    assert [r['pair_seed'] for r in result['records']] == [491, 491, 492, 492]
    assert resumed.remaining_cost('test', 'one_ply', 'random', 4) == 0
    with pytest.raises(ValueError, match='Cached match identity'):
        resumed.match('test', 'one_ply', 'heuristic', 4)


def test_budget_does_not_start_another_batch(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path/'net.pt'
    save_checkpoint(checkpoint, AzulNet(hidden=32, arch='flat'), config={'num_players': 2})
    study = SearchStudy(tmp_path/'study', str(checkpoint), str(checkpoint), device='cpu',
                        seed=1, bot_workers=1, deadline=time.time()+119)
    def unexpected(*args, **kwargs):
        raise AssertionError('must not start a match near the deadline')
    monkeypatch.setattr(search_study, 'evaluate_match', unexpected)
    with pytest.raises(BudgetExpired):
        study.match('test', 'one_ply', 'random', 4)
    assert not list((tmp_path/'study/evaluations').glob('*/batch_*.json'))


def test_combining_reports_uses_game_counts_and_rejects_duplicate_pairs() -> None:
    records = [{'pair_seed': i//2+10, 'candidate_seat': i%2,
                'outcome': 'win' if i < 2 else 'loss', 'match_score': 1 if i < 2 else 0}
               for i in range(6)]
    def shard(rows):
        return {'records': rows, 'config': {}, 'wall_s': 1., 'value_calibration': {}}
    cfg = ArenaConfig(num_games=6, game_batch_size=4, seed=10)
    combined = combine_shards([shard(records[:4]), shard(records[4:])], cfg, {})
    assert combined['summary']['match_score'] == 1/3
    assert combined['complete']
    with pytest.raises(ValueError, match='ordered'):
        combine_shards([shard(records[:4]), shard(records[:2])], cfg, {})
    assert paired_difference(combined, combined)['difference'] == 0
    with pytest.raises(ValueError, match='identical'):
        paired_difference(combined, {'records': records[:-2]})


def test_search_manifest_rejects_changed_checkpoint_or_protocol(tmp_path: Path) -> None:
    checkpoint = tmp_path/'net.pt'
    save_checkpoint(checkpoint, AzulNet(hidden=32, arch='flat'), config={'num_players': 2})
    options = dict(device='cpu', seed=1, bot_workers=1, deadline=time.time()+1000)
    SearchStudy(tmp_path/'study', str(checkpoint), str(checkpoint), **options)
    with pytest.raises(ValueError, match='identity changed'):
        SearchStudy(tmp_path/'study', str(checkpoint), str(checkpoint), **{**options, 'seed': 2})
