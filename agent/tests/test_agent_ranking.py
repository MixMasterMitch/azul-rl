from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import numpy as np

from agent.eval.agent_ranking import BUILTINS, SEED_BASE, load_records, schedule, summarize
from agent.eval.astra_tournament import play_game
from agent.eval.heuristic_astra import AstraConfig


def participants() -> dict:
    return {name: {'players': [2, 3, 4]} for name in BUILTINS} | {
        name: {'players': [2]} for name in ('league_2p', 'trained_2p')}


def test_round_robin_counts_supported_players_and_balanced_rotations() -> None:
    tasks = schedule(participants())
    assert len(tasks) == 19456
    assert len({t['key'] for t in tasks}) == len(tasks)
    assert all(t['engine_seed'] >= SEED_BASE for t in tasks)
    groups = defaultdict(list)
    for task in tasks:
        assert task['names'][task['seat']] == task['variant']
        assert all(task['n'] in participants()[name]['players'] for name in task['names'])
        groups[task['n'], task['field'], task['block']].append(task)
    for (n, _, _), rotations in groups.items():
        assert len(rotations) == n
        assert len({t['engine_seed'] for t in rotations}) == 1
        assert {t['seat'] for t in rotations} == set(range(n))
        assert all(Counter(t['names']) == Counter(rotations[0]['names']) for t in rotations)
        for name in set(rotations[0]['names']):
            assert len({sum(t['names'][seat] == name for t in rotations) for seat in range(n)}) == 1
    pairs = {tuple(sorted(t['names'])) for t in tasks if t['n'] == 2}
    assert len(pairs) == 15


def test_resume_checks_metadata_duplicates_and_partial_append(tmp_path: Path) -> None:
    tasks = schedule(participants(), 1, 1, 1)
    path = tmp_path / 'games.jsonl'
    path.write_text(json.dumps(tasks[0]) + '\n{"partial')
    assert load_records(path, tasks) == [tasks[0]]
    assert path.read_bytes().endswith(b'\n')
    path.write_text(json.dumps(tasks[0]) + '\n' + json.dumps(tasks[0]) + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_records(path, tasks)
    changed = dict(tasks[0], bot_seed=0)
    path.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='seeds'):
        load_records(path, tasks)


def game_record(n: int, seat: int, block: int, winner_seats: list[int]) -> dict:
    names = ['opus'] * n
    names[seat] = 'astra'
    return dict(n=n, block=block, seat=seat, names=names, finished=True, failure=None,
                winners=winner_seats, scores=[10] * n, rows=[1] * n)


def test_win_shares_ties_unfinished_and_reverse_two_player_results() -> None:
    games = [game_record(2, seat, block, [seat]) for block in range(4) for seat in range(2)]
    result = summarize(games)
    assert result['matchups']['2p/astra/vs_opus']['win_share'] == 1
    assert result['matchups']['2p/opus/vs_astra']['win_share'] == 0
    shared = [game_record(3, seat, block, [0, 1, 2]) for block in range(4) for seat in range(3)]
    result = summarize(shared)
    assert result['matchups']['3p/astra/vs_opus']['win_share'] == pytest.approx(1 / 3)
    assert result['matchups']['3p/astra/vs_opus']['complete_blocks'] == 4
    shared[0].update(finished=False, winners=[])
    result = summarize(shared)
    assert result['unfinished'] == 1
    assert result['matchups']['3p/astra/vs_opus']['ci95'] == [0, 1]
    assert result['matchups']['3p/astra/vs_opus']['complete_blocks'] == 3
    assert result['ratings']['unfinished_or_failed_games_skipped'] == 1


def test_custom_bot_factory_uses_authoritative_legality_and_turn_cap() -> None:
    calls = []
    class LegalBot:
        def select_action(self, engine: object, game_idx: int) -> int:
            return int(engine.legal_action_mask()[game_idx].nonzero()[0])
    def factory(name: str, *args: object) -> LegalBot:
        calls.append(name)
        return LegalBot()
    task = schedule(participants(), 1, 1, 1)[0]
    task.update(config=asdict(AstraConfig(depth=0)), max_turns=1)
    result = play_game(task, bot_factory=factory)
    assert calls == task['names']
    assert result['turns'] == 1
    assert not result['finished']
    assert result['win_share'] is None
    assert result['failure'] is None


def test_batched_bt_fit_matches_independent_rating_solver() -> None:
    from agent.scripts.report_agent_ranking import fit_counts
    from agent.train.ranking import fit_ratings_for_pc
    names = ['astra', 'opus', 'random']
    wins = np.array([[80., 99., 90.], [100., 100., 100.]])
    losses = 100 - wins
    actual = fit_counts(names, wins, losses)
    for index in range(2):
        pairs = [('astra', 'opus'), ('astra', 'random'), ('opus', 'random')]
        records = [dict(a=a, b=b, wins_a_2p=int(wins[index, j]), wins_b_2p=int(losses[index, j]))
                   for j, (a, b) in enumerate(pairs)]
        expected = fit_ratings_for_pc(records, 2, anchors={'random': 1000.},
                                      use_reference_anchors=False, prior_sigma=10000.)
        for col, name in enumerate(names):
            assert actual[index, col] == pytest.approx(expected[name], abs=2.)
        assert actual[index, 0] > actual[index, 1] > actual[index, 2]


def test_bootstrap_preserves_correlated_tables_and_tied_placements() -> None:
    from agent.scripts.report_agent_ranking import block_counts, ranking_uncertainty
    records = []
    for block in range(8):
        for seat in range(2):
            record = game_record(2, seat, block, [seat])
            record.update(names=['astra', 'random'], scores=[20, 10])
            records.append(record)
            records.append(dict(record, names=['opus', 'random'], scores=[10, 10]))
            records.append(dict(record, names=['astra', 'opus'], scores=[20, 10]))
    names, strata = block_counts(records, 2)
    assert names == ['astra', 'opus', 'random']
    assert strata['homogeneous'].shape == (8, 2, 3)
    assert strata['homogeneous'].sum() == 48
    summary = ranking_uncertainty(records, 100)
    assert summary == ranking_uncertainty(list(reversed(records)), 100)
    assert summary['per_player_count']['2']['table'][0]['agent'] == 'astra'
    assert summary['per_player_count']['2']['table'][0]['first_place_bootstrap_share'] == 1
    records[0]['finished'] = False
    with pytest.raises(ValueError, match='incomplete'):
        ranking_uncertainty(records, 100)


def test_mixed_table_shares_sum_to_one_with_shared_winners() -> None:
    from agent.scripts.report_agent_ranking import mixed_tables
    records = []
    table = ['astra', 'opus', 'random']
    for block in range(4):
        for seat in range(3):
            names = table[-seat:] + table[:-seat] if seat else table[:]
            records.append(dict(n=3, names=names, block=block, finished=True,
                                field='mixed_astra_opus_random', failure=None,
                                winners=[names.index('astra'), names.index('opus')]))
    result = mixed_tables(records)
    assert len(result) == 1
    assert result[0]['games'] == 12
    assert result[0]['agents']['astra']['win_share'] == .5
    assert result[0]['agents']['opus']['win_share'] == .5
    assert result[0]['agents']['random']['win_share'] == 0
    assert sum(x['win_share'] for x in result[0]['agents'].values()) == 1
    with pytest.raises(ValueError, match='seat rotation'):
        mixed_tables(records[:-1])
