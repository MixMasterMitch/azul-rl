"""The frozen display scale changes units without changing statistical evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from agent.train import rating_display as D
from agent.train import ranking as R
from agent.scripts.league_table import display_entry
from agent.scripts import report_agent_ranking as report_script


@pytest.mark.parametrize('players,expected', [(2, .3739824775308096), (3, .310666546179065), (4, .29498040327540787)])
def test_frozen_endpoints_and_invertible_monotone_transform(players: int, expected: float) -> None:
    assert D.SCALES[players] == pytest.approx(expected, abs=1e-12)
    assert D.to_display(1000, players) == 1000
    assert D.to_display(D.CAMPAIGN_HEURISTIC[players], players) == pytest.approx(2500)
    raw = [-500., 999., 1000., 2500., 5000., 8000.]
    values = [D.to_display(value, players) for value in raw]
    assert values == sorted(values)
    assert [D.from_display(value, players) for value in values] == pytest.approx(raw)
    a, b = torch.tensor(5000.), torch.tensor(4200.)
    probability = R.expected_score(a, b)
    restored = [torch.tensor(D.from_display(D.to_display(float(x), players), players)) for x in (a, b)]
    assert R.expected_score(*restored) == probability


@pytest.mark.parametrize('value,players,scales', [(float('nan'), 2, None), (float('inf'), 3, None),
    (2000, 1, None), (2000, 4, {4: -1}), (2000, 4, {2: .5})])
def test_invalid_display_inputs(value: float, players: int, scales: dict | None) -> None:
    with pytest.raises(ValueError):
        D.to_display(value, players, scales)


def test_reference_basis_and_legacy_inverse_do_not_double_scale() -> None:
    references = R.DEFAULT_REFERENCE_ANCHORS_PER_PC
    original = deepcopy(references)
    for n in D.PLAYER_COUNTS:
        heuristic = references[n]['heuristic']
        assert D.to_display(heuristic, n, D.scales_for(references)) == pytest.approx(2500)
        old = R.calibrate_rating(heuristic, n, R.calibration_scales_for(references))
        assert D.from_legacy_calibrated(old, n, references) == pytest.approx(2500)
    assert references == original
    with pytest.raises(ValueError, match='heuristic'):
        D.scales_for({2: {'random': 1000}})
    with pytest.raises(ValueError):
        D.scales_for({2: {'heuristic': 1000}})
    with pytest.raises(ValueError, match='random'):
        D.scales_for({2: {'random': 900, 'heuristic': 3000}})
    assert D.scales_for(json.loads(json.dumps(references))) == D.scales_for(references)


def uncertainty() -> dict:
    return {'per_player_count': {'2': {'table': [
        {'agent': 'heuristic', 'rank': 1, 'rating': D.CAMPAIGN_HEURISTIC[2],
         'rating_ci95': [4900., 5100.], 'rank_ci95': [1, 1], 'first_place_bootstrap_share': 1},
        {'agent': 'random', 'rank': 2, 'rating': 1000., 'rating_ci95': [1000., 1000.],
         'rank_ci95': [2, 2], 'first_place_bootstrap_share': 0}],
        'astra_pairwise_rating_differences': {'opus': {'rating_difference': 200., 'ci95': [100., 300.], 'astra_higher': True}}
    }}, 'bootstrap_replicates': 2000, 'mixed_tables': [{'win_share': .5}]}


def test_report_preserves_raw_fit_rank_shares_and_scales_differences_without_offset() -> None:
    raw = uncertainty()
    before = deepcopy(raw)
    displayed = report_script.display_ratings(raw)
    row = displayed['per_player_count']['2']['table'][0]
    assert raw == before
    assert row['raw_rating'] == D.CAMPAIGN_HEURISTIC[2]
    assert row['rating'] == pytest.approx(2500)
    assert row['rating_ci95'] == pytest.approx([D.to_display(4900, 2), D.to_display(5100, 2)])
    delta = displayed['per_player_count']['2']['astra_pairwise_rating_differences']['opus']
    assert delta['rating_difference'] == pytest.approx(200 * D.SCALES[2])
    assert delta['ci95'] == pytest.approx([100 * D.SCALES[2], 300 * D.SCALES[2]])
    assert delta['astra_higher'] is True
    assert row['rank'] == 1 and row['rank_ci95'] == [1, 1]
    assert displayed['mixed_tables'] == raw['mixed_tables']
    assert report_script.display_ratings(displayed) == displayed


def test_reuse_command_preserves_original_artifacts_and_checks_record_hash(tmp_path: Path, monkeypatch) -> None:
    records = tmp_path / 'games.jsonl'
    records.write_text('{}\n')
    raw = uncertainty()
    raw['records_sha256'] = report_script.digest(records)
    (tmp_path / 'ranking-uncertainty.json').write_text(json.dumps(raw))
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'report.json').write_text(json.dumps({'complete': True, 'unfinished': 0, 'failures': 0,
        'ratings': {'per_player_count': {'2': {'ratings': {'heuristic': D.CAMPAIGN_HEURISTIC[2], 'random': 1000}}}}}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(sys, 'argv', ['report_agent_ranking', str(tmp_path), '--reuse-uncertainty'])
    report_script.main()
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    display = json.loads((tmp_path / 'ranking-display-v1.json').read_text())
    assert display['raw_uncertainty_sha256'] == report_script.digest(tmp_path / 'ranking-uncertainty.json')
    assert display['per_player_count']['2']['table'][0]['rating'] == 2500
    records.write_text('changed\n')
    with pytest.raises(ValueError, match='does not match'):
        report_script.main()


def test_league_display_uses_per_format_values_and_weights_without_mutation() -> None:
    references = R.DEFAULT_REFERENCE_ANCHORS_PER_PC
    entry = {f'rating_{n}p': R.calibrate_rating(references[n]['heuristic'], n) for n in D.PLAYER_COUNTS}
    entry['rating'] = 99999  # The old combined number is not a raw per-format rating.
    before = deepcopy(entry)
    displayed = display_entry(entry, references, {2: 20, 3: 1, 4: 30})
    assert entry == before
    assert [displayed[f'rating_{n}p'] for n in D.PLAYER_COUNTS] == pytest.approx([2500] * 3)
    assert displayed['rating'] == pytest.approx(2500)
    assert display_entry({'rating': 1234}, references, {})['rating'] is None
