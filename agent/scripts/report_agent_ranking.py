"""Report direct win shares and clustered uncertainty for a ranking campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.eval.astra_tournament import atomic_json, digest, interval
from agent.train import rating_display as D


def fit_counts(names: list[str], wins: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """Batched Newton fits for the same weak-prior BT likelihood as ranking.py.

    Parameters are log odds relative to random=1000. Counts retain half-credit
    ties. Positive prior curvature keeps all-success datasets finite.
    """
    if 'random' not in names or wins.shape != losses.shape or wins.ndim != 2:
        raise ValueError('Require an anchored participant list and matching 2D counts')
    free = [name for name in names if name != 'random']
    edges = list(combinations(names, 2))
    if wins.shape[1] != len(edges) or np.any(wins < 0) or np.any(losses < 0):
        raise ValueError('Invalid pair counts')
    design = np.array([[float(name == a) - float(name == b) for name in free] for a, b in edges])
    scale = np.log(10.) / 1000.
    prior_mean = (1500. - 1000.) * scale
    prior_precision = 1. / (10000. * scale) ** 2
    theta = np.full((len(wins), len(free)), prior_mean)
    totals = wins + losses
    def objective(values: np.ndarray) -> np.ndarray:
        logits = values @ design.T
        return (totals * np.logaddexp(0., logits) - wins * logits).sum(1) + .5 * prior_precision * ((values - prior_mean) ** 2).sum(1)
    for _ in range(100):
        probability = np.exp(-np.logaddexp(0., -(theta @ design.T)))
        gradient = (totals * probability - wins) @ design + prior_precision * (theta - prior_mean)
        curvature = totals * probability * (1. - probability)
        hessian = np.einsum('ep,be,eq->bpq', design, curvature, design)
        hessian += prior_precision * np.eye(len(free))[None]
        step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
        loss = objective(theta)
        rate = np.ones(len(theta))
        proposal = theta - step
        for _ in range(30):
            reject = objective(proposal) > loss + 1e-9
            if not reject.any():
                break
            rate[reject] *= .5
            proposal = theta - rate[:, None] * step
        delta = np.max(np.abs(proposal - theta))
        theta = proposal
        if delta < 1e-8:
            break
    else:
        raise RuntimeError('Bradley–Terry bootstrap optimizer did not converge')
    result = np.full((len(wins), len(names)), 1000.)
    for col, name in enumerate(names):
        if name != 'random':
            result[:, col] += theta[:, free.index(name)] / scale
    return result


def block_counts(records: list[dict[str, Any]], n: int) -> tuple[list[str], dict[str, np.ndarray]]:
    selected = [game for game in records if game['n'] == n]
    if any(not g['finished'] or g['failure'] for g in selected):
        raise ValueError('Resolve incomplete or failed games before fitting a ranking')
    names = sorted({name for game in selected for name in game['names']})
    edges = list(combinations(names, 2))
    index = {edge: i for i, edge in enumerate(edges)}
    blocks: dict[int, np.ndarray] = {}
    for game in selected:
        counts = blocks.setdefault(game['block'], np.zeros((2, len(edges))))
        for p, q in combinations(range(n), 2):
            a, b = game['names'][p], game['names'][q]
            if a == b:
                continue
            x, y = (game['scores'][p], game['rows'][p]), (game['scores'][q], game['rows'][q])
            if a > b:
                a, b, x, y = b, a, y, x
            score = float(x > y) + .5 * float(x == y)
            counts[:, index[a, b]] += [score, 1. - score]
    strata = {}
    for mixed in (False, True):
        values = [value for block, value in sorted(blocks.items()) if (block >= 100_000) == mixed]
        if values:
            strata['mixed' if mixed else 'homogeneous'] = np.stack(values)
    return names, strata


def ranking_uncertainty(records: list[dict[str, Any]], replicates: int = 2000,
                        seed: int = 984731) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError('At least 100 bootstrap replicates are required')
    result = {}
    for n in (2, 3, 4):
        if not any(game['n'] == n for game in records):
            continue
        names, strata = block_counts(records, n)
        total = sum(values.sum(0) for values in strata.values())
        point = fit_counts(names, total[0:1], total[1:2])[0]
        rng = np.random.default_rng(seed + n)
        samples = []
        for start in range(0, replicates, 100):
            count = min(100, replicates - start)
            drawn = np.zeros((count, *total.shape))
            for values in strata.values():
                weights = rng.multinomial(len(values), np.full(len(values), 1. / len(values)), size=count)
                drawn += np.einsum('bi,icd->bcd', weights, values)
            samples.append(fit_counts(names, drawn[:, 0], drawn[:, 1]))
        sampled = np.concatenate(samples)
        order = np.argsort(-point)
        table = []
        for col in order:
            # Equal fitted values share rank; no lexicographic tie advantage.
            ranks = 1 + (sampled > sampled[:, col:col+1] + 1e-8).sum(1)
            leaders = sampled >= sampled.max(1, keepdims=True) - 1e-8
            rank = 1 + int((point > point[col] + 1e-8).sum())
            table.append(dict(agent=names[col], rank=rank, rating=float(point[col]),
                              rating_ci95=np.quantile(sampled[:, col], [.025, .975]).tolist(),
                              rank_ci95=np.quantile(ranks, [.025, .975], method='inverted_cdf').tolist(),
                              first_place_bootstrap_share=float((leaders[:, col] / leaders.sum(1)).mean())))
        deltas = {}
        a = names.index('astra')
        for b, name in enumerate(names):
            if name != 'astra':
                interval = np.quantile(sampled[:, a] - sampled[:, b], [.025, .975]).tolist()
                deltas[name] = dict(rating_difference=float(point[a] - point[b]), ci95=interval,
                                    astra_higher=interval[0] > 1e-8)
        result[str(n)] = dict(table=table, astra_pairwise_rating_differences=deltas,
                              seed_blocks={key: len(values) for key, values in strata.items()})
    return dict(per_player_count=result, bootstrap_replicates=replicates, bootstrap_seed=seed,
                method='Stratified whole-seed-block bootstrap; all matchups/rotations sharing a draw seed resampled together; homogeneous and mixed strata retain their sizes',
                rating_model='BT pairwise score/row placements; ties=0.5; random=1000; scale=1000; Gaussian prior mean=1500 sigma=10000')


def mixed_tables(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Whole-game shares in each distinct-agent table, with seat-block intervals."""
    groups = defaultdict(list)
    for game in records:
        if game.get('field', '').startswith('mixed_'):
            groups[game['n'], tuple(sorted(game['names']))].append(game)
    tables = []
    for (n, names), games in sorted(groups.items()):
        by_agent = {}
        if any(not g['finished'] or g['failure'] for g in games):
            raise ValueError('Mixed tables contain an incomplete or failed game')
        for name in names:
            blocks = defaultdict(list)
            for game in games:
                seat = game['names'].index(name)
                blocks[game['block']].append(float(seat in game['winners']) / len(game['winners']))
            if any(len(values) != n for values in blocks.values()):
                raise ValueError('Mixed table is missing a seat rotation')
            block_means = [sum(values) / n for _, values in sorted(blocks.items())]
            by_agent[name] = {'win_share': sum(block_means) / len(block_means), 'ci95': interval(block_means)}
        tables.append({'players': n, 'participants': names, 'games': len(games), 'agents': by_agent})
    return tables


def display_ratings(uncertainty: dict[str, Any]) -> dict[str, Any]:
    """Preserve the raw fit and bootstrap evidence in a separate display artifact."""
    result = deepcopy(uncertainty)
    if 'rating_display' in result:
        if result['rating_display'] != D.metadata():
            raise ValueError('Unexpected rating display scale')
        return result
    result['rating_display'] = D.metadata()
    for n, values in result['per_player_count'].items():
        for row in values['table']:
            row['raw_rating'] = row['rating']
            row['raw_rating_ci95'] = row['rating_ci95']
            row['rating'] = D.to_display(row['raw_rating'], int(n))
            row['rating_ci95'] = [D.to_display(value, int(n)) for value in row['raw_rating_ci95']]
        for delta in values.get('astra_pairwise_rating_differences', {}).values():
            delta['raw_rating_difference'] = delta['rating_difference']
            delta['raw_ci95'] = delta['ci95']
            delta['rating_difference'] *= D.SCALES[int(n)]
            delta['ci95'] = [value * D.SCALES[int(n)] for value in delta['raw_ci95']]
    return result


def markdown(report: dict[str, Any], uncertainty: dict[str, Any], manifest: dict[str, Any]) -> str:
    uncertainty = display_ratings(uncertainty)
    lines = ['# Astra: expanded agent rankings', '',
             f"Completed **{report['total_games']:,} games** with {report['unfinished']} unfinished games and {report['failures']} failures.", '',
             'The production Astra configuration was frozen throughout. This campaign used fresh seeds and included matches between every eligible pair of agents. Neural agents participated only at their trained player count (two).', '',
             '## Astra win share against homogeneous opponents', '',
             '| Players | Opponent | Games | Astra win share | 95% block interval |',
             '| --- | --- | ---: | ---: | --- |']
    for key, group in sorted(report['matchups'].items()):
        n, name, opponent = key.split('/')
        if name == 'astra':
            low, high = group['ci95']
            lines.append(f"| {n} | {opponent.removeprefix('vs_')} | {group['games']:,} | {group['win_share']:.2%} | {low:.2%}–{high:.2%} |")
    lines += ['', 'Each game contains one Astra versus the remaining copies of the named opponent. A k-way shared win earns 1/k. All-success bootstrap intervals can collapse and do not establish perfect play.', '',
              '## Bradley–Terry ranking on the heuristic-2500 display scale', '',
              'Ratings use `1000 + multiplier × (raw − 1000)`, with frozen multipliers 0.3739824775 (2p), 0.3106665462 (3p), and 0.2949804033 (4p). This keeps random at 1000 and puts the reference heuristic at 2500. These are presentation changes: the original statistical fits, game outcomes, ordering, and win probabilities are unchanged.', '',
              'The ratings summarize pairwise final score/row placements across the complete field, including mixed tables. They are secondary to game win share. This common numerical scale does not establish equivalent human ability across player counts. See [rating scale v1](RATING_SCALE.md) for reference values and treatment of legacy leagues.', '']
    for n, values in uncertainty['per_player_count'].items():
        lines += ['', f'### {n} players', '', '| Rank | Agent | Rating | Rating 95% interval | Rank 95% interval | Bootstrap first-place share |', '| ---: | --- | ---: | --- | --- | ---: |']
        for row in values['table']:
            low, high = row['rating_ci95']
            ranklow, rankhigh = row['rank_ci95']
            lines.append(f"| {row['rank']} | {row['agent']} | {row['rating']:.0f} | {low:.0f}–{high:.0f} | {ranklow}–{rankhigh} | {row['first_place_bootstrap_share']:.1%} |")
    lines += ['', f"Ranking confidence intervals use {uncertainty['bootstrap_replicates']:,} stratified bootstrap resamples of whole seed blocks. The display transformation is applied to both endpoints with the same frozen multiplier; it is not refitted in each resample. Every matchup and seat rotation sharing a draw seed moves together. Bootstrap first-place share describes resampling stability, not a Bayesian probability of being the best agent. A weak Gaussian prior keeps perfect or nearly separated results finite; absolute rating levels can depend strongly on that prior.", '',
              '## Complete matchup matrix', '', 'Cells are the row agent’s win share against a table filled by the column agent. Three- and four-player cells in opposite directions are different compositions, so they need not sum to 100%.', '']
    for n in (2, 3, 4):
        names = sorted(name for name, spec in manifest['participants'].items() if n in spec['players'])
        lines += ['', f'### {n} players', '', '| Agent | ' + ' | '.join(names) + ' |', '| --- | ' + ' | '.join('---:' for _ in names) + ' |']
        for name in names:
            cells = ['—' if other == name else f"{report['matchups'][f'{n}p/{name}/vs_{other}']['win_share']:.1%}" for other in names]
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Mixed-table win shares', '',
              'Each row uses a distinct-agent table and all seat rotations. Percentages split shared victories and sum to 100% within the table. Parentheses show 95% block-bootstrap intervals.', '',
              '| Players | Participants | Games | Astra | Opus | Heuristic | Random |',
              '| --- | --- | ---: | --- | --- | --- | --- |']
    for table in uncertainty.get('mixed_tables', []):
        cells = []
        for name in ('astra', 'opus', 'heuristic', 'random'):
            value = table['agents'].get(name)
            cells.append('—' if value is None else f"{value['win_share']:.1%} ({value['ci95'][0]:.1%}–{value['ci95'][1]:.1%})")
        lines.append(f"| {table['players']} | {', '.join(table['participants'])} | {table['games']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Provenance and reproduction', '',
              'See [RANKING_PROTOCOL.md](RANKING_PROTOCOL.md) for the fixed schedule, outcome definitions, and resume commands. Raw games, frozen sources/native extension, checkpoint bytes, and hashes are in `agent/runs/astra-ranking/full/`. `ranking-uncertainty.json` preserves the original raw fit and bootstrap intervals. `ranking-display-v1.json` adds the display scale, converted intervals, and copies of the raw values. The original Astra tests and training leagues were preserved.', '',
              'Neural participants:', '']
    for name, spec in manifest['participants'].items():
        if spec['kind'] == 'checkpoint':
            lines.append(f"- `{name}`: `{spec['model_id']}`, checkpoint SHA-256 `{spec['sha256']}`. Its registered search profile uses {spec['search']['num_simulations']} simulations with a two-second deadline for this campaign.")
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--reuse-uncertainty', action='store_true',
                        help='Render the new scale from existing verified bootstrap evidence without refitting or overwriting it')
    args = parser.parse_args()
    report = json.loads((args.campaign / 'report.json').read_text())
    manifest = json.loads((args.campaign / 'manifest.json').read_text())
    if not report['complete'] or report['unfinished'] or report['failures']:
        raise ValueError('Campaign must finish without unexplained completion failures')
    records_hash = digest(args.campaign / 'games.jsonl')
    if args.reuse_uncertainty:
        uncertainty = json.loads((args.campaign / 'ranking-uncertainty.json').read_text())
        if uncertainty.get('records_sha256') != records_hash:
            raise ValueError('Stored bootstrap evidence does not match the game records')
    else:
        records = [json.loads(line) for line in (args.campaign / 'games.jsonl').read_text().splitlines()]
        uncertainty = ranking_uncertainty(records, args.replicates)
        uncertainty['mixed_tables'] = mixed_tables(records)
        uncertainty['records_sha256'] = records_hash
        uncertainty['analysis_source_sha256'] = digest(Path(__file__))
    # Independently check the Newton estimate against the existing Torch L-BFGS fit.
    for n, values in uncertainty['per_player_count'].items():
        previous = report['ratings']['per_player_count'][n]['ratings']
        for row in values['table']:
            if abs(row['rating'] - previous[row['agent']]) > 2.:
                raise ValueError('Independent BT point estimates disagree by more than two rating points')
    if not args.reuse_uncertainty:
        atomic_json(args.campaign / 'ranking-uncertainty.json', uncertainty)
    displayed = display_ratings(uncertainty)
    displayed['raw_uncertainty_sha256'] = digest(args.campaign / 'ranking-uncertainty.json')
    atomic_json(args.campaign / 'ranking-display-v1.json', displayed)
    if args.markdown:
        notes = ''
        if args.reuse_uncertainty and args.markdown.exists():
            existing = args.markdown.read_text()
            marker = '## Interpretation and limits'
            if marker in existing:
                notes = '\n' + marker + existing.split(marker, 1)[1]
        args.markdown.write_text(markdown(report, uncertainty, manifest) + notes)
    print(json.dumps(displayed['per_player_count'], indent=2))


if __name__ == '__main__':
    main()
