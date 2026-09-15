"""Fixed-weight search comparisons with paired, resumable batches and a time budget."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any

from agent.eval.arena import (ArenaConfig, checkpoint_hash, evaluate_match, paired_interval,
                              promotion_decision, summarize_games, write_report)
from agent.eval.builtin_opponents import BUILTIN_BOTS, builtin_identity
from agent.eval.latency import benchmark_latency
from agent.search.config import SearchConfig
from agent.train.reproducibility import provenance


class BudgetExpired(Exception):
    pass


def paired_difference(candidate: dict, reference: dict) -> dict:
    """Require exactly the same completed seed pairs before comparing scores."""
    left, right = candidate['records'], reference['records']
    keys = lambda rows: [(r['pair_seed'], r['candidate_seat']) for r in rows]
    if (not left or len(left) % 2 or keys(left) != keys(right)
            or any(r['outcome'] == 'unfinished' for r in left + right)):
        raise ValueError('Comparison requires identical finished seed pairs')
    a, b = summarize_games(left), summarize_games(right)
    return {'difference': a['match_score'] - b['match_score'],
            'ci95': paired_interval([x-y for x, y in zip(a['pair_scores'], b['pair_scores'], strict=True)])}


def combine_shards(shards: list[dict], config: ArenaConfig, identity: dict) -> dict:
    records = [row for shard in shards for row in shard['records']]
    expected = [(config.seed + i//2, i % 2) for i in range(len(records))]
    if [(r['pair_seed'], r['candidate_seat']) for r in records] != expected:
        raise ValueError('Shards do not form ordered seat-swapped seed pairs')
    return {**identity, 'schema_version': 2, 'config': asdict(config),
            'search_seed_protocol': 'independent_arena_shards_v1',
            'shard_configurations': [s['config'] for s in shards],
            'records': records, 'summary': summarize_games(records),
            'complete': len(records) == config.num_games,
            'wall_s': sum(s['wall_s'] for s in shards),
            'value_calibration_by_shard': [s['value_calibration'] for s in shards]}


class SearchStudy:
    def __init__(self, root: Path, candidate: str, champion: str, *, device: str, seed: int,
                 bot_workers: int, deadline: float, batch_size: int = 32) -> None:
        self.root, self.candidate, self.champion = root, candidate, champion
        self.device, self.seed, self.bot_workers = device, seed, bot_workers
        self.deadline, self.batch_size = deadline, batch_size
        self.base = SearchConfig()
        self.profiles = {'one_ply': self.base, **{f'tree_{n}': replace(self.base,
            backend='gumbel_tree', num_simulations=n) for n in (64, 256, 1024)}}
        self.opponents = {'astra': 'astra', 'champion': champion, 'opus': 'opus'}
        self.code = provenance()
        self.manifest = {'candidate': candidate, 'candidate_sha256': checkpoint_hash(candidate),
            'champion': champion, 'champion_sha256': checkpoint_hash(champion),
            'astra_identity': builtin_identity('astra'), 'profiles': {k: asdict(v) for k, v in self.profiles.items()},
            'seed': seed, 'inference_device': device, 'bot_workers': bot_workers,
            'batch_size': batch_size, 'provenance': self.code,
            'protocol': 'independent_arena_shards_v1', 'deadline': deadline,
            'development_games': 256, 'confirmation_games': 1024,
            'selection': 'Astra match score, then champion score, then lower simulation cost; confirmation subject to remaining time',
            'automatic_promotion': False}
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = root/'search_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != self.manifest:
            raise ValueError('Search identity changed; use a new study root')
        write_report(manifest_path, self.manifest)
        self.completed_games = sum(len(json.loads(p.read_text())['records'])
                                   for p in (root/'evaluations').glob('*/batch_*.json'))
        self.last_status = 0.0
        self.latency: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.pilots: dict[str, dict] = {}
        self.confirmation: dict[str, dict] = {}
        self.skipped: list[dict] = []
        self.rates: dict[tuple[str, str], float] = {}
        self.winner = 'one_ply'
        self.decision = {'promote': False, 'reason': 'confirmation has not completed'}
        if (root/'search.json').exists():
            previous = json.loads((root/'search.json').read_text())
            for name in ('latency', 'results', 'pilots', 'confirmation', 'skipped', 'winner', 'decision'):
                setattr(self, name, previous[name])

    def status(self, label: str, profile: str, **extra: Any) -> None:
        now = time.time()
        write_report(self.root/'status.json', {'stage': 'evaluation', 'label': label,
            'search_profile': profile, 'run_id': profile, 'updated_at': now,
            'completed_games_total': self.completed_games, 'unfinished': 0,
            'remaining_s': max(0, self.deadline-now), **extra})
        self.last_status = now

    def _guard(self) -> None:
        if time.time() >= self.deadline-120:
            raise BudgetExpired

    def match(self, label: str, profile: str, opponent: str, games: int, *,
              split: str = 'development', candidate: str | None = None) -> dict:
        cfg = ArenaConfig(num_games=games, game_batch_size=self.batch_size, seed=self.seed +
            (10_000_000 if split == 'confirmation' else 0), inference_device=self.device,
            bot_workers=self.bot_workers, search=self.profiles[profile], split=split)
        candidate = candidate or self.candidate
        identity = {'candidate': candidate, 'candidate_sha256': checkpoint_hash(candidate),
            'opponent': opponent, 'opponent_sha256': None if opponent in BUILTIN_BOTS else checkpoint_hash(opponent),
            'opponent_identity': builtin_identity(opponent), 'opponent_search': asdict(self.base),
            'provenance': self.code}
        job = self.root/'evaluations'/label
        job.mkdir(parents=True, exist_ok=True)
        expected_identity = {**identity, 'config': asdict(cfg)}
        if (job/'identity.json').exists() and json.loads((job/'identity.json').read_text()) != expected_identity:
            raise ValueError(f'Cached match identity changed: {label}')
        write_report(job/'identity.json', expected_identity)
        shards = []
        for start in range(0, games, self.batch_size):
            path = job/f'batch_{start:04d}.json'
            shard_cfg = replace(cfg, num_games=min(self.batch_size, games-start), seed=cfg.seed + start//2)
            if path.exists():
                report = json.loads(path.read_text())
                if report['config'] != asdict(shard_cfg) or any(report.get(k) != v for k, v in identity.items() if k != 'provenance'):
                    raise ValueError(f'Cached shard identity changed: {path}')
            else:
                self._guard()
                self.status(label, profile, completed_in_match=start, requested_games=games)
                def activity(row: dict) -> None:
                    self._guard()
                    if time.time()-self.last_status >= 10:
                        self.status(label, profile, completed_in_match=start, requested_games=games, **row)
                report = evaluate_match(candidate, opponent, shard_cfg, self.base, activity=activity)
                if report['candidate_sha256'] != identity['candidate_sha256'] or report['opponent_identity'] != identity['opponent_identity']:
                    raise ValueError('Opponent or checkpoint changed during evaluation')
                write_report(path, report)
                self.completed_games += len(report['records'])
            if report['summary']['unfinished']:
                raise RuntimeError(f'Unfinished games in {label}; retain the failed batch for inspection')
            shards.append(report)
            combined = combine_shards(shards, cfg, identity)
            write_report(job/'result.json', combined)
            self.status(label, profile, completed_in_match=len(combined['records']), requested_games=games)
        self.rates[(profile, opponent)] = combined['wall_s']/games
        print(json.dumps({'match': label, 'wall_s': combined['wall_s'],
            'summary': {k: v for k, v in combined['summary'].items() if k != 'pair_scores'}}), flush=True)
        return combined

    def estimate(self, profile: str, opponent: str, games: int) -> float:
        rate = self.rates.get((profile, opponent))
        if rate is None:
            rate = self.rates.get((profile, 'opus'), .05 if profile == 'one_ply' else 20.)
            if opponent == 'astra':
                rate += self.rates.get(('one_ply', 'astra'), .8)
        return 1.15 * games * rate + 10

    def fits(self, cost: float, reserve: float = 0) -> bool:
        return cost + reserve < self.deadline-time.time()-120

    def remaining_cost(self, label: str, profile: str, opponent: str, games: int) -> float:
        path = self.root/'evaluations'/label/'result.json'
        finished = len(json.loads(path.read_text())['records']) if path.exists() else 0
        return self.estimate(profile, opponent, games-finished) if finished < games else 0.0

    def save(self, complete: bool = False) -> dict:
        result = {'checkpoint': self.candidate, 'winner': self.winner,
            'search': asdict(self.profiles[self.winner]), 'decision': self.decision,
            'results': self.results, 'latency': self.latency, 'pilots': self.pilots,
            'confirmation': self.confirmation, 'skipped': self.skipped,
            'completed_games': self.completed_games, 'study_complete': complete,
            'automatic_promotion': False, 'manifest': str(self.root/'search_manifest.json')}
        write_report(self.root/'search.json', result)
        return result

    def run(self) -> dict:
        try:
            # Preflight results describe the same fixed profiles and weights. They
            # guide cost estimates only; pilot wins never enter strength selection.
            for profile, search in self.profiles.items():
                self._guard()
                self.status('latency', profile)
                path = self.root/'preflight'/f'{profile}_cpu_latency.json'
                if path.exists():
                    measured = json.loads(path.read_text())
                    if measured['checkpoint_sha256'] != self.manifest['candidate_sha256'] or measured['search'] != asdict(search):
                        raise ValueError('Preflight latency used different weights or search')
                else:
                    measured = benchmark_latency(self.candidate, search, device='cpu', games=2)
                    write_report(path, measured)
                self.latency[profile] = measured
                if profile != 'one_ply' and measured['qualified']:
                    pilot_path = self.root/'preflight'/f'{profile}_throughput.json'
                    if pilot_path.exists():
                        pilot = json.loads(pilot_path.read_text())
                        if pilot['candidate_sha256'] != self.manifest['candidate_sha256'] or pilot['config']['search'] != asdict(search):
                            raise ValueError('Preflight throughput used different weights or search')
                    else:
                        cfg = ArenaConfig(num_games=8, game_batch_size=self.batch_size,
                            seed=self.seed+50_000_000, inference_device=self.device, search=search)
                        pilot = evaluate_match(self.candidate, 'opus', cfg)
                        write_report(pilot_path, pilot)
                    self.pilots[profile] = {'wall_s': pilot['wall_s'], 'games': pilot['config']['num_games'],
                                          'report': str(pilot_path)}
                    self.rates[(profile, 'opus')] = pilot['wall_s']/pilot['config']['num_games']
                self.save()

            for profile in self.profiles:
                if not self.latency[profile]['qualified']:
                    self.skipped.append({'profile': profile, 'reason': 'failed five-second latency qualification'})
                    continue
                self.results[profile] = {'checkpoint': self.candidate, 'search': asdict(self.profiles[profile]), 'scores': {}}
                # Finish all three opponents for each affordable profile. Reserve
                # 35 minutes for held-out comparisons; do not stop on win rates.
                cost = sum(self.remaining_cost(f'dev_{profile}_{name}', profile, opponent, 256)
                           for name, opponent in self.opponents.items())
                if profile != 'one_ply' and not self.fits(cost, reserve=35*60):
                    self.skipped.append({'profile': profile, 'reason': 'full development screen exceeds time budget',
                                         'estimated_s': cost, 'remaining_s': self.deadline-time.time()})
                    continue
                for name, opponent in self.opponents.items():
                    report = self.match(f'dev_{profile}_{name}', profile, opponent, 256)
                    self.results[profile]['scores'][name] = report['summary']
                    self.save()

            eligible = [p for p, r in self.results.items() if p != 'one_ply' and len(r['scores']) == 3]
            eligible.sort(key=lambda p: (self.results[p]['scores']['astra']['match_score'],
                                        self.results[p]['scores']['champion']['match_score'],
                                        -self.profiles[p].num_simulations), reverse=True)
            self.decision = {'promote': False, 'reason': 'no completed tree screen'}
            if eligible:
                self.winner = eligible[0]
                # Prefer a full Astra comparison over an incomplete, expensive
                # confirmation. Keep the development leader visible in the report.
                self.confirmation['development_leader'] = self.winner
                fitting = [p for p in eligible if self.fits(
                    self.remaining_cost(f'confirm_{p}_astra', p, 'astra', 1024)
                    + self.remaining_cost(f'confirm_{p}_one_ply_astra', 'one_ply', 'astra', 1024))]
                if not fitting:
                    self.decision = {'promote': False, 'reason': 'insufficient remaining time for paired 1024-game Astra confirmation'}
                else:
                    selected = fitting[0]
                    self.confirmation['profile'] = selected
                    for label, profile, opponent, candidate in [
                        ('astra', selected, 'astra', self.candidate),
                        ('one_ply_astra', 'one_ply', 'astra', self.candidate),
                        ('champion', selected, self.champion, self.candidate),
                        ('opus', selected, 'opus', self.candidate),
                        ('champion_opus', 'one_ply', 'opus', self.champion)]:
                        if not self.fits(self.remaining_cost(f'confirm_{selected}_{label}', profile, opponent, 1024)):
                            self.skipped.append({'confirmation': label, 'reason': 'insufficient remaining time'})
                            break
                        report = self.match(f'confirm_{selected}_{label}', profile, opponent, 1024,
                                            split='confirmation', candidate=candidate)
                        self.confirmation[label] = report['summary']
                        self.save()
                    self.decision = {'promote': False, 'reason': 'full confirmation suite incomplete'}
                    directory = self.root/'evaluations'
                    def read(label: str) -> dict:
                        return json.loads((directory/f'confirm_{selected}_{label}'/'result.json').read_text())
                    if all(k in self.confirmation for k in ('astra', 'one_ply_astra')):
                        delta = paired_difference(read('astra'), read('one_ply_astra'))
                        self.confirmation['astra_search_gain'] = delta
                    if all(k in self.confirmation for k in ('champion', 'opus', 'champion_opus', 'astra', 'one_ply_astra')):
                        self.decision = promotion_decision(read('champion'), read('opus'), read('champion_opus'))
                        if self.confirmation['astra_search_gain']['ci95'][0] <= 0:
                            self.decision.update(promote=False, reason='Astra search gain not established')
                        self.winner = selected
            return self.save(complete=True)
        except BudgetExpired:
            self.decision = {'promote': False, 'reason': 'time budget reached; completed batches retained'}
            return self.save(complete=True)


def run_study(root: Path, candidate: str, champion: str, *, device: str, seed: int,
              bot_workers: int, deadline: float) -> dict:
    return SearchStudy(root, candidate, champion, device=device, seed=seed,
                       bot_workers=bot_workers, deadline=deadline).run()
