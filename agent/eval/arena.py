"""Paired two-player evaluation with immutable, reproducible game records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from ..env.engine import BatchedEngine, SHARED_VICTORY
from ..net.encoder import encode_state
from ..search.config import SearchConfig
from ..search.gumbel_mcts import gumbel_root_act
from ..train.checkpointing import load_net_from_checkpoint
from ..train.reproducibility import provenance
from .builtin_opponents import BUILTIN_BOTS, builtin_identity
from .inference import InferenceModel


@dataclass(frozen=True)
class ArenaConfig:
    num_games: int = 256
    seed: int = 20260913
    max_turns: int = 300
    inference_device: str = 'cpu'
    game_batch_size: int = 128
    bot_workers: int = 1
    search: SearchConfig = field(default_factory=SearchConfig)
    greedy: bool = False
    split: str = 'development'

    def __post_init__(self) -> None:
        if self.num_games < 2 or self.num_games % 2:
            raise ValueError('Evaluation requires an even positive number of paired games')
        if self.game_batch_size < 2 or self.game_batch_size % 2:
            raise ValueError('game_batch_size must be even and at least two')
        if type(self.bot_workers) is not int or not 1 <= self.bot_workers <= 8:
            raise ValueError('bot_workers must be an integer from one to eight')
        if self.max_turns < 1 or self.split not in {'development', 'confirmation', 'replication'}:
            raise ValueError('Invalid arena configuration')


def checkpoint_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def paired_interval(pair_scores: list[float], seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over independent seed pairs (never individual seats)."""
    if not pair_scores:
        return (0.0, 1.0)
    values = np.asarray(pair_scores, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([values[rng.integers(0, len(values), (500, len(values)))].mean(1)
                            for _ in range(20)])
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize_games(games: list[dict]) -> dict:
    finished = [g for g in games if g['outcome'] != 'unfinished']
    wins = sum(g['outcome'] == 'win' for g in finished)
    losses = sum(g['outcome'] == 'loss' for g in finished)
    draws = sum(g['outcome'] == 'shared' for g in finished)
    pairs: dict[int, list[float]] = {}
    for g in finished:
        pairs.setdefault(g['pair_seed'], []).append(g['match_score'])
    pair_scores = [sum(v) / 2 for v in pairs.values() if len(v) == 2]
    return {'games': len(games), 'wins': wins, 'losses': losses, 'shared_victories': draws,
            'unfinished': len(games) - len(finished), 'sole_winrate': wins / max(len(finished), 1),
            'match_score': (wins + 0.5 * draws) / max(len(finished), 1),
            'match_score_ci95': paired_interval(pair_scores), 'pair_scores': pair_scores}


def _net_policy(model: InferenceModel, engine: BatchedEngine, search: SearchConfig, greedy: bool) -> torch.Tensor:
    with torch.inference_mode():
        if greedy:
            g, s = encode_state(engine)
            return model(g, s, engine.legal_action_mask(), 2)[0].argmax(1)
        if search.move_deadline_s is not None and engine.batch_size > 1:
            # A move deadline belongs to one game, never to a whole arena batch.
            return torch.cat([_net_policy(model, engine.index_select(torch.tensor([b])),
                replace(search, seed=None if search.seed is None else search.seed + b), False)
                for b in range(engine.batch_size)])
        return gumbel_root_act(engine, model, search_config=search)[0]


def _builtin_actions(bots: list, engine: BatchedEngine, indices: list[int], workers: int) -> torch.Tensor:
    # Each game owns its bot and RNG. Wait for every action before mutating the
    # shared engine; map preserves seat ordering. Astra releases the GIL in Rust.
    def select(item: tuple[int, int]) -> int:
        local, global_index = item
        return bots[global_index].select_action(engine, local)
    if workers == 1:
        return torch.tensor([select(item) for item in enumerate(indices)])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return torch.tensor(list(pool.map(select, enumerate(indices))))


def evaluate_match(candidate: str, opponent: str, config: ArenaConfig,
                   opponent_search: SearchConfig | None = None,
                   progress: Callable[[dict], None] | None = None,
                   activity: Callable[[dict], None] | None = None) -> dict:
    """Evaluate frozen weights. Every pair has identical initial draw streams and swapped seats."""
    started = time.monotonic()
    net, payload = load_net_from_checkpoint(candidate, 'cpu')
    model = InferenceModel(net, config.inference_device)
    opponent_model = None
    factories = BUILTIN_BOTS
    opponent_identity = builtin_identity(opponent)
    if opponent not in factories:
        opponent_net, _ = load_net_from_checkpoint(opponent, 'cpu')
        opponent_model = InferenceModel(opponent_net, config.inference_device)
    games: list[dict] = []
    latencies: list[float] = []
    value_pairs: list[tuple[float, float]] = []
    search_opponent = opponent_search or config.search
    for start in range(0, config.num_games, config.game_batch_size):
        n = min(config.game_batch_size, config.num_games - start)
        seeds = [config.seed + (start + b) // 2 for b in range(n)]
        seats = torch.tensor([(start + b) % 2 for b in range(n)])
        engine = BatchedEngine(n, 2, 'cpu', seed=config.seed, game_seeds=seeds)
        bots = [factories[opponent](seed=s + 17) for s in seeds] if opponent_model is None else []
        turns = torch.zeros(n, dtype=torch.long)
        predictions: list[list[float]] = [[] for _ in range(n)]
        for turn in range(config.max_turns):
            if engine.ended.all():
                break
            alive = ~engine.ended
            active_candidate = alive & (engine.current_player.long() == seats)
            actions = torch.zeros(n, dtype=torch.long)
            for candidate_turn, idx in [(True, active_candidate.nonzero().flatten()),
                                        (False, (alive & ~active_candidate).nonzero().flatten())]:
                if not len(idx):
                    continue
                sub = engine.index_select(idx)
                if candidate_turn or opponent_model is not None:
                    chosen_model = model if candidate_turn else opponent_model
                    chosen_search = config.search if candidate_turn else search_opponent
                    chosen_search = replace(chosen_search, seed=config.seed * 100003 + start * 1009 + turn * 2 + int(candidate_turn))
                    if candidate_turn:
                        g, s = encode_state(sub)
                        predicted = model.forward_value(g, s, 2)[:, 0].tolist()
                        for b, value in zip(idx.tolist(), predicted):
                            predictions[b].append(value)
                    t = time.monotonic()
                    selected = _net_policy(chosen_model, sub, chosen_search, config.greedy if candidate_turn else False)
                    latencies.append(time.monotonic() - t)
                else:
                    selected = _builtin_actions(bots, sub, idx.tolist(), config.bot_workers)
                if not sub.legal_action_mask().gather(1, selected[:, None]).all():
                    raise RuntimeError('Evaluation produced an illegal action')
                actions[idx] = selected
            engine.step(actions)
            turns[alive] += 1
            if activity:
                activity({'batch_start': start, 'turn': turn + 1,
                          'finished_in_batch': int(engine.ended.sum()), 'completed': len(games)})
        winners = engine.get_winners()
        for b in range(n):
            if not engine.ended[b]:
                outcome, score = 'unfinished', None
            elif int(winners[b]) == SHARED_VICTORY:
                outcome, score = 'shared', 0.5
            elif int(winners[b]) == int(seats[b]):
                outcome, score = 'win', 1.0
            else:
                outcome, score = 'loss', 0.0
            if outcome != 'unfinished':
                utility = -1.0 if outcome == 'loss' else 1.0
                value_pairs.extend((v, utility) for v in predictions[b])
            games.append({'pair_seed': seeds[b], 'candidate_seat': int(seats[b]),
                          'outcome': outcome, 'match_score': score, 'turns': int(turns[b]),
                          'scores': engine.scores[b, :2].tolist()})
        if progress:
            progress({'completed': len(games), 'total': config.num_games, **summarize_games(games)})
    result = {'schema_version': 1, 'candidate': str(Path(candidate).resolve()),
              'candidate_sha256': checkpoint_hash(candidate), 'opponent': opponent,
              'opponent_sha256': checkpoint_hash(opponent) if opponent_model is not None else None,
              'opponent_identity': opponent_identity,
              'config': asdict(config), 'opponent_search': asdict(search_opponent),
              'model_spec': {'arch': net.arch, 'hidden': net.hidden, 'model_version': net.model_version,
                             'encoder_version': payload.get('encoder_version', 3)},
              'summary': summarize_games(games), 'records': games,
              'value_calibration': value_calibration(value_pairs),
              'wall_s': time.monotonic() - started, 'provenance': provenance(),
              'inference_batch_latency_p95_s': float(np.quantile(latencies, .95)) if latencies else 0.0}
    return result


def value_calibration(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"samples": 0}
    values = np.asarray(pairs)
    prediction, outcome = values[:, 0], values[:, 1]
    bins = []
    for low, high in zip(np.linspace(-1, 1, 11)[:-1], np.linspace(-1, 1, 11)[1:]):
        mask = (prediction >= low) & ((prediction < high) if high < 1 else (prediction <= high))
        if mask.any():
            bins.append({"n": int(mask.sum()), "predicted": float(prediction[mask].mean()),
                         "observed": float(outcome[mask].mean())})
    return {"samples": len(pairs), "mse": float(((prediction-outcome)**2).mean()),
            "bias": float((prediction-outcome).mean()),
            "sign_accuracy": float(((prediction >= 0) == (outcome >= 0)).mean()), "bins": bins}


def promotion_decision(champion_match: dict, candidate_opus: dict, champion_opus: dict) -> dict:
    reports = (champion_match, candidate_opus, champion_opus)
    # Compare paired per-seed scores; never zip unlike protocols or seed sets.
    cfg_keys = ('num_games', 'seed', 'max_turns', 'game_batch_size', 'inference_device', 'split')
    if any(any(r['config'][k] != reports[0]['config'][k] for k in cfg_keys) for r in reports):
        raise ValueError('Promotion reports must use the same paired protocol')
    if any(r['config']['split'] not in {'confirmation', 'replication'} or r['config']['num_games'] < 1024 for r in reports):
        raise ValueError('Promotion requires at least 1024 held-out games per opponent')
    if (champion_match['candidate_sha256'] != candidate_opus['candidate_sha256']
            or champion_match['opponent_sha256'] != champion_opus['candidate_sha256']
            or candidate_opus['opponent'] != 'opus' or champion_opus['opponent'] != 'opus'):
        raise ValueError('Promotion reports refer to different candidates or champions')
    if (champion_match['config']['search'] != candidate_opus['config']['search']
            or champion_match['opponent_search'] != champion_opus['config']['search']
            or any(r['config']['greedy'] for r in reports)):
        raise ValueError('Promotion search profiles do not match')
    for report in reports:
        cfg = report['config']
        expected = [(cfg['seed'] + i//2, i % 2) for i in range(cfg['num_games'])]
        actual = [(g['pair_seed'], g['candidate_seat']) for g in report['records']]
        if actual != expected:
            raise ValueError('Promotion reports must contain every ordered, seat-swapped seed pair')
    summaries = [summarize_games(r['records']) for r in reports]
    if any(s['unfinished'] for s in summaries):
        return {'promote': False, 'reason': 'unfinished games'}
    s = summaries[0]
    candidate_pairs = summaries[1]['pair_scores']
    baseline_pairs = summaries[2]['pair_scores']
    delta = [a - b for a, b in zip(candidate_pairs, baseline_pairs, strict=True)]
    delta_ci = paired_interval(delta)
    passed = s['match_score'] >= .55 and s['match_score_ci95'][0] > .5 and delta_ci[0] >= -.03
    return {'promote': passed, 'champion_match_score': s['match_score'],
            'champion_ci95': s['match_score_ci95'], 'opus_delta_ci95': delta_ci,
            'reason': 'passed' if passed else 'strength or non-regression threshold not met'}


def write_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
