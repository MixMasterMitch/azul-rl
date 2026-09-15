"""Staged, resumable local experiments. Each training stage has an eight-hour ceiling."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
from typing import Any

import torch

from agent.eval.arena import ArenaConfig, checkpoint_hash, evaluate_match, promotion_decision, write_report
from agent.eval.builtin_opponents import BUILTIN_BOTS, builtin_identity
from agent.eval.latency import benchmark_latency
from agent.obs.run import Run
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint, load_checkpoint_payload, save_checkpoint
from agent.train.device import configure_device, resolve_device
from agent.train.league import League
from agent.train.loop import LoopConfig, run_loop
from agent.train.model_registry import ModelRegistry
from agent.train.reproducibility import provenance, require_disk_space

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / 'agent/runs/competitive'
INITIALIZERS = {
    'v3_latest': REPO / 'agent/runs/attn_256_v3/checkpoints/iter_005700.pt',
    'v4_latest': REPO / 'agent/runs/attn_256_v4/checkpoints/iter_002350.pt',
    'v4_rating_peak': REPO / 'agent/runs/league/ckpt_00294_i1350.pt',
}
BASE_SEARCH = SearchConfig()


def _stored_training_config(config: dict) -> dict:
    """Experiments frozen before Astra evaluation keep their original mix."""
    return {'eval_astra_fraction': 0.0, **config}


def training_config(root: Path, run_id: str, initializer: str, *, arch: str = 'attn',
                    seed: int = 20260913, minutes: float = 235., learner_steps: int = 72,
                    backend: str = 'one_ply', device: str = 'cuda') -> LoopConfig:
    astra_fraction = LoopConfig().eval_astra_fraction
    directory = root / 'experiments' / run_id
    for name in ('experiment_budget.json', 'experiment_complete.json'):
        path = directory / name
        if path.exists():
            astra_fraction = _stored_training_config(json.loads(path.read_text())['config'])['eval_astra_fraction']
            break
    return LoopConfig(run_id=run_id, runs_root=str(root / 'experiments'), init_from=initializer,
        num_players=2, device=device, hidden=256, arch=arch,
        seed=seed, max_iters=1_000_000, max_wall_minutes=min(minutes, 480),
        selfplay_games=1023, selfplay_sims=64 if backend == 'gumbel_tree' else 32,
        selfplay_turns_per_player=60, replay_capacity=1_000_000,
        learner_batch=256, learner_steps_per_iter=learner_steps,
        entropy_bonus=.014274964936918592, lr=.0009217470483295072,
        weight_decay=.0000896240535348826, dirichlet_alpha=.25668356196635866,
        dirichlet_mix=.5332343378344558, q_scale=26.747517709474458,
        time_discount=.999444440976462, reward_mode='binary',
        training_cycle_length=4, bot_selfplay_opus_prob=.36801279738712916,
        bot_selfplay_astra_prob=.0625, bot_selfplay_workers=8,
        league_selfplay_every=0, checkpoint_every=50, keep_recent_checkpoints=2,
        eval_games=256, eval_sims=64, eval_device=device, eval_q_scale=28, eval_astra_fraction=astra_fraction,
        eval_temperature=.25, eval_workers=1, search_backend=backend,
        league_root=str(root / 'experiments' / run_id / 'league'),
        profile_training=False, use_amp=False)


def tree_distill_config(root: Path, run_id: str, initializer: str, *, seed: int,
                        minutes: float = 450., device: str = 'cuda') -> LoopConfig:
    """Distil the qualified 64-simulation tree policy into a fresh learner run."""
    cfg = training_config(root, run_id, initializer, arch='source_attn', seed=seed,
                          minutes=minutes, learner_steps=72, backend='gumbel_tree', device=device)
    return replace(cfg, selfplay_games=256, selfplay_sims=64, learner_steps_per_iter=72,
                   checkpoint_every=25, keep_recent_checkpoints=2, eval_games=0,
                   eval_search_backend='gumbel_tree', profile_every=25)


class Campaign:
    def __init__(self, root: Path, device: str, seed: int, bot_workers: int = 1) -> None:
        self.root, self.device, self.seed = root.resolve(), device, seed
        self.bot_workers = bot_workers
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / 'status.json'
        self.started = time.monotonic()
        self.deadline = self.started + 480 * 60
        self.code = provenance()

    def status(self, stage: str, **fields: Any) -> None:
        record = {'stage': stage, 'updated_at': time.time(), 'elapsed_s': time.monotonic()-self.started, **fields}
        write_report(self.status_path, record)
        print(json.dumps(record), flush=True)

    def _read(self, name: str) -> dict:
        return json.loads((self.root / name).read_text())

    def _match(self, label: str, candidate: str, opponent: str, cfg: ArenaConfig,
               opponent_search: SearchConfig | None = None) -> dict:
        opponent_hash = None if opponent in BUILTIN_BOTS else checkpoint_hash(opponent)
        opponent_identity = builtin_identity(opponent)
        identity = {'candidate': checkpoint_hash(candidate), 'opponent': opponent,
                    'opponent_hash': opponent_hash, 'opponent_identity': opponent_identity,
                    'config': asdict(cfg), 'opponent_search': asdict(opponent_search or cfg.search), 'code': self.code}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        path = self.root / 'evaluations' / f'{label}_{fingerprint}.json'
        if time.monotonic() > self.deadline:
            raise TimeoutError('Overnight budget exhausted before evaluation; rerun the stage')
        if path.exists():
            old = json.loads(path.read_text())
            if (old['candidate_sha256'] == checkpoint_hash(candidate) and old['config'] == asdict(cfg)
                    and old['opponent'] == opponent
                    and old.get('opponent_search') == asdict(opponent_search or cfg.search)
                    and old.get('provenance', {}).get('dirty_state_sha256') == self.code.get('dirty_state_sha256')
                    and old.get('opponent_identity') == opponent_identity
                    and old.get('opponent_sha256') == opponent_hash):
                return old
        self.status('evaluation', label=label, games=cfg.num_games)
        result = evaluate_match(candidate, opponent, cfg, opponent_search)
        write_report(path, result)
        print(json.dumps({'label': label, 'summary': {k: v for k, v in result['summary'].items() if k != 'pair_scores'}}), flush=True)
        return result

    def arena(self, games: int = 256, *, split: str = 'development', search: SearchConfig = BASE_SEARCH,
              greedy: bool = False) -> ArenaConfig:
        offsets = {'development': 0, 'confirmation': 10_000_000, 'replication': 20_000_000}
        return ArenaConfig(num_games=games, seed=self.seed + offsets[split], inference_device=self.device,
                           split=split, search=search, greedy=greedy, bot_workers=self.bot_workers)

    def baseline(self) -> dict:
        pins = {}
        for label, source in INITIALIZERS.items():
            if not source.exists():
                raise FileNotFoundError(f'Required baseline checkpoint missing: {source}')
            target = self.root / 'baselines' / f'{label}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            elif checkpoint_hash(target) != checkpoint_hash(source):
                raise ValueError(f'Frozen baseline {label} differs from its source; use a new campaign root')
            pins[label] = str(target)
        champion = pins['v4_latest']
        cfg = self.arena()
        screen = {}
        for label, candidate in pins.items():
            for opponent in ('opus', 'heuristic'):
                r = self._match(f'baseline_{label}_{opponent}', candidate, opponent, cfg)
                screen[f'{label}_{opponent}'] = r['summary']
        # Challenge the provisional initializer only if a screen suggests improvement.
        opus_base = screen['v4_latest_opus']['match_score']
        challengers = sorted([k for k in pins if k != 'v4_latest'],
                             key=lambda k: screen[f'{k}_opus']['match_score'], reverse=True)
        for label in challengers:
            if screen[f'{label}_opus']['match_score'] <= opus_base:
                continue
            gate = self.confirm(f'baseline_{label}', pins[label], champion)
            if gate['promote']:
                champion = pins[label]
                break
        result = {'champion': champion, 'historical': pins['v3_latest'], 'pins': pins,
                  'search': asdict(BASE_SEARCH), 'screen': screen, 'provenance': self.code}
        latency = benchmark_latency(champion, BASE_SEARCH)
        write_report(self.root/'evaluations/baseline_latency.json', latency)
        if not latency['qualified']:
            raise RuntimeError('Baseline failed the five-second serving qualification')
        write_report(self.root / 'baseline.json', result)
        ModelRegistry(self.root / 'models/registry.json').register('trained_2p', champion,
            replace(BASE_SEARCH, move_deadline_s=5), status='baseline', evidence={'baseline_report': 'baseline.json'})
        self.status('baseline_complete', champion=champion)
        return result

    def confirm(self, label: str, candidate: str, champion: str,
                search: SearchConfig = BASE_SEARCH, split: str = 'confirmation') -> dict:
        salt = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % 1_000_000
        cfg = replace(self.arena(1024, split=split, search=search), seed=self.seed + (10_000_000 if split == 'confirmation' else 20_000_000) + salt)
        versus = self._match(f'{label}_{split}_champion', candidate, champion, cfg, BASE_SEARCH)
        opus = self._match(f'{label}_{split}_opus', candidate, 'opus', cfg)
        reference = self._match(f'champion_{checkpoint_hash(champion)[:12]}_{split}_opus', champion, 'opus', replace(cfg, search=BASE_SEARCH))
        decision = promotion_decision(versus, opus, reference)
        for name, opponent in [('heuristic', 'heuristic'), ('historical', str(self.root/'baselines/v3_latest.pt'))]:
            auxiliary = self._match(f'{label}_{split}_{name}', candidate, opponent, cfg, BASE_SEARCH)
            if auxiliary['summary']['unfinished']:
                decision.update(promote=False, reason=f'unfinished {name} games')
        write_report(self.root / 'evaluations' / f'{label}_{split}_decision.json', decision)
        return decision

    def _train(self, cfg: LoopConfig) -> str:
        directory = Path(cfg.runs_root) / cfg.run_id
        marker = directory / 'experiment_complete.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if _stored_training_config(record['config']) != asdict(cfg):
                raise ValueError('Completed experiment configuration differs; use a new run ID')
            return record['checkpoint']
        target_minutes = cfg.max_wall_minutes
        budget_path = directory / "experiment_budget.json"
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
            if (_stored_training_config(budget['config']) != asdict(cfg)
                    or budget.get('initializer_sha256') != checkpoint_hash(cfg.init_from)):
                raise ValueError("Changing an experiment's configuration or initializer requires a new run ID")
        else:
            write_report(budget_path, {"minutes": target_minutes, "config": asdict(cfg),
                                       'initializer_sha256': checkpoint_hash(cfg.init_from)})
        resume_path = directory / "checkpoints/latest_resume.pt"
        elapsed_minutes = 0.0
        if resume_path.exists():
            resume_meta = load_checkpoint_payload(resume_path, map_location="cpu")
            elapsed_minutes = resume_meta.get("progress", {}).get("training_wall_s", 0) / 60
            del resume_meta
        # Budget includes model setup, final checkpointing, and final evaluations.
        remaining = (self.deadline-time.monotonic()) / 60 - 5
        if remaining <= 0:
            raise TimeoutError('Overnight budget exhausted; rerun this stage to resume')
        cfg = replace(cfg, max_wall_minutes=max(0, min(target_minutes - elapsed_minutes, remaining)))
        league = League(cfg.league_root)
        if not league.list_entries():
            net, _ = load_net_from_checkpoint(cfg.init_from)
            net.trained_player_counts = [2]
            league.add_checkpoint(net, 'frozen_baseline', metadata={'pinned': True})
        self.status('training', run_id=cfg.run_id, config=asdict(cfg), heartbeat=str(directory/'heartbeat.json'))
        run = Run(cfg.run_id, runs_root=cfg.runs_root)
        before = time.monotonic()
        try:
            result = run_loop(run, cfg, explicit_fields=set(asdict(cfg)))
        finally:
            run.close()
        if result.get('stopped') or result.get("training_wall_s", 0) < target_minutes * 60 - 1:
            self.status('stopped', run_id=cfg.run_id)
            raise KeyboardInterrupt
        checkpoint = directory / 'checkpoints/latest_resume.pt'
        net, payload = load_net_from_checkpoint(checkpoint)
        finalist = directory / 'checkpoints/finalist.pt'
        save_checkpoint(finalist, net, iteration=payload['iteration'], config=payload['config'])
        write_report(marker, {'checkpoint': str(finalist), 'config': asdict(replace(cfg, max_wall_minutes=target_minutes)),
                             'train_wall_s': time.monotonic()-before, 'iteration': result['iter']})
        # Full replay is only retained while an experiment is active. The finalist
        # and recent weight/optimizer snapshots remain available after completion.
        checkpoint.unlink()
        write_report(directory / 'state.json', {'iter': result['iter'], 'last_checkpoint': str(finalist),
                                                'experiment_complete': True, 'replay_retained': False})
        return str(finalist)

    def _screen(self, label: str, candidate: str, baseline: dict, search: SearchConfig = BASE_SEARCH) -> dict:
        scores = {}
        for opponent_label, opponent in [('opus', 'opus'), ('heuristic', 'heuristic'), ('astra', 'astra'),
                                          ('champion', baseline['champion']), ('historical', baseline['historical'])]:
            r = self._match(f'{label}_{opponent_label}', candidate, opponent, self.arena(search=search), BASE_SEARCH)
            scores[opponent_label] = r['summary']
        greedy = self._match(f'{label}_greedy_opus', candidate, 'opus', self.arena(greedy=True))
        return {'checkpoint': candidate, 'search': asdict(search), 'scores': scores, 'greedy_opus': greedy['summary']}

    def policy(self, minutes: float = 235.) -> dict:
        baseline = self._read('baseline.json') if (self.root/'baseline.json').exists() else self.baseline()
        results = {}
        for arch in ('attn', 'source_attn'):
            cfg = training_config(self.root, f'policy_{arch}_seed{self.seed}', baseline['champion'],
                                  arch=arch, seed=self.seed, minutes=minutes, device=self.device)
            checkpoint = self._train(cfg)
            results[arch] = self._screen(f'policy_{arch}', checkpoint, baseline)
        best = max(results, key=lambda k: results[k]['scores']['champion']['match_score'])
        winner = results[best]
        gate = self.confirm('policy_winner', winner['checkpoint'], baseline['champion'])
        summary = {'results': results, 'winner_arch': best, 'winner': winner['checkpoint'],
                   'decision': gate, 'replicated': False}
        write_report(self.root/'policy.json', summary)
        self.status('policy_complete', winner=summary['winner'], promotion_pending_replication=gate['promote'])
        return summary

    def learner(self, minutes: float = 115., updates: tuple[int, ...] = (72, 144, 288)) -> dict:
        if not updates or len(set(updates)) != len(updates) or any(type(n) is not int or n < 1 for n in updates):
            raise ValueError('Learner update counts must be distinct positive integers')
        baseline, policy = self._read('baseline.json'), self._read('policy.json')
        results = {}
        for count in updates:
            cfg = training_config(self.root, f'learner_{count}_seed{self.seed}', policy['winner'],
                arch=policy['winner_arch'], seed=self.seed, learner_steps=count,
                minutes=minutes, device=self.device)
            checkpoint = self._train(cfg)
            results[str(count)] = self._screen(f'learner_{count}', checkpoint, baseline)
            initializer = self._match(f'learner_{count}_initializer', checkpoint, policy['winner'], self.arena())
            results[str(count)]['scores']['initializer'] = initializer['summary']
        best = max(results, key=lambda k: results[k]['scores']['champion']['match_score'])
        summary = {'results': results, 'winner_steps': int(best), 'winner': results[best]['checkpoint'],
                   'winner_arch': policy['winner_arch'], 'update_counts': list(updates),
                   'minutes_per_arm': minutes, 'initializer': policy['winner'], 'replicated': False,
                   'decision': self.confirm('learner_winner', results[best]['checkpoint'], baseline['champion'])}
        write_report(self.root/'learner.json', summary)
        self.status('learner_complete', winner=summary['winner'])
        return summary

    def search(self) -> dict:
        baseline = self._read('baseline.json')
        candidate = self._read('learner.json')['winner'] if (self.root/'learner.json').exists() else self._read('policy.json')['winner']
        results = {'one_ply': self._screen('search_one_ply', candidate, baseline)}
        latency = {'one_ply': benchmark_latency(candidate, BASE_SEARCH)}
        for sims in (64, 256, 1024):
            # Evaluate full, fixed budgets with batched inference; qualify the
            # same budget separately under a single-game serving deadline.
            profile = replace(BASE_SEARCH, backend='gumbel_tree', num_simulations=sims)
            latency[str(sims)] = benchmark_latency(candidate, profile)
            write_report(self.root/'evaluations/search_latency.json', latency)
            if latency[str(sims)]['qualified']:
                results[str(sims)] = self._screen(f'search_tree_{sims}', candidate, baseline, profile)
        best = max(results, key=lambda k: results[k]['scores']['champion']['match_score'])
        chosen = SearchConfig(**results[best]['search'])
        gate = self.confirm('search_winner', candidate, baseline['champion'], chosen)
        if not latency[best]['qualified']:
            gate.update(promote=False, reason='Move latency qualification failed')
        result = {'results': results, 'latency': latency, 'winner': best, 'checkpoint': candidate, 'search': asdict(chosen), 'decision': gate}
        write_report(self.root/'search.json', result)
        self.status('search_complete', winner=best)
        return result

    def bounded_search(self) -> dict:
        from agent.scripts.search_study import run_study
        baseline, policy = self._read('baseline.json'), self._read('policy.json')
        if (self.root/'supervisor.json').exists():
            deadline = self._read('supervisor.json')['deadline']
        elif (self.root/'search_manifest.json').exists():
            deadline = self._read('search_manifest.json')['deadline']
        else:
            deadline = time.time() + 2*3600
        result = run_study(self.root, policy['winner'], baseline['champion'],
                           device=self.device, seed=self.seed, bot_workers=self.bot_workers,
                           deadline=deadline)
        self.status('search_complete', winner=result['winner'], decision=result['decision'],
                    completed_games_total=result['completed_games'])
        return result

    def tree_distill(self, minutes: float = 450.) -> dict:
        """One bounded run that trains on the confirmed tree-search teacher."""
        source = REPO/'agent/runs/competitive_search_2h_20260914/initializers/source_attn.pt'
        champion = REPO/'agent/runs/competitive_search_2h_20260914/initializers/v4.pt'
        if not source.exists() or not champion.exists():
            raise FileNotFoundError('The confirmed search-study initializer or frozen champion is missing')
        pins = self.root/'initializers'
        pins.mkdir(parents=True, exist_ok=True)
        def pin(source_path: Path, name: str) -> Path:
            target = pins/name
            if not target.exists():
                shutil.copy2(source_path, target)
            if checkpoint_hash(target) != checkpoint_hash(source_path):
                raise ValueError(f'Pinned {name} differs from its source')
            return target
        initializer = pin(source, 'source_attn_tree_teacher.pt')
        frozen_champion = pin(champion, 'v4_champion.pt')
        teacher = replace(BASE_SEARCH, backend='gumbel_tree', num_simulations=64)
        cfg = tree_distill_config(self.root, f'tree_distill_seed{self.seed}', str(initializer),
                                  seed=self.seed, minutes=minutes, device=self.device)
        plan = {'purpose': 'Distil 64-simulation tree-search policy targets into source_attn.',
                'initializer': str(initializer), 'initializer_sha256': checkpoint_hash(initializer),
                'champion': str(frozen_champion), 'champion_sha256': checkpoint_hash(frozen_champion),
                'teacher_search': asdict(teacher), 'training_config': asdict(cfg),
                'tree_batch_benchmark': {'games': 256, 'positions': 13908, 'wall_s': 129.461,
                                         'seed': 20260915}, 'automatic_promotion': False,
                'provenance': self.code}
        write_report(self.root/'tree_distill_plan.json', plan)
        checkpoint = self._train(cfg)
        # The remaining bounded window is spent on fixed, comparable screens.
        screens = {}
        for label, opponent, opponent_search, games in [
            # Astra is the primary external strength bar for this experiment.
            # The larger paired sample is still small enough for the reserved
            # final half hour when its native searches run across eight workers.
            ('tree64_astra', 'astra', None, 512),
            ('tree64_initializer', str(initializer), teacher, 256),
            ('greedy_astra', 'astra', None, 256),
        ]:
            if time.monotonic() > self.deadline - 120:
                break
            arena = self.arena(games, search=teacher, greedy=label == 'greedy_astra')
            screens[label] = self._match(f'tree_distill_{label}', checkpoint, opponent, arena,
                                         opponent_search)['summary']
        result = {'checkpoint': checkpoint, 'teacher_search': asdict(teacher), 'screens': screens,
                  'training_config': asdict(cfg), 'decision': {'promote': False,
                  'reason': 'A fresh training seed requires held-out confirmation and replication'},
                  'automatic_promotion': False}
        write_report(self.root/'tree_distill.json', result)
        self.status('tree_distill_complete', checkpoint=checkpoint, screens=list(screens),
                    automatic_promotion=False)
        return result

    def tree_training(self, minutes: float = 235.) -> dict:
        search = self._read('search.json')
        if search['search']['backend'] != 'gumbel_tree' or not search['decision']['promote']:
            result = {'skipped': True, 'reason': 'Tree search did not pass the promotion gate'}
            write_report(self.root/'tree_training.json', result)
            return result
        baseline = self._read('baseline.json')
        net, _ = load_net_from_checkpoint(search['checkpoint'])
        learner_steps = self._read('learner.json')['winner_steps'] if (self.root/'learner.json').exists() else 72
        results = {}
        for backend in ('one_ply', 'gumbel_tree'):
            cfg = training_config(self.root, f'tree_training_{backend}_seed{self.seed}', search['checkpoint'],
                                  arch=net.arch, seed=self.seed, minutes=minutes, backend=backend, device=self.device,
                                  learner_steps=learner_steps)
            checkpoint = self._train(cfg)
            results[backend] = self._screen(f'tree_training_{backend}', checkpoint, baseline, SearchConfig(**search['search']))
        best = max(results, key=lambda k: results[k]['scores']['champion']['match_score'])
        result = {'results': results, 'winner': results[best]['checkpoint'], 'winner_backend': best,
                  'search': search['search'], 'decision': self.confirm('tree_training_winner', results[best]['checkpoint'], baseline['champion'], SearchConfig(**search['search']))}
        write_report(self.root/'tree_training.json', result)
        self.status('tree_training_complete', winner=result['winner'])
        return result

    def replicate(self, stage: str = 'policy', minutes: float | None = None) -> dict:
        baseline, selected = self._read('baseline.json'), self._read(f'{stage}.json')
        if not selected['decision']['promote']:
            raise ValueError('Selected configuration has not passed held-out confirmation')
        winner = selected['winner']
        # Reproduce the winning experiment from its original initializer, not from its trained weights.
        record = json.loads((Path(winner).parents[1]/'experiment_complete.json').read_text())
        cfg = LoopConfig(**_stored_training_config(record['config']))
        run_id = cfg.run_id + '_replication'
        cfg = replace(cfg, run_id=run_id, seed=cfg.seed+1_000_000, max_wall_minutes=minutes if minutes is not None else cfg.max_wall_minutes,
                      league_root=str(self.root/'experiments'/run_id/'league'))
        checkpoint = self._train(cfg)
        search = SearchConfig(**selected.get('search', asdict(BASE_SEARCH)))
        gate = self.confirm(f'{stage}_replicated', checkpoint, baseline['champion'], search, split='replication')
        latency = benchmark_latency(checkpoint, search)
        write_report(self.root/f'evaluations/{stage}_replication_latency.json', latency)
        if not latency['qualified']:
            gate.update(promote=False, reason='Replicated model failed serving latency qualification')
        gate['replicated'] = bool(gate['promote'])
        if gate['replicated']:
            ModelRegistry(self.root/'models/registry.json').register('trained_2p', checkpoint,
                replace(search, move_deadline_s=5), status='promoted', evidence=gate)
        result = {'checkpoint': checkpoint, 'decision': gate}
        write_report(self.root/f'{stage}_replication.json', result)
        self.status('replication_complete', promoted=gate['replicated'])
        return result


def _budget_watchdog(finished: threading.Event, seconds: float = 480 * 60) -> None:
    """Request a boundary save two minutes early; enforce the overnight ceiling.

    A pathological iteration may not respond promptly. The last atomic resume
    remains intact even when that process must be killed at the hard deadline.
    """
    if not finished.wait(max(0, seconds - 120)):
        os.kill(os.getpid(), signal.SIGTERM)
        if not finished.wait(min(120, seconds)):
            os.kill(os.getpid(), signal.SIGKILL)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['baseline', 'policy', 'learner', 'search', 'tree-training', 'tree-distill', 'replicate', 'status'])
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--minutes-per-arm', type=float)
    p.add_argument('--learner-updates', type=int, nargs='+', default=[72, 144, 288])
    p.add_argument('--bot-workers', type=int, choices=range(1, 9), default=1)
    p.add_argument('--bounded-search', action='store_true', help='Run resumable search screens within the supervisor time budget')
    p.add_argument('--replicate-stage', choices=['policy', 'learner', 'tree_training'], default='policy')
    args = p.parse_args()
    if args.stage == 'status':
        print((args.root/'status.json').read_text())
        return
    max_minutes = 450 if args.stage == 'tree-distill' else 235
    if args.minutes_per_arm is not None and not 0 < args.minutes_per_arm <= max_minutes:
        p.error(f'--minutes-per-arm must be in (0, {max_minutes}]')
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    configure_device(device)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'.campaign.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another campaign stage is already running')
        campaign = Campaign(args.root, device, args.seed, bot_workers=args.bot_workers)
        finished = threading.Event()
        def request_stop(signum: int, frame: object) -> None:
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, request_stop)
        threading.Thread(target=_budget_watchdog, args=(finished,), daemon=True).start()
        try:
            if args.stage == 'baseline': campaign.baseline()
            elif args.stage == 'policy': campaign.policy(args.minutes_per_arm or 235)
            elif args.stage == 'learner': campaign.learner(args.minutes_per_arm or 115, tuple(args.learner_updates))
            elif args.stage == 'search':
                campaign.bounded_search() if args.bounded_search else campaign.search()
            elif args.stage == 'tree-training': campaign.tree_training(args.minutes_per_arm or 235)
            elif args.stage == 'tree-distill': campaign.tree_distill(args.minutes_per_arm or 450)
            elif args.stage == 'replicate': campaign.replicate(args.replicate_stage, args.minutes_per_arm)
        except KeyboardInterrupt:
            campaign.status('stopped')
        except Exception as exc:
            campaign.status('failed', error=str(exc))
            raise
        finally:
            finished.set()
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
