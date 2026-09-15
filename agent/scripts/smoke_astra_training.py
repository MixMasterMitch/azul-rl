"""Validate Astra evaluation with real games and an isolated one-iteration training run."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from agent.eval.arena import ArenaConfig, evaluate_match, write_report
from agent.net.model import AzulNet
from agent.obs.run import Run
from agent.search.config import SearchConfig
from agent.train.checkpointing import save_checkpoint
from agent.train.loop import LoopConfig, run_loop
from agent.train.unified_eval import UnifiedEvalConfig, UnifiedEvalHandle


def smoke(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(916051)
    net = AzulNet(hidden=32, arch='flat')
    checkpoint = output / 'probe.pt'
    save_checkpoint(checkpoint, net, config={'num_players': 2})
    started = time.monotonic()
    config = UnifiedEvalConfig(total_games=6, num_sims=2, weight_2p=1, weight_3p=1, weight_4p=1,
                               num_workers=2, astra_opponent_fraction=1)
    handle = UnifiedEvalHandle(config, hidden=32, arch='flat')
    try:
        assert handle.launch(net.state_dict(), [], 1, 916052, context={'entity': 'probe'})
        collected = handle.wait_and_collect()
        assert collected is not None and 'error' not in collected[1], collected
        unified = collected[1]
    finally:
        handle.cleanup()
    for n in (2, 3, 4):
        suffix = f"{n}p_{'vs' if n == 2 else 'with'}_astra"
        assert unified['metrics'][f'finished_{suffix}'] == 2, unified['metrics']
        assert unified['metrics'][f'unfinished_{n}p'] == 0, unified['metrics']
    write_report(output / 'unified.json', {'config': asdict(config), **unified})
    print('Six real unified-evaluation games completed across 2p/3p/4p.', flush=True)

    paired = evaluate_match(str(checkpoint), 'astra', ArenaConfig(
        num_games=2, game_batch_size=2, seed=916053, search=SearchConfig(num_simulations=2)))
    assert paired['summary']['unfinished'] == 0, paired['summary']
    write_report(output / 'arena.json', paired)
    print('Two real paired-arena games completed.', flush=True)

    config = LoopConfig(run_id='training', runs_root=str(output), league_root=str(output / 'league'),
        device='cpu', hidden=32, arch='flat', max_iters=1, max_wall_minutes=5,
        selfplay_games=4, selfplay_sims=2, replay_capacity=1000, learner_batch=16,
        learner_steps_per_iter=2, training_cycle_length=0, league_selfplay_every=0,
        checkpoint_every=1, eval_games=4, eval_sims=2, eval_workers=2,
        eval_astra_fraction=1, save_buffer_in_checkpoints=False, seed=916054)
    run = Run(config.run_id, runs_root=config.runs_root)
    try:
        training = run_loop(run, config, explicit_fields=set(asdict(config)))
    finally:
        run.close()
    rows = [json.loads(line) for line in run.metrics_path.read_text().splitlines()]
    metrics = next(row for row in rows if 'games_2p_vs_astra' in row)
    assert metrics['finished_2p_vs_astra'] == 4 and metrics['unfinished_2p_vs_astra'] == 0, metrics
    events = [json.loads(line) for line in run.events_path.read_text().splitlines()]
    assert any(event['event'] == 'unified_eval_done' and event['fields'].get('astra_identity') for event in events)
    league = json.loads((output / 'league' / 'league.json').read_text())
    assert any('astra' in (row['a'], row['b']) for row in league['results'])
    assert 'astra' not in league['anchors']
    summary = {'validation_only': True, 'games': 12, 'finished': 12, 'unfinished': 0,
               'unified_workers': 2, 'training_workers': 2, 'player_counts': [2, 3, 4],
               'wall_s': time.monotonic() - started, 'training': training,
               'training_metrics': metrics, 'astra_identity': unified['astra_identity']}
    write_report(output / 'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New isolated output directory')
    args = parser.parse_args()
    summary = smoke(args.output.resolve())
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {'training', 'training_metrics', 'astra_identity'}}, indent=2))


if __name__ == '__main__':
    main()
