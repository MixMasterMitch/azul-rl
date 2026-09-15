"""Run a paired evaluation without modifying any training league."""
from __future__ import annotations
import argparse
import json
import torch
from agent.eval.arena import ArenaConfig, evaluate_match, write_report
from agent.search.config import SearchConfig
from agent.train.device import configure_device, resolve_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate')
    p.add_argument('opponent', help='Checkpoint, astra, opus, heuristic, or random')
    p.add_argument('--games', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--device', default='auto')
    p.add_argument('--bot-workers', type=int, default=1,
                   help='Parallel builtin searches across games (1–8; useful for Astra)')
    p.add_argument('--backend', choices=['one_ply', 'gumbel_tree'], default='one_ply')
    p.add_argument('--sims', type=int, default=64)
    p.add_argument('--q-scale', type=float, default=28.)
    p.add_argument('--temperature', type=float, default=.25)
    p.add_argument('--deadline', type=float)
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--split', choices=['development', 'confirmation', 'replication'], default='development')
    p.add_argument('--report', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    configure_device(device)
    cfg = ArenaConfig(num_games=args.games, seed=args.seed, inference_device=device, split=args.split,
                      bot_workers=args.bot_workers,
                      greedy=args.greedy, search=SearchConfig(backend=args.backend, num_simulations=args.sims,
                          q_scale=args.q_scale, temperature=args.temperature, move_deadline_s=args.deadline))
    report = evaluate_match(args.candidate, args.opponent, cfg,
                            progress=lambda r: print(json.dumps({k: v for k, v in r.items() if k != 'pair_scores'}), flush=True))
    write_report(args.report, report)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
