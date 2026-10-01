"""Measure process-parallel tree traversal with centralized GPU inference."""

from __future__ import annotations

import argparse
import json
import time

import torch

from agent.env.engine import GameEngine
from agent.search.config import SearchConfig
from agent.search.gumbel_mcts import gumbel_root_act
from agent.search.parallel import shutdown_parallel_tree_pools
from agent.train.checkpointing import load_net_from_checkpoint
from agent.train.replay_buffer import ReplayBuffer
from agent.train.selfplay import run_selfplay
from agent.net import encoder as encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--tree-core", choices=["python", "rust"], nargs="+", default=["python", "rust"]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--selfplay-turns", type=int, default=0)
    args = parser.parse_args()

    net, _ = load_net_from_checkpoint(args.checkpoint, map_location="cuda")
    net.eval()
    engine = GameEngine(args.games, 2, "cuda", seed=20260915)
    results = []
    try:
        for tree_core in args.tree_core:
            for workers in args.workers:
                config = SearchConfig(
                    backend="gumbel_tree",
                    num_simulations=args.sims,
                    tree_core=tree_core,
                    max_root_candidates=16,
                    leaf_batch_size=256,
                    seed=20260915,
                    cpu_workers=workers,
                    inference_batch_size=2048,
                    inference_wait_ms=2,
                )
                samples = []
                for repeat in range(args.repeats + 1):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actions, policies = gumbel_root_act(
                        engine, net, search_config=config
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    legal = engine.legal_action_mask()
                    assert legal.gather(1, actions[:, None]).all()
                    assert torch.allclose(
                        policies.sum(1),
                        torch.ones(args.games, device="cuda"),
                        atol=1e-5,
                    )
                    if repeat:
                        samples.append(elapsed)
                results.append(
                    {
                        "tree_core": tree_core,
                        "workers": workers,
                        "seconds_per_move": sum(samples) / len(samples),
                        "root_positions_per_second": args.games
                        * len(samples)
                        / sum(samples),
                        "samples": samples,
                    }
                )
                print(json.dumps(results[-1]), flush=True)
    finally:
        shutdown_parallel_tree_pools()
    if args.selfplay_turns:
        buffer = ReplayBuffer(
            args.games * args.selfplay_turns,
            encoder.D_GLOBAL,
            encoder.NUM_SOURCES,
            encoder.D_SOURCE,
            300,
            4,
            "cuda",
        )
        started = time.perf_counter()
        selfplay = run_selfplay(
            net,
            buffer,
            num_games=args.games,
            num_players=2,
            num_sims=args.sims,
            max_turns=args.selfplay_turns,
            seed=20260915,
            device="cuda",
            reward_mode="binary",
            search_backend="gumbel_tree",
            search_tree_core=args.tree_core[-1],
            search_cpu_workers=max(args.workers),
            search_inference_batch_size=2048,
            search_inference_wait_ms=2,
        )
        selfplay["measured_wall_s"] = time.perf_counter() - started
        print(json.dumps({"selfplay": selfplay}), flush=True)
        shutdown_parallel_tree_pools()
    print(
        json.dumps(
            {"games": args.games, "sims": args.sims, "results": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
