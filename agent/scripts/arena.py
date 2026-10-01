"""Run a paired evaluation without modifying any training league."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import torch
from agent.eval.arena import ArenaConfig, evaluate_match, write_report
from agent.search.config import SearchConfig
from agent.search.profiles import SEARCH_PROFILES, search_profile
from agent.train.device import configure_device, resolve_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate")
    p.add_argument("opponent", help="Checkpoint, astra, opus, heuristic, or random")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--bot-workers",
        type=int,
        default=1,
        help="Parallel builtin searches across games (1–8; useful for Astra)",
    )
    p.add_argument("--backend", choices=["one_ply", "gumbel_tree"], default="one_ply")
    p.add_argument("--tree-core", choices=["python", "rust"], default="python")
    p.add_argument("--profile", choices=SEARCH_PROFILES)
    p.add_argument("--opponent-profile", choices=SEARCH_PROFILES)
    p.add_argument("--root-noise-scale", type=float, default=1.0)
    p.add_argument("--max-root-candidates", type=int, default=16)
    p.add_argument("--chance-samples", type=int, default=4)
    p.add_argument("--inference-cache-size", type=int, default=0)
    p.add_argument("--opponent-root-noise-scale", type=float)
    p.add_argument("--opponent-max-root-candidates", type=int)
    p.add_argument("--opponent-chance-samples", type=int)
    p.add_argument("--opponent-inference-cache-size", type=int)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--q-scale", type=float, default=28.0)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--deadline", type=float)
    p.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="CPU tree-traversal workers for the candidate search (1–8)",
    )
    p.add_argument("--inference-batch-size", type=int, default=2048)
    p.add_argument("--inference-wait-ms", type=float, default=2.0)
    p.add_argument("--opponent-backend", choices=["one_ply", "gumbel_tree"])
    p.add_argument("--opponent-tree-core", choices=["python", "rust"])
    p.add_argument("--opponent-sims", type=int)
    p.add_argument("--opponent-q-scale", type=float)
    p.add_argument("--opponent-temperature", type=float)
    p.add_argument("--opponent-cpu-workers", type=int)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--opponent-greedy", action="store_true")
    p.add_argument(
        "--split",
        choices=["development", "confirmation", "replication"],
        default="development",
    )
    p.add_argument("--report", required=True)
    probe = p.parse_args()
    if probe.profile:
        fields = asdict(search_profile(probe.profile))
        fields["sims"] = fields.pop("num_simulations")
        fields["deadline"] = fields.pop("move_deadline_s")
        # Arena seeds define paired deals and must not inherit SearchConfig.seed=None.
        fields.pop("seed")
        destinations = {action.dest for action in p._actions}
        p.set_defaults(
            **{key: value for key, value in fields.items() if key in destinations}
        )
    args = p.parse_args()
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    configure_device(device)
    cfg = ArenaConfig(
        num_games=args.games,
        seed=args.seed,
        inference_device=device,
        split=args.split,
        bot_workers=args.bot_workers,
        greedy=args.greedy,
        opponent_greedy=args.opponent_greedy,
        search=SearchConfig(
            backend=args.backend,
            num_simulations=args.sims,
            tree_core=args.tree_core,
            root_noise_scale=args.root_noise_scale,
            max_root_candidates=args.max_root_candidates,
            chance_samples=args.chance_samples,
            inference_cache_size=args.inference_cache_size,
            q_scale=args.q_scale,
            temperature=args.temperature,
            move_deadline_s=args.deadline,
            cpu_workers=args.cpu_workers,
            inference_batch_size=args.inference_batch_size,
            inference_wait_ms=args.inference_wait_ms,
        ),
    )
    opponent_search = None
    opponent_overrides = {
        key: getattr(args, "opponent_" + arg)
        for key, arg in {
            "backend": "backend",
            "tree_core": "tree_core",
            "num_simulations": "sims",
            "q_scale": "q_scale",
            "temperature": "temperature",
            "cpu_workers": "cpu_workers",
            "root_noise_scale": "root_noise_scale",
            "max_root_candidates": "max_root_candidates",
            "chance_samples": "chance_samples",
            "inference_cache_size": "inference_cache_size",
        }.items()
        if getattr(args, "opponent_" + arg) is not None
    }
    if args.opponent_profile or opponent_overrides:
        if args.opponent_profile:
            base = search_profile(args.opponent_profile)
        elif args.opponent_backend is not None:
            base = SearchConfig(
                num_simulations=args.sims,
                q_scale=args.q_scale,
                temperature=args.temperature,
                inference_batch_size=args.inference_batch_size,
                inference_wait_ms=args.inference_wait_ms,
            )
        else:
            base = cfg.search
        opponent_search = SearchConfig(**{**asdict(base), **opponent_overrides})
    report = evaluate_match(
        args.candidate,
        args.opponent,
        cfg,
        opponent_search,
        progress=lambda r: print(
            json.dumps({k: v for k, v in r.items() if k != "pair_scores"}), flush=True
        ),
    )
    write_report(args.report, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
