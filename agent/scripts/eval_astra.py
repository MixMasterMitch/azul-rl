"""Run/resume an isolated Astra experiment: python -m agent.scripts.eval_astra --help."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from agent.eval.heuristic_astra import AstraConfig, production_config
from agent.eval.astra_tournament import run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--incumbent-config", type=Path)
    parser.add_argument("--reference-experiment", type=Path, help="Reuse this experiment's frozen engine, reference bots, neural code and checkpoint bytes")
    parser.add_argument("--checkpoint-policy", choices=["greedy", "search32"], default="greedy")
    parser.add_argument("--name", default="astra")
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--fields", nargs="+", default=["opus"])
    parser.add_argument("--split", choices=["development", "promotion", "final"], default="development")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--control", choices=["opus"])
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--center-nodes", type=int)
    parser.add_argument("--terminal-nodes", type=int,
                        help="Raise the budget once a completed pattern line guarantees this is the final round")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--time-ms", type=int, help="Native deadline in milliseconds, at most 2000")
    parser.add_argument("--weight", action="append", default=[], metavar="NAME=VALUE",
                        help="Manually override a named heuristic weight; repeat for ablations")
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    config = (AstraConfig.from_dict(json.loads(args.config.read_text())) if args.config
              else {n: production_config(n) for n in args.players})
    overrides = {k: getattr(args, k) for k in ("nodes", "center_nodes", "terminal_nodes", "depth", "time_ms") if getattr(args, k) is not None}
    try:
        weight_overrides = {name: float(value) for name, value in (x.split("=", 1) for x in args.weight)}
        def apply_overrides(c: AstraConfig) -> AstraConfig:
            return replace(c, **overrides, weights=replace(c.weights, **weight_overrides))
        config = ({n: apply_overrides(c) for n, c in config.items()} if isinstance(config, dict)
                  else apply_overrides(config))
    except (ValueError, TypeError) as exc:
        parser.error(f"Invalid configuration override: {exc}")
    incumbent_config = AstraConfig.from_dict(json.loads(args.incumbent_config.read_text())) if args.incumbent_config else None
    report = run_tournament(args.output, config=config, players=args.players, blocks=args.blocks,
                            fields=args.fields, split=args.split, seed=args.seed, workers=args.workers,
                            control=args.control, max_turns=args.max_turns, trace=args.trace, name=args.name,
                            incumbent_config=incumbent_config, checkpoint_policy=args.checkpoint_policy,
                            reference_experiment=args.reference_experiment)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "block_scores"}
                      for k, v in report["summary"].items()}, indent=2))


if __name__ == "__main__":
    main()
