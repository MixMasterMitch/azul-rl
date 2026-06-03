"""Generate reference-anchor games and write them into league.json.

Plays the heuristic triangle (random, heuristic, heuristic_opus) at 2p/3p/4p,
accumulates bot-vs-bot reference rows in ``results``, stores fitted anchors in
``reference_anchors_per_pc``, and recomputes league ratings.

Usage:
    python -m agent.scripts.gen_reference_anchors \\
        --league-dir agent/runs/league \\
        --num-games 2000 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import pathlib
import time

import torch

from ..eval import bots as B
from ..eval.heuristic_opus import HeuristicOpusBot
from ..env import batched_engine as BE
from ..train import ranking as R
from ..train.league import League


def _play_bot_matchup(
    name_a: str,
    bot_a: B.RandomBot | B.HeuristicBot | HeuristicOpusBot,
    name_b: str,
    bot_b: B.RandomBot | B.HeuristicBot | HeuristicOpusBot,
    num_games: int,
    num_players: int,
    device: torch.device,
    seed: int,
    max_turns: int,
    log_every: int,
    label: str,
    matchup_t0: float,
) -> tuple[int, int, int, int]:
    wins_a = 0
    wins_b = 0
    skipped = 0
    log_every = max(1, log_every)
    for game_idx in range(num_games):
        engine = BE.BatchedEngine(1, num_players, device, seed=seed + game_idx * 17)
        seat_a = game_idx % num_players
        for _ in range(max_turns):
            if engine.ended[0]:
                break
            cp = int(engine.current_player[0].item())
            if cp == seat_a:
                action = bot_a.select_action(engine, 0)
            else:
                action = bot_b.select_action(engine, 0)
            engine.step(torch.tensor([action], dtype=torch.long, device=device))

        winner = int(engine.get_winners()[0].item())
        if winner < 0:
            skipped += 1
        elif winner == seat_a:
            wins_a += 1
        else:
            wins_b += 1

        done = game_idx + 1
        if done % log_every == 0 or done == num_games:
            decided = wins_a + wins_b
            wr_a = wins_a / decided if decided else 0.0
            elapsed = time.monotonic() - matchup_t0
            gps = done / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{label}] {done}/{num_games} games "
                f"| {name_a} {wins_a}-{wins_b} {name_b} (skipped {skipped}) "
                f"| {name_a} wr {wr_a:.1%} "
                f"| {gps:.1f} games/s "
                f"| {elapsed:.0f}s",
                flush=True,
            )
    return wins_a, wins_b, skipped


def _reference_result_rows(results: list[dict]) -> list[dict]:
    return [row for row in results if R.is_reference_only_result_row(row)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play reference bot triangle and update league.json"
    )
    parser.add_argument(
        "--league-dir",
        type=str,
        default="agent/runs/league",
        help="League directory containing league.json",
    )
    parser.add_argument("--num-games", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress every N games per matchup (default: 100)",
    )
    args = parser.parse_args()

    league = League(pathlib.Path(args.league_dir))
    device = torch.device(args.device)

    random_bot = B.RandomBot(seed=args.seed)
    heuristic_bot = B.HeuristicBot(seed=args.seed + 1)
    opus_bot = HeuristicOpusBot(seed=args.seed + 2)

    pairs = [
        ("heuristic", heuristic_bot, "random", random_bot),
        ("heuristic_opus", opus_bot, "random", random_bot),
        ("heuristic", heuristic_bot, "heuristic_opus", opus_bot),
    ]

    t0 = time.monotonic()
    total_matchups = len(R.PLAYER_COUNTS) * len(pairs)
    matchup_idx = 0
    print(
        f"Reference triangle: {total_matchups} matchups × {args.num_games} games "
        f"(log every {args.log_every}) → {league.manifest_path}",
        flush=True,
    )
    for num_players in R.PLAYER_COUNTS:
        for name_a, bot_a, name_b, bot_b in pairs:
            matchup_idx += 1
            label = f"{matchup_idx}/{total_matchups}"
            print(
                f"\n[{label}] {name_a} vs {name_b} @ {num_players}p "
                f"({args.num_games} games, seed={args.seed + num_players * 1000})",
                flush=True,
            )
            matchup_t0 = time.monotonic()
            wins_a, wins_b, skipped = _play_bot_matchup(
                name_a,
                bot_a,
                name_b,
                bot_b,
                args.num_games,
                num_players,
                device,
                args.seed + num_players * 1000,
                args.max_turns,
                log_every=args.log_every,
                label=label,
                matchup_t0=matchup_t0,
            )
            if wins_a + wins_b > 0:
                R.add_match_result(
                    league.manifest["results"],
                    name_a,
                    name_b,
                    float(wins_a),
                    float(wins_b),
                    0.0,
                    num_players=num_players,
                )
            decided = wins_a + wins_b
            wr_a = wins_a / decided if decided else 0.0
            print(
                f"  [{label}] done: {name_a} {wins_a}-{wins_b} {name_b}, "
                f"skipped={skipped}, {name_a} wr={wr_a:.1%}, "
                f"matchup {time.monotonic() - matchup_t0:.1f}s",
                flush=True,
            )

    print("\nFitting reference anchors and recomputing league ratings...", flush=True)
    ref_rows = _reference_result_rows(league.manifest["results"])
    anchors_per_pc: dict[str, dict[str, float]] = {}
    for pc in R.PLAYER_COUNTS:
        fitted = R.fit_ratings_for_pc(
            ref_rows,
            pc,
            use_reference_anchors=False,
        )
        anchors_per_pc[str(pc)] = {k: round(float(v), 1) for k, v in fitted.items()}

    league.manifest["reference_anchors_per_pc"] = anchors_per_pc
    league.manifest["reference_games_per_matchup"] = args.num_games
    league.manifest["reference_seed"] = args.seed
    league.manifest["reference_updated_at"] = int(time.time())
    league.manifest.setdefault("anchors", dict(R.DEFAULT_ANCHORS))

    ratings = league.recompute_ratings()
    wall_s = time.monotonic() - t0

    print(f"\nWrote reference data to {league.manifest_path} ({wall_s:.1f}s)")
    print("reference_anchors_per_pc:")
    for pc in R.PLAYER_COUNTS:
        anchors = anchors_per_pc[str(pc)]
        print(f"  {pc}p: {anchors}")

    floating = league.manifest.get("floating_entities", {})
    for entity in ("random", "heuristic", "heuristic_opus"):
        if entity in floating:
            fe = floating[entity]
            print(
                f"  {entity}: rating={fe.get('rating')} games={fe.get('games')} "
                f"(2p={fe.get('rating_2p')} 3p={fe.get('rating_3p')} 4p={fe.get('rating_4p')})"
            )
        elif entity in ratings:
            print(f"  {entity}: rating={ratings[entity]}")


if __name__ == "__main__":
    main()
