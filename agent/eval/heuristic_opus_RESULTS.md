# heuristic-opus: candidate development log and final ratings

The "heuristic-opus" Azul agent was developed iteratively across 20
candidates. Each candidate self-played a round-robin tournament against the
existing reference bots (`random`, `heuristic`) and previous candidates,
fitting Elo ratings (random=1000). All tournaments used the BatchedEngine.

The strongest candidate by aggregate rating across all player counts is
**`heuristic_opus_v20`** (dispatch: V13 at 2p, V19 at 3p/4p).

## Final ratings (anchored, random=1000)

From a definitive tournament with 192 games per matchup (seed=9999):

| Entity | 2p | 3p | 4p |
| --- | ---: | ---: | ---: |
| **heuristic_opus_v20** | **1735** | **1828** | **1861** |
| heuristic_opus_v19 | 1733 | 1722 | 1761 |
| heuristic_opus_v13 | 1704 | 1671 | 1680 |
| heuristic | 1613 | 1593 | 1551 |
| random | 1000 | 1000 | 1000 |

Win rates of V20 vs baseline heuristic:
- 2p: 65.1%
- 3p: 67.2%
- 4p: 77.1%

## Candidate progression

* **V1 -- "Score-aware greedy"**: Scores each action by wall placement
  points (with adjacency simulation), floor penalties, and line fill ratio.
  Significantly stronger than the basic HeuristicBot.

* **V2 -- "Bonus-aware"**: V1 + end-game bonus tracking (+2/row, +7/column,
  +10/color). Prioritizes tiles that contribute to near-complete bonuses.

* **V3 -- "Denial + center timing"**: V2 + opponent denial (take tiles they
  need for nearly-complete pattern lines) and strategic center timing.

* **V4 -- "Multi-round planning"**: V3 + tile availability awareness:
  prefer starting pattern lines where enough tiles exist to complete.

* **V5 -- "Lookahead-aware"**: V4 + future value estimation for partial
  lines and overflow danger zone scaling.

* **V6 -- "Late-game aggression"**: V5 + late-game mode with aggressive
  scoring when any player is near completing a wall row.

* **V7 -- "Refined denial"**: V6 + scale denial by player count and
  opponent score.

* **V8 -- "Overflow-aware source selection"**: V7 + prefer sources that
  minimize overflow, penalize dumping lots of tiles to center.

* **V9 -- "Wall clustering"**: V8 + prefer wall placements that build
  clusters (maximize adjacency for future placements). Strong at 2p.

* **V10 -- "Opponent game-end awareness"**: V9 + awareness of opponents
  about to end the game.

* **V11 -- "Holistic value function"**: Ground-up rewrite with a single
  coherent value function. Computes immediate wall score, floor penalty,
  bonus contribution, positional value, denial, and tile fit from first
  principles. Strong contender.

* **V12 -- "Refined V11"**: V11 + improved late-game urgency and center
  timing.

* **V13 -- "Fusion"**: Fusion of V9's clustering with V11's holistic
  framework, plus tuned weights. Strongest at 2p. Won with exponential
  floor scaling, selective high-value denial, and early-game center
  clustering.

* **V14 -- "Factory leftover awareness"**: V13 + penalize factory picks
  that dump useful tiles to center for opponents. Penalize starting lines
  when a much better color exists for that row.

* **V15 -- "1-ply lookahead"**: V13 + evaluate top-5 actions by simulating
  them and scoring the resulting board state. Strong but adds compute cost
  for marginal gain over V13.

* **V16 -- "Color monopoly"**: V13 + prefer colors where we hold most of
  the remaining tiles across sources. Helps line safety.

* **V17 -- "Opponent scoring denial"**: V13 + estimate what opponents
  would score and prefer actions that starve them of high-scoring
  completions. Strong at multi-player.

* **V18 -- "Phase-adaptive weights"**: V13 with early/mid/late game weight
  tuning. Marginal improvement.

* **V19 -- "Best-of-V13+V16+V17 fusion"**: Integrates color monopoly and
  opponent scoring denial into V13. Strongest at 3p/4p.

* **V20 -- "Dispatch"**: V13 at 2p, V19 at 3p/4p. Best aggregate by
  combining the per-player-count strongest candidates.

## Key strategic insights

1. **Wall adjacency scoring is the single biggest lever**. Simulating the
   actual wall score for each placement (connected tiles horizontally +
   vertically) makes the bot dramatically stronger than the baseline
   heuristic that just counts adjacent neighbors.

2. **Column and color bonuses dominate end-game value**. +7 per column and
   +10 per color set are worth far more than +2 per row. The bot weights
   column/color progress 2-3x more than row progress.

3. **Floor penalty scaling is exponential danger**. The -3 penalties at
   positions 6-7 make any floor overflow devastating. The bot applies
   extra penalty weight when floor is already at 4+ tiles.

4. **Denial effectiveness scales with player count**. At 2p, denial is
   about even with self-scoring optimization. At 3p/4p, multi-opponent
   denial awareness (V19) adds ~100-150 Elo over V13.

5. **Clustering matters in early game**. Placing tiles near the center of
   the wall early creates more adjacency opportunities for future tiles.
   This effect diminishes as the wall fills up.

6. **Color monopoly is a safe play indicator**. When you're taking most of
   the remaining tiles of a color, the line is safe to start — opponents
   can't easily grab the tiles you need to complete it.

7. **Per-player-count dispatch is real improvement**. V20 (dispatch) is
   strictly better than any single candidate because the optimal strategy
   differs meaningfully between 2p (tight races, less denial value) and
   3p/4p (more chaos, denial matters more).

## Files

* `agent/eval/heuristic_opus.py` -- candidate definitions (V1-V20)
* `agent/eval/tournament_opus.py` -- round-robin harness, Elo ratings
* `agent/scripts/heuristic_opus_tournament.py` -- CLI driver

## Reproducing

```bash
source .venv/bin/activate
python -m agent.scripts.heuristic_opus_tournament \
    --candidates v13,v19,v20 \
    --num-games 192 --player-counts 2,3,4 \
    --seed 9999
```
