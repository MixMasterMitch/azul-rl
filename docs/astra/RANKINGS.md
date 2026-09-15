# Astra: expanded agent rankings

Completed **19,456 games** with 0 unfinished games and 0 failures.

The production Astra configuration was frozen throughout. This campaign used fresh seeds and included matches between every eligible pair of agents. Neural agents participated only at their trained player count (two).

## Astra win share against homogeneous opponents

| Players | Opponent | Games | Astra win share | 95% block interval |
| --- | --- | ---: | ---: | --- |
| 2p | heuristic | 512 | 99.51% | 98.83%–100.00% |
| 2p | league_2p | 512 | 82.42% | 79.30%–85.35% |
| 2p | opus | 512 | 99.02% | 98.05%–99.80% |
| 2p | random | 512 | 100.00% | 100.00%–100.00% |
| 2p | trained_2p | 512 | 79.39% | 75.98%–82.81% |
| 3p | heuristic | 384 | 57.55% | 52.86%–62.11% |
| 3p | opus | 384 | 54.38% | 49.09%–59.51% |
| 3p | random | 384 | 100.00% | 100.00%–100.00% |
| 4p | heuristic | 512 | 56.22% | 51.95%–60.45% |
| 4p | opus | 512 | 50.10% | 45.80%–54.20% |
| 4p | random | 512 | 100.00% | 100.00%–100.00% |

Each game contains one Astra versus the remaining copies of the named opponent. A k-way shared win earns 1/k. All-success bootstrap intervals can collapse and do not establish perfect play.

## Bradley–Terry ranking on the heuristic-2500 display scale

Ratings use `1000 + multiplier × (raw − 1000)`, with frozen multipliers 0.3739824775 (2p), 0.3106665462 (3p), and 0.2949804033 (4p). This keeps random at 1000 and puts the reference heuristic at 2500. These are presentation changes: the original statistical fits, game outcomes, ordering, and win probabilities are unchanged.

The ratings summarize pairwise final score/row placements across the complete field, including mixed tables. They are secondary to game win share. This common numerical scale does not establish equivalent human ability across player counts. See [rating scale v1](RATING_SCALE.md) for reference values and treatment of legacy leagues.


### 2 players

| Rank | Agent | Rating | Rating 95% interval | Rank 95% interval | Bootstrap first-place share |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | astra | 3118 | 3095–3143 | 1–1 | 100.0% |
| 2 | league_2p | 2860 | 2839–2883 | 2–3 | 0.0% |
| 3 | trained_2p | 2859 | 2840–2882 | 2–3 | 0.0% |
| 4 | opus | 2579 | 2566–2593 | 4–4 | 0.0% |
| 5 | heuristic | 2500 | 2490–2510 | 5–5 | 0.0% |
| 6 | random | 1000 | 1000–1000 | 6–6 | 0.0% |

### 3 players

| Rank | Agent | Rating | Rating 95% interval | Rank 95% interval | Bootstrap first-place share |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | astra | 2638 | 2629–2647 | 1–1 | 100.0% |
| 2 | opus | 2547 | 2540–2553 | 2–2 | 0.0% |
| 3 | heuristic | 2500 | 2495–2505 | 3–3 | 0.0% |
| 4 | random | 1000 | 1000–1000 | 4–4 | 0.0% |

### 4 players

| Rank | Agent | Rating | Rating 95% interval | Rank 95% interval | Bootstrap first-place share |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | astra | 2605 | 2599–2612 | 1–1 | 100.0% |
| 2 | opus | 2526 | 2521–2532 | 2–2 | 0.0% |
| 3 | heuristic | 2500 | 2496–2504 | 3–3 | 0.0% |
| 4 | random | 1000 | 1000–1000 | 4–4 | 0.0% |

Ranking confidence intervals use 2,000 stratified bootstrap resamples of whole seed blocks. The display transformation is applied to both endpoints with the same frozen multiplier; it is not refitted in each resample. Every matchup and seat rotation sharing a draw seed moves together. Bootstrap first-place share describes resampling stability, not a Bayesian probability of being the best agent. A weak Gaussian prior keeps perfect or nearly separated results finite; absolute rating levels can depend strongly on that prior.

## Complete matchup matrix

Cells are the row agent’s win share against a table filled by the column agent. Three- and four-player cells in opposite directions are different compositions, so they need not sum to 100%.


### 2 players

| Agent | astra | heuristic | league_2p | opus | random | trained_2p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| astra | — | 99.5% | 82.4% | 99.0% | 100.0% | 79.4% |
| heuristic | 0.5% | — | 14.6% | 34.1% | 100.0% | 10.7% |
| league_2p | 17.6% | 85.4% | — | 86.5% | 100.0% | 52.8% |
| opus | 1.0% | 65.9% | 13.5% | — | 100.0% | 15.1% |
| random | 0.0% | 0.0% | 0.0% | 0.0% | — | 0.0% |
| trained_2p | 20.6% | 89.3% | 47.2% | 84.9% | 100.0% | — |

### 3 players

| Agent | astra | heuristic | opus | random |
| --- | ---: | ---: | ---: | ---: |
| astra | — | 57.6% | 54.4% | 100.0% |
| heuristic | 10.5% | — | 26.2% | 100.0% |
| opus | 11.6% | 42.2% | — | 100.0% |
| random | 0.0% | 0.0% | 0.0% | — |

### 4 players

| Agent | astra | heuristic | opus | random |
| --- | ---: | ---: | ---: | ---: |
| astra | — | 56.2% | 50.1% | 100.0% |
| heuristic | 9.5% | — | 23.4% | 100.0% |
| opus | 11.7% | 25.1% | — | 100.0% |
| random | 0.0% | 0.0% | 0.0% | — |

## Mixed-table win shares

Each row uses a distinct-agent table and all seat rotations. Percentages split shared victories and sum to 100% within the table. Parentheses show 95% block-bootstrap intervals.

| Players | Participants | Games | Astra | Opus | Heuristic | Random |
| --- | --- | ---: | --- | --- | --- | --- |
| 3 | astra, heuristic, opus | 192 | 48.4% (41.9%–54.9%) | 41.7% (35.7%–47.9%) | 9.9% (5.7%–14.6%) | — |
| 3 | astra, heuristic, random | 192 | 90.6% (85.9%–94.8%) | — | 9.4% (5.2%–14.1%) | 0.0% (0.0%–0.0%) |
| 3 | astra, opus, random | 192 | 89.3% (84.9%–93.5%) | 10.7% (6.5%–15.1%) | — | 0.0% (0.0%–0.0%) |
| 3 | heuristic, opus, random | 192 | — | 55.2% (47.7%–62.8%) | 44.8% (37.2%–52.3%) | 0.0% (0.0%–0.0%) |
| 4 | astra, heuristic, opus, random | 256 | 67.0% (61.5%–72.5%) | 25.8% (21.1%–30.7%) | 7.2% (3.9%–10.7%) | 0.0% (0.0%–0.0%) |

## Provenance and reproduction

See [RANKING_PROTOCOL.md](RANKING_PROTOCOL.md) for the fixed schedule, outcome definitions, and resume commands. Raw games, frozen sources/native extension, checkpoint bytes, and hashes are in `agent/runs/astra-ranking/full/`. `ranking-uncertainty.json` preserves the original raw fit and bootstrap intervals. `ranking-display-v1.json` adds the display scale, converted intervals, and copies of the raw values. The original Astra tests and training leagues were preserved.

Neural participants:

- `league_2p`: `net:league:294`, checkpoint SHA-256 `538cafaa87deb5b2a13cc3e2370f57584bb4074a9f404803cbc8e9cb8f842c71`. Its registered search profile uses 64 simulations with a two-second deadline for this campaign.
- `trained_2p`: `trained_2p`, checkpoint SHA-256 `f595a14ee77d28ac36eb1d2bc42f45d2108c911bf13326fbebeda346aceb727a`. Its registered search profile uses 64 simulations with a two-second deadline for this campaign.

## Interpretation and limits

Astra ranked first at each player count in all 2,000 bootstrap resamples. The
two neural agents occupy overlapping rank intervals (second to third), so their
ordering is unresolved. Random lost every observed game; the absolute rating
gap to random is therefore set partly by the stated regularization prior. Use
the relative ranking and direct win-share tables when interpreting strength.

In the three-player table containing Astra, Opus, and heuristic, Astra's
win-share lead over Opus was 6.77 percentage points
(95% paired block interval -5.21 to 18.49 percentage points).
That individual mixed-table ordering is not statistically resolved. The overall
ranking describes this measured field and schedule; it is not a universal claim
about every opponent composition or future checkpoint.
