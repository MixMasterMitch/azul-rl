# Rating display scale v1

The selected scale is `heuristic-2500-v1`:

```text
display_rating = 1000 + multiplier[players] * (raw_rating - 1000)
```

The first 1,000 points remain fixed. Full-precision factors put the frozen
reference heuristic at exactly 2,500; rounding happens only when displaying the
final result. The factors are not refitted when another evaluation finishes.

| Players | Frozen raw heuristic reference | Multiplier |
| --- | ---: | ---: |
| 2 | 5010.883103143318 | 0.3739824775308096 |
| 3 | 5828.328052855152 | 0.3106665461790650 |
| 4 | 6085.083562651204 | 0.2949804032754079 |

The reference is the completed 19,456-game ranking campaign. Its bootstrap
point estimates supply the constants in `agent/train/rating_display.py`.
The [ranking report](RANKINGS.md) now displays this scale:

| Agent | 2p | 3p | 4p |
| --- | ---: | ---: | ---: |
| Astra | 3118 | 2638 | 2605 |
| League RL checkpoint 294 | 2860 | — | — |
| Trained RL checkpoint | 2859 | — | — |
| Opus | 2579 | 2547 | 2526 |
| Heuristic | 2500 | 2500 | 2500 |
| Random | 1000 | 1000 | 1000 |

The two neural participants have overlapping rank intervals. No multiplayer
neural rating was inferred from their two-player results.

## Statistical evidence and presentation

The Bradley–Terry fit, its odds scale and prior, raw ratings, game outcomes,
win shares, and per-format rank order are unchanged. Confidence interval endpoints
receive the same affine transformation as ratings. Rating **differences** and
their intervals are multiplied by the factor without adding 1,000.

Use the raw ratings to compute expected outcomes. The display numbers are not
inputs to the original Bradley–Terry probability formula. Aligning the reference
heuristic across formats is a convention; it does not establish equivalent human
skill across player counts. The original statistical limitations still apply.

`ranking-uncertainty.json`, `report.json`, the original archive, frozen sources,
and raw games remain unchanged. A separate `ranking-display-v1.json` contains
the display values, raw values, scale metadata, and the SHA-256 of its source
bootstrap artifact. To regenerate the display report without new matches or
bootstrap fitting:

```bash
python -m agent.scripts.report_agent_ranking agent/runs/astra-ranking/full \
  --reuse-uncertainty --markdown docs/astra/RANKINGS.md
```

This command checks that the saved bootstrap evidence matches the game records.
Existing interpretation notes are retained when reusing that evidence.

## Legacy league and play ratings

The production registry and training leagues predate this campaign and have
different raw reference ratings. Applying 0.374/0.311/0.295 directly to those
numbers would put their heuristic below 2,500. Their display factors therefore
use the **same rule against their own stored raw references**:

```text
source_multiplier[players] = 1500 / (source_reference_heuristic[players] - 1000)
```

Each source's random remains 1,000 and reference heuristic becomes 2,500.
Different evaluation datasets can still give the same checkpoint different
ratings; display normalization does not merge their evidence.

The bot catalog, human profiles, leaderboard, legacy local human-rating store,
and `azul-league` use this display conversion. The registry's `raw_ratings` and
all stored statistical anchors retain their original units. League rows that
already contain the old Opus-based calibration are converted back to their raw
source basis before applying v1. Combined display ratings are recomputed from
per-format values with the existing weighting rule.

New human result records persist raw per-format estimates, the reference table,
and the display version. Existing hosted profiles are converted on read without
changing game counts, results, or stored records. Their previous reference basis
comes from the serving catalog; new records retain it explicitly for future
version changes. Profiles and leaderboards use the same conversion.

The training fit and optimization metrics retain the existing internal scale;
old experiment records and objectives are not retroactively rewritten. Inspect
the league on the new display scale, or request the original stored values:

```bash
azul-league --league-dir agent/runs/league
azul-league --league-dir agent/runs/league --legacy-ratings
```

For application code:

```python
from agent.train import rating_display as D

# Raw values from the frozen ranking-campaign basis.
shown = D.to_display(raw_rating, players)

# Raw values from a registry/league with its own reference anchors.
shown = D.to_display(raw_rating, players, D.scales_for(reference_anchors))
```

## Verification

Regression tests cover both endpoints, exact factors, monotonicity, inversion,
finite inputs, preservation of modeled probabilities, uncertainty conversion,
idempotent report rendering, source hashes, legacy calibration reversal, weighted
combined ratings, matching bot/human displays, JSON persistence, and placement
visibility. The original raw campaign artifacts and production league/registry
were checked byte-for-byte after generating the new presentation.

Validation completed with 337 passing agent/play tests and 30 skipped tests in
the CPU-only run, plus a successful frontend production build. The focused
rating/report/play checks passed 64 tests. No live AWS deployment was performed.
