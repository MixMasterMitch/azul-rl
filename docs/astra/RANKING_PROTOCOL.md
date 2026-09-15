# Astra ranking campaign

This is additional evaluation of the frozen Astra 1.0.0 configurations, following
the original development and final test. It does not tune Astra or reuse the
original final seeds. The purpose is to establish relative rankings among the
available agents with direct comparisons across the field.

## Participants and search

- Astra: unchanged production configuration, 256,000 nodes in 2p and 4,000 in
  3p/4p, at most 1,950 ms native search.
- Opus, basic heuristic, and random: source frozen at campaign creation.
- `league_2p`: the published league champion's exact checkpoint and serving
  search profile, as selected by `play/artifacts/registry.json`.
- `trained_2p`: the competitive registry's current playable checkpoint and
  serving search profile.

Both neural agents retain their registered 64-simulation search settings, with
an explicit two-second move deadline. The original and effective settings,
checkpoint hashes, model IDs, and registry paths are retained in the manifest.
Only the player counts declared trained by each registry are eligible. The
current neural models support 2p; untrained multiplayer heads are excluded.

## Fixed schedule

The current six-agent 2p/four-agent multiplayer field produces **19,456 games**:

| Section | Schedule | Games |
| --- | --- | ---: |
| 2p | All 15 unordered pairs, 256 independent seeds, both seats | 7,680 |
| 3p | All 12 ordered minority-versus-two tables, 128 seeds, all seats | 4,608 |
| 4p | All 12 ordered minority-versus-three tables, 128 seeds, all seats | 6,144 |
| 3p mixed | All four distinct-agent triples, 64 seeds, all seats | 768 |
| 4p mixed | All four agents, 64 seeds, all seats | 256 |

Multiplayer includes both A-versus-many-B and B-versus-many-A; those are different
table compositions. All complete tables rotate, preserving relative turn order.
Seeds begin at 902,000,000, beyond all previous Astra development/promotion/final
ranges. Draw seeds and bot seeds are separate. Every table in a seed block shares
the draw seed; analyses resample the complete block across those tables together.
Mixed tables use a separate seed range. Smoke-test seeds are the first block;
smoke outcomes are used only for runner validation, never agent selection.

The explicit PyTorch reference engine owns actual draws and RNG. Search gets
only the public snapshot. Four local worker processes use one PyTorch thread
each. No GPU or live AWS deployment is involved. Source, native extension,
configuration, and checkpoints are frozen and hashed before the full campaign.

## Outcomes and ranking

Primary matchup tables report win share: 1 for a sole win, 1/k for a k-way shared
victory. Use the official score, completed-horizontal-row, shared-victory order.
Bootstrap 95% intervals resample complete seed/seat blocks. Intervals from
all-success observations can collapse; they do not prove perfect play.

Fit fresh Bradley–Terry ratings separately by player count, anchored at
`random=1000`, using pairwise final score/row placements and half-credit ties.
Use the existing rating scale of 1000 per base-10 odds unit, with the explicitly
recorded weak Gaussian prior (mean 1500, sigma 10000) to keep separated results
finite. Report ranking uncertainty by whole-block bootstrap. These fresh ratings
are not comparable to historical league values or the original narrower field.
Multiplayer pairwise placements are secondary to actual whole-game win share.

The raw fitting protocol above remains unchanged. Current presentation uses
the [heuristic-2500-v1 display scale](RATING_SCALE.md); its fixed affine
transformation is applied after fitting and to both confidence interval endpoints.
The original raw bootstrap artifact and game records are retained.

Capped games retain an unfinished outcome and no winner. Failures, illegal
actions, incomplete blocks, and time cutoffs are reported explicitly. Do not
publish a definitive ranking when unexplained completion failures remain.

## Reproduction

```bash
.venv/bin/python -m agent.scripts.rank_agents \
  --output agent/runs/astra-ranking/full --workers 4

.venv/bin/python -m agent.scripts.rank_agents \
  --output agent/runs/astra-ranking/full --resume --workers 4
```

Resume verifies archived hashes and environment versions before loading worker
code, validates every recorded game against the schedule, and plays only missing
records. Original Astra experiment records and the active training leagues are
preserved. The campaign has its own raw records, ranking table, and report.
