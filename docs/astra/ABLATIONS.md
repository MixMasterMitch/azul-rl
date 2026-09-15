# Finalist feature ablations

These exploratory comparisons remove one active term at a time, using 32 matched development seed/seat blocks against Opus, mixed tables, and the frozen search32 checkpoint. Intervals resample paired block differences and are not adjusted for multiple comparisons. Final holdout outcomes are not used here.

An ablation screen alone does not change production. For example, the initially positive two-player adjacency removal failed a separate 128-block promotion and was rejected.

## 2 players

Baseline: `ablate-2p-full`; development seed group 20, 32 complete blocks per field. 256,000 regular nodes; reply width 12; initiative 0.6.

| Removal / change | Opus share | Mixed share | Checkpoint share | Aggregate change, pp (95% CI) |
| --- | ---: | ---: | ---: | --- |
| full | 100.00% | 100.00% | 92.19% | +0.00 (+0.00 to +0.00) |
| Without adjacency | 100.00% | 100.00% | 94.53% | +0.78 (-2.34 to +4.17) |
| Without bonus | 96.88% | 98.44% | 88.28% | -2.86 (-6.77 to +1.04) |
| Without initiative | 100.00% | 100.00% | 89.06% | -1.04 (-4.17 to +2.60) |
| Without line lock | 100.00% | 98.44% | 93.75% | +0.00 (-3.12 to +3.14) |
| Without partial | 96.88% | 98.44% | 95.31% | -0.52 (-4.17 to +3.12) |
| Without safety | 97.66% | 100.00% | 87.50% | -2.34 (-5.73 to +0.52) |
| Without search | 46.88% | 77.34% | 42.97% | -41.67 (-48.96 to -34.11) |

## 3 players

Baseline: `ablate-multi-full`; development seed group 20, 32 complete blocks per field. 4,000 regular nodes; reply width 12; initiative 0.6.

| Removal / change | Opus share | Mixed share | Checkpoint share | Aggregate change, pp (95% CI) |
| --- | ---: | ---: | ---: | --- |
| full | 55.21% | 72.92% | 100.00% | +0.00 (+0.00 to +0.00) |
| Without adjacency | 54.17% | 59.90% | 100.00% | -4.69 (-10.42 to +0.87) |
| Without bonus | 33.33% | 49.83% | 100.00% | -14.99 (-20.31 to -9.72) |
| Without column prior | 36.46% | 52.08% | 100.00% | -13.19 (-19.44 to -6.94) |
| Without initiative | 52.08% | 73.96% | 100.00% | -0.69 (-4.51 to +2.78) |
| Without line lock | 56.25% | 65.10% | 100.00% | -2.26 (-6.94 to +2.43) |
| Without opponent | 43.23% | 52.08% | 100.00% | -10.94 (-15.28 to -6.42) |
| Without partial | 58.33% | 66.15% | 100.00% | -1.22 (-5.73 to +3.30) |
| Without safety | 53.12% | 67.71% | 100.00% | -2.43 (-5.21 to +0.35) |
| Without search | 47.92% | 56.25% | 100.00% | -7.99 (-13.89 to -2.08) |
| original horizon | 48.96% | 61.46% | 100.00% | -5.90 (-12.15 to +0.17) |

## 4 players

Baseline: `late-full-4p`; development seed group 31, 32 complete blocks per field. 4,000 regular nodes; reply width 12; initiative 1.2.

| Removal / change | Opus share | Mixed share | Checkpoint share | Aggregate change, pp (95% CI) |
| --- | ---: | ---: | ---: | --- |
| full | 54.69% | 63.28% | 100.00% | +0.00 (+0.00 to +0.00) |
| Without adjacency | 36.33% | 40.23% | 100.00% | -13.80 (-18.23 to -9.38) |
| Without bonus | 33.59% | 33.98% | 100.00% | -16.80 (-21.88 to -11.20) |
| Without column prior | 28.52% | 40.62% | 100.00% | -16.28 (-21.09 to -11.20) |
| Without initiative | 41.02% | 49.61% | 100.00% | -9.11 (-14.06 to -4.30) |
| Without line lock | 50.78% | 63.28% | 100.00% | -1.30 (-4.43 to +1.69) |
| Without opponent | 39.71% | 48.05% | 100.00% | -10.07 (-15.41 to -4.69) |
| Without partial | 46.09% | 64.06% | 100.00% | -2.60 (-6.77 to +1.95) |
| Without safety | 57.03% | 62.89% | 100.00% | +0.65 (-2.08 to +3.26) |
| Without search | 50.78% | 57.81% | 100.00% | -3.12 (-7.42 to +1.30) |
| original horizon | 37.89% | 50.39% | 100.00% | -9.90 (-14.19 to -5.47) |

The complete configuration and individual field deltas/intervals are retained in the ablation JSON files and experiment manifests. Larger search budgets were separate experiments; the rejected conditional terminal budget remains disabled.

The four-player safe-placement removal also received a fresh 128-block promotion:
its aggregate improvement remained inconclusive (+0.81 pp, 95% CI −0.55 to +2.18),
so it was rejected despite improving the homogeneous Opus matchup. The original
safe-placement term remains in the frozen configuration.
