# Training journal

Last reviewed: **October 1, 2026**. Training is **stopped at the user's request**.

This is the durable record of the RL training experiments recoverable from this
workspace: six May–June tuning studies, the v3/v4 runs, and the September–October
campaigns. It records unsuccessful and interrupted attempts as well as improvements.
Dates in campaign names identify their run directories; a run can finish on a later
day. Explicit clock times below use America/Los_Angeles.

The objective is the strongest two-player agent within approximately **two seconds
per turn**. Training speed matters as an experiment cost, not as the final objective.
Three- and four-player neural training remain future work. Astra's separate search
and heuristic experiments are recorded in the [Astra journal](astra/REVISIONS.md),
[ablations](astra/ABLATIONS.md), and [results](astra/RESULTS.md).

## How to read the results

- Unless labelled otherwise, a percentage is **match score**: win = 1, shared
  victory = 0.5, loss = 0. It is not the sole-win percentage. Training's binary
  reward has been +1 / 0 / −1 since the September 20 reward change.
- Confidence intervals are 95% intervals from the recorded paired evaluations.
  “Resolved” means the relevant interval clears the stated threshold. “Provisional”
  means a useful point estimate without that level of confirmation. Neither is a
  claim of robustness across training seeds; most studies used one seed.
- Development screens choose candidates. Independent confirmation is reported
  separately. A higher Astra score from another campaign, seed set, native build,
  or search budget is not a measured improvement over the current baseline.
- One-ply candidates, tree simulations, greedy policies, equal-search comparisons,
  and CPU-qualified serving profiles are different protocols. CPU timing is local
  Ryzen 5800X timing, not an AWS Lambda measurement. GPU-batched match evaluations
  do not themselves verify a two-second CPU turn budget.
- A weights-only warm start, a donor-replay fork with a fresh optimizer, and an
  exact optimizer/replay/model resume are different initializations. Earlier
  weights must never be paired with a later checkpoint's optimizer as an exact resume.
- A stopped process or missing report is not a negative strength result. A smoke
  test or sanitizer run is not a win-rate experiment. No automatic deployment was
  performed by the later campaigns.

The linked `agent/runs/` evidence and checkpoints are local and Git-ignored. The
results in this journal and the [87-trial tuning export](training-studies.json)
are version-controlled so the conclusions survive removal of local scratch data.

## Current working baseline and unresolved work

The last selected working baseline is the **width-256 source-attention model**,
descended from capacity study `control_60m`. Its recorded archive identities are:

- Historical weights: `agent/runs/competitive_distillation_resume_12h_20260928/initializers/small/weights.pt`,
  SHA-256 `6c9f4022f86578aecaeb5155f53aeb973700316af10be46cc53fdfeada8d31f1`.
- Historical matching full state: `agent/runs/competitive_distillation_resume_12h_20260928/initializers/small/resume.pt`,
  SHA-256 `ee6a6134170ac5fffd14eb8987556a76955387016e2661cd6370ff57fcac664a`.
- [Matching league](../agent/runs/competitive_distillation_resume_12h_20260928/initializers/small/league/league.json).

**Archive availability at this review:** those two original checkpoint files and
their baseline/resume copies in the surprise initializers are no longer present.
Their identities remain in the [verified preparation manifest](../agent/runs/competitive_surprise_20h_20261001/prepared.json).
The [surviving smoke initializer](../agent/runs/competitive_surprise_20h_20261001/smoke_fixed/checkpoints/latest_resume.pt)
loads successfully at iteration 54 with zero additional training time. It was
forked from the verified baseline and retains its model, optimizer, and replay,
but has the smoke configuration and a reset RNG; it is not the original full-state
archive. Verify a recovery/fork explicitly before further training. This journal
does not recreate missing checkpoints or silently substitute later model weights.

This identifies the training baseline, not necessarily the model deployed by the
play service. It was retained after the September 28 confirmation; its own earlier
selection over its predecessor was provisional, not a resolved strength gain.

The inherited recipe uses 256 games per iteration, one million replay positions,
72 learner updates of batch size 256, learning rate 0.000921747, zero extra
Dirichlet mixture, a 64/256-simulation teacher with full search on 25% of
iterations, fast-target policy weight 0.25, and 256 reanalysed positions every four
iterations. Bot games mix 50% Astra, 25% Opus, and 25% heuristic. League opponents
use 64-simulation trees. Native inference caching is disabled after memory faults.

The most useful unresolved comparisons are the saved width-512 `continuation_180m`
candidate at 384 simulations versus the baseline at 1,024; a fresh confirmation
of the early auxiliary-score candidate; and the unrun matched policy-surprise
experiment. The initial heap-corruption cause remains unproven. None of these
pending questions authorizes restarting the stopped run.

## Experiment index

| Period / run | Question | Outcome |
|---|---|---|
| May 26–June 5, six Optuna studies | Learning/search/reward/opponent hyperparameters | Mixed historical objectives; 87 trial records preserved, including invalid and stale trials |
| June, `attn_256_v3` / `attn_256_v4` | Longer training with selected tuning recipes | v4 became the corrected September reference; old league ratings are not current strength evidence |
| September 13, `competitive` | Pooled attention versus source-aware policy | Source attention selected for experiments; promotion gate not met |
| September 14, learner 2h | 72 versus 144 learner updates | Retain 72; no confirmed promotion |
| September 14, search 2h | One-ply versus multi-step tree search | Large resolved search benefit; tree64 confirmed |
| September 15, tree distillation 8h | Train on tree-search policy targets | Iteration-500 candidate beat its initializer at equal tree64 |
| September 15–16, native tree 16h | Longer Rust-tree training | Stronger Astra point score; initializer comparison inconclusive |
| September 19, noise ablation | Remove extra Dirichlet noise | Zero mixture selected provisionally |
| September 20, LR 10h | Current LR versus 0.0003, then continuation | Current LR retained; resolved head-to-head gain |
| September 20, enhancements 8h | Stronger targets, opponents, reanalysis, tie correction | Bundle improved strength; individual contributions not isolated |
| September 21, reanalysis 12h | Four times as many refreshed positions | Larger reanalysis pilot inconclusive; current recipe retained |
| September 22, league 10h | Refresh opponent ratings with measured matches | No established gain; starting weights retained |
| September 22, fine-tuning 24h | Lower LR or all-full-search teacher | Current recipe won pilot; user paused; later confirmation inconclusive |
| September 23, policy weights 10h | Downweight fast-search policy targets | +1.48pp panel estimate, interval crosses zero |
| September 24, weight refinement 8h | Fast weight 0.25 versus 0.5 | Early 0.25 checkpoint adopted provisionally; longer training regressed |
| September 24, capacity 20h | Widen 256 → 512 | Equal-search gain; serving comparison limited by profile selection |
| September 26, distillation 24h | Train small student from larger search teacher | Early student regressed; stopped for shutdown |
| September 28, resumed 12h | Continue student or larger model | Student rejected; larger finalist failed confirmation; baseline selected |
| September 28, auxiliary score 16h | Add final score-margin supervision | Early candidate promising but unconfirmed; native abort before final tests |
| September 29, surprise 20h | Reliability mitigation, then biased replay study | Main campaign never launched; diagnostics only |
| October 1, surprise restart 20h | Validate revised Rust binary, then run study | User stopped during smoke training; no campaign comparisons |

## May–June: tuning and initial neural training

Read-only SQLite extraction produced [training-studies.json](training-studies.json).
It includes every retained trial's state, parameter values (categorical indices
decoded), objectives, observed values, projection attributes, and source database
hash. `RUNNING` below is a stale database state, not a live training process.

| Study (`agent/runs/optuna_*.db`) | Recorded trials | Best recorded objective / interpretation |
|---|---|---|
| `azul-tune` | 43 complete, 8 pruned, 1 stale running | Trial 10: combined win rate 33.59%; random 98.05%, heuristic 1.17%, Opus 1.56%. Ten completed trials have negative-infinite objectives. |
| `azul-tune-curve-1023x32` | 12 complete | All objectives negative infinity; no usable winning result. |
| `azul-tune-curve-1023x32-r2` | 5 complete, 1 stale running | Trial 4: projected rating 2175.18 versus last observed 2169.63. Projection was not a measured 72-hour result. |
| `azul-tune-i1325-3h-v1` | 2 complete, 1 failed | Trial 0: projected rating 3013.97; last observed 3056.77. Different league context; do not compare to the preceding study's scale. |
| `azul-tune-v2-i1950-opus-2h-v1` | 4 complete, 1 failed | Trial 3: projected Opus win rate 53.53%; last observed 50.78%. Its recipe initialized v3. |
| `azul-tune-v3-focused-opus-v1` | 9 complete | Trial 8: projected Opus win rate 86.84%; last observed 83.59%. Its recipe initialized v4. |

The searches varied learning rate, decay, entropy, learner batch/update count,
Dirichlet concentration/mixture, Q scale, time discount, and sometimes binary versus
score-scaled rewards, self-play batch/search size, bot mixture, and cycle length.
These were joint hyperparameter studies, not isolated causal tests of each option.
Current tuning defaults to observed final evaluation rather than extrapolated curves.

`attn_256_v3` used width 256, 1,023 games × 32 candidates, 74 learner updates,
LR 0.000974602, binary rewards, and the four-phase training cycle. Its saved state
reached iteration 5700. `attn_256_v4` warmed from that model with 72 updates,
LR 0.000921747, entropy 0.01427496, and Opus bot probability 0.3680; its saved state
reached iteration 2350. Logs continue beyond those last saved iterations.
The September corrected screens, rather than historical league-rating peaks,
established the subsequent reference: v3/5700 scored 83.20% against Opus,
v4/2350 85.74%, and v4/1350 83.59% (256 games each, one-ply64).

Evidence: [v3 config](../agent/runs/attn_256_v3/config.yaml),
[v4 config](../agent/runs/attn_256_v4/config.yaml),
[corrected baseline](../agent/runs/competitive/baseline.json), and
[baseline protocol and numerical-stability investigation](competitive-training.md).

## September 13–16: architecture, search, and tree training

### Source-aware policy architecture

Equal-budget `attn` and `source_attn` arms warmed from v4 weights with fresh replay
and optimizers. Source attention preserves individual source embeddings and factory
permutation symmetry. In development (256 games), source attention scored 54.69%
against the champion versus 51.76% for pooled attention, and greedy Opus scores
were 83.79% versus 72.66%. Independent source-attention confirmation against the
champion was **51.51% [48.39, 54.64]**, so promotion was rejected. Source attention
became the experimental architecture; this was not a confirmed champion replacement.

A failed fused-attention-backward attempt was reproduced and replaced with explicit
softmax pooling. The same problematic batch's gradient norm fell from roughly
249,065 to 1.30. This was numerical repair, not a strength experiment.

Evidence: [architecture results](../agent/runs/competitive/policy.json) and
[recovery details](competitive-training.md#september-14-recovery-and-quiet-supervision).

### Learner work per iteration — two hours

Matched 50-minute arms compared **72 versus 144 updates**, using the same
source-attention initializer and fresh optimizer/replay. The 72-update arm led the
development champion comparison (54.69% versus 47.07%, 256 games each), while
Astra scores were only 21.29% and 21.88% with the old one-ply evaluation. Its
1,024-game confirmation scored **52.83% [49.85, 55.76]** against the champion.
Retain 72; no confirmed promotion. Although generic tooling offers 288 updates,
the saved two-hour experiment tested only 72 and 144.

Evidence: [results](../agent/runs/competitive_learner_2h_20260914/learner.json),
[frozen plan](../agent/runs/competitive_learner_2h_20260914/experiment_plan.json).

### Multi-step search — two hours

Fixed weights compared one-ply64 and tree search at 64, 256, and a throughput
probe at 1,024 simulations. Tree256 led development, but the available confirmation
budget supported tree64; the full tree1024 development screen was explicitly skipped.
Independent 1,024-game scores: tree64 **46.83% against Astra**, versus **18.99%**
for one-ply; paired gain **+27.83pp [24.17, 31.59]**. Tree64 scored **87.70%
[85.74, 89.60]** against the old champion and 97.41% against Opus. The search gate
passed. This isolates search improvement, not newly learned network strength.

Evidence: [search results and timing](../agent/runs/competitive_search_2h_20260914/search.json).

### Tree-policy training and native traversal

The eight-hour tree-distillation run trained on 64-simulation search targets.
An eight-worker Python-tree amendment reduced a 256-game self-play benchmark from
129.46s to 51.18s (**2.53×**). The evaluated iteration-500 model scored **63.62%
[60.84, 66.41]** against its initializer with equal tree64 (1,024 games). Astra
tree64 scores were 51.46% versus 47.56% for the initializer (512 games each);
greedy Astra scores were 31.64% versus 17.77% (256 each). Saved state reached 511.

The subsequent sixteen-hour run used native Rust tree traversal with a single
search worker and CUDA-batched network inference, starting from the preceding
model. It reached iteration 2314. Its finalist scored **62.11% [57.81, 66.31]**
against Astra (512 games), **54.49% [48.83, 60.16]** against its initializer
(256 games, inconclusive), and 37.30% greedy against Astra (256 games).
The final status says stopped, but the experiment state marks its finalist complete;
the recorded evaluations, not that status label alone, support these conclusions.

Evidence: [parallel amendment](../agent/runs/competitive_tree_distill_8h_20260915/training_amendment_result.json),
[iteration-500 confirmation](../agent/runs/competitive_tree_distill_8h_20260915/evaluations/iter500_vs_initializer_tree64_1024.json),
[native run evaluations](../agent/runs/competitive_native_tree_16h_20260915/evaluations/),
[Rust simulator measurements](../agent/runs/rust_benchmark/).

## September 19–22: exploration, targets, and opponents

### Extra Dirichlet noise — two one-hour arms

The existing mixture (about 0.5332) was compared with zero extra Dirichlet mixture;
Gumbel search itself remained. Each arm received identical initial weights and fresh
optimizer/replay. At tree64, zero-noise versus control scored 53.71% versus 49.41%
against the frozen finalist, and 59.96% versus 55.86% against Astra (256 games per
screen). The primary difference was **+4.30pp [−3.91, 12.30]**. Greedy results also
favored zero mixture. It was recommended provisionally, not statistically resolved,
and subsequent training retained zero extra mixture.

Evidence: [noise study](../agent/runs/competitive_noise_ablation_20260919/noise_ablation.json).

### Learning rate and continuation — ten hours

One-hour pilots compared LR 0.000921747 with 0.0003. The lower rate did not establish
an advantage. The current-rate arm trained for 7h29m total to iteration 1091;
the complete campaign used 9h34m. Its frozen finalist scored **58.01%
[55.1, 60.9]** against its start (1,024 games, equal tree64). Astra scores were
62.79% versus 61.52%, a **+1.27pp [−2.7, 5.2]** change; that difference was
unresolved. Greedy Astra scores were 37.7% versus 34.0% (256 games).
There were 89,064 successful updates across both arms and no skipped updates.

Evidence: [full report](../agent/runs/competitive_lr_10h_20260920/REPORT.md).

### Enhancement bundle — eight hours

This bundle added 256-simulation targets on 25% of iterations (otherwise 64),
32 root candidates, stronger/more varied league opponents, 50% Astra in bot phases,
and 256 reanalysed replay policies every four iterations. It also implemented the
requested shared-victory reward of zero, search profiles, and bounded inference
caching. **Both comparison arms used the corrected tie semantics and fresh replay**;
the reward correction was not separately ablated. Evaluation settings were frozen
for the comparisons, rather than mixing old and new serving profiles.

The enhanced pilot's advantage over control against the frozen start was
**+15.43pp [7.03, 23.44]** (256 games per arm). The selected 290-minute checkpoint then scored
**55.8% [52.8, 58.6]** against its start and **72.7%** against Astra versus
**62.0%** for the start (1,024 games each). Astra gain **+10.7pp [6.8, 14.6]**.
The full run used 6.98h. This is evidence for the bundle; it does not identify
which individual enhancement caused the gain.

Evidence: [report](../agent/runs/competitive_enhancements_8h_20260920/REPORT.md),
[arm selection](../agent/runs/competitive_enhancements_8h_20260920/arm_selection.json),
[implementation details](training-enhancements.md).

### Reanalysis volume — twelve hours

Matched pilots compared **256 versus 1,024 reanalysed positions every four
iterations**, keeping the search budget at 256. The larger refresh had a
**+4.00pp [−2.44, 10.55]** primary pilot estimate (512 games per arm), so the predeclared gate kept
the current 256-position recipe. Continuing it produced a 495-minute finalist
that scored **54.4% [51.5, 57.4]** against its frozen start (1,024 games).
However, Astra score was **71.6% versus 74.8%**, a **−3.2pp [−7.0, 0.7]** change.
This is a head-to-head improvement with a concerning, unresolved Astra tradeoff,
not proof that more reanalysis helps or that the agent improved against all opponents.
The full campaign used 11.06h.

Evidence: [report](../agent/runs/competitive_reanalysis_12h_20260921/REPORT.md),
[pilot comparison](../agent/runs/competitive_reanalysis_12h_20260921/arm_selection.json).

### Measured league ratings — ten hours

Identical full-state forks compared existing league sampling with periodic measured
rating refreshes. The rated arm paid the rating matches' cost within its budget;
these matches used 128 games per pair. The pilot's panel change was only
**+0.20pp [−3.19, 3.58]** (512 games per opponent per arm). The current arm continued, but the initializer remained
the best selected candidate. Final candidate and starting panel scores were therefore
identical: 51.2% versus start, 58.4% versus the previous model, and 72.8% versus
Astra (1,024 games per opponent). No gain was established; the campaign used 9.02h.

This campaign also encountered a native heap-corruption abort during iteration 487.
The subsequent investigation found the allocator detected corruption during inference
cache eviction; it did not establish that eviction caused the bad write. Later
campaigns disabled the optional cache and stopped automatic retries on native faults.

Evidence: [report](../agent/runs/competitive_league_10h_20260922/REPORT.md),
[plan](../agent/runs/competitive_league_10h_20260922/league_campaign_plan.json),
[memory investigation](../agent/runs/competitive_finetune_24h_20260922/RELIABILITY.md).

## September 22–24: fine-tuning and policy-loss weights

### Lower LR or deeper teacher — planned twenty-four hours, then paused

Full-state forks compared current settings, LR 0.0003, and 256-simulation search
on every iteration instead of 25%. All used the same cache-disabled mitigation.
Pilot panel changes versus current were **−3.26pp [−6.45, −0.03]** for lower LR
and **−8.82pp [−11.75, −5.79]** for all-full search. Current settings continued.
The user paused the run at iteration 1094 before independent confirmation;
67,176 campaign learner updates completed with no skips.

The next window independently evaluated the retained milestones. `pilot_current`
led development, but its held-out panel gain was only **+1.36pp [−0.24, 2.95]**.
Starting weights were retained. The paused late checkpoint was not retrospectively
declared the best merely because it had trained longest.

Evidence: [pause record](../agent/runs/competitive_finetune_24h_20260922/pause.json),
[pilot decision](../agent/runs/competitive_finetune_24h_20260922/arm_selection.json),
[independent closeout](../agent/runs/competitive_policy_weight_10h_20260923/closeout/REPORT.md).

### Fast-target policy weight — ten hours

Following that closeout, matched two-hour pilots compared fast-search policy-loss
weight **1.0 versus 0.25**. Full-search targets retained weight one; old targets
with unknown search budget retained neutral weight. Values were unchanged.
The early `weighted_1` checkpoint led development with a 59.5% panel score;
later `weighted_2` fell to 58.7% and 68.3% against Astra. Independent panel gain
was **+1.48pp [−0.20, 3.18]** (2,048 games per opponent). The strict gate retained the previous baseline.
The positive estimate remained a plausible improvement and motivated the next study;
an interval crossing zero was not interpreted as proof of no benefit.

Evidence: [report](../agent/runs/competitive_policy_weight_10h_20260923/REPORT.md),
[results](../agent/runs/competitive_policy_weight_10h_20260923/policy_weight_campaign.json).

### Refinement of policy weighting — eight hours

The selected one-hour weighted candidate initialized **0.25 versus 0.5** fast-weight
arms, with identical tagged donor replay, fresh optimizer, and fresh matched RNG.
This was explicitly **not an exact resume**. Each arm trained 150 minutes with
30/60/150-minute screens. Longer training regressed: quarter scores against the
starting candidate were 50.9%, 46.2%, 44.9%; half scores were 48.7%, 47.6%, 41.7%.

`quarter_30m` was selected. Held-out head-to-head was **50.07% [47.92, 52.27]**
(2,048 games), while Astra scores were **73.66% versus 70.97%** (2,048 each), a
paired change of **+2.69pp [0.05, 5.32]**. Greedy Astra point scores did not improve:
43.26% versus 44.53% (512 each). It was recommended as
a **provisional** improvement, not a resolved head-to-head gain. This established
the later practice of preserving promising early checkpoints and permitting an
explicitly labelled provisional choice rather than requiring every interval to
clear zero.

Evidence: [report](../agent/runs/competitive_weight_refine_8h_20260924/REPORT.md),
[full results](../agent/runs/competitive_weight_refine_8h_20260924/weight_refine_campaign.json).

## September 24–28: model capacity and distillation

### Width 256 → 512 — twenty hours

The study widened the source-attention network with a near-function-preserving
transfer and symmetry breaking, compared 512-wide current-LR and lower-LR pilots,
and retained a 256-wide control. The lower-LR wide arm supplied the continued
candidate. Training across arms used about 11h42m; setup and evaluation brought the
campaign to 16h45m, with 42,768 finite updates.

At **equal 64 simulations**, the wider finalist scored **52.73% [50.63, 54.83]**
against the smaller finalist (2,048 games). Its selected serving profile instead
used only **128 simulations against the smaller model's 1,024**: **34.33%
[31.54, 37.16]** (1,024 games), with Astra 79.59% versus 83.11%.

The serving conclusion needed qualification. The larger model completed all 512
simulations in all 24 sampled positions, with p95 1.7959s and maximum 1.8023s.
A strict 1.800s cutoff rejected it despite every position being under the user's
two-second budget. Small, noisy Astra screens then selected 128 simulations.
Thus the 34.33% result was a real loss for those **selected configurations**, not
a definitive rejection of larger networks at the best feasible two-second budget.

`control_60m` became the provisional smaller-model baseline. It scored 50.59%
[46.39, 54.69] against the preceding teacher at equal 1,024 simulations (512 games),
so that replacement itself was not statistically resolved. Later code separates
the internal search deadline from the outer two-second wall budget.

Evidence: [review correcting the serving interpretation](../agent/runs/competitive_capacity_20h_20260924/RESULTS_REVIEW.md),
[raw results](../agent/runs/competitive_capacity_20h_20260924/capacity_campaign.json),
[widening protocol](capacity-training.md).

### Larger teacher → smaller student — planned twenty-four hours

Fresh CPU qualification compared the saved wider model at 384 simulations with
the smaller model at 1,024. The wider model's initial head-to-head was **47.46%
[41.02, 53.71]** (256 games); the smaller baseline remained. A 1,280-versus-1,024
small-model probe scored **52.54% [46.09, 58.98]**, so extra simulations were not
adopted on that evidence.

The distillation treatment used a frozen width-512 teacher with 1,024 simulations,
no root noise, and 50% teacher policy minibatches. Value targets remained observed
game outcomes. The 30-minute student fell to **31.64%** against the small baseline
(256 games), and 75.0% against Astra versus the small reference's 89.06%
(128-game development screens). The user stopped for computer shutdown at
iteration 108, after 88.5 recorded training minutes; the saved state included one
million replay positions and 27,648 teacher positions. No final verdict was possible
from the interrupted campaign alone.

Evidence: [plan](../agent/runs/competitive_distillation_24h_20260926/plan.json),
[development record](../agent/runs/competitive_distillation_24h_20260926/development.json),
[verified shutdown state](../agent/runs/competitive_distillation_24h_20260926/STOPPED_FOR_SHUTDOWN.json).

### Resumed student versus larger-model training — twelve hours

The resumed student's 90-minute milestone scored only **25.20% [20.31, 30.27]**
against the baseline (256 games) and 71.88% against Astra (128 games). The student
was stopped. Wide-model training continued instead; development head-to-head scores
were 43.95% at 60 minutes, 46.09% at 150 minutes, and 44.53%, 45.51%, 51.37%,
48.44% after successive continuation milestones of 60/120/180/240 minutes.

The old Astra point-score guard excluded the otherwise promising
`continuation_180m`, because its 128-game Astra score was 83.20% against an optimistic
89.06% baseline screen. `wide_150m` went to independent confirmation instead:
**48.68% [45.70, 51.66]** against the small baseline (1,024 games, 384 versus
1,024 simulations), and **83.74% versus 85.16%** against Astra. Paired Astra
change **−1.42pp [−4.35, 1.51]**. Actual CPU timing passed; the small baseline
was retained. The run used about 10.12h.

This rejected the tested aggressive distillation recipe and did not establish a
wide-model win. It also motivated a better-powered fresh evaluation of the saved
180-minute wide candidate rather than discarding it on a noisy development guard.

Evidence: [report](../agent/runs/competitive_distillation_resume_12h_20260928/REPORT.md),
[development trajectory](../agent/runs/competitive_distillation_resume_12h_20260928/development.json),
[selection and confirmation](../agent/runs/competitive_distillation_resume_12h_20260928/distillation_resume.json),
[distillation design](distillation-training.md).

## September 28–October 1: auxiliary targets, reliability, and surprise

### Auxiliary final score-margin prediction — sixteen hours

Matched full-state width-256 forks added the same dormant score head, comparing
auxiliary loss weight **0 versus 0.1**, score scale 50, and smooth-L1 supervision
from final score margins. Binary value targets and win-oriented search were unchanged.
Unknown historical score labels were masked. Both pilots completed 102 iterations
and 7,344 updates; the score continuation completed 94 iterations and 6,768 updates.

| Development checkpoint | Head-to-head against baseline | Games |
|---|---:|---:|
| control_30m | 46.48% | 128 |
| control_120m | 45.51% | 256 |
| score_30m | 51.56% [42.97, 60.16] | 128 |
| score_120m | 46.09% | 256 |
| score continuation_60m | 47.85% | 256 |

The independent saved-wide **development/closeout** comparison was more promising
against Astra: `continuation_180m` scored **49.85% [46.73, 52.93]** against the
baseline (1,024 games; wide384 versus small1024), and **87.40% versus 81.74%**
against Astra. The paired Astra gain was **+5.66pp [2.59, 8.79]**. This was not the
campaign's final frozen-candidate confirmation and did not resolve head-to-head superiority.

The campaign aborted with `corrupted size vs. prev_size` during continuation
iteration 175, after about 10h09m of the window. Total completed learner updates:
21,456; completed self-play games: 76,288. No final confirmation or model promotion
occurred. The early score-head point advantage remains unconfirmed.

Evidence: [development results](../agent/runs/competitive_aux_score_16h_20260928/development.json),
[saved-wide comparison](../agent/runs/competitive_aux_score_16h_20260928/saved_wide_result.json),
[supervisor failure record](../agent/runs/competitive_aux_score_16h_20260928/supervisor.json),
[auxiliary-score design](auxiliary-score-training.md).

### Reliability mitigation and policy-surprise preparation — September 29

The planned twenty-hour sequence was memory investigation, saved-candidate
confirmation, two matched policy-surprise pilots, continuation, and independent
final evaluation. **The main campaign never launched.** Bounded diagnostics ended
around 9:41 PM PDT on September 29, and most of the window went unused because
the campaign was not launched afterward. This was an execution failure, not an
inconclusive completed training experiment.

The implementation records `KL(search target || raw network prior)` for completed
positions searched with at least 256 simulations. Full-target sampling weight is
`min(4, 0.5 + 0.5 * KL / mean_eligible_KL)`; fast and unknown targets keep weight one.
Sampling changes both policy and value example frequencies without importance
correction. Existing fast-target policy-loss weighting is separate. No matched
control-versus-surprise win-rate evidence exists yet.

Completed diagnostics included a 1,280-game CUDA smoke run with 360 finite updates
on the **old binary**, and a separate old-binary continuation with 3,584 games and
1,008 finite updates. Core analysis found corruption in sparse-edge hash-table
metadata. The mitigation replaces each node's edge hash map with checked linear
storage. The initial corrupting write remains unidentified; cache disabling alone
had already failed to prevent recurrence.

The revised search passed nine old/new comparisons with identical actions/policies,
18 Rust tests, and a 30-minute AddressSanitizer stress covering **18,727,243 nodes**.
Valgrind runs reported no errors. These bounded checks neither prove the original
cause is fixed nor establish improved model strength.

Evidence: [incomplete-run report](../agent/runs/competitive_surprise_20h_20260929/REPORT.md),
[sanitizer result](../agent/runs/competitive_surprise_20h_20260929/fixed_asan_validation.json),
[search parity](../agent/runs/competitive_surprise_20h_20260929/native_comparison.json),
[sampling design](policy-surprise-training.md).

### Fresh twenty-hour restart — October 1, stopped by user

A new window ran from 7:22 AM PDT October 1, with a planned cutoff of 3:22 AM
October 2. A persistent systemd job connected validation directly to the campaign
supervisor. **505 agent tests passed.** The revised-binary smoke run completed
four iterations (through 58), logging **288 finite learner updates** across standard,
bot, and league self-play. The user stopped training, then requested force stop,
during the fifth iteration (59). Systemd terminated the remaining training process.

The main campaign never reached its ready marker, saved-candidate evaluations,
matched pilots, or final CPU audit. The preserved smoke resume file is the initial
iteration-54 fork, **not** a checkpoint containing the 288 logged updates. The
champion is unchanged. No training process or automatic restart remains active.
The proposed fresh twenty-hour study is still unexecuted as a strength experiment.

Evidence: [run plan](../agent/runs/competitive_surprise_20h_20261001/RUN.md),
[test result](../agent/runs/competitive_surprise_20h_20261001/pytest_result.json),
[smoke events](../agent/runs/competitive_surprise_20h_20261001/smoke_fixed/events.log),
[stop record](../agent/runs/competitive_surprise_20h_20261001/stop_request.json).

## What to carry into the next experiment

| Finding | Evidence supports | Still unresolved |
|---|---|---|
| Multi-step search | Large gain over one-ply with fixed weights | Best allocation within two seconds for each model |
| Source attention | Better development policy results; used by subsequent successful runs | Isolated replicated architecture gain |
| Current LR / 72 updates | Repeatedly competitive with tested alternatives | Other schedules, update ratios, and multi-seed replication |
| Enhancement bundle | Resolved head-to-head and Astra gain | Individual effects of reanalysis, opponent mix, target depth, root count, and reward correction |
| More reanalysis | Positive but unresolved pilot estimate | Whether increased refresh helps with other budgets or replay ages |
| Measured league ratings | No established gain in the tested allocation | Other opponent curricula and stronger rating evidence |
| All-full search | Worse pilot in the tested training-time budget | Position-adaptive search allocation |
| Fast policy weight 0.25 | Promising early candidates and a provisional baseline | Replicated gain and why extended training regresses |
| Width 512 | Resolved equal-tree64 head-to-head gain; promising later Astra result | Superiority over small1024 at a well-used two-second serving budget |
| Teacher distillation | Tested 50% policy-only mixture regressed badly | Gentler mixtures, schedules, and better teacher/student target alignment |
| Auxiliary score head | Early positive point estimate only | Independent matched confirmation |
| Policy-surprise sampling | Implementation and bounded validation | All strength comparisons |
| Rust memory reliability | Detection sites and a tested mitigation | Original corruption cause and sustained revised-binary training |

## Updating this journal

Before planning a new run, read the current baseline, relevant prior entry, and
its caveats. Add a dated entry when a hypothesis is launched and update that entry
after completion, interruption, failure, or an explicit stop. Record:

1. Hypothesis, controlled differences, run ID, seed, requested wall budget, actual
   elapsed time, and exact initialization (weights/full-state/donor replay).
2. Training settings and the exact evaluation protocol: opponents, both search
   budgets, sample sizes, seat/draw pairing, development versus held-out seeds,
   and local serving qualification.
3. Point scores, intervals, regressions, candidate selection, hashes of important
   retained artifacts, and whether anything was actually deployed.
4. Failures and unfinished checks. Keep “planned,” “implemented,” “smoke-tested,”
   “evaluated,” and “confirmed” distinct. Never infer completion from elapsed time.
5. Links to local source artifacts and a concise portable result here. Preserve
   earlier conclusions when adding a later correction; explain the changed evidence.

Keep the tuning export historical; append a clearly identified study export when
new tuning occurs. Do not rewrite original run manifests or code hashes merely to
make an old run accept newly committed code. A stop instruction remains in effect
until the user authorizes further training.
