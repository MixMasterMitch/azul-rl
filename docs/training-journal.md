# Training journal

Last reviewed: **2026-10-10**. Shared-core confirmation is complete, including the 4p mixed-table panel and separate 2p direct match. Decision: **specialists retained; shared model remains experimental**. The evaluation service has finished.

This is the durable record of the RL training experiments recoverable from this
workspace: six May–June tuning studies, the v3/v4 runs, and the September–October
campaigns. It records unsuccessful and interrupted attempts as well as improvements.
Dates in campaign names identify their run directories; a run can finish on a later
day. Explicit clock times below use America/Los_Angeles.

The objective is maximum playing strength within approximately **two seconds
per turn**. From October 8, this includes 3p/4p training while preserving
2p playing strength; the user permits changes to 2p behavior. Training speed matters as an experiment cost, not as the final
objective. Astra's separate search
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

**Archive availability at the October 1 review:** those two original checkpoint files and
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

Unresolved comparisons include the saved width-512 `continuation_180m` candidate
at 384 simulations versus the baseline at 1,024, and a fresh confirmation of the
early auxiliary-score candidate. The matched policy-surprise experiment has now
completed two training seeds without establishing a benefit over uniform replay;
the retained method and model are unchanged. The initial heap-corruption cause
remains unproven. None of these pending questions authorizes another training run.

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
| October 1 and 6, surprise pilots 8h / resumed 6h | Uniform versus policy-surprise replay | Direct surprise score 51.32% [48.29, 54.35]; inconclusive, no promotion |
| October 7, surprise replication 9h cap | Repeat matched comparison with a fresh training seed | Direct surprise score 47.27% [44.19, 50.24]; two-seed average 49.29%, no demonstrated benefit |
| October 8, multiplayer foundation 12h cap | Transfer 2p knowledge into independent 3p/4p specialists | Completed in 8.06h; held-out Astra win shares 59.90% (3p) / 37.50% (4p); 2p unchanged, CPU timing passed |
| October 8–10, shared core / completed confirmation | Consolidate specialists and train one shared core on mixed tables | Weighted gains: 3p +4.09pp [+0.39, +7.60], 4p +3.05pp [+1.04, +5.04]; direct 2p 52.15% [50.02, 54.30]; broader 2p panel unresolved, automatic promotion gate unmet |

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

### Matched policy-surprise pilots — October 1 and 6, completed

The October 1 eight-hour study used the revised native binary and a fresh five-iteration
smoke check, then forked the same surviving iteration-54 full state (optimizer and
replay included) into two width-256 arms with seed 20261002. Both used the same
new pinned-baseline league, search budgets, learner settings, and 120 cumulative
training minutes. The sole controlled change was full-policy replay sampling:
uniform control (fraction 0) versus policy-surprise fraction 0.5, requiring at
least 256 search simulations and capping the sample weight at 4. Both recorded
surprise metadata. The historical league checkpoint files were missing, so the
new league is a documented departure from that earlier campaign. The recovered
frozen baseline reproduces the fork's model tensors but is not the lost original
full-state archive.

The user force-stopped the first service after the control arm reached iteration
156; its 256-game screen was interrupted halfway. On October 6, the user
explicitly authorized resuming. The unchanged code and native binary, checkpoint
hash, idle GPU, and disk floor were verified. A new six-hour deadline (21:22:54
UTC) with a systemd runtime cap resumed the evaluation, then trained the surprise
arm without repeating completed control training. Both arms completed 102
iterations from the fork, 7,344 successful learner updates, zero skipped updates,
and iteration-156 weight snapshots. The control and surprise 120-minute training
wall times were 7,210 and 7,296 seconds respectively.

Development screens used paired seats/draws, 1,024 simulations per model,
128 games at 30 minutes and 256 games at 120 minutes. Against the frozen
baseline, control scored 43.75% then 52.54% [46.88, 58.20]; surprise scored
50.00% then **57.23% [51.95, 62.70]**. The paired 120-minute surprise-minus-control
difference through that common opponent was +4.69pp [−2.93, +12.50]. Against
Astra at 120 minutes, control scored 79.10% and surprise 81.45%; surprise's
paired change versus the frozen baseline's 84.77% was −3.32pp [−9.38, +2.54].
No predeclared early-stop gate fired.

A separate 1,024-game, paired-seat/draw direct match on fresh confirmation seeds
put surprise against control at the same 1,024-simulation search budget:
**51.32% [48.29, 54.35]**, with no unfinished games. The result does not
establish policy-surprise superiority over uniform replay. It is one training
seed per arm; the direct match did not include a new CPU serving-time audit.
No checkpoint was promoted or deployed. All experiment services ended and GPU
compute was released. The small frozen baseline remains the retained reference;
next work should replicate the matched training comparison before selecting a
new method, if further training is desired.

Evidence: [frozen plan](../agent/runs/competitive_surprise_pilots_8h_20261001_r2/plan.json),
[resume budget](../agent/runs/competitive_surprise_pilots_8h_20261001_r2/resume_budget_20261006.json),
[development results](../agent/runs/competitive_surprise_pilots_8h_20261001_r2/result.json),
[direct comparison](../agent/runs/competitive_surprise_pilots_8h_20261001_r2/direct_comparison_20261006.json).

### Fresh-seed policy-surprise replication — October 7, completed

The user authorized one more matched training pair after the first direct comparison
remained unresolved. The replication uses fresh training seed **20261007** in
[its run directory](../agent/runs/competitive_surprise_replication_seed20261007/),
the same frozen iteration-54 full-state checkpoint, a new empty pinned-baseline
league, and the same width-256 training settings as the prior pair. Both arms
record surprise labels; uniform sampling (fraction 0) and policy-surprise
sampling (fraction 0.5) are the only intended treatment difference. Each arm
has 30- and 120-minute cumulative checkpoints and development screens; the
30-minute screens are diagnostic so both arms can reach the requested matched
two-hour budget unless a safety failure intervenes. Development screens use
128/256 paired-seat/draw games against the frozen baseline and Astra, with
1,024 simulations for learned models. A fresh-seed, direct 1,024-game paired
comparison of the 120-minute arms is planned at 1,024 simulations each.

The whole run was bounded to nine hours from launch, with a separate systemd
runtime cap and a 6 GiB disk floor. It completed at **5:20 PM PDT October 7**, after
**6h24m** including evaluations. Both arms reached iteration 156: 102 completed
iterations, 7,344 successful learner updates, and zero skipped updates each.
Control used 7,303 seconds of training and surprise 7,221 seconds. No native
corruption was observed in this run. The service exited successfully and GPU
compute was released. No continuation, deployment, or model promotion occurred.

At 120 minutes, the 256-game development screens gave control **48.24%
[41.80, 54.69]** against the frozen baseline and surprise **48.44% [42.77, 54.10]**.
Their paired difference through the baseline was +0.20pp [−8.79, +9.38]. Against
Astra, control scored **85.35%** and surprise **81.64%**, compared with the frozen
baseline's **86.72%** on the same 256-game seed set. Neither arm demonstrated an
improvement over the frozen baseline.

The fresh-seed direct evaluation completed all **1,024 games** at 1,024 simulations
per model. Surprise had 477 sole wins, 533 losses, and 14 shared victories:
**47.27% match score [44.19, 50.24]**. The point estimate favors uniform replay,
but the interval includes 50%, so this does not establish that surprise is worse.
Together with the first pair's 51.32%, the two-seed average is **49.29%**. The
pooled game-level interval is [47.17, 51.39], descriptive only: it ignores variation
between training seeds and must not be treated as a robust multi-seed interval.
The initial positive point estimate did not replicate; these experiments provide
no demonstrated strength benefit from the tested policy-surprise configuration.

The retained reference remains the frozen small baseline. There was no separate
model-promotion confirmation or new CPU serving-time audit. The control and
surprise checkpoint SHA-256 values are respectively
`11cd47e294c180b9d6c534fe06edb96fa189e2e6c9ac7ca1fc4852f79811d187` and
`cebc8963c81bc9538f4a121d514e1e68c6b4a4565ca190376fecac2391d6082b`.

Evidence: [frozen plan](../agent/runs/competitive_surprise_replication_seed20261007/plan.json),
[completed result](../agent/runs/competitive_surprise_replication_seed20261007/result.json),
[control screens](../agent/runs/competitive_surprise_replication_seed20261007/control_screens.json),
[surprise screens](../agent/runs/competitive_surprise_replication_seed20261007/surprise_screens.json),
[direct match](../agent/runs/competitive_surprise_replication_seed20261007/evaluations/surprise_120m_vs_control_120m_direct_f24034ebf9f9.json),
[completion record](../agent/runs/competitive_surprise_replication_seed20261007/replication_status.json).

## October 8: multiplayer foundation — completed

The user requested starting three- and four-player training without regressing
two-player performance. The initial window used the stated twelve-hour default
after no duration answer arrived: **7:39 AM–7:39 PM PDT October 8**, covering validation,
smoke runs, training, and evaluation after preparation. The shared network backbone
would allow 3p/4p updates to change 2p outputs, so this experiment trains independent
specialists and routes 2p to a separate frozen checkpoint. Neither the current
play-server selection nor its 2p search settings are changed.

The source is the recovered frozen small baseline
`agent/runs/competitive_wide_eval_20261001/baseline_recovered.pt`, SHA-256
`897f8c4bdabb644abcd38dd8cad8bec7595a70ea61cf8d35299056f505a9e981`.
A byte-identical private copy is retained in the new campaign. The 3p/4p
initializers copy its backbone, policy head, and embedding, then reset the
multiplayer value readout to the symmetric binary-reward expectation. Optimizers,
500,000-position replay buffers, and leagues are fresh and separate. These are
weights-only transfers, not resumes of 2p optimizer or replay. Seeds are 20261011
for 3p and 20261012 for 4p. Both use uniform replay, the established 64/256 search
schedule, 72 learner steps, fast-policy weight 0.25, and regular reanalysis.

After bounded four-iteration smoke checks for each player count, 60% of remaining
wall time is allocated equally to training, with alternating 30-minute,
120-minute, and final cumulative milestones. Development uses full seat rotations
against heuristic, Opus, and Astra at 256 simulations (96 games/opponent). One
checkpoint per player count is selected on the development panel, then compared
with its transferred initializer on held-out Opus/Astra seeds (384 games per
model/opponent). Multiplayer **win share** divides credit among the actual tied
winners; a third player's loss is not counted as a draw. This differs from the
two-player match-score shorthand used in earlier entries.

Final CPU audits cover 24 opening/middle/late positions per player count, with a
two-second outer budget. Source/2p/native hashes, independent model routing,
separate leagues, and a 6 GiB disk floor protect the retained reference and run
integrity. No automatic restart or deployment is part of this campaign. Completion,
failure, or stop will be appended to this journal by the runner.

**Launch validation:** 587 repository tests passed. Actual baseline versus the
frozen bundle route produced bitwise-identical policy/value outputs and
64-simulation tree actions/policies on the checked 2p positions. Both CUDA smoke
runs completed all four iterations (48 games and eight learner updates per player
count), with no skipped updates. Their saved full states reloaded successfully
with the correct player count, optimizer, replay, and RNG. The unchanged native
binary is the one used by the completed surprise replication. These checks
establish pipeline functionality and 2p identity, not multiplayer playing strength.

At launch, the persistent service `azul-multiplayer-foundation-20261008.service` was active;
all six initial development evaluations completed with no unfinished games, and
the first full 3p training segment started at 7:48 AM PDT. Allocation is
**214 cumulative training minutes per player count**, at 30/120/214-minute
milestones. Its runtime cap allowed graceful shutdown five minutes before the
7:39 PM deadline. The planned work finished normally at **3:42 PM**, after 8h04m;
the unused evaluation reserve was not converted into additional training.

The 3p arm finished 245 iterations / 17,640 learner updates / 62,720 games, and
4p finished 186 iterations / 13,392 updates / 47,616 games. Neither had skipped
updates or unfinished training games. The development panel selected **3p/120m,
iteration 144**, and **4p/214m, iteration 186**. The later 3p model had a slightly
higher Astra point score but a lower mean across the three development opponents;
this selection does not establish a statistically resolved regression from more
training. Held-out results are recorded in the closeout below.

Both selected models completed all 256 search simulations on 24 CPU positions.
The 3p p95/max turn times were 0.347/0.351 seconds; 4p was 0.341/0.341 seconds,
within the two-second budget. This is a local timing sample, not a guarantee for
every possible position. Source and frozen 2p hashes and bundle routing were
verified unchanged again after completion. The experimental bundle contains all
three routes; the play server remains unchanged.

The selected 3p milestone contains weights, while its latest full resume is from
iteration 245. Do not combine that later optimizer/replay with iteration-144
weights and describe it as an exact resume.

Evidence: [frozen plan](../agent/runs/multiplayer_foundation_12h_20261008/plan.json),
[player-count bundle](../agent/runs/multiplayer_foundation_12h_20261008/bundle.json),
[live status](../agent/runs/multiplayer_foundation_12h_20261008/status.json),
[regression and 2p audit](../agent/runs/multiplayer_foundation_12h_20261008/validation.json),
[smoke checkpoint audit](../agent/runs/multiplayer_foundation_12h_20261008/smoke_checkpoint_audit.json),
[completed results](../agent/runs/multiplayer_foundation_12h_20261008/result.json),
[3p CPU timing](../agent/runs/multiplayer_foundation_12h_20261008/latency_3p.json),
[4p CPU timing](../agent/runs/multiplayer_foundation_12h_20261008/latency_4p.json),
[protocol](multiplayer-training.md).

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
| Policy-surprise sampling | Two matched training seeds average 49.29% in direct matches; no demonstrated advantage over uniform replay | Small effects remain uncertain; alternative sampling settings were not tested |
| Rust memory reliability | Revised binary completed four 120-minute arms across two training seeds without observed corruption | Original corruption cause and longer-run reliability |

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

### Multiplayer campaign closeout — 2026-10-08T15:42:56-07:00

Run `multiplayer_foundation_12h_20261008` **completed**. Elapsed 8.06h. Frozen 2p checkpoint and route verified unchanged; no deployment.

- 3p: selected iteration 144; CPU latency qualified: True. Against opus: win share 89.45% versus initializer 4.69%; paired change +84.77pp [+81.12, +88.28] (384 games per model). Against astra: win share 59.90% versus initializer 6.77%; paired change +53.12pp [+47.53, +58.59] (384 games per model).
- 4p: selected iteration 186; CPU latency qualified: True. Against opus: win share 74.35% versus initializer 1.17%; paired change +73.18pp [+69.27, +77.08] (384 games per model). Against astra: win share 37.50% versus initializer 3.65%; paired change +33.85pp [+28.91, +38.80] (384 games per model).

Evidence: [run status](../agent/runs/multiplayer_foundation_12h_20261008/status.json), [frozen plan](../agent/runs/multiplayer_foundation_12h_20261008/plan.json).

## October 8–9: shared-core consolidation and mixed tables — budget stopped; partial confirmation

The user authorized the proposed 24-hour setup/training/evaluation window and
explicitly requested mixed tables in training and evaluation, with weighted win
rates. Two-player behavior may change provided playing strength is retained.
The experiment uses a fully trainable width-256 shared backbone and three separate
policy/value heads; the selected 2p/3p/4p foundation specialists remain immutable
teachers and fallbacks. This is a new weights-only initialization with a fresh
optimizer and explicitly donated observations, not an exact specialist resume.

Run: `agent/runs/shared_core_24h_20261008`. The frozen budget starts during setup.
Consolidation fits all three teachers' policies AND values with equal per-count
loss weights. Online learning adds balanced self-play/mixed tables and 15% teacher
anchors, separate 200k replay buffers, Rust 64/256 search, and regular reanalysis.
Training ends six hours before the 24-hour deadline to reserve confirmation time.
There is no automatic deployment or restart.

Evaluation rotates all distinct candidate/opponent seat assignments. Fixed table
weights emphasize combinations of Astra, Opus, and frozen neural specialists.
A sole winner earns 1, actual shared winners split 1/k, and losers earn 0; these
win shares are weighted by the declared table distribution separately for each
player count. Intervals resample complete seed blocks jointly across tables;
paired changes replace only the candidate seat. No adaptive weighting or averaging
across player counts can hide a failed per-count retention gate.

The initial CPU audit qualified all three specialists at 512 and 1,024 simulations
on 24 positions each; 1,024-simulation p95 was 1.31–1.34 seconds. Main evaluation
uses 1,024 simulations for every neural seat. Final shared-model CPU audits use
48 fresh positions per count. These timing checks are not strength results.
Development selection precedes fresh mixed-table confirmation and a separate
2,048-game 2p direct match. A predeclared two-percentage-point noninferiority margin
is an experimental retention criterion, not proof of identical strength; unresolved
comparisons keep the specialist fallback. Final decisions require all three counts
and the 2p direct test to pass, plus CPU qualification.

Evidence: [protocol](shared-core-training.md),
[budget](../agent/runs/shared_core_24h_20261008/budget.json),
[plan](../agent/runs/shared_core_24h_20261008/plan.json),
[status](../agent/runs/shared_core_24h_20261008/status.json).

**Launch validation:** 594 full-suite tests passed in 155.53 seconds; all eight
shared-core tests passed after the final bookkeeping/cache changes. The real CUDA
smoke completed 72 finished games across 2p/3p/4p, four balanced online updates,
reanalysis, and a compressed full-state restore. Functional checks do not establish
playing strength. The native extension was unchanged.

The persistent service `azul-shared-core-20261008.service` launched successfully
and began corpus construction. The budget is **October 8, 10:01 PM PDT–October 9, 10:01 PM PDT**, including setup.
Its automatic runtime cap leaves five minutes for graceful saving; no restart is
configured. Input checkpoint, native binary, and executable source hashes are
frozen. [Validation](../agent/runs/shared_core_24h_20261008/validation.json) and
[launch details](../agent/runs/shared_core_24h_20261008/launch.json) are saved.

**Initial live check:** corpus construction finished for all three player counts,
and the production run completed its first 500 joint consolidation updates with
finite losses and saved shared checkpoints. The service remains active. Early
imitation-loss changes are diagnostics only; no strength result or promotion has
been established.

### Shared-core campaign closeout — 2026-10-09T21:59:06-07:00
Run `shared_core_24h_20261008` **stopped** after 23.96h. No deployment.

Failure/stop: `Graceful stop requested`. Last phase: evaluation.

Evidence: [status](../agent/runs/shared_core_24h_20261008/status.json), [plan](../agent/runs/shared_core_24h_20261008/plan.json), [results](../agent/runs/shared_core_24h_20261008/result.json).

### Shared-core review — October 10: training finished, confirmation incomplete

The run completed **20,000 consolidation updates**, **408 online rounds / 29,376
joint RL updates**, and **117,504 online games** (39,168 per player count). The
all-count development rule selected `milestones/online_338.pt`, SHA-256
`8049d4b43d5e5f31ea9883cadc4526fb7b483195ea24007d34adfffa879a6862`.
The final full resume is iteration 408, with three populated 200k replay buffers,
optimizer, and RNG state; it was loaded successfully at review. Teacher, native,
and selected checkpoint hashes still match. No service or GPU training remains.

Completed fresh-seed, 1,024-simulation confirmation panels, recomputed from the
saved per-table reports:

| Player count | Shared weighted win share | Specialist | Paired change, 95% CI | Games per model |
|---|---:|---:|---:|---:|
| 2p | 70.55% | 71.80% | −1.25pp [−5.69, +3.29] | 768 |
| 3p mixed tables | 61.24% | 57.15% | +4.09pp [+0.39, +7.60] | 1,344 |

The 3p interval supports an improvement on this fixed panel for one training seed.
The 2p interval establishes neither a regression nor retention within the declared
two-point tolerance. Its panel included a 256-game direct score of 50.78%
[44.53, 57.23]; the separate 2,048-game direct test never started. All completed
reported games finished normally. These weighted shares are not comparable to
older homogeneous-opponent panels with different search budgets.

The selected model's 4p **development-only** score was 41.49% versus 39.32%, a
+2.18pp change [−1.61, +5.94]. The first held-out 4p table was interrupted after
864 of 1,536 games; its in-memory partial records were not saved as a table report.
The full 4p confirmation and larger direct 2p test remain outstanding. The final
`result.json` does not exist; the durable partial result is `confirmation_progress.json`.

All three shared-model CPU audits qualified 1,024 simulations on 48 positions each:
p95 1.335/1.336/1.337 seconds, sampled maxima below 1.366 seconds. This does not
establish a worst-case bound for every position.

The service reached the wall cap and saved gracefully at **9:59 PM PDT October 9**,
after 23.96 hours. Its timeout/exit status reflects the cap rather than an observed
numerical or native-engine failure. The evaluation allocation was insufficient:
4p development panels took about 55 minutes per 720 games, and the final development
screen used part of the six-hour reserve. At observed throughput, the complete
remaining frozen 4p panel and 2p direct match require roughly twelve more evaluation
hours. Future evaluation continuation should save complete seed blocks incrementally
so an interrupted table does not lose work.

**Decision:** specialists retained; no promotion, deployment, restart, or extra
training. Finish the frozen selection's outstanding evaluations before another
training study. Confirmation must not select an alternative checkpoint.

Evidence: [review report](../agent/runs/shared_core_24h_20261008/REPORT.md),
[structured review](../agent/runs/shared_core_24h_20261008/review_20261010.json),
[completed held-out panels](../agent/runs/shared_core_24h_20261008/confirmation_progress.json),
[selection](../agent/runs/shared_core_24h_20261008/selection.json),
[stop record](../agent/runs/shared_core_24h_20261008/failure.json).

### October 10: authorized evaluation-only completion — completed

The user authorized finishing the outstanding evaluations. This continuation keeps
the already-selected iteration-338 checkpoint, all specialists, 1,024-simulation
search settings, table weights, confirmation seeds, and original retention criteria.
It does not retrain, reselect, deploy, or rewrite the original campaign manifest.
The original iteration-408 full training resume and completed 2p/3p results are
hashed and preserved.

The separate 2,048-game direct 2p test runs first. Then all four 4p table strata
are completed for both the selected model and the frozen 4p specialist: 3,840 games
per model, 7,680 4p games total. This is **9,728 new evaluation games**. The first
attempt's in-memory 864 partial 4p games are unavailable and must be repeated.
The 48-position CPU timing audits for each player count are reused with verified
checkpoint, source, and native identities.

Every finished 96-game batch (including complete seat blocks) is now atomically
saved and fsynced before stop checks. Saved requests include all model/search/seed
identities; resumes preserve the original batch offset and therefore search RNG
seeds. The original and updated evaluator produced identical records in CPU
2p/3p/4p checks, and a real selected-model 4p mixed Astra/Opus/neural CUDA test
matched across an interruption and disk resume. Only an unsaved current batch
needs repeating after a forced stop. These are functional checks, not strength
results.

Expected evaluation time is roughly twelve hours. A separate eighteen-hour runtime
limit allows variability and reserves fifteen minutes for graceful shutdown. No
automatic restart is configured. The completion runner will write final results,
a readable report, and a journal closeout. Frozen historical manifests are not
modified to accept the persistence change; a new manifest records its exact scope.

Evidence: [completion manifest](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/plan.json),
[completion status](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/status.json),
[deterministic parity/CUDA resume smoke](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/parity_smoke.json).

**Launch verified:** `azul-shared-evaluation-20261010.service` is active and began
the direct 2p confirmation on **October 10 at 10:15 AM PDT**. Its maximum wall window
ends October 11 at 4:15 AM PDT; expected work is roughly twelve hours. All **41 targeted
tests passed** (28.37 seconds), including end-to-end final-report creation, frozen
artifact preservation, rejection of changed evaluation inputs, and exact record
equality after disk resume. Existing completed 2p/3p panels were recomputed and
matched their saved results after JSON normalization. The native binary and all
inherited frozen inputs passed preflight. No new learner or training process was
started.

Evidence: [launch](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/launch.json),
[validation](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/validation.json),
[preflight](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/preflight.json).

### Shared-core evaluation completion — 2026-10-10T21:00:32-07:00

Authorized evaluation-only continuation **completed**. No additional training or deployment.

| Count | Shared weighted win share | Specialist | Paired change, 95% CI |
|---|---:|---:|---:|
| 2p | 70.55% | 71.80% | -1.25pp [-5.69, +3.29] |
| 3p | 61.24% | 57.15% | +4.09pp [+0.39, +7.60] |
| 4p | 41.75% | 38.70% | +3.05pp [+1.04, +5.04] |

The separate 2,048-game 2p direct match scored **52.15%** [50.02%, 54.30%].

Decision: **specialists retained; shared model remains experimental**. The original two-point tolerance and all-count gates were unchanged. These intervals reflect games on the fixed opponent panel for one training seed.

The existing 48-position CPU audits for each count remain qualified at 1,024 simulations. All inherited checkpoint/native hashes and completed reports were verified. The selected checkpoint remains iteration 338; the final training resume remains iteration 408.

Evidence: [completion status](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/status.json), [completion manifest](../agent/runs/shared_core_24h_20261008/eval_completion_20261010/plan.json), [final results](../agent/runs/shared_core_24h_20261008/result.json).

**Completion verified at the October 10 review:** all 9,728 continuation games
are durably saved, the service exited successfully at 9:00 PM PDT after 10h45m,
and GPU compute is released. The 4p weighted aggregate was independently
recomputed from all eight completed table reports, with no unfinished games,
and the selected model hash was rechecked. The narrower 2p direct interval
favors the shared model, while the smaller weighted 2p panel still cannot rule
out a loss beyond the predeclared two-point margin; this is uncertainty in that
panel, not a demonstrated 2p regression. No acceptance criterion was changed
after seeing confirmation results.

### October 12: deployment rating preparation — no new training or matches

For the requested human comparison table, the shared iteration-338 model was
selected as the top-model reference across 2p/3p/4p. The frozen 19,456-game
Astra ranking fit anchors random, heuristic, Opus, and Astra; held-out shared-core
confirmation records fit the shared model to those opponents at 1,024 simulations.
The rounded display ratings are 3419/2796/2672 for the shared model, with
random=1000 and heuristic=2500 in each player count. These cross-campaign
Bradley–Terry estimates are diagnostic, without confidence intervals. The
formal shared-core promotion gate above remains recorded as unmet. The serving
release is staged locally, with no AWS deployment.

Evidence and method: [deployment readiness](deployment-readiness.md),
[rating data](deployment-ratings.json).

**October 12 play-scale revision:** the user chose a fixed comparison ladder
for human results: random 1000, heuristic 2000, Opus 2250, Astra 2500, and the
shared RL model 3000 in each player count. Human head-to-head fits are bounded
to 1000–3500. The earlier evaluation-derived values above remain research
diagnostics and are no longer the staged play display. No new strength claim or
training result follows from the chosen anchor values. See
[deployment readiness](deployment-readiness.md) for the serving algorithm.
