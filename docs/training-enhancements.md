# Two-player training enhancements

## Policy target weighting experiment

The September 23 follow-up closes the paused fine-tuning study with independent
evaluation, then compares two matched two-hour pilots. Both retain the mixed
64/256-simulation teacher, width-256 source-attention model, learning rate,
opponent mixture, and binary +1/0/-1 values. The only treatment is
`--policy-fast-weight 0.25`, compared with the default `1.0` control.

Replay stores the simulation budget that produced each policy target. All three
game generators set this metadata, and reanalysis replaces it with the budget
used to refresh the policy. Ring-buffer overwrites, oversized additions,
checkpoint saves, and restores preserve alignment. Legacy targets have budget
zero (unknown) and retain weight one; their depth is never guessed.

For the experiment, targets below the configured full-search budget receive
policy weight 0.25, while full-search and unknown targets receive weight one.
After filtering invalid rows, the weighted sum of per-position KL losses is
divided by the sum of weights. Value MSE and entropy regularization keep their
existing reductions. This changes which policies the learner emphasizes without
reducing the overall nominal policy coefficient or changing tie rewards.

The strongest independently validated checkpoint initializes both arms. A matching
full-state archive preserves optimizer, replay, and RNG. If only milestone weights
exist, both arms receive identical fresh optimizer/replay/RNG state; state from a
later checkpoint is never attached to earlier weights. Both use the same frozen
opponent pool. The run report records which initialization was available.

The `policy-weight` campaign preserves a fixed ten-hour deadline including setup
and closeout, saves halfway and final pilot weights, selects on a development
panel, and reserves independent confirmation. The starting checkpoint remains
eligible. Replacement requires a positive lower bound on the paired panel 95%
interval and no more than a three-point Astra score drop. No deployment is automatic.
Native diagnostics remain enabled, the optional inference cache stays disabled,
and a native-memory failure stops the supervisor without an automatic retry.

The initial run is `agent/runs/competitive_policy_weight_10h_20260923`.
Its closeout uses a frozen copy of the previous Python implementation so changes
to learner/replay code cannot affect the earlier study's evaluation. Its source
manifest, package resources, runner corrections, and results are retained there.

The September 20 learning-rate campaign's finalist scored 58.0% against its
starting checkpoint (1,024 paired games; 95% interval 55.1–60.9%). Its improvement
against Astra was not statistically resolved. The next experiments target search
and training data quality while retaining the width-256 source-attention model,
successful learning rate, 72 learner updates, and zero extra Dirichlet noise.
See the [campaign report](../agent/runs/competitive_lr_10h_20260920/REPORT.md).

## Implemented options

`azul-train --preset enhanced-2p` selects these settings. The completed eight-hour
campaign established a gain for the bundle on one seed: 55.8% against its starting
checkpoint and 72.7% against Astra, both with matched 64-simulation search. See the
[enhancement report](../agent/runs/competitive_enhancements_8h_20260920/REPORT.md).
Individual contributions remain unmeasured.

| Enhancement | Preset behavior | Main controls |
|---|---|---|
| Stronger teacher | 64 simulations normally; 256 on a reproducible random 25% of iterations; 32 root candidates | `--selfplay-sims`, `--selfplay-full-sims`, `--selfplay-full-fraction`, `--search-max-root-candidates` |
| Tougher, diverse opponents | League checkpoints use 64-simulation Rust trees; sampling mixes all available, recent, and highest-rated checkpoints. Half the bot-phase games use Astra, a quarter Opus, a quarter heuristic | `--league-search-backend`, `--league-opponent-sims`, `--league-opponent-sampling`, `--bot-selfplay-astra-prob`, `--bot-opus-prob` |
| Replay reanalysis | Refresh 256 stored policies every four iterations using current weights and 256 simulations | `--reanalysis-positions`, `--reanalysis-every`, `--reanalysis-sims`, `--reanalysis-batch-size` |
| Tie reward | Sole winner +1, shared winners 0, losers −1 in binary mode | Applied consistently in Python rules, Rust rules/search, replay values, and arena value calibration |
| Playing-time search | `strong128` and `strong256` profiles: no root exploration noise, 32 root candidates, eight sampled refill outcomes, bounded inference caching | Arena `--profile`, `--root-noise-scale`, `--max-root-candidates`, `--chance-samples`, `--inference-cache-size` |

The four-iteration cycle remains self-play, league, self-play, bot: Astra therefore
appears in about 12.5% of generated games, up from about 1.6%. League play balances
the learner's seats, pins a copy of the initializing model, and keeps an isolated
league under the new run. Ratings guide the strong-opponent sampling component
when available; unrated checkpoints have equal ratings. Periodic online evaluation
is disabled in the preset, so use external paired arena evaluations for selection.

Mixed budgets apply to entire iterations to keep inference batches large. They do
not yet prioritize difficult positions. All positions generated by both budgets
can enter replay. The loop logs the actual simulation budget and reanalysis work
in events and metrics.

Reanalysis captures a random 1/16 of recorded states, retaining at most 8,192
snapshots. Only valid positions from finished games survive trajectory filtering.
Snapshots remain attached to their ring-buffer entries and are evicted when those
entries are overwritten. They are saved with replay for deterministic resumption.
Reanalysis verifies encoded features and legality against replay before replacing
policies. It preserves observed value targets, insertion ages, and sample counters.
Future draws are resampled by search; saved RNG state never supplies a future-draw
hint. Set `--reanalysis-positions 0` to disable capture and reanalysis.

Inference caching reuses exact encoded-position evaluations within search and
across moves with unchanged weights. It is bounded per network and invalidated
by optimizer updates or checkpoint loads. Parallel worker proxies use local
caches. This reuses neural evaluations, **not complete search trees or sampled
chance outcomes**. Full subtree reuse and difficulty-based compute allocation
remain separate experiments. Cache overhead and stronger search budgets must be
assessed by win rate per training hour and move latency.

## Start a fresh pilot from the finalist

The reward change is version 2. Previous weights remain usable for inference and
warm starts, but old replay/optimizer continuation is rejected; old tie labels
cannot be corrected without recorded outcomes. Start a new run with `--init-from`.
The `score_scaled` mode also assigns zero to shared winners while retaining its
existing score-based loser values. Official winner/tiebreak rules and arena match
scores (win 1, tie 0.5, loss 0) are unchanged. Historical reports retain their
original semantics and must not be silently pooled with new search measurements.

```bash
source .venv/bin/activate
python -m pip install ./native/astra

python -m agent.scripts.train \
  --preset enhanced-2p \
  --run-id enhanced_2p_pilot \
  --init-from agent/runs/competitive_lr_10h_20260920/experiments/lr_current_seed20260920/milestones/minutes_0450.pt \
  --device cuda --seed 20260920 --max-wall-minutes 60
```

This is a one-hour training pilot; evaluation needs additional time. The preset's
default limit is 600 training minutes. The loop completes its current iteration
and checkpoint before stopping. No campaign or model promotion is launched merely
by defining a preset. Explicit CLI overrides take precedence over preset and GPU
defaults. Keep the same quality settings when resuming a run; use a new run ID for
an ablation. The one-worker Rust setting is retained from the completed campaign;
CPU utilization alone is not a reason to increase it.

For a matched control, use the same new tie rewards and successful learner settings,
then disable the additions: full-budget fraction 0, reanalysis positions 0, 16 root
candidates, no inference cache, weighted league sampling with one-ply/four-simulation
opponents, original bot fractions and original league seat probability. Prefer
adding teacher depth, opponent mix, and reanalysis in separate matched pilots when
the goal is to identify which change pays for its compute cost.

## Compare playing-time profiles

```bash
python -m agent.scripts.arena \
  path/to/candidate.pt path/to/frozen_start.pt \
  --profile strong128 --opponent-profile rust64 \
  --games 256 --seed 20260921 --device cuda \
  --report path/to/development_search.json
```

Use the same checkpoint on both sides to isolate a search-profile comparison.
Opponent profiles are independent; without an explicit opponent profile or
overrides, arena uses the candidate's search settings for a neural opponent.
Also evaluate Astra with `--bot-workers 8` and greedy play with `--greedy` to
separate search gains from policy learning. Reports record full search settings,
checkpoint hashes, reward version, elapsed time, and paired outcomes. Use fresh
confirmation seeds after selecting a candidate; 1,024 games is a useful follow-up
to a 256-game screen. No profile is automatically installed in the play server.

## Validation

Tests cover official ties for 2/3/4 players and both reward modes, snapshot alignment
through replay wrap and filtering, old-replay rejection, policy-only refresh,
mixed-budget determinism, league diversity, cache correctness/invalidation, and
uninterrupted versus resumed enhanced training. CPU and CUDA smoke runs use the
actual September 20 finalist checkpoint.

September 20 validation: the full agent/play suite passed 419 tests; the subsequent
arena regression suite passed six tests, including one newly added profile/seed
test. All 16 Rust tests passed. CPU/CUDA smoke runs finished 24/24 games, refreshed
48 policies, and completed 12 learner updates with none skipped. A two-game arena
smoke also finished with independent `strong128`/`rust64` profiles; its tiny sample
is not strength evidence. The [verification record](../agent/runs/training_enhancements_20260920/verification.json)
and adjacent logs preserve settings and results; disposable smoke weights were
retired to recover disk space. That validation did not launch a long training campaign or promotion.

## Bounded eight-hour campaign

The `enhancements` campaign stage runs one-hour control and enhanced pilots from
the same initializer, then continues the selected arm to cumulative training
milestones of 175 and 290 minutes. Both arms use fresh replay and zero tie rewards.
The control retains the September 20 finalist's training settings. Enhanced is
selected only when its paired development advantage against the frozen start has
a 95% interval above zero and its Astra score is no more than three percentage
points below control. Otherwise the control continues.

All selection and confirmation games use the same Rust 64-simulation profile for
both networks. The finalist is selected on development games before fresh held-out
games begin; the frozen initializer remains eligible. The campaign reserves 90
minutes for final evaluation plus ten minutes for the last development screen.
It automatically writes `REPORT.md`, `verification.json`, and machine-readable
results. Completed replay files are retired after freezing their milestone weights.

```bash
python -m agent.scripts.supervise_training \
  --stage enhancements --hours 8 \
  --root agent/runs/competitive_enhancements_8h_20260920 \
  --initializer agent/runs/competitive_lr_10h_20260920/experiments/lr_current_seed20260920/milestones/minutes_0450.pt \
  --device cuda --seed 20260920 --bot-workers 8
```

A `preflight_budget.json` in the campaign root can include setup in the absolute
deadline. Restarts retain that deadline and resume completed stages/checkpoints;
they never grant another eight hours. The supervisor monitors progress, retries
recoverable failures, and stops at the deadline. No automatic promotion occurs.

## Bounded twelve-hour reanalysis campaign

The `reanalysis` stage starts two matched 90-minute pilots from the latest enhanced
champion. Both use the enhanced preset and fresh optimizer/replay state. The only
training-setting difference is refreshing 256 versus 1,024 positions every four
iterations; each refreshed position receives 256 simulations. Separate run and
league directories prevent cross-contamination.

Pilot screens use 512 games per arm against the frozen champion and Astra. The
fourfold arm continues only if its paired frozen-opponent score advantage has a
95% interval above zero and its Astra score is no more than three percentage
points below the current settings. Inconclusive results retain current settings.
Development and confirmation seeds are separate and new for this campaign.

Continuation targets cumulative training minutes 210, 330, 450, and 495, with
512-game development screens against both opponents at each milestone. The final
target is shortened as necessary to preserve 90 minutes for confirmation and ten
minutes for the last development screen. This includes setup and evaluation in
the absolute twelve-hour limit. Final confirmation uses 1,024 games against the
frozen champion, 1,024 each for candidate/start against Astra, and 256 each for
candidate/start greedy play against Astra. All primary matches use identical
64-simulation Rust search.

The selected training arm retains `checkpoints/latest_resume.pt`, including
optimizer, replay, and RNG state, together with its league and milestone weights.
This resume state corresponds to the latest training milestone; the best evaluated
weights may come from an earlier milestone. The unselected pilot's replay is
retired after its milestone weights have been verified. Restarting a completed
campaign leaves the selected resume state intact.

```bash
python -m agent.scripts.supervise_training \
  --stage reanalysis --hours 12 \
  --root agent/runs/competitive_reanalysis_12h_20260921 \
  --initializer agent/runs/competitive_enhancements_8h_20260920/experiments/enhanced_seed20260920/milestones/minutes_0290.pt \
  --device cuda --seed 20260921 --bot-workers 8
```

The campaign writes `reanalysis_campaign_plan.json`, selection records,
`reanalysis_campaign.json`, `verification.json`, and `REPORT.md`. The supervisor
preserves the original deadline across restarts and monitors progress and disk
space. The campaign does not promote or deploy a model.

## Bounded ten-hour measured-league campaign

The `league-campaign` stage forks a complete resume archive, including optimizer,
replay, RNG, and prior training progress, into isolated current/rated arms. Source
archives and league weights use hard links; manifests are independent copies, and
checkpoint saves atomically replace destination files. The original run is preserved.
Both arms retain the source training seed and all learner/search settings. New,
separate seeds are used for league ratings, development, and final confirmation.

The rated arm adds pinned copies of the current model, a historical champion, and
the alternative reanalysis pilot to its inherited league. An initial six-model
round robin uses 128 paired games per matchup plus a random-opponent anchor match.
Later refreshes compare the two latest checkpoints against four fixed league
anchors. These results populate the existing per-player-count Bradley–Terry
ratings. In the rated arm's strong-opponent sampling group, models need at least
128 measured games and a 2p rating; equal measured ratings use randomized order.
The historical and recent groups remain available. The control retains its
existing sampling behavior and unrated league.

Each pilot has a 120-minute allocation, split at 60 minutes. Rating work is charged
against the rated arm's allocation, reducing its self-play time. The selected arm
continues to cumulative allocations of 225 and 330 minutes. Each arm starts from
the same inherited training progress; these allocations are additional time.

The evaluation panel is the frozen starting model, the previous champion, and
Astra. Development screens use 512 games per opponent. Arm selection requires a
resolved equal-weight mean-score advantage, bootstrapping whole seed blocks across
all three opponents, and an Astra point-score loss no worse than three percentage
points. Inconclusive results retain the control. Checkpoint selection maximizes
the same panel mean with an Astra guard against the starting model; the starting
model remains eligible. Selection is frozen before independent confirmation games.
Final evaluation runs 1,024 games per opponent for both candidate and starting model.
All neural matchups use identical 64-simulation Rust search.

The absolute ten-hour deadline includes setup and evaluation. Training reserves
90 minutes for final confirmation and 15 minutes for the last development panel.
The selected arm retains its latest full resume archive and league. The unselected
pilot's replay is retired after its milestone weights are verified. The saved
resume may be later than the best evaluated weights. No automatic deployment occurs.

Compressed resumes now use their observed archive size for storage preflight,
allowing at least 1 GiB or twice the archive size (capped at the raw replay size).
Fresh runs still reserve raw replay space. Atomic checkpoint writes continue to
check free disk space while streaming and preserve the old archive on failure.

```bash
python -m agent.scripts.supervise_training \
  --stage league-campaign --hours 10 \
  --root agent/runs/competitive_league_10h_20260922 \
  --initializer agent/runs/competitive_reanalysis_12h_20260921/experiments/current_seed20260921/checkpoints/latest_resume.pt \
  --device cuda --seed 20260922 --bot-workers 8
```

Outputs include `league_campaign_plan.json`, `league_ratings/`, `arm_budgets/`,
`league_campaign.json`, `verification.json`, and `REPORT.md`. The supervisor
retains the original deadline across retries. The campaign requires a full
resume initializer, not a weights-only milestone.

## 24-hour fine-tuning comparison

`competitive finetune-campaign --campaign-hours 24 --initializer FULL_RESUME`
starts from the retained two-player width-256 champion, including optimizer
moments/steps, the million-position replay buffer, RNG, and its original league.
`agent.scripts.finetune_campaign` creates isolated forks with explicit metadata.
The lower-rate fork changes both saved configuration and AdamW parameter-group
learning rates; ordinary resume configuration checks remain strict.

Three three-hour pilots compare the existing rate and mixed 64/256 teacher,
a learning rate of 0.0003, and a 256-simulation teacher on every iteration.
Other learning settings, including the opponent search budget and reanalysis,
are fixed. All arms disable the optional native inference cache as a common
reliability workaround: the prior campaign's core dump located the abort during
cached-result eviction. This does not establish the cause of the original heap
corruption. Rust simulation and Rust tree traversal remain enabled.

The campaign requires an explicit local `reliability.json` preflight clearance.
The September 22 launch uses Python's debug allocator and faulthandler; its
supervisor stops on SIGABRT, SIGSEGV, or SIGBUS instead of automatically retrying
native-memory failures. Other recoverable failures retain ordinary bounded
checkpoint recovery. Core findings and stress-test evidence belong in the run
folder; a passing stress test does not prove absence of memory errors.

Each pilot receives 512 seat-paired games against each of the frozen start,
previous champion, and Astra. The highest equal-weight panel score within 3pp
of starting Astra performance receives the continuation budget. Pilot statistical
significance is not required to allocate experimental time. If every pilot misses
the Astra guard, the least Astra regression receives further investigation;
this is recorded explicitly and does not authorize model promotion.

The selected arm receives a ten-hour continuation window, including intermediate
screens, with cumulative additional-training targets of 330, 480, 630, and 780
minutes. The original absolute 24-hour deadline includes preparation and survives
restarts. Training clamps to leave three hours for confirmation and 15 minutes
for the final development screen. All pilot checkpoints, later milestones, and
the starting model remain eligible for final selection. Original training state
and the selected arm's latest full resume are retained; only losing arms' new
replay archives are retired after their pilot weights have been verified.

The candidate is frozen before independent confirmation: 2,048 games per opponent
for candidate and start, plus 512 greedy games against Astra each. Development,
confirmation, and greedy seeds are separate and disjoint from earlier campaigns.
Candidate and neural opponents use identical 64-simulation search. Acceptance
requires a positive lower bound on the paired 95% panel-gain interval and an
Astra point-score decrease no worse than 3pp. Otherwise selected weights remain
the starting champion. If development already selected the start, confirmation
runs only once, and the report does not present a spurious zero-width difference
interval between duplicate evaluations. There is no automatic deployment.
