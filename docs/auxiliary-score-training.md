# Two-player auxiliary score experiment

The sixteen-hour `aux-score` campaign includes setup, fresh evaluation of the saved
width-512 `continuation_180m` weights, matched width-256 pilots, adaptive continuation,
and independent final evaluation. `preflight_budget.json` supplies an absolute deadline;
the supervisor includes a five-minute graceful-stop allowance inside it. No deployment
is automatic. Rust simulation/search, CUDA batched inference, native diagnostics, and
disabled inference caching are preserved.

Both pilots fork the champion's exact matching full training state. Existing weights,
AdamW moments/steps, replay, and league are retained. Both receive an identically
initialized, initially dormant score head and the same new seed. Training progress
resets to zero for this experiment; the iteration counter and replay ages are retained.
The only treatment difference is auxiliary loss weight 0.1 versus zero.

New finished games provide final score differences (acting player's score minus the
other player's score), including endgame bonuses. Standard, league, and bot self-play
all record these labels using the same valid/finished-position mask as ordinary replay.
The score targets are undiscounted. Shared-score games have zero score margin even
when the official completed-row tiebreak produces a sole winner. Binary value targets
remain +1/0/-1, with the existing time discount. Search continues to optimize winning.

The auxiliary head predicts margin divided by 50, with smooth L1 loss weighted by 0.1.
The loss is divided by the whole minibatch, not only its labelled subset: its effective
weight ramps naturally as fresh data replaces historical replay. Old replay entries
carry an explicit unavailable-label mask; unknown scores are never treated as ties.
Ring overwrite, oversized insertion, sampling, reanalysis, and full checkpoint resume
preserve alignment. The ordinary inference paths do not execute the auxiliary head.

Pilots save weights at 30, 60, and 120 additional training minutes. Development screens
run at 30 and 120 minutes; a head-to-head 95% upper bound below 48% stops a clearly bad
arm. Otherwise, inconclusive pilots remain eligible. The best eligible pilot continues
from its best *matching full state*, with hourly checkpoints for up to four additional
hours. Continuation shortens as needed to preserve 180 final-evaluation minutes plus
30 minutes for the last screen. All milestone weights and the best full state per arm
are retained; only this campaign's redundant scratch states and league copies are retired.

Evaluation uses fixed profiles: width 256 at 1024 simulations and width 512 at 384.
Both must complete these budgets on the local CPU within the two-second outer budget
(1.8-second internal search deadline). Development seeds are new. The saved 512
candidate receives 1024 games each against the champion and Astra, plus a historical
opponent screen. The baseline receives a matching fresh Astra evaluation; earlier
optimistic 128-game baseline scores are not reused.

Candidate ranking maximizes champion match score. Astra and historical-opponent
regressions veto a candidate only when the paired 95% interval lies entirely below
-3 percentage points. An uncertain Astra point estimate alone cannot veto a candidate.
Final selection freezes the candidate and a throughput-based count (up to 2048 games)
before any confirmation outcomes. Fresh head-to-head and matched Astra comparisons,
plus eight actual timed CPU games, decide the result. A positive primary point
advantage, Astra point loss no worse than three points, eligibility, and CPU compliance
constitute provisional improvement; a primary interval wholly above 50% additionally
constitutes resolved improvement. One training seed does not establish generality.

Run artifacts contain input hashes, source archive, validation, status, health,
supervisor events, all evaluation game records, selection decisions, and final report.
