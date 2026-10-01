# Two-player serving-strength and policy distillation campaign

The 24-hour campaign continues the retained width-512 model at learning rate
0.0003 and trains a width-256 student from the retained smaller model. Both
start from their matching optimizer, replay, and league, with explicit new RNG
seeds. No production service or deployed model changes automatically.

The frozen width-512 teacher searches 512 replay snapshots at 1024 simulations
per iteration with no root noise. Its targets enter a separate 32768-position
ring buffer. Each learner batch draws 50% teacher positions and 50% ordinary
replay positions. Ordinary fast/full policy weights are normalized within that
half, so the teacher accounts for exactly half of the policy loss. All value
targets remain observed game outcomes. Teacher replay, optimizer, regular
replay, RNG, and progress are saved together in an atomic full checkpoint.

Latency qualification uses an internal 1.8-second search stop and a separate
2-second wall budget; every qualifying position must complete the requested
simulation count. Width 512 tests 256/384/512 simulations; width 256 tests
768/1024/1280. The largest passing predeclared budget is used for development, except that
1280 simulations must also beat 1024 with a paired confidence interval above
50% before the smaller model adopts it. The final challenger is requalified
without increasing its development search budget. CPU measurements are local, not AWS
Lambda benchmarks. Timed arena games report individual move timings by side.

The controller first fills the missing 512-vs-1024 search comparison. Pilots
retain snapshots at 30/90/180 training minutes, with serving-budget development
evaluations against the frozen baseline and Astra. The better approach is
repeated from its original initializer for two hours with a fresh seed, then
the best pilot's full state is continued with hourly snapshots. Continuation
uses the remaining training allowance after measured screen and final-evaluation
costs. Best full states and leagues are retained per arm; only campaign-owned
scratch and superseded best archives are retired.

Candidates, serving profiles, and game counts are frozen before confirmation.
Fresh paired seeds compare the challenger with the baseline and both with
Astra; an actual CPU audit checks the two-second limit. A positive point
advantage with no more than a three-point Astra regression is reported as
provisional. A head-to-head confidence interval above 50% resolves the primary
advantage. Timing failures block provisional selection.

The supervisor uses the preflight absolute deadline, which includes setup and
restarts. The supervisor reserves five minutes inside that deadline for a
graceful iteration boundary and full checkpoint save. Native-memory crashes stop the campaign rather than automatically
retrying. Final results are written to `distillation_campaign.json` and
`REPORT.md` inside the run directory.

## Twelve-hour continuation after a shutdown

`competitive distillation-resume --campaign-hours 12` uses a new campaign root
and a new, explicit absolute deadline. It preserves the previous run and
imports its qualified search profiles and baseline evaluation reports as
immutable historical data. The interrupted student resumes with the same
weights, optimizer, replay, teacher bank, RNG, and cumulative training time.
Only its run paths, provenance, and wall allowance change.

The student is assessed at 90 cumulative minutes. If its head-to-head 95%
interval is entirely below 48%, that branch stops; otherwise it continues to
150 minutes. The width-512 arm gets 60/150-minute checkpoints. The better
pilot receives the remaining training allowance, with hourly strength checks.
A one-hour independent replication is included only when the pilot has a
positive point advantage and is within three points of baseline Astra strength.

A measured final reserve targets about 1024 fresh primary games per comparison.
The actual count is frozen from measured runtime before seeing final outcomes.
Both finalists are rechecked for CPU latency after the reboot. The previous
baseline remains eligible, and no model is deployed automatically.
