# Two-player capacity campaign

`agent.scripts.capacity_campaign` adds a supervised twenty-hour experiment for a
512-width `source_attn` model (10,710,118 parameters), with a 256-width control.
Its objective is playing strength within a roughly two-second move budget.
Training throughput does not determine the winner.

`agent.net.widen.widen_source_attention(teacher, 512, symmetry_noise=.01)`
uniformly replicates channels and rescales attention queries to preserve the
teacher's evaluation function. The noise sums to zero over copies of each input,
so duplicated hidden units can learn independently, including in policy/value
heads without dropout. This is an explicit transfer with a fresh optimizer;
ordinary checkpoint resumes continue to reject width mismatches.

The initializers use identical, fully search-budget-tagged replay. The campaign
trains the larger model at the inherited rate and at 0.0003, checks early
milestones, and continues from the strongest larger model's complete state. The
256-width control also starts with a fresh optimizer. The starting teacher stays
eligible throughout. All milestone weights and the best full state with its
matching league are retained for each arm. Completed experiment checkpoint and
league scratch belongs to this campaign and is retired after archival.

Development screens use 64 simulations. Finalists receive one-thread CPU timing
checks on 24 opening/middle/endgame positions, a 1.8-second search deadline, and
search-budget tuning on separate development games. Primary held-out comparisons
use each finalist's qualified budget, plus equal-64 diagnostics and 16 actual
CPU games with deadlines enabled. Primary game counts are frozen before those
games, using measured development throughput to choose up to 2,048 games per
match. Different candidate search budgets are intentional; Astra comparisons
pair the same game seeds and keep opponent settings fixed.

Local CPU qualification does not establish AWS Lambda latency. Full fixed-budget
confirmation games run in GPU batches; the CPU audit is smaller. Reports keep
provisional point-estimate improvements separate from resolved confidence
intervals. The campaign does not deploy models.

The campaign requires an input manifest, frozen transferred weights, recorded
real-replay transfer checks, a passing four-phase CUDA smoke run, `ready.json`
with the exact code provenance, and `preflight_budget.json` with the absolute
20-hour deadline including setup. Launch through
`python -m agent.scripts.supervise_training --stage capacity-campaign --hours 20`
with the usual root, device, seed, initializer, and monitoring arguments. Use
`PYTHONMALLOC=debug`, `PYTHONFAULTHANDLER=1`, and `MALLOC_PERTURB_=165`.

The September 24 run stores the exact preparation and smoke scripts, validation
logs, inputs, protocol, source archive, and persistent supervisor state under
`agent/runs/competitive_capacity_20h_20260924/`.
