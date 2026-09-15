# Astra Heuristic

Astra is a hand-coded Azul agent for two, three, and four players. Rust owns the
current-round simulator, evaluator, alpha-beta/Max-N search, and transposition
table. Python orchestrates real games, experiments, and the play service.
Astra uses no neural network, trained parameters, runtime LLM, or network calls.

The experiments keep the explicit PyTorch `BatchedEngine` as their authoritative
engine and freeze its source to preserve seeded deals. Concurrent project work
migrated the application's default simulator to Rust; Astra also supports that
`GameEngine` through its public-snapshot interface. In both cases the game engine
owns actual refills and RNG, which are separate from Astra's search state.

The frozen `astra-1.0.0` configuration achieved **99.02%, 56.08%, and 47.12%**
win share against homogeneous Opus tables at two, three, and four players.
All three passed the prespecified superiority criterion. The [final report](RESULTS.md)
contains confidence intervals, mixed controls, latency, ablations, and limitations.
All 4,608 final games completed without failures; none fed another tuning cycle.

The subsequent [expanded ranking campaign](RANKINGS.md) adds **19,456 games**
across the full eligible field, including two neural agents with their registered
64-simulation search profiles. Astra ranked first at every player count; the
ordering of the two neural agents remains uncertain. The original development
results and this additional ranking dataset are kept separate.

Current rating tables use the [heuristic-2500 display scale](RATING_SCALE.md),
which keeps random at 1,000 and places the frozen reference heuristic at 2,500.
The statistical fits, raw experiment records, and win shares are unchanged.

New training evaluations also sample Astra as a fixed opponent and record its
win share separately. See [training evaluation integration](TRAINING_EVALUATION.md)
for the default opponent fraction, configuration, paired arena, and resume behavior.

## Strategy translated into code

The [official rules](https://cdn.svc.asmodee.net/production-nextmove/uploads/sites/4/2024/06/EN-Azul-Rules-Next-Move-web.pdf)
are authoritative for transitions and scoring. Completed lines resolve from the
smallest to the largest, then floors and score clamping apply, then end detection
and bonuses. Ties use complete horizontal rows before shared victory.

The [competitive discussion](https://www.reddit.com/r/boardgames/comments/193tk77/azul_what_almost_everyone_gets_wrong/)
and [experienced-player guide](https://www.goblins.net/articoli/azul-guida-strategica-partite-due-giocatori)
informed hypotheses, rather than fixed opening rules:

- Connected placements increase future scoring opportunities. An adjacency
  feature values reusable upper lines, while exact resolution scores wall chains.
- Columns and color sets need attainable placements in the larger pattern lines.
  A hand-set completion prior, tested experimentally, discounts bonuses that require too many
  difficult placements before somebody completes a horizontal row.
- Unfinished commitments consume flexibility. Completion prospects account for
  tile availability, competing commitments, and the fraction already collected.
- A floor sacrifice can secure a better draft sequence. Floor moves stay legal
  and participate in search; there is no blanket prohibition on taking them.
- The marker combines a real floor penalty with position-dependent initiative.
- Ending the game is a decision about everyone's outcome. A terminal win share
  above the symmetric `1 / players` baseline dominates positional values; an
  all-player shared draw is neutral. A losing player can extend play.

The evaluator exposes named weights in `agent/eval/heuristic_astra.py`. Search
and evaluation changes are tested separately. All numerical revisions are
manual; there is no automatic weight optimizer.
The [research notes](SOURCES.md) distinguish competitive anecdotes from official
rules and document the strategy guide's restricted access.
The [loss review](LOSS_REVIEW.md) records reproducible game replays, deeper-search
disagreements, and a verified tactical regression without treating every deeper
choice as a correct target.

## Build and use

See [the native build guide](../../native/astra/README.md). The root package uses
setuptools; the independent `native/astra` package uses pinned Rust, PyO3, and
maturin. A prebuilt wheel needs no Rust compiler on the host.

```python
from agent.eval.heuristic_astra import HeuristicAstraBot, AstraConfig

bot = HeuristicAstraBot(seed=17)  # Defaults selected separately per player count.
action = bot.select_action(engine, game_idx=0)
diagnostics = bot.analyze(engine, game_idx=0)

# Explicit configuration overrides all player-count defaults.
controlled = HeuristicAstraBot(seed=17, config=AstraConfig(nodes=16000))
```

The play opponent ID is `astra`, displayed as **Astra Heuristic**. Native imports
are lazy. If the extension is missing, the play catalog omits it and an explicit
request reports the missing extension instead of substituting another bot.

The snapshot contains public boards, scores, sources, floor counts, and marker
ownership. It contains no bag, discard composition, future draw, or generator
state. Rust stops before the next refill. Both authoritative engines and an
independent compact Python oracle verify native transitions before refill.

## Experiment contract

`python -m agent.scripts.eval_astra --help` documents the independent runner.
Each output directory contains a manifest, frozen code and extension, copied
checkpoint bytes, append-only game records, and a summary. SHA-256 identities
prevent silent changes on resume. A partial final JSONL append can be recovered;
duplicate records, changed configurations, or changed frozen artifacts are errors.
No experiment writes the training league or historical ratings.

A seed block includes every designated seat under one deal seed. Independent
bot seeds are recorded separately. Candidate and control play the same seat
under matching schedules. Development, promotion, and final seeds occupy
separate ranges. Four spawned workers each use one PyTorch thread.

The primary outcome is win share: sole victory is one, a shared victory is
`1 / number_of_winners`, and a completed loss is zero. A capped game has no
winner and a null share; it is recorded as unfinished and blocks promotion.
The runner reports an observed lower bound and worst/best possible unfinished
outcome bounds instead of quietly deleting those games.

Confidence intervals use 10,000 bootstrap replicates of complete seed/seat
blocks. Paired comparisons resample block differences. Screening uses 32 blocks;
promotion uses 128 fresh blocks against matching fields. Promotion requires a
positive paired 95% lower bound across the field and no statistically significant
regression against Opus. Frozen checkpoints use the highest available per-count
rating, with the strongest overall checkpoint as an explicitly recorded fallback.
When every observed block has the same outcome, the empirical bootstrap interval
collapses to that value; a 100% observed result does not prove perfect play.

Final targets are 256 blocks per player count against homogeneous Opus tables
and 64 mixed blocks with an Opus control in the designated seat. Superiority is
claimed separately for each player count only if the final lower bound exceeds
`1/2`, `1/3`, or `1/4`. Final games do not feed another tuning cycle.

`agent.scripts.rate_astra` fits fresh per-count Bradley–Terry diagnostics from
pairwise final score/row placements, anchored to random=1000. A weak prior keeps
separated results finite. Those ratings are secondary and are not comparable to
historical league ratings.

## Measurement limits

Node counts include visited search states and optional rollout steps; move-ordering
evaluations are additional work. Fixed-node runs are deterministic for fixed seeds
unless the wall-clock deadline fires. A timeout returns the best completed
iteration, with an immediately available legal static choice.

An exact round result ends at the current refill boundary. It does not solve
the full game. Multiplayer Max-N assumes each opponent pursues its own evaluated
outcome and uses deterministic tie choices; it does not guarantee the same result
against an opponent choosing a different policy or cooperating with another seat.

Transition benchmarks compare equivalent compact Python and Rust operations;
the native timing includes Python binding and validation. They do not establish
a Rust-versus-Python search speed multiplier. Full move timing includes snapshot
conversion and authoritative legality verification.

Floor losses record the scheduled floor penalty before score clamping. This can
exceed the number of points actually removed from a player near zero. Unfinished
lines count pattern lines still holding tiles at the observed endpoint.

Local Lambda handler tests use a temporary JSON store. They include request
serialization, persistence, and consecutive bot turns, but do not measure AWS
cold starts, network latency, or DynamoDB. No live deployment is performed.

## Reproducible delivery

Raw records stay under `agent/runs/astra/<experiment>/games.jsonl`, alongside the
manifest, block summaries, frozen runtime, and checkpoint hashes. The compact
[experiment index](experiments.json) records each run's configuration, source
identity, sample count, and numerical results. Historical league data is separate.

Create the portable local bundle with:

```bash
python -m agent.scripts.export_astra --output agent/runs/astra/astra-delivery.zip
```

The bundle includes all experiment records and reports, final experiments with
complete resumable runtimes, release wheels, and this documentation. Historical
frozen runtimes remain in their original local directories. To resume a relocated
final experiment, use `agent.scripts.resume_astra` on its extracted experiment
directory with the recorded Python, PyTorch, and NumPy versions.
