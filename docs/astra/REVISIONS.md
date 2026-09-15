# Development record

This record separates strategic revisions from search budgets. See
[experiments.json](experiments.json) for configurations, hashes, sample counts,
latency, and complete numerical summaries. The raw records and frozen runtime
for each entry are in its listed experiment directory. No final-test seed has
been used during these revisions.
Configuration names alone do not identify an old implementation: reproduce a
historical result with its archived source and native binary, including the
correctness fixes and search behavior recorded for that run.

| Revision | Hypothesis / change | Evidence and decision |
| --- | --- | --- |
| v1 | Exact projected scores plus partial lines, adjacency, flexibility, marker, and bonus estimates; 4,000-node search. | Strong two-player results, weak multiplayer. Depth-zero comparison established that search mattered, especially at two players. |
| v2 | Incomplete lines were receiving too much optimistic completion credit. Discount the chance using investment and competing commitments. | Two-player Opus share rose from 85.9% to 96.9%; three-player from 21.7% to 45.3%; four-player from 10.2% to 20.7% on the 32-block development schedule. Retained as the first baseline. |
| v3 | More selfish multiplayer utility might predict opponents better. Reduce opponent coefficient from 1 to 0.25. | Higher own scores did not translate into higher win share. Rejected. |
| v4 | Reduce partial-line value further, to 0.4. | No convincing improvement. Rejected. |
| v5 | Increase adjacency weight from 0.25 to 1. | Substantial three-player regression. Rejected. |
| v6 | Double the general bonus weight. | Some four-player improvement, inconsistent elsewhere. Replaced by a more specific hypothesis. |
| v7 | Estimate column/color completion using remaining rounds and the difficulty of missing rows. | Fresh 128-block promotion against Opus and mixed fields passed at three and four players. The two-player improvement was inconclusive. Retain per player count. |
| v8 | At search leaves, greedily finish the current round to predict turn order and leftovers. | Regressed sharply in multiplayer; rejected. The optional rollout remains available for reproducing the ablation, disabled by default. |
| v9 | Review alpha-beta correctness: antisymmetric two-player terminal values; cache only exact transposition values inside the original search window. | Independent exhaustive round tests pass. A later proof-aware solved flag allows alpha-beta cutoffs derived entirely from resolved leaves to certify the root decision. |
| v9 budgets | Compare 4k, 16k, 64k, and 256k nodes with fixed v7 weights. | More computation was not consistently stronger. 16k was the promising four-player budget; 256k did not justify becoming a universal default. |
| v10 | An incomplete line can unlock a column or color bonus. Add its probability-weighted bonus contribution. | Three-player Opus screening improved, but mixed-table results declined. Broader promotion did not establish an improvement. Rejected. |
| v11 | The minimum number of missing row tiles may underestimate future play. Use a somewhat longer strategic horizon (`urgency=0.7`). | Passed the fixed-reference, 128-block promotion at three and four players. Two-player improvement remained inconclusive. |
| v12 | Double the penalty for blocking an unfinished pattern line. | No compelling screening improvement over v7. Rejected. |
| v13 | Spend extra nodes only when factories are empty, to solve forced leftovers without deepening speculative opening evaluations. | 256k center-only nodes did not improve the selected configurations on the matched 32-block screen. Rejected. |
| v15 | Revisit double bonus weight with the selected longer-horizon evaluator. | Regressed at two and three players; four-player improvement against Opus was offset by weaker mixed-table results. Not promoted. |
| v16 | Try the longer horizon alone at two players, without the column prior. | Little screening change against Opus/mixed fields; checkpoint improvement was small. Not promoted. |
| Evaluation reuse | Profiling showed repeated leaf evaluation after move ordering. Pass the already computed value to a static leaf. | All 300 sampled actions, node counts, depths, values, and principal variations matched exactly. Opening/middle move times fell about 9–31% in the paired microbenchmark. Retained. |
| Completion pruning | Round resolution clears completed pattern lines before reply pruning inspects the child. | Preserve those completions by checking the newly placed wall cell. A dedicated regression test passes; the fix is applied after the selected-weight budget screen. |
| v17 | Account for all close multiplayer rivals using a smooth aggregate instead of only the largest rival estimate. | With a manually chosen eight-point scale, four-player screening rose to 49.6% against Opus and 57.4% in mixed tables, from 41.0% / 51.6%. Three-player results were mixed. 128 fresh promotion blocks did not establish an aggregate gain at either player count; rejected. Disabled at two players. |
| v18 | Extend the strategic horizon further, from urgency 0.7 to 0.5. | Regressed in the mixed three-player field and both four-player fields. Rejected. |
| v19 | Increase multiplayer denial through opponent weight 1.4. | Regressed at three players; small four-player screening improvement. Not selected without promotion. |
| v20 | Use bonus weight 1.5, between the incumbent and the previously rejected 2.0. | Four-player Opus screening improved, but mixed results regressed. The field-pressure revision was stronger on the same screening seeds. Not selected. |
| v21 | Add opponent weight 1.4 to the field-pressure evaluator, restoring more attention to the leader while accounting for other rivals. | Screening did not improve on v17; rejected. |
| v22 | Add bonus weight 1.5 to v17. | Four-player mixed screening improved, but 128 fresh promotion blocks did not establish a gain over the incumbent; rejected. |
| v23 | Blend the original maximum-rival estimate and the new field estimate equally. | No aggregate screening gain over v17; rejected. |
| 1,024k nodes | Check whether a near-deadline two-player search outperforms 256k. | Opus share was 100%, mixed 99.2%, and checkpoint 92.2%; 256k had scored 96.9% against the checkpoint. Mean move time was 0.51–0.55 seconds, with 120 time cutoffs in 4,989 moves. No convincing aggregate gain; not advanced. |
| v7/v11 at 256k | Check whether attainable bonuses or a longer horizon become useful at a larger two-player search budget. | v7 gave no aggregate screening gain; v11 regressed against the checkpoint. Neither advanced. |
| Color-supply deadlocks | Check whether globally trapped colored tiles make some incomplete lines impossible to finish. | No such commitment was found in 625 recorded Astra positions before projected game end, including 390 positions from losses. The extra evaluator logic was not added. |
| Evaluation cache | Reuse exact static values when boards and total remaining colored tiles agree, despite different source distributions. | A universal cache slowed two-player searches. Restricting it to multiplayer searches starting with factories improved opening/middle benchmarks by 1.2–1.9×, with essentially unchanged two-player timings. All 300 sampled decisions and fixed-node diagnostics matched. Peak native benchmark RSS was about 48 MiB, versus 17 MiB without the cache. Retained conditionally. |

The v7 promotion against Opus and the mixed field produced paired improvements
of +10.16 percentage points at three players (95% CI +5.47 to +14.84) and +15.82
points at four players (+12.01 to +19.68). The two-player interval crossed zero.
These are promotion results, not held-out final claims.

Concurrent work changed the repository's default engine and reference-policy adapters
during a subsequent broader campaign. Cross-source comparisons were rejected;
those `broad-*`/`v11-promotion-*` runs remain archived but are superseded by the
`fixed2-*` campaign. Every fixed campaign reuses the engine, reference bots,
neural inference code, and checkpoint bytes from `broad-baseline-2p`.

On 128 fresh seed-2 promotion blocks, averaged over Opus, mixed, and checkpoint
fields, v11 improved on v7 by **+3.08 percentage points** at three players
(paired 95% CI **+0.35 to +5.86**) and **+5.34 points** at four players
(**+2.51 to +8.07**). Neither had a significant Opus regression. At two players,
v7 versus v2 was +1.50 points (−0.91 to +3.78), and v11 versus v2 was +0.72
(−1.69 to +3.13); v2 remains selected there. These comparisons include the
corrected multiplayer terminal utility: a shared win above the symmetric
`1 / players` baseline dominates positional estimates, and nonterminal values
are bounded below the terminal scale.

Loss inspection found recurring multiplayer games in which Astra built two
complete upper rows but no bottom-row tile, while the winner earned one or more
column bonuses. For example, promotion game `4:opus:0:0:astra` ended 57–75–63–62;
Astra's row occupancies were 5/5/2/4/0, and its opponent's 5/4/4/3/1. This motivated
bonus attainability and partial-line bonus experiments rather than a fixed opening.

Checkpoint diagnostics explicitly distinguish greedy inference from 32-simulation
search. The strongest available checkpoint has a two-player rating. Three- and
four-player selections fall back to the strongest overall checkpoint, whose
multiplayer results are weak; they provide limited evidence of competitive
multiplayer strength.

On promotion seed 3, field pressure versus the incumbent gave aggregate deltas
of +1.87 points at three players (95% CI −0.74 to +4.60) and +0.88 points
at four players (−1.40 to +3.06). Adding bonus weight 1.5 gave +1.06 points
at four players (−1.31 to +3.35). None passed the promotion gate.

The selected two-player budget increased from 4,000 to 256,000 nodes after
128 fresh seed-3 promotion blocks. The average competitive-field improvement
was **+5.73 percentage points** (paired 95% CI **+3.52 to +7.94**), driven chiefly
by +14.06 points against the search-enabled checkpoint. Opus improved by +1.56
points without a significant regression. Mean moves took 0.18–0.19 seconds;
the largest observed move was 1.34 seconds. The 16,000-node candidate did not
pass its promotion gate (+1.04 points, CI −1.17 to +3.26). Multiplayer retains
4,000 nodes because larger-budget screens did not establish a consistent gain.

## Finalist checks and remaining hypotheses

The seed-20 multiplayer ablations remove one feature at a time from the selected
v11 evaluator at 4,000 nodes. None improved average win share across Opus, mixed,
and checkpoint fields. Removing bonuses reduced four-player Opus share from
49.6% to 14.1%; removing the attainability prior reduced it to 26.2%. The same
32-block screen also favored retaining adjacency, opponent awareness, initiative,
safe capacity, unfinished-line penalties, and the longer horizon. These are
exploratory paired comparisons, not final-test claims.

The current losing Astra boards no longer show the original bottom-row deficit.
Against Opus at four players, losing Astra boards averaged 1.20 bottom-row tiles,
versus 0.94 for the winning opponent, but scored 62.0 versus 74.3 points. Astra
also retained more unfinished lines. These conditional summaries motivate
further tests; they do not establish which move caused a loss.

- **Previous-iteration root ordering:** try the prior iteration's best two-player
  move first, seeking better alpha-beta cutoffs at the same node budget. This is
  isolated from production because tie choices and completed depth can change.
  On the 32-block screen it matched the incumbent against Opus/mixed tables but
  scored 89.1% versus 92.2% against the checkpoint. Aggregate delta −1.04 points
  (95% CI −3.13 to +1.04); rejected despite a small latency reduction.
- **Stopping a completed pruned tree:** stop deepening once every required bound
  in the retained tree comes from a resolved round. Keep `solved=false` and label
  the cutoff `pruned_round` if reply pruning excluded alternatives. A fixture
  preserves the move/value/PV while reducing visits from 4,619 to 211. The
  broader equivalence checks below passed, and the change was retained.
- **v25:** increase initiative weight from 0.6 to 1.2. The multiplayer ablation
  suggests the first-player marker contributes useful value, especially at four
  players. Matched screening and fresh promotion are reported below.
- **v26:** revisit partial-line bonus credit conservatively at weight 0.25, with
  the selected longer horizon. The previous weight-1 experiment was rejected.
- **v28/v29:** replace or blend the adjacency-edge count with the scoring premium
  of the best two future wall connections in each row. This tests whether joining
  scoring chains is a better estimate than rewarding many adjacent empty cells.
  The isolated API-4 prototype adds `chain_adjacency` in [0,1]. Its screens did
  not justify promotion, so production remains API 3.

The completed-pruned-tree stop was retained after **400 identical position
choices/values/PVs** and **864 identical complete-game outcomes**. It stopped
early on 12 of those positions, saving 1,399,042 node visits. Two-player mean
move times in the complete-game check fell from about 0.18–0.19 to 0.16–0.17
seconds; this is a performance change with unchanged tested decisions.

The completed v25–v29 screens use the same 32 seed-20 blocks as the finalist
ablations. Doubling initiative produced aggregate gains of +2.08 points at
three players and +3.65 at four players; both advance to fresh seed-4 promotion.
Partial-line bonus credit at 0.25 was nearly neutral at three players and
regressed at four players (−4.82 points, CI −8.07 to −1.82). The complete
wall-chain replacement and the half blend did not improve aggregate results;
the API-4 prototype is rejected and production remains API 3.

At two players, removing adjacency at 256k nodes was the only ablation with a
positive screening aggregate (+0.78 points, CI −2.34 to +4.17). It advances to
fresh promotion against the unchanged selected evaluator. Removing search
reduced aggregate share by 41.67 points (CI −48.96 to −34.11). The other feature
removals were neutral or negative and do not change production.

Fresh seed-4 promotion retained v25 **only at four players**. Its aggregate
competitive-field gain was **+2.97 percentage points** (paired 95% CI **+0.81 to
+5.14**), with +4.98 points against Opus and +3.94 in mixed tables. At three
players the aggregate was +1.82 points (−0.69 to +4.30), so initiative remains
0.6 there. Four-player initiative is now 1.2. Each comparison contains 128 full
seed/seat blocks per field, with no unfinished games or agent failures.

A local disk-space interruption affected the baseline's append operation during
a Lambda builder creation. Resume retained 2,443 complete records, removed a
three-byte partial append, and completed the same 2,688-game schedule. Frozen
code and seed identities did not change; the interrupted run was not replaced
with a more favorable schedule.

The two-player no-adjacency candidate failed fresh seed-4 promotion. Its aggregate
change was **−2.47 percentage points** (95% CI **−4.75 to −0.26**), including
−5.47 points against the search-enabled checkpoint. Adjacency remains 0.25 at
256k nodes. This illustrates why a small positive ablation screen does not itself
justify a production change.

The final search screens compare reply widths 6 and 24 with width 12 at unchanged
node budgets and evaluator weights, using 32 fresh seed-31 development blocks.
A separate multiplayer candidate tests initiative 1.8 against the independently
selected 0.6/1.2 defaults. These are manually specified hypotheses; no optimizer
selects weights or searches parameter space.

The width screens advance **width 6 at two players** (+1.04 aggregate points,
95% CI −1.56 to +3.65) and **width 24 at three players** (+0.87 points,
approximately 0 to +1.91) to seed-5 promotion. Neither screen changes production.
Width 24 at two players, width 6 at three players, and both width changes at four
players failed to improve aggregate share. Raising initiative to 1.8 also failed
at both multiplayer counts. Complete paired screening results are retained in
`late-screen-decisions.json`.

A deterministic replay of 17 close losses matched every original outcome. Deeper
review inspected 100 positions per player count, producing 18, 28, and 22 move
disagreements. These positions are retained as a diagnostic corpus, without
pretending every deeper recommendation is correct. At forced endings with only
center tiles, all 21 reviewed moves agreed, so there is no evidence for a
center-only terminal-budget rule.

The broader review did find a three-player forced ending with **one factory still
active** where 4k nodes missed an exact Max-N winning continuation. This became
`astra_terminal_win.json`; the regression test checks every PV transition against
the independent Python oracle and the resulting terminal score. The manually
specified **v33** hypothesis raises the budget to 256k only when a completed
pattern line already guarantees the round ends the game. It uses public state,
keeps the same wall-clock ceiling, and is disabled by default pending experiments.

Fresh seed-5 promotion rejected three-player width 24: aggregate change **−0.30
percentage points**, 95% CI **−1.30 to +0.74**, with no checkpoint change. The
selected width remains 12. The two-player width-6 comparison is still pending.

The final summarizer sorts seed-block IDs before its seeded bootstrap, so worker
completion order cannot change its finite-resample confidence interval. A shuffled
completion-order regression verifies this. Paired promotion comparisons already
sorted their blocks, and their decisions are unchanged. Historical reports remain
with their archived summarizer and raw append order.

Fresh seed-5 promotion also rejected two-player width 6: aggregate change **+0.46
percentage points**, 95% CI **−0.98 to +1.89**. The checkpoint gain was only +0.39
points (−3.52 to +4.30), and improvement was not established. Reply width remains
12 for all player counts. All 3,840 games in the two seed-5 comparisons completed
without failures; the two-player games had no wall-clock cutoffs.

The v33 terminal-budget screens did not justify promotion. At three players the
aggregate change was **−1.56 percentage points** (95% CI **−4.51 to +1.39**); at
four players it was **−0.52 points** (**−3.26 to +2.08**). Against Opus, mean move
latency rose to about 24–26 ms from about 3–4 ms. Neither candidate improved the
screening aggregate, so no fresh promotion was run. `terminal_nodes` remains zero
in production. The recorded winning continuation remains a known limited-budget
miss; it does not justify assuming deeper Max-N is better across complete games.

The final four-player ablations use the selected initiative weight 1.2. Removing
bonus potential, the attainability prior, adjacency, opponent pressure, or
initiative reduced aggregate share by approximately 16.80, 16.28, 13.80, 10.07,
and 9.11 points, respectively; restoring the shorter horizon lost 9.90 points.
Their paired intervals exclude zero. Search, partial credit, and line-lock
removals had negative but inconclusive aggregates. The only positive removal was
safe-placement capacity: **+0.65 points** (95% CI **−2.08 to +3.26**). It advances
alone to fresh seed-6 promotion; production is unchanged until that gate passes.
Complete per-feature results are in [ABLATIONS.md](ABLATIONS.md).

Fresh seed-6 promotion rejected four-player safety removal. Its aggregate gain
was **+0.81 percentage points** (95% CI **−0.55 to +2.18**), so the original gate
was not met. Opus share improved by +4.39 points (+1.56 to +7.32), but mixed-table
share fell by 1.95 points (−4.79 to +0.98). The fixed-field criterion prevents
selecting the candidate from its better-looking Opus result alone. Safety remains
0.35, and no further strategic revision is made before the final freeze.

Production **astra-1.0.0** and its full decision runtime were frozen at **2026-09-14T06:57:24.567714+00:00**, 7.89 hours after development began. The selected configurations retain width 12, regular budgets 256k/4k/4k, initiative 0.6/0.6/1.2, and safety 0.35. No strategy revision follows the first held-out game.
