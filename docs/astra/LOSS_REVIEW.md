# Recorded loss review

Before final testing, 17 close losses from the selected configurations' seed-4
promotion runs were replayed with their original frozen code, checkpoint bytes,
engine seeds, and bot seeds. All final scores, winners, walls, floor losses,
unfinished lines, round counts, and turn counts matched the original records.

For each player count, a deeper analysis reviewed the first 100 Astra moves in
these losses. Two-player review used 1,024,000 nodes; multiplayer review used
256,000. The normal 1,950 ms ceiling remained in force. There were 18, 28, and
22 action disagreements, respectively. The corpus is
`agent/tests/fixtures/astra_loss_review.json`; complete replay trajectories and
analysis outputs are in the experiment bundle.

These deeper choices are diagnostic hypotheses. Most still end at a heuristic
position, and even a solved nonterminal round depends on the future-position
evaluator. They are not automatically treated as corrected action labels.

## A missed forced ending

In three-player promotion game `3:opus:31:2:astra`, move 70, Astra selected center
yellow to the floor. A deeper search instead takes black from factory 1 into
pattern line 4. A completed pattern line already guarantees the game ends this
round; one factory is still active. The deeper search completes the required
Max-N tree and reaches a sole win for Astra.

`astra_terminal_win.json` preserves the exact snapshot, seed, configuration,
original choice, deeper choice, and principal variation. The regression test
checks every transition in that variation against the independent Python oracle
and checks the resulting terminal scores. This is a tactical result under Max-N's
opponent-choice assumptions, not a guarantee against every multiplayer policy.

This observation motivates the v33 conditional terminal budget. Increasing the
budget in every empty-factory position had previously failed screening. A separate
review of 21 center-only forced endings found no move disagreement; restricting
extra effort to the center would miss this factory-active example.
The conditional budget did not improve the subsequent complete-game screens at
either multiplayer count. Production therefore retains its regular budget; this
particular limited-budget miss remains a documented weakness.

## Persistent strategic uncertainty

Other disagreements include deliberate floor actions, different source choices
for the same color, and completing a small line versus investing in a larger one.
Increasing search everywhere had inconsistent multiplayer results. The evaluator
still approximates future tile availability, attainable columns/colors, and the
number of rounds before another player finishes. Those estimates and Max-N's
opponent model remain the main areas for further research.

Final-test games do not enter this corpus or trigger another tuning cycle.
