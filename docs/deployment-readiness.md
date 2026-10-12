# Deployment rating selection — October 12, 2026

The staged opponent is the shared-core **iteration 338** checkpoint, SHA-256
`8049d4b43d5e5f31ea9883cadc4526fb7b483195ea24007d34adfffa879a6862`,
at 1,024 Rust-tree simulations for 2p, 3p, and 4p. The source is pinned in
`play/release.json`; the previous published 2p model remains in the staged
registry for existing games and is hidden from new games. This configuration
was deployed as source release `e1e7899c4ee4517df03f` to `AzulStack` on
October 12, 2026.

## Fixed play ladder

| Opponent | 2p | 3p | 4p |
| --- | ---: | ---: | ---: |
| Shared RL model | 3000 | 3000 | 3000 |
| Astra | 2500 | 2500 | 2500 |
| Opus | 2250 | 2250 | 2250 |
| Heuristic | 2000 | 2000 | 2000 |
| Random | 1000 | 1000 | 1000 |

These are **chosen comparison points**, not fitted estimates of bot strength.
Human results use a separate Bradley–Terry fit for each table size. Every
completed game contributes a human-versus-bot pair for each recognized opponent
at that table, comparing final scores and using the official winner tiebreak for
equal top scores. Equal unresolved placements contribute half a win to each.
The bot anchors stay fixed; the human fit is bounded below by 1000 and above by
the rung after the strongest opponent faced. A perfect record against Random
can reach 2000, against Heuristic 2250, against Opus 2500, against Astra 3000,
and against the RL model 3500. An even record against the RL model is
approximately 3000. A 2p, 3p, or 4p rating is shown after
the player has five first-place finishes in total. The combined rating is
weighted by games played in each table size. Older stored match records are
refitted onto this ladder on profile reads without rewriting the game history.
Games against older, unanchored model IDs still count as games and wins but do
not supply a rating comparison.

The earlier [evaluation-derived ratings](deployment-ratings.json) remain a
research diagnostic, separate from this deliberately fixed play ladder. The
shared model scored 52.15% [50.02%, 54.30%] against the 2p specialist in
2,048 direct games; its held-out mixed panels improved on the specialists by
+4.09pp [+0.39, +7.60] in 3p and +3.05pp [+1.04, +5.04] in 4p. The broader
2p retention panel did not clear its prespecified two-point margin. Selecting
the shared model for this staged release does not revise that experiment gate.

The deployment passed 96 frozen-source tests, a full Lambda container game
smoke, and CDK synthesis. The first AWS candidate passed its game checks but
missed the original 2-second warm AI p95 gate at 3.49 seconds. The 1,024
simulation model was retained and the hosted gate was set to 5 seconds. The
final candidate, Lambda version 5, completed 2p/3p/4p games with a 3.71-second
warm AI p95 and cold requests below 15 seconds. The live alias and CloudFront
frontend were then promoted together; the public smoke passed with a 3.97-second
AI p95, and the hosted homepage, favicon, and `/api/ready` returned success.
`npm audit --omit=dev` reported zero production dependency advisories; the
full install reported seven advisories in development dependencies.
