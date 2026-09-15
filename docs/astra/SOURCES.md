# Research notes

These sources informed hypotheses; they do not establish Astra's playing
strength. Tournament results provide that evidence.

- [Official Azul rules](https://cdn.svc.asmodee.net/production-nextmove/uploads/sites/4/2024/06/EN-Azul-Rules-Next-Move-web.pdf): the transition oracle covers drafting, floor penalties, ordered wall placement, final bonuses, and the complete-row tiebreak. Rust stops before random refill.
- [Competitive strategy discussion](https://www.reddit.com/r/boardgames/comments/193tk77/azul_what_almost_everyone_gets_wrong/): participants discuss connected development, columns, denying valuable placements, and accepting a floor loss to force a larger opposing loss. They disagree about fixed openings and first-player-marker priority. Their reported rankings are self-reports, not independently verified credentials. Astra therefore tests flexible evaluation terms and tactical search rather than copying one player's opening scheme.
- [Two-player strategy guide, La Tana dei Goblin](https://www.goblins.net/articoli/azul-guida-strategica-partite-due-giocatori): indexed excerpts emphasize opponent interaction, avoiding harmful final center takes, columns, and initiative near the expected final round. Direct retrieval returned HTTP 403, so this run used the available indexed excerpts, not a complete reading. Its strong claims about typical game length are treated as hypotheses, not universal rules.
- [Maturin project layout](https://www.maturin.rs/project_layout.html): the extension is an independent package, preserving the root setuptools project.
- [AWS Python container images](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html): native wheels are built in the matching Lambda base OS. Both Python 3.11/AL2 and the application's current Python 3.12/AL2023 image are validated locally; the runtime is explicitly x86-64.

The sources above support the plan's strategic and packaging choices. They do
not imply a particular Rust speed multiplier or superiority to expert humans.
Those claims would require separate measurements and opponent evidence.
