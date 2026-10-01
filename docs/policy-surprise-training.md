# Policy-surprise sampling

The two-player experiment records `KL(search target || raw network prior)` for
finished-game positions searched with at least 256 simulations. Fast targets and
legacy replay have unknown surprise and retain ordinary sampling weight. The
network is unchanged throughout a self-play trajectory, so priors are recomputed
in inference batches after finished samples are selected. Reanalysis replaces
surprise together with the improved policy and search-budget metadata.

This follows the policy-surprise idea described in
[KataGo's methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#policy-surprise-weighting),
with a cap and neutral handling of existing fast-search replay:

```
full-search sampling weight = min(4, 0.5 + 0.5 * KL / mean_eligible_replay_KL)
fast-search or unknown weight = 1
```

The prior is the raw, untempered network policy; the target retains the training
search's temperature and other settings. Thus this measures the complete target
correction, including sharpening, rather than isolating the contribution of extra
simulations. No inference or game-rule change is involved.

Rejection sampling applies these weights to complete training examples, including
both policy and value learning. There is no importance correction. The existing
0.25 fast-search *policy-loss* weight remains a separate setting. Weights have a
0.5 floor and fourfold cap before probability normalization. The control arm records
identical metadata and samples uniformly. Missing surprise is persisted with an
explicit validity mask. Replay overwrites and reanalysis invalidate the cached mean;
RNG checkpoints reproduce sample selection after a restart.

The CLI settings are `--policy-surprise-record`, `--policy-surprise-fraction`,
`--policy-surprise-min-sims`, and `--policy-surprise-max-weight`. Defaults leave
ordinary sampling unchanged. These settings are immutable within a resumed run;
changing them requires an explicit full-state fork. Combining biased sampling with
the separate distillation bank is currently rejected.

The `surprise` campaign checks native reliability before entry, confirms the saved
width-512 and auxiliary-score candidates on fresh seeds, then forks two matched
width-256 arms from the same verified optimizer/replay/model/league state. The score
recipe is used only if its saved candidate beats both the champion and its matched
historical control by point estimate, without the specified opponent regression.
Development and final confirmation seeds are disjoint. Final evaluation and actual
CPU moves are reserved inside the absolute twenty-hour budget. No model is deployed
automatically. Checkpoints are written every ten iterations and at each segment end;
a native memory fault stops the campaign without blind retries.
