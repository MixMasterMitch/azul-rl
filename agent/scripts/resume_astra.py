"""Resume an experiment using its verified archived code and native extension.

This explicit mode is useful after the working tree has moved to a new candidate.
It never mixes current decision code into the archived experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

DRIVER = r'''
import hashlib, inspect, json, platform, sys
from pathlib import Path
output = Path(sys.argv[1]).resolve()
frozen = output / "frozen"
manifest = json.loads((output / "manifest.json").read_text())
expected = manifest.get("frozen_sha256")
if not expected:
    raise ValueError("Frozen experiment has no integrity manifest")
for name, digest in expected.items():
    path = (output / name).resolve()
    if not path.is_relative_to(output) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen experiment artifacts failed integrity verification")
sys.path.insert(0, str(frozen))
import numpy as np
import torch
from agent.eval import astra_tournament as tournament
from agent.eval.heuristic_astra import AstraConfig, native_module
if manifest.get("frozen_sha256") != tournament.frozen_identity(output):
    raise ValueError("Frozen experiment artifacts failed integrity verification")
if not Path(native_module().__file__).resolve().is_relative_to(frozen):
    raise ValueError("Native extension did not load from the archived experiment")
actual = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}
for name, version in actual.items():
    if manifest["environment"][name] != version:
        raise ValueError(f"Restore recorded {name} version {manifest['environment'][name]} before resuming; found {version}")
spec = manifest["spec"]
# All executed files were verified above. Reuse their original logical names,
# which may include the original installed extension's absolute path.
tournament.source_identity = lambda *args, **kwargs: spec["sources"]
config = (AstraConfig.from_dict(spec["config"]) if spec.get("config") is not None
          else {int(n): AstraConfig.from_dict(c) for n, c in spec["configs_per_pc"].items()})
kwargs = {k: spec[k] for k in ("players", "blocks", "fields", "split", "seed", "control", "max_turns", "trace", "name", "checkpoint_policy") if k in spec}
kwargs.update(config=config, workers=int(sys.argv[2]))
if "incumbent_config" in inspect.signature(tournament.run_tournament).parameters:
    kwargs["incumbent_config"] = (AstraConfig.from_dict(spec["incumbent_config"]) if spec.get("incumbent_config") else None)
report = tournament.run_tournament(output, **kwargs)
print(json.dumps({"complete": report["complete"], "games": report["total_games"], "runtime": str(frozen)}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=4)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-c", DRIVER, str(args.experiment.resolve()), str(args.workers)], check=True)


if __name__ == "__main__":
    main()
