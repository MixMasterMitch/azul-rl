"""Check clean wheel installation, lazy imports, and a move without PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

MISSING = r'''
import importlib.util
from agent.eval.heuristic_astra import AstraUnavailableError, native_module
assert importlib.util.find_spec("torch") is None
assert importlib.util.find_spec("numpy") is None
try:
    native_module()
except AstraUnavailableError as error:
    assert "Rust extension" in str(error)
else:
    raise AssertionError("Missing extension was not reported")
'''

MOVE = r'''
import hashlib, importlib.util, json, sys
from importlib.resources import files
from agent.eval.heuristic_astra import AstraConfig, HeuristicAstraBot, native_module, production_config
assert "torch" not in sys.modules and "numpy" not in sys.modules
native = native_module()
snapshot = [1, 2, 0, 0, 1, 0, 0] + [0] * 50 + ([0] * 9 + [-1] * 5) * 2
snapshot[7] = 4
legal = native.legal_actions(snapshot)
class Mask:
    def __getitem__(self, key: tuple[int, int]) -> bool:
        index, action = key
        return index == 0 and action in legal
class PublicEngine:
    num_players = 2
    def public_snapshot(self, index: int) -> list[int]:
        assert index == 0
        return snapshot.copy()
    def legal_action_mask(self) -> Mask:
        return Mask()
result = HeuristicAstraBot(seed=7, config=AstraConfig(nodes=4000)).analyze(PublicEngine(), 0)
assert result["action"] in legal
assert "torch" not in sys.modules and "numpy" not in sys.modules
print(json.dumps({"action": result["action"], "native_api": native.CONFIG_VERSION,
                  "native_path": native.__file__, "python": sys.version,
                  "torch_loaded": False, "numpy_loaded": False,
                  "production_config_sha256": hashlib.sha256(files("agent.eval").joinpath("astra_configs/production.json").read_bytes()).hexdigest(),
                  "production_nodes": {str(n): production_config(n).nodes for n in (2, 3, 4)}}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-wheel", type=Path, required=True)
    parser.add_argument("--native-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.root_wheel) as wheel:
        names = wheel.namelist()
        assert "agent/eval/astra_configs/production.json" in names
        assert not any({"runs", "tests", "play_data", "artifacts", "target"}.intersection(Path(name).parts) for name in names)
    with tempfile.TemporaryDirectory(prefix="astra-wheel-smoke-") as temporary:
        directory = Path(temporary)
        environment = directory / "venv"
        # The existing pip can target a clean interpreter; no ensurepip package
        # or network bootstrap is needed on minimal Debian/Ubuntu installations.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
        python = str(environment / "bin/python")
        def install(path: Path) -> None:
            subprocess.run([sys.executable, "-m", "pip", "--python", python,
                            "install", "--no-index", "--no-deps", str(path.resolve())],
                           cwd=directory, check=True, stdout=subprocess.DEVNULL)
        install(args.root_wheel)
        subprocess.run([python, "-c", MISSING], cwd=directory, check=True)
        install(args.native_wheel)
        result = json.loads(subprocess.check_output([python, "-c", MOVE], cwd=directory, text=True))
    result.update(root_wheel=str(args.root_wheel.resolve()), native_wheel=str(args.native_wheel.resolve()),
                  root_wheel_sha256=hashlib.sha256(args.root_wheel.read_bytes()).hexdigest(),
                  native_wheel_sha256=hashlib.sha256(args.native_wheel.read_bytes()).hexdigest(),
                  root_wheel_files=len(names), missing_extension_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
