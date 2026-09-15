"""Bundle Astra records, final runtime, wheels, and reports for local delivery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile


def export(root: Path, output: Path, documents: Path) -> dict[str, int]:
    """Use an explicit allowlist; historical source archives are not exported."""
    from agent.eval.astra_tournament import frozen_identity

    entries: list[tuple[Path, str]] = []
    experiments = 0
    for report in sorted(root.glob("*/report.json")):
        experiments += 1
        for name in ("games.jsonl", "manifest.json", "report.json"):
            path = report.parent / name
            entries.append((path, f"experiments/{report.parent.name}/{name}"))
        if json.loads(report.read_text())["spec"]["split"] == "final":
            manifest = json.loads((report.parent / "manifest.json").read_text())
            if manifest["frozen_sha256"] != frozen_identity(report.parent):
                raise ValueError("Final experiment integrity check failed")
            entries.append((report.parent / "sources.zip", f"experiments/{report.parent.name}/sources.zip"))
            for folder in ("frozen", "checkpoints"):
                for path in sorted((report.parent / folder).rglob("*")):
                    relative = path.relative_to(report.parent)
                    if path.is_file() and "__pycache__" not in relative.parts:
                        if "play_data" in relative.parts:
                            raise ValueError("Unexpected application data in final experiment")
                        entries.append((path, f"experiments/{report.parent.name}/{relative}"))
    for path in sorted(root.glob("*.json")):
        entries.append((path, f"evidence/{path.name}"))
    for folder in ("validation-evidence", "release-wheels"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                entries.append((path, str(path.relative_to(root))))
    for name in ("native-performance-prototypes.zip", "late-search-prototypes.zip"):
        path = root / name
        if path.exists():
            entries.append((path, f"prototypes/{name}"))
    runtime = root / "final-runtime"
    if runtime.exists():
        manifest = json.loads((runtime / "manifest.json").read_text())
        if manifest["frozen_sha256"] != frozen_identity(runtime):
            raise ValueError("Final runtime integrity check failed")
        for path in sorted(runtime.rglob("*")):
            relative = path.relative_to(runtime)
            if path.is_file() and "__pycache__" not in relative.parts:
                if "play_data" in relative.parts:
                    raise ValueError("Unexpected application data in final runtime")
                entries.append((path, f"final-runtime/{relative}"))
    for path in sorted(documents.iterdir()):
        if path.is_file() and path.suffix in (".md", ".json"):
            entries.append((path, f"report/{path.name}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, name in entries:
                archive.write(path, name)
            archive.writestr("README.txt", (
                "Astra Heuristic development records and final delivery.\n"
                "Historical experiment directories contain raw games, manifests, and reports.\n"
                "Their full frozen runtimes remain at the original local experiment paths.\n"
                "The final-runtime directory includes verified decision code and checkpoints.\n"
                "Final experiments also include their complete resumable runtimes.\n"
                "The native build guide is final-runtime/frozen/native/astra/README.md.\n"
                "On Linux x86-64 with Python >=3.11, install the bundled root wheel and a compatible native wheel.\n"
                "The al2 wheel needs glibc >=2.26; the al2023 wheel needs glibc >=2.34.\n"
                "Restore the manifest's Python/Torch/NumPy versions before resuming.\n"
                "From this bundle directory: python -m agent.scripts.resume_astra experiments/final-opus --workers 4\n"
                "For a source build, use final-runtime/frozen as the repository root.\n"
                "Frozen binary replay requires a compatible host platform; a new build belongs in a new experiment.\n"
                "Report confidence intervals resample complete seed/seat blocks.\n"
            ))
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"experiments": experiments, "files": len(entries) + 1, "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("agent/runs/astra"))
    parser.add_argument("--documents", type=Path, default=Path("docs/astra"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output, args.documents)))


if __name__ == "__main__":
    main()
