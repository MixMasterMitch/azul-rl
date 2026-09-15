"""Run/resume an isolated, frozen all-agent ranking campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--blocks-2p', type=int, default=256)
    parser.add_argument('--blocks-mp', type=int, default=128)
    parser.add_argument('--mixed-blocks', type=int, default=64)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.resume:
        from agent.eval.agent_ranking import prepare
        manifest = prepare(output, blocks_2p=args.blocks_2p, blocks_mp=args.blocks_mp, mixed_blocks=args.mixed_blocks)
        print(json.dumps({'prepared': str(output), 'games': manifest['expected_games'],
                          'participants': list(manifest['participants'])}), flush=True)
    if args.prepare_only:
        return
    # Verify code before importing the archived CLI/worker. Workers inherit cwd.
    import hashlib
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest['frozen_sha256'].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen campaign artifact failed integrity verification')
    driver = ('from pathlib import Path; import sys; '
              'from agent.eval.agent_ranking import run_archived; '
              'r=run_archived(Path(sys.argv[1]),int(sys.argv[2])); '
              'print({k:r[k] for k in ("total_games","complete","unfinished","failures")})')
    subprocess.run([sys.executable, '-u', '-c', driver, str(output), str(args.workers)],
                   cwd=output / 'frozen', check=True)


if __name__ == '__main__':
    main()
