"""Apply one pre-recorded training amendment at a durable checkpoint.

The supervisor owns retry and restart policy.  This helper merely waits for a
known-good resume checkpoint, records the amendment, then interrupts its
campaign child so the next supervised process imports the amended code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from agent.train.checkpointing import load_checkpoint_payload


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    args = parser.parse_args()
    if args.iteration < 1 or args.poll_seconds <= 0:
        parser.error('iteration and poll-seconds must be positive')

    root = args.root.resolve()
    request = root/'training_amendment_request.json'
    result = root/'training_amendment_result.json'
    started = time.monotonic()
    while not result.exists():
        for checkpoint in root.glob('experiments/*/checkpoints/latest_resume.pt'):
            try:
                payload = load_checkpoint_payload(checkpoint, map_location='cpu')
                iteration = int(payload.get('iteration', 0))
            except Exception:
                continue  # Never interrupt based on a partial or unreadable archive.
            if iteration < args.iteration:
                continue
            pid_path = root/f'{args.stage}.pid'
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                continue
            amendment = json.loads(request.read_text()) if request.exists() else {}
            record = {
                'status': 'applied', 'checkpoint': str(checkpoint), 'iteration': iteration,
                'pid': pid, 'requested': amendment,
                'applied_at': datetime.now(timezone.utc).isoformat(),
                'wait_s': round(time.monotonic() - started, 3),
            }
            _write_json(result, record)
            # The checkpoint is already atomically visible.  Let the existing
            # supervisor account for the non-zero exit and start a fresh child.
            os.kill(pid, signal.SIGKILL)
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
