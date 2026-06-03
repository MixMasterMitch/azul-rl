"""Run a short profile cycle for throughput comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile benchmark helper")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--init-from",
        type=str,
        default="agent/runs/attn_256_v1/checkpoints/latest_resume.pt",
    )
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bot-policy", type=str, default="batched", choices=["batched", "scalar"])
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "agent.scripts.train",
        "--run-id",
        args.run_id,
        "--init-from",
        args.init_from,
        "--device",
        args.device,
        "--max-iters",
        str(args.max_iters),
        "--eval-games",
        "0",
        "--checkpoint-every",
        "999999",
        "--replay-capacity",
        str(args.replay_capacity),
        "--profile-training",
        "--profile-sync-cuda",
        "--bot-policy",
        args.bot_policy,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    summarize = [
        sys.executable,
        "-m",
        "agent.scripts.summarize_profile",
        f"agent/runs/{args.run_id}",
    ]
    print("Summarizing...")
    subprocess.run(summarize, check=False)


if __name__ == "__main__":
    main()
