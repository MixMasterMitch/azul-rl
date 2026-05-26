"""League management: persistent pool of checkpoints with ratings."""

from __future__ import annotations

import json
import os
import pathlib
import random
from typing import Optional

import torch

from ..net import model as M
from . import checkpointing as CK
from .ranking import DEFAULT_ANCHORS, fit_anchored_ratings


class League:
    """Manages a pool of training checkpoints with Bradley-Terry ratings."""

    def __init__(
        self,
        root: str | pathlib.Path,
        max_entries: int = 24,
        keep_recent: int = 8,
        anchors: Optional[dict[str, float]] = None,
    ):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "league.json"
        self.max_entries = max_entries
        self.keep_recent = keep_recent
        self._net_cache: dict[str, M.AzulNet] = {}

        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = {}
        self.manifest.setdefault("entries", [])
        self.manifest.setdefault("results", [])
        self.manifest.setdefault("anchors", dict(anchors or DEFAULT_ANCHORS))

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def _resolve_path(self, stored: str) -> pathlib.Path:
        p = pathlib.Path(stored)
        if p.is_absolute():
            return p
        return self.root / p

    def _entry_available(self, entry: dict) -> bool:
        path_str = entry.get("path", "")
        return bool(path_str) and self._resolve_path(path_str).exists()

    def list_entries(self) -> list[dict]:
        return list(self.manifest.get("entries", []))

    def latest_entry(self) -> Optional[dict]:
        entries = self.list_entries()
        return entries[-1] if entries else None

    def add_checkpoint(
        self,
        net: M.AzulNet,
        tag: str = "",
        iteration: int = 0,
    ) -> dict:
        """Save net weights into the league and append manifest entry."""
        idx = len(self.manifest["entries"])
        rel_name = f"ckpt_{idx:05d}_{tag}.pt" if tag else f"ckpt_{idx:05d}.pt"
        dest = self.root / rel_name
        CK.save_checkpoint(dest, net, iteration=iteration)
        entry = {
            "idx": idx,
            "tag": tag,
            "path": rel_name,
            "iteration": iteration,
            "games": 0,
        }
        self.manifest["entries"].append(entry)
        if self.max_entries and len(self.manifest["entries"]) > self.max_entries:
            self._prune()
        self._save_manifest()
        return entry

    def record_result(
        self,
        winner: str,
        loser: str,
        wins_w: float,
        wins_l: float,
        ties: float = 0.0,
        num_players: int = 2,
    ) -> None:
        key_w = f"wins_{winner}"
        for r in self.manifest["results"]:
            if r.get("a") == winner and r.get("b") == loser:
                r["wins_a"] = r.get("wins_a", 0) + wins_w
                r["wins_b"] = r.get("wins_b", 0) + wins_l
                r["ties"] = r.get("ties", 0) + ties
                r["num_players"] = num_players
                self._save_manifest()
                return
            if r.get("a") == loser and r.get("b") == winner:
                r["wins_a"] = r.get("wins_a", 0) + wins_l
                r["wins_b"] = r.get("wins_b", 0) + wins_w
                r["ties"] = r.get("ties", 0) + ties
                r["num_players"] = num_players
                self._save_manifest()
                return
        self.manifest["results"].append({
            "a": winner,
            "b": loser,
            "wins_a": wins_w,
            "wins_b": wins_l,
            "ties": ties,
            "num_players": num_players,
        })
        self._save_manifest()

    def recompute_ratings(self) -> dict[str, float]:
        if not self.manifest["results"]:
            return {}
        ratings = fit_anchored_ratings(
            self.manifest["results"],
            anchors=dict(self.manifest.get("anchors", DEFAULT_ANCHORS)),
        )
        self.manifest["ratings"] = ratings
        self._save_manifest()
        return ratings

    def sample_opponent_path(self, rng: Optional[random.Random] = None) -> Optional[str]:
        entries = [e for e in self.list_entries() if self._entry_available(e)]
        if not entries:
            return None
        rng = rng or random.Random()
        weights = []
        ratings = self.manifest.get("ratings", {})
        for i, entry in enumerate(entries):
            recency = 2.0 ** (i - len(entries) + 1)
            entity = f"ckpt:{entry['idx']}"
            rating = ratings.get(entity, ratings.get(entry.get("id", ""), 1500.0))
            weights.append(recency * max(float(rating) / 1000.0, 0.5))
        chosen = rng.choices(entries, weights=weights, k=1)[0]
        return str(self._resolve_path(chosen["path"]))

    def load_net(self, path: str, device: str = "cpu") -> M.AzulNet:
        if path in self._net_cache:
            return self._net_cache[path]
        net, _ = CK.load_net_from_checkpoint(path, map_location=device)
        net.eval()
        self._net_cache[path] = net
        return net

    def get_entry_path(self, entry_id: str) -> Optional[pathlib.Path]:
        for entry in self.list_entries():
            if entry.get("id") == entry_id or f"ckpt:{entry.get('idx')}" == entry_id:
                return self._resolve_path(entry["path"])
        return None

    def _prune(self) -> None:
        entries = self.manifest["entries"]
        if len(entries) <= self.max_entries:
            return
        recent = entries[-self.keep_recent :]
        older = entries[: -self.keep_recent]
        ratings = self.manifest.get("ratings", {})
        older.sort(
            key=lambda e: ratings.get(f"ckpt:{e['idx']}", 0),
            reverse=True,
        )
        keep_older = max(0, self.max_entries - len(recent))
        kept = older[:keep_older]
        removed = older[keep_older:] + entries[: max(0, len(entries) - self.max_entries)]
        for entry in removed:
            if entry not in kept and entry not in recent:
                p = self._resolve_path(entry["path"])
                if p.exists():
                    p.unlink(missing_ok=True)
        self.manifest["entries"] = kept + recent
