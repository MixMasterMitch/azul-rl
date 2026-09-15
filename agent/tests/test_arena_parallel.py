from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from agent.eval.arena import ArenaConfig, evaluate_match
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.train.checkpointing import save_checkpoint


def test_parallel_bot_actions_preserve_seeded_game_records(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    path = tmp_path/'net.pt'
    save_checkpoint(path, AzulNet(hidden=32, arch='flat'), config={'num_players': 2})
    cfg = ArenaConfig(num_games=4, game_batch_size=4, seed=831,
                      search=SearchConfig(num_simulations=2))
    serial = evaluate_match(str(path), 'random', cfg)
    parallel = evaluate_match(str(path), 'random', replace(cfg, bot_workers=4))
    assert serial['records'] == parallel['records']
    assert serial['summary']['unfinished'] == 0
    assert serial['value_calibration'] == parallel['value_calibration']


@pytest.mark.parametrize('workers', [0, 9, 1.5, True])
def test_reject_invalid_worker_count(workers: object) -> None:
    with pytest.raises(ValueError, match='bot_workers'):
        ArenaConfig(bot_workers=workers)
