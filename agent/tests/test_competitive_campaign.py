from __future__ import annotations

from dataclasses import replace
import json
import pytest

from agent.net.model import AzulNet
from agent.scripts.competitive import Campaign, training_config, tree_distill_config
from agent.train.checkpointing import save_checkpoint, load_checkpoint_payload


def test_campaign_finishes_retires_replay_and_checks_cached_configuration(tmp_path) -> None:
    source = tmp_path / 'initializer.pt'
    save_checkpoint(source, AzulNet(hidden=32, arch='flat'), config={'num_players': 2})
    campaign = Campaign(tmp_path, 'cpu', 202)
    cfg = replace(training_config(tmp_path, 'tiny', str(source), device='cpu', minutes=.003),
                  hidden=32, arch='flat', selfplay_games=4, selfplay_sims=2,
                  replay_capacity=1000, learner_steps_per_iter=2, learner_batch=16,
                  training_cycle_length=0, eval_games=0, checkpoint_every=1)
    finalist = campaign._train(cfg)
    assert load_checkpoint_payload(finalist)['arch'] == 'flat'
    directory = tmp_path / 'experiments/tiny'
    assert not (directory / 'checkpoints/latest_resume.pt').exists()
    assert json.loads((directory/'state.json').read_text())['experiment_complete']
    assert campaign._train(cfg) == finalist
    with pytest.raises(ValueError, match='configuration differs'):
        campaign._train(replace(cfg, learner_steps_per_iter=3))


def test_watchdog_allows_boundary_save_then_enforces_ceiling(monkeypatch) -> None:
    import signal
    from agent.scripts import competitive
    calls = []
    class Finished:
        def wait(self, seconds):
            calls.append(seconds)
            return False
    monkeypatch.setattr(competitive.os, 'kill', lambda pid, sig: calls.append(sig))
    competitive._budget_watchdog(Finished(), seconds=480*60)
    assert calls == [478*60, signal.SIGTERM, 120, signal.SIGKILL]


def test_tree_distillation_config_uses_the_confirmed_teacher_without_midrun_evaluation(tmp_path) -> None:
    cfg = tree_distill_config(tmp_path, 'tree', 'teacher.pt', seed=88, minutes=450, device='cuda')
    assert cfg.arch == 'source_attn' and cfg.init_from == 'teacher.pt'
    assert cfg.search_backend == 'gumbel_tree' and cfg.eval_search_backend == 'gumbel_tree'
    assert cfg.selfplay_sims == 64 and cfg.selfplay_games == 256
    assert cfg.learner_steps_per_iter == 72 and cfg.replay_capacity == 1_000_000
    assert cfg.reward_mode == 'binary' and cfg.eval_games == 0
    assert cfg.checkpoint_every == 25 and cfg.max_wall_minutes == 450
    assert cfg.bot_selfplay_astra_prob == .0625 and cfg.bot_selfplay_workers == 8


def test_short_learner_comparison_uses_matching_initializers_and_evaluation(tmp_path, monkeypatch) -> None:
    campaign = Campaign(tmp_path, 'cpu', 123, bot_workers=8)
    (tmp_path/'baseline.json').write_text(json.dumps({'champion': 'champion.pt', 'historical': 'old.pt'}))
    (tmp_path/'policy.json').write_text(json.dumps({'winner': 'source.pt', 'winner_arch': 'source_attn'}))
    configs, matches = [], []
    def train(cfg):
        configs.append(cfg)
        return f'{cfg.learner_steps_per_iter}.pt'
    def screen(label, checkpoint, baseline):
        return {'checkpoint': checkpoint, 'scores': {'champion': {'match_score': .52 if checkpoint == '72.pt' else .56}}}
    def match(label, candidate, opponent, cfg):
        matches.append((candidate, opponent, cfg))
        return {'summary': {'match_score': .5}}
    monkeypatch.setattr(campaign, '_train', train)
    monkeypatch.setattr(campaign, '_screen', screen)
    monkeypatch.setattr(campaign, '_match', match)
    monkeypatch.setattr(campaign, 'confirm', lambda *args: {'promote': False})
    result = campaign.learner(minutes=50, updates=(72, 144))
    assert [c.learner_steps_per_iter for c in configs] == [72, 144]
    assert all(c.init_from == 'source.pt' and c.seed == 123 and c.max_wall_minutes == 50 for c in configs)
    assert configs[0].league_root != configs[1].league_root
    assert len(matches) == 2 and all(opponent == 'source.pt' and cfg.bot_workers == 8 for _, opponent, cfg in matches)
    assert result['winner_steps'] == 144 and result['update_counts'] == [72, 144]
    assert not result['replicated']
    assert not (tmp_path/'models/registry.json').exists()


@pytest.mark.parametrize('updates', [(), (72, 72), (0,), (True,), (1.5,)])
def test_learner_rejects_invalid_update_counts(tmp_path, updates) -> None:
    with pytest.raises(ValueError, match='distinct positive integers'):
        Campaign(tmp_path, 'cpu', 1).learner(updates=updates)
