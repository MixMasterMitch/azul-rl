from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from agent.scripts import supervise_training as supervisor


def test_health_ignores_stale_foreign_run_and_detects_stalls(tmp_path: Path) -> None:
    supervisor.write_json(tmp_path/'status.json', {
        'stage': 'training', 'updated_at': 10,
        'heartbeat': str(tmp_path.parent/'foreign/heartbeat.json'),
    })
    health = supervisor.health_snapshot(tmp_path, launched_at=20, now=50)
    assert health['activity_age_s'] == 30
    assert 'run_id' not in health
    assert supervisor.recovery_reason(health, stale_seconds=60) is None
    health['activity_age_s'] = 61
    assert supervisor.recovery_reason(health, stale_seconds=60) == 'stalled'
    health['disk_free_gib'] = .5
    assert supervisor.recovery_reason(health) == 'low_disk'


def test_health_records_checkpoint_and_sanitizes_nonfinite_loss(tmp_path: Path) -> None:
    run = tmp_path/'experiments/training'
    (run/'checkpoints').mkdir(parents=True)
    now = time.time()
    supervisor.write_json(tmp_path/'status.json', {
        'stage': 'training', 'updated_at': now-10, 'heartbeat': str(run/'heartbeat.json'),
    })
    supervisor.write_json(run/'heartbeat.json', {
        't': datetime.fromtimestamp(now, timezone.utc).isoformat(), 'iter': 101, 'phase': 'learner',
    })
    (run/'events.log').write_text('partial line\n'+json.dumps({
        'event': 'learner_done', 'fields': {'loss': float('nan'), 'grad_norm': 1.2, 'learner_steps_skipped': 1},
    })+'\n')
    (run/'checkpoints/latest_resume.pt').write_bytes(b'resume')
    health = supervisor.health_snapshot(tmp_path, launched_at=now-20, now=now+1)
    assert health['iteration'] == 101 and health['loss'] is None
    assert health['grad_norm'] == 1.2 and health['learner_steps_skipped'] == 1
    assert health['checkpoint_mib'] > 0
    supervisor.write_json(tmp_path/'health.json', health)


def test_supervisor_restarts_failed_child_and_stops_after_completion(tmp_path: Path, monkeypatch) -> None:
    # Exercise actual process launch/reap/restart with a tiny stand-in campaign.
    repo = tmp_path/'repo'
    scripts = repo/'agent/scripts'
    scripts.mkdir(parents=True)
    (repo/'agent/__init__.py').touch()
    (scripts/'__init__.py').touch()
    (scripts/'competitive.py').write_text('''
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[sys.argv.index('--root')+1])
attempt = root/'attempt'
n = int(attempt.read_text())+1 if attempt.exists() else 1
attempt.write_text(str(n))
(root/'status.json').write_text(json.dumps({'stage': 'failed' if n == 1 else 'policy_complete', 'updated_at': time.time()}))
sys.exit(1 if n == 1 else 0)
''')
    monkeypatch.setattr(supervisor, 'REPO', repo)
    # No disk cleanup or minute-long delays in the subprocess recovery test.
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda path: argparse.Namespace(free=10*1024**3))
    real_event = threading.Event
    class FastEvent(real_event):
        def wait(self, timeout=None):
            return super().wait(min(timeout or .01, .01))
    monkeypatch.setattr(supervisor.threading, 'Event', FastEvent)
    # Avoid modifying pytest's own signal handlers.
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *args: None)
    root = tmp_path/'campaign'
    args = argparse.Namespace(root=root, stage='policy', device='cpu', seed=1, hours=.1,
                              minutes_per_arm=None, interval=.02, steady_interval=.04, stable_checks=3,
                              stale_seconds=600, max_retries=4, learner_updates=[72, 144], bot_workers=8)
    assert supervisor.supervise(args) == 0
    state = supervisor.read_json(root/'supervisor.json')
    assert state['state'] == 'complete' and state['attempts'] == 2
    assert state['child_pid'] is None
    assert state['command'][-5:] == ['--learner-updates', '72', '144', '--bot-workers', '8']
    assert supervisor.supervise(args) == 0  # A service restart never reruns completed work.
    assert int((root/'attempt').read_text()) == 2


def test_expired_supervisor_never_launches_training(tmp_path: Path, monkeypatch) -> None:
    import sys
    command = [sys.executable, '-u', '-m', 'agent.scripts.competitive', 'policy',
               '--root', str(tmp_path), '--device', 'cpu', '--seed', '1']
    supervisor.write_json(tmp_path/'supervisor.json', {
        'command': command, 'started_at': 1, 'deadline': 2, 'attempts': 0,
        'failures_without_progress': 0,
    })
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('must not launch')))
    args = argparse.Namespace(root=tmp_path, stage='policy', device='cpu', seed=1, hours=8,
                              minutes_per_arm=None, interval=300, steady_interval=1800, stable_checks=3,
                              stale_seconds=600, max_retries=4)
    assert supervisor.supervise(args) == 0
    assert supervisor.read_json(tmp_path/'supervisor.json')['state'] == 'budget_exhausted'


def test_health_requires_cycle_progress_and_stable_gradients() -> None:
    health = {'stage': 'training', 'iteration': 20, 'learner_steps_ok': 72,
              'learner_steps_skipped': 0, 'loss': 1.1, 'grad_norm': 1.4,
              'disk_free_gib': 10, 'activity_age_s': 1}
    assert supervisor.healthy_progress(health, 16)
    assert not supervisor.healthy_progress(health, 19)
    assert not supervisor.healthy_progress({**health, 'grad_norm': 20000}, 16)
    assert not supervisor.healthy_progress({**health, 'learner_steps_skipped': 1}, 16)


def test_evaluation_health_tracks_completed_games(tmp_path: Path) -> None:
    now = time.time()
    supervisor.write_json(tmp_path/'status.json', {'stage': 'evaluation', 'updated_at': now,
        'completed_games_total': 256, 'unfinished': 0, 'search_profile': 'tree_64'})
    health = supervisor.health_snapshot(tmp_path, launched_at=now-20, now=now+1)
    assert supervisor.healthy_progress(health, 224)
    assert not supervisor.healthy_progress(health, 256)
    assert not supervisor.healthy_progress({**health, 'unfinished': 1}, 224)


def test_supervisor_reduces_check_frequency_after_stable_cycles(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path/'repo'
    scripts = repo/'agent/scripts'
    scripts.mkdir(parents=True)
    (repo/'agent/__init__.py').touch()
    (scripts/'__init__.py').touch()
    (scripts/'competitive.py').write_text('''
from datetime import datetime, timezone
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[sys.argv.index('--root')+1])
run = root/'experiments/training'
run.mkdir(parents=True)
def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)
write(root/'status.json', {'stage': 'training', 'heartbeat': str(run/'heartbeat.json'), 'updated_at': time.time()})
for i in range(90):
    if i in (0, 45):
        write(root/'status.json', {'stage': 'training', 'run_id': str(i//45), 'heartbeat': str(run/'heartbeat.json'), 'updated_at': time.time()})
    write(run/'heartbeat.json', {'t': datetime.now(timezone.utc).isoformat(), 'iter': i*4, 'phase': 'learner'})
    with (run/'events.log').open('a') as output:
        output.write(json.dumps({'event': 'learner_done', 'fields': {'loss': 1, 'grad_norm': 1, 'learner_steps_ok': 72, 'learner_steps_skipped': 0}})+'\\n')
    time.sleep(.01)
write(root/'status.json', {'stage': 'policy_complete', 'updated_at': time.time()})
''')
    monkeypatch.setattr(supervisor, 'REPO', repo)
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda path: argparse.Namespace(free=10*1024**3))
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *args: None)
    root = tmp_path/'campaign'
    args = argparse.Namespace(root=root, stage='policy', device='cpu', seed=1, hours=.1,
                              minutes_per_arm=None, interval=.12, steady_interval=.3, stable_checks=3,
                              stale_seconds=600, max_retries=4)
    assert supervisor.supervise(args) == 0
    events = [json.loads(line) for line in (root/'supervisor_events.jsonl').read_text().splitlines()]
    changed = [event for event in events if event['event'] == 'monitor_interval_changed']
    assert changed and changed[0]['interval_seconds'] == .3
    assert changed[0]['healthy_checks'] == 3
    arm_changes = [event for event in events if event['event'] == 'monitor_arm_changed']
    assert len(arm_changes) == 1 and arm_changes[0]['interval_seconds'] == .12
    following = [event for event in events if event['event'] == 'health_check' and event['time'] >= arm_changes[0]['time']]
    assert following[0]['interval_seconds'] == .12
