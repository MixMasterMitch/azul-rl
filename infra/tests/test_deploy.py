from __future__ import annotations

import boto3
from infra import deploy


def test_dry_run_never_uses_aws_or_publishes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(deploy, 'command', lambda args, cwd=deploy.ROOT: calls.append(args))
    monkeypatch.setattr(deploy, 'freeze_source', lambda root: ('test', tmp_path))
    monkeypatch.setattr(deploy.sys, 'argv', ['deploy', '--dry-run'])
    def no_aws(*args, **kwargs):
        raise AssertionError('Dry run must not create an AWS session')
    monkeypatch.setattr(boto3, 'Session', no_aws)
    deploy.main()
    assert any(call[:2] == ['cdk', 'synth'] for call in calls)
    assert not any(call[:2] in [['cdk','deploy'],['cdk','bootstrap']] for call in calls)
    assert any('--read-only' in call for call in calls)
