"""Build, freeze, validate, and publish Azul releases without changing Splendor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent


def command(args: list[str], cwd: Path = ROOT) -> str:
    print('+ ' + shlex.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)
    return ''


def stack_outputs(client, name: str) -> dict:
    from botocore.exceptions import ClientError
    try:
        stack = client.describe_stacks(StackName=name)['Stacks'][0]
    except ClientError as exc:
        if 'does not exist' in str(exc):
            return {}
        raise
    return {row['OutputKey']: row['OutputValue'] for row in stack.get('Outputs', [])}


def freeze_source(root: Path) -> tuple[str, Path]:
    releases = root / '.releases'
    releases.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=releases) as temporary:
        stage = Path(temporary) / 'source'
        stage.mkdir()
        ignored = shutil.ignore_patterns('runs', 'node_modules', 'target', '__pycache__', '*.pyc',
                                         '.pytest_cache', 'play_data', '*.so', 'cdk.out')
        for name in ('agent', 'play', 'infra', 'native', 'webapp'):
            shutil.copytree(root / name, stage / name, ignore=ignored)
        for name in ('pyproject.toml', 'cdk.json', '.dockerignore', 'README.md', 'AGENTS.md'):
            shutil.copy2(root / name, stage / name)
        digest = hashlib.sha256()
        for path in sorted(p for p in stage.rglob('*') if p.is_file()):
            digest.update(str(path.relative_to(stage)).encode() + b'\0')
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
        release_id = digest.hexdigest()[:20]
        target = releases / release_id / 'source'
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(stage), target)
    print(f'Frozen source release: {release_id}', flush=True)
    return release_id, target


def extract_archive(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as bundle:
        bundle.extractall(target, filter='data')


def restore_previous_models(s3, previous: dict) -> None:
    release = previous.get('DeploymentRelease')
    if not release or not previous.get('ReleaseBucketName'):
        return
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        archive = path / 'source.tar.gz'
        s3.download_file(previous['ReleaseBucketName'], f'releases/{release}/source.tar.gz', str(archive))
        extract_archive(archive, path)
        old = path / 'source/play/artifacts'
        current = ROOT / 'play/artifacts'
        current.mkdir(parents=True, exist_ok=True)
        old_registry = json.loads((old / 'registry.json').read_text())
        manifest = current / 'registry.json'
        active = json.loads(manifest.read_text()) if manifest.exists() else old_registry
        for model_id, entry in old_registry['models'].items():
            if model_id in active['models'] and active['models'][model_id]['sha256'] != entry['sha256']:
                raise ValueError(f'Local model ID conflicts with deployed artifact: {model_id}')
            active['models'].setdefault(model_id, entry)
            shutil.copy2(old / entry['checkpoint'], current / entry['checkpoint'])
        manifest.write_text(json.dumps(active, indent=2) + '\n')


def check_active_models(session, previous: dict, source: Path) -> None:
    table = previous.get('GamesTableName')
    if not table:
        return
    models = json.loads((source / 'play/artifacts/registry.json').read_text())['models']
    db = session.resource('dynamodb').Table(table)
    args: dict = {'ProjectionExpression': '#data', 'ExpressionAttributeNames': {'#data': 'data'}, 'ConsistentRead': True}
    while True:
        result = db.scan(**args)
        for item in result.get('Items', []):
            game = json.loads(item['data'])
            if game['status'] != 'active':
                continue
            for info in game['opponent_info']:
                if 'sha256' in info and (info['id'] not in models or models[info['id']]['sha256'] != info['sha256']):
                    raise ValueError(f"Release would strand an active game using {info['id']}")
        if not result.get('LastEvaluatedKey'):
            break
        args['ExclusiveStartKey'] = result['LastEvaluatedKey']


def cdk(source: Path, action: str, args, account: str, release: str,
        live_version: str | None = None, live_frontend: str | None = None) -> None:
    cmd = ['cdk', action, '--app', shlex.join([sys.executable, '-m', 'infra.app']),
           '-c', f'account={account}', '-c', f'region={args.region}', '-c', f'stack_name={args.stack_name}',
           '-c', f'release_id={release}']
    if live_version and live_version != 'pending':
        cmd.extend(['-c', f'live_version={live_version}'])
    if live_frontend and live_frontend != 'pending':
        cmd.extend(['-c', f'live_frontend_release={live_frontend}'])
    if action == 'deploy':
        cmd.extend(['--require-approval', 'never'])
    elif action == 'diff':
        cmd.append('--change-set=false')
    elif action == 'synth':
        cmd.append('--quiet')
    command(cmd, source)


def archive_source(s3, source: Path, bucket: str, release: str) -> None:
    target = source.parent / 'source.tar.gz'
    with tarfile.open(target, 'w:gz') as bundle:
        def include(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            return None if any(part in {'cdk.out', '__pycache__', '.pytest_cache'} for part in Path(info.name).parts) else info
        bundle.add(source, arcname='source', filter=include)
    s3.upload_file(str(target), bucket, f'releases/{release}/source.tar.gz')


def clean_smoke(session, outputs: dict, report: dict) -> None:
    username = report['username']
    if not username.startswith('_smoke_'):
        raise ValueError('Refusing to clean a non-smoke user')
    games = session.resource('dynamodb').Table(outputs['GamesTableName'])
    for game_id in report['game_ids']:
        games.delete_item(Key={'game_id': game_id}, ConditionExpression='user_sub = :user',
                          ExpressionAttributeValues={':user': username})
    session.resource('dynamodb').Table(outputs['UsersTableName']).delete_item(Key={'username': username})


def smoke(source: Path, args, outputs: dict, *, public: bool = False) -> dict:
    target = source.parent / ('public-smoke.json' if public else 'candidate-smoke.json')
    cmd = [sys.executable, '-m', 'play.scripts.smoke', '--output', str(target), '--region', args.region]
    if public:
        cmd.extend(['--url', outputs['CloudFrontUrl'], '--quick'])
    else:
        cmd.extend(['--function', outputs['FunctionName'], '--version', outputs['CandidateVersion']])
    command(cmd, source)
    return json.loads(target.read_text())


def refresh_frontend(session, outputs: dict) -> None:
    """Origin-path changes do not themselves evict CloudFront's cached objects."""
    cloudfront = session.client('cloudfront')
    response = cloudfront.create_invalidation(
        DistributionId=outputs['DistributionId'],
        InvalidationBatch={'CallerReference': str(time.time_ns()),
                           'Paths': {'Quantity': 1, 'Items': ['/*']}},
    )
    cloudfront.get_waiter('invalidation_completed').wait(
        DistributionId=outputs['DistributionId'], Id=response['Invalidation']['Id'],
        WaiterConfig={'Delay': 10, 'MaxAttempts': 60},
    )


def check_frontend(source: Path, url: str) -> None:
    """Verify the actual HTML and every referenced built asset at CloudFront."""
    expected = (source / 'webapp/dist/index.html').read_bytes()
    with urlopen(url + '/', timeout=30) as response:
        actual = response.read()
    if actual != expected:
        raise ValueError('CloudFront is serving a different frontend release')
    assets = re.findall(r'''(?:src|href)=["'](/assets/[^"']+)["']''', actual.decode())
    if not assets:
        raise ValueError('Frontend HTML contains no built assets')
    for asset in assets:
        with urlopen(url + asset, timeout=30) as response:
            if response.read() != (source / 'webapp/dist' / asset.lstrip('/')).read_bytes():
                raise ValueError(f'Frontend asset differs from the release: {asset}')


def rollback(session, cloudformation, outputs: dict, args) -> None:
    s3 = session.client('s3')
    bucket = outputs['ReleaseBucketName']
    metadata = json.loads(s3.get_object(Bucket=bucket, Key=f'releases/{args.rollback}/release.json')['Body'].read())
    current = outputs['DeploymentRelease']
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)
        for release in {current, args.rollback}:
            directory = target / release
            directory.mkdir()
            archive = directory / 'source.tar.gz'
            s3.download_file(bucket, f'releases/{release}/source.tar.gz', str(archive))
            extract_archive(archive, directory)
        check_active_models(session, outputs, target / args.rollback / 'source')
        cdk(target / current / 'source', 'deploy', args, session.client('sts').get_caller_identity()['Account'],
            current, metadata['version'], args.rollback)
        refresh_frontend(session, outputs)
        check_frontend(target / args.rollback / 'source', outputs['CloudFrontUrl'])
    print(f"Rolled back to {args.rollback}: {outputs['CloudFrontUrl']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Run local checks and synth only; never publish or deploy')
    parser.add_argument('--skip-frontend', action='store_true')
    parser.add_argument('--region', default='us-west-2')
    parser.add_argument('--stack-name', default='AzulStack')
    parser.add_argument('--account')
    parser.add_argument('--rollback', metavar='RELEASE_ID')
    args = parser.parse_args()
    if args.rollback and args.dry_run:
        parser.error('--rollback cannot be combined with --dry-run')
    session, previous, account = None, {}, args.account or '000000000000'
    if not args.dry_run:
        import boto3
        session = boto3.Session(region_name=args.region)
        account = session.client('sts').get_caller_identity()['Account']
        if args.account and args.account != account:
            raise ValueError('Authenticated AWS account differs from --account')
        cloudformation = session.client('cloudformation')
        previous = stack_outputs(cloudformation, args.stack_name)
        print(f'Target: {account}/{args.region}/{args.stack_name}', flush=True)
        if args.rollback:
            rollback(session, cloudformation, previous, args)
            return
        restore_previous_models(session.client('s3'), previous)
    command([sys.executable, '-m', 'play.scripts.prepare_release'])
    if not args.skip_frontend:
        command(['npm', 'ci'], ROOT / 'webapp')
        command(['npm', 'run', 'build'], ROOT / 'webapp')
    elif not (ROOT / 'webapp/dist/index.html').exists():
        raise ValueError('--skip-frontend requires an existing frontend build')
    release, source = freeze_source(ROOT)
    command(['env', 'CUDA_VISIBLE_DEVICES=', sys.executable, '-m', 'pytest', 'play/tests', 'infra/tests',
             'agent/tests/test_checkpoint_compat.py', 'agent/tests/test_batched_step_parity.py',
             'agent/tests/test_search_correctness.py', '-q'], source)
    image = f'azul-release:{release}'
    command(['docker', 'build', '--platform', 'linux/amd64', '-f', 'infra/lambda.Dockerfile', '-t', image, '.'], source)
    command(['docker', 'run', '--rm', '--read-only', '--tmpfs', '/tmp:rw,size=512m',
             '--entrypoint', 'python', image, '-m', 'play.scripts.smoke'], source)
    cdk(source, 'synth', args, account, release, previous.get('LiveVersion'), previous.get('FrontendRelease'))
    if args.dry_run:
        print(f'Dry run passed. No AWS resources changed. Frozen release: {source}')
        return
    check_active_models(session, previous, source)
    if not stack_outputs(cloudformation, 'CDKToolkit'):
        command(['cdk', 'bootstrap', f'aws://{account}/{args.region}'], source)
    cdk(source, 'diff', args, account, release, previous.get('LiveVersion'), previous.get('FrontendRelease'))
    cdk(source, 'deploy', args, account, release, previous.get('LiveVersion'), previous.get('FrontendRelease'))
    candidate = stack_outputs(cloudformation, args.stack_name)
    archive_source(session.client('s3'), source, candidate['ReleaseBucketName'], release)
    report = smoke(source, args, candidate)
    clean_smoke(session, candidate, report)
    # Publish the API alias and matching frontend only after the candidate passes.
    cdk(source, 'deploy', args, account, release, candidate['CandidateVersion'], release)
    live = stack_outputs(cloudformation, args.stack_name)
    try:
        refresh_frontend(session, live)
        check_frontend(source, live['CloudFrontUrl'])
        public_report = smoke(source, args, live, public=True)
        clean_smoke(session, live, public_report)
    except Exception:
        if previous.get('LiveVersion') and previous['LiveVersion'] != 'pending':
            cdk(source, 'deploy', args, account, release, previous['LiveVersion'], previous['FrontendRelease'])
        else:
            # On a first release, withdraw public routes if the final check fails.
            cdk(source, 'deploy', args, account, release)
        refresh_frontend(session, live)
        raise
    metadata = {'release_id': release, 'version': candidate['CandidateVersion'], 'account': account,
                'region': args.region, 'stack_name': args.stack_name, 'url': live['CloudFrontUrl'], 'validation': report}
    (source.parent / 'release.json').write_text(json.dumps(metadata, indent=2) + '\n')
    session.client('s3').put_object(Bucket=live['ReleaseBucketName'], Key=f'releases/{release}/release.json',
                                  Body=json.dumps(metadata, indent=2).encode(), ContentType='application/json')
    print(f"Azul is live: {live['CloudFrontUrl']} (release {release})", flush=True)


if __name__ == '__main__':
    main()
