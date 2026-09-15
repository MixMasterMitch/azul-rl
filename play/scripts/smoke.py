"""End-to-end game and latency gates for local containers, Lambda versions, and HTTP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def event(method: str, path: str, body: dict | None, username: str) -> dict:
    route, _, query = path.partition('?')
    return {'version': '2.0', 'rawPath': '/api' + route, 'rawQueryString': query,
            'headers': {'content-type': 'application/json', 'x-azul-username': username},
            'body': json.dumps(body) if body is not None else '',
            'requestContext': {'http': {'method': method, 'path': '/api' + route, 'protocol': 'HTTP/1.1'}}}


def run(*, function: str | None = None, version: str | None = None, url: str | None = None,
        region: str = 'us-west-2', quick: bool = False) -> dict:
    username = '_smoke_' + uuid.uuid4().hex[:20]
    registry_path = Path(os.environ.get('AZUL_MODEL_REGISTRY', str(Path(__file__).resolve().parents[1] / 'artifacts/registry.json')))
    model_id = json.loads(registry_path.read_text())['default_model_id']
    if function:
        import boto3
        from botocore.config import Config
        client = boto3.client('lambda', region_name=region, config=Config(read_timeout=40, retries={'max_attempts': 0}))
    elif not url:
        from play.server import create_app
        from play.store import JsonPlayStore
        from play.lambda_handler import handle_wsgi
        temporary = tempfile.TemporaryDirectory()
        app = create_app(JsonPlayStore(temporary.name))
    ids: list[str] = []
    cold: list[float] = []
    warm_ai: list[float] = []
    init_durations: list[float] = []

    def call(method: str, path: str, body: dict | None = None, expected: int = 200) -> tuple[dict, float]:
        started = time.perf_counter()
        if function:
            import base64
            response = client.invoke(FunctionName=function, Qualifier=version or '$LATEST', LogType='Tail',
                                     Payload=json.dumps(event(method, path, body, username)).encode())
            payload = json.loads(response['Payload'].read())
            logs = base64.b64decode(response.get('LogResult', '')).decode(errors='replace')
            match = re.search(r'Init Duration: ([\d.]+) ms', logs)
            if match:
                init_durations.append(float(match[1]))
            if response.get('FunctionError'):
                raise RuntimeError(f'Lambda failed: {payload}')
            status, data = payload['statusCode'], json.loads(payload['body'])
        elif url:
            req = Request(url.rstrip('/') + '/api' + path, method=method,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={'Content-Type': 'application/json', 'X-Azul-Username': username})
            try:
                with urlopen(req, timeout=35) as response:
                    status, data = response.status, json.load(response)
            except HTTPError as exc:
                status, data = exc.code, json.load(exc)
        else:
            payload = handle_wsgi(event(method, path, body, username), app)
            status, data = payload['statusCode'], json.loads(payload['body'])
        elapsed = time.perf_counter() - started
        if status != expected:
            raise RuntimeError(f'{method} {path}: expected {expected}, got {status}: {data}')
        return data, elapsed

    def create(_: int) -> dict:
        state, elapsed = call('POST', '/game', {'num_players': 2, 'human_seat': 1, 'opponents': [model_id]}, 201)
        ids.append(state['game_id'])
        cold.append(elapsed)
        return state

    try:
        # Concurrent first requests to a newly published version exercise cold
        # environments; AWS REPORT logs record how many were actually cold.
        with ThreadPoolExecutor(max_workers=4 if function else 1) as pool:
            games = list(pool.map(create, range(4 if function else 1)))
        call('GET', '/ready')
        catalog, _ = call('GET', '/opponents?num_players=3')
        assert model_id not in {opponent['id'] for opponent in catalog['opponents']}
        call('POST', '/game', {'num_players': 3, 'opponents': [model_id, 'random']}, 400)
        if any(opponent['id'] == 'astra' for opponent in catalog['opponents']):
            astra, _ = call('POST', '/game', {'num_players': 2, 'human_seat': 1, 'opponents': ['astra']}, 201)
            ids.append(astra['game_id'])
            astra, _ = call('POST', f"/game/{astra['game_id']}/step-ai", {'expected_revision': 0})
            assert astra['revision'] == 1 and astra['current_player'] == 1
        call('GET', '/me')
        rng = random.Random(23)
        finished = 0
        for pc in ([2] if quick else [2, 3, 4]):
            if pc == 2:
                state = games[0]
            else:
                state, _ = call('POST', '/game', {'num_players': pc, 'opponents': ['heuristic'] * (pc - 1)}, 201)
                ids.append(state['game_id'])
            for move in range(400):
                if state['status'] == 'completed':
                    break
                is_ai = state['current_player'] != state['human_seat']
                suffix = 'step-ai' if is_ai else 'action'
                request_body = {'expected_revision': state['revision']}
                if not is_ai:
                    request_body['action'] = rng.choice(state['legal_actions'])['index']
                path = f"/game/{state['game_id']}/{suffix}"
                state, elapsed = call('POST', path, request_body)
                if is_ai:
                    warm_ai.append(elapsed)
                if move == 0:
                    call('POST', path, request_body, 409)
                if move % 10 == 0:
                    restored, _ = call('GET', f"/game/{state['game_id']}")
                    assert restored == state
                if quick and move >= 3:
                    break
            if not quick:
                assert state['status'] == 'completed', f'{pc}p game stalled'
                finished += 1
                before, _ = call('GET', '/me')
                call('POST', f"/game/{state['game_id']}/step-ai", {'expected_revision': state['revision']})
                after, _ = call('GET', '/me')
                assert before == after and after['games'] == finished
        call('GET', '/games?limit=2')
        call('GET', '/leaderboard')
        ordered = sorted(warm_ai)
        p95 = ordered[max(0, math.ceil(len(ordered) * .95) - 1)]
        report = {'username': username, 'game_ids': ids, 'completed_games': finished,
                  'cold_request_seconds': cold, 'aws_init_ms': init_durations,
                  'warm_ai_requests': len(warm_ai), 'warm_ai_p95_seconds': p95,
                  'warm_ai_max_seconds': max(warm_ai)}
        if p95 >= 2 or max(cold) >= 25:
            raise RuntimeError(f'Latency gate failed: {json.dumps(report)}')
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        # Leave no in-flight smoke games. Completed results use a unique screen
        # name; the deployment driver removes only this run's test records.
        for game_id in ids:
            try:
                state, _ = call('GET', f'/game/{game_id}')
                if state['status'] == 'active':
                    call('POST', f'/game/{game_id}/abandon', {'expected_revision': state['revision']})
            except Exception:
                pass
        if not function and not url:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--function')
    parser.add_argument('--version')
    parser.add_argument('--url')
    parser.add_argument('--region', default='us-west-2')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = run(function=args.function, version=args.version, url=args.url, region=args.region, quick=args.quick)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
