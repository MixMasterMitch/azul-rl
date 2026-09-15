"""Measure Astra through a local Lambda Runtime Interface Emulator.

Start the image with play.scripts.smoke_astra.local_handler. For a cold AI
request, prepare games, restart the container with the same local store mount,
and resume the saved state. No AWS endpoint or credentials are used.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from play.scripts.smoke import event


def run(url: str, players: list[int], prepare: bool,
        resume: Path | None = None) -> dict[str, Any]:
    endpoint = urlsplit(url)
    local = endpoint.hostname == 'localhost'
    if not local:
        try:
            local = ipaddress.ip_address(endpoint.hostname or '').is_loopback
        except ValueError:
            pass
    if not local or endpoint.scheme != 'http' or endpoint.path not in ('', '/') or endpoint.query or endpoint.fragment:
        raise ValueError('Use a local HTTP RIE base URL, such as http://127.0.0.1:9007')
    calls: list[dict[str, Any]] = []

    def call(username: str, method: str, path: str, body: dict | None = None,
             expected: int = 200) -> tuple[dict, float]:
        request = Request(url.rstrip('/') + '/2015-03-31/functions/function/invocations',
                          data=json.dumps(event(method, path, body, username)).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        started = time.perf_counter()
        with urlopen(request, timeout=35) as response:
            payload = json.load(response)
        elapsed = time.perf_counter() - started
        if payload.get('statusCode') != expected:
            raise RuntimeError(f'{method} {path}: expected {expected}, received {payload}')
        calls.append({'method': method, 'path': path, 'elapsed_s': elapsed})
        return json.loads(payload['body']), elapsed

    if resume:
        pending = json.loads(resume.read_text())['pending']
    else:
        pending = []
        for n in players:
            username = 'astra-rie-' + uuid.uuid4().hex[:12]
            catalog, _ = call(username, 'GET', f'/opponents?num_players={n}')
            assert any(bot['id'] == 'astra' for bot in catalog['opponents'])
            state, _ = call(username, 'POST', '/game', {
                'num_players': n, 'human_seat': n - 1, 'opponents': ['astra'] * (n - 1)}, 201)
            pending.append({'username': username, 'state': state, 'players': n})
    games = []
    if not prepare:
        for prepared in pending:
            state, username = prepared['state'], prepared['username']
            ai_times = []
            while state['current_player'] != state['human_seat'] and state['status'] != 'completed':
                state, elapsed = call(username, 'POST', f"/game/{state['game_id']}/step-ai",
                                      {'expected_revision': state['revision']})
                ai_times.append(elapsed)
                if len(ai_times) > 4:
                    raise RuntimeError('Opening AI chain did not reach the human seat')
            if not ai_times:
                raise ValueError('The prepared game is not waiting for an AI turn')
            games.append({'players': prepared['players'], 'ai_responses_s': ai_times,
                          'consecutive_ai_chain_s': sum(ai_times)})
    return {'prepared_for_restart': prepare, 'pending': pending if prepare else [],
            'first_invocation_s': calls[0]['elapsed_s'] if calls else None,
            'first_invocation_is_ai': bool(resume and not prepare),
            'games': games, 'calls': calls,
            'scope': 'Complete local HTTP-to-RIE-to-Flask responses and JSON persistence. A resumed first AI call includes application imports/initialization after a fresh container restart. This does not measure AWS cold starts, networking, or DynamoDB.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:9007')
    parser.add_argument('--players', type=int, nargs='+', choices=(2, 3, 4), default=[2, 3, 4])
    parser.add_argument('--prepare-for-restart', action='store_true')
    parser.add_argument('--resume-state', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.url, args.players, args.prepare_for_restart, args.resume_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'pending'}, indent=2))


if __name__ == '__main__':
    main()
