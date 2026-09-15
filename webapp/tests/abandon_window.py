"""Check the abandonment cutoff through the deployed UI and API.

Creates a unique _smoke_ account; the report lists its game IDs for cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import uuid

from playwright.sync_api import sync_playwright, expect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    username = '_smoke_' + uuid.uuid4().hex[:20]
    report = {'username': username, 'game_ids': [], 'cutoffs': [], 'page_errors': []}
    with sync_playwright() as p:
        options = {'headless': True}
        chrome = os.environ.get('CHROME_BIN', '/usr/bin/google-chrome')
        if Path(chrome).exists():
            options['executable_path'] = chrome
        browser = p.chromium.launch(**options)
        context = browser.new_context()
        context.add_init_script(f"localStorage.setItem('azul.username', {json.dumps(username)})")
        page = context.new_page()
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))

        try:
            page.goto(args.url)
            for pc in (2, 3, 4):
                expect(page.get_by_role('heading', name='Start a new game')).to_be_visible(timeout=30000)
                page.get_by_label('Players', exact=True).select_option(str(pc))
                for seat in range(2, pc + 1):
                    page.get_by_label(f'Player {seat}', exact=True).select_option('heuristic')
                page.get_by_role('button', name='Start game', exact=True).click()
                page.wait_for_url(re.compile(r'.*#game/.+'))
                game_id = page.url.split('#game/')[-1]
                report['game_ids'].append(game_id)
                abandon = page.get_by_role('button', name='Abandon game', exact=True)
                expect(abandon).to_be_visible()
                for _ in range(40):
                    page.wait_for_function("""() => [...document.querySelectorAll('[role="group"] button')]
                        .some(b => b.offsetParent !== null)""", timeout=30000)
                    current = context.request.get(f"{args.url.rstrip('/')}/api/game/{game_id}",
                        headers={'X-Azul-Username': username}).json()
                    if current['current_player'] != current['human_seat']:
                        continue
                    if not current['can_abandon']:
                        break
                    expect(abandon).to_be_visible()
                    groups = page.get_by_role('group', name=re.compile(r'^(Factory \d+|Center)$'))
                    groups.get_by_role('button').first.click()
                    lines = page.get_by_role('button', name=re.compile(r'^Place on pattern line'))
                    if lines.count():
                        lines.first.click()
                    else:
                        page.get_by_role('button', name='Place on floor', exact=True).click()
                else:
                    raise AssertionError('Abandonment window never closed')
                expect(abandon).to_have_count(0)
                assert current['status'] == 'active'
                page.reload()
                expect(page.locator('[data-game-status="active"]')).to_be_visible()
                expect(abandon).to_have_count(0)
                response = context.request.post(f"{args.url.rstrip('/')}/api/game/{game_id}/abandon",
                    headers={'X-Azul-Username': username}, data={'expected_revision': current['revision']})
                assert response.status == 400, response.text()
                assert 'four turns' in response.json()['error']
                report['cutoffs'].append({'players': pc, 'move_number': current['move_number'], 'api_status': response.status})
                page.screenshot(path=str(args.output.parent / f'abandon-{pc}p.png'), full_page=True)
                print(f'{pc}p: abandon hidden after cutoff and refresh; API rejected abandonment', flush=True)
                page.get_by_role('button', name='New game', exact=True).click()
            assert not report['page_errors'], report['page_errors']
        finally:
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            browser.close()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
