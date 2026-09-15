"""Browser acceptance: real UI games, refresh/resume, history, and mobile layout.

Run against an isolated local server, or the deployed CloudFront URL:
    python webapp/tests/hosted_play.py --url http://127.0.0.1:5177
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
    parser.add_argument('--url', default='http://127.0.0.1:5177')
    parser.add_argument('--screenshots', type=Path, default=Path('.hosting-checks/browser'))
    parser.add_argument('--opponent', help='Use one opponent ID at every seat/player count (for example astra)')
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    username = '_browser_' + uuid.uuid4().hex[:16]
    with sync_playwright() as p:
        options = {'headless': True}
        chrome = os.environ.get('CHROME_BIN', '/usr/bin/google-chrome')
        if Path(chrome).exists():
            options['executable_path'] = chrome
        browser = p.chromium.launch(**options)
        context = browser.new_context(viewport={'width':1280,'height':1000})
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.screenshot(path=str(args.screenshots / 'welcome.png'), full_page=True)
        page.get_by_label('Your screen name').fill(username)
        page.get_by_role('button', name='Let’s play').click()
        expect(page.get_by_role('heading', name='Start a new game')).to_be_visible()
        game_ids = []
        for pc in [2, 3, 4]:
            if pc != 2:
                page.get_by_role('button', name='New game', exact=True).click()
            page.get_by_label('Players', exact=True).select_option(str(pc))
            if args.opponent:
                for seat in range(2, pc + 1):
                    page.get_by_label(f'Player {seat}', exact=True).select_option(args.opponent)
            elif pc == 2:
                page.get_by_label('Player 2', exact=True).select_option(label='RL Trained Bot')
            else:
                for seat in range(2, pc + 1):
                    select = page.get_by_label(f'Player {seat}', exact=True)
                    assert 'RL Trained Bot' not in select.inner_text()
                    select.select_option('heuristic')
            page.get_by_role('button', name='Start game', exact=True).click()
            page.wait_for_url(re.compile(r'.*#game/.+'))
            game_ids.append(page.url.split('#game/')[-1])
            for move in range(400):
                page.wait_for_function("""() => document.querySelector('[data-game-status="completed"]') ||
                    [...document.querySelectorAll('[role="group"] button')].some(b => b.offsetParent !== null)""", timeout=20000)
                if page.locator('[data-game-status="completed"]').count():
                    break
                groups = page.get_by_role('group', name=re.compile(r'^(Factory \d+|Center)$'))
                groups.get_by_role('button').first.click()
                lines = page.get_by_role('button', name=re.compile(r'^Place on pattern line'))
                if lines.count():
                    lines.first.click()
                else:
                    page.get_by_role('button', name='Place on floor', exact=True).click()
                if move == 3:
                    page.reload()
                    expect(page.locator('[data-game-status]')).to_be_visible()
                if move == 0:
                    page.screenshot(path=str(args.screenshots / f'{pc}p-board.png'), full_page=True)
            else:
                raise AssertionError(f'{pc}p game did not finish through the UI')
            assert not page.get_by_role('alert').count(), page.get_by_role('alert').all_text_contents()
            page.screenshot(path=str(args.screenshots / f'{pc}p-result.png'), full_page=True)
            print(f'Completed {pc}p game through browser', flush=True)
        page.get_by_role('button', name='My games', exact=True).click()
        expect(page.locator('.history-row')).to_have_count(3)
        page.get_by_role('button', name='View result').first.click()
        expect(page.locator('[data-game-status="completed"]')).to_be_visible()
        page.get_by_role('button', name='Leaderboard', exact=True).click()
        expect(page.get_by_role('cell', name='RL Trained Bot', exact=False)).to_be_visible()
        page.screenshot(path=str(args.screenshots / 'leaderboard.png'), full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.get_by_role('button', name='Play', exact=True).click()
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile page overflows'
        page.screenshot(path=str(args.screenshots / 'mobile.png'), full_page=True)
        assert not errors, errors
        report = {'username': username, 'game_ids': game_ids, 'completed_games':3, 'page_errors':errors,
                  'opponent': args.opponent or 'default acceptance field'}
        (args.screenshots / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        browser.close()


if __name__ == '__main__':
    main()
