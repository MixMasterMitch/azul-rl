from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import torch

pytest.importorskip("azul_astra")

from agent.env.batched_engine import BatchedEngine
from agent.eval.astra_tournament import interval, play_game, schedule, summarize, winners_and_share, run_tournament
from agent.eval.heuristic_astra import AstraConfig


def test_schedule_balances_seats_and_isolates_splits() -> None:
    sets = []
    for split in ("development", "promotion", "final"):
        tasks = schedule([2, 3, 4], 3, split, 9, ["opus", "mixed"], "opus")
        assert len(tasks) == 3 * 9 * 2 * 2
        assert len({t["key"] for t in tasks}) == len(tasks)
        sets.append({t["engine_seed"] for t in tasks})
        for n in (2, 3, 4):
            group = [t for t in tasks if t["n"] == n and t["field"] == "mixed" and t["block"] == 0]
            assert {t["seat"] for t in group} == set(range(n))
            assert len({t["engine_seed"] for t in group}) == 1
            assert all(t["names"][t["seat"]] == t["variant"] for t in group)
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]


def test_official_shared_win_and_unfinished() -> None:
    e = BatchedEngine(1, 3, seed=9)
    assert winners_and_share(e) == ([], [0.0, 0.0, 0.0])
    e.ended[:] = True
    e.scores[0, :3] = torch.tensor([40, 40, 40])
    e.wall[0, :2, 0] = True
    assert winners_and_share(e) == ([0, 1], [0.5, 0.5, 0.0])
    e.wall[0, 1, 1] = True
    assert winners_and_share(e) == ([1], [0.0, 1.0, 0.0])


def fake_game(block: int, seat: int, share: float | None, variant: str = "astra") -> dict:
    return {"n": 2, "field": "mixed", "block": block, "seat": seat,
            "variant": variant, "finished": share is not None, "failure": None,
            "win_share": share, "move_latencies": [.01], "scores": [20, 20],
            "floor_losses": [2, 3], "rounds": 5, "unfinished_lines": [1, 1]}


def test_clustered_statistics_control_differences_and_stalls() -> None:
    records = [fake_game(b, s, float(b % 2)) for b in range(20) for s in range(2)]
    records += [fake_game(b, s, 0.0, "opus") for b in range(20) for s in range(2)]
    stats = summarize(records)["2p/mixed/astra"]
    assert stats["win_share"] == .5
    assert stats["ci95"] == interval([float(b % 2) for b in range(20)])
    assert stats["paired_delta_vs_opus"] == .5
    assert stats["paired_delta_ci95"][0] > 0
    records[0]["search"] = [dict(cutoff_reason="time", solved=False, nodes=100, depth=2),
                             dict(cutoff_reason="solved", solved=True, nodes=20, depth=4)]
    searches = summarize(records)["2p/mixed/astra"]
    assert searches["search_calls"] == 2
    assert searches["search_cutoffs"] == {"solved": 1, "time": 1}
    assert searches["solved_rounds"] == 1
    assert searches["mean_search_nodes"] == 60
    assert searches["mean_search_depth"] == 3
    records[0]["finished"] = False
    records[0]["win_share"] = None
    stats = summarize(records)["2p/mixed/astra"]
    assert stats["unfinished"] == 1
    assert stats["ci95"] == [0.0, 1.0]
    assert "paired_delta_ci95" not in stats
    assert not stats["opus_superiority"]


def test_capped_game_remains_explicitly_unfinished() -> None:
    t = schedule([2], 1, "development", 0, ["opus"])[0]
    t.update(config=asdict(AstraConfig(depth=0)), max_turns=1, trace=True, checkpoint=None)
    game = play_game(t)
    assert not game["finished"] and game["win_share"] is None
    assert game["turns"] == len(game["trajectory"]) == 1
    assert game["failure"] is None


def test_bootstrap_is_independent_of_worker_completion_order() -> None:
    import random

    blocks = list(range(21))
    def records(order: list[int]) -> list[dict]:
        games = []
        for block in order:
            share = (block * block % 5) / 4
            for seat in range(2):
                value = (.5 if seat == 0 else 0.0) if share == .25 else share
                games.append(fake_game(block, seat, value))
        return games
    expected = summarize(records(blocks))["2p/mixed/astra"]
    random.Random(4).shuffle(blocks)
    actual = summarize(records(blocks))["2p/mixed/astra"]
    assert actual["ci95"] == expected["ci95"]
    assert actual["block_scores"] == expected["block_scores"]
    assert actual["win_share"] == expected["win_share"]


def test_failed_record_cannot_support_a_strength_claim() -> None:
    records = [fake_game(b, s, 1.0) for b in range(8) for s in range(2)]
    for game in records:
        game["field"] = "opus"
    assert summarize(records)["2p/opus/astra"]["opus_superiority"]
    records[0]["failure"] = "unexplained outcome validation failure"
    stats = summarize(records)["2p/opus/astra"]
    assert stats["failures"] == 1
    assert not stats["opus_superiority"]


def test_shared_draws_do_not_become_superiority_through_rounding() -> None:
    records = []
    for block in range(10):
        for seat in range(3):
            game = fake_game(block, seat, 1 / 3)
            game.update(n=3, field="opus", scores=[20] * 3,
                        floor_losses=[2] * 3, unfinished_lines=[1] * 3)
            records.append(game)
    assert not summarize(records)["3p/opus/astra"]["opus_superiority"]


def test_delivery_exports_records_without_historical_source_archives(tmp_path) -> None:
    import zipfile
    from agent.scripts.export_astra import export

    root = tmp_path / "runs"
    run = root / "old-run"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"spec": {"split": "development"}}))
    (run / "manifest.json").write_text("{}")
    (run / "games.jsonl").write_text("{}\n")
    with zipfile.ZipFile(run / "sources.zip", "w") as archive:
        archive.writestr("play/play_data/store.json", "private-fixture")
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "README.md").write_text("Research report")
    output = tmp_path / "delivery.zip"
    assert export(root, output, documents)["experiments"] == 1
    with zipfile.ZipFile(output) as archive:
        assert "experiments/old-run/games.jsonl" in archive.namelist()
        assert not any("sources.zip" in name or "play_data" in name for name in archive.namelist())


def test_delivery_includes_and_verifies_final_runtime(tmp_path) -> None:
    import zipfile
    from agent.eval.astra_tournament import frozen_identity
    from agent.scripts.export_astra import export

    root = tmp_path / "runs"
    run = root / "final-opus"
    (run / "frozen").mkdir(parents=True)
    source = run / "frozen/policy.py"
    source.write_text("VALUE = 1\n")
    (run / "report.json").write_text(json.dumps({"spec": {"split": "final"}}))
    (run / "games.jsonl").write_text("{}\n")
    with zipfile.ZipFile(run / "sources.zip", "w") as archive:
        archive.write(source, "policy.py")
    (run / "manifest.json").write_text(json.dumps({"frozen_sha256": frozen_identity(run)}))
    documents = tmp_path / "docs"
    documents.mkdir()
    output = tmp_path / "delivery.zip"
    export(root, output, documents)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("experiments/final-opus/frozen/policy.py") == b"VALUE = 1\n"
        assert "experiments/final-opus/sources.zip" in archive.namelist()
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="integrity"):
        export(root, output, documents)


def test_resume_no_duplicates_and_reject_mismatched_spec(tmp_path, monkeypatch) -> None:
    import agent.eval.astra_tournament as tournament
    identity = tournament.source_identity()
    monkeypatch.setattr(tournament, "source_identity", lambda reference_experiment=None: identity)
    args = dict(config=AstraConfig(depth=0), players=[2], blocks=1,
                fields=["opus"], split="development", seed=0, workers=1,
                control=None, max_turns=1)
    first = run_tournament(tmp_path, **args)
    path = tmp_path / "games.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + b'{"interrupted":')
    second = run_tournament(tmp_path, **args)
    assert first["summary"] == second["summary"]
    assert path.read_bytes() == original
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="Resume"):
        run_tournament(tmp_path, **{**args, "seed": 1})


def test_resume_rejects_modified_worker_runtime(tmp_path, monkeypatch) -> None:
    import agent.eval.astra_tournament as tournament
    identity = tournament.source_identity()
    monkeypatch.setattr(tournament, "source_identity", lambda reference_experiment=None: identity)
    args = dict(config=AstraConfig(depth=0), players=[2], blocks=1,
                fields=["opus"], split="development", seed=0, workers=1,
                control=None, max_turns=1)
    run_tournament(tmp_path, **args)
    code = tmp_path / "frozen/agent/eval/heuristic_astra.py"
    code.write_text(code.read_text() + "\n# changed after freeze\n")
    with pytest.raises(ValueError, match="Frozen runtime"):
        run_tournament(tmp_path, **args)


def test_resume_rejects_recorded_seed_drift(tmp_path) -> None:
    args = dict(config=AstraConfig(depth=0), players=[2], blocks=1,
                fields=["opus"], split="development", seed=0, workers=1,
                control=None, max_turns=1)
    run_tournament(tmp_path, **args)
    path = tmp_path / "games.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["engine_seed"] += 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="scheduled seeds or seats"):
        run_tournament(tmp_path, **args)


def test_promotion_gate_requires_complete_matched_evidence() -> None:
    from copy import deepcopy
    from agent.scripts.compare_astra import compare
    spec = dict(players=[2], blocks=128, fields=["opus", "mixed"], seed=0,
                split="promotion", max_turns=400, control=None, sources={}, name="incumbent")
    low = {"spec": spec, "summary": {
        f"2p/{field}/astra": dict(unfinished=0, failures=0, block_scores={str(b): .25 for b in range(128)})
        for field in ("opus", "mixed")}}
    high = deepcopy(low)
    high["spec"]["name"] = "candidate"
    for x in high["summary"].values():
        x["block_scores"] = {str(b): .5 for b in range(128)}
    assert compare(high, low)["player_counts"]["2"]["promote"]
    for x in high["summary"].values():
        x["block_scores"] = {str(b): .25 + 1e-16 for b in range(128)}
    assert not compare(high, low)["player_counts"]["2"]["promote"]
    for x in high["summary"].values():
        x["block_scores"] = {str(b): .5 for b in range(128)}
    high["summary"]["2p/opus/astra"]["unfinished"] = 1
    assert not compare(high, low)["player_counts"]["2"]["promote"]
    high["spec"]["seed"] = 1
    with pytest.raises(ValueError, match="seed"):
        compare(high, low)


def test_ratings_use_official_pairwise_ties_and_require_random_connection() -> None:
    from agent.scripts.rate_astra import ratings
    game = dict(n=3, names=["astra", "opus", "random"], finished=True, failure=None,
                scores=[50, 50, 20], rows=[1, 1, 0])
    result = ratings([game])
    pair = next(p for p in result["pairwise_records"] if (p["a"], p["b"]) == ("astra", "opus"))
    assert pair["ties_3p"] == 1
    values = result["per_player_count"]["3"]["ratings"]
    assert values["random"] == 1000
    assert values["astra"] == pytest.approx(values["opus"], abs=.01)
    assert values["astra"] > values["random"]
    isolated = {**game, "n": 2, "names": ["astra", "opus"], "scores": [50, 40], "rows": [1, 0]}
    assert ratings([isolated])["per_player_count"]["2"]["ratings"] == {}


def test_reference_runtime_is_overlaid_archived_and_verified(tmp_path) -> None:
    import zipfile
    from agent.eval.astra_tournament import atomic_json, digest, frozen_identity
    ref, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    args = dict(config=AstraConfig(depth=0), players=[2], blocks=1,
                fields=['opus'], split='development', seed=0, workers=1,
                control=None, max_turns=1)
    run_tournament(ref, **args)
    old_bot = ref / 'frozen/agent/eval/heuristic_opus.py'
    old_bot.write_text(old_bot.read_text() + '\n# frozen historical policy revision\n')
    manifest = json.loads((ref / 'manifest.json').read_text())
    manifest['frozen_sha256'] = frozen_identity(ref)
    atomic_json(ref / 'manifest.json', manifest)
    run_tournament(candidate, reference_experiment=ref, **args)
    copied = candidate / 'frozen/agent/eval/heuristic_opus.py'
    assert copied.read_bytes() == old_bot.read_bytes()
    result = json.loads((candidate / 'manifest.json').read_text())
    assert result['spec']['sources']['agent/eval/heuristic_opus.py'] == digest(old_bot)
    with zipfile.ZipFile(candidate / 'sources.zip') as z:
        assert z.read('agent/eval/heuristic_opus.py') == old_bot.read_bytes()
    old_bot.write_text(old_bot.read_text() + '\n# corruption\n')
    with pytest.raises(ValueError, match='Reference experiment artifacts changed'):
        run_tournament(tmp_path / 'rejected', reference_experiment=ref, **args)


def test_concurrent_writer_cannot_enter_the_same_experiment(tmp_path) -> None:
    import fcntl
    with (tmp_path / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='already in use'):
            run_tournament(tmp_path, config=AstraConfig(depth=0), players=[2], blocks=1,
                           fields=['opus'], split='development', seed=0, workers=1,
                           control=None, max_turns=1)


def test_explicit_archive_resume_restores_missing_game_and_rejects_tamper(tmp_path) -> None:
    import subprocess
    import sys
    from agent.scripts.resume_astra import DRIVER

    run_tournament(tmp_path, config=AstraConfig(depth=0), players=[2], blocks=1,
                   fields=['opus'], split='development', seed=0, workers=1,
                   control=None, max_turns=1)
    path = tmp_path / 'games.jsonl'
    original = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(json.dumps(original[0]) + '\n' + '{"partial":')
    result = subprocess.run([sys.executable, '-c', DRIVER, str(tmp_path), '1'],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1])['games'] == 2
    restored = [json.loads(line) for line in path.read_text().splitlines()]
    assert restored[0] == original[0]
    for key in ('key', 'engine_seed', 'bot_seed', 'scores', 'finished', 'win_share'):
        assert restored[1][key] == original[1][key]
    code = tmp_path / 'frozen/agent/eval/heuristic_astra.py'
    code.write_text(code.read_text() + '\nraise RuntimeError("must never execute")\n')
    result = subprocess.run([sys.executable, '-c', DRIVER, str(tmp_path), '1'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'integrity verification' in result.stderr
    assert 'must never execute' not in result.stderr


def test_source_archive_excludes_runtime_player_data(tmp_path, monkeypatch) -> None:
    import zipfile
    import agent.eval.astra_tournament as tournament

    repo = tmp_path / 'repo'
    for name in ('agent', 'native/astra', 'play/play_data', 'play/artifacts', 'infra'):
        (repo / name).mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'README.md', 'MANIFEST.in', '.dockerignore', '.gitignore'):
        (repo / name).write_text('')
    (repo / 'play/models.py').write_text('# source\n')
    (repo / 'play/play_data/store.json').write_text('{"private_game": true}')
    (repo / 'play/artifacts/registry.json').write_text('{"runtime": true}')
    output = tmp_path / 'experiment'
    output.mkdir()
    monkeypatch.setattr(tournament, 'ROOT', repo)
    tournament.freeze_sources(output)
    with zipfile.ZipFile(output / 'sources.zip') as archive:
        assert 'play/models.py' in archive.namelist()
        assert not any('/play_data/' in name or '/artifacts/' in name for name in archive.namelist())


def test_reference_checkpoint_can_be_readonly_and_shared_by_player_counts(tmp_path) -> None:
    from agent.eval.astra_tournament import atomic_json, digest, freeze_checkpoints

    reference, output = tmp_path / 'reference', tmp_path / 'output'
    (reference / 'checkpoints').mkdir(parents=True)
    output.mkdir()
    checkpoint = reference / 'checkpoints/model.pt'
    checkpoint.write_bytes(b'frozen checkpoint bytes')
    checkpoint.chmod(0o444)
    entry = {'path': str(checkpoint), 'sha256': digest(checkpoint), 'selection': 'rating'}
    atomic_json(reference / 'manifest.json', {'checkpoints': {str(n): entry for n in (2, 3, 4)}})
    result = freeze_checkpoints(output, reference)
    assert set(result) == {'2', '3', '4'}
    assert (output / 'checkpoints/model.pt').read_bytes() == checkpoint.read_bytes()


def test_frozen_runtime_can_start_a_new_experiment_without_git(tmp_path) -> None:
    import subprocess
    import sys

    original, replay = tmp_path / 'original', tmp_path / 'replay'
    run_tournament(original, config=AstraConfig(depth=0), players=[2], blocks=1,
                   fields=['opus'], split='development', seed=0, workers=1,
                   control=None, max_turns=1)
    subprocess.run([sys.executable, '-m', 'agent.scripts.eval_astra', '--output', str(replay),
                    '--players', '2', '--blocks', '1', '--fields', 'opus', '--depth', '0',
                    '--max-turns', '1', '--workers', '1'], cwd=original / 'frozen',
                   capture_output=True, text=True, check=True)
    manifest = json.loads((replay / 'manifest.json').read_text())
    assert manifest['environment']['git_head'] is None
    assert (replay / 'frozen/agent/eval/heuristic_astra.py').read_bytes() == (
        original / 'frozen/agent/eval/heuristic_astra.py').read_bytes()
    assert json.loads((replay / 'report.json').read_text())['complete']
