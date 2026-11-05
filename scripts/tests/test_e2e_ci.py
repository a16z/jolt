import argparse
import contextlib
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).parents[1] / 'bench'
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC = importlib.util.spec_from_file_location('e2e_ci', DIRECTORY / 'ci.py')
    ci = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(ci)
finally:
    sys.path.pop(0)


class E2eCiTests(unittest.TestCase):
    def test_dispatch_and_target_bind_each_tier(self):
        for tier in ('pr', 'main', 'nightly', 'calibration'):
            plan = {'enabled': True, 'tier': tier,
                'provenance': {'run_id': '123', 'run_attempt': '2', 'pr_number': 7,
                    'workflow': ci.bench.WORKFLOW, 'event': {'pr': 'pull_request_target', 'main': 'push',
                        'nightly': 'schedule', 'calibration': 'workflow_dispatch'}[tier],
                    'ref': 'refs/heads/main'},
                'runs': [{'submitted_head': 'b' * 40, 'head': 'c' * 40}],
                'admission': {'label_event_id': 11, 'label_actor': 'maintainer'},
                'expected_builds': {'c' * 40: {}}}
            manifest = ci.dispatch_manifest(plan, 123, 2, artifact_id=456,
                index_sha256='1' * 64, digests={'c' * 40: '2' * 64})
            self.assertEqual(manifest['plan_artifact'], 'jolt-e2e-plan-123-2')
            self.assertEqual(manifest['check_sha'], ('b' if tier == 'pr' else 'c') * 40)
            self.assertEqual(manifest['builds_artifact'], None if tier == 'pr' else 'jolt-e2e-builds-123-2')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = root / 'dispatch.json'
                request.write_text(json.dumps(manifest))
                options = argparse.Namespace(plan=root / 'plan.json', dispatch=request,
                                             bundles=root / 'bundles', out=root / 'out')
                with patch.object(ci, 'load_plan', return_value=plan), patch.dict(os.environ), \
                        patch.object(ci, 'local', return_value=0) as local, \
                        patch.object(ci, 'execute', return_value=0) as execute:
                    self.assertEqual(ci.target(options), 0)
                    self.assertEqual(local.called, tier == 'pr')
                    self.assertEqual(execute.called, tier != 'pr')
                    if tier != 'pr':
                        self.assertEqual(execute.call_args.args[0].index_sha256, '1' * 64)
                    (local if tier == 'pr' else execute).side_effect = ValueError('owned failure')
                    with self.assertRaisesRegex(ValueError, 'owned failure'):
                        ci.target(options)
            for field, value in [('run_attempt', 1), ('tier', 'other'), ('check_sha', 'f' * 40)]:
                with self.subTest(tier=tier, field=field), self.assertRaises(ValueError):
                    ci.check_dispatch(dict(manifest, **{field: value}), plan)

    def test_pr_admission_rejects_unlabelled_fork_nonmaintainer_and_stale_events(self):
        pr = {'number': 7, 'state': 'open', 'labels': [{'name': 'benchmark'}],
              'head': {'sha': 'b' * 40, 'ref': 'candidate', 'repo': {'full_name': 'a16z/jolt'}},
              'base': {'sha': 'a' * 40, 'ref': 'main', 'repo': {'full_name': 'a16z/jolt'}}}
        labelled = {'id': 11, 'event': 'labeled', 'label': {'name': 'benchmark'},
                    'actor': {'login': 'maintainer'}, 'created_at': '2026-11-02T18:00:00Z'}
        for fault in ('unlabelled', 'fork', 'write-only', 'removed', 'stale-head', 'untrusted-event'):
            current, timeline, role = copy.deepcopy(pr), [labelled], 'maintain'
            event = {'action': 'synchronize', 'pull_request': copy.deepcopy(pr)}
            name = 'pull_request_target'
            if fault == 'unlabelled':
                current['labels'] = []
            elif fault == 'fork':
                current['head']['repo']['full_name'] = 'outside/jolt'
            elif fault == 'write-only':
                role = 'write'
            elif fault == 'removed':
                timeline += [dict(labelled, id=12, event='unlabeled')]
            elif fault == 'stale-head':
                current['head']['sha'] = 'e' * 40
            else:
                name = 'pull_request'
            def api(path, **_):
                return ({'role_name': role} if path.startswith('collaborators/')
                        else timeline if '/events?' in path else current)
            with self.subTest(fault=fault), patch.object(ci, 'github', side_effect=api), \
                    patch.object(ci.bench, 'git_revision', return_value='a' * 40), \
                    patch.object(ci.bench, 'capture') as capture:
                with self.assertRaises(ValueError):
                    ci.resolve(name, event, 'a16z/jolt', 'a' * 40)
                capture.assert_not_called()

    def test_maintainer_label_persists_on_synchronize_and_queue_recheck_is_public(self):
        pr = {'number': 7, 'state': 'open', 'labels': [{'name': 'benchmark'}],
              'head': {'sha': 'b' * 40, 'ref': 'candidate', 'repo': {'full_name': 'a16z/jolt'}},
              'base': {'sha': 'f' * 40, 'ref': 'main', 'repo': {'full_name': 'a16z/jolt'}},
              'merge_commit_sha': 'c' * 40}
        labelled = {'id': 11, 'event': 'labeled', 'label': {'name': 'benchmark'},
                    'actor': {'login': 'maintainer'}, 'created_at': '2026-11-02T18:00:00Z'}
        def api(path, **_):
            return ({'role_name': 'maintain'} if path.startswith('collaborators/')
                    else [labelled] if '/events?' in path else pr)
        revisions = {'HEAD': 'a' * 40, 'refs/e2e/merge': 'c' * 40,
                     'c' * 40 + '^1': 'a' * 40, 'c' * 40 + '^2': 'b' * 40,
                     'refs/e2e/main': 'a' * 40}
        for action in ('labeled', 'synchronize'):
            event = {'action': action, 'label': {'name': 'benchmark'},
                     'sender': {'login': 'maintainer'}, 'pull_request': pr}
            with self.subTest(action=action), patch.object(ci, 'github', side_effect=api), \
                    patch.object(ci.bench, 'git_revision', side_effect=revisions.__getitem__), \
                    patch.object(ci.bench, 'capture'):
                harness, runs, number, admission = ci.resolve('pull_request_target', event, 'a16z/jolt', 'a' * 40)
            self.assertEqual((harness, number), ('a' * 40, 7))
            self.assertEqual(runs, [{'base': 'a' * 40, 'head': 'c' * 40, 'submitted_head': 'b' * 40}])
            self.assertEqual(admission['label_event_id'], 11)
        plan = {'runs': runs, 'provenance': {'pr_number': 7}, 'admission': admission}
        pr['merge_commit_sha'] = 'd' * 40
        with patch.object(ci, 'github', side_effect=api) as read:
            ci.recheck_pr(plan)
            self.assertTrue(all(call.kwargs == {'authenticated': False} for call in read.call_args_list))
            pr['head']['sha'] = 'e' * 40
            with self.assertRaises(ValueError):
                ci.recheck_pr(plan)
            pr['head']['sha'] = 'b' * 40
            labelled['id'] = 12
            with self.assertRaises(ValueError):
                ci.recheck_pr(plan)

    def test_main_push_keeps_all_new_first_parent_commits_in_event_order(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            env = dict(os.environ, GIT_AUTHOR_NAME='fixture', GIT_COMMITTER_NAME='fixture',
                       GIT_AUTHOR_EMAIL='fixture@example.invalid', GIT_COMMITTER_EMAIL='fixture@example.invalid')
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            tree = subprocess.check_output(['git', 'mktree'], input='', text=True, cwd=repo).strip()
            def commit(label, parents):
                args = ['git', 'commit-tree', tree, '-m', label]
                for parent in parents:
                    args.extend(['-p', parent])
                return subprocess.check_output(args, cwd=repo, env=env, text=True).strip()
            before = commit('before', [])
            first = commit('first main commit', [before])
            side = commit('side branch', [before])
            merge = commit('main merge', [first, side])
            after = commit('tip', [merge])
            subprocess.run(['git', 'update-ref', 'HEAD', after], cwd=repo, check=True)
            real_capture = ci.bench.capture
            with patch.object(ci.bench, 'ROOT', repo), patch.object(ci.bench, 'capture',
                    side_effect=lambda argv, **kwargs: real_capture(argv, cwd=repo)):
                harness, runs, number, request = ci.resolve('push', {'before': before, 'after': after},
                                                           'a16z/jolt', after)
            self.assertEqual(harness, after)
            self.assertEqual([item['head'] for item in runs], [first, merge, after])
            self.assertIsNone(number)
            self.assertIsNone(request)

    def test_batch_artifact_has_root_tip_and_explicit_earlier_commit_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = {'workflow': ci.bench.WORKFLOW, 'workflow_file': ci.bench.WORKFLOW_FILE,
                          'run_id': '123', 'run_attempt': '1', 'event': 'push',
                          'ref': 'refs/heads/main', 'pr_number': None, 'request_run_id': None}
            plan = {'enabled': True, 'tier': 'main', 'harness': 'c' * 40, 'target_cpu': 'znver4',
                    'config': ci.bench.bundle_file(ci.bench.DEFAULT_CONFIG, ci.bench.ROOT),
                    'reason': None, 'push_before': 'a' * 40, 'push_after': 'c' * 40,
                    'provenance': provenance, 'runs': [
                        {'base': None, 'head': 'b' * 40, 'submitted_head': 'b' * 40},
                        {'base': None, 'head': 'c' * 40, 'submitted_head': 'c' * 40}]}
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            args = argparse.Namespace(plan=plan_path, out=root / 'artifact', artifact_id=456, index_sha256='1' * 64)
            def run(options):
                self.assertEqual(options.context['provenance'], provenance)
                return {'status': 'complete', 'fixture_revision': options.head.parent.name,
                        'wall_time': {'benchmark_s': 0}}
            with patch.object(ci.bench, 'git_revision', return_value='c' * 40), \
                    patch.dict(os.environ, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1'), \
                    patch.object(ci.bench, 'capture') as capture, \
                    patch.object(ci.bench, 'build', side_effect=AssertionError('target must not build')), \
                    patch.object(ci, 'verified_bundles', return_value={
                        rev * 40: root / (rev * 40) / 'manifest.json' for rev in ('b', 'c')}), \
                    patch.object(ci.bench, 'host_lock', side_effect=contextlib.nullcontext), \
                    patch.object(ci.bench, 'run', side_effect=run) as measured:
                self.assertEqual(ci.execute(args), 0)
                capture.assert_not_called()
                capture.reset_mock()
                measured.reset_mock()
                for name, value in [('GITHUB_RUN_ID', '124'), ('GITHUB_RUN_ATTEMPT', '2')]:
                    with self.subTest(mismatch=name), patch.dict(os.environ, {name: value}):
                        with self.assertRaisesRegex(ValueError, 'different run or attempt'):
                            ci.execute(args)
                        capture.assert_not_called()
                        measured.assert_not_called()
            index = json.loads((args.out / 'index.json').read_text())
            self.assertEqual(index, {'schema_version': 1, 'push_before': 'a' * 40, 'push_after': 'c' * 40,
                'documents': [{'revision': 'b' * 40, 'path': f'commits/{"b" * 40}/benchmark-results.json'},
                              {'revision': 'c' * 40, 'path': 'benchmark-results.json'}]})
            self.assertEqual(json.loads((args.out / 'benchmark-results.json').read_text())['fixture_revision'], 'c' * 40)

    def test_disabled_result_keeps_expected_cases_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(config=ci.bench.DEFAULT_CONFIG, tier='pr', out=Path(directory),
                                      status='disabled', reason='fixture: runner not enabled')
            with patch.object(ci.bench, 'provenance', return_value=({'run_id': None}, {})):
                result = ci.bench.disabled(args)
            self.assertEqual(result['status'], 'disabled')
            self.assertEqual(len(result['cases']), 17)
            self.assertEqual({case['pcs'] for case in result['cases']}, {'akita', 'dory'})
            self.assertTrue(all(c['status'] == 'missing' and not c['samples'] for c in result['cases']))
            self.assertIsNone(result['wall_time']['total_s'])

    def test_local_build_and_proof_share_the_owned_host_lock_and_keep_current_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            plan.write_text('{}')
            phases = []
            def held():
                with (root / 'host.lock').open('a') as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def prepare(options, *, locked):
                self.assertTrue(locked)
                held()
                phases.append('build')
                ci.bench.write_json(options.out / 'builds.json', {})
            def execute(options, *, locked):
                self.assertTrue(locked)
                held()
                phases.append('prove')
                self.assertEqual(phases, ['build', 'prove'])
                self.assertIsNone(options.artifact_id)
                ci.bench.write_json(options.out / 'benchmark-results.json',
                    {'status': 'complete', 'wall_time': {'benchmark_s': 0, 'build_s': 0}})
                return 0
            with patch.object(ci, 'load_plan', return_value={'tier': 'pr'}), \
                    patch.object(ci, 'recheck_pr'), patch.object(ci.bench, 'cache_root', return_value=root), \
                    patch.object(ci, 'prepare', side_effect=prepare), patch.object(ci, 'execute', side_effect=execute), \
                    patch.object(ci.bench, 'render_comment'):
                self.assertEqual(ci.local(argparse.Namespace(plan=plan, out=root / 'out')), 0)
            with (root / 'host.lock').open('a') as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = json.loads((root / 'out/benchmark-results.json').read_text())
            self.assertEqual(result['wall_time']['scope'], ci.LOCAL_SCOPE)
            self.assertGreaterEqual(result['wall_time']['total_s'], result['wall_time']['build_s'])

    def test_replayed_label_run_is_refused_before_native_input_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / 'event.json'
            event.write_text('{}')
            resolved = ('a' * 40, [{'base': 'a' * 40, 'head': 'c' * 40, 'submitted_head': 'b' * 40}], 7,
                        {'label_created_at': '2026-11-02T18:00:02Z'})
            actual = {'event': 'pull_request_target', 'path': ci.bench.WORKFLOW_FILE, 'head_sha': 'b' * 40,
                      'run_attempt': 2, 'created_at': '2026-11-02T18:00:01Z'}
            with patch.dict(os.environ, GITHUB_EVENT_PATH=str(event), GITHUB_EVENT_NAME='pull_request_target',
                    GITHUB_REPOSITORY='a16z/jolt', GITHUB_REF='refs/heads/main', GITHUB_SHA='a' * 40, GITHUB_RUN_ID='123',
                    GITHUB_RUN_ATTEMPT='2', JOLT_E2E_ENABLED='true', GITHUB_OUTPUT=str(root / 'outputs'),
                    GITHUB_STEP_SUMMARY=str(root / 'summary')), patch.object(ci, 'resolve', return_value=resolved), \
                    patch.object(ci, 'github', return_value=actual), patch.object(ci, 'git_file') as source:
                self.assertEqual(ci.plan(argparse.Namespace(config=ci.bench.DEFAULT_CONFIG, out=root / 'plan')), 0)
                source.assert_not_called()
            receipt = json.loads((root / 'plan/plan.json').read_text())
            self.assertFalse(receipt['enabled'])
            self.assertIn('stale or replayed', receipt['reason'])

    def test_unsupported_privileged_requests_keep_a_successful_receipt_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / 'event.json'
            event.write_text('{}')
            options = argparse.Namespace(config=ci.bench.DEFAULT_CONFIG, out=root / 'unsupported')
            with patch.dict(os.environ, GITHUB_EVENT_PATH=str(event),
                    GITHUB_EVENT_NAME='pull_request_target', GITHUB_REPOSITORY='a16z/jolt', GITHUB_REF='refs/heads/main',
                    GITHUB_SHA='a' * 40, JOLT_E2E_ENABLED='true',
                    GITHUB_OUTPUT=str(root / 'outputs'), GITHUB_STEP_SUMMARY=str(root / 'summary')), \
                    patch.object(ci, 'resolve', side_effect=ValueError('unsupported fixture request')):
                self.assertEqual(ci.plan(options), 0)
            self.assertFalse(json.loads((options.out / 'plan.json').read_text())['enabled'])
            receipt = json.loads((options.out / 'benchmark-results.json').read_text())
            self.assertEqual(receipt['status'], 'unsupported')
            self.assertTrue(all(not case['samples'] for case in receipt['cases']))

    def test_hosted_preparation_uses_release_for_every_tier_and_never_proves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            cache.mkdir()
            spawn = subprocess.Popen
            for tier in ('pr', 'main', 'nightly'):
                with self.subTest(tier=tier):
                    plan = {'enabled': True, 'tier': tier, 'harness': 'c' * 40,
                            'target_cpu': 'znver4', 'reason': None,
                            'config': ci.bench.bundle_file(ci.bench.DEFAULT_CONFIG, ci.bench.ROOT),
                            'expected_builds': {'a' * 40: {}, 'b' * 40: {}},
                            'push_before': None, 'push_after': 'b' * 40,
                            'provenance': {'workflow': ci.bench.WORKFLOW,
                                'workflow_file': ci.bench.WORKFLOW_FILE,
                                'run_id': '123', 'run_attempt': '1',
                                'event': 'pull_request_target' if tier == 'pr' else ('push' if tier == 'main' else 'schedule'),
                                'ref': 'refs/heads/main', 'pr_number': 7 if tier == 'pr' else None,
                                'request_run_id': 99 if tier == 'pr' else None},
                            'runs': [{'base': 'a' * 40 if tier == 'pr' else None,
                                      'head': 'b' * 40, 'submitted_head': ('d' if tier == 'pr' else 'b') * 40}]}
                    plan_path = root / 'plan.json'
                    plan_path.write_text(json.dumps(plan))
                    built = []
                    children = []
                    def launch(*args, **kwargs):
                        child = spawn(*args, **kwargs)
                        children.append(child)
                        return child
                    def build(options):
                        self.assertEqual(options.profile, 'release')
                        self.assertEqual(options.target_cpu, 'znver4')
                        self.assertEqual(options.harness_revision, 'c' * 40)
                        self.assertEqual(options.allow_harness_fallback, tier == 'pr')
                        built.append(options.revision)
                        manifest = {'status': 'complete', 'revision': options.revision}
                        ci.bench.write_json(options.out / 'manifest.json', manifest)
                        return manifest
                    with patch.dict(os.environ, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1'), \
                            patch.object(ci.bench, 'git_revision', return_value='c' * 40), \
                            patch.object(ci.bench, 'capture'), \
                            patch.object(ci.bench, 'cache_root', return_value=cache), \
                            patch.object(ci.bench, 'host_lock', side_effect=contextlib.nullcontext), \
                            patch.object(ci.subprocess, 'Popen', side_effect=launch), \
                            patch.object(ci.bench, 'run', side_effect=AssertionError('builder must not prove')), \
                            patch.object(ci.bench, 'build', side_effect=build):
                        ci.prepare(argparse.Namespace(plan=plan_path, out=root / tier))
                        if tier == 'pr':
                            with patch.object(ci.bench, 'build', side_effect=ValueError('fixture build failed')):
                                with self.assertRaisesRegex(ValueError, 'fixture build failed'):
                                    ci.prepare(argparse.Namespace(plan=plan_path, out=root / 'failure'))
                        for child in children:
                            self.assertIsNotNone(child.returncode)
                            with self.assertRaises(ChildProcessError):
                                os.waitpid(child.pid, os.WNOHANG)
                    index = json.loads((root / tier / 'builds.json').read_text())
                    self.assertEqual(built, ['a' * 40, 'b' * 40])
                    self.assertEqual(index['plan_sha256'], ci.bench.sha256(plan_path))
                    self.assertEqual(set(index['builds']), set(built))
                    argv = ['run.py', 'suite', '--tier', tier, '--head-revision', 'b' * 40,
                            '--harness-revision', 'c' * 40, '--target-cpu', 'znver4', '--out', str(root)]
                    with patch.object(sys, 'argv', argv):
                        self.assertEqual(ci.bench.parse_args().profile, 'release')

    def test_interrupted_batch_closes_partial_documents_and_keeps_missing_tip_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {'tier': 'main', 'harness': 'c' * 40, 'provenance': {},
                    'config': ci.bench.bundle_file(ci.bench.DEFAULT_CONFIG, ci.bench.ROOT), 'runs': [
                {'base': None, 'head': 'b' * 40, 'submitted_head': 'b' * 40},
                {'base': None, 'head': 'c' * 40, 'submitted_head': 'c' * 40}]}
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            partial = root / 'commits' / ('b' * 40) / 'benchmark-results.json'
            ci.bench.write_json(partial, {'status': 'incomplete', 'reason': None, 'finished_at': None})
            ci.finalize(argparse.Namespace(plan=plan_path, out=root))
            earlier = json.loads(partial.read_text())
            tip = json.loads((root / 'benchmark-results.json').read_text())
            self.assertTrue(earlier['finished_at'].endswith('Z'))
            self.assertIn('interrupted', earlier['reason'])
            self.assertEqual(tip['revisions']['head'], 'c' * 40)
            self.assertEqual(tip['status'], 'incomplete')
            self.assertTrue(all(case['status'] == 'missing' for case in tip['cases']))

    def test_run_only_verifies_planned_artifacts_before_launch_and_never_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'scripts/bench/suites.json'
            config = {'schema_version': 1, 'metric_contract': 1, 'id': 'owned-fixture',
                'tiers': {'pr': {'warmups': 0, 'repetitions': 1, 'cases': [{
                    'id': 'owned-case', 'workload': 'fibonacci', 'pcs': 'akita', 'backend': 'optimized',
                    'physical_cores': 1, 'scale_log2': 3,
                    'input': {'parameters': {'operations': 1}, 'seed': None}}]}}}
            ci.bench.write_json(config_path, config)
            script = Path(ci.bench.__file__)
            common = {'schema_version': 1, 'harness_revision': 'c' * 40,
                'native_harness_revision': 'c' * 40, 'status': 'complete',
                'toolchain': 'fixture', 'cargo_version': 'fixture', 'profile': 'release',
                'profile_config': {'lto': 'fat'}, 'guest_builder_profile': 'ci', 'guest_builder_rustflags': '',
                'target_cpu': 'znver4', 'target_triple': 'x86_64-unknown-linux-gnu',
                'target_features': [], 'rustflags': '-C target-cpu=znver4',
                'source_cargo_lock_sha256': '0' * 64, 'cargo_lock_sha256': '0' * 64,
                'harness_files': {}, 'akita_schedules': [], 'suite_config_sha256': ci.bench.sha256(config_path),
                'runner': {'path': 'scripts/bench/run.py', 'sha256': ci.bench.sha256(script)},
                'calibration_helper': {'path': 'scripts/bench/calibration.py', 'sha256': ci.bench.sha256(ci.bench.calibration.__file__)},
                'host': {}, 'started_at': '2026-11-01T00:00:00Z', 'finished_at': '2026-11-01T00:00:01Z',
                'build_s': 1, 'reason': None, 'cache': {}}
            expected = {}
            original = {}
            for revision in ('a' * 40, 'b' * 40):
                bundle = root / 'bundles' / revision
                files = {'guests/fibonacci/guest.elf': b'owned guest',
                         'guests/fibonacci/input-1.bin': b'\x01',
                         'scripts/bench/run.py': script.read_bytes(),
                         'scripts/bench/calibration.py': Path(ci.bench.calibration.__file__).read_bytes(),
                         'scripts/bench/suites.json': config_path.read_bytes()}
                for name, contents in files.items():
                    path = bundle / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(contents)
                binary = bundle / 'jolt-prover-akita'
                binary.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
assert not any(key in os.environ for key in ('GH_TOKEN', 'PIKA_CLI_TOKEN', 'ACTIONS_RUNTIME_TOKEN'))
assert '--guest-elf' in sys.argv and '--input-file' in sys.argv
Path(sys.argv[sys.argv.index('--result-json') + 1]).write_text(json.dumps({
    'workload': 'fibonacci', 'pcs': 'akita', 'backend': 'optimized', 'scale_log2': 3,
    'e2e_s': 0.003, 'prove_s': 0.001, 'setup_s': 0.001, 'verify_s': 0.0001,
    'verify_single_s': 0.0001, 'proof_bytes': 12, 'trace_rows': 4, 'padded_trace_rows': 8,
    'public_output': [42]}))
''')
                binary.chmod(0o644)
                manifest = dict(common, revision=revision,
                    binaries={'akita': ci.bench.bundle_file(binary, bundle)},
                    guests={'fibonacci': {
                        'elf': ci.bench.bundle_file(bundle / 'guests/fibonacci/guest.elf', bundle),
                        'inputs': {'1': ci.bench.bundle_file(bundle / 'guests/fibonacci/input-1.bin', bundle)}}})
                expected[revision] = {key: manifest[key] for key in (
                    'revision', 'harness_revision', 'status', 'profile', 'profile_config', 'target_cpu',
                    'rustflags', 'source_cargo_lock_sha256', 'harness_files', 'runner', 'suite_config_sha256')}
                original[revision] = manifest
                ci.bench.write_json(bundle / 'manifest.json', manifest)
            plan = {'enabled': True, 'tier': 'pr', 'target_cpu': 'znver4', 'harness': 'c' * 40,
                'push_before': None, 'config': ci.bench.bundle_file(config_path, root),
                'provenance': {'run_id': None, 'run_attempt': None}, 'expected_builds': expected,
                'runs': [{'base': 'a' * 40, 'head': 'b' * 40, 'submitted_head': 'd' * 40}]}
            plan_path = root / 'plan.json'
            ci.bench.write_json(plan_path, plan)
            index_path = root / 'bundles/builds.json'
            index = {'plan_sha256': ci.bench.sha256(plan_path), 'builds': {
                rev: ci.bench.bundle_file(root / 'bundles' / rev / 'manifest.json', root / 'bundles')
                for rev in original}}
            ci.bench.write_json(index_path, index)
            args = argparse.Namespace(plan=plan_path, bundles=index_path.parent, artifact_id=456,
                                      index_sha256=ci.bench.sha256(index_path), out=root / 'result')
            with patch.object(ci.bench, 'ROOT', root), \
                    patch.object(ci.bench, 'git_revision', return_value='c' * 40), \
                    patch.object(ci.bench, 'capture', side_effect=AssertionError('target must not fetch or compile')), \
                    patch.object(ci, 'recheck_pr'), \
                    patch.object(ci.bench, 'build', side_effect=AssertionError('target must not build')), \
                    patch.object(ci.bench, 'host_lock', side_effect=contextlib.nullcontext), \
                    patch.dict(os.environ, {'GITHUB_RUN_ID': '', 'GITHUB_RUN_ATTEMPT': '',
                        'GH_TOKEN': 'owned fixture', 'PIKA_CLI_TOKEN': 'owned fixture',
                        'ACTIONS_RUNTIME_TOKEN': 'owned fixture'}):
                manifest_path = root / 'bundles' / ('b' * 40) / 'manifest.json'
                for key, value in [('revision', 'e' * 40), ('harness_revision', 'e' * 40),
                        ('native_harness_revision', 'e' * 40),
                        ('harness_files', {'unplanned.rs': 'f' * 64}),
                        ('profile', 'ci'), ('target_cpu', 'native'), ('source_cargo_lock_sha256', 'f' * 64),
                        ('guests', {})]:
                    changed = copy.deepcopy(original['b' * 40])
                    changed[key] = value
                    ci.bench.write_json(manifest_path, changed)
                    changed_index = copy.deepcopy(index)
                    changed_index['builds']['b' * 40]['sha256'] = ci.bench.sha256(manifest_path)
                    ci.bench.write_json(index_path, changed_index)
                    args.index_sha256 = ci.bench.sha256(index_path)
                    with self.subTest(key=key), patch.object(ci.bench, 'run') as run:
                        with self.assertRaises((ValueError, KeyError)):
                            ci.execute(args)
                        run.assert_not_called()
                ci.bench.write_json(manifest_path, original['b' * 40])
                ci.bench.write_json(index_path, index)
                args.index_sha256 = ci.bench.sha256(index_path)
                input_path = manifest_path.parent / 'guests/fibonacci/input-1.bin'
                input_path.write_bytes(b'wrong input')
                with patch.object(ci.bench, 'run') as run:
                    with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                        ci.execute(args)
                    run.assert_not_called()
                input_path.write_bytes(b'\x01')
                args.index_sha256 = '0' * 64
                with patch.object(ci.bench, 'run') as run:
                    with self.assertRaisesRegex(ValueError, 'hosted job output'):
                        ci.execute(args)
                    run.assert_not_called()
                args.index_sha256 = ci.bench.sha256(index_path)
                cpu = min(os.sched_getaffinity(0))
                host = {'architecture': 'x86_64', 'cpu_flags': [],
                        'topology': [{'cpu': cpu, 'core': 0, 'socket': 0, 'online': True}]}
                with patch.object(ci.bench, 'host_snapshot', return_value=host):
                    self.assertEqual(ci.execute(args), 0)
                    fallback = copy.deepcopy(original['b' * 40])
                    fallback.update(native_harness_revision='b' * 40, harness_fallback_reason='file set differs',
                                    harness_files={'new.rs': 'f' * 64},
                                    harness_diff={'text': '+fn new_api() {}', 'truncated': False})
                    plan['own_harness_files'] = {'b' * 40: fallback['harness_files']}
                    ci.bench.write_json(plan_path, plan)
                    ci.bench.write_json(manifest_path, fallback)
                    index['plan_sha256'] = ci.bench.sha256(plan_path)
                    index['builds']['b' * 40]['sha256'] = ci.bench.sha256(manifest_path)
                    ci.bench.write_json(index_path, index)
                    args.index_sha256 = ci.bench.sha256(index_path)
                    args.out = root / 'fallback-result'
                    self.assertEqual(ci.execute(args), 0)
            result = json.loads((args.out / 'benchmark-results.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['wall_time']['build_s'], 0)
            self.assertEqual(result['wall_time']['scope'], ci.RUN_ONLY_SCOPE)
            self.assertEqual(result['artifacts']['id'], 456)
            self.assertEqual(len(result['cases'][0]['samples']), 2)


if __name__ == '__main__':
    unittest.main()


class CalibrationSchedulingTests(unittest.TestCase):
    def test_main_nightly_and_bootstrap_use_tip_null_before_optional_local_probe(self):
        for tier in ('main', 'nightly', 'calibration'):
            with self.subTest(tier=tier), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = argparse.Namespace(plan=root / 'plan.json', out=root / 'out',
                    bundles=root / 'hosted', artifact_id=44, index_sha256='a' * 64)
                args.plan.write_text('{}')
                plan = dict(tier=tier, config={'path': 'scripts/bench/suites.json'}, provenance={},
                            runs=[{'head': 'a' * 40}, {'head': 'b' * 40}])
                events = []
                def prove(options):
                    events.append((options.kind, options.tier, options.warmups, options.repetitions,
                                   options.base, options.head, len(ci.bench.config_cases(options.config, 'pr')[1])))
                    return dict(status='complete', wall_time={'benchmark_s': 0})
                def prepare(options, **kwargs):
                    events.append('local-build')
                    ci.bench.write_json(options.out / 'builds.json', {})
                with patch.object(ci, 'verified_bundles', side_effect=lambda options, *_:
                        {'b' * 40: options.bundles / 'manifest.json'}), \
                        patch.object(ci.bench, 'run', side_effect=prove), patch.object(ci, 'prepare', side_effect=prepare):
                    self.assertFalse(ci.calibration_phase(args, plan))
                self.assertEqual(events[0], ('null', 'pr', 1, 10,
                    args.bundles / 'manifest.json', args.bundles / 'manifest.json', 17))
                if tier == 'calibration':
                    self.assertEqual(events[1], 'local-build')
                    self.assertEqual(events[2][0:4], ('null', 'pr', 1, 3))
                    self.assertNotEqual(events[2][4], events[0][4])
                    self.assertTrue((args.out / 'benchmark-results.json').exists())
                else:
                    self.assertEqual(len(events), 1)

    def test_failed_refresh_preserves_every_planned_first_parent_history_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(tier='main', harness='d' * 40, config={'path': 'scripts/bench/suites.json'},
                provenance={}, push_before='a' * 40,
                runs=[dict(base=None, head=sha, submitted_head=sha) for sha in ('b' * 40, 'c' * 40)])
            args = argparse.Namespace(plan=root / 'plan.json', out=root / 'out',
                bundles=root / 'bundles', artifact_id=44, index_sha256='e' * 64)
            args.plan.write_text('{}')
            def prove(options):
                return dict(status='complete', reason=None, revisions={'head': options.head.name}, wall_time={})
            with patch.object(ci, 'load_plan', return_value=plan), \
                    patch.object(ci, 'calibration_phase', side_effect=ValueError('incomplete null')), \
                    patch.object(ci, 'verified_bundles', return_value={sha: root / sha for sha in ('b' * 40, 'c' * 40)}), \
                    patch.object(ci.bench, 'host_lock', side_effect=contextlib.nullcontext), \
                    patch.object(ci.bench, 'run', side_effect=prove):
                self.assertEqual(ci.execute(args), 0)
            self.assertTrue((args.out / 'commits' / ('b' * 40) / 'benchmark-results.json').exists())
            self.assertEqual(json.loads((args.out / 'benchmark-results.json').read_text())['status'], 'complete')
            self.assertEqual(len(json.loads((args.out / 'index.json').read_text())['documents']), 2)

    def test_only_main_calibration_dispatch_bypasses_ordinary_enablement(self):
        for tier, ref, enabled in [('calibration', 'refs/heads/main', True),
                                   ('main', 'refs/heads/main', False),
                                   ('calibration', 'refs/heads/other', False)]:
            with self.subTest(tier=tier, ref=ref), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                event = root / 'event.json'
                event.write_text(json.dumps({'inputs': {'tier': tier}}))
                def git_file(_, name):
                    if name == 'scripts/bench/suites.json':
                        return ci.bench.DEFAULT_CONFIG.read_bytes()
                    if name == 'Cargo.toml':
                        return b'[profile.release]\nlto="fat"\n'
                    return b'owned pinned source fixture'
                with patch.dict(os.environ, GITHUB_EVENT_PATH=str(event), GITHUB_EVENT_NAME='workflow_dispatch',
                        GITHUB_REF=ref, GITHUB_REPOSITORY='a16z/jolt', GITHUB_SHA='c' * 40,
                        JOLT_E2E_ENABLED='false', GITHUB_OUTPUT=str(root / 'outputs'),
                        GITHUB_STEP_SUMMARY=str(root / 'summary')), \
                        patch.object(ci, 'resolve', return_value=('d' * 40,
                            [dict(base=None, head='c' * 40, submitted_head='c' * 40)], None, None)), \
                        patch.object(ci, 'git_file', side_effect=git_file), \
                        patch.object(ci.bench, 'harness_files', return_value={'profile.rs': 'e' * 64}), \
                        patch.object(ci.bench, 'capture', return_value=''):
                    ci.plan(argparse.Namespace(config=ci.bench.DEFAULT_CONFIG, out=root / 'plan'))
                self.assertEqual(json.loads((root / 'plan/plan.json').read_text())['enabled'], enabled)
