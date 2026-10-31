import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
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
    def test_fork_request_is_rejected_before_checkout_or_compilation(self):
        request = {'name': ci.REQUEST_WORKFLOW, 'path': ci.REQUEST_FILE,
                   'event': 'pull_request', 'conclusion': 'success',
                   'head_repository': {'full_name': 'outside/jolt'}}
        with patch.object(ci, 'github', return_value=request), \
                patch.object(ci.bench, 'git_revision', return_value='a' * 40), \
                patch.object(ci.bench, 'capture') as capture:
            with self.assertRaisesRegex(ValueError, 'same-repository'):
                ci.resolve('workflow_run', {'workflow_run': {'id': 9}}, 'a16z/jolt', 'a' * 40)
        capture.assert_not_called()

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
            plan = {'enabled': True, 'tier': 'main', 'harness': 'c' * 40, 'target_cpu': 'znver4',
                    'push_before': 'a' * 40, 'provenance': {}, 'runs': [
                        {'base': None, 'head': 'b' * 40, 'submitted_head': 'b' * 40},
                        {'base': None, 'head': 'c' * 40, 'submitted_head': 'c' * 40}]}
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            args = argparse.Namespace(plan=plan_path, out=root / 'artifact')
            def suite(options):
                ci.bench.write_json(options.out / 'benchmark-results.json',
                                    {'fixture_revision': options.head_revision})
                return {'status': 'complete'}
            with patch.object(ci.bench, 'git_revision', return_value='c' * 40), \
                    patch.object(ci.bench, 'capture'), patch.object(ci.bench, 'host_lock',
                    side_effect=contextlib.nullcontext), patch.object(ci.bench, 'suite', side_effect=suite):
                self.assertEqual(ci.execute(args), 0)
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
            self.assertEqual(len(result['cases']), 6)
            self.assertTrue(all(c['status'] == 'missing' and not c['samples'] for c in result['cases']))
            self.assertIsNone(result['wall_time']['total_s'])

    def test_interrupted_batch_closes_partial_documents_and_keeps_missing_tip_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {'tier': 'main', 'harness': 'c' * 40, 'provenance': {}, 'runs': [
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


if __name__ == '__main__':
    unittest.main()
