import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.tests.test_e2e_publication import fixture

DIRECTORY = Path(__file__).parents[1] / 'bench'
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC = importlib.util.spec_from_file_location('e2e_ingest', DIRECTORY / 'ingest.py')
    ingest = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(ingest)
finally:
    sys.path.pop(0)


def zipped(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def inputs(tier='pr'):
    result, expected, config = fixture()
    spec = json.loads(config)
    if tier == 'calibration':
        result.update(kind='null')
        result['provenance'].update(event='workflow_dispatch', pr_number=None)
        result['revisions'].update(base='c' * 40, head='c' * 40, merge=None, harness='c' * 40)
        for build in result['builds']:
            build.update(revision='c' * 40, harness_revision='c' * 40, native_harness_revision='c' * 40)
        for sample in result['cases'][0]['samples']:
            sample['revision'] = 'c' * 40
    elif tier != 'pr':
        result.update(tier=tier, kind='history')
        result['provenance'].update(event='push' if tier == 'main' else 'schedule', pr_number=None)
        result['revisions'].update(base=None, head='c' * 40, merge=None)
        result['builds'] = result['builds'][1:]
        result['cases'][0]['samples'] = [sample for sample in result['cases'][0]['samples'] if sample['leg'] == 'head']
        for order, sample in enumerate(result['cases'][0]['samples']):
            sample.update(pair_index=None, order=order)
        spec['tiers'][tier] = spec['tiers'].pop('pr')
        config = json.dumps(spec).encode()
        result['suite']['config_sha256'] = hashlib.sha256(config).hexdigest()
        result['wall_time'].update(scope=ingest.ci.RUN_ONLY_SCOPE, build_s=0)
    plan = {'tier': tier, 'enabled': True, 'harness': result['revisions']['harness'], 'push_before': 'a' * 40,
            'push_after': 'c' * 40, 'provenance': result['provenance'],
            'config': {'path': 'scripts/bench/suites.json', 'sha256': hashlib.sha256(config).hexdigest()},
            'admission': {'label_event_id': 11, 'label_actor': 'maintainer', 'head_ref': 'candidate'},
            'runs': [{'base': 'a' * 40 if tier == 'pr' else None,
                      'head': 'c' * 40, 'submitted_head': ('b' if tier == 'pr' else 'c') * 40}],
            'expected_builds': {build['revision']: {key: value for key, value in build.items() if key != 'leg'}
                                for build in result['builds']}}
    dispatch = ingest.ci.dispatch_manifest(plan, 123, 1, artifact_id=44,
        index_sha256='2' * 64, digests={build['revision']: ingest.ci.manifest_digest(
            {key: value for key, value in build.items() if key != 'leg'}) for build in result['builds']})
    plan_raw = json.dumps(plan).encode()
    receipt = {'plan_sha256': hashlib.sha256(plan_raw).hexdigest()}
    result['artifacts']['plan_sha256'] = receipt['plan_sha256']
    if tier != 'pr':
        result['artifacts']['id'] = 44
    return result, plan, dispatch, config, receipt, plan_raw


class IngestTests(unittest.TestCase):
    def test_pr_comment_filters_strict_three_percent_in_either_metric_and_direction(self):
        result = inputs()[0]
        template = result['cases'].pop()
        for cores, seconds, rss in [(1, 104, 10000), (2, 96, 10000),
                                    (3, 100, 10400), (4, 100, 9600),
                                    (5, 103, 9700), (6, 97, 10300), (7, 103.001, 10000),
                                    (8, 0.97, 10000), (9, 1.03, 10000)]:
            case = copy.deepcopy(template)
            case['physical_cores'] = cores
            for sample in case['samples']:
                if sample['phase'] == 'warmup':
                    sample['metrics'].update(e2e_s=1000, peak_rss_bytes=1000000)
                else:
                    base_seconds = 1 if cores in (8, 9) else 100
                    sample['metrics'].update(e2e_s=base_seconds if sample['leg'] == 'base' else seconds,
                        peak_rss_bytes=10000 if sample['leg'] == 'base' else rss)
            result['cases'].append(case)
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.md'
            comment = ingest.render_pr_comment(result, summary, 'https://fixture.invalid/results')
            self.assertIn(f"Base: `{'a' * 40}`. Submitted head: `{'b' * 40}`. "
                          f"Measured merge: `{'c' * 40}`.", comment)
            for cores, values in [(1, '100.000000 / 104.000000 | +4.00%'),
                                  (2, '100.000000 / 96.000000 | -4.00%'),
                                  (3, '10000 / 10400 | +4.00%'),
                                  (4, '10000 / 9600 | -4.00%'),
                                  (7, '100.000000 / 103.001000 | +3.00%')]:
                row = next(line for line in comment.splitlines()
                           if line.startswith(f'| fibonacci / dory | {cores} |'))
                self.assertIn(values, row)
                full = next(line for line in summary.read_text().splitlines()
                            if line.startswith(f'| fibonacci / dory | {cores} |'))
                full_cells, short_cells = full.split('|'), row.split('|')
                self.assertEqual(short_cells[1:-1], [full_cells[i] for i in (1, 2, 3, 4, 6, 7)])
            for cores in (5, 6, 8, 9):
                self.assertNotIn(f'| fibonacci / dory | {cores} |', comment)

    def test_pr_comment_no_change_marker_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            comment = ingest.render_pr_comment(inputs()[0], Path(directory) / 'summary.md',
                'https://github.com/a16z/jolt/actions/runs/456/attempts/2')
        self.assertTrue(comment.startswith('<!-- jolt-e2e-benchmark-comment -->\n'))
        self.assertIn('[Full results](https://github.com/a16z/jolt/actions/runs/456/attempts/2)', comment)
        self.assertIn('[Benchmark history](https://a16z.github.io/jolt/dev/prover-history/)', comment)
        self.assertIn(f"Base: `{'a' * 40}`. Submitted head: `{'b' * 40}`. "
                      f"Measured merge: `{'c' * 40}`.", comment)
        self.assertIn('No change above 3%', comment)
        self.assertNotIn('| Workload', comment)

    def test_pr_comment_does_not_claim_no_change_for_unavailable_comparisons(self):
        for changes, exit_code in [({'status': 'failed'}, 0), ({'status': 'incomplete'}, 0), ({}, 1)]:
            with self.subTest(changes=changes, exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                result = dict(inputs()[0], **changes)
                comment = ingest.render_pr_comment(result, Path(directory) / 'summary.md',
                                                   'https://fixture.invalid/results', exit_code)
                self.assertIn('Comparison unavailable', comment)
                self.assertNotIn('No change', comment)
                self.assertIn(f"Base: `{'a' * 40}`. Submitted head: `{'b' * 40}`. "
                              f"Measured merge: `{'c' * 40}`.", comment)

    def test_hosted_validation_accepts_each_tier_and_refuses_result_or_build_substitution(self):
        for tier in ('pr', 'main', 'nightly', 'calibration'):
            result, plan, dispatch, config, receipt, _ = inputs(tier)
            files = {'benchmark-results.json': json.dumps(result).encode()}
            documents = ingest.validate_results(files, plan, dispatch, config, receipt)
            self.assertEqual(documents, [result])
            for key in ('provenance', 'builds', 'artifacts', 'cases'):
                altered = copy.deepcopy(result)
                if key == 'provenance':
                    altered[key]['run_attempt'] = '99'
                elif key == 'builds':
                    altered[key][0]['target_cpu'] = 'unplanned'
                elif key == 'artifacts':
                    altered[key]['plan_sha256'] = 'f' * 64
                else:
                    altered[key][0]['samples'][0]['metrics']['public_output'] = [999]
                with self.subTest(tier=tier, key=key), self.assertRaises(ValueError):
                    ingest.validate_results({'benchmark-results.json': json.dumps(altered).encode()},
                                            plan, dispatch, config, receipt)
        bad_index = dict(files, **{'index.json': b'{"documents": []}'})
        main = inputs('main')
        bad_index['benchmark-results.json'] = json.dumps(main[0]).encode()
        with self.assertRaisesRegex(ValueError, 'index differs'):
            ingest.validate_results(bad_index, *main[1:5])

    def test_plan_run_artifacts_are_listed_and_bound_to_the_original_pr_head_not_main_sha(self):
        result, plan, dispatch, config, receipt, plan_raw = inputs()
        actual = {'id': 123, 'run_attempt': 1, 'repository': {'full_name': 'a16z/jolt'},
            'head_repository': {'full_name': 'a16z/jolt'}, 'name': ingest.bench.WORKFLOW,
            'path': ingest.bench.WORKFLOW_FILE, 'status': 'completed', 'conclusion': 'success',
            'event': 'pull_request_target', 'head_branch': 'candidate', 'head_sha': 'b' * 40,
            'pull_requests': [{'number': 7, 'base': {'ref': 'main'}}]}
        artifacts = [{'id': value, 'name': f'{prefix}-123-1', 'expired': False,
                      'workflow_run': {'id': 123}, 'size_in_bytes': 100}
                     for value, prefix in [(42, 'jolt-e2e-plan'), (43, 'jolt-e2e-dispatch')]]
        archives = {42: zipped({'plan.json': plan_raw}),
                    43: zipped({'dispatch.json': json.dumps(dispatch)})}
        def github(path):
            if path == 'actions/runs/123/attempts/1':
                return actual
            if '/artifacts?' in path:
                return {'artifacts': artifacts}
            if path == 'commits/' + 'c' * 40:
                return {'parents': [{'sha': 'a' * 40}, {'sha': 'b' * 40}]}
            if path == 'contents/scripts/bench/suites.json?ref=' + 'd' * 40:
                return {'type': 'file', 'encoding': 'base64', 'content': base64.b64encode(config).decode()}
            raise AssertionError(path)
        def download(url, cap, **kwargs):
            return archives[int(url.split('/')[-2])]
        with patch.object(ingest.ci, 'github', side_effect=github), \
                patch.object(ingest, 'download', side_effect=download):
            checked, request, committed, bound = ingest.trusted_plan(123, 1)
            self.assertEqual((checked, request, committed), (plan, dispatch, config))
            self.assertEqual(bound['plan_sha256'], receipt['plan_sha256'])
            prs = actual['pull_requests']
            actual['pull_requests'] = []
            self.assertEqual(ingest.trusted_plan(123, 1)[0], plan)
            actual['pull_requests'] = prs
            actual['pull_requests'][0]['base']['ref'] = 'other'
            with self.assertRaises(ValueError):
                ingest.trusted_plan(123, 1)
            actual['pull_requests'][0]['base']['ref'] = 'main'
            artifacts.append(copy.deepcopy(artifacts[0]))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                ingest.trusted_plan(123, 1)

    def test_main_ingest_refuses_divergence_and_foreign_build_artifacts(self):
        result, plan, dispatch, config, receipt, _ = inputs('main')
        actual = {'id': 123, 'run_attempt': 1, 'repository': {'full_name': 'a16z/jolt'},
            'head_repository': {'full_name': 'a16z/jolt'}, 'name': ingest.bench.WORKFLOW,
            'path': ingest.bench.WORKFLOW_FILE, 'status': 'completed', 'conclusion': 'success',
            'event': 'push', 'head_branch': 'main', 'head_sha': 'c' * 40}
        comparison = {'status': 'ahead'}
        builds = {'name': 'jolt-e2e-builds-123-1', 'expired': False, 'workflow_run': {'id': 123}}
        def github(path):
            if path.startswith('actions/runs/'):
                return actual
            if path.startswith('compare/'):
                return comparison
            if path.startswith('actions/artifacts/'):
                return builds
            return {'type': 'file', 'encoding': 'base64', 'content': base64.b64encode(config).decode()}
        def artifact(run_id, attempt, prefix, filename):
            return (plan, json.dumps(plan).encode(), 42) if filename == 'plan.json' else (
                dispatch, json.dumps(dispatch).encode(), 43)
        with patch.object(ingest.ci, 'github', side_effect=github), \
                patch.object(ingest, 'artifact_json', side_effect=artifact):
            ingest.trusted_plan(123, 1)
            for target, key, value in [(comparison, 'status', 'diverged'),
                    (builds, 'workflow_run', {'id': 999}), (builds, 'name', 'other'),
                    (actual, 'head_branch', 'candidate'), (actual, 'head_sha', 'f' * 40)]:
                original = target[key]
                target[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ingest.trusted_plan(123, 1)
                target[key] = original

    def test_digest_mismatch_and_untrusted_actor_refuse_before_results_are_written(self):
        result, plan, dispatch, config, receipt, _ = inputs()
        raw = zipped({'benchmark-results.json': json.dumps(result)})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out'
            argv = ['ingest.py', '--job-id', '123-1', '--plan-run-id', '123', '--plan-run-attempt', '1',
                '--result-url', 'https://fixture.invalid/github-bench/' + '0' * 64 + '.zip',
                '--result-sha256', '0' * 64, '--executor-exit-code', '0', '--out', str(out)]
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, GITHUB_ACTOR=ingest.APP,
                    GITHUB_REF='refs/heads/main'), patch.object(ingest, 'trusted_plan',
                    return_value=(plan, dispatch, config, receipt)), patch.object(ingest, 'download', return_value=raw):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    ingest.main()
                self.assertFalse(out.exists())
                for env in ({'GITHUB_ACTOR': 'someone'}, {'GITHUB_REF': 'refs/heads/other'}):
                    with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'App on main'):
                        ingest.main()

    def test_ingest_publishes_canonical_receipt_and_bounded_check_output(self):
        result, plan, dispatch, config, receipt, _ = inputs()
        raw = zipped({'benchmark-results.json': json.dumps(result), 'untrusted.py': 'raise SystemExit(99)'})
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out'
            argv = ['ingest.py', '--job-id', '123-1', '--plan-run-id', '123', '--plan-run-attempt', '1',
                '--result-url', f'https://fixture.invalid/github-bench/{digest}.zip', '--result-sha256', digest,
                '--executor-exit-code', '0', '--out', str(out)]
            job_summary, outputs = Path(directory) / 'job-summary.md', Path(directory) / 'outputs'
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, GITHUB_ACTOR=ingest.APP,
                    GITHUB_REF='refs/heads/main', GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='2',
                    GITHUB_STEP_SUMMARY=str(job_summary), GITHUB_OUTPUT=str(outputs)), \
                    patch.object(ingest, 'trusted_plan', return_value=(plan, dispatch, config, receipt)), \
                    patch.object(ingest, 'download', return_value=raw):
                ingest.main()
            self.assertEqual(json.loads((out / 'result/benchmark-results.json').read_text()), result)
            self.assertEqual(json.loads((out / 'result/ingest.json').read_text())['ingest_run_id'], 456)
            check = json.loads((out / 'check/check.json').read_text())
            self.assertEqual(check['job_id'], '123-1')
            self.assertEqual(check['conclusion'], 'neutral')
            self.assertLessEqual(len(check['summary']), 60000)
            self.assertEqual(job_summary.read_text(), check['summary'] + '\n')
            self.assertEqual(outputs.read_text(), 'pr_number=7\n')
            self.assertIn('No change above 3%', (out / 'result/pr-comment.md').read_text())

    def test_check_conclusions_distinguish_execution_failure_partial_and_complete_results(self):
        result = inputs('main')[0]
        for documents, code, conclusion in [([result], 0, 'success'), ([result], 7, 'failure'),
                ([dict(result, status='failed')], 0, 'failure'), ([None, result], 0, 'neutral'),
                ([dict(result, status='incomplete')], 0, 'neutral'), ([inputs()[0]], 0, 'neutral')]:
            with self.subTest(code=code, conclusion=conclusion):
                check = ingest.check_result('123-1', documents, code, 'x' * 70000)
                self.assertEqual(check['conclusion'], conclusion)
                self.assertLessEqual(len(check['title']), 200)
                self.assertEqual(len(check['summary']), 60000)

    def test_a_finalized_missing_target_result_is_valid_partial_data_for_each_tier(self):
        for tier in ('pr', 'main', 'nightly', 'calibration'):
            result, plan, dispatch, config, receipt, plan_raw = inputs(tier)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / plan['config']['path']
                config_path.parent.mkdir(parents=True)
                config_path.write_bytes(config)
                plan_path, dispatch_path = root / 'plan.json', root / 'dispatch.json'
                plan_path.write_bytes(plan_raw)
                dispatch_path.write_text(json.dumps(dispatch))
                options = type('Options', (), {'plan': plan_path, 'dispatch': dispatch_path, 'out': root / 'out'})()
                with patch.object(ingest.bench, 'ROOT', root), \
                        patch.object(ingest.bench, 'host_snapshot', return_value={'topology': []}):
                    ingest.ci.finalize(options)
                files = {'benchmark-results.json': (options.out / 'benchmark-results.json').read_bytes()}
                checked = ingest.validate_results(files, plan, dispatch, config, receipt)
                self.assertEqual(checked[0]['status'], 'incomplete')
                self.assertEqual(checked[0]['builds'], [])
                self.assertEqual(ingest.check_result('123-1', checked, 1, 'failed')['conclusion'], 'failure')

    def test_download_sends_descriptive_user_agent(self):
        for url, authenticated in [('https://fixture.invalid/result.zip', False),
                ('https://api.github.com/repos/a16z/jolt/actions/artifacts/42/zip', True)]:
            with self.subTest(url=url), patch.dict(os.environ, GH_TOKEN='fixture-token'), \
                    patch.object(ingest.urllib.request, 'build_opener') as opener:
                opener.return_value.open.return_value = io.BytesIO(b'owned result')
                self.assertEqual(ingest.download(url, 100, authenticated=authenticated), b'owned result')
                request = opener.return_value.open.call_args.args[0]
                self.assertEqual(request.get_header('User-agent'),
                                 'jolt-e2e-ingest/1 (+https://github.com/a16z/jolt)')
                redirected = ingest.DownloadRedirect().redirect_request(request, None, 302, 'Found', {},
                                                                        'https://storage.invalid/result.zip')
                self.assertEqual(redirected.get_header('User-agent'), request.get_header('User-agent'))
                self.assertFalse(redirected.has_header('Authorization'))

    def test_untrusted_zip_paths_and_download_caps_are_enforced(self):
        self.assertEqual(ingest.result_files(zipped({'builds/fixture/build.log': b'owned log'})),
                         {'builds/fixture/build.log': b'owned log'})
        for filename in ('../outside', '/outside', './benchmark-results.json', 'a//b',
                         'builds/secret', 'case/runtime-cache/secret', 'a\\b'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                ingest.result_files(zipped({filename: b'fixture'}))
        response = io.BytesIO(b'12345')
        with patch.object(ingest.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(ValueError, 'size cap'):
                ingest.download('https://fixture.invalid/result.zip', 4)


if __name__ == '__main__':
    unittest.main()
