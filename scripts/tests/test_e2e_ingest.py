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
    if tier != 'pr':
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
    plan = {'tier': tier, 'enabled': True, 'harness': 'd' * 40, 'push_before': 'a' * 40,
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
    def test_hosted_validation_accepts_each_tier_and_refuses_result_or_build_substitution(self):
        for tier in ('pr', 'main', 'nightly'):
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
            actual['pull_requests'][0]['base']['ref'] = 'other'
            with self.assertRaises(ValueError):
                ingest.trusted_plan(123, 1)
            actual['pull_requests'][0]['base']['ref'] = 'main'
            artifacts.append(copy.deepcopy(artifacts[0]))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                ingest.trusted_plan(123, 1)

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
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, GITHUB_ACTOR=ingest.APP,
                    GITHUB_REF='refs/heads/main', GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='2'), \
                    patch.object(ingest, 'trusted_plan', return_value=(plan, dispatch, config, receipt)), \
                    patch.object(ingest, 'download', return_value=raw):
                ingest.main()
            self.assertEqual(json.loads((out / 'result/benchmark-results.json').read_text()), result)
            self.assertEqual(json.loads((out / 'result/ingest.json').read_text())['ingest_run_id'], 456)
            check = json.loads((out / 'check/check.json').read_text())
            self.assertEqual(check['job_id'], '123-1')
            self.assertEqual(check['conclusion'], 'neutral')
            self.assertLessEqual(len(check['summary']), 60000)

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
        for tier in ('pr', 'main', 'nightly'):
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

    def test_untrusted_zip_paths_and_download_caps_are_enforced(self):
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
