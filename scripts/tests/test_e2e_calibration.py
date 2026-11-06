import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.tests.test_e2e_ingest import ingest, inputs, zipped

calibration = ingest.calibration
publication = ingest.publication


def null_fixture():
    repetitions = 10
    result, plan, dispatch, config, receipt, _ = inputs('main')
    spec = json.loads(config)
    template = spec['tiers']['main']['cases'][0]
    spec['tiers']['pr'] = dict(warmups=1, repetitions=3,
        cases=[dict(copy.deepcopy(template), id=f'cell-{index}') for index in range(17)])
    config = json.dumps(spec).encode()
    result.update(tier='pr', kind='null', started_at='2026-11-05T10:00:00Z',
                  finished_at='2026-11-05T10:10:00Z')
    result['revisions'].update(base='c' * 40, head='c' * 40)
    result['suite'].update(config_sha256=hashlib.sha256(config).hexdigest(), warmups=1, repetitions=repetitions)
    host = dict(id='host', kernel='kernel', microcode='microcode',
        control_plane=dict(version='pika', cli_sha256='0' * 64, node_sha256='1' * 64),
        settings=dict(boost='1', smt='on', governor={'2': 'performance'},
                      transparent_huge_pages='[never]', swap_total_bytes=0),
        topology=[dict(cpu=2, core=0, socket=0, online=True)])
    result.update(host=host, host_after=copy.deepcopy(host),
        runtime_cache={'dory': 'initially empty, suite-local, shared by both legs and all fresh processes'})
    build = result['builds'][0]
    build.update(toolchain='rustc', cargo_version='cargo', target_triple='x86_64-unknown-linux-gnu',
        target_features=['sse2'], profile_config={'lto': 'fat'}, cargo_lock_sha256='0' * 64, guest_builder_rustflags='',
        runner={'path': 'scripts/bench/run.py', 'sha256': '2' * 64},
        calibration_helper={'path': 'scripts/bench/calibration.py', 'sha256': '3' * 64},
        suite_config_sha256=hashlib.sha256(config).hexdigest(),
        akita_schedules=[dict(path='schedule.aks', sha256='4' * 64)],
        binaries={'dory': dict(path='prover', sha256='5' * 64,
            normalized=dict(method=calibration.NORMALIZATION, sha256='6' * 64))})
    build['guests']['fibonacci']['elf'] = dict(path='guest.elf', sha256='7' * 64)
    result['builds'] = [dict(copy.deepcopy(build), leg=leg) for leg in ('base', 'head')]
    sample = result['cases'][0]['samples'][0]
    result['cases'] = []
    for index, case in enumerate(spec['tiers']['pr']['cases']):
        case = dict(copy.deepcopy(case), status='complete', cpu_ids=[2], reason=None,
                    samples=[dict(copy.deepcopy(sample), **dict(item, order=index * 2 * (repetitions + 1) + item['order']))
                        for item in ingest.bench.sample_order(True, 1, repetitions)])
        case['input']['sha256'] = '1' * 64
        result['cases'].append(case)
    plan['config']['sha256'] = hashlib.sha256(config).hexdigest()
    plan['expected_builds'] = {'c' * 40: {k: v for k, v in build.items() if k != 'leg'}}
    dispatch['manifest_digests'] = {'c' * 40: ingest.ci.manifest_digest(plan['expected_builds']['c' * 40])}
    receipt.update(schema=1, job_id='123-1', plan_run_id=123, plan_run_attempt=1,
        ingest_run_id=456, ingest_run_attempt=1, result_sha256='8' * 64,
        dispatch_sha256='9' * 64, plan_artifact_id=42, dispatch_artifact_id=43, executor_exit_code=0)
    return result, plan, dispatch, config, receipt


def evidence(result):
    raw = json.dumps(result).encode()
    root = dict(result, calibration_evidence={'raw': {
        'path': 'calibration/raw/benchmark-results.json', 'sha256': hashlib.sha256(raw).hexdigest()}})
    return root, {'benchmark-results.json': json.dumps(root).encode(),
                  'calibration/raw/benchmark-results.json': raw}


class CalibrationTests(unittest.TestCase):
    def test_reducer_matches_frozen_symmetric_median_window_vectors(self):
        result, *_ = null_fixture()
        case = result['cases'][0]
        for sample in case['samples']:
            if sample['leg'] == 'head':
                sample['metrics'].update(e2e_s=3.12, peak_rss_bytes=5120)
        bands = calibration.envelopes(case)
        self.assertEqual(bands['e2e_s']['threshold_pct'], 6.0)
        self.assertEqual(bands['peak_rss_bytes']['threshold_pct'], 37.5)
        series = [90, 90, 100, 110, 110, 110, 100, 100, 100, 100]
        for sample in case['samples']:
            sample['metrics']['e2e_s'] = (series[sample['repetition']] if sample['leg'] == 'head' else 100)
        self.assertEqual(calibration.envelopes(case)['e2e_s']['threshold_pct'], 17.0)
        for sample in case['samples']:
            if sample['leg'] == 'base':
                sample['metrics']['e2e_s'] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            calibration.envelopes(case)

    def test_normalization_admits_only_nonwritable_nonexecutable_metadata(self):
        if not all(shutil.which(tool) for tool in ('cc', 'objcopy', 'readelf')):
            self.skipTest('ELF fixture tools unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'prover'
            subprocess.run(['cc', '-x', 'c', '-', '-o', str(binary)], input=b'int main(void) { return 0; }', check=True)
            original = binary.read_bytes()
            identity = calibration.normalized(binary)
            self.assertIsNotNone(identity)
            metadata = root / 'metadata'
            metadata.write_bytes(b'other compiler label')
            changed = root / 'changed'
            subprocess.run(['objcopy', '--update-section', f'.comment={metadata}', str(binary), str(changed)], check=True)
            self.assertEqual(calibration.normalized(changed), identity)
            for section, flags in (('.comment', 'alloc,data'), ('.note.gnu.build-id', 'alloc,code')):
                subprocess.run(['objcopy', '--set-section-flags', f'{section}={flags}', str(binary), str(changed)],
                               check=True, capture_output=True)
                self.assertIsNone(calibration.normalized(changed))
            metadata.write_bytes(b'changed code bytes')
            subprocess.run(['objcopy', '--update-section', f'.text={metadata}', str(binary), str(changed)], check=True)
            self.assertNotEqual(calibration.normalized(changed), identity)
            subprocess.run(['objcopy', '--add-section', f'.calibration-data={metadata}', str(binary), str(changed)], check=True)
            self.assertNotEqual(calibration.normalized(changed), identity)
            self.assertEqual(binary.read_bytes(), original)

    def test_exact_key_binds_source_build_host_geometry_inputs_and_all_base_rows(self):
        result, *_ = null_fixture()
        self.assertEqual(calibration.encoded({'b': 2, 'a': 1}), b'{"a":1,"b":2}')
        key = calibration.cell_key(result, result['cases'][0])
        mutations = [lambda r: r['builds'][0].update(revision='a' * 40),
            lambda r: r['builds'][0]['calibration_helper'].update(sha256='e' * 64),
            lambda r: r['builds'][0]['harness_files'].update(extra='f' * 64),
            lambda r: r['cases'][0].update(cpu_ids=[3]),
            lambda r: r['cases'][0]['input'].update(seed=1),
            lambda r: r['host_after']['settings']['governor'].update({'2': 'powersave'}),
            lambda r: r['cases'][0]['samples'][2]['metrics'].update(trace_rows=5),
            lambda r: r['builds'][0]['binaries']['dory'].update(normalized=None)]
        for mutate in mutations:
            changed = copy.deepcopy(result)
            mutate(changed)
            try:
                observed = calibration.cell_key(changed, changed['cases'][0])
            except (ValueError, KeyError):
                continue
            self.assertNotEqual(observed, key)
        changed = copy.deepcopy(result)
        changed['builds'][0].update(host={'hostname': 'other builder'}, native_tools={'cc': 'different label'},
                                   harness_revision='e' * 40, finished_at='later')
        changed['host']['settings']['swap_free_bytes'] = 100
        self.assertEqual(calibration.cell_key(changed, changed['cases'][0]), key)

    def test_sidecar_validation_recomputes_bands_and_refuses_substitution(self):
        result, plan, dispatch, config, receipt = null_fixture()
        root, files = evidence(result)
        reference = ingest.calibration_reference(files, root, plan, dispatch, config, receipt)
        self.assertEqual(len(reference['cells']), 17)
        self.assertTrue(all(c['metrics']['e2e_s']['threshold_pct'] == 3 for c in reference['cells']))
        for mutate in (lambda r: r['cases'][0]['samples'][0].update(exit_code=7),
                       lambda r: r['cases'][0]['samples'].reverse(),
                       lambda r: r['builds'][1]['binaries']['dory'].update(sha256='f' * 64)):
            changed = copy.deepcopy(result)
            mutate(changed)
            changed_root, changed_files = evidence(changed)
            with self.assertRaises(ValueError):
                ingest.calibration_reference(changed_files, changed_root, plan, dispatch, config, receipt)
        files['calibration/raw/benchmark-results.json'] += b' '
        with self.assertRaisesRegex(ValueError, 'descriptor'):
            ingest.calibration_reference(files, root, plan, dispatch, config, receipt)

    def test_reference_selection_uses_newest_valid_exact_base_and_48_hour_expiry(self):
        result, *_ = null_fixture()
        raw = json.dumps(result).encode()
        source = dict(plan_run_id=123, plan_run_attempt=1, ingest_run_id=456, ingest_run_attempt=1)
        reference = calibration.reference(raw, result, source)
        probe = copy.deepcopy(result)
        probe.update(started_at='2026-11-05T11:00:00Z', finished_at='2026-11-05T11:10:00Z')
        probe['suite']['repetitions'] = 3
        older = copy.deepcopy(reference)
        older.update(finished_at='2026-11-05T09:00:00Z')
        older['cells'][0]['metrics']['e2e_s']['threshold_pct'] = 99
        probe['external_references'] = {'references': [older, reference]}
        bands, reason = calibration.for_case(probe, probe['cases'][0])
        self.assertIsNone(reason)
        self.assertEqual(bands['e2e_s']['threshold_pct'], 3)
        for change in (lambda r: r.update(finished_at='2026-11-07T10:10:01Z'),
                       lambda r: r['builds'][0].update(revision='a' * 40),
                       lambda r: r['builds'][0]['runner'].update(sha256='f' * 64),
                       lambda r: r.update(started_at='2026-11-05T08:00:00Z')):
            changed = copy.deepcopy(probe)
            change(changed)
            self.assertIsNone(calibration.for_case(changed, changed['cases'][0])[0])

    def test_missing_or_unauthenticated_catalog_preserves_raw_measurements(self):
        result, *_ = null_fixture()
        before = copy.deepcopy(result)
        with patch.object(publication.ci, 'github', side_effect=OSError('catalog unavailable')):
            rendered = publication.resolve_references(result)
        self.assertEqual(result, before)
        self.assertEqual(rendered['cases'], before['cases'])
        self.assertIn('catalog unavailable', rendered['external_references']['reason'])

    def test_canonical_reference_refuses_wrong_actor_attempt_and_artifact_identity(self):
        actual = dict(id=456, run_attempt=1, repository={'full_name': 'a16z/jolt'},
            head_repository={'full_name': 'a16z/jolt'}, name='e2e-bench ingest 123-1',
            path='.github/workflows/e2e-bench-ingest.yml', event='workflow_dispatch', head_branch='main',
            status='completed', conclusion='success', actor={'login': ingest.APP}, head_sha='a' * 40)
        artifact = dict(id=99, name='jolt-e2e-456-1', expired=False, size_in_bytes=1,
                        workflow_run=dict(id=456, head_sha='a' * 40))
        for value, field, replacement in ((actual, 'actor', {'login': 'intruder'}),
                (actual, 'run_attempt', 2), (actual, 'head_branch', 'candidate'),
                (artifact, 'name', 'jolt-e2e-456-2'), (artifact, 'id', 100), (artifact, 'expired', True)):
            changed = copy.deepcopy(value)
            changed[field] = replacement
            replies = [changed if value is actual else actual, changed if value is artifact else artifact]
            with patch.dict(sys.modules, {'ingest': ingest}), patch.object(publication.ci, 'github', side_effect=replies), \
                    patch.object(ingest, 'download', side_effect=AssertionError('must refuse before downloading')):
                with self.assertRaises(ValueError):
                    publication.canonical_artifact(99, 456, 1)

    def test_catalog_discovers_but_only_authenticated_recomputed_artifact_supplies_bands(self):
        raw_result, plan, dispatch, config, receipt = null_fixture()
        raw = json.dumps(raw_result).encode()
        main_result, *_ = inputs('main')
        main_result['suite']['config_sha256'] = hashlib.sha256(config).hexdigest()
        main_result['builds'] = [dict(copy.deepcopy(raw_result['builds'][0]), leg='head')]
        root, files = evidence(raw_result)
        root = dict(main_result, calibration_evidence=root['calibration_evidence'])
        files['benchmark-results.json'] = json.dumps(root).encode()
        reference = ingest.calibration_reference(files, root, plan, dispatch, config, receipt)
        files.update({'ingest.json': json.dumps(receipt).encode(),
                      'calibration/reference.json': calibration.encoded(reference)})
        actual = dict(id=456, run_attempt=1, repository={'full_name': 'a16z/jolt'},
            head_repository={'full_name': 'a16z/jolt'}, name='e2e-bench ingest 123-1',
            path='.github/workflows/e2e-bench-ingest.yml', event='workflow_dispatch', head_branch='main',
            status='completed', conclusion='success', actor={'login': ingest.APP}, head_sha='a' * 40,
            display_title='e2e-bench ingest 123-1')
        artifact = dict(id=99, name='jolt-e2e-456-1', expired=False, size_in_bytes=1,
                        workflow_run=dict(id=456, head_sha='a' * 40))
        catalog = {'schema': 1, 'references': [dict(artifact_id=99, ingest_run_id=456, ingest_run_attempt=1,
            sha256=calibration.digest(reference), raw_sha256=reference['raw_sha256'],
            key_sha256s=[cell['key_sha256'] for cell in reference['cells']])]}
        probe = copy.deepcopy(raw_result)
        probe['suite']['repetitions'] = 3
        for case in probe['cases']:
            case['samples'] = case['samples'][:8]
        probe.update(started_at='2026-11-05T11:00:00Z', finished_at='2026-11-05T11:10:00Z')
        original = copy.deepcopy(probe)
        def github(route, **kwargs):
            if route == 'commits/gh-pages':
                return {'sha': 'e' * 40}
            if route == 'contents/dev/prover-history/calibration/index.json?ref=' + 'e' * 40:
                self.assertEqual(kwargs, {'media': 'application/vnd.github.raw+json'})
                return copy.deepcopy(catalog)
            if route == 'actions/runs/456/attempts/1':
                return actual
            if route == 'actions/artifacts/99':
                return artifact
            raise AssertionError(route)
        with patch.dict(sys.modules, {'ingest': ingest}), patch.object(publication.ci, 'github', side_effect=github), \
                patch.object(ingest, 'download', side_effect=lambda *args, **kwargs: zipped(files)), \
                patch.object(ingest, 'trusted_plan', return_value=(plan, dispatch, config, {
                    name: receipt[name] for name in ('plan_sha256', 'dispatch_sha256', 'plan_artifact_id', 'dispatch_artifact_id')})):
            rendered = publication.resolve_references(probe)
            self.assertEqual(len(rendered['external_references']['references']), 1)
            with self.assertRaisesRegex(ValueError, 'target cannot supply'):
                publication.validate(dict(main_result, external_references={}), {}, config)
            self.assertIsNone(calibration.for_case(rendered, rendered['cases'][0])[1])
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'summary.md'
                ingest.bench.render_comment(rendered, output)
                self.assertIn('3.00%', output.read_text())
                self.assertEqual(ingest.check_result('123-1', [rendered], 0, output.read_text())['conclusion'], 'success')
                self.assertNotIn('pass/fail verdict', output.read_text())
            files['calibration/reference.json'] = b'{}'
            rejected = publication.resolve_references(probe)
            self.assertEqual(rejected['external_references']['references'], [])
            self.assertEqual(ingest.check_result('123-1', [rejected], 0, '')['conclusion'], 'neutral')
            self.assertEqual(rejected['cases'], original['cases'])
        self.assertEqual(probe, original)

    def test_verify_calibration_requires_a_bootstrap_and_all_17_resolved_rendered_cells(self):
        result, *_ = null_fixture()
        reference = calibration.reference(json.dumps(result).encode(), result, dict(tier='calibration',
            plan_run_id=123, plan_run_attempt=1, ingest_run_id=456, ingest_run_attempt=1))
        probe = copy.deepcopy(result)
        probe['suite']['repetitions'] = 3
        for case in probe['cases']:
            case['samples'] = case['samples'][:8]
        probe.update(started_at='2026-11-05T11:00:00Z', finished_at='2026-11-05T11:10:00Z',
                     external_references={'references': [reference], 'catalog_revision': 'a' * 40})
        files = {'calibration/probe/benchmark-results.json': json.dumps(probe).encode()}
        with patch.object(sys, 'argv', ['publication.py', 'verify-calibration', '--ingest-run-id', '456',
                                       '--ingest-run-attempt', '1']), \
                patch.object(publication.ci, 'github', return_value={'artifacts': [dict(id=99, name='jolt-e2e-456-1')]}), \
                patch.object(publication, 'canonical_artifact', return_value=(files, reference)), \
                patch.object(publication, 'resolve_references', return_value=probe), \
                patch.object(sys, 'stdout', new_callable=io.StringIO) as output:
            publication.main()
            self.assertEqual(json.loads(output.getvalue())['resolved_cells'], 17)
            probe['cases'].pop()
            with self.assertRaisesRegex(ValueError, 'probe references unavailable'):
                publication.main()
            reference['source']['tier'] = 'main'
            with self.assertRaisesRegex(ValueError, 'bootstrap calibration reference'):
                publication.main()

    def test_bootstrap_check_uses_reference_or_unavailable_outcome(self):
        raw_result, plan, dispatch, config, receipt = null_fixture()
        plan['tier'] = dispatch['tier'] = 'calibration'
        plan['provenance']['event'] = raw_result['provenance']['event'] = 'workflow_dispatch'
        raw = json.dumps(raw_result).encode()
        probe = copy.deepcopy(raw_result)
        probe['suite']['repetitions'] = 3
        probe['wall_time']['scope'] = ingest.ci.LOCAL_SCOPE
        probe.update(started_at='2026-11-05T11:00:00Z', finished_at='2026-11-05T11:10:00Z')
        for case in probe['cases']:
            case['samples'] = case['samples'][:8]
        for outcome, code in [('reference', 0), ('missing', 0), ('partial-probe', 0), ('executor-failed', 7)]:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                measured = copy.deepcopy(probe)
                if outcome == 'partial-probe':
                    measured['status'] = 'incomplete'
                    measured['cases'][0].update(status='missing', samples=[])
                probe_raw = json.dumps(measured).encode()
                descriptor = {'probe': dict(path='calibration/probe/benchmark-results.json',
                                           sha256=hashlib.sha256(probe_raw).hexdigest())}
                if outcome != 'missing':
                    descriptor['raw'] = dict(path='calibration/raw/benchmark-results.json',
                                             sha256=hashlib.sha256(raw).hexdigest())
                root = dict(measured, calibration_evidence=descriptor)
                files = {'benchmark-results.json': json.dumps(root),
                         'calibration/raw/benchmark-results.json': raw,
                         'calibration/probe/benchmark-results.json': probe_raw}
                archive = zipped(files)
                digest = hashlib.sha256(archive).hexdigest()
                out = Path(directory) / 'ingested'
                argv = ['ingest.py', '--job-id', '123-1', '--plan-run-id', '123', '--plan-run-attempt', '1',
                    '--result-url', f'https://fixture.invalid/{digest}.zip', '--result-sha256', digest,
                    '--executor-exit-code', str(code), '--out', str(out)]
                with patch.object(sys, 'argv', argv), patch.dict(os.environ, GITHUB_ACTOR=ingest.APP,
                        GITHUB_REF='refs/heads/main', GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='1'), \
                        patch.object(ingest, 'trusted_plan', return_value=(plan, dispatch, config, copy.deepcopy(receipt))), \
                        patch.object(ingest, 'download', return_value=archive):
                    ingest.main()
                check = json.loads((out / 'check/check.json').read_text())
                self.assertEqual(check['conclusion'], 'success' if outcome == 'reference' else 'failure')
                if outcome in ('reference', 'executor-failed'):
                    payload = (out / 'result/calibration/reference.json').read_bytes()
                    self.assertIn(hashlib.sha256(payload).hexdigest(), check['summary'])
                    if code:
                        self.assertIn('executor exited 7', check['summary'])
                else:
                    reason = json.loads((out / 'result/calibration/unavailable.json').read_text())['reason']
                    self.assertIn(reason, check['summary'])
                    self.assertFalse((out / 'result/calibration/reference.json').exists())
