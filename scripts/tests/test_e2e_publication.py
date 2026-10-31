import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).parents[1] / 'bench'
sys.path.insert(0, str(DIRECTORY))
try:
    SPEC = importlib.util.spec_from_file_location('e2e_publication', DIRECTORY / 'publication.py')
    publication = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(publication)
finally:
    sys.path.pop(0)


def fixture():
    case = {'id': 'fibonacci-dory-c1-s3-n1-v1', 'workload': 'fibonacci', 'pcs': 'dory',
            'backend': 'optimized', 'physical_cores': 1, 'scale_log2': 3,
            'input': {'parameters': {'operations': 1}, 'seed': None}}
    config = json.dumps({'schema_version': 1, 'id': 'fixture', 'metric_contract': 1,
                         'tiers': {'pr': {'warmups': 1, 'repetitions': 3, 'cases': [case]}}}).encode()
    expected = {'revisions': {'base': 'a' * 40, 'head': 'b' * 40, 'merge': 'c' * 40, 'harness': 'd' * 40},
                'provenance': {'workflow': publication.bench.WORKFLOW, 'workflow_file': publication.bench.WORKFLOW_FILE,
                               'run_id': '123', 'run_attempt': '1', 'event': 'workflow_run',
                               'ref': 'refs/heads/main', 'pr_number': 7, 'request_run_id': 99}}
    builds = [{'leg': leg, 'revision': revision, 'harness_revision': 'd' * 40, 'status': 'complete',
               'harness_files': {'profile.rs': '0' * 64}, 'rustflags': '-C target-cpu=znver4',
               'target_cpu': 'znver4', 'profile': 'release', 'profile_config': {}, 'guest_builder_profile': 'ci',
               'guests': {'fibonacci': {'inputs': {'1': {'sha256': '1' * 64}}}}}
              for leg, revision in [('base', 'a' * 40), ('head', 'c' * 40)]]
    samples = []
    for order, (leg, phase, repetition) in enumerate([
            ('base', 'warmup', 0), ('head', 'warmup', 0), ('base', 'measure', 0),
            ('head', 'measure', 0), ('head', 'measure', 1), ('base', 'measure', 1),
            ('base', 'measure', 2), ('head', 'measure', 2)]):
        samples.append({'leg': leg, 'revision': ('a' if leg == 'base' else 'c') * 40,
                        'phase': phase, 'repetition': repetition, 'pair_index': repetition,
                        'order': order, 'status': 'complete', 'exit_code': 0, 'reason': None,
                        'metrics': {'e2e_s': 3.0, 'prove_s': 1.0, 'setup_s': 1.0, 'verify_s': 0.1,
                                    'verify_single_s': 0.2, 'process_s': 4.0, 'peak_rss_bytes': 4096,
                                    'proof_bytes': 12, 'trace_rows': 4, 'padded_trace_rows': 8}})
    measured = copy.deepcopy(case)
    measured.update(cpu_ids=[2], status='complete', reason=None, samples=samples)
    measured['input']['sha256'] = '1' * 64
    result = {'schema_version': 1, 'repository': 'a16z/jolt', 'run_id': '123', 'tier': 'pr',
              'kind': 'comparison', 'status': 'complete', **copy.deepcopy(expected),
              'suite': {'id': 'fixture', 'metric_contract': 1, 'config_sha256': hashlib.sha256(config).hexdigest(),
                        'warmups': 1, 'repetitions': 3, 'selected_cases': []},
              'builds': builds, 'host': {'topology': [{'cpu': 2, 'core': 0, 'socket': 0, 'online': True}]},
              'wall_time': {'scope': 'sequential-build-preparation-through-results-on-this-host',
                            'build_s': 10.0, 'benchmark_s': 30.0, 'total_s': 40.0},
              'cases': [measured]}
    return result, expected, config


class E2ePublicationTests(unittest.TestCase):
    def test_complete_verified_matrix_renders_but_raw_artifact_scripts_are_not_read(self):
        result, expected, _ = fixture()
        config = publication.bench.DEFAULT_CONFIG.read_bytes()
        spec = json.loads(config)
        result['suite'].update(id=spec['id'], config_sha256=hashlib.sha256(config).hexdigest(),
                              calibration=spec.get('calibration'), calibration_sha256=spec.get('calibration_sha256'))
        template = result['cases'].pop()
        for index, case in enumerate(spec['tiers']['pr']['cases']):
            measured = copy.deepcopy(template)
            measured.update(copy.deepcopy(case), cpu_ids=list(range(case['physical_cores'])))
            measured['input']['sha256'] = '1' * 64
            for sample in measured['samples']:
                sample['order'] += index * len(measured['samples'])
            result['cases'].append(measured)
        result['host']['topology'] = [{'cpu': cpu, 'core': cpu, 'socket': 0, 'online': True}
                                     for cpu in range(24)]
        for build in result['builds']:
            build.update(profile='bench-off16', profile_config={
                'inherits': 'release', 'lto': 'off', 'codegen-units': 16})
            build['binaries'] = {'akita': {'path': '/do/not/open/this', 'sha256': '0' * 64}}
            build['guests'] = {case['workload']: {'inputs': {
                str(case['input']['parameters']['operations']): {'sha256': '1' * 64}}}
                for case in result['cases']}
        result['builds'][0]['binaries']['dory'] = {'path': '/do/not/open/this/either', 'sha256': '2' * 64}
        self.assertIs(publication.validate(result, expected, config), result)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comment.md'
            publication.bench.render_comment(result, output)
            text = output.read_text()
            self.assertTrue(text.startswith('<!-- jolt-e2e-benchmark-v1 -->'))
            for workload, cores in [('fibonacci', 1), ('sha2-chain', 8), ('btreemap', 24)]:
                self.assertIn(f'| {workload} / akita | {cores} | 3.000000 / 3.000000 | +0.00% |', text)
            self.assertNotIn(' / dory |', text)
            self.assertIn('Shared harness: `' + 'd' * 40 + '`', text)
            self.assertIn('Prover profiles: base `bench-off16`; head `bench-off16`.', text)
        for key, value in [('profile', 'release'), ('profile_config', {'lto': 'fat'})]:
            changed = copy.deepcopy(result)
            changed['builds'][1][key] = value
            with self.subTest(mismatch=key), self.assertRaisesRegex(ValueError, key):
                publication.validate(changed, expected, config)

    def test_comment_uses_leg_medians_and_keeps_unavailable_comparisons_visible(self):
        result, expected, config = fixture()
        for sample in result['cases'][0]['samples']:
            index = sample['repetition']
            if sample['phase'] == 'warmup':
                seconds, rss = 1000.0, 1000000
            elif sample['leg'] == 'base':
                seconds, rss = [10.0, 100.0, 20.0][index], [4096, 12288, 8192][index]
            else:
                seconds, rss = [30.0, 40.0, 50.0][index], [12288, 16384, 4096][index]
            sample['metrics'].update(e2e_s=seconds, process_s=seconds + 1, peak_rss_bytes=rss)
        publication.validate(result, expected, config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comment.md'
            publication.bench.render_comment(result, output)
            text = output.read_text()
            self.assertIn('| 20.000000 / 40.000000 | +100.00% | unavailable:', text)
            self.assertIn('| 8192 / 12288 | +50.00% | unavailable:', text)
            self.assertIn('ratio of head and base medians', text)
            self.assertIn('warmups excluded', text)
            self.assertNotIn('+150.00%', text)
            self.assertIn(result['suite']['config_sha256'], text)
            result['host'].update(id='fixture-host', kernel='fixture-kernel', microcode='fixture-microcode',
                control_plane={'version': 'fixture-shim', 'cli_sha256': '2' * 64, 'shim_sha256': '3' * 64},
                settings={'boost': '1', 'smt': 'on', 'governor': {'2': 'performance'},
                          'transparent_huge_pages': '[madvise]', 'swap_total_bytes': 0})
            for index, build in enumerate(result['builds']):
                build.update(toolchain='rustc fixture', target_triple='x86_64-unknown-linux-gnu',
                             cargo_lock_sha256='4' * 64, runner={'sha256': '5' * 64},
                             binaries={'dory': {'sha256': ('6' if index == 0 else '7') * 64}})
                build['guests']['fibonacci']['elf'] = {'sha256': ('8' if index == 0 else '9') * 64}
            for sample in result['cases'][0]['samples']:
                if sample['leg'] == 'head':
                    sample['metrics']['trace_rows'] = 6
            calibration = {
                'status': 'measured', 'reason': None, 'method': 'null-median3-symmetric-envelope-v2',
                'metric_contract': 1, 'window_pairs': 3, 'windows_overlap': True, 'independent_windows': False,
                'floor_pct': {'e2e_s': 3.0, 'peak_rss_bytes': 2.0}, 'envelope_multiplier': 1.5, 'rounding_pp': 0.5,
                'source': {'run_id': 'synthetic-null-fixture', 'result_sha256': '0' * 64,
                    'started_at': '2026-01-01T00:00:00Z', 'finished_at': '2026-01-01T00:01:00Z',
                    'revision': 'a' * 40, 'native_harness_revision': 'e' * 40, 'collector_sha256': '5' * 64,
                    'host_id': 'fixture-host', 'kernel': 'fixture-kernel', 'microcode': 'fixture-microcode',
                    'control_plane': copy.deepcopy(result['host']['control_plane']),
                    'toolchain': 'rustc fixture', 'target_cpu': 'znver4', 'target_triple': 'x86_64-unknown-linux-gnu',
                    'rustflags': '-C target-cpu=znver4', 'profile': 'release', 'cargo_lock_sha256': '4' * 64,
                    'settings': copy.deepcopy(result['host']['settings'])},
                'cases': {'fibonacci-dory-c1-s3-n1-v1': {
                    'status': 'measured', 'reason': None, 'workload': 'fibonacci', 'pcs': 'dory',
                    'backend': 'optimized', 'physical_cores': 1, 'cpu_ids': [2], 'scale_log2': 3,
                    'input_sha256': '1' * 64, 'guest_elf_sha256': '8' * 64, 'base_binary_sha256': '6' * 64,
                    'trace_rows': 4, 'padded_trace_rows': 8, 'n_pairs': 10, 'n_windows': 8,
                    'metrics': {'e2e_s': {'unit': 'seconds', 'threshold_pct': 3.0, 'max_symmetric_window_delta_pct': 1.0},
                                'peak_rss_bytes': {'unit': 'bytes', 'threshold_pct': 2.0,
                                                   'max_symmetric_window_delta_pct': 0.5}}}}}
            calibration_path = Path(directory) / 'synthetic-calibration.json'
            calibration_path.write_text(json.dumps(calibration))
            publication.bench.apply_calibration(result, calibration_path)
            self.assertEqual(result['suite']['calibration_sha256'], hashlib.sha256(calibration_path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'calibration differs'):
                publication.validate(result, expected, config)
            approved = json.loads(config)
            approved.update(calibration=calibration, calibration_sha256=result['suite']['calibration_sha256'])
            config = json.dumps(approved).encode()
            result['suite']['config_sha256'] = hashlib.sha256(config).hexdigest()
            publication.validate(result, expected, config)
            publication.bench.render_comment(result, output)
            text = output.read_text()
            self.assertIn('| +100.00% | 3.00% (max symmetric null 1.00%) |', text)
            self.assertIn('| +50.00% | 2.00% (max symmetric null 0.50%) |', text)
            self.assertIn('4,8 / 6,8', text)
            self.assertIn('666666666666 / 777777777777', text)
            self.assertIn('888888888888 / 999999999999', text)
            self.assertIn('synthetic-null-fixture', text)
            self.assertIn('e' * 40, text)
            self.assertIn('overlapping 3-pair windows', text)
            self.assertIn('not statistical significance', text)
            for reason, keys, value in [
                    ('missing host.microcode', ['host', 'microcode'], None),
                    ('mismatch: host.control_plane.version', ['host', 'control_plane', 'version'], 'changed'),
                    ('missing host.control_plane.cli_sha256', ['host', 'control_plane', 'cli_sha256'], None),
                    ('mismatch: collector_sha256', ['builds', 0, 'runner', 'sha256'], 'f' * 64),
                    ('mismatch: base.binary_sha256', ['builds', 0, 'binaries', 'dory', 'sha256'], 'f' * 64),
                    ('mismatch: base.trace_rows', ['cases', 0, 'samples', 2, 'metrics', 'trace_rows'], 5),
                    ('missing host.settings.boost', ['host', 'settings', 'boost'], None),
                    ('fixture reference unavailable', ['suite', 'calibration'],
                     {'status': 'unmeasured', 'reason': 'fixture reference unavailable'})]:
                changed = copy.deepcopy(result)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                if keys == ['host', 'microcode']:
                    changed['suite']['calibration']['source']['microcode'] = None
                with self.subTest(calibration=reason):
                    publication.bench.render_comment(changed, output)
                    text = output.read_text()
                    self.assertIn(f'| +100.00% | unavailable: {reason}', text)
                    self.assertIn('| +50.00% | unavailable:', text)
                    self.assertNotIn('3.00% (max symmetric null', text)
            ten_pairs = copy.deepcopy(result)
            ten_pairs['suite']['repetitions'] = 10
            for repetition in range(3, 10):
                for leg in (('base', 'head') if repetition % 2 == 0 else ('head', 'base')):
                    seconds, rss = (20.0, 8192) if leg == 'base' else (40.0, 12288)
                    sample = copy.deepcopy(next(item for item in result['cases'][0]['samples'] if item['leg'] == leg))
                    sample.update(phase='measure', repetition=repetition, pair_index=repetition,
                                  order=len(ten_pairs['cases'][0]['samples']))
                    sample['metrics'].update(e2e_s=seconds, peak_rss_bytes=rss, process_s=seconds + 1)
                    ten_pairs['cases'][0]['samples'].append(sample)
            publication.bench.render_comment(ten_pairs, output)
            self.assertIn('| +100.00% | unavailable: measured repetitions differ', output.read_text())
            self.assertIn('| +50.00% | unavailable:', output.read_text())
            for state in ('failed', 'failed sample', 'missing leg', 'missing repetition', 'zero base'):
                changed = copy.deepcopy(result)
                if state == 'failed':
                    changed['cases'][0]['status'] = 'failed'
                elif state == 'failed sample':
                    changed['cases'][0]['samples'][2]['status'] = 'failed'
                elif state == 'missing leg':
                    changed['cases'][0]['samples'] = [sample for sample in changed['cases'][0]['samples']
                                                     if sample['leg'] != 'head']
                elif state == 'missing repetition':
                    changed['cases'][0]['samples'].pop()
                else:
                    for sample in changed['cases'][0]['samples']:
                        if sample['leg'] == 'base':
                            sample['metrics'].update(e2e_s=0.0, peak_rss_bytes=0)
                with self.subTest(state=state):
                    publication.bench.render_comment(changed, output)
                    text = output.read_text()
                    self.assertIn('unavailable', text)
                    self.assertNotIn('+100.00%', text)
                    self.assertNotIn('+50.00%', text)
                    self.assertNotIn('+0.00%', text)

    def test_identity_missing_points_and_false_success_are_rejected(self):
        result, expected, config = fixture()
        variants = []
        changed = copy.deepcopy(result)
        changed['revisions']['head'] = 'f' * 40
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['provenance']['run_id'] = '456'
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['provenance']['request_run_id'] = 100
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['suite']['calibration'] = {'status': 'proposed', 'reason': 'artifact-selected calibration'}
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['suite']['calibration_sha256'] = '0' * 64
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'] = []
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'][0]['samples'].pop()
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'][0]['samples'][0]['exit_code'] = 7
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'][0]['samples'][0]['metrics']['e2e_s'] = float('nan')
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'][0]['samples'][0]['status'] = 'failed'
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'][0]['samples'][2], changed['cases'][0]['samples'][3] = \
            changed['cases'][0]['samples'][3], changed['cases'][0]['samples'][2]
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['wall_time']['scope'] = 'prebuilt-execution-only'
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['wall_time']['total_s'] = None
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['wall_time']['total_s'] = 20.0
        variants.append(changed)
        changed = copy.deepcopy(result)
        changed['cases'][0]['status'] = 'unsupported'
        changed['cases'][0]['samples'] = []
        variants.append(changed)
        for index, variant in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                publication.validate(variant, expected, config)

    def test_nonfinite_json_metadata_is_rejected_before_rendering(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            publication.bench.parse_json('{"metadata": {"unread_field": NaN}}')

    def test_captured_artifact_id_cannot_be_replaced_by_a_same_named_artifact(self):
        plan = {'harness': 'd' * 40}
        actual = {'name': publication.bench.WORKFLOW, 'path': publication.bench.WORKFLOW_FILE,
                  'event': 'workflow_run', 'head_branch': 'main',
                  'head_repository': {'full_name': 'a16z/jolt'}, 'head_sha': 'd' * 40, 'run_attempt': 1}
        def github(path):
            if path == 'actions/runs/123':
                return actual
            self.assertEqual(path, 'actions/artifacts/10')
            raise FileNotFoundError('captured artifact 10 was deleted; replacement artifact 11 is not accepted')
        with patch.dict(publication.os.environ, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1'), \
                patch.object(publication.ci, 'github', side_effect=github):
            with self.assertRaises(FileNotFoundError):
                publication.expected_from_github(plan, 10, 20)


if __name__ == '__main__':
    unittest.main()
