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
                         'tiers': {'pr': {'warmups': 1, 'repetitions': 2, 'cases': [case]}}}).encode()
    expected = {'revisions': {'base': 'a' * 40, 'head': 'b' * 40, 'merge': 'c' * 40, 'harness': 'd' * 40},
                'provenance': {'workflow': publication.bench.WORKFLOW, 'workflow_file': publication.bench.WORKFLOW_FILE,
                               'run_id': '123', 'run_attempt': '1', 'event': 'workflow_run',
                               'ref': 'refs/heads/main', 'pr_number': 7, 'request_run_id': 99}}
    builds = [{'leg': leg, 'revision': revision, 'harness_revision': 'd' * 40, 'status': 'complete',
               'harness_files': {'profile.rs': '0' * 64}, 'rustflags': '-C target-cpu=znver4',
               'target_cpu': 'znver4', 'profile': 'release',
               'guests': {'fibonacci': {'inputs': {'1': {'sha256': '1' * 64}}}}}
              for leg, revision in [('base', 'a' * 40), ('head', 'c' * 40)]]
    samples = []
    for order, (leg, phase, repetition) in enumerate([
            ('base', 'warmup', 0), ('head', 'warmup', 0), ('base', 'measure', 0),
            ('head', 'measure', 0), ('head', 'measure', 1), ('base', 'measure', 1)]):
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
                        'warmups': 1, 'repetitions': 2, 'selected_cases': []},
              'builds': builds, 'host': {'topology': [{'cpu': 2, 'core': 0, 'socket': 0, 'online': True}]},
              'wall_time': {'scope': 'sequential-build-preparation-through-results-on-this-host',
                            'build_s': 10.0, 'benchmark_s': 30.0, 'total_s': 40.0},
              'cases': [measured]}
    return result, expected, config


class E2ePublicationTests(unittest.TestCase):
    def test_complete_verified_matrix_renders_but_raw_artifact_scripts_are_not_read(self):
        result, expected, config = fixture()
        result['builds'][0]['binaries'] = {'dory': {'path': '/do/not/open/this', 'sha256': '0' * 64}}
        self.assertIs(publication.validate(result, expected, config), result)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comment.md'
            publication.bench.render_comment(result, output)
            text = output.read_text()
            self.assertTrue(text.startswith('<!-- jolt-e2e-benchmark-v1 -->'))
            self.assertIn('| fibonacci / dory | 1 | 3.000000 | 3.000000 | +0.00% |', text)
            self.assertIn('Shared harness: `' + 'd' * 40 + '`', text)

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
