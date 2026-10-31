import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / 'bench/run.py'
SPEC = importlib.util.spec_from_file_location('e2e_bench', SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class E2eBenchTests(unittest.TestCase):
    def test_topology_selects_one_allowed_thread_per_physical_core(self):
        rows = [{'cpu': i * 2 + sibling, 'core': i, 'socket': 0,
                 'online': True, 'allowed': True}
                for i in range(24) for sibling in (0, 1)]
        self.assertEqual(bench.physical_cpu_ids(rows, 24), list(range(0, 48, 2)))
        rows[0]['allowed'] = False
        self.assertEqual(bench.physical_cpu_ids(rows, 1), [1])
        rows[1]['online'] = False
        with self.assertRaisesRegex(ValueError, 'only 23 available'):
            bench.physical_cpu_ids(rows, 24)
        with self.assertRaises(ValueError):
            bench.physical_cpu_ids(rows, 16)

    def test_paired_order_keeps_warmups_separate(self):
        samples = list(bench.sample_order(True, 1, 3))
        self.assertEqual([s['leg'] for s in samples],
                         ['base', 'head', 'base', 'head', 'head', 'base', 'base', 'head'])
        self.assertEqual([s['repetition'] for s in samples if s['phase'] == 'measure'],
                         [0, 0, 1, 1, 2, 2])
        self.assertEqual([s['order'] for s in samples], list(range(8)))

    def test_bundle_paths_and_bytes_are_checked_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            file = root / 'proof'
            file.write_bytes(b'fixed')
            entry = {'path': 'proof', 'sha256': bench.sha256(file)}
            self.assertEqual(bench.checked_file(manifest, entry), file)
            file.write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                bench.checked_file(manifest, entry)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                bench.checked_file(manifest, {'path': '../elsewhere', 'sha256': '0' * 64})

    def test_metric_contract_rejects_nonfinite_and_incorrect_boundaries(self):
        case = {'workload': 'fibonacci', 'pcs': 'dory', 'scale_log2': 3}
        values = dict(case, backend='optimized', e2e_s=3.0, prove_s=1.0, setup_s=1.0,
                      verify_s=0.1, verify_single_s=0.1, proof_bytes=12,
                      trace_rows=4, padded_trace_rows=8)
        self.assertEqual(bench.validate_metrics(values, case)['trace_rows'], 4)
        for update in ({'e2e_s': float('nan')}, {'e2e_s': 0.5}, {'padded_trace_rows': 4},
                       {'padded_trace_rows': 7}, {'backend': 'reference'}, {'proof_bytes': 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                bench.validate_metrics(dict(values, **update), case)

    def test_build_environment_does_not_forward_service_credentials(self):
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'fixture', 'PIKA_CLI_TOKEN': 'fixture',
                                     'ACTIONS_RUNTIME_TOKEN': 'fixture', 'OPENAI_API_KEY': 'fixture'}):
            env = bench.build_environment()
        self.assertNotIn('GITHUB_TOKEN', env)
        self.assertNotIn('PIKA_CLI_TOKEN', env)
        self.assertNotIn('ACTIONS_RUNTIME_TOKEN', env)
        self.assertNotIn('OPENAI_API_KEY', env)

    def test_failed_owned_process_is_reaped_and_never_aggregated_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'schema_version': 1, 'id': 'fixture', 'metric_contract': 1,
                      'tiers': {'pr': {'warmups': 1, 'repetitions': 2, 'cases': [{
                          'id': 'fixture-dory-c1', 'workload': 'fibonacci', 'pcs': 'dory',
                          'backend': 'optimized', 'physical_cores': 1, 'scale_log2': 3,
                          'input': {'parameters': {'operations': 1}, 'seed': None}}]}}}
            config_path = root / 'suites.json'
            config_path.write_text(json.dumps(config))
            for name in ('guest.elf', 'input.bin'):
                (root / name).write_bytes(b'fixture')
            binary = root / 'prover'
            binary.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
assert not any(name in os.environ for name in ('GITHUB_TOKEN', 'PIKA_CLI_TOKEN', 'ACTIONS_RUNTIME_TOKEN'))
assert os.environ['RAYON_NUM_THREADS'] == '1'
path = Path(sys.argv[sys.argv.index('--result-json') + 1])
path.write_text(json.dumps(dict(workload='fibonacci', pcs='dory', backend='optimized', scale_log2=3,
    e2e_s=0.003, prove_s=0.001, setup_s=0.001, verify_s=0.0001, verify_single_s=0.0001,
    proof_bytes=12, trace_rows=4, padded_trace_rows=8)))
''')
            binary.chmod(0o700)
            failure = root / 'failure'
            failure.write_text(binary.read_text() + '\nsys.exit(7)\n')
            failure.chmod(0o700)
            manifests = {}
            for leg, executable in [('base', binary), ('head', failure)]:
                manifest = {'schema_version': 1, 'revision': ('a' if leg == 'base' else 'b') * 40,
                            'harness_revision': 'c' * 40, 'status': 'complete',
                            'runner': {'sha256': bench.sha256(SCRIPT)},
                            'host': {'architecture': bench.platform.machine()},
                            'target_triple': bench.platform.machine() + '-unknown-linux-gnu',
                            'target_features': [],
                            'binaries': {'dory': bench.bundle_file(executable, root)},
                            'guests': {'fibonacci': {'elf': bench.bundle_file(root / 'guest.elf', root),
                                'inputs': {'1': bench.bundle_file(root / 'input.bin', root)}}}}
                manifests[leg] = root / f'{leg}.json'
                manifests[leg].write_text(json.dumps(manifest))
            cpu = min(os.sched_getaffinity(0))
            host = {'architecture': bench.platform.machine(), 'cpu_flags': [],
                    'topology': [{'cpu': cpu, 'core': 0, 'socket': 0, 'online': True}]}
            args = argparse.Namespace(config=config_path, tier='pr', case=[], kind=None,
                                      warmups=None, repetitions=None, out=root / 'results', **manifests)
            with patch.object(bench, 'host_snapshot', return_value=host), patch.dict(os.environ,
                    {'GITHUB_TOKEN': 'fixture', 'PIKA_CLI_TOKEN': 'fixture'}), \
                    patch.object(bench, 'provenance', return_value=({'run_id': None}, {})):
                result = bench.run(args)
            samples = result['cases'][0]['samples']
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(samples), 6)
            self.assertEqual([s['exit_code'] for s in samples if s['leg'] == 'head'], [7, 7, 7])
            self.assertTrue(all(s['metrics'] is None for s in samples if s['leg'] == 'head'))
            self.assertTrue(all(s['metrics']['peak_rss_bytes'] > 0 for s in samples if s['leg'] == 'base'))
            self.assertEqual(result['wall_time']['scope'], 'prebuilt-execution-only')
            self.assertIn('| failed |', (args.out / 'pr-comment.md').read_text())

    def test_three_tiers_cover_requested_workloads_and_only_requested_core_counts(self):
        config = json.loads(bench.DEFAULT_CONFIG.read_text())
        self.assertEqual(set(config['tiers']), {'pr', 'main', 'nightly'})
        expected = {'fibonacci', 'sha2-chain', 'sha3-chain', 'btreemap', 'secp256k1-batch'}
        for tier in ('main', 'nightly'):
            _, cases = bench.config_cases(bench.DEFAULT_CONFIG, tier)
            supported = [case for case in cases if not case.get('unsupported_reason')]
            self.assertEqual({case['workload'] for case in supported}, expected)
            self.assertEqual({case['physical_cores'] for case in supported}, {1, 8, 24})
            self.assertEqual({case['pcs'] for case in supported}, {'dory', 'akita'})
        self.assertGreater(config['tiers']['nightly']['repetitions'], config['tiers']['main']['repetitions'])


if __name__ == '__main__':
    unittest.main()
