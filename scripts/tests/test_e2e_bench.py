import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / 'bench/run.py'
SPEC = importlib.util.spec_from_file_location('e2e_bench', SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class E2eBenchTests(unittest.TestCase):
    def test_host_snapshot_identifies_the_installed_node(self):
        capture = bench.capture
        hashes = {'/opt/pika/bin/pika-cli': '2' * 64, '/opt/pika/bin/pika-node': '3' * 64}

        def installed_command(argv):
            if argv[0] == 'lscpu':
                return capture(argv)
            if argv == ['/opt/pika/bin/pika-node', '--version']:
                return 'pika-node fixture'
            raise FileNotFoundError(argv[0])

        def installed_hash(filename):
            if filename not in hashes:
                raise FileNotFoundError(filename)
            return hashes[filename]

        with patch.object(bench, 'capture', side_effect=installed_command), \
                patch.object(bench, 'sha256', side_effect=installed_hash):
            identity = bench.host_snapshot()['control_plane']
        self.assertEqual(identity, {'version': 'pika-node fixture',
                                   'cli_sha256': '2' * 64, 'node_sha256': '3' * 64})

    def test_logged_commands_record_duration_and_outcome_after_owned_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for exit_code in (0, 7):
                with self.subTest(exit_code=exit_code):
                    marker, output = root / f'done-{exit_code}', root / f'log-{exit_code}'
                    argv = [sys.executable, '-c',
                            'import json, pathlib, sys; '
                            'assert json.loads(pathlib.Path(sys.argv[3]).read_text())[-3:] == sys.argv[1:]; '
                            'pathlib.Path(sys.argv[1]).write_text("finished"); '
                            'sys.stdout.write("{\\"child_done\\": true}"); sys.exit(int(sys.argv[2]))',
                            str(marker), str(exit_code), str(output)]
                    streamed = io.StringIO()
                    with output.open('w') as log, patch.object(bench.sys, 'stdout', streamed), \
                            patch.object(bench.time, 'perf_counter', side_effect=[10, 10.25]):
                        if exit_code:
                            with self.assertRaisesRegex(RuntimeError, 'command exited 7'):
                                bench.logged(argv, root, bench.build_environment(), log)
                        else:
                            bench.logged(argv, root, bench.build_environment(), log)
                        self.assertEqual(marker.read_text(), 'finished')
                        records = [json.loads(line) for line in output.read_text().splitlines() if line]
                        completed = [r for r in records if isinstance(r, dict) and 'command' in r]
                        self.assertEqual(completed, [{'command': argv, 'elapsed_s': 0.25,
                            'exit_code': exit_code, 'outcome': 'complete' if exit_code == 0 else 'failed'}])
                    self.assertIn('{"child_done": true}', streamed.getvalue())
                    self.assertIn(json.dumps(argv), streamed.getvalue())

    def test_cancellation_reaps_its_owned_process(self):
        with subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                              stdin=subprocess.PIPE, start_new_session=True) as child:
            bench.stop_child(child)
            self.assertIsNotNone(child.returncode)
            with self.assertRaises(ChildProcessError):
                os.waitpid(child.pid, os.WNOHANG)

    def test_host_lock_excludes_another_process_until_its_owner_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, XDG_CACHE_HOME=directory,
                       PYTHONPATH=str(SCRIPT.parent))
            owner_code = ('import run, sys\nwith run.host_lock():\n'
                          ' print("locked", flush=True)\n sys.stdin.read()\n')
            probe_code = ('import fcntl, pathlib, sys\n'
                          f'path = pathlib.Path({directory!r}) / "jolt-bench/host.lock"\n'
                          'with path.open("a") as lock:\n'
                          ' try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
                          ' except BlockingIOError: sys.exit(9)\n')
            with subprocess.Popen([sys.executable, '-c', owner_code], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as owner:
                try:
                    self.assertEqual(owner.stdout.readline(), 'locked\n')
                    self.assertEqual(subprocess.run([sys.executable, '-c', probe_code], env=env).returncode, 9)
                finally:
                    owner.communicate('', timeout=5)
            self.assertEqual(subprocess.run([sys.executable, '-c', probe_code], env=env).returncode, 0)

    def test_target_preflight_rejects_missing_compiled_instructions(self):
        manifest = {'target_triple': 'x86_64-unknown-linux-gnu',
                    'target_features': ['avx2', 'cmpxchg16b', 'sha']}
        host = {'architecture': 'x86_64', 'cpu_flags': ['avx2', 'cx16']}
        with self.assertRaisesRegex(ValueError, 'sha_ni'):
            bench.check_target(manifest, host)
        host['cpu_flags'].append('sha_ni')
        bench.check_target(manifest, host)

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
                      trace_rows=4, padded_trace_rows=8, public_output=[0, 255])
        self.assertEqual(bench.validate_metrics(values, case)['trace_rows'], 4)
        self.assertEqual(bench.validate_metrics(values, case)['public_output'], [0, 255])
        for update in ({'e2e_s': float('nan')}, {'e2e_s': 0.5}, {'padded_trace_rows': 4},
                       {'padded_trace_rows': 7}, {'backend': 'reference'}, {'proof_bytes': 0},
                       {'public_output': [256]}, {'public_output': [True]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                bench.validate_metrics(dict(values, **update), case)

    def test_build_environment_does_not_forward_service_credentials(self):
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'fixture', 'PIKA_CLI_TOKEN': 'fixture',
                                     'ACTIONS_RUNTIME_TOKEN': 'fixture', 'OPENAI_API_KEY': 'fixture',
                                     'CARGO_BUILD_JOBS': '3'}):
            env = bench.build_environment()
        self.assertNotIn('GITHUB_TOKEN', env)
        self.assertNotIn('PIKA_CLI_TOKEN', env)
        self.assertNotIn('ACTIONS_RUNTIME_TOKEN', env)
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertEqual(env['CARGO_BUILD_JOBS'], '3')

    def test_suite_times_current_preparation_instead_of_old_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(out=Path(directory), base_revision='a' * 40, head_revision='b' * 40)
            result = {'builds': [{'build_s': 86400}, {'build_s': 86400}],
                      'wall_time': {'benchmark_s': 2}}
            with patch.object(bench, 'build'), patch.object(bench, 'run', return_value=result), \
                    patch.object(bench, 'render_comment'), \
                    patch.object(bench.time, 'perf_counter', side_effect=[10, 13, 20]):
                bench.suite(args)
            receipt = json.loads((args.out / 'benchmark-results.json').read_text())
            self.assertEqual(receipt['wall_time']['build_s'], 3)
            self.assertEqual(receipt['wall_time']['total_s'], 10)

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
assert sys.argv[sys.argv.index('--backend') + 1] == 'optimized'
assert '--guest-elf' in sys.argv and '--input-file' in sys.argv
path = Path(sys.argv[sys.argv.index('--result-json') + 1])
path.write_text(json.dumps(dict(workload='fibonacci', pcs='dory', backend='optimized', scale_log2=3,
    e2e_s=0.003, prove_s=0.001, setup_s=0.001, verify_s=0.0001, verify_single_s=0.0001,
    proof_bytes=12, trace_rows=4, padded_trace_rows=8, public_output=[42])))
''')
            binary.chmod(0o700)
            failure = root / 'failure'
            failure.write_text(binary.read_text() + '\nsys.exit(7)\n')
            failure.chmod(0o700)
            manifests = {}
            for leg, executable in [('base', binary), ('head', failure)]:
                manifest = {'schema_version': 1, 'revision': ('a' if leg == 'base' else 'b') * 40,
                            'harness_revision': 'c' * 40, 'status': 'complete',
                            'toolchain': 'fixture', 'cargo_version': 'fixture', 'target_cpu': 'x86-64',
                            'profile': 'release', 'profile_config': {}, 'guest_builder_profile': 'ci',
                            'rustflags': '-C target-cpu=x86-64',
                            'source_cargo_lock_sha256': '0' * 64, 'cargo_lock_sha256': '0' * 64,
                            'harness_files': {}, 'akita_schedules': [], 'suite_config_sha256': '0' * 64,
                            'started_at': '2026-10-31T00:00:00Z', 'finished_at': '2026-10-31T00:00:01Z',
                            'build_s': 1.0, 'reason': None, 'cache': {},
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
            self.assertTrue(all(s['metrics']['public_output'] == [42] for s in samples if s['leg'] == 'base'))
            self.assertEqual(result['wall_time']['scope'], 'prebuilt-execution-only')
            self.assertIn('| failed |', (args.out / 'pr-comment.md').read_text())

    def test_three_tiers_cover_requested_workloads_and_only_requested_core_counts(self):
        config = json.loads(bench.DEFAULT_CONFIG.read_text())
        self.assertEqual(set(config['tiers']), {'pr', 'main', 'nightly'})
        pr = config['tiers']['pr']
        self.assertEqual((pr['warmups'], pr['repetitions']), (1, 3))
        self.assertEqual([(c['workload'], c['pcs'], c['backend'], c['physical_cores'],
                           c['scale_log2'], c['input']['parameters']['operations']) for c in pr['cases']],
                         [('fibonacci', 'akita', 'optimized', 1, 18, 16000),
                          ('sha2-chain', 'akita', 'optimized', 8, 20, 256),
                          ('btreemap', 'akita', 'optimized', 24, 22, 2048)])
        expected = {'fibonacci', 'sha2-chain', 'sha3-chain', 'btreemap', 'secp256k1-batch'}
        for tier in ('main', 'nightly'):
            _, cases = bench.config_cases(bench.DEFAULT_CONFIG, tier)
            supported = [case for case in cases if not case.get('unsupported_reason')]
            self.assertEqual({case['workload'] for case in supported}, expected)
            self.assertEqual({case['physical_cores'] for case in supported}, {1, 8, 24})
            self.assertEqual({case['pcs'] for case in supported}, {'dory', 'akita'})
        self.assertGreater(config['tiers']['nightly']['repetitions'], config['tiers']['main']['repetitions'])


class BuildCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.workspace = self.root / 'action-workspace'
        self.workspace.mkdir()
        self.compiles = []
        self.compile_envs = []
        self.prepares = []
        files = {
            '.gitignore': '/target/\n',
            'Cargo.toml': '[workspace]\nmembers = []\n\n[profile.release]\nlto = "fat"\ndebug = 1\n',
            'Cargo.lock': 'version = 3\n[[package]]\nname = "fixture"\nversion = "0.0.0"\n',
            'crates/jolt-prover/Cargo.toml': '[features]\nprofiling = []\n[dependencies]\n',
            'crates/jolt-prover/src/profile.rs': 'fn fixture() {}\n',
            'crates/jolt-akita/schedules/fixture.aks': 'owned schedule fixture',
            'scripts/bench/run.py': SCRIPT.read_text(),
        }
        self.config = self.workspace / 'scripts/bench/suites.json'
        config = {'schema_version': 1, 'metric_contract': 1, 'tiers': {}}
        for tier, counts, schemes in [('main', (1, 2), ('dory', 'akita')), ('pr', (2,), ('akita',))]:
            config['tiers'][tier] = {'cases': [
                {'id': f'{tier}-{pcs}-{operations}', 'workload': 'fibonacci', 'pcs': pcs,
                 'input': {'parameters': {'operations': operations}}}
                for pcs in schemes for operations in counts]}
        files['scripts/bench/suites.json'] = json.dumps(config)
        for name, text in files.items():
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git('init', '-q')
        self.git('add', *files)
        self.commit()
        self.revision = self.git('rev-parse', 'HEAD')
        self.enterContext(patch.object(bench, 'ROOT', self.workspace))
        self.enterContext(patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.root / 'runner-cache'),
            'GITHUB_WORKSPACE': str(self.workspace), 'CARGO_BUILD_JOBS': '3'}))
        self.enterContext(patch.object(bench, 'host_snapshot', return_value={'id': 'fixture-host'}))
        original_logged = bench.logged

        def capture(argv, cwd=None, env=None):
            if argv[0] == 'rustc':
                if '--print' in argv:
                    return 'target_feature="sse2"'
                return 'rustc fixture\nhost: x86_64-unknown-linux-gnu'
            if argv[0] == 'cargo':
                self.assertEqual(argv, ['cargo', '--version'])
                return 'cargo fixture'
            if argv[0] == 'rustup':
                return 'rust-src\nrustc-fixture'
            return subprocess.check_output(argv, cwd=cwd or self.workspace, env=env, text=True).strip()

        def logged(argv, cwd, env, log):
            if argv[0] == 'git':
                return original_logged(argv, cwd, env, log)
            self.assertEqual(env['CARGO_BUILD_JOBS'], '3')
            self.assertNotIn('CARGO_TARGET_DIR', env)
            log.write(json.dumps(argv) + '\n')
            if argv[:2] == ['cargo', 'metadata']:
                return
            if argv[:2] == ['cargo', 'build']:
                self.compiles.append(argv)
                self.compile_envs.append(env)
                target_dir = Path(argv[argv.index('--target-dir') + 1])
                expected = cwd / ('target/host-tools' if '--bin' in argv and 'jolt' in argv else 'target')
                self.assertEqual(target_dir, expected)
                target = target_dir / argv[argv.index('--target') + 1] / argv[argv.index('--profile') + 1]
                target.mkdir(parents=True, exist_ok=True)
                for name in (argv[index + 1] for index, arg in enumerate(argv) if arg == '--bin'):
                    (target / name).write_text('owned compiler output fixture: ' + argv[-1])
                return
            self.assertEqual(argv[1], 'prepare')
            self.assertTrue(Path(argv[0]).is_file())
            self.assertTrue(Path(env['JOLT_PATH']).is_file())
            self.prepares.append((argv, env['JOLT_PATH']))
            out = Path(argv[argv.index('--out') + 1])
            out.mkdir(parents=True)
            (out / 'guest.elf').write_bytes(b'owned ELF fixture')
            for count in argv[argv.index('--operations') + 1].split(','):
                (out / f'input-{count}.bin').write_text(count)

        self.enterContext(patch.object(bench, 'capture', side_effect=capture))
        self.enterContext(patch.object(bench, 'logged', side_effect=logged))

    def git(self, *args, cwd=None):
        return subprocess.check_output(['git', *args], cwd=cwd or self.workspace,
                                       stderr=subprocess.STDOUT, text=True).strip()

    def commit(self):
        self.git('-c', 'user.name=Benchmark fixture', '-c', 'user.email=fixture@example.invalid',
                 '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'fixture')

    def build(self, name, **options):
        args = argparse.Namespace(config=self.config, out=self.root / name, revision=self.revision,
                                  harness_revision=self.revision, target_cpu='x86-64', profile='release')
        vars(args).update(options)
        with bench.host_lock():
            manifest = bench.build(args)
        self.assertEqual(manifest['status'], 'complete', manifest['reason'])
        return manifest, json.loads((args.out / 'preparation.json').read_text())

    def test_checkout_cleanup_preserves_independent_cached_source_and_target(self):
        original, _ = self.build('original')
        source = bench.cache_root() / 'builds/x86-64/release/source'
        artifact = source / 'target/owned-artifact'
        artifact.write_bytes(b'owned artifact')
        self.assertFalse(source.is_relative_to(self.workspace))
        self.assertFalse((source / '.git/objects/info/alternates').exists())
        disposable = self.workspace / 'target/e2e-builds/old'
        disposable.mkdir(parents=True)
        (disposable / 'artifact').write_text('old workspace-local cache')
        # actions/checkout@3d3c42e: tryClean/tryReset run these exact commands.
        self.git('clean', '-ffdx')
        self.git('reset', '--hard', 'HEAD')
        self.assertFalse(disposable.exists())
        self.assertEqual(artifact.read_bytes(), b'owned artifact')
        reused, receipt = self.build('after-clean')
        self.assertEqual(reused, original)
        self.assertEqual(receipt['mode'], 'reused')
        (self.workspace / 'new-source').write_text('new commit after the independent clone')
        self.git('add', 'new-source')
        self.commit()
        changed = self.git('rev-parse', 'HEAD')
        result, receipt = self.build('changed-source', revision=changed)
        self.assertEqual(result['revision'], changed)
        self.assertEqual(receipt['mode'], 'built')
        self.assertTrue(result['cache']['cargo_target_preexisting'])
        self.assertEqual(artifact.read_bytes(), b'owned artifact')
        (self.workspace / '.git').rename(self.root / 'removed-workspace-git')
        self.assertEqual(self.git('show', 'HEAD:new-source', cwd=source),
                         'new commit after the independent clone')

    def test_reuse_keeps_construction_provenance_and_records_fresh_preparation(self):
        original, built = self.build('original', tier='main')
        cached = next((bench.cache_root() / 'bundles').glob('*/*/manifest.json'))
        original['started_at'] = '2000-01-01T00:00:00Z'
        original['finished_at'] = '2000-01-02T00:00:00Z'
        original['build_s'] = 86400
        bench.write_json(cached, original)
        self.compiles.clear()
        calibration = self.root / 'observations.json'
        calibration.write_text('{"status":"unavailable","reason":"fixture metadata only"}')
        with patch.object(bench, 'host_snapshot', return_value={
                'id': 'fixture-host', 'microcode': 'changed', 'control_plane': {'version': 'changed'}}):
            reused, receipt = self.build('reused', tier='pr', calibration=calibration)
        self.assertEqual(self.compiles, [])
        self.assertEqual(reused, original)
        self.assertEqual(receipt['mode'], 'reused')
        self.assertNotEqual(receipt['started_at'], original['started_at'])
        self.assertNotEqual(receipt['started_at'], built['started_at'])
        self.assertEqual(receipt['reused_from'], str(cached))
        self.assertEqual(receipt['required_inputs'], {'fibonacci': [2]})
        self.assertEqual(receipt['manifest_sha256'], bench.sha256(cached))
        self.assertGreater(receipt['elapsed_s'], 0)
        self.assertTrue((self.root / 'reused/construction.log').is_file())
        with patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.workspace / 'target/cache')}):
            with self.assertRaisesRegex(ValueError, 'outside the action workspace'):
                bench.cache_root()

    def test_checkout_preserves_mtimes_only_for_identical_effective_build_inputs(self):
        base = self.revision
        profile = 'crates/jolt-prover/src/profile.rs'
        extra = 'crates/jolt-prover/src/profile/prepare.rs'
        (self.workspace / profile).write_text('fn instrumented_fixture() {}\n')
        (self.workspace / extra).parent.mkdir()
        (self.workspace / extra).write_text('fn prepare_fixture() {}\n')
        self.git('add', profile, extra)
        self.commit()
        harness = self.git('rev-parse', 'HEAD')
        self.git('checkout', '--detach', base)
        changed = self.workspace / 'crates/jolt-akita/production.rs'
        changed.write_text('fn changed_production() {}\n')
        self.git('add', str(changed))
        self.commit()
        candidate = self.git('rev-parse', 'HEAD')
        self.build('base', revision=base, harness_revision=harness, tier='pr')
        source = bench.cache_root() / 'builds/x86-64/release/source'
        inputs = [profile, extra, 'Cargo.toml', 'crates/jolt-prover/Cargo.toml', 'Cargo.lock']
        prior = {}
        for name in inputs:
            path = source / name
            os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
            prior[name] = (path.read_bytes(), path.stat().st_mtime_ns)
        artifact = source / 'target/owned-artifact'
        artifact.write_bytes(b'preserved dependency fixture')
        (source / 'untracked-junk').write_text('discard this fixture')
        (source / '.gitignore').write_text('discard this tracked change')
        self.build('candidate', revision=candidate, harness_revision=harness, tier='pr')
        for name, expected in prior.items():
            with self.subTest(path=name):
                path = source / name
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
        self.assertEqual((source / 'crates/jolt-akita/production.rs').read_text(),
                         'fn changed_production() {}\n')
        self.assertEqual((source / '.gitignore').read_text(), '/target/\n')
        self.assertFalse((source / 'untracked-junk').exists())
        self.assertEqual(artifact.read_bytes(), b'preserved dependency fixture')
        self.git('checkout', '--detach', harness)
        (self.workspace / profile).write_text('fn changed_instrumentation() {}\n')
        self.git('rm', extra)
        self.git('add', profile)
        self.commit()
        next_harness = self.git('rev-parse', 'HEAD')
        self.build('changed-harness', revision=candidate, harness_revision=next_harness, tier='pr')
        self.assertEqual((source / profile).read_text(), 'fn changed_instrumentation() {}\n')
        self.assertNotEqual((source / profile).stat().st_mtime_ns, prior[profile][1])
        self.assertFalse((source / extra).exists())

    def test_main_bundle_missing_pr_input_is_not_reused(self):
        first, _ = self.build('main-only', tier='main', case=['main-dory-1', 'main-akita-1'])
        second, receipt = self.build('pr', tier='pr')
        self.assertEqual(first['revision'], second['revision'])
        self.assertEqual(set(first['guests']['fibonacci']['inputs']), {'1'})
        self.assertEqual(set(second['guests']['fibonacci']['inputs']), {'2'})
        self.assertEqual(receipt['mode'], 'built')
        self.assertIn('lacks required inputs', receipt['cache_misses'][0]['reason'])
        self.assertEqual(len(self.compiles), 5)

    def test_requested_pcs_builds_only_its_executable_and_keeps_distinct_cache_variants(self):
        partial, _ = self.build('pr', tier='pr')
        self.assertEqual(set(partial['binaries']), {'akita'})
        self.assertEqual([argv[argv.index('--bin') + 1] for argv in self.compiles],
                         ['jolt', 'jolt-prover'])
        for argv in self.compiles:
            self.assertIn('--message-format=json', argv)
        self.assertNotIn('RUSTFLAGS', self.compile_envs[0])
        self.assertEqual(self.compiles[0][-2:], ['--features', 'jolt-prover/profiling'])
        self.assertEqual(self.compiles[1][-2:], ['--features', 'profiling,akita'])
        self.assertEqual(self.compile_envs[1]['RUSTFLAGS'], '-C target-cpu=x86-64')
        self.assertEqual([argv[argv.index('--profile') + 1] for argv in self.compiles], ['ci', 'release'])
        self.assertEqual(partial['guest_builder_profile'], 'ci')
        self.assertEqual(partial['profile'], 'release')
        cli = bench.cache_root() / 'builds/x86-64/release/source/target/host-tools/x86_64-unknown-linux-gnu/ci/jolt'
        self.assertEqual(self.prepares[0][0][0], str(cli.with_name('jolt-prover')))
        self.assertEqual(self.prepares[0][1], str(cli))
        self.assertEqual(partial['guest_builder_sha256'], bench.sha256(cli))
        self.assertEqual(partial['guest_preparer_sha256'], bench.sha256(cli.with_name('jolt-prover')))
        partial_path = next((bench.cache_root() / 'bundles').glob('*/*/manifest.json'))
        partial_bytes = partial_path.read_bytes()
        self.compiles.clear()
        complete, receipt = self.build('both', tier='main', case=['main-dory-2', 'main-akita-2'])
        self.assertEqual(set(complete['binaries']), {'dory', 'akita'})
        self.assertEqual(len(self.compiles), 3)
        self.assertIn('lacks required PCS', receipt['cache_misses'][0]['reason'])
        self.assertEqual(len(list((bench.cache_root() / 'bundles').glob('*/*/manifest.json'))), 2)
        self.assertEqual(partial_path.read_bytes(), partial_bytes)

    def test_release_settings_are_recorded_without_profile_overrides(self):
        manifest, _ = self.build('release', tier='pr')
        self.assertEqual(manifest['profile_config'], {'lto': 'fat', 'debug': 1})
        for argv in self.compiles:
            self.assertNotIn('--config', argv)
        for key, value in [('profile', 'ci'), ('profile_config', {'lto': 'off'})]:
            with self.subTest(paired_mismatch=key), \
                    patch.object(bench, 'provenance', return_value=({'run_id': None}, {})), \
                    self.assertRaisesRegex(ValueError, key):
                bench.result_envelope(argparse.Namespace(context=None), {}, {}, {
                    'base': (None, manifest), 'head': (None, dict(manifest, **{key: value}))})
        self.compiles.clear()
        repeated, receipt = self.build('reused', tier='pr')
        self.assertEqual(repeated, manifest)
        self.assertEqual(receipt['mode'], 'reused')
        self.assertFalse(self.compiles)

    def test_completed_store_keeps_eight_bundles_and_preserves_recent_reuse(self):
        cache = bench.cache_root()
        entries, revisions = [], []
        for number in range(8):
            (self.workspace / 'candidate').write_text(str(number))
            self.git('add', 'candidate')
            self.commit()
            revision = self.git('rev-parse', 'HEAD')
            revisions.append(revision)
            self.build(f'output-{number}', revision=revision)
            entry = next(path.parent for path in (cache / 'bundles').glob('*/*/manifest.json')
                         if json.loads(path.read_text())['revision'] == revision)
            os.utime(entry, ns=(number + 1, number + 1))
            entries.append(entry)
        original_hash = bench.sha256(entries[0] / 'manifest.json')
        target_file = cache / 'builds/x86-64/release/source/target/keep'
        target_file.write_bytes(b'owned target')
        other_cache = self.root / 'unrelated-cache'
        other_cache.write_bytes(b'owned unrelated cache fixture')
        lock_inode = (cache / 'host.lock').stat().st_ino
        _, reused = self.build('reused', revision=revisions[0])
        self.assertEqual(reused['mode'], 'reused')
        self.assertEqual(bench.sha256(entries[0] / 'manifest.json'), original_hash)
        (self.workspace / 'candidate').write_text('ninth source')
        self.git('add', 'candidate')
        self.commit()
        newest = self.git('rev-parse', 'HEAD')
        self.build('output-8', revision=newest)
        retained = list((cache / 'bundles').glob('*/*/manifest.json'))
        self.assertEqual(len(retained), 8)
        self.assertIn(entries[0] / 'manifest.json', retained)
        self.assertIn(newest, [json.loads(path.read_text())['revision'] for path in retained])
        self.assertFalse(entries[1].parent.exists())
        self.assertEqual(target_file.read_bytes(), b'owned target')
        self.assertEqual(other_cache.read_bytes(), b'owned unrelated cache fixture')
        self.assertEqual((cache / 'host.lock').stat().st_ino, lock_inode)
        self.assertTrue(all((self.root / f'output-{number}/manifest.json').is_file() for number in range(9)))

    def test_cache_rejects_changed_identity_incomplete_or_corrupt_content(self):
        original, _ = self.build('original')
        cached = next((bench.cache_root() / 'bundles').glob('*/*/manifest.json'))
        for key in ('guest_builder_profile', 'profile_config'):
            missing_profile = dict(original)
            missing_profile.pop(key)
            bench.write_json(cached, missing_profile)
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, key):
                bench.load_bundle(cached)
        changes = {
            'revision': 'a' * 40, 'harness_revision': 'b' * 40, 'toolchain': 'other rustc',
            'cargo_version': 'other cargo', 'target_cpu': 'znver4',
            'target_triple': 'aarch64-unknown-linux-gnu', 'target_features': ['avx2'],
            'profile': 'ci', 'profile_config': {'lto': 'off'},
            'guest_builder_profile': 'release', 'rustflags': '-C target-cpu=znver4',
            'source_cargo_lock_sha256': '1' * 64, 'cargo_lock_sha256': '2' * 64,
            'harness_files': {}, 'suite_config_sha256': '3' * 64,
            'runner': {'path': 'scripts/bench/run.py', 'sha256': '4' * 64}, 'status': 'incomplete',
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                bench.write_json(cached, dict(original, **{key: value}))
                self.compiles.clear()
                _, receipt = self.build('invalid-' + key)
                self.assertEqual(len(self.compiles), 3)
                self.assertEqual(receipt['mode'], 'built')
                self.assertTrue(receipt['cache_misses'])
        for name, entry in [('binary', original['binaries']['akita']),
                            ('guest', original['guests']['fibonacci']['elf']),
                            ('input', original['guests']['fibonacci']['inputs']['2'])]:
            with self.subTest(name=name):
                (cached.parent / entry['path']).write_bytes(b'corrupted owned fixture')
                self.compiles.clear()
                _, receipt = self.build('corrupt-' + name)
                self.assertEqual(len(self.compiles), 3)
                self.assertEqual(receipt['mode'], 'built')
                self.assertIn('hash mismatch', receipt['cache_misses'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
