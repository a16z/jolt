import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import time
import uuid
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).parents[1] / 'bench/jolt-bench-entry'
SPEC = importlib.util.spec_from_loader('jolt_bench_entry', importlib.machinery.SourceFileLoader(
    'jolt_bench_entry', str(SCRIPT)))
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def zipped(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return output.getvalue()


def request(*, tier='pr', extra=None, harness='a' * 40, run_id=123):
    plan = {'enabled': True, 'harness': harness, 'tier': tier,
            'provenance': {'run_id': str(run_id), 'run_attempt': '1'}}
    manifest = {'schema': 1, 'tier': tier, 'run_id': run_id, 'run_attempt': 1,
                'plan_artifact': f'jolt-e2e-plan-{run_id}-1',
                'builds_artifact': None if tier == 'pr' else f'jolt-e2e-builds-{run_id}-1'}
    files = {'dispatch.json': json.dumps(manifest).encode(),
             'plan.zip': zipped({'plan.json': json.dumps(plan)})}
    if tier != 'pr':
        files['builds.zip'] = zipped({'builds.json': '{}'})
    files.update(extra or {})
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, value in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    output.seek(0)
    return output


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.job = self.home / '123-1'

    def test_executor_tiers_cover_every_planner_tier(self):
        from scripts.tests.test_e2e_ci import ci
        self.assertEqual(set(entry.TIMEOUTS), set(ci.TIERS))
        self.assertEqual(entry.TIMEOUTS['calibration'], 180 * 60)

    def test_fetch_still_refuses_symlinks_in_results(self):
        root = self.job / 'result'
        root.mkdir(parents=True)
        (root / 'benchmark-results.json').write_text('{}')
        (root / 'latest').symlink_to('benchmark-results.json')
        with patch.object(entry, 'status', return_value={'state': 'finished', 'exit_code': 0}), \
                patch.object(entry.subprocess, 'run'), \
                self.assertRaisesRegex(ValueError, 'result contains a symlink'):
            entry.fetch(self.job, io.BytesIO())

    def test_forced_command_refuses_shell_syntax_extra_arguments_and_invalid_job_ids(self):
        for command in ('', 'submit ../x', 'submit A', 'status a;id', 'fetch a b',
                        'remove a\n', 'submit ' + 'a' * 65, '--worker 123-1'):
            with self.subTest(command=command), patch.dict(os.environ, SSH_ORIGINAL_COMMAND=command), \
                    patch.object(sys, 'argv', [str(SCRIPT)]), self.assertRaises(ValueError):
                entry.main()

    def test_submit_refuses_unexpected_members_links_sizes_and_build_mismatches(self):
        for extra in ({'../outside': b'x'}, {'unexpected': b'x'}, {'builds.zip': b'x'},
                      {'plan.zip': zipped({'../plan.json': '{}'})}):
            with self.subTest(extra=list(extra)), patch.object(entry, 'fetch_harness') as fetch, \
                    patch.object(entry.subprocess, 'Popen') as spawn:
                with self.assertRaises(ValueError):
                    entry.submit(self.job, request(extra=extra), self.home)
                self.assertFalse(self.job.exists())
                fetch.assert_not_called()
                spawn.assert_not_called()
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as archive:
                member = tarfile.TarInfo('dispatch.json')
                member.type, member.linkname = kind, '../outside'
                archive.addfile(member)
            stream.seek(0)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                entry.submit(self.job, stream, self.home)
        with patch.dict(entry.LIMITS, {'dispatch.json': 1}), self.assertRaises(ValueError):
            entry.submit(self.job, request(), self.home)

    def harness(self, code, files=None):
        harness = self.home / 'fixture'
        script = harness / 'scripts/bench/ci.py'
        script.parent.mkdir(parents=True)
        script.write_text(code)
        for name, content in (files or {}).items():
            script.with_name(name).write_text(content)
        subprocess.run(['git', 'init', '-q', str(harness)], check=True)
        subprocess.run(['git', '-C', str(harness), 'add', 'scripts/bench'], check=True)
        subprocess.run(['git', '-C', str(harness), '-c', 'user.name=fixture', '-c',
                        'user.email=fixture@example.invalid', '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'test: local harness fixture'], check=True)
        revision = subprocess.check_output(['git', '-C', str(harness), 'rev-parse', 'HEAD'], text=True).strip()
        return harness, revision

    def native_manager(self):
        env = entry.environment(self.home, self.home)
        result = subprocess.run(['systemctl', '--user', 'show-environment'], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode:
            self.skipTest('native user manager is unavailable')

    def finished(self):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = entry.status(self.job)
            if current['state'] == 'finished':
                return current
        self.fail('owned unit did not finish within the test timeout')

    def test_native_worker_uses_the_local_harness_finalizes_and_submit_is_idempotent(self):
        self.native_manager()
        run_id = uuid.uuid4().int
        self.job = self.home / f'{run_id}-1'
        self.addCleanup(lambda: entry.remove(self.job))
        harness, revision = self.harness('''import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('command')
for arg in ('plan', 'out', 'dispatch', 'bundles'):
    p.add_argument('--' + arg)
a = p.parse_args()
out = Path(a.out)
out.mkdir(exist_ok=True)
assert not any(k in os.environ for k in ('GH_TOKEN', 'GITHUB_TOKEN', 'PIKA_CLI_TOKEN', 'SSH_AUTH_SOCK'))
if a.command == 'target':
    (out / 'benchmark-results.json').write_text('{}')
    print('owned worker journal output', flush=True)
    (out / 'runtime-cache').mkdir()
    (out / 'runtime-cache/secret').write_text('not published')
    (out / 'builds').mkdir()
    (out / 'builds/secret').write_text('not published')
    (out / 'builds/fixture').mkdir()
    (out / 'builds/fixture/build.log').write_text('owned failed build output')
else:
    (out / 'finalized').write_text('yes')
''')
        run = subprocess.run
        def local_remote(argv, **kwargs):
            if argv[-1] == 'https://github.com/a16z/jolt.git':
                argv = [*argv[:-1], str(harness)]
            return run(argv, **kwargs)
        with patch.object(entry.subprocess, 'run', side_effect=local_remote), \
                patch.dict(os.environ, GH_TOKEN='owned fixture', SSH_AUTH_SOCK='owned fixture'):
            self.assertEqual(entry.submit(self.job, request(harness=revision, run_id=run_id), self.home),
                             {'state': 'submitted'})
            self.assertEqual(entry.submit(self.job, io.BytesIO(b'invalid duplicate'), self.home), {'state': 'exists'})
        self.assertEqual(self.finished(), {'state': 'finished', 'exit_code': 0})
        self.assertEqual((self.job / 'result/finalized').read_text(), 'yes')
        archive = io.BytesIO()
        entry.fetch(self.job, archive)
        with zipfile.ZipFile(archive) as result:
            self.assertEqual(set(result.namelist()), {'benchmark-results.json', 'finalized', 'builds/fixture/build.log', 'executor.log'})
            self.assertEqual(result.read('builds/fixture/build.log'), b'owned failed build output')
            self.assertIn(b'owned worker journal output', result.read('executor.log'))

    def test_calibration_setup_failure_is_finalized_fetched_and_ingested(self):
        from scripts.tests.test_e2e_ingest import ingest
        self.native_manager()
        run_id = uuid.uuid4().int
        self.job = self.home / f'{run_id}-1'
        self.addCleanup(entry.remove, self.job)
        files = {name: SCRIPT.with_name(name).read_text() for name in
                 ('ci.py', 'run.py', 'calibration.py', 'suites.json')}
        files['ci_impl.py'] = files.pop('ci.py')
        harness, revision = self.harness('''from pathlib import Path
from unittest.mock import patch
import ci_impl
with patch.object(ci_impl.bench, 'cache_root', return_value=Path(__file__).resolve().parents[2] / 'cache'):
    raise SystemExit(ci_impl.main())
''', files=files)
        config = files['suites.json'].encode()
        plan = dict(enabled=True, tier='calibration', harness=revision, push_before=None, push_after=None,
            config={'path': 'scripts/bench/suites.json', 'sha256': ingest.hashlib.sha256(config).hexdigest()},
            expected_builds={revision: {}}, runs=[dict(base=None, head=revision, submitted_head=revision)],
            provenance=dict(workflow=ingest.bench.WORKFLOW, workflow_file=ingest.bench.WORKFLOW_FILE,
                run_id=str(run_id), run_attempt='1', event='workflow_dispatch', ref='refs/heads/main', pr_number=None))
        dispatch = ingest.ci.dispatch_manifest(plan, run_id, 1, artifact_id=44,
            index_sha256='0' * 64, digests={revision: '1' * 64})
        payload = request(tier='calibration', harness=revision, run_id=run_id, extra={
            'plan.zip': zipped({'plan.json': json.dumps(plan)}), 'dispatch.json': json.dumps(dispatch).encode()})
        run = subprocess.run
        def local_remote(argv, **kwargs):
            if argv[-1] == 'https://github.com/a16z/jolt.git':
                argv = [*argv[:-1], str(harness)]
            return run(argv, **kwargs)
        with patch.object(entry.subprocess, 'run', side_effect=local_remote):
            self.assertEqual(entry.submit(self.job, payload, self.home), {'state': 'submitted'})
        self.assertEqual(self.finished(), {'state': 'finished', 'exit_code': 1})
        archive = io.BytesIO()
        entry.fetch(self.job, archive)
        raw = archive.getvalue()
        receipt = {'plan_sha256': ingest.bench.sha256(self.job / 'plan.json')}
        out = self.home / 'ingested'
        digest = ingest.hashlib.sha256(raw).hexdigest()
        argv = ['ingest.py', '--job-id', f'{run_id}-1', '--plan-run-id', str(run_id), '--plan-run-attempt', '1',
            '--result-url', f'https://fixture.invalid/{digest}.zip', '--result-sha256', digest,
            '--executor-exit-code', '1', '--out', str(out)]
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, GITHUB_ACTOR=ingest.APP,
                GITHUB_REF='refs/heads/main', GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='1'), \
                patch.object(ingest, 'trusted_plan', return_value=(plan, dispatch, config, receipt)), \
                patch.object(ingest, 'download', return_value=raw):
            ingest.main()
        result = json.loads((out / 'result/benchmark-results.json').read_text())
        self.assertEqual((result['status'], result['kind'], result['wall_time']['scope']),
                         ('incomplete', 'null', ingest.ci.LOCAL_SCOPE))
        self.assertEqual(result['revisions']['base'], revision)
        check = json.loads((out / 'check/check.json').read_text())
        self.assertEqual(check['conclusion'], 'failure')
        self.assertIn('no completed calibration raw evidence', check['summary'])

    def test_native_timeout_kills_separate_session_descendants_and_preserves_finalization(self):
        self.native_manager()
        run_id = uuid.uuid4().int
        self.job = self.home / f'{run_id}-1'
        self.addCleanup(lambda: entry.remove(self.job))
        harness, revision = self.harness("""import signal, subprocess, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--out') + 1])
out.mkdir(exist_ok=True)
if sys.argv[1] == 'target':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen([sys.executable, '-c', 'import signal,sys; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); p=Path(sys.argv[1]); p.with_suffix(".tmp").write_text("ready"); p.with_suffix(".tmp").replace(p); signal.pause()', str(out / 'descendant-ready')], start_new_session=True)
    (out / 'ready.tmp').write_text('ready')
    (out / 'ready.tmp').replace(out / 'ready')
    signal.pause()
else:
    (out / 'benchmark-results.json').write_text('{}')
    (out / 'finalized').write_text('yes')
""")
        run = subprocess.run
        def local_remote(argv, **kwargs):
            if argv[-1] == 'https://github.com/a16z/jolt.git':
                argv = [*argv[:-1], str(harness)]
            if argv[0] == 'systemd-run':
                argv = [arg.replace('TimeoutStopSec=60s', 'TimeoutStopSec=1s') for arg in argv]
            return run(argv, **kwargs)
        with patch.object(entry.subprocess, 'run', side_effect=local_remote), patch.dict(entry.TIMEOUTS, pr=3):
            entry.submit(self.job, request(harness=revision, run_id=run_id), self.home)
        env = entry.environment(self.home, self.job / 'tmp')
        deadline = time.monotonic() + 30
        while not all((self.job / 'result' / name).exists() for name in ('ready', 'descendant-ready')):
            if time.monotonic() >= deadline:
                self.fail('owned worker did not publish its completed readiness record')
        self.assertEqual(self.finished(), {'state': 'finished', 'exit_code': 124})
        self.assertEqual((self.job / 'result/finalized').read_text(), 'yes')
        group = subprocess.check_output(['systemctl', '--user', 'show', entry.unit_name(self.job),
            '--property=ControlGroup', '--value'], env=env, text=True).strip()
        events = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.events'
        self.assertTrue(not events.exists() or 'populated 0' in events.read_text())

    def test_status_reads_unit_state_exit_status_and_missing_unit(self):
        with patch.object(entry, 'unit_state', return_value={'LoadState': 'not-found'}):
            self.assertEqual(entry.status(self.job), {'state': 'absent', 'exit_code': None})
        self.job.mkdir()
        base = {'LoadState': 'loaded', 'ActiveState': 'active', 'SubState': 'running',
                'Result': 'success', 'ExecMainCode': '0', 'ExecMainStatus': '0'}
        for change, expected in [({}, ('running', None)),
                ({'LoadState': 'not-found'}, ('lost', None)),
                ({'SubState': 'exited', 'ExecMainCode': '1'}, ('finished', 0)),
                ({'ActiveState': 'failed', 'SubState': 'failed', 'Result': 'exit-code',
                  'ExecMainCode': '1', 'ExecMainStatus': '7'}, ('finished', 7)),
                ({'ActiveState': 'failed', 'SubState': 'failed', 'Result': 'timeout',
                  'ExecMainCode': '2', 'ExecMainStatus': '9'}, ('finished', 124))]:
            with self.subTest(change=change), patch.object(entry, 'unit_state', return_value=dict(base, **change)):
                self.assertEqual(entry.status(self.job), dict(state=expected[0], exit_code=expected[1]))

    def test_remove_is_idempotent_and_confined_to_the_job_directory(self):
        account = type('Account', (), {'pw_dir': str(self.home), 'pw_name': 'fixture'})()
        job = self.home / '.local/share/jolt-bench/jobs/123-1'
        job.mkdir(parents=True)
        (job / 'owned').write_text('fixture')
        keep = self.home / 'keep'
        keep.write_text('unrelated fixture')
        with patch.object(entry.pwd, 'getpwuid', return_value=account), \
                patch.object(sys, 'argv', [str(SCRIPT)]), \
                patch.dict(os.environ, SSH_ORIGINAL_COMMAND='remove 123-1'), \
                patch.object(sys, 'stdout', io.StringIO()) as output, \
                patch.object(entry, 'unit_state', return_value={'LoadState': 'not-found'}):
            original_umask = os.umask(0o077)
            try:
                entry.main()
                entry.main()
            finally:
                os.umask(original_umask)
        self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()],
                         [{'state': 'removed'}, {'state': 'removed'}])
        self.assertFalse(job.exists())
        self.assertEqual(keep.read_text(), 'unrelated fixture')

    def test_submit_finishes_reading_stdin_before_acquiring_the_jobs_lock(self):
        home = self.home
        jobs = home / '.local/share/jolt-bench/jobs'
        class Input(io.BytesIO):
            def read(self, size=-1):
                with (jobs / '.lock').open('a') as probe:
                    entry.fcntl.flock(probe, entry.fcntl.LOCK_EX | entry.fcntl.LOCK_NB)
                return super().read(size)
        account = type('Account', (), {'pw_dir': str(home), 'pw_name': 'fixture'})()
        stdin = type('Stdin', (), {'buffer': Input(b'owned submission bytes')})()
        def submitted(job, incoming, selected_home):
            self.assertEqual(incoming.read(), b'owned submission bytes')
            return {'state': 'submitted'}
        with patch.object(entry.pwd, 'getpwuid', return_value=account), \
                patch.object(sys, 'argv', [str(SCRIPT)]), patch.object(sys, 'stdin', stdin), \
                patch.object(sys, 'stdout', io.StringIO()), patch.object(entry, 'submit', side_effect=submitted), \
                patch.dict(os.environ, SSH_ORIGINAL_COMMAND='submit 123-1'):
            original = os.umask(0o077)
            try:
                entry.main()
            finally:
                os.umask(original)


if __name__ == '__main__':
    unittest.main()
