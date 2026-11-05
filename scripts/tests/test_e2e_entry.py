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


def request(*, tier='pr', extra=None, harness='a' * 40):
    plan = {'enabled': True, 'harness': harness, 'tier': tier,
            'provenance': {'run_id': '123', 'run_attempt': '1'}}
    manifest = {'schema': 1, 'tier': tier, 'run_id': 123, 'run_attempt': 1,
                'plan_artifact': 'jolt-e2e-plan-123-1',
                'builds_artifact': None if tier == 'pr' else 'jolt-e2e-builds-123-1'}
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

    def test_detached_worker_uses_the_local_harness_finalizes_and_submit_is_idempotent(self):
        harness = self.home / 'fixture'
        script = harness / 'scripts/bench/ci.py'
        script.parent.mkdir(parents=True)
        script.write_text('''import argparse, json, os
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
    (out / 'benchmark-results.json').write_text(json.dumps({'sid': os.getsid(0), 'pid': os.getpid()}))
    (out / 'runtime-cache').mkdir()
    (out / 'runtime-cache/secret').write_text('not published')
    (out / 'builds').mkdir()
    (out / 'builds/secret').write_text('not published')
else:
    (out / 'finalized').write_text('yes')
''')
        subprocess.run(['git', 'init', '-q', str(harness)], check=True)
        subprocess.run(['git', '-C', str(harness), 'add', 'scripts/bench/ci.py'], check=True)
        subprocess.run(['git', '-C', str(harness), '-c', 'user.name=fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-qm', 'test: local harness fixture'], check=True)
        revision = subprocess.check_output(['git', '-C', str(harness), 'rev-parse', 'HEAD'], text=True).strip()
        children = []
        spawn = subprocess.Popen
        run = subprocess.run
        def launch(*args, **kwargs):
            child = spawn(*args, **kwargs)
            children.append(child)
            return child
        def local_remote(argv, **kwargs):
            if argv[-1] == 'https://github.com/a16z/jolt.git':
                argv = [*argv[:-1], str(harness)]
            return run(argv, **kwargs)
        with patch.object(entry.subprocess, 'run', side_effect=local_remote), \
                patch.object(entry.subprocess, 'Popen', side_effect=launch), \
                patch.dict(os.environ, GH_TOKEN='owned fixture', SSH_AUTH_SOCK='owned fixture'):
            try:
                with (self.home / 'submit.lock').open('a') as gate:
                    entry.fcntl.flock(gate, entry.fcntl.LOCK_EX)
                    self.assertEqual(entry.submit(self.job, request(harness=revision), self.home, gate.fileno()),
                                     {'state': 'submitted'})
                with (self.home / 'submit.lock').open('a') as probe:
                    entry.fcntl.flock(probe, entry.fcntl.LOCK_EX)
                    self.assertTrue((self.job / 'worker.json').is_file())
                count = len(children)
                self.assertEqual(entry.submit(self.job, io.BytesIO(b'invalid duplicate'), self.home), {'state': 'exists'})
                self.assertEqual(len(children), count)
                for child in children:
                    child.wait(timeout=10)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=10)
        self.assertEqual(entry.status(self.job), {'state': 'finished', 'exit_code': 0})
        identity = json.loads((self.job / 'result/benchmark-results.json').read_text())
        self.assertEqual(identity['sid'], identity['pid'])
        self.assertEqual((self.job / 'result/finalized').read_text(), 'yes')
        archive = io.BytesIO()
        entry.fetch(self.job, archive)
        with zipfile.ZipFile(archive) as result:
            self.assertEqual(set(result.namelist()), {'benchmark-results.json', 'finalized'})

    def test_timeout_reaps_the_owned_target_and_still_finalizes(self):
        source = self.job / 'harness/scripts/bench'
        source.mkdir(parents=True)
        (self.job / 'tmp').mkdir()
        entry.write_json(self.job / 'plan.json', {'tier': 'pr'})
        (source / 'ci.py').write_text('''import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--out') + 1])
if sys.argv[1] == 'target':
    import signal
    signal.pause()
else:
    out.mkdir(exist_ok=True)
    (out / 'finalized').write_text('yes')
''')
        children, spawn = [], subprocess.Popen
        def launch(*args, **kwargs):
            child = spawn(*args, **kwargs)
            children.append(child)
            return child
        with patch.dict(entry.TIMEOUTS, pr=0), patch.object(entry.subprocess, 'Popen', side_effect=launch):
            entry.worker(self.job, self.home)
        self.assertEqual(entry.status(self.job), {'state': 'finished', 'exit_code': 124})
        self.assertEqual((self.job / 'result/finalized').read_text(), 'yes')
        for child in children:
            self.assertIsNotNone(child.returncode)
            with self.assertRaises(ChildProcessError):
                os.waitpid(child.pid, os.WNOHANG)

    def test_status_uses_owned_process_identity_and_exit_record_not_pid_existence(self):
        self.assertEqual(entry.status(self.job), {'state': 'absent', 'exit_code': None})
        self.job.mkdir()
        self.assertEqual(entry.status(self.job), {'state': 'lost', 'exit_code': None})
        with subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                              stdin=subprocess.PIPE, start_new_session=True) as child:
            identity = entry.process_identity(child.pid)
            entry.write_json(self.job / 'worker.json', identity)
            self.assertEqual(entry.status(self.job)['state'], 'running')
            entry.write_json(self.job / 'worker.json', dict(identity, boot='different-boot'))
            self.assertEqual(entry.status(self.job)['state'], 'lost')
            child.communicate(b'', timeout=5)
        entry.write_json(self.job / 'worker.json', identity)
        self.assertEqual(entry.status(self.job)['state'], 'lost')
        with self.assertRaisesRegex(ValueError, 'finished'):
            entry.fetch(self.job, io.BytesIO())
        entry.write_json(self.job / 'exit.json', {'exit_code': 17})
        self.assertEqual(entry.status(self.job), {'state': 'finished', 'exit_code': 17})

    def test_remove_is_idempotent_and_confined_to_the_job_directory(self):
        account = type('Account', (), {'pw_dir': str(self.home)})()
        job = self.home / '.local/share/jolt-bench/jobs/123-1'
        job.mkdir(parents=True)
        (job / 'owned').write_text('fixture')
        keep = self.home / 'keep'
        keep.write_text('unrelated fixture')
        with patch.object(entry.pwd, 'getpwuid', return_value=account), \
                patch.object(sys, 'argv', [str(SCRIPT)]), \
                patch.dict(os.environ, SSH_ORIGINAL_COMMAND='remove 123-1'), \
                patch.object(sys, 'stdout', io.StringIO()) as output:
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


if __name__ == '__main__':
    unittest.main()
