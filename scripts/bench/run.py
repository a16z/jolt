#!/usr/bin/env python3
"""Reproducible CPU proof builds and fresh-process measurements (schema 1)."""

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('suites.json')
WORKFLOW = 'End-to-end prover benchmarks'
WORKFLOW_FILE = '.github/workflows/e2e-bench.yml'
METRICS = ('e2e_s', 'prove_s', 'setup_s', 'verify_s', 'verify_single_s',
           'process_s', 'peak_rss_bytes', 'proof_bytes', 'trace_rows', 'padded_trace_rows')
HARNESS_PATHS = ('crates/jolt-prover/src/profile.rs',
                 'crates/jolt-prover/src/profile',
                 'crates/jolt-prover/src/bin/jolt_prover.rs',
                 'examples/secp256k1-batch/guest')
LINUX_FEATURE_NAMES = {
    'bmi1': 'bmi1', 'cmpxchg16b': 'cx16', 'lzcnt': 'abm', 'sha': 'sha_ni',
    'sse3': 'pni', 'sse4.1': 'sse4_1', 'sse4.2': 'sse4_2',
    'avx512bf16': 'avx512_bf16', 'avx512bitalg': 'avx512_bitalg',
    'avx512vbmi2': 'avx512_vbmi2', 'avx512vnni': 'avx512_vnni',
    'avx512vpopcntdq': 'avx512_vpopcntdq',
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_json(text):
    def reject(value):
        raise ValueError(f'non-finite JSON number: {value}')
    return json.loads(text, parse_constant=reject)


def capture(argv, cwd=ROOT, env=None):
    return subprocess.check_output(argv, cwd=cwd, env=env, text=True).strip()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def build_environment():
    names = ('PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL',
             'CARGO_HOME', 'RUSTUP_HOME', 'RUSTUP_TOOLCHAIN', 'RUSTC_WRAPPER', 'RUSTC_BOOTSTRAP',
             'CARGO_BUILD_JOBS', 'RUNNER_TRACKING_ID')
    return {name: os.environ[name] for name in names if name in os.environ}


def host_snapshot():
    rows = capture(['lscpu', '--parse=CPU,CORE,SOCKET,NODE,ONLINE']).splitlines()
    topology = []
    allowed = os.sched_getaffinity(0)
    for line in rows:
        if not line or line.startswith('#'):
            continue
        cpu, core, socket, node, online = line.split(',')
        topology.append({'cpu': int(cpu), 'core': int(core), 'socket': int(socket),
                         'node': None if node == '-' else int(node),
                         'online': online == 'Y', 'allowed': int(cpu) in allowed})
    cpuinfo = Path('/proc/cpuinfo').read_text()
    model = re.search(r'^model name\s*:\s*(.+)$', cpuinfo, re.M)
    flags = re.search(r'^flags\s*:\s*(.+)$', cpuinfo, re.M)
    microcode = re.search(r'^microcode\s*:\s*(.+)$', cpuinfo, re.M)
    meminfo = dict(re.findall(r'^(\w+):\s+(\d+) kB$', Path('/proc/meminfo').read_text(), re.M))
    machine_id = read_optional('/etc/machine-id') or platform.node()
    control_plane = {'version': None, 'cli_sha256': None, 'shim_sha256': None}
    for name in ('cli', 'shim'):
        try:
            control_plane[f'{name}_sha256'] = sha256(f'/opt/pika/bin/pika-{name}')
        except OSError:
            pass
    try:
        control_plane['version'] = capture(['/opt/pika/bin/pika-shim', '--version']) or None
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        'id': hashlib.sha256(machine_id.encode()).hexdigest(),
        'hostname': platform.node(), 'architecture': platform.machine(),
        'cpu_model': model.group(1) if model else None, 'kernel': platform.release(),
        'microcode': microcode.group(1) if microcode else None, 'control_plane': control_plane,
        'physical_cores': len({(r['socket'], r['core']) for r in topology}),
        'logical_cpus': len(topology), 'memory_bytes': int(meminfo['MemTotal']) * 1024,
        'topology': topology, 'cpu_flags': flags.group(1).split() if flags else [],
        'settings': {
            'online_cpus': read_optional('/sys/devices/system/cpu/online'),
            'governor': {str(r['cpu']): read_optional(
                f"/sys/devices/system/cpu/cpu{r['cpu']}/cpufreq/scaling_governor")
                for r in topology if r['online']},
            'boost': read_optional('/sys/devices/system/cpu/cpufreq/boost'),
            'smt': read_optional('/sys/devices/system/cpu/smt/control'),
            'transparent_huge_pages': read_optional('/sys/kernel/mm/transparent_hugepage/enabled'),
            'swap_total_bytes': int(meminfo['SwapTotal']) * 1024,
            'swap_free_bytes': int(meminfo['SwapFree']) * 1024,
        },
    }


def physical_cpu_ids(topology, count):
    if count not in (1, 8, 24):
        raise ValueError('only 1, 8, or 24 physical cores are supported')
    selected = {}
    for row in sorted(topology, key=lambda r: r['cpu']):
        if row['online'] and row.get('allowed', True):
            selected.setdefault((row['socket'], row['core']), row['cpu'])
    if len(selected) < count:
        raise ValueError(f'{count} physical cores requested, only {len(selected)} available')
    return list(selected.values())[:count]


def check_target(manifest, host):
    if manifest['target_triple'].split('-')[0] != host['architecture']:
        raise ValueError('compiled target and measurement host architectures differ')
    required = {LINUX_FEATURE_NAMES.get(feature, feature) for feature in manifest['target_features']}
    missing = required - set(host['cpu_flags'])
    if missing:
        raise ValueError(f'target lacks compiled CPU features: {sorted(missing)}')


def cache_root():
    path = (Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'jolt-bench').resolve()
    workspace = Path(os.environ.get('GITHUB_WORKSPACE', ROOT)).resolve()
    if path.is_relative_to(workspace) or path.is_relative_to(ROOT.resolve()):
        raise ValueError('runner cache must be outside the action workspace and harness checkout')
    return path


@contextlib.contextmanager
def host_lock():
    path = cache_root() / 'host.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def config_cases(config_path, tier, filters=()):
    config = parse_json(Path(config_path).read_text())
    if config['schema_version'] != 1 or config['metric_contract'] != 1:
        raise ValueError('unsupported suite or metric contract')
    cases = config['tiers'][tier]['cases']
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case ids')
    if set(filters) - set(ids):
        raise ValueError(f'unknown case ids: {sorted(set(filters) - set(ids))}')
    cases = [case for case in cases if not filters or case['id'] in filters]
    for case in cases:
        if case['physical_cores'] not in (1, 8, 24) or case['backend'] != 'optimized':
            raise ValueError('suite must use optimized backend and 1/8/24 physical cores')
        if case['pcs'] not in ('dory', 'akita'):
            raise ValueError('unknown PCS')
    return config, cases


def git_revision(value):
    revision = capture(['git', 'rev-parse', '--verify', f'{value}^{{commit}}'])
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('expected a full Git commit')
    return revision


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f'signal {signum}')


def stop_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def logged(argv, cwd, env, log):
    log.write(json.dumps(argv) + '\n')
    log.flush()
    begin, code, outcome = time.perf_counter(), None, 'failed'
    try:
        with subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as child:
            try:
                code = child.wait()
            except BaseException:
                stop_child(child)
                code = child.returncode
                raise
        if code == 0:
            outcome = 'complete'
    finally:
        log.write('\n' + json.dumps({'command': argv, 'elapsed_s': time.perf_counter() - begin,
                                    'exit_code': code, 'outcome': outcome}) + '\n')
        log.flush()
    if code:
        raise RuntimeError(f'command exited {code}: {argv[0]} (see build.log)')


def add_manifest_items(source):
    path = source / 'Cargo.toml'
    text = path.read_text()
    guest = 'examples/secp256k1-batch/guest'
    if guest not in tomllib.loads(text)['workspace']['members']:
        text = re.sub(r'(\[workspace\]\s*members\s*=\s*\[)',
                      lambda m: m[1] + f'\n  "{guest}",', text, count=1)
        path.write_text(text)
    path = source / 'crates/jolt-prover/Cargo.toml'
    text = path.read_text()
    original = text
    manifest = tomllib.loads(text)
    features = ['dep:serde_json', 'dep:jolt-inlines-secp256k1', 'jolt-inlines-secp256k1/host']
    additions = [item for item in features if item not in manifest['features']['profiling']]
    if additions:
        text = re.sub(r'(\nprofiling\s*=\s*\[)',
                      lambda m: m[1] + ''.join(f'\n  "{item}",' for item in additions), text, count=1)
    deps = {'serde_json': '{ workspace = true, features = ["std"], optional = true }',
            'jolt-inlines-secp256k1': '{ workspace = true, optional = true }'}
    additions = [f'{key} = {value}\n' for key, value in deps.items()
                 if key not in manifest['dependencies']]
    if additions:
        text = text.replace('[dependencies]\n', '[dependencies]\n' + ''.join(additions), 1)
    if text != original:
        path.write_text(text)


def lock_packages(path):
    return {(p['name'], p['version'], p.get('source'), p.get('checksum'))
            for p in tomllib.loads(Path(path).read_text())['package']}


def bundle_file(path, out):
    return {'path': str(path.relative_to(out)), 'sha256': sha256(path)}


def build(args):
    started, begin = utc_now(), time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('build output already contains a manifest; choose a new receipt directory')
    if args.target_cpu.lower() == 'native' or not re.fullmatch(r'[a-zA-Z0-9_.+-]+', args.target_cpu):
        raise ValueError('native is not a portable target; use the target ISA readback')
    revision, harness = git_revision(args.revision), git_revision(args.harness_revision)
    runner_source = subprocess.check_output(['git', 'show', f'{harness}:scripts/bench/run.py'], cwd=ROOT)
    if hashlib.sha256(runner_source).hexdigest() != sha256(__file__):
        raise ValueError('build must use the committed runner at --harness-revision')
    profile_config = {}
    if args.profile == 'bench-off16':
        harness_manifest = subprocess.check_output(['git', 'show', f'{harness}:Cargo.toml'], cwd=ROOT)
        profile_config = tomllib.loads(harness_manifest.decode())['profile'][args.profile]
    toolchain = capture(['rustc', '--version', '--verbose'])
    source = cache_root() / 'builds' / args.target_cpu / args.profile / 'source'
    existed = (source / 'target').exists()
    manifest = {'schema_version': 1, 'revision': revision, 'harness_revision': harness,
                'toolchain': toolchain, 'target_cpu': args.target_cpu, 'profile': args.profile,
                'guest_builder_profile': 'ci', 'profile_config': profile_config,
                'rustflags': f'-C target-cpu={args.target_cpu}', 'host': host_snapshot(),
                'started_at': started, 'finished_at': None, 'build_s': None,
                'status': 'incomplete', 'reason': None, 'binaries': {}, 'guests': {},
                'cache': {'cargo_target_preexisting': existed,
                          'rustc_wrapper': os.environ.get('RUSTC_WRAPPER'),
                          'cargo_build_jobs': os.environ.get('CARGO_BUILD_JOBS'),
                          'hit_status': 'not inferred from target existence'}}
    write_json(out / 'manifest.json', manifest)
    env = dict(build_environment(), RUSTFLAGS=manifest['rustflags'])
    config = parse_json(Path(args.config).read_text())
    inputs, required_pcs = {}, set()
    tiers = ([config['tiers'][args.tier]] if getattr(args, 'tier', None)
             else config['tiers'].values())
    for tier in tiers:
        for case in tier['cases']:
            if (not case.get('unsupported_reason') and
                    (not getattr(args, 'workload', []) or case['workload'] in args.workload) and
                    (not getattr(args, 'case', []) or case['id'] in args.case)):
                inputs.setdefault(case['workload'], set()).add(case['input']['parameters']['operations'])
                required_pcs.add(case['pcs'])
    required_inputs = {name: sorted(counts) for name, counts in inputs.items()}
    host_id, reused, cache_misses = manifest['host']['id'], False, []
    try:
        with (out / 'build.log').open('w') as log:
            effective_inputs = [source / name for name in
                                ('Cargo.toml', 'Cargo.lock', 'crates/jolt-prover/Cargo.toml')]
            for prefix in HARNESS_PATHS:
                path = source / prefix
                effective_inputs.extend([path] if path.is_file() else path.rglob('*'))
            prior_inputs = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                            for path in effective_inputs if path.is_file()}
            if not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                logged(['git', 'clone', '--no-local', '--no-checkout', '--no-tags', str(ROOT), str(source)], ROOT, env, log)
            if not (source / '.git').is_dir() or (source / '.git/objects/info/alternates').exists():
                raise ValueError('build source must be an independent private Git checkout')
            logged(['git', 'fetch', '--no-tags', str(ROOT), revision], source, env, log)
            logged(['git', 'checkout', '--force', '--detach', revision], source, env, log)
            logged(['git', 'clean', '-ffdx', '-e', '/target/'], source, env, log)
            paths = capture(['git', 'ls-tree', '-r', '--name-only', harness, '--', *HARNESS_PATHS]).splitlines()
            for path in paths:
                destination = source / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                contents = subprocess.check_output(['git', 'show', f'{harness}:{path}'], cwd=ROOT)
                if not destination.exists() or destination.read_bytes() != contents:
                    destination.write_bytes(contents)
            add_manifest_items(source)
            original_lock = subprocess.check_output(['git', 'show', f'{revision}:Cargo.lock'], cwd=ROOT)
            manifest['source_cargo_lock_sha256'] = hashlib.sha256(original_lock).hexdigest()
            pinned = {(p['name'], p['version'], p.get('source'), p.get('checksum'))
                      for p in tomllib.loads(original_lock.decode())['package']}
            manifest['toolchain'] = capture(['rustc', '--version', '--verbose'], cwd=source, env=env)
            manifest['cargo_version'] = capture(['cargo', '--version'], cwd=source, env=env)
            triple = re.search(r'^host: (.+)$', manifest['toolchain'], re.M).group(1)
            manifest['target_triple'] = triple
            cfg = capture(['rustc', '--print', 'cfg', '--target', triple,
                           '-C', f'target-cpu={args.target_cpu}'], cwd=source, env=env)
            manifest['target_features'] = sorted(re.findall(r'^target_feature="([^"]+)"$', cfg, re.M))
            logged(['cargo', 'metadata', '--format-version', '1'], source, env, log)
            if not pinned.issubset(lock_packages(source / 'Cargo.lock')):
                raise ValueError('benchmark harness changed an existing locked dependency')
            # Cargo's default source freshness checks use mtimes. Git/overlay/lock resolution
            # may rewrite identical inputs; only byte-identical inputs retain their old mtime.
            for path, (contents, mtime_ns) in prior_inputs.items():
                if path.is_file() and path.read_bytes() == contents:
                    os.utime(path, ns=(path.stat().st_atime_ns, mtime_ns))
            manifest['cargo_lock_sha256'] = sha256(source / 'Cargo.lock')
            manifest['harness_files'] = {str(p.relative_to(source)): sha256(p)
                for prefix in HARNESS_PATHS for p in ([source / prefix] if (source / prefix).is_file()
                                                     else sorted((source / prefix).rglob('*'))) if p.is_file()}
            manifest['suite_config_sha256'] = sha256(args.config)
            manifest['runner'] = {'path': 'scripts/bench/run.py',
                                  'sha256': hashlib.sha256(runner_source).hexdigest()}
            identity = {key: manifest[key] for key in (
                'revision', 'harness_revision', 'toolchain', 'cargo_version', 'target_cpu',
                'target_triple', 'target_features', 'profile', 'profile_config', 'guest_builder_profile',
                'rustflags', 'source_cargo_lock_sha256',
                'cargo_lock_sha256', 'harness_files', 'suite_config_sha256', 'runner')}
            bucket = cache_root() / 'bundles' / hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()).hexdigest()
            for cached_path in sorted(bucket.glob('*/manifest.json')):
                try:
                    _, cached = load_bundle(cached_path)
                    if cached['status'] != 'complete' or any(cached[key] != value for key, value in identity.items()):
                        raise ValueError('cached build identity or completion differs')
                    missing_pcs = required_pcs.difference(cached['binaries'])
                    if missing_pcs:
                        raise ValueError(f'cached bundle lacks required PCS: {sorted(missing_pcs)}')
                    for name, counts in required_inputs.items():
                        if not set(map(str, counts)).issubset(cached['guests'].get(name, {}).get('inputs', {})):
                            raise ValueError(f'cached bundle lacks required inputs for {name}: {counts}')
                    entries = [cached['binaries'][pcs] for pcs in cached['binaries']]
                    entries += [cached['runner'], *cached['akita_schedules']]
                    for guest in cached['guests'].values():
                        entries += [guest['elf'], *guest['inputs'].values()]
                    for entry in entries:
                        checked_file(cached_path, entry)
                    checked_file(cached_path, {'path': 'scripts/bench/suites.json',
                                              'sha256': identity['suite_config_sha256']})
                except (OSError, ValueError, KeyError, TypeError) as error:
                    cache_misses.append({'manifest': str(cached_path), 'reason': str(error)})
                    continue
                shutil.copytree(cached_path.parent, out, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('manifest.json', 'preparation.json', 'build.log'))
                shutil.copy2(cached_path.parent / 'build.log', out / 'construction.log')
                os.utime(cached_path.parent, None)
                manifest, reused = cached, True
                return manifest
            cargo = ['cargo', 'build', '-q', '--message-format=json', '--locked',
                     '--target-dir', str(source / 'target'), '--target', triple]
            logged([*cargo, '--profile', manifest['guest_builder_profile'], '-p', 'jolt', '--bin', 'jolt'],
                   source, env, log)
            profile_args = [argument for key, value in profile_config.items()
                            for argument in ('--config', f'profile.{args.profile}.{key}={json.dumps(value)}')]
            for pcs in sorted(required_pcs):
                features = 'profiling,akita' if pcs == 'akita' else 'profiling'
                logged([*cargo, *profile_args, '--profile', args.profile, '-p', 'jolt-prover', '--bin', 'jolt-prover',
                        '--features', features], source, env, log)
                binary = out / f'jolt-prover-{pcs}'
                shutil.copy2(source / 'target' / triple / args.profile / 'jolt-prover', binary)
                manifest['binaries'][pcs] = bundle_file(binary, out)
            guest_builder = source / 'target' / triple / manifest['guest_builder_profile'] / 'jolt'
            manifest['guest_builder_sha256'] = sha256(guest_builder)
            guest_env = dict(env, JOLT_PATH=str(guest_builder))
            guest_env.pop('RUSTFLAGS', None)
            for workload, counts in inputs.items():
                prepared = out / 'guests' / workload
                prepare_binary = out / next(iter(manifest['binaries'].values()))['path']
                logged([str(prepare_binary), 'prepare', '--name', workload,
                        '--operations', ','.join(map(str, sorted(counts))),
                        '--guest-target-dir', str(source / 'target/bench-guests'), '--out', str(prepared)],
                       source, guest_env, log)
                manifest['guests'][workload] = {'elf': bundle_file(prepared / 'guest.elf', out),
                    'inputs': {str(n): bundle_file(prepared / f'input-{n}.bin', out) for n in counts}}
            manifest['requested_workloads'] = sorted(inputs)
            schedules = out / 'akita-schedules'
            schedules.mkdir()
            manifest['akita_schedules'] = []
            for path in sorted((source / 'crates/jolt-akita/schedules').glob('*.aks')):
                destination = schedules / path.name
                shutil.copy2(path, destination)
                manifest['akita_schedules'].append(bundle_file(destination, out))
            runner = out / 'scripts/bench/run.py'
            runner.parent.mkdir(parents=True)
            runner.write_bytes(runner_source)
            shutil.copy2(args.config, runner.with_name('suites.json'))
            manifest['runner'] = bundle_file(runner, out)
            manifest['status'] = 'complete'
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        manifest['status'], manifest['reason'] = 'failed', str(error)
    finally:
        if not reused:
            manifest['finished_at'], manifest['build_s'] = utc_now(), time.perf_counter() - begin
        write_json(out / 'manifest.json', manifest)
        prepared = False
        try:
            if manifest['status'] == 'complete' and not reused:
                contents = {'pcs': sorted(manifest['binaries']), 'inputs': required_inputs}
                cached_dir = bucket / hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()
                if cached_dir.exists():
                    shutil.rmtree(cached_dir)
                shutil.copytree(out, cached_dir, ignore=shutil.ignore_patterns('manifest.json', 'preparation.json'))
                write_json(cached_dir / 'manifest.json', manifest)
                # Only completed stores publish manifest.json, after all payload files are copied.
                stored_bundles = sorted(
                    (path.parent for path in bucket.parent.glob('*/*/manifest.json')),
                    key=lambda directory: (directory == cached_dir, directory.stat().st_mtime_ns), reverse=True)
                for expired in stored_bundles[8:]:
                    shutil.rmtree(expired)
                    if not any(expired.parent.iterdir()):
                        expired.parent.rmdir()
            prepared = True
        finally:
            write_json(out / 'preparation.json', {
                'mode': 'reused' if reused else 'built', 'host_id': host_id,
                'status': manifest['status'] if prepared else 'incomplete',
                'started_at': started, 'finished_at': utc_now(), 'elapsed_s': time.perf_counter() - begin,
                'manifest_sha256': sha256(out / 'manifest.json'), 'required_inputs': required_inputs,
                'reused_from': str(cached_path) if reused else None,
                'cache_misses': cache_misses,
            })
    return manifest


def load_bundle(path):
    path = Path(path).resolve()
    manifest = parse_json(path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported build manifest version')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('build revision must be a full Git commit')
    if manifest.get('status') == 'complete':
        required = {'harness_revision', 'toolchain', 'cargo_version', 'target_cpu', 'target_triple',
                    'target_features', 'profile', 'profile_config', 'guest_builder_profile', 'rustflags', 'source_cargo_lock_sha256',
                    'cargo_lock_sha256', 'harness_files', 'binaries', 'guests', 'runner',
                    'akita_schedules', 'suite_config_sha256', 'host', 'started_at',
                    'finished_at', 'build_s', 'reason', 'cache'}
        if required - manifest.keys():
            raise ValueError(f'incomplete successful build manifest: {sorted(required - manifest.keys())}')
    return path, manifest


def checked_file(manifest_path, entry):
    relative = Path(entry['path'])
    root = manifest_path.parent
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError('bundle path escapes its manifest directory')
    if sha256(path) != entry['sha256']:
        raise ValueError(f'bundle hash mismatch: {relative}')
    return path


def provenance():
    event_path = os.environ.get('GITHUB_EVENT_PATH')
    event = parse_json(Path(event_path).read_text()) if event_path else {}
    return {
        'workflow': os.environ.get('GITHUB_WORKFLOW'),
        'workflow_file': WORKFLOW_FILE if os.environ.get('GITHUB_WORKFLOW') else None,
        'run_id': os.environ.get('GITHUB_RUN_ID'), 'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
        'event': os.environ.get('GITHUB_EVENT_NAME'), 'ref': os.environ.get('GITHUB_REF'),
        'pr_number': event.get('pull_request', {}).get('number', event.get('number')),
    }, event


def sample_order(paired, warmups, repetitions):
    order = 0
    for phase, count in [('warmup', warmups), ('measure', repetitions)]:
        for repetition in range(count):
            legs = ['base', 'head'] if paired else ['head']
            if paired and repetition % 2:
                legs.reverse()
            for leg in legs:
                yield {'leg': leg, 'phase': phase, 'repetition': repetition,
                       'pair_index': repetition if paired else None, 'order': order}
                order += 1


def measure(command, env, cwd, log_path):
    with log_path.open('wb') as log:
        started = time.perf_counter()
        with subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as child:
            try:
                _, status, usage = os.wait4(child.pid, 0)
                finished = time.perf_counter()
                child.returncode = os.waitstatus_to_exitcode(status)
            except BaseException:
                stop_child(child)
                raise
    return child.returncode, finished - started, int(usage.ru_maxrss) * 1024


def validate_metrics(metrics, case):
    for key, expected in [('workload', case['workload']), ('pcs', case['pcs']),
                          ('backend', 'optimized'), ('scale_log2', case['scale_log2'])]:
        if metrics.get(key) != expected:
            raise ValueError(f'profile result {key} does not match requested case')
    result = {key: metrics[key] for key in METRICS if key not in ('process_s', 'peak_rss_bytes')}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid metric {key}')
    for key in ('proof_bytes', 'trace_rows', 'padded_trace_rows'):
        if not isinstance(result[key], int) or result[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    padded = result['padded_trace_rows']
    if padded & (padded - 1) or padded <= result['trace_rows']:
        raise ValueError('padded trace does not cover the proof rows and final no-op')
    if result['e2e_s'] < result['prove_s'] + result['setup_s']:
        raise ValueError('end-to-end interval does not contain prove and setup')
    return result


def result_envelope(args, config, host, bundles):
    prov, event = provenance()
    context = getattr(args, 'context', None)
    if context is not None:
        prov = context['provenance']
    revisions = {leg: manifest['revision'] for leg, (_, manifest) in bundles.items()}
    pr = ({'head': {'sha': context['submitted_head']}} if context and context['is_pr']
          else event.get('pull_request'))
    paired = 'base' in bundles
    harnesses = {manifest['harness_revision'] for _, manifest in bundles.values()}
    if len(harnesses) > 1:
        raise ValueError('both legs must use the same benchmark-only harness revision')
    if paired:
        base, head = bundles['base'][1], bundles['head'][1]
        for field in ('harness_files', 'rustflags', 'target_cpu', 'profile', 'profile_config'):
            if base.get(field) != head.get(field):
                raise ValueError(f'paired builds differ in {field}')
    if args.kind == 'null' and (not paired or revisions['base'] != revisions['head']):
        raise ValueError('null comparisons require the same source revision on both legs')
    if args.kind == 'null':
        for field in ('binaries', 'guests', 'akita_schedules'):
            if bundles['base'][1].get(field) != bundles['head'][1].get(field):
                raise ValueError(f'null comparison has different {field}')
    tier = config['tiers'][args.tier]
    warmups = tier['warmups'] if args.warmups is None else args.warmups
    repetitions = tier['repetitions'] if args.repetitions is None else args.repetitions
    if warmups < 0 or repetitions < 1:
        raise ValueError('warmups must be nonnegative and repetitions positive')
    return {
        'schema_version': 1, 'run_id': prov['run_id'] or Path(args.out).resolve().name,
        'tier': args.tier, 'kind': args.kind or ('comparison' if paired else 'history'),
        'status': 'incomplete', 'reason': None, 'started_at': utc_now(), 'finished_at': None,
        'repository': 'a16z/jolt',
        'revisions': {'base': revisions.get('base'),
                      'head': pr['head']['sha'] if pr else revisions.get('head'),
                      'merge': revisions.get('head') if pr else None,
                      'harness': next(iter(harnesses), None)},
        'provenance': prov,
        'suite': {'id': config['id'], 'config_sha256': sha256(args.config), 'metric_contract': 1,
                  'warmups': warmups, 'repetitions': repetitions,
                  'selected_cases': list(args.case), 'calibration': config.get('calibration'),
                  'calibration_sha256': config.get('calibration_sha256')},
        'host': host, 'builds': [dict(manifest, leg=leg) for leg, (_, manifest) in bundles.items()],
        'wall_time': {'build_s': None, 'benchmark_s': None, 'total_s': None,
                      'scope': 'prebuilt-execution-only'}, 'cases': [],
    }


def apply_calibration(result, path):
    if path is not None:
        contents = Path(path).read_bytes()
        calibration = parse_json(contents)
        if not isinstance(calibration, dict):
            raise ValueError('calibration input must be a JSON object')
        result['suite'].update(calibration=calibration, calibration_sha256=hashlib.sha256(contents).hexdigest())


def run(args):
    started = utc_now()
    begin = time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'benchmark-results.json'
    if result_path.exists():
        raise ValueError('run output already contains results; choose a new directory')
    config, cases = config_cases(args.config, args.tier, args.case)
    bundles = {'head': load_bundle(args.head)}
    if args.base:
        bundles['base'] = load_bundle(args.base)
    host = host_snapshot()
    result = result_envelope(args, config, host, bundles)
    apply_calibration(result, getattr(args, 'calibration', None))
    result['started_at'] = started
    for _, manifest in bundles.values():
        if manifest.get('runner', {}).get('sha256') != sha256(__file__):
            raise ValueError('run this build bundle with its pinned scripts/bench/run.py')
    cache = out / 'runtime-cache'
    cache.mkdir()
    result['runtime_cache'] = {
        'dory': 'initially empty, suite-local, shared by both legs and all fresh processes',
        'setup_randomness': 'Dory may generate a URS with system entropy; its files are hash-identified below',
        'files': [],
    }
    for case in cases:
        row = dict(case, cpu_ids=[], status='missing', reason='not executed', samples=[])
        row['input'] = dict(case['input'], sha256=None)
        result['cases'].append(row)
    write_json(result_path, result)
    order = 0
    for case in result['cases']:
        if case.get('unsupported_reason'):
            case['status'], case['reason'] = 'unsupported', case['unsupported_reason']
            continue
        try:
            case['cpu_ids'] = physical_cpu_ids(host['topology'], case['physical_cores'])
            files = {}
            for leg, (path, manifest) in bundles.items():
                if manifest['status'] != 'complete':
                    raise ValueError(f'{leg} build is {manifest["status"]}: {manifest.get("reason")}')
                check_target(manifest, host)
                guest = manifest['guests'][case['workload']]
                input_file = guest['inputs'][str(case['input']['parameters']['operations'])]
                files[leg] = (checked_file(path, manifest['binaries'][case['pcs']]),
                              checked_file(path, guest['elf']), checked_file(path, input_file))
                if case['input']['sha256'] not in (None, input_file['sha256']):
                    raise ValueError('base and head input bytes differ')
                case['input']['sha256'] = input_file['sha256']
                if case['pcs'] == 'akita':
                    for schedule in manifest['akita_schedules']:
                        checked_file(path, schedule)
        except (KeyError, OSError, ValueError) as error:
            case['status'], case['reason'] = 'missing', str(error)
            write_json(result_path, result)
            continue
        case['reason'] = None
        for item in sample_order('base' in bundles, result['suite']['warmups'], result['suite']['repetitions']):
            sample = dict(item, revision=bundles[item['leg']][1]['revision'], order=order,
                          status='incomplete', exit_code=None, reason=None, metrics=None)
            order += 1
            case['samples'].append(sample)
            directory = out / 'samples' / f'{sample["order"]:05d}'
            directory.mkdir(parents=True)
            raw = directory / 'metrics.json'
            binary, elf, input_file = files[item['leg']]
            command = ['taskset', '--cpu-list', ','.join(map(str, case['cpu_ids'])), str(binary),
                       'profile', '--name', case['workload'], '--scale', str(case['scale_log2']),
                       '--format', 'none', '--backend', 'optimized', '--guest-elf', str(elf),
                       '--input-file', str(input_file), '--result-json', str(raw)]
            env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LANG': 'C.UTF-8',
                   'TMPDIR': str(directory), 'LOCALAPPDATA': str(cache),
                   'RAYON_NUM_THREADS': str(case['physical_cores']), 'RUST_BACKTRACE': '1'}
            if 'RUNNER_TRACKING_ID' in os.environ:
                env['RUNNER_TRACKING_ID'] = os.environ['RUNNER_TRACKING_ID']
            if case['pcs'] == 'akita':
                env['JOLT_AKITA_SCHEDULE_DIR'] = str(bundles[item['leg']][0].parent / 'akita-schedules')
            sample['command'] = command
            sample['started_at'] = utc_now()
            write_json(result_path, result)
            try:
                code, seconds, rss = measure(command, env, directory, directory / 'process.log')
                sample['exit_code'] = code
                if code:
                    raise ValueError(f'profile exited {code}; see samples/{sample["order"]:05d}/process.log')
                metrics = validate_metrics(parse_json(raw.read_text()), case)
                metrics.update(process_s=seconds, peak_rss_bytes=rss)
                sample['metrics'], sample['status'] = metrics, 'complete'
            except (OSError, ValueError, KeyError) as error:
                sample['status'], sample['reason'] = 'failed', str(error)
            finally:
                sample['finished_at'] = utc_now()
                write_json(result_path, result)
        case['status'] = 'complete' if all(s['status'] == 'complete' for s in case['samples']) else 'failed'
        if case['status'] == 'failed':
            case['reason'] = 'one or more warmup or measured processes failed'
        write_json(result_path, result)
    statuses = {case['status'] for case in result['cases']}
    result['status'] = ('failed' if 'failed' in statuses else 'incomplete' if 'missing' in statuses
                        else 'unsupported' if statuses == {'unsupported'} else 'complete')
    result['runtime_cache']['files'] = [bundle_file(path, out) for path in sorted(cache.rglob('*.urs'))]
    result['finished_at'] = utc_now()
    seconds = time.perf_counter() - begin
    result['wall_time'].update(benchmark_s=seconds, total_s=seconds)
    write_json(result_path, result)
    if 'base' in bundles:
        render_comment(result, out / 'pr-comment.md')
    return result


def calibration_for_case(result, case):
    calibration = result['suite'].get('calibration') or {}
    if not isinstance(calibration, dict):
        return None, 'calibration is not an object'
    if calibration.get('status') != 'measured':
        return None, calibration.get('reason') or 'reference not measured'
    try:
        for key, value in {'method': 'null-median3-symmetric-envelope-v2', 'metric_contract': 1,
                           'window_pairs': 3, 'windows_overlap': True, 'independent_windows': False,
                           'floor_pct': {'e2e_s': 3.0, 'peak_rss_bytes': 2.0},
                           'envelope_multiplier': 1.5, 'rounding_pp': 0.5}.items():
            if calibration.get(key) != value:
                raise ValueError(f'calibration {key} differs from the supported method')
        if result['suite']['repetitions'] != calibration['window_pairs']:
            raise ValueError('measured repetitions differ from the 3-pair reference estimator')
        reference = calibration['cases'].get(case['id'])
        if reference is None:
            raise ValueError('case is absent from the reference')
        if reference['status'] != 'measured':
            raise ValueError(reference.get('reason') or 'reference case not measured')
        if reference['n_pairs'] != 10 or reference['n_windows'] != 8:
            raise ValueError('reference case lacks 10 pairs and 8 windows')
        source, host = calibration['source'], result['host']
        base = next((build for build in result['builds'] if build['leg'] == 'base'), None)
        if base is None:
            raise ValueError('base build is missing')
        for key in ('run_id', 'started_at', 'finished_at', 'native_harness_revision', 'result_sha256'):
            if not source.get(key):
                raise ValueError(f'missing reference source.{key}')
        for key, length in [('native_harness_revision', 40), ('result_sha256', 64)]:
            if not re.fullmatch(f'[0-9a-f]{{{length}}}', str(source[key])):
                raise ValueError(f'invalid reference source.{key}')
        guest = base.get('guests', {}).get(case['workload'], {})
        comparisons = [('metric_contract', calibration['metric_contract'], result['suite']['metric_contract']),
                       ('base.revision', source.get('revision'), base.get('revision')),
                       ('collector_sha256', source.get('collector_sha256'), base.get('runner', {}).get('sha256')),
                       ('input_sha256', reference.get('input_sha256'), case['input'].get('sha256')),
                       ('base.guest_elf_sha256', reference.get('guest_elf_sha256'), guest.get('elf', {}).get('sha256')),
                       ('base.binary_sha256', reference.get('base_binary_sha256'),
                        base.get('binaries', {}).get(case['pcs'], {}).get('sha256'))]
        comparisons += [(key, reference.get(key), case.get(key)) for key in (
            'workload', 'pcs', 'backend', 'physical_cores', 'cpu_ids', 'scale_log2')]
        comparisons += [(f'base.{key}', source.get(key), base.get(key)) for key in (
            'toolchain', 'target_cpu', 'target_triple', 'rustflags', 'profile', 'cargo_lock_sha256')]
        comparisons += [(f'host.{key}', source.get('host_id' if key == 'id' else key), host.get(key))
                        for key in ('id', 'kernel', 'microcode')]
        comparisons += [(f'host.control_plane.{key}', source['control_plane'].get(key),
                         (host.get('control_plane') or {}).get(key))
                        for key in ('version', 'cli_sha256', 'shim_sha256')]
        comparisons += [(f'host.settings.{key}', source['settings'].get(key), host.get('settings', {}).get(key))
                        for key in ('boost', 'smt', 'governor', 'transparent_huge_pages', 'swap_total_bytes')]
        for label, expected, actual in comparisons:
            if expected is None or actual is None or expected == '' or actual == '':
                raise ValueError(f'missing {label}')
            if label.endswith('sha256') and not re.fullmatch('[0-9a-f]{64}', str(expected)):
                raise ValueError(f'invalid reference {label}')
            if label == 'host.settings.governor' and (not expected or not actual or
                    any(value in (None, '') for value in [*expected.values(), *actual.values()])):
                raise ValueError(f'missing {label}')
            if expected != actual:
                raise ValueError(f'mismatch: {label}')
        samples = [sample['metrics'] for sample in case['samples']
                   if sample['leg'] == 'base' and sample['phase'] == 'measure']
        for key in ('trace_rows', 'padded_trace_rows'):
            if reference.get(key) is None or not samples:
                raise ValueError(f'missing base.{key}')
            if any(sample[key] != reference[key] for sample in samples):
                raise ValueError(f'mismatch: base.{key}')
        for metric, unit in [('e2e_s', 'seconds'), ('peak_rss_bytes', 'bytes')]:
            value = reference['metrics'][metric]
            if value['unit'] != unit:
                raise ValueError(f'mismatch: {metric} unit')
            for key in ('threshold_pct', 'max_symmetric_window_delta_pct'):
                if (type(value.get(key)) not in (int, float) or not math.isfinite(value[key])
                        or value[key] < 0):
                    raise ValueError(f'missing or invalid {metric}.{key}')
        return reference['metrics'], None
    except (KeyError, TypeError, AttributeError) as error:
        return None, f'missing or invalid calibration evidence: {error}'
    except ValueError as error:
        return None, str(error)


def render_comment(result, out):
    import html
    import statistics

    def text(value):
        return html.escape(str(value) if value is not None else 'unavailable').replace(
            '|', '&#124;').replace('`', '&#96;').replace('\n', ' ')

    if result.get('schema_version') != 1:
        raise ValueError('unsupported result schema')
    lines = ['<!-- jolt-e2e-benchmark-v1 -->', '## End-to-end CPU proofs', '',
             f"Status: **{result['status']}**. Tier: {result['tier']}. "
             f"Comparison: {result['kind']}.", '',
             'Primary time includes guest loading, both tracing passes, preprocessing, PCS setup, '
             'proving, and proof serialization. Verification is separate. Memory is process peak '
             'resident set size (RSS).', '',
             'Values are medians; changes use the ratio of head and base medians, with warmups excluded. '
             'Positive changes mean more time or memory.', '',
             '| Workload / PCS | Physical cores | End-to-end base / head (s) | Signed time change | Observed time reference | Peak RSS base / head (bytes) | Signed RSS change | Observed RSS reference |',
             '|---|---:|---:|---:|---|---:|---:|---|']
    repetitions = result['suite']['repetitions']
    builds = {build['leg']: build for build in result['builds']}
    identities = ['| Workload / PCS / physical cores | Base / head raw,padded rows | Base / head binary SHA256 | Base / head guest SHA256 |',
                  '|---|---|---|---|']
    for case in result['cases']:
        name = case['workload']
        if not re.fullmatch(r'[a-z0-9-]+', name) or case['pcs'] not in ('dory', 'akita'):
            raise ValueError('invalid workload or PCS in comment data')
        measurements = {leg: [sample for sample in case['samples']
                              if sample['leg'] == leg and sample['phase'] == 'measure']
                        for leg in ('base', 'head')}
        point_status = case['status']
        if point_status == 'complete' and any(
                sample['status'] != 'complete' or sample['exit_code'] != 0 for sample in case['samples']):
            point_status = 'failed or incomplete samples'
        if point_status == 'complete' and any(
                len(samples) != repetitions or {sample['repetition'] for sample in samples} != set(range(repetitions))
                for samples in measurements.values()):
            point_status = 'unpaired or incomplete repetitions'
        if point_status != 'complete':
            lines.append(f"| {name} / {case['pcs']} | {case['physical_cores']} | {point_status} | "
                         'unavailable | unavailable | — | unavailable | unavailable |')
            continue
        medians = {leg: {key: statistics.median(sample['metrics'][key] for sample in values)
                         for key in ('e2e_s', 'peak_rss_bytes')}
                   for leg, values in measurements.items()}
        base, head = medians['base'], medians['head']
        changes = {key: f'{(head[key] / base[key] - 1) * 100:+.2f}%' if base[key] else 'unavailable (zero base)'
                   for key in base}
        calibrated, reason = calibration_for_case(result, case)
        basis = {key: (f"{calibrated[key]['threshold_pct']:.2f}% "
                       f"(max symmetric null {calibrated[key]['max_symmetric_window_delta_pct']:.2f}%)"
                       if calibrated else f'unavailable: {text(reason)}') for key in base}
        for key in base:
            if not base[key]:
                basis[key] = 'unavailable: zero base median'
        lines.append(f"| {name} / {case['pcs']} | {case['physical_cores']} | {base['e2e_s']:.6f} / "
                     f"{head['e2e_s']:.6f} | {changes['e2e_s']} | {basis['e2e_s']} | {base['peak_rss_bytes']:.0f} / "
                     f"{head['peak_rss_bytes']:.0f} | {changes['peak_rss_bytes']} | {basis['peak_rss_bytes']} |")
        rows, binaries, guests = [], [], []
        for leg in ('base', 'head'):
            rows.append('; '.join(f'{raw},{padded}' for raw, padded in sorted({
                (sample['metrics']['trace_rows'], sample['metrics']['padded_trace_rows'])
                for sample in measurements[leg]})))
            build = builds.get(leg, {})
            binaries.append(text((build.get('binaries', {}).get(case['pcs'], {}).get('sha256') or 'unavailable')[:12]))
            guests.append(text((build.get('guests', {}).get(name, {}).get('elf', {}).get('sha256') or 'unavailable')[:12]))
        identities.append(f"| {name} / {case['pcs']} / {case['physical_cores']} | {' / '.join(rows)} | "
                          f"{' / '.join(binaries)} | {' / '.join(guests)} |")
    calibration = result['suite'].get('calibration') or {}
    source = calibration.get('source') or {}
    if len(identities) > 2:
        lines += ['', 'Identity SHA256s below are abbreviated to 12 characters; raw JSON retains the full hashes.', '', *identities]
    if calibration.get('status') == 'measured':
        lines += ['', f"Supplied reference method: `{text(calibration.get('method'))}`. "
                  'The supported v2 method requires 10 same-binary null pairs and eight overlapping 3-pair windows, '
                  'which are not independent. Each window takes each leg’s median, then the larger '
                  'of the head/base and base/head percentage increases. The displayed reference is '
                  '1.5 times the largest window increase, with floors of 3% time and 2% RSS, rounded up '
                  'to 0.5 percentage points.', '',
                  f"Reference source: run `{text(source.get('run_id'))}`, raw result SHA256 "
                  f"`{text(source.get('result_sha256'))}`, {text(source.get('started_at'))} "
                  f"through {text(source.get('finished_at'))}.", '',
                  f"Reference revision: `{text(source.get('revision'))}`; native harness: "
                  f"`{text(source.get('native_harness_revision'))}`; collector SHA256: "
                  f"`{text(source.get('collector_sha256'))}`."]
    else:
        lines += ['', f"Calibration unavailable: {text(calibration.get('reason') or 'reference not measured')}."]
    revisions = {key: value or 'unavailable' for key, value in result['revisions'].items()}
    collector = builds.get('base', builds.get('head', {})).get('runner', {}).get('sha256')
    total = result['wall_time']['total_s']
    total_label = f'{total:.3f}' if total is not None else 'not measured'
    lines += ['', f"Base: `{revisions['base']}`. Submitted head: `{revisions['head']}`. "
              f"Measured merge: `{revisions['merge']}`. Shared harness: `{revisions['harness']}`.", '',
              f"Prover profiles: base `{text(builds.get('base', {}).get('profile') or 'unavailable')}`; "
              f"head `{text(builds.get('head', {}).get('profile') or 'unavailable')}`.", '',
              f"Warmups per leg: {result['suite']['warmups']}; measured repetitions per leg: "
              f"{result['suite']['repetitions']}. Each invocation is a fresh process; measured pairs alternate order.", '',
              f"Suite configuration SHA256: `{result['suite']['config_sha256']}`.", '',
              f"Measured collector SHA256: `{text(collector or 'unavailable')}`. "
              f"Renderer SHA256: `{sha256(__file__)}`.", '',
              f"Calibration input SHA256: `{text(result['suite'].get('calibration_sha256') or 'not supplied; inline suite configuration')}`. "
              'A manual rendering does not authenticate that input.', '',
              f"Wall receipt: {result['wall_time']['scope']}. "
              f"Total seconds: {total_label}. "
              'Prebuilt execution does not establish the build-inclusive PR budget.', '',
              'Noise references are observed informational context, not statistical significance, pass/fail decisions or a CI gate. '
              'Raw measurements, proof bytes, verification times, row counts, hashes, compiler flags, '
              'and host settings are in benchmark-results.json.', '']
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text('\n'.join(lines))


def suite(args):
    begin = time.perf_counter()
    started = utc_now()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for leg, revision in [('base', args.base_revision), ('head', args.head_revision)]:
        if revision:
            build_args = argparse.Namespace(**vars(args))
            build_args.revision, build_args.out = revision, out / 'builds' / leg
            build(build_args)
            manifests[leg] = build_args.out / 'manifest.json'
    preparation_s = time.perf_counter() - begin
    run_args = argparse.Namespace(**vars(args), base=manifests.get('base'), head=manifests['head'])
    result = run(run_args)
    result['started_at'] = started
    result['wall_time'] = {
        'build_s': preparation_s,
        'benchmark_s': result['wall_time']['benchmark_s'],
        'total_s': time.perf_counter() - begin,
        'scope': 'sequential-build-preparation-through-results-on-this-host',
    }
    result['finished_at'] = utc_now()
    write_json(out / 'benchmark-results.json', result)
    if args.base_revision:
        render_comment(result, out / 'pr-comment.md')
    return result


def disabled(args):
    config, cases = config_cases(args.config, args.tier)
    args.kind = 'comparison' if args.tier == 'pr' else 'history'
    args.case, args.warmups, args.repetitions = [], None, None
    result = result_envelope(args, config, None, {})
    result.update(status=args.status, reason=args.reason, finished_at=utc_now())
    result['cases'] = [dict(case, cpu_ids=[], input=dict(case['input'], sha256=None),
                            status='missing', reason=args.reason, samples=[]) for case in cases]
    result['wall_time']['scope'] = 'not-executed'
    write_json(Path(args.out) / 'benchmark-results.json', result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build', help='build pinned prover, guest, input and manifest artifacts')
    run_parser = commands.add_parser('run', help='measure prebuilt bundles; never compiles')
    suite_parser = commands.add_parser('suite', help='measure one continuous build-plus-run interval')
    comment_parser = commands.add_parser('render-comment', help='render Markdown without posting')
    status_parser = commands.add_parser('status', help='write a non-executed receipt')
    for item in (build_parser, run_parser, suite_parser, status_parser):
        item.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        item.add_argument('--out', type=Path, required=True)
    for item in (build_parser, suite_parser):
        item.add_argument('--harness-revision', required=True)
        item.add_argument('--target-cpu', required=True)
        item.add_argument('--profile', choices=['release', 'ci', 'bench-off16'])
    build_parser.add_argument('--revision', required=True)
    build_parser.add_argument('--workload', action='append', default=[],
                              help='prepare only this workload; repeat for an early executable handoff')
    for item in (run_parser, suite_parser):
        item.add_argument('--tier', choices=['pr', 'main', 'nightly'], required=True)
        item.add_argument('--kind', choices=['comparison', 'null', 'settings', 'history'])
        item.add_argument('--case', action='append', default=[])
        item.add_argument('--warmups', type=int)
        item.add_argument('--repetitions', type=int)
    for item in (run_parser, suite_parser, comment_parser):
        item.add_argument('--calibration', type=Path, help='optional observational JSON; never a native build input')
    run_parser.add_argument('--base', type=Path)
    run_parser.add_argument('--head', type=Path, required=True)
    suite_parser.add_argument('--base-revision')
    suite_parser.add_argument('--head-revision', required=True)
    comment_parser.add_argument('--results', type=Path, required=True)
    comment_parser.add_argument('--out', type=Path, required=True)
    status_parser.add_argument('--tier', choices=['pr', 'main', 'nightly'], required=True)
    status_parser.add_argument('--status', choices=['disabled', 'unsupported', 'failed'], required=True)
    status_parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    if args.command in ('build', 'suite') and args.profile is None:
        args.profile = 'bench-off16' if getattr(args, 'tier', None) == 'pr' else 'release'
    return args


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.command == 'render-comment':
            result = parse_json(args.results.read_text())
            apply_calibration(result, args.calibration)
            render_comment(result, args.out)
            return 0
        if args.command == 'status':
            result = disabled(args)
        else:
            with host_lock():
                result = {'build': build, 'run': run, 'suite': suite}[args.command](args)
        print(json.dumps({'status': result['status'], 'out': str(args.out)}, allow_nan=False))
        return 0 if result['status'] in ('complete', 'disabled', 'unsupported') else 1
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
