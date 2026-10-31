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
import tarfile
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


def capture(argv, cwd=ROOT, env=None):
    return subprocess.check_output(argv, cwd=cwd, env=env, text=True).strip()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def build_environment():
    names = ('PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL',
             'CARGO_HOME', 'RUSTUP_HOME', 'RUSTUP_TOOLCHAIN', 'RUSTC_WRAPPER', 'RUSTC_BOOTSTRAP')
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
    meminfo = dict(re.findall(r'^(\w+):\s+(\d+) kB$', Path('/proc/meminfo').read_text(), re.M))
    machine_id = read_optional('/etc/machine-id') or platform.node()
    return {
        'id': hashlib.sha256(machine_id.encode()).hexdigest(),
        'hostname': platform.node(), 'architecture': platform.machine(),
        'cpu_model': model.group(1) if model else None, 'kernel': platform.release(),
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


@contextlib.contextmanager
def host_lock():
    path = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'jolt-bench/host.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def config_cases(config_path, tier, filters=()):
    config = json.loads(Path(config_path).read_text())
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


def logged(argv, cwd, env, log):
    log.write(json.dumps(argv) + '\n')
    log.flush()
    with subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as child:
        try:
            code = child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            raise
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
    path.write_text(text)


def lock_packages(path):
    return {(p['name'], p['version'], p.get('source'), p.get('checksum'))
            for p in tomllib.loads(Path(path).read_text())['package']}


def bundle_file(path, out):
    return {'path': str(path.relative_to(out)), 'sha256': sha256(path)}


def build(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('build output already contains a manifest; choose a new receipt directory')
    if args.target_cpu == 'native':
        raise ValueError('native is not a portable target; use the target ISA readback')
    revision, harness = git_revision(args.revision), git_revision(args.harness_revision)
    toolchain = capture(['rustc', '--version', '--verbose'])
    key = hashlib.sha256(f'{revision}:{harness}:{args.target_cpu}:{toolchain}'.encode()).hexdigest()
    source = ROOT / 'target/e2e-builds' / key / 'source'
    existed = (source / 'target').exists()
    started = utc_now()
    begin = time.perf_counter()
    manifest = {'schema_version': 1, 'revision': revision, 'harness_revision': harness,
                'toolchain': toolchain, 'target_cpu': args.target_cpu, 'profile': args.profile,
                'rustflags': f'-C target-cpu={args.target_cpu}', 'host': host_snapshot(),
                'started_at': started, 'finished_at': None, 'build_s': None,
                'status': 'incomplete', 'reason': None, 'binaries': {}, 'guests': {},
                'cache': {'cargo_target_preexisting': existed,
                          'rustc_wrapper': os.environ.get('RUSTC_WRAPPER'),
                          'hit_status': 'not inferred from target existence'}}
    write_json(out / 'manifest.json', manifest)
    env = dict(build_environment(), RUSTFLAGS=manifest['rustflags'], CARGO_BUILD_JOBS='16')
    try:
        with (out / 'build.log').open('w') as log:
            if not source.exists():
                source.mkdir(parents=True)
                archive = source.parent / 'source.tar'
                with archive.open('wb') as stream:
                    subprocess.run(['git', 'archive', revision], cwd=ROOT, stdout=stream, check=True)
                with tarfile.open(archive) as stream:
                    stream.extractall(source, filter='data')
                archive.unlink()
                paths = capture(['git', 'ls-tree', '-r', '--name-only', harness, '--', *HARNESS_PATHS]).splitlines()
                for path in paths:
                    destination = source / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(subprocess.check_output(['git', 'show', f'{harness}:{path}'], cwd=ROOT))
                add_manifest_items(source)
            original_lock = subprocess.check_output(['git', 'show', f'{revision}:Cargo.lock'], cwd=ROOT)
            manifest['source_cargo_lock_sha256'] = hashlib.sha256(original_lock).hexdigest()
            pinned = {(p['name'], p['version'], p.get('source'), p.get('checksum'))
                      for p in tomllib.loads(original_lock.decode())['package']}
            manifest['toolchain'] = capture(['rustc', '--version', '--verbose'], cwd=source, env=env)
            manifest['cargo_version'] = capture(['cargo', '--version'], cwd=source, env=env)
            logged(['cargo', 'metadata', '--format-version', '1'], source, env, log)
            if not pinned.issubset(lock_packages(source / 'Cargo.lock')):
                raise ValueError('benchmark harness changed an existing locked dependency')
            manifest['cargo_lock_sha256'] = sha256(source / 'Cargo.lock')
            manifest['harness_files'] = {str(p.relative_to(source)): sha256(p)
                for prefix in HARNESS_PATHS for p in ([source / prefix] if (source / prefix).is_file()
                                                     else sorted((source / prefix).rglob('*'))) if p.is_file()}
            for pcs, features in [('dory', 'profiling'), ('akita', 'profiling,akita')]:
                packages = ['-p', 'jolt-prover', '--bin', 'jolt-prover']
                if pcs == 'dory':
                    packages += ['-p', 'jolt', '--bin', 'jolt']
                logged(['cargo', 'build', '-q', '--message-format=short', '--locked',
                        '--profile', args.profile, *packages, '--features', features], source, env, log)
                binary = out / f'jolt-prover-{pcs}'
                shutil.copy2(source / 'target' / args.profile / 'jolt-prover', binary)
                manifest['binaries'][pcs] = bundle_file(binary, out)
            config = json.loads(Path(args.config).read_text())
            inputs = {}
            tiers = ([config['tiers'][args.tier]] if getattr(args, 'tier', None)
                     else config['tiers'].values())
            for tier in tiers:
                for case in tier['cases']:
                    if (not case.get('unsupported_reason') and
                            (not getattr(args, 'case', []) or case['id'] in args.case)):
                        inputs.setdefault(case['workload'], set()).add(case['input']['parameters']['operations'])
            guest_env = dict(env, JOLT_PATH=str(source / 'target' / args.profile / 'jolt'))
            guest_env.pop('RUSTFLAGS', None)
            for workload, counts in inputs.items():
                prepared = out / 'guests' / workload
                logged([str(out / 'jolt-prover-dory'), 'prepare', '--name', workload,
                        '--operations', ','.join(map(str, sorted(counts))),
                        '--guest-target-dir', str(source / 'target/bench-guests'), '--out', str(prepared)],
                       source, guest_env, log)
                manifest['guests'][workload] = {'elf': bundle_file(prepared / 'guest.elf', out),
                    'inputs': {str(n): bundle_file(prepared / f'input-{n}.bin', out) for n in counts}}
            schedules = out / 'akita-schedules'
            schedules.mkdir()
            manifest['akita_schedules'] = []
            for path in sorted((source / 'crates/jolt-akita/schedules').glob('*.aks')):
                destination = schedules / path.name
                shutil.copy2(path, destination)
                manifest['akita_schedules'].append(bundle_file(destination, out))
            manifest['suite_config_sha256'] = sha256(args.config)
            manifest['status'] = 'complete'
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        manifest['status'], manifest['reason'] = 'failed', str(error)
    finally:
        manifest['finished_at'], manifest['build_s'] = utc_now(), time.perf_counter() - begin
        write_json(out / 'manifest.json', manifest)
    return manifest


def load_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported build manifest version')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('build revision must be a full Git commit')
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
    event = json.loads(Path(event_path).read_text()) if event_path else {}
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
    started = time.perf_counter()
    with log_path.open('wb') as log:
        with subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as child:
            try:
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
                raise
    return child.returncode, time.perf_counter() - started, int(usage.ru_maxrss) * 1024


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
    revisions = {leg: manifest['revision'] for leg, (_, manifest) in bundles.items()}
    pr = event.get('pull_request')
    paired = 'base' in bundles
    harnesses = {manifest['harness_revision'] for _, manifest in bundles.values()}
    if len(harnesses) > 1:
        raise ValueError('both legs must use the same benchmark-only harness revision')
    if paired:
        base, head = bundles['base'][1], bundles['head'][1]
        for field in ('harness_files', 'rustflags', 'target_cpu', 'profile'):
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
                  'selected_cases': list(args.case), 'calibration': config.get('calibration')},
        'host': host, 'builds': [dict(manifest, leg=leg) for leg, (_, manifest) in bundles.items()],
        'wall_time': {'build_s': None, 'benchmark_s': None, 'total_s': None,
                      'scope': 'prebuilt-execution-only'}, 'cases': [],
    }


def run(args):
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
    begin = time.perf_counter()
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
                if manifest['host']['architecture'] != host['architecture']:
                    raise ValueError('build and target architectures differ')
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
                metrics = validate_metrics(json.loads(raw.read_text()), case)
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
    result['finished_at'] = utc_now()
    result['runtime_cache']['files'] = [bundle_file(path, out) for path in sorted(cache.rglob('*.urs'))]
    seconds = time.perf_counter() - begin
    result['wall_time'].update(benchmark_s=seconds, total_s=seconds)
    write_json(result_path, result)
    if 'base' in bundles:
        render_comment(result, out / 'pr-comment.md')
    return result


def render_comment(result, out):
    import statistics

    if result.get('schema_version') != 1:
        raise ValueError('unsupported result schema')
    lines = ['<!-- jolt-e2e-benchmark-v1 -->', '## End-to-end CPU proofs', '',
             f"Status: **{result['status']}**. Tier: {result['tier']}. "
             f"Comparison: {result['kind']}.", '',
             'Primary time includes guest loading, both tracing passes, preprocessing, PCS setup, '
             'proving, and proof serialization. Verification is separate.', '',
             '| Workload / PCS | Physical cores | Base end-to-end (s) | Head end-to-end (s) | Change | Base / head peak RSS (bytes) | Base / head proof (bytes) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for case in result['cases']:
        name = case['workload']
        if not re.fullmatch(r'[a-z0-9-]+', name) or case['pcs'] not in ('dory', 'akita'):
            raise ValueError('invalid workload or PCS in comment data')
        if case['status'] != 'complete':
            lines.append(f"| {name} / {case['pcs']} | {case['physical_cores']} | {case['status']} | — | — | — | — |")
            continue
        measurements = {leg: [sample['metrics'] for sample in case['samples']
                              if sample['leg'] == leg and sample['phase'] == 'measure'
                              and sample['status'] == 'complete'] for leg in ('base', 'head')}
        if not all(measurements.values()):
            lines.append(f"| {name} / {case['pcs']} | {case['physical_cores']} | unpaired | — | — | — | — |")
            continue
        medians = {leg: {key: statistics.median(m[key] for m in values) for key in METRICS}
                   for leg, values in measurements.items()}
        base, head = medians['base'], medians['head']
        change = f"{(head['e2e_s'] / base['e2e_s'] - 1) * 100:+.2f}%" if base['e2e_s'] else 'unavailable'
        lines.append(f"| {name} / {case['pcs']} | {case['physical_cores']} | {base['e2e_s']:.6f} | "
                     f"{head['e2e_s']:.6f} | {change} | {base['peak_rss_bytes']:.0f} / "
                     f"{head['peak_rss_bytes']:.0f} | {base['proof_bytes']:.0f} / {head['proof_bytes']:.0f} |")
    revisions = result['revisions']
    lines += ['', f"Base: `{revisions['base']}`. Submitted head: `{revisions['head']}`. "
              f"Measured merge: `{revisions['merge']}`. Shared harness: `{revisions['harness']}`.", '',
              f"Warmups per leg: {result['suite']['warmups']}; measured repetitions per leg: "
              f"{result['suite']['repetitions']}. Each invocation is a fresh process; measured pairs alternate order.", '',
              f"Wall receipt: {result['wall_time']['scope']}. "
              f"Total seconds: {result['wall_time']['total_s']}. "
              'Prebuilt execution does not establish the build-inclusive PR budget.', '',
              'No calibrated regression threshold is applied. Raw measurements, verification times, '
              'row counts, hashes, compiler flags, and host settings are in benchmark-results.json.', '']
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
    run_args = argparse.Namespace(**vars(args), base=manifests.get('base'), head=manifests['head'])
    result = run(run_args)
    result['started_at'] = started
    result['wall_time'] = {
        'build_s': sum(item['build_s'] for item in result['builds']),
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
    config, _ = config_cases(args.config, args.tier)
    args.kind, args.case, args.warmups, args.repetitions = None, [], None, None
    result = result_envelope(args, config, None, {})
    result.update(status=args.status, reason=args.reason, finished_at=utc_now())
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
        item.add_argument('--profile', choices=['release', 'ci'], default='release')
    build_parser.add_argument('--revision', required=True)
    for item in (run_parser, suite_parser):
        item.add_argument('--tier', choices=['pr', 'main', 'nightly'], required=True)
        item.add_argument('--kind', choices=['comparison', 'null', 'settings', 'history'])
        item.add_argument('--case', action='append', default=[])
        item.add_argument('--warmups', type=int)
        item.add_argument('--repetitions', type=int)
    run_parser.add_argument('--base', type=Path)
    run_parser.add_argument('--head', type=Path, required=True)
    suite_parser.add_argument('--base-revision')
    suite_parser.add_argument('--head-revision', required=True)
    comment_parser.add_argument('--results', type=Path, required=True)
    comment_parser.add_argument('--out', type=Path, required=True)
    status_parser.add_argument('--tier', choices=['pr', 'main', 'nightly'], required=True)
    status_parser.add_argument('--status', choices=['disabled', 'unsupported', 'failed'], required=True)
    status_parser.add_argument('--reason', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.command == 'render-comment':
            render_comment(json.loads(args.results.read_text()), args.out)
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
