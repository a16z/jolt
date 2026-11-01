#!/usr/bin/env python3
"""Resolve pinned inputs, prepare hosted artifacts, then measure without building."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import tomllib
import urllib.request

import run as bench


REQUEST_WORKFLOW = 'End-to-end prover benchmark requests'
REQUEST_FILE = '.github/workflows/e2e-bench-request.yml'
TARGET_CPU = 'znver4'
RUN_ONLY_SCOPE = 'artifact-verification-through-results-on-this-host'


def git_file(revision, name):
    return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=bench.ROOT)


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


def github(path):
    request = urllib.request.Request('https://api.github.com/repos/a16z/jolt/' + path,
        headers={'Accept': 'application/vnd.github+json',
                 'Authorization': 'Bearer ' + os.environ['GH_TOKEN']})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve(event_name, event, repository, sha):
    harness = bench.git_revision('HEAD')
    request_run_id = None
    if event_name == 'pull_request':
        pr = event['pull_request']
        if pr['head']['repo']['full_name'] != repository or pr['base']['ref'] != 'main':
            raise ValueError('build-only requests must be same-repository PRs targeting main')
        merge, base, head = bench.git_revision(sha), pr['base']['sha'], pr['head']['sha']
        parents = bench.capture(['git', 'rev-list', '--parents', '-n', '1', merge]).split()[1:]
        if merge != harness or parents != [base, head]:
            raise ValueError('build-only checkout does not match the event merge and its parents')
        return harness, [{'base': base, 'head': merge, 'submitted_head': head}], event['number'], None
    if event_name == 'workflow_run':
        request_run_id = event['workflow_run']['id']
        request = github(f'actions/runs/{request_run_id}')
        if (request['name'] != REQUEST_WORKFLOW or request['path'] != REQUEST_FILE
                or request['event'] != 'pull_request' or request['conclusion'] != 'success'
                or request['head_repository']['full_name'] != repository):
            raise ValueError('only successful same-repository PR requests may use the benchmark runner')
        if len(request['pull_requests']) != 1:
            raise ValueError('the request must identify exactly one PR')
        number = request['pull_requests'][0]['number']
        pr = github(f'pulls/{number}')
        if (pr['state'] != 'open' or pr['head']['repo']['full_name'] != repository
                or pr['base']['ref'] != 'main' or pr['head']['sha'] != request['head_sha']):
            raise ValueError('the same-repository PR request is no longer current')
        bench.capture(['git', 'fetch', '--no-tags', 'origin',
                       '+refs/heads/main:refs/e2e/main',
                       f'+refs/pull/{number}/merge:refs/e2e/merge'])
        merge = bench.git_revision('refs/e2e/merge')
        base = bench.git_revision(f'{merge}^1')
        head = bench.git_revision(f'{merge}^2')
        if head != pr['head']['sha'] or base != bench.git_revision('refs/e2e/main'):
            raise ValueError('the PR merge ref does not contain the current main and submitted head')
        return harness, [{'base': base, 'head': merge, 'submitted_head': head}], number, request_run_id
    if event_name == 'push':
        before, after = event['before'], event['after']
        if not re.fullmatch('[0-9a-f]{40}', before) or before == '0' * 40:
            raise ValueError('main history has no valid previous commit; explicit backfill is required')
        after = bench.git_revision(after)
        subprocess.run(['git', 'merge-base', '--is-ancestor', before, after], cwd=bench.ROOT, check=True)
        commits = bench.capture(['git', 'rev-list', '--first-parent', '--reverse', f'{before}..{after}']).splitlines()
        if not commits:
            raise ValueError('the main push contains no new first-parent commits')
        return harness, [{'base': None, 'head': commit, 'submitted_head': commit} for commit in commits], None, None
    head = bench.git_revision(sha)
    return harness, [{'base': None, 'head': head, 'submitted_head': head}], None, None


def plan(args):
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    name = os.environ['GITHUB_EVENT_NAME']
    build_only = args.build_only
    if build_only != (name == 'pull_request'):
        raise ValueError('only the unprivileged PR entrypoint may request a build-only plan')
    tier = ('pr' if name in ('workflow_run', 'pull_request') else 'main' if name == 'push'
            else event.get('inputs', {}).get('tier', 'nightly'))
    enabled = build_only or os.environ.get('JOLT_E2E_ENABLED') == 'true'
    plan = {'tier': tier, 'enabled': False, 'reason': None, 'target_cpu': TARGET_CPU,
            'push_before': event.get('before'), 'push_after': event.get('after'),
            'harness': None, 'runs': [], 'provenance': bench.provenance()[0],
            'config': bench.bundle_file(args.config.resolve(), bench.ROOT), 'expected_builds': {}}
    try:
        if not enabled:
            raise ValueError('benchmark runner is disabled; an admin must complete scripts/bench/README.md')
        plan['harness'], plan['runs'], pr, request_id = resolve(
            name, event, os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA'])
        plan['provenance'].update(pr_number=pr, request_run_id=request_id)
        if build_only:
            plan['provenance']['workflow_file'] = REQUEST_FILE
        harness = plan['harness']
        if git_file(harness, plan['config']['path']) != args.config.read_bytes():
            raise ValueError('suite configuration differs from the committed harness')
        paths = bench.capture(['git', 'ls-tree', '-r', '--name-only', harness, '--', *bench.HARNESS_PATHS]).splitlines()
        harness_files = {path: hashlib.sha256(git_file(harness, path)).hexdigest() for path in paths}
        for item in plan['runs']:
            for revision in (item['base'], item['head']):
                if not revision:
                    continue
                if not re.fullmatch('[0-9a-f]{40}', revision):
                    raise ValueError('invalid planned revision')
                schedules = bench.capture(['git', 'ls-tree', '-r', '--name-only', revision,
                                           '--', 'crates/jolt-akita/schedules']).splitlines()
                plan['expected_builds'][revision] = {
                    'revision': revision, 'harness_revision': harness, 'status': 'complete',
                    'target_cpu': TARGET_CPU, 'rustflags': f'-C target-cpu={TARGET_CPU}',
                    'target_triple': 'x86_64-unknown-linux-gnu', 'profile': 'release',
                    'profile_config': tomllib.loads(git_file(revision, 'Cargo.toml').decode())['profile']['release'],
                    'guest_builder_profile': 'ci', 'guest_builder_rustflags': '',
                    'source_cargo_lock_sha256': hashlib.sha256(git_file(revision, 'Cargo.lock')).hexdigest(),
                    'harness_files': harness_files, 'suite_config_sha256': plan['config']['sha256'],
                    'runner': {'path': 'scripts/bench/run.py',
                               'sha256': hashlib.sha256(git_file(harness, 'scripts/bench/run.py')).hexdigest()},
                    'akita_schedules': [{'path': f'akita-schedules/{Path(path).name}',
                                        'sha256': hashlib.sha256(git_file(revision, path)).hexdigest()}
                                       for path in schedules if Path(path).suffix == '.aks'],
                }
        plan['enabled'] = True
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        plan['reason'] = str(error)
        status = 'disabled' if not enabled else 'unsupported'
        bench.disabled(argparse.Namespace(config=args.config, tier=tier, out=args.out,
                                          status=status, reason=str(error)))
    bench.write_json(Path(args.out) / 'plan.json', plan)
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f'enabled={str(plan["enabled"]).lower()}\n')
        output.write(f'harness={plan["harness"] or os.environ["GITHUB_SHA"]}\n')
        output.write(f'tier={tier}\n')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
        summary.write('End-to-end benchmarks: ' + ('enabled' if plan['enabled'] else plan['reason']) + '\n')
    return int(build_only and not plan['enabled'])


def load_plan(path):
    plan = bench.parse_json(path.read_text())
    if not plan['enabled']:
        raise ValueError('disabled plan cannot be executed')
    if bench.git_revision('HEAD') != plan['harness']:
        raise ValueError('checkout is not the exact planned harness revision')
    for key, name in [('run_id', 'GITHUB_RUN_ID'), ('run_attempt', 'GITHUB_RUN_ATTEMPT')]:
        if os.environ.get(name) and str(plan['provenance'][key]) != os.environ[name]:
            raise ValueError('the plan is from a different run or attempt; rerun all jobs')
    bench.checked_file(bench.ROOT / 'plan.json', plan['config'])
    return plan


def prepare(args):
    plan = load_plan(args.plan)
    started, begin = bench.utc_now(), time.perf_counter()
    out = args.out.resolve()
    builds, digests = {}, {}
    snapshot = """date -u '+build resources %Y-%m-%dT%H:%M:%SZ'
free -b
df -B1 "$1"
awk '$1 == "oom_kill" { print "kernel", $0 }' /proc/vmstat
"""
    resource_args = ['build-resources', str(bench.cache_root())]
    env = bench.build_environment()
    with bench.host_lock(), subprocess.Popen(
            ['sh', '-c', f'while :; do\n{snapshot}sleep 10\ndone', *resource_args],
            cwd=bench.ROOT, env=env, start_new_session=True) as resources:
        try:
            for revision in plan['expected_builds']:
                if not re.fullmatch('[0-9a-f]{40}', revision):
                    raise ValueError('invalid planned revision')
                bench.capture(['git', 'fetch', '--no-tags', 'origin', revision], env=env)
                options = argparse.Namespace(config=bench.ROOT / plan['config']['path'], tier=None,
                    out=out / revision, revision=revision, harness_revision=plan['harness'],
                    target_cpu=plan['target_cpu'], profile='release', workload=[], case=[])
                manifest = bench.build(options)
                if manifest['status'] != 'complete':
                    raise ValueError(f'hosted preparation failed for {revision}: {manifest["reason"]}')
                builds[revision] = bench.bundle_file(options.out / 'manifest.json', out)
                digests[revision] = manifest_digest(manifest)
        finally:
            bench.stop_child(resources)
            subprocess.run(['sh', '-c', snapshot, *resource_args], cwd=bench.ROOT, env=env, check=False)
            print('Kernel warnings/errors (read-only; access may be unavailable):', flush=True)
            subprocess.run(['dmesg', '--level=err,warn'], env=env, check=False)
    index = {'plan_sha256': bench.sha256(args.plan), 'builds': builds,
             'started_at': started, 'finished_at': bench.utc_now(), 'elapsed_s': time.perf_counter() - begin}
    bench.write_json(out / 'builds.json', index)
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
            output.write(f'index_sha256={bench.sha256(out / "builds.json")}\n')
            output.write(f'manifest_digests={json.dumps(digests, sort_keys=True)}\n')


def verified_bundles(args, plan, revisions):
    index_path = args.bundles / 'builds.json'
    if bench.sha256(index_path) != args.index_sha256:
        raise ValueError('build index differs from the hosted job output')
    index = bench.parse_json(index_path.read_text())
    if index['plan_sha256'] != bench.sha256(args.plan) or set(index['builds']) != set(plan['expected_builds']):
        raise ValueError('artifacts do not match the exact plan and source revision set')
    inventory = bench.parse_json((bench.ROOT / plan['config']['path']).read_text())
    inputs, required_pcs = bench.build_inputs(inventory)
    config = getattr(args, 'config', None) or bench.ROOT / plan['config']['path']
    _, cases = bench.config_cases(config, plan['tier'])
    manifests = {}
    for revision in revisions:
        entry = index['builds'][revision]
        if entry['path'] != f'{revision}/manifest.json':
            raise ValueError('unexpected build manifest path')
        path, manifest = bench.load_bundle(bench.checked_file(index_path, entry))
        for key, value in plan['expected_builds'][revision].items():
            if manifest.get(key) != value:
                raise ValueError(f'build differs from the planned input: {key}')
        entries = [manifest['runner'], *manifest['akita_schedules'],
                   {'path': 'scripts/bench/suites.json', 'sha256': plan['config']['sha256']}]
        if set(manifest['binaries']) != required_pcs or set(manifest['guests']) != set(inputs):
            raise ValueError('bundle does not contain the planned binary and guest set')
        files = [(manifest['binaries'][pcs], f'jolt-prover-{pcs}') for pcs in required_pcs]
        for workload, counts in inputs.items():
            guest = manifest['guests'][workload]
            if set(guest['inputs']) != set(map(str, counts)):
                raise ValueError('bundle does not contain the complete planned input set')
            files.append((guest['elf'], f'guests/{workload}/guest.elf'))
            files.extend((guest['inputs'][str(count)], f'guests/{workload}/input-{count}.bin') for count in counts)
        for file, expected_path in files:
            if file['path'] != expected_path:
                raise ValueError('artifact file is not the expected binary or guest input')
            entries.append(file)
        for case in cases:
            if case.get('unsupported_reason'):
                continue
            if (case['pcs'] not in required_pcs or
                    case['input']['parameters']['operations'] not in inputs.get(case['workload'], set())):
                raise ValueError('measurement suite requires an input absent from the planned inventory')
        for file in {entry['path']: entry for entry in entries}.values():
            bench.checked_file(path, file)
        # Actions artifacts preserve bytes, not executable modes. Only verified
        # native binaries become executable; artifact scripts are never imported.
        for pcs in required_pcs:
            (path.parent / f'jolt-prover-{pcs}').chmod(0o755)
        manifests[revision] = path
    return manifests


def execute(args):
    plan = load_plan(args.plan)
    config = (getattr(args, 'config', None) or bench.ROOT / plan['config']['path']).resolve()
    if getattr(args, 'config', None) and git_file(plan['harness'], str(config.relative_to(bench.ROOT))) != config.read_bytes():
        raise ValueError('measurement configuration is not committed in the planned harness')
    out = args.out.resolve()
    tip = plan['runs'][-1]['head']
    index = {'schema_version': 1, 'push_before': plan['push_before'], 'push_after': tip,
             'documents': []}
    for item in plan['runs']:
        path = (Path('benchmark-results.json') if item['head'] == tip
                else Path('commits') / item['head'] / 'benchmark-results.json')
        index['documents'].append({'revision': item['head'], 'path': str(path)})
    if plan['tier'] == 'main':
        bench.write_json(out / 'index.json', index)
    failed = False
    with bench.host_lock():
        for item, entry in zip(plan['runs'], index['documents']):
            started, begin = bench.utc_now(), time.perf_counter()
            manifests = verified_bundles(args, plan, {rev for rev in (item['base'], item['head']) if rev})
            verification_s = time.perf_counter() - begin
            run_out = (out / entry['path']).parent
            options = argparse.Namespace(config=config, tier=plan['tier'],
                out=run_out, base=manifests.get(item['base']), head=manifests[item['head']],
                kind=None, case=[], warmups=None, repetitions=None,
                context={'provenance': plan['provenance'], 'submitted_head': item['submitted_head'],
                         'is_pr': plan['tier'] == 'pr'})
            result = bench.run(options)
            result['started_at'] = started
            result['artifacts'] = {'id': args.artifact_id, 'index_sha256': args.index_sha256,
                                   'plan_sha256': bench.sha256(args.plan)}
            result['wall_time'].update(build_s=0, artifact_verification_s=verification_s,
                                      total_s=time.perf_counter() - begin, scope=RUN_ONLY_SCOPE)
            result['finished_at'] = bench.utc_now()
            bench.write_json(run_out / 'benchmark-results.json', result)
            if item['base']:
                bench.render_comment(result, run_out / 'pr-comment.md')
            failed |= result['status'] not in ('complete', 'unsupported')
    return int(failed)


def verify(args):
    plan = load_plan(args.plan)
    manifests = verified_bundles(args, plan, plan['expected_builds'])
    print(json.dumps({'artifact_id': args.artifact_id, 'index_sha256': args.index_sha256,
                      'manifests': {revision: str(path) for revision, path in manifests.items()}}))



def finalize(args):
    root = args.out / 'benchmark-results.json'
    plan = json.loads(args.plan.read_text())
    for item in plan['runs']:
        path = (root if item == plan['runs'][-1]
                else args.out / 'commits' / item['head'] / 'benchmark-results.json')
        if path.exists():
            partial = json.loads(path.read_text())
            if partial['status'] == 'incomplete':
                partial['reason'] = partial['reason'] or 'suite interrupted before all planned samples finished'
                partial['finished_at'] = bench.utc_now()
                bench.write_json(path, partial)
    if root.exists():
        return
    options = argparse.Namespace(config=bench.ROOT / plan['config']['path'], tier=plan['tier'], out=args.out,
                                  status='incomplete', reason='the planned target job did not produce a root result')
    result = bench.disabled(options)
    result['provenance'] = plan['provenance']
    if plan['runs']:
        item = plan['runs'][-1]
        result['revisions'] = {'base': item['base'], 'head': item['submitted_head'],
                              'merge': item['head'] if plan['tier'] == 'pr' else None,
                              'harness': plan['harness']}
    result['kind'] = 'comparison' if plan['tier'] == 'pr' else 'history'
    bench.write_json(root, result)


def main():
    signal.signal(signal.SIGTERM, bench.interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan')
    planning.add_argument('--out', type=Path, required=True)
    planning.add_argument('--build-only', action='store_true')
    planning.add_argument('--config', type=Path, default=bench.DEFAULT_CONFIG)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--plan', type=Path, required=True)
    preparation.add_argument('--out', type=Path, required=True)
    execution = commands.add_parser('execute')
    execution.add_argument('--out', type=Path, required=True)
    execution.add_argument('--config', type=Path, help='committed candidate suite for a separately labelled measurement')
    verification = commands.add_parser('verify', help='check bytes and executable modes; never build or prove')
    for item in (execution, verification):
        item.add_argument('--plan', type=Path, required=True)
        item.add_argument('--bundles', type=Path, required=True)
        item.add_argument('--index-sha256', required=True)
        item.add_argument('--artifact-id', type=int, required=True)
    finalization = commands.add_parser('finalize')
    finalization.add_argument('--plan', type=Path, required=True)
    finalization.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return {'prepare': prepare, 'verify': verify, 'execute': execute,
            'plan': plan, 'finalize': finalize}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
