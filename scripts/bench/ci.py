#!/usr/bin/env python3
"""Admit opt-in PR builds and artifact-only main/nightly measurements."""

import argparse
import contextlib
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
from urllib.parse import quote

import run as bench


TARGET_CPU = 'znver4'
TIERS = ('pr', 'main', 'nightly', 'calibration')
RUN_ONLY_SCOPE = 'artifact-verification-through-results-on-this-host'
LOCAL_SCOPE = 'sequential-build-preparation-through-results-on-this-host'


def git_file(revision, name):
    return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=bench.ROOT)


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


def github(path, *, authenticated=True, media='application/vnd.github+json'):
    headers = {'Accept': media, 'X-GitHub-Api-Version': '2022-11-28'}
    if authenticated:
        headers['Authorization'] = 'Bearer ' + os.environ['GH_TOKEN']
    request = urllib.request.Request('https://api.github.com/repos/a16z/jolt/' + path,
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def labelled_pr(number, *, authenticated):
    pr = github(f'pulls/{number}', authenticated=authenticated)
    if (pr['state'] != 'open' or (pr['head']['repo'] or {}).get('full_name') != 'a16z/jolt'
            or pr['base']['repo']['full_name'] != 'a16z/jolt' or pr['base']['ref'] != 'main'
            or 'benchmark' not in [label['name'] for label in pr['labels']]):
        raise ValueError('benchmark requires an open, labelled, same-repository PR into main')
    latest, page = None, 1
    while True:
        events = github(f'issues/{number}/events?per_page=100&page={page}', authenticated=authenticated)
        for event in events:
            if event['event'] in ('labeled', 'unlabeled') and event['label']['name'] == 'benchmark':
                latest = event
        if len(events) < 100:
            break
        page += 1
    if latest is None or latest['event'] != 'labeled':
        raise ValueError('benchmark label has no current application event')
    admission = {'label_event_id': latest['id'], 'label_actor': latest['actor']['login'],
                 'label_created_at': latest['created_at'], 'head_ref': pr['head']['ref']}
    if authenticated:
        permission = github(f'collaborators/{quote(admission["label_actor"], safe="")}/permission')
        if permission['role_name'] not in ('maintain', 'admin'):
            raise ValueError('benchmark label must be applied by a maintainer or administrator')
    return pr, admission


def recheck_pr(plan):
    # The hosted planner checks permissions. Public reads on the target bind the
    # same label application after queue/lock waits, without passing it a token.
    # The plan pins the base and merge it builds. The API base.sha lags merge^1,
    # and merge_commit_sha changes whenever main advances, so neither is rechecked.
    pr, admission = labelled_pr(plan['provenance']['pr_number'], authenticated=False)
    if admission != plan['admission'] or pr['head']['sha'] != plan['runs'][0]['submitted_head']:
        raise ValueError('PR label, branch or head changed after trusted admission')


def resolve(event_name, event, repository, sha):
    if repository != 'a16z/jolt':
        raise ValueError('benchmark producer must run in a16z/jolt')
    harness = bench.git_revision('HEAD')
    if event_name == 'pull_request_target':
        if event['action'] not in ('labeled', 'synchronize') or (
                event['action'] == 'labeled' and event['label']['name'] != 'benchmark'):
            raise ValueError('only benchmark label applications and subsequent pushes request a PR run')
        number = event['pull_request']['number']
        pr, admission = labelled_pr(number, authenticated=True)
        submitted = event['pull_request']['head']
        if (pr['head']['sha'] != submitted['sha'] or pr['head']['ref'] != submitted['ref']
                or (submitted['repo'] or {}).get('full_name') != repository
                or (event['action'] == 'labeled' and event['sender']['login'] != admission['label_actor'])):
            raise ValueError('the labelled PR request is no longer current')
        bench.capture(['git', 'fetch', '--no-tags', 'origin',
                       '+refs/heads/main:refs/e2e/main',
                       f'+refs/pull/{number}/merge:refs/e2e/merge'])
        merge = bench.git_revision('refs/e2e/merge')
        base = bench.git_revision(f'{merge}^1')
        head = bench.git_revision(f'{merge}^2')
        if head != pr['head']['sha'] or base != bench.git_revision('refs/e2e/main'):
            raise ValueError('the PR merge ref does not contain the current main and submitted head')
        return harness, [{'base': base, 'head': merge, 'submitted_head': head}], number, admission
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
    if event_name not in ('schedule', 'workflow_dispatch'):
        raise ValueError('unsupported benchmark event')
    head = bench.git_revision(sha)
    return harness, [{'base': None, 'head': head, 'submitted_head': head}], None, None


def plan(args):
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    name = os.environ['GITHUB_EVENT_NAME']
    tier = ('pr' if name == 'pull_request_target' else 'main' if name == 'push'
            else event.get('inputs', {}).get('tier', 'nightly'))
    bootstrap = name == 'workflow_dispatch' and tier == 'calibration'
    enabled = os.environ.get('JOLT_E2E_ENABLED') == 'true' or bootstrap
    plan = {'tier': tier, 'enabled': False, 'reason': None, 'target_cpu': TARGET_CPU,
            'push_before': event.get('before'), 'push_after': event.get('after'),
            'harness': None, 'runs': [], 'provenance': bench.provenance()[0],
            'config': bench.bundle_file(args.config.resolve(), bench.ROOT), 'expected_builds': {}}
    try:
        if tier not in TIERS or os.environ['GITHUB_REF'] != 'refs/heads/main':
            raise ValueError('benchmark tiers require main')
        if not enabled:
            raise ValueError('benchmark execution is disabled; complete scripts/bench/README.md before enablement')
        plan['harness'], plan['runs'], pr, plan['admission'] = resolve(
            name, event, os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA'])
        plan['provenance'].update(pr_number=pr)
        if pr is not None:
            actual = github(f'actions/runs/{os.environ["GITHUB_RUN_ID"]}')
            if (actual['event'] != 'pull_request_target' or actual['path'] != bench.WORKFLOW_FILE
                    or actual['head_sha'] != plan['runs'][0]['submitted_head']
                    or actual['run_attempt'] != int(os.environ['GITHUB_RUN_ATTEMPT'])
                    or actual['created_at'] < plan['admission']['label_created_at']):
                raise ValueError('stale or replayed label request; apply the label again or push a new revision')
        harness = plan['harness']
        if git_file(harness, plan['config']['path']) != args.config.read_bytes():
            raise ValueError('suite configuration differs from the committed harness')
        harness_files = bench.harness_files(harness)
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
                    'calibration_helper': {'path': 'scripts/bench/calibration.py',
                        'sha256': hashlib.sha256(git_file(harness, 'scripts/bench/calibration.py')).hexdigest()},
                    'akita_schedules': [{'path': f'akita-schedules/{Path(path).name}',
                                        'sha256': hashlib.sha256(git_file(revision, path)).hexdigest()}
                                       for path in schedules if Path(path).suffix == '.aks'],
                }
                if tier == 'pr':
                    plan.setdefault('own_harness_files', {})[revision] = bench.harness_files(revision)
        plan['enabled'] = True
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        plan['reason'] = str(error)
        status = 'disabled' if not enabled else 'unsupported'
        bench.disabled(argparse.Namespace(config=args.config, tier='pr' if tier == 'calibration' else tier, out=args.out,
                                          status=status, reason=str(error)))
    bench.write_json(Path(args.out) / 'plan.json', plan)
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f'enabled={str(plan["enabled"]).lower()}\n')
        output.write(f'harness={plan["harness"] or os.environ["GITHUB_SHA"]}\n')
        output.write(f'tier={tier}\n')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
        summary.write('End-to-end benchmarks: ' + ('enabled' if plan['enabled'] else plan['reason']) + '\n')
    return 0


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


def prepare(args, *, locked=False):
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
    with (contextlib.nullcontext() if locked else bench.host_lock()), subprocess.Popen(
            ['sh', '-c', f'while :; do\n{snapshot}sleep 10\ndone', *resource_args],
            cwd=bench.ROOT, env=env, start_new_session=True) as resources:
        try:
            for revision in plan['expected_builds']:
                if not re.fullmatch('[0-9a-f]{40}', revision):
                    raise ValueError('invalid planned revision')
                bench.capture(['git', 'fetch', '--no-tags', 'origin', revision], env=env)
                options = argparse.Namespace(config=bench.ROOT / plan['config']['path'], tier=None,
                    out=out / revision, revision=revision, harness_revision=plan['harness'],
                    target_cpu=plan['target_cpu'], profile='release', workload=[], case=[],
                    allow_harness_fallback=plan['tier'] == 'pr')
                manifest = bench.build(options)
                if manifest['status'] != 'complete':
                    raise ValueError(f'build preparation failed for {revision}: {manifest["reason"]}')
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
    _, cases = bench.config_cases(config, 'pr' if plan['tier'] == 'calibration' else plan['tier'])
    manifests = {}
    for revision in revisions:
        entry = index['builds'][revision]
        if entry['path'] != f'{revision}/manifest.json':
            raise ValueError('unexpected build manifest path')
        path, manifest = bench.load_bundle(bench.checked_file(index_path, entry))
        expected = dict(plan['expected_builds'][revision])
        native = manifest['native_harness_revision']
        if native != expected['harness_revision']:
            if (plan['tier'] != 'pr' or native != revision or not manifest.get('harness_fallback_reason')):
                raise ValueError('build uses an unplanned native harness')
            expected['harness_files'] = plan['own_harness_files'][revision]
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f'build differs from the planned input: {key}')
        entries = [manifest['runner'], manifest['calibration_helper'], *manifest['akita_schedules'],
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


def execute(args, *, locked=False):
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
    with (contextlib.nullcontext() if locked else bench.host_lock()):
        if plan['tier'] == 'pr':
            recheck_pr(plan)
        else:
            try:
                calibration_failed = calibration_phase(args, plan)
            except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
                calibration_failed = True
                bench.write_json(out / 'calibration/unavailable.json', {'reason': str(error)})
            if plan['tier'] == 'calibration':
                return int(calibration_failed)
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


def calibration_phase(args, plan):
    tip = plan['runs'][-1]['head']
    started, begin = bench.utc_now(), time.perf_counter()
    manifests = verified_bundles(args, plan, {tip})
    artifacts = {'id': args.artifact_id, 'index_sha256': args.index_sha256,
                 'plan_sha256': bench.sha256(args.plan)}
    failed = False
    for name in ('raw', 'probe') if plan['tier'] == 'calibration' else ('raw',):
        manifest, repetitions, scope = manifests[tip], 10, RUN_ONLY_SCOPE
        build_s = 0
        if name == 'probe':
            started, begin = bench.utc_now(), time.perf_counter()
            preparation = argparse.Namespace(plan=args.plan, out=args.out / 'builds')
            prepare(preparation, locked=True)
            build_s = time.perf_counter() - begin
            local_args = argparse.Namespace(**dict(vars(args), bundles=preparation.out,
                index_sha256=bench.sha256(preparation.out / 'builds.json')))
            manifest = verified_bundles(local_args, plan, {tip})[tip]
            repetitions, scope = 3, LOCAL_SCOPE
        options = argparse.Namespace(config=bench.ROOT / plan['config']['path'], tier='pr',
            out=args.out / 'calibration' / name, base=manifest, head=manifest, kind='null',
            case=[], warmups=1, repetitions=repetitions,
            context={'provenance': plan['provenance'], 'submitted_head': tip, 'is_pr': False})
        result = bench.run(options)
        result['artifacts'] = artifacts
        result['started_at'] = started
        result['wall_time'].update(build_s=build_s, total_s=time.perf_counter() - begin, scope=scope)
        failed |= result['status'] != 'complete'
        bench.write_json(options.out / 'benchmark-results.json', result)
        if name == 'probe':
            bench.write_json(args.out / 'benchmark-results.json', result)
    return failed


def local(args):
    plan = load_plan(args.plan)
    if plan['tier'] != 'pr':
        raise ValueError('local builds are restricted to admitted PR comparisons')
    with bench.host_lock():
        recheck_pr(plan)
        started, begin = bench.utc_now(), time.perf_counter()
        preparation = argparse.Namespace(plan=args.plan, out=args.out / 'builds')
        prepare(preparation, locked=True)
        build_s = time.perf_counter() - begin
        execution = argparse.Namespace(plan=args.plan, out=args.out, bundles=preparation.out,
            index_sha256=bench.sha256(preparation.out / 'builds.json'), artifact_id=None)
        try:
            failed = execute(execution, locked=True)
        finally:
            path = args.out / 'benchmark-results.json'
            if path.exists():
                result = bench.parse_json(path.read_text())
                result['started_at'] = started
                result['wall_time'].update(build_s=build_s, total_s=time.perf_counter() - begin, scope=LOCAL_SCOPE)
                bench.write_json(path, result)
                bench.render_comment(result, args.out / 'pr-comment.md')
    return failed


def dispatch_manifest(plan, run_id, attempt, *, artifact_id=None, index_sha256=None, digests=None):
    if not plan['enabled'] or (str(plan['provenance']['run_id']), str(plan['provenance']['run_attempt'])) != (
            str(run_id), str(attempt)):
        raise ValueError('dispatch requires an enabled plan from this run and attempt')
    pr = None
    if plan['tier'] == 'pr':
        pr = dict(number=plan['provenance']['pr_number'], head_sha=plan['runs'][0]['submitted_head'],
                  **{key: plan['admission'][key] for key in ('label_event_id', 'label_actor')})
    elif (type(artifact_id) is not int or artifact_id <= 0
            or not re.fullmatch('[0-9a-f]{64}', index_sha256 or '')
            or not isinstance(digests, dict) or set(digests) != set(plan['expected_builds'])
            or any(not re.fullmatch('[0-9a-f]{64}', value) for value in digests.values())):
        raise ValueError('main/nightly dispatch requires the hosted build receipts')
    return {'schema': 1, 'tier': plan['tier'], 'run_id': int(run_id), 'run_attempt': int(attempt),
            'plan_artifact': f'jolt-e2e-plan-{run_id}-{attempt}',
            'builds_artifact': None if pr else f'jolt-e2e-builds-{run_id}-{attempt}',
            'check_sha': pr['head_sha'] if pr else plan['runs'][-1]['head'], 'pr': pr,
            'build_artifact_id': None if pr else artifact_id,
            'build_index_sha256': None if pr else index_sha256,
            'manifest_digests': None if pr else digests}


def check_dispatch(manifest, plan):
    expected = dispatch_manifest(plan, manifest['run_id'], manifest['run_attempt'],
        artifact_id=manifest.get('build_artifact_id'), index_sha256=manifest.get('build_index_sha256'),
        digests=manifest.get('manifest_digests'))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('dispatch differs from its trusted plan')


def dispatch(args):
    plan = bench.parse_json(args.plan.read_text())
    manifest = dispatch_manifest(plan, os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT'],
        artifact_id=int(os.environ['BUILD_ARTIFACT_ID']) if os.environ.get('BUILD_ARTIFACT_ID') else None,
        index_sha256=os.environ.get('BUILD_INDEX_SHA256'),
        digests=bench.parse_json(os.environ.get('BUILD_MANIFEST_DIGESTS') or 'null'))
    bench.write_json(args.out / 'dispatch.json', manifest)


def target(args):
    plan = load_plan(args.plan)
    manifest = bench.parse_json(args.dispatch.read_text())
    check_dispatch(manifest, plan)
    for key, name in [('run_id', 'GITHUB_RUN_ID'), ('run_attempt', 'GITHUB_RUN_ATTEMPT'),
                      ('workflow', 'GITHUB_WORKFLOW'), ('event', 'GITHUB_EVENT_NAME'), ('ref', 'GITHUB_REF')]:
        os.environ[name] = str(plan['provenance'][key])
    if plan['tier'] == 'pr':
        return local(args)
    return execute(argparse.Namespace(plan=args.plan, out=args.out, bundles=args.bundles,
        artifact_id=manifest['build_artifact_id'], index_sha256=manifest['build_index_sha256']))


def verify(args):
    plan = load_plan(args.plan)
    manifests = verified_bundles(args, plan, plan['expected_builds'])
    print(json.dumps({'artifact_id': args.artifact_id, 'index_sha256': args.index_sha256,
                      'manifests': {revision: str(path) for revision, path in manifests.items()}}))



def finalize(args):
    root = args.out / 'benchmark-results.json'
    plan = json.loads(args.plan.read_text())
    manifest = bench.parse_json(args.dispatch.read_text()) if getattr(args, 'dispatch', None) else {}
    index = args.out / 'builds/builds.json'
    artifacts = {'id': manifest.get('build_artifact_id'),
                 'index_sha256': bench.sha256(index) if plan['tier'] == 'pr' and index.exists()
                 else manifest.get('build_index_sha256'), 'plan_sha256': bench.sha256(args.plan)}
    for item in plan['runs']:
        path = (root if item == plan['runs'][-1]
                else args.out / 'commits' / item['head'] / 'benchmark-results.json')
        if path.exists():
            partial = json.loads(path.read_text())
            if partial['status'] == 'incomplete':
                partial['reason'] = partial['reason'] or 'suite interrupted before all planned samples finished'
                partial['finished_at'] = bench.utc_now()
                partial['artifacts'] = artifacts
                if 'wall_time' in partial:
                    partial['wall_time']['scope'] = LOCAL_SCOPE if plan['tier'] in ('pr', 'calibration') else RUN_ONLY_SCOPE
                bench.write_json(path, partial)
    if root.exists():
        result = bench.parse_json(root.read_text())
        result['calibration_evidence'] = {name: bench.bundle_file(path, args.out) for name in ('raw', 'probe')
            if (path := args.out / 'calibration' / name / 'benchmark-results.json').exists()}
        bench.write_json(root, result)
        return
    options = argparse.Namespace(config=bench.ROOT / plan['config']['path'], tier='pr' if plan['tier'] == 'calibration' else plan['tier'], out=args.out,
                                  status='incomplete', reason='the planned target job did not produce a root result')
    result = bench.disabled(options)
    result['provenance'] = plan['provenance']
    result['run_id'] = plan['provenance'].get('run_id')
    if plan['runs']:
        item = plan['runs'][-1]
        result['revisions'] = {'base': item['head'] if plan['tier'] == 'calibration' else item['base'],
                              'head': item['submitted_head'],
                              'merge': item['head'] if plan['tier'] == 'pr' else None,
                              'harness': plan['harness']}
    result['kind'] = ('null' if plan['tier'] == 'calibration' else
                      'comparison' if plan['tier'] == 'pr' else 'history')
    if plan['tier'] in ('pr', 'calibration'):
        result['wall_time']['scope'] = LOCAL_SCOPE
    else:
        result['wall_time']['scope'] = RUN_ONLY_SCOPE
    result['artifacts'] = artifacts
    bench.write_json(root, result)


def main():
    signal.signal(signal.SIGTERM, bench.interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan')
    planning.add_argument('--out', type=Path, required=True)
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
    finalization.add_argument('--dispatch', type=Path)
    local_run = commands.add_parser('local', help='build and prove an admitted PR under one host lock')
    local_run.add_argument('--plan', type=Path, required=True)
    local_run.add_argument('--out', type=Path, required=True)
    dispatching = commands.add_parser('dispatch', help='publish the App executor manifest')
    dispatching.add_argument('--plan', type=Path, required=True)
    dispatching.add_argument('--out', type=Path, required=True)
    targeting = commands.add_parser('target', help='execute and finalize one App-dispatched plan')
    targeting.add_argument('--dispatch', type=Path, required=True)
    targeting.add_argument('--plan', type=Path, required=True)
    targeting.add_argument('--bundles', type=Path)
    targeting.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return {'prepare': prepare, 'verify': verify, 'execute': execute,
            'local': local, 'plan': plan, 'finalize': finalize,
            'dispatch': dispatch, 'target': target}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
