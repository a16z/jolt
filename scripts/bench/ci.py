#!/usr/bin/env python3
"""Resolve exact GitHub revisions and invoke the same suite command used manually."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import urllib.request

import run as bench


REQUEST_WORKFLOW = 'End-to-end prover benchmark requests'
REQUEST_FILE = '.github/workflows/e2e-bench-request.yml'


def github(path):
    request = urllib.request.Request('https://api.github.com/repos/a16z/jolt/' + path,
        headers={'Accept': 'application/vnd.github+json',
                 'Authorization': 'Bearer ' + os.environ['GH_TOKEN']})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve(event_name, event, repository, sha):
    harness = bench.git_revision('HEAD')
    request_run_id = None
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
    tier = ('pr' if name == 'workflow_run' else 'main' if name == 'push'
            else event.get('inputs', {}).get('tier', 'nightly'))
    enabled = os.environ.get('JOLT_E2E_ENABLED') == 'true'
    cpu = os.environ.get('JOLT_E2E_TARGET_CPU', '')
    plan = {'tier': tier, 'enabled': False, 'reason': None, 'target_cpu': cpu,
            'push_before': event.get('before'), 'push_after': event.get('after'),
            'harness': None, 'runs': [], 'provenance': bench.provenance()[0]}
    try:
        if not enabled:
            raise ValueError('benchmark runner is disabled; an admin must complete scripts/bench/README.md')
        if not cpu or cpu == 'native' or not re.fullmatch('[a-zA-Z0-9_.+-]+', cpu):
            raise ValueError('an explicit target CPU supported by the observed target ISA is not configured')
        plan['harness'], plan['runs'], pr, request_id = resolve(
            name, event, os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA'])
        plan['provenance'].update(pr_number=pr, request_run_id=request_id)
        plan['enabled'] = True
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        plan['reason'] = str(error)
        status = 'disabled' if not enabled or not cpu else 'unsupported'
        bench.disabled(argparse.Namespace(config=bench.DEFAULT_CONFIG, tier=tier, out=args.out,
                                          status=status, reason=str(error)))
    bench.write_json(Path(args.out) / 'plan.json', plan)
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f'enabled={str(plan["enabled"]).lower()}\n')
        output.write(f'harness={plan["harness"] or os.environ["GITHUB_SHA"]}\n')
        output.write(f'tier={tier}\n')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
        summary.write('End-to-end benchmarks: ' + ('enabled' if plan['enabled'] else plan['reason']) + '\n')


def execute(args):
    plan = json.loads(args.plan.read_text())
    if not plan['enabled']:
        raise ValueError('disabled plan cannot be executed')
    if bench.git_revision('HEAD') != plan['harness']:
        raise ValueError('checkout is not the exact planned harness revision')
    for key, name in [('run_id', 'GITHUB_RUN_ID'), ('run_attempt', 'GITHUB_RUN_ATTEMPT')]:
        if os.environ.get(name) and str(plan['provenance'][key]) != os.environ[name]:
            raise ValueError('the plan is from a different run or attempt; rerun all jobs')
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
    for item in plan['runs']:
        for revision in (item['base'], item['head']):
            if revision:
                if not re.fullmatch('[0-9a-f]{40}', revision):
                    raise ValueError('invalid planned revision')
                bench.capture(['git', 'fetch', '--no-tags', 'origin', revision])
    with bench.host_lock():
        for item, entry in zip(plan['runs'], index['documents']):
            run_out = (out / entry['path']).parent
            options = argparse.Namespace(config=bench.DEFAULT_CONFIG, tier=plan['tier'],
                out=run_out, base_revision=item['base'], head_revision=item['head'],
                harness_revision=plan['harness'], target_cpu=plan['target_cpu'], profile='release',
                kind=None, case=[], warmups=None, repetitions=None,
                context={'provenance': plan['provenance'], 'submitted_head': item['submitted_head'],
                         'is_pr': plan['tier'] == 'pr'})
            result = bench.suite(options)
            failed |= result['status'] not in ('complete', 'unsupported')
    return int(failed)



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
    options = argparse.Namespace(config=bench.DEFAULT_CONFIG, tier=plan['tier'], out=args.out,
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
    execution = commands.add_parser('execute')
    execution.add_argument('--plan', type=Path, required=True)
    execution.add_argument('--out', type=Path, required=True)
    finalization = commands.add_parser('finalize')
    finalization.add_argument('--plan', type=Path, required=True)
    finalization.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return {'execute': execute, 'plan': plan, 'finalize': finalize}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
