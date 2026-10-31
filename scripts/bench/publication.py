#!/usr/bin/env python3
"""Validate a PR result against GitHub records before rendering trusted Markdown."""

import argparse
import base64
import hashlib
from pathlib import Path
import re
import os

import ci
import run as bench

STATUSES = {'complete', 'failed', 'incomplete', 'unsupported', 'missing'}


def validate(result, expected, config_bytes):
    if result.get('schema_version') != 1 or result.get('repository') != 'a16z/jolt':
        raise ValueError('unsupported schema or repository')
    if result['tier'] != 'pr' or result['kind'] != 'comparison':
        raise ValueError('only full PR comparisons can update the sticky comment')
    if result['status'] not in {'complete', 'failed', 'incomplete', 'unsupported'}:
        raise ValueError('result does not describe executed measurements')
    for key in ('workflow', 'workflow_file', 'run_id', 'run_attempt', 'event', 'ref', 'pr_number'):
        if str(result['provenance'][key]) != str(expected['provenance'][key]):
            raise ValueError(f'workflow provenance mismatch: {key}')
    if result['provenance']['workflow'] != bench.WORKFLOW or result['provenance']['workflow_file'] != bench.WORKFLOW_FILE:
        raise ValueError('unexpected producer workflow')
    if result['provenance'].get('request_run_id') != expected['provenance'].get('request_run_id'):
        raise ValueError('PR request provenance mismatch')
    for key in ('base', 'head', 'merge', 'harness'):
        value = result['revisions'][key]
        if not re.fullmatch('[0-9a-f]{40}', value or '') or value != expected['revisions'][key]:
            raise ValueError(f'revision mismatch: {key}')
    if str(result['run_id']) != str(expected['provenance']['run_id']):
        raise ValueError('run identity differs from the actual GitHub run')
    config = bench.parse_json(config_bytes)
    if result['suite']['config_sha256'] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError('suite is not the configuration at the measured harness commit')
    tier = config['tiers']['pr']
    for key in ('warmups', 'repetitions'):
        if result['suite'][key] != tier[key]:
            raise ValueError(f'unapproved PR sampling override: {key}')
    if (result['suite']['metric_contract'] != 1 or result['suite']['id'] != config['id']
            or result['suite'].get('selected_cases')):
        raise ValueError('PR suite identity or case selection differs')
    if result['wall_time']['scope'] != 'sequential-build-preparation-through-results-on-this-host':
        raise ValueError('PR publication requires a fresh build-inclusive receipt')
    for key in ('build_s', 'benchmark_s', 'total_s'):
        value = result['wall_time'][key]
        if value is None and result['status'] == 'complete':
            raise ValueError('complete run is missing a wall-time boundary')
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not bench.math.isfinite(value) or value < 0):
            raise ValueError('invalid wall-time receipt')
    wall = result['wall_time']
    if all(wall[key] is not None for key in ('build_s', 'benchmark_s', 'total_s')):
        if wall['total_s'] < wall['build_s'] + wall['benchmark_s']:
            raise ValueError('total wall interval does not include build and benchmark intervals')
    expected_cases = {case['id']: case for case in tier['cases']}
    cases = result['cases']
    if len(cases) != len(expected_cases) or {case['id'] for case in cases} != set(expected_cases):
        raise ValueError('expected PR cases are missing or duplicated')
    builds = {build['leg']: build for build in result['builds']}
    if len(result['builds']) != 2 or set(builds) != {'base', 'head'}:
        raise ValueError('expected exactly one build manifest per leg')
    for leg, revision in [('base', expected['revisions']['base']), ('head', expected['revisions']['merge'])]:
        build = builds[leg]
        if build['revision'] != revision or build['harness_revision'] != expected['revisions']['harness']:
            raise ValueError('build manifest is not for the measured revisions')
    for key in ('harness_files', 'rustflags', 'target_cpu', 'profile'):
        if builds['base'][key] != builds['head'][key]:
            raise ValueError(f'paired build mismatch: {key}')
    topology = {row['cpu']: row for row in result['host']['topology']}
    orders = set()
    for case in cases:
        spec = expected_cases[case['id']]
        if case['physical_cores'] not in (1, 8, 24) or case['backend'] != 'optimized':
            raise ValueError('invalid core count or backend')
        for key in ('workload', 'pcs', 'backend', 'physical_cores', 'scale_log2'):
            if case[key] != spec[key]:
                raise ValueError(f'case identity differs: {key}')
        if case['input']['parameters'] != spec['input']['parameters'] or case['input']['seed'] != spec['input']['seed']:
            raise ValueError('fixed inputs differ from the declared suite')
        if case['status'] not in STATUSES:
            raise ValueError('unknown case status')
        if case['status'] == 'unsupported' and not spec.get('unsupported_reason'):
            raise ValueError('a supported case cannot be relabeled unsupported')
        if case['status'] in ('missing', 'unsupported') and not case['samples']:
            continue
        ids = case['cpu_ids']
        if len(ids) != case['physical_cores'] or len(set(ids)) != len(ids):
            raise ValueError('CPU affinity does not match the physical-core count')
        physical = {(topology[cpu]['socket'], topology[cpu]['core']) for cpu in ids
                    if topology[cpu]['online'] and topology[cpu].get('allowed', True)}
        if len(physical) != case['physical_cores']:
            raise ValueError('affinity contains SMT siblings or unavailable CPUs')
        observed = set()
        for sample in case['samples']:
            leg, phase = sample['leg'], sample['phase']
            if leg not in builds or phase not in ('warmup', 'measure') or sample['status'] not in STATUSES:
                raise ValueError('invalid sample identity or status')
            expected_revision = builds[leg]['revision']
            if sample['revision'] != expected_revision:
                raise ValueError('sample is not for its actual measured revision')
            count = tier['warmups'] if phase == 'warmup' else tier['repetitions']
            repetition = sample['repetition']
            if type(repetition) is not int or not 0 <= repetition < count or sample['pair_index'] != repetition:
                raise ValueError('invalid repetition or pair index')
            key = (leg, phase, repetition)
            if type(sample['order']) is not int or sample['order'] < 0 or key in observed or sample['order'] in orders:
                raise ValueError('duplicate sample or execution order')
            observed.add(key)
            orders.add(sample['order'])
            if sample['status'] == 'complete':
                if sample['exit_code'] != 0 or builds[leg]['status'] != 'complete':
                    raise ValueError('successful sample has failed execution or build')
                metrics = sample['metrics']
                bench.validate_metrics(dict(metrics, workload=case['workload'], pcs=case['pcs'],
                                            backend=case['backend'], scale_log2=case['scale_log2']), case)
                for name in ('process_s', 'peak_rss_bytes'):
                    value = metrics[name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not bench.math.isfinite(value) or value <= 0:
                        raise ValueError(f'invalid {name}')
                if metrics['process_s'] < metrics['e2e_s']:
                    raise ValueError('process duration is shorter than the proving pipeline')
                inputs = builds[leg]['guests'][case['workload']]['inputs']
                if inputs[str(case['input']['parameters']['operations'])]['sha256'] != case['input']['sha256']:
                    raise ValueError('sample input hash differs from its build manifest')
            elif sample['metrics'] is not None:
                raise ValueError('failed or missing sample contains fabricated metrics')
        if case['status'] == 'complete':
            if len(observed) != 2 * (tier['warmups'] + tier['repetitions']) or any(
                    sample['status'] != 'complete' for sample in case['samples']):
                raise ValueError('complete case has missing or failed samples')
            actual_order = [(s['leg'], s['phase'], s['repetition']) for s in case['samples']]
            expected_order = [(s['leg'], s['phase'], s['repetition']) for s in
                              bench.sample_order(True, tier['warmups'], tier['repetitions'])]
            if actual_order != expected_order:
                raise ValueError('paired processes did not use the declared alternating order')
    if result['status'] == 'complete' and (not any(case['status'] == 'complete' for case in cases)
            or any(case['status'] not in ('complete', 'unsupported') for case in cases)):
        raise ValueError('complete run has no results or contains incomplete cases')
    return result


def expected_from_github(plan, plan_artifact_id, result_artifact_id):
    run_id, attempt = os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT']
    actual = ci.github(f'actions/runs/{run_id}')
    if (actual['name'] != bench.WORKFLOW or actual['path'] != bench.WORKFLOW_FILE
            or actual['event'] != 'workflow_run' or actual['head_branch'] != 'main'
            or actual['head_repository']['full_name'] != 'a16z/jolt'
            or actual['head_sha'] != plan['harness'] or actual['run_attempt'] != int(attempt)):
        raise ValueError('actual producer is not the planned trusted main-branch run')
    for artifact_id, prefix in [(plan_artifact_id, 'jolt-e2e-plan'), (result_artifact_id, 'jolt-e2e')]:
        artifact = ci.github(f'actions/artifacts/{artifact_id}')
        if (artifact['name'] != f'{prefix}-{run_id}-{attempt}' or artifact['expired']
                or artifact['workflow_run']['id'] != int(run_id)):
            raise ValueError('artifact is not from the actual producer run and attempt')
    prov = plan['provenance']
    if str(prov['run_id']) != run_id or str(prov['run_attempt']) != attempt or len(plan['runs']) != 1:
        raise ValueError('stale plan or unexpected number of PR comparisons')
    number = prov['pr_number']
    if type(number) is not int:
        raise ValueError('invalid PR number')
    item = plan['runs'][0]
    pr = ci.github(f'pulls/{number}')
    if (pr['state'] != 'open' or pr['head']['repo']['full_name'] != 'a16z/jolt'
            or pr['head']['sha'] != item['submitted_head'] or pr['base']['sha'] != item['base']
            or pr['base']['ref'] != 'main' or pr['merge_commit_sha'] != item['head']):
        return None
    request = ci.github(f'actions/runs/{prov["request_run_id"]}')
    if (request['name'] != ci.REQUEST_WORKFLOW or request['path'] != ci.REQUEST_FILE
            or request['event'] != 'pull_request' or request['conclusion'] != 'success'
            or request['head_repository']['full_name'] != 'a16z/jolt'
            or request['head_sha'] != item['submitted_head']
            or [p['number'] for p in request['pull_requests']] != [number]):
        raise ValueError('request is not the current same-repository PR run')
    commit = ci.github(f'commits/{item["head"]}')
    if [parent['sha'] for parent in commit['parents']] != [item['base'], item['submitted_head']]:
        raise ValueError('measured merge has different parents')
    config = ci.github(f'contents/scripts/bench/suites.json?ref={plan["harness"]}')
    if config['encoding'] != 'base64' or config['type'] != 'file':
        raise ValueError('expected a committed suite configuration file')
    expected = {'provenance': prov,
                'revisions': {'base': item['base'], 'head': item['submitted_head'],
                              'merge': item['head'], 'harness': plan['harness']}}
    return expected, base64.b64decode(config['content'], validate=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-artifact-id', type=int, required=True)
    parser.add_argument('--result-artifact-id', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.results.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('PR result exceeds the 8 MiB JSON limit')
    expected = expected_from_github(bench.parse_json(args.plan.read_text()), args.plan_artifact_id,
                                    args.result_artifact_id)
    if expected is None:
        print('The PR or its main base has advanced; no stale comment will be posted.')
        return
    result = validate(bench.parse_json(args.results.read_text()), *expected)
    bench.render_comment(result, args.out)
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f'publish=true\npr_number={expected[0]["provenance"]["pr_number"]}\n')
        output.write(f'head={expected[0]["revisions"]["head"]}\n')
        output.write(f'base={expected[0]["revisions"]["base"]}\n')


if __name__ == '__main__':
    main()
