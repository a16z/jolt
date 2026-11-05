#!/usr/bin/env python3
"""Validate executor results against the hosted plan before publication."""

import hashlib
import re

import ci
import run as bench

STATUSES = {'complete', 'failed', 'incomplete', 'unsupported', 'missing'}


def validate(result, expected, config_bytes):
    if result.get('schema_version') != 1 or result.get('repository') != 'a16z/jolt':
        raise ValueError('unsupported schema or repository')
    paired = expected['tier'] == 'pr'
    if result['tier'] != expected['tier'] or result['kind'] != ('comparison' if paired else 'history'):
        raise ValueError('result tier or purpose differs from the plan')
    if result['status'] not in {'complete', 'failed', 'incomplete', 'unsupported'}:
        raise ValueError('result does not describe executed measurements')
    for key in ('workflow', 'workflow_file', 'run_id', 'run_attempt', 'event', 'ref', 'pr_number'):
        if str(result['provenance'][key]) != str(expected['provenance'][key]):
            raise ValueError(f'workflow provenance mismatch: {key}')
    if result['provenance']['workflow'] != bench.WORKFLOW or result['provenance']['workflow_file'] != bench.WORKFLOW_FILE:
        raise ValueError('unexpected producer workflow')
    for key in ('base', 'head', 'merge', 'harness'):
        value = result['revisions'][key]
        if value != expected['revisions'][key] or (value is not None and not re.fullmatch('[0-9a-f]{40}', value)):
            raise ValueError(f'revision mismatch: {key}')
    if str(result['run_id']) != str(expected['provenance']['run_id']):
        raise ValueError('run identity differs from the actual GitHub run')
    config = bench.parse_json(config_bytes)
    if result['suite']['config_sha256'] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError('suite is not the configuration at the measured harness commit')
    for key in ('calibration', 'calibration_sha256'):
        if result['suite'].get(key) != config.get(key):
            raise ValueError(f'{key} differs from the pinned suite configuration')
    tier = config['tiers'][expected['tier']]
    for key in ('warmups', 'repetitions'):
        if result['suite'][key] != tier[key]:
            raise ValueError(f'unapproved sampling override: {key}')
    if (result['suite']['metric_contract'] != 1 or result['suite']['id'] != config['id']
            or result['suite'].get('selected_cases')):
        raise ValueError('suite identity or case selection differs')
    if result.get('artifacts') != expected['artifacts']:
        raise ValueError('result uses different build index or plan bytes')
    if result['wall_time']['scope'] != (ci.LOCAL_SCOPE if paired else ci.RUN_ONLY_SCOPE):
        raise ValueError('wall-time scope differs from the execution tier')
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
        raise ValueError('expected cases are missing or duplicated')
    builds = {build['leg']: build for build in result['builds']}
    legs = {'base': expected['revisions']['base'], 'head': expected['revisions']['merge']} if paired else {
        'head': expected['revisions']['head']}
    if (len(builds) != len(result['builds']) or not set(builds) <= set(legs)
            or (result['status'] == 'complete' and set(builds) != set(legs))):
        raise ValueError('build legs differ from the planned measurement')
    for leg, build in builds.items():
        revision = legs[leg]
        if build['revision'] != revision or build['harness_revision'] != expected['revisions']['harness']:
            raise ValueError('build manifest is not for the measured revisions')
        native = build['native_harness_revision']
        if native != build['harness_revision'] and (not paired or native != revision
                or not build.get('harness_fallback_reason')):
            raise ValueError('build uses an unplanned native harness')
        if expected.get('expected_builds'):
            planned = dict(expected['expected_builds'][revision])
            if native != build['harness_revision']:
                planned['harness_files'] = expected['own_harness_files'][revision]
            if any(build.get(key) != value for key, value in planned.items()):
                raise ValueError('build differs from the planned native inputs')
        if expected.get('manifest_digests') is not None:
            manifest = {key: value for key, value in build.items() if key != 'leg'}
            if ci.manifest_digest(manifest) != expected['manifest_digests'][revision]:
                raise ValueError('result build manifest differs from the builder output')
    if paired and set(builds) == set(legs):
        for key in ('rustflags', 'target_cpu', 'profile', 'profile_config'):
            if builds['base'][key] != builds['head'][key]:
                raise ValueError(f'paired build mismatch: {key}')
    if result['host'] is None and any(case['samples'] for case in cases):
        raise ValueError('measured samples have no host identity')
    topology = {row['cpu']: row for row in (result['host'] or {}).get('topology', [])}
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
            if (type(repetition) is not int or not 0 <= repetition < count
                    or sample['pair_index'] != (repetition if paired else None)):
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
            if len(observed) != len(legs) * (tier['warmups'] + tier['repetitions']) or any(
                    sample['status'] != 'complete' for sample in case['samples']):
                raise ValueError('complete case has missing or failed samples')
            actual_order = [(s['leg'], s['phase'], s['repetition']) for s in case['samples']]
            expected_order = [(s['leg'], s['phase'], s['repetition']) for s in
                              bench.sample_order(paired, tier['warmups'], tier['repetitions'])]
            if actual_order != expected_order:
                raise ValueError('paired processes did not use the declared alternating order')
    if result['status'] == 'complete' and (not any(case['status'] == 'complete' for case in cases)
            or any(case['status'] not in ('complete', 'unsupported') for case in cases)):
        raise ValueError('complete run has no results or contains incomplete cases')
    return result
