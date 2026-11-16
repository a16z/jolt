"""Exact-base observational references; normalized bytes are lookup-only."""

import datetime as dt
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

METHOD = 'null-median3-symmetric-envelope-v3'
NORMALIZATION = 'elf-no-build-id-comment-v1'
# Percent reporting thresholds by trace ceiling; a row applies from its `from_scale_log2` up to the next row's.
# This is their only definition: selection and every rendered sentence read it. scripts/bench/README.md
# ("Reporting thresholds") gives the rule that derives each value and how often published nulls reach it.
SIZE_THRESHOLDS = (
    {'from_scale_log2': 0, 'group_time': 3.0, 'case_time': 3.0, 'rss': 15.0},
    {'from_scale_log2': 20, 'group_time': 3.0, 'case_time': 4.5, 'rss': 15.0},
    {'from_scale_log2': 22, 'group_time': 2.0, 'case_time': 2.5, 'rss': 9.0},
    {'from_scale_log2': 24, 'group_time': 1.0, 'case_time': 2.0, 'rss': 3.0},
    {'from_scale_log2': 27, 'group_time': 1.0, 'case_time': 1.5, 'rss': 3.0},
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def normalized(path):
    try:
        if 'GNU objcopy' not in subprocess.check_output(['objcopy', '--version'], text=True):
            raise ValueError('GNU objcopy required')
        sections = subprocess.check_output(['readelf', '-SW', str(path)], text=True)
        for name, flags in re.findall(
                r'\[\s*\d+\]\s+(\S+)\s+\S+\s+[\da-f]+\s+[\da-f]+\s+[\da-f]+\s+[\da-f]+\s+([A-Z]*)', sections):
            if name in ('.note.gnu.build-id', '.comment') and set(flags) & {'W', 'X'}:
                raise ValueError('ignored ELF section is writable or executable')
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / 'prover'
            subprocess.run(['objcopy', '--remove-section=.note.gnu.build-id', '--remove-section=.comment',
                            str(path), str(copy)], check=True, capture_output=True)
            return {'method': NORMALIZATION, 'sha256': hashlib.sha256(copy.read_bytes()).hexdigest()}
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def host_identity(host):
    identity = {key: host[key] for key in ('id', 'kernel', 'microcode', 'control_plane')}
    identity['settings'] = {key: host['settings'][key] for key in
        ('boost', 'smt', 'governor', 'transparent_huge_pages', 'swap_total_bytes')}
    return identity


def require_evidence(value):
    if value is None or value == '' or value == {} or value == []:
        raise ValueError('missing identity evidence')
    if isinstance(value, dict):
        for item in value.values():
            require_evidence(item)
    elif isinstance(value, list):
        for item in value:
            require_evidence(item)


def cell_key(result, case):
    base = next(build for build in result['builds'] if build['leg'] == 'base')
    host = host_identity(result['host'])
    if host != host_identity(result['host_after']):
        raise ValueError('host identity changed during measurement')
    build = {key: base[key] for key in ('revision', 'toolchain', 'cargo_version', 'target_cpu',
        'target_triple', 'target_features', 'rustflags', 'profile', 'profile_config',
        'guest_builder_profile', 'cargo_lock_sha256', 'harness_files', 'akita_schedules',
        'runner', 'calibration_helper', 'suite_config_sha256')}
    build['prover'] = base['binaries'][case['pcs']]['normalized']
    guest = base['guests'][case['workload']]
    build['guest_elf_sha256'] = guest['elf']['sha256']
    rows = {name: {sample['metrics'][name] for sample in case['samples']
        if sample['leg'] == 'base' and sample['phase'] == 'measure' and sample['status'] == 'complete'}
        for name in ('trace_rows', 'padded_trace_rows')}
    if any(len(values) != 1 for values in rows.values()):
        raise ValueError('base trace rows are absent or changed')
    key = {'schema': 1, 'method': METHOD, 'normalization': NORMALIZATION, 'metric_contract': 1,
        'host': host, 'build': build, 'cache_policy': result['runtime_cache']['dory'],
        'case': {name: case[name] for name in ('id', 'workload', 'pcs', 'backend', 'physical_cores', 'cpu_ids')},
        'rows': {name: next(iter(values)) for name, values in rows.items()}}
    require_evidence(key)
    if build['prover']['method'] != NORMALIZATION or result['suite']['metric_contract'] != 1:
        raise ValueError('unsupported normalization or metrics')
    key['case'].update(scale_log2=case['scale_log2'], input=case['input'])
    require_evidence(case['input']['sha256'])
    build['guest_builder_rustflags'] = base['guest_builder_rustflags']
    if guest['inputs'][str(case['input']['parameters']['operations'])]['sha256'] != case['input']['sha256']:
        raise ValueError('input bytes differ')
    return key


def size_thresholds(scale_log2):
    """Thresholds for one case and for a size group's mean at a trace ceiling.

    A case with no ceiling gets the largest value of each column.
    """
    if scale_log2 is None:
        row = {name: max(item[name] for item in SIZE_THRESHOLDS) for name in ('group_time', 'case_time', 'rss')}
    else:
        row = [item for item in SIZE_THRESHOLDS if scale_log2 >= item['from_scale_log2']][-1]
    return {'case': {'e2e_s': row['case_time'], 'peak_rss_bytes': row['rss']},
            'group': {'e2e_s': row['group_time'], 'peak_rss_bytes': row['rss']}}


def case_thresholds(scale_log2, observed):
    """A case's thresholds: its size row, raised to each metric's observed envelope when a reference matched."""
    return {metric: max(value, observed[metric]['envelope_pct']) if observed else value
            for metric, value in size_thresholds(scale_log2)['case'].items()}


def size_thresholds_text():
    parts = []
    for row, following in zip(SIZE_THRESHOLDS, [*SIZE_THRESHOLDS[1:], None]):
        start = row['from_scale_log2']
        sizes = (f'2^{start} and above' if following is None else
                 f"below 2^{following['from_scale_log2']}" if row is SIZE_THRESHOLDS[0] else
                 f"2^{start} to 2^{following['from_scale_log2'] - 1}")
        parts.append(f"{sizes}, {row['group_time']:g}% / {row['case_time']:g}% / {row['rss']:g}%")
    return ("Trace-size thresholds (a size group's mean time / one case's time / peak RSS of either): "
            + '; '.join(parts) + '.')


def envelopes(case):
    values = {}
    for metric, unit in [('e2e_s', 'seconds'), ('peak_rss_bytes', 'bytes')]:
        legs = {leg: [s['metrics'][metric] for s in case['samples']
                     if s['leg'] == leg and s['phase'] == 'measure'] for leg in ('base', 'head')}
        if any(len(v) != 10 or any(x <= 0 or not math.isfinite(x) for x in v) for v in legs.values()):
            raise ValueError('reference requires ten positive measured pairs')
        deltas = []
        for offset in range(8):
            a, b = (Fraction(str(statistics.median(legs[leg][offset:offset + 3]))) for leg in ('base', 'head'))
            deltas.append(100 * (max(a / b, b / a) - 1))
        maximum = max(deltas)
        values[metric] = {'unit': unit, 'max_symmetric_window_delta_pct': float(maximum),
                         'envelope_pct': .5 * math.ceil(Fraction(3, 2) * maximum * 2)}
    return values


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timestamp lacks a timezone')
    return parsed


def reference(raw, result, source):
    if (result['kind'] != 'null' or result['status'] != 'complete'
            or result['suite']['warmups'] != 1 or result['suite']['repetitions'] != 10):
        raise ValueError('calibration is not a complete PR-suite ten-pair null')
    builds = [{k: v for k, v in build.items() if k != 'leg'} for build in result['builds']]
    if len(builds) != 2 or builds[0] != builds[1]:
        raise ValueError('calibration legs differ')
    cells = []
    for case in result['cases']:
        if case['status'] != 'complete' or any(s['status'] != 'complete' or s['exit_code'] != 0 for s in case['samples']):
            raise ValueError('calibration has failed processes')
        key = cell_key(result, case)
        cells.append({'key': key, 'key_sha256': digest(key), 'n_pairs': 10, 'n_windows': 8, 'metrics': envelopes(case)})
    if timestamp(result['finished_at']) < timestamp(result['started_at']):
        raise ValueError('calibration times reversed')
    return {'schema': 1, 'method': METHOD, 'cells': cells, 'source': source,
            'raw_sha256': hashlib.sha256(raw).hexdigest(), 'started_at': result['started_at'],
            'finished_at': result['finished_at'],
            'expires_at': (timestamp(result['finished_at']) + dt.timedelta(hours=48)).isoformat()}


def for_case(result, case):
    external = result.get('external_references')
    if external is None:
        return manual_reference(result, case)
    try:
        if result['suite']['warmups'] != 1 or result['suite']['repetitions'] != 3:
            raise ValueError('comparison requires the one-warmup three-pair estimator')
        key = cell_key(result, case)
        matches = []
        for reference in external['references']:
            if not (timestamp(reference['finished_at']) <= timestamp(result['started_at'])
                    <= timestamp(result['finished_at']) <= timestamp(reference['expires_at'])):
                continue
            for cell in reference['cells']:
                if cell['key'] == key and cell['key_sha256'] == digest(key):
                    matches.append((reference, cell))
        if not matches:
            raise ValueError(external.get('reason') or 'no unexpired exact-identity reference')
        reference, cell = max(matches, key=lambda pair: (timestamp(pair[0]['finished_at']),
            *[pair[0]['source'][name] for name in ('plan_run_id', 'plan_run_attempt', 'ingest_run_id', 'ingest_run_attempt')]))
        result.setdefault('comparison_references', {})[case['id']] = {
            'sha256': digest(reference), 'source': reference['source'], 'key_sha256': cell['key_sha256']}
        return cell['metrics'], None
    except (KeyError, ValueError, TypeError, StopIteration) as error:
        return None, f'reference unavailable: {error}'


def manual_reference(result, case):
    calibration = result['suite'].get('calibration') or {}
    if not isinstance(calibration, dict):
        return None, 'calibration is not an object'
    if calibration.get('status') != 'measured':
        return None, calibration.get('reason') or 'reference not measured'
    try:
        for key, value in {'method': METHOD, 'metric_contract': 1,
                           'window_pairs': 3, 'windows_overlap': True, 'independent_windows': False,
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
                        for key in ('version', 'cli_sha256', 'node_sha256')]
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
            for key in ('envelope_pct', 'max_symmetric_window_delta_pct'):
                if (type(value.get(key)) not in (int, float) or not math.isfinite(value[key])
                        or value[key] < 0):
                    raise ValueError(f'missing or invalid {metric}.{key}')
        return reference['metrics'], None
    except (KeyError, TypeError, AttributeError) as error:
        return None, f'missing or invalid calibration evidence: {error}'
    except ValueError as error:
        return None, str(error)
