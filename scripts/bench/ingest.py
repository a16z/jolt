#!/usr/bin/env python3
"""Ingest executor bytes as data using the default branch's validator."""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.request
from urllib.parse import urlsplit
import zipfile

import ci
import calibration
import publication
import run as bench

APP = '0xandoroid-ai-agent[bot]'
MAX_RESULT = 1024**3
MAX_JSON = 8 * 1024**2


class DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != 'https':
            raise ValueError('download redirect must use HTTPS')
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        redirected.remove_header('Authorization')
        return redirected


def download(url, limit, *, authenticated=False):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or not parsed.hostname:
        raise ValueError('download must use credential-free HTTPS')
    headers = {'Accept': 'application/vnd.github+json',
               'User-Agent': 'jolt-e2e-ingest/1 (+https://github.com/a16z/jolt)'}
    if authenticated:
        if parsed.hostname != 'api.github.com':
            raise ValueError('artifact authentication is limited to GitHub')
        headers['Authorization'] = 'Bearer ' + os.environ['GH_TOKEN']
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(DownloadRedirect()).open(request, timeout=60) as response:
        chunks, total = [], 0
        while chunk := response.read(min(1024 * 1024, limit + 1 - total)):
            total += len(chunk)
            if total > limit:
                raise ValueError('download exceeds size cap')
            chunks.append(chunk)
    return b''.join(chunks)


def artifact_json(run_id, attempt, prefix, filename):
    artifacts, page = [], 1
    while True:
        batch = ci.github(f'actions/runs/{run_id}/artifacts?per_page=100&page={page}')['artifacts']
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    matches = [item for item in artifacts if item['name'] == f'{prefix}-{run_id}-{attempt}']
    if len(matches) != 1:
        raise ValueError('expected exactly one plan-run artifact with the exact attempt name')
    artifact = matches[0]
    if (artifact['expired'] or artifact['workflow_run']['id'] != run_id
            or artifact['size_in_bytes'] > MAX_JSON):
        raise ValueError('plan-run artifact is expired, oversized, or from another run')
    raw = download(f'https://api.github.com/repos/a16z/jolt/actions/artifacts/{artifact["id"]}/zip',
                   MAX_JSON, authenticated=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != filename or members[0].file_size > MAX_JSON:
            raise ValueError('unexpected plan-run artifact contents')
        content = archive.read(members[0])
    return bench.parse_json(content.decode()), content, artifact['id']


def trusted_plan(run_id, attempt):
    actual = ci.github(f'actions/runs/{run_id}/attempts/{attempt}')
    if (actual['id'] != run_id or actual['run_attempt'] != attempt
            or actual['repository']['full_name'] != 'a16z/jolt'
            or (actual['head_repository'] or {}).get('full_name') != 'a16z/jolt'
            or actual['name'] != bench.WORKFLOW or actual['path'] != bench.WORKFLOW_FILE
            or actual['status'] != 'completed' or actual['conclusion'] != 'success'):
        raise ValueError('plan producer is not a successful trusted workflow attempt')
    plan, raw, plan_id = artifact_json(run_id, attempt, 'jolt-e2e-plan', 'plan.json')
    dispatch, dispatch_raw, dispatch_id = artifact_json(run_id, attempt, 'jolt-e2e-dispatch', 'dispatch.json')
    ci.check_dispatch(dispatch, plan)
    if plan['tier'] == 'calibration' and actual['event'] != 'workflow_dispatch':
        raise ValueError('bootstrap calibration requires workflow_dispatch on main')
    if (dispatch['run_id'], dispatch['run_attempt']) != (run_id, attempt):
        raise ValueError('dispatch is from another run attempt')
    prov = plan['provenance']
    if (prov['workflow'] != bench.WORKFLOW or prov['workflow_file'] != bench.WORKFLOW_FILE
            or prov['event'] != actual['event'] or prov['ref'] != 'refs/heads/main'):
        raise ValueError('plan provenance differs from its producer')
    if plan['tier'] == 'pr':
        prs = actual['pull_requests']
        item = plan['runs'][0]
        if (actual['event'] != 'pull_request_target' or len(plan['runs']) != 1
                or (prs and not any(pr['number'] == prov['pr_number'] and pr['base']['ref'] == 'main' for pr in prs))
                or actual['head_branch'] != plan['admission']['head_ref']
                or actual['head_sha'] != item['submitted_head']):
            raise ValueError('PR plan is not bound to the admitted head and main base')
        commit = ci.github(f'commits/{item["head"]}')
        if [parent['sha'] for parent in commit['parents']] != [item['base'], item['submitted_head']]:
            raise ValueError('PR merge parents differ from the plan')
    else:
        if (actual['event'] not in ('push', 'schedule', 'workflow_dispatch') or actual['head_branch'] != 'main'
                or dispatch['check_sha'] != actual['head_sha']):
            raise ValueError('history plan is not bound to main')
        comparison = ci.github(f'compare/{actual["head_sha"]}...main')
        if comparison['status'] not in ('identical', 'ahead'):
            raise ValueError('measured revision is not on main')
        builds = ci.github(f'actions/artifacts/{dispatch["build_artifact_id"]}')
        if (builds['name'] != dispatch['builds_artifact'] or builds['expired']
                or builds['workflow_run']['id'] != run_id):
            raise ValueError('build artifact differs from the hosted dispatch receipt')
    config = ci.github(f'contents/{plan["config"]["path"]}?ref={plan["harness"]}')
    if config['type'] != 'file' or config['encoding'] != 'base64':
        raise ValueError('suite configuration must be a committed file')
    config_bytes = base64.b64decode(config['content'])
    if hashlib.sha256(config_bytes).hexdigest() != plan['config']['sha256']:
        raise ValueError('plan suite hash differs from its harness')
    receipt = {'plan_artifact_id': plan_id, 'dispatch_artifact_id': dispatch_id,
               'plan_sha256': hashlib.sha256(raw).hexdigest(),
               'dispatch_sha256': hashlib.sha256(dispatch_raw).hexdigest()}
    return plan, dispatch, config_bytes, receipt


def result_files(raw):
    files, total = {}, 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (name.is_absolute() or not name.parts or '..' in name.parts or '\\' in member.filename
                    or name.as_posix() != member.filename
                    or member.filename in files or stat.S_ISLNK(mode)
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or member.is_dir()
                    or (name.parts[0] == 'builds' and not (len(name.parts) == 3 and name.name == 'build.log'))
                    or 'runtime-cache' in name.parts):
                raise ValueError('unsafe, excluded, or duplicate result member')
            total += member.file_size
            if total > 4 * 1024**3 or member.file_size > 128 * 1024**2 or len(files) >= 100000:
                raise ValueError('result ZIP exceeds extraction cap')
            files[member.filename] = archive.read(member)
    return files


def validate_results(files, plan, dispatch, config, receipt):
    documents = []
    for item in plan['runs']:
        path = ('benchmark-results.json' if item == plan['runs'][-1]
                else f'commits/{item["head"]}/benchmark-results.json')
        if path not in files:
            if path == 'benchmark-results.json':
                raise ValueError('missing root result')
            documents.append(None)
            continue
        if len(files[path]) > MAX_JSON:
            raise ValueError('result JSON exceeds 8 MiB')
        result = bench.parse_json(files[path].decode())
        paired = plan['tier'] == 'pr'
        bootstrap = plan['tier'] == 'calibration'
        artifacts = {'id': dispatch['build_artifact_id'], 'index_sha256': dispatch['build_index_sha256'],
                     'plan_sha256': receipt['plan_sha256']}
        if paired:
            index_hash = result.get('artifacts', {}).get('index_sha256')
            if index_hash is not None and not re.fullmatch('[0-9a-f]{64}', index_hash):
                raise ValueError('invalid local build index digest')
            artifacts['index_sha256'] = index_hash
        expected = {'tier': 'pr' if bootstrap else plan['tier'], 'provenance': plan['provenance'], 'artifacts': artifacts,
            'manifest_digests': dispatch['manifest_digests'], 'expected_builds': plan['expected_builds'],
            'own_harness_files': plan.get('own_harness_files', {}),
            'revisions': {'base': item['base'], 'head': item['submitted_head'],
                          'merge': item['head'] if paired else None, 'harness': plan['harness']}}
        if bootstrap:
            expected.update(kind='null', scope=ci.LOCAL_SCOPE, manifest_digests=None)
            expected['revisions'].update(base=item['head'], merge=None)
        publication.validate(result, expected, config)
        documents.append(result)
    if plan['tier'] == 'main':
        expected_index = {'schema_version': 1, 'push_before': plan['push_before'],
            'push_after': plan['runs'][-1]['head'], 'documents': [
                {'revision': item['head'], 'path': 'benchmark-results.json' if item == plan['runs'][-1]
                 else f'commits/{item["head"]}/benchmark-results.json'} for item in plan['runs']]}
        if 'index.json' in files and bench.parse_json(files['index.json'].decode()) != expected_index:
            raise ValueError('result index differs from the planned commit set')
        files['index.json'] = (json.dumps(expected_index) + '\n').encode()
    return documents


def calibration_reference(files, root, plan, dispatch, config, receipt):
    descriptor = root.get('calibration_evidence', {}).get('raw')
    if descriptor is None:
        return None
    path = 'calibration/raw/benchmark-results.json'
    if descriptor != {'path': path, 'sha256': hashlib.sha256(files[path]).hexdigest()}:
        raise ValueError('calibration raw descriptor mismatch')
    raw = files[path]
    result = bench.parse_json(raw.decode())
    tip = plan['runs'][-1]['head']
    expected = {'tier': 'pr', 'kind': 'null', 'sampling': {'warmups': 1, 'repetitions': 10},
        'scope': ci.RUN_ONLY_SCOPE, 'provenance': plan['provenance'],
        'artifacts': {'id': dispatch['build_artifact_id'], 'index_sha256': dispatch['build_index_sha256'],
                      'plan_sha256': receipt['plan_sha256']},
        'expected_builds': plan['expected_builds'], 'manifest_digests': dispatch['manifest_digests'],
        'revisions': {'base': tip, 'head': tip, 'merge': None, 'harness': plan['harness']}}
    publication.validate(result, expected, config)
    if plan['tier'] == 'calibration':
        probe_path = 'calibration/probe/benchmark-results.json'
        probe_descriptor = root.get('calibration_evidence', {}).get('probe')
        if probe_descriptor != {'path': probe_path, 'sha256': hashlib.sha256(files[probe_path]).hexdigest()}:
            raise ValueError('bootstrap probe descriptor mismatch')
        probe = bench.parse_json(files[probe_path].decode())
        expected.update(sampling={'warmups': 1, 'repetitions': 3}, scope=ci.LOCAL_SCOPE,
                        manifest_digests=None)
        publication.validate(probe, expected, config)
        if probe['status'] != 'complete':
            raise ValueError('bootstrap probe did not complete')
        if {k: v for k, v in root.items() if k != 'calibration_evidence'} != probe:
            raise ValueError('bootstrap root differs from probe')
    return calibration.reference(raw, result, dict(receipt,
        tier=plan['tier'], build_index_sha256=dispatch['build_index_sha256'],
        manifest_digests=dispatch['manifest_digests']))


def check_result(job_id, documents, exit_code, summary):
    if exit_code or any(item is not None and (item['status'] == 'failed' or any(
            case['status'] == 'failed' for case in item['cases'])) for item in documents):
        conclusion, title = 'failure', 'Benchmark execution failed'
    elif any(item is None or item['status'] != 'complete' for item in documents):
        conclusion, title = 'neutral', 'Benchmark results are partial'
    elif any(item['tier'] == 'pr' and any(calibration.for_case(item, case)[1]
             for case in item['cases'] if case['status'] == 'complete') for item in documents):
        conclusion, title = 'neutral', 'Benchmark results validated; comparison reference unavailable'
    else:
        conclusion, title = 'success', 'Benchmark results validated'
    return {'schema': 1, 'job_id': job_id, 'conclusion': conclusion, 'title': title, 'summary': summary[:60000]}


def render_pr_comment(result, summary_path, details_url, exit_code=0):
    rows = bench.render_comment(result, summary_path)
    lines = ['<!-- jolt-e2e-benchmark-comment -->', '## End-to-end CPU benchmark', '',
             f'[Full results]({details_url}) · '
             '[Benchmark history](https://a16z.github.io/jolt/dev/prover-history/)', '',
             f"Base: `{result['revisions']['base']}`. Submitted head: `{result['revisions']['head']}`. "
             f"Measured merge: `{result['revisions']['merge']}`.", '']
    if exit_code or result['status'] != 'complete':
        lines.append('Comparison unavailable; see full results for execution status.')
    elif rows:
        lines += ['Rows with more than 3% change in time or peak resident set size (RSS) are shown; '
                  'positive means an increase. '
                  'PCS is the polynomial commitment scheme: Akita is lattice-based; Dory is curve-based.', '',
                  '| Workload / PCS | Physical cores | End-to-end base / head (s) | Signed time change | Peak RSS base / head (bytes) | Signed RSS change |',
                  '|---|---:|---:|---:|---:|---:|', *rows]
    else:
        lines.append('No change above 3% in end-to-end proving time or peak RSS.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job-id', 'result-url', 'result-sha256'):
        parser.add_argument('--' + name, required=True)
    for name in ('plan-run-id', 'plan-run-attempt', 'executor-exit-code'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get('GITHUB_ACTOR') != APP or os.environ.get('GITHUB_REF') != 'refs/heads/main'
            or args.job_id != f'{args.plan_run_id}-{args.plan_run_attempt}'
            or not re.fullmatch('[0-9a-f]{64}', args.result_sha256)):
        raise ValueError('ingest requires the App on main and an exact job/digest identity')
    plan, dispatch, config, receipt = trusted_plan(args.plan_run_id, args.plan_run_attempt)
    raw = download(args.result_url, MAX_RESULT)
    if hashlib.sha256(raw).hexdigest() != args.result_sha256:
        raise ValueError('result download SHA-256 mismatch')
    files = result_files(raw)
    for name in ('calibration/reference.json', 'calibration/unavailable.json', 'comparison.json', 'ingest.json'):
        files.pop(name, None)
    documents = validate_results(files, plan, dispatch, config, receipt)
    root = documents[-1]
    receipt.update(schema=1, job_id=args.job_id, plan_run_id=args.plan_run_id,
        plan_run_attempt=args.plan_run_attempt, result_sha256=args.result_sha256,
        executor_exit_code=args.executor_exit_code, ingest_run_id=int(os.environ['GITHUB_RUN_ID']),
        ingest_run_attempt=int(os.environ['GITHUB_RUN_ATTEMPT']))
    reference = None
    if plan['tier'] != 'pr':
        try:
            reference = calibration_reference(files, root, plan, dispatch, config, receipt)
            if reference:
                files['calibration/reference.json'] = calibration.encoded(reference)
            elif plan['tier'] == 'calibration':
                raise ValueError('bootstrap has no completed calibration raw evidence')
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            files['calibration/unavailable.json'] = calibration.encoded({'reason': str(error)})
    else:
        root = publication.resolve_references(root)
        documents[-1] = root
    summary_path = args.out / 'summary.md'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if plan['tier'] == 'calibration':
        summary = ('Calibration reference SHA-256: `' + calibration.digest(reference) + '`.' if reference
                   else 'Calibration unavailable: ' + bench.parse_json(files['calibration/unavailable.json'])['reason'])
        if args.executor_exit_code:
            summary = f'Benchmark executor exited {args.executor_exit_code}. ' + summary
    elif root['tier'] == 'pr':
        details_url = (f'https://github.com/a16z/jolt/actions/runs/{receipt["ingest_run_id"]}'
                       f'/attempts/{receipt["ingest_run_attempt"]}')
        comment = render_pr_comment(root, summary_path, details_url, args.executor_exit_code)
        summary = summary_path.read_text()
        files['comparison.json'] = calibration.encoded({'references': root.get('comparison_references', {}),
            'availability': {case['id']: calibration.for_case(root, case)[1] for case in root['cases']},
            'catalog_revision': root.get('external_references', {}).get('catalog_revision'), 'summary': summary})
        files['pr-comment.md'] = comment.encode()
    else:
        summary = '\n'.join(f'{item["revisions"]["head"]}: {item["status"]}' if item else
                            'Planned commit result unavailable.' for item in documents)
    for name, content in files.items():
        target = args.out / 'result' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    bench.write_json(args.out / 'result/ingest.json', receipt)
    check = check_result(args.job_id, documents, args.executor_exit_code, summary)
    if plan['tier'] == 'calibration':
        success = reference is not None and args.executor_exit_code == 0
        check.update(conclusion='success' if success else 'failure',
                     title='Calibration reference validated' if success else 'Calibration failed')
    bench.write_json(args.out / 'check/check.json', check)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
            output.write(summary + '\n')
    if plan['tier'] == 'pr' and plan['provenance'].get('pr_number') and os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'pr_number={plan["provenance"]["pr_number"]}\n')


if __name__ == '__main__':
    main()
