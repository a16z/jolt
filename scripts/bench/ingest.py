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
    headers = {'Accept': 'application/vnd.github+json'}
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
    if (dispatch['run_id'], dispatch['run_attempt']) != (run_id, attempt):
        raise ValueError('dispatch is from another run attempt')
    prov = plan['provenance']
    if (prov['workflow'] != bench.WORKFLOW or prov['workflow_file'] != bench.WORKFLOW_FILE
            or prov['event'] != actual['event'] or prov['ref'] != 'refs/heads/main'):
        raise ValueError('plan provenance differs from its producer')
    if plan['tier'] == 'pr':
        prs = actual['pull_requests']
        item = plan['runs'][0]
        if (actual['event'] != 'pull_request_target' or len(plan['runs']) != 1 or len(prs) != 1
                or prs[0]['number'] != prov['pr_number'] or prs[0]['base']['ref'] != 'main'
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
                    or name.parts[0] == 'builds' or 'runtime-cache' in name.parts):
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
        artifacts = {'id': dispatch['build_artifact_id'], 'index_sha256': dispatch['build_index_sha256'],
                     'plan_sha256': receipt['plan_sha256']}
        if paired:
            index_hash = result.get('artifacts', {}).get('index_sha256')
            if index_hash is not None and not re.fullmatch('[0-9a-f]{64}', index_hash):
                raise ValueError('invalid local build index digest')
            artifacts['index_sha256'] = index_hash
        expected = {'tier': plan['tier'], 'provenance': plan['provenance'], 'artifacts': artifacts,
            'manifest_digests': dispatch['manifest_digests'], 'expected_builds': plan['expected_builds'],
            'own_harness_files': plan.get('own_harness_files', {}),
            'revisions': {'base': item['base'], 'head': item['submitted_head'],
                          'merge': item['head'] if paired else None, 'harness': plan['harness']}}
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


def check_result(job_id, documents, exit_code, summary):
    if exit_code or any(item is not None and (item['status'] == 'failed' or any(
            case['status'] == 'failed' for case in item['cases'])) for item in documents):
        conclusion, title = 'failure', 'Benchmark execution failed'
    elif any(item is None or item['status'] != 'complete' for item in documents):
        conclusion, title = 'neutral', 'Benchmark results are partial'
    elif any(item['tier'] == 'pr' and any(bench.calibration_for_case(item, case)[1]
             for case in item['cases'] if case['status'] == 'complete') for item in documents):
        conclusion, title = 'neutral', 'Benchmark results validated; comparison reference unavailable'
    else:
        conclusion, title = 'success', 'Benchmark results validated'
    return {'schema': 1, 'job_id': job_id, 'conclusion': conclusion, 'title': title, 'summary': summary[:60000]}


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
    documents = validate_results(files, plan, dispatch, config, receipt)
    root = documents[-1]
    summary_path = args.out / 'summary.md'
    if root['tier'] == 'pr':
        bench.render_comment(root, summary_path)
        summary = summary_path.read_text()
    else:
        summary = '\n'.join(f'{item["revisions"]["head"]}: {item["status"]}' if item else
                            'Planned commit result unavailable.' for item in documents)
    receipt.update(schema=1, job_id=args.job_id, plan_run_id=args.plan_run_id,
        plan_run_attempt=args.plan_run_attempt, result_sha256=args.result_sha256,
        executor_exit_code=args.executor_exit_code, ingest_run_id=int(os.environ['GITHUB_RUN_ID']),
        ingest_run_attempt=int(os.environ['GITHUB_RUN_ATTEMPT']))
    for name, content in files.items():
        target = args.out / 'result' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    bench.write_json(args.out / 'result/ingest.json', receipt)
    bench.write_json(args.out / 'check/check.json', check_result(args.job_id, documents,
                     args.executor_exit_code, summary))


if __name__ == '__main__':
    main()
