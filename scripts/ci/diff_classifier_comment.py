#!/usr/bin/env python3
"""Render the sticky diff-classification comment for each open pull request whose head is the head
of the workflow run that triggered .github/workflows/diff-classifier-comment.yml, and write them to
$GITHUB_OUTPUT as `pulls=<JSON list of {number, head, body}>` for the job that posts them.

The triggering run can be a pull request's own copy of any workflow, so only the head GitHub
recorded for it is used. Pull requests are re-read from the API, and the head commit is fetched as
git objects and read by diff_classifier.py, never checked out. A comment body holds fixed text and
counts; branch and file names stay out of it and out of the log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from diff_classifier import DiffClassifier, format_markdown


def log(message: str) -> None:
    # One prefixed line: a file name can hold a newline, and a line starting with `::` in an error
    # that quotes it would be a workflow command.
    print("diff-classifier-comment: " + message.replace("\r", " ").replace("\n", " "))


def gh_api(path: str) -> list[dict]:
    return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True).stdout)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def pulls_at_head(run: dict, repository: str, api) -> list[dict]:
    """Open pull requests into `repository` whose current head is the run's recorded head."""
    source = run.get("head_repository")
    if not source:
        return []
    branch, sha = run["head_branch"], run["head_sha"]
    head = quote(f"{source['owner']['login']}:{branch}", safe="")
    # GitHub ignores a `head` filter it cannot parse and lists every open pull request, so the
    # filter narrows the request and this match decides.
    return [pull for pull in api(f"repos/{repository}/pulls?state=open&per_page=100&head={head}")
            if (pull["head"].get("repo") or {}).get("id") == source["id"]
            and pull["head"]["ref"] == branch and pull["head"]["sha"] == sha]


def render(pull: dict) -> str:
    sha = pull["head"]["sha"]
    git("fetch", "--quiet", "--no-tags", "origin", sha)
    base = git("merge-base", f"refs/remotes/origin/{pull['base']['ref']}", sha)
    totals, _ = DiffClassifier(".", base, sha).classify()
    return format_markdown(totals)


def main() -> int:
    run = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"]
    rendered = []
    for pull in pulls_at_head(run, os.environ["GITHUB_REPOSITORY"], gh_api):
        try:
            body = render(pull)
        # A pull request shapes these failures, so they cost it the comment instead of failing the run.
        except Exception as error:
            log(f"pull request #{pull['number']} not classified: {type(error).__name__}: {error}")
            continue
        rendered.append({"number": pull["number"], "head": pull["head"]["sha"], "body": body})
    log(f"run {run['id']} at head {run['head_sha']}: rendered pull requests {[pull['number'] for pull in rendered]}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"pulls={json.dumps(rendered)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
