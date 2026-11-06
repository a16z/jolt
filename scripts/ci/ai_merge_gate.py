#!/usr/bin/env python3
"""Evaluate a pull request into main: approve its head as github-actions[bot] when the policy admits
the agent path, withdraw that approval otherwise, and publish the gate's labels and status.

    ai_merge_gate.py resolve      # write pr=<number> for the triggering event to $GITHUB_OUTPUT
    ai_merge_gate.py evaluate N   # evaluate pull request N
    ai_merge_gate.py refresh N    # clear stale gate labels and sync merge conflicts for N
    ai_merge_gate.py refresh-all  # refresh labels for open PRs into the default branch

The policy, its truth table and the admin setup are in .github/ai-merge-gate.md; `Gate.decide`
implements that table in order. Every PR input is read from the GitHub API when the gate runs.
The trusted preparation job exports verified base/head trees and a complete on-disk diff.
A separate job runs Codex exec in its native read-only sandbox; this publisher never executes PR bytes.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from diff_classifier import parse_diff
from classifier_agent import AgentFailure, EFFORT, MODEL, validate_verdict

ROOT = Path(__file__).resolve().parents[2]
GITHUB_API = "https://api.github.com"
GATE = "ai-merge-gate"
CLASSIFIER = "ai-merge-gate/classifier"
# Starts the body of each approving review the gate submits; only those does it ever dismiss.
APPROVAL = "<!-- ai-merge-gate/approval -->"
WORKFLOW = ".github/workflows/ai-merge-gate.yml"
# The GitHub Actions app, whose installation token GITHUB_TOKEN is.
ACTIONS_APP_ID = 15368
# GITHUB_TOKEN writes as github-actions[bot], whose login GraphQL reports without the suffix.
PUBLISHER = "github-actions"
RISK_LABELS = {f"risk-{level}" for level in range(1, 5)}
NEEDS_MAINTAINER = "needs-maintainer-review"
MERGE_ELIGIBLE = "merge-eligible"
NO_REVIEWER = "no-reviewer-assigned"
MERGE_CONFLICT = "merge-conflict"
MANAGED_LABELS = frozenset(RISK_LABELS | {NEEDS_MAINTAINER, MERGE_ELIGIBLE, NO_REVIEWER})
# REST users/0xandoroid-ai-agent[bot] and GraphQL node lookup agree on this identity.
PIKA_BOT_ID = "BOT_kgDOFAtqNQ"
LOGIN = re.compile(r"[A-Za-z0-9-]{1,39}")
ATTESTATION = re.compile(
    r"This pull request #([1-9][0-9]*) has been reviewed at commit ([0-9a-f]{40}) "
    r"using skill (.+?) and model (.+?) with (.+?) reasoning effort, and no issues were found\."
)
# The only review models and efforts an attestation may name, spelled as .github/ai-merge-gate.md
# lists them; any model may pair with any effort.
REVIEW_MODELS = ("GPT-6 Astra", "Claude Opus 5.5")
REVIEW_EFFORTS = ("max", "xhigh")
LEDGER = re.compile(r"<!-- ai-merge-gate/attempts (\{.*\}) -->")
# The run-name of .github/workflows/ai-merge-gate-events.yml, which GitHub reports as display_title.
RUN_NAME = re.compile(r"AI merge gate events for PR #([1-9][0-9]*)")
REVIEW_FIELDS = "state commit { oid } fullDatabaseId"
AGENT_WORKFLOW = ".github/workflows/ai-merge-gate-evaluate.yml"
CLASSIFIER_INPUTS = ("scripts/ci/classifier_agent.py", AGENT_WORKFLOW, "scripts/ci/ai_merge_gate.py",
                     "scripts/ci/diff_classifier.py", WORKFLOW, ".claude/skills/verifier-classifier/SKILL.md")
MAX_HEAD_ATTEMPTS = 2
MAX_PR_ATTEMPTS = 3
MAX_TREE_BYTES = 500_000_000
MAX_TEXT = 60_000

DISCUSSION_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      %s(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          author { __typename login ... on Bot { id } }
          editor { __typename login ... on Bot { id } }
          body lastEditedAt %s
        }
      }
    }
  }
}"""


class GateError(Exception):
    """An input the gate cannot judge; the gate fails closed and names it."""


def http(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def log(message: str) -> None:
    # One line, so text quoted from an API response cannot start a workflow command.
    print("ai-merge-gate: " + message.replace("\r", " ").replace("\n", " "))


def clip(text: str) -> str:
    if len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + f"\n\n[display clipped: {len(text) - MAX_TEXT} more characters]"


class GitHub:
    def __init__(self, transport, token: str, repo: str) -> None:
        self.transport, self.token, self.repo = transport, token, repo

    def call(self, method: str, path: str, body=None, accept="application/vnd.github+json",
             missing_ok=False, raw=False):
        headers = {"Authorization": f"Bearer {self.token}", "Accept": accept,
                   "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "jolt-ai-merge-gate"}
        data = None if body is None else json.dumps(body).encode()
        status, payload = self.transport(method, GITHUB_API + path, headers, data)
        if status == 404 and missing_ok:
            return None
        if not 200 <= status < 300:
            raise GateError(f"GitHub {method} {path}: HTTP {status}: {payload[:300]!r}")
        if raw:
            return payload
        return payload.decode() if accept == "application/vnd.github.diff" else json.loads(payload)

    def get(self, path: str, missing_ok=False):
        return self.call("GET", path, missing_ok=missing_ok)

    def post(self, path: str, body):
        return self.call("POST", path, body)

    def patch(self, path: str, body):
        return self.call("PATCH", path, body)

    def diff(self, path: str) -> str:
        return self.call("GET", path, accept="application/vnd.github.diff")

    def discussion(self, number: int, connection: str, fields: str) -> list[dict]:
        owner, name = self.repo.split("/")
        nodes, cursor = [], None
        while True:
            variables = {"owner": owner, "name": name, "number": number, "cursor": cursor}
            data = self.post("/graphql", {"query": DISCUSSION_QUERY % (connection, fields),
                                          "variables": variables})
            if data.get("errors"):
                raise GateError(f"GitHub GraphQL {connection}: {data['errors']}")
            page = data["data"]["repository"]["pullRequest"][connection]
            nodes += page["nodes"]
            if not page["pageInfo"]["hasNextPage"]:
                return nodes
            cursor = page["pageInfo"]["endCursor"]


class Source:
    """The pull request's merge base and head, pinned by SHA: the changed paths the classifier must cover,
    and the model's read-only view of both commits. Each commit's files come from GitHub's archive
    of that exact commit, held in memory as data."""

    def __init__(self, github: GitHub, base: str, head: str) -> None:
        self.github, self.shas = github, {"base": base, "head": head}
        self.listings: dict[str, dict[str, dict]] = {}
        self.trees: dict[str, dict[str, str | None]] = {}

    def listing(self, revision: str) -> dict[str, dict]:
        """The commit's files, links and submodules by path, from its complete tree listing."""
        if revision not in self.listings:
            sha = self.shas[revision]
            listing = self.github.get(f"/repos/{self.github.repo}/git/trees/{sha}?recursive=1")
            if listing["truncated"]:
                raise GateError(f"GitHub truncated the tree listing of {sha}")
            self.listings[revision] = {entry["path"]: entry for entry in listing["tree"] if entry["type"] != "tree"}
        return self.listings[revision]

    def changed(self) -> set[str]:
        """Each path whose file, link or submodule differs between the merge base and the head, so
        a rename contributes both names. Unlike the pull request's file list, which follows its
        branch, this is a function of the two SHAs."""
        base, head = self.listing("base"), self.listing("head")

        def entry(tree: dict[str, dict], path: str) -> tuple[str, str] | None:
            return (tree[path]["mode"], tree[path]["sha"]) if path in tree else None

        return {path for path in base.keys() | head.keys() if entry(base, path) != entry(head, path)}

    def tree(self, revision: str) -> dict[str, str | None]:
        """Path to text, or to None for a binary file."""
        if revision not in self.trees:
            sha, repo = self.shas[revision], self.github.repo
            blobs = {path: entry for path, entry in self.listing(revision).items() if entry["type"] == "blob"}
            if sum(entry["size"] for entry in blobs.values()) > MAX_TREE_BYTES:
                raise AgentFailure(f"commit {sha} holds over {MAX_TREE_BYTES} bytes of files")
            archive = self.github.call("GET", f"/repos/{repo}/tarball/{sha}", raw=True)
            files = {}
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                for member in tar:
                    if not (member.isfile() or member.issym()):
                        continue
                    # The commit's own .gitattributes can make git archive drop or rewrite files
                    # (export-ignore, export-subst), so each file must hash to its blob in the
                    # commit's tree; reading only a file of the listed size bounds the work.
                    path = member.name.partition("/")[2]
                    entry, link = blobs.pop(path, {}), member.linkname.encode()
                    data = b""
                    if (len(link) if member.issym() else member.size) == entry.get("size"):
                        data = link if member.issym() else tar.extractfile(member).read()
                    if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != entry.get("sha"):
                        raise GateError(f"GitHub's archive of {sha} differs from its tree at {path!r}")
                    files[path] = None
                    if member.issym():
                        files[path] = f"(symbolic link to {link.decode()})"
                    else:
                        try:
                            files[path] = data.decode()
                        except UnicodeDecodeError:
                            pass
            if blobs:
                raise GateError(f"GitHub's archive of {sha} lacks {len(blobs)} files, such as {min(blobs)!r}")
            self.trees[revision] = files
        return self.trees[revision]


class Prepared(Exception):
    """The trusted preparation job reserved an attempt and exported its inputs."""


class Classifier:
    def __init__(self, directory: Path, prepare: bool = False) -> None:
        self.directory, self.prepare = directory, prepare
        self.plan = json.loads((directory / "plan.json").read_text()) if (directory / "plan.json").exists() else None
        try:
            self.result = json.loads((directory / "result.json").read_text())
        except (FileNotFoundError, ValueError, UnicodeError):
            self.result = None

    def prepaid(self, source: Source) -> bool:
        if self.plan is None:
            return False
        plan = self.plan
        return plan.get("repository") == source.github.repo and plan.get("shas") == source.shas \
            and plan.get("classifier_inputs_sha256") == classifier_inputs_sha256() \
            and plan.get("paths") == sorted(source.changed())

    def ask(self, instructions: str, source: Source, admit, counts: tuple[int, int, int]) -> dict | None:
        if not self.prepare:
            if not self.prepaid(source):
                return None
            return validate_verdict(self.result, source.changed()) if self.result is not None else None
        self.directory.mkdir(parents=True, exist_ok=True)
        workspace = self.directory / "workspace"
        for revision in ("base", "head"):
            for path, text in source.tree(revision).items():
                target = workspace / revision / path
                if Path(path).is_absolute() or any(part in ("..", ".git") for part in Path(path).parts) or "\\" in path:
                    raise AgentFailure("invalid repository path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text if text is not None else "[binary file; semantic impact unknown]\n")
        diff = source.github.diff(f"/repos/{source.github.repo}/compare/{source.shas['base']}...{source.shas['head']}")
        files = parse_diff(diff)
        seen = len(files), sum(len(f.added) for f in files), sum(len(f.removed) for f in files)
        if seen != counts:
            raise GateError("the diff is incomplete")
        (workspace / "diff.patch").write_text(diff)
        plan = {"repository": source.github.repo, "shas": source.shas, "paths": sorted(source.changed()),
                "classifier_inputs_sha256": classifier_inputs_sha256()}
        (workspace / "change.json").write_text(json.dumps(plan, indent=2))
        (self.directory / "instructions.md").write_text(instructions)
        admit()
        (self.directory / "plan.json").write_text(json.dumps(plan))
        raise Prepared()


def model_text(text: str) -> str:
    """Escape untrusted verdict fields before putting them in GitHub Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|~>-])", r"\\\1", html.escape(text, quote=False)).replace("@", "@\u200b")


@dataclass
class Record:
    """A classifier result for one head, published as a check run (the cache the next evaluation
    reads) and in the gate's PR comment (the reasoning for humans). A refused attempt has no risk."""

    name: str
    conclusion: str
    title: str
    body: str
    risk: int | None


@dataclass(frozen=True)
class Outcome:
    conclusion: str
    title: str
    summary: str
    # A row that waits for a person drives the no-reviewer-assigned label.
    human: bool = False
    # The gate approves the head. Only a success on the attested agent path sets it.
    approve: bool = False


UNSETTLED = Outcome("neutral", "The pull request changed while the gate ran",
                    "Comments or reviews changed while the gate ran, and the current state needs a model "
                    "result this run did not get. The next pull request comment, review or push runs the "
                    "gate again.")
SUPERSEDED = Outcome("neutral", "Superseded while the gate ran",
                     "The pull request changed before the gate's labels and approval were confirmed; its "
                     "next evaluation decides the current head.")


def unassigned(pull: dict) -> bool:
    return not (pull["requested_reviewers"] or pull["requested_teams"] or pull["assignees"])


class Gate:
    def __init__(self, github: GitHub, model: Classifier | None, default_branch: str, maintainer_list: str,
                 run_id: str) -> None:
        self.github, self.model, self.repo = github, model, github.repo
        self.default_branch, self.maintainer_list, self.run_id = default_branch, maintainer_list, run_id
        self.maintainers: frozenset[str] = frozenset()
        self.writers: dict[str, bool] = {}
        self.gate_runs: dict[str, bool] = {}
        self.comments: list[dict] = []
        self.reviews: list[dict] = []
        self.records: list[Record] = []
        self.ledger: dict[str, int] = {}
        self.posted = 0
        self.source: Source | None = None
        self.unclassified = False

    def is_maintainer(self, login: str) -> bool:
        if not self.maintainers:
            # An empty or malformed list would otherwise admit nobody, or the wrong people, silently.
            logins = [entry.strip() for entry in self.maintainer_list.split(",") if entry.strip()]
            if not logins or not all(LOGIN.fullmatch(entry) for entry in logins):
                raise GateError("repository variable AI_MERGE_GATE_MAINTAINERS must list GitHub logins "
                                "separated by commas")
            self.maintainers = frozenset(entry.lower() for entry in logins)
        return login.lower() in self.maintainers and self.can_write(login)

    def can_write(self, login: str) -> bool:
        if login not in self.writers:
            path = f"/repos/{self.repo}/collaborators/{urllib.parse.quote(login)}/permission"
            permission = self.github.get(path, missing_ok=True)
            self.writers[login] = permission is not None and permission["permission"] in ("admin", "write")
        return self.writers[login]

    def from_gate(self, check_run: dict) -> bool:
        """Whether the claimed external ID names a gate workflow run. This trusts the writer:
        any workflow with Checks write can forge external_id and completed_at with GITHUB_TOKEN."""
        run_id = str(check_run.get("external_id") or "")
        if run_id not in self.gate_runs:
            run = self.github.get(f"/repos/{self.repo}/actions/runs/{run_id}", missing_ok=True) \
                if run_id.isdigit() else None
            self.gate_runs[run_id] = run is not None and run["path"].partition("@")[0] == WORKFLOW \
                and run["event"] in ("issue_comment", "workflow_run") \
                and (run.get("head_repository") or {}).get("full_name") == self.repo
        return self.gate_runs[run_id]

    def refresh(self, number: int) -> bool:
        pull = self.github.get(f"/repos/{self.repo}/pulls/{number}")
        present = {label["name"] for label in pull["labels"]} & MANAGED_LABELS
        if present:
            head, page = pull["head"]["sha"], 1
            while True:
                query = urllib.parse.urlencode({"check_name": GATE, "filter": "all", "per_page": 100, "page": page})
                runs = self.github.get(f"/repos/{self.repo}/commits/{head}/check-runs?{query}")["check_runs"]
                evaluated = any((run.get("app") or {}).get("id") == ACTIONS_APP_ID
                                and run["head_sha"] == head and self.from_gate(run) for run in runs)
                if evaluated or len(runs) < 100:
                    break
                page += 1
            if not evaluated:
                for name in sorted(present):
                    self.github.call("DELETE", f"/repos/{self.repo}/issues/{number}/labels/{name}", missing_ok=True)
        return self.sync_conflict(number)

    def sync_conflict(self, number: int) -> bool:
        pull = self.github.get(f"/repos/{self.repo}/pulls/{number}")
        if pull["state"] != "open" or pull["base"]["ref"] != self.default_branch:
            return True
        if pull["mergeable"] is None:
            return False
        present = any(label["name"] == MERGE_CONFLICT for label in pull["labels"])
        route = f"/repos/{self.repo}/issues/{number}/labels"
        if pull["mergeable"] is False and not present:
            if self.github.get(f"/repos/{self.repo}/labels/{MERGE_CONFLICT}", missing_ok=True) is None:
                raise GateError(f"repository label {MERGE_CONFLICT!r} is missing; an admin must create the gate's labels")
            self.github.post(route, {"labels": [MERGE_CONFLICT]})
        elif pull["mergeable"] is True and present:
            self.github.call("DELETE", route + "/" + MERGE_CONFLICT, missing_ok=True)
        return True

    def refresh_many(self, numbers: list[int]) -> None:
        deadline = time.monotonic() + 60
        pending = [number for number in numbers if not self.refresh(number)]
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(2, remaining))
            pending = [number for number in pending if not self.sync_conflict(number)]
        for number in pending:
            log(f"PR #{number}: mergeability still unknown; leaving {MERGE_CONFLICT} unchanged")

    def refresh_all(self) -> None:
        numbers, page = [], 1
        while True:
            query = urllib.parse.urlencode({"state": "open", "base": self.default_branch, "per_page": 100, "page": page})
            pulls = self.github.get(f"/repos/{self.repo}/pulls?{query}")
            numbers.extend(pull["number"] for pull in pulls)
            if len(pulls) < 100:
                break
            page += 1
        self.refresh_many(numbers)

    def recorded(self, head: str, name: str) -> Record | None:
        """A classification for this head with matching inputs, from this run or a check run
        claiming a gate workflow run. Checks-write workflows can forge that claim."""
        if mine := [record for record in self.records if record.name == name and record.risk is not None]:
            return mine[-1]
        query = urllib.parse.urlencode({"check_name": name, "filter": "all", "per_page": 100})
        runs = self.github.get(f"/repos/{self.repo}/commits/{head}/check-runs?{query}")["check_runs"]
        ours = sorted((run for run in runs if (run.get("app") or {}).get("id") == ACTIONS_APP_ID
                       and run["head_sha"] == head and run["status"] == "completed"),
                      key=lambda run: run["completed_at"], reverse=True)
        run = next((run for run in ours if self.from_gate(run)), None)
        if run is None:
            return None
        if run["conclusion"] not in ("success", "neutral"):
            return None
        output = run.get("output") or {}
        try:
            metadata = json.loads(output.get("text") or "null")
            risk = metadata["risk"]
            if metadata["classifier_inputs_sha256"] != classifier_inputs_sha256():
                return None
        except (ValueError, TypeError, KeyError):
            return None
        if type(risk) is not int or risk not in range(1, 5):
            return None
        return Record(name, run["conclusion"], output["title"], output["summary"], risk)

    def risk_level(self, pull: dict) -> int:
        if self.unclassified or not self.authorized(pull):
            return 4
        if record := self.recorded(pull["head"]["sha"], CLASSIFIER):
            return record.risk
        return 4

    def pinned(self, pull: dict) -> Source:
        """The evaluated head and its merge base with the base branch, fixed on first use. The model's
        input and its tools read these two commits."""
        if self.source is None:
            head, ref = pull["head"]["sha"], urllib.parse.quote(pull["base"]["ref"], safe="")
            base = self.github.get(f"/repos/{self.repo}/compare/{ref}...{head}?per_page=1")["merge_base_commit"]["sha"]
            self.source = Source(self.github, base, head)
        return self.source

    def attempts(self) -> dict[str, int]:
        """Paid classifier attempts per head, from the ledger line that ends each of the gate's
        classifier comments. A comment outlives pushes and force-pushes; counts merge by maximum, so
        a duplicated or injected ledger can only raise them."""
        ledger: dict[str, int] = {}
        for node in self.comments:
            if not self.ours(node, f"<!-- {CLASSIFIER} -->"):
                continue
            match = LEDGER.fullmatch(node["body"].rpartition("\n")[2])
            try:
                counts = json.loads(match.group(1)) if match else {}
            except ValueError:
                counts = None
            if not isinstance(counts, dict) or not all(
                    re.fullmatch(r"[0-9a-f]{40}", sha) and type(count) is int and count > 0
                    for sha, count in counts.items()):
                raise GateError("the gate's classifier comment holds a malformed attempt ledger")
            for sha, count in counts.items():
                ledger[sha] = max(ledger.get(sha, 0), count)
        return ledger

    def classify(self, pull: dict) -> None:
        try:
            self.classify_head(pull)
        except AgentFailure:
            self.unclassified = True
            self.records.append(Record(CLASSIFIER, "neutral", "Verifier classifier: unclassified",
                "The agent failed, reached a cap, or its result no longer matches this head. "
                "No classifier verdict was recorded; current-head maintainer approval is required.", None))

    def classify_head(self, pull: dict) -> None:
        number, head = pull["number"], pull["head"]["sha"]
        self.ledger = self.attempts()
        source = self.pinned(pull)
        paths = source.changed()
        used = sum(self.ledger.values())
        if not self.model.prepaid(source) and (self.ledger.get(head, 0) >= MAX_HEAD_ATTEMPTS or used >= MAX_PR_ATTEMPTS):
            self.records.append(Record(CLASSIFIER, "skipped", "Verifier classifier: attempts spent",
                                       f"This head has used {self.ledger.get(head, 0)} of {MAX_HEAD_ATTEMPTS} "
                                       f"paid classifier attempts and the pull request {used} of "
                                       f"{MAX_PR_ATTEMPTS}. The head stays unclassified, so only a person can "
                                       "approve it. No risk label is assigned; policy requires an independent "
                                       "current-head maintainer approval.", None))
            return

        def admit() -> None:
            # Counted durably before the first paid request, so a timeout or crash still counts.
            self.ledger[head] = self.ledger.get(head, 0) + 1
            self.comment(number, head, "Verifier classifier: running",
                         f"Paid attempt {self.ledger[head]} of {MAX_HEAD_ATTEMPTS} for this head and "
                         f"{used + 1} of {MAX_PR_ATTEMPTS} for this pull request. The verdict replaces "
                         "this text; without one, the attempt failed and the head has no classifier result.")

        skill = (ROOT / ".claude/skills/verifier-classifier/SKILL.md").read_text()
        answer = self.model.ask(skill, source, admit,
                                (pull["changed_files"], pull["additions"], pull["deletions"]))
        if answer is not None:
            answer = validate_verdict(answer, paths)
        if answer is None:
            raise AgentFailure("no classifier verdict")
        risk = max(item["level"] for item in answer["risks"])
        conclusion = "success" if risk < 4 else "neutral"
        self.records.append(Record(CLASSIFIER, conclusion, f"Verifier classifier: risk-{risk}",
                                   f"{model_text(answer['reasoning'])}\n\nRisk: risk-{risk}\n\n" +
                                   "\n".join(f"- {model_text(item['path'])}: risk-{item['level']} — {model_text(item['reasoning'])}"
                                             for item in answer["risks"]) +
                                   f"\n\n_{MODEL}, reasoning effort {EFFORT}_", risk))

    def attester(self, pull: dict) -> str | None:
        """Who attests, in the comments `decide` read, a clean review of the current head."""
        number, head = pull["number"], pull["head"]["sha"]
        for node in self.comments:
            author = node.get("author") or {}
            for line in node["body"].splitlines():
                match = ATTESTATION.fullmatch(line)
                if not match or match.group(1) != str(number) or match.group(2) != head \
                        or not match.group(3).strip() or match.group(4) not in REVIEW_MODELS \
                        or match.group(5) not in REVIEW_EFFORTS:
                    continue
                actors = [author, node.get("editor") or {}] if node.get("lastEditedAt") else [author]
                if all(self.can_attest(actor) for actor in actors):
                    return author["login"]
        return None

    def can_attest(self, actor: dict) -> bool:
        return (actor.get("__typename") == "Bot" and actor.get("id") == PIKA_BOT_ID) or \
            (actor.get("__typename") == "User" and self.is_maintainer(actor["login"]))

    def approvals(self, pull: dict) -> tuple[list[str], list[str], list[str]]:
        """From each person's latest review in `self.reviews`: the maintainers and writers whose
        change request stands, the writers other than the author who approve, and the maintainers
        among those approvers whose approval names the current head."""
        head, author = pull["head"]["sha"], pull["user"]["login"]
        latest = {}
        for review in self.reviews:
            login = (review.get("author") or {}).get("login")
            if login and review["state"] in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
                latest[login] = review
        blocking = sorted(login for login, review in latest.items() if review["state"] == "CHANGES_REQUESTED"
                          and self.can_write(login))
        # Risk-1 through risk-3 retain GitHub's approval freshness; risk-4 also checks commit.oid.
        approvers = sorted(login for login, review in latest.items()
                           if review["state"] == "APPROVED" and review["author"]["__typename"] == "User"
                           and login != author and self.can_write(login))
        current = [login for login in approvers if (latest[login].get("commit") or {}).get("oid") == head
                   and self.is_maintainer(login)]
        return blocking, approvers, current

    def authorized(self, pull: dict) -> bool:
        return self.is_maintainer(pull["user"]["login"])

    def decide(self, pull: dict, pay: bool = True) -> Outcome:
        """The table's result for `pull`, including whether the gate approves its head. With `pay`
        false, a row that needs a new paid model result decides UNSETTLED instead."""
        number, head, author = pull["number"], pull["head"]["sha"], pull["user"]["login"]
        self.comments = self.github.discussion(number, "comments", "fullDatabaseId")
        self.reviews = self.github.discussion(number, "reviews", REVIEW_FIELDS)
        blocking, approvers, current = self.approvals(pull)
        if blocking:
            return Outcome("neutral", "Changes requested by " + ", ".join(blocking),
                           "A maintainer or writer requested changes; the gate does not approve until they "
                           "approve or their review is dismissed.")
        authorized = self.authorized(pull)
        if not approvers and not authorized:
            return Outcome("neutral", "Human approval required",
                           f"@{author} is not a maintainer, so no classifier runs. Unclassified heads "
                           "require an independent current-head maintainer approval.",
                           human=True)
        # `self.records` holds this run's classification or its refusal: one paid attempt per run.
        if authorized and not current and not self.records \
                and self.recorded(head, CLASSIFIER) is None:
            if not pay:
                return UNSETTLED
            self.classify(pull)
        risk = self.risk_level(pull)
        if risk == 4:
            if current:
                return Outcome("success", "Risk 4 approved by " + ", ".join(current),
                               f"A maintainer other than the author approved the current head `{head}`.")
            return Outcome("neutral", "Risk 4: current-head maintainer approval required",
                           "Protocol or unclassified changes need an undismissed approval of the current "
                           "head from a maintainer with write access, other than the PR author. The gate "
                           "never approves risk-4 itself.", human=True)
        attested_by = self.attester(pull)
        classified = self.recorded(head, CLASSIFIER) is not None
        approve = attested_by is not None and classified and authorized
        if approvers:
            return Outcome("success", "Approved by " + ", ".join(approvers),
                           "A writer other than the author approved the pull request, and GitHub has not "
                           "dismissed the approval." + (" The gate approves it too, on the attestation by "
                                                        f"{attested_by}." if approve else ""), approve=approve)
        if approve:
            return Outcome("success", "Review attested by " + attested_by,
                           f"An authorized comment attests a clean review of the current head `{head}`, so "
                           "the gate approves it.", approve=True)
        return Outcome("neutral", "Needs an approval or a current-head review attestation",
                       "A writer other than the author can approve. Otherwise the Pika App bot or a "
                       "maintainer posts a PR comment with this standalone line, and the gate approves: "
                       f"This pull request #{number} has been reviewed at commit {head} using skill <skill name> "
                       "and model <model name> with <effort> reasoning effort, and no issues were found. "
                       f"The model must be {' or '.join(REVIEW_MODELS)}, and the effort "
                       f"{' or '.join(REVIEW_EFFORTS)}.")

    def labels(self, pull: dict, policy: Outcome) -> set[str]:
        """The managed labels for a policy result. needs-maintainer-review reports a missing maintainer
        authorization of the agent path even when a human approval satisfies the policy."""
        labels = set()
        if not self.authorized(pull):
            labels.add(NEEDS_MAINTAINER)
        elif not self.unclassified:
            if record := self.recorded(pull["head"]["sha"], CLASSIFIER):
                labels.add(f"risk-{record.risk}")
        if policy.conclusion == "success":
            labels.add(MERGE_ELIGIBLE)
        if policy.human and unassigned(pull):
            labels.add(NO_REVIEWER)
        return labels

    def evaluate(self, number: int) -> None:
        pull = check = None
        try:
            pull = self.github.get(f"/repos/{self.repo}/pulls/{number}")
            if pull["draft"]:
                self.withdraw(number, labels={label["name"] for label in pull["labels"]} & MANAGED_LABELS)
                log(f"PR #{number} is a draft; gate inactive")
                return
            if pull["state"] == "open" and pull["base"]["ref"] == self.default_branch:
                head = pull["head"]["sha"]
                check = self.check(head, GATE, None, "Evaluating", "The gate is evaluating this head.")
                outcome = self.judge(number, pull)
                self.complete(check, outcome)
                log(f"PR #{number} at {head}: {outcome.conclusion}: {outcome.title}")
                return
        except Exception as error:
            # No approval or merge-eligible may outlive a failed evaluation, whichever step failed.
            # Each cleanup step runs even if an earlier one fails; the original error is reported.
            failed = Outcome("failure", "Gate error: human approval required", f"The gate could not decide: {error}")
            draft = pull is not None and pull["draft"]
            cleanup = {label["name"] for label in pull["labels"]} & MANAGED_LABELS if draft else None
            steps = [("withdraw the gate's approval and labels", lambda: self.withdraw(number, labels=cleanup))]
            if pull is not None and not draft:
                head = pull["head"]["sha"]
                labels = set()
                managed = MANAGED_LABELS - {NEEDS_MAINTAINER}
                try:
                    if not self.authorized(pull):
                        labels, managed = {NEEDS_MAINTAINER}, MANAGED_LABELS
                    else:
                        labels = self.labels(pull, failed)
                except Exception as auth_error:
                    log(f"Could not determine classifier labels: {auth_error}")
                steps += [("label the failed evaluation", lambda: self.label(
                              number, head, labels | ({NO_REVIEWER} if unassigned(pull) else set()), managed)),
                          ("publish the classifier result", lambda: self.publish(number, head))]
            if check is not None:
                steps.append(("report the failure", lambda: self.complete(check, failed)))
            for action, step in steps:
                try:
                    step()
                except Exception as step_error:
                    log(f"Could not {action}: {step_error}")
            raise
        log(f"PR #{number} is not an open pull request into {self.default_branch}")
        # A merged pull request cannot reopen; its approval records what let it merge.
        self.withdraw(number, merged=pull["merged_at"] is not None)

    def judge(self, number: int, pull: dict) -> Outcome:
        """Decide the policy, publish the classifier result and the labels, then approve the head or
        withdraw the gate's approval. A pass rests on state read after the label writes, reusing this
        run's paid results: the model work behind it can take minutes."""
        head = pull["head"]["sha"]
        policy = self.decide(pull)
        self.publish(number, head)
        fresh = self.label(number, head, self.labels(pull, policy))
        if fresh and policy.conclusion == "success":
            self.writers = {}
            policy = self.decide(fresh, pay=False)
            fresh = self.label(number, head, self.labels(fresh, policy))
        if not fresh:
            policy = SUPERSEDED
        return policy if self.reconcile(number, head, policy) else SUPERSEDED

    def reconcile(self, number: int, head: str, policy: Outcome) -> bool:
        """Leave the gate's own approval standing on `head` exactly when `policy.approve`: dismiss its
        other undismissed approvals in `self.reviews`, then approve `head` once. GitHub dismisses stale
        approvals when a push arrives, which misses an approval of the old head submitted just after
        it, so the head is read again; if it moved, the approval is withdrawn and False returned."""
        standing = self.standing()
        for review in standing:
            if not (policy.approve and review["commit"]["oid"] == head):
                self.dismiss(number, review["fullDatabaseId"])
        if not policy.approve or any(review["commit"]["oid"] == head for review in standing):
            return True
        created = self.github.post(f"/repos/{self.repo}/pulls/{number}/reviews", {
            "commit_id": head, "event": "APPROVE",
            "body": f"{APPROVAL}\nThe AI merge gate approves `{head}`. {policy.title}: {policy.summary}"})
        if self.moved(self.github.get(f"/repos/{self.repo}/pulls/{number}"), head):
            self.dismiss(number, created["id"])
            return False
        return True

    def standing(self) -> list[dict]:
        """The gate's undismissed approvals among `self.reviews`."""
        return [review for review in self.reviews if review["state"] == "APPROVED" and self.ours(review, APPROVAL)]

    def withdraw(self, number: int, merged: bool = False, labels: set[str] | None = None) -> None:
        """Remove the requested labels (merge-eligible by default) and, unless merged, dismiss each
        standing gate approval. Dismissals run even if a label write fails."""
        try:
            for name in sorted({MERGE_ELIGIBLE} if labels is None else labels):
                self.github.call("DELETE", f"/repos/{self.repo}/issues/{number}/labels/{name}", missing_ok=True)
        finally:
            if not merged:
                self.reviews = self.github.discussion(number, "reviews", REVIEW_FIELDS)
                for review in self.standing():
                    self.dismiss(number, review["fullDatabaseId"])

    def dismiss(self, number: int, review_id) -> None:
        path = f"/repos/{self.repo}/pulls/{number}/reviews/{review_id}"
        try:
            self.github.call("PUT", path + "/dismissals", {
                "event": "DISMISS",
                "message": "The AI merge gate withdraws its approval because the pull request no longer meets its policy."})
        except GateError:
            # A push may have dismissed it as stale since the gate read it.
            if self.github.get(path)["state"] == "APPROVED":
                raise

    def publish(self, number: int, head: str) -> None:
        """Publish the classifier results of this run that are not yet published."""
        if self.records[self.posted:] and self.github.get(f"/repos/{self.repo}/pulls/{number}")["draft"]:
            return
        for record in self.records[self.posted:]:
            self.check(head, record.name, record.conclusion, record.title, record.body, record.risk)
            self.comment(number, head, record.title, record.body)
            self.posted += 1

    def moved(self, pull: dict, head: str) -> bool:
        """Whether `head` is no longer ready and open into the default branch. Remove merge-eligible
        from an inactive head, or all managed labels from a draft."""
        if pull["state"] == "open" and not pull["draft"] and pull["base"]["ref"] == self.default_branch \
                and pull["head"]["sha"] == head:
            return False
        managed = MANAGED_LABELS if pull["draft"] else {MERGE_ELIGIBLE}
        for name in sorted({label["name"] for label in pull["labels"]} & managed):
            self.github.call("DELETE", f"/repos/{self.repo}/issues/{pull['number']}/labels/{name}", missing_ok=True)
        return True

    def label(self, number: int, head: str, wanted: set[str], managed: frozenset[str] = MANAGED_LABELS) -> dict | None:
        """Make the `managed` labels exactly `wanted`, with a current-head read before and after the
        writes, and return the last read: None once the pull request no longer has `head` open into
        the default branch. GitHub has no atomic head-conditional label API, so the final read refuses
        a stale pass."""
        route = f"/repos/{self.repo}/issues/{number}/labels"
        pull = self.github.get(f"/repos/{self.repo}/pulls/{number}")
        if self.moved(pull, head):
            return None
        present = {label["name"] for label in pull["labels"]} & managed
        if present == wanted:
            return pull
        for name in sorted(wanted - present):
            if self.github.get(f"/repos/{self.repo}/labels/{name}", missing_ok=True) is None:
                raise GateError(f"repository label {name!r} is missing; an admin must create the gate's labels")
        if wanted - present:
            self.github.post(route, {"labels": sorted(wanted - present)})
        for name in sorted(present - wanted):
            self.github.call("DELETE", route + "/" + name, missing_ok=True)
        pull = self.github.get(f"/repos/{self.repo}/pulls/{number}")
        if self.moved(pull, head):
            return None
        actual = {label["name"] for label in pull["labels"]} & managed
        if actual != wanted:
            raise GateError(f"labels changed during publication: expected {sorted(wanted)}, got {sorted(actual)}")
        return pull

    def check(self, head: str, name: str, conclusion: str | None, title: str, summary: str,
              risk: int | None = None) -> int:
        """Create a check run on `head`: completed with `conclusion`, or in progress without one. Its
        external ID names this workflow run, which is how a later run trusts a cached result."""
        state = {"status": "completed", "conclusion": conclusion} if conclusion else {"status": "in_progress"}
        return self.github.post(f"/repos/{self.repo}/check-runs", {
            "name": name, "head_sha": head, "external_id": self.run_id, **state,
            "output": {"title": title, "summary": clip(summary),
                       **({"text": json.dumps({"risk": risk,
                                              "classifier_inputs_sha256": classifier_inputs_sha256()})}
                          if risk is not None else {})},
        })["id"]

    def complete(self, check: int, outcome: Outcome) -> None:
        self.github.patch(f"/repos/{self.repo}/check-runs/{check}", {
            "status": "completed", "conclusion": outcome.conclusion,
            "output": {"title": outcome.title, "summary": clip(outcome.summary)},
        })

    def ours(self, node: dict, marker: str) -> bool:
        """Whether the workflow token wrote this comment or review, starting it with `marker`."""
        author = node.get("author") or {}
        return author.get("__typename") == "Bot" and author.get("login") == PUBLISHER and node["body"].startswith(marker)

    def comment(self, number: int, head: str, title: str, text: str) -> None:
        """Create or update the gate's classifier comment: the latest result for people, then the
        attempt ledger the next evaluation counts, as its last line."""
        marker = f"<!-- {CLASSIFIER} -->"
        # Model text is shaped by the diff; a zero-width space keeps it from mentioning people.
        body = clip(f"{marker}\n### {title}\n\nHead `{head}`\n\n" + text.replace("@", "@\u200b")) \
            + f"\n\n<!-- ai-merge-gate/attempts {json.dumps(self.ledger, sort_keys=True)} -->"
        mine = next((node for node in self.comments if self.ours(node, marker)), None)
        if mine:
            self.github.patch(f"/repos/{self.repo}/issues/comments/{mine['fullDatabaseId']}", {"body": body})
            mine["body"] = body
        else:
            created = self.github.post(f"/repos/{self.repo}/issues/{number}/comments", {"body": body})
            self.comments.append({"author": {"__typename": "Bot", "login": PUBLISHER}, "body": body,
                                  "fullDatabaseId": str(created["id"])})


def classifier_inputs_sha256() -> str:
    return hashlib.sha256(json.dumps({path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                     for path in CLASSIFIER_INPUTS}, sort_keys=True).encode()).hexdigest()


def resolve(event_name: str, event: dict, github: GitHub | None = None) -> int | None:
    """The pull request an `issue_comment` or bridged `workflow_run` event is about."""
    if event_name == "issue_comment":
        author = event["comment"]["user"]
        # Ignore other bots. The gate's own comments start no run, since GITHUB_TOKEN events
        # create no workflow runs. Do not filter on the body: deleting the line is itself an
        # invalidation event.
        if author["type"] == "Bot" and author["node_id"] != PIKA_BOT_ID:
            return None
        return event["issue"]["number"] if event["issue"].get("pull_request") else None
    if event_name != "workflow_run":
        return None
    run, repo = event["workflow_run"], event["repository"]
    match = RUN_NAME.fullmatch(run.get("display_title") or "")
    if not match or run.get("path") != ".github/workflows/ai-merge-gate-events.yml" \
            or run.get("event") not in ("pull_request", "pull_request_review") \
            or (run.get("repository") or {}).get("id") != repo["id"]:
        return None
    number = int(match.group(1))
    # GitHub's run associations can omit closed PRs. Bind every proposed number to the
    # source repository, branch and exact head through the base repository's PR API.
    source = run.get("head_repository") or {}
    if github is None or not source.get("id"):
        return None
    pull = github.get(f"/repos/{github.repo}/pulls/{number}", missing_ok=True)
    if pull and pull["base"]["repo"]["id"] == repo["id"] \
            and (pull["head"].get("repo") or {}).get("id") == source.get("id") \
            and pull["head"]["ref"] == run.get("head_branch") \
            and pull["head"]["sha"] == run.get("head_sha"):
        return number
    return None


def main(argv: list[str]) -> int:
    env = os.environ
    try:
        if argv[1:] == ["resolve"]:
            event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
            number = resolve(env["GITHUB_EVENT_NAME"], event,
                             GitHub(http, env["GH_TOKEN"], env["GITHUB_REPOSITORY"]))
            with open(env["GITHUB_OUTPUT"], "a") as output:
                output.write(f"pr={number or ''}\n")
            log(f"{env['GITHUB_EVENT_NAME']} event: PR {number or 'none'}")
            return 0
        if len(argv) == 3 and argv[1] == "prepare":
            github = GitHub(http, env["GH_TOKEN"], env["GITHUB_REPOSITORY"])
            model = Classifier(Path(env["CLASSIFIER_DIRECTORY"]), prepare=True)
            gate = Gate(github, model, env["DEFAULT_BRANCH"], env.get("AI_MERGE_GATE_MAINTAINERS", ""),
                        env["GITHUB_RUN_ID"])
            pull = github.get(f"/repos/{github.repo}/pulls/{int(argv[2])}")
            needed = False
            if pull["state"] == "open" and not pull["draft"] and pull["base"]["ref"] == gate.default_branch:
                try:
                    gate.decide(pull)
                except Prepared:
                    needed = True
            with open(env["GITHUB_OUTPUT"], "a") as output:
                output.write(f"needed={str(needed).lower()}\n")
            return 0
        if len(argv) == 3 and argv[1] == "evaluate":
            github = GitHub(http, env["GH_TOKEN"], env["GITHUB_REPOSITORY"])
            model = Classifier(Path(env["CLASSIFIER_DIRECTORY"]))
            Gate(github, model, env["DEFAULT_BRANCH"], env.get("AI_MERGE_GATE_MAINTAINERS", ""),
                 env["GITHUB_RUN_ID"]).evaluate(int(argv[2]))
            return 0
        if (len(argv) == 3 and argv[1] == "refresh") or argv[1:] == ["refresh-all"]:
            github = GitHub(http, env["GH_TOKEN"], env["GITHUB_REPOSITORY"])
            gate = Gate(github, None, env["DEFAULT_BRANCH"], "", env["GITHUB_RUN_ID"])
            if argv[1] == "refresh":
                gate.refresh_many([int(argv[2])])
            else:
                gate.refresh_all()
            return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        print("::error::" + message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
