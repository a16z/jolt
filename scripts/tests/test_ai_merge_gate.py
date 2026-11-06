import difflib
import hashlib
import io
import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import tarfile
import tempfile
import tomllib
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

# The gate imports its sibling diff_classifier the way it runs in CI, from scripts/ci.
sys.path.insert(0, str(Path(__file__).parents[1] / "ci"))
import ai_merge_gate as GATE
import classifier_agent as AGENT

REPO = "a16z/jolt"
HEAD = "a" * 40
NEW_HEAD = "c" * 40
OLD = "b" * 40
BASE = "d" * 40
MAINTAINERS = "core-a, core-b"
# This evaluation's workflow run, an earlier run of the gate's workflow, a run of another
# workflow on main, and a run of the gate's file that a pull request's own workflow started.
RUN_ID = "9001"
GATE_RUN = "8001"
OTHER_RUN = "8002"
PR_RUN = "8003"
PIKA_LOGIN = "0xandoroid-ai-agent"
PIKA_ID = "BOT_kgDOFAtqNQ"
ENV = {"GH_TOKEN": "token", "GITHUB_REPOSITORY": REPO, "DEFAULT_BRANCH": "main",
       "AI_MERGE_GATE_MAINTAINERS": MAINTAINERS, "GITHUB_RUN_ID": RUN_ID, "CLASSIFIER_DIRECTORY": "/unused-fixture"}
DIFF = """diff --git a/crates/jolt-prover/src/lib.rs b/crates/jolt-prover/src/lib.rs
index 1111111..2222222 100644
--- a/crates/jolt-prover/src/lib.rs
+++ b/crates/jolt-prover/src/lib.rs
@@ -1,2 +1,2 @@ mod tests {{
 fn helper() {{}}
-// old
+// $(touch {sentinel}) `touch {sentinel}`
"""
PLAIN_DIFF = DIFF.format(sentinel="/nonexistent")
TREES = {BASE: {"crates/jolt-prover/src/lib.rs": "fn helper() {}\n// old\n"},
         HEAD: {"crates/jolt-prover/src/lib.rs": "fn helper() {}\n// $(touch /nonexistent)\n"}}


def comment(login, body, edited=False, kind="User", actor_id=None):
    author = {"__typename": kind, "login": login, "id": actor_id}
    return {"author": author, "editor": author if edited else None, "body": body,
            "lastEditedAt": "2026-10-30T12:00:00Z" if edited else None, "fullDatabaseId": "11"}


def attestation(sha=HEAD, number=7, skill="ci-code-review", model="GPT-6 Astra", effort="max"):
    return (f"This pull request #{number} has been reviewed at commit {sha} using skill {skill} "
            f"and model {model} with {effort} reasoning effort, and no issues were found.")


def review(login, state, sha=HEAD, body="", kind="User"):
    return {"author": {"__typename": kind, "login": login}, "body": body, "lastEditedAt": None,
            "state": state, "commit": {"oid": sha}}


def ledger(counts):
    """The gate's classifier comment as an earlier evaluation left it, with `counts` paid attempts."""
    return {"author": {"__typename": "Bot", "login": "github-actions"}, "editor": None,
            "body": f"<!-- {GATE.CLASSIFIER} -->\n### Verifier classifier: running\n\n"
                    f"<!-- ai-merge-gate/attempts {json.dumps(counts)} -->",
            "lastEditedAt": None, "fullDatabaseId": "21"}


def gate_approval(sha=HEAD):
    """An approving review the gate submitted for `sha` in an earlier evaluation."""
    return {"author": {"__typename": "Bot", "login": "github-actions"}, "lastEditedAt": None,
            "body": f"{GATE.APPROVAL}\nThe AI merge gate approves `{sha}`.", "state": "APPROVED",
            "commit": {"oid": sha}, "fullDatabaseId": "31"}


def record(name, sha=HEAD, conclusion="success", **run):
    """A completed check run that an earlier run of the gate's workflow published, unless `run`
    overrides its fields."""
    return {"name": name, "head_sha": sha, "conclusion": conclusion, "status": "completed",
            "completed_at": "2026-10-30T00:00:00Z", "external_id": GATE_RUN,
            "app": {"id": GATE.ACTIONS_APP_ID, "slug": "github-actions"}, **run}


def git_tree(files):
    """GitHub's recursive tree listing and tarball for a commit holding `files`."""
    contents = {path: text if isinstance(text, bytes) else text.encode() for path, text in files.items()}
    entries = [{"path": path, "type": "blob", "mode": "100644", "size": len(data),
                "sha": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
               for path, data in contents.items()]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for path, data in contents.items():
            info = tarfile.TarInfo(f"a16z-jolt-0000000/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return {"truncated": False, "tree": entries}, buffer.getvalue()


def trees_for(files):
    """Merge base and head contents in which exactly the pull request file entries `files` changed."""
    base, head = {}, {}
    for entry in files:
        new, status = entry["filename"], entry.get("status", "modified")
        if status != "added":
            base[new] = f"old {new}\n"
        if status != "removed":
            head[new] = f"new {new}\n"
    return {BASE: base, HEAD: head}


def unified_diff(path, old, new):
    return f"diff --git a/{path} b/{path}\n" + "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{path}", f"b/{path}"))


def changed_contents(base, head, **kwargs):
    paths = sorted(path for path in base.keys() | head.keys() if base.get(path) != head.get(path))
    diff = "".join(unified_diff(path, base.get(path, ""), head.get(path, "")) for path in paths)
    parsed = GATE.parse_diff(diff)
    return FakeApis(files=paths, trees={BASE: base, HEAD: head}, diff=diff,
                    counts=(len(paths), sum(len(f.added) for f in parsed), sum(len(f.removed) for f in parsed)),
                    comments=[comment("core-a", attestation())], **kwargs)


class FakeApis:
    """GitHub state and classifier answers, with every publication recorded."""

    def __init__(self, author="core-a", comments=(), reviews=(), labels=(), draft=False,
                 records=(), heads=(HEAD,), diff=PLAIN_DIFF, files=("crates/jolt-prover/src/lib.rs",),
                 counts=(1, 1, 1), risk=2, risks=None, answer_text=None,
                 served_model="gpt-6.1-sol", trees=None,
                 reviewers=(), teams=(), assignees=(), state="open", base="main", merged_at=None,
                 mergeability=(True,)) -> None:
        # The pull request's file list, which follows its branch; the commits' trees do not move.
        self.files = [entry if isinstance(entry, dict) else {"filename": entry} for entry in files]
        self.author, self.comments, self.reviews = author, list(comments), list(reviews)
        self.labels, self.draft, self.heads, self.diff, self.counts = list(labels), draft, list(heads), diff, counts
        self.reviewers, self.teams, self.assignees = list(reviewers), list(teams), list(assignees)
        self.state, self.base, self.merged_at = state, base, merged_at
        self.mergeability = list(mergeability)
        self.head_repo_id = 2
        # The pull requests branch forker:fix heads, most recently updated first; 7 stands for this one.
        self.branch_pulls = [{"number": 9, "state": "open", "base": {"ref": "release"}}, 7]
        self.other_pulls = {}
        self.answer_text = answer_text
        self.risk, self.risks = risk, risks
        self.served_model = served_model
        self.trees = dict(trees) if trees is not None else trees_for(self.files)
        self.exported, self.during_model, self.truncated = dict(self.trees), None, False
        self.during_recheck, self.pr_reads = None, 0
        self.writers = {"core-a", "core-b", "maintainer", "ci-bot"}
        self.label_definitions = set(GATE.MANAGED_LABELS) | {"merge-conflict"}
        self.label_writes, self.during_label_write = [], None
        self.review_writes, self.during_approve = [], None
        gate_run = {"path": f"{GATE.WORKFLOW}@main", "event": "workflow_run", "head_repository": {"full_name": REPO}}
        self.workflow_runs = {
            RUN_ID: gate_run, GATE_RUN: dict(gate_run, event="issue_comment"),
            OTHER_RUN: {"path": ".github/workflows/rust.yml@main", "event": "push",
                        "head_repository": {"full_name": REPO}},
            PR_RUN: {"path": f"{GATE.WORKFLOW}@refs/pull/7/merge", "event": "pull_request",
                     "head_repository": {"full_name": "forker/jolt"}},
        }
        digest = GATE.classifier_inputs_sha256()
        self.check_runs = [{"id": index, "output": {"title": "Cached classification", "summary": "cached",
                                                    "text": json.dumps({"risk": risk, "classifier_inputs_sha256": digest})},
                            **spec} for index, spec in enumerate(records)]
        self.requests, self.created, self.posted, self.model_requests = [], [], [], []

    def __call__(self, method, url, headers, body):
        self.requests.append((method, url))
        parts = urlsplit(url)
        data = json.loads(body) if body else None
        path, query = parts.path, parse_qs(parts.query)
        if method == "GET" and path in self.other_pulls:
            return self.ok(self.other_pulls[path])
        if (method, path) == ("GET", f"/repos/{REPO}/pulls/7"):
            self.pr_reads += 1
            if self.pr_reads == 2 and self.during_recheck:
                self.during_recheck(self)
            head = self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]
            mergeable = self.mergeability.pop(0) if len(self.mergeability) > 1 else self.mergeability[0]
            return self.ok({"number": 7, "state": self.state, "merged_at": self.merged_at, "draft": self.draft,
                            "mergeable": mergeable,
                            "base": {"ref": self.base, "repo": {"id": 1}},
                            "head": {"sha": head, "ref": "fix", "repo": {"id": self.head_repo_id}}, "user": {"login": self.author},
                            "labels": [{"name": name} for name in self.labels],
                            "requested_reviewers": [{"login": login} for login in self.reviewers],
                            "requested_teams": [{"slug": slug} for slug in self.teams],
                            "assignees": [{"login": login} for login in self.assignees],
                            "changed_files": self.counts[0], "additions": self.counts[1],
                            "deletions": self.counts[2]})
        if (method, path) == ("POST", "/graphql"):
            nodes = self.comments if "comments(" in data["query"] else self.reviews
            return self.ok({"data": {"repository": {"pullRequest": {
                "comments" if "comments(" in data["query"] else "reviews": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": nodes}}}}})
        if method == "GET" and path.startswith(f"/repos/{REPO}/collaborators/"):
            login = path.split("/")[-2]
            return self.ok({"permission": "write" if login in self.writers else "read"})
        if method == "GET" and path.startswith(f"/repos/{REPO}/actions/runs/"):
            run = self.workflow_runs.get(path.rsplit("/", 1)[1])
            return self.ok(run) if run else (404, b"{}")
        if (method, path) == ("GET", f"/repos/{REPO}/compare/main...{HEAD}"):
            return self.ok({"merge_base_commit": {"sha": BASE}})
        if (method, path) == ("GET", f"/repos/{REPO}/compare/{BASE}...{HEAD}"):
            assert headers["Accept"] == "application/vnd.github.diff"
            return 200, self.diff.encode()
        if method == "GET" and path.startswith(f"/repos/{REPO}/git/trees/"):
            return self.ok(dict(git_tree(self.trees[path.rsplit("/", 1)[1]])[0], truncated=self.truncated))
        if method == "GET" and path.startswith(f"/repos/{REPO}/tarball/"):
            return 200, git_tree(self.exported[path.rsplit("/", 1)[1]])[1]
        if method == "GET" and path.endswith("/check-runs"):
            sha = path.split("/")[-2]
            runs = [run for run in self.check_runs
                    if run["name"] == query["check_name"][0] and run["head_sha"] == sha]
            return self.ok({"total_count": len(runs), "check_runs": runs})
        if (method, path) == ("POST", f"/repos/{REPO}/check-runs"):
            run = dict(data, id=len(self.check_runs), completed_at="2026-10-30T01:00:00Z",
                       app={"id": GATE.ACTIONS_APP_ID, "slug": "github-actions"})
            self.created.append(run)
            self.check_runs.append(run)
            return self.ok(run)
        if method == "PATCH" and path.startswith(f"/repos/{REPO}/check-runs/"):
            run = self.check_runs[int(path.rsplit("/", 1)[1])]
            run.update(data)
            return self.ok(run)
        if (method, path) == ("POST", f"/repos/{REPO}/issues/7/comments"):
            self.posted.append(data["body"])
            self.comments.append({"author": {"__typename": "Bot", "login": "github-actions"}, "editor": None,
                                  "body": data["body"], "lastEditedAt": None,
                                  "fullDatabaseId": str(100 + len(self.posted))})
            return self.ok({"id": 100 + len(self.posted)})
        if method == "PATCH" and path.startswith(f"/repos/{REPO}/issues/comments/"):
            [node] = [node for node in self.comments if node["fullDatabaseId"] == path.rsplit("/", 1)[1]]
            node["body"] = data["body"]
            self.posted.append(data["body"])
            return self.ok({"id": int(node["fullDatabaseId"])})
        if (method, path) == ("POST", f"/repos/{REPO}/pulls/7/reviews"):
            assert data["event"] == "APPROVE", data
            node = {"author": {"__typename": "Bot", "login": "github-actions"}, "body": data["body"],
                    "lastEditedAt": None, "state": "APPROVED", "commit": {"oid": data["commit_id"]},
                    "fullDatabaseId": str(300 + len(self.review_writes))}
            self.reviews.append(node)
            self.review_writes.append(("APPROVE", data["commit_id"]))
            if self.during_approve:
                self.during_approve(self)
            return self.ok({"id": int(node["fullDatabaseId"]), "state": "APPROVED", "commit_id": data["commit_id"]})
        if path.startswith(f"/repos/{REPO}/pulls/7/reviews/"):
            [node] = [node for node in self.reviews if node.get("fullDatabaseId") == path.split("/")[7]]
            if method == "PUT" and path.endswith("/dismissals"):
                if node["state"] != "APPROVED":
                    return 422, b'{"message": "only an approval can be dismissed"}'
                node["state"] = "DISMISSED"
                self.review_writes.append(("DISMISS", node["commit"]["oid"]))
            assert method in ("GET", "PUT"), method
            return self.ok({"id": int(node["fullDatabaseId"]), "state": node["state"]})
        if method == "GET" and path.startswith(f"/repos/{REPO}/labels/"):
            name = path.rsplit("/", 1)[1]
            return self.ok({"name": name}) if name in self.label_definitions else (404, b"{}")
        if (method, path) == ("POST", f"/repos/{REPO}/issues/7/labels"):
            self.label_writes.append((method, data["labels"]))
            self.labels = sorted(set(self.labels) | (set(data["labels"]) & self.label_definitions))
            if self.during_label_write:
                self.during_label_write(self)
            return self.ok([{"name": name} for name in self.labels])
        if method == "DELETE" and path.startswith(f"/repos/{REPO}/issues/7/labels/"):
            name = path.rsplit("/", 1)[1]
            if name not in self.labels:
                return 404, b'{"message": "Label does not exist"}'
            self.label_writes.append((method, name))
            self.labels.remove(name)
            return self.ok([{"name": name} for name in self.labels])
        if (method, path) == ("GET", f"/repos/{REPO}/pulls"):
            pulls = [{"number": 7, "state": self.state, "base": {"ref": self.base}} if pull == 7 else pull
                     for pull in self.branch_pulls]
            listed = [pull for pull in pulls if query["state"] in (["all"], [pull["state"]])]
            if "base" in query:
                return self.ok([pull for pull in listed if query["base"] == [pull["base"]["ref"]]])
            return self.ok(listed if query["head"] == ["forker:fix"] else [])
        raise AssertionError(f"unexpected request {method} {url}")

    @staticmethod
    def ok(value):
        return 200, json.dumps(value).encode()

    def gate(self, name=GATE.GATE):
        runs = [run for run in self.created if run["name"] == name]
        return runs[-1].get("conclusion"), runs[-1]["output"]["title"]

    def record(self, name):
        return [run["conclusion"] for run in self.created if run["name"] == name]

    def approved(self):
        """The commits the gate's undismissed approvals name."""
        return [node["commit"]["oid"] for node in self.reviews if node["state"] == "APPROVED"
                and node["body"].startswith(GATE.APPROVAL) and node["author"]["login"] == "github-actions"]


RealClassifier = GATE.Classifier


class FakeClassifier:
    def __init__(self, apis):
        self.apis = apis

    def prepaid(self, _source):
        return False

    def ask(self, instructions, source, admit, counts):
        with tempfile.TemporaryDirectory() as directory:
            prepared = RealClassifier(Path(directory), prepare=True)
            try:
                prepared.ask(instructions, source, admit, counts)
            except GATE.Prepared:
                pass
            manifest = (Path(directory) / "workspace/change.json").read_text()
        self.apis.model_requests.append({"input": manifest, "instructions": instructions})
        if self.apis.during_model:
            self.apis.during_model(self.apis)
        if self.apis.served_model != AGENT.MODEL:
            raise AGENT.AgentFailure("unexpected served model")
        risks = self.apis.risks if self.apis.risks is not None else [
            {"path": path, "level": self.apis.risk, "reasoning": "Classified this changed path."}
            for path in sorted(source.changed())]
        try:
            return json.loads(self.apis.answer_text) if self.apis.answer_text else {
                "reasoning": "prover-only", "risks": risks}
        except ValueError as error:
            raise AGENT.AgentFailure("invalid JSON") from error


def evaluate(apis):
    GATE.Gate(GATE.GitHub(apis, "token", REPO), FakeClassifier(apis), "main", MAINTAINERS, RUN_ID).evaluate(7)
    return apis


def run_main(apis, *argv, env=ENV):
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(GATE, "http", apis), \
            mock.patch.object(GATE, "Classifier", return_value=FakeClassifier(apis)):
        return GATE.main(["ai_merge_gate.py", *argv])


class AiMergeGateTests(unittest.TestCase):
    def assert_completed(self, apis, conclusion):
        [run] = [run for run in apis.created if run["name"] == GATE.GATE]
        self.assertEqual((run["head_sha"], run["status"], run["conclusion"]), (HEAD, "completed", conclusion))

    def test_listed_maintainers_need_current_write_access_to_author_or_attest(self) -> None:
        apis = FakeApis(comments=[comment("core-b", attestation())])
        apis.writers.discard("core-b")
        evaluate(apis)
        self.assertEqual(apis.approved(), [])
        apis = FakeApis(comments=[comment("core-b", attestation())])
        apis.writers.discard("core-a")
        evaluate(apis)
        self.assertEqual((apis.model_requests, apis.approved()), ([], []))
        self.assertIn("needs-maintainer-review", apis.labels)

    def test_risk_four_requires_a_current_head_maintainer_approval_and_never_gets_the_gates(self) -> None:
        for reviews, expected in [
            ([], "neutral"),
            ([review("core-b", "APPROVED", sha=OLD)], "neutral"),
            ([review("maintainer", "APPROVED")], "neutral"),
            ([review("core-a", "APPROVED")], "neutral"),
            ([review("core-b", "DISMISSED")], "neutral"),
            ([review("core-b", "APPROVED")], "success"),
        ]:
            with self.subTest(reviews=reviews):
                apis = evaluate(FakeApis(files=["crates/jolt-verifier/src/lib.rs"], risk=4,
                                         reviews=reviews, comments=[comment("core-a", attestation())]))
                self.assert_completed(apis, expected)
                wanted = {"merge-eligible"} if expected == "success" else {"no-reviewer-assigned"}
                self.assertEqual(set(apis.labels), wanted if expected == "success" else {"risk-4"} | wanted)
                self.assertEqual(apis.review_writes, [], "the attestation never earns a risk-4 approval")

    def test_auto_approval_requires_current_head_attestation_at_each_eligible_risk(self) -> None:
        for risk in (1, 2, 3):
            for sha in (HEAD, OLD):
                with self.subTest(risk=risk, attested_head=sha):
                    apis = evaluate(FakeApis(risk=risk, comments=[comment("core-a", attestation(sha))]))
                    self.assert_completed(apis, "success" if sha == HEAD else "neutral")
                    self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"])
                    self.assertIn(f"risk-{risk}", apis.labels)
                    self.assertEqual(apis.approved(), [HEAD] if sha == HEAD else [])

    def test_current_head_maintainer_approval_satisfies_risk_four_without_a_model(self) -> None:
        apis = FakeApis(reviews=[review("core-b", "APPROVED")])
        self.assertEqual(run_main(apis, "evaluate", "7"), 0)
        self.assertEqual(apis.gate(), ("success", "Risk 4 approved by core-b"))
        self.assertEqual(set(apis.labels), {"merge-eligible"})
        self.assertEqual((apis.model_requests, apis.review_writes), ([], []))

    def test_binary_files_are_exported_and_classified(self) -> None:
        binary, text = "book/image.png", "book/chapter.md"
        diff = (f"diff --git a/{binary} b/{binary}\nBinary files a/{binary} and b/{binary} differ\n"
                + unified_diff(text, "old\n", "new\n"))
        apis = evaluate(FakeApis(files=[binary, text], diff=diff, counts=(2, 1, 1),
                                 trees={BASE: {binary: b"\xff", text: "old\n"},
                                        HEAD: {binary: b"\xfe", text: "new\n"}},
                                 risk=1, comments=[comment("core-a", attestation())]))
        self.assertIn("risk-1", apis.labels)
        self.assertEqual(apis.approved(), [HEAD])

    def test_classifier_risks_choose_the_highest_level(self) -> None:
        for levels, expected in [
            ({"README.md": 1}, 1),
            ({"crates/jolt-kernels/src/lib.rs": 2}, 2),
            ({"Cargo.lock": 2}, 2),
            ({"Cargo.lock": 1}, 1),
            ({"crates/jolt-prover/src/stages/stage3.rs": 2}, 2),
            ({"crates/jolt-verifier/src/config.rs": 1}, 1),
            ({"README.md": 1, "crates/jolt-prover/src/stages/stage3.rs": 2}, 2),
            ({"crates/jolt-akita/src/native_batching.rs": 4}, 4),
            ({"crates/jolt-akita/src/adapters.rs": 3}, 3),
            ({"new/shared.rs": 3}, 3),
            ({".claude/skills/verifier-classifier/SKILL.md": 1}, 1),
            ({"docs/a.md": 1, "src/a.rs": 2, "src/b.rs": 3}, 3),
            ({"src/a.rs": 2, "src/b.rs": 4}, 4),
        ]:
            with self.subTest(levels=levels):
                files = [{"filename": name} for name in levels]
                trees = trees_for(files)
                diff = "".join(unified_diff(name, trees[BASE][name], trees[HEAD][name]) for name in levels)
                apis = evaluate(FakeApis(files=files, trees=trees, diff=diff,
                                         counts=(len(files), len(files), len(files)),
                                         risks=[{"path": name, "level": level, "reasoning": "Scoped changed behavior."}
                                                for name, level in levels.items()],
                                         comments=[comment("core-a", attestation())]))
                wanted = {"no-reviewer-assigned"} if expected == 4 else {"merge-eligible"}
                self.assertEqual(set(apis.labels), {f"risk-{expected}"} | wanted)
                self.assert_completed(apis, "neutral" if expected == 4 else "success")
                self.assertEqual(apis.approved(), [] if expected == 4 else [HEAD])
                [classification] = [run for run in apis.created if run["name"] == GATE.CLASSIFIER]
                self.assertEqual(json.loads(classification["output"]["text"])["risk"], expected)

    def test_policy_waits_are_neutral_and_only_maintainers_receive_risk_labels(self) -> None:
        for kwargs, wanted, model_calls in [
            ({}, {"risk-2"}, 1),
            ({"author": "outsider", "files": ["docs/guide.md"]},
             {"needs-maintainer-review", "no-reviewer-assigned"}, 0),
            ({"author": "outsider"}, {"needs-maintainer-review", "no-reviewer-assigned"}, 0),
        ]:
            with self.subTest(kwargs=kwargs):
                apis = FakeApis(**kwargs)
                self.assertEqual(run_main(apis, "evaluate", "7", env=dict(ENV, OPENAI_API_KEY="sk-test")), 0)
                self.assert_completed(apis, "neutral")
                self.assertEqual(set(apis.labels), wanted)
                self.assertEqual(len(apis.model_requests), model_calls)

    def test_labels_follow_authorization_reviewers_and_merge_eligibility(self) -> None:
        outsider = {"author": "outsider"}
        for case, kwargs, wanted in [
            ("no reviewer", outsider, {"needs-maintainer-review", "no-reviewer-assigned"}),
            ("requested reviewer", dict(outsider, reviewers=["core-b"]), {"needs-maintainer-review"}),
            ("requested team", dict(outsider, teams=["core"]), {"needs-maintainer-review"}),
            ("assignee", dict(outsider, assignees=["core-b"]), {"needs-maintainer-review"}),
            ("approved, unauthorized", dict(outsider, files=["docs/guide.md"], reviews=[review("maintainer", "APPROVED")]),
             {"needs-maintainer-review", "no-reviewer-assigned"}),
            ("outsider risk-4, writer approved", dict(outsider, reviews=[review("maintainer", "APPROVED")]),
             {"needs-maintainer-review", "no-reviewer-assigned"}),
            ("outsider risk-4, maintainer approved", dict(outsider, reviews=[review("core-b", "APPROVED")]),
             {"needs-maintainer-review", "merge-eligible"}),
            ("maintainer and attested", {"comments": [comment("core-a", attestation())]}, {"risk-2", "merge-eligible"}),
            ("draft, approved and attested", {"draft": True, "records": [record(GATE.CLASSIFIER)],
                                              "reviews": [review("maintainer", "APPROVED")],
                                              "comments": [comment("core-a", attestation())]}, set()),
            ("reclassified to risk-4", {"risk": 4, "comments": [comment("core-a", attestation())]},
             {"risk-4", "no-reviewer-assigned"}),
        ]:
            with self.subTest(case=case):
                stale = ["merge-eligible", "needs-maintainer-review", "no-reviewer-assigned", "keep", *GATE.RISK_LABELS]
                apis = evaluate(FakeApis(labels=stale, **kwargs))
                self.assertEqual(set(apis.labels), wanted | {"keep"})
                self.assertEqual(apis.approved(), [HEAD] if case == "maintainer and attested" else [])
                if case == "draft, approved and attested":
                    self.assertEqual(apis.created, [])

    def test_reclassification_replaces_only_the_managed_labels(self) -> None:
        apis = evaluate(FakeApis(risk=3, records=[record(GATE.CLASSIFIER, conclusion="neutral")],
                                 labels=["risk-4", "no-reviewer-assigned", "keep-healed", "benchmark"],
                                 comments=[comment("core-a", attestation())]))
        self.assert_completed(apis, "success")
        self.assertEqual(set(apis.labels), {"risk-3", "merge-eligible", "keep-healed", "benchmark"})
        self.assertEqual(apis.label_writes, [("POST", ["merge-eligible", "risk-3"]),
                                             ("DELETE", "no-reviewer-assigned"), ("DELETE", "risk-4")])
        self.assertEqual(apis.model_requests, [])

    def test_refresh_clears_managed_labels_without_a_trusted_gate_check_on_this_head(self) -> None:
        cases = [
            (None, False),
            (record(GATE.GATE, OLD), False),
            (record(GATE.CLASSIFIER, NEW_HEAD), False),
            (record(GATE.GATE, NEW_HEAD, app={"id": 1}), False),
            (record(GATE.GATE, NEW_HEAD, external_id=OTHER_RUN), False),
            (record(GATE.GATE, NEW_HEAD, external_id=PR_RUN), False),
            (record(GATE.GATE, NEW_HEAD, status="queued", conclusion=None), True),
            (record(GATE.GATE, NEW_HEAD, status="in_progress", conclusion=None), True),
            (record(GATE.GATE, NEW_HEAD, conclusion="failure"), True),
        ]
        for command in (("refresh", "7"), ("refresh-all",)):
            for check, evaluated in cases:
                with self.subTest(command=command, check=check):
                    kept = {"merge-conflict", "keep-healed"}
                    labels = GATE.MANAGED_LABELS | kept
                    apis = FakeApis(heads=[NEW_HEAD], labels=labels, records=[check] if check else [],
                                    mergeability=[False])
                    self.assertEqual(run_main(apis, *command), 0)
                    self.assertEqual(set(apis.labels), labels if evaluated else kept)
                    self.assertEqual((apis.created, apis.review_writes, apis.model_requests), ([], [], []))

    def test_merge_conflicts_follow_mergeability_including_base_pushes_and_unknown_state(self) -> None:
        for command in (("refresh", "7"), ("refresh-all",)):
            for mergeability, present, expected, retries in [
                ([False], False, True, 0),
                ([True], True, False, 0),
                ([None, False], False, True, 1),
                ([None] * 5 + [False], False, True, 5),
                ([None], True, True, 30),
                ([None], False, False, 30),
            ]:
                with self.subTest(command=command, mergeability=mergeability, present=present):
                    labels = {"keep-healed"} | ({"merge-conflict"} if present else set())
                    reads = [None, *mergeability]
                    apis = FakeApis(labels=labels, mergeability=reads, draft=True)
                    with mock.patch("time.monotonic", return_value=0.0) as clock:
                        def advance(seconds):
                            clock.return_value += seconds

                        with mock.patch("time.sleep", side_effect=advance) as delay:
                            self.assertEqual(run_main(apis, *command), 0)
                    self.assertEqual(set(apis.labels), {"keep-healed"} | ({"merge-conflict"} if expected else set()))
                    self.assertEqual(delay.call_count, retries)
                    self.assertEqual((apis.created, apis.review_writes, apis.model_requests), ([], [], []))
        for change in ({"state": "closed"}, {"base": "release"}):
            apis = FakeApis(labels=["merge-conflict"], mergeability=[True], **change)
            self.assertEqual(run_main(apis, "refresh-all"), 0)
            self.assertEqual(apis.labels, ["merge-conflict"])
            self.assertEqual(run_main(apis, "refresh", "7"), 0)
            self.assertIn("merge-conflict", apis.labels)
        missing = FakeApis(mergeability=[False])
        missing.label_definitions.remove("merge-conflict")
        self.assertEqual(run_main(missing, "refresh-all"), 1)
        self.assertEqual(missing.label_writes, [])

        apis = FakeApis(mergeability=[None] * 4 + [False])
        settled_path = f"/repos/{REPO}/pulls/8"
        apis.branch_pulls.append({"number": 8, "state": "open", "base": {"ref": "main"}})
        apis.other_pulls[settled_path] = {"number": 8, "state": "open", "base": {"ref": "main"},
                                         "head": {"sha": HEAD}, "labels": [], "mergeable": True}
        first_pass_reads = []
        with mock.patch("time.monotonic", return_value=0.0) as clock:
            def advance(seconds):
                reads = sum(urlsplit(url).path == settled_path for method, url in apis.requests)
                self.assertGreater(reads, 0, "every PR must be read before the first retry delay")
                if not first_pass_reads:
                    first_pass_reads.append(reads)
                self.assertEqual(reads, first_pass_reads[0], "settled PRs must not be retried")
                clock.return_value += seconds

            with mock.patch("time.sleep", side_effect=advance):
                self.assertEqual(run_main(apis, "refresh-all"), 0)
        self.assertIn("merge-conflict", apis.labels)

    def test_risk_cache_reuse_is_head_bound_and_already_correct_labels_are_not_rewritten(self) -> None:
        for cached_head in (HEAD, OLD):
            with self.subTest(cached_head=cached_head):
                apis = evaluate(FakeApis(records=[record(GATE.CLASSIFIER, cached_head)],
                                         labels=["merge-eligible", "risk-2", "keep-healed"],
                                         comments=[comment("core-a", attestation())]))
                self.assert_completed(apis, "success")
                self.assertEqual(len(apis.model_requests), 0 if cached_head == HEAD else 1)
                self.assertEqual(apis.label_writes, [])
                self.assertEqual(set(apis.labels), {"merge-eligible", "risk-2", "keep-healed"})

    def test_invalid_cached_risk_results_cannot_waive_review(self) -> None:
        for case in ("failed check", "old inputs", "legacy result"):
            with self.subTest(case=case):
                apis = FakeApis(records=[record(GATE.CLASSIFIER)], risk=4,
                                comments=[comment("core-a", attestation())])
                cached = apis.check_runs[0]
                if case == "failed check":
                    cached["conclusion"] = "failure"
                elif case == "old inputs":
                    cached["output"]["text"] = json.dumps({"risk": 2, "classifier_inputs_sha256": "0" * 64})
                else:
                    cached["output"].pop("text")
                evaluate(apis)
                self.assert_completed(apis, "neutral")
                self.assertEqual(len(apis.model_requests), 1)
                self.assertEqual(set(apis.labels), {"risk-4", "no-reviewer-assigned"})
                self.assertEqual(apis.review_writes, [])

    def test_incomplete_or_invalid_risk_classification_fails_closed(self) -> None:
        path = "crates/jolt-prover/src/lib.rs"
        good = {"path": path, "level": 2, "reasoning": "Prover-only code."}
        for risks in ([], [good, good], [{**good, "path": "omitted.rs"}],
                      [{**good, "level": 0}], [{**good, "reasoning": ""}]):
            with self.subTest(risks=risks):
                apis = FakeApis(risks=risks, labels=["merge-eligible", "risk-2", "keep-healed"],
                                comments=[comment("core-a", attestation())])
                evaluate(apis)
                self.assert_completed(apis, "neutral")
                self.assertEqual(set(apis.labels), {"no-reviewer-assigned", "keep-healed"})
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
                self.assertEqual(apis.review_writes, [])

    def test_risk_labels_do_not_publish_for_a_superseded_head(self) -> None:
        apis = FakeApis(labels=["risk-3", "keep-healed"], comments=[comment("core-a", attestation())])

        def push(apis):
            apis.heads = [NEW_HEAD]

        apis.during_model = push
        evaluate(apis)
        self.assertEqual(apis.gate(), ("neutral", "Superseded while the gate ran"))
        self.assertEqual((apis.label_writes, apis.review_writes), ([], []))
        self.assertEqual(set(apis.labels), {"risk-3", "keep-healed"})

    def test_risk_four_rechecks_review_state_and_write_access(self) -> None:
        for change in (lambda apis: apis.reviews[0].update(state="DISMISSED"),
                       lambda apis: apis.reviews[0].update(commit={"oid": OLD}),
                       lambda apis: apis.writers.discard("core-b")):
            apis = FakeApis(risk=4, records=[record(GATE.CLASSIFIER, conclusion="neutral")],
                            reviews=[review("core-b", "APPROVED")], comments=[comment("core-a", attestation())])
            apis.during_recheck = change
            evaluate(apis)
            self.assert_completed(apis, "neutral")
            self.assertEqual(set(apis.labels), {"risk-4", "no-reviewer-assigned"})

    def test_approval_revoked_during_label_publication_cannot_pass(self) -> None:
        for cached in (True, False):
            with self.subTest(cached=cached):
                apis = FakeApis(risk=4, records=[record(GATE.CLASSIFIER, conclusion="neutral")] if cached else [],
                                reviews=[review("core-b", "APPROVED")], comments=[comment("core-a", attestation())])
                apis.during_label_write = lambda apis: apis.reviews[0].update(state="DISMISSED")
                evaluate(apis)
                self.assert_completed(apis, "neutral")
                title = ("Risk 4: current-head maintainer approval required" if cached else
                         "The pull request changed while the gate ran")
                self.assertEqual(apis.gate()[1], title)
                self.assertEqual(set(apis.labels), {"risk-4", "no-reviewer-assigned"} if cached else set())
                self.assertEqual((apis.model_requests, apis.review_writes), ([], []))

    def test_missing_admin_labels_fail_closed(self) -> None:
        missing = FakeApis(comments=[comment("core-a", attestation())])
        missing.label_definitions.clear()
        with self.assertRaisesRegex(GATE.GateError, "admin must create"):
            evaluate(missing)
        self.assert_completed(missing, "failure")
        self.assertEqual((missing.label_writes, missing.review_writes), ([], []))

    def test_current_head_pika_comment_earns_the_gates_approval(self) -> None:
        apis = evaluate(FakeApis(comments=[comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID)]))
        self.assertEqual(apis.gate(), ("success", "Review attested by " + PIKA_LOGIN))
        self.assertEqual((len(apis.model_requests), apis.approved()), (1, [HEAD]))

    def test_attestation_requires_an_exact_line_for_this_pull_and_head(self) -> None:
        valid = attestation()
        cases = [
            (valid, "success"),
            (f"Review complete.\r\n{valid}\r\n", "success"),
            *((attestation(skill="Independent security review", model=model, effort=effort), "success")
              for model, effort in [("GPT-6 Astra", "max"), ("GPT-6 Astra", "xhigh"),
                                    ("Claude Opus 5.5", "max"), ("Claude Opus 5.5", "xhigh")]),
            # Only those exact model and effort spellings count: no aliases, API IDs or case variants.
            (attestation(model="Astra"), "neutral"),
            (attestation(model="gpt-6-astra"), "neutral"),
            (attestation(model="GPT-6.1 Sol"), "neutral"),
            (attestation(model="claude opus 5.5"), "neutral"),
            (attestation(effort="Max"), "neutral"),
            (attestation(effort="high"), "neutral"),
            (f"Review: CONFIRMED {HEAD}", "neutral"),
            (attestation(number=8), "neutral"),
            (attestation(number="07"), "neutral"),
            (attestation(OLD), "neutral"),
            (attestation(HEAD[:7]), "neutral"),
            (attestation(HEAD.upper()), "neutral"),
            (attestation(skill=""), "neutral"),
            (attestation(model=" "), "neutral"),
            (attestation(effort="\t"), "neutral"),
            ("> " + valid, "neutral"),
            ("Quote: " + valid, "neutral"),
            (valid + " Except one issue.", "neutral"),
            (" " + valid, "neutral"),
            (valid.replace("using skill", "using a skill"), "neutral"),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                apis = evaluate(FakeApis(comments=[comment("core-b", body, edited=True)],
                                         records=[record(GATE.CLASSIFIER)]))
                self.assert_completed(apis, expected)
                self.assertEqual(apis.approved(), [HEAD] if expected == "success" else [])
                self.assertEqual(apis.model_requests, [])

    def test_attestation_author_is_a_maintainer_or_the_pinned_pika_bot(self) -> None:
        for node, expected in [
            (comment("core-b", attestation()), "success"),
            (comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID), "success"),
            (comment("outsider", attestation()), "neutral"),
            (comment("maintainer", attestation()), "neutral"),
            (comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id="BOT_other"), "neutral"),
            # The gate's own classifier comment, whose text the model's answer shapes.
            (comment("github-actions", f"<!-- {GATE.CLASSIFIER} -->\n{attestation()}", kind="Bot",
                     actor_id="BOT_actions"), "neutral"),
            (comment("outsider", attestation(), actor_id=PIKA_ID), "neutral"),
            ({**comment("core-a", attestation(), edited=True),
              "editor": {"__typename": "User", "login": "maintainer"}}, "neutral"),
            ({**comment(PIKA_LOGIN, attestation(), edited=True, kind="Bot", actor_id=PIKA_ID),
              "editor": {"__typename": "User", "login": "core-b"}}, "success"),
            ({**comment("core-a", attestation(), edited=True), "editor": None}, "neutral"),
            ({**comment("deleted", attestation()), "author": None}, "neutral"),
        ]:
            with self.subTest(author=node["author"]):
                apis = evaluate(FakeApis(comments=[node], records=[record(GATE.CLASSIFIER)]))
                self.assert_completed(apis, expected)
                self.assertEqual(apis.approved(), [HEAD] if expected == "success" else [])

    def test_a_failed_evaluation_fails_closed_and_withdraws_the_gates_approval(self) -> None:
        unset = {key: value for key, value in ENV.items() if key != "AI_MERGE_GATE_MAINTAINERS"}
        for case, env, labels in [
            ("maintainer list unset", unset, GATE.MANAGED_LABELS),
            ("maintainer list empty", dict(ENV, AI_MERGE_GATE_MAINTAINERS=" , "), GATE.MANAGED_LABELS),
            ("maintainer list not logins", dict(ENV, AI_MERGE_GATE_MAINTAINERS="core-a;core-b"), GATE.MANAGED_LABELS),
            ("label definitions missing", ENV, set()),
        ]:
            with self.subTest(case=case):
                apis = FakeApis(comments=[comment("core-a", attestation())], records=[record(GATE.CLASSIFIER)],
                                reviews=[gate_approval()])
                apis.label_definitions = set(labels)
                self.assertEqual(run_main(apis, "evaluate", "7", env=env), 1)
                self.assertEqual(apis.gate(), ("failure", "Gate error: human approval required"))
                self.assertEqual(apis.review_writes, [("DISMISS", HEAD)])
                self.assertEqual(apis.model_requests, [])

    def test_a_failed_check_run_write_still_withdraws_the_gates_approval(self) -> None:
        class ChecksApiFails(FakeApis):
            def __init__(self, failing, **kwargs):
                super().__init__(**kwargs)
                self.failing = failing

            def __call__(self, method, url, headers, body):
                if method == self.failing and urlsplit(url).path.startswith(f"/repos/{REPO}/check-runs"):
                    self.requests.append((method, url))
                    return 503, b'{"message": "Service Unavailable"}'
                return super().__call__(method, url, headers, body)

        for case, failing, comments, reviews, writes, author, wanted in [
            # The attestation was deleted, and the in-progress check cannot be created.
            ("check creation", "POST", [], [gate_approval()], [("DISMISS", HEAD)],
             "core-a", {"risk-2", "no-reviewer-assigned"}),
            # The gate approves the head, and its decision cannot be written to the check.
            ("check completion", "PATCH", [comment("core-a", attestation())], [], [("APPROVE", HEAD), ("DISMISS", HEAD)],
             "core-a", {"risk-2", "no-reviewer-assigned"}),
            ("outsider check creation", "POST", [], [gate_approval()], [("DISMISS", HEAD)],
             "outsider", {"needs-maintainer-review", "no-reviewer-assigned"}),
            ("outsider check completion", "PATCH", [], [gate_approval()], [("DISMISS", HEAD)],
             "outsider", {"needs-maintainer-review", "no-reviewer-assigned"}),
        ]:
            with self.subTest(case=case):
                apis = ChecksApiFails(failing, author=author, comments=comments, reviews=reviews,
                                      records=[record(GATE.CLASSIFIER)],
                                      labels=[*GATE.RISK_LABELS, "merge-eligible", "keep"])
                with self.assertRaisesRegex(GATE.GateError, f"GitHub {failing} /repos/{REPO}/check-runs.*HTTP 503"):
                    evaluate(apis)
                self.assertEqual((apis.review_writes, apis.approved()), (writes, []))
                self.assertEqual(set(apis.labels), wanted | {"keep"})
                self.assertEqual(apis.model_requests, [])

    def test_same_head_changes_withdraw_only_the_gates_own_approval(self) -> None:
        def delete_attestation(apis):
            apis.comments.pop(0)

        def revoke_author_write_access(apis):
            apis.writers.discard("core-a")

        def edit_attestation_away(apis):
            apis.comments[0].update(body="Review withdrawn.", lastEditedAt="2026-10-31T20:00:00Z")

        def request_changes(apis):
            apis.reviews.append(review("core-b", "CHANGES_REQUESTED"))

        def maintainer_without_write_access_requests_changes(apis):
            apis.writers.discard("core-b")
            request_changes(apis)

        def approval_names_an_earlier_head(apis):
            apis.reviews[0] = gate_approval(OLD)

        for change, writes in [
            (lambda apis: None, []),
            (delete_attestation, [("DISMISS", HEAD)]),
            (edit_attestation_away, [("DISMISS", HEAD)]),
            (revoke_author_write_access, [("DISMISS", HEAD)]),
            (request_changes, [("DISMISS", HEAD)]),
            (maintainer_without_write_access_requests_changes, []),
            # GitHub keeps an approval through a push that leaves the diff unchanged.
            (approval_names_an_earlier_head, [("DISMISS", OLD), ("APPROVE", HEAD)]),
        ]:
            with self.subTest(change=change.__name__):
                pika_review = review(PIKA_LOGIN, "COMMENTED", kind="Bot", body="No issues found.")
                apis = FakeApis(records=[record(GATE.CLASSIFIER)], comments=[
                    comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID),
                ], reviews=[gate_approval(), review("maintainer", "APPROVED"), pika_review])
                change(apis)
                evaluate(apis)
                self.assertEqual(apis.review_writes, writes)
                self.assertEqual([node["state"] for node in apis.reviews[1:3]], ["APPROVED", "COMMENTED"])

    def test_a_head_moving_as_the_gate_approves_withdraws_the_approval(self) -> None:
        apis = FakeApis(comments=[comment("core-a", attestation())], records=[record(GATE.CLASSIFIER)])

        def push(apis):
            apis.heads = [NEW_HEAD]

        apis.during_approve = push
        evaluate(apis)
        self.assertEqual(apis.review_writes, [("APPROVE", HEAD), ("DISMISS", HEAD)])
        self.assertEqual(apis.gate(), ("neutral", "Superseded while the gate ran"))
        self.assertEqual(set(apis.labels), {"risk-2"})

    def test_comment_events_evaluate_pika_but_never_other_bots_comments(self) -> None:
        for kind, login, node_id, eligible in [
            ("User", "core-a", "U_core", True),
            ("Bot", PIKA_LOGIN, PIKA_ID, True),
            ("Bot", "github-actions", "BOT_actions", False),
            ("Bot", PIKA_LOGIN, "BOT_spoof", False),
        ]:
            for action in ("created", "edited", "deleted"):
                with self.subTest(login=login, node_id=node_id, action=action), tempfile.TemporaryDirectory() as directory:
                    body = attestation() if action == "created" else "Review withdrawn."
                    event = {"action": action, "issue": {"number": 7, "pull_request": {"url": "..."}},
                             "comment": {"body": body, "user": {"type": kind, "login": login, "node_id": node_id}}}
                    event_path, output_path = Path(directory) / "event.json", Path(directory) / "output"
                    event_path.write_text(json.dumps(event))
                    comments = [] if action == "deleted" else [comment(login, body, kind=kind, actor_id=node_id)]
                    apis = FakeApis(comments=comments, records=[record(GATE.CLASSIFIER)])
                    env = dict(ENV, GITHUB_EVENT_PATH=str(event_path), GITHUB_EVENT_NAME="issue_comment",
                               GITHUB_OUTPUT=str(output_path))
                    self.assertEqual(run_main(apis, "resolve", env=env), 0)
                    self.assertEqual(output_path.read_text(), "pr=7\n" if eligible else "pr=\n")
                    if eligible:
                        self.assertEqual(run_main(apis, "evaluate", "7", env=env), 0)
                        self.assertEqual(apis.approved(), [HEAD] if action == "created" else [])
                    else:
                        self.assertEqual(apis.created, [])
                    self.assertEqual(apis.model_requests, [])

    def test_no_agent_work_for_other_authors_or_while_draft(self) -> None:
        outsider = FakeApis(author="outsider", comments=[
            comment("outsider", attestation()),
        ])
        draft = FakeApis(author="core-a", draft=True, comments=[comment("core-a", attestation())])
        for apis, title in [(outsider, "Human approval required"),
                            (draft, "Draft: agent checks start when the PR is ready for review")]:
            with self.subTest(title=title):
                evaluate(apis)
                if apis.draft:
                    self.assertEqual(apis.created, [])
                else:
                    self.assertEqual(apis.gate(), ("neutral", title))
                self.assertEqual((apis.model_requests, apis.review_writes), ([], []))
                self.assertFalse([url for _, url in apis.requests
                                  if any(part in url for part in ("api.openai.com", f"/compare/{BASE}...", "/tarball/"))])

    def test_drafts_create_no_checks_or_labels_and_clear_prior_gate_state(self) -> None:
        for prior_state in (False, True):
            with self.subTest(prior_state=prior_state):
                apis = FakeApis(draft=True, comments=[comment("core-a", attestation())],
                                labels=[*GATE.MANAGED_LABELS, "keep"] if prior_state else [],
                                reviews=[gate_approval(), review("core-b", "APPROVED")] if prior_state else [],
                                records=[record(GATE.GATE), record(GATE.CLASSIFIER)] if prior_state else [])
                previous_checks = list(apis.check_runs)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "output"
                    self.assertEqual(run_main(apis, "prepare", "7", env={
                        **ENV, "GITHUB_OUTPUT": str(output), "CLASSIFIER_DIRECTORY": directory}), 0)
                    self.assertEqual(output.read_text(), "needed=false\n")
                self.assertEqual((apis.model_requests, apis.posted), ([], []))
                evaluate(apis)
                self.assertEqual(apis.created, [])
                self.assertEqual(apis.model_requests, [])
                self.assertEqual(apis.check_runs, previous_checks)
                self.assertEqual(set(apis.labels), {"keep"} if prior_state else set())
                self.assertEqual(apis.review_writes, [("DISMISS", HEAD)] if prior_state else [])
                self.assertEqual(apis.approved(), [])
                if not prior_state:
                    self.assertEqual(apis.label_writes, [])

    def test_conversion_to_draft_during_classification_suppresses_result_publication(self) -> None:
        apis = FakeApis(comments=[comment("core-a", attestation())],
                        labels=[*GATE.MANAGED_LABELS, "keep"], reviews=[gate_approval()])
        apis.during_model = lambda apis: setattr(apis, "draft", True)
        evaluate(apis)
        self.assertEqual(apis.record(GATE.CLASSIFIER), [])
        self.assertEqual([run["name"] for run in apis.created], [GATE.GATE])
        self.assertEqual(apis.gate()[0], "neutral")
        self.assertEqual(set(apis.labels), {"keep"})
        self.assertEqual(apis.approved(), [])
        self.assertEqual(apis.review_writes, [("DISMISS", HEAD)])

    def test_unclassified_heads_have_no_risk_labels_and_need_current_maintainer_approval(self) -> None:
        for kwargs in (
            {"answer_text": "invalid JSON"},
            {"files": ["scripts/ci/build-input.py"], "comments": [ledger({HEAD: 2})]},
            {"author": "outsider", "files": ["docs/guide.md"]},
        ):
            for approver, sha, expected in (("maintainer", HEAD, "neutral"),
                                            ("core-b", OLD, "neutral"), ("core-b", HEAD, "success")):
                with self.subTest(kwargs=kwargs, approver=approver, sha=sha):
                    apis = evaluate(FakeApis(labels=[*GATE.RISK_LABELS, "keep"],
                                             reviews=[review(approver, "APPROVED", sha=sha)], **kwargs))
                    self.assertEqual(set(apis.labels) & GATE.RISK_LABELS, set())
                    self.assertEqual(apis.gate()[0], expected)
                    self.assertEqual(apis.approved(), [])

    def test_cached_low_risk_never_waives_an_outsiders_current_maintainer_approval(self) -> None:
        for risk in (1, 2, 3):
            for approver, sha, expected in (("maintainer", HEAD, "neutral"),
                                            ("core-b", OLD, "neutral"), ("core-b", HEAD, "success")):
                with self.subTest(risk=risk, approver=approver, sha=sha):
                    apis = evaluate(FakeApis(author="outsider", risk=risk, records=[record(GATE.CLASSIFIER)],
                                             reviews=[review(approver, "APPROVED", sha=sha)]))
                    self.assertEqual(apis.gate()[0], expected)
                    self.assertEqual(GATE.MERGE_ELIGIBLE in apis.labels, expected == "success")
                    self.assertEqual(set(apis.labels) & GATE.RISK_LABELS, set())
                    self.assertEqual((apis.model_requests, apis.review_writes), ([], []))

    def test_conversion_to_draft_as_the_gate_approves_withdraws_approval_and_labels(self) -> None:
        apis = FakeApis(records=[record(GATE.CLASSIFIER)], labels=["keep"],
                        comments=[comment("core-a", attestation())])
        apis.during_approve = lambda apis: setattr(apis, "draft", True)
        evaluate(apis)
        self.assertEqual(apis.review_writes, [("APPROVE", HEAD), ("DISMISS", HEAD)])
        self.assertEqual(apis.approved(), [])
        self.assertEqual(set(apis.labels), {"keep"})
        self.assertEqual(apis.gate()[0], "neutral")

    def test_maintainer_pr_content_stays_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "executed"
            diff = DIFF.format(sentinel=sentinel)
            changed = f"fn helper() {{}}\n// $(touch {sentinel}) `touch {sentinel}`\n"
            apis = evaluate(FakeApis(diff=diff, trees={**TREES, HEAD: {
                "crates/jolt-prover/src/lib.rs": changed}}, comments=[
                comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID),
            ]))
            self.assertFalse(sentinel.exists())
        self.assertEqual(apis.gate(), ("success", "Review attested by " + PIKA_LOGIN))
        [request] = apis.model_requests
        self.assertNotIn(diff, request["input"])
        self.assertEqual(json.loads(request["input"])["shas"], {"base": BASE, "head": HEAD})
        self.assertIn("# Verifier classifier", request["instructions"])
        self.assertEqual(request["instructions"], (GATE.ROOT / ".claude/skills/verifier-classifier/SKILL.md").read_text())
        self.assertNotIn(diff, request["instructions"])
        self.assertEqual([run["name"] for run in apis.created], [GATE.GATE, GATE.CLASSIFIER])
        admission, verdict = apis.posted
        self.assertTrue(admission.startswith(f"<!-- {GATE.CLASSIFIER} -->\n### Verifier classifier: running"))
        self.assertTrue(verdict.startswith(f"<!-- {GATE.CLASSIFIER} -->"))
        self.assertIn(GATE.model_text("prover-only"), verdict)
        [approval] = [node for node in apis.reviews if node["author"]["login"] == "github-actions"]
        self.assertEqual((approval["commit"]["oid"], approval["state"]), (HEAD, "APPROVED"))
        self.assertTrue(approval["body"].startswith(GATE.APPROVAL))

    def test_a_pass_rests_on_state_read_after_the_model_work(self) -> None:
        attested = [comment("core-a", attestation())]

        def revoke(apis):
            apis.writers.discard("core-a")

        def request_changes(apis):
            apis.reviews.append(review("maintainer", "CHANGES_REQUESTED"))

        def swap_attestation_for_label(apis):
            apis.comments.remove(attested[0])
            apis.labels = ["ai-review"]

        earlier_pass = [record(GATE.GATE)]
        for case, apis, change, expected in [
            ("author write access revoked", FakeApis(comments=attested, records=earlier_pass),
             revoke, ("neutral", "Human approval required")),
            ("writer requested changes", FakeApis(comments=attested, records=earlier_pass),
             request_changes, ("neutral", "Changes requested by maintainer")),
            ("attestation swapped for a CI review request", FakeApis(comments=attested, records=earlier_pass),
             swap_attestation_for_label, ("neutral", "Needs an approval or a current-head review attestation")),
        ]:
            with self.subTest(case=case):
                seen = []

                def during_model(apis, change=change, seen=seen):
                    change(apis)
                    seen.append(apis.gate())

                apis.during_model = during_model
                evaluate(apis)
                self.assertEqual(seen, [(None, "Evaluating")])
                self.assertEqual(apis.gate(), expected)
                self.assertEqual(apis.review_writes, [])
                self.assertEqual(len(apis.model_requests), 1, "the paid verdict is reused, not asked again")
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"])

    def test_push_voids_ai_evidence_for_earlier_heads(self) -> None:
        apis = evaluate(FakeApis(
            comments=[comment("core-a", attestation(OLD))],
            records=[record(GATE.CLASSIFIER, OLD)],
        ))
        self.assertEqual(apis.gate(), ("neutral", "Needs an approval or a current-head review attestation"))
        self.assertEqual(len(apis.model_requests), 1, "the new head needs its own classification")

    def test_only_the_gate_workflows_own_check_runs_are_evidence(self) -> None:
        for case, forged in [
            ("another app", record(GATE.CLASSIFIER, app={"id": 1, "slug": "other-app"})),
            ("an external ID that names no run", record(GATE.CLASSIFIER, external_id="5f2c9d1e-0000-4000-8000")),
            ("another workflow on main", record(GATE.CLASSIFIER, external_id=OTHER_RUN)),
            ("the gate's file run by a pull request", record(GATE.CLASSIFIER, external_id=PR_RUN)),
        ]:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(risk=4, records=[forged], comments=[comment("core-a", attestation())]))
                self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
                self.assertEqual((len(apis.model_requests), apis.review_writes), (1, []))

    def test_head_moving_during_evaluation_publishes_no_pass(self) -> None:
        apis = evaluate(FakeApis(heads=(HEAD, NEW_HEAD), comments=[comment("core-a", attestation())]))
        self.assertEqual(len(apis.model_requests), 1)
        self.assertEqual(apis.gate(), ("neutral", "Superseded while the gate ran"))
        self.assertEqual(apis.review_writes, [])
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"], "the paid verdict stays cached for its head")

    def test_failed_classification_requires_human_review(self) -> None:
        attested = [comment("core-a", attestation())]
        for case, overrides in [("missing risks", {"answer_text": '{"reasoning": "x"}'}),
                                ("not JSON", {"answer_text": "SAFE"}),
                                ("other model", {"served_model": "gpt-6-luna"})]:
            with self.subTest(case=case):
                apis = FakeApis(comments=attested, **overrides)
                evaluate(apis)
                self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])

    def test_human_review_path_follows_latest_review_state(self) -> None:
        cases = [
            ("writer approved head", [review("maintainer", "APPROVED")],
             ("neutral", "Risk 4: current-head maintainer approval required")),
            ("approval of an earlier commit, not dismissed", [review("maintainer", "APPROVED", sha=OLD)],
             ("neutral", "Risk 4: current-head maintainer approval required")),
            ("approval then comment", [review("maintainer", "APPROVED"), review("maintainer", "COMMENTED")],
             ("neutral", "Risk 4: current-head maintainer approval required")),
            ("changes requested", [review("maintainer", "APPROVED"), review("core-b", "CHANGES_REQUESTED", sha=OLD)],
             ("neutral", "Changes requested by core-b")),
            ("dismissed approval", [review("maintainer", "DISMISSED")], ("neutral", "Human approval required")),
            ("read-only approver", [review("fan", "APPROVED")], ("neutral", "Human approval required")),
            ("app approval", [review("ci-bot", "APPROVED", kind="Bot")], ("neutral", "Human approval required")),
            ("the gate's own approval", [gate_approval()], ("neutral", "Human approval required")),
        ]
        for case, reviews, expected in cases:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(author="outsider", reviews=reviews, records=[record(GATE.CLASSIFIER)]))
                self.assertEqual(apis.gate(), expected)
                self.assertEqual(apis.model_requests, [])

    def test_review_bodies_are_not_attestation_comments(self) -> None:
        for state in ("COMMENTED", "DISMISSED"):
            with self.subTest(state=state):
                apis = evaluate(FakeApis(reviews=[review("core-b", state, body=attestation())],
                                         records=[record(GATE.CLASSIFIER)]))
                self.assertEqual(apis.gate(), ("neutral", "Needs an approval or a current-head review attestation"))
                self.assertEqual((apis.model_requests, apis.review_writes), ([], []))

    def test_incomplete_input_fails_closed_before_the_model(self) -> None:
        two_files = ("crates/jolt-prover/src/lib.rs", "crates/jolt-prover/src/stages.rs")
        truncated = FakeApis()
        truncated.truncated = True
        for case, apis in [
            ("diff misses a file", FakeApis(counts=(2, 1, 1), files=two_files)),
            ("tree listing truncated", truncated),
        ]:
            with self.subTest(case=case):
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", "Gate error: human approval required"))
                self.assertEqual(apis.model_requests, [])
                self.assertEqual(apis.posted, [], "no paid attempt is counted")
                self.assertEqual(set(apis.labels), {"no-reviewer-assigned"})

    def test_large_diff_is_prepared_on_disk_and_classified(self) -> None:
        path = "scripts/large.py"
        apis = changed_contents({path: "x" * 160_000 + "\n"}, {path: "y" * 160_000 + "\n"}, risk=1)
        evaluate(apis)
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"])
        self.assertEqual(len(apis.model_requests), 1)
        self.assertLess(len(apis.model_requests[0]["input"]), 1000)
        self.assertIn("risk-1", apis.labels)

    def test_classifier_attempts_are_capped_per_head_and_per_pull_request(self) -> None:
        attested = comment("core-a", attestation())
        # A malformed answer fails the attempt without caching a result, as a timeout would.
        for case, comments, paid in [("two per head", [attested], 2),
                                     ("three per pull request, across heads", [ledger({OLD: 2}), attested], 1)]:
            with self.subTest(case=case):
                apis = FakeApis(comments=comments, answer_text="SAFE")
                counted = []
                apis.during_model = lambda apis, counted=counted: counted.append(json.loads(GATE.LEDGER.fullmatch(apis.posted[-1].splitlines()[-1]).group(1))[HEAD])
                for _ in range(paid):
                    evaluate(apis)
                evaluate(apis)
                self.assertEqual(len(apis.model_requests), paid)
                self.assertEqual(counted, list(range(1, paid + 1)), "each attempt is recorded before it is paid")
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"] * paid + ["skipped"])
                self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
        cached = evaluate(FakeApis(comments=[ledger({HEAD: 2, OLD: 1}), attested], records=[record(GATE.CLASSIFIER)]))
        self.assertEqual((cached.model_requests, cached.gate()), ([], ("success", "Review attested by core-a")))
        corrupt = ledger({HEAD: 1})
        corrupt["body"] = corrupt["body"].replace(": 1}", ": -1}")
        with self.assertRaisesRegex(GATE.GateError, "malformed attempt ledger"):
            evaluate(FakeApis(comments=[corrupt, attested]))

    def test_spent_classifier_attempts_never_earn_the_gates_approval(self) -> None:
        # crates/jolt-algebra/build.rs runs this script; its directory cannot establish verifier safety.
        for case, reviews, expected, labels in [
            ("attested", [], ("neutral", "Risk 4: current-head maintainer approval required"),
             {"no-reviewer-assigned"}),
            ("attested and approved by a writer", [review("maintainer", "APPROVED")],
             ("neutral", "Risk 4: current-head maintainer approval required"), {"no-reviewer-assigned"}),
        ]:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(files=["scripts/fp64_certified_matrix.py"], reviews=reviews,
                                         comments=[ledger({HEAD: 2}), comment("core-a", attestation())]))
                self.assertEqual(apis.gate(), expected)
                self.assertEqual(set(apis.labels), labels)
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["skipped"])
                self.assertEqual((apis.model_requests, apis.review_writes), ([], []))

    def test_path_rows_and_model_input_read_the_evaluated_commits(self) -> None:
        # While the gate evaluates head A, the author pushes B and then restores A. The pull
        # request's file list follows the branch, so it answers with B's one file, matching A's
        # file count; the gate must still judge A's own changes.
        class BranchMovesToBAndBack(FakeApis):
            def __init__(self, a_files, b_files, **kwargs):
                super().__init__(files=a_files, **kwargs)
                self.a_files, self.b_files, self.pull_reads = self.files, b_files, 0

            def __call__(self, method, url, headers, body):
                if (method, urlsplit(url).path) != ("GET", f"/repos/{REPO}/pulls/7"):
                    return super().__call__(method, url, headers, body)
                self.pull_reads += 1
                if self.pull_reads > 1:
                    self.files = self.a_files
                response = super().__call__(method, url, headers, body)
                if self.pull_reads == 1:
                    self.files = self.b_files
                return response

        b_files = [{"filename": "CONTRIBUTING.md", "status": "modified"}]
        attested = [comment("core-a", attestation())]
        for case, apis, expected in [
            ("classifier instruction added on A", BranchMovesToBAndBack(
                [{"filename": ".claude/skills/verifier-classifier/new.md", "status": "added"}],
                b_files, comments=attested),
             ("success", "Review attested by core-a")),
            ("prover file changed on A", BranchMovesToBAndBack(
                ["crates/jolt-prover/src/lib.rs"], b_files, comments=attested),
             ("success", "Review attested by core-a")),
        ]:
            with self.subTest(case=case):
                evaluate(apis)
                [gate] = [run for run in apis.created if run["name"] == GATE.GATE]
                self.assertEqual((gate["head_sha"], gate["status"], gate["conclusion"], gate["output"]["title"]),
                                 (HEAD, "completed", *expected))
                inputs = [request["input"] for request in apis.model_requests]
                self.assertEqual(len(inputs), expected[0] == "success")
                for material in inputs:
                    self.assertEqual(json.loads(material)["paths"], sorted(apis.trees[HEAD]))
                    self.assertNotIn("CONTRIBUTING.md", material)
                trees = {urlsplit(url).path.rsplit("/", 1)[1] for _, url in apis.requests if "/git/trees/" in url}
                self.assertEqual(trees, {BASE, HEAD})

    def test_trusted_input_changes_invalidate_verdict_without_resetting_attempts(self) -> None:
        paths = GATE.CLASSIFIER_INPUTS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in paths:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((GATE.ROOT / path).read_bytes())
            with mock.patch.object(GATE, "ROOT", root):
                for path in paths:
                    for spent in (False, True):
                        with self.subTest(path=path, spent=spent):
                            produced = evaluate(FakeApis())
                            cached = next(run for run in produced.created if run["name"] == GATE.CLASSIFIER)
                            apis = FakeApis(comments=[comment("core-a", attestation()),
                                                      ledger({HEAD: 2 if spent else 1})])
                            apis.check_runs = [cached]
                            unchanged = FakeApis()
                            unchanged.check_runs = [cached]
                            evaluate(unchanged)
                            self.assertEqual(unchanged.model_requests, [])
                            target = root / path
                            original = target.read_bytes()
                            target.write_bytes(original + b"\n ")
                            try:
                                evaluate(apis)
                            finally:
                                target.write_bytes(original)
                            self.assertEqual(len(apis.model_requests), 0 if spent else 1)
                            self.assertEqual(apis.approved(), [] if spent else [HEAD])
                            self.assertIn('"' + HEAD + '": 2', apis.comments[-1]["body"])

    def test_workflow_run_cannot_evaluate_an_unassociated_victim(self) -> None:
        event = {"repository": {"id": 1, "full_name": REPO}, "workflow_run": {
            "path": ".github/workflows/ai-merge-gate-events.yml", "event": "pull_request",
            "repository": {"id": 1, "full_name": REPO},
            "display_title": "AI merge gate events for PR #7",
            "head_repository": {"id": 3, "full_name": "attacker/jolt"},
            "head_branch": "fix", "head_sha": HEAD,
            "pull_requests": [{"number": 9, "base": {"repo": {"id": 1}}}]}}
        for association, source_id, accepted in (
            (event["workflow_run"]["pull_requests"], 3, False), ([], 3, False),
            ([{"number": 7, "base": {"repo": {"id": 1}}}], 2, True), ([], 2, True),
        ):
            with self.subTest(association=association, source_id=source_id):
                event["workflow_run"]["pull_requests"] = association
                event["workflow_run"]["head_repository"]["id"] = source_id
                apis = FakeApis(comments=[comment("core-a", attestation())])
                with tempfile.TemporaryDirectory() as directory:
                    event_path, output = Path(directory) / "event.json", Path(directory) / "output"
                    event_path.write_text(json.dumps(event))
                    env = dict(ENV, GITHUB_EVENT_PATH=str(event_path), GITHUB_EVENT_NAME="workflow_run",
                               GITHUB_OUTPUT=str(output), OPENAI_API_KEY="sk-test")
                    self.assertEqual(run_main(apis, "resolve", env=env), 0)
                    selected = output.read_text().strip().removeprefix("pr=")
                    if selected:
                        run_main(apis, "evaluate", selected, env=env)
                if accepted:
                    self.assertEqual(len(apis.model_requests), 1)
                    self.assertEqual(apis.review_writes, [("APPROVE", HEAD)])
                    self.assertEqual(set(apis.labels), {"risk-2", "merge-eligible"})
                    self.assertEqual(len(apis.created), 2)
                    self.assertEqual(len(apis.posted), 2)
                else:
                    self.assertEqual((len(apis.model_requests), apis.review_writes, apis.label_writes,
                                      apis.created, apis.posted), (0, [], [], [], []))

    def test_review_label_and_cached_ci_reviews_cannot_replace_the_comment(self) -> None:
        old_review = [record("ai-merge-gate/review")]
        for cached_classifier in ([], [record(GATE.CLASSIFIER)]):
            for comments in ([], [comment("core-a", attestation())]):
                with self.subTest(cached_classifier=bool(cached_classifier), comments=bool(comments)):
                    apis = evaluate(FakeApis(labels=["ai-review"], records=old_review + cached_classifier,
                                             comments=comments))
                    self.assert_completed(apis, "success" if comments else "neutral")
                    self.assertEqual(len(apis.model_requests), 0 if cached_classifier else 1)
                    self.assertEqual(apis.record("ai-merge-gate/review"), [])

    def test_resolve_maps_events_to_their_own_pull_request(self) -> None:
        def events_run(title, associations=(7,), fork=True):
            return {"repository": {"id": 1, "full_name": REPO}, "workflow_run": {
                "display_title": title, "path": ".github/workflows/ai-merge-gate-events.yml",
                "event": "pull_request", "repository": {"id": 1, "full_name": REPO},
                "head_repository": {"id": 2 if fork else 1}, "head_branch": "fix", "head_sha": HEAD,
                "pull_requests": [{"number": n, "base": {"repo": {"id": 1}}} for n in associations]}}

        github = GATE.GitHub(FakeApis(), "token", REPO)
        self.assertEqual(GATE.resolve("workflow_run", events_run("AI merge gate events for PR #7"), github), 7)
        self.assertEqual(GATE.resolve("workflow_run", events_run("AI merge gate events for PR #7", ()), github), 7)
        same_repo = FakeApis()
        same_repo.head_repo_id = 1
        self.assertEqual(GATE.resolve("workflow_run", events_run("AI merge gate events for PR #7", (), False),
                                      GATE.GitHub(same_repo, "token", REPO)), 7)
        # Run associations do not select a sibling or replace the exact source binding.
        self.assertEqual(GATE.resolve("workflow_run", events_run("AI merge gate events for PR #7", (9, 7)), github), 7)
        self.assertIsNone(GATE.resolve("workflow_run", events_run("AI merge gate events for PR #9")))
        for field, value in (("head_sha", OLD), ("head_branch", "other"), ("head_repository", {"id": 3})):
            event = events_run("AI merge gate events for PR #7", ())
            event["workflow_run"][field] = value
            self.assertIsNone(GATE.resolve("workflow_run", event, github))
        # Without the run name, GitHub titles the run after the pull request.
        self.assertIsNone(GATE.resolve("workflow_run", events_run("Fix the prover")))
        issue = {"issue": {"number": 3}, "comment": {"user": {"type": "User"}}}
        self.assertIsNone(GATE.resolve("issue_comment", issue))
        pr_comment = {"issue": {"number": 7, "pull_request": {"url": "..."}},
                      "comment": {"user": {"type": "User"}}}
        self.assertEqual(GATE.resolve("issue_comment", pr_comment), 7)

    def test_closing_or_retargeting_withdraws_the_gates_approval_and_merge_eligible(self) -> None:
        def route_event(apis, associations=()):
            """Resolve the events workflow's run for pull request 7, named by its run name, then evaluate."""
            with tempfile.TemporaryDirectory() as directory:
                event_path, output_path = Path(directory) / "event.json", Path(directory) / "output"
                event_path.write_text(json.dumps({"repository": {"id": 1, "full_name": REPO}, "workflow_run": {
                    "path": ".github/workflows/ai-merge-gate-events.yml", "event": "pull_request",
                    "repository": {"id": 1, "full_name": REPO},
                    "display_title": "AI merge gate events for PR #7",
                    "head_repository": {"id": apis.head_repo_id}, "head_branch": "fix", "head_sha": HEAD,
                    "pull_requests": [{"number": n, "base": {"repo": {"id": 1}}}
                                      for n in associations]}}))
                env = dict(ENV, GITHUB_EVENT_PATH=str(event_path), GITHUB_EVENT_NAME="workflow_run",
                           GITHUB_OUTPUT=str(output_path))
                self.assertEqual(run_main(apis, "resolve", env=env), 0)
                self.assertEqual(output_path.read_text(), "pr=7\n")
            self.assertEqual(run_main(apis, "evaluate", "7"), 0)

        # The pull request's branch also heads pull request 9, listed with it newest first.
        into_main = {"number": 9, "state": "open", "base": {"ref": "main"}}
        newer = {"number": 9, "state": "open", "base": {"ref": "release"}}
        for head_repo_id, associations, case, change, branch_pulls, writes in [
            (repo_id, association, *case)
            for repo_id in (1, 2) for association in ([], [9]) for case in [
            ("closed, the branch heading another into main", {"state": "closed"}, [7, into_main],
             [("DISMISS", HEAD)]),
            ("retargeted, the branch heading a newer one", {"base": "release"}, [newer, 7], [("DISMISS", HEAD)]),
            # A merged pull request cannot reopen; its approval records what let it merge.
            ("merged", {"state": "closed", "merged_at": "2026-11-01T12:00:00Z"}, [newer, 7], []),
        ]]:
            with self.subTest(case=case, head_repo_id=head_repo_id, associations=associations):
                apis = FakeApis(records=[record(GATE.CLASSIFIER)], comments=[comment("core-a", attestation())],
                                reviews=[gate_approval(), review("maintainer", "APPROVED")],
                                labels=["risk-2", "merge-eligible", "keep"], **change)
                apis.branch_pulls = branch_pulls
                apis.head_repo_id = head_repo_id
                route_event(apis, associations)
                self.assertEqual(apis.review_writes, writes)
                self.assertEqual(set(apis.labels), {"risk-2", "keep"})
                self.assertEqual(apis.reviews[1]["state"], "APPROVED")
                self.assertEqual((apis.model_requests, apis.created), ([], []))
                if change == {"base": "release"}:
                    # Targeting main again evaluates it as before.
                    apis.base = "main"
                    route_event(apis, associations)
                    self.assertEqual(apis.review_writes, writes + [("APPROVE", HEAD)])
                    self.assertEqual(set(apis.labels), {"risk-2", "merge-eligible", "keep"})


class ClassifierAgentTests(unittest.TestCase):
    def test_upstream_failures_keep_safe_bounded_diagnostics(self):
        key = "fixture-secret-key"
        for phase, status, body, details in (
            ("preflight", 401, {"error": {"type": "invalid_request_error", "code": "invalid_api_key",
                                         "message": key}}, ("invalid_request_error", "invalid_api_key")),
            ("request", 429, {"error": {"type": "insufficient_quota", "code": "insufficient_quota",
                                       "message": key}}, ("insufficient_quota",)),
            ("preflight", 502, b"not JSON: " + key.encode(), ()),
            ("request", 503, {"error": {"type": key + "x" * 5000, "code": "code_" + key}},
             ("[REDACTED]", "code_[REDACTED]")),
            ("request", None, key + "x" * 5000, ("[REDACTED]",)),
        ):
            with self.subTest(phase=phase, status=status):
                payload = body if isinstance(body, bytes) else json.dumps(body).encode()
                def upstream(request, **_kwargs):
                    if phase == "request" and request.full_url.endswith("input_tokens"):
                        response = io.BytesIO(b'{"input_tokens": 100}')
                        response.status = 200
                        return response
                    if status is None:
                        raise RuntimeError(body)
                    raise HTTPError(request.full_url, status, "upstream refused", None, io.BytesIO(payload))
                broker = AGENT.ModelBroker(key)
                with mock.patch.object(AGENT.urllib.request, "urlopen", side_effect=upstream), broker.serve() as endpoint:
                    with self.assertRaises(HTTPError) as stopped:
                        urlopen(Request(endpoint + "/v1/responses", json.dumps({"model": AGENT.MODEL}).encode(),
                                        {"Content-Type": "application/json"}), timeout=5)
                    with stopped.exception as response:
                        self.assertEqual(json.load(response)["error"]["message"], "classifier stopped")
                if status is not None:
                    self.assertIn(f"HTTP {status}", broker.failure)
                self.assertNotIn(key, broker.failure)
                self.assertLessEqual(len(broker.failure), 512)
                for detail in details:
                    self.assertIn(detail, broker.failure)
                self.assertEqual(broker.usage, [])

    def test_cached_context_can_repeat_within_spend_budget(self):
        usage = {"input_tokens": 100_000, "output_tokens": 4000,
                 "input_tokens_details": {"cached_tokens": 100_000}}
        response = {"type": "response.completed", "response": {
            "model": AGENT.MODEL, "status": "completed", "usage": usage}}
        payload = json.dumps(response).encode()
        def request(endpoint, _body, _headers):
            return 200, (json.dumps({"input_tokens": 100_000}).encode()
                         if endpoint.endswith("input_tokens") else payload)
        broker = AGENT.ModelBroker("fake", request=request)
        for _ in range(40):
            self.assertEqual(broker.forward({"model": AGENT.MODEL}, {}), (200, payload))
        self.assertEqual((broker.input_tokens, broker.output_tokens), (4_000_000, 160_000))
        self.assertAlmostEqual(broker.cost_usd, 2)

    def test_long_context_prices_the_entire_request_at_the_surcharge(self):
        for count, expected in ((272_000, .199), (272_001, .393004)):
            with self.subTest(input_tokens=count):
                self.assertAlmostEqual(AGENT.request_cost(count, 1000, cached=200_000, written=50_000), expected)

    def test_compaction_request_uses_the_budgeted_broker(self):
        calls = []
        def request(endpoint, body, _headers):
            calls.append(endpoint)
            if endpoint.endswith("input_tokens"):
                return 200, json.dumps({"input_tokens": 100_000}).encode()
            self.assertEqual(body["max_output_tokens"], 128_000)
            response = {"type": "response.completed", "response": {"model": AGENT.MODEL,
                "status": "completed", "usage": {"input_tokens": 100_000, "output_tokens": 200,
                    "input_tokens_details": {"cached_tokens": 90_000, "cache_write_tokens": 10_000}}}}
            return 200, b"data: " + json.dumps(response).encode() + b"\n\n"
        broker = AGENT.ModelBroker("fake", request=request)
        with tempfile.TemporaryDirectory() as directory, broker.serve() as endpoint:
            root = Path(directory)
            (root / "agent-state").mkdir()
            AGENT.configure_codex(endpoint, root)
            config = tomllib.loads((root / "agent-state/config.toml").read_text())
            self.assertEqual(config["model_context_window"], 1_050_000)
            self.assertEqual(config["model_providers"]["gate"]["stream_idle_timeout_ms"], 18_000_000)
            body = {"model": AGENT.MODEL, "stream": True,
                    "input": [{"role": "user", "content": [{"type": "input_text",
                        "text": "Summarize the conversation so it can continue."}]}]}
            def compact():
                return urlopen(Request(endpoint + "/v1/responses", json.dumps(body).encode(),
                                       {"Content-Type": "application/json"}), timeout=5)
            with compact() as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(broker.input_tokens, 100_000)
            self.assertAlmostEqual(broker.cost_usd, .036)
            broker.cost_usd = AGENT.MAX_SPEND_USD
            with self.assertRaises(HTTPError) as error:
                compact()
            self.assertEqual(error.exception.code, 429)
            self.assertEqual(broker.failure, "classifier spend cap")
        self.assertEqual(calls.count("/v1/responses"), 1)

    def test_budget_preflight_blocks_payment_and_incomplete_usage_is_retained(self):
        for count, cost, model, error in (
            (AGENT.MAX_INPUT_TOKENS + 1, 0, AGENT.MODEL, "per-request input cap"),
            (AGENT.MAX_INPUT_TOKENS, AGENT.MAX_SPEND_USD, AGENT.MODEL, "spend cap"),
            (300_000, 47, AGENT.MODEL, "spend cap"),
            (272_000, 48.04, AGENT.MODEL, "spend cap"),
            (None, 0, AGENT.MODEL, "input token count"),
            (100, 0, "other-model", "unexpected requested model"),
        ):
            with self.subTest(count=count, cost=cost, model=model):
                calls = []
                def request(endpoint, _body, _headers):
                    calls.append(endpoint)
                    return 200, json.dumps({"input_tokens": count}).encode()
                broker = AGENT.ModelBroker("fake", request=request)
                broker.cost_usd = cost
                with self.assertRaisesRegex(AGENT.AgentFailure, error):
                    broker.forward({"model": model}, {})
                self.assertNotIn("/v1/responses", calls)
        def incomplete(endpoint, _body, _headers):
            response = {"type": "response.incomplete", "response": {"model": AGENT.MODEL,
                "status": "incomplete", "usage": {"input_tokens": 100, "output_tokens": 20_000}}}
            return 200, (json.dumps({"input_tokens": 100}).encode() if endpoint.endswith("input_tokens")
                         else b"data: " + json.dumps(response).encode() + b"\n\n")
        broker = AGENT.ModelBroker("fake", request=incomplete)
        with self.assertRaisesRegex(AGENT.AgentFailure, "output cap"):
            broker.forward({"model": AGENT.MODEL, "stream": True}, {})
        self.assertEqual((broker.input_tokens, broker.output_tokens), (100, 20_000))
        self.assertAlmostEqual(broker.cost_usd, .2002)

    def test_wall_cap_kills_and_reaps_only_the_owned_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace").mkdir()
            (root / "instructions.md").write_text("Read only.")
            (root / "plan.json").write_text(json.dumps({"paths": ["docs/a.md"]}))
            executable = root / "fake-codex"
            executable.write_text("#!/bin/sh\nexec sleep 2\n")
            executable.chmod(0o755)
            owned = []
            real_popen = subprocess.Popen
            def start(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                owned.append(process)
                return process
            with mock.patch.object(AGENT.shutil, "which", return_value=str(executable)), \
                    mock.patch.object(AGENT, "WALL_SECONDS", .05), \
                    mock.patch.object(AGENT.subprocess, "Popen", side_effect=start):
                answer, receipt = AGENT.run_agent(root, "fake-key")
            self.assertIsNone(answer)
            self.assertEqual(receipt["failure"], "classifier wall-time cap")
            self.assertEqual(owned[0].returncode, -signal.SIGKILL)
            self.assertEqual(owned[0].wait(), -signal.SIGKILL)
            self.assertIsNone(json.loads((root / "result.json").read_text()))

    def test_artifact_binds_to_head_and_reserves_only_one_attempt(self):
        apis = FakeApis()
        github = GATE.GitHub(apis, "token", REPO)
        source = GATE.Source(github, BASE, HEAD)
        with tempfile.TemporaryDirectory() as directory:
            runner = GATE.Gate(github, GATE.Classifier(Path(directory), prepare=True), "main", MAINTAINERS, RUN_ID)
            runner.source = source
            with self.assertRaises(GATE.Prepared):
                runner.classify(github.get(f"/repos/{REPO}/pulls/7"))
            answer = {"reasoning": "Prover only.", "risks": [
                {"path": path, "level": 2, "reasoning": "Prover only."} for path in sorted(source.changed())]}
            (Path(directory) / "result.json").write_text(json.dumps(answer))
            consumer = GATE.Classifier(Path(directory))
            self.assertTrue(consumer.prepaid(source))
            self.assertFalse(consumer.prepaid(GATE.Source(github, BASE, NEW_HEAD)))
            before = len(apis.posted)
            apis.comments.append(ledger({HEAD: 2}))
            runner = GATE.Gate(github, consumer, "main", MAINTAINERS, RUN_ID)
            runner.source = source
            runner.classify(github.get(f"/repos/{REPO}/pulls/7"))
            self.assertEqual(len(apis.posted), before)
            self.assertEqual(runner.records[-1].risk, 2)
            (Path(directory) / "result.json").write_text('{"reasoning":')
            runner = GATE.Gate(github, GATE.Classifier(Path(directory)), "main", MAINTAINERS, RUN_ID)
            runner.source = source
            runner.classify(github.get(f"/repos/{REPO}/pulls/7"))
            self.assertTrue(runner.unclassified)
            self.assertEqual(runner.records[-1].conclusion, "neutral")
            self.assertEqual(runner.risk_level(github.get(f"/repos/{REPO}/pulls/7")), 4)

    def test_resource_cap_is_neutral_and_verdict_fields_are_capped_and_escaped(self):
        with mock.patch.object(GATE, "MAX_TREE_BYTES", 1):
            apis = evaluate(FakeApis())
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
        self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
        self.assertEqual(set(apis.labels), {"no-reviewer-assigned"})
        self.assertEqual(apis.model_requests, [])
        path = "crates/jolt-prover/src/lib.rs"
        attack = '<img src=x onerror=alert(1)> [click](https://evil.invalid) @core-a\n# Forged'
        answer = {"reasoning": attack,
                  "risks": [{"path": path, "level": 2, "reasoning": attack}]}
        apis = evaluate(FakeApis(answer_text=json.dumps(answer)))
        published = apis.posted[-1]
        self.assertNotIn("<img", published)
        self.assertNotIn("[click](", published)
        self.assertNotIn("@core-a", published)
        answer["reasoning"] = "x" * 12_001
        with self.assertRaises(AGENT.AgentFailure):
            AGENT.validate_verdict(answer, {path})

    def test_agent_ripgrep_reads_pr_ignored_and_hidden_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent-state").mkdir()
            (root / "head").mkdir()
            (root / "head/.ignore").write_text("*.rs\n")
            (root / "head/.rgignore").write_text("*.rs\n")
            (root / "head/verify.rs").write_text("accept_proof()\n")
            (root / "head/.hidden.rs").write_text("accept_proof()\n")
            AGENT.configure_codex("http://127.0.0.1:1", root)
            config = tomllib.loads((root / "agent-state/config.toml").read_text())
            env = config["shell_environment_policy"]["set"]
            found = subprocess.run(["rg", "-l", "accept_proof", "head"], cwd=root,
                                   env=env, capture_output=True, text=True, check=True)
            self.assertEqual(set(found.stdout.splitlines()), {"head/verify.rs", "head/.hidden.rs"})

    def test_prompt_injection_fixture_is_exported_as_data(self):
        attack = (Path(__file__).parent / "fixtures/ai_merge_gate_prompt_injection.md").read_text()
        path = "docs/untrusted-review.txt"
        apis = changed_contents({path: "Documentation.\n"}, {path: attack})
        with tempfile.TemporaryDirectory() as directory:
            source = GATE.Source(GATE.GitHub(apis, "token", REPO), BASE, HEAD)
            with self.assertRaises(GATE.Prepared):
                GATE.Classifier(Path(directory), prepare=True).ask("Trusted instructions.", source, lambda: None, apis.counts)
            self.assertEqual((Path(directory) / "workspace/head" / path).read_text(), attack)
            self.assertNotIn(attack, (Path(directory) / "instructions.md").read_text())

if __name__ == "__main__":
    unittest.main()
