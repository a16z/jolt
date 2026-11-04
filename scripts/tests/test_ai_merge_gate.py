import contextlib
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

# The gate imports its sibling diff_classifier the way it runs in CI, from scripts/ci.
sys.path.insert(0, str(Path(__file__).parents[1] / "ci"))
import ai_merge_gate as GATE

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
       "AI_MERGE_GATE_MAINTAINERS": MAINTAINERS, "GITHUB_RUN_ID": RUN_ID}
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
        if status == "renamed":
            base[entry["previous_filename"]] = head[new] = f"moved to {new}\n"
            continue
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
    """GitHub and OpenAI as the gate reaches them: canned state in, every request recorded."""

    def __init__(self, author="core-a", comments=(), reviews=(), labels=(), draft=False,
                 records=(), heads=(HEAD,), diff=PLAIN_DIFF, files=("crates/jolt-prover/src/lib.rs",),
                 counts=(1, 1, 1), verdict="SAFE", risk=2, risks=None, answer_text=None,
                 served_model="gpt-6.1-sol", trees=None, tool_calls=(),
                 reviewers=(), teams=(), assignees=(), state="open", base="main", merged_at=None) -> None:
        # The pull request's file list, which follows its branch; the commits' trees do not move.
        self.files = [entry if isinstance(entry, dict) else {"filename": entry} for entry in files]
        self.changed_paths = {entry["filename"] for entry in self.files} | {
            entry["previous_filename"] for entry in self.files if "previous_filename" in entry}
        self.author, self.comments, self.reviews = author, list(comments), list(reviews)
        self.labels, self.draft, self.heads, self.diff, self.counts = list(labels), draft, list(heads), diff, counts
        self.reviewers, self.teams, self.assignees = list(reviewers), list(teams), list(assignees)
        self.state, self.base, self.merged_at = state, base, merged_at
        self.head_repo_id = 2
        # The pull requests branch forker:fix heads, most recently updated first; 7 stands for this one.
        self.branch_pulls = [{"number": 9, "state": "open", "base": {"ref": "release"}}, 7]
        self.verdict, self.answer_text = verdict, answer_text
        self.risk, self.risks = risk, risks
        self.served_model, self.tool_calls = served_model, tool_calls
        # Input and output tokens a response reports, given the size in bytes of its request.
        self.usage = lambda size: (1_000, 100)
        self.trees = dict(trees) if trees is not None else trees_for(self.files)
        self.exported, self.during_model, self.truncated = dict(self.trees), None, False
        self.during_recheck, self.pr_reads = None, 0
        self.writers = {"core-a", "core-b", "maintainer", "ci-bot"}
        self.label_definitions = set(GATE.MANAGED_LABELS)
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
        if parts.netloc == "api.openai.com":
            return self.openai(method, parts.path, data, len(body or b""))
        path, query = parts.path, parse_qs(parts.query)
        if (method, path) == ("GET", f"/repos/{REPO}/pulls/7"):
            self.pr_reads += 1
            if self.pr_reads == 2 and self.during_recheck:
                self.during_recheck(self)
            head = self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]
            return self.ok({"number": 7, "state": self.state, "merged_at": self.merged_at, "draft": self.draft,
                            "base": {"ref": self.base, "repo": {"id": 1}},
                            "head": {"sha": head, "ref": "fix", "repo": {"id": self.head_repo_id}}, "user": {"login": self.author},
                            "labels": [{"name": name} for name in self.labels],
                            "requested_reviewers": [{"login": login} for login in self.reviewers],
                            "requested_teams": [{"slug": slug} for slug in self.teams],
                            "assignees": [{"login": login} for login in self.assignees],
                            "changed_files": self.counts[0], "additions": self.counts[1],
                            "deletions": self.counts[2]})
        if (method, path) == ("GET", f"/repos/{REPO}/pulls/7/files"):
            page = int(query["page"][0])
            return self.ok(self.files[100 * (page - 1):100 * page])
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
            return self.ok(listed if query["head"] == ["forker:fix"] else [])
        raise AssertionError(f"unexpected request {method} {url}")

    def openai(self, method, path, data, size):
        assert (method, path) == ("POST", "/v1/responses"), (method, path)
        assert data["text"]["format"]["name"] == "verifier_classification", "only the classifier may call a model"
        self.model_requests.append(data)
        if self.during_model:
            self.during_model(self)
        if self.tool_calls and "previous_response_id" not in data:
            output = [{"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}", "name": name,
                       "arguments": json.dumps(arguments)} for index, (name, arguments) in enumerate(self.tool_calls)]
        else:
            risks = self.risks if self.risks is not None else [
                {"path": path, "level": self.risk, "reasoning": "Classified this changed path."}
                for path in sorted(self.changed_paths)]
            text = self.answer_text or json.dumps({"verdict": self.verdict, "reasoning": "prover-only", "risks": risks})
            output = [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
        input_tokens, output_tokens = self.usage(size)
        return self.ok({"id": f"resp_{len(self.model_requests)}", "status": "completed", "model": self.served_model,
                        "reasoning": data["reasoning"], "output": output,
                        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})

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


def evaluate(apis):
    def no_polling(_seconds):
        raise AssertionError("a completed response needs no polling")

    model = GATE.Model(apis, "sk-test", sleep=no_polling)
    GATE.Gate(GATE.GitHub(apis, "token", REPO), model, "main", MAINTAINERS, RUN_ID).evaluate(7)
    return apis


def run_main(apis, *argv, env=ENV):
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(GATE, "http", apis):
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

    def test_verifier_floor_exempts_only_comments_blanks_and_cfg_test_code(self) -> None:
        path = "crates/jolt-verifier/src/config.rs"
        for old, new, expected in [
            ("fn verify() { reject(); }\n", "fn verify() { accept(); }\n", 4),
            ("#[cfg(test)]\nmod tests { fn check() { reject(); } }\n",
             "#[cfg(test)]\nmod tests { fn check() { accept(); } }\n", 1),
            ('#[cfg(any(test, feature = "prover-fixtures"))]\nfn check() { reject(); }\n',
             '#[cfg(any(test, feature = "prover-fixtures"))]\nfn check() { accept(); }\n', 4),
            ("#[cfg(test)]\nfn check() { reject(); }\n", "fn check() { reject(); }\n", 4),
            ("// old\n/// old\n//! old\n/* old\n * old\n */\nfn verify() {}\n",
             "// new\n/// new\n//! new\n/* new\n * new\n */\n\nfn verify() {}\n", 1),
            ("/*\nfn verify() {}\n*/\n", "fn verify() {}\n", 4),
            ('const TEXT: &str = r#"\n// old\n"#;\n', 'const TEXT: &str = r#"\n// new\n"#;\n', 4),
            ("#[cfg(test)] fn check() {}\nfn verify() { reject(); }\n",
             "#[cfg(test)] fn check() {}\nfn verify() { accept(); }\n", 4),
            ("mod tests {\n#![cfg(test)]\nfn check() {}\n}\nfn verify() { reject(); }\n",
             "mod tests {\n#![cfg(test)]\nfn check() {}\n}\nfn verify() { accept(); }\n", 4),
            ("#![cfg(test)]\nfn check() { reject(); }\n", "#![cfg(test)]\nfn check() { accept(); }\n", 1),
            ("production! {\n#[cfg(test)]\nfn input() { reject(); }\n}\n",
             "production! {\n#[cfg(test)]\nfn input() { accept(); }\n}\n", 4),
            ('#[cfg(test)]\n#[doc = "test"] fn check() {}\nfn verify() { reject(); }\n',
             '#[cfg(test)]\n#[doc = "test"] fn check() {}\nfn verify() { accept(); }\n', 4),
            ("#[cfg(test)]\nfn check() {} fn verify() { reject(); }\n",
             "#[cfg(test)]\nfn check() {} fn verify() { accept(); }\n", 4),
            ("#[cfg(test)]\nconst TEST: bool = true; fn verify() { reject(); }\n",
             "#[cfg(test)]\nconst TEST: bool = true; fn verify() { accept(); }\n", 4),
            ("#[cfg_attr(all(test), inline)]\nfn verify() { reject(); }\n",
             "#[cfg_attr(all(test), inline)]\nfn verify() { accept(); }\n", 4),
            ('#[cfg_attr(all(test, feature = "x"), derive(Debug))]\nstruct Proof { value: u8 }\n',
             '#[cfg_attr(all(test, feature = "x"), derive(Debug))]\nstruct Proof { value: bool }\n', 4),
            ("#![cfg_attr(all(test), allow(dead_code))]\nfn verify() { reject(); }\n",
             "#![cfg_attr(all(test), allow(dead_code))]\nfn verify() { accept(); }\n", 4),
            *[(f"// note{separator}/*\nfn verify() -> bool {{ false }}\n// */\n",
               f"// note{separator}/*\nfn verify() -> bool {{ true }}\n// */\n", 4)
              for separator in ("\x0c", "\r", "\u2028")],
        ]:
            with self.subTest(old=old, new=new):
                apis = evaluate(changed_contents({path: old}, {path: new}, risk=1))
                self.assertIn(f"risk-{expected}", apis.labels)
                self.assertEqual(apis.approved(), [HEAD] if expected == 1 else [])

    def test_affecting_or_uncertain_risk_three_changes_must_be_minimal_in_aggregate(self) -> None:
        for verdict in ("AFFECTS", "UNCERTAIN"):
            for crate in ("jolt-field", "jolt-prover", "jolt-akita"):
                for file_count, lines, expected in [(1, 10, 3), (1, 11, 4), (2, 6, 4), (1, 200, 4)]:
                    with self.subTest(verdict=verdict, crate=crate, file_count=file_count, lines=lines):
                        base = {f"crates/{crate}/src/part{i}.rs": "old();\n" * lines for i in range(file_count)}
                        head = {path: "new();\n" * lines for path in base}
                        apis = evaluate(changed_contents(base, head, verdict=verdict, risk=3))
                        wanted = 4 if crate == "jolt-akita" else expected
                        self.assertIn(f"risk-{wanted}", apis.labels)
                        self.assertEqual(apis.approved(), [HEAD] if wanted == 3 else [])
        prover, field = "crates/jolt-prover/src/big.rs", "crates/jolt-field/src/lib.rs"
        apis = evaluate(changed_contents({prover: "old();\n" * 200, field: "// old\nfn helper() {}\n"},
                                         {prover: "new();\n" * 200, field: "// new\nfn helper() {}\n"},
                                         verdict="UNCERTAIN", risk=3))
        self.assertIn("risk-4", apis.labels)
        self.assertEqual(apis.approved(), [])

    def test_traced_prover_only_changes_override_path_defaults(self) -> None:
        for path in (
            "crates/jolt-akita/src/trace_onehot/decomposition.rs",
            "crates/jolt-sumcheck/src/prover.rs",
            "crates/jolt-prover/src/stages/stage3.rs",
            "crates/jolt-field/src/lib.rs",
            "crates/jolt-transcript/src/prover.rs",
            "crates/jolt-verifier-derive/src/lib.rs",
        ):
            with self.subTest(path=path):
                apis = evaluate(changed_contents({path: "old();\n" * 11},
                                                 {path: "new();\n" * 11}, risk=2))
                self.assertEqual(set(apis.labels), {"risk-2", "merge-eligible"})
                self.assertEqual(apis.approved(), [HEAD])

        prover = "crates/jolt-akita/src/trace_onehot/decomposition.rs"
        shared = "crates/jolt-field/src/lib.rs"
        apis = evaluate(changed_contents(
            {prover: "old();\n" * 11, shared: "old();\n"},
            {prover: "new();\n" * 11, shared: "new();\n"}, verdict="AFFECTS",
            risks=[{"path": prover, "level": 2, "reasoning": "Only prover callers use this helper."},
                   {"path": shared, "level": 3, "reasoning": "Bounded shared arithmetic change."}]))
        self.assertEqual(set(apis.labels), {"risk-3", "merge-eligible"})
        self.assertEqual(apis.approved(), [HEAD])

    def test_unreadable_inputs_fail_early_only_on_floor_paths(self) -> None:
        for crate in ("jolt-field", "jolt-verifier"):
            path = f"crates/{crate}/src/data.bin"
            for data in (b"\xff", b"x" * (GATE.MAX_FILE_BYTES + 1)):
                with self.subTest(crate=crate, bytes=len(data)):
                    apis = FakeApis(files=[path], trees={BASE: {path: "old"}, HEAD: {path: data}})
                    with self.assertRaises(GATE.GateError):
                        evaluate(apis)
                    self.assertEqual(apis.model_requests, [])
                    self.assertEqual(apis.posted, [], "no paid attempt is admitted")
        for binary, text, level in [
            ("book/image.png", "book/chapter.md", 1),
            ("crates/jolt-verifier/fuzz/fixtures/input.bin", "crates/jolt-prover/src/lib.rs", 2),
        ]:
            with self.subTest(binary=binary):
                diff = (f"diff --git a/{binary} b/{binary}\nBinary files a/{binary} and b/{binary} differ\n"
                        + unified_diff(text, "old\n", "new\n"))
                apis = evaluate(FakeApis(files=[binary, text], diff=diff, counts=(2, 1, 1),
                                         trees={BASE: {binary: b"\xff", text: "old\n"},
                                                HEAD: {binary: b"\xfe", text: "new\n"}},
                                         risk=level, comments=[comment("core-a", attestation())]))
                self.assertIn(f"risk-{level}", apis.labels)
                self.assertEqual(apis.approved(), [HEAD])

    def test_dependency_bumps_follow_both_lockfile_closures_without_a_classifier(self) -> None:
        lock = ('version = 4\n\n[[package]]\nname = "jolt-verifier"\nversion = "0.1.0"\n'
                'dependencies = ["middle"]\n\n[[package]]\nname = "middle"\nversion = "1.0.0"\n'
                'dependencies = ["runtime-dep 1.0.0"]\n\n[[package]]\nname = "runtime-dep"\n'
                'version = "1.0.0"\n\n[[package]]\nname = "outside"\nversion = "1.0.0"\n')
        for dependency, expected in [("runtime-dep", 3), ("outside", 2)]:
            for manifest in (None, "version", "rev", "tag"):
                with self.subTest(dependency=dependency, manifest=manifest):
                    old = {"Cargo.lock": lock}
                    new = {"Cargo.lock": lock.replace(f'name = "{dependency}"\nversion = "1.0.0"',
                                                     f'name = "{dependency}"\nversion = "2.0.0"')}
                    if manifest:
                        section = "workspace.dependencies" if manifest == "version" else "patch.crates-io"
                        git = '' if manifest == "version" else ', git = "https://example.invalid/fork"'
                        old["Cargo.toml"] = (f'[{section}]\nalias = {{ package = "{dependency}"{git}, '
                                             f'{manifest} = "1" }}\n')
                        new["Cargo.toml"] = old["Cargo.toml"].replace(f'{manifest} = "1"', f'{manifest} = "2"')
                    apis = evaluate(changed_contents(old, new, verdict="UNCERTAIN", risk=4))
                    self.assertIn(f"risk-{expected}", apis.labels)
                    self.assertEqual(apis.approved(), [HEAD])
                    self.assertEqual(apis.model_requests, [])
        prover = "crates/jolt-prover/src/big.rs"
        for verdict in ("UNCERTAIN", "AFFECTS"):
            with self.subTest(mixed_bump_verdict=verdict):
                old = {"Cargo.lock": lock, prover: "old();\n" * 200}
                new = {"Cargo.lock": lock.replace('name = "runtime-dep"\nversion = "1.0.0"',
                                                  'name = "runtime-dep"\nversion = "2.0.0"'),
                       prover: "new();\n" * 200}
                apis = evaluate(changed_contents(old, new, verdict=verdict, risk=2))
                self.assertIn("risk-4", apis.labels)
                self.assertEqual(apis.approved(), [])

    def test_other_manifest_edits_keep_the_shared_code_floor(self) -> None:
        path = "Cargo.toml"
        lock = 'version = 4\n[[package]]\nname = "jolt-verifier"\nversion = "0.1.0"\n'
        for old, new in [
            ('[dependencies]\nfoo = "1"\n', '[dependencies]\nfoo = "1"\nbar = "2"\n'),
            ('[dependencies]\nfoo = { version = "1", features = ["a"] }\n',
             '[dependencies]\nfoo = { version = "2", features = ["b"] }\n'),
            ('[workspace]\nresolver = "2"\n', '[workspace]\nresolver = "3"\n'),
        ]:
            with self.subTest(new=new):
                apis = evaluate(changed_contents({path: old, "Cargo.lock": lock},
                                                 {path: new, "Cargo.lock": lock}, risk=1))
                self.assertIn("risk-3", apis.labels)
                self.assertEqual(len(apis.model_requests), 1)

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
                apis = evaluate(FakeApis(files=["crates/jolt-verifier/src/lib.rs"], verdict="AFFECTS", risk=4,
                                         reviews=reviews, comments=[comment("core-a", attestation())]))
                self.assert_completed(apis, expected)
                wanted = {"merge-eligible"} if expected == "success" else {"no-reviewer-assigned"}
                self.assertEqual(set(apis.labels), wanted if expected == "success" else {"risk-4"} | wanted)
                self.assertEqual(apis.review_writes, [], "the attestation never earns a risk-4 approval")

    def test_risk_three_attestation_earns_the_gates_approval_of_the_head(self) -> None:
        apis = evaluate(FakeApis(verdict="AFFECTS", risk=3, comments=[comment("core-a", attestation())]))
        self.assertEqual(apis.gate(), ("success", "Review attested by core-a"))
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
        self.assertEqual(set(apis.labels), {"risk-3", "merge-eligible"})
        self.assertEqual(apis.review_writes, [("APPROVE", HEAD)])

    def test_current_head_maintainer_approval_satisfies_risk_four_without_a_model(self) -> None:
        apis = FakeApis(reviews=[review("core-b", "APPROVED")])
        self.assertEqual(run_main(apis, "evaluate", "7"), 0)
        self.assertEqual(apis.gate(), ("success", "Risk 4 approved by core-b"))
        self.assertEqual(set(apis.labels), {"merge-eligible"})
        self.assertEqual((apis.model_requests, apis.review_writes), ([], []))

    def test_risk_paths_and_semantic_results_choose_the_highest_level(self) -> None:
        for levels, expected in [
            ({"README.md": 1}, 1),
            ({"crates/jolt-kernels/src/lib.rs": 2}, 2),
            ({"Cargo.lock": 2}, 2),
            ({"Cargo.lock": 1}, 3),
            ({"crates/jolt-prover/src/stages/stage3.rs": 2}, 2),
            ({"crates/jolt-verifier/src/config.rs": 1}, 4),
            ({"README.md": 1, "crates/jolt-prover/src/stages/stage3.rs": 2}, 2),
            ({"crates/jolt-akita/src/native_batching.rs": 4}, 4),
            ({"crates/jolt-akita/src/adapters.rs": 3}, 4),
            ({"new/shared.rs": 3}, 3),
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
            ({"files": [".claude/skills/verifier-classifier/SKILL.md"]}, {"no-reviewer-assigned"}, 0),
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
            ("reclassified to risk-4", {"verdict": "AFFECTS", "risk": 4, "comments": [comment("core-a", attestation())]},
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
        for case in ("failed check", "old inputs", "policy-only result", "legacy result"):
            with self.subTest(case=case):
                apis = FakeApis(records=[record(GATE.CLASSIFIER)], verdict="UNCERTAIN",
                                comments=[comment("core-a", attestation())])
                cached = apis.check_runs[0]
                if case == "failed check":
                    cached["conclusion"] = "failure"
                elif case == "old inputs":
                    cached["output"]["text"] = json.dumps({"risk": 2, "classifier_inputs_sha256": "0" * 64})
                elif case == "policy-only result":
                    cached["output"]["text"] = json.dumps({"risk": 2, "policy_sha256": hashlib.sha256(
                        (GATE.ROOT / GATE.RISK_POLICY).read_bytes()).hexdigest()})
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
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assert_completed(apis, "failure")
                self.assertEqual(set(apis.labels), {"no-reviewer-assigned", "keep-healed"})
                self.assertEqual(apis.record(GATE.CLASSIFIER), [])
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

    def test_pr_risk_policy_is_data_and_missing_admin_labels_fail_closed(self) -> None:
        files = [{"filename": GATE.RISK_POLICY}, {"filename": "crates/jolt-prover/src/stages/stage3.rs"}]
        trees = trees_for(files)
        trees[HEAD][GATE.RISK_POLICY] = '{"risk-1": {"paths": ["**"]}}'
        apis = evaluate(FakeApis(files=files, trees=trees, reviews=[review("maintainer", "APPROVED")]))
        self.assert_completed(apis, "neutral")
        self.assertEqual(set(apis.labels), {"no-reviewer-assigned"})
        self.assertEqual(apis.model_requests, [])
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
            {"files": [GATE.RISK_POLICY]},
            {"files": [GATE.RISK_POLICY], "records": [record(GATE.CLASSIFIER)]},
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
        self.assertEqual((request["model"], request["reasoning"]), ("gpt-6.1-sol", {"effort": "xhigh"}))
        self.assertEqual(request["prompt_cache_key"], "a16z/jolt#7")
        self.assertEqual((request["max_output_tokens"], request["service_tier"]), (GATE.MAX_OUTPUT_TOKENS, "default"))
        self.assertEqual(request["text"]["format"]["name"], "verifier_classification")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual([tool["name"] for tool in request["tools"]], ["read_file", "search"])
        self.assertIn("# Verifier classifier", request["instructions"])
        self.assertNotIn(diff, request["instructions"])
        self.assertIn(diff, request["input"])
        self.assertEqual([run["name"] for run in apis.created], [GATE.GATE, GATE.CLASSIFIER])
        admission, verdict = apis.posted
        self.assertTrue(admission.startswith(f"<!-- {GATE.CLASSIFIER} -->\n### Verifier classifier: running"))
        self.assertTrue(verdict.startswith(f"<!-- {GATE.CLASSIFIER} -->"))
        self.assertIn("prover-only", verdict)
        [approval] = [node for node in apis.reviews if node["author"]["login"] == "github-actions"]
        self.assertEqual((approval["commit"]["oid"], approval["state"]), (HEAD, "APPROVED"))
        self.assertTrue(approval["body"].startswith(GATE.APPROVAL))

    def test_changed_files_reach_the_model_whole(self) -> None:
        # A one-line edit inside an existing verifier test, shaped like the one in
        # crates/jolt-verifier/src/config.rs: the diff's context reaches neither attribute.
        path = "crates/jolt-verifier/src/config.rs"
        old = "\n".join(["pub struct JoltProtocolConfig;", "", "#[cfg(test)]", "mod tests {", "    #[test]",
                         "    fn protocol_config_wire_format() {",
                         *[f"        let step{index} = {index};" for index in range(8)],
                         "        assert_eq!(bytes, [0, 1, 0, 4, 0]);", "    }", "}", ""])
        new = old.replace("[0, 1, 0, 4, 0]", "[0, 1, 0, 4, 1]")
        diff = unified_diff(path, old, new)
        self.assertNotIn("#[test]", diff)
        apis = evaluate(FakeApis(files=[path], diff=diff, trees={BASE: {path: old}, HEAD: {path: new}},
                                 comments=[comment("core-a", attestation())]))
        [request] = apis.model_requests
        self.assertIn(f"## {path} at the head\n\n1\tpub struct JoltProtocolConfig;\n2\t\n3\t#[cfg(test)]\n"
                      "4\tmod tests {\n5\t    #[test]\n", request["input"])
        self.assertIn(f"Head, revision `head` for the tools: {HEAD}", request["input"])
        self.assertIn(("GET", f"https://api.github.com/repos/{REPO}/tarball/{HEAD}"), apis.requests)

    def test_tool_reads_are_pinned_to_the_merge_base_and_head_and_bounded(self) -> None:
        manifest = '[package]\nname = "jolt-verifier"\n'
        unchanged = {"crates/jolt-verifier/Cargo.toml": manifest, "crates/big.rs": "x" * 500}
        trees = {BASE: {**TREES[BASE], **unchanged}, HEAD: {**TREES[HEAD], **unchanged}}
        calls = [("read_file", {"revision": "base", "path": "crates/jolt-verifier/Cargo.toml"}),
                 ("search", {"revision": "head", "text": "fn helper", "path_prefix": "crates/"}),
                 ("read_file", {"revision": "head", "path": "crates/big.rs"})]
        attested = [comment("core-a", attestation())]
        apis = FakeApis(trees=trees, tool_calls=calls, comments=attested)
        with mock.patch.object(GATE, "MAX_ANSWER_CHARS", 300), mock.patch.object(GATE, "MAX_TOOL_ROUNDS", 1):
            evaluate(apis)
        self.assertEqual(apis.gate(), ("success", "Review attested by core-a"))
        fetched = [urlsplit(url).path.rsplit("/", 2)[1:] for _, url in apis.requests
                   if "/tarball/" in url or "/git/trees/" in url]
        self.assertEqual(sorted(fetched), sorted([["tarball", HEAD], ["trees", HEAD], ["tarball", BASE], ["trees", BASE]]))
        first, second = apis.model_requests
        self.assertEqual((second["previous_response_id"], second["tool_choice"]), ("resp_1", "none"))
        fields = ("instructions", "tools", "prompt_cache_key")
        self.assertEqual([second[field] for field in fields], [first[field] for field in fields])
        answers = {item["call_id"]: item["output"] for item in second["input"]}
        self.assertEqual(answers["call_0"], '1\t[package]\n2\tname = "jolt-verifier"\n')
        self.assertEqual(answers["call_1"], "crates/jolt-prover/src/lib.rs:1: fn helper() {}")
        self.assertTrue(answers["call_2"].startswith("error: the answer holds 503 characters"), answers["call_2"])
        # An archive that differs from the commit's tree, as export-ignore makes it, is refused.
        dropped = FakeApis(trees=trees, tool_calls=calls[:1], comments=attested)
        dropped.exported[BASE] = TREES[BASE]
        with self.assertRaisesRegex(GATE.GateError, "lacks 2 files"):
            evaluate(dropped)
        self.assertEqual(dropped.gate(), ("failure", "Gate error: human approval required"))

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
                apis = evaluate(FakeApis(verdict="AFFECTS", records=[forged], comments=[comment("core-a", attestation())]))
                self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
                self.assertEqual((len(apis.model_requests), apis.review_writes), (1, []))

    def test_head_moving_during_evaluation_publishes_no_pass(self) -> None:
        apis = evaluate(FakeApis(heads=(HEAD, NEW_HEAD), comments=[comment("core-a", attestation())]))
        self.assertEqual(len(apis.model_requests), 1)
        self.assertEqual(apis.gate(), ("neutral", "Superseded while the gate ran"))
        self.assertEqual(apis.review_writes, [])
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"], "the paid verdict stays cached for its head")

    def test_uncertain_or_inconsistent_classification_requires_human_review(self) -> None:
        attested = [comment("core-a", attestation())]
        for verdict in ("AFFECTS", "UNCERTAIN"):
            with self.subTest(verdict=verdict):
                apis = evaluate(FakeApis(verdict=verdict, comments=attested))
                self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
                self.assertEqual(set(apis.labels), {"risk-4", "no-reviewer-assigned"})
        for case, overrides in [("unknown verdict", {"answer_text": '{"verdict": "LIKELY_SAFE", "reasoning": "x"}'}),
                                ("not JSON", {"answer_text": "SAFE"}),
                                ("other model", {"served_model": "gpt-6-luna"})]:
            with self.subTest(case=case):
                apis = FakeApis(comments=attested, **overrides)
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", "Gate error: human approval required"))
                self.assertEqual(apis.record(GATE.CLASSIFIER), [], "a failed classification is not cached")

    def test_human_review_path_follows_latest_review_state(self) -> None:
        cases = [
            ("writer approved head", [review("maintainer", "APPROVED")], ("success", "Approved by maintainer")),
            ("approval of an earlier commit, not dismissed", [review("maintainer", "APPROVED", sha=OLD)],
             ("success", "Approved by maintainer")),
            ("approval then comment", [review("maintainer", "APPROVED"), review("maintainer", "COMMENTED")],
             ("success", "Approved by maintainer")),
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

    def test_incomplete_or_oversized_input_fails_closed_before_the_model(self) -> None:
        two_files = ("crates/jolt-prover/src/lib.rs", "crates/jolt-prover/src/stages.rs")
        truncated = FakeApis()
        truncated.truncated = True
        for case, apis, limit in [
            ("diff misses a file", FakeApis(counts=(2, 1, 1), files=two_files), GATE.MAX_CALL_INPUT_TOKENS),
            ("tree listing truncated", truncated, GATE.MAX_CALL_INPUT_TOKENS),
            ("request over the input bound", FakeApis(), 100),
        ]:
            with self.subTest(case=case), mock.patch.object(GATE, "MAX_CALL_INPUT_TOKENS", limit):
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", "Gate error: human approval required"))
                self.assertEqual(apis.model_requests, [])
                self.assertEqual(apis.posted, [], "no paid attempt is counted")
                self.assertEqual(set(apis.labels), {"no-reviewer-assigned"})

    def test_classifier_attempts_are_capped_per_head_and_per_pull_request(self) -> None:
        attested = comment("core-a", attestation())
        # A malformed answer fails the attempt without caching a result, as a timeout would.
        for case, comments, paid in [("two per head", [attested], 2),
                                     ("three per pull request, across heads", [ledger({OLD: 2}), attested], 1)]:
            with self.subTest(case=case):
                apis = FakeApis(comments=comments, answer_text="SAFE")
                counted = []
                apis.during_model = lambda apis, counted=counted: counted.append(len(apis.posted))
                for _ in range(paid):
                    with self.assertRaises(GATE.GateError):
                        evaluate(apis)
                evaluate(apis)
                self.assertEqual(len(apis.model_requests), paid)
                self.assertEqual(counted, list(range(1, paid + 1)), "each attempt is recorded before it is paid")
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["skipped"])
                self.assertEqual(apis.gate(), ("neutral", "Risk 4: current-head maintainer approval required"))
        cached = evaluate(FakeApis(comments=[ledger({HEAD: 2, OLD: 1}), attested], records=[record(GATE.CLASSIFIER)]))
        self.assertEqual((cached.model_requests, cached.gate()), ([], ("success", "Review attested by core-a")))
        corrupt = ledger({HEAD: 1})
        corrupt["body"] = corrupt["body"].replace(": 1}", ": -1}")
        with self.assertRaisesRegex(GATE.GateError, "malformed attempt ledger"):
            evaluate(FakeApis(comments=[corrupt, attested]))

    def test_spent_classifier_attempts_never_earn_the_gates_approval(self) -> None:
        # crates/jolt-field/build.rs runs this script; its directory cannot establish verifier safety.
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

    def test_a_continuation_over_the_input_bound_withholds_the_tool_answers(self) -> None:
        unchanged = {"crates/big.rs": "x" * 20_000}
        apis = FakeApis(trees={BASE: {**TREES[BASE], **unchanged}, HEAD: {**TREES[HEAD], **unchanged}},
                        tool_calls=[("read_file", {"revision": "head", "path": "crates/big.rs"})],
                        comments=[comment("core-a", attestation())])
        # The chain the next request continues leaves room for the request, not for the answer.
        apis.usage = lambda size: (GATE.MAX_CALL_INPUT_TOKENS - size - 2_000, 0)
        evaluate(apis)
        _, second = apis.model_requests
        self.assertEqual(([item["output"] for item in second["input"]], second["tool_choice"]),
                         ([GATE.BUDGET_SPENT], "none"))
        self.assertEqual(apis.gate(), ("success", "Review attested by core-a"))

    def test_a_model_call_past_its_deadline_is_cancelled_and_fails_closed(self) -> None:
        now, calls = [0.0], []

        def openai(method, url, headers, body):
            calls.append((method, urlsplit(url).path))
            return 200, json.dumps({"id": "resp_1", "status": "in_progress"}).encode()

        def sleep(seconds):
            now[0] += seconds

        model = GATE.Model(openai, "sk-test", sleep=sleep, clock=lambda: now[0])
        with self.assertRaisesRegex(GATE.GateError, "no answer before the gate's deadline"):
            model.run({"input": "x"}, GATE.MODEL_DEADLINE_SECONDS)
        self.assertEqual(calls[-1], ("POST", "/v1/responses/resp_1/cancel"))
        self.assertLessEqual(now[0], GATE.MODEL_CALL_SECONDS + GATE.POLL_SECONDS)

    def test_each_model_turn_logs_the_token_counts_openai_reports(self) -> None:
        reported = {"input_tokens": 30_000, "output_tokens": 900,
                    "input_tokens_details": {"cached_tokens": 24_000, "cache_write_tokens": 5_000}}
        for usage, logged in [
            (reported, ["input 30000", "cache reads 24000", "cache writes 5000", "output 900"]),
            # A count the response leaves out is named, never logged as zero.
            ({"input_tokens": 30_000, "output_tokens": 900}, ["cache reads not reported", "cache writes not reported"]),
        ]:
            with self.subTest(usage=usage):
                def openai(method, url, headers, body, usage=usage):
                    return 200, json.dumps({"id": "resp_1", "status": "completed", "model": GATE.MODEL,
                                            "reasoning": {"effort": GATE.EFFORT}, "usage": usage}).encode()

                with contextlib.redirect_stdout(io.StringIO()) as out:
                    GATE.Model(openai, "sk-test", clock=lambda: 0.0).run({"input": "x"}, GATE.MODEL_DEADLINE_SECONDS)
                [line] = out.getvalue().splitlines()
                self.assertTrue(line.startswith(f"ai-merge-gate: {GATE.MODEL} response resp_1 completed"), line)
                for count in logged:
                    self.assertIn(count, line)

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
             ("neutral", "Human approval required: the change edits the gate's inputs")),
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
                    self.assertIn("## crates/jolt-prover/src/lib.rs at the head", material)
                    self.assertNotIn("CONTRIBUTING.md", material)
                trees = {urlsplit(url).path.rsplit("/", 1)[1] for _, url in apis.requests if "/git/trees/" in url}
                self.assertEqual(trees, {BASE, HEAD})

    def test_gate_inputs_are_risk_four_with_or_without_a_cached_classification(self) -> None:
        for path in (GATE.WORKFLOW, ".github/workflows/ai-merge-gate-events.yml",
                     ".claude/skills/verifier-classifier/SKILL.md",
                     "specs/verifier-closure-lints.md", GATE.RISK_POLICY):
            for cached in (False, True):
                apis = FakeApis(files=[path], risk=1,
                                records=[record(GATE.CLASSIFIER)] if cached else [])
                github = GATE.GitHub(apis, "token", REPO)
                gate = GATE.Gate(github, GATE.Model(apis, "sk-test"), "main", MAINTAINERS, RUN_ID)
                pull = github.get(f"/repos/{REPO}/pulls/7")
                with self.subTest(path=path, cached=cached):
                    self.assertEqual(gate.risk_level(pull), 4)

    def test_changes_to_gate_inputs_never_get_the_gates_approval(self) -> None:
        attested = [comment("core-a", attestation())]
        renamed = {"filename": "docs/classifier.md", "previous_filename": ".claude/skills/verifier-classifier/SKILL.md",
                   "status": "renamed"}
        for case, files in [("classifier skill renamed away", [renamed]),
                            ("closure spec removed", [{"filename": "specs/verifier-closure-lints.md",
                                                       "status": "removed"}]),
                            ("closure spec", ["specs/verifier-closure-lints.md"]),
                            ("risk policy", [GATE.RISK_POLICY]),
                            ("classifier workflow", [GATE.WORKFLOW]),
                            ("events workflow", [".github/workflows/ai-merge-gate-events.yml"]),
                            ("classifier skill", [".claude/skills/verifier-classifier/SKILL.md"])]:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(comments=attested, files=files, counts=(len(files), 1, 1)))
                self.assertEqual(apis.gate(), ("neutral", "Human approval required: the change edits the gate's inputs"))
                self.assertEqual((apis.model_requests, apis.review_writes), ([], []))
        for approver, expected in [
            ("maintainer", ("neutral", "Risk 4: current-head maintainer approval required")),
            ("core-b", ("success", "Risk 4 approved by core-b")),
        ]:
            with self.subTest(approver=approver):
                approved = evaluate(FakeApis(comments=attested, files=[".claude/skills/verifier-classifier/SKILL.md"],
                                             reviews=[review(approver, "APPROVED")]))
                self.assertEqual((approved.gate(), approved.review_writes), (expected, []))
        path = ".github/ISSUE_TEMPLATE/bug_report.md"
        old, new = "Describe the bug\n", "Describe the bug and how to reproduce it\n"
        template = evaluate(FakeApis(comments=attested, files=[path], diff=unified_diff(path, old, new),
                                     trees={BASE: {path: old}, HEAD: {path: new}}))
        self.assertEqual(template.gate(), ("success", "Review attested by core-a"))
        self.assertEqual((len(template.model_requests), template.approved()), (1, [HEAD]))

    def test_unprotected_gate_and_review_files_use_the_classifier(self) -> None:
        for path in ("scripts/ci/ai_merge_gate.py", "scripts/ci/json.py", ".github/workflows/other.yml",
                     ".github/workflows/diff-classifier.yml",
                     ".claude/skills/ci-code-review/SKILL.md", ".claude/skills/comment-policy/SKILL.md",
                     ".claude/skills/test-policy/SKILL.md", "CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md",
                     ".github/ai-merge-gate.md"):
            with self.subTest(path=path):
                apis = evaluate(changed_contents({path: "old\n"}, {path: "new\n"}, risk=1))
                self.assertEqual(apis.gate(), ("success", "Review attested by core-a"))
                self.assertEqual(set(apis.labels), {"risk-1", "merge-eligible"})
                self.assertEqual((len(apis.model_requests), apis.approved()), (1, [HEAD]))

    def test_trusted_input_changes_invalidate_verdict_without_resetting_attempts(self) -> None:
        paths = ("scripts/ci/ai_merge_gate.py", "scripts/ci/diff_classifier.py",
                 ".github/workflows/ai-merge-gate.yml", ".claude/skills/verifier-classifier/SKILL.md",
                 "specs/verifier-closure-lints.md", GATE.RISK_POLICY)
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
                    self.assertEqual([request["text"]["format"]["name"] for request in apis.model_requests],
                                     [] if cached_classifier else ["verifier_classification"])
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


if __name__ == "__main__":
    unittest.main()
