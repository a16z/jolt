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
SLUG = "jolt-ai-merge-gate"
PIKA_LOGIN = "0xandoroid-ai-agent"
PIKA_ID = "BOT_kgDOFAtqNQ"
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


def attestation(sha=HEAD, number=7, skill="ci-code-review", model="Astra", effort="Max"):
    return (f"I have reviewed pull request #{number} at commit {sha} using skill {skill} "
            f"and model {model} with {effort} reasoning effort and found no issues.")


def review(login, state, sha=HEAD, body="", kind="User"):
    return {"author": {"__typename": kind, "login": login}, "body": body, "lastEditedAt": None,
            "state": state, "commit": {"oid": sha}}


def git_tree(files):
    """GitHub's recursive tree listing and tarball for a commit holding `files`."""
    entries = [{"path": path, "type": "blob", "mode": "100644", "size": len(text.encode()),
                "sha": hashlib.sha1(b"blob %d\0" % len(text.encode()) + text.encode()).hexdigest()}
               for path, text in files.items()]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for path, text in files.items():
            info = tarfile.TarInfo(f"a16z-jolt-0000000/{path}")
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
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


class FakeApis:
    """GitHub and OpenAI as the gate reaches them: canned state in, every request recorded."""

    def __init__(self, author="core-a", comments=(), reviews=(), labels=(), draft=False,
                 records=(), heads=(HEAD,), diff=PLAIN_DIFF, files=("crates/jolt-prover/src/lib.rs",),
                 counts=(1, 1, 1), verdict="SAFE", risk=2, risks=None, answer_text=None,
                 served_model="gpt-6-astra-2026-09-30", trees=None, tool_calls=()) -> None:
        # The pull request's file list, which follows its branch; the commits' trees do not move.
        self.files = [entry if isinstance(entry, dict) else {"filename": entry} for entry in files]
        self.changed_paths = {entry["filename"] for entry in self.files} | {
            entry["previous_filename"] for entry in self.files if "previous_filename" in entry}
        self.author, self.comments, self.reviews = author, list(comments), list(reviews)
        self.labels, self.draft, self.heads, self.diff, self.counts = list(labels), draft, list(heads), diff, counts
        self.verdict, self.answer_text = verdict, answer_text
        self.risk, self.risks = risk, risks
        self.served_model, self.tool_calls = served_model, tool_calls
        self.trees = dict(trees) if trees is not None else trees_for(self.files)
        self.exported, self.during_model, self.truncated = dict(self.trees), None, False
        self.during_recheck, self.pr_reads = None, 0
        self.core, self.writers = {"core-a", "core-b"}, {"core-a", "core-b", "maintainer", "ci-bot"}
        self.label_definitions = {"risk-1", "risk-2", "risk-3", "risk-4", "needs-human-review"}
        self.label_writes, self.during_label_write = [], None
        self.check_runs = [{"id": index, "name": name, "head_sha": sha, "conclusion": conclusion,
                            "status": "completed", "completed_at": "2026-10-30T00:00:00Z",
                            "output": {"title": "Cached classification", "summary": "cached",
                                       "text": json.dumps({"risk": risk, "policy_sha256": hashlib.sha256(
                                           (GATE.ROOT / GATE.RISK_POLICY).read_bytes()).hexdigest()})},
                            "app": {"slug": app}} for index, (name, sha, conclusion, app) in enumerate(records)]
        self.requests, self.created, self.posted, self.model_requests = [], [], [], []

    def __call__(self, method, url, headers, body):
        self.requests.append((method, url))
        parts = urlsplit(url)
        data = json.loads(body) if body else None
        if parts.netloc == "api.openai.com":
            return self.openai(method, parts.path, data)
        path, query = parts.path, parse_qs(parts.query)
        team = "/orgs/a16z/teams/core"
        if (method, path) == ("GET", f"/repos/{REPO}/pulls/7"):
            self.pr_reads += 1
            if self.pr_reads == 2 and self.during_recheck:
                self.during_recheck(self)
            head = self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]
            return self.ok({"number": 7, "state": "open", "draft": self.draft, "base": {"ref": "main"},
                            "head": {"sha": head}, "user": {"login": self.author},
                            "labels": [{"name": name} for name in self.labels],
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
        if (method, path) == ("GET", team):
            return self.ok({"slug": "core"})
        if method == "GET" and path.startswith(f"{team}/memberships/"):
            member = path.rsplit("/", 1)[1] in self.core
            return self.ok({"state": "active"}) if member else (404, b"{}")
        if method == "GET" and path.startswith(f"/repos/{REPO}/collaborators/"):
            login = path.split("/")[-2]
            return self.ok({"permission": "write" if login in self.writers else "read"})
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
            run = dict(data, id=len(self.check_runs), completed_at="2026-10-30T01:00:00Z", app={"slug": SLUG})
            self.created.append(run)
            self.check_runs.append(run)
            return self.ok(run)
        if method == "PATCH" and path.startswith(f"/repos/{REPO}/check-runs/"):
            run = self.check_runs[int(path.rsplit("/", 1)[1])]
            run.update(data)
            return self.ok(run)
        if (method, path) in (("POST", f"/repos/{REPO}/issues/7/comments"),
                              ("PATCH", f"/repos/{REPO}/issues/comments/11")):
            self.posted.append(data["body"])
            return self.ok({})
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
            self.label_writes.append((method, name))
            self.labels.remove(name)
            return self.ok([{"name": name} for name in self.labels])
        if (method, path) == ("GET", f"/repos/{REPO}/pulls"):
            pulls = [{"number": 9, "base": {"ref": "release"}}, {"number": 7, "base": {"ref": "main"}}]
            return self.ok(pulls if query["head"] == ["forker:fix"] else [])
        raise AssertionError(f"unexpected request {method} {url}")

    def openai(self, method, path, data):
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
        return self.ok({"id": f"resp_{len(self.model_requests)}", "status": "completed", "model": self.served_model,
                        "reasoning": {"effort": "max"}, "output": output})

    @staticmethod
    def ok(value):
        return 200, json.dumps(value).encode()

    def gate(self, name=GATE.GATE):
        runs = [run for run in self.created if run["name"] == name]
        return runs[-1].get("conclusion"), runs[-1]["output"]["title"]

    def record(self, name):
        return [run["conclusion"] for run in self.created if run["name"] == name]


def evaluate(apis, attest_only=False):
    def no_polling(_seconds):
        raise AssertionError("a completed response needs no polling")

    model = GATE.Model(apis, "" if attest_only else "sk-test", sleep=no_polling)
    GATE.Gate(GATE.GitHub(apis, "token", REPO), model, "main", "core", SLUG).evaluate(7, attest_only)
    return apis


class AiMergeGateTests(unittest.TestCase):
    def assert_completed(self, apis, name, conclusion):
        [run] = [run for run in apis.created if run["name"] == name]
        self.assertEqual((run["head_sha"], run["status"], run["conclusion"]), (HEAD, "completed", conclusion))

    def test_risk_four_requires_a_current_head_maintainer_approval(self) -> None:
        for reviews, expected in [
            ([], "failure"),
            ([review("core-b", "APPROVED", sha=OLD)], "failure"),
            ([review("maintainer", "APPROVED")], "failure"),
            ([review("core-a", "APPROVED")], "failure"),
            ([review("core-b", "DISMISSED")], "failure"),
            ([review("core-b", "APPROVED")], "success"),
        ]:
            with self.subTest(reviews=reviews):
                apis = evaluate(FakeApis(files=["crates/jolt-verifier/src/lib.rs"], verdict="AFFECTS", risk=4,
                                         reviews=reviews, comments=[comment("core-a", attestation())]))
                self.assert_completed(apis, GATE.GATE, expected)
                self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review"})

    def test_risk_three_can_pass_with_attestation_without_human_approval(self) -> None:
        apis = evaluate(FakeApis(verdict="AFFECTS", risk=3, comments=[comment("core-a", attestation())]))
        self.assert_completed(apis, GATE.GATE, "success")
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
        self.assertEqual(apis.labels, ["risk-3"])

    def test_current_head_core_approval_preserves_the_model_free_human_path(self) -> None:
        apis = FakeApis(reviews=[review("core-b", "APPROVED")])
        env = {"GH_TOKEN": "token", "GITHUB_REPOSITORY": REPO, "DEFAULT_BRANCH": "main",
               "AI_MERGE_GATE_CORE_TEAM": "core", "AI_MERGE_GATE_APP_SLUG": SLUG}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(GATE, "http", apis):
            self.assertEqual(GATE.main(["ai_merge_gate.py", "evaluate", "7"]), 0)
            self.assertEqual(GATE.main(["ai_merge_gate.py", "attest", "7"]), 0)
        self.assert_completed(apis, GATE.GATE, "success")
        self.assert_completed(apis, GATE.ATTESTATION_CHECK, "failure")
        self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review"})
        self.assertEqual(apis.model_requests, [])

    def test_risk_paths_and_semantic_results_choose_the_highest_level(self) -> None:
        for levels, expected in [
            ({"README.md": 1}, 1),
            ({"crates/jolt-kernels/src/lib.rs": 2}, 2),
            ({"Cargo.lock": 2}, 3),
            ({"crates/jolt-prover/src/stages/stage3.rs": 2}, 4),
            ({"crates/jolt-verifier/src/config.rs": 1}, 1),
            ({"README.md": 1, "crates/jolt-prover/src/stages/stage3.rs": 2}, 4),
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
                wanted = {f"risk-{expected}"} | ({"needs-human-review"} if expected == 4 else set())
                self.assertEqual(set(apis.labels), wanted)
                self.assert_completed(apis, GATE.GATE, "failure" if expected == 4 else "success")
                [classification] = [run for run in apis.created if run["name"] == GATE.CLASSIFIER]
                self.assertEqual(json.loads(classification["output"]["text"])["risk"], expected)

    def test_risk_labels_are_published_without_an_attestation_or_paid_authorization(self) -> None:
        for kwargs, wanted, model_calls in [
            ({}, {"risk-2"}, 1),
            ({"author": "outsider", "files": ["docs/guide.md"]}, {"risk-1"}, 0),
            ({"files": ["scripts/ci/helper.py"]}, {"risk-1"}, 0),
            ({"author": "outsider"}, {"risk-4", "needs-human-review"}, 0),
            ({"draft": True}, {"risk-4", "needs-human-review"}, 0),
        ]:
            with self.subTest(kwargs=kwargs):
                apis = evaluate(FakeApis(**kwargs))
                self.assert_completed(apis, GATE.GATE, "failure")
                self.assertEqual(set(apis.labels), wanted)
                self.assertEqual(len(apis.model_requests), model_calls)

    def test_reclassification_replaces_only_the_risk_family_and_human_label(self) -> None:
        apis = evaluate(FakeApis(risk=3, records=[(GATE.CLASSIFIER, HEAD, "neutral", SLUG)],
                                 labels=["risk-4", "needs-human-review", "keep-healed", "benchmark"],
                                 comments=[comment("core-a", attestation())]))
        self.assert_completed(apis, GATE.GATE, "success")
        self.assertEqual(set(apis.labels), {"risk-3", "keep-healed", "benchmark"})
        self.assertEqual(apis.label_writes, [("POST", ["risk-3"]), ("DELETE", "needs-human-review"), ("DELETE", "risk-4")])
        self.assertEqual(apis.model_requests, [])

    def test_risk_cache_reuse_is_head_bound_and_already_correct_labels_are_not_rewritten(self) -> None:
        for cached_head in (HEAD, OLD):
            with self.subTest(cached_head=cached_head):
                apis = evaluate(FakeApis(records=[(GATE.CLASSIFIER, cached_head, "success", SLUG)],
                                         labels=["risk-2", "keep-healed"],
                                         comments=[comment("core-a", attestation())]))
                self.assert_completed(apis, GATE.GATE, "success")
                self.assertEqual(len(apis.model_requests), 0 if cached_head == HEAD else 1)
                self.assertEqual(apis.label_writes, [])
                self.assertEqual(set(apis.labels), {"risk-2", "keep-healed"})

    def test_invalid_cached_risk_results_cannot_waive_review(self) -> None:
        for case in ("failed check", "old policy", "legacy result"):
            with self.subTest(case=case):
                apis = FakeApis(records=[(GATE.CLASSIFIER, HEAD, "success", SLUG)], verdict="UNCERTAIN",
                                comments=[comment("core-a", attestation())])
                cached = apis.check_runs[0]
                if case == "failed check":
                    cached["conclusion"] = "failure"
                elif case == "old policy":
                    cached["output"]["text"] = json.dumps({"risk": 2, "policy_sha256": "0" * 64})
                else:
                    cached["output"].pop("text")
                evaluate(apis)
                self.assert_completed(apis, GATE.GATE, "failure")
                self.assertEqual(len(apis.model_requests), 1)
                self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review"})

    def test_incomplete_or_invalid_risk_classification_fails_closed(self) -> None:
        path = "crates/jolt-prover/src/lib.rs"
        good = {"path": path, "level": 2, "reasoning": "Prover-only code."}
        for risks in ([], [good, good], [{**good, "path": "omitted.rs"}],
                      [{**good, "level": 0}], [{**good, "reasoning": ""}]):
            with self.subTest(risks=risks):
                apis = FakeApis(risks=risks, labels=["risk-2", "keep-healed"])
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assert_completed(apis, GATE.GATE, "failure")
                self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review", "keep-healed"})
                self.assertEqual(apis.record(GATE.CLASSIFIER), [])

    def test_risk_labels_do_not_publish_for_a_superseded_head(self) -> None:
        apis = FakeApis(labels=["risk-3", "keep-healed"])

        def push(apis):
            apis.heads = [NEW_HEAD]

        apis.during_model = push
        evaluate(apis)
        self.assert_completed(apis, GATE.GATE, "failure")
        self.assertEqual(apis.label_writes, [])
        self.assertEqual(set(apis.labels), {"risk-3", "keep-healed"})

    def test_risk_four_rechecks_review_state_and_maintainer_authority(self) -> None:
        for change in (lambda apis: apis.reviews[0].update(state="DISMISSED"),
                       lambda apis: apis.reviews[0].update(commit={"oid": OLD}),
                       lambda apis: apis.core.discard("core-b")):
            apis = FakeApis(risk=4, records=[(GATE.CLASSIFIER, HEAD, "neutral", SLUG)],
                            reviews=[review("core-b", "APPROVED")], comments=[comment("core-a", attestation())])
            apis.during_recheck = change
            evaluate(apis)
            self.assert_completed(apis, GATE.GATE, "failure")
            self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review"})

    def test_approval_revoked_during_label_publication_cannot_pass(self) -> None:
        apis = FakeApis(risk=4, records=[(GATE.CLASSIFIER, HEAD, "neutral", SLUG)],
                        reviews=[review("core-b", "APPROVED")], comments=[comment("core-a", attestation())])
        apis.during_label_write = lambda apis: apis.reviews[0].update(state="DISMISSED")
        evaluate(apis)
        self.assert_completed(apis, GATE.GATE, "failure")

    def test_pr_risk_policy_is_data_and_missing_admin_labels_fail_closed(self) -> None:
        files = [{"filename": GATE.RISK_POLICY}, {"filename": "crates/jolt-prover/src/stages/stage3.rs"}]
        trees = trees_for(files)
        trees[HEAD][GATE.RISK_POLICY] = '{"risk-1": {"paths": ["**"]}}'
        apis = evaluate(FakeApis(files=files, trees=trees, reviews=[review("maintainer", "APPROVED")]))
        self.assert_completed(apis, GATE.GATE, "failure")
        self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review"})
        self.assertEqual(apis.model_requests, [])
        missing = FakeApis(comments=[comment("core-a", attestation())])
        missing.label_definitions.clear()
        with self.assertRaisesRegex(GATE.GateError, "admin must create"):
            evaluate(missing)
        self.assert_completed(missing, GATE.GATE, "failure")
        self.assertEqual(missing.label_writes, [])

    def test_current_head_pika_comment_is_accepted(self) -> None:
        for attest_only in (False, True):
            with self.subTest(attest_only=attest_only):
                apis = evaluate(FakeApis(comments=[comment(PIKA_LOGIN, attestation(), kind="Bot",
                                                          actor_id=PIKA_ID)]), attest_only)
                self.assert_completed(apis, GATE.ATTESTATION_CHECK if attest_only else GATE.GATE, "success")
                self.assertEqual(len(apis.model_requests), 0 if attest_only else 1)

    def test_attestation_requires_an_exact_line_for_this_pull_and_head(self) -> None:
        valid = attestation()
        cases = [
            (valid, "success"),
            (f"Review complete.\r\n{valid}\r\n", "success"),
            (attestation(skill="Independent security review", model="A different model", effort="very high"), "success"),
            (f"Review: CONFIRMED {HEAD}", "failure"),
            (attestation(number=8), "failure"),
            (attestation(number="07"), "failure"),
            (attestation(OLD), "failure"),
            (attestation(HEAD[:7]), "failure"),
            (attestation(HEAD.upper()), "failure"),
            (attestation(skill=""), "failure"),
            (attestation(model=" "), "failure"),
            (attestation(effort="\t"), "failure"),
            ("> " + valid, "failure"),
            ("Quote: " + valid, "failure"),
            (valid + " Except one issue.", "failure"),
            (" " + valid, "failure"),
            (valid.replace("using skill", "using a skill"), "failure"),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                apis = evaluate(FakeApis(comments=[comment("core-b", body, edited=True)]), attest_only=True)
                self.assert_completed(apis, GATE.ATTESTATION_CHECK, expected)
                self.assertEqual(apis.model_requests, [])

    def test_attestation_author_is_a_current_team_member_or_the_pinned_pika_bot(self) -> None:
        for node, expected in [
            (comment("core-b", attestation()), "success"),
            (comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID), "success"),
            (comment("outsider", attestation()), "failure"),
            (comment("maintainer", attestation()), "failure"),
            (comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id="BOT_other"), "failure"),
            (comment(SLUG, attestation(), kind="Bot", actor_id="BOT_publisher"), "failure"),
            (comment("outsider", attestation(), actor_id=PIKA_ID), "failure"),
            ({**comment("core-a", attestation(), edited=True),
              "editor": {"__typename": "User", "login": "maintainer"}}, "failure"),
            ({**comment(PIKA_LOGIN, attestation(), edited=True, kind="Bot", actor_id=PIKA_ID),
              "editor": {"__typename": "User", "login": "core-b"}}, "success"),
            ({**comment("core-a", attestation(), edited=True), "editor": None}, "failure"),
            ({**comment("deleted", attestation()), "author": None}, "failure"),
        ]:
            with self.subTest(author=node["author"]):
                apis = evaluate(FakeApis(author="outsider", comments=[node]), attest_only=True)
                self.assert_completed(apis, GATE.ATTESTATION_CHECK, expected)
                self.assertEqual(apis.model_requests, [])

    def test_attestation_cli_needs_no_model_key_and_accepts_no_cached_or_human_substitute(self) -> None:
        for records, reviews, comments, expected in [
            ([], [review("maintainer", "APPROVED")], [], "failure"),
            ([("ai-merge-gate/review", HEAD, "success", SLUG)], [], [], "failure"),
            ([(GATE.ATTESTATION_CHECK, HEAD, "success", SLUG)], [], [], "failure"),
            ([], [], [comment("core-a", attestation())], "success"),
        ]:
            with self.subTest(records=records, reviews=reviews, comments=comments):
                apis = FakeApis(records=records, reviews=reviews, comments=comments)
                env = {"GH_TOKEN": "token", "GITHUB_REPOSITORY": REPO, "DEFAULT_BRANCH": "main",
                       "AI_MERGE_GATE_CORE_TEAM": "core", "AI_MERGE_GATE_APP_SLUG": SLUG}
                with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(GATE, "http", apis):
                    self.assertEqual(GATE.main(["ai_merge_gate.py", "attest", "7"]), 0)
                self.assert_completed(apis, GATE.ATTESTATION_CHECK, expected)
                self.assertEqual(apis.model_requests, [])
                self.assertEqual(apis.posted, [])
                self.assertFalse(any("/check-runs?" in url or "/tarball/" in url for _, url in apis.requests))

    def test_attestation_rechecks_edits_deletion_membership_and_new_heads_before_passing(self) -> None:
        def remove_line(apis):
            apis.comments[0].update(body="Review withdrawn.", lastEditedAt="2026-10-31T20:00:00Z")

        def change_head(apis):
            apis.heads = [NEW_HEAD]

        for case, change in [
            ("edit removes the line", remove_line),
            ("edit names a different head", lambda apis: apis.comments[0].update(body=attestation(OLD))),
            ("comment deleted", lambda apis: apis.comments.clear()),
            ("maintainer removed from team", lambda apis: apis.core.discard("core-a")),
            ("new head", change_head),
        ]:
            with self.subTest(case=case):
                apis = FakeApis(comments=[comment("core-a", attestation())],
                                records=[(GATE.ATTESTATION_CHECK, HEAD, "success", SLUG)])
                seen = []

                def during_recheck(apis):
                    seen.append(apis.gate(GATE.ATTESTATION_CHECK))
                    change(apis)

                apis.during_recheck = during_recheck
                evaluate(apis, attest_only=True)
                self.assertEqual(seen, [(None, "Evaluating")])
                self.assert_completed(apis, GATE.ATTESTATION_CHECK, "failure")
                self.assertEqual(apis.model_requests, [])

    def test_comment_events_evaluate_pika_but_never_the_publishing_bots_comments(self) -> None:
        for kind, login, node_id, eligible in [
            ("User", "core-a", "U_core", True),
            ("Bot", PIKA_LOGIN, PIKA_ID, True),
            ("Bot", SLUG, "BOT_publisher", False),
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
                    apis = FakeApis(comments=comments)
                    env = {"GH_TOKEN": "token", "GITHUB_REPOSITORY": REPO, "DEFAULT_BRANCH": "main",
                           "GITHUB_EVENT_PATH": str(event_path), "GITHUB_EVENT_NAME": "issue_comment",
                           "GITHUB_OUTPUT": str(output_path), "AI_MERGE_GATE_CORE_TEAM": "core",
                           "AI_MERGE_GATE_APP_SLUG": SLUG}
                    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(GATE, "http", apis):
                        self.assertEqual(GATE.main(["ai_merge_gate.py", "resolve"]), 0)
                        self.assertEqual(output_path.read_text(), "pr=7\n" if eligible else "pr=\n")
                        if eligible:
                            self.assertEqual(GATE.main(["ai_merge_gate.py", "attest", "7"]), 0)
                    if eligible:
                        self.assert_completed(apis, GATE.ATTESTATION_CHECK, "success" if action == "created" else "failure")
                    else:
                        self.assertEqual(apis.created, [])
                    self.assertEqual(apis.model_requests, [])

    def test_no_agent_work_without_core_authorization_or_while_draft(self) -> None:
        outsider = FakeApis(author="outsider", comments=[
            comment("drive-by", f"AI gate: AUTHORIZED {HEAD}"),
            comment("core-a", f"> AI gate: AUTHORIZED {HEAD}"),
            comment("core-a", f"AI gate: AUTHORIZED {OLD}"),
            comment("core-b", f"Fine by me: AI gate: AUTHORIZED {HEAD}"),
            comment("outsider", attestation()),
        ])
        draft = FakeApis(author="core-a", draft=True, comments=[comment("core-a", attestation())])
        for apis, title in [(outsider, "Human approval required"),
                            (draft, "Draft: agent checks start when the PR is ready for review")]:
            with self.subTest(title=title):
                evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", title))
                self.assertEqual(apis.model_requests, [])
                self.assertFalse([url for _, url in apis.requests
                                  if any(part in url for part in ("api.openai.com", f"/compare/{BASE}...", "/tarball/"))])

    def test_core_sign_off_admits_pika_attestation_and_pr_content_stays_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "executed"
            diff = DIFF.format(sentinel=sentinel)
            changed = f"fn helper() {{}}\n// $(touch {sentinel}) `touch {sentinel}`\n"
            apis = evaluate(FakeApis(author="outsider", diff=diff, trees={**TREES, HEAD: {
                "crates/jolt-prover/src/lib.rs": changed}}, comments=[
                comment("core-a", f"Read it.\r\nAI gate: AUTHORIZED {HEAD}\r\n"),
                comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID),
            ]))
            self.assertFalse(sentinel.exists())
        self.assertEqual(apis.gate(), ("success", "Review attested by " + PIKA_LOGIN))
        [request] = apis.model_requests
        self.assertEqual((request["model"], request["reasoning"]), ("gpt-6-astra", {"effort": "max"}))
        self.assertEqual(request["text"]["format"]["name"], "verifier_classification")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual([tool["name"] for tool in request["tools"]], ["read_file", "search"])
        self.assertIn("# Verifier classifier", request["instructions"])
        self.assertNotIn(diff, request["instructions"])
        self.assertIn(diff, request["input"])
        self.assertEqual([run["name"] for run in apis.created], [GATE.GATE, GATE.CLASSIFIER])
        [posted] = apis.posted
        self.assertTrue(posted.startswith(f"<!-- {GATE.CLASSIFIER} -->"))
        self.assertIn("prover-only", posted)

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
        self.assertEqual((second["instructions"], second["tools"]), (first["instructions"], first["tools"]))
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
        authorized = [comment("core-a", f"AI gate: AUTHORIZED {HEAD}"),
                      comment(PIKA_LOGIN, attestation(), kind="Bot", actor_id=PIKA_ID)]
        attested = [comment("core-a", attestation())]

        def revoke(apis):
            apis.comments = authorized[1:]

        def request_changes(apis):
            apis.reviews.append(review("maintainer", "CHANGES_REQUESTED"))

        def leave_core_team(apis):
            apis.core.discard("core-a")

        def swap_attestation_for_label(apis):
            apis.comments, apis.labels = [], ["ai-review"]

        earlier_pass = [(GATE.GATE, HEAD, "success", SLUG)]
        for case, apis, change, expected in [
            ("core authorization deleted", FakeApis(author="outsider", comments=authorized, records=earlier_pass),
             revoke, ("failure", "Human approval required")),
            ("writer requested changes", FakeApis(comments=attested, records=earlier_pass),
             request_changes, ("failure", "Changes requested by maintainer")),
            ("author left the core team", FakeApis(comments=attested, records=earlier_pass),
             leave_core_team, ("failure", "Human approval required")),
            ("attestation swapped for a CI review request", FakeApis(comments=attested, records=earlier_pass),
             swap_attestation_for_label, ("failure", "Needs a current-head review attestation")),
        ]:
            with self.subTest(case=case):
                seen = []

                def during_model(apis, change=change, seen=seen):
                    change(apis)
                    seen.append(apis.gate())

                apis.during_model = during_model
                evaluate(apis)
                self.assertEqual(seen, [(None, "Evaluating")], "the earlier pass is cleared while the model runs")
                self.assertEqual(apis.gate(), expected)
                self.assertEqual(len(apis.model_requests), 1, "the paid verdict is reused, not asked again")
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"])

    def test_push_voids_ai_evidence_for_earlier_heads(self) -> None:
        apis = evaluate(FakeApis(
            comments=[comment("core-a", attestation(OLD))],
            records=[(GATE.CLASSIFIER, OLD, "success", SLUG)],
        ))
        self.assertEqual(apis.gate(), ("failure", "Needs a current-head review attestation"))
        self.assertEqual(len(apis.model_requests), 1, "the new head needs its own classification")

    def test_only_the_gate_apps_check_runs_are_evidence(self) -> None:
        spoofed = [(GATE.CLASSIFIER, HEAD, "success", "github-actions"),
                   ("ai-merge-gate/review", HEAD, "success", "github-actions")]
        apis = evaluate(FakeApis(verdict="AFFECTS", records=spoofed))
        self.assertEqual(apis.gate(), ("failure", "Risk 4: current-head maintainer approval required"))
        self.assertEqual(len(apis.model_requests), 1)

    def test_head_moving_during_evaluation_publishes_no_pass(self) -> None:
        apis = evaluate(FakeApis(heads=(HEAD, NEW_HEAD), comments=[comment("core-a", attestation())]))
        self.assertEqual(len(apis.model_requests), 1)
        self.assertEqual(apis.gate(), ("failure", "Superseded while the gate ran"))
        self.assertEqual(apis.record(GATE.CLASSIFIER), ["success"], "the paid verdict stays cached for its head")

    def test_uncertain_or_inconsistent_classification_requires_human_review(self) -> None:
        attested = [comment("core-a", attestation())]
        for verdict in ("AFFECTS", "UNCERTAIN"):
            with self.subTest(verdict=verdict):
                apis = evaluate(FakeApis(verdict=verdict, comments=attested))
                self.assertEqual(apis.gate(), ("failure", "Risk 4: current-head maintainer approval required"))
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
                self.assertEqual(set(apis.labels), {"risk-4", "needs-human-review"})
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
             ("failure", "Changes requested by core-b")),
            ("dismissed approval", [review("maintainer", "DISMISSED")], ("failure", "Human approval required")),
            ("read-only approver", [review("fan", "APPROVED")], ("failure", "Human approval required")),
            ("app approval", [review("ci-bot", "APPROVED", kind="Bot")], ("failure", "Human approval required")),
        ]
        for case, reviews, expected in cases:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(author="outsider", reviews=reviews,
                                         records=[(GATE.CLASSIFIER, HEAD, "success", SLUG)]))
                self.assertEqual(apis.gate(), expected)
                self.assertEqual(apis.model_requests, [])

    def test_review_bodies_are_not_attestation_comments(self) -> None:
        for state in ("COMMENTED", "APPROVED", "DISMISSED"):
            with self.subTest(state=state):
                apis = evaluate(FakeApis(reviews=[review("core-b", state, body=attestation())]), attest_only=True)
                self.assert_completed(apis, GATE.ATTESTATION_CHECK, "failure")
                self.assertEqual(apis.model_requests, [])

    def test_incomplete_or_oversized_input_fails_closed_before_the_model(self) -> None:
        two_files = ("crates/jolt-prover/src/lib.rs", "crates/jolt-prover/src/stages.rs")
        truncated = FakeApis()
        truncated.truncated = True
        for case, apis, limit in [
            ("diff misses a file", FakeApis(counts=(2, 1, 1), files=two_files), GATE.MAX_INPUT_CHARS),
            ("tree listing truncated", truncated, GATE.MAX_INPUT_CHARS),
            ("over the size limit", FakeApis(), 100),
        ]:
            with self.subTest(case=case), mock.patch.object(GATE, "MAX_INPUT_CHARS", limit):
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", "Gate error: human approval required"))
                self.assertEqual(apis.model_requests, [])

    def test_event_workflow_changes_fail_even_with_a_writer_approval(self) -> None:
        renamed = {"filename": ".github/workflows/old.yml", "previous_filename": GATE.EVENT_SOURCE, "status": "renamed"}
        for case, entry in [("removed", {"filename": GATE.EVENT_SOURCE, "status": "removed"}),
                            ("edited", GATE.EVENT_SOURCE), ("renamed away", renamed)]:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(author="outsider", draft=True, files=[entry],
                                         reviews=[review("maintainer", "APPROVED")]))
                self.assertEqual(apis.gate(), ("failure", "Changes the gate's event workflow: merge it with the gate off"))
                self.assertEqual(apis.model_requests, [])

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
            ("event workflow deleted on A", BranchMovesToBAndBack(
                [{"filename": GATE.EVENT_SOURCE, "status": "removed"}], b_files, author="outsider",
                reviews=[review("maintainer", "APPROVED", sha=OLD)], counts=(1, 0, 20)),
             ("failure", "Changes the gate's event workflow: merge it with the gate off")),
            ("gate import added on A", BranchMovesToBAndBack(
                [{"filename": "scripts/ci/json.py", "status": "added"}], b_files, comments=attested),
             ("failure", "Human approval required: the change edits the gate's inputs")),
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

    def test_changes_to_gate_inputs_need_human_approval(self) -> None:
        attested = [comment("core-a", attestation())]
        renamed = {"filename": "docs/ci.yml", "previous_filename": ".github/workflows/ci.yml", "status": "renamed"}
        for case, files in [("workflow renamed away", [renamed]),
                            ("gate script", ["scripts/ci/ai_merge_gate.py"]),
                            ("module the gate's imports would find", [{"filename": "scripts/ci/json.py",
                                                                       "status": "added"}]),
                            ("classifier skill", [".claude/skills/verifier-classifier/SKILL.md"])]:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(comments=attested, files=files, counts=(len(files), 1, 1)))
                self.assertEqual(apis.gate(), ("failure", "Human approval required: the change edits the gate's inputs"))
                self.assertEqual(apis.model_requests, [])
        # Other files under .github/ feed nothing the gate runs, so the classifier judges them.
        path = ".github/ISSUE_TEMPLATE/bug_report.md"
        old, new = "Describe the bug\n", "Describe the bug and how to reproduce it\n"
        template = evaluate(FakeApis(comments=attested, files=[path], diff=unified_diff(path, old, new),
                                     trees={BASE: {path: old}, HEAD: {path: new}}))
        self.assertEqual(template.gate(), ("success", "Review attested by core-a"))
        self.assertEqual(len(template.model_requests), 1)

    def test_review_label_and_cached_ci_reviews_cannot_replace_the_comment(self) -> None:
        old_review = [("ai-merge-gate/review", HEAD, "success", SLUG)]
        for cached_classifier in ([], [(GATE.CLASSIFIER, HEAD, "success", SLUG)]):
            for comments in ([], [comment("core-a", attestation())]):
                with self.subTest(cached_classifier=bool(cached_classifier), comments=bool(comments)):
                    apis = evaluate(FakeApis(labels=["ai-review"], records=old_review + cached_classifier,
                                             comments=comments))
                    self.assert_completed(apis, GATE.GATE, "success" if comments else "failure")
                    self.assertEqual([request["text"]["format"]["name"] for request in apis.model_requests],
                                     [] if cached_classifier else ["verifier_classification"])
                    self.assertEqual(apis.record("ai-merge-gate/review"), [])

    def test_resolve_maps_events_to_open_pull_requests_into_main(self) -> None:
        github = GATE.GitHub(FakeApis(), "token", REPO)
        fork_review = {"workflow_run": {"head_repository": {"owner": {"login": "forker"}}, "head_branch": "fix"}}
        self.assertEqual(GATE.resolve(github, "workflow_run", fork_review, "main"), 7)
        issue = {"issue": {"number": 3}, "comment": {"user": {"type": "User"}}}
        self.assertIsNone(GATE.resolve(github, "issue_comment", issue, "main"))
        pr_comment = {"issue": {"number": 7, "pull_request": {"url": "..."}},
                      "comment": {"user": {"type": "User"}}}
        self.assertEqual(GATE.resolve(github, "issue_comment", pr_comment, "main"), 7)


if __name__ == "__main__":
    unittest.main()
