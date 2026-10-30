import json
from pathlib import Path
import sys
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
SLUG = "jolt-ai-merge-gate"
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


def comment(login, body, edited=False, kind="User"):
    return {"author": {"__typename": kind, "login": login}, "body": body,
            "lastEditedAt": "2026-10-30T12:00:00Z" if edited else None, "fullDatabaseId": "11"}


def review(login, state, sha=HEAD, body="", kind="User"):
    return {"author": {"__typename": kind, "login": login}, "body": body, "lastEditedAt": None,
            "state": state, "commit": {"oid": sha}}


class FakeApis:
    """GitHub and OpenAI as the gate reaches them: canned state in, every request recorded."""

    def __init__(self, author="core-a", comments=(), reviews=(), labels=(), draft=False,
                 records=(), heads=(HEAD,), diff=PLAIN_DIFF, files=("crates/jolt-prover/src/lib.rs",),
                 counts=(1, 1, 1), verdict="SAFE", review_answer=None, answer_text=None,
                 served_model="gpt-6-astra-2026-09-30") -> None:
        self.files = [entry if isinstance(entry, dict) else {"filename": entry} for entry in files]
        self.author, self.comments, self.reviews = author, list(comments), list(reviews)
        self.labels, self.draft, self.heads, self.diff, self.counts = labels, draft, list(heads), diff, counts
        self.verdict, self.review_answer, self.answer_text = verdict, review_answer, answer_text
        self.served_model = served_model
        self.core, self.writers = {"core-a", "core-b"}, {"core-a", "core-b", "maintainer", "ci-bot"}
        self.check_runs = [{"name": name, "head_sha": sha, "conclusion": conclusion,
                            "status": "completed", "completed_at": "2026-10-30T00:00:00Z",
                            "app": {"slug": app}} for name, sha, conclusion, app in records]
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
        if method == "GET" and path.startswith(f"/repos/{REPO}/compare/main..."):
            assert headers["Accept"] == "application/vnd.github.diff"
            return 200, self.diff.encode()
        if method == "GET" and path.endswith("/check-runs"):
            sha = path.split("/")[-2]
            runs = [run for run in self.check_runs
                    if run["name"] == query["check_name"][0] and run["head_sha"] == sha]
            return self.ok({"total_count": len(runs), "check_runs": runs})
        if (method, path) == ("POST", f"/repos/{REPO}/check-runs"):
            run = dict(data, completed_at="2026-10-30T01:00:00Z", app={"slug": SLUG})
            self.created.append(run)
            self.check_runs.append(run)
            return self.ok(run)
        if (method, path) in (("POST", f"/repos/{REPO}/issues/7/comments"),
                              ("PATCH", f"/repos/{REPO}/issues/comments/11")):
            self.posted.append(data["body"])
            return self.ok({})
        if (method, path) == ("GET", f"/repos/{REPO}/pulls"):
            pulls = [{"number": 9, "base": {"ref": "release"}}, {"number": 7, "base": {"ref": "main"}}]
            return self.ok(pulls if query["head"] == ["forker:fix"] else [])
        raise AssertionError(f"unexpected request {method} {url}")

    def openai(self, method, path, data):
        assert (method, path) == ("POST", "/v1/responses"), (method, path)
        self.model_requests.append(data)
        if data["text"]["format"]["name"] == "code_review":
            text = json.dumps(self.review_answer)
        else:
            text = self.answer_text or json.dumps({"verdict": self.verdict, "reasoning": "prover-only"})
        return self.ok({"id": "resp_1", "status": "completed", "model": self.served_model,
                        "reasoning": {"effort": "max"},
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]})

    @staticmethod
    def ok(value):
        return 200, json.dumps(value).encode()

    def gate(self):
        runs = [run for run in self.created if run["name"] == GATE.GATE]
        return runs[-1]["conclusion"], runs[-1]["output"]["title"]

    def record(self, name):
        return [run["conclusion"] for run in self.created if run["name"] == name]


def evaluate(apis):
    def no_polling(_seconds):
        raise AssertionError("a completed response needs no polling")

    model = GATE.Model(apis, "sk-test", sleep=no_polling)
    GATE.Gate(GATE.GitHub(apis, "token", REPO), model, "main", "core", SLUG).evaluate(7)
    return apis


class AiMergeGateTests(unittest.TestCase):
    def test_no_agent_work_without_core_authorization_or_while_draft(self) -> None:
        outsider = FakeApis(author="outsider", comments=[
            comment("drive-by", f"AI gate: AUTHORIZED {HEAD}"),
            comment("core-a", f"> AI gate: AUTHORIZED {HEAD}"),
            comment("core-a", f"AI gate: AUTHORIZED {OLD}"),
            comment("core-b", f"Fine by me: AI gate: AUTHORIZED {HEAD}"),
            comment("outsider", f"Review: CONFIRMED {HEAD}"),
        ])
        draft = FakeApis(author="core-a", draft=True, comments=[comment("core-a", f"Review: CONFIRMED {HEAD}")])
        for apis, title in [(outsider, "Human approval required"),
                            (draft, "Draft: agent checks start when the PR is ready for review")]:
            with self.subTest(title=title):
                evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", title))
                self.assertEqual(apis.model_requests, [])
                self.assertFalse(any("api.openai.com" in url or "/compare/" in url for _, url in apis.requests))

    def test_core_sign_off_admits_author_attestation_and_diff_stays_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "executed"
            diff = DIFF.format(sentinel=sentinel)
            apis = evaluate(FakeApis(author="outsider", diff=diff, comments=[
                comment("core-a", f"Read it.\r\nAI gate: AUTHORIZED {HEAD}\r\n"),
                comment("outsider", f"Review: CONFIRMED {HEAD}"),
            ]))
            self.assertFalse(sentinel.exists())
        self.assertEqual(apis.gate(), ("success", "Verifier-safe; Astra Max review attested by outsider"))
        [request] = apis.model_requests
        self.assertEqual((request["model"], request["reasoning"]), ("gpt-6-astra", {"effort": "max"}))
        self.assertEqual(request["text"]["format"]["name"], "verifier_classification")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertIn("# Verifier classifier", request["instructions"])
        self.assertNotIn(diff, request["instructions"])
        self.assertIn(diff, request["input"])
        self.assertEqual([run["name"] for run in apis.created], [GATE.CLASSIFIER, GATE.GATE])
        [posted] = apis.posted
        self.assertTrue(posted.startswith(f"<!-- {GATE.CLASSIFIER} -->"))
        self.assertIn("prover-only", posted)

    def test_push_voids_evidence_for_earlier_heads(self) -> None:
        apis = evaluate(FakeApis(
            reviews=[review("maintainer", "APPROVED", sha=OLD)],
            comments=[comment("core-a", f"Review: CONFIRMED {OLD}")],
            records=[(GATE.CLASSIFIER, OLD, "success", SLUG)],
        ))
        self.assertEqual(apis.gate(), ("failure", "Needs an Astra Max review of this head"))
        self.assertEqual(len(apis.model_requests), 1, "the new head needs its own classification")

    def test_only_the_gate_apps_check_runs_are_evidence(self) -> None:
        spoofed = [(GATE.CLASSIFIER, HEAD, "success", "github-actions"),
                   (GATE.REVIEW, HEAD, "success", "github-actions")]
        apis = evaluate(FakeApis(verdict="AFFECTS", records=spoofed))
        self.assertEqual(apis.gate(), ("failure", "Human approval required: the change may affect the verifier"))
        self.assertEqual(len(apis.model_requests), 1)

    def test_head_moving_during_evaluation_publishes_nothing(self) -> None:
        apis = evaluate(FakeApis(heads=(HEAD, NEW_HEAD), comments=[comment("core-a", f"Review: CONFIRMED {HEAD}")]))
        self.assertEqual(len(apis.model_requests), 1)
        self.assertEqual((apis.created, apis.posted), ([], []))

    def test_only_a_safe_classification_skips_human_review(self) -> None:
        attested = [comment("core-a", f"Review: CONFIRMED {HEAD}")]
        for verdict in ("AFFECTS", "UNCERTAIN"):
            with self.subTest(verdict=verdict):
                apis = evaluate(FakeApis(verdict=verdict, comments=attested))
                self.assertEqual(apis.gate(), ("failure", "Human approval required: the change may affect the verifier"))
                self.assertEqual(apis.record(GATE.CLASSIFIER), ["neutral"])
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
                apis = evaluate(FakeApis(author="outsider", reviews=reviews))
                self.assertEqual(apis.gate(), expected)
                self.assertEqual(apis.model_requests, [])

    def test_edited_or_dismissed_attestations_do_not_count(self) -> None:
        line = f"Review: CONFIRMED {HEAD}"
        safe = [(GATE.CLASSIFIER, HEAD, "success", SLUG)]
        stale = evaluate(FakeApis(records=safe, comments=[comment("core-a", line, edited=True)],
                                  reviews=[review("core-b", "DISMISSED", body=line)]))
        self.assertEqual(stale.gate(), ("failure", "Needs an Astra Max review of this head"))
        current = evaluate(FakeApis(records=safe, reviews=[review("core-b", "COMMENTED", body=line)]))
        self.assertEqual(current.gate(), ("success", "Verifier-safe; Astra Max review attested by core-b"))

    def test_incomplete_or_oversized_diff_fails_closed_before_the_model(self) -> None:
        two_files = ("crates/jolt-prover/src/lib.rs", "crates/jolt-prover/src/stages.rs")
        hidden_workflow = PLAIN_DIFF + PLAIN_DIFF.replace("crates/jolt-prover/src/lib.rs", ".github/workflows/ci.yml")
        for case, apis, limit in [
            ("diff misses a file", FakeApis(counts=(2, 1, 1), files=two_files), GATE.MAX_DIFF_CHARS),
            ("file list misses a file", FakeApis(counts=(2, 2, 2), diff=hidden_workflow), GATE.MAX_DIFF_CHARS),
            ("over the size limit", FakeApis(), 100),
        ]:
            with self.subTest(case=case), mock.patch.object(GATE, "MAX_DIFF_CHARS", limit):
                with self.assertRaises(GATE.GateError):
                    evaluate(apis)
                self.assertEqual(apis.gate(), ("failure", "Gate error: human approval required"))
                self.assertEqual(apis.model_requests, [])

    def test_changes_to_gate_inputs_always_need_human_approval(self) -> None:
        attested = [comment("core-a", f"Review: CONFIRMED {HEAD}")]
        renamed = {"filename": "docs/ci.yml", "previous_filename": ".github/workflows/ci.yml"}
        crowd = [f"crates/jolt-prover/src/f{index}.rs" for index in range(100)]
        for case, files in [("workflow renamed away", [renamed]),
                            ("gate script", ["scripts/ci/ai_merge_gate.py"]),
                            ("classifier skill", [".claude/skills/verifier-classifier/SKILL.md"]),
                            ("second page of files", crowd + [".github/dependabot.yml"])]:
            with self.subTest(case=case):
                apis = evaluate(FakeApis(comments=attested, files=files, counts=(len(files), 1, 1)))
                self.assertEqual(apis.gate(), ("failure", "Human approval required: the change edits the gate's inputs"))
                self.assertEqual(apis.model_requests, [])

    def test_ci_review_runs_only_on_request_after_a_safe_classification(self) -> None:
        safe = [(GATE.CLASSIFIER, HEAD, "success", SLUG)]
        unrequested = evaluate(FakeApis(records=safe))
        self.assertEqual(unrequested.gate(), ("failure", "Needs an Astra Max review of this head"))
        self.assertEqual(unrequested.model_requests, [])
        clean = evaluate(FakeApis(records=safe, labels=["ai-review"],
                                  review_answer={"verdict": "CLEAN", "summary": "Nothing to fix.", "findings": []}))
        self.assertEqual(clean.gate(), ("success", "Verifier-safe; CI Astra Max review clean"))
        self.assertEqual([request["text"]["format"]["name"] for request in clean.model_requests], ["code_review"])
        finding = {"path": "crates/jolt-prover/src/lib.rs", "line": 2, "body": "Off by one."}
        issues = evaluate(FakeApis(records=safe, labels=["ai-review"],
                                   review_answer={"verdict": "ISSUES", "summary": "One bug.", "findings": [finding]}))
        self.assertEqual(issues.gate(), ("failure", "Needs an Astra Max review of this head"))
        self.assertEqual(issues.record(GATE.REVIEW), ["neutral"])
        self.assertIn("`crates/jolt-prover/src/lib.rs:2`: Off by one.", issues.posted[0])

    def test_resolve_maps_events_to_open_pull_requests_into_main(self) -> None:
        github = GATE.GitHub(FakeApis(), "token", REPO)
        fork_review = {"workflow_run": {"head_repository": {"owner": {"login": "forker"}}, "head_branch": "fix"}}
        self.assertEqual(GATE.resolve(github, "workflow_run", fork_review, "main"), 7)
        issue = {"issue": {"number": 3}}
        self.assertIsNone(GATE.resolve(github, "issue_comment", issue, "main"))
        pr_comment = {"issue": {"number": 7, "pull_request": {"url": "..."}}}
        self.assertEqual(GATE.resolve(github, "issue_comment", pr_comment, "main"), 7)


if __name__ == "__main__":
    unittest.main()
