import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# The helper imports its sibling diff_classifier the way it runs in CI, from scripts/ci.
sys.path.insert(0, str(Path(__file__).parents[1] / "ci"))
import diff_classifier_comment as COMMENT

REPO = "a16z/jolt"
HEAD = "a" * 40
BRANCH = "x&state=all#$(id)"
FORK = {"id": 2, "owner": {"login": "mallory"}}
RUN = {"id": 9, "head_repository": FORK, "head_branch": BRANCH, "head_sha": HEAD}
# Git accepts newlines in file names, and the runner reads a log line starting `::` as a command.
HOSTILE_PATH = "src/pwned\n::error::pwned\n|`$(id)`|.rs"


def pull(number, repo_id=2, ref=BRANCH, sha=HEAD, base="main"):
    return {"number": number, "head": {"repo": {"id": repo_id}, "ref": ref, "sha": sha}, "base": {"ref": base}}


class PullsAtHead(unittest.TestCase):
    def test_only_pull_requests_at_the_exact_recorded_head_match(self) -> None:
        # GitHub answers a `head` filter it cannot parse with every open pull request.
        listed = [pull(7), pull(8, sha="b" * 40), pull(9, repo_id=3), pull(10, ref="other"),
                  {**pull(11), "head": {"repo": None, "ref": BRANCH, "sha": HEAD}}]
        self.assertEqual([found["number"] for found in COMMENT.pulls_at_head(RUN, REPO, lambda path: listed)], [7])
        self.assertEqual(COMMENT.pulls_at_head({**RUN, "head_repository": None}, REPO, self.fail), [])

    def test_the_branch_name_cannot_add_request_parameters(self) -> None:
        requests = []
        COMMENT.pulls_at_head(RUN, REPO, lambda path: requests.append(path) or [])
        self.assertEqual(requests, ["repos/a16z/jolt/pulls?state=open&per_page=100"
                                    "&head=mallory%3Ax%26state%3Dall%23%24%28id%29"])


class Main(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        origin = self.root / "origin"
        self.git_env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"}
        self.git("init", "--quiet", "--initial-branch=main", str(origin), cwd=self.root)
        (origin / "src").mkdir()
        (origin / "src/lib.rs").write_text("fn kept() {}\n")
        self.git("add", ".", cwd=origin)
        self.git("commit", "--quiet", "-m", "base", cwd=origin)
        # Like a fork's head in the base repository: reachable from refs/pull only.
        self.git("checkout", "--quiet", "--detach", cwd=origin)
        (origin / HOSTILE_PATH).write_text("fn a() {}\nfn b() {}\nfn c() {}\n")
        self.git("add", ".", cwd=origin)
        self.git("commit", "--quiet", "-m", "head", cwd=origin)
        self.head = self.git("rev-parse", "HEAD", cwd=origin)
        self.git("update-ref", "refs/pull/7/head", self.head, cwd=origin)
        self.git("clone", "--quiet", str(origin), str(self.root / "clone"), cwd=self.root)

    def git(self, *args: str, cwd: Path) -> str:
        return subprocess.run(["git", *args], cwd=cwd, env=self.git_env, check=True, capture_output=True,
                              text=True).stdout.strip()

    def test_renders_counts_only_and_skips_a_pull_request_it_cannot_classify(self) -> None:
        event, output = self.root / "event.json", self.root / "output"
        event.write_text(json.dumps({"workflow_run": {**RUN, "head_sha": self.head}}))
        env = {**self.git_env, "GITHUB_EVENT_PATH": str(event), "GITHUB_REPOSITORY": REPO,
               "GITHUB_OUTPUT": str(output)}
        listed = [pull(7, sha=self.head), pull(8, sha=self.head, base="deleted-base")]
        with mock.patch.dict(os.environ, env), mock.patch.object(COMMENT, "gh_api", return_value=listed), \
                contextlib.chdir(self.root / "clone"), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(COMMENT.main(), 0)
        name, _, value = output.read_text().partition("=")
        rendered = json.loads(value)
        self.assertEqual((name, [(entry["number"], entry["head"]) for entry in rendered]), ("pulls", [(7, self.head)]))
        self.assertIn("Actual code changed: **3 lines**", rendered[0]["body"])
        log = stdout.getvalue()
        self.assertNotIn("pwned", rendered[0]["body"] + log)
        self.assertTrue(all(line.startswith("diff-classifier-comment: ") for line in log.splitlines()), log)


if __name__ == "__main__":
    unittest.main()
