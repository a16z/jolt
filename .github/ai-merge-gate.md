# AI merge gate

Pull requests into `main` can merge without a human review when a core contributor wrote or
authorized them, a CI classifier finds that the change cannot change which proofs the verifier
accepts, and an Astra Max review of the exact head found no issues. Every other pull request
keeps human review. All of these conditions, including the human approval that the rest of the
pull requests still need, are folded into one required check, `ai-merge-gate`, because GitHub
cannot make its native review requirement depend on who wrote a pull request.

Astra Max means OpenAI's `gpt-6-astra` model at reasoning effort `max`. The gate calls it
through the Responses API with no tools.

This file is the policy, the contributor guide and the admin procedure. The gate's code is
`scripts/ci/ai_merge_gate.py`, `.github/workflows/ai-merge-gate.yml` and
`.github/workflows/ai-merge-gate-events.yml`. The classifier's instructions are
`.claude/skills/verifier-classifier/SKILL.md`, and the review's are
`.claude/skills/ci-code-review/SKILL.md`.

## How the gate decides

The gate reads the pull request's current head commit, which this file calls the head, and
evaluates these rows in order. The first row that matches decides. A core member is an active
member of the GitHub team named by the repository variable `AI_MERGE_GATE_CORE_TEAM`. A writer
is an account with write, maintain or admin access to the repository.

| # | Condition | Result | Paid model work |
|---|---|---|---|
| 1 | The pull request is closed, or its base is not `main` | No check | None |
| 2 | A writer has requested changes, on any commit, and has not since approved or had that review dismissed | Fail | None |
| 3 | A writer other than the author approved the head from a personal account, and has not since requested changes or had the approval dismissed | Pass | None |
| 4 | The author is not a core member, and no core member has posted `AI gate: AUTHORIZED <head>` | Fail: human approval required | None, and the diff is not read |
| 5 | The pull request is a draft | Fail | None |
| 6 | The pull request changes a gate input: a file under `.github/`, `scripts/ci/ai_merge_gate.py`, `scripts/ci/diff_classifier.py`, either skill, or `specs/verifier-closure-lints.md` | Fail: human approval required | None |
| 7 | The head has no classifier verdict yet | The classifier runs | Classifier |
| 8 | The classifier verdict for the head is not `SAFE` | Fail: human approval required | None |
| 9 | The author or a core member posted `Review: CONFIRMED <head>` | Pass | None |
| 10 | The pull request has the `ai-review` label and the head has no CI review yet | The CI review runs | Review |
| 11 | The CI review of the head found no issues | Pass | None |
| 12 | Otherwise | Fail: needs an Astra Max review | None |

Every piece of evidence names the head, so a push voids all earlier approvals, sign-offs,
attestations, classifier verdicts and reviews. Unlike the research lab's `check_review.py`,
the gate keeps no exception for a rebase that leaves the patch unchanged.

The two gate lines follow the research lab's convention for `Review: CONFIRMED`. A line counts
only when all of these hold:

- It stands alone on its line and holds the head's full 40-character SHA. Quoted text, such as
  `> Review: CONFIRMED ...`, and the phrase inside a sentence do not count.
- It is in a pull request comment or in the body of a review that is not dismissed.
- A personal account wrote it, and nobody edited the comment or review afterwards. To change a
  line, post a new comment.
- Its author has the right role when the gate runs: a core member for `AI gate: AUTHORIZED`,
  and the pull request's author or a core member for `Review: CONFIRMED`.

## Using the gate

**Core authors.** Mark the pull request ready for review. The classifier then runs once for each
new head and records its verdict as the `ai-merge-gate/classifier` check, with its reasoning in a
pull request comment. Then do one of two things:

- Run the Astra Max review locally on the exact head, as described in
  `.claude/skills/ci-code-review/SKILL.md`. If it finds nothing to fix, post a comment holding
  the line `Review: CONFIRMED <head SHA>`.
- Add the `ai-review` label. The gate then runs the review in CI on each new head while the label
  stays on, and records it as the `ai-merge-gate/review` check and a comment.

**Other authors.** The gate spends nothing on a pull request from someone outside the core team
until a core member posts `AI gate: AUTHORIZED <head SHA>`. Post that line only after reading
the change. Each push needs a new line. After that, the author's own `Review: CONFIRMED` line
counts.

**Verifier and gate changes.** A pull request that the classifier finds `AFFECTS` or
`UNCERTAIN`, or that changes a gate input listed in row 6, needs an approving review from a
writer, as before. The approval must be of the head: approve again after
each push.

**Re-running the gate.** Any pull request comment or review re-runs the gate. You can also re-run
the `AI merge gate` workflow run in the Actions tab. The Re-run button on the gate's own check
runs does nothing, because no server receives that request.

## How the gate is built

The gate never runs code from a pull request. Two workflows divide the work:

- `ai-merge-gate-events.yml` runs on `pull_request` and `pull_request_review` events. GitHub runs
  the pull request's own copy of this file, without secrets and with a token that has no
  permissions. The file does nothing except finish, which tells the gate that something changed.
- `ai-merge-gate.yml` runs on `issue_comment` and on `workflow_run` after the events workflow.
  GitHub always takes both of these workflows from the default branch. Its `resolve` job maps
  the event to a pull request number. Its `evaluate` job checks out the gate script and skills
  from the default branch, mints a token for the gate's GitHub App, reads the pull request, its
  comments and its reviews through the API, and publishes check runs as the App.

The diff reaches the model as a string inside an API request, and the model has no tools. The
gate reads the diff only after rows 2 through 6 have passed. It refuses a diff larger than
400,000 characters, and a diff whose parsed file, addition and deletion counts differ from the
pull request's, rather than truncating it.

The `ai-merge-gate` check run counts only when the gate's App created it. The ruleset below
binds the required check to the App's ID, so a job named `ai-merge-gate` in a pull request's
workflow, which GitHub Actions would report, cannot satisfy it. The App's private key and the
OpenAI key live in an environment that only the default branch can use, so a workflow pushed to
any other branch cannot read them.

Row 6 keeps the AI path from changing the gate itself. Any workflow merged into the default
branch could use the gate's environment, and the listed files decide the gate's verdicts, so
a change to them could weaken every later evaluation. Those changes always need a writer's
approval, whatever the classifier would say.

The gate stores each classifier verdict and CI review as a check run on the head, created by
the App. Only the App can create or change those, which a comment would not guarantee, since
writers can edit anyone's comments. Later runs read these check runs instead of paying for a new
model call. A failed model call is not stored, so the next event retries it.

The gate does not use `pull_request_target`. GitHub's default Actions policy for public
repositories blocks that event from December 2, 2026 unless the repository allows it, and the
gate does not need it.

## Installing the gate

A repository admin installs the gate, with an organization owner for the GitHub App. This
pull request changes no settings, and none of the steps below has been done.

A read with maintain permission on 2026-10-30 showed classic branch protection enabled on
`main` with no required status checks, and `GET /repos/a16z/jolt/rules/branches/main` returned
`[]`, so no active ruleset applied to `main`. The review settings and any organization rulesets
were not readable with that permission. Step 1 reads them with admin permission.

### 1. Record the current protection

```bash
gh api repos/a16z/jolt/branches/main/protection > main-protection.json
gh api 'repos/a16z/jolt/rulesets?includes_parents=true' > rulesets.json
gh api repos/a16z/jolt/rules/branches/main > main-rules.json
```

Write down the review settings, wherever they live: `required_pull_request_reviews` in the
classic protection, or a `pull_request` rule in a ruleset that targets `main`. Record the
approval count and the settings for dismissing stale reviews, code owner review, approval of
the last push and conversation resolution. Also record every required status check. The
gate's setup changes only the approval count and leaves the status checks alone.

Stop and ask the repository owner before continuing if the review settings require more than
one approval, a code owner review or approval of the last push. Row 3 of the table counts one
approval of the head by a writer other than the author, and implements none of those three.

Decide one more thing before step 7. The gate reads no other check, and `main` required no
status check when last read. On the human path a reviewer sees CI before approving; on the AI
path nobody does. To make CI pass before any merge, add the CI job names to the gate ruleset's
`required_status_checks` with `"integration_id": 15368`, the GitHub Actions app. That also
applies to human-reviewed pull requests.

### 2. Create the GitHub App

As an organization owner, create an App under the a16z organization's Developer settings:

- Name it, for example, `jolt-ai-merge-gate`, and clear "Active" under Webhook: the gate
  receives no events.
- Repository permissions: Checks read and write, Pull requests read and write, and the
  mandatory Metadata read-only.
- Organization permissions: Members read-only, which the team membership lookup needs.
- Allow installation only on this account.

After creating it, note the App ID, a number, for step 7, and the Client ID for step 3.
Generate a private key. Install the App on the organization for the `a16z/jolt` repository
only.

### 3. Set the repository variables

Under the repository's Actions variables, set:

- `AI_MERGE_GATE_CORE_TEAM` to the slug of the team whose active members are core contributors,
  the last part of `https://github.com/orgs/a16z/teams/<slug>`. An admin chooses the team; this
  pull request names none. The gate fails closed when the App cannot see the team.
- `AI_MERGE_GATE_CLIENT_ID` to the App's Client ID.

### 4. Create the environment

Create an environment named `ai-merge-gate`:

- Under "Deployment branches and tags", choose "Selected branches and tags" and add the branch
  `main`.
- Add no required reviewers, wait timer or custom deployment protection rule. Reviewers and
  timers would hold every evaluation, and the workflow's `deployment: false` setting rejects
  custom rules.
- Add two environment secrets: `AI_MERGE_GATE_APP_PRIVATE_KEY`, the App's private key in PEM
  form, and `OPENAI_API_KEY`, a key for an OpenAI project that may call `gpt-6-astra` through
  the Responses API.

Do not store either secret as a repository or organization secret, where workflows on any
branch could read it. The gate creates responses with `background: true` and `store: true`, so
OpenAI retains each diff and answer for at least 30 days under the project's data controls.

### 5. Create the review label

```bash
gh label create ai-review --repo a16z/jolt --description "Run the AI merge gate's Astra Max review"
```

### 6. Merge the gate and watch it

Merge this pull request with the protection unchanged. Nothing requires the gate yet, so a
failing gate blocks nothing. On the next few pull requests, confirm that the checks list shows
`ai-merge-gate` from the App, that a pull request whose head a writer approved passes, and that
a ready core-authored pull request gets an `ai-merge-gate/classifier` check and comment.

### 7. Create the two rulesets

Save these two files, replacing `<APP_ID>` with the App ID from step 2. The first requires the
App's `ai-merge-gate` check.

```json
{
  "name": "main: AI merge gate",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "rules": [
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": false,
        "required_status_checks": [{"context": "ai-merge-gate", "integration_id": <APP_ID>}]
      }
    }
  ]
}
```

The second is the off switch. It starts disabled and carries the review settings recorded in
step 1; the values shown are for one approval with stale approvals dismissed.

```json
{
  "name": "main: human review (AI merge gate off switch)",
  "target": "branch",
  "enforcement": "disabled",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "rules": [
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": false
      }
    }
  ]
}
```

Create the off switch first, and note both ruleset IDs from the responses:

```bash
gh api -X POST repos/a16z/jolt/rulesets --input human-review-ruleset.json
gh api -X POST repos/a16z/jolt/rulesets --input ai-merge-gate-ruleset.json
```

From this point a pull request needs both the existing approval and a passing gate. A writer's
approval of the head satisfies both.

### 8. Hand the approval requirement to the gate

Set the classic protection's approval count to zero, keeping "Require a pull request before
merging" and every other setting. Pass the dismiss-stale value recorded in step 1:

```bash
gh api -X PATCH repos/a16z/jolt/branches/main/protection/required_pull_request_reviews \
  -F required_approving_review_count=0 -F dismiss_stale_reviews=<recorded> \
  -F require_code_owner_reviews=false -F require_last_push_approval=false
```

If step 1 found the approval count in a ruleset instead, fetch that ruleset, set its
`pull_request` rule's `required_approving_review_count` to 0, and `PUT` the whole ruleset back.
Then confirm that `gh api repos/a16z/jolt/rules/branches/main` lists `ai-merge-gate` with the
App's `integration_id`, and that the protection now requires zero approvals.

## Turning the gate off

The off switch restores GitHub's native review requirement:

```bash
gh api -X PUT repos/a16z/jolt/rulesets/<human review ruleset ID> -f enforcement=active
```

With both rulesets active, every pull request needs a native approval, and the gate passes on
that same approval. If the gate itself is broken and publishes no check, for example because
its key expired, also disable the gate ruleset, after the command above:

```bash
gh api -X PUT repos/a16z/jolt/rulesets/<gate ruleset ID> -f enforcement=disabled
```

Activate human review before disabling the gate ruleset, so that `main` always requires a
review. To turn the gate back on, activate the gate ruleset, wait until open pull requests show
the check, then disable the human-review ruleset. To remove the gate, turn it off, delete the
gate ruleset, then delete the workflows, script, skill, environment, variables, label and App.

## Limits

- A result can stay stale briefly. Evidence that disappears without a push, such as a dismissed
  approval or an edited comment, takes effect when the next evaluation publishes. Evaluations of
  one pull request run one at a time and the newest runs last, so the final published result
  matches the final state. A head that returns to an earlier SHA shows that SHA's last result
  until its re-evaluation finishes.
- The gate reads team membership on each evaluation. Removing someone from the core team does
  not re-evaluate open pull requests until their next event.
- In CI the classifier and the review see only the diff, with git's hunk headers, and the
  default branch's `specs/verifier-closure-lints.md`. When safety depends on code outside the
  diff, the classifier should answer UNCERTAIN, which sends the pull request to human review.
- A verdict covers the diff from the head's merge base with `main`. A later change to `main` does
  not void it, just as it does not void a human approval when branches need not be up to date.
- An author's `Review: CONFIRMED` line is an attestation. The gate checks who wrote it and for
  which head, not which model ran the review.
- A fork pull request from a first-time contributor may need a maintainer to approve its
  workflow runs before the events workflow runs. Until then, a pull request comment still
  triggers the gate.
