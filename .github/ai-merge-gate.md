# AI merge gate

Once installed, merging into `main` requires both `ai-merge-gate` and
`ai-merge-gate/attestation`, published by the gate's GitHub App. The first enforces the
risk-based human-review policy below. The second requires a current-head review attestation in a PR
comment, even when a human has approved. Neither a human approval nor an earlier CI review
can replace that comment.

Only the verifier-impact classifier calls a model in this gate: OpenAI's `gpt-6-astra` at
reasoning effort `max`, through the Responses API. It reads the pull request's merge base and
head through two read-only tools. Code review happens outside CI; the attestation check has no
model key and makes no model request.

The classifier also assigns one risk label. Risk-1 through risk-3 are eligible for native
auto-merge once the gate, attestation and other required checks pass. Risk-4 also gets
`needs-human-review` and requires an approving review of the **current head** by a maintainer
other than the PR author. This draft enables neither the gate nor native auto-merge.

This file is the policy, the contributor guide and the admin procedure. The gate's code is
`scripts/ci/ai_merge_gate.py`, `.github/workflows/ai-merge-gate.yml` and
`.github/workflows/ai-merge-gate-events.yml`. The classifier's instructions are
`.claude/skills/verifier-classifier/SKILL.md`. The local review guidance is
`.claude/skills/ci-code-review/SKILL.md`; CI does not load it.

## Risk labels determine the human-review threshold

The trusted policy is [`.github/ai-merge-gate-risk.json`](ai-merge-gate-risk.json). It defines
the following levels and path globs; the classifier supplies semantic evidence for ambiguous changes.

| Label | Meaning | Human approval |
|---|---|---|
| `risk-1` | Tests, scripts, CI or documentation only | Not required on the authorized path |
| `risk-2` | Prover or other non-verifier code only | Not required on the authorized path |
| `risk-3` | Small, scoped dependency or shared-code changes that could reach the verifier | Not required on the authorized path |
| `risk-4` | Protocol, sumcheck order or symbolic relations, transcript/Fiat–Shamir, verifier logic, PCS or commitment parameters; also unknown changes | Current-head maintainer approval required |

For production changes, the highest matching path level is a minimum. The classifier can
raise it for effects elsewhere. A per-file risk-1 verdict can identify test-only or comment-only
hunks inside a production file. Every changed path must appear exactly once in the classifier's
result, including removed files and both names of a rename. The PR takes the highest effective
level; an incomplete or invalid result fails the check and receives risk-4 when GitHub is reachable.
`UNCERTAIN` always becomes risk-4. `AFFECTS` with risk-3 remains eligible without a human approval:
verifier impact is reported separately and is no longer a blanket veto.

Labels update even when the PR lacks an attestation or the policy check fails. Core authorization,
draft and gate-input restrictions still decide whether a model may run. Without a cached or new
classification, a change wholly within level-1 paths receives risk-1; other changes receive
risk-4 until they can be classified. This gives unauthorized and draft PRs a conservative label
without running a model.

A qualifying current-head maintainer approval already satisfies the highest risk level, so
the human path starts no model work. It uses an existing classification or the same conservative
fallback. The attestation check still runs independently; that approval cannot replace it.

Reclassification adds the new risk label and removes the other risk labels. It adds
`needs-human-review` for risk-4 and removes it below level 4; unrelated labels are untouched.
The gate reads the API's current head before and after these writes and refuses a stale pass.
It then rechecks review and attestation authority before completing the policy check.

## How the gate decides

The `ai-merge-gate` policy check reads the pull request's current head commit, which this file
calls the head, and evaluates these rows in order. The classifier row continues after recording
its result; the first other matching row decides. A core member is an active
member of the GitHub team named by the repository variable `AI_MERGE_GATE_CORE_TEAM`. A writer
is an account with write, maintain or admin access to the repository.

| # | Condition | Result | Paid model work |
|---|---|---|---|
| 1 | The pull request is closed, or its base is not `main` | No check | None |
| 2 | The pull request adds, changes, renames or removes `.github/workflows/ai-merge-gate-events.yml` | Fail, even when approved: merge it with the gate off | None |
| 3 | A writer has requested changes, on any commit, and has not since approved or had that review dismissed | Fail | None |
| 4 | No writer approved, the author is not a core member, and no core member posted `AI gate: AUTHORIZED <head>` | Fail: human approval required | None |
| 5 | No writer approved and the pull request is a draft | Fail | None |
| 6 | No writer approved and the PR changes a gate input: `.github/workflows/`, `scripts/ci/`, the classifier skill, closure spec or risk policy | Fail: human approval required | None |
| 7 | Core-authorized, ready, no gate-input changes, no qualifying current-head maintainer approval, and no cached classifier result | Run the classifier, then continue | Classifier |
| 8 | Risk-4 with an undismissed current-head approval by a personal-account writer in the core team, other than the author | Pass | None |
| 9 | Risk-4 without that approval | Fail: current-head maintainer approval required | None |
| 10 | Risk-1 through risk-3 and a personal-account writer other than the author approved, without a later dismissal or change request | Pass | None |
| 11 | Risk-1 through risk-3 and an allowed account posted the exact current-head attestation below | Pass | None |
| 12 | Otherwise | Fail: needs a current-head review attestation | None |

The separate `ai-merge-gate/attestation` check evaluates only the comment requirement. It
creates no check for a closed PR or a base other than `main`; otherwise it fails without a
matching authorized comment. Require **both** checks: an approval can pass the policy without
satisfying the attestation check, and an attestation alone cannot satisfy the verifier policy.

Each evaluation creates an in-progress check run and completes that run with its result.
Which same-name run GitHub selects for mergeability remains unverified; see [Limits](#limits).
Before it publishes a pass, the gate reads the pull request, its comments and reviews,
team membership and permissions again and evaluates the table once more,
reusing the model results it already has. It publishes the pass only if that second evaluation
passes too. The attestation evaluation re-reads the head, comments and team membership without
model work. When the policy's new state needs a classifier result the run does not have, for
example after an approval was dismissed, it fails with "The pull request changed while the
gate ran", and the next event runs the gate again.

The AI evidence names the head: a push voids every sign-off, attestation, classifier verdict and
review comment of an earlier head. Unlike the research lab's `check_review.py`, the gate keeps no
exception for a rebase that leaves the patch unchanged. Human approvals follow GitHub's own
review state for risk-1 through risk-3: an approval of an earlier commit counts until GitHub
dismisses it. Risk-4 is stricter: an earlier-head approval never counts, even if GitHub has not
dismissed it.

## The attestation is an exact standalone comment line

The shared Pika/Jolt format is:

```text
This pull request #<number> has been reviewed at commit <full 40-char head SHA> using skill <skill name> and model <model name> with <effort> reasoning effort, and no issues were found.
```

The number and full SHA must match the PR's current API state. Skill, model and effort must
each contain non-whitespace text; names may contain spaces, and there is no skill or model
allowlist. The line must match exactly, with no prefix, indentation, quotation marker or
suffix. It may appear between other lines in the comment. `Review: CONFIRMED`, short SHAs and
attestations only in review bodies do not count.

An allowed author is either a personal account in the configured core team (the Jolt
maintainers chosen by the admin) or the Pika App bot. Being the PR author or having repository
write access alone grants no attestation authority. Membership is checked again before a pass.
The Pika identity is pinned by its GitHub node ID, `BOT_kgDOFAtqNQ`, not by a name in the body.
The REST API identifies it as `0xandoroid-ai-agent[bot]`, numeric ID `336292405`; GraphQL reports
the same node ID with login `0xandoroid-ai-agent` (verified 2026-10-31). This is **not** the
separate App that publishes the gate's checks and classifier comments.
The authority predicate rejects the publishing App as an author or editor, even if the
publisher is misconfigured to use the Pika App.

The check reads the comment's current body, including edits, and uses its API-reported author.
For an edited comment, its last editor must also be an allowed account; write access alone
cannot add an attestation to someone else's comment.
Editing away the line or deleting the comment invalidates it on re-evaluation; it never reads
a cached attestation result. This checks the account's claim, not evidence that the named
skill, model or effort actually ran.

The unrelated `AI gate: AUTHORIZED <head>` policy is unchanged: a core member must post that
standalone line in an unedited personal-account comment or a non-dismissed review body. It
authorizes classifier work for a non-core author; it does not satisfy the attestation check.

## Using the gate

**Core authors.** Mark the pull request ready for review. The classifier then runs once for each
new head and records its verdict as the `ai-merge-gate/classifier` check, with its reasoning in a
pull request comment. Run the review outside CI on that exact head, using
`.claude/skills/ci-code-review/SKILL.md` or the chosen review skill. Once it finds no issues,
an allowed account posts the attestation with the actual skill, model and effort. The
`ai-review` label has no effect and starts no review.

**Other authors.** The gate spends nothing on a pull request from someone outside the core team
until a core member posts `AI gate: AUTHORIZED <head SHA>`. Post that line only after reading
the change. Each push needs a new authorization and a new review attestation from an allowed
account; the author's identity alone does not qualify.

**Verifier and gate changes.** Risk-3 can include a small verifier-reaching dependency change
and does not require human approval on the authorized path. Risk-4 requires a current-head
maintainer approval. Gate-input changes still need a writer's approval even when their label
is below risk-4. Every case still needs the separate review-attestation check.

**The events workflow.** The gate fails every pull request that changes
`.github/workflows/ai-merge-gate-events.yml`, approved or not (row 2). An admin merges such a
change with the gate off, as [Changing the events workflow](#changing-the-events-workflow)
describes.

**Re-running the gate.** Creating, editing or deleting a personal-account or Pika-bot PR
comment re-runs both checks, including an edit that removes the line. Other bots are ignored,
so the publishing App's classifier comment updates do not loop. PR changes and reviews also
re-run the gate through the events workflow. You can also re-run
the `AI merge gate` workflow run in the Actions tab. The Re-run button on the gate's own check
runs does nothing, because no server receives that request.

## How the gate is built

The gate never runs code from a pull request. Two workflows divide the work:

- `ai-merge-gate-events.yml` runs on `pull_request` and `pull_request_review` events. GitHub runs
  the pull request's own copy of this file, without secrets and with a token that has no
  permissions. The file does nothing except finish, which tells the gate that something changed.
  Because the pull request's copy runs, a pull request that changed it could keep its own review
  events from reaching the gate. Row 2 fails every such pull request, so the copy that runs is
  always the one on `main`.
- `ai-merge-gate.yml` runs on `issue_comment` and on `workflow_run` after the events workflow.
  GitHub always takes both of these workflows from the default branch. Its `resolve` job maps
  the event to a pull request number. Its `evaluate` job checks out the gate script and classifier skill
  from the default branch, mints a token for the gate's GitHub App, reads the pull request, its
  comments and its reviews through the API, and publishes check runs as the App. The independent
  `attest` job uses the same trusted script and environment but receives no model key; its App
  token has only Checks write, Pull requests read and Members read permissions.

GitHub documents the default-branch behavior and the `created`, `edited` and `deleted` comment
events in [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment).
The [workflow-run event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
can receive secrets even after an unprivileged workflow; this gate consumes no artifacts or
code from that run.

Each policy evaluation fixes the head it read and that head's merge base with `main` by SHA, and
compares the two commits' complete tree listings. The changed paths, old and new names of a
rename included, are the paths whose file, link or submodule differs between them. Rows 2 and 6
and the model's input all use that set. The gate does not use the pull request's file list,
which follows the branch: a push and a revert during an evaluation could make it describe
another commit with the same number of files.

The model's input is the pull request's diff from its merge base with `main`, followed by every
changed file in full, with line numbers, as the head has it. The gate reads the diff only after
rows 2 through 6 have passed. Two tools let the model read the rest of the repository:
`read_file` returns a file with line numbers or lists a directory, and `search` returns the
lines that contain a literal string. Each reads either the merge base or the head. The gate
answers them from GitHub's archive of each commit,
which it holds in memory and checks against the commit's tree: every file must hash to its blob
in that tree, so `export-ignore` or `export-subst` attributes in the pull request cannot hide or
alter a file, and a mismatch fails the gate. Nothing from the pull request is checked out,
written to disk or run.

The gate refuses rather than truncates. It fails, without calling the model, on an input over
1,000,000 characters, on a diff whose parsed file, addition and deletion counts differ from the
pull request's, on a truncated tree listing, and on a commit holding more than 500,000,000 bytes
of files. A tool answer over
200,000 characters, a search with more than 200 matching lines, and a binary file or one over
1,000,000 bytes come back to the model as errors, not as partial text. After 1,000,000
characters of answers or 40 tool turns for one question, the gate withholds the tools and the
model must answer from what it has read; the classifier skill tells it to answer `UNCERTAIN`
when that is not enough.

Both required check runs count only when the gate's App created them. The ruleset below
binds each check to the App's ID, so same-named jobs in a pull request's workflow, which GitHub
Actions would report, cannot satisfy them. The App's private key and the
OpenAI key live in an environment that only the default branch can use, so a workflow pushed to
any other branch cannot read them.

Row 6 keeps the AI path from changing the gate itself. Any workflow merged into the default
branch can use the gate's environment. Python looks up the gate's imports in `scripts/ci`
first, so any file merged there, such as a `json.py` that shadows the standard library or a
`__pycache__` entry, would run with the gate's credentials. The classifier skill, risk policy
and closure spec are the model's instructions. Changes to any of these always need a writer's approval, whatever the
classifier would say. Other files under `.github/`, such as issue templates, go to the
classifier like any other change.

The gate stores each classifier verdict and risk level as a check run on the head, created by
the App. Only the App can create or change those, which a comment would not guarantee, since
writers can edit anyone's comments. Later runs read these check runs instead of paying for a new
model call. Cached results must carry risk metadata and the current risk policy's SHA-256
digest, with a successful or neutral check conclusion. Legacy, failed or old-policy results
are classified again when authorized. A failed model call is not stored, so the next event
retries it. Attestations are always read from current comments, never from those check runs.

The gate does not use `pull_request_target`. GitHub's default Actions policy for public
repositories blocks that event from December 2, 2026 unless the repository allows it, and the
gate does not need it.

## Installing the gate

A repository admin installs the gate, with an organization owner or authorized app manager
for the GitHub App. This draft changes no settings. The examples below do not establish the
live protection settings; step 1 reads them with admin permission before any migration.

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
gate's setup changes the approval count and adds the two required gate checks, preserving
every existing status check.

Stop and ask the repository owner before continuing if the review settings require more than
one approval, a code owner review or approval of the last push. The gate counts one approval
and implements none of those three. Its risk-4 path requires a current-head core-team approval;
lower-risk human approvals follow GitHub's dismissal state. Step 7 keeps the recorded
dismiss-stale value.

Decide one more thing before step 6. The gate does not inspect other CI results. Preserve the
required checks found in step 1 and add every CI job that must pass before a merge to the gate ruleset's
`required_status_checks` with `"integration_id": 15368`, the GitHub Actions app. That also
applies to human-reviewed pull requests.

### 2. Create the GitHub App

As an organization owner, [register a separate GitHub App](https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app/registering-a-github-app)
for this gate. Do not reuse the Pika App: it is the allowed attestation author, whereas the
publishing App's classifier comments must never authorize a review or trigger another evaluation.

- Name it, for example, `jolt-ai-merge-gate`, and clear "Active" under Webhook: the gate
  receives no events.
- Repository permissions: Checks read and write, Commit statuses read and write, Contents
  read-only, Pull requests read and write, and the mandatory Metadata read-only. Contents covers
  the comparison, tree and archive reads that give the model the merge base and head.
- Organization permissions: Members read-only, which the team membership lookup needs.
- Allow installation only on this account.

After creating it, note the App ID, a number, for step 6, and the Client ID for step 3.
Generate a private key. Install the App on the organization for the `a16z/jolt` repository
only.

The [Checks API](https://docs.github.com/en/rest/checks/runs#create-a-check-run) requires Checks
write permission. The [team membership API](https://docs.github.com/en/rest/teams/members#get-team-membership-for-a-user)
requires organization Members read permission. Each workflow job requests only its needed
permissions using the [pinned App-token action](https://github.com/actions/create-github-app-token/blob/64ab729a6294000f5aa4a4bc0b73d1f32bc45510/action.yml).

GitHub's [required-check expected-source instructions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#require-status-checks-to-pass-before-merging)
require the installed App to have `statuses:write`, a recently submitted check run and an
association with a pre-existing required status check in the ruleset.
[Commit statuses write](https://docs.github.com/en/rest/commits/statuses#create-a-commit-status)
is for that admin prerequisite; the gate does not post commit statuses, and its job tokens
do not request that permission. App-bound activation has not been tested live, including
the required-check association; verify the binding during admin setup rather than treating
the configuration examples below as proof of merge blocking.

### 3. Set the repository variables

Under the repository's Actions variables, set:

- `AI_MERGE_GATE_CORE_TEAM` to the slug of the Jolt maintainer team whose active members may
  authorize classifier work, attest reviews and provide risk-4 approvals,
  the last part of `https://github.com/orgs/a16z/teams/<slug>`. An admin chooses the team; this
  pull request names none. The gate fails closed when the App cannot see the team.
- `AI_MERGE_GATE_CLIENT_ID` to the App's Client ID.

The Pika identity needs no variable. Verify the pinned identity before activation:

```bash
gh api 'users/0xandoroid-ai-agent[bot]' --jq '{login,id,node_id,type}'
```

It must return type `Bot`, numeric ID `336292405` and node ID `BOT_kgDOFAtqNQ`. A different
identity needs a reviewed code change, not a guessed replacement login.

### 4. Create the environment

Create an environment named `ai-merge-gate`, following GitHub's
[environment configuration procedure](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments):

- Under "Deployment branches and tags", choose "Selected branches and tags" and add the branch
  `main`.
- Add no required reviewers, wait timer or custom deployment protection rule. Reviewers and
  timers would hold every evaluation, and the workflow's `deployment: false` setting rejects
  custom rules.
- Add two environment secrets: `AI_MERGE_GATE_APP_PRIVATE_KEY`, the App's private key in PEM
  form, and `OPENAI_API_KEY`, a key for an OpenAI project that may call `gpt-6-astra` through
  the Responses API.

Do not store either secret as a repository or organization secret, where workflows on any
branch could read it. The classifier sends PR source to OpenAI in requests with
`background: true` and `store: true`. Confirm the project's data controls during activation;
this draft does not verify them.

### 5. Merge the gate and watch it

Create repository labels `risk-1`, `risk-2`, `risk-3`, `risk-4` and `needs-human-review` before
enabling the gate. The evaluator fails closed if a required label is missing; it never creates
label definitions. GitHub's [label API](https://docs.github.com/en/rest/issues/labels#add-labels-to-an-issue)
accepts either Issues write or Pull requests write permission. The publisher already requests
Pull requests write, so no additional Issues permission is needed. The attestation-only job
has no label-writing permission and changes no labels.

Merge this pull request with the protection unchanged. Nothing requires the gate yet, so a
failing gate blocks nothing. On the next few pull requests, confirm that the checks list shows
both `ai-merge-gate` and `ai-merge-gate/attestation` from the App. On a risk-1 through risk-3 PR,
confirm that a writer's approval passes the policy check but leaves the attestation check failing until an allowed
account posts the exact line. Check Pika-bot and maintainer comments, edits removing the line,
deletions and a new push. A ready core-authored PR should still get an
`ai-merge-gate/classifier` check and comment; that is the only model work in this gate. Confirm
exactly one risk label, risk-4's additional `needs-human-review`, and removal of both stale
managed labels when a new classification falls below level 4. Check that an earlier-head or
non-core approval cannot pass risk-4. No labels or settings were changed by this draft.

### 6. Create the two rulesets

Save these two files, replacing `<APP_ID>` with the App ID from step 2. The first requires the
App's policy and attestation checks. GitHub's [ruleset API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)
defines `context` as the check name and `integration_id` as its required publishing App.

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
        "required_status_checks": [
          {"context": "ai-merge-gate", "integration_id": <APP_ID>},
          {"context": "ai-merge-gate/attestation", "integration_id": <APP_ID>}
        ]
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

From this point a PR needs the existing approval and both passing gate checks. Risk-4 requires
the current-head maintainer approval described above. No approval satisfies the attestation check.

### 7. Hand the approval requirement to the gate

Set the classic protection's approval count to zero, keeping "Require a pull request before
merging" and every other setting. Pass the dismiss-stale value recorded in step 1:

```bash
gh api -X PATCH repos/a16z/jolt/branches/main/protection/required_pull_request_reviews \
  -F required_approving_review_count=0 -F dismiss_stale_reviews=<recorded> \
  -F require_code_owner_reviews=false -F require_last_push_approval=false
```

If step 1 found the approval count in a ruleset instead, fetch that ruleset, set its
`pull_request` rule's `required_approving_review_count` to 0, and `PUT` the whole ruleset back.
Then confirm that `gh api repos/a16z/jolt/rules/branches/main` lists both required check names
with the App's `integration_id`, and that the protection now requires zero approvals.

### Native auto-merge is an admin decision

The gate publishes checks and labels; it does not call a merge API or enable auto-merge on a
PR. After confirming both required checks and the chosen CI requirements, a repository
maintainer can enable **Allow auto-merge** under the repository's pull-request settings,
following [GitHub's auto-merge procedure](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-auto-merge-for-pull-requests-in-your-repository).
People with write permission can then enable native auto-merge on individual PRs. GitHub
waits for all required checks and reviews. Risk-1 through risk-3 need no human approval on the
authorized path; risk-4 cannot pass the policy check without the required current-head
maintainer approval. The separate attestation check remains required in either case.

This task did not inspect or change the live native auto-merge setting and did not enroll this
draft. There is no new merge service or merge-queue configuration in the patch.

## Turning the gate off

The off switch restores GitHub's native review requirement:

```bash
gh api -X PUT repos/a16z/jolt/rulesets/<human review ruleset ID> -f enforcement=active
```

With both rulesets active, every PR needs a native approval and a review attestation. To
disable the entire gate, including the attestation requirement, disable its ruleset **after**
activating the human-review ruleset:

```bash
gh api -X PUT repos/a16z/jolt/rulesets/<gate ruleset ID> -f enforcement=disabled
```

Activate human review before disabling the gate ruleset, so that `main` always requires a
review. To turn the gate back on, activate the gate ruleset, wait until open pull requests show
both checks, then disable the human-review ruleset. To remove the gate, turn it off, delete the
gate ruleset, then delete the gate workflows, script, risk policy, classifier skill, environment,
variables and publishing App. Remove the five managed label definitions if retiring risk labels.
Keep the separate Pika App and local review skill.

### Changing the events workflow

The gate fails every pull request that changes `.github/workflows/ai-merge-gate-events.yml`,
approved or not (row 2), because GitHub runs the pull request's own copy of that file for its
reviews. To merge such a change, turn the gate off with both commands above, merge the pull
request with a native approval, then turn the gate back on.

## Limits

- Event delivery delays re-evaluation of withdrawn evidence. The gate cannot reflect a
  dismissed approval or an edited or deleted comment until an evaluation reads that state
  and publishes a result. Comments reach the gate directly. Reviews and
  pull request changes reach it through the events workflow, which GitHub holds while a
  first-time contributor's workflow runs await a maintainer's approval, and does not run for
  `pull_request` activity while the pull request has a merge conflict; a comment still triggers
  the gate in both cases. Evaluations of one pull request run one at a time, and a pass rests on
  state read just before it is published. A change in the seconds between that read and the
  publication, or one GitHub delivers late, is not observed until a later evaluation. If
  the gate cannot reach GitHub, existing check runs remain unchanged.
- Team changes trigger no event. Removing someone from the core team does not re-evaluate open
  pull requests until their next event.
- The [Checks API](https://docs.github.com/en/rest/checks/runs#create-a-check-run) creates check
  runs for a commit, not a pull request. PRs with the same head share those commit-scoped runs
  and cached classifier results. Parsing rejects a line for another PR, but this does not
  establish PR-scoped merge isolation: which same-name run GitHub selects for mergeability,
  including across PRs sharing a head, has not been verified. The attestation and risk-4
  approval requirements inherit this commit-scoped limit; they are not PR-scoped merge locks.
  Risk labels themselves belong to the individual PR.
- The model reads the source but does not build or run it, and it reads only text files up to
  1,000,000 bytes. A change it cannot read, such as a binary file in a verifier crate, should
  make the classifier answer `UNCERTAIN`, which sends the pull request to human review.
- The size limits bound what the model receives and what the gate keeps, not the evaluation's
  peak memory: the gate downloads a commit's whole archive before checking its files, and builds
  a search's matches before refusing more than 200 of them. A very large tree can exhaust the
  runner, which fails the run without a pass.
- GitHub has no head-conditional label transaction. A push between a label API read and write
  can briefly leave the old head's label; the final head read refuses a stale check pass, and
  the next evaluation replaces it. Multiple label writes can also briefly show two risk labels.
  Label permissions, availability and API failures can prevent a label update; the policy
  check then fails, not passes. Labels are outputs, never merge authority supplied by a PR.
- Withdrawing an authorization while the classifier runs stops the pass, not classifier work
  already under way. There is no subsequent CI code-review request.
- A verdict covers the diff from the head's merge base with `main`. A later change to `main` does
  not void it, just as it does not void a human approval when branches need not be up to date.
- Comment authorship and last-editor identity come from GitHub. Both must be allowed accounts
  for an edited attestation. The gate does not establish who performed the review.
- The gate does not reproduce code owner review, more than one required approval, or approval of
  the last push. A maintain-permission read cannot show whether the current protection requires
  them; step 1 reads them with admin permission and stops if any is required.

This draft does not activate either check. Focused fixtures exercise API state and emitted
check conclusions; live webhook delivery, App installation, environment protections, branch
rules and classifier accuracy remain unverified until the admin activation checks above.
