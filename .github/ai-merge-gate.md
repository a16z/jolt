# AI merge gate

Once installed, `main` requires one approving review, with stale approvals dismissed on push,
plus its required CI checks. The gate is a workflow that supplies that approval for pull
requests on the agent path: it approves the exact current head as `github-actions[bot]` when
all of the following hold, and withdraws its approval when any stops holding.

- A maintainer wrote the pull request, or a maintainer posted `AI gate: AUTHORIZED <head>`.
- The Pika App bot or a maintainer attests a clean review of that head in a PR comment.
- The verifier classifier puts the head at risk-1, risk-2 or risk-3.
- The pull request changes none of the gate's own inputs.
- No maintainer or writer has a standing change request.
- The pull request is open, targets `main` and is not a draft.

The gate never approves risk-4. A human approval satisfies GitHub's rule on its own; the gate
neither repeats nor converts it. A maintainer is a login in the repository variable
`AI_MERGE_GATE_MAINTAINERS`; a writer is an account with write, maintain or admin access.

Only the verifier-impact classifier calls a model in this gate: OpenAI's `gpt-6-astra` at
reasoning effort `max`, through the Responses API, within the spend limits below. It reads the
pull request's merge base and head through two read-only tools. Code review happens outside CI,
and the attestation records it.

The gate also keeps four kinds of labels on each pull request and publishes an `ai-merge-gate`
check run that explains its decision. Neither is required for merging. This draft enables
nothing: no setting, label, ruleset or secret changed.

This file is the policy, the contributor guide and the admin procedure. The gate's code is
`scripts/ci/ai_merge_gate.py`, `.github/workflows/ai-merge-gate.yml` and
`.github/workflows/ai-merge-gate-events.yml`. The classifier's instructions are
`.claude/skills/verifier-classifier/SKILL.md`. The local review guidance is
`.claude/skills/ci-code-review/SKILL.md`; CI does not load it.

## What GitHub enforces and what only the policy asks

GitHub enforces the rules in the ruleset: the number of approvals, stale-approval dismissal and
the required checks. Under the
[rules for pull requests](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#require-a-pull-request-before-merging),
an approval counts when it comes from someone with write permission, and a reviewer who requests
changes must approve before the pull request can merge. GitHub dismisses an approval as stale
when a push changes the pull request's diff.

The gate's policy asks for more than one approval from anyone with write access. At risk-4 it
asks for an approval of the current head from a maintainer other than the author, and it asks
for a writer's approval of any change to the gate's inputs or from an author no maintainer has
authorized. GitHub cannot check maintainer-list membership or risk levels, so a human approval
from any writer satisfies the ruleset whatever the policy asks. For human approvals the policy
is advisory: the `risk-4`, `no-reviewer-assigned` and `merge-eligible` labels and the
`ai-merge-gate` check tell reviewers what it wants, and nothing blocks the merge. The gate
enforces the policy only through its own approval, which it gives and withdraws as above.

## Risk labels set the review threshold

The trusted policy is [`.github/ai-merge-gate-risk.json`](ai-merge-gate-risk.json). It defines
the following levels and path globs; the classifier supplies semantic evidence for ambiguous changes.

| Label | Meaning | Review the policy asks for |
|---|---|---|
| `risk-1` | Tests, scripts, CI or documentation only | An attestation or a writer's approval |
| `risk-2` | Prover or other non-verifier code only | An attestation or a writer's approval |
| `risk-3` | Small, scoped dependency or shared-code changes that could reach the verifier | An attestation or a writer's approval |
| `risk-4` | Protocol, sumcheck order or symbolic relations, transcript/Fiat–Shamir, verifier logic, PCS or commitment parameters; also unknown changes | A current-head maintainer approval |

For production changes, the highest matching path level is a minimum. The classifier can
raise it for effects elsewhere. A per-file risk-1 verdict can identify test-only or comment-only
hunks inside a production file. Every changed path must appear exactly once in the classifier's
result, including removed files and both names of a rename. The pull request takes the highest
effective level; an incomplete or invalid result fails the evaluation and receives risk-4 when
GitHub is reachable. `UNCERTAIN` always becomes risk-4. `AFFECTS` with risk-3 remains eligible
for the gate's approval: verifier impact is reported separately and is no longer a blanket veto.

Without a cached or new classification, a change wholly within level-1 paths receives risk-1;
other changes receive risk-4 until they can be classified. This gives unauthorized and draft
pull requests a conservative label without running a model. A current-head maintainer approval
already satisfies the highest level, so the gate starts no model work for it and keeps an
existing classification or this fallback.

## Labels the gate manages

Each evaluation sets exactly these labels and leaves every other label alone:

- `risk-1` through `risk-4`: one risk label, as above.
- `needs-maintainer-review`: the author is not a maintainer and no maintainer has posted
  `AI gate: AUTHORIZED <head>` for the current head. It reports that the agent path lacks a
  maintainer's authorization, so it stays even when a writer's approval satisfies the policy.
- `merge-eligible`: the gate's review policy is satisfied for the current head, as the table
  below decides, and the pull request is open and not a draft. It does not show that CI passed,
  that GitHub would allow the merge, or that auto-merge is enabled.
- `no-reviewer-assigned`: the policy waits for a person (table rows 3, 5 and 8, or a gate
  error), and the pull request has no requested reviewer, no requested team and no assignee.

Requesting a reviewer or team, assigning someone, or satisfying the requirement removes
`no-reviewer-assigned` at the next evaluation. Earlier drafts of this gate also managed
`needs-human-review`; `risk-4`, `no-reviewer-assigned` and the check replace it, and the gate
no longer reads or writes it. Labels are outputs: the gate never reads them as evidence.

The gate reads the API's current head before and after the label writes and refuses a stale
pass. It then evaluates the table again before approving.

## How the gate decides

The gate reads the pull request's current head commit, which this file calls the head, and
evaluates these rows in order. The classifier row continues after recording its result; the
first other matching row decides. "Withdrawn" means the gate dismisses any approval of its own
that is still standing.

| # | Condition | Policy | Gate's approval | Paid model work |
|---|---|---|---|---|
| 1 | The pull request is closed, or its base is not `main` | Not evaluated | Unchanged | None |
| 2 | A maintainer or writer has requested changes, on any commit, and has not since approved or had that review dismissed | Fail | Withdrawn | None |
| 3 | No writer approved, the author is not a maintainer, and no maintainer posted `AI gate: AUTHORIZED <head>` | Fail: human approval required | Withdrawn | None |
| 4 | No writer approved and the pull request is a draft | Fail | Withdrawn | None |
| 5 | No writer approved and the pull request changes a gate input: `.github/workflows/`, `scripts/ci/`, the classifier skill, closure spec or risk policy | Fail: human approval required | Withdrawn | None |
| 6 | Maintainer-authorized, ready, no gate-input changes, no current-head maintainer approval, and no cached classifier result | Run the classifier within its attempt limits, then continue | — | Classifier |
| 7 | Risk-4 with an undismissed current-head approval by a maintainer with write access, other than the author | Pass | Withdrawn | None |
| 8 | Risk-4 without that approval | Fail: current-head maintainer approval required | Withdrawn | None |
| 9 | Risk-1 through risk-3, and a writer other than the author approved without a later dismissal or change request | Pass | Given if the next row's attestation, maintainer authorization, readiness and gate-input conditions also hold; otherwise withdrawn | None |
| 10 | Risk-1 through risk-3, and the Pika App bot or a maintainer posted the exact current-head attestation below | Pass | Given | None |
| 11 | Otherwise | Fail: needs an approval or a current-head review attestation | Withdrawn | None |

Rows 3 through 5 apply only without a writer's approval, so reaching row 10 means the author
or a sign-off authorized the head, the pull request is ready and it changes no gate input.

Human approvals follow GitHub's own review state for risk-1 through risk-3: an approval of an
earlier commit counts until GitHub dismisses it. Risk-4 is stricter: an earlier-head approval
never counts, even if GitHub has not dismissed it. The AI evidence names the head: a push voids
every sign-off, attestation and classifier verdict of an earlier head, and the gate's approval
names the head it approved. Unlike the research lab's `check_review.py`, the gate keeps no
exception for a rebase that leaves the patch unchanged.

## The gate's approval

The gate approves with the workflow's own `GITHUB_TOKEN`, so the review's author is
`github-actions[bot]` and its body starts with `<!-- ai-merge-gate/approval -->`. It submits the
review with `commit_id` set to the head it evaluated, at most once per head: an evaluation that
finds its approval of the head still standing changes nothing. It dismisses only reviews by that
account that start with that marker. It never dismisses or changes a person's review, a Pika
review, or an approval another workflow submitted with the same token.

Each evaluation withdraws the gate's standing approvals that the current state no longer
supports: a deleted or edited attestation or sign-off, a new change request, a reclassification,
a changed maintainer list or risk policy, and any approval that names an earlier head. GitHub
dismisses stale approvals itself when a push changes the diff; a push that leaves the diff
unchanged keeps the approval, which the gate's evaluation of the new head then dismisses.

A pass rests on state read after the label writes: before it approves, the gate reads the pull
request, its comments and reviews, and permissions again and evaluates the table once more,
reusing the model results it already has. When the new state needs a classifier result the run
does not have, for example after an approval was dismissed, it fails with "The pull request
changed while the gate ran", and the next event runs the gate again. After submitting an
approval, the gate reads the head once more. GitHub dismisses stale approvals when a push
arrives, which misses an approval of the old head submitted just after it, so when the head
moved the gate dismisses the approval it just submitted.

An evaluation that fails withdraws the gate's standing approvals before reporting the error,
when GitHub is reachable. Reviews, dismissals, labels and comments made with `GITHUB_TOKEN`
start no workflow run ([GitHub's token documentation](https://docs.github.com/en/actions/concepts/security/github_token)),
so the gate's own writes never trigger another evaluation, and each evaluation finishes its own
reconciliation.

To stop the gate's approval of a head, request changes: GitHub then blocks the merge, and the
gate withdraws its approval and does not approve again until that review is approved or
dismissed. Dismissing the gate's approval by hand does not stop it; its next evaluation
approves again if the conditions still hold.

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

An allowed author is either a maintainer's personal account or the Pika App bot. Being the PR
author or having repository write access alone grants no attestation authority. The Pika
identity is pinned by its GitHub node ID, `BOT_kgDOFAtqNQ`, not by a name in the body. The REST
API identifies it as `0xandoroid-ai-agent[bot]`, numeric ID `336292405`; GraphQL reports the
same node ID with login `0xandoroid-ai-agent` (verified 2026-10-31). The gate's own comments
and reviews, written as `github-actions[bot]`, never attest.

The check reads the comment's current body, including edits, and uses its API-reported author.
For an edited comment, its last editor must also be an allowed account; write access alone
cannot add an attestation to someone else's comment. Editing away the line or deleting the
comment invalidates it on re-evaluation, which withdraws the gate's approval. This checks the
account's claim, not evidence that the named skill, model or effort actually ran.

The separate `AI gate: AUTHORIZED <head>` sign-off: a maintainer must post that standalone line
in an unedited personal-account comment or a non-dismissed review body. It authorizes classifier
work and the gate's approval for an author who is not a maintainer; it does not attest a review.

## Using the gate

**Maintainer authors.** Mark the pull request ready for review. The classifier then runs once
for each new head and records its verdict as the `ai-merge-gate/classifier` check, with its
reasoning in a pull request comment. Run the review outside CI on that exact head, using
`.claude/skills/ci-code-review/SKILL.md` or the chosen review skill. Once it finds no issues, an
allowed account posts the attestation with the actual skill, model and effort, and at risk-1
through risk-3 the gate approves. The `ai-review` label has no effect and starts no review.

**Other authors.** The gate spends nothing on a pull request from someone who is not a
maintainer until a maintainer posts `AI gate: AUTHORIZED <head SHA>`. Post that line only after
reading the change. Each push needs a new authorization and a new attestation from an allowed
account; the author's identity alone does not qualify. A writer's approval merges the pull
request without either.

**Verifier and gate changes.** Risk-3 can include a small verifier-reaching dependency change
and still gets the gate's approval on the authorized path. At risk-4 and for changes to the gate's
inputs, the gate never approves; a person's approval is needed, and the policy asks for a
current-head maintainer approval at risk-4.

**Re-running the gate.** Creating, editing or deleting a personal-account or Pika-bot PR comment
re-runs the gate, including an edit that removes the line. Comments from other bots are ignored.
Pushes, ready and draft changes, edits, review requests, assignments and reviews re-run it
through the events workflow. You can also re-run the `AI merge gate` workflow run in the Actions
tab. The Re-run button on the gate's own check runs does nothing, because no server receives
that request.

## How the gate is built

The gate never runs code from a pull request. Two workflows divide the work:

- `ai-merge-gate-events.yml` runs on `pull_request` and `pull_request_review` events. GitHub runs
  the pull request's own copy of this file, without secrets and with a token that has no
  permissions. The file does nothing except finish, which tells the gate that something changed.
  A pull request that changes it could keep its own events from reaching the gate, but such a
  pull request changes a gate input and never gets the gate's approval.
- `ai-merge-gate.yml` runs on `issue_comment` and on `workflow_run` after the events workflow.
  GitHub always takes both of these workflows from the default branch. Its `resolve` job maps the
  event to a pull request number. Its `evaluate` job checks out the gate script, risk policy,
  classifier skill and closure spec from the default branch, reads the pull request, its comments
  and its reviews through the API, calls the classifier when row 6 applies, and writes check runs,
  labels, the classifier comment and its review with the job's `GITHUB_TOKEN`. One evaluation per
  pull request runs at a time.

GitHub documents the default-branch behavior and the `created`, `edited` and `deleted` comment
events in [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment).
The [workflow-run event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
can receive secrets even after an unprivileged workflow; this gate consumes no artifacts or
code from that run.

The `evaluate` job's token has Actions read, Checks write, Contents read and Pull requests write;
the `resolve` job's has Contents read and Pull requests read. Pull requests write covers
[creating and dismissing reviews](https://docs.github.com/en/rest/pulls/reviews),
[labels](https://docs.github.com/en/rest/issues/labels) and
[the classifier comment](https://docs.github.com/en/rest/issues/comments). Checks write covers
[check runs](https://docs.github.com/en/rest/checks/runs). Actions read covers
[reading a workflow run](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run),
which the classifier cache below needs. Contents read covers the comparison, tree and archive
reads that give the model the merge base and head.

Each policy evaluation fixes the head it read and that head's merge base with `main` by SHA, and
compares the two commits' complete tree listings. The changed paths, old and new names of a
rename included, are the paths whose file, link or submodule differs between them. Row 5, the
gate's approval and the model's input all use that set. The gate does not use the pull
request's file list, which follows the branch: a push and a revert during an evaluation could make
it describe another commit with the same number of files.

The model's input is the pull request's diff from its merge base with `main`, followed by every
changed file in full, with line numbers, as the head has it. The gate reads the diff only after
rows 2 through 5 have passed. Two tools let the model read the rest of the repository:
`read_file` returns a file with line numbers or lists a directory, and `search` returns the
lines that contain a literal string. Each reads either the merge base or the head. The gate
answers them from GitHub's archive of each commit, which it holds in memory and checks against
the commit's tree: every file must hash to its blob in that tree, so `export-ignore` or
`export-subst` attributes in the pull request cannot hide or alter a file, and a mismatch fails
the gate. Nothing from the pull request is checked out, written to disk or run.

The gate refuses rather than truncates. It fails, without calling the model, on a diff whose
parsed file, addition and deletion counts differ from the pull request's, on a truncated tree
listing, on a commit holding more than 500,000,000 bytes of files, and on a request over the
input bound below. A tool answer over 60,000 characters, a search with more than 200 matching
lines, and a binary file or one over 1,000,000 bytes come back to the model as errors, not as
partial text. After 100,000 characters of answers or 4 tool turns for one question, or when the
answers would take the next request over the input bound, the gate withholds the tools and the
model must answer from what it has read; the classifier skill tells it to answer `UNCERTAIN`
when that is not enough.

Row 5 keeps the gate from approving changes to itself. Any workflow merged into the default
branch can use the gate's secret and token. Python looks up the gate's imports in `scripts/ci`
first, so any file merged there, such as a `json.py` that shadows the standard library or a
`__pycache__` entry, would run with the gate's credentials. The classifier skill, risk policy
and closure spec are the model's instructions. Other files under `.github/`, such as issue
templates, go to the classifier like any other change.

The gate stores each classifier verdict and risk level as a check run on the head, whose
external ID names the workflow run that created it. Later evaluations reuse a verdict instead of
paying for a new one only when the check run comes from the GitHub Actions app (ID 15368) and
its external ID names a run of `.github/workflows/ai-merge-gate.yml` started by `issue_comment`
or `workflow_run` in this repository: GitHub runs that file for those events only from the
default branch, and a job a pull request adds does not choose its check run's external ID.
Cached results must also carry risk metadata and the current risk policy's SHA-256 digest, with
a successful or neutral check conclusion. Legacy, failed or old-policy results are classified
again when authorized. A failed model call is not stored, so the next event may retry it
within the attempt limits. Attestations are always read from current comments, never from check
runs.

The gate does not use `pull_request_target`. GitHub's default Actions policy for public
repositories blocks that event from December 2, 2026 unless the repository allows it, and the
gate does not need it.

## Classifier spend limits

These limits bound what the classifier can spend; they apply to the CI classifier only, not to
local reviews. They are this gate's own choices, set in `scripts/ci/ai_merge_gate.py`.

- **Output.** Each request sets `max_output_tokens` to 20,000, which the
  [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
  defines as "an upper bound for the number of tokens that can be generated for a response,
  including visible output tokens and reasoning tokens". A response that reaches it ends
  incomplete, and that attempt fails.
- **Input.** No request goes out whose counted input exceeds 150,000 tokens. The gate counts a
  request by its JSON size in bytes, since no text token covers less than one byte, plus the
  input and output tokens the API reported for the response it continues: OpenAI bills "all
  previous input tokens for responses in the chain" as input
  ([conversation state](https://developers.openai.com/api/docs/guides/conversation-state)).
  The API's own message framing, a few tokens per message, is not counted. A first request over
  the bound fails the evaluation before any paid call and counts no attempt.
- **Calls per attempt.** An attempt is one question with at most 4 tool turns, so at most 5
  paid requests.
- **Attempts.** At most 2 attempts per head and 3 per pull request, across all its heads,
  retries and re-runs. The gate's classifier comment ends with a ledger line counting attempts
  per head; a new attempt is written there before its first paid request, so a timeout, crash or
  cancelled run still counts. One evaluation per pull request runs at a time, so two
  evaluations never admit attempts at once. When the attempts are spent, the head stays
  unclassified: risk-4, which needs a person's approval, unless only level-1 paths change. A
  cached verdict costs no attempt.
- **Time.** Each request is cancelled after 15 minutes and each question after 40. Cancelling
  ends the wait, not necessarily the billing: the tokens a response generated before the
  cancellation, at most its output limit, can still be charged.
- **Tier.** Requests set `service_tier` to `default`, which the API documents as "standard
  pricing and performance for the selected model", rather than inheriting the project's tier.

[OpenAI's price list for gpt-6-astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
charges, per million tokens, $10 for input, $1 for cached input, $12.50 for cache writes and $50
for output, with 2x input and 1.5x output rates for prompts over 272,000 input tokens. The input
bound keeps every request below that threshold. Charging every input token at the cache-write
rate, the highest standard input rate, one request costs at most 150,000 × $12.50/M + 20,000 ×
$50/M = $2.875, one attempt at most 5 × $2.875 = $14.375, and one pull request at most
3 × $14.375 = $43.125 at list price. This is a bound, not a measurement: actual use has not
been measured.

The limits above are enforced by the gate's code. As a second cap, give the key its own OpenAI
project and set a monthly hard spend limit on that project. OpenAI's
[spend-limit guide](https://developers.openai.com/api/docs/guides/spend-limits) distinguishes
the two controls: a spend alert "sends a notification; API traffic continues", while a hard
spend limit makes affected requests return HTTP 429 with `project_spend_limit_exceeded`.
"Enforcement is not instantaneous, so recorded spend can slightly exceed the configured amount."
When the limit is reached, classifier attempts fail and the gate approves nothing until the
limit resets or is raised.

## Trust

Writers are trusted. Anyone with write access can push a branch whose workflow uses that
branch's `GITHUB_TOKEN` to approve a pull request, to publish check runs under the gate's names,
or to read repository secrets, and can edit or delete the gate's classifier comment, which
resets its attempt counts. The gate does not defend against writers.

Workflows that GitHub runs for pull requests from forks receive no secrets and a read-only
`GITHUB_TOKEN` ([events documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)),
so a fork's workflow can neither approve nor read `OPENAI_API_KEY`. Workflows triggered by
`pull_request_target`, such as `claude-routines.yml`, run the base branch's trusted code with
secrets; they must not run a fork's code. The gate reads the key as the repository secret
`OPENAI_API_KEY`, as the routines read their `ANTHROPIC_*` secrets.

## Installing the gate

A repository admin installs the gate. This draft changes no settings. The examples below do not
establish the live protection settings; step 1 reads them with admin permission first. The
approval design has been tried only in a personal test repository, where an approval by
`GITHUB_TOKEN` counted toward a ruleset's one required approval and a push dismissed it.
Organization repositories, classic branch protection and organization-level Actions policies are
untested; the checks in step 6 establish them for this repository.

### 1. Record the current protection

```bash
gh api repos/a16z/jolt/branches/main/protection > main-protection.json
gh api 'repos/a16z/jolt/rulesets?includes_parents=true' > rulesets.json
gh api repos/a16z/jolt/rules/branches/main > main-rules.json
```

Write down the review settings, wherever they live: `required_pull_request_reviews` in the
classic protection, or a `pull_request` rule in a ruleset that targets `main`. Record the
approval count, the settings for dismissing stale reviews, code owner review, approval of the
last push, conversation resolution and who may dismiss reviews, and every required status check.

Stop and ask the repository owner before continuing if the settings require more than one
approval, a code owner review or approval of the last push: the gate supplies one approval from
an account that is not a code owner.

### 2. Set the maintainer variable

Under the repository's Actions variables, set `AI_MERGE_GATE_MAINTAINERS` to the maintainers'
GitHub logins separated by commas. They authorize agent work for other authors, attest reviews,
and give the risk-4 approvals the policy asks for. The admin chooses them; this pull request names
none. A missing, empty or malformed list fails every evaluation, which then approves nothing. It
must be a repository variable, not an environment variable, since the workflow uses no
environment.

The Pika identity needs no variable. Verify the pinned identity before activation:

```bash
gh api 'users/0xandoroid-ai-agent[bot]' --jq '{login,id,node_id,type}'
```

It must return type `Bot`, numeric ID `336292405` and node ID `BOT_kgDOFAtqNQ`. A different
identity needs a reviewed code change, not a guessed replacement login.

### 3. Provide the OpenAI key

The gate reads the repository secret `OPENAI_API_KEY`, which must hold a key for an OpenAI
project that may call `gpt-6-astra` through the Responses API. Give that project its own hard
spend limit, as [Classifier spend limits](#classifier-spend-limits) describes. The classifier sends
PR source to OpenAI in requests with `background: true` and `store: true`; confirm the project's
data controls during activation, which this draft does not verify.

Optional hardening narrows who can read the key: create an environment named `ai-merge-gate`
whose deployment branches admit only `main`, move the key there as an environment secret, delete
the repository secret, and add `environment: {name: ai-merge-gate, deployment: false}` to the
`evaluate` job in a reviewed change. Writers' branch workflows can then no longer read the key.

### 4. Create the labels

Create the labels `risk-1`, `risk-2`, `risk-3`, `risk-4`, `needs-maintainer-review`,
`merge-eligible` and `no-reviewer-assigned`. The gate fails closed if one is missing; it never
creates label definitions.

### 5. Merge the gate and watch it

Leave "Allow GitHub Actions to create and approve pull requests" off and merge this pull
request. GitHub then refuses the gate's approvals, so an evaluation that reaches an approval
fails with a gate error and nothing is approved. On the next few pull requests, confirm the
risk labels, the other managed labels, the `ai-merge-gate` check and, on a ready
maintainer-authored pull request, the `ai-merge-gate/classifier` check and comment.

### 6. Let the gate approve

Under the repository's Actions settings, in "Workflow permissions", turn on "Allow GitHub
Actions to create and approve pull requests", which
[controls whether `GITHUB_TOKEN` can create and approve pull requests](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#preventing-github-actions-from-creating-or-approving-pull-requests).
An organization setting can prevent a repository from turning it on. Every workflow whose token
has Pull requests write can then approve; today only the gate's does.

Then make `main` require one approval with stale approvals dismissed, keeping every required
check from step 1. If the protection is a ruleset, its `pull_request` rule needs:

```json
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
```

In classic protection the same settings are "Require approvals" with one approval and "Dismiss
stale pull request approvals when new commits are pushed". Leave "Restrict who can dismiss pull
request reviews" off. The [REST documentation](https://docs.github.com/en/rest/pulls/reviews#dismiss-a-review-for-a-pull-request)
says that on a protected branch a dismisser must be a repository administrator or on that list;
whether the gate can dismiss its own approval under this repository's protection is untested,
and the checks below establish it.

On a test pull request, confirm each of these before relying on the gate:

- an attestation of a risk-1 through risk-3 head makes the gate approve it, and the merge box
  counts that approval;
- deleting the attestation makes the gate dismiss its approval;
- a push dismisses the approval, and the gate approves the new head only after a new
  attestation;
- a risk-4 head gets no approval from the gate.

If the gate cannot dismiss its approval, it cannot withdraw it: turn the gate off as below.

## Turning the gate off

Turn off "Allow GitHub Actions to create and approve pull requests" and disable the `AI merge
gate` workflow. Approvals the gate already gave remain until a push dismisses them; list and
dismiss them by hand:

```bash
gh api --paginate 'repos/a16z/jolt/pulls?state=open&base=main' --jq '.[].number' | while read -r pr; do
  gh api --paginate "repos/a16z/jolt/pulls/$pr/reviews" --jq '.[] | select(.user.login == "github-actions[bot]"
    and .state == "APPROVED" and (.body | startswith("<!-- ai-merge-gate/approval -->"))) | .id' |
  while read -r id; do
    gh api -X PUT "repos/a16z/jolt/pulls/$pr/reviews/$id/dismissals" -f message='AI merge gate turned off' -f event=DISMISS
  done
done
```

To remove the gate, turn it off, then delete the gate workflows, script, risk policy, classifier
skill, the maintainer variable, the key if nothing else uses it, and the managed label definitions
if retiring them. Keep the separate Pika App and local review skill.

## Limits

- Event delivery delays withdrawal. The gate cannot withdraw its approval after a dismissed
  review, an edited or deleted comment or a new change request until an evaluation reads that
  state. Comments reach the gate directly. Reviews and pull request changes reach it through the
  events workflow, which GitHub holds while a first-time contributor's workflow runs await a
  maintainer's approval, and does not run for `pull_request` activity while the pull request has
  a merge conflict; a comment still triggers the gate in both cases. A change in the seconds
  between the gate's last read and its approval, or one GitHub delivers late, is not observed
  until a later evaluation.
- An evaluation that fails before the script starts, for example when GitHub cannot start the
  job, cannot withdraw an earlier approval of the same head; the next evaluation does.
- Variable, risk-policy and permission changes trigger no event. A maintainer removed from the
  variable, or a writer losing access, changes nothing on open pull requests until their next
  event.
- A push that leaves the diff unchanged keeps the gate's approval of the old head until the
  gate's evaluation of the push dismisses it.
- The [Checks API](https://docs.github.com/en/rest/checks/runs#create-a-check-run) creates check
  runs for a commit, not a pull request, so pull requests with the same head share cached
  classifier results. GitHub also refuses to merge a pull request while another open pull request
  whose head branch points to the same commit has pending or rejected reviews.
- The model reads the source but does not build or run it, and it reads only text files up to
  1,000,000 bytes. A change it cannot read, such as a binary file in a verifier crate, should
  make the classifier answer `UNCERTAIN`, which sends the pull request to a person.
- The size limits bound what the model receives and what the gate keeps, not the evaluation's
  peak memory: the gate downloads a commit's whole archive before checking its files, and builds
  a search's matches before refusing more than 200 of them. A very large tree can exhaust the
  runner, which fails the run without an approval.
- Whether max-effort classifications fit in 20,000 output tokens has not been measured. If they
  often do not, attempts fail and the gate falls back to human review until the limit is raised.
- GitHub has no head-conditional label transaction. A push between a label API read and write
  can briefly leave the old head's labels, and labels stay as they were while an evaluation runs
  or after a pull request closes. Label permissions, availability and API failures can prevent a
  label update; the evaluation then fails and approves nothing.
- Withdrawing an authorization while the classifier runs stops the approval, not classifier work
  already under way.
- A verdict covers the diff from the head's merge base with `main`. A later change to `main` does
  not void it, just as it does not void a human approval when branches need not be up to date.
- Comment authorship and last-editor identity come from GitHub. Both must be allowed accounts
  for an edited attestation. The gate does not establish who performed the review.

This draft activates nothing. Focused fixtures exercise API state, the emitted reviews,
dismissals, labels and check conclusions; live webhook delivery, the Actions approval setting,
dismissal permission, branch rules and classifier accuracy remain unverified until the admin
activation checks above.
