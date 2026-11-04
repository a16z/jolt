# AI merge gate

Once installed, `main` requires one approving review, with stale approvals dismissed on push,
plus its required CI checks. The gate is a workflow that supplies that approval for pull
requests on the agent path: it approves the exact current head as `github-actions[bot]` when
all of the following hold, and withdraws its approval when any stops holding, unless the pull
request has merged.

- A maintainer wrote the pull request.
- The Pika App bot or a maintainer attests a clean review of that head in a PR comment.
- A verifier classifier result for that head puts it at risk-1, risk-2 or risk-3.
- The pull request changes none of the protected inputs listed in the decision table below.
- No maintainer or writer has a standing change request.
- The pull request is open, targets `main` and is not a draft.

The gate never approves risk-4. A human approval satisfies GitHub's rule on its own; the gate
neither repeats nor converts it. A maintainer is a login in the repository variable
`AI_MERGE_GATE_MAINTAINERS` who also currently has write access; a writer is an account with
write, maintain or admin access.

Only the verifier-impact classifier calls a model in this gate: OpenAI's `gpt-6.1-sol` at
reasoning effort `xhigh`, through the Responses API, within the spend limits below. It reads the
pull request's merge base and head through two read-only tools. Code review happens outside CI,
and the attestation records it.

The gate also manages labels on each pull request and publishes an `ai-merge-gate`
check run that explains its decision. Neither is required for merging. This pull request enables
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
for a writer's approval of any change to the gate's inputs or from a non-maintainer author.
GitHub cannot check maintainer-list membership or risk levels, so a human approval
from any writer satisfies the ruleset whatever the policy asks. For human approvals the policy
is advisory: the managed labels and the `ai-merge-gate` check tell reviewers what it wants;
the check adds no merge requirement. GitHub's required-approval rule still blocks merging
without an approval. The gate enforces the policy only through its own approval, which it
gives and withdraws as above.

## Risk labels set the review threshold

The trusted policy is [`.github/ai-merge-gate-risk.json`](ai-merge-gate-risk.json). It defines
the following levels and path globs; the classifier supplies semantic evidence for ambiguous changes.

| Label | Meaning | Review the policy asks for |
|---|---|---|
| `risk-1` | Tests, scripts, CI or documentation only | An attestation or a writer's approval |
| `risk-2` | Prover or other non-verifier code; dependency bumps outside the verifier dependency closure | An attestation or a writer's approval |
| `risk-3` | Dependency bumps in the verifier dependency closure; scoped shared-code changes under the rule below | An attestation or a writer's approval |
| `risk-4` | Verifier/preprocessing, proof encoding/format, transcript/Fiat–Shamir or protocol changes; non-minimal affecting or uncertain shared-code changes; otherwise unclassified changes | A current-head maintainer approval |

**Prover-only changes are risk-2 in any crate.** The classifier must trace the changed items
through callers in both revisions, including shared helpers, trait implementations, generated
code and feature gates, and cite why no changed behavior reaches verifier logic, verifier
preprocessing, proof (de)serialization or format, transcript/Fiat–Shamir operations, sumcheck
order/expressions or PCS/commitment parameters. The policy's path lists are hints, not production
risk floors for traced prover-only code. Traced prover-only code is level 2, even when
another file makes the overall verdict `AFFECTS`; its size does not raise it to risk-3 or risk-4.
The gate trusts the classifier's reachability evidence, not a second deterministic call-graph
analysis. Direct changes to the acceptance surfaces above stay risk-4, including in prover code.
Unresolved reachability is risk-4 unless the bounded shared-code exception below applies outside
the risk-4 path hints. On risk-4-hint paths, model level 3 becomes 4; level 2 stays 2. That
exception does not apply to direct verifier, transcript, protocol or wire-format edits.

A level-1 answer still cannot waive level-3/4 runtime-code defaults, subject to the accepted
scanner limits below. The gate compares both complete source files after excluding whole comments,
blank lines and syntactically recognized top-level Rust items compiled only under `cfg(test)`
(or a file-inner `#![cfg(test)]`), including stacked multiline attributes and ordinary
external modules whose declarations are all test-only. Nested cfg-like tokens are not treated as test code, since
they can be macro input rather than Rust attributes. This also
detects runtime code exposed by removing a cfg or block-comment delimiter. A test-only or
comment/blank-only change may go below the default. `cfg(all(test, ...))` implies test-only;
`cfg(any(test, feature = "prover-fixtures"))` does not. Standalone `#[test]` or custom test
attributes alone do not qualify for this exception. Inline comment edits on lines containing
code are conservatively counted as code changes.

For paths at effective risk-3, `AFFECTS` or `UNCERTAIN` can stay risk-3 only if the change is minimal.
The effective level is the classifier's rating, except for deterministic dependency bumps,
the scanner guard on level-1 answers and level 3 becoming 4 on risk-4-hint paths.
The bounded shared-code exception applies only outside the risk-4 hints. This count includes paths outside the risk-3 hints
that the classifier rates 3. Pika's
chosen threshold is at most **20 added plus removed non-test, non-comment, non-blank lines**
across all effective level-3 paths, configured by `risk-3.max_changed_lines` in the risk JSON. For example,
replacing ten code lines counts as twenty changed lines. For non-Rust files, the count
conservatively includes every non-blank line: comment-looking lines can instead be string or
data contents. This is Pika's conservative adjustment to the proposed code-line count.
The gate compares the filtered source
lines in both revisions, excluding lockfile lines and dependency-only files handled below.
Above that threshold, either verdict becomes risk-4. A `SAFE` verdict does not trigger this
size escalation. The classifier can still raise any path's risk for semantic effects elsewhere.

Dependency bumps are deterministic: a `Cargo.toml` edit must change only dependency versions,
git revs or tags; a `Cargo.lock` edit must change only package entries. Adding a dependency,
changing features or changing workspace settings is not a bump and requires semantic classification;
the level-3 hint and minimality rule apply unless the classifier establishes another level.
Bumps get risk-3 if any changed package belongs to the verifier dependency
closure, and risk-2 otherwise. The roots are the 20 runtime crates from the October 6 addendum
in `specs/verifier-closure-lints.md`, stored in `risk-3.dependency_roots`. The gate follows every
Cargo.lock dependency edge from those roots in **both** revisions. It matches package names
across versions and includes dev/build edges, which can conservatively assign risk-3. Bump-only
pull requests need no model call; in mixed changes, the classifier cannot escalate bump paths.

Every changed path must appear exactly once in a model result, including removed files and
both names of a rename. The pull request takes the highest effective level. An incomplete or
invalid result fails the evaluation and a maintainer-authored pull request receives risk-4
when GitHub is reachable. `UNCERTAIN`
or `AFFECTS` with no non-bump path at risk-3 or above becomes risk-4. The classifier still asks about verifier acceptance;
this does not add a witness-confidentiality check for prover-only code.

Without a cached or new classification, a change wholly within level-1 paths has internal risk-1;
other changes have internal risk-4 until they can be classified. Maintainer-authored drafts
receive the corresponding label without running a model; non-maintainer pull requests receive
no risk label. Internal risk still sets their approval threshold. This path rule cannot earn the
gate's approval: the gate approves only a head that has a classifier result, because a level-1 path such as a script under
`scripts/` can still be a build input. A current-head maintainer approval already satisfies the
highest level, so the gate starts no model work for it and keeps an existing classification or
this fallback.

## Labels the gate manages

Each evaluation sets exactly these labels and leaves every other label alone:

- `risk-1` through `risk-4`: one risk label for a maintainer-authored pull request, as above.
  Non-maintainer pull requests get none; the next evaluation removes any existing risk labels.
- `needs-maintainer-review`: the author is not a maintainer. It stays even when a writer's
  approval satisfies the policy; the gate never approves non-maintainer pull requests. Without
  an approval, these pull requests get only this label and, when the existing rule below applies,
  `no-reviewer-assigned`.
- `merge-eligible`: the gate's review policy is satisfied for the current head, as the table
  below decides, and the pull request is open and not a draft. It does not show that CI passed,
  that GitHub would allow the merge, or that auto-merge is enabled. Closing or merging the pull
  request, or retargeting it away from `main`, removes it.
- `no-reviewer-assigned`: the policy waits for a person (table rows 3, 5, 8 and 11, or a gate
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
| 1 | The pull request is closed, or its base is not `main` | Not evaluated; `merge-eligible` removed | Withdrawn, unless the pull request merged | None |
| 2 | A maintainer or writer has requested changes, on any commit, and has not since approved or had that review dismissed | `neutral`: changes requested | Withdrawn | None |
| 3 | No writer approved and the author is not a maintainer | `neutral`: human approval required | Withdrawn | None |
| 4 | No writer approved and the pull request is a draft | `neutral`: draft | Withdrawn | None |
| 5 | No writer approved and the pull request changes a protected input: `.github/workflows/ai-merge-gate.yml`, `.github/workflows/ai-merge-gate-events.yml`, `.claude/skills/verifier-classifier/`, `specs/verifier-closure-lints.md` or `.github/ai-merge-gate-risk.json` | `neutral`: human approval required | Withdrawn | None |
| 6 | Maintainer-authored, ready, no gate-input changes, no current-head maintainer approval, and no cached classification | Classify dependency-only bumps deterministically; otherwise run the classifier within its attempt limits, then continue | — | Classifier unless bump-only |
| 7 | Risk-4 with an undismissed current-head approval by a maintainer with write access, other than the author | `success` | Withdrawn | None |
| 8 | Risk-4 without that approval | `neutral`: current-head maintainer approval required | Withdrawn | None |
| 9 | Risk-1 through risk-3, and a writer other than the author approved without a later dismissal or change request | `success` | Given if the next row's classification, attestation, maintainer authorship, readiness and gate-input conditions also hold; otherwise withdrawn | None |
| 10 | A classifier result for the head puts it at risk-1 through risk-3, and the Pika App bot or a maintainer posted the exact current-head attestation below | `success` | Given | None |
| 11 | Risk-1 by the path rule alone: the head has no classifier result because its attempts are spent | `neutral`: human approval required | Withdrawn | None |
| 12 | Otherwise | `neutral`: needs an approval or a current-head review attestation | Withdrawn | None |

A policy wait publishes a gray, non-failing `neutral` check with its reason in the title and
summary. Only a satisfied policy publishes `success` and can earn `merge-eligible`; `neutral`
never grants an approval. The evaluator exits successfully for both conclusions, so a human
requirement does not fail the workflow job. Only a gate execution error publishes `failure`
and exits nonzero.

Rows 3 through 5 apply only without a writer's approval. The gate's own approval additionally
requires a maintainer author, a ready pull request and no changes to gate inputs.

Human approvals follow GitHub's own review state for risk-1 through risk-3: an approval of an
earlier commit counts until GitHub dismisses it. Risk-4 is stricter: an earlier-head approval
never counts, even if GitHub has not dismissed it. The AI evidence names the head: a push voids
every attestation and classifier verdict of an earlier head, and the gate's approval
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
supports: a deleted or edited attestation, an author losing write access, a new change request, a reclassification,
a changed maintainer list or risk policy, a head left without a classifier result, any approval
that names an earlier head, and closing the pull request or retargeting it away from `main`. A
merged pull request keeps the approval that let it merge, since it cannot reopen. GitHub
dismisses stale approvals itself when a push changes the diff; a push that leaves the diff
unchanged keeps the approval, which the gate's evaluation of the new head then dismisses.

A pass rests on state read after the label writes: before it approves, the gate reads the pull
request, its comments and reviews, and permissions again and evaluates the table once more,
reusing the model results it already has. When the new state needs a classifier result the run
does not have, for example after an approval was dismissed, it returns `neutral` with "The pull
request changed while the gate ran", and the next event runs the gate again. A head change
during label or approval publication likewise returns `neutral` with "Superseded while the
gate ran". After submitting an approval, the gate reads the head once more. GitHub dismisses stale approvals when a push
arrives, which misses an approval of the old head submitted just after it, so when the head
moved the gate dismisses the approval it just submitted.

An evaluation that fails at any step after the script starts, a failed write of its own check
run included, removes `merge-eligible` and dismisses the gate's standing approvals before it
reports the error. It attempts each of these writes even if another fails. If GitHub refuses
them too, the approval stays until a later evaluation withdraws it, and the failed run shows the
error. Error cleanup also omits risk labels when the author is known not to be a maintainer.
If author authorization cannot be determined, cleanup retains the risk-4 fallback and preserves
any existing `needs-maintainer-review` label. Reviews, dismissals, labels and comments made
with `GITHUB_TOKEN`
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

The number and full SHA must match the PR's current API state. The skill name must contain
non-whitespace text and may contain spaces. The model must be exactly `GPT-6 Astra` or
`Claude Opus 5.5`, and the effort exactly `max` or `xhigh`, in any combination. Other names,
aliases, API model IDs and case variants, such as `Astra`, `gpt-6-astra`, `GPT-6.1 Sol` or
`Max`, do not count; the classifier's model is not a review model. The gate holds these lists
as `REVIEW_MODELS` and `REVIEW_EFFORTS` in `scripts/ci/ai_merge_gate.py`. The line must match
exactly, with no prefix, indentation, quotation marker or suffix. It may appear between other
lines in the comment. `Review: CONFIRMED`, short SHAs and attestations only in review bodies do
not count.

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

## Using the gate

**Maintainer authors.** Mark the pull request ready for review. The classifier then runs once
for each new head and records its verdict as the `ai-merge-gate/classifier` check, with its
reasoning in a pull request comment. Run the review outside CI on that exact head, using
`.claude/skills/ci-code-review/SKILL.md` or the chosen review skill. Once it finds no issues, an
allowed account posts the attestation with the actual skill, model and effort, which must be a
review model and effort listed above, and at risk-1 through risk-3 the gate approves. The
`ai-review` label has no effect and starts no review.

**Other authors.** The gate spends nothing and gives no approval on a pull request from someone
who is not a maintainer. A writer's approval can satisfy GitHub's review requirement.

**Verifier and gate changes.** Risk-3 can include a verifier-reaching dependency bump or a minimal
shared-code change and still gets the gate's approval on the maintainer-authored path. At risk-4 and for changes to the gate's
inputs, the gate never approves; a person's approval is needed, and the policy asks for a
current-head maintainer approval at risk-4.

**Re-running the gate.** Creating, editing or deleting a personal-account or Pika-bot PR comment
re-runs the gate, including an edit that removes the line. Comments from other bots are ignored.
Pushes, closing and reopening, ready and draft changes, edits (a new base branch included),
review requests, assignments and reviews re-run it through the events workflow. You can also
re-run the `AI merge gate` workflow run in the Actions tab. The Re-run button on the gate's own
check runs does nothing, because no server receives that request.

## How the gate is built

The gate never runs code from a pull request. Two workflows divide the work:

- `ai-merge-gate-events.yml` runs on `pull_request` and `pull_request_review` events. GitHub runs
  the pull request's own copy of this file, without secrets and with a token that has no
  permissions. The file does nothing except finish, which tells the gate that something changed,
  and its run name, `AI merge gate events for PR #<number>`, proposes which pull request changed.
  For every run, the resolver reads the named PR and requires the same base repository,
  source repository, branch and exact head as the run. It does not use the run's PR association
  list, which can omit closed PRs. A source mismatch starts no evaluation and changes no PR
  state. A PR can suppress its own events by editing this file, but cannot use its run name alone
  to start another PR's evaluation.
- `ai-merge-gate.yml` runs on `issue_comment` and on `workflow_run` after the events workflow.
  GitHub always takes both of these workflows from the default branch. Its `resolve` job takes the
  pull request number from the comment event or validates the events workflow's proposed number
  as above, so a closed or retargeted pull request is itself evaluated and loses the gate's
  approval and `merge-eligible`, whatever other pull requests its branch heads. Its `evaluate` job checks out
  the gate script, risk policy, classifier skill and closure spec from the default branch, reads
  the pull request, its comments and its reviews through the API, calls the classifier when row 6
  applies, and writes check runs, labels, the classifier comment and its review with the job's
  `GITHUB_TOKEN`. One evaluation per pull request runs at a time.

GitHub documents the default-branch behavior and the `created`, `edited` and `deleted` comment
events in [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment).
The [workflow-run event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
can receive secrets even after an unprivileged workflow; this gate consumes no artifacts or
code from that run.

The `evaluate` job's token has Actions read, Checks write, Contents read and Pull requests write;
the `resolve` job's has Contents read and Pull requests read to bind every events run to the
source repository, branch and exact PR head, including same-repository and closed pull requests.
Pull requests write covers
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

The reusable `.github/workflows/ai-merge-gate-evaluate.yml` separates three jobs:

- **Prepare:** trusted default-branch code reads GitHub's complete tree listings and archives,
  verifies every blob, and exports base/head snapshots, `diff.patch`, and a changed-path manifest.
  It records the paid attempt before handing off. No PR code, hooks, filters or build scripts run.
- **Classify:** a fresh Ubuntu runner has Contents read only and no GitHub write credential.
  Every checkout uses `persist-credentials: false`. Only this job receives `OPENAI_API_KEY`.
  Pinned Codex 0.162.0 runs GPT-6.1 Sol at `xhigh` with `codex exec --sandbox read-only`.
  Its ordinary Bash commands have no network access and inherit an explicit non-secret environment.
- **Publish:** another fresh trusted checkout receives only the plan and verdict artifacts, not
  the PR snapshots or model key. It binds the result to the repository, base/head, changed paths
  and trusted-input fingerprint, validates the verdict schema and lengths, applies the existing
  risk policy, then publishes with Checks write and Pull requests write.

No diff or complete changed file is inlined in the first model request. The agent starts with
`git diff --no-index --no-ext-diff --no-textconv --stat -- base head` and triages every changed
path. Test-only, documentation and fixture hunks get cheap path/cfg checks; production hunks
that could reach verifier acceptance, preprocessing, transcript, proof format or PCS parameters
get caller tracing in both revisions. Total diff size does not trigger a skip. The agent reads
with Git, `rg` and `sed` ranges; the old custom read/search tool loop is gone.

The preparation job checks archive contents against tree blob hashes: `export-ignore` and
`export-subst` cannot hide or rewrite files. An incomplete diff, truncated tree listing or archive
mismatch fails closed before payment. A snapshot over 500,000,000 bytes, agent failure, timeout,
invalid verdict or exhausted token/spend cap leaves the head unclassified at neutral risk-4;
a current-head maintainer approval is required. Nothing is silently truncated. Binary files are
marked as unreadable data for semantic classification rather than interpreted as source.

The model API key stays in a separate non-dumpable credential/budget broker process. Codex
receives only a placeholder key and talks to that broker; the real key is never written to
Codex auth, config or logs. Native sandbox PID isolation and environment scrubbing prevent
commands from reading the broker's or runner ancestors' environment. The shell has no network,
including access to the local broker. `project_doc_max_bytes = 0`, a private `CODEX_HOME`, and
disabled hooks, plugins, apps and web search keep PR-local agent instructions from becoming
trusted configuration. Exported files are non-executable data, `.git` path components are
rejected, system/global Git config is disabled, and hooks/fsmonitor are disabled. A trusted `RIPGREP_CONFIG_PATH` supplies `--no-ignore --hidden`, so PR-controlled
`.ignore` and `.rgignore` files cannot hide source. Read diffs
with `--no-ext-diff --no-textconv`; never invoke filters or PR Git configuration.

Never build or execute PR code: no Cargo, rustc, make, build.rs, scripts, hooks or direnv.
A small native Codex execution rule forbids common build tools. This rule and the skill are
not a general execution-proof boundary: ordinary Bash can interpret source text. The enforced
security boundary is read-only filesystem access, no network and no readable real credentials;
PR text can still attempt to mislead a verdict. The publisher treats every verdict field as
untrusted text, caps lengths, escapes HTML/Markdown and suppresses mentions before publication.

Local classification previews must stub and count all GitHub writes; they are not permission
to publish or approve anything.

Row 5 protects the classifier skill, closure spec, risk policy, both gate entry workflows,
the new reusable evaluation workflow and `scripts/ci/classifier_agent.py`. Proposed replacements
are untrusted input, not instructions for the current run. Other `scripts/ci/` files, workflows,
`CLAUDE.md`, `AGENTS.md`, this guide and review skills still go through classification with
their existing risk globs. A qualifying human approval still satisfies policy; the gate never
adds its own approval for protected-input changes. The independent `diff-classifier.yml` is
unchanged and does not receive the model key.

The gate stores each classifier verdict and risk level as a check run on the head, whose
external ID names the workflow run that created it. Later evaluations reuse a verdict instead of
paying for a new one only when the check run comes from the GitHub Actions app (ID 15368) and
its external ID names a run of `.github/workflows/ai-merge-gate.yml` started by `issue_comment`
or `workflow_run` in this repository: GitHub runs that file for those events only from the
default branch. This validates the claimed run, not who created the check. A workflow with
Checks write can use `GITHUB_TOKEN` to forge `external_id` and `completed_at`, including a
far-future completion time that wins the cache lookup. The gate trusts those fields and the
publicly computable inputs fingerprint below. This is a known limit; there is no HMAC or
dedicated App binding.
Cached results must also carry risk metadata and a SHA-256 fingerprint of the trusted classifier
inputs: the gate script, agent runner, diff-parser helper, gate and evaluation workflows, classifier
skill, closure spec and risk policy. The conclusion must be successful or neutral. Legacy, failed
or mismatched-input results are classified again when authorized and attempts remain. Changing
these inputs does not reset the paid-attempt ledger. A failed model call is not stored, so the
next event may retry it within the attempt limits. Attestations are always read from current comments, never from check
runs.

The gate does not use `pull_request_target`. GitHub's default Actions policy for public
repositories blocks that event from December 2, 2026 unless the repository allows it, and the
gate does not need it.

## Classifier spend limits

The credential broker in `scripts/ci/classifier_agent.py` applies these caps before forwarding
paid model requests. Codex, not the broker, owns the agent/tool loop.

- **Tokens:** OpenAI's `/v1/responses/input_tokens` endpoint counts each complete proposed
  request. At most 150,000 input and 20,000 output tokens per request; at most 750,000 cumulative
  input and 100,000 cumulative output tokens per attempt. Reasoning tokens count as output.
  Input includes repeated conversation context; cached input remains part of the token cap.
- **Spend and turns:** at most 32 model requests and $2.875 per attempt. Before each call the
  broker reserves its counted input at $2.50/M plus maximum output at $10/M, in addition to
  observed cost so far. It forces standard tier. Rates used for receipts are $2/M fresh input,
  $0.10/M cache reads, $2.50/M cache writes and $10/M output. These are list-price calculations,
  not invoices; model pricing changes require updating the broker.
- **Attempts:** unchanged: at most 2 paid attempts per head and 3 per PR, across retries and
  heads. The trusted preparation job records an attempt before the model job starts. Publication
  consumes that reserved attempt, never a second one. One evaluation per PR runs at a time.
  Writers can reset counts by editing/deleting the comment; see [Trust](#trust). Cached verdicts
  cost no attempt. The nominal PR spend cap remains $8.625 while that ledger stands.
- **Time:** at most 40 minutes per agent run, with at most 15 minutes waiting on one API call.
  The owned agent process group is killed on timeout. The workflow job has a 45-minute cap.
  Cancelling a wait does not guarantee cancellation of provider billing; an interrupted receipt
  marks usage incomplete rather than claiming zero cost.

A response that reaches its output cap fails classification. Available usage on incomplete
responses is still recorded. If a response is lost, the reserved request budget bounds that
attempt, but its exact billed usage is unknown. All predictable limits use neutral, unclassified
risk-4 rather than a red gate error or an automatic approval.

### Usage receipts

Each run records wall time and the API's per-request input, cache-read, cache-write and output
counts in a separate receipt. Fresh input is input minus cache reads and cache writes. Codex
manages its own conversation and provider cache key; the gate does not claim a cache hit from
elapsed time. Reusing a validated classifier check on an unchanged head is a separate cache
that makes no model request.

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
GitHub logins separated by commas. A listed login must also retain write access to qualify.
Maintainers can author gate-eligible pull requests, attest reviews, and give the risk-4 approvals
the policy asks for. The admin chooses them; this pull request names
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
project that may call `gpt-6.1-sol` through the Responses API. Give that project its own hard
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
has Pull requests write can then approve. Michael Zhu's November 3, 2026 review records default
read/write token permissions for `rust.yml`, `arch-tests.yml`, `fs-soundness.yml` and
`deploy-mdbook.yml`. All four run on push to `main` without an explicit `permissions`
restriction. They execute repository code,
so the gate is not the only workflow that can approve or create classifier check runs. Workflow
permission lockdown is left for a separate change; this pull request changes none of those
workflows or repository settings.

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
- closing the pull request without merging makes the gate dismiss its approval and remove
  `merge-eligible`;
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

- Crafted `#[cfg(test)]` attributes inside multiline generic lists can make the source scanner
  misread the rest of a runtime function as test code. The scanner is not a Rust parser.
- Submodule gitlinks on floor paths count zero code lines. None exist on those paths today.
- Out-of-file test modules use ordinary Rust module paths and all parent declarations.
  `#[path]`/`include!` targets and Cargo build/target roots keep their normal floor. Literal
  references conservatively block every file with the referenced basename. Custom references
  and macro-generated module declarations also block their declaring crate's exemption.
  The scanner does not expand macros or resolve generated module paths.
- The configured dependency roots are a hand copy of `specs/verifier-closure-lints.md`.
  Drift changes risk-2 versus risk-3 on dependency-bump paths; both levels are gate-eligible.
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
  classifier results. An events run is ignored if its head has since changed or its source
  repository is unavailable; a later event or a PR comment can evaluate the new
  state. GitHub also refuses to merge a pull request while another open pull request
  whose head branch points to the same commit has pending or rejected reviews.
- The model reads the source but does not build or run it, and it reads only text files up to
  1,000,000 bytes. Before a paid call, the gate counts code changes on floor-3/4 paths except
  dependency-only files and lockfiles. An unreadable file in that count fails closed without
  admitting a paid attempt and requires a person. Lower-floor paths are counted only if the
  classifier raises them to effective risk-3; binary docs and fixtures keep their prior treatment.
  An unreadable lower-floor file rated 3 fails closed after the paid call, even for `SAFE`;
  it can consume up to two paid attempts per head.
- The size limits bound what the model receives and what the gate keeps, not the evaluation's
  peak memory: the gate downloads a commit's whole archive before checking its files, and builds
  a search's matches before refusing more than 200 of them. A very large tree can exhaust the
  runner, which fails the run without an approval.
- In the two runs under [Prompt caching](#prompt-caching), no `xhigh` request generated more
  than 729 output tokens, reasoning included; larger or harder pull requests may need far more.
  A response that reaches the 20,000-token limit fails its attempt, and if that happens often,
  the gate falls back to human review until the limit is raised.
- GitHub has no head-conditional label transaction. A push between a label API read and write
  can briefly leave the old head's labels, and labels stay as they were while an evaluation runs.
  Closing a pull request removes only `merge-eligible`. Label permissions, availability and API
  failures can prevent a label update; the evaluation then fails and approves nothing.
- An author losing maintainer status while the classifier runs stops the approval, not classifier
  work already under way.
- A verdict covers the diff from the head's merge base with `main`. A later change to `main` does
  not void it, just as it does not void a human approval when branches need not be up to date.
- Comment authorship and last-editor identity come from GitHub. Both must be allowed accounts
  for an edited attestation. The gate does not establish who performed the review.

This pull request activates nothing. Focused fixtures exercise API state, the emitted reviews,
dismissals, labels and check conclusions. The classifier's OpenAI calls ran locally on one
public pull request, as [Prompt caching](#prompt-caching) describes; the classifier has not run
in CI. Live webhook delivery, the Actions approval setting, dismissal permission, branch rules
and classifier accuracy remain unverified until the admin activation checks above.
