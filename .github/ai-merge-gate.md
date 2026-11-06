# AI merge gate

The gate classifies a ready pull request into the default branch, publishes its risk label
and `ai-merge-gate` check, and manages its own approving review. It changes no branch rules.
Risk meanings and judging instructions live in
[verifier-classifier](../.claude/skills/verifier-classifier/SKILL.md).

## Approval policy

A maintainer is listed in `AI_MERGE_GATE_MAINTAINERS` and currently has write or admin access.
Only maintainer-authored pull requests run the classifier. A classification must account for
all changed paths exactly once; the PR's risk is the highest returned level.

| Condition, evaluated in order | Gate check | Gate approval |
| --- | --- | --- |
| Draft, closed or not targeting the default branch | Inactive | Withdrawn unless merged |
| A writer has an undismissed change request | Neutral | Withdrawn |
| Risk 4 or unclassified | Success only with a current-head maintainer approval from someone other than the author | Never supplied |
| Risk 1–3 and a writer other than the author approved | Success | Supplied only if also attested |
| Risk 1–3 and an authorized current-head review attestation | Success | Supplied for that exact head |
| Otherwise | Neutral | Withdrawn |

An unclassified head needs an independent current-head maintainer approval, including when
its author is not a maintainer. No risk label is assigned without a classifier result.
Failure, timeout, an invalid answer or exhausted resource/attempt limits leaves it unclassified.
Input integrity failures produce a failed gate check and no gate approval.

## Review attestation

The Pika App bot (`BOT_kgDOFAtqNQ`) or a maintainer must post this standalone PR-comment line:

```text
This pull request #<number> has been reviewed at commit <40-character lowercase head SHA> using skill <skill name> and model <model name> with <effort> reasoning effort, and no issues were found.
```

The model must be `GPT-6 Astra` or `Claude Opus 5.5`; effort must be `max` or `xhigh`.
For edited comments, both the author and last editor must qualify. Review bodies and labels
cannot replace the comment. A new head needs a new attestation.

## Trusted execution

The workflows run the gate and skill from the default branch, never the PR's proposed copy.
The preparation job reads pinned merge-base/head trees, verifies archive blobs against Git
hashes, checks diff counts and records an attempt before exporting read-only snapshots.
The classifier job runs Codex with no GitHub credential. The real model key remains in a
separate non-dumpable broker; Codex receives a placeholder key. The native sandbox prevents
filesystem writes and network access. Private agent configuration disables PR-local instructions,
hooks, plugins and apps; trusted ripgrep settings include ignored and hidden source files.
Never executing PR code is also an instruction: common build tools are denied, but ordinary
shell commands can interpret source text, so this is not a general execution-proof boundary.

The publisher receives only the plan and result, binds them to the repository, revisions,
changed paths and trusted-input fingerprint, validates the JSON, and escapes published text.
The classifier's JSON contains overall `reasoning` and per-path `risks` (path, level, reasoning).

Classifier results are cached on the exact head with a fingerprint of the gate, agent runner,
diff parser, gate/evaluation workflows and skill. Attestations and approvals are reread; labels
and gate approval are reconciled after classification. Drafts run no classifier and clear
managed state. Closing removes merge eligibility; merging preserves the historical approval.

An independent refresh job clears all gate-managed labels when the current head has no
`ai-merge-gate` check from a trusted gate workflow run, in any status. These labels are `risk-1`
through `risk-4`, `merge-eligible`, `needs-maintainer-review` and `no-reviewer-assigned`.
Refresh does not wait behind classification or create checks; its failure does not block evaluation.
It runs on `pull_request_target` opened, reopened and synchronize events, independently of the
evaluation bridge, and executes only default-branch code, never pull-request bytes.
Unrelated labels are left alone.

The separate `merge-conflict` label follows GitHub's `mergeable` state for open PRs into the
default branch, including drafts: false adds it; true removes it. Refresh reads every target PR
before retrying only unknown states, two seconds apart, within a shared 60-second window.
States still unknown keep their labels and are logged. Each PR refresh checks conflicts, and a
push to `main` refreshes all gate labels and conflicts for open PRs into the default branch.
Gate evaluation and draft cleanup never change `merge-conflict`.

## Limits and trust

Each classifier attempt has a $50 model-spend cap and a five-hour wall-time cap;
the classify job allows 305 minutes for setup, classification and artifact upload.
The broker reserves the model's maximum output before each request and accounts for
[standard and long-context token prices](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
There are no cumulative input-token, output-token or request-count caps.

Codex uses its own auto-compaction through the same budgeted Responses broker.
The configured API context window is 1,050,000 tokens, with up to 128,000 output tokens;
the broker reserves that output space, allowing up to 922,000 input tokens.
Pinned Codex 0.162.0 [clamps the configured window to its model catalog's maximum](https://github.com/openai/codex/blob/rust-v0.162.0/codex-rs/models-manager/src/model_info.rs),
872,000 tokens for this model, and [auto-compacts at 90%](https://github.com/openai/codex/blob/rust-v0.162.0/codex-rs/protocol/src/openai_models.rs)
(784,800 tokens). Compaction requests use the same model checks and spend accounting.

The gate allows two paid attempts per head and three per PR. Preparation reserves the attempt;
publication consumes that reservation without counting it again. Failed attempts are not cached.
Verified snapshots are bounded to 500,000,000 bytes per revision. These are resource limits,
not risk-level rules.

Classification relies on the model alone. A change to the gate or its instructions can
receive risk 1–3 and auto-approval: it cannot alter its own judge on that run, but changes
future runs after merging. Opaque external dependency updates also rely on the model's judgment.

Writers remain trusted: workflows with write tokens can forge cached check metadata or submit
approvals, and writers can edit/delete the attempt comment to reset counts. Cache validation
checks the claimed workflow run and public fingerprint, not a cryptographic publisher identity.
Event delivery and runner startup can delay label refresh; changes between API reads/writes
can race with publication. GitHub dismisses stale approvals on push when configured below;
same-head review changes require another evaluation to withdraw gate approval. A default-branch
change refreshes labels and conflicts but does not invalidate an old head's classification or approval
result until its next evaluation. No live webhook, organization protection or dismissal behavior
is proven by local fixtures.

## Installation

An admin must retain existing required checks, set `AI_MERGE_GATE_MAINTAINERS`, provide the
`OPENAI_API_KEY` repository secret, and create the labels `risk-1` through `risk-4`,
`needs-maintainer-review`, `merge-eligible`, `no-reviewer-assigned` and `merge-conflict`.
The gate never creates label definitions. A missing or malformed maintainer list fails evaluation.

The repository's Actions event policy must allow `pull_request_target`, as `pr-title.yml` and
`claude-routines.yml` already require. GitHub's [default public-repository policy blocks that event
starting December 2, 2026](https://github.blog/changelog/2026-11-02-pull_request_target-blocked-by-default-december-2/).
If blocked, per-PR refresh does not run; the independent refresh path runs only on pushes to
`main`. The refresh job checks out only the default branch and executes no pull-request bytes.

Allow GitHub Actions to create and approve pull requests, and require one approval with stale
approvals dismissed on push. More than one required approval, code-owner review or last-push
approval can require another reviewer. Verify the organization's approval/dismissal permissions
and branch rules on a test PR: attestation approves an eligible head; deleting it withdraws
approval; a push needs a new attestation; risk 4 receives no gate approval. Disable the workflow
and its approval permission if it cannot withdraw its own approval. Existing approvals must be
dismissed or invalidated separately when turning the gate off.

The standalone `diff-classifier.yml` PR-comment tool is independent of this gate.
