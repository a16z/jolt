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
It does not produce a separate semantic verdict.

Classifier results are cached on the exact head with a fingerprint of the gate, agent runner,
diff parser, gate/evaluation workflows and skill. Attestations and approvals are reread; labels
and gate approval are reconciled after classification. Drafts run no classifier and clear
managed state. Closing removes merge eligibility; merging preserves the historical approval.

## Limits and trust

The broker retains its request, token, spend and wall-time limits in `classifier_agent.py`.
The gate allows two paid attempts per head and three per PR. Preparation reserves the attempt;
publication consumes that reservation without counting it again. Failed attempts are not cached.
Verified snapshots are bounded to 500,000,000 bytes per revision. These are resource limits,
not risk-level rules.

Classification now relies on the model alone. A change to the gate or its instructions can
receive risk 1–3 and auto-approval: it cannot alter its own judge on that run, but changes
future runs after merging. Opaque external dependency updates also rely on the model's judgment.

Writers remain trusted: workflows with write tokens can forge cached check metadata or submit
approvals, and writers can edit/delete the attempt comment to reset counts. Cache validation
checks the claimed workflow run and public fingerprint, not a cryptographic publisher identity.
Event delays and changes between API reads/writes can leave approval or labels stale until the
next evaluation. A default-branch change does not invalidate an old head's result until its
next evaluation. No live webhook, organization protection or dismissal behavior is proven by
local fixtures.

## Installation

An admin must retain existing required checks, set `AI_MERGE_GATE_MAINTAINERS`, provide the
`OPENAI_API_KEY` repository secret, and create the labels `risk-1` through `risk-4`,
`needs-maintainer-review`, `merge-eligible` and `no-reviewer-assigned`. The gate never creates
label definitions. A missing or malformed maintainer list fails evaluation.

Allow GitHub Actions to create and approve pull requests, and require one approval with stale
approvals dismissed on push. More than one required approval, code-owner review or last-push
approval can require another reviewer. Verify the organization's approval/dismissal permissions
and branch rules on a test PR: attestation approves an eligible head; deleting it withdraws
approval; a push needs a new attestation; risk 4 receives no gate approval. Disable the workflow
and its approval permission if it cannot withdraw its own approval. Existing approvals must be
dismissed or invalidated separately when turning the gate off.

The standalone `diff-classifier.yml` PR-comment tool is independent of this gate.
