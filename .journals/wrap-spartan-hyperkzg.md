# wrap/spartan-hyperkzg — Spartan+HyperKZG proof-size wrapper for curve Jolt (card 135)

Worktree: /Volumes/Dev/worktrees/jolt/wrap-spartan-hyperkzg (base origin/main 756bddce3)
Private mirror: 0xAndoroid/jolt-private branch wrap/spartan-hyperkzg
Orchestrator journal — current state + kill-list + index. Completed waves → .journals/archive/.

## Spec (user, verbatim)
"same way the blindfold works, it encodes the R1CS, and then spartan proves the R1CS. Now then it would
commit using hyperkzg to this thingy. But also in that R1CS you need dory verification. So this is
essentially spartan + hyperkzg wrapper of jolt proof. The important part is to lower the proof size."

## Playbook (verbatim from parent; todo list — skipping a step requires `skip: <reason>`)
- [ ] Repo: ~/dev/jolt (a16z/jolt, origin/main 756bddce3). Work in a worktree: `cd ~/dev/jolt && wt switch wrap/spartan-hyperkzg --create` (path under /Volumes/Dev/worktrees/jolt/; CARGO_TARGET_DIR under /Volumes/Dev/cargo-target — never /tmp). Unpushed WIP may go to the private mirror `0xAndoroid/jolt-private` (branch wrap/*). Dev only inside ~/dev / /Volumes/Dev.
- [ ] Phase 0 — discovery (do first, write to the journal): locate "Blindfold" in the codebase/docs/git history/a16z zk repos (it is the existing zero-knowledge / recursion wrapper design the user refers to: R1CS-encode the Jolt verifier, prove with Spartan, commit with HyperKZG). Find the Jolt verifier's structure (sumcheck stages, Dory verification incl. G2/pairing ops, transcript), existing R1CS/constraint tooling in the repo (jolt-core r1cs, any Spartan integration, HyperKZG PCS impl), and proof-size accounting (current Jolt+Dory proof size in bytes at 2^20–2^24). Decide the minimal architecture: the wrapper circuit = Jolt verifier (all sumchecks + Dory opening verification) as R1CS; Spartan proves it; HyperKZG commits Spartan's witness polynomials; output = Spartan+HyperKZG proof (target: single-digit KB; report exact bytes). Write a plan file with the component ladder (verifier-as-R1CS pieces in dependency order; non-native field arithmetic for Dory's G2/pairing in R1CS is the hard part — evaluate whether Dory's pairing check can be deferred/folded or must be in-circuit; options: prove over BN254 scalar field with the pairing check as a public deferred check vs full in-circuit).
- [ ] Phases 1..n — implement in lanes with a planner→implementer→reviewer loop per component. Executor rules: planning/frontend/whodunit → Claude fable-max (`agent:"claude"`, `model:"fable-max"`) — ALL Jolt code work uses fable-max; well-specified kernel/impl lanes may use `agent:"codex"`, `model:"gpt-5.6-sol-xhigh"`; scoped analysis/reviews may use `kimi-k3` (keep scoped, verify claims). Model strengths: fable-max best at open-ended root-causing/integration; sol-xhigh best engineering discipline per token on hard well-specified impl; kimi-k3 scoped attribution/reviews, drifts on long sessions. Reviews: fix-then-fresh-review loop until a reviewer finds zero issues; never let a reviewer rubber-stamp its own fixes. Complexity budget in every prompt: minimal diff, no speculative abstractions, no narrating comments, no backwards-compat shims (repo rules permitting). Gates: cargo clippy -D warnings, tests, e2e: wrap a real Jolt proof (e.g. the sha2 or fibonacci example at 2^18+) and verify the wrapped proof; report proof sizes before/after in bytes and prover time added.
- [ ] Deliverable: a DRAFT PR to a16z/jolt (`gh pr create --draft`, ALL Jolt PRs open as drafts; PR body: architecture, proof-size table, security argument, what's deferred) — do not mark ready.
- [ ] Journal: .journals/wrap-spartan-hyperkzg.md in the worktree — carry this playbook VERBATIM as a todo list; skipping a step requires an explicit `skip: <reason>` line. Keep the journal lean (current state + kill-list + index; archive completed waves to .journals/archive/). Use `context_window` discipline: you are an orchestrator — keep detail in the journal, not your context.
- [ ] Kanban: card 135 is yours — `pika-cli kanban update 135 --add-link "<pr-url>|PR #N"` when the draft PR exists; move to human_review then.
- [ ] Teardown: at the end verify no stray processes (`pgrep -fl cargo|jolt`).
- [ ] Report to parent via message_parent only at real ends: draft PR opened (URL, proof-size table, what's deferred), or a genuine blocker/decision (e.g. Dory pairing check cannot be done in-circuit at acceptable cost → present options with numbers). No interim status.

## Current state
- 2026-09-02 14:30 worktree created @756bddce3; Phase 0 discovery lanes dispatched (see index).

## Kill-list (open questions / risks)
- Dory verification in R1CS over Fr = non-native Fq/Fq2/Fq12 + pairings. Need op counts + constraint estimates before architecture decision.
- Jolt transcript is Blake2b/Keccak — in-circuit hashing of thousands of absorbs may dominate; check for an algebraic (Poseidon) transcript option.
- HyperKZG: exists in legacy monolith? in modular crates? needs SRS (KZG) — where from.

## Index
- .journals/discovery/blindfold-and-r1cs-tooling.md — Blindfold internals, R1CS builder, Spartan impl, HyperKZG impl (lane A)
- .journals/discovery/verifier-structure.md — Jolt verifier stages, Dory verifier ops, transcript (lane B)
- .journals/discovery/proof-size.md — measured proof sizes 2^18–2^24, byte breakdown (lane C)
- .journals/discovery/prior-art.md — Blindfold paper, external in-circuit pairing costs, Jolt recursion prior art (lane D)
- .journals/plan.md — architecture decision + component ladder (after Phase 0)
