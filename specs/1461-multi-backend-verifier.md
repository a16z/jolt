# Spec: Multi-Backend Verifier (FieldBackend + Tracing AST)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @quangvdao                     |
| Created     | 2026-04-22                     |
| Status      | proposed                       |
| PR          | #1461                          |

## Summary

The new `VerifierOp` interpreter on `refactor/crates` is hard-coded to native
field arithmetic. Every consumer that needs a different evaluation strategy,
recursion / aggregation (proving the verifier inside another SNARK), Lean /
theorem-prover export of the verifier, or symbolic analysis of Fiat-Shamir
flow, would otherwise need its own parallel verifier and a way to keep it
in lockstep with the production one. This spec introduces a single
`FieldBackend` trait that the verifier is parameterised over, with two
concrete implementations: `Native` (zero-cost, the production path) and
`Tracing` (records every operation into an explicit AST). One verifier
codebase, two backends, by-construction equivalence.

## Intent

### Goal

Make the entire `VerifierOp` interpreter generic over a `FieldBackend` trait
that abstracts every field operation, every transcript event (init / absorb /
squeeze), and every equality assertion the verifier performs, so that swapping
the backend changes the execution semantics (concrete vs symbolic) without
changing the verifier code.

Surface introduced by this PR:

- `crates/jolt-verifier-backend` — the trait + two impls + AST + replay + viz.
  - `FieldBackend` (trait), `Native<F>`, `Tracing<F>`.
  - `AstGraph` (DAG of `AstOp` nodes + `AstAssertion` list).
  - `replay_trace(graph, wraps) -> Vec<F>` — symbolic re-execution.
  - DOT / Mermaid emitters for human inspection.
- `_with_backend` mirrors of the verifier's evaluation surface, sitting next
  to the legacy native paths so the cutover is purely additive.

A follow-up PR introduces `GroupBackend` (curve / commitment ops) and cuts the
`VerifierOp` interpreter over to the backend-aware functions, deleting the
legacy paths.

### Invariants

1. **Native parity.** For every input, `*_with_backend::<Native<F>>` returns a
   field element bit-identical to the legacy native function. The
   `cargo nextest run -p jolt-equivalence modular_self_verify` suite is the
   end-to-end witness: `modular_self_verify` (legacy path) and
   `modular_self_verify_via_tracing_backend` (Tracing + replay) must both
   pass over the same proof.
2. **Replay faithfulness.** For every `Tracing` execution that produces a
   graph `G` and a wrap-value vector `W`, `replay_trace(G, W)` returns a
   value vector of length `G.node_count()` and discharges every entry of
   `G.assertion_count()` without error.
3. **Wrap accounting.** `G.nodes.iter().filter(Wrap).count() == W.len()`. No
   orphan `Wrap` nodes; no extra wraps in `W`.
4. **Transcript provenance.** Every `transcript.challenge()` the verifier
   makes appears in the AST as a `TranscriptChallengeValue` node whose
   `state` ancestor chain leads back to a single `TranscriptInit` and a
   sequence of `TranscriptAbsorbBytes` matching what the native verifier
   absorbed, in order. Bytes appended during tracing are byte-identical to
   bytes appended in the native path (verified indirectly: the Blake2b
   instance inside `TracingTranscript` produces the same challenges as a
   Blake2b instance the native verifier would have used).
5. **Sumcheck Fiat-Shamir.** `SumcheckVerifier::verify_with_backend` appends
   the same byte sequence to the transcript that
   `SumcheckVerifier::verify` would, including degree byte and label.
6. **No legacy regressions.** The legacy verifier path stays callable and
   passes its full test suite; this PR is additive.

`jolt-eval` framework: not applicable on `refactor/crates` (the directory
does not exist on this branch). Once `refactor/crates` rebases onto main,
invariants 1–5 are candidates for `/new-invariant`. In particular,
"Tracing(G, W) replays to the same value as Native" is a clean mechanical
equivalence check that the optimizer can run on every change to either
backend.

### Non-Goals

1. **`GroupBackend` / PCS.** Polynomial commitment and group operations stay
   on raw curve types in this PR. PCS calls still take `&mut B::Transcript`
   so transcript ops are tracked, but the elliptic-curve arithmetic itself
   is not yet symbolic. Phase 2.
2. **`VerifierOp` interpreter cutover.** The new `_with_backend` functions
   sit next to the legacy ones with `#[allow(dead_code)]`. Cutting the
   interpreter over and deleting the legacy paths is Phase 3.
3. **Performance optimisation.** Goal here is correctness + a stable trait
   surface. Both backends are already fast enough for testing; the prover
   is unaffected since this is verifier-only work.
4. **ZK feature coverage (`--features host,zk`).** Not exercised yet because
   `refactor/crates` doesn't yet include the BlindFold protocol changes
   from main. Once it does, ZK-mode parity becomes invariant 7.
5. **Recursion or Lean export.** This PR builds the substrate; consumers
   come later.

## Evaluation

### Acceptance Criteria

- [x] `crates/jolt-verifier-backend` compiles standalone with
      `cargo clippy -p jolt-verifier-backend --all-targets -- -D warnings` clean.
- [x] `FieldBackend` trait covers every field op the verifier uses
      (`add`, `sub`, `mul`, `neg`, `square`, `inverse`, `wrap_*`, `const_*`,
      `assert_eq`, `Transcript`, `new_transcript`, `squeeze`).
- [x] `Native<F>` impl: `Scalar = F`, every method `#[inline]`, no allocation
      beyond what the legacy verifier does.
- [x] `Tracing<F>` impl: every method records the corresponding `AstOp` /
      `AstAssertion`, returns the new `AstNodeId` as `Scalar`.
- [x] `AstOp` covers `Wrap{origin,label}`, `Constant`, `Neg`, `Add`, `Sub`,
      `Mul`, `Square`, `Inverse{a, inv}`, plus the four transcript variants
      (`TranscriptInit`, `TranscriptAbsorbBytes`,
      `TranscriptChallengeState`, `TranscriptChallengeValue`).
- [x] `replay_trace(graph, wraps)` re-executes every node and discharges
      every assertion or returns a structured error pinpointing the failing
      node / assertion.
- [x] `viz::to_dot` and `viz::to_mermaid` render every node kind with
      distinct styling and label every transcript-state edge.
- [x] `evaluate_formula_with_backend` covers all 13 `ClaimFactor` variants;
      `evaluate_preprocessed_poly_with_backend` covers `IoMask`, `RamUnmap`,
      `ValIo`. 11 parity tests in `crates/jolt-verifier/src/verifier.rs`
      assert Native == legacy and Tracing-replay == legacy.
- [x] `SumcheckVerifier::verify_with_backend` returns
      `(final_eval_w, challenges_w, challenges_f)` matching
      `SumcheckVerifier::verify` on the Native path.
- [x] `verify_with_backend` (top-level Jolt verifier) instantiates its
      transcript via `backend.new_transcript()` and squeezes via
      `backend.squeeze()`, with no direct `Blake2bTranscript::new` /
      `transcript.challenge()` calls remaining in the function body.
- [x] `modular_self_verify_via_tracing_backend` (jolt-equivalence) drives
      the muldiv proof through `Tracing<NewFr>`, replays the captured
      graph against the recorded wraps, asserts that the AST contains
      exactly one `TranscriptInit` and a positive number of absorbs and
      squeezes, and writes Mermaid + DOT to `target/ast/`.
- [x] `examples/transcript_ast_dump.rs` produces a graph with every node
      kind for ad-hoc visual inspection.

### Testing Strategy

**Existing tests that must keep passing:**
- `jolt-verifier-backend` unit tests (was 18, now 30+ including the four new
  transcript-op tests).
- `jolt-sumcheck` unit tests (was 25, now 37+ after adding the
  `verify_with_backend` Native-parity and Tracing-replay tests).
- `jolt-verifier::verifier::tests` — 11 backend-parity tests across
  `evaluate_formula` and the three `evaluate_preprocessed_poly` variants.
- `jolt-equivalence::muldiv::modular_self_verify` (legacy path) and
  `modular_self_verify_commit_skip_alignment`.

**New tests this PR adds:**
- `jolt-verifier-backend::tracing::tests::transcript_ops_round_trip_through_replay`
  — synthesises a transcript-only graph and asserts that replay reproduces
  the squeezed challenges.
- `jolt-verifier-backend::tracing::tests::backend_squeeze_links_transcript_value_into_arithmetic`
  — asserts that a squeezed challenge is a usable AST node (e.g. an `Add`
  consuming it produces a well-formed `AstOp::Add(challenge_id, _)`).
- `jolt-equivalence::muldiv::modular_self_verify_via_tracing_backend` —
  the end-to-end witness for invariants 1, 2, 3, 4 over the muldiv proof.

**Feature coverage:** `--features host` only on this PR. ZK coverage is
explicitly deferred (see Non-Goal 4).

### Performance

- **Native path:** zero-cost over the legacy verifier. `Scalar = F`,
  every backend method is `#[inline]`, and the verifier does not branch on
  the backend type. Empirical check: `modular_self_verify` runtime delta
  before/after this PR should be inside noise (< 5%). No benchmark target
  beyond "no regression" since the verifier is already milliseconds per
  proof.
- **Tracing path:** acceptable for testing / recursion-prep. Runs over
  muldiv (~4s of which most is proof generation, not verification).
  Allocations dominated by `Vec` growth in `AstGraph.nodes` and
  `Arc<Mutex<...>>` lock acquisition on every record. Not optimised; no
  budget enforced; not on any production hot path.
- **No prover-side work.** This PR is verifier-only.

`jolt-eval/src/objective/`: not applicable on `refactor/crates`. After
rebase on main, no existing objective is expected to move.

## Design

### Architecture

```
┌──────────────────────────┐
│  jolt-verifier::verify   │  generic over <B: FieldBackend>
│  (and verify_with_backend)│
└────────────┬─────────────┘
             │ uses backend.add/mul/squeeze/...
             ▼
   ┌─────────────────────┐
   │   FieldBackend      │  trait
   │  - F, Scalar         │
   │  - Transcript        │
   │  - {add,sub,mul,...} │
   │  - {wrap_public,     │
   │     wrap_proof,      │
   │     wrap_challenge}  │
   │  - new_transcript    │
   │  - squeeze           │
   │  - assert_eq         │
   └─────┬───────────────┘
         │ impl
         ├─────────────────────┐
         ▼                     ▼
  ┌──────────────┐     ┌───────────────────┐
  │  Native<F>   │     │   Tracing<F>      │
  │ Scalar = F   │     │ Scalar = AstNodeId│
  │ everything   │     │ records every op  │
  │ #[inline]    │     │ into AstGraph      │
  │ Transcript = │     │ Transcript =      │
  │ Blake2b<F>   │     │ TracingTranscript │
  └──────────────┘     └─────────┬─────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │   AstGraph         │
                       │   nodes: Vec<AstOp>│
                       │   asserts: Vec<…>  │
                       └─────┬──────────────┘
                             │
            ┌────────────────┼─────────────────┐
            ▼                ▼                 ▼
      replay_trace      to_dot(...)       to_mermaid(...)
      (graph, wraps)    (Graphviz)         (browser-renderable)
       -> Vec<F>
```

**Module map (this PR):**
- `crates/jolt-verifier-backend/src/backend.rs` — `FieldBackend`,
  `ScalarOrigin`.
- `crates/jolt-verifier-backend/src/native.rs` — `Native<F>`.
- `crates/jolt-verifier-backend/src/tracing.rs` — `Tracing<F>`,
  `TracingTranscript<F>`, `AstGraph`, `AstOp`, `AstAssertion`,
  `replay_trace`. Shared state is `Arc<Mutex<AstGraph>>` /
  `Arc<Mutex<Vec<F>>>` because `Transcript: Sync + Send + 'static`.
- `crates/jolt-verifier-backend/src/helpers.rs` — backend-aware
  `eq_eval`, `eq_evals_table`, `lagrange_*`, `lt_mle`, `identity_mle`,
  `sparse_block_eval`, `pow_u64`, `univariate_horner`.
- `crates/jolt-verifier-backend/src/viz.rs` — DOT + Mermaid emitters with
  distinct styling per node kind (and per transcript-state node).
- `crates/jolt-sumcheck/src/verifier.rs` — `verify_with_backend`.
- `crates/jolt-verifier/src/verifier.rs` — `verify_with_backend`,
  `evaluate_formula_with_backend`,
  `evaluate_preprocessed_poly_with_backend`, `eval_io_mle_with_backend`.

**Wrap origins.** Inputs to the AST are tagged with `ScalarOrigin`:
`Public` (verifying-key data, R1CS coefficients), `Proof` (data the prover
sent), `Challenge` (Fiat-Shamir output reified back into arithmetic). This
labelling is what makes the AST useful for downstream consumers: a Lean
exporter can quantify only over `Proof` and `Challenge` inputs; a recursion
verifier can fix `Public` constants at compile time.

### Alternatives Considered

1. **Two parallel verifiers (one native, one symbolic).** Rejected —
   guarantees drift. Every change to the native verifier silently fails to
   propagate to the symbolic one until a downstream consumer breaks.
   `modular_self_verify_via_tracing_backend` over the same proof is the
   canonical witness that the two backends agree, but only if there is
   one verifier source-of-truth.
2. **Macro / build-script codegen of the symbolic verifier from the native
   one.** Rejected — opaque, hard to debug, hard to extend with new
   backends (e.g. constraint synthesis for SNARK recursion).
3. **SNARK-composed recursion only (no AST).** Rejected — the Lean export
   need is real and can't be served by SNARK composition alone. Building
   one AST that serves both keeps the surface unified.
4. **Hide the backend behind associated types of `VerifierOp` itself.**
   Considered — would couple the op enum to the backend trait. Rejected
   because it makes the op enum harder to serialise and harder for
   non-backend consumers (e.g. a future cost model).
5. **Treat transcript challenges as opaque inputs (no transcript ops in
   the AST).** Considered for the first iteration; rejected per the
   project lead's directive to capture the full AST. Without transcript
   nodes, a Lean / recursion consumer would have to re-derive challenges
   externally and trust that the labelling matches the verifier — which
   is the same drift hazard as alternative 1, just one level up.

## Documentation

No `book/` changes in this PR. The user-facing surface (`verify`) is
unchanged; this is internal refactor + a new internal abstraction. Once
the cutover phase lands and the legacy paths are deleted, a new chapter
("Verifier Backends") in the book is warranted, covering:

- The `FieldBackend` (and `GroupBackend`) trait surface.
- How to write a custom backend (e.g. constraint synthesis for a recursive
  SNARK).
- Replay + viz workflows for debugging a verifier proof failure.

## Execution

This PR is split into 5 commits to make incremental review tractable:

1. **`backend: port jolt-verifier-backend crate onto refactor/crates base`**
   — Brings `FieldBackend`, `Native`, `Tracing`, viz, replay, basic
   helpers across from the v1 prototype, dropping references to types
   Markos removed (`jolt_ir::Expr`).
2. **`backend: add Lagrange + preprocessed-poly helpers`** — Adds the
   math primitives the formula evaluator needs, with parity tests against
   `jolt-poly`.
3. **`sumcheck: add SumcheckVerifier::verify_with_backend`** — Backend-
   aware sumcheck verifier; transcript bytes match native exactly.
4. **`verifier: backend-aware evaluate_formula + preprocessed-poly mirrors`**
   — `_with_backend` versions of the formula evaluator and preprocessed-
   polynomial evaluator. 11 parity tests.
5. **`backend: full transcript AST + backend-aware Fiat-Shamir`** —
   Promotes transcript ops to first-class AST nodes; rewires
   `verify_with_backend` and `SumcheckVerifier::verify_with_backend` to
   route every transcript event through the backend.

**Phase 2 (next PR):** `GroupBackend` + cutting `VerifierOp` interpreter
over to the backend-aware functions + deleting legacy paths.

## References

- PR #1461 (this PR) — implementation.
- Markos's `refactor/crates` base commit `c7e0869d5` — the
  `VerifierOp` interpreter we're parameterising over.
- `paper-note/multi-backend-verifier-design.md` — original design sketch
  (in `~/Documents/Notes/`) referenced during the v1 prototype on
  `quang/verifier-backend`.
- Closed Dory PR (`~/Documents/SNARKs/dory`) — prior art for the same
  symbolic-AST-over-native-execution pattern in a single-protocol setting.
