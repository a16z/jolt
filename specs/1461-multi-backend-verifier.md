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

1. **PCS / commitment backend.** Polynomial commitment and group operations
   stay on raw curve types in this PR. PCS calls still take
   `&mut B::Transcript` so transcript ops are tracked, but the verifier's
   interaction with commitments themselves (wrap, absorb, opening check)
   is not yet symbolic. Deferred to Phase 2 (see "Phase 2 Amendment"
   below). The original sketch named this `GroupBackend` and proposed
   lowering Dory's pairing/MSM internals through a curve-shaped trait.
   That direction is **superseded**: Phase 2 introduces a much smaller
   `CommitmentBackend` that keeps the verifier curve-agnostic and
   future-proofs for hash-based and lattice-based PCS.
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

**Phase 2 (next PR):** `CommitmentBackend` (replaces the originally
sketched `GroupBackend`) + cutting `VerifierOp` interpreter over to the
backend-aware functions + deleting legacy paths. See "Phase 2 Amendment"
immediately below.

## Phase 2 Amendment: CommitmentBackend (supersedes "GroupBackend")

| Field       | Value                          |
|-------------|--------------------------------|
| Amended     | 2026-04-22                     |
| Status      | proposed                       |

### Why an amendment

The original Phase 2 sketch (`jolt-verifier-backend-ast-design.md` in
`~/Documents/Notes/`) introduced a `GroupBackend` trait with low-level
primitives (`g1_msm`, `pairing`, `g_combine`) modelled on Dory's verifier
internals. This bakes elliptic-curve structure into the verifier's
public surface: the `AstOp` enum grows curve-shaped opcodes, and any
future PCS that does not factor through pairings or MSMs (FRI / hash-
based, Ajtai / lattice-based; see `~/Documents/Notes/hachi-blindfold-
walkthrough-and-lattice-generalization.md` for the lattice motivation)
either gets shoe-horned into a curve-shaped AST or forces a parallel
backend trait. Both outcomes break the "one verifier source of truth"
invariant Phase 1 was built to protect.

The amendment narrows the Phase 2 trait surface so the verifier is
agnostic to *how* a commitment is verified, and pushes batching /
linear-combination logic into the PCS crate where it belongs.

### Diagnosis: where homomorphism currently leaks into the verifier

`crates/jolt-verifier/src/verifier.rs:50` and `:92` bind
`PCS: AdditivelyHomomorphic<Field = F>` and require
`PCS::Output: HomomorphicCommitment<F>`. Those bounds are not used
directly by the verifier body. They are inherited from a single blanket
implementation:

```rust
// crates/jolt-openings/src/reduction.rs:28
impl<PCS: AdditivelyHomomorphic> OpeningReduction for PCS
where PCS::Output: HomomorphicCommitment<PCS::Field>,
{ /* RLC-based reduction using PCS::combine */ }
```

So today every consumer of `OpeningReduction` is *implicitly* a
homomorphic-RLC consumer. A FRI-style scheme that batches via a Merkle
inclusion + DEEP-ALI cannot meaningfully `impl AdditivelyHomomorphic`,
yet must still implement `OpeningReduction`; under the blanket it would
have no place to put its native batching strategy.

### Plan

Phase 2 lands in seven logical steps (2.0 – 2.6); each compiles and
passes `cargo nextest run -p jolt-equivalence modular_self_verify`
before the next begins.

**Step 2.0 — De-blanket `OpeningReduction`.** Pure refactor, no
behaviour change.

- Delete the blanket `impl<PCS: AdditivelyHomomorphic> OpeningReduction
  for PCS` in `crates/jolt-openings/src/reduction.rs`.
- Add three explicit per-PCS impls (one per existing PCS):
  `crates/jolt-openings/src/mock.rs::MockCommitmentScheme`,
  `crates/jolt-hyperkzg/src/scheme.rs::HyperKZGScheme`,
  `crates/jolt-dory/src/scheme.rs::DoryScheme`. Each impl is a
  copy of the blanket body, so the byte-level transcript output is
  unchanged.
- In `crates/jolt-verifier/src/verifier.rs`, replace
  `PCS: AdditivelyHomomorphic<Field = ...>` with
  `PCS: OpeningReduction<Field = ...>` on `verify_with_backend` and
  any helpers. The `HomomorphicCommitment<F>` bound on `PCS::Output`
  goes away from the verifier entirely; it stays inside the per-PCS
  `OpeningReduction` impls where it is actually used.
- Remove `crates/jolt-verifier/src/verifier.rs:15`'s import of
  `AdditivelyHomomorphic` (no longer mentioned in the verifier).

This step is the single most important load-bearing change in Phase 2:
it severs the verifier's compile-time dependency on additive
homomorphism. Future hash-based / lattice-based PCS implement
`OpeningReduction` directly with whatever batching is natural for them.

**Step 2.1 — Extend `AstOp` with three commitment variants.**

Add to `crates/jolt-verifier-backend/src/tracing.rs::AstOp`:

```rust
CommitmentWrap   { origin: CommitmentOrigin, label: &'static str },
TranscriptAbsorbCommitment {
    transcript_state: AstNodeId,
    commitment: AstNodeId,
},
OpeningCheck    {
    commitment: AstNodeId,
    point: Vec<AstNodeId>,    // evaluation point as field nodes
    claim: AstNodeId,         // claimed evaluation as field node
    proof_handle: ProofHandle,// opaque sidecar key
    scheme_tag: SchemeTag,    // 'static str discriminator
},
```

`CommitmentOrigin` mirrors `ScalarOrigin` (`Public`, `Proof`).
`SchemeTag` is a `&'static str` (e.g. `"dory"`, `"hyperkzg"`) so the AST
record stays human-readable without coupling to crate-specific types.
`ProofHandle` is an opaque `u32` index into a per-graph
`Vec<Box<dyn Any>>` sidecar that stores the raw opening proof; replay
can hand it back to the PCS verifier as-is. This is the AST-extensibility
escape hatch flagged in alternative 6 below.

`AstAssertion` gains `OpeningHolds { check: AstNodeId }` so replay
discharges opening checks the same way it discharges field equalities.

Replay (`replay_trace`) gains:
- `CommitmentWrap` → consumes one entry from a `commitment_wraps`
  vector parameter (parallel to the existing `wraps: Vec<F>` parameter).
- `TranscriptAbsorbCommitment` → forwards to the live transcript via
  the `AppendToTranscript` impl on the wrapped commitment.
- `OpeningCheck` / `OpeningHolds` → resolves the proof sidecar and
  invokes `<PCS as CommitmentScheme>::verify`.

**Step 2.2 — `CommitmentBackend` trait (3 methods, no curve mention).**

```rust
// crates/jolt-verifier-backend/src/backend.rs
pub trait CommitmentBackend<PCS: CommitmentScheme>: FieldBackend {
    /// Backend-side handle for a commitment. `Native::Commitment = PCS::Output`;
    /// `Tracing::Commitment = AstNodeId`.
    type Commitment: Clone + std::fmt::Debug;

    fn wrap_commitment(
        &mut self,
        value: PCS::Output,
        origin: CommitmentOrigin,
        label: &'static str,
    ) -> Self::Commitment;

    fn absorb_commitment(
        &mut self,
        transcript: &mut Self::Transcript,
        commitment: &Self::Commitment,
        label: &'static [u8],
    );

    /// Verifies a single opening claim. Reduction (RLC, FRI batching, etc.)
    /// is the PCS's responsibility via `OpeningReduction::reduce_verifier_with_backend`,
    /// not the backend's.
    fn verify_opening(
        &mut self,
        vk: &PCS::VerifyingKey,
        commitment: &Self::Commitment,
        point: &[Self::Scalar],
        claim: &Self::Scalar,
        proof: &PCS::OpeningProof,
        transcript: &mut Self::Transcript,
        scheme_tag: SchemeTag,
    ) -> Result<(), OpeningsError>;
}
```

What is *not* on this trait: anything that names a curve, a pairing,
an MSM, or a linear combination of commitments. Those are PCS-internal
details, exposed (or not) through `OpeningReduction::reduce_verifier_with_backend`
on a per-PCS basis.

**Step 2.3 — `Native<F>` impl.**

```rust
type Commitment = PCS::Output;          // identity wrap
fn wrap_commitment(_, value, _, _) { value }
fn absorb_commitment(_, t, c, label) { c.append_to_transcript(t, label); }
fn verify_opening(_, vk, c, point, claim, proof, t, _) {
    PCS::verify(vk, c, point, claim, proof, t)
}
```

Zero-cost: every method is `#[inline]`, no allocation, transcript
behaviour bit-identical to the legacy verifier.

**Step 2.4 — `Tracing<F>` impl.**

`Tracing` already owns `Arc<Mutex<AstGraph>>`. It additionally owns
`Arc<Mutex<ProofSidecar>>` where `ProofSidecar = Vec<Box<dyn Any +
Send>>`. `wrap_commitment` pushes a node and stashes nothing extra;
`absorb_commitment` pushes a `TranscriptAbsorbCommitment` node *and*
forwards to the live `TracingTranscript`'s inner Blake2b so squeezed
challenges remain consistent with the native path; `verify_opening`
pushes the proof into the sidecar, returns its `ProofHandle`, records
an `OpeningCheck` node + `OpeningHolds` assertion, and returns `Ok(())`
without invoking the actual PCS verifier (the assertion will fire at
replay time).

**Step 2.5 — `OpeningReduction::reduce_verifier_with_backend`.**

Add a backend-aware mirror to the trait:

```rust
fn reduce_verifier_with_backend<B: CommitmentBackend<Self>>(
    backend: &mut B,
    claims: &[(B::Commitment, Vec<B::Scalar>, B::Scalar)],
    transcript: &mut B::Transcript,
) -> Result<Vec<(B::Commitment, Vec<B::Scalar>, B::Scalar)>, OpeningsError>;
```

The Dory / HyperKZG / Mock impls move their RLC body into this method,
calling `backend.wrap_commitment` / `backend.absorb_commitment` for the
combined commitment. A future FRI scheme implements `reduce_verifier_with_backend`
with whatever batching is natural for it; the verifier never sees
the difference.

The blocking generic combine (`PCS::combine` on `PCS::Output`) is still
called from inside the Dory / HyperKZG implementations of
`reduce_verifier_with_backend`, but only against `Native::Commitment =
PCS::Output`. For `Tracing`, the per-PCS impl emits a `CommitmentWrap`
node carrying the precomputed combined commitment (computed natively in
the Tracing prover-stub path) and records the inputs as transcript
absorptions. This keeps the AST free of curve-shaped opcodes.

**Step 2.6 — Cut `verify_with_backend` over.**

In `crates/jolt-verifier/src/verifier.rs`:
- Replace direct `c.append_to_transcript(...)` calls with
  `backend.absorb_commitment(&mut transcript, &c, label)`.
- Replace `PCS::reduce_verifier(...)` calls with
  `<PCS as OpeningReduction>::reduce_verifier_with_backend(&mut backend, ...)`.
- Replace `PCS::verify(...)` calls with
  `backend.verify_opening(vk, &c, &point, &claim, &proof, &mut transcript, "<scheme>")`.
- Replace the `where PCS: AdditivelyHomomorphic<Field = ...>` bound
  with `where PCS: OpeningReduction<Field = ...>` (already done in
  step 2.0) plus `B: CommitmentBackend<PCS>`.

After this step the legacy `verify` (without `_with_backend`) is the
only non-polymorphic caller of `PCS::verify` left in the crate, and is
slated for deletion in step 2.6's commit.

### Trait surface (summary)

```
FieldBackend  (Phase 1, unchanged)
   └── CommitmentBackend<PCS>  (Phase 2, new)
            wrap_commitment
            absorb_commitment
            verify_opening

OpeningReduction  (Phase 1, unchanged signature; blanket removed in 2.0)
   ├── reduce_verifier              (existing)
   └── reduce_verifier_with_backend (Phase 2, new — required)
   per-PCS impls in jolt-dory, jolt-hyperkzg, jolt-openings::mock
```

Crucially, no `GroupBackend` and no curve-shaped `AstOp` variants.

### AST extensibility for non-Dory schemes

`AstOp::OpeningCheck { scheme_tag, proof_handle, … }` is the
single extension point. A scheme-specific verifier consumer (e.g. a
Lean exporter, a recursion verifier) dispatches on `scheme_tag` and
loads the corresponding `proof_handle` from the sidecar. This avoids
making `AstOp` either a closed enum that knows every PCS or an open
trait-object that loses serialisability. The cost is one indirection
per opening check at replay time, which is negligible relative to the
opening verification itself.

### Phase 2 acceptance criteria

- [ ] Step 2.0 compiles and `cargo nextest run -p jolt-openings`,
      `-p jolt-verifier`, `-p jolt-equivalence modular_self_verify`,
      `-p jolt-equivalence modular_self_verify_via_tracing_backend`
      all pass with **no** `AdditivelyHomomorphic` bound on
      `verify_with_backend`.
- [ ] `cargo clippy -p jolt-verifier --all-targets -- -D warnings` shows
      `AdditivelyHomomorphic` mentioned **only** inside `jolt-zkvm`
      (prover side) and inside per-PCS crates; not in `jolt-verifier`
      and not in `jolt-verifier-backend`.
- [ ] `CommitmentBackend` trait exists with exactly three methods.
      `Native::Commitment = PCS::Output`; `Tracing::Commitment = AstNodeId`.
- [ ] `AstOp` gains exactly the three variants
      `CommitmentWrap`, `TranscriptAbsorbCommitment`, `OpeningCheck`.
      `AstAssertion` gains `OpeningHolds`. `viz::to_dot` /
      `viz::to_mermaid` render each with distinct styling.
- [ ] `OpeningReduction::reduce_verifier_with_backend` is implemented
      for `MockCommitmentScheme`, `HyperKZGScheme`, `DoryScheme`. Each
      impl is byte-identical to its legacy `reduce_verifier` on the
      `Native` backend (asserted by a parity test).
- [ ] `modular_self_verify_via_tracing_backend` (extended) asserts the
      replayed AST contains ≥ 1 `CommitmentWrap`, ≥ 1
      `TranscriptAbsorbCommitment`, ≥ 1 `OpeningCheck`, and discharges
      every `OpeningHolds` assertion.
- [ ] No mention of `g1_msm`, `pairing`, `MSM`, or `GroupBackend`
      anywhere in `crates/jolt-verifier-backend/src/`. The Phase 2
      surface is curve-agnostic.

### Phase 2 testing strategy

- **Reduction parity (per PCS).** New unit test in each of
  `crates/jolt-dory`, `crates/jolt-hyperkzg`, `crates/jolt-openings`:
  build a small bag of opening claims, call both `reduce_verifier` and
  `reduce_verifier_with_backend::<Native<_>>`, assert byte-identical
  combined commitment + transcript state.
- **End-to-end Tracing replay (extended).**
  `modular_self_verify_via_tracing_backend` runs the muldiv proof
  through `Tracing` with `CommitmentBackend` engaged, replays the AST
  + sidecar, and checks every `OpeningHolds` and field equality.
- **Negative test.** A tampered opening proof in the sidecar causes
  `OpeningHolds` to fail at replay with a structured error pointing
  to the `OpeningCheck` node id.

### Alternatives considered (Phase 2)

6. **`AstOp::PcsExtension { tag, payload: Vec<u8> }`** — a fully
   serialised escape hatch for arbitrary PCS-specific subgraphs.
   Rejected for now in favour of the narrower
   `OpeningCheck { scheme_tag, proof_handle, … }` shape: every PCS we
   know of (curve, FRI, lattice) needs the same four field-side
   ingredients (commitment, point, claim, proof), and the backend can
   route them through a single AST node + sidecar. If a future PCS
   needs a richer in-AST representation, `PcsExtension` can be added
   without disturbing the existing variants.
7. **`GroupBackend` with `g1_msm` / `pairing` (original sketch).**
   Rejected. See "Why an amendment" above.
8. **Keep the `AdditivelyHomomorphic` blanket and add a parallel
   `OpeningReductionNonHomomorphic` trait.** Rejected — splits the
   verifier surface in two and forces every consumer to handle both
   cases. Single trait + per-PCS impl is strictly simpler.

## References

- PR #1461 (this PR) — implementation.
- Markos's `refactor/crates` base commit `c7e0869d5` — the
  `VerifierOp` interpreter we're parameterising over.
- `paper-note/multi-backend-verifier-design.md` — original design sketch
  (in `~/Documents/Notes/`) referenced during the v1 prototype on
  `quang/verifier-backend`.
- Closed Dory PR (`~/Documents/SNARKs/dory`) — prior art for the same
  symbolic-AST-over-native-execution pattern in a single-protocol setting.
