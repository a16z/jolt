# Universal Sumcheck Prover Interface

**Status:** Draft design spec  
**Stack:** Modular Jolt (`prover-stack/10-jolt-prover` in `third-party/jolt`)  
**Related:** [`jolt_cuda_integration.md`](jolt_cuda_integration.md) (GPU backend integration)

---

## Summary

Modular Jolt already has a clean verifier story: each stage declares typed inputs and outputs, evaluates symbolic **claims** from `jolt-claims`, and replays sumchecks through a shared **`BatchedSumcheckVerifier`**. The prover side has no equivalent. Each stage hand-rolls the same batched-sumcheck Fiat–Shamir loop, couples to stage-specific backend traits, and duplicates logic that the verifier encodes once.

This document specifies a **universal sumcheck prover interface** that closes that gap:

- One **canonical batched-sumcheck handler** (Fiat–Shamir, RLC, front-loaded alignment, proof recording).
- One **public relation type** (`SumcheckRelation`) shared by prover, verifier, and backends.
- Three **backend capabilities** (sumcheck, commit, open) that CPU, GPU cluster, and reference implementations satisfy under the same laws.
- A **reference prover** (dense materialization) as the correctness oracle.

The design is reusable beyond Jolt (e.g. Akita). Jolt instantiates it with `JoltRelationId`, `JoltOpeningId`, and stage-specific `ProverProgram`s.

---

## Table of contents

1. [Motivation: the prover–verifier asymmetry](#1-motivation-the-proververifier-asymmetry)
2. [Goals and non-goals](#2-goals-and-non-goals)
3. [Design principles](#3-design-principles)
4. [Architecture overview](#4-architecture-overview)
5. [The public interface](#5-the-public-interface)
6. [Crate layout](#6-crate-layout)
7. [Backend instantiations](#7-backend-instantiations)
8. [Worked example: Stage 4](#8-worked-example-stage-4)
9. [Migration plan](#9-migration-plan)
10. [Testing strategy](#10-testing-strategy)
11. [Formal reading (PL and category theory)](#11-formal-reading-pl-and-category-theory)
12. [Naming conventions](#12-naming-conventions)
13. [Prior art and lineage](#13-prior-art-and-lineage)

---

## 1. Motivation: the prover–verifier asymmetry

### What the modular stack looks like today

On branch `prover-stack/10-jolt-prover`, the modular crates split roughly as follows:

| Crate | Role |
|-------|------|
| `jolt-claims` | Symbolic **relations**: `JoltRelationClaims` with `input` / `output` expressions, sumcheck shape, consistency constraints |
| `jolt-sumcheck` | **Verifier-only** batched sumcheck engine (`BatchedSumcheckVerifier`) |
| `jolt-verifier` | Per-stage `verify.rs`: typed deps, claim evaluation, sumcheck replay |
| `jolt-prover` | Per-stage `prove.rs`: hand-rolled sumcheck loops, stage-local proof recorders |
| `jolt-backends` | CPU kernels behind **five stage-specific trait bundles** |
| `jolt-cuda` (jolt-cpp) | GPU cluster backend; today mostly delegates to CPU |

The verifier path is data-driven. For example, Stage 6 verification builds `JoltRelationClaims`, calls `BatchedSumcheckVerifier`, then checks `claim.input.expression().try_evaluate(...)` and `claim.output.expression().try_evaluate(...)` against opening claims.

The prover path is code-driven. Stage 6 proving (`jolt-prover/src/stages/stage6/prove.rs`) implements its own `for round in 0..max_num_vars` loop: per-instance `evaluate_sumcheck_*_round`, RLC combination, `proof_recorder.absorb_round`, challenge squeeze, `bind_sumcheck_*_state`. Stages 2, 4, and others repeat the same pattern with different match arms.

### Concrete problems

**1. Duplicated batched-sumcheck logic**

`BatchedSumcheckVerifier` (`jolt-sumcheck/src/batched_verifier.rs`) already encodes front-loaded batching, dummy rounds for shorter instances, and RLC of round polynomials. Each prover stage reimplements the same arithmetic. Stage 4's loop (`jolt-prover/src/stages/stage4/prove.rs`, ~lines 960–991) is representative: scale claims by `2^offset`, synthesize dummy univariates for inactive instances, RLC by batching coefficients, absorb and squeeze.

Any change to batching semantics must be patched in every stage prover and kept in sync with the verifier.

**2. Prover depends on verifier for claim plumbing**

Prover stages import helpers from `jolt-verifier` (`stage6_batch_input_claims`, `stage4_expected_outputs`, …). The symbolic spec lives in three places: `jolt-claims` formulas, verifier helpers, and prover loops. That inversion makes it hard to reuse the prover stack outside Jolt.

**3. Stage-specific backend surface**

`jolt-backends/src/traits.rs` exposes separate traits per stage (`Stage4ReadWriteSumcheckBackend`, `Stage6RegularBatchSumcheckBackend`, …), each with its own state types and `materialize` / `evaluate_*_round` / `bind_*` / `output_*` methods. A GPU backend must implement the entire bundle. Adding a relation means new trait methods, new prover match arms, and a new CPU kernel file.

There is no single "sumcheck oracle" interface keyed by relation.

**4. Transcript authority is implicit**

Fiat–Shamir order (absorb input claims, draw batching coefficients, absorb round polynomials, squeeze challenges, append opening claims) is scattered across stage prove functions and stage-local `*ProofRecorder` traits. The verifier must mirror this exactly. In the legacy GPU stack, challenges are sampled only on the Rust host (`rust-bindings/src/sumchecks/batched.rs`); the cluster never draws FS challenges. That invariant should be explicit in the modular interface.

**5. No shared prove engine in `jolt-sumcheck`**

`jolt-sumcheck` is verifier-safe by design. The prove side has no `BatchedSumcheckProver`. Correctness testing relies on end-to-end proofs rather than isolating the sumcheck IOP.

**6. GPU integration mismatch**

Legacy GPU proving (`rust-bindings`) uses per-`SumcheckType` proxy drivers and a `bind_eval` round API aligned with `jolt-core`. The modular stack expects stage-specific backend state machines. Mapping a sealed GPU cluster to the modular prover requires a stable oracle boundary, not a one-to-one port of legacy drivers.

### What success looks like

After this refactor:

- Stage `prove.rs` files **lower** a stage into a `ProverProgram` (commit / batched sumcheck / open steps) plus gamma derivation and witness wiring.
- **One handler** runs every batched sumcheck in the VM.
- CPU and GPU backends implement the same four-method sumcheck oracle.
- The **reference prover** materializes relations densely and serves as the byte-identical correctness target.
- `jolt-verifier` and external verifiers depend only on `sumcheck-core`, never on prover traits.

---

## 2. Goals and non-goals

### Goals

- **Verifier parity:** Same transcript bytes, sumcheck proof payloads, challenges, opening claims, and final claims as today (observational equality).
- **Backend portability:** CPU dense kernels, fused implementations (Akita), and a sealed distributed GPU cluster are interchangeable behind one interface.
- **Relation-centric:** One `SumcheckRelation` (generalizing `JoltRelationClaims`) drives prove, verify, and reference materialization.
- **Correctness oracle:** A slow reference backend proves by literal hypercube fold; fast backends must match its round polynomials on fixtures.
- **Reuse:** Core crates carry no Jolt stage IDs; Jolt and Akita instantiate the same machinery.

### Non-goals (this spec)

- Replacing Dory PCS or Stage 8 opening logic (only the sumcheck/commit/open **step** abstraction).
- `field-inline` and full ZK / BlindFold paths (the `SumcheckProofRecorder` trait is the extension point; committed/ZK recorders plug in later).
- Deleting `jolt-core` in the upstream submodule (only the jolt-cpp modular integration path).
- Preserving legacy `SumcheckType` + `bind_eval` as the long-term GPU API (cluster code may wrap it privately inside `SumcheckBackend`).

---

## 3. Design principles

### Observational equality is the contract

Backends may differ in memory layout, residency (host vs device vs cluster), fusion, sharding, and split CPU/GPU round schedules. None of that appears in the public API. The only hard requirement is that the **observable artifacts** match:

- Transcript absorption order and bytes
- Compressed sumcheck proofs (clear or committed)
- Per-round challenges
- Opening claims and stage public outputs

### Fiat–Shamir is centralized

The handler owns every `absorb` and `challenge` in the sumcheck IOP. Backends never sample transcript challenges. This matches production GPU behavior and removes a class of host/device desync bugs.

### Relations are public; witnesses are private

`SumcheckRelation` (summand \(g\), input claim formula, sumcheck statement) is the contract both prover and verifier evaluate. How a backend computes round polynomials over a witness, trace, or device-resident MLE is entirely private.

### Byte-identical round polynomials follow from the laws

If each round polynomial equals the partial sum of the declared summand over the remaining hypercube, and commitments/openings satisfy the PCS laws, then observational equality is forced. Fast backends are tested against the reference prover, not merely against acceptance by the verifier.

### Verifier must not depend on prover code

`sumcheck-core` holds relations, claim types, and verification. `sumcheck-prover` holds the handler and backend traits. `jolt-verifier` links only to `sumcheck-core`.

---

## 4. Architecture overview

```text
                    ┌─────────────────────────────────────┐
                    │  jolt-prover (Jolt stage programs)   │
                    │  StageBuilder → ProverProgram        │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  sumcheck-prover                     │
                    │  prove(program) · prove_sumcheck()   │
                    │  SumcheckBackend / Commit / Open     │
                    │  ReferenceSumcheck (oracle)            │
                    └───────┬─────────────────┬───────────┘
                            │                 │
              ┌─────────────▼──────┐   ┌──────▼──────────────┐
              │  jolt-backends      │   │  jolt-cuda cluster  │
              │  (CPU algebra)      │   │  (GPU algebra)      │
              └─────────────┬──────┘   └──────┬──────────────┘
                            │                 │
                    ┌───────▼─────────────────▼───────────┐
                    │  sumcheck-core                       │
                    │  SumcheckRelation · OpeningClaim     │
                    │  BatchedSumcheckVerifier             │
                    └───────┬─────────────────────────────┘
                            │
                    ┌───────▼───────────┐
                    │  jolt-verifier     │
                    └───────────────────┘
```

**Proof as a program.** A Jolt proof is a sequence of protocol steps: commit polynomials, run batched sumchecks, run opening arguments. The program says *what* happens in transcript order, not *how* each backend computes it.

**Three backend capabilities** compose fractally: a Dory opening is itself a smaller program over sumcheck + commit steps interpreted against the same transcript.

---

## 5. The public interface

### 5.1 Observable protocol steps (`ProverProgram`)

```rust
pub enum ProverStep<F> {
    Commit(CommitmentRequest),
    BatchedSumcheck(BatchedSumcheckSpec<F>),
    Open(OpeningRequest<F>),
}

pub struct Stage<F> {
    pub label: &'static str,
    pub steps: Vec<ProverStep<F>>,
}

pub struct ProverProgram<F> {
    pub stages: Vec<Stage<F>>,
}
```

Jolt stage code builds a `ProverProgram` after deriving transcript challenges (gammas, domain separators, etc.). The generic handler executes steps in order.

### 5.2 Sumcheck relation (`sumcheck-core`)

Generalizes `JoltRelationClaims`:

```rust
pub struct SumcheckRelation<Open, Pub, Chal, D> {
    pub id: RelationId,
    pub statement: SumcheckStatement<D>,   // num_rounds, degree, round domain
    pub input: Expr<Open, Pub, Chal>,     // initial claim checked at batch start
    pub output: Expr<Open, Pub, Chal>,     // summand g; final-claim law at batch end
}
```

`jolt-claims` formula functions (`registers::read_write_checking`, `ram::val_check`, …) become Jolt's instantiation of this type. The verifier already evaluates `input` and `output` symbolically; the prover will use the same object for claim checks and (in the reference backend) for hypercube folding.

### 5.3 Batched sumcheck spec

```rust
pub struct BatchedSumcheckSpec<F> {
    pub label: &'static str,
    pub instances: Vec<SumcheckInstance<F>>,
    pub num_rounds: usize,   // max_i (offset_i + statement_i.num_rounds)
}

pub struct SumcheckInstance<F> {
    pub relation: RelationId,
    pub input_claim: F,
    pub alignment: RoundOffset,          // front-loaded dummy rounds
    pub bindings: WitnessMap<F>,         // reference backend only
    pub optimization_ids: Vec<&'static str>,  // backend hint; not observable
}
```

### 5.4 Canonical batched-sumcheck handler

This is the **only** place Jolt owns sumcheck IOP logic. It mirrors `BatchedSumcheckVerifier`:

```text
absorb input claims into transcript / proof recorder
sample batching coefficients r_i from transcript
scale claim_i by 2^{alignment_i}
running ← Σ r_i · claim_i

for round r in 0..num_rounds:
    active ← instances with alignment ≤ r
    polys ← backend.round_polynomials(st, r, active, active_claims)
    for inactive instances: dummy poly (constant claim · ½)
    batched ← Σ r_i · poly_i
    assert batched(0) + batched(1) == running
    challenge ← recorder.absorb_round(batched, transcript)
    running ← batched(challenge)
    update per-instance claims; backend.bind(st, r, active, challenge)

outputs ← backend.finish(st, challenges)
```

Clear vs ZK differs only in `SumcheckProofRecorder` (cleartext compressed coeffs vs Pedersen-committed rounds), analogous to today's `ClearStage4RegularBatchProofRecorder` vs `CommittedStage4RegularBatchProofRecorder`.

### 5.5 Backend capabilities

```rust
pub trait SumcheckBackend<F> {
    type State;
    fn start(&mut self, spec: &BatchedSumcheckSpec<F>, res: &mut ResourceLedger)
        -> Result<Self::State, BackendError>;
    fn round_polynomials(&mut self, st: &mut Self::State, round: usize,
                         active: &[usize], claims: &[F])
        -> Result<Vec<UnivariatePoly<F>>, BackendError>;
    fn bind(&mut self, st: &mut Self::State, round: usize,
            active: &[usize], challenge: F) -> Result<(), BackendError>;
    fn finish(&mut self, st: Self::State, challenges: &[F])
        -> Result<Vec<ProverOutput<F>>, BackendError>;
}

pub trait CommitBackend<F> { /* commit under PCS */ }
pub trait OpenBackend<F>   { /* opening protocol; may call prove(subprogram) */ }
```

**Notes for implementers:**

- `round_polynomials` returns **`Vec<UnivariatePoly<F>>` with one entry per active instance**, not one RLC-combined polynomial. Batched sumcheck runs several independent sumcheck instances in one IOP; each active instance at round `r` has its own round univariate. The **handler** (not the backend) applies batching coefficients and RLC-combines them into the single transcript-visible polynomial, matching what Stage 4 does today when it calls two `evaluate_*_round` paths then combines. A batch of size one yields a vector of length 1.
- `round_polynomials` and `bind` stay separate. A GPU backend whose native op fuses bind+eval (`bind_eval`) may defer binding to the next `round_polynomials` call, as the legacy proxy does.
- A distributed backend returns the **full** per-instance round polynomial; cross-shard field sums happen below the trait boundary.
- Stage-specific traits (`Stage4ReadWriteSumcheckBackend`, …) become **private** methods inside the CPU/GPU algebra, not public API.

### 5.6 Prover outputs

Backends return structured data for later stages and ledgers:

```rust
pub enum ProverOutput<F> {
    Opening(OpeningClaim<F>),
    Resource(ResourceHandle),
    PublicOutput(PublicOutputValue<F>),
}
```

`OpeningClaim` is shared with the verifier (`sumcheck-core`). No duplicate "opening effect" type.

### 5.7 Proof recorder and top-level prove

```rust
pub trait SumcheckProofRecorder<F> {
    type Proof;
    fn absorb_input_claims<T: Transcript<Challenge = F>>(&mut self, claims: &[F], t: &mut T);
    fn absorb_round<T: Transcript<Challenge = F>>(&mut self, m: &UnivariatePoly<F>, t: &mut T)
        -> Result<F, ProverError>;
    fn finish<T: Transcript<Challenge = F>>(self, t: &mut T, outputs: &[F])
        -> Result<Self::Proof, ProverError>;
}

pub fn prove_sumcheck<F, B, T, S>(/* spec, backend, transcript, recorder, resources */) -> ...;

pub fn prove<F, B, T, S>(
    program: &ProverProgram<F>,
    backend: &mut B,
    transcript: &mut T,
    recorder: &mut S,
    session: &mut ProvingContext<F>,
) -> Result<ProofBundle<F>, ProverError>;
```

### 5.8 Session state

```rust
pub struct ProvingContext<'a, F, B, T> {
    pub backend: &'a mut B,
    pub transcript: &'a mut T,
    pub resources: ResourceLedger,
    pub openings: OpeningLedger<F>,
    pub public_outputs: PublicOutputLedger<F>,
}
```

`ResourceRef` is opaque (host buffer, cluster trace handle, etc.). No CUDA card, stream, or shard types in the public API. `StageBuilder` is the narrow view Jolt stage code uses to inspect prior outputs and emit `ProverProgram` steps.

---

## 6. Crate layout

```text
sumcheck-core
  SumcheckRelation, Expr, OpeningClaim, CompressedSumcheckProof
  BatchedSumcheckVerifier (existing logic, relocated/generalized)
  Verifier-safe: no prover traits

sumcheck-prover
  ProverProgram, prove / prove_sumcheck, backend traits
  SumcheckProofRecorder, ReferenceSumcheck
  Depends on sumcheck-core

jolt-verifier     → sumcheck-core only
jolt-prover       → sumcheck-prover; builds Jolt ProverPrograms
jolt-backends     → CPU SumcheckBackend algebra
jolt-cuda         → GPU cluster SumcheckBackend algebra
```

**`jolt-claims` refactor:** Lift `JoltExpr` / `JoltRelationClaims` into generic `Expr<Open, Pub, Chal>` and `SumcheckRelation` in `sumcheck-core`. Jolt keeps its ID enums; formulas stay in `jolt-claims/src/protocols/jolt/formulas/`.

---

## 7. Backend instantiations

| Algebra | Source today | Role |
|---------|--------------|------|
| **Reference** | New dense folder | Materialize MLE tables, fold `output` over hypercube. Correct by construction; equivalence test oracle. |
| **CPU** | `jolt-backends` kernels | Wrap `materialize_*` / `evaluate_*_round` / `bind_*` / `output_*` behind `SumcheckBackend`. |
| **GPU cluster** | `rust-bindings` / `jolt-cuda` | Map `BatchedSumcheckSpec` to cluster RPC; `round_polynomials` wraps `batch_bind_eval_multi`. Residency, packing, split handoff stay private. |
| **Akita fused** | Future | One `SumcheckBackend` scanning shared witness tables for multiple sub-claims. |

The sealed GPU case is a first-class requirement: Jolt developers integrate a cluster knowing only that it implements `SumcheckBackend` + commit/open for declared relations. Wire formats and kernels are not part of the spec.

---

## 8. Worked example: Stage 4

Stage 4 is the smallest interesting **front-loaded batch**: two sumcheck instances with different round counts in one IOP.

| Instance | Relation | Rounds | Offset |
|----------|----------|--------|--------|
| Registers read-write | `registers::read_write_checking` | `log_t + REGISTER_ADDRESS_BITS` | 0 |
| RAM value-check | `ram::val_check` | `log_t` | `REGISTER_ADDRESS_BITS` |

### Relations (from `jolt-claims`)

Already defined in `jolt-claims/src/protocols/jolt/formulas/registers.rs` and `ram.rs` as `JoltRelationClaims`. Under this spec they become `SumcheckRelation` values with the same `input` / `output` expressions.

### Prover program (what Stage 4 emits)

After sampling `registers_gamma` and `ram_val_check_gamma` in transcript order:

```rust
ProverProgram {
    stages: vec![Stage {
        label: "stage4.regular_batch",
        steps: vec![ProverStep::BatchedSumcheck(BatchedSumcheckSpec {
            label: "stage4.regular_batch",
            num_rounds: log_t + REGISTER_ADDRESS_BITS,
            instances: vec![
                SumcheckInstance {
                    relation: RegistersReadWriteChecking,
                    input_claim: input_claims.registers_read_write,
                    alignment: RoundOffset { offset: 0 },
                    bindings: registers_sources(...),
                    optimization_ids: vec!["cpu_stage4_regular_batch_sumcheck"],
                },
                SumcheckInstance {
                    relation: RamValCheck,
                    input_claim: input_claims.ram_val_check,
                    alignment: RoundOffset { offset: REGISTER_ADDRESS_BITS },
                    bindings: ram_sources(...),
                    optimization_ids: vec!["cpu_stage4_regular_batch_sumcheck"],
                },
            ],
        })],
    }],
}
```

### Mapping from today's `stage4/prove.rs`

The handler reproduces the existing loop in `jolt-prover/src/stages/stage4/prove.rs` (materialize states, RLC loop, opening extraction). The table below is the intended correspondence; stage code shrinks to program construction plus gamma derivation.

| Today's Stage 4 prover | New handler / backend |
|------------------------|------------------------|
| `materialize_sumcheck_*_state` | `backend.start(spec, res)` |
| `proof_recorder.absorb_input_claims` + batching coeffs | `recorder.absorb_input_claims` + transcript |
| `individual_claims[1] *= 2^ram_offset` | front-loaded claim scaling in handler |
| `evaluate_sumcheck_*_round` | `backend.round_polynomials` |
| dummy poly when `round < ram_offset` | handler inactive-instance branch |
| RLC + `absorb_round` + challenge | handler RLC + `recorder.absorb_round` |
| `bind_sumcheck_*_state` | `backend.bind` |
| `output_sumcheck_*_state` | `backend.finish` → `OpeningClaim`s |
| `stage4_expected_final_claim` guard | generic `output` evaluation law |

The field-inline third instance in Stage 4 is a third `SumcheckInstance` with a different `RoundOffset`; no handler change.

### Equivalence test

Reference, CPU, and GPU algebras must produce **byte-identical round polynomials** on a Stage 4 fixture. That is stricter than verifier acceptance alone and catches transcript-order bugs early.

---

## 9. Migration plan

1. **`sumcheck-core`:** `SumcheckRelation`, shared claim/proof types, relocate/generalize `BatchedSumcheckVerifier`. Genericize `jolt-claims` IDs into type parameters.
2. **`sumcheck-prover`:** Handler, traits, `SumcheckProofRecorder`, `ReferenceSumcheck`.
3. **CPU algebra:** Wrap existing `jolt-backends` state methods; port **Stage 4** first (smallest front-loaded batch).
4. **GPU algebra:** Adapter from `BatchedSumcheckSpec` to cluster proxy (`jolt-cuda`); private `bind_eval` mapping.
5. **Stages 3, 5, 6:** Lower to `ProverProgram`; delete hand-rolled loops as each lands.
6. **Stage 6 proof recorder:** Lift into generic `SumcheckProofRecorder` (clear + committed).
7. **Equivalence tests:** Reference vs CPU vs GPU on fixtures; handler unit tests (offsets, dummy rounds, recorders).
8. **Trait cleanup:** Remove stage-specific sumcheck traits from public `jolt-backends` surface once algebras stabilize.

Phasing aligns with [`jolt_cuda_integration.md`](jolt_cuda_integration.md): GPU work targets the stable oracle boundary, not a rewrite of legacy `SumcheckProxyDriver` as the long-term API.

---

## 10. Testing strategy

| Layer | What to test |
|-------|----------------|
| **Handler** | Single instance, front-loaded batch, custom offsets, dummy rounds, clear and committed recorders |
| **Reference algebra** | Proofs verify through `BatchedSumcheckVerifier` |
| **Relation laws** | `input_claim == input.evaluate(...)`, `output == output.evaluate(...)` at sampled points |
| **Equivalence** | Fast backends byte-match reference round polynomials and transcript event order |
| **Regression** | Modular `muldiv` e2e (clear, then committed/ZK) |

---

## 11. Formal reading (PL and category theory)

This section records the formal model behind the crypto-facing API. It is not required to implement backends.

**Free program.** A proof is a term in a free algebra over protocol steps (commit, batched sumcheck, open). `ProverProgram` is that term; it says nothing about memory or devices.

**Interpreter.** A backend is an algebra interpreting the oracle sub-signature: `round_polynomial`, `commit`, `open`. The handler interprets the Fiat–Shamir sub-signature uniquely.

**Laws.** The verifier's checks are equalities:

```text
round_poly(r) = Σ_x g(challenges, X, x)   -- partial sum of declared summand
open(C, x)    = eval(commit(C), x)
```

**Observation functor.** Write `O(execution)` for `(transcript, proof bytes, opening claims, public outputs)`. If two backends satisfy the laws, then `O ∘ A_cpu = O ∘ A_gpu`. Observational equality is a **theorem**, not an extra test requirement per backend.

**Initial algebra.** The reference prover (dense fold) is the initial algebra satisfying the laws; equivalence testing is the universal property in code form.

---

## 12. Naming conventions

Locked-in public names (used consistently in §5 and downstream crates):

| Concept | Name |
|---------|------|
| Opaque resource handle in `ProverOutput` | `ResourceHandle` |
| Stage-local derived scalars | `PublicOutput`, `PublicOutputValue`, `PublicOutputLedger` |
| Front-loaded batch alignment | `RoundOffset { offset }`; `SumcheckInstance::alignment` |
| Top-level program interpreter | `prove(program, …)` |
| Final prover return type | `ProofBundle` |
| Round-polynomial oracle trait | `SumcheckBackend` (with `CommitBackend`, `OpenBackend`) |
| Symbolic relation type | `SumcheckRelation` |

`prove_sumcheck` is the batched-sumcheck handler only; it does not collide with `prove` because the latter takes a full `ProverProgram`.

---

## 13. Prior art and lineage

Markos Georghiades (`markosg04` on GitHub) has driven three overlapping modular-prover lines in [a16z/jolt](https://github.com/a16z/jolt). None is finished; each made different bets about where protocol semantics live and how sumcheck proving is factored. This proposal is a fourth direction: **handwritten orchestration** (like `prover-stack`) with a **single batched-sumcheck handler** and **relation-keyed backend oracle** (ideas partially present in all three, never unified).

Authoritative upstream repo for archaeology: sibling clone `../jolt` (or `third-party/jolt` in this workspace).

### 13.1 Timeline at a glance

```text
2026-04  jolt-v2/* merged to main
         field, transcript, poly, crypto, trace, r1cs, sumcheck-verifier, openings, dory
         (infra only; no jolt-prover on main)

2026-05  refactor/audit-prep (local branch, stack removed #1581)
         handwritten jolt-prover + request/result jolt-backends + jolt-prover-harness
         specs/jolt-prover-model-crate.md

2026-05+ jolt-v2/bolt stack (open PRs #1512–#1515)
         Bolt compiler, generated roles, jolt-equivalence oracle vs jolt-core
         quang/bolt-stack (#1523, closed): typed verifier plans on same stack

2026-06  prover-stack/* (open PRs #1596–#1606)
         restacked handwritten jolt-prover + witness + claims extensions
         pinned today as prover-stack/10-jolt-prover (GPU integration target)
```

### 13.2 Attempt 1: jolt-v2 modular crates + `refactor/audit-prep`

**What merged.** Workspace split under `jolt-v2/*` ([#1362](https://github.com/a16z/jolt/pull/1362) through [#1451](https://github.com/a16z/jolt/pull/1451)): `jolt-field`, `jolt-transcript`, `jolt-poly`, `jolt-crypto`, `jolt-trace`, `jolt-r1cs`, **`jolt-sumcheck` (verifier only)**, `jolt-openings`, `jolt-dory`. This is the foundation every later line builds on.

**What did not merge.** A full handwritten prover on branch **`refactor/audit-prep`** (inspect locally: `git log refactor/audit-prep --author=Markos`).

| Artifact | Role |
|----------|------|
| `specs/jolt-prover-model-crate.md` | 900-line design spec ([#1550](https://github.com/a16z/jolt/pull/1550), closed stack-bot PR) |
| `crates/jolt-prover/src/assembly.rs` | Large proof **assembly** layer: maps backend outputs into verifier-owned `JoltProof` / stage structs |
| `crates/jolt-prover/src/builder.rs` | **Builder** pattern for semantically keyed commitment/opening components |
| `crates/jolt-prover/src/prover.rs` | Stage orchestration (~1.7k LOC on audit-prep vs ~870 on current `prover-stack`) |
| `crates/jolt-backends/` | **Request/result** API per family (`sumcheck/request.rs`, `CONTRACT.md`), not stage-named traits |
| `crates/jolt-prover-harness/` | Frontier replay, `CorePerformanceParity`, optimization-ID ledger vs `jolt-core` |
| `specs/jolt-prover-cpu-backend-port.md` | CPU port plan (referenced from backends README) |

**Core architectural bets.**

1. **Explicit split:** `jolt-verifier` owns proof shape; `jolt-prover` owns transcript order and assembly; `jolt-backends` owns compute; **no Bolt**.
2. **Backend boundary = requests, not stages.** `SumcheckRequest` carries `SumcheckInstanceRequest` slots, `BackendRelationId`, witness `ViewRequirement`s, optimization IDs. Formula semantics come from `jolt-claims` at the prover, not inside the backend.
3. **Kernel granularity (stated in spec):** coarse (whole stage) / **medium (one batched sumcheck)** / fine (one round message). CPU `regular_batch.rs` kernel exists behind `SumcheckRegularBatchState`, but **no shared prover-side handler** factors the Fiat–Shamir loop out of stages.
4. **Acceptance oracle:** `jolt-verifier` primary; `jolt-core` parity via harness only.
5. **Performance over backend elegance:** CONTRACT.md says relation-specific coarse kernels are intentional; generic sparse kernel is reference-only.

**Gaps relative to this spec.**

- Still **no `BatchedSumcheckProver`** mirroring `BatchedSumcheckVerifier`; transcript loop lives in prover stages + assembly.
- Backend API is **relation-request-shaped** (good for GPU) but **not** the uniform `start / round_polynomials / bind / finish` oracle; dozens of per-relation request types instead.
- **Assembly/builder** layer adds complexity on top of stage `prove.rs`; current `prover-stack` dropped that for thinner `api.rs` + per-stage prove.

**Worth keeping from attempt 1.**

- `jolt-backends` **CONTRACT.md** invariants (prover owns FS, backends return slot-keyed results, optimization IDs for parity).
- Harness gates: `VerifierCorrectness` + `CorePerformanceParity` + microbenchmarks before wiring stages.
- `jolt-prover-model-crate.md` ownership table (verifier vs prover vs witness vs kernels).
- Medium-grain kernel concept aligns with `prove_sumcheck` as the one handler-owned batched-sumcheck entry point.

### 13.3 Attempt 2: Bolt (`jolt-v2/bolt`)

**Open PR stack (Markos).**

| PR | Branch | Content |
|----|--------|---------|
| [#1512](https://github.com/a16z/jolt/pull/1512) | `jolt-v2/jolt-witness` | Witness crate + modular helpers |
| [#1513](https://github.com/a16z/jolt/pull/1513) | `jolt-v2/bolt` | **`bolt` crate**: MLIR dialect, Jolt lowering, Rust artifact emission |
| [#1514](https://github.com/a16z/jolt/pull/1514) | `jolt-v2/generated-roles` | Checked-in generated `jolt-prover` / `jolt-verifier` role crates |
| [#1515](https://github.com/a16z/jolt/pull/1515) | `jolt-v2/equivalence` | **`jolt-equivalence`**: real-trace oracle vs `jolt-core` + tamper tests |

**Follow-on (Quang, same stack).** [#1523](https://github.com/a16z/jolt/pull/1523) `quang/bolt-stack` (closed): verifier refactor to **typed plans** + `bolt-verifier-runtime`; stacks on Markos's `jolt-v2/equivalence`. Docs: `crates/bolt/GOAL.md`, `VERIFIER_PROGRAM_REFACTOR_PLAN.md`, `PROVER_PROGRAM_REFACTOR_PLAN.md` (on `quang/bolt-stack`).

**Separate paper repo:** [github.com/markosg04/bolt](https://github.com/markosg04/bolt) (protocol compiler formal spec in LaTeX).

**Core architectural bets.**

1. **Compiler owns protocol facts.** MLIR → typed plans → thin generated Rust const data → kernels/runtime.
2. **Generated prover/verifier** checked into `crates/jolt-prover` and `crates/jolt-verifier` (not handwritten stage `prove.rs`).
3. **`jolt-kernels`:** coarse per-stage executors; generated Stage 4 is a static `STAGE4_PROGRAM_STEPS` table + `execute_stage4_program` (see `quang/bolt-stack:crates/jolt-prover/src/stages/stage4.rs`).
4. **Equivalence is the gate:** Bolt self-acceptance, transcript parity, splice into `jolt-core` proof, tamper rejection (`jolt-equivalence/README.md`).
5. **Verifier-first cleanup:** shrink ~21k LOC generated verifier to typed declarative plans; prover refactor explicitly **deferred** (`PROVER_PROGRAM_REFACTOR_PLAN.md`: "verifier remains the priority").

**Bolt Stage 4 shape (instructive).** Generated file declares:

```text
STAGE4_PROGRAM_STEPS = [
  transcript_squeeze(registers_gamma),
  transcript_absorb_bytes(ram_val_check_domain_separator),
  transcript_squeeze(ram_val_check_gamma),
  sumcheck_driver(stage4.sumcheck),
]
```

So Bolt already has a **prover program** abstraction at the stage level (`ProgramStepPlan`, `SumcheckDriverPlan`), but emission is compiler-owned strings/plan rows, not a shared Rust `prove_sumcheck` in `jolt-sumcheck`.

**Gaps relative to this spec.**

- No stable **handwritten** sumcheck handler crate; sumcheck driver logic is generated + `jolt-kernels`.
- **Heavy MLIR/toolchain dependency** (CI needs `llvm-config`; blocked locally in PR #1513 notes).
- Prover path still **stage-local plan types** (`jolt_kernels::stageN::*`); stringly symbols in plan rows.
- GPU integration path unclear: generated code + kernels vs cluster `bind_eval` proxy.

**Worth keeping from attempt 2.**

- **ProverProgram / ProverStep** naming and staged program tables (this spec's `ProverProgram` is the handwritten analogue).
- **Equivalence testing** mindset: byte-identical transcripts and round polynomials, not just verifier acceptance (extends attempt 1's `jolt-core` splice).
- **Declarative sumcheck batch plans** (`Stage4SumcheckBatchPlan`, instance rows) → maps to `BatchedSumcheckSpec` + `SumcheckInstance`.
- Split **generic runtime** vs **protocol-specific relations** (`bolt-verifier-runtime` pattern) → maps to `sumcheck-core` / `sumcheck-prover` / `jolt-claims`.

### 13.4 Attempt 3: `prover-stack/*` (current integration pin)

**Open PR stack (Markos).** Sequential branches `prover-stack/01-field-inline-tracing` … `prover-stack/11-akita-specs`; GPU work targets [**#1605**](https://github.com/a16z/jolt/pull/1605) `prover-stack/10-jolt-prover`.

**What it is.** A **restack** of handwritten `jolt-prover`, `jolt-witness`, extended `jolt-claims` formulas, wrapper/dory-assist/field-inline crates, without Bolt and without `audit-prep`'s `assembly.rs` / harness-first port machinery.

| vs audit-prep | vs Bolt |
|---------------|---------|
| Keeps `jolt-backends` request CONTRACT (largely same text) | Handwritten stages, not generated |
| Drops `assembly.rs` / `builder.rs` | No `jolt-equivalence` requirement |
| `api.rs` bounds **stage-specific trait bundle** on prove | No MLIR compiler |
| Stage `prove.rs` hand-rolled sumcheck loops (Stage 4 ~960–991) | `jolt-kernels` not in pin (CPU traits in `jolt-backends` only) |

**Backend surface today** (`jolt-prover/src/api.rs`):

```text
SumcheckBackend
+ RamReadWriteSumcheckBackend
+ Stage3SpartanSumcheckBackend
+ Stage4ReadWriteSumcheckBackend
+ Stage5ValueEvaluationSumcheckBackend
+ Stage6RegularBatchSumcheckBackend
+ BlindFoldProverBackend
```

Each stage trait exposes `materialize_*_state`, `evaluate_*_round`, `bind_*_state`, `output_*_state` (see `jolt-backends/src/traits.rs`). This is what `jolt-cuda` must implement today (mostly CPU delegate).

**Gaps relative to this spec (motivation for this document).**

- Same prover–verifier asymmetry as §1: **`BatchedSumcheckVerifier` without prover twin**; N copies of RLC + front-loaded batching.
- Prover imports **verifier helpers** for claim plumbing (`stage4_expected_outputs`, etc.).
- **Stage traits** instead of relation-keyed `SumcheckBackend` + canonical handler.
- GPU path must implement entire trait bundle, not a sealed round-polynomial oracle.

**Worth keeping from attempt 3.**

- Actually runs and is the **integration target** for jolt-cpp (`docs/jolt_cuda_integration.md`).
- `jolt-claims` formulas as source of truth for relations (registers, ram, stage6, …).
- Per-stage `prepare.rs` / `prove.rs` / typed outputs (become thin **StageBuilder** lowering to `ProverProgram`).
- Clear vs committed proof recorders per stage (→ `SumcheckProofRecorder` trait).

### 13.5 Comparison matrix

| Dimension | jolt-v2 infra | audit-prep | Bolt | prover-stack (now) | **This spec** |
|-----------|---------------|------------|------|-------------------|---------------|
| **Prover code** | none on main | handwritten + assembly | generated + kernels | handwritten stages | handwritten **programs** + shared handler |
| **Sumcheck IOP loop** | verifier only | per-stage + assembly | generated driver | per-stage `prove.rs` | **`prove_sumcheck` once** |
| **Relation identity** | `jolt-claims` (verifier) | `BackendRelationId` + claims | MLIR/plan symbols | `JoltRelationId` + stage traits | **`SumcheckRelation`** |
| **Backend API** | n/a | request/result families | `jolt-kernels` stage executors | stage-specific traits | **`SumcheckBackend` oracle** (3 traits) |
| **Proof assembly** | n/a | `assembly.rs` / `builder.rs` | generated | inline in stages | `ProvingContext` + recorders |
| **Correctness gate** | unit tests | harness vs `jolt-core` | `jolt-equivalence` vs core | e2e / verifier | **reference algebra** + equiv tests |
| **GPU story** | n/a | request API (plausible) | unclear | trait bundle → cuda delegate | **same oracle** as CPU |
| **Bolt dependency** | no | explicitly no | yes | no | no |
| **ZK / BlindFold** | n/a | first-class in spec | partial in stack | `BlindFoldBackend` | recorder trait extension |
| **Akita reuse** | poly/sumcheck crates | kernels medium-grain | generic Bolt IR goal | `11-akita-specs` branch | `sumcheck-core` generic |

### 13.6 How this proposal positions relative to prior work

```text
                    jolt-claims formulas
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   audit-prep          Bolt plans      prover-stack
   request API         generated       stage traits
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  ▼                 ▼
           SumcheckRelation   ProverProgram (steps)
                  │                 │
                  └────────┬────────┘
                           ▼
                  prove_sumcheck (handler)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Reference      CPU algebra   GPU cluster
        (dense)        (jolt-backends) (jolt-cuda)
```

- **From attempt 1:** request/result backend boundary, harness parity, prover owns Fiat–Shamir, `jolt-verifier` owns proof types. **Replace** dozens of request types + assembly with one handler + four-method oracle.
- **From attempt 2:** explicit **prover program** and **sumcheck driver** step; equivalence oracle discipline. **Do not** require MLIR/codegen for the near-term GPU path; optionally Bolt could *emit* `ProverProgram` later.
- **From attempt 3:** ship on `prover-stack` crates; migrate stage-by-stage starting at Stage 4; fold stage traits into private CPU/GPU algebra internals.

**Explicit non-goals inherited from prior lines:** no reintroduction of `assembly.rs`-scale proof builders; no blocking GPU integration on Bolt toolchain; no second full port via `jolt-core` harness before reference-algebra equiv tests exist (harness remains valuable as regression).

### 13.7 Where to read the old code

| Attempt | Inspect |
|---------|---------|
| **jolt-v2 merged infra** | `main` @ `jolt-sumcheck`, `jolt-openings`; PRs [#1362](https://github.com/a16z/jolt/pull/1362)–[#1450](https://github.com/a16z/jolt/pull/1450) |
| **audit-prep prover** | `git checkout refactor/audit-prep` → `specs/jolt-prover-model-crate.md`, `crates/jolt-prover/src/assembly.rs`, `crates/jolt-backends/src/sumcheck/` |
| **Bolt stack** | `origin/jolt-v2/bolt`, `origin/jolt-v2/equivalence`; `quang/bolt-stack` for verifier/runtime plans; `crates/bolt/PROVER_PROGRAM_REFACTOR_PLAN.md` |
| **Current pin** | `prover-stack/10-jolt-prover` ([#1605](https://github.com/a16z/jolt/pull/1605)); this repo's `third-party/jolt` submodule |
| **Bolt paper** | `../bolt/paper.pdf` |

Local worktree **`jolt-modular-crates-prover`** tracks `prover-stack/10-jolt-prover` for isolated development.

---

## References

### This integration

| Topic | Location |
|-------|----------|
| Modular stack integration | [`docs/jolt_cuda_integration.md`](jolt_cuda_integration.md) |
| `JoltRelationClaims` | `third-party/jolt/crates/jolt-claims/src/protocols/jolt/relation.rs` |
| Batched verifier | `third-party/jolt/crates/jolt-sumcheck/src/batched_verifier.rs` |
| Stage 4 hand-rolled loop | `third-party/jolt/crates/jolt-prover/src/stages/stage4/prove.rs` |
| Backend traits (current) | `third-party/jolt/crates/jolt-backends/src/traits.rs` |
| Sumcheck backend contract | `third-party/jolt/crates/jolt-backends/src/sumcheck/CONTRACT.md` |
| Legacy GPU sumcheck proxy | `rust-bindings/src/sumchecks/mod.rs` |
| Modular GPU delegate | `src/jolt-cuda/src/delegate/sumcheck/` |

### Prior attempts (Markos Georghiades, a16z/jolt)

| Attempt | Key links / paths |
|---------|-------------------|
| jolt-v2 scaffolding | [PR #1362](https://github.com/a16z/jolt/pull/1362) … [PR #1451](https://github.com/a16z/jolt/pull/1451) |
| audit-prep prover model | branch `refactor/audit-prep`, `specs/jolt-prover-model-crate.md` |
| Bolt emitter | [PR #1513](https://github.com/a16z/jolt/pull/1513), branch `jolt-v2/bolt` |
| Bolt equivalence | [PR #1515](https://github.com/a16z/jolt/pull/1515), `crates/jolt-equivalence/README.md` |
| Typed verifier on Bolt | [PR #1523](https://github.com/a16z/jolt/pull/1523) `quang/bolt-stack` (closed) |
| Current prover stack | [PR #1605](https://github.com/a16z/jolt/pull/1605) `prover-stack/10-jolt-prover` |
| Bolt formal spec (paper) | [github.com/markosg04/bolt](https://github.com/markosg04/bolt) |
