# Spec: Jolt Verifier — Batch-Opening Abstraction and the Akita/Lattice Path

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-07-02 |
| Status | draft |
| Branch | `feat/jolt-verifier-akita` (stacked on #1654, #1655, #1663) |

## Purpose

Make the Jolt verifier orchestrate two commitment paths as equals:

- **Homomorphic** (Dory, HyperKZG): many commitments, one RLC batch opening —
  what stage 8 hardcodes today.
- **Packed** (Akita): one packed witness commitment per lifecycle, one
  reduction-sumcheck batch opening — no homomorphism anywhere.

The verifier's only job is **orchestration**: sequence the relations
jolt-claims defines, thread claims between stages, and hand the terminal
claims to a batch-opening scheme jolt-openings defines. Everything
protocol-semantic already lives below this crate: `jolt-claims` names the
lattice relations, columns, and the final-opening map (#1663); `jolt-openings`
owns `BatchOpeningScheme`, `HomomorphicBatch`, `PackedBatch`, and
`PrefixPacking` (#1654); `jolt-akita` adapts the PCS (#1655). This spec adds
no new protocol semantics — where one seemed needed, the fix belongs in
jolt-claims first.

Prover and verifier land **in tandem in every phase** — each phase ends at a
real e2e gate, so coverage grows incrementally instead of arriving all at
once with a big-bang prover. The prover half extends `jolt-prover-legacy`
(never the verifier crate), mirroring its optimized sibling instances.

A previous attempt exists as the local branch `feat/akita-protocol-integration`
(~35k insertions, ~18k in jolt-verifier). Its architecture notes
(`specs/akita/00-roadmap.md` on that branch) are a useful map of *what* must
happen; its code is superseded rails. Section "Dropped from the reference"
records what we deliberately do not carry over — the target is **~500–1,000
non-test LOC on the verifier side** (the prover side budgets separately,
sized by its optimized siblings).

## Scope (phased — prover and verifier land together in every phase)

Each phase ships both halves so it ends at a real e2e gate; incremental
testability is a design requirement, not an afterthought. The prover side of
the lattice semantics is a handful of additional sumchecks shaped like
existing ones — extended in `jolt-prover-legacy` by mirroring sibling
instances, keeping the optimized prover code paths so nothing gets rewritten
later. Restructuring jolt-prover-legacy where that makes hosting both paths
cleaner is explicitly allowed.

```text
Phase A  batch-opening abstraction, both sides: stage 8 goes through
         BatchOpeningScheme in the verifier and the prover; Dory instantiates
         HomomorphicBatch.
         GATE: muldiv e2e green in host and host,zk ("checkable against dory")
Phase B  lattice path, both sides, feature-gated `akita`, scoped to FULL
         program mode and no advice (the packed lifecycle is exactly one W:
         Ra columns + inc chunks + msb, every column with a relation leaf):
         verifier — config axis, stage swaps, packed stage 8 via PackedBatch;
         prover  — packed witness assembly, four lattice sumcheck instances,
         packed stage-8 prove.
         GATE: muldiv e2e green over Akita (new feature-gated test variant)
Phase C  committed-program + advice over Akita, both sides: precommitted
         objects and their reconstructions (ReconstructionTerm lists become
         claim-reduction legs), advice byte column unified with
         AdviceBytesValidity, native-batch composition of W_jolt with the
         precommitted commitment objects.
         GATE: Akita parity with the committed-program and advice e2e tests
```

Out of scope (not planned for this integration):

```text
zk × lattice — there is no zk path in the Akita/lattice version of Jolt for
    now; rejected fail-closed (BlindFold itself stays Dory-only; its final
    opening may ride ZkBatchOpeningScheme for uniformity, see phase A)
field_inline × lattice
```

## Boundary Contract

| Crate | Owns | Must NOT contain |
|-------|------|------------------|
| `jolt-verifier` | stage sequencing, `ConcreteSumcheck` impls, statement assembly, config validation | relation algebra, packing/slot math, PCS internals, any prover |
| `jolt-claims` | relations, columns, `final_opening` map, reconstruction terms | — |
| `jolt-openings` | `BatchOpeningScheme` + both strategies, `PrefixPacking` | Jolt ids/semantics |
| `jolt-akita` / `jolt-dory` | PCS + native batch adapters | Jolt protocol shape |
| `jolt-prover-legacy` (every phase) | prove-side batch call + streaming RLC source (A); packed witness assembly, lattice sumcheck instances, packed stage-8 prove (B/C) — **owns the witness path entirely**; jolt-witness is not used in this integration for now | verifier logic |

## Design

### Phase A — one batch-opening seam

Today the homomorphism assumption lives in three places:

1. `verifier.rs` top-level bound `PCS::Output: HomomorphicCommitment<F>`;
2. `stages/stage8/verify.rs` + `final_openings.rs`: hand-rolled statement
   absorption → `challenge_scalar_powers` → joint claim/commitment →
   `PCS::verify`;
3. `jolt-prover-legacy/src/zkvm/prover.rs` (~2250): the mirrored prove side
   (`rlc_claims` absorption, `DoryOpeningState`, `RLCStreamingData`).

`HomomorphicBatch::{prove,verify}_batch` already internalizes (2)'s whole
sequence behind `BatchOpeningScheme`. The change:

- There is no verifier-side mode trait (as built): the seam is
  `jolt_openings::BatchOpeningScheme` itself. Stage 8 is generic over
  `B: BatchOpeningScheme<…, Statement = Vec<VerifierOpeningClaim<…>>>` and
  ends in `B::verify_batch`; entry points pin the strategy explicitly
  (`stage8::verify::<…, HomomorphicBatch<PCS>>` — the packed entry point pins
  the packed strategy in phase B). `JoltProof` carries the batch proof as a
  plain defaulted type parameter (`JointOpeningProof = PCS::Proof`), so
  stages 1–7 and the proof model stay free of capability bounds
  (`AdditivelyHomomorphic`/`HomomorphicCommitment` live on the
  `HomomorphicBatch` impl in jolt-openings). `config.rs` keeps only the
  runtime axes (`JoltProtocolConfig`: zk × commitment). Both axes are
  compile-time: `verify` takes no mode parameter — the build's `zk` feature
  fixes `JOLT_VERIFIER_CONFIG`, and a proof self-describing another config is
  rejected fail-closed.
- Stage 8 keeps its existing claim/scale/point assembly (that part is
  protocol geometry, not homomorphism) and ends in `M::verify_final` with
  `Vec<VerifierOpeningClaim>`; the inline RLC (absorb → powers → combine →
  `PCS::verify`) is deleted, byte-identical transcript preserved
  (`VerifierRlcClaims` absorption equals the old `rlc_claims` sequence).
  `Stage8ClearOutput` sheds its unconsumed `joint_claim`/`joint_commitment`/
  `constraint_coefficients` fields.
- Prover side (as built): jolt-prover-legacy is a parallel trait world (its
  own legacy `CommitmentScheme`), so the literal `B::prove_batch` call is not
  meaningful there; instead the transcript-critical block (claims absorption
  → `challenge_scalar_powers` → joint claim) is cut into
  `zkvm/final_opening.rs::homomorphic_batch_challenges`, documented as the
  prove half of the mode seam and required byte-identical to
  `HomomorphicBatch::verify_batch`'s statement absorption. The streaming RLC
  (regenerating bytecode-derived polynomials, never materializing the joint
  polynomial) is untouched and non-negotiable. Phase B adds the packed prove
  beside this seam; if unifying onto jolt-openings' trait ever pays,
  **contorting `HomomorphicBatch` and the GATs is the sanctioned move**.
- Transcript note (better than predicted): `VerifierRlcClaims`'s absorption
  is byte-identical to the old `rlc_claims` sequence, so phase A changes the
  transcript not at all — existing proofs keep verifying, which is why the
  swap could land verifier-first against unchanged prover output.
- ZK mode: the clear path moves to `B::verify_batch`; the BlindFold final
  opening **may** move to `B::verify_batch_zk` (`ZkBatchOpeningScheme`) for
  uniformity — stage 8 then has exactly two flavors, clear and zk, both
  through `B`. This is safe precisely because zk × lattice never runs:
  Akita's `ZkBatchOpeningScheme` impl exists but is unreachable behind the
  fail-closed rejection, so the migration only ever exercises the Dory path
  the `host,zk` gate covers.

Acceptance for phase A: `muldiv` e2e green in `host` and `host,zk`, plus the
jolt-verifier soundness/tampering suite against real prover proofs. The
clear final opening carries no homomorphism requirement of its own — those
bounds survive only in `HomomorphicBatch`'s jolt-openings impl and stage 8's
(homomorphic-only) ZK arm, which the packed mode never reaches.

### Phase B — config and gating (`dory` and `akita` as equals)

Two-level gating, mirroring how `zk` works today:

- **No PCS knowledge in jolt-verifier at all** (revised in review): the
  crate exposes two generic entry points — `verify` (homomorphic bounds) and
  `verify_packed` (plain `CommitmentScheme` bounds, `M = LatticeJolt`) — and
  the concrete instantiation (Dory/Fr vs Akita/fp128) happens in the
  consuming crate behind *its* cfg feature. No `dory`/`akita` features, no
  optional PCS deps, no per-PCS modules; fail-closed selection is per entry
  point (each validates the proof's self-described commitment axis against
  what that entry point is).
- **Runtime config in the proof** (kept from the reference — it is what makes
  tampering testable): `JoltProtocolConfig` gains
  `commitment: CommitmentConfig { Homomorphic, Packed }`. It is absorbed into
  the Fiat-Shamir preamble and validated fail-closed:
  `Packed` requires the `akita`-instantiated entry point, full program mode,
  no advice, and `zk: Transparent`; the declared `CommitmentConfig` must
  equal the instantiation's mode (self-description only — the payload shape
  itself is fixed by the type, see below). Every mismatch is a distinct
  `VerifierError`, checked before stage 1.
- **Commitment payload**: no enum. The proof is already generic over `B`
  (`joint_opening_proof: B::Proof`), so the commitments ride the same axis:
  the stage-8 seam trait carries `type Commitments: AppendToTranscript` —
  `JoltCommitments<C>` for the homomorphic impl, bare `C` for the packed one
  (a single packed commitment is just the degenerate case). A mode/payload
  mismatch is unrepresentable at the type level, which deletes the runtime
  payload check the reference needed for its runtime-dispatched enum. The FS
  preamble absorbs `Commitments` exactly where the per-polynomial
  commitments are absorbed today; `serde(deny_unknown_fields)` applies to
  whatever structs remain.
- **Field coupling**: selecting Akita selects the field — the whole PIOP runs
  over `AkitaField` (fp128). `verify` is already generic over `F`; the
  `akita` module pins `F = AkitaField`. No packing/layout digests travel in
  the proof: the packing is a pure function of the (absorbed) config and
  shape on both sides, and `PackedBatch` absorbs the statement it verifies.

### Phase B — mode-typed stage claims (compile time, no enums, no Options)

Four claim groups change shape between modes: the stage-6 inc reduction
(`IncClaimReductionOutputClaims` vs `IncVirtualizationOutputClaims`), the
stage-6 booleanity outputs (base vs lattice, chunk/msb columns added), the
bytecode read-raf val-stage count (5 vs 6 — a shape change only in committed
mode, so it rides as a validation const until phase C), and stage 7
(+`ChunkReconstructionOutputClaims`). As built: **jolt-claims owns the type
family** — `JoltCommitmentMode` (`protocols/jolt/mode.rs`) carries the three
varying claim-group GATs over the cell type plus `BYTECODE_VAL_STAGES`, with
unit impls `BaseJolt`/`LatticeJolt`. The verifier threads one defaulted
`M: JoltCommitmentMode = BaseJolt` through `Stage6OutputClaims`/
`Stage7OutputClaims`, the stage output carriers, `ClearProofClaims`,
`JoltProofClaims`, and `JoltProof` (which also gains a plain defaulted
`Commitments` parameter for the packed single-commitment payload); the base
mode's placeholder (`NoOutputs`) serializes to zero bytes, so the base wire
format is unchanged. Mode-specific consumers (stage-6/7 builders, stage-8
batch assembly, everything zk) pin `M` concretely, so no field access goes
through the trait. A proof for the wrong mode fails to
deserialize/typecheck rather than being runtime-rejected; the reference's
`Option`-field hollowing and two-variant enums are both off the table.

### Phase B — lattice stage orchestration

The relations land inside the existing stage batches; no new stage is
created. Each is a `ConcreteSumcheck` impl over its jolt-claims
`SymbolicSumcheck` (the same shape as every existing stage relation):

| Stage | Base mode | Lattice mode |
|---|---|---|
| 1–5 | unchanged | unchanged |
| 6a | — | `IncVirtualization` — same four consumed claims as the base inc reduction (stage-2/4/5 outputs, all upstream of 6a), produces `FusedInc` + `OpFlags(Store)`. Lives in the 6a batch (decided in review): a batch's input claims absorb at batch start, so the store claim can only feed the 6b read-raf input if it is produced strictly earlier — 6a outputs are visible to 6b inputs. Heterogeneous round counts are what batch round-offsets are for. |
| 6 | `Booleanity` | lattice `Booleanity` — same relation id, output fold extended over `UnsignedIncChunk(0..N)` + `UnsignedIncMsb` |
| 6b | `BytecodeReadRaf` (full mode) | + one val stage (`LATTICE_BYTECODE_VAL_STAGES`) — the store claim enters the lattice 6b read-raf **input** fold (not 6a's, where the five upstream stages enter); the verifier evaluates the store-flag val from public bytecode like the other five stages |
| 7 | `HammingWeightClaimReduction` | + `UnsignedIncChunkReconstruction` in the same batch — consumes the stage-6 chunk/msb/`FusedInc` openings, produces the chunk leaves |
| 8 | RLC batch (`HomomorphicBatch`) | packed statement + `PackedBatch` (below) |

Deriveds each impl must resolve (all point algebra, no new semantics):
`IncVirtualizationPublic::{Eq*}` (eq at the four consumer cycle points),
`UnsignedIncChunkReconstructionPublic::{EqBooleanityAddress,
IdentityAtAddress}` (eq/identity MLEs at bound address points).

### Phase B — packed stage 8

Stage 8 already dispatches through `BatchOpeningScheme` (phase A); the packed
path is a second impl chosen by the same `B` type parameter. Two enabling
changes: the zk arm (Dory/BlindFold-only) moves out of the shared stage-8 fn
into the homomorphic entry point's path, so the shared core drops its
`AdditivelyHomomorphic`/`ZkOpeningScheme` bounds; and the statement type goes
with `B` (`B::Statement`):

- `HomomorphicBatch<PCS>`: today's path — unified point, embedding scales,
  `Vec<VerifierOpeningClaim>`.
- `PackedBatch<PCS, LatticeColumn>`: walk
  `lattice::proof_packing(shape)` columns; for each, take the leaf claim
  named by `lattice::final_opening` (`Packed { leaf }` → the stage-6/7 output
  claim **at its own point**); assemble `PrefixPackedStatement`; call
  `verify_batch`.

Two properties of the landed `PackedBatch` make this section small:

1. **Arbitrary independent points.** The batch reduction sumcheck handles
   claims at mutually independent points (`E(z) = Σ αᵢ·eq(z, bᵢ‖γᵢ)`), so the
   verifier does **no** point unification for the packed path: the msb leaf
   stays at its stage-6 point, Ra leaves at the stage-7 hamming point, chunk
   leaves at the reconstruction point. The reference branch's
   suffix-compatibility choreography ("carry packed view row points",
   physical manifests) is unnecessary. What `prepare_statement` does enforce
   — exactly one claim per column and every column claimed — is satisfied by
   construction: `final_opening` assigns each column exactly one leaf, and
   the phase-B scope (full mode, no advice) has no reconstruction-only
   columns.
2. **No packed validity subprotocol.** The reference ran a dedicated
   validity sumcheck over `W` (cell booleanity, one-hot row sums, …) plus a
   second packed opening. On the current rails, validity for prover-supplied
   columns *is* the lattice relations (lattice Booleanity, the
   reconstruction's hamming legs), already scheduled above. Dummy/padding
   cells of `W` need no constraint at all: every claim reads only its slot's
   subcube through the eq kernel, so garbage outside claimed slots never
   enters any verified statement.

### Phase B — prover side (jolt-prover-legacy)

Landed in tandem with the verifier work above so phase B ends at a real
Akita e2e. Four pieces, each mirroring an optimized sibling that already
exists — copy the shape, keep the fast paths, do not write naive versions
that get rewritten later:

- **Packed witness assembly + commit.** The one-hot `Ra` cell streams the
  prover already materializes (the one-hot polynomial machinery —
  `RaPolynomial`/shared-eq infrastructure) are placed into `W` sparsely via
  `PrefixSlot::packed_index`; nothing is densified. The new columns derive
  from the same trace pass that builds `RdInc`/`RamInc` today: per cycle,
  the fused delta `2^64 + δ` yields the chunk symbols and msb bit —
  including the padding encoding (`δ = 0` ⇒ msb hot, chunks at symbol 0,
  per the jolt-claims padding invariant). One Akita commit of `W` through
  jolt-akita's sparse one-hot fast path replaces today's ~10+ per-polynomial
  commits.
- **Four prover sumcheck instances**, each copied from its nearest sibling
  and driven by the same jolt-claims `SymbolicSumcheck` the verifier uses:
  `IncVirtualization` (from the inc claim-reduction instance — same four
  consumed claims, different output fold), lattice `Booleanity` (the base
  booleanity instance with the chunk/msb columns joining its fold —
  chunk columns bind through the same one-hot representations as `Ra`, so
  the optimized bind/eval paths apply unchanged), `UnsignedIncChunkReconstruction`
  (hamming-reduction-shaped, `log_k_chunk` rounds over the chunk columns),
  and the read-raf store val stage (one more gamma stage in the existing
  bytecode read-raf instance; full-mode vals come from public bytecode on
  both sides). zk × lattice is rejected, so these instances carry no
  BlindFold constraint plumbing.
- **Packed stage-8 prove — with a sparse reduction prover.** The same
  `final_opening` map the verifier walks assembles the
  `PrefixPackedStatement` prover-side; `PackedBatch::prove_batch` runs the
  reduction sumcheck over `W` and the single Akita opening (with the
  commit-time hint). **The landed reduction prover densifies both operands**
  (`polynomial.to_dense()` plus a dense `2^packed_num_vars` selector table)
  — acceptable for unit tests, impossible at trace scale, since
  `packed_num_vars` is the log of the *sum of all one-hot column sizes*.
  Phase B replaces it with a sparse path in jolt-openings, in the classic
  optimized style of jolt-prover-legacy's one-hot sumcheck code: the
  selector `E(z) = Σ αᵢ·eq(z, bᵢ‖γᵢ)` is evaluated per-slot through its eq
  factorization (never materialized), and `W` binds through its one-hot
  column structure. No densifying anywhere on the prod path; the dense route
  survives only as the test oracle. Contorting `PackedBatch`'s GATs (e.g.
  the `Polynomials` source type) to make this natural is sanctioned, same as
  phase A.
- **Preamble symmetry.** The prover absorbs the `CommitmentConfig` and the
  single packed commitment exactly where the verifier expects them.

Where hosting both paths cleanly wants prover restructuring (e.g. carving
the commit/stage-8 sections behind the same mode seam instead of `if`s
around Dory-specific state), restructure — jolt-prover-legacy is not
load-bearing legacy for this integration.

### Dropped from the reference (and why)

| Reference construct | Fate |
|---|---|
| model prover inside jolt-verifier (`akita_witness/openings/validity`, ~4k, re-runs stages 1–7 to reconstruct transcript state) | never — prover work happens in the prover crates, in tandem within each phase |
| packed validity subprotocol + separate validity opening proof | dropped — validity is the lattice relations (argument above) |
| `PackingViewFormula` resolution in stage 8, per-view row points | dropped — `PackedBatch` takes claims at arbitrary points |
| layout/validity digests in the proof + FS | dropped — packing derives deterministically from the absorbed config/shape |
| `increment_mode: Dense/Separate/Fused` config enum | dropped — lattice mode is fused one-hot, period |
| per-field bespoke validation functions (~900 LOC) | replaced by one fail-closed config check before stage 1 |
| stage-6 "every claim is `Option`" shape plumbing | avoided — mode variants swap whole relation instances, they don't hollow out shared structs |

Kept from the reference (its genuinely good calls): config-in-proof absorbed
into FS with `deny_unknown_fields`; fail-closed on every invalid combination
(lattice×zk, lattice×advice and lattice×committed-program until phase C,
config/instantiation mismatch, `Packed` without the feature); the
store-binding requirement; the precommitted policy (phase C: precommitted
objects open against their own commitments, never satisfiable through the
per-proof `W`).

## LOC budget (non-test)

```text
jolt-verifier
  Phase A: stage-8 rewiring + generics threading      ~200
  Phase B: config axis + fail-closed checks           ~150
           stage-6/7 lattice ConcreteSumchecks        ~250
           read-raf store val stage                    ~50
           packed stage-8 statement assembly          ~100
                                    verifier total    ~750
jolt-prover-legacy
  Phase A: B::prove_batch seam                        ~100
  Phase B: packed witness assembly + commit           ~250
           four lattice sumcheck instances            ~350
           packed stage-8 prove + preamble            ~150
                                      prover total    ~850
jolt-openings
  Phase A: GAT adjustments for streaming sources       ~50
  Phase B: sparse one-hot packed reduction prover     ~300
```

If the verifier side trends past ~1,200 the design is wrong somewhere —
stop and find the missing abstraction in jolt-claims/jolt-openings instead
of writing more verifier code. The prover side is allowed to cost what the
optimized siblings cost, but each instance should read like its sibling.

## Testing / Acceptance

- Phase A gate: `muldiv` e2e green under `host` and `host,zk` with Dory
  through `HomomorphicBatch` — no transcript drift between the halves.
- Phase B, incrementally as pieces land: each prover instance tested against
  its `ConcreteSumcheck` expected-output (house style, draw-recording
  transcript); packed-witness assembly tested against the jolt-claims
  `lattice_semantics` identities (slot readback, chunk/msb encoding incl.
  padding); statement assembly (every `proof_packing` column receives
  exactly one leaf claim); the config/instantiation fail-closed matrix.
- Phase B gate: **`muldiv` e2e green over Akita** — a feature-gated e2e
  variant in jolt-prover-legacy (`F = AkitaField`, full mode, no advice),
  run alongside the Dory variants from then on.
- Phase C gate: Akita parity with the committed-program and advice e2e
  tests.

## Open questions

1. **Composing `W_jolt` with the precommitted objects (phase C)** — decided
   direction: trusted/precommitted objects stay separate commitment objects,
   and the final opening **black-box/native-batches `W_jolt` together with
   those objects** via `AkitaNativeBatching` (one backend proof over the
   commitment group at a common point). That requires the reduction step to
   land `W_jolt`'s claims and the precommitted claims on one shared point —
   extend jolt-openings' packed reduction to claim sets spanning several
   commitments (or co-batch per-commitment reductions sharing the bound
   point). Two separate `PackedBatch` calls remain the correct simple
   fallback while that lands. Sub-question deferred with it: whether the
   precommitted side is one packed `W′` or several individual objects.
2. **Reconstruction reductions (phase C)** — decided direction: fold them
   into the existing claim-reduction relations wherever the shapes allow
   (the natural form, and what jolt-claims spec §5 already prescribes for
   advice); a standalone cell-variable sumcheck only where folding is
   genuinely impossible.
