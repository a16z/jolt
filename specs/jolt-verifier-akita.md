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

A previous attempt exists as the local branch `feat/akita-protocol-integration`
(~35k insertions, ~18k in jolt-verifier). Its architecture notes
(`specs/akita/00-roadmap.md` on that branch) are a useful map of *what* must
happen; its code is superseded rails. Section "Dropped from the reference"
records what we deliberately do not carry over — the target for the verifier
work here is **~500–1000 non-test LOC**.

## Scope (phased)

```text
Phase A  batch-opening abstraction: stage 8 goes through BatchOpeningScheme on
         both the verifier and prover sides; Dory instantiates
         HomomorphicBatch; muldiv e2e green in both modes proves the
         abstraction is sound (this is the "checkable against dory" gate)
Phase B  lattice verifier orchestration, feature-gated `akita`:
         config/payload axis, stage swaps for the lattice relations, packed
         stage 8 via PackedBatch — scoped to FULL program mode, no advice
         (the packed lifecycle is then exactly one W: Ra columns + inc
         chunks + msb, every column with a relation leaf)
```

Out of scope (deferred, in dependency order):

```text
prover/witness lattice path (packed witness assembly + lattice relation
    proving in jolt-prover-legacy / jolt-witness) — required for Akita e2e;
    tracked separately, NOT re-implemented inside jolt-verifier
committed-program mode over Akita (precommitted W′, bytecode-lane /
    program-image reconstructions) and advice over Akita (byte-column
    reconstruction unified with AdviceBytesValidity) — the jolt-claims
    semantics exist (#1663); the verifier reduction that turns
    ReconstructionTerm lists into per-column claims lands with this
zk × lattice (rejected fail-closed until a lattice hiding layer exists)
field_inline × lattice
```

## Boundary Contract

| Crate | Owns | Must NOT contain |
|-------|------|------------------|
| `jolt-verifier` | stage sequencing, `ConcreteSumcheck` impls, statement assembly, config validation | relation algebra, packing/slot math, PCS internals, any prover |
| `jolt-claims` | relations, columns, `final_opening` map, reconstruction terms | — |
| `jolt-openings` | `BatchOpeningScheme` + both strategies, `PrefixPacking` | Jolt ids/semantics |
| `jolt-akita` / `jolt-dory` | PCS + native batch adapters | Jolt protocol shape |
| `jolt-prover-legacy` (phase A only here) | prove-side batch call, streaming RLC source | verifier logic |

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

- `JoltProof` and `verify` gain one type parameter
  `B: BatchOpeningScheme<Field = F>` (default `HomomorphicBatch<PCS>` keeps
  every existing call site compiling); `joint_opening_proof: PCS::Proof`
  becomes `B::Proof`. The `HomomorphicCommitment` bound moves off the
  top-level `verify` and into the `HomomorphicBatch` instantiation only.
- Stage 8 keeps its existing claim/scale/point assembly (that part is
  protocol geometry, not homomorphism) and ends in `B::verify_batch` with
  `Vec<VerifierOpeningClaim>` instead of the inline RLC.
- The prover calls `B::prove_batch`. Constraint: the current prover streams
  the RLC (it regenerates bytecode-derived polynomials on the fly and never
  materializes the joint polynomial). `HomomorphicBatch::Polynomials` is
  `Vec<&dyn MultilinearPoly>` combined through the lazy `RlcSource`, so the
  streaming property survives **iff** the prover's sources are exposed as
  `MultilinearPoly` views; if any source resists that shape, adapt the trait
  in jolt-openings rather than fork the prover path.
- Transcript note: `HomomorphicBatch` absorbs a typed statement
  (`VerifierRlcClaims`) where today's code appends a bare `rlc_claims` scalar
  list. Prover and verifier switch in the same commit; old proofs do not
  survive (none are stable artifacts yet).
- ZK mode: the clear path moves to `B`; the BlindFold path keeps its current
  machinery in phase A (its final opening is already committed-evaluation
  shaped), migrating to `ZkBatchOpeningScheme` only if it is a strict
  refactor. zk × packed stays rejected regardless.

Acceptance for phase A: `muldiv` e2e green in `host` and `host,zk`; no
`HomomorphicCommitment` bound anywhere outside the `HomomorphicBatch`
instantiation.

### Phase B — config and gating (`dory` and `akita` as equals)

Two-level gating, mirroring how `zk` works today:

- **Cargo features** `dory` and `akita` on jolt-verifier gate only optional
  dependencies and one instantiation module each
  (`pcs/dory.rs`, `pcs/akita.rs`: type aliases + a concrete
  `verify_dory`/`verify_akita` entry point). The core stays generic over
  `(PCS, B)`; both features can coexist in one build.
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

### Phase B — lattice stage orchestration

The relations land inside the existing stage batches; no new stage is
created. Each is a `ConcreteSumcheck` impl over its jolt-claims
`SymbolicSumcheck` (the same shape as every existing stage relation):

| Stage | Base mode | Lattice mode |
|---|---|---|
| 1–5 | unchanged | unchanged |
| 6 | `IncClaimReduction` | `IncVirtualization` — same slot, same four consumed claims, produces `FusedInc` + `OpFlags(Store)` |
| 6 | `Booleanity` | lattice `Booleanity` — same relation id, output fold extended over `UnsignedIncChunk(0..N)` + `UnsignedIncMsb` |
| 6 | `BytecodeReadRaf` (full mode) | + one val stage (`LATTICE_BYTECODE_VAL_STAGES`) consuming `OpFlags(Store)@IncVirtualization`; the verifier evaluates the store-flag val from public bytecode like the other five stages |
| 7 | `HammingWeightClaimReduction` | + `UnsignedIncChunkReconstruction` in the same batch — consumes the stage-6 chunk/msb/`FusedInc` openings, produces the chunk leaves |
| 8 | RLC batch (`HomomorphicBatch`) | packed statement + `PackedBatch` (below) |

Deriveds each impl must resolve (all point algebra, no new semantics):
`IncVirtualizationPublic::{Eq*}` (eq at the four consumer cycle points),
`UnsignedIncChunkReconstructionPublic::{EqBooleanityAddress,
IdentityAtAddress}` (eq/identity MLEs at bound address points).

### Phase B — packed stage 8

`stage8` grows a mode seam: one small trait (working name
`FinalOpeningBatch`), two impls, chosen by the same `B` type parameter phase A
introduced:

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

### Dropped from the reference (and why)

| Reference construct | Fate |
|---|---|
| model prover inside jolt-verifier (`akita_witness/openings/validity`, ~4k, re-runs stages 1–7 to reconstruct transcript state) | never — prover work happens in the prover crates, as its own phase |
| packed validity subprotocol + separate validity opening proof | dropped — validity is the lattice relations (argument above) |
| `PackingViewFormula` resolution in stage 8, per-view row points | dropped — `PackedBatch` takes claims at arbitrary points |
| layout/validity digests in the proof + FS | dropped — packing derives deterministically from the absorbed config/shape |
| `increment_mode: Dense/Separate/Fused` config enum | dropped — lattice mode is fused one-hot, period |
| per-field bespoke validation functions (~900 LOC) | replaced by one fail-closed config check before stage 1 |
| stage-6 "every claim is `Option`" shape plumbing | avoided — mode variants swap whole relation instances, they don't hollow out shared structs |

Kept from the reference (its genuinely good calls): config-in-proof absorbed
into FS with `deny_unknown_fields`; fail-closed on every invalid combination
(lattice×zk, lattice×advice, lattice×committed-program for now, payload/config
mismatch, `Packed` without the feature); the store-binding requirement; the
precommitted policy (when phase D lands, precommitted columns open against
the precommitted `W′`, never satisfiable through the per-proof `W`).

## LOC budget (non-test, jolt-verifier + prover seam)

```text
Phase A: stage-8 verifier rewiring        ~120
         prover-side B::prove_batch seam  ~100   (jolt-prover-legacy)
         generics/bounds threading         ~80
Phase B: config axis + payload + checks   ~150
         stage-6/7 lattice relation impls ~250   (4 ConcreteSumchecks + deriveds)
         read-raf store val stage          ~50
         packed stage-8 statement assembly ~100
                                    total ~850
```

If it trends past ~1,200 the design is wrong somewhere — stop and find the
missing abstraction in jolt-claims/jolt-openings instead of writing more
verifier code.

## Testing / Acceptance

- Phase A gate: `muldiv` e2e green under `host` and `host,zk` with Dory
  through `HomomorphicBatch` — no transcript drift between the halves.
- Phase B (no prover yet): `ConcreteSumcheck` impl tests in the house style
  (draw-recording transcript, expected-output evaluation against the
  symbolic relation); packed statement assembly tests (every
  `proof_packing` column receives exactly one leaf claim; config/payload
  fail-closed matrix — each invalid combination produces its error);
  `PackedBatch` round-trip against `AkitaScheme` with hand-built claims
  already exists in jolt-akita's tests.
- Full Akita e2e is the acceptance gate of the *prover* phase, not this one.

## Open questions

1. **One packed opening or two, later**: when the precommitted `W′` lands
   (phase D), per-proof `W` and `W′` are two `PackedBatch` calls; Akita's
   `AkitaNativeBatching` could open both at one shared point in a single
   backend proof if the two reduction sumchecks are co-batched. Decide when
   phase D is real; two calls are correct and simpler.
2. **Where the reconstruction reductions live** (phase D): standalone
   sumcheck over cell variables per reconstruction vs folding into the
   existing claim-reduction relations (jolt-claims spec §5 note for advice).
   Decide against real shapes when committed-program/advice come into scope.
3. **`ZkBatchOpeningScheme` adoption for BlindFold** (phase A follow-up):
   only if it is a strict refactor of the current dory ZK final opening.
