# Spec: jolt-akita Crate

| Field | Value |
|-------|-------|
| Component | jolt-akita PCS adapter |
| Depends On | 01-opening-trait-system.md |
| Unlocks | PackedWitness, logical views, verifier config |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | draft |

## Scope

`jolt-akita` is the PCS adapter crate. It presents Akita as a Jolt-compatible
batch-opening scheme before the Jolt PIOP depends on concrete packed-view
protocol details.

In scope:

```text
- Akita setup/prover/verifier types
- commitment and proof payload types
- PackedWitness commit/prove adapter over a streaming source
- transcript binding rules
- field/claim-field mode
- BatchOpeningScheme implementation
```

Out of scope:

```text
- choosing the final Jolt PackedWitness layout
- defining logical Jolt views
- fused increment protocol
- advice packing policy
```

Assumptions:

```text
- Akita verifies openings of a proof-owned prefix-packed PackedWitness.
- TrustedAdvice, BytecodeChunk(i), and ProgramImageInit are verifier-bound
  precommitted objects and are not proved by merely opening W_pack.
- Akita lattice mode is transparent until a lattice hiding protocol is
  specified.
- Claim-field mode is explicit in setup/config and transcript binding.
- The Jolt-side physical layout is supplied as a PackedWitness layout and view
  formulas.
- Akita setup is parameterized by a maximum or exact packed dimension
  `D_setup`.
- The current LayerZero Akita public API is point-opening shaped; jolt-akita
  may need an adapter for packed linear views.
```

## Architecture

Crate responsibilities:

```text
jolt-akita:
  Akita setup
  Akita commitments
  Akita proof type
  Akita PackedWitness commit adapter
  Akita verifier adapter
  Akita transcript labels
  Akita field adapters
  BatchOpeningScheme implementation
```

Public types:

```text
AkitaCommitment:
  PCS commitment object exposed to jolt-verifier.

AkitaBatchProof:
  proof for a packed final-opening relation.

AkitaSetup:
  setup material parameterized by `D_setup`.

AkitaLayoutDigest:
  transcript-bound digest of PackedWitness layout.

AkitaViewFormula:
  PCS-facing formula for how a logical claim is represented physically.

AkitaPackedViewStatement:
  normalized packed-view statement verified by the Akita backend or adapter.

AkitaCommitInput:
  PackedWitness layout digest.
  PackedWitnessSource handle.
  D_pack.

AkitaProverHint:
  backend opening hints for the committed PackedWitness.
  no Jolt semantic data.
```

Commitment class:

```text
PackedWitness:
  the single proof-owned Akita commitment used by lattice mode.

Precommitted objects:
  TrustedAdvice, BytecodeChunk(i), and ProgramImageInit keep their original
  commitments and opening proofs outside the W_pack packed-view statement.
  Their prover inputs are original polynomials plus backend hints for those
  commitments, not PackedWitnessSource entries.

Rejected:
  separate main/aux proof-owned Akita commitments.
  treating a precommitted object packed into W_pack as an opening of its
  original commitment.
  non-Akita committed object path for a proof-owned lattice-supported object.
```

Adapter rule:

```text
jolt-akita may understand:
  packed dimensions
  one-hot alphabets
  view formulas
  Akita proof syntax
  backend point-opening limitations

jolt-akita must not understand:
  RISC-V instruction semantics
  RAM/register semantics
  bytecode table semantics
  advice trust semantics
```

Jolt-specific construction of packed sources, including trace rows, bytecode
rows, program-image words, and advice bytes, lives in the verifier/protocol
layer or prover-side PIOP code. `jolt-akita` only consumes the resulting
generic PackedWitnessSource plus view formulas and layout metadata.

Field mode:

```text
base-field mode:
  Jolt field == Akita claim field.

target:
  extension claim fields are explicit in type/config and transcript binding.
```

Field conversion contract:

```text
to_akita_claim(F_jolt) -> F_akita

Required:
  injective over all values that appear in claims.
  deterministic.
  transcript representation is canonical.

Rejected:
  implicit modular reduction between unrelated fields.
```

Setup contract:

```text
AkitaSetupKey:
  security parameter
  packed dimension D_setup
  field mode
  backend/version identifier

Verifier accepts proof only if:
  proof.setup_key == verifier.setup_key
  layout.D_pack <= setup_key.D_setup, if setup is universal/up-to-D_setup
  layout.D_pack == setup_key.D_setup, if setup is exact-D_setup
```

Batch-opening input:

```text
Given:
  PackedWitness layout digest
  PackedWitness commitment
  logical point
  physical view formulas
  logical claims

Verify:
  Akita proof establishes the claimed packed opening relation.
```

Precommitted opening input:

```text
Given:
  original precommitted commitment
  point for that object
  commitment-bound layout digest for that original object
  direct physical view or deterministic linear component expansion
  claimed value

Prove/verify:
  a separate opening proof against the original commitment.
  the direct native opening transcript binds the opened commitment's layout
  digest as authoritative, even if a higher-level Stage 8 wrapper statement
  carries the PackedWitness layout digest.

Rejected:
  using the PackedWitness layout digest or W_pack commitment as evidence for
  this original commitment.
  accepting a copied precommitted value in W_pack without an explicit binding
  protocol.
```

Commit/prove contract:

```text
commit_packed_witness:
  input = AkitaCommitInput.
  output = AkitaCommitment, AkitaProverHint.

prove_batch:
  input = BatchOpeningStatement, AkitaProverHint, PackedWitnessSource.
  output = AkitaBatchProof, logical coefficients.

verify_batch:
  input = BatchOpeningStatement, AkitaBatchProof.
  output = logical coefficients and verified statement digest.
```

`jolt-akita` owns backend-specific commitment hints. `jolt-verifier` sees only
commitments, setup keys, layout digests, and batch proofs.

Batch-opening normalization:

```text
Input logical claims:
  (opening_id, relation_id, point, claim, view_formula)

Akita statement:
  PackedWitness commitment handle
  packed dimension D_pack
  normalized view relation
  normalized claim vector
  backend-specific proof payload

Result:
  logical coefficients for original Jolt claims
  verified PCS statement digest
```

Failure behavior:

```text
layout digest mismatch:
  reject before proof verification if possible.

unsupported view formula:
  reject at statement validation.

unsupported field mode:
  reject at config validation.

dimension mismatch:
  reject before or during setup lookup.
```

Transcript:

```text
1. Akita setup/config identifier.
2. PackedWitness layout digest.
3. PackedWitness commitment.
4. logical opening point.
5. logical claims or ZK claim commitments.
6. Akita batch challenges.
```

Implementation plan:

```text
crate layout:
  crates/jolt-akita/src/lib.rs
  crates/jolt-akita/src/setup.rs
  crates/jolt-akita/src/commitment.rs
  crates/jolt-akita/src/proof.rs
  crates/jolt-akita/src/layout.rs
  crates/jolt-akita/src/views.rs
  crates/jolt-akita/src/transcript.rs
  crates/jolt-akita/src/field.rs
  crates/jolt-akita/src/backend.rs
  crates/jolt-akita/src/packed_witness.rs

backend.rs:
  thin wrapper around LayerZero-Labs/akita or local Akita dependency.
  isolates backend API churn from jolt-verifier.
  exposes whether native packed linear views are supported.

layout.rs:
  AkitaLayoutDigest
  PackedWitnessLayout
  PackedFamily
  D_pack
  global padding

views.rs:
  AkitaViewFormula
  DirectOneHot
  LinearDecode
  MaskedRelationHandle
  ByteLimbDecode
  FusedIncrementDecode

packed_witness.rs:
  PackedWitnessPoly
  streaming one-hot fact source
  commit_packed_witness adapter
  prover hint construction
  adapter from PackedWitness views to Akita backend statements

field.rs:
  Jolt field -> Akita claim field conversion.
```

Software decisions:

```text
dependency direction:
  jolt-verifier depends on jolt-akita only for lattice-family builds.
  jolt-akita depends on jolt-openings and field/transcript traits.
  jolt-akita does not depend on jolt-verifier stage internals.

backend selection:
  use the LayerZero Akita backend through a thin adapter module.
  statement/layout tests may use deterministic fixtures, but they must exercise
  the real backend or the same jolt-akita adapter path used by production.
  do not add a mock Akita backend.

setup storage:
  verifier setup stores AkitaSetupKey and backend verifier material.
  prover setup stores matching prover material.

proof type:
  AkitaBatchProof is opaque to jolt-verifier except serialization and
  transcript binding.

commitment hint ownership:
  prover hints are produced by jolt-akita commitment code.
  hints are keyed by PackedWitness layout digest and commitment digest.
  verifier payload never trusts or parses hints.
```

Implementation steps:

```text
step A:
  real LayerZero Akita backend or jolt-akita adapter accepts/rejects
  BatchOpening statements by proving and verifying deterministic fixture
  witnesses.

step B:
  real Akita backend verifies a simple direct one-hot PackedWitness opening.

step C:
  real Akita backend or jolt-akita adapter verifies a mixed direct/linear
  decoded statement.

step D:
  proof-owned advice and field-inline families are supported or rejected by
  explicit error; precommitted objects are rejected from the packed-view
  statement unless an explicit binding protocol is supplied.

step E:
  prover-facing verifier helpers accept separate precommitted opening inputs
  and produce one precommitted opening proof per verifier-built precommitted
  statement, in statement order.

step F:
  direct/native precommitted opening proofs accept the layout digest of their
  own opening statement and reject commitment, hint, dimension, or claim
  mismatches.

step G:
  fused-increment Stage 6 helper derives packed Inc byte/sign and BytecodeRa
  claims from PackedWitness, but receives StoreFlag/RdPresent bytecode
  component claims from the committed-bytecode source path. It recombines those
  components into the aggregate StoreFlag/RdPresent source claims used by the
  masked translation.
```

## Invariants

```text
- Statement dimension D_pack is derived from PackedWitness layout fields.
- Setup dimension D_setup is verifier-configured and checked against D_pack.
- A proof for one PackedWitness layout digest is invalid for another digest.
- The crate does not inspect Jolt semantic IDs except through supplied view
  formulas.
- Field conversion is injective for all Jolt values accepted by Akita.
- Transparent-only config rejects ZK proof mode.
- Setup key is transcript-bound before Akita proof challenges.
- Unsupported view formulas fail closed.
- The lattice packed-view statement uses exactly one PackedWitness commitment.
- Precommitted-object commitments are verified by their own opening statements,
  not by jolt-akita's W_pack proof.
- Akita Program::Committed support is not enough to bind every Jolt
  precommitted object. Jolt supplies explicit original commitment handles for
  TrustedAdvice, BytecodeChunk(i), and ProgramImageInit.
- Direct/native Akita commitments may carry the layout digest of their own
  opening statement. Only the packed witness commitment is required to match the
  packed setup layout digest.
- AkitaProverHint is bound to the same layout digest and commitment digest used
  by the batch proof.
- The adapter result cannot introduce new logical opening IDs.
- The same layout/proof bytes produce the same Akita opening statement on
  prover and verifier.
```

## Tests

Targeted tests:

```text
akita_layout_digest_is_bound:
  proof fails under a changed PackedWitness layout digest.

akita_wrong_dimension_rejects:
  verifier rejects setup/proof dimension mismatch.

akita_batch_opening_rejects_tampered_claim:
  altered logical claim fails.

akita_batch_opening_rejects_tampered_commitment:
  altered commitment fails.

akita_transparent_rejects_zk_mode:
  config rejects unsupported ZK mode.

akita_unsupported_view_formula_rejects:
  statement validation fails before backend verification.

akita_field_conversion_is_canonical:
  same Jolt field value serializes to one Akita claim representation.

akita_setup_key_is_bound:
  proof generated under one setup key fails under another.

akita_statement_has_single_packed_witness_commitment:
  adding a second Akita commitment handle is rejected.

akita_hint_layout_mismatch_rejects:
  prover hint generated for one PackedWitness layout cannot prove another.

akita_commit_source_streams_nonzero_facts:
  commit adapter consumes PackedWitnessSource without dense W_pack
  materialization.

akita_linear_decode_requires_backend_support_or_adapter:
  byte-limb decode view cannot be treated as an ordinary point opening.
```

## Performance

Expected:

```text
setup:
  depends on D_setup, the backend setup dimension.
  setup key reuse is allowed only if backend supports universal/up-to-D setup.

prover:
  proportional to emitted one-hot facts, packed-view reduction cost, and Akita
  proof cost.

verifier:
  independent of materialized packed witness size.
```

Rejected:

```text
- recomputing setup from witness data.
- deriving D_pack or D_setup from untrusted proof header alone.
- materializing padded dummy cells to satisfy Akita APIs.
- converting semantic Jolt IDs into backend-specific behavior inside
  jolt-akita.
- using extension-field claims without explicit config and transcript binding.
- choosing a common fixed one-hot width only because the current backend
  OneHotPoly API requires it.
```

## Questions

```text
1. What is the first supported Akita field mode?
2. Does Akita setup live in jolt-verifier preprocessing or a PCS-specific
   verifier setup payload?
3. Is setup exact-D or universal/up-to-D?
4. Can the LayerZero backend verify packed linear views directly, or should
   jolt-akita reduce them to backend-supported statements?
```

## References

```text
- https://github.com/a16z/hachi: upstream Hachi PCS.
- https://github.com/LayerZero-Labs/akita: Akita fork target.
- https://github.com/a16z/dory: Dory PCS contrast.
- 01-opening-trait-system.md: trait this crate implements.
```
