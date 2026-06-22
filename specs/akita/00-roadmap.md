# Spec: Akita Roadmap

| Field | Value |
|-------|-------|
| Component | Akita integration roadmap |
| Depends On | none |
| Unlocks | opening traits, jolt-akita, packing, verifier config |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | implemented (transparent verifier/protocol target) |

## Scope

North star:

```text
Jolt supports Akita as a lattice PCS in the modular verifier path without
making the Jolt PIOP depend on additive homomorphic batching.
```

The target architecture has four implementation surfaces:

```text
jolt-claims:
  owns the base logical PIOP objects.
  adds a lattice extension, analogous to other optional extensions such as
  field-inline.

  The lattice extension names the extra PIOP semantics needed by Akita:
    prefix-packed fact families
    logical-to-physical view formulas
    one-hot increment decode and validity relations
    non-increment one-hot/decode policies
    translation relations for non-linear packed views

  Base Dory Jolt does not depend on these lattice relations.

jolt-verifier:
  owns PCS-family selection and selected PCS config.

  TOML defaults to:
    curve = true
    lattice = false

  When curve is selected:
    use the ordinary curve-PCS logical view and opening flow.
    Dory is the current curve PCS target.

  When lattice is selected:
    enable the jolt-claims lattice extension.
    derive the PackedWitness layout.
    resolve final logical claims through lattice views.
    reject unsupported feature combinations before Stage 8.

jolt-openings:
  generalizes additive homomorphism into a batch-opening interface.

  Generic batching implementations:
    additive-homomorphic batching:
      Dory keeps the current RLC joint-claim path.

    packed-view batching:
      Akita proves a batch of logical claims through packed physical views.

PCS adapters:
  Dory opens ordinary committed multilinear polynomials.
  Akita opens one proof-owned packed physical witness through view relations
  supplied by the lattice extension and verifier resolver.
  Precommitted objects are opened against their own commitments.
```

```text
- Dory remains compatible with the current Stage 8 RLC flow.
- Akita may replace RLC batching with packed-opening relations.
- The verifier sees one generic batch-opening interface.
- The PIOP can keep logical claims over Jolt objects.
- The proof-owned commit path uses one Akita-efficient PackedWitness object.
- Verifier/preprocessing commitments are not silently replaced by W_pack.
```

Precommitted opening policy:

```text
- TrustedAdvice, BytecodeChunk(i), ProgramImageInit, and committed-bytecode
  source facts derived from BytecodeChunk(i) are verifier-bound
  precommitted objects.
- They must be opened against their original precommitted commitments through
  separate direct/native opening statements.
- They must not be added to the proof-owned W_pack packed-view statement, even
  if the prover also copies their values into W_pack.
- Akita backend Program::Committed support is not this binding by itself. It
  may commit backend bytecode material, but it does not replace the Jolt
  precommitted TrustedAdvice, BytecodeChunk(i), or ProgramImageInit commitment
  handles.
- A future packed-precommitted optimization is allowed only if it proves an
  explicit binding between the packed copy and the original precommitted
  commitment before the verifier accepts the packed value as opened.
```

Precommitted opening contract:

```text
- Stage 8 partitions openings by commitment class before calling the Akita
  packed-view path.
- Proof-owned openings enter the single W_pack packed-view statement.
- Precommitted openings enter a separate direct/native opening manifest keyed by
  the original Jolt commitment handles.
- Each precommitted manifest entry has its own proof material after deterministic
  component expansion. Compatible entries may be grouped only by a native
  direct-opening API that preserves every original commitment handle.
- A proof for W_pack, or a backend Program::Committed bytecode handle, is never
  accepted as the proof for TrustedAdvice, BytecodeChunk(i), ProgramImageInit,
  StoreFlag, or RdPresent source openings.
```

In scope:

```text
- committed bytecode as background interface expected from modular Jolt.
- a jolt-claims lattice extension for packed-view PIOP semantics.
- a PCS batching trait that does not require additive homomorphism.
- a jolt-akita adapter crate.
- prefix-packed PackedWitness geometry.
- translation from logical claims to packed physical views.
- fused one-hot base increment representation.
- field-inline, advice, and precommitted-data opening policy.
- verifier config, proof payload, and rejection tests.
```

Out of scope:

```text
- implementing committed bytecode.
- changing the core Jolt sumcheck protocol for Dory.
- treating Akita-specific packed views as logical Jolt polynomials.
- adding a second proof-owned Akita PackedWitness commitment.
- supporting Akita ZK before a BlindFold-like lattice hiding layer exists.
```

## Committed Bytecode Background

The Akita integration assumes the modular Jolt stack already exposes committed
program bytecode. Akita does not implement that protocol; it consumes its public
interface.

Expected committed-program interface:

```text
ProgramMode:
  Full | Committed

trace facts:
  BytecodeRa(d)
    trace-indexed one-hot PC/address witness facts.
    These remain main-trace facts and are not the program image commitment.

preprocessing commitments:
  BytecodeChunk(i)
    committed chunk of the expanded bytecode table.

    Current core lane layout:
      rs1 one-hot lanes.
      rs2 one-hot lanes.
      rd one-hot lanes.
      unexpanded_pc scalar lane.
      imm scalar lane.
      circuit flag lanes.
      instruction flag lanes.
      lookup table selector lanes.
      raf flag lane.

  ProgramImageInit
    committed initial program-image words over a preprocessing domain.
    Words are little-endian u64 words.

staged committed-program objects:
  BytecodeValStage(i)
    verifier-hidden staged bytecode value claims when ProgramMode::Committed.

  BytecodeClaimReduction
    reduces staged bytecode claims to BytecodeChunk(i) openings.

  ProgramImageInitContributionRw
    Stage 4 contribution used by committed program-image checking.

  ProgramImageClaimReduction
    reduces program-image claims to ProgramImageInit openings.

stage outputs:
  BytecodeChunk(i) final logical openings when ProgramMode::Committed.
  ProgramImageInit final logical opening when ProgramMode::Committed.

verifier preprocessing:
  program identity fields.
  bytecode_chunk_count.
  BytecodeChunk(i) commitments.
  ProgramImageInit commitment.
  no full bytecode values used directly for committed-mode claim evaluation.
```

Interface requirements for Akita:

```text
- BytecodeRa(d), BytecodeChunk(i), and ProgramImageInit are distinct logical
  objects.
- Committed-program opening IDs and relation IDs are deterministic.
- Committed-program domains may differ from the trace domain.
- Stage 8 receives committed-program final logical claims through the same
  logical opening manifest as other Jolt claims.
- In lattice mode, committed bytecode chunks and ProgramImageInit remain
  precommitted physical objects and require separate openings against their
  original commitments.
- TrustedAdvice also remains a precommitted physical object. It is never
  discharged by copying its values into W_pack.
- Separate opening means an opening statement/proof keyed by the original Jolt
  precommitment handle. Akita's proof-owned PackedWitness commitment, and any
  Akita backend Program::Committed bytecode handle, cannot replace that handle.
- Akita cannot batch these precommitted objects with W_pack through the target
  packed-view path because that would require post-commitment additive
  combination across unrelated commitments.
- Increment selection reuses committed bytecode facts:
    Ram source = Store circuit flag.
    Rd source = sum of rd one-hot lanes.
  No extra committed-bytecode selector family is required for base RAM/RD.
  These source facts are not proof-owned W_pack families; they require
  BytecodeChunk(i) openings or a future bound precommitted packed view.
```

## Architecture

The integration proceeds by separating logical protocol objects from physical
commitment objects.

```text
logical object:
  A Jolt polynomial or relation named by jolt-claims.
  Example: RamRa(d), BytecodeRa(d), RdInc, RamInc, TrustedAdvice.

physical object:
  A committed PCS object.
  Example: a Dory polynomial commitment or Akita PackedWitness commitment.

view:
  A deterministic expression that recovers a logical object evaluation from one
  or more physical objects.

precommitted object:
  A physical object already bound by verifier preprocessing or trusted advice.
  Example: TrustedAdvice, BytecodeChunk(i), ProgramImageInit.
  Akita W_pack cannot serve as its opening unless an explicit future binding
  protocol proves equivalence to the original commitment.
```

The Dory path keeps the existing shape:

```text
Given logical claims:
  p_0(r) = y_0, ..., p_{m-1}(r) = y_{m-1}

Sample batching challenge gamma.

Open:
  P_joint(X) = sum_i gamma^i p_i(X)

Check:
  P_joint(r) = sum_i gamma^i y_i
```

The Akita path uses the same logical claims but partitions them by physical
commitment class:

```text
Given logical claims:
  p_0(r) = y_0, ..., p_{m-1}(r) = y_{m-1}

Resolve each proof-owned p_i into a physical view:
  p_i(r) = View_i(W_pack; r)

Ask Akita to prove the batched view relation:
  { View_i(W_pack; r) = y_i } for proof-owned packed claims

For precommitted p_j:
  open p_j(r) = y_j against its original commitment.

Stage 8 remains generic:
  it builds opening statements and dispatches to the selected batch-opening
  implementation. Dory may keep one homomorphic batch; Akita uses one
  packed-view batch plus separate precommitted openings.
```

For Akita, separate precommitted openings are mandatory for verifier-bound
data. TrustedAdvice, BytecodeChunk(i), ProgramImageInit, and bytecode source
views used by fused increments cannot be satisfied by copying their values into
W_pack. A future packed-precommitted optimization must include an explicit
binding proof to the original commitments.

This means the lattice proof has two different opening classes:

```text
proof-owned class:
  one PackedWitness commitment W_pack.
  one packed-view opening batch for claims whose physical facts live in W_pack.

precommitted class:
  one or more direct opening statements against the original trusted-advice,
  bytecode-chunk, or program-image commitments.
  these statements are not merged into the W_pack packed-view batch.
```

Do not treat an Akita backend `Program::Committed` mode as this binding by
itself. Jolt-level precommitted objects are the original trusted-advice,
bytecode-chunk, and program-image commitment handles exposed by preprocessing.
Lattice verification must keep those handles outside `W_pack` and verify a
separate opening proof for each required precommitted statement.

Module ownership:

```text
jolt-claims:
  owns base logical polynomial IDs and claim formulas.
  consumes committed-program IDs exposed by the modular bytecode commitment
  interface.
  owns the lattice extension that adds packed fact IDs, decode relations,
  validity relations, view formulas, and translation relation IDs.

jolt-openings:
  owns the generic batch-opening statement/output types.
  provides additive-homomorphic batching for Dory-compatible PCS modes.
  provides the packed-view batching interface Akita implements.

jolt-akita:
  owns Akita setup, commitment/proof types, PackedWitness layouts, view
  formulas, field/ring conversion, and backend dispatch.

jolt-verifier:
  owns PCS-family selection, verifier config, Stage 8 statement construction,
  logical-to-physical resolution, transcript binding, proof payload dispatch,
  and rejection order.
  enables the jolt-claims lattice extension only when the lattice family is
  selected.

prover witness path:
  owns streaming the physical PackedWitness described by the layout.
```

Milestone order:

```text
1. opening trait refactor with Dory compatibility.
2. jolt-akita crate with LayerZero Akita backend fixtures.
3. PackedWitness layout planner with no verifier integration.
4. Stage 8 logical-to-physical view resolver.
5. fused base increment facts and validity.
6. advice, field-inline, bytecode, and program-image packing.
7. Akita verifier config and proof payload.
```

Milestone details:

```text
01-opening-trait-system:
  Stage 8 depends on batch opening, not additive homomorphism.
  Dory implements the trait through the current RLC joint-claim path.

02-jolt-akita-crate:
  Akita is introduced as a PCS adapter before Jolt PIOP packing changes depend
  on it. Statement, layout, and proof fixtures use the real LayerZero Akita
  backend or the jolt-akita adapter around that backend; no mock Akita backend
  is part of the target stack.

03-prefix-packed-witness:
  Proof-owned lattice-visible facts are packed into one Akita-friendly
  PackedWitness. The layout records fact families, alphabets, row domains,
  offsets, and dimension effects. Precommitted facts are excluded unless a
  future binding protocol preserves their original commitments.

04-logical-views-and-translation:
  Final logical openings remain Jolt claims.
  Proof-owned claims resolve to view formulas over W_pack.
  Precommitted claims resolve to separate openings against their original
  commitments.

05-onehot-increments:
  RdInc and RamInc move from dense logical commitments to fused byte/sign
  facts with explicit decode and validity relations. Source facts are derived
  from committed bytecode lanes through the separate BytecodeChunk opening path.

06-advice-and-aux-onehotting:
  Advice, field-inline, precommitted program data, and future blinding data get
  canonical opening policies. Only proof-owned data enters PackedWitness.
  Precommitted data keeps original commitment handles and separate opening
  proofs.

07-verifier-config-and-tests:
  The verifier exposes curve/lattice PCS-family config, rejects unsupported
  combinations, binds the PackedWitness layout, and dispatches Stage 8 by
  selected PCS.

08-fused-increment-piop:
  Proposed replacement for the earlier byte/sign fused increment surface in 05.
  Akita uses UnsignedIncChunk(j) plus a size-T UnsignedIncMsb, adds an
  Akita-only Stage 5i/Stage 5 increment proof, and leaves Dory unchanged.
```

## Implementation Evidence

The current implementation covers the transparent verifier/protocol target:

```text
jolt-openings:
  generic BatchOpeningScheme and ZkBatchOpeningScheme interfaces.
  additive-homomorphic batching for the Dory-style path.
  PackedCombine-style packed-view batching without requiring additive
  homomorphism.

jolt-akita:
  real LayerZero Akita backend adapter.
  exact-D setup binding.
  PackedWitness layout, sparse source, and packed-view reduction path.
  separate direct/native openings for precommitted objects.

jolt-claims:
  lattice formulas, packed fact IDs, validity requirements, and fused-increment
  translation/source-link formulas.

jolt-verifier:
  PCS-family config and payload dispatch.
  PackedWitness layout derivation and validation.
  Stage 6 fused-increment claim plumbing.
  Stage 8 logical-to-physical partitioning into one W_pack packed-view batch
  plus separate precommitted direct-opening statements.
```

Verified gates for this branch:

```text
cargo nextest run -p jolt-akita --cargo-quiet
cargo nextest run -p jolt-verifier --cargo-quiet --features akita,field-inline
cargo nextest run -p jolt-openings -p jolt-claims --cargo-quiet
cargo clippy -p jolt-akita -q --all-targets -- -D warnings
cargo clippy -p jolt-verifier -q --all-targets --features akita,field-inline -- -D warnings
cargo clippy -p jolt-openings -p jolt-claims -q --all-targets --features field-inline -- -D warnings
```

This status does not claim a `jolt-core` Akita prover end-to-end path or Akita
ZK hiding. Those remain separate integration work.

## Invariants

```text
- Dory remains a first-class modular PCS path.
- The Dory verifier result is unchanged when Akita config is disabled.
- Stage 8 never requires PCS additive homomorphism.
- Akita does not expose W_pack as a logical Jolt polynomial.
- Logical claims are named before physical packing is chosen.
- PackedWitness layouts are transcript-bound before any Akita opening proof is
  accepted.
- The same verifier config determines preprocessing, proof payload decoding,
  Stage 8 dispatch, and rejection behavior.
- BytecodeRa trace facts and program bytecode commitments remain separate
  objects.
- TrustedAdvice, BytecodeChunk(i), and ProgramImageInit are precommitted
  objects; lattice verification must open them against their original
  commitments, not only through W_pack or a backend bytecode-commit mode.
- StoreFlag/RdPresent source facts used by fused increments are
  precommitted-bytecode facts, so they follow the BytecodeChunk opening path
  unless an explicit bound precommitted packed view is added.
- Any one-hot committed logical value has an explicit decode relation and an
  explicit validity relation.
- Base increment source is canonical:
    Ram = committed Store flag.
    Rd = committed rd one-hot presence.
    Store * rd_present = 0.
- Zero increments use canonical sign:
    magnitude = 0 implies sign = 0.
- Packing never silently hides a power-of-two dimension increase.
```

## Tests

The implementation stack should add targeted tests in these classes:

```text
opening_trait_dory_compat:
  Dory Stage 8 produces the same final opening claims through the new batch
  interface.

akita_batch_statement_without_homomorphic_combine:
  Akita receives physical view statements without relying on additive commitment
  combination.

packed_witness_layout_dimension_accounting:
  fact-family counts, offsets, dummy cells, and MLE dimension are deterministic.

logical_view_resolution:
  every final logical claim is resolved to a supported physical view or rejected.

increment_decode_validity:
  fused one-hot RdInc/RamInc views decode to the same logical values used by
  the PIOP.

aux_policy_rejection:
  unsupported advice, field-inline, precommitted-program, or blinding layouts
  fail before opening verification.

verifier_config_rejection:
  invalid PCS/config/proof payload combinations are rejected deterministically.
```

## Performance

The roadmap targets Akita's commit-time cost model:

```text
Dory:
  pays for ordinary committed field-element polynomials and batches final
  openings by homomorphic RLC.

Akita:
  should commit to one-hot packed objects where the ring arithmetic is cheaper
  than committing full field elements.
```

The main performance constraint is dimension control.

```text
For each family f:
  cells_f = rows_f * limbs_f * alphabet_f

PackedWitness dimension:
  D_pack = ceil_log2(sum_f cells_f)

Global dummy cells:
  2^D_pack - sum_f cells_f

Crossing a power-of-two setup threshold changes Akita setup and memory costs.
The layout must expose D_pack and global padding.
```

Target performance:

```text
- Prefix-packed W_pack is deterministic and streamable.
- Base increment magnitude facts are fused for RamInc/RdInc.
- Proof-owned advice data uses canonical byte-limb one-hot representations.
- Precommitted bytecode/program-image data keeps separate commitment openings.
- Field-inline has a separate packed family inside W_pack.
- The verifier opens one Akita packed-view relation for proof-owned same-point
  claims whenever the PIOP has reduced claims to a common r, plus separate
  openings for any precommitted claims.
```

## Resolved Decisions And Open Questions

```text
resolved:
  jolt-akita uses the LayerZero Akita backend directly and adds a generic
  packed-view reduction/adapter for Jolt PackedWitness statements.
  jolt-akita currently uses exact-D setup: D_setup must equal the derived
  PackedWitness dimension D_pack for the accepted proof.
  proof-owned field-inline FieldRdInc values use canonical field-byte families.

open:
  whether a future field-inline path can use a smaller structured encoding than
  canonical field bytes without changing the PackedWitness invariants.
  whether a future backend API can prove common byte-decode views more cheaply
  than the current packed-linear adapter representation.
```

## References

```text
- https://github.com/a16z/hachi: upstream Hachi PCS implementation.
- https://github.com/LayerZero-Labs/akita: Akita fork target.
- https://github.com/a16z/jolt/pull/1565: committed bytecode spec.
- https://github.com/a16z/jolt/pull/1571: Stage 6 split.
- https://github.com/a16z/jolt/pull/1572: precommitted reductions.
- https://github.com/a16z/jolt/pull/1583: committed bytecode foundation.
- https://github.com/a16z/jolt/pull/1584: committed reductions integration.
- https://github.com/a16z/jolt/blob/main/specs/1344-committed-bytecode-program-image.md:
  upstream committed-program spec.
- 01-opening-trait-system.md
- 02-jolt-akita-crate.md
- 03-prefix-packed-witness.md
- 04-logical-views-and-translation.md
- 05-onehot-increments.md
- 06-advice-and-aux-onehotting.md
- 07-verifier-config-and-tests.md
- 08-fused-increment-piop.md
```
