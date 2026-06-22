Here is a list of my full review comments that I want to resolve for the akita jolt integration:

### jolt-akita
- the testing for jolt-akita is not as nice as jolt-dory. we should split up the integration test file into many files and have way more rigorous testing of the akita pcs itself as an independent scheme, mirroring more how dory testing layout works.
- in general, jolt-akita should mirror much more the way that say jolt-dory or jolt-hyperkzg is structured. Like the actual file layout, how the impls are organized, where the newtypes go, etc.
- the packed semantics leak into jolt-akita. it should not know about packing, views (physical, virtual), layouts, whatever, jolt-akita is just a wrapper crate to conform to our jolt pcs interfaces to open polynomials. that's it. we need to rip out all these packing things and jolt-akita needs to align more with jolt-dory. we can keep zk impl as one that is unimplemented/panic so that runtime configs reflect this in future comments of this general review here I will discuss where these packed semantics and whatnot should live, they should def not live in jolt-akita though.

other nits: the Akita dependencies in Cargo.toml should obviously stay pinned to a concrete upstream commit/rev, but that vendor/revision detail should not leak into Rust code or public adapter abstractions (such as LAYERZERO_AKITA_REV).
- otherwise it's nice to have all the type aliases / newtypes so we can kind of of follow hte same pattern as how we bridge the other pcs impls.
- in general i just hope to simplify this crate greatly by avoiding so much over engineered packing/view stuff that doesn't belong there at all.

### packing architecture / ownership
- packing is currently spread across too many crates and the boundaries are muddy. we should make the ownership explicit and enforce it:
- `jolt-openings` should own the generic packing API: canonical layout, family refs, rank/unrank, layout digest, dummy padding, packed witness/source abstraction if needed, and the generic reduction from packed claims to a native PCS opening.
- `jolt-claims` should own Jolt semantics only: which Jolt polynomials/facts exist, which logical claims are decoded views, one-hot increment semantics, fused increment semantics, validity requirements, byte decoding formulas, etc.
- `jolt-verifier` should orchestrate: derive the Jolt packing layout from protocol config, lower Jolt claim formulas into generic `jolt-openings` packing statements, and choose the packed opening path when the lattice/Akita config requires it.
- `jolt-akita` should be a native Akita PCS adapter only. it should not define or export packed layouts, packed view catalogs, Jolt family ids, witness packing semantics, or Jolt-specific validity logic.
- the canonical packing bijection must have one source of truth. ordering, rank/unrank, layout digest, minimal dimension, and dummy padding should all live together and be tested together. avoid reimplementing or special-casing this in Akita or verifier code.
- padding should be explicit and non-semantic. the packing API should make clear that we pack the actual cells in canonical order and only pad to the final hypercube dimension when unavoidable.

### naming / api simplification
- use one vocabulary: `Packing`. avoid public API names like `PackedLinear*`. if the view happens to be a linear combination, that should be a detail or enum variant, not the root namespace for the whole mechanism.
- suggested direction: `PackingLayout`, `PackingFamily`, `PackingAddress`, `PackingTerm`, `PackingView`, `PackingBatch<PCS>`, `PackingReductionProof`. keep the naming boring and consistent.
- avoid multiple overlapping adapters (`PackedCombine`, `PackedLinearBatch`, `AkitaPackedScheme`) unless each has a real non-test role. if an adapter is test-only or transitional, keep it private or under tests.
- avoid Akita-prefixed names for generic Jolt packing work. functions like `derive_akita_packed_witness_layout` and `akita_packed_view_formula` read like Akita owns the semantics; if they are deriving Jolt packing, name them as Jolt/generic packing.
- limit indirection. the handoff should make it easy to trace: Jolt logical claim -> packing view -> generic packing reduction -> native PCS opening. if a trait/newtype does not materially vary behavior across PCS implementations, prefer a direct function or struct.
- keep PIOP semantics separate from PCS transport. one-hotting inc, fused increment, byte decoding, and validity requirements belong in Jolt claims/verifier logic; the PCS backend should only know how to commit/open/verify.

### concrete cleanup checks
- after the refactor, `jolt-akita` exports should look much closer to `jolt-dory` / `jolt-hyperkzg`: scheme, setup/config, commitment, proof, hint, and native adapter helpers. no generic packing vocabulary should be exported from `jolt-akita`.
- `rg "PackedWitnessLayout|PackedView|PackedFamilyId" crates/jolt-akita/src` should be empty or limited to a very small private compatibility shim during migration.
- `jolt-openings` tests should cover the generic packing bijection and reduction independent of Akita.
- `jolt-akita` tests should cover Akita as a PCS independent of Jolt packing, with any packed end-to-end tests moved to the generic opening/verifier integration layer.

### diff readability / auditability
- split the huge files before this is reviewable. `jolt-verifier/src/akita.rs`, `stage8/lattice.rs`, `stage8/verify.rs`, `stage6/verify.rs`, `verifier.rs`, `jolt-openings/src/schemes.rs`, `jolt-openings/src/packed_linear.rs`, and `jolt-akita/tests/batch_opening.rs` are all doing too much.
- `jolt-openings/src/schemes.rs` should mostly be trait/API surface. move homomorphic batching, packing batch/reduction, transcript binding helpers, and tests into focused modules.
- `jolt-openings/src/packed_linear.rs` currently mixes public types, layout traits, prover reduction, verifier reduction, selector evaluation, sparse proving, serialization, transcript binding, and tests. split this by audit boundary: API, layout/rank logic, reduction prover, reduction verifier, serialization/transcript, tests.
- `jolt-verifier/src/stages/stage8/lattice.rs` mixes layout derivation, validity requirements, validity sumcheck verification, physical view lowering, config validation, formula lowering, and tests. split it into smaller modules with names matching the proof concepts.
- `jolt-verifier/src/akita.rs` is especially hard to audit because it combines witness construction, commitment helpers, stage6 claim derivation, stage8 opening proving, packed validity proving, proof payload validation, and low-level validity evaluators. those should not all live in one file.
- avoid putting large integration fixtures and helper builders inside production modules. move bulky tests/fixtures out of `verifier.rs`, `stage8/verify.rs`, `stage8/lattice.rs`, and `jolt-openings/src/schemes.rs` where possible.
- remove runtime type checks such as `Any::downcast_ref::<AkitaCommitment>()` in verifier logic. layout digest / commitment metadata should be available through typed payloads or generic traits, not dynamic Akita-specific downcasts.
- reduce public API added only for proof assembly/testing. functions that return mid-verifier transcripts or stage8 batch statements should be crate-private or moved behind a clear prover-helper module, not broad verifier API surface.
- do not make Dory tests carry packed-view semantics. Dory should test Dory. generic packing/batching tests should live in `jolt-openings`; Akita/Jolt packed integration tests should live in verifier/integration tests.
- avoid duplicating validity semantics between verifier-side code and Akita prover helpers. derive one shared validity plan/factor list and have both prover and verifier consume it. duplicated logic here is a soundness audit risk.
- make transcript binding locations easy to audit. right now labels/config/proof payload/opening statements are spread across openings, verifier, stage8, and Akita backend code. centralize the transcript order for packing-related statements or document the exact sequence in one place.
- collapse duplicate config state. `PcsFamilyFlags` should probably be an enum rather than two booleans that can be invalid. `protocol.zk` and `lattice.zk` are redundant. packed witness booleans like `field_rd_inc_family` / advice family flags should either be derived from the canonical layout/validity manifest or removed.
- rename proof/config errors from Akita-specific to lattice/packing-specific where the error is not actually native-Akita-specific. reserve `Akita*` error names for native PCS adapter failures.
- avoid generic proof model names like `CommitmentPayload::Akita` if the payload is really "lattice packed witness commitment payload". this should not make future lattice/hash-based PCS work look like Akita is the protocol.
- document the invariants of `BatchOpeningStatement`: when `logical_point` differs from `pcs_point`, how `scale` composes with view coefficients, whether all claims must share one commitment for packing, and what `layout_digest` binds. these invariants are currently implicit and easy to misuse.
- avoid storing the same digest at multiple layers unless necessary. `BatchOpeningStatement.layout_digest`, each packed view's digest, setup layout digest, proof payload digest, and commitment digest all need a clear reason or this becomes a tamper-matrix problem.
- `PackedCombine` is misleading because it binds packed metadata but does not prove the packed view relation. remove it, make it test-only, or rename it so nobody can mistake it for the real packing reduction.
- keep the naming flat and boring. use `Packing` consistently rather than alternating between lattice, Akita, packed, packed-linear, physical view, view catalog, packed witness, etc. each extra synonym makes review harder.
- any broad trait-bound changes outside Akita/openings, such as relaxing `Field` or adding serde bounds to unrelated PCS implementations, should be isolated and justified. otherwise it is hard to tell whether they are real integration requirements or mechanical fallout.

### jolt-verifier diff budget
- the verifier diff should be much smaller. target roughly `3k` production lines in `jolt-verifier`, excluding tests. `4k` may be acceptable if the excess is narrowly justified, but the current shape is far above what the verifier should need.
- current tracked verifier diff is already about `+7.6k/-0.4k` excluding `crates/jolt-verifier/tests`, and that does not include the new untracked verifier files. including the current untracked verifier files puts the verifier-side addition closer to `17k` lines. this is not a reasonable review surface.
- the budget can work because the verifier should orchestrate protocols, not implement packing, Akita proving, or duplicated validity semantics.
- expected verifier production budget:
- config / proof / error surface: `500-800` lines. this should cover lattice mode selection, proof payload shape, commitment payload bindings, and clear error variants. avoid duplicate booleans and Akita-specific payload names where the concept is generic lattice/packing.
- stage6 fused increment verifier changes: `500-800` lines. this should only verify the extra fused increment relations and expose their output claims. do not put packing, Akita, or final-opening logic here.
- stage8 final-opening orchestration: `800-1100` lines. this should build the logical opening manifest, request the physical packing view lowering, partition packed vs precommitted/direct openings, and call the generic batch-opening verifier. it should not define the packing bijection or validity formulas.
- top-level verifier wiring: `400-700` lines. this should validate config/payload binding, absorb transcript commitments, invoke optional packed validity verification, and call the existing stages. avoid separate public replay APIs unless they are clearly prover-support APIs.
- misc compatibility / manifest glue: `300-500` lines. this should be small adapters only, not large semantic logic.
- total target: about `2.5k-3.9k` production lines. if it is much above this, something probably belongs in `jolt-openings`, `jolt-claims`, `jolt-akita`, or tests.
- what must move out of verifier to hit the budget:
- generic packing layout, family/address/rank/unrank, layout digest, dummy padding, and packed witness/source abstractions should move to `jolt-openings`.
- generic packing reduction from logical packed claims to a native PCS opening should move to `jolt-openings`.
- Jolt lattice semantic formulas, validity requirements, one-hot/fused increment view formulas, and byte-decoding formulas should live in `jolt-claims`.
- Akita native setup/commit/open/prove helpers should live in `jolt-akita`.
- Akita prover assembly helpers should not be in core verifier modules. if they are needed, put them behind a narrow prover-support module/crate and keep them out of verifier review surface.
- large fixtures and integration tests should live in test modules/files, not production modules.
- what should remain in verifier:
- config validation and proof payload checks.
- stage6 PIOP verification changes needed by lattice mode.
- stage8 orchestration from logical Jolt claims to generic packing statements.
- verification of the packed validity proof by invoking shared validity plans and generic opening APIs.
- transcript-order glue, with the packing-related transcript sequence centralized or documented.
- concrete acceptance check: after refactor, running `git diff --numstat origin/main -- crates/jolt-verifier ':!crates/jolt-verifier/tests'` should show verifier production additions in the low thousands, not high thousands. any file over ~`1k` new production lines should have a strong reason and a clear module boundary.
