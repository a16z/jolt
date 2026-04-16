---
name: jolt-equivalence debugging sandbox
description: jolt-equivalence crate is a debugging sandbox for fuzzing data and comparing polynomial constructions against the jolt-core reference
type: reference
originSessionId: f0719e20-9c07-4478-bf0f-7ad6f4e1a5cf
---
jolt-equivalence (crates/jolt-equivalence/) is a debugging sandbox. Key capabilities:
- `transcript_divergence` test: operation-by-operation comparison of Fiat-Shamir transcripts between jolt-core and jolt-zkvm
- `CheckpointTranscript`: records every transcript op for comparison
- Per-stage cross-system tests (stage1 through stage7)
- Targeted diagnostic tests: booleanity_debug, hash_debug, uniskip_azbz, product_left_debug, r1cs_satisfaction
- Can freely add new debug test files to fuzz specific polynomial constructions or claim formulas against jolt-core

Use it to verify transcript parity after any refactoring. Add tests freely when debugging specific divergences.
