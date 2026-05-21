# Spec Index: Recursion Protocol Composition

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-19 |
| Status | draft |
| PR | TBD |

This recursion architecture is split across four focused specs. Each spec owns
one protocol axis or the selected-verifier composition layer.

## Specs

- [Field Inline Protocol](field-inline-protocol.md)
- [Dory Assist Protocol](dory-assist-protocol.md)
- [Wrapper Protocol And SNARK Backend](wrapper-protocol.md)
- [Selected Verifier Integration](selected-verifier-integration.md)

## Architecture Summary

The recursion architecture separates protocol contracts from prover
computation:

```text
jolt-claims:
  protocol facts, dimensions, IDs, claim formulas, opening specs

jolt-verifier:
  selected verifier schedule, proof-shape validation, transcript ordering

component crates:
  reusable native and R1CS encodings for sumcheck, openings, Hyrax, BlindFold

jolt-wrapper:
  selected-verifier R1CS assembly and SNARK backend adapters
```

The protocol axes are orthogonal:

```text
Field inline:
  native field instructions, FR register memory, FieldProduct, conversion rows

Dory assist:
  auxiliary proof for expensive Dory verifier work, with Hyrax dense opening

Wrapper:
  R1CS encoding of the selected verifier computation, proved by
  Spartan + HyperKZG first
```

The composition layer is `jolt-verifier`: it selects which protocol axes are
active, validates the proof shape, fixes transcript order, and exports the
selected verifier computation for wrapper assembly.

## Related Specs

- [jolt-prover model crate spec](jolt-prover-model-crate.md)
- [jolt-verifier model crate spec](jolt-verifier-model-crate.md)
- Recursion paper repo: <https://github.com/markosg04/recursion-paper>
