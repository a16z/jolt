# crates/tasks

Task specifications for the `jolt-ir` crate (RFC finding 12, spec §4.10).

## Task dependency graph

```
01-implement-jolt-ir-core       ✅ DONE
    │
    ├──► 02-backend-evaluate    ✅ DONE
    │       │
    │       └──► 07-testing-and-fuzz ✅ DONE (integration + fuzz + README + benches)
    │
    ├──► 03-backend-r1cs        ✅ DONE
    │
    ├──► 04-backend-lean        ✅ DONE
    │
    ├──► 05-backend-circuit     ✅ DONE
    │       │
    │       └──► 08-jolt-gnark-crate  ✅ DONE (GnarkEmitter + sanitize_go_name + 14 tests)
    │
    └──► 06-integrate-downstream ← Wire into jolt-zkvm + jolt-spartan
```

## Recommended execution order

1. **Task 01** — core IR: expr, builder, claim, visitor, normalize ✅
2. **Task 02** — evaluate backend (needs `jolt-field` dep) ✅
3. **Task 03** — R1CS backend (needed for BlindFold ZK mode) ✅
4. **Task 07** — hardening: integration tests, fuzz targets, README, benches ✅
5. **Task 04** — Lean4 code generation backend ✅
6. **Task 05** — Circuit transpilation backend (CircuitEmitter trait) ✅
7. **Task 08** — `jolt-gnark` crate: gnark `CircuitEmitter` implementation (port from PR #1322) ✅
8. **Task 06** — integrate into jolt-zkvm + jolt-spartan (migrate SumcheckInstanceParams)

## Architecture decision: downstream dependency scope

Both `jolt-zkvm` and `jolt-spartan` depend on `jolt-ir`. The sumcheck crate remains generic:

- **jolt-sumcheck** — generic sumcheck protocol. Does not touch claim formulas. No `jolt-ir` dependency.
- **jolt-spartan** — depends on `jolt-ir`. Uses `jolt-ir`'s R1CS types as its native constraint representation.
- **jolt-zkvm** — orchestrator. Defines all ~20 claim formulas as `ClaimDefinition`s using `jolt-ir`. Passes them to sumcheck/spartan as needed.

## Philosophy

The IR is the source of truth for claim-level expressions. Developers write each sumcheck claim formula once as an `Expr`. All backends derive from it. No hand-written parallel implementations.

The IR does NOT own verifier orchestration (transcript, sequencing, commitments). That stays as Rust code.
