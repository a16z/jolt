# crates/tasks

Task specifications for the `jolt-ir` crate (RFC finding 12, spec §4.10).

## Task dependency graph

```
01-implement-jolt-ir-core       ✅ DONE
    │
    ├──► 02-backend-evaluate    ✅ DONE
    │       │
    │       └──► 07-testing-and-fuzz (integration + fuzz + README + benches)
    │
    ├──► 03-backend-r1cs        ✅ DONE
    │
    ├──► 04-backend-lean        — deferred (no consumer ready)
    │
    ├──► 05-backend-circuit     — deferred (no consumer ready)
    │
    └──► 06-integrate-downstream ← Wire into jolt-zkvm only. Last step.
```

## Recommended execution order

1. **Task 01** — core IR: expr, builder, claim, visitor, normalize ✅
2. **Task 02** — evaluate backend (needs `jolt-field` dep) ✅
3. **Task 03** — R1CS backend (needed for BlindFold ZK mode) ✅
4. **Task 07** — hardening: integration tests, fuzz targets, README, benches
5. **Task 06** — integrate into jolt-zkvm (migrate SumcheckInstanceParams)
6. **Tasks 04, 05** — Lean4 and circuit backends (defer until those consumers are ready)

## Architecture decision: downstream dependency scope

Only `jolt-zkvm` depends on `jolt-ir`. The sumcheck and spartan crates remain generic:

- **jolt-sumcheck** — generic sumcheck protocol. Does not touch claim formulas. No `jolt-ir` dependency.
- **jolt-spartan** — generic Spartan IOP. Claim formulas for outer/shift/product sumchecks are provided by jolt-zkvm, not defined inside spartan. No `jolt-ir` dependency.
- **jolt-zkvm** — orchestrator. Defines all ~20 claim formulas as `ClaimDefinition`s using `jolt-ir`. Passes them to sumcheck/spartan as needed.

This keeps the protocol crates reusable and the IR dependency contained to the one crate that actually defines claim formulas.

## Philosophy

The IR is the source of truth for claim-level expressions. Developers write each sumcheck claim formula once as an `Expr`. All backends derive from it. No hand-written parallel implementations.

The IR does NOT own verifier orchestration (transcript, sequencing, commitments). That stays as Rust code.
