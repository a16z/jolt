# crates/tasks

Task specifications for the modular crate workspace.

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
    └──► 06-integrate-downstream ← Wire into jolt-zkvm (jolt-spartan part done)

13-extract-split-eq-and-ra-poly   ← Extract GruenSplitEqPolynomial, RaPolynomial, mles_product_sum to jolt-poly
    │
    └──► 14-jolt-spartan-uniform  ← Uniform Spartan variant + streaming sumcheck for Jolt zkVM
             │
             └──► S1/S2 stages in jolt-zkvm (unblocked)

09-blindfold-relaxed-spartan    ✅ DONE (RelaxedR1CS trait, prove_relaxed/verify_relaxed, 4 tests)
    │
    ├──► 10-blindfold-verifier-r1cs ✅ DONE (StageConfig, BakedPublicInputs, build/assign, 10 tests)
    │
    └──► 11-blindfold-nova-folding  ✅ DONE (fold_instances/witnesses, cross_term, sample_random, 14 tests)
            │
            └──► 12-blindfold-protocol  ✅ DONE (BlindFoldProver/Verifier, e2e + negative tests, 8 tests)
```

## Recommended execution order

1. **Task 01** — core IR: expr, builder, claim, visitor, normalize ✅
2. **Task 02** — evaluate backend (needs `jolt-field` dep) ✅
3. **Task 03** — R1CS backend (needed for BlindFold ZK mode) ✅
4. **Task 07** — hardening: integration tests, fuzz targets, README, benches ✅
5. **Task 04** — Lean4 code generation backend ✅
6. **Task 05** — Circuit transpilation backend (CircuitEmitter trait) ✅
7. **Task 08** — `jolt-gnark` crate: gnark `CircuitEmitter` implementation (port from PR #1322) ✅
8. **Task 09** — relaxed R1CS in jolt-spartan ✅
9. **Task 10** — verifier R1CS construction in jolt-blindfold ✅
10. **Task 11** — Nova folding in jolt-blindfold ✅
11. **Task 12** — BlindFold protocol orchestrator ✅
12. **Task 06** — integrate IR into jolt-zkvm + jolt-spartan (spartan done, zkvm pending)
13. **Task 13** — extract GruenSplitEqPolynomial, RaPolynomial, mles_product_sum to jolt-poly
14. **Task 14** — uniform Spartan variant + streaming sumcheck in jolt-spartan (depends on 13)

## Architecture decision: downstream dependency scope

Both `jolt-zkvm` and `jolt-spartan` depend on `jolt-ir`. The sumcheck crate remains generic:

- **jolt-sumcheck** — generic sumcheck protocol. Does not touch claim formulas. No `jolt-ir` dependency.
- **jolt-spartan** — depends on `jolt-ir`. Uses `jolt-ir`'s R1CS types as its native constraint representation.
- **jolt-zkvm** — orchestrator. Defines all ~20 claim formulas as `ClaimDefinition`s using `jolt-ir`. Passes them to sumcheck/spartan as needed.

## Philosophy

The IR is the source of truth for claim-level expressions. Developers write each sumcheck claim formula once as an `Expr`. All backends derive from it. No hand-written parallel implementations.

The IR does NOT own verifier orchestration (transcript, sequencing, commitments). That stays as Rust code.
