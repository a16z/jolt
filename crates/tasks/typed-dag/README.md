# Typed DAG Implementation Tasks

Task dependency graph for the jolt-zkvm pipeline rewrite.
Full design spec: `../../typed_dag.md`

## Dependency Graph

```
Phase 1 — Foundation (all independent, parallelizable)
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ T01      │  │ T02      │  │ T03      │  │ T04      │
│ Claim    │  │ Poly     │  │ IR Audit │  │ IR→Kernel│
│ Types    │  │ Tables   │  │          │  │ Bridge   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │              │
     └──────┬──────┘             └──────┬───────┘
            │                           │
     ┌──────▼──────┐             ┌──────▼──────┐
     │ T05         │             │ T06         │
     │ Stage Output│             │ Input Claim │
     │ Types       │             │ Formulas    │
     └──────┬──────┘             └──────┬──────┘
            │                           │
            └───────────┬───────────────┘
                        │
Phase 2 — Stages (all parallelizable once T04+T05+T06 done)
     ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
     │ T07  │ T08  │ T09  │ T10  │ T11  │ T12  │ T13  │
     │ S1   │ S2   │ S3   │ S4   │ S5   │ S6   │ S7   │
     └──┬───┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘
        │       │      │      │      │      │      │
        └───────┴──────┴──┬───┴──────┴──────┴──────┘
                          │
                   ┌──────▼──────┐
                   │ T14         │
                   │ PCS Opening │
                   └──────┬──────┘
                          │
Phase 3 — Integration (sequential)
                   ┌──────▼──────┐
                   │ T15         │
                   │ Orchestrator│
                   └──────┬──────┘
                   ┌──────▼──────┐
                   │ T16         │
                   │ Verifier    │
                   └──────┬──────┘
                   ┌──────▼──────┐
                   │ T17         │
                   │ E2E muldiv  │
                   └──────┬──────┘
                   ┌──────▼──────┐
                   │ T18         │
                   │ Cleanup     │
                   └─────────────┘
```

## Task Index

| ID | Name | Crate | Deps | Status |
|----|------|-------|------|--------|
| **Phase 1: Foundation** |
| T01 | [Claim Types](T01_claim_types.md) | jolt-openings | — | `[x]` |
| T02 | [PolynomialTables](T02_polynomial_tables.md) | jolt-zkvm | — | `[x]` |
| T03 | [IR Claim Audit](T03_ir_claim_audit.md) | jolt-ir | — | `[x]` |
| T04 | [IR→Kernel Bridge](T04_ir_kernel_bridge.md) | jolt-ir | T03 | `[x]` |
| T05 | [Stage Output Types](T05_stage_output_types.md) | jolt-zkvm | T01, T02 | `[x]` |
| T06 | [Input Claim Formulas](T06_input_claim_formulas.md) | jolt-zkvm | T03, T05 | `[x]` |
| **Phase 2: Stages** |
| T07 | [S1 Spartan](T07_s1_spartan.md) | jolt-spartan, jolt-zkvm | T05, T06 | `[x]` | |
| T08 | [S2 Stage](T08_s2_stage.md) | jolt-zkvm | T04, T05, T06 | `[x]` | |
| T09 | [S3 Stage](T09_s3_stage.md) | jolt-zkvm | T04, T05, T06 | `[x]` | |
| T10 | [S4 Stage](T10_s4_stage.md) | jolt-zkvm | T04, T05, T06 | `[x]` | |
| T11 | [S5 Stage](T11_s5_stage.md) | jolt-zkvm | T04, T05, T06 | `[x]` | |
| T12 | [S6 Stage](T12_s6_stage.md) | jolt-zkvm | T04, T05, T06 | `[x]` | |
| T13 | [S7 Stage](T13_s7_stage.md) | jolt-zkvm | T04, T05, T06 | `[x]` | |
| T14 | [PCS Opening](T14_pcs_opening.md) | jolt-zkvm | T05 | `[x]` |
| **Phase 3: Integration** |
| T15 | [Prove Orchestrator](T15_prove_orchestrator.md) | jolt-zkvm | T07–T14 | `[x]` |
| T16 | [Verifier DAG](T16_verifier_dag.md) | jolt-verifier | T06, T15 | `[~]` |
| T17 | [E2E muldiv](T17_e2e_muldiv.md) | jolt-zkvm | T15, T16 | `[~]` |
| T18 | [Cleanup](T18_cleanup.md) | jolt-zkvm | T17 | `[x]` |
| **Phase 4: Full Parity** |
| T19 | [Multi-Phase Evaluator](T19_multi_phase_evaluator.md) | jolt-zkvm | T08,T11,T12 | `[ ]` |
| T20 | [Uni-Skip](T20_uni_skip.md) | jolt-zkvm, jolt-sumcheck | T07,T08 | `[ ]` |
| T21 | [RA Virtual (Toom-Cook)](T21_ra_virtual.md) | jolt-zkvm | T12,T19 | `[ ]` |
| T22 | [Verifier Descriptors](T22_verifier_descriptors.md) | jolt-zkvm, jolt-verifier | T15,T16 | `[ ]` |
| T23 | [Fiat-Shamir Parity](T23_fiat_shamir_parity.md) | jolt-zkvm | T19,T20,T21,T22 | `[ ]` |

## Phase 4 Dependency Graph

```
Phase 4 — Full Parity (fills in deferred items)

┌──────────┐  ┌──────────┐
│ T19      │  │ T20      │    ← parallelizable
│ Multi-   │  │ Uni-Skip │
│ Phase    │  │          │
└────┬─────┘  └────┬─────┘
     │             │
     └──────┬──────┘
     ┌──────▼──────┐  ┌──────────┐
     │ T21         │  │ T22      │    ← parallelizable
     │ RA Virtual  │  │ Verifier │
     │ (Toom-Cook) │  │ Descript.│
     └──────┬──────┘  └────┬─────┘
            │              │
            └──────┬───────┘
            ┌──────▼──────┐
            │ T23         │
            │ Fiat-Shamir │
            │ Parity      │
            └─────────────┘
```

## Maximum Parallelism by Wave

| Wave | Tasks | Max Agents | Notes |
|------|-------|------------|-------|
| 1 | T01, T02, T03 | 3 | All independent foundation work |
| 2 | T04, T05, T06 | 3 | T04 needs T03; T05 needs T01+T02; T06 needs T03+T05 |
| 3 | T07–T14 | 8 | All stages + PCS parallelizable |
| 4 | T15 | 1 | Wire everything together |
| 5 | T16 | 1 | Verifier mirrors prover |
| 6 | T17, T18 | 1 | Test then cleanup |
| 7 | T19, T20 | 2 | Multi-phase + uni-skip (parallel) |
| 8 | T21, T22 | 2 | RA virtual + verifier descriptors (parallel) |
| 9 | T23 | 1 | Final Fiat-Shamir parity + E2E muldiv |

## Sizing Estimates

| Size | Tasks |
|------|-------|
| Small (~50-100 lines) | T01, T18 |
| Medium (~200-300 lines) | T02, T04, T05, T06, T07, T09, T10, T11, T13, T14, T15, T17 |
| Large (~400-500 lines) | T03, T08, T12, T16 |

## Critical Path

```
T03 → T06 → T12 (S6 is the most complex stage) → T15 → T17
```

The critical path goes through the IR audit (T03), input claim formulas (T06),
Stage 6 implementation (T12, most complex due to IncCR + 5 other instances),
the orchestrator (T15), and the E2E test (T17).

## Notes

- Stage tasks (T07-T13) can be unit-tested independently with synthetic
  inputs matching their typed output types.
- The IR audit (T03) is research-heavy — read jolt-core and compare against
  existing jolt-ir definitions. May produce the most "surprises."
- T12 (S6) is the highest-risk task — it has 6 instances including IncCR
  which reads from 3 prior stages and produces the critical `r_cycle_s6`.
