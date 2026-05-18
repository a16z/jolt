# Jolt Formula Tracking

This file tracks the Jolt verifier claim formulas mirrored from `jolt-core`.
Each row corresponds to a `SumcheckId` / `JoltStageId` that can contribute an
input claim, output claim, or boundary consistency check.

## Conventions

- Use typed `JoltOpeningId` and `JoltChallengeId` values, not positional slots.
- Use semantic challenge names, e.g. `Gamma`, `EqCycle`, `LtCycle`.
- Express challenge products structurally in `Expr`; for example use
  `Gamma * Gamma` instead of a separate `GammaSquared` challenge ID.
- Model cross-stage opening equality with explicit consistency claims rather
  than relying on downstream verifier code to rediscover core assertions.
- Keep compatibility or old accumulator conversion outside these formula
  modules.
- Derive runtime one-hot formula dimensions through `JoltFormulaDimensions`; do not
  duplicate `d` arithmetic or RA polynomial ordering in individual formulas.
- Mark a row done only after the formula has dependency-order tests and a field
  evaluation test against the core verifier formula.

## Status

| Status | Stage | Core source | Formula module | Notes |
|---|---|---|---|---|
| Done | `SpartanOuter` | `jolt-core/src/zkvm/spartan/outer.rs` | `spartan.rs` | Implemented explicit uniskip and remainder constructors. The remainder uses a generic public-coefficient quadratic/linear/constant template over typed outer variables so R1CS lowering can provide coefficients without hardcoding core's R1CS tables here. |
| Done | `SpartanProductVirtualization` | `jolt-core/src/zkvm/spartan/product.rs` | `spartan.rs` | Implemented explicit uniskip and remainder constructors with public Lagrange weights and tau kernel. |
| Done | `SpartanShift` | `jolt-core/src/zkvm/spartan/shift.rs` | `spartan.rs` | Implemented shift input/output claim with `Gamma`, `EqPlusOneOuter`, and `EqPlusOneProduct`. |
| Done | `InstructionClaimReduction` | `jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs` | `claim_reductions/instruction.rs` | Implemented with `Gamma` and `EqSpartan`; powers through `Gamma^4` are structural. |
| Done | `InstructionInputVirtualization` | `jolt-core/src/zkvm/spartan/instruction_input.rs` | `instruction.rs` | Implemented with `Gamma`, `EqProduct`, and explicit same-evaluation checks against `InstructionClaimReduction`. |
| Done | `InstructionReadRaf` | `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` | `instruction.rs` | Implemented with canonical `LookupTableKind` iteration, virtual-RA dimensions, `Gamma`, table-value coefficients, RAF coefficients, and lookup-output consistency. |
| Done | `InstructionRaVirtualization` | `jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs` | `instruction.rs` | Implemented with `Gamma`, `EqCycle`, checked virtual/committed RA dimensions, and explicit gamma transcript dependency. |
| Done | `RamReadWriteChecking` | `jolt-core/src/zkvm/ram/read_write_checking.rs` | `ram.rs` | Implemented with `Gamma` and `EqCycle`; products are explicit. |
| Done | `RamRafEvaluation` | `jolt-core/src/zkvm/ram/raf_evaluation.rs` | `ram.rs` | Implemented with field-native dummy-cycle scaling and an explicit public `UnmapAddress` value. |
| Done | `RamOutputCheck` | `jolt-core/src/zkvm/ram/output_check.rs` | `ram.rs` | Implemented as public `EqIoMask` and `NegEqIoMaskValIo` coefficients over `RamValFinal`. |
| Done | `RamValCheck` | `jolt-core/src/zkvm/ram/val_check.rs` | `ram.rs` | Implemented with `Gamma`, `LtCyclePlusGamma`, full-init and public-plus-advice init decomposition. |
| Done | `RamRaClaimReduction` | `jolt-core/src/zkvm/claim_reductions/ram_ra.rs` | `ram.rs` | Implemented with `Gamma`, explicit RAF/read-write/val-check inputs, and public cycle EQ coefficients. |
| Done | `RamHammingBooleanity` | `jolt-core/src/zkvm/ram/hamming_booleanity.rs` | `ram.rs` | Implemented as `EqCycle * (H^2 - H)` over `RamHammingWeight`. |
| Done | `RamRaVirtualization` | `jolt-core/src/zkvm/ram/ra_virtual.rs` | `ram.rs` | Implemented with checked committed-RA dimensions and public `EqCycle` coefficient. |
| Done | `RegistersClaimReduction` | `jolt-core/src/zkvm/claim_reductions/registers.rs` | `claim_reductions/registers.rs` | Implemented with `Gamma` and `EqSpartan`; `Gamma * Gamma` is structural. |
| Done | `RegistersReadWriteChecking` | `jolt-core/src/zkvm/registers/read_write_checking.rs` | `registers.rs` | Implemented with `Gamma` and `EqCycle`; `Gamma * Gamma` is structural. |
| Done | `RegistersValEvaluation` | `jolt-core/src/zkvm/registers/val_evaluation.rs` | `registers.rs` | Implemented with `LtCycle`. |
| Done | `BytecodeReadRaf` | `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` | `bytecode.rs` | Implemented typed stage-folding/per-stage gammas, explicit PC and lookup flag inputs, entry public coefficient, RA product output, and unexpanded-PC consistency. |
| Done | `Booleanity` | `jolt-core/src/subprotocols/booleanity.rs` | `booleanity.rs` | Implemented generic booleanity over the canonical Jolt RA polynomial layout with explicit public `EqAddressCycle` and structural `Gamma^(2i)` powers. |
| Done | `AdviceClaimReductionCyclePhase` | `jolt-core/src/zkvm/claim_reductions/advice.rs` | `claim_reductions/advice.rs` | Implemented trusted/untrusted advice source typing, direct cycle intermediate output, and no-address-phase final-scale path. |
| Done | `AdviceClaimReduction` | `jolt-core/src/zkvm/claim_reductions/advice.rs` | `claim_reductions/advice.rs` | Implemented address/final phase with explicit public final-scale coefficient. |
| Done | `IncClaimReduction` | `jolt-core/src/zkvm/claim_reductions/increments.rs` | `claim_reductions/increments.rs` | Implemented `RamInc`/`RdInc` reduction with explicit equality publics and structural `Gamma` powers. |
| Done | `HammingWeightClaimReduction` | `jolt-core/src/zkvm/claim_reductions/hamming_weight.rs` | `claim_reductions/hamming_weight.rs` | Implemented RA hamming-weight/booleanity/virtualization reduction with the canonical Jolt RA polynomial layout and per-RA virtualization publics. |

## Suggested Order

1. Wire these formula constructors into the higher-level verifier proof model.
2. Replace the generic Spartan outer coefficient dimensions with coefficients emitted
   by the chosen R1CS lowering source once `jolt-verifier` starts consuming it.
