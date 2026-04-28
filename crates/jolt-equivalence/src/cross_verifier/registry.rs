//! `KnownGapRegistry` — explicit list of (stage, tamper) pairs where
//! V_core rejects but V_mod accepts (spec §3.6).
//!
//! Every entry must be removed in the same commit that closes its
//! underlying gap. The `known_gap_registry_consistency` test asserts
//! each remaining entry still produces a `(V_core = Reject, V_mod =
//! Accept)` mismatch; if a registry entry no longer reproduces the gap,
//! that test fails until the entry is removed.

use super::categories::TamperKind;

/// A single registered exemption from rejection equivalence (S5').
#[derive(Debug, Clone, Copy)]
pub struct KnownGap {
    /// Stage index (1..=7) the tamper targets.
    pub stage: usize,
    /// The kind of tamper that exposes the gap.
    pub kind: TamperKind,
    /// Why the gap exists today and how to read it. Updated when the
    /// underlying gap moves but is not closed.
    pub rationale: &'static str,
    /// Owner responsible for closing the gap.
    pub owner: &'static str,
}

/// The literal registry: every (stage, tamper-kind) where modular
/// currently accepts and core rejects on the muldiv fixture.
///
/// As parity work lands, entries are removed.  An empty registry plus
/// honest acceptance + KGC + S5' + S7 is the target end state.
///
/// Stage 3 closed: build_verifier_stage3_ops now emits the full
/// VerifySumcheck/CheckOutput schedule for [Shift, InstructionInput,
/// RegistersClaimReduction] (see `jolt_core_module.rs`).
///
/// Stage 4 closed: build_verifier_stage4_ops now emits the full
/// VerifySumcheck/CheckOutput schedule for [RegistersRWC, RamValCheck].
///
/// Stage 5 partially closed: build_verifier_stage5_ops emits VerifySumcheck
/// for [InstructionReadRaf, RamRaClaimReduction, RegistersValEvaluation],
/// catching T1 round-poly-coefficient and T8 degree tampers. CheckOutput
/// is deferred (output_check formulas pending — InstructionReadRaf needs
/// a `ClaimFactor::CombineEntryEval` variant for the prefix-suffix
/// composition).
///
/// Stage 6 partially closed: build_verifier_stage6_ops emits VerifySumcheck
/// for [BytecodeReadRaf, Booleanity, HammingBooleanity, RamRaVirtual,
/// InstructionRaVirtual, IncClaimReduction], catching T1 / T8 tampers.
/// CheckOutput is deferred (output_check formulas pending — Booleanity /
/// HammingBooleanity need an `EvalSquared`-style factor for `ra² − ra`,
/// BytecodeReadRaf needs the bytecode-val composition).
pub const KNOWN_GAPS: &[KnownGap] = &[
    // ── Stage 7 (HammingWeightClaimReduction +advice address phase) ──
    KnownGap {
        stage: 7,
        kind: TamperKind::T1RoundPolyCoeff,
        rationale: "Stage 7 not wired into verifier schedule",
        owner: "verifier-agent",
    },
    KnownGap {
        stage: 7,
        kind: TamperKind::T8RoundPolyDegree,
        rationale: "Stage 7 not wired into verifier schedule",
        owner: "verifier-agent",
    },
];

/// Lookup helper used by the runner: is `(stage, kind)` registered?
pub fn is_registered(stage: usize, kind: TamperKind) -> bool {
    KNOWN_GAPS
        .iter()
        .any(|g| g.stage == stage && g.kind == kind)
}

/// Number of currently-registered gaps. Used by the test summary
/// (Q1, I2 — monotonic registry shrinkage).
pub fn registered_count() -> usize {
    KNOWN_GAPS.len()
}
