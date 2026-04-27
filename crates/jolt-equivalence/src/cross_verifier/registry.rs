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
/// Initial population reflects the post-spec-authoring state: stages
/// 3 and 4 verifier ops are stubbed (squeezes only, no `VerifySumcheck`
/// or `CheckOutput`), and stages 5–7 are not assembled into the verifier
/// schedule at all (`build_module` only chains stages 1–4 — see
/// `crates/jolt-compiler/examples/jolt_core_module.rs:1406-1411`).
///
/// As parity work lands, entries are removed.  An empty registry plus
/// honest acceptance + KGC + S5' + S7 is the target end state.
pub const KNOWN_GAPS: &[KnownGap] = &[
    // ── Stage 3 (Shift / InstructionInput / RegistersClaimReduction) ──
    // Verifier stub at jolt_core_module.rs::build_verifier_stage3_ops
    // emits squeezes only; tampering any round poly coefficient or
    // committed-poly evaluation is invisible to V_mod.
    KnownGap {
        stage: 3,
        kind: TamperKind::T1RoundPolyCoeff,
        rationale: "Stage 3 verifier stubbed (squeezes only)",
        owner: "verifier-agent",
    },
    KnownGap {
        stage: 3,
        kind: TamperKind::T8RoundPolyDegree,
        rationale: "Stage 3 verifier stubbed (no sumcheck verification)",
        owner: "verifier-agent",
    },
    // ── Stage 4 (RegistersRWC / RamValCheck) ──
    KnownGap {
        stage: 4,
        kind: TamperKind::T1RoundPolyCoeff,
        rationale: "Stage 4 verifier stubbed (squeezes + ram_val_check_gamma only)",
        owner: "verifier-agent",
    },
    KnownGap {
        stage: 4,
        kind: TamperKind::T8RoundPolyDegree,
        rationale: "Stage 4 verifier stubbed (no sumcheck verification)",
        owner: "verifier-agent",
    },
    // ── Stage 5 (InstructionReadRaf / RamRaClaimReduction / RegistersValEvaluation) ──
    // Not assembled into build_module's verifier_ops chain at all.
    KnownGap {
        stage: 5,
        kind: TamperKind::T1RoundPolyCoeff,
        rationale: "Stage 5 not wired into verifier schedule",
        owner: "verifier-agent",
    },
    KnownGap {
        stage: 5,
        kind: TamperKind::T8RoundPolyDegree,
        rationale: "Stage 5 not wired into verifier schedule",
        owner: "verifier-agent",
    },
    // ── Stage 6 (BytecodeReadRaf / Booleanity / HammingBooleanity / RamRa / LookupsRa / IncReduction +advice) ──
    KnownGap {
        stage: 6,
        kind: TamperKind::T1RoundPolyCoeff,
        rationale: "Stage 6 not wired into verifier schedule",
        owner: "verifier-agent",
    },
    KnownGap {
        stage: 6,
        kind: TamperKind::T8RoundPolyDegree,
        rationale: "Stage 6 not wired into verifier schedule",
        owner: "verifier-agent",
    },
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
