//! Constraint, rejection, and tamper taxonomies (spec §3.2, §4.4).
//!
//! These enums define the static vocabulary used by the cross-verifier
//! soundness suite. The taxonomy must cover every constraint the
//! reference verifier (`V_core`) checks; the per-test coverage table
//! lives in [`super::tamper::TAMPER_COVERAGE`].

/// A structural assertion the reference verifier checks (spec §4.4).
///
/// `Constraints(V_core)` is the union of all `Constraint` instances
/// the schedule emits. The taxonomy is required to *cover* this set
/// (S7): each `Constraint` must have at least one witnessing tamper
/// in `TAMPER_COVERAGE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Constraint {
    /// Absorbed config + IO matches preprocessing.
    Preamble,
    /// `i`-th commitment matches the `AbsorbCommitment` slot.
    CommitSlot(usize),
    /// Instance `j` input-claim formula in stage `s` equals the sumcheck
    /// input claim used by the verifier.
    InputClaim { stage: usize, instance: usize },
    /// Round `r` of stage `s` satisfies `s(0) + s(1) = prior_claim`.
    RoundPoly { stage: usize, round: usize },
    /// Instance `j` output-check formula in stage `s` equals the sumcheck
    /// final eval at the instance's normalized challenge slice.
    OutputCheck { stage: usize, instance: usize },
    /// `k`-th evaluation in stage `s` is consistent with its committed poly
    /// (the value the prover claims for poly P at point r matches what
    /// PCS opening verifies).
    EvalConsistency { stage: usize, eval_idx: usize },
    /// `k`-th opening claim has eval matching the recorded eval for that poly.
    OpeningClaim(usize),
    /// PCS verifier accepts the joint opening proof.
    OpeningProof,
    /// `Vec<Option<PCS::Output>>[i]` matches the schedule's expected skip semantics
    /// (None ⇔ all-zero advice; Some ⇔ committed poly).
    CommitSkip(usize),
    /// `k`-th absorption uses the expected `DomainSeparator` tag.
    TranscriptTag(usize),
}

/// Coarse classification of a verifier rejection (spec §3.2).
///
/// Two verifiers are *cause-aligned* (S6, graded) iff their rejection
/// categories are equal — modulo `acceptable_set` collapses
/// (`OpeningClaimEvalMismatch ↔ OpeningProofInvalid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionCategory {
    PreambleMismatch,
    CommitmentMismatch,
    SumcheckRoundPolySum,
    SumcheckFinalEval,
    OpeningClaimEvalMismatch,
    OpeningProofInvalid,
    TranscriptDivergence,
    StructuralInvalid,
    Other,
}

impl RejectionCategory {
    /// Return whether `self` and `other` are equivalent under §4.3 graded
    /// equivalence (`acceptable_set`). Currently the only collapse is the
    /// opening-claim ↔ opening-proof pair: both fire on a downstream
    /// PCS-detected eval mismatch and the verifiers disagree on which
    /// surfaces first.
    pub fn graded_equivalent(self, other: Self) -> bool {
        if self == other {
            return true;
        }
        matches!(
            (self, other),
            (Self::OpeningClaimEvalMismatch, Self::OpeningProofInvalid,)
                | (Self::OpeningProofInvalid, Self::OpeningClaimEvalMismatch,)
        )
    }
}

/// The eleven tamper categories from spec §3.3.
///
/// The first column of `TAMPER_COVERAGE` (in `tamper.rs`) maps each
/// kind to the constraint(s) it witnesses, so the suite can statically
/// verify S7 (taxonomy covers `Constraints(V_core)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TamperKind {
    T1RoundPolyCoeff,
    T2Eval,
    T3CommitmentSwap,
    T4OpeningProofByte,
    T5CommitSlotNoneSome,
    T6ConfigField,
    T7CrossStage,
    T8RoundPolyDegree,
    T9BatchClaim,
    T10DomainSeparatorTag,
    T11PublicIo,
}

impl TamperKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::T1RoundPolyCoeff => "T1_RoundPolyCoeff",
            Self::T2Eval => "T2_Eval",
            Self::T3CommitmentSwap => "T3_Commitment",
            Self::T4OpeningProofByte => "T4_OpeningProof",
            Self::T5CommitSlotNoneSome => "T5_CommitSlot",
            Self::T6ConfigField => "T6_Config",
            Self::T7CrossStage => "T7_CrossStage",
            Self::T8RoundPolyDegree => "T8_Degree",
            Self::T9BatchClaim => "T9_BatchClaim",
            Self::T10DomainSeparatorTag => "T10_Tag",
            Self::T11PublicIo => "T11_Io",
        }
    }
}
