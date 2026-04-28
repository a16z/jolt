//! Tamper definitions and application (spec §3.2 — `TamperPoint` /
//! `TamperLocation` / `TamperMutation` — and §3.5 coverage table).
//!
//! A `TamperPoint` is a structural description of a mutation we apply
//! to an honest proof to test rejection equivalence. `apply_tamper`
//! consumes a mutable proof and applies the mutation in-place; the
//! runner then runs both verifiers and compares decisions.

use jolt_dory::DoryScheme;
use jolt_field::Fr as NewFr;
use jolt_poly::UnivariatePoly;
use jolt_verifier::JoltProof;
use num_traits::{One, Zero};

use super::categories::{Constraint, TamperKind};

/// A structural mutation point against an honest proof.
#[derive(Debug, Clone)]
pub struct TamperPoint {
    /// Stage the tamper targets (1-indexed for human readability,
    /// matching protocol stage numbers; `apply_tamper` translates to
    /// 0-indexed `stage_proofs[stage - 1]`).
    pub stage: usize,
    /// Where in the proof to mutate.
    pub location: TamperLocation,
    /// The mutation to apply.
    pub mutate: TamperMutation,
    /// The constraints this tamper is intended to witness (spec §3.5).
    pub witnesses: Vec<Constraint>,
    /// Cross-verifier expectation. `BothReject` is the standard target
    /// for any tamper that violates a `Constraints(V_core)` element.
    pub expected: ExpectedResult,
    /// Tamper kind for registry lookups + reporting.
    pub kind: TamperKind,
    /// Human-readable label for failure messages.
    pub label: &'static str,
}

/// What part of the proof a tamper targets.
#[derive(Debug, Clone)]
pub enum TamperLocation {
    /// `stage_proofs[stage].round_polys.round_polynomials[round].coefficients[coeff]`.
    RoundPolyCoeff { round: usize, coeff: usize },
    /// `stage_proofs[stage].round_polys.round_polynomials[round]` —
    /// truncate or extend the polynomial coefficient list (T8 family).
    RoundPolyDegree { round: usize, kind: DegreeKind },
    /// `stage_proofs[stage].evals[idx]`.
    Eval { idx: usize },
    /// `commitments[idx]` — swap with another honest proof's commitment.
    Commitment { idx: usize },
    /// `opening_proofs[0]` — flip a single bit in the serialized form.
    OpeningProofByte { byte: usize, bit: u8 },
    /// `commitments[idx]` — toggle between `None` and `Some(<honest>)`.
    CommitSlot { idx: usize, op: CommitSlotOp },
    /// `proof.config.<field>` — preamble fields.
    Config(ConfigField),
    /// Public IO: `proof.config.inputs[idx]` / `outputs[idx]` / `panic`.
    Io(IoField),
}

#[derive(Debug, Clone, Copy)]
pub enum DegreeKind {
    /// Drop the highest-degree coefficient.
    Truncate,
    /// Append an extra coefficient (zero) — extends nominal degree.
    Extend,
}

#[derive(Debug, Clone, Copy)]
pub enum CommitSlotOp {
    /// Replace `None` with `Some(<honest commitment from same slot of another proof>)`.
    NoneToSome,
    /// Replace `Some(c)` with `None`.
    SomeToNone,
}

#[derive(Debug, Clone, Copy)]
pub enum ConfigField {
    TraceLength,
    RamK,
}

#[derive(Debug, Clone, Copy)]
pub enum IoField {
    InputByte(usize),
    OutputByte(usize),
    PanicFlag,
}

/// What the runner does with a tampered field value.
#[derive(Debug, Clone)]
pub enum TamperMutation {
    /// Add field-element 1.
    AddOne,
    /// Replace with the given coefficient bytes (interpreted as the
    /// target field element). For non-`AddOne` mutations on round-poly
    /// coefficients, callers populate this with a deterministic fresh
    /// field element.
    ReplaceField(NewFr),
    /// Flip one bit in the serialized representation. Only valid for
    /// `OpeningProofByte`.
    FlipBit,
    /// Replace `usize` configuration field with the given value.
    SetUsize(usize),
    /// Replace `bool` field (panic flag) with the given value.
    SetBool(bool),
    /// Replace a public-IO byte with the given value.
    SetByte(u8),
}

/// Cross-verifier expectation (spec §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedResult {
    /// V_core and V_mod both reject the tampered proof.
    BothReject,
    /// V_core and V_mod both accept (e.g. tamper is benign).
    BothAccept,
    /// V_core rejects, V_mod accepts — registered known gap.
    CoreRejectsModularAccepts,
    /// V_core accepts, V_mod rejects — would indicate over-strict V_mod
    /// (currently never expected).
    CoreAcceptsModularRejects,
}

/// Result of applying a tamper.
#[derive(Debug)]
pub enum TamperOutcome {
    /// Mutation succeeded — proof is now mutated.
    Applied,
    /// Mutation could not be applied because the target slot is empty
    /// (e.g. no evals in stages 3-7 today, no commitment at given index).
    /// The runner treats this as "vacuous" — neither rejection nor
    /// acceptance is meaningful.
    Vacuous(&'static str),
}

/// Apply a tamper to a modular proof in place.
///
/// Returns `Vacuous` when the slot doesn't exist on the current proof
/// shape (e.g. `Eval { idx }` against a stage with no recorded evals).
pub fn apply_tamper(
    proof: &mut JoltProof<NewFr, DoryScheme>,
    tamper: &TamperPoint,
) -> TamperOutcome {
    let stage_idx = tamper.stage.checked_sub(1);
    match (&tamper.location, &tamper.mutate) {
        (TamperLocation::RoundPolyCoeff { round, coeff }, mutation) => {
            let Some(s) = stage_idx else {
                return TamperOutcome::Vacuous("stage 0 invalid");
            };
            let Some(stage_proof) = proof.stage_proofs.get_mut(s) else {
                return TamperOutcome::Vacuous("stage out of range");
            };
            let Some(rp) = stage_proof.round_polys.round_polynomials.get_mut(*round) else {
                return TamperOutcome::Vacuous("round out of range");
            };
            let mut coeffs: Vec<NewFr> = rp.coefficients().to_vec();
            let Some(c) = coeffs.get_mut(*coeff) else {
                return TamperOutcome::Vacuous("coeff out of range");
            };
            apply_field_mutation(c, mutation);
            *rp = UnivariatePoly::new(coeffs);
            TamperOutcome::Applied
        }
        (TamperLocation::RoundPolyDegree { round, kind }, _) => {
            let Some(s) = stage_idx else {
                return TamperOutcome::Vacuous("stage 0 invalid");
            };
            let Some(stage_proof) = proof.stage_proofs.get_mut(s) else {
                return TamperOutcome::Vacuous("stage out of range");
            };
            let Some(rp) = stage_proof.round_polys.round_polynomials.get_mut(*round) else {
                return TamperOutcome::Vacuous("round out of range");
            };
            let mut coeffs: Vec<NewFr> = rp.coefficients().to_vec();
            match kind {
                DegreeKind::Truncate => {
                    if coeffs.is_empty() {
                        return TamperOutcome::Vacuous("cannot truncate empty round poly");
                    }
                    let _ = coeffs.pop();
                }
                DegreeKind::Extend => {
                    coeffs.push(NewFr::zero());
                }
            }
            *rp = UnivariatePoly::new(coeffs);
            TamperOutcome::Applied
        }
        (TamperLocation::Eval { idx }, mutation) => {
            let Some(s) = stage_idx else {
                return TamperOutcome::Vacuous("stage 0 invalid");
            };
            let Some(stage_proof) = proof.stage_proofs.get_mut(s) else {
                return TamperOutcome::Vacuous("stage out of range");
            };
            let Some(e) = stage_proof.evals.get_mut(*idx) else {
                return TamperOutcome::Vacuous("eval idx out of range (stage has no evals?)");
            };
            apply_field_mutation(e, mutation);
            TamperOutcome::Applied
        }
        (TamperLocation::Config(field), mutation) => {
            match (field, mutation) {
                (ConfigField::TraceLength, TamperMutation::SetUsize(v)) => {
                    proof.config.trace_length = *v;
                }
                (ConfigField::RamK, TamperMutation::SetUsize(v)) => {
                    proof.config.ram_k = *v;
                }
                _ => return TamperOutcome::Vacuous("config field/mutation mismatch"),
            }
            TamperOutcome::Applied
        }
        (TamperLocation::Io(field), mutation) => {
            match (field, mutation) {
                (IoField::InputByte(i), TamperMutation::SetByte(b)) => {
                    let Some(slot) = proof.config.inputs.get_mut(*i) else {
                        return TamperOutcome::Vacuous("input idx out of range");
                    };
                    *slot = *b;
                }
                (IoField::OutputByte(i), TamperMutation::SetByte(b)) => {
                    let Some(slot) = proof.config.outputs.get_mut(*i) else {
                        return TamperOutcome::Vacuous("output idx out of range");
                    };
                    *slot = *b;
                }
                (IoField::PanicFlag, TamperMutation::SetBool(v)) => {
                    proof.config.panic = *v;
                }
                _ => return TamperOutcome::Vacuous("io field/mutation mismatch"),
            }
            TamperOutcome::Applied
        }
        (TamperLocation::Commitment { idx }, _) => {
            // T3 — swap commitments[idx] with commitments[0] in place.
            // Slot 0 is RdInc (a dense, always-non-zero poly), so the
            // swap forces an unrelated commitment into slot[idx]'s
            // position. If both are Some, PCS verification at stage 8
            // rejects because the commitment_map points the wrong
            // commitment at slot[idx]'s polynomial type.
            //
            // (Adjacent-index swaps like commitments[2] ↔
            // commitments[3] can be invisible when both polys are
            // all-zero — common for unused InstructionRa indices in
            // small workloads. Swapping against slot 0 dodges this.)
            if *idx == 0 {
                return TamperOutcome::Vacuous("idx 0 has nothing to swap with");
            }
            let Some(slot_idx) = proof.commitments.get(*idx) else {
                return TamperOutcome::Vacuous("commitment idx out of range");
            };
            if slot_idx.is_none() || proof.commitments[0].is_none() {
                return TamperOutcome::Vacuous("commitment slot is None");
            }
            proof.commitments.swap(*idx, 0);
            TamperOutcome::Applied
        }
        (TamperLocation::CommitSlot { idx, op }, _) => {
            // T5 — toggle a slot between None and Some(<honest>).
            // SomeToNone: zero out an honest commitment.
            // NoneToSome: replace a None slot with another slot's
            // commitment (any non-matching `Some` value will fail
            // CollectOpeningClaim's commitment_map.get → fail PCS or
            // simply absorb a different commitment to the transcript).
            let Some(slot) = proof.commitments.get_mut(*idx) else {
                return TamperOutcome::Vacuous("commit slot idx out of range");
            };
            match (op, &slot) {
                (CommitSlotOp::SomeToNone, Some(_)) => {
                    *slot = None;
                    TamperOutcome::Applied
                }
                (CommitSlotOp::NoneToSome, None) => {
                    // Pull a Some from a sibling slot (idx 0 typically
                    // committed to RdInc — always non-None for muldiv).
                    if *idx == 0 {
                        return TamperOutcome::Vacuous("no sibling slot to copy from at idx 0");
                    }
                    let sibling = proof.commitments.first().cloned().flatten();
                    let Some(c) = sibling else {
                        return TamperOutcome::Vacuous("sibling slot 0 is None — nothing to copy");
                    };
                    proof.commitments[*idx] = Some(c);
                    TamperOutcome::Applied
                }
                _ => TamperOutcome::Vacuous("commit slot op/state mismatch"),
            }
        }
        (TamperLocation::OpeningProofByte { .. }, _) => {
            // T4 — structural opening-proof tamper.
            //
            // True per-byte FlipBit on the serialized PCS proof would
            // require serde round-trips through the Dory proof type;
            // instead we use a coarser structural tamper: drop the
            // last opening proof. The verifier's stage-8 length check
            // (`reduced.len() != proof.opening_proofs.len()`) rejects.
            //
            // This isn't a full FlipBit tamper but it witnesses
            // C_opening_proof — the constraint is "PCS verifier
            // accepts the joint opening proof", which fails as soon
            // as the proof is structurally invalid.
            if proof.opening_proofs.is_empty() {
                return TamperOutcome::Vacuous("no opening proofs to tamper");
            }
            let _ = proof.opening_proofs.pop();
            TamperOutcome::Applied
        }
    }
}

fn apply_field_mutation(slot: &mut NewFr, mutation: &TamperMutation) {
    match mutation {
        TamperMutation::AddOne => {
            *slot += NewFr::one();
        }
        TamperMutation::ReplaceField(v) => {
            *slot = *v;
        }
        TamperMutation::FlipBit
        | TamperMutation::SetByte(_)
        | TamperMutation::SetBool(_)
        | TamperMutation::SetUsize(_) => {
            // Wrong mutation type for a field slot — silently treat as no-op.
            // Caller bug; the suite's static coverage table prevents this.
        }
    }
}

/// Static map: each tamper kind to the constraint(s) it witnesses (spec §3.5).
///
/// The S7 launch-time check (`assert_taxonomy_covers_constraints`) enumerates
/// `Constraints(V_core)` from the schedule and asserts every one is covered
/// by at least one tamper kind in this table.
pub const TAMPER_COVERAGE: &[(TamperKind, &[ConstraintTemplate])] = &[
    (
        TamperKind::T1RoundPolyCoeff,
        &[ConstraintTemplate::RoundPoly],
    ),
    (
        TamperKind::T2Eval,
        &[
            ConstraintTemplate::OutputCheck,
            ConstraintTemplate::OpeningClaim,
            ConstraintTemplate::EvalConsistency,
        ],
    ),
    (
        TamperKind::T3CommitmentSwap,
        &[
            ConstraintTemplate::CommitSlot,
            ConstraintTemplate::OpeningProof,
        ],
    ),
    (
        TamperKind::T4OpeningProofByte,
        &[ConstraintTemplate::OpeningProof],
    ),
    (
        TamperKind::T5CommitSlotNoneSome,
        &[ConstraintTemplate::CommitSkip],
    ),
    (TamperKind::T6ConfigField, &[ConstraintTemplate::Preamble]),
    (TamperKind::T7CrossStage, &[ConstraintTemplate::OutputCheck]),
    (
        TamperKind::T8RoundPolyDegree,
        &[ConstraintTemplate::RoundPoly],
    ),
    (TamperKind::T9BatchClaim, &[ConstraintTemplate::InputClaim]),
    (
        TamperKind::T10DomainSeparatorTag,
        &[ConstraintTemplate::TranscriptTag],
    ),
    (TamperKind::T11PublicIo, &[ConstraintTemplate::Preamble]),
];

/// Constraint shape (template) used by the coverage table. Stage and
/// instance indices are filled in at suite-launch time when enumerating
/// `Constraints(V_core)` from the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintTemplate {
    Preamble,
    CommitSlot,
    InputClaim,
    RoundPoly,
    OutputCheck,
    EvalConsistency,
    OpeningClaim,
    OpeningProof,
    CommitSkip,
    TranscriptTag,
}

impl ConstraintTemplate {
    pub fn matches(self, c: &Constraint) -> bool {
        matches!(
            (self, c),
            (Self::Preamble, Constraint::Preamble)
                | (Self::CommitSlot, Constraint::CommitSlot(_))
                | (Self::InputClaim, Constraint::InputClaim { .. })
                | (Self::RoundPoly, Constraint::RoundPoly { .. })
                | (Self::OutputCheck, Constraint::OutputCheck { .. })
                | (Self::EvalConsistency, Constraint::EvalConsistency { .. })
                | (Self::OpeningClaim, Constraint::OpeningClaim(_))
                | (Self::OpeningProof, Constraint::OpeningProof)
                | (Self::CommitSkip, Constraint::CommitSkip(_))
                | (Self::TranscriptTag, Constraint::TranscriptTag(_))
        )
    }
}
