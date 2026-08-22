//! The stage 7 `HammingWeightClaimReduction` sumcheck instance.
//!
//! Reduces the per-family RA booleanity, virtualization, and hamming-weight claims
//! (instruction, bytecode, RAM) into the one-hot `Ra` opening claims that anchor the
//! stage 8 final batched opening. Owns the shared opening-point derivation and the
//! `EqBooleanity` / `EqVirtualization` public-value computation.
//!
//! # One slot, two relations
//!
//! `JoltRelationId::HammingWeightClaimReduction` names a protocol slot, not an algebra.
//! The opening ids, transcript labels, and wire field name are identical in both modes;
//! which symbolic relation fills the slot is chosen by the `akita` feature:
//!
//! - base — `claim_reductions::hamming_weight::ClaimReduction`: three gamma legs per RA
//!   polynomial (hamming weight, booleanity, virtualization).
//! - akita — `lattice::relations::digit_zero::LatticeDigitZeroClaimReduction`: two
//!   digit-zero-recentered legs for instruction and bytecode, the base three for RAM,
//!   plus one leg per balanced-increment column and the fused-increment decode leg. See
//!   `specs/digit-zero-virtualization.md`.
//!
//! The private `mode` module below is the only place that choice is made. Everything
//! else — here and in the rest of stage 7 — spells the slot's
//! `HammingWeightClaimReduction*` names and reaches the live relation through the
//! aliases re-exported from it, so no consumer needs a `cfg` of its own.

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::geometry::balanced_inc_value;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
use jolt_claims::protocols::jolt::{
    geometry::ra::JoltRaPolynomialLayout, HammingWeightClaimReductionPublic, JoltDerivedId,
    JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage6b::outputs::{Stage6bOutputClaims, Stage6bOutputPoints};
use crate::VerifierError;

/// Base mode: the slot is a genuine Hamming-weight reduction over fully committed
/// one-hot columns.
#[cfg(not(feature = "akita"))]
mod mode {
    pub use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions as Dimensions;
    pub use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::{
        ClaimReduction as Symbolic, HammingWeightClaimReductionInputClaims as InputClaims,
        HammingWeightClaimReductionOutputClaims as OutputClaims,
    };
}

/// Akita mode: the slot is the digit-zero reduction, which also carries the
/// balanced-increment booleanity legs and the fused-increment decode leg.
#[cfg(feature = "akita")]
mod mode {
    pub use jolt_claims::protocols::jolt::lattice::relations::digit_zero::{
        LatticeDigitZeroClaimReduction as Symbolic,
        LatticeDigitZeroClaimReductionDimensions as Dimensions,
        LatticeDigitZeroClaimReductionInputClaims as InputClaims,
        LatticeDigitZeroClaimReductionOutputClaims as OutputClaims,
    };
}

/// The active relation's shape. Public because the kernel seam reads it off the
/// relation (`Self::dimensions`).
pub use mode::Dimensions as HammingWeightClaimReductionDimensions;
/// The claims the active relation consumes.
pub use mode::InputClaims as HammingWeightClaimReductionInputClaims;
/// The claims the active relation produces.
pub use mode::OutputClaims as HammingWeightClaimReductionOutputClaims;

type HammingWeightClaimReductionSymbolic = mode::Symbolic;

/// Builds the active relation's shape. Base dimensions are infallible; akita
/// additionally derives the balanced-increment chunking from `log_k_chunk`, which
/// rejects widths that do not divide the fused increment.
pub fn hamming_weight_claim_reduction_dimensions(
    layout: JoltRaPolynomialLayout,
    log_k_chunk: usize,
) -> Result<HammingWeightClaimReductionDimensions, VerifierError> {
    #[cfg(not(feature = "akita"))]
    {
        Ok(HammingWeightClaimReductionDimensions::new(
            layout,
            log_k_chunk,
        ))
    }
    #[cfg(feature = "akita")]
    {
        HammingWeightClaimReductionDimensions::new(layout, log_k_chunk).map_err(public_input_failed)
    }
}

/// The hamming reduction's consumed opening *values*, wired from the stage-6b
/// cycle-phase output claims. The relation reads only their values (its produced
/// points are derived from its own sumcheck point), so no input points are needed.
pub fn hamming_weight_input_values_from_upstream<F: JoltField>(
    cycle_phase: &Stage6bOutputClaims<F>,
) -> HammingWeightClaimReductionInputClaims<F> {
    HammingWeightClaimReductionInputClaims {
        ram_hamming_weight: cycle_phase.ram_hamming_booleanity.ram_hamming_weight,
        instruction_booleanity: cycle_phase.booleanity.instruction_ra.clone(),
        bytecode_booleanity: cycle_phase.booleanity.bytecode_ra.clone(),
        ram_booleanity: cycle_phase.booleanity.ram_ra.clone(),
        instruction_virtualization: cycle_phase
            .instruction_ra_virtualization
            .committed_instruction_ra
            .clone(),
        bytecode_virtualization: cycle_phase.bytecode_read_raf.bytecode_ra.clone(),
        ram_virtualization: cycle_phase.ram_ra_virtualization.ram_ra.clone(),
        #[cfg(feature = "akita")]
        balanced_inc_digit_booleanity: cycle_phase.booleanity.balanced_inc_digits.clone(),
        #[cfg(feature = "akita")]
        balanced_inc_carry_booleanity: cycle_phase.booleanity.balanced_inc_carry,
        #[cfg(feature = "akita")]
        fused_inc: cycle_phase.bytecode_read_raf.fused_inc,
    }
}

/// The per-RA virtualization address chunks the hamming reduction's
/// `EqVirtualization` publics compare against, in canonical (instruction, bytecode,
/// RAM) order: the leading `log_k_chunk` coordinates of each stage-6b RA
/// virtualization opening point.
pub fn stage7_hamming_virtualization_address_points<F: JoltField>(
    dimensions: HammingWeightClaimReductionDimensions,
    stage6_points: &Stage6bOutputPoints<F>,
) -> Result<Vec<Vec<F>>, VerifierError> {
    let instruction_ra_points = stage6_points
        .instruction_ra_virtualization
        .committed_instruction_ra();
    let bytecode_ra_points = stage6_points.bytecode_read_raf.bytecode_ra();
    let ram_ra_points = stage6_points.ram_ra_virtualization.ram_ra();
    if instruction_ra_points.len() != dimensions.layout.instruction()
        || bytecode_ra_points.len() != dimensions.layout.bytecode()
        || ram_ra_points.len() != dimensions.layout.ram()
    {
        return Err(public_input_failed(
            "Stage 6 RA opening point count mismatch for Stage 7",
        ));
    }

    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in instruction_ra_points
        .iter()
        .chain(bytecode_ra_points)
        .chain(ram_ra_points)
    {
        let chunk = point.get(..dimensions.log_k_chunk).ok_or_else(|| {
            public_input_failed(format!(
                "Stage 6 RA opening point is too short for HammingWeight address chunk: expected at least {}, got {}",
                dimensions.log_k_chunk,
                point.len()
            ))
        })?;
        points.push(chunk.to_vec());
    }
    Ok(points)
}

#[derive(Clone)]
pub struct HammingWeightClaimReduction<F: JoltField> {
    symbolic: HammingWeightClaimReductionSymbolic,
    dimensions: HammingWeightClaimReductionDimensions,
    /// The shared cycle suffix appended to every produced opening point (the
    /// stage-6 booleanity cycle point).
    r_cycle: Vec<F>,
    /// The stage-6 booleanity address point that `EqBooleanity` compares against.
    r_address: Vec<F>,
    /// The per-RA virtualization address chunks (one per layout polynomial, in
    /// canonical order) that `EqVirtualization(i)` compares against.
    virtualization_points: Vec<Vec<F>>,
}

impl<F: JoltField> HammingWeightClaimReduction<F> {
    pub fn new(
        dimensions: HammingWeightClaimReductionDimensions,
        r_cycle: Vec<F>,
        r_address: Vec<F>,
        virtualization_points: Vec<Vec<F>>,
    ) -> Self {
        Self {
            symbolic: HammingWeightClaimReductionSymbolic::new(dimensions),
            dimensions,
            r_cycle,
            r_address,
            virtualization_points,
        }
    }

    /// The reduction's address chunk point `rho` in reversed order: the leading
    /// `log_k_chunk` coordinates of the (shared) produced opening point. Equal to
    /// the hamming sumcheck point reversed — `opening_point` prepends the reversed
    /// challenges — so the EQ publics evaluate against it directly.
    fn rho_reversed<'a>(
        &self,
        output_points: &'a HammingWeightClaimReductionOutputClaims<Vec<F>>,
    ) -> Result<&'a [F], VerifierError> {
        let opening_point = output_points
            .instruction_ra()
            .first()
            .or_else(|| output_points.bytecode_ra().first())
            .or_else(|| output_points.ram_ra().first())
            .map(|point| point.as_slice())
            .ok_or_else(|| {
                public_input_failed("HammingWeight reduction produced no openings".to_string())
            })?;
        opening_point
            .get(..self.dimensions.log_k_chunk)
            .ok_or_else(|| {
                public_input_failed(format!(
                    "HammingWeight opening point has {} variables, fewer than log_k_chunk {}",
                    opening_point.len(),
                    self.dimensions.log_k_chunk
                ))
            })
    }

    pub fn dimensions(&self) -> HammingWeightClaimReductionDimensions {
        self.dimensions
    }

    pub fn r_cycle(&self) -> &[F] {
        &self.r_cycle
    }

    pub fn r_address(&self) -> &[F] {
        &self.r_address
    }

    pub fn virtualization_points(&self) -> &[Vec<F>] {
        &self.virtualization_points
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: reason.to_string(),
    }
}

/// `eq(point, 0) = Π (1 − point_j)` — the digit-zero weight of an `eq` leg,
/// the `w(0)` baseline the input claim folds in under digit-zero
/// virtualization (`specs/digit-zero-virtualization.md`).
fn eq_at_digit_zero<F: JoltField>(point: &[F]) -> F {
    point.iter().fold(F::one(), |accumulator, value| {
        accumulator * (F::one() - *value)
    })
}

impl<F: JoltField> ConcreteSumcheck<F> for HammingWeightClaimReduction<F> {
    type Symbolic = HammingWeightClaimReductionSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &HammingWeightClaimReductionInputClaims<Vec<F>>,
    ) -> Result<HammingWeightClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        if sumcheck_point.len() != self.dimensions.log_k_chunk {
            return Err(public_input_failed(format!(
                "HammingWeight challenge length mismatch: expected {}, got {}",
                self.dimensions.log_k_chunk,
                sumcheck_point.len()
            )));
        }
        let mut opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        opening_point.extend_from_slice(&self.r_cycle);
        let layout = self.dimensions.layout;
        Ok(HammingWeightClaimReductionOutputClaims {
            instruction_ra: vec![opening_point.clone(); layout.instruction()],
            bytecode_ra: vec![opening_point.clone(); layout.bytecode()],
            ram_ra: vec![opening_point.clone(); layout.ram()],
            #[cfg(feature = "akita")]
            balanced_inc_digits: vec![
                opening_point.clone();
                self.dimensions.chunking().chunk_count()
            ],
            #[cfg(feature = "akita")]
            balanced_inc_carry: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &HammingWeightClaimReductionInputClaims<Vec<F>>,
        output_points: &HammingWeightClaimReductionOutputClaims<Vec<F>>,
        _challenges: &HammingWeightClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::HammingWeightClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let rho_rev = self.rho_reversed(output_points)?;
        match public_id {
            HammingWeightClaimReductionPublic::EqBooleanity => {
                try_eq_mle(rho_rev, &self.r_address).map_err(public_input_failed)
            }
            HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero => {
                Ok(eq_at_digit_zero(&self.r_address))
            }
            HammingWeightClaimReductionPublic::EqVirtualization(index) => {
                let point = self.virtualization_points.get(*index).ok_or_else(|| {
                    public_input_failed(format!(
                        "missing HammingWeight virtualization point for index {index}"
                    ))
                })?;
                try_eq_mle(rho_rev, point).map_err(public_input_failed)
            }
            HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(index) => {
                let point = self.virtualization_points.get(*index).ok_or_else(|| {
                    public_input_failed(format!(
                        "missing HammingWeight virtualization point for index {index}"
                    ))
                })?;
                Ok(eq_at_digit_zero(point))
            }
            HammingWeightClaimReductionPublic::BalancedIncValueAtAddress => {
                #[cfg(feature = "akita")]
                {
                    Ok(balanced_inc_value(rho_rev))
                }
                #[cfg(not(feature = "akita"))]
                {
                    Err(VerifierError::MissingStageClaimDerived { id: *id })
                }
            }
        }
    }

    /// The lattice input expression folds each leg's digit-zero baseline into the
    /// input claim, so the `*AtDigitZero` weights are input publics too. They are
    /// pure functions of the (transcript-fixed) stage-6 points — no bound point
    /// is needed.
    fn derive_input_term(
        &self,
        id: &JoltDerivedId,
        _challenges: &HammingWeightClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::HammingWeightClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero => {
                Ok(eq_at_digit_zero(&self.r_address))
            }
            HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(index) => {
                let point = self.virtualization_points.get(*index).ok_or_else(|| {
                    public_input_failed(format!(
                        "missing HammingWeight virtualization point for index {index}"
                    ))
                })?;
                Ok(eq_at_digit_zero(point))
            }
            // Output publics — resolved in `derive_output_term`, never in the
            // input expression.
            HammingWeightClaimReductionPublic::EqBooleanity
            | HammingWeightClaimReductionPublic::EqVirtualization(_)
            | HammingWeightClaimReductionPublic::BalancedIncValueAtAddress => {
                Err(VerifierError::MissingStageClaimDerived { id: *id })
            }
        }
    }
}
