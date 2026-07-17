//! The stage 7 `HammingWeightClaimReduction` sumcheck instance.
//!
//! Reduces the per-family RA booleanity, virtualization, and hamming-weight claims
//! (instruction, bytecode, RAM) into the one-hot `Ra` opening claims that anchor the
//! stage 8 final batched opening. Owns the shared opening-point derivation and the
//! `EqBooleanity` / `EqVirtualization` public-value computation.

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::relations::hamming_weight as lattice_hamming;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::relations;
#[cfg(feature = "akita")]
pub use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
#[cfg(not(feature = "akita"))]
pub use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::{
    HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    HammingWeightClaimReductionOutputClaims,
};
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltDerivedId, JoltRelationId,
};
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;
#[cfg(feature = "akita")]
pub use lattice_hamming::{
    LatticeHammingWeightClaimReductionInputClaims as HammingWeightClaimReductionInputClaims,
    LatticeHammingWeightClaimReductionOutputClaims as HammingWeightClaimReductionOutputClaims,
};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage6b::outputs::{Stage6bOutputClaims, Stage6bOutputPoints};
use crate::VerifierError;

/// The hamming reduction's consumed opening *values*, wired from the stage-6b
/// cycle-phase output claims. The relation reads only their values (its produced
/// points are derived from its own sumcheck point), so no input points are needed.
pub fn hamming_weight_input_values_from_upstream<F: Field>(
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
        unsigned_inc_chunk_booleanity: cycle_phase.booleanity.unsigned_inc_chunks.clone(),
        #[cfg(feature = "akita")]
        unsigned_inc_msb_booleanity: cycle_phase.booleanity.unsigned_inc_msb,
        #[cfg(feature = "akita")]
        fused_inc: cycle_phase.fused_inc_claim_reduction.fused_inc,
    }
}

/// The per-RA virtualization address chunks the hamming reduction's
/// `EqVirtualization` publics compare against, in canonical (instruction, bytecode,
/// RAM) order: the leading `log_k_chunk` coordinates of each stage-6b RA
/// virtualization opening point.
pub fn stage7_hamming_virtualization_address_points<F: Field>(
    dimensions: HammingDimensions,
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

pub struct HammingWeightClaimReduction<F: Field> {
    symbolic: HammingSymbolic,
    dimensions: HammingDimensions,
    /// The shared cycle suffix appended to every produced opening point (the
    /// stage-6 booleanity cycle point).
    r_cycle: Vec<F>,
    /// The stage-6 booleanity address point that `EqBooleanity` compares against.
    r_address: Vec<F>,
    /// The per-RA virtualization address chunks (one per layout polynomial, in
    /// canonical order) that `EqVirtualization(i)` compares against.
    virtualization_points: Vec<Vec<F>>,
}

impl<F: Field> HammingWeightClaimReduction<F> {
    pub fn new(
        dimensions: HammingDimensions,
        r_cycle: Vec<F>,
        r_address: Vec<F>,
        virtualization_points: Vec<Vec<F>>,
    ) -> Self {
        Self {
            symbolic: HammingSymbolic::new(dimensions),
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
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for HammingWeightClaimReduction<F> {
    type Symbolic = HammingSymbolic;

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
            unsigned_inc_chunks: vec![
                opening_point.clone();
                self.dimensions.chunking().chunk_count()
            ],
            #[cfg(feature = "akita")]
            unsigned_inc_msb: opening_point,
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
            HammingWeightClaimReductionPublic::EqVirtualization(index) => {
                let point = self.virtualization_points.get(*index).ok_or_else(|| {
                    public_input_failed(format!(
                        "missing HammingWeight virtualization point for index {index}"
                    ))
                })?;
                try_eq_mle(rho_rev, point).map_err(public_input_failed)
            }
            HammingWeightClaimReductionPublic::IdentityAtAddress => {
                #[cfg(feature = "akita")]
                {
                    let mut value = F::zero();
                    let mut weight = F::one();
                    for challenge in rho_rev.iter().rev() {
                        value += weight * *challenge;
                        weight = weight + weight;
                    }
                    Ok(value)
                }
                #[cfg(not(feature = "akita"))]
                {
                    Err(VerifierError::MissingStageClaimDerived { id: *id })
                }
            }
        }
    }
}

#[cfg(not(feature = "akita"))]
pub(crate) type HammingDimensions =
    jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
#[cfg(feature = "akita")]
pub(crate) type HammingDimensions = lattice_hamming::LatticeHammingWeightClaimReductionDimensions;

#[cfg(not(feature = "akita"))]
type HammingSymbolic = relations::claim_reductions::hamming_weight::ClaimReduction;
#[cfg(feature = "akita")]
type HammingSymbolic = lattice_hamming::LatticeHammingWeightClaimReduction;
