//! The stage 6 `RamRaVirtualization` cycle-phase sumcheck instance.
//!
//! Virtualizes the single reduced `RamRa` claim (from the stage-5 RAM RA claim
//! reduction) into the per-chunk committed `RamRa` openings that the stage-7
//! hamming-weight reduction consumes. Each produced opening shares the cycle
//! suffix derived from this sumcheck; the address prefix comes from the stage-5
//! reduced opening point. Its only public, `EqCycle`, ties the produced cycle to
//! the reduced cycle.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamRaVirtualizationInputClaims, RamRaVirtualizationOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{dimensions::committed_address_chunks, ram::RamRaVirtualizationDimensions},
    JoltDerivedId, JoltRelationId, RamRaVirtualizationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage5::Stage5ClearOutput;
use crate::VerifierError;

/// Wire the single reduced `RamRa` opening from the stage-5 RAM RA claim
/// reduction. (Verifier-side constructor for the moved
/// [`RamRaVirtualizationInputClaims`].)
pub fn ram_ra_virtualization_inputs_from_upstream<F: Field>(
    stage5: &Stage5ClearOutput<F>,
) -> RamRaVirtualizationInputClaims<OpeningClaim<F>> {
    RamRaVirtualizationInputClaims {
        ram_ra_reduced: stage5.output_claims.ram_ra_claim_reduction.ram_ra.clone(),
    }
}

pub struct RamRaVirtualization<F: Field> {
    symbolic: relations::ram::RaVirtualization,
    dimensions: RamRaVirtualizationDimensions,
    /// The stage-5 reduced address prefix, chunked into the per-chunk committed
    /// opening points.
    ram_reduced_address: Vec<F>,
    /// The stage-5 reduced cycle suffix that `EqCycle` compares the produced cycle
    /// against.
    ram_reduced_cycle: Vec<F>,
    committed_chunk_bits: usize,
}

impl<F: Field> RamRaVirtualization<F> {
    pub fn new(
        dimensions: RamRaVirtualizationDimensions,
        ram_reduced_address: Vec<F>,
        ram_reduced_cycle: Vec<F>,
        committed_chunk_bits: usize,
    ) -> Self {
        Self {
            symbolic: relations::ram::RaVirtualization::new(dimensions),
            dimensions,
            ram_reduced_address,
            ram_reduced_cycle,
            committed_chunk_bits,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamRaVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamRaVirtualization<F> {
    type Symbolic = relations::ram::RaVirtualization;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamRaVirtualizationInputClaims<C>,
    ) -> Result<RamRaVirtualizationOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let ram_ra = committed_address_chunks(&self.ram_reduced_address, self.committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
            .collect();
        Ok(RamRaVirtualizationOutputClaims { ram_ra })
    }

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &RamRaVirtualizationInputClaims<C>,
        outputs: &RamRaVirtualizationOutputClaims<OpeningClaim<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamRaVirtualization(RamRaVirtualizationPublic::EqCycle) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let log_t = self.dimensions.log_t();
        let point = outputs
            .ram_ra
            .first()
            .map(GetPoint::point)
            .ok_or_else(|| public_input_failed("RAM RA virtualization produced no openings"))?;
        let r_cycle = point
            .get(point.len() - log_t..)
            .ok_or_else(|| public_input_failed("RAM RA opening point shorter than log_t"))?;
        try_eq_mle(&self.ram_reduced_cycle, r_cycle).map_err(public_input_failed)
    }
}
