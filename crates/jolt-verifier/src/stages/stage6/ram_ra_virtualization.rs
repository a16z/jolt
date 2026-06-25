//! The stage 6 `RamRaVirtualization` cycle-phase sumcheck instance.
//!
//! Virtualizes the single reduced `RamRa` claim (from the stage-5 RAM RA claim
//! reduction) into the per-chunk committed `RamRa` openings that the stage-7
//! hamming-weight reduction consumes. Each produced opening shares the cycle
//! suffix derived from this sumcheck; the address prefix comes from the stage-5
//! reduced opening point. Its only public, `EqCycle`, ties the produced cycle to
//! the reduced cycle.

use jolt_claims::protocols::jolt::relations;
use jolt_claims::protocols::jolt::{
    geometry::{dimensions::committed_address_chunks, ram::RamRaVirtualizationDimensions},
    JoltPublicId, JoltRelationId, RamRaVirtualizationPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage5::Stage5ClearOutput;
use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRaVirtualization)]
pub struct RamRaVirtualizationOutputClaims<C> {
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// The single reduced `RamRa` opening from the stage-5 RAM RA claim reduction.
#[derive(Clone, Debug, InputClaims)]
pub struct RamRaVirtualizationInputClaims<C> {
    #[opening(RamRa, from = RamRaClaimReduction)]
    pub ram_ra_reduced: C,
}

impl<F: Field> RamRaVirtualizationInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(stage5: &Stage5ClearOutput<F>) -> Self {
        Self {
            ram_ra_reduced: stage5.output_claims.ram_ra_claim_reduction.ram_ra.clone(),
        }
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
    type Inputs<C> = RamRaVirtualizationInputClaims<C>;
    type Outputs<C> = RamRaVirtualizationOutputClaims<C>;

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

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &RamRaVirtualizationInputClaims<C>,
        outputs: Option<&RamRaVirtualizationOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::RamRaVirtualization(RamRaVirtualizationPublic::EqCycle) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
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
