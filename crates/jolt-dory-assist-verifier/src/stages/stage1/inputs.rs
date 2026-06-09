//! Typed inputs consumed by stage 1.

use jolt_claims::protocols::dory_assist::{
    formulas::protocol::protocol_claims, DoryAssistDimensions, DoryAssistRelationId,
    DoryAssistSumcheckSpec,
};
use jolt_field::Fq;
use jolt_poly::CompressedPoly;
use jolt_sumcheck::CompressedSumcheckProof;
use serde::{Deserialize, Serialize};

use crate::verifier::CheckedInputs;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage1Proof {
    pub relations: Vec<Stage1RelationProof>,
}

impl Stage1Proof {
    pub fn canonical_for_dimensions(dimensions: DoryAssistDimensions) -> Self {
        Self {
            relations: canonical_stage1_relation_specs(dimensions)
                .into_iter()
                .map(Stage1RelationProof::zero_claim)
                .collect(),
        }
    }

    pub fn relation_count(&self) -> u32 {
        self.relations.len() as u32
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage1RelationProof {
    pub id: DoryAssistRelationId,
    pub sumcheck: DoryAssistSumcheckSpec,
    pub sumcheck_proof: CompressedSumcheckProof<Fq>,
}

impl Stage1RelationProof {
    fn zero_claim(spec: Stage1RelationSpec) -> Self {
        Self {
            id: spec.id,
            sumcheck: spec.sumcheck,
            sumcheck_proof: zero_compressed_sumcheck_proof(spec.sumcheck),
        }
    }

    pub(crate) const fn spec(&self) -> Stage1RelationSpec {
        Stage1RelationSpec {
            id: self.id,
            sumcheck: self.sumcheck,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stage1RelationSpec {
    pub id: DoryAssistRelationId,
    pub sumcheck: DoryAssistSumcheckSpec,
}

#[derive(Clone, Copy)]
pub struct Stage1Inputs<'a, 'p> {
    pub checked: &'a CheckedInputs<'p>,
    pub dimensions: DoryAssistDimensions,
    pub proof: &'a Stage1Proof,
    pub claims: &'a crate::proof::DoryAssistProofClaims,
}

#[expect(
    clippy::expect_used,
    reason = "stage 1 relation IDs are a subset of the canonical protocol catalog"
)]
pub(crate) fn canonical_stage1_relation_specs(
    dimensions: DoryAssistDimensions,
) -> Vec<Stage1RelationSpec> {
    let protocol = protocol_claims::<Fq>(dimensions);
    canonical_stage1_relation_ids(dimensions)
        .iter()
        .map(|id| {
            let relation = protocol
                .relation(*id)
                .expect("stage 1 relation belongs to canonical Dory-assist protocol");
            Stage1RelationSpec {
                id: relation.id,
                sumcheck: relation.sumcheck,
            }
        })
        .collect()
}

pub(crate) fn canonical_stage1_relation_ids(
    dimensions: DoryAssistDimensions,
) -> Vec<DoryAssistRelationId> {
    let mut ids = BASE_STAGE1_RELATION_IDS.to_vec();
    if dimensions.dory_reduce.reduce_rounds() > 1 {
        ids.push(DoryAssistRelationId::DoryReduceStateChain);
        ids.push(DoryAssistRelationId::DoryReduceBoundary);
    }
    ids
}

fn zero_compressed_sumcheck_proof(sumcheck: DoryAssistSumcheckSpec) -> CompressedSumcheckProof<Fq> {
    CompressedSumcheckProof {
        round_polynomials: (0..sumcheck.rounds)
            .map(|_| CompressedPoly::new(vec![Fq::default(); sumcheck.degree]))
            .collect(),
    }
}

pub(crate) const BASE_STAGE1_RELATION_IDS: [DoryAssistRelationId; 24] = [
    DoryAssistRelationId::GtExponentiation,
    DoryAssistRelationId::GtExponentiationDigitSelector,
    DoryAssistRelationId::GtExponentiationBasePower,
    DoryAssistRelationId::GtExponentiationDigitBitness,
    DoryAssistRelationId::GtExponentiationShift,
    DoryAssistRelationId::GtExponentiationBoundary,
    DoryAssistRelationId::GtMultiplication,
    DoryAssistRelationId::G1ScalarMultiplication,
    DoryAssistRelationId::G1ScalarMultiplicationShift,
    DoryAssistRelationId::G1ScalarMultiplicationBoundary,
    DoryAssistRelationId::G1Addition,
    DoryAssistRelationId::G2ScalarMultiplication,
    DoryAssistRelationId::G2ScalarMultiplicationShift,
    DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    DoryAssistRelationId::G2Addition,
    DoryAssistRelationId::MillerLoopLineStep,
    DoryAssistRelationId::MillerLoopLineEvaluation,
    DoryAssistRelationId::MillerLoopPairProduct,
    DoryAssistRelationId::MillerLoopAccumulator,
    DoryAssistRelationId::MillerLoopBoundary,
    DoryAssistRelationId::DoryReduceGtTransition,
    DoryAssistRelationId::DoryReduceG1Transition,
    DoryAssistRelationId::DoryReduceG2Transition,
    DoryAssistRelationId::DoryReduceScalarFold,
];

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests fail loudly on invalid fixture dimensions"
    )]

    use super::*;
    use jolt_claims::protocols::dory_assist::{
        DoryAssistDimensions, DoryReduceDimensions, G1Dimensions, G2Dimensions, GtDimensions,
        MillerLoopDimensions, PrefixPackingDimensions, WiringDimensions,
    };

    fn dimensions(reduce_rounds: usize) -> DoryAssistDimensions {
        DoryAssistDimensions::new(
            GtDimensions::new(7, 2, 3),
            G1Dimensions::new(8, 2, 3),
            G2Dimensions::new(8, 2, 3),
            MillerLoopDimensions::new(7, 2, 8),
            DoryReduceDimensions::new(2 * reduce_rounds, reduce_rounds),
            WiringDimensions::new(6),
            PrefixPackingDimensions::new(0, 0, 0).expect("valid empty packing dimensions"),
        )
    }

    #[test]
    fn singleton_stage1_relation_catalog_omits_dory_reduce_state_chain() {
        let ids = canonical_stage1_relation_ids(dimensions(1));

        assert_eq!(ids, BASE_STAGE1_RELATION_IDS);
        assert!(!ids.contains(&DoryAssistRelationId::DoryReduceStateChain));
    }

    #[test]
    fn multiround_stage1_relation_catalog_includes_dory_reduce_state_chain_and_boundary() {
        let dimensions = dimensions(2);
        let ids = canonical_stage1_relation_ids(dimensions);
        let specs = canonical_stage1_relation_specs(dimensions);

        assert_eq!(ids.len(), BASE_STAGE1_RELATION_IDS.len() + 2);
        assert_eq!(
            ids[BASE_STAGE1_RELATION_IDS.len()],
            DoryAssistRelationId::DoryReduceStateChain
        );
        assert_eq!(ids.last(), Some(&DoryAssistRelationId::DoryReduceBoundary));
        assert!(specs.iter().any(|spec| {
            spec.id == DoryAssistRelationId::DoryReduceStateChain
                && spec.sumcheck == dimensions.dory_reduce.state_chain_sumcheck()
        }));
        assert!(specs.iter().any(|spec| {
            spec.id == DoryAssistRelationId::DoryReduceBoundary
                && spec.sumcheck == dimensions.dory_reduce.boundary_sumcheck()
        }));
    }
}
