//! Typed inputs consumed by stage 2.

use jolt_claims::protocols::dory_assist::{
    formulas::{composition, dory_reduce},
    DoryAssistCopyConstraint, DoryAssistDimensions,
};
use serde::{Deserialize, Serialize};

use crate::{proof::DoryAssistProofClaims, stages::stage1::Stage1Output, verifier::CheckedInputs};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage2Proof {
    pub copy_constraints: Vec<DoryAssistCopyConstraint>,
}

impl Stage2Proof {
    pub fn canonical_for_dimensions(dimensions: DoryAssistDimensions) -> Self {
        Self {
            copy_constraints: canonical_stage2_copy_constraints(dimensions),
        }
    }

    pub fn relation_count(&self) -> u32 {
        self.copy_constraints.len() as u32
    }
}

#[derive(Clone, Copy)]
pub struct Stage2Inputs<'a, 'p> {
    pub checked: &'a CheckedInputs<'p>,
    pub dimensions: DoryAssistDimensions,
    pub proof: &'a Stage2Proof,
    pub claims: &'a DoryAssistProofClaims,
    pub stage1: &'a Stage1Output,
}

pub(crate) fn canonical_stage2_copy_constraints(
    dimensions: DoryAssistDimensions,
) -> Vec<DoryAssistCopyConstraint> {
    let dory_reduce_copies = if dimensions.dory_reduce.reduce_rounds() == 1 {
        dory_reduce::initial_state_copy_constraints()
            .into_iter()
            .chain(dory_reduce::proof_artifact_copy_constraints(0))
            .chain(dory_reduce::round_setup_artifact_copy_constraints(
                dimensions.dory_reduce.reduce_rounds(),
                0,
            ))
            .chain(dory_reduce::transition_transcript_scalar_copy_constraints(
                dimensions.dory_reduce.point_len(),
                0,
            ))
            .chain(dory_reduce::scalar_fold_transcript_scalar_copy_constraints(
                dimensions.dory_reduce.point_len(),
                0,
            ))
            .collect()
    } else {
        Vec::new()
    };

    composition::public_input_copy_constraints()
        .into_iter()
        .chain(composition::gt_copy_constraints())
        .chain(composition::g1_copy_constraints())
        .chain(composition::g2_copy_constraints())
        .chain(composition::miller_loop_copy_constraints())
        .chain(dory_reduce_copies)
        .collect()
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests fail loudly on invalid fixture dimensions"
    )]

    use super::*;
    use jolt_claims::protocols::dory_assist::{
        DoryAssistRelationId, DoryAssistValueRef, DoryReduceDimensions, G1Dimensions, G2Dimensions,
        GtDimensions, MillerLoopDimensions, PrefixPackingDimensions, WiringDimensions,
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
    fn singleton_stage2_copy_catalog_keeps_direct_dory_reduce_copies() {
        let constraints = canonical_stage2_copy_constraints(dimensions(1));

        assert!(constraints.iter().any(has_dory_reduce_endpoint));
    }

    #[test]
    fn multiround_stage2_copy_catalog_does_not_use_direct_dory_reduce_row_chains() {
        let constraints = canonical_stage2_copy_constraints(dimensions(2));

        assert!(!constraints.iter().any(has_dory_reduce_endpoint));
    }

    fn has_dory_reduce_endpoint(constraint: &DoryAssistCopyConstraint) -> bool {
        [constraint.source, constraint.target]
            .into_iter()
            .any(|endpoint| {
                matches!(
                    endpoint,
                    DoryAssistValueRef::Witness {
                        relation: DoryAssistRelationId::DoryReduceGtTransition
                            | DoryAssistRelationId::DoryReduceG1Transition
                            | DoryAssistRelationId::DoryReduceG2Transition
                            | DoryAssistRelationId::DoryReduceScalarFold
                            | DoryAssistRelationId::DoryReduceStateChain
                            | DoryAssistRelationId::DoryReduceBoundary,
                        ..
                    }
                )
            })
    }
}
