//! The Hamming-weight claim-reduction (stage 7) kernel: a naive member over
//! the committed chunk domain.
//!
//! The summand is
//! `Σ_i G_i(k) · (γ^{3i} + γ^{3i+1}·eq(r_addr_bool, k) + γ^{3i+2}·eq(r_addr_virt_i, k))`
//! — reducing each checked one-hot polynomial's Hamming-weight, booleanity,
//! and virtualization claims to one fresh opening. Each `G_i(k) =
//! Σ_j eq(r_cycle, j) · ra_i(k, j)` is the cycle fold of the committed one-hot
//! grid at the shared stage-6b cycle point (the booleanity opening point's
//! cycle suffix — every stage-6b member bound the same cycle challenges, so
//! all three reduced claim families live at that cycle). The eq publics are
//! one multilinear each over the chunk domain.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
use jolt_claims::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltDerivedId, JoltRelationId,
};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{cycle_fold, eq_table};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-7 Hamming-weight claim-reduction slot. `r_cycle` and `r_address`
/// are the stage-6b booleanity opening point's splits; `virtualization_points`
/// are the leading chunk coordinates of each stage-6b RA virtualization
/// opening point, in canonical layout order.
pub trait HammingWeightClaimReductionProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: HammingWeightClaimReductionDimensions,
        r_cycle: &[F],
        r_address: &[F],
        virtualization_points: &[Vec<F>],
        challenges: &HammingWeightClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>;
}

impl<F: Field> HammingWeightClaimReductionProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: HammingWeightClaimReductionDimensions,
        r_cycle: &[F],
        r_address: &[F],
        virtualization_points: &[Vec<F>],
        challenges: &HammingWeightClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        if r_address.len() != dimensions.log_k_chunk
            || virtualization_points.len() != dimensions.layout.total()
        {
            return Err(KernelError::Unsupported {
                reason: "hamming reduction reference point shapes disagree with the layout",
            });
        }
        let relation = HammingWeightClaimReduction::new(
            dimensions,
            r_cycle.to_vec(),
            r_address.to_vec(),
            virtualization_points.to_vec(),
        );

        let mut opening_tables = BTreeMap::new();
        for opening in dimensions
            .layout
            .openings(JoltRelationId::HammingWeightClaimReduction)
        {
            let _ = opening_tables.insert(
                opening,
                Polynomial::new(cycle_fold(
                    witness,
                    opening,
                    dimensions.log_k_chunk,
                    r_cycle,
                )?),
            );
        }

        let mut derived_tables = BTreeMap::from([(
            JoltDerivedId::from(HammingWeightClaimReductionPublic::EqBooleanity),
            Polynomial::new(eq_table(r_address)),
        )]);
        for (index, point) in virtualization_points.iter().enumerate() {
            let _ = derived_tables.insert(
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(index)),
                Polynomial::new(eq_table(point)),
            );
        }

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
