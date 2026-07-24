//! The registers claim-reduction (stage 3) kernel: a naive member over the
//! cycle domain.
//!
//! The summand is
//! `eq(τ_low, j) · (rd_write_value + γ·rs1_value + γ²·rs2_value)(j)`,
//! degree 2. Its rs1/rs2 openings alias the instruction-input member's
//! (same polynomial, same point) — the generated drivers suppress the
//! duplicate absorbs.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::claim_reductions::registers::{
    rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::registers::RegistersClaimReductionChallenges;
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersClaimReductionPublic, TraceDimensions};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
use jolt_witness::JoltWitnessOracle;

use super::views::{dense_view, eq_table};
use crate::registers_claim_reduction::RegistersClaimReductionProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> RegistersClaimReductionProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_uniskip_tau_low: &[F],
        challenges: &RegistersClaimReductionChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>
    {
        let ids = [
            rd_write_value_reduced(),
            rs1_value_reduced(),
            rs2_value_reduced(),
        ];
        let opening_tables = ids
            .into_iter()
            .map(|id| Ok((id, Polynomial::new(dense_view(witness, id)?))))
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(product_uniskip_tau_low)),
        )]);

        let relation =
            RegistersClaimReduction::new(trace_dimensions, product_uniskip_tau_low.to_vec());
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
