//! The increment claim-reduction (stage 6b) kernel: a naive member over the
//! cycle domain.
//!
//! The summand is
//! `(eq(r_rw, j) + γ·eq(r_val, j)) · RamInc(j) + γ²·(eq(s_rw, j) + γ·eq(s_val, j)) · RdInc(j)`
//! — reducing the four upstream committed increment openings (RAM read-write /
//! val-check, register read-write / val-evaluation) to two fresh openings at
//! one cycle point. Both increment tables are the committed dense trace views;
//! each eq leaf is one multilinear over its upstream cycle point.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionChallenges;
use jolt_claims::protocols::jolt::{IncClaimReductionPublic, JoltDerivedId, TraceDimensions};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_witness::JoltWitnessOracle;

use super::views::{dense_view, eq_table};
use crate::inc_claim_reduction::IncClaimReductionProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> IncClaimReductionProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        cycle_points: &[Vec<F>; 4],
        challenges: &IncClaimReductionChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = IncClaimReduction<F>>>, KernelError<F>> {
        for point in cycle_points {
            if point.len() != trace_dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "increment reduction cycle point has the wrong variable count",
                });
            }
        }
        let [ram_read_write, ram_val_check, registers_read_write, registers_val_evaluation] =
            cycle_points.clone();
        let relation = IncClaimReduction::new(
            trace_dimensions,
            ram_read_write,
            ram_val_check,
            registers_read_write,
            registers_val_evaluation,
        );

        let opening_tables = BTreeMap::from([
            (
                ram_inc_reduced(),
                Polynomial::new(dense_view(witness, ram_inc_reduced())?),
            ),
            (
                rd_inc_reduced(),
                Polynomial::new(dense_view(witness, rd_inc_reduced())?),
            ),
        ]);
        let publics = [
            IncClaimReductionPublic::EqRamReadWrite,
            IncClaimReductionPublic::EqRamValCheck,
            IncClaimReductionPublic::EqRegistersReadWrite,
            IncClaimReductionPublic::EqRegistersValEvaluation,
        ];
        let derived_tables: BTreeMap<_, _> = publics
            .into_iter()
            .zip(cycle_points)
            .map(|(public, point)| {
                (
                    JoltDerivedId::from(public),
                    Polynomial::new(eq_table(point)),
                )
            })
            .collect();

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
