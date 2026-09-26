//! The RAM RAF-evaluation (stage 2) kernel: a naive member over the address
//! domain.
//!
//! The summand `unmap(k) · ra_folded(k)` where
//! `ra_folded(k) = Σ_j eq(τ_low, j) · RamRa(k, j)` is the cycle-folded RAM
//! `ra` (its opening point is `[r_address ‖ τ_low]` — the cycle part is
//! stage 1's point, pre-folded into the table) and
//! `unmap(k) = 8k + lowest_address` is affine, hence a multilinear leaf.
//!
//! Address tables repeat across the configured unused cycle variables.
//! The naive evaluator supplies their `2^gap` scaling and constant rounds.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::ram::ram_ra_raf_evaluation;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic};
use jolt_field::JoltField;
use jolt_poly::BindingOrder;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::JoltWitnessPlane;

use super::read_write::ReadWriteTableLayout;
use super::views::cycle_fold;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: JoltField> PrepareKernel<F, RamRafEvaluation<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRafEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRafEvaluation<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.read_write_dimensions();
        let ram_log_k = relation.ram_log_k();
        let lowest_address = relation.lowest_address();
        let tau_low = relation.tau_low();
        let layout = ReadWriteTableLayout::address::<F>(dimensions)?;

        let addresses = 1usize << ram_log_k;
        let ra_folded = cycle_fold(witness, ram_ra_raf_evaluation(), ram_log_k, tau_low)?;
        let unmap: Vec<F> = (0..addresses as u64)
            .map(|k| F::from_u64(8 * k + lowest_address))
            .collect();

        let opening_tables = BTreeMap::from([(ram_ra_raf_evaluation(), layout.table(ra_folded)?)]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
            layout.table(unmap)?,
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
