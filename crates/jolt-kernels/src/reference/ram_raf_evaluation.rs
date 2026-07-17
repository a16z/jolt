//! The RAM RAF-evaluation (stage 2) kernel: a naive member over the address
//! domain.
//!
//! The summand `unmap(k) · ra_folded(k)` where
//! `ra_folded(k) = Σ_j eq(τ_low, j) · RamRa(k, j)` is the cycle-folded RAM
//! `ra` (its opening point is `[r_address ‖ τ_low]` — the cycle part is
//! stage 1's point, pre-folded into the table) and
//! `unmap(k) = 8k + lowest_address` is affine, hence a multilinear leaf.
//!
//! Only the default read-write config is supported (phase 1 = all cycle
//! rounds): then the relation's rounds equal `log_K` and no dummy cycle-gap
//! rounds or `2^gap` scalings exist.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_ra_raf_evaluation;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::cycle_fold;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamRafEvaluation<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        inputs: ProverInputs<'_, F, RamRafEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRafEvaluation<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.read_write_dimensions();
        let ram_log_k = relation.ram_log_k();
        let lowest_address = relation.lowest_address();
        let tau_low = relation.tau_low();
        if dimensions.raf_evaluation_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "reference RAM RAF evaluation supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }

        let addresses = 1usize << ram_log_k;
        let ra_folded = cycle_fold(witness, ram_ra_raf_evaluation(), ram_log_k, tau_low)?;
        let unmap: Vec<F> = (0..addresses as u64)
            .map(|k| F::from_u64(8 * k + lowest_address))
            .collect();

        let opening_tables =
            BTreeMap::from([(ram_ra_raf_evaluation(), Polynomial::new(ra_folded))]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
            Polynomial::new(unmap),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            inputs.challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
