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

use jolt_claims::protocols::jolt::geometry::ram::{
    ram_ra_raf_evaluation, RamRafEvaluationDimensions,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic, ReadWriteDimensions};
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{dense_view, eq_table};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-2 RAM RAF-evaluation slot.
pub trait RamRafEvaluationProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        raf_dimensions: RamRafEvaluationDimensions,
        ram_log_k: usize,
        lowest_address: u64,
        tau_low: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRafEvaluation<F>>>, KernelError<F>>;
}

impl<F: Field> RamRafEvaluationProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        raf_dimensions: RamRafEvaluationDimensions,
        ram_log_k: usize,
        lowest_address: u64,
        tau_low: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRafEvaluation<F>>>, KernelError<F>> {
        if dimensions.raf_evaluation_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "reference RAM RAF evaluation supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }

        let cycles = 1usize << tau_low.len();
        let addresses = 1usize << ram_log_k;
        let eq_cycle = eq_table(tau_low);
        let ram_ra_full = dense_view(witness, ram_ra_raf_evaluation())?;
        let ra_folded: Vec<F> = (0..addresses)
            .map(|k| {
                (0..cycles)
                    .map(|j| ram_ra_full[(k << tau_low.len()) | j] * eq_cycle[j])
                    .sum()
            })
            .collect();
        let unmap: Vec<F> = (0..addresses as u64)
            .map(|k| F::from_u64(8 * k + lowest_address))
            .collect();

        let opening_tables =
            BTreeMap::from([(ram_ra_raf_evaluation(), Polynomial::new(ra_folded))]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
            Polynomial::new(unmap),
        )]);

        let relation = RamRafEvaluation::new(
            dimensions,
            raf_dimensions,
            ram_log_k,
            lowest_address,
            tau_low.to_vec(),
        );
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            &NoChallenges::default(),
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
