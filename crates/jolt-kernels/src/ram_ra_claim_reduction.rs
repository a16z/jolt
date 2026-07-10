//! The RAM RA claim-reduction (stage 5) kernel: a naive member over the
//! cycle domain.
//!
//! The summand is
//! `(eq(r_cycle_raf, j) + γ·eq(r_cycle_rw, j) + γ²·eq(r_cycle_val, j)) · ra(r_address, j)`
//! — the three upstream `RamRa` openings share an address prefix (the
//! verifier's `derive_opening_points` hard-checks it), so one address-bound
//! fold of the `(K × T)` RAM `ra` grid serves all three eq-batched terms.
//! Each `EqCycle*` derived leaf is ONE multilinear: the eq table over the
//! corresponding upstream point's cycle suffix.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_ra_claim_reduction;
use jolt_claims::protocols::jolt::relations::ram::{
    RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaClaimReductionPublic, TraceDimensions};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{dense_view, eq_table};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-5 RAM RA claim-reduction slot.
pub trait RamRaClaimReductionProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        input_points: &RamRaClaimReductionInputClaims<Vec<F>>,
        challenges: &RamRaClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>>;
}

impl<F: Field> RamRaClaimReductionProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        input_points: &RamRaClaimReductionInputClaims<Vec<F>>,
        challenges: &RamRaClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>> {
        let cycles = 1usize << trace_dimensions.log_t();
        let addresses = 1usize << ram_log_k;
        let expected_len = ram_log_k + trace_dimensions.log_t();
        for point in [
            input_points.raf(),
            input_points.read_write(),
            input_points.val_check(),
        ] {
            if point.len() != expected_len {
                return Err(KernelError::Unsupported {
                    reason: "RAM RA claim-reduction input point has the wrong variable count",
                });
            }
        }
        // The shared address prefix (the relation's `derive_opening_points`
        // hard-checks that all three inputs agree on it).
        let r_address = &input_points.read_write()[..ram_log_k];

        let eq_address = eq_table(r_address);
        let ram_ra_full = dense_view(witness, ram_ra_claim_reduction())?;
        let ra_folded: Vec<F> = (0..cycles)
            .map(|j| {
                (0..addresses)
                    .map(|k| ram_ra_full[(k << trace_dimensions.log_t()) | j] * eq_address[k])
                    .sum()
            })
            .collect();

        let opening_tables =
            BTreeMap::from([(ram_ra_claim_reduction(), Polynomial::new(ra_folded))]);
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleRaf),
                Polynomial::new(eq_table(&input_points.raf()[ram_log_k..])),
            ),
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                Polynomial::new(eq_table(&input_points.read_write()[ram_log_k..])),
            ),
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleValCheck),
                Polynomial::new(eq_table(&input_points.val_check()[ram_log_k..])),
            ),
        ]);

        let relation = RamRaClaimReduction::new(trace_dimensions, ram_log_k);
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
