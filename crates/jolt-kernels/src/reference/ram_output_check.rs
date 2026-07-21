//! The RAM output-check (stage 2) kernel: a naive member over the address
//! domain with a zero input claim.
//!
//! The summand is `eq(r_address, k) · mask(k) · (val_final(k) − val_io(k))`
//! with `mask` the `[io_start, io_end)` indicator and `val_io` the committed
//! public-IO words (zero outside the segments) — each derived leaf one
//! multilinear table, built pointwise from the public IO memory and pinned
//! by the `derive_output_term` cross-check at the bound point.
//!
//! Only the default read-write config is supported (phase 1 = all cycle
//! rounds), where the relation's rounds equal `log_K`. The legacy prover's
//! leading zero-address rounds emit 1-coefficient constant polynomials whose
//! true round polynomial is zero — the naive member computes those zeros
//! literally, and the engine's batched-polynomial trim reproduces the wire
//! lengths.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamOutputCheckPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamOutputCheck<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamOutputCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamOutputCheck<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.read_write_dimensions();
        let output_address_challenges = relation.output_address_challenges();
        let ram_log_k = output_address_challenges.len();
        let public_memory = relation.public_memory();
        if dimensions.output_check_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "reference RAM output check supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }

        let addresses = 1usize << ram_log_k;
        let mut val_io = vec![F::zero(); addresses];
        for segment in &public_memory.segments {
            for (offset, &word) in segment.words.iter().enumerate() {
                let index = segment.start_index as usize + offset;
                if index < addresses {
                    val_io[index] = F::from_u64(word);
                }
            }
        }
        let io_mask: Vec<F> = (0..addresses)
            .map(|k| {
                let in_io_region = (k as u128) >= public_memory.io_mask_start
                    && (k as u128) < public_memory.io_mask_end;
                if in_io_region {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();

        let opening_tables = BTreeMap::from([(
            ram_val_final(),
            Polynomial::new(dense_view(witness, ram_val_final())?),
        )]);
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::from(RamOutputCheckPublic::EqAddress),
                Polynomial::new(eq_table(output_address_challenges)),
            ),
            (
                JoltDerivedId::from(RamOutputCheckPublic::IoMask),
                Polynomial::new(io_mask),
            ),
            (
                JoltDerivedId::from(RamOutputCheckPublic::ValIo),
                Polynomial::new(val_io),
            ),
        ]);

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            inputs.challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
