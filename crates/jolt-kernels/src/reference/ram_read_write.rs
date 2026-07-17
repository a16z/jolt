//! The RAM read/write-checking (stage 2) kernel: a naive member over the
//! joint `(address ‖ cycle)` domain.
//!
//! The summand `eq(τ_low, j) · ra(k,j) · (val(k,j) + γ·(val(k,j) + inc(j)))`
//! is one `Expr` over dense tables of size `2^(log_K + log_T)` in
//! address-major layout (`index = k·2^log_T + j`), bound `LowToHigh` — under
//! the default read-write config (phase 1 = all cycle rounds) the legacy
//! prover's phase transition only changes its data structures, never the
//! summand or the variable order, so the naive prover reproduces its round
//! polynomials exactly. The cycle-indexed `inc` and
//! eq tables are tiled across the address dimension.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::{ram_inc, ram_ra, ram_val};
use jolt_claims::protocols::jolt::relations::ram::RamReadWriteChallenges;
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{dense_view, eq_table, tile};
use crate::ram_read_write::RamReadWriteProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ReferenceBackend, SumcheckKernel};

impl<F: Field> RamReadWriteProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        tau_low: &[F],
        challenges: &RamReadWriteChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>
    {
        if dimensions.phase1_num_rounds() != dimensions.log_t() {
            return Err(KernelError::Unsupported {
                reason: "reference RAM read-write checking supports only the default \
                         read-write config (phase 1 = all cycle rounds)",
            });
        }

        let copies = 1usize << ram_log_k;
        let opening_tables = BTreeMap::from([
            (ram_val(), Polynomial::new(dense_view(witness, ram_val())?)),
            (ram_ra(), Polynomial::new(dense_view(witness, ram_ra())?)),
            (
                ram_inc(),
                Polynomial::new(tile(&dense_view(witness, ram_inc())?, copies)),
            ),
        ]);
        let derived_tables = BTreeMap::from([(
            jolt_claims::protocols::jolt::JoltDerivedId::from(
                jolt_claims::protocols::jolt::RamReadWritePublic::EqCycle,
            ),
            Polynomial::new(tile(&eq_table(tau_low), copies)),
        )]);

        let relation = RamReadWriteChecking::new(dimensions, ram_log_k, tau_low.to_vec());
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
