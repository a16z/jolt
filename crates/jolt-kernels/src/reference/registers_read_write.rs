//! The registers read/write-checking (stage 4) kernel: a naive member over
//! the joint `(register ‖ cycle)` domain.
//!
//! The summand
//! `eq(r_cycle, j) · (rd_wa·(rd_inc + val) + γ·rs1_ra·val + γ²·rs2_ra·val)(k, j)`
//! is one `Expr` over dense tables of size `2^(7 + log_T)` in register-major
//! layout (`index = k·2^log_T + j`), bound `LowToHigh` — under the default
//! read-write config (phase 1 = all cycle rounds) the legacy prover's phase
//! transition only changes its data structures. The cycle-indexed `rd_inc`
//! and eq tables are tiled across the register dimension.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::registers::{
    rd_inc_read_write, rd_wa_read_write, registers_val_read_write, rs1_ra_read_write,
    rs2_ra_read_write,
};
use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteChallenges;
use jolt_claims::protocols::jolt::{JoltDerivedId, ReadWriteDimensions, RegistersReadWritePublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{dense_view, eq_table, tile};
use crate::registers_read_write::RegistersReadWriteProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ReferenceBackend, SumcheckKernel};

impl<F: Field> RegistersReadWriteProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        r_cycle: &[F],
        challenges: &RegistersReadWriteChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        if dimensions.phase1_num_rounds() != dimensions.log_t() {
            return Err(KernelError::Unsupported {
                reason: "reference registers read-write checking supports only the default \
                         read-write config (phase 1 = all cycle rounds)",
            });
        }

        let copies = 1usize << dimensions.log_k();
        let opening_tables = BTreeMap::from([
            (
                registers_val_read_write(),
                Polynomial::new(dense_view(witness, registers_val_read_write())?),
            ),
            (
                rs1_ra_read_write(),
                Polynomial::new(dense_view(witness, rs1_ra_read_write())?),
            ),
            (
                rs2_ra_read_write(),
                Polynomial::new(dense_view(witness, rs2_ra_read_write())?),
            ),
            (
                rd_wa_read_write(),
                Polynomial::new(dense_view(witness, rd_wa_read_write())?),
            ),
            (
                rd_inc_read_write(),
                Polynomial::new(tile(&dense_view(witness, rd_inc_read_write())?, copies)),
            ),
        ]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RegistersReadWritePublic::EqCycle),
            Polynomial::new(tile(&eq_table(r_cycle), copies)),
        )]);

        let relation = RegistersReadWriteChecking::new(dimensions);
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
