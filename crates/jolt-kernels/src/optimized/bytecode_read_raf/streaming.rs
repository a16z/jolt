//! Streaming bytecode read-RAF kernels: the default kernels over the
//! witness's resident rows instead of the packed row cache the default
//! kernels share. The address phase accumulates its stage pushforwards from
//! the rows in the default kernel's walk, which stops at the rows' constant
//! suffix and weighs that suffix once.

use jolt_field::JoltField;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_witness::witnesses::BytecodePc;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::{FusedInc, ToField};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

use super::AddressKernel;
use crate::optimized::support::BundleStore;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const ROWS: &str = "bytecode read-RAF trace rows";

/// What the bytecode read-RAF kernels read of a cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct BytecodeRow {
    bytecode_pc: BytecodePc,
    #[cfg(feature = "akita")]
    fused_inc: FusedInc,
}

/// The streaming stage-6a bytecode read-RAF address slot.
pub struct StreamingBytecodeReadRafAddress;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafAddressPhase<F>>
    for StreamingBytecodeReadRafAddress
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.dimensions().log_t();
        let rows = BundleStore::<BytecodeRow>::resident(witness, log_t, ROWS)?;
        let access = rows.access();
        // The rows span `2^log_t` cycles, so the shift cannot overflow. The
        // walk weighs the suffix from `prefix` on by its row there, so
        // `prefix` stays inside the domain.
        let last = (1usize << log_t) - 1;
        let prefix = access.constant_suffix_start(last + 1)?.min(last);
        // `AddressKernel::new` reads a row only once this table is known to
        // hold the relation's address count, the length of its pushforwards.
        let table = witness.program_preprocessing().bytecode.bytecode.len();
        let row_at = |cycle: usize| {
            let row = access.row(cycle)?;
            if row.bytecode_pc.0 >= table {
                return Err(WitnessError::InvalidWitnessData {
                    label: ROWS,
                    reason: format!(
                        "cycle {cycle} has bytecode index {} in a table of {table} rows",
                        row.bytecode_pc.0
                    ),
                });
            }
            Ok(row)
        };
        #[cfg(not(feature = "akita"))]
        let row_weight = |_: &BytecodeRow| F::one();
        #[cfg(feature = "akita")]
        let row_weight = |row: &BytecodeRow| row.fused_inc.to_field::<F>();
        Ok(Box::new(AddressKernel::new(
            witness,
            &inputs,
            row_at,
            prefix,
            |row| row.bytecode_pc.0,
            row_weight,
        )?))
    }
}
