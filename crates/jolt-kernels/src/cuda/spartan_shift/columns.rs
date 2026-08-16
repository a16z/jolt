use cudarc::driver::PushKernelArg;

use super::witness::{Packed, FIRST_IN_SEQUENCE_BIT, IS_NOOP_BIT, NARROW, VIRTUAL_INSTRUCTION_BIT};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

pub const COLUMNS: usize = 5;

pub const UNEXPANDED_PC: usize = 0;
pub const PC: usize = 1;
pub const IS_VIRTUAL: usize = 2;
pub const IS_FIRST_IN_SEQUENCE: usize = 3;
pub const IS_NOOP: usize = 4;

pub fn upload(context: &CudaKernelContext, packed: &Packed) -> Result<Vec<DeviceFrVec>, CudaError> {
    let cycles = packed.flags.len();
    if cycles == 0 || packed.narrow.len() != cycles * NARROW {
        return Err(CudaError::InvariantViolation {
            reason: "the packed Spartan shift columns must hold one entry per cycle per slot",
        });
    }

    let narrow = context.upload_u64_slice(&packed.narrow)?;
    let flags = context.upload_u32_slice(&packed.flags)?;
    let mut unexpanded_pc = context.alloc(cycles)?;
    let mut pc = context.alloc(cycles)?;
    let mut is_virtual = context.alloc(cycles)?;
    let mut is_first_in_sequence = context.alloc(cycles)?;
    let mut is_noop = context.alloc(cycles)?;

    let count = CudaKernelContext::count_of(cycles)?;
    let mut builder = context.stream().launch_builder(context.ss_columns());
    let _ = builder.arg(&narrow);
    let _ = builder.arg(&flags);
    let _ = builder.arg(unexpanded_pc.limbs_mut());
    let _ = builder.arg(pc.limbs_mut());
    let _ = builder.arg(is_virtual.limbs_mut());
    let _ = builder.arg(is_first_in_sequence.limbs_mut());
    let _ = builder.arg(is_noop.limbs_mut());
    let _ = builder.arg(&VIRTUAL_INSTRUCTION_BIT);
    let _ = builder.arg(&FIRST_IN_SEQUENCE_BIT);
    let _ = builder.arg(&IS_NOOP_BIT);
    let _ = builder.arg(&count);
    // SAFETY: thread `t < cycles` reads `flags[t]` (one u32 of a `cycles`-entry
    // buffer) and the two slots of row `t` (inside `narrow`'s `cycles * NARROW`
    // u64s), and writes `out[t*LIMBS..t*LIMBS+4]` of each of the five output
    // buffers, every one a fresh `cycles`-element allocation distinct from the
    // two inputs and from each other. Threads with `t >= cycles` return before
    // any access.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

    Ok(vec![
        unexpanded_pc,
        pc,
        is_virtual,
        is_first_in_sequence,
        is_noop,
    ])
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::ToField;
    use proptest::prelude::*;

    use super::super::witness::{pack, sample_rows, SpartanShiftWitness};
    use super::{upload, COLUMNS};
    use crate::cuda::common::context::shared_context;

    fn host_columns(rows: &[SpartanShiftWitness]) -> Vec<Vec<Fr>> {
        let mut columns: Vec<Vec<Fr>> = (0..COLUMNS)
            .map(|_| Vec::with_capacity(rows.len()))
            .collect();
        for row in rows {
            columns[super::UNEXPANDED_PC].push(ToField::to_field(row.unexpanded_pc));
            columns[super::PC].push(ToField::to_field(row.pc));
            columns[super::IS_VIRTUAL].push(ToField::to_field(row.virtual_instruction));
            columns[super::IS_FIRST_IN_SEQUENCE].push(ToField::to_field(row.is_first_in_sequence));
            columns[super::IS_NOOP].push(ToField::to_field(row.is_noop));
        }
        columns
    }

    #[test]
    fn empty_packing_is_rejected() {
        let Some(context) = shared_context() else {
            return;
        };
        let packed = pack(&[]);
        assert!(
            upload(context, &packed).is_err(),
            "a zero-cycle packing must not produce columns",
        );
    }

    #[test]
    fn flag_columns_are_zero_or_one() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = sample_rows(3, 1 << 6);
        let packed = pack(&rows);
        let got = upload(context, &packed).expect("device shift columns");
        for column in [
            super::IS_VIRTUAL,
            super::IS_FIRST_IN_SEQUENCE,
            super::IS_NOOP,
        ] {
            let values = got[column].to_host().expect("download");
            assert!(
                values
                    .iter()
                    .all(|value| *value == Fr::from_u64(0) || *value == Fr::from_u64(1)),
                "flag column {column} carries a value outside {{0, 1}}",
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn shift_columns_match_cpu(seed in any::<u64>(), log_cycles in 1usize..9) {
            let Some(context) = shared_context() else { return Ok(()); };
            let rows = sample_rows(seed, 1usize << log_cycles);
            let packed = pack(&rows);
            let expected = host_columns(&rows);
            let got = upload(context, &packed).expect("device shift columns");
            prop_assert_eq!(got.len(), COLUMNS);
            for (column, expected) in expected.iter().enumerate() {
                prop_assert_eq!(
                    &got[column].to_host().expect("download"),
                    expected,
                    "shift column {} diverged",
                    column
                );
            }
        }
    }
}
