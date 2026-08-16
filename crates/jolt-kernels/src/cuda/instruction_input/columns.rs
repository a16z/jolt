use cudarc::driver::PushKernelArg;
use jolt_field::Field;

use super::witness::{Packed, COLUMNS, LAYOUT, NARROW, SIGN_BIT_BASE, WIDE};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec};
use crate::cuda::common::error::CudaError;

pub struct DeviceInstructionColumns {
    columns: Vec<DeviceFrVec>,
}

impl DeviceInstructionColumns {
    pub fn new(context: &CudaKernelContext, packed: &Packed) -> Result<Self, CudaError> {
        let cycles = packed.flags.len();
        if cycles < 2
            || !cycles.is_power_of_two()
            || packed.narrow.len() != cycles * NARROW
            || packed.wide.len() != cycles * WIDE * 2
        {
            return Err(CudaError::InvariantViolation {
                reason: "the packed instruction-input columns must hold one entry per cycle per \
                         column over a power-of-two cycle count",
            });
        }

        let narrow = context.upload_u64_slice(&packed.narrow)?;
        let wide = context.upload_u64_slice(&packed.wide)?;
        let flags = context.upload_u32_slice(&packed.flags)?;

        let count = CudaKernelContext::count_of(cycles)?;
        let narrow_width = CudaKernelContext::count_of(NARROW)?;
        let wide_width = CudaKernelContext::count_of(WIDE)?;
        let mut columns = Vec::with_capacity(COLUMNS);
        for term in LAYOUT {
            let kind = (term >> 8) & 3;
            let slot = term & 0xFF;
            let mut column = context.alloc(cycles)?;
            let mut builder = context.stream().launch_builder(context.ii_columns());
            let _ = builder.arg(&narrow);
            let _ = builder.arg(&wide);
            let _ = builder.arg(&flags);
            let _ = builder.arg(&narrow_width);
            let _ = builder.arg(&wide_width);
            let _ = builder.arg(&kind);
            let _ = builder.arg(&slot);
            let _ = builder.arg(&SIGN_BIT_BASE);
            let _ = builder.arg(&count);
            let _ = builder.arg(column.limbs_mut());
            // SAFETY: thread `t < cycles` reads `flags[t]` of `cycles` entries and,
            // by kind, either nothing else, or `narrow[t * NARROW + slot]` inside
            // `narrow`'s `cycles * NARROW` u64s, or `wide[t * WIDE * 2 + 2 * slot
            // (+ 1)]` inside `wide`'s `cycles * WIDE * 2` u64s — every `LAYOUT`
            // slot is below its kind's width by construction. `SIGN_BIT_BASE + slot`
            // is 24 for the single wide slot, inside a u32 mask. It writes only
            // `out[t]` of the freshly allocated `cycles`-element column, distinct
            // from all three inputs, so no thread reads what another writes.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
            columns.push(column);
        }

        Ok(Self { columns })
    }

    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, DeviceFrVec::len)
    }

    pub fn handles(&self) -> Vec<&DeviceFrVec> {
        self.columns.iter().collect()
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let scalar = require_fr(challenge)?;
        for column in &mut self.columns {
            let len = column.len();
            if len < 2 {
                return Err(CudaError::LengthMismatch {
                    expected: 2,
                    got: len,
                });
            }
            *column = context.bind_rows(column, len, scalar)?;
        }
        Ok(())
    }

    pub fn finals<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        let mut finals = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            if column.len() != 1 {
                return Err(CudaError::LengthMismatch {
                    expected: 1,
                    got: column.len(),
                });
            }
            finals.push(fr_into(column.first()?).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })?);
        }
        Ok(finals)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::{
        imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
        rs1_value, rs2_value, unexpanded_pc,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};
    use jolt_witness::{collect_bundles, JoltWitnessPlane};

    use super::super::witness::{self, InstructionInputWitness, COLUMNS};
    use super::DeviceInstructionColumns;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::reference::views::dense_view;

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    fn column_ids() -> [JoltOpeningId; COLUMNS] {
        [
            left_operand_is_rs1(),
            rs1_value(),
            left_operand_is_pc(),
            unexpanded_pc(),
            right_operand_is_rs2(),
            rs2_value(),
            right_operand_is_imm(),
            imm(),
        ]
    }

    fn wide_imm_rows() -> Vec<InstructionInputWitness> {
        let immediates: [i128; 8] = [
            0,
            1,
            -1,
            i128::from(u64::MAX),
            -i128::from(u64::MAX),
            (1i128 << 96) + 12_345,
            -((1i128 << 96) + 12_345),
            (1i128 << 64) - 1,
        ];
        immediates
            .into_iter()
            .enumerate()
            .map(|(index, value)| InstructionInputWitness {
                imm: Imm(value),
                rs1_value: Rs1Value(index as u64),
                rs2_value: Rs2Value(7 * index as u64),
                unexpanded_pc: UnexpandedPc(index as u64),
                right_operand_is_imm: InstructionFlag(true),
                ..InstructionInputWitness::default()
            })
            .collect()
    }

    #[test]
    fn synthetic_rows_exercise_the_high_immediate_word() {
        let rows = wide_imm_rows();
        let packed = witness::pack(&rows);
        assert!(
            packed
                .wide
                .chunks_exact(witness::WIDE * 2)
                .any(|limbs| limbs[2 * witness::IMM_SLOT + 1] != 0),
            "no synthetic row populates the immediate's high word, so the wide read's high limb \
             stays untested",
        );
        assert!(
            rows.iter().any(|row| row.imm.0 < 0),
            "no synthetic row carries a negative immediate",
        );
        assert!(
            rows.iter().any(|row| row.imm.0 > 0),
            "no synthetic row carries a positive immediate",
        );
    }

    #[test]
    fn promoted_immediates_match_the_host_conversion_beyond_64_bits() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = wide_imm_rows();
        let packed = witness::pack(&rows);
        let columns = DeviceInstructionColumns::new(context, &packed).expect("device columns");
        let handles = columns.handles();
        let got = handles[witness::IMM_COLUMN]
            .to_host()
            .expect("download the immediate column");
        let expected: Vec<Fr> = rows.iter().map(|row| Fr::from_i128(row.imm.0)).collect();
        assert_eq!(got, expected, "the promoted immediate column diverged");
    }

    #[test]
    fn promoted_columns_match_the_oracle_tables() {
        let Some(context) = shared_context() else {
            return;
        };
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 11, |witness| {
            let plane: &dyn JoltWitnessPlane<Fr> = witness;
            let rows = collect_bundles::<InstructionInputWitness>(plane, 1usize << LOG_T)
                .expect("the fixture serves the instruction-input bundle");
            let packed = witness::pack(&rows);
            let columns = DeviceInstructionColumns::new(context, &packed).expect("device columns");
            let handles = columns.handles();

            for (id, handle) in column_ids().into_iter().zip(handles) {
                let expected = dense_view::<Fr>(plane, id).expect("the fixture serves the column");
                let got = handle.to_host().expect("download column");
                assert_eq!(got, expected, "promoted column diverged for {id:?}");
            }
        });
    }
}
