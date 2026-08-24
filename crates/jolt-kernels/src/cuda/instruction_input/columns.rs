use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;
use jolt_riscv::InstructionFlags as InstructionFlagKind;
use jolt_witness::backend::cuda::{instruction_flag_bit, DeviceAtomColumns, DeviceTrace};

pub(crate) struct ColumnShard<F: Field> {
    pub(crate) ordinal: usize,
    pub(crate) columns: DeviceInstructionColumns,
    pub(crate) eq: DeviceSplitEq<F>,
    pub(crate) form: DeviceSumOfProducts,
}

pub(crate) struct ShardedInstructionColumns<F: Field> {
    shards: Vec<ColumnShard<F>>,
    collapsed: Option<DeviceInstructionColumns>,
    local_rounds: usize,
    tail_rounds: usize,
}

impl<F: Field> ShardedInstructionColumns<F> {
    #[cfg(feature = "allocative")]
    pub(crate) fn device_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.columns.device_bytes() + shard.eq.device_bytes())
            .sum::<usize>()
            + self
                .collapsed
                .as_ref()
                .map_or(0, DeviceInstructionColumns::device_bytes)
    }

    pub(crate) fn new(shards: Vec<ColumnShard<F>>, log_t: usize) -> Result<Self, CudaError> {
        let count = shards.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded instruction-input column set needs a power-of-two shard count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        if tail_rounds > log_t {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded instruction-input column set cannot split more windows than \
                         cycle rounds",
            });
        }
        if count == 1 {
            let shard = shards
                .into_iter()
                .next()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a single-shard instruction-input column set lost its state",
                })?;
            return Ok(Self {
                shards: Vec::new(),
                collapsed: Some(shard.columns),
                local_rounds: log_t,
                tail_rounds: 0,
            });
        }
        Ok(Self {
            shards,
            collapsed: None,
            local_rounds: log_t - tail_rounds,
            tail_rounds,
        })
    }

    pub(crate) fn round_endpoints(
        &self,
        form: &DeviceSumOfProducts,
        whole_eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F), CudaError> {
        if let Some(collapsed) = &self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            let half = collapsed.len() / 2;
            return form.round_gruen_endpoints(context, &collapsed.handles(), half, whole_eq);
        }
        let tasks: Vec<DeviceTask<'_, (F, F), CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, (F, F), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    let half = shard.columns.len() / 2;
                    shard.form.round_gruen_endpoints(
                        context,
                        &shard.columns.handles(),
                        half,
                        &shard.eq,
                    )
                });
                task
            })
            .collect();
        let mut total = (F::zero(), F::zero());
        for part in fan_out(tasks)? {
            total.0 += part.0;
            total.1 += part.1;
        }
        Ok(total)
    }

    pub(crate) fn whole(&self) -> Result<&DeviceInstructionColumns, CudaError> {
        self.collapsed.as_ref().ok_or(CudaError::NotImplemented {
            kernel: "the instruction-input eval-point basis is the single-device correctness \
                         reference; it does not drive a windowed column set",
        })
    }

    pub(crate) fn bind(&mut self, challenge: F, bound: usize) -> Result<(), CudaError> {
        if let Some(collapsed) = &mut self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return collapsed.bind(context, challenge);
        }
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
            .shards
            .iter_mut()
            .map(|shard| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.columns.bind(context, challenge)?;
                    shard.eq.bind(challenge);
                    Ok(())
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if bound + 1 == self.local_rounds {
            self.collapse()?;
        }
        Ok(())
    }

    fn collapse(&mut self) -> Result<(), CudaError> {
        let context = context_for(0).ok_or(absent())?;
        let shards = std::mem::take(&mut self.shards);
        let tasks: Vec<DeviceTask<'_, Vec<Fr>, CudaError>> = shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, Vec<Fr>, CudaError> = Box::new(move || {
                    let _ = context_for(shard.ordinal).ok_or(absent())?;
                    shard.columns.window_scalars()
                });
                task
            })
            .collect();
        let mut columns: Vec<Vec<Fr>> = Vec::new();
        for scalars in fan_out(tasks)? {
            if columns.is_empty() {
                columns = scalars.iter().map(|_| Vec::new()).collect();
            }
            if scalars.len() != columns.len() {
                return Err(CudaError::LengthMismatch {
                    expected: columns.len(),
                    got: scalars.len(),
                });
            }
            for (column, value) in columns.iter_mut().zip(&scalars) {
                column.push(*value);
            }
        }
        let expected = 1usize << self.tail_rounds;
        if columns.iter().any(|column| column.len() != expected) {
            return Err(CudaError::LengthMismatch {
                expected,
                got: columns.first().map_or(0, Vec::len),
            });
        }
        self.collapsed = Some(DeviceInstructionColumns::from_dense(context, &columns)?);
        Ok(())
    }

    pub(crate) fn finals<G: Field>(&self) -> Result<Vec<G>, CudaError> {
        self.collapsed
            .as_ref()
            .ok_or(CudaError::InvariantViolation {
                reason: "a sharded instruction-input column set was asked for finals before its \
                         tail rounds",
            })?
            .finals()
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a sharded instruction-input window names an absent device",
    }
}

#[cfg(test)]
use super::witness::Packed;
use super::witness::{self, COLUMNS, LAYOUT, NARROW, SIGN_BIT_BASE, WIDE};
use crate::cuda::common::context::context_for;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::cuda::common::sum_of_products::DeviceSumOfProducts;
use jolt_field::Fr;

pub struct DeviceInstructionColumns {
    columns: Vec<DeviceFrVec>,
}

impl DeviceInstructionColumns {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        self.columns.iter().map(DeviceFrVec::device_bytes).sum()
    }

    pub fn from_device(
        context: &CudaKernelContext,
        trace: &DeviceTrace,
        atoms: &DeviceAtomColumns,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        if cycles < 2 || !cycles.is_power_of_two() || trace.cycles() < cycles {
            return Err(CudaError::InvariantViolation {
                reason: "the device instruction-input sources need a power-of-two cycle count",
            });
        }
        let mut narrow = context.alloc_u64(cycles * NARROW)?;
        let mut wide = context.alloc_u64(cycles * WIDE * 2)?;
        let mut flags = context.alloc_u32(cycles)?;
        let sources = context.upload_u32_slice(&Self::gather_bit_sources()?)?;
        let sign_base = witness::SIGN_BIT_BASE;

        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.ii_gather());
        let _ = builder.arg(trace.extras());
        let _ = builder.arg(trace.unexpanded_pc());
        let _ = builder.arg(&atoms.flags);
        let _ = builder.arg(&sources);
        let _ = builder.arg(&sign_base);
        let _ = builder.arg(&mut narrow);
        let _ = builder.arg(&mut wide);
        let _ = builder.arg(&mut flags);
        let _ = builder.arg(&count);
        // SAFETY: thread `t < cycles` writes the `NARROW` u64s at `t * NARROW`,
        // the `WIDE * 2` u64s at `t * WIDE * 2` and `flags[t]`, all inside
        // allocations sized for `cycles` rows, and reads `address[t]`,
        // `canonical[t]`, the `EXTRA_WORDS` words at `t * EXTRA_WORDS` and the
        // four in-range entries of `sources`. Every buffer is distinct.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Self::split(context, narrow, wide, flags, cycles)
    }

    fn gather_bit_sources() -> Result<Vec<u32>, CudaError> {
        let missing = || CudaError::InvariantViolation {
            reason: "an instruction-input flag has no canonical device bit",
        };
        Ok(vec![
            instruction_flag_bit(InstructionFlagKind::LeftOperandIsRs1Value).ok_or_else(missing)?,
            instruction_flag_bit(InstructionFlagKind::LeftOperandIsPC).ok_or_else(missing)?,
            instruction_flag_bit(InstructionFlagKind::RightOperandIsRs2Value)
                .ok_or_else(missing)?,
            instruction_flag_bit(InstructionFlagKind::RightOperandIsImm).ok_or_else(missing)?,
        ])
    }

    #[cfg(test)]
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
        Self::split(context, narrow, wide, flags, cycles)
    }

    fn split(
        context: &CudaKernelContext,
        narrow: CudaSlice<u64>,
        wide: CudaSlice<u64>,
        flags: CudaSlice<u32>,
        cycles: usize,
    ) -> Result<Self, CudaError> {
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

    #[cfg(test)]
    pub(crate) fn from_dense_for_test(
        context: &CudaKernelContext,
        columns: &[Vec<Fr>],
    ) -> Result<Self, CudaError> {
        Self::from_dense(context, columns)
    }

    fn from_dense(context: &CudaKernelContext, columns: &[Vec<Fr>]) -> Result<Self, CudaError> {
        if columns.len() != COLUMNS {
            return Err(CudaError::LengthMismatch {
                expected: COLUMNS,
                got: columns.len(),
            });
        }
        Ok(Self {
            columns: columns
                .iter()
                .map(|column| context.upload(column))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn window_scalars(&self) -> Result<Vec<Fr>, CudaError> {
        let mut scalars = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            if column.len() != 1 {
                return Err(CudaError::LengthMismatch {
                    expected: 1,
                    got: column.len(),
                });
            }
            scalars.push(column.first()?);
        }
        Ok(scalars)
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
