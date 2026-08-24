use std::sync::Arc;

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;
use jolt_riscv::InstructionFlags as InstructionFlagKind;
use jolt_witness::backend::cuda::{
    instruction_flag_bit, DeviceAtomColumns, DeviceTrace, EXTRA_IMM_HI, EXTRA_IMM_LO, EXTRA_RS1,
    EXTRA_RS2, EXTRA_WORDS,
};

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
            return form.round_gruen_endpoints(context, &collapsed.columns()?, half, whole_eq);
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
                        &shard.columns.columns()?,
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
                    shard.columns.scalars()
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

use super::witness::{
    COLUMNS, IMM_COLUMN, LEFT_IS_PC_COLUMN, LEFT_IS_RS1_COLUMN, RIGHT_IS_IMM_COLUMN,
    RIGHT_IS_RS2_COLUMN, RS1_VALUE_COLUMN, RS2_VALUE_COLUMN, UNEXPANDED_PC_COLUMN,
};
use crate::cuda::common::context::context_for;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::half_fold::{bind_column, FoldColumn, NarrowColumn, NarrowKind};
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::cuda::common::sum_of_products::DeviceSumOfProducts;
use jolt_field::Fr;

const IMM_WORDS_ARE_ADJACENT: () = assert!(EXTRA_IMM_HI == EXTRA_IMM_LO + 1);

const FLAG_KINDS: [InstructionFlagKind; 4] = [
    InstructionFlagKind::LeftOperandIsRs1Value,
    InstructionFlagKind::LeftOperandIsPC,
    InstructionFlagKind::RightOperandIsRs2Value,
    InstructionFlagKind::RightOperandIsImm,
];

pub struct NativeInstructionColumns {
    trace: Arc<DeviceTrace>,
    flags: CudaSlice<u64>,
    bits: [u32; 4],
    entries: usize,
}

impl NativeInstructionColumns {
    fn extra(&self, word: usize, kind: NarrowKind) -> FoldColumn<'_> {
        FoldColumn::Narrow(NarrowColumn {
            words: self.trace.extras(),
            kind,
            len: self.entries,
            stride: EXTRA_WORDS,
            offset: word,
        })
    }

    fn flag(&self, slot: usize) -> Option<FoldColumn<'_>> {
        let bit = *self.bits.get(slot)?;
        Some(FoldColumn::Narrow(NarrowColumn::packed(
            &self.flags,
            NarrowKind::Bit(bit),
            self.entries,
        )))
    }

    fn column(&self, index: usize) -> Option<FoldColumn<'_>> {
        match index {
            LEFT_IS_RS1_COLUMN => self.flag(0),
            LEFT_IS_PC_COLUMN => self.flag(1),
            RIGHT_IS_RS2_COLUMN => self.flag(2),
            RIGHT_IS_IMM_COLUMN => self.flag(3),
            RS1_VALUE_COLUMN => Some(self.extra(EXTRA_RS1, NarrowKind::U64)),
            RS2_VALUE_COLUMN => Some(self.extra(EXTRA_RS2, NarrowKind::U64)),
            IMM_COLUMN => Some(self.extra(EXTRA_IMM_LO, NarrowKind::TwosI128)),
            UNEXPANDED_PC_COLUMN => Some(FoldColumn::Narrow(NarrowColumn::packed(
                self.trace.unexpanded_pc(),
                NarrowKind::U64,
                self.entries,
            ))),
            _ => None,
        }
    }
}

pub enum DeviceInstructionColumns {
    Native(NativeInstructionColumns),
    Bound(Vec<DeviceFrVec>),
}

impl DeviceInstructionColumns {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        match self {
            Self::Native(native) => native.flags.len() * size_of::<u64>(),
            Self::Bound(columns) => columns.iter().map(DeviceFrVec::device_bytes).sum(),
        }
    }

    pub fn from_device(
        context: &CudaKernelContext,
        trace: Arc<DeviceTrace>,
        atoms: &DeviceAtomColumns,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        let () = IMM_WORDS_ARE_ADJACENT;
        if cycles < 2
            || !cycles.is_power_of_two()
            || trace.cycles() < cycles
            || atoms.flags.len() < cycles
        {
            return Err(CudaError::InvariantViolation {
                reason: "the device instruction-input sources need a power-of-two cycle count",
            });
        }
        let missing = || CudaError::InvariantViolation {
            reason: "an instruction-input flag has no canonical device bit",
        };
        let mut bits = [0u32; 4];
        for (slot, kind) in FLAG_KINDS.into_iter().enumerate() {
            *bits.get_mut(slot).ok_or_else(missing)? =
                instruction_flag_bit(kind).ok_or_else(missing)?;
        }

        let mut flags = context.alloc_u64(cycles)?;
        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.ii_flag_words());
        let _ = builder.arg(&atoms.flags);
        let _ = builder.arg(&mut flags);
        let _ = builder.arg(&count);
        // SAFETY: thread `t < cycles` reads `canonical[t]`, checked above to hold
        // at least `cycles` entries, and writes `words[t]` of a fresh
        // `cycles`-word allocation distinct from it. Threads with `t >= cycles`
        // return before any access.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self::Native(NativeInstructionColumns {
            trace,
            flags,
            bits,
            entries: cycles,
        }))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Native(native) => native.entries,
            Self::Bound(columns) => columns.first().map_or(0, DeviceFrVec::len),
        }
    }

    pub fn columns(&self) -> Result<Vec<FoldColumn<'_>>, CudaError> {
        let absent = || CudaError::InvariantViolation {
            reason: "an instruction-input column set lost a column",
        };
        match self {
            Self::Native(native) => (0..COLUMNS)
                .map(|index| native.column(index).ok_or_else(absent))
                .collect(),
            Self::Bound(columns) => (0..COLUMNS)
                .map(|index| columns.get(index).map(FoldColumn::Field).ok_or_else(absent))
                .collect(),
        }
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        if self.len() < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        if let Self::Native(_) = self {
            let bound = self
                .columns()?
                .into_iter()
                .map(|column| bind_column(context, column, challenge))
                .collect::<Result<Vec<_>, _>>()?;
            *self = Self::Bound(bound);
            return Ok(());
        }
        let Self::Bound(columns) = self else {
            return Err(CudaError::InvariantViolation {
                reason: "an instruction-input column set changed state mid-bind",
            });
        };
        let scalar = require_fr(challenge)?;
        for column in columns.iter_mut() {
            let len = column.len();
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
        Ok(Self::Bound(
            columns
                .iter()
                .map(|column| context.upload(column))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    fn scalars(&self) -> Result<Vec<Fr>, CudaError> {
        let Self::Bound(columns) = self else {
            return Err(CudaError::InvariantViolation {
                reason: "an instruction-input column set was read back before its first bind",
            });
        };
        let mut scalars = Vec::with_capacity(columns.len());
        for column in columns {
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
        self.scalars()?
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
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
    use jolt_field::Fr;
    use jolt_witness::JoltWitnessPlane;

    use super::super::witness::COLUMNS;
    use super::DeviceInstructionColumns;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::half_fold::promote_for_test;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::reference::views::dense_view;
    use crate::ProofSession;

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

    #[test]
    fn native_columns_match_the_oracle_tables() {
        let Some(context) = shared_context() else {
            return;
        };
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 11, |witness| {
            let plane: &dyn JoltWitnessPlane<Fr> = witness;
            let cycles = 1usize << LOG_T;
            let mut session = ProofSession::default();
            let trace = crate::cuda::witness::session_device_trace::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
            )
            .expect("device residency");
            let atoms = crate::cuda::witness::session_atom_columns::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
            )
            .expect("atom columns");
            let columns = DeviceInstructionColumns::from_device(context, trace, &atoms, cycles)
                .expect("native columns");

            for (id, column) in column_ids()
                .into_iter()
                .zip(columns.columns().expect("columns"))
            {
                let expected = dense_view::<Fr>(plane, id).expect("the fixture serves the column");
                let got = promote_for_test(context, column).expect("promote the native column");
                assert_eq!(got, expected, "native column diverged for {id:?}");
            }
        });
    }
}
