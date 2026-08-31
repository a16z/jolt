#[cfg(feature = "test-utils")]
use std::{mem::size_of, slice};
use std::{mem::MaybeUninit, sync::Arc};

use metal::{Buffer, MTLResourceOptions};

use super::super::{
    BooleanityRows, InstructionInputRows, InstructionReadRafStage1Owner, MetalError, SolinasMetal,
};
#[cfg(feature = "test-utils")]
use crate::metal::solinas::instruction_input::{
    REGISTER_RD_INDEX_SHIFT, REGISTER_RS1_INDEX_SHIFT, REGISTER_RS2_INDEX_SHIFT,
};
#[cfg(feature = "test-utils")]
use crate::optimized::registers_read_write::PackedRegisterCycleRow;

pub(crate) const REGISTERS_READ_WRITE_STAGE1_CHUNK_ROWS: usize = 1 << 12;

pub(crate) struct RegistersReadWriteStage1Storage {
    rows: usize,
    device_registry_id: u64,
    rd_indices: Buffer,
    active_masks: Box<[u128]>,
}

pub(crate) struct RegistersReadWriteStage1ChunkWriter<'a> {
    active_mask: &'a mut u128,
    rd_indices: &'a mut [MaybeUninit<u8>],
    len: usize,
    written: usize,
}

struct RegistersReadWriteStage1SourceInner {
    instruction_input: InstructionInputRows,
    instruction_read_raf: BooleanityRows,
    rd_indices: Buffer,
    register_map: [u8; 128],
    register_unmap: [u8; 64],
    active_registers: u8,
    remap_registers: bool,
    physical_rows: usize,
    cycles: usize,
    device_registry_id: u64,
}

#[derive(Clone)]
pub(crate) struct RegistersReadWriteStage1Source(Arc<RegistersReadWriteStage1SourceInner>);

pub(crate) struct RegistersReadWriteStage1SourceView<'a> {
    pub(crate) instruction_input: &'a Buffer,
    pub(crate) instruction_read_raf: &'a Buffer,
    pub(crate) rd_indices: &'a Buffer,
    pub(crate) register_map: [u8; 128],
    pub(crate) register_unmap: [u8; 64],
    pub(crate) active_registers: usize,
    pub(crate) remap_registers: bool,
    pub(crate) physical_rows: usize,
    pub(crate) cycles: usize,
    pub(crate) device_registry_id: u64,
}

impl SolinasMetal {
    pub(crate) fn prepare_registers_read_write_stage1_storage(
        &self,
        rows: usize,
    ) -> Result<RegistersReadWriteStage1Storage, MetalError> {
        if rows == 0 {
            return Err(invalid_source("Stage-1 register source cannot be empty"));
        }
        let chunks = rows.div_ceil(REGISTERS_READ_WRITE_STAGE1_CHUNK_ROWS);
        let bytes = u64::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        Ok(RegistersReadWriteStage1Storage {
            rows,
            device_registry_id: self.device_registry_id(),
            rd_indices: self
                .device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            active_masks: vec![0; chunks].into_boxed_slice(),
        })
    }
}

impl RegistersReadWriteStage1Storage {
    pub(crate) fn with_chunk_writers<R>(
        &mut self,
        fill: impl FnOnce(&mut [RegistersReadWriteStage1ChunkWriter<'_>]) -> Result<R, MetalError>,
    ) -> Result<R, MetalError> {
        let rows = self.rows;
        // SAFETY: the unpublished buffer is exclusively borrowed and contains
        // exactly one destination-index byte per row.
        let rd_indices = unsafe {
            std::slice::from_raw_parts_mut(
                self.rd_indices.contents().cast::<MaybeUninit<u8>>(),
                rows,
            )
        };
        let mut chunks = self
            .active_masks
            .iter_mut()
            .zip(rd_indices.chunks_mut(REGISTERS_READ_WRITE_STAGE1_CHUNK_ROWS))
            .map(
                |(active_mask, rd_indices)| RegistersReadWriteStage1ChunkWriter {
                    active_mask,
                    len: rd_indices.len(),
                    rd_indices,
                    written: 0,
                },
            )
            .collect::<Vec<_>>();
        let output = fill(&mut chunks)?;
        if chunks.iter().any(|chunk| chunk.written != chunk.len) {
            return Err(invalid_source(
                "Stage-1 register source did not initialize every row",
            ));
        }
        Ok(output)
    }

    pub(crate) fn seal(
        self,
        instruction_input: InstructionInputRows,
        instruction_read_raf: &InstructionReadRafStage1Owner,
        physical_rows: usize,
    ) -> Result<RegistersReadWriteStage1Source, MetalError> {
        if instruction_input.len() != self.rows
            || instruction_input.device_registry_id() != self.device_registry_id
            || self.rd_indices.length() as usize != self.rows
            || self.rd_indices.device().registry_id() != self.device_registry_id
            || physical_rows == 0
            || physical_rows > self.rows
        {
            return Err(invalid_source(
                "Stage-1 register source dimensions disagree",
            ));
        }
        let receipt = instruction_read_raf.receipt();
        if receipt.rows() != self.rows || receipt.device_registry_id() != self.device_registry_id {
            return Err(invalid_source(
                "Stage-1 register source owners use different devices or domains",
            ));
        }
        let active_mask = self
            .active_masks
            .iter()
            .copied()
            .fold(0u128, |left, right| left | right);
        let active_registers = active_mask.count_ones() as usize;
        if active_registers > 64 {
            return Err(invalid_source(
                "Stage-1 register source has more than 64 active registers",
            ));
        }
        let remap_registers = active_mask >> 64 != 0;
        let mut register_map = [0u8; 128];
        let mut register_unmap = [0u8; 64];
        if remap_registers {
            let mut dense = 0usize;
            for (original, mapped) in register_map.iter_mut().enumerate() {
                if active_mask & (1u128 << original) != 0 {
                    *mapped = dense as u8;
                    register_unmap[dense] = original as u8;
                    dense += 1;
                }
            }
        } else {
            for (index, mapped) in register_map.iter_mut().enumerate() {
                *mapped = index as u8;
            }
            for (index, original) in register_unmap.iter_mut().enumerate() {
                *original = index as u8;
            }
        }
        Ok(RegistersReadWriteStage1Source(Arc::new(
            RegistersReadWriteStage1SourceInner {
                instruction_input,
                instruction_read_raf: instruction_read_raf.booleanity_rows(),
                rd_indices: self.rd_indices,
                register_map,
                register_unmap,
                active_registers: active_registers as u8,
                remap_registers,
                physical_rows,
                cycles: self.rows,
                device_registry_id: self.device_registry_id,
            },
        )))
    }
}

impl RegistersReadWriteStage1ChunkWriter<'_> {
    pub(crate) fn push(
        &mut self,
        register_indices: [Option<u8>; 2],
        register_write: Option<(u8, u64, u64)>,
    ) -> Result<(), MetalError> {
        if self.written == self.len {
            return Err(invalid_source(
                "Stage-1 register source chunk received too many rows",
            ));
        }
        let [rs1, rs2] = register_indices;
        for index in [rs1, rs2, register_write.map(|(index, _, _)| index)]
            .into_iter()
            .flatten()
        {
            if index >= 128 {
                return Err(invalid_source(
                    "Stage-1 register index exceeds the register domain",
                ));
            }
            *self.active_mask |= 1u128 << index;
        }
        let _ = self.rd_indices[self.written]
            .write(register_write.map_or(u8::MAX, |(index, _, _)| index));
        self.written += 1;
        Ok(())
    }

    pub(crate) fn fill_empty(&mut self, count: usize) -> Result<(), MetalError> {
        let end = self
            .written
            .checked_add(count)
            .filter(|&end| end <= self.len)
            .ok_or_else(|| {
                invalid_source("Stage-1 register source chunk received too many padding rows")
            })?;
        for index in &mut self.rd_indices[self.written..end] {
            let _ = index.write(u8::MAX);
        }
        self.written = end;
        Ok(())
    }
}

impl RegistersReadWriteStage1Source {
    pub(crate) fn device_view(&self) -> RegistersReadWriteStage1SourceView<'_> {
        RegistersReadWriteStage1SourceView {
            instruction_input: self.0.instruction_input.buffer(),
            instruction_read_raf: self.0.instruction_read_raf.buffer(),
            rd_indices: &self.0.rd_indices,
            register_map: self.0.register_map,
            register_unmap: self.0.register_unmap,
            active_registers: usize::from(self.0.active_registers),
            remap_registers: self.0.remap_registers,
            physical_rows: self.0.physical_rows,
            cycles: self.0.cycles,
            device_registry_id: self.0.device_registry_id,
        }
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn device_sidecar_bytes(&self) -> usize {
        self.0.rd_indices.length() as usize
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn decode_row(
        &self,
        row: usize,
        rd_post_value: u64,
    ) -> Option<PackedRegisterCycleRow> {
        if row >= self.0.physical_rows {
            return None;
        }
        let view = self.device_view();
        // SAFETY: the owners validate these allocation lengths when sealing.
        let instruction = unsafe {
            slice::from_raw_parts(
                view.instruction_input.contents().cast::<u64>(),
                6 * view.cycles,
            )
        };
        // SAFETY: the instruction Read-RAF owner contains four u64 columns.
        let instruction_read_raf = unsafe {
            slice::from_raw_parts(
                view.instruction_read_raf.contents().cast::<u64>(),
                4 * view.cycles,
            )
        };
        let flags = instruction[6 * row + 5];
        let decode = |shift| {
            let plus_one = ((flags >> shift) & 0xff) as u8;
            plus_one.checked_sub(1)
        };
        let rs1 = decode(REGISTER_RS1_INDEX_SHIFT);
        let rs2 = decode(REGISTER_RS2_INDEX_SHIFT);
        let metadata = instruction_read_raf[3 * view.cycles + row];
        // SAFETY: sealing validates the one-byte-per-cycle sidecar length.
        let rd_indices = unsafe {
            slice::from_raw_parts(
                view.rd_indices.contents().cast::<u8>(),
                view.rd_indices.length() as usize / size_of::<u8>(),
            )
        };
        let rd = (rd_indices[row] != u8::MAX).then_some(rd_indices[row]);
        let encoded_rd = (((flags >> REGISTER_RD_INDEX_SHIFT) & 0xff) as u8).checked_sub(1);
        if rd != encoded_rd {
            return None;
        }
        let magnitude = instruction_read_raf[2 * view.cycles + row];
        let rd_pre_value = if ((metadata >> 62) & 1) != 0 {
            rd_post_value.wrapping_add(magnitude)
        } else {
            rd_post_value.wrapping_sub(magnitude)
        };
        Some(PackedRegisterCycleRow::from_parts(
            instruction[6 * row],
            instruction[6 * row + 2],
            rd_pre_value,
            rd_post_value,
            rs1,
            rs2,
            rd,
        ))
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersReadWriteStage1Source {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(mut shared) = visitor.enter_shared(
            allocative::Key::new("owner"),
            std::mem::size_of::<*const RegistersReadWriteStage1SourceInner>(),
            Arc::as_ptr(&self.0).cast(),
        ) {
            shared.exit();
        }
        visitor.exit();
    }
}

fn invalid_source(reason: &'static str) -> MetalError {
    MetalError::InvalidRegistersReadWriteState(reason)
}
