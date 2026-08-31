use jolt_witness::{RandomAccessRows, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::rows::RegisterCycleRow;

const COLLECT_CHUNK: usize = 1 << 16;

/// Compact evaluator/accelerator source row. Missing registers use `u8::MAX`;
/// values for missing operands are zero. The explicit layout is also directly
/// consumable by device kernels without staging Rust enum discriminants.
#[cfg(feature = "metal")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PackedRegisterCycleRow {
    pub(crate) rs1_value: u64,
    pub(crate) rs2_value: u64,
    pub(crate) rd_pre_value: u64,
    pub(crate) rd_post_value: u64,
    pub(crate) rs1_index: u8,
    pub(crate) rs2_index: u8,
    pub(crate) rd_index: u8,
    _padding: [u8; 5],
}

/// One direct register cell after the cycle domain is fully bound.
#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BoundRegisterCycleRoot<F> {
    pub(crate) column: u8,
    pub(crate) previous: u64,
    pub(crate) next: u64,
    pub(crate) value: F,
    pub(crate) ra: F,
    pub(crate) wa: F,
}

#[cfg(feature = "metal")]
impl PackedRegisterCycleRow {
    pub(crate) const NO_REGISTER: u8 = u8::MAX;

    #[cfg(feature = "test-utils")]
    pub(crate) fn from_parts(
        rs1_value: u64,
        rs2_value: u64,
        rd_pre_value: u64,
        rd_post_value: u64,
        rs1_index: Option<u8>,
        rs2_index: Option<u8>,
        rd_index: Option<u8>,
    ) -> Self {
        Self {
            rs1_value,
            rs2_value,
            rd_pre_value,
            rd_post_value,
            rs1_index: rs1_index.unwrap_or(Self::NO_REGISTER),
            rs2_index: rs2_index.unwrap_or(Self::NO_REGISTER),
            rd_index: rd_index.unwrap_or(Self::NO_REGISTER),
            _padding: [0; 5],
        }
    }

    pub(crate) fn pack(row: RegisterCycleRow) -> Self {
        Self {
            rs1_value: row.rs1.map_or(0, |(_, value)| value),
            rs2_value: row.rs2.map_or(0, |(_, value)| value),
            rd_pre_value: row.rd.map_or(0, |(_, value, _)| value),
            rd_post_value: row.rd.map_or(0, |(_, _, value)| value),
            rs1_index: row.rs1.map_or(Self::NO_REGISTER, |(index, _)| index),
            rs2_index: row.rs2.map_or(Self::NO_REGISTER, |(index, _)| index),
            rd_index: row.rd.map_or(Self::NO_REGISTER, |(index, ..)| index),
            _padding: [0; 5],
        }
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn unpack(self) -> RegisterCycleRow {
        RegisterCycleRow {
            rs1: (self.rs1_index != Self::NO_REGISTER).then_some((self.rs1_index, self.rs1_value)),
            rs2: (self.rs2_index != Self::NO_REGISTER).then_some((self.rs2_index, self.rs2_value)),
            rd: (self.rd_index != Self::NO_REGISTER).then_some((
                self.rd_index,
                self.rd_pre_value,
                self.rd_post_value,
            )),
        }
    }

    pub(crate) fn rd_index(self) -> Option<u8> {
        (self.rd_index != Self::NO_REGISTER).then_some(self.rd_index)
    }

    fn set_dense_indices(&mut self, register_map: &[u8; 128]) {
        let dense = |index: u8| {
            if index == Self::NO_REGISTER {
                Self::NO_REGISTER
            } else {
                register_map[usize::from(index)]
            }
        };
        self._padding[..3].copy_from_slice(&[
            dense(self.rs1_index),
            dense(self.rs2_index),
            dense(self.rd_index),
        ]);
    }
}

#[cfg(feature = "metal")]
pub(crate) const PACKED_REGISTER_ROWS_ALIGNMENT: usize = 16 * 1024;

#[cfg(feature = "metal")]
#[derive(Debug, thiserror::Error)]
pub(crate) enum AlignedPackedRegisterRowsError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error("{0}")]
    Storage(&'static str),
}

#[cfg(feature = "metal")]
pub(crate) struct AlignedCompactRegisterIndices {
    ptr: core::ptr::NonNull<u8>,
    allocation_bytes: usize,
}

#[cfg(feature = "metal")]
impl AlignedCompactRegisterIndices {
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    pub(crate) const fn allocation_bytes(&self) -> usize {
        self.allocation_bytes
    }
}

#[cfg(feature = "metal")]
impl Drop for AlignedCompactRegisterIndices {
    fn drop(&mut self) {
        // SAFETY: these are the exact size and alignment used by construction.
        let layout = unsafe {
            std::alloc::Layout::from_size_align_unchecked(
                self.allocation_bytes,
                PACKED_REGISTER_ROWS_ALIGNMENT,
            )
        };
        // SAFETY: `ptr` owns the live allocation described by `layout`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

// SAFETY: construction owns the allocation and publishes only immutable access.
#[cfg(feature = "metal")]
unsafe impl Send for AlignedCompactRegisterIndices {}
// SAFETY: construction owns the allocation and publishes only immutable access.
#[cfg(feature = "metal")]
unsafe impl Sync for AlignedCompactRegisterIndices {}

/// Page-aligned packed rows that can back a borrowed shared Metal buffer.
#[cfg(feature = "metal")]
pub(crate) struct AlignedPackedRegisterRows {
    ptr: core::ptr::NonNull<PackedRegisterCycleRow>,
    rs1_indices: Option<std::sync::Arc<AlignedCompactRegisterIndices>>,
    register_unmap: [u8; 64],
    active_registers: u8,
    remap_registers: bool,
    rows: usize,
    row_allocation_bytes: usize,
    total_allocation_bytes: usize,
}

#[cfg(feature = "metal")]
impl AlignedPackedRegisterRows {
    pub(crate) fn collect(
        access: &RandomAccessRows,
        rows: usize,
        include_compact_rs1: bool,
    ) -> Result<Self, AlignedPackedRegisterRowsError> {
        if rows > access.cycles() {
            return Err(AlignedPackedRegisterRowsError::Storage(
                "aligned packed register collection exceeds the cycle domain",
            ));
        }
        let mut owner = Self::allocate(rows, include_compact_rs1)?;
        // SAFETY: the allocation owns `rows` properly aligned slots. They are
        // treated as initialized only after every extraction succeeds.
        let destination = unsafe {
            core::slice::from_raw_parts_mut(
                owner
                    .ptr
                    .as_ptr()
                    .cast::<core::mem::MaybeUninit<PackedRegisterCycleRow>>(),
                rows,
            )
        };
        #[cfg(feature = "parallel")]
        destination
            .par_chunks_mut(COLLECT_CHUNK)
            .enumerate()
            .try_for_each(|(chunk_index, chunk)| -> Result<(), WitnessError> {
                let base = chunk_index * COLLECT_CHUNK;
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    let row = access.window::<RegisterCycleRow>(base + offset)?;
                    let _ = slot.write(PackedRegisterCycleRow::pack(row));
                }
                Ok(())
            })?;
        #[cfg(not(feature = "parallel"))]
        for (index, slot) in destination.iter_mut().enumerate() {
            let row = access.window::<RegisterCycleRow>(index)?;
            let _ = slot.write(PackedRegisterCycleRow::pack(row));
        }
        let logical_bytes = owner.logical_bytes();
        // SAFETY: all row slots are initialized above; only the allocation's
        // sub-page tail remains to be zeroed.
        unsafe {
            owner
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(logical_bytes)
                .write_bytes(0, owner.row_allocation_bytes - logical_bytes);
        }
        owner.finish_layout()?;
        Ok(owner)
    }

    fn allocate(
        rows: usize,
        include_compact_rs1: bool,
    ) -> Result<Self, AlignedPackedRegisterRowsError> {
        if rows == 0 {
            return Err(AlignedPackedRegisterRowsError::Storage(
                "aligned packed register rows cannot be empty",
            ));
        }
        let logical_bytes = rows
            .checked_mul(core::mem::size_of::<PackedRegisterCycleRow>())
            .ok_or(AlignedPackedRegisterRowsError::Storage(
                "aligned packed register row length overflowed",
            ))?;
        let align_bytes = |bytes: usize, overflow| {
            bytes
                .checked_add(PACKED_REGISTER_ROWS_ALIGNMENT - 1)
                .map(|rounded| {
                    rounded / PACKED_REGISTER_ROWS_ALIGNMENT * PACKED_REGISTER_ROWS_ALIGNMENT
                })
                .ok_or(AlignedPackedRegisterRowsError::Storage(overflow))
        };
        let row_allocation_bytes = align_bytes(
            logical_bytes,
            "aligned packed register allocation length overflowed",
        )?;
        let rs1_indices_allocation_bytes = if include_compact_rs1 {
            align_bytes(rows, "aligned register index allocation length overflowed")?
        } else {
            0
        };
        let total_allocation_bytes = row_allocation_bytes
            .checked_add(rs1_indices_allocation_bytes)
            .ok_or(AlignedPackedRegisterRowsError::Storage(
                "aligned packed register allocation length overflowed",
            ))?;
        let rs1_indices = if include_compact_rs1 {
            let layout = std::alloc::Layout::from_size_align(
                rs1_indices_allocation_bytes,
                PACKED_REGISTER_ROWS_ALIGNMENT,
            )
            .map_err(|_| {
                AlignedPackedRegisterRowsError::Storage(
                    "aligned register index allocation layout is invalid",
                )
            })?;
            // SAFETY: `layout` has nonzero size and valid alignment.
            let raw = unsafe { std::alloc::alloc(layout) };
            let ptr = core::ptr::NonNull::new(raw).ok_or(
                AlignedPackedRegisterRowsError::Storage("aligned register index allocation failed"),
            )?;
            Some(std::sync::Arc::new(AlignedCompactRegisterIndices {
                ptr,
                allocation_bytes: rs1_indices_allocation_bytes,
            }))
        } else {
            None
        };
        let layout = std::alloc::Layout::from_size_align(
            row_allocation_bytes,
            PACKED_REGISTER_ROWS_ALIGNMENT,
        )
        .map_err(|_| {
            AlignedPackedRegisterRowsError::Storage(
                "aligned packed register allocation layout is invalid",
            )
        })?;
        // SAFETY: `layout` has nonzero size and valid alignment.
        let raw = unsafe { std::alloc::alloc(layout) };
        let ptr = core::ptr::NonNull::new(raw)
            .ok_or(AlignedPackedRegisterRowsError::Storage(
                "aligned packed register allocation failed",
            ))?
            .cast::<PackedRegisterCycleRow>();
        Ok(Self {
            ptr,
            rs1_indices,
            register_unmap: [0; 64],
            active_registers: 0,
            remap_registers: false,
            rows,
            row_allocation_bytes,
            total_allocation_bytes,
        })
    }

    fn finish_layout(&mut self) -> Result<(), AlignedPackedRegisterRowsError> {
        // SAFETY: both constructors initialize every row before calling this
        // method, and retain exclusive access to the allocation.
        let rows = unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.rows) };
        #[cfg(feature = "parallel")]
        let (active_register_mask, invalid_register) = rows
            .par_iter()
            .map(|row| {
                let mut mask = 0u128;
                let mut invalid = false;
                for index in [row.rs1_index, row.rs2_index, row.rd_index] {
                    if index != PackedRegisterCycleRow::NO_REGISTER {
                        if index < 128 {
                            mask |= 1u128 << index;
                        } else {
                            invalid = true;
                        }
                    }
                }
                (mask, invalid)
            })
            .reduce(
                || (0, false),
                |left, right| (left.0 | right.0, left.1 || right.1),
            );
        #[cfg(not(feature = "parallel"))]
        let (active_register_mask, invalid_register) =
            rows.iter()
                .fold((0u128, false), |(mut mask, mut invalid), row| {
                    for index in [row.rs1_index, row.rs2_index, row.rd_index] {
                        if index != PackedRegisterCycleRow::NO_REGISTER {
                            if index < 128 {
                                mask |= 1u128 << index;
                            } else {
                                invalid = true;
                            }
                        }
                    }
                    (mask, invalid)
                });
        if invalid_register {
            return Err(AlignedPackedRegisterRowsError::Storage(
                "packed register row index exceeds the register domain",
            ));
        }
        let active_registers = active_register_mask.count_ones() as usize;
        if active_registers > 64 {
            return Err(AlignedPackedRegisterRowsError::Storage(
                "Metal sparse register state supports at most 64 active registers",
            ));
        }
        let remap_registers = active_register_mask >> 64 != 0;
        let mut register_map = [0u8; 128];
        let mut register_unmap = [0u8; 64];
        if remap_registers {
            let mut dense = 0usize;
            for (original, mapped) in register_map.iter_mut().enumerate() {
                if active_register_mask & (1u128 << original) != 0 {
                    *mapped = dense as u8;
                    register_unmap[dense] = original as u8;
                    dense += 1;
                }
            }
            #[cfg(feature = "parallel")]
            rows.par_iter_mut()
                .for_each(|row| row.set_dense_indices(&register_map));
            #[cfg(not(feature = "parallel"))]
            rows.iter_mut()
                .for_each(|row| row.set_dense_indices(&register_map));
        } else {
            for (index, mapped) in register_map.iter_mut().enumerate() {
                *mapped = index as u8;
            }
            for (index, original) in register_unmap.iter_mut().enumerate() {
                *original = index as u8;
            }
        }
        if let Some(indices_owner) = self.rs1_indices.as_ref() {
            let indices_ptr = indices_owner.ptr;
            // SAFETY: the compact sidecar is disjoint from the row region and
            // has at least `self.rows` bytes.
            let indices =
                unsafe { core::slice::from_raw_parts_mut(indices_ptr.as_ptr(), self.rows) };
            #[cfg(feature = "parallel")]
            indices
                .par_iter_mut()
                .zip(rows.par_iter())
                .for_each(|(index, row)| *index = row.rs1_index);
            #[cfg(not(feature = "parallel"))]
            for (index, row) in indices.iter_mut().zip(rows.iter()) {
                *index = row.rs1_index;
            }
            let sidecar_bytes = indices_owner.allocation_bytes;
            // SAFETY: the initialized prefix has `self.rows` bytes and the
            // remainder is the sidecar's alignment padding.
            unsafe {
                indices_ptr
                    .as_ptr()
                    .add(self.rows)
                    .write_bytes(0, sidecar_bytes - self.rows);
            }
        }
        self.register_unmap = register_unmap;
        self.active_registers = active_registers as u8;
        self.remap_registers = remap_registers;
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> &[PackedRegisterCycleRow] {
        // SAFETY: construction initialized all `rows` elements, and the
        // allocation remains owned for this borrow.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.rows) }
    }

    pub(crate) fn device_view(&self) -> PackedRegisterRowsDeviceView<'_> {
        PackedRegisterRowsDeviceView {
            ptr: self.ptr,
            rs1_indices_ptr: self.rs1_indices.as_ref().map(|owner| owner.ptr),
            register_unmap: self.register_unmap,
            active_registers: self.active_registers,
            remap_registers: self.remap_registers,
            rows: self.rows,
            row_allocation_bytes: self.row_allocation_bytes,
            rs1_indices_allocation_bytes: self.total_allocation_bytes - self.row_allocation_bytes,
            marker: core::marker::PhantomData,
        }
    }

    pub(crate) fn compact_rs1_owner(
        &self,
    ) -> Option<std::sync::Arc<AlignedCompactRegisterIndices>> {
        self.rs1_indices.as_ref().map(std::sync::Arc::clone)
    }

    pub(crate) fn logical_bytes(&self) -> usize {
        self.rows * core::mem::size_of::<PackedRegisterCycleRow>()
    }

    pub(crate) const fn allocation_bytes(&self) -> usize {
        self.total_allocation_bytes
    }
}

#[cfg(feature = "metal")]
impl Drop for AlignedPackedRegisterRows {
    fn drop(&mut self) {
        // SAFETY: these are the exact size and alignment used by construction.
        let layout = unsafe {
            std::alloc::Layout::from_size_align_unchecked(
                self.row_allocation_bytes,
                PACKED_REGISTER_ROWS_ALIGNMENT,
            )
        };
        // SAFETY: `ptr` owns the live allocation described by `layout`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
    }
}

// SAFETY: the owner exposes only immutable row slices after construction.
#[cfg(feature = "metal")]
unsafe impl Send for AlignedPackedRegisterRows {}
// SAFETY: the owner exposes only immutable row slices after construction.
#[cfg(feature = "metal")]
unsafe impl Sync for AlignedPackedRegisterRows {}

#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
pub(crate) struct PackedRegisterRowsDeviceView<'a> {
    ptr: core::ptr::NonNull<PackedRegisterCycleRow>,
    rs1_indices_ptr: Option<core::ptr::NonNull<u8>>,
    register_unmap: [u8; 64],
    active_registers: u8,
    remap_registers: bool,
    rows: usize,
    row_allocation_bytes: usize,
    rs1_indices_allocation_bytes: usize,
    marker: core::marker::PhantomData<&'a [PackedRegisterCycleRow]>,
}

#[cfg(feature = "metal")]
impl PackedRegisterRowsDeviceView<'_> {
    pub(crate) fn as_ptr(self) -> *const PackedRegisterCycleRow {
        self.ptr.as_ptr()
    }

    pub(crate) const fn rows(self) -> usize {
        self.rows
    }

    pub(crate) const fn allocation_bytes(self) -> usize {
        self.row_allocation_bytes
    }

    pub(crate) fn compact_rs1_source(self) -> Option<(*const u8, usize)> {
        self.rs1_indices_ptr
            .map(|ptr| (ptr.as_ptr().cast_const(), self.rs1_indices_allocation_bytes))
    }

    pub(crate) const fn register_unmap(self) -> [u8; 64] {
        self.register_unmap
    }

    pub(crate) const fn active_registers(self) -> usize {
        self.active_registers as usize
    }

    pub(crate) const fn remaps_registers(self) -> bool {
        self.remap_registers
    }
}
