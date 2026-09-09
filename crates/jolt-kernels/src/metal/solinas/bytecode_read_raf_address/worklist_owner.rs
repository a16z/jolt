use std::{
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

use metal::{foreign_types::ForeignType, Buffer};

use super::{
    carrier::AddressMajorShape,
    stage1_topology::{BytecodeAddressStage1TopologyLease, BytecodeAddressStage1TopologyReceipt},
};
use crate::metal::solinas::MetalError;

static NEXT_COMPLETION_SERIAL: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressSparseStage1Receipt {
    shape: AddressMajorShape,
    physical_rows: usize,
    work_items: usize,
    first_push_pc: usize,
    device_registry_id: u64,
    source_generation: u64,
    source_completion_serial: u64,
    source_rows_storage_id: usize,
    source_claim_storage_id: usize,
    source_windows: usize,
    completion_serial: u64,
    occurrence_storage_id: usize,
    occurrence_bytes: usize,
    magnitude_storage_id: usize,
    magnitude_bytes: usize,
    work_item_storage_id: usize,
    work_item_bytes: usize,
    address_offset_storage_id: usize,
    address_offset_bytes: usize,
    complete_overwrite: bool,
    covered_rows: usize,
    additional_source_scans: usize,
    member_upload_bytes: usize,
}

impl BytecodeAddressSparseStage1Receipt {
    copy_field_getters! { pub(crate), {
        shape: AddressMajorShape,
        physical_rows: usize,
        work_items: usize,
        first_push_pc: usize,
        device_registry_id: u64,
        source_generation: u64,
        source_completion_serial: u64,
        source_rows_storage_id: usize,
        source_claim_storage_id: usize,
        source_windows: usize,
        completion_serial: u64,
        occurrence_storage_id: usize,
        occurrence_bytes: usize,
        magnitude_storage_id: usize,
        magnitude_bytes: usize,
        work_item_storage_id: usize,
        work_item_bytes: usize,
        address_offset_storage_id: usize,
        address_offset_bytes: usize,
        complete_overwrite: bool,
        covered_rows: usize,
        additional_source_scans: usize,
        member_upload_bytes: usize,
    } }
}

pub(crate) struct BytecodeAddressSparseStage1Carrier {
    occurrences: Buffer,
    magnitudes: Buffer,
    work_items: Buffer,
    address_offsets: Buffer,
    receipt: BytecodeAddressSparseStage1Receipt,
    fused_topology_receipt: Option<BytecodeAddressStage1TopologyReceipt>,
}

pub(crate) struct BytecodeAddressSparseStage1Parts {
    pub(crate) occurrences: Buffer,
    pub(crate) magnitudes: Buffer,
    pub(crate) work_items: Buffer,
    pub(crate) address_offsets: Buffer,
    pub(crate) receipt: BytecodeAddressSparseStage1Receipt,
}

pub(crate) struct BytecodeAddressFusedScatterRequest {
    topology: BytecodeAddressStage1TopologyLease,
}

impl BytecodeAddressSparseStage1Carrier {
    copy_field_getters! { pub(crate), { receipt: BytecodeAddressSparseStage1Receipt }}

    pub(crate) const fn fused_topology_receipt(
        &self,
    ) -> Option<BytecodeAddressStage1TopologyReceipt> {
        self.fused_topology_receipt
    }

    pub(crate) fn into_parts(self) -> BytecodeAddressSparseStage1Parts {
        BytecodeAddressSparseStage1Parts {
            occurrences: self.occurrences,
            magnitudes: self.magnitudes,
            work_items: self.work_items,
            address_offsets: self.address_offsets,
            receipt: self.receipt,
        }
    }
}

impl BytecodeAddressFusedScatterRequest {
    pub(crate) fn new(topology: BytecodeAddressStage1TopologyLease) -> Result<Self, MetalError> {
        let receipt = topology.receipt();
        if !receipt.complete_overwrite()
            || receipt.covered_rows() != receipt.physical_rows()
            || receipt.shared_source_row_scans() != 1
            || receipt.additional_source_row_scans() != 0
            || receipt.member_upload_bytes() != 0
        {
            return Err(invalid("fused bytecode topology has incomplete provenance"));
        }
        Ok(Self { topology })
    }

    pub(crate) fn receipt(&self) -> BytecodeAddressStage1TopologyReceipt {
        self.topology.receipt()
    }

    pub(crate) fn source_receipt(&self) -> crate::metal::solinas::InstructionReadRafStage1Receipt {
        self.topology.source().receipt()
    }

    pub(crate) fn descriptors_buffer(&self) -> &Buffer {
        self.topology.descriptors_buffer()
    }

    pub(crate) fn pivots_buffer(&self) -> &Buffer {
        self.topology.pivots_buffer()
    }

    pub(crate) fn chunk_offsets_buffer(&self) -> &Buffer {
        self.topology.chunk_offsets_buffer()
    }

    pub(crate) fn work_items_buffer(&self) -> &Buffer {
        self.topology.work_items_buffer()
    }

    pub(crate) fn address_offsets_buffer(&self) -> &Buffer {
        self.topology.address_offsets_buffer()
    }

    pub(crate) fn publish(
        self,
        source: crate::metal::solinas::InstructionReadRafStage1Receipt,
        occurrences: Buffer,
        magnitudes: Buffer,
    ) -> Result<BytecodeAddressSparseStage1Carrier, MetalError> {
        let topology = self.topology.receipt();
        if source != topology.source_receipt() || source != self.topology.source().receipt() {
            return Err(invalid(
                "fused bytecode topology belongs to another Stage-1 source",
            ));
        }
        let padded_rows = topology.padded_rows();
        let physical_rows = topology.physical_rows();
        let occurrence_bytes = byte_length::<u16>(physical_rows)?;
        let magnitude_bytes = byte_length::<u64>(physical_rows)?;
        let work_items = self.topology.work_items_buffer().clone();
        let address_offsets = self.topology.address_offsets_buffer().clone();
        let work_item_bytes = topology.work_item_bytes();
        let address_offset_bytes = topology.address_offset_bytes();
        let ids = [
            occurrences.as_ptr() as usize,
            magnitudes.as_ptr() as usize,
            work_items.as_ptr() as usize,
            address_offsets.as_ptr() as usize,
        ];
        let topology_ids = [
            topology.descriptor_allocation_identity(),
            topology.pivot_allocation_identity(),
            topology.chunk_offset_allocation_identity(),
        ];
        if padded_rows != source.rows()
            || topology
                .shape()
                .rows()
                .map_err(|error| invalid(error.to_string()))?
                != padded_rows
            || physical_rows == 0
            || physical_rows > padded_rows
            || topology.work_items() == 0
            || topology.address_offset_elements() != (1usize << super::carrier::ADDRESS_LOG2) + 1
            || occurrences.length() != occurrence_bytes as u64
            || magnitudes.length() != magnitude_bytes as u64
            || work_items.length() != work_item_bytes as u64
            || address_offsets.length() != address_offset_bytes as u64
            || occurrences.device().registry_id() != topology.device_registry_id()
            || magnitudes.device().registry_id() != topology.device_registry_id()
            || ids.contains(&0)
            || topology_ids.contains(&0)
            || ids
                .iter()
                .enumerate()
                .any(|(index, id)| ids[..index].contains(id))
            || ids.iter().any(|id| topology_ids.contains(id))
            || ids.contains(&source.row_allocation_identity())
            || ids.contains(&source.claim_allocation_identity())
        {
            return Err(invalid("fused bytecode carrier provenance is invalid"));
        }
        let completion_serial = next_nonzero(&NEXT_COMPLETION_SERIAL)?;
        let receipt = BytecodeAddressSparseStage1Receipt {
            shape: topology.shape(),
            physical_rows,
            work_items: topology.work_items(),
            first_push_pc: topology.first_push_pc(),
            device_registry_id: topology.device_registry_id(),
            source_generation: source.source_generation(),
            source_completion_serial: source.completion_serial(),
            source_rows_storage_id: source.row_allocation_identity(),
            source_claim_storage_id: source.claim_allocation_identity(),
            source_windows: source.rows(),
            completion_serial,
            occurrence_storage_id: ids[0],
            occurrence_bytes,
            magnitude_storage_id: ids[1],
            magnitude_bytes,
            work_item_storage_id: ids[2],
            work_item_bytes,
            address_offset_storage_id: ids[3],
            address_offset_bytes,
            complete_overwrite: true,
            covered_rows: physical_rows,
            additional_source_scans: 0,
            member_upload_bytes: 0,
        };
        Ok(BytecodeAddressSparseStage1Carrier {
            occurrences,
            magnitudes,
            work_items,
            address_offsets,
            receipt,
            fused_topology_receipt: Some(topology),
        })
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for BytecodeAddressSparseStage1Carrier {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_buffers"),
            self.receipt.occurrence_bytes
                + self.receipt.magnitude_bytes
                + self.receipt.work_item_bytes
                + self.receipt.address_offset_bytes,
        );
        visitor.exit();
    }
}

fn byte_length<T>(elements: usize) -> Result<usize, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::InputTooLong(elements))
}

fn next_nonzero(counter: &AtomicU64) -> Result<u64, MetalError> {
    let value = counter.fetch_add(1, Ordering::Relaxed);
    if value == 0 || value == u64::MAX {
        Err(invalid("sparse bytecode completion serial exhausted"))
    } else {
        Ok(value)
    }
}

fn invalid(message: impl Into<String>) -> MetalError {
    MetalError::InvalidInstructionReadRafGrouped(message.into())
}
