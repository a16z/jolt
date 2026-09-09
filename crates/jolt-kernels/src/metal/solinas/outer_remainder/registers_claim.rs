use std::{
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{foreign_types::ForeignType, Buffer};

use super::super::{registers_claim_reduction::RegistersClaimLinearComponents, Fp128, MetalError};

pub(super) const COMPONENTS: usize = 3;

static NEXT_COMPLETION_SERIAL: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RegistersClaimCarrierGeometry {
    pub(super) prefix_elements: usize,
    pub(super) suffix_elements: usize,
    pub(super) blocks: usize,
    pub(super) partial_elements: usize,
    pub(super) component_elements: usize,
    pub(super) partial_bytes: u64,
    pub(super) component_bytes: u64,
    pub(super) rd_bytes: u64,
    pub(super) owned_bytes: u64,
    pub(super) max_buffer_bytes: u64,
}

pub(super) fn carrier_geometry(rows: usize) -> Result<RegistersClaimCarrierGeometry, MetalError> {
    if rows < 4 || !rows.is_power_of_two() {
        return Err(MetalError::InvalidOuterRemainderRows(rows));
    }
    let log_t = rows.ilog2() as usize;
    let prefix_elements = 1usize
        .checked_shl(log_t.div_ceil(2) as u32)
        .ok_or(MetalError::InputTooLong(rows))?;
    let suffix_elements = rows / prefix_elements;
    let blocks = suffix_elements.min(256);
    let partial_elements = COMPONENTS
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(prefix_elements))
        .ok_or(MetalError::InputTooLong(rows))?;
    let component_elements = COMPONENTS
        .checked_mul(prefix_elements)
        .ok_or(MetalError::InputTooLong(rows))?;
    let partial_bytes = field_bytes(partial_elements)?;
    let component_bytes = field_bytes(component_elements)?;
    let rd_bytes = byte_length::<u64>(rows)?;
    let owned_bytes = partial_bytes
        .checked_add(component_bytes)
        .and_then(|value| value.checked_add(rd_bytes))
        .ok_or(MetalError::InputTooLong(rows))?;
    Ok(RegistersClaimCarrierGeometry {
        prefix_elements,
        suffix_elements,
        blocks,
        partial_elements,
        component_elements,
        partial_bytes,
        component_bytes,
        rd_bytes,
        owned_bytes,
        max_buffer_bytes: partial_bytes.max(component_bytes).max(rd_bytes),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OuterRegistersClaimCarrierReceipt {
    pub(crate) rows: usize,
    pub(crate) explicit_rows: usize,
    pub(crate) prefix_elements: usize,
    pub(crate) suffix_elements: usize,
    pub(crate) blocks: usize,
    pub(crate) device_registry_id: u64,
    pub(crate) source_generation: u64,
    pub(crate) source_compact_storage_id: usize,
    pub(crate) source_residual_storage_id: usize,
    pub(crate) partial_storage_id: usize,
    pub(crate) component_storage_id: usize,
    pub(crate) rd_storage_id: usize,
    pub(crate) partial_bytes: u64,
    pub(crate) component_bytes: u64,
    pub(crate) rd_bytes: u64,
    pub(crate) completion_serial: u64,
    pub(crate) row_scans: usize,
    pub(crate) command_buffers: usize,
    pub(crate) waits: usize,
    pub(crate) uploads: usize,
    pub(crate) prezero_dispatches: usize,
    pub(crate) complete_overwrite: bool,
}

pub(crate) struct OuterRegistersClaimCarrier {
    receipt: OuterRegistersClaimCarrierReceipt,
    components: RegistersClaimLinearComponents<AkitaField>,
    rd_write_value: Buffer,
}

impl OuterRegistersClaimCarrier {
    pub(super) fn new(
        receipt: OuterRegistersClaimCarrierReceipt,
        components: RegistersClaimLinearComponents<AkitaField>,
        rd_write_value: Buffer,
    ) -> Result<Self, MetalError> {
        let geometry = carrier_geometry(receipt.rows)?;
        let identities = [
            receipt.source_compact_storage_id,
            receipt.source_residual_storage_id,
            receipt.partial_storage_id,
            receipt.component_storage_id,
            receipt.rd_storage_id,
        ];
        if receipt.completion_serial == 0
            || receipt.source_generation == 0
            || receipt.device_registry_id == 0
            || receipt.explicit_rows > receipt.rows
            || receipt.prefix_elements != geometry.prefix_elements
            || receipt.suffix_elements != geometry.suffix_elements
            || receipt.blocks != geometry.blocks
            || receipt.partial_bytes != geometry.partial_bytes
            || receipt.component_bytes != geometry.component_bytes
            || receipt.rd_bytes != geometry.rd_bytes
            || !receipt.complete_overwrite
            || receipt.row_scans != 2
            || receipt.command_buffers != 1
            || receipt.waits != 1
            || receipt.uploads != 0
            || receipt.prezero_dispatches != 0
            || identities.contains(&0)
            || identities
                .iter()
                .enumerate()
                .any(|(index, identity)| identities[..index].contains(identity))
            || receipt.rd_storage_id != rd_write_value.as_ptr() as usize
            || receipt.rd_bytes != rd_write_value.length()
            || receipt.device_registry_id != rd_write_value.device().registry_id()
        {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "registers-claim carrier receipt is inconsistent",
            ));
        }
        let lengths = [
            components.rd_write_value.len(),
            components.rs1_value.len(),
            components.rs2_value.len(),
        ];
        if lengths
            .into_iter()
            .any(|length| length != receipt.prefix_elements)
        {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "registers-claim carrier component length is inconsistent",
            ));
        }
        Ok(Self {
            receipt,
            components,
            rd_write_value,
        })
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        OuterRegistersClaimCarrierReceipt,
        RegistersClaimLinearComponents<AkitaField>,
        Buffer,
    ) {
        (self.receipt, self.components, self.rd_write_value)
    }
}

pub(super) fn next_completion_serial() -> Result<u64, MetalError> {
    NEXT_COMPLETION_SERIAL
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| {
            MetalError::InvalidOuterRemainderConfig(
                "registers-claim carrier completion counter exhausted",
            )
        })
}

fn field_bytes(elements: usize) -> Result<u64, MetalError> {
    byte_length::<Fp128>(elements)
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::InputTooLong(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(elements))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "fixed geometry fixtures")]
mod tests {
    use super::carrier_geometry;

    #[test]
    fn carrier_geometry_matches_log_25_through_log_28_ledgers() {
        let log_25 = carrier_geometry(1 << 25).unwrap();
        assert_eq!(log_25.prefix_elements, 8192);
        assert_eq!(log_25.suffix_elements, 4096);
        assert_eq!(log_25.blocks, 256);
        assert_eq!(log_25.partial_bytes, 100_663_296);
        assert_eq!(log_25.component_bytes, 393_216);
        assert_eq!(log_25.rd_bytes, 268_435_456);
        assert_eq!(log_25.owned_bytes, 369_491_968);

        let log_26 = carrier_geometry(1 << 26).unwrap();
        assert_eq!(log_26.prefix_elements, 8192);
        assert_eq!(log_26.suffix_elements, 8192);
        assert_eq!(log_26.blocks, 256);
        assert_eq!(log_26.partial_bytes, 100_663_296);
        assert_eq!(log_26.component_bytes, 393_216);
        assert_eq!(log_26.rd_bytes, 536_870_912);
        assert_eq!(log_26.owned_bytes, 637_927_424);

        let log_27 = carrier_geometry(1 << 27).unwrap();
        assert_eq!(log_27.prefix_elements, 16_384);
        assert_eq!(log_27.suffix_elements, 8192);
        assert_eq!(log_27.blocks, 256);
        assert_eq!(log_27.owned_bytes, 1_275_854_848);

        let log_28 = carrier_geometry(1 << 28).unwrap();
        assert_eq!(log_28.prefix_elements, 16_384);
        assert_eq!(log_28.suffix_elements, 16_384);
        assert_eq!(log_28.blocks, 256);
        assert_eq!(log_28.owned_bytes, 2_349_596_672);
    }
}
