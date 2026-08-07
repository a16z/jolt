use core::mem::{align_of, size_of};

use super::HammingWeightV2Error;

pub const HAMMING_V2_SELECTORS: usize = 29;
pub const HAMMING_V2_BINS: usize = 256;
pub const HAMMING_V2_ADDRESS_ROUNDS: usize = 8;
pub const HAMMING_V2_HOT_PLANES: usize = HAMMING_V2_SELECTORS;
pub const HAMMING_V2_VALIDITY_PLANES: usize = 1;
pub const HAMMING_V2_SELECTOR_ORDER_VERSION: u32 = 1;
pub const HAMMING_V2_INNER_LOG2: usize = 15;
pub const HAMMING_V2_INNER_LENGTH: usize = 1 << HAMMING_V2_INNER_LOG2;
pub const HAMMING_V2_TILE_WIDTHS: [usize; 5] = [6, 6, 6, 6, 5];
pub const HAMMING_V2_ACCUMULATOR_THREADS: usize = 512;
pub const HAMMING_V2_FINALIZE_THREADS: usize = 1024;
pub const HAMMING_V2_FIELD_BYTES: u64 = 16;
pub const HAMMING_V2_ROW_BYTES: u64 = 40;

const _: () = assert!(1 << HAMMING_V2_ADDRESS_ROUNDS == HAMMING_V2_BINS);
const _: () = assert!(HAMMING_V2_TILE_WIDTHS[0] == 6);
const _: () = assert!(HAMMING_V2_TILE_WIDTHS[4] == 5);
const _: () = assert!(
    HAMMING_V2_TILE_WIDTHS[0]
        + HAMMING_V2_TILE_WIDTHS[1]
        + HAMMING_V2_TILE_WIDTHS[2]
        + HAMMING_V2_TILE_WIDTHS[3]
        + HAMMING_V2_TILE_WIDTHS[4]
        == HAMMING_V2_SELECTORS
);

/// Exact host/shader ABI of the retained consumer already in production.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightV2Params {
    pub rows: u32,
    pub e_in_length: u32,
    pub e_out_length: u32,
    pub selector_offset: u32,
    pub selectors_in_tile: u32,
    pub bins: u32,
    pub reserved: [u32; 2],
}

const _: [(); 32] = [(); size_of::<HammingWeightV2Params>()];
const _: [(); 4] = [(); align_of::<HammingWeightV2Params>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightV2Geometry {
    rows: usize,
    e_out_length: usize,
}

impl HammingWeightV2Geometry {
    pub fn new(rows: usize) -> Result<Self, HammingWeightV2Error> {
        if rows < HAMMING_V2_INNER_LENGTH || !rows.is_power_of_two() {
            return Err(HammingWeightV2Error::InvalidRows(rows));
        }
        let _ = shader_u32("rows", rows)?;
        let e_out_length = rows / HAMMING_V2_INNER_LENGTH;
        let _ = shader_u32("outer length", e_out_length)?;
        Ok(Self { rows, e_out_length })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn e_in_length(self) -> usize {
        HAMMING_V2_INNER_LENGTH
    }

    pub const fn e_out_length(self) -> usize {
        self.e_out_length
    }

    pub fn params(self, tile: usize) -> Result<HammingWeightV2Params, HammingWeightV2Error> {
        let selectors_in_tile = HAMMING_V2_TILE_WIDTHS.get(tile).copied().ok_or(
            HammingWeightV2Error::ReceiptMismatch {
                name: "tile index",
                expected: (HAMMING_V2_TILE_WIDTHS.len() - 1) as u64,
                got: tile as u64,
            },
        )?;
        let selector_offset = HAMMING_V2_TILE_WIDTHS[..tile].iter().sum::<usize>();
        Ok(HammingWeightV2Params {
            rows: shader_u32("rows", self.rows)?,
            e_in_length: HAMMING_V2_INNER_LENGTH as u32,
            e_out_length: shader_u32("outer length", self.e_out_length)?,
            selector_offset: selector_offset as u32,
            selectors_in_tile: selectors_in_tile as u32,
            bins: HAMMING_V2_BINS as u32,
            reserved: [0; 2],
        })
    }

    pub fn buffer_lengths(self) -> Result<HammingWeightV2BufferLengths, HammingWeightV2Error> {
        Ok(HammingWeightV2BufferLengths {
            hot_bytes: checked_mul(self.rows as u64, HAMMING_V2_HOT_PLANES as u64)?,
            e_in_fields: HAMMING_V2_INNER_LENGTH as u64,
            e_out_fields: self.e_out_length as u64,
            partial_fields: checked_product(&[
                self.e_out_length as u64,
                HAMMING_V2_TILE_WIDTHS[0] as u64,
                HAMMING_V2_BINS as u64,
            ])?,
            output_fields: checked_product(&[HAMMING_V2_SELECTORS as u64, HAMMING_V2_BINS as u64])?,
        })
    }

    pub const fn dispatch_plan(self) -> HammingWeightV2DispatchPlan {
        HammingWeightV2DispatchPlan {
            command_buffers: 1,
            encoders: 10,
            tile_dispatches: 5,
            finalize_dispatches: 5,
            completion_waits: 1,
            readbacks: 1,
            tile_threadgroups: self.e_out_length * 5,
            finalize_threadgroups: HAMMING_V2_SELECTORS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightV2BufferLengths {
    pub hot_bytes: u64,
    pub e_in_fields: u64,
    pub e_out_fields: u64,
    pub partial_fields: u64,
    pub output_fields: u64,
}

impl HammingWeightV2BufferLengths {
    pub fn consumer_owned_bytes(self) -> Result<u64, HammingWeightV2Error> {
        [
            checked_mul(self.e_in_fields, HAMMING_V2_FIELD_BYTES)?,
            checked_mul(self.e_out_fields, HAMMING_V2_FIELD_BYTES)?,
            checked_mul(self.partial_fields, HAMMING_V2_FIELD_BYTES)?,
            checked_mul(self.output_fields, HAMMING_V2_FIELD_BYTES)?,
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or(HammingWeightV2Error::ArithmeticOverflow)
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightV2DispatchPlan {
    pub command_buffers: u32,
    pub encoders: u32,
    pub tile_dispatches: u32,
    pub finalize_dispatches: u32,
    pub completion_waits: u32,
    pub readbacks: u32,
    pub tile_threadgroups: usize,
    pub finalize_threadgroups: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingHotLeaseEvidence {
    pub source_rows_storage_id: u64,
    pub hot_rows_storage_id: u64,
    pub device_registry_id: u64,
    pub proof_generation: u64,
    pub rows: u64,
    pub hot_bytes: u64,
    pub selector_order_version: u32,
    pub producer_command_completed: bool,
    pub complete_overwrite: bool,
    pub private_projection_dispatches: u32,
    pub row_upload_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingHotLeaseReceipt {
    evidence: HammingHotLeaseEvidence,
}

impl HammingHotLeaseReceipt {
    pub fn check(
        geometry: HammingWeightV2Geometry,
        evidence: HammingHotLeaseEvidence,
    ) -> Result<Self, HammingWeightV2Error> {
        check_identity("source rows", evidence.source_rows_storage_id)?;
        check_identity("hot rows", evidence.hot_rows_storage_id)?;
        check_identity("device registry", evidence.device_registry_id)?;
        if evidence.source_rows_storage_id == evidence.hot_rows_storage_id {
            return Err(HammingWeightV2Error::AliasedAllocations);
        }
        if evidence.proof_generation == 0 {
            return Err(HammingWeightV2Error::MissingGeneration);
        }
        check_exact(
            "lease rows",
            geometry.rows as u64,
            evidence.rows,
            |expected, got| HammingWeightV2Error::LeaseRows { expected, got },
        )?;
        let expected_hot_bytes = geometry.buffer_lengths()?.hot_bytes;
        check_exact(
            "lease bytes",
            expected_hot_bytes,
            evidence.hot_bytes,
            |expected, got| HammingWeightV2Error::LeaseBytes { expected, got },
        )?;
        if evidence.selector_order_version != HAMMING_V2_SELECTOR_ORDER_VERSION {
            return Err(HammingWeightV2Error::SelectorSchedule {
                expected: HAMMING_V2_SELECTOR_ORDER_VERSION,
                got: evidence.selector_order_version,
            });
        }
        if !evidence.producer_command_completed {
            return Err(HammingWeightV2Error::ProducerIncomplete);
        }
        if !evidence.complete_overwrite {
            return Err(HammingWeightV2Error::IncompleteOverwrite);
        }
        if evidence.private_projection_dispatches != 0 {
            return Err(HammingWeightV2Error::PrivateProjectionDispatches(
                evidence.private_projection_dispatches,
            ));
        }
        if evidence.row_upload_bytes != 0 {
            return Err(HammingWeightV2Error::RowUpload(evidence.row_upload_bytes));
        }
        Ok(Self { evidence })
    }

    pub const fn evidence(self) -> HammingHotLeaseEvidence {
        self.evidence
    }

    pub fn validate_binding(
        self,
        source_rows_storage_id: u64,
        device_registry_id: u64,
        proof_generation: u64,
    ) -> Result<(), HammingWeightV2Error> {
        for (name, expected, got) in [
            (
                "source rows storage id",
                source_rows_storage_id,
                self.evidence.source_rows_storage_id,
            ),
            (
                "device registry id",
                device_registry_id,
                self.evidence.device_registry_id,
            ),
            (
                "proof generation",
                proof_generation,
                self.evidence.proof_generation,
            ),
        ] {
            if expected != got {
                return Err(HammingWeightV2Error::ReceiptMismatch {
                    name,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightExecutionEvidence {
    pub source_rows_storage_id: u64,
    pub hot_rows_storage_id: u64,
    pub device_registry_id: u64,
    pub proof_generation: u64,
    pub command_buffers: u32,
    pub encoders: u32,
    pub tile_dispatches: u32,
    pub finalize_dispatches: u32,
    pub completion_waits: u32,
    pub readbacks: u32,
    pub row_upload_bytes: u64,
    pub private_projection_dispatches: u32,
    pub command_completed: bool,
    pub gpu_active_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightExecutionReceipt {
    evidence: HammingWeightExecutionEvidence,
}

impl HammingWeightExecutionReceipt {
    pub fn check(
        geometry: HammingWeightV2Geometry,
        lease: HammingHotLeaseReceipt,
        evidence: HammingWeightExecutionEvidence,
    ) -> Result<Self, HammingWeightV2Error> {
        let lease = lease.evidence();
        for (name, expected, got) in [
            (
                "source rows storage id",
                lease.source_rows_storage_id,
                evidence.source_rows_storage_id,
            ),
            (
                "hot rows storage id",
                lease.hot_rows_storage_id,
                evidence.hot_rows_storage_id,
            ),
            (
                "device registry id",
                lease.device_registry_id,
                evidence.device_registry_id,
            ),
            (
                "proof generation",
                lease.proof_generation,
                evidence.proof_generation,
            ),
        ] {
            check_receipt(name, expected, got)?;
        }
        let plan = geometry.dispatch_plan();
        for (name, expected, got) in [
            (
                "command buffers",
                u64::from(plan.command_buffers),
                u64::from(evidence.command_buffers),
            ),
            (
                "encoders",
                u64::from(plan.encoders),
                u64::from(evidence.encoders),
            ),
            (
                "tile dispatches",
                u64::from(plan.tile_dispatches),
                u64::from(evidence.tile_dispatches),
            ),
            (
                "finalize dispatches",
                u64::from(plan.finalize_dispatches),
                u64::from(evidence.finalize_dispatches),
            ),
            (
                "completion waits",
                u64::from(plan.completion_waits),
                u64::from(evidence.completion_waits),
            ),
            (
                "readbacks",
                u64::from(plan.readbacks),
                u64::from(evidence.readbacks),
            ),
            ("row upload bytes", 0, evidence.row_upload_bytes),
            (
                "private projection dispatches",
                0,
                u64::from(evidence.private_projection_dispatches),
            ),
        ] {
            check_receipt(name, expected, got)?;
        }
        if !evidence.command_completed {
            return Err(HammingWeightV2Error::ConsumerIncomplete);
        }
        if evidence.gpu_active_ns == 0 {
            return Err(HammingWeightV2Error::MissingGpuTimestamp);
        }
        Ok(Self { evidence })
    }

    pub const fn evidence(self) -> HammingWeightExecutionEvidence {
        self.evidence
    }
}

fn shader_u32(name: &'static str, value: usize) -> Result<u32, HammingWeightV2Error> {
    u32::try_from(value).map_err(|_| HammingWeightV2Error::ShaderIndexOverflow { name, value })
}

fn checked_mul(lhs: u64, rhs: u64) -> Result<u64, HammingWeightV2Error> {
    lhs.checked_mul(rhs)
        .ok_or(HammingWeightV2Error::ArithmeticOverflow)
}

fn checked_product(values: &[u64]) -> Result<u64, HammingWeightV2Error> {
    values
        .iter()
        .try_fold(1u64, |product, value| checked_mul(product, *value))
}

fn check_identity(name: &'static str, identity: u64) -> Result<(), HammingWeightV2Error> {
    if identity == 0 {
        Err(HammingWeightV2Error::MissingIdentity { name })
    } else {
        Ok(())
    }
}

fn check_exact(
    _name: &'static str,
    expected: u64,
    got: u64,
    error: impl FnOnce(u64, u64) -> HammingWeightV2Error,
) -> Result<(), HammingWeightV2Error> {
    if expected == got {
        Ok(())
    } else {
        Err(error(expected, got))
    }
}

fn check_receipt(name: &'static str, expected: u64, got: u64) -> Result<(), HammingWeightV2Error> {
    if expected == got {
        Ok(())
    } else {
        Err(HammingWeightV2Error::ReceiptMismatch {
            name,
            expected,
            got,
        })
    }
}
