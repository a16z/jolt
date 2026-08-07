use core::mem::{align_of, size_of};

use super::BooleanityAddressV2Error;

pub const BOOLEANITY_ADDRESS_V2_BINS: usize = 256;
pub const BOOLEANITY_ADDRESS_V2_SELECTORS: usize = 29;
pub const BOOLEANITY_ADDRESS_V2_HOT_PLANES: usize = BOOLEANITY_ADDRESS_V2_SELECTORS;
pub const BOOLEANITY_ADDRESS_V2_VALIDITY_PLANES: usize = 0;
pub const BOOLEANITY_ADDRESS_V2_ADDRESS_ROUNDS: u32 = 8;
pub const BOOLEANITY_ADDRESS_V2_FIRST_TILE_SELECTORS: usize = 6;
pub const BOOLEANITY_ADDRESS_V2_REMAINING_TILES: usize = 4;
pub const BOOLEANITY_ADDRESS_V2_REMAINING_SELECTORS: usize = 23;
pub const BOOLEANITY_ADDRESS_V2_DEFERRED_WORDS: usize = 5;
pub const BOOLEANITY_ADDRESS_V2_SIMD_WIDTH: usize = 32;
pub const BOOLEANITY_ADDRESS_V2_ACCUMULATOR_THREADS: usize = 512;
pub const BOOLEANITY_ADDRESS_V2_FINALIZE_THREADS: usize = 1024;
pub const BOOLEANITY_ADDRESS_V2_LOW_INNER_LOG2: usize = 15;
pub const BOOLEANITY_ADDRESS_V2_TARGET_INNER_LOG2: usize = 17;
pub const BOOLEANITY_ADDRESS_V2_TARGET_LOG_T: usize = 27;
pub const BOOLEANITY_ADDRESS_V2_SCHEDULE_VERSION: u32 = 2;
pub const BOOLEANITY_ADDRESS_V2_SELECTOR_ORDER_VERSION: u32 = 1;
pub const BOOLEANITY_ADDRESS_V2_INC_BIAS: u64 = 0x8080_8080_8080_8080;
pub const BOOLEANITY_ADDRESS_V2_FIELD_BYTES: u64 = 16;
pub const BOOLEANITY_ADDRESS_V2_ROW_BYTES: u64 = 40;

pub const BOOLEANITY_ADDRESS_V2_FIRST_SELECTOR_IDS: [u8; 6] = [16, 17, 18, 19, 0, 1];
pub const BOOLEANITY_ADDRESS_V2_REMAINING_SELECTOR_IDS: [u8; 23] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28,
];
pub const BOOLEANITY_ADDRESS_V2_REMAINING_TILE_OFFSETS: [usize; 5] = [0, 6, 12, 18, 23];

pub const BOOLEANITY_ADDRESS_V2_THREADGROUP_BYTES: usize =
    BOOLEANITY_ADDRESS_V2_FIRST_TILE_SELECTORS
        * BOOLEANITY_ADDRESS_V2_BINS
        * BOOLEANITY_ADDRESS_V2_DEFERRED_WORDS
        * size_of::<u32>();
pub const BOOLEANITY_ADDRESS_V2_FINALIZE_THREADGROUP_BYTES: usize =
    BOOLEANITY_ADDRESS_V2_FINALIZE_THREADS * BOOLEANITY_ADDRESS_V2_FIELD_BYTES as usize;

pub const PACK_AND_FIRST_PIPELINE: &str = "solinas_booleanity_address_v2_pack_and_first";
pub const PACKED_TILES_PIPELINE: &str = "solinas_booleanity_address_v2_packed_tiles";
pub const FINALIZE_PIPELINE: &str = "solinas_booleanity_address_v2_finalize";

pub const PACK_AND_FIRST_BUFFER_ROWS: u64 = 0;
pub const PACK_AND_FIRST_BUFFER_E_IN: u64 = 1;
pub const PACK_AND_FIRST_BUFFER_E_OUT: u64 = 2;
pub const PACK_AND_FIRST_BUFFER_HOT: u64 = 3;
pub const PACK_AND_FIRST_BUFFER_PARTIALS: u64 = 4;
pub const PACK_AND_FIRST_BUFFER_PARAMS: u64 = 5;

pub const PACKED_TILES_BUFFER_HOT: u64 = 0;
pub const PACKED_TILES_BUFFER_E_IN: u64 = 1;
pub const PACKED_TILES_BUFFER_E_OUT: u64 = 2;
pub const PACKED_TILES_BUFFER_PARTIALS: u64 = 3;
pub const PACKED_TILES_BUFFER_PARAMS: u64 = 4;

pub const FINALIZE_BUFFER_PARTIALS: u64 = 0;
pub const FINALIZE_BUFFER_OUTPUT: u64 = 1;
pub const FINALIZE_BUFFER_PARAMS: u64 = 2;

const _: () = assert!(BOOLEANITY_ADDRESS_V2_VALIDITY_PLANES == 0);
const _: () = assert!(BOOLEANITY_ADDRESS_V2_REMAINING_SELECTOR_IDS.len() == 23);
const _: () = assert!(BOOLEANITY_ADDRESS_V2_THREADGROUP_BYTES == 30_720);
const _: () = assert!(BOOLEANITY_ADDRESS_V2_FINALIZE_THREADGROUP_BYTES == 16_384);

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressV2Params {
    pub rows: u32,
    pub e_in_length: u32,
    pub e_out_length: u32,
    pub selector_count: u32,
    pub inc_bias: u64,
    pub schedule_version: u32,
    pub hot_planes: u32,
    pub remaining_tiles: u32,
    pub selector_order_version: u32,
}

const _: [(); 40] = [(); size_of::<BooleanityAddressV2Params>()];
const _: [(); 8] = [(); align_of::<BooleanityAddressV2Params>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressV2Geometry {
    rows: usize,
    log_t: usize,
    inner_log2: usize,
    e_in_length: usize,
    e_out_length: usize,
}

impl BooleanityAddressV2Geometry {
    pub fn new(rows: usize) -> Result<Self, BooleanityAddressV2Error> {
        if rows < (1 << BOOLEANITY_ADDRESS_V2_LOW_INNER_LOG2) || !rows.is_power_of_two() {
            return Err(BooleanityAddressV2Error::InvalidRows(rows));
        }
        let _ = shader_u32("rows", rows)?;
        let log_t = rows.ilog2() as usize;
        let inner_log2 = if log_t >= BOOLEANITY_ADDRESS_V2_TARGET_LOG_T {
            BOOLEANITY_ADDRESS_V2_TARGET_INNER_LOG2
        } else {
            BOOLEANITY_ADDRESS_V2_LOW_INNER_LOG2
        };
        let e_in_length = 1usize << inner_log2;
        let e_out_length = rows / e_in_length;
        let _ = shader_u32("e_out length", e_out_length)?;
        Ok(Self {
            rows,
            log_t,
            inner_log2,
            e_in_length,
            e_out_length,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn inner_log2(self) -> usize {
        self.inner_log2
    }

    pub const fn e_in_length(self) -> usize {
        self.e_in_length
    }

    pub const fn e_out_length(self) -> usize {
        self.e_out_length
    }

    pub fn params(self) -> Result<BooleanityAddressV2Params, BooleanityAddressV2Error> {
        Ok(BooleanityAddressV2Params {
            rows: shader_u32("rows", self.rows)?,
            e_in_length: shader_u32("e_in length", self.e_in_length)?,
            e_out_length: shader_u32("e_out length", self.e_out_length)?,
            selector_count: BOOLEANITY_ADDRESS_V2_SELECTORS as u32,
            inc_bias: BOOLEANITY_ADDRESS_V2_INC_BIAS,
            schedule_version: BOOLEANITY_ADDRESS_V2_SCHEDULE_VERSION,
            hot_planes: BOOLEANITY_ADDRESS_V2_HOT_PLANES as u32,
            remaining_tiles: BOOLEANITY_ADDRESS_V2_REMAINING_TILES as u32,
            selector_order_version: BOOLEANITY_ADDRESS_V2_SELECTOR_ORDER_VERSION,
        })
    }

    pub fn buffer_lengths(
        self,
    ) -> Result<BooleanityAddressV2BufferLengths, BooleanityAddressV2Error> {
        let rows = self.rows as u64;
        let output_fields = checked_mul(
            BOOLEANITY_ADDRESS_V2_SELECTORS as u64,
            BOOLEANITY_ADDRESS_V2_BINS as u64,
        )?;
        Ok(BooleanityAddressV2BufferLengths {
            resident_row_bytes: checked_mul(rows, BOOLEANITY_ADDRESS_V2_ROW_BYTES)?,
            hot_bytes: checked_mul(rows, BOOLEANITY_ADDRESS_V2_HOT_PLANES as u64)?,
            validity_bytes: 0,
            e_in_fields: self.e_in_length as u64,
            e_out_fields: self.e_out_length as u64,
            partial_fields: checked_mul(output_fields, self.e_out_length as u64)?,
            output_fields,
        })
    }

    pub fn dispatch_plan(
        self,
    ) -> Result<BooleanityAddressV2DispatchPlan, BooleanityAddressV2Error> {
        Ok(BooleanityAddressV2DispatchPlan {
            command_buffers: 1,
            encoders: 3,
            dispatches: 3,
            completion_waits: 1,
            readbacks: 1,
            original_row_scans: 1,
            pack_and_first_threadgroups: shader_u32("pack threadgroups", self.e_out_length)?,
            packed_tile_threadgroups: shader_u32(
                "packed tile threadgroups",
                self.e_out_length
                    .checked_mul(BOOLEANITY_ADDRESS_V2_REMAINING_TILES)
                    .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?,
            )?,
            finalize_threadgroups: BOOLEANITY_ADDRESS_V2_SELECTORS as u32,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressV2BufferLengths {
    pub resident_row_bytes: u64,
    pub hot_bytes: u64,
    pub validity_bytes: u64,
    pub e_in_fields: u64,
    pub e_out_fields: u64,
    pub partial_fields: u64,
    pub output_fields: u64,
}

impl BooleanityAddressV2BufferLengths {
    pub fn owned_bytes(self) -> Result<u64, BooleanityAddressV2Error> {
        checked_sum(&[
            self.hot_bytes,
            self.validity_bytes,
            checked_mul(self.e_in_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
            checked_mul(self.e_out_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
            checked_mul(self.partial_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
            checked_mul(self.output_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
        ])
    }

    pub fn validate(self, got: Self) -> Result<(), BooleanityAddressV2Error> {
        for (name, expected, got) in [
            (
                "resident rows",
                self.resident_row_bytes,
                got.resident_row_bytes,
            ),
            ("hot rows", self.hot_bytes, got.hot_bytes),
            ("validity", self.validity_bytes, got.validity_bytes),
            ("e_in", self.e_in_fields, got.e_in_fields),
            ("e_out", self.e_out_fields, got.e_out_fields),
            ("partials", self.partial_fields, got.partial_fields),
            ("output", self.output_fields, got.output_fields),
        ] {
            if expected != got {
                return Err(BooleanityAddressV2Error::BufferLength {
                    name,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressV2DispatchPlan {
    pub command_buffers: u32,
    pub encoders: u32,
    pub dispatches: u32,
    pub completion_waits: u32,
    pub readbacks: u32,
    pub original_row_scans: u32,
    pub pack_and_first_threadgroups: u32,
    pub packed_tile_threadgroups: u32,
    pub finalize_threadgroups: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressV2HotLeaseEvidence {
    pub source_rows_storage_id: u64,
    pub hot_rows_storage_id: u64,
    pub device_registry_id: u64,
    pub proof_generation: u64,
    pub rows: u64,
    pub hot_bytes: u64,
    pub validity_bytes: u64,
    pub schedule_version: u32,
    pub selector_order_version: u32,
    pub producer_command_completed: bool,
    pub complete_overwrite: bool,
    pub private_projection_dispatches: u32,
    pub row_upload_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressV2HotLeaseReceipt {
    evidence: BooleanityAddressV2HotLeaseEvidence,
}

impl BooleanityAddressV2HotLeaseReceipt {
    pub fn check(
        geometry: BooleanityAddressV2Geometry,
        evidence: BooleanityAddressV2HotLeaseEvidence,
    ) -> Result<Self, BooleanityAddressV2Error> {
        check_identity("source rows", evidence.source_rows_storage_id)?;
        check_identity("hot rows", evidence.hot_rows_storage_id)?;
        check_identity("device", evidence.device_registry_id)?;
        if evidence.source_rows_storage_id == evidence.hot_rows_storage_id {
            return Err(BooleanityAddressV2Error::AliasedAllocations);
        }
        if evidence.proof_generation == 0 {
            return Err(BooleanityAddressV2Error::MissingGeneration);
        }
        let lengths = geometry.buffer_lengths()?;
        for (name, expected, got) in [
            ("lease rows", geometry.rows as u64, evidence.rows),
            ("hot bytes", lengths.hot_bytes, evidence.hot_bytes),
            ("validity bytes", 0, evidence.validity_bytes),
            (
                "schedule version",
                BOOLEANITY_ADDRESS_V2_SCHEDULE_VERSION as u64,
                evidence.schedule_version as u64,
            ),
            (
                "selector order version",
                BOOLEANITY_ADDRESS_V2_SELECTOR_ORDER_VERSION as u64,
                evidence.selector_order_version as u64,
            ),
            (
                "private projection dispatches",
                0,
                evidence.private_projection_dispatches as u64,
            ),
            ("row upload bytes", 0, evidence.row_upload_bytes),
        ] {
            check_receipt(name, expected, got)?;
        }
        if !evidence.producer_command_completed {
            return Err(BooleanityAddressV2Error::ProducerIncomplete);
        }
        if !evidence.complete_overwrite {
            return Err(BooleanityAddressV2Error::IncompleteOverwrite);
        }
        Ok(Self { evidence })
    }

    pub const fn evidence(self) -> BooleanityAddressV2HotLeaseEvidence {
        self.evidence
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressV2LifecycleEvidence {
    pub allocation_ns: u64,
    pub first_touch_ns: u64,
    pub weight_prepare_ns: u64,
    pub encode_submit_wait_ns: u64,
    pub readback_ns: u64,
    pub host_rounds_ns: u64,
    pub unattributed_ns: u64,
    pub complete_member_ns: u64,
}

impl BooleanityAddressV2LifecycleEvidence {
    pub fn check(self) -> Result<(), BooleanityAddressV2Error> {
        let components_ns = checked_sum(&[
            self.allocation_ns,
            self.first_touch_ns,
            self.weight_prepare_ns,
            self.encode_submit_wait_ns,
            self.readback_ns,
            self.host_rounds_ns,
            self.unattributed_ns,
        ])?;
        if self.complete_member_ns == 0 || components_ns != self.complete_member_ns {
            return Err(BooleanityAddressV2Error::LifecycleMismatch {
                components_ns,
                complete_member_ns: self.complete_member_ns,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressV2ExecutionEvidence {
    pub source_rows_storage_id: u64,
    pub hot_rows_storage_id: u64,
    pub device_registry_id: u64,
    pub proof_generation: u64,
    pub command_buffers: u32,
    pub encoders: u32,
    pub dispatches: u32,
    pub completion_waits: u32,
    pub readbacks: u32,
    pub original_row_scans: u32,
    pub output_readback_bytes: u64,
    pub validity_bytes: u64,
    pub row_upload_bytes: u64,
    pub private_projection_dispatches: u32,
    pub host_fiat_shamir_rounds: u32,
    pub device_fiat_shamir_rounds: u32,
    pub command_completed: bool,
    pub gpu_active_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressV2ExecutionReceipt {
    evidence: BooleanityAddressV2ExecutionEvidence,
    lifecycle: BooleanityAddressV2LifecycleEvidence,
}

impl BooleanityAddressV2ExecutionReceipt {
    pub fn check(
        geometry: BooleanityAddressV2Geometry,
        lease: BooleanityAddressV2HotLeaseReceipt,
        evidence: BooleanityAddressV2ExecutionEvidence,
        lifecycle: BooleanityAddressV2LifecycleEvidence,
    ) -> Result<Self, BooleanityAddressV2Error> {
        let lease = lease.evidence();
        let plan = geometry.dispatch_plan()?;
        for (name, expected, got) in [
            (
                "source rows",
                lease.source_rows_storage_id,
                evidence.source_rows_storage_id,
            ),
            (
                "hot rows",
                lease.hot_rows_storage_id,
                evidence.hot_rows_storage_id,
            ),
            (
                "device",
                lease.device_registry_id,
                evidence.device_registry_id,
            ),
            (
                "proof generation",
                lease.proof_generation,
                evidence.proof_generation,
            ),
            (
                "command buffers",
                plan.command_buffers as u64,
                evidence.command_buffers as u64,
            ),
            ("encoders", plan.encoders as u64, evidence.encoders as u64),
            (
                "dispatches",
                plan.dispatches as u64,
                evidence.dispatches as u64,
            ),
            (
                "completion waits",
                plan.completion_waits as u64,
                evidence.completion_waits as u64,
            ),
            (
                "readbacks",
                plan.readbacks as u64,
                evidence.readbacks as u64,
            ),
            (
                "original row scans",
                plan.original_row_scans as u64,
                evidence.original_row_scans as u64,
            ),
            (
                "output readback bytes",
                geometry.buffer_lengths()?.output_fields * BOOLEANITY_ADDRESS_V2_FIELD_BYTES,
                evidence.output_readback_bytes,
            ),
            ("validity bytes", 0, evidence.validity_bytes),
            ("row upload bytes", 0, evidence.row_upload_bytes),
            (
                "private projection dispatches",
                0,
                evidence.private_projection_dispatches as u64,
            ),
            (
                "host Fiat-Shamir rounds",
                BOOLEANITY_ADDRESS_V2_ADDRESS_ROUNDS as u64,
                evidence.host_fiat_shamir_rounds as u64,
            ),
            (
                "device Fiat-Shamir rounds",
                0,
                evidence.device_fiat_shamir_rounds as u64,
            ),
        ] {
            check_receipt(name, expected, got)?;
        }
        if !evidence.command_completed {
            return Err(BooleanityAddressV2Error::CommandIncomplete);
        }
        if evidence.gpu_active_ns == 0 {
            return Err(BooleanityAddressV2Error::MissingGpuTimestamp);
        }
        lifecycle.check()?;
        Ok(Self {
            evidence,
            lifecycle,
        })
    }

    pub const fn evidence(self) -> BooleanityAddressV2ExecutionEvidence {
        self.evidence
    }

    pub const fn lifecycle(self) -> BooleanityAddressV2LifecycleEvidence {
        self.lifecycle
    }
}

pub(crate) fn validate_weight_shape(
    rows: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), BooleanityAddressV2Error> {
    if e_in == 0
        || e_out == 0
        || e_in
            .checked_mul(e_out)
            .is_none_or(|covered| covered != rows)
    {
        return Err(BooleanityAddressV2Error::WeightShape { rows, e_in, e_out });
    }
    Ok(())
}

pub(crate) fn checked_mul(left: u64, right: u64) -> Result<u64, BooleanityAddressV2Error> {
    left.checked_mul(right)
        .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)
}

pub(crate) fn checked_sum(values: &[u64]) -> Result<u64, BooleanityAddressV2Error> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)
    })
}

fn shader_u32(name: &'static str, value: usize) -> Result<u32, BooleanityAddressV2Error> {
    u32::try_from(value).map_err(|_| BooleanityAddressV2Error::ShaderIndexOverflow { name, value })
}

fn check_identity(name: &'static str, value: u64) -> Result<(), BooleanityAddressV2Error> {
    if value == 0 {
        return Err(BooleanityAddressV2Error::MissingIdentity(name));
    }
    Ok(())
}

fn check_receipt(
    name: &'static str,
    expected: u64,
    got: u64,
) -> Result<(), BooleanityAddressV2Error> {
    if expected != got {
        return Err(BooleanityAddressV2Error::ReceiptMismatch {
            name,
            expected,
            got,
        });
    }
    Ok(())
}
