use std::time::Duration;

use jolt_field::AkitaField;
use thiserror::Error;

use super::runtime::{BytecodeReadRafCsrObservation, BytecodeReadRafCsrTelemetry};
use super::{
    canonical_field_checksum, BytecodeReadRafError, BytecodeReadRafShape, BYTECODE_ADDRESS_ROUNDS,
    BYTECODE_ADDRESS_STAGES, BYTECODE_ADDRESS_VALUE_TABLES,
};

pub const BYTECODE_ADDRESS_DIRECT_HANDOFF_SCHEMA_VERSION: u32 = 1;
pub const BYTECODE_ADDRESS_COUNT_BYTES: usize = 4;

pub trait BytecodeReadRafResidentRowsLease {
    fn rows(&self) -> usize;
    fn device_registry_id(&self) -> u64;
    fn allocation_identity(&self) -> usize;
}

impl BytecodeReadRafResidentRowsLease for crate::metal::solinas::BooleanityRows {
    fn rows(&self) -> usize {
        crate::metal::solinas::BooleanityRows::len(self)
    }

    fn device_registry_id(&self) -> u64 {
        crate::metal::solinas::BooleanityRows::device_registry_id(self)
    }

    fn allocation_identity(&self) -> usize {
        crate::metal::solinas::BooleanityRows::allocation_identity(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerAddressCountsReceipt {
    pub device_registry_id: u64,
    pub source_rows_allocation_identity: usize,
    pub allocation_identity: usize,
    pub generation: u64,
    pub initialization_serial: u64,
    pub completed_serial: u64,
    pub elements: usize,
    pub bytes: usize,
}

impl ProducerAddressCountsReceipt {
    fn validate<L: BytecodeReadRafResidentRowsLease>(
        self,
        source: &L,
        shape: BytecodeReadRafShape,
        static_buffer_identities: &[usize; 9],
    ) -> Result<(), BytecodeReadRafHandoffError> {
        if self.device_registry_id == 0
            || self.source_rows_allocation_identity == 0
            || self.allocation_identity == 0
            || self.generation == 0
        {
            return Err(BytecodeReadRafHandoffError::ZeroProducerIdentity);
        }
        if self.device_registry_id != source.device_registry_id()
            || self.source_rows_allocation_identity != source.allocation_identity()
        {
            return Err(BytecodeReadRafHandoffError::ProducerCountSourceMismatch);
        }
        if self.initialization_serial == 0 || self.completed_serial < self.initialization_serial {
            return Err(BytecodeReadRafHandoffError::ProducerCountsIncomplete {
                initialized: self.initialization_serial,
                completed: self.completed_serial,
            });
        }
        if self.allocation_identity == source.allocation_identity()
            || static_buffer_identities.contains(&self.allocation_identity)
        {
            return Err(BytecodeReadRafHandoffError::AliasedAllocation(
                self.allocation_identity,
            ));
        }
        let expected_elements = shape
            .outer_length()
            .checked_mul(shape.addresses())
            .ok_or(BytecodeReadRafHandoffError::SizeOverflow)?;
        let expected_bytes = expected_elements
            .checked_mul(BYTECODE_ADDRESS_COUNT_BYTES)
            .ok_or(BytecodeReadRafHandoffError::SizeOverflow)?;
        if self.elements != expected_elements || self.bytes != expected_bytes {
            return Err(BytecodeReadRafHandoffError::ProducerCountShape {
                expected_elements,
                got_elements: self.elements,
                expected_bytes,
                got_bytes: self.bytes,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafHandoffTiming {
    pub submit_wall: Duration,
    pub overlap_wall: Duration,
    pub join_wall: Duration,
    pub total_wall: Duration,
    pub gpu_active: Duration,
    pub completed_before_join: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafHostShellContract {
    committed_program: bool,
}

impl BytecodeReadRafHostShellContract {
    pub const fn new(committed_program: bool) -> Self {
        Self { committed_program }
    }

    pub const fn address_rounds(self) -> usize {
        BYTECODE_ADDRESS_ROUNDS
    }

    pub const fn pushforward_tables(self) -> usize {
        BYTECODE_ADDRESS_STAGES
    }

    pub const fn raw_value_tables(self) -> usize {
        BYTECODE_ADDRESS_VALUE_TABLES
    }

    pub const fn committed_output_values(self) -> usize {
        if self.committed_program {
            BYTECODE_ADDRESS_VALUE_TABLES
        } else {
            0
        }
    }

    pub const fn committed_program(self) -> bool {
        self.committed_program
    }
}

#[must_use = "a validated bytecode address handoff must be consumed by the host shell"]
pub struct BytecodeReadRafDirectHandoff<L> {
    shape: BytecodeReadRafShape,
    output: Vec<AkitaField>,
    output_checksum: u64,
    entry_trace_address: usize,
    telemetry: BytecodeReadRafCsrTelemetry,
    timing: BytecodeReadRafHandoffTiming,
    source_rows_device_registry_id: u64,
    source_rows_allocation_identity: usize,
    static_buffer_identities: [usize; 9],
    producer_counts: Option<ProducerAddressCountsReceipt>,
    source_lease: L,
}

impl<L> BytecodeReadRafDirectHandoff<L> {
    pub const fn shape(&self) -> BytecodeReadRafShape {
        self.shape
    }

    pub const fn entry_trace_address(&self) -> usize {
        self.entry_trace_address
    }

    pub const fn output_checksum(&self) -> u64 {
        self.output_checksum
    }

    pub const fn telemetry(&self) -> BytecodeReadRafCsrTelemetry {
        self.telemetry
    }

    pub const fn timing(&self) -> BytecodeReadRafHandoffTiming {
        self.timing
    }

    pub const fn source_rows_device_registry_id(&self) -> u64 {
        self.source_rows_device_registry_id
    }

    pub const fn source_rows_allocation_identity(&self) -> usize {
        self.source_rows_allocation_identity
    }

    pub const fn output_allocation_identity(&self) -> usize {
        self.static_buffer_identities[8]
    }

    pub const fn producer_counts(&self) -> Option<ProducerAddressCountsReceipt> {
        self.producer_counts
    }

    pub fn pushforward_tables(&self) -> impl ExactSizeIterator<Item = &[AkitaField]> {
        self.output.chunks_exact(self.shape.addresses())
    }

    pub fn source_lease(&self) -> &L {
        &self.source_lease
    }

    pub fn into_flattened_pushforwards(self) -> Vec<AkitaField> {
        self.output
    }
}

/// Converts a completed CSR readback into the single-use host-shell input.
///
/// `host_entry_trace_address` must come from row zero of the same authoritative
/// witness cache, during preflight and before any transcript mutation.
pub fn admit_csr_observation<L: BytecodeReadRafResidentRowsLease>(
    source_lease: L,
    shape: BytecodeReadRafShape,
    observation: BytecodeReadRafCsrObservation,
    host_entry_trace_address: usize,
    producer_counts: Option<ProducerAddressCountsReceipt>,
) -> Result<BytecodeReadRafDirectHandoff<L>, BytecodeReadRafHandoffError> {
    if source_lease.rows() != shape.rows()
        || source_lease.device_registry_id() == 0
        || source_lease.allocation_identity() == 0
    {
        return Err(BytecodeReadRafHandoffError::SourceLeaseMismatch);
    }
    if observation.source_rows_device_registry_id != source_lease.device_registry_id()
        || observation.source_rows_storage_id != source_lease.allocation_identity()
    {
        return Err(BytecodeReadRafHandoffError::ObservationSourceMismatch);
    }
    if host_entry_trace_address >= shape.addresses() {
        return Err(BytecodeReadRafHandoffError::EntryTraceOutsideDomain {
            got: host_entry_trace_address,
            addresses: shape.addresses(),
        });
    }

    let expected_output_elements = BYTECODE_ADDRESS_STAGES
        .checked_mul(shape.addresses())
        .ok_or(BytecodeReadRafHandoffError::SizeOverflow)?;
    if observation.output.len() != expected_output_elements {
        return Err(BytecodeReadRafHandoffError::OutputElements {
            expected: expected_output_elements,
            got: observation.output.len(),
        });
    }
    validate_static_identities(
        source_lease.allocation_identity(),
        observation.static_buffer_identities,
    )?;
    let _ = observation.telemetry.status.validate(shape)?;
    observation.telemetry.diagnostics.validate(
        shape,
        observation.telemetry.status,
        super::BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )?;
    if let Some(receipt) = producer_counts {
        receipt.validate(&source_lease, shape, &observation.static_buffer_identities)?;
    }

    let output_checksum = canonical_field_checksum(&observation.output);
    Ok(BytecodeReadRafDirectHandoff {
        shape,
        output: observation.output,
        output_checksum,
        entry_trace_address: host_entry_trace_address,
        telemetry: observation.telemetry,
        timing: BytecodeReadRafHandoffTiming {
            submit_wall: observation.submit_wall,
            overlap_wall: observation.overlap_wall,
            join_wall: observation.join_wall,
            total_wall: observation.total_wall,
            gpu_active: observation.gpu_active,
            completed_before_join: observation.completed_before_join,
        },
        source_rows_device_registry_id: observation.source_rows_device_registry_id,
        source_rows_allocation_identity: observation.source_rows_storage_id,
        static_buffer_identities: observation.static_buffer_identities,
        producer_counts,
        source_lease,
    })
}

fn validate_static_identities(
    source_rows_allocation_identity: usize,
    identities: [usize; 9],
) -> Result<(), BytecodeReadRafHandoffError> {
    for (index, identity) in identities.into_iter().enumerate() {
        if identity == 0 {
            return Err(BytecodeReadRafHandoffError::ZeroStaticAllocation(index));
        }
        if identity == source_rows_allocation_identity || identities[..index].contains(&identity) {
            return Err(BytecodeReadRafHandoffError::AliasedAllocation(identity));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeReadRafCpuFallbackReason {
    TraceBelowCutoff,
    MissingResidentRows,
    CapacityUnavailable,
    PipelineUnavailable,
    PreTranscriptExecutionFailed,
}

#[must_use = "the frozen pre-transcript decision must be consumed after challenge derivation"]
pub enum BytecodeReadRafPreTranscriptDecision<L> {
    Direct(Box<BytecodeReadRafDirectHandoff<L>>),
    OptimizedCpu(BytecodeReadRafCpuFallbackReason),
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BytecodeReadRafHandoffError {
    #[error(transparent)]
    Packet(#[from] BytecodeReadRafError),
    #[error("bytecode address source lease does not match the relation shape")]
    SourceLeaseMismatch,
    #[error("bytecode address observation does not match the resident source")]
    ObservationSourceMismatch,
    #[error("bytecode address producer identity is zero")]
    ZeroProducerIdentity,
    #[error("bytecode address producer-count source does not match resident rows")]
    ProducerCountSourceMismatch,
    #[error(
        "bytecode address producer counts are incomplete: initialized {initialized}, completed {completed}"
    )]
    ProducerCountsIncomplete { initialized: u64, completed: u64 },
    #[error(
        "bytecode address producer-count shape mismatch: elements {got_elements}/{expected_elements}, bytes {got_bytes}/{expected_bytes}"
    )]
    ProducerCountShape {
        expected_elements: usize,
        got_elements: usize,
        expected_bytes: usize,
        got_bytes: usize,
    },
    #[error("bytecode address static allocation {0} is zero")]
    ZeroStaticAllocation(usize),
    #[error("bytecode address allocation {0} aliases another live allocation")]
    AliasedAllocation(usize),
    #[error("bytecode address output has {got} fields; expected {expected}")]
    OutputElements { expected: usize, got: usize },
    #[error("bytecode address entry trace {got} is outside {addresses} addresses")]
    EntryTraceOutsideDomain { got: usize, addresses: usize },
    #[error("bytecode address handoff size overflow")]
    SizeOverflow,
}
