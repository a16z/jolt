//! Checked producer receipts for the staged address-major carrier.

use core::fmt;

pub const ADDRESS_LOG2: u32 = 13;
pub const INNER_LOG2: u32 = 15;
pub const SHORT_THRESHOLD: usize = 32;
pub const SIMD_WIDTH: usize = 32;
pub const RESIDENT_ROW_BYTES: usize = 40;
pub const CELL_BYTES: usize = 4;
pub const INNER_SIGN_BYTES: usize = 4;
pub const MAGNITUDE_BYTES: usize = 8;

const INNER_INDEX_MASK: u32 = (1 << INNER_LOG2) - 1;
const INNER_SIGN_BIT: u32 = 1 << 31;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressMajorShape {
    log_rows: u32,
    log_addresses: u32,
    inner_log2: u32,
}

impl AddressMajorShape {
    pub const LOG26: Self = Self {
        log_rows: 26,
        log_addresses: ADDRESS_LOG2,
        inner_log2: INNER_LOG2,
    };

    pub const LOG28: Self = Self {
        log_rows: 28,
        log_addresses: ADDRESS_LOG2,
        inner_log2: INNER_LOG2,
    };

    pub fn new(log_rows: u32, log_addresses: u32, inner_log2: u32) -> Result<Self, CarrierError> {
        let shape = Self {
            log_rows,
            log_addresses,
            inner_log2,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn production(log_rows: u32) -> Result<Self, CarrierError> {
        Self::new(log_rows, ADDRESS_LOG2, INNER_LOG2)
    }

    pub fn validate(self) -> Result<(), CarrierError> {
        if self.inner_log2 == 0
            || self.inner_log2 > INNER_LOG2
            || self.log_rows < self.inner_log2
            || self.log_rows >= usize::BITS
            || self.log_addresses == 0
            || self.log_addresses >= usize::BITS
        {
            return Err(CarrierError::InvalidShape);
        }
        let _ = self.cells()?;
        Ok(())
    }

    pub fn rows(self) -> Result<usize, CarrierError> {
        self.validate_exponents()?;
        1usize
            .checked_shl(self.log_rows)
            .ok_or(CarrierError::Overflow("rows"))
    }

    pub fn addresses(self) -> Result<usize, CarrierError> {
        self.validate_exponents()?;
        1usize
            .checked_shl(self.log_addresses)
            .ok_or(CarrierError::Overflow("addresses"))
    }

    pub fn inner_length(self) -> Result<usize, CarrierError> {
        self.validate_exponents()?;
        1usize
            .checked_shl(self.inner_log2)
            .ok_or(CarrierError::Overflow("inner length"))
    }

    pub fn outer_length(self) -> Result<usize, CarrierError> {
        Ok(self.rows()? / self.inner_length()?)
    }

    pub fn cells(self) -> Result<usize, CarrierError> {
        checked_mul(
            "address-major cells",
            self.addresses()?,
            self.outer_length()?,
        )
    }

    pub const fn log_rows(self) -> u32 {
        self.log_rows
    }

    pub const fn log_addresses(self) -> u32 {
        self.log_addresses
    }

    pub const fn inner_log2(self) -> u32 {
        self.inner_log2
    }

    fn validate_exponents(self) -> Result<(), CarrierError> {
        if self.inner_log2 == 0
            || self.inner_log2 > INNER_LOG2
            || self.log_rows < self.inner_log2
            || self.log_rows >= usize::BITS
            || self.log_addresses == 0
            || self.log_addresses >= usize::BITS
        {
            Err(CarrierError::InvalidShape)
        } else {
            Ok(())
        }
    }
}

/// One address cell. Both fields are local to a single outer block.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedCell(u32);

impl PackedCell {
    pub fn new(start: usize, count: usize) -> Result<Self, CarrierError> {
        let start = u16::try_from(start).map_err(|_| CarrierError::InvalidCell)?;
        let count = u16::try_from(count).map_err(|_| CarrierError::InvalidCell)?;
        Ok(Self(u32::from(start) | (u32::from(count) << 16)))
    }

    pub const fn start(self) -> usize {
        (self.0 & 0xffff) as usize
    }

    pub const fn count(self) -> usize {
        (self.0 >> 16) as usize
    }

    pub const fn word(self) -> u32 {
        self.0
    }
}

/// Compact inner index plus the sign bit of the fused increment.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedInnerSign(u32);

impl PackedInnerSign {
    pub fn new(inner: usize, negative: bool) -> Result<Self, CarrierError> {
        if inner > INNER_INDEX_MASK as usize {
            return Err(CarrierError::InvalidInnerSign);
        }
        Ok(Self(inner as u32 | (u32::from(negative) << 31)))
    }

    pub fn from_word(word: u32) -> Result<Self, CarrierError> {
        if word & !(INNER_INDEX_MASK | INNER_SIGN_BIT) != 0 {
            return Err(CarrierError::InvalidInnerSign);
        }
        Ok(Self(word))
    }

    pub const fn inner(self) -> usize {
        (self.0 & INNER_INDEX_MASK) as usize
    }

    pub const fn negative(self) -> bool {
        self.0 & INNER_SIGN_BIT != 0
    }

    pub const fn word(self) -> u32 {
        self.0
    }
}

/// Stable identity of the stage-5 resident row producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerIdentity {
    device_registry_id: u64,
    source_allocation_identity: usize,
    source_allocation_bytes: usize,
    generation: u64,
    rows: usize,
}

impl ProducerIdentity {
    pub fn new(
        device_registry_id: u64,
        source_allocation_identity: usize,
        source_allocation_bytes: usize,
        generation: u64,
        rows: usize,
    ) -> Result<Self, CarrierError> {
        if device_registry_id == 0 {
            return Err(CarrierError::MissingIdentity("device"));
        }
        if source_allocation_identity == 0 {
            return Err(CarrierError::MissingIdentity("source rows"));
        }
        if generation == 0 {
            return Err(CarrierError::MissingIdentity("producer generation"));
        }
        let expected_bytes = checked_mul("resident row bytes", rows, RESIDENT_ROW_BYTES)?;
        if rows == 0 || source_allocation_bytes != expected_bytes {
            return Err(CarrierError::PlaneShape {
                plane: "source rows",
                expected_elements: rows,
                got_elements: rows,
                expected_bytes,
                got_bytes: source_allocation_bytes,
            });
        }
        Ok(Self {
            device_registry_id,
            source_allocation_identity,
            source_allocation_bytes,
            generation,
            rows,
        })
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn source_allocation_identity(self) -> usize {
        self.source_allocation_identity
    }

    pub const fn source_allocation_bytes(self) -> usize {
        self.source_allocation_bytes
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn rows(self) -> usize {
        self.rows
    }
}

/// Allocation metadata captured after a producer initializes a plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaneReceipt {
    allocation_identity: usize,
    elements: usize,
    bytes: usize,
}

impl PlaneReceipt {
    pub fn new(
        allocation_identity: usize,
        elements: usize,
        bytes: usize,
    ) -> Result<Self, CarrierError> {
        if allocation_identity == 0 {
            return Err(CarrierError::MissingIdentity("carrier plane"));
        }
        Ok(Self {
            allocation_identity,
            elements,
            bytes,
        })
    }

    pub const fn allocation_identity(self) -> usize {
        self.allocation_identity
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }

    fn validate_exact(
        self,
        plane: &'static str,
        expected_elements: usize,
        element_bytes: usize,
    ) -> Result<(), CarrierError> {
        let expected_bytes = checked_mul("carrier plane bytes", expected_elements, element_bytes)?;
        if self.elements != expected_elements || self.bytes != expected_bytes {
            return Err(CarrierError::PlaneShape {
                plane,
                expected_elements,
                got_elements: self.elements,
                expected_bytes,
                got_bytes: self.bytes,
            });
        }
        Ok(())
    }
}

/// Counters emitted when stage 5 publishes address counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountPublication {
    pub initialized_cells: usize,
    pub count_updates: usize,
    pub counted_rows: usize,
    pub completed_outer_blocks: usize,
    pub invalid_rows: u32,
    pub reserved: [u32; 3],
    pub additional_source_scans: u32,
    pub member_source_read_bytes: u64,
    pub host_staging_bytes: u64,
    pub upload_copy_bytes: u64,
}

/// Checked count state. It cannot be constructed by declaring counts resident.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedProducerCounts {
    shape: AddressMajorShape,
    producer: ProducerIdentity,
    cells: PlaneReceipt,
}

impl ValidatedProducerCounts {
    pub fn publish(
        shape: AddressMajorShape,
        producer: ProducerIdentity,
        cells: PlaneReceipt,
        publication: CountPublication,
    ) -> Result<Self, CarrierError> {
        shape.validate()?;
        let rows = shape.rows()?;
        let outer = shape.outer_length()?;
        let cell_count = shape.cells()?;
        if producer.rows != rows {
            return Err(CarrierError::ProducerRows {
                expected: rows,
                got: producer.rows,
            });
        }
        if cells.allocation_identity == producer.source_allocation_identity {
            return Err(CarrierError::AliasedAllocation("packed cells"));
        }
        cells.validate_exact("packed cells", cell_count, CELL_BYTES)?;
        if publication.initialized_cells != cell_count
            || publication.count_updates != rows
            || publication.counted_rows != rows
            || publication.completed_outer_blocks != outer
            || publication.invalid_rows != 0
            || publication.reserved != [0; 3]
        {
            return Err(CarrierError::InvalidPublication("producer counts"));
        }
        if publication.additional_source_scans != 0
            || publication.member_source_read_bytes != 0
            || publication.host_staging_bytes != 0
            || publication.upload_copy_bytes != 0
        {
            return Err(CarrierError::ForbiddenMemberTraffic {
                scans: publication.additional_source_scans,
                read_bytes: publication.member_source_read_bytes,
                staging_bytes: publication.host_staging_bytes,
                upload_bytes: publication.upload_copy_bytes,
            });
        }
        Ok(Self {
            shape,
            producer,
            cells,
        })
    }

    pub const fn shape(self) -> AddressMajorShape {
        self.shape
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub const fn cells(self) -> PlaneReceipt {
        self.cells
    }
}

/// Exact worker schedule counters derived from all published cells.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopologyScheduleReceipt {
    pub short_occurrences: u64,
    pub long_occurrences: u64,
    pub short_runs: u64,
    pub long_runs: u64,
    pub padded_short_lanes: u64,
    pub padded_long_lanes: u64,
    pub maximum_run: u64,
}

impl TopologyScheduleReceipt {
    pub fn validate(self, shape: AddressMajorShape) -> Result<(), CarrierError> {
        let rows = to_u64("rows", shape.rows()?)?;
        let outer = to_u64("outer blocks", shape.outer_length()?)?;
        let cells = to_u64("cells", shape.cells()?)?;
        let inner = to_u64("inner length", shape.inner_length()?)?;
        let short_threshold = SHORT_THRESHOLD as u64;
        let simd = SIMD_WIDTH as u64;
        let runs = add("runs", self.short_runs, self.long_runs)?;
        if add(
            "topology occurrences",
            self.short_occurrences,
            self.long_occurrences,
        )? != rows
            || runs < outer
            || runs > cells
            || self.short_occurrences < self.short_runs
            || self.short_occurrences
                > mul("short occurrence ceiling", self.short_runs, short_threshold)?
            || self.long_occurrences
                < mul("long occurrence floor", self.long_runs, short_threshold + 1)?
            || self.long_occurrences > mul("long occurrence ceiling", self.long_runs, inner)?
            || self.maximum_run == 0
            || self.maximum_run > inner
            || (self.long_runs != 0 && self.maximum_run <= short_threshold)
            || (self.long_runs == 0 && self.maximum_run > short_threshold)
            || self.long_occurrences
                > mul("maximum-run coverage", self.long_runs, self.maximum_run)?
        {
            return Err(CarrierError::InvalidTopology);
        }

        if self.short_runs == 0 {
            if self.short_occurrences != 0 || self.padded_short_lanes != 0 {
                return Err(CarrierError::InvalidTopology);
            }
        } else {
            let expected_padded = mul("short run lanes", simd, self.short_runs)?;
            if self.padded_short_lanes != expected_padded {
                return Err(CarrierError::InvalidTopology);
            }
        }

        if self.long_runs == 0 {
            if self.long_occurrences != 0 || self.padded_long_lanes != 0 {
                return Err(CarrierError::InvalidTopology);
            }
        } else {
            let maximum_padded = add(
                "maximum long padding",
                self.long_occurrences,
                mul("long masked lanes", simd - 1, self.long_runs)?,
            )?;
            if !self.padded_long_lanes.is_multiple_of(simd)
                || self.padded_long_lanes < self.long_occurrences
                || self.padded_long_lanes > maximum_padded
            {
                return Err(CarrierError::InvalidTopology);
            }
        }
        Ok(())
    }

    pub fn runs(self) -> Result<u64, CarrierError> {
        add("runs", self.short_runs, self.long_runs)
    }

    pub fn padded_lanes(self) -> Result<u64, CarrierError> {
        add(
            "padded lanes",
            self.padded_short_lanes,
            self.padded_long_lanes,
        )
    }
}

/// Counters emitted after the prefix/scatter stage publishes the carrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterPublication {
    pub scattered_rows: usize,
    pub cursor_updates: usize,
    pub completed_outer_blocks: usize,
    pub invalid_rows: u32,
    pub reserved: [u32; 3],
    pub producer_resident_scans: u32,
    pub member_resident_scans: u32,
    pub source_requested_bytes: u64,
    pub compact_write_bytes: u64,
    pub cell_write_bytes: u64,
    pub member_source_read_bytes: u64,
    pub host_staging_bytes: u64,
    pub upload_copy_bytes: u64,
    pub first_push_pc: usize,
    pub producer_incremental_wall_ns: Option<u64>,
    pub producer_gpu_active_ns: Option<u64>,
}

/// Immutable carrier accepted by the address member and its roof model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedAddressMajorCarrier {
    counts: ValidatedProducerCounts,
    inner_sign: PlaneReceipt,
    magnitude: PlaneReceipt,
    topology: TopologyScheduleReceipt,
    first_push_pc: usize,
    producer_incremental_wall_ns: Option<u64>,
    producer_gpu_active_ns: Option<u64>,
}

impl ValidatedAddressMajorCarrier {
    pub fn publish(
        counts: ValidatedProducerCounts,
        inner_sign: PlaneReceipt,
        magnitude: PlaneReceipt,
        topology: TopologyScheduleReceipt,
        publication: ScatterPublication,
    ) -> Result<Self, CarrierError> {
        let shape = counts.shape;
        let rows = shape.rows()?;
        let outer = shape.outer_length()?;
        let addresses = shape.addresses()?;
        let cells = shape.cells()?;
        inner_sign.validate_exact("inner/sign stream", rows, INNER_SIGN_BYTES)?;
        magnitude.validate_exact("magnitude stream", rows, MAGNITUDE_BYTES)?;
        for (plane, identity) in [
            ("packed cells", counts.cells.allocation_identity),
            ("inner/sign stream", inner_sign.allocation_identity),
            ("magnitude stream", magnitude.allocation_identity),
        ] {
            if identity == counts.producer.source_allocation_identity {
                return Err(CarrierError::AliasedAllocation(plane));
            }
        }
        if counts.cells.allocation_identity == inner_sign.allocation_identity
            || counts.cells.allocation_identity == magnitude.allocation_identity
            || inner_sign.allocation_identity == magnitude.allocation_identity
        {
            return Err(CarrierError::AliasedAllocation("carrier planes"));
        }
        topology.validate(shape)?;

        let expected_source_bytes = mul("scatter source bytes", 16, to_u64("rows", rows)?)?;
        let expected_compact_bytes = mul("scatter compact bytes", 12, to_u64("rows", rows)?)?;
        let expected_cell_bytes = mul("scatter cell bytes", 4, to_u64("cells", cells)?)?;
        if publication.scattered_rows != rows
            || publication.cursor_updates != rows
            || publication.completed_outer_blocks != outer
            || publication.invalid_rows != 0
            || publication.reserved != [0; 3]
            || publication.producer_resident_scans != 1
            || publication.source_requested_bytes != expected_source_bytes
            || publication.compact_write_bytes != expected_compact_bytes
            || publication.cell_write_bytes != expected_cell_bytes
            || publication.first_push_pc >= addresses
        {
            return Err(CarrierError::InvalidPublication("producer scatter"));
        }
        if publication.member_resident_scans != 0
            || publication.member_source_read_bytes != 0
            || publication.host_staging_bytes != 0
            || publication.upload_copy_bytes != 0
        {
            return Err(CarrierError::ForbiddenMemberTraffic {
                scans: publication.member_resident_scans,
                read_bytes: publication.member_source_read_bytes,
                staging_bytes: publication.host_staging_bytes,
                upload_bytes: publication.upload_copy_bytes,
            });
        }
        if let (Some(wall), Some(active)) = (
            publication.producer_incremental_wall_ns,
            publication.producer_gpu_active_ns,
        ) {
            if active > wall {
                return Err(CarrierError::InvalidPublication("producer timing"));
            }
        }
        Ok(Self {
            counts,
            inner_sign,
            magnitude,
            topology,
            first_push_pc: publication.first_push_pc,
            producer_incremental_wall_ns: publication.producer_incremental_wall_ns,
            producer_gpu_active_ns: publication.producer_gpu_active_ns,
        })
    }

    pub const fn shape(self) -> AddressMajorShape {
        self.counts.shape
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.counts.producer
    }

    pub const fn cells(self) -> PlaneReceipt {
        self.counts.cells
    }

    pub const fn inner_sign(self) -> PlaneReceipt {
        self.inner_sign
    }

    pub const fn magnitude(self) -> PlaneReceipt {
        self.magnitude
    }

    pub const fn topology(self) -> TopologyScheduleReceipt {
        self.topology
    }

    pub const fn first_push_pc(self) -> usize {
        self.first_push_pc
    }

    pub const fn producer_incremental_wall_ns(self) -> Option<u64> {
        self.producer_incremental_wall_ns
    }

    pub const fn producer_gpu_active_ns(self) -> Option<u64> {
        self.producer_gpu_active_ns
    }

    pub fn validate_consumer(self, binding: ConsumerBinding) -> Result<(), CarrierError> {
        if binding.device_registry_id != self.producer().device_registry_id {
            return Err(CarrierError::ConsumerMismatch("device"));
        }
        if binding.source_allocation_identity != self.producer().source_allocation_identity
            || binding.source_allocation_bytes != self.producer().source_allocation_bytes
            || binding.generation != self.producer().generation
        {
            return Err(CarrierError::ConsumerMismatch("source rows"));
        }
        for (plane, expected, got) in [
            ("packed cells", self.cells(), binding.cells),
            ("inner/sign stream", self.inner_sign, binding.inner_sign),
            ("magnitude stream", self.magnitude, binding.magnitude),
        ] {
            if expected != got {
                return Err(CarrierError::ConsumerMismatch(plane));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConsumerBinding {
    pub device_registry_id: u64,
    pub source_allocation_identity: usize,
    pub source_allocation_bytes: usize,
    pub generation: u64,
    pub cells: PlaneReceipt,
    pub inner_sign: PlaneReceipt,
    pub magnitude: PlaneReceipt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CarrierError {
    InvalidShape,
    InvalidCell,
    InvalidInnerSign,
    InvalidTopology,
    MissingIdentity(&'static str),
    AliasedAllocation(&'static str),
    ProducerRows {
        expected: usize,
        got: usize,
    },
    PlaneShape {
        plane: &'static str,
        expected_elements: usize,
        got_elements: usize,
        expected_bytes: usize,
        got_bytes: usize,
    },
    InvalidPublication(&'static str),
    ForbiddenMemberTraffic {
        scans: u32,
        read_bytes: u64,
        staging_bytes: u64,
        upload_bytes: u64,
    },
    ConsumerMismatch(&'static str),
    Overflow(&'static str),
}

impl fmt::Display for CarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid address-major shape"),
            Self::InvalidCell => f.write_str("address-major cell does not fit its packed ABI"),
            Self::InvalidInnerSign => f.write_str("invalid packed inner/sign word"),
            Self::InvalidTopology => f.write_str("invalid address-major schedule receipt"),
            Self::MissingIdentity(plane) => write!(f, "missing {plane} identity"),
            Self::AliasedAllocation(plane) => write!(f, "unexpected alias in {plane}"),
            Self::ProducerRows { expected, got } => {
                write!(f, "producer has {got} rows, expected {expected}")
            }
            Self::PlaneShape {
                plane,
                expected_elements,
                got_elements,
                expected_bytes,
                got_bytes,
            } => write!(
                f,
                "{plane} has {got_elements} elements/{got_bytes} bytes, expected {expected_elements}/{expected_bytes}"
            ),
            Self::InvalidPublication(stage) => write!(f, "invalid {stage} publication"),
            Self::ForbiddenMemberTraffic {
                scans,
                read_bytes,
                staging_bytes,
                upload_bytes,
            } => write!(
                f,
                "member-local carrier traffic: scans={scans}, reads={read_bytes}, staging={staging_bytes}, upload={upload_bytes}"
            ),
            Self::ConsumerMismatch(plane) => write!(f, "{plane} consumer binding changed"),
            Self::Overflow(name) => write!(f, "{name} overflowed"),
        }
    }
}

impl std::error::Error for CarrierError {}

fn checked_mul(name: &'static str, left: usize, right: usize) -> Result<usize, CarrierError> {
    left.checked_mul(right).ok_or(CarrierError::Overflow(name))
}

fn to_u64(name: &'static str, value: usize) -> Result<u64, CarrierError> {
    u64::try_from(value).map_err(|_| CarrierError::Overflow(name))
}

fn add(name: &'static str, left: u64, right: u64) -> Result<u64, CarrierError> {
    left.checked_add(right).ok_or(CarrierError::Overflow(name))
}

fn mul(name: &'static str, left: u64, right: u64) -> Result<u64, CarrierError> {
    left.checked_mul(right).ok_or(CarrierError::Overflow(name))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked fixtures")]
mod tests {
    use super::*;

    fn measured_schedule() -> TopologyScheduleReceipt {
        TopologyScheduleReceipt {
            short_occurrences: 1_239,
            long_occurrences: 67_107_625,
            short_runs: 1_059,
            long_runs: 18_949,
            padded_short_lanes: 33_888,
            padded_long_lanes: 67_695_040,
            maximum_run: 32_768,
        }
    }

    fn ready_carrier() -> ValidatedAddressMajorCarrier {
        let shape = AddressMajorShape::LOG26;
        let rows = shape.rows().unwrap();
        let cells_count = shape.cells().unwrap();
        let producer = ProducerIdentity::new(7, 11, rows * RESIDENT_ROW_BYTES, 13, rows).unwrap();
        let cells = PlaneReceipt::new(17, cells_count, cells_count * CELL_BYTES).unwrap();
        let counts = ValidatedProducerCounts::publish(
            shape,
            producer,
            cells,
            CountPublication {
                initialized_cells: cells_count,
                count_updates: rows,
                counted_rows: rows,
                completed_outer_blocks: shape.outer_length().unwrap(),
                invalid_rows: 0,
                reserved: [0; 3],
                additional_source_scans: 0,
                member_source_read_bytes: 0,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
            },
        )
        .unwrap();
        let inner_sign = PlaneReceipt::new(19, rows, rows * INNER_SIGN_BYTES).unwrap();
        let magnitude = PlaneReceipt::new(23, rows, rows * MAGNITUDE_BYTES).unwrap();
        ValidatedAddressMajorCarrier::publish(
            counts,
            inner_sign,
            magnitude,
            measured_schedule(),
            ScatterPublication {
                scattered_rows: rows,
                cursor_updates: rows,
                completed_outer_blocks: shape.outer_length().unwrap(),
                invalid_rows: 0,
                reserved: [0; 3],
                producer_resident_scans: 1,
                member_resident_scans: 0,
                source_requested_bytes: 16 * rows as u64,
                compact_write_bytes: 12 * rows as u64,
                cell_write_bytes: 4 * cells_count as u64,
                member_source_read_bytes: 0,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
                first_push_pc: 0,
                producer_incremental_wall_ns: Some(4_500_000),
                producer_gpu_active_ns: Some(4_000_000),
            },
        )
        .unwrap()
    }

    #[test]
    fn packed_words_preserve_boundary_values() {
        let cell = PackedCell::new(32_768, 32_768).unwrap();
        assert_eq!(cell.start(), 32_768);
        assert_eq!(cell.count(), 32_768);
        let inner = PackedInnerSign::new(32_767, true).unwrap();
        assert_eq!(inner.inner(), 32_767);
        assert!(inner.negative());
        assert_eq!(PackedInnerSign::from_word(inner.word()), Ok(inner));
        assert!(PackedInnerSign::from_word(1 << 30).is_err());
    }

    #[test]
    fn producer_counts_reject_a_standalone_member_scan() {
        let shape = AddressMajorShape::LOG26;
        let rows = shape.rows().unwrap();
        let cells_count = shape.cells().unwrap();
        let producer = ProducerIdentity::new(7, 11, rows * RESIDENT_ROW_BYTES, 13, rows).unwrap();
        let cells = PlaneReceipt::new(17, cells_count, cells_count * CELL_BYTES).unwrap();
        let result = ValidatedProducerCounts::publish(
            shape,
            producer,
            cells,
            CountPublication {
                initialized_cells: cells_count,
                count_updates: rows,
                counted_rows: rows,
                completed_outer_blocks: shape.outer_length().unwrap(),
                invalid_rows: 0,
                reserved: [0; 3],
                additional_source_scans: 1,
                member_source_read_bytes: (8 * rows) as u64,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
            },
        );
        assert!(matches!(
            result,
            Err(CarrierError::ForbiddenMemberTraffic { scans: 1, .. })
        ));
    }

    #[test]
    fn ready_receipt_binds_every_consumer_allocation() {
        let carrier = ready_carrier();
        let producer = carrier.producer();
        let binding = ConsumerBinding {
            device_registry_id: producer.device_registry_id(),
            source_allocation_identity: producer.source_allocation_identity(),
            source_allocation_bytes: producer.source_allocation_bytes(),
            generation: producer.generation(),
            cells: carrier.cells(),
            inner_sign: carrier.inner_sign(),
            magnitude: carrier.magnitude(),
        };
        carrier.validate_consumer(binding).unwrap();
        let changed = ConsumerBinding {
            generation: binding.generation + 1,
            ..binding
        };
        assert_eq!(
            carrier.validate_consumer(changed),
            Err(CarrierError::ConsumerMismatch("source rows"))
        );
    }

    #[test]
    fn measured_schedule_is_valid_but_unpadded_counts_are_not() {
        measured_schedule()
            .validate(AddressMajorShape::LOG26)
            .unwrap();
        let invalid = TopologyScheduleReceipt {
            padded_long_lanes: 67_107_625,
            ..measured_schedule()
        };
        assert_eq!(
            invalid.validate(AddressMajorShape::LOG26),
            Err(CarrierError::InvalidTopology)
        );
    }
}
