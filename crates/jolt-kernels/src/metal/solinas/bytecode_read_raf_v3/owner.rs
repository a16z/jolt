use thiserror::Error;

pub const BYTECODE_READ_RAF_SCHEMA_VERSION: u32 = 3;

pub(crate) const SHARED_ROW_BYTES: u128 = 40;
pub(crate) const CELL_BYTES: u128 = 4;
pub(crate) const INNER_SIGN_BYTES: u128 = 4;
pub(crate) const MAGNITUDE_BYTES: u128 = 8;

const MAX_INNER_LOG2: usize = 15;
const INNER_MASK: u32 = (1 << MAX_INNER_LOG2) - 1;
const SIGN_BIT: u32 = 1 << 31;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerIdentity {
    device_registry_id: u64,
    source_allocation_id: u64,
    generation: u64,
}

impl ProducerIdentity {
    pub fn new(
        device_registry_id: u64,
        source_allocation_id: u64,
        generation: u64,
    ) -> Result<Self, OwnerError> {
        if device_registry_id == 0 {
            return Err(OwnerError::ZeroIdentity("device registry"));
        }
        if source_allocation_id == 0 {
            return Err(OwnerError::ZeroIdentity("source allocation"));
        }
        if generation == 0 {
            return Err(OwnerError::ZeroIdentity("source generation"));
        }
        Ok(Self {
            device_registry_id,
            source_allocation_id,
            generation,
        })
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn source_allocation_id(self) -> u64 {
        self.source_allocation_id
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentPlaneIdentities {
    cells: u64,
    inner_sign: u64,
    magnitude: u64,
}

impl ResidentPlaneIdentities {
    pub fn new(cells: u64, inner_sign: u64, magnitude: u64) -> Result<Self, OwnerError> {
        if cells == 0 {
            return Err(OwnerError::ZeroIdentity("cell plane"));
        }
        if inner_sign == 0 {
            return Err(OwnerError::ZeroIdentity("inner/sign plane"));
        }
        if magnitude == 0 {
            return Err(OwnerError::ZeroIdentity("magnitude plane"));
        }
        if cells == inner_sign || cells == magnitude || inner_sign == magnitude {
            return Err(OwnerError::AliasedResidentPlane);
        }
        Ok(Self {
            cells,
            inner_sign,
            magnitude,
        })
    }

    pub const fn cells(self) -> u64 {
        self.cells
    }

    pub const fn inner_sign(self) -> u64 {
        self.inner_sign
    }

    pub const fn magnitude(self) -> u64 {
        self.magnitude
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnerConfig {
    log_t: usize,
    log_k: usize,
    inner_log2: usize,
    producer: ProducerIdentity,
    planes: ResidentPlaneIdentities,
}

impl OwnerConfig {
    pub fn new(
        log_t: usize,
        log_k: usize,
        inner_log2: usize,
        producer: ProducerIdentity,
        planes: ResidentPlaneIdentities,
    ) -> Result<Self, OwnerError> {
        if !(2..=u32::BITS as usize).contains(&log_t) {
            return Err(OwnerError::InvalidLogT(log_t));
        }
        if !(9..=2 * super::relation::COMMITTED_CHUNK_BITS).contains(&log_k) {
            return Err(OwnerError::InvalidLogK(log_k));
        }
        if inner_log2 == 0 || inner_log2 > log_t || inner_log2 > MAX_INNER_LOG2 {
            return Err(OwnerError::InvalidInnerLog2 { inner_log2, log_t });
        }
        for (name, plane) in [
            ("cell plane", planes.cells),
            ("inner/sign plane", planes.inner_sign),
            ("magnitude plane", planes.magnitude),
        ] {
            if plane == producer.source_allocation_id {
                return Err(OwnerError::AliasedSourcePlane(name));
            }
        }
        let _ = domain_size(log_t)?;
        let _ = domain_size(log_k)?;
        Ok(Self {
            log_t,
            log_k,
            inner_log2,
            producer,
            planes,
        })
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn inner_log2(self) -> usize {
        self.inner_log2
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub const fn planes(self) -> ResidentPlaneIdentities {
        self.planes
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SignedMagnitude {
    magnitude: u64,
    negative: bool,
}

impl SignedMagnitude {
    pub fn new(magnitude: u64, negative: bool) -> Result<Self, OwnerError> {
        if magnitude == 0 && negative {
            return Err(OwnerError::NegativeZeroIncrement);
        }
        Ok(Self {
            magnitude,
            negative,
        })
    }

    pub const fn zero() -> Self {
        Self {
            magnitude: 0,
            negative: false,
        }
    }

    pub fn from_i64(value: i64) -> Self {
        Self {
            magnitude: value.unsigned_abs(),
            negative: value.is_negative(),
        }
    }

    pub const fn magnitude(self) -> u64 {
        self.magnitude
    }

    pub const fn negative(self) -> bool {
        self.negative
    }

    pub fn field<F: jolt_field::Field>(self) -> F {
        let value = F::from_u64(self.magnitude);
        if self.negative {
            -value
        } else {
            value
        }
    }
}

/// Logical row decoded from the stage-5 40-byte resident allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeWitnessRow {
    mapped_pc_plus_one: u32,
    fused_increment: SignedMagnitude,
}

impl BytecodeWitnessRow {
    pub const fn cold(fused_increment: SignedMagnitude) -> Self {
        Self {
            mapped_pc_plus_one: 0,
            fused_increment,
        }
    }

    pub const fn hot(mapped_pc: u16, fused_increment: SignedMagnitude) -> Self {
        Self {
            mapped_pc_plus_one: mapped_pc as u32 + 1,
            fused_increment,
        }
    }

    pub const fn mapped_pc(self) -> Option<usize> {
        if self.mapped_pc_plus_one == 0 {
            None
        } else {
            Some((self.mapped_pc_plus_one - 1) as usize)
        }
    }

    /// Synthetic unmapped rows use padding slot zero in both protocol phases.
    pub const fn push_pc(self) -> usize {
        match self.mapped_pc() {
            Some(pc) => pc,
            None => 0,
        }
    }

    pub const fn fused_increment(self) -> SignedMagnitude {
        self.fused_increment
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedCell(u32);

impl PackedCell {
    fn new(start: usize, count: usize) -> Result<Self, OwnerError> {
        let start = u16::try_from(start).map_err(|_| OwnerError::PackedCellOverflow)?;
        let count = u16::try_from(count).map_err(|_| OwnerError::PackedCellOverflow)?;
        Ok(Self(u32::from(start) | (u32::from(count) << 16)))
    }

    pub const fn start(self) -> usize {
        (self.0 & 0xffff) as usize
    }

    pub const fn count(self) -> usize {
        (self.0 >> 16) as usize
    }

    pub const fn words(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedInnerSign(u32);

impl PackedInnerSign {
    fn new(inner: usize, negative: bool) -> Result<Self, OwnerError> {
        let inner = u32::try_from(inner).map_err(|_| OwnerError::PackedInnerOverflow)?;
        if inner > INNER_MASK {
            return Err(OwnerError::PackedInnerOverflow);
        }
        Ok(Self(inner | if negative { SIGN_BIT } else { 0 }))
    }

    pub const fn inner(self) -> usize {
        (self.0 & INNER_MASK) as usize
    }

    pub const fn negative(self) -> bool {
        self.0 & SIGN_BIT != 0
    }

    pub const fn words(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackedOccurrence {
    inner_sign: PackedInnerSign,
    magnitude: u64,
}

pub struct BytecodeReadRafOwnerBuilder {
    config: OwnerConfig,
    cycles: usize,
    addresses: usize,
    inner_length: usize,
    outer_length: usize,
    next_cycle: usize,
    rows: Vec<BytecodeWitnessRow>,
    buckets: Vec<Vec<PackedOccurrence>>,
    cells: Vec<PackedCell>,
    inner_sign: Vec<PackedInnerSign>,
    magnitude: Vec<u64>,
    hot_rows: usize,
    nonempty_cells: usize,
}

impl BytecodeReadRafOwnerBuilder {
    pub fn new(config: OwnerConfig) -> Result<Self, OwnerError> {
        let cycles = domain_size(config.log_t)?;
        let addresses = domain_size(config.log_k)?;
        let inner_length = domain_size(config.inner_log2)?;
        let outer_length = cycles / inner_length;
        let cell_capacity = outer_length
            .checked_mul(addresses)
            .ok_or(OwnerError::Overflow)?;
        Ok(Self {
            config,
            cycles,
            addresses,
            inner_length,
            outer_length,
            next_cycle: 0,
            rows: Vec::with_capacity(cycles),
            buckets: vec![Vec::new(); addresses],
            cells: Vec::with_capacity(cell_capacity),
            inner_sign: Vec::with_capacity(cycles),
            magnitude: Vec::with_capacity(cycles),
            hot_rows: 0,
            nonempty_cells: 0,
        })
    }

    pub fn push_cycle(&mut self, row: BytecodeWitnessRow) -> Result<(), OwnerError> {
        if self.next_cycle >= self.cycles {
            return Err(OwnerError::TooManyRows {
                expected: self.cycles,
            });
        }
        let push_pc = row.push_pc();
        if push_pc >= self.addresses {
            return Err(OwnerError::MappedPcOutOfRange {
                cycle: self.next_cycle,
                mapped_pc: push_pc,
                addresses: self.addresses,
            });
        }
        let inner = self.next_cycle % self.inner_length;
        let increment = row.fused_increment();
        let occurrence = PackedOccurrence {
            inner_sign: PackedInnerSign::new(inner, increment.negative())?,
            magnitude: increment.magnitude(),
        };

        self.rows.push(row);
        self.buckets[push_pc].push(occurrence);
        if row.mapped_pc().is_some() {
            self.hot_rows = self.hot_rows.checked_add(1).ok_or(OwnerError::Overflow)?;
        }
        self.next_cycle = self.next_cycle.checked_add(1).ok_or(OwnerError::Overflow)?;
        if self.next_cycle.is_multiple_of(self.inner_length) {
            self.flush_outer_block()?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<BytecodeReadRafOwner, OwnerError> {
        if self.next_cycle != self.cycles {
            return Err(OwnerError::RowCount {
                expected: self.cycles,
                got: self.next_cycle,
            });
        }
        let expected_cells = self
            .outer_length
            .checked_mul(self.addresses)
            .ok_or(OwnerError::Overflow)?;
        if self.cells.len() != expected_cells
            || self.inner_sign.len() != self.cycles
            || self.magnitude.len() != self.cycles
            || self.buckets.iter().any(|bucket| !bucket.is_empty())
        {
            return Err(OwnerError::InvalidTopology);
        }
        let row_digest = row_digest(&self.rows);
        let carrier_digest = carrier_digest(&self.cells, &self.inner_sign, &self.magnitude);
        let receipt = BytecodeReadRafReceipt {
            schema_version: BYTECODE_READ_RAF_SCHEMA_VERSION,
            config: self.config,
            cycles: self.cycles,
            addresses: self.addresses,
            inner_length: self.inner_length,
            outer_length: self.outer_length,
            hot_rows: self.hot_rows,
            cold_rows: self.cycles - self.hot_rows,
            nonempty_cells: self.nonempty_cells,
            row_digest,
            carrier_digest,
        };
        let owner = BytecodeReadRafOwner {
            rows: self.rows,
            cells: self.cells,
            inner_sign: self.inner_sign,
            magnitude: self.magnitude,
            receipt,
        };
        owner.verify_integrity()?;
        Ok(owner)
    }

    fn flush_outer_block(&mut self) -> Result<(), OwnerError> {
        let mut start = 0usize;
        for address in 0..self.addresses {
            let count = self.buckets[address].len();
            self.cells.push(PackedCell::new(start, count)?);
            if count != 0 {
                self.nonempty_cells = self
                    .nonempty_cells
                    .checked_add(1)
                    .ok_or(OwnerError::Overflow)?;
            }
            start = start.checked_add(count).ok_or(OwnerError::Overflow)?;
            for occurrence in self.buckets[address].drain(..) {
                self.inner_sign.push(occurrence.inner_sign);
                self.magnitude.push(occurrence.magnitude);
            }
        }
        if start != self.inner_length {
            return Err(OwnerError::InvalidTopology);
        }
        Ok(())
    }
}

pub struct BytecodeReadRafOwner {
    rows: Vec<BytecodeWitnessRow>,
    cells: Vec<PackedCell>,
    inner_sign: Vec<PackedInnerSign>,
    magnitude: Vec<u64>,
    receipt: BytecodeReadRafReceipt,
}

impl BytecodeReadRafOwner {
    pub const fn receipt(&self) -> BytecodeReadRafReceipt {
        self.receipt
    }

    pub fn rows(&self) -> &[BytecodeWitnessRow] {
        &self.rows
    }

    pub fn cells(&self) -> &[PackedCell] {
        &self.cells
    }

    pub fn inner_sign(&self) -> &[PackedInnerSign] {
        &self.inner_sign
    }

    pub fn magnitudes(&self) -> &[u64] {
        &self.magnitude
    }

    pub fn cell(&self, outer: usize, address: usize) -> Result<PackedCell, OwnerError> {
        if outer >= self.receipt.outer_length || address >= self.receipt.addresses {
            return Err(OwnerError::InvalidCell { outer, address });
        }
        let index = outer
            .checked_mul(self.receipt.addresses)
            .and_then(|base| base.checked_add(address))
            .ok_or(OwnerError::Overflow)?;
        self.cells
            .get(index)
            .copied()
            .ok_or(OwnerError::InvalidTopology)
    }

    pub fn occurrences(
        &self,
        outer: usize,
        address: usize,
    ) -> Result<(&[PackedInnerSign], &[u64]), OwnerError> {
        let cell = self.cell(outer, address)?;
        let base = outer
            .checked_mul(self.receipt.inner_length)
            .and_then(|value| value.checked_add(cell.start()))
            .ok_or(OwnerError::Overflow)?;
        let end = base.checked_add(cell.count()).ok_or(OwnerError::Overflow)?;
        let inner_sign = self
            .inner_sign
            .get(base..end)
            .ok_or(OwnerError::InvalidTopology)?;
        let magnitude = self
            .magnitude
            .get(base..end)
            .ok_or(OwnerError::InvalidTopology)?;
        Ok((inner_sign, magnitude))
    }

    pub fn verify_integrity(&self) -> Result<(), OwnerError> {
        let receipt = self.receipt;
        let expected_cells = receipt
            .outer_length
            .checked_mul(receipt.addresses)
            .ok_or(OwnerError::Overflow)?;
        if receipt.schema_version != BYTECODE_READ_RAF_SCHEMA_VERSION
            || self.rows.len() != receipt.cycles
            || self.cells.len() != expected_cells
            || self.inner_sign.len() != receipt.cycles
            || self.magnitude.len() != receipt.cycles
            || row_digest(&self.rows) != receipt.row_digest
            || carrier_digest(&self.cells, &self.inner_sign, &self.magnitude)
                != receipt.carrier_digest
        {
            return Err(OwnerError::ReceiptMismatch);
        }

        let mut hot_rows = 0usize;
        let mut nonempty_cells = 0usize;
        let mut seen = vec![false; receipt.inner_length];
        for outer in 0..receipt.outer_length {
            seen.fill(false);
            let mut expected_start = 0usize;
            for address in 0..receipt.addresses {
                let cell = self.cell(outer, address)?;
                if cell.start() != expected_start {
                    return Err(OwnerError::InvalidTopology);
                }
                if cell.count() != 0 {
                    nonempty_cells = nonempty_cells.checked_add(1).ok_or(OwnerError::Overflow)?;
                }
                let (inner_sign, magnitudes) = self.occurrences(outer, address)?;
                for (&packed, &magnitude) in inner_sign.iter().zip(magnitudes) {
                    let inner = packed.inner();
                    if inner >= receipt.inner_length || seen[inner] {
                        return Err(OwnerError::InvalidTopology);
                    }
                    seen[inner] = true;
                    let cycle = outer
                        .checked_mul(receipt.inner_length)
                        .and_then(|base| base.checked_add(inner))
                        .ok_or(OwnerError::Overflow)?;
                    let row = self
                        .rows
                        .get(cycle)
                        .copied()
                        .ok_or(OwnerError::InvalidTopology)?;
                    if row.push_pc() != address
                        || row.fused_increment().magnitude() != magnitude
                        || row.fused_increment().negative() != packed.negative()
                    {
                        return Err(OwnerError::InvalidTopology);
                    }
                }
                expected_start = expected_start
                    .checked_add(cell.count())
                    .ok_or(OwnerError::Overflow)?;
            }
            if expected_start != receipt.inner_length || seen.iter().any(|seen| !seen) {
                return Err(OwnerError::InvalidTopology);
            }
        }
        for row in &self.rows {
            if row.mapped_pc().is_some() {
                hot_rows = hot_rows.checked_add(1).ok_or(OwnerError::Overflow)?;
            }
            if row.push_pc() >= receipt.addresses {
                return Err(OwnerError::InvalidTopology);
            }
        }
        if hot_rows != receipt.hot_rows
            || receipt.cold_rows != receipt.cycles - hot_rows
            || nonempty_cells != receipt.nonempty_cells
        {
            return Err(OwnerError::ReceiptMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafReceipt {
    schema_version: u32,
    config: OwnerConfig,
    cycles: usize,
    addresses: usize,
    inner_length: usize,
    outer_length: usize,
    hot_rows: usize,
    cold_rows: usize,
    nonempty_cells: usize,
    row_digest: u64,
    carrier_digest: u64,
}

impl BytecodeReadRafReceipt {
    pub const fn schema_version(self) -> u32 {
        self.schema_version
    }

    pub const fn config(self) -> OwnerConfig {
        self.config
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.config.producer
    }

    pub const fn planes(self) -> ResidentPlaneIdentities {
        self.config.planes
    }

    pub const fn log_t(self) -> usize {
        self.config.log_t
    }

    pub const fn log_k(self) -> usize {
        self.config.log_k
    }

    pub const fn inner_log2(self) -> usize {
        self.config.inner_log2
    }

    pub const fn cycles(self) -> usize {
        self.cycles
    }

    pub const fn addresses(self) -> usize {
        self.addresses
    }

    pub const fn inner_length(self) -> usize {
        self.inner_length
    }

    pub const fn outer_length(self) -> usize {
        self.outer_length
    }

    pub const fn hot_rows(self) -> usize {
        self.hot_rows
    }

    pub const fn cold_rows(self) -> usize {
        self.cold_rows
    }

    pub const fn nonempty_cells(self) -> usize {
        self.nonempty_cells
    }

    /// The producer visits each authoritative witness row exactly once.
    pub const fn source_traversals(self) -> usize {
        1
    }

    pub const fn marginal_row_upload_bytes(self) -> usize {
        0
    }

    pub const fn row_digest(self) -> u64 {
        self.row_digest
    }

    pub const fn carrier_digest(self) -> u64 {
        self.carrier_digest
    }
}

fn domain_size(log_size: usize) -> Result<usize, OwnerError> {
    let shift = u32::try_from(log_size).map_err(|_| OwnerError::Overflow)?;
    1usize.checked_shl(shift).ok_or(OwnerError::Overflow)
}

fn row_digest(rows: &[BytecodeWitnessRow]) -> u64 {
    rows.iter().fold(FNV_OFFSET, |digest, row| {
        let mapped = row.mapped_pc().map_or(0, |pc| pc as u64 + 1);
        let digest = mix(digest, mapped);
        let digest = mix(digest, row.fused_increment().magnitude());
        mix(digest, u64::from(row.fused_increment().negative()))
    })
}

fn carrier_digest(cells: &[PackedCell], inner_sign: &[PackedInnerSign], magnitude: &[u64]) -> u64 {
    let digest = cells.iter().fold(FNV_OFFSET, |digest, cell| {
        mix(digest, u64::from(cell.words()))
    });
    let digest = inner_sign.iter().fold(digest, |digest, value| {
        mix(digest, u64::from(value.words()))
    });
    magnitude
        .iter()
        .fold(digest, |digest, &value| mix(digest, value))
}

fn mix(mut digest: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    digest
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum OwnerError {
    #[error("bytecode read/RAF {0} identity is zero")]
    ZeroIdentity(&'static str),
    #[error("bytecode read/RAF resident planes alias")]
    AliasedResidentPlane,
    #[error("bytecode read/RAF {0} aliases the source row allocation")]
    AliasedSourcePlane(&'static str),
    #[error("bytecode read/RAF log_T {0} is unsupported")]
    InvalidLogT(usize),
    #[error("bytecode read/RAF log_K {0} does not fit two 8-bit RA chunks")]
    InvalidLogK(usize),
    #[error("bytecode read/RAF inner split {inner_log2} is invalid for log_T {log_t}")]
    InvalidInnerLog2 { inner_log2: usize, log_t: usize },
    #[error("bytecode read/RAF signed magnitude used negative zero")]
    NegativeZeroIncrement,
    #[error("bytecode read/RAF received more than {expected} rows")]
    TooManyRows { expected: usize },
    #[error("bytecode read/RAF expected {expected} rows, got {got}")]
    RowCount { expected: usize, got: usize },
    #[error("bytecode row {cycle} maps PC {mapped_pc} outside {addresses} addresses")]
    MappedPcOutOfRange {
        cycle: usize,
        mapped_pc: usize,
        addresses: usize,
    },
    #[error("bytecode read/RAF packed cell exceeds its u16 fields")]
    PackedCellOverflow,
    #[error("bytecode read/RAF packed inner index exceeds 15 bits")]
    PackedInnerOverflow,
    #[error("bytecode read/RAF cell ({outer}, {address}) is outside the carrier")]
    InvalidCell { outer: usize, address: usize },
    #[error("bytecode read/RAF carrier topology is invalid")]
    InvalidTopology,
    #[error("bytecode read/RAF receipt does not match its private planes")]
    ReceiptMismatch,
    #[error("bytecode read/RAF owner arithmetic overflowed")]
    Overflow,
}
