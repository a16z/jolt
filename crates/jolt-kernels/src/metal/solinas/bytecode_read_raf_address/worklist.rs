use core::{fmt, mem::size_of, ops::Range};

#[cfg(test)]
use jolt_field::Field;

use super::carrier::{AddressMajorShape, CarrierError};
#[cfg(test)]
use super::oracle::Row;

pub(crate) const BYTECODE_ADDRESS_WORK_ITEM_ROWS: usize = 4096;
pub(crate) const BYTECODE_ADDRESS_PUSHFORWARD_STAGES: usize = 9;
pub(crate) const BYTECODE_ADDRESS_BASE_STAGES: usize = 5;

const INNER_SIGN_BIT: u16 = 1 << 15;
const INNER_MASK: u16 = INNER_SIGN_BIT - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseAddressRow {
    selector: u16,
    magnitude: u64,
}

impl SparseAddressRow {
    #[cfg(test)]
    pub(crate) fn new(
        mapped_pc: Option<usize>,
        fused_inc_negative: bool,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        Self::with_magnitude(mapped_pc, 0, fused_inc_negative)
    }

    pub(crate) fn with_magnitude(
        mapped_pc: Option<usize>,
        magnitude: u64,
        fused_inc_negative: bool,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        let address = mapped_pc.unwrap_or(0);
        let address = u16::try_from(address).map_err(|_| {
            BytecodeAddressWorklistError::UnsupportedAddresses(address.saturating_add(1))
        })?;
        if address & INNER_SIGN_BIT != 0 {
            return Err(BytecodeAddressWorklistError::UnsupportedAddresses(
                usize::from(address) + 1,
            ));
        }
        Ok(Self {
            selector: address
                | if fused_inc_negative {
                    INNER_SIGN_BIT
                } else {
                    0
                },
            magnitude,
        })
    }

    pub(crate) const fn address(self) -> usize {
        (self.selector & INNER_MASK) as usize
    }

    pub(crate) const fn negative(self) -> bool {
        self.selector & INNER_SIGN_BIT != 0
    }

    pub(crate) const fn magnitude(self) -> u64 {
        self.magnitude
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PackedBytecodeAddressOccurrence(u16);

impl PackedBytecodeAddressOccurrence {
    pub(crate) fn new(inner: usize, negative: bool) -> Result<Self, BytecodeAddressWorklistError> {
        let inner =
            u16::try_from(inner).map_err(|_| BytecodeAddressWorklistError::InvalidInner(inner))?;
        if inner & INNER_SIGN_BIT != 0 {
            return Err(BytecodeAddressWorklistError::InvalidInner(usize::from(
                inner,
            )));
        }
        Ok(Self(inner | if negative { INNER_SIGN_BIT } else { 0 }))
    }

    #[cfg(test)]
    pub(crate) const fn inner(self) -> usize {
        (self.0 & INNER_MASK) as usize
    }

    #[cfg(test)]
    pub(crate) const fn negative(self) -> bool {
        self.0 & INNER_SIGN_BIT != 0
    }

    #[cfg(test)]
    pub(crate) const fn word(self) -> u16 {
        self.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressWorkItem {
    pub(crate) address: u16,
    pub(crate) outer: u16,
    pub(crate) start: u16,
    pub(crate) count: u16,
}

const _: [(); 8] = [(); size_of::<BytecodeAddressWorkItem>()];

impl BytecodeAddressWorkItem {
    fn new(
        address: usize,
        outer: usize,
        start: usize,
        count: usize,
        outer_rows: usize,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        if count == 0
            || count > BYTECODE_ADDRESS_WORK_ITEM_ROWS
            || start.checked_add(count).is_none_or(|end| end > outer_rows)
        {
            return Err(BytecodeAddressWorklistError::InvalidWorkItem);
        }
        Ok(Self {
            address: u16::try_from(address).map_err(|_| {
                BytecodeAddressWorklistError::UnsupportedAddresses(address.saturating_add(1))
            })?,
            outer: u16::try_from(outer).map_err(|_| {
                BytecodeAddressWorklistError::UnsupportedOuters(outer.saturating_add(1))
            })?,
            start: u16::try_from(start)
                .map_err(|_| BytecodeAddressWorklistError::InvalidWorkItem)?,
            count: u16::try_from(count)
                .map_err(|_| BytecodeAddressWorklistError::InvalidWorkItem)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressWorklistLedger {
    physical_rows: usize,
    work_items: usize,
    occurrence_bytes: usize,
    magnitude_bytes: usize,
    work_item_bytes: usize,
    descriptor_offset_bytes: usize,
    persistent_bytes: usize,
}

impl BytecodeAddressWorklistLedger {
    #[cfg(test)]
    pub(crate) const fn physical_rows(self) -> usize {
        self.physical_rows
    }

    #[cfg(test)]
    pub(crate) const fn work_items(self) -> usize {
        self.work_items
    }

    pub(crate) const fn occurrence_bytes(self) -> usize {
        self.occurrence_bytes
    }

    pub(crate) const fn magnitude_bytes(self) -> usize {
        self.magnitude_bytes
    }

    pub(crate) const fn work_item_bytes(self) -> usize {
        self.work_item_bytes
    }

    pub(crate) const fn descriptor_offset_bytes(self) -> usize {
        self.descriptor_offset_bytes
    }

    #[cfg(test)]
    pub(crate) const fn persistent_bytes(self) -> usize {
        self.persistent_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressSparseWorklist {
    shape: AddressMajorShape,
    padded_rows: usize,
    occurrences: Vec<PackedBytecodeAddressOccurrence>,
    magnitudes: Vec<u64>,
    work_items: Vec<BytecodeAddressWorkItem>,
    descriptor_offsets: Vec<u32>,
    ledger: BytecodeAddressWorklistLedger,
}

#[cfg(test)]
pub(crate) type SparseAddressWorkItem = BytecodeAddressWorkItem;
pub(crate) type SparseAddressWorklist = BytecodeAddressSparseWorklist;
pub(crate) type SparseAddressWorklistError = BytecodeAddressWorklistError;

impl BytecodeAddressSparseWorklist {
    #[cfg(test)]
    pub(crate) fn build(
        rows: &[SparseAddressRow],
        padded_rows: usize,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        if !padded_rows.is_power_of_two() {
            return Err(BytecodeAddressWorklistError::InvalidPaddedRows(padded_rows));
        }
        let shape = AddressMajorShape::production(padded_rows.trailing_zeros())?;
        Self::build_with_shape(rows, shape)
    }

    #[cfg(test)]
    fn build_with_shape(
        rows: &[SparseAddressRow],
        shape: AddressMajorShape,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        Self::build_with(rows.len(), shape, |index| rows[index])
    }

    /// Calls `row_at` once for each physical row. One outer block of packed
    /// selectors is retained for the stable counting scatter.
    #[cfg(test)]
    pub(crate) fn build_with(
        physical_rows: usize,
        shape: AddressMajorShape,
        mut row_at: impl FnMut(usize) -> SparseAddressRow,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        Self::try_build_with(physical_rows, shape, |index| Ok(row_at(index)))
    }

    pub(crate) fn try_build_with(
        physical_rows: usize,
        shape: AddressMajorShape,
        mut row_at: impl FnMut(usize) -> Result<SparseAddressRow, BytecodeAddressWorklistError>,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        let domain_rows = shape.rows()?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        if physical_rows > domain_rows {
            return Err(BytecodeAddressWorklistError::PhysicalRows {
                physical: physical_rows,
                available: physical_rows,
                domain: domain_rows,
            });
        }
        if addresses > usize::from(INNER_SIGN_BIT) {
            return Err(BytecodeAddressWorklistError::UnsupportedAddresses(
                addresses,
            ));
        }
        let physical_outers = physical_rows.div_ceil(inner_length);
        if physical_outers > usize::from(u16::MAX) + 1 {
            return Err(BytecodeAddressWorklistError::UnsupportedOuters(
                physical_outers,
            ));
        }

        let mut occurrences = vec![PackedBytecodeAddressOccurrence::default(); physical_rows];
        let mut magnitudes = vec![0u64; physical_rows];
        let mut work_items = Vec::new();
        let mut counts = vec![0u16; addresses];
        let mut outer_scratch = Vec::with_capacity(inner_length);

        for outer in 0..physical_outers {
            let base = checked_mul("outer row base", outer, inner_length)?;
            let outer_rows = (physical_rows - base).min(inner_length);
            counts.fill(0);
            outer_scratch.clear();
            for inner in 0..outer_rows {
                let row = row_at(base + inner)?;
                let address = row.address();
                if address >= addresses {
                    return Err(BytecodeAddressWorklistError::InvalidPc {
                        row: base + inner,
                        pc: address,
                        addresses,
                    });
                }
                counts[address] = counts[address]
                    .checked_add(1)
                    .ok_or(BytecodeAddressWorklistError::Overflow("address count"))?;
                outer_scratch.push(row);
            }

            let mut start = 0usize;
            for (address, count_or_cursor) in counts.iter_mut().enumerate() {
                let count = usize::from(*count_or_cursor);
                *count_or_cursor = u16::try_from(start)
                    .map_err(|_| BytecodeAddressWorklistError::InvalidWorkItem)?;
                let mut chunk_start = start;
                let mut remaining = count;
                while remaining != 0 {
                    let chunk = remaining.min(BYTECODE_ADDRESS_WORK_ITEM_ROWS);
                    work_items.push(BytecodeAddressWorkItem::new(
                        address,
                        outer,
                        chunk_start,
                        chunk,
                        outer_rows,
                    )?);
                    chunk_start = chunk_start
                        .checked_add(chunk)
                        .ok_or(BytecodeAddressWorklistError::Overflow("work item start"))?;
                    remaining -= chunk;
                }
                start = start
                    .checked_add(count)
                    .ok_or(BytecodeAddressWorklistError::Overflow("outer occurrences"))?;
            }
            if start != outer_rows {
                return Err(BytecodeAddressWorklistError::InvalidWorkItem);
            }

            for (inner, row) in outer_scratch.iter().enumerate() {
                let address = row.address();
                let destination = base.checked_add(usize::from(counts[address])).ok_or(
                    BytecodeAddressWorklistError::Overflow("occurrence destination"),
                )?;
                counts[address] = counts[address]
                    .checked_add(1)
                    .ok_or(BytecodeAddressWorklistError::Overflow("address cursor"))?;
                occurrences[destination] =
                    PackedBytecodeAddressOccurrence::new(inner, row.negative())?;
                magnitudes[destination] = row.magnitude();
            }
        }

        // Stable sorting retains outer and chunk order within each address.
        work_items.sort_by_key(|item| item.address);
        let work_item_count = u32::try_from(work_items.len())
            .map_err(|_| BytecodeAddressWorklistError::Overflow("work item count"))?;
        let mut descriptor_offsets = vec![0u32; addresses + 1];
        let mut work_index = 0usize;
        for (address, offset) in descriptor_offsets[..addresses].iter_mut().enumerate() {
            *offset = u32::try_from(work_index)
                .map_err(|_| BytecodeAddressWorklistError::Overflow("descriptor offset"))?;
            while work_items
                .get(work_index)
                .is_some_and(|item| usize::from(item.address) == address)
            {
                work_index += 1;
            }
        }
        descriptor_offsets[addresses] = work_item_count;
        if work_index != work_items.len() {
            return Err(BytecodeAddressWorklistError::InvalidWorkItem);
        }

        let ledger = worklist_ledger(physical_rows, work_items.len(), descriptor_offsets.len())?;
        let worklist = Self {
            shape,
            padded_rows: domain_rows,
            occurrences,
            magnitudes,
            work_items,
            descriptor_offsets,
            ledger,
        };
        worklist.validate()?;
        Ok(worklist)
    }

    pub(crate) const fn shape(&self) -> AddressMajorShape {
        self.shape
    }

    pub(crate) const fn physical_rows(&self) -> usize {
        self.ledger.physical_rows
    }

    pub(crate) const fn padded_rows(&self) -> usize {
        self.padded_rows
    }

    pub(crate) fn occurrences(&self) -> &[PackedBytecodeAddressOccurrence] {
        &self.occurrences
    }

    pub(crate) fn items(&self) -> &[BytecodeAddressWorkItem] {
        &self.work_items
    }

    pub(crate) fn magnitudes(&self) -> &[u64] {
        &self.magnitudes
    }

    pub(crate) fn address_offsets(&self) -> &[u32] {
        &self.descriptor_offsets
    }

    pub(crate) const fn work_items(&self) -> usize {
        self.ledger.work_items
    }

    pub(crate) const fn persistent_bytes(&self) -> usize {
        self.ledger.persistent_bytes
    }

    pub(crate) const fn ledger(&self) -> BytecodeAddressWorklistLedger {
        self.ledger
    }

    pub(crate) fn descriptor_range(&self, address: usize) -> Option<Range<usize>> {
        let end = address.checked_add(2)?;
        let pair = self.descriptor_offsets.get(address..end)?;
        Some(pair[0] as usize..pair[1] as usize)
    }

    fn validate(&self) -> Result<(), BytecodeAddressWorklistError> {
        let addresses = self.shape.addresses()?;
        let inner_length = self.shape.inner_length()?;
        if self.descriptor_offsets.len() != addresses + 1
            || self.descriptor_offsets.first() != Some(&0)
            || self.descriptor_offsets.last().copied() != u32::try_from(self.work_items.len()).ok()
            || self
                .descriptor_offsets
                .windows(2)
                .any(|pair| pair[0] > pair[1])
        {
            return Err(BytecodeAddressWorklistError::InvalidWorkItem);
        }
        let mut covered_rows = 0usize;
        for address in 0..addresses {
            let range = self
                .descriptor_range(address)
                .ok_or(BytecodeAddressWorklistError::InvalidWorkItem)?;
            for item in &self.work_items[range] {
                let outer = usize::from(item.outer);
                let base = checked_mul("validation outer base", outer, inner_length)?;
                let outer_rows = self
                    .occurrences
                    .len()
                    .checked_sub(base)
                    .map(|remaining| remaining.min(inner_length))
                    .ok_or(BytecodeAddressWorklistError::InvalidWorkItem)?;
                let end = usize::from(item.start)
                    .checked_add(usize::from(item.count))
                    .ok_or(BytecodeAddressWorklistError::Overflow("work item end"))?;
                if usize::from(item.address) != address
                    || item.count == 0
                    || usize::from(item.count) > BYTECODE_ADDRESS_WORK_ITEM_ROWS
                    || end > outer_rows
                {
                    return Err(BytecodeAddressWorklistError::InvalidWorkItem);
                }
                covered_rows = covered_rows
                    .checked_add(usize::from(item.count))
                    .ok_or(BytecodeAddressWorklistError::Overflow("covered rows"))?;
            }
        }
        if covered_rows != self.occurrences.len() {
            return Err(BytecodeAddressWorklistError::InvalidWorkItem);
        }
        if self.magnitudes.len() != self.occurrences.len() {
            return Err(BytecodeAddressWorklistError::InvalidWorkItem);
        }
        Ok(())
    }
}

/// Computes the nine physical-prefix pushforwards without using the worklist.
#[cfg(test)]
pub(crate) fn direct_physical_pushforwards<F: Field>(
    rows: &[Row],
    physical_rows: usize,
    shape: AddressMajorShape,
    e_lo: &[Vec<F>],
    e_hi: &[Vec<F>],
) -> Result<Vec<F>, BytecodeAddressWorklistError> {
    let domain_rows = shape.rows()?;
    let addresses = shape.addresses()?;
    let inner_length = shape.inner_length()?;
    let outer_length = shape.outer_length()?;
    if physical_rows > domain_rows || physical_rows > rows.len() {
        return Err(BytecodeAddressWorklistError::PhysicalRows {
            physical: physical_rows,
            available: rows.len(),
            domain: domain_rows,
        });
    }
    if e_lo.len() != BYTECODE_ADDRESS_PUSHFORWARD_STAGES
        || e_hi.len() != BYTECODE_ADDRESS_PUSHFORWARD_STAGES
    {
        return Err(BytecodeAddressWorklistError::EqualityStageCount {
            e_lo: e_lo.len(),
            e_hi: e_hi.len(),
        });
    }
    for stage in 0..BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
        if e_lo[stage].len() != inner_length || e_hi[stage].len() != outer_length {
            return Err(BytecodeAddressWorklistError::EqualityTable {
                stage,
                e_lo: e_lo[stage].len(),
                e_hi: e_hi[stage].len(),
                expected_lo: inner_length,
                expected_hi: outer_length,
            });
        }
    }
    let output_len = checked_mul(
        "pushforward output",
        BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
        addresses,
    )?;
    let mut output = vec![F::zero(); output_len];
    for (index, row) in rows[..physical_rows].iter().copied().enumerate() {
        let address = row.push_pc();
        if address >= addresses {
            return Err(BytecodeAddressWorklistError::InvalidPc {
                row: index,
                pc: address,
                addresses,
            });
        }
        let outer = index / inner_length;
        let inner = index % inner_length;
        let magnitude = F::from_u64(row.fused_inc_magnitude);
        let increment = if row.fused_inc_negative {
            F::zero() - magnitude
        } else {
            magnitude
        };
        for stage in 0..BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
            let mut term = e_lo[stage][inner] * e_hi[stage][outer];
            if stage >= BYTECODE_ADDRESS_BASE_STAGES {
                term *= increment;
            }
            output[stage * addresses + address] += term;
        }
    }
    Ok(output)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BytecodeAddressWorklistError {
    Carrier(CarrierError),
    PhysicalRows {
        physical: usize,
        available: usize,
        domain: usize,
    },
    UnsupportedAddresses(usize),
    UnsupportedOuters(usize),
    #[cfg(test)]
    InvalidPaddedRows(usize),
    InvalidPc {
        row: usize,
        pc: usize,
        addresses: usize,
    },
    InvalidInner(usize),
    InvalidWorkItem,
    #[cfg(test)]
    EqualityStageCount {
        e_lo: usize,
        e_hi: usize,
    },
    #[cfg(test)]
    EqualityTable {
        stage: usize,
        e_lo: usize,
        e_hi: usize,
        expected_lo: usize,
        expected_hi: usize,
    },
    Overflow(&'static str),
}

impl From<CarrierError> for BytecodeAddressWorklistError {
    fn from(error: CarrierError) -> Self {
        Self::Carrier(error)
    }
}

impl fmt::Display for BytecodeAddressWorklistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carrier(error) => error.fmt(f),
            Self::PhysicalRows {
                physical,
                available,
                domain,
            } => write!(
                f,
                "physical row prefix {physical} exceeds source {available} or domain {domain}"
            ),
            Self::UnsupportedAddresses(addresses) => {
                write!(f, "{addresses} addresses do not fit the u16 work-item ABI")
            }
            Self::UnsupportedOuters(outers) => {
                write!(f, "{outers} outers do not fit the u16 work-item ABI")
            }
            #[cfg(test)]
            Self::InvalidPaddedRows(rows) => {
                write!(f, "padded row count {rows} is not a nonzero power of two")
            }
            Self::InvalidPc {
                row,
                pc,
                addresses,
            } => write!(
                f,
                "row {row} maps to address {pc}, outside the {addresses}-address domain"
            ),
            Self::InvalidInner(inner) => {
                write!(f, "inner index {inner} does not fit the 15-bit occurrence ABI")
            }
            Self::InvalidWorkItem => f.write_str("invalid sparse bytecode work-item layout"),
            #[cfg(test)]
            Self::EqualityStageCount { e_lo, e_hi } => write!(
                f,
                "equality tables have {e_lo} low and {e_hi} high stages, expected 9"
            ),
            #[cfg(test)]
            Self::EqualityTable {
                stage,
                e_lo,
                e_hi,
                expected_lo,
                expected_hi,
            } => write!(
                f,
                "stage {stage} equality tables have lengths {e_lo}/{e_hi}, expected {expected_lo}/{expected_hi}"
            ),
            Self::Overflow(name) => write!(f, "{name} overflowed"),
        }
    }
}

impl std::error::Error for BytecodeAddressWorklistError {}

fn worklist_ledger(
    physical_rows: usize,
    work_items: usize,
    descriptor_offsets: usize,
) -> Result<BytecodeAddressWorklistLedger, BytecodeAddressWorklistError> {
    let occurrence_bytes = checked_mul(
        "occurrence bytes",
        physical_rows,
        size_of::<PackedBytecodeAddressOccurrence>(),
    )?;
    let magnitude_bytes = checked_mul("magnitude bytes", physical_rows, size_of::<u64>())?;
    let work_item_bytes = checked_mul(
        "work item bytes",
        work_items,
        size_of::<BytecodeAddressWorkItem>(),
    )?;
    let descriptor_offset_bytes = checked_mul(
        "descriptor offset bytes",
        descriptor_offsets,
        size_of::<u32>(),
    )?;
    let persistent_bytes = occurrence_bytes
        .checked_add(magnitude_bytes)
        .and_then(|bytes| bytes.checked_add(work_item_bytes))
        .and_then(|bytes| bytes.checked_add(descriptor_offset_bytes))
        .ok_or(BytecodeAddressWorklistError::Overflow(
            "persistent worklist bytes",
        ))?;
    Ok(BytecodeAddressWorklistLedger {
        physical_rows,
        work_items,
        occurrence_bytes,
        magnitude_bytes,
        work_item_bytes,
        descriptor_offset_bytes,
        persistent_bytes,
    })
}

fn checked_mul(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, BytecodeAddressWorklistError> {
    left.checked_mul(right)
        .ok_or(BytecodeAddressWorklistError::Overflow(name))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked fixtures")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    fn row(mapped_pc: Option<usize>, negative: bool) -> SparseAddressRow {
        SparseAddressRow::new(mapped_pc, negative).unwrap()
    }

    fn row_with_magnitude(
        mapped_pc: Option<usize>,
        magnitude: u64,
        negative: bool,
    ) -> SparseAddressRow {
        SparseAddressRow::with_magnitude(mapped_pc, magnitude, negative).unwrap()
    }

    #[test]
    fn two_outers_are_grouped_stably_with_empty_address_ranges() {
        let shape = AddressMajorShape::new(3, 3, 2).unwrap();
        let rows = [
            row_with_magnitude(Some(2), 10, false),
            row_with_magnitude(Some(1), 11, true),
            row_with_magnitude(Some(2), 12, true),
            row_with_magnitude(None, 13, false),
            row_with_magnitude(Some(1), 14, true),
            row_with_magnitude(Some(1), 15, false),
            row_with_magnitude(Some(5), 16, true),
        ];
        let worklist = BytecodeAddressSparseWorklist::build_with_shape(&rows, shape).unwrap();

        assert_eq!(
            worklist
                .occurrences()
                .iter()
                .map(|occurrence| occurrence.word())
                .collect::<Vec<_>>(),
            vec![
                3,
                INNER_SIGN_BIT | 1,
                0,
                INNER_SIGN_BIT | 2,
                INNER_SIGN_BIT,
                1,
                INNER_SIGN_BIT | 2,
            ]
        );
        assert_eq!(worklist.magnitudes(), &[13, 11, 10, 12, 14, 15, 16]);
        assert_eq!(
            worklist.items(),
            &[
                BytecodeAddressWorkItem {
                    address: 0,
                    outer: 0,
                    start: 0,
                    count: 1,
                },
                BytecodeAddressWorkItem {
                    address: 1,
                    outer: 0,
                    start: 1,
                    count: 1,
                },
                BytecodeAddressWorkItem {
                    address: 1,
                    outer: 1,
                    start: 0,
                    count: 2,
                },
                BytecodeAddressWorkItem {
                    address: 2,
                    outer: 0,
                    start: 2,
                    count: 2,
                },
                BytecodeAddressWorkItem {
                    address: 5,
                    outer: 1,
                    start: 2,
                    count: 1,
                },
            ]
        );
        assert_eq!(worklist.address_offsets(), &[0, 1, 3, 4, 4, 4, 5, 5, 5]);
        assert_eq!(worklist.descriptor_range(3), Some(4..4));
        assert_eq!(worklist.descriptor_range(8), None);
        assert_eq!(worklist.shape(), shape);
        assert_eq!(worklist.physical_rows(), 7);
        assert_eq!(worklist.padded_rows(), 8);
        assert_eq!(worklist.work_items(), 5);
        assert_eq!(worklist.persistent_bytes(), 146);
        assert!(!worklist.occurrences()[0].negative());
        assert_eq!(worklist.occurrences()[0].inner(), 3);

        let ledger = worklist.ledger();
        assert_eq!(ledger.physical_rows(), 7);
        assert_eq!(ledger.work_items(), 5);
        assert_eq!(ledger.occurrence_bytes(), 14);
        assert_eq!(ledger.magnitude_bytes(), 56);
        assert_eq!(ledger.work_item_bytes(), 40);
        assert_eq!(ledger.descriptor_offset_bytes(), 36);
        assert_eq!(ledger.persistent_bytes(), 146);
    }

    #[test]
    fn cells_split_at_4096_rows() {
        let shape = AddressMajorShape::new(17, 2, 15).unwrap();
        let inner_length = shape.inner_length().unwrap();
        let physical_rows = 3 * inner_length;
        let mut rows = vec![row(Some(3), false); physical_rows];
        rows[..4096].fill(row(Some(0), false));
        rows[inner_length..inner_length + 4097].fill(row(Some(1), true));
        rows[2 * inner_length..3 * inner_length].fill(row(Some(2), false));

        let worklist = BytecodeAddressSparseWorklist::build_with_shape(&rows, shape).unwrap();
        let counts = |address| {
            worklist.items()[worklist.descriptor_range(address).unwrap()]
                .iter()
                .map(|item| item.count)
                .collect::<Vec<_>>()
        };

        assert_eq!(counts(0), vec![4096]);
        assert_eq!(counts(1), vec![4096, 1]);
        assert_eq!(counts(2), vec![4096; 8]);
        assert_eq!(worklist.ledger().physical_rows(), physical_rows);
        assert_eq!(worklist.ledger().work_items(), 25);
    }

    #[test]
    fn closure_builder_reads_each_physical_row_once() {
        let shape = AddressMajorShape::production(15).unwrap();
        let rows = vec![
            row(Some(3), false),
            row(Some(1), true),
            row(None, false),
            row(Some(3), true),
        ];
        let mut calls = vec![0usize; rows.len()];
        let from_closure = SparseAddressWorklist::build_with(rows.len(), shape, |index| {
            calls[index] += 1;
            rows[index]
        })
        .unwrap();
        let from_slice: SparseAddressWorklist =
            SparseAddressWorklist::build(&rows, 1 << 15).unwrap();

        assert_eq!(calls, vec![1; rows.len()]);
        assert_eq!(from_closure, from_slice);
        assert_eq!(size_of::<SparseAddressWorkItem>(), 8);
    }

    #[test]
    fn padding_suffix_is_not_read() {
        let shape = AddressMajorShape::new(2, 1, 1).unwrap();
        let rows = vec![
            row(Some(1), true),
            row(Some(7), false),
            row(Some(7), false),
            row(Some(7), false),
        ];

        let worklist = BytecodeAddressSparseWorklist::build_with_shape(&rows[..1], shape).unwrap();

        assert_eq!(worklist.occurrences().len(), 1);
        assert_eq!(worklist.work_items(), 1);
        assert_eq!(worklist.items()[0].address, 1);
    }

    #[test]
    fn direct_oracle_covers_base_and_signed_fused_stages() {
        let shape = AddressMajorShape::new(2, 1, 1).unwrap();
        let rows = vec![
            Row {
                mapped_pc: Some(0),
                fused_inc_magnitude: 1,
                fused_inc_negative: false,
            },
            Row {
                mapped_pc: Some(1),
                fused_inc_magnitude: 2,
                fused_inc_negative: false,
            },
            Row {
                mapped_pc: Some(1),
                fused_inc_magnitude: 3,
                fused_inc_negative: true,
            },
        ];
        let e_lo = vec![
            vec![AkitaField::from_u64(2), AkitaField::from_u64(3)];
            BYTECODE_ADDRESS_PUSHFORWARD_STAGES
        ];
        let e_hi = vec![
            vec![AkitaField::from_u64(5), AkitaField::from_u64(7)];
            BYTECODE_ADDRESS_PUSHFORWARD_STAGES
        ];

        let output = direct_physical_pushforwards(&rows, rows.len(), shape, &e_lo, &e_hi).unwrap();
        for stage in 0..BYTECODE_ADDRESS_BASE_STAGES {
            assert_eq!(output[2 * stage], AkitaField::from_u64(10));
            assert_eq!(output[2 * stage + 1], AkitaField::from_u64(29));
        }
        for stage in BYTECODE_ADDRESS_BASE_STAGES..BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
            assert_eq!(output[2 * stage], AkitaField::from_u64(10));
            assert_eq!(
                output[2 * stage + 1],
                AkitaField::zero() - AkitaField::from_u64(12)
            );
        }
    }

    #[test]
    fn invalid_physical_prefix_and_pc_are_rejected() {
        let shape = AddressMajorShape::new(2, 1, 1).unwrap();
        let rows = [row(Some(0), false)];
        assert!(matches!(
            BytecodeAddressSparseWorklist::build_with_shape(
                &[rows[0], rows[0], rows[0], rows[0], rows[0]],
                shape
            ),
            Err(BytecodeAddressWorklistError::PhysicalRows { .. })
        ));

        let rows = [row(Some(2), false)];
        assert!(matches!(
            BytecodeAddressSparseWorklist::build_with_shape(&rows, shape),
            Err(BytecodeAddressWorklistError::InvalidPc { row: 0, pc: 2, .. })
        ));
    }
}
