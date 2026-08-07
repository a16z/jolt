//! CPU-only oracle for the address-major layout and schedule receipt.
//!
//! This deliberately performs two host scans. It is a parity oracle, not an
//! admissible producer implementation.

use core::fmt;

use super::carrier::{
    AddressMajorShape, CarrierError, PackedCell, PackedInnerSign, TopologyScheduleReceipt,
    SHORT_THRESHOLD, SIMD_WIDTH,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Row {
    pub mapped_pc: Option<usize>,
    pub fused_inc_magnitude: u64,
    pub fused_inc_negative: bool,
}

impl Row {
    pub const fn push_pc(self) -> usize {
        match self.mapped_pc {
            Some(pc) => pc,
            None => 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostAddressMajorCarrier {
    shape: AddressMajorShape,
    cells: Vec<PackedCell>,
    inner_sign: Vec<PackedInnerSign>,
    magnitude: Vec<u64>,
    topology: TopologyScheduleReceipt,
}

impl HostAddressMajorCarrier {
    pub fn build(rows: &[Row], shape: AddressMajorShape) -> Result<Self, OracleError> {
        shape.validate()?;
        let row_count = shape.rows()?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        let outer_length = shape.outer_length()?;
        if rows.len() != row_count {
            return Err(OracleError::RowCount {
                expected: row_count,
                got: rows.len(),
            });
        }
        if rows.iter().any(|row| row.push_pc() >= addresses) {
            return Err(OracleError::InvalidPc);
        }

        let mut cells = vec![PackedCell::default(); shape.cells()?];
        let mut inner_sign = vec![PackedInnerSign::default(); row_count];
        let mut magnitude = vec![0u64; row_count];
        let mut counts = vec![0usize; addresses];
        let mut cursors = vec![0usize; addresses];

        for outer in 0..outer_length {
            counts.fill(0);
            let base = outer * inner_length;
            for row in &rows[base..base + inner_length] {
                counts[row.push_pc()] += 1;
            }

            let mut start = 0usize;
            for (address, &count) in counts.iter().enumerate() {
                cells[address * outer_length + outer] = PackedCell::new(start, count)?;
                cursors[address] = start;
                start = start.checked_add(count).ok_or(OracleError::Overflow)?;
            }
            if start != inner_length {
                return Err(OracleError::InvalidLayout);
            }

            for (inner, row) in rows[base..base + inner_length].iter().enumerate() {
                let address = row.push_pc();
                let destination = base + cursors[address];
                cursors[address] += 1;
                inner_sign[destination] = PackedInnerSign::new(inner, row.fused_inc_negative)?;
                magnitude[destination] = row.fused_inc_magnitude;
            }
        }

        let topology = schedule_from_cells(shape, &cells)?;
        let carrier = Self {
            shape,
            cells,
            inner_sign,
            magnitude,
            topology,
        };
        carrier.validate_against_rows(rows)?;
        Ok(carrier)
    }

    pub const fn shape(&self) -> AddressMajorShape {
        self.shape
    }

    pub fn cells(&self) -> &[PackedCell] {
        &self.cells
    }

    pub fn inner_sign(&self) -> &[PackedInnerSign] {
        &self.inner_sign
    }

    pub fn magnitude(&self) -> &[u64] {
        &self.magnitude
    }

    pub const fn topology(&self) -> TopologyScheduleReceipt {
        self.topology
    }

    pub fn cell(&self, address: usize, outer: usize) -> Result<PackedCell, OracleError> {
        let addresses = self.shape.addresses()?;
        let outer_length = self.shape.outer_length()?;
        if address >= addresses || outer >= outer_length {
            return Err(OracleError::InvalidLayout);
        }
        Ok(self.cells[address * outer_length + outer])
    }

    pub fn validate_against_rows(&self, rows: &[Row]) -> Result<(), OracleError> {
        let row_count = self.shape.rows()?;
        let addresses = self.shape.addresses()?;
        let inner_length = self.shape.inner_length()?;
        let outer_length = self.shape.outer_length()?;
        if rows.len() != row_count
            || self.cells.len() != self.shape.cells()?
            || self.inner_sign.len() != row_count
            || self.magnitude.len() != row_count
        {
            return Err(OracleError::InvalidLayout);
        }

        let mut seen = vec![false; inner_length];
        for outer in 0..outer_length {
            seen.fill(false);
            let base = outer * inner_length;
            let mut expected_start = 0usize;
            for address in 0..addresses {
                let cell = self.cell(address, outer)?;
                if cell.start() != expected_start
                    || cell.start().checked_add(cell.count()).is_none()
                    || cell.start() + cell.count() > inner_length
                {
                    return Err(OracleError::InvalidLayout);
                }
                for slot in cell.start()..cell.start() + cell.count() {
                    let packed = self.inner_sign[base + slot];
                    let inner = packed.inner();
                    if inner >= inner_length || seen[inner] {
                        return Err(OracleError::InvalidLayout);
                    }
                    seen[inner] = true;
                    let row = rows[base + inner];
                    if row.push_pc() != address
                        || row.fused_inc_negative != packed.negative()
                        || row.fused_inc_magnitude != self.magnitude[base + slot]
                    {
                        return Err(OracleError::InvalidLayout);
                    }
                }
                expected_start += cell.count();
            }
            if expected_start != inner_length || seen.iter().any(|seen| !seen) {
                return Err(OracleError::InvalidLayout);
            }
        }
        let topology = schedule_from_cells(self.shape, &self.cells)?;
        if topology != self.topology {
            return Err(OracleError::InvalidLayout);
        }
        Ok(())
    }
}

fn schedule_from_cells(
    shape: AddressMajorShape,
    cells: &[PackedCell],
) -> Result<TopologyScheduleReceipt, OracleError> {
    let addresses = shape.addresses()?;
    let outer_length = shape.outer_length()?;
    if cells.len() != shape.cells()? {
        return Err(OracleError::InvalidLayout);
    }

    let mut short_occurrences = 0u64;
    let mut long_occurrences = 0u64;
    let mut short_runs = 0u64;
    let mut long_runs = 0u64;
    let mut short_batches = 0u64;
    let mut padded_short_lanes = 0u64;
    let mut padded_long_lanes = 0u64;
    let mut maximum_run = 0u64;

    for address in 0..addresses {
        for batch_start in (0..outer_length).step_by(SIMD_WIDTH) {
            let batch_end = (batch_start + SIMD_WIDTH).min(outer_length);
            let mut maximum_short = 0usize;
            for outer in batch_start..batch_end {
                let count = cells[address * outer_length + outer].count();
                maximum_run = maximum_run.max(count as u64);
                if count == 0 {
                    continue;
                }
                if count <= SHORT_THRESHOLD {
                    short_runs += 1;
                    short_occurrences = short_occurrences
                        .checked_add(count as u64)
                        .ok_or(OracleError::Overflow)?;
                    maximum_short = maximum_short.max(count);
                } else {
                    long_runs += 1;
                    long_occurrences = long_occurrences
                        .checked_add(count as u64)
                        .ok_or(OracleError::Overflow)?;
                    padded_long_lanes = padded_long_lanes
                        .checked_add(round_up_usize(count, SIMD_WIDTH)? as u64)
                        .ok_or(OracleError::Overflow)?;
                }
            }
            if maximum_short != 0 {
                short_batches += 1;
                padded_short_lanes = padded_short_lanes
                    .checked_add((SIMD_WIDTH * maximum_short) as u64)
                    .ok_or(OracleError::Overflow)?;
            }
        }
    }

    let topology = TopologyScheduleReceipt {
        short_occurrences,
        long_occurrences,
        short_runs,
        long_runs,
        short_batches,
        padded_short_lanes,
        padded_long_lanes,
        maximum_run,
    };
    topology.validate(shape)?;
    Ok(topology)
}

fn round_up_usize(value: usize, multiple: usize) -> Result<usize, OracleError> {
    value
        .div_ceil(multiple)
        .checked_mul(multiple)
        .ok_or(OracleError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    Carrier(CarrierError),
    RowCount { expected: usize, got: usize },
    InvalidPc,
    InvalidLayout,
    Overflow,
}

impl From<CarrierError> for OracleError {
    fn from(value: CarrierError) -> Self {
        Self::Carrier(value)
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carrier(error) => error.fmt(f),
            Self::RowCount { expected, got } => {
                write!(f, "oracle has {got} rows, expected {expected}")
            }
            Self::InvalidPc => f.write_str("oracle row has a PC outside the address domain"),
            Self::InvalidLayout => f.write_str("invalid oracle address-major layout"),
            Self::Overflow => f.write_str("oracle topology arithmetic overflowed"),
        }
    }
}

impl std::error::Error for OracleError {}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked fixtures")]
mod tests {
    use super::*;

    fn sample_rows() -> (AddressMajorShape, Vec<Row>) {
        let shape = AddressMajorShape::new(7, 2, 6).unwrap();
        let mut rows = Vec::with_capacity(128);
        for inner in 0usize..64 {
            rows.push(Row {
                mapped_pc: Some(usize::from(inner >= 40)),
                fused_inc_magnitude: inner as u64 + 1,
                fused_inc_negative: inner.is_multiple_of(3),
            });
        }
        for inner in 0usize..64 {
            rows.push(Row {
                mapped_pc: if inner == 0 { None } else { Some(0) },
                fused_inc_magnitude: 100 + inner as u64,
                fused_inc_negative: inner.is_multiple_of(5),
            });
        }
        (shape, rows)
    }

    #[test]
    fn address_major_scatter_is_stable_and_exact() {
        let (shape, rows) = sample_rows();
        let carrier = HostAddressMajorCarrier::build(&rows, shape).unwrap();
        assert_eq!(carrier.cell(0, 0).unwrap(), PackedCell::new(0, 40).unwrap());
        assert_eq!(
            carrier.cell(1, 0).unwrap(),
            PackedCell::new(40, 24).unwrap()
        );
        assert_eq!(carrier.cell(0, 1).unwrap(), PackedCell::new(0, 64).unwrap());
        assert_eq!(carrier.topology().short_occurrences, 24);
        assert_eq!(carrier.topology().long_occurrences, 104);
        assert_eq!(carrier.topology().short_runs, 1);
        assert_eq!(carrier.topology().long_runs, 2);
        assert_eq!(carrier.topology().padded_short_lanes, 768);
        assert_eq!(carrier.topology().padded_long_lanes, 128);
        assert_eq!(carrier.topology().maximum_run, 64);
        carrier.validate_against_rows(&rows).unwrap();
    }

    #[test]
    fn absent_pc_uses_the_address_phase_zero_rule() {
        assert_eq!(
            Row {
                mapped_pc: None,
                fused_inc_magnitude: 0,
                fused_inc_negative: false,
            }
            .push_pc(),
            0
        );
    }

    #[test]
    fn invalid_pc_fails_before_publication() {
        let (shape, mut rows) = sample_rows();
        rows[0].mapped_pc = Some(shape.addresses().unwrap());
        assert_eq!(
            HostAddressMajorCarrier::build(&rows, shape),
            Err(OracleError::InvalidPc)
        );
    }

    #[test]
    fn a_corrupted_cell_is_detected_against_source_rows() {
        let (shape, rows) = sample_rows();
        let mut carrier = HostAddressMajorCarrier::build(&rows, shape).unwrap();
        carrier.cells[0] = PackedCell::new(1, 39).unwrap();
        assert_eq!(
            carrier.validate_against_rows(&rows),
            Err(OracleError::InvalidLayout)
        );
    }
}
