//! Scalar oracle independent of both retained Metal shaders.

use super::{HAMMING_V2_BINS, HAMMING_V2_HOT_PLANES, HAMMING_V2_SELECTORS};

const PC_MASK: u64 = 0x00ff_ffff_ffff_ffff;
const INC_BIAS: u64 = 0x8080_8080_8080_8080;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OracleRow([u64; 5]);

impl OracleRow {
    pub const fn from_words(words: [u64; 5]) -> Self {
        Self(words)
    }

    pub const fn words(self) -> [u64; 5] {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotProjection {
    rows: usize,
    bytes: Vec<u8>,
}

impl HotProjection {
    pub fn from_bytes(rows: usize, bytes: Vec<u8>) -> Result<Self, OracleError> {
        let expected = rows
            .checked_mul(HAMMING_V2_HOT_PLANES)
            .ok_or(OracleError::ArithmeticOverflow)?;
        if bytes.len() != expected {
            return Err(OracleError::ProjectionLength {
                expected,
                got: bytes.len(),
            });
        }
        Ok(Self { rows, bytes })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn hot(&self, selector: usize, row: usize) -> Result<u8, OracleError> {
        if selector >= HAMMING_V2_HOT_PLANES || row >= self.rows {
            return Err(OracleError::ProjectionIndex { selector, row });
        }
        Ok(self.bytes[selector * self.rows + row])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    EmptyRows,
    RowsNotPowerOfTwo(usize),
    InvalidModulus(u64),
    PointTooLong(usize),
    PointElement { index: usize, value: u64 },
    WeightLength { expected: usize, got: usize },
    ProjectionLength { expected: usize, got: usize },
    ProjectionIndex { selector: usize, row: usize },
    ArithmeticOverflow,
}

/// Producer definition. This builds plane-major bytes without using the
/// direct consumer decoder below.
pub fn encode_hot_projection(rows: &[OracleRow]) -> Result<HotProjection, OracleError> {
    validate_rows(rows)?;
    let length = rows
        .len()
        .checked_mul(HAMMING_V2_HOT_PLANES)
        .ok_or(OracleError::ArithmeticOverflow)?;
    let mut bytes = vec![0u8; length];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let hot = producer_hot_bytes(row);
        for (selector, value) in hot.into_iter().enumerate() {
            bytes[selector * rows.len() + row_index] = value;
        }
    }
    HotProjection::from_bytes(rows.len(), bytes)
}

/// Definition-level accepted-row path. Optional columns are absent rather
/// than encoded as zero, and bucket zero is removed explicitly.
pub fn direct_recentered_masses(
    rows: &[OracleRow],
    weights: &[u64],
    modulus: u64,
) -> Result<Vec<u64>, OracleError> {
    validate_inputs(rows, weights, modulus)?;
    let mut output = vec![0u64; HAMMING_V2_SELECTORS * HAMMING_V2_BINS];
    for (row, weight) in rows.iter().copied().zip(weights.iter().copied()) {
        for selector in 0..HAMMING_V2_SELECTORS {
            let Some(hot) = direct_hot(row, selector) else {
                continue;
            };
            if hot != 0 {
                let slot = selector * HAMMING_V2_BINS + hot;
                output[slot] = add_mod(output[slot], weight, modulus);
            }
        }
    }
    Ok(output)
}

/// Retained consumer definition. It cannot see optional-column flags; absent
/// and present-at-zero are equivalent only because bucket zero is removed.
pub fn retained_recentered_masses(
    projection: &HotProjection,
    weights: &[u64],
    modulus: u64,
) -> Result<Vec<u64>, OracleError> {
    if modulus < 2 {
        return Err(OracleError::InvalidModulus(modulus));
    }
    if weights.len() != projection.rows {
        return Err(OracleError::WeightLength {
            expected: projection.rows,
            got: weights.len(),
        });
    }
    let mut output = vec![0u64; HAMMING_V2_SELECTORS * HAMMING_V2_BINS];
    for (row, weight) in weights.iter().copied().enumerate() {
        if weight >= modulus {
            return Err(OracleError::PointElement {
                index: row,
                value: weight,
            });
        }
        for selector in 0..HAMMING_V2_SELECTORS {
            let hot = usize::from(projection.hot(selector, row)?);
            if hot != 0 {
                let slot = selector * HAMMING_V2_BINS + hot;
                output[slot] = add_mod(output[slot], weight, modulus);
            }
        }
    }
    Ok(output)
}

pub fn equality_weights(point: &[u64], modulus: u64) -> Result<Vec<u64>, OracleError> {
    if modulus < 2 {
        return Err(OracleError::InvalidModulus(modulus));
    }
    if point.len() >= usize::BITS as usize {
        return Err(OracleError::PointTooLong(point.len()));
    }
    for (index, value) in point.iter().copied().enumerate() {
        if value >= modulus {
            return Err(OracleError::PointElement { index, value });
        }
    }
    let rows = 1usize << point.len();
    Ok((0..rows)
        .map(|row| {
            point.iter().copied().enumerate().fold(1, |weight, (i, r)| {
                let shift = point.len() - i - 1;
                let factor = if row & (1 << shift) == 0 {
                    sub_mod(1, r, modulus)
                } else {
                    r
                };
                mul_mod(weight, factor, modulus)
            })
        })
        .collect())
}

fn producer_hot_bytes(row: OracleRow) -> [u8; HAMMING_V2_SELECTORS] {
    let [lookup_lo, lookup_hi, ram_plus_one, magnitude, packed_pc] = row.words();
    let mut hot = [0u8; HAMMING_V2_SELECTORS];
    hot[..8].copy_from_slice(&lookup_hi.to_be_bytes());
    hot[8..16].copy_from_slice(&lookup_lo.to_be_bytes());

    let pc_plus_one = packed_pc & PC_MASK;
    if pc_plus_one != 0 {
        let pc = pc_plus_one - 1;
        hot[16] = (pc >> 8) as u8;
        hot[17] = pc as u8;
    }
    if ram_plus_one != 0 {
        let ram = ram_plus_one - 1;
        hot[18] = (ram >> 8) as u8;
        hot[19] = ram as u8;
    }

    let negative = packed_pc >> 63 != 0;
    let (biased, carry) = if negative {
        (
            INC_BIAS.wrapping_sub(magnitude),
            if magnitude > INC_BIAS { -1i8 } else { 0 },
        )
    } else {
        let biased = INC_BIAS.wrapping_add(magnitude);
        (biased, i8::from(biased < INC_BIAS))
    };
    for (index, byte) in biased.to_le_bytes().into_iter().enumerate() {
        hot[20 + index] = byte.wrapping_add(128);
    }
    hot[28] = carry as u8;
    hot
}

fn direct_hot(row: OracleRow, selector: usize) -> Option<usize> {
    let [lookup_lo, lookup_hi, ram_plus_one, magnitude, packed_pc] = row.words();
    match selector {
        0..=7 => Some(((lookup_hi >> (8 * (7 - selector))) & 0xff) as usize),
        8..=15 => Some(((lookup_lo >> (8 * (15 - selector))) & 0xff) as usize),
        16..=17 => {
            let pc_plus_one = packed_pc & PC_MASK;
            (pc_plus_one != 0)
                .then(|| (((pc_plus_one - 1) >> (8 * (17 - selector))) & 0xff) as usize)
        }
        18..=19 => (ram_plus_one != 0)
            .then(|| (((ram_plus_one - 1) >> (8 * (19 - selector))) & 0xff) as usize),
        20..=27 => {
            let biased = if packed_pc >> 63 != 0 {
                INC_BIAS.wrapping_sub(magnitude)
            } else {
                INC_BIAS.wrapping_add(magnitude)
            };
            let byte = ((biased >> (8 * (selector - 20))) & 0xff) as u8;
            Some(byte.wrapping_add(128) as usize)
        }
        28 => {
            let carry = if packed_pc >> 63 != 0 {
                if magnitude > INC_BIAS {
                    -1i8
                } else {
                    0
                }
            } else {
                i8::from(INC_BIAS.wrapping_add(magnitude) < INC_BIAS)
            };
            Some(carry as u8 as usize)
        }
        _ => None,
    }
}

fn validate_rows(rows: &[OracleRow]) -> Result<(), OracleError> {
    if rows.is_empty() {
        return Err(OracleError::EmptyRows);
    }
    if !rows.len().is_power_of_two() {
        return Err(OracleError::RowsNotPowerOfTwo(rows.len()));
    }
    Ok(())
}

fn validate_inputs(rows: &[OracleRow], weights: &[u64], modulus: u64) -> Result<(), OracleError> {
    validate_rows(rows)?;
    if modulus < 2 {
        return Err(OracleError::InvalidModulus(modulus));
    }
    if weights.len() != rows.len() {
        return Err(OracleError::WeightLength {
            expected: rows.len(),
            got: weights.len(),
        });
    }
    for (index, value) in weights.iter().copied().enumerate() {
        if value >= modulus {
            return Err(OracleError::PointElement { index, value });
        }
    }
    Ok(())
}

fn add_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(modulus)) as u64
}

fn sub_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) + u128::from(modulus) - u128::from(right)) % u128::from(modulus)) as u64
}

fn mul_mod(left: u64, right: u64, modulus: u64) -> u64 {
    (u128::from(left) * u128::from(right) % u128::from(modulus)) as u64
}
