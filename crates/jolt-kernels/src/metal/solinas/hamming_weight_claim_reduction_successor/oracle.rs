//! Independent scalar oracle for the all-hot producer/consumer boundary.

pub const SELECTORS: usize = 29;
pub const BINS: usize = 256;
pub const HOT_PLANES: usize = 29;
pub const FLAG_PLANE: usize = 29;
pub const PLANES: usize = 30;
pub const BALANCED_INC_BIAS: u64 = 0x8080_8080_8080_8080;
const PACKED_PC_MASK: u64 = (1 << 56) - 1;

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
pub struct AllHotProjection {
    rows: usize,
    bytes: Vec<u8>,
}

impl AllHotProjection {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn hot(&self, selector: usize, row: usize) -> Result<u8, OracleError> {
        if selector >= HOT_PLANES || row >= self.rows {
            return Err(OracleError::ProjectionIndex { selector, row });
        }
        Ok(self.bytes[selector * self.rows + row])
    }

    pub fn flags(&self, row: usize) -> Result<u8, OracleError> {
        if row >= self.rows {
            return Err(OracleError::ProjectionIndex {
                selector: FLAG_PLANE,
                row,
            });
        }
        Ok(self.bytes[FLAG_PLANE * self.rows + row])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    EmptyRows,
    RowsNotPowerOfTwo(usize),
    InvalidModulus(u64),
    PointLength { expected: usize, got: usize },
    PointElement { index: usize, value: u64 },
    WeightLength { expected: usize, got: usize },
    ProjectionLength { expected: usize, got: usize },
    ProjectionIndex { selector: usize, row: usize },
}

/// Producer-side model. It decomposes complete words into byte arrays and does
/// not call the direct oracle's selector decoder.
pub fn encode_all_hot_projection(rows: &[OracleRow]) -> Result<AllHotProjection, OracleError> {
    if rows.is_empty() {
        return Err(OracleError::EmptyRows);
    }
    if !rows.len().is_power_of_two() {
        return Err(OracleError::RowsNotPowerOfTwo(rows.len()));
    }
    let mut bytes = vec![0u8; PLANES * rows.len()];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let (hot, flags) = producer_projection(row);
        for (selector, value) in hot.into_iter().enumerate() {
            bytes[selector * rows.len() + row_index] = value;
        }
        bytes[FLAG_PLANE * rows.len() + row_index] = flags;
    }
    Ok(AllHotProjection {
        rows: rows.len(),
        bytes,
    })
}

/// Definition-level path over raw rows. Bucket zero is removed here, exactly
/// where Akita's Hamming preparation recenters every selector table.
pub fn direct_recentered_masses(
    rows: &[OracleRow],
    eq_weights: &[u64],
    modulus: u64,
) -> Result<Vec<u64>, OracleError> {
    validate_inputs(rows, eq_weights, modulus)?;
    let mut masses = vec![0u64; SELECTORS * BINS];
    for (row, weight) in rows.iter().copied().zip(eq_weights.iter().copied()) {
        for selector in 0..SELECTORS {
            let Some(hot) = direct_hot(row, selector) else {
                continue;
            };
            if hot != 0 {
                let slot = selector * BINS + hot;
                masses[slot] = add_mod(masses[slot], weight, modulus);
            }
        }
    }
    Ok(masses)
}

/// Consumer-side model. It reads only the 29 hot planes; optional-column flags
/// are deliberately ignored because absent and present-at-zero both vanish
/// under Hamming's required bucket-zero recentering.
pub fn projected_recentered_masses(
    projection: &AllHotProjection,
    eq_weights: &[u64],
    modulus: u64,
) -> Result<Vec<u64>, OracleError> {
    if modulus < 2 {
        return Err(OracleError::InvalidModulus(modulus));
    }
    if projection.bytes.len() != PLANES * projection.rows {
        return Err(OracleError::ProjectionLength {
            expected: PLANES * projection.rows,
            got: projection.bytes.len(),
        });
    }
    if eq_weights.len() != projection.rows {
        return Err(OracleError::WeightLength {
            expected: projection.rows,
            got: eq_weights.len(),
        });
    }
    let mut masses = vec![0u64; SELECTORS * BINS];
    for (row, weight) in eq_weights.iter().copied().enumerate() {
        if weight >= modulus {
            return Err(OracleError::PointElement {
                index: row,
                value: weight,
            });
        }
        for selector in 0..SELECTORS {
            let hot = usize::from(projection.hot(selector, row)?);
            if hot != 0 {
                let slot = selector * BINS + hot;
                masses[slot] = add_mod(masses[slot], weight, modulus);
            }
        }
    }
    Ok(masses)
}

pub fn equality_weights(point: &[u64], modulus: u64) -> Result<Vec<u64>, OracleError> {
    if modulus < 2 {
        return Err(OracleError::InvalidModulus(modulus));
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

fn producer_projection(row: OracleRow) -> ([u8; SELECTORS], u8) {
    let words = row.words();
    let mut hot = [0u8; SELECTORS];
    hot[..8].copy_from_slice(&words[1].to_be_bytes());
    hot[8..16].copy_from_slice(&words[0].to_be_bytes());

    let pc_plus_one = words[4] & PACKED_PC_MASK;
    let ram_plus_one = words[2];
    let mut flags = 0u8;
    if pc_plus_one != 0 {
        flags |= 1;
        let pc = pc_plus_one - 1;
        hot[16] = (pc >> 8) as u8;
        hot[17] = pc as u8;
    }
    if ram_plus_one != 0 {
        flags |= 2;
        let ram = ram_plus_one - 1;
        hot[18] = (ram >> 8) as u8;
        hot[19] = ram as u8;
    }

    let (biased, carry) = producer_biased_inc(words[3], words[4] >> 63 != 0);
    for (index, byte) in biased.to_le_bytes().into_iter().enumerate() {
        hot[20 + index] = byte.wrapping_add(128);
    }
    hot[28] = carry as u8;
    (hot, flags)
}

fn direct_hot(row: OracleRow, selector: usize) -> Option<usize> {
    let words = row.words();
    match selector {
        0..=7 => Some(((words[1] >> (8 * (7 - selector))) & 0xff) as usize),
        8..=15 => Some(((words[0] >> (8 * (15 - selector))) & 0xff) as usize),
        16..=17 => {
            let plus_one = words[4] & PACKED_PC_MASK;
            (plus_one != 0).then(|| (((plus_one - 1) >> (8 * (17 - selector))) & 0xff) as usize)
        }
        18..=19 => {
            (words[2] != 0).then(|| (((words[2] - 1) >> (8 * (19 - selector))) & 0xff) as usize)
        }
        20..=27 => {
            let (biased, _) = direct_biased_inc(words);
            let standard = (biased >> (8 * (selector - 20))) & 0xff;
            Some(((standard + 128) & 0xff) as usize)
        }
        28 => {
            let (_, carry) = direct_biased_inc(words);
            Some(carry.rem_euclid(BINS as i32) as usize)
        }
        _ => None,
    }
}

fn producer_biased_inc(magnitude: u64, negative: bool) -> (u64, i8) {
    if negative {
        (
            BALANCED_INC_BIAS.wrapping_sub(magnitude),
            if magnitude > BALANCED_INC_BIAS { -1 } else { 0 },
        )
    } else {
        let biased = BALANCED_INC_BIAS.wrapping_add(magnitude);
        (biased, i8::from(biased < BALANCED_INC_BIAS))
    }
}

fn direct_biased_inc(words: [u64; 5]) -> (u64, i32) {
    let magnitude = words[3];
    if words[4] >> 63 != 0 {
        (
            BALANCED_INC_BIAS.wrapping_sub(magnitude),
            if magnitude > BALANCED_INC_BIAS { -1 } else { 0 },
        )
    } else {
        let biased = BALANCED_INC_BIAS.wrapping_add(magnitude);
        (biased, i32::from(biased < BALANCED_INC_BIAS))
    }
}

fn validate_inputs(
    rows: &[OracleRow],
    eq_weights: &[u64],
    modulus: u64,
) -> Result<(), OracleError> {
    if rows.is_empty() {
        return Err(OracleError::EmptyRows);
    }
    if !rows.len().is_power_of_two() {
        return Err(OracleError::RowsNotPowerOfTwo(rows.len()));
    }
    if modulus < 2 {
        return Err(OracleError::InvalidModulus(modulus));
    }
    if eq_weights.len() != rows.len() {
        return Err(OracleError::WeightLength {
            expected: rows.len(),
            got: eq_weights.len(),
        });
    }
    for (index, value) in eq_weights.iter().copied().enumerate() {
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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use super::*;

    const MODULUS: u64 = 97;

    fn rows() -> Vec<OracleRow> {
        (0u64..16)
            .map(|index| {
                let lookup_lo = index.wrapping_mul(0x0102_0304_0506_0708);
                let lookup_hi = (!index).rotate_left(11);
                let ram_plus_one = if index % 3 == 0 { 0 } else { index * 257 + 1 };
                let magnitude = index.wrapping_mul(0x1111_0001_0101);
                let pc_plus_one = if index % 4 == 0 { 0 } else { index * 513 + 1 };
                let sign = u64::from(index % 5 == 0) << 63;
                OracleRow::from_words([
                    lookup_lo,
                    lookup_hi,
                    ram_plus_one,
                    magnitude,
                    pc_plus_one | sign,
                ])
            })
            .collect()
    }

    #[test]
    fn projection_consumer_matches_independent_raw_oracle() {
        let rows = rows();
        let weights = equality_weights(&[2, 3, 5, 7], MODULUS).unwrap();
        let projection = encode_all_hot_projection(&rows).unwrap();

        let direct = direct_recentered_masses(&rows, &weights, MODULUS).unwrap();
        let projected = projected_recentered_masses(&projection, &weights, MODULUS).unwrap();
        assert_eq!(direct, projected);
        assert!(projected
            .chunks_exact(BINS)
            .all(|selector| selector[0] == 0));
    }

    #[test]
    fn optional_absence_and_present_zero_alias_only_at_recentered_bucket() {
        let absent = OracleRow::from_words([0, 0, 0, 0, 0]);
        let present_zero = OracleRow::from_words([0, 0, 1, 0, 1]);
        let rows = [absent, present_zero];
        let projection = encode_all_hot_projection(&rows).unwrap();

        assert_eq!(projection.hot(16, 0).unwrap(), 0);
        assert_eq!(projection.hot(16, 1).unwrap(), 0);
        assert_eq!(projection.flags(0).unwrap(), 0);
        assert_eq!(projection.flags(1).unwrap(), 3);
        let masses = projected_recentered_masses(&projection, &[11, 13], MODULUS).unwrap();
        assert_eq!(masses[16 * BINS], 0);
        assert_eq!(masses[18 * BINS], 0);
    }

    #[test]
    fn plane_major_mutation_is_confined_to_one_selector_table() {
        let rows = rows();
        let weights = equality_weights(&[2, 3, 5, 7], MODULUS).unwrap();
        let mut projection = encode_all_hot_projection(&rows).unwrap();
        let before = projected_recentered_masses(&projection, &weights, MODULUS).unwrap();
        let selector = 11;
        let row = 3;
        let index = selector * projection.rows + row;
        projection.bytes[index] = projection.bytes[index].wrapping_add(1).max(1);
        let after = projected_recentered_masses(&projection, &weights, MODULUS).unwrap();

        for other in 0..SELECTORS {
            if other != selector {
                assert_eq!(
                    &before[other * BINS..(other + 1) * BINS],
                    &after[other * BINS..(other + 1) * BINS]
                );
            }
        }
        assert_ne!(
            &before[selector * BINS..(selector + 1) * BINS],
            &after[selector * BINS..(selector + 1) * BINS]
        );
    }

    #[test]
    fn invalid_shapes_fail_closed() {
        assert_eq!(
            direct_recentered_masses(&[OracleRow::default(); 3], &[1, 1, 1], MODULUS),
            Err(OracleError::RowsNotPowerOfTwo(3))
        );
        assert_eq!(
            equality_weights(&[MODULUS], MODULUS),
            Err(OracleError::PointElement {
                index: 0,
                value: MODULUS,
            })
        );
    }
}
