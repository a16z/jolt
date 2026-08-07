//! Independent scalar definition and validity-free candidate model.
//!
//! The direct oracle decodes every selector from the original row and forms
//! the full equality product. The candidate model has separate decoders,
//! performs the optional-first schedule, and consumes a validity-free hot
//! projection for the remaining selectors.

use super::{
    validate_weight_shape, BooleanityAddressV2Error, BOOLEANITY_ADDRESS_V2_BINS,
    BOOLEANITY_ADDRESS_V2_FIRST_SELECTOR_IDS, BOOLEANITY_ADDRESS_V2_HOT_PLANES,
    BOOLEANITY_ADDRESS_V2_INC_BIAS, BOOLEANITY_ADDRESS_V2_REMAINING_SELECTOR_IDS,
    BOOLEANITY_ADDRESS_V2_SELECTORS,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OracleRow {
    pub lookup_lo: u64,
    pub lookup_hi: u64,
    pub ram_address_plus_one: u64,
    pub fused_inc_magnitude: u64,
    pub packed_pc_and_flags: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotProjection {
    rows: usize,
    hot: Vec<u8>,
}

impl HotProjection {
    pub fn from_parts(rows: usize, hot: Vec<u8>) -> Result<Self, BooleanityAddressV2Error> {
        let expected = rows
            .checked_mul(BOOLEANITY_ADDRESS_V2_HOT_PLANES)
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?;
        if hot.len() != expected {
            return Err(BooleanityAddressV2Error::HotStorageLength {
                expected,
                got: hot.len(),
            });
        }
        Ok(Self { rows, hot })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.hot
    }

    pub fn hot(&self, row: usize, selector: usize) -> Result<u8, BooleanityAddressV2Error> {
        if row >= self.rows {
            return Err(BooleanityAddressV2Error::RowOutOfBounds {
                rows: self.rows,
                row,
            });
        }
        if selector >= BOOLEANITY_ADDRESS_V2_SELECTORS {
            return Err(BooleanityAddressV2Error::InvalidSelector(selector));
        }
        Ok(self.hot[selector * self.rows + row])
    }
}

/// Direct relation definition, including optional-column absence.
pub fn canonical_hot_indices(row: OracleRow) -> [Option<u8>; 29] {
    let lookup = u128::from(row.lookup_lo) | (u128::from(row.lookup_hi) << 64);
    let mut hot = [None; BOOLEANITY_ADDRESS_V2_SELECTORS];
    for (selector, slot) in hot[..16].iter_mut().enumerate() {
        let shift = 8 * (15 - selector);
        *slot = Some(((lookup >> shift) & 0xff) as u8);
    }

    let pc_plus_one = row.packed_pc_and_flags & 0x00ff_ffff_ffff_ffff;
    if pc_plus_one != 0 {
        let pc = pc_plus_one - 1;
        hot[16] = Some(((pc >> 8) & 0xff) as u8);
        hot[17] = Some((pc & 0xff) as u8);
    }
    if row.ram_address_plus_one != 0 {
        let ram = row.ram_address_plus_one - 1;
        hot[18] = Some(((ram >> 8) & 0xff) as u8);
        hot[19] = Some((ram & 0xff) as u8);
    }

    let (biased, carry) = direct_biased_increment(row);
    for (selector, slot) in hot.iter_mut().enumerate().take(28).skip(20) {
        let standard = ((biased >> (8 * (selector - 20))) & 0xff) as u8;
        *slot = Some(standard.wrapping_add(128));
    }
    hot[28] = Some(carry as u8);
    hot
}

/// Candidate projection. Optional absence is encoded as zero and no validity
/// byte is materialized.
pub fn pack_rows(rows: &[OracleRow]) -> Result<HotProjection, BooleanityAddressV2Error> {
    let mut hot = vec![
        0u8;
        rows.len()
            .checked_mul(BOOLEANITY_ADDRESS_V2_HOT_PLANES)
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?
    ];
    for (row_index, row) in rows.iter().copied().enumerate() {
        for selector in 0..8 {
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                selector,
                ((row.lookup_hi >> (8 * (7 - selector))) & 0xff) as u8,
            );
        }
        for selector in 8..16 {
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                selector,
                ((row.lookup_lo >> (8 * (15 - selector))) & 0xff) as u8,
            );
        }

        let pc_plus_one = row.packed_pc_and_flags & 0x00ff_ffff_ffff_ffff;
        if pc_plus_one != 0 {
            let pc = pc_plus_one - 1;
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                16,
                ((pc >> 8) & 0xff) as u8,
            );
            store_hot(&mut hot, rows.len(), row_index, 17, (pc & 0xff) as u8);
        }
        if row.ram_address_plus_one != 0 {
            let ram = row.ram_address_plus_one - 1;
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                18,
                ((ram >> 8) & 0xff) as u8,
            );
            store_hot(&mut hot, rows.len(), row_index, 19, (ram & 0xff) as u8);
        }

        let (biased, carry) = candidate_biased_increment(row);
        for selector in 20..28 {
            let standard = ((biased >> (8 * (selector - 20))) & 0xff) as u8;
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                selector,
                standard.wrapping_add(128),
            );
        }
        store_hot(&mut hot, rows.len(), row_index, 28, carry as u8);
    }
    HotProjection::from_parts(rows.len(), hot)
}

/// Independent direct pushforward: no tensor regrouping and no projection.
pub fn unfactored_pushforward(
    rows: &[OracleRow],
    e_in: &[u64],
    e_out: &[u64],
    modulus: u64,
) -> Result<Vec<u64>, BooleanityAddressV2Error> {
    validate_oracle_inputs(rows.len(), e_in, e_out, modulus)?;
    let mut output = vec![0u64; BOOLEANITY_ADDRESS_V2_SELECTORS * BOOLEANITY_ADDRESS_V2_BINS];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let weight = mul_mod(
            e_out[row_index / e_in.len()],
            e_in[row_index % e_in.len()],
            modulus,
        );
        for (selector, hot) in canonical_hot_indices(row).into_iter().enumerate() {
            if let Some(hot) = hot {
                let index = selector * BOOLEANITY_ADDRESS_V2_BINS + hot as usize;
                output[index] = add_mod(output[index], weight, modulus);
            }
        }
    }
    Ok(output)
}

/// CPU model of the v2 raw-first and four packed accumulation tiles.
pub fn factorized_pushforward(
    rows: &[OracleRow],
    projection: &HotProjection,
    e_in: &[u64],
    e_out: &[u64],
    modulus: u64,
) -> Result<Vec<u64>, BooleanityAddressV2Error> {
    validate_oracle_inputs(rows.len(), e_in, e_out, modulus)?;
    if projection.rows() != rows.len() {
        return Err(BooleanityAddressV2Error::HotStorageLength {
            expected: rows.len() * BOOLEANITY_ADDRESS_V2_HOT_PLANES,
            got: projection.as_bytes().len(),
        });
    }
    let fields = BOOLEANITY_ADDRESS_V2_SELECTORS * BOOLEANITY_ADDRESS_V2_BINS;
    let mut output = vec![0u64; fields];
    let mut block = vec![0u64; fields];
    for (x_out, &outer) in e_out.iter().enumerate() {
        block.fill(0);
        let row_base = x_out * e_in.len();
        for (x_in, &inner) in e_in.iter().enumerate() {
            let row_index = row_base + x_in;
            let row = rows[row_index];
            for selector in BOOLEANITY_ADDRESS_V2_FIRST_SELECTOR_IDS {
                if let Some(hot) = candidate_first_hot(row, selector) {
                    let index = selector as usize * BOOLEANITY_ADDRESS_V2_BINS + hot as usize;
                    block[index] = add_mod(block[index], inner, modulus);
                }
            }
            for selector in BOOLEANITY_ADDRESS_V2_REMAINING_SELECTOR_IDS {
                let hot = projection.hot(row_index, selector as usize)?;
                let index = selector as usize * BOOLEANITY_ADDRESS_V2_BINS + hot as usize;
                block[index] = add_mod(block[index], inner, modulus);
            }
        }
        for (output, block) in output.iter_mut().zip(&block) {
            *output = add_mod(*output, mul_mod(outer, *block, modulus), modulus);
        }
    }
    Ok(output)
}

fn candidate_first_hot(row: OracleRow, selector: u8) -> Option<u8> {
    match selector {
        0 => Some((row.lookup_hi >> 56) as u8),
        1 => Some((row.lookup_hi >> 48) as u8),
        16 | 17 => {
            let pc_plus_one = row.packed_pc_and_flags & 0x00ff_ffff_ffff_ffff;
            (pc_plus_one != 0).then(|| {
                let pc = pc_plus_one - 1;
                if selector == 16 {
                    (pc >> 8) as u8
                } else {
                    pc as u8
                }
            })
        }
        18 | 19 => (row.ram_address_plus_one != 0).then(|| {
            let ram = row.ram_address_plus_one - 1;
            if selector == 18 {
                (ram >> 8) as u8
            } else {
                ram as u8
            }
        }),
        _ => None,
    }
}

fn direct_biased_increment(row: OracleRow) -> (u64, i8) {
    if row.packed_pc_and_flags >> 63 != 0 {
        let (biased, borrowed) =
            BOOLEANITY_ADDRESS_V2_INC_BIAS.overflowing_sub(row.fused_inc_magnitude);
        (biased, -i8::from(borrowed))
    } else {
        let (biased, overflowed) =
            BOOLEANITY_ADDRESS_V2_INC_BIAS.overflowing_add(row.fused_inc_magnitude);
        (biased, i8::from(overflowed))
    }
}

fn candidate_biased_increment(row: OracleRow) -> (u64, i8) {
    if row.packed_pc_and_flags & (1 << 63) != 0 {
        let biased = BOOLEANITY_ADDRESS_V2_INC_BIAS.wrapping_sub(row.fused_inc_magnitude);
        let carry = if row.fused_inc_magnitude > BOOLEANITY_ADDRESS_V2_INC_BIAS {
            -1
        } else {
            0
        };
        (biased, carry)
    } else {
        let biased = BOOLEANITY_ADDRESS_V2_INC_BIAS.wrapping_add(row.fused_inc_magnitude);
        let carry = i8::from(biased < BOOLEANITY_ADDRESS_V2_INC_BIAS);
        (biased, carry)
    }
}

fn validate_oracle_inputs(
    rows: usize,
    e_in: &[u64],
    e_out: &[u64],
    modulus: u64,
) -> Result<(), BooleanityAddressV2Error> {
    validate_weight_shape(rows, e_in.len(), e_out.len())?;
    if modulus < 2 {
        return Err(BooleanityAddressV2Error::InvalidModulus(modulus));
    }
    Ok(())
}

fn store_hot(bytes: &mut [u8], rows: usize, row: usize, selector: usize, hot: u8) {
    bytes[selector * rows + row] = hot;
}

fn add_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(modulus)) as u64
}

fn mul_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(modulus)) as u64
}
