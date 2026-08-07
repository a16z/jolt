//! Scalar correctness oracles with no Metal/runtime dependency.
//!
//! `unfactored_pushforward` evaluates the full equality weight per cycle from
//! the original 40-byte row. It does not consume the packed representation or
//! reuse the factorized implementation's selector decoder.

use jolt_field::Field;

use super::super::BooleanityRow;
use super::{
    validate_weight_shape, BooleanityAddressSuccessorError, BOOLEANITY_ADDRESS_SUCCESSOR_BINS,
    BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS, BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES,
    BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS, BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS,
};

const BYTECODE_PRESENT: u8 = 1 << 0;
const RAM_PRESENT: u8 = 1 << 1;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedAddressRows {
    rows: usize,
    hot: Vec<u8>,
    validity: Vec<u8>,
}

impl PackedAddressRows {
    pub fn from_parts(
        rows: usize,
        hot: Vec<u8>,
        validity: Vec<u8>,
    ) -> Result<Self, BooleanityAddressSuccessorError> {
        let expected = rows
            .checked_mul(BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
        if hot.len() != expected {
            return Err(BooleanityAddressSuccessorError::PackedStorageLength {
                expected,
                got: hot.len(),
            });
        }
        if validity.len() != rows {
            return Err(BooleanityAddressSuccessorError::ValidityStorageLength {
                expected: rows,
                got: validity.len(),
            });
        }
        Ok(Self {
            rows,
            hot,
            validity,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.hot
    }

    pub fn validity_bytes(&self) -> &[u8] {
        &self.validity
    }

    pub fn hot(
        &self,
        row: usize,
        selector: usize,
    ) -> Result<Option<u8>, BooleanityAddressSuccessorError> {
        if row >= self.rows {
            return Err(BooleanityAddressSuccessorError::RowOutOfBounds {
                rows: self.rows,
                row,
            });
        }
        if selector >= BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS {
            return Err(BooleanityAddressSuccessorError::InvalidPackedSelector(
                selector,
            ));
        }
        let flags = self.validity[row];
        if matches!(selector, 16 | 17) && flags & BYTECODE_PRESENT == 0 {
            return Ok(None);
        }
        if matches!(selector, 18 | 19) && flags & RAM_PRESENT == 0 {
            return Ok(None);
        }
        Ok(Some(self.hot[selector * self.rows + row]))
    }
}

/// Materializes the shader's 29 hot-index planes and separate validity plane.
pub fn pack_rows(
    rows: &[BooleanityRow],
) -> Result<PackedAddressRows, BooleanityAddressSuccessorError> {
    let mut hot = vec![
        0u8;
        rows.len()
            .checked_mul(BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?
    ];
    let mut validity = vec![0u8; rows.len()];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let [lookup_lo, lookup_hi, ram_plus_one, magnitude, packed_pc] = row.words();
        for selector in 0..8 {
            let shift = 8 * (7 - selector);
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                selector,
                ((lookup_hi >> shift) & 0xff) as u8,
            );
        }
        for selector in 8..16 {
            let shift = 8 * (15 - selector);
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                selector,
                ((lookup_lo >> shift) & 0xff) as u8,
            );
        }

        let pc_plus_one = packed_pc & 0x00ff_ffff_ffff_ffff;
        let mut flags = 0u8;
        if pc_plus_one != 0 {
            flags |= BYTECODE_PRESENT;
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
        if ram_plus_one != 0 {
            flags |= RAM_PRESENT;
            let ram = ram_plus_one - 1;
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                18,
                ((ram >> 8) & 0xff) as u8,
            );
            store_hot(&mut hot, rows.len(), row_index, 19, (ram & 0xff) as u8);
        }

        let (biased, carry) = biased_increment(magnitude, packed_pc);
        for selector in 20..28 {
            let shift = 8 * (selector - 20);
            let standard = ((biased >> shift) & 0xff) as u8;
            store_hot(
                &mut hot,
                rows.len(),
                row_index,
                selector,
                standard.wrapping_add(128),
            );
        }
        store_hot(&mut hot, rows.len(), row_index, 28, carry as u8);
        validity[row_index] = flags;
    }
    PackedAddressRows::from_parts(rows.len(), hot, validity)
}

/// Independent direct definition of all 29 production hot indices.
pub fn canonical_hot_indices(row: BooleanityRow) -> [Option<u8>; 29] {
    let [lookup_lo, lookup_hi, ram_plus_one, magnitude, packed_pc] = row.words();
    let lookup = u128::from(lookup_lo) | (u128::from(lookup_hi) << 64);
    let mut hot = [None; BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS];
    for (selector, slot) in hot[..16].iter_mut().enumerate() {
        let shift = 8 * (15 - selector);
        *slot = Some(((lookup >> shift) & 0xff) as u8);
    }

    let pc_plus_one = packed_pc & 0x00ff_ffff_ffff_ffff;
    if pc_plus_one != 0 {
        let pc = pc_plus_one - 1;
        hot[16] = Some(((pc >> 8) & 0xff) as u8);
        hot[17] = Some((pc & 0xff) as u8);
    }
    if ram_plus_one != 0 {
        let ram = ram_plus_one - 1;
        hot[18] = Some(((ram >> 8) & 0xff) as u8);
        hot[19] = Some((ram & 0xff) as u8);
    }

    let (biased, carry) = if packed_pc >> 63 != 0 {
        let (biased, borrowed) = BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS.overflowing_sub(magnitude);
        (biased, -i8::from(borrowed))
    } else {
        let (biased, overflowed) = BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS.overflowing_add(magnitude);
        (biased, i8::from(overflowed))
    };
    for (selector, slot) in hot.iter_mut().enumerate().take(28).skip(20) {
        let standard = ((biased >> (8 * (selector - 20))) & 0xff) as u8;
        *slot = Some(standard.wrapping_add(128));
    }
    hot[28] = Some(carry as u8);
    hot
}

/// Direct relation oracle: one full equality product and one original-row
/// decode per cycle, with no tensor-block regrouping.
pub fn unfactored_pushforward<F: Field>(
    rows: &[BooleanityRow],
    e_in: &[F],
    e_out: &[F],
) -> Result<Vec<F>, BooleanityAddressSuccessorError> {
    validate_weight_shape(rows.len(), e_in.len(), e_out.len())?;
    let mut output =
        vec![F::zero(); BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS * BOOLEANITY_ADDRESS_SUCCESSOR_BINS];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let weight = e_out[row_index / e_in.len()] * e_in[row_index % e_in.len()];
        for (selector, hot) in canonical_hot_indices(row).into_iter().enumerate() {
            if let Some(hot) = hot {
                output[selector * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + hot as usize] += weight;
            }
        }
    }
    Ok(output)
}

/// CPU model of the two successor accumulation dispatches. The first six
/// selectors read original rows; the remaining selectors consume packed bytes.
pub fn packed_factorized_pushforward<F: Field>(
    rows: &[BooleanityRow],
    packed: &PackedAddressRows,
    e_in: &[F],
    e_out: &[F],
) -> Result<Vec<F>, BooleanityAddressSuccessorError> {
    validate_weight_shape(rows.len(), e_in.len(), e_out.len())?;
    if packed.rows() != rows.len() {
        let expected = rows
            .len()
            .checked_mul(BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
        return Err(BooleanityAddressSuccessorError::PackedStorageLength {
            expected,
            got: packed.as_bytes().len(),
        });
    }
    let fields = BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS * BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
    let mut output = vec![F::zero(); fields];
    let mut block = vec![F::zero(); fields];
    for (x_out, &outer) in e_out.iter().enumerate() {
        block.fill(F::zero());
        let row_base = x_out * e_in.len();
        for (x_in, &inner) in e_in.iter().enumerate() {
            let row_index = row_base + x_in;
            let lookup_hi = rows[row_index].words()[1];
            for selector in 0..BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS {
                let hot = ((lookup_hi >> (8 * (7 - selector))) & 0xff) as usize;
                block[selector * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + hot] += inner;
            }
            for selector in BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS
                ..BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS
            {
                if let Some(hot) = packed.hot(row_index, selector)? {
                    block[selector * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + hot as usize] += inner;
                }
            }
        }
        for (output, block) in output.iter_mut().zip(&block) {
            *output += outer * *block;
        }
    }
    Ok(output)
}

fn store_hot(bytes: &mut [u8], rows: usize, row: usize, selector: usize, hot: u8) {
    bytes[selector * rows + row] = hot;
}

fn biased_increment(magnitude: u64, packed_pc: u64) -> (u64, i8) {
    if packed_pc >> 63 != 0 {
        (
            BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS.wrapping_sub(magnitude),
            if magnitude > BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS {
                -1
            } else {
                0
            },
        )
    } else {
        let biased = BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS.wrapping_add(magnitude);
        (
            biased,
            i8::from(biased < BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS),
        )
    }
}
