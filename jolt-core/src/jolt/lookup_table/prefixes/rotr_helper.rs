use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RotrHelperPrefix<const WORD_SIZE: usize> {}

// TODO: (0xAndoroid) - This can be optimized quite a bit
impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for RotrHelperPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let (prod_one_minus_y, mut sum_contributions) = match checkpoints[Prefixes::RotrHelper] {
            PrefixCheckpoint::RotrHelper {
                prod_one_minus_y,
                sum_contributions,
            } => (prod_one_minus_y, sum_contributions),
            _ => (F::one(), F::zero()),
        };

        let (x, y) = if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            (r_x, y)
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            (x, y_msb)
        };

        // Update sum for current position
        // This handles the case where y_j = 1 and all y_k = 0 for k > j
        let two_inv = F::from_u32(2).inverse().unwrap();
        let trailing_zeros_count = j / 2;
        let power_inv = if trailing_zeros_count == 0 {
            F::one()
        } else {
            // Compute 1/2^trailing_zeros_power
            let mut result = F::one();
            for _ in 0..trailing_zeros_count {
                result *= two_inv;
            }
            result
        };

        sum_contributions += x * y * prod_one_minus_y * power_inv;

        // Update product and power for next iteration
        let new_prod_one_minus_y = prod_one_minus_y * (F::one() - y);
        let new_trailing_zeros_power = trailing_zeros_count + 1;

        // Process remaining bits
        let (mut x_bits, mut y_bits) = b.uninterleave();
        let mut current_prod = new_prod_one_minus_y;
        let mut current_power = new_trailing_zeros_power;

        debug_assert_eq!(x_bits.len(), y_bits.len());

        // Calculate contributions from each bit position
        while x_bits.len() > 0 {
            let x_bit = x_bits.pop_msb() as u64;
            let y_bit = y_bits.pop_msb() as u64;

            if x_bit > 0 && y_bit > 0 {
                // Compute 1/2^current_power
                let mut power_inv = F::one();
                for _ in 0..current_power {
                    power_inv *= two_inv;
                }
                sum_contributions +=
                    F::from_u64(x_bit) * F::from_u64(y_bit) * current_prod * power_inv;
            }

            current_prod *= F::one() - F::from_u64(y_bit);
            current_power += 1;
        }

        // Handle all-zeros case
        // If all y values are 0, we need to add the term for 2^(-total_bits)
        let total_bits = j / 2 + WORD_SIZE;
        let mut all_zeros_power_inv = F::one();
        for _ in 0..total_bits {
            all_zeros_power_inv *= two_inv;
        }
        sum_contributions += current_prod * all_zeros_power_inv;

        sum_contributions
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let (mut prod_one_minus_y, mut sum_contributions) = match checkpoints[Prefixes::RotrHelper]
        {
            PrefixCheckpoint::RotrHelper {
                prod_one_minus_y,
                sum_contributions,
            } => (prod_one_minus_y, sum_contributions),
            _ => (F::one(), F::zero()),
        };

        // Compute 1/2^trailing_zeros_power
        let two_inv = F::from_u32(2).inverse().unwrap();
        let mut power_inv = F::one();
        for _ in 0..(j / 2) {
            power_inv *= two_inv;
        }

        // Update sum with contribution from this position
        sum_contributions += r_x * r_y * prod_one_minus_y * power_inv;

        // Update product and increment power
        prod_one_minus_y *= F::one() - r_y;

        PrefixCheckpoint::RotrHelper {
            prod_one_minus_y,
            sum_contributions,
        }
    }
}
