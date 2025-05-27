use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RotrPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for RotrPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let (prod_one_plus_y, mut first_sum, mut second_sum) = match checkpoints[Prefixes::Rotr] {
            PrefixCheckpoint::Rotr {
                prod_one_plus_y,
                first_sum,
                second_sum,
            } => (prod_one_plus_y, first_sum, second_sum),
            _ => (F::one(), F::zero(), F::zero()),
        };

        let (x, y) = if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            (r_x, y)
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            (x, y_msb)
        };

        first_sum *= F::one() + y;
        first_sum += x * y;
        second_sum +=
            x * (F::one() - y) * prod_one_plus_y * F::from_u64(1 << (WORD_SIZE - j / 2 - 1));

        // Do the remaining elements of sum
        let (x_b, y_b) = b.uninterleave();
        debug_assert_eq!(x_b.len(), y_b.len());
        let len = x_b.len();
        // We can remove trailing zeroes of y since they are useless in the formula
        let trailing_zeroes = y_b.trailing_zeros();
        let (x, y) = (
            u32::from(x_b) >> trailing_zeroes,
            u32::from(y_b) >> trailing_zeroes,
        );
        let xy = x & y;
        // Since we removed all zeroes
        let remaining_len = len - trailing_zeroes as usize;
        first_sum *= F::from_u64(1 << remaining_len);
        first_sum += F::from_u64(xy as u64);

        // Calculate second sum
        // remove leading ones of y and the same amount from x
        let mask = !((1 << remaining_len) - 1);
        let x = u32::from(x_b) & mask;
        let remaining_terms = x.unbounded_shl(WORD_SIZE as u32 - trailing_zeroes);
        second_sum += F::from_u64(remaining_terms as u64);

        first_sum + second_sum
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let (mut prod_one_plus_y, mut first_sum, mut second_sum) = match checkpoints[Prefixes::Rotr]
        {
            PrefixCheckpoint::Rotr {
                prod_one_plus_y,
                first_sum,
                second_sum,
            } => (prod_one_plus_y, first_sum, second_sum),
            _ => (F::one(), F::zero(), F::zero()),
        };

        first_sum *= F::one() + r_y;
        first_sum += r_x * r_y;
        second_sum +=
            r_x * (F::one() - r_y) * prod_one_plus_y * F::from_u64(1 << (WORD_SIZE - 1 - j / 2));
        prod_one_plus_y *= F::one() + r_y;
        PrefixCheckpoint::Rotr {
            prod_one_plus_y,
            first_sum,
            second_sum,
        }
    }
}
