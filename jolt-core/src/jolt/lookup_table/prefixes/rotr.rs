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
        let (prod_one_plus_y, mut sum_x_y_prod, mut second_sum) = match checkpoints[Prefixes::Rotr]
        {
            PrefixCheckpoint::Rotr {
                prod_one_plus_y,
                sum_x_y_prod,
                second_sum,
            } => (prod_one_plus_y, sum_x_y_prod, second_sum),
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

        sum_x_y_prod += x * y * prod_one_plus_y;
        second_sum *= F::one() + y;
        let power_2 = if j / 2 == 0 {
            F::one()
        } else {
            F::from_u32(1 << ((j / 2) - 1))
        };
        second_sum += x * (F::one() - y) * power_2;

        let (mut x, mut y) = b.uninterleave();
        let mut prod_one_plus_y_bin = 1;
        debug_assert_eq!(x.len(), y.len());
        while x.len() > 0 {
            let x = x.pop_msb() as u64;
            let y = y.pop_msb() as u64;
            sum_x_y_prod += F::from_u64(x * y * prod_one_plus_y_bin) * prod_one_plus_y;
            prod_one_plus_y_bin *= 1 + y as u64;
        }

        let (mut x, mut y) = b.uninterleave();
        // (1 + y_i)
        // (1 + y_i) * (1 + y_i+1)
        // (1 + y_i) * (1 + y_i+1) * (1 + y_i+2)
        // ...
        let mut one_plus_y = vec![1u64; y.len()];
        while y.len() > 0 {
            let i = x.len() - y.len();
            let y = y.pop_msb() as u64;
            (i..one_plus_y.len()).for_each(|j| {
                one_plus_y[j] *= 1 + y;
            });
        }

        if let Some(last) = one_plus_y.last() {
            second_sum *= F::from_u64(*last);
        }

        let mut second_sum_bin = 0;
        let start_len = x.len();
        while x.len() > 0 {
            let x_msb = x.pop_msb() as u64;
            let pow_2 = 1 << (j / 2 + (start_len - x.len()) - 1);
            second_sum_bin += x_msb * one_plus_y.pop().unwrap() as u64 * pow_2;
        }

        second_sum += F::from_u64(second_sum_bin);

        sum_x_y_prod + second_sum
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let (mut prod_one_plus_y, mut sum_x_y_prod, mut second_sum) =
            match checkpoints[Prefixes::Rotr] {
                PrefixCheckpoint::Rotr {
                    prod_one_plus_y,
                    sum_x_y_prod,
                    second_sum,
                } => (prod_one_plus_y, sum_x_y_prod, second_sum),
                _ => (F::one(), F::zero(), F::zero()),
            };
        sum_x_y_prod += r_x * r_y * prod_one_plus_y;
        let pow_2 = if j / 2 == 0 {
            F::one()
        } else {
            F::from_u32(1 << ((j / 2) - 1))
        };
        second_sum += r_x * (F::one() - r_y) * pow_2;
        prod_one_plus_y *= F::one() + r_y;
        PrefixCheckpoint::Rotr {
            prod_one_plus_y,
            sum_x_y_prod,
            second_sum,
        }
    }
}
