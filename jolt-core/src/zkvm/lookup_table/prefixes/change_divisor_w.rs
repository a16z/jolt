use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorWPrefix<const XLEN: usize> {}

/// Calculates the prefix for the change_divisor_w instruction
/// Equivalently, it's a (2 - 2^XLEN) * eq(x, 100...000) * eq(y, 111...111)
/// where x and y are the lower word parts of operands
impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for ChangeDivisorWPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN {
            return F::zero();
        }
        // println!("Hi: j is: {} checkpoint is: {:?}", j, checkpoints[Prefixes::ChangeDivisorW]);

        let mut result = if (j == XLEN || j == XLEN + 1) {
            F::from_u64(2) - F::from_u128(1u128 << XLEN)
        }
        else {
            checkpoints[Prefixes::ChangeDivisorW].unwrap()
        };

        // let mut result = checkpoints[Prefixes::ChangeDivisorW]
        //     .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN));
        if j == XLEN {
            let x_msb = b.pop_msb() as u32;
            if x_msb == 0 {
                return F::zero();
            }
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result = result.mul_u64(c as u64);
        } else if let Some(r_x) = r_x {
            if j > XLEN {
                let (x, y) = b.uninterleave();
                // println!("ChangeDivisorW prefix: c: {}, x: {}, y: {}", c, x, y);
                if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 || c == 0 {
                    return F::zero();
                }
                
                if j == XLEN + 1 {
                    // println!("HHHHHSAA");
                    result *= (r_x) * F::from_u64(c as u64);
                }
                else {
                    result *= (F::one() - r_x) * F::from_u64(c as u64);
                }
            }
        } else if j > XLEN {
            let (x, y) = b.uninterleave();
            // println!("b value is: {:?}", b.uninterleave());
            if b.len() > 0 && u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            // println!("Passs: {}", result);
            result *= F::one() - F::from_u64(c as u64);
        }
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN {
            return Some(F::zero()).into();
        }
        // println!("Updating j...: {j}, {:?}, {:?}", r_x.into(), r_y.into());

        let updated = if j == XLEN + 1 {
            (F::from_u64(2) - F::from_u128(1u128 << XLEN)) * r_x * r_y
        }
        else {
            checkpoints[Prefixes::ChangeDivisorW].unwrap() * ((F::one() - r_x) * r_y)
        };

        // let updated = checkpoints[Prefixes::ChangeDivisorW]
        //     .unwrap_or(F::from_u64(2) - F::from_u128(1u128 << XLEN))
        //     * if j == XLEN + 1 {
        //         r_x * r_y
        //     } else if j > XLEN + 1 {
        //         (F::one() - r_x) * r_y
        //     } else {
        //         F::one()
        //     };
        // println!("updated is: {}", updated);
        Some(updated).into()
    }
}
