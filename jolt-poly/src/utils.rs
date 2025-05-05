use jolt_field::JoltField;
use rayon::prelude::*;

pub trait Math {
    fn square_root(self) -> usize;
    fn pow2(self) -> usize;
    fn get_bits(self, num_bits: usize) -> Vec<bool>;
    fn log_2(self) -> usize;
    fn num_bits(self) -> usize;
}

impl Math for usize {
    #[inline]
    fn square_root(self) -> usize {
        (self as f64).sqrt() as usize
    }

    #[inline]
    fn pow2(self) -> usize {
        let base: usize = 2;
        base.pow(self as u32)
    }

    /// Returns the num_bits from n in a canonical order
    fn get_bits(self, num_bits: usize) -> Vec<bool> {
        (0..num_bits)
            .map(|shift_amount| ((self & (1 << (num_bits - shift_amount - 1))) > 0))
            .collect::<Vec<bool>>()
    }

    fn log_2(self) -> usize {
        assert_ne!(self, 0);

        if self.is_power_of_two() {
            (1usize.leading_zeros() - self.leading_zeros()) as usize
        } else {
            (0usize.leading_zeros() - self.leading_zeros()) as usize
        }
    }

    fn num_bits(self) -> usize {
        (0usize.leading_zeros() - self.leading_zeros()) as usize
    }
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i))
        .sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: JoltField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && num.is_power_of_two()
}
