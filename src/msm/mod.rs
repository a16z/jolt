/// Copy of ark_ec::VariableBaseMSM with minor modifications to speed up
/// known small element sized MSMs.
use ark_ff::{prelude::*, PrimeField};
use ark_std::{borrow::Borrow, iterable::Iterable, vec::Vec};

use ark_ec::{CurveGroup, ScalarMul};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(not(feature = "ark-msm"))]
impl<G: CurveGroup> VariableBaseMSM for G {}

pub trait VariableBaseMSM: ScalarMul {
  /// Computes an inner product between the [`PrimeField`] elements in `scalars`
  /// and the corresponding group elements in `bases`.
  ///
  /// If the elements have different length, it will chop the slices to the
  /// shortest length between `scalars.len()` and `bases.len()`.
  ///
  /// Reference: [`VariableBaseMSM::msm`]
  fn msm_unchecked(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Self {
    let bigints = ark_std::cfg_into_iter!(scalars)
      .map(|s| s.into_bigint())
      .collect::<Vec<_>>();
    Self::msm_bigint(bases, &bigints)
  }

  /// Performs multi-scalar multiplication.
  ///
  /// # Warning
  ///
  /// This method checks that `bases` and `scalars` have the same length.
  /// If they are unequal, it returns an error containing
  /// the shortest length over which the MSM can be performed.
  fn msm(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Result<Self, usize> {
    (bases.len() == scalars.len())
      .then(|| Self::msm_unchecked(bases, scalars))
      .ok_or_else(|| bases.len().min(scalars.len()))
  }

  /// Optimized implementation of multi-scalar multiplication.
  fn msm_bigint(
    bases: &[Self::MulBase],
    bigints: &[<Self::ScalarField as PrimeField>::BigInt],
  ) -> Self {
    if Self::NEGATION_IS_CHEAP {
      msm_bigint_wnaf(bases, bigints)
    } else {
      msm_bigint(bases, bigints)
    }
  }

  /// Streaming multi-scalar multiplication algorithm with hard-coded chunk
  /// size.
  fn msm_chunks<I: ?Sized, J>(bases_stream: &J, scalars_stream: &I) -> Self
  where
    I: Iterable,
    I::Item: Borrow<Self::ScalarField>,
    J: Iterable,
    J::Item: Borrow<Self::MulBase>,
  {
    assert!(scalars_stream.len() <= bases_stream.len());

    // remove offset
    let bases_init = bases_stream.iter();
    let mut scalars = scalars_stream.iter();

    // align the streams
    // TODO: change `skip` to `advance_by` once rust-lang/rust#7774 is fixed.
    // See <https://github.com/rust-lang/rust/issues/77404>
    let mut bases = bases_init.skip(bases_stream.len() - scalars_stream.len());
    let step: usize = 1 << 20;
    let mut result = Self::zero();
    for _ in 0..(scalars_stream.len() + step - 1) / step {
      let bases_step = (&mut bases)
        .take(step)
        .map(|b| *b.borrow())
        .collect::<Vec<_>>();
      let scalars_step = (&mut scalars)
        .take(step)
        .map(|s| s.borrow().into_bigint())
        .collect::<Vec<_>>();
      result += Self::msm_bigint(bases_step.as_slice(), scalars_step.as_slice());
    }
    result
  }
}

// Compute msm using windowed non-adjacent form
fn msm_bigint_wnaf<V: VariableBaseMSM>(
  bases: &[V::MulBase],
  bigints: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
  let mut max_num_bits = 1usize;
  for bigint in bigints {
    if bigint.num_bits() as usize > max_num_bits {
      max_num_bits = bigint.num_bits() as usize;
    }

    // Hack for early exit
    if max_num_bits > 60 {
      max_num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;
      break;
    }
  }

  let size = ark_std::cmp::min(bases.len(), bigints.len());
  let scalars = &bigints[..size];
  let bases = &bases[..size];

  let c = if size < 32 {
    3
  } else {
    ln_without_floats(size) + 2
  };

  let num_bits = max_num_bits;
  let digits_count = (num_bits + c - 1) / c;
  let scalar_digits = scalars
    .iter()
    .flat_map(|s| make_digits(s, c, num_bits))
    .collect::<Vec<_>>();
  let zero = V::zero();
  let window_sums: Vec<_> = ark_std::cfg_into_iter!(0..digits_count)
    .map(|i| {
      let mut buckets = vec![zero; 1 << c];
      for (digits, base) in scalar_digits.chunks(digits_count).zip(bases) {
        use ark_std::cmp::Ordering;
        // digits is the digits thing of the first scalar?
        let scalar = digits[i];
        match 0.cmp(&scalar) {
          Ordering::Less => buckets[(scalar - 1) as usize] += base,
          Ordering::Greater => buckets[(-scalar - 1) as usize] -= base,
          Ordering::Equal => (),
        }
      }

      let mut running_sum = V::zero();
      let mut res = V::zero();
      buckets.into_iter().rev().for_each(|b| {
        running_sum += &b;
        res += &running_sum;
      });
      res
    })
    .collect();

  // We store the sum for the lowest window.
  let lowest = *window_sums.first().unwrap();

  // We're traversing windows from high to low.
  lowest
    + window_sums[1..]
      .iter()
      .rev()
      .fold(zero, |mut total, sum_i| {
        total += sum_i;
        for _ in 0..c {
          total.double_in_place();
        }
        total
      })
}

/// Optimized implementation of multi-scalar multiplication.
fn msm_bigint<V: VariableBaseMSM>(
  bases: &[V::MulBase],
  bigints: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
  let size = ark_std::cmp::min(bases.len(), bigints.len());
  let scalars = &bigints[..size];
  let bases = &bases[..size];
  let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

  let c = if size < 32 {
    3
  } else {
    ln_without_floats(size) + 2
  };

  let mut max_num_bits = 1usize;
  for bigint in bigints {
    if bigint.num_bits() as usize > max_num_bits {
      max_num_bits = bigint.num_bits() as usize;
    }

    // Hack
    if max_num_bits > 60 {
      max_num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;
      break;
    }
  }

  let num_bits = max_num_bits;
  let one = V::ScalarField::one().into_bigint();

  let zero = V::zero();
  let window_starts = (0..num_bits).step_by(c);

  // Each window is of size `c`.
  // We divide up the bits 0..num_bits into windows of size `c`, and
  // in parallel process each such window.
  let window_sums: Vec<_> = window_starts
    .map(|w_start| {
      let mut res = zero;
      // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
      let mut buckets = vec![zero; (1 << c) - 1];
      // This clone is cheap, because the iterator contains just a
      // pointer and an index into the original vectors.
      scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
        if scalar == one {
          // We only process unit scalars once in the first window.
          if w_start == 0 {
            res += base;
          }
        } else {
          let mut scalar = scalar;

          // We right-shift by w_start, thus getting rid of the
          // lower bits.
          scalar.divn(w_start as u32);

          // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
          let scalar = scalar.as_ref()[0] % (1 << c);

          // If the scalar is non-zero, we update the corresponding
          // bucket.
          // (Recall that `buckets` doesn't have a zero bucket.)
          if scalar != 0 {
            buckets[(scalar - 1) as usize] += base;
          }
        }
      });

      // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
      // This is computed below for b buckets, using 2b curve additions.
      //
      // We could first normalize `buckets` and then use mixed-addition
      // here, but that's slower for the kinds of groups we care about
      // (Short Weierstrass curves and Twisted Edwards curves).
      // In the case of Short Weierstrass curves,
      // mixed addition saves ~4 field multiplications per addition.
      // However normalization (with the inversion batched) takes ~6
      // field multiplications per element,
      // hence batch normalization is a slowdown.

      // `running_sum` = sum_{j in i..num_buckets} bucket[j],
      // where we iterate backward from i = num_buckets to 0.
      let mut running_sum = V::zero();
      buckets.into_iter().rev().for_each(|b| {
        running_sum += &b;
        res += &running_sum;
      });
      res
    })
    .collect();

  // We store the sum for the lowest window.
  let lowest = *window_sums.first().unwrap();

  // We're traversing windows from high to low.
  lowest
    + window_sums[1..]
      .iter()
      .rev()
      .fold(zero, |mut total, sum_i| {
        total += sum_i;
        for _ in 0..c {
          total.double_in_place();
        }
        total
      })
}

// From: https://github.com/arkworks-rs/gemini/blob/main/src/kzg/msm/variable_base.rs#L20
fn make_digits(a: &impl BigInteger, w: usize, num_bits: usize) -> Vec<i64> {
  let scalar = a.as_ref();
  let radix: u64 = 1 << w;
  let window_mask: u64 = radix - 1;

  let mut carry = 0u64;
  let num_bits = if num_bits == 0 {
    a.num_bits() as usize
  } else {
    num_bits
  };
  let digits_count = (num_bits + w - 1) / w;
  let mut digits = vec![0i64; digits_count];
  for (i, digit) in digits.iter_mut().enumerate() {
    // Construct a buffer of bits of the scalar, starting at `bit_offset`.
    let bit_offset = i * w;
    let u64_idx = bit_offset / 64;
    let bit_idx = bit_offset % 64;
    // Read the bits from the scalar
    let bit_buf = if bit_idx < 64 - w || u64_idx == scalar.len() - 1 {
      // This window's bits are contained in a single u64,
      // or it's the last u64 anyway.
      scalar[u64_idx] >> bit_idx
    } else {
      // Combine the current u64's bits with the bits from the next u64
      (scalar[u64_idx] >> bit_idx) | (scalar[1 + u64_idx] << (64 - bit_idx))
    };

    // Read the actual coefficient value from the window
    let coef = carry + (bit_buf & window_mask); // coef = [0, 2^r)

    // Recenter coefficients from [0,2^w) to [-2^w/2, 2^w/2)
    carry = (coef + radix / 2) >> w;
    *digit = (coef as i64) - (carry << w) as i64;
  }

  digits[digits_count - 1] += (carry << w) as i64;

  digits
}

/// The result of this function is only approximately `ln(a)`
/// [`Explanation of usage`]
///
/// [`Explanation of usage`]: https://github.com/scipr-lab/zexe/issues/79#issue-556220473
fn ln_without_floats(a: usize) -> usize {
  // log2(a) * ln(2)
  (ark_std::log2(a) * 69 / 100) as usize
}
