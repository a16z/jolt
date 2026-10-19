//! BN254 base-field arithmetic used by device curve operations.
//!
//! Inputs and outputs use four Montgomery limbs per field element. Each GPU
//! thread inverts a chunk using its output range for prefix products; zero
//! entries remain zero.

use cudarc::driver::{CudaFunction, CudaSlice, PushKernelArg};

use super::context::CudaKernelContext;
use super::error::CudaError;

pub const FQ_LIMBS: usize = 4;

const BATCH_INVERSE_TARGET_THREADS: usize = 8192;

const BATCH_INVERSE_MAX_CHUNK: usize = 128;

fn take_limbs(chunk: &[u64]) -> [u64; FQ_LIMBS] {
    let mut limbs = [0u64; FQ_LIMBS];
    limbs.copy_from_slice(chunk);
    limbs
}

impl CudaKernelContext {
    fn fq_binary(
        &self,
        kernel: &CudaFunction,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        if left.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(left.as_flattened())?;
        let device_right = self.upload_u64_slice(right.as_flattened())?;
        let mut output = self.alloc_u64(left.len() * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `left[i]`/`right[i]` and writes
        // only `out[i]` (4 limbs each); all three buffers hold
        // `count * FQ_LIMBS` u64s, the length equality is checked above, and
        // `out` is a fresh allocation distinct from both inputs. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(self
            .download_u64(&output)?
            .chunks_exact(FQ_LIMBS)
            .map(take_limbs)
            .collect())
    }

    pub fn fq_add(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_add(), left, right)
    }

    pub fn fq_sub(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_sub(), left, right)
    }

    pub fn fq_mul(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_mul(), left, right)
    }

    pub fn fq_batch_inverse(
        &self,
        values: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let device = self.upload_u64_slice(values.as_flattened())?;
        let mut output = self.alloc_u64(values.len() * FQ_LIMBS)?;
        self.launch_batch_inverse(&device, &mut output, values.len())?;
        Ok(self
            .download_u64(&output)?
            .chunks_exact(FQ_LIMBS)
            .map(take_limbs)
            .collect())
    }

    pub(crate) fn launch_batch_inverse(
        &self,
        values: &CudaSlice<u64>,
        output: &mut CudaSlice<u64>,
        len: usize,
    ) -> Result<(), CudaError> {
        if len == 0 {
            return Ok(());
        }
        let chunk = len
            .div_ceil(BATCH_INVERSE_TARGET_THREADS)
            .clamp(1, BATCH_INVERSE_MAX_CHUNK);
        let threads = Self::count_of(len.div_ceil(chunk))?;
        let count = Self::count_of(len)?;
        let chunk = Self::count_of(chunk)?;
        let mut builder = self.stream().launch_builder(self.msm_fq_batch_inverse());
        let _ = builder.arg(values);
        let _ = builder.arg(output);
        let _ = builder.arg(&count);
        let _ = builder.arg(&chunk);
        // SAFETY: thread `t` owns the half-open element range
        // `[t * chunk, min((t+1) * chunk, count))`, disjoint across threads and
        // clipped to `count`. It reads only `values` and read-modify-writes
        // only `out` inside that range (4 limbs per element; both buffers hold
        // `count * FQ_LIMBS` u64s), using `out` as its own prefix-product
        // scratch. `out` is a distinct allocation from `values`. Threads whose
        // range starts at or past `count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(threads)) }?;
        self.stream().synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use ark_bn254::Fq;
    use ark_ff::{BigInt, PrimeField};
    use proptest::prelude::*;

    use super::FQ_LIMBS;

    pub fn fq_from_limbs(limbs: [u64; FQ_LIMBS]) -> Fq {
        Fq::new_unchecked(BigInt(limbs))
    }

    pub fn fq_limbs(value: Fq) -> [u64; FQ_LIMBS] {
        value.0 .0
    }

    pub fn arb_fq() -> impl Strategy<Value = Fq> {
        any::<[u64; FQ_LIMBS]>().prop_map(|limbs| {
            let mut bytes = [0u8; 32];
            for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
                chunk.copy_from_slice(&limb.to_le_bytes());
            }
            Fq::from_le_bytes_mod_order(&bytes)
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::Fq;
    use ark_ff::{PrimeField, Zero};
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::testing::{arb_fq, fq_from_limbs, fq_limbs};
    use super::FQ_LIMBS;
    use crate::cuda::common::context::{shared_context, CudaKernelContext};

    fn device() -> Option<&'static CudaKernelContext> {
        shared_context()
    }

    fn limbs(values: &[Fq]) -> Vec<[u64; FQ_LIMBS]> {
        values.iter().copied().map(fq_limbs).collect()
    }

    #[test]
    fn fq_arithmetic_carries_across_limb_and_modulus_boundaries() {
        let Some(context) = device() else {
            return;
        };
        let mut values = vec![Fq::zero(), fq_from_limbs([1, 0, 0, 0])];
        let mut near_modulus = Fq::MODULUS.0;
        near_modulus[0] -= 1;
        values.push(fq_from_limbs(near_modulus));
        for end in 1..FQ_LIMBS {
            let mut below = [0; FQ_LIMBS];
            below[..end].fill(u64::MAX);
            let mut above = [0; FQ_LIMBS];
            above[end] = 1;
            values.extend([fq_from_limbs(below), fq_from_limbs(above)]);
        }
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| (a, b)))
            .collect();
        let left: Vec<_> = pairs.iter().map(|&(a, _)| fq_limbs(a)).collect();
        let right: Vec<_> = pairs.iter().map(|&(_, b)| fq_limbs(b)).collect();
        for (actual, expected) in [
            (
                context.fq_add(&left, &right).expect("add"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a + b))
                    .collect::<Vec<_>>(),
            ),
            (
                context.fq_sub(&left, &right).expect("sub"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a - b))
                    .collect::<Vec<_>>(),
            ),
            (
                context.fq_mul(&left, &right).expect("mul"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a * b))
                    .collect::<Vec<_>>(),
            ),
        ] {
            assert_eq!(actual, expected);
        }
    }

    proptest! {
        #[test]
        fn fq_add_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got: Vec<_> = context
                .fq_add(&limbs(&left), &limbs(&right))
                .expect("device fq_add")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_sub_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a - b).collect();
            let got: Vec<_> = context
                .fq_sub(&limbs(&left), &limbs(&right))
                .expect("device fq_sub")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_batch_inverse_matches_arkworks(values in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let mut values = values;
            for (index, value) in values.iter_mut().enumerate() {
                if index % 7 == 0 {
                    *value = ark_bn254::Fq::zero();
                }
            }
            prop_assert!(
                values.iter().any(|value| value.is_zero()),
                "no zero in the input, so the pass-through-zero convention is untested",
            );
            let mut expected = values.clone();
            ark_ff::fields::batch_inversion(&mut expected);
            let got: Vec<_> = context
                .fq_batch_inverse(&limbs(&values))
                .expect("device fq_batch_inverse")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_mul_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a * b).collect();
            let got: Vec<_> = context
                .fq_mul(&limbs(&left), &limbs(&right))
                .expect("device fq_mul")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }
    }
}
