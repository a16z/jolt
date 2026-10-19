use cudarc::driver::CudaFunction;
use cudarc::driver::PushKernelArg;
use jolt_field::Fr;

use super::context::CudaKernelContext;
use super::device::DeviceFrVec;
use super::error::CudaError;

impl CudaKernelContext {
    fn binary_op(
        &self,
        kernel: &CudaFunction,
        left: &DeviceFrVec,
        right: &DeviceFrVec,
    ) -> Result<DeviceFrVec, CudaError> {
        if left.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        let len = left.len();
        let mut output = self.alloc(len)?;
        if len == 0 {
            return Ok(output);
        }
        let count = Self::count_of(len)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(left.limbs());
        let _ = builder.arg(right.limbs());
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `left[i]`/`right[i]` and writes
        // only `out[i]` (4 limbs each); all three buffers hold `count * LIMBS`
        // u64s, the length equality is checked above, and `out` is a fresh
        // allocation distinct from both inputs, so no thread aliases another's
        // write. Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    fn scalar_op(
        &self,
        kernel: &CudaFunction,
        values: &DeviceFrVec,
        scalar: Fr,
    ) -> Result<DeviceFrVec, CudaError> {
        let len = values.len();
        let mut output = self.alloc(len)?;
        if len == 0 {
            return Ok(output);
        }
        let scalar = self.upload(&[scalar])?;
        let count = Self::count_of(len)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(values.limbs());
        let _ = builder.arg(scalar.limbs());
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `values[i]` plus the single-element
        // `scalar` buffer (4 limbs, read-only and shared) and writes only
        // `out[i]`; `values` and `out` hold `count * LIMBS` u64s and `out` is a
        // fresh distinct allocation. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub fn add(&self, left: &DeviceFrVec, right: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.binary_op(&self.add, left, right)
    }

    pub fn sub(&self, left: &DeviceFrVec, right: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.binary_op(&self.sub, left, right)
    }

    pub fn mul(&self, left: &DeviceFrVec, right: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.binary_op(&self.mul, left, right)
    }

    pub fn mul_scalar(&self, values: &DeviceFrVec, scalar: Fr) -> Result<DeviceFrVec, CudaError> {
        self.scalar_op(&self.mul_scalar, values, scalar)
    }

    pub fn add_scalar(&self, values: &DeviceFrVec, scalar: Fr) -> Result<DeviceFrVec, CudaError> {
        self.scalar_op(&self.add_scalar, values, scalar)
    }

    pub fn fma(
        &self,
        accumulator: &DeviceFrVec,
        left: &DeviceFrVec,
        right: &DeviceFrVec,
    ) -> Result<DeviceFrVec, CudaError> {
        if accumulator.len() != left.len() || accumulator.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: accumulator.len(),
                got: left.len().min(right.len()),
            });
        }
        let len = accumulator.len();
        let mut output = self.alloc(len)?;
        if len == 0 {
            return Ok(output);
        }
        let count = Self::count_of(len)?;
        let mut builder = self.stream().launch_builder(&self.fma);
        let _ = builder.arg(accumulator.limbs());
        let _ = builder.arg(left.limbs());
        let _ = builder.arg(right.limbs());
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `acc[i]`, `left[i]`, `right[i]` and
        // writes only `out[i]`; all four buffers hold `count * LIMBS` u64s (the
        // three input lengths are checked equal above) and `out` is a fresh
        // allocation distinct from the inputs. Threads with `i >= count` return
        // before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, Ring};
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::super::context::{shared_context, CudaKernelContext};
    use super::super::device::DeviceFrVec;

    fn device() -> Option<&'static CudaKernelContext> {
        shared_context()
    }

    fn arb_fr() -> impl Strategy<Value = Fr> {
        any::<[u64; 4]>().prop_map(|limbs| {
            let mut bytes = [0u8; 32];
            for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
                chunk.copy_from_slice(&limb.to_le_bytes());
            }
            <ark_bn254::Fr as ark_ff::PrimeField>::from_le_bytes_mod_order(&bytes).into()
        })
    }

    fn arb_frs(min: usize, max: usize) -> impl Strategy<Value = Vec<Fr>> {
        vec(arb_fr(), min..=max)
    }

    fn upload(context: &CudaKernelContext, values: &[Fr]) -> DeviceFrVec {
        context.upload(values).expect("upload")
    }

    proptest! {
        #[test]
        fn add_matches_cpu(values in arb_frs(1, 300).prop_flat_map(|left| {
            let len = left.len();
            (Just(left), vec(arb_fr(), len))
        })) {
            let Some(context) = device() else { return Ok(()); };
            let (left, right) = values;
            let expected: Vec<Fr> = left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got = context
                .add(&upload(context, &left), &upload(context, &right))
                .expect("device add")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn sub_matches_cpu(values in arb_frs(1, 300).prop_flat_map(|left| {
            let len = left.len();
            (Just(left), vec(arb_fr(), len))
        })) {
            let Some(context) = device() else { return Ok(()); };
            let (left, right) = values;
            let expected: Vec<Fr> = left.iter().zip(&right).map(|(&a, &b)| a - b).collect();
            let got = context
                .sub(&upload(context, &left), &upload(context, &right))
                .expect("device sub")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn mul_matches_cpu(values in arb_frs(1, 300).prop_flat_map(|left| {
            let len = left.len();
            (Just(left), vec(arb_fr(), len))
        })) {
            let Some(context) = device() else { return Ok(()); };
            let (left, right) = values;
            let expected: Vec<Fr> = left.iter().zip(&right).map(|(&a, &b)| a * b).collect();
            let got = context
                .mul(&upload(context, &left), &upload(context, &right))
                .expect("device mul")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn mul_scalar_matches_cpu(values in arb_frs(1, 300), scalar in arb_fr()) {
            let Some(context) = device() else { return Ok(()); };
            let expected: Vec<Fr> = values.iter().map(|&value| value * scalar).collect();
            let got = context
                .mul_scalar(&upload(context, &values), scalar)
                .expect("device mul_scalar")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn add_scalar_matches_cpu(values in arb_frs(1, 300), scalar in arb_fr()) {
            let Some(context) = device() else { return Ok(()); };
            let expected: Vec<Fr> = values.iter().map(|&value| value + scalar).collect();
            let got = context
                .add_scalar(&upload(context, &values), scalar)
                .expect("device add_scalar")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fma_matches_cpu(values in arb_frs(1, 300).prop_flat_map(|accumulator| {
            let len = accumulator.len();
            (Just(accumulator), vec(arb_fr(), len), vec(arb_fr(), len))
        })) {
            let Some(context) = device() else { return Ok(()); };
            let (accumulator, left, right) = values;
            let expected: Vec<Fr> = accumulator
                .iter()
                .zip(&left)
                .zip(&right)
                .map(|((&acc, &a), &b)| acc + a * b)
                .collect();
            let got = context
                .fma(
                    &upload(context, &accumulator),
                    &upload(context, &left),
                    &upload(context, &right),
                )
                .expect("device fma")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }
    }

    fn boundary_values() -> Vec<Fr> {
        let modulus_minus_one = Fr::from_u64(0) - Fr::from_u64(1);
        vec![
            Fr::from_u64(0),
            Fr::from_u64(1),
            modulus_minus_one,
            modulus_minus_one - Fr::from_u64(1),
            Fr::from_u64(u64::MAX),
            Fr::from_u64(2),
        ]
    }

    #[test]
    fn arithmetic_matches_cpu_at_modulus_boundaries() {
        let Some(context) = device() else {
            return;
        };
        let values = boundary_values();
        let pairs: Vec<(Fr, Fr)> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| (a, b)))
            .collect();
        let left: Vec<Fr> = pairs.iter().map(|&(a, _)| a).collect();
        let right: Vec<Fr> = pairs.iter().map(|&(_, b)| b).collect();
        let device_left = upload(context, &left);
        let device_right = upload(context, &right);

        let expected_add: Vec<Fr> = pairs.iter().map(|&(a, b)| a + b).collect();
        let expected_sub: Vec<Fr> = pairs.iter().map(|&(a, b)| a - b).collect();
        let expected_mul: Vec<Fr> = pairs.iter().map(|&(a, b)| a * b).collect();

        assert_eq!(
            context
                .add(&device_left, &device_right)
                .expect("add")
                .to_host()
                .expect("download"),
            expected_add,
        );
        assert_eq!(
            context
                .sub(&device_left, &device_right)
                .expect("sub")
                .to_host()
                .expect("download"),
            expected_sub,
        );
        assert_eq!(
            context
                .mul(&device_left, &device_right)
                .expect("mul")
                .to_host()
                .expect("download"),
            expected_mul,
        );
    }
}
