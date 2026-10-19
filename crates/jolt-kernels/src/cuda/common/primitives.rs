use cudarc::driver::CudaFunction;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Fr, Ring};
use jolt_poly::BindingOrder;

use super::context::{CudaKernelContext, BLOCK as BLOCK_SIZE};
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

    pub fn bind(
        &self,
        values: &DeviceFrVec,
        challenge: Fr,
        order: BindingOrder,
    ) -> Result<DeviceFrVec, CudaError> {
        let len = values.len();
        if len < 2 || !len.is_power_of_two() {
            return Err(CudaError::LengthMismatch {
                expected: len.next_power_of_two().max(2),
                got: len,
            });
        }
        let half = len / 2;
        let mut output = self.alloc(half)?;
        let limbs = crate::cuda::common::device::fr_limbs(challenge);
        let count = Self::count_of(half)?;
        let kernel = match order {
            BindingOrder::LowToHigh => &self.bind_low_to_high,
            BindingOrder::HighToLow => &self.bind_high_to_low,
        };
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(values.limbs());
        let _ = builder.arg(&limbs[0]);
        let _ = builder.arg(&limbs[1]);
        let _ = builder.arg(&limbs[2]);
        let _ = builder.arg(&limbs[3]);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < half` writes only `out[i]` and reads the pair
        // (`in[2i]`,`in[2i+1]`) for LowToHigh or (`in[i]`,`in[i+half]`) for
        // HighToLow — both within `in`'s `2 * half * LIMBS` u64s. The challenge
        // arrives as four by-value limbs, so no device buffer backs it. `out`
        // holds `half * LIMBS` u64s and is a fresh allocation distinct from `in`,
        // so the reads cannot observe a partially written `out`. Threads with
        // `i >= half` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub fn sum(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        if values.is_empty() {
            return Ok(Fr::default());
        }
        let mut current = values.try_clone()?;
        loop {
            let count = Self::count_of(current.len())?;
            let blocks = count.div_ceil(BLOCK_SIZE);
            let mut partials = self.alloc(blocks as usize)?;
            let mut builder = self.stream().launch_builder(&self.sum_reduce);
            let _ = builder.arg(current.limbs());
            let _ = builder.arg(partials.limbs_mut());
            let _ = builder.arg(&count);
            // SAFETY: each block reduces its own `BLOCK_SIZE`-element window of
            // `current` (holding `count * LIMBS` u64s; out-of-range lanes seed
            // the shared-memory tree with field zero) into shared memory sized
            // `BLOCK_SIZE * LIMBS` u64s by the kernel's own declaration, with
            // `__syncthreads()` between every tree level. Only thread 0 writes,
            // to `partials[blockIdx.x]`, and `partials` holds exactly `blocks`
            // elements, so writes are one-per-block and non-aliasing.
            let _ = unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            self.stream().synchronize()?;
            if blocks == 1 {
                return partials.first();
            }
            current = partials;
        }
    }

    pub fn eq_evals(&self, point: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let mut table = self.upload(&[Fr::from_u64(1)])?;
        for &coordinate in point {
            let prev_len = table.len();
            let mut doubled = self.alloc(prev_len * 2)?;
            let coordinate = crate::cuda::common::device::fr_limbs(coordinate);
            let count = Self::count_of(prev_len)?;
            let mut builder = self.stream().launch_builder(&self.eq_double);
            let _ = builder.arg(table.limbs());
            for limb in &coordinate {
                let _ = builder.arg(limb);
            }
            let _ = builder.arg(doubled.limbs_mut());
            let _ = builder.arg(&count);
            // SAFETY: thread `j < prev_len` reads only `table[j]`, the
            // coordinate arriving as four by-value limbs, and writes exactly
            // `doubled[2j]` and `doubled[2j+1]` — disjoint across threads.
            // `table` holds `prev_len * LIMBS` u64s and `doubled` holds
            // `2 * prev_len * LIMBS`, a fresh allocation distinct from `table`
            // (the ping-pong is what makes this race-free, unlike the in-place
            // CPU form). Threads with `j >= prev_len` return first.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
            table = doubled;
        }
        Ok(table)
    }

    pub fn eq_evals_shard(
        &self,
        point: &[Fr],
        shard: usize,
        shards: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        if shards <= 1 {
            return self.eq_evals(point);
        }
        if !shards.is_power_of_two() || shard >= shards {
            return Err(CudaError::InvariantViolation {
                reason: "an eq shard needs a power-of-two shard count and an in-range index",
            });
        }
        let leading = shards.trailing_zeros() as usize;
        if point.len() < leading {
            return Err(CudaError::InvariantViolation {
                reason: "an eq shard split more variables than the point has",
            });
        }
        let (high, low) = point.split_at(leading);
        let mut scale = Fr::from_u64(1);
        for (index, &coordinate) in high.iter().enumerate() {
            let bit = (shard >> (leading - 1 - index)) & 1;
            scale *= if bit == 1 {
                coordinate
            } else {
                Fr::from_u64(1) - coordinate
            };
        }
        let table = self.eq_evals(low)?;
        self.mul_scalar(&table, scale)
    }

    pub fn lt_evals(&self, point: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let mut table = self.alloc(1usize << point.len())?;
        for (level, &coordinate) in point.iter().rev().enumerate() {
            let half = 1usize << level;
            let coordinate = crate::cuda::common::device::fr_limbs(coordinate);
            let count = Self::count_of(half)?;
            let mut builder = self.stream().launch_builder(&self.lt_double);
            let _ = builder.arg(table.limbs_mut());
            for limb in &coordinate {
                let _ = builder.arg(limb);
            }
            let _ = builder.arg(&count);
            // SAFETY: this kernel is deliberately in-place, mirroring the CPU
            // `lt_evals`. Thread `j < half` reads `table[j]` once and then writes
            // exactly `table[j]` and `table[j + half]`; across threads those
            // index sets are pairwise disjoint (`j` ranges over `[0, half)` and
            // `j + half` over `[half, 2*half)`), so no thread reads a location
            // another writes. `table` holds `2^point.len() * LIMBS` u64s and
            // `2 * half` never exceeds that. Threads with `j >= half` return
            // first. The launches are sequential on one stream (one per level),
            // which is what orders the levels.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        }
        Ok(table)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, Ring};
    use jolt_poly::{BindingOrder, EqPolynomial, LtPolynomial, Polynomial};
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

    fn arb_frs_pow2(max_log: usize) -> impl Strategy<Value = Vec<Fr>> {
        (1usize..=max_log).prop_flat_map(|log| vec(arb_fr(), 1usize << log))
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

        #[test]
        fn bind_low_to_high_matches_cpu(values in arb_frs_pow2(9), challenge in arb_fr()) {
            let Some(context) = device() else { return Ok(()); };
            let mut expected = Polynomial::new(values.clone());
            expected.bind_with_order(challenge, BindingOrder::LowToHigh);
            let got = context
                .bind(&upload(context, &values), challenge, BindingOrder::LowToHigh)
                .expect("device bind")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected.evals().to_vec());
        }

        #[test]
        fn bind_high_to_low_matches_cpu(values in arb_frs_pow2(9), challenge in arb_fr()) {
            let Some(context) = device() else { return Ok(()); };
            let mut expected = Polynomial::new(values.clone());
            expected.bind_with_order(challenge, BindingOrder::HighToLow);
            let got = context
                .bind(&upload(context, &values), challenge, BindingOrder::HighToLow)
                .expect("device bind")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected.evals().to_vec());
        }

        #[test]
        fn sum_matches_cpu(values in arb_frs(1, 2000)) {
            let Some(context) = device() else { return Ok(()); };
            let expected: Fr = values.iter().copied().sum();
            let got = context.sum(&upload(context, &values)).expect("device sum");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn eq_evals_shard_matches_the_whole_table_window(point in arb_frs(4, 10)) {
            let Some(context) = device() else { return Ok(()); };
            let whole = context.eq_evals(&point).expect("device eq_evals")
                .to_host().expect("download");
            for shards in [2usize, 4, 8] {
                let len = whole.len() / shards;
                for shard in 0..shards {
                    let got = context
                        .eq_evals_shard(&point, shard, shards)
                        .expect("device eq_evals_shard")
                        .to_host()
                        .expect("download");
                    prop_assert_eq!(
                        got,
                        whole[shard * len..(shard + 1) * len].to_vec(),
                        "eq shard {} of {} must equal the whole table's window: point[0] drives \
                         the index MSB, so a shard is the sub-table over the trailing coordinates \
                         scaled by the leading ones",
                        shard,
                        shards,
                    );
                }
            }
        }

        #[test]
        fn eq_evals_matches_cpu(point in arb_frs(1, 10)) {
            let Some(context) = device() else { return Ok(()); };
            let expected = EqPolynomial::new(point.clone()).evaluations();
            let got = context
                .eq_evals(&point)
                .expect("device eq_evals")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn lt_evals_matches_cpu(point in arb_frs(1, 10)) {
            let Some(context) = device() else { return Ok(()); };
            let expected = LtPolynomial::evaluations(&point);
            let got = context
                .lt_evals(&point)
                .expect("device lt_evals")
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

    #[test]
    fn reduction_matches_cpu_across_block_boundaries() {
        let Some(context) = device() else {
            return;
        };
        for len in [
            1usize, 2, 255, 256, 257, 511, 512, 513, 65_535, 65_536, 65_537,
        ] {
            let values: Vec<Fr> = (0..len as u64).map(|i| Fr::from_u64(i * 31 + 7)).collect();
            let expected: Fr = values.iter().copied().sum();
            assert_eq!(
                context.sum(&upload(context, &values)).expect("sum"),
                expected,
                "sum diverged at len {len}",
            );
        }
    }

    #[test]
    fn repeated_binding_matches_cpu_to_a_single_element() {
        let Some(context) = device() else {
            return;
        };
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let values: Vec<Fr> = (0..64u64).map(|i| Fr::from_u64(i * 17 + 5)).collect();
            let challenges: Vec<Fr> = (0..6u64).map(|i| Fr::from_u64(101 + 7 * i)).collect();

            let mut expected = Polynomial::new(values.clone());
            let mut device_table = upload(context, &values);
            for &challenge in &challenges {
                expected.bind_with_order(challenge, order);
                device_table = context.bind(&device_table, challenge, order).expect("bind");
                assert_eq!(
                    device_table.to_host().expect("download"),
                    expected.evals().to_vec(),
                    "bind diverged mid-loop for {order:?}",
                );
            }
            assert_eq!(device_table.len(), 1);
        }
    }
}
