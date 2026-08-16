use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::BindingOrder;

use super::context::{CudaKernelContext, BLOCK as BLOCK_SIZE};
use super::device::DeviceFrVec;
use super::error::CudaError;

impl CudaKernelContext {
    fn binary_op(
        &self,
        kernel: &cudarc::driver::CudaFunction,
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
        kernel: &cudarc::driver::CudaFunction,
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

    pub fn bind_rows(
        &self,
        values: &DeviceFrVec,
        row_len: usize,
        challenge: Fr,
    ) -> Result<DeviceFrVec, CudaError> {
        if row_len < 2 || !row_len.is_power_of_two() || !values.len().is_multiple_of(row_len) {
            return Err(CudaError::LengthMismatch {
                expected: row_len.next_power_of_two().max(2),
                got: values.len(),
            });
        }
        let half = values.len() / 2;
        let mut output = self.alloc(half)?;
        let limbs = crate::cuda::common::device::fr_limbs(challenge);
        let count = Self::count_of(half)?;
        let mut builder = self.stream().launch_builder(&self.bind_low_to_high);
        let _ = builder.arg(values.limbs());
        let _ = builder.arg(&limbs[0]);
        let _ = builder.arg(&limbs[1]);
        let _ = builder.arg(&limbs[2]);
        let _ = builder.arg(&limbs[3]);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < half` writes only `out[i]` and reads the pair
        // (`in[2i]`,`in[2i+1]`), both within `in`'s `2 * half * LIMBS` u64s. `row_len`
        // is an even divisor of `in`'s length, so no pair straddles a row boundary and
        // this is exactly a per-row low-to-high bind whose result is contiguous with
        // stride `row_len / 2`. The challenge arrives as four by-value limbs, so no
        // device buffer backs it. `out` is a fresh allocation distinct from `in`, so
        // the reads cannot observe a partially written `out`.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
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
                builder.launch(cudarc::driver::LaunchConfig {
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
            let coordinate = self.upload(&[coordinate])?;
            let count = Self::count_of(prev_len)?;
            let mut builder = self.stream().launch_builder(&self.eq_double);
            let _ = builder.arg(table.limbs());
            let _ = builder.arg(coordinate.limbs());
            let _ = builder.arg(doubled.limbs_mut());
            let _ = builder.arg(&count);
            // SAFETY: thread `j < prev_len` reads only `table[j]` and the
            // single-element coordinate buffer, and writes exactly
            // `doubled[2j]` and `doubled[2j+1]` — disjoint across threads.
            // `table` holds `prev_len * LIMBS` u64s and `doubled` holds
            // `2 * prev_len * LIMBS`, a fresh allocation distinct from `table`
            // (the ping-pong is what makes this race-free, unlike the in-place
            // CPU form). Threads with `j >= prev_len` return first.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
            self.stream().synchronize()?;
            table = doubled;
        }
        Ok(table)
    }

    pub fn lt_evals(&self, point: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let mut table = self.alloc(1usize << point.len())?;
        for (level, &coordinate) in point.iter().rev().enumerate() {
            let half = 1usize << level;
            let coordinate = self.upload(&[coordinate])?;
            let count = Self::count_of(half)?;
            let mut builder = self.stream().launch_builder(&self.lt_double);
            let _ = builder.arg(table.limbs_mut());
            let _ = builder.arg(coordinate.limbs());
            let _ = builder.arg(&count);
            // SAFETY: this kernel is deliberately in-place, mirroring the CPU
            // `lt_evals`. Thread `j < half` reads `table[j]` once and then writes
            // exactly `table[j]` and `table[j + half]`; across threads those
            // index sets are pairwise disjoint (`j` ranges over `[0, half)` and
            // `j + half` over `[half, 2*half)`), so no thread reads a location
            // another writes. `table` holds `2^point.len() * LIMBS` u64s and
            // `2 * half` never exceeds that. Threads with `j >= half` return
            // first. The launches are sequential (one per level, each followed by
            // a synchronize), which is what orders the levels.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
            self.stream().synchronize()?;
        }
        Ok(table)
    }

    pub fn u64_to_montgomery(&self, values: &[u64]) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(values.len())?;
        if values.is_empty() {
            return Ok(output);
        }
        let input = self.upload_u64_slice(values)?;
        let count = Self::count_of(values.len())?;
        let mut builder = self.stream().launch_builder(&self.u64_to_mont);
        let _ = builder.arg(&input);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `input[i]` (one u64 of a
        // `count`-element buffer) and writes only `out[i*4..i*4+4]` of a
        // `count * LIMBS` buffer; the two are distinct allocations. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub fn i128_to_montgomery(&self, values: &[i128]) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(values.len())?;
        if values.is_empty() {
            return Ok(output);
        }
        let mut magnitude = Vec::with_capacity(values.len() * 2);
        let mut negative = Vec::with_capacity(values.len());
        for &value in values {
            let unsigned = value.unsigned_abs();
            magnitude.push(unsigned as u64);
            magnitude.push((unsigned >> 64) as u64);
            negative.push(u8::from(value.is_negative()));
        }
        let magnitude = self.upload_u64_slice(&magnitude)?;
        let negative = self.upload_u8_slice(&negative)?;
        let count = Self::count_of(values.len())?;
        let mut builder = self.stream().launch_builder(&self.i128_to_mont);
        let _ = builder.arg(&magnitude);
        let _ = builder.arg(&negative);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `magnitude[2i]`, `magnitude[2i+1]`
        // (a `2 * count`-element buffer) and `negative[i]` (a `count`-element
        // buffer), and writes only `out[i*4..i*4+4]` of `count * LIMBS`. All
        // three are distinct allocations. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub fn u128_to_montgomery(&self, values: &[u128]) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(values.len())?;
        if values.is_empty() {
            return Ok(output);
        }
        let mut magnitude = Vec::with_capacity(values.len() * 2);
        for &value in values {
            magnitude.push(value as u64);
            magnitude.push((value >> 64) as u64);
        }
        let magnitude = self.upload_u64_slice(&magnitude)?;
        let negative = self.upload_u8_slice(&vec![0u8; values.len()])?;
        let count = Self::count_of(values.len())?;
        let mut builder = self.stream().launch_builder(&self.i128_to_mont);
        let _ = builder.arg(&magnitude);
        let _ = builder.arg(&negative);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: identical launch to `i128_to_montgomery`, differing only in
        // that every `negative` byte is zero: thread `i < count` reads
        // `magnitude[2i]`, `magnitude[2i+1]` (a `2 * count`-element buffer) and
        // `negative[i]` (a `count`-element buffer), and writes only
        // `out[i*4..i*4+4]` of `count * LIMBS`. All three are distinct
        // allocations. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub(crate) fn exclusive_scan_with_total_u32(
        &self,
        input: &CudaSlice<u32>,
        len: usize,
    ) -> Result<(CudaSlice<u32>, usize), CudaError> {
        let scanned = self.exclusive_scan_u32_on_device(input, len)?;
        if len == 0 {
            return Ok((scanned, 0));
        }
        let tail = self.download_u32_range(input, len - 1, len)?[0];
        let last = self.download_u32_range(&scanned, len - 1, len)?[0];
        let total = last
            .checked_add(tail)
            .ok_or(CudaError::InvariantViolation {
                reason: "an exclusive scan total overflowed u32",
            })? as usize;
        Ok((scanned, total))
    }

    pub(crate) fn exclusive_scan_u32_on_device(
        &self,
        input: &CudaSlice<u32>,
        len: usize,
    ) -> Result<CudaSlice<u32>, CudaError> {
        let count = Self::count_of(len)?;
        let blocks = count.div_ceil(BLOCK_SIZE);
        let mut output = self.alloc_u32(len)?;
        let mut block_sums = self.alloc_u32(blocks as usize)?;
        let mut builder = self.stream().launch_builder(&self.scan_u32_block);
        let _ = builder.arg(input);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&mut block_sums);
        let _ = builder.arg(&count);
        // SAFETY: identical launch to `exclusive_scan_u32`, differing only in
        // that the input already lives on device: thread `i < count` reads
        // `input[i]` and writes `out[i]`, both `count` u32s, and only the last
        // thread of each block writes `block_sums[blockIdx.x]` of `blocks` u32s.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.stream().synchronize()?;
        if blocks == 1 {
            return Ok(output);
        }

        let offsets = self.exclusive_scan_u32_on_device(&block_sums, blocks as usize)?;
        let mut builder = self.stream().launch_builder(&self.scan_u32_add_offsets);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `block_offsets[blockIdx.x]` of
        // `blocks` u32s and read-modify-writes only `out[i]` of `count` u32s,
        // one thread per element.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub fn exclusive_scan_u32(&self, values: &[u32]) -> Result<Vec<u32>, CudaError> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(values.len())?;
        let blocks = count.div_ceil(BLOCK_SIZE);
        let input = self.upload_u32_slice(values)?;
        let mut output = self.alloc_u32(values.len())?;
        let mut block_sums = self.alloc_u32(blocks as usize)?;
        let mut builder = self.stream().launch_builder(&self.scan_u32_block);
        let _ = builder.arg(&input);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&mut block_sums);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `input[i]` and writes only
        // `out[i]`; both buffers hold `count` u32s. The block scan works in
        // shared memory sized `BLOCK` u32s by the kernel's declaration, with
        // `__syncthreads()` on both sides of every update, so the read of
        // `scratch[tid - stride]` is separated from the write to `scratch[tid]`.
        // Only the last thread of each block writes `block_sums[blockIdx.x]`,
        // and `block_sums` holds exactly `blocks` u32s.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.stream().synchronize()?;

        if blocks == 1 {
            return self.download_u32(&output);
        }

        let sums = self.download_u32(&block_sums)?;
        let offsets = self.exclusive_scan_u32(&sums)?;
        let offsets = self.upload_u32_slice(&offsets)?;
        let mut builder = self.stream().launch_builder(&self.scan_u32_add_offsets);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `block_offsets[blockIdx.x]` (a
        // `blocks`-element buffer, indexed by its own block id) and
        // read-modify-writes only `out[i]` of `count` u32s — one thread per
        // element, so the update is unsynchronized but never contended. Threads
        // with `i >= count` return first.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.stream().synchronize()?;
        self.download_u32(&output)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
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
            Fr::from_le_bytes_mod_order(&bytes)
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

        #[test]
        fn u64_to_montgomery_matches_cpu(values in vec(any::<u64>(), 1..300)) {
            let Some(context) = device() else { return Ok(()); };
            let expected: Vec<Fr> = values.iter().copied().map(Fr::from_u64).collect();
            let got = context
                .u64_to_montgomery(&values)
                .expect("device u64_to_montgomery")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn i128_to_montgomery_matches_cpu(values in vec(any::<i128>(), 1..300)) {
            let Some(context) = device() else { return Ok(()); };
            let expected: Vec<Fr> = values.iter().copied().map(Fr::from_i128).collect();
            let got = context
                .i128_to_montgomery(&values)
                .expect("device i128_to_montgomery")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn u128_to_montgomery_matches_cpu(values in vec(any::<u128>(), 1..300)) {
            let Some(context) = device() else { return Ok(()); };
            let expected: Vec<Fr> = values.iter().copied().map(Fr::from_u128).collect();
            let got = context
                .u128_to_montgomery(&values)
                .expect("device u128_to_montgomery")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn exclusive_scan_u32_matches_cpu(values in vec(0u32..1_000_000, 1..5000)) {
            let Some(context) = device() else { return Ok(()); };
            let mut expected = Vec::with_capacity(values.len());
            let mut running = 0u32;
            for &value in &values {
                expected.push(running);
                running += value;
            }
            let got = context
                .exclusive_scan_u32(&values)
                .expect("device exclusive_scan_u32");
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
    fn integer_conversion_matches_cpu_at_extremes() {
        let Some(context) = device() else {
            return;
        };
        let u64_values = [0u64, 1, u64::MAX, u64::MAX - 1, 1 << 63];
        assert_eq!(
            context
                .u64_to_montgomery(&u64_values)
                .expect("u64_to_montgomery")
                .to_host()
                .expect("download"),
            u64_values
                .iter()
                .copied()
                .map(Fr::from_u64)
                .collect::<Vec<_>>(),
        );

        let i128_values = [0i128, 1, -1, i128::MAX, i128::MIN + 1, -(1i128 << 100)];
        assert_eq!(
            context
                .i128_to_montgomery(&i128_values)
                .expect("i128_to_montgomery")
                .to_host()
                .expect("download"),
            i128_values
                .iter()
                .copied()
                .map(Fr::from_i128)
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn reduction_and_scan_match_cpu_across_block_boundaries() {
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

            let counts: Vec<u32> = (0..len as u32).map(|i| i % 7).collect();
            let mut expected_scan = Vec::with_capacity(len);
            let mut running = 0u32;
            for &count in &counts {
                expected_scan.push(running);
                running += count;
            }
            assert_eq!(
                context
                    .exclusive_scan_u32(&counts)
                    .expect("exclusive_scan_u32"),
                expected_scan,
                "scan diverged at len {len}",
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
