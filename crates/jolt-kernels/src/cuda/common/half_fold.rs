use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::Field;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{fr_limbs, require_fr, DeviceFrVec, LIMBS};
use super::error::CudaError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SummedHalf {
    High,
    Low,
}

impl SummedHalf {
    const fn strides(self, out_len: usize, sum_len: usize) -> (usize, usize) {
        match self {
            Self::High => (1, out_len),
            Self::Low => (sum_len, 1),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NarrowKind {
    U64,
    U128,
    TwosI128,
}

impl NarrowKind {
    const fn words(self) -> usize {
        match self {
            Self::U64 => 1,
            Self::U128 | Self::TwosI128 => 2,
        }
    }

    const fn code(self) -> u32 {
        match self {
            Self::U64 => 0,
            Self::U128 => 1,
            Self::TwosI128 => 2,
        }
    }
}

#[derive(Clone, Copy)]
pub struct NarrowColumn<'a> {
    pub words: &'a CudaSlice<u64>,
    pub kind: NarrowKind,
    pub len: usize,
    pub stride: usize,
    pub offset: usize,
}

impl<'a> NarrowColumn<'a> {
    pub const fn packed(words: &'a CudaSlice<u64>, kind: NarrowKind, len: usize) -> Self {
        Self {
            words,
            kind,
            len,
            stride: kind.words(),
            offset: 0,
        }
    }

    const fn reach(&self) -> usize {
        if self.len == 0 {
            return 0;
        }
        self.offset + (self.len - 1) * self.stride + self.kind.words()
    }
}

#[derive(Clone, Copy)]
pub enum FoldColumn<'a> {
    Field(&'a DeviceFrVec),
    Narrow(NarrowColumn<'a>),
}

const FIELD_KIND: u32 = 3;

impl FoldColumn<'_> {
    pub const fn len(&self) -> usize {
        match self {
            Self::Field(column) => column.len(),
            Self::Narrow(column) => column.len,
        }
    }

    const fn kind(&self) -> u32 {
        match self {
            Self::Field(_) => FIELD_KIND,
            Self::Narrow(column) => column.kind.code(),
        }
    }

    const fn words(&self) -> &CudaSlice<u64> {
        match self {
            Self::Field(column) => column.limbs(),
            Self::Narrow(column) => column.words,
        }
    }

    fn covers(&self, len: usize) -> bool {
        match self {
            Self::Field(column) => column.len() == len,
            Self::Narrow(column) => {
                column.len == len
                    && column.stride >= column.kind.words()
                    && column.words.len() >= column.reach()
            }
        }
    }

    fn geometry(&self) -> Result<(u32, u32), CudaError> {
        let (stride, offset) = match self {
            Self::Field(_) => (LIMBS, 0),
            Self::Narrow(column) => (column.stride, column.offset),
        };
        Ok((
            CudaKernelContext::count_of(stride)?,
            CudaKernelContext::count_of(offset)?,
        ))
    }
}

pub fn half_fold<F: Field>(
    context: &CudaKernelContext,
    column: FoldColumn<'_>,
    weights: &DeviceFrVec,
    summed: SummedHalf,
    scale: F,
) -> Result<DeviceFrVec, CudaError> {
    if weights.is_empty() || !column.len().is_multiple_of(weights.len()) {
        return Err(CudaError::LengthMismatch {
            expected: weights.len().max(1),
            got: column.len(),
        });
    }
    let mut out = context.alloc(column.len() / weights.len())?;
    half_fold_into(
        context,
        column,
        weights,
        &mut out,
        summed,
        scale,
        F::zero(),
        false,
    )?;
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the fold's accumulate mode carries its own scale and bias so a \
              multi-column weighted sum lands in one buffer"
)]
pub fn half_fold_into<F: Field>(
    context: &CudaKernelContext,
    column: FoldColumn<'_>,
    weights: &DeviceFrVec,
    out: &mut DeviceFrVec,
    summed: SummedHalf,
    scale: F,
    bias: F,
    accumulate: bool,
) -> Result<(), CudaError> {
    let out_len = out.len();
    let sum_len = weights.len();
    if out_len == 0 || sum_len == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "a weighted half-fold needs a non-empty output and weight table",
        });
    }
    if !column.covers(out_len * sum_len) {
        return Err(CudaError::LengthMismatch {
            expected: out_len * sum_len,
            got: column.len(),
        });
    }
    let scale = fr_limbs(require_fr(scale)?);
    let bias = fr_limbs(require_fr(bias)?);
    if summed == SummedHalf::Low {
        return row_fold_into(context, column, weights, out, &scale, &bias, accumulate);
    }
    let (out_stride, sum_stride) = summed.strides(out_len, sum_len);
    let out_count = CudaKernelContext::count_of(out_len)?;
    let sum_count = CudaKernelContext::count_of(sum_len)?;
    let out_stride = CudaKernelContext::count_of(out_stride)?;
    let sum_stride = CudaKernelContext::count_of(sum_stride)?;
    let accumulate = u32::from(accumulate);
    let kind = column.kind();
    let (stride, offset) = column.geometry()?;

    let mut builder = context.stream().launch_builder(context.hf_half_fold());
    let _ = builder.arg(column.words());
    let _ = builder.arg(weights.limbs());
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&scale[0]);
    let _ = builder.arg(&scale[1]);
    let _ = builder.arg(&scale[2]);
    let _ = builder.arg(&scale[3]);
    let _ = builder.arg(&bias[0]);
    let _ = builder.arg(&bias[1]);
    let _ = builder.arg(&bias[2]);
    let _ = builder.arg(&bias[3]);
    let _ = builder.arg(&out_count);
    let _ = builder.arg(&sum_count);
    let _ = builder.arg(&out_stride);
    let _ = builder.arg(&sum_stride);
    let _ = builder.arg(&accumulate);
    let _ = builder.arg(&kind);
    let _ = builder.arg(&stride);
    let _ = builder.arg(&offset);
    // SAFETY: thread `a < out_len` reads `weights[b]` for every `b < sum_len`
    // (inside `weights`'s `sum_len` elements) and entry
    // `a * out_stride + b * sum_stride` of `column`, whose largest value is
    // `out_len * sum_len - 1` for either stride pair the axis yields. The kernel
    // turns an entry `i` into words `offset + i * stride ..= + kind.words()`, and
    // `covers` checked exactly that reach against `words.len()` for the largest
    // entry (`LIMBS`-word elements at stride `LIMBS` for the field arm), having
    // also required `stride >= kind.words()` so entries do not overlap. It reads
    // and writes only `out[a]` of `out_len`, so the
    // accumulate path's read-modify-write is per-thread and non-aliasing.
    // `scale` and `bias` arrive as by-value limbs, so no device buffer backs
    // them, and threads with `a >= out_len` return before any access.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(out_count)) }?;
    Ok(())
}

fn row_fold_into(
    context: &CudaKernelContext,
    column: FoldColumn<'_>,
    weights: &DeviceFrVec,
    out: &mut DeviceFrVec,
    scale: &[u64; LIMBS],
    bias: &[u64; LIMBS],
    accumulate: bool,
) -> Result<(), CudaError> {
    let out_len = out.len();
    let sum_len = weights.len();
    let sum_count = CudaKernelContext::count_of(sum_len)?;
    let blocks = u32::try_from(out_len).map_err(|_| CudaError::LengthMismatch {
        expected: u32::MAX as usize,
        got: out_len,
    })?;
    let accumulate = u32::from(accumulate);
    let kind = column.kind();
    let (stride, offset) = column.geometry()?;

    let mut builder = context.stream().launch_builder(context.hf_row_fold());
    let _ = builder.arg(column.words());
    let _ = builder.arg(weights.limbs());
    let _ = builder.arg(out.limbs_mut());
    for limb in scale {
        let _ = builder.arg(limb);
    }
    for limb in bias {
        let _ = builder.arg(limb);
    }
    let _ = builder.arg(&sum_count);
    let _ = builder.arg(&accumulate);
    let _ = builder.arg(&kind);
    let _ = builder.arg(&stride);
    let _ = builder.arg(&offset);
    // SAFETY: block `a = blockIdx.x < out_len` reads `weights[b]` and column
    // entry `a * sum_len + b` for `b` striding from `threadIdx.x` by
    // `blockDim.x` while `b < sum_len`, so every column entry is below
    // `out_len * sum_len`, whose words `offset + i * stride ..= + kind.words()`
    // `covers` checked against `words.len()` — and every weight index below
    // `sum_len`. Shared memory is `BLOCK * LIMBS` u64s, matching
    // `shared_mem_bytes`; every thread reaches each `__syncthreads()` because the
    // strided loop and the tree sit outside any early return, and `BLOCK` is a
    // power of two so the tree covers the block. Only thread 0 touches `out[a]`,
    // one slot per block, so the accumulate path's read-modify-write is
    // uncontended. `scale` and `bias` arrive as by-value limbs.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{half_fold, half_fold_into, FoldColumn, NarrowColumn, NarrowKind, SummedHalf};

    fn zero() -> Fr {
        Fr::from_u64(0)
    }

    fn narrow_entries(kind: NarrowKind, count: usize, seed: u64) -> (Vec<u64>, Vec<Fr>) {
        let mut words = Vec::with_capacity(count * 2);
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            let mixed = (index as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed);
            let high = mixed.rotate_left(29);
            match kind {
                NarrowKind::U64 => {
                    words.push(mixed);
                    values.push(Fr::from_u64(mixed));
                }
                NarrowKind::U128 => {
                    let value = (u128::from(high) << 64) | u128::from(mixed);
                    words.push(mixed);
                    words.push(high);
                    values.push(Fr::from_u128(value));
                }
                NarrowKind::TwosI128 => {
                    let value = (u128::from(high) << 64) | u128::from(mixed);
                    words.push(mixed);
                    words.push(high);
                    values.push(Fr::from_i128(value as i128));
                }
            }
        }
        (words, values)
    }

    fn cpu_half_fold(
        column: &[Fr],
        weights: &[Fr],
        out_len: usize,
        summed: SummedHalf,
        scale: Fr,
        bias: Fr,
    ) -> Vec<Fr> {
        let sum_len = weights.len();
        (0..out_len)
            .map(|a| {
                let mut total = zero();
                for (b, weight) in weights.iter().enumerate() {
                    let index = match summed {
                        SummedHalf::High => a + b * out_len,
                        SummedHalf::Low => b + a * sum_len,
                    };
                    total += *weight * column[index];
                }
                total * scale + bias
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]
        #[test]
        fn half_fold_matches_cpu(
            log_out in 1usize..6,
            log_sum in 1usize..6,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let out_len = 1usize << log_out;
            let sum_len = 1usize << log_sum;
            let column: Vec<Fr> = (0..out_len * sum_len)
                .map(|i| fr(seed ^ (i as u64 * 31 + 7)))
                .collect();
            let weights: Vec<Fr> = (0..sum_len).map(|b| fr(seed ^ (b as u64 * 1009 + 3))).collect();
            let scale = fr(seed ^ 0xfeed);
            let device_column = context.upload(&column).expect("upload column");
            let device_weights = context.upload(&weights).expect("upload weights");

            for summed in [SummedHalf::High, SummedHalf::Low] {
                let expected = cpu_half_fold(&column, &weights, out_len, summed, scale, zero());
                let got = half_fold(
                    context,
                    FoldColumn::Field(&device_column),
                    &device_weights,
                    summed,
                    scale,
                )
                    .expect("device half fold")
                    .to_host()
                    .expect("download fold");
                prop_assert_eq!(got, expected, "half fold diverged for {:?}", summed);
            }
        }

        #[test]
        fn narrow_half_fold_matches_the_host_fold(
            log_out in 1usize..6,
            log_sum in 1usize..6,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let out_len = 1usize << log_out;
            let sum_len = 1usize << log_sum;
            let weights: Vec<Fr> = (0..sum_len).map(|b| fr(seed ^ (b as u64 * 1009 + 3))).collect();
            let device_weights = context.upload(&weights).expect("upload weights");
            let scale = fr(seed ^ 0xfeed);

            for kind in [NarrowKind::U64, NarrowKind::U128, NarrowKind::TwosI128] {
                let (words, values) = narrow_entries(kind, out_len * sum_len, seed);
                let device_words = context.upload_u64_slice(&words).expect("upload words");
                let column = FoldColumn::Narrow(NarrowColumn::packed(
                    &device_words,
                    kind,
                    out_len * sum_len,
                ));

                for summed in [SummedHalf::High, SummedHalf::Low] {
                    let expected =
                        cpu_half_fold(&values, &weights, out_len, summed, scale, zero());
                    let got = half_fold(context, column, &device_weights, summed, scale)
                        .expect("device narrow half fold")
                        .to_host()
                        .expect("download fold");
                    prop_assert_eq!(got, expected, "{:?} narrow fold diverged for {:?}", kind, summed);
                }
            }
        }

        #[test]
        fn strided_narrow_half_fold_matches_the_host_fold(
            log_out in 1usize..6,
            log_sum in 1usize..5,
            stride_slack in 0usize..4,
            offset in 0usize..3,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let out_len = 1usize << log_out;
            let sum_len = 1usize << log_sum;
            let entries = out_len * sum_len;
            let weights: Vec<Fr> = (0..sum_len).map(|b| fr(seed ^ (b as u64 * 31 + 9))).collect();
            let device_weights = context.upload(&weights).expect("upload weights");
            let scale = fr(seed ^ 0xa11ce);

            for kind in [NarrowKind::U64, NarrowKind::U128, NarrowKind::TwosI128] {
                let (packed, values) = narrow_entries(kind, entries, seed);
                let stride = kind.words() + stride_slack;
                let mut words = vec![0u64; offset + entries * stride];
                for (index, entry) in packed.chunks_exact(kind.words()).enumerate() {
                    let base = offset + index * stride;
                    words[base..base + kind.words()].copy_from_slice(entry);
                }
                let device_words = context.upload_u64_slice(&words).expect("upload words");
                let column = FoldColumn::Narrow(NarrowColumn {
                    words: &device_words,
                    kind,
                    len: entries,
                    stride,
                    offset,
                });

                for summed in [SummedHalf::High, SummedHalf::Low] {
                    let expected =
                        cpu_half_fold(&values, &weights, out_len, summed, scale, zero());
                    let got = half_fold(context, column, &device_weights, summed, scale)
                        .expect("device strided narrow half fold")
                        .to_host()
                        .expect("download fold");
                    prop_assert_eq!(
                        got,
                        expected,
                        "{:?} strided fold diverged for {:?} at stride {} offset {}",
                        kind, summed, stride, offset
                    );
                }
            }
        }

        #[test]
        fn half_fold_accumulates_over_columns_matching_cpu(
            log_out in 1usize..6,
            log_sum in 1usize..6,
            columns in 1usize..4,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let out_len = 1usize << log_out;
            let sum_len = 1usize << log_sum;
            let weights: Vec<Fr> = (0..sum_len).map(|b| fr(seed ^ (b as u64 * 17 + 5))).collect();
            let device_weights = context.upload(&weights).expect("upload weights");
            let bias = fr(seed ^ 0xb1a5);
            let hosts: Vec<Vec<Fr>> = (0..columns)
                .map(|c| {
                    (0..out_len * sum_len)
                        .map(|i| fr(seed ^ ((c as u64) << 40) ^ (i as u64 * 13 + 1)))
                        .collect()
                })
                .collect();
            let scales: Vec<Fr> = (0..columns).map(|c| fr(seed ^ (c as u64 * 977 + 11))).collect();

            for summed in [SummedHalf::High, SummedHalf::Low] {
                let mut expected = vec![zero(); out_len];
                for (host, scale) in hosts.iter().zip(&scales) {
                    let partial = cpu_half_fold(host, &weights, out_len, summed, *scale, zero());
                    for (slot, value) in expected.iter_mut().zip(partial) {
                        *slot += value;
                    }
                }
                for slot in &mut expected {
                    *slot += bias;
                }

                let mut got = context.alloc(out_len).expect("allocate output");
                for (index, (host, scale)) in hosts.iter().zip(&scales).enumerate() {
                    let device_column = context.upload(host).expect("upload column");
                    half_fold_into(
                        context,
                        FoldColumn::Field(&device_column),
                        &device_weights,
                        &mut got,
                        summed,
                        *scale,
                        bias,
                        index > 0,
                    )
                    .expect("device half fold");
                }
                prop_assert_eq!(
                    got.to_host().expect("download fold"),
                    expected,
                    "accumulated half fold diverged for {:?}", summed
                );
            }
        }
    }

    #[test]
    fn narrow_half_fold_at_the_representation_extremes_matches_the_host_fold() {
        let Some(context) = shared_context() else {
            return;
        };
        let cases: [(NarrowKind, Vec<u64>, Vec<Fr>); 3] = [
            (
                NarrowKind::U64,
                vec![0, 1, u64::MAX, u64::MAX - 1],
                vec![
                    Fr::from_u64(0),
                    Fr::from_u64(1),
                    Fr::from_u64(u64::MAX),
                    Fr::from_u64(u64::MAX - 1),
                ],
            ),
            (
                NarrowKind::U128,
                vec![0, 0, 1, 0, u64::MAX, u64::MAX, 0, 1],
                vec![
                    Fr::from_u128(0),
                    Fr::from_u128(1),
                    Fr::from_u128(u128::MAX),
                    Fr::from_u128(1u128 << 64),
                ],
            ),
            (
                NarrowKind::TwosI128,
                vec![
                    u64::MAX,
                    u64::MAX,
                    1,
                    0,
                    0,
                    1u64 << 63,
                    u64::MAX,
                    (1u64 << 63) - 1,
                ],
                vec![
                    Fr::from_i128(-1),
                    Fr::from_i128(1),
                    Fr::from_i128(i128::MIN),
                    Fr::from_i128(i128::MAX),
                ],
            ),
        ];

        let weights: Vec<Fr> = (0..2).map(|b| fr(b as u64 * 7 + 5)).collect();
        let device_weights = context.upload(&weights).expect("upload weights");
        let scale = fr(0x5ca1e);
        for (kind, words, values) in cases {
            let device_words = context.upload_u64_slice(&words).expect("upload words");
            let column =
                FoldColumn::Narrow(NarrowColumn::packed(&device_words, kind, values.len()));
            for summed in [SummedHalf::High, SummedHalf::Low] {
                let expected = cpu_half_fold(&values, &weights, 2, summed, scale, zero());
                let got = half_fold(context, column, &device_weights, summed, scale)
                    .expect("device narrow half fold")
                    .to_host()
                    .expect("download fold");
                assert_eq!(
                    got, expected,
                    "{kind:?} narrow fold diverged for {summed:?}"
                );
            }
        }
    }

    #[test]
    fn half_fold_axes_are_different_functions() {
        let Some(context) = shared_context() else {
            return;
        };
        for (log_out, log_sum) in [(2usize, 2usize), (3, 1), (1, 3)] {
            let out_len = 1usize << log_out;
            let sum_len = 1usize << log_sum;
            let column: Vec<Fr> = (0..out_len * sum_len)
                .map(|i| fr(i as u64 * 31 + 7))
                .collect();
            let weights: Vec<Fr> = (0..sum_len).map(|b| fr(b as u64 * 1009 + 3)).collect();
            let device_column = context.upload(&column).expect("upload column");
            let device_weights = context.upload(&weights).expect("upload weights");
            let one = Fr::from_u64(1);

            let high = half_fold(
                context,
                FoldColumn::Field(&device_column),
                &device_weights,
                SummedHalf::High,
                one,
            )
            .expect("high fold")
            .to_host()
            .expect("download high");
            let low = half_fold(
                context,
                FoldColumn::Field(&device_column),
                &device_weights,
                SummedHalf::Low,
                one,
            )
            .expect("low fold")
            .to_host()
            .expect("download low");
            assert_ne!(
                high, low,
                "the two summed axes must be different functions at {out_len}x{sum_len}",
            );
            assert_eq!(
                high,
                cpu_half_fold(&column, &weights, out_len, SummedHalf::High, one, zero()),
            );
            assert_eq!(
                low,
                cpu_half_fold(&column, &weights, out_len, SummedHalf::Low, one, zero()),
            );
        }
    }
}
