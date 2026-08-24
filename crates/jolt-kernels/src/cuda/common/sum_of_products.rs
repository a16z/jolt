#![expect(
    dead_code,
    reason = "implementation target: the five G1 relations wire this up in step 3, and the \
              expectation becomes an error the moment the first one does"
)]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::Field;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{require_fr, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::half_fold::FoldColumn;
use super::primitives::reduce_lanes;
use super::split_eq::DeviceSplitEq;

pub struct SumOfProducts<F: Field> {
    offsets: Vec<u32>,
    factors: Vec<u32>,
    coefficients: Vec<F>,
    arity: Option<usize>,
}

impl<F: Field> Default for SumOfProducts<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> SumOfProducts<F> {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            factors: Vec::new(),
            coefficients: Vec::new(),
            arity: None,
        }
    }

    pub fn push(&mut self, coefficient: F, factors: &[usize]) -> Result<(), CudaError> {
        if factors.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a sum-of-products term needs at least one factor",
            });
        }
        match self.arity {
            None => self.arity = Some(factors.len()),
            Some(arity) if arity == factors.len() => {}
            Some(_) => self.arity = Some(usize::MAX),
        }
        for &factor in factors {
            self.factors.push(u32::try_from(factor).map_err(|_| {
                CudaError::InvariantViolation {
                    reason: "a sum-of-products factor index exceeds a u32",
                }
            })?);
        }
        self.offsets
            .push(u32::try_from(self.factors.len()).map_err(|_| {
                CudaError::InvariantViolation {
                    reason: "a sum-of-products term list exceeds a u32 factor count",
                }
            })?);
        self.coefficients.push(coefficient);
        Ok(())
    }

    pub fn terms(&self) -> usize {
        self.coefficients.len()
    }

    pub fn uniform_arity(&self) -> Option<usize> {
        match self.arity {
            Some(arity) if arity != usize::MAX => Some(arity),
            _ => None,
        }
    }

    pub fn upload(&self, context: &CudaKernelContext) -> Result<DeviceSumOfProducts, CudaError> {
        if self.coefficients.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a sum-of-products form needs at least one term",
            });
        }
        let coefficients: Vec<jolt_field::Fr> = self
            .coefficients
            .iter()
            .map(|value| require_fr(*value))
            .collect::<Result<_, _>>()?;
        Ok(DeviceSumOfProducts {
            offsets: context.upload_u32_slice(&self.offsets)?,
            factors: context.upload_u32_slice(&self.factors)?,
            coefficients: context.upload(&coefficients)?,
            terms: self.coefficients.len(),
            uniform_arity: self.uniform_arity(),
        })
    }
}

pub struct DeviceSumOfProducts {
    offsets: CudaSlice<u32>,
    factors: CudaSlice<u32>,
    coefficients: DeviceFrVec,
    terms: usize,
    uniform_arity: Option<usize>,
}

impl DeviceSumOfProducts {
    pub const fn uniform_arity(&self) -> Option<usize> {
        self.uniform_arity
    }

    pub fn round_lanes<F: Field>(
        &self,
        context: &CudaKernelContext,
        tables: &[FoldColumn<'_>],
        half: usize,
        first_point: u32,
        infinity_lane: bool,
        lane_count: usize,
    ) -> Result<Vec<F>, CudaError> {
        if infinity_lane && self.uniform_arity.is_none() {
            return Err(CudaError::InvariantViolation {
                reason: "an infinity lane mixes leading coefficients of different degrees \
                         unless every term has the same arity",
            });
        }
        for table in tables {
            if !table.covers(2 * half) {
                return Err(CudaError::LengthMismatch {
                    expected: 2 * half,
                    got: table.len(),
                });
            }
        }
        let lanes = CudaKernelContext::count_of(lane_count)?;
        let half_count = CudaKernelContext::count_of(half)?;
        let terms = CudaKernelContext::count_of(self.terms)?;
        let (pointers, descriptors) = context.device_columns(tables)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;
        let infinity_flag = u32::from(infinity_lane);

        let mut builder = context.stream().launch_builder(context.sop_round());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&descriptors);
        let _ = builder.arg(&self.offsets);
        let _ = builder.arg(&self.factors);
        let _ = builder.arg(self.coefficients.limbs());
        let _ = builder.arg(&terms);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(partials.limbs_mut());
        let _ = builder.arg(&first_point);
        let _ = builder.arg(&infinity_flag);
        // SAFETY: thread `y < half` reads entries `2y` and `2y+1` of every
        // table named by `factors`, each checked above by `covers` to reach
        // `2 * half` entries under its own descriptor, and every `factors` entry
        // indexes `tables`/`descriptors` because `push` recorded it against that
        // same slice. `offsets` holds `terms + 1`
        // entries so `offsets[t + 1]` is in bounds, and `coefficients` holds
        // `terms`. Writes touch only `partials[c * gridDim.x + blockIdx.x]` of
        // `lanes * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, and the block reduction sits outside the `y <
        // half` guard so every thread reaches each `__syncthreads()`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let totals = reduce_lanes(context, partials, lanes, blocks)?;
        totals
            .to_host()?
            .into_iter()
            .map(|value| {
                super::device::fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub fn round_gruen_endpoints<F: Field>(
        &self,
        context: &CudaKernelContext,
        tables: &[FoldColumn<'_>],
        half: usize,
        eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F), CudaError> {
        if self.uniform_arity.is_none() {
            return Err(CudaError::InvariantViolation {
                reason: "a Gruen leading-coefficient lane mixes leading coefficients of different \
                         degrees unless every term has the same arity",
            });
        }
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: 0,
            });
        }
        for table in tables {
            if !table.covers(2 * half) {
                return Err(CudaError::LengthMismatch {
                    expected: 2 * half,
                    got: table.len(),
                });
            }
        }
        let e_in_len = eq.e_in_len();
        if eq.e_out_current().len() * e_in_len != half {
            return Err(CudaError::LengthMismatch {
                expected: half,
                got: eq.e_out_current().len() * e_in_len,
            });
        }
        context.require_owned(self.offsets.ordinal())?;
        for table in tables {
            context.require_owned(table.words().ordinal())?;
        }
        context.require_owned(eq.e_out_current().ordinal())?;

        let half_count = CudaKernelContext::count_of(half)?;
        let terms = CudaKernelContext::count_of(self.terms)?;
        let inner_len = CudaKernelContext::count_of(e_in_len)?;
        let in_bits = CudaKernelContext::count_of(e_in_len.max(1).ilog2() as usize)?;
        let (pointers, descriptors) = context.device_columns(tables)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(2 * blocks as usize)?;

        let mut builder = context.stream().launch_builder(context.sopg_round());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&descriptors);
        let _ = builder.arg(&self.offsets);
        let _ = builder.arg(&self.factors);
        let _ = builder.arg(self.coefficients.limbs());
        let _ = builder.arg(&terms);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(eq.e_in_current().limbs());
        let _ = builder.arg(eq.e_out_current().limbs());
        let _ = builder.arg(&inner_len);
        let _ = builder.arg(&in_bits);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `y < half` reads entries `2y` and `2y+1` of every
        // table named by `factors`, each checked above by `covers` to reach
        // `2 * half` entries under its own descriptor, and every `factors` entry
        // indexes `tables`/`descriptors` because `push` recorded it against that
        // same slice. `offsets` holds `terms + 1`
        // entries so `offsets[t + 1]` is in bounds, and `coefficients` holds
        // `terms`. The eq lookup reads `e_out[y]` when `e_in_len <= 1` and
        // otherwise `e_in[y & mask]` / `e_out[y >> in_bits]`, bounded because
        // `in_bits` is `e_in`'s log length and
        // `e_out.len() * e_in_len == half` is checked above. Writes touch only
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of
        // `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, and the two block reductions sit outside the
        // `y < half` guard so every thread reaches each `__syncthreads()`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let totals = reduce_lanes(context, partials, 2, blocks)?;
        let host = totals.to_host()?;
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        let constant = super::device::fr_into(host[0]).ok_or_else(unsupported)?;
        let leading = super::device::fr_into(host[1]).ok_or_else(unsupported)?;
        Ok((constant, leading))
    }
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
    use super::SumOfProducts;

    fn cpu_lane(
        tables: &[Vec<Fr>],
        terms: &[(Fr, Vec<usize>)],
        half: usize,
        point: Option<Fr>,
    ) -> Fr {
        let mut total = Fr::from_u64(0);
        for y in 0..half {
            for (coefficient, factors) in terms {
                let mut product = Fr::from_u64(1);
                for &factor in factors {
                    let lo = tables[factor][2 * y];
                    let hi = tables[factor][2 * y + 1];
                    let value = match point {
                        Some(point) => lo + point * (hi - lo),
                        None => hi - lo,
                    };
                    product *= value;
                }
                total += *coefficient * product;
            }
        }
        total
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]
        #[test]
        fn sum_of_products_round_matches_cpu(
            log_half in 1usize..8,
            seed in any::<u64>(),
            arity in 1usize..4,
            term_count in 1usize..5,
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let half = 1usize << log_half;
            let table_count = arity + 2;
            let tables: Vec<Vec<Fr>> = (0..table_count)
                .map(|t| {
                    (0..2 * half)
                        .map(|i| fr(seed ^ ((t as u64) << 32) ^ (i as u64 * 31 + 7)))
                        .collect()
                })
                .collect();
            let terms: Vec<(Fr, Vec<usize>)> = (0..term_count)
                .map(|t| {
                    let coefficient = fr(seed ^ (t as u64 * 1009 + 3));
                    let factors = (0..arity).map(|f| (t + f) % table_count).collect();
                    (coefficient, factors)
                })
                .collect();

            let mut form = SumOfProducts::<Fr>::new();
            for (coefficient, factors) in &terms {
                form.push(*coefficient, factors).expect("push term");
            }
            prop_assert_eq!(form.uniform_arity(), Some(arity));
            let device_form = form.upload(context).expect("upload form");
            let uploaded: Vec<_> = tables
                .iter()
                .map(|table| context.upload(table).expect("upload table"))
                .collect();
            let handles: Vec<_> = uploaded.iter().map(super::FoldColumn::Field).collect();

            let lanes = 3;
            let got: Vec<Fr> = device_form
                .round_lanes(context, &handles, half, 1, true, lanes)
                .expect("device round lanes");

            let expected: Vec<Fr> = (0..lanes)
                .map(|lane| {
                    let point = if lane + 1 == lanes {
                        None
                    } else {
                        Some(Fr::from_u64(1 + lane as u64))
                    };
                    cpu_lane(&tables, &terms, half, point)
                })
                .collect();
            prop_assert_eq!(got, expected, "sum-of-products lanes diverged");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn gruen_sum_of_products_endpoints_match_cpu(
            log_t in 2usize..9,
            seed in any::<u64>(),
            arity in 1usize..4,
            term_count in 1usize..4,
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let cycles = 1usize << log_t;
            let table_count = arity + 1;
            let tables: Vec<Vec<Fr>> = (0..table_count)
                .map(|t| {
                    (0..cycles)
                        .map(|i| fr(seed ^ ((t as u64) << 40) ^ (i as u64 * 17 + 5)))
                        .collect()
                })
                .collect();
            let terms: Vec<(Fr, Vec<usize>)> = (0..term_count)
                .map(|t| {
                    let coefficient = fr(seed ^ (t as u64 * 2003 + 11));
                    let factors = (0..arity).map(|f| (t + f) % table_count).collect();
                    (coefficient, factors)
                })
                .collect();
            let point: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 97 + 41))).collect();

            let mut form = SumOfProducts::<Fr>::new();
            for (coefficient, factors) in &terms {
                form.push(*coefficient, factors).expect("push term");
            }
            let device_form = form.upload(context).expect("upload form");
            let uploaded: Vec<_> = tables
                .iter()
                .map(|table| context.upload(table).expect("upload table"))
                .collect();
            let handles: Vec<_> = uploaded.iter().map(super::FoldColumn::Field).collect();
            let eq = super::super::split_eq::DeviceSplitEq::<Fr>::new(
                context,
                &point,
                jolt_poly::BindingOrder::LowToHigh,
            )
            .expect("device split-eq");
            let host_eq = jolt_poly::GruenSplitEqPolynomial::<Fr>::new(
                &point,
                jolt_poly::BindingOrder::LowToHigh,
            );

            let got: (Fr, Fr) = device_form
                .round_gruen_endpoints(context, &handles, cycles / 2, &eq)
                .expect("device gruen endpoints");
            let expected = cpu_gruen_endpoints(&tables, &terms, cycles / 2, &host_eq);
            prop_assert_eq!(got, expected, "gruen endpoints diverged");
        }
    }

    fn cpu_gruen_endpoints(
        tables: &[Vec<Fr>],
        terms: &[(Fr, Vec<usize>)],
        half: usize,
        eq: &jolt_poly::GruenSplitEqPolynomial<Fr>,
    ) -> (Fr, Fr) {
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();
        let bits = e_in.len().max(1).ilog2();
        let mut constant = Fr::from_u64(0);
        let mut leading = Fr::from_u64(0);
        for y in 0..half {
            let weight = e_out[y >> bits]
                * if e_in.len() <= 1 {
                    Fr::from_u64(1)
                } else {
                    e_in[y & ((1usize << bits) - 1)]
                };
            let mut sum_constant = Fr::from_u64(0);
            let mut sum_leading = Fr::from_u64(0);
            for (coefficient, factors) in terms {
                let mut at_zero = Fr::from_u64(1);
                let mut delta = Fr::from_u64(1);
                for &factor in factors {
                    let lo = tables[factor][2 * y];
                    let hi = tables[factor][2 * y + 1];
                    at_zero *= lo;
                    delta *= hi - lo;
                }
                sum_constant += *coefficient * at_zero;
                sum_leading += *coefficient * delta;
            }
            constant += weight * sum_constant;
            leading += weight * sum_leading;
        }
        (constant, leading)
    }

    #[test]
    fn mixed_arity_endpoints_are_refused() {
        let Some(context) = shared_context() else {
            return;
        };
        let point = [fr(3), fr(9)];
        let eq = super::super::split_eq::DeviceSplitEq::<Fr>::new(
            context,
            &point,
            jolt_poly::BindingOrder::LowToHigh,
        )
        .expect("device split-eq");
        let mut form = SumOfProducts::<Fr>::new();
        form.push(Fr::from_u64(1), &[0]).expect("arity one");
        form.push(Fr::from_u64(1), &[0, 1]).expect("arity two");
        let device_form = form.upload(context).expect("upload form");
        let table = context
            .upload(&[fr(1), fr(2), fr(3), fr(4)])
            .expect("upload table");
        let handles = [
            super::FoldColumn::Field(&table),
            super::FoldColumn::Field(&table),
        ];
        assert!(
            device_form
                .round_gruen_endpoints::<Fr>(context, &handles, 2, &eq)
                .is_err(),
            "a leading-coefficient lane over mixed arities must be refused, not silently wrong",
        );
    }

    #[test]
    fn mixed_arity_terms_refuse_an_infinity_lane() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut form = SumOfProducts::<Fr>::new();
        form.push(Fr::from_u64(1), &[0]).expect("arity one");
        form.push(Fr::from_u64(1), &[0, 1]).expect("arity two");
        assert_eq!(form.uniform_arity(), None);
        let device_form = form.upload(context).expect("upload form");
        let table = context.upload(&[fr(1), fr(2)]).expect("upload table");
        let handles = [
            super::FoldColumn::Field(&table),
            super::FoldColumn::Field(&table),
        ];
        assert!(
            device_form
                .round_lanes::<Fr>(context, &handles, 1, 1, true, 2)
                .is_err(),
            "an infinity lane over mixed arities must be refused, not silently wrong",
        );
    }
}
