#![expect(
    dead_code,
    reason = "implementation target: the five G1 relations wire this up in step 3, and the \
              expectation becomes an error the moment the first one does"
)]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::Field;

use super::context::{CudaKernelContext, BLOCK};
use super::dense_product::DeviceDenseProduct;
use super::device::{require_fr, DeviceFrVec, LIMBS};
use super::error::CudaError;

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
        tables: &[&DeviceFrVec],
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
            if table.len() != 2 * half {
                return Err(CudaError::LengthMismatch {
                    expected: 2 * half,
                    got: table.len(),
                });
            }
        }
        let lanes = CudaKernelContext::count_of(lane_count)?;
        let half_count = CudaKernelContext::count_of(half)?;
        let terms = CudaKernelContext::count_of(self.terms)?;
        let pointers = context.device_pointers(tables)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;
        let infinity_flag = u32::from(infinity_lane);

        let mut builder = context.stream().launch_builder(context.sop_round());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&self.offsets);
        let _ = builder.arg(&self.factors);
        let _ = builder.arg(self.coefficients.limbs());
        let _ = builder.arg(&terms);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(partials.limbs_mut());
        let _ = builder.arg(&first_point);
        let _ = builder.arg(&infinity_flag);
        // SAFETY: thread `y < half` reads `table[2y]` and `table[2y+1]` of every
        // table named by `factors`, each checked above to hold `2 * half`
        // elements, and every `factors` entry indexes `tables` because `push`
        // recorded it against that same slice. `offsets` holds `terms + 1`
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

        let totals = DeviceDenseProduct::reduce_lanes(context, partials, lanes, blocks)?;
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
            let handles: Vec<_> = uploaded.iter().collect();

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
        let handles = [&table, &table];
        assert!(
            device_form
                .round_lanes::<Fr>(context, &handles, 1, 1, true, 2)
                .is_err(),
            "an infinity lane over mixed arities must be refused, not silently wrong",
        );
    }
}
