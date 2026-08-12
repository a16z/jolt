use cudarc::driver::PushKernelArg;
use jolt_field::{Field, Fr};
use jolt_poly::BindingOrder;

use super::context::CudaKernelContext;
use super::device::{require_fr, DeviceFrVec};
use super::error::CudaError;

pub struct DeviceLtPolynomial {
    lt_lo: DeviceFrVec,
    lt_hi: DeviceFrVec,
    eq_hi: DeviceFrVec,
    order: BindingOrder,
    lo_vars: usize,
    hi_vars: usize,
}

pub(crate) struct SplitLtView<'a> {
    pub(super) lt_lo: &'a DeviceFrVec,
    pub(super) lt_hi: &'a DeviceFrVec,
    pub(super) eq_hi: &'a DeviceFrVec,
    pub(super) lo_bits: u32,
    pub(super) lo_mask: u32,
}

impl DeviceLtPolynomial {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        point: &[F],
        order: BindingOrder,
    ) -> Result<Self, CudaError> {
        let mid = point.len() / 2;
        let (r_hi, r_lo) = point.split_at(point.len() - mid);
        let mut hi = Vec::with_capacity(r_hi.len());
        for &coordinate in r_hi {
            hi.push(require_fr(coordinate)?);
        }
        let mut lo = Vec::with_capacity(r_lo.len());
        for &coordinate in r_lo {
            lo.push(require_fr(coordinate)?);
        }
        Ok(Self {
            lt_lo: context.lt_evals(&lo)?,
            lt_hi: context.lt_evals(&hi)?,
            eq_hi: context.eq_evals(&hi)?,
            order,
            lo_vars: r_lo.len(),
            hi_vars: r_hi.len(),
        })
    }

    pub const fn len(&self) -> usize {
        1usize << (self.hi_vars + self.lo_vars)
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn order(&self) -> BindingOrder {
        self.order
    }

    pub const fn num_vars(&self) -> usize {
        self.hi_vars + self.lo_vars
    }

    pub(crate) fn view(&self) -> SplitLtView<'_> {
        SplitLtView {
            lt_lo: &self.lt_lo,
            lt_hi: &self.lt_hi,
            eq_hi: &self.eq_hi,
            lo_bits: self.lo_vars as u32,
            lo_mask: ((1usize << self.lo_vars) - 1) as u32,
        }
    }

    pub fn coefficients(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        let len = self.len();
        let mut output = context.alloc(len)?;
        let count = CudaKernelContext::count_of(len)?;
        let view = self.view();
        let mut builder = context.stream().launch_builder(context.lt_reconstruct());
        let _ = builder.arg(view.lt_lo.limbs());
        let _ = builder.arg(view.lt_hi.limbs());
        let _ = builder.arg(view.eq_hi.limbs());
        let _ = builder.arg(&view.lo_bits);
        let _ = builder.arg(&view.lo_mask);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `j < len` writes only `out[j]` and reads
        // `lt_hi[j >> lo_bits]`, `eq_hi[j >> lo_bits]`, `lt_lo[j & lo_mask]`.
        // Since `len == 2^(hi_vars + lo_vars)` with `lo_bits == lo_vars`, those
        // indices are bounded by `2^hi_vars` and `2^lo_vars` — the three tables'
        // element counts. `out` is a distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        context.stream().synchronize()?;
        Ok(output)
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        let drain_lo = match self.order {
            BindingOrder::LowToHigh => self.lo_vars > 0,
            BindingOrder::HighToLow => self.hi_vars == 0,
        };
        if drain_lo {
            if self.lo_vars == 0 {
                return Err(CudaError::InvariantViolation {
                    reason: "a split LT polynomial has no variables left to bind",
                });
            }
            self.lt_lo = context.bind(&self.lt_lo, challenge, self.order)?;
            self.lo_vars -= 1;
        } else {
            if self.hi_vars == 0 {
                return Err(CudaError::InvariantViolation {
                    reason: "a split LT polynomial has no variables left to bind",
                });
            }
            self.lt_hi = context.bind(&self.lt_hi, challenge, self.order)?;
            self.eq_hi = context.bind(&self.eq_hi, challenge, self.order)?;
            self.hi_vars -= 1;
        }
        Ok(())
    }

    pub fn final_claim(&self, context: &CudaKernelContext) -> Result<Fr, CudaError> {
        if self.num_vars() != 0 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len(),
            });
        }
        self.coefficients(context)?.first()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::Fr;
    use jolt_poly::{BindingOrder, LtPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::arb_point;
    use super::DeviceLtPolynomial;

    proptest! {
        #[test]
        fn lt_coefficients_match_dense_binding(
            point in arb_point(6),
            challenges in arb_point(6),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
                let mut expected = Polynomial::new(LtPolynomial::evaluations(&point));
                let mut got = DeviceLtPolynomial::new(context, &point, order)
                    .expect("device lt poly");

                for (round, &challenge) in challenges.iter().enumerate() {
                    let got_coeffs = got
                        .coefficients(context)
                        .expect("device coefficients")
                        .to_host()
                        .expect("download");
                    prop_assert_eq!(
                        got_coeffs,
                        expected.evals().to_vec(),
                        "coefficients diverged at round {} for {:?}",
                        round,
                        order
                    );
                    prop_assert_eq!(got.len(), expected.evals().len());
                    got.bind(context, challenge).expect("device bind");
                    expected.bind_with_order(challenge, order);
                }
                prop_assert_eq!(
                    got.final_claim(context).expect("final claim"),
                    expected.evals()[0]
                );
            }
        }
    }

    #[test]
    fn lt_split_matches_dense_at_odd_and_even_widths() {
        let Some(context) = shared_context() else {
            return;
        };
        for num_vars in 1usize..=7 {
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| super::super::testing::fr(31 + 7 * i as u64))
                .collect();
            for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
                let expected = LtPolynomial::evaluations(&point);
                let got = DeviceLtPolynomial::new(context, &point, order)
                    .expect("device lt poly")
                    .coefficients(context)
                    .expect("device coefficients")
                    .to_host()
                    .expect("download");
                assert_eq!(
                    got, expected,
                    "unbound coefficients diverged at num_vars {num_vars} for {order:?}",
                );
            }
        }
    }
}
