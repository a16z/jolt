use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};

use super::context::CudaKernelContext;
use super::device::DeviceFrVec;
use super::error::CudaError;

pub struct DeviceSplitEq<F: Field> {
    host: GruenSplitEqPolynomial<F>,
    e_in: Vec<DeviceFrVec>,
    e_out: Vec<DeviceFrVec>,
}

pub fn split_eq_tables<F: Field>(
    context: &CudaKernelContext,
    point: &[F],
) -> Result<(DeviceFrVec, DeviceFrVec, usize), CudaError> {
    if point.is_empty() {
        return Err(CudaError::InvariantViolation {
            reason: "an eq factor pair needs at least one variable",
        });
    }
    let split = point.len() / 2;
    let in_bits = point.len() - split;
    let (outer, inner) = point.split_at(split);
    let e_out = context.upload(super::device::require_fr_slice(&EqPolynomial::<F>::evals(
        outer, None,
    ))?)?;
    let e_in = context.upload(super::device::require_fr_slice(&EqPolynomial::<F>::evals(
        inner, None,
    ))?)?;
    Ok((e_in, e_out, in_bits))
}

impl<F: Field> DeviceSplitEq<F> {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        self.e_in
            .iter()
            .chain(self.e_out.iter())
            .map(DeviceFrVec::device_bytes)
            .sum()
    }

    pub fn new(
        context: &CudaKernelContext,
        point: &[F],
        binding_order: BindingOrder,
    ) -> Result<Self, CudaError> {
        Self::with_host(context, point, binding_order, |point, binding_order| {
            GruenSplitEqPolynomial::<F>::new(point, binding_order)
        })
    }

    pub fn new_with_scaling(
        context: &CudaKernelContext,
        point: &[F],
        binding_order: BindingOrder,
        scaling: F,
    ) -> Result<Self, CudaError> {
        Self::with_host(context, point, binding_order, |point, binding_order| {
            GruenSplitEqPolynomial::<F>::new_with_scaling(point, binding_order, Some(scaling))
        })
    }

    fn with_host(
        context: &CudaKernelContext,
        point: &[F],
        binding_order: BindingOrder,
        host: impl FnOnce(&[F], BindingOrder) -> GruenSplitEqPolynomial<F>,
    ) -> Result<Self, CudaError> {
        if binding_order != BindingOrder::LowToHigh {
            return Err(CudaError::NotImplemented {
                kernel: "the device split-eq covers LowToHigh binding only; add the \
                         HighToLow stack split and extend its equivalence test",
            });
        }
        let host = host(point, binding_order);
        let (out_point, in_point) = if point.is_empty() {
            (&point[..0], &point[..0])
        } else {
            let split = point.len() / 2;
            let head = &point[..point.len() - 1];
            head.split_at(split.min(head.len()))
        };
        let e_out = Self::upload_stack(context, &EqPolynomial::<F>::evals_cached(out_point, None))?;
        let e_in = Self::upload_stack(context, &EqPolynomial::<F>::evals_cached(in_point, None))?;
        Ok(Self { host, e_in, e_out })
    }

    fn upload_stack(
        context: &CudaKernelContext,
        levels: &[Vec<F>],
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        levels
            .iter()
            .map(|level| context.upload(super::device::require_fr_slice(level)?))
            .collect()
    }

    pub fn e_in_current(&self) -> &DeviceFrVec {
        &self.e_in[self.e_in.len() - 1]
    }

    pub fn e_out_current(&self) -> &DeviceFrVec {
        &self.e_out[self.e_out.len() - 1]
    }

    pub fn e_in_len(&self) -> usize {
        self.host.e_in_current().len()
    }

    pub fn bind(&mut self, challenge: F) {
        self.host.bind(challenge);
        Self::align(&mut self.e_in, self.host.e_in_current().len());
        Self::align(&mut self.e_out, self.host.e_out_current().len());
    }

    fn align(stack: &mut Vec<DeviceFrVec>, len: usize) {
        while stack.len() > 1 && stack[stack.len() - 1].len() > len {
            let _ = stack.pop();
        }
    }

    pub fn gruen_poly_deg_3(&self, q0: F, q2: F, previous_claim: F) -> UnivariatePoly<F> {
        self.host.gruen_poly_deg_3(q0, q2, previous_claim)
    }

    pub fn gruen_poly_from_evals(&self, q_evals: &[F], previous_claim: F) -> UnivariatePoly<F> {
        self.host.gruen_poly_from_evals(q_evals, previous_claim)
    }

    pub fn merge(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        context.upload(super::device::require_fr_slice(self.host.merge().evals())?)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::Fr;
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::DeviceSplitEq;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]
        #[test]
        fn split_eq_matches_cpu_round_for_round(
            log_t in 1usize..10,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let point: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 31 + 7))).collect();
            let mut expected = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let mut got = DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                .expect("device split-eq");

            for round in 0..log_t {
                prop_assert_eq!(
                    got.e_in_current().to_host().expect("download e_in"),
                    expected.e_in_current().to_vec(),
                    "e_in diverged at round {}", round
                );
                prop_assert_eq!(
                    got.e_out_current().to_host().expect("download e_out"),
                    expected.e_out_current().to_vec(),
                    "e_out diverged at round {}", round
                );

                let q0 = fr(round as u64 * 13 + 5);
                let q2 = fr(round as u64 * 29 + 11);
                let claim = fr(round as u64 * 7 + 3);
                prop_assert_eq!(
                    got.gruen_poly_deg_3(q0, q2, claim).coefficients().to_vec(),
                    expected.gruen_poly_deg_3(q0, q2, claim).coefficients().to_vec(),
                    "round polynomial diverged at round {}", round
                );

                let challenge = fr(round as u64 * 17 + 23);
                expected.bind(challenge);
                got.bind(challenge);
            }

            prop_assert_eq!(
                got.merge(context).expect("device merge").to_host().expect("download merge"),
                expected.merge().evals().to_vec(),
                "merged eq diverged after the final bind"
            );
        }
    }
}
