use std::any::Any;
use std::sync::OnceLock;

use jolt_field::{Field, Fr};
use jolt_poly::lagrange::{lagrange_evals, lagrange_kernel_eval};
use jolt_poly::{EqPolynomial, UnivariatePoly};

use crate::cuda::{CudaError, CudaKernelContext, DeviceFrVec};
use crate::dense::bind_dense_evals_reuse_cuda;
use crate::stage1::{
    Stage1KernelError, Stage1OuterR1csData, Stage1OuterRemainingContext, Stage1RemainingRoundProof,
    OUTER_FIRST_GROUP_ROWS, OUTER_SECOND_GROUP_ROWS, OUTER_UNISKIP_BASE_START,
    OUTER_UNISKIP_DOMAIN_SIZE,
};

fn ctx() -> Option<&'static CudaKernelContext> {
    static CTX: OnceLock<Option<CudaKernelContext>> = OnceLock::new();
    CTX.get_or_init(|| CudaKernelContext::new(0).ok()).as_ref()
}

fn as_fr_slice<F: Field>(values: &[F]) -> Option<&[Fr]> {
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<Fr>() {
        // SAFETY: F and Fr are the same type (checked above), so &[F] and &[Fr]
        // have identical layout.
        Some(unsafe { &*(std::ptr::from_ref::<[F]>(values) as *const [Fr]) })
    } else {
        None
    }
}

fn into_fr<F: Field>(value: F) -> Option<Fr> {
    (Box::new(value) as Box<dyn Any>)
        .downcast::<Fr>()
        .ok()
        .map(|boxed| *boxed)
}

fn fr_poly_into<F: Field>(poly: UnivariatePoly<Fr>) -> Option<UnivariatePoly<F>> {
    (Box::new(poly) as Box<dyn Any>)
        .downcast::<UnivariatePoly<F>>()
        .ok()
        .map(|boxed| *boxed)
}

pub fn prove_remaining_rounds_cuda<F: Field>(
    data: &Stage1OuterR1csData<'_, F>,
    context: Stage1OuterRemainingContext<'_, F>,
    num_rounds: usize,
    batching_coeff: F,
    initial_claim: F,
    observe_round: &mut dyn FnMut(&UnivariatePoly<F>) -> F,
) -> Option<Stage1RemainingRoundProof<F>> {
    let ctx = ctx()?;

    let tau_high = context.tau[context.tau.len() - 1];
    let tau_low = &context.tau[..context.tau.len() - 1];
    let lagrange_tau_r0 = lagrange_kernel_eval(
        OUTER_UNISKIP_BASE_START,
        OUTER_UNISKIP_DOMAIN_SIZE,
        tau_high,
        context.r0,
    );
    let weights = lagrange_evals(OUTER_UNISKIP_BASE_START, OUTER_UNISKIP_DOMAIN_SIZE, context.r0);
    let scale = lagrange_tau_r0 * batching_coeff;
    let eq_evals = EqPolynomial::new(tau_low.to_vec()).evaluations();
    let first_group_rows: Vec<u32> = OUTER_FIRST_GROUP_ROWS.iter().map(|&r| r as u32).collect();
    let second_group_rows: Vec<u32> = OUTER_SECOND_GROUP_ROWS.iter().map(|&r| r as u32).collect();

    let row_dots = &data.row_dots;
    let eq_evals = as_fr_slice(&eq_evals)?;
    let scale = into_fr(scale)?;
    let weights = as_fr_slice(&weights)?;
    let row_dots_a = as_fr_slice(row_dots.a())?;
    let row_dots_b = as_fr_slice(row_dots.b())?;

    let mut state = match CudaDenseOuterState::from_row_dots(
        ctx,
        DenseOuterInputs {
            eq_evals,
            scale,
            weights,
            row_dots_a,
            row_dots_b,
            row_count: row_dots.row_count(),
            first_group_rows: &first_group_rows,
            second_group_rows: &second_group_rows,
        },
    ) {
        Ok(state) => state,
        Err(error) => return Some(Err(cuda_error(error))),
    };

    let mut running_sum = initial_claim * batching_coeff;
    let mut point = Vec::with_capacity(num_rounds);
    let mut round_polynomials = Vec::with_capacity(num_rounds);

    for _round in 0..num_rounds {
        let poly_fr = match state.round_poly() {
            Ok(poly) => poly,
            Err(error) => return Some(Err(cuda_error(error))),
        };
        let poly = fr_poly_into::<F>(poly_fr)?;
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_sum {
            return Some(Err(Stage1KernelError::InvalidProof {
                driver: "stage1.outer.remaining",
                reason: "dense outer remaining claim mismatch",
            }));
        }
        let challenge = observe_round(&poly);
        running_sum = poly.evaluate(challenge);
        let challenge_fr = into_fr(challenge)?;
        if let Err(error) = state.bind(challenge_fr) {
            return Some(Err(cuda_error(error)));
        }
        point.push(challenge);
        round_polynomials.push(poly);
    }
    Some(Ok((point, round_polynomials)))
}

fn cuda_error(error: CudaError) -> Stage1KernelError {
    let _ = error;
    Stage1KernelError::InvalidProof {
        driver: "stage1.outer.remaining",
        reason: "cuda dense outer remaining failed",
    }
}

#[derive(Clone, Copy)]
pub struct DenseOuterInputs<'a> {
    pub eq_evals: &'a [Fr],
    pub scale: Fr,
    pub weights: &'a [Fr],
    pub row_dots_a: &'a [Fr],
    pub row_dots_b: &'a [Fr],
    pub row_count: usize,
    pub first_group_rows: &'a [u32],
    pub second_group_rows: &'a [u32],
}

pub struct CudaDenseOuterState<'a> {
    ctx: &'a CudaKernelContext,
    eq: DeviceFrVec,
    az: DeviceFrVec,
    bz: DeviceFrVec,
    eq_scratch: DeviceFrVec,
    az_scratch: DeviceFrVec,
    bz_scratch: DeviceFrVec,
}

impl<'a> CudaDenseOuterState<'a> {
    pub fn from_host(
        ctx: &'a CudaKernelContext,
        eq: &[Fr],
        az: &[Fr],
        bz: &[Fr],
    ) -> Result<Self, CudaError> {
        Ok(Self {
            ctx,
            eq: ctx.upload(eq)?,
            az: ctx.upload(az)?,
            bz: ctx.upload(bz)?,
            eq_scratch: ctx.upload(&[])?,
            az_scratch: ctx.upload(&[])?,
            bz_scratch: ctx.upload(&[])?,
        })
    }

    pub fn from_row_dots(
        ctx: &'a CudaKernelContext,
        inputs: DenseOuterInputs<'_>,
    ) -> Result<Self, CudaError> {
        let (eq, az, bz) = ctx.dense_outer_construct(
            inputs.eq_evals,
            inputs.scale,
            inputs.weights,
            inputs.row_dots_a,
            inputs.row_dots_b,
            inputs.row_count,
            inputs.first_group_rows,
            inputs.second_group_rows,
        )?;
        Ok(Self {
            ctx,
            eq,
            az,
            bz,
            eq_scratch: ctx.upload(&[])?,
            az_scratch: ctx.upload(&[])?,
            bz_scratch: ctx.upload(&[])?,
        })
    }

    pub fn round_poly(&self) -> Result<UnivariatePoly<Fr>, CudaError> {
        let coeffs = self.ctx.cubic_accumulate(&self.eq, &self.az, &self.bz)?;
        Ok(UnivariatePoly::new(coeffs.to_vec()))
    }

    pub fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        bind_dense_evals_reuse_cuda(self.ctx, &mut self.eq, &mut self.eq_scratch, challenge)?;
        bind_dense_evals_reuse_cuda(self.ctx, &mut self.az, &mut self.az_scratch, challenge)?;
        bind_dense_evals_reuse_cuda(self.ctx, &mut self.bz, &mut self.bz_scratch, challenge)?;
        Ok(())
    }

    pub fn eq(&self) -> Result<Vec<Fr>, CudaError> {
        self.eq.to_host()
    }

    pub fn az(&self) -> Result<Vec<Fr>, CudaError> {
        self.az.to_host()
    }

    pub fn bz(&self) -> Result<Vec<Fr>, CudaError> {
        self.bz.to_host()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stage1::DenseOuterState;
    use jolt_field::Field;
    use proptest::prelude::*;

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    fn triple_strategy(max_vars: usize) -> impl Strategy<Value = (Vec<Fr>, Vec<Fr>, Vec<Fr>)> {
        (1usize..max_vars).prop_flat_map(|num_vars| {
            let len = 1usize << num_vars;
            (
                prop::collection::vec(fr_strategy(), len),
                prop::collection::vec(fr_strategy(), len),
                prop::collection::vec(fr_strategy(), len),
            )
        })
    }

    proptest! {
        #[test]
        fn cuda_dense_outer_matches_cpu((eq, az, bz) in triple_strategy(10)) {
            let ctx = CudaKernelContext::new(0).unwrap();
            let mut cpu = DenseOuterState::from_raw(eq.clone(), az.clone(), bz.clone());
            let mut gpu = CudaDenseOuterState::from_host(&ctx, &eq, &az, &bz).unwrap();

            for round in 0..eq.len().trailing_zeros() {
                let gpu_poly = gpu.round_poly().unwrap();
                let cpu_poly = cpu.round_poly();
                prop_assert_eq!(gpu_poly.coefficients(), cpu_poly.coefficients());
                prop_assert_eq!(gpu.eq().unwrap(), cpu.eq().to_vec());
                prop_assert_eq!(gpu.az().unwrap(), cpu.az().to_vec());
                prop_assert_eq!(gpu.bz().unwrap(), cpu.bz().to_vec());

                let challenge = Fr::from_u64((round + 1) as u64);
                cpu.bind(challenge);
                gpu.bind(challenge).unwrap();
            }

            prop_assert_eq!(gpu.eq().unwrap(), cpu.eq().to_vec());
        }
    }

    const FIRST_GROUP_ROWS: [u32; 10] = [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];
    const SECOND_GROUP_ROWS: [u32; 9] = [0, 7, 8, 9, 10, 12, 13, 15, 16];
    const ROW_COUNT: usize = 19;

    fn cpu_from_row_dots(
        eq_evals: &[Fr],
        scale: Fr,
        weights: &[Fr],
        row_dots_a: &[Fr],
        row_dots_b: &[Fr],
    ) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        let len = eq_evals.len();
        let cycles = len / 2;
        let mut eq = vec![Fr::from_u64(0); len];
        let mut az = vec![Fr::from_u64(0); len];
        let mut bz = vec![Fr::from_u64(0); len];
        let matvec = |rows: &[u32], cycle: usize| -> (Fr, Fr) {
            let base = cycle * ROW_COUNT;
            let mut a = Fr::from_u64(0);
            let mut b = Fr::from_u64(0);
            for (&row, &weight) in rows.iter().zip(weights.iter()) {
                a += weight * row_dots_a[base + row as usize];
                b += weight * row_dots_b[base + row as usize];
            }
            (a, b)
        };
        for cycle in 0..cycles {
            let index = cycle << 1;
            let (az0, bz0) = matvec(&FIRST_GROUP_ROWS, cycle);
            let (az1, bz1) = matvec(&SECOND_GROUP_ROWS, cycle);
            eq[index] = eq_evals[index] * scale;
            eq[index + 1] = eq_evals[index + 1] * scale;
            az[index] = az0;
            bz[index] = bz0;
            az[index + 1] = az1;
            bz[index + 1] = bz1;
        }
        (eq, az, bz)
    }

    proptest! {
        #[test]
        fn cuda_from_row_dots_matches_cpu(
            num_vars in 1usize..10,
            scale in fr_strategy(),
            weights in prop::collection::vec(fr_strategy(), 10),
            seed in fr_strategy(),
        ) {
            let len = 1usize << num_vars;
            let cycles = len / 2;
            let eq_evals: Vec<Fr> = (0..len).map(|i| seed + Fr::from_u64(i as u64)).collect();
            let row_dots_a: Vec<Fr> =
                (0..cycles * ROW_COUNT).map(|i| seed + Fr::from_u64((i + 1) as u64)).collect();
            let row_dots_b: Vec<Fr> =
                (0..cycles * ROW_COUNT).map(|i| seed + Fr::from_u64((i + 7) as u64)).collect();

            let (eq, az, bz) =
                cpu_from_row_dots(&eq_evals, scale, &weights, &row_dots_a, &row_dots_b);

            let ctx = CudaKernelContext::new(0).unwrap();
            let gpu = CudaDenseOuterState::from_row_dots(
                &ctx,
                DenseOuterInputs {
                    eq_evals: &eq_evals,
                    scale,
                    weights: &weights,
                    row_dots_a: &row_dots_a,
                    row_dots_b: &row_dots_b,
                    row_count: ROW_COUNT,
                    first_group_rows: &FIRST_GROUP_ROWS,
                    second_group_rows: &SECOND_GROUP_ROWS,
                },
            )
            .unwrap();

            prop_assert_eq!(gpu.eq().unwrap(), eq);
            prop_assert_eq!(gpu.az().unwrap(), az);
            prop_assert_eq!(gpu.bz().unwrap(), bz);
        }
    }
}
