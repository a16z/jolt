use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};

use super::columns::DeviceInstructionColumns;
use super::witness::{
    COLUMNS, IMM_COLUMN, LEFT_IS_PC_COLUMN, LEFT_IS_RS1_COLUMN, RIGHT_IS_IMM_COLUMN,
    RIGHT_IS_RS2_COLUMN, RS1_VALUE_COLUMN, RS2_VALUE_COLUMN, UNEXPANDED_PC_COLUMN,
};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{require_fr, require_fr_slice, DeviceFrVec};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::cuda::common::sum_of_products::{DeviceSumOfProducts, SumOfProducts};

const EQ_TABLE_INDEX: usize = COLUMNS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundBasis {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the eval-point basis is the correctness reference the shipped Gruen basis is \
                      checked against; only the equivalence test constructs it"
        )
    )]
    EvalPoints,
    Gruen,
}

pub enum Basis<F: Field> {
    EvalPoints {
        eq: DeviceFrVec,
        form: DeviceSumOfProducts,
    },
    Gruen {
        eq: DeviceSplitEq<F>,
        form: DeviceSumOfProducts,
    },
}

impl<F: Field> Basis<F> {
    pub fn new(
        context: &CudaKernelContext,
        basis: RoundBasis,
        point: &[F],
        gamma: F,
    ) -> Result<Self, CudaError> {
        match basis {
            RoundBasis::EvalPoints => {
                let eq = context.eq_evals(require_fr_slice(point)?)?;
                let form = leaves(gamma, true)?.upload(context)?;
                Ok(Self::EvalPoints { eq, form })
            }
            RoundBasis::Gruen => {
                let eq = DeviceSplitEq::new(context, point, BindingOrder::LowToHigh)?;
                let form = leaves(gamma, false)?.upload(context)?;
                Ok(Self::Gruen { eq, form })
            }
        }
    }

    pub fn bind(&mut self, context: &CudaKernelContext, challenge: F) -> Result<(), CudaError> {
        match self {
            Self::EvalPoints { eq, .. } => {
                let len = eq.len();
                if len < 2 {
                    return Err(CudaError::LengthMismatch {
                        expected: 2,
                        got: len,
                    });
                }
                *eq = context.bind_rows(eq, len, require_fr(challenge)?)?;
            }
            Self::Gruen { eq, .. } => eq.bind(challenge),
        }
        Ok(())
    }

    pub fn round_poly(
        &self,
        context: &CudaKernelContext,
        columns: &DeviceInstructionColumns,
        previous_claim: F,
        degree: usize,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        let half = columns.len() / 2;
        let mut coefficients = match self {
            Self::EvalPoints { eq, form } => {
                let mut handles = columns.handles();
                handles.push(eq);
                let evals: Vec<F> = form.round_lanes(context, &handles, half, 1, true, degree)?;
                let mut toom = Vec::with_capacity(evals.len() + 1);
                toom.push(previous_claim - evals[0]);
                toom.extend_from_slice(&evals);
                UnivariatePoly::from_evals_toom(&toom).into_coefficients()
            }
            Self::Gruen { eq, form } => {
                let (constant, leading) =
                    form.round_gruen_endpoints(context, &columns.handles(), half, eq)?;
                eq.gruen_poly_deg_3(constant, leading, previous_claim)
                    .into_coefficients()
            }
        };
        coefficients.resize(degree + 1, F::zero());
        Ok(UnivariatePoly::new(coefficients))
    }
}

fn leaves<F: Field>(gamma: F, with_eq: bool) -> Result<SumOfProducts<F>, CudaError> {
    let mut form = SumOfProducts::new();
    let one = F::one();
    for (coefficient, flag, value) in [
        (one, RIGHT_IS_RS2_COLUMN, RS2_VALUE_COLUMN),
        (one, RIGHT_IS_IMM_COLUMN, IMM_COLUMN),
        (gamma, LEFT_IS_RS1_COLUMN, RS1_VALUE_COLUMN),
        (gamma, LEFT_IS_PC_COLUMN, UNEXPANDED_PC_COLUMN),
    ] {
        if with_eq {
            form.push(coefficient, &[flag, value, EQ_TABLE_INDEX])?;
        } else {
            form.push(coefficient, &[flag, value])?;
        }
    }
    Ok(form)
}
