use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};

use super::columns::ShardedInstructionColumns;
use super::witness::{
    COLUMNS, IMM_COLUMN, LEFT_IS_PC_COLUMN, LEFT_IS_RS1_COLUMN, RIGHT_IS_IMM_COLUMN,
    RIGHT_IS_RS2_COLUMN, RS1_VALUE_COLUMN, RS2_VALUE_COLUMN, UNEXPANDED_PC_COLUMN,
};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{require_fr, require_fr_slice, DeviceFrVec};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::half_fold::FoldColumn;
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
        columns: &ShardedInstructionColumns<F>,
        previous_claim: F,
        degree: usize,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        let mut coefficients = match self {
            Self::EvalPoints { eq, form } => {
                let whole = columns.whole()?;
                let mut handles = whole.columns()?;
                handles.push(FoldColumn::Field(eq));
                let evals: Vec<F> =
                    form.round_lanes(context, &handles, whole.len() / 2, 1, true, degree)?;
                let mut toom = Vec::with_capacity(evals.len() + 1);
                toom.push(previous_claim - evals[0]);
                toom.extend_from_slice(&evals);
                UnivariatePoly::from_evals_toom(&toom).into_coefficients()
            }
            Self::Gruen { eq, form } => {
                let (constant, leading) = columns.round_endpoints(form, eq)?;
                eq.gruen_poly_deg_3(constant, leading, previous_claim)
                    .into_coefficients()
            }
        };
        coefficients.resize(degree + 1, F::zero());
        Ok(UnivariatePoly::new(coefficients))
    }
}

pub(crate) fn gruen_form<F: Field>(
    context: &CudaKernelContext,
    gamma: F,
) -> Result<DeviceSumOfProducts, CudaError> {
    leaves(gamma, false)?.upload(context)
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::BindingOrder;

    use super::super::columns::{ColumnShard, DeviceInstructionColumns, ShardedInstructionColumns};
    use super::super::witness::COLUMNS;
    use super::leaves;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::split_eq::DeviceSplitEq;
    use crate::cuda::common::testing::fr;

    const LOG_T: usize = 8;

    #[test]
    fn sharded_instruction_columns_match_the_whole_domain_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        let gamma = fr(59);
        let form = leaves(gamma, false)
            .expect("gruen leaves")
            .upload(context)
            .expect("upload form");
        let all: Vec<Vec<Fr>> = (0..COLUMNS)
            .map(|column| {
                (0..cycles)
                    .map(|cycle| fr((7 + column * 31 + cycle * 13) as u64))
                    .collect()
            })
            .collect();
        let point: Vec<Fr> = (0..LOG_T).map(|i| fr(83 + 11 * i as u64)).collect();

        let build = |base: usize, len: usize| {
            let window: Vec<Vec<Fr>> = all
                .iter()
                .map(|column| column[base..base + len].to_vec())
                .collect();
            DeviceInstructionColumns::from_dense_for_test(context, &window).expect("window columns")
        };

        for shards in [2usize, 4] {
            let mut expected = build(0, cycles);
            let mut expected_eq =
                DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                    .expect("whole split-eq");
            let len = cycles / shards;
            let windows: Vec<ColumnShard<Fr>> = (0..shards)
                .map(|shard| ColumnShard {
                    ordinal: 0,
                    columns: build(shard * len, len),
                    eq: DeviceSplitEq::<Fr>::new_window(
                        context,
                        &point,
                        BindingOrder::LowToHigh,
                        shard,
                        shards,
                    )
                    .expect("window split-eq"),
                    form: super::gruen_form(context, gamma).expect("window form"),
                })
                .collect();
            let mut got = ShardedInstructionColumns::new(windows, LOG_T).expect("sharded columns");
            let mut got_eq = DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                .expect("tail split-eq");

            for round in 0..LOG_T {
                let want: (Fr, Fr) = form
                    .round_gruen_endpoints(
                        context,
                        &expected.columns().expect("whole columns"),
                        expected.len() / 2,
                        &expected_eq,
                    )
                    .expect("whole endpoints");
                let have: (Fr, Fr) = got
                    .round_endpoints(&form, &got_eq)
                    .expect("sharded endpoints");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: the Gruen endpoints are sums over the \
                     remaining cycles, so a window's pair must add to the whole domain's",
                );
                let challenge = fr(900 + 7 * round as u64);
                expected.bind(context, challenge).expect("whole bind");
                expected_eq.bind(challenge);
                got.bind(challenge, round).expect("sharded bind");
                got_eq.bind(challenge);
            }

            let want: Vec<Fr> = expected.finals().expect("whole finals");
            let have: Vec<Fr> = got.finals().expect("sharded finals");
            assert_eq!(have, want, "shards={shards}: the column finals diverged");
            assert_eq!(want.len(), COLUMNS);
            assert_ne!(
                want.first().copied(),
                Some(Fr::from_u64(0)),
                "a degenerate fixture would hide a divergence",
            );
        }
    }
}
