use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
    RegistersClaimReductionInputClaims, RegistersClaimReductionOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
use jolt_witness::JoltWitnessPlane;

use std::sync::Arc;

use jolt_witness::backend::cuda::{DeviceTrace, EXTRA_RD_POST, EXTRA_RS1, EXTRA_RS2, EXTRA_WORDS};

use crate::cuda::common::context::context_for;
use crate::cuda::common::devices::{witness_windows, CycleWindow};
use crate::cuda::common::half_fold::{NarrowColumn, NarrowKind};
use crate::cuda::common::prefix_suffix::{NarrowColumns, PrefixSuffixWindow};
use crate::cuda::witness::session_window_residency;

use super::common::prefix_suffix::{
    eq_pair, prefix_rounds_ceil, ColumnSet, PrefixSuffixGroup, PrefixSuffixRounds,
};
use super::{require_context, CudaBackend};
use crate::reference::ReferenceBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod witness;

const COLUMNS: usize = 3;

pub struct RegistersClaimReductionKernel<F: Field> {
    rounds: PrefixSuffixRounds<F>,
    total: usize,
    bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for RegistersClaimReductionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("rounds"), self.rounds.device_bytes());
        visitor.exit();
    }
}

impl<F: Field> ProveRounds<F> for RegistersClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.total
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if bind.is_some() {
            self.bound += 1;
        }
        self.rounds.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bound += 1;
        self.rounds.finish_rounds(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for RegistersClaimReductionKernel<F> {
    type Relation = RegistersClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &RegistersClaimReductionInputClaims<F>,
    ) -> Result<RegistersClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.total - self.bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let claims =
            self.rounds
                .column_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA registers claim reduction failed to read back its column claims",
                })?;
        let [rd_write_value, rs1_value, rs2_value] = claims.as_slice() else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers claim reduction produces one claim per reduced column",
            });
        };
        Ok(RegistersClaimReductionOutputClaims {
            rd_write_value: *rd_write_value,
            rs1_value: *rs1_value,
            rs2_value: *rs2_value,
        })
    }
}

struct ExtraWordColumns {
    trace: Arc<DeviceTrace>,
    entries: usize,
}

impl NarrowColumns for ExtraWordColumns {
    fn count(&self) -> usize {
        COLUMNS
    }

    fn entries(&self) -> usize {
        self.entries
    }

    fn column(&self, index: usize) -> Option<NarrowColumn<'_>> {
        let word = *[EXTRA_RD_POST, EXTRA_RS1, EXTRA_RS2].get(index)?;
        Some(NarrowColumn {
            words: self.trace.extras(),
            kind: NarrowKind::U64,
            len: self.entries,
            stride: EXTRA_WORDS,
            offset: word,
        })
    }
}
impl<F: Field> PrepareKernel<F, RegistersClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.symbolic().rounds();
        let tau_low = relation.product_uniskip_tau_low();
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "the registers reduction Spartan point has the wrong variable count",
            });
        }
        let Some(prefix_rounds) = prefix_rounds_ceil(log_t) else {
            return ReferenceBackend.prepare(session, witness, inputs);
        };

        let cycles = 1usize << log_t;
        let prefix_len = 1usize << prefix_rounds;
        let windows = witness_windows(cycles);
        let windows = if windows
            .iter()
            .all(|window| window.len.is_multiple_of(prefix_len))
        {
            windows
        } else {
            vec![CycleWindow {
                start: 0,
                len: cycles,
            }]
        };
        let mut column_windows = Vec::with_capacity(windows.len());
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a registers reduction window names an absent device",
            })?;
            let (trace, _) = session_window_residency(device, session, witness, cycles, window)?;
            column_windows.push(PrefixSuffixWindow {
                ordinal,
                columns: ColumnSet::Narrow(Arc::new(ExtraWordColumns {
                    trace,
                    entries: window.len,
                })),
                suffix_offset: window.start / prefix_len,
                suffix_len: window.len / prefix_len,
            });
        }

        let gamma = inputs.challenges.gamma;
        let mut powers = Vec::with_capacity(COLUMNS);
        let mut power = F::one();
        for _ in 0..COLUMNS {
            powers.push(power);
            power *= gamma;
        }
        let group = PrefixSuffixGroup {
            pairs: vec![eq_pair(tau_low, prefix_rounds)?],
            columns: powers.into_iter().enumerate().collect(),
            constant: F::zero(),
        };

        Ok(Box::new(RegistersClaimReductionKernel {
            rounds: PrefixSuffixRounds::new_windowed(
                context,
                column_windows,
                vec![group],
                prefix_rounds,
                log_t,
            )?,
            total: log_t,
            bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::registers::{
        rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
        RegistersClaimReductionChallenges, RegistersClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
    use jolt_witness::JoltWitnessPlane;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::half_fold::{half_fold, FoldColumn, SummedHalf};
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_r1cs_witness,
    };
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    fn column_ids() -> [JoltOpeningId; 3] {
        [
            rd_write_value_reduced(),
            rs1_value_reduced(),
            rs2_value_reduced(),
        ]
    }

    #[test]
    fn fixture_registers_claim_reduction_columns_vary() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let plane: &dyn JoltWitnessPlane<Fr> = witness;
            let zero = Fr::from_u64(0);
            for id in column_ids() {
                let column = dense_view::<Fr>(plane, id).expect("the fixture serves the column");
                assert!(
                    column.iter().any(|value| *value != zero),
                    "{id:?} is zero at every cycle, so its gamma power could be anything",
                );
                assert!(
                    column.iter().any(|value| *value != column[0]),
                    "{id:?} is constant across the fixture, so a mis-indexed read would pass",
                );
            }
        });
    }

    #[test]
    fn a_one_round_trace_is_served_below_the_prefix_suffix_minimum() {
        let Some(_) = shared_context() else {
            return;
        };
        const TINY: usize = 1;
        let point = vec![fr(7)];
        let challenges = vec![fr(11)];
        with_r1cs_witness(TINY, RAM_K, one_hot(), 3, |witness| {
            let relation =
                RegistersClaimReduction::<Fr>::new(TraceDimensions::new(TINY), point.clone());
            let claims = RegistersClaimReductionInputClaims::default();
            let points = RegistersClaimReductionInputClaims::default();
            let challenge_set = RegistersClaimReductionChallenges { gamma: fr(5) };
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };

            let input_claim = reference_input_claim(witness, make_inputs);
            let mut expected_kernel = ReferenceBackend
                .prepare(&mut ProofSession::default(), witness, make_inputs())
                .expect("reference prepare");
            let mut got_kernel = CudaBackend
                .prepare(&mut ProofSession::default(), witness, make_inputs())
                .expect("cuda prepare below the prefix-suffix minimum");

            let expected = drive(&mut *expected_kernel, input_claim, &challenges);
            let got = drive(&mut *got_kernel, input_claim, &challenges);
            assert_eq!(got, expected, "round polynomials diverged at log_T = 1");
            assert_eq!(
                got_kernel
                    .output_claims(&claims)
                    .expect("cuda claims")
                    .opening_values(),
                expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims")
                    .opening_values(),
                "output claims diverged at log_T = 1",
            );
        });
    }

    #[test]
    fn device_columns_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let cycles = 1usize << LOG_T;
            let rows: Vec<super::witness::RegistersClaimReductionWitness> =
                jolt_witness::collect_bundles(witness, cycles).expect("reference rows");
            let expected: Vec<Vec<Fr>> = super::witness::device_columns(context, &rows)
                .expect("host-encoded columns")
                .iter()
                .map(|column| column.to_host().expect("download"))
                .collect();

            let mut session = ProofSession::default();
            let trace = crate::cuda::witness::session_device_trace::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
            )
            .expect("device residency");
            let narrow = super::ExtraWordColumns {
                trace,
                entries: cycles,
            };
            let one = context.upload(&[Fr::from_u64(1)]).expect("upload weight");
            let got: Vec<Vec<Fr>> = (0..super::COLUMNS)
                .map(|index| {
                    let column =
                        super::NarrowColumns::column(&narrow, index).expect("narrow column");
                    half_fold(
                        context,
                        FoldColumn::Narrow(column),
                        &one,
                        SummedHalf::High,
                        Fr::from_u64(1),
                    )
                    .expect("promote the narrow column")
                    .to_host()
                    .expect("download")
                })
                .collect();

            assert!(
                expected[0].iter().any(|value| *value != expected[0][0]),
                "every rd write value is identical, so a kernel ignoring the row would pass",
            );
            assert_eq!(
                got, expected,
                "the device columns diverge from the host encoder",
            );
        });
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn registers_claim_reduction_matches_reference_round_for_round(
            seed in any::<u64>(),
            product_uniskip_tau_low in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = RegistersClaimReduction::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    product_uniskip_tau_low.clone(),
                );
                let claims = RegistersClaimReductionInputClaims::default();
                let points = RegistersClaimReductionInputClaims::default();
                let challenge_set = RegistersClaimReductionChallenges { gamma };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let input_claim = reference_input_claim(witness, make_inputs);
                let mut expected_kernel = ReferenceBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("reference prepare");
                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let expected_claims =
                    expected_kernel.output_claims(&claims).expect("reference claims");

                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare");
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged");

                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged"
                );
                Ok(())
            })?;
        }
    }
}
