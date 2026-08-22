use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_r1cs::constraint::ConstraintMatrices;
use jolt_r1cs::constraints::jolt::spartan_outer_constraints;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::common::device_columns::device_pc_words;
use crate::cuda::common::devices::{fan_out, witness_windows, DeviceTask};
use crate::cuda::witness::{
    session_atom_columns, session_atom_columns_window, session_device_trace,
    session_device_trace_window,
};

use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::{require_context, CudaBackend};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod columns;
pub(crate) mod remainder;
pub(crate) mod uniskip;
pub(crate) mod witness;

use columns::DeviceR1csInputs;
use remainder::{DeviceRemainder, SpartanOuterRemainderKernel};

pub struct SpartanOuterState<F: Field> {
    context: &'static CudaKernelContext,
    inputs: DeviceR1csInputs,
    matrices: ConstraintMatrices<F>,
    tau: Vec<F>,
    log_t: usize,
    extended: Vec<F>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for SpartanOuterState<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("inputs"), self.inputs.device_bytes());
        visitor.visit_simple(allocative::Key::new("tau"), self.tau.len() * size_of::<F>());
        visitor.visit_simple(
            allocative::Key::new("extended"),
            self.extended.len() * size_of::<F>(),
        );
        visitor.exit();
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the uni-skip reduce needs the session and witness to build each device's window \
              alongside the inputs, matrices and point the reduce itself consumes"
)]
fn uniskip_extended<F: Field>(
    context: &'static CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    inputs: &DeviceR1csInputs,
    matrices: &ConstraintMatrices<F>,
    tau: &[F],
    log_t: usize,
    cycles: usize,
) -> Result<Vec<F>, KernelError<F>> {
    let windows = witness_windows(cycles);
    let shards = windows.len();
    if shards <= 1 {
        return Ok(uniskip::extended_evals(
            context, inputs, matrices, tau, log_t,
        )?);
    }

    let mut shard_inputs: Vec<DeviceR1csInputs> = Vec::with_capacity(shards - 1);
    for (ordinal, window) in windows.iter().enumerate().skip(1) {
        let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
            reason: "a Spartan outer uni-skip window names an absent device",
        })?;
        let resident = window.residency(cycles);
        let trace = session_device_trace_window(device, session, witness, cycles, &resident)?;
        let atoms = session_atom_columns_window(device, session, witness, cycles, &resident)?;
        let pc_words = trace.mapped_pc_words()?;
        shard_inputs.push(DeviceR1csInputs::from_device(
            device,
            &trace,
            &atoms,
            &pc_words,
            resident.len,
        )?);
    }

    let mut tasks: Vec<DeviceTask<'_, Vec<F>, KernelError<F>>> = Vec::with_capacity(shards);
    for (ordinal, window) in windows.iter().enumerate() {
        let shard_input = if ordinal == 0 {
            inputs
        } else {
            shard_inputs
                .get(ordinal - 1)
                .ok_or(KernelError::InvariantViolation {
                    reason: "a Spartan outer uni-skip window has no device inputs",
                })?
        };
        tasks.push(Box::new(move || {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a Spartan outer uni-skip window names an absent device",
            })?;
            Ok(
                tracing::info_span!("so_uniskip_window", device = ordinal, cycles = window.len)
                    .in_scope(|| {
                        uniskip::extended_evals_window(
                            device,
                            shard_input,
                            matrices,
                            tau,
                            log_t,
                            ordinal,
                            shards,
                            window.len,
                        )
                    })?,
            )
        }));
    }

    let mut parts = fan_out(tasks)?.into_iter();
    let mut extended = parts.next().ok_or(KernelError::InvariantViolation {
        reason: "the Spartan outer uni-skip reduce produced no window",
    })?;
    for part in parts {
        if part.len() != extended.len() {
            return Err(KernelError::InvariantViolation {
                reason: "two Spartan outer uni-skip windows disagree on the extended node count",
            });
        }
        for (total, addend) in extended.iter_mut().zip(&part) {
            *total += *addend;
        }
    }
    Ok(extended)
}

impl<F: Field> UniskipKernel<F, OuterRemainder<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        let context = require_context::<F>()?;
        if log_t == 0 || tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "the Spartan outer tau spans the cycle, stream and uni-skip variables",
            });
        }

        let cycles = 1usize << log_t;
        let trace = session_device_trace(context, session, witness, cycles)?;
        let atoms = session_atom_columns(context, session, witness, cycles)?;
        let pc_words = device_pc_words::<F>(context, session, witness, cycles)?;
        let inputs = DeviceR1csInputs::from_device(context, &trace, &atoms, &pc_words, cycles)?;
        let matrices = spartan_outer_constraints::<F>();
        if matrices.num_constraints != SPARTAN_OUTER_ROWS || matrices.num_vars <= witness::VARIABLES
        {
            return Err(KernelError::Unsupported {
                reason: "the CUDA Spartan outer kernels cover the 19-row RV64 constraint system                          over its 35 inputs; the field-inline system appends rows and columns",
            });
        }
        let extended = uniskip_extended::<F>(
            context, session, witness, &inputs, &matrices, tau, log_t, cycles,
        )?;

        session.park(SpartanOuterState {
            context,
            inputs,
            matrices,
            tau: tau.to_vec(),
            log_t,
            extended,
        });
        Ok(())
    }

    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        _late_tau: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let state =
            session
                .state::<SpartanOuterState<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason:
                        "the outer uni-skip slot parked no kernel for the first-round polynomial",
                })?;
        Ok(uniskip::first_round_poly(
            &state.extended,
            state.tau[state.log_t + 1],
        )?)
    }
}

const SPARTAN_OUTER_ROWS: usize = 19;

impl<F: Field> PrepareKernel<F, OuterRemainder<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        let state =
            session
                .take::<SpartanOuterState<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "the outer uni-skip slot parked no kernel for the remainder member",
                })?;
        let dimensions = SpartanOuterDimensions::rv64(state.log_t);
        if dimensions.variables().len() != witness::VARIABLES {
            return Err(KernelError::Unsupported {
                reason: "the CUDA Spartan outer kernels cover the 35-variable RV64 input set",
            });
        }
        let device = DeviceRemainder::new(
            state.context,
            state.inputs,
            &state.matrices,
            &state.tau,
            state.log_t,
            inputs.relation.uniskip_challenge(),
        )?;
        Ok(Box::new(SpartanOuterRemainderKernel::new(
            state.context,
            device,
            inputs.relation.symbolic().degree(),
            state.log_t,
        )))
    }
}

impl<F: Field> SumcheckKernel<F> for SpartanOuterRemainderKernel<F> {
    type Relation = OuterRemainder<F>;

    fn output_claims(
        &mut self,
        _inputs: &jolt_verifier::stages::relations::SumcheckInputClaims<F, OuterRemainder<F>>,
    ) -> Result<
        jolt_verifier::stages::relations::SumcheckOutputClaims<F, OuterRemainder<F>>,
        SumcheckKernelError<F>,
    > {
        let bound = self.bound_rounds();
        let expected = <Self as jolt_sumcheck::ProveRounds<F>>::num_rounds(self);
        if bound != expected {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: expected.saturating_sub(bound),
            });
        }
        let openings = self
            .openings()
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "the CUDA Spartan outer claim pass failed",
            })?;
        jolt_claims::OutputClaims::from_opening_values(|id| openings.get(id).copied())
            .map_err(SumcheckKernelError::MissingOpeningValue)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_r1cs::constraints::jolt::spartan_outer_constraints;
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainder, OuterRemainderInputClaims,
    };
    use jolt_witness::{collect_bundles, JoltWitnessPlane};
    use proptest::prelude::*;

    use super::columns::DeviceR1csInputs;
    use super::uniskip;
    use super::witness::{self, SpartanOuterWitness};
    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::devices::CycleWindow;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, probe_input_claim, with_r1cs_witness,
    };
    use crate::cuda::witness::{session_atom_columns_window, session_device_trace_window};
    use crate::reference::spartan_outer::{
        materialize_input_tables, row_value_tables, ReferenceOuterRemainder,
    };
    use crate::reference::ReferenceBackend;
    use crate::uniskip::UniskipKernel;
    use crate::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn fixture_r1cs_rows_exercise_every_constraint() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let dimensions = SpartanOuterDimensions::rv64(LOG_T);
            let tables = materialize_input_tables::<Fr>(witness, &dimensions)
                .expect("the fixture serves every R1CS input column");
            let matrices = spartan_outer_constraints::<Fr>();
            let (az, bz) = row_value_tables(&matrices, &tables);
            let zero = Fr::from_u64(0);

            for (row, values) in az.iter().enumerate() {
                assert!(
                    values.iter().any(|value| *value != zero),
                    "row {row}: the guard is off at every cycle, so no fixture cycle exercises \
                     this constraint",
                );
                assert!(
                    values.contains(&zero),
                    "row {row}: the guard is on at every cycle, so a dropped guard would not \
                     change the round polynomials",
                );
            }
            for (row, values) in bz.iter().enumerate() {
                assert!(
                    values.iter().any(|value| *value != zero),
                    "row {row}: the magnitude is zero at every cycle, so a wrong weight on this \
                     row would not change the round polynomials",
                );
            }
        });
    }

    #[test]
    fn fixture_r1cs_columns_carry_both_signs() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let rows = collect_bundles::<SpartanOuterWitness>(witness, 1usize << LOG_T)
                .expect("the fixture serves every bundle field");
            let packed = witness::pack(&rows);
            let cycles = packed.flags.len();
            let mut both_signs = 0;
            for slot in 0..witness::WIDE {
                let bit = witness::SIGN_BIT_BASE + slot as u32;
                let negative = packed
                    .flags
                    .iter()
                    .filter(|mask| (*mask >> bit) & 1 == 1)
                    .count();
                let nonzero = packed
                    .wide
                    .chunks(witness::WIDE * 2)
                    .filter(|limbs| limbs[2 * slot] != 0 || limbs[2 * slot + 1] != 0)
                    .count();
                assert!(
                    nonzero > 0,
                    "wide column {slot} is zero at every cycle, so its coefficient could be \
                     anything",
                );
                if slot >= witness::SIGNED_WIDE {
                    assert_eq!(
                        negative, 0,
                        "wide column {slot} is the unsigned one, so it must never set a sign bit",
                    );
                } else if negative > 0 && negative < cycles {
                    both_signs += 1;
                }
            }
            assert!(
                both_signs > 0,
                "no signed wide column carries both signs across the {cycles} fixture cycles, so \
                 the packing's sign path is dead and dropping the coefficient negation would pass \
                 every equivalence test",
            );
        });
    }

    fn prepared(
        witness: &dyn JoltWitnessPlane<Fr>,
        tau: &[Fr],
        uniskip_challenge: Fr,
        cuda: bool,
    ) -> Box<dyn SumcheckKernel<Fr, Relation = OuterRemainder<Fr>>> {
        let mut session = ProofSession::default();
        if cuda {
            UniskipKernel::<Fr, OuterRemainder<Fr>>::prepare(
                &CudaBackend,
                &mut session,
                LOG_T,
                tau,
                witness,
            )
            .expect("cuda uni-skip prepare");
        } else {
            UniskipKernel::<Fr, OuterRemainder<Fr>>::prepare(
                &ReferenceBackend,
                &mut session,
                LOG_T,
                tau,
                witness,
            )
            .expect("reference uni-skip prepare");
        }

        let relation = OuterRemainder::new(
            SpartanOuterDimensions::rv64(LOG_T),
            tau.to_vec(),
            uniskip_challenge,
        );
        let claims = outer_remainder_input_values_from_uniskip_output(Fr::from_u64(0));
        let points = OuterRemainderInputClaims {
            outer_uniskip: Vec::new(),
        };
        let challenges = NoChallenges::default();
        let inputs = ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        if cuda {
            PrepareKernel::<Fr, OuterRemainder<Fr>>::prepare(
                &CudaBackend,
                &mut session,
                witness,
                inputs,
            )
            .expect("cuda remainder prepare")
        } else {
            ReferenceOuterRemainder
                .prepare(&mut session, witness, inputs)
                .expect("reference remainder prepare")
        }
    }

    #[test]
    fn uniskip_cycle_windows_sum_to_the_whole_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        let tau: Vec<Fr> = (0..LOG_T + 2).map(|i| fr(23 + 5 * i as u64)).collect();
        let matrices = spartan_outer_constraints::<Fr>();
        let cycles = 1usize << LOG_T;

        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 9, |witness| {
            let inputs = |session: &mut ProofSession, window: &CycleWindow| {
                let resident = window.residency(cycles);
                let trace =
                    session_device_trace_window::<Fr>(context, session, witness, cycles, &resident)
                        .expect("windowed residency");
                let atoms =
                    session_atom_columns_window::<Fr>(context, session, witness, cycles, &resident)
                        .expect("windowed atom columns");
                let pc_words = trace.mapped_pc_words().expect("windowed pc words");
                DeviceR1csInputs::from_device(context, &trace, &atoms, &pc_words, resident.len)
                    .expect("windowed r1cs inputs")
            };

            let mut session = ProofSession::default();
            let whole_window = CycleWindow {
                start: 0,
                len: cycles,
            };
            let whole = uniskip::extended_evals(
                context,
                &inputs(&mut session, &whole_window),
                &matrices,
                &tau,
                LOG_T,
            )
            .expect("whole-domain extended evals");

            for shards in [2usize, 4] {
                let len = cycles / shards;
                let mut summed = vec![Fr::from_u64(0); whole.len()];
                for shard in 0..shards {
                    let window = CycleWindow {
                        start: shard * len,
                        len,
                    };
                    let part = uniskip::extended_evals_window(
                        context,
                        &inputs(&mut session, &window),
                        &matrices,
                        &tau,
                        LOG_T,
                        shard,
                        shards,
                        len,
                    )
                    .expect("windowed extended evals");
                    for (total, addend) in summed.iter_mut().zip(&part) {
                        *total += *addend;
                    }
                }
                prop_assert_eq!(
                    summed,
                    whole.clone(),
                    "the uni-skip extended evaluations over {} cycle windows must sum to the \
                     whole-domain values: each node is a sum over cycles, and a contiguous cycle \
                     window is a contiguous window of the split-eq outer factor",
                    shards
                );
            }
            Ok(())
        })
        .expect("r1cs witness fixture");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn spartan_outer_uniskip_first_round_poly_matches_reference(
            seed in any::<u64>(),
            tau in arb_point(LOG_T + 2),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let mut expected_session = ProofSession::default();
                UniskipKernel::<Fr, OuterRemainder<Fr>>::prepare(
                    &ReferenceBackend, &mut expected_session, LOG_T, &tau, witness,
                ).expect("reference uni-skip prepare");
                let expected = UniskipKernel::<Fr, OuterRemainder<Fr>>::first_round_poly(
                    &ReferenceBackend, &mut expected_session, &[],
                ).expect("reference uni-skip first-round polynomial");

                let mut got_session = ProofSession::default();
                UniskipKernel::<Fr, OuterRemainder<Fr>>::prepare(
                    &CudaBackend, &mut got_session, LOG_T, &tau, witness,
                ).expect("cuda uni-skip prepare");
                let got = UniskipKernel::<Fr, OuterRemainder<Fr>>::first_round_poly(
                    &CudaBackend, &mut got_session, &[],
                ).expect("cuda uni-skip first-round polynomial");

                prop_assert_eq!(
                    got.coefficients().to_vec(),
                    expected.coefficients().to_vec(),
                    "the uni-skip first-round polynomial diverged"
                );
                Ok(())
            })?;
        }

        #[test]
        fn spartan_outer_remainder_matches_reference_round_for_round(
            seed in any::<u64>(),
            tau in arb_point(LOG_T + 2),
            uniskip_challenge in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T + 1),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let input_claim = probe_input_claim(
                    &mut *prepared(witness, &tau, uniskip_challenge, false),
                );
                let mut expected_kernel = prepared(witness, &tau, uniskip_challenge, false);
                let mut got_kernel = prepared(witness, &tau, uniskip_challenge, true);

                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged");

                let wire = outer_remainder_input_values_from_uniskip_output(input_claim);
                let expected_claims =
                    expected_kernel.output_claims(&wire).expect("reference claims");
                let got_claims = got_kernel.output_claims(&wire).expect("cuda claims");
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
