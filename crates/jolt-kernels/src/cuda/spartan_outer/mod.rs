//! CUDA Spartan outer uni-skip kernels.
//!
//! The host derives weighted linear forms from the canonical R1CS matrices;
//! device kernels evaluate them against native-width input columns and reduce
//! the extended-node sums. Unlike optimized's integer row evaluator and finite
//! differences, CUDA applies those uploaded field coefficients directly.

#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_claims::protocols::composed::r1cs::spartan_outer_constraints;
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
use jolt_r1cs::constraint::ConstraintMatrices;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessPlane;

use columns::DeviceR1csInputs;
use witness::VARIABLES;

use crate::cuda::common::context::context_for;
use crate::cuda::common::device_columns::device_trace_columns;
use crate::cuda::common::devices::{fan_out, witness_windows, CycleWindow, DeviceTask};
use crate::cuda::witness::session_window_residency;
use crate::cuda::{require_context, CudaBackend};
use crate::uniskip::UniskipKernel;
use crate::{KernelError, ProofSession};

pub(crate) mod columns;
pub(crate) mod uniskip;
pub(crate) mod witness;

#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SpartanOuterState<F: JoltField> {
    tau: Vec<F>,
    log_t: usize,
    extended: Vec<F>,
}

fn uniskip_extended<F: JoltField>(
    inputs: &[(DeviceR1csInputs, CycleWindow)],
    matrices: &ConstraintMatrices<F>,
    tau: &[F],
    log_t: usize,
) -> Result<Vec<F>, KernelError<F>> {
    let shards = inputs.len();
    let context = context_for(0).ok_or(KernelError::InvariantViolation {
        reason: "a Spartan outer uni-skip window names an absent device",
    })?;
    if shards <= 1 {
        let (whole, _) = inputs.first().ok_or(KernelError::InvariantViolation {
            reason: "the Spartan outer uni-skip reduce has no window",
        })?;
        return Ok(uniskip::extended_evals(
            context, whole, matrices, tau, log_t,
        )?);
    }

    let mut tasks: Vec<DeviceTask<'_, Vec<F>, KernelError<F>>> = Vec::with_capacity(shards);
    for (ordinal, (shard_input, window)) in inputs.iter().enumerate() {
        let cycles = window.len;
        tasks.push(Box::new(move || {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a Spartan outer uni-skip window names an absent device",
            })?;
            Ok(
                tracing::info_span!("so_uniskip_window", device = ordinal, cycles).in_scope(
                    || {
                        uniskip::extended_evals_window(
                            device,
                            shard_input,
                            matrices,
                            tau,
                            log_t,
                            ordinal,
                            shards,
                            cycles,
                        )
                    },
                )?,
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

impl<F: JoltField> UniskipKernel<F, OuterRemainder<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        let _ = require_context::<F>()?;
        if log_t == 0 || tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "the Spartan outer tau spans the cycle, stream and uni-skip variables",
            });
        }

        let cycles = 1usize << log_t;
        let matrices = spartan_outer_constraints::<F>();
        if matrices.num_constraints != SPARTAN_OUTER_ROWS || matrices.num_vars <= VARIABLES {
            return Err(KernelError::Unsupported {
                reason: "the CUDA Spartan outer kernels cover the 19-row RV64 constraint system                          over its 35 inputs; the field-inline system appends rows and columns",
            });
        }
        let windows = witness_windows(cycles);
        let mut inputs = Vec::with_capacity(windows.len());
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a Spartan outer window names an absent device",
            })?;
            let (trace, atoms) =
                session_window_residency(device, session, witness, cycles, window)?;
            let pc_words =
                device_trace_columns::<F>(device, session, witness, cycles, window, [0, 1, 0], 0)?
                    .pc;
            let resident = window.residency(cycles).len;
            inputs.push((
                DeviceR1csInputs::from_device(device, &trace, &atoms, &pc_words, resident)?,
                CycleWindow {
                    start: window.start,
                    len: window.len,
                },
            ));
        }
        let extended = uniskip_extended(&inputs, &matrices, tau, log_t)?;

        session.park(SpartanOuterState {
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
        _inputs: &(),
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::composed::r1cs::spartan_outer_constraints;
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_witness::collect_bundles;
    use proptest::prelude::*;
    use witness::SIGNED_WIDE;
    use witness::SIGN_BIT_BASE;
    use witness::WIDE;

    use super::columns::DeviceR1csInputs;
    use super::uniskip;
    use super::witness::{self, SpartanOuterWitness};
    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::devices::CycleWindow;
    use crate::cuda::common::testing::{arb_point, fr, with_r1cs_witness};
    use crate::cuda::witness::{session_atom_columns_window, session_device_trace_window};
    use crate::reference::ReferenceBackend;
    use crate::uniskip::UniskipKernel;
    use crate::ProofSession;

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn fixture_r1cs_columns_carry_both_signs() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let rows = collect_bundles::<SpartanOuterWitness>(witness, 1usize << LOG_T)
                .expect("the fixture serves every bundle field");
            let packed = witness::pack(&rows);
            let cycles = packed.flags.len();
            let mut both_signs = 0;
            for slot in 0..WIDE {
                let bit = SIGN_BIT_BASE + slot as u32;
                let negative = packed
                    .flags
                    .iter()
                    .filter(|mask| (*mask >> bit) & 1 == 1)
                    .count();
                let nonzero = packed
                    .wide
                    .chunks(WIDE * 2)
                    .filter(|limbs| limbs[2 * slot] != 0 || limbs[2 * slot + 1] != 0)
                    .count();
                assert!(
                    nonzero > 0,
                    "wide column {slot} is zero at every cycle, so its coefficient could be \
                     anything",
                );
                if slot >= SIGNED_WIDE {
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
                    &ReferenceBackend, &mut expected_session, &[], &(),
                ).expect("reference uni-skip first-round polynomial");

                let mut got_session = ProofSession::default();
                UniskipKernel::<Fr, OuterRemainder<Fr>>::prepare(
                    &CudaBackend, &mut got_session, LOG_T, &tau, witness,
                ).expect("cuda uni-skip prepare");
                let got = UniskipKernel::<Fr, OuterRemainder<Fr>>::first_round_poly(
                    &CudaBackend, &mut got_session, &[], &(),
                ).expect("cuda uni-skip first-round polynomial");

                prop_assert_eq!(
                    got.coefficients().to_vec(),
                    expected.coefficients().to_vec(),
                    "the uni-skip first-round polynomial diverged"
                );
                Ok(())
            })?;
        }


    }
}
