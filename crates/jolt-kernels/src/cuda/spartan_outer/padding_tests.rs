#![expect(
    clippy::unwrap_used,
    reason = "test failures include device and fixture errors"
)]

use std::sync::Arc;

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_claims::protocols::jolt::JoltOneHotConfig;
use jolt_claims::{NoChallenges, OutputClaims};
use jolt_field::{Fr, Zero};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{
    JoltInstructionKind as Kind, JoltInstructionRow, NormalizedOperands, RamAccess, RegisterRead,
    RegisterState, RV64IMAC_JOLT,
};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder, OuterRemainderInputClaims,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use super::super::columns::DeviceR1csInputs;
use super::super::witness::{Packed, NARROW, WIDE};
use super::super::CudaBackend;
use super::{DeviceRemainder, RemainderShard, SpartanOuterRemainderKernel};
use crate::cuda::common::context::{context_for, device_count, shared_context};
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::cuda::common::testing::{drive, fr, probe_input_claim, synthetic_row};
use crate::reference::spartan_outer::ReferenceOuterRemainder;
use crate::uniskip::UniskipKernel;
use crate::{PrepareKernel, ProofSession, ProverInputs, ReferenceBackend, SumcheckKernel};

#[test]
fn periodic_stream_tails_match_dense_polynomials() {
    let Some(context) = shared_context() else {
        return;
    };
    const LOG_T: usize = 6;
    const CYCLES: usize = 1 << LOG_T;
    for count in 1..=device_count().min(2) {
        let width = CYCLES / count;
        for prefix in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64] {
            for case in 0..4 {
                let point: Vec<_> = (0..=LOG_T)
                    .map(|i| match case {
                        1 => Fr::from((i % 2) as u64),
                        2 => Fr::from(((i + 1) % 2) as u64),
                        _ => fr(i as u64 + 11),
                    })
                    .collect();
                let scale = if case == 3 { Fr::zero() } else { fr(91) };
                let mut factors: Vec<_> = (0..2)
                    .map(|lane| {
                        Polynomial::new(
                            (0..2 * CYCLES)
                                .map(|i| {
                                    fr((lane * CYCLES * 2 + 2 * (i / 2).min(prefix) + i % 2) as u64
                                        + 19)
                                })
                                .collect(),
                        )
                    })
                    .collect();
                let shards = (0..count)
                    .map(|ordinal| {
                        let device = context_for(ordinal).unwrap();
                        let start = ordinal * width;
                        let retained = (prefix.saturating_sub(start) + 1).min(width);
                        let inputs = DeviceR1csInputs::new(
                            device,
                            &Packed {
                                narrow: vec![0; width * NARROW],
                                wide: vec![0; width * WIDE * 2],
                                flags: vec![0; width],
                            },
                        )
                        .unwrap();
                        RemainderShard {
                            ordinal,
                            inputs,
                            cycles: width,
                            az: device
                                .upload(&factors[0].evals()[2 * start..2 * (start + retained)])
                                .unwrap(),
                            bz: device
                                .upload(&factors[1].evals()[2 * start..2 * (start + retained)])
                                .unwrap(),
                            eq: DeviceSplitEq::new_window_with_scaling(
                                device,
                                &point,
                                BindingOrder::LowToHigh,
                                scale,
                                ordinal,
                                count,
                            )
                            .unwrap(),
                        }
                    })
                    .collect();
                let state = DeviceRemainder {
                    shards,
                    collapsed: None,
                    eq: DeviceSplitEq::new_with_scaling(
                        context,
                        &point,
                        BindingOrder::LowToHigh,
                        scale,
                    )
                    .unwrap(),
                    log_t: LOG_T,
                    local_rounds: LOG_T + 1 - count.ilog2() as usize,
                    challenges: Vec::new(),
                };
                let mut got = SpartanOuterRemainderKernel::new(state, 3, LOG_T);
                let mut eq = Polynomial::new(
                    EqPolynomial::new(point)
                        .evaluations()
                        .into_iter()
                        .map(|v| v * scale)
                        .collect(),
                );
                let mut claim: Fr = (0..2 * CYCLES)
                    .map(|i| eq.evals()[i] * factors[0].evals()[i] * factors[1].evals()[i])
                    .sum();
                let mut bind = None;
                for round in 0..=LOG_T {
                    let message = got.prove_round(bind, round, claim).unwrap();
                    for t in 0..=3 {
                        let t = Fr::from(t);
                        let expected: Fr = (0..eq.evals().len() / 2)
                            .map(|i| {
                                std::iter::once(&eq)
                                    .chain(&factors)
                                    .map(|p| {
                                        p.evals()[2 * i]
                                            + t * (p.evals()[2 * i + 1] - p.evals()[2 * i])
                                    })
                                    .product::<Fr>()
                            })
                            .sum();
                        assert_eq!(
                            message.evaluate(t),
                            expected,
                            "devices={count} prefix={prefix} case={case} round={round}"
                        );
                    }
                    let r = if case == 1 || case == 2 {
                        Fr::from(((round + case) % 2) as u64)
                    } else {
                        fr(round as u64 + 101)
                    };
                    claim = message.evaluate(r);
                    bind = Some(r);
                    eq.bind_with_order(r, BindingOrder::LowToHigh);
                    for factor in &mut factors {
                        factor.bind_with_order(r, BindingOrder::LowToHigh);
                    }
                }
                got.finish_rounds(bind.unwrap()).unwrap();
                let (az, bz) = got.state.collapsed.as_ref().unwrap();
                assert_eq!(az.first().unwrap(), factors[0].evals()[0]);
                assert_eq!(bz.first().unwrap(), factors[1].evals()[0]);
            }
        }
    }
}

#[test]
fn padded_outer_witness_matches_reference() {
    if shared_context().is_none() {
        return;
    }
    const LOG_T: usize = 6;
    let instruction = JoltInstructionRow {
        instruction_kind: Kind::XOR,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rs1: Some(2),
            rs2: Some(3),
            ..Default::default()
        },
        ..Default::default()
    };
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            vec![instruction],
            instruction.address as u64,
            RV64IMAC_JOLT,
        )
        .unwrap(),
        max_padded_trace_length: 1 << LOG_T,
        ram: RAMPreprocessing::default(),
        memory_layout: MemoryLayout::default(),
    });
    let program = Arc::new(JoltProgram::default());
    for prefix in [0, 1, 31, 32, 33, 63, 64] {
        let rows = (0..prefix)
            .map(|i| {
                synthetic_row(
                    instruction,
                    RegisterState {
                        rs1: Some(RegisterRead {
                            register: 2,
                            value: i as u64 + 15,
                        }),
                        rs2: Some(RegisterRead {
                            register: 3,
                            value: i as u64 * 71 + 1,
                        }),
                        ..Default::default()
                    },
                    RamAccess::NoOp,
                    1,
                )
            })
            .collect();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                LOG_T,
                64,
                JoltOneHotConfig {
                    log_k_chunk: 8,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
            ),
            JoltVmWitnessInputs::new(
                &program,
                &preprocessing,
                TraceOutput::new(TraceData::new(rows), Default::default(), None, None),
            ),
        );
        let tau: Vec<_> = (0..LOG_T + 2).map(|i| fr(i as u64 + 29)).collect();
        let relation =
            OuterRemainder::new(SpartanOuterDimensions::rv64(LOG_T), tau.clone(), fr(41));
        let claims = outer_remainder_input_values_from_uniskip_output(Fr::zero());
        let points = OuterRemainderInputClaims {
            outer_uniskip: Vec::new(),
        };
        let challenges = NoChallenges::default();
        let prepare = |cuda| -> Box<dyn SumcheckKernel<Fr, Relation = OuterRemainder<Fr>>> {
            let mut session = ProofSession::default();
            let backend: &dyn UniskipKernel<Fr, OuterRemainder<Fr>> = if cuda {
                &CudaBackend
            } else {
                &ReferenceBackend
            };
            backend
                .prepare(&mut session, LOG_T, &tau, &witness)
                .unwrap();
            let inputs = ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            if cuda {
                PrepareKernel::prepare(&CudaBackend, &mut session, &witness, inputs).unwrap()
            } else {
                ReferenceOuterRemainder
                    .prepare(&mut session, &witness, inputs)
                    .unwrap()
            }
        };
        let claim = probe_input_claim(&mut *prepare(false));
        let mut expected = prepare(false);
        let mut got = prepare(true);
        let bindings: Vec<_> = (0..=LOG_T).map(|i| fr(i as u64 + 67)).collect();
        assert_eq!(
            drive(&mut *got, claim, &bindings),
            drive(&mut *expected, claim, &bindings),
            "prefix={prefix}"
        );
        assert_eq!(
            got.output_claims(&claims).unwrap().opening_values(),
            expected.output_claims(&claims).unwrap().opening_values(),
            "prefix={prefix}"
        );
    }
}
