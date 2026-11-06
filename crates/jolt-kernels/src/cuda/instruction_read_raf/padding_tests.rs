#![expect(
    clippy::unwrap_used,
    reason = "test failures include device and fixture errors"
)]

use std::num::NonZeroUsize;
use std::sync::Arc;

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafChallenges, InstructionReadRafInputClaims,
};
use jolt_claims::protocols::jolt::JoltOneHotConfig;
use jolt_claims::OutputClaims;
use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{
    JoltInstructionKind as Kind, JoltInstructionRow, NormalizedOperands, RamAccess, RegisterRead,
    RegisterState, RV64IMAC_JOLT,
};
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use super::cycle_rounds::{CycleShard, DeviceCycleRounds};
use super::padding::eq_weights;
use super::{CudaBackend, ADDRESS_BITS};
use crate::cuda::common::context::{context_for, device_count, shared_context};
use crate::cuda::common::devices::CycleWindow;
use crate::cuda::common::testing::{drive, fr, reference_input_claim, synthetic_row};
use crate::{PrepareKernel, ProofSession, ProverInputs, ReferenceBackend};

#[test]
fn compact_weights_cover_original_cycle_windows() {
    let Some(context) = shared_context() else {
        return;
    };
    for point in [
        (1..=6).map(fr).collect::<Vec<_>>(),
        vec![Fr::from(0u64); 6],
        vec![Fr::from(1u64); 6],
    ] {
        let dense = EqPolynomial::new(point.clone()).evaluations();
        for window in [
            CycleWindow { start: 0, len: 64 },
            CycleWindow { start: 0, len: 32 },
            CycleWindow { start: 32, len: 32 },
        ] {
            for retained in [1, 2, 17, window.len] {
                let got = eq_weights(context, &point, &window, retained)
                    .unwrap()
                    .to_host()
                    .unwrap();
                assert_eq!(
                    &got[..retained - 1],
                    &dense[window.start..window.start + retained - 1]
                );
                assert_eq!(
                    got[retained - 1],
                    dense[window.start + retained - 1..window.end()]
                        .iter()
                        .copied()
                        .sum::<Fr>()
                );
            }
        }
    }
    assert!(eq_weights(context, &[fr(1)], &CycleWindow { start: 1, len: 2 }, 1).is_err());
    assert!(eq_weights(context, &[fr(1)], &CycleWindow { start: 0, len: 2 }, 0).is_err());
}

#[test]
fn constant_tail_cycle_rounds_match_dense_polynomials() {
    let Some(context) = shared_context() else {
        return;
    };
    const LOG_T: usize = 6;
    const CYCLES: usize = 1 << LOG_T;
    for shards in [1, 2, 4] {
        if shards > 1 && device_count() < 2 {
            continue;
        }
        let width = CYCLES / shards;
        for prefix in [0, 1, 15, 16, 17, 31, 32, 33, CYCLES - 1, CYCLES] {
            for exceptional in [false, true] {
                let point: Vec<_> = (0..LOG_T)
                    .map(|i| {
                        if exceptional {
                            Fr::from((i % 2) as u64)
                        } else {
                            fr(i as u64 + 1)
                        }
                    })
                    .collect();
                let mut factors: Vec<_> = (0..9)
                    .map(|lane| {
                        Polynomial::new(
                            (0..CYCLES)
                                .map(|i| fr((lane * CYCLES + i.min(prefix)) as u64 + 17))
                                .collect(),
                        )
                    })
                    .collect();
                let windows = (0..shards)
                    .map(|index| {
                        let start = index * width;
                        let ordinal = index % device_count().min(2);
                        let retained = (prefix.saturating_sub(start) + 1).min(width);
                        let device = context_for(ordinal).unwrap();
                        CycleShard {
                            ordinal,
                            factors: factors
                                .iter()
                                .map(|p| {
                                    device.upload(&p.evals()[start..start + retained]).unwrap()
                                })
                                .collect(),
                        }
                    })
                    .collect();
                let mut got = DeviceCycleRounds::from_windows(&point, windows, LOG_T).unwrap();
                let mut eq = Polynomial::new(EqPolynomial::new(point).evaluations());
                let mut claim: Fr = (0..CYCLES)
                    .map(|i| eq.evals()[i] * factors.iter().map(|p| p.evals()[i]).product::<Fr>())
                    .sum();
                for round in 0..LOG_T {
                    let message = got.round_message(context, claim).unwrap();
                    for t in 0..=10 {
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
                            "shards={shards} prefix={prefix} round={round}"
                        );
                    }
                    let r = if exceptional {
                        Fr::from((round % 2) as u64)
                    } else {
                        fr(round as u64 + 11)
                    };
                    claim = message.evaluate(r);
                    got.bind(context, r).unwrap();
                    eq.bind_with_order(r, BindingOrder::LowToHigh);
                    for p in &mut factors {
                        p.bind_with_order(r, BindingOrder::LowToHigh);
                    }
                }
                assert_eq!(
                    got.ra_finals::<Fr>(context).unwrap(),
                    factors
                        .iter()
                        .skip(1)
                        .map(|p| p.evals()[0])
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn padded_instruction_witness_matches_reference() {
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
        let relation = InstructionReadRaf::<Fr>::new(InstructionReadRafDimensions::new(
            LOG_T,
            ADDRESS_BITS,
            NonZeroUsize::new(8).unwrap(),
        ));
        let claims = InstructionReadRafInputClaims {
            lookup_output: Fr::from_u64(0),
            left_lookup_operand: Fr::from_u64(0),
            right_lookup_operand: Fr::from_u64(0),
        };
        let point: Vec<_> = (0..LOG_T).map(|i| fr(i as u64 + 1)).collect();
        let points = InstructionReadRafInputClaims {
            lookup_output: point.clone(),
            left_lookup_operand: point.clone(),
            right_lookup_operand: point,
        };
        let challenges = InstructionReadRafChallenges { gamma: fr(13) };
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let claim = reference_input_claim(&witness, inputs);
        let mut expected = ReferenceBackend
            .prepare(&mut ProofSession::default(), &witness, inputs())
            .unwrap();
        let mut got = CudaBackend
            .prepare(&mut ProofSession::default(), &witness, inputs())
            .unwrap();
        let bindings: Vec<_> = (0..ADDRESS_BITS + LOG_T)
            .map(|i| fr(i as u64 + 29))
            .collect();
        assert_eq!(
            drive(&mut *got, claim, &bindings),
            drive(&mut *expected, claim, &bindings),
            "prefix={prefix}"
        );
        assert_eq!(
            got.output_claims(&claims).unwrap().opening_values(),
            expected.output_claims(&claims).unwrap().opening_values()
        );
    }
}
