use jolt_claims::protocols::jolt::relations::ram::{
    RamRafEvaluationInputClaims, RamRafEvaluationOutputClaims,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::dense_product::{DenseProductKernel, DeviceDenseProduct};
use crate::cuda::common::one_hot_fold::{affine_table, DeviceOneHotColumns, FoldTuning};
use std::sync::Arc;

use crate::cuda::common::device_columns::{device_ram_words, DeviceTraceColumns};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const RAM_WORD_BYTES: u64 = 8;

impl<F: Field> PrepareKernel<F, RamRafEvaluation<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRafEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRafEvaluation<F>>>, KernelError<F>> {
        let context = require_context()?;
        let relation = inputs.relation;
        let ram_log_k = relation.ram_log_k();
        if relation.read_write_dimensions().raf_evaluation_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "the CUDA RAM RAF evaluation supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }
        if relation.tau_low().is_empty() {
            return Err(KernelError::InvariantViolation {
                reason: "the RAM RAF evaluation cycle point has no variables",
            });
        }

        let cycles = 1usize << relation.tau_low().len();
        let words = device_ram_words::<F>(context, session, witness, cycles, 1usize << ram_log_k)?;
        let columns = DeviceOneHotColumns::from_device(
            DeviceTraceColumns {
                lookup: Arc::new(context.alloc_u64(0)?),
                pc: Arc::new(context.alloc_u32(0)?),
                ram: words,
            },
            [0, 0, 1],
            ram_log_k,
            cycles,
        )?;

        let folded = columns.fold_cycles(context, relation.tau_low(), FoldTuning::default())?;
        drop(columns);
        let unmap = affine_table(
            context,
            relation.lowest_address(),
            RAM_WORD_BYTES,
            1usize << ram_log_k,
        )?;

        let state = DeviceDenseProduct::from_device_factors(
            None,
            vec![folded, unmap],
            None,
            None,
            ram_log_k,
            relation.degree(),
        )?;
        Ok(Box::new(DenseProductKernel {
            state,
            relation: relation.clone(),
            context,
            field: core::marker::PhantomData,
        }))
    }
}

impl<F: Field> SumcheckKernel<F> for DenseProductKernel<F, RamRafEvaluation<F>> {
    type Relation = RamRafEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &RamRafEvaluationInputClaims<F>,
    ) -> Result<RamRafEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.state.rounds_bound();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals: Vec<F> =
            self.finals()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA RAM RAF evaluation factor readback failed",
                })?;
        let [ram_ra, _unmap] = finals.as_slice() else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "the RAM RAF evaluation expects exactly two bound factors",
            });
        };
        Ok(RamRafEvaluationOutputClaims { ram_ra: *ram_ra })
    }

    fn validate_derived_tables(
        &self,
        relation: &RamRafEvaluation<F>,
        input_points: &RamRafEvaluationInputClaims<Vec<F>>,
        output_points: &RamRafEvaluationOutputClaims<Vec<F>>,
        challenges: &NoChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let id = JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let finals: Vec<F> =
            self.finals()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA RAM RAF evaluation factor readback failed",
                })?;
        let [_ram_ra, unmap] = finals.as_slice() else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "the RAM RAF evaluation expects exactly two bound factors",
            });
        };
        if *unmap != expected {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id,
                expected,
                got: *unmap,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use common::jolt_device::{MemoryConfig, MemoryLayout};
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{
        JoltProgram, OwnedTrace, RamAccess, RamRead, RamWrite, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_verifier::stages::stage2::ram_raf_evaluation::{
        RamRafEvaluation, RamRafEvaluationInputClaims,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, OneHotSource, TraceBackend};
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, drive, reference_input_claim};
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 7;

    const RAM_LOG_KS: [usize; 2] = [4, 6];

    const RAF_PATTERN: usize = 5;

    const RAF_REPEATED_WORD: u64 = 3;

    fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    const fn raf_fixture_is_cold(cycle: usize) -> bool {
        matches!(cycle % RAF_PATTERN, 0 | 1)
    }

    fn raf_rows(
        instruction: JoltInstructionRow,
        layout: &MemoryLayout,
        ram_k: usize,
        seed: u64,
    ) -> Vec<TraceRow> {
        let cycles = 1usize << LOG_T;
        let lowest = layout.get_lowest_address();
        let mut rows = Vec::with_capacity(cycles);

        for cycle in 0..cycles {
            let mix = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(cycle as u64 + 1);
            let word = if cycle % 4 == 3 {
                RAF_REPEATED_WORD % ram_k as u64
            } else {
                (mix.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (mix >> 29)) % ram_k as u64
            };
            let address = lowest + 8 * word;
            let value = 900 + cycle as u64;

            let access = if raf_fixture_is_cold(cycle) {
                match cycle % RAF_PATTERN {
                    0 => RamAccess::NoOp,
                    _ => RamAccess::Read(RamRead { address: 0, value }),
                }
            } else if cycle % RAF_PATTERN == 2 {
                RamAccess::Write(RamWrite {
                    address,
                    pre_value: value,
                    post_value: value + 1,
                })
            } else {
                RamAccess::Read(RamRead { address, value })
            };

            rows.push(TraceRow {
                instruction,
                ram_access: access,
                ..TraceRow::default()
            });
        }
        rows
    }

    fn with_raf_witness<R>(
        ram_k: usize,
        seed: u64,
        body: impl FnOnce(&TraceBackend<OwnedTrace>, u64) -> R,
    ) -> R {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::XOR,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: Some(3),
                imm: 0,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let memory_layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1 << 12),
            ..MemoryConfig::default()
        });
        let lowest_address = memory_layout.get_lowest_address();
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![instruction],
                instruction.address as u64,
                RV64IMAC_JOLT,
            )
            .expect("raf bytecode fixture"),
            ram: RAMPreprocessing::default(),
            memory_layout: memory_layout.clone(),
            max_padded_trace_length: 1usize << LOG_T,
        });
        let program = Arc::new(JoltProgram::default());
        let trace = TraceOutput::new(
            OwnedTrace::new(raf_rows(instruction, &memory_layout, ram_k, seed)),
            Default::default(),
            None,
            None,
        );
        let backend = TraceBackend::new(
            JoltVmWitnessConfig::new(LOG_T, ram_k, one_hot()),
            JoltVmWitnessInputs::new(&program, &preprocessing, trace),
        );
        body(&backend, lowest_address)
    }

    fn dimensions(ram_log_k: usize) -> ReadWriteDimensions {
        ReadWriteDimensions::new(LOG_T, ram_log_k, LOG_T, ram_log_k)
    }

    #[test]
    fn fixture_raf_fold_accumulates_over_cold_and_hot_cycles() {
        let cycles = 1usize << LOG_T;
        for ram_log_k in RAM_LOG_KS {
            let ram_k = 1usize << ram_log_k;
            with_raf_witness(ram_k, 7, |witness, _| {
                let hot = OneHotSource::hot_indices(
                    witness,
                    JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa),
                )
                .expect("the fixture serves the RAM one-hot indices");
                assert_eq!(hot.len(), cycles);

                let mut counts = vec![0usize; ram_k];
                for address in hot.iter().flatten() {
                    counts[*address] += 1;
                }
                let occupied = counts.iter().filter(|count| **count > 0).count();
                assert!(
                    occupied > 1,
                    "ram_log_k {ram_log_k}: the fold lands on {occupied} address(es), so the \
                     folded table cannot detect a wrong address",
                );
                assert!(
                    counts.iter().any(|count| *count > 1),
                    "ram_log_k {ram_log_k}: no address is hot at two cycles, so a fold that \
                     OVERWRITES instead of accumulating would still pass",
                );

                let cold = hot.iter().filter(|address| address.is_none()).count();
                assert!(
                    cold > 0 && cold < cycles,
                    "ram_log_k {ram_log_k}: {cold} of {cycles} cycles are cold, so one of the two \
                     fold paths is unexercised",
                );
                for (cycle, address) in hot.iter().enumerate() {
                    assert_eq!(
                        address.is_none(),
                        raf_fixture_is_cold(cycle),
                        "ram_log_k {ram_log_k} cycle {cycle}: the RAM one-hot column disagrees \
                         with the fixture on whether the cycle is cold",
                    );
                }

                let distinct: BTreeSet<usize> = hot.iter().flatten().copied().collect();
                assert!(
                    distinct.len() > 1,
                    "ram_log_k {ram_log_k}: the hot address is constant across every hot cycle",
                );
            });
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn ram_raf_evaluation_matches_reference_round_for_round(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            challenges in arb_point(RAM_LOG_KS[1]),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            for ram_log_k in RAM_LOG_KS {
                let read_write = dimensions(ram_log_k);
                let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write)
                    .expect("RAM RAF evaluation dimensions");

                with_raf_witness(1usize << ram_log_k, seed, |witness, lowest_address| {
                    let relation = RamRafEvaluation::<Fr>::new(
                        read_write,
                        raf_dimensions,
                        ram_log_k,
                        lowest_address,
                        tau_low.clone(),
                    );
                    let claims = RamRafEvaluationInputClaims {
                        ram_address: Fr::from_u64(0),
                    };
                    let points = RamRafEvaluationInputClaims {
                        ram_address: Vec::new(),
                    };
                    let challenge_set = NoChallenges::<Fr>::default();
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
                        .expect("cuda prepare");

                    let expected =
                        drive(&mut *expected_kernel, input_claim, &challenges[..ram_log_k]);
                    let got = drive(&mut *got_kernel, input_claim, &challenges[..ram_log_k]);
                    prop_assert_eq!(
                        got,
                        expected,
                        "round polynomials diverged at ram_log_k {}",
                        ram_log_k
                    );

                    let expected_claims =
                        expected_kernel.output_claims(&claims).expect("reference claims");
                    let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                    prop_assert_eq!(
                        got_claims.opening_values(),
                        expected_claims.opening_values(),
                        "output claims diverged at ram_log_k {}",
                        ram_log_k
                    );
                    Ok(())
                })?;
            }
        }
    }
}
