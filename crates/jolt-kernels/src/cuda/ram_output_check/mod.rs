use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::protocols::jolt::relations::ram::{
    RamOutputCheckChallenges, RamOutputCheckInputClaims, RamOutputCheckOutputClaims,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamOutputCheckPublic};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::CudaKernelContext;
use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use rounds::DeviceOutputCheck;

mod rounds;

pub struct RamOutputCheckKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: RamOutputCheck<F>,
    state: DeviceOutputCheck<F>,
    rounds_bound: usize,
}

impl<F: Field> RamOutputCheckKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.state.bind(self.context, challenge).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda RAM output-check bind",
            }
        })?;
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for RamOutputCheckKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        let mut coefficients = self
            .state
            .round_message(self.context, previous_claim)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda RAM output-check round",
            })?
            .into_coefficients();
        coefficients.resize(self.relation.degree() + 1, F::zero());
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for RamOutputCheckKernel<F> {
    type Relation = RamOutputCheck<F>;

    fn output_claims(
        &mut self,
        _inputs: &RamOutputCheckInputClaims<F>,
    ) -> Result<RamOutputCheckOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let [_io_mask, val_final, _val_io] =
            self.state
                .finals()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA RAM output-check table readback failed",
                })?;
        Ok(RamOutputCheckOutputClaims { val_final })
    }

    fn validate_derived_tables(
        &self,
        relation: &RamOutputCheck<F>,
        input_points: &RamOutputCheckInputClaims<Vec<F>>,
        output_points: &RamOutputCheckOutputClaims<Vec<F>>,
        challenges: &RamOutputCheckChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let [io_mask, _val_final, val_io] =
            self.state
                .finals()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA RAM output-check table readback failed",
                })?;
        for (id, got) in [
            (JoltDerivedId::from(RamOutputCheckPublic::IoMask), io_mask),
            (JoltDerivedId::from(RamOutputCheckPublic::ValIo), val_io),
        ] {
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

impl<F: Field> PrepareKernel<F, RamOutputCheck<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamOutputCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamOutputCheck<F>>>, KernelError<F>> {
        let context = require_context()?;
        let relation = inputs.relation;
        let address_point = inputs.challenges.output_address.as_slice();
        let ram_log_k = address_point.len();
        if relation.read_write_dimensions().output_check_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "the CUDA RAM output check supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }

        let addresses = 1usize << ram_log_k;
        let public_memory = relation.public_memory();
        let mut val_io = vec![F::zero(); addresses];
        for segment in &public_memory.segments {
            for (offset, &word) in segment.words.iter().enumerate() {
                let index = segment.start_index as usize + offset;
                if index < addresses {
                    val_io[index] = F::from_u64(word);
                }
            }
        }
        let io_mask: Vec<F> = (0..addresses)
            .map(|k| {
                let inside = (k as u128) >= public_memory.io_mask_start
                    && (k as u128) < public_memory.io_mask_end;
                if inside {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        let val_final = dense_view(witness, ram_val_final())?;
        if val_final.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", ram_val_final()),
                expected: addresses,
                got: val_final.len(),
            });
        }

        let state = DeviceOutputCheck::new(context, &io_mask, &val_final, &val_io, address_point)?;
        Ok(Box::new(RamOutputCheckKernel {
            context,
            relation: relation.clone(),
            state,
            rounds_bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{
        JoltProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, PublicIoMemory, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_verifier::stages::stage2::ram_output_check::{
        RamOutputCheck, RamOutputCheckChallenges, RamOutputCheckInputClaims,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, drive, reference_input_claim};
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 6;

    const RAM_LOG_KS: [usize; 2] = [4, 6];

    const IO_BYTES: u64 = 16;

    const WITNESS_OUTPUTS: [u8; 16] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x0F,
        0x1F,
    ];

    const PUBLIC_OUTPUTS: [u8; 16] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x0F,
        0x2F,
    ];

    fn memory_config() -> MemoryConfig {
        MemoryConfig {
            program_size: Some(1 << 12),
            max_input_size: IO_BYTES,
            max_output_size: IO_BYTES,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            ..MemoryConfig::default()
        }
    }

    fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    fn device(outputs: &[u8]) -> JoltDevice {
        let mut device = JoltDevice::new(&memory_config());
        device.inputs = (0..IO_BYTES as u8).map(|byte| byte * 3 + 1).collect();
        device.outputs = outputs.to_vec();
        device
    }

    fn final_memory() -> MemoryImage {
        MemoryImage {
            bytes: vec![(0, 0xA1), (8, 0xB2), (9, 0xC3), (24, 0xD4)],
        }
    }

    fn output_rows(instruction: JoltInstructionRow, lowest: u64, ram_k: usize) -> Vec<TraceRow> {
        let cycles = 1usize << LOG_T;
        (0..cycles)
            .map(|cycle| {
                let word = (cycle as u64 * 5 + 1) % ram_k as u64;
                let address = lowest + 8 * word;
                let value = 700 + cycle as u64;
                let access = match cycle % 3 {
                    0 => RamAccess::NoOp,
                    1 => RamAccess::Read(RamRead { address, value }),
                    _ => RamAccess::Write(RamWrite {
                        address,
                        pre_value: value,
                        post_value: value + 1,
                    }),
                };
                TraceRow {
                    instruction,
                    ram_access: access,
                    ..TraceRow::default()
                }
            })
            .collect()
    }

    fn with_output_check_witness<R>(
        ram_k: usize,
        body: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R,
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
        let witness_device = device(&WITNESS_OUTPUTS);
        let memory_layout = witness_device.memory_layout.clone();
        let lowest = memory_layout.get_lowest_address();
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![instruction],
                instruction.address as u64,
                RV64IMAC_JOLT,
            )
            .expect("output check bytecode fixture"),
            ram: RAMPreprocessing::default(),
            memory_layout,
            max_padded_trace_length: 1usize << LOG_T,
        };
        let program = JoltProgram::default();
        let trace = TraceOutput::new(
            OwnedTrace::new(output_rows(instruction, lowest, ram_k)),
            witness_device,
            Some(final_memory()),
        );
        let backend = TraceBackend::new(
            JoltVmWitnessConfig::new(LOG_T, ram_k, one_hot()),
            JoltVmWitnessInputs::new(&program, &preprocessing, trace),
        );
        body(&backend)
    }

    fn public_memory(perturbed: bool) -> PublicIoMemory {
        let outputs: &[u8] = if perturbed {
            &PUBLIC_OUTPUTS
        } else {
            &WITNESS_OUTPUTS
        };
        PublicIoMemory::new(&device(outputs)).expect("public IO memory")
    }

    fn io_tables(memory: &PublicIoMemory, addresses: usize) -> (Vec<bool>, Vec<Fr>) {
        let mut val_io = vec![Fr::from_u64(0); addresses];
        for segment in &memory.segments {
            for (offset, &word) in segment.words.iter().enumerate() {
                let index = segment.start_index as usize + offset;
                if index < addresses {
                    val_io[index] = Fr::from_u64(word);
                }
            }
        }
        let mask = (0..addresses)
            .map(|k| (k as u128) >= memory.io_mask_start && (k as u128) < memory.io_mask_end)
            .collect();
        (mask, val_io)
    }

    fn dimensions(ram_log_k: usize) -> ReadWriteDimensions {
        ReadWriteDimensions::new(LOG_T, ram_log_k, LOG_T, ram_log_k)
    }

    #[test]
    fn fixture_output_check_mask_and_values_discriminate() {
        for ram_log_k in RAM_LOG_KS {
            let addresses = 1usize << ram_log_k;
            with_output_check_witness(addresses, |witness| {
                let val_final = dense_view::<Fr>(witness, ram_val_final())
                    .expect("the fixture serves val_final");
                assert_eq!(val_final.len(), addresses);

                let (mask, faithful_io) = io_tables(&public_memory(false), addresses);
                let (perturbed_mask, perturbed_io) = io_tables(&public_memory(true), addresses);
                assert_eq!(
                    mask, perturbed_mask,
                    "the perturbation must change only the IO words, not the mask",
                );

                let inside = mask.iter().filter(|flag| **flag).count();
                assert!(
                    inside > 0 && inside < addresses,
                    "ram_log_k {ram_log_k}: {inside} of {addresses} addresses are inside the IO \
                     mask, so an all-zero or all-one mask would pass",
                );
                assert!(
                    mask.iter()
                        .zip(&faithful_io)
                        .any(|(flag, value)| *flag && *value != Fr::from_u64(0)),
                    "ram_log_k {ram_log_k}: every masked ValIo word is zero, so the ValIo table \
                     could be dropped",
                );
                assert!(
                    mask.iter()
                        .zip(&val_final)
                        .any(|(flag, value)| *flag && *value != Fr::from_u64(0)),
                    "ram_log_k {ram_log_k}: every masked val_final word is zero, so the opening \
                     table contributes nothing inside the mask",
                );
                assert!(
                    mask.iter()
                        .zip(&val_final)
                        .zip(&faithful_io)
                        .all(|((flag, final_value), io_value)| !*flag || final_value == io_value),
                    "ram_log_k {ram_log_k}: the UNPERTURBED fixture already violates the output \
                     check, so the perturbation is not what makes the test discriminating",
                );
                assert!(
                    mask.iter()
                        .zip(&val_final)
                        .zip(&perturbed_io)
                        .any(|((flag, final_value), io_value)| *flag && final_value != io_value),
                    "ram_log_k {ram_log_k}: the perturbed public memory still agrees with \
                     val_final everywhere in the mask, so every round polynomial is zero and a \
                     kernel returning zeros would pass",
                );
                assert!(
                    mask.iter()
                        .zip(&val_final)
                        .any(|(flag, value)| !*flag && *value != Fr::from_u64(0)),
                    "ram_log_k {ram_log_k}: val_final is zero everywhere outside the mask, so a \
                     kernel that dropped the IoMask factor would still pass",
                );
            });
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn ram_output_check_matches_reference_round_for_round(
            perturbed in any::<bool>(),
            output_address in arb_point(RAM_LOG_KS[1]),
            challenges in arb_point(RAM_LOG_KS[1]),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            for ram_log_k in RAM_LOG_KS {
                let addresses = 1usize << ram_log_k;
                with_output_check_witness(addresses, |witness| {
                    let relation = RamOutputCheck::<Fr>::new(
                        dimensions(ram_log_k),
                        public_memory(perturbed),
                    );
                    let claims = RamOutputCheckInputClaims::default();
                    let points = RamOutputCheckInputClaims::default();
                    let challenge_set = RamOutputCheckChallenges {
                        output_address: output_address[..ram_log_k].to_vec(),
                    };
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
                        "round polynomials diverged at ram_log_k {} perturbed {}",
                        ram_log_k,
                        perturbed
                    );

                    let expected_claims =
                        expected_kernel.output_claims(&claims).expect("reference claims");
                    let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                    prop_assert_eq!(
                        got_claims.opening_values(),
                        expected_claims.opening_values(),
                        "output claims diverged at ram_log_k {} perturbed {}",
                        ram_log_k,
                        perturbed
                    );
                    Ok(())
                })?;
            }
        }
    }
}
