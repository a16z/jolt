//! Twin-transcript engine locks: toy members driven through
//! `jolt_sumcheck::prove_batch` (and `prove_uniskip_clear`) against the
//! GENERATED `verify_clear` / `verify_zk` drivers and the shared uni-skip
//! `verify_clear` core, asserting byte-identical transcript states. This pins
//! the prove-side engine to the verifier independently of any real stage.
//! The toy locks run under both Dory and Akita. Clear RV64 stage-1 twins
//! also check repeatable exits and rejected prefixes on trace-backed witnesses.

#![expect(clippy::unwrap_used, reason = "test crate")]

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fr, JoltField, Ring};
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, CenteredIntegerDomain, ClearRound, ClearSumcheckRecorder,
    CommittedSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckDomain, SumcheckError,
    SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{prover_transcript, Keccak, Transcript};
use jolt_verifier::stages::relations::{ConcreteSumcheck as _, SumcheckBatch};
use jolt_verifier::stages::stage5::{InstructionReadRaf, RegistersValEvaluation};
use jolt_verifier::stages::uniskip::{self, UniskipParams};

#[derive(SumcheckBatch)]
struct TwinFixtureSumchecks<F: JoltField> {
    instruction_read_raf: InstructionReadRaf<F>,
    registers_val_evaluation: RegistersValEvaluation<F>,
}

/// Small geometry so a dense toy prover is feasible: the instruction
/// member gets 8 rounds (6 address + 2 cycle), the registers member 3 —
/// so the generated-driver twins also exercise front-loaded padding.
fn fixture() -> TwinFixtureSumchecks<Fr> {
    TwinFixtureSumchecks {
        instruction_read_raf: InstructionReadRaf::new(
            InstructionReadRafDimensions::try_from((2, 6, 2)).unwrap(),
        ),
        registers_val_evaluation: RegistersValEvaluation::new(TraceDimensions::new(3)),
    }
}

fn inputs() -> TwinFixtureInputClaims<Fr> {
    let fr = Fr::from_u64;
    TwinFixtureInputClaims {
        instruction_read_raf: InstructionReadRafInputClaims {
            lookup_output: fr(2),
            left_lookup_operand: fr(3),
            right_lookup_operand: fr(5),
        },
        registers_val_evaluation: RegistersValEvaluationInputClaims {
            registers_val: fr(7),
        },
    }
}

/// A dense multilinear toy batch member with a prescribed total sum
/// (HighToLow binding) — degree 1, which every relation's degree bound
/// admits.
struct DenseMember {
    evals: Vec<Fr>,
    num_rounds: usize,
}

impl DenseMember {
    fn with_sum(num_rounds: usize, sum: Fr, seed: u64) -> Self {
        let size = 1u64 << num_rounds;
        let mut evals: Vec<Fr> = (0..size)
            .map(|i| Fr::from_u64(seed + 31 * i + 11))
            .collect();
        let current: Fr = evals.iter().copied().sum();
        evals[0] += sum - current;
        Self { evals, num_rounds }
    }

    fn bind(&mut self, challenge: Fr) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
        }
        self.evals.truncate(half);
    }
}

impl ProveRounds<Fr> for DenseMember {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.evals.len() / 2;
        let eval_0: Fr = self.evals[..half].iter().copied().sum();
        let eval_1: Fr = self.evals[half..].iter().copied().sum();
        assert_eq!(eval_0 + eval_1, previous_claim);
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

fn pedersen_setup(capacity: u64) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let generators = (2..2 + capacity)
        .map(|k| generator.scalar_mul(&Fr::from_u64(k)))
        .collect();
    PedersenSetup::new(generators, generator.scalar_mul(&Fr::from_u64(99)))
}

fn synthetic_output_values() -> Vec<Fr> {
    vec![Fr::from_u64(11), Fr::from_u64(22), Fr::from_u64(33)]
}

#[test]
fn clear_engine_twin_matches_generated_verify_clear() {
    let sumchecks = fixture();
    let inputs = inputs();
    let mut prover_state =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"engine-twin"), b"");
    let prover_challenges = sumchecks.draw_challenges(&mut prover_state).unwrap();

    let instruction_sum = sumchecks
        .instruction_read_raf
        .input_claim(
            &inputs.instruction_read_raf,
            &prover_challenges.instruction_read_raf,
        )
        .unwrap();
    let registers_sum = sumchecks
        .registers_val_evaluation
        .input_claim(
            &inputs.registers_val_evaluation,
            &prover_challenges.registers_val_evaluation,
        )
        .unwrap();
    let mut instruction_member =
        DenseMember::with_sum(sumchecks.instruction_read_raf.rounds(), instruction_sum, 5);
    let mut registers_member = DenseMember::with_sum(
        sumchecks.registers_val_evaluation.rounds(),
        registers_sum,
        91,
    );

    let mut recorder = ClearSumcheckRecorder::<Fr>::new();
    let (batch, prover_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &prover_challenges,
            &mut recorder,
            &mut prover_state,
        )
        .unwrap();
    let mut members: Vec<&mut dyn ProveRounds<Fr>> =
        vec![&mut instruction_member, &mut registers_member];
    let proved = prove_batch(
        &batch,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    let output_values = synthetic_output_values();
    let recorded = recorder.finish(&output_values, &mut prover_state).unwrap();

    // Verifier: draw → begin_batch → verify_compressed_boolean → output-claim
    // absorbs (the low-level clear path the composed `verify_clear` wraps).
    let mut verifier_transcript =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"engine-twin"), b"");
    let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
    let mut verifier_recorder = ClearSumcheckRecorder::<Fr>::new();
    let (verifier_batch, verifier_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &verifier_challenges,
            &mut verifier_recorder,
            &mut verifier_transcript,
        )
        .unwrap();
    let reduction = SumcheckProof::<Fr, Bn254G1>::from(recorded.proof)
        .verify_compressed_boolean(
            verifier_batch.max_num_vars,
            verifier_batch.max_degree,
            verifier_batch.claimed_sum,
            &mut verifier_transcript,
        )
        .unwrap();
    for value in &output_values {
        verifier_transcript.append(value);
    }

    assert_eq!(reduction.value, proved.final_claim);
    assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
    assert_eq!(verifier_coefficients, prover_coefficients);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn committed_engine_twin_matches_generated_verify_zk() {
    type VC = Pedersen<Bn254G1>;
    let setup = pedersen_setup(8);

    // Prover: draw → sums → begin_batch(committed; claim absorbs no-op) →
    // prove_batch → finish (output-claim row commitments absorbed).
    let sumchecks = fixture();
    let inputs = inputs();
    let mut prover_state =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"engine-zk-twin"), b"");
    let prover_challenges = sumchecks.draw_challenges(&mut prover_state).unwrap();

    let instruction_sum = sumchecks
        .instruction_read_raf
        .input_claim(
            &inputs.instruction_read_raf,
            &prover_challenges.instruction_read_raf,
        )
        .unwrap();
    let registers_sum = sumchecks
        .registers_val_evaluation
        .input_claim(
            &inputs.registers_val_evaluation,
            &prover_challenges.registers_val_evaluation,
        )
        .unwrap();
    let mut instruction_member =
        DenseMember::with_sum(sumchecks.instruction_read_raf.rounds(), instruction_sum, 23);
    let mut registers_member = DenseMember::with_sum(
        sumchecks.registers_val_evaluation.rounds(),
        registers_sum,
        57,
    );

    let mut recorder =
        CommittedSumcheckRecorder::<Fr, VC, _>::new(&setup, rand_core::OsRng).unwrap();
    let (batch, prover_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &prover_challenges,
            &mut recorder,
            &mut prover_state,
        )
        .unwrap();
    let mut members: Vec<&mut dyn ProveRounds<Fr>> =
        vec![&mut instruction_member, &mut registers_member];
    let proved = prove_batch(
        &batch,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    let recorded = recorder
        .finish(&synthetic_output_values(), &mut prover_state)
        .unwrap();
    assert!(recorded.committed_witness.is_some());

    let mut verifier_transcript =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"engine-zk-twin"), b"");
    let _verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
    let consistency = sumchecks
        .verify_zk(&recorded.proof, &mut verifier_transcript)
        .unwrap();

    assert_eq!(consistency.challenges(), proved.challenges);
    assert_eq!(
        consistency.batching_coefficients,
        vec![
            prover_coefficients.instruction_read_raf,
            prover_coefficients.registers_val_evaluation,
        ],
    );
    assert_eq!(consistency.max_num_vars, batch.max_num_vars);
    assert_eq!(consistency.max_degree, batch.max_degree);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// Twin-transcript lock for the shared uni-skip verification core: a clear
/// uni-skip round proved through `jolt_sumcheck::prove_uniskip_clear` must be
/// accepted by `jolt-verifier`'s `uniskip::verify_clear` with byte-identical
/// transcript states (round proof, output-claim absorb, reduction challenge).
#[test]
fn uniskip_prover_twin_matches_uniskip_verify_clear() {
    let params = UniskipParams::spartan_outer();
    let degree = params.degree();
    let domain_size = params.domain_size();
    let poly = UnivariatePoly::new(
        (0..=degree as u64)
            .map(|k| Fr::from_u64(3 * k + 2))
            .collect(),
    );
    let coefficients = CenteredIntegerDomain::new(domain_size)
        .round_sum_coefficients(UnivariatePolynomial::degree(&poly))
        .unwrap();
    let input_claim = <UnivariatePoly<Fr> as ClearRound<Fr>>::coefficient_linear_combination(
        &poly,
        &coefficients,
    );

    let mut prover_state =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"uniskip-stage-twin"), b"");
    let proved =
        prove_uniskip_clear::<Fr, _>(poly, input_claim, degree, domain_size, &mut prover_state)
            .unwrap();

    let mut verifier_transcript =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"uniskip-stage-twin"), b"");
    let challenge = uniskip::verify_clear(
        &SumcheckProof::<Fr, Bn254G1>::from(proved.proof),
        &params,
        input_claim,
        proved.output_claim,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(challenge, proved.challenge);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

#[cfg(not(any(feature = "zk", feature = "akita", feature = "field-inline")))]
mod stopping {
    use std::sync::Arc;

    use super::*;
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_dory::DoryScheme;
    use jolt_field::Field as _;
    use jolt_kernels::{ProverInputs, SumcheckKernel};
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::expand::{expand_instruction, ExpansionAllocator};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_prover::stages::stage1::{prove_stage1, Stage1ProverOutput};
    use jolt_prover::{JoltBackend, ProofMode, ProverConfig, ProverError};
    use jolt_riscv::{
        JoltInstructionKind as Kind, JoltInstructionRow, JoltTraceRow, NormalizedOperands,
        RamAccess, RegisterRead, RegisterState, RegisterWrite, SourceInstruction,
        SourceInstructionKind, SourceInstructionRow, RV64IMAC_JOLT,
    };
    use jolt_transcript::ProverState;
    use jolt_verifier::config::MIN_PADDED_TRACE_LENGTH;
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainder,
    };
    use jolt_verifier::stages::stage1::outputs::{
        Stage1BatchInputClaims, Stage1BatchOutputClaims, Stage1BatchSumchecks, Stage1ClearOutput,
        Stage1OutputClaims,
    };
    use jolt_verifier::stages::uniskip::draw_spartan_outer_tau;
    use jolt_verifier::VerifierError;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    const ENTRY: u64 = 0x8000_0000;
    type Output = Stage1ProverOutput<Fr, Bn254G1>;

    #[derive(Clone, Copy, Debug)]
    enum End {
        Jal,
        Jalr,
        StableOverwritingJalr,
        CompressedJump,
        Ebreak,
        ArbitraryJump,
        UnstableJalr,
        CompressedJalr,
        ExpansionMidpoint,
        GeneratedTerminalJump,
    }

    // These rows bypass execution deliberately: the rejected cases are prefixes
    // a malicious prover can supply even though the tracer would continue.
    fn witness(real_rows: usize, end: End) -> TraceBackend {
        let mut bytecode = Vec::new();
        let mut observations = Vec::new();
        for offset in 0..real_rows - 1 {
            bytecode.push(JoltInstructionRow {
                instruction_kind: Kind::ADDI,
                address: ENTRY as usize + 4 * offset,
                operands: NormalizedOperands {
                    rd: Some(0),
                    rs1: Some(0),
                    ..Default::default()
                },
                ..Default::default()
            });
            observations.push(RegisterState::default());
        }
        let pc = ENTRY + 4 * (real_rows as u64 - 1);
        let indirect = matches!(
            end,
            End::Jalr | End::StableOverwritingJalr | End::UnstableJalr | End::CompressedJalr
        );
        let compressed = matches!(end, End::CompressedJump | End::CompressedJalr);
        let rd = if matches!(end, End::CompressedJalr) {
            1
        } else if matches!(end, End::CompressedJump) {
            0
        } else {
            5
        };
        let rs1 = if matches!(end, End::Jalr) { 6 } else { rd };
        let generated = matches!(
            end,
            End::Ebreak | End::ExpansionMidpoint | End::GeneratedTerminalJump
        );
        let last = JoltInstructionRow {
            instruction_kind: if indirect { Kind::JALR } else { Kind::JAL },
            address: pc as usize,
            operands: NormalizedOperands {
                rd: Some(rd),
                rs1: indirect.then_some(rs1),
                imm: if matches!(end, End::ArbitraryJump) {
                    4
                } else if matches!(end, End::StableOverwritingJalr) {
                    -4
                } else {
                    0
                },
                ..Default::default()
            },
            is_compressed: compressed,
            virtual_sequence_remaining: generated
                .then_some(u16::from(matches!(end, End::ExpansionMidpoint))),
            is_first_in_sequence: generated,
            source_can_end: matches!(end, End::Ebreak),
        };
        let last = if matches!(end, End::Ebreak) {
            let source = SourceInstruction::new(
                SourceInstructionKind::EBREAK,
                SourceInstructionRow {
                    address: pc as usize,
                    ..Default::default()
                },
            );
            let expansion =
                expand_instruction(&source, &mut ExpansionAllocator::new(), RV64IMAC_JOLT).unwrap();
            assert_eq!(expansion.len(), 1);
            JoltInstructionRow::from(expansion[0])
        } else {
            last
        };
        bytecode.push(last);
        observations.push(RegisterState {
            rs1: indirect.then_some(RegisterRead {
                register: rs1,
                value: pc
                    + if matches!(end, End::StableOverwritingJalr) {
                        4
                    } else {
                        0
                    },
            }),
            rd: last.operands.rd.map(|register| RegisterWrite {
                register,
                pre_value: 0,
                post_value: pc + if compressed { 2 } else { 4 },
            }),
            ..Default::default()
        });
        if matches!(end, End::ExpansionMidpoint) {
            bytecode.push(JoltInstructionRow {
                virtual_sequence_remaining: Some(0),
                is_first_in_sequence: false,
                source_can_end: false,
                ..last
            });
        }
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(bytecode.clone(), ENTRY, RV64IMAC_JOLT)
                .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: MIN_PADDED_TRACE_LENGTH,
        });
        let rows = bytecode
            .iter()
            .zip(observations)
            .map(|(&instruction, registers)| {
                JoltTraceRow::new(
                    instruction,
                    registers,
                    RamAccess::NoOp,
                    u32::try_from(preprocessing.bytecode.get_pc(&instruction).unwrap()).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let program = Arc::new(JoltProgram::from_parts(
            Vec::new(),
            bytecode,
            Vec::new(),
            pc + 4,
            ENTRY,
        ));
        let domain = ProverConfig::padded_trace_length::<Fr>(real_rows).unwrap();
        assert_eq!(domain, preprocessing.max_padded_trace_length);
        TraceBackend::new(
            JoltVmWitnessConfig::new(
                domain.ilog2() as usize,
                64,
                JoltOneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
            ),
            JoltVmWitnessInputs::new(
                &program,
                &preprocessing,
                TraceOutput::new(TraceData::new(rows), Default::default(), None, None),
            ),
        )
    }

    fn prove(
        witness: &TraceBackend,
        optimized: bool,
    ) -> Result<(Output, [u8; 32]), ProverError<Fr>> {
        let backend = if optimized {
            JoltBackend::optimized()
        } else {
            JoltBackend::reference()
        };
        let mut context = backend.with_witness(witness);
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let mut transcript =
            prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"stopping-twin"), b"");
        let output = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &mode,
            MIN_PADDED_TRACE_LENGTH.ilog2() as usize,
            &mut transcript,
        )?;
        Ok((output, transcript.verifier_message::<[u8; 32]>()))
    }

    struct ForgedEndpoints {
        inner: Box<dyn SumcheckKernel<Fr, Relation = OuterRemainder<Fr>>>,
    }

    impl ProveRounds<Fr> for ForgedEndpoints {
        fn num_rounds(&self) -> usize {
            self.inner.num_rounds()
        }

        fn prove_round(
            &mut self,
            bind: Option<Fr>,
            round: usize,
            previous_claim: Fr,
        ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
            match self.inner.prove_round(bind, round, previous_claim) {
                // The reference kernel binds before checking the endpoints.
                // Retry without binding and shift only the constant to forge
                // their sum while retaining the actual witness polynomial.
                Err(SumcheckError::RoundCheckFailed { actual, .. }) => {
                    let mut coefficients = self
                        .inner
                        .prove_round(None, round, actual)?
                        .into_coefficients();
                    let correction = (previous_claim - actual) * Fr::from_u64(2).inverse().unwrap();
                    if let Some(constant) = coefficients.first_mut() {
                        *constant += correction;
                    } else {
                        coefficients.push(correction);
                    }
                    Ok(UnivariatePoly::new(coefficients))
                }
                result => result,
            }
        }

        fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
            self.inner.finish_rounds(bind)
        }
    }

    // A malicious prover can forge endpoint sums and omit the final-claim
    // self-check. The remaining computation uses the reference kernel's real
    // witness polynomials and genuine extracted output claims.
    fn prove_with_forged_endpoints(witness: &TraceBackend) -> Result<Output, ProverError<Fr>> {
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut context = backend.with_witness(witness);
        let log_t = MIN_PADDED_TRACE_LENGTH.ilog2() as usize;
        let mut transcript =
            prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"stopping-twin"), b"");
        let tau = draw_spartan_outer_tau(&mut transcript, log_t)?;
        context.prepare_outer_uniskip(log_t, &tau)?;
        let params = UniskipParams::spartan_outer();
        let uniskip = prove_uniskip_clear::<Fr, _>(
            context.outer_uniskip_poly()?,
            Fr::from_u64(0),
            params.degree(),
            params.domain_size(),
            &mut transcript,
        )?;
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(log_t),
                tau,
                uniskip.challenge,
            ),
        };
        let challenges = sumchecks.draw_challenges(&mut transcript)?;
        let input_points = sumchecks.empty_input_points();
        let inputs = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(uniskip.output_claim),
        };
        let mut recorder = ClearSumcheckRecorder::<Fr>::new();
        let (batch, _) =
            sumchecks.begin_batch(&inputs, &challenges, &mut recorder, &mut transcript)?;
        let mut dishonest = ForgedEndpoints {
            inner: context.prepare(ProverInputs {
                relation: &sumchecks.outer_remainder,
                claims: &inputs.outer_remainder,
                points: &input_points.outer_remainder,
                challenges: &challenges.outer_remainder,
            })?,
        };
        let mut members: Vec<&mut dyn ProveRounds<Fr>> = vec![&mut dishonest];
        let proved = prove_batch(
            &batch,
            &mut members,
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )?;
        let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
        dishonest.inner.validate_derived_tables(
            &sumchecks.outer_remainder,
            &input_points.outer_remainder,
            &output_points.outer_remainder,
            &challenges.outer_remainder,
        )?;
        let outer = Stage1BatchOutputClaims {
            outer_remainder: dishonest.inner.output_claims(&inputs.outer_remainder)?,
        };
        sumchecks.validate_output_claims(&outer)?;
        let recorded = recorder.finish(&sumchecks.opening_values(&outer), &mut transcript)?;
        Ok(Output {
            tau_pow_nonce: None,
            pow_nonce: None,
            uniskip_proof: uniskip.proof.into(),
            sumcheck_proof: recorded.proof.into(),
            claims: Stage1OutputClaims::new(uniskip.output_claim, outer.clone()),
            clear_output: Stage1ClearOutput::new(outer, output_points),
        })
    }

    fn replay(output: &Output) -> Result<[u8; 32], VerifierError> {
        let log_t = MIN_PADDED_TRACE_LENGTH.ilog2() as usize;
        let mut transcript =
            prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"stopping-twin"), b"");
        let tau = draw_spartan_outer_tau(&mut transcript, log_t)?;
        let challenge = uniskip::verify_clear(
            &output.uniskip_proof,
            &UniskipParams::spartan_outer(),
            Fr::from_u64(0),
            output.claims.uniskip_output_claim,
            &mut transcript,
        )?;
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(log_t),
                tau,
                challenge,
            ),
        };
        let challenges = sumchecks.draw_challenges(&mut transcript)?;
        let inputs = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(
                output.claims.uniskip_output_claim,
            ),
        };
        let _ = sumchecks.verify_clear(
            &inputs,
            &sumchecks.empty_input_points(),
            &challenges,
            &output.claims.outer,
            &output.sumcheck_proof,
            &mut transcript,
            1,
        )?;
        sumchecks.append_output_claims(&mut transcript, &output.claims.outer);
        Ok(transcript.verifier_message::<[u8; 32]>())
    }

    #[test]
    fn repeatable_exits_match_both_engines_at_the_maximum_domain_boundary() {
        for real_rows in [MIN_PADDED_TRACE_LENGTH - 1, MIN_PADDED_TRACE_LENGTH] {
            for end in [
                End::Jal,
                End::Jalr,
                End::StableOverwritingJalr,
                End::CompressedJump,
                End::Ebreak,
            ] {
                let witness = witness(real_rows, end);
                let (reference, reference_state) = prove(&witness, false).unwrap();
                let (optimized, optimized_state) = prove(&witness, true).unwrap();
                assert_eq!(
                    reference.uniskip_proof, optimized.uniskip_proof,
                    "{end:?}, {real_rows}"
                );
                assert_eq!(
                    reference.sumcheck_proof, optimized.sumcheck_proof,
                    "{end:?}, {real_rows}"
                );
                assert_eq!(reference.claims, optimized.claims, "{end:?}, {real_rows}");
                assert_eq!(reference_state, optimized_state);
                assert_eq!(replay(&optimized).unwrap(), optimized_state);
            }
        }
    }

    #[test]
    fn endpoint_forging_path_preserves_an_honest_proof() {
        let witness = witness(MIN_PADDED_TRACE_LENGTH, End::Jal);
        let (honest, state) = prove(&witness, false).unwrap();
        let output = prove_with_forged_endpoints(&witness).unwrap();
        assert_eq!(output.uniskip_proof, honest.uniskip_proof);
        assert_eq!(output.sumcheck_proof, honest.sumcheck_proof);
        assert_eq!(output.claims, honest.claims);
        assert_eq!(replay(&output).unwrap(), state);
    }

    #[test]
    fn truncated_jump_witnesses_cannot_produce_an_accepted_stage1_proof() {
        for end in [
            End::ArbitraryJump,
            End::UnstableJalr,
            End::CompressedJalr,
            End::ExpansionMidpoint,
            End::GeneratedTerminalJump,
        ] {
            let witness = witness(MIN_PADDED_TRACE_LENGTH, end);
            let output = prove_with_forged_endpoints(&witness).unwrap();
            assert!(
                matches!(
                    replay(&output),
                    Err(VerifierError::StageClaimOutputMismatch { stage: 1 })
                ),
                "accepted {end:?}"
            );
        }
    }

    #[test]
    fn verifier_rejects_tampered_stopping_target_and_authenticated_flag() {
        let witness = witness(MIN_PADDED_TRACE_LENGTH, End::Jal);
        let (mut output, _) = prove(&witness, true).unwrap();
        let honest = output.claims.clone();
        output.claims.outer.outer_remainder.lookup_output += Fr::from_u64(1);
        assert!(replay(&output).is_err());
        output.claims = honest;
        output.claims.outer.outer_remainder.can_end = Fr::from_u64(0);
        assert!(replay(&output).is_err());
    }
}
