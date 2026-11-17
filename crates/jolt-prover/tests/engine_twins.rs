//! Twin-transcript engine locks: toy members driven through
//! `jolt_sumcheck::prove_batch` (and `prove_uniskip_clear`) against the
//! GENERATED `verify_clear` / `verify_zk` drivers and the shared uni-skip
//! `verify_clear` core, asserting byte-identical transcript states. This pins
//! the prove-side engine to the verifier independently of any real stage.
//! Protocol-agnostic: the locks exercise the shared engine/driver seam and
//! run under both the Dory and Akita builds. The base Dory build also runs
//! `halting`, which drives the real stage kernels over hand-built witnesses
//! to pin how a run may end.

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
use rand_core::OsRng;

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

    let mut recorder = CommittedSumcheckRecorder::<Fr, VC, _>::new(&setup, OsRng).unwrap();
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

/// How a run ends, against the stage-1 and stage-2 relations and bytecode
/// read-RAF. Each witness is hand-built: most ends are ones a malicious prover
/// can supply although no tracer produces them. Rejection runs on a prover that
/// forges round endpoints and skips the stage driver's final-claim self-check,
/// so only the verifier's own replay can refuse it.
#[cfg(not(any(feature = "zk", feature = "akita", feature = "field-inline")))]
mod halting {
    use std::sync::Arc;

    use super::*;
    use common::constants::RAM_START_ADDRESS;
    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_claims::protocols::jolt::geometry::spartan::{
        SpartanOuterDimensions, SpartanProductDimensions,
    };
    use jolt_claims::NoChallenges;
    use jolt_dory::DoryScheme;
    use jolt_field::Field as _;
    use jolt_kernels::uniskip::ProductUniskipUpstream;
    use jolt_kernels::{ProverInputs, SumcheckKernel};
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::expand::{expand_instruction, ExpansionAllocator};
    use jolt_program::preprocess::{compute_min_ram_k, JoltProgramPreprocessing};
    use jolt_prover::dory::stages::stage0::prove_stage0;
    use jolt_prover::stages::stage1::{prove_stage1, Stage1ProverOutput};
    use jolt_prover::stages::stage2::prove_stage2;
    use jolt_prover::stages::stage3::prove_stage3;
    use jolt_prover::stages::stage4::prove_stage4;
    use jolt_prover::stages::stage5::prove_stage5;
    use jolt_prover::stages::stage6a::prove_stage6a;
    use jolt_prover::{
        dory, JoltBackend, JoltSharedPreprocessing, ProofMode, ProverConfig, ProverError,
    };
    use jolt_riscv::{
        JoltInstructionKind as Kind, JoltInstructionRow, JoltTraceRow, NormalizedOperands,
        RamAccess, RegisterRead, RegisterState, RegisterWrite, SourceInstruction,
        SourceInstructionKind, SourceInstructionRow, RV64IMAC_JOLT,
    };
    use jolt_transcript::ProverState;
    use jolt_verifier::config::MIN_PADDED_TRACE_LENGTH;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainder,
    };
    use jolt_verifier::stages::stage1::outputs::{
        Stage1BatchInputClaims, Stage1BatchOutputClaims, Stage1BatchSumchecks, Stage1ClearOutput,
        Stage1OutputClaims,
    };
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainder,
    };
    use jolt_verifier::stages::stage2::product_tau_low;
    use jolt_verifier::stages::stage2::product_uniskip::{
        product_uniskip_input_values_from_stage1, ProductUniskip,
    };
    use jolt_verifier::stages::uniskip::{draw_spartan_outer_tau, draw_spartan_product_tau_high};
    use jolt_verifier::VerifierError;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    const ENTRY: u64 = RAM_START_ADDRESS;
    const LOG_T: usize = MIN_PADDED_TRACE_LENGTH.ilog2() as usize;
    type Stage1 = Stage1ProverOutput<Fr, Bn254G1>;

    /// The trace's last real rows; every earlier row is `ADDI x0, x0, 0`.
    #[derive(Clone, Copy, Debug)]
    enum End {
        /// `jolt.success`: one `JoltSuccess` row linking into x5.
        Success,
        /// EBREAK and c.ebreak, through the expansion's one `JoltPanic` row.
        Panic,
        CompressedPanic,
        /// `JAL x5, 4`: a jump to a real address, into padding.
        ArbitraryJump,
        /// `j .`
        SelfJump,
        /// `JALR x5, 0(x6)` with x6 = 0: a jump to address 0, as a halt's is.
        JalrToZero,
        /// The first row of a two-row JALR sequence whose target is 0; the
        /// second exists only in the bytecode. Row 14 holds, so stage 1 can
        /// reject it only through the sequence rule, row 17
        /// (`NextPC = 0 ≠ PC + 1`).
        ExpansionMidpoint,
        /// `JoltSuccess`, then one more real row.
        HaltThenReal,
        /// A `JoltPanic` trace row over a bytecode row of kind `bytecode` at
        /// the same address and PC: `JoltSuccess` forges the HaltPanic column,
        /// `JoltPanic` is the honest control.
        PanicRow {
            bytecode: Kind,
        },
    }

    fn layout() -> MemoryLayout {
        MemoryLayout::new(&MemoryConfig {
            program_size: Some(2048),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 8,
            max_output_size: 8,
            stack_size: 8,
            heap_size: 8,
        })
    }

    /// The witness of `real_rows` rows ending at `end`, and the proof shape a
    /// prover would choose for it: the padded domain and the smallest RAM
    /// domain admission accepts.
    fn run(real_rows: usize, end: End) -> (TraceBackend, ProverConfig) {
        let address = |row: usize| ENTRY as usize + 4 * row;
        let noop = |row| JoltInstructionRow {
            instruction_kind: Kind::ADDI,
            address: address(row),
            operands: NormalizedOperands {
                rd: Some(0),
                rs1: Some(0),
                ..NormalizedOperands::default()
            },
            ..JoltInstructionRow::default()
        };
        let end_row = if matches!(end, End::HaltThenReal) {
            real_rows - 2
        } else {
            real_rows - 1
        };
        let mut bytecode: Vec<_> = (0..end_row).map(noop).collect();
        let mut trace: Vec<_> = bytecode
            .iter()
            .map(|&row| (row, RegisterState::default()))
            .collect();
        let row = |instruction_kind, rs1, imm| JoltInstructionRow {
            instruction_kind,
            address: address(end_row),
            operands: NormalizedOperands {
                rd: Some(5),
                rs1,
                imm,
                ..NormalizedOperands::default()
            },
            ..JoltInstructionRow::default()
        };
        let last = match end {
            End::Success | End::HaltThenReal | End::PanicRow { .. } => {
                row(Kind::JoltSuccess, None, 0)
            }
            End::Panic | End::CompressedPanic => {
                let source = SourceInstruction::new(
                    SourceInstructionKind::EBREAK,
                    SourceInstructionRow {
                        address: address(end_row),
                        is_compressed: matches!(end, End::CompressedPanic),
                        ..SourceInstructionRow::default()
                    },
                );
                let expansion =
                    expand_instruction(&source, &mut ExpansionAllocator::new(), RV64IMAC_JOLT)
                        .unwrap();
                assert_eq!(expansion.len(), 1);
                JoltInstructionRow::from(expansion[0])
            }
            End::ArbitraryJump => row(Kind::JAL, None, 4),
            End::SelfJump => row(Kind::JAL, None, 0),
            End::JalrToZero => row(Kind::JALR, Some(6), 0),
            End::ExpansionMidpoint => JoltInstructionRow {
                virtual_sequence_remaining: Some(1),
                is_first_in_sequence: true,
                ..row(Kind::JALR, Some(6), 0)
            },
        };
        let (bytecode_row, traced) = match end {
            End::PanicRow { bytecode } => (
                JoltInstructionRow {
                    instruction_kind: bytecode,
                    ..last
                },
                JoltInstructionRow {
                    instruction_kind: Kind::JoltPanic,
                    ..last
                },
            ),
            _ => (last, last),
        };
        bytecode.push(bytecode_row);
        let link = address(end_row) as u64 + if last.is_compressed { 2 } else { 4 };
        trace.push((
            traced,
            RegisterState {
                rs1: last
                    .operands
                    .rs1
                    .map(|register| RegisterRead { register, value: 0 }),
                rd: last.operands.rd.map(|register| RegisterWrite {
                    register,
                    pre_value: 0,
                    post_value: link,
                }),
                ..RegisterState::default()
            },
        ));
        match end {
            End::ExpansionMidpoint => bytecode.push(JoltInstructionRow {
                virtual_sequence_remaining: Some(0),
                is_first_in_sequence: false,
                ..last
            }),
            End::HaltThenReal => {
                bytecode.push(noop(real_rows - 1));
                trace.push((noop(real_rows - 1), RegisterState::default()));
            }
            _ => {}
        }

        let preprocessing = Arc::new(
            JoltProgramPreprocessing::new(
                bytecode.clone(),
                vec![(ENTRY, 0)],
                layout(),
                ENTRY,
                MIN_PADDED_TRACE_LENGTH,
                RV64IMAC_JOLT,
            )
            .unwrap(),
        );
        let config = ProverConfig::from_padded_dimensions::<Fr>(
            ProverConfig::padded_trace_length::<Fr>(trace.len()).unwrap(),
            compute_min_ram_k(
                preprocessing.ram.min_bytecode_address,
                preprocessing.ram.bytecode_words.len(),
                &preprocessing.memory_layout,
            )
            .unwrap(),
            preprocessing.max_padded_trace_length,
        )
        .unwrap();
        let rows = trace
            .into_iter()
            .map(|(instruction, registers)| {
                let pc = preprocessing.bytecode.get_pc(&instruction).unwrap();
                JoltTraceRow::new(
                    instruction,
                    registers,
                    RamAccess::NoOp,
                    u32::try_from(pc).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let program = Arc::new(JoltProgram::from_parts(
            Vec::new(),
            bytecode,
            Vec::new(),
            address(real_rows) as u64,
            ENTRY,
        ));
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(LOG_T, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(
                &program,
                &preprocessing,
                TraceOutput::new(
                    TraceData::new(rows),
                    JoltDevice {
                        memory_layout: layout(),
                        ..JoltDevice::default()
                    },
                    None,
                    None,
                ),
            ),
        );
        (witness, config)
    }

    fn transcript() -> ProverState<Keccak> {
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"halting-twin"), b"")
    }

    fn prove(
        witness: &TraceBackend,
        optimized: bool,
    ) -> Result<(Stage1, [u8; 32]), ProverError<Fr>> {
        let backend = if optimized {
            JoltBackend::optimized()
        } else {
            JoltBackend::reference()
        };
        let mut context = backend.with_witness(witness);
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None)?;
        let mut transcript = transcript();
        let output = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &mode,
            LOG_T,
            &mut transcript,
        )?;
        Ok((output, transcript.verifier_message::<[u8; 32]>()))
    }

    /// A malicious prover's batch member. Where the kernel's round check
    /// fails, it keeps the kernel's actual round polynomial and shifts only
    /// its constant so the endpoints sum to the running claim.
    struct ForgedEndpoints<R: ConcreteSumcheck<Fr>> {
        inner: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    }

    impl<R: ConcreteSumcheck<Fr>> ProveRounds<Fr> for ForgedEndpoints<R> {
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
                // The kernels bind before checking the endpoints, so the
                // retry binds nothing.
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

    /// Stage 1 by the endpoint-forging prover over the reference kernels: the
    /// uni-skip round, the remainder's round polynomials and the output claims
    /// are the kernels' own over the actual witness.
    fn prove_stage1_forged(witness: &TraceBackend) -> Result<Stage1, ProverError<Fr>> {
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut context = backend.with_witness(witness);
        let mut transcript = transcript();
        let tau = draw_spartan_outer_tau(&mut transcript, LOG_T)?;
        context.prepare_outer_uniskip(LOG_T, &tau)?;
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
                SpartanOuterDimensions::rv64(LOG_T),
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
        let mut member = ForgedEndpoints {
            inner: context.prepare(ProverInputs {
                relation: &sumchecks.outer_remainder,
                claims: &inputs.outer_remainder,
                points: &input_points.outer_remainder,
                challenges: &challenges.outer_remainder,
            })?,
        };
        let proved = prove_batch(
            &batch,
            &mut [&mut member as &mut dyn ProveRounds<Fr>],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )?;
        let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
        member.inner.validate_derived_tables(
            &sumchecks.outer_remainder,
            &input_points.outer_remainder,
            &output_points.outer_remainder,
            &challenges.outer_remainder,
        )?;
        let outer = Stage1BatchOutputClaims {
            outer_remainder: member.inner.output_claims(&inputs.outer_remainder)?,
        };
        sumchecks.validate_output_claims(&outer)?;
        let recorded = recorder.finish(&sumchecks.opening_values(&outer), &mut transcript)?;
        Ok(Stage1 {
            tau_pow_nonce: None,
            pow_nonce: None,
            uniskip_proof: uniskip.proof.into(),
            sumcheck_proof: recorded.proof.into(),
            claims: Stage1OutputClaims::new(uniskip.output_claim, outer.clone()),
            clear_output: Stage1ClearOutput::new(outer, output_points),
        })
    }

    fn replay_stage1(
        stage1: &Stage1,
        transcript: &mut ProverState<Keccak>,
    ) -> Result<Stage1ClearOutput<Fr>, VerifierError> {
        let tau = draw_spartan_outer_tau(transcript, LOG_T)?;
        let challenge = uniskip::verify_clear(
            &stage1.uniskip_proof,
            &UniskipParams::spartan_outer(),
            Fr::from_u64(0),
            stage1.claims.uniskip_output_claim,
            transcript,
        )?;
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(LOG_T),
                tau,
                challenge,
            ),
        };
        let challenges = sumchecks.draw_challenges(transcript)?;
        sumchecks.validate_output_claims(&stage1.claims.outer)?;
        let inputs = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(
                stage1.claims.uniskip_output_claim,
            ),
        };
        let output_points = sumchecks.verify_clear(
            &inputs,
            &sumchecks.empty_input_points(),
            &challenges,
            &stage1.claims.outer,
            &stage1.sumcheck_proof,
            transcript,
            1,
        )?;
        sumchecks.append_output_claims(transcript, &stage1.claims.outer);
        Ok(Stage1ClearOutput::new(
            stage1.claims.outer.clone(),
            output_points,
        ))
    }

    fn replay(stage1: &Stage1) -> Result<[u8; 32], VerifierError> {
        let mut transcript = transcript();
        let _ = replay_stage1(stage1, &mut transcript)?;
        Ok(transcript.verifier_message::<[u8; 32]>())
    }

    #[derive(SumcheckBatch)]
    struct LaneSumchecks<F: JoltField> {
        product_remainder: ProductRemainder<F>,
    }

    struct LaneProof {
        uniskip_proof: SumcheckProof<Fr, Bn254G1>,
        uniskip_output_claim: Fr,
        sumcheck_proof: SumcheckProof<Fr, Bn254G1>,
        claims: LaneOutputClaims<Fr>,
    }

    /// An honest stage 1, then stage 2's product virtualization under
    /// `claimed_panic` by the endpoint-forging prover, all on the optimized
    /// backend: its uni-skip front takes lane 3's base value from the claim,
    /// so the uni-skip round accepts any claimed panic bit and the remainder
    /// carries the difference. Returns the prover's final transcript state.
    fn prove_lane(
        witness: &TraceBackend,
        claimed_panic: bool,
    ) -> Result<(Stage1, LaneProof, [u8; 32]), ProverError<Fr>> {
        let backend = JoltBackend::<Fr, DoryScheme>::optimized();
        let mut context = backend.with_witness(witness);
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None)?;
        let mut transcript = transcript();
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &mode,
            LOG_T,
            &mut transcript,
        )?;
        let tau_low = product_tau_low(&stage1.clear_output.remainder_point(), LOG_T)?;
        context.prepare_product_uniskip(LOG_T, &tau_low)?;
        let tau_high = draw_spartan_product_tau_high(&mut transcript)?;
        let dimensions = SpartanProductDimensions::new(LOG_T);
        let upstream = ProductUniskipUpstream {
            openings: product_uniskip_input_values_from_stage1(&stage1.clear_output),
            claimed_panic,
        };
        let input_claim = ProductUniskip::new(dimensions, tau_high, claimed_panic)
            .input_claim(&upstream.openings, &NoChallenges::default())?;
        let params = UniskipParams::spartan_product();
        let uniskip = prove_uniskip_clear::<Fr, _>(
            context.product_uniskip_poly(&[tau_high], &upstream)?,
            input_claim,
            params.degree(),
            params.domain_size(),
            &mut transcript,
        )?;
        let sumchecks = LaneSumchecks {
            product_remainder: ProductRemainder::new(
                dimensions,
                uniskip.challenge,
                tau_high,
                tau_low,
            ),
        };
        let challenges = sumchecks.draw_challenges(&mut transcript)?;
        let input_points = sumchecks.empty_input_points();
        let inputs = LaneInputClaims {
            product_remainder: product_remainder_input_values_from_uniskip_output(
                uniskip.output_claim,
            ),
        };
        let mut recorder = ClearSumcheckRecorder::<Fr>::new();
        let (batch, _) =
            sumchecks.begin_batch(&inputs, &challenges, &mut recorder, &mut transcript)?;
        let mut member = ForgedEndpoints {
            inner: context.prepare(ProverInputs {
                relation: &sumchecks.product_remainder,
                claims: &inputs.product_remainder,
                points: &input_points.product_remainder,
                challenges: &challenges.product_remainder,
            })?,
        };
        let proved = prove_batch(
            &batch,
            &mut [&mut member as &mut dyn ProveRounds<Fr>],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )?;
        let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
        member.inner.validate_derived_tables(
            &sumchecks.product_remainder,
            &input_points.product_remainder,
            &output_points.product_remainder,
            &challenges.product_remainder,
        )?;
        let claims = LaneOutputClaims {
            product_remainder: member.inner.output_claims(&inputs.product_remainder)?,
        };
        sumchecks.validate_output_claims(&claims)?;
        let recorded = recorder.finish(&sumchecks.opening_values(&claims), &mut transcript)?;
        let lane = LaneProof {
            uniskip_proof: uniskip.proof.into(),
            uniskip_output_claim: uniskip.output_claim,
            sumcheck_proof: recorded.proof.into(),
            claims,
        };
        Ok((stage1, lane, transcript.verifier_message::<[u8; 32]>()))
    }

    fn replay_lane(
        stage1: &Stage1,
        lane: &LaneProof,
        claimed_panic: bool,
    ) -> Result<[u8; 32], VerifierError> {
        let mut transcript = transcript();
        let stage1 = replay_stage1(stage1, &mut transcript)?;
        let tau_low = product_tau_low(&stage1.remainder_point(), LOG_T)?;
        let tau_high = draw_spartan_product_tau_high(&mut transcript)?;
        let dimensions = SpartanProductDimensions::new(LOG_T);
        let input_claim = ProductUniskip::new(dimensions, tau_high, claimed_panic).input_claim(
            &product_uniskip_input_values_from_stage1(&stage1),
            &NoChallenges::default(),
        )?;
        let challenge = uniskip::verify_clear(
            &lane.uniskip_proof,
            &UniskipParams::spartan_product(),
            input_claim,
            lane.uniskip_output_claim,
            &mut transcript,
        )?;
        let sumchecks = LaneSumchecks {
            product_remainder: ProductRemainder::new(dimensions, challenge, tau_high, tau_low),
        };
        let challenges = sumchecks.draw_challenges(&mut transcript)?;
        sumchecks.validate_output_claims(&lane.claims)?;
        let inputs = LaneInputClaims {
            product_remainder: product_remainder_input_values_from_uniskip_output(
                lane.uniskip_output_claim,
            ),
        };
        let _ = sumchecks.verify_clear(
            &inputs,
            &sumchecks.empty_input_points(),
            &challenges,
            &lane.claims,
            &lane.sumcheck_proof,
            &mut transcript,
            2,
        )?;
        sumchecks.append_output_claims(&mut transcript, &lane.claims);
        Ok(transcript.verifier_message::<[u8; 32]>())
    }

    fn prove_through_stage6a(
        witness: &TraceBackend,
        config: &ProverConfig,
        claimed_panic: bool,
    ) -> Result<(), ProverError<Fr>> {
        let preprocessing = dory::from_shared(
            JoltSharedPreprocessing::try_from(Arc::clone(&witness.preprocessing)).unwrap(),
        )
        .unwrap();
        let public_io = JoltDevice {
            panic: claimed_panic,
            ..witness.trace.device.clone()
        };
        let backend = JoltBackend::optimized();
        let mut context = backend.with_witness(witness);
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let stage0 = prove_stage0::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &preprocessing,
            config,
            None,
            &public_io,
        )
        .unwrap();
        let checked = stage0.checked;
        let mut transcript = stage0.transcript;
        let stage1 = prove_stage1(&mut context, &mode, LOG_T, &mut transcript).unwrap();
        let stage2 = prove_stage2(
            &mut context,
            &mode,
            &checked,
            config,
            &public_io,
            &stage1.clear_output,
            &mut transcript,
        )
        .unwrap();
        let stage3 = prove_stage3(
            &mut context,
            &mode,
            config,
            &stage1.clear_output,
            &stage2.clear_output,
            &mut transcript,
        )
        .unwrap();
        let stage4 = prove_stage4(
            &mut context,
            &mode,
            &checked,
            config,
            &preprocessing,
            &stage2.clear_output,
            &stage3.clear_output,
            &mut transcript,
        )
        .unwrap();
        let stage5 = prove_stage5(
            &mut context,
            &mode,
            &checked,
            config,
            &preprocessing,
            &stage2.clear_output,
            &stage4.clear_output,
            &mut transcript,
        )
        .unwrap();
        prove_stage6a(
            &mut context,
            &mode,
            &checked,
            config,
            &preprocessing,
            &stage1.clear_output,
            &stage2.clear_output,
            &stage3.clear_output,
            &stage4.clear_output,
            &stage5.clear_output,
            &mut transcript,
        )
        .map(|_| ())
    }

    #[test]
    fn halts_match_both_engines_at_the_domain_boundary() {
        for real_rows in [MIN_PADDED_TRACE_LENGTH - 1, MIN_PADDED_TRACE_LENGTH] {
            for end in [End::Success, End::Panic, End::CompressedPanic] {
                let (witness, _) = run(real_rows, end);
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
                assert_eq!(reference_state, optimized_state, "{end:?}, {real_rows}");
                assert_eq!(replay(&optimized).unwrap(), optimized_state);
            }
        }
    }

    #[test]
    fn endpoint_forging_path_preserves_an_honest_proof() {
        let (witness, _) = run(MIN_PADDED_TRACE_LENGTH, End::Success);
        let (honest, state) = prove(&witness, false).unwrap();
        let forged = prove_stage1_forged(&witness).unwrap();
        assert_eq!(forged.uniskip_proof, honest.uniskip_proof);
        assert_eq!(forged.sumcheck_proof, honest.sumcheck_proof);
        assert_eq!(forged.claims, honest.claims);
        assert_eq!(replay(&forged).unwrap(), state);
    }

    #[test]
    fn truncated_or_misplaced_ends_are_rejected_in_stage1() {
        for end in [
            End::ArbitraryJump,
            End::SelfJump,
            End::ExpansionMidpoint,
            End::HaltThenReal,
        ] {
            let (witness, _) = run(MIN_PADDED_TRACE_LENGTH, end);
            let forged = prove_stage1_forged(&witness).unwrap();
            assert!(
                matches!(
                    replay(&forged),
                    Err(VerifierError::StageClaimOutputMismatch { stage: 1 })
                ),
                "accepted {end:?}"
            );
        }
    }

    /// Flipping `public_io.panic` alone diverges the transcript before stage
    /// 1, so it cannot isolate lane 3; here the prover keeps the transcript
    /// and claims the opposite panic bit only in the product uni-skip input.
    #[test]
    fn lane_three_binds_the_claimed_panic_bit() {
        for (end, panics) in [
            (End::Success, false),
            (End::Panic, true),
            (End::CompressedPanic, true),
        ] {
            let (witness, _) = run(MIN_PADDED_TRACE_LENGTH, end);
            let (stage1, lane, state) = prove_lane(&witness, panics).unwrap();
            assert_eq!(
                replay_lane(&stage1, &lane, panics).unwrap(),
                state,
                "{end:?}"
            );
            let (stage1, lane, _) = prove_lane(&witness, !panics).unwrap();
            assert!(
                matches!(
                    replay_lane(&stage1, &lane, !panics),
                    Err(VerifierError::StageClaimOutputMismatch { stage: 2 })
                ),
                "{end:?} verified with panic = {}",
                !panics
            );
        }
    }

    /// Row 14 holds for a jump to address 0 into padding; lane 3 refuses it
    /// under either claim because the row is not a halt.
    #[test]
    fn jump_into_padding_is_rejected_in_stage2() {
        let (witness, _) = run(MIN_PADDED_TRACE_LENGTH, End::JalrToZero);
        for claimed_panic in [false, true] {
            let (stage1, lane, _) = prove_lane(&witness, claimed_panic).unwrap();
            assert!(
                matches!(
                    replay_lane(&stage1, &lane, claimed_panic),
                    Err(VerifierError::StageClaimOutputMismatch { stage: 2 })
                ),
                "panic = {claimed_panic}"
            );
        }
    }

    /// Stages 1 and 2 never read the bytecode, so a trace that claims a
    /// failure halt where the program has `jolt.success` satisfies them
    /// under panic = true. Bytecode read-RAF ties the trace's HaltPanic
    /// column to the program: stage 6a's final claim disagrees with its
    /// output fold. The control differs only in the bytecode row's kind.
    #[test]
    fn forged_halt_panic_column_is_bound_by_bytecode_read_raf() {
        let (forged, config) = run(
            MIN_PADDED_TRACE_LENGTH,
            End::PanicRow {
                bytecode: Kind::JoltSuccess,
            },
        );
        let (stage1, lane, state) = prove_lane(&forged, true).unwrap();
        assert_eq!(replay_lane(&stage1, &lane, true).unwrap(), state);
        let result = prove_through_stage6a(&forged, &config, true);
        assert!(
            matches!(
                &result,
                Err(ProverError::Verifier(VerifierError::StageClaimSumcheckFailed { stage, .. }))
                    if stage == "Stage6a"
            ),
            "{result:?}"
        );

        let (control, config) = run(
            MIN_PADDED_TRACE_LENGTH,
            End::PanicRow {
                bytecode: Kind::JoltPanic,
            },
        );
        prove_through_stage6a(&control, &config, true).unwrap();
    }
}
