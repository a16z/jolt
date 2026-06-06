use std::sync::{Arc, Mutex};

use common::jolt_device::JoltDevice;
use jolt_backends::{
    cpu::{CpuBackend, CpuBackendConfig},
    CommitmentBackend, CommitmentRequest, CommitmentRequestItem, CommitmentSlot, SumcheckBackend,
    SumcheckProductUniskipRow,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
        claim_reductions::{advice, increments},
        instruction, ram,
        ram::RamRafEvaluationDimensions,
        spartan::SPARTAN_OUTER_R1CS_INPUTS,
    },
    AdviceClaimReductionLayout, IncClaimReductionChallenge, InstructionRaVirtualizationChallenge,
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltFormulaDimensions,
    JoltOneHotConfig, JoltVirtualPolynomial,
};
use jolt_core::{
    curve::Bn254Curve,
    host,
    poly::{
        commitment::{
            commitment_scheme::{
                CommitmentScheme as CoreCommitmentScheme,
                StreamingCommitmentScheme as CoreStreamingCommitmentScheme,
            },
            dory::{DoryCommitmentScheme, DoryContext, DoryGlobals, DoryLayout},
        },
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof as core_opening,
    },
    zkvm::{
        instruction as core_instruction,
        prover::JoltProverPreprocessing,
        r1cs::inputs::ProductCycleInputs,
        ram::populate_memory_states,
        verifier::{
            JoltSharedPreprocessing, JoltVerifierPreprocessing as CoreVerifierPreprocessing,
        },
        witness as core_witness, RV64IMACProof, RV64IMACProver,
    },
};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryCommitment, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, RingCore};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_prover::initialize_proof_transcript;
use jolt_prover::stages::stage0::{prove, CommitmentStageConfig, CommitmentStageInput};
use jolt_prover::stages::stage1::{
    input::{Stage1ProverConfig, Stage1ProverInput},
    output::{stage1_claims_from_r1cs_inputs, Stage1R1csInputClaim},
    prove::{evaluate_stage1_r1cs_inputs, prove as prove_stage1},
    request::{build_stage1_r1cs_materialization_request, r1cs_input_slot},
};
use jolt_prover::stages::stage2::{
    input::{
        Stage2BatchProverConfig, Stage2ProductUniSkipInput, Stage2ProverConfig, Stage2ProverInput,
    },
    output::{
        Stage2ProductUniSkipOutput, Stage2ProverOutput, Stage2RamTerminalOutputOpeningClaims,
        Stage2RegularBatchInputClaims, Stage2RegularBatchPrefixOutput,
    },
    prove::{
        derive_stage2_regular_batch_prefix, evaluate_stage2_instruction_claim_openings,
        evaluate_stage2_product_remainder_openings, evaluate_stage2_ram_read_write_openings,
        evaluate_stage2_ram_terminal_openings, prove as prove_stage2, prove_stage2_product_uniskip,
        prove_stage2_regular_batch_sumcheck_for_frontier, Stage2RegularBatchFrontierProof,
    },
};
use jolt_prover::stages::stage3::{
    input::{Stage3ProverConfig, Stage3ProverInput},
    output::{
        Stage3ProverOutput, Stage3RegularBatchInputClaims, Stage3RegularBatchOutputOpeningClaims,
        Stage3RegularBatchPrefixOutput,
    },
    prove::{
        derive_stage3_regular_batch_prefix, evaluate_stage3_output_openings, prove as prove_stage3,
        prove_stage3_regular_batch_sumcheck_for_frontier, Stage3RegularBatchFrontierProof,
    },
};
use jolt_prover::stages::stage4::{
    input::{Stage4ProverConfig, Stage4ProverInput},
    output::{
        Stage4ProverOutput, Stage4RamValCheckAdviceContribution,
        Stage4RamValCheckInitialEvaluation, Stage4RegularBatchInputClaims,
        Stage4RegularBatchOutputOpeningClaims, Stage4RegularBatchPrefixOutput,
    },
    prove::{
        derive_stage4_regular_batch_prefix, evaluate_stage4_output_openings, prove as prove_stage4,
        prove_stage4_regular_batch_sumcheck_for_frontier, Stage4RegularBatchFrontierProof,
    },
};
use jolt_prover::stages::stage5::{
    input::{Stage5ProverConfig, Stage5ProverInput},
    output::{
        Stage5ProverOutput, Stage5RegularBatchInputClaims, Stage5RegularBatchOutputOpeningClaims,
        Stage5RegularBatchPrefixOutput,
    },
    prove::{
        derive_stage5_regular_batch_prefix, evaluate_stage5_output_openings, prove as prove_stage5,
        prove_stage5_regular_batch_sumcheck_for_frontier, Stage5RegularBatchFrontierProof,
    },
};
use jolt_prover::stages::stage6::{
    input::{Stage6ProverConfig, Stage6ProverInput},
    output::{
        Stage6ProverOutput, Stage6RegularBatchInputClaims, Stage6RegularBatchOutputOpeningClaims,
        Stage6RegularBatchPrefixOutput, Stage6RegularBatchProofOutput,
    },
    prove::{
        derive_stage6_regular_batch_prefix, evaluate_stage6_output_openings, prove as prove_stage6,
        prove_stage6_transparent_sumchecks,
    },
};
use jolt_prover::stages::stage7::{
    input::{Stage7ProverConfig, Stage7ProverInput},
    output::{Stage7ProverOutput, Stage7RegularBatchInputClaims, Stage7RegularBatchPrefixOutput},
    prove::{derive_stage7_regular_batch_prefix, prove as prove_stage7},
};
use jolt_prover::stages::stage8::{
    input::Stage8ProverConfig,
    output::Stage8OpeningStructure,
    prove::{
        evaluate_stage8_dense_constituents, evaluate_stage8_joint_polynomial,
        evaluate_stage8_ra_constituents, prove_stage8,
    },
    request::derive_stage8_opening_structure,
};
use jolt_riscv::{CircuitFlags, CIRCUIT_FLAGS};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{
    compat::convert::{CoreFieldBridge, CorePcsBridge, ImportedCoreProof},
    proof::JoltProofClaims,
    stages::stage1::{
        inputs::{SpartanOuterFlagClaims, Stage1Claims},
        Stage1ClearOutput, Stage1Output,
    },
    stages::stage2::{
        inputs::{
            InstructionClaimReductionOutputOpeningClaims, ProductRemainderOutputOpeningClaims,
            RamReadWriteOutputOpeningClaims, Stage2BatchOutputOpeningClaims,
        },
        Stage2ClearOutput, Stage2Output,
    },
    stages::stage3::{Stage3ClearOutput, Stage3Output},
    stages::stage4::{Stage4ClearOutput, Stage4Output},
    stages::stage5::{Stage5ClearOutput, Stage5Output},
    stages::stage6::{Stage6ClearOutput, Stage6Output},
    verify, JoltVerifierPreprocessing, VerifierError,
};
use jolt_witness::protocols::jolt_vm::{
    JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness, RV64_LOOKUP_ADDRESS_BITS,
};
use jolt_witness::{
    MaterializationPolicy, OracleRef, PolynomialEncoding, RetentionHint, ViewRequirement,
};
use rayon::prelude::*;
use tracer::instruction::Cycle as CoreTraceCycle;

use crate::{FeatureMode, FixtureKind, FixtureRequest, HarnessError, HarnessResult};

static CORE_FIXTURE_LOCK: Mutex<()> = Mutex::new(());

type CoreField = jolt_core::ark_bn254::Fr;
type CoreProof = RV64IMACProof;
type CoreCommitment = <DoryCommitmentScheme as CoreCommitmentScheme>::Commitment;
type CoreOpeningHint = <DoryCommitmentScheme as CoreCommitmentScheme>::OpeningProofHint;
type CoreProverSetup = <DoryCommitmentScheme as CoreCommitmentScheme>::ProverSetup;
type ConvertedProof = ImportedCoreProof<CoreField, Bn254Curve, DoryCommitmentScheme>;
type ConvertedPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type TrustedAdviceCommitter = fn(
    &JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    &[u8],
) -> (CoreCommitment, CoreOpeningHint);

#[derive(Clone)]
pub struct CoreVerifierFixture {
    pub preprocessing: ConvertedPreprocessing,
    pub public_io: JoltDevice,
    pub proof: ConvertedProof,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

impl CoreVerifierFixture {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            self.trusted_advice_commitment.as_ref(),
            false,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage0CommitmentKernelShape {
    pub trace_length: usize,
    pub log_t: usize,
    pub log_k_chunk: usize,
    pub dense_polynomials: usize,
    pub one_hot_polynomials: usize,
    pub committed_polynomials: usize,
    pub dory_rows: usize,
    pub dory_columns: usize,
    pub core_cycle_chunks: usize,
}

pub struct Stage0CommitmentKernelBenchmarkFixture {
    generated: GeneratedCoreFixture,
    core_trace: Arc<Vec<CoreTraceCycle>>,
    core_generators: CoreProverSetup,
}

impl Stage0CommitmentKernelBenchmarkFixture {
    pub fn core_trace(&self) -> &[tracer::instruction::Cycle] {
        &self.core_trace
    }

    pub fn core_bytecode(&self) -> &jolt_core::zkvm::bytecode::BytecodePreprocessing {
        &self.generated.core_preprocessing.shared.bytecode
    }

    pub fn core_memory_layout(&self) -> &common::jolt_device::MemoryLayout {
        &self.generated.trace.device.memory_layout
    }

    pub fn core_one_hot_params(&self) -> jolt_core::zkvm::config::OneHotParams {
        jolt_core::zkvm::config::OneHotParams::from_config(
            &self.generated.proof.one_hot_config,
            self.generated.core_preprocessing.shared.bytecode.code_size,
            self.generated.proof.ram_K,
        )
    }

    pub fn shape(&self) -> HarnessResult<Stage0CommitmentKernelShape> {
        let one_hot_params = self.core_one_hot_params();
        let log_t = checked_power_of_two_log(
            &FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent),
            self.generated.proof.trace_length,
        )?;
        let total_vars = log_t + one_hot_params.log_k_chunk;
        let sigma = total_vars.div_ceil(2);
        let dory_columns = 1usize << sigma;
        let dory_rows = 1usize << (total_vars - sigma);
        let dense_polynomials = 2;
        let one_hot_polynomials =
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d;
        Ok(Stage0CommitmentKernelShape {
            trace_length: self.generated.proof.trace_length,
            log_t,
            log_k_chunk: one_hot_params.log_k_chunk,
            dense_polynomials,
            one_hot_polynomials,
            committed_polynomials: dense_polynomials + one_hot_polynomials,
            dory_rows,
            dory_columns,
            core_cycle_chunks: self.generated.proof.trace_length / dory_columns,
        })
    }

    pub fn run_core_streaming_commitments(&self) -> HarnessResult<usize> {
        let fixture = &self.generated;
        if self.core_trace.len() != fixture.proof.trace_length {
            return Err(core_fixture_error(
                &FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent),
                "benchmark core stage0 commitments",
                format!(
                    "core trace length {} differs from proof trace length {}",
                    self.core_trace.len(),
                    fixture.proof.trace_length
                ),
            ));
        }

        let one_hot_params = self.core_one_hot_params();
        DoryGlobals::initialize_context(
            1 << one_hot_params.log_k_chunk,
            fixture.proof.trace_length,
            DoryContext::Main,
            Some(DoryLayout::CycleMajor),
        )
        .ok_or_else(|| {
            core_fixture_error(
                &FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent),
                "initialize core Dory streaming context",
                "DoryGlobals::initialize_context returned None",
            )
        })?;

        let polynomials = core_witness::all_committed_polynomials(&one_hot_params);
        let trace_len = DoryGlobals::get_T();
        let row_len = DoryGlobals::get_num_columns();
        if trace_len != self.core_trace.len() {
            return Err(core_fixture_error(
                &FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent),
                "benchmark core stage0 commitments",
                format!(
                    "Dory T {trace_len} differs from core trace length {}",
                    self.core_trace.len()
                ),
            ));
        }

        let chunk_rows = trace_len / DoryGlobals::get_max_num_rows();
        let mut row_commitments: Vec<
            Vec<<DoryCommitmentScheme as CoreStreamingCommitmentScheme>::ChunkState>,
        > = vec![Vec::new(); chunk_rows];

        self.core_trace
            .chunks(row_len)
            .zip(row_commitments.iter_mut())
            .par_bridge()
            .for_each(|(chunk, row_tier1_commitments)| {
                let commitments = polynomials
                    .par_iter()
                    .map(|poly| {
                        poly.stream_witness_and_commit_rows::<CoreField, DoryCommitmentScheme>(
                            &self.core_generators,
                            &fixture.core_preprocessing.shared,
                            chunk,
                            &one_hot_params,
                        )
                    })
                    .collect();
                *row_tier1_commitments = commitments;
            });

        let tier1_per_poly: Vec<
            Vec<<DoryCommitmentScheme as CoreStreamingCommitmentScheme>::ChunkState>,
        > = (0..polynomials.len())
            .into_par_iter()
            .map(|poly_index| {
                row_commitments
                    .iter()
                    .filter_map(|row| row.get(poly_index).cloned())
                    .collect()
            })
            .collect();

        let commitments = tier1_per_poly
            .into_par_iter()
            .zip(polynomials.par_iter())
            .map(|(tier1_commitments, poly)| {
                let onehot_k = poly.get_onehot_k(&one_hot_params);
                <DoryCommitmentScheme as CoreStreamingCommitmentScheme>::aggregate_chunks(
                    &self.core_generators,
                    onehot_k,
                    &tier1_commitments,
                )
                .0
            })
            .collect::<Vec<_>>();

        Ok(commitments.len())
    }

    pub fn run_modular_streaming_commitments(&self) -> HarnessResult<usize> {
        let output = prove_stage0_commitments(
            &FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent),
            &self.generated,
        )?;
        Ok(2 + output.commitments.ra.instruction.len()
            + output.commitments.ra.ram.len()
            + output.commitments.ra.bytecode.len())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage0AdviceCommitmentKernelShape {
    pub trusted_rows: usize,
    pub untrusted_rows: usize,
    pub trusted_row_width: usize,
    pub untrusted_row_width: usize,
    pub trusted_pcs_rows: usize,
    pub untrusted_pcs_rows: usize,
    pub commitment_count: usize,
}

pub struct Stage0AdviceCommitmentKernelBenchmarkFixture {
    generated: GeneratedCoreFixture,
    preprocessing: JoltProgramPreprocessing,
    witness_config: JoltVmWitnessConfig,
    core_generators: CoreProverSetup,
}

impl Stage0AdviceCommitmentKernelBenchmarkFixture {
    pub fn shape(&self) -> Stage0AdviceCommitmentKernelShape {
        let layout = &self.generated.core_preprocessing.shared.memory_layout;
        let trusted_rows = advice_rows(layout.max_trusted_advice_size);
        let untrusted_rows = advice_rows(layout.max_untrusted_advice_size);
        let trusted_row_width = dory_row_width(trusted_rows);
        let untrusted_row_width = dory_row_width(untrusted_rows);
        Stage0AdviceCommitmentKernelShape {
            trusted_rows,
            untrusted_rows,
            trusted_row_width,
            untrusted_row_width,
            trusted_pcs_rows: trusted_rows / trusted_row_width,
            untrusted_pcs_rows: untrusted_rows / untrusted_row_width,
            commitment_count: Stage0AdviceCommitments::COUNT,
        }
    }

    pub fn verify_commitment_parity(&self) -> HarnessResult<()> {
        let core = self.core_advice_commitments()?;
        let modular = self.modular_advice_commitments()?;
        if core == modular {
            return Ok(());
        }
        Err(core_fixture_error(
            &FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent),
            "compare stage0 advice commitment contexts",
            format!("core commitments {core:?} differ from modular commitments {modular:?}"),
        ))
    }

    pub fn run_core_advice_context_commitments(&self) -> HarnessResult<usize> {
        let _commitments = self.core_advice_commitments()?;
        Ok(Stage0AdviceCommitments::COUNT)
    }

    pub fn run_modular_advice_context_commitments(&self) -> HarnessResult<usize> {
        let _commitments = self.modular_advice_commitments()?;
        Ok(Stage0AdviceCommitments::COUNT)
    }

    fn core_advice_commitments(&self) -> HarnessResult<Stage0AdviceCommitments> {
        self.require_advice_bytes()?;
        let (trusted, trusted_hint) = commit_core_advice_context(
            &self.generated.core_preprocessing.shared,
            &self.core_generators,
            JoltAdviceKind::Trusted,
            &self.generated.trace.device.trusted_advice,
        );
        let (untrusted, untrusted_hint) = commit_core_advice_context(
            &self.generated.core_preprocessing.shared,
            &self.core_generators,
            JoltAdviceKind::Untrusted,
            &self.generated.trace.device.untrusted_advice,
        );
        let _retained_hints = std::hint::black_box((&trusted_hint, &untrusted_hint));
        Ok(Stage0AdviceCommitments {
            trusted: <DoryCommitmentScheme as CorePcsBridge<CoreField>>::commitment_into_verifier(
                trusted,
            ),
            untrusted: <DoryCommitmentScheme as CorePcsBridge<CoreField>>::commitment_into_verifier(
                untrusted,
            ),
        })
    }

    fn modular_advice_commitments(&self) -> HarnessResult<Stage0AdviceCommitments> {
        self.require_advice_bytes()?;
        let request = stage0_advice_commitment_request();
        let trace = TraceOutput::new(
            OwnedTrace::default(),
            self.generated.trace.device.clone(),
            None,
        );
        let witness = TraceBackedJoltVmWitness::new(
            self.witness_config.clone(),
            JoltVmWitnessInputs::new(&self.generated.program, &self.preprocessing, trace),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let result = <CpuBackend as CommitmentBackend<Fr, _, DoryScheme>>::commit(
            &mut backend,
            &request,
            &witness,
            &self.generated.prover_setup,
        )
        .map_err(|error| {
            core_fixture_error(
                &FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent),
                "prove modular stage0 advice commitment contexts",
                error,
            )
        })?;
        Stage0AdviceCommitments::from_modular_result(result)
    }

    fn require_advice_bytes(&self) -> HarnessResult<()> {
        if !self.generated.trace.device.trusted_advice.is_empty()
            && !self.generated.trace.device.untrusted_advice.is_empty()
        {
            return Ok(());
        }
        Err(core_fixture_error(
            &FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent),
            "load stage0 advice commitment kernel fixture",
            "fixture must contain both trusted and untrusted advice bytes",
        ))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Stage0AdviceCommitments {
    trusted: DoryCommitment,
    untrusted: DoryCommitment,
}

impl Stage0AdviceCommitments {
    const COUNT: usize = 2;

    fn from_modular_result(
        result: jolt_backends::CommitmentResult<
            jolt_witness::protocols::jolt_vm::JoltVmNamespace,
            DoryScheme,
        >,
    ) -> HarnessResult<Self> {
        let mut trusted = None;
        let mut untrusted = None;
        for output in result.commitments {
            match output.oracle.kind {
                jolt_witness::OracleKind::Committed(JoltCommittedPolynomial::TrustedAdvice) => {
                    trusted = Some(output.commitment);
                }
                jolt_witness::OracleKind::Committed(JoltCommittedPolynomial::UntrustedAdvice) => {
                    untrusted = Some(output.commitment);
                }
                _ => {
                    return Err(core_fixture_error(
                        &FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent),
                        "collect modular stage0 advice commitments",
                        format!("unexpected commitment output {:?}", output.oracle.kind),
                    ));
                }
            }
        }
        Ok(Self {
            trusted: trusted.ok_or_else(|| {
                core_fixture_error(
                    &FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent),
                    "collect modular stage0 advice commitments",
                    "missing trusted advice commitment",
                )
            })?,
            untrusted: untrusted.ok_or_else(|| {
                core_fixture_error(
                    &FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent),
                    "collect modular stage0 advice commitments",
                    "missing untrusted advice commitment",
                )
            })?,
        })
    }
}

#[derive(Clone)]
pub struct Stage1SpartanOuterCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub opening_point: Vec<Fr>,
    pub claim_count: usize,
}

#[derive(Clone)]
pub struct Stage1SpartanOuterKernelBenchmarkFixture {
    pub core_trace: Arc<Vec<CoreTraceCycle>>,
    pub core_bytecode: Arc<jolt_core::zkvm::bytecode::BytecodePreprocessing>,
    pub log_t: usize,
    pub r1cs_inputs: Vec<Vec<Fr>>,
}

#[derive(Clone)]
pub struct Stage2RegularBatchInputCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage2RegularBatchPrefixOutput<Fr>,
    pub expected: Stage2RegularBatchInputClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage2RegularBatchInputKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage2BatchProverConfig,
    pub stage1: Stage1ClearOutput<Fr>,
    pub product_uniskip: Stage2ProductUniSkipOutput<Fr, Bn254G1>,
    pub transcript: Blake2bTranscript<Fr>,
    pub expected: Stage2RegularBatchPrefixOutput<Fr>,
}

impl Stage2RegularBatchInputKernelBenchmarkFixture {
    pub fn run_reference_prefix(&self) -> HarnessResult<Stage2RegularBatchPrefixOutput<Fr>> {
        reference_stage2_regular_batch_prefix(
            self.config,
            &self.stage1,
            &self.product_uniskip,
            &mut self.transcript.clone(),
        )
    }

    pub fn run_modular_prefix(&self) -> HarnessResult<Stage2RegularBatchPrefixOutput<Fr>> {
        derive_stage2_regular_batch_prefix(
            self.config,
            &self.stage1,
            &self.product_uniskip,
            &mut self.transcript.clone(),
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "derive modular Stage 2 regular-batch input claims",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage3RegularBatchInputKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage3ProverConfig,
    pub stage1: Stage1ClearOutput<Fr>,
    pub stage2: Stage2ClearOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub expected: Stage3RegularBatchPrefixOutput<Fr>,
}

impl Stage3RegularBatchInputKernelBenchmarkFixture {
    pub fn run_reference_prefix(&self) -> Stage3RegularBatchPrefixOutput<Fr> {
        reference_stage3_regular_batch_prefix(
            &self.stage1,
            &self.stage2,
            &mut self.transcript.clone(),
        )
    }

    pub fn run_modular_prefix(&self) -> HarnessResult<Stage3RegularBatchPrefixOutput<Fr>> {
        derive_stage3_regular_batch_prefix(
            self.config,
            &self.stage1,
            &self.stage2,
            &mut self.transcript.clone(),
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "derive modular Stage 3 regular-batch input claims",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage4RegularBatchInputKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage4ProverConfig,
    pub stage2: Stage2ClearOutput<Fr>,
    pub stage3: Stage3ClearOutput<Fr>,
    pub ram_val_check_init: Stage4RamValCheckInitialEvaluation<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub expected: Stage4RegularBatchPrefixOutput<Fr>,
}

impl Stage4RegularBatchInputKernelBenchmarkFixture {
    pub fn run_reference_prefix(&self) -> Stage4RegularBatchPrefixOutput<Fr> {
        reference_stage4_regular_batch_prefix(
            &self.stage2,
            &self.stage3,
            self.ram_val_check_init.clone(),
            &mut self.transcript.clone(),
        )
    }

    pub fn run_modular_prefix(&self) -> HarnessResult<Stage4RegularBatchPrefixOutput<Fr>> {
        derive_stage4_regular_batch_prefix(
            self.config,
            &self.stage2,
            &self.stage3,
            self.ram_val_check_init.clone(),
            &mut self.transcript.clone(),
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "derive modular Stage 4 regular-batch input claims",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage5RegularBatchInputKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage5ProverConfig,
    pub stage2: Stage2ClearOutput<Fr>,
    pub stage4: Stage4ClearOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub expected: Stage5RegularBatchPrefixOutput<Fr>,
}

impl Stage5RegularBatchInputKernelBenchmarkFixture {
    pub fn run_reference_prefix(&self) -> Stage5RegularBatchPrefixOutput<Fr> {
        reference_stage5_regular_batch_prefix(
            &self.stage2,
            &self.stage4,
            &mut self.transcript.clone(),
        )
    }

    pub fn run_modular_prefix(&self) -> HarnessResult<Stage5RegularBatchPrefixOutput<Fr>> {
        derive_stage5_regular_batch_prefix(
            self.config,
            &self.stage2,
            &self.stage4,
            &mut self.transcript.clone(),
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "derive modular Stage 5 regular-batch input claims",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage5RegularBatchSumcheckExpected {
    pub proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    pub challenges: Vec<Fr>,
    pub batching_coefficients: Vec<Fr>,
    pub output_claim: Fr,
}

#[derive(Clone)]
pub struct Stage5RegularBatchSumcheckKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage5ProverConfig,
    pub checked: jolt_verifier::CheckedInputs,
    pub one_hot_config: JoltOneHotConfig,
    pub stage2: Stage2ClearOutput<Fr>,
    pub stage4: Stage4ClearOutput<Fr>,
    pub prefix: Stage5RegularBatchPrefixOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub expected: Stage5RegularBatchSumcheckExpected,
    core: Stage2RegularBatchCoreBenchmarkContext,
}

impl Stage5RegularBatchSumcheckKernelBenchmarkFixture {
    pub fn run_reference_sumcheck(&self) -> HarnessResult<usize> {
        use jolt_core::{
            poly::opening_proof::ProverOpeningAccumulator,
            subprotocols::{sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
            transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _},
            zkvm::{
                claim_reductions::{RaReductionParams, RamRaClaimReductionSumcheckProver},
                instruction_lookups::read_raf_checking::{
                    InstructionReadRafSumcheckParams, InstructionReadRafSumcheckProver,
                },
                registers::val_evaluation::{
                    RegistersValEvaluationSumcheckParams, ValEvaluationSumcheckProver,
                },
            },
        };

        let mut opening_accumulator = ProverOpeningAccumulator::<CoreField>::new(self.config.log_t);
        seed_core_stage5_openings(&mut opening_accumulator, &self.stage2, &self.stage4);

        let mut transcript = CoreBlake2bTranscript::new(b"Jolt");
        let instruction_params = InstructionReadRafSumcheckParams::new(
            self.core.trace.len().ilog2() as usize,
            &self.core.one_hot_params,
            &opening_accumulator,
            &mut transcript,
        );
        let ram_params = RaReductionParams::new(
            self.core.trace.len(),
            &self.core.one_hot_params,
            &opening_accumulator,
            &mut transcript,
        );
        let registers_params = RegistersValEvaluationSumcheckParams::new(&opening_accumulator);

        let instruction = InstructionReadRafSumcheckProver::initialize(
            instruction_params,
            Arc::clone(&self.core.trace),
        );
        let ram = RamRaClaimReductionSumcheckProver::initialize(
            ram_params,
            &self.core.trace,
            &self.core.program_io.memory_layout,
            &self.core.one_hot_params,
        );
        let registers = ValEvaluationSumcheckProver::initialize(
            registers_params,
            &self.core.trace,
            &self.core.bytecode,
            &self.core.program_io.memory_layout,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(instruction), Box::new(ram), Box::new(registers)];
        let (proof, challenges, _) = BatchedSumcheck::prove(
            instances
                .iter_mut()
                .map(|instance| &mut **instance as _)
                .collect(),
            &mut opening_accumulator,
            &mut transcript,
        );
        Ok(proof.compressed_polys.len() + challenges.len())
    }

    pub fn run_modular_sumcheck(
        &self,
    ) -> HarnessResult<Stage5RegularBatchFrontierProof<Fr, Bn254G1>> {
        let witness_config =
            JoltVmWitnessConfig::new(self.config.log_t, self.checked.ram_K, self.one_hot_config)
                .include_trusted_advice(!self.trace.device.trusted_advice.is_empty())
                .include_untrusted_advice(!self.trace.device.untrusted_advice.is_empty());
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&self.program, &self.preprocessing, self.trace.clone()),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let mut transcript = self.transcript.clone();
        let input = Stage5ProverInput::new(
            self.config,
            &self.checked,
            &self.stage2,
            &self.stage4,
            &witness,
        );
        prove_stage5_regular_batch_sumcheck_for_frontier(
            &input,
            &mut backend,
            &self.prefix,
            &mut transcript,
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "prove modular Stage 5 regular-batch sumcheck",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage6RegularBatchInputKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage6ProverConfig,
    pub stage1: Stage1ClearOutput<Fr>,
    pub stage2: Stage2ClearOutput<Fr>,
    pub stage3: Stage3ClearOutput<Fr>,
    pub stage4: Stage4ClearOutput<Fr>,
    pub stage5: Stage5ClearOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub expected: Stage6RegularBatchPrefixOutput<Fr>,
}

impl Stage6RegularBatchInputKernelBenchmarkFixture {
    pub fn run_reference_prefix(&self) -> HarnessResult<Stage6RegularBatchPrefixOutput<Fr>> {
        reference_stage6_regular_batch_prefix(
            &self.request,
            self.config.clone(),
            &self.stage1,
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
            &mut self.transcript.clone(),
        )
    }

    pub fn run_modular_prefix(&self) -> HarnessResult<Stage6RegularBatchPrefixOutput<Fr>> {
        derive_stage6_regular_batch_prefix(
            self.config.clone(),
            &self.stage1,
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
            &mut self.transcript.clone(),
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "derive modular Stage 6 regular-batch input claims",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage6RegularBatchSumcheckExpected {
    pub proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    pub challenges: Vec<Fr>,
    pub batching_coefficients: Vec<Fr>,
    pub output_claim: Fr,
}

#[derive(Clone)]
pub struct Stage6RegularBatchSumcheckKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage6ProverConfig,
    pub checked: jolt_verifier::CheckedInputs,
    pub one_hot_config: JoltOneHotConfig,
    pub stage1: Stage1ClearOutput<Fr>,
    pub stage2: Stage2ClearOutput<Fr>,
    pub stage3: Stage3ClearOutput<Fr>,
    pub stage4: Stage4ClearOutput<Fr>,
    pub stage5: Stage5ClearOutput<Fr>,
    pub prefix: Stage6RegularBatchPrefixOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub expected: Stage6RegularBatchSumcheckExpected,
    core: Stage2RegularBatchCoreBenchmarkContext,
}

impl Stage6RegularBatchSumcheckKernelBenchmarkFixture {
    pub fn run_reference_sumcheck(&self) -> HarnessResult<usize> {
        use jolt_core::{
            poly::opening_proof::{OpeningId, ProverOpeningAccumulator, SumcheckId},
            subprotocols::{sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
            transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _},
            zkvm::{
                bytecode::read_raf_checking::{
                    BytecodeReadRafSumcheckParams, BytecodeReadRafSumcheckProver,
                },
                claim_reductions::increments::{
                    IncClaimReductionSumcheckParams, IncClaimReductionSumcheckProver,
                },
                instruction_lookups::ra_virtual::{
                    InstructionRaSumcheckParams, InstructionRaSumcheckProver,
                },
                ram::{
                    hamming_booleanity::{
                        HammingBooleanitySumcheckParams, HammingBooleanitySumcheckProver,
                    },
                    ra_virtual::{RamRaVirtualParams, RamRaVirtualSumcheckProver},
                },
            },
        };

        let mut opening_accumulator = ProverOpeningAccumulator::<CoreField>::new(self.config.log_t);
        *opening_accumulator.evaluation_openings_mut() = self
            .core
            .openings
            .iter()
            .filter(|(id, _)| {
                let sumcheck_id = match id {
                    OpeningId::Polynomial(_, sumcheck_id)
                    | OpeningId::UntrustedAdvice(sumcheck_id)
                    | OpeningId::TrustedAdvice(sumcheck_id) => sumcheck_id,
                };
                !matches!(
                    sumcheck_id,
                    SumcheckId::BytecodeReadRaf
                        | SumcheckId::Booleanity
                        | SumcheckId::RamHammingBooleanity
                        | SumcheckId::RamRaVirtualization
                        | SumcheckId::InstructionRaVirtualization
                        | SumcheckId::AdviceClaimReductionCyclePhase
                        | SumcheckId::IncClaimReduction
                        | SumcheckId::HammingWeightClaimReduction
                        | SumcheckId::AdviceClaimReduction
                )
            })
            .map(|(id, opening)| (*id, opening.clone()))
            .collect();
        seed_core_stage6_openings(
            &mut opening_accumulator,
            &self.stage1,
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
        );

        let mut transcript = CoreBlake2bTranscript::new(b"Jolt");
        let trace_len = self.core.trace.len();
        let log_t = trace_len.ilog2() as usize;
        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.core.bytecode,
            log_t,
            &self.core.one_hot_params,
            &opening_accumulator,
            &mut transcript,
        );
        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&opening_accumulator);
        let booleanity_params = jolt_core::subprotocols::booleanity::BooleanitySumcheckParams::new(
            log_t,
            &self.core.one_hot_params,
            &opening_accumulator,
            &mut transcript,
        );
        let ram_ra_virtual_params =
            RamRaVirtualParams::new(trace_len, &self.core.one_hot_params, &opening_accumulator);
        let instruction_ra_virtual_params = InstructionRaSumcheckParams::new(
            &self.core.one_hot_params,
            &opening_accumulator,
            &mut transcript,
        );
        let inc_reduction_params =
            IncClaimReductionSumcheckParams::new(trace_len, &opening_accumulator, &mut transcript);

        let bytecode_read_raf = BytecodeReadRafSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.core.trace),
            Arc::clone(&self.core.bytecode),
        );
        let booleanity = jolt_core::subprotocols::booleanity::BooleanitySumcheckProver::initialize(
            booleanity_params,
            &self.core.trace,
            &self.core.bytecode,
            &self.core.program_io.memory_layout,
        );
        let ram_hamming_booleanity = HammingBooleanitySumcheckProver::initialize(
            ram_hamming_booleanity_params,
            &self.core.trace,
        );
        let ram_ra_virtual = RamRaVirtualSumcheckProver::initialize(
            ram_ra_virtual_params,
            &self.core.trace,
            &self.core.program_io.memory_layout,
            &self.core.one_hot_params,
        );
        let instruction_ra_virtual = InstructionRaSumcheckProver::initialize(
            instruction_ra_virtual_params,
            &self.core.trace,
        );
        let inc_reduction = IncClaimReductionSumcheckProver::initialize(
            inc_reduction_params,
            self.core.trace.clone(),
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(bytecode_read_raf),
            Box::new(booleanity),
            Box::new(ram_hamming_booleanity),
            Box::new(ram_ra_virtual),
            Box::new(instruction_ra_virtual),
            Box::new(inc_reduction),
        ];
        let (proof, challenges, _) = BatchedSumcheck::prove(
            instances
                .iter_mut()
                .map(|instance| &mut **instance as _)
                .collect(),
            &mut opening_accumulator,
            &mut transcript,
        );
        Ok(proof.compressed_polys.len() + challenges.len())
    }

    pub fn run_modular_sumcheck(
        &self,
    ) -> HarnessResult<Stage6RegularBatchProofOutput<Fr, Bn254G1>> {
        let witness_config =
            JoltVmWitnessConfig::new(self.config.log_t, self.checked.ram_K, self.one_hot_config)
                .include_trusted_advice(!self.trace.device.trusted_advice.is_empty())
                .include_untrusted_advice(!self.trace.device.untrusted_advice.is_empty());
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&self.program, &self.preprocessing, self.trace.clone()),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let mut transcript = self.transcript.clone();
        prove_stage6_transparent_sumchecks(
            self.config.clone(),
            &witness,
            None,
            &mut backend,
            &self.stage1,
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
            &self.prefix,
            &mut transcript,
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "prove modular Stage 6 regular-batch sumcheck",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage7RegularBatchInputKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage7ProverConfig,
    pub stage6: Stage6ClearOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub expected: Stage7RegularBatchPrefixOutput<Fr>,
}

impl Stage7RegularBatchInputKernelBenchmarkFixture {
    pub fn run_reference_prefix(&self) -> Stage7RegularBatchPrefixOutput<Fr> {
        reference_stage7_regular_batch_prefix(
            &self.config,
            &self.stage6,
            &mut self.transcript.clone(),
        )
    }

    pub fn run_modular_prefix(&self) -> HarnessResult<Stage7RegularBatchPrefixOutput<Fr>> {
        derive_stage7_regular_batch_prefix(&self.config, &self.stage6, &mut self.transcript.clone())
            .map_err(|error| {
                core_fixture_error(
                    &self.request,
                    "derive modular Stage 7 regular-batch input claims",
                    error,
                )
            })
    }
}

#[derive(Clone)]
pub struct Stage7RegularBatchSumcheckExpected {
    pub proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    pub challenges: Vec<Fr>,
    pub batching_coefficients: Vec<Fr>,
    pub output_claim: Fr,
}

#[derive(Clone)]
pub struct Stage7RegularBatchSumcheckKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage7ProverConfig,
    pub checked: jolt_verifier::CheckedInputs,
    pub one_hot_config: JoltOneHotConfig,
    pub stage4: Stage4ClearOutput<Fr>,
    pub stage6: Stage6ClearOutput<Fr>,
    pub prefix: Stage7RegularBatchPrefixOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub expected: Stage7RegularBatchSumcheckExpected,
    core: Stage7RegularBatchCoreBenchmarkContext,
}

impl Stage7RegularBatchSumcheckKernelBenchmarkFixture {
    pub fn run_reference_sumcheck(&self) -> HarnessResult<usize> {
        use jolt_core::{
            poly::opening_proof::{OpeningId, ProverOpeningAccumulator, SumcheckId},
            subprotocols::{sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
            transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _},
            zkvm::claim_reductions::{
                AdviceKind, HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
            },
        };

        let mut opening_accumulator = ProverOpeningAccumulator::<CoreField>::new(self.config.log_t);
        *opening_accumulator.evaluation_openings_mut() = self
            .core
            .openings
            .iter()
            .filter(|(id, _)| {
                let sumcheck_id = match id {
                    OpeningId::Polynomial(_, sumcheck_id)
                    | OpeningId::UntrustedAdvice(sumcheck_id)
                    | OpeningId::TrustedAdvice(sumcheck_id) => sumcheck_id,
                };
                !matches!(
                    sumcheck_id,
                    SumcheckId::HammingWeightClaimReduction | SumcheckId::AdviceClaimReduction
                )
            })
            .map(|(id, opening)| (*id, opening.clone()))
            .collect();

        let mut transcript = CoreBlake2bTranscript::new(b"Jolt");
        let params = HammingWeightClaimReductionParams::new(
            &self.core.one_hot_params,
            &opening_accumulator,
            &mut transcript,
        );
        let hamming = HammingWeightClaimReductionProver::initialize(
            params,
            &self.core.trace,
            &self.core.preprocessing,
            &self.core.one_hot_params,
        );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![Box::new(hamming)];
        if let Some(advice) =
            self.reference_advice_address_prover(AdviceKind::Trusted, &opening_accumulator)?
        {
            instances.push(Box::new(advice));
        }
        if let Some(advice) =
            self.reference_advice_address_prover(AdviceKind::Untrusted, &opening_accumulator)?
        {
            instances.push(Box::new(advice));
        }
        let (proof, challenges, _) = BatchedSumcheck::prove(
            instances
                .iter_mut()
                .map(|instance| &mut **instance as _)
                .collect(),
            &mut opening_accumulator,
            &mut transcript,
        );
        Ok(proof.compressed_polys.len() + challenges.len())
    }

    fn reference_advice_address_prover(
        &self,
        kind: jolt_core::zkvm::claim_reductions::AdviceKind,
        opening_accumulator: &jolt_core::poly::opening_proof::ProverOpeningAccumulator<CoreField>,
    ) -> HarnessResult<
        Option<jolt_core::zkvm::claim_reductions::AdviceClaimReductionProver<CoreField>>,
    > {
        use jolt_core::{
            poly::multilinear_polynomial::MultilinearPolynomial,
            subprotocols::sumcheck_prover::SumcheckInstanceProver,
            zkvm::claim_reductions::{
                AdviceClaimReductionParams, AdviceClaimReductionProver, AdviceKind, ReductionPhase,
            },
        };

        let (cycle_phase, bytes, max_bytes) = match kind {
            AdviceKind::Trusted => (
                self.stage6.batch.trusted_advice_cycle_phase.as_ref(),
                self.trace.device.trusted_advice.as_slice(),
                self.preprocessing.memory_layout.max_trusted_advice_size as usize,
            ),
            AdviceKind::Untrusted => (
                self.stage6.batch.untrusted_advice_cycle_phase.as_ref(),
                self.trace.device.untrusted_advice.as_slice(),
                self.preprocessing.memory_layout.max_untrusted_advice_size as usize,
            ),
        };
        let Some(cycle_phase) = cycle_phase else {
            return Ok(None);
        };
        let trace_len = 1usize << self.config.log_t;
        let params = AdviceClaimReductionParams::new(
            kind,
            &self.core.preprocessing.memory_layout,
            trace_len,
            opening_accumulator,
        );
        if params.num_address_phase_rounds() == 0 {
            return Ok(None);
        }
        let words = advice_words_for_core(bytes, max_bytes);
        let mut prover =
            AdviceClaimReductionProver::initialize(params, MultilinearPolynomial::from(words));
        for (round, challenge) in cycle_phase.sumcheck_point.iter().copied().enumerate() {
            <AdviceClaimReductionProver<CoreField> as SumcheckInstanceProver<
                CoreField,
                jolt_core::transcripts::Blake2bTranscript,
            >>::ingest_challenge(&mut prover, core_challenge_from_fr(challenge), round);
        }
        prover.params.phase = ReductionPhase::AddressVariables;
        Ok(Some(prover))
    }

    pub fn run_modular_sumcheck(
        &self,
    ) -> HarnessResult<Stage7ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>> {
        let witness_config =
            JoltVmWitnessConfig::new(self.config.log_t, self.checked.ram_K, self.one_hot_config)
                .include_trusted_advice(!self.trace.device.trusted_advice.is_empty())
                .include_untrusted_advice(!self.trace.device.untrusted_advice.is_empty());
        let trace = TraceOutput::new(self.trace.trace.clone(), self.trace.device.clone(), None);
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&self.program, &self.preprocessing, trace),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let mut transcript = self.transcript.clone();
        let input = Stage7ProverInput::new(
            &self.config,
            &self.checked,
            &self.stage4,
            &self.stage6,
            &witness,
        );
        prove_stage7::<Fr, _, _, _, Bn254G1>(input, &mut backend, &mut transcript).map_err(
            |error| {
                core_fixture_error(
                    &self.request,
                    "prove modular Stage 7 regular-batch sumcheck",
                    error,
                )
            },
        )
    }
}

fn advice_words_for_core(bytes: &[u8], max_bytes: usize) -> Vec<u64> {
    let rows = (max_bytes / 8).next_power_of_two().max(1);
    (0..rows)
        .map(|word_index| advice_word_le_for_core(bytes, word_index))
        .collect()
}

fn advice_word_le_for_core(bytes: &[u8], word_index: usize) -> u64 {
    let Some(start) = word_index.checked_mul(8) else {
        return 0;
    };
    if start >= bytes.len() {
        return 0;
    }
    let end = start.saturating_add(8).min(bytes.len());
    let mut word = [0_u8; 8];
    word[..end - start].copy_from_slice(&bytes[start..end]);
    u64::from_le_bytes(word)
}

fn core_challenge_from_fr(challenge: Fr) -> <CoreField as jolt_core::field::JoltField>::Challenge {
    let limbs = challenge.inner_limbs().0;
    <CoreField as jolt_core::field::JoltField>::Challenge::from(
        (limbs[2] as u128) | ((limbs[3] as u128) << 64),
    )
}

#[derive(Clone)]
pub struct Stage2RegularBatchSumcheckExpected {
    pub proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    pub challenges: Vec<Fr>,
    pub batching_coefficients: Vec<Fr>,
    pub output_claim: Fr,
}

#[derive(Clone)]
pub struct Stage2RegularBatchSumcheckKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage2BatchProverConfig,
    pub checked: jolt_verifier::CheckedInputs,
    pub one_hot_config: JoltOneHotConfig,
    pub stage1: Stage1ClearOutput<Fr>,
    pub product_uniskip: Stage2ProductUniSkipOutput<Fr, Bn254G1>,
    pub prefix: Stage2RegularBatchPrefixOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub expected: Stage2RegularBatchSumcheckExpected,
    core: Stage2RegularBatchCoreBenchmarkContext,
}

impl Stage2RegularBatchSumcheckKernelBenchmarkFixture {
    pub fn run_reference_sumcheck(&self) -> HarnessResult<usize> {
        use jolt_core::{
            poly::opening_proof::{
                OpeningAccumulator, OpeningId, PolynomialId, ProverOpeningAccumulator, SumcheckId,
            },
            subprotocols::{sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
            transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _},
            zkvm::{
                claim_reductions::instruction_lookups::{
                    InstructionLookupsClaimReductionSumcheckParams,
                    InstructionLookupsClaimReductionSumcheckProver,
                },
                ram::{
                    output_check::{OutputSumcheckParams, OutputSumcheckProver},
                    raf_evaluation::{RafEvaluationSumcheckParams, RafEvaluationSumcheckProver},
                    read_write_checking::{RamReadWriteCheckingParams, RamReadWriteCheckingProver},
                },
                spartan::product::{
                    ProductVirtualRemainderParams, ProductVirtualRemainderProver,
                    ProductVirtualUniSkipParams,
                },
                witness::VirtualPolynomial,
            },
        };

        let mut opening_accumulator = ProverOpeningAccumulator::<CoreField>::new(self.config.log_t);
        *opening_accumulator.evaluation_openings_mut() = self
            .core
            .openings
            .iter()
            .filter(|(id, _)| {
                matches!(
                    id,
                    OpeningId::Polynomial(_, SumcheckId::SpartanOuter)
                        | OpeningId::Polynomial(
                            PolynomialId::Virtual(VirtualPolynomial::UnivariateSkip),
                            SumcheckId::SpartanProductVirtualization
                        )
                )
            })
            .map(|(id, opening)| (*id, opening.clone()))
            .collect();

        let mut transcript = CoreBlake2bTranscript::new(b"Jolt");
        let (mut tau, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);
        tau.r
            .push(<CoreField as jolt_core::field::JoltField>::Challenge::from(
                17u128,
            ));
        let uni_skip_params = ProductVirtualUniSkipParams::<CoreField> {
            tau: tau.r,
            base_evals: [<CoreField as jolt_core::field::JoltField>::from_u64(0); 3],
        };
        let ram_read_write_checking_params = RamReadWriteCheckingParams::new(
            &opening_accumulator,
            &mut transcript,
            &self.core.one_hot_params,
            self.core.trace.len(),
            &self.core.rw_config,
        );
        let spartan_product_virtual_remainder_params = ProductVirtualRemainderParams::new(
            self.core.trace.len(),
            uni_skip_params,
            &opening_accumulator,
        );
        let instruction_claim_reduction_params =
            InstructionLookupsClaimReductionSumcheckParams::new(
                self.core.trace.len(),
                &opening_accumulator,
                &mut transcript,
            );
        let ram_raf_evaluation_params = RafEvaluationSumcheckParams::new(
            &self.core.program_io.memory_layout,
            &self.core.one_hot_params,
            &opening_accumulator,
            self.core.trace.len(),
            &self.core.rw_config,
        );
        let ram_output_check_params = OutputSumcheckParams::new(
            self.core.one_hot_params.ram_k,
            &self.core.program_io,
            &mut transcript,
            self.core.trace.len(),
            &self.core.rw_config,
        );

        let ram_read_write_checking = RamReadWriteCheckingProver::initialize(
            ram_read_write_checking_params,
            &self.core.trace,
            &self.core.bytecode,
            &self.core.program_io.memory_layout,
            &self.core.initial_ram_state,
        );
        let spartan_product_virtual_remainder = ProductVirtualRemainderProver::initialize(
            spartan_product_virtual_remainder_params,
            Arc::clone(&self.core.trace),
        );
        let instruction_claim_reduction =
            InstructionLookupsClaimReductionSumcheckProver::initialize(
                instruction_claim_reduction_params,
                Arc::clone(&self.core.trace),
            );
        let ram_raf_evaluation = RafEvaluationSumcheckProver::initialize(
            ram_raf_evaluation_params,
            &self.core.trace,
            &self.core.program_io.memory_layout,
        );
        let ram_output_check = OutputSumcheckProver::initialize(
            ram_output_check_params,
            &self.core.initial_ram_state,
            &self.core.final_ram_state,
            &self.core.program_io.memory_layout,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(ram_read_write_checking),
            Box::new(spartan_product_virtual_remainder),
            Box::new(instruction_claim_reduction),
            Box::new(ram_raf_evaluation),
            Box::new(ram_output_check),
        ];
        let (proof, challenges, _) = BatchedSumcheck::prove(
            instances
                .iter_mut()
                .map(|instance| &mut **instance as _)
                .collect(),
            &mut opening_accumulator,
            &mut transcript,
        );
        Ok(proof.compressed_polys.len() + challenges.len())
    }

    pub fn run_modular_sumcheck(
        &self,
    ) -> HarnessResult<Stage2RegularBatchFrontierProof<Fr, Bn254G1>> {
        let witness_config = JoltVmWitnessConfig::new(
            self.config.log_t,
            1usize << self.config.log_k,
            self.one_hot_config,
        )
        .include_trusted_advice(!self.trace.device.trusted_advice.is_empty())
        .include_untrusted_advice(!self.trace.device.untrusted_advice.is_empty());
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&self.program, &self.preprocessing, self.trace.clone()),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let mut transcript = self.transcript.clone();
        let input = Stage2ProverInput::new(self.config, &self.checked, &self.stage1, &witness);
        prove_stage2_regular_batch_sumcheck_for_frontier(
            &input,
            &mut backend,
            &self.product_uniskip,
            &self.prefix,
            &mut transcript,
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "prove modular Stage 2 regular-batch sumcheck",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage3RegularBatchSumcheckExpected {
    pub proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    pub challenges: Vec<Fr>,
    pub batching_coefficients: Vec<Fr>,
    pub output_claim: Fr,
}

#[derive(Clone)]
pub struct Stage3RegularBatchSumcheckKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage3ProverConfig,
    pub checked: jolt_verifier::CheckedInputs,
    pub one_hot_config: JoltOneHotConfig,
    pub stage1: Stage1ClearOutput<Fr>,
    pub stage2: Stage2ClearOutput<Fr>,
    pub prefix: Stage3RegularBatchPrefixOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub expected: Stage3RegularBatchSumcheckExpected,
    core: Stage2RegularBatchCoreBenchmarkContext,
}

impl Stage3RegularBatchSumcheckKernelBenchmarkFixture {
    pub fn run_reference_sumcheck(&self) -> HarnessResult<usize> {
        use jolt_core::{
            poly::opening_proof::{
                OpeningId, OpeningPoint, PolynomialId, ProverOpeningAccumulator, SumcheckId,
            },
            subprotocols::{sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
            transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _},
            zkvm::{
                claim_reductions::registers::{
                    RegistersClaimReductionSumcheckParams, RegistersClaimReductionSumcheckProver,
                },
                spartan::{
                    instruction_input::{InstructionInputParams, InstructionInputSumcheckProver},
                    shift::{ShiftSumcheckParams, ShiftSumcheckProver},
                },
                witness::VirtualPolynomial,
            },
        };

        let mut opening_accumulator = ProverOpeningAccumulator::<CoreField>::new(self.config.log_t);
        *opening_accumulator.evaluation_openings_mut() = self
            .core
            .openings
            .iter()
            .filter(|(id, _)| {
                matches!(
                    id,
                    OpeningId::Polynomial(
                        PolynomialId::Virtual(
                            VirtualPolynomial::NextPC
                                | VirtualPolynomial::NextUnexpandedPC
                                | VirtualPolynomial::NextIsVirtual
                                | VirtualPolynomial::NextIsFirstInSequence
                                | VirtualPolynomial::RdWriteValue
                                | VirtualPolynomial::Rs1Value
                                | VirtualPolynomial::Rs2Value
                                | VirtualPolynomial::LookupOutput,
                        ),
                        SumcheckId::SpartanOuter,
                    ) | OpeningId::Polynomial(
                        PolynomialId::Virtual(
                            VirtualPolynomial::NextIsNoop
                                | VirtualPolynomial::LeftInstructionInput
                                | VirtualPolynomial::RightInstructionInput,
                        ),
                        SumcheckId::SpartanProductVirtualization,
                    )
                )
            })
            .map(|(id, opening)| (*id, opening.clone()))
            .collect();
        let product_remainder_point = OpeningPoint::new(
            self.stage2
                .batch
                .product_remainder
                .opening_point
                .iter()
                .map(|challenge| {
                    let limbs = challenge.inner_limbs().0;
                    <CoreField as jolt_core::field::JoltField>::Challenge::from(
                        (limbs[2] as u128) | ((limbs[3] as u128) << 64),
                    )
                })
                .collect(),
        );
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::InstructionClaimReduction,
            ))
            .or_insert((
                product_remainder_point.clone(),
                self.stage2
                    .output_claims
                    .product_remainder
                    .left_instruction_input
                    .into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::InstructionClaimReduction,
            ))
            .or_insert((
                product_remainder_point,
                self.stage2
                    .output_claims
                    .product_remainder
                    .right_instruction_input
                    .into(),
            ));

        let mut transcript = CoreBlake2bTranscript::new(b"Jolt");
        let shift_params =
            ShiftSumcheckParams::new(self.config.log_t, &opening_accumulator, &mut transcript);
        let instruction_input_params =
            InstructionInputParams::new(&opening_accumulator, &mut transcript);
        let registers_claim_reduction_params = RegistersClaimReductionSumcheckParams::new(
            self.core.trace.len(),
            &opening_accumulator,
            &mut transcript,
        );

        let shift = ShiftSumcheckProver::initialize(
            shift_params,
            Arc::clone(&self.core.trace),
            &self.core.bytecode,
        );
        let instruction_input = InstructionInputSumcheckProver::initialize(
            instruction_input_params,
            &self.core.trace,
            &opening_accumulator,
        );
        let registers_claim_reduction = RegistersClaimReductionSumcheckProver::initialize(
            registers_claim_reduction_params,
            Arc::clone(&self.core.trace),
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(shift),
            Box::new(instruction_input),
            Box::new(registers_claim_reduction),
        ];
        let (proof, challenges, _) = BatchedSumcheck::prove(
            instances
                .iter_mut()
                .map(|instance| &mut **instance as _)
                .collect(),
            &mut opening_accumulator,
            &mut transcript,
        );
        Ok(proof.compressed_polys.len() + challenges.len())
    }

    pub fn run_modular_sumcheck(
        &self,
    ) -> HarnessResult<Stage3RegularBatchFrontierProof<Fr, Bn254G1>> {
        let witness_config =
            JoltVmWitnessConfig::new(self.config.log_t, self.checked.ram_K, self.one_hot_config)
                .include_trusted_advice(!self.trace.device.trusted_advice.is_empty())
                .include_untrusted_advice(!self.trace.device.untrusted_advice.is_empty());
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&self.program, &self.preprocessing, self.trace.clone()),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let mut transcript = self.transcript.clone();
        let input = Stage3ProverInput::new(
            self.config,
            &self.checked,
            &self.stage1,
            &self.stage2,
            &witness,
        );
        prove_stage3_regular_batch_sumcheck_for_frontier(
            &input,
            &mut backend,
            &self.prefix,
            &mut transcript,
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "prove modular Stage 3 regular-batch sumcheck",
                error,
            )
        })
    }
}

#[derive(Clone)]
pub struct Stage4RegularBatchSumcheckExpected {
    pub proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    pub challenges: Vec<Fr>,
    pub batching_coefficients: Vec<Fr>,
    pub output_claim: Fr,
}

#[derive(Clone)]
pub struct Stage4RegularBatchSumcheckKernelBenchmarkFixture {
    pub request: FixtureRequest,
    pub config: Stage4ProverConfig,
    pub checked: jolt_verifier::CheckedInputs,
    pub one_hot_config: JoltOneHotConfig,
    pub stage2: Stage2ClearOutput<Fr>,
    pub stage3: Stage3ClearOutput<Fr>,
    pub ram_val_check_init: Stage4RamValCheckInitialEvaluation<Fr>,
    pub prefix: Stage4RegularBatchPrefixOutput<Fr>,
    pub transcript: Blake2bTranscript<Fr>,
    pub program: JoltProgram,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<OwnedTrace>,
    pub expected: Stage4RegularBatchSumcheckExpected,
    core: Stage2RegularBatchCoreBenchmarkContext,
}

impl Stage4RegularBatchSumcheckKernelBenchmarkFixture {
    pub fn run_reference_sumcheck(&self) -> HarnessResult<usize> {
        use jolt_core::{
            poly::opening_proof::{OpeningId, ProverOpeningAccumulator, SumcheckId},
            subprotocols::{sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
            transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _},
            zkvm::{
                ram::val_check::{RamValCheckSumcheckParams, RamValCheckSumcheckProver},
                registers::read_write_checking::{
                    RegistersReadWriteCheckingParams, RegistersReadWriteCheckingProver,
                },
                witness::VirtualPolynomial,
            },
        };

        let mut opening_accumulator = ProverOpeningAccumulator::<CoreField>::new(self.config.log_t);
        let registers_point =
            core_opening_point_from_fr(&self.stage3.batch.registers_claim_reduction.opening_point);
        let instruction_point =
            core_opening_point_from_fr(&self.stage3.batch.instruction_input.opening_point);
        let ram_read_write_point =
            core_opening_point_from_fr(&self.stage2.batch.ram_read_write.opening_point);
        let ram_output_check_point =
            core_opening_point_from_fr(&self.stage2.batch.ram_output_check.opening_point);
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::RdWriteValue,
                SumcheckId::RegistersClaimReduction,
            ))
            .or_insert((
                registers_point.clone(),
                self.stage3
                    .output_claims
                    .registers_claim_reduction
                    .rd_write_value
                    .into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::Rs1Value,
                SumcheckId::RegistersClaimReduction,
            ))
            .or_insert((
                registers_point.clone(),
                self.stage3
                    .output_claims
                    .registers_claim_reduction
                    .rs1_value
                    .into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::Rs2Value,
                SumcheckId::RegistersClaimReduction,
            ))
            .or_insert((
                registers_point,
                self.stage3
                    .output_claims
                    .registers_claim_reduction
                    .rs2_value
                    .into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::Rs1Value,
                SumcheckId::InstructionInputVirtualization,
            ))
            .or_insert((
                instruction_point.clone(),
                self.stage3.output_claims.instruction_input.rs1_value.into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::Rs2Value,
                SumcheckId::InstructionInputVirtualization,
            ))
            .or_insert((
                instruction_point,
                self.stage3.output_claims.instruction_input.rs2_value.into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            ))
            .or_insert((
                ram_read_write_point,
                self.stage2.output_claims.ram_read_write.val.into(),
            ));
        let _ = opening_accumulator
            .openings
            .entry(OpeningId::virt(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            ))
            .or_insert((
                ram_output_check_point,
                self.stage2.output_claims.ram_output_check.into(),
            ));

        let mut transcript = CoreBlake2bTranscript::new(b"Jolt");
        let registers_params = RegistersReadWriteCheckingParams::new(
            self.core.trace.len(),
            &opening_accumulator,
            &mut transcript,
            &self.core.rw_config,
        );
        transcript.append_bytes(b"ram_val_check_gamma", &[]);
        let ram_val_check_gamma = transcript.challenge_scalar::<CoreField>();
        let ram_params = RamValCheckSumcheckParams::new_from_prover(
            &self.core.one_hot_params,
            &opening_accumulator,
            &self.core.initial_ram_state,
            self.core.trace.len(),
            ram_val_check_gamma,
            &self.core.ram_preprocessing,
            &self.core.program_io,
        );

        let registers = RegistersReadWriteCheckingProver::initialize(
            registers_params,
            Arc::clone(&self.core.trace),
            &self.core.bytecode,
            &self.core.program_io.memory_layout,
        );
        let ram_val_check = RamValCheckSumcheckProver::initialize(
            ram_params,
            &self.core.trace,
            &self.core.bytecode,
            &self.core.program_io.memory_layout,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(registers), Box::new(ram_val_check)];
        let (proof, challenges, _) = BatchedSumcheck::prove(
            instances
                .iter_mut()
                .map(|instance| &mut **instance as _)
                .collect(),
            &mut opening_accumulator,
            &mut transcript,
        );
        Ok(proof.compressed_polys.len() + challenges.len())
    }

    pub fn run_modular_sumcheck(
        &self,
    ) -> HarnessResult<Stage4RegularBatchFrontierProof<Fr, Bn254G1>> {
        let witness_config =
            JoltVmWitnessConfig::new(self.config.log_t, self.checked.ram_K, self.one_hot_config)
                .include_trusted_advice(!self.trace.device.trusted_advice.is_empty())
                .include_untrusted_advice(!self.trace.device.untrusted_advice.is_empty());
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&self.program, &self.preprocessing, self.trace.clone()),
        );
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1024,
        });
        let mut transcript = self.transcript.clone();
        let input = Stage4ProverInput::new(
            self.config,
            &self.checked,
            &self.stage2,
            &self.stage3,
            self.ram_val_check_init.clone(),
            &witness,
        );
        prove_stage4_regular_batch_sumcheck_for_frontier(
            &input,
            &mut backend,
            &self.prefix,
            &mut transcript,
        )
        .map_err(|error| {
            core_fixture_error(
                &self.request,
                "prove modular Stage 4 regular-batch sumcheck",
                error,
            )
        })
    }
}

fn core_opening_point_from_fr(
    point: &[Fr],
) -> core_opening::OpeningPoint<{ core_opening::BIG_ENDIAN }, CoreField> {
    core_opening::OpeningPoint::new(
        point
            .iter()
            .map(|challenge| {
                let limbs = challenge.inner_limbs().0;
                <CoreField as jolt_core::field::JoltField>::Challenge::from(
                    (limbs[2] as u128) | ((limbs[3] as u128) << 64),
                )
            })
            .collect(),
    )
}

fn seed_core_opening(
    opening_accumulator: &mut core_opening::ProverOpeningAccumulator<CoreField>,
    id: core_opening::OpeningId,
    point: &[Fr],
    claim: Fr,
) {
    let _ = opening_accumulator
        .openings
        .entry(id)
        .or_insert((core_opening_point_from_fr(point), claim.into()));
}

fn seed_core_stage5_openings(
    opening_accumulator: &mut core_opening::ProverOpeningAccumulator<CoreField>,
    stage2: &Stage2ClearOutput<Fr>,
    stage4: &Stage4ClearOutput<Fr>,
) {
    use core_opening::{OpeningId, SumcheckId};
    use core_witness::VirtualPolynomial;

    let instruction_claim_point =
        core_opening_point_from_fr(&stage2.batch.instruction_claim_reduction.opening_point);
    let product_point = core_opening_point_from_fr(&stage2.batch.product_remainder.opening_point);
    let ram_raf_point = core_opening_point_from_fr(&stage2.batch.ram_raf_evaluation.opening_point);
    let ram_read_write_point =
        core_opening_point_from_fr(&stage2.batch.ram_read_write.opening_point);
    let ram_val_check_point = core_opening_point_from_fr(&stage4.batch.ram_val_check.opening_point);
    let registers_point =
        core_opening_point_from_fr(&stage4.batch.registers_read_write.opening_point);

    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output;
    let instruction_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);

    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        ),
        (
            instruction_claim_point.clone(),
            instruction_lookup_output.into(),
        ),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
        ),
        (product_point, product_lookup_output.into()),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
        ),
        (
            instruction_claim_point.clone(),
            stage2
                .output_claims
                .instruction_claim_reduction
                .left_lookup_operand
                .into(),
        ),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
        ),
        (
            instruction_claim_point,
            stage2
                .output_claims
                .instruction_claim_reduction
                .right_lookup_operand
                .into(),
        ),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation),
        (
            ram_raf_point,
            stage2.output_claims.ram_raf_evaluation.into(),
        ),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamReadWriteChecking),
        (
            ram_read_write_point,
            stage2.output_claims.ram_read_write.ra.into(),
        ),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamValCheck),
        (
            ram_val_check_point,
            stage4.output_claims.ram_val_check.ram_ra.into(),
        ),
    );
    let _ = opening_accumulator.openings.insert(
        OpeningId::virt(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        ),
        (
            registers_point,
            stage4
                .output_claims
                .registers_read_write
                .registers_val
                .into(),
        ),
    );
}

fn seed_core_stage6_openings(
    opening_accumulator: &mut core_opening::ProverOpeningAccumulator<CoreField>,
    stage1: &Stage1ClearOutput<Fr>,
    stage2: &Stage2ClearOutput<Fr>,
    stage3: &Stage3ClearOutput<Fr>,
    stage4: &Stage4ClearOutput<Fr>,
    stage5: &Stage5ClearOutput<Fr>,
) {
    use core_instruction::{
        CircuitFlags as CoreCircuitFlags, InstructionFlags as CoreInstructionFlags,
    };
    use core_opening::{OpeningId, SumcheckId};
    use core_witness::{CommittedPolynomial, VirtualPolynomial};

    let spartan_outer_point = stage1.public.tau.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
        spartan_outer_point,
        stage1.outer.pc,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanOuter),
        spartan_outer_point,
        stage1.outer.unexpanded_pc,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::Imm, SumcheckId::SpartanOuter),
        spartan_outer_point,
        stage1.outer.imm,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter),
        spartan_outer_point,
        stage1.outer.lookup_output,
    );
    for (flag, claim) in [
        (
            CoreCircuitFlags::AddOperands,
            stage1.outer.flags.add_operands,
        ),
        (
            CoreCircuitFlags::SubtractOperands,
            stage1.outer.flags.subtract_operands,
        ),
        (
            CoreCircuitFlags::MultiplyOperands,
            stage1.outer.flags.multiply_operands,
        ),
        (CoreCircuitFlags::Load, stage1.outer.flags.load),
        (CoreCircuitFlags::Store, stage1.outer.flags.store),
        (CoreCircuitFlags::Jump, stage1.outer.flags.jump),
        (
            CoreCircuitFlags::WriteLookupOutputToRD,
            stage1.outer.flags.write_lookup_output_to_rd,
        ),
        (
            CoreCircuitFlags::VirtualInstruction,
            stage1.outer.flags.virtual_instruction,
        ),
        (CoreCircuitFlags::Assert, stage1.outer.flags.assert),
        (
            CoreCircuitFlags::DoNotUpdateUnexpandedPC,
            stage1.outer.flags.do_not_update_unexpanded_pc,
        ),
        (CoreCircuitFlags::Advice, stage1.outer.flags.advice),
        (
            CoreCircuitFlags::IsCompressed,
            stage1.outer.flags.is_compressed,
        ),
        (
            CoreCircuitFlags::IsFirstInSequence,
            stage1.outer.flags.is_first_in_sequence,
        ),
        (
            CoreCircuitFlags::IsLastInSequence,
            stage1.outer.flags.is_last_in_sequence,
        ),
    ] {
        seed_core_opening(
            opening_accumulator,
            OpeningId::virt(VirtualPolynomial::OpFlags(flag), SumcheckId::SpartanOuter),
            spartan_outer_point,
            claim,
        );
    }

    let product_point = stage2.batch.product_remainder.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::OpFlags(CoreCircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        ),
        product_point,
        stage2.output_claims.product_remainder.jump_flag,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::Branch),
            SumcheckId::SpartanProductVirtualization,
        ),
        product_point,
        stage2.output_claims.product_remainder.branch_flag,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::OpFlags(CoreCircuitFlags::WriteLookupOutputToRD),
            SumcheckId::SpartanProductVirtualization,
        ),
        product_point,
        stage2
            .output_claims
            .product_remainder
            .write_lookup_output_to_rd,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::OpFlags(CoreCircuitFlags::VirtualInstruction),
            SumcheckId::SpartanProductVirtualization,
        ),
        product_point,
        stage2.output_claims.product_remainder.virtual_instruction,
    );

    let ram_read_write_point = stage2.batch.ram_read_write.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::RamVal, SumcheckId::RamReadWriteChecking),
        ram_read_write_point,
        stage2.output_claims.ram_read_write.val,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamReadWriteChecking),
        ram_read_write_point,
        stage2.output_claims.ram_read_write.ra,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::committed(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        ),
        ram_read_write_point,
        stage2.output_claims.ram_read_write.inc,
    );

    let shift_point = stage3.batch.shift.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanShift),
        shift_point,
        stage3.output_claims.shift.pc,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanShift),
        shift_point,
        stage3.output_claims.shift.unexpanded_pc,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::OpFlags(CoreCircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        ),
        shift_point,
        stage3.output_claims.shift.is_virtual,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::OpFlags(CoreCircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        ),
        shift_point,
        stage3.output_claims.shift.is_first_in_sequence,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        ),
        shift_point,
        stage3.output_claims.shift.is_noop,
    );

    let instruction_input_point = stage3.batch.instruction_input.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.left_operand_is_rs1,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.rs1_value,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.left_operand_is_pc,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.unexpanded_pc,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.right_operand_is_rs2,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.rs2_value,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.right_operand_is_imm,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        ),
        instruction_input_point,
        stage3.output_claims.instruction_input.imm,
    );

    let registers_point = stage4.batch.registers_read_write.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        ),
        registers_point,
        stage4.output_claims.registers_read_write.registers_val,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        registers_point,
        stage4.output_claims.registers_read_write.rs1_ra,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        registers_point,
        stage4.output_claims.registers_read_write.rs2_ra,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        ),
        registers_point,
        stage4.output_claims.registers_read_write.rd_wa,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::committed(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        ),
        registers_point,
        stage4.output_claims.registers_read_write.rd_inc,
    );

    let ram_val_check_point = stage4.batch.ram_val_check.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamValCheck),
        ram_val_check_point,
        stage4.output_claims.ram_val_check.ram_ra,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::committed(CommittedPolynomial::RamInc, SumcheckId::RamValCheck),
        ram_val_check_point,
        stage4.output_claims.ram_val_check.ram_inc,
    );

    let instruction_read_raf = &stage5.batch.instruction_read_raf;
    let instruction_claims = &stage5.output_claims.instruction_read_raf;
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        ),
        &instruction_read_raf.instruction_raf_flag_opening_point,
        instruction_claims.instruction_raf_flag,
    );
    for (index, claim) in instruction_claims
        .lookup_table_flags
        .iter()
        .copied()
        .enumerate()
    {
        seed_core_opening(
            opening_accumulator,
            OpeningId::virt(
                VirtualPolynomial::LookupTableFlag(index),
                SumcheckId::InstructionReadRaf,
            ),
            &instruction_read_raf.lookup_table_flag_opening_point,
            claim,
        );
    }
    for (index, (point, claim)) in instruction_read_raf
        .instruction_ra_opening_points
        .iter()
        .zip(instruction_claims.instruction_ra.iter().copied())
        .enumerate()
    {
        seed_core_opening(
            opening_accumulator,
            OpeningId::virt(
                VirtualPolynomial::InstructionRa(index),
                SumcheckId::InstructionReadRaf,
            ),
            point,
            claim,
        );
    }

    let ram_ra_reduction_point = stage5.batch.ram_ra_claim_reduction.opening_point.as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamRaClaimReduction),
        ram_ra_reduction_point,
        stage5.output_claims.ram_ra_claim_reduction.ram_ra,
    );

    let registers_val_point = stage5
        .batch
        .registers_val_evaluation
        .opening_point
        .as_slice();
    seed_core_opening(
        opening_accumulator,
        OpeningId::virt(VirtualPolynomial::RdWa, SumcheckId::RegistersValEvaluation),
        registers_val_point,
        stage5.output_claims.registers_val_evaluation.rd_wa,
    );
    seed_core_opening(
        opening_accumulator,
        OpeningId::committed(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        ),
        registers_val_point,
        stage5.output_claims.registers_val_evaluation.rd_inc,
    );
}

#[derive(Clone)]
struct Stage2RegularBatchCoreBenchmarkContext {
    trace: Arc<Vec<CoreTraceCycle>>,
    bytecode: Arc<jolt_core::zkvm::bytecode::BytecodePreprocessing>,
    ram_preprocessing: jolt_core::zkvm::ram::RAMPreprocessing,
    program_io: JoltDevice,
    initial_ram_state: Vec<u64>,
    final_ram_state: Vec<u64>,
    one_hot_params: jolt_core::zkvm::config::OneHotParams,
    rw_config: jolt_core::zkvm::config::ReadWriteConfig,
    openings: core_opening::Openings<CoreField>,
}

#[derive(Clone)]
struct Stage7RegularBatchCoreBenchmarkContext {
    trace: Arc<Vec<CoreTraceCycle>>,
    preprocessing: JoltSharedPreprocessing,
    one_hot_params: jolt_core::zkvm::config::OneHotParams,
    openings: core_opening::Openings<CoreField>,
}

#[derive(Clone)]
pub struct Stage2ProductRemainderOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: ProductRemainderOutputOpeningClaims<Fr>,
    pub expected: ProductRemainderOutputOpeningClaims<Fr>,
    pub opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage2ProductUniskipKernelBenchmarkFixture {
    pub core_trace: Arc<Vec<CoreTraceCycle>>,
    pub rows: Vec<SumcheckProductUniskipRow>,
    pub log_t: usize,
}

#[derive(Clone)]
pub struct Stage2RamReadWriteOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: RamReadWriteOutputOpeningClaims<Fr>,
    pub expected: RamReadWriteOutputOpeningClaims<Fr>,
    pub opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage2RamTerminalOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage2RamTerminalOutputOpeningClaims<Fr>,
    pub expected: Stage2RamTerminalOutputOpeningClaims<Fr>,
    pub ram_raf_opening_point: Vec<Fr>,
    pub ram_output_check_opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage2InstructionClaimOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: InstructionClaimReductionOutputOpeningClaims<Fr>,
    pub expected: InstructionClaimReductionOutputOpeningClaims<Fr>,
    pub opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage2RegularBatchVerifierReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage2BatchOutputOpeningClaims<Fr>,
    pub verified: Stage2BatchOutputOpeningClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage3RegularBatchInputCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage3RegularBatchPrefixOutput<Fr>,
    pub expected: Stage3RegularBatchInputClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage3OutputOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage3RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage3RegularBatchOutputOpeningClaims<Fr>,
    pub shift_opening_point: Vec<Fr>,
    pub instruction_input_opening_point: Vec<Fr>,
    pub registers_claim_reduction_opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage3RegularBatchVerifierReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage3RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage3RegularBatchOutputOpeningClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage4RegularBatchInputCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage4RegularBatchPrefixOutput<Fr>,
    pub expected: Stage4RegularBatchInputClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage4OutputOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage4RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage4RegularBatchOutputOpeningClaims<Fr>,
    pub registers_read_write_opening_point: Vec<Fr>,
    pub ram_val_check_opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage4RegularBatchVerifierReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage4RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage4RegularBatchOutputOpeningClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage5RegularBatchInputCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage5RegularBatchPrefixOutput<Fr>,
    pub expected: Stage5RegularBatchInputClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage5OutputOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage5RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage5RegularBatchOutputOpeningClaims<Fr>,
    pub instruction_lookup_table_flag_opening_point: Vec<Fr>,
    pub instruction_ra_opening_points: Vec<Vec<Fr>>,
    pub instruction_raf_flag_opening_point: Vec<Fr>,
    pub ram_ra_claim_reduction_opening_point: Vec<Fr>,
    pub registers_val_evaluation_opening_point: Vec<Fr>,
}

#[derive(Clone)]
pub struct Stage5RegularBatchVerifierReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage5RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage5RegularBatchOutputOpeningClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage6RegularBatchInputCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage6RegularBatchPrefixOutput<Fr>,
    pub expected: Stage6RegularBatchInputClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage6OutputOpeningCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage6RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage6RegularBatchOutputOpeningClaims<Fr>,
}

#[derive(Clone)]
pub struct Stage6RegularBatchVerifierReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage6RegularBatchOutputOpeningClaims<Fr>,
    pub expected: Stage6RegularBatchOutputOpeningClaims<Fr>,
}

#[derive(Clone)]
struct Stage3OutputOpeningPoints {
    shift: Vec<Fr>,
    instruction_input: Vec<Fr>,
    registers_claim_reduction: Vec<Fr>,
}

pub fn load_core_verifier_fixture(request: &FixtureRequest) -> HarnessResult<CoreVerifierFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "core verifier fixtures currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    fixture
        .verify()
        .map_err(|error| core_fixture_error(request, "verify converted core fixture", error))?;
    Ok(fixture)
}

pub fn load_stage0_commitment_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<CoreVerifierFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage0 commitment verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let stage0 = prove_stage0_commitments(request, &generated)?;
    let mut fixture = convert_fixture(request, generated)?;
    if let Some(mismatch) =
        stage0_commitment_mismatch(&fixture.proof.commitments, &stage0.commitments)
    {
        return Err(core_fixture_error(
            request,
            "compare stage0 commitment verifier_replay",
            mismatch,
        ));
    }
    if fixture.proof.untrusted_advice_commitment != stage0.untrusted_advice_commitment {
        return Err(core_fixture_error(
            request,
            "compare stage0 commitment verifier_replay",
            "untrusted advice commitment differs",
        ));
    }
    fixture.proof.commitments = stage0.commitments;
    fixture.proof.untrusted_advice_commitment = stage0.untrusted_advice_commitment;
    fixture.trusted_advice_commitment = stage0
        .trusted_advice_commitment
        .or(fixture.trusted_advice_commitment);
    fixture.verify().map_err(|error| {
        core_fixture_error(request, "verify stage0 commitment verifier_replay", error)
    })?;
    Ok(fixture)
}

pub fn load_stage0_commitment_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage0CommitmentKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage0 commitment kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let core_trace = generate_core_trace(request)?;
    if core_trace.len() != generated.proof.trace_length {
        return Err(core_fixture_error(
            request,
            "load stage0 commitment kernel benchmark fixture",
            format!(
                "generated core trace length {} differs from proof trace length {}",
                core_trace.len(),
                generated.proof.trace_length
            ),
        ));
    }
    Ok(Stage0CommitmentKernelBenchmarkFixture {
        core_generators: stage0_core_generators(&generated),
        generated,
        core_trace,
    })
}

pub fn load_stage0_advice_commitment_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage0AdviceCommitmentKernelBenchmarkFixture> {
    if request.kind != FixtureKind::AdviceConsumer
        || request.feature_mode != FeatureMode::Transparent
    {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage0 advice commitment kernel benchmarks currently cover transparent advice fixtures",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let log_t = checked_power_of_two_log(request, generated.proof.trace_length)?;
    let one_hot_config = convert_one_hot_config(generated.proof.one_hot_config.clone());
    let witness_config = JoltVmWitnessConfig::new(log_t, generated.proof.ram_K, one_hot_config)
        .include_trusted_advice(true)
        .include_untrusted_advice(true);
    let preprocessing = convert_program_preprocessing(&generated.core_preprocessing);
    Ok(Stage0AdviceCommitmentKernelBenchmarkFixture {
        core_generators: stage0_core_generators(&generated),
        generated,
        preprocessing,
        witness_config,
    })
}

pub fn load_stage1_spartan_outer_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage1SpartanOuterCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage1 Spartan outer checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let stage1 = prove_stage1_spartan_outer_checkpoint(request, &generated)?;
    let fixture = convert_fixture(request, generated)?;
    let expected = clear_stage1_claims(request, &fixture)?;
    if stage1.claims != *expected {
        return Err(core_fixture_error(
            request,
            "compare stage1 Spartan outer checkpoint",
            "modular Stage 1 claims differ from converted core proof claims",
        ));
    }
    fixture
        .verify()
        .map_err(|error| core_fixture_error(request, "verify stage1 checkpoint fixture", error))?;
    Ok(Stage1SpartanOuterCheckpoint {
        fixture,
        opening_point: stage1.opening_point,
        claim_count: stage1.claim_count,
    })
}

pub fn load_stage1_spartan_outer_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<CoreVerifierFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage1 Spartan outer verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let stage1 = prove_stage1_spartan_outer_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    verifier_replay_stage1_spartan_outer(request, &mut fixture.proof, stage1)?;
    fixture.verify().map_err(|error| {
        core_fixture_error(
            request,
            "verify stage1 Spartan outer verifier_replay",
            error,
        )
    })?;
    Ok(fixture)
}

pub fn load_stage1_spartan_outer_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage1SpartanOuterKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage1 Spartan outer kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let core_trace = generate_core_trace(request)?;
    if core_trace.len() != generated.proof.trace_length {
        return Err(core_fixture_error(
            request,
            "load stage1 Spartan outer kernel benchmark fixture",
            format!(
                "generated core trace length {} differs from proof trace length {}",
                core_trace.len(),
                generated.proof.trace_length
            ),
        ));
    }
    let log_t = checked_power_of_two_log(request, generated.proof.trace_length)?;
    let one_hot_config = convert_one_hot_config(generated.proof.one_hot_config.clone());
    let witness_config = JoltVmWitnessConfig::new(log_t, generated.proof.ram_K, one_hot_config)
        .include_trusted_advice(!generated.trace.device.trusted_advice.is_empty())
        .include_untrusted_advice(!generated.trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(&generated.core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&generated.program, &preprocessing, generated.trace.clone()),
    );
    let config = Stage1ProverConfig::new(log_t);
    let materialization_request =
        build_stage1_r1cs_materialization_request::<Fr, _>(config, &witness).map_err(|error| {
            core_fixture_error(
                request,
                "build stage1 Spartan outer kernel materialization request",
                error,
            )
        })?;
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let materializations = <CpuBackend as SumcheckBackend<
        Fr,
        jolt_witness::protocols::jolt_vm::JoltVmNamespace,
    >>::materialize_sumcheck_views(
        &mut backend,
        &materialization_request.materializations,
        &witness,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "materialize stage1 Spartan outer kernel R1CS inputs",
            error,
        )
    })?;
    let expected_len = 1usize << log_t;
    let mut by_slot = std::collections::HashMap::with_capacity(materializations.len());
    for output in materializations {
        if by_slot.insert(output.slot, output.values).is_some() {
            return Err(core_fixture_error(
                request,
                "materialize stage1 Spartan outer kernel R1CS inputs",
                format!("duplicate materialization slot {:?}", output.slot),
            ));
        }
    }
    let r1cs_inputs = materialization_request
        .r1cs_inputs
        .iter()
        .map(|input| {
            let values = by_slot.remove(&input.slot).ok_or_else(|| {
                core_fixture_error(
                    request,
                    "materialize stage1 Spartan outer kernel R1CS inputs",
                    format!("missing materialization slot {:?}", input.slot),
                )
            })?;
            if values.len() != expected_len {
                return Err(core_fixture_error(
                    request,
                    "materialize stage1 Spartan outer kernel R1CS inputs",
                    format!(
                        "materialization for {:?} has {} rows, expected {expected_len}",
                        input.variable,
                        values.len()
                    ),
                ));
            }
            Ok(values)
        })
        .collect::<HarnessResult<Vec<_>>>()?;
    if let Some(slot) = by_slot.keys().next() {
        return Err(core_fixture_error(
            request,
            "materialize stage1 Spartan outer kernel R1CS inputs",
            format!("unexpected materialization slot {slot:?}"),
        ));
    }

    Ok(Stage1SpartanOuterKernelBenchmarkFixture {
        core_trace,
        core_bytecode: generated.core_preprocessing.shared.bytecode,
        log_t,
        r1cs_inputs,
    })
}

pub fn load_stage2_product_uniskip_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<CoreVerifierFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage2 product uni-skip verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let stage2 = prove_stage2_product_uniskip_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    verifier_replay_stage2_product_uniskip(request, &mut fixture.proof, stage2)?;
    fixture.verify().map_err(|error| {
        core_fixture_error(
            request,
            "verify stage2 product uni-skip verifier_replay",
            error,
        )
    })?;
    Ok(fixture)
}

pub fn load_stage2_regular_batch_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2RegularBatchVerifierReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage2 regular-batch verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let stage2 = prove_stage2_regular_batch_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    let modular = stage2.output.claims.batch_outputs.clone();
    verifier_replay_stage2_regular_batch(request, &mut fixture.proof, stage2)?;
    let verified = verify_core_stage2_clear(request, &fixture)?.output_claims;
    if modular != verified {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular-batch verifier_replay output claims",
            first_stage2_batch_output_mismatch(&verified, &modular)
                .unwrap_or_else(|| "Stage 2 batch output opening claims differ".to_owned()),
        ));
    }
    Ok(Stage2RegularBatchVerifierReplayCheckpoint {
        fixture,
        modular,
        verified,
    })
}

pub fn load_stage2_product_uniskip_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2ProductUniskipKernelBenchmarkFixture> {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|_| core_fixture_error(request, "lock core fixture generator", "poisoned lock"))?;
    let core_trace = generate_core_trace(request)?;
    let log_t = checked_power_of_two_log(request, core_trace.len())?;
    let rows = product_uniskip_rows_from_core_trace(&core_trace);
    Ok(Stage2ProductUniskipKernelBenchmarkFixture {
        core_trace,
        rows,
        log_t,
    })
}

pub fn load_stage2_regular_batch_input_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2RegularBatchInputCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage2 regular batch input checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let modular = prove_stage2_regular_batch_input_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    let expected = stage2_regular_batch_input_claims(&expected_stage2);
    if modular.input_claims != expected {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch input claims",
            first_stage2_batch_input_mismatch(&expected, &modular.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if modular.ram_read_write_gamma != expected_stage2.public.ram_read_write_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch transcript challenges",
            "RAM read-write gamma differs",
        ));
    }
    if modular.instruction_gamma != expected_stage2.public.instruction_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch transcript challenges",
            "instruction gamma differs",
        ));
    }
    if modular.output_address_challenges != expected_stage2.public.output_address_challenges {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch transcript challenges",
            "output-address challenges differ",
        ));
    }

    Ok(Stage2RegularBatchInputCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage2_regular_batch_input_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2RegularBatchInputKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage2 regular batch input kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, fixture.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, fixture.proof.ram_K, fixture.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(&core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage2 input benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage2 input benchmark transcript through stage1",
            error,
        )
    })?;
    let Stage1Output::Clear(stage1) = stage1 else {
        return Err(core_fixture_error(
            request,
            "advance modular stage2 input benchmark transcript through stage1",
            "expected clear Stage 1 output",
        ));
    };
    let input = Stage2ProductUniSkipInput::from_stage1(&stage1);

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let product_uniskip = prove_stage2_product_uniskip::<Fr, _, _, _, Bn254G1>(
        Stage2ProverConfig::new(log_t),
        &input,
        &witness,
        &mut backend,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "prove modular stage2 product uni-skip for input benchmark",
            error,
        )
    })?;

    let config = Stage2BatchProverConfig::new(log_t, log_k, fixture.proof.rw_config);
    let transcript_before_prefix = transcript.clone();
    let reference = reference_stage2_regular_batch_prefix(
        config,
        &stage1,
        &product_uniskip,
        &mut transcript.clone(),
    )?;
    let expected_input_claims = stage2_regular_batch_input_claims(&expected_stage2);
    if reference.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare reference stage2 regular batch input claims",
            first_stage2_batch_input_mismatch(&expected_input_claims, &reference.input_claims)
                .unwrap_or_else(|| "reference input claim vector differs".to_owned()),
        ));
    }
    if reference.ram_read_write_gamma != expected_stage2.public.ram_read_write_gamma {
        return Err(core_fixture_error(
            request,
            "compare reference stage2 regular batch challenges",
            "RAM read-write gamma differs",
        ));
    }
    if reference.instruction_gamma != expected_stage2.public.instruction_gamma {
        return Err(core_fixture_error(
            request,
            "compare reference stage2 regular batch challenges",
            "instruction gamma differs",
        ));
    }
    if reference.output_address_challenges != expected_stage2.public.output_address_challenges {
        return Err(core_fixture_error(
            request,
            "compare reference stage2 regular batch challenges",
            "output-address challenges differ",
        ));
    }

    Ok(Stage2RegularBatchInputKernelBenchmarkFixture {
        request: request.clone(),
        config,
        stage1,
        product_uniskip,
        transcript: transcript_before_prefix,
        expected: reference,
    })
}

pub fn load_stage3_regular_batch_input_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage3RegularBatchInputKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage3 regular batch input kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program: _,
        trace: _,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage3 = verify_core_stage3_clear(request, &fixture)?;
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage3 input benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage3 input benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage3 input benchmark transcript through stage2",
            error,
        )
    })?;
    let (Stage1Output::Clear(stage1), Stage2Output::Clear(stage2)) = (stage1, stage2) else {
        return Err(core_fixture_error(
            request,
            "advance modular stage3 input benchmark transcript through stage2",
            "expected clear Stage 1 and Stage 2 outputs",
        ));
    };

    let config = Stage3ProverConfig::new(log_t);
    let transcript_before_prefix = transcript.clone();
    let reference =
        reference_stage3_regular_batch_prefix(&stage1, &stage2, &mut transcript.clone());
    let expected_input_claims = stage3_regular_batch_input_claims(&expected_stage3);
    if reference.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare reference stage3 regular batch input claims",
            first_stage3_batch_input_mismatch(&expected_input_claims, &reference.input_claims)
                .unwrap_or_else(|| "reference input claim vector differs".to_owned()),
        ));
    }
    if reference.shift_gamma != expected_stage3.public.shift_gamma
        || reference.instruction_gamma != expected_stage3.public.instruction_gamma
        || reference.registers_gamma != expected_stage3.public.registers_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare reference stage3 regular batch challenges",
            "Stage 3 reference gammas differ from core",
        ));
    }

    Ok(Stage3RegularBatchInputKernelBenchmarkFixture {
        request: request.clone(),
        config,
        stage1,
        stage2,
        transcript: transcript_before_prefix,
        expected: reference,
    })
}

pub fn load_stage4_regular_batch_input_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage4RegularBatchInputKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage4 regular batch input kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program: _,
        trace: _,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage4 = verify_core_stage4_clear(request, &fixture)?;
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, fixture.proof.ram_K)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage4 input benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage4 input benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage4 input benchmark transcript through stage2",
            error,
        )
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage4 input benchmark deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage4 input benchmark transcript through stage3",
            error,
        )
    })?;
    let (Stage2Output::Clear(stage2), Stage3Output::Clear(stage3)) = (stage2, stage3) else {
        return Err(core_fixture_error(
            request,
            "advance modular stage4 input benchmark transcript through stage3",
            "expected clear Stage 2 and Stage 3 outputs",
        ));
    };

    let config = Stage4ProverConfig::new(log_t, log_k, fixture.proof.rw_config);
    let ram_val_check_init = stage4_ram_val_check_initial_evaluation(&expected_stage4);
    let transcript_before_prefix = transcript.clone();
    let reference = reference_stage4_regular_batch_prefix(
        &stage2,
        &stage3,
        ram_val_check_init.clone(),
        &mut transcript.clone(),
    );
    let expected_input_claims = stage4_regular_batch_input_claims(&expected_stage4);
    if reference.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare reference stage4 regular batch input claims",
            first_stage4_batch_input_mismatch(&expected_input_claims, &reference.input_claims)
                .unwrap_or_else(|| "reference input claim vector differs".to_owned()),
        ));
    }
    if reference.registers_gamma != expected_stage4.public.registers_gamma
        || reference.ram_val_check_gamma != expected_stage4.public.ram_val_check_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare reference stage4 regular batch challenges",
            "Stage 4 reference gammas differ from core",
        ));
    }

    Ok(Stage4RegularBatchInputKernelBenchmarkFixture {
        request: request.clone(),
        config,
        stage2,
        stage3,
        ram_val_check_init,
        transcript: transcript_before_prefix,
        expected: reference,
    })
}

pub fn load_stage4_regular_batch_sumcheck_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage4RegularBatchSumcheckKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage4 regular batch sumcheck benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let core = generate_stage2_regular_batch_core_context(request, proof.opening_claims.0.clone())?;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage4 = verify_core_stage4_clear(request, &fixture)?;
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, fixture.proof.ram_K)?;
    let preprocessing = convert_program_preprocessing(&core_preprocessing);
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage4 sumcheck benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage4 sumcheck benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage4 sumcheck benchmark transcript through stage2",
            error,
        )
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage4 sumcheck benchmark deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage4 sumcheck benchmark transcript through stage3",
            error,
        )
    })?;
    let (Stage2Output::Clear(stage2), Stage3Output::Clear(stage3)) = (stage2, stage3) else {
        return Err(core_fixture_error(
            request,
            "advance modular stage4 sumcheck benchmark transcript through stage3",
            "expected clear Stage 2 and Stage 3 outputs",
        ));
    };

    let config = Stage4ProverConfig::new(log_t, log_k, fixture.proof.rw_config);
    let ram_val_check_init = stage4_ram_val_check_initial_evaluation(&expected_stage4);
    let prefix = derive_stage4_regular_batch_prefix(
        config,
        &stage2,
        &stage3,
        ram_val_check_init.clone(),
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "derive modular Stage 4 regular-batch prefix for sumcheck benchmark",
            error,
        )
    })?;
    let expected_input_claims = stage4_regular_batch_input_claims(&expected_stage4);
    if prefix.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch sumcheck input claims",
            first_stage4_batch_input_mismatch(&expected_input_claims, &prefix.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if prefix.registers_gamma != expected_stage4.public.registers_gamma
        || prefix.ram_val_check_gamma != expected_stage4.public.ram_val_check_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch sumcheck prefix challenges",
            "Stage 4 prefix gammas differ from core",
        ));
    }

    Ok(Stage4RegularBatchSumcheckKernelBenchmarkFixture {
        request: request.clone(),
        config,
        checked,
        one_hot_config: fixture.proof.one_hot_config,
        stage2,
        stage3,
        ram_val_check_init,
        prefix,
        transcript,
        program,
        preprocessing,
        trace,
        expected: Stage4RegularBatchSumcheckExpected {
            proof: fixture.proof.stages.stage4_sumcheck_proof.clone(),
            challenges: expected_stage4.public.challenges,
            batching_coefficients: expected_stage4.public.batching_coefficients,
            output_claim: expected_stage4.batch.sumcheck_final_claim,
        },
        core,
    })
}

pub fn load_stage5_regular_batch_input_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage5RegularBatchInputKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage5 regular batch input kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let expected_stage5 = verify_core_stage5_clear(request, &fixture)?;
    let config = stage5_prover_config(request, &fixture)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage5 input benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 input benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 input benchmark transcript through stage2",
            error,
        )
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage5 input benchmark stage3 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 input benchmark transcript through stage3",
            error,
        )
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage5 input benchmark stage4 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 input benchmark transcript through stage4",
            error,
        )
    })?;
    let (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) = (stage2, stage4) else {
        return Err(core_fixture_error(
            request,
            "advance modular stage5 input benchmark transcript through stage4",
            "expected clear Stage 2 and Stage 4 outputs",
        ));
    };

    let transcript_before_prefix = transcript.clone();
    let reference =
        reference_stage5_regular_batch_prefix(&stage2, &stage4, &mut transcript.clone());
    let expected_input_claims = stage5_regular_batch_input_claims(&expected_stage5);
    if reference.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare reference stage5 regular batch input claims",
            first_stage5_batch_input_mismatch(&expected_input_claims, &reference.input_claims)
                .unwrap_or_else(|| "reference input claim vector differs".to_owned()),
        ));
    }
    if reference.instruction_gamma != expected_stage5.public.instruction_gamma
        || reference.ram_gamma != expected_stage5.public.ram_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare reference stage5 regular batch challenges",
            "Stage 5 reference gammas differ from core",
        ));
    }

    Ok(Stage5RegularBatchInputKernelBenchmarkFixture {
        request: request.clone(),
        config,
        stage2,
        stage4,
        transcript: transcript_before_prefix,
        expected: reference,
    })
}

pub fn load_stage5_regular_batch_sumcheck_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage5RegularBatchSumcheckKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage5 regular batch sumcheck benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let core = generate_stage2_regular_batch_core_context(request, proof.opening_claims.0.clone())?;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage5 = verify_core_stage5_clear(request, &fixture)?;
    let config = stage5_prover_config(request, &fixture)?;
    let preprocessing = convert_program_preprocessing(&core_preprocessing);

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage5 sumcheck benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 sumcheck benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 sumcheck benchmark transcript through stage2",
            error,
        )
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage5 sumcheck benchmark stage3 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 sumcheck benchmark transcript through stage3",
            error,
        )
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage5 sumcheck benchmark stage4 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage5 sumcheck benchmark transcript through stage4",
            error,
        )
    })?;
    let (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) = (stage2, stage4) else {
        return Err(core_fixture_error(
            request,
            "advance modular stage5 sumcheck benchmark transcript through stage4",
            "expected clear Stage 2 and Stage 4 outputs",
        ));
    };

    let prefix = derive_stage5_regular_batch_prefix(config, &stage2, &stage4, &mut transcript)
        .map_err(|error| {
            core_fixture_error(
                request,
                "derive modular Stage 5 regular-batch prefix for sumcheck benchmark",
                error,
            )
        })?;
    let expected_input_claims = stage5_regular_batch_input_claims(&expected_stage5);
    if prefix.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch sumcheck input claims",
            first_stage5_batch_input_mismatch(&expected_input_claims, &prefix.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if prefix.instruction_gamma != expected_stage5.public.instruction_gamma
        || prefix.ram_gamma != expected_stage5.public.ram_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch sumcheck prefix challenges",
            "Stage 5 prefix gammas differ from core",
        ));
    }

    Ok(Stage5RegularBatchSumcheckKernelBenchmarkFixture {
        request: request.clone(),
        config,
        checked,
        one_hot_config: fixture.proof.one_hot_config,
        stage2,
        stage4,
        prefix,
        transcript,
        program,
        preprocessing,
        trace,
        expected: Stage5RegularBatchSumcheckExpected {
            proof: fixture.proof.stages.stage5_sumcheck_proof.clone(),
            challenges: expected_stage5.public.challenges,
            batching_coefficients: expected_stage5.public.batching_coefficients,
            output_claim: expected_stage5.batch.sumcheck_final_claim,
        },
        core,
    })
}

pub fn load_stage6_regular_batch_input_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage6RegularBatchInputKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage6 regular batch input kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let expected_stage6 = verify_core_stage6_clear(request, &fixture)?;
    let config = stage6_prover_config(request, &fixture)?;

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage6 input benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 input benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 input benchmark transcript through stage2",
            error,
        )
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage6 input benchmark stage3 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 input benchmark transcript through stage3",
            error,
        )
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage6 input benchmark stage4 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 input benchmark transcript through stage4",
            error,
        )
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage6 input benchmark stage5 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 input benchmark transcript through stage5",
            error,
        )
    })?;
    let (
        Stage1Output::Clear(stage1),
        Stage2Output::Clear(stage2),
        Stage3Output::Clear(stage3),
        Stage4Output::Clear(stage4),
        Stage5Output::Clear(stage5),
    ) = (stage1, stage2, stage3, stage4, stage5)
    else {
        return Err(core_fixture_error(
            request,
            "advance modular stage6 input benchmark transcript through stage5",
            "expected clear Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5 outputs",
        ));
    };

    let transcript_before_prefix = transcript.clone();
    let reference = reference_stage6_regular_batch_prefix(
        request,
        config.clone(),
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
        &mut transcript.clone(),
    )?;
    compare_stage6_regular_batch_input_checkpoint(request, &expected_stage6, &reference)?;

    Ok(Stage6RegularBatchInputKernelBenchmarkFixture {
        request: request.clone(),
        config,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        transcript: transcript_before_prefix,
        expected: reference,
    })
}

pub fn load_stage6_regular_batch_sumcheck_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage6RegularBatchSumcheckKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage6 regular batch sumcheck benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let core = generate_stage2_regular_batch_core_context(request, proof.opening_claims.0.clone())?;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage6 = verify_core_stage6_clear(request, &fixture)?;
    let config = stage6_prover_config(request, &fixture)?;
    let preprocessing = convert_program_preprocessing(&core_preprocessing);

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage6 sumcheck benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 sumcheck benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 sumcheck benchmark transcript through stage2",
            error,
        )
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage6 sumcheck benchmark stage3 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 sumcheck benchmark transcript through stage3",
            error,
        )
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage6 sumcheck benchmark stage4 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 sumcheck benchmark transcript through stage4",
            error,
        )
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4).map_err(|error| {
            core_fixture_error(
                request,
                "assemble modular stage6 sumcheck benchmark stage5 deps",
                error,
            )
        })?,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage6 sumcheck benchmark transcript through stage5",
            error,
        )
    })?;
    let (
        Stage1Output::Clear(stage1),
        Stage2Output::Clear(stage2),
        Stage3Output::Clear(stage3),
        Stage4Output::Clear(stage4),
        Stage5Output::Clear(stage5),
    ) = (stage1, stage2, stage3, stage4, stage5)
    else {
        return Err(core_fixture_error(
            request,
            "advance modular stage6 sumcheck benchmark transcript through stage5",
            "expected clear Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5 outputs",
        ));
    };

    let prefix = derive_stage6_regular_batch_prefix(
        config.clone(),
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "derive modular Stage 6 regular-batch prefix for sumcheck benchmark",
            error,
        )
    })?;
    compare_stage6_regular_batch_input_checkpoint(request, &expected_stage6, &prefix)?;

    Ok(Stage6RegularBatchSumcheckKernelBenchmarkFixture {
        request: request.clone(),
        config,
        checked,
        one_hot_config: fixture.proof.one_hot_config,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        prefix,
        transcript,
        program,
        preprocessing,
        trace,
        expected: Stage6RegularBatchSumcheckExpected {
            proof: fixture.proof.stages.stage6_sumcheck_proof.clone(),
            challenges: expected_stage6.public.challenges,
            batching_coefficients: expected_stage6.public.batching_coefficients,
            output_claim: expected_stage6.batch.sumcheck_final_claim,
        },
        core,
    })
}

pub fn load_stage7_regular_batch_input_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage7RegularBatchInputKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage7 regular batch input kernel benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let config = stage7_prover_config(request, &fixture)?;

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage7 input benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "stage7 input benchmark stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "stage7 input benchmark stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(request, "stage7 input benchmark stage3 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 input benchmark stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(request, "stage7 input benchmark stage4 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 input benchmark stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4).map_err(|error| {
            core_fixture_error(request, "stage7 input benchmark stage5 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 input benchmark stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| {
                core_fixture_error(request, "stage7 input benchmark stage6 deps", error)
            })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 input benchmark stage6", error))?;
    let Stage6Output::Clear(stage6) = stage6 else {
        return Err(core_fixture_error(
            request,
            "stage7 input benchmark stage6",
            "expected clear Stage 6 output",
        ));
    };

    let transcript_before_prefix = transcript.clone();
    let reference =
        reference_stage7_regular_batch_prefix(&config, &stage6, &mut transcript.clone());
    let modular =
        derive_stage7_regular_batch_prefix(&config, &stage6, &mut transcript).map_err(|error| {
            core_fixture_error(request, "derive stage7 input benchmark claims", error)
        })?;
    if reference != modular {
        return Err(core_fixture_error(
            request,
            "compare reference stage7 regular batch input claims",
            "Stage 7 reference prefix differs from the modular derivation",
        ));
    }

    Ok(Stage7RegularBatchInputKernelBenchmarkFixture {
        request: request.clone(),
        config,
        stage6,
        transcript: transcript_before_prefix,
        expected: reference,
    })
}

pub fn load_stage7_regular_batch_sumcheck_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage7RegularBatchSumcheckKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage7 regular batch sumcheck benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let core_stage2 =
        generate_stage2_regular_batch_core_context(request, proof.opening_claims.0.clone())?;
    let core = Stage7RegularBatchCoreBenchmarkContext {
        trace: core_stage2.trace,
        preprocessing: core_preprocessing.shared.clone(),
        one_hot_params: core_stage2.one_hot_params,
        openings: core_stage2.openings,
    };
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let config = stage7_prover_config(request, &fixture)?;
    let preprocessing = convert_program_preprocessing(&core_preprocessing);

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage7 sumcheck benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(request, "stage7 sumcheck benchmark stage3 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(request, "stage7 sumcheck benchmark stage4 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4).map_err(|error| {
            core_fixture_error(request, "stage7 sumcheck benchmark stage5 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| {
                core_fixture_error(request, "stage7 sumcheck benchmark stage6 deps", error)
            })?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage6", error))?;
    let (Stage4Output::Clear(stage4), Stage6Output::Clear(stage6)) = (stage4, stage6) else {
        return Err(core_fixture_error(
            request,
            "stage7 sumcheck benchmark stage6",
            "expected clear Stage 4 and Stage 6 outputs",
        ));
    };

    let transcript_before_stage7 = transcript.clone();
    let expected_stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage7::inputs::Deps::Clear {
            stage4: &stage4,
            stage6: &stage6,
        },
    )
    .map_err(|error| core_fixture_error(request, "stage7 sumcheck benchmark stage7", error))?;
    let jolt_verifier::stages::stage7::Stage7Output::Clear(expected_stage7) = expected_stage7
    else {
        return Err(core_fixture_error(
            request,
            "stage7 sumcheck benchmark stage7",
            "expected clear Stage 7 output",
        ));
    };

    let mut prefix_transcript = transcript_before_stage7.clone();
    let prefix = derive_stage7_regular_batch_prefix(&config, &stage6, &mut prefix_transcript)
        .map_err(|error| {
            core_fixture_error(
                request,
                "derive modular Stage 7 regular-batch prefix for sumcheck benchmark",
                error,
            )
        })?;
    if prefix.input_claims.hamming_weight_claim_reduction
        != expected_stage7
            .batch
            .hamming_weight_claim_reduction
            .input_claim
        || prefix.hamming_gamma != expected_stage7.public.hamming_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare Stage 7 sumcheck benchmark prefix",
            "modular prefix differs from verifier Stage 7 output",
        ));
    }

    Ok(Stage7RegularBatchSumcheckKernelBenchmarkFixture {
        request: request.clone(),
        config,
        checked,
        one_hot_config: fixture.proof.one_hot_config,
        stage4,
        stage6,
        prefix,
        transcript: transcript_before_stage7,
        program,
        preprocessing,
        trace,
        expected: Stage7RegularBatchSumcheckExpected {
            proof: fixture.proof.stages.stage7_sumcheck_proof.clone(),
            challenges: expected_stage7.public.challenges,
            batching_coefficients: expected_stage7.public.batching_coefficients,
            output_claim: expected_stage7.batch.sumcheck_final_claim,
        },
        core,
    })
}

pub fn load_stage2_regular_batch_sumcheck_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2RegularBatchSumcheckKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage2 regular batch sumcheck benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let core = generate_stage2_regular_batch_core_context(request, proof.opening_claims.0.clone())?;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, fixture.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, fixture.proof.ram_K, fixture.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(&core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace.clone()),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage2 sumcheck benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage2 sumcheck benchmark transcript through stage1",
            error,
        )
    })?;
    let Stage1Output::Clear(stage1) = stage1 else {
        return Err(core_fixture_error(
            request,
            "advance modular stage2 sumcheck benchmark transcript through stage1",
            "expected clear Stage 1 output",
        ));
    };
    let product_input = Stage2ProductUniSkipInput::from_stage1(&stage1);

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let product_uniskip = prove_stage2_product_uniskip::<Fr, _, _, _, Bn254G1>(
        Stage2ProverConfig::new(log_t),
        &product_input,
        &witness,
        &mut backend,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "prove modular stage2 product uni-skip for sumcheck benchmark",
            error,
        )
    })?;

    let config = Stage2BatchProverConfig::new(log_t, log_k, fixture.proof.rw_config);
    let prefix =
        derive_stage2_regular_batch_prefix(config, &stage1, &product_uniskip, &mut transcript)
            .map_err(|error| {
                core_fixture_error(
                    request,
                    "derive modular Stage 2 regular-batch prefix for sumcheck benchmark",
                    error,
                )
            })?;
    let expected_input_claims = stage2_regular_batch_input_claims(&expected_stage2);
    if prefix.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch sumcheck input claims",
            first_stage2_batch_input_mismatch(&expected_input_claims, &prefix.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if prefix.ram_read_write_gamma != expected_stage2.public.ram_read_write_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch sumcheck prefix challenges",
            "RAM read-write gamma differs",
        ));
    }
    if prefix.instruction_gamma != expected_stage2.public.instruction_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch sumcheck prefix challenges",
            "instruction gamma differs",
        ));
    }
    if prefix.output_address_challenges != expected_stage2.public.output_address_challenges {
        return Err(core_fixture_error(
            request,
            "compare stage2 regular batch sumcheck prefix challenges",
            "output-address challenges differ",
        ));
    }

    Ok(Stage2RegularBatchSumcheckKernelBenchmarkFixture {
        request: request.clone(),
        config,
        checked,
        one_hot_config: fixture.proof.one_hot_config,
        stage1,
        product_uniskip,
        prefix,
        transcript,
        program,
        preprocessing,
        trace,
        expected: Stage2RegularBatchSumcheckExpected {
            proof: fixture.proof.stages.stage2_sumcheck_proof.clone(),
            challenges: expected_stage2.public.challenges,
            batching_coefficients: expected_stage2.public.batching_coefficients,
            output_claim: expected_stage2.batch.sumcheck_final_claim,
        },
        core,
    })
}

pub fn load_stage3_regular_batch_sumcheck_kernel_benchmark_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage3RegularBatchSumcheckKernelBenchmarkFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage3 regular batch sumcheck benchmarks currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let core = generate_stage2_regular_batch_core_context(request, proof.opening_claims.0.clone())?;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage3 = verify_core_stage3_clear(request, &fixture)?;
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let preprocessing = convert_program_preprocessing(&core_preprocessing);
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "initialize modular stage3 sumcheck benchmark transcript",
            error,
        )
    })?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage3 sumcheck benchmark transcript through stage1",
            error,
        )
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "advance modular stage3 sumcheck benchmark transcript through stage2",
            error,
        )
    })?;
    let (Stage1Output::Clear(stage1), Stage2Output::Clear(stage2)) = (stage1, stage2) else {
        return Err(core_fixture_error(
            request,
            "advance modular stage3 sumcheck benchmark transcript through stage2",
            "expected clear Stage 1 and Stage 2 outputs",
        ));
    };

    let config = Stage3ProverConfig::new(log_t);
    let prefix = derive_stage3_regular_batch_prefix(config, &stage1, &stage2, &mut transcript)
        .map_err(|error| {
            core_fixture_error(
                request,
                "derive modular Stage 3 regular-batch prefix for sumcheck benchmark",
                error,
            )
        })?;
    let expected_input_claims = stage3_regular_batch_input_claims(&expected_stage3);
    if prefix.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch sumcheck input claims",
            first_stage3_batch_input_mismatch(&expected_input_claims, &prefix.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if prefix.shift_gamma != expected_stage3.public.shift_gamma
        || prefix.instruction_gamma != expected_stage3.public.instruction_gamma
        || prefix.registers_gamma != expected_stage3.public.registers_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch sumcheck prefix challenges",
            "Stage 3 prefix gammas differ from core",
        ));
    }

    Ok(Stage3RegularBatchSumcheckKernelBenchmarkFixture {
        request: request.clone(),
        config,
        checked,
        one_hot_config: fixture.proof.one_hot_config,
        stage1,
        stage2,
        prefix,
        transcript,
        program,
        preprocessing,
        trace,
        expected: Stage3RegularBatchSumcheckExpected {
            proof: fixture.proof.stages.stage3_sumcheck_proof.clone(),
            challenges: expected_stage3.public.challenges,
            batching_coefficients: expected_stage3.public.batching_coefficients,
            output_claim: expected_stage3.batch.sumcheck_final_claim,
        },
        core,
    })
}

pub fn load_stage2_product_remainder_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2ProductRemainderOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage2 product-remainder opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let opening_point = expected_stage2
        .batch
        .product_remainder
        .opening_point
        .clone();
    let modular = evaluate_stage2_product_remainder_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        opening_point.clone(),
    )?;
    let expected = expected_stage2.output_claims.product_remainder;
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage2 product-remainder opening claims",
            first_product_remainder_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "product-remainder opening claims differ".to_owned()),
        ));
    }

    Ok(Stage2ProductRemainderOpeningCheckpoint {
        fixture,
        modular,
        expected,
        opening_point,
    })
}

pub fn load_stage2_ram_read_write_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2RamReadWriteOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage2 RAM read-write opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let opening_point = expected_stage2.batch.ram_read_write.opening_point.clone();
    let modular = evaluate_stage2_ram_read_write_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        opening_point.clone(),
    )?;
    let expected = expected_stage2.output_claims.ram_read_write;
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage2 RAM read-write opening claims",
            first_ram_read_write_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "RAM read-write opening claims differ".to_owned()),
        ));
    }

    Ok(Stage2RamReadWriteOpeningCheckpoint {
        fixture,
        modular,
        expected,
        opening_point,
    })
}

pub fn load_stage2_ram_terminal_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2RamTerminalOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage2 RAM terminal opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let ram_raf_opening_point = expected_stage2
        .batch
        .ram_raf_evaluation
        .opening_point
        .clone();
    let ram_output_check_opening_point =
        expected_stage2.batch.ram_output_check.opening_point.clone();
    let modular = evaluate_stage2_ram_terminal_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        ram_raf_opening_point.clone(),
        ram_output_check_opening_point.clone(),
    )?;
    let expected = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: expected_stage2.output_claims.ram_raf_evaluation,
        ram_output_check: expected_stage2.output_claims.ram_output_check,
    };
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage2 RAM terminal opening claims",
            first_ram_terminal_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "RAM terminal opening claims differ".to_owned()),
        ));
    }

    Ok(Stage2RamTerminalOpeningCheckpoint {
        fixture,
        modular,
        expected,
        ram_raf_opening_point,
        ram_output_check_opening_point,
    })
}

pub fn load_stage2_instruction_claim_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage2InstructionClaimOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context:
                "stage2 instruction-claim opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage2 = verify_core_stage2_clear(request, &fixture)?;
    let opening_point = expected_stage2
        .batch
        .instruction_claim_reduction
        .opening_point
        .clone();
    let modular = evaluate_stage2_instruction_claim_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        opening_point.clone(),
    )?;
    let expected = expected_stage2.output_claims.instruction_claim_reduction;
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage2 instruction-claim opening claims",
            first_instruction_claim_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "instruction-claim opening claims differ".to_owned()),
        ));
    }

    Ok(Stage2InstructionClaimOpeningCheckpoint {
        fixture,
        modular,
        expected,
        opening_point,
    })
}

pub fn load_stage3_regular_batch_input_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage3RegularBatchInputCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage3 regular batch input checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let expected_stage3 = verify_core_stage3_clear(request, &fixture)?;
    let modular = prove_stage3_regular_batch_input_checkpoint(request, &fixture)?;
    let expected = stage3_regular_batch_input_claims(&expected_stage3);
    if modular.input_claims != expected {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch input claims",
            first_stage3_batch_input_mismatch(&expected, &modular.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if modular.shift_gamma != expected_stage3.public.shift_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch transcript challenges",
            "shift gamma differs",
        ));
    }
    if modular.instruction_gamma != expected_stage3.public.instruction_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch transcript challenges",
            "instruction gamma differs",
        ));
    }
    if modular.registers_gamma != expected_stage3.public.registers_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch transcript challenges",
            "registers gamma differs",
        ));
    }

    Ok(Stage3RegularBatchInputCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage3_output_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage3OutputOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage3 output opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage3 = verify_core_stage3_clear(request, &fixture)?;
    let opening_points = Stage3OutputOpeningPoints {
        shift: expected_stage3.batch.shift.opening_point.clone(),
        instruction_input: expected_stage3
            .batch
            .instruction_input
            .opening_point
            .clone(),
        registers_claim_reduction: expected_stage3
            .batch
            .registers_claim_reduction
            .opening_point
            .clone(),
    };
    let modular = evaluate_stage3_output_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        opening_points.clone(),
    )?;
    let expected = stage3_output_opening_claims(&expected_stage3);
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage3 output opening claims",
            first_stage3_output_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "Stage 3 output opening claims differ".to_owned()),
        ));
    }

    Ok(Stage3OutputOpeningCheckpoint {
        fixture,
        modular,
        expected,
        shift_opening_point: opening_points.shift,
        instruction_input_opening_point: opening_points.instruction_input,
        registers_claim_reduction_opening_point: opening_points.registers_claim_reduction,
    })
}

pub fn load_stage3_regular_batch_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage3RegularBatchVerifierReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage3 regular batch verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage3 = verify_core_stage3_clear(request, &fixture)?;
    let stage3 = prove_stage3_regular_batch_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    compare_stage3_regular_batch_verifier_replay(request, &expected_stage3, &stage3.output)?;
    let modular = Stage3RegularBatchOutputOpeningClaims {
        shift: stage3.output.claims.shift.clone(),
        instruction_input: stage3.output.claims.instruction_input.clone(),
        registers_claim_reduction: stage3.output.claims.registers_claim_reduction.clone(),
    };
    let expected = stage3_output_opening_claims(&expected_stage3);
    verifier_replay_stage3_regular_batch(request, &mut fixture.proof, stage3)?;
    fixture.verify().map_err(|error| {
        core_fixture_error(
            request,
            "verify stage3 regular batch verifier_replay",
            error,
        )
    })?;
    Ok(Stage3RegularBatchVerifierReplayCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage4_regular_batch_input_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage4RegularBatchInputCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage4 regular batch input checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let expected_stage4 = verify_core_stage4_clear(request, &fixture)?;
    let modular = prove_stage4_regular_batch_input_checkpoint(
        request,
        &fixture,
        stage4_ram_val_check_initial_evaluation(&expected_stage4),
    )?;
    let expected = stage4_regular_batch_input_claims(&expected_stage4);
    if modular.input_claims != expected {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch input claims",
            first_stage4_batch_input_mismatch(&expected, &modular.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if modular.registers_gamma != expected_stage4.public.registers_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch transcript challenges",
            "registers gamma differs",
        ));
    }
    if modular.ram_val_check_gamma != expected_stage4.public.ram_val_check_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch transcript challenges",
            "RAM value-check gamma differs",
        ));
    }
    if modular.ram_val_check_init != stage4_ram_val_check_initial_evaluation(&expected_stage4) {
        return Err(core_fixture_error(
            request,
            "compare stage4 RAM value-check initial evaluation",
            "RAM value-check initial evaluation differs",
        ));
    }

    Ok(Stage4RegularBatchInputCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage4_output_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage4OutputOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage4 output opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage4 = verify_core_stage4_clear(request, &fixture)?;
    let prefix = prove_stage4_regular_batch_input_checkpoint(
        request,
        &fixture,
        stage4_ram_val_check_initial_evaluation(&expected_stage4),
    )?;
    let modular = evaluate_stage4_output_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        &prefix,
        expected_stage4
            .batch
            .registers_read_write
            .opening_point
            .clone(),
        expected_stage4.batch.ram_val_check.opening_point.clone(),
    )?;
    let expected = stage4_output_opening_claims(&expected_stage4);
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage4 output opening claims",
            first_stage4_output_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "Stage 4 output opening claims differ".to_owned()),
        ));
    }

    Ok(Stage4OutputOpeningCheckpoint {
        fixture,
        modular,
        expected,
        registers_read_write_opening_point: expected_stage4
            .batch
            .registers_read_write
            .opening_point,
        ram_val_check_opening_point: expected_stage4.batch.ram_val_check.opening_point,
    })
}

pub fn load_stage4_regular_batch_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage4RegularBatchVerifierReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage4 regular batch verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage4 = verify_core_stage4_clear(request, &fixture)?;
    let stage4 = prove_stage4_regular_batch_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        stage4_ram_val_check_initial_evaluation(&expected_stage4),
    )?;
    compare_stage4_regular_batch_verifier_replay(request, &expected_stage4, &stage4.output)?;
    let modular = Stage4RegularBatchOutputOpeningClaims {
        advice: stage4.output.claims.advice.clone(),
        registers_read_write: stage4.output.claims.registers_read_write.clone(),
        ram_val_check: stage4.output.claims.ram_val_check.clone(),
    };
    let expected = stage4_output_opening_claims(&expected_stage4);
    verifier_replay_stage4_regular_batch(request, &mut fixture.proof, stage4)?;
    fixture.verify().map_err(|error| {
        core_fixture_error(
            request,
            "verify stage4 regular batch verifier_replay",
            error,
        )
    })?;
    Ok(Stage4RegularBatchVerifierReplayCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage5_regular_batch_input_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage5RegularBatchInputCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage5 regular batch input checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let expected_stage5 = verify_core_stage5_clear(request, &fixture)?;
    let modular = prove_stage5_regular_batch_input_checkpoint(request, &fixture)?;
    let expected = stage5_regular_batch_input_claims(&expected_stage5);
    if modular.input_claims != expected {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch input claims",
            first_stage5_batch_input_mismatch(&expected, &modular.input_claims)
                .unwrap_or_else(|| "input claim vectors differ".to_owned()),
        ));
    }
    if modular.instruction_gamma != expected_stage5.public.instruction_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch transcript challenges",
            "instruction gamma differs",
        ));
    }
    if modular.ram_gamma != expected_stage5.public.ram_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch transcript challenges",
            "RAM gamma differs",
        ));
    }

    Ok(Stage5RegularBatchInputCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage5_output_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage5OutputOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage5 output opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage5 = verify_core_stage5_clear(request, &fixture)?;
    let modular = evaluate_stage5_output_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        expected_stage5
            .batch
            .instruction_read_raf
            .lookup_table_flag_opening_point
            .clone(),
        expected_stage5
            .batch
            .instruction_read_raf
            .instruction_ra_opening_points
            .clone(),
        expected_stage5
            .batch
            .instruction_read_raf
            .instruction_raf_flag_opening_point
            .clone(),
        expected_stage5
            .batch
            .ram_ra_claim_reduction
            .opening_point
            .clone(),
        expected_stage5
            .batch
            .registers_val_evaluation
            .opening_point
            .clone(),
    )?;
    let expected = stage5_output_opening_claims(&expected_stage5);
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage5 output opening claims",
            first_stage5_output_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "Stage 5 output opening claims differ".to_owned()),
        ));
    }

    Ok(Stage5OutputOpeningCheckpoint {
        fixture,
        modular,
        expected,
        instruction_lookup_table_flag_opening_point: expected_stage5
            .batch
            .instruction_read_raf
            .lookup_table_flag_opening_point,
        instruction_ra_opening_points: expected_stage5
            .batch
            .instruction_read_raf
            .instruction_ra_opening_points,
        instruction_raf_flag_opening_point: expected_stage5
            .batch
            .instruction_read_raf
            .instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_opening_point: expected_stage5
            .batch
            .ram_ra_claim_reduction
            .opening_point,
        registers_val_evaluation_opening_point: expected_stage5
            .batch
            .registers_val_evaluation
            .opening_point,
    })
}

pub fn load_stage5_regular_batch_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage5RegularBatchVerifierReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage5 regular batch verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage5 = verify_core_stage5_clear(request, &fixture)?;
    let stage5 = prove_stage5_regular_batch_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    compare_stage5_regular_batch_verifier_replay(request, &expected_stage5, &stage5.output)?;
    let modular = stage5.output.claims.clone();
    let expected = stage5_output_opening_claims(&expected_stage5);
    verifier_replay_stage5_regular_batch(request, &mut fixture.proof, stage5)?;
    fixture.verify().map_err(|error| {
        core_fixture_error(
            request,
            "verify stage5 regular batch verifier_replay",
            error,
        )
    })?;
    Ok(Stage5RegularBatchVerifierReplayCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage6_regular_batch_input_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage6RegularBatchInputCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage6 regular batch input checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let expected_stage6 = verify_core_stage6_clear(request, &fixture)?;
    let modular = prove_stage6_regular_batch_input_checkpoint(request, &fixture)?;
    let expected = stage6_regular_batch_input_claims(&expected_stage6);
    compare_stage6_regular_batch_input_checkpoint(request, &expected_stage6, &modular)?;

    Ok(Stage6RegularBatchInputCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage6_output_opening_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage6OutputOpeningCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage6 output opening checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage6 = verify_core_stage6_clear(request, &fixture)?;
    let expected_stage4 = verify_core_stage4_clear(request, &fixture)?;
    let modular = evaluate_stage6_output_opening_checkpoint(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
        &expected_stage4,
        &expected_stage6,
    )?;
    let expected = stage6_output_opening_claims(&expected_stage6);
    if modular != expected {
        return Err(core_fixture_error(
            request,
            "compare stage6 output opening claims",
            first_stage6_output_opening_mismatch(&expected, &modular)
                .unwrap_or_else(|| "Stage 6 output opening claims differ".to_owned()),
        ));
    }

    Ok(Stage6OutputOpeningCheckpoint {
        fixture,
        modular,
        expected,
    })
}

pub fn load_stage6_regular_batch_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage6RegularBatchVerifierReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage6 regular batch verifier_replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected_stage6 = verify_core_stage6_clear(request, &fixture)?;
    let stage6 = prove_stage6_regular_batch_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    compare_stage6_regular_batch_verifier_replay(
        request,
        &expected_stage6,
        &fixture.proof.stages.stage6_sumcheck_proof,
        &stage6.output,
    )?;
    let modular = stage6.output.claims.clone();
    let expected = stage6_output_opening_claims(&expected_stage6);
    verifier_replay_stage6_regular_batch(request, &mut fixture.proof, stage6)?;
    fixture.verify().map_err(|error| {
        core_fixture_error(
            request,
            "verify stage6 regular batch verifier_replay",
            error,
        )
    })?;
    Ok(Stage6RegularBatchVerifierReplayCheckpoint {
        fixture,
        modular,
        expected,
    })
}

fn stage0_commitment_mismatch(
    expected: &jolt_verifier::proof::JoltCommitments<DoryCommitment>,
    actual: &jolt_verifier::proof::JoltCommitments<DoryCommitment>,
) -> Option<String> {
    if expected.rd_inc != actual.rd_inc {
        return Some("RdInc commitment differs".to_owned());
    }
    if expected.ram_inc != actual.ram_inc {
        return Some("RamInc commitment differs".to_owned());
    }
    first_vec_mismatch(
        "InstructionRa",
        &expected.ra.instruction,
        &actual.ra.instruction,
    )
    .or_else(|| first_vec_mismatch("RamRa", &expected.ra.ram, &actual.ra.ram))
    .or_else(|| first_vec_mismatch("BytecodeRa", &expected.ra.bytecode, &actual.ra.bytecode))
}

fn first_vec_mismatch<T: PartialEq>(label: &str, expected: &[T], actual: &[T]) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "{label} commitment count differs: expected {}, got {}",
            expected.len(),
            actual.len(),
        ));
    }
    expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .map(|index| format!("{label}({index}) commitment differs"))
}

fn first_point_mismatch<T: PartialEq + std::fmt::Debug>(
    label: &str,
    expected: &[T],
    actual: &[T],
) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "{label} length differs: expected {}, got {}",
            expected.len(),
            actual.len(),
        ));
    }
    expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .map(|index| {
            format!(
                "{label} differs at index {index}: expected {:?}, got {:?}",
                expected[index], actual[index],
            )
        })
}

fn first_sumcheck_round_mismatch(
    expected: &jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    actual: &jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
) -> Option<String> {
    let (
        jolt_sumcheck::SumcheckProof::Clear(jolt_sumcheck::ClearProof::Compressed(expected)),
        jolt_sumcheck::SumcheckProof::Clear(jolt_sumcheck::ClearProof::Compressed(actual)),
    ) = (expected, actual)
    else {
        return Some("Stage 6 proof encoding differs".to_owned());
    };

    if expected.round_polynomials.len() != actual.round_polynomials.len() {
        return Some(format!(
            "Stage 6 round count differs: expected {}, got {}",
            expected.round_polynomials.len(),
            actual.round_polynomials.len(),
        ));
    }
    expected
        .round_polynomials
        .iter()
        .zip(&actual.round_polynomials)
        .enumerate()
        .find_map(|(round, (expected, actual))| {
            first_point_mismatch(
                &format!("Stage 6 round {round} compressed coefficients"),
                expected.coeffs_except_linear_term(),
                actual.coeffs_except_linear_term(),
            )
        })
}

fn generate_fixture(request: &FixtureRequest) -> HarnessResult<GeneratedCoreFixture> {
    match request.kind {
        FixtureKind::MuldivSmall => generate_muldiv(request),
        FixtureKind::AdviceConsumer => generate_advice_consumer(request),
        _ => Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "no core verifier fixture is registered for this kind",
        }),
    }
}

fn generate_core_trace(request: &FixtureRequest) -> HarnessResult<Arc<Vec<CoreTraceCycle>>> {
    match request.kind {
        FixtureKind::MuldivSmall => {
            let inputs = postcard::to_stdvec(&[9u32, 5, 3])
                .map_err(|error| core_fixture_error(request, "serialize muldiv inputs", error))?;
            generate_core_trace_from_program(
                request,
                host::Program::new("muldiv-guest"),
                inputs,
                Vec::new(),
                Vec::new(),
            )
        }
        FixtureKind::AdviceConsumer => {
            let inputs = postcard::to_stdvec(&12u64).map_err(|error| {
                core_fixture_error(request, "serialize advice public input", error)
            })?;
            let untrusted_advice = postcard::to_stdvec(&5u64).map_err(|error| {
                core_fixture_error(request, "serialize untrusted advice", error)
            })?;
            let trusted_advice = postcard::to_stdvec(&7u64)
                .map_err(|error| core_fixture_error(request, "serialize trusted advice", error))?;
            generate_core_trace_from_program(
                request,
                host::Program::new("advice-consumer-guest"),
                inputs,
                untrusted_advice,
                trusted_advice,
            )
        }
        _ => Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "no core trace fixture is registered for this kind",
        }),
    }
}

fn generate_stage2_regular_batch_core_context(
    request: &FixtureRequest,
    openings: core_opening::Openings<CoreField>,
) -> HarnessResult<Stage2RegularBatchCoreBenchmarkContext> {
    match request.kind {
        FixtureKind::MuldivSmall => {
            let inputs = postcard::to_stdvec(&[9u32, 5, 3])
                .map_err(|error| core_fixture_error(request, "serialize muldiv inputs", error))?;
            generate_stage2_regular_batch_core_context_from_program(
                request,
                host::Program::new("muldiv-guest"),
                inputs,
                Vec::new(),
                Vec::new(),
                None,
                openings,
            )
        }
        FixtureKind::AdviceConsumer => {
            let inputs = postcard::to_stdvec(&12u64).map_err(|error| {
                core_fixture_error(request, "serialize advice public input", error)
            })?;
            let untrusted_advice = postcard::to_stdvec(&5u64).map_err(|error| {
                core_fixture_error(request, "serialize untrusted advice", error)
            })?;
            let trusted_advice = postcard::to_stdvec(&7u64)
                .map_err(|error| core_fixture_error(request, "serialize trusted advice", error))?;
            generate_stage2_regular_batch_core_context_from_program(
                request,
                host::Program::new("advice-consumer-guest"),
                inputs,
                untrusted_advice,
                trusted_advice,
                Some(commit_trusted_advice_preprocessing_only),
                openings,
            )
        }
        _ => Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "no core Stage 2 regular-batch benchmark fixture is registered for this kind",
        }),
    }
}

fn generate_stage2_regular_batch_core_context_from_program(
    request: &FixtureRequest,
    mut program: host::Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
    trusted_advice_committer: Option<TrustedAdviceCommitter>,
    openings: core_opening::Openings<CoreField>,
) -> HarnessResult<Stage2RegularBatchCoreBenchmarkContext> {
    let jolt_program = program
        .jolt_program()
        .map_err(|error| core_fixture_error(request, "build Jolt program", error))?;
    let mut tracer_backend = tracer::TracerBackend::new();
    let trace = program
        .trace_with_backend(
            &mut tracer_backend,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        )
        .map_err(|error| core_fixture_error(request, "trace Jolt program", error))?;

    let shared_preprocessing = JoltSharedPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        trace.device.memory_layout.clone(),
        jolt_program.memory_init,
        1 << 16,
        jolt_program.entry_address,
    )
    .map_err(|error| core_fixture_error(request, "preprocess core Stage 2 benchmark", error))?;
    let prover_preprocessing: JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| core_fixture_error(request, "read guest ELF", "missing ELF contents"))?;
    let (trusted_advice_commitment, trusted_advice_hint) = trusted_advice_committer
        .map(|commit| commit(&prover_preprocessing, &trusted_advice))
        .map_or((None, None), |(commitment, hint)| {
            (Some(commitment), Some(hint))
        });
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        trusted_advice_commitment,
        trusted_advice_hint,
        None,
    );

    Ok(Stage2RegularBatchCoreBenchmarkContext {
        trace: Arc::clone(&prover.trace),
        bytecode: prover.preprocessing.shared.bytecode.clone(),
        ram_preprocessing: prover.preprocessing.shared.ram.clone(),
        program_io: prover.program_io.clone(),
        initial_ram_state: prover.initial_ram_state.clone(),
        final_ram_state: prover.final_ram_state.clone(),
        one_hot_params: prover.one_hot_params,
        rw_config: prover.rw_config,
        openings,
    })
}

fn generate_core_trace_from_program(
    request: &FixtureRequest,
    mut program: host::Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
) -> HarnessResult<Arc<Vec<CoreTraceCycle>>> {
    let jolt_program = program
        .jolt_program()
        .map_err(|error| core_fixture_error(request, "build Jolt program", error))?;
    let mut tracer_backend = tracer::TracerBackend::new();
    let trace = program
        .trace_with_backend(
            &mut tracer_backend,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        )
        .map_err(|error| core_fixture_error(request, "trace Jolt program", error))?;

    let shared_preprocessing = JoltSharedPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        trace.device.memory_layout.clone(),
        jolt_program.memory_init.clone(),
        1 << 16,
        jolt_program.entry_address,
    )
    .map_err(|error| core_fixture_error(request, "preprocess core trace fixture", error))?;
    let prover_preprocessing: JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| core_fixture_error(request, "read guest ELF", "missing ELF contents"))?;
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        None,
        None,
        None,
    );
    Ok(prover.trace)
}

fn product_uniskip_rows_from_core_trace(
    trace: &[CoreTraceCycle],
) -> Vec<SumcheckProductUniskipRow> {
    (0..trace.len())
        .map(|index| {
            let row = ProductCycleInputs::from_trace::<CoreField>(trace, index);
            SumcheckProductUniskipRow::new(
                row.instruction_left_input,
                row.should_branch_lookup_output,
                row.jump_flag,
                row.instruction_right_input,
                row.should_branch_flag,
                !row.not_next_noop,
            )
        })
        .collect()
}

fn generate_muldiv(request: &FixtureRequest) -> HarnessResult<GeneratedCoreFixture> {
    let inputs = postcard::to_stdvec(&[9u32, 5, 3])
        .map_err(|error| core_fixture_error(request, "serialize muldiv inputs", error))?;
    generate_core_fixture(
        request,
        host::Program::new("muldiv-guest"),
        inputs,
        Vec::new(),
        Vec::new(),
        None,
    )
}

fn generate_advice_consumer(request: &FixtureRequest) -> HarnessResult<GeneratedCoreFixture> {
    let inputs = postcard::to_stdvec(&12u64)
        .map_err(|error| core_fixture_error(request, "serialize advice public input", error))?;
    let untrusted_advice = postcard::to_stdvec(&5u64)
        .map_err(|error| core_fixture_error(request, "serialize untrusted advice", error))?;
    let trusted_advice = postcard::to_stdvec(&7u64)
        .map_err(|error| core_fixture_error(request, "serialize trusted advice", error))?;

    generate_core_fixture(
        request,
        host::Program::new("advice-consumer-guest"),
        inputs,
        untrusted_advice,
        trusted_advice,
        Some(commit_trusted_advice_preprocessing_only),
    )
}

fn generate_core_fixture(
    request: &FixtureRequest,
    mut program: host::Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
    trusted_advice_committer: Option<TrustedAdviceCommitter>,
) -> HarnessResult<GeneratedCoreFixture> {
    let jolt_program = program
        .jolt_program()
        .map_err(|error| core_fixture_error(request, "build Jolt program", error))?;
    let mut tracer_backend = tracer::TracerBackend::new();
    let trace = program
        .trace_with_backend(
            &mut tracer_backend,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        )
        .map_err(|error| core_fixture_error(request, "trace Jolt program", error))?;

    let shared_preprocessing = JoltSharedPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        trace.device.memory_layout.clone(),
        jolt_program.memory_init.clone(),
        1 << 16,
        jolt_program.entry_address,
    )
    .map_err(|error| core_fixture_error(request, "preprocess core fixture", error))?;
    let prover_preprocessing: JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing);
    let prover_setup = DoryProverSetup(prover_preprocessing.generators.clone());
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| core_fixture_error(request, "read guest ELF", "missing ELF contents"))?;

    let (trusted_advice_commitment, trusted_advice_hint) = trusted_advice_committer
        .map(|commit| commit(&prover_preprocessing, &trusted_advice))
        .map_or((None, None), |(commitment, hint)| {
            (Some(commitment), Some(hint))
        });

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        trusted_advice_commitment,
        trusted_advice_hint,
        None,
    );
    let public_io = prover.program_io.clone();
    let (proof, _) = prover.prove();
    let core_preprocessing = CoreVerifierPreprocessing::from(&prover_preprocessing);
    let trace = TraceOutput::new(trace.trace, public_io.clone(), trace.final_memory);

    Ok(GeneratedCoreFixture {
        core_preprocessing,
        prover_setup,
        program: jolt_program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    })
}

fn commit_trusted_advice_preprocessing_only(
    preprocessing: &JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (CoreCommitment, CoreOpeningHint) {
    commit_core_advice_context(
        &preprocessing.shared,
        &preprocessing.generators,
        JoltAdviceKind::Trusted,
        trusted_advice_bytes,
    )
}

fn commit_core_advice_context(
    shared: &JoltSharedPreprocessing,
    generators: &CoreProverSetup,
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
) -> (CoreCommitment, CoreOpeningHint) {
    let (max_advice_size, context) = match kind {
        JoltAdviceKind::Trusted => (
            shared.memory_layout.max_trusted_advice_size,
            DoryContext::TrustedAdvice,
        ),
        JoltAdviceKind::Untrusted => (
            shared.memory_layout.max_untrusted_advice_size,
            DoryContext::UntrustedAdvice,
        ),
    };
    let mut advice_words = vec![0_u64; (max_advice_size as usize) / 8];
    populate_memory_states(0, advice_bytes, Some(&mut advice_words), None);

    let poly = MultilinearPolynomial::<CoreField>::from(advice_words);
    let advice_len = poly.len().next_power_of_two().max(1);

    let _guard = DoryGlobals::initialize_context(1, advice_len, context, None);
    let _ctx = DoryGlobals::with_context(context);
    DoryCommitmentScheme::commit(&poly, generators)
}

fn stage0_advice_commitment_request(
) -> CommitmentRequest<jolt_witness::protocols::jolt_vm::JoltVmNamespace> {
    CommitmentRequest::new(vec![
        CommitmentRequestItem::new(
            CommitmentSlot(0),
            ViewRequirement::new(
                OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice),
                PolynomialEncoding::Compact,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughBlindFold,
            ),
        ),
        CommitmentRequestItem::new(
            CommitmentSlot(1),
            ViewRequirement::new(
                OracleRef::committed(JoltCommittedPolynomial::UntrustedAdvice),
                PolynomialEncoding::Compact,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughBlindFold,
            ),
        ),
    ])
}

fn advice_rows(max_advice_size: u64) -> usize {
    ((max_advice_size as usize) / 8).next_power_of_two().max(1)
}

fn dory_row_width(rows: usize) -> usize {
    let log_rows = rows.trailing_zeros() as usize;
    1usize << log_rows.div_ceil(2)
}

fn convert_fixture(
    request: &FixtureRequest,
    fixture: GeneratedCoreFixture,
) -> HarnessResult<CoreVerifierFixture> {
    convert_fixture_parts(
        request,
        &fixture.core_preprocessing,
        fixture.public_io,
        fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
    )
}

fn convert_fixture_parts(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    public_io: JoltDevice,
    proof: CoreProof,
    trusted_advice_commitment: Option<&CoreCommitment>,
) -> HarnessResult<CoreVerifierFixture> {
    Ok(CoreVerifierFixture {
        preprocessing: convert_preprocessing(core_preprocessing),
        public_io,
        proof: proof
            .try_into()
            .map_err(|error| core_fixture_error(request, "convert core proof", error))?,
        trusted_advice_commitment: trusted_advice_commitment
            .copied()
            .map(<DoryCommitmentScheme as CorePcsBridge<CoreField>>::commitment_into_verifier),
    })
}

fn convert_preprocessing(
    preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> ConvertedPreprocessing {
    JoltVerifierPreprocessing::new(
        JoltProgramPreprocessing {
            bytecode: preprocessing.shared.bytecode.as_ref().clone(),
            ram: preprocessing.shared.ram.clone(),
            memory_layout: preprocessing.shared.memory_layout.clone(),
            max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
        },
        preprocessing.shared.digest(),
        DoryVerifierSetup(preprocessing.generators.clone()),
        None,
    )
}

struct Stage0CommitmentVerifierReplay {
    commitments: jolt_verifier::proof::JoltCommitments<DoryCommitment>,
    trusted_advice_commitment: Option<DoryCommitment>,
    untrusted_advice_commitment: Option<DoryCommitment>,
}

fn prove_stage0_commitments(
    request: &FixtureRequest,
    fixture: &GeneratedCoreFixture,
) -> HarnessResult<Stage0CommitmentVerifierReplay> {
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let one_hot_config = convert_one_hot_config(fixture.proof.one_hot_config.clone());
    let dimensions = JoltFormulaDimensions::try_from(one_hot_config.dimensions(
        log_t,
        RV64_LOOKUP_ADDRESS_BITS,
        fixture.core_preprocessing.shared.bytecode.code_size,
        fixture.proof.ram_K,
    ))
    .map_err(|error| core_fixture_error(request, "derive stage0 formula dimensions", error))?;
    let include_trusted_advice = !fixture.trace.device.trusted_advice.is_empty();
    let include_untrusted_advice = !fixture.trace.device.untrusted_advice.is_empty();
    let witness_config = JoltVmWitnessConfig::new(log_t, fixture.proof.ram_K, one_hot_config)
        .include_trusted_advice(include_trusted_advice)
        .include_untrusted_advice(include_untrusted_advice);
    let preprocessing = convert_program_preprocessing(&fixture.core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&fixture.program, &preprocessing, fixture.trace.clone()),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let output = prove::<Fr, _, _, DoryScheme>(
        CommitmentStageInput::new(
            &witness,
            &fixture.prover_setup,
            CommitmentStageConfig::new(
                dimensions.ra_layout,
                include_trusted_advice,
                include_untrusted_advice,
            ),
            jolt_verifier::JoltProtocolConfig::for_zk(false),
        ),
        &mut backend,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage0 commitments", error))?;

    Ok(Stage0CommitmentVerifierReplay {
        commitments: output.commitments,
        trusted_advice_commitment: output.trusted_advice_commitment,
        untrusted_advice_commitment: output.untrusted_advice_commitment,
    })
}

fn stage0_core_generators(
    fixture: &GeneratedCoreFixture,
) -> <DoryCommitmentScheme as CoreCommitmentScheme>::ProverSetup {
    let max_t = fixture
        .core_preprocessing
        .shared
        .max_padded_trace_length
        .next_power_of_two();
    let max_log_t = max_t.trailing_zeros() as usize;
    let max_log_k_chunk = if max_log_t < common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T {
        4
    } else {
        8
    };
    <DoryCommitmentScheme as CoreCommitmentScheme>::setup_prover(max_log_k_chunk + max_log_t)
}

struct Stage1SpartanOuterCoreParity {
    claims: Stage1Claims<Fr>,
    opening_point: Vec<Fr>,
    claim_count: usize,
}

struct Stage1SpartanOuterVerifierReplay {
    uniskip_proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    remainder_proof: jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    claims: Stage1Claims<Fr>,
}

struct Stage2ProductUniSkipVerifierReplay {
    output: Stage2ProductUniSkipOutput<Fr, Bn254G1>,
}

struct Stage2RegularBatchVerifierReplay {
    output: Stage2ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
}

struct Stage3RegularBatchVerifierReplay {
    output: Stage3ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
}

struct Stage4RegularBatchVerifierReplay {
    output: Stage4ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
}

struct Stage5RegularBatchVerifierReplay {
    output: Stage5ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
}

struct Stage6RegularBatchVerifierReplay {
    output: Stage6ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
}

fn prove_stage1_spartan_outer_checkpoint(
    request: &FixtureRequest,
    fixture: &GeneratedCoreFixture,
) -> HarnessResult<Stage1SpartanOuterCoreParity> {
    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let one_hot_config = convert_one_hot_config(fixture.proof.one_hot_config.clone());
    let witness_config = JoltVmWitnessConfig::new(log_t, fixture.proof.ram_K, one_hot_config)
        .include_trusted_advice(!fixture.trace.device.trusted_advice.is_empty())
        .include_untrusted_advice(!fixture.trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(&fixture.core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&fixture.program, &preprocessing, fixture.trace.clone()),
    );
    let (opening_point, expected_claims, uniskip_output_claim) =
        core_stage1_spartan_outer_claims(request, &fixture.proof)?;

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let actual_claims = evaluate_stage1_r1cs_inputs(
        Stage1ProverConfig::new(log_t),
        &witness,
        &mut backend,
        opening_point.clone(),
    )
    .map_err(|error| core_fixture_error(request, "evaluate modular stage1 R1CS claims", error))?;
    if actual_claims != expected_claims {
        return Err(core_fixture_error(
            request,
            "compare stage1 R1CS opening claims",
            first_stage1_claim_mismatch(&expected_claims, &actual_claims)
                .unwrap_or_else(|| "claim vectors differ".to_owned()),
        ));
    }

    let claims = stage1_claims_from_r1cs_inputs(uniskip_output_claim, &actual_claims)
        .map_err(|error| core_fixture_error(request, "assemble modular stage1 claims", error))?;
    Ok(Stage1SpartanOuterCoreParity {
        claims,
        opening_point,
        claim_count: actual_claims.len(),
    })
}

fn prove_stage1_spartan_outer_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage1SpartanOuterVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(converted.trusted_advice_commitment.is_some())
            .include_untrusted_advice(converted.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage1 transcript", error))?;

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let output = prove_stage1::<Fr, _, _, _, Bn254G1>(
        Stage1ProverInput::new(Stage1ProverConfig::new(log_t), &witness),
        &mut backend,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage1 Spartan outer", error))?;
    let claims =
        stage1_claims_from_r1cs_inputs(output.uniskip_output_claim, &output.r1cs_input_claims)
            .map_err(|error| {
                core_fixture_error(
                    request,
                    "assemble modular stage1 verifier_replay claims",
                    error,
                )
            })?;

    // The transparent prover must emit a verifier-mirroring `Stage1ClearOutput`
    // that the full-proof orchestrator threads into Stage 2. Validate it equals
    // what `jolt-verifier`'s Stage 1 derives from the (transcript-identical) core
    // proof, on a fresh transcript replayed to the pre-Stage-1 position.
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let _ = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut verifier_transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize stage1 verifier transcript", error))?;
    let verifier_stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut verifier_transcript,
    )
    .map_err(|error| core_fixture_error(request, "derive verifier stage1 clear output", error))?;
    let Stage1Output::Clear(verifier_clear) = verifier_stage1 else {
        return Err(core_fixture_error(
            request,
            "compare modular stage1 verifier output",
            "verifier produced a ZK stage1 output in clear mode",
        ));
    };
    match output.verifier_output.as_ref() {
        Some(prover_clear) if *prover_clear == verifier_clear => {}
        Some(_) => {
            return Err(core_fixture_error(
                request,
                "compare modular stage1 verifier output",
                "prover Stage1ClearOutput differs from jolt-verifier",
            ))
        }
        None => {
            return Err(core_fixture_error(
                request,
                "compare modular stage1 verifier output",
                "transparent prover did not emit a Stage1ClearOutput",
            ))
        }
    }

    Ok(Stage1SpartanOuterVerifierReplay {
        uniskip_proof: output.uniskip_proof,
        remainder_proof: output.remainder_proof,
        claims,
    })
}

fn verifier_replay_stage1_spartan_outer(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage1: Stage1SpartanOuterVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage1_uni_skip_first_round_proof = stage1.uniskip_proof;
    proof.stages.stage1_sumcheck_proof = stage1.remainder_proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage1 Spartan outer",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage1 = stage1.claims;
    Ok(())
}

fn prove_stage2_product_uniskip_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage2ProductUniSkipVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(converted.trusted_advice_commitment.is_some())
            .include_untrusted_advice(converted.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage2 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let Stage1Output::Clear(stage1) = stage1 else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage1",
            "expected clear Stage 1 output",
        ));
    };
    let input = Stage2ProductUniSkipInput::from_stage1(&stage1);

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let output = prove_stage2_product_uniskip::<Fr, _, _, _, Bn254G1>(
        Stage2ProverConfig::new(log_t),
        &input,
        &witness,
        &mut backend,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage2 product uni-skip", error))?;
    Ok(Stage2ProductUniSkipVerifierReplay { output })
}

fn prove_stage2_regular_batch_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage2RegularBatchVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(converted.trusted_advice_commitment.is_some())
            .include_untrusted_advice(converted.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage2 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let Stage1Output::Clear(stage1) = stage1 else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage1",
            "expected clear Stage 1 output",
        ));
    };

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let output = prove_stage2::<Fr, _, _, _, Bn254G1>(
        Stage2ProverInput::new(
            Stage2BatchProverConfig::new(log_t, log_k, converted.proof.rw_config),
            &checked,
            &stage1,
            &witness,
        ),
        &mut backend,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage2 regular batch", error))?;
    Ok(Stage2RegularBatchVerifierReplay { output })
}

fn prove_stage2_regular_batch_input_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage2RegularBatchPrefixOutput<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage2 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let Stage1Output::Clear(stage1) = stage1 else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage1",
            "expected clear Stage 1 output",
        ));
    };
    let input = Stage2ProductUniSkipInput::from_stage1(&stage1);

    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let product_uniskip = prove_stage2_product_uniskip::<Fr, _, _, _, Bn254G1>(
        Stage2ProverConfig::new(log_t),
        &input,
        &witness,
        &mut backend,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage2 product uni-skip", error))?;

    derive_stage2_regular_batch_prefix(
        Stage2BatchProverConfig::new(log_t, log_k, converted.proof.rw_config),
        &stage1,
        &product_uniskip,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "derive modular stage2 batch input claims", error))
}

fn evaluate_stage2_product_remainder_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    opening_point: Vec<Fr>,
) -> HarnessResult<ProductRemainderOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage2_product_remainder_openings(
        Stage2ProverConfig::new(log_t),
        &witness,
        &mut backend,
        opening_point,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "evaluate modular stage2 product-remainder openings",
            error,
        )
    })
}

fn evaluate_stage2_instruction_claim_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    opening_point: Vec<Fr>,
) -> HarnessResult<InstructionClaimReductionOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage2_instruction_claim_openings(
        Stage2ProverConfig::new(log_t),
        &witness,
        &mut backend,
        opening_point,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "evaluate modular stage2 instruction-claim openings",
            error,
        )
    })
}

fn evaluate_stage2_ram_read_write_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    opening_point: Vec<Fr>,
) -> HarnessResult<RamReadWriteOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage2_ram_read_write_openings(
        Stage2BatchProverConfig::new(log_t, log_k, converted.proof.rw_config),
        &witness,
        &mut backend,
        opening_point,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "evaluate modular stage2 RAM read-write openings",
            error,
        )
    })
}

fn evaluate_stage2_ram_terminal_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    ram_raf_opening_point: Vec<Fr>,
    ram_output_check_opening_point: Vec<Fr>,
) -> HarnessResult<Stage2RamTerminalOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage2_ram_terminal_openings(
        Stage2BatchProverConfig::new(log_t, log_k, converted.proof.rw_config),
        &witness,
        &mut backend,
        ram_raf_opening_point,
        ram_output_check_opening_point,
    )
    .map_err(|error| {
        core_fixture_error(
            request,
            "evaluate modular stage2 RAM terminal openings",
            error,
        )
    })
}

fn prove_stage3_regular_batch_input_checkpoint(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage3RegularBatchPrefixOutput<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage3 transcript", error))?;
    let stage1_output = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2_output = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let (Stage1Output::Clear(stage1), Stage2Output::Clear(stage2)) = (stage1_output, stage2_output)
    else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage2",
            "expected clear Stage 1 and Stage 2 outputs",
        ));
    };

    derive_stage3_regular_batch_prefix(
        Stage3ProverConfig::new(log_t),
        &stage1,
        &stage2,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "derive modular stage3 batch input claims", error))
}

fn evaluate_stage3_output_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    opening_points: Stage3OutputOpeningPoints,
) -> HarnessResult<Stage3RegularBatchOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage3_output_openings(
        Stage3ProverConfig::new(log_t),
        &witness,
        &mut backend,
        opening_points.shift,
        opening_points.instruction_input,
        opening_points.registers_claim_reduction,
    )
    .map_err(|error| core_fixture_error(request, "evaluate modular stage3 output openings", error))
}

fn prove_stage3_regular_batch_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage3RegularBatchVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage3 transcript", error))?;
    let stage1_output = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2_output = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let (Stage1Output::Clear(stage1), Stage2Output::Clear(stage2)) = (stage1_output, stage2_output)
    else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage2",
            "expected clear Stage 1 and Stage 2 outputs",
        ));
    };
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let input = Stage3ProverInput::new(
        Stage3ProverConfig::new(log_t),
        &checked,
        &stage1,
        &stage2,
        &witness,
    );
    let output = prove_stage3::<Fr, _, _, _, Bn254G1>(input, &mut backend, &mut transcript)
        .map_err(|error| {
            core_fixture_error(request, "prove modular stage3 regular batch", error)
        })?;
    Ok(Stage3RegularBatchVerifierReplay { output })
}

fn prove_stage4_regular_batch_input_checkpoint(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
    ram_val_check_init: Stage4RamValCheckInitialEvaluation<Fr>,
) -> HarnessResult<Stage4RegularBatchPrefixOutput<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage4 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble modular stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage3", error)
    })?;
    let (Stage2Output::Clear(stage2), Stage3Output::Clear(stage3)) = (stage2, stage3) else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage3",
            "expected clear Stage 2 and Stage 3 outputs",
        ));
    };

    derive_stage4_regular_batch_prefix(
        Stage4ProverConfig::new(log_t, log_k, converted.proof.rw_config),
        &stage2,
        &stage3,
        ram_val_check_init,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "derive modular stage4 batch input claims", error))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Harness fixture helper threads core fixture artifacts explicitly."
)]
fn evaluate_stage4_output_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    prefix: &Stage4RegularBatchPrefixOutput<Fr>,
    registers_read_write_opening_point: Vec<Fr>,
    ram_val_check_opening_point: Vec<Fr>,
) -> HarnessResult<Stage4RegularBatchOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage4_output_openings(
        Stage4ProverConfig::new(log_t, log_k, converted.proof.rw_config),
        &witness,
        &mut backend,
        prefix,
        registers_read_write_opening_point,
        ram_val_check_opening_point,
    )
    .map_err(|error| core_fixture_error(request, "evaluate modular stage4 output openings", error))
}

fn prove_stage4_regular_batch_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    ram_val_check_init: Stage4RamValCheckInitialEvaluation<Fr>,
) -> HarnessResult<Stage4RegularBatchVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage4 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble modular stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage3", error)
    })?;
    let (Stage2Output::Clear(stage2), Stage3Output::Clear(stage3)) = (stage2, stage3) else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage3",
            "expected clear Stage 2 and Stage 3 outputs",
        ));
    };
    let config = Stage4ProverConfig::new(log_t, log_k, converted.proof.rw_config);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let input = Stage4ProverInput::new(
        config,
        &checked,
        &stage2,
        &stage3,
        ram_val_check_init,
        &witness,
    );
    let output = prove_stage4::<Fr, _, _, _, Bn254G1>(input, &mut backend, &mut transcript)
        .map_err(|error| {
            core_fixture_error(request, "prove modular stage4 regular batch", error)
        })?;
    Ok(Stage4RegularBatchVerifierReplay { output })
}

fn prove_stage5_regular_batch_input_checkpoint(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage5RegularBatchPrefixOutput<Fr>> {
    let config = stage5_prover_config(request, converted)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage5 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble modular stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble modular stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage4", error)
    })?;
    let (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) = (stage2, stage4) else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage4",
            "expected clear Stage 2 and Stage 4 outputs",
        ));
    };

    derive_stage5_regular_batch_prefix(config, &stage2, &stage4, &mut transcript).map_err(|error| {
        core_fixture_error(request, "derive modular stage5 batch input claims", error)
    })
}

fn prove_stage6_regular_batch_input_checkpoint(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage6RegularBatchPrefixOutput<Fr>> {
    let config = stage6_prover_config(request, converted)?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage6 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble modular stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble modular stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage4", error)
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "assemble modular stage5 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage5", error)
    })?;
    let (
        Stage1Output::Clear(stage1),
        Stage2Output::Clear(stage2),
        Stage3Output::Clear(stage3),
        Stage4Output::Clear(stage4),
        Stage5Output::Clear(stage5),
    ) = (stage1, stage2, stage3, stage4, stage5)
    else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage5",
            "expected clear Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5 outputs",
        ));
    };

    derive_stage6_regular_batch_prefix(
        config,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "derive modular stage6 batch input claims", error))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Harness fixture helper threads core fixture artifacts and Stage 5 opening points explicitly."
)]
fn evaluate_stage5_output_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    instruction_lookup_table_flag_opening_point: Vec<Fr>,
    instruction_ra_opening_points: Vec<Vec<Fr>>,
    instruction_raf_flag_opening_point: Vec<Fr>,
    ram_ra_claim_reduction_opening_point: Vec<Fr>,
    registers_val_evaluation_opening_point: Vec<Fr>,
) -> HarnessResult<Stage5RegularBatchOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage5_output_openings(
        stage5_prover_config(request, converted)?,
        &witness,
        &mut backend,
        instruction_lookup_table_flag_opening_point,
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_opening_point,
        registers_val_evaluation_opening_point,
    )
    .map_err(|error| core_fixture_error(request, "evaluate modular stage5 output openings", error))
}

fn evaluate_stage6_output_opening_checkpoint(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
    expected_stage4: &Stage4ClearOutput<Fr>,
    expected_stage6: &Stage6ClearOutput<Fr>,
) -> HarnessResult<Stage6RegularBatchOutputOpeningClaims<Fr>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(converted.trusted_advice_commitment.is_some())
            .include_untrusted_advice(converted.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });

    evaluate_stage6_output_openings(
        stage6_prover_config(request, converted)?,
        &witness,
        &mut backend,
        expected_stage6
            .batch
            .bytecode_read_raf
            .bytecode_ra_opening_points
            .clone(),
        expected_stage6.batch.booleanity.opening_point.clone(),
        expected_stage6
            .batch
            .ram_hamming_booleanity
            .opening_point
            .clone(),
        expected_stage6
            .batch
            .ram_ra_virtualization
            .ram_ra_opening_points
            .clone(),
        expected_stage6
            .batch
            .instruction_ra_virtualization
            .instruction_ra_opening_points
            .clone(),
        expected_stage6
            .batch
            .inc_claim_reduction
            .opening_point
            .clone(),
        expected_stage4
            .ram_val_check_init
            .advice_contributions
            .iter()
            .find(|contribution| contribution.kind == JoltAdviceKind::Trusted)
            .map(|contribution| contribution.opening_point.clone()),
        expected_stage6
            .batch
            .trusted_advice_cycle_phase
            .as_ref()
            .map(|verified| verified.opening_point.clone()),
        expected_stage4
            .ram_val_check_init
            .advice_contributions
            .iter()
            .find(|contribution| contribution.kind == JoltAdviceKind::Untrusted)
            .map(|contribution| contribution.opening_point.clone()),
        expected_stage6
            .batch
            .untrusted_advice_cycle_phase
            .as_ref()
            .map(|verified| verified.opening_point.clone()),
    )
    .map_err(|error| core_fixture_error(request, "evaluate modular stage6 output openings", error))
}

fn prove_stage5_regular_batch_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage5RegularBatchVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(!trace.device.trusted_advice.is_empty())
            .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage5 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble modular stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble modular stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage4", error)
    })?;
    let (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) = (stage2, stage4) else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage4",
            "expected clear Stage 2 and Stage 4 outputs",
        ));
    };
    let config = stage5_prover_config(request, converted)?;
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let input = Stage5ProverInput::new(config, &checked, &stage2, &stage4, &witness);
    let output = prove_stage5::<Fr, _, _, _, Bn254G1>(input, &mut backend, &mut transcript)
        .map_err(|error| {
            core_fixture_error(request, "prove modular stage5 regular batch", error)
        })?;
    Ok(Stage5RegularBatchVerifierReplay { output })
}

fn prove_stage6_regular_batch_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage6RegularBatchVerifierReplay> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(converted.trusted_advice_commitment.is_some())
            .include_untrusted_advice(converted.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage6 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble modular stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble modular stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage4", error)
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "assemble modular stage5 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance modular transcript through stage5", error)
    })?;
    let (
        Stage1Output::Clear(stage1),
        Stage2Output::Clear(stage2),
        Stage3Output::Clear(stage3),
        Stage4Output::Clear(stage4),
        Stage5Output::Clear(stage5),
    ) = (stage1, stage2, stage3, stage4, stage5)
    else {
        return Err(core_fixture_error(
            request,
            "advance modular transcript through stage5",
            "expected clear Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5 outputs",
        ));
    };

    let config = stage6_prover_config(request, converted)?;
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let input = Stage6ProverInput::new(
        &config, &checked, &stage1, &stage2, &stage3, &stage4, &stage5, &witness,
    );
    let output = prove_stage6::<Fr, _, _, _, _, Bn254G1>(input, &mut backend, &mut transcript)
        .map_err(|error| {
            core_fixture_error(request, "prove modular stage6 regular batch", error)
        })?;
    Ok(Stage6RegularBatchVerifierReplay { output })
}

fn stage5_prover_config(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage5ProverConfig> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let formula_dimensions =
        JoltFormulaDimensions::try_from(converted.proof.one_hot_config.dimensions(
            log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            converted.preprocessing.program.bytecode.code_size,
            converted.proof.ram_K,
        ))
        .map_err(|error| core_fixture_error(request, "derive stage5 formula dimensions", error))?;
    Ok(Stage5ProverConfig::new(
        log_t,
        log_k,
        formula_dimensions.instruction_read_raf,
    ))
}

fn stage6_prover_config(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage6ProverConfig> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let log_k = checked_power_of_two_log(request, converted.proof.ram_K)?;
    let formula_dimensions =
        JoltFormulaDimensions::try_from(converted.proof.one_hot_config.dimensions(
            log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            converted.preprocessing.program.bytecode.code_size,
            converted.proof.ram_K,
        ))
        .map_err(|error| core_fixture_error(request, "derive stage6 formula dimensions", error))?;
    let committed_chunk_bits = converted.proof.one_hot_config.committed_chunk_bits();
    let trusted_advice_layout = converted.trusted_advice_commitment.is_some().then(|| {
        jolt_claims::protocols::jolt::AdviceClaimReductionLayout::balanced(
            converted.proof.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            converted.public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = converted
        .proof
        .untrusted_advice_commitment
        .as_ref()
        .map(|_| {
            jolt_claims::protocols::jolt::AdviceClaimReductionLayout::balanced(
                converted.proof.trace_polynomial_order,
                log_t,
                committed_chunk_bits,
                converted.public_io.memory_layout.max_untrusted_advice_size as usize,
            )
        });
    let entry_bytecode_index = converted
        .preprocessing
        .program
        .bytecode
        .entry_bytecode_index()
        .ok_or_else(|| {
            core_fixture_error(
                request,
                "derive stage6 bytecode context",
                "entry address was not found in bytecode preprocessing",
            )
        })?;
    Ok(Stage6ProverConfig::new(
        log_t,
        log_k,
        committed_chunk_bits,
        formula_dimensions.bytecode_read_raf,
        jolt_claims::protocols::jolt::formulas::booleanity::BooleanityDimensions::new(
            formula_dimensions.ra_layout,
            log_t,
            committed_chunk_bits,
        ),
        formula_dimensions.ram_ra_virtualization,
        formula_dimensions.instruction_ra_virtualization,
        trusted_advice_layout,
        untrusted_advice_layout,
    )
    .with_bytecode_context(
        converted.preprocessing.program.bytecode.bytecode.clone(),
        entry_bytecode_index,
    ))
}

fn verifier_replay_stage2_product_uniskip(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage2: Stage2ProductUniSkipVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage2_uni_skip_first_round_proof = stage2.output.proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage2 product uni-skip",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage2.product_uniskip_output_claim = stage2.output.output_claim;
    Ok(())
}

fn verifier_replay_stage2_regular_batch(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage2: Stage2RegularBatchVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage2_uni_skip_first_round_proof = stage2.output.product_uniskip_proof;
    proof.stages.stage2_sumcheck_proof = stage2.output.regular_batch_proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage2 regular batch",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage2 = stage2.output.claims;
    Ok(())
}

fn verifier_replay_stage3_regular_batch(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage3: Stage3RegularBatchVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage3_sumcheck_proof = stage3.output.stage3_sumcheck_proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage3 regular batch",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage3 = stage3.output.claims;
    Ok(())
}

fn verifier_replay_stage4_regular_batch(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage4: Stage4RegularBatchVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage4_sumcheck_proof = stage4.output.stage4_sumcheck_proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage4 regular batch",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage4 = stage4.output.claims;
    Ok(())
}

fn verifier_replay_stage5_regular_batch(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage5: Stage5RegularBatchVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage5_sumcheck_proof = stage5.output.stage5_sumcheck_proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage5 regular batch",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage5 = stage5.output.claims;
    Ok(())
}

fn verifier_replay_stage6_regular_batch(
    request: &FixtureRequest,
    proof: &mut ConvertedProof,
    stage6: Stage6RegularBatchVerifierReplay,
) -> HarnessResult<()> {
    proof.stages.stage6_sumcheck_proof = stage6.output.stage6_sumcheck_proof;
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage6 regular batch",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage6 = stage6.output.claims;
    Ok(())
}

fn verify_core_stage2_clear(
    request: &FixtureRequest,
    fixture: &CoreVerifierFixture,
) -> HarnessResult<Stage2ClearOutput<Fr>> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize core stage2 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage2", error)
    })?;
    let Stage2Output::Clear(stage2) = stage2 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage2",
            "expected clear Stage 2 output",
        ));
    };
    Ok(stage2)
}

fn verify_core_stage3_clear(
    request: &FixtureRequest,
    fixture: &CoreVerifierFixture,
) -> HarnessResult<Stage3ClearOutput<Fr>> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize core stage3 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble core stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage3", error)
    })?;
    let Stage3Output::Clear(stage3) = stage3 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage3",
            "expected clear Stage 3 output",
        ));
    };
    Ok(stage3)
}

fn verify_core_stage4_clear(
    request: &FixtureRequest,
    fixture: &CoreVerifierFixture,
) -> HarnessResult<Stage4ClearOutput<Fr>> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize core stage4 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble core stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble core stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage4", error)
    })?;
    let Stage4Output::Clear(stage4) = stage4 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage4",
            "expected clear Stage 4 output",
        ));
    };
    Ok(stage4)
}

fn verify_core_stage5_clear(
    request: &FixtureRequest,
    fixture: &CoreVerifierFixture,
) -> HarnessResult<Stage5ClearOutput<Fr>> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize core stage5 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble core stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble core stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage4", error)
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "assemble core stage5 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage5", error)
    })?;
    let Stage5Output::Clear(stage5) = stage5 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage5",
            "expected clear Stage 5 output",
        ));
    };
    Ok(stage5)
}

#[derive(Clone)]
pub struct Stage7RegularBatchInputCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage7RegularBatchPrefixOutput<Fr>,
    pub expected: Stage7RegularBatchInputClaims<Fr>,
    pub expected_hamming_gamma: Fr,
}

fn stage7_prover_config(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage7ProverConfig> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let formula_dimensions =
        JoltFormulaDimensions::try_from(converted.proof.one_hot_config.dimensions(
            log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            converted.preprocessing.program.bytecode.code_size,
            converted.proof.ram_K,
        ))
        .map_err(|error| core_fixture_error(request, "derive stage7 formula dimensions", error))?;
    let committed_chunk_bits = converted.proof.one_hot_config.committed_chunk_bits();
    let hamming_dimensions = HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        committed_chunk_bits,
    );
    let trusted_advice_layout = converted.trusted_advice_commitment.is_some().then(|| {
        jolt_claims::protocols::jolt::AdviceClaimReductionLayout::balanced(
            converted.proof.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            converted.public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = converted
        .proof
        .untrusted_advice_commitment
        .as_ref()
        .map(|_| {
            jolt_claims::protocols::jolt::AdviceClaimReductionLayout::balanced(
                converted.proof.trace_polynomial_order,
                log_t,
                committed_chunk_bits,
                converted.public_io.memory_layout.max_untrusted_advice_size as usize,
            )
        });
    Ok(Stage7ProverConfig::new(
        log_t,
        hamming_dimensions,
        trusted_advice_layout,
        untrusted_advice_layout,
    ))
}

pub fn load_stage7_regular_batch_input_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage7RegularBatchInputCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage7 regular batch input checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let config = stage7_prover_config(request, &fixture)?;

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize core stage7 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance stage7 transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance stage7 transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble stage7 stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance stage7 transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble stage7 stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance stage7 transcript through stage4", error)
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "assemble stage7 stage5 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance stage7 transcript through stage5", error)
    })?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| core_fixture_error(request, "assemble stage7 stage6 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance stage7 transcript through stage6", error)
    })?;

    let Stage6Output::Clear(stage6_clear) = &stage6 else {
        return Err(core_fixture_error(
            request,
            "advance stage7 transcript through stage6",
            "expected clear Stage 6 output",
        ));
    };

    let mut derive_transcript = transcript.clone();
    let modular = derive_stage7_regular_batch_prefix(&config, stage6_clear, &mut derive_transcript)
        .map_err(|error| {
            core_fixture_error(request, "derive modular stage7 batch input claims", error)
        })?;

    let stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage7::inputs::deps(&stage4, &stage6)
            .map_err(|error| core_fixture_error(request, "assemble stage7 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage7", error)
    })?;
    let jolt_verifier::stages::stage7::Stage7Output::Clear(stage7) = stage7 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage7",
            "expected clear Stage 7 output",
        ));
    };

    let expected = Stage7RegularBatchInputClaims {
        hamming_weight_claim_reduction: stage7.batch.hamming_weight_claim_reduction.input_claim,
        trusted_advice_address_phase: stage7
            .batch
            .trusted_advice_address_phase
            .as_ref()
            .map(|verified| verified.input_claim),
        untrusted_advice_address_phase: stage7
            .batch
            .untrusted_advice_address_phase
            .as_ref()
            .map(|verified| verified.input_claim),
    };
    let expected_hamming_gamma = stage7.public.hamming_gamma;

    Ok(Stage7RegularBatchInputCheckpoint {
        fixture,
        modular,
        expected,
        expected_hamming_gamma,
    })
}

#[derive(Clone)]
pub struct Stage7RegularBatchVerifierReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: jolt_verifier::stages::stage7::inputs::Stage7Claims<Fr>,
    pub expected: jolt_verifier::stages::stage7::inputs::Stage7Claims<Fr>,
}

fn prove_stage7_regular_batch_verifier_replay(
    request: &FixtureRequest,
    core_preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    program: &JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage7ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, converted.proof.ram_K, converted.proof.one_hot_config)
            .include_trusted_advice(converted.trusted_advice_commitment.is_some())
            .include_untrusted_advice(converted.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(program, &preprocessing, trace),
    );
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &converted.preprocessing,
        &converted.public_io,
        &converted.proof,
        converted.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize modular stage7 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "advance stage7 replay through stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "advance stage7 replay through stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2).map_err(|error| {
            core_fixture_error(request, "assemble stage7 replay stage3 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "advance stage7 replay through stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3).map_err(|error| {
            core_fixture_error(request, "assemble stage7 replay stage4 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "advance stage7 replay through stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4).map_err(|error| {
            core_fixture_error(request, "assemble stage7 replay stage5 deps", error)
        })?,
    )
    .map_err(|error| core_fixture_error(request, "advance stage7 replay through stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &converted.preprocessing,
        &converted.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| {
                core_fixture_error(request, "assemble stage7 replay stage6 deps", error)
            })?,
    )
    .map_err(|error| core_fixture_error(request, "advance stage7 replay through stage6", error))?;
    let (Stage4Output::Clear(stage4), Stage6Output::Clear(stage6)) = (stage4, stage6) else {
        return Err(core_fixture_error(
            request,
            "advance stage7 replay through stage6",
            "expected clear Stage 4 and Stage 6 outputs",
        ));
    };

    let config = stage7_prover_config(request, converted)?;
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let input = Stage7ProverInput::new(&config, &checked, &stage4, &stage6, &witness);
    prove_stage7::<Fr, _, _, _, Bn254G1>(input, &mut backend, &mut transcript)
        .map_err(|error| core_fixture_error(request, "prove modular stage7", error))
}

pub fn load_stage7_regular_batch_verifier_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage7RegularBatchVerifierReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage7 regular batch verifier replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let expected = stage7_input_checkpoint_expected(request, &fixture)?;
    let output = prove_stage7_regular_batch_verifier_replay(
        request,
        &core_preprocessing,
        &program,
        trace,
        &fixture,
    )?;
    let modular = output.claims.clone();

    fixture.proof.stages.stage7_sumcheck_proof = output.stage7_sumcheck_proof;
    let JoltProofClaims::Clear(claims) = &mut fixture.proof.claims else {
        return Err(core_fixture_error(
            request,
            "verifier_replay modular stage7",
            "converted proof did not contain clear claims",
        ));
    };
    claims.stage7 = output.claims;

    fixture
        .verify()
        .map_err(|error| core_fixture_error(request, "verify stage7 verifier replay", error))?;
    Ok(Stage7RegularBatchVerifierReplayCheckpoint {
        fixture,
        modular,
        expected,
    })
}

fn stage7_input_checkpoint_expected(
    request: &FixtureRequest,
    fixture: &CoreVerifierFixture,
) -> HarnessResult<jolt_verifier::stages::stage7::inputs::Stage7Claims<Fr>> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize stage7 expected transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "stage7 expected stage3 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "stage7 expected stage4 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "stage7 expected stage5 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| core_fixture_error(request, "stage7 expected stage6 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected stage6", error))?;
    let stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage7::inputs::deps(&stage4, &stage6)
            .map_err(|error| core_fixture_error(request, "stage7 expected deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage7 expected verify", error))?;
    let jolt_verifier::stages::stage7::Stage7Output::Clear(stage7) = stage7 else {
        return Err(core_fixture_error(
            request,
            "stage7 expected verify",
            "expected clear Stage 7 output",
        ));
    };
    Ok(stage7.output_claims)
}

#[derive(Clone)]
pub struct Stage8OpeningStructureCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub modular: Stage8OpeningStructure<Fr>,
    pub expected: Stage8OpeningStructure<Fr>,
}

fn stage8_prover_config(
    request: &FixtureRequest,
    converted: &CoreVerifierFixture,
) -> HarnessResult<Stage8ProverConfig> {
    let log_t = checked_power_of_two_log(request, converted.proof.trace_length)?;
    let formula_dimensions =
        JoltFormulaDimensions::try_from(converted.proof.one_hot_config.dimensions(
            log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            converted.preprocessing.program.bytecode.code_size,
            converted.proof.ram_K,
        ))
        .map_err(|error| core_fixture_error(request, "derive stage8 formula dimensions", error))?;
    let committed_chunk_bits = converted.proof.one_hot_config.committed_chunk_bits();
    let trusted_advice_layout = converted.trusted_advice_commitment.is_some().then(|| {
        jolt_claims::protocols::jolt::AdviceClaimReductionLayout::balanced(
            converted.proof.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            converted.public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = converted
        .proof
        .untrusted_advice_commitment
        .as_ref()
        .map(|_| {
            jolt_claims::protocols::jolt::AdviceClaimReductionLayout::balanced(
                converted.proof.trace_polynomial_order,
                log_t,
                committed_chunk_bits,
                converted.public_io.memory_layout.max_untrusted_advice_size as usize,
            )
        });
    Ok(Stage8ProverConfig::new(
        log_t,
        committed_chunk_bits,
        formula_dimensions.ra_layout,
        converted.proof.trace_polynomial_order,
        trusted_advice_layout,
        untrusted_advice_layout,
    ))
}

pub fn load_stage8_opening_structure_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage8OpeningStructureCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage8 opening-structure checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let fixture = convert_fixture(request, generated)?;
    let config = stage8_prover_config(request, &fixture)?;

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize stage8 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "stage8 stage3 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "stage8 stage4 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "stage8 stage5 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| core_fixture_error(request, "stage8 stage6 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage6", error))?;
    let stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage7::inputs::deps(&stage4, &stage6)
            .map_err(|error| core_fixture_error(request, "stage8 stage7 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 stage7", error))?;

    let (
        Stage6Output::Clear(stage6_clear),
        jolt_verifier::stages::stage7::Stage7Output::Clear(stage7_clear),
    ) = (&stage6, &stage7)
    else {
        return Err(core_fixture_error(
            request,
            "stage8 dependency unwrap",
            "expected clear Stage 6 and Stage 7 outputs",
        ));
    };

    let mut derive_transcript = transcript.clone();
    let modular = derive_stage8_opening_structure(
        &config,
        stage6_clear,
        stage7_clear,
        &mut derive_transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "derive modular stage8 opening structure", error)
    })?;

    let stage8 = jolt_verifier::stages::stage8::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        &mut transcript,
        jolt_verifier::stages::stage8::inputs::deps(&stage6, &stage7)
            .map_err(|error| core_fixture_error(request, "stage8 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage8", error)
    })?;
    let jolt_verifier::stages::stage8::outputs::Stage8Output::Clear(stage8) = stage8 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage8",
            "expected clear Stage 8 output",
        ));
    };

    let expected = Stage8OpeningStructure {
        opening_ids: stage8.opening_ids.clone(),
        scaled_opening_values: stage8
            .opening_claims
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect(),
        constraint_coefficients: stage8.constraint_coefficients.clone(),
        opening_point: stage8.opening_point.clone(),
        pcs_opening_point: stage8.pcs_opening_point.clone(),
        joint_claim: stage8.joint_claim,
    };

    Ok(Stage8OpeningStructureCheckpoint {
        fixture,
        modular,
        expected,
    })
}

#[derive(Clone)]
pub struct Stage8RaConstituentCheckpoint {
    pub fixture: CoreVerifierFixture,
    pub evaluated: Vec<Fr>,
    pub expected: Vec<Fr>,
    pub dense_evaluated: Vec<Fr>,
    pub dense_expected: Vec<Fr>,
    pub joint_evaluation: Fr,
    pub joint_claim: Fr,
}

pub fn load_stage8_ra_constituent_checkpoint_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage8RaConstituentCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage8 RA constituent checkpoints currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup: _,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let config = stage8_prover_config(request, &fixture)?;

    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let witness_config =
        JoltVmWitnessConfig::new(log_t, fixture.proof.ram_K, fixture.proof.one_hot_config)
            .include_trusted_advice(fixture.trusted_advice_commitment.is_some())
            .include_untrusted_advice(fixture.proof.untrusted_advice_commitment.is_some());
    let preprocessing = convert_program_preprocessing(&core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize stage8 RA transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "stage8 RA stage3 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "stage8 RA stage4 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "stage8 RA stage5 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| core_fixture_error(request, "stage8 RA stage6 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage6", error))?;
    let stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage7::inputs::deps(&stage4, &stage6)
            .map_err(|error| core_fixture_error(request, "stage8 RA stage7 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 RA stage7", error))?;

    let (
        Stage6Output::Clear(stage6_clear),
        jolt_verifier::stages::stage7::Stage7Output::Clear(stage7_clear),
    ) = (&stage6, &stage7)
    else {
        return Err(core_fixture_error(
            request,
            "stage8 RA dependency unwrap",
            "expected clear Stage 6 and Stage 7 outputs",
        ));
    };

    let (joint_evaluation, joint_claim) = evaluate_stage8_joint_polynomial::<Fr, _, _>(
        &config,
        stage6_clear,
        stage7_clear,
        &witness,
        &mut transcript.clone(),
    )
    .map_err(|error| core_fixture_error(request, "evaluate stage8 joint polynomial", error))?;

    let structure =
        derive_stage8_opening_structure(&config, stage6_clear, stage7_clear, &mut transcript)
            .map_err(|error| core_fixture_error(request, "derive stage8 structure", error))?;

    let evaluated = evaluate_stage8_ra_constituents::<Fr, _>(
        config.layout,
        config.committed_chunk_bits,
        config.log_t,
        config.trace_polynomial_order,
        &witness,
        structure.pcs_opening_point.as_slice(),
    )
    .map_err(|error| core_fixture_error(request, "evaluate stage8 RA constituents", error))?;
    // The RA openings follow `RamInc` and `RdInc` (indices 0,1) in the batch.
    let expected = structure.scaled_opening_values[2..2 + config.layout.total()].to_vec();

    let dense_evaluated = evaluate_stage8_dense_constituents::<Fr, _>(
        config.committed_chunk_bits,
        config.log_t,
        config.trace_polynomial_order,
        &witness,
        structure.pcs_opening_point.as_slice(),
    )
    .map_err(|error| core_fixture_error(request, "evaluate stage8 dense constituents", error))?;
    // `RamInc` and `RdInc` are the first two openings in the batch.
    let dense_expected = structure.scaled_opening_values[0..2].to_vec();

    Ok(Stage8RaConstituentCheckpoint {
        fixture,
        evaluated,
        expected,
        dense_evaluated,
        dense_expected,
        joint_evaluation,
        joint_claim,
    })
}

#[derive(Clone)]
pub struct Stage8JointOpeningReplayCheckpoint {
    pub fixture: CoreVerifierFixture,
}

#[derive(Clone)]
pub struct TopLevelClearProverFixture {
    pub request: FixtureRequest,
    pub fixture: CoreVerifierFixture,
}

pub fn load_top_level_clear_prover_fixture(
    request: &FixtureRequest,
) -> HarnessResult<TopLevelClearProverFixture> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "top-level clear prover fixtures currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment: _,
    } = generated;

    let proof_shape_source: ConvertedProof = proof
        .try_into()
        .map_err(|error| core_fixture_error(request, "convert core proof shape", error))?;
    let log_t = checked_power_of_two_log(request, proof_shape_source.trace_length)?;
    let include_trusted_advice = !trace.device.trusted_advice.is_empty();
    let include_untrusted_advice = !trace.device.untrusted_advice.is_empty();
    let witness_config = JoltVmWitnessConfig::new(
        log_t,
        proof_shape_source.ram_K,
        proof_shape_source.one_hot_config,
    )
    .include_trusted_advice(include_trusted_advice)
    .include_untrusted_advice(include_untrusted_advice);
    let program_preprocessing = convert_program_preprocessing(&core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &program_preprocessing, trace),
    );

    let verifier_preprocessing = convert_preprocessing(&core_preprocessing);
    let prover_preprocessing =
        jolt_prover::JoltProverPreprocessing::new(verifier_preprocessing.clone(), prover_setup);
    let proof_shape = jolt_prover::ProverProofShape::new(
        proof_shape_source.trace_length,
        proof_shape_source.ram_K,
        proof_shape_source.rw_config,
        proof_shape_source.one_hot_config,
        proof_shape_source.trace_polynomial_order,
    );
    let config = jolt_prover::ProverConfig::default().with_proof_shape(proof_shape);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let output = jolt_prover::prove_with_output::<DoryScheme, Pedersen<Bn254G1>, _, _>(
        &prover_preprocessing,
        &public_io,
        &witness,
        config,
        &mut backend,
    )
    .map_err(|error| core_fixture_error(request, "prove top-level modular clear proof", error))?;

    let fixture = CoreVerifierFixture {
        preprocessing: verifier_preprocessing,
        public_io,
        proof: output.proof,
        trusted_advice_commitment: output.trusted_advice_commitment,
    };
    fixture.verify().map_err(|error| {
        core_fixture_error(request, "verify top-level modular clear proof", error)
    })?;

    Ok(TopLevelClearProverFixture {
        request: request.clone(),
        fixture,
    })
}

pub fn load_stage8_joint_opening_replay_fixture(
    request: &FixtureRequest,
) -> HarnessResult<Stage8JointOpeningReplayCheckpoint> {
    if request.feature_mode != FeatureMode::Transparent {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "stage8 joint-opening replays currently cover transparent mode",
        });
    }

    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_fixture_error(request, "lock core fixture generator", error))?;
    let generated = generate_fixture(request)?;
    let GeneratedCoreFixture {
        core_preprocessing,
        prover_setup,
        program,
        trace,
        public_io,
        proof,
        trusted_advice_commitment,
    } = generated;
    let mut fixture = convert_fixture_parts(
        request,
        &core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.as_ref(),
    )?;
    let config = stage8_prover_config(request, &fixture)?;

    let log_t = checked_power_of_two_log(request, fixture.proof.trace_length)?;
    let include_trusted_advice = !trace.device.trusted_advice.is_empty();
    let include_untrusted_advice = !trace.device.untrusted_advice.is_empty();
    let witness_config =
        JoltVmWitnessConfig::new(log_t, fixture.proof.ram_K, fixture.proof.one_hot_config)
            .include_trusted_advice(include_trusted_advice)
            .include_untrusted_advice(include_untrusted_advice);
    let preprocessing = convert_program_preprocessing(&core_preprocessing);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );

    // Stage 0 commit: produce the prover-side commitments and retained opening hints.
    let mut commit_backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 1024,
    });
    let stage0 = prove::<Fr, _, _, DoryScheme>(
        CommitmentStageInput::new(
            &witness,
            &prover_setup,
            CommitmentStageConfig::new(
                config.layout,
                include_trusted_advice,
                include_untrusted_advice,
            ),
            jolt_verifier::JoltProtocolConfig::for_zk(false),
        ),
        &mut commit_backend,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage0 commitments", error))?;

    fixture.proof.commitments = stage0.commitments.clone();
    fixture
        .proof
        .untrusted_advice_commitment
        .clone_from(&stage0.untrusted_advice_commitment);
    fixture
        .trusted_advice_commitment
        .clone_from(&stage0.trusted_advice_commitment);

    let mut commitments_ordered = vec![
        stage0.commitments.ram_inc.clone(),
        stage0.commitments.rd_inc.clone(),
    ];
    commitments_ordered.extend(stage0.commitments.ra.instruction.iter().cloned());
    commitments_ordered.extend(stage0.commitments.ra.bytecode.iter().cloned());
    commitments_ordered.extend(stage0.commitments.ra.ram.iter().cloned());
    if let Some(commitment) = &stage0.trusted_advice_commitment {
        commitments_ordered.push(commitment.clone());
    }
    if let Some(commitment) = &stage0.untrusted_advice_commitment {
        commitments_ordered.push(commitment.clone());
    }

    let opening_hints = &stage0.prover_state.opening_hints;
    let mut hints_ordered = Vec::with_capacity(commitments_ordered.len());
    for polynomial in [
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::RdInc,
    ]
    .into_iter()
    .chain((0..config.layout.instruction()).map(JoltCommittedPolynomial::InstructionRa))
    .chain((0..config.layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa))
    .chain((0..config.layout.ram()).map(JoltCommittedPolynomial::RamRa))
    .chain(
        stage0
            .trusted_advice_commitment
            .as_ref()
            .map(|_| JoltCommittedPolynomial::TrustedAdvice),
    )
    .chain(
        stage0
            .untrusted_advice_commitment
            .as_ref()
            .map(|_| JoltCommittedPolynomial::UntrustedAdvice),
    ) {
        let hint = opening_hints.get(&polynomial).cloned().ok_or_else(|| {
            core_fixture_error(
                request,
                "stage8 joint-opening replay missing opening hint",
                format!("{polynomial:?}"),
            )
        })?;
        hints_ordered.push(hint);
    }

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize stage8 replay transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage1", error))?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage2", error))?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "stage8 replay stage3 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage3", error))?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "stage8 replay stage4 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage4", error))?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "stage8 replay stage5 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage5", error))?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| core_fixture_error(request, "stage8 replay stage6 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage6", error))?;
    let stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage7::inputs::deps(&stage4, &stage6)
            .map_err(|error| core_fixture_error(request, "stage8 replay stage7 deps", error))?,
    )
    .map_err(|error| core_fixture_error(request, "stage8 replay stage7", error))?;

    let (
        Stage6Output::Clear(stage6_clear),
        jolt_verifier::stages::stage7::Stage7Output::Clear(stage7_clear),
    ) = (&stage6, &stage7)
    else {
        return Err(core_fixture_error(
            request,
            "stage8 replay dependency unwrap",
            "expected clear Stage 6 and Stage 7 outputs",
        ));
    };

    let output = prove_stage8::<Fr, DoryScheme, _, _>(
        &config,
        stage6_clear,
        stage7_clear,
        &witness,
        &commitments_ordered,
        hints_ordered,
        &prover_setup,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "prove modular stage8 joint opening", error))?;

    fixture.proof.joint_opening_proof = output.joint_opening_proof;

    fixture.verify().map_err(|error| {
        core_fixture_error(request, "verify stage8 joint-opening replay", error)
    })?;
    Ok(Stage8JointOpeningReplayCheckpoint { fixture })
}

fn verify_core_stage6_clear(
    request: &FixtureRequest,
    fixture: &CoreVerifierFixture,
) -> HarnessResult<Stage6ClearOutput<Fr>> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    let checked = initialize_proof_transcript(
        &fixture.preprocessing,
        &fixture.public_io,
        &fixture.proof,
        fixture.trusted_advice_commitment.as_ref(),
        false,
        &mut transcript,
    )
    .map_err(|error| core_fixture_error(request, "initialize core stage6 transcript", error))?;
    let stage1 = jolt_verifier::stages::stage1::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage1", error)
    })?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1),
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage2", error)
    })?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage3::inputs::deps(&stage1, &stage2)
            .map_err(|error| core_fixture_error(request, "assemble core stage3 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage3", error)
    })?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage4::inputs::deps(&stage2, &stage3)
            .map_err(|error| core_fixture_error(request, "assemble core stage4 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage4", error)
    })?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage5::inputs::deps(&stage2, &stage4)
            .map_err(|error| core_fixture_error(request, "assemble core stage5 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage5", error)
    })?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        &fixture.preprocessing,
        &fixture.proof,
        &mut transcript,
        jolt_verifier::stages::stage6::inputs::deps(&stage1, &stage2, &stage3, &stage4, &stage5)
            .map_err(|error| core_fixture_error(request, "assemble core stage6 deps", error))?,
    )
    .map_err(|error| {
        core_fixture_error(request, "advance core transcript through stage6", error)
    })?;
    let Stage6Output::Clear(stage6) = stage6 else {
        return Err(core_fixture_error(
            request,
            "advance core transcript through stage6",
            "expected clear Stage 6 output",
        ));
    };
    Ok(stage6)
}

fn stage2_regular_batch_input_claims(
    stage2: &Stage2ClearOutput<Fr>,
) -> Stage2RegularBatchInputClaims<Fr> {
    Stage2RegularBatchInputClaims {
        ram_read_write: stage2.batch.ram_read_write.input_claim,
        product_remainder: stage2.batch.product_remainder.input_claim,
        instruction_claim_reduction: stage2.batch.instruction_claim_reduction.input_claim,
        ram_raf_evaluation: stage2.batch.ram_raf_evaluation.input_claim,
        ram_output_check: stage2.batch.ram_output_check.input_claim,
    }
}

fn reference_stage2_regular_batch_prefix<T>(
    config: Stage2BatchProverConfig,
    stage1: &Stage1ClearOutput<Fr>,
    product_uniskip: &Stage2ProductUniSkipOutput<Fr, Bn254G1>,
    transcript: &mut T,
) -> HarnessResult<Stage2RegularBatchPrefixOutput<Fr>>
where
    T: Transcript<Challenge = Fr>,
{
    let read_write_dimensions = config.rw_config.ram_dimensions(config.log_t, config.log_k);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            core_fixture_error(
                &FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent),
                "derive reference Stage 2 regular-batch RAF dimensions",
                error,
            )
        })?;

    let ram_read_write_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let output_address_challenges = (0..config.log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    let gamma2 = instruction_gamma * instruction_gamma;
    let gamma3 = gamma2 * instruction_gamma;
    let gamma4 = gamma3 * instruction_gamma;
    Ok(Stage2RegularBatchPrefixOutput {
        input_claims: Stage2RegularBatchInputClaims {
            ram_read_write: stage1.outer.ram_read_value
                + ram_read_write_gamma * stage1.outer.ram_write_value,
            product_remainder: product_uniskip.output_claim,
            instruction_claim_reduction: stage1.outer.lookup_output
                + instruction_gamma * stage1.outer.left_lookup_operand
                + gamma2 * stage1.outer.right_lookup_operand
                + gamma3 * stage1.outer.left_instruction_input
                + gamma4 * stage1.outer.right_instruction_input,
            ram_raf_evaluation: Fr::pow2(raf_dimensions.phase3_cycle_rounds())
                * stage1.outer.ram_address,
            ram_output_check: Fr::default(),
        },
        ram_read_write_gamma,
        instruction_gamma,
        output_address_challenges,
    })
}

fn reference_stage3_regular_batch_prefix<T>(
    stage1: &Stage1ClearOutput<Fr>,
    stage2: &Stage2ClearOutput<Fr>,
    transcript: &mut T,
) -> Stage3RegularBatchPrefixOutput<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let shift_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let registers_gamma = transcript.challenge_scalar();
    let shift_gamma2 = shift_gamma * shift_gamma;
    let shift_gamma3 = shift_gamma2 * shift_gamma;
    let shift_gamma4 = shift_gamma3 * shift_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;
    let product_left = stage2
        .output_claims
        .product_remainder
        .left_instruction_input;
    let product_right = stage2
        .output_claims
        .product_remainder
        .right_instruction_input;
    Stage3RegularBatchPrefixOutput {
        input_claims: Stage3RegularBatchInputClaims {
            shift: stage1.outer.next_unexpanded_pc
                + shift_gamma * stage1.outer.next_pc
                + shift_gamma2 * stage1.outer.next_is_virtual
                + shift_gamma3 * stage1.outer.next_is_first_in_sequence
                + shift_gamma4
                    * (Fr::pow2(0) - stage2.output_claims.product_remainder.next_is_noop),
            instruction_input: product_right + instruction_gamma * product_left,
            registers_claim_reduction: stage1.outer.rd_write_value
                + registers_gamma * stage1.outer.rs1_value
                + registers_gamma2 * stage1.outer.rs2_value,
        },
        shift_gamma,
        instruction_gamma,
        registers_gamma,
    }
}

fn reference_stage4_regular_batch_prefix<T>(
    stage2: &Stage2ClearOutput<Fr>,
    stage3: &Stage3ClearOutput<Fr>,
    ram_val_check_init: Stage4RamValCheckInitialEvaluation<Fr>,
    transcript: &mut T,
) -> Stage4RegularBatchPrefixOutput<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let registers_gamma = transcript.challenge_scalar();
    transcript.append(&jolt_transcript::LabelWithCount(b"ram_val_check_gamma", 0));
    transcript.append_bytes(&[]);
    let ram_val_check_gamma = transcript.challenge_scalar();
    let registers_gamma2 = registers_gamma * registers_gamma;
    let reduction = &stage3.output_claims.registers_claim_reduction;
    Stage4RegularBatchPrefixOutput {
        input_claims: Stage4RegularBatchInputClaims {
            registers_read_write: reduction.rd_write_value
                + registers_gamma * reduction.rs1_value
                + registers_gamma2 * reduction.rs2_value,
            ram_val_check: stage2.output_claims.ram_read_write.val
                + ram_val_check_gamma * stage2.output_claims.ram_output_check
                - (Fr::pow2(0) + ram_val_check_gamma) * ram_val_check_init.full_eval,
        },
        registers_gamma,
        ram_val_check_gamma,
        ram_val_check_init,
    }
}

fn reference_stage5_regular_batch_prefix<T>(
    stage2: &Stage2ClearOutput<Fr>,
    stage4: &Stage4ClearOutput<Fr>,
    transcript: &mut T,
) -> Stage5RegularBatchPrefixOutput<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let instruction_gamma = transcript.challenge_scalar();
    let instruction_gamma2 = instruction_gamma * instruction_gamma;
    let ram_gamma = transcript.challenge_scalar();
    let ram_gamma2 = ram_gamma * ram_gamma;

    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output;
    let reduced_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);

    Stage5RegularBatchPrefixOutput {
        input_claims: Stage5RegularBatchInputClaims {
            instruction_read_raf: reduced_lookup_output
                + instruction_gamma
                    * stage2
                        .output_claims
                        .instruction_claim_reduction
                        .left_lookup_operand
                + instruction_gamma2
                    * stage2
                        .output_claims
                        .instruction_claim_reduction
                        .right_lookup_operand,
            ram_ra_claim_reduction: stage2.output_claims.ram_raf_evaluation
                + ram_gamma * stage2.output_claims.ram_read_write.ra
                + ram_gamma2 * stage4.output_claims.ram_val_check.ram_ra,
            registers_val_evaluation: stage4.output_claims.registers_read_write.registers_val,
        },
        instruction_gamma,
        ram_gamma,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 batches every prior clear-stage boundary into one prefix."
)]
fn reference_stage6_regular_batch_prefix<T>(
    request: &FixtureRequest,
    config: Stage6ProverConfig,
    stage1: &Stage1ClearOutput<Fr>,
    stage2: &Stage2ClearOutput<Fr>,
    stage3: &Stage3ClearOutput<Fr>,
    stage4: &Stage4ClearOutput<Fr>,
    stage5: &Stage5ClearOutput<Fr>,
    transcript: &mut T,
) -> HarnessResult<Stage6RegularBatchPrefixOutput<Fr>>
where
    T: Transcript<Challenge = Fr>,
{
    if stage3.output_claims.shift.unexpanded_pc
        != stage3.output_claims.instruction_input.unexpanded_pc
    {
        return Err(core_fixture_error(
            request,
            "derive reference stage6 regular batch input claims",
            "Stage 6 bytecode read-RAF unexpanded-PC dependencies disagree",
        ));
    }

    let bytecode_gamma_powers = transcript.challenge_scalar_powers(8);
    let stage1_gammas = transcript.challenge_scalar_powers(2 + CIRCUIT_FLAGS.len());
    let stage2_gammas = transcript.challenge_scalar_powers(4);
    let stage3_gammas = transcript.challenge_scalar_powers(9);
    let stage4_gammas = transcript.challenge_scalar_powers(3);
    let stage5_gammas = transcript.challenge_scalar_powers(
        2 + stage5
            .output_claims
            .instruction_read_raf
            .lookup_table_flags
            .len(),
    );

    let mut booleanity_reference_address = stage5.batch.instruction_read_raf.r_address.clone();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < config.committed_chunk_bits {
        booleanity_reference_address
            .extend(transcript.challenge_vector(
                config.committed_chunk_bits - booleanity_reference_address.len(),
            ));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - config.committed_chunk_bits..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5.batch.instruction_read_raf.r_cycle.clone();
    booleanity_reference_cycle.reverse();
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma == Fr::default() {
        booleanity_gamma = Fr::pow2(0);
    }

    let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
        config
            .instruction_ra_virtualization_dimensions
            .num_virtual_ra_polys(),
    );
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(|| Fr::pow2(0));
    let inc_gamma = transcript.challenge_scalar();

    Ok(Stage6RegularBatchPrefixOutput {
        input_claims: Stage6RegularBatchInputClaims {
            bytecode_read_raf: reference_stage6_bytecode_read_raf_input_claim(
                request,
                stage1,
                stage2,
                stage3,
                stage4,
                stage5,
                &bytecode_gamma_powers,
                &stage1_gammas,
                &stage2_gammas,
                &stage3_gammas,
                &stage4_gammas,
                &stage5_gammas,
            )?,
            booleanity: Fr::default(),
            ram_hamming_booleanity: Fr::default(),
            ram_ra_virtualization: reference_stage6_ram_ra_virtualization_input_claim(
                request, &config, stage5,
            )?,
            instruction_ra_virtualization:
                reference_stage6_instruction_ra_virtualization_input_claim(
                    request,
                    &config,
                    stage5,
                    instruction_ra_gamma,
                )?,
            inc_claim_reduction: reference_stage6_inc_claim_reduction_input_claim(
                request, &config, stage2, stage4, stage5, inc_gamma,
            )?,
            trusted_advice_cycle_phase: reference_stage6_advice_cycle_phase_input_claim(
                request,
                config.trusted_advice_layout.as_ref(),
                stage4,
                JoltAdviceKind::Trusted,
            )?,
            untrusted_advice_cycle_phase: reference_stage6_advice_cycle_phase_input_claim(
                request,
                config.untrusted_advice_layout.as_ref(),
                stage4,
                JoltAdviceKind::Untrusted,
            )?,
        },
        bytecode_gamma_powers,
        stage1_gammas,
        stage2_gammas,
        stage3_gammas,
        stage4_gammas,
        stage5_gammas,
        booleanity_reference_address,
        booleanity_reference_cycle,
        booleanity_gamma,
        instruction_ra_gamma_powers,
        inc_gamma,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "The bytecode input claim batches several prior stage outputs."
)]
fn reference_stage6_bytecode_read_raf_input_claim(
    request: &FixtureRequest,
    stage1: &Stage1ClearOutput<Fr>,
    stage2: &Stage2ClearOutput<Fr>,
    stage3: &Stage3ClearOutput<Fr>,
    stage4: &Stage4ClearOutput<Fr>,
    stage5: &Stage5ClearOutput<Fr>,
    bytecode_gamma_powers: &[Fr],
    stage1_gammas: &[Fr],
    stage2_gammas: &[Fr],
    stage3_gammas: &[Fr],
    stage4_gammas: &[Fr],
    stage5_gammas: &[Fr],
) -> HarnessResult<Fr> {
    reference_require_len(
        request,
        "Stage 6 bytecode gamma powers",
        bytecode_gamma_powers,
        8,
    )?;
    reference_require_len(
        request,
        "Stage 6 stage1 gammas",
        stage1_gammas,
        2 + CIRCUIT_FLAGS.len(),
    )?;
    reference_require_len(request, "Stage 6 stage2 gammas", stage2_gammas, 4)?;
    reference_require_len(request, "Stage 6 stage3 gammas", stage3_gammas, 9)?;
    reference_require_len(request, "Stage 6 stage4 gammas", stage4_gammas, 3)?;
    reference_require_len(
        request,
        "Stage 6 stage5 gammas",
        stage5_gammas,
        2 + stage5
            .output_claims
            .instruction_read_raf
            .lookup_table_flags
            .len(),
    )?;

    let mut stage1_claim = stage1.outer.unexpanded_pc + stage1_gammas[1] * stage1.outer.imm;
    for (index, &flag) in CIRCUIT_FLAGS.iter().enumerate() {
        stage1_claim += stage1_gammas[index + 2]
            * reference_spartan_outer_flag_claim(&stage1.outer.flags, flag);
    }

    let stage2_claim = stage2.output_claims.product_remainder.jump_flag
        + stage2_gammas[1] * stage2.output_claims.product_remainder.branch_flag
        + stage2_gammas[2]
            * stage2
                .output_claims
                .product_remainder
                .write_lookup_output_to_rd
        + stage2_gammas[3] * stage2.output_claims.product_remainder.virtual_instruction;

    let stage3_claim = stage3.output_claims.instruction_input.imm
        + stage3_gammas[1] * stage3.output_claims.shift.unexpanded_pc
        + stage3_gammas[2] * stage3.output_claims.instruction_input.left_operand_is_rs1
        + stage3_gammas[3] * stage3.output_claims.instruction_input.left_operand_is_pc
        + stage3_gammas[4] * stage3.output_claims.instruction_input.right_operand_is_rs2
        + stage3_gammas[5] * stage3.output_claims.instruction_input.right_operand_is_imm
        + stage3_gammas[6] * stage3.output_claims.shift.is_noop
        + stage3_gammas[7] * stage3.output_claims.shift.is_virtual
        + stage3_gammas[8] * stage3.output_claims.shift.is_first_in_sequence;

    let stage4_claim = stage4.output_claims.registers_read_write.rd_wa
        + stage4_gammas[1] * stage4.output_claims.registers_read_write.rs1_ra
        + stage4_gammas[2] * stage4.output_claims.registers_read_write.rs2_ra;

    let mut stage5_claim = stage5.output_claims.registers_val_evaluation.rd_wa
        + stage5_gammas[1]
            * stage5
                .output_claims
                .instruction_read_raf
                .instruction_raf_flag;
    for (index, &flag_claim) in stage5
        .output_claims
        .instruction_read_raf
        .lookup_table_flags
        .iter()
        .enumerate()
    {
        stage5_claim += stage5_gammas[index + 2] * flag_claim;
    }

    Ok(bytecode_gamma_powers[7]
        + stage1_claim
        + bytecode_gamma_powers[1] * stage2_claim
        + bytecode_gamma_powers[2] * stage3_claim
        + bytecode_gamma_powers[3] * stage4_claim
        + bytecode_gamma_powers[4] * stage5_claim
        + bytecode_gamma_powers[5] * stage1.outer.pc
        + bytecode_gamma_powers[6] * stage3.output_claims.shift.pc)
}

fn reference_stage6_ram_ra_virtualization_input_claim(
    request: &FixtureRequest,
    config: &Stage6ProverConfig,
    stage5: &Stage5ClearOutput<Fr>,
) -> HarnessResult<Fr> {
    let claims = ram::ra_virtualization::<Fr>(config.ram_ra_virtualization_dimensions);
    let [ram_ra_reduced] = ram::ra_virtualization_input_openings();
    claims.input.expression().try_evaluate(
        |id| match *id {
            id if id == ram_ra_reduced => Ok(stage5.output_claims.ram_ra_claim_reduction.ram_ra),
            id => Err(core_fixture_error(
                request,
                "derive reference stage6 RAM RA virtualization input claim",
                format!("missing opening {id:?}"),
            )),
        },
        |id| {
            Err(core_fixture_error(
                request,
                "derive reference stage6 RAM RA virtualization input claim",
                format!("unexpected challenge {id:?}"),
            ))
        },
        |id| {
            Err(core_fixture_error(
                request,
                "derive reference stage6 RAM RA virtualization input claim",
                format!("unexpected public input {id:?}"),
            ))
        },
    )
}

fn reference_stage6_instruction_ra_virtualization_input_claim(
    request: &FixtureRequest,
    config: &Stage6ProverConfig,
    stage5: &Stage5ClearOutput<Fr>,
    gamma: Fr,
) -> HarnessResult<Fr> {
    let claims =
        instruction::ra_virtualization::<Fr>(config.instruction_ra_virtualization_dimensions);
    let input_openings = instruction::ra_virtualization_input_openings(
        config.instruction_ra_virtualization_dimensions,
    );
    claims.input.expression().try_evaluate(
        |id| {
            for (index, opening) in input_openings.iter().enumerate() {
                if *id == *opening {
                    return stage5
                        .output_claims
                        .instruction_read_raf
                        .instruction_ra
                        .get(index)
                        .copied()
                        .ok_or_else(|| {
                            core_fixture_error(
                                request,
                                "derive reference stage6 instruction RA virtualization input claim",
                                format!("missing instruction RA input claim {index}"),
                            )
                        });
                }
            }
            Err(core_fixture_error(
                request,
                "derive reference stage6 instruction RA virtualization input claim",
                format!("missing opening {id:?}"),
            ))
        },
        |id| match id {
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::Gamma,
            ) => Ok(gamma),
            _ => Err(core_fixture_error(
                request,
                "derive reference stage6 instruction RA virtualization input claim",
                format!("unexpected challenge {id:?}"),
            )),
        },
        |id| {
            Err(core_fixture_error(
                request,
                "derive reference stage6 instruction RA virtualization input claim",
                format!("unexpected public input {id:?}"),
            ))
        },
    )
}

fn reference_stage6_inc_claim_reduction_input_claim(
    request: &FixtureRequest,
    config: &Stage6ProverConfig,
    stage2: &Stage2ClearOutput<Fr>,
    stage4: &Stage4ClearOutput<Fr>,
    stage5: &Stage5ClearOutput<Fr>,
    gamma: Fr,
) -> HarnessResult<Fr> {
    let claims = increments::claim_reduction::<Fr>(config.trace_dimensions());
    let [ram_inc_read_write, ram_inc_val_check, rd_inc_read_write, rd_inc_val_evaluation] =
        increments::claim_reduction_input_openings();
    claims.input.expression().try_evaluate(
        |id| match *id {
            id if id == ram_inc_read_write => Ok(stage2.output_claims.ram_read_write.inc),
            id if id == ram_inc_val_check => Ok(stage4.output_claims.ram_val_check.ram_inc),
            id if id == rd_inc_read_write => Ok(stage4.output_claims.registers_read_write.rd_inc),
            id if id == rd_inc_val_evaluation => {
                Ok(stage5.output_claims.registers_val_evaluation.rd_inc)
            }
            id => Err(core_fixture_error(
                request,
                "derive reference stage6 increment claim-reduction input claim",
                format!("missing opening {id:?}"),
            )),
        },
        |id| match id {
            JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => Ok(gamma),
            _ => Err(core_fixture_error(
                request,
                "derive reference stage6 increment claim-reduction input claim",
                format!("unexpected challenge {id:?}"),
            )),
        },
        |id| {
            Err(core_fixture_error(
                request,
                "derive reference stage6 increment claim-reduction input claim",
                format!("unexpected public input {id:?}"),
            ))
        },
    )
}

fn reference_stage6_advice_cycle_phase_input_claim(
    request: &FixtureRequest,
    layout: Option<&AdviceClaimReductionLayout>,
    stage4: &Stage4ClearOutput<Fr>,
    kind: JoltAdviceKind,
) -> HarnessResult<Option<Fr>> {
    let Some(layout) = layout else {
        return Ok(None);
    };
    let claim = advice::cycle_phase::<Fr>(kind, layout.dimensions());
    let [advice_input] = advice::cycle_phase_input_openings(kind);
    claim
        .input
        .expression()
        .try_evaluate(
            |id| match *id {
                id if id == advice_input => stage4
                    .ram_val_check_init
                    .advice_contributions
                    .iter()
                    .find(|contribution| contribution.kind == kind)
                    .map(|contribution| contribution.opening_claim)
                    .ok_or_else(|| {
                        core_fixture_error(
                            request,
                            "derive reference stage6 advice cycle-phase input claim",
                            format!("missing {kind:?} advice contribution"),
                        )
                    }),
                id => Err(core_fixture_error(
                    request,
                    "derive reference stage6 advice cycle-phase input claim",
                    format!("missing opening {id:?}"),
                )),
            },
            |id| {
                Err(core_fixture_error(
                    request,
                    "derive reference stage6 advice cycle-phase input claim",
                    format!("unexpected challenge {id:?}"),
                ))
            },
            |id| {
                Err(core_fixture_error(
                    request,
                    "derive reference stage6 advice cycle-phase input claim",
                    format!("unexpected public input {id:?}"),
                ))
            },
        )
        .map(Some)
}

fn reference_require_len(
    request: &FixtureRequest,
    label: &'static str,
    actual: &[Fr],
    expected: usize,
) -> HarnessResult<()> {
    if actual.len() < expected {
        return Err(core_fixture_error(
            request,
            "derive reference stage6 bytecode read-RAF input claim",
            format!(
                "{label} length {} is less than required {expected}",
                actual.len()
            ),
        ));
    }
    Ok(())
}

fn reference_spartan_outer_flag_claim(
    claims: &SpartanOuterFlagClaims<Fr>,
    flag: CircuitFlags,
) -> Fr {
    match flag {
        CircuitFlags::AddOperands => claims.add_operands,
        CircuitFlags::SubtractOperands => claims.subtract_operands,
        CircuitFlags::MultiplyOperands => claims.multiply_operands,
        CircuitFlags::Load => claims.load,
        CircuitFlags::Store => claims.store,
        CircuitFlags::Jump => claims.jump,
        CircuitFlags::WriteLookupOutputToRD => claims.write_lookup_output_to_rd,
        CircuitFlags::VirtualInstruction => claims.virtual_instruction,
        CircuitFlags::Assert => claims.assert,
        CircuitFlags::DoNotUpdateUnexpandedPC => claims.do_not_update_unexpanded_pc,
        CircuitFlags::Advice => claims.advice,
        CircuitFlags::IsCompressed => claims.is_compressed,
        CircuitFlags::IsFirstInSequence => claims.is_first_in_sequence,
        CircuitFlags::IsLastInSequence => claims.is_last_in_sequence,
    }
}

fn reference_stage7_regular_batch_prefix<T>(
    config: &Stage7ProverConfig,
    stage6: &Stage6ClearOutput<Fr>,
    transcript: &mut T,
) -> Stage7RegularBatchPrefixOutput<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let hamming_gamma = transcript.challenge_scalar();
    let layout = config.hamming_dimensions.layout;
    let total = layout.total();
    let one = Fr::pow2(0);

    let ram_hamming_weight = stage6
        .output_claims
        .ram_hamming_booleanity
        .ram_hamming_weight;
    let booleanity = stage6
        .output_claims
        .booleanity
        .instruction_ra
        .iter()
        .chain(&stage6.output_claims.booleanity.bytecode_ra)
        .chain(&stage6.output_claims.booleanity.ram_ra)
        .copied()
        .collect::<Vec<_>>();
    let virtualization = stage6
        .output_claims
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter()
        .chain(&stage6.output_claims.bytecode_read_raf.bytecode_ra)
        .chain(&stage6.output_claims.ram_ra_virtualization.ram_ra)
        .copied()
        .collect::<Vec<_>>();

    let first_ram = layout.instruction() + layout.bytecode();
    let mut terms = Vec::with_capacity(3 * total);
    let mut gamma_power = one;
    for index in 0..total {
        let hamming_weight = if index < first_ram {
            one
        } else {
            ram_hamming_weight
        };
        terms.push(gamma_power * hamming_weight);
        gamma_power *= hamming_gamma;
        terms.push(gamma_power * booleanity[index]);
        gamma_power *= hamming_gamma;
        terms.push(gamma_power * virtualization[index]);
        gamma_power *= hamming_gamma;
    }
    let hamming_weight_claim_reduction = terms.into_iter().sum::<Fr>();

    Stage7RegularBatchPrefixOutput {
        input_claims: Stage7RegularBatchInputClaims {
            hamming_weight_claim_reduction,
            trusted_advice_address_phase: None,
            untrusted_advice_address_phase: None,
        },
        hamming_gamma,
    }
}

fn first_stage2_batch_input_mismatch(
    expected: &Stage2RegularBatchInputClaims<Fr>,
    actual: &Stage2RegularBatchInputClaims<Fr>,
) -> Option<String> {
    [
        (
            "ram_read_write",
            expected.ram_read_write,
            actual.ram_read_write,
        ),
        (
            "product_remainder",
            expected.product_remainder,
            actual.product_remainder,
        ),
        (
            "instruction_claim_reduction",
            expected.instruction_claim_reduction,
            actual.instruction_claim_reduction,
        ),
        (
            "ram_raf_evaluation",
            expected.ram_raf_evaluation,
            actual.ram_raf_evaluation,
        ),
        (
            "ram_output_check",
            expected.ram_output_check,
            actual.ram_output_check,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_stage2_batch_output_mismatch(
    expected: &Stage2BatchOutputOpeningClaims<Fr>,
    actual: &Stage2BatchOutputOpeningClaims<Fr>,
) -> Option<String> {
    let scalar_mismatch = [
        (
            "ram_read_write.val",
            expected.ram_read_write.val,
            actual.ram_read_write.val,
        ),
        (
            "ram_read_write.ra",
            expected.ram_read_write.ra,
            actual.ram_read_write.ra,
        ),
        (
            "ram_read_write.inc",
            expected.ram_read_write.inc,
            actual.ram_read_write.inc,
        ),
        (
            "product_remainder.left_instruction_input",
            expected.product_remainder.left_instruction_input,
            actual.product_remainder.left_instruction_input,
        ),
        (
            "product_remainder.right_instruction_input",
            expected.product_remainder.right_instruction_input,
            actual.product_remainder.right_instruction_input,
        ),
        (
            "product_remainder.jump_flag",
            expected.product_remainder.jump_flag,
            actual.product_remainder.jump_flag,
        ),
        (
            "product_remainder.write_lookup_output_to_rd",
            expected.product_remainder.write_lookup_output_to_rd,
            actual.product_remainder.write_lookup_output_to_rd,
        ),
        (
            "product_remainder.lookup_output",
            expected.product_remainder.lookup_output,
            actual.product_remainder.lookup_output,
        ),
        (
            "product_remainder.branch_flag",
            expected.product_remainder.branch_flag,
            actual.product_remainder.branch_flag,
        ),
        (
            "product_remainder.next_is_noop",
            expected.product_remainder.next_is_noop,
            actual.product_remainder.next_is_noop,
        ),
        (
            "product_remainder.virtual_instruction",
            expected.product_remainder.virtual_instruction,
            actual.product_remainder.virtual_instruction,
        ),
        (
            "instruction_claim_reduction.left_lookup_operand",
            expected.instruction_claim_reduction.left_lookup_operand,
            actual.instruction_claim_reduction.left_lookup_operand,
        ),
        (
            "instruction_claim_reduction.right_lookup_operand",
            expected.instruction_claim_reduction.right_lookup_operand,
            actual.instruction_claim_reduction.right_lookup_operand,
        ),
        (
            "ram_raf_evaluation",
            expected.ram_raf_evaluation,
            actual.ram_raf_evaluation,
        ),
        (
            "ram_output_check",
            expected.ram_output_check,
            actual.ram_output_check,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    });
    if scalar_mismatch.is_some() {
        return scalar_mismatch;
    }

    [
        (
            "instruction_claim_reduction.lookup_output",
            expected.instruction_claim_reduction.lookup_output,
            actual.instruction_claim_reduction.lookup_output,
        ),
        (
            "instruction_claim_reduction.left_instruction_input",
            expected.instruction_claim_reduction.left_instruction_input,
            actual.instruction_claim_reduction.left_instruction_input,
        ),
        (
            "instruction_claim_reduction.right_instruction_input",
            expected.instruction_claim_reduction.right_instruction_input,
            actual.instruction_claim_reduction.right_instruction_input,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn stage3_regular_batch_input_claims(
    stage3: &Stage3ClearOutput<Fr>,
) -> Stage3RegularBatchInputClaims<Fr> {
    Stage3RegularBatchInputClaims {
        shift: stage3.batch.shift.input_claim,
        instruction_input: stage3.batch.instruction_input.input_claim,
        registers_claim_reduction: stage3.batch.registers_claim_reduction.input_claim,
    }
}

fn stage3_output_opening_claims(
    stage3: &Stage3ClearOutput<Fr>,
) -> Stage3RegularBatchOutputOpeningClaims<Fr> {
    Stage3RegularBatchOutputOpeningClaims {
        shift: stage3.output_claims.shift.clone(),
        instruction_input: stage3.output_claims.instruction_input.clone(),
        registers_claim_reduction: stage3.output_claims.registers_claim_reduction.clone(),
    }
}

fn stage4_regular_batch_input_claims(
    stage4: &Stage4ClearOutput<Fr>,
) -> Stage4RegularBatchInputClaims<Fr> {
    Stage4RegularBatchInputClaims {
        registers_read_write: stage4.batch.registers_read_write.input_claim,
        ram_val_check: stage4.batch.ram_val_check.input_claim,
    }
}

fn stage4_output_opening_claims(
    stage4: &Stage4ClearOutput<Fr>,
) -> Stage4RegularBatchOutputOpeningClaims<Fr> {
    Stage4RegularBatchOutputOpeningClaims {
        advice: stage4.output_claims.advice.clone(),
        registers_read_write: stage4.output_claims.registers_read_write.clone(),
        ram_val_check: stage4.output_claims.ram_val_check.clone(),
    }
}

fn stage4_ram_val_check_initial_evaluation(
    stage4: &Stage4ClearOutput<Fr>,
) -> Stage4RamValCheckInitialEvaluation<Fr> {
    Stage4RamValCheckInitialEvaluation {
        public_eval: stage4.ram_val_check_init.public_eval,
        advice_contributions: stage4
            .ram_val_check_init
            .advice_contributions
            .iter()
            .map(|contribution| Stage4RamValCheckAdviceContribution {
                kind: contribution.kind,
                selector: contribution.selector,
                opening_claim: contribution.opening_claim,
                opening_point: contribution.opening_point.clone(),
            })
            .collect(),
        full_eval: stage4.ram_val_check_init.full_eval,
    }
}

fn stage5_regular_batch_input_claims(
    stage5: &Stage5ClearOutput<Fr>,
) -> Stage5RegularBatchInputClaims<Fr> {
    Stage5RegularBatchInputClaims {
        instruction_read_raf: stage5.batch.instruction_read_raf.input_claim,
        ram_ra_claim_reduction: stage5.batch.ram_ra_claim_reduction.input_claim,
        registers_val_evaluation: stage5.batch.registers_val_evaluation.input_claim,
    }
}

fn stage5_output_opening_claims(
    stage5: &Stage5ClearOutput<Fr>,
) -> Stage5RegularBatchOutputOpeningClaims<Fr> {
    stage5.output_claims.clone()
}

fn stage6_regular_batch_input_claims(
    stage6: &Stage6ClearOutput<Fr>,
) -> Stage6RegularBatchInputClaims<Fr> {
    Stage6RegularBatchInputClaims {
        bytecode_read_raf: stage6.batch.bytecode_read_raf.input_claim,
        booleanity: stage6.batch.booleanity.input_claim,
        ram_hamming_booleanity: stage6.batch.ram_hamming_booleanity.input_claim,
        ram_ra_virtualization: stage6.batch.ram_ra_virtualization.input_claim,
        instruction_ra_virtualization: stage6.batch.instruction_ra_virtualization.input_claim,
        inc_claim_reduction: stage6.batch.inc_claim_reduction.input_claim,
        trusted_advice_cycle_phase: stage6
            .batch
            .trusted_advice_cycle_phase
            .as_ref()
            .map(|verified| verified.input_claim),
        untrusted_advice_cycle_phase: stage6
            .batch
            .untrusted_advice_cycle_phase
            .as_ref()
            .map(|verified| verified.input_claim),
    }
}

fn stage6_output_opening_claims(
    stage6: &Stage6ClearOutput<Fr>,
) -> Stage6RegularBatchOutputOpeningClaims<Fr> {
    stage6.output_claims.clone()
}

fn compare_stage6_regular_batch_input_checkpoint(
    request: &FixtureRequest,
    expected: &Stage6ClearOutput<Fr>,
    actual: &Stage6RegularBatchPrefixOutput<Fr>,
) -> HarnessResult<()> {
    let expected_input_claims = stage6_regular_batch_input_claims(expected);
    if actual.input_claims != expected_input_claims {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch input claims",
            first_stage6_batch_input_mismatch(&expected_input_claims, &actual.input_claims)
                .unwrap_or_else(|| "Stage 6 input claims differ".to_owned()),
        ));
    }
    if actual.bytecode_gamma_powers != expected.public.bytecode_gamma_powers {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "bytecode gamma powers differ",
        ));
    }
    if actual.stage1_gammas != expected.public.stage1_gammas {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "stage1 gammas differ",
        ));
    }
    if actual.stage2_gammas != expected.public.stage2_gammas {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "stage2 gammas differ",
        ));
    }
    if actual.stage3_gammas != expected.public.stage3_gammas {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "stage3 gammas differ",
        ));
    }
    if actual.stage4_gammas != expected.public.stage4_gammas {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "stage4 gammas differ",
        ));
    }
    if actual.stage5_gammas != expected.public.stage5_gammas {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "stage5 gammas differ",
        ));
    }
    if actual.booleanity_reference_address != expected.public.booleanity_reference_address {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "booleanity reference address differs",
        ));
    }
    if actual.booleanity_reference_cycle != expected.public.booleanity_reference_cycle {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "booleanity reference cycle differs",
        ));
    }
    if actual.booleanity_gamma != expected.public.booleanity_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "booleanity gamma differs",
        ));
    }
    if actual.instruction_ra_gamma_powers != expected.public.instruction_ra_gamma_powers {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "instruction RA gamma powers differ",
        ));
    }
    if actual.inc_gamma != expected.public.inc_gamma {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch transcript challenges",
            "increment gamma differs",
        ));
    }
    Ok(())
}

fn compare_stage3_regular_batch_verifier_replay(
    request: &FixtureRequest,
    expected: &Stage3ClearOutput<Fr>,
    actual: &Stage3ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
) -> HarnessResult<()> {
    let expected_openings = stage3_output_opening_claims(expected);
    let actual_openings = Stage3RegularBatchOutputOpeningClaims {
        shift: actual.claims.shift.clone(),
        instruction_input: actual.claims.instruction_input.clone(),
        registers_claim_reduction: actual.claims.registers_claim_reduction.clone(),
    };
    if actual_openings != expected_openings {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch verifier_replay output openings",
            first_stage3_output_opening_mismatch(&expected_openings, &actual_openings)
                .unwrap_or_else(|| "Stage 3 output openings differ".to_owned()),
        ));
    }
    if actual.verifier_output.public.shift_gamma != expected.public.shift_gamma
        || actual.verifier_output.public.instruction_gamma != expected.public.instruction_gamma
        || actual.verifier_output.public.registers_gamma != expected.public.registers_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch verifier_replay transcript",
            "Stage 3 gammas differ from core",
        ));
    }
    if actual.verifier_output.batch.sumcheck_point.as_slice()
        != expected.batch.sumcheck_point.as_slice()
        || actual.verifier_output.batch.expected_final_claim != expected.batch.expected_final_claim
    {
        return Err(core_fixture_error(
            request,
            "compare stage3 regular batch verifier_replay batch",
            "Stage 3 batch point or final claim differs from core",
        ));
    }
    Ok(())
}

fn compare_stage4_regular_batch_verifier_replay(
    request: &FixtureRequest,
    expected: &Stage4ClearOutput<Fr>,
    actual: &Stage4ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
) -> HarnessResult<()> {
    let expected_openings = stage4_output_opening_claims(expected);
    let actual_openings = Stage4RegularBatchOutputOpeningClaims {
        advice: actual.claims.advice.clone(),
        registers_read_write: actual.claims.registers_read_write.clone(),
        ram_val_check: actual.claims.ram_val_check.clone(),
    };
    if actual_openings != expected_openings {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch verifier_replay output openings",
            first_stage4_output_opening_mismatch(&expected_openings, &actual_openings)
                .unwrap_or_else(|| "Stage 4 output openings differ".to_owned()),
        ));
    }
    if actual.verifier_output.public.registers_gamma != expected.public.registers_gamma
        || actual.verifier_output.public.ram_val_check_gamma != expected.public.ram_val_check_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch verifier_replay transcript",
            "Stage 4 gammas differ from core",
        ));
    }
    if actual.verifier_output.ram_val_check_init != expected.ram_val_check_init {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch verifier_replay initial evaluation",
            "Stage 4 RAM value-check initial evaluation differs from core",
        ));
    }
    if actual.verifier_output.batch.sumcheck_point.as_slice()
        != expected.batch.sumcheck_point.as_slice()
        || actual.verifier_output.batch.expected_final_claim != expected.batch.expected_final_claim
    {
        return Err(core_fixture_error(
            request,
            "compare stage4 regular batch verifier_replay batch",
            "Stage 4 batch point or final claim differs from core",
        ));
    }
    Ok(())
}

fn compare_stage5_regular_batch_verifier_replay(
    request: &FixtureRequest,
    expected: &Stage5ClearOutput<Fr>,
    actual: &Stage5ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
) -> HarnessResult<()> {
    let expected_openings = stage5_output_opening_claims(expected);
    if actual.claims != expected_openings {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch verifier_replay output openings",
            first_stage5_output_opening_mismatch(&expected_openings, &actual.claims)
                .unwrap_or_else(|| "Stage 5 output openings differ".to_owned()),
        ));
    }
    if actual.verifier_output.public.instruction_gamma != expected.public.instruction_gamma
        || actual.verifier_output.public.ram_gamma != expected.public.ram_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch verifier_replay transcript",
            "Stage 5 gammas differ from core",
        ));
    }
    if actual.verifier_output.batch.sumcheck_point.as_slice()
        != expected.batch.sumcheck_point.as_slice()
        || actual.verifier_output.batch.expected_final_claim != expected.batch.expected_final_claim
    {
        return Err(core_fixture_error(
            request,
            "compare stage5 regular batch verifier_replay batch",
            "Stage 5 batch point or final claim differs from core",
        ));
    }
    Ok(())
}

fn compare_stage6_regular_batch_verifier_replay(
    request: &FixtureRequest,
    expected: &Stage6ClearOutput<Fr>,
    expected_proof: &jolt_sumcheck::SumcheckProof<Fr, Bn254G1>,
    actual: &Stage6ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, Bn254G1>>,
) -> HarnessResult<()> {
    let expected_openings = stage6_output_opening_claims(expected);
    if actual.claims != expected_openings {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch verifier_replay output openings",
            first_stage6_output_opening_mismatch(&expected_openings, &actual.claims)
                .unwrap_or_else(|| "Stage 6 output openings differ".to_owned()),
        ));
    }
    if let Some(mismatch) =
        first_sumcheck_round_mismatch(expected_proof, &actual.stage6_sumcheck_proof)
    {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch verifier_replay proof",
            mismatch,
        ));
    }
    if actual.verifier_output.public.inc_gamma != expected.public.inc_gamma
        || actual.verifier_output.public.booleanity_gamma != expected.public.booleanity_gamma
    {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch verifier_replay transcript",
            "Stage 6 gammas differ from core",
        ));
    }
    if actual.verifier_output.batch.sumcheck_point.as_slice()
        != expected.batch.sumcheck_point.as_slice()
        || actual.verifier_output.batch.expected_final_claim != expected.batch.expected_final_claim
    {
        return Err(core_fixture_error(
            request,
            "compare stage6 regular batch verifier_replay batch",
            "Stage 6 batch point or final claim differs from core",
        ));
    }
    Ok(())
}

fn first_stage3_batch_input_mismatch(
    expected: &Stage3RegularBatchInputClaims<Fr>,
    actual: &Stage3RegularBatchInputClaims<Fr>,
) -> Option<String> {
    [
        ("shift", expected.shift, actual.shift),
        (
            "instruction_input",
            expected.instruction_input,
            actual.instruction_input,
        ),
        (
            "registers_claim_reduction",
            expected.registers_claim_reduction,
            actual.registers_claim_reduction,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_stage4_batch_input_mismatch(
    expected: &Stage4RegularBatchInputClaims<Fr>,
    actual: &Stage4RegularBatchInputClaims<Fr>,
) -> Option<String> {
    [
        (
            "registers_read_write",
            expected.registers_read_write,
            actual.registers_read_write,
        ),
        (
            "ram_val_check",
            expected.ram_val_check,
            actual.ram_val_check,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_stage5_batch_input_mismatch(
    expected: &Stage5RegularBatchInputClaims<Fr>,
    actual: &Stage5RegularBatchInputClaims<Fr>,
) -> Option<String> {
    [
        (
            "instruction_read_raf",
            expected.instruction_read_raf,
            actual.instruction_read_raf,
        ),
        (
            "ram_ra_claim_reduction",
            expected.ram_ra_claim_reduction,
            actual.ram_ra_claim_reduction,
        ),
        (
            "registers_val_evaluation",
            expected.registers_val_evaluation,
            actual.registers_val_evaluation,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_stage6_batch_input_mismatch(
    expected: &Stage6RegularBatchInputClaims<Fr>,
    actual: &Stage6RegularBatchInputClaims<Fr>,
) -> Option<String> {
    [
        (
            "bytecode_read_raf",
            expected.bytecode_read_raf,
            actual.bytecode_read_raf,
        ),
        ("booleanity", expected.booleanity, actual.booleanity),
        (
            "ram_hamming_booleanity",
            expected.ram_hamming_booleanity,
            actual.ram_hamming_booleanity,
        ),
        (
            "ram_ra_virtualization",
            expected.ram_ra_virtualization,
            actual.ram_ra_virtualization,
        ),
        (
            "instruction_ra_virtualization",
            expected.instruction_ra_virtualization,
            actual.instruction_ra_virtualization,
        ),
        (
            "inc_claim_reduction",
            expected.inc_claim_reduction,
            actual.inc_claim_reduction,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
    .or_else(|| {
        (expected.trusted_advice_cycle_phase != actual.trusted_advice_cycle_phase).then(|| {
            format!(
                "trusted_advice_cycle_phase differs: expected {:?}, got {:?}",
                expected.trusted_advice_cycle_phase, actual.trusted_advice_cycle_phase
            )
        })
    })
    .or_else(|| {
        (expected.untrusted_advice_cycle_phase != actual.untrusted_advice_cycle_phase).then(|| {
            format!(
                "untrusted_advice_cycle_phase differs: expected {:?}, got {:?}",
                expected.untrusted_advice_cycle_phase, actual.untrusted_advice_cycle_phase
            )
        })
    })
}

fn first_stage6_output_opening_mismatch(
    expected: &Stage6RegularBatchOutputOpeningClaims<Fr>,
    actual: &Stage6RegularBatchOutputOpeningClaims<Fr>,
) -> Option<String> {
    if expected.bytecode_read_raf.bytecode_ra != actual.bytecode_read_raf.bytecode_ra {
        return Some(format!(
            "bytecode_read_raf.bytecode_ra differs: expected {:?}, got {:?}",
            expected.bytecode_read_raf.bytecode_ra, actual.bytecode_read_raf.bytecode_ra
        ));
    }
    if expected.booleanity.instruction_ra != actual.booleanity.instruction_ra {
        return Some(format!(
            "booleanity.instruction_ra differs: expected {:?}, got {:?}",
            expected.booleanity.instruction_ra, actual.booleanity.instruction_ra
        ));
    }
    if expected.booleanity.bytecode_ra != actual.booleanity.bytecode_ra {
        return Some(format!(
            "booleanity.bytecode_ra differs: expected {:?}, got {:?}",
            expected.booleanity.bytecode_ra, actual.booleanity.bytecode_ra
        ));
    }
    if expected.booleanity.ram_ra != actual.booleanity.ram_ra {
        return Some(format!(
            "booleanity.ram_ra differs: expected {:?}, got {:?}",
            expected.booleanity.ram_ra, actual.booleanity.ram_ra
        ));
    }
    if expected.ram_ra_virtualization.ram_ra != actual.ram_ra_virtualization.ram_ra {
        return Some(format!(
            "ram_ra_virtualization.ram_ra differs: expected {:?}, got {:?}",
            expected.ram_ra_virtualization.ram_ra, actual.ram_ra_virtualization.ram_ra
        ));
    }
    if expected
        .instruction_ra_virtualization
        .committed_instruction_ra
        != actual
            .instruction_ra_virtualization
            .committed_instruction_ra
    {
        return Some(format!(
            "instruction_ra_virtualization.committed_instruction_ra differs: expected {:?}, got {:?}",
            expected
                .instruction_ra_virtualization
                .committed_instruction_ra,
            actual
                .instruction_ra_virtualization
                .committed_instruction_ra
        ));
    }
    [
        (
            "ram_hamming_booleanity.ram_hamming_weight",
            expected.ram_hamming_booleanity.ram_hamming_weight,
            actual.ram_hamming_booleanity.ram_hamming_weight,
        ),
        (
            "inc_claim_reduction.ram_inc",
            expected.inc_claim_reduction.ram_inc,
            actual.inc_claim_reduction.ram_inc,
        ),
        (
            "inc_claim_reduction.rd_inc",
            expected.inc_claim_reduction.rd_inc,
            actual.inc_claim_reduction.rd_inc,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
    .or_else(|| {
        (expected.advice_cycle_phase.trusted != actual.advice_cycle_phase.trusted).then(|| {
            format!(
                "advice_cycle_phase.trusted differs: expected {:?}, got {:?}",
                expected.advice_cycle_phase.trusted, actual.advice_cycle_phase.trusted
            )
        })
    })
    .or_else(|| {
        (expected.advice_cycle_phase.untrusted != actual.advice_cycle_phase.untrusted).then(|| {
            format!(
                "advice_cycle_phase.untrusted differs: expected {:?}, got {:?}",
                expected.advice_cycle_phase.untrusted, actual.advice_cycle_phase.untrusted
            )
        })
    })
}

fn first_stage5_output_opening_mismatch(
    expected: &Stage5RegularBatchOutputOpeningClaims<Fr>,
    actual: &Stage5RegularBatchOutputOpeningClaims<Fr>,
) -> Option<String> {
    if expected.instruction_read_raf.lookup_table_flags
        != actual.instruction_read_raf.lookup_table_flags
    {
        return Some(format!(
            "instruction_read_raf.lookup_table_flags differs: expected {:?}, got {:?}",
            expected.instruction_read_raf.lookup_table_flags,
            actual.instruction_read_raf.lookup_table_flags
        ));
    }
    if expected.instruction_read_raf.instruction_ra != actual.instruction_read_raf.instruction_ra {
        return Some(format!(
            "instruction_read_raf.instruction_ra differs: expected {:?}, got {:?}",
            expected.instruction_read_raf.instruction_ra,
            actual.instruction_read_raf.instruction_ra
        ));
    }
    [
        (
            "instruction_read_raf.instruction_raf_flag",
            expected.instruction_read_raf.instruction_raf_flag,
            actual.instruction_read_raf.instruction_raf_flag,
        ),
        (
            "ram_ra_claim_reduction.ram_ra",
            expected.ram_ra_claim_reduction.ram_ra,
            actual.ram_ra_claim_reduction.ram_ra,
        ),
        (
            "registers_val_evaluation.rd_inc",
            expected.registers_val_evaluation.rd_inc,
            actual.registers_val_evaluation.rd_inc,
        ),
        (
            "registers_val_evaluation.rd_wa",
            expected.registers_val_evaluation.rd_wa,
            actual.registers_val_evaluation.rd_wa,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_stage4_output_opening_mismatch(
    expected: &Stage4RegularBatchOutputOpeningClaims<Fr>,
    actual: &Stage4RegularBatchOutputOpeningClaims<Fr>,
) -> Option<String> {
    if expected.advice.untrusted != actual.advice.untrusted {
        return Some(format!(
            "advice.untrusted differs: expected {:?}, got {:?}",
            expected.advice.untrusted, actual.advice.untrusted
        ));
    }
    if expected.advice.trusted != actual.advice.trusted {
        return Some(format!(
            "advice.trusted differs: expected {:?}, got {:?}",
            expected.advice.trusted, actual.advice.trusted
        ));
    }
    [
        (
            "registers_read_write.registers_val",
            expected.registers_read_write.registers_val,
            actual.registers_read_write.registers_val,
        ),
        (
            "registers_read_write.rs1_ra",
            expected.registers_read_write.rs1_ra,
            actual.registers_read_write.rs1_ra,
        ),
        (
            "registers_read_write.rs2_ra",
            expected.registers_read_write.rs2_ra,
            actual.registers_read_write.rs2_ra,
        ),
        (
            "registers_read_write.rd_wa",
            expected.registers_read_write.rd_wa,
            actual.registers_read_write.rd_wa,
        ),
        (
            "registers_read_write.rd_inc",
            expected.registers_read_write.rd_inc,
            actual.registers_read_write.rd_inc,
        ),
        (
            "ram_val_check.ram_ra",
            expected.ram_val_check.ram_ra,
            actual.ram_val_check.ram_ra,
        ),
        (
            "ram_val_check.ram_inc",
            expected.ram_val_check.ram_inc,
            actual.ram_val_check.ram_inc,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_stage3_output_opening_mismatch(
    expected: &Stage3RegularBatchOutputOpeningClaims<Fr>,
    actual: &Stage3RegularBatchOutputOpeningClaims<Fr>,
) -> Option<String> {
    [
        (
            "shift.unexpanded_pc",
            expected.shift.unexpanded_pc,
            actual.shift.unexpanded_pc,
        ),
        ("shift.pc", expected.shift.pc, actual.shift.pc),
        (
            "shift.is_virtual",
            expected.shift.is_virtual,
            actual.shift.is_virtual,
        ),
        (
            "shift.is_first_in_sequence",
            expected.shift.is_first_in_sequence,
            actual.shift.is_first_in_sequence,
        ),
        (
            "shift.is_noop",
            expected.shift.is_noop,
            actual.shift.is_noop,
        ),
        (
            "instruction_input.left_operand_is_rs1",
            expected.instruction_input.left_operand_is_rs1,
            actual.instruction_input.left_operand_is_rs1,
        ),
        (
            "instruction_input.rs1_value",
            expected.instruction_input.rs1_value,
            actual.instruction_input.rs1_value,
        ),
        (
            "instruction_input.left_operand_is_pc",
            expected.instruction_input.left_operand_is_pc,
            actual.instruction_input.left_operand_is_pc,
        ),
        (
            "instruction_input.unexpanded_pc",
            expected.instruction_input.unexpanded_pc,
            actual.instruction_input.unexpanded_pc,
        ),
        (
            "instruction_input.right_operand_is_rs2",
            expected.instruction_input.right_operand_is_rs2,
            actual.instruction_input.right_operand_is_rs2,
        ),
        (
            "instruction_input.rs2_value",
            expected.instruction_input.rs2_value,
            actual.instruction_input.rs2_value,
        ),
        (
            "instruction_input.right_operand_is_imm",
            expected.instruction_input.right_operand_is_imm,
            actual.instruction_input.right_operand_is_imm,
        ),
        (
            "instruction_input.imm",
            expected.instruction_input.imm,
            actual.instruction_input.imm,
        ),
        (
            "registers_claim_reduction.rd_write_value",
            expected.registers_claim_reduction.rd_write_value,
            actual.registers_claim_reduction.rd_write_value,
        ),
        (
            "registers_claim_reduction.rs1_value",
            expected.registers_claim_reduction.rs1_value,
            actual.registers_claim_reduction.rs1_value,
        ),
        (
            "registers_claim_reduction.rs2_value",
            expected.registers_claim_reduction.rs2_value,
            actual.registers_claim_reduction.rs2_value,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_product_remainder_opening_mismatch(
    expected: &ProductRemainderOutputOpeningClaims<Fr>,
    actual: &ProductRemainderOutputOpeningClaims<Fr>,
) -> Option<String> {
    [
        (
            "left_instruction_input",
            expected.left_instruction_input,
            actual.left_instruction_input,
        ),
        (
            "right_instruction_input",
            expected.right_instruction_input,
            actual.right_instruction_input,
        ),
        ("jump_flag", expected.jump_flag, actual.jump_flag),
        (
            "write_lookup_output_to_rd",
            expected.write_lookup_output_to_rd,
            actual.write_lookup_output_to_rd,
        ),
        (
            "lookup_output",
            expected.lookup_output,
            actual.lookup_output,
        ),
        ("branch_flag", expected.branch_flag, actual.branch_flag),
        ("next_is_noop", expected.next_is_noop, actual.next_is_noop),
        (
            "virtual_instruction",
            expected.virtual_instruction,
            actual.virtual_instruction,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_ram_read_write_opening_mismatch(
    expected: &RamReadWriteOutputOpeningClaims<Fr>,
    actual: &RamReadWriteOutputOpeningClaims<Fr>,
) -> Option<String> {
    [
        ("val", expected.val, actual.val),
        ("ra", expected.ra, actual.ra),
        ("inc", expected.inc, actual.inc),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_ram_terminal_opening_mismatch(
    expected: &Stage2RamTerminalOutputOpeningClaims<Fr>,
    actual: &Stage2RamTerminalOutputOpeningClaims<Fr>,
) -> Option<String> {
    [
        (
            "ram_raf_evaluation",
            expected.ram_raf_evaluation,
            actual.ram_raf_evaluation,
        ),
        (
            "ram_output_check",
            expected.ram_output_check,
            actual.ram_output_check,
        ),
    ]
    .into_iter()
    .find(|(_, expected, actual)| expected != actual)
    .map(|(name, expected, actual)| {
        format!("{name} differs: expected {expected:?}, got {actual:?}")
    })
}

fn first_instruction_claim_opening_mismatch(
    expected: &InstructionClaimReductionOutputOpeningClaims<Fr>,
    actual: &InstructionClaimReductionOutputOpeningClaims<Fr>,
) -> Option<String> {
    if expected.lookup_output != actual.lookup_output {
        return Some(format!(
            "lookup_output differs: expected {:?}, got {:?}",
            expected.lookup_output, actual.lookup_output
        ));
    }
    for (name, expected, actual) in [
        (
            "left_lookup_operand",
            expected.left_lookup_operand,
            actual.left_lookup_operand,
        ),
        (
            "right_lookup_operand",
            expected.right_lookup_operand,
            actual.right_lookup_operand,
        ),
    ] {
        if expected != actual {
            return Some(format!(
                "{name} differs: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    if expected.left_instruction_input != actual.left_instruction_input {
        return Some(format!(
            "left_instruction_input differs: expected {:?}, got {:?}",
            expected.left_instruction_input, actual.left_instruction_input
        ));
    }
    if expected.right_instruction_input != actual.right_instruction_input {
        return Some(format!(
            "right_instruction_input differs: expected {:?}, got {:?}",
            expected.right_instruction_input, actual.right_instruction_input
        ));
    }
    None
}

fn core_stage1_spartan_outer_claims(
    request: &FixtureRequest,
    proof: &CoreProof,
) -> HarnessResult<(Vec<Fr>, Vec<Stage1R1csInputClaim<Fr>>, Fr)> {
    let openings = &proof.opening_claims.0;
    let mut point = None;
    let mut claims = Vec::with_capacity(SPARTAN_OUTER_R1CS_INPUTS.len());
    for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().copied().enumerate() {
        let core_variable = core_stage1_variable(request, variable)?;
        let id =
            core_opening::OpeningId::virt(core_variable, core_opening::SumcheckId::SpartanOuter);
        let (opening_point, claim) = openings.get(&id).ok_or_else(|| {
            core_fixture_error(
                request,
                "read core stage1 opening claims",
                format!("missing opening for {variable:?}"),
            )
        })?;
        let opening_point = opening_point
            .r
            .iter()
            .map(|challenge| {
                let core_field: CoreField = challenge.into();
                core_field.into_verifier_field()
            })
            .collect::<Vec<Fr>>();
        match &point {
            Some(expected) if *expected != opening_point => {
                return Err(core_fixture_error(
                    request,
                    "read core stage1 opening claims",
                    format!("opening point for {variable:?} differs from prior stage1 point"),
                ));
            }
            Some(_) => {}
            None => point = Some(opening_point),
        }
        claims.push(Stage1R1csInputClaim {
            variable,
            slot: r1cs_input_slot(index),
            value: (*claim).into_verifier_field(),
        });
    }
    let uniskip_id = core_opening::OpeningId::virt(
        core_witness::VirtualPolynomial::UnivariateSkip,
        core_opening::SumcheckId::SpartanOuter,
    );
    let (_, uniskip_output_claim) = openings.get(&uniskip_id).ok_or_else(|| {
        core_fixture_error(
            request,
            "read core stage1 opening claims",
            "missing univariate-skip output opening",
        )
    })?;
    let point = point.ok_or_else(|| {
        core_fixture_error(
            request,
            "read core stage1 opening claims",
            "missing Spartan outer R1CS openings",
        )
    })?;
    Ok((point, claims, (*uniskip_output_claim).into_verifier_field()))
}

fn first_stage1_claim_mismatch(
    expected: &[Stage1R1csInputClaim<Fr>],
    actual: &[Stage1R1csInputClaim<Fr>],
) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "claim count differs: expected {}, got {}",
            expected.len(),
            actual.len()
        ));
    }
    expected
        .iter()
        .zip(actual)
        .find(|(expected, actual)| expected != actual)
        .map(|(expected, actual)| {
            format!(
                "{:?} differs: expected {:?}, got {:?}",
                expected.variable, expected.value, actual.value
            )
        })
}

fn clear_stage1_claims<'a>(
    request: &FixtureRequest,
    fixture: &'a CoreVerifierFixture,
) -> HarnessResult<&'a Stage1Claims<Fr>> {
    match &fixture.proof.claims {
        JoltProofClaims::Clear(claims) => Ok(&claims.stage1),
        JoltProofClaims::Zk { .. } => Err(core_fixture_error(
            request,
            "read converted stage1 claims",
            "expected transparent proof claims",
        )),
    }
}

fn core_stage1_variable(
    request: &FixtureRequest,
    variable: JoltVirtualPolynomial,
) -> HarnessResult<core_witness::VirtualPolynomial> {
    Ok(match variable {
        JoltVirtualPolynomial::LeftInstructionInput => {
            core_witness::VirtualPolynomial::LeftInstructionInput
        }
        JoltVirtualPolynomial::RightInstructionInput => {
            core_witness::VirtualPolynomial::RightInstructionInput
        }
        JoltVirtualPolynomial::Product => core_witness::VirtualPolynomial::Product,
        JoltVirtualPolynomial::ShouldBranch => core_witness::VirtualPolynomial::ShouldBranch,
        JoltVirtualPolynomial::PC => core_witness::VirtualPolynomial::PC,
        JoltVirtualPolynomial::UnexpandedPC => core_witness::VirtualPolynomial::UnexpandedPC,
        JoltVirtualPolynomial::Imm => core_witness::VirtualPolynomial::Imm,
        JoltVirtualPolynomial::RamAddress => core_witness::VirtualPolynomial::RamAddress,
        JoltVirtualPolynomial::Rs1Value => core_witness::VirtualPolynomial::Rs1Value,
        JoltVirtualPolynomial::Rs2Value => core_witness::VirtualPolynomial::Rs2Value,
        JoltVirtualPolynomial::RdWriteValue => core_witness::VirtualPolynomial::RdWriteValue,
        JoltVirtualPolynomial::RamReadValue => core_witness::VirtualPolynomial::RamReadValue,
        JoltVirtualPolynomial::RamWriteValue => core_witness::VirtualPolynomial::RamWriteValue,
        JoltVirtualPolynomial::LeftLookupOperand => {
            core_witness::VirtualPolynomial::LeftLookupOperand
        }
        JoltVirtualPolynomial::RightLookupOperand => {
            core_witness::VirtualPolynomial::RightLookupOperand
        }
        JoltVirtualPolynomial::NextUnexpandedPC => {
            core_witness::VirtualPolynomial::NextUnexpandedPC
        }
        JoltVirtualPolynomial::NextPC => core_witness::VirtualPolynomial::NextPC,
        JoltVirtualPolynomial::NextIsVirtual => core_witness::VirtualPolynomial::NextIsVirtual,
        JoltVirtualPolynomial::NextIsFirstInSequence => {
            core_witness::VirtualPolynomial::NextIsFirstInSequence
        }
        JoltVirtualPolynomial::LookupOutput => core_witness::VirtualPolynomial::LookupOutput,
        JoltVirtualPolynomial::ShouldJump => core_witness::VirtualPolynomial::ShouldJump,
        JoltVirtualPolynomial::OpFlags(flag) => {
            core_witness::VirtualPolynomial::OpFlags(core_circuit_flag(flag))
        }
        _ => {
            return Err(core_fixture_error(
                request,
                "map stage1 R1CS variable",
                format!("unsupported Spartan outer variable {variable:?}"),
            ));
        }
    })
}

const fn core_circuit_flag(flag: CircuitFlags) -> core_instruction::CircuitFlags {
    match flag {
        CircuitFlags::AddOperands => core_instruction::CircuitFlags::AddOperands,
        CircuitFlags::SubtractOperands => core_instruction::CircuitFlags::SubtractOperands,
        CircuitFlags::MultiplyOperands => core_instruction::CircuitFlags::MultiplyOperands,
        CircuitFlags::Load => core_instruction::CircuitFlags::Load,
        CircuitFlags::Store => core_instruction::CircuitFlags::Store,
        CircuitFlags::Jump => core_instruction::CircuitFlags::Jump,
        CircuitFlags::WriteLookupOutputToRD => {
            core_instruction::CircuitFlags::WriteLookupOutputToRD
        }
        CircuitFlags::VirtualInstruction => core_instruction::CircuitFlags::VirtualInstruction,
        CircuitFlags::Assert => core_instruction::CircuitFlags::Assert,
        CircuitFlags::DoNotUpdateUnexpandedPC => {
            core_instruction::CircuitFlags::DoNotUpdateUnexpandedPC
        }
        CircuitFlags::Advice => core_instruction::CircuitFlags::Advice,
        CircuitFlags::IsCompressed => core_instruction::CircuitFlags::IsCompressed,
        CircuitFlags::IsFirstInSequence => core_instruction::CircuitFlags::IsFirstInSequence,
        CircuitFlags::IsLastInSequence => core_instruction::CircuitFlags::IsLastInSequence,
    }
}

fn checked_power_of_two_log(request: &FixtureRequest, value: usize) -> HarnessResult<usize> {
    if value == 0 || !value.is_power_of_two() {
        return Err(core_fixture_error(
            request,
            "derive stage0 witness dimensions",
            format!("trace length {value} is not a nonzero power of two"),
        ));
    }
    Ok(value.trailing_zeros() as usize)
}

fn convert_one_hot_config(
    config: jolt_core::zkvm::config::OneHotConfig,
) -> jolt_claims::protocols::jolt::JoltOneHotConfig {
    jolt_claims::protocols::jolt::JoltOneHotConfig {
        log_k_chunk: config.log_k_chunk,
        lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
    }
}

fn convert_program_preprocessing(
    preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing {
        bytecode: preprocessing.shared.bytecode.as_ref().clone(),
        ram: preprocessing.shared.ram.clone(),
        memory_layout: preprocessing.shared.memory_layout.clone(),
        max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
    }
}

struct GeneratedCoreFixture {
    core_preprocessing: CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    prover_setup: DoryProverSetup,
    program: JoltProgram,
    trace: TraceOutput<OwnedTrace>,
    public_io: JoltDevice,
    proof: CoreProof,
    trusted_advice_commitment: Option<CoreCommitment>,
}

fn core_fixture_error(
    request: &FixtureRequest,
    context: &'static str,
    error: impl ToString,
) -> HarnessError {
    HarnessError::CoreFixture {
        fixture: request.kind,
        context,
        reason: error.to_string(),
    }
}
