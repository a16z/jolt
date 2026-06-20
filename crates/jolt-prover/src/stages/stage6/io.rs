use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity::BooleanityDimensions, bytecode::BytecodeReadRafDimensions,
        dimensions::TraceDimensions, instruction::InstructionRaVirtualizationDimensions,
        ram::RamRaVirtualizationDimensions,
    },
    AdviceClaimReductionLayout,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_riscv::JoltInstructionRow;
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage6::outputs::Stage6PublicOutput;
use jolt_verifier::stages::stage6::{
    outputs::{Stage6ClearOutput, Stage6OutputClaims},
    Stage6BatchInputClaims, Stage6TranscriptChallenges,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use jolt_verifier::CheckedInputs;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BytecodeContext {
    pub rows: Vec<JoltInstructionRow>,
    pub entry_bytecode_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub committed_chunk_bits: usize,
    pub bytecode_read_raf_dimensions: BytecodeReadRafDimensions,
    pub booleanity_dimensions: BooleanityDimensions,
    pub ram_ra_virtualization_dimensions: RamRaVirtualizationDimensions,
    pub instruction_ra_virtualization_dimensions: InstructionRaVirtualizationDimensions,
    pub trusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub bytecode_context: Option<Stage6BytecodeContext>,
}

impl Stage6ProverConfig {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 dimensions are derived by distinct protocol subsystems."
    )]
    pub const fn new(
        log_t: usize,
        log_k: usize,
        committed_chunk_bits: usize,
        bytecode_read_raf_dimensions: BytecodeReadRafDimensions,
        booleanity_dimensions: BooleanityDimensions,
        ram_ra_virtualization_dimensions: RamRaVirtualizationDimensions,
        instruction_ra_virtualization_dimensions: InstructionRaVirtualizationDimensions,
        trusted_advice_layout: Option<AdviceClaimReductionLayout>,
        untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    ) -> Self {
        Self {
            log_t,
            log_k,
            committed_chunk_bits,
            bytecode_read_raf_dimensions,
            booleanity_dimensions,
            ram_ra_virtualization_dimensions,
            instruction_ra_virtualization_dimensions,
            trusted_advice_layout,
            untrusted_advice_layout,
            bytecode_context: None,
        }
    }

    pub fn with_bytecode_context(
        mut self,
        rows: Vec<JoltInstructionRow>,
        entry_bytecode_index: usize,
    ) -> Self {
        self.bytecode_context = Some(Stage6BytecodeContext {
            rows,
            entry_bytecode_index,
        });
        self
    }

    pub const fn trace_dimensions(&self) -> TraceDimensions {
        TraceDimensions::new(self.log_t)
    }
}

#[derive(Clone, Debug)]
pub struct Stage6ProverInput<'a, F: Field, W> {
    pub config: &'a Stage6ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub stage5: &'a Stage5ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage6ProverInput<'a, F, W> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 consumes every prior clear stage output."
    )]
    pub const fn new(
        config: &'a Stage6ProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            witness,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage6RegularBatchPrefixOutput<F: Field> {
    pub(super) input_claims: Stage6BatchInputClaims<F>,
    pub(super) challenges: Stage6TranscriptChallenges<F>,
}

impl<F: Field> Stage6RegularBatchPrefixOutput<F> {
    pub(super) const fn new(
        input_claims: Stage6BatchInputClaims<F>,
        challenges: Stage6TranscriptChallenges<F>,
    ) -> Self {
        Self {
            input_claims,
            challenges,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ProofComponent<F: Field, Proof> {
    pub stage6a_sumcheck_proof: Proof,
    pub stage6b_sumcheck_proof: Proof,
    pub claims: Stage6OutputClaims<F>,
    pub verifier_output: Stage6ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage6a_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage6b_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage6PublicOutput<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage6ClearOutput<F>,
    /// ZK committed witness for stage 6a (address phase).
    pub(crate) stage6a_committed_witness: CommittedSumcheckWitness<F>,
    /// ZK committed witness for stage 6b (cycle phase).
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage6RegularBatchProofOutput<F: Field, Proof> {
    pub(super) proof: SumcheckProof<F, Proof>,
    pub(super) verifier_output: Stage6ClearOutput<F>,
}
