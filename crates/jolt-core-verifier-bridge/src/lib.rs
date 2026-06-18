//! Conversion from current `jolt-core` proof artifacts into `jolt-verifier` artifacts.

use ark_serialize::CanonicalDeserialize;
#[cfg(not(feature = "zk"))]
use jolt_verifier::compat::claims::clear_claims_from_native;
use jolt_verifier::{
    config::JoltProtocolConfig,
    preprocessing::{
        CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
    },
    proof::{
        JoltCommitments, JoltProof, JoltProofClaims, JoltRaCommitments, JoltStageProofs,
        TracePolynomialOrder,
    },
    VerifierError,
};

#[cfg(not(feature = "zk"))]
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitmentOpening;
use jolt_crypto::{
    Bn254G1, Bn254GT, Commitment as VerifierCommitment, HomomorphicCommitment, Pedersen,
    PedersenSetup, VectorCommitment as VerifierVectorCommitment,
};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field as VerifierFieldTrait, Fr as VerifierFr};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use jolt_sumcheck::{
    ClearProof, ClearSumcheckProof, CommittedOutputClaims, CommittedRound, CommittedSumcheckProof,
    CompressedSumcheckProof, SumcheckProof,
};

#[cfg(feature = "zk")]
use jolt_blindfold::BlindFoldProof as VerifierBlindFoldProof;
use jolt_core::{
    curve::{Bn254Curve, JoltCurve},
    field::JoltField,
    poly::commitment::{
        commitment_scheme::CommitmentScheme,
        dory::{ArkGT as CoreDoryCommitment, DoryCommitmentScheme, DoryLayout as CoreDoryLayout},
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig as CoreOneHotConfig, ReadWriteConfig as CoreReadWriteConfig},
        program::ProgramPreprocessing as CoreProgramPreprocessing,
        proof_serialization::JoltProof as CoreJoltProof,
        verifier::JoltVerifierPreprocessing as CoreJoltVerifierPreprocessing,
    },
};
#[cfg(feature = "zk")]
use jolt_core::{
    poly::commitment::hyrax::HyraxOpeningProof, poly::unipoly::CompressedUniPoly,
    subprotocols::blindfold::BlindFoldProof as CoreBlindFoldProof,
};
#[cfg(not(feature = "zk"))]
use jolt_core::{
    poly::opening_proof as core_opening,
    zkvm::{instruction as core_instruction, witness as core_witness},
};

pub type RV64IMACProof =
    ImportedCoreProof<jolt_core::ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>;
pub type VerifierTrustedAdviceCommitment = DoryCommitment;
pub type CoreRV64IMACVerifierPreprocessing =
    CoreJoltVerifierPreprocessing<jolt_core::ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>;

#[derive(Debug)]
pub enum CoreImportError {
    Deserialize(ark_serialize::SerializationError),
    Verifier(VerifierError),
}

impl From<ark_serialize::SerializationError> for CoreImportError {
    fn from(error: ark_serialize::SerializationError) -> Self {
        Self::Deserialize(error)
    }
}

impl From<VerifierError> for CoreImportError {
    fn from(error: VerifierError) -> Self {
        Self::Verifier(error)
    }
}

pub trait CoreFieldBridge: JoltField {
    type VerifierField: VerifierFieldTrait;

    fn into_verifier_field(self) -> Self::VerifierField;
}

impl CoreFieldBridge for jolt_core::ark_bn254::Fr {
    type VerifierField = VerifierFr;

    fn into_verifier_field(self) -> Self::VerifierField {
        self.into()
    }
}

pub trait CorePcsBridge<F>: CommitmentScheme<Field = F>
where
    F: CoreFieldBridge,
{
    type VerifierPcs: VerifierCommitmentScheme<Field = F::VerifierField>;

    fn commitment_into_verifier(
        commitment: Self::Commitment,
    ) -> <Self::VerifierPcs as VerifierCommitment>::Output;

    fn proof_into_verifier(
        proof: Self::Proof,
    ) -> <Self::VerifierPcs as VerifierCommitmentScheme>::Proof;

    fn verifier_setup_into_verifier(
        setup: Self::VerifierSetup,
    ) -> <Self::VerifierPcs as VerifierCommitmentScheme>::VerifierSetup;
}

impl CorePcsBridge<jolt_core::ark_bn254::Fr> for DoryCommitmentScheme {
    type VerifierPcs = DoryScheme;

    fn commitment_into_verifier(commitment: Self::Commitment) -> DoryCommitment {
        DoryCommitment(core_dory_commitment_into_verifier(&commitment))
    }

    fn proof_into_verifier(proof: Self::Proof) -> DoryProof {
        DoryProof(proof)
    }

    fn verifier_setup_into_verifier(setup: Self::VerifierSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(setup)
    }
}

pub trait CoreCurveBridge<F>: JoltCurve<F = F> + Sized
where
    F: CoreFieldBridge,
{
    type VerifierVectorCommitment: VerifierVectorCommitment<
        Field = F::VerifierField,
        Output = Self::VerifierRoundCommitment,
    >;
    type VerifierRoundCommitment: Copy
        + HomomorphicCommitment<F::VerifierField>
        + jolt_transcript::AppendToTranscript
        + serde::Serialize
        + serde::de::DeserializeOwned;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment;

    fn vc_setup_from_core_blindfold(
        setup: &jolt_core::poly::commitment::pedersen::PedersenGenerators<Self>,
    ) -> <Self::VerifierVectorCommitment as VerifierVectorCommitment>::Setup;
}

impl CoreCurveBridge<jolt_core::ark_bn254::Fr> for Bn254Curve {
    type VerifierVectorCommitment = Pedersen<Bn254G1>;
    type VerifierRoundCommitment = Bn254G1;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment {
        commitment.0.into()
    }

    fn vc_setup_from_core_blindfold(
        setup: &jolt_core::poly::commitment::pedersen::PedersenGenerators<Self>,
    ) -> <Self::VerifierVectorCommitment as VerifierVectorCommitment>::Setup {
        PedersenSetup::new(
            setup
                .message_generators
                .iter()
                .copied()
                .map(Self::g1_into_verifier)
                .collect(),
            Self::g1_into_verifier(setup.blinding_generator),
        )
    }
}

pub type ImportedCoreProof<F, C, PCS> = JoltProof<
    <PCS as CorePcsBridge<F>>::VerifierPcs,
    <C as CoreCurveBridge<F>>::VerifierVectorCommitment,
>;

pub type ImportedCorePreprocessing<F, C, PCS> = JoltVerifierPreprocessing<
    <PCS as CorePcsBridge<F>>::VerifierPcs,
    <C as CoreCurveBridge<F>>::VerifierVectorCommitment,
>;

pub fn verifier_preprocessing_from_core<F, C, PCS>(
    preprocessing: &CoreJoltVerifierPreprocessing<F, C, PCS>,
) -> ImportedCorePreprocessing<F, C, PCS>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
{
    let program = match &preprocessing.shared.program {
        CoreProgramPreprocessing::Full(full) => {
            ProgramPreprocessing::Full(JoltProgramPreprocessing {
                bytecode: full.bytecode.as_ref().clone(),
                ram: full.ram.clone(),
                memory_layout: preprocessing.shared.memory_layout.clone(),
                max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
            })
        }
        CoreProgramPreprocessing::Committed(committed) => {
            ProgramPreprocessing::Committed(CommittedProgramPreprocessing {
                meta: ProgramMetadata {
                    entry_address: committed.meta.entry_address,
                    min_bytecode_address: committed.meta.min_bytecode_address,
                    entry_bytecode_index: committed.meta.entry_bytecode_index,
                    program_image_len_words: committed.meta.program_image_len_words,
                    bytecode_len: committed.meta.bytecode_len,
                },
                memory_layout: preprocessing.shared.memory_layout.clone(),
                max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
                bytecode_chunk_commitments: committed
                    .bytecode_commitments
                    .commitments
                    .iter()
                    .cloned()
                    .map(PCS::commitment_into_verifier)
                    .collect(),
                program_image_commitment: PCS::commitment_into_verifier(
                    committed
                        .program_commitments
                        .program_image_commitment
                        .clone(),
                ),
            })
        }
    };

    JoltVerifierPreprocessing::new(
        program,
        preprocessing.shared.digest(),
        PCS::verifier_setup_into_verifier(preprocessing.generators.clone()),
        convert_vc_setup::<F, C, PCS>(preprocessing),
    )
}

pub fn commitment_into_verifier<F, PCS>(
    commitment: PCS::Commitment,
) -> <PCS::VerifierPcs as VerifierCommitment>::Output
where
    F: CoreFieldBridge,
    PCS: CorePcsBridge<F>,
{
    PCS::commitment_into_verifier(commitment)
}

pub fn verifier_preprocessing_from_core_bytes(
    bytes: &[u8],
) -> Result<
    ImportedCorePreprocessing<jolt_core::ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>,
    ark_serialize::SerializationError,
> {
    let preprocessing = CoreRV64IMACVerifierPreprocessing::deserialize_compressed(bytes)?;
    Ok(verifier_preprocessing_from_core(&preprocessing))
}

pub fn proof_into_verifier_from_core_bytes(bytes: &[u8]) -> Result<RV64IMACProof, CoreImportError> {
    let proof = jolt_core::zkvm::RV64IMACProof::deserialize_compressed(bytes)?;
    Ok(proof_into_verifier(proof)?)
}

pub fn trusted_advice_commitment_from_core_bytes(
    bytes: &[u8],
) -> Result<VerifierTrustedAdviceCommitment, ark_serialize::SerializationError> {
    let commitment =
        <DoryCommitmentScheme as CommitmentScheme>::Commitment::deserialize_compressed(bytes)?;
    Ok(<DoryCommitmentScheme as CorePcsBridge<
        jolt_core::ark_bn254::Fr,
    >>::commitment_into_verifier(commitment))
}

pub fn verify_rv64imac(
    preprocessing: &ImportedCorePreprocessing<
        jolt_core::ark_bn254::Fr,
        Bn254Curve,
        DoryCommitmentScheme,
    >,
    public_io: &common::jolt_device::JoltDevice,
    proof: &RV64IMACProof,
    trusted_advice_commitment: Option<&VerifierTrustedAdviceCommitment>,
    zk: bool,
) -> Result<(), VerifierError> {
    jolt_verifier::verify::<
        VerifierFr,
        DoryScheme,
        Pedersen<Bn254G1>,
        jolt_transcript::LegacyBlake2bTranscript<VerifierFr>,
    >(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        zk,
    )
}

#[cfg(not(feature = "zk"))]
fn convert_vc_setup<F, C, PCS>(
    _preprocessing: &CoreJoltVerifierPreprocessing<F, C, PCS>,
) -> Option<<C::VerifierVectorCommitment as VerifierVectorCommitment>::Setup>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
{
    None
}

#[cfg(feature = "zk")]
#[expect(
    clippy::expect_used,
    reason = "core ZK verifier preprocessing must carry the BlindFold setup needed by modular verification"
)]
fn convert_vc_setup<F, C, PCS>(
    preprocessing: &CoreJoltVerifierPreprocessing<F, C, PCS>,
) -> Option<<C::VerifierVectorCommitment as VerifierVectorCommitment>::Setup>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
{
    Some(C::vc_setup_from_core_blindfold(
        &preprocessing
            .blindfold_setup
            .as_ref()
            .expect("ZK core preprocessing must carry BlindFold setup")
            .0,
    ))
}

fn convert_read_write_config(config: CoreReadWriteConfig) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: config.ram_rw_phase1_num_rounds,
        ram_rw_phase2_num_rounds: config.ram_rw_phase2_num_rounds,
        registers_rw_phase1_num_rounds: config.registers_rw_phase1_num_rounds,
        registers_rw_phase2_num_rounds: config.registers_rw_phase2_num_rounds,
    }
}

fn convert_one_hot_config(config: CoreOneHotConfig) -> JoltOneHotConfig {
    JoltOneHotConfig {
        log_k_chunk: config.log_k_chunk,
        lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
    }
}

fn convert_trace_polynomial_order(layout: CoreDoryLayout) -> TracePolynomialOrder {
    match layout {
        CoreDoryLayout::CycleMajor => TracePolynomialOrder::CycleMajor,
        CoreDoryLayout::AddressMajor => TracePolynomialOrder::AddressMajor,
    }
}

fn convert_proof_commitments<F, PCS>(
    commitments: Vec<PCS::Commitment>,
    one_hot_config: JoltOneHotConfig,
    ram_k: usize,
) -> Result<JoltCommitments<<PCS::VerifierPcs as VerifierCommitment>::Output>, VerifierError>
where
    F: CoreFieldBridge,
    PCS: CorePcsBridge<F>,
{
    let committed_chunk_bits = one_hot_config.committed_chunk_bits();
    if committed_chunk_bits == 0 {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: 2,
            got: commitments.len(),
        });
    }
    let instruction_ra_count = (2 * RISCV_XLEN).div_ceil(committed_chunk_bits);
    let ram_ra_count = ceil_log_2(ram_k).div_ceil(committed_chunk_bits);
    let commitments = commitments
        .into_iter()
        .map(PCS::commitment_into_verifier)
        .collect::<Vec<_>>();

    commitments_from_proof_payload_order(commitments, instruction_ra_count, ram_ra_count)
}

fn ceil_log_2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn commitments_from_proof_payload_order<C>(
    commitments: Vec<C>,
    instruction_ra_count: usize,
    ram_ra_count: usize,
) -> Result<JoltCommitments<C>, VerifierError> {
    let minimum = 2 + instruction_ra_count + ram_ra_count;
    if commitments.len() < minimum {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: minimum,
            got: commitments.len(),
        });
    }

    let mut commitments = commitments.into_iter();
    let Some(rd_inc) = commitments.next() else {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: minimum,
            got: 0,
        });
    };
    let Some(ram_inc) = commitments.next() else {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: minimum,
            got: 1,
        });
    };
    let instruction_ra = commitments
        .by_ref()
        .take(instruction_ra_count)
        .collect::<Vec<_>>();
    let ram_ra = commitments.by_ref().take(ram_ra_count).collect::<Vec<_>>();
    let bytecode_ra = commitments.collect::<Vec<_>>();

    Ok(JoltCommitments::new(
        rd_inc,
        ram_inc,
        JoltRaCommitments::new(instruction_ra, ram_ra, bytecode_ra),
    ))
}

#[cfg(not(feature = "zk"))]
pub fn proof_into_verifier<F, C, PCS, FS>(
    proof: CoreJoltProof<F, C, PCS, FS>,
) -> Result<ImportedCoreProof<F, C, PCS>, VerifierError>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
    FS: Transcript,
{
    let one_hot_config = convert_one_hot_config(proof.one_hot_config);
    let commitments =
        convert_proof_commitments::<F, PCS>(proof.commitments, one_hot_config, proof.ram_K)?;
    let stages = JoltStageProofs {
        stage1_uni_skip_first_round_proof: convert_uniskip(proof.stage1_uni_skip_first_round_proof),
        stage1_sumcheck_proof: convert_sumcheck(proof.stage1_sumcheck_proof),
        stage2_uni_skip_first_round_proof: convert_uniskip(proof.stage2_uni_skip_first_round_proof),
        stage2_sumcheck_proof: convert_sumcheck(proof.stage2_sumcheck_proof),
        stage3_sumcheck_proof: convert_sumcheck(proof.stage3_sumcheck_proof),
        stage4_sumcheck_proof: convert_sumcheck(proof.stage4_sumcheck_proof),
        stage5_sumcheck_proof: convert_sumcheck(proof.stage5_sumcheck_proof),
        stage6a_sumcheck_proof: convert_sumcheck(proof.stage6a_sumcheck_proof),
        stage6b_sumcheck_proof: convert_sumcheck(proof.stage6b_sumcheck_proof),
        stage7_sumcheck_proof: convert_sumcheck(proof.stage7_sumcheck_proof),
    };

    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(false),
        commitments,
        stages,
        joint_opening_proof: PCS::proof_into_verifier(proof.joint_opening_proof),
        untrusted_advice_commitment: proof
            .untrusted_advice_commitment
            .map(PCS::commitment_into_verifier),
        claims: JoltProofClaims::Clear(convert_opening_claims(
            proof.opening_claims,
            proof.trace_length,
        )),
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        rw_config: convert_read_write_config(proof.rw_config),
        one_hot_config,
        trace_polynomial_order: convert_trace_polynomial_order(proof.dory_layout),
    })
}

#[cfg(feature = "zk")]
pub fn proof_into_verifier<F, C, PCS, FS>(
    proof: CoreJoltProof<F, C, PCS, FS>,
) -> Result<ImportedCoreProof<F, C, PCS>, VerifierError>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
    FS: Transcript,
{
    let one_hot_config = convert_one_hot_config(proof.one_hot_config);
    let commitments =
        convert_proof_commitments::<F, PCS>(proof.commitments, one_hot_config, proof.ram_K)?;
    let stages = JoltStageProofs {
        stage1_uni_skip_first_round_proof: convert_uniskip(proof.stage1_uni_skip_first_round_proof),
        stage1_sumcheck_proof: convert_sumcheck(proof.stage1_sumcheck_proof),
        stage2_uni_skip_first_round_proof: convert_uniskip(proof.stage2_uni_skip_first_round_proof),
        stage2_sumcheck_proof: convert_sumcheck(proof.stage2_sumcheck_proof),
        stage3_sumcheck_proof: convert_sumcheck(proof.stage3_sumcheck_proof),
        stage4_sumcheck_proof: convert_sumcheck(proof.stage4_sumcheck_proof),
        stage5_sumcheck_proof: convert_sumcheck(proof.stage5_sumcheck_proof),
        stage6a_sumcheck_proof: convert_sumcheck(proof.stage6a_sumcheck_proof),
        stage6b_sumcheck_proof: convert_sumcheck(proof.stage6b_sumcheck_proof),
        stage7_sumcheck_proof: convert_sumcheck(proof.stage7_sumcheck_proof),
    };

    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(true),
        commitments,
        stages,
        joint_opening_proof: PCS::proof_into_verifier(proof.joint_opening_proof),
        untrusted_advice_commitment: proof
            .untrusted_advice_commitment
            .map(PCS::commitment_into_verifier),
        claims: JoltProofClaims::Zk {
            blindfold_proof: core_blindfold_proof_into_verifier::<F, C>(&proof.blindfold_proof),
        },
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        rw_config: convert_read_write_config(proof.rw_config),
        one_hot_config,
        trace_polynomial_order: convert_trace_polynomial_order(proof.dory_layout),
    })
}

fn convert_uniskip<F, C, FS>(
    proof: UniSkipFirstRoundProofVariant<F, C, FS>,
) -> SumcheckProof<F::VerifierField, C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    FS: Transcript,
{
    match proof {
        UniSkipFirstRoundProofVariant::Standard(proof) => {
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
                round_polynomials: vec![convert_univariate(proof.uni_poly)],
            }))
        }
        UniSkipFirstRoundProofVariant::Zk(proof) => {
            SumcheckProof::Committed(committed_proof_from_parts::<F, C>(
                vec![proof.commitment],
                vec![proof.poly_degree],
                proof.output_claims_commitments,
            ))
        }
    }
}

fn convert_sumcheck<F, C, FS>(
    proof: SumcheckInstanceProof<F, C, FS>,
) -> SumcheckProof<F::VerifierField, C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    FS: Transcript,
{
    match proof {
        SumcheckInstanceProof::Clear(proof) => {
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: proof
                    .compressed_polys
                    .into_iter()
                    .map(convert_compressed_poly)
                    .collect(),
            }))
        }
        SumcheckInstanceProof::Zk(proof) => {
            SumcheckProof::Committed(committed_proof_from_parts::<F, C>(
                proof.round_commitments,
                proof.poly_degrees,
                proof.output_claims_commitments,
            ))
        }
    }
}

fn committed_proof_from_parts<F, C>(
    round_commitments: Vec<C::G1>,
    poly_degrees: Vec<usize>,
    output_claims_commitments: Vec<C::G1>,
) -> CommittedSumcheckProof<C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
{
    let rounds = round_commitments
        .into_iter()
        .zip(poly_degrees)
        .map(|(commitment, degree)| CommittedRound {
            commitment: C::g1_into_verifier(commitment),
            degree,
        })
        .collect();

    CommittedSumcheckProof {
        rounds,
        output_claims: CommittedOutputClaims {
            commitments: output_claims_commitments
                .into_iter()
                .map(C::g1_into_verifier)
                .collect(),
        },
    }
}

fn convert_univariate<F>(
    poly: jolt_core::poly::unipoly::UniPoly<F>,
) -> UnivariatePoly<F::VerifierField>
where
    F: CoreFieldBridge,
{
    UnivariatePoly::new(convert_field_vec(poly.coeffs))
}

fn convert_compressed_poly<F>(
    poly: jolt_core::poly::unipoly::CompressedUniPoly<F>,
) -> CompressedPoly<F::VerifierField>
where
    F: CoreFieldBridge,
{
    CompressedPoly::new(convert_field_vec(poly.coeffs_except_linear_term))
}

fn convert_field_vec<F>(values: Vec<F>) -> Vec<F::VerifierField>
where
    F: CoreFieldBridge,
{
    values
        .into_iter()
        .map(CoreFieldBridge::into_verifier_field)
        .collect()
}

fn core_dory_commitment_into_verifier(commitment: &CoreDoryCommitment) -> Bn254GT {
    // SAFETY: `jolt-core` Dory and modular `jolt-dory` use thin wrappers
    // over the same arkworks `Fq12` target-group element.
    unsafe { std::mem::transmute_copy(commitment) }
}

#[cfg(feature = "zk")]
fn core_blindfold_proof_into_verifier<F, C>(
    proof: &CoreBlindFoldProof<F, C>,
) -> VerifierBlindFoldProof<F::VerifierField, C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
{
    VerifierBlindFoldProof {
        auxiliary_row_commitments: convert_g1_slice::<F, C>(&proof.noncoeff_row_commitments),
        random_round_commitments: convert_g1_slice::<F, C>(
            &proof.random_instance.round_commitments,
        ),
        random_output_claim_row_commitments: convert_g1_slice::<F, C>(
            &proof.random_instance.output_claims_row_commitments,
        ),
        random_auxiliary_row_commitments: convert_g1_slice::<F, C>(
            &proof.random_instance.noncoeff_row_commitments,
        ),
        random_error_row_commitments: convert_g1_slice::<F, C>(
            &proof.random_instance.e_row_commitments,
        ),
        random_eval_commitments: convert_g1_slice::<F, C>(&proof.random_instance.eval_commitments),
        random_u: proof.random_instance.u.into_verifier_field(),
        cross_term_error_row_commitments: convert_g1_slice::<F, C>(
            &proof.cross_term_row_commitments,
        ),
        outer_sumcheck: convert_compressed_sumcheck::<F>(&proof.spartan_proof),
        az_rx: proof.az_r.into_verifier_field(),
        bz_rx: proof.bz_r.into_verifier_field(),
        cz_rx: proof.cz_r.into_verifier_field(),
        inner_sumcheck: convert_compressed_sumcheck::<F>(&proof.inner_sumcheck_proof),
        witness_opening: convert_hyrax_opening::<F>(&proof.w_opening),
        error_opening: convert_hyrax_opening::<F>(&proof.e_opening),
        folded_eval_outputs: convert_field_slice(&proof.folded_eval_outputs),
        folded_eval_blindings: convert_field_slice(&proof.folded_eval_blindings),
        folded_eval_output_openings: proof
            .folded_eval_output_openings
            .iter()
            .map(convert_hyrax_opening::<F>)
            .collect(),
        folded_eval_blinding_openings: proof
            .folded_eval_blinding_openings
            .iter()
            .map(convert_hyrax_opening::<F>)
            .collect(),
    }
}

#[cfg(feature = "zk")]
fn convert_g1_slice<F, C>(commitments: &[C::G1]) -> Vec<C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
{
    commitments
        .iter()
        .copied()
        .map(C::g1_into_verifier)
        .collect()
}

#[cfg(feature = "zk")]
fn convert_compressed_sumcheck<F>(
    proof: &[CompressedUniPoly<F>],
) -> CompressedSumcheckProof<F::VerifierField>
where
    F: CoreFieldBridge,
{
    CompressedSumcheckProof {
        round_polynomials: proof.iter().map(convert_compressed_poly_ref).collect(),
    }
}

#[cfg(feature = "zk")]
fn convert_compressed_poly_ref<F>(poly: &CompressedUniPoly<F>) -> CompressedPoly<F::VerifierField>
where
    F: CoreFieldBridge,
{
    CompressedPoly::new(convert_field_slice(&poly.coeffs_except_linear_term))
}

#[cfg(feature = "zk")]
fn convert_hyrax_opening<F>(
    proof: &HyraxOpeningProof<F>,
) -> VectorCommitmentOpening<F::VerifierField>
where
    F: CoreFieldBridge,
{
    VectorCommitmentOpening {
        combined_vector: convert_field_slice(&proof.combined_row),
        combined_blinding: proof.combined_blinding.into_verifier_field(),
    }
}

#[cfg(feature = "zk")]
fn convert_field_slice<F>(values: &[F]) -> Vec<F::VerifierField>
where
    F: CoreFieldBridge,
{
    values
        .iter()
        .copied()
        .map(CoreFieldBridge::into_verifier_field)
        .collect()
}

#[cfg(not(feature = "zk"))]
#[expect(
    clippy::expect_used,
    reason = "core standard proofs are expected to carry a complete clear-claim payload"
)]
fn convert_opening_claims<F>(
    claims: jolt_core::zkvm::proof_serialization::Claims<F>,
    trace_length: usize,
) -> jolt_verifier::proof::ClearProofClaims<F::VerifierField>
where
    F: CoreFieldBridge,
{
    clear_claims_from_native(
        claims.0.into_iter().map(|(id, (_point, claim))| {
            (
                convert_opening_id(id),
                CoreFieldBridge::into_verifier_field(claim),
            )
        }),
        trace_length,
    )
    .expect("core standard proof must contain all typed clear opening claims")
}

#[cfg(not(feature = "zk"))]
fn convert_opening_id(id: core_opening::OpeningId) -> JoltOpeningId {
    match id {
        core_opening::OpeningId::Polynomial(poly, sumcheck) => {
            JoltOpeningId::polynomial(convert_polynomial_id(poly), convert_sumcheck_id(sumcheck))
        }
        core_opening::OpeningId::UntrustedAdvice(sumcheck) => {
            JoltOpeningId::untrusted_advice(convert_sumcheck_id(sumcheck))
        }
        core_opening::OpeningId::TrustedAdvice(sumcheck) => {
            JoltOpeningId::trusted_advice(convert_sumcheck_id(sumcheck))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_polynomial_id(id: core_opening::PolynomialId) -> JoltPolynomialId {
    match id {
        core_opening::PolynomialId::Committed(poly) => {
            JoltPolynomialId::Committed(convert_committed_polynomial(poly))
        }
        core_opening::PolynomialId::Virtual(poly) => {
            JoltPolynomialId::Virtual(convert_virtual_polynomial(poly))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_sumcheck_id(id: core_opening::SumcheckId) -> JoltRelationId {
    match id {
        core_opening::SumcheckId::SpartanOuter => JoltRelationId::SpartanOuter,
        core_opening::SumcheckId::SpartanProductVirtualization => {
            JoltRelationId::SpartanProductVirtualization
        }
        core_opening::SumcheckId::SpartanShift => JoltRelationId::SpartanShift,
        core_opening::SumcheckId::InstructionClaimReduction => {
            JoltRelationId::InstructionClaimReduction
        }
        core_opening::SumcheckId::InstructionInputVirtualization => {
            JoltRelationId::InstructionInputVirtualization
        }
        core_opening::SumcheckId::InstructionReadRaf => JoltRelationId::InstructionReadRaf,
        core_opening::SumcheckId::InstructionRaVirtualization => {
            JoltRelationId::InstructionRaVirtualization
        }
        core_opening::SumcheckId::RamReadWriteChecking => JoltRelationId::RamReadWriteChecking,
        core_opening::SumcheckId::RamRafEvaluation => JoltRelationId::RamRafEvaluation,
        core_opening::SumcheckId::RamOutputCheck => JoltRelationId::RamOutputCheck,
        core_opening::SumcheckId::RamValCheck => JoltRelationId::RamValCheck,
        core_opening::SumcheckId::RamRaClaimReduction => JoltRelationId::RamRaClaimReduction,
        core_opening::SumcheckId::RamHammingBooleanity => JoltRelationId::RamHammingBooleanity,
        core_opening::SumcheckId::RamRaVirtualization => JoltRelationId::RamRaVirtualization,
        core_opening::SumcheckId::RegistersClaimReduction => {
            JoltRelationId::RegistersClaimReduction
        }
        core_opening::SumcheckId::RegistersReadWriteChecking => {
            JoltRelationId::RegistersReadWriteChecking
        }
        core_opening::SumcheckId::RegistersValEvaluation => JoltRelationId::RegistersValEvaluation,
        core_opening::SumcheckId::BytecodeReadRafAddressPhase => JoltRelationId::BytecodeReadRaf,
        core_opening::SumcheckId::BytecodeReadRaf => JoltRelationId::BytecodeReadRaf,
        core_opening::SumcheckId::BooleanityAddressPhase => JoltRelationId::Booleanity,
        core_opening::SumcheckId::Booleanity => JoltRelationId::Booleanity,
        core_opening::SumcheckId::AdviceClaimReductionCyclePhase => {
            JoltRelationId::AdviceClaimReductionCyclePhase
        }
        core_opening::SumcheckId::AdviceClaimReduction => JoltRelationId::AdviceClaimReduction,
        core_opening::SumcheckId::BytecodeClaimReductionCyclePhase => {
            JoltRelationId::BytecodeClaimReductionCyclePhase
        }
        core_opening::SumcheckId::BytecodeClaimReduction => JoltRelationId::BytecodeClaimReduction,
        core_opening::SumcheckId::ProgramImageClaimReductionCyclePhase => {
            JoltRelationId::ProgramImageClaimReductionCyclePhase
        }
        core_opening::SumcheckId::ProgramImageClaimReduction => {
            JoltRelationId::ProgramImageClaimReduction
        }
        core_opening::SumcheckId::IncClaimReduction => JoltRelationId::IncClaimReduction,
        core_opening::SumcheckId::HammingWeightClaimReduction => {
            JoltRelationId::HammingWeightClaimReduction
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_committed_polynomial(
    poly: core_witness::CommittedPolynomial,
) -> JoltCommittedPolynomial {
    match poly {
        core_witness::CommittedPolynomial::RdInc => JoltCommittedPolynomial::RdInc,
        core_witness::CommittedPolynomial::RamInc => JoltCommittedPolynomial::RamInc,
        core_witness::CommittedPolynomial::InstructionRa(index) => {
            JoltCommittedPolynomial::InstructionRa(index)
        }
        core_witness::CommittedPolynomial::BytecodeRa(index) => {
            JoltCommittedPolynomial::BytecodeRa(index)
        }
        core_witness::CommittedPolynomial::BytecodeChunk(index) => {
            JoltCommittedPolynomial::BytecodeChunk(index)
        }
        core_witness::CommittedPolynomial::RamRa(index) => JoltCommittedPolynomial::RamRa(index),
        core_witness::CommittedPolynomial::TrustedAdvice => JoltCommittedPolynomial::TrustedAdvice,
        core_witness::CommittedPolynomial::UntrustedAdvice => {
            JoltCommittedPolynomial::UntrustedAdvice
        }
        core_witness::CommittedPolynomial::ProgramImageInit => {
            JoltCommittedPolynomial::ProgramImageInit
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_virtual_polynomial(poly: core_witness::VirtualPolynomial) -> JoltVirtualPolynomial {
    match poly {
        core_witness::VirtualPolynomial::PC => JoltVirtualPolynomial::PC,
        core_witness::VirtualPolynomial::UnexpandedPC => JoltVirtualPolynomial::UnexpandedPC,
        core_witness::VirtualPolynomial::NextPC => JoltVirtualPolynomial::NextPC,
        core_witness::VirtualPolynomial::NextUnexpandedPC => {
            JoltVirtualPolynomial::NextUnexpandedPC
        }
        core_witness::VirtualPolynomial::NextIsNoop => JoltVirtualPolynomial::NextIsNoop,
        core_witness::VirtualPolynomial::NextIsVirtual => JoltVirtualPolynomial::NextIsVirtual,
        core_witness::VirtualPolynomial::NextIsFirstInSequence => {
            JoltVirtualPolynomial::NextIsFirstInSequence
        }
        core_witness::VirtualPolynomial::LeftLookupOperand => {
            JoltVirtualPolynomial::LeftLookupOperand
        }
        core_witness::VirtualPolynomial::RightLookupOperand => {
            JoltVirtualPolynomial::RightLookupOperand
        }
        core_witness::VirtualPolynomial::LeftInstructionInput => {
            JoltVirtualPolynomial::LeftInstructionInput
        }
        core_witness::VirtualPolynomial::RightInstructionInput => {
            JoltVirtualPolynomial::RightInstructionInput
        }
        core_witness::VirtualPolynomial::Product => JoltVirtualPolynomial::Product,
        core_witness::VirtualPolynomial::ShouldJump => JoltVirtualPolynomial::ShouldJump,
        core_witness::VirtualPolynomial::ShouldBranch => JoltVirtualPolynomial::ShouldBranch,
        core_witness::VirtualPolynomial::Rd => JoltVirtualPolynomial::Rd,
        core_witness::VirtualPolynomial::Imm => JoltVirtualPolynomial::Imm,
        core_witness::VirtualPolynomial::Rs1Value => JoltVirtualPolynomial::Rs1Value,
        core_witness::VirtualPolynomial::Rs2Value => JoltVirtualPolynomial::Rs2Value,
        core_witness::VirtualPolynomial::RdWriteValue => JoltVirtualPolynomial::RdWriteValue,
        core_witness::VirtualPolynomial::Rs1Ra => JoltVirtualPolynomial::Rs1Ra,
        core_witness::VirtualPolynomial::Rs2Ra => JoltVirtualPolynomial::Rs2Ra,
        core_witness::VirtualPolynomial::RdWa => JoltVirtualPolynomial::RdWa,
        core_witness::VirtualPolynomial::LookupOutput => JoltVirtualPolynomial::LookupOutput,
        core_witness::VirtualPolynomial::InstructionRaf => JoltVirtualPolynomial::InstructionRaf,
        core_witness::VirtualPolynomial::InstructionRafFlag => {
            JoltVirtualPolynomial::InstructionRafFlag
        }
        core_witness::VirtualPolynomial::InstructionRa(index) => {
            JoltVirtualPolynomial::InstructionRa(index)
        }
        core_witness::VirtualPolynomial::RegistersVal => JoltVirtualPolynomial::RegistersVal,
        core_witness::VirtualPolynomial::RamAddress => JoltVirtualPolynomial::RamAddress,
        core_witness::VirtualPolynomial::RamRa => JoltVirtualPolynomial::RamRa,
        core_witness::VirtualPolynomial::RamReadValue => JoltVirtualPolynomial::RamReadValue,
        core_witness::VirtualPolynomial::RamWriteValue => JoltVirtualPolynomial::RamWriteValue,
        core_witness::VirtualPolynomial::RamVal => JoltVirtualPolynomial::RamVal,
        core_witness::VirtualPolynomial::RamValInit => JoltVirtualPolynomial::RamValInit,
        core_witness::VirtualPolynomial::RamValFinal => JoltVirtualPolynomial::RamValFinal,
        core_witness::VirtualPolynomial::RamHammingWeight => {
            JoltVirtualPolynomial::RamHammingWeight
        }
        core_witness::VirtualPolynomial::UnivariateSkip => JoltVirtualPolynomial::UnivariateSkip,
        core_witness::VirtualPolynomial::OpFlags(flag) => {
            JoltVirtualPolynomial::OpFlags(convert_circuit_flag(flag))
        }
        core_witness::VirtualPolynomial::InstructionFlags(flag) => {
            JoltVirtualPolynomial::InstructionFlags(convert_instruction_flag(flag))
        }
        core_witness::VirtualPolynomial::LookupTableFlag(index) => {
            JoltVirtualPolynomial::LookupTableFlag(index)
        }
        core_witness::VirtualPolynomial::BytecodeValStage(stage) => {
            JoltVirtualPolynomial::BytecodeValStage(stage)
        }
        core_witness::VirtualPolynomial::BytecodeReadRafAddrClaim => {
            JoltVirtualPolynomial::BytecodeReadRafAddrClaim
        }
        core_witness::VirtualPolynomial::BooleanityAddrClaim => {
            JoltVirtualPolynomial::BooleanityAddrClaim
        }
        core_witness::VirtualPolynomial::BytecodeClaimReductionIntermediate => {
            JoltVirtualPolynomial::BytecodeClaimReductionIntermediate
        }
        core_witness::VirtualPolynomial::ProgramImageInitContributionRw => {
            JoltVirtualPolynomial::ProgramImageInitContributionRw
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_circuit_flag(flag: core_instruction::CircuitFlags) -> jolt_riscv::CircuitFlags {
    match flag {
        core_instruction::CircuitFlags::AddOperands => jolt_riscv::CircuitFlags::AddOperands,
        core_instruction::CircuitFlags::SubtractOperands => {
            jolt_riscv::CircuitFlags::SubtractOperands
        }
        core_instruction::CircuitFlags::MultiplyOperands => {
            jolt_riscv::CircuitFlags::MultiplyOperands
        }
        core_instruction::CircuitFlags::Load => jolt_riscv::CircuitFlags::Load,
        core_instruction::CircuitFlags::Store => jolt_riscv::CircuitFlags::Store,
        core_instruction::CircuitFlags::Jump => jolt_riscv::CircuitFlags::Jump,
        core_instruction::CircuitFlags::WriteLookupOutputToRD => {
            jolt_riscv::CircuitFlags::WriteLookupOutputToRD
        }
        core_instruction::CircuitFlags::VirtualInstruction => {
            jolt_riscv::CircuitFlags::VirtualInstruction
        }
        core_instruction::CircuitFlags::Assert => jolt_riscv::CircuitFlags::Assert,
        core_instruction::CircuitFlags::DoNotUpdateUnexpandedPC => {
            jolt_riscv::CircuitFlags::DoNotUpdateUnexpandedPC
        }
        core_instruction::CircuitFlags::Advice => jolt_riscv::CircuitFlags::Advice,
        core_instruction::CircuitFlags::IsCompressed => jolt_riscv::CircuitFlags::IsCompressed,
        core_instruction::CircuitFlags::IsFirstInSequence => {
            jolt_riscv::CircuitFlags::IsFirstInSequence
        }
        core_instruction::CircuitFlags::IsLastInSequence => {
            jolt_riscv::CircuitFlags::IsLastInSequence
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_instruction_flag(
    flag: core_instruction::InstructionFlags,
) -> jolt_riscv::InstructionFlags {
    match flag {
        core_instruction::InstructionFlags::LeftOperandIsPC => {
            jolt_riscv::InstructionFlags::LeftOperandIsPC
        }
        core_instruction::InstructionFlags::RightOperandIsImm => {
            jolt_riscv::InstructionFlags::RightOperandIsImm
        }
        core_instruction::InstructionFlags::LeftOperandIsRs1Value => {
            jolt_riscv::InstructionFlags::LeftOperandIsRs1Value
        }
        core_instruction::InstructionFlags::RightOperandIsRs2Value => {
            jolt_riscv::InstructionFlags::RightOperandIsRs2Value
        }
        core_instruction::InstructionFlags::Branch => jolt_riscv::InstructionFlags::Branch,
        core_instruction::InstructionFlags::IsNoop => jolt_riscv::InstructionFlags::IsNoop,
    }
}
