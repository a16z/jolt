#![expect(
    clippy::module_name_repetitions,
    reason = "verifier-facing aliases intentionally include protocol/module names"
)]
//! Native proof construction for prover outputs accepted by `jolt-verifier`.

#[cfg(not(feature = "zk"))]
use crate::zkvm::clear_claims::build_clear_claims;
pub use jolt_verifier::VerifierError;
use jolt_verifier::{
    config::{CommitmentConfig, JoltProtocolConfig, ZkConfig},
    preprocessing::{
        CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
    },
    proof::{
        JoltCommitments, JoltProof, JoltProofClaims, JoltRaCommitments, JoltStageProofs,
        TracePolynomialOrder,
    },
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

use crate::{
    curve::{Bn254Curve, JoltCurve},
    field::JoltField,
    poly::commitment::{
        commitment_scheme::{CommitmentScheme, ZkEvalCommitment},
        dory::{
            ArkGT as ProverDoryCommitment, DoryCommitmentScheme, DoryLayout as ProverDoryLayout,
        },
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig as ProverOneHotConfig, ReadWriteConfig as ProverReadWriteConfig},
        preprocessing::{BlindfoldSetup, JoltSharedPreprocessing},
        program::ProgramPreprocessing as ProverProgramPreprocessing,
        proof_parts::JoltProofParts as ProverProofParts,
        prover::JoltProverPreprocessing,
    },
};
#[cfg(feature = "zk")]
use crate::{
    poly::commitment::hyrax::HyraxOpeningProof, poly::unipoly::CompressedUniPoly,
    subprotocols::blindfold::BlindFoldProof as ProverBlindFoldProof,
};
#[cfg(not(feature = "zk"))]
use crate::{
    poly::opening_proof as prover_opening,
    zkvm::{instruction as prover_instruction, witness as prover_witness},
};
#[cfg(feature = "zk")]
use jolt_blindfold::BlindFoldProof as VerifierBlindFoldProof;

pub type RV64IMACProof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

pub trait ProofField: JoltField {
    type VerifierField: VerifierFieldTrait;

    fn into_verifier_field(self) -> Self::VerifierField;
}

impl ProofField for crate::ark_bn254::Fr {
    type VerifierField = VerifierFr;

    fn into_verifier_field(self) -> Self::VerifierField {
        self.into()
    }
}

pub trait ProofCommitmentScheme<F>: CommitmentScheme<Field = F>
where
    F: ProofField,
{
    type VerifierPcs: VerifierCommitmentScheme<Field = F::VerifierField>;

    fn commitment_into_verifier(
        commitment: Self::Commitment,
    ) -> <Self::VerifierPcs as VerifierCommitment>::Output;

    fn opening_proof_into_verifier(
        proof: Self::Proof,
    ) -> <Self::VerifierPcs as VerifierCommitmentScheme>::Proof;

    fn verifier_setup_into_verifier(
        setup: Self::VerifierSetup,
    ) -> <Self::VerifierPcs as VerifierCommitmentScheme>::VerifierSetup;
}

impl ProofCommitmentScheme<crate::ark_bn254::Fr> for DoryCommitmentScheme {
    type VerifierPcs = DoryScheme;

    fn commitment_into_verifier(commitment: Self::Commitment) -> DoryCommitment {
        DoryCommitment(prover_dory_commitment_into_verifier(&commitment))
    }

    fn opening_proof_into_verifier(proof: Self::Proof) -> DoryProof {
        DoryProof(proof)
    }

    fn verifier_setup_into_verifier(setup: Self::VerifierSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(setup)
    }
}

pub trait ProofCurve<F>: JoltCurve<F = F> + Sized
where
    F: ProofField,
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

    fn vc_setup_from_prover_blindfold(
        setup: &crate::poly::commitment::pedersen::PedersenGenerators<Self>,
    ) -> <Self::VerifierVectorCommitment as VerifierVectorCommitment>::Setup;
}

impl ProofCurve<crate::ark_bn254::Fr> for Bn254Curve {
    type VerifierVectorCommitment = Pedersen<Bn254G1>;
    type VerifierRoundCommitment = Bn254G1;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment {
        commitment.0.into()
    }

    fn vc_setup_from_prover_blindfold(
        setup: &crate::poly::commitment::pedersen::PedersenGenerators<Self>,
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

fn verifier_preprocessing_from_parts<F, C, PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
    generators: PCS::VerifierSetup,
    blindfold_setup: Option<&BlindfoldSetup<C>>,
) -> JoltVerifierPreprocessing<
    <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
    <C as ProofCurve<F>>::VerifierVectorCommitment,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
{
    JoltVerifierPreprocessing::new(
        verifier_program_from_shared::<F, PCS>(shared),
        shared.digest(),
        PCS::verifier_setup_into_verifier(generators),
        convert_vc_setup::<F, C>(blindfold_setup),
    )
}

fn verifier_program_from_shared<F, PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
) -> ProgramPreprocessing<<PCS as ProofCommitmentScheme<F>>::VerifierPcs>
where
    F: ProofField,
    PCS: ProofCommitmentScheme<F>,
{
    match &shared.program {
        ProverProgramPreprocessing::Full(full) => {
            ProgramPreprocessing::Full(JoltProgramPreprocessing {
                bytecode: full.bytecode.as_ref().clone(),
                ram: full.ram.clone(),
                memory_layout: shared.memory_layout.clone(),
                max_padded_trace_length: shared.max_padded_trace_length,
            })
        }
        ProverProgramPreprocessing::Committed(committed) => {
            ProgramPreprocessing::Committed(CommittedProgramPreprocessing {
                meta: ProgramMetadata {
                    entry_address: committed.meta.entry_address,
                    min_bytecode_address: committed.meta.min_bytecode_address,
                    entry_bytecode_index: committed.meta.entry_bytecode_index,
                    program_image_len_words: committed.meta.program_image_len_words,
                    bytecode_len: committed.meta.bytecode_len,
                },
                memory_layout: shared.memory_layout.clone(),
                max_padded_trace_length: shared.max_padded_trace_length,
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
    }
}

pub fn verifier_preprocessing_from_shared<F, C, PCS>(
    shared: JoltSharedPreprocessing<PCS>,
    generators: PCS::VerifierSetup,
    blindfold_setup: Option<BlindfoldSetup<C>>,
) -> JoltVerifierPreprocessing<
    <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
    <C as ProofCurve<F>>::VerifierVectorCommitment,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
{
    verifier_preprocessing_from_parts(&shared, generators, blindfold_setup.as_ref())
}

pub fn verifier_preprocessing_from_prover<F, C, PCS>(
    preprocessing: &JoltProverPreprocessing<F, C, PCS>,
) -> JoltVerifierPreprocessing<
    <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
    <C as ProofCurve<F>>::VerifierVectorCommitment,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F> + ZkEvalCommitment<C>,
{
    let generators = PCS::setup_verifier(&preprocessing.generators);
    #[cfg(not(feature = "zk"))]
    let blindfold_setup: Option<BlindfoldSetup<C>> = None;
    #[cfg(feature = "zk")]
    let blindfold_setup = Some(preprocessing.blindfold_setup());

    verifier_preprocessing_from_parts(&preprocessing.shared, generators, blindfold_setup.as_ref())
}

#[cfg(not(feature = "zk"))]
fn convert_vc_setup<F, C>(
    _blindfold_setup: Option<&BlindfoldSetup<C>>,
) -> Option<<C::VerifierVectorCommitment as VerifierVectorCommitment>::Setup>
where
    F: ProofField,
    C: ProofCurve<F>,
{
    None
}

#[cfg(feature = "zk")]
#[expect(
    clippy::expect_used,
    reason = "prover ZK verifier preprocessing must carry the BlindFold setup needed by modular verification"
)]
fn convert_vc_setup<F, C>(
    blindfold_setup: Option<&BlindfoldSetup<C>>,
) -> Option<<C::VerifierVectorCommitment as VerifierVectorCommitment>::Setup>
where
    F: ProofField,
    C: ProofCurve<F>,
{
    Some(C::vc_setup_from_prover_blindfold(
        &blindfold_setup
            .expect("ZK prover preprocessing must carry BlindFold setup")
            .0,
    ))
}

fn convert_read_write_config(config: ProverReadWriteConfig) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: config.ram_rw_phase1_num_rounds,
        ram_rw_phase2_num_rounds: config.ram_rw_phase2_num_rounds,
        registers_rw_phase1_num_rounds: config.registers_rw_phase1_num_rounds,
        registers_rw_phase2_num_rounds: config.registers_rw_phase2_num_rounds,
    }
}

fn convert_one_hot_config(config: ProverOneHotConfig) -> JoltOneHotConfig {
    JoltOneHotConfig {
        log_k_chunk: config.log_k_chunk,
        lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
    }
}

fn convert_trace_polynomial_order(layout: ProverDoryLayout) -> TracePolynomialOrder {
    match layout {
        ProverDoryLayout::CycleMajor => TracePolynomialOrder::CycleMajor,
        ProverDoryLayout::AddressMajor => TracePolynomialOrder::AddressMajor,
    }
}

fn convert_proof_commitments<F, PCS>(
    commitments: Vec<PCS::Commitment>,
    one_hot_config: JoltOneHotConfig,
    ram_k: usize,
) -> Result<JoltCommitments<<PCS::VerifierPcs as VerifierCommitment>::Output>, VerifierError>
where
    F: ProofField,
    PCS: ProofCommitmentScheme<F>,
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
#[expect(
    clippy::type_complexity,
    reason = "private converter returns the verifier-native proof with projected backend types"
)]
pub(crate) fn proof_parts_into_verifier<F, C, PCS, FS>(
    proof: ProverProofParts<F, C, PCS, FS>,
) -> Result<
    JoltProof<
        <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
        <C as ProofCurve<F>>::VerifierVectorCommitment,
    >,
    VerifierError,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
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
        protocol: JoltProtocolConfig {
            zk: ZkConfig::Transparent,
            commitment: CommitmentConfig::Homomorphic,
        },
        commitments,
        stages,
        joint_opening_proof: PCS::opening_proof_into_verifier(proof.joint_opening_proof),
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
#[expect(
    clippy::type_complexity,
    reason = "private converter returns the verifier-native proof with projected backend types"
)]
pub(crate) fn proof_parts_into_verifier<F, C, PCS, FS>(
    proof: ProverProofParts<F, C, PCS, FS>,
) -> Result<
    JoltProof<
        <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
        <C as ProofCurve<F>>::VerifierVectorCommitment,
    >,
    VerifierError,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
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
        protocol: JoltProtocolConfig {
            zk: ZkConfig::BlindFold,
            commitment: CommitmentConfig::Homomorphic,
        },
        commitments,
        stages,
        joint_opening_proof: PCS::opening_proof_into_verifier(proof.joint_opening_proof),
        untrusted_advice_commitment: proof
            .untrusted_advice_commitment
            .map(PCS::commitment_into_verifier),
        claims: JoltProofClaims::Zk {
            blindfold_proof: prover_blindfold_proof_into_verifier::<F, C>(&proof.blindfold_proof),
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
    F: ProofField,
    C: ProofCurve<F>,
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
    F: ProofField,
    C: ProofCurve<F>,
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
    F: ProofField,
    C: ProofCurve<F>,
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

fn convert_univariate<F>(poly: crate::poly::unipoly::UniPoly<F>) -> UnivariatePoly<F::VerifierField>
where
    F: ProofField,
{
    UnivariatePoly::new(convert_field_vec(poly.coeffs))
}

fn convert_compressed_poly<F>(
    poly: crate::poly::unipoly::CompressedUniPoly<F>,
) -> CompressedPoly<F::VerifierField>
where
    F: ProofField,
{
    CompressedPoly::new(convert_field_vec(poly.coeffs_except_linear_term))
}

fn convert_field_vec<F>(values: Vec<F>) -> Vec<F::VerifierField>
where
    F: ProofField,
{
    values
        .into_iter()
        .map(ProofField::into_verifier_field)
        .collect()
}

fn prover_dory_commitment_into_verifier(commitment: &ProverDoryCommitment) -> Bn254GT {
    // `ArkGT` (prover Dory) and `Bn254GT` (modular `jolt-dory`) both wrap the same
    // patched `ark_bn254::Fq12`, so convert through the shared inner element. This
    // is equivalent to the previous `transmute_copy` but layout-independent: a
    // future change to either wrapper becomes a type error here instead of silent
    // memory corruption on the soundness-critical commitment path.
    Bn254GT::from(commitment.0)
}

#[cfg(feature = "zk")]
fn prover_blindfold_proof_into_verifier<F, C>(
    proof: &ProverBlindFoldProof<F, C>,
) -> VerifierBlindFoldProof<F::VerifierField, C::VerifierRoundCommitment>
where
    F: ProofField,
    C: ProofCurve<F>,
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
    F: ProofField,
    C: ProofCurve<F>,
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
    F: ProofField,
{
    CompressedSumcheckProof {
        round_polynomials: proof.iter().map(convert_compressed_poly_ref).collect(),
    }
}

#[cfg(feature = "zk")]
fn convert_compressed_poly_ref<F>(poly: &CompressedUniPoly<F>) -> CompressedPoly<F::VerifierField>
where
    F: ProofField,
{
    CompressedPoly::new(convert_field_slice(&poly.coeffs_except_linear_term))
}

#[cfg(feature = "zk")]
fn convert_hyrax_opening<F>(
    proof: &HyraxOpeningProof<F>,
) -> VectorCommitmentOpening<F::VerifierField>
where
    F: ProofField,
{
    VectorCommitmentOpening {
        combined_vector: convert_field_slice(&proof.combined_row),
        combined_blinding: proof.combined_blinding.into_verifier_field(),
    }
}

#[cfg(feature = "zk")]
fn convert_field_slice<F>(values: &[F]) -> Vec<F::VerifierField>
where
    F: ProofField,
{
    values
        .iter()
        .copied()
        .map(ProofField::into_verifier_field)
        .collect()
}

#[cfg(not(feature = "zk"))]
#[expect(
    clippy::expect_used,
    reason = "standard prover proofs are expected to carry a complete clear-claim payload"
)]
fn convert_opening_claims<F>(
    claims: crate::zkvm::proof_parts::ProverOpeningClaims<F>,
    trace_length: usize,
) -> jolt_verifier::proof::ClearProofClaims<F::VerifierField>
where
    F: ProofField,
{
    build_clear_claims(
        claims.0.into_iter().map(|(id, (_point, claim))| {
            (
                convert_opening_id(id),
                ProofField::into_verifier_field(claim),
            )
        }),
        trace_length,
    )
    .expect("standard prover proof must contain all typed clear opening claims")
}

#[cfg(not(feature = "zk"))]
fn convert_opening_id(id: prover_opening::OpeningId) -> JoltOpeningId {
    match id {
        prover_opening::OpeningId::Polynomial(poly, sumcheck) => {
            JoltOpeningId::polynomial(convert_polynomial_id(poly), convert_sumcheck_id(sumcheck))
        }
        prover_opening::OpeningId::UntrustedAdvice(sumcheck) => {
            JoltOpeningId::untrusted_advice(convert_sumcheck_id(sumcheck))
        }
        prover_opening::OpeningId::TrustedAdvice(sumcheck) => {
            JoltOpeningId::trusted_advice(convert_sumcheck_id(sumcheck))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_polynomial_id(id: prover_opening::PolynomialId) -> JoltPolynomialId {
    match id {
        prover_opening::PolynomialId::Committed(poly) => {
            JoltPolynomialId::Committed(convert_committed_polynomial(poly))
        }
        prover_opening::PolynomialId::Virtual(poly) => {
            JoltPolynomialId::Virtual(convert_virtual_polynomial(poly))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_sumcheck_id(id: prover_opening::SumcheckId) -> JoltRelationId {
    match id {
        prover_opening::SumcheckId::SpartanOuter => JoltRelationId::SpartanOuter,
        prover_opening::SumcheckId::SpartanProductVirtualization => {
            JoltRelationId::SpartanProductVirtualization
        }
        prover_opening::SumcheckId::SpartanShift => JoltRelationId::SpartanShift,
        prover_opening::SumcheckId::InstructionClaimReduction => {
            JoltRelationId::InstructionClaimReduction
        }
        prover_opening::SumcheckId::InstructionInputVirtualization => {
            JoltRelationId::InstructionInputVirtualization
        }
        prover_opening::SumcheckId::InstructionReadRaf => JoltRelationId::InstructionReadRaf,
        prover_opening::SumcheckId::InstructionRaVirtualization => {
            JoltRelationId::InstructionRaVirtualization
        }
        prover_opening::SumcheckId::RamReadWriteChecking => JoltRelationId::RamReadWriteChecking,
        prover_opening::SumcheckId::RamRafEvaluation => JoltRelationId::RamRafEvaluation,
        prover_opening::SumcheckId::RamOutputCheck => JoltRelationId::RamOutputCheck,
        prover_opening::SumcheckId::RamValCheck => JoltRelationId::RamValCheck,
        prover_opening::SumcheckId::RamRaClaimReduction => JoltRelationId::RamRaClaimReduction,
        prover_opening::SumcheckId::RamHammingBooleanity => JoltRelationId::RamHammingBooleanity,
        prover_opening::SumcheckId::RamRaVirtualization => JoltRelationId::RamRaVirtualization,
        prover_opening::SumcheckId::RegistersClaimReduction => {
            JoltRelationId::RegistersClaimReduction
        }
        prover_opening::SumcheckId::RegistersReadWriteChecking => {
            JoltRelationId::RegistersReadWriteChecking
        }
        prover_opening::SumcheckId::RegistersValEvaluation => {
            JoltRelationId::RegistersValEvaluation
        }
        prover_opening::SumcheckId::BytecodeReadRafAddressPhase => JoltRelationId::BytecodeReadRaf,
        prover_opening::SumcheckId::BytecodeReadRaf => JoltRelationId::BytecodeReadRaf,
        prover_opening::SumcheckId::BooleanityAddressPhase => JoltRelationId::Booleanity,
        prover_opening::SumcheckId::Booleanity => JoltRelationId::Booleanity,
        prover_opening::SumcheckId::AdviceClaimReductionCyclePhase => {
            JoltRelationId::AdviceClaimReductionCyclePhase
        }
        prover_opening::SumcheckId::AdviceClaimReduction => JoltRelationId::AdviceClaimReduction,
        prover_opening::SumcheckId::BytecodeClaimReductionCyclePhase => {
            JoltRelationId::BytecodeClaimReductionCyclePhase
        }
        prover_opening::SumcheckId::BytecodeClaimReduction => {
            JoltRelationId::BytecodeClaimReduction
        }
        prover_opening::SumcheckId::ProgramImageClaimReductionCyclePhase => {
            JoltRelationId::ProgramImageClaimReductionCyclePhase
        }
        prover_opening::SumcheckId::ProgramImageClaimReduction => {
            JoltRelationId::ProgramImageClaimReduction
        }
        prover_opening::SumcheckId::IncClaimReduction => JoltRelationId::IncClaimReduction,
        prover_opening::SumcheckId::HammingWeightClaimReduction => {
            JoltRelationId::HammingWeightClaimReduction
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_committed_polynomial(
    poly: prover_witness::CommittedPolynomial,
) -> JoltCommittedPolynomial {
    match poly {
        prover_witness::CommittedPolynomial::RdInc => JoltCommittedPolynomial::RdInc,
        prover_witness::CommittedPolynomial::RamInc => JoltCommittedPolynomial::RamInc,
        prover_witness::CommittedPolynomial::InstructionRa(index) => {
            JoltCommittedPolynomial::InstructionRa(index)
        }
        prover_witness::CommittedPolynomial::BytecodeRa(index) => {
            JoltCommittedPolynomial::BytecodeRa(index)
        }
        prover_witness::CommittedPolynomial::BytecodeChunk(index) => {
            JoltCommittedPolynomial::BytecodeChunk(index)
        }
        prover_witness::CommittedPolynomial::RamRa(index) => JoltCommittedPolynomial::RamRa(index),
        prover_witness::CommittedPolynomial::TrustedAdvice => {
            JoltCommittedPolynomial::TrustedAdvice
        }
        prover_witness::CommittedPolynomial::UntrustedAdvice => {
            JoltCommittedPolynomial::UntrustedAdvice
        }
        prover_witness::CommittedPolynomial::ProgramImageInit => {
            JoltCommittedPolynomial::ProgramImageInit
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_virtual_polynomial(poly: prover_witness::VirtualPolynomial) -> JoltVirtualPolynomial {
    match poly {
        prover_witness::VirtualPolynomial::PC => JoltVirtualPolynomial::PC,
        prover_witness::VirtualPolynomial::UnexpandedPC => JoltVirtualPolynomial::UnexpandedPC,
        prover_witness::VirtualPolynomial::NextPC => JoltVirtualPolynomial::NextPC,
        prover_witness::VirtualPolynomial::NextUnexpandedPC => {
            JoltVirtualPolynomial::NextUnexpandedPC
        }
        prover_witness::VirtualPolynomial::NextIsNoop => JoltVirtualPolynomial::NextIsNoop,
        prover_witness::VirtualPolynomial::NextIsVirtual => JoltVirtualPolynomial::NextIsVirtual,
        prover_witness::VirtualPolynomial::NextIsFirstInSequence => {
            JoltVirtualPolynomial::NextIsFirstInSequence
        }
        prover_witness::VirtualPolynomial::LeftLookupOperand => {
            JoltVirtualPolynomial::LeftLookupOperand
        }
        prover_witness::VirtualPolynomial::RightLookupOperand => {
            JoltVirtualPolynomial::RightLookupOperand
        }
        prover_witness::VirtualPolynomial::LeftInstructionInput => {
            JoltVirtualPolynomial::LeftInstructionInput
        }
        prover_witness::VirtualPolynomial::RightInstructionInput => {
            JoltVirtualPolynomial::RightInstructionInput
        }
        prover_witness::VirtualPolynomial::Product => JoltVirtualPolynomial::Product,
        prover_witness::VirtualPolynomial::ShouldJump => JoltVirtualPolynomial::ShouldJump,
        prover_witness::VirtualPolynomial::ShouldBranch => JoltVirtualPolynomial::ShouldBranch,
        prover_witness::VirtualPolynomial::Rd => JoltVirtualPolynomial::Rd,
        prover_witness::VirtualPolynomial::Imm => JoltVirtualPolynomial::Imm,
        prover_witness::VirtualPolynomial::Rs1Value => JoltVirtualPolynomial::Rs1Value,
        prover_witness::VirtualPolynomial::Rs2Value => JoltVirtualPolynomial::Rs2Value,
        prover_witness::VirtualPolynomial::RdWriteValue => JoltVirtualPolynomial::RdWriteValue,
        prover_witness::VirtualPolynomial::Rs1Ra => JoltVirtualPolynomial::Rs1Ra,
        prover_witness::VirtualPolynomial::Rs2Ra => JoltVirtualPolynomial::Rs2Ra,
        prover_witness::VirtualPolynomial::RdWa => JoltVirtualPolynomial::RdWa,
        prover_witness::VirtualPolynomial::LookupOutput => JoltVirtualPolynomial::LookupOutput,
        prover_witness::VirtualPolynomial::InstructionRaf => JoltVirtualPolynomial::InstructionRaf,
        prover_witness::VirtualPolynomial::InstructionRafFlag => {
            JoltVirtualPolynomial::InstructionRafFlag
        }
        prover_witness::VirtualPolynomial::InstructionRa(index) => {
            JoltVirtualPolynomial::InstructionRa(index)
        }
        prover_witness::VirtualPolynomial::RegistersVal => JoltVirtualPolynomial::RegistersVal,
        prover_witness::VirtualPolynomial::RamAddress => JoltVirtualPolynomial::RamAddress,
        prover_witness::VirtualPolynomial::RamRa => JoltVirtualPolynomial::RamRa,
        prover_witness::VirtualPolynomial::RamReadValue => JoltVirtualPolynomial::RamReadValue,
        prover_witness::VirtualPolynomial::RamWriteValue => JoltVirtualPolynomial::RamWriteValue,
        prover_witness::VirtualPolynomial::RamVal => JoltVirtualPolynomial::RamVal,
        prover_witness::VirtualPolynomial::RamValInit => JoltVirtualPolynomial::RamValInit,
        prover_witness::VirtualPolynomial::RamValFinal => JoltVirtualPolynomial::RamValFinal,
        prover_witness::VirtualPolynomial::RamHammingWeight => {
            JoltVirtualPolynomial::RamHammingWeight
        }
        prover_witness::VirtualPolynomial::UnivariateSkip => JoltVirtualPolynomial::UnivariateSkip,
        prover_witness::VirtualPolynomial::OpFlags(flag) => {
            JoltVirtualPolynomial::OpFlags(convert_circuit_flag(flag))
        }
        prover_witness::VirtualPolynomial::InstructionFlags(flag) => {
            JoltVirtualPolynomial::InstructionFlags(convert_instruction_flag(flag))
        }
        prover_witness::VirtualPolynomial::LookupTableFlag(index) => {
            JoltVirtualPolynomial::LookupTableFlag(index)
        }
        prover_witness::VirtualPolynomial::BytecodeValStage(stage) => {
            JoltVirtualPolynomial::BytecodeValStage(stage)
        }
        prover_witness::VirtualPolynomial::BytecodeReadRafAddrClaim => {
            JoltVirtualPolynomial::BytecodeReadRafAddrClaim
        }
        prover_witness::VirtualPolynomial::BooleanityAddrClaim => {
            JoltVirtualPolynomial::BooleanityAddrClaim
        }
        prover_witness::VirtualPolynomial::BytecodeClaimReductionIntermediate => {
            JoltVirtualPolynomial::BytecodeClaimReductionIntermediate
        }
        prover_witness::VirtualPolynomial::ProgramImageInitContributionRw => {
            JoltVirtualPolynomial::ProgramImageInitContributionRw
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_circuit_flag(flag: prover_instruction::CircuitFlags) -> jolt_riscv::CircuitFlags {
    match flag {
        prover_instruction::CircuitFlags::AddOperands => jolt_riscv::CircuitFlags::AddOperands,
        prover_instruction::CircuitFlags::SubtractOperands => {
            jolt_riscv::CircuitFlags::SubtractOperands
        }
        prover_instruction::CircuitFlags::MultiplyOperands => {
            jolt_riscv::CircuitFlags::MultiplyOperands
        }
        prover_instruction::CircuitFlags::Load => jolt_riscv::CircuitFlags::Load,
        prover_instruction::CircuitFlags::Store => jolt_riscv::CircuitFlags::Store,
        prover_instruction::CircuitFlags::Jump => jolt_riscv::CircuitFlags::Jump,
        prover_instruction::CircuitFlags::WriteLookupOutputToRD => {
            jolt_riscv::CircuitFlags::WriteLookupOutputToRD
        }
        prover_instruction::CircuitFlags::VirtualInstruction => {
            jolt_riscv::CircuitFlags::VirtualInstruction
        }
        prover_instruction::CircuitFlags::Assert => jolt_riscv::CircuitFlags::Assert,
        prover_instruction::CircuitFlags::DoNotUpdateUnexpandedPC => {
            jolt_riscv::CircuitFlags::DoNotUpdateUnexpandedPC
        }
        prover_instruction::CircuitFlags::Advice => jolt_riscv::CircuitFlags::Advice,
        prover_instruction::CircuitFlags::IsCompressed => jolt_riscv::CircuitFlags::IsCompressed,
        prover_instruction::CircuitFlags::IsFirstInSequence => {
            jolt_riscv::CircuitFlags::IsFirstInSequence
        }
        prover_instruction::CircuitFlags::IsLastInSequence => {
            jolt_riscv::CircuitFlags::IsLastInSequence
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_instruction_flag(
    flag: prover_instruction::InstructionFlags,
) -> jolt_riscv::InstructionFlags {
    match flag {
        prover_instruction::InstructionFlags::LeftOperandIsPC => {
            jolt_riscv::InstructionFlags::LeftOperandIsPC
        }
        prover_instruction::InstructionFlags::RightOperandIsImm => {
            jolt_riscv::InstructionFlags::RightOperandIsImm
        }
        prover_instruction::InstructionFlags::LeftOperandIsRs1Value => {
            jolt_riscv::InstructionFlags::LeftOperandIsRs1Value
        }
        prover_instruction::InstructionFlags::RightOperandIsRs2Value => {
            jolt_riscv::InstructionFlags::RightOperandIsRs2Value
        }
        prover_instruction::InstructionFlags::Branch => jolt_riscv::InstructionFlags::Branch,
        prover_instruction::InstructionFlags::IsNoop => jolt_riscv::InstructionFlags::IsNoop,
    }
}
