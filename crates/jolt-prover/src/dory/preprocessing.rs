use std::sync::Arc;

use ark_serialize::CanonicalSerialize;
#[cfg(feature = "zk")]
use common::constants::MAX_BLINDFOLD_GENERATORS;
use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::geometry::{
    claim_reductions::{bytecode, program_image},
    dimensions::{CommitmentMatrixShape, TracePolynomialOrder},
};
#[cfg(feature = "zk")]
use jolt_crypto::DeriveSetup;
use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_field::{Fr, Ring};
use jolt_kernels::committed_program::{
    build_committed_bytecode_chunk_coeffs, program_image_words_padded,
};
use jolt_openings::{CommitmentScheme, StreamingCommitment};
#[cfg(feature = "zk")]
use jolt_openings::{ZkOpeningScheme, ZkStreamingCommitment};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};

use super::stages::stage0::TrustedAdviceCommitment;
use crate::preprocessing::{
    canonical_preprocessing_digest, encode_program_metadata, encode_shared_preprocessing_tail,
    COMMITTED_PROGRAM_TAG,
};
use crate::{
    CommittedProgramProverData, JoltProverPreprocessing, JoltSharedPreprocessing,
    PreprocessingError,
};

pub type DoryProverPreprocessing = JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
pub type DoryVerifierPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

impl JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>> {
    pub fn verifier_preprocessing(&self) -> DoryVerifierPreprocessing {
        self.verifier.clone()
    }

    #[cfg(feature = "zk")]
    #[expect(
        clippy::expect_used,
        reason = "the ZK constructor always installs the BlindFold setup"
    )]
    pub fn blindfold_setup(&self) -> PedersenSetup<Bn254G1> {
        self.verifier
            .vc_setup
            .clone()
            .expect("ZK preprocessing carries a BlindFold setup")
    }
}

pub fn from_shared(shared: JoltSharedPreprocessing) -> DoryProverPreprocessing {
    let total_vars = setup_total_vars(
        &shared.program.memory_layout,
        &[],
        shared.program.max_padded_trace_length,
    );
    let pcs_setup = DoryScheme::setup_prover(total_vars);
    let verifier = from_shared_parts(
        &shared,
        DoryScheme::verifier_setup(&pcs_setup),
        blindfold_setup(&pcs_setup),
    );
    JoltProverPreprocessing {
        verifier,
        pcs_setup,
        committed_program: None,
    }
}

pub fn from_shared_parts(
    shared: &JoltSharedPreprocessing,
    pcs_setup: <DoryScheme as CommitmentScheme>::VerifierSetup,
    vc_setup: Option<PedersenSetup<Bn254G1>>,
) -> DoryVerifierPreprocessing {
    JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Full(Arc::clone(&shared.program)),
        shared.preprocessing_digest,
        pcs_setup,
        vc_setup,
    )
}

pub fn preprocess_committed(
    full: JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
) -> Result<DoryProverPreprocessing, PreprocessingError> {
    preprocess_committed_with_order(full, bytecode_chunk_count, TracePolynomialOrder::CycleMajor)
}

pub fn preprocess_committed_with_order(
    full: JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
    trace_order: TracePolynomialOrder,
) -> Result<DoryProverPreprocessing, PreprocessingError> {
    let metadata = full
        .metadata()
        .ok_or_else(|| PreprocessingError::InvalidCommittedProgram {
            reason: "entry address is absent from bytecode preprocessing".to_owned(),
        })?;
    let bytecode_candidate =
        bytecode::precommitted_candidate(full.bytecode.code_size, bytecode_chunk_count).map_err(
            |error| PreprocessingError::InvalidCommittedProgram {
                reason: error.to_string(),
            },
        )?;
    let image_candidate = program_image::precommitted_candidate(full.ram.bytecode_words.len());
    let pcs_setup = DoryScheme::setup_prover(setup_total_vars(
        &full.memory_layout,
        &[bytecode_candidate, image_candidate],
        full.max_padded_trace_length,
    ));

    let (bytecode_chunk_commitments, bytecode_chunk_hints) =
        commit_bytecode_chunks(&full, bytecode_chunk_count, trace_order, &pcs_setup)?;
    let (program_image_commitment, program_image_hint) =
        commit_program_image(&full, image_candidate, &pcs_setup);
    let committed_program = CommittedProgramPreprocessing {
        meta: metadata,
        memory_layout: full.memory_layout.clone(),
        max_padded_trace_length: full.max_padded_trace_length,
        bytecode_chunk_commitments,
        program_image_commitment,
    };
    let digest = committed_program_digest(&committed_program)?;
    let program = ProgramPreprocessing::Committed(committed_program);
    let verifier = JoltVerifierPreprocessing::new(
        program,
        digest,
        DoryScheme::verifier_setup(&pcs_setup),
        blindfold_setup(&pcs_setup),
    );
    Ok(JoltProverPreprocessing {
        verifier,
        pcs_setup,
        committed_program: Some(CommittedProgramProverData {
            full: Arc::new(full),
            bytecode_chunk_hints,
            program_image_hint,
            trace_order,
        }),
    })
}

fn committed_program_digest(
    program: &CommittedProgramPreprocessing<DoryScheme>,
) -> Result<[u8; 32], PreprocessingError> {
    let bytecode_chunk_count = program.bytecode_chunk_commitments.len();
    let bytecode_t = program.meta.bytecode_len / bytecode_chunk_count;
    let bytecode_total_vars =
        bytecode::precommitted_candidate(program.meta.bytecode_len, bytecode_chunk_count).map_err(
            |error| PreprocessingError::InvalidCommittedProgram {
                reason: error.to_string(),
            },
        )?;
    let bytecode_columns =
        1usize << CommitmentMatrixShape::balanced(bytecode_total_vars).column_vars();
    let program_image_words = program
        .meta
        .program_image_len_words
        .next_power_of_two()
        .max(2);
    let program_image_columns = 1usize
        << CommitmentMatrixShape::balanced(program_image_words.ilog2() as usize).column_vars();
    let max_log_t = program.max_padded_trace_length.next_power_of_two().ilog2() as usize;
    let max_log_k_chunk = if max_log_t >= ONEHOT_CHUNK_THRESHOLD_LOG_T {
        8u8
    } else {
        4u8
    };

    canonical_preprocessing_digest(|encoded| {
        COMMITTED_PROGRAM_TAG.serialize_compressed(&mut *encoded)?;
        encode_program_metadata(&program.meta, encoded)?;

        (bytecode_chunk_count as u64).serialize_compressed(&mut *encoded)?;
        for commitment in &program.bytecode_chunk_commitments {
            commitment.0.serialize_compressed(&mut *encoded)?;
        }
        bytecode_columns.serialize_compressed(&mut *encoded)?;
        max_log_k_chunk.serialize_compressed(&mut *encoded)?;
        bytecode_chunk_count.serialize_compressed(&mut *encoded)?;
        program
            .meta
            .bytecode_len
            .serialize_compressed(&mut *encoded)?;
        bytecode_t.serialize_compressed(&mut *encoded)?;

        program
            .program_image_commitment
            .0
            .serialize_compressed(&mut *encoded)?;
        program_image_columns.serialize_compressed(&mut *encoded)?;
        program_image_words.serialize_compressed(&mut *encoded)?;

        encode_shared_preprocessing_tail(
            &program.meta,
            &program.memory_layout,
            program.max_padded_trace_length,
            bytecode_chunk_count,
            encoded,
        )
    })
}

pub fn commit_trusted_advice(
    preprocessing: &DoryProverPreprocessing,
    advice_bytes: &[u8],
) -> Result<TrustedAdviceCommitment<DoryScheme>, PreprocessingError> {
    let max_bytes = usize::try_from(
        preprocessing
            .verifier
            .program
            .memory_layout()
            .max_trusted_advice_size,
    )
    .map_err(|_| PreprocessingError::InvalidAdvice {
        reason: "trusted advice size does not fit usize".to_owned(),
    })?;
    let words =
        common::advice::canonical_advice_words(advice_bytes, max_bytes).map_err(|error| {
            PreprocessingError::InvalidAdvice {
                reason: error.to_string(),
            }
        })?;
    let evaluations: Vec<Fr> = words.into_iter().map(Fr::from_u64).collect();
    #[cfg(feature = "zk")]
    let (commitment, hint) = DoryScheme::commit_zk(&evaluations, &preprocessing.pcs_setup)?;
    #[cfg(not(feature = "zk"))]
    let (commitment, hint) = DoryScheme::commit(&evaluations, &preprocessing.pcs_setup)?;
    Ok(TrustedAdviceCommitment { commitment, hint })
}

fn blindfold_setup(
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> Option<PedersenSetup<Bn254G1>> {
    #[cfg(feature = "zk")]
    {
        Some(PedersenSetup::derive(setup, MAX_BLINDFOLD_GENERATORS))
    }
    #[cfg(not(feature = "zk"))]
    {
        let _ = setup;
        None
    }
}

fn advice_vars(max_advice_size_bytes: u64) -> usize {
    let words = (max_advice_size_bytes / 8).max(1);
    if words == 1 {
        0
    } else {
        (words - 1).ilog2() as usize + 1
    }
}

fn setup_total_vars(
    memory_layout: &MemoryLayout,
    extra_candidates: &[usize],
    max_padded_trace_length: usize,
) -> usize {
    let max_log_t = max_padded_trace_length.next_power_of_two().ilog2() as usize;
    let max_log_k_chunk = if max_log_t >= ONEHOT_CHUNK_THRESHOLD_LOG_T {
        8
    } else {
        4
    };
    extra_candidates.iter().copied().fold(
        (max_log_k_chunk + max_log_t)
            .max(advice_vars(memory_layout.max_trusted_advice_size))
            .max(advice_vars(memory_layout.max_untrusted_advice_size)),
        usize::max,
    )
}

fn commit_table(
    table: &[Fr],
    row_width: usize,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (
    DoryCommitment,
    <DoryScheme as CommitmentScheme>::OpeningHint,
) {
    let mut partial = DoryScheme::begin(setup);
    for row in table.chunks(row_width) {
        DoryScheme::feed(&mut partial, row, setup);
    }
    #[cfg(feature = "zk")]
    {
        DoryScheme::finish_zk_with_hint(partial, setup)
    }
    #[cfg(not(feature = "zk"))]
    {
        DoryScheme::finish_with_hint(partial, setup)
    }
}

fn commit_bytecode_chunks(
    full: &JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
    trace_order: TracePolynomialOrder,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> Result<
    (
        Vec<DoryCommitment>,
        Vec<<DoryScheme as CommitmentScheme>::OpeningHint>,
    ),
    PreprocessingError,
> {
    let candidate = bytecode::precommitted_candidate(full.bytecode.code_size, bytecode_chunk_count)
        .map_err(|error| PreprocessingError::InvalidCommittedProgram {
            reason: error.to_string(),
        })?;
    let tables = build_committed_bytecode_chunk_coeffs::<Fr>(
        &full.bytecode.bytecode,
        bytecode_chunk_count,
        trace_order,
    )
    .map_err(|error| PreprocessingError::InvalidCommittedProgram {
        reason: error.to_string(),
    })?;
    let row_width = 1usize << CommitmentMatrixShape::balanced(candidate).column_vars();
    Ok(tables
        .iter()
        .map(|table| commit_table(table, row_width, setup))
        .unzip())
}

fn commit_program_image(
    full: &JoltProgramPreprocessing,
    candidate: usize,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (
    DoryCommitment,
    <DoryScheme as CommitmentScheme>::OpeningHint,
) {
    let evaluations: Vec<Fr> = program_image_words_padded(&full.ram.bytecode_words)
        .into_iter()
        .map(Fr::from_u64)
        .collect();
    let row_width = 1usize << CommitmentMatrixShape::balanced(candidate).column_vars();
    commit_table(&evaluations, row_width, setup)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::RV64IMAC_JOLT;
    use jolt_verifier::CommittedProgramPreprocessing;

    use super::{committed_program_digest, from_shared, preprocess_committed};
    use crate::JoltSharedPreprocessing;

    fn assert_prover_preprocessing_round_trips(preprocessing: &super::DoryProverPreprocessing) {
        let encoded =
            bincode::serde::encode_to_vec(preprocessing, bincode::config::standard()).unwrap();
        let (decoded, consumed): (super::DoryProverPreprocessing, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

        assert_eq!(consumed, encoded.len());
        assert_eq!(
            bincode::serde::encode_to_vec(&decoded, bincode::config::standard()).unwrap(),
            encoded
        );
        assert_eq!(
            decoded.verifier.preprocessing_digest,
            preprocessing.verifier.preprocessing_digest
        );
        assert_eq!(decoded.program(), preprocessing.program());
    }

    #[test]
    fn committed_preprocessing_digest_is_legacy_compatible() {
        let full = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();
        let committed = CommittedProgramPreprocessing::<DoryScheme> {
            meta: full.metadata().unwrap(),
            memory_layout: full.memory_layout,
            max_padded_trace_length: full.max_padded_trace_length,
            bytecode_chunk_commitments: vec![DoryCommitment::default()],
            program_image_commitment: DoryCommitment::default(),
        };

        assert_eq!(
            committed_program_digest(&committed).unwrap(),
            [
                59, 40, 197, 10, 217, 59, 24, 236, 134, 68, 40, 181, 195, 223, 5, 176, 53, 66, 211,
                95, 29, 19, 80, 25, 95, 199, 196, 52, 42, 106, 98, 139,
            ]
        );
    }

    #[test]
    fn full_prover_preprocessing_round_trips() {
        let full = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();
        let preprocessing = from_shared(JoltSharedPreprocessing::new(full).unwrap());
        assert_prover_preprocessing_round_trips(&preprocessing);
    }

    #[test]
    fn committed_prover_preprocessing_round_trips() {
        let full = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();
        let preprocessing = preprocess_committed(full, 1).unwrap();
        assert_prover_preprocessing_round_trips(&preprocessing);
    }
}
