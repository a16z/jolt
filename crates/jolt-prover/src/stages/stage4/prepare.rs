use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::sparse_segments_mle_msb;
use jolt_program::preprocess::PublicInitialRam;
use jolt_verifier::stages::relations::OpeningClaim;
use jolt_verifier::stages::stage4::{
    ram_val_check_advice_block, RamValCheckInitialEvaluation, VerifiedRamValCheckAdviceContribution,
};
use jolt_verifier::{stages::stage2::outputs::Stage2ClearOutput, CheckedInputs};

use crate::{JoltProverPreprocessing, ProverError};

pub(crate) fn ram_val_check_initial_evaluation<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    checked: &CheckedInputs,
    stage2: &Stage2ClearOutput<PCS::Field>,
    log_k: usize,
) -> Result<RamValCheckInitialEvaluation<PCS::Field>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let opening_point = stage2.output_claims.ram_read_write_point();
    if opening_point.len() < log_k {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 RAM read-write opening point has {} variables, fewer than log_k {log_k}",
                opening_point.len()
            ),
        });
    }
    let (r_address, _) = opening_point.split_at(log_k);
    let full_program = preprocessing.verifier.program.as_full().ok_or_else(|| {
        ProverError::InvalidStageRequest {
            reason: "Stage 4 requires full program preprocessing (committed-program mode is not supported by the prover)".to_string(),
        }
    })?;
    let public_initial_ram =
        PublicInitialRam::new(&full_program.ram, &checked.public_io).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: format!("Stage 4 public initial RAM construction failed: {error}"),
            }
        })?;
    for segment in &public_initial_ram.segments {
        let end = segment.start_index + segment.words.len();
        if end > checked.ram_K {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 4 public initial RAM segment [{}, {}) exceeds RAM domain {}",
                    segment.start_index, end, checked.ram_K
                ),
            });
        }
    }

    let public_eval = sparse_segments_mle_msb(
        public_initial_ram
            .segments
            .iter()
            .map(|segment| (segment.start_index, segment.words.as_slice())),
        r_address,
    );
    let mut advice_contributions = Vec::new();
    collect_advice_contribution(
        JoltAdviceKind::Untrusted,
        !checked.public_io.untrusted_advice.is_empty(),
        &checked.public_io.untrusted_advice,
        checked,
        r_address,
        &mut advice_contributions,
    )?;
    collect_advice_contribution(
        JoltAdviceKind::Trusted,
        checked.trusted_advice_commitment_present,
        &checked.public_io.trusted_advice,
        checked,
        r_address,
        &mut advice_contributions,
    )?;

    Ok(RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution: None,
        advice_contributions,
    })
}

fn collect_advice_contribution<F: Field>(
    kind: JoltAdviceKind,
    present: bool,
    bytes: &[u8],
    checked: &CheckedInputs,
    r_address: &[F],
    contributions: &mut Vec<VerifiedRamValCheckAdviceContribution<F>>,
) -> Result<(), ProverError> {
    if !present {
        return Ok(());
    }

    let max_size = match kind {
        JoltAdviceKind::Trusted => checked.public_io.memory_layout.max_trusted_advice_size,
        JoltAdviceKind::Untrusted => checked.public_io.memory_layout.max_untrusted_advice_size,
    };
    if bytes.len() > max_size as usize {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 {kind:?} advice has {} bytes, exceeding configured max {max_size}",
                bytes.len()
            ),
        });
    }

    // The selector and opening point are the shared advice-block geometry the
    // verifier checks against; compute the opening value here from the witness bytes.
    let block = ram_val_check_advice_block(kind, checked, r_address).map_err(|error| {
        ProverError::InvalidStageRequest {
            reason: error.to_string(),
        }
    })?;
    let words = advice_words_le(bytes);
    let opening_claim = sparse_segments_mle_msb([(0, words.as_slice())], &block.opening_point);
    contributions.push(VerifiedRamValCheckAdviceContribution {
        kind,
        selector: block.selector,
        opening: OpeningClaim {
            point: block.opening_point,
            value: opening_claim,
        },
    });
    Ok(())
}

fn advice_words_le(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect()
}
