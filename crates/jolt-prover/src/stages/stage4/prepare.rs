use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{block_selector_mle_msb, sparse_segments_mle_msb};
use jolt_program::preprocess::PublicInitialRam;
use jolt_verifier::{stages::stage2::outputs::Stage2ClearOutput, CheckedInputs};

use super::prove::{Stage4RamValCheckAdviceContribution, Stage4RamValCheckInitialEvaluation};
use crate::{JoltProverPreprocessing, ProverError};

pub(crate) fn ram_val_check_initial_evaluation<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    checked: &CheckedInputs,
    stage2: &Stage2ClearOutput<PCS::Field>,
    log_k: usize,
) -> Result<Stage4RamValCheckInitialEvaluation<PCS::Field>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let opening_point = &stage2.batch.ram_read_write.opening_point;
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
    let mut full_eval = public_eval;
    let mut advice_contributions = Vec::new();
    collect_advice_contribution(
        JoltAdviceKind::Untrusted,
        !checked.public_io.untrusted_advice.is_empty(),
        &checked.public_io.untrusted_advice,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;
    collect_advice_contribution(
        JoltAdviceKind::Trusted,
        checked.trusted_advice_commitment_present,
        &checked.public_io.trusted_advice,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;

    Ok(Stage4RamValCheckInitialEvaluation {
        public_eval,
        advice_contributions,
        full_eval,
    })
}

fn collect_advice_contribution<F: Field>(
    kind: JoltAdviceKind,
    present: bool,
    bytes: &[u8],
    checked: &CheckedInputs,
    r_address: &[F],
    full_eval: &mut F,
    contributions: &mut Vec<Stage4RamValCheckAdviceContribution<F>>,
) -> Result<(), ProverError> {
    if !present {
        return Ok(());
    }

    let layout = &checked.public_io.memory_layout;
    let (start_address, max_size) = match kind {
        JoltAdviceKind::Trusted => (layout.trusted_advice_start, layout.max_trusted_advice_size),
        JoltAdviceKind::Untrusted => (
            layout.untrusted_advice_start,
            layout.max_untrusted_advice_size,
        ),
    };
    if max_size == 0 {
        return Err(ProverError::InvalidStageRequest {
            reason: format!("Stage 4 {kind:?} advice is present but configured size is zero"),
        });
    }
    if bytes.len() > max_size as usize {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 {kind:?} advice has {} bytes, exceeding configured max {max_size}",
                bytes.len()
            ),
        });
    }

    let start_index = layout
        .remapped_word_address(start_address)
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: format!("Stage 4 {kind:?} advice start address is invalid: {error}"),
        })? as usize;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    if advice_num_vars > r_address.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 {kind:?} advice point needs {advice_num_vars} variables but RAM address has {}",
                r_address.len()
            ),
        });
    }
    let selector =
        block_selector_mle_msb(start_index, advice_num_vars, r_address).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: format!("Stage 4 {kind:?} advice selector failed: {error}"),
            }
        })?;
    let opening_point = r_address[r_address.len() - advice_num_vars..].to_vec();
    let words = advice_words_le(bytes);
    let opening_claim = sparse_segments_mle_msb([(0, words.as_slice())], &opening_point);
    *full_eval += selector * opening_claim;
    contributions.push(Stage4RamValCheckAdviceContribution {
        kind,
        selector,
        opening_claim,
        opening_point,
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
