//! Stage 8: the joint batched opening — no sumcheck, one homomorphic PCS
//! batch over every committed polynomial.
//!
//! Pure orchestration mirroring `stage8::verify`: the batch entries come from
//! the verifier's promoted `batch_entries` assembly (the prover passes its
//! own stage-0 commitments where the verifier passes the proof's), the
//! unified opening point and per-entry embedding scales are the verifier's
//! own `final_opening_point` / `commitment_embedding_scale` math, and the
//! whole transcript sequence — scaled-claim absorbs, one gamma-powers
//! squeeze, the PCS opening, the final evaluation-claim absorb — is
//! `HomomorphicBatch::prove_batch`, byte-identical to the verifier's inlined
//! sequence. The prover-only work is the grid-embedded witness materialization
//! (the backend's joint-opening slot; advice polynomials block-embed rather
//! than prefix-embed) and the hint reordering + combination (stage 0 retains
//! hints in proof-commitment order, advice hints included; the batch runs in
//! final-opening order, and the ragged advice hints pad with identity rows in
//! `combine_hints`).

use jolt_claims::protocols::jolt::geometry::committed_openings::{
    final_opening_point, final_opening_polynomial_order, FinalOpeningPointInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use std::collections::BTreeMap;

use jolt_kernels::committed_program::{
    build_committed_bytecode_chunk_coeffs, program_image_words_padded,
};
use jolt_kernels::{CommitmentGrid, JoltBackend, KernelError, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{
    AdditivelyHomomorphic, BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch,
    VerifierOpeningClaim,
};
use jolt_poly::Point;
use jolt_transcript::Transcript;
use jolt_verifier::proof::JoltCommitments;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::{batch_entries, precommitted_final_openings};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{CommittedProgramCandidates, JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 8's output: the joint PCS opening proof (the last wire component of
/// a clear proof).
pub struct Stage8ProverOutput<PCS: CommitmentScheme> {
    pub joint_opening_proof: PCS::Proof,
}

/// Prove stage 8 on `transcript` (positioned at the stage-7 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage8<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    commitments: &JoltCommitments<PCS::Output>,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    hints: &[(JoltCommittedPolynomial, PCS::OpeningHint)],
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    transcript: &mut T,
) -> Result<Stage8ProverOutput<PCS>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let precommitted = &checked.precommitted;
    let formula_dimensions = JoltFormulaDimensions::try_from(config.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.verifier.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: error.to_string(),
    })?;
    let layout = formula_dimensions.ra_layout;

    // The assembly, exactly as `stage8::verify` performs it before any
    // transcript operation.
    let hamming_opening_point = stage7
        .output_points
        .hamming_weight_opening_point()
        .map(<[F]>::to_vec)
        .ok_or(ProverError::Unsupported {
            reason: "stage 7 produced no hamming-weight openings",
        })?;
    let inc_opening_point = stage6b.output_points.inc_opening_point();
    let precommitted_finals = precommitted_final_openings(
        precommitted,
        &stage7.output_points,
        &stage6b.output_points,
        Some((&stage7.output_values, &stage6b.output_values)),
    )?;
    let anchor_points: Vec<&[F]> = precommitted_finals
        .iter()
        .map(|opening| opening.point.as_slice())
        .collect();
    let opening_point = final_opening_point(FinalOpeningPointInputs {
        log_t,
        log_k_chunk: config.one_hot_config.committed_chunk_bits(),
        trace_order: config.trace_polynomial_order,
        hamming_weight_opening_point: hamming_opening_point.as_slice(),
        inc_claim_reduction_opening_point: inc_opening_point,
        precommitted_anchor_points: &anchor_points,
    })
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;
    let pcs_opening_point = Point::high_to_low(opening_point.clone());

    let entries = batch_entries::<F, PCS, VC>(
        &preprocessing.verifier,
        commitments,
        untrusted_advice_commitment,
        layout,
        trusted_advice_commitment,
        &opening_point,
        hamming_opening_point.as_slice(),
        inc_opening_point,
        &precommitted_finals,
        Some((&stage6b.output_values, &stage7.output_values)),
    )?;
    let statement: Vec<VerifierOpeningClaim<F, PCS::Output>> = entries
        .iter()
        .map(|entry| {
            let opening_claim = entry.opening_claim.ok_or(ProverError::Unsupported {
                reason: "stage-8 batch entry carries no clear opening claim",
            })?;
            Ok(VerifierOpeningClaim {
                commitment: entry.commitment.clone(),
                evaluation: EvaluationClaim::new(
                    pcs_opening_point.clone(),
                    opening_claim * entry.scale,
                ),
            })
        })
        .collect::<Result<_, ProverError<F>>>()?;

    // Witness materialization (grid-embedded, batch order) and the hints
    // reordered from stage 0's proof-commitment order.
    let include_trusted = precommitted.trusted_advice.is_some();
    let include_untrusted = precommitted.untrusted_advice.is_some();
    let chunk_count = precommitted
        .bytecode
        .as_ref()
        .map(|layout| layout.chunk_count());
    let order =
        final_opening_polynomial_order(layout, include_trusted, include_untrusted, chunk_count);
    let grid = CommitmentGrid {
        total_vars: config.commitment_total_vars(
            preprocessing.verifier.program.memory_layout(),
            include_trusted,
            include_untrusted,
            CommittedProgramCandidates::from_schedule(precommitted),
        ),
        log_t,
        log_k_chunk: config.one_hot_config.committed_chunk_bits(),
        order: config.trace_polynomial_order,
    };
    if grid.total_vars != opening_point.len() {
        return Err(ProverError::Unsupported {
            reason: "commitment grid width disagrees with the unified opening point",
        });
    }
    // The committed-program polynomials are preprocessing data (not witness
    // oracles): materialize them from the prover-retained full program.
    let mut precommitted_tables: BTreeMap<JoltCommittedPolynomial, Vec<F>> = BTreeMap::new();
    if let Some(bytecode_layout) = &precommitted.bytecode {
        let program = preprocessing.program().ok_or(ProverError::Unsupported {
            reason: "full program preprocessing is unavailable",
        })?;
        let chunk_coeffs = build_committed_bytecode_chunk_coeffs::<F>(
            &program.bytecode.bytecode,
            bytecode_layout.chunk_count(),
            bytecode_layout.trace_order(),
        )?;
        for (index, coeffs) in chunk_coeffs.into_iter().enumerate() {
            let _ =
                precommitted_tables.insert(JoltCommittedPolynomial::BytecodeChunk(index), coeffs);
        }
        let image_words = program_image_words_padded(&program.ram.bytecode_words);
        let _ = precommitted_tables.insert(
            JoltCommittedPolynomial::ProgramImageInit,
            image_words.into_iter().map(F::from_u64).collect(),
        );
    }
    let polynomials =
        backend
            .joint_opening
            .prepare(session, witness, &order, &precommitted_tables, grid)?;
    let ordered_hints: Vec<PCS::OpeningHint> = order
        .iter()
        .map(|polynomial| {
            hints
                .iter()
                .find(|(id, _)| id == polynomial)
                .map(|(_, hint)| hint.clone())
                .ok_or(ProverError::Unsupported {
                    reason: "missing stage-0 opening hint for a batched polynomial",
                })
        })
        .collect::<Result<_, _>>()?;

    let joint_opening_proof = HomomorphicBatch::<PCS>::prove_batch(
        &preprocessing.pcs_setup,
        statement,
        polynomials.iter().map(|poly| &**poly).collect(),
        ordered_hints,
        transcript,
    )
    .map_err(KernelError::<F>::from)?;

    Ok(Stage8ProverOutput {
        joint_opening_proof,
    })
}
