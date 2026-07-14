//! Statement assembly for the homomorphic final opening: resolves each
//! committed polynomial's commitment, clear-mode claim, and unified-point
//! embedding scale into the batch entries [`super::verify`] discharges.

use super::precommitted::PrecommittedFinalOpening;
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltCommitments, JoltProof},
    stages::{stage6b::outputs::Stage6bOutputClaims, stage7::outputs::Stage7OutputClaims},
    VerifierError,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        committed_openings::{
            commitment_embedding_scale, final_opening_id, final_opening_polynomial_order,
        },
        ra::JoltRaPolynomialLayout,
    },
    JoltCommittedPolynomial, JoltOpeningId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;

pub(super) struct Stage8BatchEntry<'a, F: Field, C> {
    pub(super) id: JoltOpeningId,
    pub(super) commitment: &'a C,
    /// `None` in ZK mode, where opening claims stay committed.
    pub(super) opening_claim: Option<F>,
    /// Lagrange factor embedding this polynomial's own opening point into the
    /// unified opening point.
    pub(super) scale: F,
}

/// One-hot RA entry: the family's `index`-th commitment, its (clear mode)
/// hamming-weight claim, and the hamming point's embedding scale.
fn ra_entry<'a, F: Field, C>(
    commitments: &'a [C],
    claims: Option<&[F]>,
    index: usize,
    polynomial: JoltCommittedPolynomial,
    opening_point: &[F],
    hamming_opening_point: &[F],
) -> Result<Stage8BatchEntry<'a, F, C>, VerifierError> {
    let id = final_opening_id(polynomial);
    let commitment = commitments
        .get(index)
        .ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
    let opening_claim = claims
        .map(|claims| {
            claims
                .get(index)
                .copied()
                .ok_or(VerifierError::MissingOpeningClaim { id })
        })
        .transpose()?;
    Ok(Stage8BatchEntry {
        id,
        commitment,
        opening_claim,
        scale: commitment_embedding_scale(opening_point, hamming_opening_point),
    })
}

/// Precommitted entry: the polynomial's externally-supplied commitment and
/// its resolved final opening (point anchor plus clear-mode claim).
fn precommitted_entry<'a, F: Field, C>(
    commitment: Option<&'a C>,
    opening: Option<&'a PrecommittedFinalOpening<F>>,
    polynomial: JoltCommittedPolynomial,
    opening_point: &[F],
) -> Result<Stage8BatchEntry<'a, F, C>, VerifierError> {
    let id = final_opening_id(polynomial);
    let opening = opening.ok_or(VerifierError::MissingOpeningClaim { id })?;
    let commitment =
        commitment.ok_or(VerifierError::MissingFinalOpeningCommitment { polynomial })?;
    Ok(Stage8BatchEntry {
        id,
        commitment,
        opening_claim: opening.opening_claim,
        scale: commitment_embedding_scale(opening_point, opening.point.as_slice()),
    })
}

/// Builds the final PCS batch in the canonical order from
/// [`final_opening_polynomial_order`], resolving each polynomial's commitment,
/// opening claim (clear mode only), and unified-point embedding scale.
#[expect(
    clippy::too_many_arguments,
    reason = "gathers per-polynomial sources from several stages"
)]
pub(super) fn final_opening_entries<'a, F, PCS, VC, ZkProof>(
    preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    proof: &'a JoltProof<PCS, VC, ZkProof>,
    layout: JoltRaPolynomialLayout,
    trusted_advice_commitment: Option<&'a PCS::Output>,
    opening_point: &[F],
    hamming_opening_point: &[F],
    inc_opening_point: &[F],
    precommitted_finals: &'a [PrecommittedFinalOpening<F>],
    clear_claims: Option<(&Stage6bOutputClaims<F>, &Stage7OutputClaims<F>)>,
) -> Result<Vec<Stage8BatchEntry<'a, F, PCS::Output>>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let precommitted_final = |polynomial: JoltCommittedPolynomial| {
        precommitted_finals
            .iter()
            .find(|opening| opening.polynomial == polynomial)
    };
    let include_trusted = precommitted_final(JoltCommittedPolynomial::TrustedAdvice).is_some();
    let include_untrusted = precommitted_final(JoltCommittedPolynomial::UntrustedAdvice).is_some();
    let committed_program = preprocessing.program.committed();
    let order = final_opening_polynomial_order(
        layout,
        include_trusted,
        include_untrusted,
        committed_program.map(|committed| committed.bytecode_chunk_count()),
    );
    let hamming_claims =
        |select: fn(&Stage7OutputClaims<F>) -> &[F]| clear_claims.map(|(_, stage7)| select(stage7));

    let mut entries = Vec::with_capacity(order.len());
    // The prover's final PCS batch order intentionally differs from proof payload order.
    for polynomial in order {
        let entry = match polynomial {
            JoltCommittedPolynomial::RamInc => Ok(Stage8BatchEntry {
                id: final_opening_id(polynomial),
                commitment: &proof.commitments.ram_inc,
                opening_claim: clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.ram_inc),
                scale: commitment_embedding_scale(opening_point, inc_opening_point),
            }),
            JoltCommittedPolynomial::RdInc => Ok(Stage8BatchEntry {
                id: final_opening_id(polynomial),
                commitment: &proof.commitments.rd_inc,
                opening_claim: clear_claims.map(|(stage6, _)| stage6.inc_claim_reduction.rd_inc),
                scale: commitment_embedding_scale(opening_point, inc_opening_point),
            }),
            JoltCommittedPolynomial::InstructionRa(index) => ra_entry(
                &proof.commitments.ra.instruction,
                hamming_claims(|stage7| &stage7.hamming_weight_claim_reduction.instruction_ra),
                index,
                polynomial,
                opening_point,
                hamming_opening_point,
            ),
            JoltCommittedPolynomial::BytecodeRa(index) => ra_entry(
                &proof.commitments.ra.bytecode,
                hamming_claims(|stage7| &stage7.hamming_weight_claim_reduction.bytecode_ra),
                index,
                polynomial,
                opening_point,
                hamming_opening_point,
            ),
            JoltCommittedPolynomial::RamRa(index) => ra_entry(
                &proof.commitments.ra.ram,
                hamming_claims(|stage7| &stage7.hamming_weight_claim_reduction.ram_ra),
                index,
                polynomial,
                opening_point,
                hamming_opening_point,
            ),
            JoltCommittedPolynomial::TrustedAdvice => precommitted_entry(
                trusted_advice_commitment,
                precommitted_final(polynomial),
                polynomial,
                opening_point,
            ),
            JoltCommittedPolynomial::UntrustedAdvice => precommitted_entry(
                proof.untrusted_advice_commitment.as_ref(),
                precommitted_final(polynomial),
                polynomial,
                opening_point,
            ),
            JoltCommittedPolynomial::BytecodeChunk(index) => precommitted_entry(
                committed_program
                    .and_then(|committed| committed.bytecode_chunk_commitments.get(index)),
                precommitted_final(polynomial),
                polynomial,
                opening_point,
            ),
            JoltCommittedPolynomial::ProgramImageInit => precommitted_entry(
                committed_program.map(|committed| &committed.program_image_commitment),
                precommitted_final(polynomial),
                polynomial,
                opening_point,
            ),
            // Lattice-mode polynomials open through the packed opening
            // (`lattice::packing::final_opening`), never the homomorphic
            // stage 8 RLC batch.
            _ => Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "polynomial {polynomial:?} is not part of the stage 8 prover order"
                ),
            }),
        };
        entries.push(entry?);
    }
    Ok(entries)
}

pub(super) fn require_commitment_layout<C>(
    commitments: &JoltCommitments<C>,
    layout: JoltRaPolynomialLayout,
) -> Result<(), VerifierError> {
    let expected = 2 + layout.total();
    let got = 2
        + commitments.ra.instruction.len()
        + commitments.ra.bytecode.len()
        + commitments.ra.ram.len();
    if got != expected {
        return Err(VerifierError::InvalidCommitmentCount { expected, got });
    }
    if commitments.ra.instruction.len() != layout.instruction()
        || commitments.ra.bytecode.len() != layout.bytecode()
        || commitments.ra.ram.len() != layout.ram()
    {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "commitment layout mismatch: expected instruction={}, bytecode={}, ram={}; got instruction={}, bytecode={}, ram={}",
                layout.instruction(),
                layout.bytecode(),
                layout.ram(),
                commitments.ra.instruction.len(),
                commitments.ra.bytecode.len(),
                commitments.ra.ram.len()
            ),
        });
    }
    Ok(())
}
