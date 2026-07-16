//! The Akita final opening.
//!
//! Wjolt is a native group of uniform one-hot members, all opened directly at
//! one canonical point. Optional advice and committed-program objects have
//! distinct domains and are discharged separately through
//! [`jolt_openings::verify_packed_openings`].

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::lattice::geometry::word_byte_num_vars;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_bytes_packing, precommitted_packing, PrecommittedPackingShape, WJoltShape,
};
use jolt_claims::protocols::jolt::lattice::strategy::{WJoltLayoutPlan, W_JOLT_LAYOUT};
use jolt_claims::protocols::jolt::{
    JoltAdviceKind, JoltCommittedPolynomial, JoltOneHotConfig, JoltOpeningId, JoltPolynomialId,
};
use jolt_field::{Field, FixedByteSize};
use jolt_openings::{
    verify_packed_openings, CommitmentScheme, EvaluationClaim, PackedObjectGroup,
    PackedVerifierObject, PrefixPackedStatement, PrefixPacking,
};
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, Transcript};

use super::reconstruction::ReconstructionClearOutput;
use crate::stages::stage7::outputs::Stage7ClearOutput;
use crate::stages::stage8::{WJoltCommitmentMetadata, WJoltSetupMetadata};
use crate::VerifierError;

fn batch_failed(reason: impl ToString) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    }
}

fn opening_failed(reason: impl ToString) -> VerifierError {
    VerifierError::FinalOpeningVerificationFailed {
        reason: reason.to_string(),
    }
}

fn validate_wjolt_metadata<C, S>(
    commitment: &C,
    setup: &S,
    canonical_digest: [u8; 32],
    member_arity: usize,
    member_count: usize,
) -> Result<(), VerifierError>
where
    C: WJoltCommitmentMetadata,
    S: WJoltSetupMetadata,
{
    if !commitment.is_one_hot_backend() {
        return Err(batch_failed(
            "Wjolt commitment must use Akita's one-hot backend",
        ));
    }
    if commitment.layout_digest() != canonical_digest {
        return Err(batch_failed(
            "Wjolt commitment has a noncanonical layout digest",
        ));
    }
    if commitment.num_vars() != member_arity || setup.max_num_vars() != member_arity {
        return Err(batch_failed(format!(
            "Wjolt commitment/setup arity must equal canonical arity {member_arity}"
        )));
    }
    if commitment.poly_count() != member_count
        || setup.max_num_polys_per_commitment_group() != member_count
    {
        return Err(batch_failed(format!(
            "Wjolt commitment/setup member count must equal canonical count {member_count}"
        )));
    }
    if setup.default_layout_digest() != canonical_digest {
        return Err(batch_failed(
            "Wjolt verifier setup has a noncanonical layout digest",
        ));
    }
    Ok(())
}

/// A byte column's word-variable count, recovered from its leaf claim's
/// arity (the `(byte ‖ place)` cell prefix is fixed).
fn leaf_word_vars(cell_vars: usize) -> Result<usize, VerifierError> {
    let cell_prefix_vars = word_byte_num_vars(0);
    cell_vars.checked_sub(cell_prefix_vars).ok_or_else(|| {
        batch_failed(format!(
            "byte-column leaf has {cell_vars} variables, below the \
             {cell_prefix_vars}-variable cell prefix"
        ))
    })
}

/// One resolved commitment object: its canonical packing plus the borrowed
/// commitment and shape-exact setup the final PCS opening runs against.
type ResolvedObject<'a, PCS> = (
    PrefixPacking<JoltCommittedPolynomial>,
    &'a <PCS as jolt_crypto::Commitment>::Output,
    &'a <PCS as CommitmentScheme>::VerifierSetup,
);

/// Resolve one advice object's packing/commitment/setup triple, or `None`
/// when the kind is absent; a commitment without a reconstruction leaf, or a
/// present object missing its commitment or setup, is rejected fail-closed.
fn advice_object<'a, PCS: CommitmentScheme>(
    present: Option<&Vec<PCS::Field>>,
    commitment: Option<&'a PCS::Output>,
    setup: Option<&'a PCS::VerifierSetup>,
    kind: JoltAdviceKind,
) -> Result<Option<ResolvedObject<'a, PCS>>, VerifierError> {
    let Some(leaf_point) = present else {
        if commitment.is_some() {
            return Err(batch_failed(format!(
                "{kind:?} advice commitment supplied without a reconstruction leaf"
            )));
        }
        return Ok(None);
    };
    let (Some(commitment), Some(setup)) = (commitment, setup) else {
        return Err(batch_failed(format!(
            "{kind:?} advice object without a commitment or setup"
        )));
    };
    let packing =
        advice_bytes_packing(kind, leaf_word_vars(leaf_point.len())?).map_err(batch_failed)?;
    Ok(Some((packing, commitment, setup)))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the per-object commitments and their preprocessing setups, resolved here in one place"
)]
pub fn verify<PCS, VC, T>(
    formula_dimensions: &JoltFormulaDimensions,
    one_hot_config: JoltOneHotConfig,
    preprocessing: &crate::preprocessing::JoltVerifierPreprocessing<PCS, VC>,
    w_jolt_commitment: &PCS::Output,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    proof: &crate::proof::AkitaJointOpeningProof<PCS::Field, PCS::Proof>,
    transcript: &mut T,
    stage7: &Stage7ClearOutput<PCS::Field>,
    reconstruction: &ReconstructionClearOutput<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: Clone + AppendToTranscript + WJoltCommitmentMetadata,
    PCS::VerifierSetup: WJoltSetupMetadata,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    // Per-object packings, commitments, and setups in canonical object order:
    // `W_jolt` is one native group of uniform one-hot members, followed by the
    // optional auxiliary commitment objects. The shared layout is the same
    // one the prover committed under.
    // Optional objects join exactly when their reconstruction outputs exist;
    // presence must agree with the proof/preprocessing commitment slots.
    let chunk_width = one_hot_config.committed_chunk_bits();
    let wjolt_shape = WJoltShape {
        ra_layout: formula_dimensions.ra_layout,
        log_t: formula_dimensions.trace.log_t(),
        log_k_chunk: chunk_width,
    };
    let plan = W_JOLT_LAYOUT.plan(&wjolt_shape).map_err(batch_failed)?;
    let canonical_digest = W_JOLT_LAYOUT
        .layout_digest(&wjolt_shape)
        .map_err(batch_failed)?;
    let WJoltLayoutPlan {
        members,
        member_arity,
    } = &plan;
    validate_wjolt_metadata(
        w_jolt_commitment,
        &preprocessing.pcs_setup,
        canonical_digest,
        *member_arity,
        members.len(),
    )?;
    let leaves = leaf_claims(stage7, reconstruction);
    let mut common_point: Option<Vec<PCS::Field>> = None;
    let mut evaluations = Vec::with_capacity(members.len());
    for polynomial in members {
        let claim = leaves
            .get(polynomial)
            .ok_or_else(|| batch_failed(format!("missing final Wjolt claim for {polynomial:?}")))?;
        let point = W_JOLT_LAYOUT
            .member_point(*polynomial, chunk_width, claim.point.as_slice())
            .map_err(batch_failed)?;
        if let Some(expected) = &common_point {
            if expected != &point {
                return Err(batch_failed(format!(
                    "Wjolt member {polynomial:?} does not share the canonical opening point"
                )));
            }
        } else {
            common_point = Some(point);
        }
        evaluations.push(claim.value);
    }
    let common_point = common_point.ok_or_else(|| batch_failed("Wjolt has no members"))?;
    PCS::verify_batch(
        w_jolt_commitment,
        &common_point,
        &evaluations,
        &proof.w_jolt,
        &preprocessing.pcs_setup,
        transcript,
    )
    .map_err(opening_failed)?;

    let mut packings = Vec::new();
    let mut commitments = Vec::new();
    let mut setups = Vec::new();
    let mut groups = Vec::new();

    if let Some((packing, commitment, setup)) = advice_object::<PCS>(
        reconstruction
            .output_points
            .untrusted_advice
            .as_ref()
            .map(|points| &points.bytes),
        untrusted_advice_commitment,
        preprocessing.untrusted_advice_setup.as_ref(),
        JoltAdviceKind::Untrusted,
    )? {
        groups.push(PackedObjectGroup::singleton(packings.len()));
        packings.push(packing);
        commitments.push(commitment);
        setups.push(setup);
    }
    if let Some((packing, commitment, setup)) = advice_object::<PCS>(
        reconstruction
            .output_points
            .trusted_advice
            .as_ref()
            .map(|points| &points.bytes),
        trusted_advice_commitment,
        preprocessing.trusted_advice_setup.as_ref(),
        JoltAdviceKind::Trusted,
    )? {
        groups.push(PackedObjectGroup::singleton(packings.len()));
        packings.push(packing);
        commitments.push(commitment);
        setups.push(setup);
    }
    match (
        reconstruction.output_points.bytecode.as_ref(),
        preprocessing.program.committed(),
    ) {
        (Some(bytecode_points), Some(committed)) => {
            let setup = preprocessing.w_prog_setup.as_ref().ok_or_else(|| {
                batch_failed("committed-program object without a verifier setup in preprocessing")
            })?;
            // The `W_prog` shape is claim-derived: the packing must match the
            // committed witness or its PCS opening fails, so the lane/image
            // point arities are an honest source for the row/word counts.
            let log_bytecode_rows = bytecode_points
                .pc_bytes
                .first()
                .map(|point| leaf_word_vars(point.len()))
                .transpose()?
                .ok_or_else(|| batch_failed("program reconstruction has no pc lanes"))?;
            let program_image_log_words = reconstruction
                .output_points
                .program_image
                .as_ref()
                .map(|points| leaf_word_vars(points.bytes.len()))
                .transpose()?;
            groups.push(PackedObjectGroup::singleton(packings.len()));
            packings.push(
                precommitted_packing(&PrecommittedPackingShape {
                    bytecode_chunks: committed.bytecode_chunk_count(),
                    log_bytecode_rows,
                    imm_byte_width: <PCS::Field as FixedByteSize>::NUM_BYTES,
                    program_image_log_words,
                })
                .map_err(batch_failed)?,
            );
            commitments.push(&committed.w_prog_commitment);
            setups.push(setup);
        }
        (None, None) => {}
        (Some(_), None) => {
            return Err(batch_failed(
                "program reconstruction leaves without a W_prog commitment",
            ));
        }
        (None, Some(_)) => {
            return Err(batch_failed(
                "W_prog commitment supplied without program reconstruction leaves",
            ));
        }
    }

    // Auxiliary objects retain their own logical leaf points and packings.
    let statements = packings
        .iter()
        .zip(&commitments)
        .map(|(packing, commitment)| object_statement(packing, (*commitment).clone(), &leaves))
        .collect::<Result<Vec<_>, _>>()?;
    let objects: Vec<PackedVerifierObject<'_, PCS, JoltCommittedPolynomial>> = packings
        .iter()
        .zip(&statements)
        .zip(setups)
        .map(|((packing, statement), setup)| PackedVerifierObject {
            packing,
            statement,
            setup,
        })
        .collect();

    match (&proof.auxiliary, objects.is_empty()) {
        (None, true) => Ok(()),
        (Some(auxiliary), false) => {
            verify_packed_openings(&objects, &groups, auxiliary, transcript).map_err(opening_failed)
        }
        (None, false) => Err(batch_failed("missing auxiliary packed opening proof")),
        (Some(_), true) => Err(batch_failed("unexpected auxiliary packed opening proof")),
    }
}

/// Assembles one object's statement: each of its packing's canonical columns
/// paired with that column's leaf claim.
fn object_statement<F, C>(
    packing: &PrefixPacking<JoltCommittedPolynomial>,
    commitment: C,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
) -> Result<PrefixPackedStatement<F, JoltCommittedPolynomial, C>, VerifierError>
where
    F: Field,
{
    let claims = packing
        .iter()
        .map(|(polynomial, _slot)| {
            leaves
                .get(polynomial)
                .cloned()
                .map(|claim| (*polynomial, claim))
                .ok_or_else(|| {
                    batch_failed(format!(
                        "missing stage output claim for packed leaf {polynomial:?}"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PrefixPackedStatement::new(commitment, claims))
}

/// Every packed column's single leaf claim, resolved from the stage-7 and
/// reconstruction outputs and keyed by committed polynomial. Coverage against
/// the packings is machine-checked downstream by `prepare_statement`
/// (one-claim-per-slot, no gaps, per-slot point arity).
fn leaf_claims<F: Field>(
    stage7: &Stage7ClearOutput<F>,
    reconstruction: &ReconstructionClearOutput<F>,
) -> BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>> {
    use JoltCommittedPolynomial as Poly;

    fn leaf<F: Field>(value: F, point: &[F]) -> EvaluationClaim<F> {
        EvaluationClaim::new(Point::high_to_low(point.to_vec()), value)
    }
    fn insert<F: Field>(
        leaves: &mut BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
        polynomial: JoltCommittedPolynomial,
        claim: EvaluationClaim<F>,
    ) {
        // Keys are distinct by construction, so no entry is ever displaced.
        let _previous = BTreeMap::insert(leaves, polynomial, claim);
    }
    fn insert_indexed<F: Field>(
        leaves: &mut BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
        values: &[F],
        points: &[Vec<F>],
        polynomial: impl Fn(usize) -> JoltCommittedPolynomial,
    ) {
        for (index, (value, point)) in values.iter().zip(points).enumerate() {
            insert(leaves, polynomial(index), leaf(*value, point));
        }
    }
    let mut leaves = BTreeMap::new();

    let hamming_values = &stage7.output_values.hamming_weight_claim_reduction;
    let hamming_points = &stage7.output_points.hamming_weight_claim_reduction;
    insert_indexed(
        &mut leaves,
        &hamming_values.instruction_ra,
        &hamming_points.instruction_ra,
        Poly::InstructionRa,
    );
    insert_indexed(
        &mut leaves,
        &hamming_values.bytecode_ra,
        &hamming_points.bytecode_ra,
        Poly::BytecodeRa,
    );
    insert_indexed(
        &mut leaves,
        &hamming_values.ram_ra,
        &hamming_points.ram_ra,
        Poly::RamRa,
    );

    insert_indexed(
        &mut leaves,
        &hamming_values.unsigned_inc_chunks,
        &hamming_points.unsigned_inc_chunks,
        Poly::UnsignedIncChunk,
    );
    insert(
        &mut leaves,
        Poly::UnsignedIncMsb,
        leaf(
            hamming_values.unsigned_inc_msb,
            &hamming_points.unsigned_inc_msb,
        ),
    );

    if let Some((values, points)) = reconstruction
        .output_values
        .untrusted_advice
        .as_ref()
        .zip(reconstruction.output_points.untrusted_advice.as_ref())
    {
        insert(
            &mut leaves,
            Poly::UntrustedAdviceBytes,
            leaf(values.bytes, &points.bytes),
        );
    }
    if let Some((values, points)) = reconstruction
        .output_values
        .trusted_advice
        .as_ref()
        .zip(reconstruction.output_points.trusted_advice.as_ref())
    {
        insert(
            &mut leaves,
            Poly::TrustedAdviceBytes,
            leaf(values.bytes, &points.bytes),
        );
    }
    if let Some((values, points)) = reconstruction
        .output_values
        .program_image
        .as_ref()
        .zip(reconstruction.output_points.program_image.as_ref())
    {
        insert(
            &mut leaves,
            Poly::ProgramImageBytes,
            leaf(values.bytes, &points.bytes),
        );
    }

    // The bytecode leaf keys are read off the canonical cell order jolt-claims
    // pins (`leaves()` pairs one-for-one with `opening_order`), instead of
    // re-deriving the chunk/lane index arithmetic here.
    if let Some((values, points)) = reconstruction
        .output_values
        .bytecode
        .as_ref()
        .zip(reconstruction.output_points.bytecode.as_ref())
    {
        for ((id, value), (_, point)) in values.leaves().zip(points.leaves()) {
            let JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(polynomial),
                ..
            } = id
            else {
                continue;
            };
            insert(&mut leaves, polynomial, leaf(*value, point));
        }
    }

    leaves
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
        committed_lane_vars, BYTECODE_LANE_LAYOUT,
    };
    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::lattice::relations::advice_reconstruction::{
        TrustedAdviceReconstructionOutputClaims, UntrustedAdviceReconstructionOutputClaims,
    };
    use jolt_claims::protocols::jolt::lattice::relations::bytecode_reconstruction::BytecodeChunkReconstructionOutputClaims;
    use jolt_claims::protocols::jolt::lattice::relations::program_image_reconstruction::ProgramImageReconstructionOutputClaims;
    use jolt_claims::protocols::jolt::BytecodeRegisterLane;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::math::Math;
    use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

    use super::super::reconstruction::{ReconstructionOutputClaims, ReconstructionOutputPoints};
    use crate::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims;
    use crate::stages::stage7::outputs::{Stage7OutputClaims, Stage7OutputPoints};

    const LOG_T: usize = 4;
    const LOG_K_CHUNK: usize = 8;
    const INC_CHUNKS: usize = 8;
    const BYTECODE_CHUNKS: usize = 2;
    const LOG_BYTECODE_ROWS: usize = 6;
    const LOG_IMAGE_WORDS: usize = 5;
    const ADVICE_WORD_VARS: usize = 3;

    #[derive(Clone, Copy)]
    struct CommitmentMetadata {
        one_hot: bool,
        digest: [u8; 32],
        num_vars: usize,
        poly_count: usize,
    }

    impl WJoltCommitmentMetadata for CommitmentMetadata {
        fn is_one_hot_backend(&self) -> bool {
            self.one_hot
        }

        fn layout_digest(&self) -> [u8; 32] {
            self.digest
        }

        fn num_vars(&self) -> usize {
            self.num_vars
        }

        fn poly_count(&self) -> usize {
            self.poly_count
        }
    }

    #[derive(Clone, Copy)]
    struct SetupMetadata {
        digest: [u8; 32],
        num_vars: usize,
        poly_count: usize,
    }

    impl WJoltSetupMetadata for SetupMetadata {
        fn max_num_vars(&self) -> usize {
            self.num_vars
        }

        fn max_num_polys_per_commitment_group(&self) -> usize {
            self.poly_count
        }

        fn default_layout_digest(&self) -> [u8; 32] {
            self.digest
        }
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[test]
    fn wjolt_metadata_is_enforced_before_pcs_verification() {
        let digest = [7; 32];
        let commitment = CommitmentMetadata {
            one_hot: true,
            digest,
            num_vars: 12,
            poly_count: 17,
        };
        let setup = SetupMetadata {
            digest,
            num_vars: 12,
            poly_count: 17,
        };
        assert!(validate_wjolt_metadata(&commitment, &setup, digest, 12, 17).is_ok());

        for invalid in [
            CommitmentMetadata {
                one_hot: false,
                ..commitment
            },
            CommitmentMetadata {
                digest: [8; 32],
                ..commitment
            },
            CommitmentMetadata {
                num_vars: 13,
                ..commitment
            },
            CommitmentMetadata {
                poly_count: 18,
                ..commitment
            },
        ] {
            assert!(validate_wjolt_metadata(&invalid, &setup, digest, 12, 17).is_err());
        }
        for invalid in [
            SetupMetadata {
                digest: [9; 32],
                ..setup
            },
            SetupMetadata {
                num_vars: 13,
                ..setup
            },
            SetupMetadata {
                poly_count: 18,
                ..setup
            },
        ] {
            assert!(validate_wjolt_metadata(&commitment, &invalid, digest, 12, 17).is_err());
        }
    }

    fn point(arity: usize) -> Vec<Fr> {
        vec![fr(1); arity]
    }

    fn stage7(layout: JoltRaPolynomialLayout) -> Stage7ClearOutput<Fr> {
        let one_hot_arity = LOG_K_CHUNK + LOG_T;
        let hamming_values = HammingWeightClaimReductionOutputClaims {
            instruction_ra: (0..layout.instruction())
                .map(|i| fr(100 + i as u64))
                .collect(),
            bytecode_ra: (0..layout.bytecode()).map(|i| fr(200 + i as u64)).collect(),
            ram_ra: (0..layout.ram()).map(|i| fr(300 + i as u64)).collect(),
            unsigned_inc_chunks: (0..INC_CHUNKS).map(|i| fr(400 + i as u64)).collect(),
            unsigned_inc_msb: fr(500),
        };
        let hamming_points = HammingWeightClaimReductionOutputClaims {
            instruction_ra: vec![point(one_hot_arity); layout.instruction()],
            bytecode_ra: vec![point(one_hot_arity); layout.bytecode()],
            ram_ra: vec![point(one_hot_arity); layout.ram()],
            unsigned_inc_chunks: vec![point(one_hot_arity); INC_CHUNKS],
            unsigned_inc_msb: point(one_hot_arity),
        };
        Stage7ClearOutput {
            output_values: Stage7OutputClaims {
                hamming_weight_claim_reduction: hamming_values,
                trusted_advice: None,
                untrusted_advice: None,
                bytecode_address_phase: None,
                program_image_address_phase: None,
            },
            output_points: Stage7OutputPoints {
                hamming_weight_claim_reduction: hamming_points,
                trusted_advice: None,
                untrusted_advice: None,
                bytecode_address_phase: None,
                program_image_address_phase: None,
            },
        }
    }

    fn reconstruction() -> ReconstructionClearOutput<Fr> {
        let advice_arity = word_byte_num_vars(ADVICE_WORD_VARS);
        let selectors = BYTECODE_CHUNKS * BytecodeRegisterLane::ALL.len();
        let bytecode_values = BytecodeChunkReconstructionOutputClaims {
            register_selectors: (0..selectors).map(|i| fr(600 + i as u64)).collect(),
            circuit_flags: (0..BYTECODE_CHUNKS * NUM_CIRCUIT_FLAGS)
                .map(|i| fr(700 + i as u64))
                .collect(),
            instruction_flags: (0..BYTECODE_CHUNKS * NUM_INSTRUCTION_FLAGS)
                .map(|i| fr(800 + i as u64))
                .collect(),
            lookup_selectors: (0..BYTECODE_CHUNKS).map(|i| fr(900 + i as u64)).collect(),
            raf_flags: (0..BYTECODE_CHUNKS).map(|i| fr(910 + i as u64)).collect(),
            pc_bytes: (0..BYTECODE_CHUNKS).map(|i| fr(920 + i as u64)).collect(),
            imm_bytes: (0..BYTECODE_CHUNKS).map(|i| fr(930 + i as u64)).collect(),
        };
        let lookup_arity = (BYTECODE_LANE_LAYOUT.raf_flag_idx - BYTECODE_LANE_LAYOUT.lookup_start)
            .log_2()
            + LOG_BYTECODE_ROWS;
        let bytecode_points = BytecodeChunkReconstructionOutputClaims {
            register_selectors: vec![point(REGISTER_ADDRESS_BITS + LOG_BYTECODE_ROWS); selectors],
            circuit_flags: vec![point(LOG_BYTECODE_ROWS); BYTECODE_CHUNKS * NUM_CIRCUIT_FLAGS],
            instruction_flags: vec![
                point(LOG_BYTECODE_ROWS);
                BYTECODE_CHUNKS * NUM_INSTRUCTION_FLAGS
            ],
            lookup_selectors: vec![point(lookup_arity); BYTECODE_CHUNKS],
            raf_flags: vec![point(LOG_BYTECODE_ROWS); BYTECODE_CHUNKS],
            pc_bytes: vec![point(word_byte_num_vars(LOG_BYTECODE_ROWS)); BYTECODE_CHUNKS],
            imm_bytes: vec![
                point(
                    jolt_claims::protocols::jolt::lattice::geometry::byte_num_vars(
                        <Fr as FixedByteSize>::NUM_BYTES,
                        LOG_BYTECODE_ROWS,
                    )
                    .unwrap()
                );
                BYTECODE_CHUNKS
            ],
        };
        ReconstructionClearOutput {
            output_values: ReconstructionOutputClaims {
                untrusted_advice: Some(UntrustedAdviceReconstructionOutputClaims { bytes: fr(41) }),
                trusted_advice: Some(TrustedAdviceReconstructionOutputClaims { bytes: fr(43) }),
                bytecode: Some(bytecode_values),
                program_image: Some(ProgramImageReconstructionOutputClaims { bytes: fr(47) }),
            },
            output_points: ReconstructionOutputPoints {
                untrusted_advice: Some(UntrustedAdviceReconstructionOutputClaims {
                    bytes: point(advice_arity),
                }),
                trusted_advice: Some(TrustedAdviceReconstructionOutputClaims {
                    bytes: point(advice_arity),
                }),
                bytecode: Some(bytecode_points),
                program_image: Some(ProgramImageReconstructionOutputClaims {
                    bytes: point(word_byte_num_vars(LOG_IMAGE_WORDS)),
                }),
            },
        }
    }

    /// Every auxiliary object's packing resolves exactly one leaf claim per
    /// column at the slot's arity — `prepare_statement` machine-checks
    /// one-claim-per-slot, full coverage, and per-slot point arity, so a
    /// passing preparation pins the leaf-resolution map against the canonical
    /// packings.
    #[test]
    fn auxiliary_packed_statements_cover_every_column_at_slot_arity() {
        let layout = JoltRaPolynomialLayout::new(2, 1, 1).unwrap();
        let leaves = leaf_claims(&stage7(layout), &reconstruction());

        let objects = [
            advice_bytes_packing(JoltAdviceKind::Untrusted, ADVICE_WORD_VARS).unwrap(),
            advice_bytes_packing(JoltAdviceKind::Trusted, ADVICE_WORD_VARS).unwrap(),
            precommitted_packing(&PrecommittedPackingShape {
                bytecode_chunks: BYTECODE_CHUNKS,
                log_bytecode_rows: LOG_BYTECODE_ROWS,
                imm_byte_width: <Fr as FixedByteSize>::NUM_BYTES,
                program_image_log_words: Some(LOG_IMAGE_WORDS),
            })
            .unwrap(),
        ];
        for packing in &objects {
            let statement = object_statement(packing, (), &leaves).unwrap();
            assert_eq!(statement.claims.len(), packing.iter().count());
            let prepared = packing.prepare_statement(&statement).unwrap();
            assert_eq!(prepared.num_claims(), packing.iter().count());
        }
    }

    /// The lane-vars split the leaf resolver relies on matches the completed
    /// chunk claims the reconstruction consumes.
    #[test]
    fn committed_lane_split_matches_layout() {
        assert_eq!(
            committed_lane_vars(),
            jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY
                .log_2()
        );
    }
}
