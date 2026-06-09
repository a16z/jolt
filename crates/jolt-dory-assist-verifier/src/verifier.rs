//! Top-level Dory-assist verifier entry point.

use jolt_claims::protocols::dory_assist::formulas::composition;
use jolt_crypto::{Bn254G1, Bn254G2, Bn254GT, JoltGroup};
use jolt_dory::{
    DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup, DoryVerifierTranscriptScalars,
};
use jolt_field::{CanonicalBytes, FixedByteSize, Fq, Fr, FromPrimitiveInt};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{Label, LabelWithCount, Transcript};
use jolt_verifier::{PcsAssistClearInput, PcsAssistZkInput, PcsProofAssist};

use crate::{
    artifacts::{
        DoryProofArtifactLayout, G1_ARTIFACT_COORDS, G2_ARTIFACT_COORDS, GT_ARTIFACT_COEFFS,
    },
    config::DoryAssistConfig,
    error::DoryAssistVerifierError,
    native_final::{transparent_final_pairing_check, zk_final_pairing_check},
    proof::{default_dory_assist_dimensions, DoryAssistInputPublicClaims, DoryAssistProof},
    stages,
};

const MAX_DORY_ASSIST_OPENING_POINT_LEN: usize = 64;
const CHECKED_INPUT_DIGEST_LABEL: &[u8] = b"dory_assist_checked_input_digest";
const VERIFIER_SETUP_DIGEST_LABEL: &[u8] = b"dory_assist_setup_digest";
const DORY_PROOF_DIGEST_LABEL: &[u8] = b"dory_assist_proof_digest";
const JOLT_COMMITMENT_DIGEST_LABEL: &[u8] = b"dory_assist_commitment_digest";
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DoryAssist;

impl PcsProofAssist<DoryScheme> for DoryAssist {
    type Proof = DoryAssistProof;
    type Config = DoryAssistConfig;
    type Error = DoryAssistVerifierError;

    fn selected_config() -> Self::Config {
        DoryAssistConfig
    }

    fn verify_clear<T>(
        config: &Self::Config,
        input: PcsAssistClearInput<'_, DoryScheme>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
    {
        crate::verifier::verify_clear(config, input, proof, transcript)
    }

    fn verify_zk<T>(
        config: &Self::Config,
        input: PcsAssistZkInput<'_, DoryScheme>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<<DoryScheme as ZkOpeningScheme>::HidingCommitment, Self::Error>
    where
        T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
    {
        crate::verifier::verify_zk(config, input, proof, transcript)
    }
}

#[derive(Clone, Copy)]
pub enum CheckedInputs<'a> {
    Clear(ClearInputs<'a>),
    Zk(ZkInputs<'a>),
}

impl CheckedInputs<'_> {
    pub const fn zk(&self) -> bool {
        matches!(self, Self::Zk(_))
    }

    pub const fn mode_name(&self) -> &'static str {
        if self.zk() {
            "zk"
        } else {
            "clear"
        }
    }

    pub const fn point(&self) -> &[Fr] {
        match self {
            Self::Clear(inputs) => inputs.opening.point,
            Self::Zk(inputs) => inputs.opening.point,
        }
    }

    pub const fn pcs_proof(&self) -> &DoryProof {
        match self {
            Self::Clear(inputs) => inputs.opening.pcs_proof,
            Self::Zk(inputs) => inputs.opening.pcs_proof,
        }
    }

    pub const fn setup(&self) -> &DoryVerifierSetup {
        match self {
            Self::Clear(inputs) => inputs.opening.setup,
            Self::Zk(inputs) => inputs.opening.setup,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ClearInputs<'a> {
    pub opening: ClearOpeningStatement<'a>,
}

#[derive(Clone, Copy)]
pub struct ZkInputs<'a> {
    pub opening: ZkOpeningStatement<'a>,
}

#[derive(Clone, Copy)]
pub struct ClearOpeningStatement<'a> {
    pub setup: &'a DoryVerifierSetup,
    pub pcs_proof: &'a DoryProof,
    pub commitment: &'a DoryCommitment,
    pub point: &'a [Fr],
    pub eval: Fr,
}

#[derive(Clone, Copy)]
pub struct ZkOpeningStatement<'a> {
    pub setup: &'a DoryVerifierSetup,
    pub pcs_proof: &'a DoryProof,
    pub commitment: &'a DoryCommitment,
    pub point: &'a [Fr],
}

pub fn verify_clear<T>(
    config: &DoryAssistConfig,
    input: PcsAssistClearInput<'_, DoryScheme>,
    proof: &DoryAssistProof,
    transcript: &mut T,
) -> Result<(), DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    let _ = config;
    let checked = checked_clear_inputs(input);
    validate_checked_inputs(&checked)?;
    validate_proof_dimensions(&checked, proof)?;
    let dory_verifier_scalars = dory_verifier_transcript_scalars(&checked, transcript);
    validate_dory_verifier_transcript_scalars(&checked, &dory_verifier_scalars)?;
    let input_public_claims = absorb_checked_inputs(&checked, &dory_verifier_scalars, transcript);
    verify_checked_input_public_claims(proof, input_public_claims, transcript)?;
    let _stage_output = run_stages(&checked, proof, transcript)?;
    match checked {
        CheckedInputs::Clear(inputs) => {
            verify_clear_native_outputs(&inputs.opening, proof, &dory_verifier_scalars)
        }
        CheckedInputs::Zk(_) => Err(DoryAssistVerifierError::InvalidMode {
            expected: "clear",
            got: "zk",
        }),
    }
}

pub fn verify_zk<T>(
    config: &DoryAssistConfig,
    input: PcsAssistZkInput<'_, DoryScheme>,
    proof: &DoryAssistProof,
    transcript: &mut T,
) -> Result<<DoryScheme as ZkOpeningScheme>::HidingCommitment, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    let _ = config;
    let checked = checked_zk_inputs(input);
    validate_checked_inputs(&checked)?;
    validate_proof_dimensions(&checked, proof)?;
    let dory_verifier_scalars = dory_verifier_transcript_scalars(&checked, transcript);
    validate_dory_verifier_transcript_scalars(&checked, &dory_verifier_scalars)?;
    let input_public_claims = absorb_checked_inputs(&checked, &dory_verifier_scalars, transcript);
    verify_checked_input_public_claims(proof, input_public_claims, transcript)?;
    let _stage_output = run_stages(&checked, proof, transcript)?;
    match checked {
        CheckedInputs::Zk(inputs) => {
            verify_zk_native_outputs(&inputs.opening, proof, &dory_verifier_scalars)
        }
        CheckedInputs::Clear(_) => Err(DoryAssistVerifierError::InvalidMode {
            expected: "zk",
            got: "clear",
        }),
    }
}

pub fn checked_clear_inputs(input: PcsAssistClearInput<'_, DoryScheme>) -> CheckedInputs<'_> {
    CheckedInputs::Clear(ClearInputs {
        opening: ClearOpeningStatement::from(input),
    })
}

pub fn checked_zk_inputs(input: PcsAssistZkInput<'_, DoryScheme>) -> CheckedInputs<'_> {
    CheckedInputs::Zk(ZkInputs {
        opening: ZkOpeningStatement::from(input),
    })
}

pub fn validate_checked_inputs(checked: &CheckedInputs<'_>) -> Result<(), DoryAssistVerifierError> {
    let point_len = checked.point().len();

    if point_len > MAX_DORY_ASSIST_OPENING_POINT_LEN {
        return Err(DoryAssistVerifierError::CheckedInputMismatch {
            reason: format!(
                "opening point length {point_len} exceeds maximum {MAX_DORY_ASSIST_OPENING_POINT_LEN}"
            ),
        });
    }

    let pcs_proof = checked.pcs_proof();
    if point_len != pcs_proof.point_len() {
        return Err(DoryAssistVerifierError::CheckedInputMismatch {
            reason: format!(
                "opening point length {point_len} does not match Dory proof point length {}",
                pcs_proof.point_len()
            ),
        });
    }
    if !pcs_proof.has_canonical_reduce_round_shape() {
        return Err(DoryAssistVerifierError::CheckedInputMismatch {
            reason: format!(
                "Dory proof reduce message counts must both equal sigma={}: first={}, second={}",
                pcs_proof.reduce_round_count(),
                pcs_proof.first_reduce_message_count(),
                pcs_proof.second_reduce_message_count()
            ),
        });
    }
    if !checked
        .setup()
        .supports_reduce_round_count(pcs_proof.reduce_round_count())
    {
        return Err(DoryAssistVerifierError::CheckedInputMismatch {
            reason: format!(
                "Dory verifier setup supports {} reduce rounds with consistent artifacts, but proof requires {}",
                checked.setup().max_reduce_rounds(),
                pcs_proof.reduce_round_count()
            ),
        });
    }
    match checked {
        CheckedInputs::Clear(_) if !pcs_proof.has_transparent_opening_artifacts() => {
            return Err(DoryAssistVerifierError::CheckedInputMismatch {
                reason: "clear Dory-assist input requires a transparent Dory opening proof"
                    .to_string(),
            });
        }
        CheckedInputs::Zk(_) if !pcs_proof.has_zk_opening_artifacts() => {
            return Err(DoryAssistVerifierError::CheckedInputMismatch {
                reason: "ZK Dory-assist input requires Dory ZK, sigma, and scalar-product proof artifacts"
                    .to_string(),
            });
        }
        _ => {}
    }

    Ok(())
}

pub fn validate_proof_dimensions(
    checked: &CheckedInputs<'_>,
    proof: &DoryAssistProof,
) -> Result<(), DoryAssistVerifierError> {
    let dory_reduce = proof.dimensions.dory_reduce;
    let expected_point_len = checked.point().len();
    if dory_reduce.point_len() != expected_point_len {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.dory_reduce.point_len",
            reason: format!(
                "Dory-reduce point length {} must match checked opening point length {expected_point_len}",
                dory_reduce.point_len()
            ),
        });
    }

    let expected_reduce_rounds = checked.pcs_proof().reduce_round_count();
    if dory_reduce.reduce_rounds() != expected_reduce_rounds {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.dory_reduce.reduce_rounds",
            reason: format!(
                "Dory-reduce round count {} must match Dory proof reduce round count {expected_reduce_rounds}",
                dory_reduce.reduce_rounds()
            ),
        });
    }

    let supported = default_dory_assist_dimensions();
    if proof.dimensions.gt != supported.gt {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.gt",
            reason: format!(
                "only canonical GT dimensions {:?} are currently supported, got {:?}",
                supported.gt, proof.dimensions.gt
            ),
        });
    }
    if proof.dimensions.g1 != supported.g1 {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.g1",
            reason: format!(
                "only canonical G1 dimensions {:?} are currently supported, got {:?}",
                supported.g1, proof.dimensions.g1
            ),
        });
    }
    if proof.dimensions.g2 != supported.g2 {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.g2",
            reason: format!(
                "only canonical G2 dimensions {:?} are currently supported, got {:?}",
                supported.g2, proof.dimensions.g2
            ),
        });
    }
    if proof.dimensions.miller_loop != supported.miller_loop {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.miller_loop",
            reason: format!(
                "only canonical Miller-loop dimensions {:?} are currently supported, got {:?}",
                supported.miller_loop, proof.dimensions.miller_loop
            ),
        });
    }
    if proof.dimensions.wiring != supported.wiring {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.wiring",
            reason: format!(
                "only canonical wiring dimensions {:?} are currently supported, got {:?}",
                supported.wiring, proof.dimensions.wiring
            ),
        });
    }

    let expected_packing = composition::prefix_packing_catalog(proof.dimensions)
        .minimal_dimensions()
        .map_err(|error| DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.packing",
            reason: error.to_string(),
        })?;
    if proof.dimensions.packing != expected_packing {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "proof.dimensions.packing",
            reason: format!(
                "packing dimensions {:?} must match catalog-derived minimal dimensions {:?}",
                proof.dimensions.packing, expected_packing
            ),
        });
    }

    Ok(())
}

pub(crate) fn absorb_checked_inputs<T>(
    checked: &CheckedInputs<'_>,
    dory_verifier_scalars: &DoryVerifierTranscriptScalars,
    transcript: &mut T,
) -> DoryAssistInputPublicClaims
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    let dory_verifier_scalar_claims = dory_verifier_transcript_scalar_claims(dory_verifier_scalars);
    transcript.append(&Label(b"DoryAssist"));
    transcript.append(&Label(b"checked_inputs"));
    transcript.append(&Label(checked.mode_name().as_bytes()));
    let mut input_public_claims = match checked {
        CheckedInputs::Clear(inputs) => {
            let mut input_public_claims = absorb_common_opening_inputs(
                inputs.opening.setup,
                inputs.opening.pcs_proof,
                inputs.opening.commitment,
                inputs.opening.point,
                transcript,
            );
            transcript.append(&Label(b"dory_assist_eval"));
            transcript.append(&inputs.opening.eval);
            input_public_claims
                .jolt_evaluation_claims
                .push(inject_fr_to_fq(inputs.opening.eval));
            input_public_claims
                .dory_reduce_initial_e2
                .extend(g2_artifact_coordinates(
                    inputs
                        .opening
                        .setup
                        .artifacts()
                        .g2_0
                        .scalar_mul(&inputs.opening.eval),
                ));
            input_public_claims
        }
        CheckedInputs::Zk(inputs) => {
            let mut input_public_claims = absorb_common_opening_inputs(
                inputs.opening.setup,
                inputs.opening.pcs_proof,
                inputs.opening.commitment,
                inputs.opening.point,
                transcript,
            );
            input_public_claims
                .dory_reduce_initial_e2
                .extend(g2_artifact_coordinates(
                    inputs
                        .opening
                        .pcs_proof
                        .zk_artifacts()
                        .e2
                        .unwrap_or_default(),
                ));
            input_public_claims
        }
    };
    input_public_claims
        .transcript_scalars
        .extend(checked.point().iter().copied().map(inject_fr_to_fq));
    input_public_claims
        .transcript_scalars
        .extend(dory_verifier_scalar_claims);
    input_public_claims
}

fn dory_verifier_transcript_scalar_claims(scalars: &DoryVerifierTranscriptScalars) -> Vec<Fq> {
    let mut claims = Vec::with_capacity(
        8 * scalars.reduce_rounds.len() + 4 + usize::from(scalars.scalar_product_sigma_c.is_some()),
    );
    for round in &scalars.reduce_rounds {
        claims.push(inject_fr_to_fq(round.beta));
        claims.push(inject_fr_to_fq(round.beta_inverse));
        claims.push(inject_fr_to_fq(round.alpha));
        claims.push(inject_fr_to_fq(round.alpha_inverse));
        claims.push(inject_fr_to_fq(round.alpha_beta));
        claims.push(inject_fr_to_fq(round.alpha_inverse_beta_inverse));
        claims.push(inject_fr_to_fq(round.s1_fold_factor));
        claims.push(inject_fr_to_fq(round.s2_fold_factor));
    }
    claims.push(inject_fr_to_fq(scalars.gamma));
    claims.push(inject_fr_to_fq(scalars.gamma_inverse));
    if let Some(sigma_c) = scalars.scalar_product_sigma_c {
        claims.push(inject_fr_to_fq(sigma_c));
    }
    claims.push(inject_fr_to_fq(scalars.d));
    claims.push(inject_fr_to_fq(scalars.d_inverse));
    claims.push(inject_fr_to_fq(scalars.d_squared));
    claims
}

fn dory_verifier_transcript_scalars<T>(
    checked: &CheckedInputs<'_>,
    transcript: &T,
) -> DoryVerifierTranscriptScalars
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    checked
        .pcs_proof()
        .verifier_transcript_scalars(transcript, checked.point())
}

fn validate_dory_verifier_transcript_scalars(
    checked: &CheckedInputs<'_>,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<(), DoryAssistVerifierError> {
    if scalars.has_valid_replay_relations_for_point(checked.point()) {
        Ok(())
    } else {
        Err(DoryAssistVerifierError::TranscriptMismatch {
            reason: "Dory verifier transcript produced a non-invertible challenge or inconsistent derived scalar"
                .to_string(),
        })
    }
}

fn absorb_common_opening_inputs<T>(
    setup: &DoryVerifierSetup,
    pcs_proof: &DoryProof,
    commitment: &DoryCommitment,
    point: &[Fr],
    transcript: &mut T,
) -> DoryAssistInputPublicClaims
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    let mut input_public_claims = DoryAssistInputPublicClaims::default();

    transcript.append(&Label(b"dory_assist_setup"));
    transcript.append(setup);
    input_public_claims.verifier_setup_digest =
        forked_fq_challenge(transcript, VERIFIER_SETUP_DIGEST_LABEL);
    append_dory_verifier_setup_artifacts(&mut input_public_claims.verifier_setup_artifacts, setup);

    transcript.append(&Label(b"dory_assist_pcs_proof"));
    transcript.append(pcs_proof);
    input_public_claims
        .dory_proof_artifacts
        .push(forked_fq_challenge(transcript, DORY_PROOF_DIGEST_LABEL));
    append_dory_proof_artifacts(&mut input_public_claims.dory_proof_artifacts, pcs_proof);

    transcript.append(&Label(b"dory_assist_commitment"));
    transcript.append(commitment);
    input_public_claims
        .jolt_commitments
        .push(forked_fq_challenge(
            transcript,
            JOLT_COMMITMENT_DIGEST_LABEL,
        ));
    input_public_claims
        .jolt_commitments
        .extend(gt_artifact_coefficients(&commitment.0));

    transcript.append(&LabelWithCount(b"dory_assist_point", point.len() as u64));
    for point_coordinate in point {
        transcript.append(point_coordinate);
    }

    input_public_claims
}

fn append_dory_proof_artifacts(artifacts: &mut Vec<Fq>, pcs_proof: &DoryProof) {
    let vmv = pcs_proof.vmv_artifacts();
    artifacts.extend(gt_artifact_coefficients(&vmv.c));
    artifacts.extend(gt_artifact_coefficients(&vmv.d2));
    artifacts.extend(g1_artifact_coordinates(vmv.e1));

    let zk = pcs_proof.zk_artifacts();
    artifacts.extend(match zk.e2 {
        Some(e2) => g2_artifact_coordinates(e2),
        None => identity_g2_artifact_coordinates(),
    });
    artifacts.extend(match zk.y_com {
        Some(y_com) => g1_artifact_coordinates(y_com),
        None => identity_g1_artifact_coordinates(),
    });
    if let Some(scalar_product) = pcs_proof.scalar_product_artifacts() {
        artifacts.extend(gt_artifact_coefficients(&scalar_product.p1));
        artifacts.extend(gt_artifact_coefficients(&scalar_product.p2));
        artifacts.extend(gt_artifact_coefficients(&scalar_product.q));
        artifacts.extend(gt_artifact_coefficients(&scalar_product.r));
        artifacts.extend(g1_artifact_coordinates(scalar_product.e1));
        artifacts.extend(g2_artifact_coordinates(scalar_product.e2));
        artifacts.push(inject_fr_to_fq(scalar_product.r1));
        artifacts.push(inject_fr_to_fq(scalar_product.r2));
        artifacts.push(inject_fr_to_fq(scalar_product.r3));
    } else {
        let identity_gt = Bn254GT::default();
        artifacts.extend(gt_artifact_coefficients(&identity_gt));
        artifacts.extend(gt_artifact_coefficients(&identity_gt));
        artifacts.extend(gt_artifact_coefficients(&identity_gt));
        artifacts.extend(gt_artifact_coefficients(&identity_gt));
        artifacts.extend(identity_g1_artifact_coordinates());
        artifacts.extend(identity_g2_artifact_coordinates());
        artifacts.extend([Fq::default(), Fq::default(), Fq::default()]);
    }

    let layout = DoryProofArtifactLayout::for_proof(pcs_proof);
    artifacts.reserve(layout.expected_len().saturating_sub(artifacts.len()));
    for round in pcs_proof.reduce_round_artifacts() {
        artifacts.extend(gt_artifact_coefficients(&round.first.d1_left));
        artifacts.extend(gt_artifact_coefficients(&round.first.d1_right));
        artifacts.extend(gt_artifact_coefficients(&round.first.d2_left));
        artifacts.extend(gt_artifact_coefficients(&round.first.d2_right));
        artifacts.extend(g1_artifact_coordinates(round.first.e1_beta));
        artifacts.extend(g2_artifact_coordinates(round.first.e2_beta));

        artifacts.extend(gt_artifact_coefficients(&round.second.c_plus));
        artifacts.extend(gt_artifact_coefficients(&round.second.c_minus));
        artifacts.extend(g1_artifact_coordinates(round.second.e1_plus));
        artifacts.extend(g1_artifact_coordinates(round.second.e1_minus));
        artifacts.extend(g2_artifact_coordinates(round.second.e2_plus));
        artifacts.extend(g2_artifact_coordinates(round.second.e2_minus));
    }
    let final_artifacts = pcs_proof.final_artifacts();
    artifacts.extend(g1_artifact_coordinates(final_artifacts.e1));
    artifacts.extend(g2_artifact_coordinates(final_artifacts.e2));
}

fn append_dory_verifier_setup_artifacts(artifacts: &mut Vec<Fq>, setup: &DoryVerifierSetup) {
    let setup_artifacts = setup.artifacts();
    for value in &setup_artifacts.chi {
        artifacts.extend(gt_artifact_coefficients(value));
    }
    for value in &setup_artifacts.delta_1l {
        artifacts.extend(gt_artifact_coefficients(value));
    }
    for value in &setup_artifacts.delta_1r {
        artifacts.extend(gt_artifact_coefficients(value));
    }
    for value in &setup_artifacts.delta_2l {
        artifacts.extend(gt_artifact_coefficients(value));
    }
    for value in &setup_artifacts.delta_2r {
        artifacts.extend(gt_artifact_coefficients(value));
    }
    artifacts.extend(g1_artifact_coordinates(setup_artifacts.g1_0));
    artifacts.extend(g2_artifact_coordinates(setup_artifacts.g2_0));
    artifacts.extend(g1_artifact_coordinates(setup_artifacts.h1));
    artifacts.extend(g2_artifact_coordinates(setup_artifacts.h2));
    artifacts.extend(gt_artifact_coefficients(&setup_artifacts.ht));
}

fn gt_artifact_coefficients(value: &Bn254GT) -> [Fq; GT_ARTIFACT_COEFFS] {
    let mut coefficients = [Fq::default(); GT_ARTIFACT_COEFFS];
    coefficients[..Bn254GT::FQ12_COEFFICIENTS].copy_from_slice(&value.fq12_coefficients());
    coefficients
}

fn g1_artifact_coordinates(value: Bn254G1) -> [Fq; G1_ARTIFACT_COORDS] {
    value.affine_coordinates_with_infinity()
}

fn g2_artifact_coordinates(value: Bn254G2) -> [Fq; G2_ARTIFACT_COORDS] {
    value.affine_coordinates_with_infinity()
}

fn identity_g1_artifact_coordinates() -> [Fq; G1_ARTIFACT_COORDS] {
    [Fq::default(), Fq::default(), Fq::from_u64(1)]
}

fn identity_g2_artifact_coordinates() -> [Fq; G2_ARTIFACT_COORDS] {
    [
        Fq::default(),
        Fq::default(),
        Fq::default(),
        Fq::default(),
        Fq::from_u64(1),
    ]
}

pub(crate) fn inject_fr_to_fq(value: Fr) -> Fq {
    let mut bytes = [0_u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    Fq::from_le_bytes_mod_order(&bytes)
}

pub(crate) fn squeeze_fq<T>(transcript: &mut T) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    inject_fr_to_fq(transcript.challenge_scalar())
}

pub(crate) fn squeeze_fq_challenge<T>(transcript: &mut T, label: &'static [u8]) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label));
    squeeze_fq(transcript)
}

pub(crate) fn forked_fq_challenge<T>(transcript: &T, label: &'static [u8]) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    let mut fork = transcript.clone();
    squeeze_fq_challenge(&mut fork, label)
}

pub(crate) fn squeeze_checked_input_digest<T>(transcript: &mut T) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    squeeze_fq_challenge(transcript, CHECKED_INPUT_DIGEST_LABEL)
}

pub(crate) fn verify_checked_input_public_claims<T>(
    proof: &DoryAssistProof,
    mut expected: DoryAssistInputPublicClaims,
    transcript: &mut T,
) -> Result<(), DoryAssistVerifierError>
where
    T: Transcript<Challenge = Fr>,
{
    expected.checked_input_digest = squeeze_checked_input_digest(transcript);
    let actual = &proof.claims.stage1.public.input;
    if actual != &expected {
        return Err(DoryAssistVerifierError::CheckedInputMismatch {
            reason: checked_input_public_mismatch_reason(actual, &expected),
        });
    }

    Ok(())
}

fn checked_input_public_mismatch_reason(
    actual: &DoryAssistInputPublicClaims,
    expected: &DoryAssistInputPublicClaims,
) -> String {
    if actual.checked_input_digest != expected.checked_input_digest {
        return format!(
            "checked-input digest claim {:?} does not match continued transcript digest {:?}",
            actual.checked_input_digest, expected.checked_input_digest
        );
    }
    if actual.verifier_setup_digest != expected.verifier_setup_digest {
        return format!(
            "verifier setup digest claim {:?} does not match expected {:?}",
            actual.verifier_setup_digest, expected.verifier_setup_digest
        );
    }
    if actual.verifier_setup_artifacts != expected.verifier_setup_artifacts {
        return format!(
            "verifier setup artifact claims {:?} do not match expected {:?}",
            actual.verifier_setup_artifacts, expected.verifier_setup_artifacts
        );
    }
    if actual.dory_proof_artifacts != expected.dory_proof_artifacts {
        return format!(
            "Dory proof artifact claims {:?} do not match expected {:?}",
            actual.dory_proof_artifacts, expected.dory_proof_artifacts
        );
    }
    if actual.jolt_commitments != expected.jolt_commitments {
        return format!(
            "Jolt commitment claims {:?} do not match expected {:?}",
            actual.jolt_commitments, expected.jolt_commitments
        );
    }
    if actual.jolt_evaluation_claims != expected.jolt_evaluation_claims {
        return format!(
            "Jolt evaluation claims {:?} do not match expected {:?}",
            actual.jolt_evaluation_claims, expected.jolt_evaluation_claims
        );
    }
    if actual.dory_reduce_initial_e2 != expected.dory_reduce_initial_e2 {
        return format!(
            "Dory-reduce initial E2 claims {:?} do not match expected {:?}",
            actual.dory_reduce_initial_e2, expected.dory_reduce_initial_e2
        );
    }
    if actual.transcript_scalars != expected.transcript_scalars {
        return format!(
            "transcript scalar claims {:?} do not match expected {:?}",
            actual.transcript_scalars, expected.transcript_scalars
        );
    }

    "checked-input public claims do not match expected values".to_string()
}

pub(crate) fn run_stages<T>(
    checked: &CheckedInputs<'_>,
    proof: &DoryAssistProof,
    transcript: &mut T,
) -> Result<stages::stage3::Stage3Output, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    validate_proof_dimensions(checked, proof)?;

    let stage1 = stages::stage1::verify(
        stages::stage1::Stage1Inputs {
            checked,
            dimensions: proof.dimensions,
            proof: &proof.stages.stage1,
            claims: &proof.claims,
        },
        transcript,
    )?;
    let stage2 = stages::stage2::verify(
        stages::stage2::Stage2Inputs {
            checked,
            dimensions: proof.dimensions,
            proof: &proof.stages.stage2,
            claims: &proof.claims,
            stage1: &stage1,
        },
        transcript,
    )?;
    stages::stage3::verify(
        stages::stage3::Stage3Inputs {
            checked,
            proof: &proof.stages.stage3,
            opening_proof: &proof.opening_proof,
            claims: &proof.claims,
            dense_commitment: &proof.dense_commitment,
            public_outputs: &proof.public_outputs,
            stage1: &stage1,
            stage2: &stage2,
        },
        transcript,
    )
}

pub(crate) fn verify_native_outputs(
    proof: &DoryAssistProof,
) -> Result<(), DoryAssistVerifierError> {
    let expected = proof.public_outputs.pre_final_exponentiation_coefficients();
    let actual = proof.claims.stage1.public.miller_loop.output_gt;

    if let Some((component, (actual, expected))) = actual
        .iter()
        .zip(expected.iter())
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        return Err(DoryAssistVerifierError::PublicOutputMismatch {
            reason: format!(
                "MillerLoopOutputGt({component}) claim {actual:?} does not match pre-final-exponentiation coefficient {expected:?}"
            ),
        });
    }

    Ok(())
}

pub(crate) fn verify_clear_native_outputs(
    input: &ClearOpeningStatement<'_>,
    proof: &DoryAssistProof,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<(), DoryAssistVerifierError> {
    verify_native_outputs(proof)?;

    let final_check = transparent_final_pairing_check(
        input,
        scalars,
        &proof.claims.stage1.public.native_final.inputs,
    )?;
    let final_value = proof
        .public_outputs
        .pre_final_exponentiation
        .final_exponentiation()
        .ok_or_else(|| DoryAssistVerifierError::PublicOutputMismatch {
            reason: "pre-final-exponentiation output did not admit BN254 final exponentiation"
                .to_string(),
        })?;

    if final_value != final_check.rhs {
        return Err(DoryAssistVerifierError::PublicOutputMismatch {
            reason: "final exponentiation of MillerLoopOutputGt did not match Dory final RHS"
                .to_string(),
        });
    }

    Ok(())
}

pub(crate) fn verify_zk_native_outputs(
    input: &ZkOpeningStatement<'_>,
    proof: &DoryAssistProof,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<Bn254G1, DoryAssistVerifierError> {
    verify_native_outputs(proof)?;

    let final_check = zk_final_pairing_check(
        input,
        scalars,
        &proof.claims.stage1.public.native_final.inputs,
    )?;
    let final_value = proof
        .public_outputs
        .pre_final_exponentiation
        .final_exponentiation()
        .ok_or_else(|| DoryAssistVerifierError::PublicOutputMismatch {
            reason: "ZK pre-final-exponentiation output did not admit BN254 final exponentiation"
                .to_string(),
        })?;

    if final_value != final_check.rhs {
        return Err(DoryAssistVerifierError::PublicOutputMismatch {
            reason: "final exponentiation of ZK MillerLoopOutputGt did not match Dory scalar-product RHS"
                .to_string(),
        });
    }

    input.pcs_proof.zk_artifacts().y_com.ok_or_else(|| {
        DoryAssistVerifierError::CheckedInputMismatch {
            reason: "ZK Dory-assist input is missing y_com hiding commitment".to_string(),
        }
    })
}

impl<'a> From<PcsAssistClearInput<'a, DoryScheme>> for ClearOpeningStatement<'a> {
    fn from(input: PcsAssistClearInput<'a, DoryScheme>) -> Self {
        Self {
            setup: input.setup,
            pcs_proof: input.pcs_proof,
            commitment: input.commitment,
            point: input.point,
            eval: input.eval,
        }
    }
}

impl<'a> From<PcsAssistZkInput<'a, DoryScheme>> for ZkOpeningStatement<'a> {
    fn from(input: PcsAssistZkInput<'a, DoryScheme>) -> Self {
        Self {
            setup: input.setup,
            pcs_proof: input.pcs_proof,
            commitment: input.commitment,
            point: input.point,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tests may panic on invalid local setup"
)]
mod tests {
    use super::*;
    use crate::{
        artifacts::{
            DoryProofArtifactLayout, DORY_VMV_C_START, DORY_VMV_E1_START, G1_ARTIFACT_COORDS,
            GT_ARTIFACT_COEFFS,
        },
        derive_hyrax_prover_setup,
        native_final::{
            transparent_native_final_input_claims, transparent_replayed_final_pairing_check,
            zk_native_final_input_claims, zk_replayed_final_pairing_check,
        },
        proof::{
            DoryAssistOpeningClaim, DoryAssistStage1PublicClaims, NATIVE_FINAL_D1_START,
            NATIVE_FINAL_GT_C_START, NATIVE_FINAL_INPUT_LEN,
        },
        stages::stage3::Stage3Output,
        DoryAssistHyrax, DoryAssistStage,
    };
    use jolt_claims::protocols::dory_assist::{
        formulas::{
            composition, dory_reduce,
            protocol::{protocol_claims, CANONICAL_RELATION_ORDER},
            setup_artifacts, transcript_scalars,
        },
        DoryAssistChallengeId, DoryAssistCopyConstraint, DoryAssistDimensions, DoryAssistOpeningId,
        DoryAssistPublicId, DoryAssistRelationId, DoryAssistSumcheckSpec, DoryAssistValueRef,
        DoryAssistVirtualPolynomial, DoryReduceDimensions, DoryReducePolynomial, G1Dimensions,
        G2Dimensions, GtDimensions, MillerLoopDimensions, PrefixPackingDimensions,
        WiringDimensions,
    };
    use jolt_crypto::JoltGroup;
    use jolt_field::{Fq, FromPrimitiveInt, Invertible};
    use jolt_hyrax::HyraxDimensions;
    use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
    use jolt_poly::{CompressedPoly, EqPolynomial, Polynomial};
    use jolt_sumcheck::SUMCHECK_ROUND_TRANSCRIPT_LABEL;
    use jolt_transcript::{Blake2bTranscript, U64Word};

    #[test]
    fn dory_assist_implements_pcs_assist_for_dory() {
        fn assert_impl<T: PcsProofAssist<DoryScheme>>() {}
        assert_impl::<DoryAssist>();
    }

    #[test]
    fn selected_config_is_deterministic() {
        let left = DoryAssist::selected_config();
        let right = DoryAssist::selected_config();

        assert_eq!(left, right);
    }

    fn absorb_checked_inputs_for_test<T>(
        checked: &CheckedInputs<'_>,
        transcript: &mut T,
    ) -> DoryAssistInputPublicClaims
    where
        T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
    {
        let scalars = dory_verifier_transcript_scalars(checked, transcript);
        assert!(scalars.has_valid_replay_relations_for_point(checked.point()));
        absorb_checked_inputs(checked, &scalars, transcript)
    }

    #[test]
    fn checked_clear_inputs_preserve_opening_statement() -> Result<(), &'static str> {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());

        let CheckedInputs::Clear(inputs) = checked else {
            return Err("expected clear checked inputs");
        };
        assert_clear_opening_matches(&inputs.opening, &fixture);
        Ok(())
    }

    #[test]
    fn checked_zk_inputs_preserve_opening_statement() -> Result<(), &'static str> {
        let fixture = dory_opening_fixture();
        let checked = checked_zk_inputs(fixture.zk_input());

        let CheckedInputs::Zk(inputs) = checked else {
            return Err("expected zk checked inputs");
        };
        assert_zk_opening_matches(&inputs.opening, &fixture);
        Ok(())
    }

    #[test]
    fn checked_inputs_expose_jolt_like_mode_flag() {
        let fixture = dory_opening_fixture();
        let clear = checked_clear_inputs(fixture.clear_input());
        let zk = checked_zk_inputs(fixture.zk_input());

        assert!(!clear.zk());
        assert_eq!(clear.mode_name(), "clear");
        assert!(zk.zk());
        assert_eq!(zk.mode_name(), "zk");
    }

    #[test]
    fn fr_to_fq_injection_preserves_small_scalars() {
        let scalar = Fr::from_u64(42);

        assert_eq!(inject_fr_to_fq(scalar), Fq::from_u64(42));
        assert_eq!(inject_fr_to_fq(scalar), inject_fr_to_fq(scalar));
    }

    #[test]
    fn checked_input_validation_rejects_excessively_long_points() {
        let fixture = dory_opening_fixture();
        let point = vec![Fr::from_u64(1); MAX_DORY_ASSIST_OPENING_POINT_LEN + 1];
        let checked = checked_clear_inputs(PcsAssistClearInput {
            setup: &fixture.verifier_setup,
            pcs_proof: &fixture.proof,
            commitment: &fixture.commitment,
            point: &point,
            eval: fixture.eval,
        });

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_point_len_mismatch() {
        let fixture = dory_opening_fixture();
        let point = vec![Fr::from_u64(1)];
        let checked = checked_clear_inputs(PcsAssistClearInput {
            setup: &fixture.verifier_setup,
            pcs_proof: &fixture.proof,
            commitment: &fixture.commitment,
            point: &point,
            eval: fixture.eval,
        });

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_reduce_round_shape_mismatch() {
        let mut fixture = dory_opening_fixture();
        let _ = fixture.proof.0.first_messages.pop();
        let checked = checked_clear_inputs(fixture.clear_input());

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_setup_with_insufficient_reduce_round_capacity() {
        let mut fixture = dory_opening_fixture_with_num_vars(4);
        assert_eq!(fixture.proof.reduce_round_count(), 2);
        let (_, smaller_setup) = DoryScheme::setup(2);
        assert_eq!(smaller_setup.max_reduce_rounds(), 1);
        fixture.verifier_setup = smaller_setup;
        let checked = checked_clear_inputs(fixture.clear_input());

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_inconsistent_setup_artifact_lengths() {
        let mut fixture = dory_opening_fixture();
        let _ = fixture.verifier_setup.0.delta_1l.pop();
        let checked = checked_clear_inputs(fixture.clear_input());

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_clear_with_zk_dory_proof() {
        let fixture = dory_zk_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_zk_without_zk_dory_artifacts() {
        let fixture = dory_opening_fixture();
        let checked = checked_zk_inputs(fixture.zk_input());

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_validation_rejects_zk_missing_scalar_product_artifacts() {
        let mut fixture = dory_zk_opening_fixture();
        fixture.proof.0.scalar_product_proof = None;
        let checked = checked_zk_inputs(fixture.zk_input());

        let result = validate_checked_inputs(&checked);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_dory_reduce_point_len_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.dimensions.dory_reduce =
            DoryReduceDimensions::new(fixture.point.len() + 1, fixture.proof.reduce_round_count());

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.dory_reduce.point_len",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_dory_reduce_round_count_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.dimensions.dory_reduce =
            DoryReduceDimensions::new(fixture.point.len(), fixture.proof.reduce_round_count() + 1);

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.dory_reduce.reduce_rounds",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_accepts_matching_multiround_dory_reduce() {
        let fixture = dory_opening_fixture_with_num_vars(4);
        assert_eq!(fixture.proof.point_len(), 4);
        assert_eq!(fixture.proof.reduce_round_count(), 2);
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof_for_checked(&checked);

        assert_eq!(
            proof.dimensions.dory_reduce,
            DoryReduceDimensions::new(
                fixture.proof.point_len(),
                fixture.proof.reduce_round_count(),
            )
        );

        let result = validate_proof_dimensions(&checked, &proof);

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn verify_clear_accepts_well_shaped_multiround_transparent_proof() {
        let fixture = dory_opening_fixture_with_num_vars(4);
        assert_eq!(fixture.proof.point_len(), 4);
        assert_eq!(fixture.proof.reduce_round_count(), 2);
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof_for_checked(&checked);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = DoryAssist::verify_clear(
            &DoryAssist::selected_config(),
            fixture.clear_input(),
            &proof,
            &mut transcript,
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn run_stages_rejects_multiround_dory_reduce_public_fold_mismatch() {
        let fixture = dory_opening_fixture_with_num_vars(4);
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        let opening = dory_reduce::s1_fold_factor_opening();
        let tampered = get_dory_reduce_opening(&proof, opening) + Fq::from_u64(1);
        set_dory_reduce_opening(&mut proof, opening, tampered);
        rebalance_dory_reduce_scalar_fold_relation(&mut proof, &checked);
        populate_valid_hyrax_opening(&mut proof);

        let result = try_run_stages_after_checked_preamble(&checked, &proof);

        assert!(
            matches!(
                result,
                Err(DoryAssistVerifierError::StageOutputMismatch {
                    stage: DoryAssistStage::Stage2,
                    ..
                })
            ),
            "{result:?}"
        );
    }

    #[test]
    fn run_stages_rejects_multiround_dory_reduce_boundary_output_mismatch() {
        let fixture = dory_opening_fixture_with_num_vars(4);
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        set_dory_reduce_opening(
            &mut proof,
            dory_reduce_opening(
                DoryAssistRelationId::DoryReduceBoundary,
                DoryReducePolynomial::S1Accumulator,
            ),
            Fq::from_u64(9),
        );
        populate_valid_hyrax_opening(&mut proof);

        let result = try_run_stages_after_checked_preamble(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_multiround_dory_reduce_state_chain_output_mismatch() {
        let fixture = dory_opening_fixture_with_num_vars(4);
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        set_dory_reduce_opening(
            &mut proof,
            dory_reduce_opening(
                DoryAssistRelationId::DoryReduceStateChain,
                DoryReducePolynomial::S1Accumulator,
            ),
            Fq::from_u64(7),
        );
        populate_valid_hyrax_opening(&mut proof);

        let result = try_run_stages_after_checked_preamble(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_multiround_dory_reduce_point_mismatch() {
        let fixture = dory_opening_fixture_with_num_vars(4);
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.dimensions.dory_reduce = DoryReduceDimensions::new(
            fixture.proof.point_len() + 1,
            fixture.proof.reduce_round_count(),
        );

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.dory_reduce.point_len",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_noncanonical_gt_dimensions() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        let gt = proof.dimensions.gt;
        proof.dimensions.gt = GtDimensions::new(
            gt.exp_step_vars() + 1,
            gt.exp_instance_vars(),
            gt.mul_instance_vars(),
        );

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.gt",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_noncanonical_g1_dimensions() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        let g1 = proof.dimensions.g1;
        proof.dimensions.g1 = G1Dimensions::new(
            g1.scalar_mul_step_vars() + 1,
            g1.scalar_mul_instance_vars(),
            g1.add_instance_vars(),
        );

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.g1",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_noncanonical_g2_dimensions() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        let g2 = proof.dimensions.g2;
        proof.dimensions.g2 = G2Dimensions::new(
            g2.scalar_mul_step_vars() + 1,
            g2.scalar_mul_instance_vars(),
            g2.add_instance_vars(),
        );

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.g2",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_noncanonical_miller_loop_dimensions() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        let miller_loop = proof.dimensions.miller_loop;
        proof.dimensions.miller_loop = MillerLoopDimensions::new(
            miller_loop.line_event_vars() + 1,
            miller_loop.pair_vars(),
            miller_loop.accumulator_op_vars(),
        );

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.miller_loop",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_noncanonical_wiring_dimensions() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.dimensions.wiring = WiringDimensions::new(proof.dimensions.wiring.log_edges() + 1);

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.wiring",
                ..
            })
        ));
    }

    #[test]
    fn proof_dimension_validation_rejects_nonminimal_packing_dimensions() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        let packing = proof.dimensions.packing;
        proof.dimensions.packing = PrefixPackingDimensions::new(
            packing.packed_vars() + 1,
            packing.max_poly_vars(),
            packing.num_claims(),
        )
        .expect("valid non-minimal packing dimensions");

        let result = validate_proof_dimensions(&checked, &proof);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "proof.dimensions.packing",
                ..
            })
        ));
    }

    #[test]
    fn checked_input_preamble_is_deterministic_for_clear() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut left = Blake2bTranscript::new(b"dory-assist-test");
        let mut right = Blake2bTranscript::new(b"dory-assist-test");

        let _ = absorb_checked_inputs_for_test(&checked, &mut left);
        let _ = absorb_checked_inputs_for_test(&checked, &mut right);

        assert_eq!(left.state(), right.state());
    }

    #[test]
    fn checked_input_preamble_changes_when_clear_eval_changes() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut changed = fixture.clear_input();
        changed.eval += Fr::from_u64(1);
        let changed = checked_clear_inputs(changed);
        let mut left = Blake2bTranscript::new(b"dory-assist-test");
        let mut right = Blake2bTranscript::new(b"dory-assist-test");

        let _ = absorb_checked_inputs_for_test(&checked, &mut left);
        let _ = absorb_checked_inputs_for_test(&changed, &mut right);

        assert_ne!(left.state(), right.state());
    }

    #[test]
    fn checked_input_challenge_changes_when_setup_changes() {
        let fixture = dory_opening_fixture();
        let (_, changed_setup) = DoryScheme::setup(3);
        let checked = checked_clear_inputs(fixture.clear_input());
        let changed = checked_clear_inputs(PcsAssistClearInput {
            setup: &changed_setup,
            pcs_proof: &fixture.proof,
            commitment: &fixture.commitment,
            point: &fixture.point,
            eval: fixture.eval,
        });

        assert_ne!(
            checked_input_test_challenge(&checked),
            checked_input_test_challenge(&changed)
        );
    }

    #[test]
    fn checked_input_challenge_changes_when_pcs_proof_changes() {
        let fixture = dory_opening_fixture();
        let changed_fixture = dory_opening_fixture_with_shift(9);
        let checked = checked_clear_inputs(fixture.clear_input());
        let changed = checked_clear_inputs(PcsAssistClearInput {
            setup: &fixture.verifier_setup,
            pcs_proof: &changed_fixture.proof,
            commitment: &fixture.commitment,
            point: &fixture.point,
            eval: fixture.eval,
        });

        assert_ne!(
            checked_input_test_challenge(&checked),
            checked_input_test_challenge(&changed)
        );
    }

    #[test]
    fn checked_input_digest_is_deterministic_for_clear() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());

        assert_eq!(
            checked_input_public_claims_for_test(&checked).checked_input_digest,
            checked_input_public_claims_for_test(&checked).checked_input_digest
        );
    }

    #[test]
    fn checked_input_digest_rejects_mismatched_claim() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.claims.stage1.public.input.checked_input_digest += Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let input_public_claims = absorb_checked_inputs_for_test(&checked, &mut transcript);

        let result =
            verify_checked_input_public_claims(&proof, input_public_claims, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_public_claims_reject_mismatched_dory_reduce_initial_e2() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.claims.stage1.public.input.dory_reduce_initial_e2[0] += Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let input_public_claims = absorb_checked_inputs_for_test(&checked, &mut transcript);

        let result =
            verify_checked_input_public_claims(&proof, input_public_claims, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::CheckedInputMismatch { .. })
        ));
    }

    #[test]
    fn checked_input_public_claims_resolve_public_ids() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let input_claims = checked_input_public_claims_for_test(&checked);

        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::VerifierSetupDigest),
            Some(input_claims.verifier_setup_digest)
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::VerifierSetupArtifact(0)),
            input_claims.verifier_setup_artifacts.first().copied()
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::DoryProofArtifact(0)),
            input_claims.dory_proof_artifacts.first().copied()
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::JoltCommitment(0)),
            input_claims.jolt_commitments.first().copied()
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::JoltCommitment(1)),
            input_claims.jolt_commitments.get(1).copied()
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::JoltEvaluationClaim(0)),
            Some(inject_fr_to_fq(fixture.eval))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::DoryReduceInitialE2(0)),
            input_claims.dory_reduce_initial_e2.first().copied()
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(1)),
            Some(inject_fr_to_fq(fixture.point[1]))
        );
        let dory_transcript = Blake2bTranscript::<Fr>::new(b"dory-assist-test");
        let dory_scalars = fixture
            .proof
            .verifier_transcript_scalars(&dory_transcript, &fixture.point);
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_reduce_beta(fixture.point.len(), 0),
            )),
            Some(inject_fr_to_fq(dory_scalars.reduce_rounds[0].beta))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_reduce_alpha(fixture.point.len(), 0),
            )),
            Some(inject_fr_to_fq(dory_scalars.reduce_rounds[0].alpha))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_reduce_alpha_beta(fixture.point.len(), 0),
            )),
            Some(inject_fr_to_fq(dory_scalars.reduce_rounds[0].alpha_beta))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_reduce_s1_fold_factor(fixture.point.len(), 0),
            )),
            Some(inject_fr_to_fq(
                dory_scalars.reduce_rounds[0].s1_fold_factor
            ))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_gamma(fixture.point.len(), 1),
            )),
            Some(inject_fr_to_fq(dory_scalars.gamma))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_gamma_inverse(fixture.point.len(), 1),
            )),
            Some(inject_fr_to_fq(dory_scalars.gamma_inverse))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_final_d(fixture.point.len(), 1, false),
            )),
            Some(inject_fr_to_fq(dory_scalars.d))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_final_d_squared(fixture.point.len(), 1, false),
            )),
            Some(inject_fr_to_fq(dory_scalars.d_squared))
        );
        assert_eq!(
            input_claims.transcript_scalars.len(),
            transcript_scalars::transcript_scalar_count(fixture.point.len(), 1, false)
        );

        let vmv = fixture.proof.vmv_artifacts();
        let layout = DoryProofArtifactLayout::for_proof(&fixture.proof);
        let setup_artifacts = fixture.verifier_setup.artifacts();
        assert_eq!(
            input_claims.verifier_setup_artifacts.len(),
            setup_artifacts::dory_setup_artifact_count(fixture.verifier_setup.max_reduce_rounds())
        );
        assert_eq!(
            &input_claims.verifier_setup_artifacts[setup_artifacts::dory_setup_chi_start(0)
                ..setup_artifacts::dory_setup_chi_start(0) + GT_ARTIFACT_COEFFS],
            gt_artifact_coefficients(&setup_artifacts.chi[0]).as_slice()
        );
        assert_eq!(
            &input_claims.verifier_setup_artifacts[setup_artifacts::dory_setup_delta_1l_start(
                fixture.verifier_setup.max_reduce_rounds(),
                0,
            )
                ..setup_artifacts::dory_setup_delta_1l_start(
                    fixture.verifier_setup.max_reduce_rounds(),
                    0,
                ) + GT_ARTIFACT_COEFFS],
            gt_artifact_coefficients(&setup_artifacts.delta_1l[0]).as_slice()
        );
        assert_eq!(
            &input_claims.verifier_setup_artifacts[setup_artifacts::dory_setup_g1_0_start(
                fixture.verifier_setup.max_reduce_rounds()
            )
                ..setup_artifacts::dory_setup_g1_0_start(
                    fixture.verifier_setup.max_reduce_rounds()
                ) + G1_ARTIFACT_COORDS],
            g1_artifact_coordinates(setup_artifacts.g1_0).as_slice()
        );
        assert_eq!(
            input_claims.dory_proof_artifacts.len(),
            layout.expected_len()
        );
        assert_eq!(input_claims.jolt_commitments.len(), 17);
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.vmv_c()],
            gt_artifact_coefficients(&vmv.c).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.vmv_d2()],
            gt_artifact_coefficients(&vmv.d2).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.vmv_e1()],
            g1_artifact_coordinates(vmv.e1).as_slice()
        );
        let round = fixture
            .proof
            .reduce_round_artifacts()
            .into_iter()
            .next()
            .expect("fixture has one Dory reduce round");
        let round_layout = layout.reduce_round(0);
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.first_d1_left()],
            gt_artifact_coefficients(&round.first.d1_left).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.first_d1_right()],
            gt_artifact_coefficients(&round.first.d1_right).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.first_d2_left()],
            gt_artifact_coefficients(&round.first.d2_left).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.first_d2_right()],
            gt_artifact_coefficients(&round.first.d2_right).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.first_e1_beta()],
            g1_artifact_coordinates(round.first.e1_beta).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.first_e2_beta()],
            g2_artifact_coordinates(round.first.e2_beta).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.second_c_plus()],
            gt_artifact_coefficients(&round.second.c_plus).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.second_c_minus()],
            gt_artifact_coefficients(&round.second.c_minus).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.second_e1_plus()],
            g1_artifact_coordinates(round.second.e1_plus).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.second_e1_minus()],
            g1_artifact_coordinates(round.second.e1_minus).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.second_e2_plus()],
            g2_artifact_coordinates(round.second.e2_plus).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[round_layout.second_e2_minus()],
            g2_artifact_coordinates(round.second.e2_minus).as_slice()
        );
        let final_artifacts = fixture.proof.final_artifacts();
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.final_e1()],
            g1_artifact_coordinates(final_artifacts.e1).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.final_e2()],
            g2_artifact_coordinates(final_artifacts.e2).as_slice()
        );
        assert_eq!(
            &input_claims.jolt_commitments[1..17],
            gt_artifact_coefficients(&fixture.commitment.0).as_slice()
        );
    }

    #[test]
    fn checked_input_public_claims_resolve_zk_artifacts_and_sigma_c() {
        let fixture = dory_zk_opening_fixture();
        let checked = checked_zk_inputs(fixture.zk_input());
        let input_claims = checked_input_public_claims_for_test(&checked);
        let layout = DoryProofArtifactLayout::for_proof(&fixture.proof);
        let zk_artifacts = fixture.proof.zk_artifacts();
        let scalar_product = fixture
            .proof
            .scalar_product_artifacts()
            .expect("ZK fixture carries scalar-product artifacts");
        let dory_transcript = Blake2bTranscript::<Fr>::new(b"dory-assist-test");
        let dory_scalars = fixture
            .proof
            .verifier_transcript_scalars(&dory_transcript, &fixture.point);
        let e2 = zk_artifacts.e2.expect("ZK fixture carries E2 artifact");
        let y_com = zk_artifacts
            .y_com
            .expect("ZK fixture carries y_com artifact");
        let sigma_c = dory_scalars
            .scalar_product_sigma_c
            .expect("ZK fixture has scalar-product sigma_c");

        assert!(input_claims.jolt_evaluation_claims.is_empty());
        assert_eq!(
            input_claims.dory_reduce_initial_e2,
            g2_artifact_coordinates(e2).to_vec()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.zk_e2()],
            g2_artifact_coordinates(e2).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.zk_y_com()],
            g1_artifact_coordinates(y_com).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.scalar_product_p1()],
            gt_artifact_coefficients(&scalar_product.p1).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.scalar_product_p2()],
            gt_artifact_coefficients(&scalar_product.p2).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.scalar_product_q()],
            gt_artifact_coefficients(&scalar_product.q).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.scalar_product_r()],
            gt_artifact_coefficients(&scalar_product.r).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.scalar_product_e1()],
            g1_artifact_coordinates(scalar_product.e1).as_slice()
        );
        assert_eq!(
            &input_claims.dory_proof_artifacts[layout.scalar_product_e2()],
            g2_artifact_coordinates(scalar_product.e2).as_slice()
        );
        assert_eq!(
            input_claims.dory_proof_artifacts[layout.scalar_product_r1()],
            inject_fr_to_fq(scalar_product.r1)
        );
        assert_eq!(
            input_claims.dory_proof_artifacts[layout.scalar_product_r2()],
            inject_fr_to_fq(scalar_product.r2)
        );
        assert_eq!(
            input_claims.dory_proof_artifacts[layout.scalar_product_r3()],
            inject_fr_to_fq(scalar_product.r3)
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_scalar_product_sigma_c(
                    fixture.point.len(),
                    fixture.proof.reduce_round_count(),
                ),
            )),
            Some(inject_fr_to_fq(sigma_c))
        );
        assert_eq!(
            input_claims.claim(&DoryAssistPublicId::TranscriptScalar(
                transcript_scalars::dory_final_d(
                    fixture.point.len(),
                    fixture.proof.reduce_round_count(),
                    true,
                ),
            )),
            Some(inject_fr_to_fq(dory_scalars.d))
        );
        assert_eq!(
            input_claims.transcript_scalars.len(),
            transcript_scalars::transcript_scalar_count(
                fixture.point.len(),
                fixture.proof.reduce_round_count(),
                true,
            )
        );
    }

    #[test]
    fn stage1_public_claims_resolve_dory_reduce_shift_kernel() {
        let public_claims = DoryAssistStage1PublicClaims {
            dory_reduce_shift_eq_kernel: Fq::from_u64(17),
            ..Default::default()
        };

        assert_eq!(
            public_claims.claim(&DoryAssistPublicId::DoryReduceShiftEqKernel),
            Some(Fq::from_u64(17))
        );
    }

    #[test]
    fn stage1_public_claims_resolve_native_final_inputs() {
        let mut public_claims = DoryAssistStage1PublicClaims::default();
        public_claims.native_final.bind(
            (0..NATIVE_FINAL_INPUT_LEN)
                .map(|index| Fq::from_u64(u64::try_from(index + 1).expect("index fits")))
                .collect(),
        );

        assert_eq!(
            public_claims.claim(&DoryAssistPublicId::NativeFinalCheckInput(
                NATIVE_FINAL_D1_START
            )),
            Some(Fq::from_u64(
                u64::try_from(NATIVE_FINAL_D1_START + 1).expect("index fits")
            ))
        );
    }

    #[test]
    fn checked_input_preamble_changes_when_zk_point_changes() {
        let fixture = dory_opening_fixture();
        let checked = checked_zk_inputs(fixture.zk_input());
        let mut point = fixture.point.clone();
        point[0] += Fr::from_u64(1);
        let changed = checked_zk_inputs(PcsAssistZkInput {
            setup: &fixture.verifier_setup,
            pcs_proof: &fixture.proof,
            commitment: &fixture.commitment,
            point: &point,
        });
        let mut left = Blake2bTranscript::new(b"dory-assist-test");
        let mut right = Blake2bTranscript::new(b"dory-assist-test");

        let _ = absorb_checked_inputs_for_test(&checked, &mut left);
        let _ = absorb_checked_inputs_for_test(&changed, &mut right);

        assert_ne!(left.state(), right.state());
    }

    #[test]
    fn verify_clear_accepts_well_shaped_transparent_proof() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof_for_checked(&checked);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = DoryAssist::verify_clear(
            &DoryAssist::selected_config(),
            fixture.clear_input(),
            &proof,
            &mut transcript,
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn verify_zk_accepts_well_shaped_zk_proof() {
        let fixture = dory_zk_opening_fixture();
        let checked = checked_zk_inputs(fixture.zk_input());
        let proof = well_shaped_assist_proof_for_checked(&checked);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = DoryAssist::verify_zk(
            &DoryAssist::selected_config(),
            fixture.zk_input(),
            &proof,
            &mut transcript,
        );

        assert_eq!(
            result,
            Ok(fixture
                .proof
                .zk_artifacts()
                .y_com
                .expect("ZK fixture has y_com"))
        );
    }

    #[test]
    fn dory_verifier_transcript_scalars_reject_invalid_replay_relations() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let transcript = Blake2bTranscript::new(b"dory-assist-test");
        let mut scalars = dory_verifier_transcript_scalars(&checked, &transcript);
        assert!(validate_dory_verifier_transcript_scalars(&checked, &scalars).is_ok());

        scalars.reduce_rounds[0].alpha_inverse = Fr::from_u64(0);

        assert!(matches!(
            validate_dory_verifier_transcript_scalars(&checked, &scalars),
            Err(DoryAssistVerifierError::TranscriptMismatch { .. })
        ));

        let mut scalars = dory_verifier_transcript_scalars(&checked, &transcript);
        scalars.reduce_rounds[0].s2_fold_factor += Fr::from_u64(1);

        assert!(matches!(
            validate_dory_verifier_transcript_scalars(&checked, &scalars),
            Err(DoryAssistVerifierError::TranscriptMismatch { .. })
        ));
    }

    #[test]
    fn native_outputs_accept_bound_pre_final_exponentiation() {
        let mut proof = DoryAssistProof::default();
        proof
            .claims
            .stage1
            .public
            .miller_loop
            .bind_pre_final_exponentiation(&proof.public_outputs);

        assert_eq!(verify_native_outputs(&proof), Ok(()));
    }

    #[test]
    fn native_outputs_reject_mismatched_pre_final_exponentiation() {
        let proof = DoryAssistProof::default();

        assert!(matches!(
            verify_native_outputs(&proof),
            Err(DoryAssistVerifierError::PublicOutputMismatch { .. })
        ));
    }

    #[test]
    fn clear_native_outputs_reject_final_exponentiation_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof.public_outputs.pre_final_exponentiation = Default::default();
        proof
            .claims
            .stage1
            .public
            .miller_loop
            .bind_pre_final_exponentiation(&proof.public_outputs);
        let CheckedInputs::Clear(inputs) = checked else {
            panic!("fixture is clear")
        };
        let transcript = Blake2bTranscript::new(b"dory-assist-test");
        let scalars = dory_verifier_transcript_scalars(&checked, &transcript);

        assert!(matches!(
            verify_clear_native_outputs(&inputs.opening, &proof, &scalars),
            Err(DoryAssistVerifierError::PublicOutputMismatch { .. })
        ));
    }

    #[test]
    fn clear_native_outputs_reject_mismatched_native_final_input_claim() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        tamper_native_final_c_acc(&mut proof);
        let CheckedInputs::Clear(inputs) = checked else {
            panic!("fixture is clear")
        };
        let transcript = Blake2bTranscript::new(b"dory-assist-test");
        let scalars = dory_verifier_transcript_scalars(&checked, &transcript);

        assert!(matches!(
            verify_clear_native_outputs(&inputs.opening, &proof, &scalars),
            Err(DoryAssistVerifierError::PublicOutputMismatch { .. })
        ));
    }

    #[test]
    fn zk_native_outputs_reject_mismatched_native_final_input_claim() {
        let fixture = dory_zk_opening_fixture();
        let checked = checked_zk_inputs(fixture.zk_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        tamper_native_final_c_acc(&mut proof);
        let CheckedInputs::Zk(inputs) = checked else {
            panic!("fixture is ZK")
        };
        let transcript = Blake2bTranscript::new(b"dory-assist-test");
        let scalars = dory_verifier_transcript_scalars(&checked, &transcript);

        assert!(matches!(
            verify_zk_native_outputs(&inputs.opening, &proof, &scalars),
            Err(DoryAssistVerifierError::PublicOutputMismatch { .. })
        ));
    }

    #[test]
    fn native_final_input_claims_reject_wrong_vector_length() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof_for_checked(&checked);
        proof
            .claims
            .stage1
            .public
            .native_final
            .inputs
            .truncate(NATIVE_FINAL_INPUT_LEN - 1);
        let CheckedInputs::Clear(inputs) = checked else {
            panic!("fixture is clear")
        };
        let transcript = Blake2bTranscript::new(b"dory-assist-test");
        let scalars = dory_verifier_transcript_scalars(&checked, &transcript);

        assert!(matches!(
            verify_clear_native_outputs(&inputs.opening, &proof, &scalars),
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "claims.stage1.public.native_final.inputs",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_returns_stage3_output() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert_eq!(
            result.ok().map(|output| output.packed_eval),
            Some(proof.stages.stage3.packed_eval)
        );
    }

    #[test]
    fn stage1_output_records_verified_relation_claims() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let stage1 = stages::stage1::verify(
            stages::stage1::Stage1Inputs {
                checked: &checked,
                dimensions: proof.dimensions,
                proof: &proof.stages.stage1,
                claims: &proof.claims,
            },
            &mut transcript,
        )
        .expect("stage 1 verifies");

        assert_eq!(
            stage1.relation_outputs.len(),
            proof.stages.stage1.relations.len()
        );
        assert_eq!(stage1.relation_outputs[0].input_claim, Fq::default());
        assert_eq!(
            stage1.relation_outputs[0].sumcheck_final_claim,
            stage1.relation_outputs[0].expected_output_claim
        );
        assert_eq!(stage1.relation_outputs[0].opening_claims.len(), 5);
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::G1ScalarMultiplication));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::G1Addition));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::G2ScalarMultiplication));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::G2Addition));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::MillerLoopLineStep));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::MillerLoopLineEvaluation));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::MillerLoopPairProduct));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::MillerLoopAccumulator));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::MillerLoopBoundary));
        assert!(stage1
            .relation_outputs
            .iter()
            .any(|relation| relation.id == DoryAssistRelationId::DoryReduceScalarFold));
    }

    #[test]
    fn stage2_output_records_verified_copy_constraints() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let stage1 = stages::stage1::verify(
            stages::stage1::Stage1Inputs {
                checked: &checked,
                dimensions: proof.dimensions,
                proof: &proof.stages.stage1,
                claims: &proof.claims,
            },
            &mut transcript,
        )
        .expect("stage 1 verifies");

        let stage2 = stages::stage2::verify(
            stages::stage2::Stage2Inputs {
                checked: &checked,
                dimensions: proof.dimensions,
                proof: &proof.stages.stage2,
                claims: &proof.claims,
                stage1: &stage1,
            },
            &mut transcript,
        )
        .expect("stage 2 verifies");

        assert_eq!(
            stage2.relation_count as usize,
            proof.stages.stage2.copy_constraints.len()
        );
        assert_eq!(
            stage2.copy_constraints[0].constraint,
            proof.stages.stage2.copy_constraints[0]
        );
        assert_eq!(
            stage2.copy_constraints[0].source_value,
            stage2.copy_constraints[0].target_value
        );
        assert!(!stage2.dory_reduce_public_folds.is_empty());
        assert_eq!(
            stage2.dory_reduce_public_folds[0].expected_value,
            stage2.dory_reduce_public_folds[0].target_value
        );
    }

    #[test]
    fn run_stages_challenge_is_deterministic() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof();
        let mut left_transcript = Blake2bTranscript::new(b"dory-assist-test");
        let mut right_transcript = Blake2bTranscript::new(b"dory-assist-test");

        let left = run_stages(&checked, &proof, &mut left_transcript).expect("left stages verify");
        let right =
            run_stages(&checked, &proof, &mut right_transcript).expect("right stages verify");

        assert_eq!(left.challenge, right.challenge);
    }

    #[test]
    fn run_stages_rejects_stage1_relation_catalog_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.stages.stage1.relations[0].sumcheck.degree += 1;
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageClaimMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_sumcheck_round_count_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        let _ = proof.stages.stage1.relations[0]
            .sumcheck_proof
            .round_polynomials
            .pop();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageSumcheckFailed {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_sumcheck_degree_bound_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        let relation = &mut proof.stages.stage1.relations[0];
        relation.sumcheck_proof.round_polynomials[0] =
            CompressedPoly::new(vec![Fq::default(); relation.sumcheck.degree + 1]);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageSumcheckFailed {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_empty_compressed_round() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.stages.stage1.relations[0]
            .sumcheck_proof
            .round_polynomials[0] = CompressedPoly::new(Vec::new());
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageSumcheckFailed {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_relation_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.stage1.gt_exponentiation.accumulator = Fq::from_u64(1);
        proof.claims.stage1.gt_exponentiation.digit_selector = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_digit_selector_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof
            .claims
            .stage1
            .gt_exponentiation_digit_selector
            .digit_lo = Fq::default();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_shift_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.stage1.gt_exponentiation_shift.accumulator = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_boundary_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.stage1.gt_exponentiation_boundary.accumulator = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_multiplication_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.stage1.gt_multiplication.opening.output = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_line_step_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.stage1.miller_loop.line_step.shifted_state_x[0] = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage1_line_evaluation_output_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof
            .claims
            .stage1
            .miller_loop
            .line_evaluation
            .line_evaluation_coeffs[6] = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage2_copy_catalog_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let proof = well_shaped_assist_proof();
        let mut changed_proof = proof.clone();
        let _ = changed_proof.stages.stage2.copy_constraints.pop();
        let mut right_transcript = Blake2bTranscript::new(b"dory-assist-test");

        let _ = run_stages_after_checked_preamble(&checked, &proof);
        let right = run_stages(&checked, &changed_proof, &mut right_transcript);

        assert!(matches!(
            right,
            Err(DoryAssistVerifierError::StageClaimMismatch {
                stage: DoryAssistStage::Stage2,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage2_copy_value_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.stage1.gt_exponentiation_digit_bitness.digit_lo = Fq::default();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage2,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage2_line_copy_value_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof
            .claims
            .stage1
            .miller_loop
            .line_evaluation
            .line_coefficients[0][0] = Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage2,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_challenge_changes_when_checked_preamble_changes() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut changed_input = fixture.clear_input();
        changed_input.eval += Fr::from_u64(1);
        let changed = checked_clear_inputs(changed_input);
        let proof = well_shaped_assist_proof();

        let left = run_stages_after_checked_preamble(&checked, &proof);
        let right = run_stages_after_checked_preamble(&changed, &proof);

        assert_ne!(left.challenge, right.challenge);
    }

    #[test]
    fn run_stages_rejects_empty_stage1_shape() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.stages.stage1.relations.clear();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage1.relations",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_empty_stage2_shape() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.stages.stage2.copy_constraints.clear();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage2.copy_constraints",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_empty_stage3_opening_row() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.opening_proof.combined_row.clear();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage3.opening_proof.combined_row",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_empty_stage3_packed_point() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.opening.packed_point.clear();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage3.claims.opening.packed_point",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_empty_stage3_dense_commitment() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.dense_commitment.rows.clear();
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage3.dense_commitment.rows",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_stage3_packed_eval_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.stages.stage3.packed_eval += Fq::from_u64(1);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage3,
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_non_power_of_two_hyrax_row_count() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.dense_commitment.rows.push(Default::default());
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage3.dense_commitment.rows",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_non_power_of_two_hyrax_row_len() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.opening_proof.combined_row.push(Fq::from_u64(37));
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage3.opening_proof.combined_row",
                ..
            })
        ));
    }

    #[test]
    fn run_stages_rejects_hyrax_dimension_mismatch() {
        let fixture = dory_opening_fixture();
        let checked = checked_clear_inputs(fixture.clear_input());
        let mut proof = well_shaped_assist_proof();
        proof.claims.opening.packed_point.push(Fq::from_u64(41));
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");

        let result = run_stages(&checked, &proof, &mut transcript);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage3.hyrax_dimensions",
                ..
            })
        ));
    }

    struct DoryOpeningFixture {
        verifier_setup: DoryVerifierSetup,
        proof: DoryProof,
        commitment: DoryCommitment,
        point: Vec<Fr>,
        eval: Fr,
    }

    impl DoryOpeningFixture {
        fn clear_input(&self) -> PcsAssistClearInput<'_, DoryScheme> {
            PcsAssistClearInput {
                setup: &self.verifier_setup,
                pcs_proof: &self.proof,
                commitment: &self.commitment,
                point: &self.point,
                eval: self.eval,
            }
        }

        fn zk_input(&self) -> PcsAssistZkInput<'_, DoryScheme> {
            PcsAssistZkInput {
                setup: &self.verifier_setup,
                pcs_proof: &self.proof,
                commitment: &self.commitment,
                point: &self.point,
            }
        }
    }

    fn assert_clear_opening_matches(
        opening: &ClearOpeningStatement<'_>,
        fixture: &DoryOpeningFixture,
    ) {
        assert_zk_opening_matches(
            &ZkOpeningStatement {
                setup: opening.setup,
                pcs_proof: opening.pcs_proof,
                commitment: opening.commitment,
                point: opening.point,
            },
            fixture,
        );
        assert_eq!(opening.eval, fixture.eval);
    }

    fn assert_zk_opening_matches(opening: &ZkOpeningStatement<'_>, fixture: &DoryOpeningFixture) {
        assert!(std::ptr::eq(
            opening.setup,
            std::ptr::from_ref(&fixture.verifier_setup)
        ));
        assert_eq!(opening.pcs_proof, &fixture.proof);
        assert_eq!(opening.commitment, &fixture.commitment);
        assert_eq!(opening.point, fixture.point.as_slice());
    }

    fn dory_opening_fixture() -> DoryOpeningFixture {
        dory_opening_fixture_with_shift(0)
    }

    fn dory_zk_opening_fixture() -> DoryOpeningFixture {
        dory_zk_opening_fixture_with_shift(0)
    }

    fn dory_opening_fixture_with_shift(shift: u64) -> DoryOpeningFixture {
        dory_opening_fixture_with_num_vars_and_shift(2, shift)
    }

    fn dory_opening_fixture_with_num_vars(num_vars: usize) -> DoryOpeningFixture {
        dory_opening_fixture_with_num_vars_and_shift(num_vars, 0)
    }

    fn dory_opening_fixture_with_num_vars_and_shift(
        num_vars: usize,
        shift: u64,
    ) -> DoryOpeningFixture {
        let (prover_setup, verifier_setup) = DoryScheme::setup(num_vars);
        let offset = Fr::from_u64(shift);
        let poly = Polynomial::<Fr>::from(
            (0..(1usize << num_vars))
                .map(|i| Fr::from_u64(u64::try_from(i + 1).expect("fixture index fits")) + offset)
                .collect::<Vec<_>>(),
        );
        let point = (0..num_vars)
            .map(|i| Fr::from_u64(u64::try_from(5 + 2 * i).expect("fixture point fits")) + offset)
            .collect::<Vec<_>>();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        );

        DoryOpeningFixture {
            verifier_setup,
            proof,
            commitment,
            point,
            eval,
        }
    }

    fn dory_zk_opening_fixture_with_shift(shift: u64) -> DoryOpeningFixture {
        let num_vars = 2;
        let (prover_setup, verifier_setup) = DoryScheme::setup(num_vars);
        let offset = Fr::from_u64(shift);
        let poly = Polynomial::<Fr>::from(vec![
            Fr::from_u64(1) + offset,
            Fr::from_u64(2) + offset,
            Fr::from_u64(3) + offset,
            Fr::from_u64(4) + offset,
        ]);
        let point = vec![Fr::from_u64(5) + offset, Fr::from_u64(7) + offset];
        let eval = poly.evaluate(&point);
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let (proof, _hiding_commitment, _blind) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript);

        DoryOpeningFixture {
            verifier_setup,
            proof,
            commitment,
            point,
            eval,
        }
    }

    fn checked_input_test_challenge(checked: &CheckedInputs<'_>) -> Fq {
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let _ = absorb_checked_inputs_for_test(checked, &mut transcript);
        squeeze_fq_challenge(&mut transcript, b"checked_test_challenge")
    }

    fn run_stages_after_checked_preamble(
        checked: &CheckedInputs<'_>,
        proof: &DoryAssistProof,
    ) -> Stage3Output {
        try_run_stages_after_checked_preamble(checked, proof).expect("stages verify")
    }

    fn try_run_stages_after_checked_preamble(
        checked: &CheckedInputs<'_>,
        proof: &DoryAssistProof,
    ) -> Result<Stage3Output, DoryAssistVerifierError> {
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let _ = absorb_checked_inputs_for_test(checked, &mut transcript);
        let _ = squeeze_checked_input_digest(&mut transcript);
        run_stages(checked, proof, &mut transcript)
    }

    #[expect(
        clippy::expect_used,
        reason = "test fixture dimensions are derived from checked Dory inputs"
    )]
    fn dory_assist_dimensions_for_checked(checked: &CheckedInputs<'_>) -> DoryAssistDimensions {
        let supported = default_dory_assist_dimensions();
        let unpacked = DoryAssistDimensions::new(
            supported.gt,
            supported.g1,
            supported.g2,
            supported.miller_loop,
            DoryReduceDimensions::new(
                checked.point().len(),
                checked.pcs_proof().reduce_round_count(),
            ),
            supported.wiring,
            PrefixPackingDimensions::new(0, 0, 0).expect("valid empty packing dimensions"),
        );
        let packing = composition::prefix_packing_catalog(unpacked)
            .minimal_dimensions()
            .expect("valid checked Dory-assist packing dimensions");

        DoryAssistDimensions::new(
            unpacked.gt,
            unpacked.g1,
            unpacked.g2,
            unpacked.miller_loop,
            unpacked.dory_reduce,
            unpacked.wiring,
            packing,
        )
    }

    fn well_shaped_assist_proof() -> DoryAssistProof {
        well_shaped_assist_proof_with_dimensions(default_dory_assist_dimensions())
    }

    fn well_shaped_assist_proof_with_dimensions(
        dimensions: DoryAssistDimensions,
    ) -> DoryAssistProof {
        let mut proof = DoryAssistProof {
            dimensions,
            ..DoryAssistProof::default()
        };
        proof
            .claims
            .stage1
            .public
            .input
            .dory_proof_artifacts
            .resize(
                DoryProofArtifactLayout::new(proof.dimensions.dory_reduce.reduce_rounds())
                    .expected_len(),
                Fq::default(),
            );
        proof
            .claims
            .stage1
            .public
            .input
            .verifier_setup_artifacts
            .resize(
                setup_artifacts::dory_setup_artifact_count(
                    proof.dimensions.dory_reduce.reduce_rounds(),
                ),
                Fq::default(),
            );
        proof.claims.stage1.public.input.transcript_scalars.resize(
            transcript_scalars::dory_reduce_s2_fold_factor(
                proof.dimensions.dory_reduce.point_len(),
                0,
            ) + 1,
            Fq::default(),
        );
        proof
            .claims
            .stage1
            .public
            .input
            .jolt_commitments
            .resize(1 + GT_ARTIFACT_COEFFS, Fq::default());
        proof.stages.stage1 =
            stages::stage1::Stage1Proof::canonical_for_dimensions(proof.dimensions);
        proof.stages.stage2 =
            stages::stage2::Stage2Proof::canonical_for_dimensions(proof.dimensions);
        bind_zero_dory_reduce_fixture(&mut proof);
        for constraint in dory_reduce::initial_state_copy_constraints() {
            bind_dory_reduce_copy_target(&mut proof, constraint);
        }
        populate_valid_hyrax_opening(&mut proof);
        proof
    }

    fn well_shaped_assist_proof_for_checked(checked: &CheckedInputs<'_>) -> DoryAssistProof {
        let mut proof =
            well_shaped_assist_proof_with_dimensions(dory_assist_dimensions_for_checked(checked));
        proof.claims.stage1.public.input = checked_input_public_claims_for_test(checked);
        bind_public_input_copy_fixture(&mut proof);
        bind_native_public_output_fixture(&mut proof, checked);
        proof
    }

    fn bind_zero_dory_reduce_fixture(proof: &mut DoryAssistProof) {
        proof.claims.stage1.dory_reduce.transitions.clear();
        proof.claims.stage1.dory_reduce.state_chain.clear();
        proof.claims.stage1.dory_reduce.boundary.clear();
        let protocol = protocol_claims::<Fq>(proof.dimensions);
        for relation_id in [
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryAssistRelationId::DoryReduceG1Transition,
            DoryAssistRelationId::DoryReduceG2Transition,
        ] {
            let relation = protocol
                .relation(relation_id)
                .expect("Dory-reduce transition relation is in the protocol catalog");
            proof.claims.stage1.dory_reduce.transitions.extend(
                relation
                    .required_openings()
                    .into_iter()
                    .map(|id| DoryAssistOpeningClaim {
                        id,
                        value: Fq::default(),
                    }),
            );
        }
        if proof.dimensions.dory_reduce.reduce_rounds() > 1 {
            for relation_id in [
                DoryAssistRelationId::DoryReduceStateChain,
                DoryAssistRelationId::DoryReduceBoundary,
            ] {
                let relation = protocol
                    .relation(relation_id)
                    .expect("Dory-reduce multi-round relation is in the protocol catalog");
                let claims =
                    relation
                        .required_openings()
                        .into_iter()
                        .map(|id| DoryAssistOpeningClaim {
                            id,
                            value: Fq::default(),
                        });
                match relation_id {
                    DoryAssistRelationId::DoryReduceStateChain => {
                        proof.claims.stage1.dory_reduce.state_chain.extend(claims);
                    }
                    DoryAssistRelationId::DoryReduceBoundary => {
                        proof.claims.stage1.dory_reduce.boundary.extend(claims);
                    }
                    _ => unreachable!("only multi-round Dory-reduce relations are handled here"),
                }
            }
        }
    }

    fn bind_public_input_copy_fixture(proof: &mut DoryAssistProof) {
        let vmv_c0 = proof.claims.stage1.public.input.dory_proof_artifacts[DORY_VMV_C_START];
        proof.claims.stage1.gt_exponentiation.accumulator = vmv_c0;
        proof.claims.stage1.gt_exponentiation_shift.accumulator = vmv_c0;
        proof.claims.stage1.gt_exponentiation_boundary.accumulator = vmv_c0;
        proof.claims.stage1.public.gt_shift_eq_kernel = Fq::default();
        proof
            .claims
            .stage1
            .public
            .gt_exponentiation_boundary
            .initial_value = vmv_c0;
        proof.claims.stage1.miller_loop.line_evaluation.g1_point_x =
            proof.claims.stage1.public.input.dory_proof_artifacts[DORY_VMV_E1_START];
        proof.claims.stage1.miller_loop.line_evaluation.g1_point_y =
            proof.claims.stage1.public.input.dory_proof_artifacts[DORY_VMV_E1_START + 1];
    }

    fn bind_native_public_output_fixture(proof: &mut DoryAssistProof, checked: &CheckedInputs<'_>) {
        let transcript = Blake2bTranscript::new(b"dory-assist-test");
        let scalars = dory_verifier_transcript_scalars(checked, &transcript);
        match checked {
            CheckedInputs::Clear(inputs) => {
                let native_final_inputs =
                    transparent_native_final_input_claims(&inputs.opening, &scalars)
                        .expect("transparent native-final inputs are well shaped");
                proof
                    .claims
                    .stage1
                    .public
                    .native_final
                    .bind(native_final_inputs);
                proof.public_outputs.pre_final_exponentiation =
                    transparent_replayed_final_pairing_check(&inputs.opening, &scalars)
                        .expect("transparent final fixture is well shaped")
                        .pre_final_exponentiation();
            }
            CheckedInputs::Zk(inputs) => {
                let native_final_inputs = zk_native_final_input_claims(&inputs.opening, &scalars)
                    .expect("ZK native-final inputs are well shaped");
                proof
                    .claims
                    .stage1
                    .public
                    .native_final
                    .bind(native_final_inputs);
                proof.public_outputs.pre_final_exponentiation =
                    zk_replayed_final_pairing_check(&inputs.opening, &scalars)
                        .expect("ZK final fixture is well shaped")
                        .pre_final_exponentiation();
            }
        }

        let output_coefficients = proof.public_outputs.pre_final_exponentiation_coefficients();
        proof
            .claims
            .stage1
            .public
            .miller_loop
            .bind_pre_final_exponentiation(&proof.public_outputs);
        proof
            .claims
            .stage1
            .public
            .miller_loop
            .boundary_initial_value = output_coefficients;
        proof.claims.stage1.miller_loop.accumulator.accumulator = output_coefficients;
        proof
            .claims
            .stage1
            .miller_loop
            .accumulator
            .shifted_accumulator = output_coefficients;
        proof.claims.stage1.miller_loop.boundary.accumulator = output_coefficients;
        proof.claims.stage1.miller_loop.boundary.shifted_accumulator = output_coefficients;

        let square_row =
            &mut proof.claims.stage1.gt_multiplication.rows[composition::ACCUMULATOR_SQUARE_GT_ROW];
        square_row.left = output_coefficients;
        square_row.right = output_coefficients;
        square_row.output = output_coefficients;

        let mul_row =
            &mut proof.claims.stage1.gt_multiplication.rows[composition::ACCUMULATOR_MUL_GT_ROW];
        mul_row.left = output_coefficients;
        mul_row.output = output_coefficients;

        proof
            .claims
            .stage1
            .public
            .miller_loop
            .accumulator_shift_eq_kernel = accumulator_zero_sumcheck_kernel(proof, checked);
        bind_dory_reduce_transition_copy_fixture(proof, checked);
        populate_valid_hyrax_opening(proof);
    }

    fn tamper_native_final_c_acc(proof: &mut DoryAssistProof) {
        let inputs = &mut proof.claims.stage1.public.native_final.inputs;
        let identity = Bn254GT::identity().fq12_coefficients();
        let current_is_identity = inputs
            [NATIVE_FINAL_GT_C_START..NATIVE_FINAL_GT_C_START + Bn254GT::FQ12_COEFFICIENTS]
            .iter()
            .copied()
            .eq(identity);
        if current_is_identity {
            inputs.copy_within(
                NATIVE_FINAL_D1_START..NATIVE_FINAL_D1_START + GT_ARTIFACT_COEFFS,
                NATIVE_FINAL_GT_C_START,
            );
            return;
        }

        inputs[NATIVE_FINAL_GT_C_START..NATIVE_FINAL_GT_C_START + Bn254GT::FQ12_COEFFICIENTS]
            .copy_from_slice(&identity);
        inputs[NATIVE_FINAL_GT_C_START + Bn254GT::FQ12_COEFFICIENTS
            ..NATIVE_FINAL_GT_C_START + GT_ARTIFACT_COEFFS]
            .fill(Fq::default());
    }

    fn bind_dory_reduce_transition_copy_fixture(
        proof: &mut DoryAssistProof,
        checked: &CheckedInputs<'_>,
    ) {
        for constraint in dory_reduce::initial_state_copy_constraints() {
            bind_dory_reduce_copy_target(proof, constraint);
        }
        if proof.dimensions.dory_reduce.reduce_rounds() == 1 {
            for constraint in dory_reduce_transition_copy_constraints(proof.dimensions) {
                bind_dory_reduce_copy_target(proof, constraint);
            }
        }

        bind_dory_reduce_public_fold_fixture(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceGtTransition,
        );
        rebalance_dory_reduce_transition_relation(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceGtTransition,
        );
        bind_dory_reduce_public_fold_fixture(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceG1Transition,
        );
        rebalance_dory_reduce_transition_relation(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceG1Transition,
        );
        bind_dory_reduce_public_fold_fixture(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceG2Transition,
        );
        rebalance_dory_reduce_transition_relation(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceG2Transition,
        );
        bind_dory_reduce_public_fold_fixture(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceScalarFold,
        );
        rebalance_dory_reduce_scalar_fold_relation(proof, checked);
        if proof.dimensions.dory_reduce.reduce_rounds() > 1 {
            bind_dory_reduce_boundary_fixture(proof);
        }
    }

    fn dory_reduce_transition_copy_constraints(
        dimensions: jolt_claims::protocols::dory_assist::DoryAssistDimensions,
    ) -> Vec<DoryAssistCopyConstraint> {
        dory_reduce::proof_artifact_copy_constraints(0)
            .into_iter()
            .chain(dory_reduce::round_setup_artifact_copy_constraints(
                dimensions.dory_reduce.reduce_rounds(),
                0,
            ))
            .chain(dory_reduce::transition_transcript_scalar_copy_constraints(
                dimensions.dory_reduce.point_len(),
                0,
            ))
            .collect()
    }

    fn bind_dory_reduce_copy_target(
        proof: &mut DoryAssistProof,
        constraint: DoryAssistCopyConstraint,
    ) {
        let value = match constraint.source {
            DoryAssistValueRef::Public { id, .. } => proof
                .claims
                .stage1
                .public
                .claim(&id)
                .expect("fixture public claim exists"),
            DoryAssistValueRef::Constant(value) => Fq::from_u64(value as u64),
            DoryAssistValueRef::Witness { .. } | DoryAssistValueRef::Challenge(_) => {
                panic!("Dory-reduce fixture copy source must be public or constant")
            }
        };
        let opening = constraint
            .target
            .witness_opening()
            .expect("Dory-reduce transition fixture copy target must be witness");
        set_dory_reduce_opening(proof, opening, value);
    }

    #[expect(
        clippy::expect_used,
        reason = "test fixture public-fold sources are derived from canonical Dory-reduce dimensions"
    )]
    fn bind_dory_reduce_public_fold_fixture(
        proof: &mut DoryAssistProof,
        checked: &CheckedInputs<'_>,
        relation_id: DoryAssistRelationId,
    ) {
        let context = stage1_relation_context_for_test(proof, checked, relation_id);
        let weights = EqPolynomial::new(context.sumcheck_point).evaluations();
        for constraint in dory_reduce::public_fold_constraints(proof.dimensions.dory_reduce) {
            let opening = constraint
                .target
                .witness_opening()
                .expect("Dory-reduce public fold target is a witness opening");
            if opening_relation(opening) != relation_id {
                continue;
            }
            assert!(
                constraint.sources.len() <= weights.len(),
                "public fold sources fit the relation point domain"
            );
            let value =
                constraint
                    .sources
                    .iter()
                    .zip(&weights)
                    .fold(Fq::default(), |acc, (id, weight)| {
                        let public = proof
                            .claims
                            .stage1
                            .public_claim(id)
                            .expect("fixture public-fold source exists");
                        acc + public * *weight
                    });
            set_dory_reduce_opening(proof, opening, value);
        }
    }

    fn bind_dory_reduce_boundary_fixture(proof: &mut DoryAssistProof) {
        for term in dory_reduce::initial_boundary_terms()
            .into_iter()
            .chain(dory_reduce::final_boundary_terms())
        {
            let value = match term.value {
                dory_reduce::DoryReduceBoundaryValue::ConstantOne => Fq::from_u64(1),
                dory_reduce::DoryReduceBoundaryValue::Public(id) => proof
                    .claims
                    .stage1
                    .public_claim(&id)
                    .expect("fixture Dory-reduce boundary public claim exists"),
            };
            set_dory_reduce_opening(proof, term.opening, value);
        }
    }

    fn rebalance_dory_reduce_transition_relation(
        proof: &mut DoryAssistProof,
        checked: &CheckedInputs<'_>,
        relation_id: DoryAssistRelationId,
    ) {
        let protocol = protocol_claims::<Fq>(proof.dimensions);
        let relation = protocol
            .relation(relation_id)
            .expect("Dory-reduce transition relation is in the protocol catalog");
        let context = stage1_relation_context_for_test(proof, checked, relation_id);
        let target = dory_reduce_transition_target_opening(relation_id);
        let input = relation
            .input
            .expression()
            .try_evaluate(
                |id| {
                    proof
                        .claims
                        .stage1
                        .opening_claim(id)
                        .ok_or("missing opening")
                },
                |id| resolve_test_challenge(&context.relation_challenges, id),
                |id| proof.claims.stage1.public_claim(id).ok_or("missing public"),
            )
            .expect("fixture Dory-reduce transition input evaluates");
        let output = relation
            .output
            .expression()
            .try_evaluate(
                |id| {
                    proof
                        .claims
                        .stage1
                        .opening_claim(id)
                        .ok_or("missing opening")
                },
                |id| resolve_test_challenge(&context.relation_challenges, id),
                |id| proof.claims.stage1.public_claim(id).ok_or("missing public"),
            )
            .expect("fixture Dory-reduce transition output evaluates");
        let factor = sumcheck_linear_factor(&context.sumcheck_point);
        let delta = (output - input * factor)
            * factor
                .inverse()
                .expect("fixture Dory-reduce sumcheck factor is nonzero");
        set_dory_reduce_opening(
            proof,
            target,
            get_dory_reduce_opening(proof, target) + delta,
        );
    }

    fn rebalance_dory_reduce_scalar_fold_relation(
        proof: &mut DoryAssistProof,
        checked: &CheckedInputs<'_>,
    ) {
        let protocol = protocol_claims::<Fq>(proof.dimensions);
        let relation = protocol
            .relation(DoryAssistRelationId::DoryReduceScalarFold)
            .expect("Dory-reduce scalar-fold relation is in the protocol catalog");
        let context = stage1_relation_context_for_test(
            proof,
            checked,
            DoryAssistRelationId::DoryReduceScalarFold,
        );
        let input = relation
            .input
            .expression()
            .try_evaluate(
                |id| {
                    proof
                        .claims
                        .stage1
                        .opening_claim(id)
                        .ok_or("missing opening")
                },
                |id| resolve_test_challenge(&context.relation_challenges, id),
                |id| proof.claims.stage1.public_claim(id).ok_or("missing public"),
            )
            .expect("fixture Dory-reduce scalar-fold input evaluates");
        let output = relation
            .output
            .expression()
            .try_evaluate(
                |id| {
                    proof
                        .claims
                        .stage1
                        .opening_claim(id)
                        .ok_or("missing opening")
                },
                |id| resolve_test_challenge(&context.relation_challenges, id),
                |id| proof.claims.stage1.public_claim(id).ok_or("missing public"),
            )
            .expect("fixture Dory-reduce scalar-fold output evaluates");
        let factor = sumcheck_linear_factor(&context.sumcheck_point);
        let target = dory_reduce::s1_next_accumulator_opening();
        let delta = (output - input * factor)
            * factor
                .inverse()
                .expect("fixture Dory-reduce scalar-fold sumcheck factor is nonzero");
        set_dory_reduce_opening(
            proof,
            target,
            get_dory_reduce_opening(proof, target) + delta,
        );
    }

    fn dory_reduce_transition_target_opening(
        relation: DoryAssistRelationId,
    ) -> DoryAssistOpeningId {
        let polynomial = match relation {
            DoryAssistRelationId::DoryReduceGtTransition => DoryReducePolynomial::NextC(0),
            DoryAssistRelationId::DoryReduceG1Transition => DoryReducePolynomial::NextE1X,
            DoryAssistRelationId::DoryReduceG2Transition => DoryReducePolynomial::NextE2X0,
            _ => panic!("not a Dory-reduce transition relation"),
        };
        dory_reduce_opening(relation, polynomial)
    }

    fn dory_reduce_opening(
        relation: DoryAssistRelationId,
        polynomial: DoryReducePolynomial,
    ) -> DoryAssistOpeningId {
        DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::DoryReduce(polynomial),
            relation,
        )
    }

    fn set_dory_reduce_transition_opening(
        proof: &mut DoryAssistProof,
        opening: DoryAssistOpeningId,
        value: Fq,
    ) {
        let claim = proof
            .claims
            .stage1
            .dory_reduce
            .transitions
            .iter_mut()
            .find(|claim| claim.id == opening)
            .expect("fixture contains Dory-reduce transition opening");
        claim.value = value;
    }

    fn set_dory_reduce_opening_claim(
        claims: &mut [DoryAssistOpeningClaim],
        opening: DoryAssistOpeningId,
        value: Fq,
    ) {
        let claim = claims
            .iter_mut()
            .find(|claim| claim.id == opening)
            .expect("fixture contains Dory-reduce relation opening");
        claim.value = value;
    }

    fn set_dory_reduce_opening(
        proof: &mut DoryAssistProof,
        opening: DoryAssistOpeningId,
        value: Fq,
    ) {
        if opening == dory_reduce::s1_accumulator_opening() {
            proof.claims.stage1.dory_reduce.scalar_fold.s1_accumulator = value;
        } else if opening == dory_reduce::s1_next_accumulator_opening() {
            proof
                .claims
                .stage1
                .dory_reduce
                .scalar_fold
                .s1_next_accumulator = value;
        } else if opening == dory_reduce::s1_fold_factor_opening() {
            proof.claims.stage1.dory_reduce.scalar_fold.s1_fold_factor = value;
        } else if opening == dory_reduce::s2_accumulator_opening() {
            proof.claims.stage1.dory_reduce.scalar_fold.s2_accumulator = value;
        } else if opening == dory_reduce::s2_next_accumulator_opening() {
            proof
                .claims
                .stage1
                .dory_reduce
                .scalar_fold
                .s2_next_accumulator = value;
        } else if opening == dory_reduce::s2_fold_factor_opening() {
            proof.claims.stage1.dory_reduce.scalar_fold.s2_fold_factor = value;
        } else {
            match opening_relation(opening) {
                DoryAssistRelationId::DoryReduceGtTransition
                | DoryAssistRelationId::DoryReduceG1Transition
                | DoryAssistRelationId::DoryReduceG2Transition => {
                    set_dory_reduce_transition_opening(proof, opening, value);
                }
                DoryAssistRelationId::DoryReduceStateChain => set_dory_reduce_opening_claim(
                    &mut proof.claims.stage1.dory_reduce.state_chain,
                    opening,
                    value,
                ),
                DoryAssistRelationId::DoryReduceBoundary => set_dory_reduce_opening_claim(
                    &mut proof.claims.stage1.dory_reduce.boundary,
                    opening,
                    value,
                ),
                relation => panic!("fixture cannot set non-Dory-reduce opening {relation:?}"),
            }
        }
    }

    fn get_dory_reduce_opening(proof: &DoryAssistProof, opening: DoryAssistOpeningId) -> Fq {
        proof
            .claims
            .stage1
            .opening_claim(&opening)
            .expect("fixture Dory-reduce opening exists")
    }

    fn opening_relation(opening: DoryAssistOpeningId) -> DoryAssistRelationId {
        let DoryAssistOpeningId::Polynomial { relation, .. } = opening;
        relation
    }

    fn checked_input_public_claims_for_test(
        checked: &CheckedInputs<'_>,
    ) -> DoryAssistInputPublicClaims {
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let mut input_public_claims = absorb_checked_inputs_for_test(checked, &mut transcript);
        input_public_claims.checked_input_digest = squeeze_checked_input_digest(&mut transcript);
        input_public_claims
    }

    fn accumulator_zero_sumcheck_kernel(
        proof: &DoryAssistProof,
        checked: &CheckedInputs<'_>,
    ) -> Fq {
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let _ = absorb_checked_inputs_for_test(checked, &mut transcript);
        let _ = squeeze_checked_input_digest(&mut transcript);
        absorb_stage1_preamble_for_test(
            checked.mode_name().as_bytes(),
            proof.stages.stage1.relation_count(),
            &mut transcript,
        );

        let protocol = protocol_claims::<Fq>(proof.dimensions);
        for relation in &proof.stages.stage1.relations {
            let relation_claims = protocol
                .relation(relation.id)
                .expect("stage 1 relation belongs to Dory-assist protocol");
            absorb_stage1_relation_for_test(relation.id, &relation.sumcheck, &mut transcript);
            let relation_challenges = relation_claims
                .required_challenges()
                .into_iter()
                .map(|id| (id, squeeze_fq(&mut transcript)))
                .collect::<Vec<_>>();

            if relation.id == DoryAssistRelationId::MillerLoopAccumulator {
                let input_claim = relation_claims
                    .input
                    .expression()
                    .try_evaluate(
                        |id| {
                            proof
                                .claims
                                .stage1
                                .opening_claim(id)
                                .ok_or("missing opening")
                        },
                        |id| resolve_test_challenge(&relation_challenges, id),
                        |id| proof.claims.stage1.public_claim(id).ok_or("missing public"),
                    )
                    .expect("fixture accumulator input evaluates");
                let final_claim = relation.sumcheck_proof.round_polynomials.iter().fold(
                    input_claim,
                    |running_sum, round_proof| {
                        absorb_sumcheck_round_for_test(round_proof, &mut transcript);
                        let challenge = squeeze_fq(&mut transcript);
                        round_proof.evaluate_with_hint(running_sum, challenge)
                    },
                );
                return final_claim
                    * input_claim
                        .inverse()
                        .expect("native-output fixture has nonzero accumulator claim");
            }

            for round_proof in &relation.sumcheck_proof.round_polynomials {
                absorb_sumcheck_round_for_test(round_proof, &mut transcript);
                let _ = squeeze_fq(&mut transcript);
            }
            for id in relation_claims.required_openings() {
                let value = proof
                    .claims
                    .stage1
                    .opening_claim(&id)
                    .expect("fixture has canonical opening claim");
                transcript.append_labeled(b"opening_claim", &value);
            }
        }

        panic!("canonical Stage 1 relation catalog has no Miller-loop accumulator relation");
    }

    struct Stage1RelationContextForTest {
        relation_challenges: Vec<(DoryAssistChallengeId, Fq)>,
        sumcheck_point: Vec<Fq>,
    }

    fn stage1_relation_context_for_test(
        proof: &DoryAssistProof,
        checked: &CheckedInputs<'_>,
        target: DoryAssistRelationId,
    ) -> Stage1RelationContextForTest {
        let mut transcript = Blake2bTranscript::new(b"dory-assist-test");
        let _ = absorb_checked_inputs_for_test(checked, &mut transcript);
        let _ = squeeze_checked_input_digest(&mut transcript);
        absorb_stage1_preamble_for_test(
            checked.mode_name().as_bytes(),
            proof.stages.stage1.relation_count(),
            &mut transcript,
        );

        let protocol = protocol_claims::<Fq>(proof.dimensions);
        for relation in &proof.stages.stage1.relations {
            let relation_claims = protocol
                .relation(relation.id)
                .expect("stage 1 relation belongs to Dory-assist protocol");
            absorb_stage1_relation_for_test(relation.id, &relation.sumcheck, &mut transcript);
            let relation_challenges = relation_claims
                .required_challenges()
                .into_iter()
                .map(|id| (id, squeeze_fq(&mut transcript)))
                .collect::<Vec<_>>();

            if relation.id == target {
                let mut sumcheck_point = Vec::with_capacity(relation.sumcheck.rounds);
                for round_proof in &relation.sumcheck_proof.round_polynomials {
                    absorb_sumcheck_round_for_test(round_proof, &mut transcript);
                    sumcheck_point.push(squeeze_fq(&mut transcript));
                }
                return Stage1RelationContextForTest {
                    relation_challenges,
                    sumcheck_point,
                };
            }

            for round_proof in &relation.sumcheck_proof.round_polynomials {
                absorb_sumcheck_round_for_test(round_proof, &mut transcript);
                let _ = squeeze_fq(&mut transcript);
            }
            for id in relation_claims.required_openings() {
                let value = proof
                    .claims
                    .stage1
                    .opening_claim(&id)
                    .expect("fixture has canonical opening claim");
                transcript.append_labeled(b"opening_claim", &value);
            }
        }

        panic!("target relation {target:?} is absent from the canonical Stage 1 catalog");
    }

    fn sumcheck_linear_factor(point: &[Fq]) -> Fq {
        point
            .iter()
            .copied()
            .fold(Fq::from_u64(1), |acc, challenge| acc * challenge)
    }

    fn resolve_test_challenge(
        challenges: &[(DoryAssistChallengeId, Fq)],
        id: &DoryAssistChallengeId,
    ) -> Result<Fq, &'static str> {
        challenges
            .iter()
            .find(|(candidate, _)| candidate == id)
            .map(|(_, value)| *value)
            .ok_or("missing challenge")
    }

    fn absorb_stage1_preamble_for_test(
        mode_name: &'static [u8],
        relation_count: u32,
        transcript: &mut Blake2bTranscript,
    ) {
        transcript.append(&Label(b"dory_assist_stage1"));
        transcript.append(&Label(mode_name));
        transcript.append(&Label(b"stage1_relations"));
        transcript.append(&U64Word(relation_count as u64));
    }

    fn absorb_stage1_relation_for_test(
        id: DoryAssistRelationId,
        sumcheck: &DoryAssistSumcheckSpec,
        transcript: &mut Blake2bTranscript,
    ) {
        transcript.append(&Label(b"stage1_relation_id"));
        transcript.append(&U64Word(relation_transcript_tag_for_test(id) as u64));
        transcript.append(&Label(b"stage1_sumcheck_domain"));
        transcript.append(&U64Word(0));
        transcript.append(&Label(b"stage1_sumcheck_rounds"));
        transcript.append(&U64Word(sumcheck.rounds as u64));
        transcript.append(&Label(b"stage1_sumcheck_degree"));
        transcript.append(&U64Word(sumcheck.degree as u64));
    }

    fn absorb_sumcheck_round_for_test(
        round_proof: &CompressedPoly<Fq>,
        transcript: &mut Blake2bTranscript,
    ) {
        let coeffs = round_proof.coeffs_except_linear_term();
        transcript.append(&LabelWithCount(
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            coeffs.len() as u64,
        ));
        for coeff in coeffs {
            transcript.append(coeff);
        }
    }

    fn relation_transcript_tag_for_test(id: DoryAssistRelationId) -> usize {
        CANONICAL_RELATION_ORDER
            .iter()
            .position(|candidate| *candidate == id)
            .expect("stage 1 relation has a canonical transcript tag")
    }

    fn populate_valid_hyrax_opening(proof: &mut DoryAssistProof) {
        let reduced_claims = reduced_opening_claims(proof);
        let poly_len = reduced_claims.len().next_power_of_two();
        let num_vars = poly_len.trailing_zeros() as usize;
        let row_vars = num_vars / 2;
        let col_vars = num_vars - row_vars;
        let dimensions =
            HyraxDimensions::new(num_vars, row_vars, col_vars).expect("valid Hyrax dimensions");
        let hyrax_setup = derive_hyrax_prover_setup(dimensions).expect("seed-derived Hyrax setup");
        let mut evaluations = vec![Fq::default(); poly_len];
        for (slot, claim) in evaluations.iter_mut().zip(&reduced_claims) {
            *slot = claim.value;
        }
        let packed_poly = Polynomial::<Fq>::from(evaluations);
        let packed_point = (0..num_vars)
            .map(|index| Fq::from_u64(13 + 6 * index as u64))
            .collect::<Vec<_>>();
        let packed_eval = packed_poly.evaluate(&packed_point);
        let (dense_commitment, hint) = DoryAssistHyrax::commit(&packed_poly, &hyrax_setup);
        let mut transcript = Blake2bTranscript::new(b"dory-assist-hyrax-unit-fixture");
        let opening_proof = DoryAssistHyrax::open(
            &packed_poly,
            &packed_point,
            packed_eval,
            &hyrax_setup,
            Some(hint),
            &mut transcript,
        );

        proof.stages.stage3.packed_eval = packed_eval;
        proof.stages.stage3.reduced_openings =
            reduced_claims.iter().map(|claim| claim.id).collect();
        proof.claims.opening.packed_point = packed_point;
        proof.claims.opening.packed_eval = packed_eval;
        proof.opening_proof = opening_proof;
        proof.dense_commitment = dense_commitment;
    }

    fn reduced_opening_claims(proof: &DoryAssistProof) -> Vec<DoryAssistOpeningClaim> {
        let protocol = protocol_claims::<Fq>(proof.dimensions);
        let mut reduced_claims = Vec::new();
        for relation in &proof.stages.stage1.relations {
            let relation_claims = protocol
                .relation(relation.id)
                .expect("stage 1 relation belongs to Dory-assist protocol");
            for id in relation_claims.required_openings() {
                if reduced_claims
                    .iter()
                    .any(|claim: &DoryAssistOpeningClaim| claim.id == id)
                {
                    continue;
                }
                let value = proof
                    .claims
                    .stage1
                    .opening_claim(&id)
                    .expect("stage 1 claim value exists for canonical opening");
                reduced_claims.push(DoryAssistOpeningClaim { id, value });
            }
        }
        reduced_claims
    }
}
