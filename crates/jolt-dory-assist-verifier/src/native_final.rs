use jolt_crypto::{Bn254, Bn254Fq12, Bn254G1, Bn254G2, Bn254GT, JoltGroup};
use jolt_dory::{DoryReduceRoundArtifacts, DoryVerifierTranscriptScalars};
use jolt_field::{CanonicalBytes, FixedByteSize, Fq, Fr, FromPrimitiveInt};

use crate::{
    artifacts::{G1_ARTIFACT_COORDS, G2_ARTIFACT_COORDS, GT_ARTIFACT_COEFFS},
    proof::{
        NATIVE_FINAL_D1_START, NATIVE_FINAL_D2_INIT_START, NATIVE_FINAL_D2_START,
        NATIVE_FINAL_E1_INIT_START, NATIVE_FINAL_E1_START, NATIVE_FINAL_E2_START,
        NATIVE_FINAL_GT_C_START, NATIVE_FINAL_INPUT_LEN, NATIVE_FINAL_S1_ACC_INDEX,
        NATIVE_FINAL_S2_ACC_INDEX,
    },
    verifier::{inject_fr_to_fq, ClearOpeningStatement, ZkOpeningStatement},
    DoryAssistVerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeFinalPairingCheck {
    pub g1_terms: [Bn254G1; 4],
    pub g2_terms: [Bn254G2; 4],
    pub rhs: Bn254GT,
}

impl NativeFinalPairingCheck {
    pub fn pre_final_exponentiation(&self) -> Bn254Fq12 {
        Bn254::multi_miller_loop(&self.g1_terms, &self.g2_terms)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkNativeFinalPairingCheck {
    pub g1_term: Bn254G1,
    pub g2_term: Bn254G2,
    pub rhs: Bn254GT,
}

impl ZkNativeFinalPairingCheck {
    pub fn pre_final_exponentiation(&self) -> Bn254Fq12 {
        Bn254::multi_miller_loop(&[self.g1_term], &[self.g2_term])
    }
}

pub fn transparent_final_pairing_check(
    input: &ClearOpeningStatement<'_>,
    scalars: &DoryVerifierTranscriptScalars,
    native_final_inputs: &[Fq],
) -> Result<NativeFinalPairingCheck, DoryAssistVerifierError> {
    let setup = input.setup.artifacts();
    let proof = input.pcs_proof;
    let final_artifacts = proof.final_artifacts();
    let state = native_final_input_state(native_final_inputs)?;

    transparent_final_pairing_check_from_state(&setup, &final_artifacts, &state, scalars)
}

pub fn transparent_replayed_final_pairing_check(
    input: &ClearOpeningStatement<'_>,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<NativeFinalPairingCheck, DoryAssistVerifierError> {
    let setup = input.setup.artifacts();
    let final_artifacts = input.pcs_proof.final_artifacts();
    let state = replay_reduce_native_state(
        input.pcs_proof,
        &setup,
        &input.commitment.0,
        setup.g2_0.scalar_mul(&input.eval),
        scalars,
    )?;

    transparent_final_pairing_check_from_state(&setup, &final_artifacts, &state, scalars)
}

fn transparent_final_pairing_check_from_state(
    setup: &jolt_dory::DoryVerifierSetupArtifacts,
    final_artifacts: &jolt_dory::DoryFinalArtifacts,
    state: &DoryReduceNativeState,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<NativeFinalPairingCheck, DoryAssistVerifierError> {
    let d = scalars.d;
    let d_inverse = scalars.d_inverse;
    let d_squared = scalars.d_squared;
    let gamma = scalars.gamma;
    let gamma_inverse = scalars.gamma_inverse;
    let s_product = state.s1_acc * state.s2_acc;
    let chi_0 = *setup.chi.first().ok_or_else(|| {
        invalid_final_shape("verifier setup chi must contain the final chi_0 term")
    })?;

    let rhs = state.c
        + setup.ht.scalar_mul(&s_product)
        + chi_0
        + state.d2.scalar_mul(&d)
        + state.d1.scalar_mul(&d_inverse)
        + state.d2_init.scalar_mul(&d_squared);

    Ok(NativeFinalPairingCheck {
        g1_terms: [
            final_artifacts.e1 + setup.g1_0.scalar_mul(&d),
            setup.h1,
            (state.e1 + setup.g1_0.scalar_mul(&(d * state.s2_acc))).scalar_mul(&(-gamma_inverse)),
            state.e1_init.scalar_mul(&d_squared),
        ],
        g2_terms: [
            final_artifacts.e2 + setup.g2_0.scalar_mul(&d_inverse),
            (state.e2 + setup.g2_0.scalar_mul(&(d_inverse * state.s1_acc))).scalar_mul(&(-gamma)),
            setup.h2,
            setup.g2_0,
        ],
        rhs,
    })
}

pub fn zk_final_pairing_check(
    input: &ZkOpeningStatement<'_>,
    scalars: &DoryVerifierTranscriptScalars,
    native_final_inputs: &[Fq],
) -> Result<ZkNativeFinalPairingCheck, DoryAssistVerifierError> {
    let setup = input.setup.artifacts();
    let proof = input.pcs_proof;
    let scalar_product = proof.scalar_product_artifacts().ok_or_else(|| {
        invalid_final_shape("ZK final check missing Dory scalar-product proof artifacts")
    })?;
    let sigma_c = scalars
        .scalar_product_sigma_c
        .ok_or_else(|| invalid_final_shape("ZK final check missing sigma_c transcript scalar"))?;
    let state = native_final_input_state(native_final_inputs)?;

    zk_final_pairing_check_from_state(&setup, &scalar_product, &state, scalars, sigma_c)
}

pub fn zk_replayed_final_pairing_check(
    input: &ZkOpeningStatement<'_>,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<ZkNativeFinalPairingCheck, DoryAssistVerifierError> {
    let setup = input.setup.artifacts();
    let proof = input.pcs_proof;
    let e2 = proof
        .zk_artifacts()
        .e2
        .ok_or_else(|| invalid_final_shape("ZK final check missing Dory proof e2 artifact"))?;
    let scalar_product = proof.scalar_product_artifacts().ok_or_else(|| {
        invalid_final_shape("ZK final check missing Dory scalar-product proof artifacts")
    })?;
    let sigma_c = scalars
        .scalar_product_sigma_c
        .ok_or_else(|| invalid_final_shape("ZK final check missing sigma_c transcript scalar"))?;
    let state = replay_reduce_native_state(proof, &setup, &input.commitment.0, e2, scalars)?;

    zk_final_pairing_check_from_state(&setup, &scalar_product, &state, scalars, sigma_c)
}

fn zk_final_pairing_check_from_state(
    setup: &jolt_dory::DoryVerifierSetupArtifacts,
    scalar_product: &jolt_dory::DoryScalarProductProofArtifacts,
    state: &DoryReduceNativeState,
    scalars: &DoryVerifierTranscriptScalars,
    sigma_c: Fr,
) -> Result<ZkNativeFinalPairingCheck, DoryAssistVerifierError> {
    let d = scalars.d;
    let d_inverse = scalars.d_inverse;
    let sigma_c_squared = sigma_c * sigma_c;
    let chi_0 = *setup.chi.first().ok_or_else(|| {
        invalid_final_shape("verifier setup chi must contain the final chi_0 term")
    })?;
    let ht_scalar = scalar_product.r3 + d * scalar_product.r2 + d_inverse * scalar_product.r1;

    let rhs = chi_0
        + scalar_product.r
        + scalar_product.q.scalar_mul(&sigma_c)
        + state.c.scalar_mul(&sigma_c_squared)
        + scalar_product.p2.scalar_mul(&d)
        + state.d2.scalar_mul(&(d * sigma_c))
        + scalar_product.p1.scalar_mul(&d_inverse)
        + state.d1.scalar_mul(&(d_inverse * sigma_c))
        - setup.ht.scalar_mul(&ht_scalar);

    Ok(ZkNativeFinalPairingCheck {
        g1_term: scalar_product.e1 + setup.g1_0.scalar_mul(&d),
        g2_term: scalar_product.e2 + setup.g2_0.scalar_mul(&d_inverse),
        rhs,
    })
}

pub fn transparent_native_final_input_claims(
    input: &ClearOpeningStatement<'_>,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<Vec<Fq>, DoryAssistVerifierError> {
    let setup = input.setup.artifacts();
    let state = replay_reduce_native_state(
        input.pcs_proof,
        &setup,
        &input.commitment.0,
        setup.g2_0.scalar_mul(&input.eval),
        scalars,
    )?;

    Ok(native_final_input_claims(&state))
}

pub fn zk_native_final_input_claims(
    input: &ZkOpeningStatement<'_>,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<Vec<Fq>, DoryAssistVerifierError> {
    let setup = input.setup.artifacts();
    let e2 = input
        .pcs_proof
        .zk_artifacts()
        .e2
        .ok_or_else(|| invalid_final_shape("ZK final inputs missing Dory proof e2 artifact"))?;
    let state =
        replay_reduce_native_state(input.pcs_proof, &setup, &input.commitment.0, e2, scalars)?;

    Ok(native_final_input_claims(&state))
}

fn replay_reduce_native_state(
    proof: &jolt_dory::DoryProof,
    setup: &jolt_dory::DoryVerifierSetupArtifacts,
    commitment: &Bn254GT,
    e2: Bn254G2,
    scalars: &DoryVerifierTranscriptScalars,
) -> Result<DoryReduceNativeState, DoryAssistVerifierError> {
    let vmv = proof.vmv_artifacts();
    let mut state = DoryReduceNativeState {
        c: vmv.c,
        d1: *commitment,
        d2: vmv.d2,
        e1: vmv.e1,
        e2,
        e1_init: vmv.e1,
        d2_init: vmv.d2,
        s1_acc: Fr::from_u64(1),
        s2_acc: Fr::from_u64(1),
    };

    let round_artifacts = proof.reduce_round_artifacts();
    if round_artifacts.len() != scalars.reduce_rounds.len() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "dory_final.reduce_rounds",
            reason: format!(
                "Dory final native replay has {} reduce artifacts but {} scalar rounds",
                round_artifacts.len(),
                scalars.reduce_rounds.len()
            ),
        });
    }

    for (round, (artifacts, scalar_round)) in round_artifacts
        .iter()
        .zip(&scalars.reduce_rounds)
        .enumerate()
    {
        let setup_index = round_artifacts.len() - round;
        state.process_round(
            artifacts,
            setup,
            setup_index,
            DoryReduceNativeScalars {
                beta: scalar_round.beta,
                beta_inverse: scalar_round.beta_inverse,
                alpha: scalar_round.alpha,
                alpha_inverse: scalar_round.alpha_inverse,
                alpha_beta: scalar_round.alpha_beta,
                alpha_inverse_beta_inverse: scalar_round.alpha_inverse_beta_inverse,
                s1_fold_factor: scalar_round.s1_fold_factor,
                s2_fold_factor: scalar_round.s2_fold_factor,
            },
        )?;
    }

    Ok(state)
}

fn native_final_input_claims(state: &DoryReduceNativeState) -> Vec<Fq> {
    let mut claims = vec![Fq::default(); NATIVE_FINAL_INPUT_LEN];
    claims[NATIVE_FINAL_GT_C_START..NATIVE_FINAL_GT_C_START + GT_ARTIFACT_COEFFS]
        .copy_from_slice(&gt_artifact_coefficients(&state.c));
    claims[NATIVE_FINAL_D1_START..NATIVE_FINAL_D1_START + GT_ARTIFACT_COEFFS]
        .copy_from_slice(&gt_artifact_coefficients(&state.d1));
    claims[NATIVE_FINAL_D2_START..NATIVE_FINAL_D2_START + GT_ARTIFACT_COEFFS]
        .copy_from_slice(&gt_artifact_coefficients(&state.d2));
    claims[NATIVE_FINAL_E1_START..NATIVE_FINAL_E1_START + G1_ARTIFACT_COORDS]
        .copy_from_slice(&g1_artifact_coordinates(state.e1));
    claims[NATIVE_FINAL_E2_START..NATIVE_FINAL_E2_START + G2_ARTIFACT_COORDS]
        .copy_from_slice(&g2_artifact_coordinates(state.e2));
    claims[NATIVE_FINAL_E1_INIT_START..NATIVE_FINAL_E1_INIT_START + G1_ARTIFACT_COORDS]
        .copy_from_slice(&g1_artifact_coordinates(state.e1_init));
    claims[NATIVE_FINAL_D2_INIT_START..NATIVE_FINAL_D2_INIT_START + GT_ARTIFACT_COEFFS]
        .copy_from_slice(&gt_artifact_coefficients(&state.d2_init));
    claims[NATIVE_FINAL_S1_ACC_INDEX] = inject_fr_to_fq(state.s1_acc);
    claims[NATIVE_FINAL_S2_ACC_INDEX] = inject_fr_to_fq(state.s2_acc);
    claims
}

fn native_final_input_state(
    native_final_inputs: &[Fq],
) -> Result<DoryReduceNativeState, DoryAssistVerifierError> {
    let actual_len = native_final_inputs.len();
    if actual_len != NATIVE_FINAL_INPUT_LEN {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "claims.stage1.public.native_final.inputs",
            reason: format!(
                "native final input claim vector has length {actual_len}, expected {NATIVE_FINAL_INPUT_LEN}"
            ),
        });
    }

    Ok(DoryReduceNativeState {
        c: decode_native_final_gt(native_final_inputs, NATIVE_FINAL_GT_C_START, "C_acc")?,
        d1: decode_native_final_gt(native_final_inputs, NATIVE_FINAL_D1_START, "D1_acc")?,
        d2: decode_native_final_gt(native_final_inputs, NATIVE_FINAL_D2_START, "D2_acc")?,
        e1: decode_native_final_g1(native_final_inputs, NATIVE_FINAL_E1_START, "E1_acc")?,
        e2: decode_native_final_g2(native_final_inputs, NATIVE_FINAL_E2_START, "E2_acc")?,
        e1_init: decode_native_final_g1(
            native_final_inputs,
            NATIVE_FINAL_E1_INIT_START,
            "E1_init",
        )?,
        d2_init: decode_native_final_gt(
            native_final_inputs,
            NATIVE_FINAL_D2_INIT_START,
            "D2_init",
        )?,
        s1_acc: decode_native_final_fr(native_final_inputs, NATIVE_FINAL_S1_ACC_INDEX, "s1_acc")?,
        s2_acc: decode_native_final_fr(native_final_inputs, NATIVE_FINAL_S2_ACC_INDEX, "s2_acc")?,
    })
}

fn decode_native_final_gt(
    inputs: &[Fq],
    start: usize,
    label: &'static str,
) -> Result<Bn254GT, DoryAssistVerifierError> {
    let mut coefficients = [Fq::default(); Bn254GT::FQ12_COEFFICIENTS];
    coefficients.copy_from_slice(&inputs[start..start + Bn254GT::FQ12_COEFFICIENTS]);

    for (offset, value) in inputs[start + Bn254GT::FQ12_COEFFICIENTS..start + GT_ARTIFACT_COEFFS]
        .iter()
        .enumerate()
    {
        if *value != Fq::default() {
            return Err(invalid_final_shape(format!(
                "{label} GT padding slot {} must be zero",
                Bn254GT::FQ12_COEFFICIENTS + offset
            )));
        }
    }

    Bn254GT::from_fq12_coefficients(coefficients)
        .ok_or_else(|| invalid_final_shape(format!("{label} is not a valid BN254 GT element")))
}

fn decode_native_final_g1(
    inputs: &[Fq],
    start: usize,
    label: &'static str,
) -> Result<Bn254G1, DoryAssistVerifierError> {
    let mut coordinates = [Fq::default(); G1_ARTIFACT_COORDS];
    coordinates.copy_from_slice(&inputs[start..start + G1_ARTIFACT_COORDS]);

    Bn254G1::from_affine_coordinates_with_infinity(coordinates)
        .ok_or_else(|| invalid_final_shape(format!("{label} is not a valid BN254 G1 point")))
}

fn decode_native_final_g2(
    inputs: &[Fq],
    start: usize,
    label: &'static str,
) -> Result<Bn254G2, DoryAssistVerifierError> {
    let mut coordinates = [Fq::default(); G2_ARTIFACT_COORDS];
    coordinates.copy_from_slice(&inputs[start..start + G2_ARTIFACT_COORDS]);

    Bn254G2::from_affine_coordinates_with_infinity(coordinates)
        .ok_or_else(|| invalid_final_shape(format!("{label} is not a valid BN254 G2 point")))
}

fn decode_native_final_fr(
    inputs: &[Fq],
    index: usize,
    label: &'static str,
) -> Result<Fr, DoryAssistVerifierError> {
    let value = inputs[index];
    let mut bytes = [0_u8; Fq::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    let scalar = Fr::from_le_bytes_mod_order(&bytes);
    if inject_fr_to_fq(scalar) != value {
        return Err(invalid_final_shape(format!(
            "{label} is not a canonical injected BN254 Fr scalar"
        )));
    }

    Ok(scalar)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DoryReduceNativeState {
    c: Bn254GT,
    d1: Bn254GT,
    d2: Bn254GT,
    e1: Bn254G1,
    e2: Bn254G2,
    e1_init: Bn254G1,
    d2_init: Bn254GT,
    s1_acc: Fr,
    s2_acc: Fr,
}

impl DoryReduceNativeState {
    fn process_round(
        &mut self,
        artifacts: &DoryReduceRoundArtifacts,
        setup: &jolt_dory::DoryVerifierSetupArtifacts,
        setup_index: usize,
        scalars: DoryReduceNativeScalars,
    ) -> Result<(), DoryAssistVerifierError> {
        self.c = self.c
            + setup_gt(&setup.chi, setup_index, "chi")?
            + self.d2.scalar_mul(&scalars.beta)
            + self.d1.scalar_mul(&scalars.beta_inverse)
            + artifacts.second.c_plus.scalar_mul(&scalars.alpha)
            + artifacts.second.c_minus.scalar_mul(&scalars.alpha_inverse);

        self.d1 = artifacts.first.d1_left.scalar_mul(&scalars.alpha)
            + artifacts.first.d1_right
            + setup_gt(&setup.delta_1l, setup_index, "delta_1l")?.scalar_mul(&scalars.alpha_beta)
            + setup_gt(&setup.delta_1r, setup_index, "delta_1r")?.scalar_mul(&scalars.beta);

        self.d2 = artifacts.first.d2_left.scalar_mul(&scalars.alpha_inverse)
            + artifacts.first.d2_right
            + setup_gt(&setup.delta_2l, setup_index, "delta_2l")?
                .scalar_mul(&scalars.alpha_inverse_beta_inverse)
            + setup_gt(&setup.delta_2r, setup_index, "delta_2r")?.scalar_mul(&scalars.beta_inverse);

        self.e1 = self.e1
            + artifacts.first.e1_beta.scalar_mul(&scalars.beta)
            + artifacts.second.e1_plus.scalar_mul(&scalars.alpha)
            + artifacts.second.e1_minus.scalar_mul(&scalars.alpha_inverse);

        self.e2 = self.e2
            + artifacts.first.e2_beta.scalar_mul(&scalars.beta_inverse)
            + artifacts.second.e2_plus.scalar_mul(&scalars.alpha)
            + artifacts.second.e2_minus.scalar_mul(&scalars.alpha_inverse);

        self.s1_acc *= scalars.s1_fold_factor;
        self.s2_acc *= scalars.s2_fold_factor;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DoryReduceNativeScalars {
    beta: Fr,
    beta_inverse: Fr,
    alpha: Fr,
    alpha_inverse: Fr,
    alpha_beta: Fr,
    alpha_inverse_beta_inverse: Fr,
    s1_fold_factor: Fr,
    s2_fold_factor: Fr,
}

fn setup_gt(
    values: &[Bn254GT],
    index: usize,
    name: &'static str,
) -> Result<Bn254GT, DoryAssistVerifierError> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| invalid_final_shape(format!("verifier setup {name} missing index {index}")))
}

fn invalid_final_shape(reason: impl Into<String>) -> DoryAssistVerifierError {
    DoryAssistVerifierError::InvalidProofShape {
        component: "dory_final",
        reason: reason.into(),
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests fail loudly on invalid fixtures")]
mod tests {
    use jolt_crypto::PairingGroup;
    use jolt_dory::{DoryScheme, DoryVerifierSetup};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
    use jolt_poly::Polynomial;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use super::*;

    #[test]
    fn transparent_final_pairing_check_matches_dory_verifier_equation() {
        let fixture = final_check_fixture();
        let scalars = fixture
            .proof
            .verifier_transcript_scalars(&fixture.verifier_transcript, &fixture.point);
        let native_final_inputs =
            transparent_native_final_input_claims(&fixture.clear_statement(), &scalars)
                .expect("transparent native-final inputs are well shaped");
        let check = transparent_final_pairing_check(
            &fixture.clear_statement(),
            &scalars,
            &native_final_inputs,
        )
        .expect("transparent final check is well shaped");
        let replayed =
            transparent_replayed_final_pairing_check(&fixture.clear_statement(), &scalars)
                .expect("transparent replayed final check is well shaped");

        assert_eq!(check, replayed);
        assert_eq!(
            check.pre_final_exponentiation().final_exponentiation(),
            Some(check.rhs)
        );
        assert_eq!(
            Bn254::multi_pairing(&check.g1_terms, &check.g2_terms),
            check.rhs
        );
    }

    #[test]
    fn zk_final_pairing_check_matches_dory_verifier_equation() {
        let fixture = zk_final_check_fixture();
        let scalars = fixture
            .proof
            .verifier_transcript_scalars(&fixture.verifier_transcript, &fixture.point);
        let native_final_inputs = zk_native_final_input_claims(&fixture.zk_statement(), &scalars)
            .expect("ZK native-final inputs are well shaped");
        let check = zk_final_pairing_check(&fixture.zk_statement(), &scalars, &native_final_inputs)
            .expect("ZK final check");
        let replayed = zk_replayed_final_pairing_check(&fixture.zk_statement(), &scalars)
            .expect("ZK replayed final check");

        assert!(scalars.scalar_product_sigma_c.is_some());
        assert_eq!(check, replayed);
        assert_eq!(
            check.pre_final_exponentiation().final_exponentiation(),
            Some(check.rhs)
        );
        assert_eq!(Bn254::pairing(&check.g1_term, &check.g2_term), check.rhs);
    }

    struct FinalCheckFixture {
        verifier_setup: DoryVerifierSetup,
        proof: jolt_dory::DoryProof,
        commitment: jolt_dory::DoryCommitment,
        point: Vec<Fr>,
        eval: Fr,
        verifier_transcript: Blake2bTranscript,
    }

    impl FinalCheckFixture {
        fn clear_statement(&self) -> ClearOpeningStatement<'_> {
            ClearOpeningStatement {
                setup: &self.verifier_setup,
                pcs_proof: &self.proof,
                commitment: &self.commitment,
                point: &self.point,
                eval: self.eval,
            }
        }

        fn zk_statement(&self) -> ZkOpeningStatement<'_> {
            ZkOpeningStatement {
                setup: &self.verifier_setup,
                pcs_proof: &self.proof,
                commitment: &self.commitment,
                point: &self.point,
            }
        }
    }

    fn final_check_fixture() -> FinalCheckFixture {
        let (prover_setup, verifier_setup) = DoryScheme::setup(2);
        let poly = Polynomial::<Fr>::from(vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ]);
        let point = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        let mut prover_transcript = Blake2bTranscript::new(b"dory-assist-native-final-fixture");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );
        let verifier_transcript = Blake2bTranscript::new(b"dory-assist-native-final-fixture");

        FinalCheckFixture {
            verifier_setup,
            proof,
            commitment,
            point,
            eval,
            verifier_transcript,
        }
    }

    fn zk_final_check_fixture() -> FinalCheckFixture {
        let (prover_setup, verifier_setup) = DoryScheme::setup(2);
        let poly = Polynomial::<Fr>::from(vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ]);
        let point = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let eval = poly.evaluate(&point);
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);
        let mut prover_transcript = Blake2bTranscript::new(b"dory-assist-native-final-zk");
        let (proof, _hiding_commitment, _blind) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            hint,
            &mut prover_transcript,
        );
        let verifier_transcript = Blake2bTranscript::new(b"dory-assist-native-final-zk");

        FinalCheckFixture {
            verifier_setup,
            proof,
            commitment,
            point,
            eval,
            verifier_transcript,
        }
    }
}
