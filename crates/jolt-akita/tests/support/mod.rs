use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaNativeBatchPolynomials, AkitaNativeBatchStatement,
    AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams,
};
use jolt_field::Ring;
use jolt_openings::{CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-akita/test");

pub fn new_prover_transcript(session: &[u8]) -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&PROTOCOL, session)
}

pub fn new_verifier_transcript<'a>(
    session: &[u8],
    proof: &'a [u8],
) -> VerifierTranscript<'a, Blake2b512> {
    VerifierTranscript::new(&PROTOCOL, session, proof)
}

/// The verifier consumed exactly the prover's argument string and ended in the
/// same sponge state.
pub fn assert_transcripts_agree(
    mut prover: ProverTranscript<Blake2b512>,
    mut verifier: VerifierTranscript<'_, Blake2b512>,
) {
    assert_eq!(
        prover.challenge_bytes::<32>(),
        verifier.challenge_bytes::<32>()
    );
    verifier
        .finish()
        .expect("the verifier consumes the whole proof");
}

pub fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

pub fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

pub fn polynomial(num_vars: usize, offset: u64) -> Polynomial<AkitaField> {
    let len = 1usize << num_vars;
    Polynomial::new(
        (0..len)
            .map(|index| f(offset + 1 + 3 * index as u64 + (index as u64 % 5)))
            .collect(),
    )
}

pub fn setup_for(
    num_vars: usize,
    max_num_polys_per_commitment_group: usize,
    layout_digest: [u8; 32],
) -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::new(
        num_vars,
        max_num_polys_per_commitment_group,
        layout_digest,
        AkitaScheduleArtifacts::shared_from_default_directory(),
    ))
    .expect("Akita setup should succeed")
}

pub fn native_setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    setup_for(16, 2, layout(7))
}

pub fn native_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    evaluations: impl IntoIterator<Item = AkitaField>,
) -> AkitaNativeBatchStatement {
    evaluations
        .into_iter()
        .map(|evaluation| VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), evaluation),
        })
        .collect()
}

pub fn single_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval: AkitaField,
) -> AkitaNativeBatchStatement {
    native_statement(commitment, point, [eval])
}

pub fn batch_polynomials<'a>(
    polynomials: impl IntoIterator<Item = &'a Polynomial<AkitaField>>,
) -> AkitaNativeBatchPolynomials<'a> {
    polynomials
        .into_iter()
        .map(|polynomial| polynomial as &(dyn MultilinearPoly<AkitaField> + 'a))
        .collect()
}
