use jolt_field::{Field, Fr, Ring};
use jolt_openings::{CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

pub fn protocol() -> ProtocolId {
    ProtocolId::new::<Blake2b512>("jolt-openings/tests")
}

pub fn prover(session: &[u8]) -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&protocol(), session)
}

pub fn verifier<'a>(session: &[u8], narg: &'a [u8]) -> VerifierTranscript<'a, Blake2b512> {
    VerifierTranscript::new(&protocol(), session, narg)
}

/// Sponge state after both sides finish, which must agree for an honest run.
pub fn fingerprint<C: Channel>(channel: &mut C) -> [u8; 32] {
    channel.challenge_bytes::<32>()
}

pub fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

pub fn sources(polynomials: &[Polynomial<Fr>]) -> Vec<&dyn MultilinearPoly<Fr>> {
    polynomials
        .iter()
        .map(|polynomial| polynomial as &dyn MultilinearPoly<Fr>)
        .collect()
}

pub fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Point<HIGH_TO_LOW, Fr> {
    Point::new((0..num_vars).map(|_| Fr::random(rng)).collect::<Vec<_>>())
}

pub fn homomorphic_polynomials(
    count: usize,
    num_vars: usize,
    seed: u64,
) -> (Vec<Polynomial<Fr>>, Point<HIGH_TO_LOW, Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let polynomials = (0..count)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let point = random_point(num_vars, &mut rng);
    (polynomials, point)
}

pub type ClaimsAndHints<PCS> = (
    Vec<VerifierOpeningClaim<Fr, <PCS as jolt_crypto::Commitment>::Output>>,
    Vec<<PCS as CommitmentScheme>::OpeningHint>,
);

/// Commits every polynomial and returns same-point opening claims plus hints.
pub fn clear_claims<PCS: CommitmentScheme<Field = Fr>>(
    polynomials: &[Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
    setup: &PCS::ProverSetup,
) -> ClaimsAndHints<PCS> {
    let mut claims = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        let (commitment, hint) = PCS::commit(polynomial, setup).expect("commit should succeed");
        claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
        });
        hints.push(hint);
    }
    (claims, hints)
}
