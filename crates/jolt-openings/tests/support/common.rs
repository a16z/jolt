use jolt_crypto::Bn254;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

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
        let (commitment, hint) = PCS::commit(polynomial, setup);
        claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
        });
        hints.push(hint);
    }
    (claims, hints)
}

pub fn kzg_setup(
    max_num_vars: usize,
) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let prover = HyperKZGScheme::<Bn254>::setup(
        &mut rng,
        1usize << max_num_vars,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier = HyperKZGScheme::<Bn254>::verifier_setup(&prover);
    (prover, verifier)
}
