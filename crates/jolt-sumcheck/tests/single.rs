#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixed tables"
)]

use jolt_algebra::{
    Ext2, ExtField, Field, Fr, One, Prime128Offset275 as P, Prime64Offset59, Ring, Zero,
};
use jolt_poly::{OmittedConstantPoly, Point, UnivariatePoly};
use jolt_sumcheck::{
    prove_single, verify_single, EqFactoredRound, ProveRounds, SingleFormat, SingleInstance,
    SingleReduction, SingleSumcheckProof, SumcheckDomain, SumcheckError,
};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Blake2b512, ChallengeCodec, Encoding, ProverState,
    TAG_BYTES,
};

type Quadratic = Ext2<Prime64Offset59>;

/// A field the challenge codecs cover, absorbed through its encoding.
trait ChallengeField: Field + Encoding + ChallengeCodec + 'static {}

impl<F: Field + Encoding + ChallengeCodec + 'static> ChallengeField for F {}

fn session() -> [u8; TAG_BYTES] {
    jolt_transcript::protocol_tag(b"single-sumcheck")
}

fn sample_word(seed: u64) -> u64 {
    let mut word = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

fn sample<F: Field, E: ExtField<F>>(seed: u64) -> E {
    E::from_base_fn(|index| F::from_u64(sample_word(seed.wrapping_mul(7) + index as u64)))
}

/// `Σ_x w(x) a(x) b(x)` over the hypercube, binding the lowest index bit
/// first. With `tau`, `w = eq(tau, ·)` and each round returns the full `q`
/// (the eq factor of the bound bit removed); without it, `w = 1`.
struct ProductKernel<E> {
    a: Vec<E>,
    b: Vec<E>,
    tau: Option<Vec<E>>,
    seen_claims: Vec<E>,
}

fn bind<E: Field>(table: &mut Vec<E>, r: E) {
    let half = table.len() / 2;
    for j in 0..half {
        table[j] = table[2 * j] + r * (table[2 * j + 1] - table[2 * j]);
    }
    table.truncate(half);
}

/// `eq(point, x)` with `x`'s bit `k` paired with `point[k]`.
fn eq<E: Field>(point: &[E], x: usize) -> E {
    point
        .iter()
        .enumerate()
        .map(|(k, &p)| if x >> k & 1 == 1 { p } else { E::one() - p })
        .product()
}

/// The multilinear extension of `table` at `point`, by its definition.
fn mle<E: Field>(table: &[E], point: &[E]) -> E {
    table
        .iter()
        .enumerate()
        .map(|(x, &value)| eq(point, x) * value)
        .sum()
}

impl<E: Field> ProductKernel<E> {
    fn sampled(rounds: usize, seed: u64, eq_factored: bool, draw: impl Fn(u64) -> E) -> Self {
        let table = |offset: u64| {
            (0..1u64 << rounds)
                .map(|x| draw(seed + offset + x))
                .collect()
        };
        Self {
            a: table(0),
            b: table(1 << 20),
            tau: eq_factored.then(|| {
                (0..rounds as u64)
                    .map(|k| draw(seed + (2 << 20) + k))
                    .collect()
            }),
            seen_claims: Vec::new(),
        }
    }

    fn claim(&self) -> E {
        (0..self.a.len())
            .map(|x| {
                let weight = self.tau.as_ref().map_or(E::one(), |tau| eq(tau, x));
                weight * self.a[x] * self.b[x]
            })
            .sum()
    }
}

impl<E: Field> ProveRounds<E> for ProductKernel<E> {
    fn num_rounds(&self) -> usize {
        self.a.len().trailing_zeros() as usize
    }

    fn prove_round(
        &mut self,
        bind_challenge: Option<E>,
        round: usize,
        previous_claim: E,
    ) -> Result<UnivariatePoly<E>, SumcheckError<E>> {
        if let Some(r) = bind_challenge {
            bind(&mut self.a, r);
            bind(&mut self.b, r);
        }
        assert_eq!(round, self.seen_claims.len());
        self.seen_claims.push(previous_claim);
        let mut coefficients = vec![E::zero(); 3];
        for j in 0..self.a.len() / 2 {
            let weight = self
                .tau
                .as_ref()
                .map_or(E::one(), |tau| eq(&tau[round + 1..], j));
            let (a0, a1) = (self.a[2 * j], self.a[2 * j + 1] - self.a[2 * j]);
            let (b0, b1) = (self.b[2 * j], self.b[2 * j + 1] - self.b[2 * j]);
            coefficients[0] += weight * a0 * b0;
            coefficients[1] += weight * (a0 * b1 + a1 * b0);
            coefficients[2] += weight * a1 * b1;
        }
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind_challenge: E) -> Result<(), SumcheckError<E>> {
        bind(&mut self.a, bind_challenge);
        bind(&mut self.b, bind_challenge);
        Ok(())
    }
}

fn instance<E: Field>(kernel: &ProductKernel<E>) -> SingleInstance<E> {
    SingleInstance {
        input_claim: kernel.claim(),
        rounds: kernel.num_rounds(),
        degree: 2,
        format: kernel.tau.clone().map_or(SingleFormat::Compressed, |tau| {
            SingleFormat::EqFactored {
                equality_point: Point::low_to_high(tau),
            }
        }),
    }
}

fn prove<E: ChallengeField>(
    kernel: &mut ProductKernel<E>,
    instance: &SingleInstance<E>,
) -> (SingleSumcheckProof<E>, SingleReduction<E>, [u8; 32]) {
    let mut transcript = prover_transcript::<Blake2b512>(&session(), b"");
    let (proof, reduction) = prove_single(kernel, instance, &mut transcript).unwrap();
    (proof, reduction, transcript.verifier_message::<[u8; 32]>())
}

fn replay<E: ChallengeField>(
    instance: &SingleInstance<E>,
    proof: &SingleSumcheckProof<E>,
) -> Result<(SingleReduction<E>, [u8; 32]), SumcheckError<E>> {
    let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
    let reduction = verify_single(instance, proof, &mut transcript)?;
    Ok((reduction, transcript.verifier_message::<[u8; 32]>()))
}

/// The terminal oracle check of the product relation: the normalized claim
/// is `a(r) b(r)`, the eq factors `Π eq(τ_i, r_i)` being the caller's.
fn oracle_accepts<E: Field>(tables: &(Vec<E>, Vec<E>), reduction: &SingleReduction<E>) -> bool {
    reduction.output_claim
        == mle(&tables.0, &reduction.challenges) * mle(&tables.1, &reduction.challenges)
}

fn round_trip<F: Field, E: ChallengeField + ExtField<F>>(eq_factored: bool, seed: u64) {
    let mut kernel = ProductKernel::<E>::sampled(4, seed, eq_factored, sample::<F, E>);
    let tables = (kernel.a.clone(), kernel.b.clone());
    let instance = instance(&kernel);
    let (proof, proved, prover_state) = prove(&mut kernel, &instance);
    assert_eq!(proof.rounds.len(), 4);
    assert!(proof.rounds.iter().all(|wire| wire.len() == 2));

    let (replayed, verifier_state) = replay(&instance, &proof).unwrap();
    assert_eq!(replayed, proved);
    assert_eq!(verifier_state, prover_state);
    assert!(oracle_accepts(&tables, &replayed));
}

#[test]
fn honest_compressed_and_eq_factored_proofs_verify() {
    for eq_factored in [false, true] {
        round_trip::<P, P>(eq_factored, 11);
        round_trip::<Prime64Offset59, Quadratic>(eq_factored, 12);
    }
}

#[test]
fn eq_factored_round_check_and_recovery_match_direct_evaluation() {
    type E = Quadratic;
    for seed in 0..8u64 {
        let q = UnivariatePoly::new(
            (0..4)
                .map(|k| sample::<Prime64Offset59, E>(100 * seed + k))
                .collect(),
        );
        let tau = sample::<Prime64Offset59, E>(100 * seed + 50);
        let point = sample::<Prime64Offset59, E>(100 * seed + 51);
        let claim = (E::one() - tau) * q.evaluate(E::zero()) + tau * q.evaluate(E::one());

        let round = EqFactoredRound { tau };
        round.check_round_sum(0, claim, &q).unwrap();
        assert!(matches!(
            round.check_round_sum(0, claim + E::one(), &q),
            Err(SumcheckError::RoundCheckFailed { .. })
        ));

        let omitted = OmittedConstantPoly::from_q_coefficients(q.coefficients().to_vec());
        assert_eq!(omitted.constant_from_claim(claim, tau), q.coefficients()[0]);
        assert_eq!(
            omitted.evaluate_from_claim(claim, tau, point),
            q.evaluate(point)
        );
    }
}

#[test]
fn eq_factored_claims_stay_normalized_across_rounds() {
    // The running claim entering round i is the remaining eq-weighted sum
    // with the bound variables fixed, without the bound rounds' eq factors.
    type E = Quadratic;
    let rounds = 5;
    let mut kernel = ProductKernel::<E>::sampled(rounds, 21, true, sample::<Prime64Offset59, E>);
    let tables = (kernel.a.clone(), kernel.b.clone());
    let tau = kernel.tau.clone().unwrap();
    let instance = instance(&kernel);
    let (proof, proved, _) = prove(&mut kernel, &instance);
    let (replayed, _) = replay(&instance, &proof).unwrap();
    let point = proved.challenges;

    let normalized = |round: usize| -> E {
        (0..1usize << (rounds - round))
            .map(|rest| {
                let mut full_point = point[..round].to_vec();
                full_point.extend((0..rounds - round).map(|k| {
                    if rest >> k & 1 == 1 {
                        E::one()
                    } else {
                        E::zero()
                    }
                }));
                eq(&tau[round..], rest) * mle(&tables.0, &full_point) * mle(&tables.1, &full_point)
            })
            .sum()
    };
    for (round, &claim) in kernel.seen_claims.iter().enumerate() {
        assert_eq!(claim, normalized(round), "round {round}");
    }
    assert_eq!(replayed.output_claim, normalized(rounds));
}

/// BN254 challenges need no proof of work, so a tampered proof replays to a
/// reduction and the oracle check is what rejects it.
#[test]
fn tampered_coefficient_or_claim_fails_the_oracle_check() {
    let draw = |seed: u64| Fr::from_u64(sample_word(seed));
    for eq_factored in [false, true] {
        let mut kernel = ProductKernel::<Fr>::sampled(3, 31, eq_factored, draw);
        let tables = (kernel.a.clone(), kernel.b.clone());
        let instance = instance(&kernel);
        let (proof, _, _) = prove(&mut kernel, &instance);
        assert_eq!(proof.pow_nonces, []);

        for round in 0..3 {
            for coefficient in 0..2 {
                let mut tampered = proof.clone();
                tampered.rounds[round][coefficient] += Fr::one();
                let (replayed, _) = replay(&instance, &tampered).unwrap();
                assert!(!oracle_accepts(&tables, &replayed));
            }
        }

        let mut wrong = instance.clone();
        wrong.input_claim += Fr::one();
        let (replayed, _) = replay(&wrong, &proof).unwrap();
        assert!(!oracle_accepts(&tables, &replayed));

        let mut kernel = ProductKernel::<Fr>::sampled(3, 31, eq_factored, draw);
        let mut transcript = prover_transcript::<Blake2b512>(&session(), b"");
        assert!(matches!(
            prove_single(&mut kernel, &wrong, &mut transcript),
            Err(SumcheckError::RoundCheckFailed { round: 0, .. })
        ));
    }
}

#[test]
fn malformed_proof_shapes_are_rejected() {
    let mut kernel = ProductKernel::<P>::sampled(3, 41, true, sample::<P, P>);
    let instance = instance(&kernel);
    let (proof, _, _) = prove(&mut kernel, &instance);

    let mut short = proof.clone();
    let _ = short.rounds[1].pop();
    assert!(matches!(
        replay(&instance, &short),
        Err(SumcheckError::WrongRoundWidth {
            round: 1,
            expected: 2,
            got: 1
        })
    ));

    let mut missing = proof.clone();
    let _ = missing.rounds.pop();
    assert!(matches!(
        replay(&instance, &missing),
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 3,
            got: 2
        })
    ));

    let mut dropped = proof;
    assert!(!dropped.pow_nonces.is_empty(), "a 128-bit field grinds");
    let _ = dropped.pow_nonces.pop();
    assert!(matches!(
        replay(&instance, &dropped),
        Err(SumcheckError::WrongNumberOfPowNonces { .. })
    ));
}

#[test]
fn invalid_shapes_are_rejected_before_any_round() {
    let base = SingleInstance {
        input_claim: P::from_u64(17),
        rounds: 2,
        degree: 2,
        format: SingleFormat::EqFactored {
            equality_point: Point::low_to_high(vec![P::from_u64(3); 2]),
        },
    };
    let empty = SingleSumcheckProof::default();
    let shape_error = |instance: &SingleInstance<P>| replay(instance, &empty).unwrap_err();

    let mut zero_degree = base.clone();
    zero_degree.degree = 0;
    assert!(matches!(
        shape_error(&zero_degree),
        SumcheckError::ZeroDegreeBound
    ));

    let mut short_point = base.clone();
    short_point.format = SingleFormat::EqFactored {
        equality_point: Point::low_to_high(vec![P::from_u64(3)]),
    };
    assert!(matches!(
        shape_error(&short_point),
        SumcheckError::EqualityPointLengthMismatch {
            expected: 2,
            got: 1
        }
    ));

    let mut no_rounds = base.clone();
    no_rounds.rounds = 0;
    no_rounds.format = SingleFormat::EqFactored {
        equality_point: Point::low_to_high(Vec::new()),
    };
    assert!(matches!(
        shape_error(&no_rounds),
        SumcheckError::EqFactoredZeroRounds
    ));

    let mut overflow = base;
    overflow.rounds = usize::MAX;
    overflow.format = SingleFormat::Compressed;
    assert!(matches!(
        shape_error(&overflow),
        SumcheckError::ShapeOverflow { .. }
    ));
}

#[test]
fn zero_round_compressed_instance_reduces_to_its_input_claim() {
    let instance = SingleInstance {
        input_claim: P::from_u64(17),
        rounds: 0,
        degree: 2,
        format: SingleFormat::Compressed,
    };
    let mut kernel = ProductKernel::<P>::sampled(0, 51, false, sample::<P, P>);
    let (proof, proved, prover_state) = prove(&mut kernel, &instance);
    assert_eq!(proof, SingleSumcheckProof::default());
    assert_eq!(proved.challenges, []);
    let (replayed, verifier_state) = replay(&instance, &proof).unwrap();
    assert_eq!(replayed.output_claim, instance.input_claim);
    assert_eq!(verifier_state, prover_state);
}

/// A kernel round above the instance's degree is rejected before the round
/// is absorbed.
#[test]
fn prover_rejects_a_round_above_the_degree() {
    let mut kernel = ProductKernel::<P>::sampled(2, 61, false, sample::<P, P>);
    let mut instance = instance(&kernel);
    instance.degree = 1;
    let mut transcript: ProverState<Blake2b512> = prover_transcript(&session(), b"");
    assert!(matches!(
        prove_single(&mut kernel, &instance, &mut transcript),
        Err(SumcheckError::DegreeBoundExceeded { got: 2, max: 1 })
    ));
}
