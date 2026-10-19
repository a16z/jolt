#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use std::vec::IntoIter;

use jolt_field::{Field, Fr, JoltField, One, Ring, F128, F64};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, send_compressed_round, send_full_round, BatchMember,
    BatchPrelude, BooleanHypercube, CenteredIntegerDomain, ClearSumcheckRecorder, EvaluationClaim,
    ProveRounds, ProvedBatch, SequentialRounds, SumcheckClaim, SumcheckDomain, SumcheckDomainSpec,
    SumcheckError, SumcheckRecorder, SumcheckVerifier,
};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, Sponge, VerifierTranscript,
};

const LABEL: &[u8] = b"binary-sumcheck";

trait BinaryField: JoltField {
    fn sample(seed: u64) -> Self;
}

fn sample_word(seed: u64) -> u64 {
    let mut word = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

impl BinaryField for F128 {
    fn sample(seed: u64) -> Self {
        Self::from_raw(
            u128::from(sample_word(seed)) | (u128::from(sample_word(seed.wrapping_add(1))) << 64),
        )
    }
}

impl BinaryField for F64 {
    fn sample(seed: u64) -> Self {
        Self::from_raw(sample_word(seed))
    }
}

#[derive(Clone)]
struct DenseProduct<F> {
    factors: Vec<Vec<F>>,
    rounds: usize,
}

impl<F: BinaryField> DenseProduct<F> {
    fn sampled(rounds: usize, degree: usize, seed: u64) -> Self {
        Self {
            factors: (0..degree)
                .map(|factor| {
                    (0..1 << rounds)
                        .map(|index| F::sample(seed + 37 * factor as u64 + index as u64))
                        .collect()
                })
                .collect(),
            rounds,
        }
    }
}

impl<F: JoltField> DenseProduct<F> {
    fn claimed_sum(&self) -> F {
        (0..self.factors[0].len())
            .map(|index| {
                self.factors
                    .iter()
                    .map(|factor| factor[index])
                    .product::<F>()
            })
            .sum()
    }

    fn evaluate(&self, point: &[F]) -> F {
        self.factors
            .iter()
            .map(|factor| Polynomial::new(factor.clone()).evaluate_and_consume(point))
            .product()
    }

    fn bind(&mut self, challenge: F) {
        for factor in &mut self.factors {
            let half = factor.len() / 2;
            for index in 0..half {
                let low = factor[index];
                let high = factor[index + half];
                factor[index] = low + challenge * (high - low);
            }
            factor.truncate(half);
        }
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.factors[0].len() / 2;
        let mut sum = vec![F::zero(); self.factors.len() + 1];
        for index in 0..half {
            let mut product = vec![F::one()];
            for factor in &self.factors {
                let constant = factor[index];
                let linear = factor[index + half] - constant;
                let mut next = vec![F::zero(); product.len() + 1];
                for (power, coefficient) in product.into_iter().enumerate() {
                    next[power] += coefficient * constant;
                    next[power + 1] += coefficient * linear;
                }
                product = next;
            }
            for (coefficient, term) in sum.iter_mut().zip(product) {
                *coefficient += term;
            }
        }
        UnivariatePoly::new(sum)
    }
}

impl<F: JoltField> ProveRounds<F> for DenseProduct<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        assert_eq!(bind.is_none(), round == 0);
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        assert_eq!(self.factors[0].len(), 1 << (self.rounds - round));
        assert_eq!(self.claimed_sum(), previous_claim);
        Ok(self.round_polynomial())
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        assert!(self.factors.iter().all(|factor| factor.len() == 1));
        Ok(())
    }
}

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck/binary-fields");

fn prover() -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&PROTOCOL, LABEL)
}

fn verifier(narg: &[u8]) -> VerifierTranscript<'_, Blake2b512> {
    VerifierTranscript::new(&PROTOCOL, LABEL, narg)
}

/// The sponge state both roles must agree on after the same message sequence.
fn fingerprint<C: Channel>(channel: &mut C) -> [u8; 32] {
    channel.challenge_bytes::<32>()
}

/// A clear recorder whose round challenges come from a fixed script instead of
/// the transcript, to reach challenge values a sponge would draw only by
/// chance. Its argument string has no verifier counterpart.
struct ScriptedRecorder<F> {
    script: IntoIter<F>,
}

impl<F: BinaryField> SumcheckRecorder<F> for ScriptedRecorder<F> {
    type Witness = ();

    fn absorb_input_claims<C: Channel>(&mut self, input_claims: &[F], channel: &mut C) {
        channel.public_all(input_claims);
    }

    fn absorb_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F>> {
        send_compressed_round(round_poly, degree, transcript)?;
        Ok(self.script.next().unwrap())
    }

    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<F>> {
        transcript.send_all(output_claim_values);
        Ok(())
    }
}

/// How a fixture draws its batching coefficients and round challenges.
enum Challenges<F> {
    Transcript,
    Scripted {
        coefficients: Vec<F>,
        rounds: Vec<F>,
    },
}

struct BatchFixture<F: Field> {
    prelude: BatchPrelude<F>,
    proved: ProvedBatch<F>,
    narg: Vec<u8>,
    fingerprint: [u8; 32],
}

impl<F: BinaryField> BatchFixture<F> {
    fn prove(
        original: &[DenseProduct<F>],
        offsets: &[usize],
        max_num_vars: usize,
        max_degree: usize,
        challenges: Challenges<F>,
    ) -> Self {
        let mut transcript = prover();
        let claims: Vec<_> = original.iter().map(DenseProduct::claimed_sum).collect();
        transcript.public_all(&claims);
        let (mut coefficients, scripted_rounds) = match challenges {
            Challenges::Transcript => (None, None),
            Challenges::Scripted {
                coefficients,
                rounds,
            } => (Some(coefficients.into_iter()), Some(rounds)),
        };
        let members = original
            .iter()
            .zip(offsets)
            .zip(claims)
            .map(|((member, &offset), input_claim)| BatchMember {
                input_claim,
                coefficient: coefficients
                    .as_mut()
                    .map_or_else(|| transcript.challenge(), |script| script.next().unwrap()),
                rounds: member.rounds,
                offset,
            })
            .collect();
        let prelude = BatchPrelude::try_new(members, max_num_vars, max_degree).unwrap();
        let expected_sum: F = prelude
            .members
            .iter()
            .map(|member| member.coefficient * member.input_claim)
            .sum();
        assert_eq!(prelude.claimed_sum, expected_sum);
        let mut products = original.to_vec();
        let mut members: Vec<&mut dyn ProveRounds<F>> = products
            .iter_mut()
            .map(|member| member as &mut dyn ProveRounds<F>)
            .collect();
        let proved = if let Some(rounds) = scripted_rounds {
            let mut recorder = ScriptedRecorder {
                script: rounds.into_iter(),
            };
            let proved = prove_batch(
                &prelude,
                &mut members,
                &mut SequentialRounds,
                &mut recorder,
                &mut transcript,
            )
            .unwrap();
            recorder
                .finish(&proved.member_claims, &mut transcript)
                .unwrap();
            proved
        } else {
            let mut recorder = ClearSumcheckRecorder::<F>::new();
            let proved = prove_batch(
                &prelude,
                &mut members,
                &mut SequentialRounds,
                &mut recorder,
                &mut transcript,
            )
            .unwrap();
            recorder
                .finish(&proved.member_claims, &mut transcript)
                .unwrap();
            proved
        };
        let narg = transcript.narg().to_vec();
        Self {
            prelude,
            proved,
            narg,
            fingerprint: fingerprint(&mut transcript),
        }
    }

    fn replay_head(&self, transcript: &mut VerifierTranscript<'_, Blake2b512>) {
        let claims: Vec<F> = self
            .prelude
            .members
            .iter()
            .map(|member| member.input_claim)
            .collect();
        transcript.public_all(&claims);
        for member in &self.prelude.members {
            assert_eq!(transcript.challenge::<F>(), member.coefficient);
        }
    }

    fn reduce(&self, narg: &[u8]) -> Result<EvaluationClaim<F>, SumcheckError<F>> {
        let mut transcript = verifier(narg);
        self.replay_head(&mut transcript);
        SumcheckVerifier::verify_compressed(
            &SumcheckClaim::new(
                self.prelude.max_num_vars,
                self.prelude.max_degree,
                self.prelude.claimed_sum,
            ),
            &mut transcript,
        )
    }

    fn verify(&self) -> EvaluationClaim<F> {
        let mut transcript = verifier(&self.narg);
        self.replay_head(&mut transcript);
        let reduced = SumcheckVerifier::verify_compressed(
            &SumcheckClaim::new(
                self.prelude.max_num_vars,
                self.prelude.max_degree,
                self.prelude.claimed_sum,
            ),
            &mut transcript,
        )
        .unwrap();
        let claims: Vec<F> = transcript
            .receive_n(self.proved.member_claims.len())
            .unwrap();
        assert_eq!(claims, self.proved.member_claims);
        assert_eq!(reduced.point.as_slice(), self.proved.challenges);
        assert_eq!(reduced.value, self.proved.final_claim);
        assert_eq!(fingerprint(&mut transcript), self.fingerprint);
        transcript.finish().unwrap();
        reduced
    }

    fn check_output_scales(&self, original: &[DenseProduct<F>]) {
        let point = &self.proved.challenges;
        for (index, (member, product)) in self.prelude.members.iter().zip(original).enumerate() {
            let native = product.evaluate(&point[member.offset..member.offset + member.rounds]);
            assert_eq!(
                self.proved.member_claims[index],
                self.prelude.member_output_scale(index, point).unwrap() * native,
            );
        }
        assert_eq!(
            self.proved.final_claim,
            self.prelude
                .members
                .iter()
                .zip(&self.proved.member_claims)
                .map(|(member, &claim)| member.coefficient * claim)
                .sum::<F>()
        );
    }
}

fn single_members<F: BinaryField>() {
    for degree in 1..=3 {
        let products = [DenseProduct::<F>::sampled(4, degree, 11)];
        let fixture = BatchFixture::prove(&products, &[0], 4, degree, Challenges::Transcript);
        let reduced = fixture.verify();
        assert_eq!(
            fixture.prelude.members[0].input_claim,
            products[0].claimed_sum()
        );
        let expected = products[0].evaluate(reduced.point.as_slice());
        assert_eq!(fixture.proved.member_claims[0], expected);
        let scaled = fixture.prelude.members[0].coefficient * expected;
        assert_eq!(fixture.proved.final_claim, scaled);
        assert_eq!(reduced.value, scaled);
        fixture.check_output_scales(&products);
    }
}

fn mixed_linear_members<F: BinaryField>() {
    let products = [
        DenseProduct::<F>::sampled(4, 1, 100),
        DenseProduct::<F>::sampled(2, 1, 200),
        DenseProduct::<F>::sampled(2, 1, 300),
        DenseProduct::<F>::sampled(2, 1, 400),
    ];
    let fixture = BatchFixture::prove(&products, &[0, 2, 0, 1], 4, 1, Challenges::Transcript);
    let _ = fixture.verify();
    for (member, indices) in [(1, [0, 1, 2, 3]), (2, [0, 4, 8, 12]), (3, [0, 2, 4, 6])] {
        let mut extended = vec![F::zero(); 16];
        for (&value, index) in products[member].factors[0].iter().zip(indices) {
            extended[index] = value;
        }
        assert_eq!(
            fixture.proved.member_claims[member],
            Polynomial::new(extended).evaluate_and_consume(&fixture.proved.challenges),
        );
    }
    fixture.check_output_scales(&products);
}

fn mixed_quadratic_members<F: BinaryField>() {
    let products = [
        DenseProduct::<F>::sampled(4, 2, 100),
        DenseProduct::<F>::sampled(2, 2, 200),
        DenseProduct::<F>::sampled(2, 2, 300),
    ];
    let fixture = BatchFixture::prove(&products, &[0, 2, 1], 4, 2, Challenges::Transcript);
    let _ = fixture.verify();
    let r = &fixture.proved.challenges;
    assert_eq!(
        fixture.proved.member_claims[1],
        products[1].evaluate(&r[2..4]) * (F::one() - r[0]) * (F::one() - r[1])
    );
    assert_eq!(
        fixture.proved.member_claims[2],
        products[2].evaluate(&r[1..3]) * (F::one() - r[0]) * (F::one() - r[3])
    );
    fixture.check_output_scales(&products);
}

fn zero_native_claim<F: BinaryField>() {
    let u = F::sample(79);
    assert!(!u.is_zero());
    let products = [
        DenseProduct::<F>::sampled(2, 1, 100),
        DenseProduct {
            factors: vec![vec![u, u]],
            rounds: 1,
        },
    ];
    let fixture = BatchFixture::prove(&products, &[0, 1], 2, 1, Challenges::Transcript);
    assert_eq!(fixture.prelude.members[1].input_claim, F::zero());
    assert_eq!(
        fixture.proved.member_claims[1],
        u * (F::one() - fixture.proved.challenges[0])
    );
    assert!(!fixture.proved.member_claims[1].is_zero());
    let _ = fixture.verify();
    fixture.check_output_scales(&products);
}

fn zero_padding_factor<F: BinaryField>() {
    for (offset, challenges) in [(1, [F::one(), F::sample(6)]), (0, [F::sample(6), F::one()])] {
        let products = [DenseProduct::<F>::sampled(1, 2, 200)];
        let fixture = BatchFixture::prove(
            &products,
            &[offset],
            2,
            2,
            Challenges::Scripted {
                coefficients: vec![F::sample(19)],
                rounds: challenges.to_vec(),
            },
        );
        assert_eq!(fixture.proved.challenges, challenges);
        assert_eq!(
            fixture.prelude.member_output_scale(0, &challenges).unwrap(),
            F::zero()
        );
        assert_eq!(fixture.proved.member_claims, vec![F::zero()]);
        assert_eq!(fixture.proved.final_claim, F::zero());
        fixture.check_output_scales(&products);
    }
}

fn rejection<F: BinaryField>() {
    let products = [
        DenseProduct::<F>::sampled(4, 2, 31),
        DenseProduct::<F>::sampled(2, 2, 97),
    ];
    let fixture = BatchFixture::prove(&products, &[0, 1], 4, 2, Challenges::Transcript);
    let mut changed_prelude = fixture.prelude.clone();
    changed_prelude.claimed_sum += F::one();
    let mut members = products.clone();
    let mut references: Vec<&mut dyn ProveRounds<F>> = members
        .iter_mut()
        .map(|member| member as &mut dyn ProveRounds<F>)
        .collect();
    assert!(matches!(
        prove_batch(
            &changed_prelude,
            &mut references,
            &mut SequentialRounds,
            &mut ClearSumcheckRecorder::<F>::new(),
            &mut prover()
        ),
        Err(SumcheckError::RoundCheckFailed { round: 0, .. }),
    ));

    // Each compressed degree-2 round carries two coefficients; shift the last
    // round's constant term.
    let mut altered = fixture.narg.clone();
    let last_round = 3 * 2 * F::NUM_BYTES;
    let constant = F::from_bytes_le_checked(&altered[last_round..last_round + F::NUM_BYTES])
        .unwrap()
        + F::one();
    altered[last_round..last_round + F::NUM_BYTES].copy_from_slice(&constant.to_bytes_le_vec());
    let reduced = fixture.reduce(&altered).unwrap();
    let expected: F = fixture
        .prelude
        .members
        .iter()
        .zip(&products)
        .enumerate()
        .map(|(index, (member, product))| {
            member.coefficient
                * fixture
                    .prelude
                    .member_output_scale(index, reduced.point.as_slice())
                    .unwrap()
                * product.evaluate(
                    &reduced.point.as_slice()[member.offset..member.offset + member.rounds],
                )
        })
        .sum();
    assert_ne!(reduced.value, expected);

    let missing_round = &fixture.narg[..last_round];
    assert!(matches!(
        fixture.reduce(missing_round),
        Err(SumcheckError::Transcript(_))
    ));
}

fn uncompressed<F: BinaryField>() {
    let original = DenseProduct::<F>::sampled(4, 3, 27);
    let input_claim = original.claimed_sum();
    let mut member = original.clone();
    let mut transcript = prover();
    let mut previous_claim = input_claim;
    let mut bind = None;
    let mut challenges = Vec::new();
    for round in 0..4 {
        let polynomial = member.prove_round(bind, round, previous_claim).unwrap();
        send_full_round(&polynomial, 3, &mut transcript).unwrap();
        let challenge = transcript.challenge_small();
        previous_claim = polynomial.evaluate(challenge);
        bind = Some(challenge);
        challenges.push(challenge);
    }
    member.finish_rounds(bind.unwrap()).unwrap();
    let narg = transcript.narg().to_vec();
    let mut reader = verifier(&narg);
    let reduced = SumcheckVerifier::verify(
        &SumcheckClaim::new(4, 3, input_claim),
        BooleanHypercube,
        &mut reader,
    )
    .unwrap();
    assert_eq!(reduced.point.as_slice(), challenges);
    assert_eq!(reduced.value, original.evaluate(&challenges));
    assert_eq!(reduced.value, previous_claim);
    assert_eq!(fingerprint(&mut transcript), fingerprint(&mut reader));
}

fn output_scale_errors<F: BinaryField>() {
    let member = BatchMember {
        input_claim: F::sample(17),
        coefficient: F::sample(28),
        rounds: 2,
        offset: 1,
    };
    let prelude = BatchPrelude::try_new(vec![member.clone()], 4, 1).unwrap();
    assert!(matches!(
        prelude.member_output_scale(1, &[F::sample(7); 4]),
        Err(SumcheckError::RoundMemberIndexOutOfRange {
            member: 1,
            members: 1
        })
    ));
    assert!(matches!(
        prelude.member_output_scale(0, &[F::sample(7); 3]),
        Err(SumcheckError::WrongNumberOfChallenges {
            expected: 4,
            got: 3
        })
    ));
    let mut invalid = prelude.clone();
    invalid.members[0].offset = 3;
    assert!(matches!(
        invalid.member_output_scale(0, &[F::sample(7); 4]),
        Err(SumcheckError::BatchMemberWindowOutOfRange { member: 0, .. })
    ));
    invalid = prelude.clone();
    invalid.max_num_vars = 1;
    assert!(matches!(
        invalid.member_output_scale(0, &[F::sample(7)]),
        Err(SumcheckError::BatchMemberRoundsOutOfRange { member: 0, .. })
    ));
    invalid = prelude.clone();
    invalid.max_degree = 0;
    assert!(matches!(
        invalid.member_output_scale(0, &[F::sample(7); 4]),
        Err(SumcheckError::ZeroBatchDegree { max_num_vars: 4 })
    ));
    invalid = prelude;
    invalid.max_num_vars = 258;
    assert!(matches!(
        invalid.member_output_scale(0, &[F::sample(7); 258]),
        Err(SumcheckError::BatchPaddingExponentOutOfRange {
            member: 0,
            exponent: 256
        })
    ));
    assert!(matches!(
        BatchPrelude::try_new(vec![member], 258, 1),
        Err(SumcheckError::BatchPaddingExponentOutOfRange {
            member: 0,
            exponent: 256
        })
    ));
}

fn integer_domains<F: BinaryField>() {
    for (domain_size, degree) in [(3, 2), (4, 127)] {
        let result: Result<Vec<F>, _> =
            CenteredIntegerDomain::new(domain_size).round_sum_coefficients(degree);
        assert!(
            matches!(result, Err(SumcheckError::IntegerDomainNotDistinct { domain_size: size }) if size == domain_size)
        );
    }
    let coefficients: Vec<F> = CenteredIntegerDomain::new(2)
        .round_sum_coefficients(2)
        .unwrap();
    assert_eq!(coefficients, vec![F::zero(), F::one(), F::one()]);
    let invalid: Result<Vec<F>, _> = CenteredIntegerDomain::new(0).round_sum_coefficients(2);
    assert!(matches!(
        invalid,
        Err(SumcheckError::InvalidIntegerDomain { domain_size: 0 })
    ));
    let spec: Result<Vec<F>, _> = SumcheckDomainSpec::centered_integer(3).round_sum_coefficients(2);
    assert!(matches!(
        spec,
        Err(SumcheckError::IntegerDomainNotDistinct { domain_size: 3 })
    ));
    let polynomial = UnivariatePoly::new(vec![F::sample(1), F::sample(2), F::sample(3)]);
    assert!(matches!(
        CenteredIntegerDomain::new(3).check_round_sum(0, F::sample(4), &polynomial),
        Err(SumcheckError::IntegerDomainNotDistinct { domain_size: 3 })
    ));
    let mut transcript = prover();
    let mut untouched = prover();
    assert!(matches!(
        prove_uniskip_clear::<F, _>(polynomial.clone(), F::sample(4), 2, 3, &mut transcript),
        Err(SumcheckError::IntegerDomainNotDistinct { domain_size: 3 })
    ));
    assert!(matches!(
        prove_uniskip_clear::<F, _>(polynomial, F::sample(4), 1, 3, &mut transcript),
        Err(SumcheckError::DegreeBoundExceeded { got: 2, max: 1 })
    ));
    assert!(transcript.narg().is_empty());
    assert_eq!(fingerprint(&mut transcript), fingerprint(&mut untouched));
}

macro_rules! binary_tests {
    ($module:ident, $field:ty) => {
        mod $module {
            use super::*;
            #[test]
            fn single_member_degrees_one_through_three() {
                single_members::<$field>();
            }
            #[test]
            fn mixed_lengths_degree_one() {
                mixed_linear_members::<$field>();
            }
            #[test]
            fn mixed_lengths_degree_two() {
                mixed_quadratic_members::<$field>();
            }
            #[test]
            fn zero_native_claim_after_leading_padding() {
                zero_native_claim::<$field>();
            }
            #[test]
            fn zero_padding_factor_in_engine_and_output_scale() {
                zero_padding_factor::<$field>();
            }
            #[test]
            fn incorrect_claim_coefficient_and_round_count() {
                rejection::<$field>();
            }
            #[test]
            fn full_coefficient_rounds() {
                uncompressed::<$field>();
            }
            #[test]
            fn output_scale_rejects_invalid_requests() {
                output_scale_errors::<$field>();
            }
            #[test]
            fn integer_domain_rejection_before_transcript_absorption() {
                integer_domains::<$field>();
            }
        }
    };
}

binary_tests!(f128, F128);
binary_tests!(f64, F64);

#[test]
fn odd_characteristic_output_scales_are_one() {
    let members = [2, 0]
        .into_iter()
        .map(|offset| BatchMember {
            input_claim: Fr::from_u64(11),
            coefficient: Fr::from_u64(7),
            rounds: 2,
            offset,
        })
        .collect();
    let prelude = BatchPrelude::try_new(members, 4, 1).unwrap();
    for member in 0..2 {
        assert_eq!(
            prelude
                .member_output_scale(
                    member,
                    &[Fr::from_u64(3), Fr::one(), Fr::from_u64(5), Fr::from_u64(7)]
                )
                .unwrap(),
            Fr::one()
        );
    }
}
