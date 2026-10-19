#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use core::convert::Infallible;
use core::marker::PhantomData;

use jolt_field::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, Ext2, ExtField, Field, Fr};
use jolt_field::{One, Prime64Offset59, Zero};
use jolt_poly::{OmittedConstantPoly, Point, UnivariatePoly};
use jolt_sumcheck::{
    prove_single, verify_single, EqFactoredRound, ProveRounds, ProverRoundContext, SingleFormat,
    SingleInstance, SingleReplay, SumcheckDomain, SumcheckError, SumcheckSite,
    VerifierRoundContext,
};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, SiteId, VerifierTranscript,
};

type H = Blake2b512;
const PROTOCOL: ProtocolId = ProtocolId::new::<H>("jolt-sumcheck/single-test");
const SESSION: &[u8] = b"single";

fn sample_word(seed: u64) -> u64 {
    let mut word = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

fn sample<F: Field, E: ExtField<F>>(seed: u64) -> E {
    E::from_base_fn(|index| F::from_u64(sample_word(seed.wrapping_mul(7) + index as u64)))
}

fn site(site: SumcheckSite) -> SiteId {
    let mut id = [0u8; 32];
    id[..4].copy_from_slice(&site.invocation.to_le_bytes());
    id[4..8].copy_from_slice(&site.round.to_le_bytes());
    id[8..12].copy_from_slice(&(site.role as u32).to_le_bytes());
    SiteId(id)
}

struct ProverContext<E> {
    transcript: ProverTranscript<H>,
    _field: PhantomData<E>,
}

impl<E: Field> ProverRoundContext<E> for ProverContext<E> {
    type Sponge = H;
    type Error = Infallible;

    fn transcript_mut(&mut self) -> &mut ProverTranscript<H> {
        &mut self.transcript
    }

    fn site(&self, sumcheck_site: SumcheckSite) -> Option<SiteId> {
        Some(site(sumcheck_site))
    }

    fn challenge(&mut self, sumcheck_site: SumcheckSite) -> Result<E, Infallible> {
        self.transcript.site(site(sumcheck_site));
        Ok(self.transcript.challenge())
    }
}

struct VerifierContext<'proof, E> {
    transcript: VerifierTranscript<'proof, H>,
    _field: PhantomData<E>,
}

impl<'proof, E: Field> VerifierRoundContext<'proof, E> for VerifierContext<'proof, E> {
    type Sponge = H;
    type Error = Infallible;

    fn transcript_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
        &mut self.transcript
    }

    fn site(&self, sumcheck_site: SumcheckSite) -> Option<SiteId> {
        Some(site(sumcheck_site))
    }

    fn challenge(&mut self, sumcheck_site: SumcheckSite) -> Result<E, Infallible> {
        self.transcript.site(site(sumcheck_site));
        Ok(self.transcript.challenge())
    }
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
    fn sampled<F: Field>(rounds: usize, seed: u64, eq_factored: bool) -> Self
    where
        E: ExtField<F>,
    {
        let table = |offset: u64| {
            (0..1u64 << rounds)
                .map(|x| sample::<F, E>(seed + offset + x))
                .collect()
        };
        Self {
            a: table(0),
            b: table(1 << 20),
            tau: eq_factored.then(|| {
                (0..rounds as u64)
                    .map(|k| sample::<F, E>(seed + (2 << 20) + k))
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

fn instance<E: Field>(kernel: &ProductKernel<E>, invocation: u32) -> SingleInstance<E> {
    SingleInstance {
        input_claim: kernel.claim(),
        rounds: kernel.num_rounds(),
        degree: 2,
        format: kernel.tau.clone().map_or(SingleFormat::Compressed, |tau| {
            SingleFormat::EqFactored {
                equality_point: Point::low_to_high(tau),
            }
        }),
        invocation,
    }
}

fn prove<F, E>(kernel: &mut ProductKernel<E>, instance: &SingleInstance<E>) -> (Vec<E>, Vec<u8>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
{
    let mut context = ProverContext {
        transcript: ProverTranscript::new(&PROTOCOL, SESSION),
        _field: PhantomData,
    };
    let proved = prove_single::<F, E, _, _>(kernel, instance, &mut context).unwrap();
    (proved.challenges, context.transcript.finish())
}

fn replay<F, E>(
    instance: &SingleInstance<E>,
    proof: &[u8],
) -> Result<SingleReplay<E>, SumcheckError<E>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
{
    let mut context = VerifierContext {
        transcript: VerifierTranscript::new(&PROTOCOL, SESSION, proof),
        _field: PhantomData,
    };
    let replay = verify_single::<F, E, _, _>(instance, &mut context)?;
    context.transcript.finish()?;
    Ok(replay)
}

/// The terminal oracle check of the product relation: the normalized claim
/// is `a(r) b(r)`, the eq factors `Π eq(τ_i, r_i)` being the caller's.
fn oracle_accepts<E: Field>(tables: &(Vec<E>, Vec<E>), replay: &SingleReplay<E>) -> bool {
    replay.output_claim == mle(&tables.0, &replay.challenges) * mle(&tables.1, &replay.challenges)
}

fn round_trip<F, E>(eq_factored: bool, seed: u64)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
{
    let mut kernel = ProductKernel::<E>::sampled::<F>(4, seed, eq_factored);
    let tables = (kernel.a.clone(), kernel.b.clone());
    let instance = instance(&kernel, 9);
    let (prover_point, proof) = prove::<F, E>(&mut kernel, &instance);
    assert_eq!(proof.len(), 4 * 2 * E::NUM_BYTES);

    let replay = replay::<F, E>(&instance, &proof).unwrap();
    assert_eq!(replay.challenges, prover_point);
    assert!(oracle_accepts(&tables, &replay));
}

#[test]
fn honest_compressed_and_eq_factored_proofs_verify() {
    for eq_factored in [false, true] {
        round_trip::<Fr, Fr>(eq_factored, 11);
        round_trip::<Prime64Offset59, Ext2<Prime64Offset59>>(eq_factored, 12);
    }
}

#[test]
fn eq_factored_round_check_and_recovery_match_direct_evaluation() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    for seed in 0..8u64 {
        let q = UnivariatePoly::new((0..4).map(|k| sample::<F, E>(100 * seed + k)).collect());
        let tau = sample::<F, E>(100 * seed + 50);
        let point = sample::<F, E>(100 * seed + 51);
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
    type F = Prime64Offset59;
    type E = Ext2<F>;
    let rounds = 5;
    let mut kernel = ProductKernel::<E>::sampled::<F>(rounds, 21, true);
    let tables = (kernel.a.clone(), kernel.b.clone());
    let tau = kernel.tau.clone().unwrap();
    let instance = instance(&kernel, 0);
    let (point, proof) = prove::<F, E>(&mut kernel, &instance);
    let replay = replay::<F, E>(&instance, &proof).unwrap();

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
    assert_eq!(replay.output_claim, normalized(rounds));
}

#[test]
fn tampered_coefficient_or_claim_is_rejected() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    for eq_factored in [false, true] {
        let mut kernel = ProductKernel::<E>::sampled::<F>(3, 31, eq_factored);
        let tables = (kernel.a.clone(), kernel.b.clone());
        let instance = instance(&kernel, 2);
        let (_, proof) = prove::<F, E>(&mut kernel, &instance);

        for round in 0..3 {
            for coefficient in 0..2 {
                let offset = (2 * round + coefficient) * E::NUM_BYTES;
                let mut tampered = proof.clone();
                let value = E::from_bytes_le_checked(&tampered[offset..offset + E::NUM_BYTES])
                    .unwrap()
                    + E::one();
                tampered[offset..offset + E::NUM_BYTES].copy_from_slice(&value.to_bytes_le_vec());
                let replay = replay::<F, E>(&instance, &tampered).unwrap();
                assert!(!oracle_accepts(&tables, &replay));
            }
        }

        let mut wrong = instance.clone();
        wrong.input_claim += E::one();
        let replay = replay::<F, E>(&wrong, &proof).unwrap();
        assert!(!oracle_accepts(&tables, &replay));

        let mut kernel = ProductKernel::<E>::sampled::<F>(3, 31, eq_factored);
        let mut context = ProverContext {
            transcript: ProverTranscript::new(&PROTOCOL, SESSION),
            _field: PhantomData,
        };
        assert!(matches!(
            prove_single::<F, E, _, _>(&mut kernel, &wrong, &mut context),
            Err(SumcheckError::RoundCheckFailed { round: 0, .. })
        ));
    }
}

#[test]
fn truncated_proof_is_rejected() {
    let mut kernel = ProductKernel::<Fr>::sampled::<Fr>(3, 41, true);
    let instance = instance(&kernel, 0);
    let (_, proof) = prove::<Fr, Fr>(&mut kernel, &instance);
    assert!(matches!(
        replay::<Fr, Fr>(&instance, &proof[..proof.len() - 1]),
        Err(SumcheckError::Transcript(_))
    ));
}
