#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixed tables"
)]

use jolt_field::{
    Ext2, ExtField, Field, Fr, One, Prime128Offset275 as P, Prime64Offset59, Ring, Zero,
};
use jolt_poly::{eval_round_wire, Point, UnivariatePoly, WireFormat};
use jolt_sumcheck::{
    prove_single, verify_single, EqFactoredRound, ProveRounds, ProverRoundChannel, SingleFormat,
    SingleInstance, SingleReduction, SingleSumcheckProof, SumcheckDomain, SumcheckError,
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

        assert_eq!(
            eval_round_wire(
                WireFormat::EqFactored { tau },
                &q.coefficients()[1..],
                claim,
                point
            ),
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

mod allocation {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub struct Counting;
    thread_local! {
        static COUNTS: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
    }
    // SAFETY: every allocation operation forwards the original layout and pointer
    // to System; the thread-local counters neither allocate nor affect ownership.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            COUNTS.with(|counts| {
                if let Some((n, bytes)) = counts.get() {
                    counts.set(Some((n + 1, bytes + layout.size())));
                }
            });
            // SAFETY: the caller supplies the unchanged GlobalAlloc contract arguments.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the caller supplies the unchanged GlobalAlloc contract arguments.
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            COUNTS.with(|counts| {
                if let Some((n, bytes)) = counts.get() {
                    counts.set(Some((n + 1, bytes + size)));
                }
            });
            // SAFETY: the caller supplies the unchanged GlobalAlloc contract arguments.
            unsafe { System.realloc(ptr, layout, size) }
        }
    }
    pub fn measure<R>(f: impl FnOnce() -> R) -> (R, (usize, usize)) {
        COUNTS.with(|counts| counts.set(Some((0, 0))));
        let result = f();
        let counts = COUNTS.with(|counts| counts.replace(None).unwrap());
        (result, counts)
    }
}

use allocation::Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Replays a four-round fixture under the counting allocator.
/// Counts for `verify_single` and for the same transcript operations issued
/// directly, without the driver.
type Counts = (usize, usize);

fn verify_fixture_counted(factored: bool) -> (Counts, Counts, [u8; 32], [u8; 32]) {
    use jolt_transcript::{FieldTranscript, PrivateChannel};
    let mut kernel = ProductKernel::<Fr>::sampled(4, 71, factored, Fr::from_u64);
    let instance = instance(&kernel);
    let (proof, proved, state) = prove(&mut kernel, &instance);
    let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
    let (replayed, counts) =
        allocation::measure(|| verify_single(&instance, &proof, &mut transcript).unwrap());
    assert_eq!(replayed, proved);
    let mut direct = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
    let ((), transcript_only) = allocation::measure(|| {
        direct.bind_derived(&instance.input_claim);
        for wire in &proof.rounds {
            direct.append_private(wire.as_slice());
            let _: Fr = direct.challenge().unwrap();
        }
    });
    (
        counts,
        transcript_only,
        state,
        transcript.verifier_message::<[u8; 32]>(),
    )
}

/// Beyond the transcript's own allocations, the verifier allocates only the
/// challenge vector: it borrows each round's wire from the proof.
#[test]
fn verifier_allocates_only_its_challenges() {
    for factored in [false, true] {
        let (driver, transcript_only, _, _) = verify_fixture_counted(factored);
        assert_eq!(
            driver,
            (
                transcript_only.0 + 1,
                transcript_only.1 + 4 * std::mem::size_of::<Fr>()
            )
        );
    }
}

/// Pinned for jolt-transcript's default SHA3-256 digests; its
/// `keccak256-digests` feature changes the session identifier.
#[test]
fn verifier_transcript_digest_is_frozen() {
    for factored in [false, true] {
        let (_, _, prover_state, verifier_state) = verify_fixture_counted(factored);
        let expected = if factored {
            [
                97, 113, 63, 193, 220, 197, 166, 119, 64, 24, 22, 35, 5, 37, 249, 171, 2, 17, 145,
                252, 109, 231, 81, 118, 225, 66, 118, 123, 230, 55, 65, 74,
            ]
        } else {
            [
                252, 152, 71, 241, 238, 192, 1, 107, 99, 35, 137, 148, 145, 207, 215, 31, 179, 114,
                155, 134, 130, 217, 32, 196, 102, 94, 5, 41, 22, 123, 137, 32,
            ]
        };
        assert_eq!(prover_state, expected);
        assert_eq!(verifier_state, expected);
    }
}

#[test]
fn checked_shape_boundaries() {
    use jolt_sumcheck::SumcheckShape;
    assert!(matches!(
        SumcheckShape::new::<Fr>(1, 0),
        Err(SumcheckError::ZeroDegreeBound)
    ));
    assert!(matches!(
        SumcheckShape::new::<Fr>(0, usize::MAX),
        Err(SumcheckError::DegreeOverflow { .. })
    ));
    assert!(matches!(
        SumcheckShape::new::<Fr>(2, usize::MAX),
        Err(SumcheckError::ShapeOverflow { .. })
    ));
    assert_eq!(SumcheckShape::new::<Fr>(0, 1).unwrap().degree_bound(), 1);
}

#[test]
fn round_wires_evaluate_without_allocating() {
    let point = Fr::from_u64(13);
    let tau = Fr::from_u64(7);
    for coefficients in [
        vec![],
        vec![Fr::from_u64(9)],
        vec![Fr::from_u64(9), Fr::from_u64(3)],
        vec![
            Fr::from_u64(9),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(11),
        ],
    ] {
        // Direct power sum, independent of the wire evaluation.
        let oracle = |x: Fr| {
            coefficients
                .iter()
                .enumerate()
                .map(|(i, &c)| c * (0..i).fold(Fr::one(), |a, _| a * x))
                .sum::<Fr>()
        };
        let eq_claim = (Fr::one() - tau) * oracle(Fr::zero()) + tau * oracle(Fr::one());
        let eq_wire: Vec<Fr> = coefficients.iter().skip(1).copied().collect();
        let (value, counts) = allocation::measure(|| {
            eval_round_wire(WireFormat::EqFactored { tau }, &eq_wire, eq_claim, point)
        });
        assert_eq!(counts, (0, 0));
        assert_eq!(value, oracle(point));
        if !coefficients.is_empty() {
            let wire: Vec<Fr> = std::iter::once(coefficients[0])
                .chain(coefficients.iter().skip(2).copied())
                .collect();
            let claim = oracle(Fr::zero()) + oracle(Fr::one());
            let (value, counts) = allocation::measure(|| {
                eval_round_wire(WireFormat::Compressed, &wire, claim, point)
            });
            assert_eq!(counts, (0, 0));
            assert_eq!(value, oracle(point));
        }
    }
}

#[derive(Debug)]
enum ChannelError {
    Sumcheck(SumcheckError<Fr>),
    Transport,
}

impl From<SumcheckError<Fr>> for ChannelError {
    fn from(error: SumcheckError<Fr>) -> Self {
        Self::Sumcheck(error)
    }
}

/// One round whose polynomial is fixed by the test.
struct OneRound(Option<UnivariatePoly<Fr>>);
impl<Err> ProveRounds<Fr, Err> for OneRound {
    fn num_rounds(&self) -> usize {
        1
    }
    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        _claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, Err> {
        assert_eq!(round, 0);
        assert!(bind.is_none());
        Ok(self.0.take().unwrap())
    }
    fn finish_rounds(&mut self, _bind: Fr) -> Result<(), Err> {
        Ok(())
    }
}

/// Counts transport calls and draws the challenge 3.
#[derive(Default)]
struct Transport {
    binds: usize,
    sends: usize,
    draws: usize,
    fail: bool,
    wires: Vec<Vec<Fr>>,
}
impl ProverRoundChannel<Fr> for Transport {
    type Error = ChannelError;
    fn bind_input_claim(&mut self, _claim: &Fr) -> Result<(), ChannelError> {
        self.binds += 1;
        Ok(())
    }
    fn send_round(&mut self, _round: usize, wire: &[Fr]) -> Result<(), ChannelError> {
        self.sends += 1;
        self.wires.push(wire.to_vec());
        if self.fail {
            Err(ChannelError::Transport)
        } else {
            Ok(())
        }
    }
    fn round_challenge(&mut self, _round: usize, _degree: usize) -> Result<Fr, ChannelError> {
        self.draws += 1;
        Ok(Fr::from_u64(3))
    }
}

fn prove_one(
    instance: &SingleInstance<Fr>,
    coefficients: &[u64],
    channel: &mut Transport,
) -> Result<SingleReduction<Fr>, ChannelError> {
    let poly = UnivariatePoly::new(coefficients.iter().map(|&c| Fr::from_u64(c)).collect());
    jolt_sumcheck::prove_single_with(&mut OneRound(Some(poly)), instance, channel)
}

#[test]
fn prover_checks_each_round_before_sending_it() {
    let compressed = SingleInstance {
        rounds: 1,
        degree: 2,
        input_claim: Fr::from_u64(5),
        format: SingleFormat::Compressed,
    };
    // s = 1 + 2X + X^2: s(0) + s(1) = 5 and s(3) = 16; trailing zeros are padding.
    for coefficients in [&[1, 2, 1][..], &[1, 2, 1, 0, 0]] {
        let mut channel = Transport::default();
        let reduction = prove_one(&compressed, coefficients, &mut channel).unwrap();
        assert_eq!(reduction.output_claim, Fr::from_u64(16));
        assert_eq!((channel.sends, channel.draws), (1, 1));
        assert_eq!(channel.wires, [vec![Fr::from_u64(1), Fr::from_u64(1)]]);
    }
    // The kernel proves one round; the instance asks for two.
    let two = SingleInstance {
        rounds: 2,
        ..compressed.clone()
    };
    let mut channel = Transport::default();
    assert!(matches!(
        prove_one(&two, &[1, 2, 1], &mut channel),
        Err(ChannelError::Sumcheck(SumcheckError::RoundCountMismatch {
            expected: 2,
            got: 1
        }))
    ));
    assert_eq!((channel.binds, channel.sends, channel.draws), (0, 0, 0));
    let mut channel = Transport::default();
    assert!(matches!(
        prove_one(&compressed, &[1, 1, 1], &mut channel),
        Err(ChannelError::Sumcheck(SumcheckError::RoundCheckFailed {
            round: 0,
            ..
        }))
    ));
    assert_eq!((channel.sends, channel.draws), (0, 0));
    // The round sum holds, but a coefficient past the first excess one is nonzero.
    let mut channel = Transport::default();
    assert!(matches!(
        prove_one(&compressed, &[1, 1, 1, 0, 1], &mut channel),
        Err(ChannelError::Sumcheck(SumcheckError::DegreeBoundExceeded {
            got: 4,
            max: 2
        }))
    ));
    assert_eq!((channel.sends, channel.draws), (0, 0));
    let mut channel = Transport {
        fail: true,
        ..Transport::default()
    };
    assert!(matches!(
        prove_one(&compressed, &[1, 2, 1], &mut channel),
        Err(ChannelError::Transport)
    ));
    assert_eq!((channel.sends, channel.draws), (1, 0));

    let factored = SingleInstance {
        format: SingleFormat::EqFactored {
            equality_point: Point::low_to_high(vec![Fr::from_u64(7)]),
        },
        ..compressed
    };
    for instance in [&compressed, &factored] {
        let zero = SingleInstance {
            input_claim: Fr::zero(),
            ..instance.clone()
        };
        let mut channel = Transport::default();
        let _ = prove_one(&zero, &[], &mut channel).unwrap();
        assert_eq!(channel.wires, [vec![Fr::zero(); 2]]);
    }
    // q = 1 + X fails q(0) + 7 (q(1) - q(0)) = 5.
    let mut channel = Transport::default();
    assert!(matches!(
        prove_one(&factored, &[1, 1], &mut channel),
        Err(ChannelError::Sumcheck(SumcheckError::RoundCheckFailed {
            round: 0,
            ..
        }))
    ));
    assert_eq!((channel.sends, channel.draws), (0, 0));
}

#[test]
fn streaming_visitor_borrows_wires_and_stops_on_failure() {
    use jolt_sumcheck::{visit_single_rounds, RoundSpec, SumcheckShape, VerifierRoundChannel};
    use std::borrow::Cow;
    struct Borrowed {
        wire: [Fr; 2],
        calls: usize,
    }
    impl VerifierRoundChannel<Fr> for Borrowed {
        type Error = ChannelError;
        type Slots = usize;
        fn verify_round<R>(
            &mut self,
            round: usize,
            _spec: RoundSpec,
            visit: impl FnOnce(Cow<'_, [Fr]>, usize, Fr) -> Result<R, ChannelError>,
        ) -> Result<R, ChannelError> {
            self.calls += 1;
            visit(Cow::Borrowed(&self.wire), round, Fr::one())
        }
    }
    let mut channel = Borrowed {
        wire: [Fr::one(); 2],
        calls: 0,
    };
    let pointer = channel.wire.as_ptr();
    let mut visits = 0;
    let result = visit_single_rounds(
        SumcheckShape::new::<Fr>(4, 2).unwrap(),
        &mut channel,
        |round, wire, slot, challenge| {
            assert!(matches!(wire, Cow::Borrowed(_)));
            assert_eq!(wire.as_ptr(), pointer);
            assert_eq!(slot, round);
            assert_eq!(challenge, Fr::one());
            visits += 1;
            Err(ChannelError::Transport)
        },
    );
    assert!(matches!(result, Err(ChannelError::Transport)));
    assert_eq!((channel.calls, visits), (1, 1));
    channel.calls = 0;
    let result = visit_single_rounds(
        SumcheckShape::new::<Fr>(4, 1).unwrap(),
        &mut channel,
        |_, _, _, _| {
            visits += 1;
            Ok(())
        },
    );
    assert!(matches!(
        result,
        Err(ChannelError::Sumcheck(SumcheckError::WrongRoundWidth {
            round: 0,
            ..
        }))
    ));
    assert_eq!((channel.calls, visits), (1, 1));
}

#[test]
fn clear_prover_moves_the_kernel_buffer_into_the_proof() {
    use jolt_sumcheck::{prove_single_with, ClearProverRounds};
    // q = -2 + X: q(0) + 7 (q(1) - q(0)) = 5.
    let coefficients = vec![-Fr::from_u64(2), Fr::one(), Fr::zero()];
    let pointer = coefficients.as_ptr();
    let instance = SingleInstance {
        input_claim: Fr::from_u64(5),
        rounds: 1,
        degree: 2,
        format: SingleFormat::EqFactored {
            equality_point: Point::low_to_high(vec![Fr::from_u64(7)]),
        },
    };
    let mut kernel = OneRound(Some(UnivariatePoly::new(coefficients)));
    let mut transcript = prover_transcript::<Blake2b512>(&session(), b"");
    let mut channel = ClearProverRounds::<_, _, SumcheckError<Fr>>::new(&mut transcript);
    let reduction = prove_single_with(&mut kernel, &instance, &mut channel).unwrap();
    let proof = channel.into_proof();
    assert_eq!(proof.rounds[0].as_ptr(), pointer);
    assert_eq!(proof.rounds[0], vec![Fr::one(), Fr::zero()]);
    assert_eq!(replay(&instance, &proof).unwrap().0, reduction);
}
