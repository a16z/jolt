#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixed tables"
)]

use jolt_field::{
    Ext2, ExtField, Field, Fr, One, Prime128Offset275 as P, Prime64Offset59, Ring, Zero,
};
use jolt_poly::{OmittedConstantPoly, Point, UnivariatePoly};
use jolt_sumcheck::{
    prove_single, verify_single, EqFactoredRound, ProveRounds, ProveSingleRounds, ProverRound,
    ProverRoundChannel, SingleFormat, SingleInstance, SingleReduction, SingleSumcheckProof,
    SumcheckDomain, SumcheckError,
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
            Err(SumcheckError::InvalidInput { .. })
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

#[test]
fn verifier_allocation_and_transcript_baseline() {
    for factored in [false, true] {
        let mut kernel = ProductKernel::<Fr>::sampled(4, 71, factored, Fr::from_u64);
        let instance = instance(&kernel);
        let (proof, proved, state) = prove(&mut kernel, &instance);
        let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
        let (replayed, counts) =
            allocation::measure(|| verify_single(&instance, &proof, &mut transcript).unwrap());
        assert_eq!(replayed, proved);
        // Captured at ec0f33b73 with the same fixture and counting allocator.
        assert!(counts.0 <= 22 && counts.1 <= 1152, "{counts:?}");
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
        assert_eq!(state, expected);
        assert_eq!(transcript.verifier_message::<[u8; 32]>(), expected);
    }
}

#[test]
fn checked_shape_boundaries() {
    use jolt_sumcheck::{SumcheckShape, SumcheckShapeError};
    assert_eq!(
        SumcheckShape::new(1, 0),
        Err(SumcheckShapeError::ZeroDegree)
    );
    assert!(matches!(
        SumcheckShape::new(0, usize::MAX),
        Err(SumcheckShapeError::DegreeOverflow { .. })
    ));
    assert!(matches!(
        SumcheckShape::new(2, usize::MAX),
        Err(SumcheckShapeError::Overflow { .. })
    ));
    assert_eq!(
        SumcheckShape::new(u32::MAX as usize, 1)
            .unwrap()
            .num_rounds(),
        u32::MAX as usize
    );
    assert!(matches!(
        SumcheckShape::new(u32::MAX as usize + 1, 1),
        Err(SumcheckShapeError::RoundCountOutOfRange { .. })
    ));
    assert_eq!(SumcheckShape::new(0, 1).unwrap().degree_bound(), 1);
}

#[test]
fn borrowed_and_owned_rounds_match_polynomial_oracle_without_allocating() {
    use jolt_poly::{eval_round_wire, CompressedPoly, WireFormat};
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
        // Direct power sum, independent of both evaluation implementations.
        let oracle = |x: Fr| {
            coefficients
                .iter()
                .enumerate()
                .map(|(i, &c)| c * (0..i).fold(Fr::one(), |a, _| a * x))
                .sum::<Fr>()
        };
        let eq_claim = (Fr::one() - tau) * oracle(Fr::zero()) + tau * oracle(Fr::one());
        let omitted = OmittedConstantPoly::from_q_coefficients(coefficients.clone());
        let (borrowed, counts) = allocation::measure(|| {
            eval_round_wire(
                WireFormat::EqFactored { tau },
                omitted.coefficients(),
                eq_claim,
                point,
            )
        });
        assert_eq!(counts, (0, 0));
        assert_eq!(borrowed, oracle(point));
        assert_eq!(
            omitted.evaluate_from_claim(eq_claim, tau, point),
            oracle(point)
        );
        if !coefficients.is_empty() {
            let wire = std::iter::once(coefficients[0])
                .chain(coefficients.iter().skip(2).copied())
                .collect();
            let compressed = CompressedPoly::new(wire);
            let claim = oracle(Fr::zero()) + oracle(Fr::one());
            let (borrowed, counts) = allocation::measure(|| {
                eval_round_wire(
                    WireFormat::Compressed,
                    compressed.coeffs_except_linear_term(),
                    claim,
                    point,
                )
            });
            assert_eq!(counts, (0, 0));
            assert_eq!(borrowed, oracle(point));
            assert_eq!(compressed.eval_from_hint(&claim, &point), oracle(point));
        }
    }
}

#[test]
fn clear_channels_preserve_wrapper_proof_and_transcript() {
    use jolt_sumcheck::{
        prove_single_with, verify_single_with, ClearProverRounds, ClearVerifierRounds, FullRounds,
    };
    for factored in [false, true] {
        let mut kernel = ProductKernel::<P>::sampled(3, 81, factored, sample::<P, P>);
        let instance = instance(&kernel);
        let (expected_proof, expected_reduction, expected_state) = prove(&mut kernel, &instance);
        let mut kernel = ProductKernel::<P>::sampled(3, 81, factored, sample::<P, P>);
        let mut transcript = prover_transcript::<Blake2b512>(&session(), b"");
        let mut channel = ClearProverRounds::new(instance.shape().unwrap(), &mut transcript);
        let reduction =
            prove_single_with(&mut FullRounds::new(&mut kernel), &instance, &mut channel).unwrap();
        let proof = channel.into_proof();
        assert_eq!(proof, expected_proof);
        assert_eq!(reduction, expected_reduction);
        assert_eq!(transcript.verifier_message::<[u8; 32]>(), expected_state);
        let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
        let mut channel =
            ClearVerifierRounds::new(instance.shape().unwrap(), &proof, &mut transcript).unwrap();
        assert_eq!(
            verify_single_with(&instance, &mut channel).unwrap(),
            expected_reduction
        );
        assert_eq!(transcript.verifier_message::<[u8; 32]>(), expected_state);
    }
}

#[derive(Debug, PartialEq)]
enum ChannelError {
    InvalidInput,
    Transport,
}

struct NativeRound(Option<ProverRound<Fr>>);
impl ProveSingleRounds<Fr, ChannelError> for NativeRound {
    fn num_rounds(&self) -> usize {
        1
    }
    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        _claim: Fr,
    ) -> Result<ProverRound<Fr>, ChannelError> {
        assert_eq!(round, 0);
        assert!(bind.is_none());
        self.0.take().ok_or(ChannelError::InvalidInput)
    }
    fn finish_rounds(&mut self, _bind: Fr) -> Result<(), ChannelError> {
        Ok(())
    }
}

#[derive(Default)]
struct Transport {
    sends: usize,
    draws: usize,
    pointer: Option<usize>,
    fail: bool,
}
impl ProverRoundChannel<Fr> for Transport {
    type Error = ChannelError;
    fn map_sumcheck_error(&self, _error: SumcheckError<Fr>) -> ChannelError {
        ChannelError::InvalidInput
    }
    fn bind_input_claim(&mut self, _claim: &Fr) -> Result<(), ChannelError> {
        Ok(())
    }
    fn send_round(&mut self, _round: usize, wire: &mut [Fr]) -> Result<(), ChannelError> {
        self.sends += 1;
        if let Some(pointer) = self.pointer {
            assert_eq!(pointer, wire.as_ptr() as usize);
        }
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

#[test]
fn native_round_validation_precedes_send_and_preserves_owned_buffer() {
    use jolt_sumcheck::{prove_single_with, ProverRound};
    let tau = Fr::from_u64(7);
    let mut instance = SingleInstance {
        rounds: 1,
        degree: 2,
        input_claim: Fr::from_u64(5),
        format: SingleFormat::EqFactored {
            equality_point: Point::low_to_high(vec![tau]),
        },
    };
    for wire in [vec![Fr::one(), -Fr::one()], vec![Fr::one()]] {
        let mut kernel = NativeRound(Some(ProverRound::EqFactored {
            wire,
            tau: tau + Fr::one(),
        }));
        let mut channel = Transport::default();
        assert_eq!(
            prove_single_with(&mut kernel, &instance, &mut channel),
            Err(ChannelError::InvalidInput)
        );
        assert_eq!((channel.sends, channel.draws), (0, 0));
    }
    let mut wire = Vec::with_capacity(2);
    wire.push(Fr::one());
    let pointer = wire.as_ptr() as usize;
    let mut kernel = NativeRound(Some(ProverRound::EqFactored { wire, tau }));
    let mut channel = Transport {
        pointer: Some(pointer),
        ..Transport::default()
    };
    let reduction = prove_single_with(&mut kernel, &instance, &mut channel).unwrap();
    assert_eq!(reduction.output_claim, Fr::one()); // q0 = 5 - 7, q(3) = 1
    assert_eq!((channel.sends, channel.draws), (1, 1));
    for output in [
        ProverRound::EqFactored {
            wire: vec![Fr::zero(); 3],
            tau,
        },
        ProverRound::Full(UnivariatePoly::new(vec![Fr::from_u64(5), Fr::one()])),
    ] {
        let mut channel = Transport::default();
        assert_eq!(
            prove_single_with(&mut NativeRound(Some(output)), &instance, &mut channel),
            Err(ChannelError::InvalidInput)
        );
        assert_eq!((channel.sends, channel.draws), (0, 0));
    }
    instance.format = SingleFormat::Compressed;
    for output in [
        ProverRound::EqFactored {
            wire: vec![Fr::zero(); 2],
            tau,
        },
        ProverRound::Full(UnivariatePoly::new(vec![Fr::one(), Fr::one(), Fr::one()])),
    ] {
        let mut channel = Transport::default();
        assert_eq!(
            prove_single_with(&mut NativeRound(Some(output)), &instance, &mut channel),
            Err(ChannelError::InvalidInput)
        );
        assert_eq!((channel.sends, channel.draws), (0, 0));
    }
    let mut channel = Transport::default();
    let padded = ProverRound::Full(UnivariatePoly::new(vec![
        Fr::one(),
        Fr::from_u64(2),
        Fr::one(),
        Fr::zero(),
        Fr::zero(),
    ]));
    let reduction =
        prove_single_with(&mut NativeRound(Some(padded)), &instance, &mut channel).unwrap();
    assert_eq!(reduction.output_claim, Fr::from_u64(16));
    let excess = ProverRound::Full(UnivariatePoly::new(vec![
        Fr::one(),
        Fr::from_u64(2),
        Fr::one(),
        Fr::zero(),
        Fr::one(),
    ]));
    let mut channel = Transport::default();
    assert_eq!(
        prove_single_with(&mut NativeRound(Some(excess)), &instance, &mut channel),
        Err(ChannelError::InvalidInput)
    );
    assert_eq!((channel.sends, channel.draws), (0, 0));
    let mut channel = Transport {
        fail: true,
        ..Transport::default()
    };
    let valid = ProverRound::Full(UnivariatePoly::new(vec![
        Fr::one(),
        Fr::from_u64(2),
        Fr::one(),
    ]));
    assert_eq!(
        prove_single_with(&mut NativeRound(Some(valid)), &instance, &mut channel),
        Err(ChannelError::Transport)
    );
    assert_eq!((channel.sends, channel.draws), (1, 0));
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
        fn validate_shape(&self, _shape: SumcheckShape) -> Result<(), ChannelError> {
            Ok(())
        }
        fn map_sumcheck_error(&self, _error: SumcheckError<Fr>) -> ChannelError {
            ChannelError::InvalidInput
        }
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
        SumcheckShape::new(4, 2).unwrap(),
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
    assert_eq!(result, Err(ChannelError::Transport));
    assert_eq!((channel.calls, visits), (1, 1));
    channel.calls = 0;
    let result = visit_single_rounds(
        SumcheckShape::new(4, 1).unwrap(),
        &mut channel,
        |_, _, _, _| {
            visits += 1;
            Ok(())
        },
    );
    assert_eq!(result, Err(ChannelError::InvalidInput));
    assert_eq!((channel.calls, visits), (1, 1));
}

#[test]
fn full_adapter_rejects_invalid_sequences_without_touching_kernel() {
    use jolt_sumcheck::FullRounds;
    let mut kernel = ProductKernel::<Fr>::sampled(2, 91, false, Fr::from_u64);
    let claim = kernel.claim();
    let mut adapter = FullRounds::new(&mut kernel);
    assert!(matches!(
        adapter.prove_round(None, 1, claim),
        Err(SumcheckError::InvalidInput { .. })
    ));
    assert!(matches!(
        adapter.prove_round(Some(Fr::one()), 0, claim),
        Err(SumcheckError::InvalidInput { .. })
    ));
    assert!(matches!(
        adapter.finish_rounds(Fr::one()),
        Err(SumcheckError::InvalidInput { .. })
    ));
    let _ = adapter.prove_round(None, 0, claim).unwrap();
    assert!(matches!(
        adapter.prove_round(None, 0, claim),
        Err(SumcheckError::InvalidInput { .. })
    ));
    assert!(matches!(
        adapter.prove_round(None, 1, claim),
        Err(SumcheckError::InvalidInput { .. })
    ));
    let _ = adapter.prove_round(Some(Fr::one()), 1, claim).unwrap();
    assert!(matches!(
        adapter.prove_round(Some(Fr::one()), 2, claim),
        Err(SumcheckError::InvalidInput { .. })
    ));
    adapter.finish_rounds(Fr::one()).unwrap();
    assert!(matches!(
        adapter.finish_rounds(Fr::one()),
        Err(SumcheckError::InvalidInput { .. })
    ));
    assert!(matches!(
        adapter.prove_round(None, 0, claim),
        Err(SumcheckError::InvalidInput { .. })
    ));
    assert_eq!(kernel.seen_claims.len(), 2);
}

#[test]
fn verifier_rejects_channel_shape_mismatch_before_claim_binding() {
    use jolt_sumcheck::{
        verify_single_with, visit_single_rounds, ClearVerifierRounds, RoundSpec, SumcheckShape,
        VerifierRoundChannel,
    };
    for factored in [false, true] {
        let mut kernel = ProductKernel::<Fr>::sampled(2, 101, factored, Fr::from_u64);
        let original = instance(&kernel);
        let (proof, _, _) = prove(&mut kernel, &original);
        for (rounds, degree) in [(0, 2), (1, 2), (3, 2), (2, 1)] {
            let mut wrong = original.clone();
            wrong.rounds = rounds;
            wrong.degree = degree;
            if factored {
                wrong.format = SingleFormat::EqFactored {
                    equality_point: Point::low_to_high(vec![Fr::from_u64(7); rounds]),
                };
            }
            let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
            let mut untouched = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
            let result = {
                let mut channel =
                    ClearVerifierRounds::new(original.shape().unwrap(), &proof, &mut transcript)
                        .unwrap();
                verify_single_with(&wrong, &mut channel)
            };
            if factored && rounds == 0 {
                assert!(matches!(result, Err(SumcheckError::EqFactoredZeroRounds)));
            } else {
                assert!(matches!(
                    result,
                    Err(SumcheckError::ChannelShapeMismatch { .. })
                ));
            }
            assert_eq!(
                transcript.verifier_message::<[u8; 32]>(),
                untouched.verifier_message::<[u8; 32]>()
            );
        }
        // Shape checks also protect the raw visitor, including an empty loop.
        let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
        let mut channel =
            ClearVerifierRounds::new(original.shape().unwrap(), &proof, &mut transcript).unwrap();
        let mut visits = 0;
        for shape in [
            SumcheckShape::new(0, 2).unwrap(),
            SumcheckShape::new(1, 2).unwrap(),
            SumcheckShape::new(2, 1).unwrap(),
        ] {
            let result = visit_single_rounds(shape, &mut channel, |_, _, (), _| {
                visits += 1;
                Ok(())
            });
            assert!(matches!(
                result,
                Err(SumcheckError::ChannelShapeMismatch { .. })
            ));
        }
        assert_eq!(visits, 0);
        // A direct caller cannot price a round's proof of work at another
        // degree than the channel's nonces were checked for.
        let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
        let mut untouched = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
        let mut channel =
            ClearVerifierRounds::new(original.shape().unwrap(), &proof, &mut transcript).unwrap();
        let result = channel.verify_round(
            0,
            RoundSpec {
                width: 2,
                degree: 1,
            },
            |_, (), _| Ok(()),
        );
        assert!(matches!(
            result,
            Err(SumcheckError::RoundDegreeMismatch {
                round: 0,
                expected: 2,
                got: 1
            })
        ));
        assert_eq!(
            transcript.verifier_message::<[u8; 32]>(),
            untouched.verifier_message::<[u8; 32]>()
        );
    }
    for format in [
        SingleFormat::Compressed,
        SingleFormat::EqFactored {
            equality_point: Point::low_to_high(vec![Fr::one()]),
        },
    ] {
        let empty = SingleSumcheckProof::<Fr>::default();
        let mut transcript = verifier_transcript::<Blake2b512>(&session(), b"", &[]);
        let mut channel =
            ClearVerifierRounds::new(SumcheckShape::new(0, 2).unwrap(), &empty, &mut transcript)
                .unwrap();
        let instance = SingleInstance {
            input_claim: Fr::one(),
            rounds: 1,
            degree: 2,
            format,
        };
        assert!(matches!(
            verify_single_with(&instance, &mut channel),
            Err(SumcheckError::ChannelShapeMismatch { .. })
        ));
    }
}

#[test]
fn clear_prover_moves_native_wire_into_proof() {
    use jolt_sumcheck::{prove_single_with, ClearProverRounds};
    struct OwnedWire(Vec<Fr>);
    impl ProveSingleRounds<Fr> for OwnedWire {
        fn num_rounds(&self) -> usize {
            1
        }
        fn prove_round(
            &mut self,
            _bind: Option<Fr>,
            _round: usize,
            _claim: Fr,
        ) -> Result<ProverRound<Fr>, SumcheckError<Fr>> {
            Ok(ProverRound::EqFactored {
                wire: std::mem::take(&mut self.0),
                tau: Fr::from_u64(7),
            })
        }
        fn finish_rounds(&mut self, _bind: Fr) -> Result<(), SumcheckError<Fr>> {
            Ok(())
        }
    }
    let mut kernel = OwnedWire(vec![Fr::one(), Fr::zero()]);
    let pointer = kernel.0.as_ptr();
    let instance = SingleInstance {
        input_claim: Fr::from_u64(5),
        rounds: 1,
        degree: 2,
        format: SingleFormat::EqFactored {
            equality_point: Point::low_to_high(vec![Fr::from_u64(7)]),
        },
    };
    let mut transcript = prover_transcript::<Blake2b512>(&session(), b"");
    let mut channel = ClearProverRounds::new(instance.shape().unwrap(), &mut transcript);
    let reduction = prove_single_with(&mut kernel, &instance, &mut channel).unwrap();
    let proof = channel.into_proof();
    assert_eq!(proof.rounds[0].as_ptr(), pointer);
    assert_eq!(proof.rounds[0], vec![Fr::one(), Fr::zero()]);
    assert_eq!(replay(&instance, &proof).unwrap().0, reduction);
}
