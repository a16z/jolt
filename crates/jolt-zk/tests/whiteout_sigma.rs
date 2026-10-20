//! The Whiteout Σ-protocol on synthetic residual relations: completeness,
//! the abort rate, the accepted response's distribution, and tamper
//! rejection, with a response prover that checks the statement directly.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]

use std::cell::RefCell;
use std::f64::consts::LN_2;
use std::sync::Arc;

use jolt_algebra::{CyclotomicRing, Field, One, Prime128OffsetA7F7, Ring, Zero};
use jolt_claims::{opening, FactoredExpr, Source};
use jolt_crypto::challenge::SparseChallenge;
use jolt_crypto::lattice::{
    derive_public_matrix_prefix, sample_box_digits, MatrixShape, MatrixView, PreparedMatrix,
    PublicMatrixSeed, CHALLENGE_RING_DIM,
};
use jolt_poly::EqPolynomial;
use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, Sponge, VerifierTranscript};
use jolt_zk::masking::{
    lower_factored, AffineRow, MaskedOperand, OperandShapeDomain, PadIndex, PadLayout, ProductRow,
    ResidualRelation, ResidualVar, RowsDomain, ZkError,
};
use jolt_zk::whiteout::{
    prove_sigma, sample_zero_constant_term, verify_sigma, EvaluationClaims, GadgetTerm,
    LatticePadCommitment, LiftedRelation, LinearBlock, PadPlacement, ResponseProver,
    ResponseSegment, ResponseStatement, ResponseWitness, RingLinearMap, Segment, SegmentMask,
    SegmentRole, SegmentShape, SegmentTable, SigmaParams, SigmaSetup, WhiteoutError, FIELD_LEVELS,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

type F = Prime128OffsetA7F7;
const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-zk/whiteout-sigma");
const LABEL: &[u8] = b"whiteout-sigma";
const BASE_PADS: usize = 5;

/// A response prover that sends `e` in the clear and has the verifier check
/// the statement directly: not zero knowledge, not succinct, the statement's
/// definition.
struct CheckingResponseProver;

impl<const D: usize> ResponseProver<F, D> for CheckingResponseProver {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        let bytes: Vec<u8> = witness.digits.iter().map(|&digit| digit as u8).collect();
        transcript.send_bytes(&bytes);
        Ok(())
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        let bytes = transcript.receive_bytes(statement.layout().len())?;
        let witness = ResponseWitness {
            digits: bytes.iter().map(|&byte| byte as i8).collect(),
        };
        Ok(statement.check_witness(&witness)?)
    }
}

/// Records every accepted response and its challenge, then defers.
#[derive(Default)]
struct CapturingResponseProver {
    accepted: RefCell<Vec<(Vec<i64>, SparseChallenge)>>,
}

impl<const D: usize> ResponseProver<F, D> for CapturingResponseProver {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let mut z = Vec::new();
        let mut rest = witness.digits.as_slice();
        for segment in statement.layout().segments {
            if let ResponseSegment::Response {
                coefficients,
                digits,
                ..
            } = segment
            {
                let (planes, tail) = rest.split_at(coefficients * digits);
                rest = tail;
                let mut values = vec![0i64; coefficients];
                for plane in planes.chunks_exact(coefficients).rev() {
                    for (value, &digit) in values.iter_mut().zip(plane) {
                        *value = *value * 16 + i64::from(digit);
                    }
                }
                z.extend(values);
            }
        }
        self.accepted
            .borrow_mut()
            .push((z, statement.challenge.clone()));
        CheckingResponseProver.prove_response(statement, witness, transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        CheckingResponseProver.verify_response(statement, transcript)
    }
}

fn stream(len: usize, seed: u8) -> Vec<F> {
    derive_public_matrix_prefix(len, &PublicMatrixSeed::shake256_paged_v1([seed; 32])).unwrap()
}

fn prepared<const D: usize>(rows: usize, width: usize, seed: u8) -> Arc<PreparedMatrix<F, D>> {
    let shape = MatrixShape {
        rows,
        width,
        ring_dim: D,
    };
    let data = stream(shape.num_field_elements().unwrap(), seed);
    Arc::new(PreparedMatrix::new(&MatrixView::new(shape, &data).unwrap()).unwrap())
}

fn digits(rng: &mut impl RngCore, len: usize) -> Vec<i64> {
    (0..len)
        .map(|_| i64::from(rng.next_u32() % 16) - 8)
        .collect()
}

fn ring_digits<const D: usize>(ring: &CyclotomicRing<F, D>) -> Vec<i64> {
    use jolt_algebra::balanced_decompose_coefficients_pow2_i8_into;
    use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
    let params = BalancedDecomposePow2Params::<F>::new(FIELD_LEVELS, 4);
    let mut out = vec![0i8; FIELD_LEVELS * D];
    balanced_decompose_coefficients_pow2_i8_into(ring.coefficients(), &mut out, &params);
    out.into_iter().map(i64::from).collect()
}

/// The prover's side of one synthetic instance: a residual witness `w` (a
/// public part and a Gaussian part) under a two-level commitment, five masked
/// messages and one masked product, all over `R_D`.
struct Instance<const D: usize> {
    relation: ResidualRelation<F>,
    claims: EvaluationClaims<F>,
    placement: PadPlacement,
    s: Vec<i64>,
    /// Commitment images as sent: `t` of the inner rows is zero.
    u: Vec<CyclotomicRing<F, D>>,
    t_p: Vec<CyclotomicRing<F, D>>,
}

/// Segment lengths, in challenge-ring elements.
struct Shape {
    public: usize,
    witness: usize,
    inner: usize,
    randomness: usize,
}

fn shape<const D: usize>() -> Shape {
    let m = D / CHALLENGE_RING_DIM;
    Shape {
        public: 2 * m,
        witness: 4 * m,
        inner: FIELD_LEVELS * m,
        randomness: m,
    }
}

fn pad_commitment<const D: usize>(placement: PadPlacement) -> LatticePadCommitment<F, D> {
    LatticePadCommitment::new(placement, &stream(1 << 18, 21), 2, 2).unwrap()
}

fn table<const D: usize>(pads: &LatticePadCommitment<F, D>) -> SegmentTable {
    let shape = shape::<D>();
    let segment = |role, len, mask| Segment {
        role,
        len,
        native_dim: D,
        log_basis: 4,
        shape: SegmentShape::Dense,
        mask,
    };
    let mut segments = vec![
        segment(SegmentRole::Witness, shape.public, SegmentMask::Public),
        segment(SegmentRole::Witness, shape.witness, SegmentMask::Gaussian),
        segment(SegmentRole::InnerDigits, shape.inner, SegmentMask::Gaussian),
        segment(
            SegmentRole::ResidualRandomness,
            shape.randomness,
            SegmentMask::Gaussian,
        ),
    ];
    segments.extend(pads.segments());
    SegmentTable::new(segments).unwrap()
}

/// Inner `A` (rank 1 over the witness) and outer `B` (rank 2 over the inner
/// digits and the randomness).
fn residual_matrices<const D: usize>() -> (Arc<PreparedMatrix<F, D>>, Arc<PreparedMatrix<F, D>>) {
    let shape = shape::<D>();
    let m = D / CHALLENGE_RING_DIM;
    (
        prepared::<D>(1, (shape.public + shape.witness) / m, 11),
        prepared::<D>(2, (shape.inner + shape.randomness) / m, 12),
    )
}

/// `N`: the inner rows `A w − G t̂ = 0`, the outer rows
/// `B (t̂ ‖ r_a) = u`, and the pad opening.
fn setup<const D: usize>(
    table: SegmentTable,
    pads: &LatticePadCommitment<F, D>,
    u: Vec<CyclotomicRing<F, D>>,
    t_p: Vec<CyclotomicRing<F, D>>,
) -> SigmaSetup<F, D> {
    let shape = shape::<D>();
    let c = CHALLENGE_RING_DIM;
    let witness_end = (shape.public + shape.witness) * c;
    let inner_end = witness_end + shape.inner * c;
    let residual_end = inner_end + shape.randomness * c;
    let (a, b) = residual_matrices::<D>();
    let linear = RingLinearMap::new(vec![
        LinearBlock {
            matrix: a,
            input: 0..witness_end,
            gadget: Some(GadgetTerm {
                input: witness_end..inner_end,
                log_basis: 4,
                levels: FIELD_LEVELS,
            }),
            target: vec![CyclotomicRing::zero()],
        },
        LinearBlock {
            matrix: b,
            input: witness_end..residual_end,
            gadget: None,
            target: u,
        },
        pads.linear_block(residual_end, t_p),
    ])
    .unwrap();
    SigmaSetup::new(
        table,
        linear,
        pads.l_row(residual_end),
        1,
        &stream(1 << 18, 31),
    )
    .unwrap()
}

/// Five base messages and one auxiliary product: rows `m2 + m3 = sum`,
/// `w̃(r) = m4`, and `m0 · m1 = aux`.
fn instance<const D: usize>(seed: u64, witness_scale: i64) -> Instance<D> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let shape = shape::<D>();
    let c = CHALLENGE_RING_DIM;
    let layout = PadLayout::new(BASE_PADS, 1).unwrap();
    let pads: Vec<F> = (0..layout.num_pads).map(|_| F::random(&mut rng)).collect();
    let mut messages: Vec<F> = (0..BASE_PADS).map(|_| F::random(&mut rng)).collect();

    let w: Vec<i64> = digits(&mut rng, (shape.public + shape.witness) * c)
        .into_iter()
        .map(|digit| (digit * witness_scale).clamp(-8, 7))
        .collect();
    let num_vars = w.len().next_power_of_two().trailing_zeros() as usize;
    let point: Vec<F> = (0..num_vars).map(|_| F::random(&mut rng)).collect();
    let eq = EqPolynomial::new(point.clone()).evaluations();
    messages[4] = w.iter().zip(&eq).map(|(&w, e)| F::from_i64(w) * *e).sum();
    let auxiliary = messages[0] * messages[1];
    let masked: Vec<F> = messages
        .iter()
        .chain(std::iter::once(&auxiliary))
        .zip(&pads)
        .map(|(m, p)| *m + *p)
        .collect();
    let slot = |index: usize| AffineRow::masked(masked[index], PadIndex(index));

    let mut relation = ResidualRelation::new(layout, 1);
    relation.affine.push(
        slot(2)
            .add_scaled(&slot(3), F::one())
            .add_scaled(&AffineRow::constant(-(messages[2] + messages[3])), F::one()),
    );
    relation
        .affine
        .push(AffineRow::variable(ResidualVar::Witness(0)).add_scaled(&slot(4), -F::one()));
    relation.products.push(ProductRow {
        x: MaskedOperand {
            masked: masked[0],
            pad: PadIndex(0),
        },
        y: slot(1),
        z: slot(5),
    });

    let placement = PadPlacement::new(relation.layout.num_pads, vec![PadIndex(0)]).unwrap();
    let pad_commitment = pad_commitment::<D>(placement.clone());
    let g = sample_zero_constant_term::<F, CHALLENGE_RING_DIM, _>(&mut rng);
    let r_p = sample_box_digits(&mut rng, 2 * D, 4).unwrap();
    let (t_p, opening) = pad_commitment.commit(&pads, g, &r_p).unwrap();

    let (a, b) = residual_matrices::<D>();
    let t = a.apply_centered(&w, 8).unwrap();
    let t_hat: Vec<i64> = t.iter().flat_map(ring_digits::<D>).collect();
    let r_a = digits(&mut rng, shape.randomness * c);
    let outer_input: Vec<i64> = t_hat.iter().chain(&r_a).copied().collect();
    let u = b.apply_centered(&outer_input, 8).unwrap();

    let s: Vec<i64> = w
        .iter()
        .chain(&t_hat)
        .chain(&r_a)
        .chain(&opening.digits)
        .copied()
        .collect();
    Instance {
        relation,
        claims: EvaluationClaims {
            points: vec![point],
        },
        placement,
        s,
        u,
        t_p,
    }
}

/// Proves `instance` with `response`; returns the proof and attempt count.
fn prove<const D: usize, P: ResponseProver<F, D>>(
    instance: &Instance<D>,
    response: &P,
    seed: u64,
) -> Result<(Vec<u8>, usize), ZkError> {
    let (setup, lifted) = prover_setup(instance)?;
    prove_with(instance, &setup, &lifted, response, seed)
}

/// The prover's setup and lifted relation for `instance`.
fn prover_setup<const D: usize>(
    instance: &Instance<D>,
) -> Result<(SigmaSetup<F, D>, LiftedRelation<F>), ZkError> {
    let pads = pad_commitment::<D>(instance.placement.clone());
    let table = table(&pads);
    let lifted = LiftedRelation::new(
        &instance.relation,
        &instance.placement,
        &table,
        &instance.claims,
    )?;
    Ok((
        setup(table, &pads, instance.u.clone(), instance.t_p.clone()),
        lifted,
    ))
}

fn prove_with<const D: usize, P: ResponseProver<F, D>>(
    instance: &Instance<D>,
    setup: &SigmaSetup<F, D>,
    lifted: &LiftedRelation<F>,
    response: &P,
    seed: u64,
) -> Result<(Vec<u8>, usize), ZkError> {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, LABEL);
    for ring in instance.u.iter().chain(&instance.t_p) {
        transcript.send_all(ring.coefficients());
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let attempts = prove_sigma(
        setup,
        lifted,
        &instance.s,
        &mut transcript,
        &mut rng,
        response,
    )?;
    Ok((transcript.finish(), attempts))
}

/// Verifies a proof of `instance`'s public relation.
fn verify<const D: usize>(instance: &Instance<D>, proof: &[u8]) -> Result<(), ZkError> {
    let pads = pad_commitment::<D>(instance.placement.clone());
    let table = table(&pads);
    let lifted = LiftedRelation::new(
        &instance.relation,
        pads.placement(),
        &table,
        &instance.claims,
    )?;
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, proof);
    let mut receive = |count: usize| -> Result<Vec<CyclotomicRing<F, D>>, ZkError> {
        (0..count)
            .map(|_| Ok(CyclotomicRing::from_slice(&transcript.receive_n::<F>(D)?)))
            .collect()
    };
    let u = receive(instance.u.len())?;
    let t_p = receive(instance.t_p.len())?;
    let setup = setup(table, &pads, u, t_p);
    verify_sigma(&setup, &lifted, &mut transcript, &CheckingResponseProver)?;
    Ok(transcript.finish()?)
}

#[test]
fn honest_proofs_verify_over_two_ring_dimensions() {
    let small = instance::<64>(1, 1);
    let (proof, _) = prove(&small, &CheckingResponseProver, 2).unwrap();
    verify(&small, &proof).unwrap();
    let large = instance::<256>(3, 1);
    let (proof, _) = prove(&large, &CheckingResponseProver, 4).unwrap();
    verify(&large, &proof).unwrap();
}

/// `M = exp(t/α + 1/(2α²))`, `t = √(2·137·ln 2)`: the expected attempt count
/// when the box holds essentially all of `π_0`.
fn predicted_repetition_rate() -> f64 {
    let t = (2.0 * 137.0 * LN_2).sqrt();
    (t / 16.0 + 1.0 / 512.0).exp()
}

#[test]
fn abort_rate_and_response_distribution_match_the_parameters() {
    const PROOFS: usize = 1_000;
    let instance = instance::<64>(5, 1);
    let capture = CapturingResponseProver::default();
    let (setup, lifted) = prover_setup(&instance).unwrap();
    let mut attempts = Vec::with_capacity(PROOFS);
    for seed in 0..PROOFS as u64 {
        attempts.push(
            prove_with(&instance, &setup, &lifted, &capture, 1_000 + seed)
                .unwrap()
                .1 as f64,
        );
    }
    // Geometric attempts with acceptance 1/M: mean M, variance (1 − 1/M) M².
    let m = predicted_repetition_rate();
    let mean = attempts.iter().sum::<f64>() / PROOFS as f64;
    let standard_error = ((1.0 - 1.0 / m) * m * m / PROOFS as f64).sqrt();
    assert!(
        (mean - m).abs() < 5.0 * standard_error,
        "mean attempts {mean:.4}, predicted M = {m:.4} ± {standard_error:.4}"
    );

    // On the Gaussian witness segment, the accepted z has the configured
    // Gaussian's mean, variance and kurtosis. (Independence from the shift
    // v = c·s is not testable here: ‖v‖_σ ≤ 1/α makes its effect on z far
    // below sampling error. M5a tests the rule on large shifts.)
    let pads = pad_commitment::<64>(instance.placement.clone());
    let table = table(&pads);
    let params = SigmaParams::derive(&table).unwrap();
    let block = params.blocks()[0];
    let start = table.starts()[block.segment].0;
    let sigma = block.width as f64 / (2.0 * LN_2).sqrt();
    let samples: Vec<f64> = capture
        .accepted
        .borrow()
        .iter()
        .flat_map(|(z, _)| {
            z[start..start + block.len]
                .iter()
                .map(|&z| z as f64 / sigma)
        })
        .collect();
    let n = samples.len() as f64;
    let moment = |k: i32| samples.iter().map(|z| z.powi(k)).sum::<f64>() / n;
    assert!(moment(1).abs() < 5.0 / n.sqrt(), "mean {}", moment(1));
    assert!(
        (moment(2) - 1.0).abs() < 5.0 * (2.0 / n).sqrt(),
        "variance ratio {}",
        moment(2)
    );
    assert!(
        (moment(4) - 3.0).abs() < 5.0 * (96.0 / n).sqrt(),
        "fourth moment {}",
        moment(4)
    );
}

/// Flips one byte of the proof at `position` and expects rejection.
fn assert_tamper_rejected<const D: usize>(instance: &Instance<D>, proof: &[u8], position: usize) {
    let mut tampered = proof.to_vec();
    tampered[position] ^= 1;
    assert!(
        verify(instance, &tampered).is_err(),
        "tampered byte {position} was accepted"
    );
}

#[test]
fn tampered_proofs_are_rejected() {
    let instance = instance::<64>(7, 1);
    let (proof, _) = prove(&instance, &CheckingResponseProver, 8).unwrap();
    let images = (instance.u.len() + instance.t_p.len()) * 64 * 16;
    let t_d = 64 * 16;
    // An image byte, a t_D byte, a nonce byte, and a response digit.
    for position in [3, images + 5, images + t_d + 7, images + t_d + 32 + 11] {
        assert_tamper_rejected(&instance, &proof, position);
    }
    // A shortened proof.
    assert!(verify(&instance, &proof[..proof.len() - 1]).is_err());
}

fn assert_prover_rejects<const D: usize>(
    instance: &Instance<D>,
    expected: fn(&WhiteoutError) -> bool,
) {
    match prove(instance, &CheckingResponseProver, 9) {
        Err(ZkError::Whiteout(error)) => assert!(expected(&error), "unexpected {error:?}"),
        other => panic!("expected a Whiteout self-check failure, got {other:?}"),
    }
}

#[test]
fn malformed_witnesses_fail_the_self_checks() {
    // A violated affine row.
    let mut affine = instance::<64>(10, 1);
    affine.relation.affine[0].constant += F::one();
    assert_prover_rejects(&affine, |error| {
        matches!(error, WhiteoutError::ResidualUnsatisfied { row: 0 })
    });
    // A violated product row.
    let mut product = instance::<64>(11, 1);
    product.relation.products[0].x.masked += F::one();
    assert_prover_rejects(&product, |error| {
        matches!(error, WhiteoutError::ResidualUnsatisfied { .. })
    });
    // A non-constant operand digit polynomial (a scalarity violation) fails
    // the shape check first.
    let mut operand = instance::<64>(12, 1);
    let operand_start = {
        let pads = pad_commitment::<64>(operand.placement.clone());
        let table = table(&pads);
        let index = table.unique(SegmentRole::OperandDigits).unwrap();
        table.starts()[index].0
    };
    operand.s[operand_start + 1] = 1;
    assert_prover_rejects(&operand, |error| {
        matches!(error, WhiteoutError::WitnessOutsideAlphabet)
    });
    // A digit outside its alphabet.
    let mut alphabet = instance::<64>(13, 1);
    alphabet.s[150] = 9;
    assert_prover_rejects(&alphabet, |error| {
        matches!(error, WhiteoutError::WitnessOutsideAlphabet)
    });
    // ct(g) ≠ 0: the last row.
    let mut mask = instance::<64>(14, 1);
    let g_start = {
        let pads = pad_commitment::<64>(mask.placement.clone());
        let table = table(&pads);
        let index = table.unique(SegmentRole::PadDigits).unwrap();
        table.starts()[index].0 + mask.placement.regular_groups() * 32 * 64
    };
    mask.s[g_start] = if mask.s[g_start] == 7 {
        6
    } else {
        mask.s[g_start] + 1
    };
    assert_prover_rejects(&mask, |error| {
        matches!(
            error,
            WhiteoutError::ResidualUnsatisfied { .. } | WhiteoutError::ResidualOpeningMismatch
        )
    });
    // A witness that does not open its commitment.
    let mut opening = instance::<64>(15, 1);
    opening.u[0] += CyclotomicRing::one();
    assert_prover_rejects(&opening, |error| {
        matches!(error, WhiteoutError::ResidualOpeningMismatch)
    });
}

/// The masking layer's shape routine names the same operand pads as the
/// product rows its row routine emits.
#[test]
fn operand_shape_routine_matches_the_rows() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        A,
        B,
        C,
    }
    let expr: FactoredExpr<F, Opening, (), ()> = FactoredExpr::product(vec![
        opening(Opening::A),
        opening(Opening::B),
        opening(Opening::C),
    ]);
    let layout = PadLayout::new(3, 2).unwrap();
    let mut shape = OperandShapeDomain::new(layout.auxiliary.clone());
    let _ = lower_factored(
        &expr,
        &mut shape,
        &mut |id| {
            Ok(Some(PadIndex(match id {
                Opening::A => 0,
                Opening::B => 1,
                Opening::C => 2,
            })))
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<F, ZkError> { Ok(F::zero()) },
    )
    .unwrap();
    let mut relation = ResidualRelation::new(layout, 0);
    let auxiliary = [F::from_u64(7), F::from_u64(9)];
    let mut rows = RowsDomain::new(&mut relation, &auxiliary);
    let _ = lower_factored(
        &expr,
        &mut rows,
        &mut |id| {
            let index = match id {
                Opening::A => 0,
                Opening::B => 1,
                Opening::C => 2,
            };
            Ok(AffineRow::masked(
                F::from_u64(index as u64),
                PadIndex(index),
            ))
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<F, ZkError> { Ok(F::zero()) },
    )
    .unwrap();
    rows.finish().unwrap();
    assert_eq!(shape.operands, PadPlacement::operands_of(&relation));
    assert_eq!(shape.operands, vec![PadIndex(0), PadIndex(3)]);
}
