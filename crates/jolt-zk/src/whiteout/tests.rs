//! The Whiteout Σ-protocol on synthetic residual relations.
//!
//! - completeness at two ring dimensions;
//! - the verifier's rejection of every violated row class, with the
//!   prover's self-checks bypassed;
//! - each self-check firing on its own case;
//! - an independent oracle for `ct Q_μ(1, s)`;
//! - the pinned row `L`, algebraic properties, the abort rate at two α, the
//!   response's moments, and tamper rejection.

use std::f64::consts::LN_2;
use std::sync::Mutex;

use jolt_algebra::{CyclotomicRing, Field, One, Ring, Zero};
use jolt_crypto::lattice::{MatrixShape, MatrixView, CHALLENGE_RING_DIM};
use jolt_poly::EqPolynomial;
use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, Sponge, VerifierTranscript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::challenge::{challenge_ring, challenge_times_i64, challenge_times_ring};
use super::fixture::{
    digits, layout, lengths, pad_commitment, residual_matrices, residual_segments, stream,
    Instance, F, PAD_RANDOMNESS_COLUMNS, PAD_RANK,
};
use super::labels::RowLabels;
use super::lift::constant_ring;
use super::params::{ALPHA, EPSILON_BITS};
use super::placement::{PadPlace, PadPlacement, OPERAND_LEVELS, PAD_LEVELS};
use super::response::{ResponseProver, ResponseStatement, ResponseWitness};
use super::segments::{Segment, SegmentMask, SegmentRole, SegmentShape};
use super::sigma::{prove_sigma_with, take_rejected, SelfChecks};
use super::{
    verify_sigma, LiftedRelation, ResidualBlock, ResponseSegment, SegmentTable, SigmaParams,
    SigmaSetup, WhiteoutError,
};
use crate::masking::{AffineRow, PadIndex, ResidualRelation, ResidualVar, ZkError};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-zk/whiteout-sigma");
const LABEL: &[u8] = b"whiteout-sigma";

/// Sends `e` in the clear; the verifier checks the statement directly.
struct CheckingResponseProver;

impl<const D: usize> ResponseProver<F, D> for CheckingResponseProver {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        SendingResponseProver.prove_response(statement, witness, transcript)
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

/// Sends `e` without checking it, so the verifier's check is what rejects.
struct SendingResponseProver;

impl<const D: usize> ResponseProver<F, D> for SendingResponseProver {
    fn prove_response<H: Sponge>(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let bytes: Vec<u8> = witness.digits.iter().map(|&digit| digit as u8).collect();
        transcript.send_bytes(&bytes);
        Ok(())
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        CheckingResponseProver.verify_response(statement, transcript)
    }
}

/// Records each accepted response's Gaussian-block coefficients.
#[derive(Default)]
struct CapturingResponseProver {
    accepted: Mutex<Vec<Vec<i64>>>,
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
        self.accepted.lock().unwrap().push(z);
        SendingResponseProver.prove_response(statement, witness, transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        CheckingResponseProver.verify_response(statement, transcript)
    }
}

/// Proves `instance` under `setup`, sending the images first.
fn prove<const D: usize, P: ResponseProver<F, D> + Sync>(
    instance: &Instance<D>,
    setup: &SigmaSetup<F, D>,
    lifted: &LiftedRelation<F>,
    response: &P,
    checks: SelfChecks,
    seed: u64,
) -> Result<(Vec<u8>, usize), ZkError> {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, LABEL);
    for ring in instance.u.iter().chain(&instance.t_p) {
        transcript.send_all(ring.coefficients());
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let attempts = prove_sigma_with(
        setup,
        lifted,
        &instance.s,
        &mut transcript,
        &mut rng,
        response,
        checks,
    )?;
    Ok((transcript.finish(), attempts))
}

fn prove_checked<const D: usize>(instance: &Instance<D>, seed: u64) -> Result<Vec<u8>, ZkError> {
    let setup = instance.setup();
    let lifted = setup.lift(&instance.relation, &instance.claims)?;
    prove(
        instance,
        &setup,
        &lifted,
        &CheckingResponseProver,
        SelfChecks::Run,
        seed,
    )
    .map(|(proof, _)| proof)
}

/// Verifies a proof of `relation`, reading the images from the proof.
fn verify<const D: usize>(
    instance: &Instance<D>,
    relation: &ResidualRelation<F>,
    proof: &[u8],
) -> Result<(), ZkError> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, proof);
    let mut receive = |count: usize| -> Result<Vec<CyclotomicRing<F, D>>, ZkError> {
        (0..count)
            .map(|_| Ok(CyclotomicRing::from_slice(&transcript.receive_n::<F>(D)?)))
            .collect()
    };
    let received = Instance {
        relation: relation.clone(),
        claims: instance.claims.clone(),
        s: Vec::new(),
        u: receive(instance.u.len())?,
        t_p: receive(instance.t_p.len())?,
        scale: instance.scale,
    };
    let setup = received.setup();
    let lifted = setup.lift(relation, &instance.claims)?;
    verify_sigma(&setup, &lifted, &mut transcript, &CheckingResponseProver)?;
    Ok(transcript.finish()?)
}

fn whiteout_error(error: &ZkError) -> Option<&WhiteoutError> {
    if let ZkError::Whiteout(source) = error {
        source.downcast_ref::<WhiteoutError>()
    } else {
        None
    }
}

#[test]
fn honest_proofs_verify_over_two_ring_dimensions() {
    let small = Instance::<64>::new(1);
    verify(&small, &small.relation, &prove_checked(&small, 2).unwrap()).unwrap();
    let large = Instance::<256>::new(3);
    verify(&large, &large.relation, &prove_checked(&large, 4).unwrap()).unwrap();
}

/// The cases each self-check exists for, with the prover's checks
/// bypassed: the verifier rejects every one at Item 5, `ct(h) = 0`.
#[test]
fn verifier_rejects_each_violated_row_class_at_item_five() {
    let cases: Vec<(&str, Instance<64>)> = vec![
        ("affine row", {
            let mut instance = Instance::new(20);
            instance.relation.affine[0].constant += F::one();
            instance
        }),
        ("product row", {
            let mut instance = Instance::new(21);
            instance.relation.products[0].x.masked += F::one();
            instance
        }),
        ("scalarity row", {
            let mut instance = Instance::new(22);
            let operand = instance.start(SegmentRole::OperandDigits);
            instance.s[operand + 1] = 1;
            instance.reopen();
            instance
        }),
        ("ct(g)", {
            let mut instance = Instance::new(23);
            let pad = instance.start(SegmentRole::PadDigits);
            let g = pad + PadPlacement::new(&layout()).unwrap().g_ring(0) * CHALLENGE_RING_DIM;
            instance.s[g] = if instance.s[g] == 7 {
                6
            } else {
                instance.s[g] + 1
            };
            instance.reopen();
            instance
        }),
    ];
    for (name, instance) in cases {
        let setup = instance.setup();
        let lifted = setup.lift(&instance.relation, &instance.claims).unwrap();
        let (proof, _) = prove(
            &instance,
            &setup,
            &lifted,
            &SendingResponseProver,
            SelfChecks::Skip,
            7,
        )
        .unwrap();
        let error = verify(&instance, &instance.relation, &proof).unwrap_err();
        assert_eq!(
            whiteout_error(&error),
            Some(&WhiteoutError::ResponseRejected("ct(h)")),
            "{name}: {error:?}"
        );
    }
}

/// Each self-check fires on its own case, with the exact row or variant.
#[test]
fn each_self_check_fires_on_its_own_case() {
    let affine_rows = 2;
    let scalarity_rows = 63;
    let check = |instance: &Instance<64>, expected: WhiteoutError| {
        let error = prove_checked(instance, 9).unwrap_err();
        assert_eq!(whiteout_error(&error), Some(&expected), "{error:?}");
    };

    let mut affine = Instance::<64>::new(10);
    affine.relation.affine[0].constant += F::one();
    check(&affine, WhiteoutError::ResidualUnsatisfied { row: 0 });

    // Coefficient 1 of operand 0's digit polynomial 0: scalarity row r = 1.
    let mut scalarity = Instance::<64>::new(11);
    let operand = scalarity.start(SegmentRole::OperandDigits);
    scalarity.s[operand + 1] = 1;
    scalarity.reopen();
    check(
        &scalarity,
        WhiteoutError::ResidualUnsatisfied { row: affine_rows },
    );

    let mut product = Instance::<64>::new(12);
    product.relation.products[0].x.masked += F::one();
    check(
        &product,
        WhiteoutError::ResidualUnsatisfied {
            row: affine_rows + scalarity_rows,
        },
    );

    let mut mask = Instance::<64>::new(13);
    let pad = mask.start(SegmentRole::PadDigits);
    let g = pad + PadPlacement::new(&layout()).unwrap().g_ring(0) * CHALLENGE_RING_DIM;
    mask.s[g] = if mask.s[g] == 7 { 6 } else { mask.s[g] + 1 };
    mask.reopen();
    check(
        &mask,
        WhiteoutError::ResidualUnsatisfied {
            row: affine_rows + scalarity_rows + 1,
        },
    );

    let mut opening = Instance::<64>::new(14);
    opening.u[0] += CyclotomicRing::one();
    check(&opening, WhiteoutError::ResidualOpeningMismatch);

    // A randomness digit outside its alphabet, with the images recomputed:
    // no row and no opening is affected.
    let mut alphabet = Instance::<64>::new(15);
    let randomness = alphabet.start(SegmentRole::ResidualRandomness);
    alphabet.s[randomness] = 9;
    alphabet.reopen();
    check(&alphabet, WhiteoutError::WitnessOutsideAlphabet);
}

/// `ct Q_μ(1, s)` on arbitrary `s` against an independent decoding:
/// `Σ μ_row · row(s) + ct(g)`, where a product row's term includes the
/// difference `Σ_r P_x[r]·Y_{64−r}` that a non-constant operand adds. At
/// `D = 256` the decoding reads challenge-ring element `E·m + k` as native
/// coefficients `m·i + k` of element `E` (the stride map of §0).
#[test]
fn quadratic_constant_term_matches_an_independent_decoding() {
    check_quadratic_against_decoding::<64>(30);
    check_quadratic_against_decoding::<256>(32);
}

fn check_quadratic_against_decoding<const D: usize>(seed: u64) {
    let instance = Instance::<D>::new(seed);
    let setup = instance.setup();
    let lifted = setup.lift(&instance.relation, &instance.claims).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let rho: Vec<F> = (0..lifted.labels().num_challenges())
        .map(|_| F::random(&mut rng))
        .collect();
    // Arbitrary digits everywhere, including non-constant operand digits
    // and a g with a nonzero constant term.
    let s: Vec<i64> = digits(&mut rng, instance.s.len());

    let aggregated = lifted.aggregate(&rho).unwrap();
    let tables = lifted.claim_tables().unwrap();
    let parts = aggregated.linear_parts(&tables, &s, false).unwrap();
    let actual = aggregated
        .quadratic(&constant_ring(F::one()), &parts)
        .constant_term();

    // Independent decoding.
    let placement = PadPlacement::new(&layout()).unwrap();
    let pad = instance.start(SegmentRole::PadDigits);
    let operand_start = instance.start(SegmentRole::OperandDigits);
    // Challenge-ring element `ring` (counted from the start of s) and its
    // coefficient `i`.
    let m = D / 64;
    let coefficient = |ring: usize, i: usize| F::from_i64(s[(ring / m) * D + m * i + ring % m]);
    let radix16 = F::from_u64(16);
    let radix256 = F::from_u64(256);
    let operand_poly = |operand: usize| -> Vec<F> {
        (0..64)
            .map(|i| {
                (0..OPERAND_LEVELS).rev().fold(F::zero(), |acc, level| {
                    acc * radix256 + coefficient(operand_start / 64 + operand * 16 + level, i)
                })
            })
            .collect()
    };
    let pad_value = |index: usize| match placement.place(PadIndex(index)).unwrap() {
        PadPlace::Regular { group, position } => {
            (0..PAD_LEVELS).rev().fold(F::zero(), |acc, level| {
                acc * radix16 + coefficient(pad / 64 + group * 32 + level, position)
            })
        }
        PadPlace::Operand(operand) => operand_poly(operand)[0],
    };
    let w_len: usize = lengths::<D>(1)[..2].iter().sum::<usize>() * 64;
    let eq = EqPolynomial::new(instance.claims.points[0].clone()).evaluations();
    let claim: F = s[..w_len]
        .iter()
        .zip(&eq)
        .map(|(&w, e)| F::from_i64(w) * *e)
        .sum();
    let row_value = |row: &AffineRow<F>| {
        row.terms
            .iter()
            .fold(row.constant, |acc, &(variable, weight)| {
                acc + weight
                    * match variable {
                        ResidualVar::Pad(pad) => pad_value(pad.0),
                        ResidualVar::Witness(_) => claim,
                    }
            })
    };
    // Y(1, s)'s ring coefficients, from Y's field weights: coefficient t of
    // σ(ℓ̂)·s_k is Σ_i ℓ̂_i s_k[i + t], negated past X^64.
    let lift_ring = |row: &AffineRow<F>| -> Vec<F> {
        // Direct correlation: coefficient t = Σ_k Σ_i ℓ̂_{k,i} s_k[i + t].
        let mut weights: Vec<(usize, usize, F)> = Vec::new();
        for &(variable, weight) in &row.terms {
            match variable {
                ResidualVar::Pad(pad_index) => match placement.place(pad_index).unwrap() {
                    PadPlace::Regular { group, position } => {
                        let mut power = weight;
                        for level in 0..PAD_LEVELS {
                            weights.push((pad / 64 + group * 32 + level, position, power));
                            power *= radix16;
                        }
                    }
                    PadPlace::Operand(operand) => {
                        let mut power = weight;
                        for level in 0..OPERAND_LEVELS {
                            weights.push((operand_start / 64 + operand * 16 + level, 0, power));
                            power *= radix256;
                        }
                    }
                },
                ResidualVar::Witness(_) => {
                    for (n, e) in eq.iter().enumerate().take(w_len) {
                        let (element, j) = (n / D, n % D);
                        weights.push((element * m + j % m, j / m, *e * weight));
                    }
                }
            }
        }
        let mut ring = vec![F::zero(); 64];
        for (ring_index, i, weight) in weights {
            for (t, dst) in ring.iter_mut().enumerate() {
                let value = if i + t < 64 {
                    coefficient(ring_index, i + t)
                } else {
                    -coefficient(ring_index, i + t - 64)
                };
                *dst += weight * value;
            }
        }
        ring
    };
    let labels = lifted.labels();
    let weight = |label: usize| labels.weight(label, &rho).unwrap();
    let mut expected = F::zero();
    for (index, row) in instance.relation.affine.iter().enumerate() {
        expected += weight(labels.affine(index)) * row_value(row);
    }
    let p_x = operand_poly(0);
    for (r, coefficient) in p_x.iter().enumerate().skip(1) {
        expected += weight(RowLabels::scalarity(0, r)) * *coefficient;
    }
    for (index, product) in instance.relation.products.iter().enumerate() {
        let x = product.x.masked - p_x[0];
        let y = lift_ring(&product.y);
        let difference: F = (1..64).map(|r| p_x[r] * y[64 - r]).sum();
        let term = x * row_value(&product.y) - row_value(&product.z) + difference;
        expected += weight(labels.product(index)) * term;
    }
    expected += (0..PAD_LEVELS).rev().fold(F::zero(), |acc, level| {
        acc * radix16 + coefficient(pad / 64 + placement.g_ring(level), 0)
    });
    assert_eq!(actual, expected);
}

/// `L·r` is `π_0` of row `rank` of the pad view on the randomness columns.
#[test]
fn l_row_is_row_rank_of_the_pad_view() {
    let instance = Instance::<256>::new(40);
    let setup = instance.setup();
    let pads = pad_commitment::<256>();
    let width = pads.segments().iter().map(|s| s.len / 4).sum::<usize>();
    let data = stream(1 << 18, 21);
    let view = MatrixView::new(
        MatrixShape {
            rows: PAD_RANK + 1,
            width,
            ring_dim: 256,
        },
        &data,
    )
    .unwrap();
    let randomness = setup.table().unique(SegmentRole::PadRandomness).unwrap();
    let r = &instance.s[randomness.coefficients()];
    let mut row = CyclotomicRing::<F, 256>::zero();
    for (j, chunk) in r.chunks_exact(256).enumerate() {
        let entry = CyclotomicRing::<F, 256>::from_slice(
            view.entry(PAD_RANK, width - PAD_RANDOMNESS_COLUMNS + j)
                .unwrap(),
        );
        let value =
            CyclotomicRing::from_slice(&chunk.iter().map(|&v| F::from_i64(v)).collect::<Vec<_>>());
        row += entry * value;
    }
    let expected: Vec<F> = row.coefficients().iter().step_by(4).copied().collect();
    assert_eq!(
        setup
            .l_row()
            .apply(&instance.s, 8)
            .unwrap()
            .coefficients()
            .to_vec(),
        expected
    );
}

/// `Q_μ(c, z) = c² h + c B_μ + Q_μ(0, y)` for `z = c·s + y`, and
/// `N(c·s) = c·N(s)`. (θ is pinned by the statement check on every honest
/// proof and by the θ-digit tamper.)
#[test]
fn quadratic_polar_quotient_and_linearity_identities() {
    let instance = Instance::<256>::new(50);
    let setup = instance.setup();
    let lifted = setup.lift(&instance.relation, &instance.claims).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(51);
    let rho: Vec<F> = (0..lifted.labels().num_challenges())
        .map(|_| F::random(&mut rng))
        .collect();
    let aggregated = lifted.aggregate(&rho).unwrap();
    let tables = lifted.claim_tables().unwrap();
    let s = &instance.s;
    let y: Vec<i64> = (0..s.len())
        .map(|_| i64::from(rng.next_u32() % 2001) - 1000)
        .collect();
    let mut c = [0i8; 64];
    for coefficient in c.iter_mut().take(20) {
        *coefficient = (rng.next_u32() % 5) as i8 - 2;
    }
    let shift = challenge_times_i64(&c, 256, s).unwrap();
    let z: Vec<i64> = y.iter().zip(&shift).map(|(y, v)| y + v).collect();
    let c_ring = challenge_ring::<F>(&c);
    let parts_s = aggregated.linear_parts(&tables, s, false).unwrap();
    let parts_y = aggregated.linear_parts(&tables, &y, false).unwrap();
    let parts_z = aggregated.linear_parts(&tables, &z, false).unwrap();
    let h = aggregated.quadratic(&constant_ring(F::one()), &parts_s);
    assert_eq!(
        aggregated.quadratic(&c_ring, &parts_z),
        c_ring * c_ring * h
            + c_ring * aggregated.polar(&parts_s, &parts_y)
            + aggregated.quadratic_at_zero(&parts_y)
    );

    // N(c·s) = c·N(s), gadget block included.
    let n_cs = setup.linear().apply(&shift, 1 << 12).unwrap();
    let n_s = setup.linear().apply(s, 1 << 7).unwrap();
    for (lhs, rhs) in n_cs.iter().zip(&n_s) {
        assert_eq!(*lhs, challenge_times_ring(&c, rhs));
    }
}

#[test]
fn setups_reject_mixed_ring_dimensions() {
    let mut segments = residual_segments::<64>(1);
    segments[1].native_dim = 128;
    segments[1].len = 8;
    let (a, _) = residual_matrices::<64>(1);
    let result = SigmaSetup::new(
        segments,
        vec![ResidualBlock {
            matrix: a,
            input: 0..2,
            gadget: None,
            target: vec![CyclotomicRing::zero()],
        }],
        &pad_commitment::<64>(),
        vec![CyclotomicRing::zero(); PAD_RANK],
        1,
        &stream(1 << 18, 31),
    );
    assert!(matches!(result, Err(WhiteoutError::InvalidRelation(_))));
}

/// The expected attempt count `M = exp(t/α + 1/(2α²))`, `t = √(2 ε ln 2)`,
/// when the box holds essentially all of `π_0`.
fn predicted_repetition_rate(alpha: u32) -> f64 {
    let t = (2.0 * f64::from(EPSILON_BITS) * LN_2).sqrt();
    let alpha = f64::from(alpha);
    (t / alpha + 1.0 / (2.0 * alpha * alpha)).exp()
}

/// The mean attempt count over 300 proofs is within 4σ of `M`, at the
/// paper's α and at α = 10 (`M ≈ 3.99`), which a wrong rate would leave.
#[test]
fn abort_rate_matches_the_repetition_rate() {
    const PROOFS: u64 = 300;
    let instance = Instance::<64>::new(60);
    for alpha in [ALPHA, 10] {
        let setup = instance.setup().with_alpha(alpha).unwrap();
        let lifted = setup.lift(&instance.relation, &instance.claims).unwrap();
        let attempts: Vec<f64> = (0..PROOFS)
            .map(|seed| {
                prove(
                    &instance,
                    &setup,
                    &lifted,
                    &SendingResponseProver,
                    SelfChecks::Run,
                    seed,
                )
                .unwrap()
                .1 as f64
            })
            .collect();
        let m = predicted_repetition_rate(alpha);
        let mean = attempts.iter().sum::<f64>() / PROOFS as f64;
        let standard_error = ((1.0 - 1.0 / m) * m * m / PROOFS as f64).sqrt();
        assert!(
            (mean - m).abs() < 4.0 * standard_error,
            "α = {alpha}: mean attempts {mean:.3}, predicted M = {m:.3} ± {standard_error:.3}"
        );
    }
}

/// On the Gaussian witness block, the accepted `z` has the configured
/// Gaussian's mean, variance and fourth moment.
#[test]
fn accepted_responses_have_the_configured_moments() {
    let instance = Instance::<64>::new(61);
    let setup = instance.setup();
    let lifted = setup.lift(&instance.relation, &instance.claims).unwrap();
    let capture = CapturingResponseProver::default();
    for seed in 0..300 {
        let _ = prove(&instance, &setup, &lifted, &capture, SelfChecks::Run, seed).unwrap();
    }
    let params = SigmaParams::derive(setup.table()).unwrap();
    let block = params.blocks()[0];
    let start = setup.table().placed()[block.segment].coefficient;
    let sigma = block.width as f64 / (2.0 * LN_2).sqrt();
    let samples: Vec<f64> = capture
        .accepted
        .lock()
        .unwrap()
        .iter()
        .flat_map(|z| {
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

#[test]
fn tampered_proofs_are_rejected() {
    let instance = Instance::<64>::new(70);
    let proof = prove_checked(&instance, 71).unwrap();
    let u_bytes = instance.u.len() * 64 * 16;
    let images = u_bytes + instance.t_p.len() * 64 * 16;
    let t_d = 64 * 16;
    // A byte of u_a, of t_p, of t_D, of the nonce, of a z digit, and of a θ
    // digit (θ ends the proof).
    for position in [
        3,
        u_bytes + 5,
        images + 7,
        images + t_d + 9,
        images + t_d + 32 + 11,
        proof.len() - 5,
    ] {
        let mut tampered = proof.clone();
        tampered[position] ^= 1;
        assert!(
            verify(&instance, &instance.relation, &tampered).is_err(),
            "tampered byte {position} was accepted"
        );
    }
    assert!(verify(&instance, &instance.relation, &proof[..proof.len() - 1]).is_err());

    // The verifier's relation, changed in one constant, rejects.
    let mut relation = instance.relation.clone();
    relation.affine[0].constant += F::one();
    assert!(verify(&instance, &relation, &proof).is_err());
}

/// A rejected attempt's `(t_D, n)` with the accepted `z` does not verify.
#[test]
fn rejected_attempts_cannot_be_substituted() {
    let instance = Instance::<64>::new(80);
    let images = (instance.u.len() + instance.t_p.len()) * 64 * 16;
    let mut seed = 81;
    let (proof, rejected) = loop {
        let _ = take_rejected();
        let setup = instance.setup();
        let lifted = setup.lift(&instance.relation, &instance.claims).unwrap();
        let (proof, attempts) = prove(
            &instance,
            &setup,
            &lifted,
            &CheckingResponseProver,
            SelfChecks::Run,
            seed,
        )
        .unwrap();
        if attempts > 1 {
            break (proof, take_rejected().remove(0));
        }
        seed += 1;
    };
    let (t_d, nonce) = rejected;
    let mut substituted = proof.clone();
    substituted[images..images + t_d.len()].copy_from_slice(&t_d);
    substituted[images + t_d.len()..images + t_d.len() + 32].copy_from_slice(&nonce);
    assert_ne!(substituted, proof);
    assert!(verify(&instance, &instance.relation, &substituted).is_err());
}

#[test]
fn oversized_tables_are_rejected_before_the_lift_overflows() {
    let table = SegmentTable::new(residual_segments::<64>(1)).unwrap();
    let widths = vec![1u64 << 31; 3];
    // Widths this large need at least nine digits: past the i32 lift.
    assert!(matches!(
        SigmaParams::new(&table, &widths),
        Err(WhiteoutError::ResponseBoundTooLarge | WhiteoutError::RejectionArithmetic)
    ));
    let big = SegmentTable::new(vec![Segment {
        role: SegmentRole::Witness,
        len: 1 << 20,
        native_dim: 64,
        log_basis: 4,
        shape: SegmentShape::Dense,
        mask: SegmentMask::Gaussian,
    }])
    .unwrap();
    assert_eq!(
        SigmaParams::derive(&big),
        Err(WhiteoutError::ResponseBoundTooLarge)
    );
}
