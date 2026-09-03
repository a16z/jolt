#![expect(clippy::expect_used, reason = "measurement assertions may panic")]
#![expect(
    clippy::indexing_slicing,
    reason = "fixed protocol shapes are checked before indexing"
)]
#![expect(
    clippy::print_stdout,
    reason = "the measurement bench emits its result table"
)]
#![expect(clippy::unwrap_used, reason = "measurement assertions may panic")]

use std::time::Instant;

use ark_bn254::{
    Bn254, Config as Bn254Config, Fq, Fq12, Fr as ArkScalar, G1Projective, G2Projective,
};
use ark_ec::bn::BnConfig;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, CyclotomicMultSubgroup, Field as ArkField, MontFp, One, PrimeField};
use dory::backends::arkworks::{ArkDoryProof, ArkFr, ArkGT, ArkworksVerifierSetup, BN254};
use dory::primitives::arithmetic::{Field as DoryField, Group as DoryGroup};
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_crypto::Bn254GT;
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_field::{Field, Fr};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::domain::{Label, LabelWithCount};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: usize = 22;
const NUM_COMMITMENTS: usize = 41;
const WINDOWS: [usize; 4] = [4, 6, 7, 8];
const SCALAR_BITS: usize = ArkScalar::MODULUS_BIT_SIZE as usize;

mod g2_constants {
    include!("../../jolt-crypto/src/ec/bn254/glv/constants.rs");
}

use g2_constants::POWER_OF_2_DECOMPOSITIONS;

#[derive(Clone, Copy, Debug, Default)]
struct GtOps {
    mul: usize,
    cyclotomic_sqr: usize,
    generic_sqr: usize,
}

impl GtOps {
    fn fq_products(self) -> usize {
        54 * self.mul + 18 * self.cyclotomic_sqr + 36 * self.generic_sqr
    }

    fn rows(self) -> usize {
        12 * (self.mul + self.cyclotomic_sqr + self.generic_sqr)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EcOps {
    mixed_add: usize,
    projective_add: usize,
    double: usize,
}

impl EcOps {
    fn g1_fq_products(self) -> usize {
        11 * self.mixed_add + 15 * self.projective_add + 7 * self.double
    }

    fn g2_fq_products(self) -> usize {
        30 * self.mixed_add + 41 * self.projective_add + 17 * self.double
    }
}

#[derive(Clone, Debug)]
struct SignedMagnitude {
    magnitude: BigUint,
    positive: bool,
}

#[derive(Clone)]
struct Challenges {
    beta: Vec<ArkFr>,
    alpha: Vec<ArkFr>,
    gamma: ArkFr,
    d: ArkFr,
}

struct Relation {
    bases: Vec<ArkGT>,
    scalars: Vec<ArkFr>,
    names: Vec<String>,
    lhs: ArkGT,
    e1_bases: Vec<G1Projective>,
    e1_scalars: Vec<ArkFr>,
    e2_bases: Vec<G2Projective>,
    e2_scalars: Vec<ArkFr>,
}

struct TranscriptAdapter<'a, T: Transcript<Challenge = Fr>> {
    transcript: &'a mut T,
}

impl<T: Transcript<Challenge = Fr>> DoryTranscript for TranscriptAdapter<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript
            .append(&LabelWithCount(b"dory_bytes", bytes.len() as u64));
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], value: &ArkFr) {
        self.transcript.append(&Label(b"dory_field"));
        ark_to_jolt_fr(*value).append_to_transcript(self.transcript);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], value: &G) {
        let mut bytes = Vec::new();
        value.serialize_compressed(&mut bytes).unwrap();
        self.transcript
            .append(&LabelWithCount(b"dory_group", bytes.len() as u64));
        self.transcript.append_bytes(&bytes);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], value: &S) {
        let mut bytes = Vec::new();
        value.serialize_compressed(&mut bytes).unwrap();
        self.transcript
            .append(&LabelWithCount(b"dory_serde", bytes.len() as u64));
        self.transcript.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        jolt_to_ark_fr(self.transcript.challenge_scalar())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("dory-pcs does not reset the transcript")
    }
}

fn jolt_to_ark_fr(value: Fr) -> ArkFr {
    ArkFr(value.into())
}

fn ark_to_jolt_fr(value: ArkFr) -> Fr {
    value.0.into()
}

fn inv(value: ArkFr) -> ArkFr {
    ArkFr(
        value
            .0
            .inverse()
            .expect("transcript challenges are nonzero"),
    )
}

fn replay_transcript(proof: &ArkDoryProof, label: &'static [u8]) -> Challenges {
    let mut transcript = Blake2bTranscript::new(label);
    let mut transcript = TranscriptAdapter {
        transcript: &mut transcript,
    };
    let vmv = &proof.vmv_message;
    transcript.append_serde(b"vmv_c", &vmv.c);
    transcript.append_serde(b"vmv_d2", &vmv.d2);
    transcript.append_serde(b"vmv_e1", &vmv.e1);

    let mut beta = Vec::with_capacity(proof.sigma);
    let mut alpha = Vec::with_capacity(proof.sigma);
    for (first, second) in proof.first_messages.iter().zip(&proof.second_messages) {
        transcript.append_serde(b"d1_left", &first.d1_left);
        transcript.append_serde(b"d1_right", &first.d1_right);
        transcript.append_serde(b"d2_left", &first.d2_left);
        transcript.append_serde(b"d2_right", &first.d2_right);
        transcript.append_serde(b"e1_beta", &first.e1_beta);
        transcript.append_serde(b"e2_beta", &first.e2_beta);
        beta.push(transcript.challenge_scalar(b"beta"));

        transcript.append_serde(b"c_plus", &second.c_plus);
        transcript.append_serde(b"c_minus", &second.c_minus);
        transcript.append_serde(b"e1_plus", &second.e1_plus);
        transcript.append_serde(b"e1_minus", &second.e1_minus);
        transcript.append_serde(b"e2_plus", &second.e2_plus);
        transcript.append_serde(b"e2_minus", &second.e2_minus);
        alpha.push(transcript.challenge_scalar(b"alpha"));
    }

    let gamma = transcript.challenge_scalar(b"gamma");
    let final_message = proof
        .final_message
        .as_ref()
        .expect("transparent proofs carry a final message");
    transcript.append_serde(b"final_e1", &final_message.e1);
    transcript.append_serde(b"final_e2", &final_message.e2);
    let d = transcript.challenge_scalar(b"d");
    Challenges {
        beta,
        alpha,
        gamma,
        d,
    }
}

fn push_term(
    bases: &mut Vec<ArkGT>,
    scalars: &mut Vec<ArkFr>,
    names: &mut Vec<String>,
    name: impl Into<String>,
    base: &ArkGT,
    scalar: ArkFr,
) {
    bases.push(*base);
    scalars.push(scalar);
    names.push(name.into());
}

fn build_relation(
    commitments: &[ArkGT],
    rho: ArkFr,
    evaluation: ArkFr,
    point: &[ArkFr],
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    label: &'static [u8],
) -> Relation {
    let sigma = proof.sigma;
    let nu = proof.nu;
    assert_eq!(sigma, proof.first_messages.len());
    assert_eq!(sigma, proof.second_messages.len());
    assert_eq!(point.len(), sigma + nu);
    let challenges = replay_transcript(proof, label);
    let beta0_inv = inv(challenges.beta[0]);
    let d_inv = inv(challenges.d);
    let d_sq = challenges.d * challenges.d;

    let mut s2_coords = vec![ArkFr::zero(); sigma];
    s2_coords[..nu].copy_from_slice(&point[sigma..]);
    let mut s1_acc = ArkFr::one();
    let mut s2_acc = ArkFr::one();
    let mut chi_scalars = vec![ArkFr::one(); sigma + 1];
    for round in 0..sigma {
        let alpha = challenges.alpha[round];
        let alpha_inv = inv(alpha);
        let idx = sigma - round - 1;
        s1_acc = s1_acc * (alpha * (ArkFr::one() - point[idx]) + point[idx]);
        s2_acc = s2_acc * (alpha_inv * (ArkFr::one() - s2_coords[idx]) + s2_coords[idx]);
        let u = if round + 1 < sigma {
            inv(challenges.beta[round + 1])
        } else {
            d_inv
        };
        let v = if round + 1 < sigma {
            challenges.beta[round + 1]
        } else {
            challenges.d
        };
        chi_scalars[idx] = chi_scalars[idx]
            + u * alpha * challenges.beta[round]
            + v * alpha_inv * inv(challenges.beta[round]);
    }

    let mut bases = Vec::with_capacity(9 * sigma + commitments.len() + 4);
    let mut scalars = Vec::with_capacity(bases.capacity());
    let mut names = Vec::with_capacity(bases.capacity());
    push_term(
        &mut bases,
        &mut scalars,
        &mut names,
        "C_init",
        &proof.vmv_message.c,
        ArkFr::one(),
    );
    let mut rho_power = ArkFr::one();
    for (index, commitment) in commitments.iter().enumerate() {
        push_term(
            &mut bases,
            &mut scalars,
            &mut names,
            format!("C_{index}"),
            commitment,
            beta0_inv * rho_power,
        );
        rho_power = rho_power * rho;
    }
    push_term(
        &mut bases,
        &mut scalars,
        &mut names,
        "D2_init",
        &proof.vmv_message.d2,
        challenges.beta[0] + d_sq,
    );

    for round in 0..sigma {
        let first = &proof.first_messages[round];
        let second = &proof.second_messages[round];
        let alpha = challenges.alpha[round];
        let alpha_inv = inv(alpha);
        let beta = challenges.beta[round];
        let beta_inv = inv(beta);
        let u = if round + 1 < sigma {
            inv(challenges.beta[round + 1])
        } else {
            d_inv
        };
        let v = if round + 1 < sigma {
            challenges.beta[round + 1]
        } else {
            challenges.d
        };
        for (name, base, scalar) in [
            ("C+", &second.c_plus, alpha),
            ("C-", &second.c_minus, alpha_inv),
            ("D1L", &first.d1_left, u * alpha),
            ("D1R", &first.d1_right, u),
            ("D2L", &first.d2_left, v * alpha_inv),
            ("D2R", &first.d2_right, v),
        ] {
            push_term(
                &mut bases,
                &mut scalars,
                &mut names,
                format!("{name}_{round}"),
                base,
                scalar,
            );
        }
        let setup_index = sigma - round;
        push_term(
            &mut bases,
            &mut scalars,
            &mut names,
            format!("Delta1R_{setup_index}"),
            &setup.delta_1r[setup_index],
            u * beta,
        );
        push_term(
            &mut bases,
            &mut scalars,
            &mut names,
            format!("Delta2R_{setup_index}"),
            &setup.delta_2r[setup_index],
            v * beta_inv,
        );
    }
    for (index, scalar) in chi_scalars.into_iter().enumerate() {
        push_term(
            &mut bases,
            &mut scalars,
            &mut names,
            format!("chi_{index}"),
            &setup.chi[index],
            scalar,
        );
    }
    push_term(
        &mut bases,
        &mut scalars,
        &mut names,
        "HT",
        &setup.ht,
        s1_acc * s2_acc,
    );
    assert_eq!(bases.len(), 9 * sigma + commitments.len() + 4);
    assert!(scalars.iter().all(|scalar| !scalar.0.is_zero()));

    let mut e1_acc = proof.vmv_message.e1;
    let mut e2_acc = setup.g2_0.scale(&evaluation);
    for round in 0..sigma {
        let first = &proof.first_messages[round];
        let second = &proof.second_messages[round];
        e1_acc = e1_acc
            + first.e1_beta.scale(&challenges.beta[round])
            + second.e1_plus.scale(&challenges.alpha[round])
            + second.e1_minus.scale(&inv(challenges.alpha[round]));
        e2_acc = e2_acc
            + first.e2_beta.scale(&inv(challenges.beta[round]))
            + second.e2_plus.scale(&challenges.alpha[round])
            + second.e2_minus.scale(&inv(challenges.alpha[round]));
    }
    let final_message = proof.final_message.as_ref().unwrap();
    let gamma_inv = inv(challenges.gamma);
    let p1_g1 = final_message.e1 + setup.g1_0.scale(&challenges.d);
    let p1_g2 = final_message.e2 + setup.g2_0.scale(&d_inv);
    let p2_g1 = setup.h1;
    let p2_g2 = (e2_acc + setup.g2_0.scale(&(d_inv * s1_acc))).scale(&(-challenges.gamma));
    let p3_g1 = (e1_acc + setup.g1_0.scale(&(challenges.d * s2_acc))).scale(&(-gamma_inv));
    let p3_g2 = setup.h2;
    let p4_g1 = proof.vmv_message.e1.scale(&d_sq);
    let p4_g2 = setup.g2_0;
    let lhs = ArkGT(Bn254::multi_pairing(
        [p1_g1.0, p2_g1.0, p3_g1.0, p4_g1.0],
        [p1_g2.0, p2_g2.0, p3_g2.0, p4_g2.0],
    ));

    let mut e1_bases = Vec::with_capacity(3 * sigma + 4);
    let mut e1_scalars = Vec::with_capacity(3 * sigma + 4);
    let mut e2_bases = vec![setup.g2_0.0];
    let mut e2_scalars = vec![evaluation];
    for round in 0..sigma {
        let first = &proof.first_messages[round];
        let second = &proof.second_messages[round];
        e1_bases.extend([first.e1_beta.0, second.e1_plus.0, second.e1_minus.0]);
        e1_scalars.extend([
            challenges.beta[round],
            challenges.alpha[round],
            inv(challenges.alpha[round]),
        ]);
        e2_bases.extend([first.e2_beta.0, second.e2_plus.0, second.e2_minus.0]);
        e2_scalars.extend([
            inv(challenges.beta[round]),
            challenges.alpha[round],
            inv(challenges.alpha[round]),
        ]);
    }
    e1_scalars.extend([
        -gamma_inv,
        -gamma_inv * challenges.d * s2_acc,
        challenges.d,
        d_sq,
    ]);
    e1_bases.extend([e1_acc.0, setup.g1_0.0, setup.g1_0.0, proof.vmv_message.e1.0]);
    e2_scalars.extend([-challenges.gamma, -challenges.gamma * d_inv * s1_acc, d_inv]);
    e2_bases.extend([e2_acc.0, setup.g2_0.0, setup.g2_0.0]);
    assert_eq!(e1_scalars.len(), 3 * sigma + 4);
    assert_eq!(e2_scalars.len(), 3 * sigma + 4);
    assert_eq!(e1_bases.len(), e1_scalars.len());
    assert_eq!(e2_bases.len(), e2_scalars.len());

    Relation {
        bases,
        scalars,
        names,
        lhs,
        e1_bases,
        e1_scalars,
        e2_bases,
        e2_scalars,
    }
}

fn scalar_magnitudes(scalars: &[ArkFr]) -> Vec<SignedMagnitude> {
    scalars
        .iter()
        .map(|scalar| SignedMagnitude {
            magnitude: BigUint::from_bytes_be(&scalar.0.into_bigint().to_bytes_be()),
            positive: true,
        })
        .collect()
}

fn window_digit(value: &BigUint, start: usize, width: usize) -> usize {
    (0..width).fold(0, |digit, bit| {
        digit | usize::from(value.bit((start + bit) as u64)) << bit
    })
}

fn pippenger_counts(scalars: &[SignedMagnitude], window: usize, bits: usize) -> GtOps {
    let num_windows = bits.div_ceil(window);
    let mut ops = GtOps::default();
    let mut result_started = false;
    for window_index in (0..num_windows).rev() {
        if result_started {
            ops.cyclotomic_sqr += window;
        }
        let mut buckets = vec![false; (1 << window) - 1];
        for scalar in scalars {
            let digit = window_digit(&scalar.magnitude, window_index * window, window);
            if digit != 0 {
                if buckets[digit - 1] {
                    ops.mul += 1;
                } else {
                    buckets[digit - 1] = true;
                }
            }
        }
        let mut running = false;
        let mut sum = false;
        for occupied in buckets.into_iter().rev() {
            if occupied {
                if running {
                    ops.mul += 1;
                } else {
                    running = true;
                }
            }
            if running {
                if sum {
                    ops.mul += 1;
                } else {
                    sum = true;
                }
            }
        }
        if sum {
            if result_started {
                ops.mul += 1;
            } else {
                result_started = true;
            }
        }
    }
    ops
}

fn gt_pippenger(
    bases: &[Fq12],
    scalars: &[SignedMagnitude],
    window: usize,
    bits: usize,
) -> (Fq12, GtOps) {
    assert_eq!(bases.len(), scalars.len());
    let num_windows = bits.div_ceil(window);
    let mut ops = GtOps::default();
    let mut result: Option<Fq12> = None;
    for window_index in (0..num_windows).rev() {
        if let Some(acc) = result.as_mut() {
            for _ in 0..window {
                let _ = acc.cyclotomic_square_in_place();
                ops.cyclotomic_sqr += 1;
            }
        }
        let mut buckets = vec![None; (1 << window) - 1];
        for (base, scalar) in bases.iter().zip(scalars) {
            let digit = window_digit(&scalar.magnitude, window_index * window, window);
            if digit != 0 {
                if let Some(bucket) = buckets[digit - 1].as_mut() {
                    *bucket *= base;
                    ops.mul += 1;
                } else {
                    buckets[digit - 1] = Some(*base);
                }
            }
        }
        let mut running = None;
        let mut sum = None;
        for bucket in buckets.into_iter().rev() {
            if let Some(bucket) = bucket {
                if let Some(running) = running.as_mut() {
                    *running *= &bucket;
                    ops.mul += 1;
                } else {
                    running = Some(bucket);
                }
            }
            if let Some(running) = running {
                if let Some(sum) = sum.as_mut() {
                    *sum *= &running;
                    ops.mul += 1;
                } else {
                    sum = Some(running);
                }
            }
        }
        if let Some(sum) = sum {
            if let Some(result) = result.as_mut() {
                *result *= &sum;
                ops.mul += 1;
            } else {
                result = Some(sum);
            }
        }
    }
    let result = result.unwrap_or_else(Fq12::one);
    assert_eq!(ops.mul, pippenger_counts(scalars, window, bits).mul);
    (result, ops)
}

fn naive_counts(scalars: &[SignedMagnitude]) -> GtOps {
    GtOps {
        mul: scalars
            .iter()
            .map(|scalar| scalar.magnitude.count_ones() as usize)
            .sum::<usize>()
            + scalars.len().saturating_sub(1),
        generic_sqr: SCALAR_BITS * scalars.len(),
        cyclotomic_sqr: 0,
    }
}

fn round_div(numerator: BigInt, denominator: &BigInt) -> BigInt {
    let (mut quotient, remainder) = numerator.div_rem(denominator);
    if (&remainder + &remainder).abs() >= *denominator {
        quotient += if remainder.sign() == Sign::Minus {
            BigInt::from(-1)
        } else {
            BigInt::from(1)
        };
    }
    quotient
}

fn decompose_2d(scalar: &BigUint) -> Vec<SignedMagnitude> {
    let u = BigUint::from(4_965_661_367_192_848_881u64);
    let lambda = BigUint::from(6u64) * &u * &u;
    let (mut k1, remainder) = scalar.div_rem(&lambda);
    if &remainder + &remainder > lambda {
        k1 += BigUint::from(1u64);
    }
    let k0 = BigInt::from_biguint(Sign::Plus, scalar.clone())
        - BigInt::from_biguint(Sign::Plus, &k1 * lambda);
    vec![
        signed_magnitude(k0),
        SignedMagnitude {
            magnitude: k1,
            positive: true,
        },
    ]
}

fn decompose_4d(scalar: &BigUint) -> Vec<SignedMagnitude> {
    const BASIS: [[i128; 4]; 4] = [
        [
            9_931_322_734_385_697_762,
            4_965_661_367_192_848_882,
            -4_965_661_367_192_848_881,
            4_965_661_367_192_848_881,
        ],
        [
            -4_965_661_367_192_848_881,
            4_965_661_367_192_848_881,
            -4_965_661_367_192_848_881,
            -9_931_322_734_385_697_763,
        ],
        [
            4_965_661_367_192_848_882,
            4_965_661_367_192_848_881,
            4_965_661_367_192_848_881,
            -9_931_322_734_385_697_762,
        ],
        [
            9_931_322_734_385_697_763,
            -4_965_661_367_192_848_881,
            -4_965_661_367_192_848_882,
            -4_965_661_367_192_848_881,
        ],
    ];
    const INVERSE_ROW_NUMERATORS: [&str; 4] = [
        "734653495049373973658254490726798021314063399421879442165",
        "-734653495049373973806201247608587340319794091592875701774",
        "734653495049373973806201247608587340329725414327261399537",
        "734653495049373973806201247608587340314828430225682852893",
    ];
    let modulus = BigInt::from_biguint(Sign::Plus, modulus_biguint());
    let scalar_int = BigInt::from_biguint(Sign::Plus, scalar.clone());
    let lattice_coefficients: Vec<BigInt> = INVERSE_ROW_NUMERATORS
        .iter()
        .map(|numerator| round_div(&scalar_int * numerator.parse::<BigInt>().unwrap(), &modulus))
        .collect();
    (0..4)
        .map(|coordinate| {
            let lattice_coordinate = lattice_coefficients
                .iter()
                .zip(BASIS)
                .map(|(coefficient, row)| coefficient * BigInt::from(row[coordinate]))
                .sum::<BigInt>();
            let target = if coordinate == 0 {
                scalar_int.clone()
            } else {
                BigInt::zero()
            };
            signed_magnitude(target - lattice_coordinate)
        })
        .collect()
}

fn signed_magnitude(value: BigInt) -> SignedMagnitude {
    SignedMagnitude {
        magnitude: value.magnitude().clone(),
        positive: value.sign() != Sign::Minus,
    }
}

fn modulus_biguint() -> BigUint {
    BigUint::from_bytes_be(&ArkScalar::MODULUS.to_bytes_be())
}

fn scalar_from_magnitude(value: &SignedMagnitude) -> ArkScalar {
    let scalar = ArkScalar::from_be_bytes_mod_order(&value.magnitude.to_bytes_be());
    if value.positive {
        scalar
    } else {
        -scalar
    }
}

fn assert_decomposition(scalar: ArkScalar, decomposition: &[SignedMagnitude]) {
    let lambda = ArkScalar::from(6u64) * ArkScalar::from(4_965_661_367_192_848_881u64).square();
    let mut power = ArkScalar::one();
    let mut recomposed = ArkScalar::zero();
    for component in decomposition {
        recomposed += scalar_from_magnitude(component) * power;
        power *= lambda;
    }
    assert_eq!(recomposed, scalar);
}

fn glv_expansion(
    bases: &[Fq12],
    scalars: &[SignedMagnitude],
    dimensions: usize,
) -> (Vec<Fq12>, Vec<SignedMagnitude>, usize) {
    let mut expanded_bases = Vec::with_capacity(dimensions * bases.len());
    let mut expanded_scalars = Vec::with_capacity(dimensions * bases.len());
    let mut max_bits = 0;
    for (base, scalar) in bases.iter().zip(scalars) {
        let decomposition = match dimensions {
            2 => decompose_2d(&scalar.magnitude),
            4 => decompose_4d(&scalar.magnitude),
            _ => unreachable!(),
        };
        let full_scalar = ArkScalar::from_be_bytes_mod_order(&scalar.magnitude.to_bytes_be());
        assert_decomposition(full_scalar, &decomposition);
        for (power, component) in decomposition.into_iter().enumerate() {
            let mut conjugate = *base;
            conjugate.frobenius_map_in_place(power);
            if !component.positive {
                let _ = conjugate.cyclotomic_inverse_in_place().unwrap();
            }
            max_bits = max_bits.max(component.magnitude.bits() as usize);
            expanded_bases.push(conjugate);
            expanded_scalars.push(component);
        }
    }
    (expanded_bases, expanded_scalars, max_bits)
}

fn decompose_g1_scalar(scalar: ArkScalar) -> Vec<SignedMagnitude> {
    let [n11, n12, n21, n22] = [
        BigInt::from(-147_946_756_881_789_319_000_765_030_803_803_410_728i128),
        BigInt::from(9_931_322_734_385_697_763i128),
        BigInt::from(-9_931_322_734_385_697_763i128),
        BigInt::from(-147_946_756_881_789_319_010_696_353_538_189_108_491i128),
    ];
    let scalar = BigInt::from_biguint(
        Sign::Plus,
        BigUint::from_bytes_be(&scalar.into_bigint().to_bytes_be()),
    );
    let modulus = BigInt::from_biguint(Sign::Plus, modulus_biguint());
    let beta1 = {
        let (mut quotient, remainder) = (&scalar * &n22).div_rem(&modulus);
        if &remainder + &remainder > modulus {
            quotient += 1;
        }
        quotient
    };
    let beta2 = {
        let (mut quotient, remainder) = (&scalar * -&n12).div_rem(&modulus);
        if &remainder + &remainder > modulus {
            quotient += 1;
        }
        quotient
    };
    let b1 = &beta1 * n11 + &beta2 * n21;
    let b2 = beta1 * n12 + beta2 * n22;
    vec![signed_magnitude(scalar - b1), signed_magnitude(-b2)]
}

fn g1_endomorphism(point: &G1Projective) -> G1Projective {
    const ENDO_COEFF: Fq =
        MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");
    let mut image = *point;
    image.x *= ENDO_COEFF;
    image
}

fn decompose_g2_scalar(scalar: ArkScalar) -> Vec<SignedMagnitude> {
    let scalar = scalar.into_bigint();
    let mut accumulators = [0u128; 4];
    for (bit, &(k0, k1, k2, k3, neg0, neg1, neg2, neg3)) in
        POWER_OF_2_DECOMPOSITIONS.iter().enumerate()
    {
        if scalar.get_bit(bit) {
            for ((accumulator, coefficient), negative) in accumulators
                .iter_mut()
                .zip([k0, k1, k2, k3])
                .zip([neg0, neg1, neg2, neg3])
            {
                *accumulator = if negative {
                    accumulator.wrapping_sub(coefficient)
                } else {
                    accumulator.wrapping_add(coefficient)
                };
            }
        }
    }
    accumulators
        .into_iter()
        .map(|coefficient| {
            let value = coefficient as i128;
            SignedMagnitude {
                magnitude: BigUint::from(value.unsigned_abs()),
                positive: value >= 0,
            }
        })
        .collect()
}

fn g2_frobenius(point: &G2Projective, power: usize) -> G2Projective {
    if point.is_zero() {
        return *point;
    }
    let mut image = *point;
    let coefficients = g2_constants::get_frobenius_coefficients();
    if power & 1 == 1 {
        let _ = image.x.conjugate_in_place();
        let _ = image.y.conjugate_in_place();
        let _ = image.z.conjugate_in_place();
    }
    match power % 4 {
        0 => image,
        1 => {
            image.x *= coefficients.psi1_coef2;
            image.y *= coefficients.psi1_coef3;
            image
        }
        2 => {
            image.x *= coefficients.psi2_coef2;
            image.y *= coefficients.psi2_coef3;
            image
        }
        _ => {
            image.x *= coefficients.psi3_coef2;
            image.y *= coefficients.psi3_coef3;
            image
        }
    }
}

fn g1_components(scalars: &[ArkFr]) -> Vec<SignedMagnitude> {
    scalars
        .iter()
        .flat_map(|scalar| decompose_g1_scalar(scalar.0))
        .collect()
}

fn g2_components(scalars: &[ArkFr]) -> Vec<SignedMagnitude> {
    scalars
        .iter()
        .flat_map(|scalar| decompose_g2_scalar(scalar.0))
        .collect()
}

fn gt_components(scalars: &[ArkFr]) -> Vec<SignedMagnitude> {
    scalar_magnitudes(scalars)
        .into_iter()
        .flat_map(|scalar| decompose_4d(&scalar.magnitude))
        .collect()
}

fn max_bits(scalars: &[SignedMagnitude]) -> usize {
    scalars
        .iter()
        .map(|scalar| scalar.magnitude.bits() as usize)
        .max()
        .unwrap_or(0)
}

fn centered_digits(scalar: &SignedMagnitude, window: usize) -> Vec<isize> {
    let radix = BigInt::from(1usize << window);
    let half = BigInt::from(1usize << (window - 1));
    let mut value = BigInt::from_biguint(
        if scalar.positive {
            Sign::Plus
        } else {
            Sign::Minus
        },
        scalar.magnitude.clone(),
    );
    let mut digits = Vec::new();
    while !value.is_zero() {
        let mut residue = &value % &radix;
        if residue.sign() == Sign::Minus {
            residue += &radix;
        }
        let digit = if residue >= half {
            residue - &radix
        } else {
            residue
        };
        digits.push(digit.to_isize().unwrap());
        value = (value - digit) / &radix;
    }
    digits
}

fn centered_windows(scalars: &[SignedMagnitude], window: usize) -> usize {
    scalars
        .iter()
        .map(|scalar| centered_digits(scalar, window).len())
        .max()
        .unwrap_or(0)
}

fn gt_glv_expansion_raw(bases: &[Fq12], scalars: &[ArkFr]) -> (Vec<Fq12>, Vec<SignedMagnitude>) {
    let mut expanded_bases = Vec::with_capacity(4 * bases.len());
    let mut expanded_scalars = Vec::with_capacity(4 * bases.len());
    for (base, scalar) in bases.iter().zip(scalars) {
        let components = decompose_4d(&BigUint::from_bytes_be(
            &scalar.0.into_bigint().to_bytes_be(),
        ));
        assert_decomposition(scalar.0, &components);
        for (power, component) in components.into_iter().enumerate() {
            let mut image = *base;
            image.frobenius_map_in_place(power);
            expanded_bases.push(image);
            expanded_scalars.push(component);
        }
    }
    (expanded_bases, expanded_scalars)
}

fn g1_glv_expansion_raw(
    bases: &[G1Projective],
    scalars: &[ArkFr],
) -> (Vec<G1Projective>, Vec<SignedMagnitude>) {
    let mut expanded_bases = Vec::with_capacity(2 * bases.len());
    let mut expanded_scalars = Vec::with_capacity(2 * bases.len());
    for (base, scalar) in bases.iter().zip(scalars) {
        expanded_bases.extend([*base, g1_endomorphism(base)]);
        expanded_scalars.extend(decompose_g1_scalar(scalar.0));
    }
    (expanded_bases, expanded_scalars)
}

fn g2_glv_expansion_raw(
    bases: &[G2Projective],
    scalars: &[ArkFr],
) -> (Vec<G2Projective>, Vec<SignedMagnitude>) {
    let mut expanded_bases = Vec::with_capacity(4 * bases.len());
    let mut expanded_scalars = Vec::with_capacity(4 * bases.len());
    for (base, scalar) in bases.iter().zip(scalars) {
        for power in 0..4 {
            expanded_bases.push(g2_frobenius(base, power));
        }
        expanded_scalars.extend(decompose_g2_scalar(scalar.0));
    }
    (expanded_bases, expanded_scalars)
}

fn gt_straus(bases: &[Fq12], scalars: &[SignedMagnitude], window: usize, signed: bool) -> Fq12 {
    let windows = if signed {
        centered_windows(scalars, window)
    } else {
        max_bits(scalars).div_ceil(window)
    };
    let table_limit = if signed {
        1 << (window - 1)
    } else {
        (1 << window) - 1
    };
    let tables: Vec<Vec<Fq12>> = bases
        .iter()
        .map(|base| {
            let mut table = vec![Fq12::one(), *base];
            while table.len() <= table_limit {
                let next = *table.last().unwrap() * base;
                table.push(next);
            }
            table
        })
        .collect();
    let signed_digits: Vec<Vec<isize>> = if signed {
        scalars
            .iter()
            .map(|scalar| centered_digits(scalar, window))
            .collect()
    } else {
        Vec::new()
    };
    let mut result = Fq12::one();
    for window_index in (0..windows).rev() {
        for _ in 0..window {
            let _ = result.cyclotomic_square_in_place();
        }
        for (index, scalar) in scalars.iter().enumerate() {
            let digit = if signed {
                signed_digits[index].get(window_index).copied().unwrap_or(0)
            } else {
                window_digit(&scalar.magnitude, window_index * window, window) as isize
            };
            let mut operand = tables[index][digit.unsigned_abs()];
            let negative = if signed { digit < 0 } else { !scalar.positive };
            if negative {
                let _ = operand.cyclotomic_inverse_in_place().unwrap();
            }
            result *= operand;
        }
    }
    result
}

fn ec_straus<G>(bases: &[G], scalars: &[SignedMagnitude], window: usize, signed: bool) -> G
where
    G: CurveGroup<ScalarField = ArkScalar> + Copy,
{
    let windows = if signed {
        centered_windows(scalars, window)
    } else {
        max_bits(scalars).div_ceil(window)
    };
    let table_limit = if signed {
        1 << (window - 1)
    } else {
        (1 << window) - 1
    };
    let tables: Vec<Vec<G>> = bases
        .iter()
        .map(|base| {
            let mut table = vec![G::zero(), *base];
            while table.len() <= table_limit {
                let next = *table.last().unwrap() + base;
                table.push(next);
            }
            table
        })
        .collect();
    let signed_digits: Vec<Vec<isize>> = if signed {
        scalars
            .iter()
            .map(|scalar| centered_digits(scalar, window))
            .collect()
    } else {
        Vec::new()
    };
    let mut result = G::zero();
    for window_index in (0..windows).rev() {
        for _ in 0..window {
            let _ = result.double_in_place();
        }
        for (index, scalar) in scalars.iter().enumerate() {
            let digit = if signed {
                signed_digits[index].get(window_index).copied().unwrap_or(0)
            } else {
                window_digit(&scalar.magnitude, window_index * window, window) as isize
            };
            let mut operand = tables[index][digit.unsigned_abs()];
            let negative = if signed { digit < 0 } else { !scalar.positive };
            if negative {
                operand = -operand;
            }
            result += operand;
        }
    }
    result
}

#[derive(Clone, Copy)]
enum StrausVariant {
    Fixed,
    Signed,
}

impl StrausVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Signed => "signed",
        }
    }

    fn is_signed(self) -> bool {
        matches!(self, Self::Signed)
    }
}

#[derive(Clone, Copy)]
struct StrausCount {
    bits: usize,
    windows: usize,
    table_ops: usize,
    square_ops: usize,
    online_ops: usize,
    fq_products: usize,
    rows: usize,
    public_digit_selectors: usize,
    selector_bits: usize,
    fixed_operand_offsets: usize,
    choices_per_selector: usize,
    shift_relations: usize,
}

fn straus_shape(
    original_bases: usize,
    public_original_bases: usize,
    dimensions: usize,
    scalars: &[SignedMagnitude],
    window: usize,
    variant: StrausVariant,
) -> (usize, usize, usize, usize, usize, usize, usize) {
    let bits = max_bits(scalars);
    let windows = if variant.is_signed() {
        centered_windows(scalars, window)
    } else {
        bits.div_ceil(window)
    };
    let table_entries = if variant.is_signed() {
        1 << (window - 1)
    } else {
        (1 << window) - 1
    };
    let table_ops_per_base = table_entries - 1;
    let public_expanded_bases = public_original_bases * dimensions;
    let public_digit_selectors = public_expanded_bases * windows;
    let selector_bits = if variant.is_signed() {
        public_digit_selectors * window
    } else {
        public_digit_selectors * window + public_expanded_bases
    };
    (
        bits,
        windows,
        original_bases * table_ops_per_base,
        dimensions * original_bases * windows,
        public_digit_selectors,
        selector_bits,
        original_bases * table_entries,
    )
}

fn gt_straus_count(
    original_bases: usize,
    public_original_bases: usize,
    scalars: &[SignedMagnitude],
    window: usize,
    variant: StrausVariant,
) -> StrausCount {
    let (
        bits,
        windows,
        table_ops,
        online_ops,
        public_digit_selectors,
        selector_bits,
        fixed_operand_offsets,
    ) = straus_shape(
        original_bases,
        public_original_bases,
        4,
        scalars,
        window,
        variant,
    );
    let square_ops = windows * window;
    StrausCount {
        bits,
        windows,
        table_ops,
        square_ops,
        online_ops,
        fq_products: 54 * (table_ops + online_ops) + 18 * square_ops,
        rows: 12 * (table_ops + square_ops + online_ops),
        public_digit_selectors,
        selector_bits,
        fixed_operand_offsets,
        choices_per_selector: if variant.is_signed() {
            (1 << (window - 1)) + 1
        } else {
            1 << window
        },
        shift_relations: 1,
    }
}

fn g1_straus_count(
    original_bases: usize,
    scalars: &[SignedMagnitude],
    window: usize,
    variant: StrausVariant,
) -> StrausCount {
    let (
        bits,
        windows,
        table_ops,
        online_ops,
        public_digit_selectors,
        selector_bits,
        fixed_operand_offsets,
    ) = straus_shape(original_bases, original_bases, 2, scalars, window, variant);
    let square_ops = windows * window;
    StrausCount {
        bits,
        windows,
        table_ops,
        square_ops,
        online_ops,
        fq_products: 11 * table_ops + 7 * square_ops + 15 * online_ops,
        rows: 11 * table_ops + 7 * square_ops + 15 * online_ops,
        public_digit_selectors,
        selector_bits,
        fixed_operand_offsets,
        choices_per_selector: if variant.is_signed() {
            (1 << (window - 1)) + 1
        } else {
            1 << window
        },
        shift_relations: 5,
    }
}

fn g2_straus_count(
    original_bases: usize,
    scalars: &[SignedMagnitude],
    window: usize,
    variant: StrausVariant,
) -> StrausCount {
    let (
        bits,
        windows,
        table_ops,
        online_ops,
        public_digit_selectors,
        selector_bits,
        fixed_operand_offsets,
    ) = straus_shape(original_bases, original_bases, 4, scalars, window, variant);
    let square_ops = windows * window;
    StrausCount {
        bits,
        windows,
        table_ops,
        square_ops,
        online_ops,
        fq_products: 30 * table_ops + 17 * square_ops + 41 * online_ops,
        rows: 22 * table_ops + 14 * square_ops + 30 * online_ops,
        public_digit_selectors,
        selector_bits,
        fixed_operand_offsets,
        choices_per_selector: if variant.is_signed() {
            (1 << (window - 1)) + 1
        } else {
            1 << window
        },
        shift_relations: 5,
    }
}

fn print_straus_component(
    sigma: usize,
    group: &str,
    window: usize,
    variant: StrausVariant,
    original_bases: usize,
    dimensions: usize,
    count: StrausCount,
) {
    println!(
        "STRAUS sigma={sigma} group={group} variant={} w={window} original_bases={original_bases} expanded_bases={} mini_bits={} windows={} table_ops={} square_or_double_ops={} online_ops={} fq_products={} rows={} public_digit_selectors={} public_selector_bits={} fixed_operand_offsets={} choices_per_selector={} shift_relations={}",
        variant.name(),
        original_bases * dimensions,
        count.bits,
        count.windows,
        count.table_ops,
        count.square_ops,
        count.online_ops,
        count.fq_products,
        count.rows,
        count.public_digit_selectors,
        count.selector_bits,
        count.fixed_operand_offsets,
        count.choices_per_selector,
        count.shift_relations,
    );
}

fn print_straus_counts(
    sigma: usize,
    gt_scalars: &[ArkFr],
    g1_scalars: &[ArkFr],
    g2_scalars: &[ArkFr],
) {
    const PAIRING_FQ_PRODUCTS: usize = 27_340 + 7_992;
    const PAIRING_ROWS: usize = 14_380 + 3_288;

    let gt_components = gt_components(gt_scalars);
    let g1_components = g1_components(g1_scalars);
    let g2_components = g2_components(g2_scalars);
    assert_eq!(gt_components.len(), 4 * gt_scalars.len());
    assert_eq!(g1_components.len(), 2 * g1_scalars.len());
    assert_eq!(g2_components.len(), 4 * g2_scalars.len());

    for window in [3, 4, 5] {
        for variant in [StrausVariant::Fixed, StrausVariant::Signed] {
            let gt = gt_straus_count(
                gt_scalars.len(),
                gt_scalars.len() - 1,
                &gt_components,
                window,
                variant,
            );
            let g1 = g1_straus_count(g1_scalars.len(), &g1_components, window, variant);
            let g2 = g2_straus_count(g2_scalars.len(), &g2_components, window, variant);
            print_straus_component(sigma, "GT", window, variant, gt_scalars.len(), 4, gt);
            print_straus_component(sigma, "G1", window, variant, g1_scalars.len(), 2, g1);
            print_straus_component(sigma, "G2", window, variant, g2_scalars.len(), 4, g2);
            let fq_products =
                gt.fq_products + g1.fq_products + g2.fq_products + PAIRING_FQ_PRODUCTS;
            let rows = gt.rows + g1.rows + g2.rows + PAIRING_ROWS;
            println!(
                "STRAUS_TOTAL sigma={sigma} variant={} w={window} fq_products={fq_products} rows={rows} domain=2^{} public_digit_selectors={} public_selector_bits={} fixed_operand_offsets={} shift_relations={}",
                variant.name(),
                rows.next_power_of_two().ilog2(),
                gt.public_digit_selectors
                    + g1.public_digit_selectors
                    + g2.public_digit_selectors,
                gt.selector_bits + g1.selector_bits + g2.selector_bits,
                gt.fixed_operand_offsets
                    + g1.fixed_operand_offsets
                    + g2.fixed_operand_offsets,
                gt.shift_relations + g1.shift_relations + g2.shift_relations,
            );
        }
    }
}

fn ec_pippenger_counts(scalars: &[SignedMagnitude], window: usize, bits: usize) -> EcOps {
    let mut ops = EcOps::default();
    let mut result_started = false;
    for window_index in (0..bits.div_ceil(window)).rev() {
        if result_started {
            ops.double += window;
        }
        let mut bucket_counts = vec![0usize; (1 << window) - 1];
        for scalar in scalars {
            let digit = window_digit(&scalar.magnitude, window_index * window, window);
            if digit != 0 {
                if bucket_counts[digit - 1] != 0 {
                    ops.mixed_add += 1;
                }
                bucket_counts[digit - 1] += 1;
            }
        }
        let mut running = false;
        let mut sum = false;
        for count in bucket_counts.into_iter().rev() {
            if count != 0 {
                if running {
                    ops.projective_add += 1;
                } else {
                    running = true;
                }
            }
            if running {
                if sum {
                    ops.projective_add += 1;
                } else {
                    sum = true;
                }
            }
        }
        if sum {
            if result_started {
                ops.projective_add += 1;
            } else {
                result_started = true;
            }
        }
    }
    ops
}

fn print_gt_counts(sigma: usize, num_commitments: usize, scalars: &[ArkFr]) {
    let scalars = scalar_magnitudes(scalars);
    let naive = naive_counts(&scalars);
    println!(
        "GT sigma={sigma} N={num_commitments} algorithm=naive mul={} generic_sqr={} cyclotomic_sqr={} fq_products={} rows={}",
        naive.mul,
        naive.generic_sqr,
        naive.cyclotomic_sqr,
        naive.fq_products(),
        naive.rows()
    );
    for dimensions in [1, 2, 4] {
        let (count_scalars, bits) = if dimensions == 1 {
            (scalars.clone(), SCALAR_BITS)
        } else {
            let mut decomposed = Vec::with_capacity(dimensions * scalars.len());
            let mut max_bits = 0;
            for scalar in &scalars {
                let components = if dimensions == 2 {
                    decompose_2d(&scalar.magnitude)
                } else {
                    decompose_4d(&scalar.magnitude)
                };
                assert_decomposition(
                    ArkScalar::from_be_bytes_mod_order(&scalar.magnitude.to_bytes_be()),
                    &components,
                );
                max_bits = max_bits.max(
                    components
                        .iter()
                        .map(|component| component.magnitude.bits() as usize)
                        .max()
                        .unwrap_or(0),
                );
                decomposed.extend(components);
            }
            (decomposed, max_bits)
        };
        for window in WINDOWS {
            let ops = pippenger_counts(&count_scalars, window, bits);
            println!(
                "GT sigma={sigma} N={num_commitments} algorithm=glv{dimensions}-pippenger c={window} bits={bits} mul={} cyclotomic_sqr={} fq_products={} rows={}",
                ops.mul,
                ops.cyclotomic_sqr,
                ops.fq_products(),
                ops.rows()
            );
        }
    }
}

fn print_ec_counts(sigma: usize, group: &str, scalars: &[ArkFr]) {
    let scalars = scalar_magnitudes(scalars);
    for window in WINDOWS {
        let ops = ec_pippenger_counts(&scalars, window, SCALAR_BITS);
        let fq_products = if group == "G1" {
            ops.g1_fq_products()
        } else {
            ops.g2_fq_products()
        };
        println!(
            "EC sigma={sigma} group={group} terms={} c={window} mixed_add={} projective_add={} double={} fq_products={fq_products}",
            scalars.len(), ops.mixed_add, ops.projective_add, ops.double
        );
    }
}

fn synthetic_scalars(sigma: usize, num_commitments: usize) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xD0_12_00 + sigma as u64);
    let random = |rng: &mut ChaCha20Rng| ArkFr(<ArkScalar as ark_ff::UniformRand>::rand(rng));
    let beta: Vec<_> = (0..sigma).map(|_| random(&mut rng)).collect();
    let alpha: Vec<_> = (0..sigma).map(|_| random(&mut rng)).collect();
    let rho = random(&mut rng);
    let d = random(&mut rng);
    let gamma = random(&mut rng);
    let evaluation = random(&mut rng);
    let s1: Vec<_> = (0..sigma).map(|_| random(&mut rng)).collect();
    let s2: Vec<_> = (0..sigma).map(|_| random(&mut rng)).collect();
    let challenges = Challenges {
        beta,
        alpha,
        gamma,
        d,
    };
    let beta0_inv = inv(challenges.beta[0]);
    let d_inv = inv(d);
    let mut scalars = vec![ArkFr::one()];
    let mut rho_power = ArkFr::one();
    for _ in 0..num_commitments {
        scalars.push(beta0_inv * rho_power);
        rho_power = rho_power * rho;
    }
    scalars.push(challenges.beta[0] + d * d);
    let mut chi_scalars = vec![ArkFr::one(); sigma + 1];
    let mut s1_acc = ArkFr::one();
    let mut s2_acc = ArkFr::one();
    for round in 0..sigma {
        let alpha = challenges.alpha[round];
        let alpha_inv = inv(alpha);
        let beta = challenges.beta[round];
        let beta_inv = inv(beta);
        let u = if round + 1 < sigma {
            inv(challenges.beta[round + 1])
        } else {
            d_inv
        };
        let v = if round + 1 < sigma {
            challenges.beta[round + 1]
        } else {
            d
        };
        scalars.extend([
            alpha,
            alpha_inv,
            u * alpha,
            u,
            v * alpha_inv,
            v,
            u * beta,
            v * beta_inv,
        ]);
        let idx = sigma - round - 1;
        chi_scalars[idx] = chi_scalars[idx] + u * alpha * beta + v * alpha_inv * beta_inv;
        s1_acc = s1_acc * (alpha * (ArkFr::one() - s1[idx]) + s1[idx]);
        s2_acc = s2_acc * (alpha_inv * (ArkFr::one() - s2[idx]) + s2[idx]);
    }
    scalars.extend(chi_scalars);
    scalars.push(s1_acc * s2_acc);
    assert_eq!(scalars.len(), 9 * sigma + num_commitments + 4);

    let mut e1_scalars = Vec::with_capacity(3 * sigma + 4);
    let mut e2_scalars = vec![evaluation];
    for round in 0..sigma {
        e1_scalars.extend([
            challenges.beta[round],
            challenges.alpha[round],
            inv(challenges.alpha[round]),
        ]);
        e2_scalars.extend([
            inv(challenges.beta[round]),
            challenges.alpha[round],
            inv(challenges.alpha[round]),
        ]);
    }
    let gamma_inv = inv(gamma);
    e1_scalars.extend([-gamma_inv, -gamma_inv * d * s2_acc, d, d * d]);
    e2_scalars.extend([-gamma, -gamma * d_inv * s1_acc, d_inv]);
    (scalars, e1_scalars, e2_scalars)
}

fn print_pairing_counts() {
    let ate = Bn254Config::ATE_LOOP_COUNT;
    let doubling_lines = ate.len() - 1;
    let add_lines = ate[..ate.len() - 1]
        .iter()
        .filter(|digit| **digit != 0)
        .count();
    assert_eq!((doubling_lines, add_lines), (64, 21));
    let lines_per_pair = doubling_lines + add_lines + 2;
    let accumulator_squares = doubling_lines - 1;
    let accumulator_lines = 4 * lines_per_pair;
    let precompute_fq_products_per_pair = doubling_lines * 26 + (add_lines + 2) * 37 + 12;
    let precompute_rows_per_pair = doubling_lines * 22 + (add_lines + 2) * 26 + 8;
    let miller_fq_products =
        4 * precompute_fq_products_per_pair + accumulator_squares * 36 + accumulator_lines * 43;
    let miller_rows =
        4 * precompute_rows_per_pair + accumulator_squares * 12 + accumulator_lines * 16;

    let mut x = 4_965_661_367_192_848_881u64;
    let mut naf = Vec::new();
    while x != 0 {
        let digit = if x & 1 == 1 { 2i8 - (x % 4) as i8 } else { 0 };
        naf.push(digit);
        x = if digit < 0 {
            x.midpoint((-digit) as u64)
        } else {
            (x - digit as u64) / 2
        };
    }
    assert_eq!(
        (naf.len(), naf.iter().filter(|digit| **digit != 0).count()),
        (63, 24)
    );
    let fe_mul = 3 * 24 + 10 + 2;
    let fe_inverse_check_mul = 1;
    let fe_cyclotomic_sqr = 3 * 62 + 3;
    let fe_relation_mul = fe_mul + fe_inverse_check_mul;
    let fe_fq_products = fe_relation_mul * 54 + fe_cyclotomic_sqr * 18;
    let fe_rows = (fe_relation_mul + fe_cyclotomic_sqr) * 12;
    println!(
        "PAIRING pairs=4 doubling_lines_per_pair={doubling_lines} signed_add_lines_per_pair={add_lines} frobenius_lines_per_pair=2 accumulator_generic_sqr={accumulator_squares} accumulator_sparse_mul={accumulator_lines} miller_fq_products={miller_fq_products} miller_rows={miller_rows}"
    );
    println!(
        "FINAL_EXP easy_mul=2 easy_inverse=1 inverse_check_mul={fe_inverse_check_mul} hard_mul={} hard_cyclotomic_sqr={fe_cyclotomic_sqr} frobenius_maps=4 total_mul={fe_relation_mul} fq_products={fe_fq_products} rows={fe_rows}",
        fe_mul - 2
    );
}

fn ark_commitment(commitment: &DoryCommitment) -> ArkGT {
    let fq12: Fq12 = commitment.0.into();
    ArkGT(PairingOutput(fq12))
}

fn jolt_commitment(value: &Fq12) -> DoryCommitment {
    DoryCommitment(Bn254GT::from(*value))
}

fn split_commitment(
    commitment: &Fq12,
    ht: &Fq12,
    rho: ArkFr,
    count: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<ArkGT> {
    let mut exponents = vec![ArkScalar::zero(); count];
    let mut weighted_sum = ArkScalar::zero();
    let mut rho_power = rho.0;
    for exponent in &mut exponents[1..] {
        *exponent = <ArkScalar as ark_ff::UniformRand>::rand(rng);
        weighted_sum += *exponent * rho_power;
        rho_power *= rho.0;
    }
    exponents[0] = -weighted_sum;
    exponents
        .into_iter()
        .enumerate()
        .map(|(index, exponent)| {
            let offset = ht.pow(exponent.into_bigint());
            ArkGT(PairingOutput(if index == 0 {
                *commitment * offset
            } else {
                offset
            }))
        })
        .collect()
}

fn main() {
    let label = b"deferred-check-counts";
    let mut rng = ChaCha20Rng::seed_from_u64(0xD0_11_00);
    let setup_start = Instant::now();
    let prover_setup = DoryScheme::setup_prover(NUM_VARS);
    let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
    let setup_time = setup_start.elapsed();
    let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
    let point: Vec<Fr> = (0..NUM_VARS)
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let evaluation = poly.evaluate(&point);
    let commit_start = Instant::now();
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();
    let commit_time = commit_start.elapsed();
    let open_start = Instant::now();
    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof = DoryScheme::open(
        &poly,
        &point,
        evaluation,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .unwrap();
    let open_time = open_start.elapsed();

    let verify_start = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(label);
    DoryScheme::verify(
        &commitment,
        &point,
        evaluation,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .unwrap();
    let verify_time = verify_start.elapsed();

    let rho = ArkFr(<ArkScalar as ark_ff::UniformRand>::rand(&mut rng));
    let ark_commitment = ark_commitment(&commitment);
    let commitments = split_commitment(
        &ark_commitment.0 .0,
        &verifier_setup.0.ht.0 .0,
        rho,
        NUM_COMMITMENTS,
        &mut rng,
    );
    let ark_point: Vec<_> = point.iter().rev().copied().map(jolt_to_ark_fr).collect();
    let deferred_start = Instant::now();
    let relation = build_relation(
        &commitments,
        rho,
        jolt_to_ark_fr(evaluation),
        &ark_point,
        &proof.0,
        &verifier_setup.0,
        label,
    );
    let full_scalars = scalar_magnitudes(&relation.scalars);
    let fq12_bases: Vec<_> = relation.bases.iter().map(|base| base.0 .0).collect();
    let (glv4_bases, glv4_scalars, glv4_bits) = glv_expansion(&fq12_bases, &full_scalars, 4);
    let (rhs, _) = gt_pippenger(&glv4_bases, &glv4_scalars, 7, glv4_bits);
    assert_eq!(relation.lhs.0 .0, rhs);
    let deferred_time = deferred_start.elapsed();

    let (gt_straus_bases, gt_straus_scalars) = gt_glv_expansion_raw(&fq12_bases, &relation.scalars);
    let g1_expected = relation
        .e1_bases
        .iter()
        .zip(&relation.e1_scalars)
        .fold(G1Projective::zero(), |sum, (base, scalar)| {
            sum + *base * scalar.0
        });
    let (g1_straus_bases, g1_straus_scalars) =
        g1_glv_expansion_raw(&relation.e1_bases, &relation.e1_scalars);
    let g2_expected = relation
        .e2_bases
        .iter()
        .zip(&relation.e2_scalars)
        .fold(G2Projective::zero(), |sum, (base, scalar)| {
            sum + *base * scalar.0
        });
    let (g2_straus_bases, g2_straus_scalars) =
        g2_glv_expansion_raw(&relation.e2_bases, &relation.e2_scalars);
    for window in [3, 4, 5] {
        for signed in [false, true] {
            assert_eq!(
                gt_straus(&gt_straus_bases, &gt_straus_scalars, window, signed,),
                rhs,
            );
            assert_eq!(
                ec_straus(&g1_straus_bases, &g1_straus_scalars, window, signed,),
                g1_expected,
            );
            assert_eq!(
                ec_straus(&g2_straus_bases, &g2_straus_scalars, window, signed,),
                g2_expected,
            );
        }
    }
    println!("STRAUS_EQUALITY real_proof=true windows=3,4,5 variants=fixed,signed groups=GT,G1,G2");

    let (rhs_plain, _) = gt_pippenger(&fq12_bases, &full_scalars, 8, SCALAR_BITS);
    assert_eq!(rhs, rhs_plain);
    let (glv2_bases, glv2_scalars, glv2_bits) = glv_expansion(&fq12_bases, &full_scalars, 2);
    let (rhs_glv2, _) = gt_pippenger(&glv2_bases, &glv2_scalars, 8, glv2_bits);
    assert_eq!(rhs, rhs_glv2);

    for index in 0..relation.bases.len() {
        let mut tampered = fq12_bases.clone();
        tampered[index] *= &verifier_setup.0.ht.0 .0;
        let (tampered_rhs, _) = gt_pippenger(&tampered, &full_scalars, 8, SCALAR_BITS);
        assert_ne!(
            relation.lhs.0 .0, tampered_rhs,
            "tampered {} accepted",
            relation.names[index]
        );
    }
    let wrong_commitment_value = commitments[0].0 .0 * verifier_setup.0.ht.0 .0;
    let wrong_commitment = jolt_commitment(&wrong_commitment_value);
    let mut wrong_transcript = Blake2bTranscript::new(label);
    assert!(DoryScheme::verify(
        &wrong_commitment,
        &point,
        evaluation,
        &proof,
        &verifier_setup,
        &mut wrong_transcript,
    )
    .is_err());

    println!(
        "WALL setup_ms={:.3} commit_ms={:.3} open_ms={:.3} dory_verify_ms={:.3} deferred_ms={:.3}",
        setup_time.as_secs_f64() * 1e3,
        commit_time.as_secs_f64() * 1e3,
        open_time.as_secs_f64() * 1e3,
        verify_time.as_secs_f64() * 1e3,
        deferred_time.as_secs_f64() * 1e3,
    );
    println!(
        "RELATION sigma={} N={} gt_bases={} e1_terms={} e2_terms={} tampered_bases_checked={}",
        proof.0.sigma,
        NUM_COMMITMENTS,
        relation.bases.len(),
        relation.e1_scalars.len(),
        relation.e2_scalars.len(),
        relation.bases.len()
    );
    print_gt_counts(proof.0.sigma, NUM_COMMITMENTS, &relation.scalars);
    print_ec_counts(proof.0.sigma, "G1", &relation.e1_scalars);
    print_ec_counts(proof.0.sigma, "G2", &relation.e2_scalars);
    print_straus_counts(
        proof.0.sigma,
        &relation.scalars,
        &relation.e1_scalars,
        &relation.e2_scalars,
    );

    let commitments_42 = split_commitment(
        &ark_commitment.0 .0,
        &verifier_setup.0.ht.0 .0,
        rho,
        42,
        &mut rng,
    );
    let relation_42 = build_relation(
        &commitments_42,
        rho,
        jolt_to_ark_fr(evaluation),
        &ark_point,
        &proof.0,
        &verifier_setup.0,
        label,
    );
    print_gt_counts(proof.0.sigma, 42, &relation_42.scalars);

    let (scalars_12, e1_scalars_12, e2_scalars_12) = synthetic_scalars(12, NUM_COMMITMENTS);
    print_gt_counts(12, NUM_COMMITMENTS, &scalars_12);
    print_ec_counts(12, "G1", &e1_scalars_12);
    print_ec_counts(12, "G2", &e2_scalars_12);
    print_straus_counts(12, &scalars_12, &e1_scalars_12, &e2_scalars_12);
    print_pairing_counts();
}
