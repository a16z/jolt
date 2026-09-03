//! The Dory deferred final check as one flattened equation
//! (`.journals/lanes/dory-offload-study.md` §1.4): the verifier-key constants
//! and committed proof elements the table reads, the public scalars derived
//! from the opening statement, and the native oracle that evaluates the same
//! equation with arkworks arithmetic.

use ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{Field, One, PrimeField, Zero};
use dory::backends::arkworks::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksVerifierSetup, BN254,
};
use dory::messages::ScalarProductMessage;
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_transcript::domain::{Label, LabelWithCount};
use jolt_transcript::{AppendToTranscript, Transcript};

use super::tower::fq12_coords;

/// Fiat-Shamir challenges of one transparent Dory opening, in derivation order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryChallenges {
    pub beta: Vec<Fr>,
    pub alpha: Vec<Fr>,
    pub gamma: Fr,
    pub d: Fr,
}

impl DoryChallenges {
    /// Replays dory-pcs's `verify_evaluation_proof` absorption sequence on a
    /// transcript positioned at the Dory boundary (the adapter is
    /// `jolt_dory`'s, byte for byte).
    pub fn replay<T: Transcript<Challenge = jolt_field::Fr>>(
        proof: &ArkDoryProof,
        transcript: &mut T,
    ) -> Self {
        let mut transcript = TranscriptAdapter { transcript };
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
            beta.push(transcript.challenge_scalar(b"beta").0);
            transcript.append_serde(b"c_plus", &second.c_plus);
            transcript.append_serde(b"c_minus", &second.c_minus);
            transcript.append_serde(b"e1_plus", &second.e1_plus);
            transcript.append_serde(b"e1_minus", &second.e1_minus);
            transcript.append_serde(b"e2_plus", &second.e2_plus);
            transcript.append_serde(b"e2_minus", &second.e2_minus);
            alpha.push(transcript.challenge_scalar(b"alpha").0);
        }
        let gamma = transcript.challenge_scalar(b"gamma").0;
        let final_message = proof
            .final_message
            .as_ref()
            .unwrap_or_else(|| unreachable!("transparent proofs carry a final message"));
        transcript.append_serde(b"final_e1", &final_message.e1);
        transcript.append_serde(b"final_e2", &final_message.e2);
        let d = transcript.challenge_scalar(b"d").0;
        Self {
            beta,
            alpha,
            gamma,
            d,
        }
    }
}

/// Public scalars of the opening statement: the joint-commitment weight `rho`
/// (`C = Σ rho^i C_i`), the opening point in dory-pcs order (the Jolt point
/// reversed), the claimed evaluation, and the challenges.
#[derive(Clone, Debug)]
pub struct DoryStatement {
    pub rho: Fr,
    pub point: Vec<Fr>,
    pub evaluation: Fr,
    pub challenges: DoryChallenges,
}

/// Verifier-key elements the table reads as public rows.
#[derive(Clone, Debug)]
pub struct DorySetupInputs {
    pub chi: Vec<Fq12>,
    pub delta_1r: Vec<Fq12>,
    pub delta_2r: Vec<Fq12>,
    pub ht: Fq12,
    pub g1_0: G1Affine,
    pub g2_0: G2Affine,
    pub h1: G1Affine,
    pub h2: G2Affine,
}

impl From<&ArkworksVerifierSetup> for DorySetupInputs {
    fn from(setup: &ArkworksVerifierSetup) -> Self {
        let gt = |v: &[ArkGT]| v.iter().map(|x| x.0 .0).collect();
        Self {
            chi: gt(&setup.chi),
            delta_1r: gt(&setup.delta_1r),
            delta_2r: gt(&setup.delta_2r),
            ht: setup.ht.0 .0,
            g1_0: setup.g1_0.0.into_affine(),
            g2_0: setup.g2_0.0.into_affine(),
            h1: setup.h1.0.into_affine(),
            h2: setup.h2.0.into_affine(),
        }
    }
}

/// One committed element the table reads as input rows; the order of
/// [`input_elements`] is the byte-link order for the transcript table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputElement {
    Commitment(usize),
    VmvC,
    VmvD2,
    VmvE1,
    D1Left(usize),
    D1Right(usize),
    D2Left(usize),
    D2Right(usize),
    E1Beta(usize),
    E2Beta(usize),
    CPlus(usize),
    CMinus(usize),
    E1Plus(usize),
    E1Minus(usize),
    E2Plus(usize),
    E2Minus(usize),
    FinalE1,
    FinalE2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    Gt,
    G1,
    G2,
}

impl ElementKind {
    /// `Fq` coordinates per element: twelve for GT, affine `(x, y)` for G1,
    /// affine `(x.c0, x.c1, y.c0, y.c1)` for G2.
    pub const fn coords(self) -> usize {
        match self {
            Self::Gt => 12,
            Self::G1 => 2,
            Self::G2 => 4,
        }
    }
}

impl InputElement {
    pub const fn kind(self) -> ElementKind {
        match self {
            Self::Commitment(_)
            | Self::VmvC
            | Self::VmvD2
            | Self::D1Left(_)
            | Self::D1Right(_)
            | Self::D2Left(_)
            | Self::D2Right(_)
            | Self::CPlus(_)
            | Self::CMinus(_) => ElementKind::Gt,
            Self::VmvE1 | Self::E1Beta(_) | Self::E1Plus(_) | Self::E1Minus(_) | Self::FinalE1 => {
                ElementKind::G1
            }
            Self::E2Beta(_) | Self::E2Plus(_) | Self::E2Minus(_) | Self::FinalE2 => ElementKind::G2,
        }
    }
}

/// Canonical order of the committed elements for `sigma` rounds and `n`
/// commitments: commitments, VMV message, per round the first then second
/// reduce message, final message.
pub fn input_elements(sigma: usize, n: usize) -> Vec<InputElement> {
    let mut out = Vec::with_capacity(n + 3 + 12 * sigma + 2);
    out.extend((0..n).map(InputElement::Commitment));
    out.extend([InputElement::VmvC, InputElement::VmvD2, InputElement::VmvE1]);
    for j in 0..sigma {
        out.extend([
            InputElement::D1Left(j),
            InputElement::D1Right(j),
            InputElement::D2Left(j),
            InputElement::D2Right(j),
            InputElement::E1Beta(j),
            InputElement::E2Beta(j),
            InputElement::CPlus(j),
            InputElement::CMinus(j),
            InputElement::E1Plus(j),
            InputElement::E1Minus(j),
            InputElement::E2Plus(j),
            InputElement::E2Minus(j),
        ]);
    }
    out.extend([InputElement::FinalE1, InputElement::FinalE2]);
    out
}

/// The committed elements' values, keyed by [`InputElement`].
#[derive(Clone, Debug)]
pub struct DoryWitnessInputs {
    pub commitments: Vec<Fq12>,
    pub proof: ArkDoryProof,
}

impl DoryWitnessInputs {
    pub fn sigma(&self) -> usize {
        self.proof.sigma
    }

    pub fn gt(&self, element: InputElement) -> Fq12 {
        let (first, second) = (&self.proof.first_messages, &self.proof.second_messages);
        match element {
            InputElement::Commitment(i) => self.commitments[i],
            InputElement::VmvC => self.proof.vmv_message.c.0 .0,
            InputElement::VmvD2 => self.proof.vmv_message.d2.0 .0,
            InputElement::D1Left(j) => first[j].d1_left.0 .0,
            InputElement::D1Right(j) => first[j].d1_right.0 .0,
            InputElement::D2Left(j) => first[j].d2_left.0 .0,
            InputElement::D2Right(j) => first[j].d2_right.0 .0,
            InputElement::CPlus(j) => second[j].c_plus.0 .0,
            InputElement::CMinus(j) => second[j].c_minus.0 .0,
            _ => unreachable!("not a GT element"),
        }
    }

    pub fn g1(&self, element: InputElement) -> G1Affine {
        let (first, second) = (&self.proof.first_messages, &self.proof.second_messages);
        let point = match element {
            InputElement::VmvE1 => self.proof.vmv_message.e1.0,
            InputElement::E1Beta(j) => first[j].e1_beta.0,
            InputElement::E1Plus(j) => second[j].e1_plus.0,
            InputElement::E1Minus(j) => second[j].e1_minus.0,
            InputElement::FinalE1 => self.final_message().e1.0,
            _ => unreachable!("not a G1 element"),
        };
        point.into_affine()
    }

    pub fn g2(&self, element: InputElement) -> G2Affine {
        let (first, second) = (&self.proof.first_messages, &self.proof.second_messages);
        let point = match element {
            InputElement::E2Beta(j) => first[j].e2_beta.0,
            InputElement::E2Plus(j) => second[j].e2_plus.0,
            InputElement::E2Minus(j) => second[j].e2_minus.0,
            InputElement::FinalE2 => self.final_message().e2.0,
            _ => unreachable!("not a G2 element"),
        };
        point.into_affine()
    }

    fn final_message(&self) -> &ScalarProductMessage<ArkG1, ArkG2> {
        self.proof
            .final_message
            .as_ref()
            .unwrap_or_else(|| unreachable!("transparent proofs carry a final message"))
    }

    /// Every element's `Fq` coordinates in [`input_elements`] order; the
    /// table's input rows hold exactly this vector.
    pub fn coordinates(&self) -> Vec<ark_bn254::Fq> {
        let mut out = Vec::new();
        for element in input_elements(self.sigma(), self.commitments.len()) {
            match element.kind() {
                ElementKind::Gt => out.extend(fq12_coords(&self.gt(element))),
                ElementKind::G1 => {
                    let p = self.g1(element);
                    out.extend([p.x, p.y]);
                }
                ElementKind::G2 => {
                    let p = self.g2(element);
                    out.extend([p.x.c0, p.x.c1, p.y.c0, p.y.c1]);
                }
            }
        }
        out
    }
}

/// A base of one of the multi-exponentiations: a committed input or a
/// verifier-key constant (`chi[k]`, `delta_1r[k]`, `delta_2r[k]`, `ht`,
/// `g1_0`, `g2_0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Base {
    Input(InputElement),
    Chi(usize),
    Delta1R(usize),
    Delta2R(usize),
    Ht,
    G1Zero,
    G2Zero,
}

/// One multi-exponentiation `Σ scalar_i · base_i` of the flattened check.
#[derive(Clone, Debug)]
pub struct MultiExp {
    pub bases: Vec<Base>,
    pub scalars: Vec<Fr>,
}

/// The flattened deferred check: `RHS = Σ_k s_k X_k` over the GT bases and
/// the four pairing inputs, each a G1/G2 multi-exponentiation or a constant.
/// `LHS = e(p1) e(H1, p2_g2) e(p3_g1, H2) e(p4_g1, Γ2_0)`.
#[derive(Clone, Debug)]
pub struct FlattenedCheck {
    pub gt: MultiExp,
    /// `E1_fin + d·Γ1_0`, `-γ⁻¹·(E1_acc + d·s2·Γ1_0)`, `d²·E1_init`.
    pub g1: [MultiExp; 3],
    /// `E2_fin + d⁻¹·Γ2_0`, `-γ·(E2_acc + d⁻¹·s1·Γ2_0)`.
    pub g2: [MultiExp; 2],
}

fn inv(x: Fr) -> Fr {
    x.inverse()
        .unwrap_or_else(|| unreachable!("transcript challenges are nonzero"))
}

impl FlattenedCheck {
    /// The closed form of `dory-offload-study.md` §1.4 (the one owner of the
    /// base/scalar pairing).
    pub fn derive(statement: &DoryStatement, sigma: usize, n: usize) -> Self {
        let ch = &statement.challenges;
        assert_eq!(ch.beta.len(), sigma);
        assert_eq!(ch.alpha.len(), sigma);
        let nu = statement.point.len() - sigma;
        assert!(nu <= sigma);
        let point = &statement.point;
        let d = ch.d;
        let d_inv = inv(d);
        let d_sq = d * d;
        let gamma_inv = inv(ch.gamma);
        let beta0_inv = inv(ch.beta[0]);

        let mut s2_coords = vec![Fr::zero(); sigma];
        s2_coords[..nu].copy_from_slice(&point[sigma..]);
        let mut s1_acc = Fr::one();
        let mut s2_acc = Fr::one();
        let mut chi_scalars = vec![Fr::one(); sigma + 1];
        let uv = |round: usize| {
            if round + 1 < sigma {
                (inv(ch.beta[round + 1]), ch.beta[round + 1])
            } else {
                (d_inv, d)
            }
        };
        for round in 0..sigma {
            let alpha = ch.alpha[round];
            let alpha_inv = inv(alpha);
            let idx = sigma - round - 1;
            s1_acc *= alpha * (Fr::one() - point[idx]) + point[idx];
            s2_acc *= alpha_inv * (Fr::one() - s2_coords[idx]) + s2_coords[idx];
            let (u, v) = uv(round);
            chi_scalars[idx] += u * alpha * ch.beta[round] + v * alpha_inv * inv(ch.beta[round]);
        }

        let mut gt = MultiExp {
            bases: Vec::with_capacity(9 * sigma + n + 4),
            scalars: Vec::with_capacity(9 * sigma + n + 4),
        };
        let push = |m: &mut MultiExp, base: Base, scalar: Fr| {
            m.bases.push(base);
            m.scalars.push(scalar);
        };
        push(&mut gt, Base::Input(InputElement::VmvC), Fr::one());
        let mut rho_power = Fr::one();
        for i in 0..n {
            push(
                &mut gt,
                Base::Input(InputElement::Commitment(i)),
                beta0_inv * rho_power,
            );
            rho_power *= statement.rho;
        }
        push(&mut gt, Base::Input(InputElement::VmvD2), ch.beta[0] + d_sq);
        for round in 0..sigma {
            let alpha = ch.alpha[round];
            let alpha_inv = inv(alpha);
            let beta = ch.beta[round];
            let (u, v) = uv(round);
            push(&mut gt, Base::Input(InputElement::CPlus(round)), alpha);
            push(&mut gt, Base::Input(InputElement::CMinus(round)), alpha_inv);
            push(&mut gt, Base::Input(InputElement::D1Left(round)), u * alpha);
            push(&mut gt, Base::Input(InputElement::D1Right(round)), u);
            push(
                &mut gt,
                Base::Input(InputElement::D2Left(round)),
                v * alpha_inv,
            );
            push(&mut gt, Base::Input(InputElement::D2Right(round)), v);
            push(&mut gt, Base::Delta1R(sigma - round), u * beta);
            push(&mut gt, Base::Delta2R(sigma - round), v * inv(beta));
        }
        for (k, scalar) in chi_scalars.into_iter().enumerate() {
            push(&mut gt, Base::Chi(k), scalar);
        }
        push(&mut gt, Base::Ht, s1_acc * s2_acc);

        let mut p3_g1 = MultiExp {
            bases: Vec::with_capacity(3 * sigma + 2),
            scalars: Vec::with_capacity(3 * sigma + 2),
        };
        let mut p2_g2 = MultiExp {
            bases: Vec::with_capacity(3 * sigma + 1),
            scalars: Vec::with_capacity(3 * sigma + 1),
        };
        for round in 0..sigma {
            let alpha = ch.alpha[round];
            let alpha_inv = inv(alpha);
            let beta = ch.beta[round];
            push(
                &mut p3_g1,
                Base::Input(InputElement::E1Beta(round)),
                -gamma_inv * beta,
            );
            push(
                &mut p3_g1,
                Base::Input(InputElement::E1Plus(round)),
                -gamma_inv * alpha,
            );
            push(
                &mut p3_g1,
                Base::Input(InputElement::E1Minus(round)),
                -gamma_inv * alpha_inv,
            );
            push(
                &mut p2_g2,
                Base::Input(InputElement::E2Beta(round)),
                -ch.gamma * inv(beta),
            );
            push(
                &mut p2_g2,
                Base::Input(InputElement::E2Plus(round)),
                -ch.gamma * alpha,
            );
            push(
                &mut p2_g2,
                Base::Input(InputElement::E2Minus(round)),
                -ch.gamma * alpha_inv,
            );
        }
        push(&mut p3_g1, Base::Input(InputElement::VmvE1), -gamma_inv);
        push(&mut p3_g1, Base::G1Zero, -gamma_inv * d * s2_acc);
        push(
            &mut p2_g2,
            Base::G2Zero,
            -ch.gamma * (statement.evaluation + d_inv * s1_acc),
        );
        let p1_g1 = MultiExp {
            bases: vec![Base::Input(InputElement::FinalE1), Base::G1Zero],
            scalars: vec![Fr::one(), d],
        };
        let p4_g1 = MultiExp {
            bases: vec![Base::Input(InputElement::VmvE1)],
            scalars: vec![d_sq],
        };
        let p1_g2 = MultiExp {
            bases: vec![Base::Input(InputElement::FinalE2), Base::G2Zero],
            scalars: vec![Fr::one(), d_inv],
        };
        Self {
            gt,
            g1: [p1_g1, p3_g1, p4_g1],
            g2: [p1_g2, p2_g2],
        }
    }
}

/// Native evaluation of the flattened check (the test oracle): plain
/// exponentiations and arkworks' multi-pairing.
pub struct NativeCheck {
    pub rhs: Fq12,
    pub pairs: [(G1Affine, G2Affine); 4],
    pub miller: Fq12,
    pub lhs: Fq12,
}

impl NativeCheck {
    pub fn evaluate(
        check: &FlattenedCheck,
        setup: &DorySetupInputs,
        witness: &DoryWitnessInputs,
    ) -> Self {
        let gt_base = |base: Base| match base {
            Base::Input(e) => witness.gt(e),
            Base::Chi(k) => setup.chi[k],
            Base::Delta1R(k) => setup.delta_1r[k],
            Base::Delta2R(k) => setup.delta_2r[k],
            Base::Ht => setup.ht,
            Base::G1Zero | Base::G2Zero => unreachable!("not a GT base"),
        };
        let mut rhs = Fq12::one();
        for (base, scalar) in check.gt.bases.iter().zip(&check.gt.scalars) {
            rhs *= gt_base(*base).pow(scalar.into_bigint());
        }
        let g1 = |m: &MultiExp| {
            m.bases
                .iter()
                .zip(&m.scalars)
                .fold(G1Projective::zero(), |acc, (base, scalar)| {
                    let point = match base {
                        Base::Input(e) => witness.g1(*e),
                        Base::G1Zero => setup.g1_0,
                        _ => unreachable!("not a G1 base"),
                    };
                    acc + point.mul_bigint(scalar.into_bigint())
                })
                .into_affine()
        };
        let g2 = |m: &MultiExp| {
            m.bases
                .iter()
                .zip(&m.scalars)
                .fold(G2Projective::zero(), |acc, (base, scalar)| {
                    let point = match base {
                        Base::Input(e) => witness.g2(*e),
                        Base::G2Zero => setup.g2_0,
                        _ => unreachable!("not a G2 base"),
                    };
                    acc + point.mul_bigint(scalar.into_bigint())
                })
                .into_affine()
        };
        let pairs = [
            (g1(&check.g1[0]), g2(&check.g2[0])),
            (setup.h1, g2(&check.g2[1])),
            (g1(&check.g1[1]), setup.h2),
            (g1(&check.g1[2]), setup.g2_0),
        ];
        let miller = Bn254::multi_miller_loop(pairs.map(|p| p.0), pairs.map(|p| p.1)).0;
        let lhs = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(miller))
            .unwrap_or_else(|| unreachable!("Miller loop output is invertible"))
            .0;
        Self {
            rhs,
            pairs,
            miller,
            lhs,
        }
    }

    pub fn holds(&self) -> bool {
        self.lhs == self.rhs
    }
}

/// `jolt_dory`'s private Jolt-to-dory-pcs transcript adapter, replicated
/// byte for byte (labels are dropped; lengths are absorbed with the bytes).
struct TranscriptAdapter<'a, T: Transcript<Challenge = jolt_field::Fr>> {
    transcript: &'a mut T,
}

impl<T: Transcript<Challenge = jolt_field::Fr>> DoryTranscript for TranscriptAdapter<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript
            .append(&LabelWithCount(b"dory_bytes", bytes.len() as u64));
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], value: &ArkFr) {
        self.transcript.append(&Label(b"dory_field"));
        jolt_field::Fr::from(value.0).append_to_transcript(self.transcript);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], value: &G) {
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .unwrap_or_else(|_| unreachable!("group serialization cannot fail"));
        self.transcript
            .append(&LabelWithCount(b"dory_group", bytes.len() as u64));
        self.transcript.append_bytes(&bytes);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], value: &S) {
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .unwrap_or_else(|_| unreachable!("DorySerialize cannot fail"));
        self.transcript
            .append(&LabelWithCount(b"dory_serde", bytes.len() as u64));
        self.transcript.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        ArkFr(self.transcript.challenge_scalar().into())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("dory-pcs does not reset the transcript")
    }
}
