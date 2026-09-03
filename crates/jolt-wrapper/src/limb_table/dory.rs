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
use crate::relation::DoryScalar;
use std::collections::HashMap;

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

    /// Every element's `Fq` coordinates in [`input_elements`] order.
    pub fn coordinates(&self) -> Vec<ark_bn254::Fq> {
        self.coordinates_in(&input_elements(self.sigma(), self.commitments.len()))
    }

    /// The elements' `Fq` coordinates in the given order (the layout's
    /// `input_order`); the table's `Input` rows hold exactly this vector.
    pub fn coordinates_in(&self, order: &[InputElement]) -> Vec<ark_bn254::Fq> {
        let mut out = Vec::new();
        for &element in order {
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

fn inv(x: Fr) -> Fr {
    x.inverse()
        .unwrap_or_else(|| unreachable!("transcript challenges are nonzero"))
}

/// The scalar of one base: a named verifier wire of the R1CS lane
/// (`jolt_wrapper::relation::DoryScalar`) or the constant one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Wire {
    Named(DoryScalar),
    One,
}

/// A GT base of `RHS = Σ_k s_k X_k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GtBase {
    Input(InputElement),
    /// `χ[k]` of the verifier key.
    Chi(usize),
    /// `Δ1R[k]` of the verifier key (`k = σ − round`).
    Delta1R(usize),
    Delta2R(usize),
    Ht,
}

/// A G1 base: a committed element, the setup generator `Γ1_0`, or the
/// negated output of the accumulator chain (`−E1_acc`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum G1Base {
    Input(InputElement),
    Gamma1Zero,
    NegAcc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum G2Base {
    Input(InputElement),
    Gamma2Zero,
    NegAcc,
}

/// `Σ_i wire_i · base_i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Msm<B> {
    pub bases: Vec<(B, Wire)>,
}

/// The flattened deferred check with the pairing inputs regrouped so that
/// every scalar is a named wire: `RHS = Σ_k s_k X_k` and
/// `LHS = e(A1, E2_fin) · e(H1, B2) · e(A3, H2) · e(A4, Γ2_0)` with
/// `A1 = E1_fin + d·Γ1_0`, `A3 = γ⁻¹·(−E1_acc) + (−γ⁻¹ d s2)·Γ1_0`,
/// `A4 = d²·E1_init + d⁻¹·E1_fin + Γ1_0` (the `d⁻¹·A1` term of the original
/// `e(A1, E2_fin + d⁻¹Γ2_0)` moved onto the `Γ2_0` pair by bilinearity),
/// `B2 = γ·(−E2_acc') + (−γ d⁻¹ s1)·Γ2_0`, `E2_acc' = E2_acc + y·Γ2_0`.
/// The structure depends only on `(σ, n)`; the wire values on the statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlattenedCheck {
    pub sigma: usize,
    pub n: usize,
    pub gt: Msm<GtBase>,
    /// `E1_acc = Σ_j β_j E1β_j + α_j E1+_j + α_j⁻¹ E1−_j + E1_init` (`3σ + 1` bases).
    pub g1_acc: Msm<G1Base>,
    pub g1_a3: Msm<G1Base>,
    pub g1_a1: Msm<G1Base>,
    pub g1_a4: Msm<G1Base>,
    /// `E2_acc' = Σ_j β_j⁻¹ E2β_j + α_j E2+_j + α_j⁻¹ E2−_j + y·Γ2_0` (`3σ + 1` bases).
    pub g2_acc: Msm<G2Base>,
    pub g2_b2: Msm<G2Base>,
}

impl FlattenedCheck {
    /// The closed form of `dory-offload-study.md` §1.4 (the one owner of the
    /// base/wire pairing).
    pub fn derive(sigma: usize, n: usize) -> Self {
        use DoryScalar as S;
        let mut gt: Vec<(GtBase, Wire)> = Vec::with_capacity(9 * sigma + n + 4);
        gt.push((GtBase::Input(InputElement::VmvC), Wire::One));
        for i in 0..n {
            gt.push((
                GtBase::Input(InputElement::Commitment(i)),
                Wire::Named(S::CommitmentWeight(i)),
            ));
        }
        gt.push((GtBase::Input(InputElement::VmvD2), Wire::Named(S::D2Init)));
        for round in 0..sigma {
            let k = sigma - 1 - round;
            gt.extend([
                (
                    GtBase::Input(InputElement::CPlus(round)),
                    Wire::Named(S::Alpha(round)),
                ),
                (
                    GtBase::Input(InputElement::CMinus(round)),
                    Wire::Named(S::AlphaInv(round)),
                ),
                (
                    GtBase::Input(InputElement::D1Left(round)),
                    Wire::Named(S::UAlpha(round)),
                ),
                (
                    GtBase::Input(InputElement::D1Right(round)),
                    Wire::Named(S::U(round)),
                ),
                (
                    GtBase::Input(InputElement::D2Left(round)),
                    Wire::Named(S::VAlphaInv(round)),
                ),
                (
                    GtBase::Input(InputElement::D2Right(round)),
                    Wire::Named(S::V(round)),
                ),
                (GtBase::Delta1R(sigma - round), Wire::Named(S::Delta1R(k))),
                (GtBase::Delta2R(sigma - round), Wire::Named(S::Delta2R(k))),
            ]);
        }
        for k in 0..sigma {
            gt.push((GtBase::Chi(k), Wire::Named(S::Chi(k))));
        }
        gt.push((GtBase::Chi(sigma), Wire::One));
        gt.push((GtBase::Ht, Wire::Named(S::Ht)));

        let mut g1_acc = Vec::with_capacity(3 * sigma + 1);
        let mut g2_acc = Vec::with_capacity(3 * sigma + 1);
        for round in 0..sigma {
            g1_acc.extend([
                (
                    G1Base::Input(InputElement::E1Beta(round)),
                    Wire::Named(S::Beta(round)),
                ),
                (
                    G1Base::Input(InputElement::E1Plus(round)),
                    Wire::Named(S::Alpha(round)),
                ),
                (
                    G1Base::Input(InputElement::E1Minus(round)),
                    Wire::Named(S::AlphaInv(round)),
                ),
            ]);
            g2_acc.extend([
                (
                    G2Base::Input(InputElement::E2Beta(round)),
                    Wire::Named(S::BetaInv(round)),
                ),
                (
                    G2Base::Input(InputElement::E2Plus(round)),
                    Wire::Named(S::Alpha(round)),
                ),
                (
                    G2Base::Input(InputElement::E2Minus(round)),
                    Wire::Named(S::AlphaInv(round)),
                ),
            ]);
        }
        g1_acc.push((G1Base::Input(InputElement::VmvE1), Wire::One));
        g2_acc.push((G2Base::Gamma2Zero, Wire::Named(S::Evaluation)));
        Self {
            sigma,
            n,
            gt: Msm { bases: gt },
            g1_acc: Msm { bases: g1_acc },
            g1_a3: Msm {
                bases: vec![
                    (G1Base::NegAcc, Wire::Named(S::GammaInv)),
                    (G1Base::Gamma1Zero, Wire::Named(S::PairingG1ZeroScalar)),
                ],
            },
            g1_a1: Msm {
                bases: vec![
                    (G1Base::Input(InputElement::FinalE1), Wire::One),
                    (G1Base::Gamma1Zero, Wire::Named(S::D)),
                ],
            },
            g1_a4: Msm {
                bases: vec![
                    (G1Base::Input(InputElement::VmvE1), Wire::Named(S::DSquared)),
                    (G1Base::Input(InputElement::FinalE1), Wire::Named(S::DInv)),
                    (G1Base::Gamma1Zero, Wire::One),
                ],
            },
            g2_acc: Msm { bases: g2_acc },
            g2_b2: Msm {
                bases: vec![
                    (G2Base::NegAcc, Wire::Named(S::Gamma)),
                    (G2Base::Gamma2Zero, Wire::Named(S::PairingG2ZeroScalar)),
                ],
            },
        }
    }

    /// Every named wire in first-use order (GT bases, then the G1 and G2
    /// chains): the published digit-base order when no link order is given.
    pub fn wires(&self) -> Vec<DoryScalar> {
        let mut out: Vec<DoryScalar> = Vec::new();
        for wire in self.all_wires() {
            if let Wire::Named(scalar) = wire {
                if !out.contains(scalar) {
                    out.push(scalar.clone());
                }
            }
        }
        out
    }

    /// Every base's wire over all seven MSMs.
    pub fn all_wires(&self) -> impl Iterator<Item = &Wire> {
        self.gt
            .bases
            .iter()
            .map(|(_, w)| w)
            .chain(
                self.g1_chains()
                    .into_iter()
                    .flat_map(|m| m.bases.iter().map(|(_, w)| w)),
            )
            .chain(
                self.g2_chains()
                    .into_iter()
                    .flat_map(|m| m.bases.iter().map(|(_, w)| w)),
            )
    }

    /// How many MSM bases carry `wire` (the digit link divides by it).
    pub fn wire_multiplicity(&self, wire: &Wire) -> usize {
        self.all_wires().filter(|w| *w == wire).count()
    }

    /// The four G1 chains in evaluation order (`A3` reads the accumulator).
    pub fn g1_chains(&self) -> [&Msm<G1Base>; 4] {
        [&self.g1_acc, &self.g1_a3, &self.g1_a1, &self.g1_a4]
    }

    pub fn g2_chains(&self) -> [&Msm<G2Base>; 2] {
        [&self.g2_acc, &self.g2_b2]
    }
}

/// The verifier wires' values for one statement (what the R1CS lane's
/// witness holds; the oracle for the digit link).
#[derive(Clone, Debug)]
pub struct WireValues {
    values: HashMap<DoryScalar, Fr>,
}

impl WireValues {
    pub fn derive(statement: &DoryStatement, sigma: usize, n: usize) -> Self {
        use DoryScalar as S;
        let ch = &statement.challenges;
        assert_eq!(ch.beta.len(), sigma);
        assert_eq!(ch.alpha.len(), sigma);
        let nu = statement.point.len() - sigma;
        assert!(nu <= sigma);
        let point = &statement.point;
        let d = ch.d;
        let d_inv = inv(d);
        let gamma_inv = inv(ch.gamma);
        let mut s2_coords = vec![Fr::zero(); sigma];
        s2_coords[..nu].copy_from_slice(&point[sigma..]);
        let mut values = HashMap::new();
        let mut set = |wire: S, value: Fr| {
            let _ = values.insert(wire, value);
        };
        set(S::Evaluation, statement.evaluation);
        let mut rho_power = inv(ch.beta[0]);
        for i in 0..n {
            set(S::CommitmentWeight(i), rho_power);
            rho_power *= statement.rho;
        }
        set(S::Gamma, ch.gamma);
        set(S::GammaInv, gamma_inv);
        set(S::D, d);
        set(S::DInv, d_inv);
        set(S::DSquared, d * d);
        set(S::D2Init, ch.beta[0] + d * d);
        let mut s1_acc = Fr::one();
        let mut s2_acc = Fr::one();
        let mut chi = vec![Fr::one(); sigma];
        for round in 0..sigma {
            let alpha = ch.alpha[round];
            let alpha_inv = inv(alpha);
            let beta = ch.beta[round];
            let beta_inv = inv(beta);
            let (u, v) = if round + 1 < sigma {
                (inv(ch.beta[round + 1]), ch.beta[round + 1])
            } else {
                (d_inv, d)
            };
            let k = sigma - 1 - round;
            s1_acc *= alpha * (Fr::one() - point[k]) + point[k];
            s2_acc *= alpha_inv * (Fr::one() - s2_coords[k]) + s2_coords[k];
            chi[k] += u * alpha * beta + v * alpha_inv * beta_inv;
            set(S::Beta(round), beta);
            set(S::BetaInv(round), beta_inv);
            set(S::Alpha(round), alpha);
            set(S::AlphaInv(round), alpha_inv);
            set(S::U(round), u);
            set(S::V(round), v);
            set(S::UAlpha(round), u * alpha);
            set(S::VAlphaInv(round), v * alpha_inv);
            set(S::Delta1R(k), u * beta);
            set(S::Delta2R(k), v * beta_inv);
        }
        for (k, value) in chi.into_iter().enumerate() {
            set(S::Chi(k), value);
        }
        set(S::S1Acc, s1_acc);
        set(S::S2Acc, s2_acc);
        set(S::Ht, s1_acc * s2_acc);
        set(S::PairingG2ZeroScalar, -ch.gamma * d_inv * s1_acc);
        set(S::PairingG1ZeroScalar, -gamma_inv * d * s2_acc);
        Self { values }
    }

    /// Wire values taken from an R1CS witness (the committed adapter path).
    pub fn from_wires(pairs: Vec<(DoryScalar, Fr)>) -> Self {
        Self {
            values: pairs.into_iter().collect(),
        }
    }

    pub fn get(&self, wire: &Wire) -> Fr {
        match wire {
            Wire::One => Fr::one(),
            Wire::Named(name) => *self
                .values
                .get(name)
                .unwrap_or_else(|| unreachable!("wire {name:?} has a value")),
        }
    }

    pub fn scalars<B>(&self, msm: &Msm<B>) -> Vec<Fr> {
        msm.bases.iter().map(|(_, wire)| self.get(wire)).collect()
    }
}

/// Native evaluation of the flattened check (the test oracle): plain
/// exponentiations and arkworks' multi-pairing over the regrouped pairs.
pub struct NativeCheck {
    pub rhs: Fq12,
    pub e1_acc: G1Affine,
    pub e2_acc: G2Affine,
    pub pairs: [(G1Affine, G2Affine); 4],
    pub miller: Fq12,
    pub lhs: Fq12,
}

impl NativeCheck {
    pub fn evaluate(
        check: &FlattenedCheck,
        values: &WireValues,
        setup: &DorySetupInputs,
        witness: &DoryWitnessInputs,
    ) -> Self {
        let gt_base = |base: GtBase| match base {
            GtBase::Input(e) => witness.gt(e),
            GtBase::Chi(k) => setup.chi[k],
            GtBase::Delta1R(k) => setup.delta_1r[k],
            GtBase::Delta2R(k) => setup.delta_2r[k],
            GtBase::Ht => setup.ht,
        };
        let mut rhs = Fq12::one();
        for (base, wire) in &check.gt.bases {
            rhs *= gt_base(*base).pow(values.get(wire).into_bigint());
        }
        let g1 = |m: &Msm<G1Base>, acc: G1Affine| {
            m.bases
                .iter()
                .fold(G1Projective::zero(), |sum, (base, wire)| {
                    let point = match base {
                        G1Base::Input(e) => witness.g1(*e),
                        G1Base::Gamma1Zero => setup.g1_0,
                        G1Base::NegAcc => -acc,
                    };
                    sum + point.mul_bigint(values.get(wire).into_bigint())
                })
                .into_affine()
        };
        let g2 = |m: &Msm<G2Base>, acc: G2Affine| {
            m.bases
                .iter()
                .fold(G2Projective::zero(), |sum, (base, wire)| {
                    let point = match base {
                        G2Base::Input(e) => witness.g2(*e),
                        G2Base::Gamma2Zero => setup.g2_0,
                        G2Base::NegAcc => -acc,
                    };
                    sum + point.mul_bigint(values.get(wire).into_bigint())
                })
                .into_affine()
        };
        let e1_acc = g1(&check.g1_acc, G1Affine::identity());
        let e2_acc = g2(&check.g2_acc, G2Affine::identity());
        let pairs = [
            (g1(&check.g1_a1, e1_acc), witness.g2(InputElement::FinalE2)),
            (setup.h1, g2(&check.g2_b2, e2_acc)),
            (g1(&check.g1_a3, e1_acc), setup.h2),
            (g1(&check.g1_a4, e1_acc), setup.g2_0),
        ];
        let miller = Bn254::multi_miller_loop(pairs.map(|p| p.0), pairs.map(|p| p.1)).0;
        let lhs = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(miller))
            .unwrap_or_else(|| unreachable!("Miller loop output is invertible"))
            .0;
        Self {
            rhs,
            e1_acc,
            e2_acc,
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
