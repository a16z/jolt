//! HyperKZG commitment scheme implementing `jolt-openings` traits.
//!
//! [`HyperKZGScheme`] is generic over `P: PairingGroup` — instantiate with
//! `Bn254` for the concrete BN254 curve.

#![expect(
    clippy::expect_used,
    reason = "KZG operations return Result for API symmetry; with a correctly-sized SRS and well-formed inputs these errors are unreachable"
)]

use std::marker::PhantomData;

use jolt_crypto::{Commitment, JoltGroup, PairingGroup};
use jolt_field::FromPrimitiveInt;
#[cfg(feature = "zk")]
use jolt_field::Invertible;
#[cfg(feature = "zk")]
use jolt_field::RandomSampling;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use num_traits::{One, Zero};
#[cfg(feature = "zk")]
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::HyperKZGError;
use crate::kzg::{self, kzg_open_batch, kzg_verify_batch};
use crate::types::{
    HyperKZGCommitment, HyperKZGOpeningHint, HyperKZGProof, HyperKZGProofKind, HyperKZGProverSetup,
    HyperKZGVerifierSetup,
};
#[cfg(feature = "zk")]
use crate::types::{
    HyperKZGHiddenEvaluationCommitments, HyperKZGProofPayload, HyperKZGZkOpenOutput,
};

#[cfg(feature = "zk")]
type HyperKZGTauRows<P> = [Vec<<P as PairingGroup>::ScalarField>; 3];

trait HyperKZGMode {
    const PROOF_KIND: HyperKZGProofKind;
}

struct Transparent;

impl HyperKZGMode for Transparent {
    const PROOF_KIND: HyperKZGProofKind = HyperKZGProofKind::Clear;
}

#[cfg(feature = "zk")]
struct Zk;

#[cfg(feature = "zk")]
impl HyperKZGMode for Zk {
    const PROOF_KIND: HyperKZGProofKind = HyperKZGProofKind::Zk;
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy)]
struct V2Bases<P: PairingGroup> {
    value: P::G1,
    hiding: P::G1,
    beta_hiding: P::G1,
}

/// HyperKZG multilinear polynomial commitment scheme.
///
/// Generic over `P: PairingGroup`. Implements [`CommitmentScheme`] and
/// [`AdditivelyHomomorphic`] from `jolt-openings`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]

pub struct HyperKZGScheme<P: PairingGroup> {
    #[serde(skip)]
    _phantom: PhantomData<P>,
}

impl<P: PairingGroup> HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    /// Phase 1 of the HyperKZG protocol: fold the multilinear polynomial.
    ///
    /// Given polynomial $P$ with $2^\ell$ evaluations and opening point
    /// $x = (x_1, \ldots, x_\ell)$, produces $\ell$ polynomials
    /// $P_0 = P, P_1, \ldots, P_{\ell-1}$ where each $P_i$ has half
    /// the length of $P_{i-1}$.
    ///
    /// The folding relation is:
    /// $P_i[j] = (1 - x_{\ell-i}) \cdot P_{i-1}[2j] + x_{\ell-i} \cdot P_{i-1}[2j+1]$
    fn fold_polynomials(
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
    ) -> Vec<Vec<P::ScalarField>> {
        let ell = point.len();
        let mut polys = Vec::with_capacity(ell);
        polys.push(evals.to_vec());

        for i in 0..ell - 1 {
            let prev = &polys[i];
            let half = prev.len() / 2;
            let xi = point[ell - i - 1];
            let mut pi = vec![P::ScalarField::zero(); half];
            pi.par_iter_mut().enumerate().for_each(|(j, pj)| {
                *pj = prev[2 * j] + xi * (prev[2 * j + 1] - prev[2 * j]);
            });
            polys.push(pi);
        }

        polys
    }

    fn dense_evaluations<S: MultilinearPoly<P::ScalarField> + ?Sized>(
        poly: &S,
    ) -> Vec<P::ScalarField> {
        let mut evaluations = Vec::with_capacity(1 << poly.num_vars());
        poly.for_each_row(poly.num_vars(), &mut |_, row| {
            evaluations.extend_from_slice(row);
        });
        evaluations
    }

    fn commit_with_mode<M, S>(
        poly: &S,
        setup: &HyperKZGProverSetup<P>,
    ) -> (HyperKZGCommitment<P>, HyperKZGOpeningHint<P>)
    where
        M: HyperKZGMode,
        S: MultilinearPoly<P::ScalarField> + ?Sized,
    {
        let evaluations = Self::dense_evaluations(poly);
        #[cfg(not(feature = "zk"))]
        {
            debug_assert_eq!(M::PROOF_KIND, HyperKZGProofKind::Clear);
            let point = kzg::kzg_commit::<P>(&evaluations, setup)
                .expect("SRS must be large enough for the polynomial");
            (HyperKZGCommitment { point }, HyperKZGOpeningHint::clear())
        }
        #[cfg(feature = "zk")]
        {
            match M::PROOF_KIND {
                HyperKZGProofKind::Clear => {
                    let point = kzg::kzg_commit::<P>(&evaluations, setup)
                        .expect("SRS must be large enough for the polynomial");
                    (HyperKZGCommitment { point }, HyperKZGOpeningHint::clear())
                }
                HyperKZGProofKind::Zk => {
                    let (hiding_base, _) =
                        Self::zk_hiding_bases(setup).expect("ZK SRS must contain hiding bases");
                    let mut rng = Self::zk_rng();
                    let blind = P::ScalarField::random(&mut rng);
                    let transparent = kzg::kzg_commit::<P>(&evaluations, setup)
                        .expect("SRS must be large enough for the polynomial");
                    let point = kzg::blind_commitment::<P>(transparent, hiding_base, blind);
                    (HyperKZGCommitment { point }, HyperKZGOpeningHint::zk(blind))
                }
            }
        }
    }

    /// Full HyperKZG opening proof.
    #[tracing::instrument(skip_all, name = "HyperKZG::open")]
    pub fn open<T: Transcript<Challenge = P::ScalarField>>(
        setup: &HyperKZGProverSetup<P>,
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
        transcript: &mut T,
    ) -> Result<HyperKZGProof<P>, HyperKZGError> {
        Self::open_with_mode::<Transparent, T>(setup, evals, point, transcript)
    }

    fn open_with_mode<M, T>(
        setup: &HyperKZGProverSetup<P>,
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
        transcript: &mut T,
    ) -> Result<HyperKZGProof<P>, HyperKZGError>
    where
        M: HyperKZGMode,
        T: Transcript<Challenge = P::ScalarField>,
    {
        debug_assert_eq!(M::PROOF_KIND, HyperKZGProofKind::Clear);
        let ell = point.len();
        if ell == 0 {
            return Err(HyperKZGError::EmptyPoint);
        }
        let n = evals.len();
        assert_eq!(n, 1 << ell, "evaluation count must be 2^ell");

        // Phase 1: fold
        let polys = Self::fold_polynomials(evals, point);
        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // Commit to intermediate polynomials (skip polys[0] — already committed)
        let com: Vec<P::G1> = polys[1..]
            .par_iter()
            .map(|p| kzg::kzg_commit::<P>(p, setup).expect("SRS large enough for intermediate"))
            .collect();

        // Phase 2: derive challenge r
        for c in &com {
            transcript.append(c);
        }
        let r: P::ScalarField = transcript.challenge();
        let u = [r, -r, r * r];

        // Phase 3: batch open all polynomials at the three points
        let (w, v) = kzg_open_batch::<P, T>(&polys, &u, setup, transcript);

        Ok(HyperKZGProof::clear(com, w, v))
    }

    #[cfg(feature = "zk")]
    fn combine_polys_with_powers(
        polys: &[Vec<P::ScalarField>],
        powers: &[P::ScalarField],
    ) -> Vec<P::ScalarField> {
        let mut combined = vec![P::ScalarField::zero(); polys[0].len()];
        for (poly, &power) in polys.iter().zip(powers) {
            for (acc, &coeff) in combined.iter_mut().zip(poly) {
                *acc += power * coeff;
            }
        }
        combined
    }

    #[cfg(feature = "zk")]
    fn combine_scalars_with_powers(
        scalars: &[P::ScalarField],
        powers: &[P::ScalarField],
    ) -> P::ScalarField {
        scalars
            .iter()
            .zip(powers)
            .map(|(&scalar, &power)| scalar * power)
            .fold(P::ScalarField::zero(), |acc, value| acc + value)
    }

    #[cfg(feature = "zk")]
    fn zk_rng() -> rand_chacha::ChaCha20Rng {
        let mut seed = <rand_chacha::ChaCha20Rng as SeedableRng>::Seed::default();
        rand_core::OsRng.fill_bytes(&mut seed);
        rand_chacha::ChaCha20Rng::from_seed(seed)
    }

    #[cfg(feature = "zk")]
    fn zk_hiding_bases(setup: &HyperKZGProverSetup<P>) -> Result<(P::G1, P::G1), HyperKZGError> {
        let hiding_g1_sequence = setup
            .hiding_g1_sequence
            .as_ref()
            .ok_or(HyperKZGError::MissingZkSrs)?;
        if hiding_g1_sequence.len() < 2 {
            return Err(HyperKZGError::SrsTooSmall {
                have: hiding_g1_sequence.len(),
                need: 2,
            });
        }
        Ok((hiding_g1_sequence[0], hiding_g1_sequence[1]))
    }

    #[cfg(feature = "zk")]
    fn append_zk_opening_prefix<T>(
        point: &[P::ScalarField],
        y_out: &P::G1,
        com: &[P::G1],
        transcript: &mut T,
    ) -> P::ScalarField
    where
        T: Transcript<Challenge = P::ScalarField>,
    {
        transcript.append(&LabelWithCount(b"hyperkzg_zk_point", point.len() as u64));
        for value in point {
            value.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"hyperkzg_zk_y_out"));
        y_out.append_to_transcript(transcript);
        transcript.append(&LabelWithCount(b"hyperkzg_zk_fold_com", com.len() as u64));
        for commitment in com {
            commitment.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"hyperkzg_zk_gemini"));
        transcript.challenge()
    }

    #[cfg(feature = "zk")]
    fn append_zk_eval_commitments<T>(
        y: &[Vec<P::G1>; 3],
        transcript: &mut T,
    ) -> (P::ScalarField, P::ScalarField)
    where
        T: Transcript<Challenge = P::ScalarField>,
    {
        for (label, row) in [
            b"hyperkzg_zk_eval_r".as_slice(),
            b"hyperkzg_zk_eval_neg_r".as_slice(),
            b"hyperkzg_zk_eval_r2".as_slice(),
        ]
        .into_iter()
        .zip(y)
        {
            transcript.append(&LabelWithCount(label, row.len() as u64));
            for commitment in row {
                commitment.append_to_transcript(transcript);
            }
        }
        transcript.append(&Label(b"hyperkzg_zk_gemini_batch"));
        let alpha = transcript.challenge();
        transcript.append(&Label(b"hyperkzg_zk_eval_batch"));
        let q = transcript.challenge();
        (alpha, q)
    }

    #[cfg(feature = "zk")]
    fn append_zk_witnesses<T>(w: &[P::G1; 3], transcript: &mut T) -> P::ScalarField
    where
        T: Transcript<Challenge = P::ScalarField>,
    {
        for (label, witness) in [
            b"hyperkzg_zk_wit_r".as_slice(),
            b"hyperkzg_zk_wit_nr".as_slice(),
            b"hyperkzg_zk_wit_r2".as_slice(),
        ]
        .into_iter()
        .zip(w)
        {
            transcript.append(&Label(label));
            witness.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"hyperkzg_zk_wit_batch"));
        transcript.challenge()
    }

    #[cfg(feature = "zk")]
    fn solve_v2_fold_randomizers(
        r: P::ScalarField,
        xi: P::ScalarField,
        rho_i: P::ScalarField,
        next_a: P::ScalarField,
        next_b: P::ScalarField,
    ) -> Result<(P::ScalarField, P::ScalarField), HyperKZGError> {
        let one_minus_xi = P::ScalarField::one() - xi;
        let denom_pos = r * one_minus_xi + xi;
        let denom_neg = xi - r * one_minus_xi;
        let denom_pos_inv = denom_pos
            .inverse()
            .ok_or(HyperKZGError::DegenerateChallenge)?;
        let denom_neg_inv = denom_neg
            .inverse()
            .ok_or(HyperKZGError::DegenerateChallenge)?;

        let base = next_a - one_minus_xi * rho_i;
        let tau_pos = (base - r * next_b) * denom_pos_inv;
        let tau_neg = (base + r * next_b) * denom_neg_inv;
        Ok((tau_pos, tau_neg))
    }

    #[cfg(feature = "zk")]
    fn v2_hidden_evaluation_rows(
        polys: &[Vec<P::ScalarField>],
        point: &[P::ScalarField],
        rho: &[P::ScalarField],
        lambda: P::ScalarField,
        u: &[P::ScalarField; 3],
        bases: V2Bases<P>,
        rng: &mut rand_chacha::ChaCha20Rng,
    ) -> Result<(HyperKZGHiddenEvaluationCommitments<P>, HyperKZGTauRows<P>), HyperKZGError> {
        let ell = polys.len();
        debug_assert_eq!(point.len(), ell);
        debug_assert_eq!(rho.len(), ell);

        let tau_sq: Vec<P::ScalarField> = (0..ell).map(|_| P::ScalarField::random(rng)).collect();
        let mut tau_pos = vec![P::ScalarField::zero(); ell];
        let mut tau_neg = vec![P::ScalarField::zero(); ell];

        for i in 0..ell {
            let (next_a, next_b) = if i + 1 < ell {
                (rho[i + 1] + u[2] * tau_sq[i + 1], -tau_sq[i + 1])
            } else {
                (lambda, P::ScalarField::zero())
            };
            let xi = point[ell - i - 1];
            let (pos, neg) = Self::solve_v2_fold_randomizers(u[0], xi, rho[i], next_a, next_b)?;
            tau_pos[i] = pos;
            tau_neg[i] = neg;
        }

        let tau = [tau_pos, tau_neg, tau_sq];
        let row = |u_i, tau_i: &[P::ScalarField]| {
            polys
                .iter()
                .zip(rho)
                .zip(tau_i)
                .map(|((poly, &rho_i), &tau_i_u)| {
                    let value = kzg::eval_univariate(poly, u_i);
                    kzg::randomized_eval_commitment::<P>(
                        bases.value,
                        bases.hiding,
                        bases.beta_hiding,
                        value,
                        rho_i,
                        tau_i_u,
                        u_i,
                    )
                })
                .collect()
        };
        let (row_r, (row_neg_r, row_r2)) = rayon::join(
            || row(u[0], &tau[0]),
            || rayon::join(|| row(u[1], &tau[1]), || row(u[2], &tau[2])),
        );
        Ok(([row_r, row_neg_r, row_r2], tau))
    }

    #[cfg(feature = "zk")]
    fn open_zk_inner<T>(
        setup: &HyperKZGProverSetup<P>,
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
        claimed_eval: P::ScalarField,
        rho_0: P::ScalarField,
        transcript: &mut T,
    ) -> Result<HyperKZGZkOpenOutput<P>, HyperKZGError>
    where
        T: Transcript<Challenge = P::ScalarField>,
    {
        let ell = point.len();
        if ell == 0 {
            return Err(HyperKZGError::EmptyPoint);
        }
        let n = evals.len();
        assert_eq!(n, 1 << ell, "evaluation count must be 2^ell");

        let (hiding_base, beta_hiding_base) = Self::zk_hiding_bases(setup)?;
        let mut rng = Self::zk_rng();

        let polys = Self::fold_polynomials(evals, point);
        let output_blind = P::ScalarField::random(&mut rng);
        let y_out =
            setup.g1_powers[0].scalar_mul(&claimed_eval) + hiding_base.scalar_mul(&output_blind);

        let mut rho = vec![P::ScalarField::zero(); ell];
        rho[0] = rho_0;
        for blind in &mut rho[1..] {
            *blind = P::ScalarField::random(&mut rng);
        }

        let com: Vec<P::G1> = polys[1..]
            .par_iter()
            .zip(&rho[1..])
            .map(|(poly, &blind)| {
                let transparent =
                    kzg::kzg_commit::<P>(poly, setup).expect("SRS large enough for intermediate");
                kzg::blind_commitment::<P>(transparent, hiding_base, blind)
            })
            .collect();

        let r = Self::append_zk_opening_prefix(point, &y_out, &com, transcript);
        if r.is_zero() {
            return Err(HyperKZGError::DegenerateChallenge);
        }
        let u = [r, -r, r * r];

        let (y, tau) = Self::v2_hidden_evaluation_rows(
            &polys,
            point,
            &rho,
            output_blind,
            &u,
            V2Bases {
                value: setup.g1_powers[0],
                hiding: hiding_base,
                beta_hiding: beta_hiding_base,
            },
            &mut rng,
        )?;
        let (_, q) = Self::append_zk_eval_commitments(&y, transcript);
        let q_powers = kzg::challenge_powers(q, polys.len());
        let batched_poly = Self::combine_polys_with_powers(&polys, &q_powers);

        let tau_q = [
            Self::combine_scalars_with_powers(&tau[0], &q_powers),
            Self::combine_scalars_with_powers(&tau[1], &q_powers),
            Self::combine_scalars_with_powers(&tau[2], &q_powers),
        ];
        let witness = |i: usize| {
            let transparent = kzg::kzg_witness_commitment::<P>(&batched_poly, u[i], setup)
                .expect("SRS large enough for witness");
            transparent + hiding_base.scalar_mul(&tau_q[i])
        };
        let (w_r, (w_neg_r, w_r2)) =
            rayon::join(|| witness(0), || rayon::join(|| witness(1), || witness(2)));
        let w = [w_r, w_neg_r, w_r2];
        let _d = Self::append_zk_witnesses(&w, transcript);

        Ok((HyperKZGProof::zk(com, w, y, y_out), y_out, output_blind))
    }

    /// HyperKZG verification.
    #[tracing::instrument(skip_all, name = "HyperKZG::verify")]
    pub fn verify<T: Transcript<Challenge = P::ScalarField>>(
        vk: &HyperKZGVerifierSetup<P>,
        commitment: &HyperKZGCommitment<P>,
        point: &[P::ScalarField],
        claimed_eval: &P::ScalarField,
        proof: &HyperKZGProof<P>,
        transcript: &mut T,
    ) -> Result<(), HyperKZGError> {
        let ell = point.len();
        if ell == 0 {
            return Err(HyperKZGError::EmptyPoint);
        }

        if proof.com.len() + 1 != ell {
            return Err(HyperKZGError::WrongCommitmentCount {
                expected: ell - 1,
                got: proof.com.len(),
            });
        }

        // Validate inner evaluation widths before mutating the transcript.
        let v = proof
            .clear_evaluations()
            .ok_or_else(|| HyperKZGError::WrongProofPayload {
                expected: HyperKZGProofKind::Clear,
                got: proof.payload_kind(),
            })?;
        if v[0].len() != ell || v[1].len() != ell || v[2].len() != ell {
            return Err(HyperKZGError::WrongEvaluationWidth { expected: ell });
        }

        // Absorb intermediate commitments
        for c in &proof.com {
            transcript.append(c);
        }
        let r: P::ScalarField = transcript.challenge();

        if r.is_zero() {
            return Err(HyperKZGError::DegenerateChallenge);
        }

        // Prepend the original commitment as C_0
        let mut com = Vec::with_capacity(ell);
        com.push(commitment.point);
        com.extend_from_slice(&proof.com);

        let u = [r, -r, r * r];

        let ypos = &v[0]; // evaluations at r
        let yneg = &v[1]; // evaluations at -r
        let mut y_sq = v[2].clone(); // evaluations at r^2
        y_sq.push(*claimed_eval);

        // Consistency check: the folding relation must hold across evaluations
        //
        // For each level i, the polynomial P_i is defined by:
        //   P_i(x) = (1 - x_{ell-i}) * P_{i-1,even}(x) + x_{ell-i} * P_{i-1,odd}(x)
        //
        // This implies:
        //   2*r * P_{i+1}(r^2) = r * (1 - x_{ell-i-1}) * (P_i(r) + P_i(-r))
        //                       + x_{ell-i-1} * (P_i(r) - P_i(-r))
        let two = P::ScalarField::from_u64(2);
        for i in 0..ell {
            let lhs = two * r * y_sq[i + 1];
            let rhs = r * (P::ScalarField::one() - point[ell - i - 1]) * (ypos[i] + yneg[i])
                + point[ell - i - 1] * (ypos[i] - yneg[i]);
            if lhs != rhs {
                return Err(HyperKZGError::FoldingConsistencyFailed { level: i });
            }
        }

        // Batch KZG pairing check
        if !kzg_verify_batch::<P, T>(vk, &com, &proof.w, &u, v, transcript) {
            return Err(HyperKZGError::PairingCheckFailed);
        }

        Ok(())
    }

    #[cfg(feature = "zk")]
    fn verify_zk_inner<T>(
        vk: &HyperKZGVerifierSetup<P>,
        commitment: &HyperKZGCommitment<P>,
        point: &[P::ScalarField],
        proof: &HyperKZGProof<P>,
        transcript: &mut T,
    ) -> Result<P::G1, HyperKZGError>
    where
        T: Transcript<Challenge = P::ScalarField>,
    {
        let ell = point.len();
        if ell == 0 {
            return Err(HyperKZGError::EmptyPoint);
        }
        if proof.com.len() + 1 != ell {
            return Err(HyperKZGError::WrongCommitmentCount {
                expected: ell - 1,
                got: proof.com.len(),
            });
        }

        let _ = vk.hiding_g1.ok_or(HyperKZGError::MissingZkSrs)?;
        let (y, y_out) = match &proof.payload {
            HyperKZGProofPayload::Zk { y, y_out } => (y, y_out),
            HyperKZGProofPayload::Clear { .. } => {
                return Err(HyperKZGError::WrongProofPayload {
                    expected: HyperKZGProofKind::Zk,
                    got: proof.payload_kind(),
                });
            }
        };
        if y[0].len() != ell || y[1].len() != ell || y[2].len() != ell {
            return Err(HyperKZGError::WrongEvaluationWidth { expected: ell });
        }

        let r = Self::append_zk_opening_prefix(point, y_out, &proof.com, transcript);
        if r.is_zero() {
            return Err(HyperKZGError::DegenerateChallenge);
        }
        let u = [r, -r, r * r];

        let (alpha, q) = Self::append_zk_eval_commitments(y, transcript);
        let alpha_powers = kzg::challenge_powers(alpha, ell);
        let q_powers = kzg::challenge_powers(q, ell);

        let two_r = P::ScalarField::from_u64(2) * r;
        let mut gemini_bases = Vec::with_capacity(3 * ell);
        let mut gemini_scalars = Vec::with_capacity(3 * ell);
        for i in 0..ell {
            let alpha_i = alpha_powers[i];
            let xi = point[ell - i - 1];
            let r_one_minus_xi = r * (P::ScalarField::one() - xi);

            if i + 1 < ell {
                gemini_bases.push(y[2][i + 1]);
            } else {
                gemini_bases.push(*y_out);
            }
            gemini_scalars.push(alpha_i * two_r);
            gemini_bases.push(y[0][i]);
            gemini_scalars.push(-(alpha_i * (r_one_minus_xi + xi)));
            gemini_bases.push(y[1][i]);
            gemini_scalars.push(-(alpha_i * (r_one_minus_xi - xi)));
        }
        if !P::G1::msm(&gemini_bases, &gemini_scalars).is_identity() {
            return Err(HyperKZGError::BatchedFoldingConsistencyFailed);
        }

        let d = Self::append_zk_witnesses(&proof.w, transcript);
        let d_powers = [P::ScalarField::one(), d, d * d];

        let mut com = Vec::with_capacity(ell);
        com.push(commitment.point);
        com.extend_from_slice(&proof.com);
        let commitment_multiplier = d_powers
            .iter()
            .copied()
            .fold(P::ScalarField::zero(), |acc, power| acc + power);
        let mut kzg_bases = Vec::with_capacity(4 * ell + 3);
        let mut kzg_scalars = Vec::with_capacity(4 * ell + 3);

        for (&commitment_i, &q_i) in com.iter().zip(&q_powers) {
            kzg_bases.push(commitment_i);
            kzg_scalars.push(q_i * commitment_multiplier);
        }
        for t in 0..3 {
            for (&y_t_i, &q_i) in y[t].iter().zip(&q_powers) {
                kzg_bases.push(y_t_i);
                kzg_scalars.push(-(d_powers[t] * q_i));
            }
        }
        for t in 0..3 {
            kzg_bases.push(proof.w[t]);
            kzg_scalars.push(d_powers[t] * u[t]);
        }

        let batched_lhs = P::G1::msm(&kzg_bases, &kzg_scalars);
        let batched_rhs = P::G1::msm(&proof.w, &d_powers);

        let result = P::multi_pairing(&[batched_lhs, -batched_rhs], &[vk.g2, vk.beta_g2]);
        if !result.is_identity() {
            return Err(HyperKZGError::PairingCheckFailed);
        }

        Ok(*y_out)
    }
}

impl<P: PairingGroup> Commitment for HyperKZGScheme<P> {
    type Output = HyperKZGCommitment<P>;
}

impl<P: PairingGroup> CommitmentScheme for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    type Field = P::ScalarField;
    type Proof = HyperKZGProof<P>;
    type ProverSetup = HyperKZGProverSetup<P>;
    type VerifierSetup = HyperKZGVerifierSetup<P>;
    type Polynomial = Polynomial<P::ScalarField>;
    type OpeningHint = HyperKZGOpeningHint<P>;
    type SetupParams = (usize, P::G1, P::G2);

    fn setup(
        (max_num_vars, g1, g2): Self::SetupParams,
    ) -> (Self::ProverSetup, Self::VerifierSetup) {
        let mut rng = rand_core::OsRng;
        let max_degree = 1usize << max_num_vars;
        let prover = HyperKZGScheme::setup(&mut rng, max_degree, g1, g2);
        let verifier = Self::verifier_setup(&prover);
        (prover, verifier)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HyperKZGVerifierSetup::from(prover_setup)
    }

    fn commit<S: jolt_poly::MultilinearPoly<Self::Field> + ?Sized>(
        poly: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<Transparent, S>(poly, setup)
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        _eval: Self::Field,
        setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        Self::open(setup, poly.evaluations(), point, transcript)
            .expect("HyperKZG open should not fail with valid inputs")
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        Self::verify(setup, commitment, point, &eval, proof, transcript)
            .map_err(|_| OpeningsError::VerificationFailed)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&LabelWithCount(
            b"hyperkzg_opening_point",
            point.len() as u64,
        ));
        for p in point {
            p.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"hyperkzg_opening_eval"));
        eval.append_to_transcript(transcript);
    }
}

impl<P: PairingGroup> AdditivelyHomomorphic for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let bases: Vec<P::G1> = commitments.iter().map(|c| c.point).collect();
        HyperKZGCommitment {
            point: P::G1::msm(&bases, scalars),
        }
    }

    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        assert_eq!(hints.len(), scalars.len());
        if hints.is_empty() {
            return HyperKZGOpeningHint::clear();
        }

        let zk_hint_count = hints.iter().filter(|hint| hint.is_zk()).count();
        if zk_hint_count == 0 {
            return HyperKZGOpeningHint::clear();
        }
        assert_eq!(
            zk_hint_count,
            hints.len(),
            "cannot combine mixed transparent and ZK HyperKZG opening hints"
        );

        let blind = hints
            .iter()
            .zip(scalars)
            .map(|(hint, &scalar)| scalar * hint.blind.expect("ZK hint must contain a blind"))
            .fold(P::ScalarField::zero(), |acc, value| acc + value);

        HyperKZGOpeningHint { blind: Some(blind) }
    }
}

#[cfg(feature = "zk")]
impl<P: PairingGroup> ZkOpeningScheme for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    type HidingCommitment = P::G1;
    type Blind = P::ScalarField;

    fn commit_zk<S: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<Zk, S>(poly, setup)
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::open_zk")]
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let blind = hint
            .into_zk_blind()
            .expect("ZK HyperKZG opening requires a ZK opening hint");
        Self::open_zk_inner(setup, poly.evaluations(), point, eval, blind, transcript)
            .expect("HyperKZG ZK open should not fail with valid inputs")
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        Self::verify_zk_inner(setup, commitment, point, proof, transcript)
            .map_err(|_| OpeningsError::VerificationFailed)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&LabelWithCount(b"hyperkzg_zk_point", point.len() as u64));
        for p in point {
            p.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"hyperkzg_zk_eval_com"));
        hiding_commitment.append_to_transcript(transcript);
    }
}
