//! HyperKZG commitment scheme implementing `jolt-openings` traits.
//!
//! [`HyperKZGScheme`] is generic over `P: PairingGroup` — instantiate with
//! `Bn254` for the concrete BN254 curve.

use std::marker::PhantomData;

use jolt_crypto::{Commitment, DeriveSetup, JoltGroup, PairingGroup, PedersenSetup};
use jolt_field::{CanonicalBytes, Field, JoltField};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::One;
use rand_core::{OsRng, RngCore};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::HyperKZGError;
use crate::kzg::{self, kzg_open_batch, kzg_verify_batch, FoldPoints};
use crate::types::{
    HyperKZGCommitment, HyperKZGProof, HyperKZGProverSetup, HyperKZGVerifierSetup,
    NoopVerifierObserver, VerifierObserver,
};

/// HyperKZG multilinear polynomial commitment scheme.
///
/// Generic over `P: PairingGroup`. Implements [`CommitmentScheme`] and
/// [`AdditivelyHomomorphic`] from `jolt-openings`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperKZGScheme<P: PairingGroup> {
    _phantom: PhantomData<P>,
}

impl<P: PairingGroup> HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    /// Generates an SRS from a random generator and secret scalar.
    ///
    /// `max_degree` is the maximum polynomial length (number of evaluations).
    /// The SRS contains `max(max_degree, 7)` G1 powers, G2 powers at exponents
    /// 0, 1, 4, 5, and the G2 shifts for degree-five/six univariate checks.
    pub fn setup<R: RngCore>(
        rng: &mut R,
        max_degree: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let beta = P::ScalarField::random(rng);
        Self::setup_from_secret(beta, max_degree, g1, g2)
    }

    /// Generates SRS from a known secret.
    ///
    /// WARNING: this is only appropriate for deterministic tests or trusted
    /// setup tooling that destroys `beta`; anyone who knows `beta` can break
    /// KZG binding.
    #[expect(
        clippy::indexing_slicing,
        reason = "num_powers is at least seven and scalars contains each published exponent"
    )]
    pub fn setup_from_secret(
        beta: P::ScalarField,
        max_degree: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let num_powers = max_degree.max(7);
        let mut scalars = Vec::with_capacity(num_powers);
        let mut cur = P::ScalarField::one();
        for _ in 0..num_powers {
            scalars.push(cur);
            cur *= beta;
        }
        let g1_powers = P::g1_to_affine(&fixed_base_powers::<P>(g1, &scalars));

        let g2_powers = [0, 1, 4, 5].map(|exponent| g2.scalar_mul(&scalars[exponent]));

        let degree_five_shift_g2 = g2.scalar_mul(&scalars[num_powers - 6]);
        let degree_six_shift_g2 = g2.scalar_mul(&scalars[num_powers - 7]);
        HyperKZGProverSetup {
            g1_powers,
            g2_powers,
            degree_five_shift_g2,
            degree_six_shift_g2,
        }
    }

    /// Number of committed two-variable fold levels for a multilinear opening.
    /// The final one or two variables are checked without another commitment.
    pub const fn fold_level_count(num_vars: usize) -> usize {
        num_vars.saturating_sub(1) / 2
    }

    /// Phase 1 of the HyperKZG protocol: fold two variables per level.
    ///
    /// Given polynomial $P$ with $2^\ell$ evaluations and opening point
    /// $x = (x_1, \ldots, x_\ell)$, each level reduces the coefficient
    /// vector to one quarter of its prior length. The terminal check handles
    /// the first variable for odd $\ell$ and the first two for even $\ell$.
    ///
    /// For a chunk $(a_{00}, a_{01}, a_{10}, a_{11})$ and variables
    /// $(x, y)$, the next coefficient is
    /// $(1-x)((1-y)a_{00}+ya_{01})+x((1-y)a_{10}+ya_{11})$.
    #[expect(
        clippy::expect_used,
        reason = "polys is seeded with one element before the fold loop"
    )]
    fn fold_polynomials(
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
    ) -> Vec<Vec<P::ScalarField>> {
        let levels = Self::fold_level_count(point.len());
        let mut polys = Vec::with_capacity(levels + 1);
        polys.push(evals.to_vec());

        for variables in point.rchunks_exact(2).take(levels) {
            let prev = polys.last().expect("polys starts with one element");
            let pi: Vec<P::ScalarField> = prev
                .par_chunks_exact(4)
                .map(|values| {
                    let [a00, a01, a10, a11] = values else {
                        unreachable!("par_chunks_exact(4) fixes the chunk width")
                    };
                    let [x, y] = variables else {
                        unreachable!("rchunks_exact(2) fixes the variable width")
                    };
                    fold_two_variables(
                        [*a00, *a01, *a10, *a11],
                        [*x, *y],
                        &mut NoopVerifierObserver,
                    )
                })
                .collect();
            polys.push(pi);
        }

        polys
    }

    /// Full HyperKZG opening proof.
    #[tracing::instrument(skip_all, name = "HyperKZG::open")]
    #[expect(
        clippy::expect_used,
        reason = "intermediate fold polynomials shrink, so an SRS sized for the input covers every kzg_commit"
    )]
    pub fn open<T: Transcript<Challenge = P::ScalarField>>(
        setup: &HyperKZGProverSetup<P>,
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
        claimed_eval: &P::ScalarField,
        transcript: &mut T,
    ) -> Result<HyperKZGProof<P>, HyperKZGError> {
        let num_vars = point.len();
        if num_vars == 0 {
            return Err(HyperKZGError::EmptyPoint);
        }
        let n = evals.len();
        assert_eq!(n, 1 << num_vars, "evaluation count must be 2^ell");

        // Phase 1: fold
        let polys = Self::fold_polynomials(evals, point);
        let levels = Self::fold_level_count(num_vars);
        assert_eq!(polys.len(), levels + 1);
        let final_fold = polys.last().ok_or(HyperKZGError::InvalidBatchShape)?;
        let derived_claim = match final_fold.as_slice() {
            [even, odd] => {
                let x = point.first().ok_or(HyperKZGError::EmptyPoint)?;
                *even + *x * (*odd - *even)
            }
            [a00, a01, a10, a11] => {
                let [x, y, ..] = point else {
                    return Err(HyperKZGError::InvalidBatchShape);
                };
                fold_two_variables(
                    [*a00, *a01, *a10, *a11],
                    [*x, *y],
                    &mut NoopVerifierObserver,
                )
            }
            _ => return Err(HyperKZGError::InvalidBatchShape),
        };
        if derived_claim != *claimed_eval {
            return Err(HyperKZGError::FoldingConsistencyFailed { level: levels });
        }

        // Commit to intermediate polynomials (skip polys[0] — already committed)
        let com: Vec<P::G1> = polys
            .iter()
            .skip(1)
            .map(|p| kzg::kzg_commit::<P>(p, setup).expect("SRS large enough for intermediate"))
            .collect();

        // Phase 2: derive challenge r
        for c in &com {
            transcript.append(c);
        }
        let r: P::ScalarField = transcript.challenge();
        let points = FoldPoints::new(r, &mut NoopVerifierObserver)?;

        let (w, v, p0_at_r_fourth) = kzg_open_batch::<P, T>(&polys, &points, setup, transcript)?;

        Ok(HyperKZGProof {
            com,
            w,
            v,
            p0_at_r_fourth,
        })
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
        Self::verify_observed(
            vk,
            commitment,
            point,
            claimed_eval,
            proof,
            transcript,
            &mut NoopVerifierObserver,
        )
    }

    pub fn verify_observed<T, O>(
        vk: &HyperKZGVerifierSetup<P>,
        commitment: &HyperKZGCommitment<P>,
        point: &[P::ScalarField],
        claimed_eval: &P::ScalarField,
        proof: &HyperKZGProof<P>,
        transcript: &mut T,
        observer: &mut O,
    ) -> Result<(), HyperKZGError>
    where
        T: Transcript<Challenge = P::ScalarField>,
        O: VerifierObserver,
    {
        let num_vars = point.len();
        if num_vars == 0 {
            return Err(HyperKZGError::EmptyPoint);
        }
        let levels = Self::fold_level_count(num_vars);
        let polynomial_count = levels + 1;

        if proof.com.len() != levels {
            return Err(HyperKZGError::WrongCommitmentCount {
                expected: levels,
                got: proof.com.len(),
            });
        }

        // Validate inner evaluation widths before mutating the transcript.
        let v = &proof.v;
        if v.iter().any(|row| row.len() != polynomial_count) {
            return Err(HyperKZGError::WrongEvaluationWidth {
                expected: polynomial_count,
            });
        }

        // Absorb intermediate commitments
        for c in &proof.com {
            transcript.append(c);
        }
        let r: P::ScalarField = transcript.challenge();

        // Prepend the original commitment as C_0
        let mut com = Vec::with_capacity(polynomial_count);
        com.push(commitment.point);
        com.extend_from_slice(&proof.com);

        let points = FoldPoints::new(r, observer)?;

        // The 4x4 DFT at r, ir, -r, -ir is invertible because r != 0 and
        // i^2 = -1. It determines the four residue evaluations at r^4; their
        // two-variable fold becomes the claimed fifth-point evaluation of the
        // next commitment. The five-point KZG batch binds those claims. The
        // final HyperKZG path has no separate per-level degree-bound proof.
        let mut y_fourth = Vec::with_capacity(polynomial_count);
        y_fourth.push(proof.p0_at_r_fourth);
        let [at_r, at_ir, at_neg_r, at_neg_ir] = v;
        for (variables, (((&y_r, &y_ir), &y_neg_r), &y_neg_ir)) in point
            .rchunks_exact(2)
            .take(levels)
            .zip(at_r.iter().zip(at_ir).zip(at_neg_r).zip(at_neg_ir))
        {
            let [x, y] = variables else {
                return Err(HyperKZGError::InvalidBatchShape);
            };
            let residues = points.residues([y_r, y_ir, y_neg_r, y_neg_ir], observer);
            y_fourth.push(fold_two_variables(residues, [*x, *y], observer));
        }

        let terminal_values: [P::ScalarField; 4] = [
            *at_r.last().ok_or(HyperKZGError::InvalidBatchShape)?,
            *at_ir.last().ok_or(HyperKZGError::InvalidBatchShape)?,
            *at_neg_r.last().ok_or(HyperKZGError::InvalidBatchShape)?,
            *at_neg_ir.last().ok_or(HyperKZGError::InvalidBatchShape)?,
        ];
        let terminal = if num_vars.is_multiple_of(2) {
            let [x, y, ..] = point else {
                return Err(HyperKZGError::InvalidBatchShape);
            };
            let residues = points.residues(terminal_values, observer);
            fold_two_variables(residues, [*x, *y], observer)
        } else {
            let x = point.first().ok_or(HyperKZGError::EmptyPoint)?;
            let [at_r, _, at_neg_r, _] = terminal_values;
            let [even, odd] = points.binary_residues(at_r, at_neg_r, observer);
            even + observer.fr_mul(*x, odd - even)
        };
        if terminal != *claimed_eval {
            return Err(HyperKZGError::FoldingConsistencyFailed { level: levels });
        }

        // Batch KZG pairing check
        let full_evaluations = [
            v[0].clone(),
            v[1].clone(),
            v[2].clone(),
            v[3].clone(),
            y_fourth,
        ];
        if !kzg_verify_batch::<P, T, O>(
            vk,
            &com,
            proof.w,
            &points,
            &full_evaluations,
            transcript,
            observer,
        ) {
            return Err(HyperKZGError::PairingCheckFailed);
        }

        Ok(())
    }
}

fn fold_two_variables<F, O>(values: [F; 4], variables: [F; 2], observer: &mut O) -> F
where
    F: JoltField,
    O: VerifierObserver,
{
    let [a00, a01, a10, a11] = values;
    let [x, y] = variables;
    let low = a00 + observer.fr_mul(y, a01 - a00);
    let high = a10 + observer.fr_mul(y, a11 - a10);
    low + observer.fr_mul(x, high - low)
}

/// `[s · g1]` for every scalar through a fixed-base table: sixteen 16-bit
/// windows of `g1` multiples, one table add per window instead of a full
/// double-and-add per power.
fn fixed_base_powers<P: PairingGroup>(g1: P::G1, scalars: &[P::ScalarField]) -> Vec<P::G1> {
    const WINDOW_BITS: usize = 16;
    let windows = <P::ScalarField as CanonicalBytes>::NUM_BYTES.div_ceil(2);
    let mut window_bases = Vec::with_capacity(windows);
    let mut base = g1;
    for _ in 0..windows {
        window_bases.push(base);
        for _ in 0..WINDOW_BITS {
            base = base.double();
        }
    }
    let tables: Vec<Vec<P::G1>> = window_bases
        .into_par_iter()
        .map(|base| {
            let mut table = Vec::with_capacity(1 << WINDOW_BITS);
            let mut multiple = P::G1::identity();
            for _ in 0..1usize << WINDOW_BITS {
                table.push(multiple);
                multiple += base;
            }
            table
        })
        .collect();
    scalars
        .par_iter()
        .map(|scalar| {
            tables.iter().zip(scalar.to_bytes_le_vec().chunks(2)).fold(
                P::G1::identity(),
                |acc, (table, digit_bytes)| {
                    let digit = digit_bytes
                        .iter()
                        .rev()
                        .fold(0usize, |digit, &byte| digit << 8 | usize::from(byte));
                    table.get(digit).map_or(acc, |multiple| acc + multiple)
                },
            )
        })
        .collect()
}

/// # Security note
///
/// Uses KZG SRS powers as Pedersen generators — Pedersen binding shares the
/// KZG trapdoor `beta`. Both are sound once `beta` is destroyed, but the two
/// schemes do not have independent security assumptions.
impl<P: PairingGroup> DeriveSetup<HyperKZGProverSetup<P>> for PedersenSetup<P::G1> {
    /// # Panics
    ///
    /// Panics when the SRS is smaller than `capacity + 1`. `derive` runs at
    /// setup time on operator-provided parameters (the infallible
    /// `DeriveSetup` trait offers no error channel), never on the
    /// proof-verification path.
    #[expect(clippy::expect_used, reason = "length checked by the assert above")]
    fn derive(source: &HyperKZGProverSetup<P>, capacity: usize) -> Self {
        assert!(
            source.g1_powers.len() > capacity,
            "SRS has {} G1 powers, need at least {} (capacity + 1 for blinding)",
            source.g1_powers.len(),
            capacity + 1,
        );
        let (message_generators, rest) = source.g1_powers.split_at(capacity);
        let message_generators = message_generators.iter().map(P::g1_from_affine).collect();
        let blinding_generator =
            P::g1_from_affine(rest.first().expect("length checked by the assert above"));
        PedersenSetup::new(message_generators, blinding_generator)
    }
}

impl<P: PairingGroup> Commitment for HyperKZGScheme<P> {
    type Output = HyperKZGCommitment<P>;
}

impl<P: PairingGroup> CommitmentScheme for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript + Serialize + DeserializeOwned,
    P::G1: AppendToTranscript + Serialize + DeserializeOwned,
    P::G2: Serialize + DeserializeOwned,
{
    type Field = P::ScalarField;
    type Proof = HyperKZGProof<P>;
    type ProverSetup = HyperKZGProverSetup<P>;
    type VerifierSetup = HyperKZGVerifierSetup<P>;
    type OpeningHint = ();
    type SetupParams = (usize, P::G1, P::G2);

    fn setup(
        (max_num_vars, g1, g2): Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
        let mut rng = OsRng;
        let max_degree = 1usize << max_num_vars;
        let prover = HyperKZGScheme::setup(&mut rng, max_degree, g1, g2);
        let verifier = Self::verifier_setup(&prover);
        Ok((prover, verifier))
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HyperKZGVerifierSetup::from(prover_setup)
    }

    fn commit<S: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &S,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        // HyperKZG always works on dense evaluations; `to_dense` borrows them
        // when the source is already dense.
        let evaluations = poly.to_dense();
        let point = kzg::kzg_commit::<P>(&evaluations, setup)
            .map_err(|e| OpeningsError::CommitFailed(format!("HyperKZG commit failed: {e:?}")))?;
        Ok((HyperKZGCommitment { point }, ()))
    }

    fn open<S: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &S,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        let evaluations = poly.to_dense();
        Self::open(setup, &evaluations, point, &eval, transcript)
            .map_err(|e| OpeningsError::ProveFailed(format!("HyperKZG open failed: {e:?}")))
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
}

impl<P: PairingGroup> AdditivelyHomomorphic for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript + Serialize + DeserializeOwned,
    P::G1: AppendToTranscript + Serialize + DeserializeOwned,
    P::G2: Serialize + DeserializeOwned,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let bases: Vec<P::G1> = commitments.iter().map(|c| c.point).collect();
        HyperKZGCommitment {
            point: P::g1_msm(&bases, scalars),
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        clippy::expect_used,
        reason = "tests unwrap successful PCS operations"
    )]

    use super::*;
    use jolt_crypto::Bn254;
    use jolt_field::{Fr, Ring};
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type TestScheme = HyperKZGScheme<Bn254>;

    fn test_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
        let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        let prover = TestScheme::setup(&mut rng, max_degree, g1, g2);
        let verifier = TestScheme::verifier_setup(&prover);
        (prover, verifier)
    }

    #[test]
    fn two_variable_folds_preserve_multilinear_evaluation() {
        assert_eq!(TestScheme::fold_level_count(23), 11);
        assert_eq!(TestScheme::fold_level_count(22), 10);
        let mut rng = ChaCha20Rng::seed_from_u64(87);
        for num_vars in 1..=8 {
            let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
            let point: Vec<_> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
            let folds = TestScheme::fold_polynomials(poly.evaluations(), &point);
            assert_eq!(folds.len(), TestScheme::fold_level_count(num_vars) + 1);
            for (level, coefficients) in folds.into_iter().enumerate() {
                let remaining = num_vars - 2 * level;
                assert_eq!(coefficients.len(), 1 << remaining);
                let folded = Polynomial::from(coefficients);
                assert_eq!(
                    folded.evaluate(point.split_at(remaining).0),
                    poly.evaluate(&point)
                );
            }
        }
    }

    #[test]
    fn commit_open_verify_roundtrip() {
        for ell in [2, 3, 4, 5, 6, 8] {
            let n = 1 << ell;
            let mut rng = ChaCha20Rng::seed_from_u64(ell as u64);
            let (pk, vk) = test_setup(n);

            let poly = Polynomial::<Fr>::random(ell, &mut rng);
            let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
            let eval = poly.evaluate(&point);

            let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk).unwrap();

            let mut prover_transcript = Blake2bTranscript::new(b"test");
            let proof = <TestScheme as CommitmentScheme>::open(
                &poly,
                &point,
                eval,
                &pk,
                None,
                &mut prover_transcript,
            )
            .unwrap();

            let mut verifier_transcript = Blake2bTranscript::new(b"test");
            let result = <TestScheme as CommitmentScheme>::verify(
                &commitment,
                &point,
                eval,
                &proof,
                &vk,
                &mut verifier_transcript,
            );
            assert!(result.is_ok(), "ell={ell}: verification failed: {result:?}");
        }
    }

    #[test]
    fn wrong_eval_rejects() {
        let ell = 4;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let wrong_eval = eval + Fr::from_u64(1);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk).unwrap();

        let mut prover_transcript = Blake2bTranscript::new(b"test-bad");
        let proof = <TestScheme as CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &pk,
            None,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::new(b"test-bad");
        let result = <TestScheme as CommitmentScheme>::verify(
            &commitment,
            &point,
            wrong_eval,
            &proof,
            &vk,
            &mut verifier_transcript,
        );
        assert!(result.is_err(), "wrong evaluation should be rejected");
    }

    #[test]
    fn missing_intermediate_commitment_rejects() {
        let ell = 4;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(43);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk).unwrap();

        let mut prover_transcript = Blake2bTranscript::new(b"test-missing-com");
        let mut proof = <TestScheme as CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &pk,
            None,
            &mut prover_transcript,
        )
        .unwrap();
        let _ = proof.com.pop();

        let mut verifier_transcript = Blake2bTranscript::new(b"test-missing-com");
        let result = TestScheme::verify(
            &vk,
            &commitment,
            &point,
            &eval,
            &proof,
            &mut verifier_transcript,
        );
        assert!(matches!(
            result,
            Err(HyperKZGError::WrongCommitmentCount { .. })
        ));
    }

    #[test]
    fn trait_setup_uses_fresh_randomness() {
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();

        let (_pk1, vk1) = <TestScheme as CommitmentScheme>::setup((4, g1, g2)).unwrap();
        let (_pk2, vk2) = <TestScheme as CommitmentScheme>::setup((4, g1, g2)).unwrap();

        assert_ne!(vk1.beta_g2, vk2.beta_g2);
    }

    #[test]
    fn tampered_proof_rejects() {
        let ell = 4;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk).unwrap();

        let mut prover_transcript = Blake2bTranscript::new(b"test-tamper");
        let mut proof = <TestScheme as CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &pk,
            None,
            &mut prover_transcript,
        )
        .unwrap();

        // Tamper with proof: swap v[0] and v[1]
        let v1 = proof.v[1].clone();
        proof.v[0].clone_from(&v1);

        let mut verifier_transcript = Blake2bTranscript::new(b"test-tamper");
        let result = <TestScheme as CommitmentScheme>::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &vk,
            &mut verifier_transcript,
        );
        assert!(result.is_err(), "tampered proof should be rejected");
    }

    #[test]
    fn combine_is_homomorphic() {
        let ell = 3;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let (pk, _vk) = test_setup(n);

        let poly_a = Polynomial::<Fr>::random(ell, &mut rng);
        let poly_b = Polynomial::<Fr>::random(ell, &mut rng);

        let (ca, ()) = TestScheme::commit(poly_a.evaluations(), &pk).unwrap();
        let (cb, ()) = TestScheme::commit(poly_b.evaluations(), &pk).unwrap();

        let sum_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let (c_sum_direct, ()) = TestScheme::commit(&sum_evals, &pk).unwrap();

        let c_sum_combined = TestScheme::combine(&[ca, cb], &[Fr::from_u64(1), Fr::from_u64(1)]);

        assert_eq!(
            c_sum_direct, c_sum_combined,
            "combine([1,1]) must match commitment to sum"
        );
    }

    #[test]
    fn combine_with_scalars() {
        let ell = 3;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let (pk, _vk) = test_setup(n);

        let poly_a = Polynomial::<Fr>::random(ell, &mut rng);
        let poly_b = Polynomial::<Fr>::random(ell, &mut rng);
        let s_a = Fr::random(&mut rng);
        let s_b = Fr::random(&mut rng);

        let (ca, ()) = TestScheme::commit(poly_a.evaluations(), &pk).unwrap();
        let (cb, ()) = TestScheme::commit(poly_b.evaluations(), &pk).unwrap();

        let combined_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| s_a * *a + s_b * *b)
            .collect();
        let (c_direct, ()) = TestScheme::commit(&combined_evals, &pk).unwrap();

        let c_combined = TestScheme::combine(&[ca, cb], &[s_a, s_b]);

        assert_eq!(c_direct, c_combined);
    }

    #[test]
    fn open_verify_with_random_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xcafe);

        for _ in 0..5 {
            let ell = 4;
            let n = 1 << ell;
            let (pk, vk) = test_setup(n);

            let poly = Polynomial::<Fr>::random(ell, &mut rng);
            let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
            let eval = poly.evaluate(&point);

            let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk).unwrap();

            let mut pt = Blake2bTranscript::new(b"rand-test");
            let proof =
                <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt)
                    .unwrap();

            let mut vt = Blake2bTranscript::new(b"rand-test");
            <TestScheme as CommitmentScheme>::verify(
                &commitment,
                &point,
                eval,
                &proof,
                &vk,
                &mut vt,
            )
            .expect("random instance should verify");
        }
    }

    #[test]
    fn extract_vc_setup_produces_valid_pedersen() {
        use jolt_crypto::{Pedersen, VectorCommitment};

        let n = 1 << 4;
        let (pk, _vk) = test_setup(n);

        let capacity = 5;
        let vc_setup = PedersenSetup::<jolt_crypto::Bn254G1>::derive(&pk, capacity);

        assert_eq!(
            <Pedersen<jolt_crypto::Bn254G1> as VectorCommitment>::capacity(&vc_setup),
            capacity,
        );

        // Commit and verify a small vector.
        let values = vec![Fr::one(), Fr::from_u64(2), Fr::from_u64(3)];
        let blinding = Fr::from_u64(42);
        let commitment = <Pedersen<jolt_crypto::Bn254G1> as VectorCommitment>::commit(
            &vc_setup, &values, &blinding,
        );
        assert!(
            <Pedersen<jolt_crypto::Bn254G1> as VectorCommitment>::verify(
                &vc_setup,
                &commitment,
                &values,
                &blinding,
            )
        );
    }

    #[test]
    fn fixed_base_powers_match_scalar_mul() {
        let g1 = Bn254::g1_generator();
        let scalars = [
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(65_535),
            Fr::from_u64(65_536),
            Fr::from_u64(u64::MAX),
            -Fr::from_u64(1),
            Fr::from_u64(23).inverse().unwrap(),
        ];
        let powers = fixed_base_powers::<Bn254>(g1, &scalars);
        for (power, scalar) in powers.iter().zip(&scalars) {
            assert_eq!(*power, g1.scalar_mul(scalar));
        }
    }

    #[test]
    fn setup_opening_and_degree_shift_exponents() {
        let beta = Fr::from_u64(3);
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        let setup = TestScheme::setup_from_secret(beta, 8, g1, g2);
        let vk = TestScheme::verifier_setup(&setup);
        assert_eq!(setup.g1_powers().len(), 8);
        assert_eq!(vk.beta_cu_g1, g1.scalar_mul(&Fr::from_u64(27)));
        assert_eq!(vk.beta_fourth_g1, g1.scalar_mul(&Fr::from_u64(81)));
        assert_eq!(vk.g2, g2);
        assert_eq!(vk.beta_g2, g2.scalar_mul(&beta));
        assert_eq!(vk.beta_fourth_g2, g2.scalar_mul(&Fr::from_u64(81)));
        assert_eq!(vk.beta_fifth_g2, g2.scalar_mul(&Fr::from_u64(243)));
        assert_eq!(vk.degree_five_shift_g2, g2.scalar_mul(&Fr::from_u64(9)));
        assert_eq!(vk.degree_six_shift_g2, g2.scalar_mul(&beta));
    }

    #[test]
    fn trivial_polynomial() {
        // 1-variable polynomial: [a, b]
        let ell = 1;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk).unwrap();

        let mut pt = Blake2bTranscript::new(b"trivial");
        let proof = <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt)
            .unwrap();

        let mut vt = Blake2bTranscript::new(b"trivial");
        <TestScheme as CommitmentScheme>::verify(&commitment, &point, eval, &proof, &vk, &mut vt)
            .expect("trivial polynomial should verify");
    }
}
