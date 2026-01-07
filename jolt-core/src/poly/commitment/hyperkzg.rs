//! This is a port of https://github.com/microsoft/Nova/blob/main/src/provider/hyperkzg.rs
//! and such code is Copyright (c) Microsoft Corporation.
//!
//! This module implements `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
use super::{
    commitment_scheme::CommitmentScheme,
    kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG},
};
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::poly::rlc_polynomial::RLCPolynomial;
use crate::zkvm::witness::CommittedPolynomial;
use crate::{
    msm::VariableBaseMSM,
    poly::{commitment::kzg::SRS, dense_mlpoly::DensePolynomial, unipoly::UniPoly},
    transcripts::{AppendToTranscript, Transcript},
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::borrow::Borrow;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::{marker::PhantomData, sync::Arc};

pub struct HyperKZGSRS<P: Pairing>(Arc<SRS<P>>);

impl<P: Pairing> HyperKZGSRS<P> {
    pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, max_degree: usize) -> Self
    where
        P::ScalarField: JoltField,
    {
        Self(Arc::new(SRS::setup(rng, max_degree, 2)))
    }

    pub fn trim(self, max_degree: usize) -> (HyperKZGProverKey<P>, HyperKZGVerifierKey<P>) {
        let (kzg_pk, kzg_vk) = SRS::trim(self.0, max_degree);
        (HyperKZGProverKey { kzg_pk }, HyperKZGVerifierKey { kzg_vk })
    }

    /// Load SRS from a file using compressed serialization.
    pub fn load_from_file<Pth: AsRef<Path>>(path: Pth) -> Result<Self, ark_serialize::SerializationError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let srs = SRS::<P>::deserialize_compressed(reader)?;
        Ok(Self(Arc::new(srs)))
    }

    /// Save SRS to a file using compressed serialization.
    pub fn save_to_file<Pth: AsRef<Path>>(&self, path: Pth) -> Result<(), ark_serialize::SerializationError> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        self.0.serialize_compressed(writer)?;
        Ok(())
    }

    /// Load SRS from a reader using compressed serialization.
    pub fn load_from_reader<R: IoRead>(reader: R) -> Result<Self, ark_serialize::SerializationError> {
        let srs = SRS::<P>::deserialize_compressed(reader)?;
        Ok(Self(Arc::new(srs)))
    }

    /// Save SRS to a writer using compressed serialization.
    pub fn save_to_writer<W: IoWrite>(&self, writer: W) -> Result<(), ark_serialize::SerializationError> {
        self.0.serialize_compressed(writer)?;
        Ok(())
    }

    /// Get the maximum degree this SRS supports.
    pub fn max_degree(&self) -> usize {
        self.0.g1_powers.len().saturating_sub(1)
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGProverKey<P: Pairing> {
    pub kzg_pk: KZGProverKey<P>,
}

#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGVerifierKey<P: Pairing> {
    pub kzg_vk: KZGVerifierKey<P>,
}

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGCommitment<P: Pairing>(pub P::G1Affine);

impl<P: Pairing> Default for HyperKZGCommitment<P> {
    fn default() -> Self {
        Self(P::G1Affine::zero())
    }
}

impl<P: Pairing> AppendToTranscript for HyperKZGCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_point(&self.0.into_group());
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperKZGProof<P: Pairing> {
    pub com: Vec<P::G1Affine>,
    pub w: Vec<P::G1Affine>,
    pub v: Vec<Vec<P::ScalarField>>,
}

// On input f(x) and u compute the witness polynomial used to prove
// that f(u) = v. The main part of this is to compute the
// division (f(x) - f(u)) / (x - u), but we don't use a general
// division algorithm, we make use of the fact that the division
// never has a remainder, and that the denominator is always a linear
// polynomial. The cost is (d-1) mults + (d-1) adds in P::ScalarField, where
// d is the degree of f.
//
// We use the fact that if we compute the quotient of f(x)/(x-u),
// there will be a remainder, but it'll be v = f(u).  Put another way
// the quotient of f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
// same.  One advantage is that computing f(u) could be decoupled
// from kzg_open, it could be done later or separate from computing W.
fn kzg_batch_open_no_rem<P: Pairing>(
    f: &MultilinearPolynomial<P::ScalarField>,
    u: &[P::ScalarField],
    pk: &HyperKZGProverKey<P>,
) -> Vec<P::G1Affine>
where
    <P as Pairing>::ScalarField: JoltField,
{
    let f: &DensePolynomial<P::ScalarField> = f.try_into().unwrap();
    let h = u
        .par_iter()
        .map(|ui| {
            let h = compute_witness_polynomial::<P>(&f.evals(), *ui);
            MultilinearPolynomial::from(h)
        })
        .collect::<Vec<_>>();

    UnivariateKZG::commit_batch(&pk.kzg_pk, &h).unwrap()
}

fn compute_witness_polynomial<P: Pairing>(
    f: &[P::ScalarField],
    u: P::ScalarField,
) -> Vec<P::ScalarField>
where
    <P as Pairing>::ScalarField: JoltField,
{
    let d = f.len();

    // Compute h(x) = f(x)/(x - u)
    let mut h = vec![P::ScalarField::zero(); d];
    for i in (1..d).rev() {
        h[i - 1] = f[i] + h[i] * u;
    }

    h
}

fn kzg_open_batch<P: Pairing, ProofTranscript: Transcript>(
    f: &[MultilinearPolynomial<P::ScalarField>],
    u: &[P::ScalarField],
    pk: &HyperKZGProverKey<P>,
    transcript: &mut ProofTranscript,
) -> (Vec<P::G1Affine>, Vec<Vec<P::ScalarField>>)
where
    <P as Pairing>::ScalarField: JoltField,
{
    let k = f.len();
    let t = u.len();

    // The verifier needs f_i(u_j), so we compute them here
    // (V will compute B(u_j) itself)
    let mut v = vec![vec!(P::ScalarField::zero(); k); t];
    v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
        // for each point u
        v_i.par_iter_mut().zip_eq(f).for_each(|(v_ij, f)| {
            // for each poly f
            *v_ij = UniPoly::eval_as_univariate(f, &u[i]);
        });
    });

    // TODO(moodlezoup): Avoid cloned()
    let scalars = v.iter().flatten().collect::<Vec<&P::ScalarField>>();
    transcript.append_scalars::<P::ScalarField>(&scalars);
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(f.len());
    let f_arc: Vec<Arc<MultilinearPolynomial<P::ScalarField>>> =
        f.iter().map(|poly| Arc::new(poly.clone())).collect();

    // HyperKZG is not currently used in Jolt, but we handle both dense-only and mixed cases
    let has_one_hot = f_arc
        .iter()
        .any(|poly| matches!(poly.as_ref(), MultilinearPolynomial::OneHot(_)));

    let B = if has_one_hot {
        // Use RLCPolynomial::linear_combination for mixed dense + one-hot polynomials
        let dummy_poly_ids = vec![CommittedPolynomial::RdInc; f_arc.len()];
        let rlc_result = RLCPolynomial::linear_combination(dummy_poly_ids, f_arc, &q_powers, None);
        MultilinearPolynomial::RLC(rlc_result)
    } else {
        let poly_refs: Vec<&MultilinearPolynomial<P::ScalarField>> =
            f_arc.iter().map(|arc| arc.as_ref()).collect();
        let dense_result = DensePolynomial::linear_combination(&poly_refs, &q_powers);
        MultilinearPolynomial::from(dense_result.Z)
    };

    // Now open B at u0, ..., u_{t-1}
    let w = kzg_batch_open_no_rem(&B, u, pk);

    // The prover computes the challenge to keep the transcript in the same
    // state as that of the verifier
    transcript.append_points(&w.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
    let _d_0: P::ScalarField = transcript.challenge_scalar();

    (w, v)
}

// vk is hashed in transcript already, so we do not add it here
fn kzg_verify_batch<P: Pairing, ProofTranscript: Transcript>(
    vk: &HyperKZGVerifierKey<P>,
    C: &[P::G1Affine],
    W: &[P::G1Affine],
    u: &[P::ScalarField],
    v: &[Vec<P::ScalarField>],
    transcript: &mut ProofTranscript,
) -> bool
where
    <P as Pairing>::ScalarField: JoltField,
{
    let k = C.len();
    let t = u.len();

    let scalars = v.iter().flatten().collect::<Vec<&P::ScalarField>>();
    transcript.append_scalars::<P::ScalarField>(&scalars);
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(k);

    transcript.append_points(&W.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
    let d_0: P::ScalarField = transcript.challenge_scalar();
    let d_1 = d_0 * d_0;

    assert_eq!(t, 3);
    assert_eq!(W.len(), 3);
    // We write a special case for t=3, since this what is required for
    // hyperkzg. Following the paper directly, we must compute:
    // let L0 = C_B - vk.G * B_u[0] + W[0] * u[0];
    // let L1 = C_B - vk.G * B_u[1] + W[1] * u[1];
    // let L2 = C_B - vk.G * B_u[2] + W[2] * u[2];
    // let R0 = -W[0];
    // let R1 = -W[1];
    // let R2 = -W[2];
    // let L = L0 + L1*d_0 + L2*d_1;
    // let R = R0 + R1*d_0 + R2*d_1;
    //
    // We group terms to reduce the number of scalar mults (to seven):
    // In Rust, we could use MSMs for these, and speed up verification.
    //
    // Note, that while computing L, the intermediate computation of C_B together with computing
    // L0, L1, L2 can be replaced by single MSM of C with the powers of q multiplied by (1 + d_0 + d_1)
    // with additionally concatenated inputs for scalars/bases.

    let q_power_multiplier: P::ScalarField = P::ScalarField::one() + d_0 + d_1;

    let q_powers_multiplied: Vec<P::ScalarField> = q_powers
        .par_iter()
        .map(|q_power| *q_power * q_power_multiplier)
        .collect();

    // Compute the batched openings
    // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
    let B_u = v
        .into_par_iter()
        .map(|v_i| {
            v_i.into_par_iter()
                .zip(q_powers.par_iter())
                .map(|(a, b)| *a * *b)
                .sum()
        })
        .collect::<Vec<P::ScalarField>>();

    let L = <P::G1 as VariableBaseMSM>::msm_field_elements(
        &[&C[..k], &[W[0], W[1], W[2], vk.kzg_vk.g1]].concat(),
        &[
            &q_powers_multiplied[..k],
            &[
                u[0],
                (u[1] * d_0),
                (u[2] * d_1),
                -(B_u[0] + d_0 * B_u[1] + d_1 * B_u[2]),
            ],
        ]
        .concat(),
    )
    .unwrap();

    let R = W[0] + W[1] * d_0 + W[2] * d_1;

    // Check that e(L, vk.H) == e(R, vk.tau_H)
    P::multi_pairing([L, -R], [vk.kzg_vk.g2, vk.kzg_vk.beta_g2]).is_zero()
}

#[derive(Clone)]
pub struct HyperKZG<P: Pairing> {
    _phantom: PhantomData<P>,
}

impl<P: Pairing> HyperKZG<P>
where
    <P as Pairing>::ScalarField: JoltField,
{
    pub fn protocol_name() -> &'static [u8] {
        b"HyperKZG"
    }

    /// Setup prover key from a file containing the SRS.
    pub fn setup_prover_from_file<Pth: AsRef<Path>>(
        path: Pth,
        max_degree: usize,
    ) -> Result<HyperKZGProverKey<P>, ark_serialize::SerializationError> {
        let srs = HyperKZGSRS::load_from_file(path)?;
        Ok(srs.trim(max_degree).0)
    }

    /// Setup prover key from a reader containing the SRS.
    pub fn setup_prover_from_reader<R: IoRead>(
        reader: R,
        max_degree: usize,
    ) -> Result<HyperKZGProverKey<P>, ark_serialize::SerializationError> {
        let srs = HyperKZGSRS::load_from_reader(reader)?;
        Ok(srs.trim(max_degree).0)
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::commit_poly")]
    pub fn commit(
        pp: &HyperKZGProverKey<P>,
        poly: &MultilinearPolynomial<P::ScalarField>,
    ) -> Result<HyperKZGCommitment<P>, ProofVerifyError> {
        if pp.kzg_pk.g1_powers().len() < poly.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pp.kzg_pk.g1_powers().len(),
                poly.len(),
            ));
        }
        Ok(HyperKZGCommitment(UnivariateKZG::commit_as_univariate(
            &pp.kzg_pk, poly,
        )?))
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::open")]
    pub fn open<ProofTranscript: Transcript>(
        pk: &HyperKZGProverKey<P>,
        poly: &MultilinearPolynomial<P::ScalarField>,
        point: &[<P::ScalarField as JoltField>::Challenge],
        _eval: &P::ScalarField,
        transcript: &mut ProofTranscript,
    ) -> Result<HyperKZGProof<P>, ProofVerifyError> {
        let ell = point.len();
        let n = poly.len();
        assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

        // Phase 1  -- create commitments com_1, ..., com_\ell
        // We do not compute final Pi (and its commitment) as it is constant and equals to 'eval'
        // also known to verifier, so can be derived on its side as well
        let mut polys: Vec<MultilinearPolynomial<P::ScalarField>> = Vec::new();
        polys.push(poly.clone());
        for i in 0..ell - 1 {
            let previous_poly: &DensePolynomial<P::ScalarField> = (&polys[i]).try_into().unwrap();
            let Pi_len = previous_poly.len() / 2;
            let mut Pi = vec![P::ScalarField::zero(); Pi_len];
            Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
                *Pi_j = point[ell - i - 1] * (previous_poly[2 * j + 1] - previous_poly[2 * j])
                    + previous_poly[2 * j];
            });

            polys.push(MultilinearPolynomial::from(Pi));
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // We do not need to commit to the first polynomial as it is already committed.
        let com: Vec<P::G1Affine> = UnivariateKZG::commit_variable_batch(&pk.kzg_pk, &polys[1..])?;

        // Phase 2
        // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
        // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
        transcript.append_points(&com.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();
        let u = vec![r, -r, r * r];

        // Phase 3 -- create response
        let (w, v) = kzg_open_batch(&polys, &u, pk, transcript);

        Ok(HyperKZGProof { com, w, v })
    }

    /// A method to verify purported evaluations of a batch of polynomials
    #[tracing::instrument(skip_all, name = "HyperKZG::verify")]
    pub fn verify<ProofTranscript: Transcript>(
        vk: &HyperKZGVerifierKey<P>,
        C: &HyperKZGCommitment<P>,
        point: &[<P::ScalarField as JoltField>::Challenge],
        P_of_x: &P::ScalarField,
        pi: &HyperKZGProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let y = P_of_x;

        let ell = point.len();

        let mut com = pi.com.clone();

        // we do not need to add x to the transcript, because in our context x was
        // obtained from the transcript
        transcript.append_points(&com.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();

        if r == P::ScalarField::zero() || C.0 == P::G1Affine::zero() {
            return Err(ProofVerifyError::InternalError);
        }
        com.insert(0, C.0); // set com_0 = C, shifts other commitments to the right

        let u = vec![r, -r, r * r];

        // Setup vectors (Y, ypos, yneg) from pi.v
        let v = &pi.v;
        if v.len() != 3 {
            return Err(ProofVerifyError::InternalError);
        }
        if v[0].len() != ell || v[1].len() != ell || v[2].len() != ell {
            return Err(ProofVerifyError::InternalError);
        }
        let ypos = &v[0];
        let yneg = &v[1];
        let mut Y = v[2].to_vec();
        Y.push(*y);

        // Check consistency of (Y, ypos, yneg)
        let two = P::ScalarField::from(2u64);
        for i in 0..ell {
            if two * r * Y[i + 1]
                != r * (P::ScalarField::one() - point[ell - i - 1]) * (ypos[i] + yneg[i])
                    + point[ell - i - 1] * (ypos[i] - yneg[i])
            {
                return Err(ProofVerifyError::InternalError);
            }
            // Note that we don't make any checks about Y[0] here, but our batching
            // check below requires it
        }

        // Check commitments to (Y, ypos, yneg) are valid
        if !kzg_verify_batch(vk, &com, &pi.w, &u, &pi.v, transcript) {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(())
    }
}

/// Specialized implementation for BN254 that provides optimized sparse OneHot commitment.
impl HyperKZG<ark_bn254::Bn254> {
    /// Commit to a OneHotPolynomial without materializing the dense representation.
    /// This exploits the sparsity of OneHot polynomials (only T nonzero coefficients
    /// out of K*T total) by performing T point additions instead of a full MSM.
    ///
    /// The polynomial layout is: coefficient at index `k * T + t` is 1 if `nonzero_indices[t] == Some(k)`.
    #[tracing::instrument(skip_all, name = "HyperKZG::commit_one_hot")]
    pub fn commit_one_hot(
        pk: &HyperKZGProverKey<ark_bn254::Bn254>,
        poly: &OneHotPolynomial<ark_bn254::Fr>,
    ) -> Result<HyperKZGCommitment<ark_bn254::Bn254>, ProofVerifyError> {
        let T = poly.nonzero_indices.len();
        let K = poly.K;
        let required_size = K * T;

        if pk.kzg_pk.g1_powers().len() < required_size {
            return Err(ProofVerifyError::KeyLengthError(
                pk.kzg_pk.g1_powers().len(),
                required_size,
            ));
        }

        // Collect all indices where the coefficient is 1
        // Index formula: k * T + t (where k is the address at timestep t)
        let indices: Vec<usize> = poly
            .nonzero_indices
            .iter()
            .enumerate()
            .filter_map(|(t, k)| k.map(|k| k as usize * T + t))
            .collect();

        if indices.is_empty() {
            return Ok(HyperKZGCommitment(ark_bn254::G1Affine::zero()));
        }

        // Use optimized batch point addition (all coefficients are 1)
        let g1_bases = pk.kzg_pk.g1_powers();
        let indices_slice = [indices];
        let results = jolt_optimizations::batch_g1_additions_multi(g1_bases, &indices_slice);

        Ok(HyperKZGCommitment(results[0]))
    }

    /// Batch commit to multiple OneHotPolynomials efficiently.
    #[tracing::instrument(skip_all, name = "HyperKZG::batch_commit_one_hot")]
    pub fn batch_commit_one_hot(
        pk: &HyperKZGProverKey<ark_bn254::Bn254>,
        polys: &[OneHotPolynomial<ark_bn254::Fr>],
    ) -> Result<Vec<HyperKZGCommitment<ark_bn254::Bn254>>, ProofVerifyError> {
        if polys.is_empty() {
            return Ok(vec![]);
        }

        // Verify SRS is large enough for all polynomials
        let max_required = polys.iter().map(|p| p.K * p.nonzero_indices.len()).max().unwrap_or(0);
        if pk.kzg_pk.g1_powers().len() < max_required {
            return Err(ProofVerifyError::KeyLengthError(
                pk.kzg_pk.g1_powers().len(),
                max_required,
            ));
        }

        // Collect indices for all polynomials
        let all_indices: Vec<Vec<usize>> = polys
            .iter()
            .map(|poly| {
                let T = poly.nonzero_indices.len();
                poly.nonzero_indices
                    .iter()
                    .enumerate()
                    .filter_map(|(t, k)| k.map(|k| k as usize * T + t))
                    .collect()
            })
            .collect();

        let g1_bases = pk.kzg_pk.g1_powers();
        let results = jolt_optimizations::batch_g1_additions_multi(g1_bases, &all_indices);

        Ok(results.into_iter().map(HyperKZGCommitment).collect())
    }

    /// Commit to an RLCPolynomial without materializing one-hot components to dense.
    /// Uses linear homomorphism: commit(Σ c_i * p_i) = Σ c_i * commit(p_i)
    #[tracing::instrument(skip_all, name = "HyperKZG::commit_rlc")]
    pub fn commit_rlc(
        pk: &HyperKZGProverKey<ark_bn254::Bn254>,
        rlc: &RLCPolynomial<ark_bn254::Fr>,
    ) -> Result<HyperKZGCommitment<ark_bn254::Bn254>, ProofVerifyError> {
        use ark_ec::CurveGroup;

        let mut total = ark_bn254::G1Projective::zero();

        // Commit to dense part via standard MSM
        if !rlc.dense_rlc.is_empty() {
            let g1_powers = pk.kzg_pk.g1_powers();
            if g1_powers.len() < rlc.dense_rlc.len() {
                return Err(ProofVerifyError::KeyLengthError(
                    g1_powers.len(),
                    rlc.dense_rlc.len(),
                ));
            }
            let dense_commit: ark_bn254::G1Projective =
                VariableBaseMSM::msm_field_elements(&g1_powers[..rlc.dense_rlc.len()], &rlc.dense_rlc).unwrap();
            total += dense_commit;
        }

        // For each one-hot polynomial: compute sparse commitment and scale by coefficient
        for (coeff, poly_arc) in rlc.one_hot_rlc.iter() {
            let one_hot = match poly_arc.as_ref() {
                MultilinearPolynomial::OneHot(oh) => oh,
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            };
            let oh_commit = Self::commit_one_hot(pk, one_hot)?;
            // Scale by RLC coefficient: coeff * C_oh
            total += oh_commit.0.into_group() * coeff;
        }

        Ok(HyperKZGCommitment(total.into_affine()))
    }
}

impl<P: Pairing> CommitmentScheme for HyperKZG<P>
where
    <P as Pairing>::ScalarField: JoltField,
{
    type Field = P::ScalarField;
    type ProverSetup = HyperKZGProverKey<P>;
    type VerifierSetup = HyperKZGVerifierKey<P>;

    type Commitment = HyperKZGCommitment<P>;
    type Proof = HyperKZGProof<P>;
    type BatchedProof = HyperKZGProof<P>;
    type OpeningProofHint = ();

    const REQUIRES_MATERIALIZED_POLYS: bool = true;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        HyperKZGSRS(Arc::new(SRS::setup(
            &mut ChaCha20Rng::from_seed(*b"HyperKZG_POLY_COMMITMENT_SCHEMEE"),
            1 << max_num_vars,
            2,
        )))
        .trim(1 << max_num_vars)
        .0
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HyperKZGVerifierKey {
            kzg_vk: KZGVerifierKey::from(&setup.kzg_pk),
        }
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::commit")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        assert!(
            setup.kzg_pk.g1_powers().len() >= poly.len(),
            "COMMIT KEY LENGTH ERROR {}, {}",
            setup.kzg_pk.g1_powers().len(),
            poly.len()
        );
        let commitment =
            HyperKZGCommitment(UnivariateKZG::commit_as_univariate(&setup.kzg_pk, poly).unwrap());
        (commitment, ())
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::batch_commit")]
    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        UnivariateKZG::commit_batch(&gens.kzg_pk, polys)
            .unwrap()
            .into_par_iter()
            .map(|c| (HyperKZGCommitment(c), ()))
            .collect()
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let combined_commitment: P::G1 = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.borrow().0 * coeff)
            .sum();
        HyperKZGCommitment(combined_commitment.into_affine())
    }

    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        // HyperKZG doesn't use hints, so combining is trivial
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        eval: &Self::Field,
        _hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        HyperKZG::<P>::open(setup, poly, opening_point, eval, transcript).unwrap()
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge], // point at which the polynomial is evaluated
        opening: &Self::Field,                                   // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        HyperKZG::<P>::verify(setup, commitment, opening_point, opening, proof, transcript)
    }

    fn protocol_name() -> &'static [u8] {
        b"hyperkzg"
    }
}

/// Chunk data stored during streaming for HyperKZG.
/// Since HyperKZG accepts linear memory (for small traces), we store the actual data
/// and commit at the end rather than computing partial commitments.
#[derive(Clone, Debug, PartialEq)]
pub enum HyperKZGChunkData {
    /// Dense polynomial chunk: stores field elements directly
    Dense(Vec<ark_bn254::Fr>),
    /// OneHot polynomial chunk: stores indices per timestep (None = no contribution)
    OneHot { k: usize, indices: Vec<Option<usize>> },
}

impl super::commitment_scheme::StreamingCommitmentScheme for HyperKZG<ark_bn254::Bn254> {
    type ChunkState = HyperKZGChunkData;

    fn process_chunk<T: SmallScalar>(
        _setup: &Self::ProverSetup,
        chunk: &[T],
    ) -> Self::ChunkState {
        let data: Vec<ark_bn254::Fr> = chunk.iter().map(|x| x.to_field()).collect();
        HyperKZGChunkData::Dense(data)
    }

    fn process_chunk_onehot(
        _setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        HyperKZGChunkData::OneHot {
            k: onehot_k,
            indices: chunk.to_vec(),
        }
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::aggregate_chunks")]
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        use crate::poly::commitment::dory::DoryGlobals;

        if chunks.is_empty() {
            return (HyperKZGCommitment(ark_bn254::G1Affine::zero()), ());
        }

        let commitment = if onehot_k.is_some() {
            // OneHot polynomial: reconstruct indices and use sparse commit
            let T = DoryGlobals::get_T();
            let K = match &chunks[0] {
                HyperKZGChunkData::OneHot { k, .. } => *k,
                _ => panic!("Expected OneHot chunk data"),
            };

            // Collect all indices across chunks
            let mut all_indices: Vec<Option<u8>> = Vec::with_capacity(T);
            for chunk in chunks {
                match chunk {
                    HyperKZGChunkData::OneHot { indices, .. } => {
                        for &idx in indices {
                            all_indices.push(idx.map(|i| i as u8));
                        }
                    }
                    _ => panic!("Mixed chunk types"),
                }
            }

            // Build OneHotPolynomial and commit
            let one_hot_poly = OneHotPolynomial::<ark_bn254::Fr>::from_indices(all_indices, K);
            Self::commit_one_hot(setup, &one_hot_poly)
                .expect("OneHot commit failed")
        } else {
            // Dense polynomial: concatenate all chunks and do MSM
            let total_len: usize = chunks
                .iter()
                .map(|c| match c {
                    HyperKZGChunkData::Dense(data) => data.len(),
                    _ => panic!("Expected Dense chunk data"),
                })
                .sum();

            let mut full_poly = Vec::with_capacity(total_len);
            for chunk in chunks {
                match chunk {
                    HyperKZGChunkData::Dense(data) => full_poly.extend_from_slice(data),
                    _ => panic!("Mixed chunk types"),
                }
            }

            let poly: MultilinearPolynomial<ark_bn254::Fr> = full_poly.into();
            Self::commit(setup, &poly).expect("Commit failed")
        };

        (commitment, ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::multilinear_polynomial::PolynomialEvaluation;
    use crate::transcripts::{Blake2bTranscript, Transcript};
    use ark_bn254::Bn254;
    use ark_std::UniformRand;
    use rand::Rng;
    use rand_core::SeedableRng;

    type Fr = <Bn254 as Pairing>::ScalarField;
    type Challenge = <Fr as JoltField>::Challenge;

    #[test]
    fn test_hyperkzg_eval() {
        // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let srs = HyperKZGSRS::setup(&mut rng, 3);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

        // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
        let poly =
            MultilinearPolynomial::from(vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)]);

        let c = HyperKZG::commit(&pk, &poly).unwrap();

        let test_inner =
            |point: Vec<Challenge>| -> Result<(), ProofVerifyError> {
                // Compute the expected evaluation dynamically
                let eval = poly.evaluate(&point);
                let mut tr = Blake2bTranscript::new(b"TestEval");
                let proof = HyperKZG::open(&pk, &poly, &point, &eval, &mut tr).unwrap();
                let mut tr = Blake2bTranscript::new(b"TestEval");
                HyperKZG::verify(&vk, &c, &point, &eval, &proof, &mut tr)
            };

        let test_inner_wrong =
            |point: Vec<Challenge>, wrong_eval: Fr| -> Result<(), ProofVerifyError> {
                let mut tr = Blake2bTranscript::new(b"TestEval");
                let proof = HyperKZG::open(&pk, &poly, &point, &wrong_eval, &mut tr).unwrap();
                let mut tr = Blake2bTranscript::new(b"TestEval");
                HyperKZG::verify(&vk, &c, &point, &wrong_eval, &proof, &mut tr)
            };

        // Test various evaluation points - eval is computed dynamically
        let point = vec![Challenge::from(0u128), Challenge::from(0u128)];
        assert!(test_inner(point).is_ok());

        let point = vec![Challenge::from(0u128), Challenge::from(1u128)];
        assert!(test_inner(point).is_ok());

        let point = vec![Challenge::from(1u128), Challenge::from(1u128)];
        assert!(test_inner(point).is_ok());

        let point = vec![Challenge::from(0u128), Challenge::from(2u128)];
        assert!(test_inner(point).is_ok());

        let point = vec![Challenge::from(2u128), Challenge::from(2u128)];
        assert!(test_inner(point).is_ok());

        // Random points
        let point = vec![Challenge::from(12345u128), Challenge::from(67890u128)];
        assert!(test_inner(point).is_ok());

        // Try incorrect evaluations and expect failure
        let point = vec![Challenge::from(2u128), Challenge::from(2u128)];
        let correct_eval = poly.evaluate(&point);
        assert!(test_inner_wrong(point, correct_eval + Fr::from(1)).is_err());
    }

    #[test]
    fn test_hyperkzg_small() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

        // poly = [1, 2, 1, 4]
        let poly =
            MultilinearPolynomial::from(vec![Fr::from(1), Fr::from(2), Fr::from(1), Fr::from(4)]);

        // point = [4,3] using MontU128Challenge
        let point = vec![Challenge::from(4u128), Challenge::from(3u128)];

        // Compute eval dynamically
        let eval = poly.evaluate(&point);

        let srs = HyperKZGSRS::setup(&mut rng, 3);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

        // make a commitment
        let c = HyperKZG::commit(&pk, &poly).unwrap();

        // prove an evaluation
        let mut tr = Blake2bTranscript::new(b"TestEval");
        let proof = HyperKZG::open(&pk, &poly, &point, &eval, &mut tr).unwrap();
        let post_c_p = tr.challenge_scalar::<Fr>();

        // verify the evaluation
        let mut verifier_transcript = Blake2bTranscript::new(b"TestEval");
        assert!(
            HyperKZG::verify(&vk, &c, &point, &eval, &proof, &mut verifier_transcript,).is_ok()
        );
        let post_c_v = verifier_transcript.challenge_scalar::<Fr>();

        // check if the prover transcript and verifier transcript are kept in the same state
        assert_eq!(post_c_p, post_c_v);

        let mut proof_bytes = Vec::new();
        proof.serialize_compressed(&mut proof_bytes).unwrap();
        assert_eq!(proof_bytes.len(), 368);

        // Change the proof and expect verification to fail
        let mut bad_proof = proof.clone();
        let v1 = bad_proof.v[1].clone();
        bad_proof.v[0].clone_from(&v1);
        let mut verifier_transcript2 = Blake2bTranscript::new(b"TestEval");
        assert!(HyperKZG::verify(
            &vk,
            &c,
            &point,
            &eval,
            &bad_proof,
            &mut verifier_transcript2
        )
        .is_err());
    }

    #[test]
    fn test_hyperkzg_srs_file_roundtrip() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let max_degree = 1 << 8; // 256 elements

        // Create SRS
        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, max_degree);
        assert_eq!(srs.max_degree(), max_degree);

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let srs_path = temp_dir.join("test_hyperkzg_srs.bin");
        srs.save_to_file(&srs_path).unwrap();

        // Load from file
        let loaded_srs = HyperKZGSRS::<Bn254>::load_from_file(&srs_path).unwrap();
        assert_eq!(loaded_srs.max_degree(), max_degree);

        // Verify they produce the same keys
        let (pk1, _vk1) = srs.trim(max_degree);
        let (pk2, vk2) = loaded_srs.trim(max_degree);

        // Create a test polynomial and verify both setups produce the same commitment
        let poly = MultilinearPolynomial::from(
            (0..max_degree)
                .map(|i| Fr::from(i as u64))
                .collect::<Vec<_>>(),
        );

        let c1 = HyperKZG::commit(&pk1, &poly).unwrap();
        let c2 = HyperKZG::commit(&pk2, &poly).unwrap();
        assert_eq!(c1, c2);

        // Verify proof works with loaded SRS
        let point: Vec<Challenge> = (0..8)
            .map(|i| Challenge::from((i * 7) as u128))
            .collect();
        let eval = poly.evaluate(&point);

        let mut tr1 = Blake2bTranscript::new(b"TestSRS");
        let proof = HyperKZG::open(&pk2, &poly, &point, &eval, &mut tr1).unwrap();

        let mut tr2 = Blake2bTranscript::new(b"TestSRS");
        assert!(HyperKZG::verify(&vk2, &c2, &point, &eval, &proof, &mut tr2).is_ok());

        // Cleanup
        std::fs::remove_file(&srs_path).ok();
    }

    #[test]
    fn test_hyperkzg_large() {
        // test the hyperkzg prover and verifier with random instances (derived from a seed)
        for ell in [8, 9, 10] {
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(ell as u64);

            let n = 1 << ell; // n = 2^ell

            let poly_raw = (0..n)
                .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();
            let poly = MultilinearPolynomial::from(poly_raw.clone());
            let point = (0..ell)
                .map(|_| {
                    <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::from(
                        rng.gen::<u128>(),
                    )
                })
                .collect::<Vec<_>>();
            let eval = poly.evaluate(&point);

            let srs = HyperKZGSRS::setup(&mut rng, n);
            let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(n);

            // make a commitment
            let C = HyperKZG::commit(&pk, &poly).unwrap();

            // prove an evaluation
            let mut prover_transcript = Blake2bTranscript::new(b"TestEval");
            let proof: HyperKZGProof<Bn254> =
                HyperKZG::open(&pk, &poly, &point, &eval, &mut prover_transcript).unwrap();

            // verify the evaluation
            let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
            assert!(HyperKZG::verify(&vk, &C, &point, &eval, &proof, &mut verifier_tr,).is_ok());

            // Change the proof and expect verification to fail
            let mut bad_proof = proof.clone();
            let v1 = bad_proof.v[1].clone();
            bad_proof.v[0].clone_from(&v1);
            let mut verifier_tr2 = Blake2bTranscript::new(b"TestEval");
            assert!(
                HyperKZG::verify(&vk, &C, &point, &eval, &bad_proof, &mut verifier_tr2,).is_err()
            );
        }
    }

    #[test]
    fn test_hyperkzg_one_hot_commit() {
        use crate::poly::commitment::dory::DoryGlobals;
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        // Test parameters: K (address space) and T (trace length)
        // Using small values for fast testing
        let K: usize = 16; // 2^4 addresses
        let T: usize = 64; // 2^6 timesteps
        let total_size = K * T; // 1024 elements

        // Initialize DoryGlobals (required by OneHotPolynomial::from_indices)
        let _guard = DoryGlobals::initialize(K, T);

        // Setup SRS large enough for K * T
        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
        let (pk, _vk) = srs.trim(total_size);

        // Generate random nonzero indices (simulating random addresses per timestep)
        let nonzero_indices: Vec<Option<u8>> = (0..T)
            .map(|_| Some((rng.gen::<u64>() % K as u64) as u8))
            .collect();

        // Create OneHotPolynomial
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

        // Commit using sparse method
        let sparse_commitment = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

        // Create equivalent dense polynomial for comparison
        let mut dense_coeffs = vec![Fr::zero(); total_size];
        for (t, k) in nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let idx = *k as usize * T + t;
                dense_coeffs[idx] = Fr::one();
            }
        }
        let dense_poly = MultilinearPolynomial::from(dense_coeffs);

        // Commit using dense method
        let dense_commitment = HyperKZG::<Bn254>::commit(&pk, &dense_poly).unwrap();

        // Both commitments should be equal
        assert_eq!(
            sparse_commitment.0, dense_commitment.0,
            "Sparse and dense OneHot commitments should match"
        );
    }

    #[test]
    fn test_hyperkzg_one_hot_commit_with_none_indices() {
        use crate::poly::commitment::dory::DoryGlobals;
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(123);

        let K: usize = 8;
        let T: usize = 32;
        let total_size = K * T;

        // Initialize DoryGlobals
        let _guard = DoryGlobals::initialize(K, T);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
        let (pk, _vk) = srs.trim(total_size);

        // Generate indices with some None values (simulating no memory access at some timesteps)
        let nonzero_indices: Vec<Option<u8>> = (0..T)
            .map(|i| {
                if i % 3 == 0 {
                    None // No access at every 3rd timestep
                } else {
                    Some((rng.gen::<u64>() % K as u64) as u8)
                }
            })
            .collect();

        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

        // Commit using sparse method
        let sparse_commitment = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

        // Create equivalent dense polynomial
        let mut dense_coeffs = vec![Fr::zero(); total_size];
        for (t, k) in nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let idx = *k as usize * T + t;
                dense_coeffs[idx] = Fr::one();
            }
        }
        let dense_poly = MultilinearPolynomial::from(dense_coeffs);

        let dense_commitment = HyperKZG::<Bn254>::commit(&pk, &dense_poly).unwrap();

        assert_eq!(
            sparse_commitment.0, dense_commitment.0,
            "Sparse and dense commitments should match even with None indices"
        );
    }

    #[test]
    fn test_hyperkzg_batch_one_hot_commit() {
        use crate::poly::commitment::dory::DoryGlobals;
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(999);

        let K: usize = 16;
        let T: usize = 64;
        let total_size = K * T;
        let num_polys = 5;

        // Initialize DoryGlobals
        let _guard = DoryGlobals::initialize(K, T);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
        let (pk, _vk) = srs.trim(total_size);

        // Generate multiple OneHotPolynomials
        let polys: Vec<OneHotPolynomial<Fr>> = (0..num_polys)
            .map(|_| {
                let nonzero_indices: Vec<Option<u8>> = (0..T)
                    .map(|_| Some((rng.gen::<u64>() % K as u64) as u8))
                    .collect();
                OneHotPolynomial::from_indices(nonzero_indices, K)
            })
            .collect();

        // Batch commit
        let batch_commitments = HyperKZG::<Bn254>::batch_commit_one_hot(&pk, &polys).unwrap();

        // Individual commits
        let individual_commitments: Vec<_> = polys
            .iter()
            .map(|p| HyperKZG::<Bn254>::commit_one_hot(&pk, p).unwrap())
            .collect();

        // All should match
        for (batch, individual) in batch_commitments.iter().zip(individual_commitments.iter()) {
            assert_eq!(batch.0, individual.0, "Batch and individual commits should match");
        }
    }

    #[test]
    fn test_hyperkzg_one_hot_empty() {
        use crate::poly::commitment::dory::DoryGlobals;
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(456);

        let K: usize = 8;
        let T: usize = 16;
        let total_size = K * T;

        // Initialize DoryGlobals
        let _guard = DoryGlobals::initialize(K, T);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
        let (pk, _vk) = srs.trim(total_size);

        // All None indices (completely zero polynomial)
        let nonzero_indices: Vec<Option<u8>> = vec![None; T];

        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);

        let commitment = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

        // Commitment to zero polynomial should be the identity element
        assert!(
            commitment.0.is_zero(),
            "Commitment to all-None OneHot should be zero"
        );
    }

    #[test]
    fn test_hyperkzg_commit_rlc() {
        use crate::poly::commitment::dory::DoryGlobals;
        use crate::poly::multilinear_polynomial::MultilinearPolynomial;
        use crate::poly::one_hot_polynomial::OneHotPolynomial;
        use crate::poly::rlc_polynomial::RLCPolynomial;
        use std::sync::Arc;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(789);

        let K: usize = 8;
        let T: usize = 16;
        let total_size = K * T;

        let _guard = DoryGlobals::initialize(K, T);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
        let (pk, _vk) = srs.trim(total_size);

        // Create a dense polynomial (length T)
        let dense_coeffs: Vec<Fr> = (0..T).map(|i| Fr::from(i as u64 + 1)).collect();

        // Create a one-hot polynomial
        let nonzero_indices: Vec<Option<u8>> = (0..T).map(|t| Some((t % K) as u8)).collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

        // RLC coefficients
        let coeff_dense = Fr::from(3u64);
        let coeff_onehot = Fr::from(5u64);

        // Build RLCPolynomial manually
        let rlc = RLCPolynomial {
            dense_rlc: dense_coeffs.iter().map(|&c| c * coeff_dense).collect(),
            one_hot_rlc: vec![(
                coeff_onehot,
                Arc::new(MultilinearPolynomial::OneHot(one_hot_poly.clone())),
            )],
            streaming_context: None,
        };

        // Commit via commit_rlc (sparse one-hot path)
        let rlc_commit = HyperKZG::<Bn254>::commit_rlc(&pk, &rlc).unwrap();

        // Now compute expected commitment by materializing everything
        // dense contribution: coeff_dense * dense_coeffs[i] at index i
        // one-hot contribution: coeff_onehot * 1 at index k*T+t where k = nonzero_indices[t]
        let mut expected_dense = vec![Fr::zero(); total_size];
        for (i, &c) in dense_coeffs.iter().enumerate() {
            expected_dense[i] = coeff_dense * c;
        }
        for (t, &maybe_k) in nonzero_indices.iter().enumerate() {
            if let Some(k) = maybe_k {
                let idx = (k as usize) * T + t;
                expected_dense[idx] += coeff_onehot;
            }
        }

        let expected_poly: MultilinearPolynomial<Fr> = expected_dense.into();
        let expected_commit = HyperKZG::<Bn254>::commit(&pk, &expected_poly).unwrap();

        assert_eq!(
            rlc_commit.0, expected_commit.0,
            "commit_rlc should match materialized dense commit"
        );
    }

    #[test]
    fn test_hyperkzg_streaming_dense() {
        use super::super::commitment_scheme::StreamingCommitmentScheme;
        use crate::poly::commitment::dory::DoryGlobals;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(999);

        let row_len: usize = 8;
        let num_rows: usize = 4;
        let T = row_len * num_rows;

        let _guard = DoryGlobals::initialize(1, T);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, T);
        let (pk, _vk) = srs.trim(T);

        // Create polynomial data as i64 (SmallScalar)
        let poly_data_small: Vec<i64> = (0..T).map(|i| i as i64 + 1).collect();

        // Streaming path: process chunks and aggregate
        let chunks: Vec<HyperKZGChunkData> = poly_data_small
            .chunks(row_len)
            .map(|chunk| HyperKZG::<Bn254>::process_chunk(&pk, chunk))
            .collect();

        let (streaming_commit, _) =
            HyperKZG::<Bn254>::aggregate_chunks(&pk, None, &chunks);

        // Direct path: commit all at once
        let poly_data_fr: Vec<Fr> = poly_data_small.iter().map(|&x| Fr::from(x)).collect();
        let poly: MultilinearPolynomial<Fr> = poly_data_fr.into();
        let direct_commit = HyperKZG::<Bn254>::commit(&pk, &poly).unwrap();

        assert_eq!(
            streaming_commit.0, direct_commit.0,
            "Streaming commit should match direct commit for dense polynomial"
        );
    }

    #[test]
    fn test_hyperkzg_streaming_onehot() {
        use super::super::commitment_scheme::StreamingCommitmentScheme;
        use crate::poly::commitment::dory::DoryGlobals;
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(888);

        let K: usize = 8;
        let row_len: usize = 4;
        let num_rows: usize = 4;
        let T = row_len * num_rows;
        let total_size = K * T;

        let _guard = DoryGlobals::initialize(K, T);

        let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
        let (pk, _vk) = srs.trim(total_size);

        // Create onehot indices
        let indices: Vec<Option<usize>> = (0..T).map(|t| Some(t % K)).collect();

        // Streaming path: process chunks and aggregate
        let chunks: Vec<HyperKZGChunkData> = indices
            .chunks(row_len)
            .map(|chunk| HyperKZG::<Bn254>::process_chunk_onehot(&pk, K, chunk))
            .collect();

        let (streaming_commit, _) =
            HyperKZG::<Bn254>::aggregate_chunks(&pk, Some(K), &chunks);

        // Direct path: commit using commit_one_hot
        let indices_u8: Vec<Option<u8>> = indices.iter().map(|&i| i.map(|x| x as u8)).collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(indices_u8, K);
        let direct_commit = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

        assert_eq!(
            streaming_commit.0, direct_commit.0,
            "Streaming commit should match direct commit for onehot polynomial"
        );
    }
}
