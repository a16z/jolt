use crate::poly::hyrax::BatchedHyraxOpeningProof;
use crate::utils::compute_dotproduct_low_optimized;
use crate::utils::thread::allocate_vec_in_background;
use crate::utils::thread::allocate_zero_vec_background;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::AppendToTranscript;
use crate::utils::transcript::ProofTranscript;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rayon::prelude::*;
use thiserror::Error;

use super::r1cs_shape::R1CSShape;
use crate::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        hyrax::{HyraxCommitment, HyraxGenerators},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
};
use common::constants::NUM_R1CS_POLYS;

pub struct UniformSpartanKey<F: PrimeField> {
    shape_single_step: R1CSShape<F>, // Single step shape
    num_cons_total: usize,           // Number of constraints
    num_vars_total: usize,           // Number of variables
    num_steps: usize,                // Number of steps
    vk_digest: F,                    // digest of the verifier's key
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,

    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidInnerSumcheckProof")]
    InvalidInnerSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidInnerSumcheckClaim")]
    InvalidInnerSumcheckClaim,

    /// returned if the supplied witness is not of the right length
    #[error("InvalidWitnessLength")]
    InvalidWitnessLength,

    /// returned when an invalid Hyrax proof is provided
    #[error("InvalidHyraxProof")]
    InvalidHyraxProof,
}

// Trait which will kick out a small and big R1CS shape
pub trait UniformShapeBuilder<F: PrimeField> {
    fn single_step_shape(&self) -> R1CSShape<F>;
}

// TODO: Rather than use these adhoc virtual indexable polys – create a DensePolynomial which takes any impl Index<usize> inner
// and can run all the normal DensePolynomial ops.
pub struct SegmentedPaddedWitness<F: PrimeField> {
    total_len: usize,
    segments: Vec<Vec<F>>,
    segment_len: usize,
    zero: F,
}

impl<F: PrimeField> SegmentedPaddedWitness<F> {
    pub fn new(total_len: usize, segments: Vec<Vec<F>>) -> Self {
        let segment_len = segments[0].len();
        for segment in &segments {
            assert_eq!(
                segment.len(),
                segment_len,
                "All segments must be the same length"
            );
        }
        SegmentedPaddedWitness {
            total_len,
            segments,
            segment_len,
            zero: F::ZERO,
        }
    }

    pub fn len(&self) -> usize {
        self.total_len
    }

    #[tracing::instrument(skip_all, name = "SegmentedPaddedWitness::evaluate_all")]
    pub fn evaluate_all(&self, point: Vec<F>) -> Vec<F> {
        let chi = EqPolynomial::new(point).evals();
        let results = self.segments
            .par_iter()
            .map(|segment| compute_dotproduct_low_optimized(&chi, segment))
            .collect();
        // drop_in_background_thread(chi);
        std::mem::forget(chi);
        results
    }

    pub fn into_dense_polys(self) -> Vec<DensePolynomial<F>> {
        self.segments
            .into_iter()
            .map(|poly| DensePolynomial::new(poly))
            .collect()
    }
}

impl<F: PrimeField> std::ops::Index<usize> for SegmentedPaddedWitness<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.segments.len() * self.segment_len {
            &self.zero
        } else if index >= self.total_len {
            panic!("index too high");
        } else {
            let segment_index = index / self.segment_len;
            let inner_index = index % self.segment_len;
            &self.segments[segment_index][inner_index]
        }
    }
}

pub trait IndexablePoly<F: PrimeField>: std::ops::Index<usize, Output = F> + Sync {
    fn len(&self) -> usize;
}

impl<F: PrimeField> IndexablePoly<F> for SegmentedPaddedWitness<F> {
    fn len(&self) -> usize {
        self.total_len
    }
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
pub struct UniformSpartanProof<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    outer_sumcheck_proof: SumcheckInstanceProof<F>,
    outer_sumcheck_claims: (F, F, F),
    inner_sumcheck_proof: SumcheckInstanceProof<F>,
    eval_arg: Vec<F>, // TODO(arasuarun / sragss): better name
    claimed_witnesss_evals: Vec<F>,
    opening_proof: BatchedHyraxOpeningProof<NUM_R1CS_POLYS, G>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> UniformSpartanProof<F, G> {
    #[tracing::instrument(skip_all, name = "UniformSpartanProof::setup_precommitted")]
    pub fn setup_precommitted<C: UniformShapeBuilder<F>>(
        circuit: &C,
        num_steps: usize,
    ) -> Result<UniformSpartanKey<F>, SpartanError> {
        let shape_single_step = circuit.single_step_shape();

        let num_constraints_total = shape_single_step.num_cons * num_steps;
        let num_aux_total = shape_single_step.num_vars * num_steps;

        let pad_num_constraints = num_constraints_total.next_power_of_two();
        let pad_num_aux = num_aux_total.next_power_of_two();

        // TODO(sragss / arasuarun): Verifier key digest
        let vk_digest = F::one();

        let key = UniformSpartanKey {
            shape_single_step,
            num_cons_total: pad_num_constraints,
            num_vars_total: pad_num_aux,
            num_steps,
            vk_digest,
        };
        Ok(key)
    }

    /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
    #[tracing::instrument(skip_all, name = "UniformSpartanProof::prove_precommitted")]
    pub fn prove_precommitted(
        key: &UniformSpartanKey<F>,
        witness_segments: Vec<Vec<F>>,
        transcript: &mut Transcript,
    ) -> Result<Self, SpartanError> {
        // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
        <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"vk", &key.vk_digest);

        let poly_ABC_len = 2 * key.num_vars_total;
        let RLC_evals_alloc = allocate_zero_vec_background::<F>( poly_ABC_len);

        let segmented_padded_witness =
            SegmentedPaddedWitness::new(key.num_vars_total, witness_segments);

        let num_rounds_x = key.num_cons_total.ilog2() as usize;
        let num_rounds_y = key.num_vars_total.ilog2() as usize + 1;

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"t"))
            .collect::<Vec<F>>();

        let combined_witness_size = (key.num_steps * key.shape_single_step.num_cons).next_power_of_two();
        let A_z = allocate_vec_in_background(F::zero(), combined_witness_size);
        let B_z = allocate_vec_in_background(F::zero(), combined_witness_size);
        let C_z = allocate_vec_in_background(F::zero(), combined_witness_size);

        let mut poly_tau = DensePolynomial::new(EqPolynomial::new(tau).evals());

        let span = tracing::span!(tracing::Level::TRACE, "wait_join");
        let _enter = span.enter();
        let mut A_z = A_z.join().unwrap();
        let mut B_z = B_z.join().unwrap();
        let mut C_z = C_z.join().unwrap();
        drop(_enter);

        key.shape_single_step.multiply_vec_uniform(
            &segmented_padded_witness,
            key.num_steps,
            &mut A_z,
            &mut B_z,
            &mut C_z
        )?;
        let mut poly_Az = DensePolynomial::new(A_z);
        let mut poly_Bz = DensePolynomial::new(B_z);
        let mut poly_Cz = DensePolynomial::new(C_z);



        let comb_func_outer = |A: &F, B: &F, C: &F, D: &F| -> F {
            // Below is an optimized form of: *A * (*B * *C - *D)
            if B.is_zero() || C.is_zero() {
                if D.is_zero() {
                    F::zero()
                } else {
                    *A * (-(*D))
                }
            } else {
                *A * (*B * *C - *D)
            }
        };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic::<G, _>(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut poly_tau,
                &mut poly_Az,
                &mut poly_Bz,
                &mut poly_Cz,
                comb_func_outer,
                transcript,
            );
        drop_in_background_thread(poly_Az);
        drop_in_background_thread(poly_Bz);
        drop_in_background_thread(poly_Cz);
        drop_in_background_thread(poly_tau);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );
        <Transcript as ProofTranscript<G>>::append_scalars(
            transcript,
            b"claims_outer",
            &[claim_Az, claim_Bz, claim_Cz].as_slice(),
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F =
            <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"r");
        let claim_inner_joint = claim_Az
            + r_inner_sumcheck_RLC * claim_Bz
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * claim_Cz;

        let span = tracing::span!(tracing::Level::TRACE, "poly_ABC");
        let _enter = span.enter();

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let poly_ABC = {
            let num_steps_bits = key.num_steps.trailing_zeros();
            let (rx_con, rx_ts) =
                outer_sumcheck_r.split_at(outer_sumcheck_r.len() - num_steps_bits as usize);
            let (eq_rx_con, eq_rx_ts) = rayon::join(
                || EqPolynomial::new(rx_con.to_vec()).evals(),
                || EqPolynomial::new(rx_ts.to_vec()).evals(),
            );
            let n_steps = key.num_steps;

            // With uniformity, each entry of the RLC of A, B, C can be expressed using
            // the RLC of the small_A, small_B, small_C matrices.

            // 1. Evaluate \tilde smallM(r_x, y) for all y
            let compute_eval_table_sparse_single = |small_M: &Vec<(usize, usize, F)>| -> Vec<F> {
                let mut small_M_evals = vec![F::zero(); key.shape_single_step.num_vars + 1];
                for (row, col, val) in small_M.iter() {
                    small_M_evals[*col] += eq_rx_con[*row] * val;
                }
                small_M_evals
            };

            let (small_A_evals, (small_B_evals, small_C_evals)) = rayon::join(
                || compute_eval_table_sparse_single(&key.shape_single_step.A),
                || {
                    rayon::join(
                        || compute_eval_table_sparse_single(&key.shape_single_step.B),
                        || compute_eval_table_sparse_single(&key.shape_single_step.C),
                    )
                },
            );

            let span = tracing::span!(tracing::Level::TRACE, "poly_ABC_small_RLC_evals");
            let _enter = span.enter();
            let r_sq = r_inner_sumcheck_RLC * r_inner_sumcheck_RLC;
            let small_RLC_evals = (0..small_A_evals.len())
                .into_par_iter()
                .map(|i| {
                    small_A_evals[i]
                        + small_B_evals[i] * r_inner_sumcheck_RLC
                        + small_C_evals[i] * r_sq
                })
                .collect::<Vec<F>>();
            drop(_enter);

            // 2. Handles all entries but the last one with the constant 1 variable
            let other_span = tracing::span!(tracing::Level::TRACE, "poly_ABC_wait_alloc_complete");
            let _other_enter = other_span.enter();
            let mut RLC_evals = RLC_evals_alloc.join().unwrap();
            drop(_other_enter);

            let span = tracing::span!(tracing::Level::TRACE, "poly_ABC_big_RLC_evals");
            let _enter = span.enter();
            // small_RLC_evals is 30% ones.
            RLC_evals.par_chunks_mut(n_steps).take(key.num_vars_total / n_steps).enumerate().for_each(|(chunk_index, rlc_chunk)| {
                if !small_RLC_evals[chunk_index].is_zero() {
                    for (eq_index, item) in rlc_chunk.iter_mut().enumerate() {
                        *item = eq_rx_ts[eq_index] * small_RLC_evals[chunk_index];
                    }
                }
            });
            drop(_enter);

            // 3. Handles the constant 1 variable
            let span = tracing::span!(tracing::Level::TRACE, "poly_ABC_constant");
            let _enter = span.enter();
            let eq_sum = eq_rx_ts.par_iter().sum::<F>();
            let compute_eval_constant_column = |small_M: &Vec<(usize, usize, F)>| -> F {
                let constant_sum: F = small_M.iter()
                    .filter(|(_, col, _)| *col == key.shape_single_step.num_vars)   // expecting ~1
                    .map(|(row, _, val)| {
                        *val * eq_rx_con[*row] * eq_sum
                    }).sum();

                constant_sum
            };

            let (constant_term_A, (constant_term_B, constant_term_C)) = rayon::join(
                || compute_eval_constant_column(&key.shape_single_step.A),
                || {
                    rayon::join(
                        || compute_eval_constant_column(&key.shape_single_step.B),
                        || compute_eval_constant_column(&key.shape_single_step.C),
                    )
                },
            );
            drop(_enter);

            RLC_evals[key.num_vars_total] = constant_term_A
                + r_inner_sumcheck_RLC * constant_term_B
                + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * constant_term_C;

            RLC_evals
        };
        drop(_enter);
        drop(span);

        let mut poly_ABC = DensePolynomial::new(poly_ABC);
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_spartan_quadratic::<G, _>(
                &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
                num_rounds_y,
                &mut poly_ABC, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
                &segmented_padded_witness,
                transcript,
            );
        drop_in_background_thread(poly_ABC);

        // The number of prefix bits needed to identify a segment within the witness vector
        // assuming that num_vars_total is a power of 2 and each segment has length num_steps, which is also a power of 2.
        // The +1 is the first element used to separate the inputs and the witness.
        // TODO(sragss): Are these `.trailing_zeros()` calls in place of log_2()?
        let n_prefix = (key.num_vars_total.trailing_zeros() as usize
            - key.num_steps.trailing_zeros() as usize)
            + 1; // TODO(sragss): This is a hack!
        let r_y_point = &inner_sumcheck_r[n_prefix..];

        // Evaluate each segment on r_y_point
        let witness_evals = segmented_padded_witness.evaluate_all(r_y_point.to_owned());

        let witness_segment_polys: Vec<DensePolynomial<F>> =
            segmented_padded_witness.into_dense_polys();
        let witness_segment_polys_ref: Vec<&DensePolynomial<F>> = witness_segment_polys
            .iter()
            .map(|poly_ref| poly_ref)
            .collect();
        let opening_proof = BatchedHyraxOpeningProof::prove(
            &witness_segment_polys_ref,
            &r_y_point,
            &witness_evals,
            transcript,
        );

        drop_in_background_thread(witness_segment_polys);

        // Outer sumcheck claims: [eq(r_x), A(r_x), B(r_x), C(r_x)]
        let outer_sumcheck_claims = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );
        Ok(UniformSpartanProof {
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            eval_arg: vec![],
            claimed_witnesss_evals: witness_evals,
            opening_proof,
        })
    }

    /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
    #[tracing::instrument(skip_all, name = "SNARK::verify")]
    pub fn verify_precommitted(
        &self,
        witness_segment_commitments: Vec<&HyraxCommitment<NUM_R1CS_POLYS, G>>,
        key: &UniformSpartanKey<F>,
        io: &[F],
        generators: &HyraxGenerators<NUM_R1CS_POLYS, G>,
        transcript: &mut Transcript,
    ) -> Result<(), SpartanError> {
        assert_eq!(io.len(), 0); // Currently not using io

        let N_SEGMENTS = witness_segment_commitments.len();

        // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
        <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"vk", &key.vk_digest);

        let (num_rounds_x, num_rounds_y) = (
            usize::try_from(key.num_cons_total.ilog2()).unwrap(),
            (usize::try_from(key.num_vars_total.ilog2()).unwrap() + 1),
        );

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"t"))
            .collect::<Vec<F>>();

        let (claim_outer_final, r_x) = self
            .outer_sumcheck_proof
            .verify::<G, Transcript>(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // verify claim_outer_final
        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidOuterSumcheckClaim);
        }

        <Transcript as ProofTranscript<G>>::append_scalars(
            transcript,
            b"claims_outer",
            &[
                self.outer_sumcheck_claims.0,
                self.outer_sumcheck_claims.1,
                self.outer_sumcheck_claims.2,
            ]
            .as_slice(),
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F =
            <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"r");
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + r_inner_sumcheck_RLC * self.outer_sumcheck_claims.1
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * self.outer_sumcheck_claims.2;

        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify::<G, Transcript>(claim_inner_joint, num_rounds_y, 2, transcript)
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        // verify claim_inner_final
        // this should be log (num segments)
        let n_prefix = (key.num_vars_total.trailing_zeros() as usize
            - key.num_steps.trailing_zeros() as usize)
            + 1; // TODO(sragss): HACK!

        let eval_Z = {
            let eval_X = {
                // constant term
                let mut poly_X = vec![(0, F::one())];
                //remaining inputs
                // TODO(sragss / arasuarun): I believe this is supposed to be io -- which is empty??
                // poly_X.extend(
                //     (0..self.witness_segment_commitments.len())
                //         .map(|i| (i + 1, self.witness_segment_commitments[i]))
                //         .collect::<Vec<(usize, F)>>(),
                // );
                SparsePolynomial::new(usize::try_from(key.num_vars_total.ilog2()).unwrap(), poly_X)
                    .evaluate(&inner_sumcheck_r[1..])
            };

            // evaluate the segments of W
            let r_y_witness = &inner_sumcheck_r[1..n_prefix]; // skip the first as it's used to separate the inputs and the witness
            let eval_W = (0..N_SEGMENTS)
                .map(|i| {
                    let bin = format!("{:0width$b}", i, width = n_prefix - 1); // write i in binary using N_PREFIX bits

                    let product = bin.chars().enumerate().fold(F::one(), |acc, (j, bit)| {
                        acc * if bit == '0' {
                            F::one() - r_y_witness[j]
                        } else {
                            r_y_witness[j]
                        }
                    });

                    product * self.claimed_witnesss_evals[i]
                })
                .sum::<F>();

            (F::one() - inner_sumcheck_r[0]) * eval_W + inner_sumcheck_r[0] * eval_X
        };

        // compute evaluations of R1CS matrices
        let multi_evaluate_uniform =
            |M_vec: &[&[(usize, usize, F)]], r_x: &[F], r_y: &[F], num_steps: usize| -> Vec<F> {
                let evaluate_with_table_uniform =
                    |M: &[(usize, usize, F)], T_x: &[F], T_y: &[F], num_steps: usize| -> F {
                        (0..M.len())
                            .into_par_iter()
                            .map(|i| {
                                let (row, col, val) = M[i];
                                (0..num_steps)
                                    .into_par_iter()
                                    .map(|j| {
                                        let row = row * num_steps + j;
                                        let col = if col != key.shape_single_step.num_vars {
                                            col * num_steps + j
                                        } else {
                                            key.num_vars_total
                                        };
                                        let val = val * T_x[row] * T_y[col];
                                        val
                                    })
                                    .sum::<F>()
                            })
                            .sum()
                    };

                let (T_x, T_y) = rayon::join(
                    || EqPolynomial::new(r_x.to_vec()).evals(),
                    || EqPolynomial::new(r_y.to_vec()).evals(),
                );

                (0..M_vec.len())
                    .into_par_iter()
                    .map(|i| evaluate_with_table_uniform(M_vec[i], &T_x, &T_y, num_steps))
                    .collect()
            };

        let evals = multi_evaluate_uniform(
            &[
                &key.shape_single_step.A,
                &key.shape_single_step.B,
                &key.shape_single_step.C,
            ],
            &r_x,
            &inner_sumcheck_r,
            key.num_steps,
        );

        let left_expected = evals[0]
            + r_inner_sumcheck_RLC * evals[1]
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * evals[2];
        let right_expected = eval_Z;
        let claim_inner_final_expected = left_expected * right_expected;
        if claim_inner_final != claim_inner_final_expected {
            // DEDUPE(arasuarun): add
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let r_y_point = &inner_sumcheck_r[n_prefix..];
        self.opening_proof
            .verify(
                &generators,
                &r_y_point,
                &self.claimed_witnesss_evals,
                &witness_segment_commitments,
                transcript,
            )
            .map_err(|_| SpartanError::InvalidHyraxProof)?;

        Ok(())
    }
}

struct SparsePolynomial<F: PrimeField> {
    num_vars: usize,
    Z: Vec<(usize, F)>,
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
    pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
        SparsePolynomial { num_vars, Z }
    }

    /// Computes the $\tilde{eq}$ extension polynomial.
    /// return 1 when a == r, otherwise return 0.
    fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
        assert_eq!(a.len(), r.len());
        let mut chi_i = Scalar::ONE;
        for j in 0..r.len() {
            if a[j] {
                chi_i *= r[j];
            } else {
                chi_i *= Scalar::ONE - r[j];
            }
        }
        chi_i
    }

    // Takes O(n log n)
    pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
        assert_eq!(self.num_vars, r.len());

        (0..self.Z.len())
            .into_par_iter()
            .map(|i| {
                let bits = get_bits(self.Z[0].0, r.len());
                SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
            })
            .sum()
    }
}

/// Returns the `num_bits` from n in a canonical order
fn get_bits(operand: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits)
        .map(|shift_amount| ((operand & (1 << (num_bits - shift_amount - 1))) > 0))
        .collect::<Vec<bool>>()
}
