use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rayon::prelude::*;
use thiserror::Error;
use crate::poly::hyrax::BatchedHyraxOpeningProof;
use crate::utils::transcript::ProofTranscript;
use crate::utils::transcript::AppendToTranscript;

use super::r1cs_shape::R1CSShape;
use crate::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        hyrax::{HyraxCommitment, HyraxGenerators},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
};

pub struct UniformSpartanKey<F: PrimeField> {
    shape_single_step: R1CSShape<F>, // Single step shape
    shape_full: R1CSShape<F>,        // Single step shape
    num_cons_total: usize,           // Number of constraints
    num_vars_total: usize,           // Number of variables
    num_steps: usize,                // Number of steps
    vk_digest: F,                    // digest of the verifier's key
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
pub struct UniformSpartanProof<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    witness_segment_commitments: Vec<HyraxCommitment<1, G>>,
    outer_sumcheck_proof: SumcheckInstanceProof<F>,
    outer_sumcheck_claims: (F, F, F),
    inner_sumcheck_proof: SumcheckInstanceProof<F>,
    eval_arg: Vec<F>, // TODO(arasuarun): better name
    claimed_witnesss_evals: Vec<F>,
    opening_proof: BatchedHyraxOpeningProof<1, G>
}

pub struct PrecommittedR1CSInstance<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    comm_W: Vec<HyraxCommitment<1, G>>,
    X: Vec<F>,
}

// Trait which will kick out a small and big R1CS shape
pub trait UniformShapeBuilder<F: PrimeField> {
    fn single_step_shape(&self) -> R1CSShape<F>;
    fn full_shape(&self, N: usize, single_step_shape: &R1CSShape<F>) -> R1CSShape<F>;
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,
    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,
    /// returned if the supplied witness is not of the right length
    #[error("InvalidWitnessLength")]
    InvalidWitnessLength,
    /// returned when an invalid Hyrax proof is provided
    #[error("InvalidHyraxProof")]
    InvalidHyraxProof,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> UniformSpartanProof<F, G> {
    #[tracing::instrument(skip_all, name = "SNARK::setup_precommitted")]
    fn setup_precommitted<C: UniformShapeBuilder<F>>(
        circuit: C,
        num_steps: usize,
        generators: HyraxGenerators<1, G>,
    ) -> Result<UniformSpartanKey<F>, SpartanError> {
        let shape_single_step = circuit.single_step_shape();
        let shape_full = circuit.full_shape(num_steps, &shape_single_step);

        let num_constraints_total = shape_single_step.num_cons * num_steps;
        let num_aux_total = shape_single_step.num_vars * num_steps;

        let pad_num_constraints = num_constraints_total.next_power_of_two();
        let pad_num_aux = num_aux_total.next_power_of_two();

        // TODO(sragss / arasuarun): Verifier key digest
        let vk_digest = F::one();

        let key = UniformSpartanKey {
            shape_single_step,
            shape_full,
            num_cons_total: pad_num_constraints,
            num_vars_total: pad_num_aux,
            num_steps,
            vk_digest,
        };
        Ok(key)
    }

    /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
    #[tracing::instrument(skip_all, name = "Spartan2::UPSnark::prove")]
    fn prove_precommitted(
        prover_generators: HyraxGenerators<1, G>,
        key: &UniformSpartanKey<F>,
        w_segments: Vec<Vec<F>>,
        witness_commitments: Vec<HyraxCommitment<1, G>>,
        transcript: &mut Transcript,
    ) -> Result<Self, SpartanError> {
        let witness_segments = w_segments;

        // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
        <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"vk", &key.vk_digest);

        let span_u = tracing::span!(tracing::Level::INFO, "absorb_u");
        let _guard_u = span_u.enter();
        witness_commitments.iter().for_each(|commitment| {
            commitment.append_to_transcript(b"U", transcript);
        });
        drop(_guard_u);

        // TODO(sragss/arasuarun/moodlezoup): We can do this by reference in prove_quad_batched_unrolled.
        let span = tracing::span!(tracing::Level::INFO, "witness_batching");
        let _guard = span.enter();
        let mut witness = Vec::with_capacity(witness_segments.len() * witness_segments[0].len());
        witness_segments.iter().for_each(|segment| {
            witness.par_extend(segment);
        });
        drop(_guard);

        let span = tracing::span!(tracing::Level::INFO, "witness_resizing");
        let _guard = span.enter();
        witness.resize(key.num_vars_total, F::zero());
        drop(_guard);

        let (num_rounds_x, num_rounds_y) = (
            usize::try_from(key.num_cons_total.ilog2()).unwrap(),
            (usize::try_from(key.num_vars_total.ilog2()).unwrap() + 1),
        );

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"t"))
            .collect::<Vec<F>>();

        let mut poly_tau = DensePolynomial::new(EqPolynomial::new(tau).evals());
        // poly_Az is the polynomial extended from the vector Az
        let (mut poly_Az, mut poly_Bz, mut poly_Cz) = {
            let (poly_Az, poly_Bz, poly_Cz) =
                key.shape_single_step.multiply_vec_uniform(&witness, &vec![], key.num_steps)?; // TODO(sragss): witness_commitments param is wrong [W, 1, X] -- I think it's just IO??
            (
                DensePolynomial::new(poly_Az),
                DensePolynomial::new(poly_Bz),
                DensePolynomial::new(poly_Cz),
            )
        };

        let comb_func_outer = |poly_A_comp: &F, poly_B_comp: &F, poly_C_comp: &F, poly_D_comp: &F| -> F {
                // Below is an optimized form of: *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp)
                if poly_B_comp.is_zero() || poly_C_comp.is_zero() {
                    if poly_D_comp.is_zero() {
                        F::zero()
                    } else {
                        *poly_A_comp * (-(*poly_D_comp))
                    }
                } else {
                    *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp)
                }
            };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) = SumcheckInstanceProof::prove_cubic_with_additive_term::<G, _>(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut poly_tau,
                &mut poly_Az,
                &mut poly_Bz,
                &mut poly_Cz,
                comb_func_outer,
                transcript,
            );
        std::thread::spawn(|| drop(poly_Az));
        std::thread::spawn(|| drop(poly_Bz));
        std::thread::spawn(|| drop(poly_Cz));
        std::thread::spawn(|| drop(poly_tau));

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) =
            (outer_sumcheck_claims[1], outer_sumcheck_claims[2], outer_sumcheck_claims[3]);
        <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claims_outer", &[claim_Az, claim_Bz, claim_Cz].as_slice());

        // inner sum-check
        let r: F = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"r");
        let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

        let span = tracing::span!(tracing::Level::TRACE, "poly_ABC");
        let _enter = span.enter();

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let poly_ABC = {
            let num_steps_bits = key.num_steps.trailing_zeros();
            let (rx_con, rx_ts) = outer_sumcheck_r.split_at(outer_sumcheck_r.len() - num_steps_bits as usize);
            let (eq_rx_con, eq_rx_ts) = rayon::join(
                || EqPolynomial::new(rx_con.to_vec()).evals(),
                || EqPolynomial::new(rx_ts.to_vec()).evals(),
            );

            let n_steps = key.num_steps;

            // With uniformity, each entry of the RLC of A, B, C can be expressed using
            // the RLC of the small_A, small_B, small_C matrices.

            // 1. Evaluate \tilde smallM(r_x, y) for all y
            let compute_eval_table_sparse_single = |small_M: &Vec<(usize, usize, F)>| -> Vec<F> {
                let mut small_M_evals = vec![F::zero(); key.shape_full.num_vars + 1];
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

            let r_sq = r * r;
            let small_RLC_evals = (0..small_A_evals.len())
                .into_par_iter()
                .map(|i| small_A_evals[i] + small_B_evals[i] * r + small_C_evals[i] * r_sq)
                .collect::<Vec<F>>();

            // 2. Handles all entries but the last one with the constant 1 variable
            let mut RLC_evals: Vec<F> = (0..key.num_vars_total)
                .into_par_iter()
                .map(|col| eq_rx_ts[col % n_steps] * small_RLC_evals[col / n_steps])
                .collect();
            let next_pow_2 = 2 * key.num_vars_total;
            RLC_evals.resize(next_pow_2, F::zero());

            // 3. Handles the constant 1 variable
            let compute_eval_constant_column = |small_M: &Vec<(usize, usize, F)>| -> F {
                let constant_sum: F = small_M.iter()
              .filter(|(_, col, _)| *col == key.shape_single_step.num_vars)   // expecting ~1
              .map(|(row, _, val)| {
                  let eq_sum = (0..n_steps).into_par_iter().map(|t| eq_rx_ts[t]).sum::<F>();
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

            RLC_evals[key.num_vars_total] =
                constant_term_A + r * constant_term_B + r * r * constant_term_C;

            RLC_evals
        };
        drop(_enter);
        drop(span);

        let comb_func = |poly_A_comp: &F, poly_B_comp: &F| -> F {
            if *poly_A_comp == F::zero() || *poly_B_comp == F::zero() {
                F::zero()
            } else {
                *poly_A_comp * *poly_B_comp
            }
        };
        let mut poly_ABC = DensePolynomial::new(poly_ABC);
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) = SumcheckInstanceProof::prove_quad_unrolled::<G, _>(
            &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
            num_rounds_y,
            &mut poly_ABC, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
            &witness,
            &vec![], // TODO(sragss): I believe this is IO??
            comb_func,
            transcript,
        );
        std::thread::spawn(|| drop(poly_ABC));

        // The number of prefix bits needed to identify a segment within the witness vector
        // assuming that num_vars_total is a power of 2 and each segment has length num_steps, which is also a power of 2.
        // The +1 is the first element used to separate the inputs and the witness.
        let n_prefix = (key.num_vars_total.trailing_zeros() as usize
            - key.num_steps.trailing_zeros() as usize)
            + 1;
        let r_y_point = &inner_sumcheck_r[n_prefix..];

        // Evaluate each segment on r_y_point
        let span = tracing::span!(tracing::Level::TRACE, "evaluate_segments");
        let _enter = span.enter();
        let witness_segment_polys: Vec<DensePolynomial<F>> = witness_segments.into_iter().map(|segment| DensePolynomial::new(segment)).collect();
        let chi = EqPolynomial::new(r_y_point.to_owned()).evals();
        let witness_evals: Vec<F> = witness_segment_polys.iter().map(|segment| segment.evaluate_at_chi_low_optimized(&chi)).collect();
        drop(_enter);

        // now batch these together
        let c: F = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");
        // todo!("change batching strategy");
        // let w: PolyEvalWitness<G> = PolyEvalWitness::batch(&w.W.as_slice().iter().map(|v| v.as_ref()).collect::<Vec<_>>(), &c);
        // let u: PolyEvalInstance<G> = PolyEvalInstance::batch(&comm_vec, &r_y_point, &witness_evals, &c);

        // TODO(sragss/arasuarun): switch to hyrax
        let witness_segment_polys_ref: Vec<&DensePolynomial<F>> = witness_segment_polys.iter().map(|poly_ref| poly_ref).collect();
        let opening_proof = BatchedHyraxOpeningProof::prove(&witness_segment_polys_ref, &r_y_point, &witness_evals, transcript);

        // todo!("finish the stuff");

        // TODO(sragss): Compress commitments?

      let outer_sumcheck_claims = (outer_sumcheck_claims[0], outer_sumcheck_claims[1], outer_sumcheck_claims[2]);
      Ok(UniformSpartanProof {
        witness_segment_commitments: witness_commitments,
        outer_sumcheck_proof,
        outer_sumcheck_claims,
        inner_sumcheck_proof,
        eval_arg: vec![],
        claimed_witnesss_evals: witness_evals,
        opening_proof
      })
    }

    /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
    #[tracing::instrument(skip_all, name = "SNARK::verify")]
    fn verify_precommitted(
        &self,
        key: &UniformSpartanKey<F>,
        io: &[F],
        generators: HyraxGenerators<1, G>,
        transcript: &mut Transcript
    ) -> Result<(), SpartanError> {
        let N_SEGMENTS = self.witness_segment_commitments.len();

        // construct an instance using the provided commitment to the witness and IO
        // let comm_W_vec = self.witness_segment_commitments.iter()
        //   .map(|c| Commitment::<G>::decompress(c).unwrap())
        //   .collect::<Vec::<<<G as Group>::CE as CommitmentEngineTrait<G>>::Commitment>>();

        assert_eq!(io.len(), 0);
        // let witness_segment_commitments = PrecommittedR1CSInstance::new(&hollow_S, comm_W_vec.clone(), io)?;

        // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
        <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"vk", &key.vk_digest);
        self.witness_segment_commitments.iter().for_each(|commitment| {
          commitment.append_to_transcript(b"U", transcript)
        });

        let (num_rounds_x, num_rounds_y) = (
            usize::try_from(key.num_cons_total.ilog2()).unwrap(),
            (usize::try_from(key.num_vars_total.ilog2()).unwrap() + 1),
        );

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"t"))
            .collect::<Vec<F>>();

        let (claim_outer_final, r_x) =
            self.outer_sumcheck_proof
                .verify::<G, Transcript>(F::zero(), num_rounds_x, 3, transcript)
                .map_err(|_| SpartanError::InvalidSumcheckProof)?;

        // verify claim_outer_final
        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidSumcheckProof);
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
        let r: F = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"r");
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + r * self.outer_sumcheck_claims.1
            + r * r * self.outer_sumcheck_claims.2;

        let (claim_inner_final, r_y) = self.inner_sumcheck_proof.verify::<G, Transcript>(
            claim_inner_joint,
            num_rounds_y,
            2,
            transcript,
        ).map_err(|_| SpartanError::InvalidSumcheckProof)?;

        // verify claim_inner_final
        // this should be log (num segments)
        let n_prefix = (key.num_vars_total.trailing_zeros() as usize
            - key.num_steps.trailing_zeros() as usize)
            + 1;

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
                    .evaluate(&r_y[1..])
            };

            // evaluate the segments of W
            let r_y_witness = &r_y[1..n_prefix]; // skip the first as it's used to separate the inputs and the witness
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

            (F::one() - r_y[0]) * eval_W + r_y[0] * eval_X
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
            &r_y,
            key.num_steps,
        );

        let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
        if claim_inner_final != claim_inner_final_expected {
            // DEDUPE(arasuarun): add
            return Err(SpartanError::InvalidSumcheckProof);
        }

        // we now combine evaluation claims at the same point rz into one
        // let comm_vec = self.witness_segment_commitments;
        // let eval_vec = &self.eval_W;

        let r_y_point = &r_y[n_prefix..];
        let c: F = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");
        // let u: PolyEvalInstance<G> = PolyEvalInstance::batch(&comm_vec, &r_y_point, &eval_vec, &c);
        let hyrax_commitment_refs: Vec<&HyraxCommitment<1, G>> = self.witness_segment_commitments.iter().map(|commit_ref| commit_ref).collect(); // TODO(sragss): Fix
        self.opening_proof.verify(&generators, &r_y_point, &self.claimed_witnesss_evals, &hyrax_commitment_refs, transcript)
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