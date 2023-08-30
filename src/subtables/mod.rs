use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use merlin::Transcript;

use crate::{
  lasso::{densified::DensifiedRepresentation, memory_checking::GrandProducts},
  poly::dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentGens, PolyEvalProof},
  poly::eq_poly::EqPolynomial,
  utils::errors::ProofVerifyError,
  utils::math::Math,
  utils::random::RandomTape,
  utils::transcript::{AppendToTranscript, ProofTranscript}, jolt::jolt_strategy::JoltStrategy,
};

#[cfg(feature = "multicore")]
use rayon::prelude::*;

pub mod subtable_strategy;
// pub mod and;
// pub mod lt;
// pub mod or;
// pub mod range_check;
// pub mod xor;

#[cfg(test)]
pub mod test;

pub struct Subtables<F: PrimeField, S: JoltStrategy<F>>
{
  pub subtable_entries: Vec<Vec<F>>,
  pub lookup_polys: Vec<DensePolynomial<F>>,
  pub combined_poly: DensePolynomial<F>,
  _marker: PhantomData<S>
}

/// Stores the non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials, enables combination and commitment.
impl<F: PrimeField, S: JoltStrategy<F>> Subtables<F, S> {
  /// Create new Subtables
  /// - `nz`: non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials
  /// - `s`: number of lookups
  pub fn new(nz: &Vec<Vec<usize>>, s: usize) -> Self {
    nz.iter().for_each(|nz_dim| assert_eq!(nz_dim.len(), s));
    let subtable_entries = S::materialize_subtables();
    let lookup_polys: Vec<DensePolynomial<F>> = S::to_lookup_polys(&subtable_entries, nz, s);
    let combined_poly = DensePolynomial::merge(&lookup_polys);

    Subtables {
      subtable_entries,
      lookup_polys,
      combined_poly,
      _marker: PhantomData
    }
  }

  /// Converts subtables T_1, ..., T_{\alpha} and densified multilinear polynomial
  /// into grand products for memory-checking.
  #[tracing::instrument(skip_all, name = "Subtables.to_grand_products")]
  pub fn to_grand_products(
    &self,
    dense: &DensifiedRepresentation<F, S>,
    r_mem_check: &(F, F),
  ) -> Vec<GrandProducts<F>> {
    (0..S::num_memories()).map(|i| {
      let subtable = &self.subtable_entries[S::memory_to_subtable_index(i)];
      let j = S::memory_to_dimension_index(i);
      GrandProducts::new(
        subtable,
        &dense.dim[j],
        &dense.dim_usize[j],
        &dense.read[j],
        &dense.r#final[j],
        r_mem_check,
      )
    }).collect()
  }

  #[tracing::instrument(skip_all, name = "Subtables.commit")]
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &PolyCommitmentGens<G>,
  ) -> CombinedTableCommitment<G> {
    let (comm_ops_val, _blinds) = self.combined_poly.commit(gens, None);
    CombinedTableCommitment { comm_ops_val }
  }

  #[tracing::instrument(skip_all, name = "Subtables.compute_sumcheck_claim")]
  pub fn compute_sumcheck_claim(&self, eq: &EqPolynomial<F>) -> F {
    let g_operands = self.lookup_polys.clone();
    let hypercube_size = g_operands[0].len();
    g_operands
      .iter()
      .for_each(|operand| assert_eq!(operand.len(), hypercube_size));
    
    let eq_evals = eq.evals();
    
    #[cfg(feature = "multicore")]
    let claim = (0..hypercube_size)
      .into_par_iter()
      .map(|k| {
        let g_operands: Vec<F> = (0..S::num_memories()).map(|j| g_operands[j][k]).collect();
        // eq * g(T_1[k], ..., T_\alpha[k])
        eq_evals[k] * S::combine_lookups(&g_operands)
      })
      .sum();

    #[cfg(not(feature = "multicore"))]
    let claim = (0..hypercube_size)
      .map(|k| {
        let g_operands: Vec<F> = (0..S::num_memories())(|j| g_operands[j][k]).collect();
        // eq * g(T_1[k], ..., T_\alpha[k])
        eq_evals[k] * S::combine_lookups(&g_operands)
      })
      .sum();

    claim
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableCommitment<G: CurveGroup> {
  comm_ops_val: PolyCommitment<G>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableEvalProof<G: CurveGroup, S: JoltStrategy<G::ScalarField>> {
  proof_table_eval: PolyEvalProof<G>,
  _marker: PhantomData<S>
}

impl<G: CurveGroup, S: JoltStrategy<G::ScalarField>> CombinedTableEvalProof<G, S> {
  fn prove_single(
    joint_poly: &DensePolynomial<G::ScalarField>,
    r: &[G::ScalarField],
    evals: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> PolyEvalProof<G> {
    assert_eq!(
      joint_poly.get_num_vars(),
      r.len() + evals.len().log_2() 
    );

    // append the claimed evaluations to transcript
    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", evals);

    // n-to-1 reduction
    let (r_joint, eval_joint) = {
      let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
        transcript,
        b"challenge_combine_n_to_one",
        evals.len().log_2(),
      );

      let mut poly_evals = DensePolynomial::new(evals.to_vec());
      for i in (0..challenges.len()).rev() {
        poly_evals.bound_poly_var_bot(&challenges[i]);
      }
      assert_eq!(poly_evals.len(), 1);
      let joint_claim_eval = poly_evals[0];
      let mut r_joint = challenges;
      r_joint.extend(r);

      debug_assert_eq!(joint_poly.evaluate(&r_joint), joint_claim_eval);
      (r_joint, joint_claim_eval)
    };
    // decommit the joint polynomial at r_joint
    <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"joint_claim_eval", &eval_joint);

    let (proof_table_eval, _comm_table_eval) = PolyEvalProof::prove(
      joint_poly,
      None,
      &r_joint,
      &eval_joint,
      None,
      gens,
      transcript,
      random_tape,
    );

    proof_table_eval
  }

  /// evalues both polynomials at r and produces a joint proof of opening
  #[tracing::instrument(skip_all, name = "CombinedEval.prove")]
  pub fn prove(
    combined_poly: &DensePolynomial<G::ScalarField>,
    eval_ops_val_vec: &[G::ScalarField],
    r: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      CombinedTableEvalProof::<G, S>::protocol_name(),
    );

    let evals = {
      let mut evals: Vec<G::ScalarField> = eval_ops_val_vec.to_vec();
      evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());
      evals.to_vec()
    };
    let proof_table_eval = CombinedTableEvalProof::<G, S>::prove_single(
      combined_poly,
      r,
      &evals,
      gens,
      transcript,
      random_tape,
    );

    CombinedTableEvalProof { proof_table_eval, _marker: PhantomData }
  }

  fn verify_single(
    proof: &PolyEvalProof<G>,
    comm: &PolyCommitment<G>,
    r: &[G::ScalarField],
    evals: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // append the claimed evaluations to transcript
    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", evals);

    // n-to-1 reduction
    let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_n_to_one",
      evals.len().log_2(),
    );
    let mut poly_evals = DensePolynomial::new(evals.to_vec());
    for i in (0..challenges.len()).rev() {
      poly_evals.bound_poly_var_bot(&challenges[i]);
    }
    assert_eq!(poly_evals.len(), 1);
    let joint_claim_eval = poly_evals[0];
    let mut r_joint = challenges;
    r_joint.extend(r);

    // decommit the joint polynomial at r_joint
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval",
      &joint_claim_eval,
    );

    proof.verify_plain(gens, transcript, &r_joint, &joint_claim_eval, comm)
  }

  // verify evaluations of both polynomials at r
  pub fn verify(
    &self,
    r: &[G::ScalarField],
    evals: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    comm: &CombinedTableCommitment<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      CombinedTableEvalProof::<G, S>::protocol_name(),
    );
    let mut evals = evals.to_owned();
    evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());

    CombinedTableEvalProof::<G, S>::verify_single(
      &self.proof_table_eval,
      &comm.comm_ops_val,
      r,
      &evals,
      gens,
      transcript,
    )
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso CombinedTableEvalProof"
  }
}

impl<G: CurveGroup> AppendToTranscript<G> for CombinedTableCommitment<G> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, label: &'static [u8], transcript: &mut T) {
    transcript.append_message(
      b"subtable_evals_commitment",
      b"begin_subtable_evals_commitment",
    );
    self.comm_ops_val.append_to_transcript(label, transcript);
    transcript.append_message(
      b"subtable_evals_commitment",
      b"end_subtable_evals_commitment",
    );
  }
}