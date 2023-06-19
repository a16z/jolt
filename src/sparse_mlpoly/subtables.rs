use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use merlin::Transcript;

use crate::{
  dense_mlpoly::{
    DensePolynomial, EqPolynomial, PolyCommitment, PolyCommitmentGens, PolyEvalProof,
  },
  errors::ProofVerifyError,
  math::Math,
  random::RandomTape,
  transcript::{AppendToTranscript, ProofTranscript},
};

use super::{densified::DensifiedRepresentation, memory_checking::GrandProducts};

pub trait SubtableStrategy<F: PrimeField, const C: usize, const ALPHA: usize> {
  /// Materialize subtables indexed [1, ..., \alpha]
  /// Note: Some materializations will not use the parameter r.
  ///
  /// Params
  /// - `m`: size of subtable / number of evaluations to materialize
  /// - `r`: point at which to materialize the table (potentially unused)
  fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> [Vec<F>; ALPHA];

  /// Converts subtables T_1, ..., T_{\alpha} and lookup indices nz_1, ..., nz_c
  /// into log(m)-variate "lookup polynomials" E_1, ..., E_{\alpha}.
  fn to_lookup_polys(
    subtable_entries: &[Vec<F>; ALPHA],
    nz: &[Vec<usize>; C],
    s: usize,
  ) -> [DensePolynomial<F>; ALPHA];

  /// Converts subtables T_1, ..., T_{\alpha} and densified multilinear polynomial
  /// into grand products for memory-checking.
  fn to_grand_products(
    subtable_entries: &[Vec<F>; ALPHA],
    dense: &DensifiedRepresentation<F, C>,
    r_mem_check: &(F, F),
  ) -> [GrandProducts<F>; ALPHA];

  /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
  fn combine_lookups(vals: &[F; ALPHA]) -> F;

  /// The total degree of `g`, i.e. considering `combine_lookups` as a log(m)-variate polynomial.
  /// Determines the number of evaluation points in each sumcheck round.
  fn sumcheck_poly_degree() -> usize;
}

pub enum EqSubtableStrategy {}

impl<F: PrimeField, const C: usize, const ALPHA: usize> SubtableStrategy<F, C, ALPHA>
  for EqSubtableStrategy
{
  fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> [Vec<F>; ALPHA] {
    std::array::from_fn(|i| {
      let eq_evals = EqPolynomial::new(r[i].clone()).evals();
      assert_eq!(eq_evals.len(), m);
      eq_evals
    })
  }

  fn to_lookup_polys(
    subtable_entries: &[Vec<F>; ALPHA],
    nz: &[Vec<usize>; C],
    s: usize,
  ) -> [DensePolynomial<F>; ALPHA] {
    std::array::from_fn(|i| {
      let mut subtable_lookups: Vec<F> = Vec::with_capacity(s);
      for j in 0..s {
        subtable_lookups.push(subtable_entries[i][nz[i][j]]);
      }
      DensePolynomial::new(subtable_lookups)
    })
  }

  fn to_grand_products(
    subtable_entries: &[Vec<F>; ALPHA],
    dense: &DensifiedRepresentation<F, C>,
    r_mem_check: &(F, F),
  ) -> [GrandProducts<F>; ALPHA] {
    std::array::from_fn(|i| {
      GrandProducts::new(
        &subtable_entries[i],
        &dense.dim[i],
        &dense.dim_usize[i],
        &dense.read[i],
        &dense.r#final[i],
        r_mem_check,
      )
    })
  }

  fn combine_lookups(vals: &[F; ALPHA]) -> F {
    vals.iter().product()
  }

  fn sumcheck_poly_degree() -> usize {
    ALPHA // TODO(moodlezoup) +1 for \tilde{eq} poly?
  }
}

pub struct Subtables<F: PrimeField, const C: usize, const ALPHA: usize, S>
where
  S: SubtableStrategy<F, C, ALPHA>,
{
  pub subtable_entries: [Vec<F>; ALPHA],
  pub lookup_polys: [DensePolynomial<F>; ALPHA],
  pub combined_poly: DensePolynomial<F>,
  strategy: PhantomData<S>,
}

/// Stores the non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials, enables combination and commitment.
impl<F: PrimeField, const C: usize, const ALPHA: usize, S> Subtables<F, C, ALPHA, S>
where
  S: SubtableStrategy<F, C, ALPHA>,
{
  /// Create new SubtableEvaluations
  /// - `evaluations`: non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials
  pub fn new(nz: &[Vec<usize>; C], r: &[Vec<F>; C], m: usize, s: usize) -> Self {
    nz.iter().for_each(|nz_dim| assert_eq!(nz_dim.len(), s));
    let subtable_entries: [Vec<F>; ALPHA] = S::materialize_subtables(m, r);
    let lookup_polys: [DensePolynomial<F>; ALPHA] =
      S::to_lookup_polys(&subtable_entries, nz, s);
    let combined_poly = DensePolynomial::merge(&lookup_polys);

    Subtables {
      subtable_entries,
      lookup_polys,
      combined_poly,
      strategy: PhantomData,
    }
  }

  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &PolyCommitmentGens<G>,
  ) -> CombinedTableCommitment<G> {
    let (comm_ops_val, _blinds) = self.combined_poly.commit(gens, None);
    CombinedTableCommitment { comm_ops_val }
  }

  pub fn compute_sumcheck_claim(&self) -> F {
    let g_operands = self.lookup_polys.clone();
    let hypercube_size = g_operands[0].len();
    g_operands
      .iter()
      .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

    (0..hypercube_size)
      .map(|k| {
        let g_operands: [F; ALPHA] = std::array::from_fn(|j| g_operands[j][k]);
        S::combine_lookups(&g_operands)
        // TODO(moodlezoup): \tilde{eq}(r, k)
      })
      .sum()
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableCommitment<G: CurveGroup> {
  comm_ops_val: PolyCommitment<G>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableEvalProof<G: CurveGroup, const C: usize> {
  proof_table_eval: PolyEvalProof<G>,
}

impl<G: CurveGroup, const C: usize> CombinedTableEvalProof<G, C> {
  fn protocol_name() -> &'static [u8] {
    b"Surge CombinedTableEvalProof"
  }

  fn prove_single(
    joint_poly: &DensePolynomial<G::ScalarField>,
    r: &[G::ScalarField],
    evals: Vec<G::ScalarField>,
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> PolyEvalProof<G> {
    assert_eq!(
      joint_poly.get_num_vars(),
      r.len() + evals.len().log_2() as usize
    );

    // append the claimed evaluations to transcript
    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

    // n-to-1 reduction
    let (r_joint, eval_joint) = {
      let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
        transcript,
        b"challenge_combine_n_to_one",
        evals.len().log_2() as usize,
      );

      let mut poly_evals = DensePolynomial::new(evals);
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

  // evalues both polynomials at r and produces a joint proof of opening
  pub fn prove(
    combined_poly: &DensePolynomial<G::ScalarField>,
    eval_ops_val_vec: &Vec<G::ScalarField>,
    r: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      CombinedTableEvalProof::<G, C>::protocol_name(),
    );

    let evals = {
      let mut evals = eval_ops_val_vec.clone();
      evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());
      evals.to_vec()
    };
    let proof_table_eval = CombinedTableEvalProof::<G, C>::prove_single(
      &combined_poly,
      r,
      evals,
      gens,
      transcript,
      random_tape,
    );

    CombinedTableEvalProof { proof_table_eval }
  }

  fn verify_single(
    proof: &PolyEvalProof<G>,
    comm: &PolyCommitment<G>,
    r: &[G::ScalarField],
    evals: Vec<G::ScalarField>,
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // append the claimed evaluations to transcript
    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

    // n-to-1 reduction
    let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_n_to_one",
      evals.len().log_2() as usize,
    );
    let mut poly_evals = DensePolynomial::new(evals);
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
      CombinedTableEvalProof::<G, C>::protocol_name(),
    );
    let mut evals = evals.to_owned();
    evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());

    CombinedTableEvalProof::<G, C>::verify_single(
      &self.proof_table_eval,
      &comm.comm_ops_val,
      r,
      evals,
      gens,
      transcript,
    )
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

// #[cfg(test)]
// mod test {
//   use super::*;

//   use crate::dense_mlpoly::EqPolynomial;
//   use crate::utils::index_to_field_bitvector;
//   use ark_bls12_381::Fr;

//   #[test]
//   fn forms_valid_merged_dense_poly() {
//     // Pass in the eq evaluations over log_m boolean variables and log_m fixed variables r
//     let log_m = 2;
//     const c: usize = 2;

//     let r_x: Vec<Fr> = vec![Fr::from(3), Fr::from(4)];
//     let r_y: Vec<Fr> = vec![Fr::from(5), Fr::from(6)];
//     let eq_evals_x: Vec<Fr> = EqPolynomial::new(r_x.clone()).evals();
//     let eq_evals_y: Vec<Fr> = EqPolynomial::new(r_y.clone()).evals();
//     assert_eq!(eq_evals_x.len(), log_m.pow2());
//     assert_eq!(eq_evals_y.len(), log_m.pow2());

//     let eq_evals_x_poly: DensePolynomial<Fr> = DensePolynomial::new(eq_evals_x);
//     let eq_evals_y_poly: DensePolynomial<Fr> = DensePolynomial::new(eq_evals_y);

//     // You can think of the concatenation as adding a log(c) bits to eq to specify the dimension
//     let eq_index_bits = 3;
//     // eq(x,y) = prod{x_i * y_i + (1-x_i) * (1-y_i)}
//     // eq(0, 0, 3, 4) = (0 * 3 + (1-0) * (1-3)) * (0 * 4 + (1-0) * (1-4)) = (-2)(-3) = 6
//     // eq(0, 1, 3, 4) = (0 * 3 + (1-0) * (1-3)) * (1 * 4 + (1-1) * (1-4)) = (-2)(4) = -8
//     // eq(1, 0, 3, 4) = (1 * 3 + (1-1) * (1-3)) * (0 * 4 + (1-0) * (1-4)) = (3)(-3) = -9
//     // eq(1, 1, 3, 4) = (1 * 3 + (1-1) * (1-3)) * (1 * 4 + (1-1) * (1-4)) = (3)(4) = 12
//     // eq(0, 0, 5, 6) = (0 * 5 + (1-0) * (1-5)) * (0 * 6 + (1-0) + (1-6)) = (-4)(-5) = 20
//     // eq(0, 1, 5, 6) = (0 * 5 + (1-0) * (1-5)) * (1 * 6 + (1-1) + (1-6)) = (-4)(6) = -24
//     // eq(1, 0, 5, 6) = (1 * 5 + (1-1) * (1-5)) * (0 * 6 + (1-0) + (1-6)) = (5)(-5) = -25
//     // eq(1, 1, 5, 6) = (1 * 5 + (1-1) * (1-5)) * (1 * 6 + (1-1) + (1-6)) = (5)(6) = 30

//     let subtable_evals: Subtables<Fr, c, c, EqSubtableStrategy> =
//       Subtables::new([eq_evals_x_poly, eq_evals_y_poly], &[r_x, r_y], 1 << log_m, );

//     for (x, eval) in vec![
//       (0, 6),
//       (1, -8),
//       (2, -9),
//       (3, 12),
//       (4, 20),
//       (5, -24),
//       (6, -25),
//       (7, 30),
//     ] {
//       assert_eq!(
//         subtable_evals
//           .combined_poly
//           .evaluate(&index_to_field_bitvector(x, eq_index_bits)),
//         Fr::from(eval)
//       );
//     }
//   }
// }
