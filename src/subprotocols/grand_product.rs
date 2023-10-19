#![allow(dead_code)]

use super::sumcheck::SumcheckInstanceProof;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::math::Math;
use crate::utils::transcript::ProofTranscript;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use merlin::Transcript;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

/// Contains grand product circuits to evaluate multi-set checks on memories.
/// Evaluating each circuit is equivalent to computing the hash/fingerprint
/// H_{\tau, \gamma} of the corresponding set.
#[derive(Debug)]
pub struct GrandProducts<F> {
  /// Corresponds to the Init_{row/col} hash in the Spartan paper.
  pub init: GrandProductCircuit<F>,
  /// Corresponds to the RS_{row/col} hash in the Spartan paper.
  pub read: GrandProductCircuit<F>,
  /// Corresponds to the WS_{row/col} hash in the Spartan paper.
  pub write: GrandProductCircuit<F>,
  /// Corresponds to the Audit_{row/col} hash in the Spartan paper.
  pub r#final: GrandProductCircuit<F>,
}

impl<F: PrimeField> GrandProducts<F> {
  /// Creates the grand product circuits used for memory checking of read-only memories.
  ///
  /// Params
  /// - `eval_table`: M-sized list of table entries
  /// - `dim_i`: log(s)-variate polynomial evaluating to the table index corresponding to each access.
  /// - `dim_i_usize`: Vector of table indices accessed, as `usize`s.
  /// - `read_i`: "Counter polynomial" for memory reads.
  /// - `final_i` "Counter polynomial" for the final memory state.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting.
  pub fn new_read_only(
    eval_table: &[F],
    dim_i: &DensePolynomial<F>,
    dim_i_usize: &[usize],
    read_i: &DensePolynomial<F>,
    final_i: &DensePolynomial<F>,
    r_mem_check: &(F, F),
  ) -> Self {
    let (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    ) = GrandProducts::build_read_only_inputs(
      eval_table,
      dim_i,
      dim_i_usize,
      read_i,
      final_i,
      r_mem_check,
    );

    let prod_init = GrandProductCircuit::new(&grand_product_input_init);
    let prod_read = GrandProductCircuit::new(&grand_product_input_read);
    let prod_write = GrandProductCircuit::new(&grand_product_input_write);
    let prod_final = GrandProductCircuit::new(&grand_product_input_final);

    #[cfg(debug)]
    {
      let hashed_write_set: F = prod_init.evaluate() * prod_write.evaluate();
      let hashed_read_set: F = prod_read.evaluate() * prod_final.evaluate();
      // H(Init) * H(WS) ?= H(RS) * H(Audit)
      // analogous to H(WS) = H(RS) * H(S) in the Lasso paper
      debug_assert_eq!(hashed_read_set, hashed_write_set);
    }

    GrandProducts {
      init: prod_init,
      read: prod_read,
      write: prod_write,
      r#final: prod_final,
    }
  }

  /// Builds the multilinear polynomials that will serve as the inputs to the grand product circuits
  /// used for memory checking. Specifically, this function computes the hash (Reed-Solomon fingerprint)
  /// for each tuple in the "init", "read", "write", and "final" sets (named "Init", "WS", "RS", "Audit"
  /// in the Spartan paper).
  ///
  /// Params
  /// - `v_table`: Memory-sized list of 'v' evaluations
  /// - `a_i`: log(s)-variate polynomial evaluating to the address or table index corresponding to each access.
  /// - `a_i_usize`: Vector of table indices accessed, as `usize`s.
  /// - `t_read_i`: "Counter polynomial" for memory reads.
  /// - `t_final_i` "Counter polynomial" for the final memory state.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting (see `hash_func` closure).
  ///
  /// Returns
  /// - `(init, read, write, final)`: These are the memory polynomials as described in the Spartan paper.
  /// Note that the Lasso describes using `RS`, `WS`, and `S` (using fewer grand products for efficiency),
  /// but that they serve the same purpose: to prove/verify memory consistency.
  fn build_read_only_inputs(
    v_table: &[F],
    a_i: &DensePolynomial<F>,
    a_i_usize: &[usize],
    t_read_i: &DensePolynomial<F>,
    t_final_i: &DensePolynomial<F>,
    r_mem_check: &(F, F),
  ) -> (
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
  ) {
    let (gamma, tau) = r_mem_check;

    // TODO(moodlezoup): (t * gamma^2 + v * gamma + a - tau) * flags + (1 - flags)
    // hash(a, v, t) = t * gamma^2 + v * gamma + a - tau
    let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

    // init: M hash evaluations => log(M)-variate polynomial
    assert_eq!(v_table.len(), t_final_i.len());
    let num_mem_cells = v_table.len();
    let grand_product_input_init = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // addr is given by i, init value is given by eval_table, and ts = 0
          hash_func(&F::from(i as u64), &v_table[i], &F::zero())
        })
        .collect::<Vec<F>>(),
    );

    // final: M hash evaluations => log(M)-variate polynomial
    let grand_product_input_final = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // addr is given by i, value is given by eval_table, and ts is given by audit_ts
          hash_func(&F::from(i as u64), &v_table[i], &t_final_i[i])
        })
        .collect::<Vec<F>>(),
    );

    // TODO(#30): Parallelize

    // read: s hash evaluations => log(s)-variate polynomial
    assert_eq!(a_i.len(), t_read_i.len());

    #[cfg(feature = "multicore")]
    let num_ops = (0..a_i.len()).into_par_iter();
    #[cfg(not(feature = "multicore"))]
    let num_ops = 0..dim_i.len();
    let grand_product_input_read = DensePolynomial::new(
      num_ops
        .clone()
        .map(|i| {
          // addr is given by dim_i, value is given by eval_table, and ts is given by read_ts
          hash_func(&a_i[i], &v_table[a_i_usize[i]], &t_read_i[i])
        })
        .collect::<Vec<F>>(),
    );

    // write: s hash evaluation => log(s)-variate polynomial
    let grand_product_input_write = DensePolynomial::new(
      num_ops
        .map(|i| {
          // addr is given by dim_i, value is given by eval_table, and ts is given by write_ts = read_ts + 1
          hash_func(
            &a_i[i],
            &v_table[a_i_usize[i]],
            &(t_read_i[i] + F::one()),
          )
        })
        .collect::<Vec<F>>(),
    );

    (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    )
  }

  // Creates the grand product circuits used for memory checking of read-only memories including flags.
  ///
  /// Params
  /// - `eval_table`: M-sized list of table entries
  /// - `dim_i`: log(s)-variate polynomial evaluating to the table index corresponding to each access.
  /// - `dim_i_usize`: Vector of table indices accessed, as `usize`s.
  /// - `read_i`: "Counter polynomial" for memory reads.
  /// - `final_i` "Counter polynomial" for the final memory state.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting.
  pub fn new_read_only_with_flags(
    eval_table: &[F],
    dim_i: &DensePolynomial<F>,
    dim_i_usize: &[usize],
    read_i: &DensePolynomial<F>,
    final_i: &DensePolynomial<F>,
    flag_table_i: &Vec<bool>,
    r_mem_check: &(F, F),
  ) -> Self {
    debug_assert_eq!(dim_i.len(), dim_i_usize.len());
    debug_assert_eq!(dim_i.len(), read_i.len());
    debug_assert_eq!(dim_i.len(), flag_table_i.len());

    let (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    ) = GrandProducts::build_read_only_inputs_flags(
      eval_table,
      dim_i,
      dim_i_usize,
      read_i,
      final_i,
      flag_table_i,
      r_mem_check,
    );

    let prod_init = GrandProductCircuit::new(&grand_product_input_init);
    let prod_read = GrandProductCircuit::new(&grand_product_input_read);
    let prod_write = GrandProductCircuit::new(&grand_product_input_write);
    let prod_final = GrandProductCircuit::new(&grand_product_input_final);

    println!("GrandProducts::new_read_only_with_flags");

    #[cfg(test)]
    {
      let hashed_write_set: F = prod_init.evaluate() * prod_write.evaluate();
      let hashed_read_set: F = prod_read.evaluate() * prod_final.evaluate();
      // H(Init) * H(WS) ?= H(RS) * H(Audit)
      // analogous to H(WS) = H(RS) * H(S) in the Lasso paper
      debug_assert_eq!(hashed_read_set, hashed_write_set);
    }

    GrandProducts {
      init: prod_init,
      read: prod_read,
      write: prod_write,
      r#final: prod_final,
    }
  }

   /// Builds the multilinear polynomials that will serve as the inputs to the grand product circuits
  /// used for memory checking. Specifically, this function computes the hash (Reed-Solomon fingerprint)
  /// for each tuple in the "init", "read", "write", and "final" sets (named "Init", "WS", "RS", "Audit"
  /// in the Spartan paper).
  ///
  /// Params
  /// - `v_table`: Memory-sized list of 'v' evaluations
  /// - `a_i`: log(s)-variate polynomial evaluating to the address or table index corresponding to each access.
  /// - `a_i_usize`: Vector of table indices accessed, as `usize`s.
  /// - `t_read_i`: "Counter polynomial" for memory reads.
  /// - `t_final_i` "Counter polynomial" for the final memory state.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting (see `hash_func` closure).
  ///
  /// Returns
  /// - `(init, read, write, final)`: These are the memory polynomials as described in the Spartan paper.
  /// Note that the Lasso describes using `RS`, `WS`, and `S` (using fewer grand products for efficiency),
  /// but that they serve the same purpose: to prove/verify memory consistency.
  fn build_read_only_inputs_flags(
    v_table: &[F],
    a_i: &DensePolynomial<F>,
    a_i_usize: &[usize],
    t_read_i: &DensePolynomial<F>,
    t_final_i: &DensePolynomial<F>,
    flag_table_i: &Vec<bool>,
    r_mem_check: &(F, F),
  ) -> (
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
  ) {
    let (gamma, tau) = r_mem_check;

    // TODO(moodlezoup): (t * gamma^2 + v * gamma + a - tau) * flags + (1 - flags)
    // hash(a, v, t) = t * gamma^2 + v * gamma + a - tau
    let hash_func = |a: &F, v: &F, t: &F| -> F { 
      *t * gamma.square() + *v * *gamma + *a - tau 
    };
    let hash_func_read_write = |a: &F, v: &F, t: &F, flags: bool| -> F { 
      if flags {
        *t * gamma.square() + *v * *gamma + *a - tau 
      } else {
        // TODO(sragss): Assert other terms 0 for cheap(er) commitment.
        F::one()
      }
    };

    // init: M hash evaluations => log(M)-variate polynomial
    assert_eq!(v_table.len(), t_final_i.len());
    let num_mem_cells = v_table.len();
    let grand_product_input_init = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // addr is given by i, init value is given by eval_table, and ts = 0
          hash_func(&F::from(i as u64), &v_table[i], &F::zero())
        })
        .collect::<Vec<F>>(),
    );

    // final: M hash evaluations => log(M)-variate polynomial
    let grand_product_input_final = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // addr is given by i, value is given by eval_table, and ts is given by audit_ts
          hash_func(&F::from(i as u64), &v_table[i], &t_final_i[i])
        })
        .collect::<Vec<F>>(),
    );

    // TODO(#30): Parallelize

    // read: s hash evaluations => log(s)-variate polynomial
    assert_eq!(a_i.len(), t_read_i.len());

    #[cfg(feature = "multicore")]
    let num_ops = (0..a_i.len()).into_par_iter();
    #[cfg(not(feature = "multicore"))]
    let num_ops = 0..dim_i.len();
    let grand_product_input_read = DensePolynomial::new(
      num_ops
        .clone()
        .map(|i| {
          // addr is given by dim_i, value is given by eval_table, and ts is given by read_ts
          hash_func_read_write(&a_i[i], &v_table[a_i_usize[i]], &t_read_i[i], flag_table_i[i])
        })
        .collect::<Vec<F>>(),
    );

    // write: s hash evaluation => log(s)-variate polynomial
    let grand_product_input_write = DensePolynomial::new(
      num_ops
        .map(|i| {
          // addr is given by dim_i, value is given by eval_table, and ts is given by write_ts = read_ts + 1
          hash_func_read_write(
            &a_i[i],
            &v_table[a_i_usize[i]],
            &(t_read_i[i] + F::one()),
        flag_table_i[i]
          )
        })
        .collect::<Vec<F>>(),
    );

    (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    )
  }
}

#[derive(Debug)]
pub struct GrandProductCircuit<F> {
  left_vec: Vec<DensePolynomial<F>>,
  right_vec: Vec<DensePolynomial<F>>,
}

impl<F: PrimeField> GrandProductCircuit<F> {
  fn compute_layer(
    inp_left: &DensePolynomial<F>,
    inp_right: &DensePolynomial<F>,
  ) -> (DensePolynomial<F>, DensePolynomial<F>) {
    let len = inp_left.len() + inp_right.len();
    let outp_left = (0..len / 4)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<F>>();
    let outp_right = (len / 4..len / 2)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<F>>();

    (
      DensePolynomial::new(outp_left),
      DensePolynomial::new(outp_right),
    )
  }

  pub fn new(poly: &DensePolynomial<F>) -> Self {
    let mut left_vec: Vec<DensePolynomial<F>> = Vec::new();
    let mut right_vec: Vec<DensePolynomial<F>> = Vec::new();

    let num_layers = poly.len().log_2();
    let (outp_left, outp_right) = poly.split(poly.len() / 2);

    left_vec.push(outp_left);
    right_vec.push(outp_right);

    for i in 0..num_layers - 1 {
      let (outp_left, outp_right) = GrandProductCircuit::compute_layer(&left_vec[i], &right_vec[i]);
      left_vec.push(outp_left);
      right_vec.push(outp_right);
    }

    GrandProductCircuit {
      left_vec,
      right_vec,
    }
  }

  pub fn evaluate(&self) -> F {
    let len = self.left_vec.len();
    assert_eq!(self.left_vec[len - 1].get_num_vars(), 0);
    assert_eq!(self.right_vec[len - 1].get_num_vars(), 0);
    self.left_vec[len - 1][0] * self.right_vec[len - 1][0]
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProofBatched<F: PrimeField> {
  pub proof: SumcheckInstanceProof<F>,
  pub claims_prod_left: Vec<F>,
  pub claims_prod_right: Vec<F>,
}

#[allow(dead_code)]
impl<F: PrimeField> LayerProofBatched<F> {
  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    claim: F,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut T,
  ) -> (F, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    self
      .proof
      .verify::<G, T>(claim, num_rounds, degree_bound, transcript)
      .unwrap()
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductArgument<F: PrimeField> {
  proof: Vec<LayerProofBatched<F>>,
}

impl<F: PrimeField> BatchedGrandProductArgument<F> {
  #[tracing::instrument(skip_all, name = "BatchedGrandProductArgument.prove")]
  pub fn prove<G>(
    grand_product_circuits: &mut Vec<&mut GrandProductCircuit<F>>,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    assert!(!grand_product_circuits.is_empty());

    let mut proof_layers: Vec<LayerProofBatched<F>> = Vec::new();
    let num_layers = grand_product_circuits[0].left_vec.len();
    let mut claims_to_verify = (0..grand_product_circuits.len())
      .map(|i| grand_product_circuits[i].evaluate())
      .collect::<Vec<F>>();

    let mut rand = Vec::new();
    for layer_id in (0..num_layers).rev() {
      // prepare parallel instance that share poly_C first
      let len = grand_product_circuits[0].left_vec[layer_id].len()
        + grand_product_circuits[0].right_vec[layer_id].len();

      let mut poly_C_par = DensePolynomial::new(EqPolynomial::<F>::new(rand.clone()).evals());
      assert_eq!(poly_C_par.len(), len / 2);

      let num_rounds_prod = poly_C_par.len().log_2();
      let comb_func_prod = |poly_A_comp: &F, poly_B_comp: &F, poly_C_comp: &F| -> F {
        *poly_A_comp * *poly_B_comp * *poly_C_comp
      };

      let mut poly_A_batched_par: Vec<&mut DensePolynomial<F>> = Vec::new();
      let mut poly_B_batched_par: Vec<&mut DensePolynomial<F>> = Vec::new();
      for prod_circuit in grand_product_circuits.iter_mut() {
        poly_A_batched_par.push(&mut prod_circuit.left_vec[layer_id]);
        poly_B_batched_par.push(&mut prod_circuit.right_vec[layer_id])
      }
      let poly_vec_par = (
        &mut poly_A_batched_par,
        &mut poly_B_batched_par,
        &mut poly_C_par,
      );

      // produce a fresh set of coeffs and a joint claim
      let coeff_vec: Vec<F> = <Transcript as ProofTranscript<G>>::challenge_vector(
        transcript,
        b"rand_coeffs_next_layer",
        claims_to_verify.len(),
      );
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      // TODO(moodlezoup): Degree 5 for last layer
      let (proof, rand_prod, claims_prod) = SumcheckInstanceProof::<F>::prove_cubic_batched::<_, G>(
        &claim,
        num_rounds_prod,
        poly_vec_par,
        &coeff_vec,
        comb_func_prod,
        transcript,
      );

      let (claims_prod_left, claims_prod_right, _claims_eq) = claims_prod;
      for i in 0..grand_product_circuits.len() {
        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_prod_left",
          &claims_prod_left[i],
        );

        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_prod_right",
          &claims_prod_right[i],
        );
      }

      // produce a random challenge to condense two claims into a single claim
      let r_layer =
        <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"challenge_r_layer");

      claims_to_verify = (0..grand_product_circuits.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<F>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;

      proof_layers.push(LayerProofBatched {
        proof,
        claims_prod_left,
        claims_prod_right,
      });
    }

    (
      BatchedGrandProductArgument {
        proof: proof_layers,
      },
      rand,
    )
  }

  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    claims_prod_vec: &Vec<F>,
    len: usize,
    transcript: &mut T,
  ) -> (Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    let num_layers = len.log_2();
    let mut rand: Vec<F> = Vec::new();
    assert_eq!(self.proof.len(), num_layers);

    let mut claims_to_verify = claims_prod_vec.to_owned();
    for (num_rounds, i) in (0..num_layers).enumerate() {
      // produce random coefficients, one for each instance
      let coeff_vec =
        transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());

      // produce a joint claim
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      let (claim_last, rand_prod) = self.proof[i].verify::<G, T>(claim, num_rounds, 3, transcript);

      let claims_prod_left = &self.proof[i].claims_prod_left;
      let claims_prod_right = &self.proof[i].claims_prod_right;
      assert_eq!(claims_prod_left.len(), claims_prod_vec.len());
      assert_eq!(claims_prod_right.len(), claims_prod_vec.len());

      for i in 0..claims_prod_vec.len() {
        transcript.append_scalar(b"claim_prod_left", &claims_prod_left[i]);
        transcript.append_scalar(b"claim_prod_right", &claims_prod_right[i]);
      }

      assert_eq!(rand.len(), rand_prod.len());
      let eq: F = (0..rand.len())
        .map(|i| rand[i] * rand_prod[i] + (F::one() - rand[i]) * (F::one() - rand_prod[i]))
        .product();
      let claim_expected: F = (0..claims_prod_vec.len())
        .map(|i| coeff_vec[i] * (claims_prod_left[i] * claims_prod_right[i] * eq))
        .sum();

      assert_eq!(claim_expected, claim_last);

      // produce a random challenge
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

      claims_to_verify = (0..claims_prod_left.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<F>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;
    }
    (claims_to_verify, rand)
  }
}

#[cfg(test)]
mod grand_product_circuit_tests {
  use super::*;
  use ark_curve25519::{EdwardsProjective as G1Projective, Fr};

  #[test]
  fn prove_verify() {
    let factorial = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
    let mut factorial_circuit = GrandProductCircuit::new(&factorial);
    let expected_eval = vec![Fr::from(24)];
    assert_eq!(factorial_circuit.evaluate(), Fr::from(24));

    let mut transcript = Transcript::new(b"test_transcript");
    let mut circuits_vec = vec![&mut factorial_circuit];
    let (proof, _) =
      BatchedGrandProductArgument::prove::<G1Projective>(&mut circuits_vec, &mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    proof.verify::<G1Projective, _>(&expected_eval, 4, &mut transcript);
  }
}
