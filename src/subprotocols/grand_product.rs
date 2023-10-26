#![allow(dead_code)]

use super::sumcheck::{CubicSumcheckParams, SumcheckInstanceProof};
use crate::jolt::vm::PolynomialRepresentation;
use crate::lasso::fingerprint_strategy::MemBatchInfo;
use crate::lasso::gp_evals::GPEvals;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::sumcheck::CubicSumcheckType;
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
#[derive(Debug, Clone)]
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
          hash_func(&a_i[i], &v_table[a_i_usize[i]], &(t_read_i[i] + F::one()))
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

    #[cfg(test)]
    {
      let hashed_write_set: F = prod_init.evaluate() * prod_write.evaluate();
      let hashed_read_set: F = prod_read.evaluate() * prod_final.evaluate();
      // H(Init) * H(WS) ?= H(RS) * H(Audit)
      // analogous to H(WS) = H(RS) * H(S) in the Lasso paper
      assert_eq!(hashed_read_set, hashed_write_set);
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
    let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };
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
          hash_func_read_write(
            &a_i[i],
            &v_table[a_i_usize[i]],
            &t_read_i[i],
            flag_table_i[i],
          )
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
            flag_table_i[i],
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

#[derive(Debug, Clone)]
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

  pub fn new(leaves: &DensePolynomial<F>) -> Self {
    let mut left_vec: Vec<DensePolynomial<F>> = Vec::new();
    let mut right_vec: Vec<DensePolynomial<F>> = Vec::new();

    let num_layers = leaves.len().log_2();
    let (outp_left, outp_right) = leaves.split(leaves.len() / 2);

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
  pub claims_poly_A: Vec<F>,
  pub claims_poly_B: Vec<F>,
  pub combine_prod: bool, // TODO(sragss): Use enum. Sumcheck.rs/CubicType
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

/// BatchedGrandProductCircuitInterpretable
pub trait BGPCInterpretable<F: PrimeField>: MemBatchInfo {
  // a for init, final
  fn a_mem(&self, _memory_index: usize, leaf_index: usize) -> F {
    F::from(leaf_index as u64)
  }
  // a for read, write
  fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F;

  // v for init, final
  fn v_mem(&self, memory_index: usize, leaf_index: usize) -> F;
  // v for read, write
  fn v_ops(&self, memory_index: usize, leaf_index: usize) -> F;

  // t for init, final
  fn t_init(&self, _memory_index: usize, _leaf_index: usize) -> F {
    F::zero()
  }
  fn t_final(&self, memory_index: usize, leaf_index: usize) -> F;
  // t for read, write
  fn t_read(&self, memory_index: usize, leaf_index: usize) -> F;
  fn t_write(&self, memory_index: usize, leaf_index: usize) -> F {
    self.t_read(memory_index, leaf_index) + F::one()
  }

  fn fingerprint_read(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_ops(memory_index, leaf_index),
      self.v_ops(memory_index, leaf_index),
      self.t_read(memory_index, leaf_index),
      gamma,
      tau,
    )
  }

  fn fingerprint_write(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_ops(memory_index, leaf_index),
      self.v_ops(memory_index, leaf_index),
      self.t_write(memory_index, leaf_index),
      gamma,
      tau,
    )
  }

  fn fingerprint_init(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_mem(memory_index, leaf_index),
      self.v_mem(memory_index, leaf_index),
      self.t_init(memory_index, leaf_index),
      gamma,
      tau,
    )
  }

  fn fingerprint_final(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_mem(memory_index, leaf_index),
      self.v_mem(memory_index, leaf_index),
      self.t_final(memory_index, leaf_index),
      gamma,
      tau,
    )
  }

  fn fingerprint(a: F, v: F, t: F, gamma: &F, tau: &F) -> F {
    t * gamma.square() + v * gamma + a - tau
  }

  fn compute_leaves(
    &self,
    memory_index: usize,
    r_hash: (&F, &F),
  ) -> (
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
  ) {
    let init_evals = (0..self.mem_size())
      .map(|i| self.fingerprint_init(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    let read_evals = (0..self.ops_size())
      .map(|i| self.fingerprint_read(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    let write_evals = (0..self.ops_size())
      .map(|i| self.fingerprint_write(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    let final_evals = (0..self.mem_size())
      .map(|i| self.fingerprint_final(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    (
      DensePolynomial::new(init_evals),
      DensePolynomial::new(read_evals),
      DensePolynomial::new(write_evals),
      DensePolynomial::new(final_evals),
    )
  }

  fn construct_batches(
    &self,
    r_hash: (&F, &F),
  ) -> (
    BatchedGrandProductCircuit<F>,
    BatchedGrandProductCircuit<F>,
    Vec<GPEvals<F>>,
  ) {
    let mut rw_circuits = Vec::with_capacity(self.num_memories() * 2);
    let mut if_circuits = Vec::with_capacity(self.num_memories() * 2);
    let mut gp_evals = Vec::with_capacity(self.num_memories());
    for memory_index in 0..self.num_memories() {
      let (init_leaves, read_leaves, write_leaves, final_leaves) =
        self.compute_leaves(memory_index, r_hash);
      let (init_gpc, read_gpc, write_gpc, final_gpc) = (
        GrandProductCircuit::new(&init_leaves),
        GrandProductCircuit::new(&read_leaves),
        GrandProductCircuit::new(&write_leaves),
        GrandProductCircuit::new(&final_leaves),
      );

      gp_evals.push(GPEvals::new(
        init_gpc.evaluate(),
        read_gpc.evaluate(),
        write_gpc.evaluate(),
        final_gpc.evaluate(),
      ));

      rw_circuits.push(read_gpc);
      rw_circuits.push(write_gpc);
      if_circuits.push(init_gpc);
      if_circuits.push(final_gpc);
    }
    (
      BatchedGrandProductCircuit::new_batch(rw_circuits),
      BatchedGrandProductCircuit::new_batch(if_circuits),
      gp_evals,
    )
  }
}

impl<F: PrimeField> BGPCInterpretable<F> for PolynomialRepresentation<F> {
  fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F {
    self.dim[memory_index % self.C][leaf_index]
  }

  fn v_mem(&self, memory_index: usize, leaf_index: usize) -> F {
    self.materialized_subtables[self.memory_to_subtable_map[memory_index]][leaf_index]
  }

  fn v_ops(&self, memory_index: usize, leaf_index: usize) -> F {
    self.E_polys[memory_index][leaf_index]
  }

  fn t_final(&self, memory_index: usize, leaf_index: usize) -> F {
    self.final_cts[memory_index][leaf_index]
  }

  fn t_read(&self, memory_index: usize, leaf_index: usize) -> F {
    self.read_cts[memory_index][leaf_index]
  }

  // TODO(sragss): Some if this logic is sharable.
  fn construct_batches(
    &self,
    r_hash: (&F, &F),
  ) -> (
    BatchedGrandProductCircuit<F>,
    BatchedGrandProductCircuit<F>,
    Vec<GPEvals<F>>,
  ) {
    // compute leaves for all the batches                     (shared)
    // convert the rw leaves to flagged leaves                (custom)
    // create GPCs for each of the leaves (&leaves)           (custom)
    // evaluate the GPCs                                      (shared)
    // construct 1x batch with flags, 1x batch without flags  (custom)

    let mut rw_circuits = Vec::with_capacity(self.num_memories() * 2);
    let mut if_circuits = Vec::with_capacity(self.num_memories() * 2);
    let mut gp_evals = Vec::with_capacity(self.num_memories());

    // Stores the initial fingerprinted values for read and write memories. GPC stores the upper portion of the tree after the fingerprints at the leaves
    // experience flagging (toggling based on the flag value at that leaf).
    let mut rw_fingerprints: Vec<DensePolynomial<F>> = Vec::with_capacity(self.num_memories() * 2);
    for memory_index in 0..self.num_memories() {
      let (init_fingerprints, read_fingerprints, write_fingerprints, final_fingerprints) =
        self.compute_leaves(memory_index, r_hash);

      let (mut read_leaves, mut write_leaves) =
        (read_fingerprints.evals(), write_fingerprints.evals());
      rw_fingerprints.push(read_fingerprints);
      rw_fingerprints.push(write_fingerprints);
      for leaf_index in 0..self.ops_size() {
        // TODO(sragss): Would be faster if flags were non-FF repr
        let flag = self.subtable_flag_polys[self.memory_to_subtable_map[memory_index]][leaf_index];
        if flag == F::zero() {
          read_leaves[leaf_index] = F::one();
          write_leaves[leaf_index] = F::one();
        }
      }

      let (init_gpc, final_gpc) = (
        GrandProductCircuit::new(&init_fingerprints),
        GrandProductCircuit::new(&final_fingerprints),
      );
      let (read_gpc, write_gpc) = (
        GrandProductCircuit::new(&DensePolynomial::new(read_leaves)),
        GrandProductCircuit::new(&DensePolynomial::new(write_leaves)),
      );

      gp_evals.push(GPEvals::new(
        init_gpc.evaluate(),
        read_gpc.evaluate(),
        write_gpc.evaluate(),
        final_gpc.evaluate(),
      ));

      rw_circuits.push(read_gpc);
      rw_circuits.push(write_gpc);
      if_circuits.push(init_gpc);
      if_circuits.push(final_gpc);
    }

    // self.memory_to_subtable map has to be expanded because we've doubled the number of "grand products memorys": [read_0, write_0, ... read_NUM_MEMOREIS, write_NUM_MEMORIES]
    let expanded_flag_map: Vec<usize> = self
      .memory_to_subtable_map
      .iter()
      .flat_map(|subtable_index| [*subtable_index, *subtable_index])
      .collect();

    // Prover has access to subtable_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
    let rw_batch = BatchedGrandProductCircuit::new_batch_flags(
      rw_circuits,
      self.subtable_flag_polys.clone(),
      expanded_flag_map,
      rw_fingerprints,
    );

    let if_batch = BatchedGrandProductCircuit::new_batch(if_circuits);

    (rw_batch, if_batch, gp_evals)
  }
}

pub struct BatchedGrandProductCircuit<F: PrimeField> {
  pub circuits: Vec<GrandProductCircuit<F>>,

  flags: Option<Vec<DensePolynomial<F>>>,
  flag_map: Option<Vec<usize>>,
  fingerprint_polys: Option<Vec<DensePolynomial<F>>>,
}

impl<F: PrimeField> BatchedGrandProductCircuit<F> {
  pub fn new_batch(circuits: Vec<GrandProductCircuit<F>>) -> Self {
    Self {
      circuits,
      flags: None,
      flag_map: None,
      fingerprint_polys: None,
    }
  }

  pub fn new_batch_flags(
    circuits: Vec<GrandProductCircuit<F>>,
    flags: Vec<DensePolynomial<F>>,
    flag_map: Vec<usize>,
    fingerprint_polys: Vec<DensePolynomial<F>>,
  ) -> Self {
    assert_eq!(circuits.len(), flag_map.len());
    assert_eq!(circuits.len(), fingerprint_polys.len());
    flag_map.iter().for_each(|i| assert!(*i < flags.len()));

    Self {
      circuits,
      flags: Some(flags),
      flag_map: Some(flag_map),
      fingerprint_polys: Some(fingerprint_polys),
    }
  }

  fn num_layers(&self) -> usize {
    let prod_layers = self.circuits[0].left_vec.len();

    if self.flags.is_some() {
      prod_layers + 1
    } else {
      prod_layers
    }
  }

  fn sumcheck_layer_params(
    &self,
    layer_id: usize,
    eq: DensePolynomial<F>,
  ) -> CubicSumcheckParams<F> {
    if self.flags.is_some() && layer_id == 0 {
      let flags = self.flags.as_ref().unwrap();
      debug_assert_eq!(flags[0].len(), eq.len());

      let num_rounds = eq.get_num_vars();
      // TODO(sragss): Handle .as_ref().unwrap().clone() without cloning.
      CubicSumcheckParams::new_flags(
        self
          .fingerprint_polys
          .as_ref()
          .unwrap()
          .iter()
          .map(|poly| poly.clone())
          .collect(),
        self.flags.as_ref().unwrap().clone(),
        eq,
        self.flag_map.as_ref().unwrap().clone(),
        num_rounds,
      )
    } else {
      // If flags is present layer_id 1 corresponds to circuits.left_vec/right_vec[0]
      let layer_id = if self.flags.is_some() {
        layer_id - 1
      } else {
        layer_id
      };

      let num_rounds = self.circuits[0].left_vec[layer_id].get_num_vars();

      // TODO(sragss): rm clone – use remove / take
      CubicSumcheckParams::new_prod(
        self
          .circuits
          .iter()
          .map(|circuit| circuit.left_vec[layer_id].clone())
          .collect(),
        self
          .circuits
          .iter()
          .map(|circuit| circuit.right_vec[layer_id].clone())
          .collect(),
        eq,
        num_rounds,
      )
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductArgument<F: PrimeField> {
  proof: Vec<LayerProofBatched<F>>,
}

impl<F: PrimeField> BatchedGrandProductArgument<F> {
  #[tracing::instrument(skip_all, name = "BatchedGrandProductArgument.prove")]
  pub fn prove<G>(
    batch: BatchedGrandProductCircuit<F>,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    println!("BatchedGrandProductArgument::prove()");
    let mut proof_layers: Vec<LayerProofBatched<F>> = Vec::new();
    let mut claims_to_verify = (0..batch.circuits.len())
      .map(|i| batch.circuits[i].evaluate())
      .collect::<Vec<F>>();

    let mut rand = Vec::new();
    for layer_id in (0..batch.num_layers()).rev() {
      // produce a fresh set of coeffs and a joint claim
      let coeff_vec: Vec<F> = <Transcript as ProofTranscript<G>>::challenge_vector(
        transcript,
        b"rand_coeffs_next_layer",
        claims_to_verify.len(),
      );
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      let eq = DensePolynomial::new(EqPolynomial::<F>::new(rand.clone()).evals());
      let params = batch.sumcheck_layer_params(layer_id, eq);
      // TODO(sragss): Are these params constructed properly?
      let sumcheck_type = params.sumcheck_type.clone();
      let (proof, rand_prod, claims_prod) = SumcheckInstanceProof::prove_cubic_batched_special::<G>(
        &claim, params, &coeff_vec, transcript,
      );

      let (claims_poly_A, claims_poly_B, _claim_eq) = claims_prod;
      for i in 0..batch.circuits.len() {
        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_prod_left",
          &claims_poly_A[i],
        );

        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_prod_right",
          &claims_poly_B[i],
        );
      }

      if sumcheck_type == CubicSumcheckType::Prod {
        // Prod layers must generate an additional random coefficient. The sumcheck randomness indexes into the current layer,
        // but the resulting randomness and claims are about the next layer. The next layer is indexed by an additional variable
        // in the MSB. We use the evaluations V_i(r,0), V_i(r,1) to compute V_i(r, r').

        // produce a random challenge to condense two claims into a single claim
        let r_layer =
          <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"challenge_r_layer");

        claims_to_verify = (0..batch.circuits.len())
          .map(|i| claims_poly_A[i] + r_layer * (claims_poly_B[i] - claims_poly_A[i]))
          .collect::<Vec<F>>();

        let mut ext = vec![r_layer];
        ext.extend(rand_prod);
        rand = ext;

        proof_layers.push(LayerProofBatched {
          proof,
          claims_poly_A,
          claims_poly_B,
          combine_prod: true,
        });
      } else {
        // CubicSumcheckType::Flags
        // Flag layers do not need the additional bit as the randomness from the previous layers have already fully determined
        assert_eq!(layer_id, 0);
        rand = rand_prod;

        // TODO(sragss): Only needed for debugging
        claims_to_verify = (0..batch.circuits.len())
          .map(|i| claims_poly_A[i] * claims_poly_B[i] + (F::one() - claims_poly_B[i]))
          .collect::<Vec<F>>();

        proof_layers.push(LayerProofBatched {
          proof,
          claims_poly_A,
          claims_poly_B,
          combine_prod: false,
        });
      }
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
    transcript: &mut T,
  ) -> (Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    let mut rand: Vec<F> = Vec::new();
    let num_layers = self.proof.len();

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

      let claims_prod_left = &self.proof[i].claims_poly_A;
      let claims_prod_right = &self.proof[i].claims_poly_B;
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

      // TODO(sragss): Comment about what is going on here.
      let claim_expected = if self.proof[i].combine_prod {
        let claim_expected: F = (0..claims_prod_vec.len())
          .map(|i| {
            coeff_vec[i]
              * CubicSumcheckParams::combine_prod(&claims_prod_left[i], &claims_prod_right[i], &eq)
          })
          .sum();

        // produce a random challenge
        let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

        claims_to_verify = (0..claims_prod_left.len())
          .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
          .collect::<Vec<F>>();

        let mut ext = vec![r_layer];
        ext.extend(rand_prod);
        rand = ext;

        claim_expected
      } else {
        let claim_expected: F = (0..claims_prod_vec.len())
          .map(|i| {
            coeff_vec[i]
              * CubicSumcheckParams::combine_flags(&claims_prod_left[i], &claims_prod_right[i], &eq)
          })
          .sum();

        rand = rand_prod;

        claims_to_verify = (0..claims_prod_left.len())
          .map(|i| claims_prod_left[i] * claims_prod_right[i] + (F::one() - claims_prod_right[i]))
          .collect::<Vec<F>>();

        claim_expected
      };

      assert_eq!(claim_expected, claim_last);
    }
    (claims_to_verify, rand)
  }
}

#[cfg(test)]
mod grand_product_circuit_tests {
  use super::*;
  use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
  use ark_std::{One, Zero};

  #[test]
  fn prove_verify() {
    let factorial = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
    let factorial_circuit = GrandProductCircuit::new(&factorial);
    let expected_eval = vec![Fr::from(24)];
    assert_eq!(factorial_circuit.evaluate(), Fr::from(24));

    let mut transcript = Transcript::new(b"test_transcript");
    let circuits_vec = vec![factorial_circuit];
    let batch = BatchedGrandProductCircuit::new_batch(circuits_vec);
    let (proof, _) = BatchedGrandProductArgument::prove::<G1Projective>(batch, &mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    proof.verify::<G1Projective, _>(&expected_eval, &mut transcript);
  }

  #[test]
  fn gp_unflagged() {
    // Fundamentally grand products performs a multi-set check, so skip fingerprinting and all that, construct GP circuits directly
    let read_leaves = vec![Fr::from(10), Fr::from(20)];
    let write_leaves = vec![Fr::from(100), Fr::from(200)];

    let read_poly = DensePolynomial::new(read_leaves);
    let write_poly = DensePolynomial::new(write_leaves);

    let (read_gpc, write_gpc) = (
      GrandProductCircuit::new(&read_poly),
      GrandProductCircuit::new(&write_poly),
    );
    let batch = BatchedGrandProductCircuit::new_batch(vec![read_gpc, write_gpc]);

    let mut transcript = Transcript::new(b"test_transcript");
    let (proof, prove_rand) =
      BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(batch, &mut transcript);

    let expected_eval_read = Fr::from(10) * Fr::from(20);
    let expected_eval_write = Fr::from(100) * Fr::from(200);
    let mut transcript = Transcript::new(b"test_transcript");
    let (verify_claims, verify_rand) = proof.verify::<G1Projective, _>(
      &vec![expected_eval_read, expected_eval_write],
      &mut transcript,
    );

    assert_eq!(prove_rand, verify_rand);
    assert_eq!(verify_claims.len(), 2);
    assert_eq!(verify_claims[0], read_poly.evaluate(&verify_rand));
    assert_eq!(verify_claims[1], write_poly.evaluate(&verify_rand));
  }

  #[test]
  fn gp_flagged() {
    let read_fingerprints = vec![Fr::from(10), Fr::from(20), Fr::from(30), Fr::from(40)];
    let write_fingerprints = vec![Fr::from(100), Fr::from(200), Fr::from(300), Fr::from(400)];

    // toggle off index '2'
    let flag_poly = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::zero(), Fr::one()]);

    // Grand Product Circuit leaves are those that are toggled
    let mut read_leaves = read_fingerprints.clone();
    read_leaves[2] = Fr::one();
    let mut write_leaves = write_fingerprints.clone();
    write_leaves[2] = Fr::one();

    let read_leaf_poly = DensePolynomial::new(read_leaves);
    let write_leaf_poly = DensePolynomial::new(write_leaves);

    let flag_map = vec![0, 0];

    let fingerprint_polys = vec![
      DensePolynomial::new(read_fingerprints),
      DensePolynomial::new(write_fingerprints),
    ];

    // Construct the GPCs not from the raw fingerprints, but from the flagged fingerprints!
    let (read_gpc, write_gpc) = (
      GrandProductCircuit::new(&read_leaf_poly),
      GrandProductCircuit::new(&write_leaf_poly),
    );

    // Batch takes reference to the "untoggled" fingerprint_polys for the final flag layer that feeds into the leaves, which have been flagged (set to 1 if the flag is not 1)
    let batch = BatchedGrandProductCircuit::new_batch_flags(
      vec![read_gpc, write_gpc],
      vec![flag_poly.clone()],
      flag_map,
      fingerprint_polys.clone(),
    );

    let mut transcript = Transcript::new(b"test_transcript");
    let (proof, prove_rand) =
      BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(batch, &mut transcript);

    let expected_eval_read: Fr = Fr::from(10) * Fr::from(20) * Fr::from(40);
    let expected_eval_write: Fr = Fr::from(100) * Fr::from(200) * Fr::from(400);
    let expected_evals = vec![expected_eval_read, expected_eval_write];

    let mut transcript = Transcript::new(b"test_transcript");
    let (verify_claims, verify_rand) =
      proof.verify::<G1Projective, _>(&expected_evals, &mut transcript);

    assert_eq!(prove_rand, verify_rand);
    assert_eq!(verify_claims.len(), 2);

    assert_eq!(proof.proof.len(), 3);
    // Claims about raw fingerprints bound to r_z
    assert_eq!(
      proof.proof[2].claims_poly_A[0],
      fingerprint_polys[0].evaluate(&verify_rand)
    );
    assert_eq!(
      proof.proof[2].claims_poly_A[1],
      fingerprint_polys[1].evaluate(&verify_rand)
    );

    // Claims about flags bound to r_z
    assert_eq!(
      proof.proof[2].claims_poly_B[0],
      flag_poly.evaluate(&verify_rand)
    );
    assert_eq!(
      proof.proof[2].claims_poly_B[1],
      flag_poly.evaluate(&verify_rand)
    );

    let verifier_flag_eval = flag_poly.evaluate(&verify_rand);
    let verifier_read_eval = verifier_flag_eval * fingerprint_polys[0].evaluate(&verify_rand)
      + Fr::one()
      - verifier_flag_eval;
    let verifier_write_eval = verifier_flag_eval * fingerprint_polys[1].evaluate(&verify_rand)
      + Fr::one()
      - verifier_flag_eval;
    assert_eq!(verify_claims[0], verifier_read_eval);
    assert_eq!(verify_claims[1], verifier_write_eval);
  }
}
