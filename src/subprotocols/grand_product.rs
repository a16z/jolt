#![allow(dead_code)]

use super::sumcheck::{SumcheckInstanceProof, CubicSumcheckParams};
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

    let prod_init = GrandProductCircuit::new(grand_product_input_init);
    let prod_read = GrandProductCircuit::new(grand_product_input_read);
    let prod_write = GrandProductCircuit::new(grand_product_input_write);
    let prod_final = GrandProductCircuit::new(grand_product_input_final);

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

    let prod_init = GrandProductCircuit::new(grand_product_input_init);
    let prod_read = GrandProductCircuit::new(grand_product_input_read);
    let prod_write = GrandProductCircuit::new(grand_product_input_write);
    let prod_final = GrandProductCircuit::new(grand_product_input_final);

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

#[derive(Debug, Clone)]
pub struct GrandProductCircuit<F> {
  left_vec: Vec<DensePolynomial<F>>,
  right_vec: Vec<DensePolynomial<F>>,
  leaves: DensePolynomial<F> // TODO(sragss): Wasted RAM for non-flags GPCs. Swap to Option<DensePolynomial<F>>
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

  pub fn new(leaves: DensePolynomial<F>) -> Self {
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
      leaves
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
  pub combine_prod: bool // TODO(sragss): Use enum. Sumcheck.rs/CubicType
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
pub trait BGPCInterpretable<F: PrimeField> : MemBatchInfo {
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
      tau
    )
	}

	fn fingerprint_write(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_ops(memory_index, leaf_index), 
      self.v_ops(memory_index, leaf_index), 
      self.t_write(memory_index, leaf_index),
      gamma,
      tau
    )
	}

	fn fingerprint_init(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_mem(memory_index, leaf_index), 
      self.v_mem(memory_index, leaf_index), 
      self.t_init(memory_index, leaf_index),
      gamma,
      tau
    )
	}

	fn fingerprint_final(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    Self::fingerprint(
      self.a_mem(memory_index, leaf_index), 
      self.v_mem(memory_index, leaf_index), 
      self.t_final(memory_index, leaf_index),
      gamma,
      tau
    )
	}

  fn fingerprint(a: F, v: F, t: F, gamma: &F, tau: &F) -> F {
    t * gamma.square() + v * gamma + a - tau 
  }

	fn compute_leaves(&self, memory_index: usize, r_hash: (&F, &F)) -> (
		DensePolynomial<F>, 
		DensePolynomial<F>, 
		DensePolynomial<F>, 
		DensePolynomial<F>) {
      println!("BGPCInterpable::compute_leaves()");
			let init_evals = (0..self.mem_size()).map(|i| 
        self.fingerprint_init(memory_index, i, r_hash.0, r_hash.1)
      ).collect();
			let read_evals = (0..self.ops_size()).map(|i| 
        self.fingerprint_read(memory_index, i, r_hash.0, r_hash.1)
      ).collect();
			let write_evals = (0..self.ops_size()).map(|i| 
        self.fingerprint_write(memory_index, i, r_hash.0, r_hash.1)
      ).collect();
			let final_evals = (0..self.mem_size()).map(|i| 
        self.fingerprint_final(memory_index, i, r_hash.0, r_hash.1)
      ).collect();
      println!("inits {:?}", init_evals);
      println!("reads {:?}", read_evals);
      println!("writes {:?}", write_evals);
      println!("finals {:?}", final_evals);
			(
        DensePolynomial::new(init_evals), 
        DensePolynomial::new(read_evals), 
        DensePolynomial::new(write_evals), 
        DensePolynomial::new(final_evals)
      )
	}

	fn construct_batched_read_write(&self, reads: Vec<GrandProductCircuit<F>>, writes: Vec<GrandProductCircuit<F>>) -> BatchedGrandProductCircuit<F> {
    debug_assert_eq!(reads.len(), writes.len());
    let interleaves = reads.into_iter().zip(writes).flat_map(|(read, write)| [read, write]).collect();

		BatchedGrandProductCircuit::new_batch(interleaves)	
  }
	fn construct_batched_init_final(&self, inits: Vec<GrandProductCircuit<F>>, finals: Vec<GrandProductCircuit<F>>) -> BatchedGrandProductCircuit<F> {
    debug_assert_eq!(inits.len(), finals.len());
    let interleaves = inits.into_iter().zip(finals).flat_map(|(init, fin)| [init, fin]).collect();

		BatchedGrandProductCircuit::new_batch(interleaves)	
	}
}

impl<F: PrimeField> BGPCInterpretable<F> for PolynomialRepresentation<F> {
    fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F {
      self.dim[memory_index % self.C][leaf_index]
    }

    fn v_mem(&self, memory_index: usize, leaf_index: usize) -> F {
      self.materialized_subtables[memory_index][leaf_index]
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

    // TODO(sragss): Flags overrides
    // fn fingerprint_read(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    //   todo!("flags")
    // }
  
    // fn fingerprint_write(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    //   todo!("flags")
    // }

    // fn construct_batched_read_write(
    //   &self, 
    //   reads: Vec<GrandProductCircuit<F>>, 
    //   writes: Vec<GrandProductCircuit<F>>) -> BatchedGrandProductCircuit<F> {
    //     debug_assert_eq!(reads.len(), writes.len());
    //     let interleaves = reads.into_iter().zip(writes).flat_map(|(read, write)| [read, write]).collect();
    
    //     let flag_map = vec![]; // TODO: Flag map
    //     BatchedGrandProductCircuit::new_batch_flags(interleaves, self.flag_polys.clone(), flag_map)	
    // }
}

pub struct BatchedGrandProductCircuit<F: PrimeField> {
  pub circuits: Vec<GrandProductCircuit<F>>,

  flags: Option<Vec<DensePolynomial<F>>>,
  flag_map: Option<Vec<usize>>
}

impl<F: PrimeField> BatchedGrandProductCircuit<F> {

  pub fn construct<P: BGPCInterpretable<F>>(polys: &P, r_fingerprint: (&F, &F)) -> (Self, Self, Vec<GPEvals<F>>) {
    let mut gp_evals = Vec::with_capacity(polys.num_memories());
    let mut read_circuits: Vec<GrandProductCircuit<F>> = Vec::with_capacity(polys.num_memories());
    let mut write_circuits: Vec<GrandProductCircuit<F>> = Vec::with_capacity(polys.num_memories());
    let mut init_circuits: Vec<GrandProductCircuit<F>> = Vec::with_capacity(polys.num_memories());
    let mut final_circuits: Vec<GrandProductCircuit<F>> = Vec::with_capacity(polys.num_memories());

    // For each: Compute leaf hash, compute GrandProductCircuit
    for memory_index in 0..polys.num_memories() {
      let (init_leaves, read_leaves, write_leaves, final_leaves) 
        = polys.compute_leaves(memory_index, r_fingerprint);
      let init_circuit = GrandProductCircuit::new(init_leaves);
      let read_circuit = GrandProductCircuit::new(read_leaves);
      let write_circuit = GrandProductCircuit::new(write_leaves);
      let final_circuit = GrandProductCircuit::new(final_leaves);

      gp_evals.push(
        GPEvals::new(
          init_circuit.evaluate(), 
          read_circuit.evaluate(), 
          write_circuit.evaluate(), 
          final_circuit.evaluate()
        )
      );

      init_circuits.push(init_circuit);
      read_circuits.push(read_circuit);
      write_circuits.push(write_circuit);
      final_circuits.push(final_circuit);
    }
	
    let batched_read_write = polys.construct_batched_read_write(read_circuits, write_circuits);
    let batched_init_final = polys.construct_batched_init_final(init_circuits, final_circuits);
    (batched_read_write, batched_init_final, gp_evals)
  }

  /// flag_map: [flag_index_0, ... flag_index_NUM_MEMORIES]
  pub fn new_batches_flags(
    gps: Vec<GrandProducts<F>>, 
    flags: Vec<DensePolynomial<F>>, 
    flag_map: Vec<usize>) -> (Self, Self) {
      debug_assert_eq!(gps.len(), flag_map.len());
      let mut read_write_circuits = Vec::with_capacity(gps.len() * 2);
      let mut init_final_circuits = Vec::with_capacity(gps.len() * 2);

      for gp in gps {
        read_write_circuits.push(gp.read);
        read_write_circuits.push(gp.write);

        init_final_circuits.push(gp.init);
        init_final_circuits.push(gp.r#final);
      }

      // gps: [mem_0, ... mem_NUM_MEMORIES]
      // flags: [flag_0, ... flag_NUM_MEMORIES]
      // flag_map: [flag_index_0, ... flag_index_NUM_MEMORIES]
      // read_write_circuits: [read_0, write_0, read_NUM_MEMORIES, write_NUM_MEMORIES]
      // expanded_flag_map[circuit_index] => flag_index
      let expanded_flag_map = flag_map.iter().flat_map(|&i| vec![i; 2]).collect();

      (Self {
        circuits: read_write_circuits,
        flags: Some(flags),
        flag_map: Some(expanded_flag_map)
      },
      Self {
        circuits: init_final_circuits,
        flags: None,
        flag_map: None
      })
  }

  pub fn new_batch(circuits: Vec<GrandProductCircuit<F>>) -> Self {
    Self {
      circuits,
      flags: None,
      flag_map: None
    }
  }

  pub fn new_batch_flags(
    circuits: Vec<GrandProductCircuit<F>>,
    flags: Vec<DensePolynomial<F>>,
    flag_map: Vec<usize>
  ) -> Self {
    assert_eq!(flag_map.len(), circuits.len());
    flag_map.iter().for_each(|i| assert!(*i < flags.len()));

    Self {
      circuits,
      flags: Some(flags),
      flag_map: Some(flag_map)
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

  fn sumcheck_layer_params(&self, layer_id: usize, eq: DensePolynomial<F>) -> CubicSumcheckParams<F> {
    if self.flags.is_some() && layer_id == 0 {
      let flags = self.flags.as_ref().unwrap();
      debug_assert_eq!(self.circuits[0].leaves.len(), flags[0].len());
      debug_assert_eq!(flags[0].len(), eq.len());

      let num_rounds = eq.get_num_vars();
      // TODO(sragss): Handle .as_ref().unwrap().clone() without cloning.
      CubicSumcheckParams::new_flags(
        self.circuits.iter().map(|circuit| circuit.leaves.clone()).collect(),
        self.flags.as_ref().unwrap().clone(),
        eq, 
        self.flag_map.as_ref().unwrap().clone(), 
        num_rounds
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
        self.circuits.iter().map(|circuit| 
          circuit.left_vec[layer_id].clone()
        ).collect(),
        self.circuits.iter().map(|circuit| 
          circuit.right_vec[layer_id].clone()
        ).collect(),
        eq,
        num_rounds
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
    println!("claims_to_verify {claims_to_verify:?}");

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
      let sumcheck_type = params.sumcheck_type.clone();
      let (proof, rand_prod, claims_prod) = 
        SumcheckInstanceProof::prove_cubic_batched_special::<G>(
          &claim, 
          params, 
          &coeff_vec, 
          transcript
        );

      let (claims_prod_left, claims_prod_right, _claims_eq) = claims_prod;
      for i in 0..batch.circuits.len() {
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

      claims_to_verify = (0..batch.circuits.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<F>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;

      let combine_prod = match sumcheck_type {
        CubicSumcheckType::Prod => true,
        CubicSumcheckType::Flags => false,
        _ => panic!("ruh roh")
      };

      proof_layers.push(LayerProofBatched {
        proof,
        claims_prod_left,
        claims_prod_right,
        combine_prod
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

      let claim_expected: F = if self.proof[i].combine_prod {
        println!("combine_prod");
        (0..claims_prod_vec.len())
          .map(|i| coeff_vec[i] * CubicSumcheckParams::combine_prod(
            &claims_prod_left[i], 
            &claims_prod_right[i],
             &eq))
          .sum()
      } else {
        println!("combine_flags");
        (0..claims_prod_vec.len())
          .map(|i| coeff_vec[i] * CubicSumcheckParams::combine_flags(
            &claims_prod_left[i], 
          &claims_prod_right[i], 
            &eq))
          .sum()
      };

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
    let factorial_circuit = GrandProductCircuit::new(factorial);
    let expected_eval = vec![Fr::from(24)];
    assert_eq!(factorial_circuit.evaluate(), Fr::from(24));

    let mut transcript = Transcript::new(b"test_transcript");
    let circuits_vec = vec![factorial_circuit];
    let batch = BatchedGrandProductCircuit::new_batch(circuits_vec);
    let (proof, _) =
      BatchedGrandProductArgument::prove::<G1Projective>(batch, &mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    proof.verify::<G1Projective, _>(&expected_eval, &mut transcript);
  }

  #[test]
  fn gp_read_only_trivial_e2e() {
    let eval_table = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
    let dim_i = DensePolynomial::new(vec![Fr::from(2), Fr::from(3)]);
    let dim_i_usize = vec![2, 3];
    let read_i = DensePolynomial::new(vec![Fr::from(0), Fr::from(0)]);
    let final_i = DensePolynomial::new(vec![Fr::from(0), Fr::from(0), Fr::from(1), Fr::from(1)]);
    let r_mem_check = (Fr::from(12), Fr::from(35));

    let flags = vec![DensePolynomial::new(vec![Fr::from(1), Fr::from(1)])];
    let flag_map = vec![0];

    let gps = GrandProducts::new_read_only(&eval_table, &dim_i, &dim_i_usize, &read_i, &final_i, &r_mem_check);

    let expected_eval_ops = vec![gps.read.evaluate(), gps.write.evaluate()];
    let expected_eval_mem = vec![gps.init.evaluate(), gps.r#final.evaluate()];

    let (rw_batch, if_batch) = 
      BatchedGrandProductCircuit::new_batches_flags(vec![gps], flags, flag_map);

    let mut transcript = Transcript::new(b"test_transcript");
    let (proof_ops, _rand_ops_sized_gps) =
      BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(rw_batch, &mut transcript);
    let (proof_mem, _rand_mem_sized_gps) =
      BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(if_batch, &mut transcript);

    let mut verify_transcript = Transcript::new(b"test_transcript");
    proof_ops.verify::<G1Projective, _>(&expected_eval_ops, &mut verify_transcript);
    proof_mem.verify::<G1Projective, _>(&expected_eval_mem, &mut verify_transcript);
  }

  #[test]
  fn gp_read_only_e2e() {
    let eval_table = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
    let dim_i = DensePolynomial::new(vec![Fr::from(2), Fr::from(3)]);
    let dim_i_usize = vec![2, 3];
    let read_i = DensePolynomial::new(vec![Fr::from(0), Fr::from(0)]);
    let final_i = DensePolynomial::new(vec![Fr::from(0), Fr::from(0), Fr::from(1), Fr::from(0)]);
    let r_mem_check = (Fr::from(12), Fr::from(35));

    let bool_flags: Vec<bool> = vec![true, false];
    let flags = vec![DensePolynomial::new(vec![Fr::from(1), Fr::from(0)])];
    let flag_map = vec![0];

    let gps = GrandProducts::new_read_only_with_flags(
      &eval_table, 
      &dim_i, 
      &dim_i_usize, 
      &read_i, 
      &final_i, 
      &bool_flags, 
      &r_mem_check);

    let expected_eval_ops = vec![gps.read.evaluate(), gps.write.evaluate()];
    let expected_eval_mem = vec![gps.init.evaluate(), gps.r#final.evaluate()];

    let (rw_batch, if_batch) = 
      BatchedGrandProductCircuit::new_batches_flags(vec![gps], flags, flag_map);

    let mut transcript = Transcript::new(b"test_transcript");
    let (proof_ops, _rand_ops_sized_gps) =
      BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(rw_batch, &mut transcript);
    let (proof_mem, _rand_mem_sized_gps) =
      BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(if_batch, &mut transcript);

    let mut verify_transcript = Transcript::new(b"test_transcript");
    proof_ops.verify::<G1Projective, _>(&expected_eval_ops, &mut verify_transcript);
    proof_mem.verify::<G1Projective, _>(&expected_eval_mem, &mut verify_transcript);
  }
}
