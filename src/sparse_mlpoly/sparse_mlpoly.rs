#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
use crate::dense_mlpoly::DensePolynomial;
use crate::dense_mlpoly::{
  EqPolynomial, IdentityPolynomial, PolyCommitment, PolyCommitmentGens, PolyEvalProof,
};
use crate::errors::ProofVerifyError;
use crate::math::Math;
use crate::product_tree::{DotProductCircuit, GrandProductCircuit, ProductCircuitEvalProofBatched};
use crate::random::RandomTape;
use crate::sparse_mlpoly::densified::DensifiedRepresentation;
use crate::sparse_mlpoly::derefs::{Derefs, DerefsCommitment, DerefsEvalProof};
use crate::timer::Timer;
use crate::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::{Field, PrimeField};
use ark_serialize::*;
use ark_std::{One, Zero};
use itertools::Itertools;
use std::convert::From;

use merlin::Transcript;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatEntry<F: PrimeField, const c: usize> {
  pub indices: [usize; c],
  pub val: F, // TODO(moodlezoup) always 1 for Lasso; delete?
}

impl<F: PrimeField, const c: usize> SparseMatEntry<F, c> {
  pub fn new(indices: [usize; c], val: F) -> Self {
    SparseMatEntry { indices, val }
  }
}

pub struct SparseMatPolyCommitmentGens<G> {
  pub gens_combined_l_variate: PolyCommitmentGens<G>,
  pub gens_combined_log_m_variate: PolyCommitmentGens<G>,
  pub gens_derefs: PolyCommitmentGens<G>,
}

impl<G: CurveGroup> SparseMatPolyCommitmentGens<G> {
  pub fn new(
    label: &'static [u8],
    c: usize,
    s: usize,
    log_m: usize,
  ) -> SparseMatPolyCommitmentGens<G> {
    // dim + read + val
    // log_2(cs + cs + s) = log_2(2cs + s)
    let num_vars_combined_l_variate = (2 * c * s + s).next_power_of_two().log_2();
    // final
    // log_2(c * m) = log_2(c) + log_2(m)
    let num_vars_combined_log_m_variate = c.next_power_of_two().log_2() + log_m;
    // TODO(moodlezoup)
    let num_vars_derefs = s.next_power_of_two().log_2() as usize + 1;

    let gens_combined_l_variate = PolyCommitmentGens::new(num_vars_combined_l_variate, label);
    let gens_combined_log_m_variate =
      PolyCommitmentGens::new(num_vars_combined_log_m_variate, label);
    let gens_derefs = PolyCommitmentGens::new(num_vars_derefs, label);
    SparseMatPolyCommitmentGens {
      gens_combined_l_variate: gens_combined_l_variate,
      gens_combined_log_m_variate: gens_combined_log_m_variate,
      gens_derefs,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparsePolynomialCommitment<G: CurveGroup> {
  pub l_variate_polys_commitment: PolyCommitment<G>,
  pub log_m_variate_polys_commitment: PolyCommitment<G>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize, // TODO: big integer
}

impl<G: CurveGroup> AppendToTranscript<G> for SparsePolynomialCommitment<G> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, _label: &'static [u8], transcript: &mut T) {
    self
      .l_variate_polys_commitment
      .append_to_transcript(b"l_variate_polys_commitment", transcript);
    self
      .log_m_variate_polys_commitment
      .append_to_transcript(b"log_m_variate_polys_commitment", transcript);
    transcript.append_u64(b"s", self.s as u64);
    transcript.append_u64(b"log_m", self.log_m as u64);
    transcript.append_u64(b"m", self.m as u64);
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatPolynomial<F: PrimeField, const c: usize> {
  pub nonzero_entries: Vec<SparseMatEntry<F, c>>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize, // TODO: big integer
}

impl<F: PrimeField, const c: usize> SparseMatPolynomial<F, c> {
  pub fn new(nonzero_entries: Vec<SparseMatEntry<F, c>>, log_m: usize) -> Self {
    let s = nonzero_entries.len().next_power_of_two();
    // TODO(moodlezoup):
    // nonzero_entries.resize(s, F::zero());

    SparseMatPolynomial {
      nonzero_entries,
      s,
      log_m,
      m: log_m.pow2(),
    }
  }

  pub fn evaluate(&self, r: &Vec<F>) -> F {
    assert_eq!(c * self.log_m, r.len());

    // \tilde{M}(r) = \sum_k [val(k) * \prod_i E_i(k)]
    // where E_i(k) = \tilde{eq}(to-bits(dim_i(k)), r_i)
    self
      .nonzero_entries
      .iter()
      .map(|entry| {
        r.chunks_exact(self.log_m)
          .enumerate()
          .map(|(i, r_i)| {
            let E_i = EqPolynomial::new(r_i.to_vec()).evals();
            E_i[entry.indices[i]]
          })
          .product::<F>()
          .mul(entry.val)
      })
      .sum()
  }

  fn to_densified(&self) -> DensifiedRepresentation<F, c> {
    // TODO(moodlezoup) Initialize as arrays using std::array::from_fn ?
    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(c);
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(c);
    let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(c);
    let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(c);

    for i in 0..c {
      let mut access_sequence = self
        .nonzero_entries
        .iter()
        .map(|entry| entry.indices[i])
        .collect::<Vec<usize>>();
      // TODO(moodlezoup) Is this resize necessary/in the right place?
      access_sequence.resize(self.s, 0usize);

      let mut final_timestamps = vec![0usize; self.m];
      let mut read_timestamps = vec![0usize; self.s];

      // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
      // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
      for i in 0..self.s {
        let memory_address = access_sequence[i];
        assert!(memory_address < self.m);
        let ts = final_timestamps[memory_address];
        read_timestamps[i] = ts;
        let write_timestamp = ts + 1;
        final_timestamps[memory_address] = write_timestamp;
      }

      dim.push(DensePolynomial::from_usize(&access_sequence));
      read.push(DensePolynomial::from_usize(&read_timestamps));
      r#final.push(DensePolynomial::from_usize(&final_timestamps));
      dim_usize.push(access_sequence);
    }

    let mut values: Vec<F> = self.nonzero_entries.iter().map(|entry| entry.val).collect();
    // TODO(moodlezoup) Is this resize necessary/in the right place?
    values.resize(self.s, F::zero());

    let val = DensePolynomial::new(values);

    let mut l_variate_polys = [dim.as_slice(), read.as_slice()].concat();
    l_variate_polys.push(val.clone());

    let combined_l_variate_polys = DensePolynomial::merge(&l_variate_polys);
    let combined_log_m_variate_polys = DensePolynomial::merge(&r#final);

    DensifiedRepresentation {
      dim_usize,
      dim: dim.try_into().unwrap(),
      read: read.try_into().unwrap(),
      r#final: r#final.try_into().unwrap(),
      val,
      combined_l_variate_polys,
      combined_log_m_variate_polys,
      s: self.s,
      log_m: self.log_m,
      m: self.m,
    }
  }
}

// impl<F: PrimeField> MultiSparseMatPolynomialAsDense<F> {
//   pub fn deref(&self, mem_vals: &Vec<Vec<F>>) -> Derefs<F> {
//     let ops_vals: Vec<_> = self
//       .dim
//       .iter()
//       .zip(mem_vals)
//       .map(|(&dim_i, mem_val)| dim_i.deref(&mem_val))
//       .collect();

//     Derefs::new(ops_vals)
//   }
// }

// TODO(moodlezoup): Combine init and write, read and final
#[derive(Debug)]
struct GrandProducts<F> {
  init: GrandProductCircuit<F>,
  read: GrandProductCircuit<F>,
  write: GrandProductCircuit<F>,
  r#final: GrandProductCircuit<F>,
}

impl<F: PrimeField> GrandProducts<F> {
  fn build_hash_layer(
    eval_table: &[F],
    dim_i: &DensePolynomial<F>,
    read_i: &DensePolynomial<F>,
    final_i: &DensePolynomial<F>,
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

    // hash init and audit
    assert_eq!(eval_table.len(), final_i.len());
    let num_mem_cells = eval_table.len();
    let grand_product_input_init = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // at init time, addr is given by i, init value is given by eval_table, and ts = 0
          hash_func(&F::from(i as u64), &eval_table[i], &F::zero())
        })
        .collect::<Vec<F>>(),
    );
    let grand_product_input_final = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // at audit time, addr is given by i, value is given by eval_table, and ts is given by audit_ts
          hash_func(&F::from(i as u64), &eval_table[i], &final_i[i])
        })
        .collect::<Vec<F>>(),
    );

    // hash read and write
    assert_eq!(dim_i.len(), read_i.len());
    let num_ops = dim_i.len();
    let grand_product_input_read = DensePolynomial::new(
      (0..num_ops)
        .map(|i| {
          // at read time, addr is given by dim_i, value is given by derefs, and ts is given by read_ts
          hash_func(&dim_i[i], &eval_table[i], &read_i[i])
        })
        .collect::<Vec<F>>(),
    );

    let grand_product_input_write = DensePolynomial::new(
      (0..num_ops)
        .map(|i| {
          // at write time, addr is given by dim_i, value is given by derefs, and ts is given by write_ts = read_ts + 1
          hash_func(&dim_i[i], &eval_table[i], &(read_i[i] + F::one()))
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

  pub fn new(
    eval_table: &[F],
    dim_i: &DensePolynomial<F>,
    read_i: &DensePolynomial<F>,
    final_i: &DensePolynomial<F>,
    r_mem_check: &(F, F),
  ) -> Self {
    let (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    ) = GrandProducts::build_hash_layer(eval_table, dim_i, read_i, final_i, r_mem_check);

    let prod_init = GrandProductCircuit::new(&grand_product_input_init);
    let prod_read = GrandProductCircuit::new(&grand_product_input_read);
    let prod_write = GrandProductCircuit::new(&grand_product_input_write);
    let prod_final = GrandProductCircuit::new(&grand_product_input_final);

    // TODO(moodlezoup): delete?
    let hashed_write_set: F = prod_init.evaluate() * prod_write.evaluate();
    let hashed_read_set: F = prod_read.evaluate() * prod_final.evaluate();
    assert_eq!(hashed_read_set, hashed_write_set);

    GrandProducts {
      init: prod_init,
      read: prod_read,
      write: prod_write,
      r#final: prod_final,
    }
  }
}

#[derive(Debug)]
struct PolyEvalNetwork<F, const c: usize> {
  grand_products: [GrandProducts<F>; c],
}

impl<F: PrimeField, const c: usize> PolyEvalNetwork<F, c> {
  pub fn new(
    dense: &DensifiedRepresentation<F, c>,
    mems: &Vec<Vec<F>>,
    r_mem_check: &(F, F),
  ) -> Self {
    let grand_products: [GrandProducts<F>; c] = std::array::from_fn(|i| {
      GrandProducts::new(
        &mems[i],
        &dense.dim[i],
        &dense.read[i],
        &dense.r#final[i],
        r_mem_check,
      )
    });
    PolyEvalNetwork { grand_products }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct HashLayerProof<G: CurveGroup, const c: usize> {
  eval_dim: Vec<(G::ScalarField, G::ScalarField, G::ScalarField)>,
  eval_val: G::ScalarField,
  eval_derefs: Vec<G::ScalarField>,
  proof_ops: PolyEvalProof<G>,
  proof_mem: PolyEvalProof<G>,
  proof_derefs: DerefsEvalProof<G>,
}

impl<G: CurveGroup, const c: usize> HashLayerProof<G, c> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial hash layer proof"
  }

  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    dense: &DensifiedRepresentation<G::ScalarField, c>,
    derefs: &Derefs<G::ScalarField>,
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      HashLayerProof::<G, c>::protocol_name(),
    );

    let (rand_mem, rand_ops) = rand;

    // decommit derefs at rand_ops
    let eval_derefs: Vec<G::ScalarField> = derefs
      .eq_evals
      .iter()
      .map(|eq| eq.evaluate::<G>(rand_ops))
      .collect();
    let proof_derefs = DerefsEvalProof::prove(
      derefs,
      &eval_derefs,
      rand_ops,
      &gens.gens_derefs,
      transcript,
      random_tape,
    );

    // form a single decommitment using comm_comb_ops
    let mut evals_ops: Vec<G::ScalarField> = Vec::new();
    let mut evals_final: Vec<G::ScalarField> = Vec::new();

    let mut eval_dim = Vec::new();
    for i in 0..c {
      let dim_eval = dense.dim[i].evaluate::<G>(rand_ops);
      let read_eval = dense.read[i].evaluate::<G>(rand_ops);
      let final_eval = dense.r#final[i].evaluate::<G>(rand_mem);
      eval_dim.push((dim_eval, read_eval, final_eval));
      evals_ops.push(dim_eval);
      evals_ops.push(read_eval);
      evals_final.push(final_eval);
    }

    let eval_val = dense.val.evaluate::<G>(rand_ops);

    evals_ops.push(eval_val);
    evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_ops", &evals_ops);

    let challenges_ops = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_n_to_one",
      evals_ops.len().log_2() as usize,
    );

    let mut poly_evals_ops = DensePolynomial::new(evals_ops);
    for i in (0..challenges_ops.len()).rev() {
      poly_evals_ops.bound_poly_var_bot(&challenges_ops[i]);
    }
    assert_eq!(poly_evals_ops.len(), 1);

    let joint_claim_eval_ops = poly_evals_ops[0];
    let mut r_joint_ops = challenges_ops;
    r_joint_ops.extend(rand_ops);
    debug_assert_eq!(
      dense.combined_l_variate_polys.evaluate::<G>(&r_joint_ops),
      joint_claim_eval_ops
    );

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_ops",
      &joint_claim_eval_ops,
    );

    let (proof_ops, _) = PolyEvalProof::prove(
      &dense.combined_l_variate_polys,
      None,
      &r_joint_ops,
      &joint_claim_eval_ops,
      None,
      &gens.gens_combined_l_variate,
      transcript,
      random_tape,
    );

    <Transcript as ProofTranscript<G>>::append_scalars(
      transcript,
      b"claim_evals_mem",
      &evals_final,
    );
    let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_two_to_one",
      evals_final.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new(evals_final);
    for i in (0..challenges_mem.len()).rev() {
      poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
    }
    assert_eq!(poly_evals_mem.len(), 1);

    let joint_claim_eval_mem = poly_evals_mem[0];
    let mut r_joint_mem = challenges_mem;
    r_joint_mem.extend(rand_mem);
    debug_assert_eq!(
      dense
        .combined_log_m_variate_polys
        .evaluate::<G>(&r_joint_mem),
      joint_claim_eval_mem
    );

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_mem",
      &joint_claim_eval_mem,
    );

    let (proof_mem, _) = PolyEvalProof::prove(
      &dense.combined_log_m_variate_polys,
      None,
      &r_joint_mem,
      &joint_claim_eval_mem,
      None,
      &gens.gens_combined_log_m_variate,
      transcript,
      random_tape,
    );

    HashLayerProof {
      eval_dim,
      eval_val,
      proof_ops,
      proof_mem,
      eval_derefs,
      proof_derefs,
    }
  }

  // fn verify_helper(
  //   rand_mem: &Vec<G::ScalarField>,
  //   claims: &(
  //     G::ScalarField,
  //     Vec<G::ScalarField>,
  //     Vec<G::ScalarField>,
  //     G::ScalarField,
  //   ),
  //   eval_ops_val: &[G::ScalarField],
  //   eval_ops_addr: &[G::ScalarField],
  //   eval_read_ts: &[G::ScalarField],
  //   eval_audit_ts: &G::ScalarField,
  //   r: &[G::ScalarField],
  //   gamma: &G::ScalarField,
  //   tau: &G::ScalarField,
  // ) -> Result<(), ProofVerifyError> {
  //   let hash_func = |a: &G::ScalarField,
  //                    v: &G::ScalarField,
  //                    t: &G::ScalarField|
  //    -> G::ScalarField { *t * gamma.square() + *v * *gamma + *a - tau };
  //   // moodlezoup: (t * gamma^2 + v * gamma + a) instead of (a * gamma^2 + v * gamma + t)

  //   let (claim_init, claim_read, claim_write, claim_audit) = claims;

  //   // init
  //   // moodlezoup: Collapses Init_row into a single field element
  //   // Spartan 7.2.3 #4
  //   let eval_init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem); // [0, 1, ..., m-1]
  //   let eval_init_val = EqPolynomial::new(r.to_vec()).evaluate(rand_mem); // [\tilde{eq}(0, r_x), \tilde{eq}(1, r_x), ..., \tilde{eq}(m-1, r_x)]
  //                                                                         // H_\gamma_1(a, v, t)
  //   let hash_init_at_rand_mem = hash_func(&eval_init_addr, &eval_init_val, &G::ScalarField::zero()); // verify the claim_last of init chunk
  //   assert_eq!(&hash_init_at_rand_mem, claim_init);

  //   // read
  //   for i in 0..eval_ops_addr.len() {
  //     let hash_read_at_rand_ops = hash_func(&eval_ops_addr[i], &eval_ops_val[i], &eval_read_ts[i]); // verify the claim_last of init chunk
  //     assert_eq!(&hash_read_at_rand_ops, &claim_read[i]);
  //   }

  //   // write: shares addr, val component; only decommit write_ts
  //   for i in 0..eval_ops_addr.len() {
  //     let eval_write_ts = eval_read_ts[i] + G::ScalarField::one();
  //     let hash_write_at_rand_ops = hash_func(&eval_ops_addr[i], &eval_ops_val[i], &eval_write_ts); // verify the claim_last of init chunk
  //     assert_eq!(&hash_write_at_rand_ops, &claim_write[i]);
  //   }

  //   // audit: shares addr and val with init
  //   let eval_audit_addr = eval_init_addr;
  //   let eval_audit_val = eval_init_val;
  //   let hash_audit_at_rand_mem = hash_func(&eval_audit_addr, &eval_audit_val, eval_audit_ts);
  //   assert_eq!(&hash_audit_at_rand_mem, claim_audit); // verify the last step of the sum-check for audit

  //   Ok(())
  // }

  // fn verify(
  //   &self,
  //   rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
  //   claims_dim: &Vec<(
  //     G::ScalarField,
  //     Vec<G::ScalarField>,
  //     Vec<G::ScalarField>,
  //     G::ScalarField,
  //   )>,
  //   claims_dotp: &[G::ScalarField],
  //   comm: &SparsePolynomialCommitment<G>,
  //   gens: &SparseMatPolyCommitmentGens<G>,
  //   comm_derefs: &DerefsCommitment<G>,
  //   r: &Vec<Vec<G::ScalarField>>,
  //   r_hash: &G::ScalarField,
  //   r_multiset_check: &G::ScalarField,
  //   transcript: &mut Transcript,
  // ) -> Result<(), ProofVerifyError> {
  //   let timer = Timer::new("verify_hash_proof");
  //   <Transcript as ProofTranscript<G>>::append_protocol_name(
  //     transcript,
  //     HashLayerProof::protocol_name(),
  //   );

  //   let (rand_mem, rand_ops) = rand;

  //   // struct HashLayerProof<G: CurveGroup> {
  //   //   eval_dim: Vec<(Vec<G::ScalarField>, Vec<G::ScalarField>, G::ScalarField)>,
  //   //                       [addr]            [read_ts]            audit_ts
  //   //   eval_val: Vec<G::ScalarField>,
  //   //   eval_derefs: Vec<Vec<G::ScalarField>>,
  //   //   proof_ops: PolyEvalProof<G>,
  //   //   proof_mem: PolyEvalProof<G>,
  //   //   proof_derefs: DerefsEvalProof<G>,
  //   // }

  //   // comm_derefs
  //   // self.eval_derefs = E_i
  //   // self.eval_dim
  //   //
  //   // claims_dim
  //   // claims_dotp

  //   // verify derefs at rand_ops
  //   // TODO(moodlezoup)
  //   // assert_eq!(eval_row_ops_val.len(), eval_col_ops_val.len());
  //   self.proof_derefs.verify(
  //     rand_ops,
  //     &self.eval_derefs,
  //     &gens.gens_derefs,
  //     comm_derefs,
  //     transcript,
  //   )?;

  //   // verify the decommitments used in evaluation sum-check
  //   let eval_val_vec = &self.eval_val;
  //   assert_eq!(claims_dotp.len(), 3 * eval_row_ops_val.len());
  //   for i in 0..claims_dotp.len() / 3 {
  //     let claim_row_ops_val = claims_dotp[3 * i];
  //     let claim_col_ops_val = claims_dotp[3 * i + 1];
  //     let claim_val = claims_dotp[3 * i + 2];

  //     assert_eq!(claim_row_ops_val, eval_row_ops_val[i]);
  //     assert_eq!(claim_col_ops_val, eval_col_ops_val[i]);
  //     assert_eq!(claim_val, eval_val_vec[i]);
  //   }

  //   // verify addr-timestamps using comm_comb_ops at rand_ops
  //   let (eval_row_addr_vec, eval_row_read_ts_vec, eval_row_audit_ts) = &self.eval_row;
  //   let (eval_col_addr_vec, eval_col_read_ts_vec, eval_col_audit_ts) = &self.eval_col;

  //   let mut evals_ops: Vec<G::ScalarField> = Vec::new();
  //   evals_ops.extend(eval_row_addr_vec);
  //   evals_ops.extend(eval_row_read_ts_vec);
  //   evals_ops.extend(eval_col_addr_vec);
  //   evals_ops.extend(eval_col_read_ts_vec);
  //   evals_ops.extend(eval_val_vec);
  //   evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

  //   <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_ops", &evals_ops);

  //   let challenges_ops = <Transcript as ProofTranscript<G>>::challenge_vector(
  //     transcript,
  //     b"challenge_combine_n_to_one",
  //     evals_ops.len().log_2() as usize,
  //   );

  //   let mut poly_evals_ops = DensePolynomial::new(evals_ops);
  //   for i in (0..challenges_ops.len()).rev() {
  //     poly_evals_ops.bound_poly_var_bot(&challenges_ops[i]);
  //   }
  //   assert_eq!(poly_evals_ops.len(), 1);
  //   let joint_claim_eval_ops = poly_evals_ops[0];
  //   let mut r_joint_ops = challenges_ops;
  //   r_joint_ops.extend(rand_ops);
  //   <Transcript as ProofTranscript<G>>::append_scalar(
  //     transcript,
  //     b"joint_claim_eval_ops",
  //     &joint_claim_eval_ops,
  //   );
  //   self.proof_ops.verify_plain(
  //     &gens.gens_ops,
  //     transcript,
  //     &r_joint_ops,
  //     &joint_claim_eval_ops,
  //     &comm.comm_comb_ops,
  //   )?;

  //   // verify proof-mem using comm_comb_mem at rand_mem
  //   // form a single decommitment using comm_comb_mem at rand_mem
  //   let evals_mem: Vec<G::ScalarField> = vec![*eval_row_audit_ts, *eval_col_audit_ts];
  //   <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_mem", &evals_mem);
  //   let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
  //     transcript,
  //     b"challenge_combine_two_to_one",
  //     evals_mem.len().log_2() as usize,
  //   );

  //   let mut poly_evals_mem = DensePolynomial::new(evals_mem);
  //   for i in (0..challenges_mem.len()).rev() {
  //     poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
  //   }
  //   assert_eq!(poly_evals_mem.len(), 1);
  //   let joint_claim_eval_mem = poly_evals_mem[0];
  //   let mut r_joint_mem = challenges_mem;
  //   r_joint_mem.extend(rand_mem);
  //   <Transcript as ProofTranscript<G>>::append_scalar(
  //     transcript,
  //     b"joint_claim_eval_mem",
  //     &joint_claim_eval_mem,
  //   );
  //   self.proof_mem.verify_plain(
  //     &gens.gens_mem,
  //     transcript,
  //     &r_joint_mem,
  //     &joint_claim_eval_mem,
  //     &comm.comm_comb_mem,
  //   )?;

  //   // // verify the claims from the product layer

  //   // for (claims) in claims_dim.iter().zip() {
  //   //   let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_row;
  //   //   HashLayerProof::<G>::verify_helper(
  //   //     &(rand_mem, rand_ops),
  //   //     claims,
  //   //     eval_row_ops_val,
  //   //     eval_ops_addr,
  //   //     eval_read_ts,
  //   //     eval_audit_ts,
  //   //     r,
  //   //     r_hash,
  //   //     r_multiset_check,
  //   //   )?;
  //   // }
  //   let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_row;
  //   HashLayerProof::<G>::verify_helper(
  //     &(rand_mem, rand_ops),
  //     claims_row,
  //     eval_row_ops_val,
  //     eval_ops_addr,
  //     eval_read_ts,
  //     eval_audit_ts,
  //     rx,
  //     r_hash,
  //     r_multiset_check,
  //   )?;

  //   let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_col;
  //   HashLayerProof::<G>::verify_helper(
  //     &(rand_mem, rand_ops),
  //     claims_col,
  //     eval_col_ops_val,
  //     eval_ops_addr,
  //     eval_read_ts,
  //     eval_audit_ts,
  //     ry,
  //     r_hash,
  //     r_multiset_check,
  //   )?;

  //   timer.stop();
  //   Ok(())
  // }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct ProductLayerProof<F: PrimeField, const c: usize> {
  grand_product_evals: [(F, F, F, F); c],
  eval_val: (Vec<F>, Vec<F>),
  proof_mem: ProductCircuitEvalProofBatched<F>,
  proof_ops: ProductCircuitEvalProofBatched<F>,
}

impl<F: PrimeField, const c: usize> ProductLayerProof<F, c> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial product layer proof"
  }

  pub fn prove<G>(
    grand_products: &[GrandProducts<F>; c],
    dense: &DensifiedRepresentation<F, c>,
    derefs: &Derefs<F>,
    eval: &F,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    todo!("unimpl");
    // <Transcript as ProofTranscript<G>>::append_protocol_name(
    //   transcript,
    //   ProductLayerProof::protocol_name(),
    // );

    // std::array::from_fn(|i| i);

    // for (i, grand_product) in grand_products.iter().enumerate() {
    //   let hash_init = grand_product.init.evaluate();
    //   let hash_final = grand_product.r#final.evaluate();
    //   let hash_read = grand_product.read.evaluate();
    //   let hash_write = grand_product.write.evaluate();

    //   assert_eq!(hash_init * hash_write, hash_read * hash_final);

    // TODO(moodlezoup)
    // <Transcript as ProofTranscript<G>>::append_scalar(
    //   transcript,
    //   b"claim_row_eval_init",
    //   &dim_eval_init,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalars(
    //   transcript,
    //   b"claim_row_eval_read",
    //   &dim_eval_read,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalars(
    //   transcript,
    //   b"claim_row_eval_write",
    //   &dim_eval_write,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalar(
    //   transcript,
    //   b"claim_row_eval_audit",
    //   &dim_eval_audit,
    // );
    // }

    // TODO(moodlezoup)
    // prepare dotproduct circuit for batching them with ops-related product circuits
    // derefs
    //   .ops_vals
    //   .iter()
    //   .for_each(|ops| assert_eq!(eval.len(), ops.len()));
    // assert_eq!(eval.len(), dense.val.len());
    // let mut dotp_circuit_left_vec: Vec<DotProductCircuit<F>> = Vec::new();
    // let mut dotp_circuit_right_vec: Vec<DotProductCircuit<F>> = Vec::new();
    // let mut eval_dotp_left_vec: Vec<F> = Vec::new();
    // let mut eval_dotp_right_vec: Vec<F> = Vec::new();
    // for i in 0..derefs.row_ops_val.len() {
    //   // evaluate sparse polynomial evaluation using two dotp checks
    //   let left = derefs.row_ops_val[i].clone();
    //   let right = derefs.col_ops_val[i].clone();
    //   let weights = dense.val[i].clone();

    //   // build two dot product circuits to prove evaluation of sparse polynomial
    //   let mut dotp_circuit = DotProductCircuit::new(left, right, weights);
    //   let (dotp_circuit_left, dotp_circuit_right) = dotp_circuit.split();

    //   let (eval_dotp_left, eval_dotp_right) =
    //     (dotp_circuit_left.evaluate(), dotp_circuit_right.evaluate());

    //   <Transcript as ProofTranscript<G>>::append_scalar(
    //     transcript,
    //     b"claim_eval_dotp_left",
    //     &eval_dotp_left,
    //   );

    //   <Transcript as ProofTranscript<G>>::append_scalar(
    //     transcript,
    //     b"claim_eval_dotp_right",
    //     &eval_dotp_right,
    //   );

    //   assert_eq!(eval_dotp_left + eval_dotp_right, eval[i]);
    //   eval_dotp_left_vec.push(eval_dotp_left);
    //   eval_dotp_right_vec.push(eval_dotp_right);

    //   dotp_circuit_left_vec.push(dotp_circuit_left);
    //   dotp_circuit_right_vec.push(dotp_circuit_right);
    // }

    // // The number of operations into the memory encoded by rx and ry are always the same (by design)
    // // So we can produce a batched product proof for all of them at the same time.
    // // prove the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write
    // // TODO: we currently only produce proofs for 3 batched sparse polynomial evaluations
    // assert_eq!(row_prod_layer.read_vec.len(), 3);
    // let (row_read_A, row_read_B, row_read_C) = {
    //   let (vec_A, vec_BC) = row_prod_layer.read_vec.split_at_mut(1);
    //   let (vec_B, vec_C) = vec_BC.split_at_mut(1);
    //   (vec_A, vec_B, vec_C)
    // };

    // let (row_write_A, row_write_B, row_write_C) = {
    //   let (vec_A, vec_BC) = row_prod_layer.write_vec.split_at_mut(1);
    //   let (vec_B, vec_C) = vec_BC.split_at_mut(1);
    //   (vec_A, vec_B, vec_C)
    // };

    // let (col_read_A, col_read_B, col_read_C) = {
    //   let (vec_A, vec_BC) = col_prod_layer.read_vec.split_at_mut(1);
    //   let (vec_B, vec_C) = vec_BC.split_at_mut(1);
    //   (vec_A, vec_B, vec_C)
    // };

    // let (col_write_A, col_write_B, col_write_C) = {
    //   let (vec_A, vec_BC) = col_prod_layer.write_vec.split_at_mut(1);
    //   let (vec_B, vec_C) = vec_BC.split_at_mut(1);
    //   (vec_A, vec_B, vec_C)
    // };

    // let (dotp_left_A, dotp_left_B, dotp_left_C) = {
    //   let (vec_A, vec_BC) = dotp_circuit_left_vec.split_at_mut(1);
    //   let (vec_B, vec_C) = vec_BC.split_at_mut(1);
    //   (vec_A, vec_B, vec_C)
    // };

    // let (dotp_right_A, dotp_right_B, dotp_right_C) = {
    //   let (vec_A, vec_BC) = dotp_circuit_right_vec.split_at_mut(1);
    //   let (vec_B, vec_C) = vec_BC.split_at_mut(1);
    //   (vec_A, vec_B, vec_C)
    // };

    // let (proof_ops, rand_ops) = ProductCircuitEvalProofBatched::<F>::prove::<G>(
    //   &mut vec![
    //     &mut row_read_A[0],
    //     &mut row_read_B[0],
    //     &mut row_read_C[0],
    //     &mut row_write_A[0],
    //     &mut row_write_B[0],
    //     &mut row_write_C[0],
    //     &mut col_read_A[0],
    //     &mut col_read_B[0],
    //     &mut col_read_C[0],
    //     &mut col_write_A[0],
    //     &mut col_write_B[0],
    //     &mut col_write_C[0],
    //   ],
    //   // &mut vec![
    //   //   &mut dotp_left_A[0],
    //   //   &mut dotp_right_A[0],
    //   //   &mut dotp_left_B[0],
    //   //   &mut dotp_right_B[0],
    //   //   &mut dotp_left_C[0],
    //   //   &mut dotp_right_C[0],
    //   // ],
    //   &mut Vec::new(),
    //   transcript,
    // );

    // // produce a batched proof of memory-related product circuits
    // let (proof_mem, rand_mem) = ProductCircuitEvalProofBatched::<F>::prove::<G>(
    //   &mut vec![
    //     &mut row_prod_layer.init,
    //     &mut row_prod_layer.audit,
    //     &mut col_prod_layer.init,
    //     &mut col_prod_layer.audit,
    //   ],
    //   &mut Vec::new(),
    //   transcript,
    // );

    // let product_layer_proof = ProductLayerProof {
    //   grand_product_evals,
    //   eval_val: (eval_dotp_left_vec, eval_dotp_right_vec),
    //   proof_mem,
    //   proof_ops,
    // };

    // let mut product_layer_proof_encoded = vec![];
    // product_layer_proof
    //   .serialize_compressed(&mut product_layer_proof_encoded)
    //   .unwrap();

    // (product_layer_proof, rand_mem, rand_ops)
  }

  pub fn verify<G>(
    &self,
    num_ops: usize,
    num_cells: usize,
    eval: &[F],
    transcript: &mut Transcript,
  ) -> Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
  where
    G: CurveGroup<ScalarField = F>,
  {
    // <Transcript as ProofTranscript<G>>::append_protocol_name(
    //   transcript,
    //   ProductLayerProof::protocol_name(),
    // );

    // let timer = Timer::new("verify_prod_proof");
    // let num_instances = eval.len();

    // // subset check
    // let (row_eval_init, row_eval_read, row_eval_write, row_eval_audit) = &self.eval_row;
    // assert_eq!(row_eval_write.len(), num_instances);
    // assert_eq!(row_eval_read.len(), num_instances);
    // let ws: F = (0..row_eval_write.len())
    //   .map(|i| row_eval_write[i])
    //   .product();
    // let rs: F = (0..row_eval_read.len()).map(|i| row_eval_read[i]).product();
    // assert_eq!(*row_eval_init * ws, rs * row_eval_audit);

    // <Transcript as ProofTranscript<G>>::append_scalar(
    //   transcript,
    //   b"claim_row_eval_init",
    //   row_eval_init,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalars(
    //   transcript,
    //   b"claim_row_eval_read",
    //   row_eval_read,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalars(
    //   transcript,
    //   b"claim_row_eval_write",
    //   row_eval_write,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalar(
    //   transcript,
    //   b"claim_row_eval_audit",
    //   row_eval_audit,
    // );

    // // subset check
    // let (col_eval_init, col_eval_read, col_eval_write, col_eval_audit) = &self.eval_col;
    // assert_eq!(col_eval_write.len(), num_instances);
    // assert_eq!(col_eval_read.len(), num_instances);
    // let ws: F = (0..col_eval_write.len())
    //   .map(|i| col_eval_write[i])
    //   .product();
    // let rs: F = (0..col_eval_read.len()).map(|i| col_eval_read[i]).product();
    // assert_eq!(*col_eval_init * ws, rs * col_eval_audit);

    // <Transcript as ProofTranscript<G>>::append_scalar(
    //   transcript,
    //   b"claim_col_eval_init",
    //   col_eval_init,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalars(
    //   transcript,
    //   b"claim_col_eval_read",
    //   col_eval_read,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalars(
    //   transcript,
    //   b"claim_col_eval_write",
    //   col_eval_write,
    // );
    // <Transcript as ProofTranscript<G>>::append_scalar(
    //   transcript,
    //   b"claim_col_eval_audit",
    //   col_eval_audit,
    // );

    // // // verify the evaluation of the sparse polynomial
    // // let (eval_dotp_left, eval_dotp_right) = &self.eval_val;
    // // assert_eq!(eval_dotp_left.len(), eval_dotp_left.len());
    // // assert_eq!(eval_dotp_left.len(), num_instances);
    // // let mut claims_dotp_circuit: Vec<F> = Vec::new();
    // // for i in 0..num_instances {
    // //   assert_eq!(eval_dotp_left[i] + eval_dotp_right[i], eval[i]);

    // //   <Transcript as ProofTranscript<G>>::append_scalar(
    // //     transcript,
    // //     b"claim_eval_dotp_left",
    // //     &eval_dotp_left[i],
    // //   );

    // //   <Transcript as ProofTranscript<G>>::append_scalar(
    // //     transcript,
    // //     b"claim_eval_dotp_right",
    // //     &eval_dotp_right[i],
    // //   );

    // //   claims_dotp_circuit.push(eval_dotp_left[i]);
    // //   claims_dotp_circuit.push(eval_dotp_right[i]);
    // // }

    // // verify the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write
    // let mut claims_prod_circuit: Vec<F> = Vec::new();
    // claims_prod_circuit.extend(row_eval_read);
    // claims_prod_circuit.extend(row_eval_write);
    // claims_prod_circuit.extend(col_eval_read);
    // claims_prod_circuit.extend(col_eval_write);

    // let (claims_ops, claims_dotp, rand_ops) = self.proof_ops.verify::<G>(
    //   &claims_prod_circuit,
    //   // &claims_dotp_circuit,
    //   &Vec::new(),
    //   num_ops,
    //   transcript,
    // );
    // // verify the correctness of claim_row_eval_init and claim_row_eval_audit
    // let (claims_mem, _claims_mem_dotp, rand_mem) = self.proof_mem.verify::<G>(
    //   &[
    //     *row_eval_init,
    //     *row_eval_audit,
    //     *col_eval_init,
    //     *col_eval_audit,
    //   ],
    //   &Vec::new(),
    //   num_cells,
    //   transcript,
    // );
    // timer.stop();

    // Ok((claims_mem, rand_mem, claims_ops, claims_dotp, rand_ops))

    Err(ProofVerifyError::InternalError)
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PolyEvalNetworkProof<G: CurveGroup, const c: usize> {
  proof_prod_layer: ProductLayerProof<G::ScalarField, c>,
  proof_hash_layer: HashLayerProof<G, c>,
}

impl<G: CurveGroup, const c: usize> PolyEvalNetworkProof<G, c> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }

  pub fn prove(
    network: &mut PolyEvalNetwork<G::ScalarField, c>,
    dense: &DensifiedRepresentation<G::ScalarField, c>,
    derefs: &Derefs<G::ScalarField>,
    eval: &G::ScalarField,
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    // <Transcript as ProofTranscript<G>>::append_protocol_name(
    //   transcript,
    //   PolyEvalNetworkProof::protocol_name(),
    // );

    // let (proof_prod_layer, rand_mem, rand_ops) = ProductLayerProof::prove::<G>(
    //   &mut network
    //     .layers_by_dimension
    //     .iter()
    //     .map(|&layer| layer.prod_layer)
    //     .collect(),
    //   dense,
    //   derefs,
    //   eval,
    //   transcript,
    // );

    // // proof of hash layer for row and col
    // let proof_hash_layer = HashLayerProof::prove(
    //   (&rand_mem, &rand_ops),
    //   dense,
    //   derefs,
    //   gens,
    //   transcript,
    //   random_tape,
    // );

    // PolyEvalNetworkProof {
    //   proof_prod_layer,
    //   proof_hash_layer,
    // }

    todo!("unimpl!")
  }

  pub fn verify(
    &self,
    comm: &SparsePolynomialCommitment<G>,
    comm_derefs: &DerefsCommitment<G>,
    eval: &G::ScalarField,
    gens: &SparseMatPolyCommitmentGens<G>,
    r: &Vec<G::ScalarField>,
    r_mem_check: &(G::ScalarField, G::ScalarField),
    nz: usize,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // let timer = Timer::new("verify_polyeval_proof");
    // <Transcript as ProofTranscript<G>>::append_protocol_name(
    //   transcript,
    //   PolyEvalNetworkProof::protocol_name(),
    // );

    // let (r_hash, r_multiset_check) = r_mem_check;

    // let num_ops = nz.next_power_of_two();
    // let num_cells = rx.len().pow2();
    // assert_eq!(rx.len(), ry.len());

    // let (claims_mem, rand_mem, mut claims_ops, claims_dotp, rand_ops) = self
    //   .proof_prod_layer
    //   .verify::<G>(num_ops, num_cells, eval, transcript)?;
    // assert_eq!(claims_mem.len(), 4);
    // assert_eq!(claims_ops.len(), 4 * num_instances);
    // // TODO(moodlezoup)
    // // assert_eq!(claims_dotp.len(), 3 * num_instances);

    // let (claims_ops_row, claims_ops_col) = claims_ops.split_at_mut(2 * num_instances);
    // let (claims_ops_row_read, claims_ops_row_write) = claims_ops_row.split_at_mut(num_instances);
    // let (claims_ops_col_read, claims_ops_col_write) = claims_ops_col.split_at_mut(num_instances);

    // // verify the proof of hash layer
    // self.proof_hash_layer.verify(
    //   (&rand_mem, &rand_ops),
    //   &(
    //     claims_mem[0],
    //     claims_ops_row_read.to_vec(),
    //     claims_ops_row_write.to_vec(),
    //     claims_mem[1],
    //   ),
    //   // &(
    //   //   claims_mem[2],
    //   //   claims_ops_col_read.to_vec(),
    //   //   claims_ops_col_write.to_vec(),
    //   //   claims_mem[3],
    //   // ),
    //   &claims_dotp,
    //   comm,
    //   gens,
    //   comm_derefs,
    //   r,
    //   r_hash,
    //   r_multiset_check,
    //   transcript,
    // )?;
    // timer.stop();

    Ok(())
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparsePolynomialEvaluationProof<G: CurveGroup, const c: usize> {
  comm_derefs: DerefsCommitment<G>,
  poly_eval_network_proof: PolyEvalNetworkProof<G, c>,
}

impl<G: CurveGroup, const c: usize> SparsePolynomialEvaluationProof<G, c> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }
  /// Prove an opening of the Sparse Matrix Polynomial
  /// - `dense`: SparseMatPolynomialAsDense
  /// - `r`: c log_m sized coordinates at which to prove the evaluation of the sparse polynomial
  /// - `eval`: evaluation of \widetilde{M}(r = (r_0, ..., r_logM))
  /// - `gens`: Commitment generator
  pub fn prove(
    dense: &DensifiedRepresentation<G::ScalarField, c>,
    r: &Vec<Vec<G::ScalarField>>, // 'log-m' sized point at which the polynomial is evaluated across 'c' dimensions
    eval: &G::ScalarField,        // a evaluation of \widetilde{M}(r = (r_0, ..., r_logM))
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    assert_eq!(r.len(), c);
    r.iter()
      .for_each(|dimensional_coordinate| assert_eq!(dimensional_coordinate.len(), dense.log_m));

    // Create an \widetilde{eq}(r) polynomial for each dimension, which we will memory check
    let eqs: Vec<Vec<G::ScalarField>> = r
      .iter()
      .map(|r_dim| {
        let eq_evals = EqPolynomial::new(r_dim.clone()).evals();
        assert_eq!(eq_evals.len(), dense.m);
        eq_evals
      })
      .collect();

    // eqs are the evaluations of eq(i_0, r_0) , eq(i_1, r_1) , ... , eq(i_c, r_c)
    // Where i_0, ... i_c are all \in {0, 1}^logM for the non-sparse indices (s)-sized
    // And r_0, ... r_c are all \in F^logM
    // Derefs converts each eqs into E_{r_i}
    let derefs = dense.deref(&eqs);
    // assert_eq!(derefs.len(), dense.s);

    // commit to non-deterministic choices of the prover
    let timer_commit = Timer::new("commit_nondet_witness");
    let comm_derefs = {
      let comm = derefs.commit(&gens.gens_derefs);
      comm.append_to_transcript(b"comm_poly_row_col_ops_val", transcript);
      comm
    };
    timer_commit.stop();

    let poly_eval_network_proof = {
      // produce a random element from the transcript for hash function
      let r_hash_params: Vec<G::ScalarField> =
        <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

      // build a network to evaluate the sparse polynomial
      let timer_build_network = Timer::new("build_layered_network");

      let mut net: PolyEvalNetwork<G::ScalarField, c> =
        PolyEvalNetwork::new(dense, &eqs, &(r_hash_params[0], r_hash_params[1]));
      timer_build_network.stop();

      let timer_eval_network = Timer::new("evalproof_layered_network");
      let poly_eval_network_proof = PolyEvalNetworkProof::prove(
        &mut net,
        dense,
        &derefs,
        eval,
        gens,
        transcript,
        random_tape,
      );
      timer_eval_network.stop();

      poly_eval_network_proof
    };

    Self {
      comm_derefs,
      poly_eval_network_proof,
    }
  }

  pub fn verify(
    &self,
    commitment: &SparsePolynomialCommitment<G>,
    r: &Vec<G::ScalarField>,     // point at which the polynomial is evaluated
    evaluation: &G::ScalarField, // evaluation of \widetilde{M}(r = (rx,ry))
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      SparsePolynomialEvaluationProof::<G, c>::protocol_name(),
    );

    assert_eq!(r.len(), c * commitment.log_m);

    // add claims to transcript and obtain challenges for randomized mem-check circuit
    self
      .comm_derefs
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    // produce a random element from the transcript for hash function
    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

    self.poly_eval_network_proof.verify(
      commitment,
      &self.comm_derefs,
      evaluation,
      gens,
      r,
      &(r_mem_check[0], r_mem_check[1]),
      commitment.s,
      transcript,
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bls12_381::{Fr, G1Projective};
  use ark_std::rand::RngCore;
  use ark_std::test_rng;
  use ark_std::UniformRand;

  use crate::utils::{ff_bitvector_dbg, index_to_field_bitvector};

  #[test]
  fn check_evaluation() {
    check_evaluation_helper::<G1Projective>()
  }
  fn check_evaluation_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    // parameters
    let num_entries: usize = 64;
    let s: usize = 8;
    const c: usize = 3;
    let log_m: usize = num_entries.log_2() / c; // 2
    let m: usize = log_m.pow2(); // 2 ^ 2 = 4

    let mut nonzero_entries: Vec<SparseMatEntry<G::ScalarField, c>> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
      ];
      let entry = SparseMatEntry::new(indices, G::ScalarField::rand(&mut prng));
      println!("{:?}", entry);
      nonzero_entries.push(entry);
    }

    let sparse_poly = SparseMatPolynomial::new(nonzero_entries, log_m);
    let gens = SparseMatPolyCommitmentGens::<G>::new(b"gens_sparse_poly", c, s, log_m);

    // evaluation
    let r: Vec<G::ScalarField> = (0..c * log_m)
      .map(|_| G::ScalarField::rand(&mut prng))
      .collect();
    let evaluation = sparse_poly.evaluate(&r);
    // println!("r: {:?}", r);
    // println!("eval: {}", eval);

    let dense: DensifiedRepresentation<G::ScalarField, c> = sparse_poly.into();
    // for i in 0..c {
    //   println!("i: {:?}", i);
    //   println!("dim: {:?}", dense.dim[i]);
    //   println!("read: {:?}", dense.read[i]);
    //   println!("final: {:?}\n", dense.r#final[i]);
    // }
    // println!("val: {:?}", dense.val);

    // // dim + read + val => log2((2c + 1) * s)
    // println!(
    //   "combined l-variate multilinear polynomial has {} variables",
    //   dense.combined_l_variate_polys.get_num_vars()
    // );
    // // final => log2(c * m)
    // println!(
    //   "combined log(m)-variate multilinear polynomial has {} variables",
    //   dense.combined_log_m_variate_polys.get_num_vars()
    // );

    let (gens, commitment) = dense.commit::<G>();

    let mut random_tape = RandomTape::<G>::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    // let proof = SparsePolynomialEvaluationProof::prove(
    //   &dense,
    //   &r,
    //   &evaluation,
    //   &gens,
    //   &mut prover_transcript,
    //   &mut random_tape,
    // );

    // let mut verifier_transcript = Transcript::new(b"example");
    // assert!(proof
    //   .verify(&commitment, &r, &evals, &gens, &mut verifier_transcript)
    //   .is_ok());
  }

  // #[test]
  fn check_sparse_polyeval_proof() {
    check_sparse_polyeval_proof_helper::<G1Projective>()
  }
  fn check_sparse_polyeval_proof_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    // parameters
    let num_entries: usize = 256 * 256;
    let s: usize = 256;
    const c: usize = 4;
    let log_m: usize = num_entries.log_2() / c; // 4
    let m: usize = log_m.pow2(); // 2 ^ 4 = 16

    // generate sparse polynomial
    let mut nonzero_entries: Vec<SparseMatEntry<G::ScalarField, c>> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
      ];
      let entry = SparseMatEntry::new(indices, G::ScalarField::rand(&mut prng));
      nonzero_entries.push(entry);
    }

    let sparse_poly = SparseMatPolynomial::new(nonzero_entries, log_m);

    // evaluation
    let r: Vec<G::ScalarField> = (0..c * log_m)
      .map(|_| G::ScalarField::rand(&mut prng))
      .collect();
    let eval = sparse_poly.evaluate(&r);

    // commitment
    let dense: DensifiedRepresentation<G::ScalarField, c> = sparse_poly.into();
    let (gens, commitment) = dense.commit::<G>();

    // let mut random_tape = RandomTape::new(b"proof");
    // let mut prover_transcript = Transcript::new(b"example");
    // let proof = SparseMatPolyEvalProof::prove(
    //   &dense,
    //   &r,
    //   &evals,
    //   &gens,
    //   &mut prover_transcript,
    //   &mut random_tape,
    // );

    // let mut verifier_transcript = Transcript::new(b"example");
    // assert!(proof
    //   .verify(&commitment, &r, &evals, &gens, &mut verifier_transcript)
    //   .is_ok());
  }

  /// Construct a 2d sparse integer matrix like the following:
  /// ```
  ///     let M: Vec<usize> = vec! [
  ///         0, 0, 0, 0,
  ///         2, 0, 4, 0,
  ///         0, 8, 0, 9,
  ///         0, 0, 0, 0
  ///    ];
  /// ```
  fn construct_2d_sparse_mat_polynomial_from_ints<F: PrimeField>(
    ints: Vec<usize>,
    m: usize,
    log_m: usize,
    s: usize,
  ) -> SparseMatPolynomial<F, 2> {
    assert_eq!(m, log_m.pow2());
    let mut row_index = 0usize;
    let mut column_index = 0usize;
    let mut sparse_evals: Vec<SparseMatEntry<F, 2>> = Vec::new();
    for entry_index in 0..ints.len() {
      if ints[entry_index] != 0 {
        println!(
          "Non-sparse: (row, col, val): ({row_index}, {column_index}, {})",
          ints[entry_index]
        );
        sparse_evals.push(SparseMatEntry::new(
          [row_index, column_index],
          F::from(ints[entry_index] as u64),
        ));
      }

      column_index += 1;
      if column_index >= m {
        column_index = 0;
        row_index += 1;
      }
    }

    SparseMatPolynomial::<F, 2>::new(sparse_evals, log_m)
  }

  /// Returns a tuple of (c, s, m, log_m, SparsePoly)
  fn construct_2d_small<G: CurveGroup>() -> (
    usize,
    usize,
    usize,
    usize,
    SparseMatPolynomial<G::ScalarField, 2>,
  ) {
    let c = 2usize;
    let s = 4usize;
    let m = 4usize;
    let log_m = 2usize;

    let M: Vec<usize> = vec![0, 0, 0, 0, 2, 0, 4, 0, 0, 8, 0, 9, 0, 0, 0, 0];
    (
      c,
      s,
      m,
      log_m,
      construct_2d_sparse_mat_polynomial_from_ints(M, m, log_m, s),
    )
  }

  #[test]
  fn evaluate_over_known_indices() {
    // Create SparseMLE and then evaluate over known indices and confirm correct evaluations
    let (c, s, m, log_m, sparse_poly) = construct_2d_small::<G1Projective>();

    // Evaluations
    // poly[row, col] = eval
    // poly[1, 0] = 2
    // poly[1, 2] = 4
    // poly[2, 1] = 8
    // poly[2, 3] = 9
    // Check each and a few others over the boolean hypercube to be 0

    // poly[1, 0] = 2
    let row: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(0, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(2));

    // poly[1, 2] = 4
    let row: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(4));

    // poly[2, 1] = 8
    let row: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(8));

    // poly[2, 3] = 9
    let row: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(3, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(9));
  }

  #[test]
  fn prove() {
    let mut prng = test_rng();
    const c: usize = 2;

    let (_, s, m, log_m, sparse_poly) = construct_2d_small::<G1Projective>();

    // Commit
    let dense: DensifiedRepresentation<Fr, c> = sparse_poly.to_densified();
    let (gens, commitment) = dense.commit();

    // Eval
    let mut r: Vec<Vec<Fr>> = Vec::new();
    for dim in 0..(c) {
      let mut dimension: Vec<Fr> = Vec::with_capacity(log_m);
      for i in 0..log_m {
        dimension.push(Fr::rand(&mut prng));
      }
      r.push(dimension);
    }
    let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();
    let eval = sparse_poly.evaluate(&flat_r);

    // Prove
    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, c>::prove(
      &dense,
      &r,
      &eval,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );
  }
}
