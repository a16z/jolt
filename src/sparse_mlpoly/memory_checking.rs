use crate::dense_mlpoly::DensePolynomial;
use crate::dense_mlpoly::PolyEvalProof;
use crate::errors::ProofVerifyError;
use crate::math::Math;
use crate::product_tree::{
  BatchedGrandProductArgument, GeneralizedScalarProduct, GrandProductCircuit,
};
use crate::random::RandomTape;
use crate::sparse_mlpoly::densified::DensifiedRepresentation;
use crate::sparse_mlpoly::derefs::{Derefs, DerefsCommitment, DerefsEvalProof};
use crate::sparse_mlpoly::sparse_mlpoly::{
  SparseMatPolyCommitmentGens, SparsePolynomialCommitment,
};
use crate::transcript::ProofTranscript;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use ark_std::Zero;

use merlin::Transcript;

// TODO(moodlezoup): Combine init and write, read and final
#[derive(Debug)]
pub struct GrandProducts<F> {
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

impl<F: PrimeField, const c: usize> DensifiedRepresentation<F, c> {
  pub fn to_grand_products(
    &self,
    mems: &Vec<Vec<F>>,
    r_mem_check: &(F, F),
  ) -> [GrandProducts<F>; c] {
    std::array::from_fn(|i| {
      GrandProducts::new(
        &mems[i],
        &self.dim[i],
        &self.read[i],
        &self.r#final[i],
        r_mem_check,
      )
    })
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
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

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
  proof_mem: BatchedGrandProductArgument<F>,
  proof_ops: BatchedGrandProductArgument<F>,
}

impl<F: PrimeField, const c: usize> ProductLayerProof<F, c> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial product layer proof"
  }

  pub fn prove<G>(
    grand_products: &mut [GrandProducts<F>; c],
    dense: &DensifiedRepresentation<F, c>,
    derefs: &Derefs<F>,
    eval: F,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    // TODO(moodlezoup): Move scalar product stuff into separate prove/verify
    // prepare scalar product
    let mut scalar_product_operands = derefs.eq_evals.clone();
    scalar_product_operands.push(dense.val.clone());
    let mut scalar_product = GeneralizedScalarProduct::new(scalar_product_operands);
    // build two dot product circuits to prove evaluation of sparse polynomial
    let (scalar_product_left, scalar_product_right) = scalar_product.split();

    let (eval_scalar_product_left, eval_scalar_product_right) = (
      scalar_product_left.evaluate(),
      scalar_product_right.evaluate(),
    );

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product_left",
      &eval_scalar_product_left,
    );

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product_right",
      &eval_scalar_product_right,
    );

    assert_eq!(eval_scalar_product_left + eval_scalar_product_right, eval);
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////

    // let mut claims_dotp_final = (Vec::new(), Vec::new(), Vec::new());

    // // prepare sequential instances that don't share poly_C
    // let mut poly_A_batched_seq: Vec<&mut DensePolynomial<F>> = Vec::new();
    // let mut poly_B_batched_seq: Vec<&mut DensePolynomial<F>> = Vec::new();
    // let mut poly_C_batched_seq: Vec<&mut DensePolynomial<F>> = Vec::new();
    // // add additional claims
    // for item in scalar_product_circuits.iter() {
    //   claims_to_verify.push(item.evaluate());
    //   assert_eq!(len / 2, item.left.len());
    //   assert_eq!(len / 2, item.right.len());
    //   assert_eq!(len / 2, item.weight.len());
    // }

    // for dotp_circuit in scalar_product_circuits.iter_mut() {
    //   poly_A_batched_seq.push(&mut dotp_circuit.left);
    //   poly_B_batched_seq.push(&mut dotp_circuit.right);
    //   poly_C_batched_seq.push(&mut dotp_circuit.weight);
    // }
    // let poly_vec_seq = (
    //   &mut poly_A_batched_seq,
    //   &mut poly_B_batched_seq,
    //   &mut poly_C_batched_seq,
    // );

    // //////////////////////////////////////////////////////////////////////////////////////////////////////////

    // if !scalar_product_circuits.is_empty() {
    //   let (claims_dotp_left, claims_dotp_right, claims_dotp_weight) = claims_dotp;
    //   for i in 0..scalar_product_circuits.len() {
    //     <Transcript as ProofTranscript<G>>::append_scalar(
    //       transcript,
    //       b"claim_dotp_left",
    //       &claims_dotp_left[i],
    //     );

    //     <Transcript as ProofTranscript<G>>::append_scalar(
    //       transcript,
    //       b"claim_dotp_right",
    //       &claims_dotp_right[i],
    //     );

    //     <Transcript as ProofTranscript<G>>::append_scalar(
    //       transcript,
    //       b"claim_dotp_weight",
    //       &claims_dotp_weight[i],
    //     );
    //   }
    //   claims_dotp_final = (claims_dotp_left, claims_dotp_right, claims_dotp_weight);
    // }

    // //////////////////////////////////////////////////////////////////////////////////////////////////////////

    let grand_product_evals: [(F, F, F, F); c] = std::array::from_fn(|i| {
      let hash_init = grand_products[i].init.evaluate();
      let hash_read = grand_products[i].read.evaluate();
      let hash_write = grand_products[i].write.evaluate();
      let hash_final = grand_products[i].r#final.evaluate();

      assert_eq!(hash_init * hash_write, hash_read * hash_final);

      <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_hash_init", &hash_init);
      <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_hash_read", &hash_read);
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_write",
        &hash_write,
      );
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_final",
        &hash_final,
      );

      (hash_init, hash_read, hash_write, hash_final)
    });

    // The number of operations into the memory encoded by rx and ry are always the same (by design)
    // So we can produce a batched product proof for all of them at the same time.
    // prove the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write

    let mut read_write_grand_products: Vec<&mut GrandProductCircuit<F>> = grand_products
      .iter_mut()
      .map(|grand_product| [&mut grand_product.read, &mut grand_product.write])
      .flatten()
      .collect();

    let (proof_ops, rand_ops) =
      BatchedGrandProductArgument::<F>::prove::<G>(&mut read_write_grand_products, transcript);

    let mut init_final_grand_products: Vec<&mut GrandProductCircuit<F>> = grand_products
      .iter_mut()
      .map(|grand_product| [&mut grand_product.init, &mut grand_product.r#final])
      .flatten()
      .collect();

    // produce a batched proof of memory-related product circuits
    let (proof_mem, rand_mem) =
      BatchedGrandProductArgument::<F>::prove::<G>(&mut init_final_grand_products, transcript);

    let product_layer_proof = ProductLayerProof {
      grand_product_evals,
      proof_mem,
      proof_ops,
    };

    let mut product_layer_proof_encoded = vec![];
    product_layer_proof
      .serialize_compressed(&mut product_layer_proof_encoded)
      .unwrap();

    (product_layer_proof, rand_mem, rand_ops)
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
pub struct MemoryCheckingProof<G: CurveGroup, const c: usize> {
  proof_prod_layer: ProductLayerProof<G::ScalarField, c>,
  proof_hash_layer: HashLayerProof<G, c>,
}

impl<G: CurveGroup, const c: usize> MemoryCheckingProof<G, c> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }

  pub fn prove(
    // network: &mut PolyEvalNetwork<G::ScalarField, c>,
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
