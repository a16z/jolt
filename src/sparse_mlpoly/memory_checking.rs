use crate::dense_mlpoly::{DensePolynomial, EqPolynomial, IdentityPolynomial, PolyEvalProof};
use crate::errors::ProofVerifyError;
use crate::math::Math;
use crate::product_tree::{BatchedGrandProductArgument, GrandProductCircuit};
use crate::random::RandomTape;
use crate::sparse_mlpoly::densified::DensifiedRepresentation;
use crate::sparse_mlpoly::derefs::{Derefs, DerefsCommitment, DerefsEvalProof};
use crate::sparse_mlpoly::sparse_mlpoly::{
  SparseMatPolyCommitmentGens, SparsePolynomialCommitment,
};
use crate::transcript::ProofTranscript;

use ark_ec::CurveGroup;
use ark_ff::{Field, PrimeField};
use ark_serialize::*;
use ark_std::{One, Zero};
use itertools::izip;
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
  fn build_grand_product_inputs(
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
    ) = GrandProducts::build_grand_product_inputs(eval_table, dim_i, read_i, final_i, r_mem_check);

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

impl<F: PrimeField, const C: usize> DensifiedRepresentation<F, C> {
  pub fn to_grand_products(
    &self,
    mems: &Vec<Vec<F>>,
    r_mem_check: &(F, F),
  ) -> [GrandProducts<F>; C] {
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
struct HashLayerProof<G: CurveGroup, const C: usize> {
  eval_dim: [G::ScalarField; C],
  eval_read: [G::ScalarField; C],
  eval_final: [G::ScalarField; C],
  eval_val: G::ScalarField,
  eval_derefs: Vec<G::ScalarField>,
  proof_ops: PolyEvalProof<G>,
  proof_mem: PolyEvalProof<G>,
  proof_derefs: DerefsEvalProof<G, C>,
}

impl<G: CurveGroup, const C: usize> HashLayerProof<G, C> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial hash layer proof"
  }

  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    dense: &DensifiedRepresentation<G::ScalarField, C>,
    derefs: &Derefs<G::ScalarField, C>,
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
    let mut evals_ops: Vec<G::ScalarField> = Vec::new(); // moodlezoup: changed order of evals_ops

    let eval_dim: [G::ScalarField; C] =
      std::array::from_fn(|i| dense.dim[i].evaluate::<G>(rand_ops));
    let eval_read: [G::ScalarField; C] =
      std::array::from_fn(|i| dense.read[i].evaluate::<G>(rand_ops));
    let eval_final: [G::ScalarField; C] =
      std::array::from_fn(|i| dense.r#final[i].evaluate::<G>(rand_mem));
    let eval_val = dense.val.evaluate::<G>(rand_ops);

    evals_ops.extend(eval_dim);
    evals_ops.extend(eval_read);
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
    // TODO(moodlezoup) remove debug asserts
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

    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_mem", &eval_final);
    let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_two_to_one",
      eval_final.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new(eval_final.to_vec());
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
      eval_read,
      eval_final,
      eval_val,
      proof_ops,
      proof_mem,
      eval_derefs,
      proof_derefs,
    }
  }

  fn check_reed_solomon_fingerprints(
    rand_mem: &Vec<G::ScalarField>,
    claims: &(
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
    ),
    eval_deref: &G::ScalarField,
    eval_dim: &G::ScalarField,
    eval_read: &G::ScalarField,
    eval_final: &G::ScalarField,
    r: &Vec<G::ScalarField>,
    gamma: &G::ScalarField,
    tau: &G::ScalarField,
  ) -> Result<(), ProofVerifyError> {
    let hash_func = |a: &G::ScalarField,
                     v: &G::ScalarField,
                     t: &G::ScalarField|
     -> G::ScalarField { *t * gamma.square() + *v * *gamma + *a - tau };
    // moodlezoup: (t * gamma^2 + v * gamma + a) instead of (a * gamma^2 + v * gamma + t)

    let (claim_init, claim_read, claim_write, claim_final) = claims;

    // init
    let eval_init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem); // [0, 1, ..., m-1]
    let eval_init_val = EqPolynomial::new(r.to_vec()).evaluate(rand_mem); // [\tilde{eq}(0, r_x), \tilde{eq}(1, r_x), ..., \tilde{eq}(m-1, r_x)]
    let hash_init = hash_func(&eval_init_addr, &eval_init_val, &G::ScalarField::zero()); // verify the claim_last of init chunk
    assert_eq!(&hash_init, claim_init);

    // read
    let hash_read = hash_func(&eval_dim, &eval_deref, &eval_read); // verify the claim_last of init chunk
    assert_eq!(hash_read, *claim_read);

    // write: shares addr, val component
    let eval_write = *eval_read + G::ScalarField::one();
    let hash_write = hash_func(&eval_dim, &eval_deref, &eval_write); // verify the claim_last of init chunk
    assert_eq!(hash_write, *claim_write);

    // final: shares addr and val with init
    let eval_final_addr = eval_init_addr;
    let eval_final_val = eval_init_val;
    let hash_final = hash_func(&eval_final_addr, &eval_final_val, eval_final);
    assert_eq!(hash_final, *claim_final); // verify the last step of the sum-check for audit

    Ok(())
  }

  fn verify(
    &self,
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    claims_dim: &[(
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
    ); C],
    comm: &SparsePolynomialCommitment<G>,
    gens: &SparseMatPolyCommitmentGens<G>,
    comm_derefs: &DerefsCommitment<G>,
    r: &[Vec<G::ScalarField>; C],
    r_hash: &G::ScalarField,
    r_multiset_check: &G::ScalarField,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // verify derefs at rand_ops
    self.proof_derefs.verify(
      rand_ops,
      &self.eval_derefs,
      &gens.gens_derefs,
      comm_derefs,
      transcript,
    )?;

    let mut evals_ops: Vec<G::ScalarField> = Vec::new();
    evals_ops.extend(self.eval_dim);
    evals_ops.extend(self.eval_read);
    evals_ops.push(self.eval_val);
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
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_ops",
      &joint_claim_eval_ops,
    );

    self.proof_ops.verify_plain(
      &gens.gens_combined_l_variate,
      transcript,
      &r_joint_ops,
      &joint_claim_eval_ops,
      &comm.l_variate_polys_commitment,
    )?;

    <Transcript as ProofTranscript<G>>::append_scalars(
      transcript,
      b"claim_evals_mem",
      &self.eval_final,
    );
    let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_two_to_one",
      self.eval_final.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new(self.eval_final.to_vec());
    for i in (0..challenges_mem.len()).rev() {
      poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
    }
    assert_eq!(poly_evals_mem.len(), 1);

    let joint_claim_eval_mem = poly_evals_mem[0];
    let mut r_joint_mem = challenges_mem;
    r_joint_mem.extend(rand_mem);
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_mem",
      &joint_claim_eval_mem,
    );

    self.proof_mem.verify_plain(
      &gens.gens_combined_log_m_variate,
      transcript,
      &r_joint_mem,
      &joint_claim_eval_mem,
      &comm.log_m_variate_polys_commitment,
    )?;

    // verify the claims from the product layer
    for (claims, eval_deref, eval_dim, eval_read, eval_final, r_i) in izip!(
      claims_dim,
      &self.eval_derefs,
      &self.eval_dim,
      &self.eval_read,
      &self.eval_final,
      r
    ) {
      Self::check_reed_solomon_fingerprints(
        rand_mem,
        claims,
        &eval_deref,
        &eval_dim,
        &eval_read,
        &eval_final,
        r_i,
        r_hash,
        r_multiset_check,
      )?;
    }

    Ok(())
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct ProductLayerProof<F: PrimeField, const C: usize> {
  grand_product_evals: [(F, F, F, F); C],
  proof_mem: BatchedGrandProductArgument<F>,
  proof_ops: BatchedGrandProductArgument<F>,
}

impl<F: PrimeField, const C: usize> ProductLayerProof<F, C> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial product layer proof"
  }

  pub fn prove<G>(
    grand_products: &mut [GrandProducts<F>; C],
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let grand_product_evals: [(F, F, F, F); C] = std::array::from_fn(|i| {
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
    transcript: &mut Transcript,
  ) -> Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    for (hash_init, hash_read, hash_write, hash_final) in self.grand_product_evals {
      // Multiset equality check
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
    }

    let read_write_claims: Vec<F> = self
      .grand_product_evals
      .iter()
      .map(|(_, hash_read, hash_write, _)| [*hash_read, *hash_write])
      .flatten()
      .collect();

    let (claims_ops, rand_ops) =
      self
        .proof_ops
        .verify::<G, Transcript>(&read_write_claims, num_ops, transcript);

    let init_final_claims: Vec<F> = self
      .grand_product_evals
      .iter()
      .map(|(hash_init, _, _, hash_final)| [*hash_init, *hash_final])
      .flatten()
      .collect();

    let (claims_mem, rand_mem) =
      self
        .proof_mem
        .verify::<G, Transcript>(&init_final_claims, num_cells, transcript);

    Ok((claims_mem, rand_mem, claims_ops, rand_ops))
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<G: CurveGroup, const C: usize> {
  proof_prod_layer: ProductLayerProof<G::ScalarField, C>,
  proof_hash_layer: HashLayerProof<G, C>,
}

impl<G: CurveGroup, const C: usize> MemoryCheckingProof<G, C> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }

  pub fn prove(
    grand_products: &mut [GrandProducts<G::ScalarField>; C],
    dense: &DensifiedRepresentation<G::ScalarField, C>,
    derefs: &Derefs<G::ScalarField, C>,
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (proof_prod_layer, rand_mem, rand_ops) =
      ProductLayerProof::prove::<G>(grand_products, transcript);

    // proof of hash layer for row and col
    let proof_hash_layer = HashLayerProof::prove(
      (&rand_mem, &rand_ops),
      dense,
      derefs,
      gens,
      transcript,
      random_tape,
    );

    MemoryCheckingProof {
      proof_prod_layer,
      proof_hash_layer,
    }
  }

  pub fn verify(
    &self,
    comm: &SparsePolynomialCommitment<G>,
    comm_derefs: &DerefsCommitment<G>,
    gens: &SparseMatPolyCommitmentGens<G>,
    r: &[Vec<G::ScalarField>; C],
    r_mem_check: &(G::ScalarField, G::ScalarField),
    s: usize,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (r_hash, r_multiset_check) = r_mem_check;

    let num_ops = s.next_power_of_two();
    let num_cells = r[0].len().pow2();

    let (claims_mem, rand_mem, claims_ops, rand_ops) = self
      .proof_prod_layer
      .verify::<G>(num_ops, num_cells, transcript)?;

    let claims: [(
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
    ); C] = std::array::from_fn(|i| {
      (
        claims_mem[2 * i],     // init
        claims_ops[2 * i],     // read
        claims_ops[2 * i + 1], // write
        claims_mem[2 * i + 1], // final
      )
    });

    // verify the proof of hash layer
    self.proof_hash_layer.verify(
      (&rand_mem, &rand_ops),
      &claims,
      comm,
      gens,
      comm_derefs,
      r,
      r_hash,
      r_multiset_check,
      transcript,
    )?;

    Ok(())
  }
}
