#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
use super::dense_mlpoly::DensePolynomial;
use super::dense_mlpoly::{
  EqPolynomial, IdentityPolynomial, PolyCommitment, PolyCommitmentGens, PolyEvalProof,
};
use super::errors::ProofVerifyError;
use super::math::Math;
use super::product_tree::{DotProductCircuit, ProductCircuit, ProductCircuitEvalProofBatched};
use super::random::RandomTape;
use super::timer::Timer;
use super::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::{Field, PrimeField};
use ark_serialize::*;
use ark_std::{One, Zero};
use itertools::Itertools;
use std::convert::From;

use merlin::Transcript;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatEntry<F: PrimeField, const c: usize> {
  indices: [usize; c],
  val: F, // TODO(moodlezoup) always 1 for Lasso; delete?
}

impl<F: PrimeField, const c: usize> SparseMatEntry<F, c> {
  pub fn new(indices: [usize; c], val: F) -> Self {
    SparseMatEntry { indices, val }
  }
}

pub struct Derefs<F> {
  ops_vals: Vec<Vec<DensePolynomial<F>>>,
  combined_poly: DensePolynomial<F>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DerefsCommitment<G: CurveGroup> {
  comm_ops_val: PolyCommitment<G>,
}

// // Optimization: Combines k \mu-variate dense multilinear polynomials into a single
// //               (\mu + \log k)-variate dense multilinear polynomial.
// //               See Spartan 7.2.3 #5
// impl<F: PrimeField> Derefs<F> {
//   pub fn new(ops_vals: Vec<Vec<DensePolynomial<F>>>) -> Self {
//     // TODO(moodlezoup)
//     // assert_eq!(row_ops_val.len(), col_ops_val.len());

//     let derefs = {
//       // combine all polynomials into a single polynomial (used below to produce a single commitment)
//       let combined_poly = DensePolynomial::merge(ops_vals.concat().as_slice());

//       Derefs {
//         ops_vals,
//         combined_poly,
//       }
//     };

//     derefs
//   }

//   pub fn commit<G: CurveGroup<ScalarField = F>>(
//     &self,
//     gens: &PolyCommitmentGens<G>,
//   ) -> DerefsCommitment<G> {
//     let (comm_ops_val, _blinds) = self.combined_poly.commit(gens, None);
//     DerefsCommitment { comm_ops_val }
//   }
// }

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DerefsEvalProof<G: CurveGroup> {
  proof_derefs: PolyEvalProof<G>,
}

// impl<G: CurveGroup> DerefsEvalProof<G> {
//   fn protocol_name() -> &'static [u8] {
//     b"Derefs evaluation proof"
//   }

//   fn prove_single(
//     joint_poly: &DensePolynomial<G::ScalarField>,
//     r: &[G::ScalarField],
//     evals: Vec<G::ScalarField>,
//     gens: &PolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> PolyEvalProof<G> {
//     assert_eq!(
//       joint_poly.get_num_vars(),
//       r.len() + evals.len().log_2() as usize
//     );

//     // append the claimed evaluations to transcript
//     <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

//     // n-to-1 reduction
//     let (r_joint, eval_joint) = {
//       let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
//         transcript,
//         b"challenge_combine_n_to_one",
//         evals.len().log_2() as usize,
//       );

//       let mut poly_evals = DensePolynomial::new(evals);
//       for i in (0..challenges.len()).rev() {
//         poly_evals.bound_poly_var_bot(&challenges[i]);
//       }
//       assert_eq!(poly_evals.len(), 1);
//       let joint_claim_eval = poly_evals[0];
//       let mut r_joint = challenges;
//       r_joint.extend(r);

//       debug_assert_eq!(joint_poly.evaluate::<G>(&r_joint), joint_claim_eval);
//       (r_joint, joint_claim_eval)
//     };
//     // decommit the joint polynomial at r_joint
//     <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"joint_claim_eval", &eval_joint);

//     let (proof_derefs, _comm_derefs_eval) = PolyEvalProof::prove(
//       joint_poly,
//       None,
//       &r_joint,
//       &eval_joint,
//       None,
//       gens,
//       transcript,
//       random_tape,
//     );

//     proof_derefs
//   }

//   // evalues both polynomials at r and produces a joint proof of opening
//   pub fn prove(
//     derefs: &Derefs<G::ScalarField>,
//     eval_ops_val_vec: &Vec<Vec<G::ScalarField>>,
//     r: &[G::ScalarField],
//     gens: &PolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> Self {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       DerefsEvalProof::<G>::protocol_name(),
//     );

//     let evals = {
//       let mut evals = eval_ops_val_vec.concat();
//       evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());
//       evals
//     };
//     let proof_derefs = DerefsEvalProof::prove_single(
//       &derefs.combined_poly,
//       r,
//       evals,
//       gens,
//       transcript,
//       random_tape,
//     );

//     DerefsEvalProof { proof_derefs }
//   }

//   fn verify_single(
//     proof: &PolyEvalProof<G>,
//     comm: &PolyCommitment<G>,
//     r: &[G::ScalarField],
//     evals: Vec<G::ScalarField>,
//     gens: &PolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     // append the claimed evaluations to transcript
//     <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

//     // n-to-1 reduction
//     let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
//       transcript,
//       b"challenge_combine_n_to_one",
//       evals.len().log_2() as usize,
//     );
//     let mut poly_evals = DensePolynomial::new(evals);
//     for i in (0..challenges.len()).rev() {
//       poly_evals.bound_poly_var_bot(&challenges[i]);
//     }
//     assert_eq!(poly_evals.len(), 1);
//     let joint_claim_eval = poly_evals[0];
//     let mut r_joint = challenges;
//     r_joint.extend(r);

//     // decommit the joint polynomial at r_joint
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"joint_claim_eval",
//       &joint_claim_eval,
//     );

//     proof.verify_plain(gens, transcript, &r_joint, &joint_claim_eval, comm)
//   }

//   // verify evaluations of both polynomials at r
//   pub fn verify(
//     &self,
//     r: &[G::ScalarField],
//     eval_ops_val_vec: &Vec<Vec<G::ScalarField>>,
//     gens: &PolyCommitmentGens<G>,
//     comm: &DerefsCommitment<G>,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       DerefsEvalProof::<G>::protocol_name(),
//     );
//     let mut evals = eval_ops_val_vec.concat();
//     evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());

//     DerefsEvalProof::verify_single(
//       &self.proof_derefs,
//       &comm.comm_ops_val,
//       r,
//       evals,
//       gens,
//       transcript,
//     )
//   }
// }

// impl<G: CurveGroup> AppendToTranscript<G> for DerefsCommitment<G> {
//   fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
//     transcript.append_message(b"derefs_commitment", b"begin_derefs_commitment");
//     self.comm_ops_val.append_to_transcript(label, transcript);
//     transcript.append_message(b"derefs_commitment", b"end_derefs_commitment");
//   }
// }

pub struct SparseMatPolyCommitmentGens<G> {
  gens_combined_l_variate: PolyCommitmentGens<G>,
  gens_combined_log_m_variate: PolyCommitmentGens<G>,
  gens_derefs: PolyCommitmentGens<G>,
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
  l_variate_polys_commitment: PolyCommitment<G>,
  log_m_variate_polys_commitment: PolyCommitment<G>,
  s: usize, // sparsity
  log_m: usize,
  m: usize, // TODO: big integer
}

impl<G: CurveGroup> AppendToTranscript<G> for SparsePolynomialCommitment<G> {
  fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
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

pub struct DensifiedRepresentation<F: PrimeField, const c: usize> {
  dim: [DensePolynomial<F>; c],
  read: [DensePolynomial<F>; c],
  r#final: [DensePolynomial<F>; c],
  val: DensePolynomial<F>,
  combined_l_variate_polys: DensePolynomial<F>,
  combined_log_m_variate_polys: DensePolynomial<F>,
  s: usize, // sparsity
  log_m: usize,
  m: usize, // TODO: big integer
}

impl<F: PrimeField, const c: usize> DensifiedRepresentation<F, c> {
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &SparseMatPolyCommitmentGens<G>,
  ) -> SparsePolynomialCommitment<G> {
    let (l_variate_polys_commitment, _) = self
      .combined_l_variate_polys
      .commit(&gens.gens_combined_l_variate, None);
    let (log_m_variate_polys_commitment, _) = self
      .combined_log_m_variate_polys
      .commit(&gens.gens_combined_log_m_variate, None);

    SparsePolynomialCommitment {
      l_variate_polys_commitment,
      log_m_variate_polys_commitment,
      s: self.s,
      log_m: self.log_m,
      m: self.m,
    }
  }
  // TODO(moodlezoup): implement deref?
}

impl<F: PrimeField, const c: usize> From<SparseMatPolynomial<F, c>>
  for DensifiedRepresentation<F, c>
{
  fn from(sparse_poly: SparseMatPolynomial<F, c>) -> Self {
    // TODO(moodlezoup) Initialize as arrays using std::array::from_fn ?
    let mut dim: Vec<DensePolynomial<F>> = Vec::new();
    let mut read: Vec<DensePolynomial<F>> = Vec::new();
    let mut r#final: Vec<DensePolynomial<F>> = Vec::new();

    for i in 0..c {
      let mut access_sequence = sparse_poly
        .nonzero_entries
        .iter()
        .map(|entry| entry.indices[i])
        .collect::<Vec<usize>>();
      // TODO(moodlezoup) Is this resize necessary/in the right place?
      access_sequence.resize(sparse_poly.s, 0usize);

      let mut final_timestamps = vec![0usize; sparse_poly.m];
      let mut read_timestamps = vec![0usize; sparse_poly.s];

      // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
      // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
      for i in 0..sparse_poly.s {
        let memory_address = access_sequence[i];
        assert!(memory_address < sparse_poly.m);
        let ts = final_timestamps[memory_address];
        read_timestamps[i] = ts;
        let write_timestamp = ts + 1;
        final_timestamps[memory_address] = write_timestamp;
      }

      dim.push(DensePolynomial::from_usize(&access_sequence));
      read.push(DensePolynomial::from_usize(&read_timestamps));
      r#final.push(DensePolynomial::from_usize(&final_timestamps));
    }

    let mut values: Vec<F> = sparse_poly
      .nonzero_entries
      .iter()
      .map(|entry| entry.val)
      .collect();
    // TODO(moodlezoup) Is this resize necessary/in the right place?
    values.resize(sparse_poly.s, F::zero());

    let val = DensePolynomial::new(values);

    let mut l_variate_polys = [dim.as_slice(), read.as_slice()].concat();
    l_variate_polys.push(val.clone());

    let combined_l_variate_polys = DensePolynomial::merge(&l_variate_polys);
    let combined_log_m_variate_polys = DensePolynomial::merge(&r#final);

    DensifiedRepresentation {
      dim: dim.try_into().unwrap(),
      read: read.try_into().unwrap(),
      r#final: r#final.try_into().unwrap(),
      val,
      combined_l_variate_polys,
      combined_log_m_variate_polys,
      s: sparse_poly.s,
      log_m: sparse_poly.log_m,
      m: sparse_poly.m,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatPolynomial<F: PrimeField, const c: usize> {
  nonzero_entries: Vec<SparseMatEntry<F, c>>,
  s: usize, // sparsity
  log_m: usize,
  m: usize, // TODO: big integer
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

#[derive(Debug)]
struct ProductLayer<F> {
  init: ProductCircuit<F>,
  read_vec: Vec<ProductCircuit<F>>,
  write_vec: Vec<ProductCircuit<F>>,
  audit: ProductCircuit<F>,
}

#[derive(Debug)]
struct Layers<F> {
  prod_layer: ProductLayer<F>,
}

// impl<F: PrimeField> Layers<F> {
//   fn build_hash_layer(
//     eval_table: &[F],
//     addrs_vec: &[DensePolynomial<F>],
//     derefs_vec: &[DensePolynomial<F>],
//     read_ts_vec: &[DensePolynomial<F>],
//     audit_ts: &DensePolynomial<F>,
//     r_mem_check: &(F, F),
//   ) -> (
//     DensePolynomial<F>,
//     Vec<DensePolynomial<F>>,
//     Vec<DensePolynomial<F>>,
//     DensePolynomial<F>,
//   ) {
//     let (r_hash, r_multiset_check) = r_mem_check;

//     //hash(addr, val, ts) = ts * r_hash_sqr + val * r_hash + addr
//     let r_hash_sqr = r_hash.square();
//     let hash_func = |addr: &F, val: &F, ts: &F| -> F { *ts * r_hash_sqr + *val * *r_hash + *addr };

//     // hash init and audit that does not depend on #instances
//     let num_mem_cells = eval_table.len();
//     let poly_init_hashed = DensePolynomial::new(
//       (0..num_mem_cells)
//         .map(|i| {
//           // at init time, addr is given by i, init value is given by eval_table, and ts = 0
//           hash_func(&F::from(i as u64), &eval_table[i], &F::zero()) - r_multiset_check
//         })
//         .collect::<Vec<F>>(),
//     );
//     let poly_audit_hashed = DensePolynomial::new(
//       (0..num_mem_cells)
//         .map(|i| {
//           // at audit time, addr is given by i, value is given by eval_table, and ts is given by audit_ts
//           hash_func(&F::from(i as u64), &eval_table[i], &audit_ts[i]) - r_multiset_check
//         })
//         .collect::<Vec<F>>(),
//     );

//     // hash read and write that depends on #instances
//     let mut poly_read_hashed_vec: Vec<DensePolynomial<F>> = Vec::new();
//     let mut poly_write_hashed_vec: Vec<DensePolynomial<F>> = Vec::new();
//     for i in 0..addrs_vec.len() {
//       let (addrs, derefs, read_ts) = (&addrs_vec[i], &derefs_vec[i], &read_ts_vec[i]);
//       assert_eq!(addrs.len(), derefs.len());
//       assert_eq!(addrs.len(), read_ts.len());
//       let num_ops = addrs.len();
//       let poly_read_hashed = DensePolynomial::new(
//         (0..num_ops)
//           .map(|i| {
//             // at read time, addr is given by addrs, value is given by derefs, and ts is given by read_ts
//             hash_func(&addrs[i], &derefs[i], &read_ts[i]) - r_multiset_check
//           })
//           .collect::<Vec<F>>(),
//       );
//       poly_read_hashed_vec.push(poly_read_hashed);

//       let poly_write_hashed = DensePolynomial::new(
//         (0..num_ops)
//           .map(|i| {
//             // at write time, addr is given by addrs, value is given by derefs, and ts is given by write_ts = read_ts + 1
//             hash_func(&addrs[i], &derefs[i], &(read_ts[i] + F::one())) - r_multiset_check
//           })
//           .collect::<Vec<F>>(),
//       );
//       poly_write_hashed_vec.push(poly_write_hashed);
//     }

//     (
//       poly_init_hashed,
//       poly_read_hashed_vec,
//       poly_write_hashed_vec,
//       poly_audit_hashed,
//     )
//   }

//   pub fn new(
//     eval_table: &[F],
//     addr_timestamps: &MemoryPolynomials<F>,
//     poly_ops_val: &[DensePolynomial<F>],
//     r_mem_check: &(F, F),
//   ) -> Self {
//     let (poly_init_hashed, poly_read_hashed_vec, poly_write_hashed_vec, poly_audit_hashed) =
//       Layers::build_hash_layer(
//         eval_table,
//         &addr_timestamps.ops_addr,
//         poly_ops_val,
//         &addr_timestamps.read_ts,
//         &addr_timestamps.audit_ts,
//         r_mem_check,
//       );

//     let prod_init = ProductCircuit::new(&poly_init_hashed);
//     let prod_read_vec = (0..poly_read_hashed_vec.len())
//       .map(|i| ProductCircuit::new(&poly_read_hashed_vec[i]))
//       .collect::<Vec<ProductCircuit<F>>>();
//     let prod_write_vec = (0..poly_write_hashed_vec.len())
//       .map(|i| ProductCircuit::new(&poly_write_hashed_vec[i]))
//       .collect::<Vec<ProductCircuit<F>>>();
//     let prod_audit = ProductCircuit::new(&poly_audit_hashed);

//     // subset audit check
//     let hashed_writes: F = (0..prod_write_vec.len())
//       .map(|i| prod_write_vec[i].evaluate())
//       .product();
//     let hashed_write_set: F = prod_init.evaluate() * hashed_writes;

//     let hashed_reads: F = (0..prod_read_vec.len())
//       .map(|i| prod_read_vec[i].evaluate())
//       .product();
//     let hashed_read_set: F = hashed_reads * prod_audit.evaluate();

//     //assert_eq!(hashed_read_set, hashed_write_set);
//     debug_assert_eq!(hashed_read_set, hashed_write_set);

//     Layers {
//       prod_layer: ProductLayer {
//         init: prod_init,
//         read_vec: prod_read_vec,
//         write_vec: prod_write_vec,
//         audit: prod_audit,
//       },
//     }
//   }
// }

#[derive(Debug)]
struct PolyEvalNetwork<F> {
  layers_by_dimension: Vec<Layers<F>>,
}

// impl<F: PrimeField> PolyEvalNetwork<F> {
//   pub fn new(
//     dense: &MultiSparseMatPolynomialAsDense<F>,
//     derefs: &Derefs<F>,
//     mems: &Vec<Vec<F>>,
//     r_mem_check: &(F, F),
//   ) -> Self {
//     let layers_by_dimension: Vec<_> = mems
//       .iter()
//       .zip(dense.dim)
//       .zip(derefs.ops_vals)
//       .map(|((eval_table, addr_timestamps), poly_ops_val)| {
//         Layers::new(eval_table, &addr_timestamps, &poly_ops_val, r_mem_check)
//       })
//       .collect();

//     PolyEvalNetwork {
//       layers_by_dimension,
//     }
//   }
// }

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct HashLayerProof<G: CurveGroup> {
  eval_dim: Vec<(Vec<G::ScalarField>, Vec<G::ScalarField>, G::ScalarField)>,
  eval_val: Vec<G::ScalarField>,
  eval_derefs: Vec<Vec<G::ScalarField>>,
  proof_ops: PolyEvalProof<G>,
  proof_mem: PolyEvalProof<G>,
  proof_derefs: DerefsEvalProof<G>,
}

// impl<G: CurveGroup> HashLayerProof<G> {
//   fn protocol_name() -> &'static [u8] {
//     b"Sparse polynomial hash layer proof"
//   }

//   fn prove_helper(
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     addr_timestamps: &MemoryPolynomials<G::ScalarField>,
//   ) -> (Vec<G::ScalarField>, Vec<G::ScalarField>, G::ScalarField) {
//     let (rand_mem, rand_ops) = rand;

//     // decommit ops-addr at rand_ops
//     let mut eval_ops_addr_vec: Vec<G::ScalarField> = Vec::new();
//     for i in 0..addr_timestamps.ops_addr.len() {
//       let eval_ops_addr = addr_timestamps.ops_addr[i].evaluate::<G>(rand_ops);
//       eval_ops_addr_vec.push(eval_ops_addr);
//     }

//     // decommit read_ts at rand_ops
//     let mut eval_read_ts_vec: Vec<G::ScalarField> = Vec::new();
//     for i in 0..addr_timestamps.read_ts.len() {
//       let eval_read_ts = addr_timestamps.read_ts[i].evaluate::<G>(rand_ops);
//       eval_read_ts_vec.push(eval_read_ts);
//     }

//     // decommit audit-ts at rand_mem
//     let eval_audit_ts = addr_timestamps.audit_ts.evaluate::<G>(rand_mem);

//     (eval_ops_addr_vec, eval_read_ts_vec, eval_audit_ts)
//   }

//   fn prove(
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     dense: &MultiSparseMatPolynomialAsDense<G::ScalarField>,
//     derefs: &Derefs<G::ScalarField>,
//     gens: &SparseMatPolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> Self {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       HashLayerProof::<G>::protocol_name(),
//     );

//     let (rand_mem, rand_ops) = rand;

//     // decommit derefs at rand_ops
//     let eval_ops_val = derefs
//       .ops_vals
//       .iter()
//       .map(|ops| ops.iter().map(|op| op.evaluate::<G>(rand_ops)).collect())
//       .collect();
//     let proof_derefs = DerefsEvalProof::prove(
//       derefs,
//       &eval_ops_val,
//       rand_ops,
//       &gens.gens_derefs,
//       transcript,
//       random_tape,
//     );

//     // form a single decommitment using comm_comb_ops
//     let mut evals_ops: Vec<G::ScalarField> = Vec::new();
//     let mut evals_mem: Vec<G::ScalarField> = Vec::new();

//     let eval_dim = Vec::new();
//     for dim_i in dense.dim.iter() {
//       let (addr, read_ts, audit_ts) =
//         HashLayerProof::<G>::prove_helper((rand_mem, rand_ops), dim_i);
//       eval_dim.push((addr, read_ts, audit_ts));
//       evals_mem.push(audit_ts);
//       evals_ops.extend(&addr);
//       evals_ops.extend(&read_ts);
//     }

//     let eval_val_vec = (0..dense.val.len())
//       .map(|i| dense.val[i].evaluate::<G>(rand_ops))
//       .collect::<Vec<G::ScalarField>>();
//     evals_ops.extend(&eval_val_vec);
//     evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

//     <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_ops", &evals_ops);

//     let challenges_ops = <Transcript as ProofTranscript<G>>::challenge_vector(
//       transcript,
//       b"challenge_combine_n_to_one",
//       evals_ops.len().log_2() as usize,
//     );

//     let mut poly_evals_ops = DensePolynomial::new(evals_ops);
//     for i in (0..challenges_ops.len()).rev() {
//       poly_evals_ops.bound_poly_var_bot(&challenges_ops[i]);
//     }
//     assert_eq!(poly_evals_ops.len(), 1);
//     let joint_claim_eval_ops = poly_evals_ops[0];
//     let mut r_joint_ops = challenges_ops;
//     r_joint_ops.extend(rand_ops);
//     debug_assert_eq!(
//       dense.comb_ops.evaluate::<G>(&r_joint_ops),
//       joint_claim_eval_ops
//     );
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"joint_claim_eval_ops",
//       &joint_claim_eval_ops,
//     );

//     let (proof_ops, _comm_ops_eval) = PolyEvalProof::prove(
//       &dense.comb_ops,
//       None,
//       &r_joint_ops,
//       &joint_claim_eval_ops,
//       None,
//       &gens.gens_ops,
//       transcript,
//       random_tape,
//     );

//     <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_mem", &evals_mem);
//     let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
//       transcript,
//       b"challenge_combine_two_to_one",
//       evals_mem.len().log_2() as usize,
//     );

//     let mut poly_evals_mem = DensePolynomial::new(evals_mem);
//     for i in (0..challenges_mem.len()).rev() {
//       poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
//     }
//     assert_eq!(poly_evals_mem.len(), 1);
//     let joint_claim_eval_mem = poly_evals_mem[0];
//     let mut r_joint_mem = challenges_mem;
//     r_joint_mem.extend(rand_mem);
//     debug_assert_eq!(
//       dense.comb_mem.evaluate::<G>(&r_joint_mem),
//       joint_claim_eval_mem
//     );
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"joint_claim_eval_mem",
//       &joint_claim_eval_mem,
//     );

//     let (proof_mem, _comm_mem_eval) = PolyEvalProof::prove(
//       &dense.comb_mem,
//       None,
//       &r_joint_mem,
//       &joint_claim_eval_mem,
//       None,
//       &gens.gens_mem,
//       transcript,
//       random_tape,
//     );

//     HashLayerProof {
//       eval_dim,
//       eval_val: eval_val_vec,
//       eval_derefs: eval_ops_val,
//       proof_ops,
//       proof_mem,
//       proof_derefs,
//     }
//   }

//   fn verify_helper(
//     rand: &(&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     claims: &(
//       G::ScalarField,
//       Vec<G::ScalarField>,
//       Vec<G::ScalarField>,
//       G::ScalarField,
//     ),
//     eval_ops_val: &[G::ScalarField],
//     eval_ops_addr: &[G::ScalarField],
//     eval_read_ts: &[G::ScalarField],
//     eval_audit_ts: &G::ScalarField,
//     r: &[G::ScalarField],
//     gamma: &G::ScalarField,
//     tau: &G::ScalarField,
//   ) -> Result<(), ProofVerifyError> {
//     let hash_func = |a: &G::ScalarField,
//                      v: &G::ScalarField,
//                      t: &G::ScalarField|
//      -> G::ScalarField { *t * gamma.square() + *v * *gamma + *a - tau };
//     // moodlezoup: (t * gamma^2 + v * gamma + a) instead of (a * gamma^2 + v * gamma + t)

//     let (rand_mem, _rand_ops) = rand;
//     let (claim_init, claim_read, claim_write, claim_audit) = claims;

//     // init
//     // moodlezoup: Collapses Init_row into a single field element
//     // Spartan 7.2.3 #4
//     let eval_init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem); // [0, 1, ..., m-1]
//     let eval_init_val = EqPolynomial::new(r.to_vec()).evaluate(rand_mem); // [\tilde{eq}(0, r_x), \tilde{eq}(1, r_x), ..., \tilde{eq}(m-1, r_x)]
//     let hash_init_at_rand_mem = hash_func(&eval_init_addr, &eval_init_val, &G::ScalarField::zero()); // verify the claim_last of init chunk
//                                                                                                      // H_\gamma_1(a, v, t)
//     assert_eq!(&hash_init_at_rand_mem, claim_init);

//     // read
//     for i in 0..eval_ops_addr.len() {
//       let hash_read_at_rand_ops = hash_func(&eval_ops_addr[i], &eval_ops_val[i], &eval_read_ts[i]); // verify the claim_last of init chunk
//       assert_eq!(&hash_read_at_rand_ops, &claim_read[i]);
//     }

//     // write: shares addr, val component; only decommit write_ts
//     for i in 0..eval_ops_addr.len() {
//       let eval_write_ts = eval_read_ts[i] + G::ScalarField::one();
//       let hash_write_at_rand_ops = hash_func(&eval_ops_addr[i], &eval_ops_val[i], &eval_write_ts); // verify the claim_last of init chunk
//       assert_eq!(&hash_write_at_rand_ops, &claim_write[i]);
//     }

//     // audit: shares addr and val with init
//     let eval_audit_addr = eval_init_addr;
//     let eval_audit_val = eval_init_val;
//     let hash_audit_at_rand_mem = hash_func(&eval_audit_addr, &eval_audit_val, eval_audit_ts);
//     assert_eq!(&hash_audit_at_rand_mem, claim_audit); // verify the last step of the sum-check for audit

//     Ok(())
//   }

//   fn verify(
//     &self,
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     claims_dim: &Vec<(
//       G::ScalarField,
//       Vec<G::ScalarField>,
//       Vec<G::ScalarField>,
//       G::ScalarField,
//     )>,
//     claims_dotp: &[G::ScalarField],
//     comm: &SparsePolynomialCommitment<G>,
//     gens: &SparseMatPolyCommitmentGens<G>,
//     comm_derefs: &DerefsCommitment<G>,
//     r: &Vec<Vec<G::ScalarField>>,
//     r_hash: &G::ScalarField,
//     r_multiset_check: &G::ScalarField,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     let timer = Timer::new("verify_hash_proof");
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       HashLayerProof::<G>::protocol_name(),
//     );

//     let (rand_mem, rand_ops) = rand;

//     // struct HashLayerProof<G: CurveGroup> {
//     //   eval_dim: Vec<(Vec<G::ScalarField>, Vec<G::ScalarField>, G::ScalarField)>,
//     //                       [addr]            [read_ts]            audit_ts
//     //   eval_val: Vec<G::ScalarField>,
//     //   eval_derefs: Vec<Vec<G::ScalarField>>,
//     //   proof_ops: PolyEvalProof<G>,
//     //   proof_mem: PolyEvalProof<G>,
//     //   proof_derefs: DerefsEvalProof<G>,
//     // }

//     // comm_derefs
//     // self.eval_derefs = E_i
//     // self.eval_dim
//     //
//     // claims_dim
//     // claims_dotp

//     // verify derefs at rand_ops
//     // TODO(moodlezoup)
//     // assert_eq!(eval_row_ops_val.len(), eval_col_ops_val.len());
//     self.proof_derefs.verify(
//       rand_ops,
//       &self.eval_derefs,
//       &gens.gens_derefs,
//       comm_derefs,
//       transcript,
//     )?;

//     // verify the decommitments used in evaluation sum-check
//     let eval_val_vec = &self.eval_val;
//     assert_eq!(claims_dotp.len(), 3 * eval_row_ops_val.len());
//     for i in 0..claims_dotp.len() / 3 {
//       let claim_row_ops_val = claims_dotp[3 * i];
//       let claim_col_ops_val = claims_dotp[3 * i + 1];
//       let claim_val = claims_dotp[3 * i + 2];

//       assert_eq!(claim_row_ops_val, eval_row_ops_val[i]);
//       assert_eq!(claim_col_ops_val, eval_col_ops_val[i]);
//       assert_eq!(claim_val, eval_val_vec[i]);
//     }

//     // verify addr-timestamps using comm_comb_ops at rand_ops
//     let (eval_row_addr_vec, eval_row_read_ts_vec, eval_row_audit_ts) = &self.eval_row;
//     let (eval_col_addr_vec, eval_col_read_ts_vec, eval_col_audit_ts) = &self.eval_col;

//     let mut evals_ops: Vec<G::ScalarField> = Vec::new();
//     evals_ops.extend(eval_row_addr_vec);
//     evals_ops.extend(eval_row_read_ts_vec);
//     evals_ops.extend(eval_col_addr_vec);
//     evals_ops.extend(eval_col_read_ts_vec);
//     evals_ops.extend(eval_val_vec);
//     evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

//     <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_ops", &evals_ops);

//     let challenges_ops = <Transcript as ProofTranscript<G>>::challenge_vector(
//       transcript,
//       b"challenge_combine_n_to_one",
//       evals_ops.len().log_2() as usize,
//     );

//     let mut poly_evals_ops = DensePolynomial::new(evals_ops);
//     for i in (0..challenges_ops.len()).rev() {
//       poly_evals_ops.bound_poly_var_bot(&challenges_ops[i]);
//     }
//     assert_eq!(poly_evals_ops.len(), 1);
//     let joint_claim_eval_ops = poly_evals_ops[0];
//     let mut r_joint_ops = challenges_ops;
//     r_joint_ops.extend(rand_ops);
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"joint_claim_eval_ops",
//       &joint_claim_eval_ops,
//     );
//     self.proof_ops.verify_plain(
//       &gens.gens_ops,
//       transcript,
//       &r_joint_ops,
//       &joint_claim_eval_ops,
//       &comm.comm_comb_ops,
//     )?;

//     // verify proof-mem using comm_comb_mem at rand_mem
//     // form a single decommitment using comm_comb_mem at rand_mem
//     let evals_mem: Vec<G::ScalarField> = vec![*eval_row_audit_ts, *eval_col_audit_ts];
//     <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_mem", &evals_mem);
//     let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
//       transcript,
//       b"challenge_combine_two_to_one",
//       evals_mem.len().log_2() as usize,
//     );

//     let mut poly_evals_mem = DensePolynomial::new(evals_mem);
//     for i in (0..challenges_mem.len()).rev() {
//       poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
//     }
//     assert_eq!(poly_evals_mem.len(), 1);
//     let joint_claim_eval_mem = poly_evals_mem[0];
//     let mut r_joint_mem = challenges_mem;
//     r_joint_mem.extend(rand_mem);
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"joint_claim_eval_mem",
//       &joint_claim_eval_mem,
//     );
//     self.proof_mem.verify_plain(
//       &gens.gens_mem,
//       transcript,
//       &r_joint_mem,
//       &joint_claim_eval_mem,
//       &comm.comm_comb_mem,
//     )?;

//     // // verify the claims from the product layer

//     // for (claims) in claims_dim.iter().zip() {
//     //   let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_row;
//     //   HashLayerProof::<G>::verify_helper(
//     //     &(rand_mem, rand_ops),
//     //     claims,
//     //     eval_row_ops_val,
//     //     eval_ops_addr,
//     //     eval_read_ts,
//     //     eval_audit_ts,
//     //     r,
//     //     r_hash,
//     //     r_multiset_check,
//     //   )?;
//     // }
//     let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_row;
//     HashLayerProof::<G>::verify_helper(
//       &(rand_mem, rand_ops),
//       claims_row,
//       eval_row_ops_val,
//       eval_ops_addr,
//       eval_read_ts,
//       eval_audit_ts,
//       rx,
//       r_hash,
//       r_multiset_check,
//     )?;

//     let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_col;
//     HashLayerProof::<G>::verify_helper(
//       &(rand_mem, rand_ops),
//       claims_col,
//       eval_col_ops_val,
//       eval_ops_addr,
//       eval_read_ts,
//       eval_audit_ts,
//       ry,
//       r_hash,
//       r_multiset_check,
//     )?;

//     timer.stop();
//     Ok(())
//   }
// }

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct ProductLayerProof<F: PrimeField> {
  eval_dim: Vec<(F, Vec<F>, Vec<F>, F)>,
  eval_val: (Vec<F>, Vec<F>),
  proof_mem: ProductCircuitEvalProofBatched<F>,
  proof_ops: ProductCircuitEvalProofBatched<F>,
}

// impl<F: PrimeField> ProductLayerProof<F> {
//   fn protocol_name() -> &'static [u8] {
//     b"Sparse polynomial product layer proof"
//   }

//   pub fn prove<G>(
//     prod_layers: &Vec<ProductLayer<F>>,
//     dense: &MultiSparseMatPolynomialAsDense<F>,
//     derefs: &Derefs<F>,
//     eval: &[F],
//     transcript: &mut Transcript,
//   ) -> (Self, Vec<F>, Vec<F>)
//   where
//     G: CurveGroup<ScalarField = F>,
//   {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       ProductLayerProof::<F>::protocol_name(),
//     );

//     for (i, prod_layer) in prod_layers.iter().enumerate() {
//       let dim_eval_init = prod_layer.init.evaluate();
//       let dim_eval_audit = prod_layer.audit.evaluate();
//       let dim_eval_read = (0..prod_layer.read_vec.len())
//         .map(|i| prod_layer.read_vec[i].evaluate())
//         .collect::<Vec<F>>();
//       let dim_eval_write = (0..prod_layer.write_vec.len())
//         .map(|i| prod_layer.write_vec[i].evaluate())
//         .collect::<Vec<F>>();

//       // subset check
//       let ws: F = (0..dim_eval_write.len())
//         .map(|i| dim_eval_write[i])
//         .product();
//       let rs: F = (0..dim_eval_read.len()).map(|i| dim_eval_read[i]).product();
//       assert_eq!(dim_eval_init * ws, rs * dim_eval_audit);

//       // TODO(moodlezoup)
//       // <Transcript as ProofTranscript<G>>::append_scalar(
//       //   transcript,
//       //   b"claim_row_eval_init",
//       //   &dim_eval_init,
//       // );
//       // <Transcript as ProofTranscript<G>>::append_scalars(
//       //   transcript,
//       //   b"claim_row_eval_read",
//       //   &dim_eval_read,
//       // );
//       // <Transcript as ProofTranscript<G>>::append_scalars(
//       //   transcript,
//       //   b"claim_row_eval_write",
//       //   &dim_eval_write,
//       // );
//       // <Transcript as ProofTranscript<G>>::append_scalar(
//       //   transcript,
//       //   b"claim_row_eval_audit",
//       //   &dim_eval_audit,
//       // );
//     }

//     // TODO(moodlezoup)
//     // prepare dotproduct circuit for batching then with ops-related product circuits
//     derefs
//       .ops_vals
//       .iter()
//       .for_each(|ops| assert_eq!(eval.len(), ops.len()));
//     assert_eq!(eval.len(), dense.val.len());
//     let mut dotp_circuit_left_vec: Vec<DotProductCircuit<F>> = Vec::new();
//     let mut dotp_circuit_right_vec: Vec<DotProductCircuit<F>> = Vec::new();
//     let mut eval_dotp_left_vec: Vec<F> = Vec::new();
//     let mut eval_dotp_right_vec: Vec<F> = Vec::new();
//     for i in 0..derefs.row_ops_val.len() {
//       // evaluate sparse polynomial evaluation using two dotp checks
//       let left = derefs.row_ops_val[i].clone();
//       let right = derefs.col_ops_val[i].clone();
//       let weights = dense.val[i].clone();

//       // build two dot product circuits to prove evaluation of sparse polynomial
//       let mut dotp_circuit = DotProductCircuit::new(left, right, weights);
//       let (dotp_circuit_left, dotp_circuit_right) = dotp_circuit.split();

//       let (eval_dotp_left, eval_dotp_right) =
//         (dotp_circuit_left.evaluate(), dotp_circuit_right.evaluate());

//       <Transcript as ProofTranscript<G>>::append_scalar(
//         transcript,
//         b"claim_eval_dotp_left",
//         &eval_dotp_left,
//       );

//       <Transcript as ProofTranscript<G>>::append_scalar(
//         transcript,
//         b"claim_eval_dotp_right",
//         &eval_dotp_right,
//       );

//       assert_eq!(eval_dotp_left + eval_dotp_right, eval[i]);
//       eval_dotp_left_vec.push(eval_dotp_left);
//       eval_dotp_right_vec.push(eval_dotp_right);

//       dotp_circuit_left_vec.push(dotp_circuit_left);
//       dotp_circuit_right_vec.push(dotp_circuit_right);
//     }

//     // The number of operations into the memory encoded by rx and ry are always the same (by design)
//     // So we can produce a batched product proof for all of them at the same time.
//     // prove the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write
//     // TODO: we currently only produce proofs for 3 batched sparse polynomial evaluations
//     assert_eq!(row_prod_layer.read_vec.len(), 3);
//     let (row_read_A, row_read_B, row_read_C) = {
//       let (vec_A, vec_BC) = row_prod_layer.read_vec.split_at_mut(1);
//       let (vec_B, vec_C) = vec_BC.split_at_mut(1);
//       (vec_A, vec_B, vec_C)
//     };

//     let (row_write_A, row_write_B, row_write_C) = {
//       let (vec_A, vec_BC) = row_prod_layer.write_vec.split_at_mut(1);
//       let (vec_B, vec_C) = vec_BC.split_at_mut(1);
//       (vec_A, vec_B, vec_C)
//     };

//     let (col_read_A, col_read_B, col_read_C) = {
//       let (vec_A, vec_BC) = col_prod_layer.read_vec.split_at_mut(1);
//       let (vec_B, vec_C) = vec_BC.split_at_mut(1);
//       (vec_A, vec_B, vec_C)
//     };

//     let (col_write_A, col_write_B, col_write_C) = {
//       let (vec_A, vec_BC) = col_prod_layer.write_vec.split_at_mut(1);
//       let (vec_B, vec_C) = vec_BC.split_at_mut(1);
//       (vec_A, vec_B, vec_C)
//     };

//     let (dotp_left_A, dotp_left_B, dotp_left_C) = {
//       let (vec_A, vec_BC) = dotp_circuit_left_vec.split_at_mut(1);
//       let (vec_B, vec_C) = vec_BC.split_at_mut(1);
//       (vec_A, vec_B, vec_C)
//     };

//     let (dotp_right_A, dotp_right_B, dotp_right_C) = {
//       let (vec_A, vec_BC) = dotp_circuit_right_vec.split_at_mut(1);
//       let (vec_B, vec_C) = vec_BC.split_at_mut(1);
//       (vec_A, vec_B, vec_C)
//     };

//     let (proof_ops, rand_ops) = ProductCircuitEvalProofBatched::<F>::prove::<G>(
//       &mut vec![
//         &mut row_read_A[0],
//         &mut row_read_B[0],
//         &mut row_read_C[0],
//         &mut row_write_A[0],
//         &mut row_write_B[0],
//         &mut row_write_C[0],
//         &mut col_read_A[0],
//         &mut col_read_B[0],
//         &mut col_read_C[0],
//         &mut col_write_A[0],
//         &mut col_write_B[0],
//         &mut col_write_C[0],
//       ],
//       // &mut vec![
//       //   &mut dotp_left_A[0],
//       //   &mut dotp_right_A[0],
//       //   &mut dotp_left_B[0],
//       //   &mut dotp_right_B[0],
//       //   &mut dotp_left_C[0],
//       //   &mut dotp_right_C[0],
//       // ],
//       &mut Vec::new(),
//       transcript,
//     );

//     // produce a batched proof of memory-related product circuits
//     let (proof_mem, rand_mem) = ProductCircuitEvalProofBatched::<F>::prove::<G>(
//       &mut vec![
//         &mut row_prod_layer.init,
//         &mut row_prod_layer.audit,
//         &mut col_prod_layer.init,
//         &mut col_prod_layer.audit,
//       ],
//       &mut Vec::new(),
//       transcript,
//     );

//     let product_layer_proof = ProductLayerProof {
//       eval_dim: (row_eval_init, row_eval_read, row_eval_write, row_eval_audit),
//       eval_val: (eval_dotp_left_vec, eval_dotp_right_vec),
//       proof_mem,
//       proof_ops,
//     };

//     let mut product_layer_proof_encoded = vec![];
//     product_layer_proof
//       .serialize_compressed(&mut product_layer_proof_encoded)
//       .unwrap();

//     let msg = format!(
//       "len_product_layer_proof {:?}",
//       product_layer_proof_encoded.len()
//     );
//     Timer::print(&msg);

//     (product_layer_proof, rand_mem, rand_ops)
//   }

//   pub fn verify<G>(
//     &self,
//     num_ops: usize,
//     num_cells: usize,
//     eval: &[F],
//     transcript: &mut Transcript,
//   ) -> Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
//   where
//     G: CurveGroup<ScalarField = F>,
//   {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       ProductLayerProof::<F>::protocol_name(),
//     );

//     let timer = Timer::new("verify_prod_proof");
//     let num_instances = eval.len();

//     // subset check
//     let (row_eval_init, row_eval_read, row_eval_write, row_eval_audit) = &self.eval_row;
//     assert_eq!(row_eval_write.len(), num_instances);
//     assert_eq!(row_eval_read.len(), num_instances);
//     let ws: F = (0..row_eval_write.len())
//       .map(|i| row_eval_write[i])
//       .product();
//     let rs: F = (0..row_eval_read.len()).map(|i| row_eval_read[i]).product();
//     assert_eq!(*row_eval_init * ws, rs * row_eval_audit);

//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"claim_row_eval_init",
//       row_eval_init,
//     );
//     <Transcript as ProofTranscript<G>>::append_scalars(
//       transcript,
//       b"claim_row_eval_read",
//       row_eval_read,
//     );
//     <Transcript as ProofTranscript<G>>::append_scalars(
//       transcript,
//       b"claim_row_eval_write",
//       row_eval_write,
//     );
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"claim_row_eval_audit",
//       row_eval_audit,
//     );

//     // subset check
//     let (col_eval_init, col_eval_read, col_eval_write, col_eval_audit) = &self.eval_col;
//     assert_eq!(col_eval_write.len(), num_instances);
//     assert_eq!(col_eval_read.len(), num_instances);
//     let ws: F = (0..col_eval_write.len())
//       .map(|i| col_eval_write[i])
//       .product();
//     let rs: F = (0..col_eval_read.len()).map(|i| col_eval_read[i]).product();
//     assert_eq!(*col_eval_init * ws, rs * col_eval_audit);

//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"claim_col_eval_init",
//       col_eval_init,
//     );
//     <Transcript as ProofTranscript<G>>::append_scalars(
//       transcript,
//       b"claim_col_eval_read",
//       col_eval_read,
//     );
//     <Transcript as ProofTranscript<G>>::append_scalars(
//       transcript,
//       b"claim_col_eval_write",
//       col_eval_write,
//     );
//     <Transcript as ProofTranscript<G>>::append_scalar(
//       transcript,
//       b"claim_col_eval_audit",
//       col_eval_audit,
//     );

//     // // verify the evaluation of the sparse polynomial
//     // let (eval_dotp_left, eval_dotp_right) = &self.eval_val;
//     // assert_eq!(eval_dotp_left.len(), eval_dotp_left.len());
//     // assert_eq!(eval_dotp_left.len(), num_instances);
//     // let mut claims_dotp_circuit: Vec<F> = Vec::new();
//     // for i in 0..num_instances {
//     //   assert_eq!(eval_dotp_left[i] + eval_dotp_right[i], eval[i]);

//     //   <Transcript as ProofTranscript<G>>::append_scalar(
//     //     transcript,
//     //     b"claim_eval_dotp_left",
//     //     &eval_dotp_left[i],
//     //   );

//     //   <Transcript as ProofTranscript<G>>::append_scalar(
//     //     transcript,
//     //     b"claim_eval_dotp_right",
//     //     &eval_dotp_right[i],
//     //   );

//     //   claims_dotp_circuit.push(eval_dotp_left[i]);
//     //   claims_dotp_circuit.push(eval_dotp_right[i]);
//     // }

//     // verify the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write
//     let mut claims_prod_circuit: Vec<F> = Vec::new();
//     claims_prod_circuit.extend(row_eval_read);
//     claims_prod_circuit.extend(row_eval_write);
//     claims_prod_circuit.extend(col_eval_read);
//     claims_prod_circuit.extend(col_eval_write);

//     let (claims_ops, claims_dotp, rand_ops) = self.proof_ops.verify::<G>(
//       &claims_prod_circuit,
//       // &claims_dotp_circuit,
//       &Vec::new(),
//       num_ops,
//       transcript,
//     );
//     // verify the correctness of claim_row_eval_init and claim_row_eval_audit
//     let (claims_mem, _claims_mem_dotp, rand_mem) = self.proof_mem.verify::<G>(
//       &[
//         *row_eval_init,
//         *row_eval_audit,
//         *col_eval_init,
//         *col_eval_audit,
//       ],
//       &Vec::new(),
//       num_cells,
//       transcript,
//     );
//     timer.stop();

//     Ok((claims_mem, rand_mem, claims_ops, claims_dotp, rand_ops))
//   }
// }

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PolyEvalNetworkProof<G: CurveGroup> {
  proof_prod_layer: ProductLayerProof<G::ScalarField>,
  proof_hash_layer: HashLayerProof<G>,
}

// impl<G: CurveGroup> PolyEvalNetworkProof<G> {
//   fn protocol_name() -> &'static [u8] {
//     b"Sparse polynomial evaluation proof"
//   }

//   pub fn prove(
//     network: &mut PolyEvalNetwork<G::ScalarField>,
//     dense: &MultiSparseMatPolynomialAsDense<G::ScalarField>,
//     derefs: &Derefs<G::ScalarField>,
//     evals: &[G::ScalarField],
//     gens: &SparseMatPolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> Self {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       PolyEvalNetworkProof::<G>::protocol_name(),
//     );

//     let (proof_prod_layer, rand_mem, rand_ops) = ProductLayerProof::<G::ScalarField>::prove::<G>(
//       &mut network
//         .layers_by_dimension
//         .iter()
//         .map(|&layer| layer.prod_layer)
//         .collect(),
//       dense,
//       derefs,
//       evals,
//       transcript,
//     );

//     // proof of hash layer for row and col
//     let proof_hash_layer = HashLayerProof::prove(
//       (&rand_mem, &rand_ops),
//       dense,
//       derefs,
//       gens,
//       transcript,
//       random_tape,
//     );

//     PolyEvalNetworkProof {
//       proof_prod_layer,
//       proof_hash_layer,
//     }
//   }

//   pub fn verify(
//     &self,
//     comm: &SparsePolynomialCommitment<G>,
//     comm_derefs: &DerefsCommitment<G>,
//     evals: &[G::ScalarField],
//     gens: &SparseMatPolyCommitmentGens<G>,
//     r: &Vec<Vec<G::ScalarField>>,
//     r_mem_check: &(G::ScalarField, G::ScalarField),
//     nz: usize,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     let timer = Timer::new("verify_polyeval_proof");
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       PolyEvalNetworkProof::<G>::protocol_name(),
//     );

//     let num_instances = evals.len();
//     let (r_hash, r_multiset_check) = r_mem_check;

//     let num_ops = nz.next_power_of_two();
//     let num_cells = rx.len().pow2();
//     assert_eq!(rx.len(), ry.len());

//     let (claims_mem, rand_mem, mut claims_ops, claims_dotp, rand_ops) = self
//       .proof_prod_layer
//       .verify::<G>(num_ops, num_cells, evals, transcript)?;
//     assert_eq!(claims_mem.len(), 4);
//     assert_eq!(claims_ops.len(), 4 * num_instances);
//     // TODO(moodlezoup)
//     // assert_eq!(claims_dotp.len(), 3 * num_instances);

//     let (claims_ops_row, claims_ops_col) = claims_ops.split_at_mut(2 * num_instances);
//     let (claims_ops_row_read, claims_ops_row_write) = claims_ops_row.split_at_mut(num_instances);
//     let (claims_ops_col_read, claims_ops_col_write) = claims_ops_col.split_at_mut(num_instances);

//     // verify the proof of hash layer
//     self.proof_hash_layer.verify(
//       (&rand_mem, &rand_ops),
//       &(
//         claims_mem[0],
//         claims_ops_row_read.to_vec(),
//         claims_ops_row_write.to_vec(),
//         claims_mem[1],
//       ),
//       // &(
//       //   claims_mem[2],
//       //   claims_ops_col_read.to_vec(),
//       //   claims_ops_col_write.to_vec(),
//       //   claims_mem[3],
//       // ),
//       &claims_dotp,
//       comm,
//       gens,
//       comm_derefs,
//       r,
//       r_hash,
//       r_multiset_check,
//       transcript,
//     )?;
//     timer.stop();

//     Ok(())
//   }
// }

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatPolyEvalProof<G: CurveGroup> {
  comm_derefs: DerefsCommitment<G>,
  poly_eval_network_proof: PolyEvalNetworkProof<G>,
}

// impl<G: CurveGroup> SparseMatPolyEvalProof<G> {
//   fn protocol_name() -> &'static [u8] {
//     b"Sparse polynomial evaluation proof"
//   }

//   fn equalize(r: &mut Vec<Vec<G::ScalarField>>) {
//     let max_len: usize = r.iter().map(|r_dim| r_dim.len()).max().unwrap();
//     for i in 0..r.len() {
//       if r[i].len() < max_len {
//         let diff = max_len - r[i].len();
//         let mut r_ext = vec![G::ScalarField::zero(); diff];
//         r_ext.extend(r[i]);
//         r[i] = r_ext;
//       }
//     }
//   }

//   pub fn prove(
//     dense: &MultiSparseMatPolynomialAsDense<G::ScalarField>,
//     r: &Vec<Vec<G::ScalarField>>, // point at which the polynomial is evaluated
//     evals: &[G::ScalarField],     // a vector evaluation of \widetilde{M}(r = (rx,ry)) for each M
//     gens: &SparseMatPolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> SparseMatPolyEvalProof<G> {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       SparseMatPolyEvalProof::<G>::protocol_name(),
//     );

//     // ensure there is one eval for each polynomial in dense
//     assert_eq!(evals.len(), dense.batch_size);

//     // TODO(moodlezoup)
//     let mut r_equalized = r;
//     SparseMatPolyEvalProof::<G>::equalize(&mut r_equalized);

//     let mems: Vec<_> = r_equalized
//       .iter()
//       .map(|&r_i| EqPolynomial::new(r_i).evals())
//       .collect();

//     let derefs = dense.deref(&mems);

//     // commit to non-deterministic choices of the prover
//     let timer_commit = Timer::new("commit_nondet_witness");
//     let comm_derefs = {
//       let comm = derefs.commit(&gens.gens_derefs);
//       comm.append_to_transcript(b"comm_poly_row_col_ops_val", transcript);
//       comm
//     };
//     timer_commit.stop();

//     let poly_eval_network_proof = {
//       // produce a random element from the transcript for hash function
//       let r_mem_check =
//         <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

//       // build a network to evaluate the sparse polynomial
//       let timer_build_network = Timer::new("build_layered_network");
//       let mut net = PolyEvalNetwork::new(dense, &derefs, &mems, &(r_mem_check[0], r_mem_check[1]));
//       timer_build_network.stop();

//       let timer_eval_network = Timer::new("evalproof_layered_network");
//       let poly_eval_network_proof = PolyEvalNetworkProof::prove(
//         &mut net,
//         dense,
//         &derefs,
//         evals,
//         gens,
//         transcript,
//         random_tape,
//       );
//       timer_eval_network.stop();

//       poly_eval_network_proof
//     };

//     SparseMatPolyEvalProof {
//       comm_derefs,
//       poly_eval_network_proof,
//     }
//   }

//   pub fn verify(
//     &self,
//     comm: &SparsePolynomialCommitment<G>,
//     r: &Vec<Vec<G::ScalarField>>, // point at which the polynomial is evaluated
//     evals: &[G::ScalarField],     // evaluation of \widetilde{M}(r = (rx,ry))
//     gens: &SparseMatPolyCommitmentGens<G>,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(
//       transcript,
//       SparseMatPolyEvalProof::<G>::protocol_name(),
//     );

//     // TODO(moodlezoup)
//     let mut r_equalized = r;
//     SparseMatPolyEvalProof::<G>::equalize(&mut r_equalized);

//     let (nz, num_mem_cells) = (comm.num_ops, comm.num_mem_cells);
//     assert_eq!(r_equalized[0].len().pow2(), num_mem_cells);

//     // add claims to transcript and obtain challenges for randomized mem-check circuit
//     self
//       .comm_derefs
//       .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

//     // produce a random element from the transcript for hash function
//     let r_mem_check =
//       <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

//     self.poly_eval_network_proof.verify(
//       comm,
//       &self.comm_derefs,
//       evals,
//       gens,
//       r_equalized,
//       &(r_mem_check[0], r_mem_check[1]),
//       nz,
//       transcript,
//     )
//   }
// }

// pub struct LagrangeBasisCoefficient<F> {
//   idx: usize,
//   coeff: F,
// }

// impl<F> LagrangeBasisCoefficient<F> {
//   pub fn new(idx: usize, coeff: F) -> Self {
//     LagrangeBasisCoefficient { idx, coeff }
//   }
// }

// pub struct SparsePolynomial<F> {
//   num_vars: usize,
//   lagrange_representation: Vec<LagrangeBasisCoefficient<F>>,
// }

// impl<F: PrimeField> SparsePolynomial<F> {
//   pub fn new(num_vars: usize, lagrange_representation: Vec<LagrangeBasisCoefficient<F>>) -> Self {
//     SparsePolynomial {
//       num_vars,
//       lagrange_representation,
//     }
//   }

//   fn evaluate_lagrange_basis_poly(idx_bits: &[bool], r: &[F]) -> F {
//     assert_eq!(idx_bits.len(), r.len());
//     let mut chi_i = F::one();
//     for j in 0..r.len() {
//       if idx_bits[j] {
//         chi_i *= r[j];
//       } else {
//         chi_i *= F::one() - r[j];
//       }
//     }
//     chi_i
//   }

//   // Takes O(n log n). TODO: do this in O(n) where n is the number of entries in Z
//   pub fn evaluate(&self, r: &[F]) -> F {
//     assert_eq!(self.num_vars, r.len());

//     (0..self.lagrange_representation.len())
//       .map(|i| {
//         let idx_bits = self.lagrange_representation[i].idx.get_bits(r.len());
//         SparsePolynomial::evaluate_lagrange_basis_poly(&idx_bits, r)
//           * self.lagrange_representation[i].coeff
//       })
//       .sum()
//   }
// }

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bls12_381::G1Projective;
  use ark_std::rand::RngCore;
  use ark_std::test_rng;
  use ark_std::UniformRand;

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
    let eval = sparse_poly.evaluate(&r);
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

    let commitment = dense.commit(&gens);
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
    let gens = SparseMatPolyCommitmentGens::<G>::new(b"gens_sparse_poly", c, s, log_m);

    // evaluation
    let r: Vec<G::ScalarField> = (0..c * log_m)
      .map(|_| G::ScalarField::rand(&mut prng))
      .collect();
    let eval = sparse_poly.evaluate(&r);

    // commitment
    let dense: DensifiedRepresentation<G::ScalarField, c> = sparse_poly.into();
    let commitment = dense.commit(&gens);

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
}
