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
use super::scalar::Scalar;
use super::timer::Timer;
use super::transcript::{AppendToTranscript, ProofTranscript};
use ark_ff::Field;
use ark_serialize::*;
use ark_std::{One, Zero};
use core::cmp::Ordering;
use merlin::Transcript;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatEntry {
  row: usize,
  col: usize,
  val: Scalar,
}

impl SparseMatEntry {
  pub fn new(row: usize, col: usize, val: Scalar) -> Self {
    SparseMatEntry { row, col, val }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatPolynomial {
  num_vars_x: usize,
  num_vars_y: usize,
  M: Vec<SparseMatEntry>,
}

pub struct Derefs {
  row_ops_val: Vec<DensePolynomial>,
  col_ops_val: Vec<DensePolynomial>,
  comb: DensePolynomial,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DerefsCommitment {
  comm_ops_val: PolyCommitment,
}

impl Derefs {
  pub fn new(row_ops_val: Vec<DensePolynomial>, col_ops_val: Vec<DensePolynomial>) -> Self {
    assert_eq!(row_ops_val.len(), col_ops_val.len());

    let derefs = {
      // combine all polynomials into a single polynomial (used below to produce a single commitment)
      let comb = DensePolynomial::merge(row_ops_val.iter().chain(col_ops_val.iter()));

      Derefs {
        row_ops_val,
        col_ops_val,
        comb,
      }
    };

    derefs
  }

  pub fn commit(&self, gens: &PolyCommitmentGens) -> DerefsCommitment {
    let (comm_ops_val, _blinds) = self.comb.commit(gens, None);
    DerefsCommitment { comm_ops_val }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DerefsEvalProof {
  proof_derefs: PolyEvalProof,
}

impl DerefsEvalProof {
  fn protocol_name() -> &'static [u8] {
    b"Derefs evaluation proof"
  }

  fn prove_single(
    joint_poly: &DensePolynomial,
    r: &[Scalar],
    evals: Vec<Scalar>,
    gens: &PolyCommitmentGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> PolyEvalProof {
    assert_eq!(
      joint_poly.get_num_vars(),
      r.len() + evals.len().log_2() as usize
    );

    // append the claimed evaluations to transcript
    evals.append_to_transcript(b"evals_ops_val", transcript);

    // n-to-1 reduction
    let (r_joint, eval_joint) = {
      let challenges =
        transcript.challenge_vector(b"challenge_combine_n_to_one", evals.len().log_2() as usize);
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
    eval_joint.append_to_transcript(b"joint_claim_eval", transcript);
    let (proof_derefs, _comm_derefs_eval) = PolyEvalProof::prove(
      joint_poly,
      None,
      &r_joint,
      &eval_joint,
      None,
      gens,
      transcript,
      random_tape,
    );

    proof_derefs
  }

  // evalues both polynomials at r and produces a joint proof of opening
  pub fn prove(
    derefs: &Derefs,
    eval_row_ops_val_vec: &[Scalar],
    eval_col_ops_val_vec: &[Scalar],
    r: &[Scalar],
    gens: &PolyCommitmentGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> Self {
    transcript.append_protocol_name(DerefsEvalProof::protocol_name());

    let evals = {
      let mut evals = eval_row_ops_val_vec.to_owned();
      evals.extend(eval_col_ops_val_vec);
      evals.resize(evals.len().next_power_of_two(), Scalar::zero());
      evals
    };
    let proof_derefs =
      DerefsEvalProof::prove_single(&derefs.comb, r, evals, gens, transcript, random_tape);

    DerefsEvalProof { proof_derefs }
  }

  fn verify_single(
    proof: &PolyEvalProof,
    comm: &PolyCommitment,
    r: &[Scalar],
    evals: Vec<Scalar>,
    gens: &PolyCommitmentGens,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // append the claimed evaluations to transcript
    evals.append_to_transcript(b"evals_ops_val", transcript);

    // n-to-1 reduction
    let challenges =
      transcript.challenge_vector(b"challenge_combine_n_to_one", evals.len().log_2() as usize);
    let mut poly_evals = DensePolynomial::new(evals);
    for i in (0..challenges.len()).rev() {
      poly_evals.bound_poly_var_bot(&challenges[i]);
    }
    assert_eq!(poly_evals.len(), 1);
    let joint_claim_eval = poly_evals[0];
    let mut r_joint = challenges;
    r_joint.extend(r);

    // decommit the joint polynomial at r_joint
    joint_claim_eval.append_to_transcript(b"joint_claim_eval", transcript);

    proof.verify_plain(gens, transcript, &r_joint, &joint_claim_eval, comm)
  }

  // verify evaluations of both polynomials at r
  pub fn verify(
    &self,
    r: &[Scalar],
    eval_row_ops_val_vec: &[Scalar],
    eval_col_ops_val_vec: &[Scalar],
    gens: &PolyCommitmentGens,
    comm: &DerefsCommitment,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    transcript.append_protocol_name(DerefsEvalProof::protocol_name());
    let mut evals = eval_row_ops_val_vec.to_owned();
    evals.extend(eval_col_ops_val_vec);
    evals.resize(evals.len().next_power_of_two(), Scalar::zero());

    DerefsEvalProof::verify_single(
      &self.proof_derefs,
      &comm.comm_ops_val,
      r,
      evals,
      gens,
      transcript,
    )
  }
}

impl AppendToTranscript for DerefsCommitment {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(b"derefs_commitment", b"begin_derefs_commitment");
    self.comm_ops_val.append_to_transcript(label, transcript);
    transcript.append_message(b"derefs_commitment", b"end_derefs_commitment");
  }
}

struct AddrTimestamps {
  ops_addr_usize: Vec<Vec<usize>>,
  ops_addr: Vec<DensePolynomial>,
  read_ts: Vec<DensePolynomial>,
  audit_ts: DensePolynomial,
}

impl AddrTimestamps {
  pub fn new(num_cells: usize, num_ops: usize, ops_addr: Vec<Vec<usize>>) -> Self {
    for item in ops_addr.iter() {
      assert_eq!(item.len(), num_ops);
    }

    let mut audit_ts = vec![0usize; num_cells];
    let mut ops_addr_vec: Vec<DensePolynomial> = Vec::new();
    let mut read_ts_vec: Vec<DensePolynomial> = Vec::new();
    for ops_addr_inst in ops_addr.iter() {
      let mut read_ts = vec![0usize; num_ops];

      // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
      // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
      for i in 0..num_ops {
        let addr = ops_addr_inst[i];
        assert!(addr < num_cells);
        let r_ts = audit_ts[addr];
        read_ts[i] = r_ts;

        let w_ts = r_ts + 1;
        audit_ts[addr] = w_ts;
      }

      ops_addr_vec.push(DensePolynomial::from_usize(ops_addr_inst));
      read_ts_vec.push(DensePolynomial::from_usize(&read_ts));
    }

    AddrTimestamps {
      ops_addr: ops_addr_vec,
      ops_addr_usize: ops_addr,
      read_ts: read_ts_vec,
      audit_ts: DensePolynomial::from_usize(&audit_ts),
    }
  }

  fn deref_mem(addr: &[usize], mem_val: &[Scalar]) -> DensePolynomial {
    DensePolynomial::new(
      (0..addr.len())
        .map(|i| {
          let a = addr[i];
          mem_val[a]
        })
        .collect::<Vec<Scalar>>(),
    )
  }

  pub fn deref(&self, mem_val: &[Scalar]) -> Vec<DensePolynomial> {
    (0..self.ops_addr.len())
      .map(|i| AddrTimestamps::deref_mem(&self.ops_addr_usize[i], mem_val))
      .collect::<Vec<DensePolynomial>>()
  }
}

pub struct MultiSparseMatPolynomialAsDense {
  batch_size: usize,
  val: Vec<DensePolynomial>,
  row: AddrTimestamps,
  col: AddrTimestamps,
  comb_ops: DensePolynomial,
  comb_mem: DensePolynomial,
}

pub struct SparseMatPolyCommitmentGens {
  gens_ops: PolyCommitmentGens,
  gens_mem: PolyCommitmentGens,
  gens_derefs: PolyCommitmentGens,
}

impl SparseMatPolyCommitmentGens {
  pub fn new(
    label: &'static [u8],
    num_vars_x: usize,
    num_vars_y: usize,
    num_nz_entries: usize,
    batch_size: usize,
  ) -> SparseMatPolyCommitmentGens {
    let num_vars_ops = num_nz_entries.next_power_of_two().log_2() as usize
      + (batch_size * 5).next_power_of_two().log_2() as usize;
    let num_vars_mem = if num_vars_x > num_vars_y {
      num_vars_x
    } else {
      num_vars_y
    } + 1;
    let num_vars_derefs = num_nz_entries.next_power_of_two().log_2() as usize
      + (batch_size * 2).next_power_of_two().log_2() as usize;

    let gens_ops = PolyCommitmentGens::new(num_vars_ops, label);
    let gens_mem = PolyCommitmentGens::new(num_vars_mem, label);
    let gens_derefs = PolyCommitmentGens::new(num_vars_derefs, label);
    SparseMatPolyCommitmentGens {
      gens_ops,
      gens_mem,
      gens_derefs,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatPolyCommitment {
  batch_size: usize,
  num_ops: usize,
  num_mem_cells: usize,
  comm_comb_ops: PolyCommitment,
  comm_comb_mem: PolyCommitment,
}

impl AppendToTranscript for SparseMatPolyCommitment {
  fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_u64(b"batch_size", self.batch_size as u64);
    transcript.append_u64(b"num_ops", self.num_ops as u64);
    transcript.append_u64(b"num_mem_cells", self.num_mem_cells as u64);
    self
      .comm_comb_ops
      .append_to_transcript(b"comm_comb_ops", transcript);
    self
      .comm_comb_mem
      .append_to_transcript(b"comm_comb_mem", transcript);
  }
}

impl SparseMatPolynomial {
  pub fn new(num_vars_x: usize, num_vars_y: usize, M: Vec<SparseMatEntry>) -> Self {
    SparseMatPolynomial {
      num_vars_x,
      num_vars_y,
      M,
    }
  }

  pub fn get_num_nz_entries(&self) -> usize {
    self.M.len().next_power_of_two()
  }

  fn sparse_to_dense_vecs(&self, N: usize) -> (Vec<usize>, Vec<usize>, Vec<Scalar>) {
    assert!(N >= self.get_num_nz_entries());
    let mut ops_row: Vec<usize> = vec![0; N];
    let mut ops_col: Vec<usize> = vec![0; N];
    let mut val: Vec<Scalar> = vec![Scalar::zero(); N];

    for i in 0..self.M.len() {
      ops_row[i] = self.M[i].row;
      ops_col[i] = self.M[i].col;
      val[i] = self.M[i].val;
    }
    (ops_row, ops_col, val)
  }

  fn multi_sparse_to_dense_rep(
    sparse_polys: &[&SparseMatPolynomial],
  ) -> MultiSparseMatPolynomialAsDense {
    assert!(!sparse_polys.is_empty());
    for i in 1..sparse_polys.len() {
      assert_eq!(sparse_polys[i].num_vars_x, sparse_polys[0].num_vars_x);
      assert_eq!(sparse_polys[i].num_vars_y, sparse_polys[0].num_vars_y);
    }

    let N = (0..sparse_polys.len())
      .map(|i| sparse_polys[i].get_num_nz_entries())
      .max()
      .unwrap()
      .next_power_of_two();

    let mut ops_row_vec: Vec<Vec<usize>> = Vec::new();
    let mut ops_col_vec: Vec<Vec<usize>> = Vec::new();
    let mut val_vec: Vec<DensePolynomial> = Vec::new();
    for poly in sparse_polys {
      let (ops_row, ops_col, val) = poly.sparse_to_dense_vecs(N);
      ops_row_vec.push(ops_row);
      ops_col_vec.push(ops_col);
      val_vec.push(DensePolynomial::new(val));
    }

    let any_poly = &sparse_polys[0];

    let num_mem_cells = if any_poly.num_vars_x > any_poly.num_vars_y {
      any_poly.num_vars_x.pow2()
    } else {
      any_poly.num_vars_y.pow2()
    };

    let row = AddrTimestamps::new(num_mem_cells, N, ops_row_vec);
    let col = AddrTimestamps::new(num_mem_cells, N, ops_col_vec);

    // combine polynomials into a single polynomial for commitment purposes
    let comb_ops = DensePolynomial::merge(
      row
        .ops_addr
        .iter()
        .chain(row.read_ts.iter())
        .chain(col.ops_addr.iter())
        .chain(col.read_ts.iter())
        .chain(val_vec.iter()),
    );
    let mut comb_mem = row.audit_ts.clone();
    comb_mem.extend(&col.audit_ts);

    MultiSparseMatPolynomialAsDense {
      batch_size: sparse_polys.len(),
      row,
      col,
      val: val_vec,
      comb_ops,
      comb_mem,
    }
  }

  fn evaluate_with_tables(&self, eval_table_rx: &[Scalar], eval_table_ry: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars_x.pow2(), eval_table_rx.len());
    assert_eq!(self.num_vars_y.pow2(), eval_table_ry.len());

    (0..self.M.len())
      .map(|i| {
        let row = self.M[i].row;
        let col = self.M[i].col;
        let val = &self.M[i].val;
        eval_table_rx[row] * eval_table_ry[col] * val
      })
      .sum()
  }

  pub fn multi_evaluate(
    polys: &[&SparseMatPolynomial],
    rx: &[Scalar],
    ry: &[Scalar],
  ) -> Vec<Scalar> {
    let eval_table_rx = EqPolynomial::new(rx.to_vec()).evals();
    let eval_table_ry = EqPolynomial::new(ry.to_vec()).evals();

    (0..polys.len())
      .map(|i| polys[i].evaluate_with_tables(&eval_table_rx, &eval_table_ry))
      .collect::<Vec<Scalar>>()
  }

  pub fn multiply_vec(&self, num_rows: usize, num_cols: usize, z: &[Scalar]) -> Vec<Scalar> {
    assert_eq!(z.len(), num_cols);

    (0..self.M.len())
      .map(|i| {
        let row = self.M[i].row;
        let col = self.M[i].col;
        let val = self.M[i].val;
        (row, val * z[col])
      })
      .fold(vec![Scalar::zero(); num_rows], |mut Mz, (r, v)| {
        Mz[r] += v;
        Mz
      })
  }

  pub fn compute_eval_table_sparse(
    &self,
    rx: &[Scalar],
    num_rows: usize,
    num_cols: usize,
  ) -> Vec<Scalar> {
    assert_eq!(rx.len(), num_rows);

    let mut M_evals: Vec<Scalar> = vec![Scalar::zero(); num_cols];

    for i in 0..self.M.len() {
      let entry = &self.M[i];
      M_evals[entry.col] += rx[entry.row] * entry.val;
    }
    M_evals
  }

  pub fn multi_commit(
    sparse_polys: &[&SparseMatPolynomial],
    gens: &SparseMatPolyCommitmentGens,
  ) -> (SparseMatPolyCommitment, MultiSparseMatPolynomialAsDense) {
    let batch_size = sparse_polys.len();
    let dense = SparseMatPolynomial::multi_sparse_to_dense_rep(sparse_polys);

    let (comm_comb_ops, _blinds_comb_ops) = dense.comb_ops.commit(&gens.gens_ops, None);
    let (comm_comb_mem, _blinds_comb_mem) = dense.comb_mem.commit(&gens.gens_mem, None);

    (
      SparseMatPolyCommitment {
        batch_size,
        num_mem_cells: dense.row.audit_ts.len(),
        num_ops: dense.row.read_ts[0].len(),
        comm_comb_ops,
        comm_comb_mem,
      },
      dense,
    )
  }
}

impl MultiSparseMatPolynomialAsDense {
  pub fn deref(&self, row_mem_val: &[Scalar], col_mem_val: &[Scalar]) -> Derefs {
    let row_ops_val = self.row.deref(row_mem_val);
    let col_ops_val = self.col.deref(col_mem_val);

    Derefs::new(row_ops_val, col_ops_val)
  }
}

#[derive(Debug)]
struct ProductLayer {
  init: ProductCircuit,
  read_vec: Vec<ProductCircuit>,
  write_vec: Vec<ProductCircuit>,
  audit: ProductCircuit,
}

#[derive(Debug)]
struct Layers {
  prod_layer: ProductLayer,
}

impl Layers {
  fn build_hash_layer(
    eval_table: &[Scalar],
    addrs_vec: &[DensePolynomial],
    derefs_vec: &[DensePolynomial],
    read_ts_vec: &[DensePolynomial],
    audit_ts: &DensePolynomial,
    r_mem_check: &(Scalar, Scalar),
  ) -> (
    DensePolynomial,
    Vec<DensePolynomial>,
    Vec<DensePolynomial>,
    DensePolynomial,
  ) {
    let (r_hash, r_multiset_check) = r_mem_check;

    //hash(addr, val, ts) = ts * r_hash_sqr + val * r_hash + addr
    let r_hash_sqr = r_hash.square();
    let hash_func = |addr: &Scalar, val: &Scalar, ts: &Scalar| -> Scalar {
      *ts * r_hash_sqr + *val * *r_hash + *addr
    };

    // hash init and audit that does not depend on #instances
    let num_mem_cells = eval_table.len();
    let poly_init_hashed = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // at init time, addr is given by i, init value is given by eval_table, and ts = 0
          hash_func(&Scalar::from(i as u64), &eval_table[i], &Scalar::zero()) - r_multiset_check
        })
        .collect::<Vec<Scalar>>(),
    );
    let poly_audit_hashed = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // at audit time, addr is given by i, value is given by eval_table, and ts is given by audit_ts
          hash_func(&Scalar::from(i as u64), &eval_table[i], &audit_ts[i]) - r_multiset_check
        })
        .collect::<Vec<Scalar>>(),
    );

    // hash read and write that depends on #instances
    let mut poly_read_hashed_vec: Vec<DensePolynomial> = Vec::new();
    let mut poly_write_hashed_vec: Vec<DensePolynomial> = Vec::new();
    for i in 0..addrs_vec.len() {
      let (addrs, derefs, read_ts) = (&addrs_vec[i], &derefs_vec[i], &read_ts_vec[i]);
      assert_eq!(addrs.len(), derefs.len());
      assert_eq!(addrs.len(), read_ts.len());
      let num_ops = addrs.len();
      let poly_read_hashed = DensePolynomial::new(
        (0..num_ops)
          .map(|i| {
            // at read time, addr is given by addrs, value is given by derefs, and ts is given by read_ts
            hash_func(&addrs[i], &derefs[i], &read_ts[i]) - r_multiset_check
          })
          .collect::<Vec<Scalar>>(),
      );
      poly_read_hashed_vec.push(poly_read_hashed);

      let poly_write_hashed = DensePolynomial::new(
        (0..num_ops)
          .map(|i| {
            // at write time, addr is given by addrs, value is given by derefs, and ts is given by write_ts = read_ts + 1
            hash_func(&addrs[i], &derefs[i], &(read_ts[i] + Scalar::one())) - r_multiset_check
          })
          .collect::<Vec<Scalar>>(),
      );
      poly_write_hashed_vec.push(poly_write_hashed);
    }

    (
      poly_init_hashed,
      poly_read_hashed_vec,
      poly_write_hashed_vec,
      poly_audit_hashed,
    )
  }

  pub fn new(
    eval_table: &[Scalar],
    addr_timestamps: &AddrTimestamps,
    poly_ops_val: &[DensePolynomial],
    r_mem_check: &(Scalar, Scalar),
  ) -> Self {
    let (poly_init_hashed, poly_read_hashed_vec, poly_write_hashed_vec, poly_audit_hashed) =
      Layers::build_hash_layer(
        eval_table,
        &addr_timestamps.ops_addr,
        poly_ops_val,
        &addr_timestamps.read_ts,
        &addr_timestamps.audit_ts,
        r_mem_check,
      );

    let prod_init = ProductCircuit::new(&poly_init_hashed);
    let prod_read_vec = (0..poly_read_hashed_vec.len())
      .map(|i| ProductCircuit::new(&poly_read_hashed_vec[i]))
      .collect::<Vec<ProductCircuit>>();
    let prod_write_vec = (0..poly_write_hashed_vec.len())
      .map(|i| ProductCircuit::new(&poly_write_hashed_vec[i]))
      .collect::<Vec<ProductCircuit>>();
    let prod_audit = ProductCircuit::new(&poly_audit_hashed);

    // subset audit check
    let hashed_writes: Scalar = (0..prod_write_vec.len())
      .map(|i| prod_write_vec[i].evaluate())
      .product();
    let hashed_write_set: Scalar = prod_init.evaluate() * hashed_writes;

    let hashed_reads: Scalar = (0..prod_read_vec.len())
      .map(|i| prod_read_vec[i].evaluate())
      .product();
    let hashed_read_set: Scalar = hashed_reads * prod_audit.evaluate();

    //assert_eq!(hashed_read_set, hashed_write_set);
    debug_assert_eq!(hashed_read_set, hashed_write_set);

    Layers {
      prod_layer: ProductLayer {
        init: prod_init,
        read_vec: prod_read_vec,
        write_vec: prod_write_vec,
        audit: prod_audit,
      },
    }
  }
}

#[derive(Debug)]
struct PolyEvalNetwork {
  row_layers: Layers,
  col_layers: Layers,
}

impl PolyEvalNetwork {
  pub fn new(
    dense: &MultiSparseMatPolynomialAsDense,
    derefs: &Derefs,
    mem_rx: &[Scalar],
    mem_ry: &[Scalar],
    r_mem_check: &(Scalar, Scalar),
  ) -> Self {
    let row_layers = Layers::new(mem_rx, &dense.row, &derefs.row_ops_val, r_mem_check);
    let col_layers = Layers::new(mem_ry, &dense.col, &derefs.col_ops_val, r_mem_check);

    PolyEvalNetwork {
      row_layers,
      col_layers,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct HashLayerProof {
  eval_row: (Vec<Scalar>, Vec<Scalar>, Scalar),
  eval_col: (Vec<Scalar>, Vec<Scalar>, Scalar),
  eval_val: Vec<Scalar>,
  eval_derefs: (Vec<Scalar>, Vec<Scalar>),
  proof_ops: PolyEvalProof,
  proof_mem: PolyEvalProof,
  proof_derefs: DerefsEvalProof,
}

impl HashLayerProof {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial hash layer proof"
  }

  fn prove_helper(
    rand: (&Vec<Scalar>, &Vec<Scalar>),
    addr_timestamps: &AddrTimestamps,
  ) -> (Vec<Scalar>, Vec<Scalar>, Scalar) {
    let (rand_mem, rand_ops) = rand;

    // decommit ops-addr at rand_ops
    let mut eval_ops_addr_vec: Vec<Scalar> = Vec::new();
    for i in 0..addr_timestamps.ops_addr.len() {
      let eval_ops_addr = addr_timestamps.ops_addr[i].evaluate(rand_ops);
      eval_ops_addr_vec.push(eval_ops_addr);
    }

    // decommit read_ts at rand_ops
    let mut eval_read_ts_vec: Vec<Scalar> = Vec::new();
    for i in 0..addr_timestamps.read_ts.len() {
      let eval_read_ts = addr_timestamps.read_ts[i].evaluate(rand_ops);
      eval_read_ts_vec.push(eval_read_ts);
    }

    // decommit audit-ts at rand_mem
    let eval_audit_ts = addr_timestamps.audit_ts.evaluate(rand_mem);

    (eval_ops_addr_vec, eval_read_ts_vec, eval_audit_ts)
  }

  fn prove(
    rand: (&Vec<Scalar>, &Vec<Scalar>),
    dense: &MultiSparseMatPolynomialAsDense,
    derefs: &Derefs,
    gens: &SparseMatPolyCommitmentGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> Self {
    transcript.append_protocol_name(HashLayerProof::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // decommit derefs at rand_ops
    let eval_row_ops_val = (0..derefs.row_ops_val.len())
      .map(|i| derefs.row_ops_val[i].evaluate(rand_ops))
      .collect::<Vec<Scalar>>();
    let eval_col_ops_val = (0..derefs.col_ops_val.len())
      .map(|i| derefs.col_ops_val[i].evaluate(rand_ops))
      .collect::<Vec<Scalar>>();
    let proof_derefs = DerefsEvalProof::prove(
      derefs,
      &eval_row_ops_val,
      &eval_col_ops_val,
      rand_ops,
      &gens.gens_derefs,
      transcript,
      random_tape,
    );
    let eval_derefs = (eval_row_ops_val, eval_col_ops_val);

    // evaluate row_addr, row_read-ts, col_addr, col_read-ts, val at rand_ops
    // evaluate row_audit_ts and col_audit_ts at rand_mem
    let (eval_row_addr_vec, eval_row_read_ts_vec, eval_row_audit_ts) =
      HashLayerProof::prove_helper((rand_mem, rand_ops), &dense.row);
    let (eval_col_addr_vec, eval_col_read_ts_vec, eval_col_audit_ts) =
      HashLayerProof::prove_helper((rand_mem, rand_ops), &dense.col);
    let eval_val_vec = (0..dense.val.len())
      .map(|i| dense.val[i].evaluate(rand_ops))
      .collect::<Vec<Scalar>>();

    // form a single decommitment using comm_comb_ops
    let mut evals_ops: Vec<Scalar> = Vec::new();
    evals_ops.extend(&eval_row_addr_vec);
    evals_ops.extend(&eval_row_read_ts_vec);
    evals_ops.extend(&eval_col_addr_vec);
    evals_ops.extend(&eval_col_read_ts_vec);
    evals_ops.extend(&eval_val_vec);
    evals_ops.resize(evals_ops.len().next_power_of_two(), Scalar::zero());
    evals_ops.append_to_transcript(b"claim_evals_ops", transcript);
    let challenges_ops = transcript.challenge_vector(
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
    debug_assert_eq!(dense.comb_ops.evaluate(&r_joint_ops), joint_claim_eval_ops);
    joint_claim_eval_ops.append_to_transcript(b"joint_claim_eval_ops", transcript);
    let (proof_ops, _comm_ops_eval) = PolyEvalProof::prove(
      &dense.comb_ops,
      None,
      &r_joint_ops,
      &joint_claim_eval_ops,
      None,
      &gens.gens_ops,
      transcript,
      random_tape,
    );

    // form a single decommitment using comb_comb_mem at rand_mem
    let evals_mem: Vec<Scalar> = vec![eval_row_audit_ts, eval_col_audit_ts];
    evals_mem.append_to_transcript(b"claim_evals_mem", transcript);
    let challenges_mem = transcript.challenge_vector(
      b"challenge_combine_two_to_one",
      evals_mem.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new(evals_mem);
    for i in (0..challenges_mem.len()).rev() {
      poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
    }
    assert_eq!(poly_evals_mem.len(), 1);
    let joint_claim_eval_mem = poly_evals_mem[0];
    let mut r_joint_mem = challenges_mem;
    r_joint_mem.extend(rand_mem);
    debug_assert_eq!(dense.comb_mem.evaluate(&r_joint_mem), joint_claim_eval_mem);
    joint_claim_eval_mem.append_to_transcript(b"joint_claim_eval_mem", transcript);
    let (proof_mem, _comm_mem_eval) = PolyEvalProof::prove(
      &dense.comb_mem,
      None,
      &r_joint_mem,
      &joint_claim_eval_mem,
      None,
      &gens.gens_mem,
      transcript,
      random_tape,
    );

    HashLayerProof {
      eval_row: (eval_row_addr_vec, eval_row_read_ts_vec, eval_row_audit_ts),
      eval_col: (eval_col_addr_vec, eval_col_read_ts_vec, eval_col_audit_ts),
      eval_val: eval_val_vec,
      eval_derefs,
      proof_ops,
      proof_mem,
      proof_derefs,
    }
  }

  fn verify_helper(
    rand: &(&Vec<Scalar>, &Vec<Scalar>),
    claims: &(Scalar, Vec<Scalar>, Vec<Scalar>, Scalar),
    eval_ops_val: &[Scalar],
    eval_ops_addr: &[Scalar],
    eval_read_ts: &[Scalar],
    eval_audit_ts: &Scalar,
    r: &[Scalar],
    r_hash: &Scalar,
    r_multiset_check: &Scalar,
  ) -> Result<(), ProofVerifyError> {
    let r_hash_sqr = r_hash.square();
    let hash_func = |addr: &Scalar, val: &Scalar, ts: &Scalar| -> Scalar {
      *ts * r_hash_sqr + *val * *r_hash + *addr
    };

    let (rand_mem, _rand_ops) = rand;
    let (claim_init, claim_read, claim_write, claim_audit) = claims;

    // init
    let eval_init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem);
    let eval_init_val = EqPolynomial::new(r.to_vec()).evaluate(rand_mem);
    let hash_init_at_rand_mem =
      hash_func(&eval_init_addr, &eval_init_val, &Scalar::zero()) - r_multiset_check; // verify the claim_last of init chunk
    assert_eq!(&hash_init_at_rand_mem, claim_init);

    // read
    for i in 0..eval_ops_addr.len() {
      let hash_read_at_rand_ops =
        hash_func(&eval_ops_addr[i], &eval_ops_val[i], &eval_read_ts[i]) - r_multiset_check; // verify the claim_last of init chunk
      assert_eq!(&hash_read_at_rand_ops, &claim_read[i]);
    }

    // write: shares addr, val component; only decommit write_ts
    for i in 0..eval_ops_addr.len() {
      let eval_write_ts = eval_read_ts[i] + Scalar::one();
      let hash_write_at_rand_ops =
        hash_func(&eval_ops_addr[i], &eval_ops_val[i], &eval_write_ts) - r_multiset_check; // verify the claim_last of init chunk
      assert_eq!(&hash_write_at_rand_ops, &claim_write[i]);
    }

    // audit: shares addr and val with init
    let eval_audit_addr = eval_init_addr;
    let eval_audit_val = eval_init_val;
    let hash_audit_at_rand_mem =
      hash_func(&eval_audit_addr, &eval_audit_val, eval_audit_ts) - r_multiset_check;
    assert_eq!(&hash_audit_at_rand_mem, claim_audit); // verify the last step of the sum-check for audit

    Ok(())
  }

  fn verify(
    &self,
    rand: (&Vec<Scalar>, &Vec<Scalar>),
    claims_row: &(Scalar, Vec<Scalar>, Vec<Scalar>, Scalar),
    claims_col: &(Scalar, Vec<Scalar>, Vec<Scalar>, Scalar),
    claims_dotp: &[Scalar],
    comm: &SparseMatPolyCommitment,
    gens: &SparseMatPolyCommitmentGens,
    comm_derefs: &DerefsCommitment,
    rx: &[Scalar],
    ry: &[Scalar],
    r_hash: &Scalar,
    r_multiset_check: &Scalar,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    let timer = Timer::new("verify_hash_proof");
    transcript.append_protocol_name(HashLayerProof::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // verify derefs at rand_ops
    let (eval_row_ops_val, eval_col_ops_val) = &self.eval_derefs;
    assert_eq!(eval_row_ops_val.len(), eval_col_ops_val.len());
    self.proof_derefs.verify(
      rand_ops,
      eval_row_ops_val,
      eval_col_ops_val,
      &gens.gens_derefs,
      comm_derefs,
      transcript,
    )?;

    // verify the decommitments used in evaluation sum-check
    let eval_val_vec = &self.eval_val;
    assert_eq!(claims_dotp.len(), 3 * eval_row_ops_val.len());
    for i in 0..claims_dotp.len() / 3 {
      let claim_row_ops_val = claims_dotp[3 * i];
      let claim_col_ops_val = claims_dotp[3 * i + 1];
      let claim_val = claims_dotp[3 * i + 2];

      assert_eq!(claim_row_ops_val, eval_row_ops_val[i]);
      assert_eq!(claim_col_ops_val, eval_col_ops_val[i]);
      assert_eq!(claim_val, eval_val_vec[i]);
    }

    // verify addr-timestamps using comm_comb_ops at rand_ops
    let (eval_row_addr_vec, eval_row_read_ts_vec, eval_row_audit_ts) = &self.eval_row;
    let (eval_col_addr_vec, eval_col_read_ts_vec, eval_col_audit_ts) = &self.eval_col;

    let mut evals_ops: Vec<Scalar> = Vec::new();
    evals_ops.extend(eval_row_addr_vec);
    evals_ops.extend(eval_row_read_ts_vec);
    evals_ops.extend(eval_col_addr_vec);
    evals_ops.extend(eval_col_read_ts_vec);
    evals_ops.extend(eval_val_vec);
    evals_ops.resize(evals_ops.len().next_power_of_two(), Scalar::zero());
    evals_ops.append_to_transcript(b"claim_evals_ops", transcript);
    let challenges_ops = transcript.challenge_vector(
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
    joint_claim_eval_ops.append_to_transcript(b"joint_claim_eval_ops", transcript);
    self.proof_ops.verify_plain(
      &gens.gens_ops,
      transcript,
      &r_joint_ops,
      &joint_claim_eval_ops,
      &comm.comm_comb_ops,
    )?;

    // verify proof-mem using comm_comb_mem at rand_mem
    // form a single decommitment using comb_comb_mem at rand_mem
    let evals_mem: Vec<Scalar> = vec![*eval_row_audit_ts, *eval_col_audit_ts];
    evals_mem.append_to_transcript(b"claim_evals_mem", transcript);
    let challenges_mem = transcript.challenge_vector(
      b"challenge_combine_two_to_one",
      evals_mem.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new(evals_mem);
    for i in (0..challenges_mem.len()).rev() {
      poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
    }
    assert_eq!(poly_evals_mem.len(), 1);
    let joint_claim_eval_mem = poly_evals_mem[0];
    let mut r_joint_mem = challenges_mem;
    r_joint_mem.extend(rand_mem);
    joint_claim_eval_mem.append_to_transcript(b"joint_claim_eval_mem", transcript);
    self.proof_mem.verify_plain(
      &gens.gens_mem,
      transcript,
      &r_joint_mem,
      &joint_claim_eval_mem,
      &comm.comm_comb_mem,
    )?;

    // verify the claims from the product layer
    let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_row;
    HashLayerProof::verify_helper(
      &(rand_mem, rand_ops),
      claims_row,
      eval_row_ops_val,
      eval_ops_addr,
      eval_read_ts,
      eval_audit_ts,
      rx,
      r_hash,
      r_multiset_check,
    )?;

    let (eval_ops_addr, eval_read_ts, eval_audit_ts) = &self.eval_col;
    HashLayerProof::verify_helper(
      &(rand_mem, rand_ops),
      claims_col,
      eval_col_ops_val,
      eval_ops_addr,
      eval_read_ts,
      eval_audit_ts,
      ry,
      r_hash,
      r_multiset_check,
    )?;

    timer.stop();
    Ok(())
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct ProductLayerProof {
  eval_row: (Scalar, Vec<Scalar>, Vec<Scalar>, Scalar),
  eval_col: (Scalar, Vec<Scalar>, Vec<Scalar>, Scalar),
  eval_val: (Vec<Scalar>, Vec<Scalar>),
  proof_mem: ProductCircuitEvalProofBatched,
  proof_ops: ProductCircuitEvalProofBatched,
}

impl ProductLayerProof {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial product layer proof"
  }

  pub fn prove(
    row_prod_layer: &mut ProductLayer,
    col_prod_layer: &mut ProductLayer,
    dense: &MultiSparseMatPolynomialAsDense,
    derefs: &Derefs,
    eval: &[Scalar],
    transcript: &mut Transcript,
  ) -> (Self, Vec<Scalar>, Vec<Scalar>) {
    transcript.append_protocol_name(ProductLayerProof::protocol_name());

    let row_eval_init = row_prod_layer.init.evaluate();
    let row_eval_audit = row_prod_layer.audit.evaluate();
    let row_eval_read = (0..row_prod_layer.read_vec.len())
      .map(|i| row_prod_layer.read_vec[i].evaluate())
      .collect::<Vec<Scalar>>();
    let row_eval_write = (0..row_prod_layer.write_vec.len())
      .map(|i| row_prod_layer.write_vec[i].evaluate())
      .collect::<Vec<Scalar>>();

    // subset check
    let ws: Scalar = (0..row_eval_write.len())
      .map(|i| row_eval_write[i])
      .product();
    let rs: Scalar = (0..row_eval_read.len()).map(|i| row_eval_read[i]).product();
    assert_eq!(row_eval_init * ws, rs * row_eval_audit);

    row_eval_init.append_to_transcript(b"claim_row_eval_init", transcript);
    row_eval_read.append_to_transcript(b"claim_row_eval_read", transcript);
    row_eval_write.append_to_transcript(b"claim_row_eval_write", transcript);
    row_eval_audit.append_to_transcript(b"claim_row_eval_audit", transcript);

    let col_eval_init = col_prod_layer.init.evaluate();
    let col_eval_audit = col_prod_layer.audit.evaluate();
    let col_eval_read: Vec<Scalar> = (0..col_prod_layer.read_vec.len())
      .map(|i| col_prod_layer.read_vec[i].evaluate())
      .collect();
    let col_eval_write: Vec<Scalar> = (0..col_prod_layer.write_vec.len())
      .map(|i| col_prod_layer.write_vec[i].evaluate())
      .collect();

    // subset check
    let ws: Scalar = (0..col_eval_write.len())
      .map(|i| col_eval_write[i])
      .product();
    let rs: Scalar = (0..col_eval_read.len()).map(|i| col_eval_read[i]).product();
    assert_eq!(col_eval_init * ws, rs * col_eval_audit);

    col_eval_init.append_to_transcript(b"claim_col_eval_init", transcript);
    col_eval_read.append_to_transcript(b"claim_col_eval_read", transcript);
    col_eval_write.append_to_transcript(b"claim_col_eval_write", transcript);
    col_eval_audit.append_to_transcript(b"claim_col_eval_audit", transcript);

    // prepare dotproduct circuit for batching then with ops-related product circuits
    assert_eq!(eval.len(), derefs.row_ops_val.len());
    assert_eq!(eval.len(), derefs.col_ops_val.len());
    assert_eq!(eval.len(), dense.val.len());
    let mut dotp_circuit_left_vec: Vec<DotProductCircuit> = Vec::new();
    let mut dotp_circuit_right_vec: Vec<DotProductCircuit> = Vec::new();
    let mut eval_dotp_left_vec: Vec<Scalar> = Vec::new();
    let mut eval_dotp_right_vec: Vec<Scalar> = Vec::new();
    for i in 0..derefs.row_ops_val.len() {
      // evaluate sparse polynomial evaluation using two dotp checks
      let left = derefs.row_ops_val[i].clone();
      let right = derefs.col_ops_val[i].clone();
      let weights = dense.val[i].clone();

      // build two dot product circuits to prove evaluation of sparse polynomial
      let mut dotp_circuit = DotProductCircuit::new(left, right, weights);
      let (dotp_circuit_left, dotp_circuit_right) = dotp_circuit.split();

      let (eval_dotp_left, eval_dotp_right) =
        (dotp_circuit_left.evaluate(), dotp_circuit_right.evaluate());

      eval_dotp_left.append_to_transcript(b"claim_eval_dotp_left", transcript);
      eval_dotp_right.append_to_transcript(b"claim_eval_dotp_right", transcript);
      assert_eq!(eval_dotp_left + eval_dotp_right, eval[i]);
      eval_dotp_left_vec.push(eval_dotp_left);
      eval_dotp_right_vec.push(eval_dotp_right);

      dotp_circuit_left_vec.push(dotp_circuit_left);
      dotp_circuit_right_vec.push(dotp_circuit_right);
    }

    // The number of operations into the memory encoded by rx and ry are always the same (by design)
    // So we can produce a batched product proof for all of them at the same time.
    // prove the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write
    // TODO: we currently only produce proofs for 3 batched sparse polynomial evaluations
    assert_eq!(row_prod_layer.read_vec.len(), 3);
    let (row_read_A, row_read_B, row_read_C) = {
      let (vec_A, vec_BC) = row_prod_layer.read_vec.split_at_mut(1);
      let (vec_B, vec_C) = vec_BC.split_at_mut(1);
      (vec_A, vec_B, vec_C)
    };

    let (row_write_A, row_write_B, row_write_C) = {
      let (vec_A, vec_BC) = row_prod_layer.write_vec.split_at_mut(1);
      let (vec_B, vec_C) = vec_BC.split_at_mut(1);
      (vec_A, vec_B, vec_C)
    };

    let (col_read_A, col_read_B, col_read_C) = {
      let (vec_A, vec_BC) = col_prod_layer.read_vec.split_at_mut(1);
      let (vec_B, vec_C) = vec_BC.split_at_mut(1);
      (vec_A, vec_B, vec_C)
    };

    let (col_write_A, col_write_B, col_write_C) = {
      let (vec_A, vec_BC) = col_prod_layer.write_vec.split_at_mut(1);
      let (vec_B, vec_C) = vec_BC.split_at_mut(1);
      (vec_A, vec_B, vec_C)
    };

    let (dotp_left_A, dotp_left_B, dotp_left_C) = {
      let (vec_A, vec_BC) = dotp_circuit_left_vec.split_at_mut(1);
      let (vec_B, vec_C) = vec_BC.split_at_mut(1);
      (vec_A, vec_B, vec_C)
    };

    let (dotp_right_A, dotp_right_B, dotp_right_C) = {
      let (vec_A, vec_BC) = dotp_circuit_right_vec.split_at_mut(1);
      let (vec_B, vec_C) = vec_BC.split_at_mut(1);
      (vec_A, vec_B, vec_C)
    };

    let (proof_ops, rand_ops) = ProductCircuitEvalProofBatched::prove(
      &mut vec![
        &mut row_read_A[0],
        &mut row_read_B[0],
        &mut row_read_C[0],
        &mut row_write_A[0],
        &mut row_write_B[0],
        &mut row_write_C[0],
        &mut col_read_A[0],
        &mut col_read_B[0],
        &mut col_read_C[0],
        &mut col_write_A[0],
        &mut col_write_B[0],
        &mut col_write_C[0],
      ],
      &mut vec![
        &mut dotp_left_A[0],
        &mut dotp_right_A[0],
        &mut dotp_left_B[0],
        &mut dotp_right_B[0],
        &mut dotp_left_C[0],
        &mut dotp_right_C[0],
      ],
      transcript,
    );

    // produce a batched proof of memory-related product circuits
    let (proof_mem, rand_mem) = ProductCircuitEvalProofBatched::prove(
      &mut vec![
        &mut row_prod_layer.init,
        &mut row_prod_layer.audit,
        &mut col_prod_layer.init,
        &mut col_prod_layer.audit,
      ],
      &mut Vec::new(),
      transcript,
    );

    let product_layer_proof = ProductLayerProof {
      eval_row: (row_eval_init, row_eval_read, row_eval_write, row_eval_audit),
      eval_col: (col_eval_init, col_eval_read, col_eval_write, col_eval_audit),
      eval_val: (eval_dotp_left_vec, eval_dotp_right_vec),
      proof_mem,
      proof_ops,
    };

    let mut product_layer_proof_encoded = vec![];
    product_layer_proof
      .serialize(&mut product_layer_proof_encoded)
      .unwrap();

    let msg = format!(
      "len_product_layer_proof {:?}",
      product_layer_proof_encoded.len()
    );
    Timer::print(&msg);

    (product_layer_proof, rand_mem, rand_ops)
  }

  pub fn verify(
    &self,
    num_ops: usize,
    num_cells: usize,
    eval: &[Scalar],
    transcript: &mut Transcript,
  ) -> Result<
    (
      Vec<Scalar>,
      Vec<Scalar>,
      Vec<Scalar>,
      Vec<Scalar>,
      Vec<Scalar>,
    ),
    ProofVerifyError,
  > {
    transcript.append_protocol_name(ProductLayerProof::protocol_name());

    let timer = Timer::new("verify_prod_proof");
    let num_instances = eval.len();

    // subset check
    let (row_eval_init, row_eval_read, row_eval_write, row_eval_audit) = &self.eval_row;
    assert_eq!(row_eval_write.len(), num_instances);
    assert_eq!(row_eval_read.len(), num_instances);
    let ws: Scalar = (0..row_eval_write.len())
      .map(|i| row_eval_write[i])
      .product();
    let rs: Scalar = (0..row_eval_read.len()).map(|i| row_eval_read[i]).product();
    assert_eq!(*row_eval_init * ws, rs * row_eval_audit);

    row_eval_init.append_to_transcript(b"claim_row_eval_init", transcript);
    row_eval_read.append_to_transcript(b"claim_row_eval_read", transcript);
    row_eval_write.append_to_transcript(b"claim_row_eval_write", transcript);
    row_eval_audit.append_to_transcript(b"claim_row_eval_audit", transcript);

    // subset check
    let (col_eval_init, col_eval_read, col_eval_write, col_eval_audit) = &self.eval_col;
    assert_eq!(col_eval_write.len(), num_instances);
    assert_eq!(col_eval_read.len(), num_instances);
    let ws: Scalar = (0..col_eval_write.len())
      .map(|i| col_eval_write[i])
      .product();
    let rs: Scalar = (0..col_eval_read.len()).map(|i| col_eval_read[i]).product();
    assert_eq!(*col_eval_init * ws, rs * col_eval_audit);

    col_eval_init.append_to_transcript(b"claim_col_eval_init", transcript);
    col_eval_read.append_to_transcript(b"claim_col_eval_read", transcript);
    col_eval_write.append_to_transcript(b"claim_col_eval_write", transcript);
    col_eval_audit.append_to_transcript(b"claim_col_eval_audit", transcript);

    // verify the evaluation of the sparse polynomial
    let (eval_dotp_left, eval_dotp_right) = &self.eval_val;
    assert_eq!(eval_dotp_left.len(), eval_dotp_left.len());
    assert_eq!(eval_dotp_left.len(), num_instances);
    let mut claims_dotp_circuit: Vec<Scalar> = Vec::new();
    for i in 0..num_instances {
      assert_eq!(eval_dotp_left[i] + eval_dotp_right[i], eval[i]);
      eval_dotp_left[i].append_to_transcript(b"claim_eval_dotp_left", transcript);
      eval_dotp_right[i].append_to_transcript(b"claim_eval_dotp_right", transcript);

      claims_dotp_circuit.push(eval_dotp_left[i]);
      claims_dotp_circuit.push(eval_dotp_right[i]);
    }

    // verify the correctness of claim_row_eval_read, claim_row_eval_write, claim_col_eval_read, and claim_col_eval_write
    let mut claims_prod_circuit: Vec<Scalar> = Vec::new();
    claims_prod_circuit.extend(row_eval_read);
    claims_prod_circuit.extend(row_eval_write);
    claims_prod_circuit.extend(col_eval_read);
    claims_prod_circuit.extend(col_eval_write);

    let (claims_ops, claims_dotp, rand_ops) = self.proof_ops.verify(
      &claims_prod_circuit,
      &claims_dotp_circuit,
      num_ops,
      transcript,
    );
    // verify the correctness of claim_row_eval_init and claim_row_eval_audit
    let (claims_mem, _claims_mem_dotp, rand_mem) = self.proof_mem.verify(
      &[
        *row_eval_init,
        *row_eval_audit,
        *col_eval_init,
        *col_eval_audit,
      ],
      &Vec::new(),
      num_cells,
      transcript,
    );
    timer.stop();

    Ok((claims_mem, rand_mem, claims_ops, claims_dotp, rand_ops))
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PolyEvalNetworkProof {
  proof_prod_layer: ProductLayerProof,
  proof_hash_layer: HashLayerProof,
}

impl PolyEvalNetworkProof {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }

  pub fn prove(
    network: &mut PolyEvalNetwork,
    dense: &MultiSparseMatPolynomialAsDense,
    derefs: &Derefs,
    evals: &[Scalar],
    gens: &SparseMatPolyCommitmentGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> Self {
    transcript.append_protocol_name(PolyEvalNetworkProof::protocol_name());

    let (proof_prod_layer, rand_mem, rand_ops) = ProductLayerProof::prove(
      &mut network.row_layers.prod_layer,
      &mut network.col_layers.prod_layer,
      dense,
      derefs,
      evals,
      transcript,
    );

    // proof of hash layer for row and col
    let proof_hash_layer = HashLayerProof::prove(
      (&rand_mem, &rand_ops),
      dense,
      derefs,
      gens,
      transcript,
      random_tape,
    );

    PolyEvalNetworkProof {
      proof_prod_layer,
      proof_hash_layer,
    }
  }

  pub fn verify(
    &self,
    comm: &SparseMatPolyCommitment,
    comm_derefs: &DerefsCommitment,
    evals: &[Scalar],
    gens: &SparseMatPolyCommitmentGens,
    rx: &[Scalar],
    ry: &[Scalar],
    r_mem_check: &(Scalar, Scalar),
    nz: usize,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    let timer = Timer::new("verify_polyeval_proof");
    transcript.append_protocol_name(PolyEvalNetworkProof::protocol_name());

    let num_instances = evals.len();
    let (r_hash, r_multiset_check) = r_mem_check;

    let num_ops = nz.next_power_of_two();
    let num_cells = rx.len().pow2();
    assert_eq!(rx.len(), ry.len());

    let (claims_mem, rand_mem, mut claims_ops, claims_dotp, rand_ops) = self
      .proof_prod_layer
      .verify(num_ops, num_cells, evals, transcript)?;
    assert_eq!(claims_mem.len(), 4);
    assert_eq!(claims_ops.len(), 4 * num_instances);
    assert_eq!(claims_dotp.len(), 3 * num_instances);

    let (claims_ops_row, claims_ops_col) = claims_ops.split_at_mut(2 * num_instances);
    let (claims_ops_row_read, claims_ops_row_write) = claims_ops_row.split_at_mut(num_instances);
    let (claims_ops_col_read, claims_ops_col_write) = claims_ops_col.split_at_mut(num_instances);

    // verify the proof of hash layer
    self.proof_hash_layer.verify(
      (&rand_mem, &rand_ops),
      &(
        claims_mem[0],
        claims_ops_row_read.to_vec(),
        claims_ops_row_write.to_vec(),
        claims_mem[1],
      ),
      &(
        claims_mem[2],
        claims_ops_col_read.to_vec(),
        claims_ops_col_write.to_vec(),
        claims_mem[3],
      ),
      &claims_dotp,
      comm,
      gens,
      comm_derefs,
      rx,
      ry,
      r_hash,
      r_multiset_check,
      transcript,
    )?;
    timer.stop();

    Ok(())
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatPolyEvalProof {
  comm_derefs: DerefsCommitment,
  poly_eval_network_proof: PolyEvalNetworkProof,
}

impl SparseMatPolyEvalProof {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }

  fn equalize(rx: &[Scalar], ry: &[Scalar]) -> (Vec<Scalar>, Vec<Scalar>) {
    match rx.len().cmp(&ry.len()) {
      Ordering::Less => {
        let diff = ry.len() - rx.len();
        let mut rx_ext = vec![Scalar::zero(); diff];
        rx_ext.extend(rx);
        (rx_ext, ry.to_vec())
      }
      Ordering::Greater => {
        let diff = rx.len() - ry.len();
        let mut ry_ext = vec![Scalar::zero(); diff];
        ry_ext.extend(ry);
        (rx.to_vec(), ry_ext)
      }
      Ordering::Equal => (rx.to_vec(), ry.to_vec()),
    }
  }

  pub fn prove(
    dense: &MultiSparseMatPolynomialAsDense,
    rx: &[Scalar], // point at which the polynomial is evaluated
    ry: &[Scalar],
    evals: &[Scalar], // a vector evaluation of \widetilde{M}(r = (rx,ry)) for each M
    gens: &SparseMatPolyCommitmentGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> SparseMatPolyEvalProof {
    transcript.append_protocol_name(SparseMatPolyEvalProof::protocol_name());

    // ensure there is one eval for each polynomial in dense
    assert_eq!(evals.len(), dense.batch_size);

    let (mem_rx, mem_ry) = {
      // equalize the lengths of rx and ry
      let (rx_ext, ry_ext) = SparseMatPolyEvalProof::equalize(rx, ry);
      let poly_rx = EqPolynomial::new(rx_ext).evals();
      let poly_ry = EqPolynomial::new(ry_ext).evals();
      (poly_rx, poly_ry)
    };

    let derefs = dense.deref(&mem_rx, &mem_ry);

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
      let r_mem_check = transcript.challenge_vector(b"challenge_r_hash", 2);

      // build a network to evaluate the sparse polynomial
      let timer_build_network = Timer::new("build_layered_network");
      let mut net = PolyEvalNetwork::new(
        dense,
        &derefs,
        &mem_rx,
        &mem_ry,
        &(r_mem_check[0], r_mem_check[1]),
      );
      timer_build_network.stop();

      let timer_eval_network = Timer::new("evalproof_layered_network");
      let poly_eval_network_proof = PolyEvalNetworkProof::prove(
        &mut net,
        dense,
        &derefs,
        evals,
        gens,
        transcript,
        random_tape,
      );
      timer_eval_network.stop();

      poly_eval_network_proof
    };

    SparseMatPolyEvalProof {
      comm_derefs,
      poly_eval_network_proof,
    }
  }

  pub fn verify(
    &self,
    comm: &SparseMatPolyCommitment,
    rx: &[Scalar], // point at which the polynomial is evaluated
    ry: &[Scalar],
    evals: &[Scalar], // evaluation of \widetilde{M}(r = (rx,ry))
    gens: &SparseMatPolyCommitmentGens,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    transcript.append_protocol_name(SparseMatPolyEvalProof::protocol_name());

    // equalize the lengths of rx and ry
    let (rx_ext, ry_ext) = SparseMatPolyEvalProof::equalize(rx, ry);

    let (nz, num_mem_cells) = (comm.num_ops, comm.num_mem_cells);
    assert_eq!(rx_ext.len().pow2(), num_mem_cells);

    // add claims to transcript and obtain challenges for randomized mem-check circuit
    self
      .comm_derefs
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    // produce a random element from the transcript for hash function
    let r_mem_check = transcript.challenge_vector(b"challenge_r_hash", 2);

    self.poly_eval_network_proof.verify(
      comm,
      &self.comm_derefs,
      evals,
      gens,
      &rx_ext,
      &ry_ext,
      &(r_mem_check[0], r_mem_check[1]),
      nz,
      transcript,
    )
  }
}

pub struct SparsePolyEntry {
  idx: usize,
  val: Scalar,
}

impl SparsePolyEntry {
  pub fn new(idx: usize, val: Scalar) -> Self {
    SparsePolyEntry { idx, val }
  }
}

pub struct SparsePolynomial {
  num_vars: usize,
  Z: Vec<SparsePolyEntry>,
}

impl SparsePolynomial {
  pub fn new(num_vars: usize, Z: Vec<SparsePolyEntry>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
    assert_eq!(a.len(), r.len());
    let mut chi_i = Scalar::one();
    for j in 0..r.len() {
      if a[j] {
        chi_i *= r[j];
      } else {
        chi_i *= Scalar::one() - r[j];
      }
    }
    chi_i
  }

  // Takes O(n log n). TODO: do this in O(n) where n is the number of entries in Z
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    (0..self.Z.len())
      .map(|i| {
        let bits = self.Z[i].idx.get_bits(r.len());
        SparsePolynomial::compute_chi(&bits, r) * self.Z[i].val
      })
      .sum()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_std::rand::RngCore;
  use ark_std::test_rng;
  use ark_std::UniformRand;

  #[test]
  fn check_sparse_polyeval_proof() {
    let mut prng = test_rng();

    let num_nz_entries: usize = 256;
    let num_rows: usize = 256;
    let num_cols: usize = 256;
    let num_vars_x: usize = num_rows.log_2() as usize;
    let num_vars_y: usize = num_cols.log_2() as usize;

    let mut M: Vec<SparseMatEntry> = Vec::new();

    for _i in 0..num_nz_entries {
      M.push(SparseMatEntry::new(
        (prng.next_u64() % (num_rows as u64)) as usize,
        (prng.next_u64() % (num_cols as u64)) as usize,
        Scalar::rand(&mut prng),
      ));
    }

    let poly_M = SparseMatPolynomial::new(num_vars_x, num_vars_y, M);
    let gens = SparseMatPolyCommitmentGens::new(
      b"gens_sparse_poly",
      num_vars_x,
      num_vars_y,
      num_nz_entries,
      3,
    );

    // commitment
    let (poly_comm, dense) = SparseMatPolynomial::multi_commit(&[&poly_M, &poly_M, &poly_M], &gens);

    // evaluation
    let rx: Vec<Scalar> = (0..num_vars_x)
      .map(|_i| Scalar::rand(&mut prng))
      .collect::<Vec<Scalar>>();
    let ry: Vec<Scalar> = (0..num_vars_y)
      .map(|_i| Scalar::rand(&mut prng))
      .collect::<Vec<Scalar>>();
    let eval = SparseMatPolynomial::multi_evaluate(&[&poly_M], &rx, &ry);
    let evals = vec![eval[0], eval[0], eval[0]];

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparseMatPolyEvalProof::prove(
      &dense,
      &rx,
      &ry,
      &evals,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(
        &poly_comm,
        &rx,
        &ry,
        &evals,
        &gens,
        &mut verifier_transcript,
      )
      .is_ok());
  }
}
