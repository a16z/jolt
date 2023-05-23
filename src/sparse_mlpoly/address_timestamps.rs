use ark_ff::PrimeField;

use crate::dense_mlpoly::DensePolynomial;

pub struct AddrTimestamps<F> {
  /// The vector of indices within this span of memory which are looked up (size = sparsity)
  pub ops_addr_usize: Vec<usize>,

  /// The vector of indices within this span of memory which are looked up, encoded as a polynomial (size = sparsity)
  pub ops_addr: DensePolynomial<F>,

  /// 's'-sized Counter of the number of existing reads at the time this address was looked up
  pub read_ts: DensePolynomial<F>,

  // TODO: Is this 'm'-sized or logM sized?
  /// 'm'-sized polynomial evaluating to the count of accesses to each slot of 'm' memories
  pub audit_ts: DensePolynomial<F>,

  /// Sparsity: Number of non-zero / non-sparse indices
  pub s: usize,

  /// M is the maximum value for each tensor index, shared across all dimensions. log_m is the number of bits to represent a sparse index
  pub log_m: usize,

  /// Total number of cells, maximum cell value
  pub m: usize,
}

impl<F: PrimeField> AddrTimestamps<F> {
  pub fn new(num_cells: usize, num_ops: usize, addrs: Vec<usize>, log_m: usize, m: usize) -> Self {
    assert_eq!(addrs.len(), num_ops);
    assert_eq!(num_cells, m);

    let mut audit_ts = vec![0usize; num_cells];
    let mut read_ts = vec![0usize; num_ops];

    // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
    // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
    for i in 0..num_ops {
      let addr = addrs[i];
      assert!(addr < num_cells);
      let r_ts = audit_ts[addr];
      read_ts[i] = r_ts;

      let w_ts = r_ts + 1;
      audit_ts[addr] = w_ts;
    }

    let ops_addr_poly = DensePolynomial::from_usize(&addrs);
    let read_ts_poly = DensePolynomial::from_usize(&read_ts);

    AddrTimestamps {
      ops_addr: ops_addr_poly,
      ops_addr_usize: addrs,
      read_ts: read_ts_poly,
      audit_ts: DensePolynomial::from_usize(&audit_ts),
      s: num_ops,
      log_m,
      m,
    }
  }

  /// For each of the non-sparse indices in each dimension, evaluate eq at that point, and form a 's'-sized dense polynomial from it.
  /// - `mem_val`: evaluations of eq over all 'm' points {0,1}^log(m) at a verifier selected point r_i: eq({0,1}^log(m), r_i)
  pub fn deref(&self, mem_val: &[F]) -> DensePolynomial<F> {
    assert_eq!(mem_val.len(), self.m);

    let mut items: Vec<F> = Vec::with_capacity(self.s);
    for i in 0..self.s {
      items.push(mem_val[self.ops_addr_usize[i]]);
    }
    DensePolynomial::new(items)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::utils::index_to_field_bitvector;
  use ark_bls12_381::{Fr, G1Projective};

  #[test]
  fn non_overlapping() {
    let m = 8;
    let log_m = 3;
    let sparsity = 2; // 2 lookups
    let log_s = 1;

    // All lookups must be in the range 0..m
    let addrs = vec![1usize, 5];

    let addr_timestamps = AddrTimestamps::<Fr>::new(m, sparsity, addrs, log_m, m);

    // Ops addr is an 's'-sized array of the addrs as usizes
    assert_eq!(addr_timestamps.ops_addr_usize.len(), sparsity);
    assert_eq!(addr_timestamps.ops_addr_usize[0], 1usize);
    assert_eq!(addr_timestamps.ops_addr_usize[1], 5usize);

    // Ops addr is an 's'-sized multi-linear polynomial, evaluating to the non-sparse addrs over sequential indices
    assert_eq!(addr_timestamps.ops_addr.len(), sparsity);
    let index_0: Vec<Fr> = index_to_field_bitvector(0, log_s);
    let index_1: Vec<Fr> = index_to_field_bitvector(1, log_s);
    assert_eq!(
      addr_timestamps
        .ops_addr
        .evaluate::<G1Projective>(index_0.as_slice()),
      Fr::from(1)
    );
    assert_eq!(
      addr_timestamps
        .ops_addr
        .evaluate::<G1Projective>(index_1.as_slice()),
      Fr::from(5)
    );

    // read-ts is an 's'-sized multi-linear polynomial
    assert_eq!(addr_timestamps.read_ts.len(), sparsity);
    let index_0: Vec<Fr> = index_to_field_bitvector(0, log_s);
    let index_1: Vec<Fr> = index_to_field_bitvector(1, log_s);
    // With none overlapping, the reads of existing is at 0
    assert_eq!(
      addr_timestamps
        .read_ts
        .evaluate::<G1Projective>(index_0.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .read_ts
        .evaluate::<G1Projective>(index_1.as_slice()),
      Fr::from(0)
    );

    // audit-ts is an 'm'-sized multi-linear polynomial
    assert_eq!(addr_timestamps.audit_ts.len(), m);
    todo!("This should be logm sized");
    let index_0: Vec<Fr> = index_to_field_bitvector(0, log_m);
    let index_1: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let index_2: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let index_3: Vec<Fr> = index_to_field_bitvector(3, log_m);
    let index_4: Vec<Fr> = index_to_field_bitvector(4, log_m);
    let index_5: Vec<Fr> = index_to_field_bitvector(5, log_m);
    let index_6: Vec<Fr> = index_to_field_bitvector(6, log_m);
    let index_7: Vec<Fr> = index_to_field_bitvector(7, log_m);
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_0.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_1.as_slice()),
      Fr::from(1)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_2.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_3.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_4.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_5.as_slice()),
      Fr::from(1)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_6.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_7.as_slice()),
      Fr::from(0)
    );
  }

  #[test]
  fn overlapping() {
    let m = 8;
    let log_m = 3;
    let sparsity = 4; // 6 lookups
    let log_s = 2;

    // All lookups must be in the range 0..m
    let addrs = vec![1usize, 3, 3, 3];

    let addr_timestamps = AddrTimestamps::<Fr>::new(m, sparsity, addrs, log_m, m);

    // Ops addr is an 's'-sized array of the addrs as usizes
    assert_eq!(addr_timestamps.ops_addr_usize.len(), sparsity);
    assert_eq!(addr_timestamps.ops_addr_usize[0], 1usize);
    assert_eq!(addr_timestamps.ops_addr_usize[1], 3usize);
    assert_eq!(addr_timestamps.ops_addr_usize[2], 3usize);
    assert_eq!(addr_timestamps.ops_addr_usize[3], 3usize);

    // read-ts is an 's'-sized multi-linear polynomial
    assert_eq!(addr_timestamps.read_ts.len(), sparsity);
    let index_0: Vec<Fr> = index_to_field_bitvector(0, log_s);
    let index_1: Vec<Fr> = index_to_field_bitvector(1, log_s);
    let index_2: Vec<Fr> = index_to_field_bitvector(2, log_s);
    let index_3: Vec<Fr> = index_to_field_bitvector(3, log_s);
    assert_eq!(
      addr_timestamps
        .read_ts
        .evaluate::<G1Projective>(index_0.as_slice()),
      Fr::from(0)
    ); // 1, has been looked up 0 times
    assert_eq!(
      addr_timestamps
        .read_ts
        .evaluate::<G1Projective>(index_1.as_slice()),
      Fr::from(0)
    ); // 3, has been looked up 0 times
    assert_eq!(
      addr_timestamps
        .read_ts
        .evaluate::<G1Projective>(index_2.as_slice()),
      Fr::from(1)
    ); // 3, has been looked up 1 times
    assert_eq!(
      addr_timestamps
        .read_ts
        .evaluate::<G1Projective>(index_3.as_slice()),
      Fr::from(2)
    ); // 3, has been looked up 2 times

    // Ops addr is an 's'-sized multi-linear polynomial, evaluating to the non-sparse addrs over sequential indices
    assert_eq!(addr_timestamps.ops_addr.len(), sparsity);
    let index_0: Vec<Fr> = index_to_field_bitvector(0, log_s);
    let index_1: Vec<Fr> = index_to_field_bitvector(1, log_s);
    let index_2: Vec<Fr> = index_to_field_bitvector(2, log_s);
    let index_3: Vec<Fr> = index_to_field_bitvector(3, log_s);
    assert_eq!(
      addr_timestamps
        .ops_addr
        .evaluate::<G1Projective>(index_0.as_slice()),
      Fr::from(1)
    );
    assert_eq!(
      addr_timestamps
        .ops_addr
        .evaluate::<G1Projective>(index_1.as_slice()),
      Fr::from(3)
    );
    assert_eq!(
      addr_timestamps
        .ops_addr
        .evaluate::<G1Projective>(index_2.as_slice()),
      Fr::from(3)
    );
    assert_eq!(
      addr_timestamps
        .ops_addr
        .evaluate::<G1Projective>(index_3.as_slice()),
      Fr::from(3)
    );

    // audit-ts is an 'm'-sized multi-linear polynomial
    todo!("This should be log m sized");
    assert_eq!(addr_timestamps.audit_ts.len(), m);
    let index_0: Vec<Fr> = index_to_field_bitvector(0, log_m);
    let index_1: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let index_2: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let index_3: Vec<Fr> = index_to_field_bitvector(3, log_m);
    let index_4: Vec<Fr> = index_to_field_bitvector(4, log_m);
    let index_5: Vec<Fr> = index_to_field_bitvector(5, log_m);
    let index_6: Vec<Fr> = index_to_field_bitvector(6, log_m);
    let index_7: Vec<Fr> = index_to_field_bitvector(7, log_m);
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_0.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_1.as_slice()),
      Fr::from(1)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_2.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_3.as_slice()),
      Fr::from(3)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_4.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_5.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_6.as_slice()),
      Fr::from(0)
    );
    assert_eq!(
      addr_timestamps
        .audit_ts
        .evaluate::<G1Projective>(index_7.as_slice()),
      Fr::from(0)
    );
  }

  // All lookups must be in the range 0..m
  #[test]
  #[should_panic]
  fn out_of_range_lookup_fails() {
    let m = 8;
    let log_m = 3;
    let sparsity = 2; // 2 lookups

    let addrs = vec![1usize, 10usize]; // 10 is out of range

    AddrTimestamps::<Fr>::new(m, sparsity, addrs, log_m, m);
  }
}
