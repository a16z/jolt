use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;

use super::surge::{SparsePolyCommitmentGens, SparsePolynomialCommitment};
use crate::jolt::JoltStrategy;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::math::Math;

pub struct DensifiedRepresentation<F: PrimeField, S: JoltStrategy<F>> {
  pub dim_usize: Vec<Vec<usize>>,
  pub dim: Vec<DensePolynomial<F>>,
  pub read: Vec<DensePolynomial<F>>,
  pub r#final: Vec<DensePolynomial<F>>,
  pub combined_l_variate_polys: DensePolynomial<F>,
  pub combined_log_m_variate_polys: DensePolynomial<F>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize,
  _marker: PhantomData<S>
}

impl<F: PrimeField, S: JoltStrategy<F>> DensifiedRepresentation<F, S> {
  /// Densifies indices.
  /// - `indices`: `sparsity` x `subtable_dimensionality` sized matrix of lookups.
  #[tracing::instrument(skip_all, name = "Densify")]
  pub fn from_lookup_indices(indices: &Vec<Vec<usize>>, log_m: usize) -> Self {
    debug_assert_eq!(indices[0].len(), S::subtable_dimensionality());

    let s = indices.len().next_power_of_two();
    let m = log_m.pow2();

    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(S::subtable_dimensionality());
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(S::subtable_dimensionality());
    let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(S::subtable_dimensionality());
    let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(S::subtable_dimensionality());

    // TODO(#29): Parallelize
    for i in 0..S::subtable_dimensionality() {
      let mut access_sequence = indices
        .iter()
        .map(|indices| indices[i])
        .collect::<Vec<usize>>();
      access_sequence.resize(s, 0usize);

      let mut final_timestamps = vec![0usize; m];
      let mut read_timestamps = vec![0usize; s];

      // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
      // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
      for i in 0..s {
        let memory_address = access_sequence[i];
        debug_assert!(memory_address < m);
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

    let l_variate_polys = [dim.as_slice(), read.as_slice()].concat();

    let combined_l_variate_polys = DensePolynomial::merge(&l_variate_polys);
    let combined_log_m_variate_polys = DensePolynomial::merge(&r#final);

    DensifiedRepresentation {
      dim_usize: dim_usize.try_into().unwrap(),
      dim: dim.try_into().unwrap(),
      read: read.try_into().unwrap(),
      r#final: r#final.try_into().unwrap(),
      combined_l_variate_polys,
      combined_log_m_variate_polys,
      s,
      log_m,
      m,
      _marker: PhantomData
    }
  }

  #[tracing::instrument(skip_all, name = "DensifiedRepresentation.commit")]
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &SparsePolyCommitmentGens<G>,
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
}
