use ark_ec::CurveGroup;
use ark_ff::PrimeField;

#[cfg(feature = "multicore")]
use rayon::prelude::*;
#[cfg(feature = "multicore")]
use std::sync::Mutex;

use super::surge::{SparsePolyCommitmentGens, SparsePolynomialCommitment};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::math::Math;

pub struct DensifiedRepresentation<F: PrimeField, const C: usize> {
  pub dim_usize: [Vec<usize>; C],
  pub dim: [DensePolynomial<F>; C],
  pub read: [DensePolynomial<F>; C],
  pub r#final: [DensePolynomial<F>; C],
  pub combined_l_variate_polys: DensePolynomial<F>,
  pub combined_log_m_variate_polys: DensePolynomial<F>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize,
}

impl<F: PrimeField, const C: usize> DensifiedRepresentation<F, C> {
  #[tracing::instrument(skip_all, name = "Densify")]
  pub fn from_lookup_indices(indices: &Vec<[usize; C]>, log_m: usize) -> Self {
    let s = indices.len().next_power_of_two();
    let m = log_m.pow2();

    #[cfg(feature = "multicore")]
    let (iterator, dim_usize, dim, read, r#final) = {
      let iterator = (0..C).into_par_iter();
      let dim_usize: Mutex<Vec<Vec<usize>>> = Mutex::new(Vec::with_capacity(C));
      let dim: Mutex<Vec<DensePolynomial<F>>> = Mutex::new(Vec::with_capacity(C));
      let read: Mutex<Vec<DensePolynomial<F>>> = Mutex::new(Vec::with_capacity(C));
      let r#final: Mutex<Vec<DensePolynomial<F>>> = Mutex::new(Vec::with_capacity(C));
      (iterator, dim_usize, dim, read, r#final)
    };

    #[cfg(not(feature = "multicore"))]
    let (iterator, dim_usize, dim, read, r#final) = {
      let iterator = (0..C).iter();
      let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(C);
      let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
      let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
      let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
      (iterator, dim_usize, dim, read, r#final)
    };

    iterator.for_each(|i| {
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

      #[cfg(feature = "multicore")]
      let (mut dim_usize, mut dim, mut read, mut r#final) = {
        let dim_usize = dim_usize.lock().unwrap();
        let dim = dim.lock().unwrap();
        let read = read.lock().unwrap();
        let r#final = r#final.lock().unwrap();
        (dim_usize, dim, read, r#final)
      };

      dim.push(DensePolynomial::from_usize(&access_sequence));
      read.push(DensePolynomial::from_usize(&read_timestamps));
      r#final.push(DensePolynomial::from_usize(&final_timestamps));
      dim_usize.push(access_sequence);
    });

    #[cfg(feature = "multicore")]
    let (dim_usize, dim, read, r#final) = {
      let dim_usize = dim_usize.into_inner().unwrap();
      let dim = dim.into_inner().unwrap();
      let read = read.into_inner().unwrap();
      let r#final = r#final.into_inner().unwrap();
      (dim_usize, dim, read, r#final)
    };

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
