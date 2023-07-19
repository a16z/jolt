use ark_ec::CurveGroup;
use ark_ff::PrimeField;

use crate::poly::dense_mlpoly::DensePolynomial;
use super::surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialCommitment};

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

impl<F: PrimeField, const C: usize> From<&SparseLookupMatrix<C>> for DensifiedRepresentation<F, C> {
  #[tracing::instrument(skip_all, name = "Densify")]
  fn from(sparse: &SparseLookupMatrix<C>) -> Self {
    // TODO(moodlezoup) Initialize as arrays using std::array::from_fn ?
    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(C);
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(C);

    for i in 0..C {
      let mut access_sequence = sparse
        .nz
        .iter()
        .map(|indices| indices[i])
        .collect::<Vec<usize>>();
      // TODO(moodlezoup) Is this resize necessary/in the right place?
      access_sequence.resize(sparse.s, 0usize);

      let mut final_timestamps = vec![0usize; sparse.m];
      let mut read_timestamps = vec![0usize; sparse.s];

      // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
      // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
      for i in 0..sparse.s {
        let memory_address = access_sequence[i];
        assert!(memory_address < sparse.m);
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
      s: sparse.s,
      log_m: sparse.log_m,
      m: sparse.m,
    }
  }
}

impl<F: PrimeField, const C: usize> DensifiedRepresentation<F, C> {
  #[tracing::instrument(skip_all, name = "Dense.commit")]
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
