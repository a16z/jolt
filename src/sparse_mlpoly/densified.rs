use ark_ec::CurveGroup;
use ark_ff::PrimeField;

use crate::dense_mlpoly::DensePolynomial;

use super::{
  derefs::Derefs,
  sparse_mlpoly::{SparseMatPolyCommitmentGens, SparseMatPolynomial, SparsePolynomialCommitment},
};

pub struct DensifiedRepresentation<F: PrimeField, const C: usize> {
  pub dim_usize: Vec<Vec<usize>>, // c-dimensional
  pub dim: [DensePolynomial<F>; C],
  pub read: [DensePolynomial<F>; C],
  pub r#final: [DensePolynomial<F>; C],
  pub val: DensePolynomial<F>,
  pub combined_l_variate_polys: DensePolynomial<F>,
  pub combined_log_m_variate_polys: DensePolynomial<F>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize, // TODO: big integer
}

impl<F: PrimeField, const C: usize> DensifiedRepresentation<F, C> {
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
  ) -> (
    SparseMatPolyCommitmentGens<G>,
    SparsePolynomialCommitment<G>,
  ) {
    let gens = SparseMatPolyCommitmentGens::<G>::new(b"gens_sparse_poly", C, self.s, self.log_m);
    let (l_variate_polys_commitment, _) = self
      .combined_l_variate_polys
      .commit(&gens.gens_combined_l_variate, None);
    let (log_m_variate_polys_commitment, _) = self
      .combined_log_m_variate_polys
      .commit(&gens.gens_combined_log_m_variate, None);

    (
      gens,
      SparsePolynomialCommitment {
        l_variate_polys_commitment,
        log_m_variate_polys_commitment,
        s: self.s,
        log_m: self.log_m,
        m: self.m,
      },
    )
  }

  /// Dereference memory. Create 'c' Dense(multi-linear)Polynomials with 's' evaluations of \tilde{eq}(i_dim, r_dim) corresponding to the non-zero indicies of M along the 'c'-th dimension.
  /// Where r is the randomly selected point by the verifier and r \in F^{log(M)} and i \in {0,1}^{log(M)} for all non-sparse indices along the 'c'-th dimension.
  /// - `eqs`: c-dimensional vector containing an M-sized vector for each dimension with evaulations of
  /// \tilde{eq}(i_0, r_0), ..., \tilde{eq}(i_c, r_c) where i_0, ..., i_c \in {0,1}^{logM} (for the non-sparse indices in each dimension)
  /// and r_0, ... r_c are the randomly selected evaluation points by the verifier.
  pub fn deref(&self, eqs: &Vec<Vec<F>>) -> Derefs<F, C> {
    // TODO(moodlezoup) std::array::from_fn
    // Iterate over each of the 'c' dimensions and their corresponding audit timestamps / counters
    let mut derefs: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    for (c_index, dim_i) in self.dim_usize.iter().enumerate() {
      let mut dim_deref: Vec<F> = Vec::with_capacity(self.s);
      for sparsity_index in 0..self.s {
        dim_deref.push(eqs[c_index][dim_i[sparsity_index]]);
      }
      derefs.push(DensePolynomial::new(dim_deref));
    }

    Derefs::new(derefs.try_into().unwrap())
  }
}

impl<F: PrimeField, const C: usize> From<SparseMatPolynomial<F, C>>
  for DensifiedRepresentation<F, C>
{
  fn from(sparse_poly: SparseMatPolynomial<F, C>) -> Self {
    // TODO(moodlezoup) Initialize as arrays using std::array::from_fn ?
    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(C);
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(C);

    for i in 0..C {
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
      dim_usize.push(access_sequence);
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
      dim_usize,
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
