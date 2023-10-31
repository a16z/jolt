use std::marker::{PhantomData, Sync};

use ark_ec::CurveGroup;
use ark_ff::PrimeField;

use crate::{
  jolt::jolt_strategy::JoltStrategy,
  lasso::{densified::DensifiedRepresentation},
  poly::dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
  poly::eq_poly::EqPolynomial,
  subprotocols::{combined_table_proof::CombinedTableCommitment, grand_product::GrandProducts},
  utils::errors::ProofVerifyError,
  utils::math::Math,
  utils::random::RandomTape,
  utils::transcript::{AppendToTranscript, ProofTranscript},
};

#[cfg(feature = "multicore")]
use rayon::prelude::*;

pub mod subtable_strategy;
// pub mod and;
// pub mod lt;
// pub mod or;
// pub mod range_check;
// pub mod xor;

#[cfg(test)]
pub mod test;

pub struct Subtables<F: PrimeField, S: JoltStrategy<F>> {
  pub subtable_entries: Vec<Vec<F>>,
  pub lookup_polys: Vec<DensePolynomial<F>>,
  pub combined_poly: DensePolynomial<F>,
  _marker: PhantomData<S>,
}

/// Stores the non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials, enables combination and commitment.
impl<F: PrimeField, S: JoltStrategy<F>> Subtables<F, S> {
  /// Create new Subtables
  /// - `nz`: non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials
  /// - `s`: number of lookups
  pub fn new(nz: &Vec<Vec<usize>>, s: usize) -> Self {
    nz.iter().for_each(|nz_dim| assert_eq!(nz_dim.len(), s));
    let subtable_entries = S::materialize_subtables();
    let lookup_polys: Vec<DensePolynomial<F>> = S::to_lookup_polys(&subtable_entries, nz, s);
    let combined_poly = DensePolynomial::merge(&lookup_polys);

    Subtables {
      subtable_entries,
      lookup_polys,
      combined_poly,
      _marker: PhantomData,
    }
  }

  /// Converts subtables T_1, ..., T_{\alpha} and densified multilinear polynomial
  /// into grand products for memory-checking.
  #[tracing::instrument(skip_all, name = "Subtables.to_grand_products")]
  pub fn to_grand_products(
    &self,
    dense: &DensifiedRepresentation<F, S>,
    r_mem_check: &(F, F),
  ) -> Vec<GrandProducts<F>> {
    #[cfg(feature = "multicore")]
    {
      (0..S::num_memories())
        .into_par_iter()
        .map(|i| {
          let subtable = &self.subtable_entries[S::memory_to_subtable_index(i)];
          let j = S::memory_to_dimension_index(i);
          GrandProducts::new_read_only(
            subtable,
            &dense.dim[j],
            &dense.dim_usize[j],
            &dense.read[j],
            &dense.r#final[j],
            r_mem_check,
          )
        })
        .collect()
    }
    #[cfg(not(feature = "multicore"))]
    {
      (0..S::num_memories())
        .map(|i| {
          let subtable = &self.subtable_entries[S::memory_to_subtable_index(i)];
          let j = S::memory_to_dimension_index(i);
          GrandProducts::new(
            subtable,
            &dense.dim[j],
            &dense.dim_usize[j],
            &dense.read[j],
            &dense.r#final[j],
            r_mem_check,
          )
        })
        .collect()
    }
  }

  #[tracing::instrument(skip_all, name = "Subtables.commit")]
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &PolyCommitmentGens<G>,
  ) -> CombinedTableCommitment<G> {
    let (joint_commitment, _blinds) = self.combined_poly.commit(gens, None);
    CombinedTableCommitment::new(joint_commitment)
  }

  #[tracing::instrument(skip_all, name = "Subtables.compute_sumcheck_claim")]
  pub fn compute_sumcheck_claim(&self, eq: &EqPolynomial<F>) -> F {
    let g_operands = self.lookup_polys.clone();
    let hypercube_size = g_operands[0].len();
    g_operands
      .iter()
      .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

    let eq_evals = eq.evals();

    #[cfg(feature = "multicore")]
    let claim = (0..hypercube_size)
      .into_par_iter()
      .map(|k| {
        let g_operands: Vec<F> = (0..S::num_memories()).map(|j| g_operands[j][k]).collect();
        // eq * g(T_1[k], ..., T_\alpha[k])
        eq_evals[k] * S::combine_lookups(&g_operands)
      })
      .sum();

    #[cfg(not(feature = "multicore"))]
    let claim = (0..hypercube_size)
      .map(|k| {
        let g_operands: Vec<F> = (0..S::num_memories()).map(|j| g_operands[j][k]).collect();
        // eq * g(T_1[k], ..., T_\alpha[k])
        eq_evals[k] * S::combine_lookups(&g_operands)
      })
      .sum();

    claim
  }
}
