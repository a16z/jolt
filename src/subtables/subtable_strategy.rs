use ark_ff::PrimeField;

use crate::poly::dense_mlpoly::DensePolynomial;

pub trait SubtableStrategy<F: PrimeField, const C: usize, const M: usize> {
    const NUM_SUBTABLES: usize;
    const NUM_MEMORIES: usize;
  
    /// Materialize subtables indexed [1, ..., \alpha]
    fn materialize_subtables() -> Vec<Vec<F>>;
  
    /// Evaluates the MLE of a subtable at the given point. Used by the verifier in memory-checking.
    ///
    /// Params
    /// - `subtable_index`: Which subtable to evaluate the MLE of. Ranges 0..ALPHA
    /// - `point`: Point at which to evaluate the MLE
    fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F;
  
    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(vals: &[F]) -> F;
  
    /// The total degree of `g`, i.e. considering `combine_lookups` as a log(m)-variate polynomial.
    /// Determines the number of evaluation points in each sumcheck round.
    fn g_poly_degree() -> usize;
  
    /// Computes eq * g(T_1[k], ..., T_\alpha[k]) assuming the eq evaluation is the last element in vals
    fn combine_lookups_eq(vals: &[F]) -> F {
      // len(vals) == Self::NUM_MEMORIES + 1
      // len(combine_lookups.vals) == Self::NUM_MEMORIES
      // let mut table_evals: Vec<F> = Vec::with_capacity(Self::NUM_MEMORIES);
      // table_evals.copy_from_slice(&vals[0..Self::NUM_MEMORIES]);
      Self::combine_lookups(&vals[0..Self::NUM_MEMORIES]) * vals[Self::NUM_MEMORIES]
    }
  
    /// Total degree of eq * g(T_1[k], ..., T_\alpha[k])
    fn sumcheck_poly_degree() -> usize {
      Self::g_poly_degree() + 1
    }
  
    fn memory_to_subtable_index(memory_index: usize) -> usize {
      assert_eq!(Self::NUM_SUBTABLES * C, Self::NUM_MEMORIES);
      assert!(memory_index < Self::NUM_MEMORIES);
      memory_index % Self::NUM_SUBTABLES
    }
  
    fn memory_to_dimension_index(memory_index: usize) -> usize {
      assert_eq!(Self::NUM_SUBTABLES * C, Self::NUM_MEMORIES);
      assert!(memory_index < Self::NUM_MEMORIES);
      memory_index / Self::NUM_SUBTABLES
    }
  
    /// Converts subtables T_1, ..., T_{\alpha} and lookup indices nz_1, ..., nz_c
    /// into log(m)-variate "lookup polynomials" E_1, ..., E_{\alpha}.
    fn to_lookup_polys(
      subtable_entries: &Vec<Vec<F>>,
      nz: &Vec<Vec<usize>>,
      s: usize,
    ) -> Vec<DensePolynomial<F>> {
      (0..Self::NUM_MEMORIES).map(|i: usize|{
        let mut subtable_lookups: Vec<F> = Vec::with_capacity(s);
        for j in 0..s {
          let subtable = &subtable_entries[Self::memory_to_subtable_index(i)];
          let nz = nz[Self::memory_to_dimension_index(i)][j];
          subtable_lookups.push(subtable[nz]);
        }
        DensePolynomial::new(subtable_lookups)
      }).collect()
    }
  }