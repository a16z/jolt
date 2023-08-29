use ark_ff::PrimeField;

use crate::poly::dense_mlpoly::DensePolynomial;

pub trait JoltStrategy<F: PrimeField>: Sync + Send {
    type Instruction;
  
    fn instructions() -> Vec<Box<dyn InstructionStrategy<F>>>;
    fn primary_poly_degree() -> usize; // TODO: Move poly degree functions to instructions and find max, rename top level to sumcheck_poly_degree()
  
    fn combine_lookups(vals: &[F]) -> F {
      assert_eq!(vals.len(), Self::num_memories());
  
      let mut memory_index: usize = 0;
      let mut sum: F = F::zero();
      for instruction in Self::instructions() {
        sum += instruction
          .combine_lookups(&vals[memory_index..(memory_index + instruction.num_memories())]);
  
        memory_index += instruction.num_memories();
      }
      sum
    }
  
    /// Computes eq * g(T_1[k], ..., T_\alpha[k]) assuming the eq evaluation is the last element in vals
    fn combine_lookups_eq(vals: &[F]) -> F {
      // len(vals) == Self::NUM_MEMORIES + 1
      // len(combine_lookups.vals) == Self::NUM_MEMORIES
      // let mut table_evals: Vec<F> = Vec::with_capacity(Self::NUM_MEMORIES);
      // table_evals.copy_from_slice(&vals[0..Self::NUM_MEMORIES]);
      Self::combine_lookups(&vals[0..Self::num_memories()]) * vals[Self::num_memories()]
    }
  
    /// Converts materialized subtables (`subtable_entries`) and subtable lookup indices (`nz`) into `num_memories` different `DensePolynomials` storing 
    /// `sparsity` evalutations of each subtable at the corresponding index.
    /// 
    /// DensePolynomials are ordered for subtables ST1, ST2, ST3: [ST1[0], ST2[0], ST3[0], ... ST1[C-1], ST2[C-1], ST3[C-1]]
    /// 
    /// Params:
    /// - `subtable_entries`: Materialized subtables of size `num_subtables` x `subtable.memory_size` 
    /// - `nz`: Non-zero indices of size `subtable_dimensionality` x `sparisty`
    fn to_lookup_polys(
      subtable_entries: &Vec<Vec<F>>,
      nz: &Vec<Vec<usize>>,
      s: usize,
    ) -> Vec<DensePolynomial<F>> {
      debug_assert_eq!(subtable_entries.len(), Self::num_subtables());
      debug_assert_eq!(nz.len(), Self::subtable_dimensionality());
  
      (0..Self::num_memories())
        .map(|memory_index| {
          debug_assert_eq!(
              subtable_entries[Self::memory_to_subtable_index(memory_index)].len(), 
              Self::flat_subtables()[Self::memory_to_subtable_index(memory_index)].memory_size()
          );
  
          let mut subtable_lookups: Vec<F> = Vec::with_capacity(s);
          for sparsity_index in 0..s {
            let subtable = &subtable_entries[Self::memory_to_subtable_index(memory_index)];
            let nz = nz[Self::memory_to_dimension_index(memory_index)][sparsity_index];
  
            subtable_lookups.push(subtable[nz]);
          }
          DensePolynomial::new(subtable_lookups)
        })
        .collect()
    }
  
    fn num_instructions() -> usize {
      Self::instructions().len()
    }
  
    /// Total number of subtables across all instructions (potentially duplicated).
    fn num_subtables() -> usize {
      let mut sum = 0;
      for instruction in Self::instructions() {
        sum += instruction.num_subtables()
      }
      sum
    }
  
    /// Total number of memories across all subtables of associated instructions.
    fn num_memories() -> usize {
      let mut sum = 0;
      for instruction in Self::instructions() {
        sum += instruction.num_memories()
      }
      sum
    }
  
    fn materialize_subtables() -> Vec<Vec<F>> {
      let mut subtables: Vec<Vec<F>> = Vec::with_capacity(Self::num_subtables());
  
      for instruction in Self::instructions() {
        for subtable in instruction.subtables() {
          subtables.push(subtable.materialize());
        }
      }
      subtables
    }
  
    fn evaluate_memory_mle(memory_index: usize, point: &[F]) -> F {
      Self::flat_subtables()[Self::memory_to_subtable_index(memory_index)].evaluate_mle(point)
    }
  
    /// Maps an index [0, num_memories) -> [0, num_subtables)
    fn memory_to_subtable_index(i: usize) -> usize {
      i % Self::num_subtables()
    }
  
    /// Maps an index [0, num_memories) -> [0, subtable_dimensionality]
    fn memory_to_dimension_index(i: usize) -> usize {
      i / Self::num_subtables()
    }
  
    fn flat_subtables() -> Vec<Box<dyn SubtableStrategy<F>>> {
      Self::instructions()
        .iter()
        .map(|instruction| instruction.subtables())
        .flatten()
        .collect()
    }
  
    /// Maximum dimensionality of all subtables of all instructions. Used for splitting big table indices
    /// into memory indices.
    // TODO: Better name?
    fn subtable_dimensionality() -> usize {
      // TODO: assert uniform dimensionality here or in macros?
      Self::flat_subtables()
        .iter()
        .map(|strategy| strategy.dimensions())
        .max()
        .unwrap()
    }
  }
  
  pub trait InstructionStrategy<F: PrimeField> {
    fn subtables(&self) -> Vec<Box<dyn SubtableStrategy<F>>>;
  
    fn num_subtables(&self) -> usize {
      self.subtables().len()
    }
  
    fn num_memories(&self) -> usize {
      let mut memories = 0;
      for subtable in self.subtables() {
        memories += subtable.dimensions();
      }
      memories
    }
  
    fn combine_lookups(&self, vals: &[F]) -> F;
  
    fn g_poly_degree(&self) -> usize;
  
    // TODO: Unneeded?
    fn evaluate_subtable_mle(&self, subtable_index: usize, point: &[F]) -> F {
      self.subtables()[subtable_index].evaluate_mle(point)
    }
  }
  
  pub trait SubtableStrategy<F: PrimeField> {
    /// C: Number of memories in this subtable.
    fn dimensions(&self) -> usize;
  
    /// M: Size of a single memory of this subtable.
    fn memory_size(&self) -> usize;
  
    /// Fill out a single M-sized subtable.
    fn materialize(&self) -> Vec<F>;
  
    /// Evaluate the MLE of this subtable at a single point.
    fn evaluate_mle(&self, point: &[F]) -> F;
  }
  