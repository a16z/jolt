use ark_ff::PrimeField;

use crate::poly::dense_mlpoly::DensePolynomial;

pub trait JoltStrategy<F: PrimeField>: Sync + Send {
  type Instruction;

  /// All instructions used by this VM.
  fn instructions() -> Vec<Box<dyn InstructionStrategy<F>>>;

  /// Computes g(T_1[k], ..., T_\alpha[k])
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
    assert_eq!(vals.len(), Self::num_memories() + 1);
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

  /// Materialize each subtable across all instructions. May include duplicate subtable materialization
  /// if instructions reuse subtables, but does not duplicate per subtable dimension.
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

  /// Degree of the primary sumcheck univariate polynomial. Describes number of points required to uniquely define
  /// the associated polynomial.
  fn primary_poly_degree() -> usize {
    let max_instruction_degree = Self::instructions()
    .iter()
    .map(|instruction| instruction.g_poly_degree())
    .max()
    .unwrap();
    
    // Add evaluation of eq
    max_instruction_degree + 1
  }

  /// Maps an index [0, num_memories) -> [0, num_subtables)
  fn memory_to_subtable_index(i: usize) -> usize {
    i % Self::num_subtables()
  }

  /// Maps an index [0, num_memories) -> [0, subtable_dimensionality]
  fn memory_to_dimension_index(i: usize) -> usize {
    i / Self::num_subtables()
  }

  /// Flattened subtables across all instructions, including duplicates. Defines order
  /// of memories.
  fn flat_subtables() -> Vec<Box<dyn SubtableStrategy<F>>> {
    Self::instructions()
      .iter()
      .map(|instruction| instruction.subtables())
      .flatten()
      .collect()
  }

  /// Maximum dimensionality of all subtables of all instructions. Used for splitting big table indices
  /// into memory indices.
  fn subtable_dimensionality() -> usize {
    // TODO: assert uniform dimensionality here or in macros?
    Self::flat_subtables()
      .iter()
      .map(|strategy| strategy.dimensions())
      .max()
      .unwrap()
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
}

pub trait InstructionStrategy<F: PrimeField> {
    /// Returns subtables used by this InstructionStrategy in order.
    fn subtables(&self) -> Vec<Box<dyn SubtableStrategy<F>>>;

    /// Combines g(T1[0], T2[0], T3[0], ... T1[C-1], T2[C-1], T3[C-1])
    fn combine_lookups(&self, vals: &[F]) -> F;

    /// Degree of the g polynomial considering all T (subtable) polynomials a 
    /// degree-1 univariate polynomial.
    fn g_poly_degree(&self) -> usize;

    /// Evaluate the MLE of the `subtable_index` subtable with ordered defined in 
    /// `InstructionStrategy::subtables()`.
    fn evaluate_subtable_mle(&self, subtable_index: usize, point: &[F]) -> F {
        self.subtables()[subtable_index].evaluate_mle(point)
    }

    /// Total number of unique subtables.
    fn num_subtables(&self) -> usize {
        self.subtables().len()
    }

    /// Total number of memories across all subtables.
    fn num_memories(&self) -> usize {
        let mut memories = 0;
        for subtable in self.subtables() {
        memories += subtable.dimensions();
        }
        memories
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
