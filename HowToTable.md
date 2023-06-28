# Custom Subtable Strategies

This document explains how to fill out custom Subtable Strategies that implement the `SubtableStrategy` trait. The trait provides a set of methods and associated constants that need to be implemented to define a custom subtable strategy. The trait is defined as follows:

```rust
#[cfg(test)]
pub mod test;

pub trait SubtableStrategy<F: PrimeField, const C: usize, const M: usize> {
  const NUM_SUBTABLES: usize;
  const NUM_MEMORIES: usize;

  fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> [Vec<F>; Self::NUM_SUBTABLES];

  fn evaluate_subtable_mle(subtable_index: usize, r: &[Vec<F>; C], point: &Vec<F>) -> F;

  fn combine_lookups(vals: &[F; Self::NUM_MEMORIES]) -> F;

  fn sumcheck_poly_degree() -> usize;

  fn memory_to_subtable_index(memory_index: usize) -> usize;

  fn memory_to_dimension_index(memory_index: usize) -> usize;

  fn to_lookup_polys(
    subtable_entries: &[Vec<F>; Self::NUM_SUBTABLES],
    nz: &[Vec<usize>; C],
    s: usize,
  ) -> [DensePolynomial<F>; Self::NUM_MEMORIES];
}
```

## Implementing the SubtableStrategy Trait

To implement the `SubtableStrategy` trait, you need to define the following methods and associated constants:

### 1. `NUM_SUBTABLES` and `NUM_MEMORIES`

These associated constants specify the number of subtables and memories used in the strategy. Replace `NUM_SUBTABLES` and `NUM_MEMORIES` with the desired values. Often these constant parameters will be a function of `C`.

```rust
const NUM_SUBTABLES: usize = <desired_value>;
const NUM_MEMORIES: usize = <desired_value>;
```

### 2. `materialize_subtables`

This method materializes the subtables for a given size `m` and point `r`.

*Note: Most subtables will not use the parameter `r`, this is vestigial and used for Spark (see `SparkSubtableStrategy`).*

```rust
fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> [Vec<F>; Self::NUM_SUBTABLES] {
    // Implementation goes here
}
```

Replace the implementation with your logic to generate and return the materialized subtables.

### 3. `evaluate_subtable_mle`

This method evaluates the mutlilinear extension (MLE) of a subtable at the given `point`. The `subtable_index` parameter specifies which subtable to evaluate.

*Note: Most subtables will not use the parameter `r`, this is vestigial and used for Spark (see `SparkSubtableStrategy`).*


```rust
fn evaluate_subtable_mle(subtable_index: usize, r: &[Vec<F>; C], point: &Vec<F>) -> F {
    // Implementation goes here
}
```

Replace the implementation with your logic to evaluate and return the MLE of the specified subtable at the given point.

### 4. `combine_lookups`
This method defines the g function that computes the combined lookup value T[r] based on the lookup values of different subtables and memories.

```rust
fn combine_lookups(vals: &[F; Self::NUM_MEMORIES]) -> F {
    // Implementation goes here
}
```

Replace the implementation with your logic to combine the lookup values and return the combined result.

### 5. `sumcheck_poly_degree`

This method specifies the total degree of the g function, considering combine_lookups as a log(m)-variate polynomial. The total degree determines the number of evaluation points in each sumcheck round.

```rust
fn sumcheck_poly_degree() -> usize {
    // Specify the desired sumcheck polynomial degree
}
```

Replace the implementation with the desired value for the sumcheck polynomial degree.

### Remaining
The remaining trait functions should be implemented by default.