# Surge
![surge](imgs/surge.jpg)
Here, we Surge. Ever heard of the Cola? Yeah. Like that, but proving. Proving zero-knowledge succinctness things.

# Todo
- [ ] Generalize `SubtableEvaluations` beyond eq table
- [ ] Generalize `Densified.materialize_table()` beyond eq: Evaluate $T[k] \forall k \in [0, ... M]$ 
- [ ] Generalize `SparseLookupMatrix.evaluate_mle()` beyond eq (reuse `Densified.materialize_table()`)
- [ ] Lazy version of `SubtableStrategy::materialize_subtables` which does not materialize all [M] copies up-front, rather it evaluates subT(k) in O(1) time.
- [ ] Consider containing `DensifiedRepresentation` as a property of a mutable `SparseLookupMatrix` instead of requiring the implementer to hold them separately.
- [ ] Investigate multiple dense PCS (notably IPA)
- [ ] Investigate MSM speedups ([paper](https://eprint.iacr.org/2022/1400.pdf), [arkworks](https://github.com/arkworks-rs/algebra/blob/c015ea331674368461ff466bc7cbc69806f61628/ec/src/scalar_mul/variable_base/mod.rs#L112-L122))
- [ ] Clippy / Cargo fmt
- [ ] SubtableStrategy: move M to generic param

## Current usage
```rust
  let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_M);
  let mut dense: DensifiedRepresentation<Fr, C, EqSubtableStrategy> = DensifiedRepresentation::from(&lookup_matrix);
  let (gens, commitment) = dense.commit::<EdwardsProjective>();
  let eval = lookup_matrix.evaluate_mle(&flat_r);

  let proof = SparsePolynomialEvaluationProof::<EdwardsProjective, C>::prove(
    &mut dense,
    &r,
    &eval,
    &gens,
    &mut prover_transcript,
    &mut random_tape,
  );
```

## Cmds
- `cargo build --release`
- `sudo cargo flamegraph`: Run `main.rs` and generate `flamegraph.svg`
- `cargo bench`
- `cargo clippy --fix`

*Note on benching / flamegraphing: Turn off the parallel feature in Cargo.toml to make flamegraph more interpretable"

## Notes
Densified::from_sparse
- sparse.nz – already split into nz_1, nz_2, ... nz_c
- Basically just need actual tables at `combine_subtable_evaluations` step
- Don't need separate `materialize_table` can be done in `combine_subtable_evaluations`
- The `GeneralizedScalarProduct` which evaluates $T[r] = g(T_1[r_1], ... T_\alpha[r_c])$  uses `DensifiedRepresentation.combine_subtable_evaluations().subtable_evaluations`

Many of the tables can compute subT[k] in O(1) time and don't need the eq(k, r) amortization across all M. This is an artifact of supporting Sparkplug.

Glue strategy
- Add a another combination function analogous to 'g' which combines tables by adding some high order selector bits, which further generalizes the `GeneralizedScalarProduct` thing.
- Still commitments and memory checking happen on a per-table level (although you can imagine combining these across Tables)
