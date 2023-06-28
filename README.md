# Surge
![surge](imgs/surge.jpg)
Here, we Surge. Ever heard of the Cola? Yeah. Like that, but proving. Proving zero-knowledge succinctness things.

# Todo
- [x] Generalize `Densified.materialize_table()` beyond eq: Evaluate $T[k] \forall k \in [0, ... M]$ 
- [x] `Subtables` / `memory_checking.rs`: Move 'K' / 'ALPHA' into `Subtables`. `memory_checking.rs` should not handle the `K` generic.
- [x] SubtableStrategy: move M to generic param
- [ ] SubtableStrategy: Consider removing M as generic param and moving onto `AndSubtableStrategy` exclusively. If not, remove `m` from params of `SubtableStrategy::materialize_subtable`.
- [ ] Lazy version of `SubtableStrategy::materialize_subtables` which does not materialize all [M] copies up-front, rather it evaluates subT(k) in O(1) time.
- [ ] Consider containing `DensifiedRepresentation` as a property of a mutable `SparseLookupMatrix` instead of requiring the implementer to hold them separately.
- [ ] Investigate multiple dense PCS (notably IPA)
- [x] Investigate MSM speedups ([paper](https://eprint.iacr.org/2022/1400.pdf), [arkworks](https://github.com/arkworks-rs/algebra/blob/c015ea331674368461ff466bc7cbc69806f61628/ec/src/scalar_mul/variable_base/mod.rs#L112-L122))
- [ ] Clippy / Cargo fmt
- [ ] Consider killing const generics as a whole.

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

*Note on benching / flamegraphing: Turn off the parallel feature in Cargo.toml to make flamegraph more interpretable*

## Notes
Many of the tables can compute subT[k] in O(1) time and don't need the eq(k, r) amortization across all M. This is an artifact of supporting Sparkplug.

### Glue strategy
- Add a another combination function analogous to 'g' which combines tables by adding some high order selector bits, which further generalizes the `GeneralizedScalarProduct` thing.
- Still commitments and memory checking happen on a per-table level (although you can imagine combining these across Tables)
