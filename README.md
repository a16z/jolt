![imgs/lasso_logo.png](imgs/lasso_logo.png)

# Lasso
Lookup Arguments via Sum-check and Sparse polynomial commitments, including for Oversized tables

## Overview
- See [EngineeringOverview.md](EngineeringOverview.md) for a high level technical outline
- See [HowToTable.md](HowToTable.md) for instructions on adding a new table type

## Current usage
```rust
  let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_M);
  let mut dense: DensifiedRepresentation<Fr, C, EqSubtableStrategy> = DensifiedRepresentation::from(&lookup_matrix);
  let (gens, commitment) = dense.commit::<EdwardsProjective>();

  let proof = 
    SparsePolynomialEvaluationProof::<EdwardsProjective, C, M, SubtableStrategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
    );
```

## Cmds
- `cargo build --release`
- `cargo run --release -- --name <bench_name>`
- `cargo run --release -- --name <bench_name> --chart`: Display performance gant chart
- `cargo run --release --name <bench_name> --features ark-msm`: Run without MSM small field optimizations
- `sudo cargo flamegraph`

*Note on benching / flamegraphing: Turn off the parallel feature in Cargo.toml (`multicore`) and / or `export RAYON_NUM_THREADS=1` to make flamegraph more interpretable. Turning off `multicore` and leaving `unset RAYON_NUM_THREADS` allows testing Arkworks MSM parallelism without Lasso parallelism.*