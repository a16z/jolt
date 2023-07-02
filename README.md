# Surge
![surge](imgs/surge.jpg)
Here, we Surge. Ever heard of the Cola? Yeah. Like that, but proving. Proving zero-knowledge succinctness things.

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
        &spark_randomness,
        &eq_randomness,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
    );
```

## Cmds
- `cargo build --release`
- `cargo build --release -- --chart`
- `sudo cargo flamegraph`
- `cargo bench`

*Note on benching / flamegraphing: Turn off the parallel feature in Cargo.toml (`multicore`) and / or `export RAYON_NUM_THREADS=1` to make flamegraph more interpretable. Turning off `multicore` and leaving `unset RAYON_NUM_THREADS` allows testing Arkworks MSM parallelism without Surge parallelism.*