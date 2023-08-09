# Lasso

![imgs/lasso_logo.png](imgs/lasso_logo.png)

Lookup Arguments via Sum-check and Sparse polynomial commitments, including for Oversized tables. Code originally derived from [Spartan](https://github.com/microsoft/Spartan) by Srinath Setty.

## Overview

-   See [EngineeringOverview.md](EngineeringOverview.md) for a high level technical outline
-   See [HowToTable.md](HowToTable.md) for instructions on adding a new table type

## Current usage

```rust
  let lookup_matrix = SparseLookupMatrix::new(nz, log_M);
  let mut dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&lookup_matrix);
  let commitment = dense.commit::<G>(&gens);

  let proof =
    SparsePolynomialEvaluationProof::<G, C, M, SubtableStrategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
    );
```

## Cmds

-   `cargo build --release`
-   `cargo run --release -- --name <bench_name>`
-   `cargo run --release -- --name <bench_name> --chart`: Display performance gant chart
-   `cargo run --release --features ark-msm -- --name <bench_name>`: Run without MSM small field optimizations
-   `sudo cargo flamegraph`

_Note on benching / flamegraphing: Turn off the parallel feature in Cargo.toml (`multicore`) and / or `export RAYON_NUM_THREADS=1` to make flamegraph more interpretable. Turning off `multicore` and leaving `unset RAYON_NUM_THREADS` allows testing Arkworks MSM parallelism without Lasso parallelism._

## Disclaimer

*This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. a16z is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. See a16z.com/disclosures for more info.*

*This repository was forked from https://github.com/arkworks-rs/spartan.*