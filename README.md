# Lasso

![imgs/lasso_logo.png](imgs/lasso_logo.png)

Lookup Arguments via Sum-check and Sparse polynomial commitments, including for Oversized tables. 

*This repository was forked from https://github.com/arkworks-rs/spartan. Original Spartan [code](https://github.com/microsoft/Spartan) by Srinath Setty.*

## Resources

-   [Introducing Lasso and Jolt](https://a16zcrypto.com/posts/article/introducing-lasso-and-jolt/)
-   [Understanding Lasso and Jolt, from theory to code](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/)
-   See [EngineeringOverview.md](EngineeringOverview.md) for a high level technical outline
-   See [HowToTable.md](HowToTable.md) for instructions on adding a new table type
-   [Lasso paper](https://people.cs.georgetown.edu/jthaler/Lasso-paper.pdf)
-   [Jolt paper](https://people.cs.georgetown.edu/jthaler/Jolt-paper.pdf)

## Current usage

```rust
  let mut dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&nz, log_M);
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

Note: requires nightly Rust

-   `cargo build --release`
-   `cargo run --release -- --name <bench_name>`
-   `cargo run --release -- --name <bench_name> --chart`: Display performance gant chart
-   `cargo run --release --features ark-msm -- --name <bench_name>`: Run without MSM small field optimizations
-   `sudo cargo flamegraph`

_Note on flamegraphing: Turn off the parallel feature in Cargo.toml (`multicore`) and / or `export RAYON_NUM_THREADS=1` to make flamegraph more interpretable._

## Disclaimer

*This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. a16z is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. See a16z.com/disclosures for more info.*
