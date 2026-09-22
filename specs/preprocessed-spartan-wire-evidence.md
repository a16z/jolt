# Wire v1 implementation evidence

Code commit `c692c8273fd9616f0505f2542d08fc5e75a3f9bb`, isolated `preprocessed-wire`, base `c281860b26b4550f79343548d673080cb8a080a4`. No Akita/field-inline source or dependency changes; Cargo.lock is unchanged, SHA256 `258cb2f0c7d6cb7c99a912d57c4f7c212fbf91526b439dc859917a01f1662ef3`.

The entry point `jolt_spartan_verifier::preprocessed::wire::verify_bytes` authenticates the canonical computation key and setup, bounds geometry and input lengths, checks canonical field/group representations, decodes the fixed-layout proof, and invokes the complete existing verifier. `jolt-crypto` owns checked compressed G1 decoding. Existing typed verification and transcript semantics are unchanged. The exact ABI, limits and claim-to-code ownership are in [preprocessed-spartan-wire.md](preprocessed-spartan-wire.md).

## Observed controls

The first compile succeeded. Initial tests exposed an honest all-zero empty-N2 proof with shortened compressed rounds: the provisional exact3/2-width encoder returned `RoundEncoding`. Initial run `433b0f7c-7482-49cf-beec-ac32451c13ed` exited100 (3pass/1fail). The final transport stores one bounded original-width tag plus fixed degree-width slots, requiring unused slots to be zero. Allocation uses the key-derived degree, never the tag. Decoding reconstructs the exact typed message length; no padding enters Fiat–Shamir.

Repair run `95115eca-f9a9-4c73-80d0-d70bcb32488d` passed4/4. Final regression run `6f2f0c72-2606-40ba-9d34-549e84f2ed81` passed14/14, with15 unrelated tests excluded by the explicit filter. It includes canonical key vectors, the original full v2 transcript/payload regression, actual setup binding, private SPARK controls, a positive zero-factor network, and the new wire controls. New tests reject truncation/trailing bytes, wrong proof/header/key/setup, oversized geometry, invalid group/field values (including Fr modulus alias), malformed public inputs, changed public/proof values, width tags0/>degree, nonzero padding, and extra round data in a zero-round layer. Honest toy and all-zero empty-N2 proofs pass full byte verification and canonical re-encoding. The curve-owner test also rejects noncanonical infinity and x=0 (noncurve), while accepting identity and the generator.

The toy golden is `crates/jolt-spartan-prover/tests/fixtures/preprocessed-wire-v1-toy.bin`: **9,149 bytes**,248 Fr slots and36 G1, including52-byte header and9 width tags. SHA256 `4479eddec1b1291240169650799eff76f0190d592afd1b8944603cdbb9dfd151`. It came from the accepted live4×8/p3 prover and is pinned by a permanent test; temporary fixture-writing code was removed. An independent Python grammar parser checked exact consumption, scalar bounds, compressed-point curve/flag rules, tags/padding and counts without importing Rust/arkworks. This is encoding evidence, not an independent algebraic verifier or security proof.

The intentional `wire_norm_fixture` example hash-authenticates the historical measured norm proof/key **before** bincode parsing, regenerates only its known-beta7 test setup, and runs the production byte verifier. It never proves. The accepted norm wire artifact is **54,547 bytes**,1,595 Fr slots and100 G1, including52-byte header and255 tags. SHA256 `0a5c4932d266868cfd8f2664d28ab1e99c2ebe294d4c361b23c712fc1130c6af`. The same independent parser accepted it. Differences from historical bincode lengths9,176/54,716 bytes are transport framing differences, not protocol compression gains.

## Commands and retained evidence

Task root is `/private/var/folders/g4/6bp_sb413fjg5gtf5gc6p85c0000gp/T/jolt-wrapper-goal-2zc2x9g1`. Logs, explicit `.exit` records, initial source/lock hashes, independent parser/results and frozen diagnostic executable are under `wire-run/`. Commands ran from `preprocessed-wire` with `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=../crypto-r1cs/target`:

```
cargo nextest run --locked --offline -p jolt-spartan-prover -p jolt-spartan-verifier -p jolt-crypto --features jolt-spartan-prover/preprocessed --lib -E 'test(preprocessed) | test(checked_compressed_point)' --cargo-quiet
cargo clippy --locked --offline -p jolt-spartan-prover -p jolt-spartan-verifier -p jolt-crypto --features jolt-spartan-prover/preprocessed --all-targets -q -- -D warnings
cargo build --locked --offline -p jolt-spartan-prover --features preprocessed --example wire_norm_fixture -q
```

All exited0 (`nextest-final`, `clippy-initial`, `norm-build` logs/exit files). `cargo fmt --all -- --check`, `git diff --check`, and `taplo fmt --check crates/jolt-spartan-prover/Cargo.toml` also passed. Neither Spartan crate has a default feature; the exercised explicit preprocessed feature selects the BN254 closure. No duplicate minimal-mode test run, workspace host/ZK suite, or unrelated Akita integration run is represented as performed.

The frozen norm conversion command and all hashes are `wire-run/norm.command.json`; code head c692, executable SHA256 `bf16f23ba6c161ee238975995bd65a83e48fb85d16f4a6d8aacd9e3b3b7694af`. Governor command `python3 ../recursion-blake-run/govern.py ../wire-run/norm.command.json`, session37220, exit0. Raw stage/output and resource record are `recursion-blake-run/logs/wire-norm-c692c8273.{log,result.json}`. Limits32GiB sampled process-group RSS/900seconds/12GiB free disk; no stop, cleanup error or surviving group member. Elapsed42.689seconds includes debug-build setup regeneration; sampled RSS105,578,496 bytes, child rusage108,445,696 bytes, minimum free disk16,460,173,312 bytes. These are bounded-control observations, not a verifier benchmark. No proof regeneration, repetitions, remote pushes or toolchain installation occurred.

Remaining unverified: independent security review of this parser; deployed network framing; Solidity decoding/gas; operational joint extraction/ROM; ZK; setup ceremony provenance and complete Akita-wrapper composition. Typed serde APIs remain outside the bounded byte-entry contract.
