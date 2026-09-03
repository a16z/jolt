# W4-S — shared stream and Spartan + HyperKZG

Date: 2026-09-02. Branch: `wrap/spartan-hyperkzg`.

## Module map

- `stream.rs`: packing/commit kernels, generic compressed stages, degree-bounded KZG stage A,
  shared-point column reductions, and final claim reduction.
- `stream/protocol.rs`: statement-specific A→B→C prover/verifier and one multilinear HyperKZG
  opening.
- `stream/types.rs`: statement/proof/error types and exact payload/bincode accounting.
- `jolt-hyperkzg::multi_open`: BDFG20 §4 variable-point batching plus the random-RLC degree check.
- `spartan.rs`: clear Spartan outer/inner sumchecks over `ConstraintMatrices<Fr>` with public
  columns `[0, 1 + |x|)` and witness columns `[1 + |x|, num_vars)`.
- `jolt-r1cs::ConstraintMatrices::column_range_contributions`: public-range evaluation in one pass
  per matrix.

## Packed layout

For group `g`, slot `j < k`, and row `i`:

```text
P_g[i*k+j] = column[g*k+j][i]
opening point = (r_row, s_slot)
```

The group domain pads to `next_power_of_two(ceil(column_count/k))`. Missing slots/groups are zero;
missing groups emit no commitments. `PackingLayout` owns the group/slot split and truncated group
eq weights. Homogeneous bit groups use `g1_bit_columns_msm`, homogeneous u16 groups use the
small-scalar MSM, and mixed/Fr groups use HyperKZG commitment.

## Stream stages

1. **A — rows.** `StageAEncoding::Compressed` uses the generic clear stage. The gate selects
   `KzgCommitted`: 17 degree-at-most-five round commitments, two Fr evaluations per round, one
   batched shifted degree proof, and the BDFG variable-point opening.
2. **B — columns.** One degree-two `prove_batch` stream contains every tensor-factor reduction.
   All members share the same `log2(padded_columns)` challenges. Transcript-derived member RLCs
   combine them without sending coefficients.
3. **C — packed opening.** One reduced `T(s)` claim becomes an eq-weighted RLC of the packed
   polynomials. A degree-two claim reduction and one HyperKZG proof open that RLC.

For a tensor term, the proof sends its factor values at A's final point. The verifier checks the
public tensor formula against A's derived output, uses those values as B's member inputs, and
derives every B output from the one `T(s)` reduced claim. A/B/C singleton outputs are not sent.

## Fiat–Shamir order

`Blake3Transcript<Fr>::new(b"jolt-wrapper-v1")`:

1. Verifier-key/profile digest; public statement scalars; packed commitments; stage-A encoding.
2. KZG A: input claim; for each round, `C_i`, challenge `r_i`, `s_i(0)`, `s_i(r_i)`.
3. Degree batch: `rho`; `C_shift`. BDFG: `gamma`; `W`; `z`; `W'`.
4. B: factor input claims; one member coefficient per factor; eight compressed round polynomials;
   verifier-derived outputs `eq(column_i,s)T(s)`.
5. Reduced `T(s)`; claim RLC coefficient; C input/coefficient; compressed C rounds; derived output.
6. Multilinear HyperKZG fold commitments/challenge and three-point KZG batch.

The degree proof uses shift `L-1-D`, where `L` is the G1 SRS length and `D=5`. The verifier checks
`e(C_shift,[1]_2)=e(sum rho^i C_i,[beta^(L-1-D)]_2)`. This prevents an algebraic prover from using
the full multilinear SRS to submit higher-degree round polynomials. BDFG's two proof elements bind
the three disclosed evaluations of every already-bound round polynomial.

Spartan retains compressed stages:

1. R1CS/profile digest; public inputs; witness commitment; `tau`.
2. Outer input, coefficient, degree-three rounds, derived output, then `Az(rx),Bz(rx),Cz(rx)`.
3. Matrix weights; public-column subtraction; inner input/coefficient; degree-two rounds; `W(ry)`.
4. HyperKZG opening of `W` at `ry`.

## Exact G-shape bytes and timing

See `w4s-byte-audit.md` for line items.

```text
k=8:  payload 7,232 B; bincode 7,347 B
      setup 14.712 s; commit 2.056 s; proof 8.220 s; verify 0.122 s
k=16: payload 6,944 B; bincode 7,046 B
      setup 28.291 s; commit 1.454 s; proof 11.614 s; verify 0.042 s
```

## Verification

```text
cargo fmt -q --message-format=short -p jolt-hyperkzg -p jolt-wrapper
cargo clippy -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings
cargo clippy -p jolt-wrapper --all-targets -q --message-format=short -- -D warnings
cargo nextest run -p jolt-wrapper --cargo-quiet
cargo nextest run -p jolt-wrapper --release n3_g_shape_timing \
  --run-ignored ignored-only --cargo-quiet
```

Isolated scratch result: both clippy gates clean; wrapper suite 10 passed / 1 ignored; release gate
passed for both packing factors. Tamper tests reject changed round commitments, next claims,
BDFG witnesses, factor/RLC inputs, all three stage streams, packed commitments, and final opening
components.
