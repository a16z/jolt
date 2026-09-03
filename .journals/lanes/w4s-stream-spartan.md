# W4-S — shared stream and Spartan + HyperKZG

Date: 2026-09-02. Branch: `wrap/spartan-hyperkzg`.

## Module map

- `stream.rs`: packing/commit kernels, generic compressed stages, degree-bounded KZG stage A, and
  shared-point column reductions.
- `stream/protocol.rs`: statement-specific A→B→C prover/verifier and one multilinear HyperKZG
  opening.
- `stream/types.rs`: statement/proof/error types and exact payload/bincode accounting.
- `jolt-hyperkzg::multi_open`: BDFG20 §4 variable-point batching plus the random-RLC degree check.
- `jolt-transcript::Keccak256Transcript`: Keccak-256 chained digest used only by the outer wrapper.
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
3. **Direct packed opening.** B leaves one `T(s)` claim. Group weights `eq(s_group,g)` define the
   commitment RLC, opened directly at `(r_A,s_slot)`; no point-reduction sumcheck remains.

For a tensor term, the proof sends its factor values at A's final point. The verifier checks the
public tensor formula against A's derived output, uses those values as B's member inputs, and
derives every B output from the one `T(s)` reduced claim. A/B/C singleton outputs are not sent.

## Fiat–Shamir order

`Keccak256Transcript<Fr>::new(b"jolt-wrapper-v1")`:

1. Verifier-key/profile digest; public statement scalars; packed commitments; stage-A encoding.
2. KZG A: input claim; for each round, `C_i`, challenge `r_i`, `s_i(0)`, `s_i(r_i)`.
3. Degree batch: `rho`; `C_shift`. BDFG: `gamma`; `W`; `z`; `W'`.
4. B: factor input claims; one member coefficient per factor; eight compressed round polynomials;
   verifier-derived outputs `eq(column_i,s)T(s)`.
5. Reduced `T(s)`; multilinear HyperKZG fold commitments/challenge and three-point KZG batch at
   `(r_A,s_slot)`.

The degree proof uses shift `L-1-D`, where `L` is the G1 SRS length and `D=5`. The verifier checks
`e(C_shift,[1]_2)=e(sum rho^i C_i,[beta^(L-1-D)]_2)`. This prevents an algebraic prover from using
the full multilinear SRS to submit higher-degree round polynomials. BDFG's two proof elements bind
the three disclosed evaluations of every already-bound round polynomial.

### Chained-digest encoding

Initialization is `state = keccak256(label || zero_pad_to_32)`. Each append is a separate hash:

```text
state' = keccak256(state[32] || round[32] || payload)
round  = uint32 big-endian in bytes 28..31 of an otherwise-zero word
```

Each squeeze hashes `state || round_word` with no payload, promotes its first 16 bytes through
`Fr::from_challenge_bytes`, stores that digest as the next state, and increments the round.
Fr payloads are 32-byte canonical big-endian values. G1 payloads are arkworks canonical compressed
32-byte encodings. A `Label` is right-zero-padded to 32 bytes; `LabelWithCount` is a 24-byte
right-zero-padded label followed by an 8-byte big-endian count. `append_labeled` performs two
hashes: label, then value. The inner Jolt transcript and hash-table replay remain Blake3.

Consensus labels are `sumcheck_claim`, `sumcheck_poly` (with coefficient count),
`opening_claim`, `sumcheck_kzg_commitment`, `sumcheck_kzg_zero`, and `sumcheck_kzg_next`.
Unlabeled events are the key digest, public Fr values, packed commitments, stage-encoding byte,
reduced claims, KZG/Gemini commitments and witnesses, and every squeeze.

### EVM pairing form

Every outer-verifier pairing now uses fixed G2 SRS points. BDFG checks
`e(F+zW',[1]_2)e(-W',[beta]_2)=1`; its degree check already uses fixed `[1]_2` and the shift G2.
HyperKZG expands its cubic divisor into
`e(B-R-z0W,[1]_2)e(-z1W,[beta]_2)e(-z2W,[beta^2]_2)e(-W,[beta^3]_2)=1`.
All divisor scalar multiplication is on G1.

`HyperKZGProof::v[2]` remains on the wire. The fold equations derive its entries for `P_1...`, but
not `P_0(r^2)`. Removing the whole vector soundly needs a second KZG witness for `P_0`'s two-point
opening, saving `32*ell-32` bytes and adding a half-size prover MSM; that is not the plan's claimed
free `32*ell` deletion.

### Co-pointing inventory

- Tensor factors enter B as evaluations at A's row point `r_A`; B shares one column challenge
  `s=(s_group,s_slot)` across every factor.
- The only packed-polynomial output is `T(s)`, claiming
  `sum_g eq(s_group,g) P_g(r_A,s_slot)`. It is already a single-point claim.
- Standalone Spartan opens only `W(ry)`; `Az(rx),Bz(rx),Cz(rx)` are matrix scalars, not packed
  opening claims. In the combined wrapper, indexing W by the T1 row domain and head-aligning its
  inner sumcheck in A makes its row point `r_A`; adding W's fixed column to the shared B batch
  moves it to the same `(r_A,s_slot)` opening.
- Smaller tables pad to the common row/column domains. No second point is required by the current
  claim set.

Thus stage C was redundant after B became a shared-point batch and is deleted. A three-point PCS
opening would add `3(ell+1)+1` G1 and lose to the 20–23-round block it was meant to replace.

Spartan retains compressed stages:

1. R1CS/profile digest; public inputs; witness commitment; `tau`.
2. Outer input, coefficient, degree-three rounds, derived output, then `Az(rx),Bz(rx),Cz(rx)`.
3. Matrix weights; public-column subtraction; inner input/coefficient; degree-two rounds; `W(ry)`.
4. HyperKZG opening of `W` at `ry`.

Spartan splits public inputs into verifier-known values and 128-bit transcript challenges. The
latter serialize as 16-byte big-endian words inside `WrapperProof` and are expanded to Fr before
transcript replay/R1CS evaluation. The 28-challenge fixture occupies 448 B instead of 896 B.

## Exact G-shape bytes and timing

See `w4s-byte-audit.md` for line items.

```text
k=8:  payload 5,952 B; bincode 6,046 B
      setup 14.779 s; commit 1.533 s; proof 2.213 s; verify 0.008 s
      109 ecMul; 108 ecAdd; 8 pairing pairs; 1,112 modeled Fr ops; 301 Keccaks
k=16: payload 5,600 B; bincode 5,680 B
      setup 17.374 s; commit 1.434 s; proof 2.890 s; verify 0.006 s
      95 ecMul; 94 ecAdd; 8 pairing pairs; 1,079 modeled Fr ops; 290 Keccaks
```

## Verification

```text
cargo fmt -q --message-format=short -p jolt-transcript -p jolt-hyperkzg -p jolt-wrapper
cargo clippy -p jolt-transcript --all-targets -q --message-format=short -- -D warnings
cargo clippy -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings
cargo clippy -p jolt-wrapper --all-targets -q --message-format=short -- -D warnings
cargo nextest run -p jolt-wrapper --cargo-quiet
cargo nextest run -p jolt-wrapper --release n3_g_shape_timing \
  --run-ignored ignored-only --cargo-quiet
```

Isolated scratch result: both clippy gates clean; wrapper suite 10 passed / 1 ignored; release gate
passed for both packing factors. Tamper tests reject changed round commitments, next claims,
BDFG witnesses, factor/RLC inputs, both stage streams, packed commitments, and final opening
components.
