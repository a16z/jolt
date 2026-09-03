# W4-S — shared stream and Spartan + HyperKZG

Date: 2026-09-02. Branch: `wrap/spartan-hyperkzg`.

## Result

- `stream.rs`: low-variable column packing, type-directed commitment kernels, staged `prove_batch`,
  tensor column batching, and claim reduction.
- `stream/protocol.rs`: canonical A→B→C verifier driver, commitment/evaluation RLC, and one
  HyperKZG opening.
- `stream/types.rs`: proof/statement types, errors, and exact payload/bincode size accounting.
- `spartan.rs`: plain Spartan outer/inner sumchecks over `ConstraintMatrices<Fr>` with public columns
  `[0, 1 + |x|)` and committed witness columns `[1 + |x|, num_vars)`.
- `tests/stream_synthetic.rs`: `2^12 × (40 bit + 20 u16)`, degree-5 row relation, stages A/B/C,
  full opening verification, 33/237-column padding, and link/point/index/tensor/claim tampers.
- `tests/spartan_core.rs`: satisfiable `m = n = 2^12`, `|x| = 50`, public-column terms, and
  public/round/claim/opening tampers.
- `tests/stream_stage.rs`: full, head-aligned, and tail-aligned members in one 12-round batch.
- `jolt-r1cs::ConstraintMatrices::column_range_contributions`: public range in one pass per matrix.

## Packed layout

For group `g`, slot `j < k`, and row `i`:

```text
P_g[i * k + j] = column[g * k + j][i]
opening point   = (r_row, s_slot)
                  high vars  low log2(k) vars
```

Missing slots in the last group are zero. The group domain is padded to
`next_power_of_two(ceil(column_count / k))`; missing groups are implicit zero polynomials and emit
no commitment. `PackingLayout` owns the group/slot split and truncated group eq weights.
Homogeneous bit groups use
`g1_bit_columns_msm`; homogeneous `u16` groups use `Bn254::g1_affine_msm_small`; mixed or `Fr`
groups use the full-field HyperKZG commitment path. All paths commit the same dense layout.

## Sumcheck stages

1. **A — row space.** Any `ProveRounds<Fr>` batch; shorter head-aligned members are scaled by
   `2^(max_rounds - offset - member_rounds)` inside the stage adapter.
2. **B — column space.** `ColumnBatching` proves sparse tensor forms
   `sum Q[j_1,...,j_D] * product T(j_d)` in `D * log2(columns)` degree-2 rounds. Its input is the
   stage-A final relation value; its final values are `T(s_1), ..., T(s_D)`.
3. **C — claim reduction.** For each stage-B point, the verifier derives
   `T(s_i) = sum_g eq(s_group_i, g) P_g(r_A, s_slot_i)`. `ClaimReduction` proves
   `sum_g q_g(x) P_g(x)`, where
   `q_g = sum_i rho_i eq(s_group_i, g) eq((r_A, s_slot_i), x)`, in degree-2 rounds. The final
   weights `q_g(r)` define the commitment/evaluation RLC and one HyperKZG opening.

`StageProof` stores only compressed round polynomials. `WrapperProof::stage_claims` transmits each
member output for heterogeneous batches; the verifier recomputes every claim before absorbing it.
`StageResult::member_point` owns each offset/round slice.

## Fiat–Shamir order

Generic stream, `Blake3Transcript<Fr>::new(b"jolt-wrapper-v1")`:

1. Caller-supplied canonical verifier-key digest; public statement scalars; packed commitments.
2. Stage A input claim; batch coefficient; compressed rounds; verifier-derived output claim.
3. Stage B input equal to stage A's derived output; batch coefficient; compressed rounds; output
   recomputed from `T(s_i)` and the public tensor terms.
4. The same `T(s_i)` values; one `rho` per value; stage C input claim; batch coefficient;
   compressed rounds; final output claim.
5. HyperKZG: fold commitments; fold challenge; KZG batch-opening transcript.

Spartan uses the same stage encoding with this prefix/order:

1. Caller-supplied R1CS/profile digest; public inputs; witness commitment; `tau`.
2. Outer input `0`; batch coefficient; outer rounds; derived outer final claim; `Az(rx), Bz(rx),
   Cz(rx)`.
3. Matrix weights `ra, rb, rc`; inner input claim after subtracting columns `[0, 1 + |x|)`; batch
   coefficient; inner rounds; derived inner final claim; `W(ry)`.
4. HyperKZG opening of `W` at `ry`.

## Bytes

Ignoring serde length prefixes, with `C` packed commitments, `S` transmitted compressed-round
scalars, `A` stage-member output claims, `R` reduced claims, and opening dimension `ell`:

```text
payload = 32 * (C + S + A + R + 4 * ell)
                       opening: (ell - 1) fold G1 + 1 witness G1 + 3*ell Fr
```

`WrapperProof::bincode_bytes()` adds every standard-bincode vector-length prefix and the
`serialize_bytes` length prefix on each G1.

Spartan worked example, `m = n = 2^12`, `|x| = 50`:

```text
C = 1
S = 12*3 + 12*2 = 60
R = 4                   # Az, Bz, Cz, W(ry)
A = 2                   # outer and inner stage outputs
ell = 12
payload = 32 * (1 + 60 + 2 + 4 + 48) = 3,680 B
bincode = 3,680 + 49 = 3,729 B
```

The ignored `2^17`, `k = 8` gate constructs this crate's 163 bit, 54 u16, 19 helper-Fr, and one
witness-Fr column shape, then executes `commit_packed`, `prove_stream`, and `verify_stream`.

Release result on the 16 GiB M4 mini:

```text
setup 6.600 s; commit 1.013 s; prove after commit 4.948 s; verify 0.002 s
payload 10,304 B; bincode 10,445 B; max RSS 3,867,017,216 B
```

## Verification

```text
cargo nextest run -p jolt-wrapper --cargo-quiet
cargo clippy -p jolt-wrapper --all-targets -q --message-format=short -- -D warnings
cargo fmt -q
cargo nextest run --release -p jolt-wrapper n3_g_shape_timing --cargo-quiet \
  --run-ignored ignored-only
```

Isolated scratch-worktree results: wrapper clippy clean; wrapper normal suite 4 passed / 1 ignored;
R1CS range-evaluator test passed; release timing gate passed; diff-scoped style checker clean.

Review fixes: stage outputs/claim points are verifier-derived and the exact `T(s_i)` values feed C;
stage claims support heterogeneous batches; final shape comes from the statement; key/public values
precede commitments; group padding has one owner; a head/tail aligned test exercises stage scaling;
the timing gate executes production code; public-column contributions take one pass per matrix;
tamper cases cover shapes, degrees, every stage, claims, commitment order/count, openings, keys, and
R1CS witnesses. `StageWindow`, per-round inversions, duplicate output-claim absorption, avoidable
shape-G clones, redundant Spartan extraction, string errors, and per-point table passes were removed.
