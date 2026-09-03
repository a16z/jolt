# W4-S — shared stream and Spartan + HyperKZG

Date: 2026-09-02. Branch: `wrap/spartan-hyperkzg`.

## Result

- `stream.rs`: low-variable column packing, type-directed commitment kernels, staged `prove_batch`,
  tensor column batching, evaluation-claim reduction, commitment/evaluation RLC, and one HyperKZG
  opening.
- `stream/types.rs`: proof/statement types, errors, and exact payload/bincode size accounting.
- `spartan.rs`: plain Spartan outer/inner sumchecks over `ConstraintMatrices<Fr>` with public columns
  `[0, 1 + |x|)` and committed witness columns `[1 + |x|, num_vars)`.
- `tests/stream_synthetic.rs`: `2^12 × (40 bit + 20 u16)`, degree-5 row relation, stages A/B/C,
  full opening verification, and column/round/claim tampers.
- `tests/spartan_core.rs`: satisfiable `m = n = 2^12`, `|x| = 50`, public-column terms, and
  public/round/claim/opening tampers.

## Packed layout

For group `g`, slot `j < k`, and row `i`:

```text
P_g[i * k + j] = column[g * k + j][i]
opening point   = (r_row, s_slot)
                  high vars  low log2(k) vars
```

Missing slots in the last group are zero. Homogeneous bit groups use
`g1_bit_columns_msm`; homogeneous `u16` groups use `Bn254::g1_affine_msm_small`; mixed or `Fr`
groups use the full-field HyperKZG commitment path. All paths commit the same dense layout.

## Sumcheck stages

1. **A — row space.** Any `ProveRounds<Fr>` batch; shorter head-aligned members are scaled by
   `2^(max_rounds - offset - member_rounds)` inside the stage adapter.
2. **B — column space.** `ColumnBatching` proves sparse tensor forms
   `sum Q[j_1,...,j_D] * product T(j_d)` in `D * log2(columns)` degree-2 rounds. Its input is the
   stage-A final relation value; its final values are `T(s_1), ..., T(s_D)`.
3. **C — claim reduction.** `ClaimReduction` proves
   `sum_g q_g(x) P_g(x)`, where `q_g = sum_j rho_j eq(p_j, x)`, in degree-2 rounds. At the final
   point it yields weights `q_g(r)` for the commitment/evaluation RLC and one HyperKZG opening.

`StageProof` stores only compressed round polynomials. Input/output claims are statement-derived
and absorbed in Fiat–Shamir, but are not transmitted.

## Fiat–Shamir order

Generic stream, `Blake3Transcript<Fr>::new(b"jolt-wrapper-v1")`:

1. Packed commitments, declaration order.
2. For each prior stage: input claims; one batch coefficient per member; compressed round
   polynomials; derived member output claims.
3. Reduction claim values; one `rho` per claim.
4. Stage C input claim; its batch coefficient; compressed rounds; final output claim.
5. HyperKZG: fold commitments; fold challenge; KZG batch-opening transcript.

Spartan uses the same stage encoding with this prefix/order:

1. Witness commitment; public inputs; `tau`.
2. Outer input `0`; batch coefficient; outer rounds; derived outer final claim; `Az(rx), Bz(rx),
   Cz(rx)`.
3. Matrix weights `ra, rb, rc`; inner input claim after subtracting columns `[0, 1 + |x|)`; batch
   coefficient; inner rounds; derived inner final claim; `W(ry)`.
4. HyperKZG opening of `W` at `ry`.

## Bytes

Ignoring serde length prefixes, with `C` packed commitments, `S` transmitted compressed-round
scalars, `R` reduced claims, and opening dimension `ell`:

```text
payload = 32 * (C + S + R + 4 * ell)
                       opening: (ell - 1) fold G1 + 1 witness G1 + 3*ell Fr
```

`WrapperProof::bincode_bytes()` adds every standard-bincode vector-length prefix and the
`serialize_bytes` length prefix on each G1.

Spartan worked example, `m = n = 2^12`, `|x| = 50`:

```text
C = 1
S = 12*3 + 12*2 = 60
R = 4                   # Az, Bz, Cz, W(ry)
ell = 12
payload = 32 * (1 + 60 + 4 + 48) = 3,616 B
bincode = 3,616 + 46 = 3,662 B
```

N3 G timing shape, `2^17`, `s = 3`, `k = 8`:

```text
commitments 1,024 + rounds 2,720 + claims 5,632 + opening 2,560 = 11,936 B
+ public IO estimate 1,024 = 12,960 B
```

## Verification

```text
cargo nextest run -p jolt-wrapper --cargo-quiet
cargo clippy -p jolt-wrapper --all-targets -q --message-format=short -- -D warnings
cargo fmt -q
cargo nextest run -p jolt-wrapper n3_g_shape_timing --cargo-quiet --run-ignored ignored-only
```

Results: 2 passed, 1 ignored in the normal suite; ignored timing gate passed. Timing gate measured
setup 7.1 s, prover 4.765 s, verifier 0.007 s, and maximum RSS 2,678,161,408 B. Prover split:
commitments 2.087 s, instance build 0.248 s, stream 0.958 s, reduction/combine 0.147 s, opening
1.326 s.
