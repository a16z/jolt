# W4-S review #3

Scope: committed tree `37134dbb5`, reviewed in a detached scratch worktree against
`19872523d`; staged-stream sources/tests and the `jolt-hyperkzg` variable-point opening/setup
changes.

## Findings

None.

## Stage A: committed round polynomials

### Fiat-Shamir order

The prover and verifier have the same per-round trace:

```text
claim_i
  -> C_i
  -> r_i
  -> s_i(0)
  -> claim_(i+1) = s_i(r_i)
```

`stream.rs:456-497` and `:543-560` append every `C_i` before drawing `r_i`, append both disclosed
values after `r_i`, finish all 17 rounds, and only then enter the variable-point batch. Thus `rho`,
`gamma`, and `z` are drawn after every round commitment and disclosed evaluation is fixed.
`open_variable_batch`/`verify_variable_batch` then use `rho -> C_shift -> gamma -> W -> z -> W'`
at `multi_open.rs:48-125` and `:151-194`.

### BDFG20 section 4

The implementation matches [BDFG20 section 4](https://eprint.iacr.org/2020/081):

```text
r_i(X)    = degree-2 interpolation through (0, s_i(0)), (1, s_i(1)), (r_i, s_i(r_i))
f(X)      = sum_i gamma^i Z_(T\S_i)(X) (s_i(X) - r_i(X))
W         = commit(f / Z_T)
L(X)      = sum_i gamma^i Z_(T\S_i)(z) (s_i(X) - r_i(z)) - Z_T(z) (f / Z_T)(X)
W'        = commit(L / (X-z))
```

The verifier builds the same interpolation remainders and checks
`e(F, [1]_2) = e(W', [beta-z]_2)`, with
`F = sum_i gamma^i Z_(T\S_i)(z)(C_i-[r_i(z)]_1)-Z_T(z)W`. Signs and complement sets agree with
the paper. Repeated points inside one set are rejected; cross-round repeated challenges are
handled by the union.

### Degree bound

Let `L = setup.g1_powers.len()` and `D = 5`. `multi_open.rs:54-69` commits the six coefficients of
the round-polynomial RLC against powers `L-1-D .. L-1`; `scheme.rs:65-86` places
`[beta^(L-1-D)]_2` in the verifier setup. The pairing at `multi_open.rs:153-160` is therefore

```text
e(commit(X^(L-1-D) sum_i rho^i s_i), [1]_2)
  = e(sum_i rho^i C_i, [beta^(L-1-D)]_2).
```

If any fixed `s_i` has a coefficient above degree five, choose the highest offending degree. Its
RLC coefficient is a nonzero polynomial in `rho` of degree at most `rounds-1`, so it survives
except with probability at most `(rounds-1)/|Fr|`; the shifted polynomial would then require a G1
power beyond index `L-1`. The index is exact: `setup_from_secret` uses SRS degree `L-1`, hence its
literal `srs_degree - 5` is `L-1-D`.

Degree below five is accepted by zero-padding the shifted RLC. A zero round polynomial is also
valid exactly when its running claim is zero; all three commitments can be the identity and the
exact divisions produce zero quotients.

### Tamper reach

- `stream_synthetic.rs:356-363` changes `C_shift`; verification reaches and fails the degree
  pairing before the later BDFG challenge.
- `stream_synthetic.rs:348-354` changes `W'`; verification reaches and fails the final BDFG
  pairing, with no later challenge acting as a checksum.
- `stream_synthetic.rs:332-346` changes a round commitment and a disclosed next claim. Both alter
  the post-round `rho` and reject before acceptance.

## Stage B and elided claims

`protocol.rs:57-86` fixes all five factor claims before `prove_stage`; `stream.rs:384-398` and
`:635-650` absorb them before drawing the five member coefficients. All five `ColumnReduction`s
start with the same column-evaluation polynomial and receive the same eight challenges.

At the final point `s`, `protocol.rs:233-251` reconstructs every member output as
`eq(column_i,s) * T(s)` using statement-derived column indices and the one reduced `T(s)`. The
batched final-claim equality checks their verifier-computed RLC. Stage C then binds that `T(s)` to
the packed commitments at the derived `(row_point, s)` through the claim reduction and HyperKZG
opening (`protocol.rs:252-320`). Swapping two factor claims preserves the tensor product but is
rejected by the Stage-B RLC path (`stream_synthetic.rs:277-279`).

The three old singleton outputs are absent from the wire:

- A: recovered from the single member's final claim/coefficient (`protocol.rs:214`, `:434-443`);
  committed A already absorbed the same value as its last `claim_(i+1)`.
- B: reconstructed by `ColumnReduction::expected_final` for every factor.
- C: recovered from its final claim/coefficient, then checked as the HyperKZG evaluation
  (`protocol.rs:301-319`).

The only `stage_claims` vector contains the five A-to-B factor inputs; the only `reduced_claims`
entry is `T(s)`.

## Byte recount

| item | `k=8` | `k=16` |
|---|---:|---:|
| packed commitments | 960 | 480 |
| A: `17 * (1 G1 + 2 Fr) + 3 G1` | 1,728 | 1,728 |
| B: `8 * 2 Fr` | 512 | 512 |
| C: `ell * 2 Fr` | 1,280 (`ell=20`) | 1,344 (`ell=21`) |
| five factor claims + one reduced claim | 192 | 192 |
| HyperKZG: `ell G1 + 3*ell Fr` | 2,560 | 2,688 |
| **payload** | **7,232** | **6,944** |

This follows `WrapperProof::payload_bytes` field-for-field. The timing gate's prior serialized
measurements, 7,347 and 7,046 bytes, also matched `bincode_bytes()`.

## Prover-time assessment

The reported `k=8` commit time doubled from 1.01 s to 2.06 s although `commit_packed` is unchanged
by `37134dbb5`. Setup generation also roughly doubled. The new Stage-A curve work is bounded by 17
six-term round MSMs plus MSMs of sizes 6, at most 3, and at most 5 for `C_shift`, `W`, and `W'`
(at most 116 scalar terms total); no new path commits a round polynomial against the full SRS.
BDFG polynomial arithmetic is over at most 19 distinct points.

Conclusion: the 8.22 s versus 4.95 s sample is host contention, not evidence of a protocol
regression. A fresh timing run was skipped because another lane continuously occupied about eight
cores (`~778%` CPU; load average about 11), which would repeat the same measurement error.

## Complexity and checks

- Scoped source files are at most 946 lines; no `#[allow]`, nominal-path violation, dead mode, or
  duplicated protocol formula found.
- BDFG formulas live in `multi_open.rs`; packing geometry, Stage-B final values, and payload sizing
  each retain one owner.
- `cargo clippy -p jolt-wrapper -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings`:
  passed.
- `cargo nextest run -p jolt-wrapper --cargo-quiet`: 10 passed, 1 ignored.

VERDICT: 0 blockers, 0 majors, 0 minors
