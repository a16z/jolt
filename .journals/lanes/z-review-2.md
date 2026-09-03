# Zeromorph review 2 — `686cacf94`

## Findings

1. **MINOR — `crates/jolt-zeromorph/src/lib.rs:596`, `crates/jolt-zeromorph/src/lib.rs:641` — the final identity weights have two owners.**

   `combined_identity` and `combined_identity_commitment` independently implement the same
   `-rho^j (y^k x^(N-2^k) + z A_k(x))` quotient coefficient, plus the same lifted,
   polynomial-commitment, and constant coefficients. Round-trip tests catch an honest-path drift,
   but this duplication violates the repo rule that each protocol formula has one owner.

   **Fix:** make `identity_scalars` return the complete `y`/`z`-weighted coefficient row and use it
   in both accumulators; apply `rho^j` at each call site. This keeps the prover loop allocation shape
   unchanged.

## Prior findings

1. **FIXED — statement binding.**

   - Prover absorbs commitment, arity, point count, every point coordinate, and every claim before
     quotient construction or challenge sampling: `crates/jolt-zeromorph/src/lib.rs:227-233` and
     `crates/jolt-zeromorph/src/lib.rs:455-477`.
   - Verifier uses the same owner and order: `crates/jolt-zeromorph/src/lib.rs:318-330`.
   - Quotient commitments precede `y`; the `y`-dependent lifted commitments precede `x`; `z` and
     `rho` follow; the challenge-dependent witness is the last prover message:
     `crates/jolt-zeromorph/src/lib.rs:250-277`.
   - The trait path returns the commitment as its opening hint and rejects a missing hint:
     `crates/jolt-zeromorph/src/lib.rs:724-756`.
   - `combine_hints` calls the same `combine` implementation with the same scalar slice as the
     verifier commitment RLC: `crates/jolt-zeromorph/src/lib.rs:778-792`. The outer homomorphic
     batch absorbs the parts before deriving those scalars, then both PCS sides absorb the resulting
     combined commitment and combined evaluation: `crates/jolt-openings/src/schemes.rs:513-525`
     and `crates/jolt-openings/src/schemes.rs:540-552`.

2. **FIXED — runtime G2 scalar multiplication.**

   `verify_multi` computes only `x * pi` in G1 and pairs against the fixed verifier-key values
   `[beta]_2` and `[1]_2`: `crates/jolt-zeromorph/src/lib.rs:343-347`.

3. **FIXED — decisive and boundary tests.**

   - Fixed-challenge mutations cover a quotient commitment, lifted commitment, and final witness:
     `crates/jolt-zeromorph/tests/zeromorph.rs:171-218`.
   - The adaptive regression uses identity quotient/lift commitments, derives `x` from the exact
     post-statement-omission schedule, sets `v = f_hat(x) / Phi_n(x)`, constructs `[X Q(X)]_1`,
     proves the forged claim is false, and checks rejection:
     `crates/jolt-zeromorph/tests/zeromorph.rs:220-279`.
   - Arity one, zero/constant tables, exact SRS length, high-to-low Boolean point order, and the
     2,048-byte three-point shape are pinned at
     `crates/jolt-zeromorph/tests/zeromorph.rs:137-147`,
     `crates/jolt-zeromorph/tests/zeromorph.rs:281-309`, and
     `crates/jolt-zeromorph/tests/zeromorph.rs:354-370`.

## Pairing and degree-check derivation

For `H(X) = (X - x) Q(X)`:

```text
C_H = [(beta - x) Q(beta)]_1
pi  = [beta Q(beta)]_1

C_H - pi = [-x Q(beta)]_1
x pi     = [x beta Q(beta)]_1

e(C_H - pi, [beta]_2) * e(x pi, [1]_2)
= e(g1, g2)^(-x beta Q(beta) + x beta Q(beta))
= 1.
```

The signs and exponents at `crates/jolt-zeromorph/src/lib.rs:343-347` match this identity.

Kohrita–Towa §5.3 and §6 use an SRS with exponents `0..Nmax-1`; for an `N = 2^ell`
opening with `Nmax = N`, the terminal degree bound is `N-1` and the shift is
`Nmax - (N-1) = 1`. The implementation publishes exactly `N` Zeromorph G1 powers
(`crates/jolt-zeromorph/src/lib.rs:159-177`), commits `pi = [X Q(X)]_1`
(`crates/jolt-zeromorph/src/lib.rs:274-275`), and can touch only SRS indices `1..N-1`
because `Q` has at most `N-1` coefficients (`crates/jolt-zeromorph/src/lib.rs:683-692`).
No path consumes `[beta^N]_1`.

The individual `deg q_k < 2^k` checks remain present. Each `q_k` commitment is fixed before `y`;
the prover then commits to `sum_k y^k X^(N-2^k) q_k` in the upper SRS half
(`crates/jolt-zeromorph/src/lib.rs:235-260`, `crates/jolt-zeromorph/src/lib.rs:540-553`).
After random `x`, the final shifted opening proves the lifted identity at `x`. A high-degree
`q_k` makes the intended lift exceed the `0..N-1` SRS except with the paper's `y`/`x` failure
probability.

The shared HyperKZG SRS correction is consistent: `N` G1 powers and degree-five shift
`beta^(N-6)` at `crates/jolt-hyperkzg/src/scheme.rs:65-86`; its degree-five commitment uses
indices `N-6..N-1` at `crates/jolt-hyperkzg/src/multi_open.rs:54-71`.

## Three-point and verifier-path audit

- All three `(u_j, v_j)` statements are absorbed before any quotient message. All three quotient
  families precede shared `y`; all three lifted commitments precede shared `x`, `z`, and `rho`.
  The single witness follows `rho`: `crates/jolt-zeromorph/src/lib.rs:227-277`.
- At fixed `y`, `x`, and `z`, every per-point `H_j` is fixed before `rho`; the verifier checks
  `sum_j rho^j H_j`. A nonzero false component cancels for at most `t-1` values of `rho`.
- Grep of both crates' verify paths found no G2 scalar multiplication, `mul_bigint`, or
  `G2Projective` arithmetic. Zeromorph uses fixed `[beta]_2`, `[1]_2`; HyperKZG uses fixed
  low G2 powers and fixed `beta^(N-6)`.
- Journal protocol, proof-shape, SRS, and EVM-operation claims match the code and
  [Kohrita–Towa §5–6](https://eprint.iacr.org/2023/917.pdf). The timing table was not rerun.

## Style and checks

- Production source: 859 lines; no `#[allow]`; no added nominal-path violation found.
- `cargo clippy -p jolt-zeromorph -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings`: passed.
- `cargo nextest run -p jolt-zeromorph -p jolt-hyperkzg --cargo-quiet`: 33 passed, 0 skipped.

VERDICT: 0 blockers, 0 majors, 1 minor
