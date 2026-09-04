# PERF-5 lane 5b — four-ary HyperKZG folds

Date: 2026-09-03. Base: `dbe2a2f9e`. Tests and measurements deferred for the
lane-4 idle-machine window.

## Design

- Each committed fold consumes the last two variables and combines each four-coefficient
  chunk in multilinear order. The shared fold-count owner is
  `HyperKZGScheme::fold_level_count(num_vars) = (num_vars - 1) / 2`.
- At `ell = 23`, eleven four-ary folds produce polynomials of lengths
  `2^21, 2^19, ..., 2`; the last width-two polynomial supplies the claimed value through
  one uncommitted binary fold at `x_0`. Committed points: 8,388,606 -> 2,796,202.
- The BN254 Fr modulus is 1 modulo 4. The configured canonical field element is checked in
  production to satisfy `i^2 = -1` and `i != +/-1`.
- The verifier opens each polynomial at `r, ir, -r, -ir, r^4`. The first four values recover
  the four residues through one 4-by-4 DFT; the fifth row binds the next fold commitment.
  The divisor is `(X^4 - r^4)(X - r^4)`. Its only nonzero coefficients are at exponents
  0, 1, 4, 5; the pairing uses exactly those four G2 powers. The VK adds G1 powers
  beta^3 and beta^4, and replaces G2 beta^2/beta^3 with beta^4/beta^5.
- The shifted-commitment check still bounds the shared round polynomials. HyperKZG fold
  commitments retain the pre-existing SRS-wide degree bound; this change adds no per-level
  degree-bound proof.

## Expected deltas before the gate

| item (`ell = 23`, `k = 32`) | binary | four-ary | delta |
|---|---:|---:|---:|
| intermediate G1 commitments | 22 | 11 | -11 / -352 B |
| transmitted Fr evaluations | 47 | 49 | +2 / +64 B |
| HyperKZG opening | 2,240 B | 1,952 B | -288 B |
| wrapper payload | 7,392 B | 7,104 B | -288 B |
| bincode proof | 7,529 B | 7,232 B | -297 B |
| statement | 352 B | 352 B | 0 |
| pairing pairs | 8 | 8 | 0 |
| ecMul / ecAdd | 226 / 225 | 216 / 216 | -10 / -9 |
| Fr mul / inversions | 123,229 / 10 | 123,121 / 8 | -108 / -2 |
| Keccak calls | 848 | 839 | -9 |
| N4 gas | 4,890,645 | 4,800,225 | -90,420 |

The dense five-point model's two extra pairing pairs multiply identity G1 elements,
so they are omitted. The two equal divisor coefficients also share one G1 scalar
multiplication. The gate derives pairing gas from `VerifierCost::pairing_pairs`.
At `ell = 22`, ten commitments and 45 evaluations make the opening 1,792 B;
the `k = 16` wrapper model is
7,392 B payload / 7,534 B bincode.

`FoldPoints` owns the five points and inverse-DFT scales; one observed inversion
produces all four scales. The sparse quotient recurrence and its parallel-block
correction each use three Fr multiplications per coefficient, with no dense
five-by-five matvec in the coefficient loop. Counts above are source-derived,
not measured. The real gate pins both packed shapes, every evaluation row, all
fold commitments, and the complete k=32 operation vector.

## Compile gate

| command | result |
|---|---|
| `cargo check -p jolt-hyperkzg -p jolt-wrapper --all-targets --features prover-fixtures` | pass |
| matching clippy with `-D warnings` | pass |

No tests, nextest, benchmark, or real fixture gate ran in this phase.
The retained tests cover multilinear fold ordering, inverse-DFT residues, quintic
division/interpolation (including a short parallel-block tail), and exact SRS powers.
Run these and the existing full tamper gate only after the orchestrator releases
the lane-4 measurement window.
