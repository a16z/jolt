# PERF-5 lane 5b — four-ary HyperKZG folds

Date: 2026-09-03. Original base: `dbe2a2f9e`; measured against integrated
`a244203fb` (lanes 4 and 6 included). Fixture: `fibonacci_2_18_blake3.bin`.
Ten Rayon threads for timed gates. Default packing remains `k=32`.

## Result

The matched `k=32` idle comparison cuts online wall **22.636 -> 19.671 s**
(-2.965 s, 13.1%), fold commitments **4.071859 -> 1.422473 s**, and total
HyperKZG opening **8.553994 -> 5.693959 s**. Quotient MSM:
**3.811224 -> 3.713000 s**. Against the earlier integrated 22.410 s gate,
the observed online reduction is 2.739 s.

Proof payload decreases 288 B and modeled N4 gas decreases 90,420 at `k=32`.
The `k=16` idle gate measures **16.978 s** online, **0.751742 s** folds,
**1.963131 s** quotient, and **3.042152 s** total opening. Relative to
four-ary `k=32`, this saves 2.693 s for 288 B payload and 143,924 modeled gas.

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

## Measured bytes and verifier operations

| item | binary k=32 | four-ary k=32 | four-ary k=16 |
|---|---:|---:|---:|
| opening variables | 23 | 23 | 22 |
| intermediate G1 commitments | 22 | 11 | 10 |
| transmitted Fr evaluations | 47 | 49 | 45 |
| HyperKZG opening | 2,240 B | 1,952 B | 1,792 B |
| wrapper payload | 7,392 B | 7,104 B | 7,392 B |
| bincode proof | 7,529 B | 7,232 B | 7,533 B |
| statement | 352 B | 352 B | 352 B |
| pairing pairs | 8 | 8 | 8 |
| ecMul / ecAdd | 226 / 225 | 216 / 216 | 233 / 233 |
| Fr mul / inversions | 123,229 / 10 | 123,121 / 8 | 123,144 / 8 |
| Keccak calls | 848 | 839 | 852 |
| N4 gas, modeled from observed operations | 4,890,645 | 4,800,225 | 4,944,149 |

The dense five-point model's two extra pairing pairs multiply identity G1 elements,
so they are omitted. The two equal divisor coefficients also share one G1 scalar
multiplication. The gate derives pairing gas from `VerifierCost::pairing_pairs`.
At `ell = 22`, ten folds leave four coefficients for the terminal two-variable
check. The measured k=16 bincode size corrects the compile-phase estimate by
one byte; its payload and operation counts match the recorded layout.

`FoldPoints` owns the five points and inverse-DFT scales; one observed inversion
produces all four scales. The sparse quotient recurrence and its parallel-block
correction each use three Fr multiplications per coefficient, with no dense
five-by-five matvec in the coefficient loop. The real gate pins both packed
shapes, every evaluation row, all
fold commitments, and the complete k=32 operation vector.

## Idle phase table

| phase | binary k=32 | four-ary k=32 | four-ary k=16 |
|---|---:|---:|---:|
| SRS setup, offline | 8,045 ms | 7,704 ms | 3,805 ms |
| key/profile, offline | 196 ms | 207 ms | 166 ms |
| key commitments, offline | 395 ms | 418 ms | 305 ms |
| preparation | 437 ms | 451 ms | 443 ms |
| T1/R adaptation | 70 ms | 75 ms | 65 ms |
| T2 adaptation | 654 ms | 647 ms | 652 ms |
| phase 1a commitment | 794 ms | 887 ms | 734 ms |
| phase 1b commitment | 887 ms | 821 ms | 769 ms |
| phase 2a commitment | 5,332 ms | 5,425 ms | 5,576 ms |
| phase 2b commitment | 98 ms | 91 ms | 66 ms |
| CopyLink helpers | 36 ms | 34 ms | 35 ms |
| phase 2c commitment | 331 ms | 341 ms | 325 ms |
| T2 finish | 263 ms | 207 ms | 242 ms |
| all member constructors | 781 ms | 780 ms | 792 ms |
| T2 constructor, included above | 16 ms | 15 ms | 17 ms |
| proof stages/opening | 12,948 ms | 9,907 ms | 7,273 ms |
| T2 stage A, included above | 2,597 ms | 2,556 ms | 2,459 ms |
| fold commitments | 4.071859 s | 1.422473 s | 0.751742 s |
| quotient MSM | 3.811224 s | 3.713000 s | 1.963131 s |
| total HyperKZG opening | 8.553994 s | 5.693959 s | 3.042152 s |
| honest online wall / phase sum | 22.636 / 22.631 s | 19.671 / 19.666 s | 16.978 / 16.972 s |
| process CPU / CPU-to-wall | 186.900 s / 8.257 | 160.050 s / 8.136 | 136.610 s / 8.046 |
| verifier, outside online clock | 28 ms | 27 ms | 24 ms |

All three accepted windows held `/tmp/wrapper-gate.lock`, started below
one-minute load 4 with no compiler or other test process, and sampled processes
once per second. No competing job was observed during an accepted window.

| window (ET) | command-start load, 1/5/15 min | online-start/end, 1 min | command-end load, 1/5/15 min |
|---|---|---|---|
| baseline 22:36:49–22:37:26 | 3.06 / 4.90 / 5.74 | 4.44 / 6.67 | 6.21 / 5.53 / 5.93 |
| k=32 22:43:10–22:43:45 | 3.49 / 4.98 / 5.57 | 4.72 / 5.95 | 5.80 / 5.44 / 5.72 |
| k=16 22:52:00–22:52:27 | 3.71 / 4.85 / 5.29 | 4.38 / 6.00 | 6.24 / 5.35 / 5.46 |

The online clock starts after SRS/key preparation, which raises the
one-minute load before that clock begins. Timing probes were removed before
handoff. Logs: `/tmp/perf5-lane5b.mAGQf1/{baseline-idle,candidate32-rebuilt-idle,candidate16-clean-idle}.log`.

Excluded attempts: one k=32 run reused the baseline executable in the shared
target directory; the two changed crates were cleaned and rebuilt before the
accepted candidate. One k=16 run stopped at the one-byte size ratchet; the next
passed correctness but overlapped 70 external Cargo/compiler processes and
measured 23.697 s online. Neither supplies a timing row above. A daemon restart
also interrupted an idle wait, before a test started.

Prebuild with `cargo nextest run -p jolt-wrapper --features prover-fixtures
--cargo-quiet real_wrapper --no-run`. The locked executions use saved nextest
binary/Cargo metadata, so no build occurs inside the mutex; this nextest version
rejects `--cargo-quiet` together with `--binaries-metadata`, so metadata runs omit
that flag. Do not share a target directory between comparison worktrees without
invalidating the changed crates' artifacts.

## Gates

| command | result |
|---|---|
| `cargo check -p jolt-hyperkzg -p jolt-wrapper --all-targets --features prover-fixtures` | pass |
| matching clippy with `-D warnings` | pass |
| wrapper + HyperKZG nextest, six workers | 94/94 pass |
| feature-enabled real k=32 | pass; every tamper rejects |
| feature-enabled real k=16 | pass; every tamper rejects |

Rebased onto `d7601a2fb`, a journal-only advance beyond `a244203fb`.
After removing the timing probes, the final six-worker nextest run passed
94/94 in 200.378 s; its log is
`/tmp/perf5-lane5b.mAGQf1/final-unit-complete.log`.

The retained tests cover multilinear fold ordering, inverse-DFT residues, quintic
division/interpolation (including a short parallel-block tail), and exact SRS powers.
An inconsistent middle fold is rejected for both odd and even dimensions despite
independently valid KZG openings at all five points. Corrupting each newly needed
G1/G2 VK power rejects; the wrong-claim test exercises the odd terminal binary step.
The synthetic statement-cost ratchet changes 9,900 -> 9,815 Fr multiplications;
its independently counted 703-operation statement contribution is unchanged.
