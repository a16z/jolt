# PERF-5 lane 6 — stream tail

Date: 2026-09-03. Base: `dbe2a2f9e`; rebased code: `b8cfcd247`. Fixture:
`fibonacci_2_18_blake3.bin`, `k = 32`, `N = 2^23`. Prior landing:
**26.072 s**; the paired canonical rerun measured **26.025 s**.

## Result

Unit and real-fixture proofs are byte-identical to `dbe2a2f9e`. The wrapper
suite passed 64/64 with `NEXTEST_TEST_THREADS=4` and `RAYON_NUM_THREADS=1`.
The idle gate measured **26.025→25.200 s** online (**−0.825 s**).
Column evaluation falls **371.557→1.770 ms**, packed RLC falls
**659.126→132.695 ms**, and member construction falls **806→792 ms**.
No `jolt-hyperkzg` files changed.

## Changes

1. Stage-A members publish the multilinears they already reduced to one value:
   T1 committed/wired columns, each CopyLink's fixed and helper columns, every
   T2 claimed column, and the Spartan witness carry. The key owns the mapping
   between those values and physical packed slots.
2. The only live columns without a member binding are T1's six verifier-key
   columns. They are evaluated with one row-equality table and typed sparse
   scans. The key enumerates every physical padding slot as zero. The term
   stage therefore receives a complete column vector without scanning the 26
   packed group polynomials.
3. Packed RLC now runs by parallel row blocks. Type dispatch occurs once per
   group and block; bit/u16/u32 sources stay typed; zero values, zero weights,
   and the key's padding slots are skipped. No group becomes a full-field
   polynomial for this pass.
4. T2 digit-link setup reads the row-major typed matrix once into its three
   owned working columns. This removes nine temporary full-field column vectors
   and the digit clone.

## Column ownership

At `k = 32`, 26 physical groups contain 832 slots: 598 live columns and
234 padding slots. The stage-B domain pads further to 1,024 slots.

| source | live columns | owner at the stage-A point |
|---|---:|---|
| T1 bits / wired bits / words | 307 | T1 row member |
| T1 VK | 6 | typed VK evaluation |
| CopyLink fixed / helpers | 120 / 20 | ten sparse CopyLink members |
| T2 claimed columns, including VK | 144 | T2 row member |
| Spartan witness | 1 | carry member |

## Measurements

All real runs used 10 Rayon threads. Each binary was compiled with
`cargo nextest run --no-run` before the mutex. Saved nextest binary/cargo
metadata bypassed compilation inside the mutex. The mutex was released after
each run; acquisition retried every 60 s with one-minute load below 4.

| phase | canonical idle | lane 6 idle | delta |
|---|---:|---:|---:|
| column evaluations at `r_A` | 371.557 ms | 1.770 ms | −369.787 ms |
| packed RLC | 659.126 ms | 132.695 ms | −526.431 ms |
| all member constructors | 806 ms | 792 ms | −14 ms |
| T2 member setup | 33 ms | 19 ms | −14 ms |
| unchanged T2 phase 2a | 6,254 ms | 6,256 ms | +2 ms |
| unchanged T2 stage A | 2,503 ms | 2,491 ms | −12 ms |
| proof stages/opening | 15,227 ms | 14,473 ms | −754 ms |
| **honest online wall** | **26,025 ms** | **25,200 ms** | **−825 ms** |
| online phase sum | 26,021 ms | 25,195 ms | −826 ms |
| process CPU | 218.500 s | 211.700 s | −6.800 s |
| CPU / wall | 8.396 | 8.401 | +0.005 |
| verifier, outside online clock | 26 ms | 25 ms | −1 ms |

The successful candidate held the mutex at **22:03:45–22:04:26 ET**.
Command-start load was **2.41** for the baseline and **1.90** for lane 6.
Online-start/end loads were **3.88/6.19** and **3.22/5.73**, respectively.
No compiler was present at candidate entry or observed during its one-second
process sampling. The aggregate member bucket includes T1's precomputed first
round; its 14 ms saving is entirely the T2 constructor.

Two earlier candidate attempts measured 30.281 s and 33.122 s and are
excluded: external compilers started during their mutex-held windows.
The second entered at 21:49:59 with no compiler present, but two nightly
compiler processes started around 21:50:20 before it finished at 21:50:48.
The successful retry used the same prebuilt binary; no code tuning followed
these contaminated measurements.

The measured tail saving is **0.896 s**, plus **14 ms** in T2 setup.
The initial 1.4–1.9 s estimate used the planning run's 34-group timings;
the current 26-group baseline is cheaper and replaces that estimate.

If lane 5b exposes a first-fold accumulation hook, the row-block RLC can feed
that hook without an intermediate pass. This branch leaves RLC materialized
once because concurrent `jolt-hyperkzg` edits are out of scope.

## Byte identity

The synthetic `verifier_cost_includes_statement_derivation` fixture produced
identical 4,734-byte bincode proofs:
`380d6cee6fcc5613f419650ea92157a15758d8b74563dab4ce4c699351aaf6d9`.
The complete real fixture produced identical 7,529-byte bincode proofs:
`1defa055d1a9445bc58814df939611da88c011003c787a4905b7d24739365aae`.
Both comparisons used `cmp` against a scratch checkout at `dbe2a2f9e`.
Temporary proof-output hooks and tail timers were removed before handoff.
Every tamper rejected; payload/bincode/statement remain **7,392/7,529/352 B**.
Verifier counts remain **226 ecMul / 225 ecAdd / 8 pairing pairs / 123,229
Fr mul / 10 inversions / 848 Keccak**, or **4,890,645 gas**.

## Gates

| command | result |
|---|---|
| `cargo check -p jolt-wrapper --all-targets --features prover-fixtures` | pass |
| wrapper clippy, all targets, feature-enabled, warnings denied | pass |
| `cargo fmt -q --message-format=short` | pass |
| `cargo nextest run -p jolt-wrapper --cargo-quiet` | 64/64 pass |
| feature-enabled real fixture under the mutex | baseline and three candidates pass; all tampers reject |
| feature-enabled all-target check after rebase | pass |
| wrapper suite after rebase | 64/64 pass |

The rebase onto `f4c2dc3d4` contained only upstream journal changes;
production sources match the measured prebuilt binary.
