# PERF-5 lane 2 — sparse CopyLink and column packing

Date: 2026-09-03. Base after lane 1: `1996e1dc2`. Code commit after final
rebase: `320c65abe`. Machine: Mac mini M4, 10 Rayon threads. Fixture:
`fibonacci_2_18_blake3.bin`, `k = 32`, `N = 2^23`.

## Result

- Honest online wall: **37.523 s -> 29.802 s**.
- CopyLink helpers: **2.960 s -> 0.034 s**.
- Ten CopyLink stage-A members: **2.830 s -> 0.348 s**. The before value is
  the planning run; lane 1 did not change this path.
- Proof: **7,488 -> 7,392 B payload**, **7,628 -> 7,530 B bincode**;
  statement stays **352 B**.
- Verifier: **5,048,805 -> 4,868,177 gas**.

Both lane targets passed: helper construction is below 0.5 s and the ten
CopyLink members are below 0.8 s.

## Idle gate

The warm run started with the measurement mutex held, no other Cargo process,
and command-start load `3.44 / 11.26 / 16.04`. The honest interval reported
`4.05 / 11.02 / 15.87` at start and `6.11 / 10.85 / 15.64` at end; the rise is
this proof's own work.

| clock | lane 1 | lane 2 | delta |
|---|---:|---:|---:|
| honest online wall | 37.523 s | 29.802 s | **-7.721 s** |
| online phase sum | 37.518 s | 29.796 s | -7.722 s |
| process CPU | 274.380 s | 242.390 s | -31.990 s |
| CPU / wall | 7.312 | 8.133 | +0.821 |

### Printed phases

| phase | lane 1 ms | lane 2 ms | delta ms |
|---|---:|---:|---:|
| wrapper preparation | 550 | 564 | +14 |
| T1/R stream adaptation | 270 | 72 | -198 |
| phase 1a commitment | 1,945 | 770 | -1,175 |
| T2 adaptation | 1,261 | 1,426 | +165 |
| T2 phase 1b commitment | 1,050 | 1,043 | -7 |
| T2 phase 2a commitment | 7,271 | 7,198 | -73 |
| T2 phase 2b commitment | 101 | 99 | -2 |
| CopyLink helpers | 2,960 | **34** | **-2,926** |
| T2 phase 2c + helpers | 344 | 383 | +39 |
| T2 finish | 457 | 598 | +141 |
| member construction | 1,983 | 1,440 | -543 |
| proof stages/opening | 19,326 | 16,169 | -3,157 |

Offline lane-2 values were 7,794 ms deterministic SRS, 193 ms key/profile,
and 457 ms fixed-key commitments. Verification took 27 ms.

### CopyLink member split

| work | before ms | after ms |
|---|---:|---:|
| helper construction | 2,960 | **34** |
| all member constructors | 1,983 | 1,440 |
| ten CopyLink stage-A members | 2,830 | **348** |

The after CopyLink value came from a temporary aggregate timer around only
`CopyLinkProver::{prove_round, finish_rounds}`. The timer was removed before
the final checks.

## Packing and wire cost

| geometry | before | after |
|---|---:|---:|
| CopyLink fixed columns | 120 | 120 |
| CopyLink fixed groups | 10 | **4** |
| final-phase CopyLink helper groups | 1 | **0** |
| proof wire groups | 21 | **20** |
| pinned key groups | 13 | **7** |
| full groups | 34 | **27** |
| stage-B rounds | 11 | **10** |

| proof section | before B | after B | delta B |
|---|---:|---:|---:|
| phase 2c wire commitments | 64 | 32 | -32 |
| stage B | 704 | 640 | -64 |
| all other sections | 6,720 | 6,720 | 0 |
| **payload** | **7,488** | **7,392** | **-96** |

The fixed columns remain key-pinned and now occupy one contiguous 120-column
range. The twenty helper columns follow T2's three phase-2c columns, using its
29 available wire slots before the pinned T2 VK group. `LimbTableKey` owns the
filled phase geometry, helper IDs, and shifted VK range used by the assembly.

## CopyLink prover representation

- Each key side retains active `(row, wire, selector, id)` entries instead of
  six dense field columns.
- Hash forms, Spartan W, T2 chunks, and sign flags stay borrowed or typed;
  helper construction reads only active positions.
- All ten denominator sets use one batch inversion. Helpers remain sparse
  until their twenty commitment columns are materialized once.
- Stage A keeps selectors, IDs, and helpers sparse. Value sources stay
  borrowed for three high-to-low binds, then materialize at one-eighth of the
  row domain. The prover evaluates `eq(tau, row)` only on active pairs and
  builds no dense per-link equality table.

The CopyLink equations, members, challenges, terms, and degrees are unchanged.

## Fiat-Shamir schedule

Challenge counts remain **`[39, 23, 1, 3, 232]`**. CopyLink `(beta, gamma)`
still follows both linked sides' phase-1b commitment. The helper columns remain
in the final commitment phase after every helper input challenge and before
the CopyLink row points and relation weights. Slot filling changes physical
positions only; it does not move a commitment or challenge boundary.

## Verifier and gates

| operation | before | after | delta |
|---|---:|---:|---:|
| ecMul | 234 | 227 | -7 |
| ecAdd | 233 | 226 | -7 |
| pairing pairs | 8 | 8 | 0 |
| Fr multiplications | 127,884 | 121,705 | -6,179 |
| Fr inversions | 10 | 10 | 0 |
| Keccak | 857 | 846 | -11 |
| **N4 gas** | **5,048,805** | **4,868,177** | **-180,628** |

| command | result |
|---|---|
| wrapper feature-enabled check after lane-1 rebase | pass |
| wrapper clippy, all targets, warnings denied | pass |
| `cargo nextest run -p jolt-wrapper --cargo-quiet` after rebase | 64/64 pass |
| locked feature-enabled `real_wrapper --no-capture` | 1/1 pass; every tamper rejects |
