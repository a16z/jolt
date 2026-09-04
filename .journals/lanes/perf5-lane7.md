# PERF-5 lane 7 — default packing k=16

Date: 2026-09-03. Base: `9a6643df5`; rebased onto `b8e05dc35`
(journal-only changes). Fixture: `fibonacci_2_18_blake3.bin`, Mac mini M4,
ten Rayon threads for isolated real gates.

## Decision and scope

The lane-5b idle pair selected k=16: 16.978 s versus 19.671 s at k=32,
with 288 B more payload and 143,924 more modeled gas (3.0%). The k=16
payload equals the pre-four-ary k=32 payload, 7,392 B.

- `DEFAULT_PACKING_FACTOR` is 16. The real gate reads `WrapConfig::default()`;
  `WRAP_K=32` remains the comparison override. The gate pins the default itself.
- Both packings pin wire-phase/full group geometry, opening shape, payload,
  bincode size, statement bytes, the complete verifier operation vector, and gas.
- Phase challenges remain `39 / 23 / 1 / 3 / 232` at both packings. The final
  count is 22 T2 challenges plus ten CopyLink points and three weights per link:
  `22 + 10 * (18 + 3)`. Packing does not enter that count.
- `pr-tables.md` uses k=16 as the primary column and this lane's measured pair.
  Historical journals are unchanged. `pr-body.md` belongs to the orchestrator
  and is excluded from this lane.

## Measured real gates

Both gates accepted the honest proof and rejected every existing tamper.

| measurement | k=16 default | WRAP_K=32 |
|---|---:|---:|
| honest online wall / phase sum | 16.507 / 16.500 s | 19.785 / 19.779 s |
| fold commitments | 0.733848 s | 1.332783 s |
| quotient MSM | 1.869708 s | 3.723125 s |
| total HyperKZG opening | 2.794118 s | 5.644792 s |
| payload / bincode / statement | 7,392 / 7,533 / 352 B | 7,104 / 7,232 / 352 B |
| proof wire / key / full groups | 33 / 11 / 44 | 19 / 7 / 26 |
| opening variables / committed folds | 22 / 10 | 23 / 11 |
| ecMul / ecAdd | 233 / 233 | 216 / 216 |
| pairing pairs | 8 | 8 |
| Fr mul / inversions | 123,144 / 8 | 123,121 / 8 |
| Keccak | 852 | 839 |
| N4 gas, modeled from observed operations | 4,944,149 | 4,800,225 |
| process CPU / CPU-to-wall | 136.600 s / 8.275 | 158.380 s / 8.005 |
| native verifier, outside online clock | 27 ms | 25 ms |

This pair saves 3.278 s online at k=16. No proof or verifier-cost change
relative to the corresponding lane-5b packing was observed.

## Idle windows

| gate window (ET) | command-start load, 1/5/15 min | online-start/end, 1 min | command-end load, 1/5/15 min |
|---|---|---|---|
| k=16, 23:21:07–23:21:33 | 2.47 / 9.28 / 8.81 | 3.16 / 5.12 | 5.51 / 9.45 / 8.89 |
| k=32, 23:45:15–23:45:49 | 1.41 / 2.04 / 3.28 | 2.25 / 4.66 | 4.44 / 2.74 / 3.48 |

Each window acquired `/tmp/wrapper-gate.lock` with `mkdir`, checked
one-minute load below 4 and no compiler/other test process, then used saved
nextest binary and Cargo metadata. Unavailable windows retried after 60 s.
The runner sampled processes once per second; neither accepted window saw
a competing job. Each runner exited 0 and removed the mutex with `rmdir`.
SRS/key setup precedes the online clock and raises load before that clock starts.

The default gate completed before daemon restart #222; its log records normal
test and runner completion. The comparison reused the same prebuilt binary
after restart. Default execution explicitly unset `WRAP_K`.

Logs and runner: `/tmp/perf5-lane7/{default,k32}.log`,
`/tmp/perf5-lane7/gate.py`; metadata: `binaries.json`, `cargo.json` in that directory.
Prebuild used `cargo nextest run -p jolt-wrapper --features prover-fixtures
--cargo-quiet real_wrapper --no-run`, with
`CARGO_TARGET_DIR=/Volumes/Dev/target/perf5-lane7`.
Metadata-based nextest runs omit `--cargo-quiet`, which nextest rejects with
`--binaries-metadata`. Temporary fold/quotient timers were removed before final checks.

## Verification

`cargo fmt -q --message-format=short`, feature-enabled all-target check, and
matching clippy with `-D warnings` passed. The initial six-worker wrapper suite
passed 64/64; both feature-enabled real gates passed 1/1 with all tampers rejecting.
No new test helper, ignored test, timing probe, or benchmark was retained.

After rebase and timer removal, all-target check/clippy passed, and
`NEXTEST_TEST_THREADS=6 cargo nextest run -p jolt-wrapper --cargo-quiet`
passed 64/64 in 131.522 s. Logs: `/tmp/perf5-lane7/final-{check,clippy,unit}.log`.
