# PERF-3 — nest-safe affine MSM

Date: 2026-09-03 · Mac mini M4 · 10 Rayon threads · CPU only.

## Result

No code landed. The best candidate made every production context faster than the projective
control and fixed the nested-Rayon utilization loss, but missed two requested gates:

| gate | projective control | candidate min / median | target | result |
|---|---:|---:|---:|---|
| isolated 2^21 MSM | 0.552 / 0.555 µs/point | 0.420 / 0.430 µs/point | ≤0.400 | miss |
| four mixed/Fr helper groups | 2.661 s | 2.085 / 2.109 s | ≤2.000 s | miss |
| HyperKZG open | 2.466 / 2.486 s | 2.000 / 2.025 s | ≤2.100 s | pass |

Control loads: isolated 8.07; full gate 8.45–10.37. Candidate repeat loads: 5.59–6.16.
The candidate keeps proof bytes and the `g1_msm` API unchanged.

## Diagnosis

PERF-2's regression was nested Rayon scheduling, confirmed by the utilization progression:

| variant | helper block | busy threads |
|---|---:|---:|
| PERF-2 window × nested chunk iterators | 5.396 s | 6.6 |
| flat tasks, one point chunk/window | 8.136 s | 3.62 |
| flat tasks, three point chunks/window | 4.624 s | 7.09 |
| flat tasks + scalar-width split | 2.085 s | 9.51 |

The flat `(window, point chunk)` graph removes one Rayon level. Scalar-width splitting is also
required: packing group 22 is entirely bit/u16 data, group 29 is 5/8 full-width, group 30 is
full-width, and group 31 is 6/8 full-width. Running all four through the full-width affine pass
spends the saved curve work on empty high windows.

## Best candidate

- PERF-2 signed-Booth batch-affine buckets, 16-bit windows at sizes ≥2^20.
- One flat parallel iterator over window × point-chunk tasks.
- Exact all-u16 dispatch to the existing small-scalar kernel.
- Mixed scalar vectors split into compact full-width and u16 subsets when a 4K sample has at
  least 1/8 u16 values; the split affects scheduling only, never the group result.
- Uninitialized linked-list `next` storage; every reachable node initializes its slot before the
  slot is read.

Three clean targeted repeats:

| load | helpers | isolated 2^21 | HyperKZG open |
|---:|---:|---:|---:|
| 5.59 | 2.085 s | 0.430 µs/point | 2.025 s |
| 6.01 | 2.109 s | 0.420 µs/point | 2.096 s |
| 5.83–6.16 | 2.246 s | 0.442 µs/point | 2.000 s |

Command:

```bash
CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/perf1 cargo nextest run -p jolt-wrapper --release \
  perf2_commit_open_profile --run-ignored ignored-only --cargo-quiet --no-capture
```

## Rejected variants

- Six nested chunks raised utilization to 8.50 threads but added bucket overhead: helpers 3.044 s,
  open 2.209 s.
- Sequential mixed groups did not solve the CPU floor. Per-group times summed to 2.179 s at load
  7.11: 0.065 + 0.618 + 0.878 + 0.619 s.
- Cache-local nested windows for top-level calls, an active-bucket list, affine partial-bucket
  merges, two chunks for compact subsets, and retained prefix buffers all lost or were neutral.
- No `stream.rs` change was made; W5 ownership was respected.

## Verification and status

- Focused MSM tests pass: naive random/full-width equality; zero, identity, duplicate, inverse,
  non-power-of-two, mixed-width, and small-scalar consistency cases.
- The earlier full `jolt-crypto` suite passed with the affine kernel: 143 tests.
- Full statement gate not run; the targeted gates already reject the candidate.
- `perf1/profile` retains the unstaged experiment for follow-up. `wrap/spartan-hyperkzg` received
  no MSM code.

## Landing decision

The orchestrator accepted the uniform production gain despite the two stretch-target misses.

- Candidate committed on `perf1/profile` as `cdc5fc9a6`, then rebased onto main/W5 as
  `add9b7a31`.
- W5's stream stack is present at `c305fb312`.
- `cargo clippy -p jolt-crypto --all-targets -q --message-format=short -- -D warnings` passed.
- `cargo clippy -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings` passed.
- Combined `jolt-crypto` + `jolt-hyperkzg` nextest: 169 passed.
- Main was not fast-forwarded: `crates/jolt-hyperkzg/tests/commit_open_verify.rs` is modified and
  `crates/jolt-hyperkzg/src/.journals/` is untracked in the shared worktree.

The required post-rebase profile launched at load 9.10. A separate six-core `wrap_real_t1` test
started during the run. Helpers measured 2.383 s at 8.58 busy threads; isolated MSM measured
0.461 µs/point at 8.45; HyperKZG open measured 4.063 s at 4.52. These contended values do not
replace the clean three-repeat results above.
