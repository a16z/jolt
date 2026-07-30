# Release packed opening rows after the accepted root fold

Date: 2026-07-30 EDT

## Question

Can Akita release the packed trace-one-hot row cache during opening, once the
accepted root fold has consumed it, without changing the protocol or slowing
the prover?

The hint retained 29 lane bytes per cycle until the entire opening returned.
Only the root evaluation and root decomposition read those rows. The
recursive fold tail consumes the accepted root-fold witness instead.

## Safe release boundary

Akita can retry root decomposition while grinding its nonce, so releasing the
rows inside an evaluation or decomposition kernel would be unsound. The safe
boundary is immediately after `prepare_root` returns successfully:

1. Root evaluation and every decomposition retry have finished.
2. The accepted nonce and root-fold witness have been constructed.
3. All source views have ended.
4. The recursive fold consumes that witness and does not reread the root
   polynomial.

`RootPolyMeta` now has a default no-op release hook. The root-fold driver calls
it at this boundary, and Jolt's trace-backed source drops its packed rows.
Shape metadata remains cached in the source. A view holds one guard for its
entire kernel, which prevents a concurrent release while rows are being read.

Jolt also drops its typed `Arc<JoltOneHotTraceRows>` after deferred RA-row
materialization. Without that ownership fix, the Akita hint would not be the
last owner and the hook would not return the allocation.

Cloned hints preserve their prior behavior. Each clone gets an independent
release handle but shares the immutable row allocation. Releasing one hint
drops only its reference; another clone remains usable until it is released.

This changes no commitment, transcript message, Fiat-Shamir challenge,
opening claim, proof object, or verifier behavior.

## Expected result

The exact late-lifetime saving is 29 B/cycle:

| Trace size | Rows released |
|---|---:|
| `2^22` | 121,634,816 B (0.1133 GiB) |
| `2^26` | 1,946,157,056 B (1.8125 GiB) |
| `2^28` | 7.25 GiB |

The root kernels are unchanged. The only new target-scale work is an
`Arc`/lock ownership transition and freeing the row allocation. A repeatable
opening or whole-prover slowdown would reject the candidate.

## Measurements

All runs forced K256 and verified the proof.

### `2^22` screen

| Run | Prove | Commitment | Packed opening |
|---|---:|---:|---:|
| Control A | 5.765 s | 1.575 s | 2.668 s |
| Control B | 5.683 s | 1.559 s | 2.665 s |
| Candidate A | 5.784 s | 1.602 s | 2.692 s |
| Candidate B | 5.721 s | 1.583 s | 2.683 s |

The two pairs overlap at ordinary run-to-run scale. Both candidate traces
report the exact 121,634,816-byte cache, and the release takes approximately
1 ms.

### `2^26` target

| Run | Prove | Commitment | Stage 6b | Packed opening | Maximum RSS |
|---|---:|---:|---:|---:|---:|
| Control | 53.543 s | 22.488 s | 5.194 s | 11.144 s | 38.924 GB |
| Candidate A | 54.208 s | 22.923 s | 5.225 s | 11.250 s | 38.928 GB |
| Candidate B | 53.697 s | 22.685 s | 5.241 s | 11.124 s | 38.876 GB |

Candidate A's 0.665-second whole-proof movement includes a 0.435-second
movement in the unchanged commitment. The repeat is 0.155 seconds above the
control while the affected opening is 0.019 seconds faster. The evidence
supports performance neutrality, not a speedup.

The trace marker reports exactly 1,946,157,056 retained bytes in both target
runs. It begins 0.54 ms after the accepted root fold ends and takes 12.84 and
12.94 ms.

| Run | Sampled global max | Opening max | Opening end |
|---|---:|---:|---:|
| Control | 35.80 GiB | 32.62 GiB | 20.42 GiB |
| Candidate A | 34.29 GiB | 31.92 GiB | 18.62 GiB |
| Candidate B | 34.22 GiB | 30.24 GiB | 18.09 GiB |

Opening-end RSS falls by 1.80–2.33 GiB, bracketing the exact 1.8125 GiB
allocation after allocator and sampling effects. Opening maximum falls by
0.70–2.38 GiB. The process maximum is effectively unchanged because Stage 6b
precedes the release and remains the limiting phase.

## Outcome

Accepted. Akita commit `82b3b856` adds the generic lifecycle boundary; Jolt
commit `cffef8618` releases the trace-backed rows and removes the extra local
owner. At `2^28`, this removes 7.25 GiB from the recursive opening tail with
no measured target-scale runtime regression.

The result improves late capacity and swap headroom, but it does not lower the
current Stage-6b peak. The next experiment should address overlapping
Stage-6b field vectors rather than further opening-source lifetime.

## Validation

- `cargo nextest run -p jolt-akita --cargo-quiet`: 43 passed, 5 skipped
- `cargo nextest run -p jolt-prover-legacy --features host,akita -E
  'test(e2e_akita)' --cargo-quiet`: 5 passed
- Forced-K256 `2^22` proofs: two passed and verified
- Forced-K256 `2^26` proofs: two passed and verified, zero swaps
- `cargo clippy -p jolt-akita --lib --no-deps -- -D warnings`: passed
- `cargo clippy -p jolt-prover-legacy --features akita --lib --no-deps -- -D warnings`:
  passed

Workspace `--all-targets` clippy remains blocked by the user's untracked
`crates/jolt-akita/tests/schedule_probe.rs`, whose debug prints and unwraps are
outside this change.

## Retained artifacts

- `benchmark-runs/perfetto_traces/mem-opening-release-2e22.json`
- `benchmark-runs/perfetto_traces/mem-opening-release-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-opening-release-2e26.json`
- `benchmark-runs/perfetto_traces/mem-opening-release-2e26-b.json`
- `benchmark-runs/akita-memory-2e28-2026-07-29/logs/mem-opening-release-2e26.log`
- `benchmark-runs/akita-memory-2e28-2026-07-29/logs/mem-opening-release-2e26.rss`
- `benchmark-runs/akita-memory-2e28-2026-07-29/logs/mem-opening-release-2e26-b.log`
- `benchmark-runs/akita-memory-2e28-2026-07-29/logs/mem-opening-release-2e26-b.rss`
