# Metal W11 st4 — re-attribution + lazy-zero bind buffers

## Verdict

**RETAIN.** st4's top item was not device work: 2.84 s @2^27 of the 8.16 s
stage is a single-threaded host `memset` — `PageAlignedVec::from_elem`
zero-filling each cycle round's fresh multi-GiB output CSR (~44 GiB of
eager zero-fill per proof) before the GPU overwrites every byte. Cut:
bind outputs become kernel-zeroed `MmapVec` mappings (`own_mmap`), the
exact pattern wave-3's prepare salvage already used for the initial CSR.
Measured @2^24 e2e ABBA: **st4 −0.120 s (−14.7%)**, pairs agree; modeled
**−1.95 s st4 @2^27** (conservative −1.4 s). Buffer contents are
byte-identical (zeros then the same GPU writes) — proof bytes unchanged
by construction; byte-diff 20/20 first pass. Kill switch
`JOLT_REGRW_MMAP_BIND=0`. Commit: see below.

## Phase 1 — re-attribution @2^27 (one instrumented profile)

Run: sha2-chain 2^27 metal, `--format chrome`, GPU-locked, quiet box,
FrBind 252.9 µs (healthy), wall 54.55 s (ordinary daytime window; matches
the wave-10 51-54 s class + tracing overhead), RSS 72.36 GiB (= trunk).
st4 traced 8.158 s (stage vector 8.04 — consistent).

st4 is a two-member batch: `RegistersReadWriteChecking` (Metal slot) +
`RamValCheck` (CPU-only kernel, log_t rounds, no Metal slot).

| component | s | share |
|---|---:|---:|
| **RegRw::alloc_entries (host memset)** | **2.844** | **34.9%** |
| RegRw::bind_run (bind CB wait) | 1.180 | 14.5% |
| RegRw::msg_run (message CB wait) | 0.824 | 10.1% |
| RamValCheck rounds (CPU: message 1.218 + bind 0.293) | 1.511 | 18.5% |
| RegRW prepare (meta 0.635 + GPU build 0.317 + residual 0.506) | 1.458 | 17.9% |
| RegRw::install / scan_offsets / msg_sums | 0.182 | 2.2% |
| RamValCheck prepare (inc_column 0.069) | 0.109 | 1.3% |
| batch engine + stage misc + host address tail | ~0.05 | 0.6% |

- Host address rounds (post-transition, ≤128 entries) ≈ 0 — the wave-2
  intuition holds; no address-side mass.
- The suspected serial count scan is a non-item: 0.077 s total.
- Per-round output sizes (`new_count`): r0 176.4M entries (Indexed 56 B =
  9.2 GiB), r1 128.1M, r2 92.2M (deref → Direct 120 B = 10.3 GiB),
  r3 66.1M, then geometric decay. Eager zero-fill ran at ~17 GB/s on one
  core inside `plan_bind`, on the critical path of every cycle round.

## Phase 2 — the cut

`DeviceRegistersRwState::plan_bind`: output CSR allocation switches from
`own_page_aligned(PageAlignedVec::from_elem(default, new_count))` (serial
element-write loop) to `own_mmap(MmapVec::zeroed(new_count))` (MAP_ANON,
kernel lazy zero-fill, munmap on drop). The `DeviceEntries` enum already
holds mmap-backed buffers on the prepare path, so this is a constructor
swap; zero pattern == `default()` for both entry types, and the GPU bind
scatter writes every slot in `[offsets[p], offsets[p+1])`.

### Receipts — @2^24 e2e ABBA (one binary, kill switch, 40 s cooldowns)

| | ON-A | OFF-B | OFF-B | ON-A |
|---|---:|---:|---:|---:|
| wall (s) | 7.94 | 8.00 | 7.96 | 7.88 |
| st4 (s) | 0.702 | 0.818 | 0.816 | 0.692 |
| RegRw::alloc_entries | 0.001 | 0.183 | 0.183 | 0.001 |
| RegRw::bind_run | 0.224 | 0.166 | 0.164 | 0.219 |
| RegRw::msg_run | 0.117 | 0.125 | 0.123 | 0.119 |
| RamValCheck rounds | 0.156 | 0.138 | 0.139 | 0.148 |

st4 delta −0.116/−0.124 (pairs agree). Mechanism exactly as modeled:
alloc −0.182, bind CB +0.057 — the first-touch page faults move into the
GPU write window (31% clawback). RSS @2^24: 13.09 → 12.89 GiB.

### Model @2^27

Removed: 2.844 s measured. Clawback: byte-proportional fault cost at the
measured 31.3% → +0.89 s in bind CBs. **Net −1.95 s st4** (8.16 → ~6.2 s);
conservative bound at 50% clawback −1.42 s. Both clear the ≥1.0 s bar.
Transient CSRs additionally munmap out of the footprint at drop instead
of parking in libmalloc (the W3A citizen pattern).

### Soundness / parity

No protocol content touched: allocation strategy only, identical zeroed
contents, identical kernel writes, identical readbacks. Proofs byte-equal
by construction; gates below confirm.

## Gates

- metal suites (jolt-kernels + jolt-dory + jolt-eval, metal): **411/411**.
- byte-diff ratchet `-p jolt-prover --features prover-fixtures`: **20/20
  first pass**.
- `cargo clippy --all --features host --all-targets -- -D warnings`: clean.
- `cargo fmt`: applied.
- E2e verify passed on every profile/bench run (4×@2^24, 1×@2^27).

## Doors closed / priced (with receipts)

1. **Batch overlap, priced SUB-BAR (~0.75–0.9 s @2^27 post-cut):** the
   sumcheck engine already runs `begin_round` (launch) → synchronous
   members → `collect_round` (wait); the st4 Metal slot never opted in, so
   RegRW's blocking CB waits serialize with RamValCheck's CPU rounds. The
   prize is Σ_r min(RegRW CB_r, RamVal CPU_r): per-round traces give r0
   0.37, r1 0.20, r2 0.08, decaying — RamVal's dense-CPU halving decays
   faster than the CB series, so the overlap window is front-loaded and
   caps out below the bar. Implementation path exists (w10
   `registers_val_evaluation` detach exemplar + the in-tree
   `bind_and_message` single-CB encoder). This is also the honest
   REPRICING of the w3 round-fusion NO-GO: fusion-as-detach-enabler is a
   different mechanism than fusion-for-CB-count (−5.7%), but its prize is
   ~0.8 s, not ≥1.0 s. Kill-list entry stands for standalone fusion.
2. **Grandparent arena reuse, priced ~0.9 s, not taken:** entry counts are
   monotone non-increasing (pair-merge unions), so round r+2's output fits
   in round r's buffer — reuse would kill the +0.89 s fault clawback. Adds
   cross-round lifetime coupling; sizing a persistent arena at Direct
   width is the W2B fat trap (~+21 GiB). Revisit only if the clawback
   measures larger than modeled at a wave gate.
3. **W2B "middle ground" (parked since gpu-util): SUBSUMED.** Its target
   was the per-round representation/allocation overhead between fast+fat
   and lean+slow. With the CSR exact-legacy (w3) and allocation now lazy
   (this lane), the representation-side host cost is 0.001 s + fault tail;
   the remaining round-loop mass is in-kernel CB time. No coherent
   middle-ground prize survives.
4. RegRW prepare residual 0.506 s (shared-record fetch, `own_vec` adopts,
   counts init, RoundTable adopt) — unattributed below span granularity,
   sub-bar; next re-attribution can sub-span if prepare grows.
5. RamValCheck device port: superseded by the overlap framing — porting it
   to Metal would re-serialize it into the same GPU queue it currently
   complements; only worth it combined with door 1's scheduling.

## Discipline

- Timed budget: ONE 2^27 instrumented profile (phase 1); ONE @2^24 ABBA
  (4 runs) for the phase-2 decision. No 2^27 certification runs.
- All cargo under `/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock`;
  every GPU run additionally under `/tmp/jolt-metal-gpu.lock`; FrBind
  health-checked (252.9 µs) before timed work; 40 s cooldowns.
- Attribution spans (`RegRw::*`, `RamVal::*`) are shipped — per-round
  frequency, matching the in-tree `IrrKernel::*`/`IrrScanner::*`
  precedent; st4 stays decomposable at future gates.
- No pushes; no touch of scratch/metal-saturation or sibling worktrees.
- KernelId::ALL unchanged (83) — no kernels added. commitment.rs untouched.
