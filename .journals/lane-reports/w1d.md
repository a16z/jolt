# Lane W1D report — 2^27 pressure-tier root-cause (2026-08-04)

Branch `gpu/util-w1d`, worktree `~/dev/jolt/.worktrees/gpuutil-w1d`, never
pushed. Companion artifact with full evidence:
[`w1d-rootcause.md`](w1d-rootcause.md).

## Headline

**The "2^27 pressure tier" does not exist on trunk as theorized.** Direct
measurement (vm_stat sidecar, footprint ledger, park-vs-free ablation)
rules out every pressure mechanism the lane was chartered to fix: zero
compressor activity, zero swap, free memory ≥ 39 GiB throughout a 2^27 run,
and — decisively — freeing the stage-5 30 GiB pair at retire instead of
parking it moves the total by **+0.12 s (+0.14%)** and st6b by **−0.17 s
(noise)** in a same-binary same-window A/B. The W4 U1 door ("structurally
end or decommit stage-5 ownership before the stage-6b adoptions") is closed
with a measured null.

## What the degradation actually is

- **st6b (×4.33 for the 2^26→2^27 doubling, ~+7.4 s canonical excess):**
  intrinsic CPU working-set shape. The inflated prepares/rounds run
  parallel-busy (11.5 avg cores, 64% CPU, 0% GPU) on warm malloc-recycled
  pages at low fault rates — DRAM-latency-bound table builds, not stalls.
  This is exactly the surface lane W1B's ports remove (BRRC prepare+rounds
  7.1 s, IncCR prepare 2.6 s, eq/oracle feeds ~3.8 s instrumented). No
  allocator or lifetime change moves it.
- **st4 (constant ×2.05-2.06 per doubling 2^25→26→27, no tier cliff):**
  shape-coupled, dominated by `RegistersRWC::prepare` = 4.7 s **serial**
  host build (1.9 cores / 10% CPU). Wave-2 parallelize/port target; CSR
  rewrite remains rejected (W4 U3). Handed off.
- **W3-era T2-vs-T6 ±8 s st6b variance, re-attributed:** the TraceRecord
  lane family (~28 GiB at 2^27) died before st6b in the fast trees and
  inside/after it in the slow ones (st6b entry footprint 36.9 vs 67.7 GiB).
  W4 U2's lifetime restructuring already banked this on trunk (entry
  41 GiB). The arena pair was a bystander.
- **Why U1's `MADV_FREE_REUSABLE` failed:** measured with a new ignored
  probe test — REUSABLE instantly drops phys_footprint on virgin malloc
  memory and is a *silent success-returning no-op* on any range ever
  wrapped by `newBufferWithBytesNoCopy` (live or released buffer). It never
  could have worked on Metal-wrapped slabs, and footprint residency wasn't
  the tax anyway.

## What landed

1. `08baac471` + `bdfdb1556` — root-cause artifact (census-measured slab
   inventory: factors=5 pair, 29.5 s parked window at 2^27; per-stage
   RSS/footprint ledger; sidecar counters; scaling tables; ablation
   verdict).
2. `0e5741ca5` — madvise-REUSABLE probe (ignored diagnostic test in
   `metal/buffers.rs`, run via `--run-ignored all`).
3. `18beb5cd6` — `JOLT_METAL_NO_PARK` ablation knob (diagnostic; superseded
   by the deletion, kept in history for the A/B record).
4. `1c8df377c` — **delete the retired-buffer placement arena** (−381/+28
   lines): `ArenaSlab`/`ArenaLease`, the `RETIRED` pool, `RetiredPoolGuard`,
   the carve path, and both retire sites are gone; producers drop their
   ping-pong pairs where they used to park them. malloc's large cache
   provides the identical warm-page recycling. Removes the machinery that
   attracted two failed fix waves and a latent footprint stack-up mode.
5. `w1d-rootcause.md` §8 + this report.

## Validation

- 2^25 interleaved A/B ×3: park 23.19 min vs deletion 23.17 min (−0.09%,
  neutral inside ±1%).
- 2^27: controlled knob A/B +0.12 s total (park 88.249 / free 88.369, same
  window); deletion sanity run 91.56 s with ~3-4% same-day ambient
  inflation (wave-2 sibling builds), zero swap, no localization at the
  changed sites.
- Full gate matrix green on the deletion tree: kernels 231/231, dory 46/46,
  **byte-diff 11/11 host + 11/11 metal (proof bytes identical)**, muldiv
  3/3 + 3/3, witness 34/34, clippy `-D warnings` ×3 feature sets, fmt.

## Accept-gate accounting (honest)

Target was st4+st6b −4 s vs canonical 24.32 s. **Delivered: ~0 s wall** —
the door the target assumed was already worth ~0 on trunk; no
allocator/lifetime fix can reach it. Deliverables instead: the mechanism
verdict with receipts (redirects campaign effort to W1B ports + a wave-2
st4-prepare parallelization, which own the real ~12 s), the REUSABLE pin,
and a perf-neutral −353-line simplification removing the failure-mode
attractor. 2^27 run budget: 4 of ≈6 used.

## Doors opened / recommendations

- **st4 wave-2 lane:** parallelize `RegistersRWC::prepare` (serial 4.7 s
  @2^27, 1.9 cores). Bounded, port-free, ~3-4 s prize if it parallelizes
  like the other prepares.
- **st6b residual after W1B:** eq-evals (2.2 s) and `oracle_table` (1.6 s)
  feeds inflate superlinearly at 2^27; if W1B's ports land and st6b still
  lags 2×2^26, profile those two next.
- The st6a footprint cliff (record family death) is scale-parity-dependent
  (dies at st8 for even log_T) — if peak footprint ever becomes the binding
  constraint at 2^28+, moving the record drop earlier is the lever U2
  already demonstrated.
