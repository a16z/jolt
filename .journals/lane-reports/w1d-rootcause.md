# W1D root-cause artifact — 2^27 st4/st6b degradation (2026-08-04)

Status: evidence phase complete except the decisive park-vs-free ablation
(§6). Committed before any fix per lane protocol. Ablation result appended in
§7 after run #2.

## 1. Slab inventory at the st5→st6b boundary (who allocates, parks, adopts)

One producer, one pool, five adoption families. All numbers measured via
`JOLT_METAL_ALLOC_TRACE=1` census runs at 2^26 and 2^27 (this tree, runs of
2026-08-04) — not inferred.

**Producer.** Stage 5 `DeviceIrrScanner` (`metal/slots/instruction_read_raf.rs`)
materializes its flat cycle ping-pong via `own_uninit_frs`:
`cur = factors·T·32 B`, `nxt = factors·(T/2)·32 B`, **factors = 5**
(1 + ra_count=4; read off the census: 10+5 GiB @2^26, 20+10 GiB @2^27).
Allocated mid-st5 when the address→cycle handoff runs (57.8 s @2^27
timeline), device-write-faulted under the materialize dispatch.

**Park.** When device cycle rounds end (`take_cycle_tables`, or scanner drop
at st5 end), the pair retires into the process-global `RETIRED` pool
(`metal/slots/mod.rs`) as two `ArenaSlab`s. The pool holds a strong `Arc`
until first carve; the slab keeps the original `MTLBuffer` (`_owner`) alive
the whole time. Pages stay dirty-anonymous-live in `phys_footprint` for the
entire parked window.

**Adopt.** st6b carves (silent in the census — leases are untraced) and
fresh-allocates (visible): IncClaimReduction's four `RoundTable.nxt` (4×T/2·32
= 4 GiB @2^27) carve the pool; its four `cur` (16 GiB @2^27) are
prepare-built host Vecs (`vec_adopt` events, ids 989–992); ra_lazy
materialize allocates 16-poly dense pairs fresh (8+4 GiB @2^27, ids
994/995 — pool best-fit misses); BRRC (CPU member) never touches the pool.

**Lifetime (measured, 2^27):** pair allocated 57.8 s; `cur` (20 GiB) freed
87.35 s = st6b batch end; `nxt` (10 GiB) frees with the last lease at st6b
close. **Parked-idle window ≈ 29.5 s** spanning late-st5 CPU tail (~8.5 s),
all of st6a (4.1 s), and all of st6b (17.4 s). Identical structure at 2^26
(9.0 s window). The pool serves nobody before st6b — every earlier stage's
ping-pong pairs alloc and free inside their own stage (census ids 929–948).

## 2. Footprint/RSS/page-fault evidence at the boundary (2^27, this tree)

Per-stage ledger (instrumented run, 2026-08-04; matches the 2026-08-03
campaign trace within noise):

| stage | RSS open→close | footprint open→close |
|---|---|---|
| st4 | 61.9 → 71.8 | 60.4 → 70.8 |
| st5 | 71.8 → **46.8** | 70.8 → **71.0** |
| st6a | 46.8 → 36.7 | 71.0 → **41.0** |
| st6b | 36.7 → 36.2 | 41.0 → 51.3 |

- **st5: RSS −25 GiB with footprint flat.** Not the compressor (see below) —
  this is Metal *unwiring* the IRR working set when device rounds end. The
  parked pair stays footprint-live (dirty anonymous), just no longer wired.
- **st6a: footprint −30 GiB = genuine frees.** The census shows zero device
  buffer frees in the window → host memory: the TraceRecord lane family
  (~150 B/cycle ≈ 19 GiB) + RamAccessColumns (24 B/cycle ≈ 3 GiB) + related
  walk co-products, dropped by their last consumers (st6a address-phase
  prepares). Scale-parity artifact: at 2^26 the same family dies at st8
  instead (0 GiB frees in st6a — boundary-sample tables differ by log_T
  parity), which is why stage-boundary footprints looked contradictory
  across scales.
- **st6b entry footprint on this tree is 41 GiB — already T6-like healthy.**
  The W3-era catastrophe correlate (st6b entry 67.7 GiB → 19.8–21.8 s st6b)
  is structurally absent from trunk: W4 U2's lifetime restructuring moved
  the record family's death ahead of st6b. What remains parked across st6b
  is exactly the 30 GiB IRR pair.

**vm_stat sidecar (1 Hz, whole run): compressions = 0, decompressions ≈ 0,
swap 0.25 MB constant, system free ≥ 39 GiB at all times.** Zero-fill fault
bursts reach 8+ GiB/s during st4's transient churn (the machine faults fast
when asked); st6b's inflated prepares run at *low* fault rates on
malloc-recycled warm pages.

## 3. The mechanism, named (and the W4 story corrected)

**On the current tree the 2^27 st6b degradation is NOT OS page pressure.**
No compression, no swap, no free-memory exhaustion, no Metal device
failures/fallbacks (traces clean), one-hot config identical across
2^25/26/27 (`ONEHOT_CHUNK_THRESHOLD_LOG_T = 25`; log_k_chunk = 8 at all
three scales).

What the instrumented counters show instead, during the inflated windows
(2026-08-03 2^27 trace):

| span | wall | cpu% | avg cores | gpu% |
|---|---|---|---|---|
| BRRC::prepare (st6b) | 2.24 | 64 | 11.5 | 0 |
| IncCR::prepare (st6b) | 2.58 | 64 | 11.5 | 0 |
| BoolAddr::prepare (st6a) | 2.88 | 69 | 12.4 | 0 |
| RegRWC::prepare (st4) | 4.70 | **10** | **1.9** | 0 |
| st6b prove_batch | 10.24 | 32 | 5.8 | 38.5 |

The st6a/st6b prepares are **parallel and busy** (11–12 cores at work), not
stalled on faults — consistent with DRAM-latency/bandwidth-bound table
builds at 2^27 working-set sizes, and *possibly* aggravated by 30 GiB of
idle parked pages (pmap/TLB scale, allocator placement). The two candidate
mechanisms left standing:

- **H-park:** the parked 30 GiB harms the st5-tail→st6b window even without
  compressor involvement (the W3 T2-vs-T6 correlation was real and causal).
- **H-shape:** the superlinear 2^26→2^27 member scaling is intrinsic
  (latency curve of the memory system at 2× working set), the T2-era
  catastrophe was a *different*, pre-U2 mechanism (record family resident →
  entry footprint 67.7 GiB → genuine kernel pressure on those trees), and
  parking is now harmless on trunk.

These are indistinguishable by observation on this tree — every trunk run
parks identically. They separate **only by ablation**: free the pair at
retire (structural end of stage-5 ownership) vs park, same binary, 2^27.
That ablation is run #2; result in §7.

**Why this artifact still supports the structural fix either way:** the
parked pair pins 30 GiB of garbage-content pages across a 29.5 s window in
which at most ~4 GiB is ever carved before mid-st6b (IncCR nxt's), the
biggest adoption (ra_lazy 12 GiB) *misses* the pool anyway, and the pool's
design premise — avoid fresh-page zero-fill for st6b — is worth at most
~1.6 s of faulting at the measured 8 GiB/s while the demonstrated downside
risk of holding it (T2-era mode: +8 s st6b; U1 confirmation: +8.0 s st6b at
entry footprint 71 GiB) is 5× larger and timing-fragile.

## 4. Why W4 U1's MADV_FREE_REUSABLE failed

Commit `69e7d75d4` (reverted `276396aed`) applied `MADV_FREE_REUSABLE` to
the whole slab range at adopt (retire) time and `MADV_FREE_REUSE` per carve.
U1's own confirmation measured **st6b-entry footprint 71.02 GiB** — the
madvise did not remove the parked pages from the footprint ledger at all
(a working REUSABLE drops phys_footprint immediately).

Root cause of the failure, established by micro-experiment (§5): the slab
keeps its `MTLBuffer` (and leases mint more) over the same pages —
**pages referenced by a Metal no-copy buffer's IOGPU mapping do not
transition to reusable; the madvise returns 0 but is a silent no-op** while
the buffer object lives. The unit test that "verified" the transition only
checked the return code, which lies. Additionally the intent was misaimed:
on trunk there is no compressor activity for REUSABLE to short-circuit
(§2) — the pages' footprint residency was never the tax it was assumed
to be.

## 5. Micro-experiment: REUSABLE vs Metal wrap (footprint ledger)

Ignored test `metal::buffers::madvise_probe::madvise_reusable_vs_metal_wrap`
(this branch): dirty 1 GiB page-aligned, apply `MADV_FREE_REUSABLE`, read
`phys_footprint` deltas via `proc_pid_rusage(RUSAGE_INFO_V4)` (ledger sanity
leg: dirtying moves the ledger +1024 MiB). Measured 2026-08-04:

| variant | madvise rc | footprint delta |
|---|---|---|
| plain malloc'd, never Metal-touched | 0 | **−1024 MiB** (works) |
| wrapped in live no-copy `MTLBuffer` | 0 | **0 (silent no-op)** |
| wrapped, then buffer released before madvise | 0 | **0 (silent no-op)** |

`MADV_FREE_REUSABLE` works exactly as documented on virgin malloc memory
and does nothing — while still returning success — on any range that has
ever been mapped by `newBufferWithBytesNoCopy` (IOGPU holds a second
reference to the VM object; release does not restore eligibility on any
relevant timescale). U1 applied it under a live owner `MTLBuffer` plus
per-carve lease wraps: it could never have worked, and its return-code
check (and the unit test asserting `reclaimable`) verified nothing.

Corollary for fix design: any madvise-shaped decommit of Metal-wrapped
pages is dead on arrival. Ending stage-5 ownership means actually dropping
the buffer AND its backing allocation. A dropped `Vec` backing goes to
malloc's large-entry cache (the sanity leg shows `free()` alone also leaves
footprint: +1024 MiB residual — `MADV_FREE`-tagged, reclaim-on-demand,
reusable warm by the next large allocation), which is precisely the brief's
candidate "st6b carves the same physical pages" — via malloc, with no
arena code at all.
| wrapped + dispatched once, buffer live | |

## 6. st4: pressure vs shape — verdict SHAPE (hand off, do not fix here)

Three-point canonical scaling (cool anchors: 2^25/2^27 trunk close-out,
2^26 W4-U1 journal):

| stage | 2^25 | 2^26 | 2^27 | 25→26 | 26→27 |
|---|---|---|---|---|---|
| st4 | 2.468 | 5.089 | 10.442 | ×2.06 | ×2.05 |
| st5 | 3.048 | 6.109 | 14.653 | ×2.00 | ×2.40 |
| st6a | 0.493 | 0.688 | 2.265 | ×1.40 | ×3.29 |
| st6b | 1.680 | 3.208 | 13.874 | ×1.91 | **×4.33** |

st4 scales at a *constant* ×2.05–2.06 per doubling across the entire range —
no tier cliff whatsoever. Its GPU-utilization halving is explained by
composition: `RegistersRWC::prepare` (4.70 s @2^27) runs at **1.9 cores /
10% CPU — a serial host table build** — and grows linearly, so the zero-GPU
fraction of the stage grows while wall scales linearly. The fix class is
parallelize-the-prepare / port (W4 U3 already rejected the bounded
prototype; CSR rewrite is out of lane scope). **st4 is shape-coupled;
recommended wave-2 port-lane target; no allocator/lifetime change will move
it.** D-lane's fix targets st5-tail/st6a/st6b only.

st6b is the opposite: sub-linear into 2^26, then ×4.33 — all of its excess
(~+7.4 s canonical vs 2×2^26) appears at the last doubling, where the
parked window and the 90 GiB footprint peak also appear.

## 7. Ablation result (runs #2/#3) and fix decision

Back-to-back same-binary 2^27 pair (chrome format, no monitor, quiet box,
one lock window; knob verified live by a census 2^23 run showing the pair
freeing mid-st5 at 4.745 s vs st6b-end when parked):

| stage | park (control) | free-at-retire | delta |
|---|---:|---:|---:|
| st4 | 11.658 | 11.404 | −0.254 |
| st5 | 15.072 | 14.957 | −0.115 |
| st6a | 3.772 | 3.555 | −0.217 |
| st6b | 14.330 | 14.161 | −0.169 |
| st7 | 1.712 | 1.993 | +0.281 |
| st8 | 7.954 | 8.703 | +0.749 |
| **total** | **88.249** | **88.369** | **+0.120 (+0.14%)** |

Peak footprint identical (89.40 vs 89.39 GiB — `free()` of the Vec backing
routes through malloc's large cache as `MADV_FREE`, so the ledger doesn't
move either way; the pages recycle warm into st6b's fresh allocations in
both arms).

**Verdict: H-shape. The parked 30 GiB is performance-neutral at the 2^27
tier on this tree.** The W4 U1 door — "structurally end or decommit stage-5
ownership before the stage-6b adoptions" — is now closed with a measured
null: ending stage-5 ownership at retire moves st6b by −0.17 s (noise), not
the hoped ~−7 s. The 2^26→2^27 st6b excess (~+7.4 s canonical vs 2×2^26) is
intrinsic working-set shape in the CPU members and eq/oracle feeds
(parallel-busy at 11.5 cores, zero-GPU), which is exactly the surface lane
W1B's ports remove. The W3-era T2-vs-T6 ±8 s variance correlated with the
*record family's* death timing (pre-U2 trees carried ~28 GiB more into
st6b), not with the arena pair; U2's lifetime restructuring already banked
that win on trunk.

**Fix decision:** land free-at-retire as the structural default by deleting
the retired-buffer arena outright (`RETIRED` pool, `ArenaSlab`/`ArenaLease`,
`RetiredPoolGuard`, `own_uninit_frs`'s carve path, the IRR session guard).
Rationale: its premise (fresh-page zero-fill worth avoiding) is disproven
end-to-end at 2^27 (+0.12 s total when every adoption goes fresh), malloc's
large cache provides the same warm-page recycling with zero bespoke code,
and the machinery is the proven attractor of failed fixes (U1's madvise,
the poisoning logic) plus a latent reintroduction path for the T2-mode
footprint stack-up if lifetimes shift again. Gate: 2^25/2^26 neutral ±1%,
full matrix, byte-identical proofs.

**Accept-gate honesty:** the lane's −4 s @2^27 target is not reachable
through the allocator/lifetime door — the door was already worth ~0 on
trunk. D's deliverables are the mechanism verdict (this artifact), the
REUSABLE failure pin (§5), the st4 shape hand-off (§6), and the arena
deletion (risk removal + simplification, perf-neutral by construction).

## 8. Deletion validation (post-fix)

- **2^25 interleaved A/B ×3** (park binary vs deletion binary, one lock
  window, alternating): park 23.19/23.23/23.25 vs deletion
  23.17/23.21/23.29 — minima 23.19 vs 23.17 (**−0.09%**, spread ≤ 0.12 s).
  Neutral well inside ±1%. (Absolute walls carry ambient+chrome overhead vs
  the 19.82 cool anchor; both arms identically.)
- **2^27 deletion sanity (run #4):** 91.56 s, vector `[12.376, 12.093,
  5.691, 3.001, 11.534, 15.927, 3.676, 16.277, 2.237, 8.698]`, peak
  footprint 89.36 GiB, zero swap. Carries ~3-4% same-day ambient inflation
  vs the controlled knob A/B window (88.25/88.37 — wave-2 sibling lanes were
  building all day; inflation spreads across CPU-heavy stages, no
  localization at the retire/adoption sites). The controlled same-window
  knob A/B (§7) remains the perf evidence; cool canonical certification is
  the orchestrator's wave-close run.
- **Gate matrix (deletion tree):** jolt-kernels 231/231 (incl. the new
  `own_uninit_frs_wraps_nocopy`; madvise probe stays ignored), jolt-dory
  46/46, byte-diff 11/11 `prover-fixtures` + 11/11 `prover-fixtures,metal`
  (proof bytes identical — hard gate; first host run hit the known
  fixture-build-race flake, clean full rerun), legacy muldiv 3/3 host + 3/3
  host,zk, jolt-witness 34/34, workspace clippy `-D warnings`
  host/host,zk/host,metal, fmt check clean.
- 2^27 budget: 4 of ≈6 used (1 instrumented diagnostic, 2 ablation A/B,
  1 deletion sanity); 2 unspent.
