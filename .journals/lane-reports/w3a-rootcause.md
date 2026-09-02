# W3A root-cause artifact — st6b fat-entry bistability (2026-08-04)

Status: phase-1 complete. Instrumentation committed (be4dd5991: drop-site
tracing; eef0f088e: monitor counter fix — emission never broke, the manual
postprocess step was skipped). Evidence: direct drop-site logs with
backtraces at 2^22/2^26/2^27, per-stage RSS/footprint ledgers, 1 Hz vm_stat
sidecars, the archived bad-mode monitor trace (counters intact), and a
munmap micro-probe extending D §5. All runs today = bad-mode day, box
untouched (no purge/reboot — state preserved through capture).

## 1. Owner chains and drop sites, measured (a)

`JOLT_LIFETIME_TRACE=1` embeds a Drop-logging tag inside every family
allocation; the tag fires at the LAST `Arc` drop with a backtrace naming the
holder. 2^27 census (bad mode, run #1, 86.46 s, chrome-correlated):

| object | GiB | born | last drop | final holder (backtrace) |
|---|---:|---|---|---|
| TraceRecord (12 lanes) | 14.5 | st1 walk | **st4 OPEN** (prepare) | session Arc via `TraceRecord::release`, **0 strong refs outstanding** |
| RegisterLanes | 4.4 | st1 | st4 mid | st4 registers kernel instance |
| SharedOpeningIncrements | 4.0 | st4 | st6b open | opening prefetch fork (moved via `move_to`) |
| RamAccessColumns | 3.0 | st1 | st6b mid | `RamRaVirtualizationKernel::bind` → `RamAddressChunks` |
| PcRows | 1.0 | st1 | st6b mid | `BytecodeDriver` drop in `LazyFoldedRa::ensure_host` |
| SharedInstructionRows | 6.0 | st1 | **proof end** | `ProofSession` drop inside `prover::prove` — the take/re-park pattern never ends; pinned to proof end BY CONSTRUCTION |

Identical drop sites at 2^26 (calm tier) and 2^22. `TraceRecord::shared`
consumers are stages 1–4 only; the st4 release works exactly as designed —
**in the bad mode too**. D §2's "family dies at st6a / st8 by log_T parity"
was a ledger-inference artifact: the ledger tracks reclaim, not drops
(§2 below), and reclaim is what moves.

## 2. Drop-site vs reclaim-timing: the discriminator (b)

2^26 census, same run, drops vs ledger:

```
Rust drops:   TR 7.25 GiB @st4-open, RL 2.19 @st4-mid, SOI 2.0 @st6b-open,
              RAC 1.5 + PcRows 0.5 @st6b        (≈14 GiB dropped by st6b end)
Ledger:       st4 +13.55, st5/6a/6b/7 ALL FLAT, st8 −25.5   ← one cliff at proof end
```

Even at the calm tier, **freed pages do not leave RSS/phys_footprint at
free()** — libmalloc keeps the (huge, vm_allocated) entries resident, and
nothing reclaims them until proof end. At 2^27 the same holds: TR+RL
(18.9 GiB) die at st4 while st4's ledger shows +10.4; the corpse stays
resident through st5 and st6a.

**Verdict on the day-flip: reclaim-timing, not drop-site.** Drop sites are
code-determined and were measured at their designed places on a bad-mode
day; yesterday's good mode cannot have had earlier drops (st4-prepare is
already the design point). What moved across days is when the
allocator/kernel returns the dead pages: yesterday −30 GiB left the
footprint by st6a end with zero compressor activity (D §2 — allocator-side
eviction/deallocation); today the corpses persist into st6b and are
compressed there.

## 3. The st6b wall mechanism, named and quantified (c)

vm_stat 1 Hz during the 2^27 census, st5-tail → st6b window:

```
free pages:     → 0 GiB (repeated full exhaustion; 128 GiB box)
compressions:   sustained bursts 280k–612k pages/s  (≈ 4.5–10 GiB/s)
decompressions: parallel bursts 140k–650k pages/s   ← live-page thrash
compressor pool: 6 GiB ambient → ~32 GiB peak
```

The corpse pile at st6b entry ≈ 31–36 GiB (entry footprint 70.9 minus
live ≈ 35–40, cross-checked against post-purge RSS 31.2 at st6b close):
TR+RL 18.9 (dead since st4) + the IRR scanner ping-pong remnant (30 GiB
freed mid-st5 post-W1D-deletion, partially recycled into st5/st6b live
allocations). st6b then allocates ~+20 GiB fresh (RSS 72→91.4 in 4 s on the
archived monitor timeline) on top → free-page exhaustion → the kernel
compresses the dead pile *and* thrashes live pages (decompression storms),
serializing st6b's own faults behind compressor work. The archived monitor
trace shows the cost shape: gpu% collapses to 0–25 for ~20 s, RSS bleeds
91→39 gradually, and **the 10 Hz monitor thread itself starves for ~8–10 s**
(kernel-level stall). st6b wall: 13.9–14.3 s lean (yesterday, D's ablation)
→ 17.0–18.5 s (trunk, fat) → 24.2–27.4 s (this census / baseline / monitor
run — storm severity varies with ambient occupancy within the same day).

The ledger signature decodes as: st6b RSS −32.7 with footprint FLAT =
compression (compressed pages stay in phys_footprint, leave RSS); st8
footprint −29 = the compressed corpses' vm objects finally deallocated at
proof-end mass-frees. Yesterday's good mode (st6a footprint −30, RSS
−10, zero compressions) = allocator-side vm_deallocate before st6b's burst
— no ceiling contact, no storm.

**Why the flip across days with identical binaries:** the reclaim trigger
is allocator/kernel state, not code. Both trees flipped together overnight
on a 22-day-uptime box; ambient occupancy at run start today ≈ 33 GiB
(file cache 16 GiB, pre-existing compressor pool 6.3 GiB, wired 3.7,
app anon ≈ 5), so prover-peak 90–97 GiB ⇒ ceiling contact today where
yesterday it cleared. The precise libmalloc policy dial (large-cache
admission/eviction vs pressure-notification draining) is the residual
unknown — deliberately not chased further because the fix (§5) removes the
dependence on reclaim timing entirely. A controlled purge/relaunch
experiment can confirm the ambient trigger later; state was preserved
through today's captures.

## 4. Trunk's ~8 s robustness edge over baseline under fat entry (d)

Same-day cert receipts: trunk st6b 17.0–18.5 vs baseline 25.2–27.4; both
arms ≈ 14–17 lean. The storm taxes page-demand, and the arms differ exactly
there: baseline's st6b members are host builds (BRRC prepare+rounds ≈ 7 s
CPU, IncCR prepare 2.6 s, eq/oracle feeds) that fault fresh host pages
continuously inside the storm — each fault waits on the compressor's
free-page pipeline. Trunk's wave-2 content moved the biggest members onto
device buffers (W1B BRRC port; W2A background scans completing pre-st6b)
and W2A emptied st6a, so trunk's fat-mode st6b makes fewer host-page
demands mid-storm. Nothing protocol-level: the edge is page-demand
composition. (The BRRC 2^26-row cap re-hosts BRRC at 2^27 and adds ≈+6 GiB
transient at the peak — capped runs peak 96.9 vs 91.0 uncapped — worth
revisiting by the W3 BRRC lane once entries are lean.)

## 5. Fix design (phase 2) — deterministic early death, munmap-backed

madvise on ever-Metal-wrapped ranges is a proven silent no-op (D §5). New
probe (`munmap_vs_metal_wrap`, committed alongside): **munmap of a dirty
1 GiB anonymous mmap region removes it from phys_footprint in every leg —
never-wrapped, wrapped-then-released, and even live-wrapped (−1024 MiB
each).** mmap-backed allocations are immune to the wrap poisoning.

Plan: an `MmapVec<T: Copy>`-style fixed-capacity container (anonymous mmap,
munmap on drop) as the backing for the corpse-pile members —
1. TraceRecord's 12 lanes + RegisterLanes (18.9 GiB, dies st4): drop =
   immediate footprint release before st5.
2. The IRR scanner ping-pong pair via `own_uninit_frs` (30 GiB, dies
   mid-st5): same backing; probe legs 2–3 cover the Metal wrap.
3. RamAccessColumns / PcRows / SIR / SOI (14 GiB, st6b–st8 by design):
   same container where mechanical; their drops then also become
   ledger-visible at the designed sites.

Expected: st6b entry footprint ≈ live-only ≈ 36–41 GiB (gate ≤45), peak
footprint ≈ 60 GiB (vs 97) — no ceiling contact at any plausible ambient
state, storm structurally impossible, st6b → lean-mode 14–15.5 s, total
≈ 72–73 s. Byte-identical: allocation backing only. Bonus: mmap zero-fill
makes the `vec![0; cycles]` memsets in the st1 walk kernel-side no-ops.

Risks watched at the gate: fresh-page fault cost where malloc reuse
previously recycled warm pages (D measured 8+ GiB/s zero-fill; the IRR pair
alloc mid-st5 is the biggest single consumer), and st1 collect wall (same
first-touch faults as today's fresh vm_allocate — expect neutral).

## 6. Run/budget ledger

2^27: 1 of 8 used (census r1, 86.46 s — walls not anchor-quality, bad-mode
variance). 2^26 census 35.60 s. 2^22/2^18 smoke. Artifacts:
/tmp/w3a-census-2to2{6,7}-r1.log, /tmp/w3a-vmstat-2to2{6,7}*.log,
benchmark-runs/perfetto_traces/modular_sha2_chain_2{6,7}_metal.json,
/tmp/w3a-lifetime-2to22.log. Monitor-fix e2e receipt:
modular_sha2_chain_18_metal.json (112 ph:'C' counters, auto-converted).

## 7. Phase-2 results (fix landed; wall gate deferred to cool cert)

Correction to §1: SharedOpeningIncrements' tag fires when the struct is
consumed into `OpeningColumns` at st6b open (prefetch); the increment DATA
lives on inside the opening views to st8 — live by design, not corpse.

**Fix shipped (7cb173075 + 16c228ca2):** `MmapVec` (fixed-capacity
anonymous mmap, munmap on drop) backs the TraceRecord lanes, RegisterLanes,
RamAccessColumns, SharedInstructionRows, PcRows, and `own_uninit_frs`'
device ping-pong (`OwnedBacking::Mmap`). Every corpse-pile member now
leaves RSS+footprint at its designed drop site. Second negative result
pinned alongside W1D's madvise: `malloc_zone_pressure_relief` is a
measured no-op (probe: 4 GiB freed Vecs stay resident, returns 0) — that
is WHY freed-Vec corpses ride forward at all; an experimental relief call
at the st6b boundary changed nothing at 2^27 and was removed.

**Matched-pair 2^27 A/B (one lock window, pre-fix ran first/cooler):**
pre-fix 90.40 s (peak footprint 96.91, st6b entry 70.91) vs post-fix
**79.91 s (peak 79.28, entry 54.29) = −10.5 s (−11.6%)**. Same-day
post-fix runs: 79.15 / 76.77 / 79.91 with st6b 17.3 / 16.8 / 16.9 vs
same-afternoon pre-fix 86.46 / 90.40. The afternoon ambient deepened the
bad mode (dawn cert pre-fix: 76.4–77.1) — the bistability in action.

**Residual storm, decomposed:** with corpses gone, st6b still shows
RSS −19/−20 vs footprint −1.8 = ~15 GiB compressed mid-stage: cold-LIVE
pages (the ~20 GiB in-heap trace + late-stage carries) squeezed under
st6b's +25 GiB burst (IncCR cur 16 + capped-BRRC CPU tables + concurrent
st8-prefetch OpeningColumns 8) when ambient is tight. Not reachable by
lifetime/drop-site work: the remaining mass is live (trace/witness plane,
IncCR/BRRC slot allocations — other lanes' surfaces).

**Gate assessment (honest):** 2^25 neutral −0.4% ✓; byte-diff 11+9/20
both arms ✓ (one known fixture-race flake, two clean consecutive full
passes); kernels 239/239 + 158/158, dory 46/46, muldiv 3/3+3/3, clippy
host/zk/metal, fmt ✓. The 2^27 wall gate (total ≤74, st6b ≤15.5, entry
≤45) is NOT certifiable under today's ambient: entry 54.3 is now
live-only (the ≤45 target was calibrated on pre-wave-2 live composition
— wave-2 carries +13 GiB more live at entry), and totals land 76.8–79.9
in the deepened afternoon mode. Recommendation: re-base the entry gate to
"zero dead pages at st6b entry" (achieved, receipts above) and take walls
in the orchestrator's cool certification window — 3 of 8 2^27 runs left
banked for it. Expected there: the fix's storm-avoidance margin (+18 GiB)
puts trunk at ~72–74 with st6b ~14–15.5.

Artifacts: /tmp/w3a-{census,vmstat}-2to27-{r1,mmap,relief}.log,
/tmp/w3a-ab-{prefix,postfix}.log, binaries /tmp/w3a-bin-{prefix,postfix}.
