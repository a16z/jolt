# Metal-sat lane: address-major layout scope (Phase 1 — design/attribution)

Lane: `scratch/metal-sat-address`. No cargo, no benches run by this lane (velocity v3).
All file:line refs against this worktree @88b063db3. Fr = 32 B throughout.

## Verdict (summary)

| # | candidate | verdict | prize @2^27 | why |
|---|---|---|---:|---|
| C1 | Dory committed-order flip (`TracePolynomialOrder::AddressMajor`) | **NO-GO** | ≤0 | measured >68× regression @2^22 (orchestrator: 3.52 s cycle-major vs >240 s terminated); mechanism = reference-tier fallback; even a native rewrite targets the two healthiest stages and inverts sharding |
| C2 | st6b member-table relayout / address-major streams | **NO-GO** (vacuous) | 0 | no address axis remains in st6b — address vars bound at 6a; tables are 1-D cycle-domain by construction; streaming is ≤2% of stage wall |
| C3 | st4 RegistersRW **address-first bind order** + fixed-slot layout | **PROBE** | −1.5..−2.5 s | converts the sole unfused slot into an IncCR-shaped dense fused slot; protocol change, probe discriminates the one unknown (address-phase cost) for ~0.5 day, no protocol wiring |
| C4 | RAM st2 rw_matrix order flip | NO-GO | ~0 | already flips to address-major for free at the phase boundary (O(accesses) map, no sort); st2 healthy at 2.7 s |
| C5 | Field-element limb interleave (SoA limbs) | NO-GO | 0 | 32 B flat AoS is already coalesced for every existing access shape |
| C6 | One-hot u8 stream relayout | NO-GO (vacuous) | 0 | nothing is materialized to relayout — hot indices are computed on the fly from lanes |

**No address-major GO exists at current evidence.** One probe (C3) is worth a
half-day to discriminate; everything else closes with mechanism. Deep reason,
one line: **layout follows bind order, and Jolt-on-Metal is a cycle-major
machine because every hot sumcheck phase has only cycle variables unbound;
the address side is either tiny (≤8 KB chunk tables, 128-entry register
file) or crossed exactly once per proof via already-hoisted pushforwards.**

## 0. Empirical anchor (orchestrator, 2026-08-04)

`modular_benchmark` @2^22: cycle-major **3.52 s**; `TracePolynomialOrder::
AddressMajor` override **>240 s, terminated** (>68× lower-bound regression).
Mechanism from code, not mystery:

- Optimized streaming commit requires cycle-major windows — non-cycle-major
  routes to the reference materializing path
  (`optimized/commitment.rs:61-65` → `reference/commitment.rs:320-388`
  `MaterializedColumn`, which walks the full widened grid per committed poly
  with none of the one-hot sparsity exploited).
- Metal commit gates on `order == CycleMajor && row_width <= cycles`
  (`metal/commitment.rs:209-217`); Metal joint-opening fold is cycle-major
  only (`metal/slots/joint_opening.rs:20-32`).

So the >68× is the *fallback-tier tax*, a lower bound for the axis as wired
— not a measurement of a hypothetical native address-major backend. §2 (C1)
argues the native version is also worthless. The axis itself stays (it is a
wire/FS parameter with test coverage); it just must never be derived.

## 1. Current layout map (code facts)

### 1a. Variable & index order

- Committed ra grid index = `k·T + j` — **address bits HIGH, cycle bits
  LOW** (`TracePolynomialOrder::CycleMajor`,
  `jolt-claims/.../geometry/dimensions.rs:19-58`); production always derives
  CycleMajor (`jolt-prover/src/config.rs:95`). `AddressMajor` (= `j·K + k`)
  exists as a wire axis, absorbed into FS as `b"dory_layout"`
  (`jolt-verifier/src/verifier.rs:622-626`), never derived.
- One-hot representation: **per-cycle hot index, `Option<u8>`, length T**
  (`jolt-poly/src/one_hot.rs:24-29`; legacy twin
  `one_hot_polynomial.rs:24-35`). Never a K×T grid anywhere in the
  optimized/metal tiers (dense grids exist only in the reference tier and
  tests). `k_chunk = 16` below 2^25, `256` at/above
  (`ONEHOT_CHUNK_THRESHOLD_LOG_T = 25`, `common/src/constants.rs:13`);
  `instruction_d = 128/log_k_chunk` → 32 @2^24, 16 @2^25+.
- Trace-derived SoA lanes, all cycle-indexed `MmapVec`s (position t = cycle
  t): TraceRecord 116 B/c, RegisterLanes 35 B/c, RamAccessColumns 24 B/c,
  SharedInstructionRows 48 B/c (repr(C), asserted), PcRows 8 B/c
  (`optimized/trace_record.rs:96-116`, `optimized/ram_trace.rs:32-42`,
  `optimized/instruction_read_raf.rs:85-102`,
  `optimized/bytecode_read_raf.rs:80-85`).

### 1b. Metal access patterns (all hot slots)

- Field elements: BN254 Fr = flat AoS 32 B (8×u32 LE), zero host↔device
  conversion (`metal/field.rs:19-26`); `fr_load/store` walk
  `p[idx·FR_LIMBS + i]`. Adjacent threads touch adjacent 32-B rows —
  element-stride-1, coalesced; bind kernels read 128-B quads/thread.
- Every round kernel is a **stride-1 cycle-domain ping-pong**
  (`jk_round_pair`, `shaders/kernels.metal:142-164`; `RoundTable{cur,nxt}`
  `slots/mod.rs:79-119`): binding round reads `len`, writes `len/2`, swap.
  Bind is fused into the eval dispatch (challenge delivered as
  `pending_bind`, `jolt-sumcheck/src/prover.rs:232-326`).
- The only gathers on device index **chunk-sized eq/branch tables**:
  `width·2^cb·32 B ≤ 8·256·32 = 64 KB` (bytecode `jk_bytecode_gather`,
  RAV `jk_ra_gather`, IRR `v_tables[phase·256+chunk8]`) — L2/SLC-resident,
  random-within-256, never K-wide, never T-wide.
- **No scatter, no sort, no transposition exists on device.** The only
  inter-round reshapes are the one-time third-bind adopt/materialize gathers
  into flat factor-major pairs (pure gather maps, one thread/output).
- Unified memory throughout: `storageModeShared`, zero-copy
  `newBufferWithBytesNoCopy` page wraps, mmap-backed pongs munmapped on drop
  (W3A). No threadgroup-memory data tiling anywhere; threadgroup memory is
  reduction scratch only (limb-major, bank-conflict-free, `fr.metal:55-69`).

### 1c. Dory commitment/opening consumption

- Commit streams **contiguous cycle windows** (tier-1 chunk = `row_width`
  cycles); one-hot rows commit as **sums of SRS generators at hot cycle
  positions** bucketed by k — zero scalar muls
  (`jolt-dory/src/streaming.rs:400-449`); tier-2 row = `k·windows + window`
  (`streaming.rs:366-396`; Metal `(column,window,k)` segment map
  `metal/commitment.rs:279-285,1151`).
- Openings: stage-8 point order **matches committed order by construction**
  (`committed_openings.rs:147-198` has both arms); sparse one-hot fold =
  per-cycle scatter into a `2^⌈n/2⌉`-entry accumulator (4–8 MB, SLC-resident,
  `optimized/opening.rs:491-551`).
- st0 and st8 are the campaign's two healthiest stages (GPU ratio-to-healthy
  1.0 in the final attribution). **There is no commit/opening prize.**

### 1d. Cycle→address transposition tax (complete ledger)

~6 O(T) pushforward walks/proof + 2 O(entries) representation flips:

| site | when | size out | status |
|---|---|---|---|
| booleanity `cycle_pushforward` | 6a | N × k_chunk | backgrounded (W2A R1 anchor) |
| bytecode `stage_pushforwards` ×2 walks | 6a | 5 × 2^log_k_bc | 4 points backgrounded (R2), 1 inline |
| HWCR `build_hamming_weight_tables` | st7 | N × k_chunk | inline, point forced (no early anchor exists — w2a.md §R3) |
| RAM RAF `fold_cycles` | st2 | ram_K | inline, ~cheap |
| `reconstruct_val_init` | st2 | ram_K | inline, one T-walk |
| rw_matrix `into_address_major` | st2 | O(accesses) | free flip (rows already collapsed; no sort) |
| registers K-scatter | st4 end | 3×128 | O(remaining entries) |

Residual critical-path tax ≈ st7's walk (1.2–1.8 s @2^27) + inline
fallbacks. **An address-major primary layout would invert this tax**: every
cycle-phase sumcheck (the bulk of the prover — log_T rounds vs log_K rounds,
27 vs 7–8) would need an address→cycle transposition instead. The current
direction is the cheap one by a factor ≈ (cycle-phase work)/(address-phase
work) ≫ 1.

## 2. Byte quantification (from code dimensions)

T-scale structures (GB, decimal; B/cycle × T):

| structure | B/c | 2^24 | 2^25 | 2^27 |
|---|---:|---:|---:|---:|
| TraceRecord lanes | 116 | 1.95 | 3.89 | 15.6 |
| RegisterLanes | 35 | 0.59 | 1.17 | 4.70 |
| RamAccessColumns | 24 | 0.40 | 0.81 | 3.22 |
| SharedInstructionRows | 48 | 0.81 | 1.61 | 6.44 |
| SpartanOuter pair (Az‖Bz, 6T Fr) | 192 | 3.22 | 6.44 | 25.8 |
| IRR cycle pair (7.5T Fr, factors=5) | 240 | 4.03 | 8.05 | 32.2 |
| InstrInput pair (6T Fr, 8 tables @T/2) | 192 | 3.22 | 6.44 | 25.8 |
| IncCR tables (4 pairs, 6T Fr) | 192 | 3.22 | 6.44 | 25.8 |

Full-phase streamed bytes for a dense fused n-table member ≈ `128·n·T`
(read T + read/write 3T over the halving cascade):

- IncCR (n=4): 8.6 / 17.2 / **68.7 GB** — measured rounds 3.17 s @2^27
  ⇒ effective ~22 GB/s end-to-end, i.e. **~4% of the ~0.5 TB/s roofline**.
- RamHB (n=1): 2.1 / 4.3 / 17.2 GB.
- st6b aggregate device streams (inc + hamming + BRRC + 2×RAV dense tails +
  48 B-row gather traffic) ≈ **170–190 GB @2^27 ≈ 0.31–0.35 s at roofline**
  vs **16.3 s measured stage wall**.

**Conclusion the whole lane hangs on: st6b (and st4/st3's low ratios) are
not table-streaming-bandwidth-bound.** ≤2% of st6b's wall is explainable as
DRAM streaming of its tables. The residual is gather ALU (Montgomery mults),
per-round dispatch+wait serialization, shrinking-tail occupancy, host glue,
and CPU-member overlap. Relayouting the streamed tables — by address or any
other key — moves none of those costs. This kills the generic "address-major
for bandwidth/SLC locality" hypothesis at the root: the bytes are already
minimal, contiguous, and coalesced; there is no strided or scattered T-scale
DRAM traffic anywhere in the hot path to fix (§1b).

## 3. Candidates in detail

### C1 — Dory committed-order flip: NO-GO

What it would change: ra grid index `k·T+j → j·K+k`; commit tier-1 rows
become high-cycle-bit windows with (low-cycle, address) columns; stage-8
point arm flips (already wired); Metal commit/opening fast paths would need
full rewrites (currently hard-gated CycleMajor).

- **Gain: none.** st0/st8 healthy (ratio 1.0). The one-hot generator-sum
  trick survives either order (one hot cell per cycle either way), so
  address-major cannot even beat cycle-major at its own game — at best it
  ties commit throughput after a full rewrite of
  `streaming.rs`/`metal/commitment.rs`/`joint_opening.rs`.
- **Sharding inverts (the structural kill).** Cycle-major tier-1 shards =
  contiguous trace segments: streamable during the trace walk, natural for
  multi-device/multi-pass, tier-2 aggregates per k. Address-major shards =
  address slices, and every slice must scan the **full** trace (any cycle's
  hot address can land in any slice) — shard count multiplies T-passes;
  anti-streaming. This also blocks the future shard-by-trace-chunk door.
- **Empirical:** §0 — >68× as wired.
- Soundness: n/a (axis already sound, fail-closed, FS-absorbed). Cost saved.

### C2 — st6b/st5 member relayout: NO-GO (vacuous axis)

st6b members are the **cycle phases** of relations whose address variables
were bound in 6a (booleanity cycle, BRRC, RAV×2, RamHB, IncCR). Their device
state is 1-D over cycles because those are the unbound variables; "address-
major" is not expressible. The chunk-table gathers are ≤64 KB and cache-
resident. §2 shows streaming ≈ 2% of wall. What st6b actually needs — fewer
waits, cross-member fusion, occupancy at tails — is parked door #3 and the
**fusion lane's** territory, not a layout axis. Same logic covers st5 IRR
(16 address phases already run device-side against 256-entry buckets) and
st3 InstrInput (dense table-major pair; its 17 GB round-1 write is ~60 ms of
roofline traffic — the stage cost is elsewhere).

### C3 — st4 address-first bind order: PROBE (the one justified design)

**Today** (`optimized/registers_read_write.rs`,
`metal/slots/registers_read_write.rs`): cycle-major sparse CSR, ≤3 entries/
cycle (56 B idx / 120 B field entries), binds **27 cycle rounds first** with
per-round merge→count→host-scan→alloc→scatter (2 command buffers + 2 waits +
host boundary per round — the campaign's sole unfused slot), then collapses
to K=128 dense and runs 7 cheap address rounds
(`registers_read_write.rs:1302-1333,1205-1238`). Measured @2^27: prepare
2.0 s (W3C), device prefix 5.862 s at 30.5% GPU-eq, st4 total 8.08–8.3 s.
W2B died here twice: fixed segments grow as rows merge (cols union ⇒
`min(3·2^r, K)` width ⇒ +52 GiB), and in-place sparse device rounds are
+46.8% slow. **Both deaths are artifacts of binding cycle first.**

**Flip: bind the 7 address variables first, then 27 cycle variables.**

- Address rounds (0–6): rows stay = cycles, so segments are **fixed 3-wide
  forever** — no merging across cycles, no count/scan/alloc, offsets are
  literally `3j`. Pairing is within-cycle (col k with k+2^b). The sibling
  problem (Val is dense in k) is solved by a **running bound register file**:
  128 entries → 64 → … folding with each challenge, updated per cycle by
  `file[rd] += inc` weighted by the partial-eq of rd's bound bits; L1-
  resident (≤4 KB), chunk-checkpointable (checkpoint = file snapshot per
  chunk, 128×32 B) for full parallelism. Each address round = one O(T) pass
  over RegisterLanes (35 B/c, already resident — **no new T-scale
  allocation**): 4.7 GB/pass @2^27, 7 passes ≈ 33 GB ≈ CPU-streamable in
  0.5–1.0 s on 12 cores, or a trivially fused device kernel later.
- After round 7 the member is **3–4 dense T-length tables** (ra-bound A,
  wa-bound W, val-bound V, inc) → 27 stride-1 fused rounds, byte-for-byte
  the IncCR shape (n=4 ⇒ 512·T B total, measured analogue **3.17 s @2^27**).
- **Modeled win @2^27: rounds 5.86 → ~3.7–4.2 s ⇒ st4 −1.5..−2.5 s**
  (8.3 → ~5.8–6.8). @2^25 ≈ −0.2..−0.4 s; @2^24 ≈ −0.10..−0.20 s (st4
  −20..−30%, usable as the kill gate). Prepare also simplifies: fixed slots
  need no sort and no exact scan — pure parallel scatter.
- **Memory: neutral-to-better.** Dense tables 4×T×32 = 17.2 GB @2^27 vs
  current sparse entries ≈ 2.5T×56 B ≈ 18.8 GB; no K-growing segments.
- **Batch mechanics:** stage-4 batch = {RegistersRW (log_T+7 vars),
  RamValCheck} (`jolt-verifier/src/stages/stage4/verify.rs:128-129`). The
  batch driver already supports offset-windowed members — head-aligned
  prefix vs tail-aligned suffix (`jolt-sumcheck/src/prover.rs:39-42,245`).
  RegRW becomes head-aligned (rounds 0..34); RamValCheck's window shifts to
  rounds 7..34, sharing the cycle challenges. **Must-confirm in code before
  build:** RamValCheck and downstream point consumers tolerate the shifted
  window (they take explicit point slices, so expected yes — verify).

**Soundness (protocol change, required argument):** binding order over the
variables of a fixed multilinear claim is a verifier-known permutation of
sumcheck rounds. Completeness, per-round degree bounds, and the soundness
error Σᵢ degᵢ/|F| are order-independent — the verifier checks the same
`g_r(0)+g_r(1) = claim` chain over the same variable set. Fiat–Shamir: every
round message is still absorbed before the challenge binding it is drawn; no
draw moves relative to its message; the challenge→variable assignment is a
fixed public function of the (proof-self-described) axis. Downstream
openings (val/ra/wa at (r_addr‖r_cycle), consumed by registers val-eval and
claim reductions) source their point components from rounds 0..6 ∪ 7..34
instead of 7..34 ∪ 0..6 — explicit slices, no aliasing. Axis follows the
BooleanityAnchor V1 pattern: `JoltProtocolConfig` gains
`registers_bind_order: {CycleFirstLegacy, AddressFirstV1}`; legacy + zk pin
`CycleFirstLegacy` (BlindFold stage-config constraints untouched under zk);
verifier validates fail-closed **before** stage work; V1+BlindFold rejected.

**e2e/tamper gate (protocol-visible):** (a) e2e prove+verify @2^22 and 2^24,
both backends, under V1; (b) tamper: V1 proof re-tagged legacy → clean
`ProtocolConfigMismatch` (and vice versa); mutated st4 round message →
reject; (c) byte-diff twins stay green with legacy pinned; (d) full
integrated suite once, at wave close, per velocity rules.

**Cost/risk:** 3–5 days full build (prepare rework, address-round algorithm,
dense slot reusing existing fused kernels, verifier + axis + tests). Risk
moderate-high: W2B graveyard adjacency, the batch-window must-confirm,
BlindFold surface (mitigated by pinning legacy under zk). Hence: **probe
first, protocol untouched.**

### The minimal probe (discriminates without a rewrite)

The dense-phase side is already measured (IncCR 3.17 s @2^27 is the exact
shape). The **only unknown is the address-phase cost** (7 running-file
passes + fixed-slot pair folds). Probe:

- ~150–200 LOC standalone (ignored test or bin): consume `RegisterLanes`
  as-is, run the 7 address rounds with the running-file algorithm
  (chunk-checkpointed, rayon), produce the collapsed A/W/V/inc tables and
  round messages. Correctness oracle: feed the collapsed tables into a plain
  dense sumcheck for the remaining 27 rounds and compare the **final claim**
  against the existing prover's st4 output claim on the same small fixture
  (2^12–2^16) — same total sum ⇒ algorithm right, no protocol wiring at all.
- Timing (deferred to a cargo-permitted phase, ONE pair @2^24 under the
  bench lock): **kill if the 7-pass address phase > 0.15 s @2^24 parallel**
  (that scales to >1.2 s @2^27 and eats the fused-round savings; the net
  must clear ≈ −1.5 s @2^27 to beat the protocol-neutral alternative).
- Zero soundness exposure, zero slot/verifier changes, half a day.

## 4. Lane boundaries & sequencing (for orchestrator)

- **Fusion-lane overlap:** parked door #2 (st4 round-loop fusion under a
  memory-viable representation, "the middle is unexplored") is the
  protocol-NEUTRAL route to the same 5.9 s wall; C3 is the protocol-changing
  route that makes fusion trivial by construction. Sequence: if the fusion
  lane lands the middle first, re-measure st4 residual before spending C3's
  protocol change; if it doesn't, C3's probe result decides. Do not run both
  builds concurrently — same files (`registers_read_write.rs` both tiers).
- st6b tiling/fusion (parked door #3) and st0 walk↔commit contention
  (parked door #1): out of this lane's scope, no address-major angle found.
- The `TracePolynomialOrder::AddressMajor` axis: keep as wire/test artifact;
  never derive; no code change proposed.

## 5. Negative results to pin in the campaign journal

11. Dory committed-order flip (AddressMajor axis as wired): >68× @2^22
    (3.52 s vs >240 s, terminated) — reference-tier fallback by design;
    native rewrite has no prize (st0/st8 healthy) and inverts sharding from
    trace-streamable to full-trace-per-shard. Door closed.
12. Address-major relayout of cycle-phase sumcheck state: vacuous — after
    6a, no address variable is unbound anywhere in st6b; layout follows bind
    order. (Generalizes journal negative result #5 beyond booleanity, on the
    layout side rather than the bind-order side.)
13. Hot-path DRAM traffic audit @2^27: all T-scale device streams are
    stride-1 coalesced ping-pongs totaling ~0.3 s of roofline inside a
    16.3 s stage — bandwidth/locality is not the st6b limiter; no strided or
    scattered T-scale access exists to fix. (Numbers in §2, from code dims +
    certified walls; no new benches run.)

— lane metal-sat-address, phase 1, 2026-08-04
