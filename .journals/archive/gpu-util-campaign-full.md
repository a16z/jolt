# Metal M5 Max GPU-Utilization Campaign

Mandate (2026-08-03): the 2^27 trace shows literal 0% GPU on big portions of the
sumcheck stages. Get GPU utilization high and the GPU utilized EVERYWHERE on the
Metal backend. ~~Hard gate: proof bytes identical (byte_diff / sha A/B) — pure
engineering, no protocol changes.~~

**DIRECTIVE UPDATE (2026-08-04 00:20 EDT, supersedes byte-parity):** byte-identical
proof compatibility NO LONGER required. Anything SOUND may be tried — layout
changes (address-major), two sumcheck rounds at once (extra compute for fewer
challenge round-trips), protocol restructuring. Remaining constraints:
1. Proof must verify end-to-end (prover+verifier may change together).
2. Soundness preserved — journal a short soundness argument for any
   protocol-touching change.
3. Full test matrix stays green (protocol-touching changes update fixtures/tests
   rather than delete them; byte-diff fixture tests are superseded by e2e
   prove+verify equivalents where the format legitimately changed).
Gate shifts from byte-identical → **e2e verify + full tests**. In-flight wave-1
lanes finish as scoped (their ports are exact-math; byte parity is not a perf
constraint for pure ports since field arithmetic is associative — the freedom
matters for restructuring, i.e. follow-on lanes).

Predecessor: M5 Max campaign (closed) — `.metal-m5-box-journal.md`. Retained
ceiling there: **2^25 = 19.822 s / 1.693 MHz** (two-run mean), **2^27 = 77.168 s
/ 1.739 MHz**, 3.196× vs same-binary CPU arm @2^25.

## Lineage

- Campaign trunk: `scratch/gpu-util-trace` @ `1c6bbff4e` = `gpu/metal-backend`
  (`042b2c7ab`, W4 merge + journal) + GPU-counter MetricsMonitor commit.
  Worktree: `~/dev/jolt/.worktrees/metal-gputrace`. Never push.
- Lane branches: `gpu/util-w1{a,b,d}` in worktrees
  `~/dev/jolt/.worktrees/gpuutil-w1{a,b,d}`.
- Instrumentation: `jolt-profiling` feature `monitor` now samples
  `gpu_percent` (ioreg) as a Perfetto counter track.

## Baseline attribution (traces: /tmp/metal-m5-gputrace-2to{25,27}-20260803.json.gz)

Instrumented runs (monitor feature on; walls are ~10-16% above canonical — use
for attribution only, never as wall anchors).

### 2^27 per-stage (instrumented prove ≈ 89.9 s; canonical 77.17 s)

| stage | wall | gpu_avg | dominant CPU spans |
|---|---:|---:|---|
| st0 | 11.04 | 91.8% | healthy |
| st1 | 10.23 | 30.8% | SpartanOuterUniskip::prepare 5.94 (TraceRecord::collect 3.94) — 4.2 s zero-hole |
| st2 | 5.12 | 27.2% | tail feeds st3 zero-hole (4.3 s spanning st2→st3) |
| st3 | 3.09 | 21.7% | prove_batch CPU members |
| st4 | 13.25 | 18.1% | RegistersRWC::prepare 4.70 (zero-hole) + rounds 6.74 @ low util; RamValCheck rounds 1.44 (unported slot) |
| st5 | 17.74 | 62.3% | late InstructionReadRaf rounds 2.5 s zero-hole (CPU tail) |
| st6a | 3.78 | **0.0%** | BooleanityAddressPhase::prepare 2.88 + BytecodeReadRafAddressPhase::prepare 0.89 |
| st6b | 15.76 | 25.1% | BytecodeReadRafCycle prepare 2.24 + rounds 4.82 (CPU member); IncClaimReduction::prepare 2.58; EqPolynomial::evals 2.21; oracle_table 1.63 |
| st7 | 1.87 | **0.0%** | HammingWeightClaimReduction::prepare 1.86 (CPU table build; device rounds negligible) |
| st8 | 7.96 | 86.1% | healthy |

Zero-GPU windows >1.5 s during prove: 4.20 (st1 prepare), 4.34 (st2 tail→st3),
4.81 (st4 prepare), 1.63 (st4 round 1), 2.47 (st5 late rounds),
**9.16 (st5-end→st6b mid: st6a entire + st6b prepare prefix)**, 2.17 (st7).
Sum ≈ 28.8 s ≈ 32% of instrumented prove wall at literal 0%.

### 2^25 per-stage (instrumented prove ≈ 20.6 s; canonical 19.82 s)

st0 93%, st1 58%, st2 44%, st3 28%, st4 41%, st5 82%, **st6a 3.5%**, st6b 54%,
**st7 0%**, st8 92%. Only zero-windows >0.5 s: two ≈0.5-0.9 s. The crisis is
2^27-specific (≈90 GiB pressure tier) except st6a/st7 which are 0% at every scale.

## Slot registry status (crates/jolt-kernels/src/metal/mod.rs `metal()`)

Metal-installed: commit, spartan outer/product (uniskip+remainder), ram_read_write,
registers_read_write, instruction_{claim_reduction,input,read_raf,ra_virtualization},
ram_{raf_evaluation,hamming_booleanity,ra_virtualization}, inc_claim_reduction,
hamming_weight_claim_reduction, booleanity_cycle, joint_opening.

CPU-only (optimized fallback): **booleanity_address**, **bytecode_read_raf_address**,
**bytecode_read_raf_cycle**, ram_output_check, spartan_shift (W2 no-go),
registers_claim_reduction, ram_val_check, ram_ra_claim_reduction,
registers_val_evaluation, advice_opening, trusted/untrusted advice (cycle+address),
bytecode_reduction (cycle+address), program_image_reduction (cycle+address).

Installed-but-CPU-prepare: hamming_weight_claim_reduction (build_hamming_weight_tables),
inc_claim_reduction (prepare inflates under 2^27 page pressure — W4 U1 mechanism).

## Wave 1 lane cut

Priority per mandate: (1) st6a+st7 ports, (2) st5→st6b seam 9.16 s, (3) st3 feed,
(4) st4/st6b 2^27 degradation. Lanes A+B jointly cover (1)+(2); D covers (4);
(3) deferred to wave 2 (1.97 s @2^27, smallest prize).

| lane | scope | prize @2^27 (instr.) | model | branch |
|---|---|---:|---|---|
| A | st6a+st7 ports: Metal slots for booleanity_address, bytecode_read_raf_address; device path for HWCR table build | ~5.6 s of 0%-GPU wall | codex gpt-5.6-sol-xhigh | gpu/util-w1a |
| B | st6b CPU members: bytecode_read_raf_cycle full port (prepare+rounds 7.06 s); inc_claim_reduction prepare device path (2.58 s); eq-evals/oracle_table feed as encountered | ~9-11 s | codex gpt-5.6-sol-xhigh | gpu/util-w1b |
| D | 2^27 pressure tier root-cause+fix: stage-5 arena ownership → st6b prepare inflation (parked W4 U1 door: "structurally end or decommit stage-5 ownership before stage-6b adoptions"); st4 degradation (41%→18% util, prepare hole 4.7 s) | ~6-9 s | claude fable-max | gpu/util-w1d |

Non-overlap contract: A owns st6a+st7 slot files; B owns st6b member slots; D owns
allocator/arena/lifetime + st4 (registers_read_write internals). B and D both touch
IncClaimReduction::prepare — B owns its device port, D treats it only as a
pressure symptom (no code edits to that slot without coordinating through
orchestrator).

## Protocol (all lanes)

- Build: `cargo build --release -p jolt-prover --example modular_benchmark --features prover-fixtures,metal -q --message-format=short`
- Timed run: `cargo run --release -p jolt-prover --example modular_benchmark --features prover-fixtures,metal -- --name sha2-chain --scale N --format chrome --backend metal` (cwd = worktree root; trace lands in `benchmark-runs/perfetto_traces/`)
- GPU-util attribution run: add `-F jolt-profiling/monitor`. Never quote monitor-run walls as anchors.
- CPU ablation arm: `JOLT_METAL_DISABLE=1` prefix, same binary.
- Bench lock for ANY timed run: `while ! mkdir /tmp/jolt-gpu.lock.d 2>/dev/null; do sleep 15; done; echo "$LANE $$" > /tmp/jolt-gpu.lock.d/owner` … release `rm -rf /tmp/jolt-gpu.lock.d`. Builds don't need the lock. Check owner file before force-clearing anything stale (>30 min with no jolt process running).
- Iterate at 2^22-24; confirm 2^25 (cool: ≥3 min quiet, AC, `pmset -g batt` shows AC). 2^27 runs (~90 GiB, ~4 min + 2.5 min fixture gen): lane D only for diagnosis, one at a time, under lock; wave-close certification is orchestrator-run.
- Memory discipline: never start a 2^27 while another lane holds >20 GiB; no swap storms (check `sysctl vm.swapusage` before/after).
- Gate matrix before any retained commit: jolt-kernels 231/231 (+ your new tests), jolt-dory 46/46, jolt-prover byte-diff 19/19 in BOTH `prover-fixtures` and `prover-fixtures,metal`, legacy muldiv 3/3 `host` + 3/3 `host,zk`, clippy `-D warnings` host / host,zk / metal, fmt. Proof bytes identical is the hard gate — a Metal slot must produce byte-identical proofs vs CPU path (parity tests force device thresholds low and require positive dispatch count; follow existing slot test patterns).
- Journal: lane report at `.journals/lane-reports/w1<x>.md` (own worktree), committed. Report to orchestrator via message_parent at: (i) root-cause/decomposition done, (ii) parity green with first A/B numbers, (iii) final, or any hard blocker.

## Kill gates

- A: at 2^24 with parity green, combined st6a+st7 wall must drop ≥35% vs same-tree baseline; else report and stop (don't polish a dead port).
- B: bytecode_read_raf_cycle port at 2^24: st6b wall −15% or member-attributed −40%; inc prepare device path judged at 2^25.
- D: needs a written root-cause artifact BEFORE any fix attempt (which allocations, which stage owns them, page-fault evidence). Fix gate at 2^27: st4+st6b combined −4 s vs canonical 24.32 s with 2^25 neutral (±1%).

## Canonical anchors (from M5 close-out, binary b9799347e…)

- 2^25 Metal cool: 19.822 s mean — vector [4.539, 1.757, 0.991, 0.412, 2.468, 3.048, 0.493, 1.680, 0.247, 4.219] (run 1)
- 2^27 Metal: 77.168 s — vector [10.766, 7.997, 4.343, 1.970, 10.442, 14.653, 2.265, 13.874, 2.072, 8.788]
- 2^25 CPU arm: 63.354 s

Honest wave-1 projection if all three lanes hit: 2^27 ≈ 63-67 s (2.0-2.1 MHz),
2^25 ≈ 17.5-18.5 s (1.81-1.90 MHz). Above 2 MHz @2^27 requires D to land, not
just A+B.

## Wave-1 gate results

- **W1A (task 2bbe078e): KILLED at the 2^24 gate, cleanly.** Exact-math device
  prepares for st6a+st7 regressed the targeted span +26.2% (needed ≥−35%).
  Diagnosis: SIMD bucket shape leaves lanes inactive scanning every inner-eq
  row (work inflation scales with T, not fixed overhead — kill is scale-honest);
  bytecode counting-sort adds two full trace passes. Forced-device parity was
  3/3 green — correctness fine, shape wrong. Prototype reverted; only report
  retained (lane-reports/w1a.md in gpuutil-w1a, commits 2dd709f38+80f69a3e0).
  Artifacts /tmp/w1a-*.{log,json}. CONSEQUENCE: priority-1 target (st6a/st7 0%
  GPU) does NOT fall to a like-for-like port → new-freedom rethink lane W2A.
- **W1B (task 83cf4e87): checkpoint-2 kill gate PASSED.** BytecodeReadRafCycle
  full port: st6b −42.9% @2^24 (2.102→1.200 s), member −53.3%, whole prove
  −8.7% (11.94→10.90 s). Parity: forced-device lockstep green; byte_diff 11/11
  in both prover-fixtures and prover-fixtures,metal (brief's "19/19" count was
  stale — current tree discovers 11 fixtures). Continues: IncClaimReduction
  prepare device path + 2^25 cool ABBA + full gate matrix.

- **W1D (task 5c8623e5): root-cause artifact complete** (lane-reports/
  w1d-rootcause.md in gpuutil-w1d — read it in full before any memory work).
  Headlines: (1) st5 DeviceIrrScanner parks a 30 GiB ping-pong pair in the
  global RETIRED pool across a 29.5 s idle window (late-st5 tail + st6a + st6b);
  ≤4 GiB ever carved before mid-st6b; biggest adoption misses the pool. (2) On
  trunk the 2^27 st6b degradation is NOT OS page pressure — zero compression/
  swap; prepares run parallel-busy (11-12 cores) ⇒ DRAM-bound at 2^27 working
  sets; H-park vs H-shape separable only by free-at-retire ablation (running).
  (3) **W4-U1 madvise failure root-caused: MADV_FREE_REUSABLE is a silent
  no-op (rc=0, footprint unchanged) on any range ever mapped via
  newBufferWithBytesNoCopy — IOGPU holds a second VM-object ref; release does
  not restore eligibility.** Micro-experiment committed as ignored test
  (madvise_probe). Any madvise-shaped decommit of Metal-wrapped pages is DOA;
  structural fix = actually drop buffer + backing. (4) **st4 verdict: SHAPE,
  not pressure** — constant ×2.05/doubling, no tier cliff; RegistersRWC::prepare
  is a SERIAL host build (1.9 cores, 4.70 s @2^27). D hands st4 off; fix class
  = parallelize/port ⇒ converges with scope-lane finding (unfused round loop).
  st4 becomes lane W2B (port+fuse+optional pairing). st6a footprint drop @2^27
  (−30 GiB) = TraceRecord family death, log_T-parity-dependent (explains the
  cross-scale contradictions).

## Post-directive lever board (sound-but-not-byte-identical, wave-1.5+)

Ranked candidates unlocked by the 2026-08-04 directive; each needs a journal
soundness note before merge.

1. **Round-pairing (two sumcheck rounds per GPU round-trip).** SCOPED
   2026-08-04 (lane-reports/w15-roundpair-scope.md, task 5068310d): **narrow GO
   only — st4 RegistersRW device-only prefix rounds 0..6, pairs (0,1)(2,3)(4,5),
   never across the r6/r7 join; generic rollout NO-GO** (normal Metal slots
   already fuse bind+eval in one command buffer — measured exposed round-boundary
   gaps: st5 5 ms, st6b 0.14 ms, st3 54 ms; pairing there buys ~nothing and d5/d6
   members make paired messages 3-3.5× ALU). RegistersRW is the sole UNFUSED
   slot: message pass + host count-scan/alloc + bind pass per round = 2 waits +
   1 host boundary/round; prefix = 5.862 s @2^27 at 30.5% GPU-eq (3.73 s
   sampled-0% inside its own spans). Modeled pairing win 1.2-1.8 s @2^27.
   Soundness: verifier checks Σ_{x,y∈{0,1}} g(x,y) = prev claim, samples both
   challenges after the bivariate is absorbed; 2d/|F| per pair = two single
   rounds. zk: protocol-profile knob in JoltProtocolConfig, pairing rejected
   fail-closed under zk (report §5, Option A). NOTE: a protocol-NEUTRAL
   alternative captures much of the same prize — fuse RegistersRW (device
   prefix-scan + single-CB bind+message, 13→7 passes); sequenced as route (a)
   before pairing route (b). Both routed through lane D's st4 root-cause
   (findings forwarded to D 04:45 UTC) — st4 owner implements, wave 2.
2. **Address-major layout for booleanity: DEAD (W2A design §2, journaled
   negative result).** Cycle-major binding (j before k) is vacuous — on the
   boolean cube ra² = ra, so Σ_k eq(r_a,k)(ra²−ra) ≡ 0 pointwise and a j-first
   sumcheck proves 0=0; keeping k open forces collision-aware sparse state ≈
   the K×T grid in disguise. Per-round stateless gather (H[j]=E[addr(j)]) is
   valid math but recomputes the pushforward per address round: ~10× ALU vs
   one O(T) add-only build. The pushforward IS the compression. (May still
   apply to non-booleanity address relations — none currently hot.)
3. **Stage-boundary restructuring on the st5→st6b seam** → ACTIVE as W2A R1:
   booleanity eq-anchor moved to the stage-1 cycle binding (pure anchor, never
   absorbed/drawn; soundness §3 of w2a.md — point sampled after ra commitments
   bound, no FS draw moves, fail-closed BooleanityAnchor axis in
   JoltProtocolConfig, zk pinned legacy) ⇒ pushforward builds on a capped
   background pool overlapping st5's GPU window. + R2 bytecode 4-early/1-late
   split (byte-identical) + R3-lite HWCR on-the-fly tensor eq (kills the 4.3 GB
   eq_table materialization at the pressure tier, byte-identical). GO granted
   2026-08-04 ~06:45 UTC; modeled st6a+st7 −43…−56% @2^24 (gate −30%).
4. **st4 RegistersRWC restructure** → ACTIVE as W2B: implicit-CSR
   fixed-segment layout (W_r = min(3·2^r, K) per row, offset = row·W_r — no
   counts/scans/dynamic compaction, ≤3T entries), host-parallel prepare first,
   single-command-buffer rounds. Protocol-neutral steps 1-2 (byte-diff oracle);
   optional round-pairing step 3 gated on orchestrator GO. Memory-viability
   gate added (peak-storage projection @2^27 before 2^25 confirm).

## Wave 3 — W3A certified

- **W3A CLOSED — merged to trunk @91c4d5700** (kernels 239/239, prover 20/20).
  Fix: `MmapVec` (anonymous mmap, munmap on drop) backs TraceRecord lanes,
  RegisterLanes, RamAccessColumns, SharedInstructionRows, PcRows, and the IRR
  device ping-pong (`OwnedBacking::Mmap`) — every corpse leaves RSS+footprint
  at its designed drop site; compressor storm structurally avoided (peak
  96.9→79.3 GiB). Matched-pair @2^27: −10.5 s (−11.6%). Third negative result
  pinned: `malloc_zone_pressure_relief` is a measured no-op on freed huge
  Vecs (why corpses ride forward). Post-merge certification (locked, deep
  afternoon ambient, 22-day-uptime box): **76.92 / 73.66 s — best 73.66 s =
  1.822 MHz** (vector [10.73, 8.10, 4.32, 1.97, 10.83, 12.88, 0.131, 15.45,
  1.24, 7.95]); st5 12.88 = all-time best (mmap zero-fill elides st1-walk
  memsets); st6b 15.4 lean. Campaign standing vs pre-campaign canonical
  77.168: **−3.5 s at equal luck, −11..−13 s same-day, and the ±9 s ambient
  lottery is dead.** 2^25: 19.24 s (1.744 MHz). Expect ~72-73 in a clean
  ambient window (3 banked 2^27 runs remain for a flagship number after the
  box gets a natural reboot/quiet day — do NOT reboot while lanes live).

## Wave 3 (continued)

- **W3D CLOSED — RETAINED, merged to trunk @513b1e195** (kernels 242/242,
  prover 20/20 metal arm post-merge; byte-diff 12/12 BOTH arms on the lane
  tree with the hoist live). F1: TraceRecord::collect (challenge-independent,
  proven by signature) hoisted into st0's commit window — scoped thread, 8-
  thread capped pool, extends W2A's BACKGROUND_BUILD_TOKEN (record holds it
  st0-st1, 6a builds post-st4; zero interaction measured), mpsc carry with
  inline-rebuild fallback, JOLT_RECORD_HOIST=off ablation knob. F2: the four
  post-rounds claims walks (st1 outer 35-opening, st2 product+ICR, both
  twins) lose their full-T eq materializations (3× 4.3 GiB @2^27) via run-
  factored e_hi/e_lo + fmadd_s256 unreduced accumulation. F3 mapped not done
  (InstrInput q0 device promotion, ~0.4 s door). Gates: st1+st2 −43% at both
  2^24 and 2^25 cool (gate −25%); st3 bonus −15..−23% (eq-flood relief).
  **st0 tax saga (model-failure lesson):** 2^25 cool tax +5.8% (+264 ms),
  mechanism pinned = bandwidth contention (pool-width and QoS probes both
  null); lane deferred per W3C amendment at 5.2× win/loss. At 2^27 the tax
  model FAILED — the walk ran 13.0 s (not the 7.2 modeled) and st0 stretched
  10.7→17.8 under mutual starvation — BUT the same ambient inflates the
  trunk's INLINE walk worse (st1 12.1 s): same-window certification pair
  (afternoon deep-bad ambient) = **W3D 77.72 vs trunk 87.22 = net −9.5 s**;
  st1 4.65/st2 2.89 confirmed at tier. Two regimes, same sign ⇒ retained.
  Clean-window magnitude → tonight's banked flagship pass. Trunk binary
  rebuilt: ca653254ec4ce1eb….

- **W3A (task 743016e4, fable-max, worktree gpuutil-w3a) — PHASE 2 (fix).**
  Phase-1 artifact (w3a-rootcause.md, ACCEPTED 2026-08-04 17:0x UTC) overturns
  the parity theory: (1) Rust drop sites are at design points in every mode
  (TraceRecord dies at st4-OPEN; SharedInstructionRows pinned to proof end by
  construction) — D's "family dies at st6a/st8" was a ledger artifact, the
  ledger tracks RECLAIM not drops; (2) libmalloc keeps freed huge entries
  resident until proof end even at calm tiers; (3) the day-flip = ambient
  occupancy (33 GiB today) → prover peak 90-97 GiB touches the ceiling →
  compressor storm inside st6b (free pages→0, 4.5-10 GiB/s compression +
  decompression thrash, monitor thread starved 8-10 s) — yesterday cleared the
  ceiling, allocator vm_deallocated before st6b; (4) trunk's +8 s robustness =
  page-demand composition (device buffers + emptied st6a fault less mid-storm);
  (5) NEW PROBE: munmap removes Metal-wrapped pages from footprint in every
  leg (madvise = silent no-op, munmap works). Fix in flight: munmap-backed
  MmapVec for corpse-pile members (TR lanes, RegisterLanes, IRR ping-pong,
  RAC/PcRows/SIR/SOI) → peak ~60 GiB, storm structurally impossible, expected
  st6b 14-15.5 / total 72-73 s, byte-identical. Orchestrator cautions issued:
  munmap strictly after buffer release + completed waits (live-wrapped probe
  leg is the DANGEROUS case, not a green light), Metal never owns the
  deallocator, st1 walk swap stays mechanical. Task 0 resolved: counters never
  broke — a manual postprocess step converts monitor events; now automated
  (eef0f088e). Also: capped-BRRC re-hosting adds +6 GiB at peak (96.9 vs 91.0)
  — W3B input.
- **W3B CLOSED — UNCAPPED, merged to trunk @445ff4479** (kernels 239/239,
  prover 20/20 incl. byte-diff 12/12 both arms — fixture count grew again;
  the gate is "all discovered", currently 12). Verdict: the 2^27 device cliff
  was the STORM, not the kernels — BRRC GPU execution is 1.279 s @2^27 and
  scales linearly (2^26→2^27 = 1.92× for 2× rows); the certification-day
  9.89 s was ordered-queue WAIT inflated by compressor stalls. Under W3A's
  lean regime, two opposite-order locked pairs: st6b 16.382 device vs 18.804
  CPU mean = **−2.42 s (−12.9%)**, IncCR canary improves both pairs, CB
  timestamps (new JOLT_METAL_CB_TRACE) show no member's GPU execution
  regressing, and device-on runs 3.8 GiB LOWER peak (no CPU-table rebuild).
  All four cliff hypotheses discriminated: SLC blowout no, 64-bit offsets no,
  occupancy no, batch-scheduling-under-storm YES (historical). metal_gate_
  capped + MAX_DEVICE_ROWS removed. Sequencing vindication: W3B after W3A
  turned a kernel-rewrite lane into a measurement + one-line-revert lane.
- **W3C CLOSED — RETAINED at certification scale, merged to trunk @5ce6c19d1.**
  Lane delivered the parallel build (3-pass count/scan/scatter onto the frozen
  legacy representation, byte-identical, serial builders kept as equality
  oracle + JOLT_REGISTERS_PREPARE_SERIAL timing arm) and honestly REJECTED
  per gate letter: prepare −67…−79% and st4 −31…−38% @2^25, but st8 +5.3-5.6%
  (+220-240 ms) and st6b +3.6% violated the no-stage->+2% clause at both
  scales/fan-outs ("shared-SoC pressure" aftereffect). ORCHESTRATOR OVERRIDE
  (gate amendment, journaled reasoning): the per-stage clause exists to catch
  hidden regressions, not to veto a favorable net trade; the aftereffect is a
  fixed-ish tax (thermal/power-state class) while the st4 saving scales with
  T. Certification A/B @2^27 (same binary, env-switched, serial arm FIRST =
  thermal ordering against the candidate): serial 74.40 s vs parallel
  **71.43 s = net −2.97 s; 1.879 MHz — new campaign best.** st4 10.8→8.08,
  reg prepare 4.7→2.0 s, st8 8.62 within session range (no visible 2^27
  penalty). Retention matrix run by orchestrator (lane had skipped it after
  self-rejection): kernels 241/241 (incl. 2 new equality tests), prover metal
  20/20, byte-diff 12/12 CPU arm, muldiv 3/3+3/3, clippy host/zk/metal, fmt —
  all green. Lane discipline exemplary; the override is a scale-of-decision
  question, not a gate failure.
- **Fresh attribution 2026-08-04 17:5x UTC** (trace /tmp/gpu-util-trace-2to27-
  wave3-20260804.json.gz, monitor binary on trunk @5ce6c19d1; NOTE: the fixed
  counter reads a different scale — healthy stages now ≈44-45%, use
  ratios-to-healthy, not absolutes): **zero-GPU windows >1.5 s: NONE** (campaign
  open: 7 windows, 28.8 s). Per-stage (instr. wall / gpu% / ratio-to-healthy):
  st0 11.3/44.7/1.0, st1 9.2/21.8/0.49, st2 4.7/19.2/0.43, st3 2.3/15.8/0.35,
  st4 8.5/19.6/0.44, st5 13.8/42.2/0.94, st6a 0.17/0, st6b 16.3/18.0/0.40,
  st7 1.9/6.8/0.15, st8 8.2/44.7/1.0.
- **st1/st2/st3 attribution (the last big door):** st1 = 85% prepare —
  SpartanOuterUniskip::prepare 6.23 s (TraceRecord::collect 4.17 s inside) +
  OuterRemainder::prepare 1.65 s vs rounds 0.36 s. st2 = Stage2Batch 4.20 s
  with rounds 0.98 s + ProductUniskip::prepare 0.50 + ~1.8 s unattributed
  host glue. st3 = InstrInput rounds 1.61 s + SpartanShift::prepare 0.44.
  Fix classes: hoist/overlap challenge-independent walk work into st0's
  11.3 s GPU-heavy window (W2A pattern, process token exists) and/or
  parallelize the serial walks (W1D/W3C pattern). → **W3D ACTIVE: task
  0d5cf2fa, fable-max, worktree gpuutil-w3d.** Phase-1 dependency/shape
  artifact before fixes; protocol changes need GO; W3C gate amendment
  codified in its brief (surface SoC-pressure tradeoffs, defer retention to
  2^27 certification instead of self-killing). Prize ~−3.5..−5 s @2^27.
- st3 feed → **W3E CLOSED — NO-GO with mechanism** (task 4a023a76): InstrInput q0 device
  promotion per W3D F3 map (~0.4 s @2^27 + st3 util). Time-boxed to 00:30 UTC
  for the flagship window. Byte-identical, standard gates.

## Parked doors (inherited)

- Co-issue probe (M5: is a wide mul dual-slot? add32 only 1.83× mul32) — ALU-roof
  interpretation, not util; parked.
- NTZ small-space — parked (CUDA-side door; no Metal evidence yet).
- st3 feed, st1 TraceRecord::collect residual (3.94 s @2^27), st5 late-round CPU
  tail (2.5 s), st2-tail→st3 hole (4.3 s) — wave-2 candidates.

## Log

- 2026-08-04 04:0x UTC: campaign opened; nosleep on; traces analyzed; wave-1 cut
  published; lanes A/B/D dispatched.
- 2026-08-04 04:25 UTC: USER DIRECTIVE logged — byte-parity gate lifted, replaced
  by e2e verify + full tests + journaled soundness notes for protocol changes.
  Wave-1 lanes finish as scoped (unaffected: exact-math ports). Lever board
  published; round-pairing scoping lane dispatched (read-only, no bench lock):
  task 5068310d, gpt-5.6-sol-xhigh, report → lane-reports/w15-roundpair-scope.md.
- 2026-08-04 04:36-04:50 UTC: scope lane returned in ~15 min — narrow GO (st4
  RegistersRW prefix only), generic pairing no-go; lever-board entry updated
  with verdict + protocol-neutral fusion alternative. Findings + gate change
  forwarded to lane D (task 5c8623e5) for its st4 root-cause artifact. W1A/W1B
  checkpoint-1 decomposition reports read (w1a.md, w1b.md — both sound, byte
  parity retained as their retention gate, stricter than required = fine).
  Lane task IDs (recorded late, lesson): A=2bbe078e, B=83cf4e87 (confirmed:
  sole surviving codex process = 83cf4e87 = B, still implementing after A's
  kill-gate exit), D=5c8623e5, scope=5068310d (done).
- 2026-08-04 15:5x UTC: **WAVE-2 CERTIFIED (same-day interleaved, non-monitor,
  bench-locked, 22-day-uptime box).**
  | arm | 2^27 runs | mean | st6b | st6b entry |
  |---|---|---|---|---|
  | trunk @cap commit (V1 anchor) | 76.37 / 76.86 / 77.10 | **76.8** | 17.0-18.5 | 67.6-69.0 GiB |
  | baseline (wave-1-open ≈ canonical code) | 85.08 / 87.53 | **86.3** | 25.2-27.4 | 69-70 GiB |
  **Trunk −9.5 s (−11%) same-day; 1.75 MHz vs baseline's 1.53-1.58 today.**
  2^25: trunk 19.233/19.248 (1.744 MHz) vs canonical 19.822. Stage story
  @2^27: st6a 2.27→0.13-0.16 (−94%), st7 2.07→1.06-1.49, st5 14.65→13.5-13.8
  (−1 s), st4 flat (W2B rejected), st6b = everything (see below).
  **BISTABILITY EXPOSED (memory-ledger receipts in /tmp/cert-s27-*.log):**
  yesterday's canonical 77.17 rode the GOOD mode (record family dies at st6a;
  st6b entry 41 GiB — D's ledger). TODAY every run of BOTH trees enters st6b
  at ~68-70 GiB (family death migrated inside st6b; st6a frees ≈ 0, st6b
  frees −32..−44 GiB) — trigger unknown (same binary+fixture as yesterday's
  good-mode runs; NOT log_T parity; session/page-cache state suspected).
  Under this adverse mode trunk holds 17-18.5 st6b vs baseline 25-27 —
  wave-2 content is ~8 s more robust — but the lean-entry upside (~14 s st6b
  → ~72-73 s total) is unrealized. ⇒ WAVE-3 LANE #1: make the record-family
  drop deterministic before st6b (U2-demonstrated lever, D's doors note).
  Anchor A/B @2^27: V1 76.86 vs legacy 76.37 — wash at tier (V1 keeps its
  clear 2^25 win; V1 stays default). BRRC-device cliff note: measured on a
  fat-entry run — cliff magnitude under lean entry unknown; W3 BRRC lane is
  sequenced AFTER the lifetime lane for that reason. Instrumentation bug
  found: monitor build emitted ZERO gpu_percent counters into the chrome
  trace this session (ledger prints fine) — fix before next attribution
  pass. Attribution trace archived: /tmp/gpu-util-trace-2to27-wave2-20260804
  .json.gz (walls-only value). 2^27 budget spent this session: 10 runs.
- 2026-08-04 14:3x UTC: **CERTIFICATION FOUND A 2^27 CLIFF IN W1B'S PORT —
  root-caused and capped same-session.** First trunk 2^27: 89.12 s (st6b
  DOUBLED 13.87→27.73; BRRC device rounds 9.89 s vs 4.82 CPU before; IncCR
  rounds 6.37 s = contention victim). Discriminator: rerun with
  JOLT_METAL_MIN_TERMS_BYTECODE_READ_RAF_CYCLE=huge → **74.91 s / 1.792 MHz**
  (st6b 16.82: BRRC CPU 3.81, IncCR back to 3.17). Verdict: BRRC device path
  wins ≤2^25 (−53% member) but at 2^27 runs 2.6× slower than its CPU twin AND
  stalls the batch's other five device members (bandwidth/queue contention).
  W1B never ran 2^27 (correctly, per lane rules) — the tier cliff is exactly
  the class D diagnosed for CPU members, now observed device-side. FIX:
  metal_gate_capped + MAX_DEVICE_ROWS = 2^26 on the slot (byte-identical
  fallback; env-overridable; 2^26 kept on device per sub-linear-scaling bet).
  Kernels 236/236 green. WAVE-2 CERTIFICATION (hot-box, 22-day uptime — cool
  finals pending): 2^25 = 19.233/19.248 s (~1.744 MHz, vs canonical 19.822);
  2^27 capped = 74.9-77.1 s across thermal states (vs canonical 77.168 —
  cool run pending). FREE WINS CONFIRMED @2^27: st5 13.59-13.70 vs 14.653
  canonical (≈−1 s — record-family dies mid-st5 via W2A's background
  consumption); st7 1.13-1.49 vs 2.072 (R3-lite's 4.3 GiB alloc kill worth
  more at the pressure tier, as predicted). WAVE-3 DOORS from the new trace:
  st6b residual vs canonical (+2.9 s in matched hot runs — IncCR/RamHB/
  InstrRA device rounds at tier; needs cool confirm first), BRRC device
  2^27 root-cause (why 2.6×: 5×T×32B flat ping-pong streaming? 64-bit MSL
  addressing? width-8 materialization?), st4 prepare salvage (W2B's −65%
  prepare onto old rounds), st1/st2 holes untouched (4.2+4.3 s @2^27).
- 2026-08-04 13:0x UTC: **W2B CLOSED — REJECTED at 2^25, not merged.** Two
  variants, both fail retention: two-buffer fixed segments (st4 −33.4% @2^24)
  = footprint-dead @2^27 (+52 GiB projected); one-buffer scale-safe variant =
  memory OK (+~5 GiB @2^25) but in-place device rounds +46.8% cancel the
  prepare win → st4 −3.0% cool @2^25 (gate −15%) + cross-stage violations.
  Honest self-rejection; branch gpu/util-w2b kept as experimental handoff
  (binary 116ac881…, artifacts /tmp/w2b-*). DURABLE FINDINGS: (1) the prepare
  parallelization works in isolation — device build −64.6% cool @2^25; (2) the
  round-loop rewrite is where both variants die (memory or speed); (3) pairing
  confirmed dead for st4 (fused-vs-unfused = noise). WAVE-3 SALVAGE CANDIDATE:
  bounded parallel prepare onto the EXISTING representation (old rounds
  untouched, byte-identical, no footprint delta) — D's original ~3-4 s @2^27
  prize; U3 lesson: parallelize the whole build (count+scan+scatter), not just
  the scan. Wave-2 net trunk state: W1B + W1D + W2A. Certification next.
- 2026-08-04 12:0x UTC: **W2A CLOSED — merged to trunk @cc7a5c5d5** (validated:
  kernels 236/236, jolt-prover 20/20 metal arm). Final: combined st6a+st7
  −51.6% @2^24 / −67.3% @2^25 cool (st6a −85.7% → −90.7% [0.53→0.05 s];
  st7 −10.7%; st6b bonus −8.1%/−4.8% from guaranteed-parked shared scans);
  st5 contention (+4.5% from two concurrent 4-thread pools) root-caused and
  fixed via process-wide build token → final st5 +1.1% ✓. R1 = FIRST LANDED
  PROTOCOL CHANGE under the directive: BooleanityAnchor axis (V1 = stage-1
  anchor, DEFAULT for modular non-zk prover; legacy/zk pinned; fail-closed;
  anchor_v1 e2e proves V1 verifies AND a re-tagged anchor fails = axis
  load-bearing). Soundness note: w2a.md §3 (accepted 06:45). NOTE for
  certification: canonical anchors predate V1 — wave-2 close certifies the
  new default protocol; wall comparison remains the metric. Session thermal
  caveat: lanes measured absolute walls 15-20% above canonical anchors
  (22-day-uptime box after a full bench day) — A/B ratios valid, absolutes
  need a cool certification pass.
  **W2B checkpoint 2: steps 1-2 gate PASSED** (st4 −33.4% @2^24, prepare
  −82.8% [0.788→0.136 s], byte-diff 11/11 both, fused CB; step 3 round-pairing
  correctly self-killed: fused-vs-unfused delta = noise @2^24, residual sync
  <0.3 s projected — scope report's modeled pairing win is fully absorbed by
  the fusion+rewrite, pairing dead for st4 too). **BUT memory-viability gate
  TRIPPED by orchestrator:** +6.57 GiB st4 transient @2^24 → ~+52 GiB linear
  @2^27 → ~112-123 GiB peak = compressor cliff. W2B ordered to reduce the
  transient (compact-scalar segments / narrow layout / windowed conversion)
  to ≤+10 GiB projected @2^27 and re-measure before 2^25 retention.
- 2026-08-04 08:0x UTC: **W1D CLOSED — merged to trunk @79b83e0e3.** Final
  report lane-reports/w1d.md: mechanism verdict with receipts (pressure tier
  does not exist on trunk: no compressor/swap/starvation; park-vs-free null),
  arena deletion −381/+28 (gates: 2^25 neutral −0.09%, byte-diff 11/11 both,
  kernels 231/231 on its tree, muldiv, clippy, fmt). Honest accounting: 0 s
  wall delivered vs −4 s target — the chartered door was worth ~0; redirected
  campaign truth instead. Orchestrator validated MERGED trunk (B+D):
  kernels 233/233, byte-diff 11/11 metal + 11/11 CPU. Doors it opened:
  st6b residual = eq-evals 2.2 s + oracle_table 1.6 s feeds (superlinear at
  2^27) — wave-2/3 candidate, unowned. W2A/W2B notified to merge trunk before
  their A/B phases. Wave-1 fully closed: A killed, B retained (−0.91 s @2^25),
  D null-with-simplification. 2^27 certification deferred to wave-2 close.
- 2026-08-04 06:5x UTC: **W1D ablation verdict: H-shape — park is perf-neutral
  @2^27 (+0.12 s total = noise; st6b −0.17 s).** W4-U1 door closed with a
  measured null: the −4 s allocator prize does not exist on trunk. D's fix =
  delete the retired-buffer arena outright (risk removal + simplification,
  perf-neutral by construction; gate 2^25/2^26 neutral ±1% + full matrix +
  byte-identical) — committed on its branch, gating now. st6b's 2^27 excess is
  intrinsic CPU-member working-set shape (the surface W1B's port removes).
  **W2A design GO** (R1 anchor + R2 split + R3-lite; amendments: trunk merge
  first, 11-fixture count, additive config-axis pattern, st2-st5 regression
  watch, journal duty). **W2B design ack** (fixed-segment CSR; memory-viability
  gate added). Lever board items 2/3/4 updated: 2 dead (negative result), 3/4
  active as W2A/W2B.
- 2026-08-04 06:1x UTC: **W1B CLOSED — retained + merged to trunk @c8b5841e0.**
  Final 2^25 cool ABBA: st6b −25.5% (2.070→1.543 s), member −38.1%, total
  −0.910 s. IncClaimReduction prepare device port REJECTED (−9 ms prepare /
  +22 ms stage = noise; reverted; monitor showed no util shift). Full matrix
  green: kernels 233/233, dory 46/46, muldiv 3/3+3/3, byte-diff 11/11 both
  feature sets, clippy/fmt clean. Binary sha 333ad9ee…. W1D artifact complete
  (§Wave-1 gate results); D ablating park-vs-free @2^27 under lock
  (JOLT_METAL_NO_PARK knob committed). **W2B dispatched: task 6d1f61bb,
  gpt-5.6-sol-xhigh, worktree gpuutil-w2b (branch gpu/util-w2b off trunk
  @c8b5841e0)** — st4 CSR/fixed-segment rewrite (step 1) + round-loop fusion
  (step 2, both byte-identical) + optional gated round-pairing (step 3,
  orchestrator GO required). st4 ownership transferred D→W2B. W2A + D notified
  of merge; W2A told to merge trunk into its branch now.
- 2026-08-04 05:0x UTC: wave-1 gate results journaled (A killed / B passed —
  see §Wave-1 gate results). W2A dispatched: task 0eb25fd2, fable-max,
  worktree gpuutil-w2a (branch gpu/util-w2a off trunk @2fc5e877f) — st6a+st7
  rethink under protocol freedom (hypothesis menu H1 address-major bind order,
  H2 device pushforward shape, H3 r_cycle-independent hoisting, H4 HWCR
  restructure). Design-before-code hard gate: W2A must get orchestrator GO on
  its design artifact before implementing. D notified of scope findings +
  directive earlier (04:45); D currently holds bench lock for 2^27 diagnosis.

## Utilization-surface status after W3E (2026-08-04 20:2x UTC)

W3E's no-go (best device shape — coefficient form via boolean-endpoint finite
differences, 6 Montgomery products/pair — only TIES the 18-core host
exact-integer pipeline at 2^24; host is not a field-arithmetic twin, W1A's
lesson in miniature) closes the last mapped door. Remaining parked-with-reason:
SpartanShift γ-split (≤0.3 s, plumbing-deep), st1 claimed_inputs device port
(~0.6 s, W1A-shaped risk), round-pairing (dead twice over). No stage retains a
>1 s GPU-idle host mass; residual low-util is bandwidth-bound device work
(InstrInput r1 17 GiB write, st6b member streams) and small stages. The
mandate's surface is EXHAUSTED pending flagship certification numbers.

## Standing velocity rules (USER DIRECTIVES, 2026-08-04 16:48 EDT)

1. **Full gate battery ONCE per wave**, at wave close on the integrated trunk
   — lanes run targeted tests + their parity oracle only (retro note: this
   codifies what W3C's close already practiced).
2. **Max 2 timed benches per decision.** A pair is the default; a third run
   only when the pair disagrees beyond noise. Flagship cert = 2 canonical
   runs + conditional third.
3. **Small scales (2^22-24) for iteration; big scales for gates/certification
   only.**

## CAMPAIGN CLOSE — flagship certification (2026-08-04 night, quiet box, user-idle 16 h, uptime 22 d)

Trunk @ec9db38f6 (+velocity-rules docs), binary ca653254…. Bench-locked, 4-min
gaps unless noted. Per velocity rule: pair + conditional third (+1 F1-audit).

| run | conditions | total | note |
|---|---|---:|---|
| r1 | after ~30 min idle | **71.46 s = 1.878 MHz** | good st0 mode (walk 9.3 s) |
| r2 | 4-min gap | 78.63 | bad st0 mode (walk 13.2 s) |
| r3 | 4-min gap | 78.47 | bad mode — central tendency back-to-back |
| r4 | JOLT_RECORD_HOIST=off, same window | 79.73 | F1 HOLDS in bad mode too (st1 reverts to 11.7) |
| 2^25 pair | warm, post-marathon | 19.71 / 19.89 | cool reference = W3D ABBA ≈18.0 summed |

**Final attribution (monitor run, walls-invalid): zero-GPU windows >1.0 s:
NONE.** GPU ratio-to-healthy: st1 0.97 (was 0.49), st2 0.65, st5 0.95,
st6a n/a (0.26 s), remaining sub-healthy = bandwidth-bound or small: st3
0.28 (2.9 s stage), st4 0.39, st6b 0.40, st7 0.15 (1.9 s). Trace:
/tmp/gpu-util-trace-2to27-final-20260804.json.gz.

### Campaign ledger vs open (2^27 canonical 77.168 / 1.69 MHz; 2^25 19.822)

Best certified: **71.43-71.46 s = 1.88 MHz @2^27** (−7.4%); 2^25 ≈18.0-19.2
(−3..−9% by regime). More important than the mean: the tails. Campaign open
had a hidden ±9 s ambient lottery (77 lucky / 85-90 unlucky, mechanism:
corpse-pile compressor storms). Closing trunk: worst observed back-to-back
mode 78.6 ≈ the OLD LUCKY case; old unlucky case eliminated (storm
structurally impossible). Same-window controls: pre-W3D trunk 87.2 (afternoon
adverse), F1-off 79.7 (tonight).

Stage walls, campaign open → close (canonical-ish best): st0 10.8→12.3-12.6
(hosts the hoisted walk), st1 8.0→4.5, st2 4.3→2.7, st3 2.0→2.1, st4
10.4→8.3, st5 14.7→12.9-13.7, st6a 2.3→0.13-0.18, st6b 13.9(lucky)→15.4-16.5
(storm-proof), st7 2.1→1.2-1.8, st8 8.8→9.0.

Waves: W1A killed / W1B retained (+uncap after W3A) / W1D null+arena-deletion
/ W2A retained (protocol: BooleanityAnchor V1) / W2B rejected / W3A retained
(mmap lifetimes) / W3B uncap / W3C retained-at-certification / W3D retained /
W3E no-go. One protocol change total, fail-closed, load-bearing-tested;
everything else byte-identical.

### Negative-results index (durable)

1. MADV_FREE_REUSABLE: silent no-op on any range ever wrapped by a no-copy
   MTLBuffer (probe-pinned; munmap works, incl. live-wrapped — ordering!).
2. malloc_zone_pressure_relief: no-op on freed huge Vecs.
3. libmalloc never returns multi-GiB frees mid-proof; drop-site ≠ reclaim.
4. Round-pairing: dead — slots already fuse; the one unfused slot's fix was
   fusion, and after fusion the residual sync is noise.
5. Cycle-major booleanity sumcheck: mathematically vacuous (ra²=ra).
6. Per-round stateless gather for one-hot address phases: 10× ALU.
7. Device pushforward scatter (all shapes tried): loses to 12-core CPU.
8. Device q0 for InstrInput: best shape TIES host exact-integer pipeline.
9. Device IncCR prepare: no better than parallel CPU (DRAM-bound).
10. "The 2^27 pressure tier" as OS pressure: false on trunk — it was
    working-set shape + (pre-W3A) the corpse-pile storm.

### Parked doors (future waves)

1. **st0 walk↔commit contention** — the new dominant variance (±5 s @2^27,
   bimodal, idle-time-correlated). Mitigations untried: spawn stagger, walk
   chunking with commit-aware yielding, QoS on the METAL side instead.
2. st4 round-loop fusion under a memory-viable representation (W2B's two
   variants bracketed it: fast+fat vs lean+slow; the middle is unexplored).
3. st6b bandwidth tier (members are device but DRAM-bound; SLC tiling).
4. SpartanShift γ-split (≤0.3 s), st1 claimed_inputs device port (~0.6 s).
5. Co-issue probe, NTZ small-space (inherited, still parked).

Campaign mandate status: **utilization surface exhausted — no zero-GPU
windows >1 s remain; every >1 s GPU-idle host mass eliminated or
evidence-closed.**
