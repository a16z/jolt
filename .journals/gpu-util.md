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

## Post-directive lever board (sound-but-not-byte-identical, wave-1.5+)

Ranked candidates unlocked by the 2026-08-04 directive; each needs a journal
soundness note before merge.

1. **Round-pairing (two sumcheck rounds per GPU round-trip).** Per pair: one
   eval pass producing a bivariate g(X,Y) on a (d+1)² grid + one bind-by-2 pass,
   vs two eval + two bind passes today → ~2× fewer big-array traversals and ~2×
   fewer host↔GPU sync/transcript round-trips on round loops. Cost: (d+1)²
   coeffs vs 2(d+1) per pair in the proof (d=3: 16 vs 8 Fr) and more ALU per
   pass. Soundness sketch: verifier checks Σ_{x,y∈{0,1}} g(x,y) = prev claim,
   samples (r_i, r_{i+1}), next claim g(r_i,r_{i+1}); Schwartz-Zippel gives
   2d/|F| per pair = identical total to two single rounds. Best on the mid-util
   stages (st1/2/3/4/6b, 18-31% @2^27) where round-boundary stalls dominate.
   SCOPING LANE dispatched (read-only) before any implementation.
2. **Address-major layout / materialization** for address-phase sumchecks
   (booleanity_address, bytecode_read_raf_address — the st6a pair). Only if
   lane A's exact-math port is gather-bound; wait for A's root-cause report.
3. **Stage-boundary restructuring on the st5→st6b seam**: hoist st6a/st6b
   prepares to overlap st5 GPU rounds if challenge deps allow (scheduling =
   sound trivially); with the directive, reordering transcript absorptions
   across the seam is also legal if FS ordering keeps every challenge derived
   after everything it must bind (journal the argument per change).
4. **st4 RegistersRWC restructure** — if D's root-cause shows the prepare hole
   is inherent to the current formulation rather than allocator pressure.

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
  published; round-pairing scoping lane dispatched (read-only, no bench lock).
