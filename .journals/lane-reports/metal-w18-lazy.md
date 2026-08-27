# W18 lane L18 — st6b BytecodeLazyRound + RamRAV base: both doors CLOSED (premise-false)

**Verdict: double NO-GO with receipts, tree unchanged.** The two B16 parked
doors dissolve under isolated single-kernel measurement: both "inefficiencies"
were co-run **window** readings taken as exec time, on kernels that are in
fact at (or near) the compound Fr ALU roof. Bytecode device kernels total
**~155 ms isolated @2^27** — the 0.3-0.7 s modeled prize never existed. No
production change ships (nothing to kill-switch); the parity-checked
attribution rig is retained (`jolt-eval --bench bytecode_lazy`).

## 1. Rig

`BytecodeLazyFixture` / `RavLazyFixture` (jolt-kernels, `bench-utils`) +
`jolt-eval/benches/metal/bytecode_lazy.rs`: production geometry (bytecode
`num_ra=2`/factors 3, chunk 8; RAM RAV 2 polys × batch 2 kind 2; Instr RAV
16 × 4 kind 0), loopy hot-every-cycle mapped-PC rows / production column-mix
InstructionCycleRows, `compile_variant` probes, same-window interleaved
A/B pairs only. Rig validity: source-identical generic twins ±0.3% of
production on every shape. Probe parity: `jkx_bc_f3`, `jkx_bc_simdsum`,
`jkx_bc_sgbar`, `jkx_rav_p2_b2`, `jkx_rav_p16_b4` byte-identical to
production (lane sums + fold bytes). Linear scaling verified over three
octaves 2^22→2^25 (×3.97-4.0 per two octaves; r0 2.05→8.13→16.23 ms) — the
@2^27 extrapolation below is ×4 off measured 2^25.

## 2. Door 1 — BytecodeLazyRound "1.7-3.6 Gmul/s underrun": CLOSED

Component ladder @2^24 (r0 = production 8.13 ms; interleaved pairs):

| cell (strip) | r0 w1 | r2 w4 | reading |
|---|---:|---:|---|
| notab (table loads → idx const) | −2.3% | −6.7% | branch tables L1-resident, free |
| nogather (rows+idx+adds gone) | −9.8% | −22.2% | whole gather = 10-22% |
| nopair (cur stream gone) | −1.5% | −19.7% | r0 cur read fully hidden; bind fold real |
| floor (grid+tg_sum only) | −14.7% | −41.6% | memory total 15/42% |
| onetg (1 tg_sum) | −30.9% | −52.8% | tg_sum ≈ 0.66 ms each |
| notg (grid only) | −36.5% | −56.7% | **grid ALU = 64% of r0** |
| null (launch) | −99.1% | −99.3% | launch base ~1% |

**Mechanism:** the kernel is ~64% product-grid Fr ALU **at the compound
roof** (6 muls + ~11 adds/group ≈ 8.5-9 mul-equiv × 2^23 groups in 5.17 ms
≈ the 11.6 Gmul/s saturated reference), ~22% `jk_tg_sum` (3 lanes × 9
barriers), ~10% gathers, ~1.5% cur stream. **Gmul/s was the wrong metric**:
it counts muls only, and a factors-3 kernel has 6 muls/group where its 7-12
Gmul/s siblings carry 8-48 — same roof, different op mix. Occupancy proxies
1024/32 everywhere: no register cliff; the runtime-bounded
`JK_BYTECODE_MAX_FACTORS` arrays cost exactly the −10..−12% the f3 probe
recovers, not a spill catastrophe.

Sub-door receipts (all parity-exact where applicable):
- **Factor-specialization f3** (arrays register-pinned, full unroll):
  −11.6/−10.4/−10.2% on r0/r1/r2 ⇒ ~**−14 ms @2^27** — 20× under the 0.3 s
  bar. NO-SHIP (matches w15's −2.9..−6.6% on the dense-shape twin).
- **simd-shuffle reduction** (5 shuffle levels, no barriers): **+21/+9/+8%**
  — 256-bit shuffle reconstruction costs more than AGX barriers save.
- **simdgroup-barrier tree** (full barrier only at the cross-simd step):
  **+31/+18/+15%** — `jk_tg_sum`'s plain barriered tree is already the
  right shape on this hardware. Reduction door closed with two receipts.

**Kernel mass @2^27 (isolated, ×4 off 2^25):** r0 64.9 + r1 42.8 + r2 23.7
+ adopt 8.8 + dense tail ~13 ≈ **155 ms**. B16's "0.3-0.7 s (2-4×
per-element)" priced 60-132 ms co-run windows @2^25 against 8-16 ms actual
exec — the queue time billed to those windows is Bool/InstrRAV gather
execution (B16's own footnote: windows are bounds, the union is exact).
Cross-check: Σ isolated kernels @2^25 (instr RAV r0-r2 204 ms + ram 20 +
bytecode 39 + Bool/adopt/Inc/Hamming est.) ≈ 620-700 ms vs B16's exact
union 728 ms — consistent.

## 3. Door 2 — RamRAV base anomaly: CLOSED (explained)

Isolated @2^24 r0: **ram (2 polys) 5.02 ms vs instr (16 polys) 53.8 ms** —
per-poly 2.51 vs 3.36 ms: RAM RAV is **25% cheaper per poly** than
instruction RAV. B16's anomaly ("2 polys ≈ 66% of the 16-poly window",
~1.7 Gmul/s) was window overlap on detached co-running CBs — isolated
@2^25 is 10.0 ms vs the ≤87.8 ms window it was read from.

Residual composition (ladder @2^24, r0): floor (batch-2 grid + 2 tg_sums +
launch) 74%, eq 11.6%, gathers 12.9% — the fixed per-thread base amortizes
over 2 polys instead of 16, which is why per-poly sits above the instr
marginal cost but below its average. Batch-2 specialization −1.7%, instr
p16b4 −0.9%: nothing there. Whole ram-RAV lazy mass @2^27 ≈ 78 ms — no cut
above noise exists; cross-driver base-sharing stays dead (w3 pricing
unchanged, blast radius unchanged).

## 4. What this means for st6b (for the orchestrator)

st6b's 4.0-4.5 s @2^27 is NOT hiding a bytecode/ram kernel prize. The
device queue is Bool + InstrRAV gather ALU (~0.8-1.0 s+ isolated: instr
r0-r2 alone ≈ 820 ms @2^27) — w3's "gathers are the mass" verdict, already
on the kill list — plus the serial host prepares and blocked waits B16's
detach/prelaunch already attacked. The B16 member-span wall (bytecode
rounds 2.93 s) is queue wait, not bytecode exec. **Suggested gate
measurement: none — tree unchanged; no 2^27 run owed.** If a future lane
wants st6b mass, the only doors with real tonnage are the killed gather
kernels themselves or scheduling, not these two.

## 5. Discipline

- FrBind 256.5 µs pre-window (gate <350). No e2e runs consumed (no
  production change to certify); all evidence single-kernel interleaved
  pairs @2^22/2^24 + 2^25 spot check under the GPU lock.
- Gates: metal suites **414/414** (1 known leaky) · byte-diff **20/20** ·
  `clippy --all --features host -D warnings` green · metal+bench-utils
  clippy green · fmt green. Zero production-path lines touched; proof
  bytes trivially identical.
- Diff: attribution rig only (fixtures `bench` module in the bytecode slot,
  bench-utils exports, `bytecode_lazy` bench). **Flag for PR-handoff
  audit** alongside `irr_cycle` (same retained-rig precedent). KernelId
  count unchanged (92).
- Kill-list adds: st6b BytecodeLazyRound factor-specialization (≈−14 ms
  @2^27, sub-bar 20×; premise was window mispricing) · lazy-round tg_sum
  replacements (simd-shuffle +21%, simdgroup-barrier tree +31% — the plain
  tree is optimal on AGX) · RamRAV base anomaly (artifact; per-poly cost
  below instr's).
