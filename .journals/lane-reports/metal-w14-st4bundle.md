# Metal W14 st4 sub-bar bundle — batch overlap + bind-slab arena reuse

## Verdict

**RETAIN both doors.** st4's two pre-priced sub-bar doors (W11 S11) land
bundled, each default-on behind its own kill switch:

1. **Batch overlap (`JOLT_REGRW_OVERLAP=0` kills):** the RegRW Metal slot
   opts into the sumcheck engine's `begin_round`/`collect_round` contract —
   each cycle round commits a detached fused bind+message CB and RamValCheck's
   synchronous CPU rounds run underneath it. Measured @2^24: **st4 −0.05 s
   alone**, exposed RegRW CB wait 0.335 → 0.28 s and RamVal hidden.
2. **Arena reuse (`JOLT_REGRW_ARENA=0` kills):** entry CSRs become untyped
   byte slabs (one currency across the Indexed→Direct deref) and each bind's
   retired input slab recycles as a later round's output — the mmap
   first-touch fault tax (S11's 31% clawback) is paid once per slab, not
   every round. Fresh slabs take Direct-width **virtual** headroom (MAP_ANON
   address space is free) so the deref round fits without a growth remap.
   Measured @2^24: **st4 −0.07 s alone**, sync bind CB 0.213 → 0.148 s.

Combined @2^24 (clean window): **st4 0.697/0.685 → 0.545 s (−0.15 s,
−21.5%)**. Modeled @2^27: **−1.4 s conservative / −1.8 s central** (bar ≥1.2
✓; the naive −21.5% ratio transfer gives −1.18 — the component model is the
honest one because the fault-tax share of st4 is 2× larger @2^27, see below).
Commit `88bb15ea4` on `lane/metal-w14-st4bundle`.

## Receipts — @2^24 e2e, sha2-chain metal, chrome spans

One binary, kill-switch matrix, GPU-locked, 40 s cooldowns, FrBind probe
257.97 µs (<350 healthy), window probe 2^22 e2e 2.44 s. Clean-window quad
(walls 6.61–6.99 s, tight):

| config | st4 | RegRw::bind_run | RegRw::msg_run | RegRw::bind_msg_run | RamVal rounds |
|---|---:|---:|---:|---:|---:|
| B = both OFF (S11 baseline) ×2 | 0.697 / 0.685 | 0.211 / 0.215 | 0.124 / 0.119 | — | 0.159 / 0.162 |
| C = overlap only ×2 | 0.627 / 0.658 | ~0 | 0.030 / 0.036 | 0.248 / 0.245 | 0.156 / 0.161 |
| D = arena only | 0.627 | **0.148** | 0.117 | — | 0.158 |
| A = both ON | **0.545** | ~0 | 0.036 | **0.174** | 0.148 |

- **Mechanisms land exactly as priced.** Arena: sync bind CB −0.065 s — the
  S11-measured @2^24 clawback was +0.057 s, so slab reuse recovers the full
  fault tax plus the per-round munmap/remap churn. Overlap: message wait
  0.12 → 0.03 s exposed; the fused CB (0.246 exposed alone) shrinks to 0.174
  once arena removes the in-CB faults — the doors compose super-additively
  (−0.146 combined > −0.049 + −0.065).
- Confirming pair after the window degraded (walls 6.9/9.5 s): B5 0.711 →
  A5 0.636 st4; every same-window comparison across 12 runs, including the
  degraded ones, has A < B (A4 0.832 vs B3 1.095 / B4 1.361). Two mid-day
  runs (D2, A2 wall 9.6–9.7 s) hit an ambient window shift and are excluded
  per the standing same-window rule (their spans show RamVal CPU 0.16→0.32
  — host throttle, not the doors).
- Peak RSS @2^24: 13.16 (A) vs 13.15 GiB (B) — neutral at this scale.

## Model @2^27 (post-S11 st4 = 5.49 s)

- **Arena:** the fault tax is byte-proportional (S11: +0.057 @2^24 ↔ +0.89
  @2^27 modeled). Recovery measured ≥ the @2^24 clawback → **−0.89 s
  conservative, −1.0 s at the measured @2^24 ratio** (extra munmap churn
  scales with bytes too). Residual paid once: first touch of the two
  retained slabs (~11 GiB vs ~44 GiB/proof today).
- **Overlap:** prize = Σ_r min(RegRW CB_r, RamVal CPU_r); S11's @2^27
  per-round trace priced 0.75–0.9 s. Arena-shrunk CBs haircut the window →
  **−0.5..−0.7 s**.
- **Combined: −1.4 s (both conservative) .. −1.8 s** → st4 ~5.5 → ~3.9±0.2 s.
  The percentage transfer (−1.18 s) understates because the clawback is 16%
  of st4 @2^27 vs 8% @2^24.

## Scale-transfer / residency flag (for the orchestrator 2^27 ABBA)

Arena grows retained residency: the prepare slab (~entry_count×56 B ≈
12.8 GiB @2^27) plus one output arena (high-water ≈ 11.1 GiB at the deref
round) stay resident through the ~27 cycle rounds instead of munmapping
per round; both drop at the host transition. **Δ resident ≈ +5–6 GiB during
st4's cycle phase only**, vs the status-quo transient in+out peak (~18 GiB
at the deref round). Neutral @2^24; must be kill-switch ABBA'd at 2^27
before default-on per the w13 scale-transfer rule (72 GiB working set /
compressor interaction). Also @2^27 the r0 headroom slab is a ~21 GiB
**virtual** MTLBuffer (touched ≈ 11 GiB) — if `newBufferWithBytesNoCopy`
ever rejects it, `alloc_entries` falls back to exact sizing (deref-round
realloc = S11 parity, no regression).

## Soundness / parity

Scheduling + allocation only; no protocol content. Overlap: the engine
assembles the batched polynomial as an exact field sum in declaration order
— wire bytes cannot depend on scheduling (engine contract); the detached CB
encodes the same kernels/buffers as the in-tree fused probe. Arena: every
readable slot `[0, new_count)` is fully written by the bind kernel
(out_offsets are exact prefix sums of the message kernel's merge counts —
identical merge walks), host readbacks are offset-bounded, and struct pad
bytes are never read; a reused slab is therefore value-identical to a
zeroed one everywhere any kernel or host read lands. Failure paths: a
failed detached round never installs its plan — pre-bind state is intact
and the CPU fallback redoes the round (same contract as the fused probe).

## Gates

- metal suites (jolt-kernels + jolt-dory + jolt-eval): **412/412** (was
  411; +1 new overlapped-parity test).
- byte-diff ratchet `-p jolt-prover --features prover-fixtures`: **20/20**.
- `cargo clippy --all --features host --all-targets -- -D warnings`: clean;
  `cargo fmt` applied.
- E2e verify passed on all 13 benchmark runs (1×2^22, 12×2^24).
- New parity coverage: `registers_rw_overlapped_matches_reference` drives
  the engine's begin/collect contract vs the reference kernel and pins the
  detached CB schedule (log_t+1 CBs, log_t launches); existing parity tests
  pin arena-on (Idx handoff + field rounds) and arena-off (legacy schedule)
  with exact CB/dispatch counts.

## Mechanism (one paragraph)

`DeviceEntries` (typed per-representation buffers) collapsed into
`EntrySlab` (an `OwnedDeviceBuffer<u8>` + kind tag) with unsafe typed views
(both entry structs are repr(C) with explicit pad fields — no implicit
padding). `plan_bind` pulls the output from `spare` when it fits, else
allocates with Direct-width virtual headroom; `install_bind` retires the
input slab into `spare`. `bind_and_message` refactored into
`launch_bind_and_message` (encode + `commit().detach()`, plan rides in the
flight) + `collect_flight` (wait, install, sums) — the sync path calls both
back-to-back, `begin_round`/`collect_round` split them across the engine's
two phases so RamValCheck's CPU rounds fill the gap. Message-only round 0
detaches the same way; the final cycle bind (host transition) and address
tail stay synchronous.

## Discipline

- Timed budget: 13 e2e runs @2^24 total, of which the decision evidence is
  one clean-window quad (A/B/C/D + mirrored C/B) + one confirming pair;
  2 degraded-window runs discarded by the same-window rule. No 2^27 runs
  of any kind; no 2^25.
- All cargo under `/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock`;
  every GPU run + bench under `/tmp/jolt-metal-gpu.lock`; FrBind health
  before pairs; 40–180 s cooldowns.
- KernelId::ALL unchanged (83) — no kernels added; shaders untouched;
  commitment.rs untouched. No pushes; siblings' worktrees untouched.
