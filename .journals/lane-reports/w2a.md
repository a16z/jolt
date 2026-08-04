# W2A — st6a + st7 restructure (R1 early-anchor + R2 bytecode split + R3-lite)

**Status: IMPLEMENTED — kill gate cleared @2^24 (combined st6a+st7 −50.6% vs
trunk, gate −30%). §8b has results; §§1-7 below are the reviewed design (GO
2026-08-04 ~06:45 UTC with five amendments, all honored: trunk merged before
work — twice, W1B then W1D; 11-fixture byte-diff count confirmed; config axis
additive; st2–st5 walls reported explicitly; §3 kept current).**

Lane: gpu/util-w2a. Targets: BooleanityAddressPhase::prepare 2.88 s,
BytecodeReadRafAddressPhase::prepare 0.89 s (st6a, 0% GPU),
HammingWeightClaimReduction::prepare 1.86 s (st7, 0% GPU) — 2^27 instrumented.

## 0. The structural finding

All three prepares are the same primitive: the **one-hot pushforward**
`G_i[k] = Σ_{j: hot_i(j)=k} eq(r_cycle, j)` — an O(T·N) scatter of eq weights
into K-sized buckets, keyed by hot chunk indices that are pure witness data.
The pushforward is information-theoretically the right compression (one O(T)
add-only pass making all subsequent address rounds O(K)); W1A proved the
scatter shape loses on device. The winning move is not a better kernel — it is
**anchoring the eq point early enough that the build runs off the critical
path**, plus exact-regrouping splits for the parts whose points are forced.

## 1. Chosen route

Three independent pieces, ordered by prize; only R1 touches the protocol.

### R1 — booleanity early anchor + background build (protocol change, Option A)

Booleanity's reference cycle point is a **pure eq anchor**: the address-phase
input claim is identically zero (`BooleanityAddressPhaseInputClaims` resolves
nothing), no upstream claim lives at the point, it is never absorbed or drawn
("pure construction geometry" — stage6a/booleanity.rs). Today it is the
reversed stage-5 instruction cycle point solely because 6a runs right after
st5. **Any transcript-prior random cycle point works identically** — and the
codebase already exploits this: `RamHammingBooleanity::new(trace_dimensions,
stage1_cycle_binding)` (stage6b/batch.rs:396) anchors its booleanity-family
relation at the **stage-1 cycle binding**.

Change: booleanity (6a address phase + 6b cycle phase) anchors its reference
cycle at the reversed **stage-1 cycle binding** instead of stage-5's point.
`stage1_cycle_binding` is already a field of both `Stage6aBuildParts` and the
6b parts — the swap is which field the two `Booleanity*::new` calls read
(stage6a/batch.rs:78, stage6b/batch.rs:394), single-sourced across both
fronts.

Scheduling payoff: the anchor is known at stage-1 end; the packed
`SharedInstructionRows` are parked by stage-1's trace-record walk
(trace_record.rs:580). The prover orchestrator (prover.rs, between
`prove_stage4` and `prove_stage5`) Arc-clones the rows, builds the column
selectors inline (O(N) shape checks), and spawns `cycle_pushforward` on a
**dedicated capped pool (~4 threads)**; the result channel is parked in the
`ProofSession`. St6a's booleanity prepare takes the carry and joins (≈0 ms);
if the carry is absent (tests, exotic registries) it builds inline —
bit-identical values either way (exact field arithmetic, order-independent).
Window: st5 wall 14.65 s canonical @2^27 vs ~8–9 s capped build; @2^24 ~1.3 s
vs ~0.7 s. Join blocks at worst until the build finishes — never slower than
today's inline build minus overlap.

Protocol gate (scope-report §5 Option A, adapted): `JoltProtocolConfig` gains
a third axis, e.g. `booleanity_anchor: BooleanityAnchor { Stage5Instruction,
Stage1CycleV1 }`. `for_zk()` pins `Stage5Instruction` (legacy prover and all
zk callers keep it — proof.rs:453/509 construct via `for_zk`, so legacy moves
with the struct automatically). The modular non-zk prover defaults to
`Stage1CycleV1` via `ProverConfig` (benchmarks pick it up with no command
change). Verifier: fail-closed — accept `Stage1CycleV1` only when
`zk == Transparent`; reject the `V1 && BlindFold` combination before any
stage work; relation construction switches on the validated axis. zk/BlindFold
paths are untouched behaviorally (legacy anchor), so no BlindFold stage-config
changes.

### R2 — bytecode 4-early/1-late pushforward split (byte-identical)

`stage_pushforwards` (optimized/bytecode_read_raf.rs:182) builds **five
independent per-stage tables** `F_s(k) = Σ_{j:pc(j)=k} eq(r_cycle_s, j)`; the
6a gammas weight them only inside the round loop. Stages 1–4's cycle points
exist before st5; `pc_rows` are parked from stage-1's record walk. Split the
one shared walk into a background walk over stages 1–4 (same spawn site as
R1) and an inline stage-5-only walk at 6a prepare. Per-stage tables are
computed independently → identical field values → **byte-identical proof**, no
protocol gate. Residual inline cost ≈ 35–45% of the walk (1 pc load + 1 add +
touched-bookkeeping vs 1 load + 5 adds).

### R3-lite — HWCR on-the-fly tensor eq (byte-identical)

HWCR's build (optimized/hamming_weight_claim_reduction.rs:174) materializes
`eq_table(r_cycle)` — a full T-sized table (4.3 GB @2^27, at the 90 GiB
pressure tier, right after st6b's peak) — then streams it back through the
scatter. Replace with `eq[j] = E_out[j_hi]·E_in[j_lo]` computed on the fly
(two √T-sized tables, one extra mult per row), keeping HWCR's existing plain
`+=` scatter otherwise. Kills the alloc+fill+read; exact same field values →
byte-identical. NOTE: HWCR's r_cycle is **forced** (it is the 6b booleanity
opening's cycle part, stage7/verify.rs:181–196; its input claims live there)
— no early anchor exists for st7; this shape fix is the available lever.

## 2. Rejected routes

- **H1 as stated (address-first stateless T-scans, no pushforward).** Valid
  math: round r of the address phase is computable statelessly per cycle as
  `E_hi[a_hi]·(L[a_lo]²·χ_{a_bit}(X)² − L[a_lo]·χ_{a_bit}(X))` with L = eq
  over bound bits — the `H_i[j] = E_r[addr_i(j)]` gather shape, byte-identical
  round polys. But it re-scans all T cycles **per address round**: 8 × T × N
  × ~2 mults ≈ 7·10⁹ field mults @2^24 (vs a one-pass add-only build) —
  loses by an order of magnitude on raw ALU before any dispatch overhead.
  The pushforward IS the compression; recomputing it round-wise is strictly
  worse.
- **H1 variant, cycle-major (bind j before k) — mathematically dead, worth
  journaling.** On boolean j the k-sum collapses: ra one-hot ⇒ ra² = ra ⇒
  `Σ_k eq(r_a,k)(ra²−ra) ≡ 0` pointwise — a j-first sumcheck that folds k on
  the cube proves 0 = 0 (vacuous). Keeping k open during j-binding requires
  collision-aware state: after r binds, `Σ_k eq_a(k)·ra~(k,·)² =
  Σ_{p,q: a_p=a_q} w_p w_q E_a[a_p]` — the collision indicator breaks the
  H(H−γ) gather form and forces sparse per-column (k, weight) multisets ≈
  the K×T grid in disguise, O(T) sparse binds per round for 27 rounds. The
  existing cheap 6b cycle kernel works only because k is fully bound first.
  This kills lever-board item 2 as originally imagined; the campaign journal
  entry should record it.
- **Absorb booleanity's address phase into st7 to share HWCR's G tables.**
  The tables match (same polys, K-sized, at r_c6b) — but booleanity's cycle
  phase must then run *after* the new address rounds, i.e. a fresh 27-round
  post-st7 cycle batch with one member: unshared T-scale rounds, a third
  cycle point, extra opening claims. Net loss.
- **H2 (device pushforward, improved shape).** W1A: +26.2% with
  SIMD-bucket and sort+segmented-reduce shapes. Threadgroup-privatized
  K-buckets don't fit (256 buckets × Fr-accumulator × simdgroups ≫ 32 KB);
  presorted index arrays cost ~13 GB @2^27 (25 polys × T × u32) — dead at
  the pressure tier. After R1/R2 there is no residual worth a device port.
- **Full tensor/deferred reshape of HWCR's build (original R3).** Measured
  evidence against: booleanity's tensor+deferred `cycle_pushforward` (2.88 s)
  is *slower* than HWCR's naive build (1.86 s) on the same rows, selectors,
  and N — the per-block K×N clear/reduce/fmadd machinery isn't free on CPU.
  Only the eq-materialization elimination (R3-lite) is unambiguous.

## 3. Soundness argument (R1 — the only protocol change; CURRENT as implemented)

Statement: every one-hot `ra_i(k,j)` is boolean on the cube. Check:
`0 = Σ_{k,j} eq(r_a,k)·eq(r_c,j)·Σ_i γ^{2i}(ra~_i(k,j)² − ra~_i(k,j))` — a
random evaluation of `B(y_a,y_c) := Σ_{k,j} eq(y_a,k)eq(y_c,j)(ra~²−ra~)`,
which is the zero polynomial iff every ra_i is boolean everywhere (γ-batching
adds the standard union term). Schwartz–Zippel requires only that `(r_a,
r_c)` be sampled after the `ra` commitments are transcript-bound. The stage-1
cycle binding is drawn after stage-0's witness commitments are absorbed —
the same property as stage-5's point, and the same construction
`RamHammingBooleanity` already relies on. Fiat–Shamir order is unchanged: no
draw moves (the anchor is derived, never drawn); every round message is still
absorbed before the challenges binding it; the 6a draw sequence (bytecode
gammas → booleanity pad draws → gamma) is untouched. Soundness error: bound
unchanged term-for-term vs today. The proof self-describes the anchor axis
and the verifier validates it fail-closed before constructing relations, so
no cross-protocol replay: a V1 proof cannot be verified as legacy or vice
versa, and V1+BlindFold is rejected outright.

## 4. Opening-coupling analysis

**No claim, point, or wire-shape changes anywhere.** 6a booleanity still
outputs the single `BooleanityAddrClaim` intermediate (value differs); 6b
still outputs the same ra openings at (6a-bound address point, r_c6b); HWCR
still consumes the 6b booleanity opening + hamming + virtualization claims at
the same points and reduces to the same joint openings at (r_a7, r_c6b);
st8's joint opening is untouched. R1 changes only which transcript-prior
point feeds one eq factor: proof format identical, round/claim counts
identical, all values shift. The `JoltProtocolConfig` field addition changes
the proof preamble encoding for **both** provers simultaneously (legacy
constructs the shared struct via `for_zk`), so the byte-diff twins stay
aligned on the legacy arm.

## 5. Modeled win (honest, per component)

Sources: 2^27 instrumented attribution (journal), canonical anchors, W1A's
2^24 stock arm (st6a 303.4 ms + st7 314.1 ms = 617.5 ms combined).

| component | 2^24 (model) | 2^25 canonical (model) | 2^27 canonical (model) |
|---|---|---|---|
| st6a booleanity prep (R1) | −~225 ms (76% share of 6a prep) | 0.493 → ~0.13 with R2 | st6a 2.265 → ~0.35–0.55 |
| st6a bytecode prep (R2) | −~30–40 ms (24% share × ~55% hoisted) | (included above) | (included above) |
| st7 eq materialization (R3-lite) | −30–80 ms | 0.247 → ~0.19–0.22 | st7 2.072 → ~1.6–1.9 |
| **combined st6a+st7** | **617 → ~270–350 ms (−43% to −56%)** | **0.740 → ~0.32–0.35 (−53% to −57%)** | **4.337 → ~1.95–2.45 (−44% to −55%)** |

Kill gate is −30% @2^24: modeled −43% worst-case, dominated by R1 alone
(−36% from R1 by itself). The 9.16 s zero-GPU seam loses its st6a chunk
(~2.0–2.2 s canonical) from the front; st6b prepare (lane B) owns the rest.
2^27 st7 upside beyond the table (page-pressure relief from the 4.3 GB
alloc) is plausible but unmodeled — orchestrator's certification run will
show it. Risk watch: st2–st5 walls must not absorb the background build
(capped pool; gate: no other stage regressing >2% @2^24 A/B; fallback = move
spawn later or shrink the pool).

## 6. Files touched

- `crates/jolt-verifier/src/config.rs` — anchor axis + fail-closed rule.
- `crates/jolt-verifier/src/verifier.rs` — validation + anchor into builders.
- `crates/jolt-verifier/src/stages/stage6a/{batch,booleanity,verify}.rs`,
  `stage6b/batch.rs` — anchor parameter through `Stage6a/6bBuildParts`.
- `crates/jolt-prover/src/{config,prover}.rs`, `stages/stage6a.rs` — knob,
  spawn site (between stage4/stage5), wiring.
- `crates/jolt-kernels/src/optimized/booleanity.rs` — prepare consumes the
  parked masses carry; session key + capped-pool build helper.
- `crates/jolt-kernels/src/optimized/bytecode_read_raf.rs` — R2 split
  (`stage_pushforwards` generalized over a point subset). **Lane-B collision
  flag:** B owns the bytecode CYCLE port (Metal slot + slots file); this
  optimized file hosts both phases. R2 confines itself to
  `stage_pushforwards` + the address prepare; R2 is sequenced LAST and
  rebased after B merges to trunk if B lands first.
- `crates/jolt-kernels/src/optimized/hamming_weight_claim_reduction.rs` —
  R3-lite.
- `crates/jolt-prover/tests/byte_diff.rs` — pin `booleanity_anchor =
  Stage5Instruction` (legacy arm) in `derive_config_pinned`.
- New tests (see §7). No Metal shader or slot changes at all; no
  jolt-sumcheck changes; legacy touched only via the shared config struct.

## 7. Test plan

1. **R2/R3-lite are byte-exact:** byte-diff 19/19 green in `prover-fixtures`
   AND `prover-fixtures,metal` — the strongest possible parity evidence;
   land these behind that gate.
2. **R1:** byte-diff 19/19 stays green with the legacy anchor pinned (live
   twin comparison — no snapshot fixtures exist to regen). New tests:
   (a) V1 e2e prove+verify small-scale, both backends; (b) rejection: proof
   declaring `Stage1CycleV1 + BlindFold` → clean `ProtocolConfigMismatch`
   before stage work; V1 proof vs legacy-expecting verifier → mismatch;
   (c) background-vs-inline mass equality (spawned build == inline values);
   (d) 6a prepare fallback when the carry is absent.
3. muldiv legacy `host` + `host,zk` green (legacy behavior unchanged).
4. jolt-kernels 231 + new, jolt-dory 46; clippy `-D warnings` host /
   host,zk / metal; fmt.
5. e2e prove+verify at 2^22 and 2^24 in `prover-fixtures` and
   `prover-fixtures,metal` under the new default (V1).
6. A/B @2^24 under the bench lock (ABBA): kill gate combined st6a+st7 −30%,
   no other stage >+2% (explicitly watching st5 for build contention).
   Confirm @2^25 cool (≥3 min quiet, AC). No 2^27 from this lane.

## 8. Sequencing & ETA

R1 core (1–1.5 d) → A/B checkpoint → R3-lite (2–3 h) → R2 (0.5 d, lane-B
collision-gated) → 2^25 confirm + final report. Total ~2–2.5 days.

## 8b. Implementation results (2026-08-04)

### What landed (commits on gpu/util-w2a)

- `3c8b96680` R1 — `BooleanityAnchor` axis in `JoltProtocolConfig`
  (additive, appended field; `for_zk()` and legacy pin `Stage5Instruction`;
  `validate_proof_config` admits `Stage1CycleV1` only when both the proof and
  the build are transparent — fail-closed before stage work). Anchor source
  single-sourced in `booleanity_reference_cycle_source` (6a batch) + the 6b
  builder match; prover spawns `cycle_pushforward` at the reversed stage-1
  binding on a dedicated 4-thread pool between stages 4 and 5; the 6a prepare
  joins a validated session carry or rebuilds inline (identical values).
  `ProverConfig::derive` defaults to V1; byte-diff pins legacy.
- `71272d8d3` R3-lite — HWCR pushforward computes `eq(r,j) =
  eq(r[..hi],j_hi)·eq(r[hi..],j_lo)` on the fly; the T-sized eq_table
  materialization (4.3 GiB @2^27) is gone. Byte-identical.
- `82024b418` R2 — `stage_pushforwards_for` generalized over point subsets;
  stages 1–4 (all pre-stage-5 points, via the new single-sourced
  `bytecode_early_stage_points`) build on a background pool at the same spawn
  site; only stage 5's single-point walk stays on the 6a path. Byte-identical
  (per-stage lanes regroup exactly). Bench ablation knob
  `JOLT_BOOLEANITY_ANCHOR=legacy` added to modular_benchmark.
- Trunk merges: `3abf99069` (W1B) before work, `b3eb2e893` (W1D) before the
  final A/B — the gate comparison is against trunk `9b7a111ce`.

### Gate matrix (post-W1D merge, all green)

jolt-kernels 155/155 default + 236/236 metal (3 new tests:
background-vs-inline ×2 relations, stale-carry fallback ×2 arms inside them);
jolt-dory 46/46; jolt-prover 20/20 in `prover-fixtures` AND
`prover-fixtures,metal` (11 byte-diff fixtures wire-equal with legacy —
R2/R3-lite byte-exactness proven there with the carries live — plus the new
`anchor_v1` e2e: V1 proves+verifies on both CPU backends and a re-tagged
anchor fails verification = the axis is load-bearing); legacy muldiv 3/3
`host` + 3/3 `host,zk`; clippy `-D warnings` host / host,zk / metal; fmt.
Known flakes, both pre-existing and logged: (i) 1–3 nondeterministic nextest
"leaky" flags on byte-diff tests (legacy-anchored arms — no spawn runs there);
(ii) a guest-ELF race on `/tmp/jolt-guest-targets/muldiv-guest-/` when narrow
filters start several muldiv-guest tests simultaneously ("could not open elf
file" at ~0.55 s) — full-suite invocations passed 3/3.

### A/B @2^24 (bench lock, ABBA T-W-W-T, trunk binary `ffdd71ad4079…` vs
W2A binary `277ae5aa14ee…`, sha2-chain, metal)

| stage | trunk mean | W2A mean | Δ |
|---|---:|---:|---:|
| st0 | 2.623 | 2.626 | +0.1% |
| st1 | 1.073 | 1.052 | −2.0% |
| st2 | 0.604 | 0.589 | −2.4% |
| st3 | 0.288 | 0.290 | +0.6% |
| st4 | 1.459 | 1.452 | −0.5% |
| **st5** | **1.726** | **1.803** | **+4.5%** |
| **st6a** | **0.207** | **0.038** | **−81.6%** |
| st6b | 1.203 | 1.173 | −2.5% |
| **st7** | **0.219** | **0.173** | **−21.2%** |
| st8 | 1.205 | 1.165 | −3.3% |
| **prove** | **11.627** | **11.325** | **−2.6%** |

**Combined st6a+st7: 0.426 → 0.211 = −50.6% (kill gate −30%: PASS).**
Decomposition (earlier same-tree R1-only ABBA): R1 alone took st6a
0.199→0.114 (−42.5%); R2 takes it to 0.038; R3-lite accounts for the st7 drop
(trunk st7 runs varied 0.180–0.258, W2A's 0.171–0.174 are tight — honest st7
read: −10…−30%).

**st5 watch (amendment 4) → root-caused and FIXED (`fc3b87367`):** the first
A/B showed st5 +4.5% at 2^24 AND 2^25 — two concurrent 4-thread background
pools (R1's booleanity + R2's bytecode; the R1-only ABBA had measured +2.4%
with one). A process-wide token now serializes the background builds (at most
one capped pool competes with the foreground stage; prepares only join
handles, never lock — no deadlock; poisoned token → inline rebuild).

### FINAL A/B @2^24 (tokened, W-T-W-T, cool, cross-run variance ≤ ±2%;
trunk `9b7a111ce` binary `ffdd71ad…` vs W2A binary `f5b7ad7c75ce…`)

| stage | trunk mean | W2A mean | Δ |
|---|---:|---:|---:|
| st0 | 2.618 | 2.630 | +0.4% |
| st1 | 0.895 | 0.887 | −0.9% |
| st2 | 0.515 | 0.509 | −1.2% |
| st3 | 0.210 | 0.212 | +0.9% |
| st4 | 1.218 | 1.200 | −1.4% |
| **st5** | **1.712** | **1.730** | **+1.1%** ✓ (≤2% gate) |
| **st6a** | **0.228** | **0.033** | **−85.7%** |
| st6b | 1.230 | 1.131 | **−8.1%** (consistent bonus: guaranteed-parked shared scans) |
| **st7** | **0.190** | **0.170** | **−10.7%** |
| st8 | 1.123 | 1.122 | −0.1% |
| **prove** | **10.895** | **10.562** | **−3.1%** |

**Combined st6a+st7: 0.418 → 0.203 = −51.6%. Kill gate −30%: PASS. No stage
above +2%: PASS.**

### 2^25 confirm (tokened, W-T-T-W, ≥3 min cool + AC; second pair thermally
inflated ~8–10% — the box ramps ~2 s/run at this scale, cool pair is the
signal)

- st6a: T 0.530/0.723 → W 0.052/0.064 (**−90.7%** on means; canonical anchor
  for context: 0.493)
- st7: T 0.229/0.275 → W 0.208/0.250 (−9.1%)
- combined: 0.879 → 0.287 = **−67.3%**
- st5: cool pair +1.4% ✓; all-runs mean +3.3% is thermally confounded (the
  hot-slot W2 run inflates st0 +0.8% and st8 +14% alike)
- st6b: −4.8% (bonus holds); prove: cool pair −0.8%
- 2^24/2^25 absolute walls this session run ~15–20% above the campaign's
  canonical anchors (22-day-uptime box after a full bench day) — the A/B is
  same-session interleaved, so the relative numbers stand.

### Handoff notes for wave-close certification (2^27, orchestrator-run)

- Expected @2^27 canonical: st6a 2.265 → ~0.3–0.5 (booleanity 2.88-instr
  build fully off-path; bytecode's stage-5-only walk + join residual remain);
  st7 2.072 → the R3-lite question mark — the 4.3 GiB eq_table alloc is gone,
  which the 2^24/2^25 scales price at only −9…−11% but the 90-GiB pressure
  tier may reward much harder (W4 U1 mechanism). Watch the 9.16 s seam:
  the st6a chunk of it should collapse.
- The background builds hold the token for ~8–9 s (booleanity, 4 threads)
  + ~2–3 s (bytecode) inside st5's ~14.6 s window at 2^27 — fits serially;
  if st5 shows contention at 2^27, the lever is pool width (constants
  `BOOLEANITY_BACKGROUND_THREADS` / `BYTECODE_BACKGROUND_THREADS`).
- Flakes (pre-existing, worth an infra lane eventually): the
  `/tmp/jolt-guest-targets/muldiv-guest-/` ELF race fails ~1 in 6 full-suite
  runs at ~0.5–1.0 s ("could not open elf file") — always passes isolated
  and on retry; nondeterministic nextest "leaky" flags on byte-diff tests.
- zk stays entirely on the legacy anchor/path: `for_zk()` pins it, the
  verifier rejects V1+BlindFold fail-closed, BlindFold code untouched.

### Retained state

Commits: `3c8b96680` (R1) → `71272d8d3` (R3-lite) → `82024b418` (R2) →
`b3eb2e893` (trunk/W1D merge) → `fc3b87367` (token). Binary sha-256
`f5b7ad7c75ce1ffba3748a186140e5738d8b49c17a6309114ae60915bc6c17de`
(`prover-fixtures,metal` release, modular_benchmark). Artifacts:
`/tmp/w2a-{s24,final,tok,s25,tok25}-*.{log,json}` + `/tmp/w2a-stages.py`.
