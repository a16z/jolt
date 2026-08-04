# W1.5 scope — paired sumcheck rounds on Metal

## Verdict

**GO for one narrow, non-zk A/B: Stage 4 `RegistersRW`, pairs `(0,1)`, `(2,3)`, `(4,5)` only; leave round 6 single and never cross the round-6/7 join. Do not implement generic/broad pairing yet. Confidence: 75%.**

Why this slice:

- The normal Metal slot is already fused: it folds the previous challenge and accumulates the next univariate in **one command buffer / one synchronization** (`crates/jolt-kernels/src/metal/slots/mod.rs:14-25`). Generic pairing therefore primarily removes host↔GPU synchronization; it does not automatically remove a standalone evaluation pass because none exists.
- `RegistersRW` is the material exception. `message()` runs and waits in its own pass (`metal/slots/registers_read_write.rs:117-179`); `bind()` performs host prefix-scan/allocation, then runs and waits in a second pass (`:182-313`); every post-first round calls both (`:630-695`). Pairing can replace 13 pre-tail passes across rounds 0..6 with about 7, avoid the intermediate half-size sparse representation, and amortize the host compaction boundary.
- Its stable device-only prefix consumes **5.862 s** at 2^27 with only **1.789 GPU%-equivalent seconds / 30.5% sampled utilization** and **3.730 s sampled at 0% GPU**. This is the only measured slice where extra degree-3 ALU has clear stall headroom and the implementation can remove real sparse-table traffic.
- Modeled A/B win: **1.2–1.8 s at 2^27**, or **9–14% of Stage 4's 13.248 s**. Model: 13→7 command buffers across the seven-round prefix, discounted from the raw 46% pass reduction to a 20–30% net prefix reduction for doubled d3 message ALU and unchanged host compaction (`5.862 s × 20–30% = 1.17–1.76 s`). Treat this as a gate: keep pairing only if the 2^24 and 2^27 A/Bs confirm it.

Broad rollout is a no-go on current evidence: Stage 3 exposes only 54 ms of pair-selected boundary gap; Stage 5 only 5 ms and its device-only prefix is 94.2% GPU%-equivalent; Stage 6b only 0.14 ms, while its CPU bytecode member and degree-5 instruction member make paired messages substantially more expensive.

## 1. Current round-loop anatomy

### Host protocol

```text
generated stage driver
  begin_batch → prepare all members                         driver.rs:474-519
  prove_batch
    for round r:
      launch prove_round(previous challenge) for active members
                                                               prover.rs:232-257
      collect each launch / obtain its univariate               prover.rs:264-284
      RLC-combine → assert p(0)+p(1)=claim                       prover.rs:285-305
      transcript absorb one round → sample one challenge        prover.rs:306-317
    finish_rounds(last challenge)                               prover.rs:320-325
```

`ProveRounds` deliberately passes the previous challenge into the next call (`crates/jolt-sumcheck/src/prover.rs:44-50`). Batch membership is an interval `[offset, offset + num_rounds)`; default members are tail-aligned, with explicit head alignment for special members (`crates/jolt-sumcheck/src/batch.rs:13-29`). The verifier mirrors exactly one compressed univariate and one transcript challenge per variable (`crates/jolt-sumcheck/src/verifier.rs:87-131`).

The generated batch head absorbs input claims and samples the RLC coefficients before entering the loop (`crates/jolt-verifier-derive/src/lib.rs:466-513`). Thus a paired message must be formed after the same batch prelude and before **both** pair challenges.

### Normal Metal slot: fused

For the normal slot, one round call encodes one pass, dispatches once, waits once, then sums unified-memory partials on the host. The contract is explicit in the slot header (`crates/jolt-kernels/src/metal/slots/mod.rs:14-32`); runtime `run()` commits and waits (`crates/jolt-kernels/src/metal/runtime.rs:491-527`). A representative implementation binds and evaluates in the same pass (`metal/slots/instruction_input.rs:246-282`, `:374-445`); the async read-RAF slot retains the same fused device work behind launch/collect (`metal/slots/instruction_read_raf.rs:565-625`, `:760-825`).

```text
round r call
  GPU: cur --bind(r-1)--> nxt  + accumulate p_r evaluations  [one CB]
  CPU: wait/read unified partials → exact sum → transcript → r_r
```

Consequences:

1. Pairing halves transcript/readback synchronizations in a pair.
2. Pairing does **not** inherit a free 40% traffic win on these slots: evaluation and fold are already fused. Avoiding the intermediate `n/2` table requires a new bind-by-two kernel and layout, not merely a bivariate proof type.
3. Unified memory removes an explicit download, but not the wait or host partial reduction (`metal/slots/mod.rs:27-32`, partial summation at `:145-180`).

### `RegistersRW`: unfused special case

```text
round 0: message(n) → wait/read counts+partials
round r>0:
  host scan prior counts + allocate compacted sparse table
  bind(n, challenge) → wait
  message(n/2)       → wait/read counts+partials
```

This slot is not the fused pattern advertised by the common header. The separation is structural: `message()` produces counts (`registers_read_write.rs:117-179`), `scanned_offsets()` reads them on the host (`:182-199`), and `bind()` allocates/dispatches the compacted output (`:201-313`). This is the traffic/synchronization target; a paired sparse kernel can consume four rows per output and bind two challenges directly.

## 2. 2^27 measurements

Source: `/tmp/metal-m5-gputrace-2to27-20260803.json.gz`, 333 `sumcheck_round` spans and 1,637 `gpu_percent` samples. Stage walls agree with the campaign journal. `gpu_percent` is sampled at about 10 Hz, so the integral below is **utilization-equivalent time**, not exact kernel-active time. “Boundary gap” is the exact trace lower bound from the end of the last Metal member span in round `r` to the start of the first Metal member span in `r+1`; it includes exposed CPU-member/transcript/launch delay, but excludes device-idle time hidden inside a wait/readback span.

| Stage batch | Rounds | Round-loop wall | Median / p90 / max round | GPU%-equiv. | 0%-sampled wall | All device boundary gaps | Gaps selected by legal adjacent pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| st3 | 27 | 2.473 s | 1.731 / 127.204 / 892.778 ms | 0.506 s (20.5%) | 1.941 s | 0.056 s | **0.054 s** |
| st4 | 34 | 8.412 s | 11.931 / 714.761 / 2,250.382 ms | 2.393 s (28.4%) | 5.267 s | **1.539 s** | **0.841 s** |
| st5 | 155 | 15.868 s | 0.221 / 586.156 / 4,807.452 ms | 10.021 s (63.1%) | 5.230 s | 0.031 s | **0.005 s** |
| st6b | 27 | 10.241 s | 7.344 / 874.417 / 4,136.250 ms | 3.836 s (37.5%) | 5.803 s | 0.000287 s | **0.000140 s** |

For the three requested representative mid/healthy batches, the exact exposed device-boundary lower bound is **1.570 s total** (st4 1.539 + st5 0.031 + st6b 0.000287); the deterministic legal pairing schedule selects **0.846 s** of those gaps. The much larger 16.300 s of 0%-sampled wall is not all pair-removable: it includes CPU members, unified-memory/page pressure, device waits, and long kernels sampled between ticks.

### Representative rounds

`GPU-eq` and `0%-wall` are counter integrals over the round; `next gap` is exact by the definition above.

| Stage / round | Wall | GPU-eq | 0%-wall | Next device gap | Trace anatomy |
|---|---:|---:|---:|---:|---|
| st4 r0 | 392.608 ms | 133.658 ms | 230.966 ms | 0.015 ms | Registers 392.596 ms |
| st4 r1 | 2,250.382 ms | 453.264 ms | 1,780.638 ms | 0.019 ms | Registers 2,250.368 ms |
| st4 r7 | 701.168 ms | 300.217 ms | 320.550 ms | **494.551 ms** | Registers 206.621 + CPU RamVal 494.521 ms |
| st4 r8 | 818.586 ms | 86.028 ms | 620.324 ms | **522.193 ms** | Registers 296.397 + CPU RamVal 522.163 ms |
| st4 r9 | 407.236 ms | 46.772 ms | 338.454 ms | **219.417 ms** | Registers 187.822 + CPU RamVal 219.392 ms |
| st5 r8 | 619.885 ms | 351.620 ms | 260.027 ms | 0.013 ms | InstrRead launch/collect envelope 619.868 ms |
| st5 r120 | 593.430 ms | 578.023 ms | 0 | 0.009 ms | InstrRead 593.417 ms |
| st5 r128 | 4,807.452 ms | 581.849 ms | 4,165.335 ms | 0.043 ms | InstrRead 4,291.040 + CPU RegVal 515.870 ms |
| st5 r141 | 26.048 ms | 14.847 ms | 0 | 22.465 ms | InstrRead 3.586 + CPU RamRA 21.143 + RegVal 1.294 ms |
| st6b r0 | 2,246.789 ms | 1,028.454 ms | 1,133.612 ms | 0.018 ms | CPU Bytecode 1,583.073; Metal RamH 204.291 + Inc 439.457 ms |
| st6b r3 | 4,136.250 ms | 1,342.262 ms | 2,586.463 ms | 0.018 ms | CPU Bytecode 713.897; Metal InstrRA launch 1,899.234 ms among five device members |
| st6b r13 | 158.169 ms | 94.678 ms | 0 | 0.015 ms | Metal RamRA launch 144.693 ms dominates |

Idle location:

- **st4:** the exposed 1.539 s sits after the Registers device member, mainly in CPU `RamVal` before the next Registers launch. More importantly, the 5.862 s device-only prefix hides two separate waits and a host count scan inside each Registers span; the trace lacks Metal GPU start/end events, so the counter is the available evidence for that internal stall.
- **st5:** the device-only prefix is effectively busy (9.147 s wall, 8.613 GPU-eq, 94.2%). In the mixed tail, CPU `RegVal` is mostly overlapped between asynchronous InstrRead launch and collect. Only 31 ms is exposed after the final device span, and the legal pair parity captures 5 ms.
- **st6b:** CPU Bytecode and five Metal members are serialized/overlapped inside each round, but another Metal member reaches almost to the next round. Only 0.287 ms is exposed at the boundary. Pairing would increase all members' message work rather than remove the dominant Bytecode/Metal work.

## 3. Batch membership and legal pair ranges

The stage calls are `Stage*::prove` → generated `prove_batch`, not a concrete `BatchedSumcheck::prove`: st1 `crates/jolt-prover/src/stages/stage1.rs:71-96`, st2 `stage2.rs:115-152`, st3 `stage3.rs:59-82`, st4 `stage4.rs:156-190`, st5 `stage5.rs:75-97`, st6b `stage6b.rs:118-181`.

| Stage | Trace-stable member ranges at 2^27 | Legal pairing segments | Metal/CPU mixture |
|---|---|---|---|
| st1 | 0..27 OuterRem | 0..27 | Metal only |
| st2 | 0..12 RamRW; 13..26 +ProductRem+InstrClaim; 27..39 +RamRAF+RamOutput | each range separately | mixed only 27..39 (`RamOutput` CPU) |
| st3 | 0..26 Shift+InstrInput+RegClaim | 0..26 | mixed all rounds; only InstrInput Metal |
| st4 | 0..6 RegistersRW; 7..33 +RamVal | **0..6**, 7..33 separately | Metal-only prefix; mixed tail (`RamVal` CPU) |
| st5 | 0..127 InstrReadRAF; 128..154 +RamRAClaim+RegVal | each range separately | Metal-only prefix; mixed tail |
| st6a | 0..5 Bytecode address; 6..13 +Booleanity address | each range separately | CPU-only in this trace baseline |
| st6b | 0..26 six members | 0..26 | mixed all rounds; Bytecode cycle CPU; five Metal |
| st7 | 0..7 HWCR | 0..7 | Metal only in this trace; optional address members would be CPU |

Member definitions: st1 `stages/stage1/outputs.rs:37-44`; st2 `stage2/outputs.rs:44-55`; st3 `stage3/outputs.rs:31-37`; st4 `stage4/outputs.rs:38-41`; st5 `stage5/outputs.rs:23-27`; st6a `stage6a/outputs.rs:44-47`; st6b `stage6b/outputs.rs:67-92`; st7 `stage7/outputs.rs:34-52`.

Rules:

1. Never pair across a join: e.g. st4 r6/r7 is invalid because the pair would have different RLC member sets.
2. A CPU member in either round must emit the **same bivariate message**; pairing only the Metal member changes the batch identity and is unsound.
3. Device-failure fallback must also implement the bivariate path. A paired proof cannot silently revert one member to two univariates after the pair transcript has begun.
4. Optional members alter segment geometry. The proof/config must carry a deterministic pairing profile from actual batch metadata, not hard-code the observed trace's absent optionals.

## 4. Degree and cost inventory

For degree `d`, the dense paired message has `(d+1)^2` grid values versus `2(d+1)` values for two univariates. Raw evaluation-count ratio: d2 **1.5×**, d3 **2×**, d5 **3×**, d6 **3.5×**. Compression can omit one dependent corner, but does not change the ALU ranking.

| Stage | Member degrees | Pairing assessment |
|---|---|---|
| st1 | OuterRem d3 | stall-bound stage, but normal fused slot; measure only after st4 proves the mechanism |
| st2 | RamRW d3; ProductRem d3; InstrClaim d2; RamRAF d2; RamOutput d3 | modest gap; mixed tail requires CPU bivariates |
| st3 | Shift d2; InstrInput d3; RegClaim d2 | only 54 ms selected gap; no implementation case |
| **st4** | **RegistersRW d3; RamVal d3** | **best case: unfused sparse Metal prefix, d3, 30.5% GPU-eq** |
| st5 | InstrReadRAF **d6**; RamRAClaim d2; RegVal d3 | reject initially: prefix 94.2% GPU-eq and d6 costs 3.5× raw evaluations |
| st6a | Bytecode address d3; Booleanity d3 | CPU baseline; unrelated lane may change placement |
| st6b | Bytecode d3; Booleanity d3; RamH d3; RamRA d3; InstrRA **d5**; Inc d2; optional precommitted d2 | reject initially: near-zero exposed gap, CPU member, d5 costs 3× |
| st7 | HWCR/optional members d2 | too small to matter |

Degree sources include Spartan constants (`crates/jolt-prover/src/geometry/spartan.rs:18-20`), instruction constants (`geometry/instruction.rs:21-22`), RegistersRW (`subprotocols/registers/read_write_checking.rs:89-91`), RamVal (`subprotocols/ram/val_check.rs:129-131`), InstrReadRAF (`subprotocols/instruction/read_raf.rs:84-86`), and InstrRA virtualization (`subprotocols/instruction/ra_virtualization.rs:73-75`). At logT=27 the config uses committed chunk 8 and virtual chunk 32 (`crates/jolt-prover/src/config.rs:145-163`), producing the d6/d5 cases above.

The one-round uniskips are separate protocols—outer d27/domain 10 and product d6/domain 3 (`geometry/dimensions.rs:14-17`)—whose output claims feed later batches. They are not pairable with the first remainder round without a larger protocol redesign.

The campaign's healthy stages st0 (92%) and st8 (86%) are not these sumcheck batches. The relevant warning is st5's measured 94.2%-equivalent InstrRead prefix: paired d6 ALU is likely to swamp any saved sync.

### Ranked 2^27 opportunity

| Rank | Slice | Pair-selected exposed gap | Expected net stage win | Decision |
|---:|---|---:|---:|---|
| 1 | **st4 Registers rounds 0..6** | 0.042 ms; gain is inside its unfused bind/message spans | **1.2–1.8 s** from sparse pass/traffic reduction | implement A/B |
| 2 | st4 mixed rounds 7..33 | 0.841 s | 0–0.4 s after extra CPU RamVal d3 work | defer until prefix result |
| 3 | st3 | 0.054 s | ≤0.05 s before extra d2/d3 ALU | no-go |
| 4 | st2 | 0.00136 s | ≤0.01 s | no-go |
| 5 | st1 | 0.00010 s | ≤0.01 s | no-go |
| 6 | st5 | 0.005 s | likely negative: d6 prefix already 94.2% GPU-eq | no-go |
| 7 | st6b | 0.00014 s | negative risk: CPU member plus d5 member | no-go |

The st4 tail's 0.841 s is an upper bound on synchronization-only savings, not a forecast: nearly all of it is CPU RamVal work that a pair must replace with a larger bivariate computation.

## 5. Proof, transcript, and zk blast radius

### Transparent proof changes

Current clear serialization is one `Vec<UnivariatePoly>` / compressed `Vec<CompressedPoly>` (`crates/jolt-sumcheck/src/proof.rs:26-61`). A compressed univariate omits its dependent linear coefficient and absorbs a fixed label/count (`round_proof.rs:102-136`). The recorder compresses one round and squeezes one challenge (`recorder.rs:49-58`, `:118-129`). A paired profile therefore needs:

- a sumcheck message enum (`Uni`, `BivariateGrid`) or an arity-tagged message vector;
- a distinct transcript label plus dimensions/count, followed by **two** challenge squeezes;
- verifier enforcement of exact degree, arity, schedule, batch-segment boundary, and total variables;
- paired corner reconstruction/compression and tensor-Lagrange evaluation at `(r_i,r_{i+1})`.

Do not reinterpret an old vector by length. Make the proof self-describing, then require its profile to equal the verifier's expected `JoltProtocolConfig`.

### Option A — recommended: pairing off in zk mode

`JoltProtocolConfig` currently contains only `zk` and commitment and validates by exact equality (`crates/jolt-verifier/src/config.rs:36-53`, `:73-84`); the proof carries it (`crates/jolt-verifier/src/proof.rs:53-72`) and verification checks it (`crates/jolt-verifier/src/verifier.rs:293-303`). Add a versioned sumcheck profile, e.g. `Single` / `MetalRegistersPrefixV1`, to the config and the transcript preamble (`verifier.rs:570-627`). Reject `profile != Single && zk` before proof parsing/commitment work.

Config plumbing estimate: **5–8 constructors/validation/preamble/test sites, under one engineering day alongside the core work**. The paired proof self-describes for decoding, but fail-closed equality—not the proof's own claim—authorizes it. `host,zk` remains `Single`.

### Option B — full BlindFold support: not part of this cut

Committed sumcheck currently has one commitment/degree and one coefficient row per round (`crates/jolt-sumcheck/src/committed.rs:17-55`, `:176-180`, `:230-260`). BlindFold R1CS allocates one power-of-two coefficient row per univariate (`crates/jolt-blindfold/src/r1cs.rs:164-225`, `:266-289`) and constrains a univariate Boolean sum/evaluation (`crates/jolt-sumcheck/src/r1cs.rs:150-180`, `:283-295`). Legacy orchestration likewise assumes one coefficient row and one challenge per sumcheck round (`crates/jolt-prover-legacy/src/subprotocols/blindfold/mod.rs:241-308`).

Full support means a new committed pair message, `(d+1)^2` coefficient witnesses/blindings, two challenges per row, bivariate corner-sum/evaluation constraints, layout changes (d3 row 16; d5/d6 padded row 64), legacy conversions, and fixture/test rewrites. Estimate: **15–25 files and 1–2 engineering weeks**, with materially more proving memory. It has no justification for the first Metal A/B.

## 6. Smallest sound implementation cut

### Protocol

For each configured pair, prover commits/absorbs a bidegree-`≤d` polynomial `g(X,Y)` before sampling either challenge:

```text
g(0,0) + g(0,1) + g(1,0) + g(1,1) == previous_claim
(r_i, r_{i+1}) = transcript.challenge(), transcript.challenge()
next_claim = g(r_i, r_{i+1})
```

Schwartz–Zippel error is at most `2d/|F|` per pair, equal to the union bound for two degree-d univariates. RLC batching is unchanged. The pairing profile and proof message shape must be transcript-bound and verifier-validated; the pair message must be fixed before both Fiat–Shamir challenges.

For the first A/B, enable only st4 Registers' stable device-only prefix when its initial table exceeds the existing Metal gate. Schedule `(0,1)`, `(2,3)`, `(4,5)`, then r6 single. The CPU optimized twin implements the same grid for fallback; the st4 mixed tail and every other stage remain single-round.

### Files touched

| Area | Minimum likely files |
|---|---|
| Sumcheck wire/loop | `crates/jolt-sumcheck/src/{batch,prover,round_proof,proof,recorder,verifier}.rs`; possibly one bivariate type in `jolt-poly` |
| Generated batch API | `crates/jolt-verifier-derive/src/lib.rs` |
| Protocol fail-closed config | `crates/jolt-verifier/src/{config,proof,verifier}.rs`; `crates/jolt-prover/src/prover.rs` and st4/profile plumbing |
| CPU fallback | optimized RegistersRW prover/kernel |
| Metal | `crates/jolt-kernels/src/metal/slots/registers_read_write.rs`, its `.metal` shader, `metal/runtime.rs` kernel IDs |
| Tests/fixtures | sumcheck message/verifier tests, Registers CPU↔Metal parity, protocol-config rejection, e2e/serde fixtures |

No `committed.rs`, BlindFold R1CS, or legacy BlindFold changes beyond an explicit unsupported-combination rejection.

### A/B gate and test plan

1. **Algebra/unit:** paired grid equals two sequential binds; four-corner sum; tensor evaluation; RLC inactive constant is `claim/4`; odd tail and join-crossing rejected.
2. **Backend parity:** optimized/Metal Registers grid parity; direct bind-by-two parity; forced gate and injected mid-proof device failure exercise the CPU paired fallback; assert expected dispatch-count reduction.
3. **Malformed proof:** reject wrong arity/grid/degree/schedule, a pair across r6/r7, config mismatch, and every `paired + zk` combination.
4. **E2E:** paired Metal prove+verify at 2^22 and 2^24; then one locked 2^27 A/B. Keep `host`, `host,zk`, `muldiv host`, and `muldiv host,zk` on `Single` and green.
5. **Fixtures:** preserve existing byte fixtures for `Single`; version/key new fixtures by protocol profile. Within one paired profile, CPU/Metal should remain field-exact. Do not require byte equality across `Single` and paired protocols.

**Acceptance rule:** proceed beyond st4 only if paired st4 saves at least 10% stage wall at 2^24 and 2^27 without increasing end-to-end proof size/memory enough to move another stage. Otherwise delete/disable the profile; the broad trace case does not support rollout.
