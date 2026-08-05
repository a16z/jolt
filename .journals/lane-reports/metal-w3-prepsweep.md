# Metal W3 prepare sweep

**Status: retain st6b IncCR; reject st5 row-bucketing follow-up.** Isolated
lane only: no end-to-end prover run.

## Inventory

Source of record: `monitor-2to27-20260804.json`, paired Chrome `B`/`E`
events. Times are inclusive and nested rows are not additive. “Current”
dispositions account for the W2/W3 changes already present on
`scratch/metal-saturation`; when no fresh post-adoption span exists, the
table labels the value as modeled. The cutoff is 0.2 s at `2^27`.

| stage | prepare / setup / materialization | `2^27` | execution shape | GPU / overlap disposition |
|---|---|---:|---|---|
| st0 | `TraceRecord::collect` | 13.337 s | Rayon-parallel, eight-thread background pool | Challenge-independent and already hoisted into the st0 commitment window by W3D; `TraceRecord::join = 0`, so exposed wall is zero. Further GPU work would compete for the same memory fabric. |
| st0 | `DoryScheme::prepare_tier2` | 0.457 s | Rayon-parallel G2 preparation | Already runs beside the stage-0 streaming commitment work; no exposed join in the trace. |
| st1 | `SpartanOuterUniskip::prepare` | 1.994 s | Metal `OuterT1`; host dispatch + synchronous wait | Data-parallel core already on GPU. Point depends on the st0 transcript, so no earlier-stage hoist. |
| st1 | `OuterRemainder::prepare` | 1.659 s | Metal `OuterAzbz`; host dispatch + synchronous wait | Data-parallel core already on GPU; depends on the uniskip challenge. |
| st1 | closing `claimed_inputs_from_record` + eq setup | ~0.37 s current; 0.954 s trace | Rayon host walk in trace; W2 Metal claims + W3D split-eq adopted | Already cut. W2's isolated ratio modeled 0.468 s removed; W3D also removed the full-T eq table. No independent previous-stage tail after the final point exists. |
| st2 | `SpartanProductUniskip::prepare` | 0.538 s | mixed host setup + Metal `ProductT1` | Dominant data-parallel pass already on GPU; point arrives at st1 end. |
| st2 | `RamReadWriteChecking::prepare` | 0.501 s | mixed Rayon setup + Metal table path | Already mixed/offloaded; independent members prepare serially in the batch driver, so a detached launch is possible only with lifetime plumbing. |
| st2 | `RamRafEvaluation::prepare` | 0.392 s | mixed; includes ~0.160 s Rayon eq setup | Device path already present; remaining host setup is below the 0.4 s modeled-save bar. |
| st2 | `ProductRemainder::prepare` | 0.317 s | mixed host setup + Metal `ProductLr` | Core already on GPU; below the modeled-save bar. |
| st2 | post-round product/ICR claim materialization | ~1.0–1.3 s current; 2.148 s trace | Rayon host walks | W3D split-eq + unreduced accumulators already landed, modeled 0.9–1.15 s removed. Final points prevent overlap with earlier GPU work. |
| st3 | `SpartanShift::prepare` | 0.657 s | Rayon-parallel Q-table build | Data-parallel and GPU-suitable. A st2-tail hoist needs a five-column gamma decomposition; prior analysis caps the likely win near 0.3 s, below this lane's 0.4 s bar. |
| st3 | `InstructionInput` round-0 setup/message | 0.455 s trace | Rayon host weighted reduction | Already ported to `jk_instr_input_q0`; isolated reduction 88–89%, modeled 0.401 s removed. |
| st4 | `RegistersReadWriteChecking::prepare` | 2.449 s | host scan/allocation plus Metal tables | **Excluded: sibling W3 lane owns this function.** Largest remaining exposed prepare. |
| st5 | `InstructionReadRaf::prepare` | 1.371 s | serial stable row buckets, Rayon eq, first Metal phase scan/wait | Mixed dependency chain. The serial bucket itself is only ~0.30 s modeled and a stable Rayon version saved 20.6–23.9%; rejected below. The eq table and first scan cannot simply overlap because the scan consumes both. |
| st5 | `RegistersValEvaluation::prepare` / witness `materialize_cycle` | 0.371 / 0.370 s | witness materialization, Rayon-capable | Data-parallel but below the 0.4 s absolute bar even at a hypothetical 100% cut; no cut. |
| st6a | `BytecodeReadRafAddressPhase::prepare` | 0.225 s trace; stage now 0.189 s | Rayon/background preparation | W2A already moved independent selector work into the post-st4 background window. Below bar. |
| st6b | `BytecodeReadRafCycle::prepare` | 0.281 s | mixed host setup + Metal combined init | Already offloaded; below bar. |
| st6b | `IncClaimReduction::prepare` | **1.790 s** | four full eq/table builds plus two raw-column materializations | **Cut in this lane:** direct Metal paired-eq weights, detached and overlapped with the independent host column walks. |
| st7 | `HammingWeightClaimReduction::prepare` | ~1.106 s current; 1.887 s trace | Rayon one-hot pushforward | W2 split-eq pushforward already reduced the slice 41.37% (modeled 0.781 s removed). Prior Metal scatter prototype lost; final st6b point prevents a hoist. |
| st8 | `DoryScheme::combine_hints` | 0.532 s trace | data-parallel combination | Metal hook already adopted. |
| st8 | `JoltG2Routines::fixed_base_vector_scalar_mul` | 0.441 s | device routine + host wait | Already on device. |
| st8 | `BN254::multi_pair_{g1,g2}_setup` | 0.203–1.704 s each, concurrent | worker-thread setup enclosing device Miller work | Already GPU-dominated and overlapped across openings; not a host prepare cut. |

Below cutoff: st3 `RegistersClaimReduction::prepare` 0.196 s, st4
`RamValCheck::prepare` 0.114 s, st5 `RamRaClaimReduction::prepare` 0.111 s,
st6b Hamming booleanity 0.123 s, st6b RAM-RA virtualization 0.062 s, and
st8 `MetalJointOpening::prepare` 0.091 s.

One-time setup before the st0–st8 proof window is intentionally outside the
ranked inventory: legacy Dory prover/verifier setup 49.954/18.163 s and
modular `DoryScheme::setup_prover` 38.821 s. These key/setup costs are not
charged to a proof stage.

### Exposed top five before this lane

| rank | slice | wall | state |
|---:|---|---:|---|
| 1 | st4 `RegistersReadWriteChecking::prepare` | 2.449 s | sibling-owned |
| 2 | st6b `IncClaimReduction::prepare` | 1.790 s | cut here |
| 3 | st5 `InstructionReadRaf::prepare` | 1.371 s | mixed chain; bucket attempt rejected |
| 4 | st7 `HammingWeightClaimReduction::prepare` | ~1.106 s current | W2 already cut 41.37% |
| 5 | st3 `SpartanShift::prepare` | 0.657 s | parked; modeled save below bar |

`TraceRecord::collect` is larger than all five but fully hidden behind st0,
so it is not an exposed critical-path candidate.

## Retained cut: st6b increment prepare

`jk_inc_prepare` receives balanced split-eq factors for the four cycle
points and writes the two paired weight tables directly:

```text
GPU:  split-eq(ram-rw, ram-val, rd-rw, rd-val) ──> A[], B[] ──┐
                                                               ├─ one wait
CPU:  materialize RamInc[] + RdInc[] ──────────────────────────┘
```

This removes four full intermediate eq tables. The command buffer is detached
before the independent witness-column walks and waited exactly once. Device
failure falls back to the unchanged optimized constructor. `RoundTable`
allocates device-filled `cur` plus the existing bind target; the detached pass
owns all backings through completion.

The setup-time oracle compares all four full tables (`RamInc`, `RdInc`, and
both paired weights) against the old host constructor before Criterion starts.
The existing full-round lockstep parity test pins round-polynomial and output
claim bytes, hence transcript bytes.

### Isolated timing

Command:

```bash
/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock cargo bench \
  -p jolt-eval --features metal --bench inc_prepare -- --quick
```

One quiet ABBA Criterion session, 10 samples/arm, 5 s measurement windows.
The two same-window host arms agree within 5.1%; Metal arms within 3.6%.

| size | arm | median | 95% median CI |
|---|---|---:|---:|
| `2^24` | host A | 465.30 ms | 457.15–473.44 ms |
| `2^24` | Metal A | **79.97 ms** | 79.72–80.22 ms |
| `2^24` | Metal B | **82.90 ms** | 81.44–84.36 ms |
| `2^24` | host B | 489.66 ms | 489.25–490.08 ms |

Both adjacent comparisons remove **82.81–83.07%**. Applying those ratios to
the measured 1.790499 s `2^27` slice models **0.303–0.308 s after**, or
**1.483–1.487 s removed**. Retention bars pass: isolated reduction >40% and
modeled removal >0.4 s.

The `2^22` diagnostic arm was memory-state sensitive (host A 35.44 ms, host B
132.33 ms), so it is not used for retention; both full-table oracles passed.

## Rejected follow-up: st5 stable row buckets

A production-width fixture compared the old serial stable bucketing with
Rayon chunk-local buckets and stable serial concatenation. Full flat-index,
table-id, and range oracles passed. At `2^24`, serial A/B were 37.07/38.62 ms
and parallel A/B 29.45/29.38 ms: only **20.6–23.9%** removed. At `2^22` one
pair regressed and the other improved only 11%. This misses the 40% slice bar
and can remove only ~0.07 s modeled at `2^27`, far below 0.4 s. The change and
harness were reverted.

## Verification

- `cargo fmt -q --message-format=short --all` — pass.
- `cargo clippy -q --message-format=short -p jolt-kernels -p jolt-eval
  --all-targets -- -D warnings` — pass.
- Same clippy command with `--features metal` — pass.
- Targeted `cargo nextest` with Metal: IncCR optimized/reference parity,
  Metal/optimized lockstep parity, and forced-device `2^16` parity — 3/3
  pass.
- `inc_prepare` Criterion harness — pass; exact full-table oracles at `2^22`
  and `2^24` ran before timing.
- `git diff --check` — pass.

No end-to-end prover run, per lane scope.
