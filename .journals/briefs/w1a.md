# Lane W1A — st6a + st7 GPU ports (0%-GPU stages)

## Mission

Stages 6a and 7 of the Metal-backend modular prover run at literal 0% GPU at
every scale. Port their CPU-dominant work to Metal device kernels with
byte-identical proofs. Prize at 2^27 (instrumented): ~5.6 s of pure-CPU wall
(canonical st6a 2.265 s + st7 2.072 s, plus their share of the 9.16 s
st5→st6b zero-GPU seam).

## Targets, in order

1. **`booleanity_address` slot** — `BooleanityAddressPhase::prepare` is 2.88 s
   @2^27 (76% of st6a). CPU impl: `crates/jolt-kernels/src/optimized/booleanity.rs`
   (+ `reference/booleanity.rs` for semantics). Note: the CYCLE phase already has
   a Metal slot (`MetalBooleanityCycle` in `metal/slots/ra_lazy.rs`,
   `impl PrepareKernel<Fr, Booleanity<Fr>>`) — study it; the address phase
   builds masses/eq tables/gamma weights off the stage-6a challenge aggregate
   (see `crates/jolt-prover/src/stages/stage6a.rs` doc comment).
2. **`hamming_weight_claim_reduction` prepare** — st7 is 1.86 s @2^27, ~100% =
   `HammingWeightClaimReduction::prepare` = `build_hamming_weight_tables`
   (CPU pushforward G_i + combined weights W_i) in
   `metal/slots/hamming_weight_claim_reduction.rs`. The device ROUND kernel
   already exists and works — only the table build is CPU. Port the table
   build (pushforward/scatter shape) to device, or restructure so tables are
   produced on-device from already-resident witness data.
3. **`bytecode_read_raf_address` slot** — `BytecodeReadRafAddressPhase::prepare`
   0.89 s @2^27. CPU impl: `optimized/bytecode_read_raf.rs`. Same batch as (1);
   its stage-value fold reads the witness plane's program view + PC pushforward
   from typed stage-6 rows.

st7 small members (advice/bytecode-reduction/program-image address phases) are
NOT in scope unless trivially absorbed — they're negligible at sha2-chain shapes.

## Where things live

- Backend slot registry: `crates/jolt-kernels/src/metal/mod.rs` (`fn metal()`);
  slot fields in `crates/jolt-kernels/src/backend.rs` (`JoltBackend`).
- 14 existing Metal slots in `crates/jolt-kernels/src/metal/slots/` — copy their
  structure: `metal_gate(KIND, work_items)` threshold → device kernel, fallback
  to optimized twin below threshold or on device failure; structural errors
  propagate. Env knobs: `JOLT_METAL_MIN_TERMS[_<KIND>]`.
- Shaders: `crates/jolt-kernels/src/metal/shaders/*.metal`. BN254 Fr Montgomery
  arithmetic device tier — reuse existing field primitives (`jk_fr_*`).
  WARNING (from M5 W3 T6): flat-table word offsets must be `ulong` — 32-bit
  word-offset products overflow at 2^27 (5 tables × 2^27 × 8 limbs hits 2^32).
- Stage drivers: `crates/jolt-prover/src/stages/{stage6a,stage7}.rs`,
  `drivers.rs`.

## Ground rules

- Worktree `~/dev/jolt/.worktrees/gpuutil-w1a`, branch `gpu/util-w1a`. Never
  push. Commit per unit with proper messages.
- **Hard gate: proof bytes identical.** Each new slot needs a parity test
  following the existing slot-test pattern (force device threshold low, assert
  positive device dispatch count, compare against CPU path bytes).
- Full gate matrix before any retained commit (see `.journals/gpu-util.md`
  Protocol section — kernels/dory/byte-diff both modes/muldiv both modes/clippy
  three arms/fmt).
- Timed runs only under `/tmp/jolt-gpu.lock.d` mkdir-lock (protocol in
  `.journals/gpu-util.md`). Iterate at 2^22-24; confirm at 2^25 cool. NO 2^27
  runs from this lane — the orchestrator certifies scale.
- Measure GPU util with `-F jolt-profiling/monitor` runs (gpu_percent counter
  in the chrome trace); measure walls WITHOUT monitor.
- Kill gate: at 2^24 with parity green, st6a+st7 combined wall −35% vs your own
  same-tree baseline, else stop and report honestly.

## Baselines to beat (canonical, 2^25 / 2^27)

st6a 0.493 / 2.265 s; st7 0.247 / 2.072 s. Your worktree baseline will differ
slightly — measure your own before/after pairs (ABBA at 2^24/2^25).

## Reporting

- Decomposition note first: for each target, what is portable kernel work vs
  host-serial glue (transcript draws, small folds), expected device shape.
  Write to `.journals/lane-reports/w1a.md`, then message_parent (checkpoint i).
- Checkpoint ii: parity green + first 2^24 A/B. Final: 2^25 cool confirm,
  lane report committed, branch summary (commits, binary sha256, gate results).
