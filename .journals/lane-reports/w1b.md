# Lane W1B — stage 6b CPU members

## Checkpoint 1 — BytecodeReadRafCycle decomposition

### Retained host glue

- Validate relation dimensions, committed chunk count, stage-point widths, and output-opening count.
- Reclaim the session-shared packed `PcRow` scan; build only the small address-chunk eq tables and selector metadata.
- Derive the five scalar stage weights, the entry scalar, and the output-opening IDs.
- Assemble each round polynomial from device lane sums with the existing `round_poly_from_skipped_evals` recipe; retain transcript/bind order and output-claim extraction unchanged.
- Fail closed to the optimized CPU kernel before dispatch, or reclaim the live host-visible tables after a mid-sumcheck decline/failure.

### Device kernels

1. **Combined-table init:** tensor-split each stage cycle point on the host at `O(sqrt(T))`; one device map writes `C(j) = sum_s weight_s * E_hi_s[j_hi] * E_lo_s[j_lo]`, including the entry term at `j = 0`. This removes the five `O(T)` host eq-table builds and additions.
2. **Lazy rounds (width 1/2/4):** gather every `ra_i` pair directly from packed mapped-PC rows and the small branch tables; fuse any pending bind of `C`, evaluate `C(t) * product_i ra_i(t)` at `t in {0, 2, ..., degree}`, and reduce lane-major partials.
3. **Third-bind adoption:** fuse the pending `C` bind with width-8 RA materialization into one flat, device-owned factor ping-pong at length `T/8`.
4. **Dense rounds:** fuse the pending bind of every flat factor with the same product-grid lane reduction. Use 64-bit MSL word offsets for every `factor * len * FR_LIMBS` rebase; table-local indices remain 32-bit.

### Geometry and recovery invariants

- Synthetic `T = 2^27` offset parity test precedes the slot implementation; it must expose the fifth full-width factor start as word `2^32` without allocating the domain.
- Every mutating round writes only the next ping-pong buffer; the current state remains recoverable until command-buffer success.
- Device threshold controls prepare and rounds; forced-device parity asserts a positive dispatch count. CPU and Metal proof-byte matrices remain the retention gate.
- Reuse existing Metal buffer/runtime utilities only. No allocator, arena, madvise, parking-lifetime, or stage-driver ownership changes.

## Checkpoint 2 — parity green; `2^24` kill gate passed

- Forced-device slot lockstep: optimized and Metal round coefficients/output claims match; combined init plus every fixture round dispatched. Synthetic `2^27` flat-offset device parity remains green (`factor=4`, `len=2^27` => word `2^32`).
- Proof bytes: `byte_diff` 11/11 with `prover-fixtures`; 11/11 with `prover-fixtures,metal` (the current revision discovers 11 fixtures; the campaign brief's 19 count is stale).
- Same-binary `2^24` A/B, wall mode without monitor, bench lock held, pre-run load 4.35/3.47:

| metric | slot disabled | slot enabled | delta |
|---|---:|---:|---:|
| stage 6b | 2.102 s | 1.200 s | **-0.901 s (-42.9%)** |
| BytecodeReadRafCycle prepare + rounds | 1.278 s | 0.596 s | **-0.682 s (-53.3%)** |
| total prove | 11.94 s | 10.90 s | -1.04 s (-8.7%) |

Kill gate required stage 6b `-15%` or member `-40%`; both pass. Logs/traces: `/tmp/w1b-bytecode-s24-{A,B}.{log,json}`.

## Final — `2^25` cool confirmation

Non-monitor release binary; bench lock held; 12 consecutive 15-second load samples below 6 (2.52 down to 1.13); AC power at 100%. ABBA disables only `bytecode_read_raf_cycle` in A and forces its threshold to zero in B.

| run | total | stage 6b | Bytecode prepare | Bytecode rounds | Bytecode member |
|---|---:|---:|---:|---:|---:|
| A1 disabled | 20.59 s | 1.845 s | 0.328 s | 0.601 s | 0.929 s |
| B1 enabled | 20.40 s | 1.490 s | 0.095 s | 0.565 s | 0.660 s |
| B2 enabled | 21.49 s | 1.595 s | 0.102 s | 0.569 s | 0.672 s |
| A2 disabled | 23.12 s | 2.296 s | 0.398 s | 0.825 s | 1.223 s |
| **A mean** | **21.855 s** | **2.070 s** | **0.363 s** | **0.713 s** | **1.076 s** |
| **B mean** | **20.945 s** | **1.543 s** | **0.099 s** | **0.567 s** | **0.666 s** |
| **delta** | **-0.910 s (-4.2%)** | **-0.528 s (-25.5%)** | **-0.264 s (-72.8%)** | **-0.146 s (-20.5%)** | **-0.410 s (-38.1%)** |

Logs/traces: `/tmp/w1b-bytecode-s25-{A1,B1,B2,A2}.{log,json}`. The slot remains retained: its `2^24` kill gate passed both allowed criteria, and the cool `2^25` confirm preserves a 25.5% stage-6b win.

### IncClaimReduction prepare — rejected

- Prototype initialized both increment columns and both combined cycle-eq weights from the parked stage-4 raw lanes in one Metal kernel; existing device rounds consumed the resulting ping-pong tables.
- Cool `2^25` ABBA: prepare 0.325 s CPU vs 0.317 s device (`-0.009 s`); stage 6b 1.537 s vs 1.559 s (`+0.022 s`). This misses the `-0.3 s` stage gate.
- Monitor attribution did not show a positive utilization shift inside the prepare span: two samples per arm, 44.5% mean / 89% max CPU path vs 40.0% mean / 80% max device path.
- Prototype reverted completely; no allocator, arena, madvise, lifetime, or parked-column ownership change retained. Logs/traces: `/tmp/w1b-final-s25-{A1,B1,B2,A2}.{log,json}` and `/tmp/w1b-inc-monitor-{A,B}.{log,json}`.

### Final gate matrix

- `jolt-kernels --features metal`: 233/233, including forced-device lockstep and synthetic `2^27` offset parity.
- `jolt-dory`: 46/46. Legacy muldiv: 3/3 `host`; 3/3 `host,zk`.
- Proof bytes: 11/11 `prover-fixtures`; 11/11 `prover-fixtures,metal` (current discovered suite count; brief's 19 is stale).
- Clippy `-D warnings`: workspace `allocative,host`; workspace `allocative,host,zk`; `jolt-kernels` + `jolt-prover` Metal targets. `cargo fmt --all --check` and `git diff --check`: clean.
- Final non-monitor release binary SHA-256: `333ad9ee21f5f503388eb89c0e6bc4375e522b4c22c18a547553268e2d33a29b`.

### Commits

- `524eef1b2` `docs(metal): decompose bytecode read RAF cycle port`
- `14f2a9289` `test(metal): cover bytecode RAF large offsets`
- `b251011cf` `docs(metal): record bytecode RAF kill gate`
- `724621194` `feat(metal): port bytecode read RAF cycle`
