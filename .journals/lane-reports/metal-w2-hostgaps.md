# Metal wave 2: host-serial gaps

## Audit

Ranked by the largest attributable host-serial span in the fresh `2^27`
trace:

| rank | stage / routine | isolated wall | disposition |
|---:|---|---:|---|
| 1 | st4 `RegistersRWC::prepare` and scan/allocation boundaries | 2.449 s prepare; 4.761 s total sampled-zero | Larger ceiling, but the repeated scan/bind boundary is not one isolated routine. |
| 2 | st3 `InstructionInput::prove_batch` round 0 | 2.140 s | Mixed host message construction and dense device write; no reusable isolated kernel. |
| 3 | st7 Hamming-weight pushforward preparation | 1.887 s | Selected: 99.7% of st7, one transcript-free routine, with an existing exact split-eq implementation to reuse. |

## Isolated objective

Added `jolt-eval` objective/bench `hamming_weight_pushforward`:

- `2^22` real 48-byte `InstructionCycleRow` values;
- production `2^27` one-hot geometry: 8-bit chunks and 16/2/3
  instruction/bytecode/RAM columns;
- deterministic hot/cold bytecode and RAM distribution;
- setup-time oracle: full `eq(r, ·)` materialization plus direct per-row
  scatter, compared exactly against every output bucket before timing;
- timed body: the stage-7 pushforward only; no prover or transcript.

## Change

Extracted stage 6a's one-hot selector and split-eq deferred-bucket algorithm
into `optimized::one_hot_pushforward`. Booleanity keeps its balanced split.
Stage 7 chooses four outer blocks per Rayon worker: inner eq weights accumulate
by addition, then one outer multiplication is applied per reduced bucket.
This removes the old per-row eq multiplication while retaining enough blocks
for work stealing.

## Timing decision

Criterion, 10 flat samples per arm, 12 s measurement window:

| arm | median | 95% median CI |
|---|---:|---:|
| before | 79.816671 ms | 78.355649–80.282463 ms |
| after | 46.795669 ms | 46.535044–47.512235 ms |

Result: **33.021002 ms removed, 41.3711% reduction, 1.7056x speedup**.
The confidence intervals do not overlap; retain.

Applying the isolated ratio to the measured 1.887 s stage-7 prepare estimates
1.106 s after, **0.781 s stage gain**. Against the 71.77 s flagship proof,
the estimated whole-proof gain is **1.09%** (`71.77 -> 70.99 s`). This is a
stage-calibrated estimate, not an end-to-end measurement.

## Verification

- `cargo check --message-format=short -p jolt-eval`
- targeted `cargo nextest`: Hamming reduction reference parity plus three
  booleanity address parity geometries, **4/4 passed**
- `cargo clippy -p jolt-eval --all-targets ... -D warnings`
- `cargo fmt --all`
- no end-to-end prover run, per lane scope

## st3: instruction-input round 0

### Attribution

The isolated fixture uses the production native lane types, eight-table
geometry, low-to-high Gruen split, and first-bind dense allocation. A traced
single pass separates the host message from the bind command buffer's GPU
timestamp; residual is synchronous bind wall minus GPU execution, covering
first-write residency, queue/wait, and host partial collection.

| slice / exact function | `2^22` | share | `2^24` | share |
|---|---:|---:|---:|---:|
| Host message: `native_q_evals` | 0.011752 s | 33.41% | 0.044612 s | 40.41% |
| Dense write + evaluation: `jk_instr_input_bind_native` GPU window | 0.010899 s | 30.99% | 0.029308 s | 26.55% |
| First-write/sync residual: `ComputePass::run` / `PendingPass::wait` / `Partials::sums` | 0.012520 s | 35.60% | 0.036477 s | 33.04% |
| **Combined front** | **0.035171 s** | **100%** | **0.110397 s** | **100%** |

The `2^27` stage trace independently measures `native_q_evals` at 0.455 s
and the dense bind round at 0.793 s. The host message is the largest growing,
separately removable slice; the bind's device write and first-touch residual
remain unchanged.

### Isolated objective and change

Added `jolt-eval` bench `instruction_input_round0` at `2^22` and `2^24`:

- deterministic packed flags, u64 operands, full-range i128 immediates;
- production split-eq factorization and eight `T/2` dense bind target;
- exact setup-time oracle: device q(0..=3) equals `native_q_evals`;
- shared `gpu_lock()`, setup/buffers outside Criterion timing.

`jk_instr_input_q0` computes q(0), q(1), and the quadratic coefficient from
native lanes. Boolean endpoint selection removes field products at q(0/1);
flag transitions times operand slopes produce the quadratic term. Three
device reductions replace the host walk, and the host reconstructs q(2/3).
The existing host implementation remains the fail-closed fallback. Round
polynomials and transcript bytes are unchanged.

### Timing decision

Criterion, 10 samples, 5 s measurement per case:

| size | before median (95% median CI) | after median (95% median CI) | reduction |
|---|---:|---:|---:|
| `2^22` primary | 32.055792 ms (30.621938–33.365754) | 3.810223 ms (3.804239–3.812924) | 88.1138%; 8.4131x |
| `2^24` confirmation | 139.642703 ms (135.926600–142.529087) | 15.148045 ms (14.628770–15.630624) | 89.1523%; 9.2185x |

Both confidence intervals are disjoint. Applying the primary isolated ratio
to the measured 0.455 s `2^27` host slice estimates **0.400918 s removed**:
stage 3 `2.340 -> 1.939 s` (**17.13%**) and flagship proof
`71.77 -> 71.37 s` (**0.56%**). This is a stage-calibrated estimate, not an
end-to-end measurement. Retention bar (`>= 0.3 s`) clears; **GO**.

### Verification

- exact `2^22` and `2^24` fixture oracle passed;
- targeted Metal instruction-input parity: **3/3 passed**;
- `cargo fmt --all` passed;
- touched-crate clippy with `-D warnings` passed;
- no end-to-end prover run.

## st1: Spartan outer remaining

### Attribution

The isolated fixture uses all 17 production `TraceRecord` lanes, the exact
`log_t + 1` split-eq geometry, production T1/AzBz kernels, and the fused
shrinking-round dispatch. Command-buffer GPU timestamps split synchronous
wall into device execution, queue/wait residual, and host preparation.

| slice / exact function | `2^22` | share | `2^24` | share |
|---|---:|---:|---:|---:|
| Uniskip message: `dispatch_t1` / `jk_outer_t1` | 0.059810 s | 38.42% | 0.237246 s | 39.80% |
| First remainder message: `dispatch_azbz` / `jk_outer_azbz` | 0.053279 s | 34.23% | 0.207250 s | 34.77% |
| Key evaluation: `claimed_inputs_from_record` | 0.025111 s | 16.13% | 0.101753 s | 17.07% |
| Bound rounds: `bind_and_endpoints` / `jk_outer_round` | 0.017021 s | 10.93% | 0.048994 s | 8.22% |
| Interpolation, derived weights, allocation, final host bind | 0.000439 s | 0.28% | 0.000819 s | 0.14% |
| **Attributed total** | **0.155660 s** | **100%** | **0.596062 s** | **100%** |

At `2^24`, T1 splits into 0.207570 s GPU + 0.025517 s wait +
0.004159 s host; Az/Bz into 0.143864 + 0.062656 + 0.000730 s; the
round loop into 0.019315 + 0.028435 + 0.001244 s. At `2^22`, the same
splits are T1 0.052101 + 0.006711 + 0.000998 s, Az/Bz 0.035787 +
0.017206 + 0.000286 s, and rounds 0.005021 + 0.011317 + 0.000683 s.

The round loop is already fused: each `jk_outer_round` command buffer folds
Az/Bz and produces q(0)/q(∞) in one dispatch. There is no st4-style
message/bind host boundary to remove. The largest separately removable host
slice is the final 35-opening `claimed_inputs_from_record` walk.

### Isolated objective and change

Added `jolt-eval` bench `spartan_outer_claims` at `2^22` and `2^24`:

- deterministic full-width record lanes and production split-eq point;
- exact setup oracle against `claimed_inputs_from_record` for all 35 claims;
- shared `gpu_lock()`, record construction and oracle outside timing;
- same-window host/Metal Criterion arms; no prover or transcript.

`jk_outer_claims` maps one threadgroup to one outer-eq index and a four-column
tile. Threads reduce inner-eq weighted native lane values; the host applies
one outer-eq multiplication per partial. Nine tiles cover the 18 wide and 17
boolean inputs without a full-T eq table. `output_claims` keeps the host
fallback and gates Metal below the measured winning scale, `2^24`. Opening
values and transcript bytes are unchanged.

### Timing decision

Criterion quick profile, 10 samples per arm, adjacent host/Metal arms:

| size | host median (95% median CI) | Metal median (95% median CI) | result |
|---|---:|---:|---:|
| `2^22` primary | 25.373 ms (24.995–25.468) | 31.438 ms (31.301–31.983) | host wins 19.29%; gated |
| `2^24` confirmation | 300.34 ms (289.86–302.95) | 115.43 ms (114.49–119.20) | **61.57% removed; 2.60x** |

The `2^24` intervals do not overlap. Applying its 61.57% isolated reduction
to the measured 17.07% final-key-evaluation share estimates **0.468 s removed**
from the measured 4.456 s `2^27` stage: `4.456 -> 3.988 s` (**10.51%**).
Against the 71.77 s flagship proof, the estimated whole-proof gain is
**0.65%** (`71.77 -> 71.30 s`). Stage-calibrated estimate only; no end-to-end
run. Retention bar clears; **GO**.

### Verification

- exact `2^22`, `2^24`, and targeted `2^8` host/device claim oracle passed;
- existing full Metal/optimized Spartan outer package parity passed;
- `cargo fmt` passed;
- touched-crate clippy with `-D warnings` passed;
- no end-to-end prover run.
