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
