# Metal M5 saturation campaign

Opened 2026-08-04 22:19 EDT from `feat/metal` / `88b063db3`.

## Mandate and gates

- Byte parity is not a gate. Prover and verifier may change together.
- Retention gate for protocol changes: end-to-end accept, tamper rejection,
  full integrated test battery, and a written soundness argument here.
- Two-round fusion may use only a single radix-4 univariate message `q(Z)` of
  degree at most `3d`, followed by one challenge. The two-polynomial/shared-
  challenge construction is forbidden: for `d >= 2`, its diagonal kernel
  admits `Delta = c X (X - Y)`.
- Velocity v3: small-scale iteration (`2^22..2^24`), at most two timed benches
  per decision unless they disagree, one full battery at the integrated gate,
  certification at `2^25` and `2^27`.

## Baseline

Fresh non-monitor runs from `88b063db3`, benchmark-locked on AC power:

| scale | prove | padded throughput | peak RSS | peak footprint |
|---|---:|---:|---:|---:|
| `2^25` | 19.67 s | 1.706 MHz | 27.42 GiB | 26.78 GiB |
| `2^27` | 71.77 s | 1.870 MHz | 76.87 GiB | 75.39 GiB |

The fresh `2^27` run reproduces the prior 71.46 s flagship within 0.31 s;
the campaign starts from the good st0 mode rather than the ~78.5 s mode.
Artifacts: `.journals/artifacts/baseline-2to{25,27}-20260804.log` and
`benchmark-runs/perfetto_traces/modular_sha2_chain_{25,27}_metal.json`.

Stage vectors (`st0..st8`, seconds):

- `2^25`: `[5.335, 1.104, 0.759, 0.580, 2.228, 3.348, 0.070, 1.625, 0.249, 4.367]`
- `2^27`: `[12.079, 4.456, 2.775, 2.403, 8.768, 14.220, 0.250, 16.345, 1.909, 8.514]`

Prior certified anchor: `2^27 = 71.46 s / 1.878 MHz`, with a bimodal bad mode
near `78.5 s`; `2^25 = 19.71 / 19.89 s` warm, with a prior cool reference near
`18.0 s`. Final prior monitor trace had no GPU-idle window over 1 s.

## Wave 1

1. Radix-4 sumcheck fusion: concrete polynomial derivation and pro-model
   soundness review before implementation; prototype only after review.
2. Address-major probe: include Dory commitment/opening, not only the already-
   closed booleanity address-phase ordering.
3. Saturation attribution: fresh stage traces/counters separating ALU,
   bandwidth, occupancy, host serialization, and launch/synchronization.

## Decisions and results

- Wave-1 scoping lanes dispatched after both baselines: radix-4 fusion
  (`a346b564`), address-major/Dory (`1af3a092`), saturation attribution
  (`3cceafee`). Phase 1 is static only so Cargo and timed runs remain serial.
- GPT-5.6 pro oracle dispatched on the concrete radix-4 polynomial and current
  Jolt round-loop contracts (`55fd4b105b90`). Implementation is blocked on its
  review. Source-file attachments were unavailable across the external-machine
  boundary, so the oracle prompt carries the concrete protocol and integration
  contracts inline.

### Fresh saturation evidence

The fresh `2^27` monitor run completed in 80.99 s; its wall is attribution-only.
Time-weighted `ioreg` device-utilization samples by stage:

| stage | monitor wall | GPU | CPU | active cores |
|---|---:|---:|---:|---:|
| st0 | 18.01 s | 79.4% | 61.9% | 11.1 |
| st1 | 4.70 s | 77.1% | 12.4% | 2.2 |
| st2 | 2.96 s | 48.3% | 18.4% | 3.3 |
| st3 | 3.00 s | 16.1% | 37.0% | 6.7 |
| st4 | 9.28 s | 40.2% | 22.3% | 4.0 |
| st5 | 14.67 s | 77.6% | 25.0% | 4.5 |
| st6a | 0.24 s | 36.0% | 44.7% | 8.1 |
| st6b | 17.49 s | 34.5% | 18.6% | 3.3 |
| st7 | 1.89 s | 13.5% | 65.6% | 11.8 |
| st8 | 8.70 s | 86.6% | 10.2% | 1.8 |

**Correction to the closed campaign:** re-analysis of both the old final trace
and this fresh trace finds sampled-zero GPU intervals over 2 s in st3, st4,
st6b, and st7. The prior `NONE >1 s` statement is not reproducible. In the
fresh trace, the longest are st3 2.48 s, st4 2.58 s, st6b 2.23 s (several),
and st7 2.11 s. `ioreg` is a sampled activity signal, not an ALU-occupancy
counter, but a multi-second exact zero is enough to reject continuous device
occupancy.

Dominant host/round structure at `2^27`: st3 `prove_batch` 2.14 s; st4
`RegistersRWC::prepare` 2.45 s plus 6.69 s rounds; st6b 15.21 s rounds plus
`IncCR::prepare` 1.79 s; st7 is almost entirely `HWCR::prepare` 1.887 s.

The built-in `JOLT_METAL_CB_TRACE` audit at `2^25` records 646 command buffers.
Fresh empty-CB cost is 133.8 us; the absolute launch/round-trip ceiling is
therefore about 86 ms, under 0.5% of a 19.72 s proof. Launch overhead is not
the campaign bottleneck, though it matters in tiny tail rounds.

Fresh roof/pressure probes on the M5 Max:

- streaming bind: 357 GB/s sustained in the contention probe, 485 GB/s best
  isolated large pass; compute roof: 11.30 Gmont-mul/s;
- concurrent GPU bind + CPU field-mul cuts GPU bandwidth 55% and CPU bandwidth
  45%, identifying shared-memory contention as a first-order limiter;
- Miller kernels prefer one/two pair-evaluations per thread, and per-pair cost
  collapses until 4k-8k threads are exposed: occupancy/register pressure is
  material inside st0/st8, but those stages already show 79-87% device use;
- Miller + CPU ALU soak is neutral on the device, separating compute occupancy
  from the memory-walk contention behind st0's bimodality.

Initial verdict: the remaining wall is a mix of serial/parallel host mass in
st3/st4/st7 and bandwidth/queue contention in st5/st6b. ALU/occupancy is local
to the Miller-heavy endpoints; fixed launch overhead is negligible globally.

Hardware-counter limitation: this host has no `xctrace` developer tool, and
Metal exposes only the `timestamp/GPUTimestamp` counter set (`counterSets`
enumerated directly on the M5 Max). ALU occupancy, SIMD utilization, cache
misses, and DRAM bytes are therefore not directly observable from public
Metal counters here. The audit distinguishes them through fresh device-active
samples, command-buffer GPU timestamps, controlled bandwidth/compute roofs,
thread-scaling, and contention experiments; it does not relabel `ioreg` GPU%
as ALU saturation.

### Radix-4 pro-model gate (`55fd4b105b90`)

**NO-GO for the proposed `3d` message with ordinary bind-by-two/Dory MLE
openings.** For digit embedding `0..3 -> (00,01,10,11)`, the cubic coordinate
maps are

`x(Z) = -Z^3/3 + 3Z^2/2 - 7Z/6` and
`y(Z) = 2Z^3/3 - 3Z^2 + 10Z/3`.

The four-point identity is valid:

`q(0)+q(1)+q(2)+q(3) = G(0,0)+G(0,1)+G(1,0)+G(1,1)`.

But a four-entry oracle's degree-3 digit interpolation is not its ordinary
bilinear MLE at `(x(z), y(z))`. Counterexample: `U(X,Y)=XY` has digit values
`[0,0,0,1]`; its cubic interpolation is `Z(Z-1)(Z-2)/6`, while ordinary
binding/opening yields `x(Z)y(Z)`. They agree only at the four digit nodes.
The original proposal would therefore propagate one claim and open another.

Sound alternatives:

1. Preserve ordinary MLE/Dory semantics and define
   `q(Z)=G(x(Z),y(Z))`. A generic relation with per-variable degree `d` has
   bidegree `(d,d)`, hence `deg q <= 6d`, with error `6d/|F|` per fused pair.
2. Preserve `3d` by changing every table bind and commitment/opening to the
   quaternary Lagrange extension. This is a commitment-protocol redesign, not
   a Metal-only prototype.

For either consistent extension, coordinate correlation is not itself a
soundness failure: the univariate root bound applies in `z`, and Dory may open
an MLE at transcript-derived correlated coordinates. RLC batching remains
linear. A member inactive across both fused variables contributes constant
`claim/4`; pairs must split at every active-set, degree, optional, uniskip, or
binding-order boundary. The linear coefficient remains compressible because
the four-point functional has weight `0+1+2+3=6`, invertible in the field.

Decision: never build the inconsistent `3d + ordinary Dory` shape. The fusion
lane must first prove either a relation-specific lower bound, a cheap
quaternary-to-Dory bridge, or an honest `6d` prototype with a positive cost
model. Mandatory regression: `U=XY` distinguishes digit interpolation from
ordinary MLE binding.

### Quaternary Dory bridge under review

`dory-pcs 0.4.0` commits only the Boolean-corner value vector, so the
commitment itself is extension-agnostic. Its prover already accepts arbitrary
public evaluation vectors through `MultilinearLagrange::compute_evaluation_vectors`.
The verifier is the missing seam: it currently stores one binary coordinate
per Dory reduce round and reconstructs the folded scalar as a product of
`alpha * (1-r) + r` terms.

A radix-4 Lagrange vector `[l0(z),l1(z),l2(z),l3(z)]` is generically rank two
when reshaped as 2x2, so it cannot be represented by two ordinary MLE
coordinates. It is nevertheless one tensor factor. Under Dory's half-split
folds `s <- alpha*s_L+s_R`, two consecutive reduction challenges fold that
factor to

`alpha_2*alpha_1*l0 + alpha_1*l1 + alpha_2*l2 + l3`

(`alpha_i^-1` on Dory's opposite scalar vector). A typed binary/radix-4 factor
schedule can therefore keep verifier work logarithmic: hold one four-weight
factor across two Dory reductions, multiply the accumulator by the expression
above, and never materialize the full vector. A radix-4 factor must not
straddle Dory's row/column matrix split; an odd boundary remains binary.

This changes Dory's public evaluation-point API and verifier scalar-fold
logic, but not the commitment or reduce proof. GPT-5.6 pro job
`ec0b50d07d63` is auditing the exact construction, degree bound, folding order,
and soundness before any implementation.

### Metal radix-4 bind microprototype

Implemented an isolated `jk_fr_bind4` prototype without changing protocol
bytes or the prover driver. For one four-entry Lagrange block it uses

`a0 + l1*(a1-a0) + l2*(a2-a0) + l3*(a3-a0)`.

Because `l0+l1+l2+l3=1`, this is the exact quaternary Lagrange bind. It costs
the same three Montgomery products as two binary binds, but removes the
intermediate `N/2` table read+write and one dispatch. The device property test
matches a host four-point Lagrange evaluation for dense and ragged shapes.

Two timing decisions (minimum of five warm passes inside each run):

| input | run 1 speedup | run 2 speedup | verdict |
|---:|---:|---:|---|
| `2^20` | 1.25x | 1.10x | small positive |
| `2^22` | 0.49x | 1.06x | unstable / no proven gain |
| `2^24` | 1.98x | 1.51x | strong positive |

Second-run absolute large-table result: two binary binds in one command buffer
2.39 ms versus direct radix-4 1.59 ms (`1.51x`). The first run was 3.11 ms
versus 1.57 ms (`1.98x`). At production-sized dense tables, direct bind-by-two
therefore has real Metal bandwidth upside; small/mid tables remain launch/cache
mode sensitive. This is only the binding half of the cost model: higher-degree
message generation and generalized Dory evaluation can still erase the gain.

The prototype now exercises the full algebraic shape for a dense degree-2
relation `G=U*V`: interpolate the seven values needed for `deg q <= 3d = 6`,
check the four digit-node sum against the Boolean claim, draw `z`, bind both
tables on Metal with the quaternary weights, and match `q(z)` to the terminal
bound-table relation. A tampered polynomial adds
`c*Z*(Z-1)*(Z-2)*(Z-3)`: it preserves all four node values and the input-claim
check but is rejected at the terminal random point. Targeted device/algebra
tests pass. This establishes a sound single fused-round Metal prototype; it
does not yet alter production transcripts or Dory.

### Address-major Metal probe

Address-major is already a valid end-to-end protocol mode, including verifier
and Dory parity tests, but the current Metal implementation is structurally
hostile to it:

- cycle-major commitment streams every trace column from one packed pass;
  address-major materializes one full strided grid table per polynomial;
- `MetalJointOpening::prepare` accepts only cycle-major grids, so address-major
  falls back wholesale to the optimized CPU fold path in stage 8;
- Metal sumcheck slots are cycle-major gather/coalescing designs. Global
  address-major would require new kernels, not a free layout flip.

A benchmark-only `JOLT_TRACE_ORDER=address` selector allowed a direct `2^22`
A/B. Cycle-major completed in **3.52 s / 1.192 MHz padded**, peak RSS 3.81 GiB.
The address-major arm was still consuming roughly 12 CPU cores and 4.14 GB RSS
after **more than 240 s** and was terminated. This is a decisive **>68x lower
bound regression** at the small-scale gate, so no second address-major run is
warranted under Velocity v3.

Decision: kill global address-major on the current Metal backend. The CUDA
sharding prize does not transfer to a unified-memory machine whose fast path
already streams cycle-major columns. Retain address-major only as a correctness
mode and use targeted internal AoSoA/block-local transposes when an individual
kernel demonstrates a locality win. A production address-major campaign would
first need a streaming commitment builder, Metal joint-opening fold, and
address-major sumcheck kernels; it is not an optimization knob today.
