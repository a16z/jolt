# Registers claim-reduction Metal design

This directory contains the first executable slice for stage 3's
`RegistersClaimReduction` member: the resident native-plane projection that
builds the prefix `q` table. The slice is registered in the shared Metal
library and has a Criterion microbenchmark, but it is not a production backend
and does not alter the protocol.

The production relation was traced through the symbolic relation, concrete
verifier, generated stage-3 output aliases, reference kernel, optimized CPU
kernel, and generic batched-sumcheck driver. Projected numbers below remain
models unless explicitly labeled as observed.

## Resident BCSR amendment (2026-08-07)

This section is the current implementation contract. It supersedes the older
dense-SoA producer choice, roof table, blockers, and integration order later
in this file. The durable denominator below remains authoritative. The older
sections remain as evidence for the already-executed standalone kernel and the
alternatives that were rejected. The component/reducer shaders and checked
standalone runtime are executable; the production resident adapter and
midpoint remain unwired.

The authoritative five-pair exact-log-26 optimized-CPU denominator remains
`99.905582 ms`. The hard 5x gate is `19.981116 ms`, the working 7x target is
`14.272226 ms`, and the 8x pursue-if-credible gate is `12.488197 ms`. The newer
`101.146122-ms` one-pair observation is diagnostic context only and cannot
loosen a promotion cap. All gates are complete-member PIOP wall times with
host Fiat--Shamir included; GPU-active time alone cannot pass one.

### Selected source boundary

The stage-4 `registers_read_write_v3` BCSR-256 state-flow receipt is the
selected carrier. Its pre-sumcheck producer must publish an admitted
`RegisterBcsrReceipt` before the stage-1 command is encoded, and the same
immutable device allocations must remain live through the stage-3 midpoint.
Do not publish three dense `u64[T]` planes, and do not add rs1/rs2 event-value
planes. At the measured log-26 event census, the alternatives are:

| source | new persistent bytes | new producer writes | component reads | midpoint reads | charged source traffic |
|---|---:|---:|---:|---:|---:|
| three dense value planes | 1,610,612,736 | 1,610,612,736 | 1,610,612,736 | 536,870,912 | 3,758,096,384 |
| sparse event-value planes | 924,611,008 | 924,611,008 | 1,494,745,080 | 453,509,120 | 2,872,865,208 |
| resident BCSR state flow, column replay | 0 | 0 | 1,039,896,120 | 520,617,984 | 1,560,514,104 |
| resident BCSR plus dense read indices | 134,217,728 | 134,217,728 | 1,074,266,112 | 520,617,984 | 1,729,101,824 |

The selected indexed candidate adds one `u8[T]` map for each read operand. The
producer must emit them during its existing row traversal; a second host scan
is not part of the production contract. The measured row counts are `59,652,323` rs1 events, `55,924,053` rs2
events, and `50,331,648` rd events. The existing BCSR allocation is
2,350,383,104 bytes and its initialized producer writes 2,180,746,808 bytes.
Those shared costs are charged exactly once in a paired resident-PIOP
evaluation. The incremental numbers above are valid only after an admitted
BCSR receipt is already resident; a standalone member may not hide its
producer.

BCSR construction cannot directly produce `q`: `tau_hi` is not available
then, `gamma` is not drawn until stage 3, and a challenge-combined table cannot
recover the three stage-1 openings. The correct co-production point is stage
1, after `product_uniskip_tau_low` is known. It projects three canonical
component tables and delays only the gamma combination.

### Kernel contract

For log 26, `P = H = 8192`, there are 262,144 BCSR blocks, and

```text
j         = x_hi * 8192 + x_lo
low_block = x_lo / 256
block     = x_hi * 32 + low_block.
```

The retained control, `solinas_registers_claim_bcsr_components`, assigns its
first 128 threads to register columns and replays each column's merged rs1,
rs2, and rd runs. Target-scale measurement falsified that architecture: a hot
register serializes one thread and the curve flattens near 42 ms even as the
partial count rises.

The selected candidate,
`solinas_registers_claim_bcsr_indexed_components`, keeps only the rd state-flow
topology plus dense `u8[T]` rs1/rs2 register-index maps. Its first 128 threads
scatter the block's rd event numbers into a 256-entry threadgroup array. Each
cycle thread then finds the last rd position strictly before its cycle by a
binary predecessor lookup within that register's short rd run. No predecessor
means the block start value. The strict comparison preserves reads-before-write
at a same-cycle read/write; the rd component uses the same-cycle post value.
The 256 position threads accumulate

```text
eq(tau_hi)[x_hi] * (rd, rs1, rs2)
```

for their `x_lo`. The candidate uses canonical half-width
multiplication and addition per nonzero term. This matches the retained
33.168-billion-term/s control and keeps three four-limb accumulators live,
rather than the register-heavy deferred 224-bit state. The group writes
`partials[component][partial][x_lo]`. Its dynamic workspace is 528 bytes: 256
`u16` rd-event numbers and one shared 16-byte equality coefficient. The
control's three dense threadgroup value arrays require 6,160 bytes.

The runtime admits power-of-two partial counts that divide `H`. Log-26
screens compare 32, 64, 128, and 256 partials: reducing the count cuts
the partial write/read traffic while increasing each group's lifetime. The
observed winner is 128 partials; it dispatches 4,096 component groups and
allocates 50,331,648 partial bytes.

`solinas_registers_claim_bcsr_reduce_components` dispatches 96 groups of 256
threads and reduces the configured partials to three canonical `Fp128[P]` tables:
`Q_rd`, `Q_rs1`, and `Q_rs2`. The host reconstructs the three omitted stage-1
openings by dotting each with `eq(tau_lo)` (`3P` full products). Once stage 3
draws `gamma`, the host forms

```text
q = Q_rd + gamma * Q_rs1 + gamma^2 * Q_rs2
```

with `2P` full products. This keeps the exact prefix `P * Q` rounds; it makes
no protocol or transcript change.

After the 13th host prefix challenge has been absorbed,
`solinas_registers_claim_bcsr_fold_rd_midpoint` dispatches 8,192 groups of 256
threads. One group owns one `x_hi`. For each of its 32 BCSR blocks, the first
128 threads scatter rd event numbers into a 256-entry threadgroup map; all 256
cycle threads then accumulate independently. The 640-byte workspace holds the
event map and eight SIMDgroup sums. The kernel scans only rd offsets,
positions, and post values and reduces

```text
rd_dense[x_hi] = sum_x_lo eq(reverse(r_prefix))[x_lo] * rd_write_value(j).
```

InstructionInput supplies its resident rs1/rs2 midpoint tables instead of
replaying them again. The handoff must fail closed on device, allocation
identity, generation, length, table IDs 1 and 5, and the ordered prefix-point
digest. The BCSR component receipt must likewise carry the BCSR source
identity and generation, full `product_uniskip_tau_low` digest, split
geometry, three allocation identities, canonical encoding, and one-shot
consumption. Inputs remain immutable until the midpoint receipt is published.

All Fiat--Shamir work remains on the host. A shader receives a challenge only
after the host has checked and absorbed the preceding round polynomial. The
last challenge is applied exactly once before the three output openings are
published in `rd`, `rs1`, `rs2` order.

The exact Rust ABI, buffer slots, geometry, source alternatives, and executable
log-26 roof census are in `resident_bcsr.rs`. The component pair is encoded in
the existing stage-1 command buffer and adds no command buffer or wait. The
midpoint adds one dispatch, one command buffer, and one wait.

### Falsified column-replay ceiling and observed indexed result

| phase | useful half-width terms | cache-unique bytes | shader-requested bytes | compute floor | traffic floor/envelope |
|---|---:|---:|---:|---:|---:|
| BCSR components + reduction | 165,908,024 | 1,241,747,000 | 1,245,810,232 | 5.002051 ms | 2.749043 / 2.758038 ms |
| rd midpoint | 50,331,648 | 520,880,128 | 1,326,055,424 | 1.517477 ms | 1.153151 / 2.935689 ms |

The table is the pre-measurement column-replay model and is retained to record
the failed assumption. The arithmetic rate is the measured canonical half-width rate of 33.168
Gterm/s; traffic uses 451.702 GB/s. The optimistic combined floor is
6.519528 ms and its 80%-roof cap is 8.149410 ms. Charging every midpoint
equality load as a memory request gives a conservative 7.937740-ms floor and
9.922175-ms 80%-roof cap. The latter leaves 10.058941 ms under 5x, 4.350051 ms
under 7x, and 2.566022 ms under 8x for host work, equality generation,
publication, command/wait latency, and adapter overhead.

Host work remains explicit: `3P` stage-1 opening products, `2P` q-combination
products, and `4P + 8H - 12` prefix/dense products, or 139,252 full products
total. Excluding equality-table generation, their logical traffic is exactly
4,980,272 bytes. The envelope was not predictive because it treated event
replay as parallel useful work and omitted the hot-column critical path. At log
26 the column control flattened near `42.323 ms` GPU-active. The
indexed-predecessor candidate with 128 partials measured a Criterion interval
of `7.8876..7.9926 ms` GPU-active and `8.4411..8.6812 ms` resident wall. Its
warm diagnostic was `7.895 ms` active and `8.387 ms` wall. The candidate
consumes 1,074,266,112 source bytes, including the two dense index maps, and
improves the observed component mechanism by about `5.3x`. This is exact
component evidence, not complete-member promotion evidence. The first
register-column midpoint measured `6.7240 ms` active and an unstable
`18.792 ms` wall. Reusing the indexed position-parallel mechanism reduced that
to `2.0963 ms` active and `2.4249 ms` wall. The measured component and midpoint
wall medians sum to `10.9702 ms`, leaving `9.0109 ms` under the 5x cap for the
host prefix/suffix, Fiat--Shamir, receipt checks, and shared scheduling.

### Algebraic obligations

The protocol algebra is unchanged. Both component strategies pass the same
scalar parity fixture at log 16 across partial counts 8, 32, 64, 128, and 256.
The fixture includes a same-cycle read/write and shows that every component is
exactly

```text
Q_v[x_lo] = sum_x_hi eq(tau_hi)[x_hi] * v(x_hi || x_lo).
```

The remaining midpoint oracle must show that sparse rd-post replay equals the dense
partial bind at `reverse(r_prefix)`. Finally, InstructionInput must establish
that tables 1 and 5 are the same rs1/rs2 polynomials at that exact ordered
point, not merely buffers of the right length. Maximal-value fixtures must
also validate that 32 accumulated 192-bit products fit the stated 197-bit
bound before the one canonical reduction.

### Unresolved implementation evidence

1. The sibling BCSR receipt currently proves provenance and layout, but the
   runtime needs a borrow-only device-buffer view; it must not allocate or
   copy a replacement carrier.
2. The indexed rs1/rs2 maps need typed producer ownership and allocation
   identities; benchmark-local allocation is not admissible in production.
3. InstructionInput's two midpoint buffers need a typed same-point resident
   handoff. CPU reconstruction is not an admissible performance fallback.
4. A Metal capture must confirm register allocation, no spills, active SIMD
   residency, and whether midpoint
   equality reads achieve the cache-unique or requested-byte envelope.
5. Logs 27 and 28 need explicit partial tiling and max-buffer admission; the
   log-26 constants must not silently generalize.
6. The paired evaluator must charge the shared 2.181-GB BCSR producer once,
   include all host FS and waits, and compare five alternating complete-member
   samples against the durable five-pair CPU denominator.

The smallest next integration slice stays in this package: publish the
three-component receipt from the checked indexed runtime without copying its
resident BCSR/index inputs. The rd midpoint and InstructionInput alias receipt
are the second slice. Global
prover selection waits until both slices pass parity and the complete fair
boundary clears 5x; optimization continues toward 7x and then 8x while the
measured ceilings leave credible headroom.

## Observed Q-slice screen

On the retained Apple M4 Max, both registered accumulator variants passed
parity against the independent unfactored oracle, including maximal carry
chains. At log 26 with resident inputs and 128 threads per threadgroup, ten
same-binary alternating pairs measured:

```text
                         canonical 128-bit   deferred 224-bit   speedup
GPU active median:          6.092375 ms        13.246667 ms      2.174x
resident wall median:       6.701125 ms        13.892833 ms      2.073x
useful half-width terms: 201,326,592
```

Criterion independently measured the canonical path at `6.0699 ms` active
(`33.168` billion useful terms/s) and `6.5903 ms` resident wall. The canonical
path is now the default. It beats the Q slice's `9.580085 ms` conservative
80%-roof cap and uses 33.0% of the full member's `19.981116 ms` 5x budget.
Threadgroup widths 32, 64, 128, and 256 were indistinguishable on the deferred
control, so the gain comes from avoiding its register-heavy 224-bit live state,
not dispatch geometry. The complete member still needs the remaining rounds,
host transcript bridge, and end-to-end alternating validation before promotion.

## Durable denominator and historical local analysis

The production artifact
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json` was recorded
at revision `5f520c21e338632aa0bf5936ceb02be6c22fa40f`, log 26, on the retained M4
Max with 16 Rayon threads. Its five optimized-CPU attributed samples are:

```text
98.79929000001401 ms
101.61374800000340 ms
102.54645899995417 ms
99.90558200001716 ms
97.84945799999684 ms
```

The frozen median is `99.905582000 ms`. The hard local 5x cap is
`19.981116400 ms`; the pursue-if-credible 8x cap is `12.488197750 ms`.
The same artifact reports:

| log2(T) | optimized CPU attributed wall |
|---:|---:|
| 18 | 0.854751 ms |
| 20 | 1.863002 ms |
| 22 | 5.635414 ms |
| 25 | 58.908957 ms |
| 26 | 97.849458 ms |

One log-26 component trace assigns `80.362208 ms` to `prepare` and
`17.486916 ms` to all `prove_round` spans. That split motivates a resident
producer, but it does not permit input production to disappear from the Metal
numerator.

The final evaluator must freeze one fair boundary:

1. **Standalone member:** charge production of all resident inputs to Metal
   and retain the current complete optimized-CPU denominator.
2. **Resident PIOP:** produce the shared register planes once and charge the
   same producer policy to both arms. Do not divide a resident-input Metal time
   by the current CPU denominator if the CPU arm is charged a different input
   boundary.

Five alternating pairs from one stable binary, identical host Fiat--Shamir
accounting, proof verification, and complete producer/command/wait/host-tail
time are required for promotion.

## Exact relation

For cycle row `j`, define

```text
C(j) = rd_write_value(j)
     + gamma * rs1_value(j)
     + gamma^2 * rs2_value(j).
```

The summand is

```text
eq(product_uniskip_tau_low, j) * C(j),
```

has degree two, and binds cycle variables low-to-high. If binding-order
challenges are `r_0, ..., r_(n-1)`, all three outputs are opened at

```text
reverse([r_0, ..., r_(n-1)]).
```

The canonical outputs are `rd_write_value`, `rs1_value`, and `rs2_value`.
The latter two alias InstructionInput outputs at the identical stage-3 point.
The terminal verifier expression is

```text
eq(reverse(r), product_uniskip_tau_low)
  * (rd + gamma*rs1 + gamma^2*rs2).
```

## Split geometry and transcript ownership

Let

```text
n = log2(T)
prefix_vars = ceil(n / 2)
suffix_vars = floor(n / 2)
P = 2^prefix_vars
H = 2^suffix_vars.
```

The protocol point is big-endian and splits as `tau_hi || tau_lo`, where
`tau_hi.len = suffix_vars` and `tau_lo.len = prefix_vars`. The exact row map is

```text
j = (x_hi << prefix_vars) | x_lo.
```

The prefix decomposition is

```text
p[x_lo] = eq(tau_lo)[x_lo]
q[x_lo] = sum_x_hi eq(tau_hi)[x_hi] * C(x_hi || x_lo).
```

For every prefix round, the host computes the degree-two endpoints

```text
s(0) = sum_y p[2y] * q[2y]
s(2) = sum_y (2*p[2y+1] - p[2y])
                 * (2*q[2y+1] - q[2y]).
```

`s(1)` comes from the previous-claim hint. After the last prefix challenge,
the contiguous low-index fold point is `reverse(prefix_challenges)`.

The generic batch driver alone checks the round sum, combines members, absorbs
the round polynomial, and draws the next challenge. The Metal adapter receives
the pending challenge on the next `prove_round` call. It never hashes or draws
on device, and `finish_rounds` applies the final challenge exactly once.

## Historical resident PIOP architecture: dense-SoA partial-q handoff

There is a stronger producer boundary than a stage-3 scan. Stage 1 already
evaluates all 35 outer-remainder openings at

```text
point = reverse(stage1_remainder_challenges[1..])
      = product_uniskip_tau_low.
```

Its opening code splits that point at `floor(n / 2)`, so its `e_out` is
exactly `eq(tau_hi)`, its `e_in` is exactly `eq(tau_lo)`, and its row index is
`x_out * P + x_in`. Columns 8, 9, and 10 are respectively `rs1`, `rs2`, and
`rd_write`. Therefore the three `Q` component tables needed here are useful
intermediates of openings stage 1 must already produce:

```text
Q_v[x_in] = sum_x_out e_out[x_out] * v(x_out || x_in)
opening_v = sum_x_in e_in[x_in] * Q_v[x_in].
```

The current stage-1 opening partials cannot simply be retained: they reduce
over `x_in` for each `x_out`, the transpose of the table required here. A real
handoff must change the producer, not relabel its current output.

The implementation-ready producer is a companion projection over the canonical
register SoA, dispatched while stage 1 still owns the rows:

1. The existing opening shader omits columns 8, 9, and 10 but continues to
   evaluate the other 32 columns unchanged.
2. One projection thread owns one `x_in`, scans all `x_out`, and accumulates
   three exact 224-bit `e_out * u64` sums. It writes the three canonical
   `P`-element component tables. This is the same coalesced geometry and
   accumulator as the standalone q builder, without `gamma` or its final
   combination.
3. After the existing stage-1 completion wait, the host dots the component
   tables with `e_in` to supply the omitted three scalar openings. This is
   `3P` full products over 384 KiB at log 26.
4. Retain the component tables through stage 2. Once stage 3 draws `gamma`,
   combine them on the host into `q` with `2P` full products.

This removes the separate stage-3 full-domain q scan without requiring a
second projection. More importantly, it replaces stage 1's current work for
these openings--`3T` full-field inner products plus `3H` full-field outer
products--with the same `3T` half-width projection stage 3 would otherwise
need plus `3P` full products. The companion reads `24T` native bytes; those
bytes are charged to the shared producer even though the SoA itself is already
resident.

The typed handoff is:

```text
Stage1RegisterPartialQ {
    producer = OuterRemainder,
    generation,
    product_tau_low_digest,
    prefix_elements = P,
    rd_write_value: canonical Fp128[P],
    rs1_value:       canonical Fp128[P],
    rs2_value:       canonical Fp128[P],
}
```

Stage 3 rejects a producer id, generation, point digest, or length mismatch.
The carrier is only `48P` bytes: 384 KiB at log 26. It is immutable after
stage 1, survives stage 2, and is released immediately after stage 3 forms
`q`. No protocol or transcript change is needed.

At log 26, the projection moves exactly 1,611,137,024 GPU bytes:

```text
native planes + e_out + component writes = 24T + 16H + 48P.
```

Its traffic floor is 3.566816 ms, below its 7.663162-ms half-width arithmetic
floor. The host stage-1 dot moves another `64P + 48` logical bytes and the
stage-3 q combination moves `64P`. This accounting prevents the producer from
disappearing at either the standalone-member or resident-PIOP boundary.

A later 2D-tiled stage-1 opening/projection fusion may remove the `24T` read,
but the transpose needs extra partial tables because Metal has no 128-bit field
atomic. It is not the baseline and gets no analytical credit until its exact
partial traffic and capture beat the companion projection.

The stage-1 refactor is outside this isolated directory. Until it lands, the
self-contained architecture below is the implementation and microbenchmark
baseline. It has the same leading `4T` half-width work when the q producer is
charged, so it remains a fair falsification path for the frozen per-member CPU
denominator.

## Standalone architecture: alias-linear projection

The old sketch cached a `T`-element field-valued `C` plane and counted every
zero-extended `u64` multiplication as a full field product. That is not the
best architecture. Register values are native scalars, and the already-running
InstructionInput member computes the same partially bound `rs1` and `rs2`
tables at the same batch challenges.

The self-contained path is:

1. A shared producer exposes cycle-ordered `u64` structure-of-arrays planes for
   `rd`, `rs1`, and `rs2`.
2. Before round zero, one device thread owns one `x_lo`. It accumulates three
   exact 224-bit sums

   ```text
   Q_rd   = sum_x_hi eq(tau_hi)[x_hi] * rd(x_hi || x_lo)
   Q_rs1  = sum_x_hi eq(tau_hi)[x_hi] * rs1(x_hi || x_lo)
   Q_rs2  = sum_x_hi eq(tau_hi)[x_hi] * rs2(x_hi || x_lo),
   ```

   reduces each once, and writes

   ```text
   q = Q_rd + gamma*Q_rs1 + gamma^2*Q_rs2.
   ```

   A seven-limb accumulator is sufficient: every term is strictly below
   `2^192`, the shader ABI bounds the term count by `2^32`, and the sum is
   strictly below `2^224`. Per-term modular reduction is unnecessary.
3. The host runs all prefix messages and binds over the `P`-element `p` and
   `q` tables.
4. Because InstructionInput precedes this member in the stage-3 batch order,
   it has already applied the last prefix challenge when this member receives
   it. InstructionInput publishes immutable copies of dense table indices 1
   and 5 (`rs1`, `rs2`), their remaining length `H`, and the exact bound-prefix
   identity.
5. The second GPU command reads only the `rd` plane and computes

   ```text
   rd_dense[x_hi] = sum_x_lo eq(reverse(r_prefix))[x_lo]
                              * rd(x_hi || x_lo).
   ```

   It again accumulates exact 128-by-64 products before one reduction per
   thread.
6. The host forms

   ```text
   C_dense = rd_dense + gamma*rs1_dense + gamma^2*rs2_dense
   ```

   and discards `rd_dense`. It runs suffix messages over `eq_dense` and
   `C_dense` while also binding the two copied alias tables. At the terminal
   point it returns the bound alias scalars and recovers

   ```text
   rd = C* - gamma*rs1 - gamma^2*rs2.
   ```

   This needs no inversion, is exact for `gamma = 0`, and avoids carrying a
   fourth suffix table or adding a second end-of-stage dependency.

This path uses two large GPU commands and two synchronization boundaries. It
does not allocate a `T`-element field buffer.

### Why the midpoint alias is sound

Both members have the same number of stage-3 rounds, zero offsets, and the same
batch challenges. The generated verifier already declares the final `rs1` and
`rs2` openings as aliases and checks their values at identical point slices.
The handoff is therefore reuse of an existing prover table, not a protocol
change. Admission must nevertheless validate:

- producer member and table ids (`InstructionInput`, tables 1 and 5);
- exactly `prefix_vars` applied challenges;
- remaining length exactly `H`;
- the same ordered prefix-challenge digest or generation counter; and
- one-shot publication before either member applies the next challenge.

A stale or wrong-point table must be rejected. Copying `2H` fields at log 26
is only 256 KiB and is preferable to a lifetime-unsafe borrowed mutable slice.

### Shared producer and lifetimes

The fallback producer boundary is a canonical register-value SoA emitted
during the existing stage-1 witness/row walk:

| plane | ABI | log-26 bytes |
|---|---:|---:|
| `rd_write_value` | little-endian `u64[T]` | 512 MiB |
| `rs1_value` | little-endian `u64[T]` | 512 MiB |
| `rs2_value` | little-endian `u64[T]` | 512 MiB |

This should replace duplicated value storage in later row formats rather than
be an uncharged fourth copy. InstructionInput consumes `rs1`/`rs2`, this member
consumes all three, and RegistersReadWrite can retain them and add its own
`rd_pre` and metadata planes. Stage 1 already extracts all three values, so the
producer needs no second witness scan.

At log 28 each individual plane is exactly 2 GiB. Runtime admission must check
every plane against `maxBufferLength`; equality, q, alias, and dense tables are
lower order. The preferred lifetime is

```text
resident preferred:
q components: stage-1 opening projection -> stage-3 q combine -> release
rd/rs1/rs2: shared producer -> midpoint rd fold -> later consumers

standalone fallback:
rd/rs1/rs2: shared producer -> q build -> midpoint rd fold -> later consumers
q:          q build -> prefix host rounds -> release
rs aliases: InstructionInput midpoint snapshot -> suffix host rounds -> release
rd_dense:   midpoint fold -> suffix host rounds -> release
```

## Fallbacks and ablations

`DirectLinear` uses the same q build but rescans all three native planes at the
midpoint and returns three dense output tables. It requires no sibling handoff
and is the correctness fallback. Its leading work is `6T` half-width products
and `48T` native bytes.

`CachedCombinedControl` is an analytical ablation, not a preferred shader. It
would build `C` with `2T` half-width products, use `T` full products for q, and
use another `T` full products for the midpoint fold. It also moves `56T` bytes
and allocates `16T` extra bytes. It can win arithmetically only if the matched
full-field product rate exceeds the matched batched half-width rate; its larger
traffic and allocation still have to be charged. Do not restore the original
all-full-width implementation.

## Shader ABI and dispatch

`RegistersClaimParams` is four little-endian `u32` words with size 16 and
alignment 4:

```text
rows, prefix_elements, suffix_elements, reserved_zero.
```

Field buffers contain canonical four-limb, little-endian 16-byte
`SolinasFp128` elements. Concatenate the shader after the offset-specialized
`fp128.metal` and `simd_reduce.metal` sources.

The stage-1 producer adds
`solinas_registers_claim_build_components` outside this isolated source. Its
slots are 0 rd, 1 rs1, 2 rs2, 3 `eq_suffix`, 4 column-major
`[Q_rd[P], Q_rs1[P], Q_rs2[P]]`, and 5 params. It dispatches `P` threads at
width 128. Stage 1 encodes it in the same command buffer as the 32-column
opening scan, waits once, computes the three host dots, and publishes the
typed carrier. The standalone entry points in this directory are:

| entry point | buffer slots | dynamic threadgroup slot |
|---|---|---|
| `solinas_registers_claim_build_linear_q` | 0 rd, 1 rs1, 2 rs2, 3 `[gamma,gamma^2]`, 4 `eq_suffix`, 5 q, 6 params | none |
| `solinas_registers_claim_fold_alias_rd` | 0 rd, 1 `eq_prefix`, 2 `rd_dense`, 3 params | 0: one field per SIMD group |
| `solinas_registers_claim_fold_direct` | 0 rd, 1 rs1, 2 rs2, 3 `eq_prefix`, 4 column-major `[rd,rs1,rs2]`, 5 params | 0: three fields per SIMD group |

The resident PIOP path does not dispatch
`solinas_registers_claim_build_linear_q`; it consumes the typed stage-1
component carrier and forms `q` on the host. The listed build entry point is
the self-contained benchmark and fallback. It dispatches exactly `P` threads.
Adjacent lanes own adjacent `x_lo`s, so every fixed-`x_hi` plane read is
coalesced. The fold dispatches exactly `H` threadgroups; lanes cover contiguous
`x_lo`s. Initial widths are 128 threads, nonzero multiples of the measured
execution width 32. At width 128 the alias fold uses 64 bytes and the direct
fold 192 bytes of dynamic threadgroup memory.

The complete resident schedule is one stage-1 completion wait shared with the
outer openings, no register-claim command before round zero, one midpoint rd
fold command/wait, then a host suffix. Fiat--Shamir remains on the host. The
standalone schedule adds the q-build command/wait before round zero and is the
microbenchmark boundary used until the stage-1 carrier exists.

### First executable slice

The first slice contains only `solinas_registers_claim_build_linear_q`. Its
Rust invocation accepts three typed resident native planes, derives the four
shader parameters from their checked geometry, builds `eq_suffix` and
`[gamma, gamma^2]` during preparation, and allocates `q`. Production source
registration remains withheld until this slice passes parity and capture.

At log 26 the resident planes occupy 1,610,612,736 bytes. Preparation adds
32 bytes of gamma powers, 131,072 bytes of equality weights, and 131,072 bytes
of output, for 262,176 private bytes and 1,610,874,912 total resident bytes.
Parameters are inline command bytes and do not allocate a buffer. The runtime
checks the aggregate private addition against the device working-set limit and
checks every allocation against `maxBufferLength` before allocation.
The compulsory roof charges 1,610,874,880 bytes. Source-level loads issue
2,684,485,632 bytes if every repeated `eq_suffix` read reaches memory; capture
must determine how much cache/broadcast removes before using the lower number.

`resident_wall` starts before invocation-state validation and ends after the
command completes, the entire canonical output is read, and its FNV-1a digest
is computed. It excludes native-plane upload, equality generation, allocation,
and pipeline compilation because those happen in preparation. `gpu_active`
comes from the command buffer timestamps around the single dispatch. A fair
standalone-member benchmark must report preparation and native producer cost
separately; neither timer licenses hiding them behind another member.

The invocation rejects a non-Akita context, the wrong point length, mismatched
resident geometry, wrong buffer lengths or devices, changed or aliased
allocation identities, a non-32 execution width, an invalid threadgroup width,
noncanonical controls or output, an exceeded working set, and any non-completed
command. Callers cannot supply raw shader parameters.

## Exact work and projected ceilings

Retained same-machine controls are:

| control | measured rate |
|---|---:|
| large streaming device copy | 420.68 GiB/s = 451,701,710,520 B/s |
| best relevant full-field product | 32.33 Gproduct/s |
| six-accumulator full-field control | 18.10 Gproduct/s |
| conservative full-field control | 16.42 Gproduct/s |

The batched 128-by-64 accumulator is **unmeasured**. Its pre-registered
promotion floor is 26.272 Gterm/s. That is an admission requirement, not a
measured property of this shader.

For standalone `AliasLinear`, the exact GPU work is:

| phase | half-width terms | full products | compulsory GPU bytes |
|---|---:|---:|---:|
| q build | `3T` | `2P` | `24T + 16(H + P)` |
| rd fold | `T` | 0 | `8T + 16(P + H)` |
| total | `4T` | `2P` | `32T + 32(P + H)` |

The host additionally performs `2H` full products to form `C_dense`. Excluding
equality-table generation, its prefix/suffix message-and-bind core is exactly
`4P + 8H - 12` full products when it owns and binds both alias tables.

For the preferred resident path, keep two accounting views:

| boundary | producer | stage-3 q build | midpoint fold |
|---|---:|---:|---:|
| resident PIOP incremental | `3T` half terms + `3P` full products, fused into stage 1 | `2P` host full products | `T` half terms |
| frozen standalone member | charge `3T` half terms + `3P` full products and native producer bytes | `2P` host full products | `T` half terms |

At log 26, the standalone-charged resident design has `4T` half terms, `5P`
full products, and 2,149,318,704 projection/fold/handoff logical bytes,
including the host opening-dot and q-combine traffic but excluding later host
round-table traffic. The `4T` GPU floor is 10.217549 ms and its 80%-roof cap is
12.771937 ms; the `5P = 40,960` host full products remain explicit and must be
measured rather than priced at a GPU product rate. Before that host and fixed
work, the frozen CPU median leaves 9.763567 ms of headroom for 5x and
2.270648 ms for 8x. Thus the resident producer does not create an unfairly low
local numerator.

Within the complete PIOP, stage 3's incremental GPU work after the handoff is
only the `T`-term rd fold: 2.554388 ms at the promotion floor and 3.192985 ms
at 80% of roof. The shared projection remains charged to stage 1. Against the
current stage-1 implementation, the three register openings' isolated
arithmetic floor changes from 11.124374 ms (`3T + 3H` full products) to
7.664520 ms (`3T` half terms + `3P` full products), a projected 3.459854-ms
reduction. The resulting tables also eliminate the otherwise-identical
stage-3 q projection. This is an architectural projection, not an additive
wall-time claim: stage 1 also computes 32 other columns in the same shader.

At `T = 2^26`, `P = H = 8192`:

| phase | exact work | projected 100%-roof floor | projected 80%-roof cap |
|---|---:|---:|---:|
| q build arithmetic | 201,326,592 half terms + 16,384 full products | 7.664068 ms | 9.580085 ms |
| q build traffic | 1,610,874,880 bytes | 3.566236 ms | not binding |
| rd fold arithmetic | 67,108,864 half terms | 2.554388 ms | 3.192985 ms |
| rd fold traffic | 537,133,056 bytes | 1.189133 ms | not binding |
| complete GPU-active | 268,435,456 half terms + 16,384 full products | 10.218456 ms | 12.773070 ms |

The projected floor uses 26.272 Gterm/s for half width and the conservative
18.10 Gproduct/s for the lower-order full products. The complete GPU traffic is
2,148,007,936 bytes. Compute, not bandwidth, is the expected phase limiter.

Against the frozen denominator, the projected 100%-roof floor leaves
`9.762660 ms` for producer attribution, command/wait time, midpoint alias
publication, host rounds, and adapter work while retaining 5x. It leaves only
`2.269742 ms` against the 8x cap. Equivalently, with 1 ms of non-half-width
overhead the leading half-width rate needed for 8x is about 23.37 Gterm/s; with
2 ms it is about 25.59 Gterm/s. The 26.272-Gterm/s promotion floor therefore
makes 8x credible, so optimization must continue toward 8x rather than stop at
5x.

The self-contained `DirectLinear` fallback has `6T` leading half-width terms.
At the minimum half-width rate its arithmetic floor is about 15.326 ms and its
80%-roof cap about 19.157 ms before host and fixed work. It is a parity path,
not the default performance claim.

## Occupancy and capture gates

The q build's source-level live state is three seven-limb accumulators plus one
field coefficient, three native values, loop state, and reduction temporaries.
This is a structural estimate, not compiler register evidence. Per-term
modular reduction was removed both to reduce instruction count and to shorten
the dependency path.

The stage-1 component producer has the same long-loop state but omits the two
final gamma products. At log 26 its `P` threads form 64 width-128 groups, or
256 SIMD groups. That is enough logical parallelism for the 40-core target only
if capture shows the long loops remain distributed and register allocation
does not serialize residency; otherwise use two or four `x_out` tiles and
reduce lower-order partials.

Promotion requires a capture for every entry point reporting:

- execution width and maximum threads per threadgroup;
- compiler register allocation and register-limited resident SIMD groups;
- threadgroup-memory and thread limits;
- no spills or local-memory traffic;
- active cores/SIMD groups during the long q build;
- achieved useful half terms/s, full products/s, and bytes/s; and
- command-buffer gaps and both host synchronization intervals.

The matched arithmetic control must use the same seven-limb accumulation and
one final reduction, not the existing pointwise half-width chain. Require at
least 26.272 Gterm/s at two saturated sizes with relative MAD at most 3%.

## Hybrid cutoff

The conservative initial dispatch policy is CPU below log 25 and Metal at log
25 or above, only when the shared planes, stage-1 partial-q carrier, midpoint
alias handoff, promoted half-width primitive, and complete 5x admission all
pass. Without the stage-1 carrier, the same cutoff may select standalone
`AliasLinear` if its separately charged q build passes. At log 25 the projected
leading half-width floor is about 5.109 ms versus the measured 58.909-ms CPU
member, leaving about 6.67 ms of fixed headroom for 5x and about 2.25 ms for
8x. Logs 20 and 22 have much smaller fixed-latency budgets and are not safe
static cutoffs without measurements.

Freeze the final cutoff from an alternating sweep at logs 18, 20, 22, 25, and
26, then validate it unchanged. A per-input winner picker is inadmissible. The
cutoff may move below log 25 only after the complete path, not GPU-active time
alone, wins its fixed CPU control.

Choose the resident route before stage 1. That decision controls whether the
outer opening scan omits the three register columns and must therefore not be
made opportunistically when stage 3 starts. If a later admission check fails,
the CPU fallback remains correct, but the abandoned projection stays charged
to that sample. The standalone route does not alter stage 1 and can be chosen
at stage 3 from the same frozen log cutoff.

## Concrete blockers

1. The exact seven-limb batched half-width primitive has an isolated runtime
   and independent oracle tests, but neither has been executed. It has not
   been captured or shown to clear 26.272 Gterm/s.
2. No typed stage-1 partial-q handoff currently publishes the three component
   tables with a producer generation and `product_tau_low` identity. The
   current stage-1 opening partials have the wrong reduction orientation.
3. No typed stage-3 midpoint handoff currently publishes InstructionInput
   tables 1 and 5 with a bound-prefix identity.
4. The canonical shared register-value SoA producer does not yet exist. The
   current 48-byte InstructionInput row is not the preferred q-build ABI.
5. Producer attribution for the frozen per-member denominator is not fixed.
6. Production module/source registration is withheld. The isolated q-build
   runtime and GPU parity test exist, but no alternating benchmark is wired.

Until blockers 1--6 are resolved, the production dispatcher must stay on the
optimized CPU implementation.

### Withheld root integration

To execute this slice without admitting it to the prover:

1. declare `pub mod registers_claim_reduction` in `solinas/mod.rs`;
2. add this directory's `SOURCE` as a named fragment in `solinas/source.rs`
   after `fp128.metal` and before library assembly completes;
3. update the exact source-assembly test in `source.rs` with that fragment;
4. run the host oracle tests, then the single Metal parity test; and
5. add a Criterion target that prepares resident planes once and reports both
   `resident_wall` and `gpu_active`, plus preparation and producer charges as
   separate columns.

Do not add a prover adapter or hybrid selector in this integration. Promotion
still requires the parity, compiler-shape, occupancy, and timing artifacts in
the gates below.

## Required parity and evaluation cases

The integration harness must cover:

1. odd/even geometries and minimal one- and two-round domains;
2. the exact `x_hi * P + x_lo` row map and reversed prefix point;
3. `gamma = 0`, `gamma = 1`, and seeded nonzero gamma;
4. native values 0, both sides of `2^32`, and `u64::MAX`;
5. seven-limb pre-reduction carry chains at maximal coefficient/value/count;
6. linear q against cached, direct, and fully dense scalar oracles;
7. stage-1 partial-q handoff identity, its three reconstructed stage-1
   openings, and its stage-3 q against the standalone q build;
8. every prefix endpoint, previous-claim check, and bind;
9. midpoint `rd_dense` and both handed-off aliases element-for-element;
10. stale generation, wrong table id, wrong length, and wrong point rejection;
11. every suffix message/bind and final three output scalars;
12. final recombination and verifier `EqSpartan`, including `gamma = 0`;
13. canonical output order and clear/ZK generated-driver alias paths;
14. Rust/Metal size, alignment, slot order, and zero-reserved word;
15. max-buffer admission at logs 26, 27, and 28;
16. shader limits, spills, occupancy, and useful-rate capture; and
17. five alternating exact log-26 pairs, proof verification, and an untuned
    log-27 transfer run.

## Kill and escalation bars

- Reject Metal if the complete fair-boundary log-26 median exceeds
  `19.981116 ms`.
- Treat `14.272226 ms` as the working target and continue toward 8x whenever
  the complete result reaches `12.488197 ms` or capture plus measured fixed
  overhead makes it credible.
- Reject `AliasLinear` if the midpoint handoff cannot fail closed on table and
  point identity; retain `DirectLinear` for correctness.
- Redesign the q build if it misses its matched 80%-roof cap or capture shows
  insufficient active SIMD groups, spills, or local memory.
- Do not hide producer or synchronization time behind another member. Cross-
  member fusion is allowed only with a fixed PIOP attribution rule and a paired
  end-to-end result.
- A phase below 80% of its matched roof is unfinished even if another phase
  makes the aggregate clear 5x.

This slice is an implementation contract and falsification plan, not
performance evidence.
