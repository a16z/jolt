# Metal sumcheck backend

The Metal backend will specialize Jolt's Akita field, keep dense round state on the
GPU, and synchronize only the round polynomial with the host transcript. Each slot
starts as the optimized CPU implementation. A slot moves to Metal only after an exact
oracle benchmark shows that its complete hybrid path, including handoff and any final
readback, beats that CPU implementation above a measured crossover.

The implementation base is draft PR #1732 at `c10b26986`. That branch contains the
Akita prover and its modified copy of the optimized CPU backend from draft PR #1714.
Merging #1714 back into it is not a valid update procedure because the Akita branch
changes the same kernels. Future rebases must reconcile the two PRs semantically.

## Requirements and limits

- **Requirement:** round polynomials, challenges, output claims, and proof bytes must
  equal the optimized CPU path. Tests compare field values exactly; tolerance-based
  GPU checks are invalid.
- **Decision:** Fiat-Shamir stays on the host. A round reads at most the univariate
  message from shared memory, absorbs it, draws the challenge, and writes that
  challenge to a small shared parameter buffer.
- **Decision:** a Metal round does not allocate. Two full-capacity buffers hold the
  shrinking factor tables and alternate as source and destination. Equality-table
  levels, reduction scratch, parameter buffers, and pipelines are allocated during
  `prepare`.
- **Decision:** portfolio acceptance compares optimized-CPU PIOP wall time with
  Metal-hybrid PIOP wall time. The `jolt_prover::piop` span contains stages 1-7 and
  Akita reconstruction. It excludes trace and witness generation, stage-0
  commitments, stage-8 PCS openings, and verification.
- **Decision:** each slot's primary microbenchmark is its complete hybrid wall time.
  It includes direct handoff, every command submission and host transcript step, and
  any Metal-to-CPU tail handoff. Resident GPU-active time remains diagnostic.
- **Constraint:** Metal currently supports only
  `jolt_field::AkitaField`, whose modulus is
  `2^128 - 0xffffa7f7`. Buffers contain canonical little-endian values at a 16-byte
  stride. The first implementation converts through `FixedBytes<16>`; a direct cast
  needs separate size, alignment, representation, and aliasing proofs.
- **Constraint:** public Metal properties do not expose enough register and resident
  SIMD-group data to prove occupancy. Throughput and pipeline limits can be measured
  now; an Instruments capture is still required for an occupancy claim.

Commitment, opening, and Akita NTT work are outside this sumcheck port. The backend
may retain their existing CPU implementations.

## Round state and control flow

`MetalBackend` owns one device, command queue, compiled library, and immutable tuning
table. A `ProofSession` owns buffers whose lifetime spans a proof. Each Metal slot
implements the existing `PrepareKernel` and `SumcheckKernel` interfaces, so the stage
drivers and transcript code remain unchanged.

```text
optimized CPU prepare / irregular prefix rounds
                    |
                    v
       canonical handoff into resident buffer A
                    |
          +---------+------------------------------+
          |                                        |
          | GPU: bind A -> B and form next message |
          | GPU: reduce to degree+1 field values   |
          | host: read message, Fiat-Shamir, write r|
          | swap A/B                               |
          +------------------- while n > cutoff ---+
                    |
                    v
       one canonical readback, optimized CPU tail
```

The first slot is `InstructionReadRaf`. Its 16 sparse 128-bit address phases use a
resident table-major Metal layout, while the small address round tables and
Fiat-Shamir challenges return to the host. The current cycle path derives the first
message on the CPU, then writes five half-domain tables directly into one shared
Metal buffer. Metal uses a resident five-factor fused bind-and-message schedule
until the measured cutoff and the optimized CPU finishes the short tail. The open
iteration removes that CPU cycle seam by deriving the factors from the resident
compact rows on device.

A command failure before a message is absorbed may retry on the CPU only while an
equivalent host state still exists. After the host state is released, or after the
message changes the transcript, the prover returns
`SumcheckError::ComputeBackend`; device errors never enter proof data.

## Throughput model and crossover

For round `i`, let `M_i` be useful field multiplications, `B_i` the optimistic unique
device bytes, `R_mul` the measured arithmetic ceiling, `R_mem` the measured copy
ceiling, and `L_cmd` the command wall-time floor. Host transcript time is `H_i`.
Because every challenge depends on the preceding message, rounds cannot overlap:

```text
T_metal >= T_handoff
         + sum_i (L_cmd + max(M_i / R_mul, B_i / R_mem) + H_i)
         + T_readback
```

Every kernel worksheet records `M_i`, `B_i`, arithmetic intensity, buffer capacity,
message size, and the measured fractions of the arithmetic and copy roofs. Logical
bytes count one read of each factor and one write of each bound value. Cache reuse,
equality weights, reduction scratch, and coherence traffic are reported separately,
so the optimistic byte count is never described as physical DRAM traffic.

Fusion is preferred when it consumes a value in registers before storing it:
linear-combination leaves first, product samples second, reduction third, and the
next bound state last. The candidate loses if it writes a table that the next kernel
immediately rereads without a protocol dependency between them.

The crossover is selected over complete tails, not isolated rounds. For each possible
cutoff `k`, the evaluator measures GPU rounds down to `2^k`, one readback, and the
optimized CPU tail. The retained cutoff minimizes median hybrid wall time and must
beat its neighbors by more than the measured noise floor. Unsupported geometry uses
the CPU slot from the start.

## Portfolio target

The primary scalar is

```text
S_piop = optimized CPU jolt_prover::piop wall time
         / Metal hybrid jolt_prover::piop wall time.
```

The minimum accepted result is `S_piop >= 4` at a padded `2^26` trace. Four is a
floor, not an optimization cap. After reaching it, the goal loop continues whenever
the remaining independently attributed kernel shares and conservative local
speedups predict at least another 5% PIOP improvement. With current-Metal PIOP shares
`f_i` and conservative speedups `s_i` over the currently selected paths, the
projection is

```text
S_projected = S_current / (1 - sum_i f_i * (1 - 1 / s_i)).
```

Only disjoint shares may be combined. Before implementation, a candidate may use a
current Metal profile and a conservative traffic/arithmetic ceiling. Once a working
kernel exists, measured complete-hybrid speedup replaces that estimate. This keeps
the stretch policy uncapped without treating an unsupported peak-ALU number as
attainable throughput.

The local bars are promotion minimums, not targets:

| CPU PIOP share before the port | Promotion minimum | Working target |
|---:|---:|---:|
| at least 5% | 4x complete-hybrid speedup | 5x or the measured ceiling |
| 1-5% | 3x | 4x or the measured ceiling |
| below 1% | 2x, or reuse already-resident state | 3x; otherwise retain CPU |

Before shader work, the conservative analytical ceiling must exceed the promotion
minimum by 25%. A kernel that misses its bar stays on CPU unless it removes a handoff
or supplies residency reused by a later hot kernel.

## Port ledger

The 31 ordinary sumcheck slots and two uni-skip fronts are tracked below. `analyze`
means that no shader work starts until the relation expression, CPU data layout,
`M_i`, `B_i`, and expected crossover are recorded. Profiling data will determine the
order after the first slot establishes the harness.

| Stage | Slot or shared implementation | Initial Metal plan | State |
|---|---|---|---|
| 1 | `spartan_outer_uniskip` | centered-domain reduction | shader retained at 8.53x seam estimate; backend handoff integrating |
| 1 | `spartan_outer_remainder` | dense fused product | analyze |
| 2 | `spartan_product_uniskip` | centered-domain reduction | analyze |
| 2 | `spartan_product_remainder` | dense fused product | analyze |
| 2 | `ram_read_write` | sparse CPU front, dense cycle tail | analyze |
| 2 | `instruction_claim_reduction` | dense fused product | analyze |
| 2 | `ram_raf_evaluation` | sparse reduction | analyze |
| 2 | `ram_output_check` | address-domain product | analyze |
| 3 | `spartan_shift` | dense fused product | analyze |
| 3 | `instruction_input` | dense fused product | analyze |
| 3 | `registers_claim_reduction` | dense fused product | analyze |
| 4 | `registers_read_write` | sparse CPU front, dense cycle tail | analyze |
| 4 | `ram_val_check` | dense fused product | analyze |
| 5 | `instruction_read_raf` | resident address scans, five-factor Metal cycle | address + cycle integrated; cycle seam optimizing |
| 5 | `ram_ra_claim_reduction` | dense fused product | analyze |
| 5 | `registers_val_evaluation` | dense fused product | analyze |
| 6a | `bytecode_read_raf_address` | address pushforward | analyze |
| 6a | `booleanity_address` | address pushforward | analyze |
| 6b | `bytecode_read_raf_cycle` | sparse-to-dense cycle reduction | analyze |
| 6b | `booleanity_cycle` | sparse-to-dense cycle reduction | integrated; 4.85x real kernel seam at `2^26`, further ceiling work open |
| 6b | `ram_hamming_booleanity` | dense cubic | analyze |
| 6b | `ram_ra_virtualization` | one-hot virtualization | analyze |
| 6b | `instruction_ra_virtualization` | one-hot virtualization | analyze |
| 6b | `inc_claim_reduction` | dense reduction | analyze |
| 6b/7 | trusted-advice cycle/address | resident two-phase reduction | analyze |
| 6b/7 | untrusted-advice cycle/address | resident two-phase reduction | analyze |
| 6b/7 | bytecode-reduction cycle/address | resident two-phase reduction | analyze |
| 6b/7 | program-image cycle/address | resident two-phase reduction | analyze |
| 7 | `hamming_weight_claim_reduction` | packed one-hot reduction | analyze |

The four shared precommitted implementations account for eight backend slots. Their
cycle state already crosses stages through `ProofSession`; the Metal version must
retain the same ownership rule rather than upload it twice.

## Fixed experiment contract

Each port gets a versioned run directory containing an immutable `run.json`, its
SHA-256 digest, append-only `events.jsonl`, and raw output for every trial. A trial may
change only its slot's shader, Rust dispatch code, and declared tuning values. The
CPU oracle, input generator, transcript implementation, metric parser, workload,
warmup, sample count, and correctness checks are frozen for that run.

The primary scalar is optimized-CPU wall time divided by hybrid wall time at the
contract's primary size; larger is better. Guards require exact messages,
challenges, final table values, and output claims at all validation sizes, no Metal
validation errors, no extra round allocation, and peak buffer bytes within the
contract. A candidate is retained only when order-inverted runs separate by more
than the baseline noise floor. The final validation repeats the winner in a fresh
process, runs adjacent sizes and cutoff values, and exercises the real
`PrepareKernel` path.

The initial foreground budget is 12 parameter trials or 30 minutes, whichever comes
first. This covers threadgroup and cutoff selection; shader edits start a new phase
with a new contract. No unattended loop starts until a valid baseline and noise floor
have been recorded.

### Goal-mode controller

`autoresearch/piop_goal.json` freezes the PIOP boundary, 4x floor, uncapped stretch
policy, local promotion bars, and phase budget. Goal mode uses the repository as its
memory: recover any interrupted kernel transaction, finish its fixed phase, validate
the retained candidate, integrate it, emit a fresh `2^26` PIOP profile, then select
the largest remaining conservative time saving. Reaching 4x alone is not a stopping
condition.

The portfolio evaluator runs identical Akita workloads with the optimized and Metal
backends in alternating order. Both runs must verify, and each trace must contain one
complete PIOP span:

```bash
python3 scripts/metal_piop_eval.py --log-n 26 --repeats 3
```

After profiling residual kernels, the deterministic continuation check is:

```bash
python3 scripts/metal_autoresearch.py goal-decision \
  crates/jolt-kernels/autoresearch/piop_goal.json \
  --current-speedup <S> \
  --candidate '<kernel>:<current-PIOP-share>:<conservative-local-speedup>'
```

Below 4x it always returns `continue: true`. At or above 4x, it still returns true
when the conservative aggregate projection clears the 5% continuation threshold or
any active kernel has a conservative local estimate strictly above 4x.
Each kernel phase retains its own immutable evaluator, snapshots, and JSONL lineage;
changing the kernel, algorithm, or evaluator starts a new phase rather than mutating
the previous run.

### Retained first-slot experiment

The robust `2^22` search used three fresh evaluator processes per candidate. Its
optimized-CPU/hybrid baseline was 1.5225x, the comparison relative MAD was 1.82%, and
the promotion threshold was fixed at 5.46%. Twelve cutoff/threadgroup candidates all
passed exactness guards; none cleared the promotion threshold. The retained values
are cutoff `2^16`, 128 message threads, and 64 transition threads.

A separate `2^24` Criterion validation measured 130.85 ms for the optimized CPU
cycle sequence, 26.898 ms for resident/direct-handoff Metal (4.87x), and 50.277 ms
when copying the five initial tables was included (2.60x). The real slot uses direct
materialization into shared memory. Round-polynomial parity, final-claim parity, and
the modular Akita proof/verifier path all pass with Metal forced above a cutoff of
eight elements.

### Target-scale profile and port order

The first exact evaluator run at a padded `2^26` Fibonacci trace produced a 20.418 s
optimized-CPU PIOP and a 20.500 s Metal-hybrid PIOP, or 0.996x. Both proofs verified
and each trace contained one complete PIOP span. This is one CPU-first pair, so it is
baseline evidence rather than a promoted comparison. Its purpose is to establish the
target-scale attribution before further shader work.

The optimized trace's largest disjoint kernel seams were:

| Kernel | PIOP wall time | PIOP share |
|---|---:|---:|
| `Booleanity` | 3.929 s | 19.24% |
| `InstructionReadRaf` | 3.583 s | 17.55% |
| `SpartanOuterUniskip` | 2.355 s | 11.53% |
| `InstructionRaVirtualization` | 2.227 s | 10.90% |
| `BytecodeReadRafCycle` | 1.273 s | 6.24% |
| `RegistersReadWriteChecking` | 1.053 s | 5.16% |
| `BooleanityAddressPhase` | 1.011 s | 4.95% |
| `InstructionInput` | 0.777 s | 3.80% |

The current full `InstructionReadRaf` path measured 3.429 s under Metal, only 1.04x
faster than its 3.583 s optimized-CPU oracle. The retained cycle-only result remains
4.87x at `2^24`; it did not satisfy the complete-hybrid bar because the 128 CPU
address rounds dominate. Stage 5 is therefore reopened before moving to Booleanity.

After integrating all 16 resident address phases, a fresh exact `2^26` pair at
`c44c6e368` measured 20.642 s optimized CPU and 19.500 s Metal-hybrid PIOP, or
1.059x. `InstructionReadRaf` fell from 3.697 s to 2.181 s (1.69x), and stage 5 fell
from 4.089 s to 2.561 s. Both proofs verified. This is a retained correctness and
directional-performance checkpoint, not a completed port: the complete kernel still
misses its 4x promotion floor.

Applying the working targets (5x for shares above 5%, 4x for 1-5%, 3x below 1%) to
the target profile gives an Amdahl projection of about 4.19x from the current path.
That calculation is not a forecast: it shows that the 4x portfolio floor requires
broad coverage and cannot be obtained from the cycle tail alone. Exact hybrid
measurements replace each local target as ports land.

After the Booleanity cycle port, the exact CPU-first `2^26` pair in
`benchmark-runs/metal-piop-eval/20260804-105011` measured 21.308 s optimized CPU and
15.515 s Metal-hybrid PIOP, or 1.373x. `Booleanity` fell from 3.986 s to 821.1 ms
(4.85x at its real kernel seam), while `InstructionReadRaf` measured 3.715 s versus
1.030 s because its stage-5 prepare now also creates the shared Booleanity row
buffer. Both proofs verified and each trace contained exactly one PIOP span. One
pair is directional evidence; promotion of the aggregate result still requires the
contract's interleaved repetitions.

### Instruction-read address worksheet

There are 16 eight-variable address phases. In every phase the RAF scan reads one
40-byte packed row and one 16-byte condensed equality weight. Phases 1-15 also write
the updated 16-byte weight after multiplying by the previous phase's 256-entry
equality table. In the conservative all-rows-have-a-table case, the suffix scan reads
one 4-byte bucket index, the same 40-byte row, and the same 16-byte weight. Ignoring
cache reuse and reduction scratch, the complete address prefix therefore moves

```text
B_raf    = T * (56 + 15 * 72) = 1,136 T bytes
B_suffix = T * (16 * 60)      =   960 T bytes
B_total  = 2,096 T bytes.
```

At `T = 2^26`, this is 131 GiB. The measured copy roof of roughly 420 GiB/s gives an
optimistic traffic floor near 0.31 s, compared with 3.583 s for the complete CPU
kernel. A 5x complete-kernel result permits about 0.717 s, or 45 ms per address phase
before crediting the already-fast cycle tail. The ceiling has enough margin over the
4x promotion bar to enter implementation.

The first probe is deliberately narrower than the final port. It executes one exact
RAF phase scan and produces the six 256-bin reductions (`shift_half`, `left`,
`right`, `shift_full`, `identity`, and `upper_all_ones`). Packed rows and weights
remain device-resident. The phase loses if its complete wall time at `2^26` cannot
stay below 45 ms after tuning its occupancy controls. Passing that gate admits the
table-suffix scan and the full 16-phase sequence; failing it forces a different keyed
reduction design before more of the relation is ported.

The first keyed-reduction candidate gave each SIMD group a private 1,536-field device
table, bitonic-sorted each 32-row batch by `(chunk, RAF flag)`, performed segmented
field sums in registers, and reduced the group tables in a second kernel. It used
zero static threadgroup memory, so occupancy was not limited by the 32 KiB
threadgroup-memory ceiling. Exact Akita-field tests passed for all phase shapes. The
candidate was nevertheless discarded:

| Rows | Optimized CPU | Metal wall | Metal active | Speedup |
|---:|---:|---:|---:|---:|
| `2^16` | 0.233 ms | 4.499 ms | 4.691 ms | 0.052x |
| `2^22` | 3.432 ms | 4.799 ms | 4.596 ms | 0.715x |

At `2^22` the GPU had 64 independent SIMD groups, and the projected `2^26` wall time
was roughly 77 ms. The in-register sort, not input traffic or static occupancy, was
the limiting work.

The next candidate is a byte-radix schedule: form per-group 256-bin histograms,
prefix them on device, scatter 32-bit row indices, then assign one dense reduction
threadgroup to each chunk. This rereads the row key during scatter and adds an index
stream, but removes every per-row sort and gives the field reduction coalesced bucket
ranges with only a few KiB of dynamic threadgroup memory.

The index-radix candidate improved the `2^22` phase to 1.499 ms wall (1.319 ms
active), 2.30x faster than the 3.444 ms CPU scan and 3.20x faster than SIMD-sort
Metal. At `2^26`, however, it measured 38.66 ms wall versus 47.22 ms CPU, only 1.22x.
Replacing the two strided row-key reads with a resident one-byte key plane did not
change the target-size time. This falsified key extraction as the scale bottleneck;
the final reduction's 32-bit indices still gather sparse 40-byte rows and weights.

The successor radix layout uses a 9-bit `(chunk, RAF flag)` key and scatters a compact
32-byte contribution: the field weight and two packed scalar operands. Scatter reads
each source row and weight once in cycle order. Reduction reads contributions
contiguously, performs the exact scalar field products, and writes three disjoint
lanes per key. Its optimistic phase traffic is about 100 bytes per row, all sequential
except the contribution scatter.

This layout restored locality at the target scale. The exploratory width sweep at
`2^26` reduced Metal wall time from 23.65 ms at 128 threads to 18.27 ms at 256,
17.46 ms at 512, and 17.01 ms at the device maximum of 1024. Halving the 64K-row
group to 32K was flat at 16.98 ms; increasing it to 128K regressed to 21.72 ms with
high variance. The retained geometry is therefore 64K rows and 1024 threads.

The benchmark was then corrected to keep the CPU weights natively in `AkitaField`
instead of converting from the Metal ABI in every row. At `2^26`, the corrected
uncondensed scan measured 49.74 ms CPU versus 16.42 ms Metal, or 3.03x. Fusing the
previous phase's field-weight condensation into scatter raised Metal to only
18.19 ms, while the production-shaped CPU condensation followed by scan measured
62.36 ms, a conservative 3.43x speedup. Inverting benchmark order left Metal at
18.19 ms and moved CPU to 70.30 ms, so the conservative comparison uses the faster
CPU-first result.

At `2^28`, the uncondensed scan measured 193.85 ms CPU versus 66.43 ms Metal (2.92x),
and the fused scan measured 256.45 ms versus 73.50 ms (3.49x). Metal took 4.04x as
long for 4x as many fused rows. Its approximately 116 bytes of logical per-row
traffic correspond to 424 GB/s at that point, so the contribution layout is at its
bandwidth ceiling. The 32-byte contribution write and read account for 64 bytes per
row; replacing that materialization with compact tile partials is the next candidate
because it has enough analytical headroom to clear 4x rather than merely approach
it. Table suffix accumulation remains outside both sides of these microbenchmarks.

The direct candidate realizes that layout without assuming lossy atomics. Each
threadgroup holds all `512 keys * 3 lanes` as four wrapping 32-bit limbs plus a
fifth word that counts `2^128` carries. This uses 30,720 of the M4 Max's 32,768
threadgroup-memory bytes. A final exact reduction maps each carry to the Solinas
offset before reducing the compact per-tile field partials. With 1024 threads this
permits one full-width resident threadgroup per GPU core; the target sizes expose
thousands of independent tiles.

For 32K-row tiles, fused source traffic is 50 bytes per row (key, lookup, weight
read/write) and partial write/read traffic adds 1.5 bytes per row, down from 116.
The `2^26` sweep measured 9.80 ms wall at 64K rows, 9.28 ms at 32K, and 9.49 ms at
16K; 32K is the provisional winner. Against the 68.14 ms native-Akita CPU result,
the 32K cold run is 7.34x. A later heat-soaked order-inverted run measured 14.69 ms
Metal and 86.42 ms CPU; comparing that slower Metal result with the earlier faster
CPU still gives a conservative 4.64x. At `2^28`, the paired result was 293.06 ms
CPU versus 38.55 ms Metal, or 7.60x, with 4.16x Metal scaling for 4x the rows.
The candidate therefore clears the address RAF-plus-condensation bar even under
the observed thermal range. It is not yet the complete address phase: the next
experiment must fold per-table suffix accumulation into the same resident scan.

The first suffix-layout probe accumulated the `One` suffix for all 40 tables. A
cycle-major source with resident table buckets was exact but scaled poorly: at
`2^26` it took 18.48 ms versus 70.87 ms CPU (3.83x), because each table stream
gathered roughly every fortieth weight and repeatedly fetched mostly unused cache
lines. Reordering the address-phase rows and weights table-major reduced the same
Metal kernel to 2.60 ms versus 71.43 ms CPU, or 27.45x. RAF accumulation is
order-independent and can consume that same layout. Stage 5 does not need an inverse
permutation: the condensed weights die at the address/cycle handoff, while the cycle
bases use the original packed rows and the 16 small phase tables. The retained
address-session design is therefore table-major; the next implementation expands
the suffix tile from `One` to each table's actual one-to-four suffix functions.

The full suffix tile evaluates all 43 `Suffixes` variants in the shader and selects
each table's actual one-to-four terms from a compact descriptor. It reads one 16-byte
lookup and one 16-byte field weight per selected row, evaluates the table's suffixes
in registers, and accumulates directly into `4 * 256` exact Solinas fields. Each
field uses four wrapping atomic limbs plus a fifth `2^128`-carry counter, for 20,480
bytes of dynamic threadgroup memory. The 65,536-row tile bound keeps every carry
correction within `u64`; a final kernel reduces the compact per-tile fields.

Exact tests cover every table and suffix at lengths 0, 8, 32, 56, 64, 112, and 120.
At `2^22`, the production-shaped optimized CPU scan measured 12.84 ms, versus
0.612 ms Metal wall and 0.433 ms active (20.98x and 29.6x). At `2^26`, 32K-row
tiles measured 223.22 ms CPU, 6.25 ms Metal wall, and 6.19 ms active (35.7x wall).
Increasing to 64K rows halved partial storage from 34.1 MB to 17.0 MB and measured
222.68 ms CPU, 6.07 ms wall, and 6.19 ms active; the active-time difference is below
the noise floor, so 64K is retained for its smaller resident scratch.

The 64K layout moves an optimistic 32 bytes of source data per selected row plus
about 0.51 bytes per row of partial write/read traffic at `2^26`. Its 6.19 ms active
time corresponds to roughly 328 GiB/s, 78% of the measured 420.68-GiB/s copy roof.
That roof gives a 4.83-ms lower bound and only about 1.28x remaining local headroom.
The kernel is therefore ready for the resident 16-phase evaluator: further suffix-
only tuning cannot save enough PIOP time to justify a more complex reduction before
the real table distribution and direct-RAF handoff are measured together.

The resident phase sequence allocates the table-major lookup, RAF flag, and weight
planes once. A phase runs condensation and direct RAF first, then reads the updated
weights in the full suffix tile, all in one command buffer. It performs no device
allocation, upload, or permutation between phases and returns only the exact RAF and
suffix bin tables used to build the host round message. A two-phase test checks the
in-place condensation handoff against independently prepared direct-RAF and suffix
invocations.

For a condensed `2^26` phase with every table equally represented, the complete
production-shaped CPU control measured 290.39 ms and the resident Metal phase
measured 15.81 ms wall / 16.20 ms heat-soaked active, a wall speedup of 18.36x. The
optimistic logical traffic is about 82.3 bytes per row including compact partial
write/read, corresponding to roughly 317 GiB/s at the active time, or 75% of the
measured copy roof. The empirical traffic floor is about 12.2 ms, leaving only
1.33x local headroom. The next evaluator therefore measures all 16 real address
phases, their one-time table-major handoff, CPU message construction, and the cycle
tail rather than further tuning this isolated dispatch.

That evaluator attributes 718.2 ms to preparation and 1,463.2 ms to round proving.
Within the rounds, all address work takes 296.6 ms and the resident dense cycle
sequence takes 56.7 ms, but the first compact-row cycle message takes 650.7 ms and
the following CPU-derived dense handoff takes 462.0 ms. The raw address shaders are
therefore no longer the limiter; the CPU/device representation boundary is.

The next candidate keeps an inverse cycle-to-table-major index beside the resident
compact rows. Its first cycle-message shader reads the two compact rows for each
cycle pair, derives the four RA factors from the 16 cache-sized phase tables, and
reduces the five Gruen evaluations without materializing dense inputs. After host
Fiat-Shamir, a second shader fuses the first cycle bind, half-domain materialization,
and the following message into the resident product buffers.

For `T` rows, the two dispatches derive `12T` RA multiplications each. Including
the five-factor message and bind work gives an optimistic total near `43T` useful
field multiplications. At `T = 2^26`, the measured 16.4-Gmul/s direct-handoff rate
places a conservative compute floor near 0.18 s, versus 1.11 s for the two CPU seams.
The compact-row reads plus the one required half-domain write are below that compute
floor at the measured copy roof. Direct parallel population of shared Metal buffers
must then reduce the 718-ms preparation cost; a 0.20-s preparation budget, the
measured 0.30-s address work, and a 0.18-s cycle seam put a roughly 5x complete-slot
result within the empirical roofs. Because this is clearly above the 4x floor, the
goal controller keeps the slot open.

The compact-row seam and preparation iterations were retained through `9167000c7`.
Against the 3.697-s optimized-CPU `InstructionReadRaf` control at `2^26`, the current
Metal slot measures 821.0 ms: 314.6 ms of preparation and 506.4 ms of proving, or
4.50x. The 16 slow address-phase calls account for 280.6 ms; resident first-message,
handoff, dense rounds, and readback spans account for another 190.8 ms. The same
single-profile pair measures 20.642 s CPU versus 17.490 s Metal for the full PIOP,
or 1.18x. These are bottleneck and direction measurements, not the final three-pair
promotion result.

A cache-reuse candidate fused each RAF tile with its suffix tile and reused the same
30 KiB threadgroup allocation sequentially. At synthetic `2^22`, a 16K tile improved
complete phase wall time from 1.686 ms to 1.453 ms; 32K and 64K were worse, and 4K
reversed the trend. The gain did not transfer to the real `2^26` distribution: the
16 phase calls were effectively flat at 279.5 ms and the complete slot regressed to
832.6 ms. That design is rejected. A successful successor must remove the second
logical row scan, not merely schedule it closer to the first scan.

The next candidate did remove the second scan. Partitioning each table by RAF flag
reduced the RAF accumulator to 15 KiB; deriving the `One` suffix from its lane zero
left another 15 KiB for the three nontrivial suffix lanes. Both relations therefore
fit in the M4 Max's 32 KiB threadgroup store and consume every selected row once.
The table-major implementation paid for an inverse permutation and compact-row
scatter on the host. In a fresh `2^26` pair it measured 895.5 ms for the complete
slot versus 3.725 s optimized CPU, or 4.16x. Its 253.4-ms address sequence beat the
retained two-scan sequence, but preparation rose to 441.5 ms, including 136.1 ms of
layout and 146.9 ms of scatter. It lost to the retained 821.0-ms slot.

A cycle-order variant replaced the inverse permutation with a grouped 4-byte index
stream. Compact rows stayed in cycle order, making preparation sequential and the
cycle handoff direct, while the single-scan shader gathered rows through the index
stream. After removing redundant internal permutation and tag validation, its
`2^24` exact evaluator measured 101.4 ms preparation and 22.0 ms phase wall time;
the table-major form measured 148.5 ms and 16.6 ms. At `2^26`, the complete slot was
871.6 ms: 334.1 ms preparation and 537.5 ms proving, with the 16 address calls taking
339.4 ms. The indexed form recovered 23.8 ms relative to table-major single-scan but
remained 50.6 ms behind the retained two-scan checkpoint. It is also rejected.

These results separate logical traffic from effective traffic. Single-scan moves
about 49 bytes per selected row in table-major order or 53 bytes with an index,
versus about 81 bytes for two scans, but the host permutation or random device
gathers consume the theoretical saving. A successor needs either an already-resident
row layout shared by later kernels or a producer that emits grouped rows directly;
paying to transform stage-5 rows for this slot alone does not clear the evaluator.

### Booleanity cycle worksheet

`Booleanity` was the next profiled slot. At the Akita `K = 256` geometry it checks
`P = 29` one-hot columns over `T` cycles. The CPU keeps the columns index-encoded for
four binds and materializes at width 16. A dense-from-start Metal implementation
would discard that optimization, so the retained sequence has two states:

- A 40-byte row contains the instruction index, mapped PC, RAM address, and signed
  fused increment. One SIMD group handles a cycle pair; its first five lanes load
  the row words and broadcast them, and one lane per polynomial derives its hot
  chunk and gathers the resident `K`-entry branch table.
- The initial message and lazy binds double the branch width. The dispatch that
  reaches width `W` also materializes `P * T/W` dense values directly into the first
  ping-pong buffer and computes that round's message from the same registers.
- Dense dispatches fuse one bind, the next two-lane Booleanity message, and the
  bound-table write. The two field message values return to host Fiat-Shamir after
  each command. One readback hands a short dense tail to the optimized CPU.

There are no per-round device allocations. The proof-scoped row buffer is prepared
during stage 5, parked in `ProofSession`, and consumed by stage 6b. The stage-5 and
Booleanity row structs are both five `u64` words; a compile-time size/alignment check
allows one bulk Metal-buffer copy instead of reconstructing 67 million rows in a
Rust loop. Unsupported shapes and traces below the configured cutoff retain the CPU
kernel.

For materialization width `W`, the evaluator's optimistic non-cache traffic and
useful field-multiplication counts are

```text
L                 = log2(W) + 1
B_device          = 40 T L + 64 P T / W - 48 P
B_cache_logical   = 16 P T L
M_base            = P(2T + T/W - 3) + 2T - 2 + 2PK(W - 1)
M_metal           = M_base - PT + P((K+1) + (K+1)^2).
```

`B_device` counts the repeated packed-row scans and geometric dense read/write
traffic. `B_cache_logical` is reported separately because the branch tables occupy
only `P * W * K * 16` bytes and are intended to remain cache-resident. Actual peak
resident storage is one row buffer plus dense buffers of `P*T/W` and `P*T/(2W)`
field elements, not the full-sequence traffic sum.

At `T = 2^26`, `P = 29`, `K = 256`, and retained `W = 8`, the original schedule did
4.270 billion useful Metal multiplications. The retained initial-pair tables remove
`PT` initial-round multiplications and add only 1.923 million setup
multiplications, leaving 2.326 billion. The tables occupy 30.77 MB. The measured
16.4-Gmul/s arithmetic roof gives a 142 ms floor; 26.31 GB of optimistic non-cache
traffic gives a 58 ms floor at the 420-GiB/s copy roof. Logical cache-table loads
remain 124.55 GB and peak row-plus-dense storage is about 7.94 GiB.

The target-size materialization sweep used exact messages, host challenges, final
tables, and transcript state in every run:

| Width | Hybrid cycle wall | Modeled non-cache traffic | Peak row+dense storage |
|---:|---:|---:|---:|
| 2 | 868.9 ms | 67.65 GB | 24.25 GiB |
| 4 | 681.8 ms | 39.19 GB | 13.37 GiB |
| 8 | 640.4 ms | 26.31 GB | 7.94 GiB |
| 16 | 659.3 ms | 21.21 GB | 5.22 GiB |
| 32 | 700.6 ms | 20.00 GB | 3.86 GiB |

Width 8 is retained: later widths save dense traffic but add another full lazy row
and cache-table scan. Lazy threadgroup widths 64, 128, and 256 were within about 1%
at `2^26`; 512 regressed, so the stable 256-thread setting remains. Dense rounds use
128 threads and the CPU cutoff is `2^10` elements.

Before initial-pair specialization, the exact local evaluator measured 3.095 s
optimized CPU versus 640.4 ms hybrid at `2^26`, or 4.83x cycle-only. The bulk row
handoff took 125.6 ms, so charging it entirely to this slot gave 4.04x. At `2^22`,
width 8 measured 202.5 ms CPU, 40.0 ms hybrid, and 8.5 ms preparation: 5.06x
cycle-only and 4.17x all-in. The first real PIOP profile measured a 4.85x Booleanity
seam because preparation is performed and attributed in stage 5.

Seven shader schedules were rejected against the earlier fixed evaluator: pair
unrolling, selected-word shuffles, raw wide limbs, pair-major first-round work,
half-limb SIMD reduction, specialized Comba squaring, and selector-grouped SIMD.
None cleared the 6.44% noise-qualified promotion threshold. A smaller follow-up that
precomputed only `H(H-rho)` reduced the first round by 6.8% but improved the complete
cycle by only 2.63%, below the fixed 3% bar, and was rejected. The retained successor
precomputes both that constant and `(H_1-H_0)^2` for the `K+1` possible endpoints;
the extra endpoint represents a cold bytecode/RAM row. It changes no protocol state
and later rounds use the original shader path.

Against a matched seven-repeat baseline, endpoint-pair tables reduced median hybrid
wall time from 637.8 to 582.9 ms (8.62%) and the initial round from 286.7 to 228.5 ms
(20.3%). Their timed refresh cost was 5.6 ms. The contemporaneous CPU median was
3.344 s, giving 5.74x cycle-only and about 4.67x with the 133.8-ms one-time prepare
fully charged. Exactness and no-round-allocation guards passed. The new 142-ms
compute floor still leaves material headroom, so the uncapped controller does not
treat this promotion as the end of Booleanity analysis.

The post-promotion CPU-first PIOP pair in
`benchmark-runs/metal-piop-eval/20260804-111730` measured 21.071 s optimized CPU and
15.116 s Metal, or 1.394x. The real Booleanity seam was 4.007 s versus 714.6 ms
(5.61x), 106.5 ms below the first retained Metal seam. Both proofs verified. The
aggregate remains a one-pair checkpoint; the next port is selected from the Metal
profile, where Booleanity is now 4.73% of PIOP rather than the leading bottleneck.

## Spartan outer uni-skip ceiling

The post-Booleanity Metal profile makes `SpartanOuterUniskip` the largest remaining
seam: 2.626 s, or 17.37% of the 15.116-s Metal PIOP. The optimized CPU kernel already
uses the right protocol reduction. For each cycle and each of the two stream values,
it extends ten consecutive constraint-row evaluations to the nine nodes outside the
centered domain, multiplies the extended `Az` and `Bz`, and accumulates against the
factored equality polynomial. The device therefore returns only nine field values;
interpolation, the degree-27 round polynomial, transcript absorption, and the
uni-skip challenge remain on the host.

The first Metal schedule maps three rows onto one 32-lane SIMD group. Each row owns
nine lanes, one per extended node, so 27 lanes do useful work. A lane computes both
stream products. One 256-thread threadgroup owns one `E_out` block and walks its
`E_in` rows in strides of 24. It reduces nine values in 3.4 KiB of dynamic
threadgroup memory, multiplies by the block's single `E_out` value, and writes nine
block sums. A second dispatch reduces those block sums. This retains the CPU's
`E_out ⊗ E_in` factorization and avoids a field multiplication by `E_out` per row.

The coefficient path is expanded before shader compilation rather than constructing
19 signed row values. For example, the first-stream `Bz` coefficient of
`RamReadValue` is `c1 + c2`, and the second-stream coefficient of
`RightLookupOperand` is `c1 + c2 + c3 + c4`. This leaves roughly 522 widening
32-bit multiply-adds for the two signed dot products across all nine nodes, 18
small-by-wide integer products, and exactly 18 full field multiplications per cycle.
The largest Lagrange coefficient is 140,140; the existing CPU bounds
`|Az| < 2^22`, `|Bz| < 2^152`, and `|Az Bz| < 2^174` make a seven-limb product plus
the reusable Solinas reducer exact.

At `T = 2^26`, the useful count is 1,207,959,552 field multiplications. A standalone
160-byte packed-row ABI plus two 16-byte `E_in` weights moves 12.0 logical GiB, a
28.5-ms floor at the measured 420.68-GiB/s copy roof. The intended direct handoff is
64 bytes of canonical trace state plus a reusable 64-byte lookup-value sidecar, or
10.0 logical GiB and a 23.8-ms floor. Output traffic is about 1.2 MiB before the
final nine-value reduction.

The same binary measures 16.42 Gfield-mul/s on the retained compute-dense path and
958 G independent `u32` multiply-adds/s. Charging only one third of each measured
rate gives 220.7 ms for the field products and about 147.6 ms for the expanded
integer path; adding the standalone traffic floor gives a 397-ms conservative
budget. Against the measured 2.626-s seam this is 6.62x, above the hot-kernel
shader-entry bar of `1.25 · 5x = 6.25x`. The working target is at most 375 ms (7x),
not 5x. A 5x measured port alone projects the current PIOP from 1.394x to 1.619x;
6.25x projects 1.632x, before any residency reused by `OuterRemainder`.

The frozen evaluator must time three boundaries separately: GPU-active arithmetic
over resident packed rows, a complete standalone invocation including row-buffer
preparation, and the real `UniskipKernel` seam including direct witness handoff. Only
the last promotes the backend. The first two identify whether a miss comes from the
shader or the row boundary. Exact guards compare all nine extended-node values, the
assembled round polynomial, the host challenge, and the parked carry consumed by
`OuterRemainder`.

The target-scale dispatch search is recorded in
`benchmark-runs/metal-autoresearch/spartan-outer-uniskip-v3`. Its fixed dispatch-time
baseline was 348.860 ms with 0.83% relative MAD. The first retained candidate defers
canonical field additions in a 192-bit lane accumulator and performs one Solinas
fold after the row loop; its 320.499-ms median is 8.13% faster. The second retained
candidate evaluates six nodes over five rows and then three nodes over ten rows,
using 30 lanes in both phases without keeping two wide accumulators live. Its
307.757-ms median is another 3.98% gain and 11.78% faster than the baseline. All six
source trials passed exact extended-value, round-polynomial, challenge, output-claim,
host-transcript, and allocation guards.

The retained two-phase schedule rereads resident inputs. With the standalone packed
ABI, each phase reads 160 row bytes and 32 equality-weight bytes, or 24 logical GiB
at `2^26`; its 57.1-ms optimistic copy floor remains well below dispatch wall time.
The intended 128-byte trace-plus-sidecar handoff would read 20 logical GiB across the
two phases, a 47.5-ms floor. The v3 JSON `direct_handoff_logical_bytes` field records
the one-pass ABI minimum, not this two-phase shader traffic. A three-phase
full-occupancy trial improved only 2.06% and was rejected, showing that the extra row
scan consumed most of its utilization gain.

The retained synthetic packed-row evaluator reports a 10.56x complete resident
hybrid speedup in its middle fresh process and 4.05x when a new 10-GiB packed-row
copy is charged. Against the optimized CPU PIOP seam, 307.757 ms corresponds to
8.53x before handoff cost. The shader therefore clears the 7x working target; the
promotion question has moved to the real `UniskipKernel` path and whether witness
residency avoids repacking 160 bytes per row inside the PIOP boundary.

The first real backend profile is
`benchmark-runs/metal-piop-eval/20260804-133730`. Exact proofs verified at
`2^26`; optimized CPU PIOP was 21.134 s and Metal-hybrid PIOP was 13.572 s, a
1.557x paired speedup. `SpartanOuterUniskip` fell from 2.485 s on CPU to
667.445 ms on Metal (3.72x complete). Its child spans attribute 310.897 ms to
extracting and packing rows directly into the shared Metal buffer and 329.303
ms to dispatch. Removing only that redundant repack projects 6.98x locally,
so the next retained design must make the packed rows a reusable witness
representation while continuing to charge the actual buffer attachment inside
the PIOP boundary.

## Requirement map and open points

| Requirement | Planned mechanism | Acceptance evidence |
|---|---|---|
| Exact protocol output | Existing kernel traits; host transcript | optimized/Metal round and proof byte parity |
| No round allocation | proof-scoped ping-pong and scratch buffers | allocation counter plus code review |
| Honest portfolio speedup | PIOP-span wall time | interleaved optimized/Metal runs at `2^26` |
| Honest local speedup | complete hybrid wall time | order-inverted per-kernel runs |
| Useful work per read | fused bind, evaluate, reduce | worksheet counts and roof utilization |
| Safe small-size behavior | measured complete-tail cutoff | cutoff-neighbor validation |
| Recoverable tuning | immutable contract and JSONL lineage | controller recovery audit |

Open points that can change the implementation:

1. Register allocation, achieved occupancy, and the limiting execution resource need
   Xcode Instruments. Runtime pipeline limits alone cannot settle them.
