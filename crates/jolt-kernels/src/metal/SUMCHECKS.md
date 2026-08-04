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

The first slot is `InstructionReadRaf`. Its sparse 128-bit address phases and first
cycle message remain on the CPU. When the first cycle challenge arrives, the CPU
computes the five half-domain tables directly into one shared Metal buffer; no
full-domain dense table or intermediate host `Vec` exists. Metal then uses the
resident five-factor fused bind-and-message schedule until the measured cutoff and
the optimized CPU finishes the short tail.

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
| 1 | `spartan_outer_uniskip` | centered-domain reduction | analyze |
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
| 5 | `instruction_read_raf` | resident address scans, five-factor Metal cycle | cycle integrated; address reopened |
| 5 | `ram_ra_claim_reduction` | dense fused product | analyze |
| 5 | `registers_val_evaluation` | dense fused product | analyze |
| 6a | `bytecode_read_raf_address` | address pushforward | analyze |
| 6a | `booleanity_address` | address pushforward | analyze |
| 6b | `bytecode_read_raf_cycle` | sparse-to-dense cycle reduction | analyze |
| 6b | `booleanity_cycle` | sparse-to-dense cycle reduction | worksheet complete; queued after stage 5 |
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
when the conservative aggregate projection clears the 5% continuation threshold.
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

Applying the working targets (5x for shares above 5%, 4x for 1-5%, 3x below 1%) to
the target profile gives an Amdahl projection of about 4.19x from the current path.
That calculation is not a forecast: it shows that the 4x portfolio floor requires
broad coverage and cannot be obtained from the cycle tail alone. Exact hybrid
measurements replace each local target as ports land.

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
order-independent and can consume that same layout. The only required inverse
permutation is the weight vector at the address/cycle handoff: about 36 bytes per
row once, amortized to 2.25 bytes per row over 16 address phases. The retained
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

### Booleanity cycle worksheet

The next slot is `Booleanity`, selected by the profile. Let `T` be the cycle count,
`P` the number of checked one-hot columns, `K = 2^w` the chunk domain, and `b < 4`
the number of bound cycle bits. A dense-from-start implementation moves roughly
`64PT` factor bytes over the full sequence and discards the CPU kernel's most useful
optimization. The retained design mirrors the mathematical sparsity instead:

- upload each packed 40-byte instruction/cycle row once and reuse it across the
  Booleanity, instruction-virtualization, bytecode, and RAM consumers;
- keep `P * 2^b * K` pre-scaled address tables resident for the first four rounds;
- in a lazy message, read each packed row once, derive every selector index from that
  row in registers, gather the small address tables, and accumulate the two quadratic
  lanes;
- fuse the fourth bind, `T/16` dense materialization, and following message, writing
  each dense field value once; then use resident dense bind-and-message transitions.

Each of the first four lazy messages reads `40T` row bytes, independent of `P` and
`b`; its small address tables are cache-sized. At `b = 0`, it performs about `PT`
useful quadratic field multiplications, for optimistic intensity `P/40` useful
multiplications per byte. At `b = 3`, the relation work is about `PT/8`, while the
row traffic stays `40T`. Materialization reads another `40T` row bytes and writes
`PT` bytes. The dense tail adds approximately `3PT` factor bytes, giving an optimistic
full-sequence floor of

```text
B_booleanity ~= T * (200 + 4P) bytes,
```

versus `64PT` bytes for eager dense tables. For a representative `P = 50`, this is
about an 8x reduction in factor/row traffic before cache and reduction scratch. The
falsifying benchmark is the complete hybrid sequence: if selector decoding and
irregular gathers erase that traffic advantage, the fallback handoff starts at the
CPU's `T/16` dense materialization instead.

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
