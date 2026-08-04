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
| 5 | `instruction_read_raf` | CPU address, five-factor Metal cycle | integrated; exact e2e |
| 5 | `ram_ra_claim_reduction` | dense fused product | analyze |
| 5 | `registers_val_evaluation` | dense fused product | analyze |
| 6a | `bytecode_read_raf_address` | address pushforward | analyze |
| 6a | `booleanity_address` | address pushforward | analyze |
| 6b | `bytecode_read_raf_cycle` | sparse-to-dense cycle reduction | analyze |
| 6b | `booleanity_cycle` | sparse-to-dense cycle reduction | worksheet complete; evaluator next |
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

### Measured port order

An optimized Akita Fibonacci proof at `2^20` took 1.411 s. Inclusive round time from
the retained Perfetto trace ranks the unported cycle kernels as follows:

| Kernel | Inclusive round time |
|---|---:|
| `Booleanity` | 132.28 ms |
| `InstructionRaVirtualization` | 83.12 ms |
| `BytecodeReadRafCycle` | 41.33 ms |
| `RegistersReadWriteChecking` | 20.41 ms |
| `InstructionInput` | 20.57 ms |
| `RamRaVirtualization` | 18.96 ms |

`InstructionReadRaf` itself used 110.84 ms across 128 address and 20 cycle rounds.
The current Metal backend changed the full `2^20` proof from 1.411 s to 1.384 s in
one matched process pair, but stage 5 remained about 130 ms: this size is close to the
cycle cutoff and the CPU address prefix dominates. This is directionally useful, not
a promoted end-to-end speedup result.

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
