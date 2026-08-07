# Spartan dense Metal family v3

Status: canonical implementation packet, 2026-08-07. This packet supersedes
the product/shift successor packets as an implementation order, but not as
evidence. It makes no protocol change and does not promote an unmeasured path.

## Decision

Preserve the current product arithmetic. `ProductRemainder` is already a
promoted kernel: at `log_T = 26`, the clean five-pair evidence reports
`439.132709 ms` optimized CPU, `30.341540 ms` Metal, and `14.3988x`. The
carried Outer+Product family is `1579.798297 / 266.668917 = 5.9242x`.
Reopening its shaders before counters show a new bottleneck would spend the
iteration budget on a solved member.

The immediate problem is ownership. The exact Phase-A shift runtime exists,
but its promotable input must be produced while the Stage-1 witness row is
already live. A second projection is useful only as an executable measurement
bridge. The promotion architecture is one proof-scoped resident row owner
that emits typed leases to Stage 1, product, instruction claim, and shift.

The ordered objective is therefore:

1. measure the staged shift adapter through a checked second-projection bridge;
2. replace the bridge and the current product collection with one Stage-1
   producer traversal and a receipt-bearing resident owner;
3. retain the current carried product uni-skip and 4,096-element CPU tail;
4. add upstream shift partial carriers only after the producer-inclusive
   Phase-A member clears `26.210324 ms`;
5. continue to the `16.381453 ms` Phase-B target while its measured component
   rates still make 8x credible.

The first promotable slice is item 2, not the measurement bridge.

## Fixed protocol and transcript boundaries

### Product uni-skip

Let `N = 2^n`. For cycle `j`, define

```text
u(j) = [left_instruction_input(j), lookup_output(j), jump_flag(j)]
v(j) = [right_instruction_input(j), branch_flag(j), 1-next_is_noop(j)].
```

For the centered domain `{-1, 0, 1}`, let `L_i(y)` be its Lagrange basis and

```text
u_y(j) = sum_i L_i(y) u_i(j)
v_y(j) = sum_i L_i(y) v_i(j)
t1(y)  = sum_j eq(tau_low, j) u_y(j) v_y(j).
```

Stage 1 already supplies `t1(-1) = product`, `t1(0) = should_branch`, and
`t1(1) = should_jump`. The device work is only `t1(-2)` and `t1(2)`. The host
draws `tau_high`, inserts the three known values, interpolates the five-node
degree-four `t1`, multiplies it by the degree-two centered kernel
`K_3(tau_high, y)`, absorbs the degree-six first-round polynomial, and draws
`r0`. The device never hashes and never sees transcript state.

The production route already obtains the two missing endpoints from the
Stage-1 outer opening command. It must continue to have zero standalone
product-uni-skip dispatches and zero standalone readbacks. The endpoint
receipt binds `tau_low`, row count, source allocation, device, and producer
generation; it does not pretend that an endpoint computed from outer rows was
computed from the product-row allocation.

### Product remainder

After `r0`, set

```text
w_i      = L_i(r0)
scale    = K_3(tau_high, r0)
left(j)  = w_0*u_0(j) + w_1*u_1(j) + w_2*u_2(j)
right(j) = w_0*v_0(j) + w_1*v_1(j) + w_2*v_2(j)
f(j)     = scale * eq(tau_low, j) * left(j) * right(j).
```

This is the existing degree-three, `n`-round, low-to-high sumcheck. A Metal
message is exactly `[q(0), q(infinity)]`; the host reconstructs the cubic with
`GruenSplitEqPolynomial`, combines it with the other Stage-2 members, checks
the round sum, absorbs the one batched polynomial, and supplies the shared
challenge. Consecutive product rounds cannot fuse across this boundary.

After the terminal challenge, the common product opening point is
`reverse([c_0, ..., c_(n-1)])`. Outputs remain in this exact order:

```text
left_instruction_input
right_instruction_input
jump_flag
write_lookup_output_to_rd
lookup_output
branch_flag
next_is_noop
virtual_instruction
```

In Stage 2, the host order remains:

1. derive `tau_low` from Stage 1 and prepare the two uni-skip endpoints;
2. draw `tau_high`;
3. prove the uni-skip and draw `r0`;
4. construct `ProductRemainder(r0, tau_high, tau_low)` and submit its
   transcript-free prefetch;
5. draw the Stage-2 batch challenges;
6. run `prove_batch`, with one host absorption per global round;
7. deliver the terminal bind, evaluate outputs, and park residue.

### Spartan shift

For current-cycle columns,

```text
outer(j)   = upc(j) + gamma*pc(j)
           + gamma^2*virtual(j) + gamma^3*first(j)
product(j) = gamma^4*(1-noop(j)).
```

The exact relation is

```text
EqPlusOne(r_outer, j)*outer(j)
  + EqPlusOne(r_product, j)*product(j),
```

where `r_outer = tau_low` and `r_product` is the reversed product-remainder
opening point. Shift also binds low-to-high and opens its five current columns
at the reverse of its own 26 challenges. The exact current
`InstructionFlags::IsNoop` bit is required; deriving it from unrelated flags
is forbidden.

Stage 3 draws `gamma`, prepares members in generated order (`shift`,
`instruction_input`, `registers_claim_reduction`), and then absorbs one
batched polynomial per round. Phase B may advance transcript-free
InstructionInput state from the shift handle at the midpoint, but the later
InstructionInput call must consume the tagged cached message. It must not bind
twice or alter member aggregation or absorption order.

## One resident row owner

Use one logical `SpartanDenseResidentOwner` with multiple physical
allocations. A monolithic allocation would worsen alignment, maximum-buffer,
and lifetime constraints. The owner is proof-scoped and contains an allocation
registry, liveness state, and producer receipt. Typed leases clone Metal
resource handles but never claim a new allocation identity or account the
bytes twice.

At `log_T = 26`, the current physical source views are:

| source lease | layout | bytes |
|---|---:|---:|
| Stage-1 outer | `48N + 112N` | 10,737,418,240 |
| product | `40N` | 2,684,354,560 |
| instruction companion, when admitted | `24N` | 1,610,612,736 |
| shift UPC/PC/three masks | `16N + 12(N/32)` | 1,098,907,648 |
| maximum row-source set | | 16,131,293,184 |

The source owner is separate from a `SpartanDenseWorkspaceArena`. Product
state A/B and reduction scratch must not be allocated or first-touched merely
because Stage 1 starts. They are acquired after outer dense storage is dead,
or are reused from that released storage. The row owner outlives those
workspaces.

The ownership sequence is:

```text
backend_witness_prepare
  -> one row traversal creates outer/product/optional-instruction/shift views
Stage 1
  -> outer lease; parks product endpoints and, in Phase B, outer partials
  -> releases outer-only residual/workspace at output
Stage 2
  -> product lease; materialize, dense ladder, one tail handoff, openings
  -> parks three instruction aliases and, in Phase B, nonnoop partials
  -> releases product state; product row lives only while another consumer needs it
Stage 3
  -> shift lease plus upstream point receipts
  -> Phase A releases all three shift buffers after midpoint fold
  -> Phase B releases UPC after outer carrier production and PC/masks after residual fold
```

The owner should be an `Rc`-backed proof-local object because `ProofSession`
and the current cross-member services are proof-local. Allocation accounting
lives only on the owner; leases report metadata, not resident bytes.

### Producer receipts

`SpartanDenseProducerReceipt` contains, at minimum:

```text
proof/witness generation
N and log_T
Metal device registry id
for every allocation: semantic kind, allocation identity, exact byte length
exact-current-noop certificate
producer completion/event generation
row extraction count and each destination write count
late upload/copy dispatch count
```

A point carrier adds the relation id, producer stage, big-endian point or its
canonical digest, `P/H`, canonical field encoding, and source allocation ids.
A round receipt adds service generation, local/global round, current element
count, source/destination ids, and the exact canonical bytes of the pending
bind. The InstructionInput midpoint receipt additionally binds the ordered
first-13-challenge digest and the exact `H`-element UPC allocation.

Missing, duplicate, stale, foreign-device, wrong-length, wrong-point, or
wrong-challenge receipts fail closed. Before a command is submitted, a
capability/admission miss may select optimized CPU. After successful submit,
the affected service is authoritative; a command or receipt failure is
terminal. After any affected round polynomial is absorbed, fallback is never
legal.

## Concrete pass schedule

### Current product service: retain

1. **Outer opening carrier.** Under the existing outer-opening command,
   compute the two uni-skip endpoint scalars. No new command or wait.
2. **Product materialize/message.** After `r0`,
   `solinas_product_remainder_materialize_message` reads the 40-byte row once,
   writes L/R state, and reduces the first message. Submit from
   `prefetch_relation`; join only when product round zero needs it.
3. **Product transitions.** Each
   `solinas_product_remainder_bind_message` binds both factor planes and
   reduces the next message in one command. Never split bind from message.
4. **CPU tail.** At 4,096 source elements, read L/R exactly once (131,072
   bytes) and finish the late rounds on the host.
5. **Product openings.** Scan the resident 40-byte row once and reduce all
   eight canonical outputs. Split into two four-column scans only if generated
   code proves that avoiding spills wins after paying the second 2.5-GiB read.

The default threadgroup widths remain 64 for the endpoint control, 128 for
materialize, 64 for transitions, and 128 for openings. Dynamic threadgroup
storage is tiny (at most 512 bytes for eight opening columns at width 128), so
register allocation and command service, not threadgroup memory, are the
occupancy risks. Every full-size pass has enough independent groups to cover
40 GPU cores; the 4,096 cutoff prevents the smallest device rounds from
becoming a launch-latency tail.

### Phase-A shift: first integrated service

1. `solinas_spartan_shift_build_mixed_partials` and
   `solinas_spartan_shift_reduce_prefix` execute in one prefix command.
2. The host proves 13 prefix rounds from four `P = 8192` tables.
3. After `c_12`, `solinas_spartan_shift_fold_native` folds the five native
   columns into `H = 8192` dense tables.
4. The host proves the 13 suffix rounds and extracts the five outputs.

This is two full native scans and two command completions. The currently
selected launch is mixed `(64 threads, high tile 128)` and fold width 32.
Keep scratch reusable and pre-sized. There is no warm-up dispatch in a target
sample.

### Phase-B shift: upstream carriers

Let `P = H = 8192`, `j = hP+l`, and `e[h] = eq(r_hi,h)`. For every native
column `v`, retain

```text
C_v[l] = sum_h e[h] v(h,l)
S_v[l] = sum_(h=1)^(H-1) e[h-1] v(h,l).
```

`S` has no wraparound. Stage 1 emits current/successor UPC, PC, virtual, and
first tables. Stage 2 emits current/successor nonnoop. After `gamma`, a small
combine forms

```text
q0 = C_upc + gamma*C_pc + gamma^2*C_virtual + gamma^3*C_first
q1 = S_upc + gamma*S_pc + gamma^2*S_virtual + gamma^3*S_first
q2 = gamma^4*C_nonnoop
q3 = gamma^4*S_nonnoop.
```

The host proves the prefix. At the midpoint, borrow InstructionInput's exact
partially bound UPC table and run only
`solinas_spartan_shift_successor_fold_residual` for PC, virtual, first, and
noop. The outer and product carriers encode under their existing output
command buffers; only the midpoint adds a Stage-3 wait.

Start the outer numeric carrier at width 128. At log 26 it has 64 threadgroups
and 256 SIMD groups, enough to distribute across 40 cores. Its structural live
set is roughly four field accumulators plus native/weight temporaries; capture
must show no local-memory spill and at least two resident SIMD groups per core.
Metal does not expose a stable source-level register-file formula, so a more
precise occupancy percentage without compiler output would be fabricated.

### Legal and illegal fusion

Legal:

- one producer traversal for all native views;
- product endpoints in the existing outer-opening command;
- product nonnoop partials in the existing product-opening command;
- outer shift partials in the existing outer-opening command;
- product plus instruction materialize/transition/opening encoders in one
  command buffer, after the shared service has both members' challenges;
- one L/R/C tail handoff for a later paired product/instruction service.

Illegal without a protocol change:

- product uni-skip endpoint scan with remainder materialization (`r0` is not
  known yet);
- product materialization with final openings (the terminal point is not
  known yet);
- any two dense rounds (the second bind is a Fiat-Shamir result of the first
  batched polynomial);
- moving Fiat-Shamir or batch polynomial absorption to Metal;
- inferring current noop from `next_is_noop` without the exact boundary-cell
  proof and current cycle-zero value.

## Log-26 work, traffic, and budgets

### Product with the production 4,096 tail

Here `N = 67,108,864`. The GPU transition source sum is exactly
`2N-2C = 134,209,536`, the split-equality `e_out` sum is 16,256, and the tail
readback is 131,072 bytes.

| phase | logical bytes | useful full products | products/byte |
|---|---:|---:|---:|
| carried uni-skip endpoints | no incremental source scan | 268,451,840 | n/a |
| materialize + first message | 4,831,838,208 | 335,560,704 | 0.069448 |
| 14 GPU transitions | 6,442,057,728 | 268,451,584 | 0.041672 |
| eight openings | 2,684,354,560 | 201,392,128 | 0.075024 |
| remainder core plus tail | 13,958,381,568 | 805,404,416 | 0.057700 |
| core plus fused endpoint arithmetic | same local bytes | 1,073,856,256 | 0.076933 |

The endpoint row load belongs to outer opening and must not be charged as a
fictional standalone product scan. Conversely, a fallback that dispatches the
endpoint shader standalone must charge the full row scan and wait.

Observed complete-member bars replace the older analytical control rates:

| product boundary | wall |
|---|---:|
| optimized CPU median | 439.132709 ms |
| hard 5x cap | 87.826542 ms |
| 8x cap | 54.891589 ms |
| 10x non-regression target | 43.913271 ms |
| current Metal median | 30.341540 ms |

The implementation must preserve at least 10x at log 26; 5x is only the
project kill bar. Do not replace observed current performance with the stale
2^16-tail model.

### Shift Phase A

The selected mixed build performs 134,209,536 useful full products and
67,624,960 half-width terms. The midpoint fold performs 134,217,728
half-width terms. Cache-coalesced logical traffic for both commands is
2,274,948,096 bytes, in addition to the 1,098,907,648-byte producer output;
host-visible tables total 1,179,648 bytes.

The optimized CPU median is `131.051624 ms`; 5x and 8x caps are
`26.210324 ms` and `16.381453 ms`. The prepared-service control is
`22.636833 ms` (`5.789x`), but it excludes production and had one
`45.228959-ms` outlier. The host-written first-use median was `79.592083 ms`.
Therefore:

- Slice 0 is successful as a bridge if it produces exact complete-member
  evidence, even if it is slow; it is not promotable evidence.
- the co-produced Phase-A promotion target is 22--26 ms, a realistic
  `5.0--6.0x`, with five alternating pairs and no hidden warm-up;
- a median above 26.210324 ms stops Phase B until ownership/service, rather
  than shader arithmetic, explains the miss.

### Shift Phase B

| phase | useful work | logical bytes |
|---|---:|---:|
| outer eight-table carrier | 268,419,072 half-width terms | 1,091,698,688 |
| product two-table carrier | at most `2N-P` selected adds | 8,781,824 |
| ten-to-four combine | 65,536 full products | 1,835,008 |
| prefix host ladder | 131,048 full products | host |
| residual midpoint | 67,108,864 half-width terms | 562,692,096 |
| suffix host ladder | 155,629 full products | host |
| total | 335,527,936 half-width + 352,213 full | 1,665,007,616 kernel bytes |

Arithmetic intensity is 0.2015 half-width terms per kernel byte. Including
the 1,098,907,648 producer writes gives 2,763,915,264 logical bytes and 0.1214
terms/byte. The retained matched-rate model gives 15.145 ms complete
(`8.65x`); the conservative promotion-rate model gives 18.464 ms (`7.10x`).
These are targets, not evidence. Use phase bars of 10.2 ms outer-carrier
active, 0.5 ms product-carrier incremental wall when fused, 2.6 ms midpoint
active, and 2.5 ms total host/service. Continue toward 16.381453 ms while
captures still admit it; the hard kill remains 26.210324 ms.

Adding shift to the observed Outer+Product family gives an illustrative dense
family bound of about `1710.850 ms` CPU. At the 5x shift cap the corresponding
Metal sum is about `292.879 ms` (`5.84x`); at the 8x cap it is about
`283.050 ms` (`6.04x`). These are sums of member medians, not an end-to-end
PIOP prediction. Phase A saves roughly 105 ms and Phase B roughly 115 ms
against CPU shift; the PIOP improvement must be measured because other batch
members and overlap determine how much reaches the critical path.

## Hybrid policy

- Product keeps `trace_cutoff_elements = 2^18` and `cpu_tail_elements = 2^12`
  until a frozen multi-scale sweep disproves them. Metal admission also
  requires the endpoint and row-owner receipts; otherwise select CPU before
  uni-skip transcript work.
- Shift uses CPU below `2^25`. At `2^25` and above, Phase A is eligible only
  after its producer-inclusive five-pair gate. If log 25 does not clear 5x,
  raise the selector to `2^26`; do not extrapolate from the existing single
  70.9-ms CPU observation.
- Phase B inherits the validated Phase-A crossover. A carrier miss selects
  Phase A or CPU before round zero; it never reconstructs carriers through a
  fresh full-row scan.

Every selector report includes complete member wall, PIOP wall, and
PIOP-plus-`backend_witness_prepare`. Excluding preparation from the fair PIOP
comparison does not authorize hiding duplicate witness walks, uploads, peak
residency, or first-consumption service.

## Source-level implementation map

### Slice 0: measurement bridge, not promotion architecture

- `crates/jolt-prover/src/akita/prover.rs::prove` already calls
  `spartan_outer_uniskip.prepare_witness` under
  `jolt_prover::backend_witness_prepare`. Keep the checked second projection
  there, outside the PIOP span, and park `SpartanShiftResidentRows` before
  Stage 1.
- `crates/jolt-kernels/src/metal/spartan_shift.rs` consumes that parked owner
  through the staged prefix/fold adapter. Record projection, first-consumer,
  command, host ladder, and output walls separately.
- `crates/jolt-kernels/src/metal/solinas/spartan_shift/runtime.rs` remains the
  checked attachment/submission layer. The bridge must use real allocation
  identities and `exact_current_flags = true`.

This bridge deliberately performs a second witness projection. Its result can
validate parity and lifecycle instrumentation, but it cannot satisfy the
one-pass receipt or zero-copy promotion condition.

### Slice 1: canonical co-producer and owner

- In `crates/jolt-witness/src/witnesses/spartan.rs`, extend
  `SpartanOuterRow` with exact current `InstructionFlags::IsNoop` and populate
  it in `WitnessBundle::from_row`. This is witness storage, not a proof or
  verifier change.
- Move the Metal-specific multi-destination producer out of
  `crates/jolt-kernels/src/optimized/spartan_outer.rs` into a focused
  `crates/jolt-kernels/src/metal/spartan_dense.rs`. The current seam to replace
  is `prepare_metal_spartan_outer_rows`.
- Extend
  `SolinasMetal::prepare_spartan_outer_uniskip_rows_with_fill` in
  `metal/solinas/spartan_outer_uniskip/mod.rs`, or add a sibling checked
  allocator, so one 32-row chunk writes:
  outer compact/residual rows, product rows, optional instruction companion,
  UPC/PC, and one complete `SpartanShiftFlagWord`. Chunk ownership avoids
  atomics and a later flag-pack pass.
- In `crates/jolt-kernels/src/metal/spartan_outer.rs`, replace the separate
  calls from `UniskipKernel::prepare_witness` with one
  `prepare_spartan_dense_witness` call and park the logical owner.
- Split
  `crates/jolt-kernels/src/metal/spartan_product.rs::prepare_product_remainder_witness`
  into source production and `prepare_product_remainder_storage_from_lease`.
  Remove its `collect_bundles` traversal and host `Vec` repack once Slice 1 is
  authoritative.
- Change `metal/spartan_shift.rs` from owning a one-off
  `PreparedSpartanShiftRows` to taking a typed lease from the owner. Preserve
  the pre-submit CPU fallback and post-submit terminal behavior.
- Add owner exports and config plumbing in `metal/mod.rs` and
  `metal/backend.rs`; this adds no new sumcheck slot.

### Slice 2: product lifetime and arena

- Keep `MetalProductUniskipEndpointCarrier` production in
  `metal/spartan_outer.rs::MetalOuterRemainderKernel::output_claims` and its
  outer-remainder opening shader. Strengthen its receipt; do not add a
  standalone path.
- Override the outer kernel's `park_residue` or add an equivalent release
  handoff so outer-only dense storage enters `SpartanDenseWorkspaceArena`
  before Stage 2. Product state allocation must occur after that release.
- In `metal/spartan_product.rs`, make `ProductRemainderSequence` consume the
  product lease and arena workspace. Preserve relation-dependent prefetch,
  the 4,096 tail, one tail readback, and eight-opening alias publication.
- Keep `metal/solinas/product_remainder/{mod.rs,shader.metal}` unchanged until
  the lifecycle version reproduces current product evidence.

### Slice 3: Phase-B product and outer carriers

- Add encoder-level carrier functions under
  `metal/solinas/spartan_shift_successor`; register the shader fragment in
  `metal/solinas/source.rs` only when the runtime calls it.
- In `metal/solinas/outer_remainder/{shader.metal,sequence.rs}`, produce the
  eight outer partial tables under the existing opening command. The endpoint
  implementation must replace compatible opening work or account its full
  delta; it may not rescan the 160-byte physical row layout.
- In `metal/spartan_product.rs::MetalProductRemainderKernel::output_claims`,
  encode the two nonnoop tables under the existing product-opening command.
  Read the compact current-noop masks from the owner rather than scanning the
  40-byte product rows again.
- Park typed outer/product carrier leases in `ProofSession`; validate them in
  `metal/spartan_shift.rs` before round zero.

### Slice 4: midpoint UPC service

- In `crates/jolt-kernels/src/metal/instruction_input.rs`, add a proof-local
  shared service. Shift preparation creates it and InstructionInput
  preparation registers its state before `prove_batch` starts.
- On shift round 13, advance InstructionInput with the exact pending `c_12`,
  publish its `H`-element UPC table, and cache only the raw InstructionInput
  message. Its later `prove_round` reconstructs its own polynomial using its
  own `previous_claim`; it does not bind again.
- In `metal/spartan_shift.rs`, consume the midpoint receipt and run the
  four-column residual fold. Keep `crates/jolt-sumcheck/src/prover.rs::prove_batch`,
  `crates/jolt-prover/src/stages/stage2.rs`, and `stage3.rs` unchanged. They
  are transcript authorities, not GPU scheduler extension points.

### Slice 5: optional product/instruction scheduler

Only after the above family is stable, reuse the existing
`Stage2ProductInstructionRow` producer relationship to encode product and
instruction materializers, transitions, and openings into shared command
buffers. Use one tagged service and one L/R/C tail handoff. Do not introduce a
joint high-register shader until the existing two encoders demonstrate that
completion/wait fusion is the remaining bottleneck.

## Exact oracles and promotion evidence

Independent authority is:

- `crates/jolt-kernels/src/reference/spartan_product.rs` for direct dense
  product uni-skip/remainder;
- `crates/jolt-kernels/src/reference/spartan_shift.rs` for seven-table direct
  dense shift;
- `crates/jolt-kernels/src/optimized/spartan_product.rs` and
  `optimized/spartan_shift.rs` for byte-parity against the production CPU
  algorithms;
- the symbolic relations in
  `crates/jolt-claims/src/protocols/jolt/relations/spartan/{product_uniskip,product_remainder,shift}.rs`
  and their verifier-derived publics.

Tests must compare, in order:

1. direct `t1(-2/+2)` and the full degree-six uni-skip polynomial;
2. direct L/R materialization and every `[q(0),q(infinity)]` message;
3. all eight product outputs at the reversed point;
4. direct dense `EqPlusOne` (`j=0 -> 0`, `j>0 -> eq(r,j-1)`) against all four
   prefix Q tables and all five midpoint tables;
5. every shift round polynomial and all five outputs;
6. clear Akita/Metal transcript events, challenge order, proof bytes, and
   verifier acceptance, plus the CPU backend's `host,zk` regressions (the
   workspace intentionally forbids `akita` and `zk` together);
7. owner/receipt rejection for stale generations, wrong points, duplicate or
   foreign allocations, wrong lengths, missing exact-noop certification, and
   replayed binds.

Vectors include odd and even logs, row zero and final row, `i128::MIN/MAX`,
`u64::MAX`, all-zero/all-one/mixed flags, cycle-zero noop, and challenges
`0`, `1`, `p-1`, plus randomized cases. A carrier oracle computes current and
successor tables directly from dense rows and checks the no-wrap final cell;
it does not call the carrier implementation.

Performance promotion requires five fresh alternating CPU/Metal pairs at log
26, one holdout scale, exact proofs, stable source/binary, first-consumer wall,
GPU-active and exposed join time per command, uploads/copies/allocations/waits,
producer before/after wall, peak resident bytes, achieved useful rate and
bytes/s, compiler register allocation, local-memory spills, resident SIMD
groups, and active cores. Product must reproduce at least 10x; Phase A shift
must clear 5x; Phase B targets 8x while its component evidence leaves that
route physically credible.

## Highest-leverage first implementation

After Slice 0 supplies a runnable measurement, implement Slice 1 as the first
real change: extend `SpartanOuterRow` with current noop, emit the shift planes
and existing product/lookup rows from the Stage-1 producer traversal, park one
receipt-bearing owner, and make product and shift consume leases. It removes
the discovery and repeated-extraction bottleneck without touching the already
fast product shaders, and it turns the exact staged shift runtime into a fair
first-consumer experiment.

The realistic local outcome is unchanged product at roughly 30--35 ms and
producer-resident Phase-A shift at 22--26 ms (`5.0--6.0x`). Phase B has a
credible 15--18.5-ms design range (`7.1--8.6x`), but that range is conditional
on upstream carrier delta, no spills, and the midpoint alias. No end-to-end
PIOP ratio should be quoted until those critical-path walls are measured.
