# Increment claim ownership for Metal

Decision: do not register a standalone increment claim-reduction kernel on the
Akita path. Akita already discharges the four increment claims inside the
bytecode read-RAF address and cycle phases. The Metal performance boundary is
therefore the fused read-RAF pair. A two-scan projection remains the preferred
base-protocol fallback after the field and protocol feature split; it stays
disabled until a matching base Solinas CPU denominator exists.

This packet is unregistered. It changes no transcript, relation, backend slot,
or shared shader source.

## Frozen owner denominator

The target artifact is
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, SHA-256
`587e00a65bde003a7c3481f58b1ea047ed2c908b0e3d9808bbc7eec6f894b2df`.
It is an alternating five-pair Fibonacci log-26 run on the 40-core M4 Max with
16 Rayon threads, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, clean worktree, binary SHA-256
`a8b5f918c4a86ebdd2e4be3da10511ea071df4ea4949a23e02e5b286397d0e8b`.

The optimized CPU samples for
`BytecodeReadRafAddressPhase + BytecodeReadRafCycle` are:

```text
1156.804791, 1254.933123, 1168.671791, 1206.041790, 1203.638208 ms
```

The frozen median is `1203.638208 ms`. The complete fused-owner caps are
`240.727641 ms` at 5x and `150.454776 ms` at 8x. The corresponding Metal-arm
samples were `350.824336, 363.178339, 349.465502, 348.842671, 341.891626 ms`,
a `349.465502 ms` median and `3.444x` screening ratio. The cycle phase was on
Metal; the address phase was still CPU work, so this is not a completed fused
owner.

There is no valid standalone log-26 denominator. `IncClaimReduction` is
compiled out under `akita`, and the current `metal` feature selects Akita.
The historical BN254 trace at revision `1a6c6ff58` recorded a 1.790499-second
log-27 prepare span, but it is the wrong field, scale, and boundary. It cannot
set a selector or a speedup claim.

## Algebra and host boundary

The base relation is

```text
A(j) = eq(u0,j) + gamma*eq(u1,j)
B(j) = gamma^2*eq(u2,j) + gamma^3*eq(u3,j)
S(j) = A(j)*RamInc(j) + B(j)*RdInc(j).
```

The four points are RAM read-write, RAM value-check, register read-write, and
register value-evaluation, in that order. The degree-two sumcheck binds cycle
variables low-to-high and emits `RamInc` then `RdInc` at
`reverse(sumcheck_challenges)`.

Akita replaces this member with four stages at powers `gamma^5..gamma^8` in
the bytecode read-RAF relation. A bytecode row's one-hot store selector proves
`FusedInc*Store = RamInc` and `FusedInc*(1-Store) = RdInc`; preprocessing
checks store/rd disjointness. The address phase consumes the four claims, and
the cycle phase returns one `FusedInc` opening. That relation is the selected
owner. Reintroducing `IncClaimReduction` on Akita would duplicate protocol
work.

Fiat-Shamir remains on the host in both designs. The device may use points and
challenges already drawn by the generated driver, but it never absorbs,
hashes, draws, reorders claims, or crosses the next challenge dependency.

## Base fallback: two scans, not dense rounds

Let `p=floor(log_T/2)`, `q=log_T-p`, `P=2^p`, `H=2^q`, and
`j=x_hi*P+x_lo`. Before the stage-6b gamma draw, compute four projections:

```text
Q0[x_lo] = sum_x_hi eq(u0_hi,x_hi)*RamInc[j]
Q1[x_lo] = sum_x_hi eq(u1_hi,x_hi)*RamInc[j]
Q2[x_lo] = sum_x_hi eq(u2_hi,x_hi)*RdInc[j]
Q3[x_lo] = sum_x_hi eq(u3_hi,x_hi)*RdInc[j].
```

The host proves the first `p` messages over `Q0..Q3` and the four low equality
tables. Once the prefix challenges exist, a second scan computes

```text
RamDense[x_hi] = sum_x_lo eq(reverse(r_prefix),x_lo)*RamInc[j]
RdDense[x_hi]  = sum_x_lo eq(reverse(r_prefix),x_lo)*RdInc[j].
```

The host finishes over those two arrays and the four high equality tables.
Before the first suffix message, it multiplies each of the four gamma powers
by the corresponding fully bound low equality leaf. These four scalar
products preserve the original factorization without scaling an `H`-element
table.
The second scan is necessary for arbitrary Fiat-Shamir prefix challenges; the
four fixed-point Q projections cannot recover it.

The preferred Q owner is the existing stage-6a resident-row command. On the
base path it may append Q partials to the Booleanity-address or bytecode-address
pass while each 16-byte increment row is already staged. The paired benchmark
charges the producer and any occupancy loss exactly once. A separate Q primer
during the stage-6a window is the fallback, not free work. At the midpoint,
the fold can join the bytecode cycle command for that global round; a private
command is allowed only if its wait is included in complete-member wall.

## Log-26 roof

For `T=2^26`, `P=H=8192`, eight Q partitions, and dense homogeneous selectors:

| quantity | exact value |
|---|---:|
| useful Q / fold signed products | 134,217,728 / 67,108,864 |
| issued Q / fold products | 134,217,728 / 67,108,864 |
| perfect Q / fold bytes | 1,082,654,720 / 1,074,003,968 |
| cache-unique total bytes | 2,157,051,904 |
| source-request total bytes | 3,297,509,376 |
| mapped output bytes | 786,432 |
| sequence-owned storage | 5,636,128 |

With retained controls of `451.701710520 GB/s` and `26.272 G signed
products/s`, arithmetic binds both perfect phases. The 80%-of-roof two-phase
gate is `9.579 ms`; the pessimistic request model raises it to `12.329 ms`.
If every SIMD block contains both selectors, useful work is unchanged but Q
issue doubles. The cache-unique and request gates become `15.965 ms` and
`18.715 ms`.

Sharing the Q row read removes `1,073,741,824` incremental bytes, leaving only
`8,912,896` Q output/reduction bytes, but it does not remove Q arithmetic.
Report its active time in the paired owner even when overlap hides its wall.

The Q launch has 2,048 long-lived SIMD-group threadgroups, its reducer has 256,
and the fold has 8,192. This supplies 51.2, 6.4, and 204.8 groups per GPU core.
The structural accumulator minima are four fields (16 words) for Q and two
fields (8 words) for fold. Promotion still requires compiled register counts,
resident SIMD groups, no spills/local memory, all 40 cores active outside
tails, and at least 80% of the topology-matched signed-product roof.

## Rejected ownership routes

The generic precommitted-reduction round service is the wrong owner. Even an
alias-aware version that binds only two value states must complete 14 device
messages through the first suffix message and moves at least
`6,441,664,512` bytes at log 26. Its traffic floor is `14.261 ms` before
factor reads, products, reductions, waits, or the CPU tail. The two-scan
design moves `2,156,658,688` perfect bytes and uses two completions.

Four independent precommitted lanes are worse because they duplicate the two
aliased increment states. A full-table Metal port is also rejected: it builds
and ping-pongs four `T`-field tables even though equality factors are fixed and
the useful witness has one signed increment per row. Shared reducers and host
adapters may be reused; neither route owns the state or benchmark.

## ABI, lifetime, and cutover

The base fallback borrows a producer-validated 16-byte row:

```text
u64 magnitude
u32 flags: negative, RamInc selector
u32 reserved
```

The descriptor includes device registry id, storage id, generation, rows,
byte length, point digest, and exact selector topology. It must reject a stale
generation, wrong device, wrong point order, negative zero, reserved bits, or
an aliased output. The row is retained from its single producer through the
midpoint fold, then released when no later row consumer remains. Q outputs die
after prefix round 12; dense outputs die after terminal claims.

Akita has no increment cutoff: selection follows the fused read-RAF pair.
The base fallback defaults to CPU with `cutoff=None` until five alternating
same-binary log-26 pairs establish the complete Solinas member, followed by a
log-27 transfer. Producer work, commands, waits, host rounds, Fiat-Shamir,
validation, and outputs are in both arms.

## Implementation ladder and falsifiers

1. Keep Akita routing unchanged and add fused-owner attribution for the four
   increment stages. No new kernel is needed.
2. Bring the fused read-RAF address and cycle pair below `240.727641 ms` while
   preserving every round polynomial and opening. If counters and active roofs
   make `150.454776 ms` credible, continue to 8x.
3. After feature separation, freeze a base Solinas denominator. Recheck the
   two-scan model against that exact row producer before compiling a shader.
4. Implement Q as an encode-only extension of the resident stage-6a pass,
   then the one midpoint fold. Keep a separate-Q diagnostic for attribution.
5. Add odd/even, sparse/mixed, `gamma=0/1/-1/random`, sign-boundary, transcript,
   clear-proof, and ZK-proof parity before any selector changes.

Reject or redesign if Akita registers a standalone member, a producer or row
read is omitted from paired accounting, the base path needs more than two new
completions, Q or fold misses 80% of its matched roof, a fused owner regresses
unrelated work by more than 2%, or any message, output order, transcript byte,
proof byte, or verification result differs from optimized CPU.

Unverified here: compiled occupancy and spills, physical cache-line traffic,
the best stage-6a Q host pass, base Solinas CPU time, log-27 capacity, and
end-to-end clear/ZK parity.
