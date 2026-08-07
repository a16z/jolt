# Bytecode read/RAF address v2: staged resident carrier

Status: design and CPU-model packet. This directory is deliberately not
registered in the Metal backend or shader source. It defines the producer
receipts, address-major topology contract, and pre-registered performance
screen for the eventual stage-6a address kernel.

## Decision and boundary

Keep the fixed address-major worker from
`bytecode_read_raf_successor/WIRING.md`, but make its residency claim a checked
type rather than a modeling switch. The stage-6a member may use the v2 worker
only when a producer receipt proves that:

- address counts were emitted while stage 5 already visited the authoritative
  `BooleanityRows`;
- compact address-major streams were published before the member began;
- the member performs no standalone count/CSR scan, row repack, host staging,
  or row upload; and
- every allocation, device, generation, row count, and topology counter still
  matches the producer.

Without that receipt, the complete optimized CPU member is the fallback. A
naked `ResidentProducer` enum is insufficient evidence.

For `N = 2^log_T`, `K = 2^13`, `I = 2^15`, and `O = N / I`, the device still
produces exactly nine canonical Akita tables:

```text
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j)                    s = 0..4
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j) * inc(j)           s = 5..8.
```

An absent mapped PC pushes to address zero. Metal owns only these nine
pushforwards. Stage values, entry tables, all 13 address rounds, output claims,
and Fiat--Shamir remain in the unchanged host shell.

The hard log-26 target is `27.700000 ms` complete member wall. This is slightly
stricter than the `27.735322 ms` address allowance derived from the frozen
seven-times address-plus-cycle family denominator. The standalone five-times
cap is `40.817025 ms`; it is a minimum gate, not the stopping condition.

## Staged carrier

The carrier has two producer transitions and one consumer transition.

### 1. Count publication

While stage 5 writes each resident row, it increments the `(outer, address)`
count in the final packed-cell allocation. The receipt requires exactly `N`
count updates and `O` completed outer blocks, with zero additional source-row
scans. PC masking is the address-phase decoder: the low 56 bits of word four
are mapped-PC-plus-one and zero maps to address zero.

### 2. Prefix and scatter publication

One producer-owned dispatch converts counts to

```text
cell[k * O + outer] = { start: u16, count: u16 }
```

and scatters two compact outer-major streams:

```text
inner_sign[outer * I + slot] = inner | (negative << 31)
magnitude[outer * I + slot]  = abs(inc).
```

It may make one charged scan of the already-resident PC/sign and magnitude
words. It is not a stage-6a member scan. Every cell and stream slot is written
once; no run arena, occurrence index, indirect dispatch list, or global output
atomic survives into the carrier.

The ready receipt binds the source allocation and generation, all three
carrier allocations, the first pushed PC used by `EntryTrace`, exact schedule
counters, producer wall/active time when measured, and zero member-local row
traffic.

### 3. Address-owned consumption

One 256-thread group owns each address. Long cells use one SIMDgroup per cell;
short cells batch up to 32 consecutive outers for the same address. Each group
accumulates its address locally and writes nine fields once.

The default `5 + 4` stage schedule makes two compact passes:

- base: read `inner_sign`, perform five cached `E_lo` lookups/additions;
- fused: read `inner_sign` and `magnitude`, perform four signed products and
  additions.

This yields nine useful stage updates per 16 compact bytes, or `0.5625`
updates/byte and 4.5 updates per compact occurrence read. A single-pass
nine-accumulator variant reaches `0.75` updates/byte and saves `4N` bytes, but
it is admitted only if compiler evidence shows no spills, at least two
resident groups per core, and at least a three-percent complete-wall win. The
two-pass default structurally caps live stage accumulators at five.

## Frozen production topology

The supplied Fibonacci log-26 census is:

```text
short occurrences = 1,239       short cells = 1,059
long occurrences  = 67,107,625  long cells  = 18,949
maximum cell count = 32,768
```

Thus `U = 20,008` nonempty cells and 99.998154% of rows are long-cell
occurrences. A compact short-only alternative is not justified before an
exact schedule receipt predicts at least five percent complete-wall benefit.

The aggregate census does not determine short batches or SIMD padding. The
model therefore derives honest bounds and requires the final producer receipt
to supply the exact values:

```text
34 <= B_s <= 1,059
1,248 <= P_s <= 39,648
67,107,648 <= P_l <= 67,695,040.
```

For the pessimistic bound, useful and issued work is:

```text
useful signed products                 4N = 268,435,456
useful outer full products             9U = 180,072
issued signed-product/update lanes        = 270,938,752
issued outer full-product lanes           = 911,360
equality-generation full products         = 626,652
issued accumulation-add lanes             = 609,612,192
issued reduction-add lanes                = 30,908,672
producer count/cursor atomics           2N = 134,217,728
member global output atomics                = 0.
```

## Traffic equations

The producer-count target removes the rejected standalone CSR count scan. Its
prefix/scatter request is

```text
source PC/sign and magnitude reads       16N
compact stream writes                    12N
packed-cell publication                  4OK
--------------------------------------------
producer target                    28N + 4OK.
```

At log 26 this is `1,946,157,056` bytes, a `4.308501 ms` copy-rate floor.
Count/cursor atomic throughput and producer interference remain separate
measured terms; moving this work earlier does not erase it from the PIOP
ledger.

For the default worker:

```text
packed cells                              4OK
compact streams                           16N
logical E_lo requests                    144N
logical E_hi requests                    144U
output writes                            144K
--------------------------------------------
shader requested       4OK + 160N + 144U + 144K.
```

For the production census this is `10,808,587,904` requested bytes. The
forced streaming floor, counting the second compact-index pass but only one
copy of the small equality tables, is `1,147,043,840` bytes (`2.539384 ms` at
the retained `451.701710520 GB/s` copy control). The unique carrier/equality/
output allocation is `878,608,384` bytes (`1.945108 ms`). A one-pass worker
reduces the forced floor to that unique value and can save at most
`0.594276 ms` at the copy control before occupancy costs.

## Receipt-bound roof

`model.rs` separates an incomplete product/traffic screen from a promotable
roof. The complete roof requires:

- a validated ready-carrier receipt with exact `B_s`, `P_s`, and `P_l`;
- matched base-update, fused-update, outer/equality-product, SIMD-reduction,
  and copy rates from the compiled worker shape;
- measured host shell, 13-round, finish, and output-claim wall; and
- the producer's separately charged incremental wall for whole-PIOP claims.

Using pessimistic schedule bounds, 80% utilization, the old `7.918251 ms`
host-round proxy, and one `0.141 ms` command boundary gives only an incomplete
screen:

| product control | partial complete-member screen | remaining to 27.7 ms |
| --- | ---: | ---: |
| full-width at 18.10 Gupdate/s | `26.876708 ms` | `0.823292 ms` |
| exact-u64 at 26.272 Gterm/s | `21.056511 ms` | `6.643489 ms` |

Both omit 609.6 million accumulation lanes, 30.9 million reduction lanes,
shell construction, finish/output work, and any producer charge. Therefore
neither row is a promotion claim. Full-width remains the first measured
candidate because it won the old CSR worker, but the fixed worker must retest
both paths; the receipt-bound model makes the missing headroom explicit.

## Falsification and integration gates

- No stage-6a source-row scan, repack, staging allocation, or upload.
- Exact parity for every cell and compact occurrence, all 73,728
  pushforwards, all 13 round polynomials, final intermediate value, and six
  committed value claims.
- Exact producer and consumer device, allocation, generation, length, and
  byte-size matches; invalid/reserved counters fail closed.
- One worker command and one host wait. Producer construction is separately
  metered and charged once.
- No spills and at least two resident address groups per core for the
  selected stage tiling.
- Counter-measured external worker traffic no more than twice the
  `878,608,384`-byte unique minimum.
- Five alternating clean log-26 pairs with every complete address sample at
  or below `27.700000 ms`, followed by log-27/log-28 capacity checks.

The verifier, transcript schedule, proof shape, and public protocol do not
change.
