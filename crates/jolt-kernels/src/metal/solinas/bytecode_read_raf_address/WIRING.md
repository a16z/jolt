# Bytecode read/RAF address worker

Status: executable resident-worker probe. The Metal shader, Rust runtime,
independent oracle, accounting model, and Criterion harness are live. The
probe still uploads a host-built carrier; it is not the production stage-6a
route until stage 5 publishes checked resident buffers.

## Boundary

For `N = 2^log_T`, `K = 2^13`, `I = 2^15`, and `O = N / I`, the worker emits
nine stage-major address tables:

```text
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j)                    s = 0..4
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j) * inc(j)           s = 5..8.
```

An absent mapped PC pushes to address zero. Metal owns only these `9K`
fields. The host retains entry-table construction, all 13 address rounds,
output claims, and Fiat--Shamir.

The carrier is address-major by cell and outer-major within each compact
stream:

```text
cell[k * O + outer] = { start: u16, count: u16 }
inner_sign[outer * I + slot] = inner | (negative << 31)
magnitude[outer * I + slot]  = abs(inc)
```

## Executable schedule

The `5 + 4` shader launches one 256-thread group per `(address, outer_tile)`.
Its eight SIMDgroups stride the tile's outer blocks. Every nonempty cell is
handled by one SIMDgroup; there is no short-cell batcher.

- Pass 1 rereads every packed cell, streams `inner_sign`, and accumulates five
  `E_lo` sums. It reduces each cell and multiplies by the corresponding
  `E_hi` value.
- Pass 2 rereads every packed cell and `inner_sign`, streams `magnitude`, and
  performs four signed field products before the same reduction.
- Each group writes nine tile partials. A second encoder in the same command
  buffer deterministically reduces the tile dimension. There are no output
  atomics or write races.

The grid is flattened with tile as the fastest coordinate so the groups for
one address can be resident together. Dynamic threadgroup memory is 640
bytes. Pipeline checks prove SIMD width 32 and the 256-thread launch limit;
registers, spills, and actual resident-group count still require capture.

For an exact topology receipt, let:

```text
U = number of nonempty cells
P = sum over nonempty cells of 32 * ceil(count / 32)
T = outer tile count.
```

The current worker issues:

```text
base update lanes              5P
signed product lanes           4P
outer-product lanes        9 * 32U
reduction lanes        1440U + 521KT
useful field products          4N + 9U.
```

Equality-table expansion is host work outside the timed worker command. The
model reports it separately and does not price it as device work.

## Traffic and storage

The current `5 + 4` worker requests:

```text
packed cells, two passes                         8OK
compact streams                                  16N
logical E_lo requests                            144N
logical E_hi requests                            144U
tile partial write + read                        288KT
output write                                      144K
-----------------------------------------------------
total                         8OK + 160N + 144U + 288KT + 144K.
```

Its forced resident floor counts the two cell passes, compact streams, one
copy of each equality table, partial write/read, and output write. Physical
ownership counts each allocation once.

The frozen Fibonacci log-26 census is:

```text
short occurrences = 1,239       short cells = 1,059
long occurrences  = 67,107,625  long cells  = 18,949
maximum cell count = 32,768      U = 20,008.
```

The aggregate census bounds `P` but does not determine its exact value:

```text
67,141,536 <= P <= 67,728,928.
```

At tile 8, the pessimistic accounting is:

```text
issued signed products             270,915,712
issued outer products                5,762,304
issued reduction lanes              62,955,776
forced resident traffic          1,233,027,072 B
physical worker ownership          888,045,568 B.
```

Tile 16 raises the reduction term to `97,100,032` lanes, the traffic floor to
`1,251,901,440 B`, and ownership to `897,482,752 B`.

## Producer accounting

The production choice must state which owner pays any transient rank plane:

- Cursor-atomic scatter with no `u16[N]` rank plane: `28N + 4OK` requested
  bytes, with `2N` count/cursor atomics priced separately.
- Rank scatter when stage 5 already owns the rank write: the address producer
  is charged `30N + 4OK` and uses `N` count atomics.
- Whole-PIOP incremental charge for the same rank design: `32N + 4OK`, because
  the rank plane is written and read once.

At log 26 those ledgers are respectively 1,946,157,056, 2,080,374,784, and
2,214,592,512 bytes. The ready carrier must bind the source allocation,
device, proof generation, all carrier allocation identities and lengths,
completion serial, exact topology counters, and zero member-local source
scan, staging, or upload.

## Current evidence and next gate

The balanced support-10 probe approximates total useful work and padded lanes
but not the production schedule: its 20,480 cells are evenly spread over ten
addresses and have counts 3,276/3,277, while production contains a count of
32,768. It is therefore a primitive screen, not complete-member evidence.

The stable log-26 tile-8 screen measured 10.478 ms GPU active and 10.945 ms
resident wall, or 25.64/24.54 G useful products/s. Tile 16 reached a slightly
lower active median but had noisy wall samples; tile 32 was 10.684/11.237 ms.

Before production integration, replay the captured `(outer,address,count)`
layout and compact stream with the actual nine Fiat--Shamir equality tables.
Alternate tiles 8, 16, and 32; compare all 73,728 fields with the optimized
CPU oracle; capture wall, active time, external bytes, registers, spills, and
resident groups.

Promotion requires:

- exact carrier, table, 13-round, final-claim, and clear/ZK proof parity;
- no stage-6a row scan, repack, staging allocation, or upload;
- one worker command and one host wait;
- complete address member at or below 27.700 ms over five alternating pairs;
- whole address charge within the frozen family allowance after charging the
  producer once;
- no spills and at least two resident groups per core; and
- a paired log-27 CPU/Metal speedup of at least 5x.

The verifier, transcript schedule, and public protocol do not change.
