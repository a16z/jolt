# Bytecode address direct-consumption amendment

This amendment replaces the CSR shadow comparison with a checked handoff into
the existing optimized host address-round shell. It does not move a sumcheck
round or a Fiat--Shamir operation to Metal, and it does not change the stage-6b
cycle member.

## Exact boundary

At `log_T = 26` and `log_K = 13`, Metal consumes the stage-5 resident
`BooleanityRows` allocation and the nine cycle points carried by the relation.
It produces one stage-major array containing nine canonical Akita
pushforwards, each of length `K = 8,192`:

```text
F_s(k) = sum_{j: push_pc(j) = k} eq(r_s, j)                    s < 5
F_s(k) = sum_{j: push_pc(j) = k} eq(r_s, j) * fused_inc(j)     s >= 5.
```

An absent mapped PC pushes to address zero. The source allocation stays alive
through an owned `BooleanityRows` lease. Admission checks its length, device
registry identity, allocation identity, the completed command's source
identity, every static allocation identity, status, diagnostics, output
shape, and the domain of the host-supplied row-zero pushed PC used by
`EntryTrace`.

The host remains authoritative for:

- the six raw value tables and the stage mapping
  `T0,T1,T2,T3,T4,T5,T5,1-T5,1-T5`;
- `Int(k)`, `EntryTrace(k)`, and `EntryExpected(k)`;
- gamma powers, the two within-stage RAF weights, and the entry weight;
- all 13 low-to-high address rounds, polynomial absorption, challenge draws,
  the reversed output point, and output-claim construction;
- the six raw committed-program outputs. Complemented uses of `T5` never
  become output claims.

The optimized module needs one narrow constructor that accepts the checked
stage-major pushforwards and row-zero address, then builds the same private
`AddressKernel` shell as the CPU path. The stage algebra must remain
single-sourced there; it is not duplicated in this Metal packet. The current
observation owns one flat vector while `AddressKernel` owns nine polynomials.
The constructor must either teach the shell to retain the flat backing store or
charge the partition copy to the complete-member timing.

Pushforwards are independent of the six stage-6a Fiat--Shamir challenges.
Stage 6a must therefore call relation-dependent Metal preflight after building
the relation but before `draw_challenges`. The preflight joins and validates
the command, then parks either a checked handoff or a frozen CPU-fallback
decision in the proof session. After the challenges and `begin_batch`,
`PrepareKernel::prepare` only consumes that decision. It must not discover a
new recoverable Metal failure after transcript mutation.

## Lower bound and working set

The durable five-pair CPU denominator is `190,915,958 ns`; the strict integer 5x
cap is `38,183,191 ns` (`38.183191 ms` rounded for display). The latest
one-pair diagnostic measured `206,118,083 ns`; it is context, not a replacement
acceptance denominator. The tracked
26-address CSR evidence is `29,109,917 ns` for the complete slice, including
status validation and pushforward readback but excluding the real host shell.
It leaves `9,073,274 ns` below the strict cap.

The retained 13-round CPU component is `7,918,251 ns`. Adding it to the CSR
slice gives a scheduling screen of `37,028,168 ns`, leaving `1,155,023 ns` for
shell construction and costs absent from that component. Component medians are
not additive performance evidence, so this makes `<= 38.18 ms` credible but
unproved. Only an alternating complete-member measurement can establish 5x.

The initial host shell contains 18 field tables of length `K`: nine
pushforwards, six raw values, `Int`, `EntryTrace`, and `EntryExpected`. Its
field payload is `18 * 8,192 * 16 = 2,359,296` bytes. The pushforward readback
is `1,179,648` bytes and is already included in the complete-slice evidence.
Committed-program mode adds six scalar outputs, not six additional tables.

The current log-26 device working set is:

```text
successor-owned             544,538,768 bytes
shared resident rows      2,684,354,560 bytes
aggregate                 3,228,893,328 bytes
```

Producer-side address counts use one `u32` for each `(outer,address)` cell.
There are `2,048 * 8,192 = 16,777,216` cells, so the plane is `67,108,864`
bytes and the aggregate resident set becomes `3,296,002,192` bytes. The plane
belongs to the producer; it is not another bytecode-consumer row copy.

For the observed `U = 53,248` runs, the existing two-pass CSR model charges:

```text
B_two_pass = 84N + 40U + 32O = 5,639,340,032 bytes.
```

Consuming resident counts removes the first 40-byte row scan but adds one read
of the count plane:

```text
B_counts = 44N + 40U + 32O + 4OK = 3,022,094,336 bytes.
```

The consumer-side reduction is `2,617,245,696` bytes, or `5,794,191 ns` at
the measured `451,701,710,520 B/s` copy rate. If this consumer is the sole
reason the producer writes the count plane, fair whole-PIOP accounting adds
`67,108,864` write bytes; net traffic reduction is `2,550,136,832` bytes, or
`5,645,622 ns` at the same rate. These are movement deltas, not predicted
stopwatch wins: atomics, cache residency, and overlap can make the realized
delta smaller.

## Bound-changing candidates

1. **Direct host-shell consumption.** Remove the optimized CPU pushforward
   preparation and elementwise comparison. Correctness is unchanged because
   the checked handoff supplies exactly the nine tables the common shell
   consumes. This is required before the current CSR result has end-to-end
   meaning.
2. **Producer-owned address counts.** Produce `u32[O*K]` counts during the
   authoritative stage-5 traversal, retain their allocation identity and
   initialization completion, and use them for prefix/run construction. The
   CSR scatter still reads every row once. This removes one full row scan;
   it does not justify the existing `ReusedProducer` zero-CSR charge, which is
   reserved for a future producer-owned complete topology.
3. **Producer-owned complete topology.** Cells plus compact occurrences could
   remove CSR entirely, but changes producer storage and scheduling more
   substantially. It is out of scope until the count-plane experiment shows
   the remaining CSR cost matters.
4. **Device address rounds.** Rejected. Thirteen host/device dependencies for
   less than 2.4 MiB of host tables cannot improve the movement bound, and
   moving Fiat--Shamir would enlarge the protocol boundary.

## Falsification and promotion bar

- Before any transcript mutation, admission must yield either a valid direct
  handoff or the optimized CPU decision. Identity or telemetry corruption is
  fail-closed, not silently promoted.
- Exact parity covers all nine pushforwards, every one of the 13 round
  polynomials and challenges, the reversed opening point, `intermediate`, and
  all six committed-program raw value outputs. Canonical checksums are retained.
- Alternating complete CPU/direct-Metal members at log 26 must have median at
  most `38,183,191 ns`, relative MAD at most 3%, no first-sample winner, and no
  capacity fallback. Report log 19, 20, 21, and 26 to locate the crossover.
- The producer-count candidate is accepted only with the count allocation's
  identity, initialization/completion evidence, and producer write attribution
  in the artifact. Measure it against the same complete-member boundary; do
  not subtract the analytical `5.79 ms` from a stopwatch result.
- Promotion also requires the existing shader binary, register/spill,
  occupancy, status/diagnostic, and matched-control evidence in the same
  device record.

## Unresolved integration risks

- `BooleanityRows` currently exposes allocation and device identity but no
  generation or initialization serial. Its `Arc` is a real lifetime lease,
  but durable producer-completion evidence still needs an ABI extension.
- The row-zero pushed PC needed for `EntryTrace` is absent from the current
  observation. The handoff therefore accepts a domain-checked preflight host
  value, which integration must read from the same authoritative witness row
  cache. A producer receipt or dedicated completed scalar would bind it more
  strongly; it must never be inferred from the program entry index.
- The generic stage driver calls `begin_batch` before `prepare`. Stage 6a must
  explicitly run and freeze the relation-dependent preflight before its six
  challenge draws, or the claimed fallback boundary is false.
- The optimized host shell has no constructor for precomputed pushforwards.
  That refactor is the only required edit outside this isolated packet. Its
  current per-polynomial ownership may add a 1,179,648-byte partition copy.
- The `29.109917 ms` evidence does not measure host shell construction or the
  complete member, and the producer-count traffic reduction has not been
  observed with counters.
