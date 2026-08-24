# Akita Metal Stage 4/5 compatibility scatter

## Boundary

At BTreeMap T28, Stage 4 starts the Instruction Read-RAF compatibility worker after
`RegistersReadWriteChecking::prepare`. The worker groups the Stage-1 resident rows,
constructs four dense consumer planes, and computes the first address-phase message.
Stage 5 joins an already-complete worker. The transcript point, grouped-row order,
scatter kernel, address-phase schedule, proof, and verifier are unchanged by the B1
campaign below.

The currently accepted trace measures 5.4600 s for Stage 4 and 2.4543 s for Stage 5,
or 7.9143 s combined. The untraced compatibility construction occupies 1.5349 s
from worker release to the visible address-prefetch span; that span adds 0.2684 s.
Both finish during Stage 4's third register round. Treating their 1.8033 s total as
additive latency is therefore invalid.

The optimized-CPU trace measures 2.5317 s across the register rounds versus 3.6151 s
in the Metal trace. The first three Metal rounds overlap the worker and take 1.9762 s
versus 1.3245 s on CPU, a 0.6517 s difference. Later-round differences and the
0.4579 s prepare difference have no simultaneous compatibility work and are excluded
from this candidate's causal ceiling.

## Traffic and storage

For `T = 2^28`, the grouped consumer writes one byte of packed metadata, 16 bytes of
lookup limbs, four bytes of inverse permutation, and 16 bytes of equality weight per
row. The four planes total `37*T = 9.25 GiB`. It also reads 33 bytes per Stage-1 row.
The fused bytecode outputs are already private Metal buffers. The four grouped planes
are allocated with `StorageModeShared`, although the resident address sequence only
binds them to later GPU commands. Its validation reads buffer lengths and allocation
identities, not their contents; CPU materialization exists only in the separate
non-resident construction branch.

Changing these four outputs to `StorageModePrivate` does not reduce their physical
capacity or the scatter's compulsory reads and writes. It removes CPU mapping and
coherence from a GPU-only lifetime. At the measured 412.5 GiB/s, the roughly
19.9 GiB read/write stream has a traffic floor near 0.05 s; 268 million fp128 weight
multiplications add about 0.02 s at the measured arithmetic rate. The observed
1.5546 s construction interval is therefore dominated by allocation, residency,
synchronization, or inefficient execution rather than compulsory traffic. Private
storage can address the first two mechanisms, not the floor.

## Candidate B1: private grouped consumer planes

Change only the four compatibility-scatter output allocations from shared to private
Metal storage. Keep the source, equality tables, status word, bytecode outputs,
dispatch geometry, worker release point, and address prefetch unchanged. Record the
storage mode, exact output bytes, compatibility wall time, and address-prefetch wall
time at the existing `jolt::metal` telemetry boundary. The capacity ledger must
continue charging all 9.25 GiB even if process RSS accounts private storage
differently.

The prediction is a 0.3--0.7 s complete-prover saving. The upper edge is the
0.7091 s difference in the only two register rounds that overlap the worker; no later
Stage 4 difference is credited. The lower edge allows most of the interval to be GPU
execution or unavoidable physical allocation.

Add a red storage-lifetime assertion to the existing fused Stage-1 scatter oracle,
then run that one focused Metal test. Because the production prefetch is admitted
only at T28, a T25 proof does not exercise the candidate. Admit one BTreeMap T28
treatment after the focused test passes. The full proof must verify, telemetry must
report 9.25 GiB of private grouped outputs with no fallback, and the exact working-set
ledger must stay unchanged. Retain only at 47.58 s or below against the 48.08 s
parent, with peak RSS below 90 GiB. One repeat is legal only if a promotion result is
within ordinary whole-prover noise. Otherwise revert the storage change and preserve
the result; do not tune threadgroup width or worker timing under the same candidate.

This is a prover-local resource-policy change. It changes no protocol message,
challenge, polynomial, proof byte, verifier code, or soundness assumption.

## B1 result

The focused oracle passed and the T28 proof verified with exactly 9,932,111,872
private grouped-output bytes. The compatibility scatter took 1.5439 s, only 0.0107 s
below the 1.5546 s trace anchor, while address prefetch rose from 0.1805 s to
0.2571 s. Complete proving regressed from 48.08 s to 50.25 s at 80.09 GiB RSS.
This misses the 47.58 s promotion bar by 2.67 s and was reverted in `a3709b12b`.

Private storage did not remove the construction cost on this unified-memory system.
The first-two-round difference cannot be attributed to CPU mapping or coherence of
the four outputs. Do not retry storage modes, cache flags, or hazard flags without a
new mechanism and a separately measured ceiling. Any further Stage 4/5 candidate
must address scheduling or eliminate work rather than relabel the same allocations.

## Candidate B2: cycle-major source with one grouped index

### Exact dataflow boundary

The four compatibility outputs are not all semantic state. They are a byte of packed
table/RAF metadata, a 16-byte lookup index, a four-byte cycle-to-table-major inverse,
and a 16-byte equality weight per row. The packed byte and lookup index duplicate the
resident Stage-1 claim and the first two column-major source words. The inverse exists
only because later kernels must recover cycle order after consuming those copies.

Only the suffix scan requires table-major iteration. The RAF scan and the Stage-5
cycle phase are naturally cycle-major, and Stage-6b Instruction RA also asks for
lookup indices in cycle order. B2 therefore keeps exactly one four-byte
table-major-to-cycle index for suffix jobs and one 16-byte evolving weight per cycle.
RAF, cycle, and Instruction RA kernels read lookup limbs directly from the existing
Stage-1 column-major source. The claim byte supplies the table and RAF flags. The
first RAF phase initializes the weight from the split reduction equality tables;
later phases update it in place exactly as today.

The scatter still uses the published per-chunk selector counts and the same stable
segment partition. Its only Instruction Read-RAF output becomes
`table_major_to_cycle[grouped] = cycle`; the fused bytecode outputs remain unchanged.
The address/cycle challenge order, 16 eight-bit address phases, Product5 handoff,
round polynomials, output claims, transcript, proof bytes, and verifier all remain
unchanged. This is an ownership/layout change, not a protocol change.

### Traffic, storage, and lower bound

At T28 the current four planes occupy `37*T = 9.25 GiB`. B2 occupies `20*T =
5.00 GiB`: 1 GiB of grouped indices and 4 GiB of weights. It removes exactly
`17*T = 4.25 GiB` of live capacity. The Stage-1 source is not a new charge: its row
buffer is already retained through Stage 6b/7 by the booleanity and other published
consumers. A later lookup carrier must clone only that row buffer, not the owner that
also retains the claim allocation.

With fused bytecode enabled, the scatter must still read the 32-byte Stage-1 row and
one-byte claim and write the bytecode carrier. It now writes four rather than 37
Instruction Read-RAF bytes per row, removing `33*T = 8.25 GiB` from the overlapping
scatter. Initializing weights in the first RAF pass preserves that pass's 33-byte
per-row compulsory stream: 17 source bytes read and 16 weight bytes written replace
the old 33 grouped bytes read. Later RAF phases preserve their 49-byte stream.

Suffix phases add a four-byte grouped-index load per selected row. For BTreeMap's
150-million-row physical trace, all 16 phases add at most 8.94 GiB. The first two
Stage-5 cycle traversals each remove the old four-byte inverse load, saving 2 GiB in
total. The five Stage-6b lazy/materialization traversals remove another 5 GiB. Thus
the complete BTreeMap path saves about 6.3 GiB of compulsory traffic; an all-selected
trace is approximately traffic-neutral. This candidate is not justified by a
bandwidth roofline alone. Its mechanism is eliminating 4.25 GiB of allocation and
residency plus an indirection from seven downstream full-domain traversals.

At the accepted 412.5 GiB/s calibration, the raw traffic term is only about 0.02 s.
The measurable ceiling comes from ownership: the scatter overlaps 0.6517 s of excess
register-round time, and C2 showed that adding one 4 GiB resident table can displace
at least 2.65 s of later work. Credit only 0.2--0.5 s of the overlap and 0.1--0.4 s
from direct Stage-5/6b reads and reduced residency. The preregistered complete-prover
prediction is therefore a 0.3--0.9 s saving, with an exact 4.25 GiB capacity saving.

### Rejected nearby layouts

Recomputing all prior phase weights would remove the 4 GiB weight plane but add 120
fp128 multiplications per row across the 16 sequential phases; it is outside the
compute floor. Scanning the cycle-major source once per table removes the grouped
index but multiplies source traffic by up to 40. Keeping copied grouped lookups merely
to avoid the four-byte suffix indirection preserves 4 GiB of the allocation being
targeted. Chunk-local 16-bit indices require a much larger job/partial schedule and
are deferred unless the simple four-byte index proves a measured bottleneck.

### Falsifier and admission gates

First add a resident-source parity test that compares every Stage-5 round polynomial
and output claim with the optimized CPU kernel, plus a route assertion that fails on
the four-plane layout. The receipt must report exactly 5.00 GiB at T28, no copied
packed or lookup plane, one grouped-index allocation, one weight allocation, no full
readback, and the original fused-bytecode carrier.

Run one verified T25 BTreeMap sentinel after focused parity. It must exercise the
resident-source route, preserve exact proof verification and fallback counters, and
improve either the compatibility construction by at least 10% or Stage-6b
Instruction RA by at least 5% without regressing complete proving by 0.2 s. Only then
admit one T28 treatment. At T28 retain only if complete proving is at most 47.58 s,
or if it improves by at least 0.3 s while exact telemetry confirms a 4.25 GiB capacity
reduction and peak RSS falls by at least 2 GiB. Reject on a slower Stage-5 address
phase that erases the downstream gain, any extra source scan, stale ownership,
verification failure, fallback, swapping, or RSS above 90 GiB. Do not sweep index
widths, storage modes, worker timing, or dispatch widths under B2.
