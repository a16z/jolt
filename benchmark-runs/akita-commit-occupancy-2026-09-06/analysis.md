# Working model: root commit only

Source audit: ../akita-commit-eval-boost-2026-09-05/commit-eval-kernel-audit-2026-09-06.md,
sections2 and6. Source and public counter availability are verified; occupancy,
physical register count, spills and physical DRAM bytes remain unknown.

## Exact boundary

Root panel computes rank3 negacyclic shifted sums at D128, P524288, K256,
29 live/32 capacity columns,1024 blocks/column,16 position partials. It emits
canonical fp128 partial coefficients for the existing final reduction.
Production code, matrix footprint, selector loop, arithmetic,1024-thread
launch and32KiB shared tile are unchanged in initial diagnostic D0.
Full traced H3263846381 gives U=H*384=1253317010304 updates;
panel12.319893832s, device envelope12.485318084s. Production commands contain
8 streams *3rank *16partials=384 threadgroups, not the full16752 simultaneously.

## Bounds and unknowns

Matrix requests349 *3GiB=1047GiB. At measured copy440.544806GiB/s, the
all-requests-hit-DRAM proxy is2.376602653s; this is conditional, not an
irreducible traffic floor because caches reuse matrix data. Compulsory A,
allocated selector and hint bytes alone give(3+7.25+0.1875)/440.544806=
0.023692s, ignoring shared gathers, partials and overfetch. Neither is an
achievable prediction. Logical shared gathers16U=20.053TB.

Direct source tally~36 scalar operations/update =>45.119412371T operations.
Compute proxy is45.119412371/R seconds for effective issue rate R Tops/s.
R and actual ISA count are not measured; fitting R to12.32s is circular.
No defensible numerical compute floor or dominant limiter is established.
The initial work is therefore identification, not code promotion against an
invented optimum. Exact occupancy/optimality certification is unavailable.

## Ranked mechanisms and costs

1. Resource/concurrency pressure:32KiB shared plus40 persistent source words
per lane,32SIMDs/group. D0 tests fixed production PSO saturation at48..768
groups (production384); controlled shared-reservation follows only if useful.
2. Barrier imbalance: two task-specific ballots/loops per SIMD followed by
group synchronization. Count selected iterations per SIMD on real inputs
before changing mapping. Source-level binomial estimate is not idle time.
3. Live-state relief: price matrix sweep growth, instructions and barriers.
R4 already showed arithmetic-only throughput did not transfer; reshuffling
accumulators remains closed until a new causal observation justifies it.

## D0 preregistration: unchanged production shader saturation

Question: does the current384-group dispatch lie below the throughput plateau
for the complete production kernel body at its P19/3GiB matrix footprint?
Host-only harness loads the accepted unmodified MSL and compiles with safe
math/default production pipeline policy. No kernel edits in this diagnostic.
Use deterministic synthetic selectors with~56% hot entries, native stride29;
clearly not a Fibonacci input or an end-to-end prediction. Preserve total
matrix footprint and per-group2048tile work. Allocate real output geometry;
only initialize/touch the source prefix needed for the largest dispatch.

First exact reduced-shape full-output CPU oracle covers zero bytes, selected
zero mask, rotations/sign, tail tasks, rank/position partials, padded columns.
At P19, independent sampled oracle at first/last/random output coordinates
and exact H of dispatched tasks. Report GPU ms, updates/s, group count,
pipeline limits, input hash/seed, command status and telemetry if available.
One bounded warmup; initial order streams1,8,2,16,4,8. The repeated8 control
detects thermal/order drift, not a license for repeated optimization samples.

Expected: saturation by production384 groups. Falsifier:768 groups gives
>=10% higher useful updates/s than384 with <=3% control drift. That unlocks
a separately priced batching/layout test, not an occupancy claim. If plateau
is already reached, deprioritize grid-size tuning and investigate shared
resource/barrier mechanisms. >3% control drift or mismatched oracle =>invalid;
at most one targeted ordering repeat after explanation. Budget: <=15min
implementation/compile, <=5min guarded run including120s initial cooling.

Tooling: no-sudo macmon source fetched for inspection; frequency/power useful
as controls, not occupancy. IORegistry coarse GPU statistics work without
sudo. M4 IOReport bandwidth fallback may saturate and is not a roofline.
No telemetry installation, pipeline change or candidate promotion yet.
