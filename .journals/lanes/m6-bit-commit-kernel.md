# Lane M6 — shared-base bit-column commitment kernel

Date: 2026-09-02. Machine: Mac mini, 10 Rayon threads. Times are wall-clock release builds; the
reported ns/add divides by the original number of selected-base additions.

## Result

`g1_bit_columns_msm` commits the 2^17 × 163 wrapper table in **77.84 ms / 7.29 ns per original
addition**, down from **777.70 ms / 72.81 ns** for `batch_g1_additions_multi`: **9.99× faster** and
35% below the 120 ms gate.

HyperKZG already owns `g1_powers: Vec<G1Affine>`, so the production kernel has **zero SRS conversion
cost**. Converting the benchmark's independently generated projective bases took 6.36 ms at 2^18.

## Baseline profile

A one-second `sample(1)` trace during the old 2^18 run recorded 728 active main-thread samples. Each
Rayon worker spent about 689/728 samples asleep and about 39/728 in the parallel inversion pass.
This matches the code shape:

- Selected-point gathering is parallel, then every affine result loop is serial on the caller.
- Each tree level builds denominators, pair metadata, one new `Vec` per column, and copies every
  surviving point into those new vectors.
- Arkworks parallelizes the batch inversion, but the affine formulas and odd-point carries remain
  serial; its serial inversion chunks allocate another prefix vector.
- Projective-to-affine normalization is 6.36 ms at 2^18, under 1% of the old 1.59 s result.
- The benchmark's old timer excludes index-set construction, so the measured delta is inside the
  commitment kernel.

## Kernel

Dense columns take a six-row lookup path:

1. Split bases into groups of six; a six-bit column mask selects one of 64 subset sums.
2. Process 256 groups per Rayon work item. Build each local subset table by Hamming-weight level,
   with one Montgomery inversion batch per level.
3. Interleave every column in the work item. One selected lookup point replaces up to six original
   bases, then all column trees share one inversion batch per level.
4. Keep subset tables, column points, denominators, and inversion prefixes in flat, reused scratch
   buffers. Merge the work-item results per column in projective form.

For random 163-column input this changes about 10.68 M affine additions into about 1.25 M subset-table
adds + 3.51 M column-tree adds. A density gate selects the direct in-place column tree below 24
selected points per base, avoiding the fixed lookup-table cost on sparse or small column sets.

Every denominator batch handles identity, equal, and inverse pairs explicitly. Tests compare both
the direct and grouped paths against projective sums, including zero, singleton, all-one, duplicated,
inverse, and identity bases.

## Timings

| rows | columns | selected additions | old | new | old ns/add | new ns/add |
|---:|---:|---:|---:|---:|---:|---:|
| 2^17 | 1 | 65,523 | 7.97 ms | 6.25 ms | 121.66 | 95.30 |
| 2^17 | 16 | 1,048,399 | 77.33 ms | 19.79 ms | 73.76 | 18.87 |
| 2^17 | 163 | 10,681,067 | 777.70 ms | **77.84 ms** | 72.81 | **7.29** |
| 2^18 | 163 | 21,361,678 | 1,591.31 ms | **154.87 ms** | 74.49 | **7.25** |

Repeated clean 2^17 × 163 runs were 77.84, 78.93, and 79.01 ms. An active 10-core Criterion run in
another lane raised it to 149.41 ms and doubled the old result too; that contended run is excluded
from the table.

Before a concurrent lane replaced the wrapper benchmark entrypoint, its wired 2^17 run measured the
new commitment phase at **82 ms**, checked column zero against `HyperKZGScheme::commit`, and completed
the full prover path in 575 ms.
