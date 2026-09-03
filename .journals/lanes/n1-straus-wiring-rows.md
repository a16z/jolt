# Lane N1 — fixed-wiring Straus row counts

Date: 2026-09-02. Source: M1's seeded `2^22` Dory opening (`sigma=11`, `N=41`) plus its seeded synthetic `sigma=12`, `N=41` scalar set.

## Decision

**Fixed unsigned: `w=4`, 283,162 rows at `sigma=11`; 301,972 at `sigma=12`; both require `2^19`. Signed centered digits: `w=5`, 245,449 rows at `sigma=11`; 261,550 at `sigma=12`; both fit `2^18`.** The signed `sigma=12` case has 594 spare rows before operand/input rows, so a combined table that also stores those rows still spills to `2^19`.

The fixed `w=4` schedules need 56,938 / 60,880 proof-dependent selector bits at `sigma=11/12`. Signed `w=5` needs 57,160 / 61,120 bits.

## Total rows

Totals include the GT, G1, and G2 schedules, the 14,380-row four-pair Miller loop, and the 3,288-row final-exponentiation relation. They exclude separate operand/input storage rows, matching M1's 205,079 / 230,641 totals.

| sigma | variant | w | GT rows | G1 rows | G2 rows | pairing + FE | total | domain |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | fixed | 3 | 163,224 | 51,075 | 107,970 | 17,668 | 339,937 | `2^19` |
| 11 | fixed | **4** | 135,552 | 42,114 | 87,828 | 17,668 | **283,162** | `2^19` |
| 11 | fixed | 5 | 142,476 | 41,980 | 87,560 | 17,668 | 289,684 | `2^19` |
| 11 | signed | 3 | 158,040 | 49,854 | 105,528 | 17,668 | 331,090 | `2^19` |
| 11 | signed | 4 | 130,416 | 39,265 | 82,130 | 17,668 | 269,479 | `2^19` |
| 11 | signed | **5** | 116,556 | 35,875 | 75,350 | 17,668 | **245,449** | `2^18` |
| 12 | fixed | 3 | 173,376 | 55,143 | 116,646 | 17,668 | 362,833 | `2^19` |
| 12 | fixed | **4** | 143,976 | 45,456 | 94,872 | 17,668 | **301,972** | `2^19` |
| 12 | fixed | 5 | 151,332 | 45,310 | 94,580 | 17,668 | 308,890 | `2^19` |
| 12 | signed | 3 | 167,868 | 53,823 | 114,006 | 17,668 | 353,365 | `2^19` |
| 12 | signed | 4 | 138,516 | 42,376 | 88,712 | 17,668 | 287,272 | `2^19` |
| 12 | signed | **5** | 123,792 | 38,710 | 81,380 | 17,668 | **261,550** | `2^18` |

## Exact operation counts

Each cell is `table precomputation / window square-or-double / online table operation`. Every online slot executes, including digit zero via the identity. Every window executes all `w` leading squares/doubles, including the top identity window.

| sigma | variant | w | GT ops | G1 ops | G2 ops |
|---:|---|---:|---:|---:|---:|
| 11 | fixed | 3 | 864 / 66 / 12,672 | 222 / 129 / 3,182 | 222 / 69 / 3,404 |
| 11 | fixed | 4 | 2,016 / 64 / 9,216 | 518 / 128 / 2,368 | 518 / 68 / 2,516 |
| 11 | fixed | 5 | 4,320 / 65 / 7,488 | 1,110 / 130 / 1,924 | 1,110 / 70 / 2,072 |
| 11 | signed | 3 | 432 / 66 / 12,672 | 111 / 129 / 3,182 | 111 / 69 / 3,404 |
| 11 | signed | 4 | 1,008 / 68 / 9,792 | 259 / 128 / 2,368 | 259 / 68 / 2,516 |
| 11 | signed | 5 | 2,160 / 65 / 7,488 | 555 / 130 / 1,924 | 555 / 70 / 2,072 |
| 12 | fixed | 3 | 918 / 66 / 13,464 | 240 / 129 / 3,440 | 240 / 69 / 3,680 |
| 12 | fixed | 4 | 2,142 / 64 / 9,792 | 560 / 128 / 2,560 | 560 / 68 / 2,720 |
| 12 | fixed | 5 | 4,590 / 65 / 7,956 | 1,200 / 130 / 2,080 | 1,200 / 70 / 2,240 |
| 12 | signed | 3 | 459 / 66 / 13,464 | 120 / 129 / 3,440 | 120 / 69 / 3,680 |
| 12 | signed | 4 | 1,071 / 68 / 10,404 | 280 / 128 / 2,560 | 280 / 68 / 2,720 |
| 12 | signed | 5 | 2,295 / 65 / 7,956 | 600 / 130 / 2,080 | 600 / 70 / 2,240 |

Original/expanded bases are `144/576` GT and `37/74` G1 and `37/148` G2 at `sigma=11`; `153/612`, `40/80`, and `40/160` at `sigma=12`. Observed mini-scalar widths are 64-bit GT 4D, 127-bit G1 2D, and 67-bit G2 4D. Endomorphisms map each original base's table to all GLV-image tables, so table construction is paid once per original base. Fixed construction costs `2^w - 2` group operations per original base. Signed construction stores magnitudes `1..=2^(w-1)` and costs `2^(w-1) - 1`; inverse/negation is a linear coordinate map.

Centered digits use `[-2^(w-1), 2^(w-1))`. A carry gives GT 17 windows at `w=4`, versus 16 unsigned windows; the other window counts are visible in the square/double column divided by `w`.

## Row formulas

| group operation | Fq products | output-coefficient rows |
|---|---:|---:|
| GT table/online Fq12 multiply | 54 | 12 |
| GT cyclotomic square | 18 | 12 |
| G1 table mixed add | 11 | 11 |
| G1 double | 7 | 7 |
| G1 online projective add | 15 | 15 |
| G2 table mixed add | 30 | 22 |
| G2 double | 17 | 14 |
| G2 online projective add | 41 | 30 |

G1 uses arkworks' `a=0` Jacobian `8M+3S`, `3M+4S`, and `11M+4S` formulas. G2 applies Fq2 `M=3` / `S=2` Fq products to the same formulas. Table construction repeatedly adds the affine original base; online selection consumes the projective table output. GT inverse is cyclotomic conjugation; EC inverse is point negation. Both cost zero bilinear rows.

## Wiring data

`public selector bits` counts proof-dependent binary data. Fixed schedules encode `w` magnitude bits per digit plus one GLV-component sign bit. Signed schedules encode each centered digit in `w` two's-complement bits. GT's fixed scalar `1` is omitted from public bits but remains in the operation schedule.

`fixed operand offsets` counts stored nonidentity table locations after GLV-image sharing; identity is one shared constant. Per selector, fixed has `2^w` choices including identity. Signed has `2^(w-1)+1` stored-offset choices (zero plus magnitudes) and applies sign as a linear inverse/negation map.

| sigma | variant | w | public digit selectors | public selector bits | fixed operand offsets | choices/selector | shift relations |
|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | fixed | 3 | 19,170 | 58,304 | 1,526 | 8 | 11 |
| 11 | fixed | 4 | 14,036 | 56,938 | 3,270 | 16 | 11 |
| 11 | fixed | 5 | 11,432 | 57,954 | 6,758 | 32 | 11 |
| 11 | signed | 3 | 19,170 | 57,510 | 872 | 5 | 11 |
| 11 | signed | 4 | 14,608 | 58,432 | 1,744 | 9 | 11 |
| 11 | signed | 5 | 11,432 | 57,160 | 3,488 | 17 | 11 |
| 12 | fixed | 3 | 20,496 | 62,336 | 1,631 | 8 | 11 |
| 12 | fixed | 4 | 15,008 | 60,880 | 3,495 | 16 | 11 |
| 12 | fixed | 5 | 12,224 | 61,968 | 7,223 | 32 | 11 |
| 12 | signed | 3 | 20,496 | 61,488 | 932 | 5 | 11 |
| 12 | signed | 4 | 15,616 | 62,464 | 1,864 | 9 | 11 |
| 12 | signed | 5 | 12,224 | 61,120 | 3,728 | 17 | 11 |

The structural shift set has 11 operation-type relations, independent of `sigma`, `w`, and digit variant: one 12-row GT accumulator/table shift; five G1 and five G2 point-state shifts (`table<-table`, `double<-double`, `add<-double`, `add<-add`, `double<-add`). Selector edges use the fixed table-offset sets above and are not row-shift relations.

## Executable check

```bash
CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg RAYON_NUM_THREADS=1 RUST_MIN_STACK=67108864 cargo bench -p jolt-dory --bench deferred_check_counts -- --nocapture
```

The bench asserts both variants at `w in {3,4,5}` reproduce the Pippenger/plain GT result on the real proof, including the flattened final equation. It separately asserts the G1 2D and G2 4D schedules equal naive scalar multiplication on M1's real `3sigma+4` operation-term multisets.
