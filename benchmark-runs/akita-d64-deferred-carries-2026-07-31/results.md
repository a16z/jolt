# D64 deferred-carry commitment results

Date: 2026-07-31 EDT
Machine: Apple M4 Max

## Result

Accepted in `6536b7af9`. Extending the deferred fp128 accumulator from D128 to
D64 reduced the adjacent `T = 2^22` root-accumulation span by **21.7%**. At
`T = 2^26`, two proofs averaged **45.383091 s**, **15.882733 s** of which was
the root accumulation kernel. Peak RSS averaged **33.8169 GiB**, and neither
run swapped.

Two fresh Dory controls averaged **92.544715 s**. The current comparison is
therefore **2.039189x Dory/Akita**. Both paired ratios independently exceed
2x, although the first does so by only 0.44 seconds against the half-Dory
threshold. The result meets the original 2x target with a modest, not large,
margin.

## Mechanism

The D64 rank-tiled root kernel previously used one
`WideCyclotomicRing<Fp128x8i32, 64>` per committed column. Every source
coefficient was expanded into eight signed 32-bit limbs, and each shifted
accumulation updated that representation.

The candidate uses the same `DeferredFp128Ring` introduced for D128:

```text
coefficient state = (low u64, high u64, signed i16 wrap count)
```

Addition and subtraction update the two native limbs and record only the
carry or borrow across bit 127. At the end of each 8,192-row tile, one
reduction per coefficient applies the accumulated correction.

This is not lazy field multiplication and it does not change any polynomial
value. It replaces the representation used by a prover-local commitment
accumulator.

### Correctness argument

Let `p = 2^128 - C`. The current `Prime128OffsetA7F7` preset uses
`C = 2^32 - 22537 = 0xFFFFA7F7`. For an accumulator coefficient, keep a
wrapped 128-bit value `r` and a signed wrap count `w`. The represented integer
is

```text
S = r + w * 2^128.
```

An overflowing addition subtracts `2^128` from `r` and increments `w`; a
borrowing subtraction adds `2^128` to `r` and decrements `w`. The invariant
therefore holds after every update. At the tile boundary,

```text
S = r + w * 2^128 = r + w * C  (mod p),
```

so reducing `r` and adding or subtracting `|w| * C` gives the canonical field
result. Negacyclic shifts only choose whether a source coefficient is added
or subtracted, so the same invariant covers the ring operation.

At most 8,192 contributions reach a coefficient before a flush. Thus
`|w| <= 8,192`, safely inside `i16`. The bound test runs the maximum-depth
positive and negative chains for both D64 and D128 and compares them with
canonical ring accumulation.

### Memory

The old D64 state was 32 bytes per coefficient (`8 * i32`). The new state is
18 bytes per coefficient (`2 * u64 + i16`), checked by a layout assertion in
the bound test.

For 29 committed columns in one rank task:

| State | Bytes per ring | Bytes per task |
|---|---:|---:|
| Old D64 wide accumulator | 2,048 | 59,392 |
| Deferred D64 accumulator | 1,152 | 33,408 |
| Difference | -896 | **-25,984** |

This state is task-local rather than trace-scaled. The change cannot explain
a large RSS movement, but it does not spend memory to obtain the CPU gain.

### Follow-on rank-batch screen

The compact state reopened one earlier question: batching two A ranks now
needs 66,816 bytes of destination state, versus 118,784 bytes with the old
D64 accumulator. A candidate scanned each decoded row and active-column mask
once for two ranks while leaving D128 at rank batch one.

It remained decisively slower:

| `T = 2^22` span | Deferred rank-one mean | Rank batch two | Change |
|---|---:|---:|---:|
| Root accumulation | 0.895740 s | 1.232115 s | **+37.6%** |
| Commitment | 1.162690 s | 1.502068 s | +29.2% |
| Whole prover | 4.741647 s | 5.202443 s | +9.7% |

Even with the smaller working set, interleaving writes to two destination
ranks costs more than sharing the row-mask traversal saves. The candidate
was reverted without a large run.

## Adjacent `T = 2^22` causal screen

The control and candidates differ only in the D64 accumulator. Candidate
means are from two runs.

| Span | D64 wide control | Deferred mean | Change |
|---|---:|---:|---:|
| Root accumulation | 1.143925 s | 0.895740 s | **-21.7%** |
| Commitment | 1.416222 s | 1.162690 s | **-17.9%** |
| Whole prover | 4.892803 s | 4.741647 s | -3.1% |

The two candidate root spans were 0.898764 s and 0.892715 s, a 0.67% spread.
Unrelated sumcheck spans were stable enough to localize the gain to the
modified kernel.

## `T = 2^26` validation

| Trial | Prover | Commitment | Root accumulation | Batched opening | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Akita 1 | 45.368753 s | 16.340658 s | 15.889985 s | 10.134431 s | 33.8208 GiB |
| Akita 2 | 45.397429 s | 16.333641 s | 15.875481 s | 10.102049 s | 33.8131 GiB |
| **Mean** | **45.383091 s** | **16.337150 s** | **15.882733 s** | **10.118240 s** | **33.8169 GiB** |

Both proofs verified and both `/usr/bin/time -l` reports recorded zero swaps.
The prover spread is 0.028676 s, or 0.063% of the mean.

There is no exact adjacent `T = 2^26` parent run. Two recent traces with the
same old D64 root kernel measured 22.959642 s and 22.813186 s in root
accumulation. The candidate mean is 30.4--30.8% lower. That comparison
corroborates scaling but is not used as the causal estimate; the adjacent
`T = 2^22` screen above is the controlled attribution.

## Fresh Dory comparison

The older retained Dory pair averaged 111.167123 s, but current-tree Dory is
materially faster. Two fresh controls were therefore run after the Akita
pair:

| Trial | Akita | Dory | Dory/Akita | Half-Dory margin |
|---|---:|---:|---:|---:|
| Pair 1 | 45.368753 s | 91.621822 s | 2.019492x | 0.442158 s |
| Pair 2 | 45.397429 s | 93.467609 s | 2.058874x | 1.336375 s |
| **Mean** | **45.383091 s** | **92.544715 s** | **2.039189x** | **0.889266 s** |

Dory peak RSS averaged 31.9431 GiB. Akita is currently 1.8739 GiB, or 5.9%,
higher. The ratio clears the fixed 2x objective in both pairs, but the small
first-pair margin makes further prover work useful rather than optional if
the target must remain robust across machine-state variance.

## Validation

Passed:

- all 49 `jolt-akita` tests;
- Akita natural, forced-K256, and committed-program `muldiv` tests;
- Dory `muldiv` in standard and ZK modes;
- scoped `jolt-akita` and legacy `host,akita` clippy;
- workspace clippy with `host`;
- workspace clippy with `host,zk`;
- `cargo fmt -q`.

No verifier, transcript, opening claim, or protocol code changed.

## Retained traces

| Trace | Purpose | SHA-256 |
|---|---|---|
| `akita_22_d64_deferred.json` | adjacent candidate A | `fb455bb6b07b1de82b0ce800af21b79ca7c5d16568979c6cbe54a2669c0ea200` |
| `akita_22_d64_deferred_repeat.json` | adjacent candidate B | `106af829b47e0ef272e0283c0cc98164b7be8bdeabd8bbf0dd20e6f916703343` |
| `akita_26_d64_deferred.json` | large candidate A | `8cbce08593d218929947e4fea62fa3adfbb743371340a0c53105cbcb572c599d` |
| `akita_26_d64_deferred_repeat.json` | large candidate B | `a8c5a8e38dccb9435668b7fedefd1e21c75ce1299a236cf9c88aedbe04637aee` |
| `dory_26_refresh.json` | fresh Dory control A | `fc5ad6c7454193e35d6dd2757f464a3e5cff64a672e44409150b3046e2f0700d` |
| `dory_26_refresh_repeat.json` | fresh Dory control B | `0eaa89d1614e3bcb0a97aac0b67c952cd5875af9ed2d15595a069f898972531b` |
| `akita_22_d64_rank_batch2.json` | rejected two-rank follow-on | `d8be83b7fafce925422274a45d8c1afa9b72323c88dc1b0a8139c51647b60ab5` |

All files are in `benchmark-runs/perfetto_traces/`.
