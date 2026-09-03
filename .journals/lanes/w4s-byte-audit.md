# W4-S byte audit — `2^17` G-shape gate

Date: 2026-09-02. Audited commit: `19872523d`. The release gate was instrumented in a detached
scratch worktree; no protocol source changed. Stable result across two release runs: payload
**10,304 B**, standard-bincode **10,445 B**. Timing used below is the quieter run: setup 6.600 s,
commit 1.013 s, post-commit proof 4.948 s, verify 0.002 s; commit + proof = **5.961 s**.

## Exact current proof

| item | count / origin | bytes |
|---|---|---:|
| packed commitments | 30 G1 | 960 |
| stage A rounds | `4 + 16*5 = 84 Fr` | 2,688 |
| stage B rounds | `40*2 = 80 Fr` | 2,560 |
| stage C rounds | `20*2 = 40 Fr` | 1,280 |
| stage-member outputs | 3 Fr | 96 |
| reduced claims | 5 Fr | 160 |
| HyperKZG | `19 G1 + 1 G1 + 60 Fr` | 2,560 |
| **payload** | `322 * 32` | **10,304** |
| bincode framing | G1 byte lengths + vector lengths | 141 |
| **standard bincode** | | **10,445** |

### 1. Packed commitments — 960 B

The fixture has 163 bit, 54 u16, 19 helper-Fr, and one witness-Fr column: 237 columns total at
`k = 8`. `commit_packed` groups the flat declaration order, including groups that cross a type
boundary:

```text
groups  0..19   160 bit columns                 20
group      20     3 bit + 5 u16                  1
groups 21..26    48 u16                          6
group      27     1 u16 + 7 Fr                   1
groups 28..29    13 Fr + 3 zero slots            2
                                                    --
                                                    30 G1 = 960 B
```

The group-index domain pads 30 to 32. The two missing groups are implicit zero polynomials: they
affect the eight-variable column point (`log2(32) + log2(8) = 8`) but emit no commitment.

Minimum among the named options: `k = 32` gives `ceil(237/32) = 8` commitments = **256 B**, saving
**704 B**. Its polynomial grows from `2^20` to `2^22`; measured HyperKZG open cost rises from
1.35 s to 5.85 s (**+4.50 s**). Total committed entries rise from `30*2^20` to `8*2^22` because
the last polynomial has zero slots; commit-time change is not measured.

### 2. Stage A — 2,688 B

Rows are `2^17`, hence 17 rounds. The declared degree-five envelope matches `s + 2` at LogUp
grouping `s = 3` (one eq factor plus the degree-four relation); the timing fixture realizes the
same envelope as a product of five multilinears. Compressed rounds omit the linear coefficient.
Instrumentation found:

```text
round 0:       degree 4 -> 4 Fr
rounds 1..16:  degree 5 -> 80 Fr
total                     84 Fr = 2,688 B
```

The first round's degree-five coefficient cancels for this fixture, so the actual proof is 32 B
below `17*5*32 = 2,720 B`.

KZG-committed round polynomials, under the requested wire convention, cost one G1 commitment and
one Fr evaluation per round, plus one two-G1 batched multi-point proof:

```text
17 * (1 G1 + 1 Fr) + 2 G1 = 1,152 B
```

Exact saving against this proof: **1,536 B**. Plan-v3's measured/estimated prover increment for the
batched round opening is about **0.05 s**. The verifier derives each round challenge from its KZG
commitment, then checks all claimed round evaluations in one multi-point KZG batch.

Degree three (`s = 1`) instead sends `17*3*32 = 1,632 B`, saving **1,056 B** here, but needs 54
extra full-width helper columns. At 72 ms per `2^17` helper column: **+3.888 s**. At `k = 8` those
columns add seven commitments (+224 B) and raise the padded column domain from 256 to 512, so stage
B adds five degree-two rounds (+320 B). Net proof change without committed rounds: **−512 B**.
After applying committed rounds, `s = 1` no longer shrinks stage A and instead adds 544 B, so the
two techniques should not be combined for byte minimization.

### 3. Stage B — 2,560 B

There are `D = 5` tensor factors. Each factor has an eight-variable column point: five group bits
for the padded 32-group domain and three slot bits for `k = 8`. The tensor coefficient eq term and
the active multilinear `T` make every round degree two:

```text
D * log2(padded_groups * k) * degree * 32
= 5 * 8 * 2 * 32 = 2,560 B
```

`k = 32` leaves the full column domain at `8 groups * 32 slots = 256`, so this line stays **2,560
B**. `s = 1` adds enough columns to pad the domain to 512 and raises it to **2,880 B**. No named
option reduces this line.

### 4. Stage C — 1,280 B

Each packed polynomial has `rows*k = 2^20` evaluations, hence 20 degree-two rounds for
`q_g(x) * P_g(x)`:

```text
20 * 2 Fr * 32 = 1,280 B
```

At `k = 32`, `ell = 22`, so this line becomes `22*2*32 = 1,408 B` (+128 B).

A direct two-point option deletes these 40 Fr plus C's one output claim (1,312 B), but a second
`ell = 20` HyperKZG point contributes **20 extra G1 + 60 extra Fr = 2,560 B**: 19 fold G1, one
opening-witness G1, and three 20-Fr evaluation vectors. Net proof change: **+1,248 B** and one
additional `2^20` open, about **+1.35 s**. A Shplonk-style verifier derives two point-specific
commitment RLCs and batches their final KZG equations into one pairing equation; it does not remove
the second point's Gemini fold data. This timing fixture has five B points, so the two-point figure
requires an earlier staging change and is not directly applicable to its five claims.

### 5. Stage-member outputs — 96 B

The proof sends one Fr for each singleton stage member: row relation A, column tensor B, and claim
reduction C. All three are recoverable here: A and C from `final_claim / batch_coefficient`, B from
the five reduced values and public tensor. A derived-singleton encoding can retain the
`stage_claims` field for heterogeneous batches while omitting these entries: minimum **0 B**,
saving **96 B**, with no prover-time change.

### 6. Reduced claims — 160 B

These are **five `T(s_i)` values, one per tensor factor**, not one value per packed polynomial or
group. Each value claims the eq-weighted combination of all 30 commitments at `(r_A, s_slot_i)`;
stage C binds those same five values to the final opening. `k = 32` therefore leaves this line at
**5 Fr = 160 B**. The suggested “reduced claims divided by four” does not apply to this proof.

### 7. HyperKZG — 2,560 B

For `ell = log2(2^17 * 8) = 20`:

```text
(ell - 1) fold G1 + 1 witness G1 + 3*ell Fr
= 20 G1 + 60 Fr = 2,560 B
```

A Zeromorph-style `ell + 1` G1 proof with no `3*ell` Fr vector is **21 G1 = 672 B**, saving
**1,888 B**. At `k = 32`, it is `23 G1 = 736 B`. No repo implementation has measured its time;
the budget tables below price it at the current opening's time class, not as a measured result.

### 8. External statement and bincode framing — 141 B framing

The 32-byte key digest, row-input Fr, dimensions, and tensor terms are verifier-known statement
data and add **0 proof bytes**. Standard bincode adds:

```text
50 one-byte G1 byte-string lengths                         50
commitments/stages/stage-claims/reduced/com/v Vec lengths 14
77 compressed-round coefficient-Vec lengths               77
                                                             ---
                                                             141 B
```

A fixed-shape protocol encoding removes these length tags: minimum **0 B framing**, no prover-time
change. The payload totals below assume that encoding; retaining serde/bincode adds format-specific
length bytes.

## Size frontiers

### Approximately 5-second class — 6,784 B payload

Measured baseline is 5.961 s including commitments, so no measured configuration is strictly at or
below 5.0 s. With the 0.05 s committed-round cost and Zeromorph priced at HyperKZG's current time
class:

```text
packed commitments, k=8                         960
stage A, KZG-committed rounds                  1,152
stage B                                         2,560
stage C                                         1,280
derived singleton stage outputs                    0
five reduced claims                               160
Zeromorph, ell=20                                 672
fixed-shape framing                                 0
                                                  -----
                                                  6,784 B
```

Levers: committed degree-five rounds (**−1,536 B, +0.05 s**), Zeromorph (**−1,888 B, unmeasured
time**), singleton-output elision (**−96 B, 0 s**), fixed-shape wire encoding (**−141 B from
bincode, 0 s**). Estimated measured-part prover time: **about 6.01 s**.

### Approximately 10-second class — 6,272 B payload

Add `k = 32` to the preceding set:

```text
packed commitments, k=32                        256
stage A, KZG-committed rounds                  1,152
stage B                                         2,560
stage C, ell=22                                 1,408
derived singleton stage outputs                    0
five reduced claims                               160
Zeromorph, ell=22                                 736
fixed-shape framing                                 0
                                                  -----
                                                  6,272 B
```

Measured opening interpolation adds 4.50 s; estimated measured-part prover time is **about 10.51
s**. If 10.0 s is a hard ceiling, use `s = 1` instead of `k = 32`: **7,328 B** at about **9.90 s**
under the same committed-round/Zeromorph/derived-output assumptions.

## Decision

- Byte-first, approximately 5–6 s: 6,784 B conditional on committed rounds and Zeromorph.
- Byte-first, approximately 10–11 s: 6,272 B by adding `k = 32`.
- Built and measured today: 10,304 B payload / 10,445 B bincode at 5.961 s commit + proof.
- Two-point opening and `s = 1` are not byte winners once committed rounds are selected.
