# W5 assembly review #3

Target: `0789e5a14` (`wrap/spartan-hyperkzg`). Scope: wrapper assembly,
R1CS/Spartan, native matrix evaluation, links, keys, cost accounting, and the
real fixture gate. T1/T2 internals were treated as reviewed dependencies except
at their assembly boundaries. Review performed in detached
`/Volumes/Dev/worktrees/jolt/w5-review3`.

## Verdict

**1 blocker / 0 majors / 4 minors.** Spartan and the W carry are sound, but the
cross-table batching challenges are sampled before T2 commits the values they
are meant to bind.

## Findings

### Blocker 1 — T2 chooses linked values after seeing the link challenges

**Files:**

- `crates/jolt-wrapper/src/wrap/key.rs:98-112`
- `crates/jolt-wrapper/src/limb_table/stream.rs:341-368`
- `crates/jolt-wrapper/src/limb_table/export.rs:16-30`

Phase 1a draws all ten CopyLink `(beta, gamma)` pairs and scalar-link `rho`:

```text
phase 1a: commit T1 + W           -> beta/gamma[0..10], T1 randomizers, theta, rho
phase 1b: commit T2 chunks/D/sign -> xi, alpha
```

Eight CopyLinks bind T1 element bytes to T2 phase-1b chunks/signs. The scalar
link binds W to T2's phase-1b digit values. The right side of each link is
therefore chosen after its batching challenge.

For the scalar link, after seeing `rho`, choose two T2 scalar values with
`delta_i = a * weight_j(rho)` and `delta_j = -a * weight_i(rho)`. Their weighted
sum is unchanged, while both individual values differ. T2's unique-recoding
checks pin the digit columns to this altered vector; they do not recover the W
vector. The random-linear-combination argument has no collision bound when one
vector is adaptive. The element CopyLinks have the same commit-order defect for
their inverse-sum identity.

**Fix:** phase 1a should draw only T1's 38 randomizers and `theta`. After the T2
phase-1b commitment, draw all CopyLink `(beta, gamma)` pairs and scalar-link
`rho`, along with `xi, alpha`. Update the key-owned offsets and the real-gate
challenge extraction. Helper commitments already occur later, so this reordering
does not add proof bytes.

Regression: `.journals/lanes/w5-review-3-tests.patch`. On the target tree it
fails before proving: phase 1a exposes 60 challenges; the safe prefix is 39.

### Minor 1 — the shared opening admits degree-five Spartan outer rounds

**Files:**

- `crates/jolt-wrapper/src/stream/shared_rounds.rs:179-249`
- `crates/jolt-wrapper/src/stream/shared_rounds.rs:373-446`

The shared BDFG opening selects the maximum stage degree, five, for the Spartan
outer, stage A, and the term stage. This is a valid common bound, but it does not
enforce the outer stage's declared degree three. A malicious outer round may be
degree four or five; the protocol remains statistically sound using a `5/|Fr|`
per-round bound instead of `3/|Fr|`.

**Fix:** either account for the outer rounds as degree five, or give the outer
commitments a degree-three shift check while retaining one multi-opening.

### Minor 2 — the native-matrix subcount is not the observer count and misses an obvious trim

**Files:**

- `crates/jolt-wrapper/src/spartan.rs:231-307`
- `crates/jolt-wrapper/src/stream/term_stage.rs:415-428`
- `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:613-633`

The reported `136,946` is the test formula `2 * public_nnz + 4 * private_nnz`,
not an observed delta. With 1,417 public and 33,528 private nonzeros, the code
executes:

```text
public matrix terms       2 * 1,417       =   2,834
private matrix terms      3 * 33,528      = 100,584
two eq tables             4 * (8,192 - 1) =  32,764
                                               -----
                                             136,182
claim/final composition                         +4
```

The total `VerifierCost.fr_mul = 179,547` is execution-counted and reproduced;
only the matrix attribution is wrong by 760–764 multiplications, depending on
whether the four composition products are included.

**Fix:** record a verifier-cost delta around native matrix evaluation. For the
hot path, form each eq-table sibling as `parent - first_child` (one multiply per
parent), accumulate A/B/C separately with two multiplies per nonzero, then apply
the three matrix weights once. This cuts this block to about 86.3k
multiplications.

### Minor 3 — restored Spartan/carry internals remain public or dead

**Files:**

- `crates/jolt-wrapper/src/lib.rs:6-13`
- `crates/jolt-wrapper/src/carry.rs:13-131`
- `crates/jolt-wrapper/src/spartan.rs:55-119,310-317`
- `crates/jolt-wrapper/src/stream.rs:37-41`
- `crates/jolt-wrapper/src/stream/protocol.rs:63-85`

The old standalone `prove_spartan`/`verify_spartan` API is gone, but the restored
modules export wrapper internals. `carried_final`, `SharedWitnessColumn::inner_member`,
and `SharedWitnessColumn::into_column` have no caller. `matrix_nnz` is used only
by the real report. `SpartanAssembly` and `prove_spartan_assembly` are public but
called only by `wrap` inside this crate.

**Fix:** keep `carry` and `spartan` private, expose only the error type needed by
`WrapError`, delete the unused methods/function, and make the Spartan assembly
entry point crate-private. The real test can count nonzeros locally.

### Minor 4 — the claimed program-preamble tamper is absent

**File:** `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:195-198,374-410,569-573`

The gate checks a T2-layout key, one of the seven public values, and a modified
profile. The value named `program_hash_key` is used only with the modified
`bytecode_ra_commitments` profile; no test changes the 54-byte T1 initial
state/tail that transitively binds the program/IO preamble.

**Fix:** build a key with one `hash_public.state_in` word changed and require the
original wrapper proof to fail. The attached patch adds this negative; it
compiled in the second fixture run but did not execute because the new
commit-order assertion fails earlier.

## Spartan and matrix checks

- R is exactly 5,254 constraints / 6,761 variables. `z = [1, seven public,
  6,753 W]`; the verifier inserts the constant one and seven key-owned statement
  fields. No other relation variable is public.
- `tau` is sampled after every column commitment. The outer stage has 13
  committed rounds; every commitment precedes its challenge and next claim.
  `Az(rx), Bz(rx), Cz(rx)` are absorbed before the three matrix weights.
- The inner claim subtracts verifier-computed public-column contributions, then
  checks a random A/B/C combination against 13 clear degree-two rounds. Its
  final value is the verifier's native matrix MLE times `W(ry)`.
- The common W column stores the 6,753 private variables at rows `0..6,753`.
  The honest suffix is zero. Spartan needs only the first 8,192 rows; arbitrary
  committed padding cannot affect the R1CS sum because its projected matrix
  coefficients are zero there.
- The 5,254 matrix rows are extended to 8,192 with zero `Az/Bz/Cz` entries.
  Verifier matrix loops read only the native rows, so padded constraints cannot
  inject a matrix coefficient.
- The 18-round carry starts at `[0; 5] || ry` and ends at stage A's point. Its
  term is `eq([0;5] || ry, r_A) * W(r_A)`, so stage A, term reduction, and the
  final HyperKZG opening consume the same W claim.
- Matrix MLEs use the key's sparse A/B/C matrices. No proof field supplies a
  matrix entry.

## Link ledger

| Inner-verifier value | Count | Binding | Result |
|---|---:|---|---|
| T1 squeeze outputs | 376 | one CopyLink into private W cells; up to three occurrence slots per W row | complete, subject to Blocker 1 only for link challenge ordering |
| Fr absorbs before the final squeeze | 1,199 | aligned/shifted T1 word forms into private W cells | complete |
| Fr absorbs after the final squeeze | 23 | R variables: 22 opening coordinates plus the evaluation | no later inner challenge; point-derived/evaluation Dory scalars enter T2 |
| Dory elements | 45,152 bytes / 1,526 T2 rows | eight CopyLinks, including compressed signs and zero infinity bits | values complete; challenge order unsound per Blocker 1 |
| Dory scalars | 173 ordered entries | occurrence weights accumulated at each private W row, then subtracted from T2's digit-link member | values complete; `rho` order unsound per Blocker 1 |
| Statement fields | 7 Fr | Spartan public segment after the constant one | complete |
| T1 initial state/tail | 54 bytes / 4 Fr | wrapper statement suffix | complete |

`Chi(sigma) = 1` is R1CS-pinned but not exported. `S1Acc` and `S2Acc` remain
internal; T2 consumes their derived `Ht` and pairing coefficients. The 173-entry
order excludes all three by construction and matches `FlattenedCheck::wires()`.

## Tamper matrix and cleanup

- The real tamper loop mutates every nonempty serialized vector: all wire
  commitments, outer/A/term commitments and claims, inner/B coefficients,
  shared BDFG fields, factor evaluations, all five reduced claims, and every
  HyperKZG group/scalar field.
- Witness negatives cover W, an absorbed-Fr row, T2 window and input rows,
  sign, psi, digit, T2 VK pins, and one of the seven uniformly handled statement
  fields. Profile mismatch is covered; the program-preamble negative is Minor 4.
- The Plonkish row table, SPARK, row selectors/permutation witness, and old
  standalone Spartan proof API are deleted. Scoped source and real-test files
  are below 1,000 lines. No `#[allow]` remains; reviewed imports and `#[expect]`
  sites follow repository policy. Minor 3 is the remaining public-surface work.

## Numbers and verification

- Baseline real k=32 gate: passed in 102.61 s test time.
- Payload / bincode / statement: 7,488 / 7,628 / 352 bytes.
- Geometry: T=510; nine term rounds; eleven stage-B rounds.
- Cost: 234 ecMul / 233 ecAdd / 8 pairing pairs / 179,547 Fr mul / 10 Fr inv /
  857 Keccak; gas model 6,082,065.
- Schedule regression: failed as intended in 24.88 s (`phase 1a challenges:
  left 60, right 39`). No broad suite, clippy, or fmt run; the requested real
  gate consumed both permitted fixture runs.
- Hunk had no active review session. No GitHub comments were posted.
