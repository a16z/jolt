# W5 integration

Date: 2026-09-02. Target: one T1 + T2 + Spartan/SPARK stream and one HyperKZG opening.

## Compiling milestone

`WrapPreparation::new` now performs the proof-independent front half against committed APIs:

1. derive and hash `WrapperProfile`;
2. build the 5,254-row verifier relation and generate its satisfying 6,761-variable assignment;
3. replay the real Jolt proof through `RecordingTranscript<Blake3Transcript>`;
4. build and verify the T1 schedule/table in the configured common row domain;
5. extract the relation's public column with the production challenge decoders;
6. embed the 6,715 private Spartan coordinates as the 13-round shared W column over `2^18` rows.

Cached fibonacci `2^18` measurement: 889 ms preparation, T1 219,784 used rows / `2^18`
padded rows, R1CS 5,254 constraints / 6,761 variables, public column 7 verifier-computed values +
38 canonical challenge preimages.

The committed relation exports 45 public entries, split 7+38 for this profile:
`log_k_ram=13`, `log_k_bytecode=12`, six bytecode gammas, and seven register coordinates.
The proof wire transmits the 38 canonical 16-byte challenge preimages: **608 B**. The seven known
field elements are supplied by the external statement and are not serialized in `WrapperProof`.

## Interfaces still blocking `wrap` / `verify_wrapped`

### Fiat–Shamir dependency

The requested stage order has two Fiat–Shamir dependencies:

```text
commitments
  -> Spartan outer rounds
  -> rx, Az(rx), Bz(rx), Cz(rx)
  -> inner linear form + input claim
  -> Spartan inner rounds
  -> ry
  -> SPARK matrix-memory members
```

Spartan outer can share stage A with T1/T2 because it has no dependency on their final point. The
inner prover can only start after A fixes `rx`, so it starts stage S and fixes `ry` after 13 fresh
degree-2 rounds. A SPARK matrix member contains `eq(ry, col(k))`; it cannot emit any round
polynomial until all of `ry` is known. Starting it in the same 13-round batch lets the prover choose
its round polynomial after seeing the matching evaluation point. Offsetting SPARK after the inner
member is sound, but is 13 + 16 sequential rounds rather than a 16-round batch. `prove_stage`
absorbs every member's input claim and draws every batching coefficient before round zero, while
`prove_batch` calls `finish_rounds` only after the batch's maximum round. An offset cannot expose
the inner member's final bind or construct SPARK midway through the batch.

The sound order supported by the current sumcheck API is therefore: (A) T1/T2 + Spartan outer;
(S) Spartan inner; (P) SPARK after `ry`; (R) point reduction; (B) column batching; one opening.
Stages A, S, and P end at different points (`r_A`, `ry`, and `r_P`). Packing T1/T2, W, and SPARK
columns into one polynomial does not make these points equal. A single final opening needs an
explicit eq-weighted point-reduction member for each earlier claim; no such reduction was specified
or implemented. Without R, the protocol needs separate openings.

### Stream assembly

The stage primitive itself is already generic: `prove_stage(&mut [StageMember], transcript)` accepts
heterogeneous member counts, degrees, round counts, and offsets; `verify_stage_with` runs a
verifier-owned final-claim callback. No new stage-builder type is needed. `stream::prove_stream`
remains a two-stage synthetic driver. The full orchestration still needs to construct these members
at their dependency boundaries:

- T1: degree-3 row relation plus 64 wired bits / 3 wired words;
- T2: degree-5 row relation plus operand limbs and its degree-2 wiring member;
- Spartan outer, then its data-dependent inner member;
- SPARK row/column/value and LogUp members.

Reusing the current tensor-only `TensorStreamStatement` would omit T1/T2 wiring and point-reduction
claims.

### T1 links

The committed `HashTableProver` proves the row relation and returns virtual wired evaluations, but
`HashTableProver::new` consumes `HashTable`, and no committed adapter turns `MessageLink`,
`ChallengeLink`, and `RowFeeds` into the succinct verifier kernels plus stage members. Packing its
163 bit columns today requires cloning about 43 MB before the prover consumes the table. Needed
interface: borrowed/shared committed columns and a `HashTableLinks` prover/verifier pair returning
the linked R1CS/T2 claims at the common point.

### T2 adapter

The committed limb table exposes `FlattenedCheck`, `DoryWitnessInputs`, `schedule::build`,
`columns::Columns`, `RowSumcheck`, and `WiringSumcheck`, but no constructor maps the real Jolt
objects into those inputs. Needed interface:

```text
from_jolt(
  preprocessing.pcs_setup,
  proof.commitments,
  proof.joint_opening_proof,
  relation.link.dory,
  relation_witness.values,
) -> { DorySetupInputs, FlattenedCheck, DoryWitnessInputs }
```

It must own the final-opening commitment order. The padded-cell layout and committed one-hot digit
columns are still uncommitted; consuming their working-tree types would make this milestone
non-reproducible.

### SPARK

No committed key-time row/column/value/multiplicity tables or LogUp prover/verifier exist. The
current Spartan verifier can evaluate matrix MLEs in `O(nnz)` and remains the named native fallback,
but it is a standalone transcript with a standalone W opening. Treating that component proof as the
full wrapper would leave T1/T2 and their links unbound, so it is measured only as an R component.

The native fallback can replace SPARK's matrix-memory argument only after A and S fix `rx` and
`ry`: it computes the same matrix MLEs from the verifier key in `O(nnz)` and adds no proof bytes,
but it does not remove the outer→inner dependency above.

## Measured native-R component

The cached fibonacci `2^18` relation was padded to 8,192 rows and 8,192 private witness entries,
proved with the current standalone Spartan/HyperKZG path, and verified:

| item | bytes |
|---|---:|
| 38 canonical challenge words | 608 |
| W commitment | 32 |
| outer: 13 degree-3 rounds | 1,248 |
| inner: 13 degree-2 rounds | 832 |
| two stage claims | 64 |
| `Az(rx), Bz(rx), Cz(rx), W(ry)` | 128 |
| standalone HyperKZG opening | 1,280 |
| **payload** | **4,192** |
| **bincode** | **4,246** |

Preparation took 559 ms, proving 29 ms, and verification 4 ms. Host load was 9.07 immediately
before the run after another 167 s ten-thread performance gate, so timings include contention. This
is an R-only measurement; it does not bind T1, T2, or SPARK and is not a wrapped proof.

## Size projection before SPARK implementation

The measured synthetic k=16 stream is 4,960 B. With the requested stages added literally:

| item | bytes |
|---|---:|
| current k=16 stream | 4,960 |
| Spartan inner stage S, 13 degree-2 rounds | 832 |
| SPARK stage P, 16 degree-3 rounds | 1,536 |
| one new packed commitment + two residual claims | 96 |
| 38 canonical challenge words | 608 |
| **projection before point reduction** | **8,032** |

This is **2,032 B over the 6,000 B target** (1,888 B over 6 KiB) before the missing point-reduction
proof and T1/T2 link claims. A minimal common-domain point reduction adds another 18 degree-2 rounds
(1,152 B), so the sound full-statement lower bound is **9,184 B** before its final-value claims and
the link claims. The failing item is the sequential `rx -> ry -> SPARK` round-polynomial block:
SPARK cannot consume the same challenges as the inner sumcheck under the current protocol.

## Current result

- Real proof preparation: **passes**.
- Full wrapped proof / verification: **blocked before stage assembly**; no unsound partial proof.
- Full payload/prover/gas: not measurable until the T1/T2 adapters, SPARK protocol, and point
  reduction exist.
- The exact stage primitive needed for heterogeneous batching already exists; no duplicate builder
  was added.

## 23:50 fallback baseline and carry-in-A

Implemented the missing claim transport as `CarryProver`:

```text
input  = f(r_old)
check  = sum_x eq(r_old, x) f(x)
output = f(r_A)
final  = eq(r_old, r_A) f(r_A)
```

All carries enter the existing 18-round stage A as degree-2 members. The generalized
`prove_kzg_batch_stage` / `verify_kzg_batch_stage` path batches them with a degree-5 row member,
retaining the 96 B/round KZG wire and adding no rounds. Each carry needs its new `f(r_A)` claim;
the prior-point value is already an output claim of its owning earlier stage.

The fallback SPARK relation has 2^16 entries for the real 5,254-row relation and 6,715 private
columns. Its VK fixes row, column, A/B/C value, and row/column multiplicity columns; the prover
commits `E_row`, `E_col`, and four inverse columns. One degree-3 sumcheck checks the matrix MLE,
four pointwise inverse identities under a random eq weight, and two LogUp multiset sums. Fixed and
dynamic partial commitments add to the final packed commitment. Table-evaluation and inverse
tamper tests reject.

Real fibonacci `2^18`, k=16 measurement (load 3.47 at start; the test's setup raised it to 15):

| phase | ms | proof bytes |
|---|---:|---:|
| SPARK table build (VK) | 3 | 0 |
| deterministic test SRS (not prover) | 62,027 | 0 |
| fixed-table VK commitment | 172 | 0 |
| SPARK witness | 615 | 0 |
| dynamic witness commitment | 1,457 | 32 |
| SPARK sumcheck, 16 degree-3 rounds | 60 | 1,536 |
| SPARK prior-point claims | — | 416 |
| 14 carries in stage A | 170 | 448 new `r_A` claims |
| **fallback SPARK + carry online delta** | **2,302** | **2,432** |

The measured k=16 G-shape core at 2^18 is 5,184 B. Adding 13 SPARK columns crosses 16 packed
groups, adding one commitment (32 B) and one stage-B round (64 B), for a 5,280 B core. Full
fallback projection:

| item | bytes |
|---|---:|
| k=16 core with 267 columns, A/B/opening | 5,280 |
| stage 0 Spartan outer | 1,248 |
| stage 1 Spartan inner | 832 |
| stage 2 SPARK | 1,536 |
| outer `Az/Bz/Cz` | 96 |
| W + 13 SPARK prior-point claims | 448 |
| 14 carry outputs at `r_A` | 448 |
| canonical public challenge words | 608 |
| **projected fallback payload** | **10,496** |

Gap: **+4,496 B over 6,000 B**. The largest individual item is the 2,144 B HyperKZG opening;
the largest removable block is the 3,616 B sequential outer + inner + SPARK round wire. The core
verifier count remains 100 ecMul / 99 ecAdd / 8 pairing pairs / 7,133 Fr mul / 281 Keccak before
the fallback stages; their field/hash additions are not yet observer-instrumented, so no exact
combined gas claim is made.

## R-slot change

The 23:40 decision replaces the fallback with W6-RT's R row table in stage A, deleting outer,
inner, SPARK, W, and the 608 B public challenge block. W5 assembly will consume one pluggable R
slot with:

1. VK packed-group commitments and column descriptors;
2. prover columns in the common `2^18` row domain;
3. owned `ProveRounds` members plus degree/offset/input-claim metadata;
4. verifier final-claim evaluation from `r_A` and stage-B column evaluations;
5. link members and the column indices they expose to stage B.

`.journals/lanes/w6-relation-table.md` is not present yet, so no speculative Rust trait was added.
The existing Spartan/SPARK path remains a measured fallback. L4 stays after the first assembled
e2e baseline.

## 00:29 generic assembly and L4 checkpoint

Commits `8536caf1e` and `4cfd41f2a` add the generic assembly core plus the named `wrap` /
`verify_wrapped` entry points. The caller supplies the packed columns, arbitrary stage-A members,
their verifier-owned final checks, and the stage-B factor-column set. The engine commits the
columns, proves one KZG-batched stage A, reduces every exposed column through stage B, and opens
one packed RLC. `tests/assembly.rs` verifies the round trip and rejects independent changes to a
packed commitment, stage-A KZG data, a member-final claim, stage B, the reduced claim, and the
HyperKZG opening.

This is an executable synthetic-honest assembly seam, not the real fibonacci wrapper yet. The T2
`from_jolt` input adapter is now committed, but it does not yet export its `Column` list,
`StageMember`s, or final-check evaluator. T1 still lacks the same stream adapter. W6-RT adds a
second requirement: its `a,b,c` commitments precede `(beta,gamma)`, while `h_id,h_sigma`
commitments follow them. The current one-phase `wrap` seeds all commitments together. The next
assembly API must let an adapter build and absorb a challenge-dependent commitment phase before
stage A; flattening W6-RT into the current call would be unsound.

L4 is isolated on `wrap/l4-typed` at `dc1e9ff42`: `PackedPolynomial::{Bits,U16,Fr}` replaces the
unconditional `Vec<Fr>` copy per packed group. Pure bit groups use conditional additions for row
evaluation and RLC; pure u16 groups convert only in the active arithmetic loop; mixed groups keep
the field path. Focused assembly/stream/SPARK tests pass, as do debug all-target and release-lib
clippy. It remains isolated because W6-RT's uncommitted protocol constructs `PackedColumns`
directly and must switch `evaluations` to `polynomials` with the same commit.

W4-R review #2's production diagnostic is fixed by `c0592f75c` + `9d9c344c9`: `native::check`,
`NativeParity`, and the witness field exist only for debug/test builds. Release witness generation
still executes the native verifier once to obtain the transcript schedule; it no longer performs
the second parity reconstruction. Debug relation tests, debug all-target clippy, and release-lib
clippy pass.

L4 release timings use the PERF-1 shape at `rows=2^18`, `k=8`: 180 bit columns, 54 u16 columns,
20 Fr columns, 32 packed groups. The prior dense profiler measured column evaluations at 0.168 s
and RLC at 0.315 s. One initial typed run (load 5.29) measured 0.210/0.158 s; replacing u16 field
multiplication with `Fr::mul_u64` then measured **0.121/0.148 s** at load 8.29. Combined time is
0.483 -> 0.269 s (-0.214 s, 44%). Packed coefficient storage is exactly 2,048 -> 324 MiB
(-1,724 MiB); this excludes the caller-owned source columns and SRS. The 0.05/0.08 s estimate was
not reached; bit-column field additions now dominate the evaluation pass.

## 00:52 term-stage export contract

Published in `stream::types`:

```rust
pub struct ColumnId { pub group: usize, pub slot: usize }
pub struct AffineForm { pub constant: Fr, pub weights: Vec<(ColumnId, Fr)> }
pub struct Term { pub coefficient: Fr, pub factors: Vec<AffineForm> }
pub struct TermContext<'a> {
    pub row_point: &'a [Fr],
    pub batching_coefficients: &'a [Fr],
    pub challenges: &'a [Fr],
}
pub trait TermExporter {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term>;
}
```

`ColumnId` is the physical `(packed group, slot)` so VK and prover phases share one namespace.
Each table exporter captures its typed verifier state; `TermContext::challenges` is the canonical
Fiat-Shamir vector for member-specific α/β/γ/kernel values. Exporters return products of at most
five affine factors. The assembly pads shorter products with the constant-one form and pads the
term table to a power of two with zero-coefficient terms. Prover and verifier call the same export
method after stage A, using the same row point and member batching coefficients.

## 01:04 compressed-claim stream gate

Commits `5ca9bfa52` and `d5873f4fc` replace assembly's per-column claims with a term-index
sumcheck and one weighted stage-B reduction. Stage A defers member final checks to the shared
`TermExporter` list. The term stage pads `T` to a power of two, commits degree-six rounds, sends
the five final factor evaluations, and derives one column weight vector. Stage B proves that
weighted column functional and the existing HyperKZG proof opens the resulting packed polynomial.

Committed stages now send one G1 commitment and one next-claim scalar per round. One aggregated
`S(0)` scalar plus the three-G1 variable-batch proof checks all Boolean-sum identities. The
assembly test rejects changes to either `S(0)`, a term-round commitment, a final factor
evaluation, and a phase-2 commitment.

Commitment phases are statement-owned `(group_count, challenge_count)` entries. The caller uses
`commitment_prefix_challenges` after committing each available prefix, constructs the next helper
phase, and joins aligned packed phases with `combine_packed_phases`. Verification replays the same
ordered absorption before stage A. Non-final phases must end on a packed-group boundary.

One release gate at `rows=2^18`, `k=16`, `T=600`, five factors, 32 synthetic stand-in columns:

| item | bytes |
|---|---:|
| two phase commitments | 64 |
| stage A: 18 × 64 + `S(0)` + BDFG/shift | 1,280 |
| term stage: 10 × 64 + `S(0)` + BDFG/shift | 768 |
| five final factor evaluations | 160 |
| stage B: five degree-two rounds | 320 |
| reduced claim | 32 |
| HyperKZG opening | 2,144 |
| proof payload | **4,768** |
| canonical IO wire | 608 |
| **total** | **5,376** |

Exact bincode size is 4,850 B. Load was 7.01/9.37/9.53. Deterministic test setup 3,166 ms;
wire commit 41 ms; helper commit 41 ms; assembly prove 4,638 ms; verify 10 ms. Executed verifier:
152 ecMul, 151 ecAdd, 12 pairing pairs, 18,825 Fr mul, 8 inversions, 306 Keccak; N4 estimate
2,128,781 gas including 608 B IO. The extra 56 Fr multiplications are the now-observed term
export (54 for `eq`, two coefficient products); every production exporter must implement the
object-safe `TermObserver` path, so its verifier arithmetic cannot bypass the counter.

The committed R exporter contributes exactly 26 terms; T1/T2 exporters are still uncommitted, so
600 is the requested stand-in count. Up to 1,024 total terms keeps ten term rounds and the same
768 B term stage. Replacing the gate's two groups with 17 groups for approximately 270 real
columns changes commitments 64 → 544 B and stage B five → nine rounds (320 → 576 B): projected
full total **6,112 B**, 112 B over the decimal 6,000 B target (32 B below 6 KiB). The 2,144 B
HyperKZG opening is the largest fixed item.

Command:

```bash
cargo nextest run -p jolt-wrapper --release --test assembly_term_gate term_compression_gate \
  --run-ignored ignored-only --cargo-quiet --no-capture
```

## 01:28 shared committed-round opening

The assembly now delays stage-A and term-stage round openings until both stages have emitted all
round commitments and next claims. All aggregation coefficients are then drawn together. Each
stage keeps its own `S(0)` scalar; one variable-batch proof supplies the shared `W`, `W'`, and
degree-shift commitment. Delta: **−96 B** and **−4 pairing pairs**.

The BDFG verifier evaluates complement vanishing products directly at its challenge rather than
building each polynomial. The term verifier materializes `eq(t*, ·)` once for its coefficient and
five factor reductions. Measured Fr work falls 36,936 → **7,945 multiplications**.

One release gate at `rows=2^18`, `k=16`, `T=600`, five factors:

| item | bytes |
|---|---:|
| two phase commitments | 64 |
| stage A: 18 × 64 + `S_A(0)` | 1,184 |
| term stage: 10 × 64 + `S_T(0)` | 672 |
| shared BDFG/shift | 96 |
| five final factor evaluations | 160 |
| stage B | 320 |
| reduced claim | 32 |
| HyperKZG opening | 2,144 |
| **proof payload** | **4,672** |
| temporary public challenge IO | 608 |
| **baseline total** | **5,280** |

Exact bincode: 4,754 B. Executed verifier: 150 ecMul, 149 ecAdd, 8 pairing pairs, 7,945 Fr
mul, 8 inversions, 300 Keccak; **1,892,109 N4 gas**. Contended timing (load
8.95/9.16/10.28): setup 5,947 ms, phase commits 85 + 91 ms, prove 10,805 ms, verify 64 ms.

At 270 columns the current projection is **5,408 B**. T1's CopyLink-bound challenge outputs remove
the temporary 608 B public block, leaving **0 B challenge IO** and 592 B headroom against decimal
6,000 B. T1 contributes 230 terms; R contributes 26; T2 remains pending.

## T1 stream adapter

`hash_table::StreamColumns` exports typed groups without materializing its 14 u32 word columns as
field elements: 15 bit groups, one u32 group, one VK-bit group, one VK-u16 group at k=16.
`hash_table::StreamTermExporter` maps T1's 230 local terms to physical `(group, slot)` IDs and
binds its two stage-A member coefficients. The adapter is ready on top of committed T1
`c4a218b14`; the real assembly waits for T2's phase/helper/term export.

## Real T1 + relation-table gate after T1 review fixes (02:24)

- `WrapVerifierKey` owns the profile-fixed T1 `SymbolicSchedule`, its `LinkMap`, the assembly
  statement, and pinned T1/R/link VK commitments. Verification does not derive them from proof
  bytes.
- Phase 1 contains canonical T1 columns, R fixed/wire columns, and the T1↔R challenge-link VK
  columns. It is absorbed before T1's 38 randomizers and both R/CopyLink `(beta, gamma)` pairs.
  Phase 2 contains R/CopyLink inverse helpers plus the T2 stand-in, then supplies the remaining
  `tau` and relation weights.
- Real cached fibonacci `2^18`, k=16: T1's two members, R row member, the 376-item T1↔R challenge
  `CopyLink`, and a synthetic-honest T2 member prove and verify together. **T=322** = T1 296 + R
  15 + CopyLink 10 + stand-in 1; nine term rounds; maximum term degree 4.
- Payload **5,600 B** / bincode **5,709 B**: phase-1 commitments 800; phase-2 32; stage A 1,184;
  term stage 608; shared BDFG/shift 96; four factor evaluations 128; stage B 576; reduced claim 32;
  HyperKZG 2,144; challenge-word IO 0. The seven known public fields are 224 external calldata
  bytes. Margin below decimal 6 KB: **400 B** before replacing the T2 stand-in.
- One release run at load 9.43/6.40/6.22: prepare 531 ms; SRS setup 3,164; adapters 446; phase-1
  commit 2,451; helpers 319; phase-2 commit 198; proof 6,332; verify 12. Executed verifier:
  **170 ecMul, 169 ecAdd, 8 pairing pairs, 12,643 Fr mul, 8 inversions, 449 Keccak; 2,175,449 N4
  gas**. T1 term construction now routes every multiplication through the observer.
- Tampering a T1 commitment, phase-2 commitment, stage-A aggregate, term-round commitment, factor
  evaluation, reduced claim, or HyperKZG value is rejected. T2 remains a zero-column stand-in; the
  missing real phase columns, two members, term exporter, and T2↔R scalar link gate full soundness.
