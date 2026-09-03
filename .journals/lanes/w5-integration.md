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
