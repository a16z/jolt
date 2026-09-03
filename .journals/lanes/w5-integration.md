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
