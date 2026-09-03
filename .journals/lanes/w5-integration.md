# W5 integration

Date: 2026-09-02. Target: one T1 + T2 + Spartan/SPARK stream and one HyperKZG opening.

## Compiling milestone

`WrapPreparation::new` now performs the proof-independent front half against committed APIs:

1. derive and hash `WrapperProfile`;
2. build the 5,253-row verifier relation and generate its satisfying 6,760-variable assignment;
3. replay the real Jolt proof through `RecordingTranscript<Blake3Transcript>`;
4. build and verify the T1 schedule/table in the configured common row domain;
5. extract the relation's public column with the production challenge decoders;
6. embed the 6,714 private Spartan coordinates as the 13-round shared W column over `2^18` rows.

Cached fibonacci `2^18` measurement: 889 ms preparation, T1 219,784 used rows / `2^18`
padded rows, R1CS 5,253 constraints / 6,760 variables, public column 7 verifier-computed values +
38 canonical challenge preimages.

The committed relation exports 45 public entries, but they split 7+38 for this profile:
`log_k_ram=13`, `log_k_bytecode=12`, six bytecode gammas, and seven register coordinates.
The requested 7+28 split is not representable by the committed `PublicLayout`; ten coordinates
must be derived or removed before the proof-size target can use 28 words.

## Interfaces still blocking `wrap` / `verify_wrapped`

### Stream assembly

`stream::prove_stream` accepts one row prover plus a fixed equal-arity tensor over committed column
indices. The full relation needs heterogeneous row members and virtual-column links:

- T1: degree-3 row relation plus 64 wired bits / 3 wired words;
- T2: degree-5 row relation plus operand limbs and its degree-2 wiring member;
- Spartan outer, then its data-dependent inner member;
- SPARK row/column/value and LogUp members.

Needed committed interface: a stage builder accepting arbitrary `StageMember`s, verifier-owned final
claim callbacks, and column-opening claims that may come from virtual link members. Reusing the
current tensor-only `TensorStreamStatement` would omit T1/T2 wiring claims.

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
full wrapper would leave T1/T2 and their links unbound, so no such proof is emitted.

## Current result

- Real proof preparation: **passes**.
- Full wrapped proof / verification: **blocked before stage assembly**; no unsound partial proof.
- Payload/bincode/prover phase/gas: not measurable until the stream/T1/T2/SPARK interfaces above
  exist.
- W co-pointing remains 0 extra rounds and projects a 1,312 B saving once the shared proof merger
  removes its standalone commitment/opening.
