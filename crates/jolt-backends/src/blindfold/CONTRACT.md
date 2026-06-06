# BlindFold Request Contract

## Constructed By

`jolt-prover` ZK orchestration after all committed sumcheck and opening
material is available.

## Slots

`BlindFoldSlot` identifies committed round rows. `BackendValueSlot` identifies
private openings and output-claim values.

Requests must be structurally valid before a backend attempts execution:

- labels are non-empty;
- the request includes at least one committed round or output-claim row;
- committed round slots are unique;
- each committed round names at least one coefficient slot;
- coefficient slots are unique within a round;
- output-claim slots are unique.

## Witness Views

BlindFold requests consume private sumcheck/opening material already produced
by earlier backend requests.

`BlindFoldRowCommitmentRequest` is the transcript-free backend hook for the
active critical path. It carries already-assembled private rows plus their
blindings and asks the backend only for vector commitments. The caller remains
responsible for transcript scheduling, protocol ordering, Fiat-Shamir
challenges, final proof object construction, and verifier-visible proof data.

Row commitment requests must be structurally valid before execution:

- labels are non-empty;
- the row count equals the blinding count;
- every row fits the vector-commitment setup capacity.

`BlindFoldRowOpeningRequest` is the matching hook for opening committed row
matrices at caller-supplied row/entry points. It carries rectangular private
rows, row blindings, and the two evaluation points. The caller remains
responsible for when the opening is appended to the proof flow.

Row opening requests must be structurally valid before execution:

- labels are non-empty;
- the row point selects exactly the provided number of rows;
- the entry point selects exactly each row's width;
- the blinding count equals the row count;
- every row fits the vector-commitment setup capacity.

`BlindFoldErrorRowsRequest` and `BlindFoldCrossTermErrorRowsRequest` are the
matching transcript-free hooks for relaxed-R1CS error row materialization. They
carry the R1CS matrices, relaxed-instance scalar(s), flattened witness vector(s),
and target error-row shape.

Error row requests must be structurally valid before execution:

- labels are non-empty;
- row counts and row lengths are nonzero powers of two;
- the padded row shape has enough slots for every constraint;
- matrix rows and referenced witness columns are in bounds;
- cross-term requests use matching real/random witness lengths.

`BlindFoldFoldRowsRequest`, `BlindFoldFoldScalarsRequest`,
`BlindFoldFoldErrorRowsRequest`, and `BlindFoldFoldErrorScalarsRequest` are the
transcript-free hooks for folding already-sampled private material after the
caller has derived the folding challenge. They own only arithmetic over the
provided rows/scalars:

- regular folding computes `real + challenge * random`;
- error folding computes `real + challenge * cross + challenge^2 * random`;
- labels are non-empty;
- input lengths match, and row requests also require matching row widths.

## Backend Cache Scope

Backends may retain row commitments, private coefficients, blindings, layout
scratch, and folding witness material.

## Result Semantics

Row commitment results return commitments only. Row opening results return the
row-combined opening plus the opened evaluation. Error row results return padded
error rows only. Folding results return folded rows or folded scalars only. Full
BlindFold results return the BlindFold proof payload and private opening values
required for proof construction. They do not define verifier equations or own
transcript mutation.

## Optimization IDs

Primary IDs: `OPT-ZK-001`, `OPT-ZK-002`, `OPT-ZK-003`, `OPT-ZK-005`,
`OPT-ZK-006`.

## Feature Behavior

Compiled only with `zk`. Disabled builds expose no BlindFold backend trait or
request/result types.
