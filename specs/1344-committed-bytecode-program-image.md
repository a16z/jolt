# Committed bytecode and program image

Committed-program mode replaces the verifier's full bytecode and initial program
image with metadata and two trusted commitments. Both Dory and Akita commit one
whole padded bytecode polynomial (`ProgramBytecode`) and one initial-image
polynomial (`ProgramImageInit`).

## Encoding and geometry

The canonical row encoder in `jolt-kernels/src/committed_program.rs` includes
register one-hot lanes, unexpanded PC, signed immediate, circuit/instruction
flags, lookup selectors, and the RAF flag. Immediate magnitudes must fit u64.
The logical bytecode arity is `committed_lane_vars() + log2(padded_bytecode_rows)`.
The validated bytecode length is a nonzero power of two.

Dory uses a balanced matrix and sizes setup over the trace, advice, bytecode,
and image candidates. Its final opening grid includes precommitted anchor
points, allowing bytecode larger than the trace without changing the trace
length. Both CycleMajor and AddressMajor encodings remain supported. The
existing two-phase reduction schedule adds rounds for larger static objects.

Akita retains an independent bounded-dense bytecode group, its zero-prefix
physical-floor embedding, CycleMajor-only admission, and trusted producer/grouped
schedule checks before allocation. With nine lane variables, physical floor 14,
and physical cap 34, its structural bytecode limit is 2^25 padded rows. This is
not a deployment memory guarantee.

Address-bit decomposition (`BytecodeRa`) and internal PCS response profiles are
separate from bytecode-table partitioning and remain part of the protocol.

## Reduction and opening

Stage 6b batches the staged `BytecodeValClaim` values with eta powers and reduces
them over the whole row/lane grid. Cycle output is exactly one of intermediate
handoff or final whole-bytecode claim. Stage 7 finishes the address phase when
required. The final output is the scalar `ProgramBytecode` opening times the
canonical `OutputWeight`: lane-weight evaluation, bytecode-address equality,
and skipped-round normalization. There are no partition weights or dropped
address bits. The symbolic expression remains owned by `jolt-claims` and is
consumed by clear verification and BlindFold lowering.

The image reduction remains separate. Stage 0 absorbs the bytecode commitment
then the image after trace/advice commitments. Dory stage 8 includes the two
precommitted polynomials in its homomorphic joint-opening batch; Akita includes
the two independent groups in its role-bound opening.

## APIs and transport

Dory and Akita committed preprocessing take no bytecode-table count. Verifier
preprocessing has named bytecode/image commitments; Dory prover preprocessing
retains one bytecode hint and one image hint. Generated SDK committed helpers
also take no count. SDK program preprocessing uses `ProgramPreprocessingMode`
to select Full or Committed. Example `--committed-bytecode` flags take no value.

Retired serialized identifier slots use uninhabited payloads to preserve live
tags while rejecting old identifiers. Both backends introduce new preprocessing
digest domains in this change:

- Dory: `jolt/program-preprocessing/dory-whole-bytecode/v2`.
- Dory field-inline: `jolt/program-preprocessing/dory-whole-bytecode/field-inline/v2`.
- Akita: `jolt/program-preprocessing/akita-whole-bytecode/v1`.
- Akita field-inline: `jolt/program-preprocessing/akita-whole-bytecode/field-inline/v1`.

All existing Dory and Akita preprocessing and proofs must be regenerated,
including full mode. Committed preprocessing records the coefficient trace order,
and verification rejects a proof declaring a different order. Dory's domain
version 2 includes this order binding. There is no old-format migration or
automatic partitioning. Committed mode remains unsupported with field-inline.

## Validation and limits

Committed proofs must verify in clear and ZK modes, with both Dory trace orders
and with bytecode larger than the trace. Verify transported preprocessing and
proofs, altered claims/commitments, and malformed cycle output states. Preserve
Akita singleton/profile coverage and all guest acceptance matrices. Frozen
preprocessing digests and Fiat–Shamir inventories bind deliberate wire changes.

Whole-table materialization and a larger Dory joint grid can increase memory and
opening cost. This change simplifies the protocol and implementation; it makes
no performance claim. Existing structural, setup, and trusted schedule checks
remain the admission mechanisms.
