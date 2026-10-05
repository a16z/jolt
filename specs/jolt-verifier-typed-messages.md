# Design note: typed, declared proof messages

| Field     | Value       |
| --------- | ----------- |
| Author(s) | @markosg04  |
| Created   | 2026-10-05  |
| Status    | implemented |

## Decision

Every claim a proof carries is received into, and sent from, a typed claims
struct by code generated from that struct's declaration. The shape of a
stage's output claims is the shape of its typed output-points struct, which
`derive_opening_points` builds from typed dimensions. No wire shape is
computed from symbolic expressions. Each claim cell has a typed `ClaimRoute`:
sent, aliased to a source cell, or staged before the batch. One generated
function per stage turns points plus routes into the clear receive, the clear
send, the committed (ZK) rows, and the BlindFold row layout. Non-claim message
groups are typed structs that own their `send` and `receive` in one `impl`.

## Problem

The NARG port moved proof data from a typed `JoltProof` struct into the
argument string. Stage data flow stayed typed, but the proof boundary became
less declarative:

- The verifier received claims by building a placeholder from
  `wire_output_openings`, a set derived at runtime from each relation's
  symbolic output expression. That set disagreed with the claims struct for
  the address-major committed-program cycle phases. The mismatch surfaced only
  as a rejected honest proof.
- The same claim order existed in several hand-written copies: stage 6b's
  `stage6b_opening_values` and its receive, stage 4's
  `opening_values` / `post_round_opening_values` and its receive, and one
  BlindFold id list per stage (plus the field-inline splices).
- Stage 4's staged openings were sent and received by unrelated functions.

## Invariants

1. A claims struct's field declaration order is the only definition of claim
   order on the wire, in committed rows, and in the BlindFold lowering.
2. The verifier receives claims in the shape of its own output points, derived
   from public dimensions and the batch point. The prover checks its values
   against the same points (`validate_output_shape`) before recording them.
3. Every non-claim message group is a type with `send` and `receive` in one
   `impl`.
4. Routes are a function of the derived points alone, declared once per stage.
   An alias reads a cell already received; a missing source is a typed error.

## Mechanism

**`MapCells` (jolt-claims).** `#[derive(OutputClaims)]` also implements
`MapCells`: `try_map_cells(&S<A>, f) -> S<B>` visits every cell in canonical
order with its opening id. The composed aggregates delegate to their parts.

**`ClaimRoute` / `ClaimRoutes` (jolt-verifier `relations`).** `Sent`,
`Alias(source)`, or `Staged`. A cell's route is its override if one is set,
else its member's static alias (`aliased_output_openings`), else `Sent`.

**Generated per stage (`#[derive(SumcheckBatch)]`).**

- `claim_routes(points)`: the default routes, or, under
  `#[sumcheck_batch(routes)]`, the output-points struct's inherent
  `claim_routes`. Stage 4 routes its staged contribution cells `Staged`.
  Stage 6b aliases a booleanity `bytecode_ra` cell whose point equals its
  bytecode read-RAF source's point.
- `receive_output_claims(points, routes, staged, t)`: `Sent` cells are
  received, `Alias` cells copy their source, `Staged` cells come from
  `staged`.
- `wire_claim_values` (the clear send) and `committed_claim_values` (the ZK
  rows) over claims and routes.
- `committed_claim_layout(points, routes)`: the committed ids in row order plus
  the `OpeningAlias` rows. `verify_zk` derives the points, builds this layout,
  and returns it in the `CommittedOutputClaimShape`. BlindFold's `add_stage`
  reads ids and aliases from the shape, so no lowering keeps its own list.
- `validate_output_shape(claims, points)`: the prover's self-check.

`ConcreteSumcheck::wire_output_openings`, `output_claim_count`, the
`no_opening_values` / `no_output_shape` opt-outs, and the prover driver's
`curate` hook are deleted.

**Recording.** The prover driver records `committed_claim_values` when its
recorder's `ClaimRecorder::RECORDS_STAGED` is true (committed) and
`wire_claim_values` otherwise (clear). Stage 4 sends its staged openings
before the batch exactly when the mode recorder does not record them.

**Typed message groups.** `RamValCheckStagedOpenings<C>` is a cell-generic
claims struct over the RAM value check's staged fields, with `send`,
`receive(points, t)`, and `by_id` (the `staged` pre-fill).
`RamValCheckInitStructure::staged_points` builds its points, which
`RamValCheck::derive_opening_points` now fills instead of leaving absent.

## Compatibility

Clear wire bytes are unchanged. ZK proofs change: stage 4's committed rows are
now in declaration order (registers, field registers, then the RAM value
check's fields with its staged cells in place) instead of staged cells first.
The BlindFold lowering follows automatically. Fixtures regenerate.

## Verification

- Prover e2e in clear, ZK, and Akita builds; the verifier fixture suite.
- Order pins with distinct sentinels against the generated functions: stages
  2–7 `wire_claim_values`, stage 4 `committed_claim_values`, and the 6b alias
  route.
- `MapCells` visit order equals `canonical_order`, keeps shape, and stops at
  the first error.
- The BlindFold stage-1 row order equals the composed R1CS column order the
  factored output formula consumes, from the production layout.
- The engine ZK twin checks `verify_zk`'s derived points and row count against
  the prover's.
