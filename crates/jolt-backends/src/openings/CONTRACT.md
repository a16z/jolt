# Opening Request Contract

## Constructed By

`jolt-prover` Stage 8 request builders.

## Slots

`OpeningSlot` identifies query order for backend execution. `BackendValueSlot`
identifies claimed evaluations and joint claims.

## Witness Views

Requests carry committed/virtual oracle references, opening points, scalars,
and hint requirements.

## Backend Cache Scope

Backends may retain joint polynomial state, Dory opening hints, folded one-hot
tables, and RLC scratch buffers.

## Result Semantics

Results return PCS proof payloads, joint claims, and query evaluations. They do
not choose final opening order or transcript binding.

## Optimization IDs

Primary IDs: `OPT-OPEN-001`, `OPT-OPEN-003`, `OPT-OPEN-006`, `OPT-OPEN-007`.

## Feature Behavior

ZK opening requests hide evaluations through the selected PCS ZK opening path.
Transparent requests return clear joint claims.
