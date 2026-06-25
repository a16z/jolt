# Commitment Request Contract

## Constructed By

`jolt-prover` stage request builders.

## Slots

`CommitmentSlot` is a scheduling/result key. It is not proof order.

## Witness Views

Each item carries a committed `OracleRef` and a `ViewRequirement`.

## Backend Cache Scope

Backends may retain PCS partial state, opening hints, stream metadata, and
representation-specific scratch buffers.

## Result Semantics

Results return resolved witness descriptors, streamed chunk metadata, PCS
commitments, and PCS opening hints by slot. `jolt-prover` maps them to
verifier-visible commitments by logical oracle ID.

## Optimization IDs

Primary IDs: `OPT-COM-001`, `OPT-COM-002`, `OPT-COM-006`, `OPT-COM-007`.

## Feature Behavior

Field-inline adds explicit committed field-register surfaces through typed
field-inline outputs. ZK commitment behavior is selected by the request/proof
path, not by hidden backend mode.
