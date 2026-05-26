# Sumcheck Request Contract

## Constructed By

`jolt-prover` stage request builders.

## Slots

`SumcheckSlot` identifies each requested sumcheck instance. `BackendValueSlot`
identifies input/output claims and evaluations.

## Witness Views

Requests carry backend-local relation IDs and witness `ViewRequirement` values.
Formula semantics are selected by `jolt-prover` from `jolt-claims`, not by the
backend.

## Backend Cache Scope

Backends may retain relation state, equality tables, bound polynomials,
streaming schedules, and round scratch buffers.

## Result Semantics

Results return proof payloads and output evaluations by slot. They do not
mutate transcripts or construct verifier proof structs.

## Optimization IDs

Primary IDs: `OPT-SC-001`, `OPT-SC-008`, `OPT-EQ-004`, `OPT-REL-001`.

## Feature Behavior

ZK committed-round material is requested explicitly under `zk`; transparent
paths should not allocate BlindFold private material.
