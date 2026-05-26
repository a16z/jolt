# BlindFold Request Contract

## Constructed By

`jolt-prover` ZK orchestration after all committed sumcheck and opening
material is available.

## Slots

`BlindFoldSlot` identifies committed round rows. `BackendValueSlot` identifies
private openings and output-claim values.

## Witness Views

BlindFold requests consume private sumcheck/opening material already produced
by earlier backend requests.

## Backend Cache Scope

Backends may retain row commitments, private coefficients, blindings, layout
scratch, and folding witness material.

## Result Semantics

Results return the BlindFold proof payload and private opening values required
for proof construction. They do not define verifier equations.

## Optimization IDs

Primary IDs: `OPT-ZK-001`, `OPT-ZK-002`, `OPT-ZK-003`, `OPT-ZK-005`.

## Feature Behavior

Compiled only with `zk`. Disabled builds expose no BlindFold backend trait or
request/result types.
