# Sumcheck Request Contract

## Constructed By

`jolt-prover` stage request builders. `jolt-prover` owns the Fiat-Shamir
transcript lifecycle and constructs backend requests between transcript
operations.

## Slots

`SumcheckSlot` identifies each requested sumcheck instance. `BackendValueSlot`
identifies input/output claims and evaluations.

## Witness Views

Requests carry backend-local relation IDs and witness `ViewRequirement` values.
Formula semantics are selected by `jolt-prover` from `jolt-claims`, not by the
backend.

Product-evaluation requests also carry optional kernel metadata: the relation
being accelerated and the optimization-inventory IDs the caller expects the CPU
backend to preserve. Metadata is accounting and dispatch context; it must not
change Fiat-Shamir challenge derivation or verifier-visible proof semantics.

## Optimization Porting

The CPU backend is expected to carry the optimized core sumcheck closure or
equivalent algorithm for the requested slice. A prover path is not complete just
because a clear reference implementation is correct; the backend kernel must
preserve the relevant `jolt-core` optimization IDs and pass the harness
`CorePerformanceParity` gate.

Performance wins over backend modularity for the initial CPU backend. Kernels
may be coarse, relation-specific, and locally duplicated when that preserves the
optimized `jolt-core` algorithm shape or avoids abstraction overhead in hot
loops. The stable boundary is the request/result API, not a generic internal
kernel hierarchy.

The generic CPU sparse-product kernel is a fallback/reference layer for shared
validation and small slices. It is not sufficient evidence of core parity by
itself. Performance-bearing Spartan, lookup, RAM/register, and claim-reduction
ports should use coarse kernels with the memory layout, streaming windows,
split-eq tables, sparse handling, and rayon parallelism required by the
corresponding core routine.

Every performance-bearing CPU kernel must have a microbenchmark covering the
target shape and a measured peak-allocation check against analytical memory for
the required data structures. Use:

```bash
cargo bench -p jolt-backends --bench sumcheck_kernels
```

## Backend Cache Scope

Backends may retain relation state, equality tables, bound polynomials,
streaming schedules, and round scratch buffers.

## Materialization

When a prover stage needs dense witness material between Fiat-Shamir transcript
operations, it requests materialization through the backend. This keeps witness
layout, validation, and backend caching policy out of `jolt-prover` while still
leaving transcript mutation and verifier-proof construction to `jolt-prover`.

## Linear Products

Transparent prover paths may request batched linear-product evaluations over
materialized witness polynomials. `jolt-prover` supplies the transcript-derived
points, row weights, batching scales, sparse rows, and column mapping. Backends
only evaluate the requested multilinear witnesses and sparse linear forms; they
do not derive Fiat-Shamir challenges or append transcript material.

## Product Uni-Skip Rows

Stage 2 may request product uni-skip evaluation over primitive row data instead
of six dense field-materialized factor polynomials. The prover or witness layer
still owns the semantic projection from trace rows into these primitive fields;
the backend only applies the supplied equality point, row weights, and batching
scale.

## Result Semantics

Results return proof payloads and output evaluations by slot. They do not
mutate transcripts or construct verifier proof structs.

## Harness Gates

Each sumcheck prover path using this contract must pass:

- `VerifierCorrectness`: `jolt-verifier` accepts the modular proof component
  produced by `jolt-prover` from replayed fixture inputs.
- `CorePerformanceParity`: CPU backend time, memory, and other reasonable
  metrics match the corresponding `jolt-core` optimized slice within the
  prover-path `PerfGate`.

Fixture replay is the default inner loop. Broad proof-flow tests are regression
checks and should not be the only evidence that a backend sumcheck slice is
done.

## Optimization IDs

Primary IDs: `OPT-SC-001`, `OPT-SC-008`, `OPT-EQ-004`, `OPT-REL-001`.

## Feature Behavior

ZK committed-round material is requested explicitly under `zk`; transparent
paths should not allocate BlindFold private material.
