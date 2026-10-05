# Spec: Akita Multi-Chunk Trace Witnesses

| Field       | Value                                           |
|-------------|-------------------------------------------------|
| Author(s)   | @RadNi                                          |
| Created     | 2026-09-23                                      |
| Status      | implemented                                     |
| PR          | [#1896](https://github.com/a16z/jolt/pull/1896) |

## Summary

Jolt's Akita adapter previously exposed only single-chunk fold witnesses for the packed
`OneHotTrace`. PR #1896 adopts Akita's chunk-aware batch decompose-fold kernel and adds
trusted schedule artifacts for two-, four-, and eight-chunk trace witnesses. All chunk
witnesses are produced during one traversal of the streamed trace, allowing setup to select
multi-chunk geometry without materializing the one-hot polynomial or rereading the trace for
each chunk.

## Intent

### Goal

Support selectable one-, two-, four-, and eight-chunk Akita witnesses for K=16 and K=256
`OneHotTrace` commitments while preserving Jolt's existing claims, commitment layout, and
grouped-opening statement.

### Invariants

- `Single` remains the default trace profile and preserves the existing one-hot single-chunk catalogs.
- `Two`, `Four`, and `Eight` select Akita's W2R2, W4R2, and W8R2 geometry respectively: the
  root and first recursive fold are chunked, and later folds are single-chunk.
- The selected `(one_hot_k, chunk_profile)` determines one typed Akita configuration and one
  validated catalog. Setup-owned grouped rows for advice and committed-program precommits are
  adapted from that same catalog.
- Both bounded and full-width dense producer catalogs use a fixed eight-chunk security
  budget, covering response counts 1, 2, 4, and 8. Advice, bytecode, and field-valued
  precommit setup selects only the source shape and source contract; it receives no trace
  chunk profile and does not replan producers. Every grouped opening preserves those
  profiles and audits the actual response geometry. Dense-only setup rejects a one-hot
  profile selection; its own dense opening schedule uses the fixed eight-chunk budget.
- Chunk ownership uses Akita's canonical dyadic live-block partition. The streamed kernel's
  witness for every chunk equals the corresponding witness from Akita's materialized one-hot
  kernel.
- Producing multiple chunk witnesses traverses every trace row once, not once per chunk.
- The batch kernel rejects malformed challenge counts, non-uniform live-block geometry, and
  invalid chunk partitions before witness construction.
- The verifier setup serializes the selected profile and exact finalized catalog. The
  Fiat-Shamir setup preamble absorbs the catalog digest for every setup and a labeled chunk
  count for nondefault one-hot profiles. `Single` and `Dense` keep the #1948 preamble.

No existing `jolt-eval` invariant changes. Its transcript invariants exercise the generic
sponge API rather than Akita's setup preamble; the Akita-specific binding is covered by the
Fiat-Shamir inventory and setup round-trip tests.

### Non-Goals

- Changing Jolt's Akita claims, group order, commitment layout, or proof type.
- Supporting chunk counts other than 1, 2, 4, and 8, or chunking more than the first two fold
  levels.
- Adding ZK Akita support; the `akita` and `zk` features remain mutually exclusive.
- Dynamically choosing a chunk profile during proving or verification. The profile is fixed by
  preprocessing.

## Evaluation

### Acceptance Criteria

- [x] K=16 and K=256 each round-trip under the `Single`, `Two`, `Four`, and `Eight` profiles.
- [x] One trusted-advice commitment and hint created before trace-profile selection verify
  under all four profiles. Committed-bytecode proofs cover one and two bytecode objects
  under each profile.
- [x] Six companion `.aks` artifacts cover the pinned planner's admitted one- and
  two-polynomial shapes through arity 40 for K=16 and 43 for K=256. Minimum
  arities are 12/12 for `Two`, 13/12 for `Four`, and 14/13 for `Eight`
  (one/two polynomials).
- [x] Every companion artifact applies its selected chunk geometry to the root and first
  recursive fold and uses single-chunk geometry thereafter.
- [x] Streamed multi-chunk decomposition matches Akita's materialized one-hot decomposition
  across the supported ring dimensions; the scalar path remains equivalent in dense, sparse,
  and compact rotation modes.
- [x] A traversal-count test proves that chunked decomposition reads each trace row exactly
  once.
- [x] Grouped schedule provisioning inherits the selected trace profile. Dense producer rows
  come from conservative checked-in catalogs; grouped search preserves their exact profiles and
  may retry without the scalar guide for at most two bounded producers within the opening
  assignment budget. Dense-only setup rejects a one-hot profile selection.
- [x] Prover and verifier bind a labeled chunk count for nondefault one-hot profiles and use
  the same profile-specific catalog digest. `Single` and `Dense` retain the #1948 preamble.

### Testing Strategy

The primary gate is:

```text
cargo nextest run -p jolt-akita --cargo-quiet
```

The suite compares streamed and materialized chunk witnesses, counts trace-row visits, checks
the complete artifact grids, and round-trips both K values under all profiles. The Akita
Fiat-Shamir inventory retains the conditional chunk-profile absorption sites, and a frozen
digest test pins the #1948 `Single` setup preamble. Lint and formatting gates are:

```text
cargo clippy -p jolt-akita --all-targets -- -D warnings
cargo clippy -p jolt-prover --features akita --all-targets -- -D warnings
cargo fmt --check
```

No `host,zk` variant applies because Akita and ZK are compile-time-exclusive protocols.

### Performance

There is no numeric performance target in this PR. The structural requirement is that all chunk
witnesses share one trace traversal and one prepared rotation set. Position-task working-set
sizing includes the number of chunk accumulators so additional chunks do not silently multiply
the intended cache budget.

No `jolt-eval` objective is added: the current framework has no Akita-specific decomposition or
prover objective. The traversal-count test mechanically guards the principal performance
property introduced here.

## Design

### Architecture

`ProverConfig` passes `AkitaOneHotChunkProfile` to Akita setup, which selects one of eight typed
one-hot families: K=16 or K=256 crossed with one, two, four, or eight chunks. The selected
profile is serialized in `AkitaVerifierSetup`. The two dense catalogs plus those eight
families form a ten-artifact runtime bundle. `from_directory` requires the four base
files and loads any of the six companion files that are present. A selected profile
whose companion artifact is absent fails during setup. `AkitaScheduleArtifacts::new`
accepts the four base catalogs for the default single-chunk trace path.

The multi-chunk catalogs admit minimum arities 12/12 (`Two`), 13/12 (`Four`),
and 14/13 (`Eight`) for one/two-polynomial shapes under the pinned planner.
Setup validation and artifact generation share these shape-dependent bounds.
Their planner configurations use W2R2, W4R2, or W8R2 witness geometry.
The single-chunk one-hot catalogs retain non-chunked
geometry. Both dense catalogs use W8R2, independently of the trace profile, to certify
the maximum supported response envelope. Program-specific grouped rows are derived during setup
from the selected base family, then the exact extended catalog is serialized into the verifier
setup.

At the kernel boundary, Akita supplies a `DecomposeFoldBatchPlan::SparseChunked` with a
claim-major challenge carrier and canonical chunk ranges. Jolt validates the singleton
streamed batch, checks the ranges, prepares rotations once, and accumulates into a
position-by-chunk buffer. It expands digits independently for each chunk and returns
`CpuFoldResponses::chunked`, which includes Akita's aggregated global response. The scalar
fold path calls the same streamed implementation with one chunk.

Internal enums dispatch the eight concrete Akita scheme and verifier types through commitment,
opening, and verification. Akita owns proof-byte parsing. This keeps the profile and trusted
catalog type-aligned without duplicating those protocol paths.

The workspace keeps #1948's Akita and Spongefish pins. The six companion catalogs are generated
under that pinned planner, as are the regenerated conservative dense catalogs.

### Alternatives Considered

- **Run the scalar decompose-fold kernel once per chunk.** Rejected because it rereads the
  streamed trace and rebuilds rotations for every chunk.
- **Materialize `OneHotTrace` before using Akita's built-in kernel.** Rejected because the packed
  trace representation exists to avoid a trace-sized dense one-hot allocation.
- **Mutate one runtime schedule family with a chunk count.** Rejected because Akita configuration
  types and validated catalog identities are part of setup and transcript binding. Separate
  trusted families fail closed on profile/catalog mismatches.

## Documentation

`crates/jolt-akita/schedules/README.md` documents the three required artifacts, six optional
companions, profile geometry, supported arities, and regeneration selectors. The Jolt book
documents profile selection through `ProverConfig`; `specs/lattice-claims.md` remains the
normative statement-level Akita contract.

## Execution

- Define the public chunk-profile selector and profile-specific K=16/K=256 configurations.
- Generate and check in the six companion schedule catalogs, and extend loading, setup-owned
  grouped provisioning, typed backend dispatch, and generator selectors.
- Implement the chunk-aware streamed `OpeningBatchKernel` with one-pass accumulation and retain
  the one-chunk scalar path.
- Serialize the profile in verifier setup, transcript-bind nondefault one-hot profiles, update the
  Fiat-Shamir inventory, and cover all profiles with differential, traversal-count, catalog, and
  end-to-end tests.
- Retain #1948's Akita pin and regenerate the six companion catalogs under its planner.

## References

- [Jolt PR #1896](https://github.com/a16z/jolt/pull/1896)
- [Jolt PR #1948](https://github.com/a16z/jolt/pull/1948)
- [`specs/lattice-claims.md`](lattice-claims.md)
- [`crates/jolt-akita/schedules/README.md`](../crates/jolt-akita/schedules/README.md)
