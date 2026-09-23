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

- `Single` remains the default profile and preserves the existing single-chunk catalogs.
- `Two`, `Four`, and `Eight` select Akita's W2R2, W4R2, and W8R2 geometry respectively: the
  root and first recursive fold are chunked, and later folds are single-chunk.
- The selected `(one_hot_k, chunk_profile)` determines one typed Akita configuration and one
  validated catalog. Setup-owned grouped rows for advice and committed-program precommits are
  adapted from that same catalog.
- Dense advice and committed-program objects remain single-chunk. A dense-only setup rejects a
  multi-chunk profile.
- Chunk ownership uses Akita's canonical dyadic live-block partition. The streamed kernel's
  witness for every chunk equals the corresponding witness from Akita's materialized one-hot
  kernel.
- Producing multiple chunk witnesses traverses every trace row once, not once per chunk.
- The batch kernel rejects malformed challenge counts, non-uniform live-block geometry, and
  invalid chunk partitions before witness construction.
- The verifier setup serializes the selected profile and exact finalized catalog. The
  Fiat-Shamir setup preamble absorbs the catalog digest for every setup and a labeled chunk
  count for nondefault one-hot profiles. `Single` and `Dense` keep their legacy preamble.

No existing `jolt-eval` invariant changes. Its transcript invariants exercise the generic
sponge API rather than Akita's setup preamble; the Akita-specific binding is covered by the
Fiat-Shamir inventory and setup round-trip tests.

### Non-Goals

- Changing Jolt's Akita claims, group order, commitment layout, or proof type.
- Adding multi-chunk support to bounded-dense advice or committed-program objects.
- Supporting chunk counts other than 1, 2, 4, and 8, or chunking more than the first two fold
  levels.
- Adding ZK Akita support; the `akita` and `zk` features remain mutually exclusive.
- Dynamically choosing a chunk profile during proving or verification. The profile is fixed by
  preprocessing.

## Evaluation

### Acceptance Criteria

- [x] K=16 and K=256 each round-trip under the `Single`, `Two`, `Four`, and `Eight` profiles.
- [x] Six companion `.aks` artifacts cover one- and two-polynomial shapes from physical arity
  16 through 34 for K=16 and through 43 for K=256.
- [x] Every companion artifact applies its selected chunk geometry to the root and first
  recursive fold and uses single-chunk geometry thereafter.
- [x] Streamed multi-chunk decomposition matches Akita's materialized one-hot decomposition
  across the supported ring dimensions; the scalar path remains equivalent in dense, sparse,
  and compact rotation modes.
- [x] A traversal-count test proves that chunked decomposition reads each trace row exactly
  once.
- [x] Grouped schedule provisioning inherits the selected trace profile, while dense-only setup
  rejects multi-chunk selection.
- [x] Prover and verifier bind a labeled chunk count for nondefault one-hot profiles and use
  the same profile-specific catalog digest. `Single` and `Dense` retain their legacy preamble.

### Testing Strategy

The primary gate is:

```text
cargo nextest run -p jolt-akita --cargo-quiet
```

The suite compares streamed and materialized chunk witnesses, counts trace-row visits, checks
the complete artifact grids, and round-trips both K values under all profiles. The Akita
Fiat-Shamir inventory retains the conditional chunk-profile absorption sites, and a frozen
digest test pins the legacy `Single` setup preamble. Lint and formatting gates are:

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
profile is serialized in `AkitaVerifierSetup`. The existing dense catalog plus those eight
families form a nine-artifact runtime bundle. `from_directory` requires
the original three files and loads any of the six companion files that are present. A selected
profile whose companion artifact is absent fails during setup. The three-artifact
`AkitaScheduleArtifacts::new` constructor remains sufficient for the default single-chunk path.

The multi-chunk catalogs admit arities of at least 16. Their planner configurations use W2R2,
W4R2, or W8R2 witness geometry. Existing single-chunk one-hot catalogs and the dense catalog
explicitly retain non-chunked geometry. Program-specific grouped rows are derived during setup
from the selected base family, then the exact extended catalog is serialized into the verifier
setup.

At the kernel boundary, Akita supplies a `DecomposeFoldBatchPlan` containing
`challenges_per_poly`, `num_chunks`, block positions, and digit geometry. Jolt validates the
singleton streamed batch, maps each live block to its canonical dyadic chunk, prepares rotations
once, and accumulates into a position-by-chunk buffer. It then expands digits independently for
each chunk and returns one `DecomposeFoldWitness` per chunk. The scalar fold path calls the same
implementation with one chunk.

An internal enum dispatches the eight concrete Akita scheme types through commitment, opening,
proof deserialization, and verification. This keeps the profile and trusted catalog type-aligned
without duplicating those protocol paths.

The workspace pins all Akita crates to the merged Akita `main` revision that provides the
per-chunk batch-fold return contract and current committed-source planner API.

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
- Pin Akita to the merged `main` revision and use its committed-source planner interface.

## References

- [Jolt PR #1896](https://github.com/a16z/jolt/pull/1896)
- [Akita `cdb2d843`](https://github.com/LayerZero-Labs/akita/commit/cdb2d8430d09db0fc8715c8af8c67f578bfcc5fc)
- [`specs/lattice-claims.md`](lattice-claims.md)
- [`crates/jolt-akita/schedules/README.md`](../crates/jolt-akita/schedules/README.md)
