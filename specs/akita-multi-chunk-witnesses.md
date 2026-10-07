# Spec: Akita Multi-Chunk Trace Witnesses

| Field       | Value                                           |
|-------------|-------------------------------------------------|
| Author(s)   | @RadNi                                          |
| Created     | 2026-09-23                                      |
| Status      | implemented                                     |
| PR          | [#1976](https://github.com/a16z/jolt/pull/1976) |

## Summary

Jolt's Akita adapter previously exposed only single-chunk fold witnesses for the packed
`OneHotTrace`. PR #1976 ports #1896's chunk-aware batch decompose-fold integration and adds
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
  count under `akita_chunk_profile` for nondefault one-hot profiles. `Single` and `Dense`
  omit this label.
- Stage 0 rejects a `ProverConfig.akita_chunk_profile` that differs from the prepared
  verifier setup before making commitments; proving cannot change the setup's profile.

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
  come from conservative checked-in catalogs. Akita first attempts guided adaptation and
  automatically falls back to full planning on `UnsupportedSchedule`, preserving exact producer
  profiles and auditing the result. Jolt's admission gates apply before search, as documented in
  the [schedule README](../crates/jolt-akita/schedules/README.md). Dense-only setup rejects a
  one-hot profile selection.
- [x] Prover and verifier bind a labeled chunk count for nondefault one-hot profiles and use
  the same profile-specific catalog digest. The label rename leaves `Single` and `Dense`
  absorption unchanged; regenerating the dense catalogs changes their catalog binding.
- [x] Chunk-profile APIs and named setup serialization use the Akita-specific names below,
  with no legacy aliases. Nondefault profiles bind the `akita_chunk_profile` transcript label.
- [x] Proving rejects profile mismatches for both single-chunk and chunked preprocessing.

### Testing Strategy

The focused checks are:

```text
cargo nextest run -p jolt-akita --cargo-quiet
cargo nextest run -p jolt-prover --features akita,prover-fixtures -E 'binary(akita_e2e)' --cargo-quiet
cargo nextest run -p jolt-verifier --features fs-audit --test fs_obligations --cargo-quiet
```

The suite compares streamed and materialized chunk witnesses, counts trace-row visits, checks
the complete artifact grids, and round-trips both K values under all profiles. These round trips
also reject transported verifier setups whose profile disagrees with their catalog, including
chunked catalogs encoded as the legacy single-chunk variant. The prover
suite covers profile-mismatch rejection and streamed proofs under every profile. The Akita
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

On 2026-10-05, an isolated synthetic streamed-kernel comparison used the pre-chunk
decomposition and kernel entry points from `4f442a3caf6ecc0621986c133789f16eef2bc0f2`
against the current implementation, with the same remaining sources and dependencies.
Both builds used optimization level 3 without LTO, four Rayon workers, 65,536 generated
trace rows, K=16, 64 selector slots, and 8,192 positions. The 24 cases crossed D=64/256,
3/48 semantic columns, 1/3 digits, and Dense/Sparse/Compact rotations. Each process
discarded two warmups and measured eleven samples; two passes reversed revision order.
The reported changes compare the means of the two per-process medians.

The 48-column cases ranged from a 7.7% improvement to a 1.5% slowdown. Small D64 cases
showed measurable overhead: Dense took 1.091 ms versus 0.892 ms with one digit (+22.2%),
and 1.266 ms versus 1.086 ms with three digits (+16.7%); Compact with one digit took
1.942 ms versus 1.643 ms (+18.2%). These measurements include traversal, rotation
preparation, accumulation, digit expansion, and witness construction, but exclude setup,
fixture creation, and the rest of proving. They do not establish a full-proof speedup or
rule out single-chunk regressions. Peak process RSS was recorded separately; it is not
an allocation count. The one-chunk, one-digit expansion reuses its coefficient buffer.

No `jolt-eval` objective is added: the current framework has no Akita-specific decomposition or
prover objective. The traversal-count test mechanically guards the principal performance
property introduced here.

## Design

### Architecture

`ProverConfig` passes its `akita_chunk_profile: AkitaChunkProfile` choice to Akita setup,
which selects one of eight typed one-hot families: K=16 or K=256 crossed with one, two,
four, or eight chunks. The selected
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

Grouped provisioning uses the pinned Akita planner's guided-then-full search. Full planning
may change the scalar row's fold geometry, opening parameters, relation modes, and
direct/offloaded topology; the selected trace chunk profile and frozen dense producer profiles
remain fixed. Jolt rejects multiple full-width producers and a full-width producer with more
than two auxiliary producers before search. It also rejects bounded-only `Single` requests
whose scalar guide has no recursive child fold. Other bounded-only requests may use full
planning, including `Single` requests and batches with more than two producers. Akita owns
the search limits and opening-assignment enumeration. The exact audited result is frozen
during preprocessing and consumed by proving and verification.

At the kernel boundary, Akita supplies a `DecomposeFoldBatchPlan::SparseChunked` with a
claim-major challenge carrier and canonical chunk ranges. With native trace batching,
Jolt validates the ordered column views over one shared trace owner, checks the ranges,
prepares rotations once, and accumulates into a position-by-chunk buffer across all
columns. It expands digits independently for each chunk and returns
`CpuFoldResponses::chunked`, which includes Akita's aggregated global response. The
single-chunk batch uses the same streamed implementation with one chunk. Native batching
establishes first-fold cycle locality; recursive ownership alignment remains the
follow-up described in [the native batching spec](akita-native-trace-batching.md).

Internal enums dispatch the eight concrete Akita scheme and verifier types through commitment,
opening, and verification. Akita owns proof-byte parsing. This keeps the profile and trusted
catalog type-aligned without duplicating those protocol paths.

Schedule catalogs are generated with the Akita planner revision pinned in the workspace.

### Alternatives Considered

- **Run the scalar decompose-fold kernel once per chunk.** Rejected because it rereads the
  streamed trace and rebuilds rotations for every chunk.
- **Materialize `OneHotTrace` before using Akita's built-in kernel.** Rejected because the packed
  trace representation exists to avoid a trace-sized dense one-hot allocation.
- **Mutate one runtime schedule family with a chunk count.** Rejected because Akita configuration
  types and validated catalog identities are part of setup and transcript binding. Separate
  trusted families fail closed on profile/catalog mismatches.

## Documentation

### Breaking changes and migration

The chunk-profile API is renamed throughout setup, proving, and verification:

| Previous name | Current name |
|---------------|--------------|
| `AkitaOneHotChunkProfile` | `AkitaChunkProfile` |
| `one_hot_chunk_profile` field and accessor | `akita_chunk_profile` |
| `with_one_hot_chunk_profile` builder | `with_akita_chunk_profile` |

Callers must use the new names; no deprecated aliases are provided. The `Single`,
`Two`, `Four`, and `Eight` variants retain their meanings and discriminants.

`AkitaSetupParams` also serializes the field as `akita_chunk_profile` in named formats
such as JSON. Recipes containing the old `one_hot_chunk_profile` key are rejected by
`deny_unknown_fields`; rename the key or regenerate the recipe. There is no serde alias
for the old key. Renaming fields and types does not itself change positional bincode
encoding; the added profile and artifact fields cause the cache break described below.

For nondefault one-hot profiles, the setup preamble's transcript label changes from
`akita_one_hot_chunk_profile` to `akita_chunk_profile`. This changes Fiat-Shamir challenges:
proofs created with the previous label must be regenerated with the updated prover and
verified with the updated verifier. No legacy transcript path is provided. `Single` and
`Dense` omit this label, so the label rename does not change their preamble or the legacy
`Single` verifier-setup encoding. The Fiat-Shamir absorption inventory records the new label.

Proving now rejects a `config.akita_chunk_profile` that differs from preprocessing with
`ProverError::Unsupported` at stage 0, before commitments. Previously, the prove-time
setting was silently ignored. Reuse the preprocessing configuration or regenerate
preprocessing for the requested profile. A newly derived `ProverConfig` defaults to
`Single`; when reusing a chunked setup, explicitly retain its selected profile.

This change intentionally breaks bincode compatibility for previously serialized
`AkitaSetupParams` and `AkitaScheduleArtifacts`. These are regenerable preprocessing
inputs: discard caches written before the multi-chunk fields were added, reload
compatible schedule catalogs, and rerun preprocessing with the current implementation.
`#[serde(default)]` supports omitted fields in map-based formats such as JSON; it does
not make older bincode encodings compatible. No legacy decoder or migration is provided.
The legacy `Single` verifier-setup encoding remains unchanged. Compatible `.aks`
catalogs can be reused; rebuilding these caches does not itself require catalog
regeneration.

### Supporting documentation

`crates/jolt-akita/schedules/README.md` documents the four required artifacts, six optional
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
- Generate schedule catalogs with the workspace's pinned Akita planner.

## References

- [Jolt PR #1976](https://github.com/a16z/jolt/pull/1976)
- [Jolt PR #1896](https://github.com/a16z/jolt/pull/1896)
- [Jolt PR #1948](https://github.com/a16z/jolt/pull/1948)
- [`specs/lattice-claims.md`](lattice-claims.md)
- [`crates/jolt-akita/schedules/README.md`](../crates/jolt-akita/schedules/README.md)
