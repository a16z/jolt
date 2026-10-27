# Jolt Akita schedule artifacts

These canonical `.aks` catalogs are public runtime data. The four base catalogs
support Jolt's default single-chunk execution and its adapter fixtures. The two
W8R2 fixture catalogs contain only the small guest, roundtrip, grouped
cutover, and field-inline boundary shapes exercised by this repository. They do
not promise production coverage for downstream multi-chunk deployments.

Applications own their supported shapes, generation policy, and deployment
catalogs. Generate those catalogs with the pinned Akita planner and validate them
against the corresponding `jolt_akita::configs` policy. Supply the complete bundle
explicitly through `AkitaScheduleArtifacts::from_directory(path, dense_profile)`,
retain it in `Arc`,
and pass it to preprocessing. No source change or custom verifier is required
when an application supplies additional admitted rows for a supported family.

`shared_from_default_directory` is a host/dev convenience: it reads
`JOLT_AKITA_SCHEDULE_DIR` or falls back to this directory. It requires the dense-bounded
and dense-full files and discovers one-hot catalogs named
`jolt-fp128-onehot-k<K>-w<chunks>r<levels>.aks`. Unsuffixed one-hot filenames denote
`w1r0`. All geometries for a given K share the same catalog family identity;
the policy binds the exact geometry. The bundle stores entries keyed by K and geometry; a missing selected
catalog, duplicate key, or missing exact row fails loading or setup. The application's
producer and consumer setups must use compatible frozen producer profiles.

Committed programs use one whole-bytecode group and one image group. Their
physical arities are `max(14, logical_arity)` with a ceiling of 34. Bytecode
logical arity is `9 + log2(padded_bytecode_rows)` and may exceed the trace
group arity. Producer and grouped schedule admission run before whole-table
allocation. Existing dense catalogs are reused; this changes the requested
group shapes, not the offline response-chunk budget.

During preprocessing, Jolt adapts rows whose shapes depend on advice, field
increments, or direct committed-program sizes. Those rows are merged with the
relevant base catalog.
Preprocessing validates and serializes the exact catalog in `AkitaVerifierSetup`.
An independent upstream verifier reconstructs its verification tables from that
public setup, without reading source catalogs, running the planner, or needing
the application's proving backend. The catalog digest is bound into the transcript. Every one-hot setup also binds
its chunk count and leading chunked fold count, including `w1r0`. Verification only admits selections from
its installed trusted setup; a proof cannot install a catalog.

Schedule catalogs are part of the verifier's trusted preprocessing. Before use,
catalogs pass through Akita's validated constructor, including when reconstructed
from serialized verifier preprocessing. Admission checks the expected configuration
and chunk geometry and applies Akita's schedule audit. The proof carries a
schedule-row digest, which the verifier resolves within its own validated catalog.
Unknown rows and rows incompatible with the opening or verifier setup are rejected.

`AkitaChunkProfile { chunks, levels }` selects runtime geometry, validated by
Akita at admission. The packaged multi-chunk fixtures use `w8r2`: the root and first recursive fold are chunked. Applications may supply
catalogs with other admitted chunk counts and leading chunked fold counts.
Folds after the selected prefix are single-chunk.
Recursive witness bodies inherit ownership and align to successor blocks.
The two packaged dense fixtures use an eight-chunk producer budget so the
cross-profile commitment-reuse tests exercise compatible producer profiles.
Changing that policy requires regenerating compatible producer and consumer
catalogs and their prepared setups.

The default K=16 catalogs cover native arities 16–35 and widths 51–64, that is,
traces through `2^31` cycles. K=256 single-chunk native rows are explicit
adapter and benchmark fixtures. One-hot trace rows use direct schedules below
logical `2^21` cycles. Every single-chunk trace row from that cutoff uses setup
offloading; multi-chunk rows there follow the planner's objective.
`one_hot_catalogs_switch_to_setup_offloading_at_the_trace_cutover` checks both
rules for every emitted trace key. Multi-chunk fixture inventories are declared
in `src/schedules/mod.rs`; deployment inventories belong to their applications.

Regenerate this repository's catalogs:

```sh
cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
```

Selectors include `k16`, `k256`, `k16-single`, `k256-single`, `w8r2`, `dense-bounded`, and `dense-full`. Append `--check` to replan and compare
canonical bytes without replacing the files. The catalogs are generated with
the Akita revision pinned in the workspace `Cargo.toml`. Every catalog binds
Akita's planner policy digest, which includes the SIS security policy and its
estimator tables, so catalogs planned under an earlier policy are rejected at
load time.

`external_catalog` generates a two-chunk shape absent from the packaged
fixtures, proves it with that external catalog, and verifies the transported
public setup in a fresh process after the source catalogs have been deleted. Other tests cover malformed setup, profile
mismatch, canonical shape checks, and the selected boundary geometries. Production
coverage matrices and application catalog freshness belong downstream.

The child verifier uses only the exported public inputs and deliberately points
its catalog environment at a nonexistent directory.

The dense profile is supplied by the caller and travels with the catalog in verifier
preprocessing. The verifier does not impose the packaged fixtures' geometry.
