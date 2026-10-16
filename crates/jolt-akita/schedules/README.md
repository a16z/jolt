# Jolt Akita schedule artifacts

These canonical `.aks` catalogs are public runtime data. The four base catalogs
support Jolt's default single-chunk execution and its adapter fixtures. The six
multi-chunk companion catalogs contain only the small guest, roundtrip, grouped
cutover, and field-inline boundary shapes exercised by this repository. They do
not promise production coverage for downstream multi-chunk deployments.

Applications own their supported shapes, generation policy, and deployment
catalogs. Generate those catalogs with the pinned Akita planner and validate them
against the corresponding `jolt_akita::configs` policy. Supply the complete bundle
explicitly through `AkitaScheduleArtifacts::from_directory`, retain it in `Arc`,
and pass it to preprocessing. No source change or custom verifier is required
when an application supplies additional admitted rows for a supported family.

`shared_from_default_directory` is a host/dev convenience: it reads
`JOLT_AKITA_SCHEDULE_DIR` or falls back to this directory. It requires dense-bounded,
dense-full, onehot-k16, and onehot-k256; profile companions are optional until
selected. A missing companion or exact row fails setup. The application's
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
the application's proving backend. The catalog digest and nondefault chunk
profile are bound into the transcript. Verification only admits selections from
its installed trusted setup; a proof cannot install a catalog.

Single, Two, Four, and Eight profiles remain supported. The chunked profiles
split the root and first recursive fold; subsequent folds are single-chunk.
Recursive witness bodies inherit ownership and align to successor blocks.
The two dense fixtures retain their fixed eight-chunk producer budget so the
cross-profile commitment-reuse tests exercise compatible producer profiles.
Changing that policy requires regenerating compatible producer and consumer
catalogs and their prepared setups.

The default single-chunk K=16 catalog covers native arities 16–35 and widths
51–64. K=256 single-chunk native rows are explicit adapter and benchmark fixtures.
One-hot rows use direct schedules below logical `2^21` cycles and setup
offloading from that cutoff. Multi-chunk fixture inventories are declared in
`src/schedules/mod.rs`; deployment inventories belong to their applications.

The default single-chunk catalog covers traces through `2^31` cycles: each
K=16 native column then has 35 variables (`31 + 4`), with 51–64 columns
according to the bytecode and RAM domains. Every auxiliary object must also
admit a grouped row under the pinned planner policy.

Catalog identity is transcript-bound. Extending a catalog changes newly generated
proof bytes, including proofs selecting previously provisioned rows. Existing
proofs keep their matching serialized verifier preprocessing; newly generated
preprocessing is a distinct setup and is not interchangeable with the old one.

Regenerate this repository's catalogs:

```sh
cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
```

Selectors include `k16`, `k256`, `k16-single`, `k256-single`, `w2r2`, `w4r2`,
`w8r2`, `dense-bounded`, and `dense-full`. Append `--check` to replan and compare
canonical bytes without replacing the files. The catalogs are generated with
the Akita revision pinned in the workspace `Cargo.toml`.

`external_catalog` generates a two-chunk shape absent from the packaged
fixtures, proves it with that external catalog, and verifies the transported
public setup in a fresh process after the source catalogs have been deleted.
Other tests cover malformed setup, profile mismatch, canonical shape checks, and
the selected boundary geometries. Production coverage matrices and application
catalog freshness belong downstream.

The child verifier uses only the exported public inputs and deliberately points
its catalog environment at a nonexistent directory.
