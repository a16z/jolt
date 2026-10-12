# Jolt Akita schedule artifacts

This directory contains Jolt's base Akita schedule catalogs as canonical
`.aks` files. They are runtime data, not generated Rust modules and not
embedded into the executable.

Application preprocessing loads the four original files and any present
multi-chunk companions once, wraps the resulting
`AkitaScheduleArtifacts` in `Arc`, and passes that immutable bundle explicitly
to every `AkitaSetupParams` constructor. Production deployments should call
`AkitaScheduleArtifacts::from_directory` with a versioned, deployment-owned
path. `shared_from_default_directory` is the host/dev loader: it reads
`JOLT_AKITA_SCHEDULE_DIR`, falling back to this packaged source directory, and
aborts if the catalogs cannot be read. Protocol setup and verification never
discover files or consult the environment.

Advice and committed-program objects use the bounded dense catalog. Field-register
increments use the full-width dense catalog, with each group's source contract
preserved in the joint opening. Both dense catalogs are planned for eight response
chunks, the maximum supported count. For fixed dense response geometry, A's
collision envelope grows linearly with chunk count; eight therefore covers one,
two, and four as well. B is certified for the selected opening geometry. These
producer profiles are independent of the eventual trace chunk selection.

During preprocessing, Jolt adapts rows whose shapes depend on advice, field
increments, or direct committed-program sizes. Those rows are merged with the
relevant base catalog.
The resulting exact catalog is serialized inside `AkitaVerifierSetup`, so a
transported verifier setup does not depend on process-global state or on these
source-tree files.

Serialized `AkitaSetupParams` and `AkitaScheduleArtifacts` are regenerable
preprocessing inputs whose bincode format is tied to the implementation version.
The multi-chunk fields intentionally break compatibility with older serialized
values. Discard those caches, reload compatible `.aks` catalogs, and rerun
preprocessing with the current code. Serde defaults do not provide bincode
backward compatibility. Compatible catalogs can be reused without regeneration.
The legacy `Single` verifier-setup encoding remains unchanged.

The one-hot artifacts are hybrid catalogs. A logical trace shorter than
`2^21` uses a direct schedule. A trace of `2^21` cycles or longer uses a
setup-offloaded schedule. Akita uses K=16 committed chunks at every trace
length, with catalog coverage through `2^30`. Virtual lookup chunks are 16 bits
below `2^25` and 32 bits at or above it. The K=256 catalogs retain explicit native trace test and benchmark keys plus
the scalar adapter/planner grid; they do not provide a general native trace range.
This is an offline catalog policy: proving and verification simply
resolve the exact admitted row and never choose a mode dynamically.

Each K=16 and K=256 family has W2R2, W4R2, and W8R2 multi-chunk companion
catalogs. The selected profile splits the root and first recursive fold into
two, four, or eight chunks, while later folds remain single-chunk. Native K=16
trace groups cover column arities 16–34 and widths 51–64 in every profile.
The adapter and grouped-planner diagnostic grids retain the base branch's
one- and two-polynomial rows, with these minimum physical arities for both K values:

| Profile | One polynomial | Two polynomials |
| --- | ---: | ---: |
| Single | 12 | 12 |
| Two (W2R2) | 12 | 12 |
| Four (W4R2) | 13 | 12 |
| Eight (W8R2) | 14 | 13 |

The scalar diagnostic upper arity is 40 for K=16 and 43 for K=256. These are
physical scalar shapes, not selector-packed Jolt traces. Setup and grouped
provisioning check the profile floors before constructing backend matrices
and resolve the exact native group shape, including its column count.
The original one-hot catalogs remain single-chunk. Dense standalone opening
schedules use the fixed eight-chunk budget. Four-file directories with the
current dense catalogs support `Single`;
selecting a profile whose companion catalog is absent fails during setup.
Grouped opening rows inherit the selected trace profile; precommit producers
keep their fixed profiles.

### Field-inline provisioning coverage

PR CI resolves every K=16 production key under Single and the six
width 51/64 × arity 16/25/34 boundary shapes under each of W2R2, W4R2,
and W8R2. The three exhaustive chunked-grid tests remain available for
catalog/provisioner changes but are excluded from the PR CI invocation:

```sh
cargo nextest run --cargo-profile ci -p jolt-akita --features field-inline \
  -E 'test(/field_inline_rows_cover_k16_w[248]r2_production_grid/)' --cargo-quiet
```

These tests run real schedule adaptation for 266 keys per profile and can
consume tens of minutes. They are separate from catalog freshness checking
with `gen_jolt_schedules … --check`.

### Grouped provisioning diagnostic

Audit the complete advice matrix without allocating backend matrices or proving
traces:

```text
cargo run --release -p jolt-akita --bin check_grouped_schedules -- crates/jolt-akita/schedules /tmp/jolt-grouped-schedules.csv full
```

The diagnostic reads admitted one-polynomial final arities from each catalog,
checks both K values and all four trace profiles, and crosses absent advice with
physical producer arities 11 through 22 for both advice roles. It skips only the
empty batch. `unsupported_producer` records arities missing from the dense
catalog; `provisioning_failed` records failures for admitted producers and makes
the command fail. `unsupported_grouped_shape` records an explicit capability
rejection matching the provisioner's Single-profile scalar-guide error for that
K and final arity. Other schedule rejections count as `provisioning_failed`,
including all chunked-profile rejections. For bounded-only `Single` requests,
Jolt requires the selected scalar row to have a recursive child fold. The
smallest Single scalar rows reach their terminal immediately (K=16 arities
12–15 and K=256 arities 12–16), so preprocessing rejects these requests before
either guided or full search, naming K, profile, final arity, and the requested
groups. This is a Jolt admission rule; it does not establish that full planning
could never find a grouped schedule. Scalar-only setups and full-width producer
batches retain their existing paths.
Successful rows are audited by the production provisioner and
checked for unchanged producer profiles and the requested trace chunk count.
The CSV is flushed after each final arity; this is an expensive offline check.
Record the code revision and catalog checksums alongside the report when sharing
results. This command never changes schedule artifacts.

Use `boundary` instead of `full` for the 24 K=16 cutover cases: final arities
31 and 32, both advice roles at 21 or 22, and Two/Four/Eight profiles. The unit test `grouped_advice_rows_cover_recursive_cutover` also covers native
51-column groups at arities 25 and 26, the same logical trace cutover after
removing selector variables.

The cutoff comes from same-shape, release-mode K=16 comparisons on a 16-core
Apple M4 Max host:

| Logical trace | Direct single-thread verify | Offloaded single-thread verify | Verifier speedup | Direct commit + prove | Offloaded commit + prove |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2^20` | 22.564 ms | 13.370 ms | 1.69x | 6.635 s | 6.603 s |
| `2^21` | 27.120 ms | 11.168 ms | 2.43x | 13.305 s | 13.191 s |

`2^20` therefore misses the 2x verifier gate. `2^21` is the first measured
shape to clear it while keeping total prover time within the 10% budget.

These timings were measured on the rows that preceded the catalog
regeneration under akita `db5efa20`. The regenerated hybrid catalogs switch
from direct to setup-offloaded rows at the same logical trace length; the
table was not re-measured.

Grouped planning calls Akita's `find_adapted_schedule`, which first tries the
selected scalar trace row's fold geometry, opening parameters, relation modes,
and direct/offloaded topology. If guided adaptation returns `UnsupportedSchedule`,
Akita automatically runs a full schedule search for the same request under the
same audited policy. The resulting row may use different trace fold geometry,
opening parameters, relation modes, or direct/offloaded topology. Invalid
requests and other errors propagate without this fallback.

Advice, bytecode, and field-valued precommits resolve their producer rows directly
from the dense catalogs. Commitment setup receives no trace chunk count and
performs no producer replanning. Grouped planning preserves the exact producer
profiles and audits their compatibility with the selected opening schedule.
Deployment directories must regenerate both dense catalogs for the conservative
policy; previous catalogs have a different policy digest and are rejected.

Jolt rejects batches with multiple full-width producers, or one full-width
producer and more than two auxiliary producers, before planner search. This
retains the supported field-increment batch: one full-width increment and at
most two bounded advice groups. Bounded-only requests that pass the admission
checks above can use Akita's automatic full-search fallback, including `Single`
requests and batches with more than two producers. They are not restricted to
the old chunked, two-producer fallback rule.

Akita owns the search limits and precommit-opening assignment enumeration;
Jolt does not impose the former separate opening-assignment product check.
Every auxiliary commitment's profile stays fixed through both searches, and
the resulting grouped row passes the usual schedule audit before entering the
setup-owned catalog. Preprocessing leaves the base artifacts intact; proving
and verification use the resulting frozen grouped row.

Regenerate all base catalogs from the planner with:

```sh
cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
```

Pass `k16`, `k256`, `w2r2`, `w4r2`, `w8r2`, `dense-bounded`, `dense-full`, or `dense`
as a selector to narrow regeneration to matching families. `k16-single` and
`k256-single` select only the corresponding standard single-chunk catalog.

Check complete artifact freshness without overwriting the catalogs:

```sh
cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules --check
```

This replans every selected row with the pinned Akita revision, renders canonical
artifacts in a temporary directory, and fails on any byte difference or missing
file. Ordinary catalog tests check admitted keys and coverage of production
trace geometry; key agreement alone does not establish schedule freshness.

Trace groups use native columns with arity `log_T + log_K`, without selector
variables. Production trace keys cover K=16 arities 16–34 with 51–64 columns.
The bounds follow the 32-bit bytecode PCs, 61-bit remapped RAM word addresses,
and 64-column row mask. The K=256 single-chunk keys `(num_vars, num_polys)` are
`(14,1)`, `(15,1)`, `(16,1)`, `(20,1)`, `(20,29)`, `(25,1)`,
`(28,27)`, `(29,27)`, and `(34,27)`, retained for adapter, benchmark, cutover,
advice, and forced-K tests. Each K=256 multi-chunk catalog retains the `(16,1)` roundtrip fixture alongside
the scalar diagnostic grid described above. K=16 fixtures also remain explicit
in `one_hot_keys`.
These K=256 keys are fixture and benchmark coverage, not a supported general
trace range. Other explicit configurations require a deployment-owned catalog
containing that exact shape. Grouped preprocessing rejects a missing final shape
during setup and reports the requested K, arity, and column count.

The catalogs match Akita revision
`318c322113dc66aeed31f9f3c640e84099ba043f`, as verified by the complete
artifact freshness check above.
Native trace batching establishes first-fold cycle locality. Recursive folds
inherit chunk ownership, align chunk bodies to the successor's source block
width, and keep the shared tail in the last owner. A reduction in chunk count
merges owners. Check artifact freshness after updating Akita and regenerate
catalogs if they differ.
