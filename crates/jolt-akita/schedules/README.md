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
preserved in the joint opening.

During preprocessing, Jolt adapts rows whose shapes depend on advice, field
increments, or direct committed-program sizes. Those rows are merged with the
relevant base catalog.
The resulting exact catalog is serialized inside `AkitaVerifierSetup`, so a
transported verifier setup does not depend on process-global state or on these
source-tree files.

The one-hot artifacts are hybrid catalogs. A logical trace shorter than
`2^21` uses a direct schedule. A trace of `2^21` cycles or longer uses a
setup-offloaded schedule. Akita uses K=16 committed chunks at every trace
length, with catalog coverage through `2^30`. Virtual lookup chunks are 16 bits
below `2^25` and 32 bits at or above it. The K=256 catalog remains available
for explicitly configured layouts. This is an offline catalog policy: proving and verification simply
resolve the exact admitted row and never choose a mode dynamically.

Each K=16 and K=256 family has W2R2, W4R2, and W8R2 multi-chunk companion
catalogs. The selected profile splits the root and first recursive fold into
two, four, or eight chunks, while later folds remain single-chunk; their
smallest admitted physical arity is 16 variables. The original one-hot
catalogs and the dense advice and committed-program catalog remain
single-chunk. Existing four-file directories continue to support `Single`;
selecting a profile whose companion catalog is absent fails during setup.
Grouped precommit setups inherit the selected trace profile.

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

Grouped planning first preserves the selected trace row's fold geometry,
opening parameters, relation modes, and direct/offloaded topology, adapting
only the auxiliary object profiles and the sizes they induce. Requests with
only bounded dense objects fail if that fixed geometry is infeasible.

A full-width field increment can require different trace fold geometry. If
guided planning returns `UnsupportedSchedule` for the supported field batch—
exactly one full-width field increment and at most two bounded advice groups—
preprocessing runs the full planner under the same audited policy. Larger
batches and batches with multiple full-width objects retain the guided-planning
rejection, including its opening-assignment budget. Every auxiliary commitment's
profile stays fixed, and the resulting grouped row passes the usual schedule
audit before entering the setup-owned catalog. Other errors propagate. The
checked-in base catalogs are unchanged; proving and verification use the
resulting frozen grouped row.

Regenerate all base catalogs from the planner with:

```sh
cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
```

Pass `k16`, `k256`, `w2r2`, `w4r2`, `w8r2`, `dense-bounded`, `dense-full`, or `dense` as a final
argument to narrow regeneration to matching families. `k16-single` and
`k256-single` select only the corresponding standard single-chunk catalog.
