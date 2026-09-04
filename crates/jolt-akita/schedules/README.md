# Jolt Akita schedule artifacts

This directory contains Jolt's base Akita schedule catalogs as canonical
`.aks` files. They are runtime data, not generated Rust modules and not
embedded into the executable.

`AkitaScheduleArtifacts::from_default_directory` reads this directory by
default. Set `JOLT_AKITA_SCHEDULE_DIR` to load the same three file names from a
deployment-owned storage location, or attach bytes explicitly with
`AkitaSetupParams::with_schedule_artifacts`.

During preprocessing, Jolt plans rows whose shapes depend on advice or direct
committed-program sizes. Those rows are merged with the relevant base catalog.
The resulting exact catalog is serialized inside `AkitaVerifierSetup`, so a
transported verifier setup does not depend on process-global state or on these
source-tree files.

Regenerate all base catalogs from the planner with:

```sh
cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
```

Pass `k16`, `k256`, or `dense` as a final argument to regenerate one family.
