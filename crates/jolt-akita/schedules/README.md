# Jolt Akita schedule artifacts

This directory contains Jolt's base Akita schedule catalogs as canonical
`.aks` files. They are runtime data, not generated Rust modules and not
embedded into the executable.

Application preprocessing loads the three files once, wraps the resulting
`AkitaScheduleArtifacts` in `Arc`, and passes that immutable bundle explicitly
to every `AkitaSetupParams` constructor. Production deployments should call
`AkitaScheduleArtifacts::from_directory` with a versioned, deployment-owned
path. `from_default_directory` is a host/dev helper: it reads
`JOLT_AKITA_SCHEDULE_DIR`, falling back to this packaged source directory.
Protocol setup and verification never discover files or consult the
environment.

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
