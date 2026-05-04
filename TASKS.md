# Tasks

The active Bolt/Jolt task is documented in `crates/bolt/GOAL.md`.

Current focus:

```text
make the Bolt-generated jolt-verifier compact, human-readable, auditable,
security-hardened, and driven by explicit MLIR-derived verifier plans
```

Supporting docs:

- `crates/bolt/GOAL.md`: long-haul verifier cleanup objective, LOC targets,
  hardening plan, and definition of done.
- `crates/bolt/GENERIC_PROTOCOL_GOAL.md`: generic protocol cleanup objective,
  Bolt/Jolt boundary rules, and genericity acceptance criteria.
- `crates/bolt/JOLT_PROTOCOL_IMPLEMENTATION.md`: durable compiler-boundary
  rules after the original stage-by-stage bring-up.
- `crates/bolt/TESTING.md`: equivalence, tamper, MLIR, import, LOC, and
  readability gates.
- `crates/bolt/README.md`: compact orientation for the Bolt crate.

The old root task list for the retired compiler/runtime path has been cleared
to avoid confusing it with the active Bolt verifier-pipeline cleanup.
