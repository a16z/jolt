# Jolt Protocol Implementation Notes

The original stage-by-stage bring-up plan has been completed for the first
full-field, non-zk Jolt-on-Bolt implementation. The active long-haul goal now
lives in `GOAL.md`: make the generated Jolt verifier much smaller, cleaner,
and better hardened.

This file keeps the durable implementation rules that should continue to guide
that cleanup.

The companion genericity goal lives in `GENERIC_PROTOCOL_GOAL.md`. It defines
the rule that Jolt is a protocol package over Bolt, not a special case inside
generic IRs, passes, validation, or Rust artifact assembly.

## Permanent Compiler Rules

- Protocol facts live in `crates/bolt/src/protocols/jolt` and typed MLIR/plan
  structures, not in generated Rust control flow.
- Generic dialects should remain generic. Jolt-only names and parameters may be
  ordinary attrs or SSA values carried by the Jolt protocol definition, but
  they should not become hidden assumptions in generic lowering code.
- Generic artifact assembly should consume protocol config and ordered stage
  artifacts; Jolt artifact names, stage enums, relation mappings, and eval-proof
  composition belong under `crates/bolt/src/protocols/jolt`.
- Lowering order remains:

```text
protocol -> concrete -> party -> compute -> cpu -> Rust
```

- Rust emission is the final target. Before emission, behavior should be
  represented as dialect ops, validation passes, analyses, rewrites, lowerings,
  or typed plan extraction.
- Prover code may use coarse CPU kernels while performance work continues.
  Those kernels are below the dialect boundary.
- Verifier code must stay kernel-free and audit-stable. It should use modular
  verifier crates and generated plan data, not `jolt-kernels` or `jolt-core`.

## Verifier Cleanup Algorithm

For every verifier cleanup iteration:

1. Measure generated verifier LOC, stage LOC, duplicate plan structs, duplicate
   helper functions, forbidden imports, and string-dispatch sites.
2. Pick one duplication class or compiler hygiene issue.
3. Move generic mechanics into shared verifier runtime only when protocol
   semantics remain explicit in MLIR-derived typed plan data.
4. Regenerate checked-in `jolt-prover` and `jolt-verifier` artifacts through the
   canonical artifact rail.
5. Run schema, import, equivalence, and tamper gates.
6. Keep the change only when generated code is easier to read and no semantic
   oracle weakens.

## Do Not Regress

- Verifier CPU IR must not contain kernel attrs or prover-only ops.
- Generated verifier Rust must not import `jolt-kernels`, `jolt-core`,
  `jolt-prover`, `jolt-equivalence`, `jolt-profiling`, or tracer internals.
- Transcript state must be explicitly threaded through MLIR.
- Opening batches must preserve ordered claim lists.
- Opening equality checks must reject incompatible claim metadata.
- Sumcheck relation dispatch should be typed or explicitly allowlisted.
- Full-field transcript challenges are the intended path:
  `Transcript<Challenge = Fr>`.

## Regeneration Rail

Checked-in generated role crates are not hand-maintained. Regenerate them with:

```bash
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules --cargo-quiet
```

Then run the gates in `TESTING.md`.
