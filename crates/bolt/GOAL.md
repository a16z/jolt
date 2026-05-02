# Bolt Jolt Goal

Bolt should produce a full Jolt implementation from compiler-owned protocol
IR. The final checked-in artifacts are generated Rust role crates under
`crates/jolt-prover` and `crates/jolt-verifier`; they must faithfully match
`jolt-core` semantics while using the modular primitive crates and preserving
the MLIR/Bolt boundaries from the paper.

## Source Of Truth

- `../bolt/paper.pdf` and `../bolt/mlir/paper-alignment.md`: formal Bolt
  design and dialect philosophy. Treat these as the oracle for compiler
  structure, updating implementation details only when Jolt protocol needs
  force a defensible refinement.
- `crates/bolt/JOLT_PROTOCOL_IMPLEMENTATION.md`: stage implementation
  algorithm and acceptance criteria. New work should follow this playbook
  exactly unless the file is deliberately updated first.
- `crates/bolt/README.md`: current compiler architecture, dialect boundaries,
  and generated-artifact shape.
- `crates/bolt/TESTING.md`: concrete correctness, generated-artifact,
  equivalence, tamper, and perf gates.
- `TASKS.md` and `PERF_TASKS.md`: repository-level correctness/performance
  program counters.

## Target Protocol Scope

Implement Jolt on Bolt cumulatively:

1. Commitment phase.
2. Stages 1 through 7.
3. Evaluation proof.

Each stage must extend the same end-to-end chain. The prover/verifier artifacts
should not be isolated examples; after every stage, `crates/jolt-prover` and
`crates/jolt-verifier` should represent the complete implemented prefix.

## Compiler Boundaries

- Protocol knowledge lives in `crates/bolt/src/protocols/jolt` and MLIR
  dialects/passes, not in generated Rust control flow.
- Generic dialects (`protocol`, `piop`, `poly`, `field`, `transcript`,
  `commit`, `pcs`, `party`, `compute`, `cpu`) must stay generic; Jolt-only
  names and parameters should not leak into their schemas except as ordinary
  attributes or SSA values carried by the Jolt protocol definition.
- Lowering order remains `protocol -> concrete -> party -> compute -> cpu ->
  Rust`.
- Rust emission is the final non-MLIR target. Anything before emission is a
  dialect op, validation pass, analysis, rewrite, or lowering.
- Prover code may call coarse CPU kernels in `crates/jolt-kernels` while we
  bootstrap performance. Those kernels are below the dialect boundary and can
  be refined into smaller compute/cpu ops later.
- Verifier code must stay audit-stable: no `jolt-kernels`, no `jolt-core`, no
  equivalence-only imports, and no tracer internals. It should be readable glue
  over modular audit-scope crates such as `jolt-sumcheck`, `jolt-openings`,
  `jolt-dory`, `jolt-poly`, `jolt-field`, and `jolt-transcript`.

## Generated Artifacts

- `crates/jolt-prover`: canonical generated prover crate for the implemented
  Jolt prefix. Stage modules live in `src/stages/<stage>.rs`.
- `crates/jolt-verifier`: canonical generated verifier crate for the same
  implemented prefix. Stage modules mirror the prover stage names.
- `bolt::assemble_jolt_workspace_generated_crates` and
  `bolt::write_jolt_generated_crates` are the compiler rails for materializing
  these crates under `crates/`.
- The generated crates expose a stage registry plus top-level `prover.rs` and
  `verifier.rs` APIs synthesized by the Rust artifact emitter. These files are
  not hand-maintained.
- The top-level API generator derives stage fields, proof conversion, executor
  bounds, verifier inputs, and module wiring from emitted role artifacts and
  generated Rust item names. Avoid adding hand-coded Jolt-stage branches here.
- `jolt-verifier` owns `JoltProof` and all verification structure. It must not
  depend on `jolt-prover`. `jolt-prover` may import verifier-owned proof types
  to construct a proof, but it must not import verifier stage internals.
- `crates/jolt-equivalence` and `crates/jolt-bench` should discover the
  implemented prefix from the generated stage registry rather than temp files
  or hand-written artifact lists.
- Regenerate the checked-in crates through the compiler artifact rail with
  `JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules --cargo-quiet`.

## Stage Completion Criteria

A stage is done only when all criteria below pass cumulatively through that
stage boundary on real trace data:

- MLIR schemas pass for protocol, concrete, party, compute, kernelized compute,
  and CPU IR for both roles.
- Generated `jolt-prover` and `jolt-verifier` crates compile and expose the
  full implemented prefix.
- Bolt prover output is accepted by the Bolt verifier.
- Bolt prover and verifier transcript states match step-for-step.
- Bolt proof components can be spliced into the matching `jolt-core` proof
  prefix and accepted by `jolt-core`.
- Bolt and `jolt-core` transcript states and observable proof components match
  through the stage boundary.
- Generated verifier tamper tests reject representative mutations for every
  new soundness obligation introduced by the stage.
- Prover performance for the newly added stage is within 20% of `jolt-core` on
  the agreed `sha2-chain` workload, with perf gates capped at three timing
  iterations.

Synthetic fixtures are useful while building a stage, but they never satisfy a
stage completion criterion.

## Harnesses

- `crates/jolt-equivalence`: real-data correctness oracle. It should import
  `jolt-prover` and `jolt-verifier` and grow one cumulative stage gate per
  implemented prefix.
- `crates/jolt-bench`: perf oracle. It should benchmark the same generated
  prover stage prefix against `jolt-core`, using the generated stage registry
  rather than hard-coded temp artifacts.
- `crates/jolt-kernels`: temporary coarse prover kernels. Optimize here for
  performance parity without moving protocol semantics out of MLIR.
- `crates/jolt-witness`: primitive witness/oracle construction helpers for
  generated prover code.
- `jolt-core`: reference implementation and semantic oracle. It is not a code
  quality template for Bolt, but it is the ground truth for transcript/proof
  parity until Bolt is independently audited.

## Current Position

Commitment, Stage 1, Stage 2, and Stage 3 have compiler/codegen rails in
progress. Before adding further stages, keep the generated crates, verifier
pipeline, equivalence hooks, and bench hooks aligned with the completion
criteria above so every subsequent stage lands through the same repeatable
process.
