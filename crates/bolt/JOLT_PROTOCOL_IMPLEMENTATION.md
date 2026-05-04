# Jolt Protocol Implementation Plan

This is the stage-addition playbook for the Bolt implementation of Jolt. The
goal is to make every new protocol stage boring to add: first specify the stage
in MLIR, then lower it through the role/compute/CPU pipeline, then emit uniform
Rust artifacts, then wire arithmetic, real-data equivalence, tamper tests, and
performance gates.

## Current State

Commitment, Stage 1, Stage 2, and Stage 3 have the active compiler rails: protocol MLIR,
per-party lowering, compute/CPU lowering, generated Rust fixtures, and real-data
equivalence tests against `jolt-core`. After the full-field challenge repair,
the focused Stage 1/2/3 real-data equivalence gates pass on the current
`muldiv` fixture, including Bolt verifier acceptance, transcript parity, core
acceptance, core proof-component parity, and generated verifier tamper
coverage. The `jolt-bench` `bolt-stage` selector supports Stage 1 through Stage
5 as correctness-gated perf runs, and their `sha2-chain` release perf gates are
green (`ratio_vs_core ~= 0.79` for Stage 1, `~= 1.19` for Stage 2, `~= 0.89`
for Stage 3, `~= 1.06` for Stage 4, and `~= 0.97` for Stage 5). Stage 3 also
has monolithic generated verifier `verify_jolt_with_programs`
acceptance/tamper coverage and core acceptance/parity on `muldiv`. The default
checked-in verifier programs still describe the synthetic fixture shape, but
the generated top-level verifier can take real program plans for the `muldiv`
path.

Stage 4 now has protocol/lowering/generated Rust rails for register read/write
checking and RAM value checking. The MLIR includes the core transcript shape,
including `transcript.absorb_bytes` for the `ram_val_check_gamma` domain
separator, and lowers through concrete, party, compute, kernelized compute, and
CPU IR fixtures. The focused `muldiv` equivalence gate now runs real
Bolt-produced Stage 4 prover output and checks Bolt verifier acceptance,
transcript parity, core acceptance of the spliced Stage 4 proof, core/Bolt
artifact parity, generated Stage 4 verifier tamper rejection, and monolithic
generated verifier acceptance/tamper rejection. The `bolt-stage` selector
supports Stage 4 as a correctness-gated perf run. After the linear-time RAM LT
materialization, sparse register parallelization, split-eq/Gruen register cycle
rounds, reusable sparse bind buffers, and sparse read-selector optimizations,
the standard three-iteration gate passes the 1.2x target:
`perf/bolt-stage4-last.json` records core at `23.236ms`, Bolt at `24.522ms`,
and `ratio_vs_core ~= 1.06`.

Stage 5 now has protocol/lowering/generated Rust rails for instruction read
RAF, RAM RA claim reduction, and register value evaluation. The focused
`muldiv` equivalence gate runs real Bolt-produced Stage 5 prover output using
full-field opening inputs derived from Bolt Stage 2/4 artifacts, checks kernel
and generated verifier acceptance, compares the Stage 5 opening inputs and
observable proof artifacts against `jolt-core`, verifies transcript-state
parity through Stage 5, and covers representative generated verifier and
monolithic tamper rejection for Stage 5 sumcheck coefficients, output evals,
and points. The `bolt-stage` selector supports Stage 5 as a correctness-gated
perf run, and the Stage 5 `sha2-chain` perf gate is green. After grouping
duplicate instruction lookup keys and moving instruction-read table evaluation
onto the same prefix/suffix address-phase path used by core, adding full-field
small-scalar accumulation for suffix buckets, flattening RAF Q bucket
materialization, hoisting RAF shift scaling, avoiding Rayon overhead on late
address binds, and avoiding intermediate read-table message allocation, the
latest permissive-timeout three-iteration run in
`perf/bolt-stage5-last.json` records core at `100.891ms`, Bolt at `98.196ms`,
and `ratio_vs_core ~= 0.97`.

Stage 6 has the first compiler-owned IR/lowering slice for the six batched
instances: bytecode read RAF, booleanity, hamming booleanity, RAM RA
virtualization, instruction RA virtualization, and increment claim reduction.
The protocol MLIR models the nine pre-sumcheck transcript squeezes, the
cross-stage full-field input-claim wiring from Stages 1 through 5, booleanity
power placeholders, ordered sumcheck claims, output-opening obligations, and
the shared `jolt_core_stage6_aligned` batched driver. The focused
`commitment_ir` Stage 6 tests cover protocol schema validation, concrete
Fiat-Shamir threading, prover/verifier party projection, compute lowering,
prover-only kernel resolution, CPU lowering, compileable generated
prover/verifier Rust for the Stage 6 CPU plans, and checked-in full-prefix
generated prover/verifier crates. The generated verifier now checks Stage 6
output claims for booleanity, hamming booleanity, RAM RA virtualization,
instruction RA virtualization, increment claim reduction, and bytecode read RAF
when callers provide `Stage6VerifierData` with the bytecode preprocessing table
and entry bytecode index. The generated verifier derives the five bytecode Val
evaluations internally at the sumcheck address point. For correctness, the
`muldiv` equivalence gate now derives and passes that bytecode
verifier data, produces a Bolt Stage 6 proof from real trace witnesses, compares
its proof artifacts and transcript state against `jolt-core`, verifies it with
the standalone generated verifier, verifies it through the monolithic generated
verifier after the generated Stage 5 prefix, replays the `jolt-core` proof
through the generated Stage 6 prover CPU plan's proof-carrying kernel bridge,
runs the top-level generated prover `prove_jolt_with_programs` with real
commitment and Stage 1-6 CPU plans, verifies that monolithic prover proof with
the monolithic generated verifier, and rejects representative standalone and
monolithic Stage 6 sumcheck tampering. `jolt-kernels` now includes real Stage 6
prover executor slices for bytecode read RAF, booleanity, hamming booleanity,
RAM RA virtualization, instruction RA virtualization, and increment claim
reduction, plus final-claim tamper gates for those single-relation batches. The
Stage 6 prover path now avoids the earlier full `K*T` dense bytecode read RAF
construction by using a core-aligned address/cycle split, and booleanity uses a
specialized sparse degree-3 evaluator. The `bolt-stage` selector supports Stage
6 as a correctness-gated timing run; the release `muldiv` smoke passes the 1.2x
gate, and the documented `sha2-chain` three-iteration release perf gate is
green. `perf/bolt-stage6-last.json` records core at `2011.115ms`, Bolt at
`1266.910ms`, and `ratio_vs_core ~= 0.63`.

Stage 7 has generated prefix rails, standalone verifier coverage, and
monolithic generated prover/verifier coverage on real `muldiv` traces. The
equivalence gate compares Bolt-produced Stage 7 artifacts and transcript state
against `jolt-core`, verifies the spliced core proof prefix, replays the core
proof through the proof-carrying Stage 7 kernel bridge, runs the top-level
generated prover through Stage 7, and rejects representative Stage 7 tampering.
The generated crates and monolithic APIs now use the full-field transcript path
(`Transcript<Challenge = Fr>`), matching `jolt-core` after its
`JoltField::Challenge` aliases were changed to the full field for
`ark_bn254::Fr` and `TrackedFr`. The `bolt-stage` selector supports Stage 7,
and the `sha2-chain` smoke perf gate is green:
`perf/bolt-stage7-smoke.json` records core at `2.551ms`, Bolt at `2.701ms`,
and `ratio_vs_core ~= 1.06`.

The evaluation proof is wired into the generated monolithic prover/verifier on
that same full-field transcript path. The generated prover emits a Dory joint
opening proof when Stage 7 opening inputs are supplied, and the generated
verifier checks it when an `evaluation_setup` is present. The generated
monolithic program tables now pass the compiler-owned Stage 8 evaluation plan
into the prover and verifier helpers, so Dory batching uses the real trace's
opening-claim shape instead of a static fixture-shaped `STAGE8_PROGRAM`. The
`muldiv` equivalence gate covers generated Stage 8 acceptance, core acceptance
after substituting Bolt's joint opening proof, full Stage 8 transcript parity,
missing-proof/setup rejection, and unrelated Dory proof rejection. The
`bolt-stage` selector has a correctness-gated `stage8` timing path. Stage 8 is
correctness-green and perf-green after skipping zero one-hot cells during joint
polynomial materialization and moving modular Dory hint/opening routines onto
the same GLV vector operations used by core. `perf/bolt-stage8-smoke.json`
records the current `muldiv` `log_t = 10` smoke with core at `152.451ms`, Bolt
at `162.147ms`, and `ratio_vs_core ~= 1.06`. The documented `sha2-chain`
three-iteration release gate is also green: `perf/bolt-stage8-last.json`
records core at `796.183ms`, Bolt at `922.799ms`, and
`ratio_vs_core ~= 1.16`.

## Stage Algorithm

For a new stage `N`, do these steps in order. Do not treat synthetic tests as an
acceptance gate; they are only unit scaffolding before real trace data is wired.

1. **Scope the stage contract**
   - Identify the paper-level stage purpose, claims, sumchecks, transcript
     labels, opening obligations, and proof slots.
   - Identify the matching `jolt-core` verifier/prover fields and the exact
     prefix boundary where Bolt artifacts can be spliced into a core proof.
   - Write down any new data dependencies from earlier stages as typed opening
     inputs, not ad hoc runtime lookups.

2. **Model protocol MLIR first**
   - Encode protocol facts under `protocol`, `piop`, `poly`, `field`,
     `transcript`, `commit`, and `pcs`.
   - Keep Jolt-specific facts inside `protocols/jolt`; generic dialect schemas
     must not learn Jolt-only names.
   - Use SSA operands/results for claims, points, transcript state, and opening
     flow. Attributes are for stable metadata such as names, domains, counts,
     labels, policies, and proof slots.
   - Add validation before lowering: ordered claims, compatible opening
     equalities, expected point arities, round counts, degree, proof-slot
     uniqueness, transcript threading, and role visibility.

3. **Project roles and lower through MLIR**
   - Lower `protocol -> concrete` with explicit Fiat-Shamir state.
   - Project `concrete -> party` for prover and verifier.
   - Lower each party to `compute`, preserving SSA dataflow.
   - Resolve prover kernels as a compute-to-compute pass. Verifier IR must not
     receive kernels.
   - Lower `compute -> cpu`. CPU IR is still MLIR; Rust is only emitted after
     this point.

4. **Emit uniform Rust artifacts**
   - The compiler output is a pair of role crates:
     `jolt-prover` and `jolt-verifier`.
   - Stage modules live at `src/stages/<stage>.rs` in both crates.
   - `assemble_jolt_generated_crates` is the canonical crate assembly rail; it
     must emit manifests, `src/lib.rs`, `src/stages/mod.rs`, and every stage
     module in a shape that `cargo check` accepts.
   - `jolt-prover` may import coarse, unstable CPU kernels from
     `jolt-kernels`.
   - `jolt-verifier` must import only audit-scope modular crates and local
     generated modules; it must not import `jolt-kernels`, `jolt-core`,
     `jolt-equivalence`, `jolt-bench`, or tracer internals.
   - Generated verifier code should be explicit, human-readable glue around
     modular verifier APIs. If verifier logic is hidden in `jolt-kernels`, the
     stage is not audit-stable.

5. **Wire arithmetic**
   - Start with the coarsest prover kernels that can plausibly reach performance
     parity with `jolt-core`.
   - Implement kernels below the dialect boundary. Kernel internals can be
     optimized aggressively, but the protocol order, claims, transcript labels,
     and opening obligations must remain owned by MLIR.
   - Reuse or improve modular primitive crates for verifier-side proof
     structure and checking. Treat `jolt-kernels` as prover-side implementation
     detail, not soundness-critical verifier infrastructure.

6. **Add real-data equivalence**
   - Use real traced program data for the acceptance gate. Current standard
     workload is `sha2-chain` for perf and real program fixtures such as
     `muldiv` for focused semantic debugging.
   - Run Bolt prover and Bolt verifier on the same trace and assert proof
     acceptance plus transcript-state equality through the new stage boundary.
   - Splice Bolt stage artifacts into the matching `jolt-core` proof prefix and
     assert the core verifier accepts through the same boundary.
   - Compare Bolt vs core transcript states and observable proof components
     step-for-step through the stage: sumcheck round polynomials, opening
     claims, normalized points, commitments, and proof slots as applicable.

7. **Harden with negative tests**
   - Add tamper tests for each new verifier obligation: sumcheck coefficients,
     claimed output, opening evaluations, opening points, equality constraints,
     batching order, transcript labels, and proof-slot shape.
   - Each tamper should fail in the generated/Bolt verifier before any core
     fallback is consulted.
   - Add schema-level negative tests for malformed IR that could otherwise
     erase or reorder a soundness obligation.

8. **Optimize after correctness**
   - Use `jolt-bench`, `jolt-profiling`, and targeted instrumentation in
     `jolt-core`/Bolt kernels to isolate gaps.
   - Add the stage to the `bolt-stage` bench selector before claiming perf
     parity.
   - Perf gates use at most three timing iterations for iteration speed; the
     selector exits nonzero when Bolt exceeds the ratio gate (`--max-ratio`,
     default `1.2`).
   - A stage is performance-complete only when Bolt prover time for the newly
     wired stage is within 20% of the corresponding `jolt-core` stage on the
     agreed workload. Re-run correctness after any accepted optimization.

9. **Mark the stage done**
   - Check in MLIR fixtures for protocol/concrete/prover party/verifier
     party/prover compute/verifier compute/prover CPU/verifier CPU.
   - Check in generated Rust fixtures for both role artifacts.
   - Keep `jolt-equivalence` gates named by stage and using real data.
   - Update the stage status matrix and only then start scoping the next stage.

## Acceptance Criteria

A stage is done only when all conditions below are true:

- **MLIR correctness**: protocol, concrete, party, compute, kernelized compute,
  and CPU schemas pass for both roles; verifier IR is kernel-free.
- **Artifact correctness**: generated prover and verifier modules compile under
  the canonical `jolt-prover`/`jolt-verifier` layout, including whole-crate
  `cargo check` of the generated manifests and stage module graph.
- **Import policy**: generated verifier imports only modular audit-scope crates;
  generated prover does not import `jolt-core` or equivalence-only crates.
- **Internal real-data parity**: Bolt prover output is accepted by the Bolt
  verifier on real trace data, and prover/verifier transcript states match
  step-for-step through the stage boundary.
- **Core acceptance**: `jolt-core` accepts a proof prefix whose stage artifacts
  were produced by Bolt.
- **Core transcript/artifact parity**: Bolt and `jolt-core` match transcript
  states and observable stage artifacts through the implemented boundary.
- **Tamper coverage**: generated verifier rejects representative mutations for
  every new soundness obligation introduced by the stage.
- **Performance**: Bolt prover time for the new stage is within 20% of
  `jolt-core` on the agreed `sha2-chain` workload, with perf gates capped at
  three iterations.

## Required Gates

Use focused gates during development, then run the full stage gate before moving
on:

```bash
cargo nextest run -p bolt --cargo-quiet
cargo check -p jolt-equivalence --tests --quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
cargo nextest run -p jolt-kernels --cargo-quiet
```

For stage perf:

```bash
cargo run --release -p jolt-bench --bin bolt-stage -- \
  --program sha2-chain --stage <stage> --log-t 16 \
  --num-iters 16 --iters 3 --warmup 1 \
  --json perf/bolt-<stage>-last.json
```

Run `jolt-core` first when comparing wall time. If Bolt exceeds a 10x timeout
relative to core, stop timing and profile rather than waiting for a full run.
