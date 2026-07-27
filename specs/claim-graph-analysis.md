# Spec: Claim-Graph Analysis

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Markos Georghiades             |
| Created     | 2026-07-27                     |
| Status      | implemented                    |
| PR          | #1705                          |

## Summary

The highest-severity protocol bug class in Jolt is a mis-wired or unbound claim: a sumcheck output that no downstream relation consumes and no PCS opening checks is a prover-controlled free variable, and no existing gate (lints, fuzzing, coverage, mutation, tamper manifest) can detect it, because a missing check has no code to exercise. jolt-claims already describes the protocol declaratively — every relation implements `SymbolicSumcheck` with typed input/output expressions, and the claim-struct derives parse producer/consumer wiring (`#[relation(..)]`, `#[opening(.., from = ..)]`) on every claim struct. This spec turns that description into a whole-protocol static analysis: build the claim-flow graph from two independent sources (derive-emitted edge metadata and the symbolic expressions), and check it — no dangling claims, unique production, declared producers match actual, schedule consistency, Dory/Akita PIOP correspondence — as integration tests in `crates/jolt-claims/tests/`.

## Intent

### Goal

A claim-flow graph checker covering both PIOPs (Dory and Akita), fed by derive-generated metadata so the graph updates automatically whenever claim wiring changes.

Mechanism, in two layers plus the vertex set:

1. **Derive-emitted edges (struct level, config-independent).** `jolt-claims-derive`'s existing `OutputClaims`/`InputClaims` derives additionally emit an implementation of a new `ClaimAdjacency` trait: a `const` list of `ClaimEdge { polynomial, arity: Scalar | Optional | Family, relation }` records, produced from the same tokens the derive already parses. Editing a claim struct regenerates its edges in the same compile; the metadata cannot drift from the resolution code because both come from one macro invocation. Hand-written aggregate claim structs implement `ClaimAdjacency` manually (short, explicit, and cross-validated below).
2. **Symbolic-expression edges (expansion level, config-dependent).** In the test tree, a fold over `input_expression()` / `output_expression()` (public `Expr.terms` / `Term.factors`) collects the opening ids each relation instance consumes and produces under a concrete protocol configuration, mirroring the existing `expected_output_openings`.
3. **Vertex set (test tree only).** A test-local `ProtocolVertex: SymbolicSumcheck` trait declaring how each relation instantiates (`fn instances(config) -> Vec<Self::Shape>`; conditional relations return no instances and are carried as declaration-only records for the exhaustiveness backstop), implemented for each relation type, legal because the trait is local. PIOP membership comes from the manifest's `shared` / `dory` / `akita` sections — one source of truth rather than a per-impl const. A vertex's produced/consumed sets union the expression openings with the struct-declared scalar/optional edges: *forwarded* openings — opened at the relation's point, constrained downstream — are a documented protocol pattern (`wire_output_openings` in the verifier's relation driver). The graph also carries *alias edges*: `(aliased, source)` wire-copy pairs the stage drivers enforce (`aliased_output_openings`), assembled from the same jolt-claims consistency-openings geometry the verifier uses; a pair's members discharge together, and pairs citing unproduced ids are violations. The graph carries no stage data: many stagings are valid for one graph, so staging is an assignment applied to the graph by a scheduler, not a property of it (see Non-Goals). One `claim_graph_vertices!` `macro_rules!` lists the vertex set in `shared` / `dory` / `akita` sections, and a generic `vertices::<S>(config)` extractor erases the non-object-safe trait into plain `VertexRecord { in_edges, out_edges, degree_bound, rounds }` data; everything downstream is ordinary graph algorithms over owned records.

Checks, run per (PIOP, config) cell — Dory and Akita, each under three configurations (small; with both advice kinds; with a committed program). The planned (Dory, zk) cell collapsed during implementation: stage 8 computes the opening-id batch before its zk branch and BlindFold binds the same list value-level, so the zk graph is id-identical to clear (documented on the Dory sink builder with code citations):

- **Unique production**: each opening id is produced by exactly one relation instance; collisions report both producers.
- **Resolution**: every consumed id is in the produced set or a source (PCS-committed polynomials, advice, baked publics), and the producer relation embedded in the id (a `JoltOpeningId` is a `(polynomial, relation)` pair) matches the actual producer.
- **No dangling claims**: `produced − (consumed ∪ sinks ∪ alias-closure ∪ terminal_allowlist)` is empty. Sinks per PIOP are computed from the same public functions the verifier uses — Dory from `final_opening_polynomial_order`/`final_opening_id` (the zk path binds the same id list value-level, so no separate zk sink set exists), Akita (`SinkKind::PackedOpened`) from the lattice packing geometry: native one-hot batch columns, advice byte objects, and per-chunk committed-program lanes — so the sink set tracks the batch order in the same commit. Sink matching is on the full `JoltOpeningId`, not the polynomial: the batch opens each polynomial against one specific relation's claim, and an opening of the right polynomial bound to a different relation's claim leaves the produced claim dangling. The `PcsOpened` sink is typed to reject virtual-polynomial ids: virtual claims have no commitments and must reach committed claims through the claim-reduction chain, so an unconsumed virtual claim is always an error and can never be allowlisted as PCS-terminal. The allowlist starts empty; entries require written justification and may only shrink. Outcome: it stayed empty — every first-run finding resolved into a typed mechanism (exact-id PCS sinks, forwarded openings, alias equality, finalize-at-cycle-handoff termini, staged wire openings), each modeled with source citations rather than allowlisted.
- **Acyclicity**: Kahn's toposort over produce→consume edges; the graph must be a DAG, and the toposort's generations give the canonical layering used by the `Display` renderer.
- **Struct/expression cross-validation**: the derive-emitted family-level edges and the expression-level edges must describe the same graph (expression ids collapse to family granularity for comparison). Divergence means a claim struct and its relation's algebra disagree — the drift class CLAUDE.md's hand-maintained sync invariants warn about.
- **Cross-PIOP correspondence**: under one configuration, the Dory and Akita graphs must share an identical common subgraph, and their symmetric difference must equal the substitution set documented in `specs/lattice-claims.md`, asserted by name.
- **Exhaustiveness backstop**: every `JoltRelationId` variant is covered by a registered sumcheck or an allowlist entry; unregistered relations fail by name.

`ClaimGraph` implements `Display`, rendering the DAG for the terminal: Kahn generations in topological order, each vertex with its in/out edges, then the source and sink sets. That rendering is also the golden-snapshot serialization — one format for humans and for the pinned artifact, so the pretty-printer and the snapshot cannot drift. Snapshots per (PIOP, config) use env-var regeneration following the `jolt-claims-derive` testdata convention, so any protocol rewiring appears as a reviewable diff; check failures print the offending vertex's neighborhood via the same renderer, and an ignored `dump_claim_graph` test exists for on-demand inspection (`cargo nextest run -p jolt-claims dump_claim_graph --run-ignored all --no-capture`).

### Invariants

- **Minimal src surface**: `crates/jolt-claims/src` gains only the `ClaimAdjacency` trait and `ClaimEdge` type (declarations, no logic); everything else lives in `crates/jolt-claims/tests/` and `jolt-claims-derive`. No protocol behavior changes anywhere; the derives' existing generated impls are byte-identical (guarded by the expansion snapshot tests).
- The analysis consumes only public API; if a relation's `Shape` is not publicly constructible, the fix is a per-case visibility change, not new infrastructure.
- Metadata emission and claim-struct semantics come from the same macro parse — no separately maintained wiring tables.
- Both PIOPs are first-class: every check runs for Dory and Akita, and the Akita×zk cell is rejected by construction (one PIOP per compiled verifier).

### Non-Goals

- Proving the constraints encode RISC-V semantics (z3-verifier and the Lean extraction own that layer) or witness-level soundness probing.
- Opening-point binding. The graph verifies that every claim *reaches* the PCS batch or a consumer at id level; whether each polynomial's own opening point is correctly embedded into the unified point (the Lagrange scaling in stage-8 assembly) is value-level verifier semantics, exercised by the tamper manifest's point-tampering targets. A green graph check does not assert PCS binding correctness.
- Index-level family-length agreement beyond what the shared dimension objects already guarantee; the analysis works at family granularity plus expression expansion under fixed configs.
- Staging validation. The graph deliberately carries no stage assignments (many stagings are valid for one graph). The planned follow-up, after this analysis is stable: a jolt-verifier integration test that takes the claim graph, applies the verifier's actual staging to it, and checks validity (every edge's producer staged no later than its consumer, sinks reachable). The graph builder is structured so it can be exposed to that test when the time comes.
- Automatic relation enumeration (no `inventory`/linker registration: explicit manifests with exhaustiveness backstops, per this repo's idiom).

## Evaluation

### Acceptance Criteria

- [x] `ClaimAdjacency` is emitted by both derives; the derive crate's expansion snapshots cover the new output, and hand-written aggregates have manual impls verified by the cross-validation check.
- [x] All checks above pass for every (PIOP, mode, config) cell, or fail with named ids/relations; the terminal allowlist entries (if any) carry written justifications.
- [x] Golden snapshots exist per (PIOP, config) with env-var regeneration, serialized via `ClaimGraph`'s `Display` (layered, human-readable); the ignored dump test renders the graph on demand.
- [x] CI runs both test targets: `claim_graph_dory` under default features and `claim_graph_akita` under `--features akita` (`required-features` on the target).
- [x] First-run findings are triaged in the PR description: every dangling claim found is either a fixed bug, an allowlisted terminal with justification, or a documented false positive with a checker fix.
- [x] `cargo clippy --all --features host --all-targets` and `--features host,zk` stay clean; full workspace `cargo nextest run --workspace` (minus z3-verifier locally) passes.

### Testing Strategy

The deliverable is tests. The derive changes are additionally covered by the existing expansion snapshots, error-path tests, and the nightly mutation target for `jolt-claims-derive`. The checker's own logic gets unit tests with synthetic graphs (a planted dangling claim, a planted double-producer, a planted cycle) so the checks are demonstrated to fail — not just to pass on the real protocol.

### Performance

Test-time only: graphs of a few hundred edges, microseconds per cell; no measurable CI cost. Zero runtime impact on prover or verifier (the emitted metadata is `const` data on claim structs; no codegen changes to existing impls).

## Design

### Architecture

```
crates/jolt-claims/src:            ClaimAdjacency trait + ClaimEdge type (only src addition)
crates/jolt-claims-derive:         derives additionally emit ClaimAdjacency impls
crates/jolt-claims/tests/
  claim_graph_dory.rs               #[path] wrapper, default features
  claim_graph_akita.rs              #[path] wrapper, required-features = ["akita"]
  claim_graph/
    mod.rs                         Piop, ProtocolVertex (test-local)
    registry.rs                    claim_graph_vertices! sections + vertices::<S>() extractor
    graph.rs                       ClaimGraph build, checks, Kahn toposort, Display renderer
    snapshots/                     golden edge lists
```

The registry's `shapes(config)` mapping is the irreducible hand-written knowledge (a derive sees one struct; no macro can enumerate the protocol or construct `Shape`s from a config). It changes when relations are added or reshaped — rare — while claim wiring inside relations, the high-churn part, tracks automatically via the derives.

### Alternatives Considered

- **Test-only registry, zero src additions** (edges extracted solely from expressions): rejected because struct-level wiring changes would be checked only through the expression layer, losing the two-independent-views cross-validation and the automatic-tracking property that motivated the derive route. Kept as fallback if the src additions are vetoed in review.
- **`inventory`/linkme self-registration**: rejected — link-time constructors are invisible to review and can silently drop registrations; explicit manifests with exhaustiveness backstops match the repo's tamper-manifest/floors idiom.
- **Analysis in jolt-verifier's test tree**: rejected — jolt-claims owns every type involved; the verifier only contributes sink lists, which are public.

## Documentation

This spec, plus a paragraph in the future book dev section (deferred with the lint-campaign docs until `specs/test-quality-ci.md`'s dev section lands).

## Execution

1. `ClaimAdjacency`/`ClaimEdge` in src; derive emission + snapshot updates + manual impls for hand-written aggregates. Open item to resolve here: `ClaimEdge` must be `const`-constructible, and payload-carrying opening variants (`OpFlags(CircuitFlags::..)`) embed payload values — if the payload types are not const-constructible, the edge uses a family-level polynomial identifier instead of a full `JoltOpeningId`, and the cross-validation collapses expression ids to that granularity.
2. Registry, extractor, graph builder, and the checks for (Dory, clear) under one config; planted-defect unit tests for the checker.
3. Remaining cells: zk sink set, Akita target with lattice relations and packed sinks, cross-PIOP correspondence, config matrix.
4. Goldens, CI wiring, first-run triage (fix, allowlist with justification, or checker fix), spec status update.

## References

- `specs/symbolic-sumcheck.md` — the symbolic relation layer this analysis consumes.
- `specs/lattice-claims.md` — the documented Dory/Akita substitution set the correspondence check asserts.
- `specs/self-contained-sumcheck-relations.md`, `specs/sumcheck-instance-data-model.md` — relation data-model background.
- `specs/verifier-closure-lints.md` — companion implementation-level hardening; this spec targets the protocol level.
