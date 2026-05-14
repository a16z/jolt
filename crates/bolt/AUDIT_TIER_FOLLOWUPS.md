# Audit-Tier Follow-Ups (S2 — S6)

This is the implementation plan for the post-S1 verifier-cleanup track. It is
a companion to `crates/bolt/GOAL.md`. Read GOAL.md first; this document
assumes the `Audit Tiers` framing introduced there.

S1 (the audit-tier split that landed in PR #1523) reframes the shared
verifier runtime as two explicitly-bounded tiers:

```text
Tier A (Bolt verifier runtime)        stages/common.rs           1,265 LOC
Tier B (audited Jolt verifier core)   stages/jolt_relations.rs     638 LOC
Tier C (generated stage data + verifier.rs)                      6,430 LOC
```

S2 — S6 below progressively shrink Tier B and Tier C by lifting today's
hand-written Rust into MLIR vocabulary. Each slice is intended to be one
reviewable PR stacked on top of the previous one.

## Guiding principle

> Most code in the verifier templates is an *interpreter* for plans, not
> protocol math. Lift the rest of the protocol math into MLIR vocabulary so
> the interpreter is the only thing humans need to audit.

Concrete corollary: **Tier A should not be per-protocol generated code at
all**. It is a small, generic interpreter that happens to be checked into
each protocol's verifier crate by historical accident. S2 corrects this.

The remaining slices (S3 — S6) close the gap between "generic interpreter"
and "Jolt-specific math evaluators" by extending the compute dialect with
the polynomial primitives, point reorderings, indexed-eval addressing, and
relation-expression vocabulary that Tier B currently fills with hand-written
Rust.

## Scoreboard (target trajectory)

| Slice | Tier A | Tier B | Tier C | jolt-verifier total | Notes |
|-------|-------:|-------:|-------:|--------------------:|-------|
| post-S1 (today) | 1,265 | 638 | 6,430 | 7,905 | hard ceilings: A 1,400 / B 700 / C surface 6,100 |
| post-S2 | ~50 | 638 | 6,430 | ~6,640 | Tier A moves to `bolt-verifier-runtime` crate |
| post-S3 | ~50 | ~500 | 6,430 | ~6,500 | poly, point reordering, gamma-power vector ops |
| post-S4 | ~50 | ~350 | 6,500 | ~6,500 | typed indexed-eval addressing |
| post-S5 | ~50 | ~290 | 6,700 | ~6,700 | relations as typed plans (Tier C grows) |
| post-S6 | ~50 | ~50 | 6,800 | ~6,800 | bytecode encoding as typed plans (optional) |

LOC numbers are targets, not contracts. Each slice should re-baseline the
ceilings in `crates/bolt/tests/verifier_cleanup.rs` based on what actually
lands.

---

## S2: Promote Tier A to `bolt-verifier-runtime` crate

**Goal.** Stop emitting Tier A as a per-protocol template. Instead, ship it
once as a real workspace crate that the generated `jolt-verifier` crate
declares as a Cargo dependency.

**Why first.** Largest leverage, smallest semantics change. Tier A contains
no Jolt-specific code; the only reason it lives under `crates/jolt-verifier`
is that the artifact pipeline currently emits it there. Once Tier A is a
real crate, every subsequent slice operates against a stable, versionable
surface.

### Concrete plumbing

1. New crate `crates/bolt-verifier-runtime/` with `src/lib.rs` containing
   today's `crates/bolt/src/protocols/jolt/verifier_common.rs.template`
   verbatim, minus the per-protocol `super::common`-style internal
   references (which don't exist in Tier A anyway).
2. Workspace `Cargo.toml` registers the new crate.
3. `crates/bolt/src/protocols/jolt/artifacts.rs::verifier_runtime_modules`
   stops emitting the `common` module (the `ProtocolRuntimeModule` entry
   for `common` is removed). The `jolt_relations` entry stays.
4. Stage emitters (`crates/bolt/src/protocols/jolt/emit/rust/stage{1..8}.rs`)
   change `super::common::{...}` imports to
   `bolt_verifier_runtime::{...}`. The `super::jolt_relations::{...}`
   imports are unchanged.
5. The Bolt manifest emitter (the part of `artifacts.rs` that writes the
   generated `Cargo.toml` for `jolt-verifier`) adds a `bolt-verifier-runtime`
   dependency entry, mirroring how `jolt-field`, `jolt-poly`, etc. are
   already wired.
6. `crates/jolt-verifier/src/stages/common.rs` is deleted from the goldens.
   `crates/jolt-verifier/src/stages/mod.rs` drops `pub mod common;` (it
   re-exports `bolt_verifier_runtime` items at the same path if any
   existing call site references `super::common::*`).
7. `crates/jolt-equivalence/src/plan_adapters/generated_stage*.rs` and the
   oracle modules update their `use jolt_verifier::stages::common::*`
   imports to `use bolt_verifier_runtime::*`.

### Test/gate updates

- `verifier_cleanup.rs` removes `bolt_runtime_loc` (it no longer counts
  toward the verifier crate). Add an equivalent gate over the new crate:
  `BOLT_VERIFIER_RUNTIME_LOC_CEILING ≈ 1,400`.
- `checked_in_generated_verifier_respects_boundary_hygiene` adds
  `bolt-verifier-runtime` to the *allowed* import list (it is the
  intentional runtime dependency).
- `commitment_ir.rs::assert_rust_source_compiles` no longer needs to stage
  `common.rs` into the test harness.

### Blockers and complications

- **Cross-crate blast radius.** `jolt-equivalence` has many import sites
  that name `jolt_verifier::stages::common::*`. The full-cutover rule
  (workspace policy) forbids leaving back-compat re-exports, so all sites
  must move to `bolt_verifier_runtime` in the same PR. Mechanical, but
  large.
- **Type aliases in stage files.** Generated stage files contain
  `pub type Stage6FieldExprPlan = FieldExprPlan;` and similar. After S2
  the right-hand side is `bolt_verifier_runtime::FieldExprPlan`. The
  per-stage Bolt emitter that produces these aliases must update its import
  path and the alias targets. This is one localized change per stage
  emitter.
- **`impl_runtime_plan_error_conversion!` macro.** Currently
  `pub(crate) use`. Either promote to `pub` in the new crate, or keep it
  per-stage by re-exporting it through `jolt-verifier`. Lean: `pub` in the
  new crate.
- **Generated `Cargo.toml` dependency injection.** The Bolt manifest
  emitter must learn that `bolt-verifier-runtime` is a workspace-relative
  path dependency for non-published builds, but a versioned
  registry dependency for published builds. Check how `jolt-field` etc.
  are currently wired and follow the same pattern.
- **`cargo check` chain.** Existing CI runs
  `cargo check -p bolt -p jolt-verifier -p jolt-prover -p jolt-equivalence`.
  Add `-p bolt-verifier-runtime`.

### Acceptance criteria

```text
cargo check -p bolt -p bolt-verifier-runtime -p jolt-verifier
            -p jolt-prover -p jolt-equivalence  --quiet
cargo nextest run -p bolt --test verifier_cleanup --no-capture
cargo nextest run -p bolt --test commitment_ir --cargo-quiet
cargo nextest run -p jolt-equivalence --test generated_role_crates --cargo-quiet
cargo nextest run -p jolt-equivalence --test bolt_commitment --no-capture
```

All green. The `verifier_cleanup` metrics output reports
`bolt_runtime_loc = 0` (Tier A no longer in the verifier crate) and a new
`bolt_verifier_runtime_loc` line.

### Rollback

S2 is reversible by re-introducing the `ProtocolRuntimeModule` entry for
`common` and reverting the import-path changes. The new crate can stay (it
is harmless) or be deleted. No data or proof-format changes are involved.

### Estimated wall-clock

One agent session, 60-90 minutes. The work is mechanical but touches many
files because of the cross-crate import changes.

---

## S3: Polynomial primitives + point reordering + gamma-power vectors

**Goal.** Lift the small set of pure-dataflow primitives in Tier B into
new MLIR ops. Tier B's relation evaluators stop calling
`EqPolynomial::mle`, `bytecode_gamma_powers`, `field_powers`,
`reverse_slice`, `prefix_point`, `suffix_point`, and the `normalize_*_point`
helpers directly.

### Dialect changes

Extend `crates/bolt/irdl/compute.mlir` (no new dialect; these are dataflow
ops, same family as the existing `point_*` and `field_*` ops):

```mlir
compute.poly_mle              %p1, %p2 -> %scalar
compute.poly_eq_indexed       %point, index = N -> %scalar
compute.poly_identity_eval    %point -> %scalar
compute.poly_lt_eval          %x, %y -> %scalar
compute.poly_operand_eval     %point, side = "left" | "right" -> %scalar

compute.point_reverse         %p -> %p'
compute.point_split           %p, at = log_k -> %lo, %hi
compute.point_prefix          %p, length = N -> %p'
compute.point_suffix          %p, length = N -> %p'

compute.field_pow_vector      %base, count = N -> %vec : tensor<N x !field_value>
```

`compute::poly_mle` is the workhorse. The other `poly_*` ops are convenient
specializations for relations that already exist (`identity`, `lt`,
`operand`, `eq_indexed`).

### Runtime additions

`bolt-verifier-runtime` (post-S2) gains the corresponding interpreter
dispatches in `evaluate_field_expr`. Each new op extends `FieldExprKind`
with a typed variant, e.g.

```rust
FieldExprKind::PolyMle { /* operand symbols carried by FieldExprPlan */ }
FieldExprKind::PointReverse { /* same */ }
FieldExprKind::FieldPowVector { count: usize }
```

The interpreter calls into `jolt-poly`'s `EqPolynomial::mle` and the local
helpers for the rest.

### Tier B impact

After S3, `verifier_jolt_relations.rs.template` no longer defines:
`bytecode_gamma_powers`, `field_powers`, `reverse_slice`, `prefix_point`,
`suffix_point`, `indexed_boolean_eq`, `normalize_bytecode_read_raf_point`,
`normalize_instruction_read_raf_point`, `operand_polynomial_eval`,
`identity_polynomial_eval`, `lt_polynomial_eval`. Total: ~140 LOC removed.

The `expected_stage67_*` evaluators stay hand-written for now, but their
*bodies* shrink because the primitives they invoke are now provided by the
runtime.

### Blockers and complications

- **`jolt-poly` API stability.** `EqPolynomial::mle` must have a stable
  signature and bit-exact semantics. `jolt-poly` is on the Bolt side of the
  generic compiler, so this is fine, but worth a sanity check that we're
  not pinning to an evolving signature.
- **`tensor<N x !field_value>` as MLIR type.** The compute dialect already
  uses `irdl.parametric @compute::@field_value<>` for scalars and
  `@compute::@point<>` for points. We may need either a new
  `@compute::@field_vector<>` carrier type or re-use `@point<>` for
  `field_pow_vector` results. Lean: re-use `@point<>` since gamma-power
  vectors are semantically just "ordered scalar tuples." Confirm with a
  small smoke test before formalizing.
- **Stage emitter updates.** The Stage 6/7 emitters (the largest in
  `crates/bolt/src/protocols/jolt/emit/rust/`) currently inline calls to
  these helpers when building `expected_stage67_*` bodies. They must learn
  to lower those call sites to `compute.poly_mle` etc. and let the
  generic interpreter run them.
- **Validator updates.** Each new `compute::*` op needs an entry in the
  Bolt validator so malformed plans (wrong arity, wrong operand types) are
  rejected at compile time.

### Acceptance criteria

`muldiv` e2e test passes in both `--features host` and `--features host,zk`
(this is the workspace's primary correctness check). Tier B drops to
~500 LOC. New metrics in `verifier_cleanup.rs`:
`compute_poly_op_call_sites`, `compute_point_op_call_sites` reported but
not gated yet.

### Rollback

Each new MLIR op is independently revertible. Worst case we keep the
`poly_mle` op (which is unambiguous) and revert the `normalize_*_point`
lowering if the dataflow ordering turns out to be subtle.

### Estimated wall-clock

Two agent sessions, ~120-180 minutes total. Most of the time is in the
emitter changes (Stage 6/7 bodies) and validator updates.

---

## S4: Typed indexed-eval addressing

**Goal.** Eliminate the last big string-dispatch site in Tier B:
`indexed_evals_by_prefix_any(evals, "stage6.booleanity.eval.InstructionRa_")`
and friends. Replace with a typed eval-family vocabulary.

### The problem today

Stage 6/7 evaluators do:

```rust
let booleanity_evals =
    indexed_evals_by_prefix_any(evals, "stage6.booleanity.eval.InstructionRa_")?;
```

This works because the prover emits evals named `..._0, ..._1, ..._2` and
the verifier wants them as `Vec<Fr>`. The verifier reconstructs the family
by string-prefix matching plus integer-suffix parsing. This is the *only*
remaining "execution-relevant string dispatch" that the current
`verifier_cleanup` gates do not yet enforce-to-zero.

### Dialect changes

Extend `compute::sumcheck_eval` with an `eval_family` attribute (or add a
sibling `compute::sumcheck_eval_family` op):

```mlir
compute.sumcheck_eval_family {
  sym_name = "stage6.booleanity.instruction_ra_family"
  produced_by = @stage6_booleanity_driver
  oracle_family = "InstructionRa"
  count = 3
} -> %eval_vec : !field_vector
```

The existing `compute::sumcheck_eval` ops with names like
`...InstructionRa_0`, `...InstructionRa_1`, ... are replaced by a single
family op. The prover side already emits these as a contiguous block; the
verifier side gains a typed handle to the whole block.

### Runtime additions

Add `EvalFamilyPlan` to `bolt-verifier-runtime`:

```rust
pub struct EvalFamilyPlan {
    pub symbol: &'static str,
    pub source: &'static str,        // sumcheck driver symbol
    pub oracle_family: &'static str, // diagnostic
    pub count: usize,
}
```

`ValueStore` learns to materialize `Vec<F>` for family symbols when the
parent sumcheck output is observed.

### Tier B impact

Each `expected_stage67_*` that today calls `indexed_evals_by_prefix_any`
instead reads a typed `&[Fr]` from the store. Estimated ~60 LOC reduction
in Tier B.

### Blockers and complications

- **Proof format.** The proof's `evals` field is already
  `Vec<StageNamedEval<F>>` with explicit `name` strings (verified in
  `verifier_common.rs.template:393-397`). S4 changes how the *verifier
  consumes* named evals, not how they are serialized. No back-compat
  break. *This was the biggest risk; it is not real.*
- **Prover/verifier symmetry.** The prover-side emitter must annotate the
  same eval block as a family. If this is a separate emitter, both must be
  updated together.
- **`indexed_evals_by_prefix` and `indexed_evals_by_prefix_any` removal.**
  These two helpers in Tier A become unused after S4. They can be deleted
  from `bolt-verifier-runtime` to keep the API surface minimal.
- **Testing.** Add a `verifier_cleanup` gate
  `RELATION_INDEXED_EVAL_PREFIX_SITES_CEILING = 0` that fires if any
  generated stage source still calls `indexed_evals_by_prefix*`.

### Acceptance criteria

`muldiv` passes in both modes. Zero `indexed_evals_by_prefix*` call sites
in the generated verifier. Tier B drops to ~350 LOC.

### Rollback

Pure refactor. Keep `indexed_evals_by_prefix_any` in the runtime
indefinitely as a fallback if the family-typing turns out to have edge
cases (e.g., variable-count families that depend on runtime data).

### Estimated wall-clock

One agent session, 60-90 minutes.

---

## S5: Relations as typed plan data

**Goal.** Make the Stage 6/7 relation evaluators (`expected_stage67_*`)
declarative MLIR-derived plans rather than hand-written Rust functions.
After S5, the only thing left in Tier B is the bytecode-row encoding
(addressed by S6).

### The shape

After S3 + S4, every `expected_stage67_*` body is structurally:

```text
val = Σ_i γ^i · (Π_j eq_mle(P_ij, query_point) · eval_k_ij) + boundary_terms
```

That is a flat algebraic expression DAG over plan operands. Capture the
DAG as data:

```rust
pub struct RelationPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub kind: RelationKind,           // existing typed enum
    pub terms: &'static [RelationTerm],
}

pub enum RelationTerm {
    GammaScaled { gamma_index: usize, term: &'static RelationTerm },
    EqMle { lhs_point: &'static str, rhs_point: &'static str },
    EvalProduct { evals: &'static [&'static str] },
    EvalFamilyProduct { family: &'static str },
    Sum(&'static [&'static RelationTerm]),
    // boundary terms as needed
}
```

The runtime gains a generic `evaluate_relation_plan(plan, store, evals,
query_point) -> Fr`. The interpreter is similar to today's
`evaluate_field_expr` but specialized for the relation shape.

### Dialect changes

Add a `compute::relation_term` op family that mirrors the
`RelationTerm` enum, plus a top-level `compute::relation` op that ties
them together for a sumcheck instance:

```mlir
compute.relation {
  sym_name = "stage6.booleanity"
  kind = "Stage6Booleanity"
  // body is a region of relation_term ops
}
```

Stage emitters lower these to typed relation plans in the generated stage
files.

### Tier C impact (acknowledged growth)

The relation plans live in the *generated* stage Rust, not in the runtime.
Stage 6/7 will grow by an estimated ~200 LOC of relation tables. This is
the explicit trade: Tier B shrinks by ~200 LOC and Tier C grows by ~200,
but Tier C is declarative data (audit-easy) while Tier B is hand-written
Rust (audit-hard).

The `GENERATED_VERIFIER_TARGET_LOC` ceiling will need to bump from 6,100 to
~6,300. That is OK provided the structure of stage files becomes more
declarative.

### Blockers and complications

- **Expressiveness limit.** The relation interpreter must express every
  algebraic shape that the Stage 6/7 relations currently use. Today's
  shapes are uniform; future relations may not be. If Bolt acquires a new
  protocol with relations that have control flow (e.g., conditional
  structure based on entry data), S5's plan vocabulary must be extended
  rather than escaped.
- **Bytecode-read-RAF special case.** The bytecode-read-RAF evaluator has
  a `for index in 0..stage_value_evals.len()` loop with a per-index
  `int_contrib` mask. That loop can be captured as `RelationTerm::Sum`
  over a static slice, but the mask values come from
  `stage67_bytecode_stage_value_evals`, which is bytecode-encoding-specific
  (S6 territory). For S5, we treat the mask as an opaque scalar derived by
  Tier B (or an S6 typed plan); the relation expression itself stays
  generic.
- **Performance.** The verifier-side relation interpreter walks a small
  DAG. We expect zero perf concern. Keep a smoke benchmark anyway since
  this code runs once per Stage 6/7 sumcheck instance.
- **Audit story.** Tier B drops to ~290 LOC, almost all of which is the
  bytecode encoder. We should explicitly document in `GOAL.md` that the
  relation algebra is *no longer* part of the audit surface; only the
  generic interpreter (in `bolt-verifier-runtime`) and the bytecode
  encoder are.

### Acceptance criteria

Same correctness gates plus: `expected_stage67_*` functions in Tier B
are deleted. New `compute::relation`/`relation_term` ops have validator
coverage. `muldiv` passes in both modes.

### Rollback

This is the most invasive slice. If the relation interpreter design is
wrong, S5 should be revertible by restoring the deleted
`expected_stage67_*` functions and removing the relation-plan emitter
output. Recommend keeping the per-relation hand-written code in a
preserved-for-comparison file in the worklog while S5 is being shaken
down.

### Estimated wall-clock

Three to four agent sessions, ~6 hours of agent runtime in total. Highest
risk slice in this plan.

---

## S6: Bytecode encoding as typed plan data (optional)

**Goal.** Capture the Jolt-specific bytecode-row encoding rules as a
typed table rather than hand-written Rust. After S6, Tier B is ~50 LOC
and is essentially just `Stage67BytecodeEntry` (the data table itself,
not the math).

### The encoding today

`stage67_bytecode_entry_stage_values` decomposes a bytecode row into 5
gamma-weighted sums (one per Stage 1-5). The pattern for each stage:

```rust
let mut stage1 = entry.address() + entry.imm() * stage1_gamma_powers[1];
for (flag, gamma) in flags.iter().zip(stage1_gamma_powers.iter().skip(2)) {
    if *flag { stage1 += *gamma; }
}
```

This is a small DSL for "for each entry field, conditionally add a
gamma-weighted contribution." The fields and their conditions vary per
stage but the *shape* is uniform.

### Capture as data

```rust
pub struct BytecodeEncodingPlan {
    pub stage_terms: [&'static [BytecodeTerm]; 5],
}

pub enum BytecodeTerm {
    Address                                    { gamma_index: usize },
    Imm                                        { gamma_index: usize },
    CircuitFlag       { index: usize,            gamma_index: usize },
    EntryFlag         { which: BytecodeFlag,     gamma_index: usize },
    RegisterEq        { which: RegisterRole,     gamma_index: usize },
    LookupTableIndex  { offset: usize,           gamma_base: usize },
}

pub enum BytecodeFlag { IsInterleaved, IsBranch, IsNoop, LeftIsRs1, ... }
pub enum RegisterRole { Rd, Rs1, Rs2 }
```

The runtime gains `evaluate_bytecode_encoding(entries, plan, gamma_powers,
register_points) -> [Fr; 5]`. Tier B keeps only the
`Stage67BytecodeEntry` trait and the per-stage `Stage6BytecodeEntry` /
`Stage7BytecodeEntry` impls (~50 LOC).

### Dialect changes

Optional: a `bytecode.encoding_plan` op family in a new `bytecode.mlir`
sub-dialect, OR keep `BytecodeEncodingPlan` purely as runtime-side data
that the Bolt emitter populates from a small declarative source. Lean:
declarative source first, dialect later if a second protocol uses the
same shape.

### Blockers and complications

- **Protocol-specificity.** This encoding is genuinely Jolt-specific. If
  Bolt only ever has one protocol with this exact shape, S6 is overkill
  and we should leave Tier B at ~290 LOC of typed Rust.
- **Lookup-table indexing.** `stage5 += stage5_gamma_powers[2 + table];`
  is a *variable-index* contribution where the index depends on entry
  data. The plan vocabulary must support this (`LookupTableIndex` above
  handles it).
- **Conditional terms.** `if entry.is_interleaved() { ... }` becomes
  `BytecodeTerm::EntryFlag { which: IsInterleaved, ... }`. Mechanical.
- **Audit win is small.** The remaining 290 LOC of bytecode encoding is
  *already* easy to audit; it is a flat sequence of "if flag then add
  gamma^k" lines. The audit benefit of S6 is lower than S2-S5.

### Recommendation

**Skip S6 unless** Bolt acquires a second protocol that wants the same
encoding shape. The 290 LOC of typed bytecode-encoding Rust is acceptable
audit surface as long as it is well-organized and tested. Mark S6 as
"optional follow-up; revisit when there is a second user."

### Estimated wall-clock

If pursued: two agent sessions, ~3 hours. Otherwise zero.

---

## Cross-cutting concerns

### Coordination with Markos' equivalence track

Each slice that touches generated stage outputs (S2, S3, S4, S5) will
require equivalence-side adapter updates in
`crates/jolt-equivalence/src/plan_adapters/generated_stage*.rs`. The
correct workflow is:

1. Land the slice (compiler change + regenerated stage Rust + equivalence
   adapter update) in one PR.
2. Stack on top of the most recent Markos tip
   (`origin/jolt-v2/equivalence` today).
3. Do not split the slice across two PRs (one for compiler, one for
   equivalence). The cross-crate type changes are too tightly coupled.

If Markos pushes new commits to `jolt-v2/equivalence` while a slice is in
flight, the right move is `git rebase origin/jolt-v2/equivalence` and
re-run the full check matrix, not `git merge`.

### Generic compiler vs Jolt-specific quarantine

GOAL.md's "Locked Genericity Decisions" require that generic Bolt
compiler infrastructure not contain Jolt-specific code, and that Jolt
emitters be progressively lifted into generic CPU emitters.

S2 strengthens this: `bolt-verifier-runtime` is the *generic* verifier
runtime; nothing Jolt-specific lives there.

S3 + S4 are pure compute-dialect extensions; no Jolt content.

S5 introduces `compute::relation` which is generic algebra, not Jolt.
However, the *RelationKind* enum (which lives in the generic runtime
today) is still populated with Jolt-specific variants
(`Stage6BytecodeReadRaf`, etc.). After S5, consider moving the
`RelationKind` enum out of `bolt-verifier-runtime` and into a per-protocol
typed enum, with the runtime carrying only an opaque `RelationId`. This
cleans the trust boundary further but is not strictly necessary for S5
itself.

S6 is explicitly Jolt-specific. If pursued, it goes under
`crates/bolt/src/protocols/jolt/`, not in `bolt-verifier-runtime`.

### Trust boundary and audit surface evolution

The audit surface contracts as we move through the slices:

```text
post-S1:  Tier A (1,265 LOC)  + Tier B (638 LOC)  = 1,903 LOC audited
post-S2:  bolt-runtime crate  + Tier B (638 LOC)  = ~1,900 LOC, but
                                                    Tier A is now
                                                    versioned and reviewed
                                                    once across protocols
post-S3:  bolt-runtime crate  + Tier B (~500 LOC) = ~1,750 LOC
post-S4:  bolt-runtime crate  + Tier B (~350 LOC) = ~1,600 LOC
post-S5:  bolt-runtime crate  + Tier B (~290 LOC) = ~1,540 LOC
post-S6:  bolt-runtime crate  + Tier B (~50 LOC)  = ~1,300 LOC
```

The shape of the audit also changes: post-S5, almost all of Tier B is the
bytecode encoder, which is *data-table-shaped* hand-written Rust. That is
qualitatively easier to review than algebraic relation evaluators.

### MLIR dialect growth

`compute.mlir` is currently 808 lines. After S3 + S4 + S5 it will probably
land around 1,000-1,100 lines with the new poly, point-reordering,
pow-vector, eval-family, and relation ops. Still small enough to be one
dialect; no need for a `poly` or `relation` sub-dialect at that point.
Revisit if `compute.mlir` exceeds ~1,500 lines.

### Performance

Verifier perf is not the goal of this track, but every slice should
include a quick `muldiv` e2e timing check to catch accidental
quadratic-in-trace-length loops in the generic interpreters. The expected
trend is "no measurable change," because the interpreter dispatches are
straightforward and Stage 6/7 evaluators are called O(1) times per
verification.

### Compatibility with the `zk` feature

All five slices are `cfg`-independent. They affect plan generation and
verifier interpretation, both of which are identical between the standard
and `zk` modes. Each slice must run `muldiv` in both
`--features host` and `--features host,zk` per the workspace's primary
correctness check.

---

## Open questions

These deserve human (Markos / Quang) input before S2 starts:

1. **`bolt-verifier-runtime` location.** Standalone workspace crate under
   `crates/bolt-verifier-runtime/`, or sub-crate of `bolt`? Recommend
   standalone for versionability.
2. **`bolt-verifier-runtime` re-export through `bolt`.** Should `bolt`
   re-export `bolt_verifier_runtime` for convenience, or keep them
   separate? Recommend separate (stricter trust boundary).
3. **Per-stage type aliases.** Generated stage files contain
   `pub type Stage6FieldExprPlan = FieldExprPlan` and ~20 similar aliases
   per stage. Are these aliases load-bearing for downstream code (e.g.,
   `jolt-equivalence` adapters) or can they be deleted along with S2?
   Spot-checking suggests they exist purely for readability and can go.
4. **`RelationKind` enum location.** After S5, does it stay in
   `bolt-verifier-runtime` (which means the runtime carries Jolt
   protocol vocabulary) or migrate to a per-protocol `RelationId`? Lean:
   migrate, but treat as a follow-up to S5 not a precondition.
5. **S6 threshold.** Is "second protocol with bytecode-row encoding"
   the right trigger to actually do S6, or is there a different
   readability / audit reason to do it preemptively?

---

## Sequencing decision tree

```text
Land S2 first.                                   [unconditional]
  Re-baseline ceilings.

Land S3 next.                                    [unconditional]
  Tier B should drop ~140 LOC.

Decide on S4 vs Markos' next slice.              [coordinate]
  S4 is independent of Markos' work; either order is fine.

Pause before S5.                                 [explicit human review]
  Decide whether the relation-as-data shape is right for Jolt's
  current and likely-future relations. If yes, land S5. If not,
  stop here; the verifier is in a defensible state with Tier B at
  ~350 LOC of typed Rust.

S6 only on demand.                               [optional]
  Defer until there is a concrete second user or audit pressure.
```

---

This plan is the agent-side complement to GOAL.md's `Concrete Gates` and
`Iteration Algorithm`. Update both files together when a slice lands so
the goal definition and the implementation plan stay in sync.
