# Spec: Field-Inline Verifier-Surface Reduction

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-08-29 |
| Status | draft |
| PR | #1808 (in-branch refactor) |

## Decision

The field-inline feature adds ~4,700 production code lines to jolt-verifier's
dependency closure — the protocol-load-bearing surface. Most of that is not
protocol content: formulas already single-sourced in the symbolic layer are
hand-written a second time as verifier impls, the BlindFold lowering repeats a
per-stage shell pattern, and the per-stage clear seams repeat an
absorb/attach/validate pattern eight times. This refactor derives the repeated
forms from their existing single sources and compresses the closure to a
target of ~2,000–2,200 code lines with zero wire change: every proof byte,
transcript byte, and fixture stays identical in every mode.

## Invariants (all machine-checked, all pre-existing)

- Wire identity: both byte-diff ratchets, the recorded `.jvcf` fixtures, and
  all e2e suites (dory FR-on/off, akita FR-on/off, zk) pass unmodified. A
  regenerated fixture is a spec violation, not a fix.
- The formula-pin tests are the independent ground truth for every derived
  formula; they are not rewritten to match the refactor.
- Protocol separation stands: jolt-claims protocol modules stay
  import-disjoint; jolt-verifier keeps one visible `cfg(feature =
  "field-inline")` divergence point per stage (compressed from a seam module
  to a registration line, not removed); the boundary tests keep enforcing
  both, with caps retightened to the new (smaller) counts.
- The FS absorb inventory changes only by call-site text/ordinal shifts from
  the refactor itself; the re-bless diff is reviewed entry-by-entry and must
  contain no new or lost absorb identities (same absorbs, same order, new
  spellings).

## Mechanisms

1. **Derive the verifier relation impls from the symbolic layer.** Each FR
   relation's `ConcreteSumcheck` surface (input/output claim evaluation,
   `derive_output_term`, challenge resolution, typed claim structs' plumbing)
   is generated from the relation's `SymbolicSumcheck` definition plus the
   existing claims-derive tables, instead of hand-written per relation. The
   symbolic expression is already the declared single source (BlindFold
   consumes it); the clear path now consumes it too. Hand code remains only
   where a relation has genuinely non-symbolic behavior (lazy coefficient
   tables, uniskip domain handling) and each such site carries a WHY comment
   naming what could not be derived.
2. **Table-drive the BlindFold stage shells.** The per-stage FR lowering
   files reduce to one interpreter over a stage-domain table: {stage, member
   list, committed-round row counts, publics sources, output-claim binding}.
   Expression lowering is already generic; the shells are the residue and
   become rows.
3. **Compress clear seams to registration.** `stages/stageN/field_inline.rs`
   modules are replaced by per-stage FR member registrations (one cfg'd
   declaration naming members + absorb order) consumed by one shared
   executor. The cfg'd line per stage preserves the visible divergence the
   architecture ruling requires.
4. **Consume composed geometry instead of re-deriving it.** The FR extensions
   in `outer_remainder` / `product_remainder` / `product_uniskip` read the
   feature-aware jolt-r1cs tables (`spartan_outer_opening_columns`, lane
   tables) rather than carrying their own composition arithmetic.
5. **Schema-drive the bytecode side-table converter.** The role mapping in
   `field_inline_bytecode.rs` becomes a table; the fail-closed validation
   surface is kept verbatim.

## Non-goals

- No change to the instance-parameterized-jolt-namespace question: the
  2026-08-18 separation ruling stands.
- No prover/kernel-side reduction (out of closure); kernels keep calling the
  same `ConcreteSumcheck` surface, whatever generates it.
- No comment-density campaign; doc trimming happens only as a by-product of
  deleted code.

## Acceptance

- Closure production code lines (the region-aware classifier over
  `git diff origin/main...HEAD`, comments/blanks excluded) lands at or under
  ~2,400; report the exact figure per mechanism.
- Full battery green: both workspace clippy lanes, all feature clippy lanes,
  all nextest lanes including `field-inline,akita`, both `-j1` fixture
  suites, all seven prover fixture lanes, legacy muldiv ×2, style script,
  FS inventory, boundary tests.
- The audit-tension check: `cargo expand` output for one derived relation is
  attached to the PR discussion so reviewers can see the expanded form the
  derive produces.

## Outcome (2026-08-29): the surface is at its floor; target superseded

The campaign ran as three census-gated units and disproved this spec's
premise. R1 (mechanism 1) landed −20 lines (commit a018601b3): the
input-claim and challenge halves were already derived — `ConcreteSumcheck`'s
trait defaults interpret the symbolic expressions, and only ~53 lines of
struct-fill/id-destructure residue existed to remove. R2 (mechanism 2) was a
no-go at ~55 recoverable lines: the BlindFold lowering already is the
rows-plus-interpreter form this spec proposed (generic `map_expr` +
jolt-claims geometry tables), the per-stage code is genuine math over
differently-typed inputs, and stages 2/4/6b hand-curate wire-bearing absorb
orders no static table can pin. R3 (mechanisms 3–5) was a no-go at ~48/~27/~5
recoverable lines respectively: the seams' member constructors and attach
targets are type-distinct (an executor costs what it removes at n=8), the
composed Spartan/product code already consumes the feature-aware jolt-r1cs
tables, and the bytecode converter is fail-closed validation plus
already-schema-driven suppression.

The measured floor under the 2026-08-18 separation ruling is **~4,800
closure production code lines** (comments excluded). The acceptance target
above (~2,400) was calibrated against a duplication hypothesis the three
censuses refuted: the original construction had already banked the
single-sourcing this spec set out to introduce. Further reduction requires
one of: type-erasing per-stage claim structs into a dynamic registry
(surrenders compile-time shape facts, net-negative at n=8); revisiting the
separation ruling (explicit non-goal); or deleting spec-recorded protocol
deviations (wire changes, not refactors). The only surviving crumb is a
~40-line micro-helper harvest in the seams, excluded because its largest
item relocates an absorb call site and forces an FS-inventory re-bless for
single-digit savings.

## Open questions

- Whether stage-2's curated absorb + `validate_product_aliases` equality
  deviation can ride the shared executor or stays bespoke (it is a recorded
  spec deviation; keeping it bespoke is acceptable).
- Whether the uniskip first-round handling (stage 1/2) fits mechanism 1 or
  remains hand-held (likely the latter; it predates the FR work).
