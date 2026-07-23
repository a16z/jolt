# Branch review: `feat/jolt-prover` (ec3c30433) vs merge-base a75713215

Scope: full PR diff (175 files, +6632/−4162) against `specs/prover-stage-drivers.md` (v2) and the
carried-over `specs/clean-slate-prover.md` invariants. Method: spec-first read, four parallel
area reviews (stage fronts, kernels, verifier, engine/tests) with old-vs-new tracing, acceptance-
criteria greps, and all verification gates run locally (five full byte-diff suite runs).

## Verdict

**Sound.** The v2 driver architecture is implemented as specified: jolt-verifier is prover-free
(grep-verified), member lists are single-sourced through the derive-emitted callback macros, the
wire is frozen (zero fixture changes, `jolt-prover-legacy` untouched, byte-diff oracle in-process),
and transcript-op parity was traced stage by stage with no drift found. No branch-introduced
correctness regression identified. One HIGH operational finding (a nondeterministic byte-diff
flake whose entire compare path is branch-untouched), one MEDIUM perf papercut, one MEDIUM
error-ordering deviation, and a tail of LOW/NIT robustness and doc items.

## Gates (run at ec3c30433)

| Gate | Result |
|---|---|
| `nextest -p jolt-prover --features prover-fixtures` | 16/16 (4 of 5 runs; 1 run hit finding 1's flake, re-passes) |
| `nextest -p jolt-kernels` | 5/5 |
| `nextest -p jolt-verifier` | 74/74 |
| `nextest -p jolt-verifier --features akita` | 76/76 |
| `clippy --all --features host` / `host,zk` | clean |
| `cargo fmt --check` | clean |
| `cargo machete --with-metadata` | **FAILS** — finding 4, fixed in follow-up commit |

## Findings

### 1. HIGH — byte-diff harness is nondeterministically flaky: bytecode chunk-0 commitment diverges from legacy under load

`crates/jolt-prover/tests/byte_diff.rs:386` —
`advice_committed::prover_matches_legacy_on_committed_advice_consumer_address_major` failed once
in five full-suite runs (parallel, 16 tests): `bytecode chunk 0 commitment diverged from legacy's`
(distinct GT elements). The same test passes in isolation and in the other four full runs.

Provenance: every function on the failing compare path is **untouched by this branch** —
`build_committed_bytecode_chunk_coeffs` (`jolt-kernels/src/committed_program.rs`, single-threaded,
deterministic; file has no diff vs merge-base), `DoryScheme::begin/feed/finish_with_hint`
(jolt-dory: no diff), legacy `verifier_preprocessing_from_prover` (jolt-prover-legacy: no diff),
and the test helper + assert themselves (pre-existing; the branch's byte_diff.rs changes are
signature plumbing + the `RetainedProgram` park). So this is almost certainly a **pre-existing
latent nondeterminism**, not branch drift — but it fails the branch's primary gate, and the spec
leans everything on that gate.

Group arithmetic is exact, so parallel MSM bracketing cannot legitimately change a commitment;
nextest is process-per-test, so cross-test interference is out. The remaining candidate class is a
scheduling-dependent race or uninitialized read inside the one test process (legacy `prove()` runs
first and dirties heap/thread state; commitments are computed after). `jolt-poly`'s
`unsafe_allocate_zero_vec` was checked — it is `alloc_zeroed`, not the classic assumed-zero bug.
Root cause not localized here; needs a dedicated hunt (rerun under `--test-threads` sweep /
ThreadSanitizer on the Dory streaming commit and legacy preprocessing commitment paths).
Two runs also flagged nextest `LEAK`/`leaky` on twin tests — likely detection noise, but note it.

### 2. MEDIUM — per-proof deep clone of the full program preprocessing

`crates/jolt-prover/src/prover.rs:73-78` — `session.park(RetainedProgram { program:
Arc::new(program.clone()) })` clones the entire `JoltProgramPreprocessing` (bytecode rows, RAM
image, layout) once per `prove()` call, only to wrap it in an `Arc`. Merge-base recipes threaded
`&program` borrows (zero copy). Session residency needs `'static` ownership, but the fix is to
hold `Arc<JoltProgramPreprocessing>` in `JoltProverPreprocessing` and park a refcount bump.
O(program size) per proof on the production entry point; also duplicated in the byte-diff harness
(`byte_diff.rs:530-535`). Spec disclaims prover perf for this PR, so not blocking — but this is
the one regression on the real `prove()` path.

### 3. MEDIUM — verify-path error *precedence* reordered on multi-defect proofs

`crates/jolt-verifier/src/stages/stage6b/batch.rs:172-247` — `build` now hoists entry-index /
bytecode-table-rows / clear-aux extraction ahead of the register/RAM point-split checks (old
order: splits first); the "preserving the fallible-check precedence" module-doc line was
deliberately dropped (063910d23). Same class: `stage2/verify.rs` runs τ_low/τ_high/uniskip before
the `PublicIoMemory` check; `stage6a/verify.rs:56-68` surfaces cycle-binding/entry-index/split
errors one stage earlier than 6b did. All reordered checks are pure (no transcript ops in the
windows), the accept/reject set is unchanged, and single-defect inputs produce byte-identical
`VerifierError`s — only *which* error a multi-defect proof reports first changes. Tampering
fixtures inject single defects, so they cannot catch this. Fine under a "same accept/reject +
same single-defect errors" reading of the wire freeze; flagging because the strict reading
("same errors, all inputs") is violated and the change was intentional but undocumented outside
the commit message.

### 4. MEDIUM — cargo-machete CI job fails: stale `rand_core` dev-dep in jolt-verifier

`crates/jolt-verifier/Cargo.toml:53` — the branch moved the engine twin tests (the only
`rand_core::OsRng` users) out of jolt-verifier into `jolt-prover/tests/engine_twins.rs` and
correctly added `rand_core` to jolt-prover's dev-deps, but left the now-unused dev-dep behind.
`cargo machete --with-metadata` (the exact CI invocation) fails on it. Genuinely unused — remove,
don't ignore. Fixed in the follow-up commit on this branch.

### 5. LOW — `prepare_optional` presence handling is asymmetric

`crates/jolt-prover/src/driver.rs:172-185` — `(Some(relation), missing cell)` is a hard error
attributed to the relation id, but `(None, Some cells)` silently returns `Ok(None)`: upstream
handing claims/points/challenges for a member this batch didn't instantiate is dropped without
diagnosis. Mirrors `begin_batch`'s convention (relation is the presence authority), so both sides
agree and nothing desynchronizes — but the wiring bug surfaces downstream instead of at its
source. A mirror check would close it.

### 6. LOW — jolt-prover's stage-6b surface is not `akita`-gated

`crates/jolt-verifier/src/stages/stage6b/outputs.rs:82-83` makes `inc_claim_reduction`
`#[cfg(not(feature = "akita"))]`, but jolt-prover's stage6b front (unconditional field init +
`IncClaimReductionChallenges` import) and `stages/drivers.rs:97` are not gated. Latent only:
nothing today co-builds jolt-prover with akita-enabled jolt-verifier (CI's akita job tests
jolt-verifier alone), but the first shared feature-unified build breaks. Worth a
`cfg`/compile-guard before akita reaches the prover stack.

### 7. LOW — dual-role inference premise is assumed, not enforced

`crates/jolt-kernels/src/reference/naive.rs:122-147, 296-301` — the id-intersection inference is
provably equivalent to the deleted explicit RamValCheck attach *today* (verified: only
RamValCheck's three ids intersect input/output structs workspace-wide, and opening ids are
relation-tagged so over-attach requires `from = Self`). But the spec premise "dual-role cells are
not Expr leaves" is unchecked: a future relation restaging a dual-role id as an output-Expr leaf
would silently emit the bound table value instead of the consumed claim, failing only as a distant
FS divergence. Extra caller-supplied opening tables under non-leaf ids are also accepted
unvalidated and outrank carried claims. A construction-time check —
`(input ∩ output ids) ∩ expr leaves == ∅`, reject tables for ids the Expr doesn't reference —
would move the failure to the seam.

### 8. LOW — no head-aligned member exercises the generated driver in any twin

Twin fixtures (`drivers.rs` twin_tests, `engine_twins.rs`) are all tail-aligned; the head-aligned
delayed-`finish_rounds` path is covered only at the raw-engine level
(`jolt-sumcheck/src/tests.rs:1376-1457`) and by the real 6b/7 precommitted members under the
byte-diff ratchet. The gap is the combination (generated head + driver + head-aligned member),
not the path. Cheap to add a head-aligned toy member.

### 9. LOW — stale/overstating docs (three spots)

- `specs/clean-slate-prover.md:345-352` — the new pointer paragraph is v1 wording: names
  `prove_clear` and says the derive generates the driver; v2 emits only the member-list macro and
  jolt-prover expands one recorder-generic `prove`.
- `crates/jolt-verifier-derive/src/lib.rs:335-338` — `stage_relation_id` doc claims the generated
  verify drivers read it; only jolt-prover's driver does (`driver.rs:435`).
- `crates/jolt-kernels/src/reference/mod.rs:6-7` — "implements the slot traits its top-level
  sibling defines": those ~27 sibling files are deleted.

### 10. NIT roundup

- `driver.rs:154`, `stages/drivers.rs:613` — `#[expect(clippy::type_complexity)]` without
  `reason`, inconsistent with sibling expects.
- `backend.rs:232-237` — `ProofSession::take` swallows an impossible downcast (`.ok()` drops the
  entry, masquerading as "never parked"); `state_or_insert_with` handles the same impossibility
  with an annotated `expect`.
- `jolt-kernels-derive/src/lib.rs:137-143` — `Box<dyn PrepareKernel<F,R> + Send>` (any second
  bound) is silently skipped per the documented contract; surfaces as a distant missing-bound
  error. Narrow trap, worth a doc line at the registry.
- `precommitted_reduction.rs:379-396` — bytecode kind re-spells the intermediate-vs-final branch
  `scalar_claim()` centralizes (condition itself single-sourced via `has_address_phase`;
  deliberate per spec §Execution 6d).
- `stage1/outputs.rs:100,164` — `.ok_or(empty_remainder_point(stage))` allocates the reason
  String on the success path; `ok_or_else` matches house style.
- `jolt-claims/.../ram/output_check.rs:69-77`, `.../booleanity/address_phase.rs:74-82` —
  fail-closed `from_transcript_values` stubs with fabricated `required/populated` counts;
  unreachable (both relations override `draw_challenges`), mildly misleading if ever surfaced.
- stage6a/verify.rs + stage6b/batch.rs both recompute cycle-binding/stage-points/entry-index per
  proof (small Vec clones, verifier cold path).
- `engine_twins.rs` duplicates `DenseMember`/`pedersen_setup` from jolt-sumcheck's tests (no
  shared test-util crate; acceptable for a cross-crate integration test).
- Spec §Design lists curation *after* the final-claim check; the implementation curates before
  (the defensible order for a mutating curation — both sides fold post-curation claims). Update
  the spec paragraph, not the code.

## Clean verdicts (explicitly verified)

**Spec invariant 1 — FS byte-identity, body-wide.** Transcript-op sequences traced old-vs-new for
every stage front: stage2's τ_high/uniskip/γ/log_k-output-address draws (now via the last member's
`draw_challenges` override — position and `challenge()`-vs-`challenge_scalar()` distinction
preserved, pin tests updated to values not just events); stage6a's pad/truncate + γ draws moved
into `BooleanityAddressPhase::draw_challenges` byte-for-byte; stage6b's front-kept γ/conditional-η
draws and curated dedup absorb (`stage6b_opening_values` via the invocation-site override);
stages 1/3/4/5/7 mechanical. Declaration order == old hand member-vector order for all eight
batches; the round-loop slot array preserves it (`Option` flatten), and `begin_batch`'s
presence-gated dims fold is the same generated code on both sides.

**Spec invariant 2 — declaration order is the only order.** No stage recipe contains a member
vector, `StageNOutputClaims` literal, or `expected_final_claim`/`FinalClaimMismatch` block; the
single macro-expanded instance lives in `impl_stage_prover!`. `stages/drivers.rs` is the complete
prove-side member-list surface (one callback invocation per stage; only 6b carries an override).

**Spec invariant 3 — prover-free verifier.** Acceptance grep
(`SumcheckKernel|ProverInputs|PrepareKernel|PrepareSumcheck|SumcheckPreparer|ProveRounds|prove_batch|prove_uniskip`)
→ zero hits in `crates/jolt-verifier/src` and tests; the 324-line engine-twin module was excised
from `relations.rs`; `#[sumcheck(external)]`/`ExternalMembers`/`BackendPreparer`/`prove_clear`/
`ProvedStage*` do not exist in the workspace; no `extern crate self` in jolt-verifier or
jolt-kernels (serde-style `crate = "..."` overrides on both derives). The emitted members macro is
inert token forwarding; the `crate` override correctly does not apply to it.

**Spec invariant 4 — wire freeze.** Zero fixture/binary changes in the diff; `jolt-prover-legacy`
diff is empty (in-process oracle unmodified); the harness has no disk-fixture regeneration
channel — stage-granular transcript-state ratchets plus whole-proof `assert_eq!` against a legacy
proof computed in-process. byte_diff.rs changes are signature plumbing only (the dropped verify
`zk` flag is now derived from the proof inside the verifier — semantics unchanged for clear
proofs).

**Fused round API.** Engine bookkeeping traced (`jolt-sumcheck/src/prover.rs:171-243`): `bind` is
`None` exactly on each member's first active round (pending_binds take/set alternation over
contiguous activation windows), `finish_rounds` exactly once per ever-active member, zero-round
members never called, inactive halving/degree checks/recorder sequence byte-identical to the
two-call engine. Kernel-side round-index reconstruction (`round − 1`, `num_rounds() − 1`) valid
for the full-window precommitted members. New engine test locks fused-vs-separate byte equality
including transcript state.

**Session carries.** Full inventory: 2 uni-skip kernels (park at front prepare → non-destructive
`state` read for the first-round poly → `take` by remainder kernel), 4 precommitted
`PrecommittedReductionCarry<F, {address relation}>` keyed by four distinct nominal structs
(collision-free), `RetainedProgram` (park at proof start, read-only). Session is per-proof
(`begin_proof()` → fresh default), so no cross-proof staleness. 6b→7 carry conditionality
(`park_carry` no-ops iff `!has_address_phase`) provably matches the stage-7 member's
schedule-driven presence (same field). Missing-carry paths are loud `KernelError`s and
twin-tested.

**Dual-role inference (today).** Equivalent to the deleted explicit wrapper: workspace scan shows
only RamValCheck's three ids intersect input/output structs; table-first/carried-claim-fallback is
deterministic; required-but-unserved cells error loudly; no other kernel re-attaches wire claims.

**Kernels.** The 27 deleted root slot-trait files contained zero checks (pure trait defs); all
prepare-time validation survives in the reference impls, several kernels gained geometry/row
validation. Precommitted reduction math byte-identical incl. the hinted `s(1)` padding subtlety
(WARNING doc carried over). No `macro_rules!` in jolt-kernels src. Program-image eq binding
against `relation.r_addr_rw()` traced equal to the old staged point.

**KernelSlots derive.** Syntactic match handles qualified `Box`/trait paths via `segments.last()`;
duplicate relation types in two fields → E0119 (desired one-slot-per-relation enforcement);
generics/where-clause propagate verbatim; non-kernel fields skipped per documented contract
(compile-tested in `backend.rs` toy registry).

**Members-macro emission.** `snake_case` handles digit-suffixed names (`Stage6aSumchecks` →
`stage6a_sumchecks_members`); `relation_path` strips generics from the last segment and forwards
module-relative paths resolved at the invocation site (all production batch fields are bare
idents — a qualified path would fail the consumer's `$relation:ident` match at compile time);
all nine macro names unique at crate root; `stage_relation_id`'s all-optional fallback is safe
(`plans` non-empty enforced; `ConcreteSumcheck::id` never overridden, so instance-id ≡ type-id).

**Verifier promotions.** `build_from_parts` is a faithful hoist (same splits/labels/stage-ids,
same `ram_reduced` length check, akita 9-leg order preserved, clones → moves); `product_tau_low`,
`Stage1ClearOutput::remainder_point`/`cycle_binding`, `advice_reference_point_from_upstream`,
`formula_dimensions_from_parts` all byte-identical to the inline copies they replace, plus
strictly-additive guards (η⟷committed-layout coupling, τ_low length). New accessor blocks are
read-only projections; the removal of `RamOutputCheck::set_output_address_challenges` eliminates
the one two-phase mutable-construction hole; pub widening (uniskip `verify_clear`,
`UniskipParams`, promoted helpers) is minimal and doc-justified. Blindfold/ZK paths value-
identical (stage2 blindfold reads the same drawn vector from its new home).

**Twin coverage.** Spec acceptance list fully present: plain member, `Option` member absent AND
present, session-carried member (park → reclaim in `prepare`), missing-carry error path,
`CommittedSumcheckRecorder` compile-witness, byte-match against generated `verify_clear` on twin
transcripts, prepare/park order pinned via session logs, uni-skip and committed engine twins.

**Line-count criterion.** `stages/` baseline 2832 lines (zero test code) → 2530 total now, of
which ~608 are the relocated `#[cfg(test)]` twins: production ≈1922, **−32%** (criterion ≥30%
met on production code; −10.7% raw including tests).

**Lint/comment policy.** No `#[allow]`; non-test `unwrap`/`expect` all `#[expect]`-annotated with
reasons (modulo the two NIT reasons above); comments are WHY/WARNING-grade; load-bearing warnings
(the `challenge()` decode notes, hinted-`s(1)` byte parity, macro-hygiene notes) preserved or
improved.
