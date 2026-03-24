# Crate Review Framework

Systematic bottom-up review of every `jolt-*` crate to crates.io-publishable quality.

## Dependency Graph & Review Order

```
Level 0 ── jolt-field, jolt-profiling, jolt-host
Level 1 ── jolt-transcript, jolt-ir, jolt-instructions
Level 2 ── jolt-crypto, jolt-poly, jolt-compute, jolt-witness
Level 3 ── jolt-openings, jolt-cpu
Level 4 ── jolt-dory, jolt-hyperkzg, jolt-sumcheck, jolt-metal
Level 5 ── jolt-spartan, jolt-wrapper
Level 6 ── jolt-blindfold
Level 7 ── jolt-verifier
Level 8 ── jolt-zkvm
```

Review proceeds left-to-right within a level, bottom-to-top across levels.
A crate is only marked DONE after all its findings are resolved AND downstream crates are adjusted.

---

## Review Categories

Each crate is evaluated across two dimensions: **Code Quality** and **Crate Design**.
Every criterion has a specific ID (e.g., `CQ-1`) for traceability in findings.

### A. Code Quality

#### CQ-1: Idiomatic Rust

- [ ] Types use correct ownership: borrows where possible, owned where necessary, no gratuitous `Clone`
- [ ] Error handling: `Result` over panics at API boundaries; `unwrap`/`expect` only where invariants are proven
- [ ] Iterator usage: prefer combinators over manual loops; no `collect()` into intermediate vecs when streaming suffices
- [ ] Pattern matching: exhaustive matches, no wildcard arms that silently swallow new variants
- [ ] Trait usage: blanket impls where appropriate, no orphan-rule workarounds via newtypes unless justified
- [ ] `From`/`Into`/`TryFrom` for conversions; no ad-hoc `to_foo()` that duplicates a standard trait
- [ ] Correct `derive` usage: no manual impls of `Clone`/`Debug`/`PartialEq` when derive works
- [ ] `Default` impl where there's a natural default
- [ ] `#[must_use]` on pure functions that return values callers shouldn't ignore
- [ ] No `as` casts for potentially lossy numeric conversions — use `try_into()` or explicit truncation

#### CQ-2: Cognitive Load & Clarity

- [ ] Functions < ~60 lines; complex functions decomposed into named helpers
- [ ] No deep nesting (> 3 levels of indent outside match arms)
- [ ] Variable/function names are descriptive; math variables follow `non_snake_case` convention per CLAUDE.md
- [ ] Type aliases clarify intent (e.g., `type LogSize = usize`) rather than obscure it
- [ ] No stringly-typed interfaces — use enums/newtypes for domain concepts
- [ ] Control flow is linear where possible (early returns over nested if/else)
- [ ] Generic parameters are constrained tightly — no `T: Clone + Debug + Send + Sync + ...` grab-bags

#### CQ-3: Redundancy & DRY

- [ ] No duplicated logic across functions/modules (extract shared code)
- [ ] No near-identical struct/enum variants that should be parameterized
- [ ] No reimplementation of functionality available in dependencies
- [ ] Constants/magic numbers extracted and named
- [ ] No dead code: unused functions, unreachable branches, vestigial feature gates
- [ ] No stale imports

#### CQ-4: Indirection & Abstraction

- [ ] Every trait is justified: has ≥ 2 real impls OR is a crate boundary contract
- [ ] No "pass-through" wrapper types that add no behavior
- [ ] No unnecessary `Box<dyn Trait>` when static dispatch works and the type set is closed
- [ ] Generics earn their keep — a generic param with only one concrete instantiation is premature
- [ ] Module hierarchy reflects logical grouping, not arbitrary splitting
- [ ] `pub` visibility is intentional — nothing is `pub` just because it was easier

#### CQ-5: Unsafe & Invariants

- [ ] Every `unsafe` block has a `// SAFETY:` comment explaining why it's sound
- [ ] Unsafe is used only where it provides measurable perf gain or is required by FFI
- [ ] Invariants that unsafe code relies on are documented and enforced by the module's safe API
- [ ] No UB: no uninitialized memory read, no aliasing violations, no data races

#### CQ-6: Performance (Hot Path)

- [ ] `#[inline]` on small functions called in tight loops (sumcheck inner loops, polynomial ops)
- [ ] Pre-allocated buffers where size is known; no grow-in-loop patterns
- [ ] No unnecessary allocations in hot paths (clones, to_vec, collect)
- [ ] Correct use of `rayon` parallelism: work per task justifies overhead, no over-parallelization
- [ ] Cache-friendly access patterns (sequential > random, SoA over AoS where it matters)
- [ ] No redundant computation (recompute vs cache tradeoff explicit and justified)

#### CQ-7: Comments & Documentation

- [ ] All `pub` items have doc comments explaining behavior, constraints, and invariants
- [ ] Doc comments include examples for non-obvious APIs
- [ ] No comment types banned by CLAUDE.md (section separators, restating names, commented-out code, TODOs without links)
- [ ] WHY comments on non-obvious decisions; SAFETY comments on unsafe blocks
- [ ] Crate-level `//!` doc describes purpose, key types, and usage
- [ ] No stale comments that contradict the code

#### CQ-8: Test Quality & Coverage

- [ ] Unit tests for all non-trivial public functions
- [ ] Edge cases tested: empty inputs, boundary values, max sizes
- [ ] Property-based tests for algebraic invariants (field axioms, polynomial identities)
- [ ] Integration tests for cross-module workflows
- [ ] Test names describe the scenario, not the function (`test_bind_preserves_evaluation` not `test_bind`)
- [ ] No tests that only assert `!panicked` — test actual outputs
- [ ] Benchmarks for performance-critical paths (if the crate contains hot-path code)
- [ ] Fuzz targets for parsing/deserialization (if applicable)
- [ ] All tests pass: `cargo nextest run -p <crate> --cargo-quiet`
- [ ] Zero clippy warnings: both standard and (if applicable) `zk` feature modes

### B. Nits & Aesthetics

#### NIT-1: Import Style

- [ ] No fully qualified paths in function bodies — use `use` imports at the top
- [ ] Imports are grouped: std, external crates, internal crates, `self`/`super` — separated by blank lines
- [ ] No glob imports (`use foo::*`) except in test modules or preludes
- [ ] No redundant path prefixes (`self::`, `crate::`) where the short form resolves
- [ ] Re-imports of the same item from different locations consolidated

#### NIT-2: Formatting & Layout

- [ ] No separator comments (`// =========`, `// ---------`, `// ***`)
- [ ] No banner/ASCII-art comments
- [ ] No trailing whitespace or inconsistent blank lines (more than 2 consecutive)
- [ ] Consistent brace style (Rust standard — opening brace on same line)
- [ ] `impl` blocks ordered: trait impls grouped together, then inherent impls; within each, public methods before private
- [ ] Match arms aligned consistently; single-expression arms use `=>` without braces
- [ ] Tuple structs vs named structs: named fields when there are ≥ 3 fields or meaning isn't obvious from context

#### NIT-3: Naming Conventions

- [ ] No Hungarian notation or type-encoding in names (`vec_of_polys` → `polynomials`)
- [ ] No single-letter variables outside of tight math contexts (closures, short loops)
- [ ] Boolean variables/functions read as predicates (`is_empty`, `has_next`, `should_skip`)
- [ ] No abbreviations that aren't universally understood in the domain (`eval` ok, `proc` not ok)
- [ ] Consistent naming across the crate (don't call the same concept `size` in one place and `length` in another)

#### NIT-4: Misc Aesthetics

- [ ] No `return` keyword at the end of functions — use expression position
- [ ] No `if x == true` or `if x == false` — use `if x` / `if !x`
- [ ] No `.clone()` immediately followed by a consuming call — pass ownership directly
- [ ] No `&String` or `&Vec<T>` in function signatures — use `&str` / `&[T]`
- [ ] No `impl Trait` in return position when the concrete type is simple and known
- [ ] Empty trait impls on one line: `impl Foo for Bar {}`
- [ ] Prefer `Option::map`/`unwrap_or` over `match Some/None` for simple transforms
- [ ] No `pub(crate)` on items inside private modules (already crate-private by the module)
- [ ] Lifetime elision used wherever the compiler allows it — no explicit `'a` when unnecessary
- [ ] `where` clauses over inline bounds when bounds exceed ~80 chars

### C. Crate Design

#### CD-1: Purpose & Scope

- [ ] Crate has a single, clear responsibility (one sentence description)
- [ ] No functionality that belongs in a different crate (upward or sideways leakage)
- [ ] No missing functionality that callers must re-implement (incomplete abstraction)
- [ ] Feature flags (if any) are orthogonal and well-documented
- [ ] Crate name accurately reflects its contents

#### CD-2: API Surface

- [ ] Public API is minimal: only expose what downstream crates need
- [ ] API is hard to misuse: type-level enforcement over runtime checks where feasible
- [ ] Re-exports are intentional and documented (no accidental `pub use` chains)
- [ ] Breaking changes to the API have clear justification
- [ ] Generic parameters serve a real extensibility need, not hypothetical future use
- [ ] Trait methods have sensible defaults where applicable
- [ ] Builder/config patterns where construction has many optional parameters

#### CD-3: Dependency Hygiene

- [ ] Only depends on crates it actually uses (no dead deps)
- [ ] No circular or near-circular dependency chains
- [ ] Dev-dependencies don't leak into the public API
- [ ] External dependencies (non-jolt) are justified and not excessive
- [ ] `[features]` don't pull in heavy deps unless explicitly opted in
- [ ] Version constraints are reasonable (not overly pinned, not overly loose)

#### CD-4: Crate Boundary Fitness

- [ ] Would a user of this crate ever need to depend on a sibling crate to use it? (If yes, boundary is wrong)
- [ ] Does the crate re-export enough from its dependencies, or do users need to add them manually?
- [ ] Is the crate too thin (< 200 lines of real logic → merge candidate)?
- [ ] Is the crate too fat (> 10k lines with separable concerns → split candidate)?
- [ ] Could two crates at the same level be merged without increasing coupling?

#### CD-5: Publishability (crates.io readiness)

- [ ] `Cargo.toml` has correct metadata: name, version, description, license, repository, edition
- [ ] `README.md` exists with: purpose, usage example, MSRV if relevant
- [ ] No path dependencies that would break on publish (use `version` + `path`)
- [ ] No `[patch]` section in the crate's own Cargo.toml (workspace-level only)
- [ ] Semver-compatible versioning (0.x is fine, but API should be stable within a version)

#### CD-6: Downstream Usage Fitness

- [ ] Every downstream crate's `use` of this crate's public API examined
- [ ] No type leakage: downstream crates don't re-export or pattern-match on types that should be abstracted away
- [ ] No missing abstractions: downstream crates don't duplicate logic that belongs in this crate's API
- [ ] Trait coverage: every downstream call site's needs are served by the trait/API surface (no workarounds)
- [ ] No forced re-imports: downstream crates don't need to add this crate's transitive deps to use its API
- [ ] The API enables the intended usage pattern — not just today's code, but the design goal of the abstraction boundary

---

## Review Process

### Per-Crate Review Workflow

```
1. AUDIT        — Subagents run each CQ/CD criterion, produce raw findings
2. AGGREGATE    — Compile findings into crate review doc: crates/<name>/REVIEW.md
3. DISCUSS      — Walk through findings together, classify as: FIX / WONTFIX / DEFER
4. IMPLEMENT    — Fix all FIX items; update downstream crates for API changes
5. VERIFY       — Re-run clippy, tests, downstream compilation
6. CLOSE        — Mark crate DONE in tracker below
```

### Subagent Decomposition

Each review spawns parallel subagents per criterion group:

| Agent | Criteria | What it does |
|-------|----------|--------------|
| `idiom-check` | CQ-1 | Scan for non-idiomatic patterns: raw `as` casts, unnecessary clones, manual impls |
| `clarity-check` | CQ-2 | Flag long functions, deep nesting, unclear names |
| `redundancy-check` | CQ-3 | Find duplicated logic, dead code, magic numbers |
| `abstraction-check` | CQ-4 | Audit traits (impl count), wrapper types, visibility |
| `unsafe-check` | CQ-5 | Enumerate all unsafe blocks, verify SAFETY comments |
| `perf-check` | CQ-6 | Find allocations in hot paths, missing inlines, redundant computation |
| `docs-check` | CQ-7 | Audit doc coverage, stale comments, banned patterns |
| `test-check` | CQ-8 | Run tests, measure coverage gaps, check test quality |
| `nits-check` | NIT-1 to NIT-4 | Fully qualified paths, separators, naming, formatting aesthetics |
| `design-check` | CD-1 to CD-5 | Crate scope, API surface, deps, boundaries, publishability |
| `downstream-check` | CD-6 | Examine how downstream crates actually use the API; flag leaky abstractions, misused types, missing trait coverage |

### Finding Format

Each finding in the review doc follows this structure:

```markdown
### [CQ-4.3] Unnecessary `Box<dyn Trait>` in `FooProcessor`

**File:** `src/processor.rs:42`
**Severity:** MEDIUM
**Finding:** `Box<dyn SomeHandler>` is used but `SomeHandler` only has one implementor (`DefaultHandler`).
Static dispatch via a generic `H: SomeHandler` eliminates the vtable indirection.

**Suggested fix:**
​```rust
// Before
struct FooProcessor {
    handler: Box<dyn SomeHandler>,
}

// After
struct FooProcessor<H: SomeHandler = DefaultHandler> {
    handler: H,
}
​```

**Status:** [ ] OPEN
```

Severities:
- **HIGH** — Correctness issue, unsound unsafe, or significant API design flaw
- **MEDIUM** — Performance issue, unnecessary complexity, or poor ergonomics
- **LOW** — Style nit, minor redundancy, or documentation gap

---

## Tracker

| Crate | Level | LOC | Status | Review Doc | Findings | Date |
|-------|-------|-----|--------|------------|----------|------|
| jolt-field | 0 | 3982 | **DONE** | [REVIEW.md](jolt-field/REVIEW.md) | 32 (all resolved) | 2026-03-24 |
| jolt-profiling | 0 | 475 | **DONE** | [REVIEW.md](jolt-profiling/REVIEW.md) | 22 (18 resolved, 4 pass/wontfix) | 2026-03-24 |
| jolt-host | 0 | 547 | **DONE** | [REVIEW.md](jolt-host/REVIEW.md) | 29 (22 resolved, 6 pass/wontfix, 1 deferred: CycleRow) | 2026-03-24 |
| jolt-transcript | 1 | 672 | **DONE** | [REVIEW.md](jolt-transcript/REVIEW.md) | 13 (10 resolved, 3 pass) | 2026-03-24 |
| jolt-ir | 1 | 12389 | PENDING | | | |
| jolt-instructions | 1 | 10280 | **DONE** | [REVIEW.md](jolt-instructions/REVIEW.md) | 12 (5 resolved, 7 pass/wontfix) | 2026-03-24 |
| jolt-crypto | 2 | 5070 | PENDING | | | |
| jolt-poly | 2 | 4841 | PENDING | | | |
| jolt-compute | 2 | 1480 | PENDING | | | |
| jolt-witness | 2 | 1150 | PENDING | | | |
| jolt-openings | 3 | 1424 | PENDING | | | |
| jolt-cpu | 3 | 4113 | PENDING | | | |
| jolt-dory | 4 | 1253 | PENDING | | | |
| jolt-hyperkzg | 4 | 996 | PENDING | | | |
| jolt-sumcheck | 4 | 4371 | PENDING | | | |
| jolt-metal | 4 | 4351 | PENDING | | | |
| jolt-spartan | 5 | 4257 | PENDING | | | |
| jolt-wrapper | 5 | 4355 | PENDING | | | |
| jolt-blindfold | 6 | 2458 | PENDING | | | |
| jolt-verifier | 7 | 1156 | PENDING | | | |
| jolt-zkvm | 8 | 6778 | PENDING | | | |
