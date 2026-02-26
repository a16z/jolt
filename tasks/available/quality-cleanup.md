# quality-cleanup: Post-implementation code quality pass

**Scope:** crates/jolt-poly/, crates/jolt-openings/, crates/jolt-sumcheck/, crates/jolt-spartan/, crates/jolt-instructions/, crates/jolt-dory/, crates/jolt-zkvm/

**Depends:** integrate-workspace

**Verifier:** ./verifiers/default.sh

**Context:**

All implementations, tests, and integration are complete. This is the final quality pass before the new crates are considered production-ready.

### Tasks

#### 1. Documentation completeness

- Every public item in every crate has a rustdoc comment
- Math-heavy items use LaTeX (`$...$`, `$$...$$`) in doc comments
- Each crate has a `README.md` with:
  - One-paragraph description of the crate's purpose
  - Link to the relevant Jolt Book section (https://jolt.a16zcrypto.com/)
  - Minimal usage example (3-10 lines of code)
  - Feature flags documented
  - Dependency diagram showing where this crate fits
- Module-level doc comments (`//!`) on every `lib.rs` and `mod.rs`

#### 2. API surface audit

- No unnecessary `pub` items — everything that doesn't need to be public should be `pub(crate)` or private
- No leaked implementation details in public APIs
- No arkworks types in any public API signature (they should be behind Jolt traits)
- No `CanonicalSerialize`/`CanonicalDeserialize` in any public trait bounds
- Consistent naming conventions across crates (e.g., all provers are `XxxProver`, all errors are `XxxError`)

#### 3. Code deduplication

- No logic duplicated across crate boundaries
- Shared utilities (if any) are in the appropriate lowest-common-ancestor crate
- No copy-pasted test helpers — extract shared test utilities into a `test-utils` module if needed

#### 4. Clippy pedantic

Run `cargo clippy --workspace -- -W clippy::pedantic` and fix reasonable warnings:
- Missing docs (should be fixed by step 1)
- Unnecessary clones
- Unidiomatic patterns
- Large enum variants
- Missing `#[must_use]` on functions that return values

Skip the pedantic lints that are too noisy (e.g., `too_many_arguments` for legitimate API signatures).

#### 5. File size audit

- No source file exceeds 500 lines
- Files over 300 lines should be reviewed for potential splitting
- Each file has a single clear responsibility

#### 6. Dead code removal

- No unused imports
- No unused functions, types, or modules
- No commented-out code
- No `#[allow(dead_code)]` attributes (unless genuinely temporary and annotated with a TODO)

#### 7. Consistency check

- All error types use `thiserror`
- All serialization uses `serde` (not arkworks serialization)
- All crates have the `parallel` feature flag with consistent behavior
- All public types that should be `Clone + Debug + Send + Sync` are
- Consistent use of `Result<T, E>` (no unwraps in library code)

**Acceptance:**

- Every public item documented with rustdoc
- README.md exists for every new crate
- No arkworks types in public APIs
- `cargo clippy --workspace -- -W clippy::pedantic` has no actionable warnings
- No file exceeds 500 lines
- No dead code, unused imports, or commented-out code
- Consistent error handling, serialization, and feature flags across all crates
- `cargo doc --workspace --no-deps` builds without warnings
