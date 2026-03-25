# jolt-instructions Review

**Crate:** jolt-instructions (Level 1)
**LOC:** 10,280 (was 10,672 — reduced ~4% via macro-based dispatch deduplication)
**Baseline:** 0 clippy warnings, 260 tests passing, 2 doc tests (ignored)

## Overview

RISC-V instruction definitions and lookup table decompositions for the Jolt zkVM.
Provides the `Instruction` trait (execution semantics + flags), `LookupTable` trait
(MLE evaluation + prefix/suffix decomposition), 105 concrete instructions (RV64IMAC +
virtual), 41 lookup tables, and the `JoltInstructionSet` registry. Used by 2 downstream
crates (jolt-host, jolt-zkvm).

**Verdict:** Well-structured crate with excellent test coverage (260 tests including
exhaustive MLE verification). The `define_instruction!` macro is clean and the
prefix/suffix decomposition is well-tested. A few maintenance hazards from manually
synchronized constants and duplicated enum variant lists.

---

## Findings

### [CD-1.1] LookupTableKind::COUNT is 40 but there are 41 variants

**File:** `src/tables/mod.rs:180`
**Severity:** MEDIUM
**Finding:** `pub const COUNT: usize = 40;` but the `LookupTableKind` enum has 41 variants
(0-indexed: `RangeCheck` = 0 through `VirtualXORROTW7` = 40). This means `COUNT` is
off-by-one. Currently unused outside the crate, but any downstream code using it for
array sizing would silently drop the last variant.

**Status:** [x] RESOLVED — Changed to `COUNT: usize = 41`, added compile-time const assertion.

---

### [CQ-1.1] No compile-time assertions for enum-vs-constant sync

**File:** `src/tables/mod.rs`, `src/tables/prefixes/mod.rs`, `src/flags.rs`
**Severity:** MEDIUM
**Finding:** Three manually maintained constants must match their enum variant counts:
- `LookupTableKind::COUNT` (wrong — see CD-1.1)
- `NUM_PREFIXES = 46` (correct today)
- `NUM_CIRCUIT_FLAGS = 14`, `NUM_INSTRUCTION_FLAGS = 7` (correct today)

The flags module has tests (`circuit_flags_count_matches_enum`,
`instruction_flags_count_matches_enum`) that verify last-variant + 1 == count.
But `LookupTableKind::COUNT` and `NUM_PREFIXES` have no such tests.

**Status:** [x] RESOLVED — Added `const _: ()` assertions for both `LookupTableKind::COUNT` and `NUM_PREFIXES` (compile-time, not just tests).

---

### [CQ-1.2] `unsafe transmute` for Prefixes iteration

**File:** `src/tables/test_utils.rs:130`, `src/tables/prefixes/mod.rs`
**Severity:** LOW
**Finding:** Two call sites use `unsafe { std::mem::transmute(i as u8) }` to iterate
over `Prefixes` variants. This is UB if `NUM_PREFIXES` doesn't match the actual
variant count (the transmute creates an invalid enum discriminant).

**Status:** [x] RESOLVED — Added `pub const ALL_PREFIXES: [Prefixes; NUM_PREFIXES]` array. Replaced both transmute sites with safe `ALL_PREFIXES[index]` indexing.

---

### [CQ-2.1] LookupBits::PartialEq ignores len

**File:** `src/lookup_bits.rs:168-173`
**Severity:** LOW
**Finding:** `PartialEq` compares only `self.as_u128() == other.as_u128()`, ignoring
the `len` field. Two bitvectors with different logical lengths but the same raw bits
(e.g., `LookupBits::new(5, 4)` vs `LookupBits::new(5, 8)`) would compare as equal.

In practice this doesn't cause bugs because `new()` masks excess bits and callers
always use the same `len` when comparing. But it's surprising semantics.

**Status:** [ ] PASS — not worth changing behavior; masking in `new()` prevents real issues.

---

### [CQ-3.1] Five sync points when adding a new lookup table

**File:** `src/tables/mod.rs`
**Severity:** LOW
**Finding:** Adding a new lookup table requires coordinated changes in 5 places within
`tables/mod.rs` alone: (1) `LookupTableKind` enum, (2) `LookupTables` enum,
(3) `dispatch_table!` macro arms, (4) `LookupTables::kind()` match,
(5) `From<LookupTableKind>` match, plus updating `COUNT`.

**Status:** [x] RESOLVED — Reduced to 3 sync points via `kind_table_identity!` macro that auto-generates the `kind()`, `From<LookupTableKind>`, and `From<LookupTables>` impls from a single variant list.

---

### [CQ-3.2] Duplicated prefix dispatch boilerplate

**File:** `src/tables/prefixes/mod.rs`
**Severity:** LOW
**Finding:** `prefix_mle()` and `update_prefix_checkpoint()` each had ~40 lines of
identical `use` imports and ~46-arm match blocks dispatching to concrete prefix types.
The two methods duplicated ~350 lines of near-identical boilerplate.

**Status:** [x] RESOLVED — Introduced `dispatch_prefix!` macro that encodes the variant-to-type mapping once. Both methods now delegate via a one-line macro call. Reduced `prefixes/mod.rs` from 690 to 350 lines (-49%).

---

### [CQ-4.1] Manual Instruction impls duplicate macro-generated boilerplate

**File:** `src/rv/arithmetic.rs` (MulH, MulHSU, MulHU, Div, DivU, Rem, RemU),
`src/rv/arithmetic_w.rs` (DivW, DivUW, RemW, RemUW)
**Severity:** LOW
**Finding:** 11 instructions are implemented manually (full struct + trait impls)
instead of using the `define_instruction!` macro. The macro doesn't support multi-line
execute bodies with `let` bindings, so these complex instructions opt out.

The manual impls are correct and consistent (same derives, same `#[inline]` placement).
Extending the macro to support block-expression bodies would reduce ~350 lines but
isn't clearly worth the macro complexity.

**Status:** [ ] PASS — manual impls are correct and consistent.

---

### [CD-6.1] Dispatch table duplication between jolt-host and jolt-zkvm

**File:** `jolt-host/src/cycle_row_impl.rs`, `jolt-zkvm/src/witness/flags.rs`
**Severity:** LOW (architecture, not this crate's fault)
**Finding:** Both downstream crates contain ~270-line match statements mapping
`tracer::Instruction` variants to `jolt-instructions` flag arrays. These are
mechanically identical and acknowledged as absorbed copies (comment: "ISA dispatch
tables absorbed from jolt-zkvm/src/witness/flags.rs").

The duplication exists because `jolt-instructions` correctly doesn't depend on
`tracer`, so it can't provide the mapping. This is proper layering. The CycleRow
trait (planned in `jolt-host/PLAN.md`) will consolidate this.

**Status:** [ ] PASS — proper layering; CycleRow plan addresses this.

---

### [CQ-5.1] JoltInstructionSet uses Vec<Box<dyn Instruction>>

**File:** `src/instruction_set.rs:14-16`
**Severity:** LOW
**Finding:** The registry heap-allocates 105 `Box<dyn Instruction>` for zero-sized
unit structs. A flat array of function pointers or an enum dispatch would avoid the
allocations and virtual dispatch overhead.

However, `JoltInstructionSet::new()` is called once at startup, not in hot paths.
The dynamic dispatch through `instruction()` is also cold-path (used for tests and
debugging, not the prover's inner loop). The proving system uses direct struct
instantiation, not the registry.

**Status:** [ ] PASS — cold-path only, not worth optimizing.

---

### [CQ-6.1] Shift instructions don't set lookup table

**File:** `src/rv/shift.rs`
**Severity:** PASS
**Finding:** `Sll`, `SllI`, `Srl`, `SrlI`, `Sra`, `SraI` have `table: None` (no
`table:` clause in the macro). This is correct — these are decomposed into virtual
shift sequences (`VirtualSrl`, `VirtualSra`, etc.) which DO have lookup tables.
The real shift instructions don't directly participate in lookup-based proving.

**Status:** [x] PASS

---

### [CD-2.1] `is_multiple_of` uses nightly-only API

**File:** `src/virtual_/assert.rs:63,71`
**Severity:** LOW
**Finding:** `x.is_multiple_of(4)` and `x.is_multiple_of(2)` use
`u64::is_multiple_of()` which was stabilized in Rust 1.85. If the MSRV is below 1.85,
this would fail to compile. Given the crate already compiles and the workspace likely
targets recent nightly, this is fine.

**Status:** [ ] PASS — compiles on the target toolchain.

---

### [CQ-7.1] Doc examples use `ignore` attribute

**File:** `src/tables/mod.rs:229-232`
**Severity:** LOW
**Finding:** The `LookupTables` doc example uses `ignore`:
```rust
/// ```ignore
/// let table = LookupTables::<64>::from(LookupTableKind::And);
/// ```
```
This means the example is never compiled or tested. It could use `no_run` instead
if the issue is runtime requirements, or be made into a compilable example.

**Status:** [ ] PASS — minor.

---

## Summary

| Severity | Count | Resolved | Pass/WontFix |
|----------|-------|----------|-------------|
| HIGH     | 0     | 0        | 0           |
| MEDIUM   | 2     | 2        | 0           |
| LOW      | 10    | 3        | 7           |
| **Total** | **12** | **5** | **7** |

**Final state:** 0 clippy warnings, 260 tests passing.

### Changes made:

1. **Fixed LookupTableKind::COUNT** — changed 40 → 41
2. **Added compile-time const assertions** — `LookupTableKind::COUNT` and `NUM_PREFIXES` now verified at compile time
3. **Eliminated unsafe transmute** — replaced with `ALL_PREFIXES` const array (exported from crate root)
4. **`dispatch_prefix!` macro** — eliminated ~340 lines of duplicated import+match boilerplate in `prefixes/mod.rs`
5. **`kind_table_identity!` macro** — eliminated ~80 lines of duplicated identity mapping in `tables/mod.rs`
