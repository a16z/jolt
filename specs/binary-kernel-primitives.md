# Spec: Kernel Primitives for Sum-Checks over Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The optimized sum-check kernels of `jolt-kernels` are assembled from a few primitives: the split equality polynomial and its Gruen round in `jolt-poly`, lazily bound one-hot columns (`LazyFoldedRa`), the split less-than table (`SplitLt`), the deferred-reduction accumulators of `jolt-field`, and a lock-step harness that runs an optimized kernel against the reference kernel of its relation. `specs/binary-protocol-family-seams.md` lets a crate outside the workspace declare a protocol family over a binary field, prove it with the reference kernel and drive it with the generated stage driver. It lists optimized kernels for such a family, and the export of the internals of `jolt_kernels::optimized` and of the parity harness, among its Non-Goals.

A kernel written in such a crate over `F128` cannot use the primitives today, for three reasons. `LazyFoldedRa`, `SplitLt` and the harness are private to `jolt-kernels`. The public Gruen round methods interpolate over integer nodes, which collapse in characteristic 2, and the two steps of that round that hold in every characteristic are private or inline. And three products that such kernels use in their inner loops are cheaper than a general multiplication but need the representation that `jolt_field::binary` keeps private.

This spec makes the primitives available in one pull request, stacked on the branch of `specs/binary-protocol-family-seams.md`. It continues the binary-field stack (`specs/binary-field.md`, `specs/binary-sumcheck.md`, `specs/binary-accumulators.md`, `specs/binary-uniskip-domain.md`, `specs/binary-protocol-family-seams.md`) and is motivated by generality: prover kernels for sum-checks over binary fields, written in a crate outside the workspace's own prover, as the seams spec allows. It is meant to be complete for that purpose, so that such kernels need no later change to `jolt-field`, `jolt-poly`, `jolt-kernels` or `jolt-sumcheck`; `jolt-sumcheck` needs none now. No kernel, stage, relation, proof type or transcript changes.

## Intent

### Goal

Export, with a stated contract and a validated constructor where one is needed, the kernel primitives that are valid in every characteristic, and add the three `F128` products. The public surface added is exactly this:

```rust
// jolt_field, feature `binary`
impl F128 {
    pub const fn mul_x(self) -> Self;
    pub fn mul_word(self, word: u64) -> Self;
}
impl F128Accumulator { pub fn fmadd_word(&mut self, a: F128, word: u64); }
pub use binary::F128Accumulator; // at the crate root, beside `F128`

// jolt_poly, defined in `split_eq.rs` and re-exported at the crate root; no feature
pub fn gruen_recover_q_one<F: Field>(
    linear_evals: (F, F), q_zero: F, s_0_plus_s_1: F, q_at_one: impl FnOnce() -> F,
) -> Result<F, F>;
pub fn gruen_mul_linear<F: Field>(linear_evals: (F, F), q_coeffs: &[F]) -> UnivariatePoly<F>;
impl<F: JoltField> GruenSplitEqPolynomial<F> {
    pub fn recover_q_one(&self, q_zero: F, s_0_plus_s_1: F, q_at_one: impl FnOnce() -> F) -> Result<F, F>;
    pub fn round_poly_from_q_coeffs(&self, q_coeffs: &[F]) -> UnivariatePoly<F>;
}

// jolt_kernels::optimized::lazy_ra, which becomes `pub mod`
pub trait ChunkIndexSource: Send + Sync + 'static {
    fn num_polys(&self) -> usize; fn cycles(&self) -> usize;
    fn index(&self, i: usize, j: usize) -> Option<usize>;
    fn index_bound(&self, i: usize) -> Option<usize> { None } // new, provided
}
pub enum LazyFoldedRa<F: JoltField, S> { Lazy(LazyRaBranches<F, S>), Dense(LazyRaDense<F>) }
pub struct LazyRaBranches<F: JoltField, S> { /* pub(crate) */ } pub struct LazyRaDense<F: JoltField>(/* pub(crate) */);
impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    pub fn try_new(tables: Vec<Vec<F>>, source: S) -> Result<Self, LazyRaError>; // new
    pub fn num_polys(&self) -> usize;
    pub fn value(&self, i: usize, j: usize) -> F;
    pub fn lo_hi(&self, i: usize, row: usize) -> (F, F);
    pub fn lo_hi_all(&self, row: usize, out: &mut [(F, F)]);
    pub fn final_values(&self) -> Vec<F>; pub fn bind(&mut self, challenge: F);
}
pub enum LazyRaError {
    TableCount { tables: usize, polys: usize }, CyclesNotPowerOfTwo { cycles: usize },
    IndexBoundExceedsTable { poly: usize, bound: usize, len: usize },
    IndexOutOfRange { poly: usize, cycle: usize, index: usize, len: usize },
}

// jolt_kernels::optimized, re-exported from `support`
pub enum SplitLt<F> { Split(SplitLtTables<F>), Dense(SplitLtDense<F>) }
pub struct SplitLtTables<F> { /* private */ } pub struct SplitLtDense<F>(/* private */);
impl<F: JoltField> SplitLt<F> {
    pub fn new(r_cycle: &[F]) -> Self;
    pub fn new_plus_constant(r_cycle: &[F], constant: F) -> Self;
    pub fn pair(&self, y: usize) -> (F, F); pub fn bind(&mut self, r: F);
    pub fn bound_value(&self) -> Option<F>; // new
}

// jolt_kernels::optimized::parity: `pub mod` under `cfg(any(test, feature = "test-utils"))`, a new feature
pub fn probe_input_claim<F: JoltField, R>(kernel: &mut dyn SumcheckKernel<F, Relation = R>) -> F where R: ConcreteSumcheck<F>;
pub fn run_lockstep<F: JoltField, R>(
    reference: &mut dyn SumcheckKernel<F, Relation = R>, optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F, challenges: &[F],
) where R: ConcreteSumcheck<F>;
pub fn run_lockstep_checked<F: JoltField, R>( // new
    inputs: &ProverInputs<'_, F, R>,
    reference: &mut dyn SumcheckKernel<F, Relation = R>, optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F, challenges: &[F],
) -> SumcheckOutputClaims<F, R>
where R: ConcreteSumcheck<F>, SumcheckOutputClaims<F, R>: OutputClaims<F, OpeningIdOf<F, R>>, OpeningIdOf<F, R>: PartialEq + Debug;
```

Functions not marked new exist today with these signatures and change in visibility only; the two enums change in the shape of their variants (invariant 1). `LazyFoldedRa::new` and `SplitLt::final_value` stay `pub(crate)`.

### Invariants

1. **Additive.** In `jolt-field` and `jolt-poly` no existing function body or signature changes, and `jolt-sumcheck` is not edited. In `jolt-kernels` no signature changes and no function returns a different value on an input on which it returns. The edits to existing code there are: visibility keywords; the one release assertion that `LazyFoldedRa::bind` gains (invariant 4), which no call sequence within the documented bound reaches; the payloads of the variants of `LazyFoldedRa` and `SplitLt` move into named types, which rewrites match patterns and constructor expressions and nothing else, in `lazy_ra.rs`, in the `SplitLt` impl of `support.rs`, at `booleanity.rs:752`, and in the test fixtures at `booleanity.rs:1672` and `bytecode_read_raf.rs:1939`; and in `parity.rs` the round loop of `run_lockstep` moves into a private function that also returns the final running claim, and the three fixtures that are not exported (`synthetic_point`, `ExceptionalEq`, `probe_one_hot_family`) are compiled under `cfg(test)` only. Every existing test passes with no change to an assertion. No arithmetic on a path of a shipped prover changes, so round messages, claims and proofs over the prime fields are byte-identical.
2. **`F128` products.** `a.mul_x() == a * F128::from_raw(2)`: one shift and one conditional XOR of `0x87`, the same code on every target. `a.mul_word(w) == a * F128::from_raw(w as u128)`. After `acc.fmadd_word(a, w)`, `reduce` and every later operation on `acc` give what they give after `acc.fmadd(a, F128::from_raw(w as u128))`. The word is a polynomial of degree below 64, not an integer: these are not `Ring::mul_u64` and `Accumulator::fmadd_u64`, which keep the parity of the scalar (`specs/binary-accumulators.md`, invariant 2), and the rustdoc of each says so.
3. **Gruen steps.** For a round polynomial `s = l·q` with `l` linear and `linear_evals = (l(0), l(1))`, `gruen_recover_q_one` returns the `q(1)` that satisfies `l(0)·q(0) + l(1)·q(1) = s_0_plus_s_1`. When `l(1)` is nonzero it returns `(s_0_plus_s_1 − l(0)·q(0))·l(1)⁻¹` and does not call `q_at_one`. When `l(1)` is zero it calls `q_at_one` once and returns its value `e` if `l(0)·q(0) + l(1)·e` equals `s_0_plus_s_1`, and `Err` of that sum if it does not, which is the error convention of `gruen_poly_deg_3`. `gruen_mul_linear` returns the `q_coeffs.len() + 1` coefficients of `l·q`, low degree first and with trailing zeros kept, for `q` given low degree first; an empty slice gives one zero coefficient. Neither samples at an integer, halves, or inverts anything but `l(1)`. The two methods pass `self.current_linear_evals()` and inherit its precondition that a variable is left to bind. A zero `current_scalar` is the case `l(0) = l(1) = 0`: the closure is called, a nonzero claim is `Err(0)`, and the round polynomial is zero.
4. **Lazy one-hot columns.** A `LazyFoldedRa` built from tables `T_i` and a source represents the `N = num_polys()` columns `c_i[j] = T_i[index(i, j)]`, zero where `index(i, j)` is `None`, over `j < cycles()`, bound low to high. After `b` binds, `value(i, j)` for `j < cycles() / 2^b` is entry `j` of column `i` bound `b` times by `t[y] ← t[2y] + ρ·(t[2y+1] − t[2y])`. `lo_hi(i, row)` is `(value(i, 2·row), value(i, 2·row + 1))`, and `lo_hi_all(row, out)` writes those pairs for the first `min(out.len(), N)` columns. `final_values()` is `value(i, 0)` for every column, which is the fully bound value once `log2(cycles())` binds are done. The first three binds rescale the tables. The fourth builds dense tables of `cycles() / 16` entries, hands the tables and the source to `mem::drop_in_background_thread` and calls `mem::purge_retained_memory`, which is why a source is `Send + Sync + 'static`. The conditions on the call sequence are the caller's, and the rustdoc states them: `i < N`, `j` below the current length, and at most `log2(cycles())` binds. While they hold, `index` is called only with `i < num_polys()` and `j < cycles()`. A bind beyond the last panics in release builds, and `bind` says so under `# Panics`. Once the tables are dense, `Polynomial::bind_low_to_high` asserts that a variable is left. The branch-table state gains the assertion `width < source.cycles()`, which is new: a source of at most eight cycles is fully bound while still in that state, and one more bind would otherwise call `index` beyond `cycles()` or leave empty dense tables. It is not a `LazyRaError`. `try_new` knows the bound and cannot know how many binds will follow, and a `Result` from `bind` would change a signature that invariant 1 keeps.
5. **Validated construction.** `try_new` is the only public constructor of `LazyFoldedRa`. It checks, in this order and before it stores anything: the number of tables equals `source.num_polys()` (`TableCount`); `source.cycles()` is a power of two (`CyclesNotPowerOfTwo`, which covers zero); and, column by column, every digit is below its table's length. A table may have any length. The type reads a table only at the digits of its column and at those digits offset by a multiple of the table's own length, so it needs no property of that length, and an empty table is accepted for a column that has no digit on any cycle. For the digit check a column whose `index_bound(i)` is `Some(b)` is accepted when `b ≤ len` and rejected with `IndexBoundExceedsTable` otherwise, without a scan. A column that gives `None` is scanned over all cycles, and `IndexOutOfRange` names the least cycle that fails, with and without the `parallel` feature. `index_bound(i) = Some(b)` is the implementor's statement that every `Some(k)` returned by `index(i, ·)` has `k < b`; when that is false the source breaks the contract of the trait, and `value` may panic or read an entry of another branch table. `LazyRaError` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq` and `thiserror::Error`.
6. **Split less-than.** For a point `r` of `n` coordinates, high variable first, `SplitLt::new(r)` represents the table `t[j] = LT(j, r) = Σ_{k > j} eq(k, r)` over `j < 2^n`, and `new_plus_constant(r, c)` the table `t[j] + c`. `pair(y)` returns `(t[2y], t[2y+1])` of the current table and panics, by slice index, when `y` is not below half its length. `bind(ρ)` replaces the table by `t[y] ← t[2y] + ρ·(t[2y+1] − t[2y])`. `bound_value()` is `Some(t[0])` when exactly one entry is left, which is after `n` binds, and `None` otherwise. It is new because `final_value` checks that state with a `debug_assert`. A bind of a one-entry table leaves an empty table, on which `pair` panics and `bound_value` is `None`, so no call returns a wrong value. An empty point gives the one-entry table `[c]`. `n` is limited by `LtPolynomial::evaluations`, which panics above the shiftable dimension.
7. **Lock-step harness.** `run_lockstep_checked` drives a reference kernel and an optimized kernel of one relation from the same `ProverInputs`, initial claim and challenges. It panics, naming the check and the round, unless: (a) every round's coefficient vectors are equal, which is what `run_lockstep` checks; (b) after `finish_rounds`, `output_claims(inputs.claims)` succeeds on both and the two results have equal `canonical_order()` and equal `opening_values()`; (c) with `output_points = inputs.relation.derive_opening_points(challenges, inputs.points)`, `validate_derived_tables` returns `Ok` on both; (d) `inputs.relation.expected_output(inputs.points, &outputs, &output_points, inputs.challenges)` equals the running claim after the last round. It returns the reference kernel's output claims, so that a caller adds the checks that depend on its fixture. `initial_claim` is the honest input claim, which `probe_input_claim` recovers. The three functions name no protocol family. The module is compiled for tests and under `test-utils`, a feature that adds no dependency; its functions panic by design and are not for use in a prover.
8. **External contract.** The rule that a public item needs a production caller in the repository or a documented external contract is met by the second. This spec and the rustdoc of each item are the contract, and the caller is a kernel crate outside the workspace. Each item is exercised by tests in the repository: the three `F128` methods on both multiplication backends, and every other item over `F128` and over a prime field.

### Non-Goals

- Any kernel, and any change to a stage, a relation, a proof type or a transcript.
- Routing the existing private helpers `recover_q_zero` and `multiply_linear_factor`, or the recovery inside `gruen_poly_deg_3`, through the new functions. It changes bodies on the path of the shipped provers, which invariant 1 leaves alone.
- A Gruen round method that takes evaluation nodes. A kernel over a binary field interpolates `q` at the nodes it chose, with `interpolate_nodes_to_coeffs` of `specs/binary-uniskip-domain.md`, and passes the coefficients to `round_poly_from_q_coeffs`.
- `mul_x` and `mul_word` for `F64` and `F192`, `fmadd_word` for their accumulators, and a public unreduced type. `specs/binary-accumulators.md` keeps the binary fields outside `Unreduced`.
- Exporting anything else of `optimized::support` or `optimized::testing`, or the fixtures of `parity`.
- Moving `SplitLt` into `jolt-poly`. It binds low to high, which `LtPolynomial` does not, and it uses the crate's `bind_pairs`.
- Checking in the harness that an aliased opening equals its source. The source belongs to another member, so the equality is a property of a batch, enforced by the generated `validate_aliases`.

## Evaluation

### Acceptance Criteria

- [ ] The public surface added is the one listed under Goal: same names, signatures, bounds and gates, and nothing else is exported. `jolt-kernels` gains `test-utils = []`.
- [ ] `mul_x` equals multiplication by `F128::from_raw(2)` on the 128 elements `F128::from_raw(1 << i)`, on `0` and `u128::MAX`, and on 10,000 seeded values; `F128::from_raw(1 << 127).mul_x()` equals `F128::from_raw(0x87)`, written out.
- [ ] `mul_word` equals multiplication by `F128::from_raw(w as u128)` for `a` in `{0, 1, 1 << 127, u128::MAX}` and 10,000 seeded values and `w` in `{0, 1, 2, 1 << 63, u64::MAX}` and seeded values. The portable implementation and, where one is compiled, the carry-less one are each checked against the portable shift-and-XOR product, as `kernel_matches_portable` checks `multiply128`.
- [ ] `fmadd_word`, on both backends: for seeded lists of 1, 2 and 20 terms, an accumulator that receives a mix of `fmadd_word`, `fmadd` and `add`, and a `merge` of a second accumulator filled the same way, reduces to the sum of the terms computed with `*` and `+`.
- [ ] Gruen functions, over `Fr`, `Prime128OffsetA7F7` and, under `binary`, `F128`, for seeded `(l(0), l(1))` and `q` of degree 0 to 5: `gruen_mul_linear` returns `q_coeffs.len() + 1` coefficients whose Horner value equals `l(x)·q(x)` at that many distinct points, also when the top coefficient of `q` is zero. `gruen_recover_q_one` returns `q(1)` without calling the closure when `l(1) ≠ 0`. With `l(1) = 0` it calls the closure once, returns its value for the claim `l(0)·q(0)`, and returns `Err(l(0)·q(0))` for that claim plus one. With `l(0) = l(1) = 0` it returns `Ok` for a zero claim and `Err(0)` for the claim one.
- [ ] Gruen methods, over the same fields and in both binding orders: for seeded tables `A`, `B` of 16 entries and a seeded point `w`, with the claim `Σ_j eq(w, j)·A[j]·B[j]` computed by direct summation, each round's polynomial is built from `q(0)` and the leading coefficient of `q`, both by direct summation, `recover_q_one` and `round_poly_from_q_coeffs`. At 0, at 1 and at two further points it equals the direct sum of the summand with the round variable set to that point. The same holds with one coordinate of `w` equal to 0 and with one equal to 1. Over `Fr` the rounds built with `gruen_poly_deg_3` meet the same direct sums.
- [ ] `LazyFoldedRa`, in an integration test of `jolt-kernels`, which sees the public API only, over `Fr` and `F128`: for `cycles` in `{1, 2, 8, 16, 64}`, three columns with tables of 2, 5 and 16 seeded entries and seeded digits that include `None`, and a fourth with an empty table and no digit, before any bind and after each of the `b` binds, `value(i, j)` equals the partial evaluation `Σ_{u < 2^b} eq(ρ, u)·c_i[j·2^b + u]` of the column of invariant 4, computed by direct summation, and `lo_hi` and `lo_hi_all` return the corresponding pairs. After the last bind `final_values()` equals `Σ_j eq(ρ, j)·c_i[j]`, and for each `cycles` one further `bind` panics.
- [ ] `try_new` returns each error: one table too few; 12 cycles; 0 cycles; a digit equal to its table's length, once through `index_bound` and once through the scan, the latter with two failing cycles and the lesser one reported. A source with two faults reports the one checked first. A source whose `index_bound` is `Some` for every column counts its `index` calls, and the count is zero after `try_new`.
- [ ] `SplitLt`, in the same integration test, over `Fr` and `F128`, for `n` in `{0, 1, 2, 3, 6, 7}`, with constant zero and with a seeded constant: before any bind and after each of the `b` binds, `pair(y)` for every `y` equals the two adjacent entries of the partial evaluation `Σ_{u < 2^b} eq(ρ, u)·t[j·2^b + u]` of the table `t[j] = Σ_{k > j} eq(k, r) + c`, both sums computed directly. `bound_value()` is `None` before bind `n` and `Some` of the remaining entry after it. `pair` at half the current length panics.
- [ ] Lock-step harness, in unit tests over `Fr` and `F128`, with the toy relation of `reference/naive.rs`, which is generic in the field, and seeded tables: two `NaiveSumcheckProver`s pass, and the returned claims equal `Polynomial::evaluate` of the tables at the derived opening points. A wrapper kernel that adds one to a coefficient panics at (a). One that adds one to an output value on the optimized side panics at (b). One whose `validate_derived_tables` returns an error panics at (c). The output perturbation applied to both sides panics at (d).
- [ ] No assertion of an existing test changes, and outside the new items `git diff` shows only the edits listed in invariant 1.
- [ ] The checks under Execution pass, and the pull request description reports the benchmark rows of Performance.

### Testing Strategy

Ground truth is independent of the code under test: field multiplication for the three `F128` products, with the portable shift-and-XOR product beneath it; Horner evaluation and direct summation over the cube for the Gruen steps; for the two bound-table types, the tables written from their definitions and their partial evaluations as sums against `eq`, never a second binding routine; and, for the harness, kernels perturbed in one known place. No test compares an old body with a new one, because no body is replaced. Where two round constructions both ship, each is checked against the same direct sums and not against the other.

### Performance

No path of a shipped prover changes, so none of their figures is at risk. The three `F128` methods are the only items with a performance purpose. `benches/binary_kernels.rs` gains the rows `F128/mul_x_slice` and `F128/mul_word_slice`, over 1,024 seeded operands as `mul_slice` is, and `F128/accumulator/deferred_word`, as `deferred` is; no existing bench function changes. Counted from the code and not measured: a general product is four carry-less multiplications on aarch64 and three on x86_64, and its reduction two more; `mul_word` is two and one; `fmadd_word` is two, against four or three for `fmadd`. The pull request description reports the new rows beside `mul_slice` and `deferred` for one aarch64 host and one x86_64 host. A method whose row is not below its general counterpart on both is withdrawn from the pull request, since that difference is its only reason to exist.

## Design

### Architecture

| Files | Change |
|---|---|
| `crates/jolt-field/src/binary/{f128.rs, accumulator.rs, kernels.rs, portable.rs, mod.rs, tests.rs}`, `src/lib.rs`, `benches/binary_kernels.rs` | The three methods; a private word product and a private word accumulation in each backend; the re-export; tests; bench rows |
| `crates/jolt-poly/src/{split_eq.rs, lib.rs}` | The two functions, the two methods, the re-exports, tests |
| `crates/jolt-kernels/src/optimized/{lazy_ra.rs, mod.rs, booleanity.rs, bytecode_read_raf.rs}` | `pub mod`, visibility, payload types, `index_bound`, `try_new`, `LazyRaError` |
| `crates/jolt-kernels/src/optimized/{support.rs, mod.rs}` | `SplitLt`: visibility, payload types, `bound_value`, the re-exports |
| `crates/jolt-kernels/src/optimized/{parity.rs, mod.rs}`, `Cargo.toml`, tests in `src/reference/naive.rs` | The feature, the module gate, visibility, `run_lockstep_checked` and its tests |
| `crates/jolt-kernels/tests/kernel_primitives.rs` | New: the integration tests of `LazyFoldedRa` and `SplitLt` |

**Word products live in `jolt-field`.** A product of an element with a 64-bit word has 191 bits. It needs two carry-less multiplications where a general product needs three or four, and its reduction folds one word where the general one folds two. Writing it requires the lane type of the backend, the unreduced representation, which differs between the carry-less and the portable backends, and the state of `F128Accumulator`. All three are private to `jolt_field::binary`, and `specs/binary-accumulators.md` keeps them so. `F128Accumulator` is re-exported because it gains an inherent method that rustdoc has to show; `F64Accumulator` and `F192Accumulator` gain none and stay as they are.

**Payload types.** A public enum has public variants, and fields of a variant are as public as the enum. Exporting the two enums as they are would let outside code build a state that `try_new` rejects, or read the tables of a state. Each payload therefore moves into a type with no public field and no public constructor: outside code can hold a value and see which state it is in, and nothing more. The fields of the `LazyFoldedRa` payloads are `pub(crate)`, because `booleanity.rs` reads the branch tables of the lazy state; those of `SplitLt` are private to `support.rs`.

**A feature for the harness.** `parity` is compiled under `cfg(test)` today, so no other crate can call it. `test-utils` follows `jolt-witness` and `jolt-claims`, which gate their test support on a feature of that name. `run_lockstep` keeps its signature, because each of its call sites would otherwise have to discard a result under the workspace lint `unused_results`; the checked variant shares its round loop through a private function.

### Alternatives Considered

1. **Opaque wrapper types around the two enums,** which would leave every existing body untouched. Rejected: one state machine would have two names, the kernels of this crate and those outside it would hold different types, and every method would be forwarded.
2. **`Result` from `value`, `pair` and `bind`.** Rejected: `value` and `pair` are called per pair inside round loops, where a slice index already panics. `bind` is called once per round and its one condition is on the call sequence, so it asserts in release builds (invariant 4) and keeps its signature.
3. **A typed error for the Gruen steps.** Rejected: `Result<_, F>` carrying the actual sum is the convention of the three existing round methods, and their callers map it to `SumcheckError::RoundCheckFailed`.
4. **`fmadd_word` as a provided method of `Accumulator`.** Rejected: every field's accumulator implements the trait, and over a prime field a word has no meaning distinct from `fmadd_u64`.
5. **Always scan, with no `index_bound`.** Rejected: the scan is `N · cycles` calls to `index`, as many as the gathers of a kernel's first round, and a source whose digits are bounded by construction can say so.
6. **Change `run_lockstep` to return the output claims.** Rejected for the reason given under Architecture.

## Documentation

The rustdoc of each item states its contract in the terms of the invariants, including the conditions it does not check. The module documentation of `lazy_ra.rs` and of `parity.rs` says what is exported and to whom, and the binary-fields section of `jolt-field`'s crate documentation names the three methods. In `specs/binary-accumulators.md`, the sentence on re-exports gains the exception for `F128Accumulator`. No book change.

## Execution

One pull request of five commits on the branch of `specs/binary-protocol-family-seams.md`, each of which compiles and passes its crate's tests: `jolt-field`; `jolt-poly`; `lazy_ra`; `SplitLt`; the feature and the harness. The last three share `optimized/mod.rs`.

```bash
cargo fmt -q --check
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo clippy --all --features allocative,host -q --all-targets -- -D warnings
cargo clippy -p jolt-field -p jolt-poly --features binary -q --all-targets -- -D warnings
cargo clippy -p jolt-kernels --features test-utils -q --all-targets -- -D warnings
cargo clippy -p jolt-kernels --features test-utils,field-inline -q --all-targets -- -D warnings
cargo nextest run -p jolt-field --features binary --cargo-quiet
cargo nextest run -p jolt-poly --cargo-quiet
cargo nextest run -p jolt-poly --features binary --cargo-quiet
cargo nextest run -p jolt-kernels --cargo-quiet
cargo nextest run -p jolt-kernels --features field-inline --cargo-quiet
cargo nextest run -p jolt-sumcheck --cargo-quiet
cargo bench -p jolt-field --features binary --bench binary_kernels
```

The `test-utils` lanes are the only ones that compile `parity` outside a test build. The remaining feature lanes of `jolt-kernels` that CI runs are run as well, because `parity.rs` and the fixtures carry feature gates.

## References

- `specs/binary-field.md`, `specs/binary-sumcheck.md`, `specs/binary-accumulators.md`, `specs/binary-uniskip-domain.md`, `specs/binary-protocol-family-seams.md`
- `crates/jolt-field/src/binary/`; `crates/jolt-poly/src/split_eq.rs`; `crates/jolt-poly/src/lt.rs`
- `crates/jolt-kernels/src/optimized/{lazy_ra.rs, support.rs, parity.rs}`; `crates/jolt-kernels/src/kernel.rs`; `crates/jolt-verifier/src/stages/relations.rs`
