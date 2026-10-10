# Spec: Kernel Primitives for Sum-Checks over Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          | #2045                          |

## Summary

The optimized sum-check kernels of `jolt-kernels` are assembled from a few primitives: the split equality polynomial and its Gruen round in `jolt-poly`, lazily bound one-hot columns (`LazyFoldedRa`), the split less-than table (`SplitLt`), the deferred-reduction accumulators of `jolt-field`, and a lock-step harness that runs an optimized kernel against the reference kernel of its relation. `specs/binary-protocol-family-seams.md` lets a crate outside the workspace declare a protocol family over a binary field, prove it with the reference kernel and drive it with the generated stage driver. It lists optimized kernels for such a family, and the export of the internals of `jolt_kernels::optimized` and of the parity harness, among its Non-Goals.

A kernel written in such a crate over `F128` cannot use the primitives today, for three reasons. `LazyFoldedRa`, `SplitLt` and the harness are private to `jolt-kernels`. Of the public Gruen round methods, `gruen_poly_deg_3` and `gruen_poly_from_evals` interpolate over integer nodes, which collapse in characteristic 2, and `gruen_poly_deg_2`, which does not, serves a linear `q` only; the two steps of the round that hold in every characteristic and for every degree are private or inline. And three products that such kernels use in their inner loops are cheaper than a general multiplication but need the representation that `jolt_field::binary` keeps private.

This spec makes the primitives available in one pull request, stacked on the branch of `specs/binary-protocol-family-seams.md`. It continues the binary-field stack (`specs/binary-field.md`, `specs/binary-sumcheck.md`, `specs/binary-accumulators.md`, `specs/binary-uniskip-domain.md`, `specs/binary-protocol-family-seams.md`) and is motivated by generality: prover kernels for sum-checks over binary fields, written in a crate outside the workspace's own prover, as the seams spec allows. It is meant to be complete for the operations it documents, so that a kernel built from them needs no later change to `jolt-field`, `jolt-poly`, `jolt-kernels` or `jolt-sumcheck` on their account; `jolt-sumcheck` needs none now. No kernel, stage, relation, proof type or transcript changes.

## Intent

### Goal

Export, with a stated contract and a validated constructor where one is needed, the kernel primitives that are valid in every characteristic, and add the three `F128` products. The public surface added is exactly this:

```rust
// jolt_field, feature `binary`
impl F128 {
    pub const fn mul_x(self) -> Self; // new
    pub fn mul_word(self, word: u64) -> Self; // new
}
impl F128Accumulator { pub fn fmadd_word(&mut self, a: F128, word: u64); } // new
pub use binary::F128Accumulator; // new re-export, at the crate root beside `F128`

// jolt_poly, defined in `split_eq.rs` and re-exported at the crate root; no feature
pub fn gruen_recover_endpoint<F: Field>( // new
    s_known: F, l_missing: F, s_0_plus_s_1: F, q_missing: impl FnOnce() -> F,
) -> Result<F, F>;
pub fn gruen_mul_linear<F: Field>(linear_evals: (F, F), q_coeffs: &[F]) -> UnivariatePoly<F>; // new
impl<F: JoltField> GruenSplitEqPolynomial<F> {
    pub fn recover_q_one(&self, q_zero: F, s_0_plus_s_1: F, q_at_one: impl FnOnce() -> F) -> Result<F, F>; // new
    pub fn round_poly_from_q_coeffs(&self, q_coeffs: &[F]) -> UnivariatePoly<F>; // new name of the private `multiply_linear_factor`
}

// jolt_kernels::optimized::lazy_ra, which becomes `pub mod`
pub trait ChunkIndexSource: Send + Sync + 'static {
    fn num_polys(&self) -> usize; fn cycles(&self) -> usize;
    fn index(&self, i: usize, j: usize) -> Option<usize>;
    fn index_bound(&self, i: usize) -> Option<usize> { None } // new, provided
}
pub enum LazyFoldedRa<F: JoltField, S> { Lazy(LazyRaBranches<F, S>), Dense(LazyRaDense<F>) }
pub struct LazyRaBranches<F: JoltField, S> { /* pub(crate) */ } pub struct LazyRaDense<F: JoltField>(/* pub(crate) */); // both new
impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    pub fn try_new(tables: Vec<Vec<F>>, source: S) -> Result<Self, LazyRaError>; // new
    pub fn num_polys(&self) -> usize;
    pub fn value(&self, i: usize, j: usize) -> F;
    pub fn lo_hi(&self, i: usize, row: usize) -> (F, F);
    pub fn lo_hi_all(&self, row: usize, out: &mut [(F, F)]);
    pub fn final_values(&self) -> Vec<F>; pub fn bind(&mut self, challenge: F);
}
pub enum LazyRaError { // new
    TableCount { tables: usize, polys: usize }, NoColumns, CyclesNotPowerOfTwo { cycles: usize },
    IndexBoundExceedsTable { poly: usize, bound: usize, len: usize },
    IndexOutOfRange { poly: usize, cycle: usize, index: usize, len: usize },
}

// jolt_kernels::optimized, re-exported from `support`
pub enum SplitLt<F> { Split(SplitLtTables<F>), Dense(SplitLtDense<F>) }
pub struct SplitLtTables<F> { /* private */ } pub struct SplitLtDense<F>(/* private */); // both new
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

The items not marked new exist today with these signatures and change in visibility only: `ChunkIndexSource` with its three required methods, six methods of `LazyFoldedRa`, four of `SplitLt`, `probe_input_claim` and `run_lockstep`. The two enums change in the shape of their variants (invariant 1). `LazyFoldedRa::new` and `SplitLt::final_value` stay `pub(crate)` and unchanged.

### Invariants

1. **Additive on the documented call sequences.** `jolt-sumcheck` is not edited, no existing signature changes in any crate, and in `jolt-field` no existing function body changes. In `jolt-poly` three bodies of `split_eq.rs` change, and only to call the function that owns their formula (invariant 3): the recovery written inline in `gruen_poly_deg_3` and the private `recover_q_zero` call `gruen_recover_endpoint`, and the private `multiply_linear_factor` becomes the public `round_poly_from_q_coeffs`, with no second name kept, and calls `gruen_mul_linear`, where its loop now is; its two call sites follow the rename. The return on a zero `current_scalar` stays in front of these calls in `gruen_poly_deg_2`, `gruen_poly_deg_3` and `gruen_poly_from_evals`. On every input these three therefore return the same `Ok` polynomial with the same number of coefficients and the same `Err` value, and call their closure in the same cases, never when `current_scalar` is zero. In `jolt-kernels` a call sequence within the documented conditions keeps its results and its arithmetic. A bind of an exhausted lazy state now panics where it returned (invariant 4), and no equality of behaviour is claimed outside those conditions. The edits to existing code in `jolt-kernels` are: visibility keywords, with the imports and re-exports that the surface needs; documentation comments; the release assertion in `LazyFoldedRa::bind`; the payloads of the variants of `LazyFoldedRa` and `SplitLt` move into named types, which rewrites match patterns and constructor expressions and nothing else, in `lazy_ra.rs`, in the `SplitLt` impl of `support.rs`, at `booleanity.rs:752`, and in the test fixtures at `booleanity.rs:1672` and `bytecode_read_raf.rs:1939`; and in `parity.rs` the round loop of `run_lockstep` moves into a private function that also returns the final running claim, and the three fixtures that are not exported (`synthetic_point`, `ExceptionalEq` with its impl, `probe_one_hot_family`), with the imports that only they use, are compiled under `cfg(test)` only. The four payload types carry, under the `allocative` feature, the `Allocative` derive of their enum with its bounds, `S: Allocative` for the lazy payload included, so the kernels that derive `Allocative` over the two enums still do. Every existing test passes with no change to an assertion. For the three round methods that is the evidence: the tests of `split_eq.rs` pin their coefficients, error values and closure calls on the exceptional rounds. Round messages, claims and proofs over the prime fields are byte-identical.
2. **`F128` products.** `a.mul_x() == a * F128::from_raw(2)`: one shift and one conditional XOR of `0x87`, the same code on every target. `a.mul_word(w) == a * F128::from_raw(w as u128)`. After `acc.fmadd_word(a, w)`, `reduce` and every later operation on `acc` give what they give after `acc.fmadd(a, F128::from_raw(w as u128))`. The word is a polynomial of degree below 64, not an integer: these are not `Ring::mul_u64` and `Accumulator::fmadd_u64`, which keep the parity of the scalar (`specs/binary-accumulators.md`, invariant 2), and the rustdoc of each says so.
3. **Gruen steps, one owner each.** For a round polynomial `s = l·q` with `l` linear, the claim fixes `s(0) + s(1)`, an equation that is symmetric in the two endpoints, and `gruen_recover_endpoint` owns it in both directions. Given `s_known`, the value of `s` at one endpoint, and `l_missing`, the value of `l` at the other, it returns the `q` at the other endpoint that satisfies `s_known + l_missing·q = s_0_plus_s_1`. When `l_missing` is nonzero it returns `(s_0_plus_s_1 − s_known)·l_missing⁻¹` and does not call `q_missing`. When `l_missing` is zero it calls `q_missing` once and returns its value `e` if `s_known + l_missing·e` equals `s_0_plus_s_1`, and `Err` of that sum if it does not, which is the error convention of the existing round methods. `gruen_mul_linear` owns the product with the linear factor: for `linear_evals = (l(0), l(1))` and `q` given low degree first, it returns the `q_coeffs.len() + 1` coefficients of `l·q`, low degree first and with trailing zeros kept; an empty slice gives one zero coefficient. Neither samples at an integer, halves, or inverts anything but `l_missing`, and no other code of `split_eq.rs` computes either formula. `recover_q_one` passes `l(0)·q_zero` and `l(1)`, and `round_poly_from_q_coeffs` passes `l`, both from `self.current_linear_evals()`, whose precondition that a variable is left to bind they inherit. Unlike the three existing round methods, these two do not return early on a zero `current_scalar`. That is the case `l(0) = l(1) = 0`: the closure is called, a nonzero claim is `Err(0)`, and the round polynomial is zero.
4. **Lazy one-hot columns.** A `LazyFoldedRa` built from tables `T_i` and a source represents the `N = num_polys()` columns `c_i[j] = T_i[index(i, j)]`, zero where `index(i, j)` is `None`, over `j < cycles()`, bound low to high: bind `k`, counted from zero, binds bit `k` of `j`, the least significant first. A source returns the same `num_polys()`, the same `cycles()` and the same `index(i, j)` on every call for as long as the `LazyFoldedRa` holds it. `try_new` cannot check that, and the type calls `index` again in every gather and `cycles` again in every bind of the lazy state. State that changes no returned value, such as a count of calls, is allowed. After `b` binds, `value(i, j)` for `j < cycles() / 2^b` is entry `j` of column `i` bound `b` times by `t[y] ← t[2y] + ρ·(t[2y+1] − t[2y])`. `lo_hi(i, row)` is `(value(i, 2·row), value(i, 2·row + 1))`, and `lo_hi_all(row, out)` writes those pairs for the first `min(out.len(), N)` columns. `final_values()` is `value(i, 0)` for every column, which is the fully bound value once `log2(cycles())` binds are done. The first three binds rescale the tables. The fourth builds dense tables of `cycles() / 16` entries, hands the tables and the source to `mem::drop_in_background_thread` and calls `mem::purge_retained_memory`, which is why a source is `Send + Sync + 'static`. The conditions on the call sequence are the caller's, and the rustdoc states them: `i < N`, `j` below the current length, and at most `log2(cycles())` binds. While they hold, `index` is called only with `i < num_polys()` and `j < cycles()`. A bind beyond the last panics in release builds, for every value that `try_new` returns, and `bind` says so under `# Panics`. Once the tables are dense, `Polynomial::bind_low_to_high` asserts that a variable is left, on the first column, which invariant 5 guarantees to exist. The branch-table state gains the assertion `width < source.cycles()`, which is new: a source of at most eight cycles is fully bound while still in that state, and one more bind would otherwise call `index` beyond `cycles()` or leave empty dense tables. It is not a `LazyRaError`. `try_new` knows the bound and cannot know how many binds will follow, and a `Result` from `bind` would change a signature that invariant 1 keeps.
5. **Validated construction.** `try_new` is the only public constructor of `LazyFoldedRa`. It checks, in this order and before it stores anything: the number of tables equals `source.num_polys()` (`TableCount`); that number is not zero (`NoColumns`), because a family with no column has no value to read and, once dense, no column whose bind could panic; `source.cycles()` is a power of two (`CyclesNotPowerOfTwo`, which covers zero); and, column by column, every digit is below its table's length. A table may have any length. The type reads a table only at the digits of its column and at those digits offset by a multiple of the table's own length, so it needs no property of that length, and an empty table is accepted for a column that has no digit on any cycle. For the digit check a column whose `index_bound(i)` is `Some(b)` is accepted when `b ≤ len` and rejected with `IndexBoundExceedsTable` otherwise, without a scan. A column that gives `None` is scanned over all cycles, and `IndexOutOfRange` names the least cycle that fails, with and without the `parallel` feature. `index_bound(i) = Some(b)` is the implementor's statement that every `Some(k)` returned by `index(i, ·)` has `k < b`; when that is false the source breaks the contract of the trait, and `value` may panic or read an entry of another branch table. `LazyRaError` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq` and `thiserror::Error`.
6. **Split less-than.** For a point `r` of `n` coordinates, high variable first, `SplitLt::new(r)` represents the table `t[j] = LT(j, r) = Σ_{k > j} eq(k, r)` over `j < 2^n`, and `new_plus_constant(r, c)` the table `t[j] + c`. `pair(y)` returns `(t[2y], t[2y+1])` of the current table. Its precondition, which it does not check, is that `y` is below half the current length. Outside it, the slice index panics for every `y` at or above half the length for which `2y + 1` does not overflow `usize`, and nothing is promised for a larger `y`. `bind(ρ)` replaces the table by `t[y] ← t[2y] + ρ·(t[2y+1] − t[2y])`. `bound_value()` is `Some(t[0])` when exactly one entry is left, which is after `n` binds, and `None` otherwise. It is new because `final_value` checks that state with a `debug_assert`. A bind of a one-entry table leaves an empty table, on which every `pair` panics and `bound_value` is `None`. An empty point gives the one-entry table `[c]`. The constructors require `n < usize::BITS`, so that the length `2^n` of the table is a `usize`, and they do not check it. For `n ≥ 2` they pass `LtPolynomial::evaluations` the high `⌈n/2⌉` coordinates and the low `⌊n/2⌋` as two slices, never the whole point, and hold `2·2^⌈n/2⌉ + 2^⌊n/2⌋` elements; for `n ≤ 1` they pass the whole point. That function panics on a slice of `usize::BITS` coordinates or more, which is a check on each half and not on `n`. Allocation is a further limit.
7. **Lock-step harness.** `run_lockstep_checked` drives a reference kernel and an optimized kernel of one relation from the same `ProverInputs`, initial claim and challenges. It panics, naming the check and the round, unless: (a) every round's coefficient vectors are equal, which is what `run_lockstep` checks; (b) after `finish_rounds`, `output_claims(inputs.claims)` succeeds on both and the two results have equal `canonical_order()` and equal `opening_values()`; (c) with `output_points = inputs.relation.derive_opening_points(challenges, inputs.points)`, `validate_derived_tables` returns `Ok` on both; (d) `inputs.relation.expected_output(inputs.points, &outputs, &output_points, inputs.challenges)` equals the running claim after the last round. It returns the reference kernel's output claims, so that a caller adds the checks that depend on its fixture. `initial_claim` is the honest input claim. `probe_input_claim` calls `prove_round(None, 0, 0)`, returns `actual` of a `SumcheckError::RoundCheckFailed`, and returns zero on `Ok`. That is the input claim for a kernel that meets two conditions, and for no other: its first round computes the endpoint sum `s(0) + s(1)` from its tables, independently of the claim it is given, and reports it in that error; and the call may be repeated, with the same result. `NaiveSumcheckProver` meets both. A kernel that recovers an endpoint from the claim returns `Ok` for every claim and is probed as zero, and one whose first round consumes state fails the second condition. A caller therefore probes the reference kernel. Both runners require two kernels on which no challenge has been bound, which they cannot check, and they panic unless the two round counts are equal, positive and equal to `challenges.len()`. The three functions name no protocol family. The module is compiled for tests and under `test-utils`, a feature that adds no dependency; its functions panic by design and are not for use in a prover.
8. **External contract.** The rule that a public item needs a production caller in the repository or a documented external contract is met by the second. This spec and the rustdoc of each item are the contract, and the caller is a kernel crate outside the workspace. Each item is exercised by tests in the repository: the three `F128` methods on both multiplication backends, and every other item over `F128` and over a prime field.

### Non-Goals

- Any kernel, and any change to a stage, a relation, a proof type or a transcript.
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
- [ ] `fmadd_word`, on both backends: for seeded lists of 1, 2 and 20 terms, an accumulator that receives a mix of `fmadd_word`, `fmadd` and `add`, merged with a second accumulator that receives a different seeded list through the same mix, reduces to the sum of the terms of both lists computed with `*` and `+`. The seeds are ones for which that sum is nonzero, so that two states that cancel, or an accumulator that discards its terms, fail.
- [ ] Gruen functions, over `Fr`, `Prime128OffsetA7F7` and, under `binary`, `F128`, for seeded `(l(0), l(1))` and `q` of degree 0 to 5: `gruen_mul_linear` returns `q_coeffs.len() + 1` coefficients whose Horner value equals `l(x)·q(x)` at that many distinct points, also when the top coefficient of `q` is zero, and one zero coefficient for the empty slice. `gruen_recover_endpoint`, with endpoint 0 known and with endpoint 1 known, returns `q` at the other endpoint without calling the closure when `l_missing ≠ 0`. With `l_missing = 0` it calls the closure once, returns its value for the claim `s_known`, and returns `Err(s_known)` for that claim plus one. With `s_known = 0` as well it returns `Ok` for a zero claim and `Err(0)` for the claim one.
- [ ] Gruen methods, over the same fields and in both binding orders: for seeded tables `A`, `B` of 16 entries and a seeded point `w`, with the claim `Σ_j eq(w, j)·A[j]·B[j]` computed by direct summation, each round's polynomial is built from `q(0)` and the leading coefficient of `q`, both by direct summation, `recover_q_one` and `round_poly_from_q_coeffs`. At 0, at 1 and at two further points it equals the direct sum of the summand with the round variable set to that point. The same holds with one coordinate of `w` equal to 0 and with one equal to 1. On a value built by `new_with_scaling` with a zero scaling factor, `recover_q_one` calls its closure and returns `Ok` for a zero claim and `Err(0)` for the claim one, and `round_poly_from_q_coeffs` returns `q_coeffs.len() + 1` zero coefficients. Over `Fr` the rounds built with `gruen_poly_deg_3` meet the same direct sums.
- [ ] `LazyFoldedRa`, in an integration test of `jolt-kernels`, which sees the public API only, over `Fr` and `F128`: for `cycles` in `{1, 2, 8, 16, 64}`, three columns with tables of 2, 5 and 16 seeded entries and seeded digits that include `None`, and a fourth with an empty table and no digit, before any bind and after each of the `b` binds, `value(i, j)` equals the partial evaluation `Σ_{u < 2^b} W_b(u)·c_i[j·2^b + u]` of the column of invariant 4, computed by direct summation, where `W_b(u) = Π_{k < b} (ρ_k·u_k + (1 − ρ_k)·(1 − u_k))`, `ρ_k` is the challenge of bind `k` and `u_k` is bit `k` of `u`, least significant first. `lo_hi` returns the corresponding pairs. `lo_hi_all` writes them into an `out` of `N − 1`, of `N` and of `N + 1` entries, and leaves the last entry of the longest as it was. After the last bind `final_values()` equals `Σ_j W_b(j)·c_i[j]`, and for each `cycles` one further `bind` panics.
- [ ] `try_new` returns each error: one table too few; no table, for a source of no column and 16 cycles; 12 cycles; 0 cycles; a digit equal to its table's length, once through `index_bound` and once through the scan, the latter with two failing cycles and the lesser one reported. A source with two faults reports the one checked first. A source whose `index_bound` is `Some` for every column counts its `index` calls, and the count is zero after `try_new`.
- [ ] `SplitLt`, in the same integration test, over `Fr` and `F128`, for `n` in `{0, 1, 2, 3, 6, 7}`, with constant zero and with a seeded constant: before any bind and after each of the `b` binds, `pair(y)` for every `y` equals the two adjacent entries of the partial evaluation `Σ_{u < 2^b} W_b(u)·t[j·2^b + u]`, with the weight `W_b` of the previous criterion, of the table `t[j] = Σ_{k > j} eq(k, r) + c`, both sums computed directly. `bound_value()` is `None` before bind `n` and `Some` of the remaining entry after it. `pair` at half the current length panics. One bind more, of the one-entry table, leaves a state on which `bound_value()` is `None` and `pair(0)` panics.
- [ ] Lock-step harness, in unit tests over `Fr` and `F128`, with the toy relation of `reference/naive.rs`, which is generic in the field, and seeded tables: `probe_input_claim` on a `NaiveSumcheckProver` returns the sum of the summand over the cube, computed in the test from the tables, for seeds that make it nonzero. Two `NaiveSumcheckProver`s pass. Of the returned claims, each one that a table backs equals `Polynomial::evaluate` of that table at its derived opening point, and the dual-role one equals the input claim it consumes. A wrapper kernel that adds one to a coefficient panics at (a). One that adds one to an output value on the optimized side panics at (b). One whose `validate_derived_tables` returns an error panics at (c). Wrappers that add one to the additive output `c` of the toy relation on both sides, which moves the expected output by exactly one where a change to a factor of a product could be annihilated, panic at (d).
- [ ] No assertion of an existing test changes, and outside the new items `git diff` shows only the edits listed in invariant 1.
- [ ] The checks under Execution pass, and the pull request description reports the benchmark rows of Performance.

### Testing Strategy

Ground truth is independent of the code under test: field multiplication for the three `F128` products, with the portable shift-and-XOR product beneath it; Horner evaluation and direct summation over the cube for the Gruen steps; for the two bound-table types, the tables written from their definitions and their partial evaluations as sums against the weight that the criterion writes out, never a second binding routine; and, for the harness, kernels perturbed in one known place. No test compares an old body with a new one. The three bodies that change in `split_eq.rs` are held by their existing tests, which compare against products computed directly. Where two round constructions both ship, each is checked against the same direct sums and not against the other. The sizes of the criteria reach the two state transitions that this spec owns, from both sides: the fourth bind of `LazyFoldedRa`, with 8, 16 and 64 cycles, and the fold of `SplitLt` into its dense table, with `n` from 0 to 7. They do not reach the thresholds of the arithmetic that these types delegate to: the parallel bind of `Polynomial` from 1,024 pairs, the parallel path of `EqPolynomial::evals` above 16 coordinates, and the release of memory in `mem::purge_retained_memory` from `2^22` cycles. Those belong to the tests of the primitives that own them.

### Performance

No arithmetic formula on a path of a shipped prover changes. Two things are added to those paths, neither per row. `LazyFoldedRa::bind` makes one integer comparison per bind while the state is lazy, at most four in the life of a value. And the three round methods reach their recovery and their product with the linear factor through the functions of invariant 3, once per round. Counted from the code, that adds no multiplication and no inversion to any of them. Ahead of its interpolation `gruen_poly_deg_3` makes five multiplications, and one inversion when `l(1)` is nonzero, and it makes the same after the change: it passes the product `l(0)·q(0)` it already holds as `s_known`, and when `l(1)` is nonzero it computes the difference `s_0_plus_s_1 − l(0)·q(0)` a second time, which is one subtraction per round. `gruen_poly_deg_2` and `gruen_poly_from_evals` do what they did, through one more call.

The three `F128` methods are the only items with a performance purpose. `benches/binary_kernels.rs` gains three pairs of rows, each a method and its generic control on the same 1,024 seeded operands, with the same placement of `black_box` and the same timed scope: `F128/mul_x_slice` and `a * F128::from_raw(2)`; `F128/mul_word_slice` and `a * F128::from_raw(w as u128)`; `F128/accumulator/deferred_word` and `fmadd(a, F128::from_raw(w as u128))`. The existing rows `mul_slice` and `deferred` are reported beside them as context, and no existing bench function changes. Counted from the code and not measured: a general product is four carry-less multiplications on aarch64 and three on x86_64, and its reduction two more; `mul_word` is two and one; `fmadd_word` is two, against four or three for `fmadd`. The pull request description reports the pairs for one aarch64 host and one x86_64 host, and with each report the host, the toolchain and the target features that select the multiplication backend (`aes` on aarch64, `pclmulqdq` on x86_64, and the portable backend without them). A method is below its control when the upper end of the Criterion interval of its row is under the lower end of the interval of its control, on every backend reported. A method that is not below its control is withdrawn, since that difference is its only reason to exist. Withdrawal amends the Goal, the Documentation and the criteria of this spec in the same change, before the pull request is approved.

## Design

### Architecture

| Files | Change |
|---|---|
| `crates/jolt-field/src/binary/{f128.rs, accumulator.rs, kernels.rs, portable.rs, mod.rs, tests.rs}`, `src/lib.rs`, `benches/binary_kernels.rs` | The three methods; a private word product and a private word accumulation in each backend; the re-export; tests; bench rows |
| `crates/jolt-poly/src/{split_eq.rs, lib.rs}` | The two functions, which the existing recovery and product now call; the two methods; the re-exports; tests |
| `crates/jolt-kernels/src/optimized/{lazy_ra.rs, mod.rs, booleanity.rs, bytecode_read_raf.rs}` | `pub mod`, visibility, payload types, `index_bound`, `try_new`, `LazyRaError` |
| `crates/jolt-kernels/src/optimized/{support.rs, mod.rs}` | `SplitLt`: visibility, payload types, `bound_value`, the re-exports |
| `crates/jolt-kernels/src/optimized/{parity.rs, mod.rs}`, `Cargo.toml`, tests in `src/reference/naive.rs` | The feature, the module gate, visibility, `run_lockstep_checked` and its tests |
| `crates/jolt-kernels/tests/kernel_primitives.rs` | New: the integration tests of `LazyFoldedRa` and `SplitLt` |

**Word products live in `jolt-field`.** A product of an element with a 64-bit word has 191 bits. It needs two carry-less multiplications where a general product needs three or four, and its reduction folds one word where the general one folds two. Writing it requires the lane type of the backend, the unreduced representation, which differs between the carry-less and the portable backends, and the state of `F128Accumulator`. All three are private to `jolt_field::binary`, and `specs/binary-accumulators.md` keeps them so. `F128Accumulator` is re-exported because it gains an inherent method that rustdoc has to show; `F64Accumulator` and `F192Accumulator` gain none and stay as they are.

**One owner per Gruen formula.** `gruen_recover_endpoint` takes the value of `s` at the known endpoint, and not `l` and `q` there, for two reasons. The equation then reads the same in both directions, so the recovery of `q(0)` in `recover_q_zero` and that of `q(1)` in `gruen_poly_deg_3` and `recover_q_one` are one function with the endpoints exchanged, and no caller passes a pair in reversed order. And `gruen_poly_deg_3` already holds `l(0)·q(0)`, as the value `s(0)` that it interpolates, so it calls the function without a second multiplication. Leaving the existing bodies as they were, with the two functions added beside them, would give each formula two owners in one file, and a kernel in characteristic 2 would run code that the shipped rounds do not.

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
cargo nextest run -p jolt-kernels --no-default-features --cargo-quiet
cargo nextest run -p jolt-sumcheck --cargo-quiet
cargo bench -p jolt-field --features binary --bench binary_kernels
```

The `test-utils` lanes are the only ones that compile `parity` outside a test build, and the `--no-default-features` lane is the only one that runs the serial scan of `try_new`, which has to name the same cycle as the parallel one. The remaining feature lanes of `jolt-kernels` that CI runs are run as well, because `parity.rs` and the fixtures carry feature gates.

## References

- `specs/binary-field.md`, `specs/binary-sumcheck.md`, `specs/binary-accumulators.md`, `specs/binary-uniskip-domain.md`, `specs/binary-protocol-family-seams.md`
- `crates/jolt-field/src/binary/`; `crates/jolt-poly/src/split_eq.rs`; `crates/jolt-poly/src/lt.rs`
- `crates/jolt-kernels/src/optimized/{lazy_ra.rs, support.rs, parity.rs}`; `crates/jolt-kernels/src/kernel.rs`; `crates/jolt-verifier/src/stages/relations.rs`
