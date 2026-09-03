# W4-T2 limb-table review #5

Target: `28d195054` (`crates/jolt-wrapper/src/limb_table/`; review #5, closing)

## Verdict

**0 blockers / 1 major / 0 minors**

### Major

1. **`limb_table/columns.rs:397,406-420`, `limb_table/lookup.rs:465-469`, called by
   `limb_table/relation.rs:228` and `limb_table/lookup.rs:380` — the 9,973-Fr count is warm-process
   only; a cold verifier executes 751 hidden field multiplications and one hidden inversion.**
   `LazyLock` defers work until the first verifier dereference rather than precomputing it. The first
   `StreamTermExporter::terms_observed` call initializes `CONSTANTS` through five families of
   `Fr::pow2` calls (**172 multiplications**) and `SIXTEEN` through 64 more `Fr::pow2` calls plus a
   separate `Fr::pow2(4)` (**579 multiplications**) and `16.inverse()` (**one inversion**). `Fr::pow2(e)`
   at `jolt-field/src/algebra.rs:110-121` performs `floor(log2(e)) + popcount(e)` multiplications for
   nonzero `e`, giving those exact totals. The observer starts below the initializers and therefore
   still reports `9,973 = 162 + 9,669 + 139 + 3`; the component-sum assertion excludes the same work
   on both sides. A standalone verifier's cold total is at least **10,724 Fr multiplications + one
   inversion**, above the 10,000 cap, and `fr_inv = 0` is false for that call. **Fix:** encode the fixed
   field values as compile-time/precomputed representations (including `16^-1`), or include cold
   initialization in the verifier budget and its component assertion.

## Unique-recoding derivation

Let

`t = V_hi·2^192 + V_lo`,

where `V_hi = Σ_{i=48}^{63} 16^(i-48)d_i` and
`|V_lo| ≤ 8(16^48 − 1)/15 < 2^192·8/15`. The admitted range

`0 ≤ V_hi ≤ R_HI − 2`, `R_HI = 0x30644e72e131a029 = floor(r/2^192)`

puts `t` in

`[-L, (R_HI − 2)·2^192 + L]`, `L < 2^192·8/15`.

Its length is below `(R_HI − 2 + 16/15)·2^192 < R_HI·2^192 ≤ r`; one residue class therefore has at
most one admitted signed recoding. `WINDOW_BOUND = R_HI − 2` matches this range. An honest centered
recoding has `V_hi ∈ {s_hi, s_hi + 1}` and fails only in a top interval shorter than
`3·2^192`, hence probability `< 3/R_HI ≈ 2^-60` per scalar.

At each `Source::Window` row, chunks `0..4` reconstruct `V` and chunks `4..8` reconstruct `V'` with
weights `2^(16j)`. All eight are among the 61 committed chunk columns covered by the existing
16-bit LogUp (`RANGE_COLUMNS = CHUNK_COLUMNS + DIGIT_COLUMNS`); each half is therefore an integer in
`[0, 2^64)`. There is no carry variable: the link reads those four columns directly. The equations
`V = V_hi` and `V + V' = WINDOW_BOUND` give exactly `0 ≤ V_hi ≤ WINDOW_BOUND`.

## Occurrence and row coverage

- Fibonacci profile: **173 named wires**, **175 digit bases** after constant one and `theta`, and
  **230 link occurrences**. Scratch test confirms every link index `0..229` owns exactly one op at
  each window `0..63`.
- `Builder::link_base` assigns disjoint contiguous occurrence ranges to the GT, four G1, and two G2
  chains. `window_rows()` runs after all seven chains and writes all 256 fixed rows starting at
  `WINDOW_ROW_BASE = 153,856` (`Cells::WINDOW = 9,616`).
- Link powers split into `0..M` (scalar recodings), `M..M+256` (top-digit equality), and
  `M+256..M+512` (window sums). For unused rows `o = 230..255`, the missing top-digit contribution is
  zero, so the two latter coefficients pin `V(o) = 0` and `V'(o) = WINDOW_BOUND`.
- A prover cannot reuse one valid window row for another occurrence: its row position changes the
  `rho^o` coefficient. Scratch test copies all eight in-range chunks between two occurrence rows,
  preserving `V + V' = WINDOW_BOUND`; the link claim changes.

The batched identity has degree at most `M + 511`; a false set of occurrence/window equations passes
with probability at most `(M + 511)/r`.

## Link and term contract

`LinkPowers::base_weights` still exports `W_k(rho) = Σ_{o:base(o)=k} rho^o`. R's contract remains
`Σ_{k<173} W_k(rho)s_k`; the added powers do not touch R. `link_input_claim` adds

`W_173(rho) + W_174(rho)·theta + WINDOW_BOUND·rho^(M+256)·Σ_{o<256}rho^o`,

matching the member expansion term by term. `LinkMember`, `LinkFinals`, `LinkEvals`, and
`link_terms = [omega·D, kappa·V, kappa'·V']` agree at the final point. The packed-column oracle passes:
**177 terms, maximum degree 4**.

## Theta-prefix argument

Proof-scalar values are fixed before `theta` through R; per-occurrence uniqueness makes every
proof-base digit string a function of that value. `theta` is drawn after phase 1a, then its own digits
and window rows are committed in phase 1b. Its link equation binds those digits to `theta`, and the
same admitted interval leaves at most one digit string. Thus proof-base digits cannot be selected
after `theta`; `theta`'s digits remain a function of `theta` alone. The EC module's prefix/suffix count
now applies, subject to the documented completeness loss.

## Prior findings and code shape

- Permanent `modulus_alias_recodings_are_rejected` passes for `1 + r` and `1 - r`. Its forged matching
  window rows satisfy the digit link but fail the row member because one chunk lies outside 16 bits.
- Permanent phase test passes at packing 4/16/32; k = 32 is `[3, 3, 1, 2]`.
  `StreamBuilder::end(phase)` release-asserts the exact `phases()[phase].columns` list before padding;
  `commitment_phases` derives group counts from the same ranges and appends the VK group count.
- Averaging text and `wire_multiplicity()` are gone. All limb-table source files remain below 1,000
  lines (maximum `relation.rs`, 930). Nominal imports and `#[expect]` use match repository policy; no
  test-only control field was added to a production type.
- Reviews #1-#4's soundness items remain fixed: range-table tail gates, phase ordering, Fq
  canonicality, GT norm-one projection, pairing-input subgroup checks, sign bits, correction-add
  guards, MSB-first binding, per-occurrence scalar weights, and fixed phase geometry.

## Scratch tests and verification

Patch: `.journals/lanes/w4t2-review-5-tests.patch`

- `fibonacci_profile_binds_every_link_occurrence_to_one_window_row` — pass; 173/175/230 counts,
  complete 64-window coverage, and 256 distinct `Source::Window` rows.
- `one_window_row_cannot_be_reused_for_two_occurrences` — pass; copying a fully in-range window row
  to a different occurrence changes the link claim.
- T2 suite — **48/48 pass**: library plus `limb_table_e2e`, `limb_table_program`, and
  `limb_table_miller`; permanent modulus-alias and packed-column oracle negatives included.
- Targeted budget output — `9,973 = 162 + 9,669 + 139 + 3`, 177 terms, degree 4, 230 occurrences,
  175 bases. Major 1 accounts for the cold initializers omitted by this observer.
- Clippy — pass for `--lib`, `limb_table_e2e`, `limb_table_program`, and `limb_table_miller` with
  `-D warnings`.
