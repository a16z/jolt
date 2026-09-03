# W4-S review #2

Scope: committed tree at `19872523d`; `crates/jolt-wrapper/src/{stream*,spartan*}`, scoped
integration tests, and `jolt-r1cs::ConstraintMatrices::column_range_contributions`.

## Findings

1. **MINOR — `crates/jolt-wrapper/tests/stream_synthetic.rs:1`,
   `crates/jolt-wrapper/tests/spartan_core.rs:83`: prior review 1b finding 15's test-lint
   inconsistency remains.** `stream_synthetic` explicitly expects `clippy::indexing_slicing`, while
   `spartan_core` uses the same fixed-shape direct indexing without the expectation. This has no
   protocol effect and clippy passes, but the scoped tests still use two policies for the same
   fixture pattern. **Fix:** apply the same module-level expectation to `spartan_core`, or replace
   the direct indexing in both tests and remove the expectation.

## Prior finding closure

### Review 1

1. **Fixed — A/B/C binding.** `crates/jolt-wrapper/src/stream/protocol.rs:122-195` derives stages A
   and B from the statement and transcript results; `:316-339` derives every stage-C reference
   point and group weight. `crates/jolt-wrapper/src/stream/types.rs:21-27` carries stage outputs,
   and `stream/protocol.rs:253-262` checks them against verifier-derived values before absorption.
2. **Fixed — key/profile transcript binding.** `stream/protocol.rs:264-275` absorbs the key digest,
   public statement scalars, then commitments. Spartan passes public inputs through this path at
   `crates/jolt-wrapper/src/spartan.rs:65` and `:158`.
3. **Fixed — non-power-of-two group layout.** `crates/jolt-wrapper/src/stream.rs:60-154` owns actual
   and padded group counts, the group/slot split, truncated group weights, and packed points.
   `crates/jolt-wrapper/tests/stream_synthetic.rs:288-323` covers 33 and 237 columns, a mixed final
   group, zero slots, and a missing padded group.
4. **Fixed — timing path.** `crates/jolt-wrapper/tests/stream_timing.rs:17-19` imports this crate's
   stream API; `:164-178` executes `commit_packed`, `prove_stream`, `verify_stream`, serialization,
   and byte accounting.
5. **Fixed — unreachable `StageWindow`.** The variant is absent from
   `crates/jolt-wrapper/src/stream/types.rs:149-219`; stage construction now proceeds directly from
   computed maxima at `crates/jolt-wrapper/src/stream.rs:368-415`.

### Review 1b

1. **Fixed — verifier-runnable staged protocol.** `stream/protocol.rs:122-195` is the canonical
   verifier driver; stage claims and member-point slicing live at
   `stream/types.rs:21-27` and `:108-123`.
2. **Fixed — final-stage shape.** `stream/protocol.rs:127-136` derives and checks groups and opening
   width from the statement; `:209-213` derives stage-C rounds from `PackingLayout::packed_vars`.
3. **Fixed — batch padding coverage.** `crates/jolt-wrapper/tests/stream_stage.rs:69-183` proves and
   verifies full, head-aligned, and tail-aligned members, including a short-member output tamper.
   The scale inverse is computed once at `crates/jolt-wrapper/src/stream.rs:393-404`.
4. **Fixed — production timing fixture.** `stream_timing.rs:164-178` directly runs the wrapper
   implementation and checks `WrapperProof::bincode_bytes()` against serialization.
5. **Fixed — public-column passes.** `crates/jolt-r1cs/src/constraint.rs:182-223` computes A, B, and
   C with one pass per matrix; `crates/jolt-wrapper/src/spartan.rs:292-301` is its production caller.
6. **Fixed — tamper set.** `stream_synthetic.rs:179-285` covers digest, stage outputs, columns,
   degree, round counts, all three stages, tensor, claims, commitment shape/order, and all opening
   components. `crates/jolt-wrapper/tests/spartan_core.rs:70-159` covers key/R1CS mismatch,
   unsatisfied witnesses, public input, both stages, all reduced claims, commitment, and opening.
7. **Fixed — verifier-key digest.** `stream/protocol.rs:264-275` places it before commitments;
   `spartan.rs:53-65` and `:126-158` require it on both sides.
8. **Fixed — canonical group padding.** `crates/jolt-wrapper/src/stream.rs:69-154` owns the padded
   geometry; `stream_synthetic.rs:288-323` tests both requested non-power-of-two shapes.
9. **Fixed — dead `StageWindow`.** Removed; see review 1 finding 5 above.
10. **Fixed — per-round inverse and false error.** `stream.rs:393-404` computes the inverse during
    adapter construction and reports `StageScale` honestly.
11. **Fixed — output-claim absorption owner.** `stream.rs:467-470` is the one helper used by staged
    verification and the reduced-opening verifier.
12. **Fixed — shape-G clones.** Full-field commitment accepts a slice at `stream.rs:308-311`;
    stage-C evaluates the combined slice without cloning at `stream/protocol.rs:105-111`; the old
    per-claim polynomial evaluation clone is absent from `stream.rs:772-805`.
13. **Fixed — unused column evaluation path.** `stream/protocol.rs:42` uses
    `PackedColumns::column_evaluations` to build stage B.
14. **Fixed — Spartan cleanup.** Fixed-array extraction is at `spartan.rs:149-150`; the inner-input
    error is at `:461-469`; outer and inner rounds use one table pass at `:412-445` and `:504-532`;
    typed source errors are at `stream/types.rs:211-218`.
15. **Partially fixed.** Column enforcement docs are honest at `stream.rs:26-32`, and the journal's
    staged-claim description matches `stream/protocol.rs:122-195`. The test-lint inconsistency is
    finding 1 above.

## Verifier-independence trace

```text
statement rows/columns/k/degree/terms
  -> PackingLayout + exact proof widths                 protocol.rs:127-136
  -> A(statement row claim) -> r_A, out_A               protocol.rs:143-156
  -> B(input = out_A) -> s_1..s_D                       protocol.rs:157-180
       final = Q(s_1..s_D) * product_i T(s_i)
  -> claims_i = (eq(s_i.group, g), (r_A, s_i.slot), T(s_i))
                                                         protocol.rs:181-186,316-339
  -> C(input = sum_i rho_i T(s_i)) -> r_C               protocol.rs:198-228
  -> sum_g q_g(r_C) C_g opened at derived evaluation    protocol.rs:223-249
```

All proof-controlled vectors have statement-derived exact counts at `protocol.rs:130-136`.
Missing padded groups are canonical zero polynomials: the prover pads the column-value vector at
`stream.rs:189-193`, while the verifier truncates group weights to committed groups at
`stream.rs:129-134`. Group-high/slot-low reconstruction is shared by `PackingLayout`.

## Transcript audit

- Generic prefix: key digest, public row claim, commitments (`protocol.rs:264-275`) before stage A.
- Each stage: input claims, batching coefficients, compressed rounds/challenges, checked output
  claims (`stream.rs:378-416`, `:492-518`, `:437-464`).
- Reduced values are absorbed before `rho` (`protocol.rs:89-93`, `:205-208`). No challenge occurs
  between stage-B's derived output and those raw values.
- HyperKZG follows the checked stage-C output (`protocol.rs:238-249`).
- Spartan: key/public inputs/commitment precede `tau` (`spartan.rs:65-69`, `:158-162`); A/B/C
  evaluations precede matrix weights (`:181-184`); witness evaluation precedes HyperKZG (`:214-223`).

## Tamper-path spot checks

1. Degree append (`stream_synthetic.rs:214-219`) preserves all outer shapes and reaches the row
   verifier's declared degree bound.
2. Stage-B round edit (`:239-246`) preserves lengths; stage A completes before the changed stage-B
   polynomial is read.
3. Stage-C round edit (`:248-253`) preserves lengths; stages A/B and canonical-claim construction
   complete before the changed stage-C polynomial is read.
4. Opening evaluation edit (`:275-277`) preserves every shape; all three sumchecks complete before
   HyperKZG reads the changed evaluation.
5. Fold-commitment swap (`:279-281`) preserves the required commitment count; rejection occurs in
   HyperKZG rather than the wrapper's early shape gate.

## Timing and bytes

The ignored gate passed on this review machine and wrote:

```text
setup=16.974s commit=2.547s prove=8.276s verify=0.004s payload=10304B bincode=10445B
```

For 237 columns, `k = 8`, five tensor factors, and `ell = 20`:

```text
C = 30
S = 204   # A: 84 after one zero-leading-term trim; B: 80; C: 40
A = 3
R = 5
opening = 4*ell = 80
payload = 32 * (30 + 204 + 3 + 5 + 80) = 10,304 B
```

`prove_batch` canonically trims zero leading coefficients, explaining the one-scalar difference
from the nominal degree-5 maximum. The reported 4.95 s on the stated M4 mini is plausible; this
run measured 8.28 s with no timing threshold.

## Complexity and checks

- Scoped source files: 904, 394, 228, and 539 lines; all below 1,000.
- No `#[allow]`; no scoped dead variant/helper found; packing geometry and output absorption each
  have one owner.
- `cargo nextest run -p jolt-wrapper --cargo-quiet`: 4 passed, 1 ignored.
- `cargo clippy -p jolt-wrapper --all-targets -q -- -D warnings`: passed.
- `cargo clippy -p jolt-r1cs --all-targets -q -- -D warnings`: passed.
- `cargo nextest run --release -p jolt-wrapper n3_g_shape_timing --cargo-quiet --run-ignored
  ignored-only`: 1 passed.

VERDICT: 0 blockers, 0 majors, 1 minor
