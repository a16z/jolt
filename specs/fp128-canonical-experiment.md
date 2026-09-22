# q-specific canonicality experiment

Base: `01cbfefc7e620584d67e46def44a46bde976a20f`, incorporating the independently
reviewed arithmetic component. Candidate branch: `wrap/foreign-arithmetic-canonical`.

## Preregistered decision

Question: can the special shape of q reduce the complete canonical allocation
without changing its accepted integer range or its public API? Editable surface:
canonical allocation and its bound-specific tests/docs only. Multiplication,
addition, polynomial evaluation and independent integer oracles remain unchanged.

Budget: one candidate, at most 30 minutes for derivation, implementation and
targeted validation. Use the existing exact matrix diagnostic as evaluator;
count all rows, variables and normalized A/B/C nonzeros. Expected allocation:
147 rows, 146 new variables, 906 nonzeros, versus 258/257/1029 at baseline.
Expected product with both inputs: 593 rows, 588 variables including one, 3653
nonzeros. Reject on a counterexample, missing completeness, shape dependence,
or a cost increase; otherwise submit for fresh proof review before promotion.
Prover time and on-chain gas are outside this experiment. Timing circuit
construction is optional and must not be represented as SNARK proving time.

Workload: the existing canonical/product/degree-three cost circuits; all existing
arithmetic and Horner oracle tests; added tests around the high-word transition,
low-word threshold, selector and inverse tampering, and the gated slack range.
Confounds: an unconstrained zero-test selector, dropping slack bitness, charging
only one branch, or measuring witness-dependent matrix structure would invalidate
the result regardless of row count.

## Integer argument before implementation

Write `q=(2^96-1)*2^32+22537` and `x=H*2^32+L`. Existing linked Boolean bits
give `0<=H<=2^96-1` and `0<=L<=2^32-1`. Let `S` be the sum of complements of
the 96 high bits. Then `0<=S<=96<r`. Constraints `S*h=0` and `S*inv=1-h`
force h=1 when S=0 and h=0 otherwise, without a separate Boolean row for h.

Allocate d with 15 Boolean bits and enforce `h*(L+d-22536)=0`. If h=1,
`L+d<=2^32-1+32767<r`; equality modulo r is integer equality, proving
`L<=22536` and thus x<q. If h=0, at least one high bit is zero, so
`H<=2^96-2` and x<q regardless of L. For completeness, choose d=22536-L in
the saturated-high case, which lies in [0,22536], and d=0 otherwise. Choose
inv=S^-1 for nonzero S, or zero when S=0. The latter inverse need not be
uniquely constrained: it cannot alter the accepted x.

All bounds rely on the existing fixed-one column requirement and same-builder
caller precondition. No builder provenance mechanism changes.

## Append-only observations

- Planned: baseline and candidate exact diagnostic; targeted tests and clippy;
  report complete totals and obtain independent review.

- Baseline diagnostic before edits: canonical 258 rows / 258 variables including
  one / 1029 nonzeros; product 1037/1032/4145; degree-three Horner 3633/3611/14517.
- Candidate diagnostic: canonical 147/147/806; product 593/588/3253;
  degree-three Horner 2079/2057/11395. Verdict: observed improvement, pending
  independent proof review. Row/variable predictions match. The predicted 906
  nonzeros was a manual addition error: 384+130+98+100+45+49=806. The evaluator
  was unchanged; it reported the actual normalized sparse rows.
- Tests: all existing arithmetic/shape tests and three new canonicality tests
  passed (219 total). New coverage includes all 96 possible single missing high
  bits, selector/inverse forgeries, and q/q+1/u128::MAX with a non-Boolean
  negative slack crafted to satisfy the gated sum.
- Timing: not claimed. The baseline executable was overwritten when Cargo built
  the example during test preparation. Counts were captured before mutation;
  no process timing result or prover-time inference is reported.

## Witness privacy and limits

h, inv, slack and decomposed bits are assigned witness coordinates and must
remain private in the eventual outer SNARK; this slice introduces no public
input binding for them. Both prefix cases emit
exactly the same matrix shape. The arithmetic layer does not supply ZK itself,
and witness generation remains explicitly not constant-time. An eventual outer
SNARK must hide these coordinates and constrain its fixed-one column. This
experiment changes canonical range enforcement only, not the accepted source
field, CRT multiplication proof, or same-builder/index contracts.

- Final targeted validation: 219 tests passed with default features and 219
  with `--no-default-features`; both corresponding all-target clippy commands
  passed. `cargo fmt -q` and `git diff --check` passed. Raw quiet-mode logs:
  `/private/tmp/fp128-canonical-nextest.log`,
  `/private/tmp/fp128-canonical-nextest-minimal.log`,
  `/private/tmp/fp128-canonical-clippy.log`,
  `/private/tmp/fp128-canonical-clippy-minimal.log`.
- Integration caveat reported by the conductor: these two-package checks can
  unify serde allocation features and do not establish isolated `jolt-r1cs`
  packaging. A separate ownership repair handles that pre-existing manifest
  gap; no manifest change is part of this candidate.
