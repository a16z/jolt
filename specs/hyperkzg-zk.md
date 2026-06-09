# Spec: ZK HyperKZG

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-26 |
| Status | exploratory draft |
| PR | TBD |

## Purpose

This spec describes zero-knowledge variants of `jolt-hyperkzg`.

The goal is to keep HyperKZG's Gemini reduction and KZG batching structure, but
make the transcript-visible commitment and opening data statistically hiding:

```text
plain HyperKZG:
  commitments to folded polynomials
  raw scalar evaluations v[i][u]
  KZG witnesses

ZK HyperKZG:
  hiding commitments to folded polynomials
  hiding commitments to evaluations
  hiding KZG witnesses
  no raw opened evaluations
```

V1 is retained as the conservative full-mask reference construction. The
implementation target is V2 only: it keeps commitments hiding while avoiding
the full second mask-polynomial prover cost. There is no requirement to preserve
V1 opening-hint or prover-path compatibility.

## Background

Plain HyperKZG uses Gemini to reduce a multilinear opening claim to univariate
KZG openings. For a multilinear table of size `N = 2^ell`, define:

```text
P_0 = original evaluation table, interpreted as a univariate coefficient vector
P_{i+1} = Fold_{x_i}(P_i)
```

where:

```text
Fold_x(A)[j] = A[2j] + x * (A[2j + 1] - A[2j])
```

After all folds, `P_ell` is the claimed multilinear evaluation at the opening
point `x`.

Gemini commits to `P_1, ..., P_{ell-1}`, derives a Fiat-Shamir challenge `r`,
opens all `P_i` at:

```text
r, -r, r^2
```

and checks that adjacent folds are consistent. In the current plain protocol,
those checks use raw scalar evaluations. That is not zero-knowledge.

## V1 Design Intuition

Gemini's verifier checks are linear. Therefore they can be checked
homomorphically over hiding commitments.

The prover samples a random mask table:

```text
B_0
```

and folds it with the same Gemini folds:

```text
B_{i+1} = Fold_{x_i}(B_i)
```

The prover commits to each pair `(P_i, B_i)` with a Pedersenized KZG commitment:

```text
C_i = P_i(beta) * G + B_i(beta) * H
```

Evaluation values are also hidden:

```text
Y_{i,u} = P_i(u) * G + B_i(u) * H
```

Since the masks satisfy the same fold equations as the real values, Gemini's
linear checks can be performed directly on the `Y` group elements.

## V1 SRS

ZK HyperKZG needs a structured reference string with two independent G1 power
sequences over the same KZG trapdoor:

```text
G_i = beta^i * G
H_i = beta^i * H
```

and ordinary KZG G2 powers:

```text
K
beta * K
```

The prover SRS for capacity `2^k` contains:

```text
G_0, ..., G_{2^k}
H_0, ..., H_{2^k}
K, beta * K
```

The verifier SRS needs enough public data to verify pairings and expose the
evaluation-commitment base to callers:

```text
G_0
H_0
K
beta * K
```

`H` must be an independent generator. The ceremony must destroy the KZG
trapdoor `beta`; no production prover or verifier should see it.

Plain HyperKZG keeps the current canonical filename:

```text
hyperkzg_{k}.srs
```

The ZK variant should use a separate SRS artifact:

```text
hyperkzg_zk_{k}.srs
```

where `k` is the exponent for `2^k` supported evaluations. This mirrors the
plain SRS naming convention while making the ceremony output explicit: the ZK
SRS contains two structured G1 power sequences plus the G2 powers needed for
hidden KZG verification, while the plain SRS contains only the non-hiding
material. The serialized file format must still be versioned with a name
header, mode, and capacity field. A ZK production loader should reject a plain
`hyperkzg_{k}.srs` rather than silently upgrading or accepting it.

The file envelope should include an explicit discriminant:

```text
name:     "HYPERKZG_SRS"
version:  1
kind:     Plain | Zk
k:        supported exponent
capacity: supported evaluation count
payload:  plain or ZK setup material
```

`read_srs_from_dir(dir, k)` loads `hyperkzg_{k}.srs` and requires
`kind = Plain`. `read_zk_srs_from_dir(dir, k)` loads `hyperkzg_zk_{k}.srs` and
requires `kind = Zk`. Both paths should reject mismatched discriminants.

## V1 Commitment

For a polynomial/table `P_0` of length `N = 2^ell`, sample a uniform random mask
table `B_0` of the same length.

Commit:

```text
C_0 = sum_j P_0[j] * G_j + sum_j B_0[j] * H_j
```

The commitment is statistically hiding in `P_0` because `B_0` is uniform and the
`H_j` sequence is independent of the `G_j` sequence.

`commit_zk` must return an opening hint containing the mask material needed by
`open_zk`. The simplest hint is `B_0`. More memory-efficient hints, such as a
private seed expanded deterministically by the prover, are possible but weaken
the information-theoretic proof unless modeled carefully.

## V1 Opening

Input:

```text
P_0
B_0
opening point x = (x_0, ..., x_{ell-1})
commitment C_0
```

The prover computes folded chains:

```text
P_{i+1} = Fold_{x_i}(P_i)
B_{i+1} = Fold_{x_i}(B_i)
```

The hidden final evaluation commitment is:

```text
Y_out = P_ell * G_0 + B_ell * H_0
```

The value `B_ell` is the output blinding returned to the prover-side caller.
This mirrors Dory ZK mode's `y_com` / `y_blinding` split.

The prover commits to intermediate folds:

```text
C_i = P_i(beta) * G + B_i(beta) * H
for i = 1..ell-1
```

Then it appends the opening point, `Y_out`, and the intermediate commitments,
and derives Gemini challenge `r`. `Y_out` must be included before `r` because
it participates in the final Gemini fold equation.

For each `i = 0..ell-1`, define:

```text
Y_i^+  = P_i(r)   * G_0 + B_i(r)   * H_0
Y_i^-  = P_i(-r)  * G_0 + B_i(-r)  * H_0
Y_i^sq = P_i(r^2) * G_0 + B_i(r^2) * H_0
```

These are appended to the transcript. The verifier later checks Gemini fold
consistency directly on these group elements.

## V1 Hidden KZG Opening

For a single hidden polynomial pair `(P, B)` and point `u`, the opening relation
is:

```text
C - Y_u
  = (P(beta) - P(u)) * G + (B(beta) - B(u)) * H
```

The witness is:

```text
W_u =
  ((P(X) - P(u)) / (X - u))(beta) * G
+ ((B(X) - B(u)) / (X - u))(beta) * H
```

The verifier checks:

```text
e(C - Y_u, K) = e(W_u, beta*K - u*K)
```

Equivalently:

```text
e(C - Y_u + u*W_u, K) = e(W_u, beta*K)
```

This is the normal KZG pairing equation with the scalar evaluation replaced by
a hiding evaluation commitment.

## V1 Batching

ZK HyperKZG should preserve the existing batching shape.

After appending all `Y` commitments, derive a batching challenge `q` and combine
all folded polynomials:

```text
P_q = sum_i q^i * P_i
B_q = sum_i q^i * B_i
C_q = sum_i q^i * C_i
```

For each audit point:

```text
u_0 = r
u_1 = -r
u_2 = r^2
```

compute:

```text
Y_q,u = sum_i q^i * Y_i,u
W_u   = hidden KZG witness for (P_q, B_q) at u
```

After appending `W_{u_0}, W_{u_1}, W_{u_2}`, derive a point-batching challenge
`d`. The verifier forms:

```text
L = sum_t d^t * (C_q - Y_q,u_t + u_t * W_u_t)
R = sum_t d^t * W_u_t
```

and checks:

```text
e(L, K) = e(R, beta*K)
```

This gives one constant-size multi-pairing check for all `3 * ell` hidden
opening claims.

## V1 Gemini Group Check

Plain Gemini checks, for each fold:

```text
2*r * P_{i+1}(r^2)
  = r*(1 - x_i)*(P_i(r) + P_i(-r))
  + x_i*(P_i(r) - P_i(-r))
```

ZK Gemini checks the same equation in G1:

```text
2*r * Y_next^sq
  = r*(1 - x_i)*(Y_i^+ + Y_i^-)
  + x_i*(Y_i^+ - Y_i^-)
```

where:

```text
Y_next^sq = Y_{i+1}^sq  if i + 1 < ell
Y_next^sq = Y_out       if i + 1 = ell
```

This works because both the value side and the mask side satisfy the same fold
relation.

## V1 Proof Object

For `N = 2^ell`, a ZK HyperKZG proof contains:

```text
com:       C_1, ..., C_{ell-1}                 // ell - 1 G1
y_pos:     Y_0^+, ..., Y_{ell-1}^+             // ell G1
y_neg:     Y_0^-, ..., Y_{ell-1}^-             // ell G1
y_sq:      Y_0^sq, ..., Y_{ell-1}^sq           // ell G1
y_out:     Y_out                               // 1 G1
w:         W_r, W_-r, W_r2                     // 3 G1
```

Total proof size:

```text
(ell - 1) + 3*ell + 1 + 3 = 4*ell + 3 G1 elements
```

For `ell = 20`:

```text
83 G1 elements
```

With 32-byte compressed BN254 G1 encodings:

```text
83 * 32 = 2656 bytes
```

The verifier returns `y_out` as the hiding commitment to the evaluation. The
prover-side API also returns `B_ell` as the output blinding.

## V1 Verifier Work

Verifier work remains logarithmic in the evaluation table size:

```text
N = 2^ell
verifier group work = O(ell)
pairing work        = O(1)
```

More concretely:

```text
Gemini group checks:
  about 3*ell variable-base scalar multiplications

commitment batching:
  ell scalar multiplications for C_q
  3*ell scalar multiplications for Y_q,u

hidden KZG point batching:
  O(1) scalar multiplications over W_r, W_-r, W_r2

pairing:
  one constant-size multi-pairing check
```

For `ell = 20`, this is roughly:

```text
about 7*ell + O(1) = about 145 G1 scalar-mul equivalents
plus one constant-size multi-pairing
```

This is heavier than plain HyperKZG by constants because scalar Gemini checks
become group checks, but the asymptotic verifier time is unchanged.

## V1 Prover Work

Prover work remains linear in `N`:

```text
fold P chain:             O(N) field work
fold B chain:             O(N) field work
commit folded P/B pairs:  O(N) group MSM work, roughly 2x plain
build batched witnesses:  O(N) field and group work, roughly 2x plain
```

The straightforward implementation materializes both folded chains. A later
implementation can optimize memory by streaming folds and retaining only the
data needed for commitments, evaluation commitments, and batched witnesses.

## V2: Commitment Blinding + KZG Proof Randomization

V1 is conservative but expensive: every hidden commitment is a commitment to
both `P_i` and a full folded mask polynomial `B_i`. V2 keeps the same verifier
equations and proof shape, but changes the prover-side hiding method. This is
the implemented ZK HyperKZG path.

Instead of a full mask polynomial, each Gemini commitment gets one scalar
blinder:

```text
C_i' = C_i + rho_i * H
```

where:

```text
C_i = P_i(beta) * G
H_0 = H
H_1 = beta * H
```

For an opening at point `u`, start from the transparent KZG witness:

```text
W_i,u = ((P_i(X) - P_i(u)) / (X - u))(beta) * G
```

Then randomize the witness by one scalar `tau_i,u`:

```text
W_i,u' = W_i,u + tau_i,u * H
```

The corresponding hidden evaluation commitment is:

```text
Y_i,u' =
  P_i(u) * G
+ (rho_i + u * tau_i,u) * H
- tau_i,u * H_1
```

The KZG equation is preserved:

```text
C_i' - Y_i,u' + u * W_i,u' = beta * W_i,u'
```

because the `H` coefficient cancels and the remaining `H_1` coefficient equals
`tau_i,u`.

V2 therefore hides:

```text
commitments:   with rho_i * H
evaluations:   with rho_i, tau_i,u, and H_1
witnesses:     with tau_i,u * H
```

without committing to or opening a full random mask polynomial.

Information-theoretically, V2 is the BlindFold-style idea applied directly to
HyperKZG's transcript solution space. V1 samples a random preimage in the large
mask-polynomial space and maps it through Gemini/KZG. V2 samples only the
minimal visible degrees of freedom needed by the verifier relations, then solves
the linear constraints so the resulting transcript is a random satisfying
transcript.

## V2 SRS

V2 requires less hiding SRS material than V1:

```text
G_0, ..., G_{2^k}
H_0 = H
H_1 = beta * H
K
beta * K
```

The existing V1 ZK SRS with the full `H_i = beta^i * H` sequence is sufficient
for V2, but not minimal. A later implementation can either:

- keep using `hyperkzg_zk_{k}.srs` and read only `H_0, H_1`, or
- introduce an explicit reduced V2 SRS kind if ceremony size matters.

In both cases the ceremony must destroy `beta` and must not reveal the
discrete-log relation between `G` and `H`.

## V2 Commitment

For the original table:

```text
C_0' = sum_j P_0[j] * G_j + rho_0 * H
```

`commit_zk` returns `C_0'` and an opening hint containing `rho_0`. The hint is a
single scalar, not a full mask table. Homomorphic combination of hints is just:

```text
rho_joint = sum_i gamma_i * rho_i
```

This preserves the existing `AdditivelyHomomorphic` interface for stage-8
joint openings.

## V2 Opening Protocol

Input:

```text
P_0
rho_0
opening point x = (x_0, ..., x_{ell-1})
commitment C_0'
```

The prover computes the ordinary folded chain:

```text
P_{i+1} = Fold_{x_i}(P_i)
```

For `i = 1..ell-1`, sample fresh commitment blinders `rho_i` and send:

```text
C_i' = P_i(beta) * G + rho_i * H
```

Also sample an output blinder `lambda` and send:

```text
Y_out = P_ell * G + lambda * H
```

Append the opening point, `Y_out`, and `C_1'..C_{ell-1}'`, then derive Gemini
challenge `r`. Define audit points:

```text
u_+  = r
u_-  = -r
u_sq = r^2
```

After `r` is known, sample free variables:

```text
tau_i,sq  for i = 0..ell-1
```

Then solve for:

```text
tau_i,+ and tau_i,-
```

so that the `H` and `H_1` coefficients satisfy the Gemini fold equations.

For any point `u`, define:

```text
a_i,u = rho_i + u * tau_i,u
b_i,u = -tau_i,u

Y_i,u' = P_i(u) * G + a_i,u * H + b_i,u * H_1
```

For each fold level `i`, let:

```text
next_a =
  a_{i+1,sq}                 if i + 1 < ell
  lambda                     if i + 1 = ell

next_b =
  b_{i+1,sq}                 if i + 1 < ell
  0                          if i + 1 = ell
```

The prover chooses `tau_i,+, tau_i,-` to satisfy:

```text
2*r * next_a =
  r*(1 - x_i)*(a_i,+ + a_i,-)
+ x_i*(a_i,+ - a_i,-)

2*r * next_b =
  r*(1 - x_i)*(b_i,+ + b_i,-)
+ x_i*(b_i,+ - b_i,-)
```

This is a `2 x 2` linear system in `tau_i,+, tau_i,-` once `rho_i` and
the next hidden coefficients are fixed. For the final fold those next
coefficients are determined by `lambda`; otherwise they are determined by
`rho_{i+1}` and the sampled free variable `tau_{i+1,sq}`.

The system determinant is:

```text
-2*r * (r^2*(1 - x_i)^2 - x_i^2)
```

So it is invertible except for `r = 0` or `r*(1 - x_i) = +/- x_i`. These are
Fiat-Shamir degeneracies with negligible probability. A prover-side API that
cannot return errors may panic on this event; a fallible implementation should
return a degenerate-challenge error.

After solving, the prover appends:

```text
Y_i,+  = Y_i,r'
Y_i,-  = Y_i,-r'
Y_i,sq = Y_i,r^2'
```

for all `i`, then derives batching challenge `q`.

## V2 Batched KZG Opening

As in plain HyperKZG, combine folded polynomials:

```text
P_q = sum_i q^i * P_i
C_q' = sum_i q^i * C_i'
```

For each audit point `u`, define:

```text
tau_q,u = sum_i q^i * tau_i,u
Y_q,u'  = sum_i q^i * Y_i,u'
```

Compute the transparent batched witness:

```text
W_q,u = ((P_q(X) - P_q(u)) / (X - u))(beta) * G
```

and send the randomized witness:

```text
W_q,u' = W_q,u + tau_q,u * H
```

The verifier uses exactly the same hidden KZG batch equation as V1:

```text
L = sum_t d^t * (C_q' - Y_q,u_t' + u_t * W_q,u_t')
R = sum_t d^t * W_q,u_t'

e(L, K) = e(R, beta*K)
```

The proof object is unchanged from V1:

```text
com:       C_1', ..., C_{ell-1}'
y_pos:     Y_{0,+}, ..., Y_{ell-1,+}
y_neg:     Y_{0,-}, ..., Y_{ell-1,-}
y_sq:      Y_{0,sq}, ..., Y_{ell-1,sq}
y_out:     Y_out
w:         W_q,r', W_q,-r', W_q,r2'
```

The verifier does not need to know whether the prover used V1 full-mask hiding
or V2 proof randomization. Both constructions produce group elements satisfying
the same Gemini and hidden-KZG equations. If both prover modes remain
implemented, the proof payload can stay the same; the opening hint should carry
the mode-specific private data.

## V2 Prover Work

V2 removes the full second mask-polynomial path:

```text
fold P chain:                   O(N) field work
commit folded P_i:              same MSMs as transparent
transparent batched witnesses:  same as transparent
commitment blinds:              O(ell) scalar muls by H
evaluation randomizers:         O(ell) scalar arithmetic + O(ell) G1 ops
witness randomizers:            O(1) scalar muls by H
```

The expected prover cost is therefore close to transparent HyperKZG plus
logarithmic overhead, instead of V1's roughly `2x` large-MSM work.

Verifier work is essentially the same as V1 because the verifier still receives
group-valued hidden evaluations and checks Gemini in G1. V2 primarily improves
prover time and prover memory.

The verifier should batch the group-valued checks:

```text
after absorbing all Y_i,u:
  derive alpha and check one random-linear Gemini residual MSM
  derive q for hidden KZG batching

after absorbing all W_u:
  derive d and build the hidden-KZG pairing L side with one direct MSM
```

The important transcript condition is that all hidden evaluation commitments
`Y_i(r), Y_i(-r), Y_i(r^2)` and `Y_out` are bound before deriving `alpha`.
This gives the batched Gemini check the usual Schwartz-Zippel soundness loss
without letting the prover choose `Y` after seeing the batching randomness.

## V2 Hiding Sketch

For a fixed Fiat-Shamir transcript prefix and challenge `r`, the V2 prover
samples:

```text
rho_0, ..., rho_{ell-1}
lambda
tau_0,sq, ..., tau_{ell-1,sq}
```

and derives `tau_i,+, tau_i,-` by solving the Gemini coefficient equations.
Except for the negligible degenerate `r` values above, this is an affine
bijection between the sampled variables and the space of `H/H_1` coefficients
that satisfy all Gemini equations with `Y_out` constrained to have output blind
`lambda`.

Thus the visible proof elements are:

```text
transparent P-dependent G components
+ a uniformly random point in the linear solution space generated by H and H_1
```

Adding the fixed `P`-dependent component only shifts that solution-space
distribution. Because the transparent `P`-dependent tuple itself satisfies the
same homogeneous Gemini and KZG group relations, that shift is by an element of
the same solution space. The distribution is therefore identical for every
polynomial, modulo the public relations the verifier checks. The verifier learns
the public Gemini and KZG relations, but not the scalar evaluations or
commitment contents.

Commitments are statistically hiding because each `C_i'` is shifted by an
independent uniform `rho_i * H`. `Y_out` is statistically hiding because it is
shifted by uniform `lambda * H`.

The KZG witnesses do not need independent full-polynomial masks. Once `Y_q,u'`
and `C_q'` are fixed, the randomized witness:

```text
W_q,u' = W_q,u + tau_q,u * H
```

is the unique honest witness satisfying the hidden KZG equation. Its randomness
is exactly the same `tau` randomness already used in the corresponding
evaluation commitments, so it reveals only the public KZG opening relation.

The Fiat-Shamir hiding argument follows the same prefix-by-prefix structure as
V1: every challenge is derived from statistically hiding prior messages, so the
challenge distribution is independent of the committed polynomial.

## V2 Soundness Sketch

The verifier checks the same equations as V1:

1. Gemini consistency over group-valued hidden evaluations.
2. One random-linear batched hidden KZG pairing check.

If a prover forges a hidden KZG opening for some `C_i'`, `Y_i,u'`, and `W_i,u'`,
then it has produced group elements satisfying:

```text
C_i' - Y_i,u' = (beta - u) * W_i,u'
```

without knowing `beta`. As in KZG, this is binding except with negligible
probability under the structured reference string assumption. Random-linear
batching in `q` and `d` preserves the usual Schwartz-Zippel soundness loss.

If the hidden KZG openings are sound, then each `Y_i,u'` is bound to the opened
hidden commitment at the audit point. The Gemini group equations then imply
that the folded chain is consistent with the returned `Y_out`, except with the
same Gemini audit soundness error as plain HyperKZG.

The soundness model excludes:

```text
knowledge of beta
knowledge of log_G(H)
ability to program Fiat-Shamir challenges
degenerate r values where the V2 coefficient solve is singular
```

These are the same setup and Fiat-Shamir assumptions already required by the
KZG-based construction.

## Trait Integration

The natural integration point is `jolt_openings::ZkOpeningScheme`:

```text
type HidingCommitment = P::G1
type Blind = P::ScalarField

commit_zk:
  samples rho_0 and returns a hint carrying rho_0

open_zk:
  computes ZK Gemini proof
  returns (proof, Y_out, lambda)

verify_zk:
  verifies group Gemini checks and hidden KZG batch check
  returns Y_out

bind_zk_opening_inputs:
  binds opening point and Y_out, not the raw scalar evaluation
```

As with Dory's `y_com`, the HyperKZG proof should carry `Y_out`. `verify_zk`
checks that commitment as part of the proof, then returns it through the generic
`ZkOpeningScheme` API for the outer transcript or wrapper protocol.

For V2, `lambda` is the output blinding for:

```text
Y_out = P_ell * G + lambda * H
```

This preserves the generic `Blind = P::ScalarField` API while avoiding a full
mask-table opening hint.

The PCS should keep `HidingCommitment = P::G1`. BlindFold, wrapper SNARKs, and
other consumers should remain generic over the relevant `jolt-crypto` group
types and resolve `P::G1` at concrete instantiation time. HyperKZG should not add
an adapter commitment type solely to satisfy the current Dory-shaped
`VC::Output` coupling.

The existing non-ZK `CommitmentScheme` implementation should remain available
for transparent mode unless the crate later decides to make ZK mode the only
production path.

## Implementation Architecture

Use one `HyperKZGScheme<P>` implementation with transparent and ZK behavior
selected by a small internal mode abstraction. This mirrors the organization of
`dory-pcs`, where the same protocol code is parameterized by a `Mode`, and the
Jolt wrapper maps transparent trait methods to `Transparent` and ZK trait
methods to `Zk`.

```text
commit      -> commit_with_mode::<Transparent>
open        -> open_with_mode::<Transparent>
verify      -> verify transparent proof payload

commit_zk   -> commit_with_mode::<Zk>
open_zk     -> open_with_mode::<Zk>
verify_zk   -> verify ZK proof payload and return Y_out
```

The internal mode should be lightweight:

```text
trait Mode {
  const HIDING: bool
  sample_opening_hint(len) -> none or rho_0
  require_opening_hint(hint) -> mode-specific private opening data
}

Transparent:
  HIDING = false
  no blind

Zk:
  HIDING = true
  samples uniform rho_0
```

The public setup and proof types stay unified, but carry optional ZK material:

```text
HyperKZGProverSetup:
  g1_powers
  g2_powers
  hiding_g1_powers: optional beta^i * H

HyperKZGVerifierSetup:
  g1
  g2
  beta_g2
  hiding_g1: optional H

HyperKZGOpeningHint:
  none | ZkBlind(rho_0)

HyperKZGProof:
  com
  w
  payload:
    Clear { v }
    Zk { y, y_out }
```

The proof payload should always use an explicit `Clear | Zk` discriminant in the
serialized type. This is less footgun-prone than cfg-gating the wire shape: clear
verification can reject a `Zk` payload, ZK verification can reject a `Clear`
payload, and non-ZK builds do not silently reinterpret bytes under a different
schema. The `zk` feature gates construction and verification machinery, not the
basic ability to deserialize and reject the wrong payload mode.

`commit` returns a transparent hint with no private opening data. `commit_zk`
requires a ZK-capable SRS and returns mode-specific hiding data. Stage-8
batching must combine hints homomorphically.

```text
P_joint   = sum_i gamma_i * P_i
rho_joint = sum_i gamma_i * rho_i
C_joint'  = sum_i gamma_i * C_i'
```

Thus `combine_hints` returns no private data if all inputs are transparent,
returns the linear combination of scalar blinds if all inputs are ZK, and
rejects mixed transparent/ZK hints as a caller precondition violation.

Error policy:

```text
prover-side APIs:
  may panic on caller-precondition failures that cannot be represented by the
  trait, e.g. commit_zk with a plain SRS, open_zk without the right hint kind,
  mixed transparent/ZK hints, or the negligible V2 singular linear solve if the
  trait remains infallible

verifier-side APIs:
  must not panic on proof, transcript, setup, or payload errors; return
  HyperKZGError/OpeningsError for wrong discriminants, missing ZK SRS material,
  malformed lengths, failed Gemini checks, or failed pairing checks
```

The `jolt-hyperkzg` crate should expose a local feature flag:

```toml
[features]
default = []
zk = []
```

The `zk` feature gates only the ZK surface area: the `Zk` mode, the
`ZkOpeningScheme` implementation, ZK SRS import/export helpers, and ZK
construction helpers. It should not gate the Gemini folding logic, clear
`CommitmentScheme` implementation, shared KZG helpers, or the proof payload
discriminant. The workspace-level `zk` feature should enable
`jolt-hyperkzg/zk` when HyperKZG is wired into a ZK prover.

## Transcript Order

A candidate proof-internal transcript order:

```text
proof verification:
  append opening point
  append Y_out
  append intermediate commitments C_1..C_{ell-1}
  derive r
  append Y_i^+, Y_i^-, Y_i^sq for all i
  derive alpha for batched Gemini group check
  derive q
  append W_r, W_-r, W_r2
  derive d
  verify batched Gemini group equation
  verify batched hidden KZG pairing equation
```

The outer `ZkOpeningScheme::bind_zk_opening_inputs` call should still bind the
opening point and `Y_out` into the Jolt transcript after `verify_zk` returns
`Y_out`, matching the current stage-8 flow. That outer binding is separate from
the proof-internal absorption above.

Use canonical labels that name the protocol part and separate clear scalar
evaluations from hidden evaluation commitments:

```text
clear outer binding:
  hyperkzg_opening_point
  hyperkzg_opening_eval

ZK proof-internal transcript:
  hyperkzg_zk_point
  hyperkzg_zk_y_out
  hyperkzg_zk_fold_com
  hyperkzg_zk_gemini
  hyperkzg_zk_eval_r
  hyperkzg_zk_eval_neg_r
  hyperkzg_zk_eval_r2
  hyperkzg_zk_gemini_batch
  hyperkzg_zk_eval_batch
  hyperkzg_zk_wit_r
  hyperkzg_zk_wit_nr
  hyperkzg_zk_wit_r2
  hyperkzg_zk_wit_batch

ZK outer binding:
  hyperkzg_zk_point
  hyperkzg_zk_eval_com
```

Vector labels should include a count, following the existing
`LabelWithCount` pattern. Labels must stay within the transcript library's
24-byte label limit. The verifier should use the same labels and reject
payload-mode mismatches before absorbing mode-specific proof elements.

## Security Target

Binding remains computational and KZG-style:

```text
no adversary knowing only the public SRS can open one commitment to two
different polynomial/evaluation statements, except with negligible probability
under the KZG binding assumption and discrete-log binding of the G/H bases
```

Hiding should be statistical for transcript-visible values:

```text
Given the public SRS and public hidden output commitment Y_out, the distribution
of C_i, Y_i^+, Y_i^-, Y_i^sq, and W_u should be independent of P_0, except for
the public linear relations checked by the verifier.
```

For V1, the intended reason is that all visible scalar linear functionals of
`P_0` are masked by corresponding linear functionals of a uniform random `B_0`.
For V2, the intended reason is that commitments are independently shifted by
`rho_i * H`, while evaluation and witness commitments are sampled from the
linear solution space of the Gemini and KZG equations using `tau` randomizers.

## V1 Proof Sketch

This section sketches the security argument for the construction. It is not a
replacement for a final proof, but it identifies the right mathematical object:
the image of one linear map from the original table to all transcript-visible
linear functionals.

### Hiding, Interactive View

First consider the interactive protocol where `r`, `q`, and `d` are sampled by
the verifier after the prover has sent the relevant prior messages.

Fix the verifier challenges and define a linear map:

```text
Phi : F^N -> F^M
```

where `Phi(T)` contains every scalar functional of table `T` that appears in the
proof transcript:

```text
fold commitments:
  T_i(beta)

hidden evaluation commitments:
  T_i(r), T_i(-r), T_i(r^2)

batched KZG witnesses:
  ((T_q(X) - T_q(u)) / (X - u))(beta)
  for u in {r, -r, r^2}
```

Here `T_i` denotes the Gemini-folded chain derived from `T`, and `T_q` denotes
the Fiat-Shamir linear combination of the folded polynomials. All entries of
`Phi(T)` are linear in `T`.

The visible proof elements have scalar representation:

```text
Phi(P_0) * G + Phi(B_0) * H
```

componentwise in G1. Since `B_0` is uniform in `F^N`, `Phi(B_0)` is uniform over
`im(Phi)`. Since `Phi(P_0)` is itself in `im(Phi)`, adding it only shifts a
uniform distribution over the same subspace. Therefore:

```text
Phi(P_0) * G + Phi(B_0) * H
```

has the same distribution for every `P_0`, modulo the public linear relations
defining `im(Phi)`.

Those public relations are exactly the Gemini fold equations and KZG opening
relations checked by the verifier. The verifier learns that the committed
messages are mutually consistent, but not their scalar values.

### Hiding, Fiat-Shamir View

In Fiat-Shamir, challenges are hashes of prior transcript messages. The fixed
challenge argument above should be applied prefix-by-prefix.

At each challenge point, the prior group elements are already statistically
hiding because they are masked by the same linear-image argument. Therefore the
challenge distribution is independent of `P_0`. Conditional on any concrete
prefix and resulting challenge, the next block of messages is again an affine
shift of a linear image of the remaining random mask degrees of freedom.

The final formal proof should write this as a simulator over transcript
prefixes, or equivalently over the image of the full Fiat-Shamir-adapted map.
The important design condition is that masks are generated by a full random
table `B_0`, not by independent ad hoc per-message masks.

### Witness Hiding

The KZG witnesses do not require separate randomizers. For every point `u`, the
hidden witness is:

```text
W_u =
  ((P_q(X) - P_q(u)) / (X - u))(beta) * G
+ ((B_q(X) - B_q(u)) / (X - u))(beta) * H
```

The quotient map is linear. Thus the witness mask is exactly the same linear
functional applied to `B_0`. The witness is correlated with `C_q` and `Y_q,u`,
but only through the public KZG equation:

```text
C_q - Y_q,u = (beta - u) * W_u
```

That relation is necessary for verifiability and does not expose an unmasked
linear functional of `P_0`.

### Soundness Sketch

Soundness has the same two layers as plain HyperKZG:

1. Hidden KZG openings bind each `Y_i,u` to the corresponding committed folded
   polynomial commitment `C_i`.
2. Gemini group equations bind the committed folded chain to the final hidden
   evaluation commitment `Y_out`.

The hidden KZG opening check proves, under the structured KZG binding
assumption, that each accepted opening commitment `Y_i,u` is the actual hidden
evaluation commitment of `C_i` at `u`. Random-linear batching with challenges
`q` and `d` preserves this except with Schwartz-Zippel probability, as in plain
HyperKZG.

The Gemini checks are linear group equations. If the folded chain is
inconsistent, then the residual is a nonzero low-degree expression in the
Gemini challenge `r`; it can vanish only with Schwartz-Zippel probability. This
is the same information-theoretic audit used by plain Gemini, with scalar
residuals lifted into G1 commitments.

Binding additionally requires that the adversary cannot exploit a known
relation between `G` and `H` or the trapdoor `beta` to equivocate between
different `(P, B)` pairs. This is why the SRS ceremony must destroy `beta` and
must not reveal a discrete-log relation between the two G1 bases.

## V1 Required Proof Obligations

Current assessment of the proof obligations:

1. **Mask rank.**
   One full random mask table `B_0` appears sufficient. For fixed challenges,
   all transcript-visible value data is a linear map of the original table:

   ```text
   Phi(P_0) = (C_i value parts, Y_i,u value parts, W_u value parts)
   ```

   The mask data is the same linear map applied to a uniform random table:

   ```text
   Phi(B_0)
   ```

   The visible group elements have the form:

   ```text
   Phi(P_0) * G + Phi(B_0) * H
   ```

   Since `B_0` is uniform, `Phi(B_0)` is uniform over `im(Phi)`. Therefore
   adding `Phi(P_0)` shifts a uniform distribution over the same image and does
   not change it. Any rank deficiency corresponds to public linear relations in
   `im(Phi)`, which are exactly the Gemini/KZG consistency relations the
   verifier checks.

2. **Witness hiding.**
   No extra independent quotient-witness mask appears necessary. For fixed
   point `u`, the quotient witness map

   ```text
   P_0 -> ((P_q(X) - P_q(u)) / (X - u))(beta)
   ```

   is also linear in `P_0`, and its mask is the same linear map applied to
   `B_0`. The witness is correlated with `C_q` and `Y_q,u`, but that
   correlation is the public KZG opening relation. It does not add an
   unmasked linear functional of `P_0`.

3. **Batching soundness.**
   Random-linear batching should have the same soundness shape as plain
   HyperKZG. Invalid hidden opening claims define nonzero group/pairing
   residuals. Combining them with Fiat-Shamir powers in `q` and `d` can cancel
   only with Schwartz-Zippel probability, up to the usual KZG binding
   assumptions.

4. **SRS independence.**
   This is mandatory. The ceremony must produce two structured G1 sequences

   ```text
   beta^i * G
   beta^i * H
   ```

   with no known discrete-log relation between `G` and `H`, and no surviving
   knowledge of `beta`. Hiding only needs random masks, but binding depends on
   the adversary not knowing the `G/H` relation or `beta`.

5. **Transcript binding.**
   `Y_out` must be absorbed inside `verify_zk` before deriving Gemini challenge
   `r`, because it participates in the final Gemini group equation. Separately,
   the `ZkOpeningScheme` integration may call `bind_zk_opening_inputs` after
   `verify_zk` returns `Y_out`; that later call binds the opening statement into
   the outer Jolt transcript and does not replace the proof-internal absorption.

Remaining proof details:

- Exclude standard KZG degeneracies such as `u = beta`; this happens with
  negligible probability because `beta` is hidden and `u` is Fiat-Shamir
  derived.
- Reject or domain-separate degenerate Gemini challenges already rejected in
  plain HyperKZG, e.g. `r = 0`.
- Write the formal simulator over the image of `Phi` rather than over
  independent per-message masks.

## Statistical Integration Tests

Add crate-local statistical tests under `crates/jolt-hyperkzg/tests/`, gated by
the `zk` feature. These should follow the same shape as `dory-pcs`'
`zk_statistical.rs` example and the `jolt-verifier` ZK statistical-independence
harness:

```text
NUM_BUCKETS = 16
chi-squared uniformity check over projected components
split-half two-sample check over even/odd samples
witness-family two-sample checks across different polynomial distributions
stable public statement/proof shape checks
```

The tests should collect transcript-visible ZK HyperKZG components:

```text
C_0
C_1..C_{ell-1}
Y_i(r), Y_i(-r), Y_i(r^2)
Y_out
W_r, W_-r, W_r2
returned output blinding `lambda`
```

For large vectors, sample stable positions such as first, middle, and last. Each
component should be projected by canonical serialization or `AppendToTranscript`
into a field challenge, then bucketed by low bits. The tests should verify:

- repeated proofs for the same polynomial, point, and claimed public statement
  have indistinguishable projected distributions;
- repeated proofs for different polynomial families, e.g. zero, low-Hamming,
  structured, and uniform random tables, have indistinguishable projected
  distributions for the transcript-visible proof components;
- two `commit_zk` calls for the same polynomial produce different commitments;
- homomorphically combined ZK hints preserve the same statistical behavior for
  the joint opening proof.

Because these tests are probabilistic and potentially expensive, keep a small
non-ignored smoke test for randomized ZK roundtrips, and mark the full
statistical test `#[ignore]`. The full test should require a release build unless
an explicit environment override is set, and should read its sample count from an
environment variable such as `JOLT_HYPERKZG_ZK_STAT_SAMPLES`.

## Current Design Decisions

- Use a separate ZK SRS file, likely `hyperkzg_zk_{k}.srs`.
  `hyperkzg_{k}.srs` remains the plain HyperKZG SRS naming convention.
- Store `Y_out` inside the proof, mirroring Dory's `y_com` pattern. `verify_zk`
  absorbs and checks it, then returns it through `ZkOpeningScheme` for the outer
  transcript or wrapper protocol.
- Have `commit_zk` store only the scalar `rho_0` opening hint. ZK HyperKZG uses
  commitment blinding plus proof randomization; it does not carry a full random
  mask table.
- Do not optimize for streaming folds in the first spec or implementation. The
  straightforward materialized-fold construction is the reference path. V2 still
  benefits from ordinary transparent HyperKZG folding and does not require a
  second folded mask chain.
- Treat ZK HyperKZG as HyperKZG's ZK mode, implemented inside the same
  `HyperKZGScheme` codebase. It remains distinct from the existing Dory PCS
  internals and interacts with BlindFold or a wrapper SNARK through the generic
  PCS interface for the chosen proof system.
- Keep `HidingCommitment = P::G1`; generic consumers resolve the concrete group
  type when HyperKZG is paired with BlindFold or a wrapper.
- Use explicit discriminants for SRS files and proof payloads. Separate file
  paths still distinguish plain and ZK SRS artifacts.
- Allow prover-side panics for invalid caller preconditions in non-`Result`
  trait methods, but make verifier-side handling return errors rather than
  panicking.
- Use canonical transcript labels with separate `hyperkzg_*` and
  `hyperkzg_zk_*` namespaces.

## V1 Historical Reference Plan

This was the conservative full-mask implementation plan. It is retained only as
reference material; the implementation target is V2.

1. Add versioned SRS file envelopes with explicit `Plain | Zk` discriminants.
   The ZK payload contains `G_i`, `H_i`, `K`, and `beta*K`, serialized as
   separate `hyperkzg_zk_{k}.srs` artifacts.
2. Add hidden univariate KZG helpers and unit tests for one polynomial, one
   point.
3. Add hidden KZG batching across many polynomials and the three Gemini points.
4. Add ZK Gemini proof types with group evaluation commitments.
5. Implement `ZkOpeningScheme` for the existing `HyperKZGScheme` using the
   shared mode-parametrized internals.
6. Add roundtrip tests:
   - valid hidden opening verifies
   - two `commit_zk` calls for the same polynomial produce different commitments
   - tampering any `Y_i,u`, `C_i`, `W_u`, or `Y_out` rejects
   - wrong hidden output commitment rejects
   - batched openings reject if any folded commitment is inconsistent
   - verifier rejects wrong proof/SRS discriminants without panicking
   - prover panics on invalid ZK caller preconditions that the trait cannot
     return as errors
7. Add statistical integration tests for transcript-visible ZK proof components,
   including uniformity, split-half, witness-family independence, and combined
   hint behavior.
8. Add serialization tests for ZK SRS and proof payloads.
9. Add criterion benches comparing plain and ZK HyperKZG at `ell = 8, 10, 12,
   14, 20`.

## V2 Implementation Plan

Implement V2 as the preferred prover path:

1. Store `HyperKZGOpeningHint` as `None | ZkBlind(F)`.
2. Change `commit_zk` for V2 to compute the transparent KZG commitment and add
   `rho_0 * H`.
3. Keep the existing proof payload shape:

   ```text
   Zk { y: [Vec<G1>; 3], y_out: G1 }
   ```

   The verifier equations are unchanged.
4. Add V2 prover helpers:
   - sample `rho_1..rho_{ell-1}`, `lambda`, and free `tau_i,sq`;
   - solve the per-level `2 x 2` systems for `tau_i,+`, `tau_i,-`;
   - build `Y_i,u'` using `H_0` and `H_1`;
   - build randomized witnesses `W_q,u' = W_q,u + tau_q,u * H`.
5. Reuse the transparent folding, intermediate commitment, and batched witness
   code wherever possible. V2 should not materialize a mask polynomial.
   The prover needs an internal helper that computes transparent batched
   evaluations and witnesses without appending clear scalar evaluations to the
   transcript; `q` must be derived from the hidden `Y` commitments.
6. Optimize verifier batching:
   - derive `alpha` after absorbing all hidden `Y` commitments and before `q`;
   - verify one random-linear Gemini residual MSM instead of `ell` separate
     group equations;
   - construct the hidden-KZG pairing `L` side with one direct MSM over
     `C_i`, `Y_i,u`, and `W_u`.
7. Add V2-specific tests:
   - roundtrip opening verifies and returns `Y_out`;
   - returned blind `lambda` satisfies `Y_out = eval * G + lambda * H`;
   - two commitments to the same polynomial differ;
   - homomorphic combination of `rho` hints verifies;
   - tampering `rho`-derived commitments, `Y_i,u`, `Y_out`, or randomized
     witnesses rejects;
   - artificially forced singular coefficient systems are detected in prover
     code.
8. Rerun the statistical harness against V2. The transcript-visible components
   should remain independent across polynomial families.
9. Bench V2 against transparent HyperKZG. Expected V2 prover overhead is
   logarithmic plus small G1 operations, while verifier overhead remains close
   to the group-check ZK verifier described above.

## Non-Goals

This spec does not require:

```text
wrapping plain HyperKZG in a SNARK
reusing the existing Dory/BlindFold PCS internals for Gemini
revealing any scalar evaluations
changing Dory's ZK path
making setup_from_secret a production API
```

The construction is deliberately PCS-native: hide the commitments, evaluations,
and witnesses at the KZG/Gemini layer, then use homomorphic checks for Gemini's
linear verifier. Wrapper SNARKs or BlindFold-style layers can consume this PCS
through `ZkOpeningScheme` and impose their own constraints over the returned
hiding commitment and blinding data.
