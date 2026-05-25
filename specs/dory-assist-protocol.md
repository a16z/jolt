# Spec: Dory Assist Protocol

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

Dory assist is the recursion-paper path for replacing expensive ordinary Dory
stage-8 verifier work with an auxiliary proof. It is the Dory-specific
implementation of the generic PCS proof-assist boundary exposed by
`jolt-verifier`.

From the `jolt-verifier` point of view, the Jolt proof carries a generic
optional PCS-assist payload, e.g. `Option<PcsAssist::Proof>` where
`PcsAssist: PcsProofAssist<PCS>`. The verifier config decides whether such a
payload is required and which assist implementation is selected. Dory-specific
staging does not live in `jolt-verifier`; it lives in the
`jolt-dory-assist-verifier` crate that implements the generic PCS-assist
interface for the Dory PCS.

This spec owns the Dory-assist protocol semantics in `jolt-claims`, the
`jolt-dory-assist-verifier` crate, and the Hyrax commitment layer used by that
proof. Composition with field inline, wrapper proving, ZK, and proof
configuration is specified in
[selected-verifier-integration.md](selected-verifier-integration.md). The
wrapper R1CS compiler is specified in [wrapper-protocol.md](wrapper-protocol.md).

References:

- Recursion paper repo: <https://github.com/markosg04/recursion-paper>
- Quang's `quang/recursion-temp` branch is the concrete protocol reference for
  Dory-assist staging, operation families, wiring, prefix packing, and Hyrax
  opening flow.

The modular implementation is new code. The reference branch informs staging
and formulas; it is not copied as an architectural dependency.

## Curve Choice

Dory assist uses Grumpkin for the auxiliary Hyrax commitment layer in the BN254
instantiation. This is a semantic requirement from the recursion-paper design,
not just an implementation preference:

```text
BN254 Dory verifier:
  group coordinates and pairing-target arithmetic live over BN254 Fq

ordinary Jolt proof:
  committed polynomial field is BN254 Fr

Grumpkin:
  base field is BN254 Fr
  scalar field is BN254 Fq
  no pairings
```

The Dory verifier computation we offload needs `Fq`-native arithmetic. Grumpkin
forms a 2-cycle with BN254, so a Hyrax/Pedersen assist proof over Grumpkin has
the right scalar field for the Dory verifier's `Fq` arithmetic while keeping the
cryptographic opening check to ordinary Grumpkin MSMs rather than pairings.

## Scope

V1 scope:

```text
Dory-assist protocol facts in jolt-claims
`jolt-dory-assist-verifier` crate implementing PcsProofAssist for the Dory PCS
generic PCS-assist payload consumed by jolt-verifier
Hyrax dense-witness opening over Grumpkin-backed Pedersen row commitments
Miller-loop work proven inside the assist proof
final exponentiation checked natively by the Dory-assist verifier
wrapper-compatible R1CS hooks for sumcheck, claims, packing, and Hyrax
```

Out of scope:

```text
ordinary Dory verifier inside wrapper R1CS
pairing final exponentiation inside the Dory-assist SNARK
one-shot port of all recursion branch implementation internals
making Dory assist depend on field-inline-specific machinery
Dory-specific stage modules inside jolt-verifier
```

## Boundary Model

The ownership split is:

```text
jolt-verifier:
  generic configured verifier flow
  PcsProofAssist trait or equivalent verifier-facing abstraction
  Option<PcsAssist::Proof> proof payload slot
  proof/config shape validation
  reduced opening statement construction
  dispatch to the selected assist implementation

jolt-claims:
  Dory-assist protocol semantics
  relation IDs, public IDs, opening IDs, dimensions, and claim formulas
  operation-family, wiring, and packing constraints

jolt-dory-assist-verifier:
  Dory-specific staged verifier organization
  native verifier API mirroring jolt-verifier for Dory-assist semantics
  wrapper-facing R1CS hook for the selected Dory-assist verifier path
  DoryAssist implementation type and proof payload for the Dory PCS
  transcript ordering for the Dory-assist stages
  Hyrax dense-witness opening verification
  native final-exponentiation/public pairing equality check

jolt-hyrax:
  reusable Hyrax commitment, opening, verifier, and R1CS helpers

future jolt-trace:
  runtime instrumentation for the Dory verifier program
  emission of operation-family events and wiring data used by witness gen
  no ownership of Dory-assist protocol semantics
```

`jolt-verifier` should not match on Dory-assist stages directly. A Dory-enabled
build selects the Dory PCS and a Dory-assist proof type; the generic opening
phase then validates `proof.pcs_assist` against the configured shape and calls
the selected `PcsProofAssist` implementation.

## Protocol Target

The Dory assist proof is a semantic relation for the Dory verifier's expensive
cryptographic component:

```text
public inputs:
  Dory proof artifact / joint PCS opening proof
  Dory verifier setup inputs
  Jolt evaluation claims and commitments from stage 8
  transcript-derived scalars

private witness:
  operation-family polynomial assignments:
    GT exponentiation and multiplication witnesses
    G1/G2 scalar multiplication and addition witnesses
    Miller-loop witnesses
    operation outputs and wiring values

commitment:
  Grumpkin/Pedersen-backed Hyrax commitment to the prefix-packed dense witness
```

The proof establishes:

```text
local correctness:
  each operation family satisfies its algebraic relation

wiring correctness:
  outputs consumed by later operations match producer outputs

public-input consistency:
  public Dory proof inputs and Jolt evaluation claims are the values used in
  the operation-family witnesses

packing correctness:
  many native-size operation-family polynomials are prefix-packed into one
  dense polynomial

opening correctness:
  Hyrax opens that dense polynomial at the verifier's required point
```

`jolt-claims::protocols::dory_assist` defines this relation. It does not define
how the Dory verifier program is instrumented, how runtime events are recorded,
or how concrete witnesses are extracted. That belongs to a later generic
`jolt-trace` path plus Dory-assist witness generation.

## Statement

Dory assist proves the following statement:

```text
Given the public Dory opening-verifier inputs, there exists a correctly wired
assignment to the Dory-assist operation-family witness polynomials whose final
public result is the pre-final-exponentiation GT value consumed by the native
pairing check, and whose prefix-packed dense witness is committed/opened
consistently by Hyrax.
```

Public inputs:

```text
joint Dory opening proof artifact
Dory verifier setup inputs / verifier setup digest
Jolt reduced opening statement:
  commitments
  opening points
  claimed evaluations
  transcript-derived scalars
Dory-assist transcript challenges
Grumpkin/Pedersen-backed Hyrax dense-witness commitment
Hyrax opening claim
Miller-loop output GT value and native final-check inputs
```

Private witness:

```text
operation-family witness polynomials:
  GT exponentiation rows
  GT multiplication rows
  G1 scalar-mul rows
  G1 addition rows
  G2 scalar-mul rows
  G2 addition rows
  Miller-loop rows
  intermediate values
  wiring values
prefix-packed dense witness evaluations
```

The proof establishes:

```text
local correctness:
  every operation-family row satisfies its algebraic relation

wiring correctness:
  every value consumed by a later operation equals the value produced earlier

public-input consistency:
  witness inputs equal the public Dory proof fields, verifier setup data, Jolt
  opening claims, commitments, and transcript challenges

Dory verifier consistency:
  the witness assignment computes the same pre-final-exponentiation GT value
  that the ordinary Dory verifier would compute for the public opening proof

packing correctness:
  all operation-family witness polynomials are packed into one dense
  multilinear polynomial according to the prefix-packing layout

opening correctness:
  the Hyrax commitment opens that dense packed witness at the claimed verifier
  point to the claimed packed evaluation

final public check:
  the Dory-assist verifier uses the public GT output, performs native final
  exponentiation, and checks the public pairing equality
```

The SNARK portion proves the expensive Dory verifier computation through the
Miller-loop output. Final exponentiation and pairing equality remain native
Dory-assist verifier work in v1.

Wrapper caveat: "native verifier work" means outside the Dory-assist auxiliary
PIOP, not outside the configured verifier. If a wrapper proves the Dory-assisted
configured verifier, it must still account for the final exponentiation and
pairing-equality acceptance condition. The primary self-contained wrapper path
should do this with a `jolt_dory_assist_verifier::r1cs` final-check hook. An
external native side check is possible only as an explicit deployment choice and
must be bound to the wrapper statement; it is not the default self-contained
wrapper semantics.

## Techniques

This protocol should use the recursion paper's terminology for the auxiliary
proof. The Dory-assist SNARK follows the standard
commit--PIOP--evaluation-proof pattern and uses three classes of techniques:

```text
PIOPs for EC arithmetic:
  efficient sum-check PIOPs for computations of pairing curves

copy constraints via sum-check:
  lightweight wiring checks for composing the PIOPs

prefix packing:
  packing multilinear polynomials to avoid homomorphism-related and
  padding-related costs
```

### PIOPs For EC Arithmetic

The operation-family formulas are PIOPs for the elliptic-curve and
extension-field operations that dominate Dory verification:

```text
GT arithmetic:
  GT multiplication via the polynomial division identity over Fq[X] modulo the
  degree-12 irreducible polynomial
  GT exponentiation via exponentiate-and-multiply with accumulator, shifted
  accumulator, digit selector, quotient, shift sum-check, and boundary checks
  batching GT instances across operation instances by stacking over a Boolean
  constraint index

G1 arithmetic:
  affine Weierstrass point addition with branch selectors and inverse witnesses
  scalar multiplication via double-and-add, shift sum-check, and boundary checks

G2 arithmetic:
  the same strategy as G1, with Fq2 coordinates split into Fq components

Miller-loop boundary:
  BN254 multi-Miller line schedule, line evaluation, pair-line products, and
  accumulator recurrence in the assist proof
  final exponentiation and pairing equality checked natively by the
  Dory-assist verifier in v1
```

### GT Semantics

`GT` elements are represented as 4-variate MLEs over the coefficient index,
encoding the 12 `Fq12` coefficients with four padding slots. Arithmetic is
checked by the polynomial division identity modulo the degree-12 irreducible
polynomial `gpoly(X)`.

Multiplication proves, for each coefficient-index point `x` and batched
operation instance:

```text
MulLeft(x) * MulRight(x) = MulOutput(x) + MulQuotient(x) * Modulus(x)
```

Equivalently the zero-check polynomial is:

```text
C_mul(x) = MulLeft(x) * MulRight(x)
         - MulOutput(x)
         - MulQuotient(x) * Modulus(x)
```

Exponentiation uses the paper's base-4 exponentiate-and-multiply recurrence.
For step bits `s`, coefficient bits `x`, accumulator `rho`, shifted accumulator
`rho_shift(s, x) = rho(s + 1, x)`, quotient `q`, and digit selector
`digit(s, x) = g^{d_s}(x)`, the local relation is:

```text
rho_shift(s, x) = rho(s, x)^4 * digit(s, x) + q(s, x) * Modulus(x)
```

The `digit` term is the generalized selector for the concrete base-4 digit
decomposition used by the reference implementation. Let `d0,d1` be the low and
high digit bits and let `B`, `B2`, `B3` encode `g`, `g^2`, and `g^3`. The
concrete base-4 selector relation is:

```text
digit = (1-d0)*(1-d1)
      + d0*(1-d1)*B
      + (1-d0)*d1*B2
      + d0*d1*B3
```

The digit bits are not trusted by construction. A separate bitness relation
checks:

```text
d0*(1-d0) = 0
d1*(1-d1) = 0
```

The base-power rows are checked with the same GT multiplication quotient
identity:

```text
B  * B = B2 + Q2 * Modulus
B2 * B = B3 + Q3 * Modulus
```

Those two checks are batched by a transcript challenge in the
`GtExponentiationBasePower` relation. Thus `GtExponentiation` can stay generic
over a selected `digit` value while the base-4 instantiation separately proves
that the selector is exactly Quang's `digit_lo/digit_hi/base/base2/base3`
construction.

`rho_shift` is not an independent committed object; it is connected to `rho`
by the GT shift sum-check. If the exponentiation zero-check emits the claim
`rho_shift(r_s, r_u, r_c)`, the shift relation proves:

```text
rho_shift(r_s, r_u, r_c)
  = sum_{s,u,c} EqPlusOne(r_s, s) * eq(r_u, u) * eq(r_c, c) * rho(s, u, c)
```

The final shift-round claim is therefore a public shift-kernel value times a
fresh opening of `rho` under the `GtExponentiationShift` relation. Boundary
checks use fixed public boundary selector evaluations to assert that the first
accumulator row is `1_GT` and the final shifted-accumulator row is the public
output `h` used by the Dory verifier operation DAG. Base powers for the digit
selector are part of the GT operation-family witness/public-input interface and
are checked with the same polynomial-division semantics.

### G1 Semantics

G1 points use affine coordinates `(x, y, iota)` where `iota = 1` encodes the
point at infinity as `(0, 0, 1)`. Addition proves `R = P + Q` with slope
`lambda`, inverse witness `mu = (x_Q - x_P)^-1` on the ordinary-add branch, and
branch selectors `sigma_1` for doubling and `sigma_2` for inverse points. With:

```text
dx  = x_Q - x_P
dy  = y_Q - y_P
phi = (1 - iota_P) * (1 - iota_Q)
```

the relation is the random linear combination of the 27 appendix constraints
plus an explicit branch-exclusivity check:

```text
booleanity:
  iota_P(1-iota_P), iota_Q(1-iota_Q), iota_R(1-iota_R)

infinity encoding:
  iota_P*x_P, iota_P*y_P, iota_Q*x_Q, iota_Q*y_Q, iota_R*x_R, iota_R*y_R

identity cases:
  iota_P*(R-Q), iota_P*(iota_R-iota_Q)
  iota_Q*(1-iota_P)*(R-P), iota_Q*(1-iota_P)*(iota_R-iota_P)

branch checks:
  phi*sigma_1(1-sigma_1)
  phi*sigma_2(1-sigma_2)
  phi*sigma_1*sigma_2
  phi*(1-sigma_1-sigma_2)*(1-mu*dx)
  phi*sigma_1*dx, phi*sigma_1*dy
  phi*sigma_2*dx, phi*sigma_2*(y_Q+y_P)

slope and output:
  phi*((1-sigma_1-sigma_2)*(lambda*dx-dy)
       + sigma_1*(2*y_P*lambda - 3*x_P^2))
  phi*sigma_2*(1-iota_R)
  phi*(1-sigma_2)*iota_R
  phi*(1-sigma_2)*(x_R - lambda^2 + x_P + x_Q)
  phi*(1-sigma_2)*(y_R - lambda*(x_P-x_R) + y_P)
```

Scalar multiplication uses the double-and-add recurrence
`A_{i+1} = [2]A_i + b_i P`. The local recurrence witnesses are accumulator
`A`, doubled point `T`, shifted accumulator `A'`, bit `b`, base `P`, and
infinity indicators `iota_A`, `iota_T`. With `dx = x_P - x_T` and
`dy = y_P - y_T`, the local constraints are:

```text
b*(1-b)
4*y_A^2*(x_T + 2*x_A) - 9*x_A^4
3*x_A^2*(x_T - x_A) + 2*y_A*(y_T + y_A)
(1-b)*(x_A' - x_T)
  + b*iota_T*(x_A' - x_P)
  + b*(1-iota_T)*((x_A' + x_T + x_P)*dx^2 - dy^2)
(1-b)*(y_A' - y_T)
  + b*iota_T*(y_A' - y_P)
  + b*(1-iota_T)*((y_A' + y_T)*dx - dy*(x_T - x_A'))
iota_A*(1-iota_T)
iota_T*x_T
iota_T*y_T
```

The scalar-mul shift relation separately enforces that `A'` is the one-step
shift of `A`. The scalar bit is explicitly constrained by `b*(1-b)=0`.
Boundary checks use fixed public boundary selector evaluations: the initial
accumulator row is constrained as `(0,0,1)`, and the final shifted accumulator
coordinates are constrained to the public affine scalar-mul output consumed by
the operation DAG.

### G2 Semantics

G2 mirrors G1 over `Fq2 = Fq[u]/(u^2 + 1)`. Every coordinate is split into
base-field components:

```text
a = a0 + a1*u
(a*b)_0 = a0*b0 - a1*b1
(a*b)_1 = a0*b1 + a1*b0
(a^2)_0 = a0^2 - a1^2
(a^2)_1 = 2*a0*a1
```

G2 addition uses the same branch, identity, slope, and output semantics as G1,
but each `Fq2` coordinate equation is split into its `c0` and `c1`
components. This yields 48 constraints: three scalar infinity booleanity
constraints, twelve infinity-encoding constraints, five `P = O` identity
constraints, five `Q = O` identity constraints, two branch booleanity
constraints, one branch-exclusivity constraint, two branch-selection
constraints for `mu*dx = 1`, four doubling enforcement constraints, four
inverse enforcement constraints, four split slope constraints, two
output-indicator constraints, and four output-coordinate constraints.

G2 scalar multiplication is the split-coordinate version of the G1 recurrence.
The two denominator-free doubling equations and two conditional-add equations
each produce `c0` and `c1` constraints, followed by the scalar infinity
constraints:

```text
b*(1-b)
iota_A*(1-iota_T)
iota_T*x_T0
iota_T*x_T1
iota_T*y_T0
iota_T*y_T1
```

The scalar-mul shift relation separately batches the four shifted accumulator
components `(x_A0', x_A1', y_A0', y_A1')` against `(x_A0, x_A1, y_A0, y_A1)`.
As in G1, the scalar bit is explicitly boolean-constrained, and fixed public
boundary selectors constrain the initial accumulator `(0,0,1)` and final
shifted affine coordinates.

The formulas should expose local constraints, virtual-polynomial dependencies,
opening IDs, public IDs, challenge IDs, and relation claim expressions. They
should not expose runtime tracing mechanics.

### Miller-Loop Semantics

The assist proof proves the BN254 multi-Miller loop, not the full pairing. The
ordinary final exponentiation and final equality check stay native in the
Dory-assist verifier.

For a configured pair list `(P_j, Q_j)`, the canonical BN254 schedule follows
the arkworks-style ate loop:

```text
prepared lines per nonzero pair:
  64 double lines
  21 signed-add lines from the non-leading loop digits
  2 Frobenius correction lines
  total = 87 line events

Miller accumulator ops:
  63 squares
  87 multiplications by pair-line products
  total = 150 accumulator ops
```

The PIOP should factor the work into four algebraic layers:

```text
line step:
  prove the G2 homogeneous projective double/add/Frobenius-correction schedule
  and produce the prepared line coefficients; the public line selector chooses
  the double formula or the generic add formula, while copy constraints bind
  the addend to Q, -Q, pi(Q), or -pi^2(Q)

line evaluation:
  evaluate each prepared line at the matching G1 point as the sparse Fq12
  element used by the BN254 D-twist `mul_by_034` path; the six nonzero Fq
  components are constrained and the remaining coefficient slots are zero

pair product:
  for each line event, multiply the line evaluations across all pairs to
  obtain one GT line product for that event; this relation owns the scan
  shift/boundary checks and uses GT multiplication rows for arithmetic

accumulator:
  run the Miller accumulator recurrence over the canonical 150-op schedule and
  expose the final pre-final-exponentiation GT value; this relation owns the
  accumulator shift checks and uses GT multiplication rows for square/multiply
  arithmetic
```

GT multiplications in pair-product and accumulator rows should be ordinary
`GtMultiplication` instances connected by direct copy constraints. Miller-loop
relations own schedule, shift, line formulas, sparse embedding, product order,
and boundaries; they do not reimplement the generic Fq12 multiplication
identity.

The `jolt-claims` semantics now exposes this decomposition directly:

```text
MillerLoopSelector:
  public line-step selector: LineDouble | LineAdd

MillerLoopConstant:
  public BN254 line-formula constants: TwoInverse, TwistB0, TwistB1

MillerLoopShiftEqKernel:
  public shift-kernel evaluations for pair-product and accumulator scans
```

### Copy Constraints Via Sum-Check

The paper also calls copy constraints "wiring": they enforce consistency when
composing EC PIOPs, so an operation output equals the downstream operation input
that consumes it.

V1 uses the paper's sum-check-based wiring perspective, but keeps the simplest
semantics: declared, directed equality edges. It must not use a PLONK-style
permutation argument, grand product accumulator, sorted table, or multiset
equality argument. The important recursion-paper point is that the verifier can
pay linear work in the small number of wiring edges, avoiding accumulator
polynomial commitments and the cost of verifying additional committed data.

### Prefix Packing

At the end of the PIOP, the verifier holds many evaluation claims on committed
multilinear polynomials of varying sizes. Homomorphic batching would require
many separate Hyrax commitments; padding everything to the largest shape would
pay too much padding. Prefix packing assigns each polynomial a prefix-free code,
places it on a disjoint subcube of one dense multilinear polynomial, and reduces
all virtual openings to one dense opening claim:

```text
packed_eval = sum_i w_i(r_pack) * v_i
```

where `w_i` is the multilinear subcube selector for the prefix assigned to the
`i`th polynomial. This is the stage-3 claim-reduction step. It can be expressed
as a degree-1 sum-check over the prefix variables, with the dense-witness
opening as input and the weighted reduced virtual claims as output.

Family packing should be used before prefix packing where the paper does:
witness polynomials of the same type across operation instances share a single
family-packed polynomial whose domain is extended by a family-local
constraint-index suffix.

### Staging Point Reduction

Dory assist should end with one Hyrax opening of one dense witness. To make that
true, relation outputs must enter prefix packing at a common suffix point.

Let the dense opening point be:

```text
r_dense = (r_prefix, r_suffix)
|r_suffix| = max_poly_vars
```

Each virtual witness polynomial `V_i` has native width `w_i <= max_poly_vars`.
The packing layout embeds it as a subcube:

```text
DenseWitness(prefix_i, z, pad) = V_i(z)
```

for `|z| = w_i`; the remaining padding variables are ignored by that embedded
polynomial. Therefore the claim entering prefix packing for `V_i` must be:

```text
V_i(r_suffix[..w_i])
```

not an independently sampled point. V1 should stage all component sum-checks so
their verifier challenges are deterministic projections of the common packing
suffix. If a later optimization genuinely needs a non-packing point, that
component must add an explicit same-polynomial claim reduction before stage 3;
such claims must not be passed directly into prefix packing.

This is the main invariant for preserving a single packed Hyrax opening:

```text
Every reduced virtual claim consumed by PrefixPacking is evaluated at the
canonical packing suffix projection for that virtual polynomial.
```

The prefix weights are public evaluations of the prefix selectors at
`r_prefix`. Padding slots for coefficient-indexed encodings, such as the four
unused slots in a 16-slot `GT` element layout, are constrained to zero or are
left unreachable by the component's canonical coefficient range.

## Stage Shape

The reference branch has moved toward a three-stage prefix-packing pipeline.
The modular protocol should preserve this high-level shape:

```text
stage 1:
  packed GT exponentiation sumcheck

stage 2:
  batched constraints:
    GT shift and claim reduction
    GT multiplication
    G1/G2 scalar multiplication
    G1/G2 addition
    Miller-loop line-step, line-evaluation, pair-product, accumulator, boundary
    declared wiring and public-input constraints

stage 3:
  prefix packing reduction to one dense polynomial opening

PCS opening:
  Hyrax opening of the dense witness using Grumpkin-backed Pedersen row
  commitments
```

Dory assist proves the Miller-loop work. The Dory-assist verifier receives the
resulting public GT value, computes final exponentiation directly, and checks
the public pairing equality. Final exponentiation is cheap deterministic
verifier work and stays native in v1, but it is still Dory-specific verifier
logic owned by the Dory-assist verifier implementation rather than by
`jolt-verifier`.

This split is an implementation choice, not a soundness omission. Native
verification checks final exponentiation natively after the Dory-assist PIOP.
Wrapped verification must prove or bind that same selected-verifier final check.
The simplest first implementation is an R1CS final-check gadget in
`jolt_dory_assist_verifier::r1cs`; a future final-exp PIOP can move that work
into the packed assist witness if wrapper cost warrants it.

## Copy Constraints

Dory-assist copy constraints, also referred to as wiring, are directed typed
equality edges over a declared copy-edge table. `jolt-claims` defines what it
means for those edges to be satisfied. It does not define how the edge table is
produced; a later `jolt-trace` plus witness-generation path derives those edges
from the Dory verifier program/runtime data.

The protocol must not use a permutation, grand-product, sorted-table, or
multiset-equality argument for copy constraints in v1.

Semantic object:

```text
CopyConstraint:
  id: CopyId
  value_type: GT | G1 | G2 | Scalar | Fp2 | ...
  source: ValueRef
  target: ValueRef

ValueRef:
  WitnessValue { relation, family, row, column, component }
  PublicValue { id, component }
  ChallengeValue { id }
  Constant
```

Fanout is represented as multiple copy edges with the same producer and
different targets. Public-input consistency is represented by the same
mechanism with `source = PublicValue(...)` and `target = WitnessValue(...)`.

Let the copy table have `2^m` rows after padding. Define virtual polynomials
over copy-index variables:

```text
Src(i)        = compressed value read from source endpoint of copy edge i
Dst(i)        = compressed value read from target endpoint of copy edge i
Enabled(i)    = fixed public mask, 1 for real copy edges and 0 for padding
EdgeWeight(i) = edge-batching challenge weight for edge i
```

For vector-valued objects, component values are compressed with a
Fiat-Shamir-derived challenge:

```text
compress_eta(v_0, ..., v_k) = v_0 + eta * v_1 + eta^2 * v_2 + ...
```

The copy constraint relation is:

```text
CopyDiff(i) = EdgeWeight(i) * Enabled(i) * (Src(i) - Dst(i))
CopyDiff == 0
```

The verifier samples `r_copy` and the prover reduces the ordered equality
check to:

```text
sum_i eq(r_copy, i) * EdgeWeight(i) * Enabled(i) * (Src(i) - Dst(i)) = 0
```

`Src` and `Dst` are virtual polynomials. `Enabled` is not prover-controlled: the
verifier evaluates the canonical edge-table mask at `r_copy` and exposes that
value as a public input to the wiring relation. `Src` and `Dst` must be derived
from typed `ValueRef`s into public inputs, transcript challenges, constants, or
operation-family witness columns. Stage 2 owns the copy zero-check claim; stage
3 and prefix packing bind the resulting witness-column claims to the single
dense Hyrax opening.

The `relation` field in `WitnessValue` is part of the semantic identity. The
same virtual polynomial family may appear under multiple relations, so copy
edges must distinguish, for example, `MillerLoopAccumulator.AccumulatorCoeff`
from `MillerLoopBoundary.AccumulatorCoeff`.

The paper's full wiring view indexes both a port domain `x` and a padded
family-local constraint domain `c`:

```text
sum_x sum_c eq(r_x, x) *
  sum_e lambda_e * (
      beta_src(e) * eq(c^src(e), idx_src(e)) * V_src(e)(x)
    - beta_dst(e) * eq(c^dst(e), idx_dst(e)) * V_dst(e)(x)
  ) = 0
```

The `jolt-claims` v1 semantics expose the same equality-edge meaning with typed
virtual `Src`/`Dst`, a public `Enabled` mask, tuple-compression, copy-point, and
edge-batch challenges. The later tracer/witness layer is responsible for
deriving the canonical edge table and any family-local selector/normalization
data from the Dory verifier operation DAG.

The prover may do work linear in the number of copy constraints. The verifier
may also do linear work in the small declared edge set. This follows the
recursion paper's motivation for sum-check-based wiring: avoid accumulator
polynomial commitments and the corresponding Hyrax verification cost. The
protocol semantics remain explicit equality edges rather than a permutation
argument.

## Component Model

The component split tracks the Dory verifier's algebraic domains:

```text
constraints:
  shared constraint families, poly types, arity, and sumcheck shapes

gt:
  GT exponentiation, GT multiplication, base powers, GT shift, GT wiring

g1:
  G1 addition, G1 scalar multiplication, scalar-mul shift, G1 wiring

g2:
  G2 addition, G2 scalar multiplication, scalar-mul shift, G2 wiring

miller_loop:
  line-step, line-evaluation, pair-product, accumulator, boundary, output GT

composition:
  prefix-packing claim catalog, native-width catalog, and declarative direct
  copy-edge stencils that compose Miller-loop rows with GT multiplication and
  boundary/public-output rows

packing:
  prefix-packing layout and dense-opening claim

protocol:
  canonical semantic relation bundle and relation order used by later verifier
  staging
```

The Dory assist proof is a multi-stage virtualization protocol in the same
sense that base Jolt has multiple protocol components. The implementation
should mirror the base Jolt claim organization: small top-level modules define
IDs and relation claim types, while `formulas/*` owns dimensions, semantic
claim formulas, openings, and wiring. Stage shape is verifier organization.

## Dimension And Reduction Catalog

The dimensions below are the semantic domains exposed by
`jolt-claims::protocols::dory_assist::formulas::dimensions`. They describe
where each relation lives before the final prefix-packing reduction.

Global constants:

```text
GT_ELEMENT_VARS = 4
  16 coefficient-index slots for Fq12 values; 12 are real coefficients and
  four are padding.

GT_EXP_BASE = 4
  base-4 exponentiation digits.

GT_EXP_STEP_VARS = 7
  enough for the base-4 exponentiation schedule used by the Dory verifier.

EC_SCALAR_MUL_STEP_VARS = 8
  enough for the fixed scalar-mul windows used by the operation DAG.

BN254_MILLER_LOOP_LINE_EVENTS = 87
  64 double line events, 21 signed-add line events from the non-leading
  BN254 ate-loop digits, plus 2 Frobenius correction line events.

BN254_MILLER_LOOP_SQUARE_OPS = 63
  one Miller accumulator square before every loop iteration except the first.

BN254_MILLER_LOOP_ACCUMULATOR_OPS = 150
  87 line-product multiplications plus 63 squares.

BN254_MILLER_LOOP_LINE_EVENT_VARS = 7
  padding 87 line events to 128.

BN254_MILLER_LOOP_ACCUMULATOR_OP_VARS = 8
  padding 150 accumulator ops to 256.
```

### GT Dimensions

`GtDimensions { exp_step_vars, exp_instance_vars, mul_instance_vars }` defines:

```text
GT exponentiation native domain:
  (exp_step, gt_coeff)
  rounds = exp_step_vars + GT_ELEMENT_VARS

GT exponentiation batched domain:
  (exp_step, exp_instance, gt_coeff)
  rounds = exp_step_vars + exp_instance_vars + GT_ELEMENT_VARS

GT base-power domain:
  (exp_instance, gt_coeff)
  rounds = exp_instance_vars + GT_ELEMENT_VARS

GT multiplication domain:
  (mul_instance, gt_coeff)
  rounds = mul_instance_vars + GT_ELEMENT_VARS
```

Reductions:

```text
GtExponentiation:
  proves the base-4 recurrence and emits claims on accumulator, shifted
  accumulator, digit selector, quotient, and modulus at the packing suffix
  projection for the exponentiation domain.

GtExponentiationDigitSelector:
  reduces digit bits and base powers to the selected digit value. This is the
  concrete base-4 Quang-style selector generalized in the formula layer.

GtExponentiationDigitBitness:
  proves each exponent digit bit is Boolean.

GtExponentiationBasePower:
  proves B^2 = B2 and B2*B = B3 using the GT multiplication quotient identity.

GtExponentiationShift:
  reduces shifted-accumulator claims to accumulator claims with the public
  one-step shift kernel.

GtExponentiationBoundary:
  reduces boundary claims to public initial/final selector values.

GtMultiplication:
  is the shared Fq12 multiplication relation. Miller-loop pair-product and
  accumulator multiplications should be staged as ordinary GT multiplication
  rows and connected back to the Miller-loop virtual columns with copy edges.
```

The GT component's output to prefix packing is a list of reduced scalar
coefficient claims, each already evaluated at the canonical suffix projection
for its GT virtual polynomial.

### G1 Dimensions

`G1Dimensions { scalar_mul_step_vars, scalar_mul_instance_vars,
add_instance_vars }` defines:

```text
G1 scalar-mul domain:
  (scalar_step, scalar_mul_instance)
  rounds = scalar_mul_step_vars + scalar_mul_instance_vars

G1 scalar-mul shift domain:
  (scalar_step)
  rounds = scalar_mul_step_vars

G1 addition domain:
  (add_instance)
  rounds = add_instance_vars
```

Reductions:

```text
G1ScalarMultiplication:
  proves the double-and-add row relation for x, y, infinity, doubled point,
  shifted accumulator, scalar bit, and base point.

G1ScalarMultiplicationShift:
  reduces shifted-accumulator claims to the next accumulator row.

G1ScalarMultiplicationBoundary:
  reduces initial accumulator and final output checks to public boundary
  selectors and boundary values.

G1Addition:
  proves affine addition with identity, doubling, inverse-point, slope, inverse
  witness, and branch-selector constraints.
```

The G1 component emits scalar-coordinate reduced claims for every G1 virtual
column consumed by later wiring or prefix packing.

### G2 Dimensions

`G2Dimensions` has the same domain shape as `G1Dimensions`, but each affine
coordinate is split into two Fq components:

```text
G2 scalar-mul domain:
  (scalar_step, scalar_mul_instance)

G2 scalar-mul shift domain:
  (scalar_step)

G2 addition domain:
  (add_instance)
```

Reductions mirror G1, except every Fq2 equation is split into `c0` and `c1`
base-field equations. G2 scalar-mul and addition outputs feed both ordinary
Dory verifier operations and the Miller-loop line-step schedule through direct
copy constraints.

### Miller-Loop Dimensions

`MillerLoopDimensions { line_event_vars, pair_vars, accumulator_op_vars }`
uses separate domains for line work and accumulator work:

```text
line-step domain:
  (line_event, pair)
  rounds = line_event_vars + pair_vars

line-evaluation domain:
  (line_event, pair)
  rounds = line_event_vars + pair_vars

pair-product domain:
  (line_event, pair)
  rounds = line_event_vars + pair_vars

accumulator domain:
  (accumulator_op)
  rounds = accumulator_op_vars

boundary domain:
  (accumulator_op)
  rounds = accumulator_op_vars
```

For BN254, `line_event_vars = 7` and `accumulator_op_vars = 8`. `pair_vars` is
configured from the Dory opening verifier's pair list and pads the number of
pairs to a power of two.

The Miller-loop subrelations are:

```text
MillerLoopLineStep:
  proves the canonical BN254 G2 prepared-line schedule over homogeneous
  projective coordinates. Each row is one of: double, signed add by Q, signed
  add by -Q, Frobenius correction add by pi(Q), or Frobenius correction add by
  -pi^2(Q). The local formula has public `LineDouble` and `LineAdd` selectors.
  Signed-add and Frobenius-correction rows share the same add formula; direct
  copy constraints bind the addend columns to the canonical Q, -Q, pi(Q), or
  -pi^2(Q) value. The double formula uses public BN254 constants
  `TwoInverse`, `TwistB0`, and `TwistB1`.

MillerLoopLineEvaluation:
  evaluates each prepared line at the corresponding G1 point. For BN254's D
  twist this is the sparse Fq12 line used by arkworks' `mul_by_034` path:
  coefficient 0 is scaled by P.y, coefficient 1 by P.x, and coefficient 2 is
  copied directly. In scalar-column form this gives six nonzero Fq components;
  the remaining GT coefficient slots are fixed to zero.

MillerLoopPairProduct:
  for each line_event, scans over pair and proves the product of all pair line
  evaluations. The actual Fq12 products are GT multiplication rows; this
  relation owns the scan order, shift-kernel claim reduction, initial
  accumulator boundary, and final line-product boundary. Direct copy
  constraints connect scan inputs/outputs to `GtMultiplication`.

MillerLoopAccumulator:
  scans the flattened BN254 Miller accumulator schedule. The 150 real ops are
  63 squares and 87 multiplications by pair-line products. Squares and
  multiplications are GT multiplication rows; this relation owns the canonical
  op ordering and accumulator shift-kernel claim reduction. Direct copy
  constraints connect accumulator rows, selected right operands, quotients, and
  outputs to `GtMultiplication`.

MillerLoopBoundary:
  enforces accumulator[0] = 1_GT and accumulator[last] =
  MillerLoopOutputGt. The output GT value is public and is the value consumed by
  native final exponentiation and equality checking.
```

Miller-loop reductions:

```text
line-step claims:
  reduce G2 state, addend, shifted state, line coefficients, public
  double/add selectors, and public BN254 line constants to suffix-projected
  scalar/Fq2 component claims.

line-evaluation claims:
  reduce G1 coordinates, line coefficients, sparse line-evaluation components,
  and zero sparse slots to suffix-projected claims.

pair-product claims:
  reduce pair-product scan state, shifted scan state, shift-kernel claim, pair
  boundaries, and final per-event line products to suffix-projected claims.
  GT multiplication row inputs/outputs are connected by copy constraints.

accumulator claims:
  reduce accumulator state, shifted accumulator state, and shift-kernel claim
  to suffix-projected claims. Pair-line-product inputs, square/multiply
  operands, GT multiplication outputs, and quotient columns are connected by
  copy constraints.

boundary claims:
  reduce initial/final accumulator checks to public selectors and the public
  `MillerLoopOutputGt(k)` coefficients.
```

The Miller-loop component must not duplicate generic GT multiplication
semantics. It composes G2 line semantics, G1 evaluation-point semantics, GT
multiplication semantics, and direct copy constraints. Final exponentiation and
the final equality check are outside the SNARK portion and remain native
Dory-assist verifier work.

### Wiring Dimensions

`WiringDimensions { log_edges }` defines the copy-edge domain:

```text
copy domain:
  (edge)
  rounds = log_edges
```

Reductions:

```text
WiringGt / WiringG1 / WiringG2:
  compress typed vector values with the tuple-compression challenge, apply the
  public enabled mask and edge-batch weight, and zero-check Src - Dst.
```

The output claims are only `Source` and `Destination` virtual claims at the
copy-point projection. Endpoint expansion from typed `ValueRef`s to concrete
witness columns/publics/challenges is a verifier/witness-generation concern,
but the equality semantics are owned here.

### Prefix-Packing Dimensions

`PrefixPackingDimensions { packed_vars, max_poly_vars, num_claims }` defines:

```text
prefix_vars = packed_vars - max_poly_vars
r_dense = (r_prefix, r_suffix)
```

Reductions:

```text
PrefixPacking:
  input claim:
    DenseWitness(r_dense)

  output claim:
    sum_i PrefixPackingWeight(i) * ReducedClaim_i
```

`num_claims` is the number of reduced virtual claims emitted by GT, G1, G2,
Miller-loop, and wiring components. `max_poly_vars` is the maximum native width
among those reduced claims. Every claim entering this reduction must satisfy the
canonical suffix-projection invariant described above.

The formula layer also exposes a composition catalog that derives the prefix
packing inputs from `DoryAssistDimensions`. It records, for each reduced opening:

```text
PackingEntry:
  opening
  native_vars
```

The catalog includes GT, G1, G2, Miller-loop, wiring outputs, and every witness
endpoint referenced by the Miller-loop direct copy-edge stencil. Its minimal
packing shape is:

```text
max_poly_vars = max_i native_vars_i
prefix_vars  = ceil_log2(num_claims)
packed_vars  = max_poly_vars + prefix_vars
```

The resulting `PrefixPacking` claim uses the catalog order exactly:

```text
DenseWitness(r_prefix, r_suffix)
  = sum_i PrefixPackingWeight(i) * ReducedClaim_i(r_suffix[..native_vars_i])
```

The Miller-loop copy-edge stencil is not the runtime trace. It is the semantic
template that the later tracing/witness-generation layer expands into concrete
copy rows. It includes:

```text
line-step shifted G2 state -> next line-step G2 state
line-step line coefficients -> line-evaluation line coefficients
line-evaluation sparse GT coefficients -> GT multiplication right operand
pair-product scan/current/output/quotient columns <-> GT multiplication rows
pair-line products -> accumulator GT multiplication right operand
accumulator square/multiply operands/outputs/quotients <-> GT multiplication rows
accumulator columns -> Miller-loop boundary columns
Miller-loop boundary final GT -> public MillerLoopOutputGt
```

Tests in `jolt-claims` assert that the catalog has no duplicate openings, spans
the intended component families, produces a fitting minimal prefix-packing
shape, uses the catalog order in the prefix-packing claim, and that all witness
endpoints used by the Miller-loop copy-edge stencil are present in the packing
catalog.

### Canonical Protocol Bundle

`jolt-claims` exposes a semantic bundle constructor:

```rust
dory_assist::formulas::protocol_claims(dimensions)
```

This is not transcript replay and not verifier staging. It is the canonical
ordered list of semantic relation claims:

```text
GT:
  GtExponentiation
  GtExponentiationDigitSelector
  GtExponentiationBasePower
  GtExponentiationDigitBitness
  GtExponentiationShift
  GtExponentiationBoundary
  GtMultiplication

G1:
  G1ScalarMultiplication
  G1ScalarMultiplicationShift
  G1ScalarMultiplicationBoundary
  G1Addition

G2:
  G2ScalarMultiplication
  G2ScalarMultiplicationShift
  G2ScalarMultiplicationBoundary
  G2Addition

Miller loop:
  MillerLoopLineStep
  MillerLoopLineEvaluation
  MillerLoopPairProduct
  MillerLoopAccumulator
  MillerLoopBoundary

Wiring:
  WiringGt
  WiringG1
  WiringG2

Packing:
  PrefixPacking
```

The prefix-packing relation in the bundle is built from the composition catalog
so its reduced openings and `PrefixPackingWeight(i)` publics use the same
catalog order. Tests assert canonical relation order, no duplicate relation IDs,
protocol-level dependency aggregation, catalog-backed prefix packing, copy-edge
endpoint packability, and the single dense-witness opening invariant.

## `jolt-claims` Layout

Target layout:

```text
crates/jolt-claims/src/protocols/dory_assist/
  mod.rs
  ids.rs
  relation.rs
  formulas/
    mod.rs
    dimensions.rs
    error.rs
    gt.rs
    g1.rs
    g2.rs
    miller_loop.rs
    composition.rs
    wiring.rs
    packing.rs
    protocol.rs
    committed_openings.rs
    claim_reductions/
      mod.rs
      gt.rs
      g1.rs
      g2.rs
      miller_loop.rs
```

This intentionally mirrors `protocols::jolt`, not a verifier crate. In
particular:

```text
mod.rs:
  declares formulas, ids, relation
  reexports dimensions/errors, IDs, and relation claim types

ids.rs:
  relation IDs
  committed/virtual polynomial IDs
  opening IDs
  public IDs
  challenge IDs

relation.rs:
  DoryAssistExpr
  DoryAssistInputClaimExpression
  DoryAssistOutputClaimExpression
  DoryAssistConsistencyClaim
  DoryAssistRelationClaims
  DoryAssistProtocolClaims

formulas/:
  dimensions and formula constructors for semantic relations
```

`jolt-claims` should not contain Dory-assist proof payloads, verifier-stage
payloads, transcript replay code, opening-snapshot construction, or runtime
tracing plans. Those belong to `jolt-dory-assist-verifier`, `jolt-verifier`, or
the later `jolt-trace`/witness-generation layer.

Top-level API shape:

```rust
pub mod formulas;

mod ids;
mod relation;

pub use formulas::{
    dimensions::{DoryAssistDimensions, DoryAssistSumcheckSpec},
    error::{DoryAssistFormulaDimensionsError, DoryAssistFormulaPointError},
};
pub use ids::{
    DoryAssistChallengeId, DoryAssistCommittedPolynomial, DoryAssistOpeningId,
    DoryAssistPolynomialId, DoryAssistPublicId, DoryAssistRelationId,
    DoryAssistVirtualPolynomial, G1Challenge, G2Challenge, GtChallenge,
    MillerLoopChallenge, PackingChallenge, WiringChallenge,
};
pub use relation::{
    DoryAssistConsistencyClaim, DoryAssistExpr, DoryAssistInputClaimExpression,
    DoryAssistOutputClaimExpression, DoryAssistProtocolClaims,
    DoryAssistRelationClaims,
};
```

Relation API shape:

```rust
pub type DoryAssistExpr<F> =
    Expr<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;
pub type DoryAssistInputClaimExpression<F> =
    InputClaimExpression<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;
pub type DoryAssistOutputClaimExpression<F> =
    OutputClaimExpression<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;
pub type DoryAssistConsistencyClaim<F> =
    ConsistencyClaim<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;

pub struct DoryAssistRelationClaims<F> {
    pub id: DoryAssistRelationId,
    pub sumcheck: DoryAssistSumcheckSpec,
    pub input: DoryAssistInputClaimExpression<F>,
    pub output: DoryAssistOutputClaimExpression<F>,
    pub consistency: Vec<DoryAssistConsistencyClaim<F>>,
}

pub struct DoryAssistProtocolClaims<F> {
    pub relations: Vec<DoryAssistRelationClaims<F>>,
}
```

Representative relation IDs:

```rust
pub enum DoryAssistRelationId {
    GtExponentiation,
    GtExponentiationDigitSelector,
    GtExponentiationBasePower,
    GtExponentiationDigitBitness,
    GtExponentiationShift,
    GtExponentiationBoundary,
    GtMultiplication,
    G1ScalarMultiplication,
    G1ScalarMultiplicationShift,
    G1ScalarMultiplicationBoundary,
    G1Addition,
    G2ScalarMultiplication,
    G2ScalarMultiplicationShift,
    G2ScalarMultiplicationBoundary,
    G2Addition,
    MillerLoopLineStep,
    MillerLoopLineEvaluation,
    MillerLoopPairProduct,
    MillerLoopAccumulator,
    MillerLoopBoundary,
    WiringGt,
    WiringG1,
    WiringG2,
    PrefixPacking,
}
```

The three-stage schedule is verifier organization, not a `jolt-claims` module.
`jolt-dory-assist-verifier` should map these semantic relations into stage 1,
stage 2, stage 3, Hyrax opening verification, and the native final check.

## Hyrax

`jolt-hyrax` factors Hyrax out as a reusable modular crate. Dory assist uses
Hyrax to commit to and open the packed dense witness. The first implementation is
a native transparent PCS adapter over `jolt_crypto::VectorCommitment`. Wrapper
assembly later adds `jolt-hyrax::r1cs` to prove the same opening verification
inside R1CS when the configured verifier is wrapped.

V1 native layout:

```text
crates/jolt-hyrax/
  Cargo.toml
  src/
    lib.rs
    dimensions.rs
    error.rs
    commitment.rs
    proof.rs
    setup.rs
    scheme.rs
```

Future R1CS layout:

```text
crates/jolt-hyrax/src/r1cs/
  mod.rs
  inputs.rs
  witness.rs
  constraints.rs
```

Hyrax should use existing abstractions rather than defining parallel ones:

```text
VectorCommitment
VectorCommitmentOpening
HomomorphicCommitment
Pedersen / PedersenSetup
JoltGroup
AppendToTranscript
jolt_openings::CommitmentScheme
jolt_openings::AdditivelyHomomorphic
```

Native API sketch:

```rust
pub struct HyraxDimensions {
    pub num_vars: usize,
    pub row_vars: usize,
    pub col_vars: usize,
}

pub struct HyraxCommitment<C> {
    pub rows: Vec<C>,
}

pub struct HyraxOpeningProof<F> {
    pub combined_row: Vec<F>,
    pub combined_row_opening_scalar: F,
}

pub struct HyraxScheme<VC: VectorCommitment> { ... }

impl<VC> CommitmentScheme for HyraxScheme<VC>
where
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>;
```

The point split is fixed as:

```text
num_vars = row_vars + col_vars
row_point = point[..row_vars]
col_point = point[row_vars..]
```

Hyrax combines committed rows using `row_point`, then verifies the combined row
opening at `col_point`. The implementation should delegate row commitments,
row-combination openings, and commitment homomorphism to `VectorCommitment`,
`VectorCommitmentOpening`, and `HomomorphicCommitment`.

The default instantiation can be Pedersen-backed:

```text
PedersenHyrax<G> = Hyrax<Pedersen<G>>
```

Dory assist's production instantiation uses Grumpkin for those Pedersen row
commitments:

```text
DoryAssistHyrax = Hyrax<Pedersen<GrumpkinPoint>>
```

Verifier APIs should still be generic over the vector commitment abstraction.
Pedersen is generic over `JoltGroup`, and the vector-commitment scalar field is
the group's associated `ScalarField`; for Grumpkin this is BN254 `Fq`. Arkworks
types must stay behind `jolt-field` and `jolt-crypto` backend modules. Public
assist, Hyrax, and verifier APIs should use Jolt wrapper types such as
`jolt_field::Fq` and `jolt_crypto::GrumpkinPoint`, not arkworks types.

Hyrax setup should compose with existing `jolt_crypto::DeriveSetup` impls. For
Dory assist, the intended path is:

```text
DoryProverSetup
  -> PedersenSetup<Grumpkin> via DeriveSetup<DoryProverSetup>
  -> HyraxProverSetup<Pedersen<Grumpkin>> using HyraxDimensions.row_len()
```

`HyraxDimensions` are not derivable from the PCS SRS and must be supplied by the
Dory-assist dense-witness layout. If the initial native `jolt-hyrax` tests use a
different available `JoltGroup` implementation, that is only a test fixture; the
Dory-assist verifier configuration should select Grumpkin.

## Generic PCS-Assist Integration

`jolt-verifier` owns the configured linear verifier flow and the generic
opening-phase assist boundary. Dory assist is one implementation of that
boundary, selected only when the configured PCS is Dory and the verifier config
requires the Dory-assist payload.

Verifier-facing API sketch:

```rust
pub trait PcsProofAssist<PCS: CommitmentScheme> {
    type Proof;
    type Config;
    type Error;

    fn verify_clear<T>(
        config: &Self::Config,
        input: PcsAssistClearInput<'_, PCS>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = PCS::Field>;

    fn verify_zk<T>(
        config: &Self::Config,
        input: PcsAssistZkInput<'_, PCS>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<<PCS as ZkOpeningScheme>::HidingCommitment, Self::Error>
    where
        PCS: ZkOpeningScheme,
        T: Transcript<Challenge = PCS::Field>;
}
```

The exact trait spelling can move during implementation, but the boundary
should preserve these semantics:

```text
ordinary configured Jolt:
  stages 1-7
  -> build the reduced opening statement
  -> proof.pcs_assist must be None
  -> verify joint_opening_proof through the ordinary PCS verifier

Dory-assisted configured Jolt:
  stages 1-7
  -> build the same reduced opening statement
  -> proof.pcs_assist must be Some(DoryAssistProof)
  -> call DoryAssist::verify_clear / verify_zk through the PcsProofAssist boundary
  -> pass the whole Dory PCS proof as `input.pcs_proof`
  -> do not also run the expensive ordinary Dory verifier path
```

The Dory-assist implementation consumes the entire Dory PCS proof. Generic
`jolt-verifier` must not parse or project Dory proof internals. Its job is to
construct the same reduced opening statement that ordinary PCS verification
would use and pass that statement, the `PCS::Proof`, the selected assist proof,
and the continued transcript to the selected assist implementation.

Because this assist proof is conceptually over Grumpkin, its scalar challenges
are BN254 `Fq`. The surrounding Jolt verifier transcript remains over BN254
`Fr`. Dory assist therefore continues the ordinary Jolt transcript, squeezes
`Fr` challenges from it, and injects those canonical `Fr` values into `Fq` on
both prover and verifier sides. There is no hidden independent `Fq` transcript.

The configured verifier flow and proof-shape rules live in
[selected-verifier-integration.md](selected-verifier-integration.md).

## `jolt-dory-assist-verifier` Layout

The `jolt-dory-assist-verifier` crate owns the concrete organization of the
semantics defined in `jolt-claims::protocols::dory_assist`.

Target layout:

```text
crates/jolt-dory-assist-verifier/
  Cargo.toml
  src/
    lib.rs
    config.rs
    proof.rs
    public_inputs.rs
    public_outputs.rs
    transcript.rs
    opening_statement.rs
    stages/
      mod.rs
      stage1/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      stage2/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
      stage3/
        mod.rs
        inputs.rs
        outputs.rs
        verify.rs
    hyrax_opening/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    final_check.rs
```

The crate should expose a typed proof payload and implement the generic
PCS-assist boundary for the Dory PCS:

```rust
pub struct DoryAssistProof<F, T, HyraxOpeningProof> {
    pub stage1_proof: SumcheckInstanceProof<F, T>,
    pub stage2_proof: SumcheckInstanceProof<F, T>,
    pub stage3_packed_eval: F,
    pub opening_proof: HyraxOpeningProof,
    pub opening_claims: DoryAssistOpeningClaims<F>,
    pub dense_commitment: HyraxCommitment,
    pub public_outputs: DoryAssistPublicOutputs<F>,
}

pub struct DoryAssist;

impl PcsProofAssist<DoryCommitmentScheme> for DoryAssist {
    type Proof = DoryAssistProof<...>;
    type Config = DoryAssistConfig;
    type Error = DoryAssistVerifierError;

    fn verify_clear<T>(
        config: &Self::Config,
        input: PcsAssistClearInput<'_, DoryCommitmentScheme>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = <DoryCommitmentScheme as CommitmentScheme>::Field>,
    {
        // stage 1, stage 2, stage 3, Hyrax opening, native final check
    }

    fn verify_zk<T>(
        config: &Self::Config,
        input: PcsAssistZkInput<'_, DoryCommitmentScheme>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<<DoryCommitmentScheme as ZkOpeningScheme>::HidingCommitment, Self::Error>
    where
        T: Transcript<Challenge = <DoryCommitmentScheme as CommitmentScheme>::Field>,
    {
        // stage 1, stage 2, stage 3, Hyrax opening, native final check
    }
}
```

The proof fields should remain typed by stage and operation family. Production
code should not route claims through an untyped opening map.

## R1CS And Wrapper Hooks

Dory assist contributes verifier computation to the wrapper:

```text
stage 1/2 sumcheck checks:
  jolt-sumcheck::r1cs

operation-family claim equations:
  jolt-claims formulas + jolt-r1cs lowering

wiring and public-input equations:
  dory_assist component claims + jolt-r1cs lowering

prefix packing:
  dory_assist::packing R1CS helper

dense Hyrax opening:
  jolt-hyrax::r1cs
```

The wrapper consumes these helpers through the configured verifier computation
and the selected `PcsProofAssist` implementation. Dory-specific formulas stay
in `jolt-claims`; Dory-specific stage organization stays in the Dory-assist
verifier crate; Hyrax-specific constraints stay in `jolt-hyrax`.

## Interaction With Field Inline

Dory assist is independent of field inline. If field inline is enabled, its
effects are already reflected in the composed verifier stage outputs and
opening data consumed by the Dory-opening path. Dory assist does not need a
separate field-inline mode.

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Add `jolt-hyrax` native API.
   - Define setup, commitment, proof, opening claim, and verifier APIs.
   - Use `jolt-crypto` vector commitment and group abstractions.
   - Review gate: Hyrax opening fixture tests pass.

2. Add `jolt-hyrax::r1cs`.
   - Encode the verifier equations needed by Dory assist.
   - Keep R1CS helpers generic over `jolt-r1cs` builders.
   - Review gate: R1CS verifier constraints match native Hyrax verification on
     small fixtures.

3. Add `jolt-claims::protocols::dory_assist`.
   - Define IDs, dimensions, relation IDs, public IDs, and opening IDs.
   - Mirror the base Jolt structure rather than inventing a separate verifier
     spec abstraction.
   - Review gate: API shape matches `protocols::jolt` and covers the relation
     set needed by the three-stage reference pipeline.

4. Add operation-family formulas.
   - Add GT, G1, G2, Miller-loop, wiring, and claim-reduction formulas.
   - Encode local constraints, output claims, virtual-polynomial dependencies,
     and wiring formulas.
   - Review gate: formula tests cover each operation family independently.

5. Add prefix-packing formulas.
   - Define dense witness layout, prefix codewords, and dense-opening claim.
   - Review gate: packing tests catch incorrect prefix layout and opening-point
     normalization.

6. Add the `jolt-dory-assist-verifier` crate and proof shape.
   - Add typed proof payloads for stage 1, stage 2, stage 3, and Hyrax opening.
   - Organize verification stages around the semantic claims from
     `jolt-claims::protocols::dory_assist`.
   - Review gate: proof-shape validation rejects missing, extra, or misordered
     stage payloads.

7. Implement the generic PCS-assist boundary for Dory.
   - Implement `PcsProofAssist<DoryCommitmentScheme>` for the
     `jolt_dory_assist_verifier::DoryAssist` implementation type.
   - Build Dory-assist public inputs from the generic reduced opening statement
     and the ordinary joint opening proof.
   - Review gate: `jolt-verifier` can dispatch to Dory-assist
     fixtures without Dory-specific stage modules.

8. Add Miller-loop verification path.
   - Prove Miller-loop witness constraints in Dory assist.
   - Keep final exponentiation as native Dory-assist verifier work.
   - Review gate: fixtures match the Quang reference branch for equivalent
     inputs.

9. Add wrapper R1CS hooks.
   - Lower Dory-assist stages through claim, sumcheck, packing, and Hyrax R1CS
     helpers.
   - Review gate: wrapper assembly can include configured Dory-assist stages in
     a satisfied R1CS.
