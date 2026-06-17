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
polynomial `gpoly(X)`. The native `Bn254GT` and `Bn254Fq12` bindings expand the
real slots in arkworks tower order `(c0.c0, c0.c1, c0.c2, c1.c0, c1.c1,
c1.c2)`, emitting each Fq2 coefficient as `(c0, c1)`, and pad slots 12 through
15 with zero.

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

The verifier implementation uses direct-copy machinery for the active GT, G1,
G2, Miller-loop, and singleton Dory-reduce Stage 1 building blocks. The
canonical GT-local copy stencil is declared by
`jolt_claims::protocols::dory_assist::formulas::composition::
gt_copy_constraints()` and includes these equality edges:

```text
GtExponentiation.ExpDigitSelector
  == GtExponentiationDigitSelector.ExpDigitSelector

GtExponentiationDigitSelector.ExpDigitBit(0/1)
  == GtExponentiationDigitBitness.ExpDigitBit(0/1)

GtExponentiationDigitSelector.ExpBasePower(1/2/3)
  == GtExponentiationBasePower.ExpBasePower(1/2/3)

GtExponentiation.Modulus
  == GtExponentiationBasePower.Modulus

GtExponentiation.ExpAccumulator
  == GtExponentiationShift.ExpAccumulator
  == GtExponentiationBoundary.ExpAccumulator

GtExponentiation.ExpShiftedAccumulator
  == GtExponentiationBoundary.ExpShiftedAccumulator
```

This is deliberately direct equality over typed endpoints. It does not use a
permutation argument. The same verifier pattern is active for G1/G2
scalar-multiplication shift and boundary edges, Miller-loop line/evaluation,
pair-product, accumulator, quotient, and boundary edges, and singleton
Dory-reduce public-artifact/setup/transcript-scalar copy edges. Multi-round
Dory-reduce public vectors use Stage 2 public folds instead of row-indexed
direct copy edges.

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
  copy-edge stencils that compose G1/G2 scalar-mul reductions, Miller-loop rows
  with GT multiplication, and boundary/public-output rows

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
base-field equations. The static copy stencil wires scalar-mul local columns to
their shift and boundary reductions. Concrete operation-DAG edges from G2
scalar-mul/addition outputs into ordinary Dory verifier operations and the
Miller-loop line-step schedule are trace-generated direct copy rows.

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

### Dory-Reduce Semantics

The Dory opening verifier maintains per-round state
`(C, D1, D2, E1, E2, s1_acc, s2_acc)` and updates it with the two reduce
messages, verifier setup constants, and transcript scalars. The formula layer
declares this as six semantic families:

```text
DoryReduceGtTransition:
  C'  = C + chi_i + beta * D2 + beta^{-1} * D1
        + alpha * C_plus + alpha^{-1} * C_minus
  D1' = alpha * D1_left + D1_right
        + alpha * beta * delta_1l_i + beta * delta_1r_i
  D2' = alpha^{-1} * D2_left + D2_right
        + alpha^{-1} * beta^{-1} * delta_2l_i
        + beta^{-1} * delta_2r_i

DoryReduceG1Transition:
  E1' = E1 + beta * E1_beta + alpha * E1_plus + alpha^{-1} * E1_minus

DoryReduceG2Transition:
  E2' = E2 + beta^{-1} * E2_beta + alpha * E2_plus
        + alpha^{-1} * E2_minus

DoryReduceScalarFold:
  s1_acc' = s1_acc * s1_fold_factor
  s2_acc' = s2_acc * s2_fold_factor

DoryReduceStateChain:
  combine_gamma(C', D1', D2', E1', E2', s1_acc', s2_acc')
    = DoryReduceShiftEqKernel
      * combine_gamma(C, D1, D2, E1, E2, s1_acc, s2_acc)

DoryReduceBoundary:
  BoundarySelector(initial)
    * combine_gamma(CurrentC - VMV_C,
                    CurrentD1 - JoltCommitment,
                    CurrentD2 - VMV_D2,
                    CurrentE1 - VMV_E1,
                    CurrentE2 - DoryReduceInitialE2,
                    s1_acc - 1,
                    s2_acc - 1)
  +
  BoundarySelector(final)
    * combine_gamma(NextC - NativeFinalCheckInput(C_acc),
                    NextD1 - NativeFinalCheckInput(D1_acc),
                    NextD2 - NativeFinalCheckInput(D2_acc),
                    NextE1 - NativeFinalCheckInput(E1_acc),
                    NextE2 - NativeFinalCheckInput(E2_acc),
                    s1_acc' - NativeFinalCheckInput(s1_acc),
                    s2_acc' - NativeFinalCheckInput(s2_acc))
  = 0
```

Here `i = reduce_rounds - round`, matching `DoryVerifierState::process_round`
where setup arrays are indexed by the remaining round count. The equations
above use Dory additive group notation: GT, G1, and G2 additions/scalar
multiplications are group operations that the Dory-assist protocol decomposes
through the corresponding algebraic component semantics, not raw coefficient
linear equalities.

The direct-copy semantics for each reduce round bind:

```text
Initial reducer state:
  DoryProofArtifact VMV C
    -> DoryReduce CurrentC

  JoltCommitment GT coefficients
    -> DoryReduce CurrentD1

  DoryProofArtifact VMV D2
    -> DoryReduce CurrentD2

  DoryProofArtifact VMV E1
    -> DoryReduce CurrentE1

  DoryReduceInitialE2
    -> DoryReduce CurrentE2

  constant 1
    -> DoryReduce s1_acc, s2_acc

DoryProofArtifact reduce messages
  -> DoryReduce message columns

VerifierSetupArtifact chi/delta constants at remaining-round index i
  -> DoryReduce setup columns

TranscriptScalar beta, beta^{-1}, alpha, alpha^{-1},
alpha*beta, alpha^{-1}*beta^{-1}, s1_fold_factor, s2_fold_factor
  -> DoryReduce scalar columns

Row chaining:
  DoryReduce NextC, NextD1, NextD2 at reduce row r
    -> DoryReduce CurrentC, CurrentD1, CurrentD2 at reduce row r + 1

  DoryReduce NextE1, NextE2 at reduce row r
    -> DoryReduce CurrentE1, CurrentE2 at reduce row r + 1

  DoryReduce s1_acc', s2_acc' at reduce row r
    -> DoryReduce s1_acc, s2_acc at reduce row r + 1
```

The final Dory scalar-product check is separate from the per-round reduce
transitions. Its `gamma`, optional `sigma_c`, `d`, `d^{-1}`, and `d^2` scalars
are staged in the same `TranscriptScalar` space so the final semantic layer can
reuse the native-Fr transcript replay without recomputing Fr arithmetic in Fq.
The scalar replay follows `jolt-dory`'s verifier adapter exactly: the opening
point is first reversed before entering upstream Dory, so the folded scalar
coordinates use Dory's `(sigma columns, nu rows)` order rather than Jolt's
external opening-point order. The external point coordinates are still exposed
in `TranscriptScalar` in Jolt order for checked-input binding.

Current verifier staging activates the Fq-native Dory-reduce algebra in Stage
1: GT, G1, and G2 transition relations batch the Dory verifier's per-round
linear update formulas, `DoryReduceScalarFold` batches the two scalar
accumulator folds, `DoryReduceStateChain` batches next-row/current-row
chaining, and `DoryReduceBoundary` binds the first and last reducer rows to
checked/public verifier data. Initial `E2` is exposed as a checked
mode-aware public input: transparent openings use `eval * g2_0`, while ZK
openings use the Dory proof `e2` artifact. The final reducer state is exposed
through `NativeFinalCheckInput`, which is also consumed by the native final
pairing check.

In the singleton reduce-round case, Stage 2 still directly copies the initial
`C/D1/D2/E1/s1_acc/s2_acc` reducer state, Dory proof reduce messages, verifier
setup `chi`/`delta` constants, and the relevant Fr-derived transcript scalars
into the GT/G1/G2 transition openings, and copies `s1_fold_factor` and
`s2_fold_factor` from `TranscriptScalar` into the scalar-fold relation
openings. Stage 2 additionally computes row-aware public folds for
round-indexed Dory-reduce inputs: for each public vector it evaluates the
multilinear extension at the relation's reduce-round sumcheck point and checks
that value against the corresponding Dory-reduce opening. For multi-round
dimensions, direct row-chain copy edges are deliberately omitted; row shifting
is owned by `DoryReduceStateChain`, and initial/final state binding is owned by
`DoryReduceBoundary`.

The verifier's Stage 1 catalog is dimension-aware: singleton reducer
dimensions use the transition/scalar-fold catalog, while multi-round reducer
dimensions add `DoryReduceStateChain` and `DoryReduceBoundary`. Runtime
verification accepts Dory-reduce dimensions that match the checked PCS proof's
point length and reduce-round count, derives the corresponding Stage 1/2/3
catalogs, and checks the packed Hyrax opening over the resulting reduced
opening set. The verifier test suite pins this with real clear and ZK
multi-round Dory openings (`num_vars = 4`, two reduce rounds), plus public-fold,
state-chain, and boundary tamper tests in both modes. Prover/runtime tracing
still has to replace the synthetic verifier fixtures with traced Dory-verifier
witness generation.

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
endpoint referenced by the active direct copy-edge stencils. Its minimal
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

The active copy-edge stencils are not the runtime trace. They are the semantic
templates that the later tracing/witness-generation layer expands into concrete
copy rows. They include:

```text
G1 scalar-mul accumulator/shifted-accumulator columns -> G1 scalar-mul shift
G1 scalar-mul accumulator/shifted-accumulator columns -> G1 boundary columns
G2 scalar-mul accumulator/shifted-accumulator columns -> G2 scalar-mul shift
G2 scalar-mul accumulator/shifted-accumulator columns -> G2 boundary columns
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
Dory assist, the Grumpkin setup must not be derived from the Dory/Bn254 SRS.
The intended path is a transparent Grumpkin-native seed:

```text
GrumpkinPedersenSetupSeed(domain, seed)
  -> PedersenSetup<GrumpkinPoint> via DeriveSetup<GrumpkinPedersenSetupSeed>
  -> HyraxProverSetup<Pedersen<GrumpkinPoint>> using HyraxDimensions.row_len()
  -> HyraxVerifierSetup<Pedersen<GrumpkinPoint>> using the same dimensions
```

The seed derivation hashes a domain-separated seed to Grumpkin curve points,
validates each point, and uses separate roles for message generators and the
Pedersen blinding generator. It must not derive generators as public scalar
multiples of the standard Grumpkin generator, because known discrete-log
relations between Pedersen bases would break binding.

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
The injection is the canonical little-endian integer representative of the
sampled `Fr` value, reduced into `Fq`; for the BN254 cycle this preserves the
intended scalar value and gives prover/verifier identical `Fq` challenges.

The configured verifier flow and proof-shape rules live in
[selected-verifier-integration.md](selected-verifier-integration.md).

## `jolt-dory-assist-verifier` Layout

The `jolt-dory-assist-verifier` crate owns the concrete organization of the
semantics defined in `jolt-claims::protocols::dory_assist`.

Current layout:

```text
crates/jolt-dory-assist-verifier/
  Cargo.toml
  src/
    lib.rs
    config.rs
    error.rs
    proof.rs
    setup.rs
    verifier.rs
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
  tests/
    completeness.rs
    completeness/
      mod.rs
      cases.rs
      fixtures.rs
      oracle.rs
    soundness.rs
    soundness/
      mod.rs
      fixtures.rs
      tampering/
        mod.rs
        manifest.rs
        inputs.rs
        stages.rs
        openings.rs
        public_outputs.rs
    support/
      mod.rs
```

The crate mirrors `jolt-verifier` rather than splitting every small concept
into its own module:

- `lib.rs` is a thin module/export surface.
- `error.rs` owns the stage-aware verifier error taxonomy.
- `proof.rs` owns serialized proof data: the proof envelope, stage-proof
  envelope, opening claims, and public outputs.
- `setup.rs` owns the fixed Dory-assist Grumpkin Hyrax seed and helper aliases
  for deriving prover/verifier Hyrax setup from `HyraxDimensions`.
- `verifier.rs` owns the `DoryAssist` implementation type, checked/normalized
  inputs, the generic `PcsProofAssist` implementation, and top-level stage
  orchestration.
- `stages/stage*/{inputs,outputs,verify}.rs` owns stage-local proof inputs,
  typed outputs, and semantic checks.
- There is no standalone `public_inputs.rs`, `public_outputs.rs`,
  `opening_statement.rs`, or `transcript.rs`. If transcript labels become
  concrete, they should live next to the verifier or stage that consumes them.
  If a later Hyrax/final-check component becomes large enough to justify its
  own module, it can be factored out then; it should not be pre-split before
  the semantics exist.

The crate exposes a typed proof payload and implements the generic PCS-assist
boundary for the Dory PCS:

```rust
pub struct DoryAssistProof {
    pub dimensions: DoryAssistDimensions,
    pub stages: DoryAssistStageProofs,
    pub opening_proof: HyraxOpeningProof<Fq>,
    pub claims: DoryAssistProofClaims,
    pub dense_commitment: HyraxCommitment<GrumpkinPoint>,
    pub public_outputs: DoryAssistPublicOutputs,
}

pub struct DoryAssistStageProofs {
    pub stage1: stage1::Stage1Proof,
    pub stage2: stage2::Stage2Proof,
    pub stage3: stage3::Stage3Proof,
}

pub struct Stage1Proof {
    pub relations: Vec<Stage1RelationProof>,
}

pub struct Stage1RelationProof {
    pub id: DoryAssistRelationId,
    pub sumcheck: DoryAssistSumcheckSpec,
    pub sumcheck_proof: CompressedSumcheckProof<Fq>,
}

pub struct Stage2Proof {
    pub copy_constraints: Vec<DoryAssistCopyConstraint>,
}

pub struct Stage3Proof {
    pub packed_eval: Fq,
    pub reduced_openings: Vec<DoryAssistOpeningId>,
}

pub struct DoryAssistProofClaims {
    pub stage1: DoryAssistStage1Claims,
    pub opening: DoryAssistOpeningClaims,
}

pub struct DoryAssistStage1Claims {
    pub public: DoryAssistStage1PublicClaims,
    pub gt_exponentiation: DoryAssistGtExponentiationClaims,
    pub gt_exponentiation_digit_selector: DoryAssistGtExponentiationDigitSelectorClaims,
    pub gt_exponentiation_base_power: DoryAssistGtExponentiationBasePowerClaims,
    pub gt_exponentiation_digit_bitness: DoryAssistGtExponentiationDigitBitnessClaims,
    pub gt_exponentiation_shift: DoryAssistGtExponentiationShiftClaims,
    pub gt_exponentiation_boundary: DoryAssistGtExponentiationBoundaryClaims,
    pub gt_multiplication: DoryAssistGtMultiplicationClaims,
    pub g1: DoryAssistG1Claims,
    pub g2: DoryAssistG2Claims,
    pub miller_loop: DoryAssistMillerLoopClaims,
}

pub struct DoryAssistStage1PublicClaims {
    pub input: DoryAssistInputPublicClaims,
    pub gt_shift_eq_kernel: Fq,
    pub gt_exponentiation_boundary: DoryAssistGtExponentiationBoundaryPublicClaims,
    pub g1: DoryAssistG1PublicClaims,
    pub g2: DoryAssistG2PublicClaims,
    pub miller_loop: DoryAssistMillerLoopPublicClaims,
}

pub struct DoryAssistInputPublicClaims {
    pub checked_input_digest: Fq,
    pub verifier_setup_digest: Fq,
    pub dory_proof_artifacts: Vec<Fq>,
    pub jolt_commitments: Vec<Fq>,
    pub jolt_evaluation_claims: Vec<Fq>,
    pub transcript_scalars: Vec<Fq>,
}

pub struct DoryAssistGtExponentiationBoundaryPublicClaims {
    pub initial_selector: Fq,
    pub final_selector: Fq,
    pub initial_value: Fq,
    pub final_value: Fq,
}

pub struct DoryAssistGtExponentiationClaims {
    pub shifted_accumulator: Fq,
    pub accumulator: Fq,
    pub digit_selector: Fq,
    pub quotient: Fq,
    pub modulus: Fq,
}

pub struct DoryAssistGtExponentiationDigitSelectorClaims {
    pub digit_selector: Fq,
    pub digit_lo: Fq,
    pub digit_hi: Fq,
    pub base: Fq,
    pub base_squared: Fq,
    pub base_cubed: Fq,
}

pub struct DoryAssistGtExponentiationBasePowerClaims {
    pub base: Fq,
    pub base_squared: Fq,
    pub quotient_squared: Fq,
    pub modulus: Fq,
    pub base_cubed: Fq,
    pub quotient_cubed: Fq,
}

pub struct DoryAssistGtExponentiationDigitBitnessClaims {
    pub digit_lo: Fq,
    pub digit_hi: Fq,
}

pub struct DoryAssistGtExponentiationShiftClaims {
    pub accumulator: Fq,
}

pub struct DoryAssistGtExponentiationBoundaryClaims {
    pub accumulator: Fq,
    pub shifted_accumulator: Fq,
}

pub struct DoryAssistGtMultiplicationClaims {
    pub opening: DoryAssistGtMultiplicationOpeningClaims,
    pub rows: [DoryAssistGtMultiplicationRowClaims; 3],
}

pub struct DoryAssistGtMultiplicationOpeningClaims {
    pub left: Fq,
    pub right: Fq,
    pub output: Fq,
    pub quotient: Fq,
    pub modulus: Fq,
}

pub struct DoryAssistGtMultiplicationRowClaims {
    pub left: [Fq; 16],
    pub right: [Fq; 16],
    pub output: [Fq; 16],
    pub quotient: [Fq; 16],
}

pub struct DoryAssistG1Claims {
    pub scalar_multiplication: DoryAssistG1ScalarMultiplicationClaims,
    pub scalar_multiplication_shift: DoryAssistG1ScalarMultiplicationShiftClaims,
    pub scalar_multiplication_boundary: DoryAssistG1ScalarMultiplicationBoundaryClaims,
    pub addition: DoryAssistG1AdditionClaims,
}

pub struct DoryAssistG2Claims {
    pub scalar_multiplication: DoryAssistG2ScalarMultiplicationClaims,
    pub scalar_multiplication_shift: DoryAssistG2ScalarMultiplicationShiftClaims,
    pub scalar_multiplication_boundary: DoryAssistG2ScalarMultiplicationBoundaryClaims,
    pub addition: DoryAssistG2AdditionClaims,
}

pub struct DoryAssistG1PublicClaims {
    pub scalar_multiplication_boundary: DoryAssistG1ScalarMultiplicationBoundaryPublicClaims,
}

pub struct DoryAssistG2PublicClaims {
    pub scalar_multiplication_boundary: DoryAssistG2ScalarMultiplicationBoundaryPublicClaims,
}

pub struct DoryAssistOpeningClaim {
    pub id: DoryAssistOpeningId,
    pub value: Fq,
}

pub struct DoryAssistOpeningClaims {
    pub packed_point: Vec<Fq>,
    pub packed_eval: Fq,
}

pub struct DoryAssistPublicOutputs {
    pub pre_final_exponentiation: Bn254Fq12,
}

pub struct DoryAssist;

pub const DORY_ASSIST_HYRAX_GRUMPKIN_SETUP_SEED:
    GrumpkinPedersenSetupSeed<'static>;

pub type DoryAssistHyrax = HyraxScheme<Pedersen<GrumpkinPoint>>;

impl PcsProofAssist<DoryScheme> for DoryAssist {
    type Proof = DoryAssistProof;
    type Config = DoryAssistConfig;
    type Error = DoryAssistVerifierError;

    fn verify_clear<T>(
        config: &Self::Config,
        input: PcsAssistClearInput<'_, DoryScheme>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
    {
        // stage 1, stage 2, stage 3, Hyrax opening, native final check
    }

    fn verify_zk<T>(
        config: &Self::Config,
        input: PcsAssistZkInput<'_, DoryScheme>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<<DoryScheme as ZkOpeningScheme>::HidingCommitment, Self::Error>
    where
        T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
    {
        // stage 1, stage 2, stage 3, Hyrax opening, native final check,
        // hiding-commitment return
    }
}
```

`verifier.rs` normalizes the generic assist input into `CheckedInputs`, matching
the pattern used by `jolt-verifier`:

```rust
pub enum CheckedInputs<'a> {
    Clear(ClearInputs<'a>),
    Zk(ZkInputs<'a>),
}

pub struct ClearInputs<'a> {
    pub opening: ClearOpeningStatement<'a>,
}

pub struct ZkInputs<'a> {
    pub opening: ZkOpeningStatement<'a>,
}
```

The verifier flow is:

```text
PcsAssistClearInput / PcsAssistZkInput
  -> CheckedInputs::{Clear,Zk}
  -> absorb Dory-assist checked-input preamble into the continued Fr transcript
  -> derive typed input public claims from the continued Fr transcript
  -> squeeze Fr transcript challenge, inject Fr -> Fq, and compare against
     proof.claims.stage1.public.input.checked_input_digest
  -> stage1::verify
  -> stage2::verify
  -> stage3::verify
  -> packed Hyrax dense-witness opening
  -> clear mode: native final-exponentiation/public pairing check
  -> ZK mode: bind the public Miller-loop output; scalar-product final equation
     and hiding-commitment return are still explicit construction work
```

Checked input handling does the following:

- validates obvious shape only: the Dory opening point length must not exceed
  the Dory proof round cap used by the implementation (`64`);
- absorbs a deterministic Dory-assist checked-input preamble into the continued
  Jolt `Fr` transcript before stage verification;
- derives typed Stage 1 input-public claims for `VerifierSetupDigest`,
  `VerifierSetupArtifact(i)`, `DoryProofArtifact(0)`, `JoltCommitment(0)`,
  `JoltEvaluationClaim(0)` in clear mode, and `TranscriptScalar(i)` for each
  opening-point coordinate followed by the Dory verifier transcript scalars
  needed by the reduce/final semantics;
- immediately squeezes `Label("dory_assist_checked_input_digest")` from that
  continued `Fr` transcript, injects the scalar into `Fq`, and checks it against
  `proof.claims.stage1.public.input.checked_input_digest`.

The setup/proof/commitment digests are sampled from forked copies of the
continued `Fr` transcript after the corresponding artifact has been absorbed:
`Label("dory_assist_setup_digest")`,
`Label("dory_assist_proof_digest")`, and
`Label("dory_assist_commitment_digest")`. Forking makes these public values
available to Dory-assist relations without inserting extra challenges into the
main stage transcript. Opening-point coordinates, clear evaluations, and Dory
verifier transcript scalars are injected directly from `Fr` to `Fq`. Any
inverse, product, square, or fold factor used by the Dory verifier is computed
in native `Fr` first and then injected into `Fq`; Dory assist does not recompute
those scalar-arithmetic facts in `Fq`. The verifier boundary still enforces the
native Dory scalar consistency before absorption: for every reduce round,
`beta * beta^{-1} = 1`, `alpha * alpha^{-1} = 1`,
`alpha_beta = alpha * beta`, and
`alpha_inverse_beta_inverse = alpha^{-1} * beta^{-1}`; after reduction,
`gamma * gamma^{-1} = 1`, `d * d^{-1} = 1`, and `d_squared = d^2`. It also
checks the reduce-round fold factors against the checked opening point using
the native Dory coordinate order:
`s1_fold_factor = alpha * (1 - s1_coord) + s1_coord` and
`s2_fold_factor = alpha^{-1} * (1 - s2_coord) + s2_coord`. This
matches native Dory verifier behavior, which rejects non-invertible transcript
challenges. The same validated scalar replay is then used both for
`TranscriptScalar` public-claim absorption and for the native final pairing
check. The `TranscriptScalar` layout is:

```text
0 .. point_len - 1                         opening-point coordinates
base_r = point_len + 8*r
base_r + 0                                  reduce round r beta
base_r + 1                                  reduce round r beta^{-1}
base_r + 2                                  reduce round r alpha
base_r + 3                                  reduce round r alpha^{-1}
base_r + 4                                  reduce round r alpha * beta
base_r + 5                                  reduce round r alpha^{-1} * beta^{-1}
base_r + 6                                  reduce round r s1 fold factor
base_r + 7                                  reduce round r s2 fold factor
final_base = point_len + 8*reduce_rounds
final_base + 0                              gamma
final_base + 1                              gamma^{-1}
final_base + 2                              sigma_c when present
final_base + 2 + zk_sigma                   final d
final_base + 3 + zk_sigma                   final d^{-1}
final_base + 4 + zk_sigma                   final d^2
```

where `zk_sigma = 0` in transparent mode and `zk_sigma = 1` when the optional
ZK scalar-product `sigma_c` challenge is present.

Concrete Dory and Jolt PCS artifacts are also staged as public Fq coordinates
under the same typed input-public claim:

```text
DoryProofArtifact:
  0                  full Dory proof digest
  1..16              VMV C as padded GT coefficients
  17..32             VMV D2 as padded GT coefficients
  33..35             VMV E1 as G1 affine x, y, infinity
  36..40             ZK E2 as G2 affine x0, x1, y0, y1, infinity
                      or the identity tuple when absent
  41..43             ZK y_com as G1 affine x, y, infinity
                      or the identity tuple when absent
  44..59             ZK scalar-product p1 as padded GT coefficients
                      or the GT identity when absent
  60..75             ZK scalar-product p2 as padded GT coefficients
                      or the GT identity when absent
  76..91             ZK scalar-product q as padded GT coefficients
                      or the GT identity when absent
  92..107            ZK scalar-product r as padded GT coefficients
                      or the GT identity when absent
  108..110           ZK scalar-product e1 as G1 affine x, y, infinity
                      or the identity tuple when absent
  111..115           ZK scalar-product e2 as G2 affine x0, x1, y0, y1, infinity
                      or the identity tuple when absent
  116                ZK scalar-product r1 injected Fr -> Fq, or zero when absent
  117                ZK scalar-product r2 injected Fr -> Fq, or zero when absent
  118                ZK scalar-product r3 injected Fr -> Fq, or zero when absent
  for each reduce round, starting at 119:
    first message:
      D1_left, D1_right, D2_left, D2_right as padded GT coefficients
      E1_beta as G1 affine x, y, infinity
      E2_beta as G2 affine x0, x1, y0, y1, infinity
    second message:
      C_plus, C_minus as padded GT coefficients
      E1_plus, E1_minus as G1 affine x, y, infinity
      E2_plus, E2_minus as G2 affine x0, x1, y0, y1, infinity
  after all reduce rounds:
    final E1 as G1 affine x, y, infinity
    final E2 as G2 affine x0, x1, y0, y1, infinity

JoltCommitment:
  0                  joint commitment digest
  1..16              joint commitment as padded GT coefficients
```

The padded GT layout is the 12 BN254 Fq12 coefficients followed by four zero
slots, matching the GT algebraic relation layout. G1/G2 public coordinates use
an explicit infinity flag so later curve relations can consume these public
values without relying on an exceptional affine encoding.
`jolt-claims::protocols::dory_assist::formulas::artifacts` is the semantic
source of truth for these offsets and range widths; the verifier artifact
helpers re-export those constants and verifier tests assert that the staged
proof artifacts match that named layout rather than raw numeric offsets.
The checked-input validator enforces proof mode at this boundary: clear assist
inputs must carry transparent Dory proofs with no ZK/sigma/scalar-product
artifacts, while ZK assist inputs must carry Dory ZK, sigma, and scalar-product
proof artifacts. This prevents a clear assist input from relying on a blinded
Dory proof whose verifier path would otherwise ignore the explicit clear
evaluation.
The verifier tests also pin the ZK checked-input layout: `DoryReduceInitialE2`
comes from the Dory ZK `e2` artifact, `y_com` and scalar-product artifacts
occupy the fixed public-artifact slots above, and `sigma_c` shifts the final
`d`, `d^{-1}`, and `d^2` `TranscriptScalar` indices exactly as specified.
The active soundness harness exercises those bindings on the ZK multi-round
path by tampering `e2`, `y_com`, scalar-product artifacts, reduce-round
artifacts, reduce dimensions, and the `sigma_c` transcript-scalar slot.
The verifier setup is staged separately under `VerifierSetupArtifact(i)`:

```text
VerifierSetupArtifact:
  chi[0..=max_rounds]                       padded GT coefficients
  delta_1l[0..=max_rounds]                  padded GT coefficients
  delta_1r[0..=max_rounds]                  padded GT coefficients
  delta_2l[0..=max_rounds]                  padded GT coefficients
  delta_2r[0..=max_rounds]                  padded GT coefficients
  g1_0                                      G1 affine x, y, infinity
  g2_0                                      G2 affine x0, x1, y0, y1, infinity
  h1                                        G1 affine x, y, infinity
  h2                                        G2 affine x0, x1, y0, y1, infinity
  ht                                        padded GT coefficients
```

`jolt-claims::protocols::dory_assist::formulas::setup_artifacts` is the
semantic source of truth for these verifier-setup offsets. The Dory-reduce
semantic catalog binds the active round's `chi`, `delta_1l`, `delta_1r`,
`delta_2l`, and `delta_2r` constants from these public slots into the reduce
transition witness columns by direct copy constraints.
The verifier additionally checks that the opening point length equals
`proof.nu + proof.sigma`, and that the Dory proof carries exactly `sigma`
first-reduce and `sigma` second-reduce messages before staging the reduce-round
and final scalar-product artifacts.

The preamble is:

```text
Label("DoryAssist")
Label("checked_inputs")

clear mode:
  Label("clear")
  Label("dory_assist_setup")
  Dory verifier setup
  Label("dory_assist_pcs_proof")
  full Dory PCS proof
  Label("dory_assist_commitment")
  Dory commitment
  LabelWithCount("dory_assist_point", point.len())
  point coordinates
  Label("dory_assist_eval")
  clear evaluation

ZK mode:
  Label("zk")
  Label("dory_assist_setup")
  Dory verifier setup
  Label("dory_assist_pcs_proof")
  full Dory PCS proof
  Label("dory_assist_commitment")
  Dory commitment
  LabelWithCount("dory_assist_point", point.len())
  point coordinates
```

This preamble makes transcript order deterministic and sensitive to the opening
statement values that are already available at the generic PCS-assist boundary,
the Dory verifier setup, and the full Dory PCS proof. The checked-input digest
then makes the proof payload context-bound to that continued transcript state:
changing the setup, Dory PCS proof, Jolt commitment, opening point, or clear
evaluation changes the expected digest before any algebraic stage runs. This is
not just a digest check: Stage 2 already binds
`DoryProofArtifact(DORY_VMV_C_START)` to the first coefficient of the
`GtExponentiation.ExpAccumulator` witness endpoint, and binds the VMV `E1`
affine `x,y` coordinates to the Miller-loop G1 evaluation-point witness
endpoint. The remaining artifact coordinates are input-bound here and will be
connected to the Dory verifier trace by later algebraic/copy/opening stages as
those trace columns are enabled.

Each current stage also binds its local payload into the continued `Fr`
transcript and exposes the sampled challenge after Fr-to-Fq injection:

```text
stage 1:
  Label("dory_assist_stage1")
  mode label
  Label("stage1_relations")
  stage1 relation count
  for each declared relation:
    relation ID
    sumcheck domain
    sumcheck rounds
    sumcheck degree
    relation-local semantic challenges are squeezed from the continued transcript
    input claim is computed from typed proof claims and `jolt-claims`
    compressed Boolean sumcheck round polynomials
    referenced opening-claim values are appended as `opening_claim`
  squeeze Label("dory_stage1_challenge") -> Fq

stage 2:
  Label("dory_assist_stage2")
  mode label
  Label("stage1_relations")
  stage1 relation count
  Label("stage2_copy_constraints")
  stage2 copy-constraint count
  for each canonical copy constraint:
    copy-constraint index
    resolved source endpoint value
    resolved target endpoint value
  squeeze Label("dory_stage2_challenge") -> Fq

stage 3:
  Label("dory_assist_stage3")
  mode label
  stage 1/2 relation counts
  stage3.reduced_openings count
  stage3.packed_eval
  packed opening claim point/eval
  Hyrax opening proof
  dense Hyrax commitment
  public pre-final-exponentiation GT output
  squeeze Label("dory_stage3_challenge") -> Fq
```

These stage challenges are verifier organization for the concrete assist proof:
they are deterministic, depend on the checked-input preamble and stage payloads,
and will be consumed by the algebraic/copy-constraint sumcheck stages as those
checks land.

The current verifier layer rejects malformed stage payloads before full
sumcheck semantics are attempted, and Stage 3 already verifies the packed
Hyrax opening:

```text
stage 1:
  proof Dory-reduce dimensions must match the checked PCS proof point length
  and reduce-round count before the proof dimensions drive any stage catalog
  verifier setup artifact vectors must be internally consistent and support at
  least the proof's Dory reduce-round count, so setup-indexed reducer checks
  fail at the checked-input boundary instead of later stage/native-final lookup
  until dynamic trace-sized catalogs are wired, active GT/G1/G2/Miller-loop,
  and wiring dimensions must equal the verifier's canonical supported shape
  prefix-packing dimensions must equal the minimal dimensions derived from the
  canonical packing catalog, so the proof cannot select a weaker packed-opening
  shape
  stage1.relations must be nonempty
  stage1.relations must equal the canonical Stage 1 relation catalog derived
  from `jolt_claims::protocols::dory_assist::formulas::protocol_claims` and
  the public Dory-assist dimensions
  each relation's input claim is computed from the relation input expression
  over typed proof claims, sampled challenges, and public values
  GT shift-kernel, GT boundary publics, G1/G2 scalar-mul boundary publics,
  Miller-loop selectors/constants, Miller-loop shift kernels, Miller-loop
  boundary selectors/values, and MillerLoopOutputGt coefficients are read from
  typed Stage 1 public claim fields; DoryReduceInitialE2 is checked as part of
  the checked-input public claims; current fixtures still use synthetic
  zero-boundary and zero-line values until prover/runtime tracing supplies
  verifier-derived evaluations
  each relation's compressed Boolean sumcheck proof is verified using
  `jolt-sumcheck` proof encoding and Fr-to-Fq injected round challenges
  each relation's output expression is evaluated from `jolt-claims`, and the
  sumcheck final claim must match that expected output claim
  Dory-reduce GT/G1/G2 transition relations check the batched per-round linear
  state updates, and DoryReduceScalarFold checks the batched scalar accumulator
  folds
  for multi-round Dory-reduce dimensions, `DoryReduceStateChain` checks
  next-row/current-row reducer chaining and `DoryReduceBoundary` checks initial
  reducer inputs plus final `NativeFinalCheckInput` bindings
  each declared `DoryAssistConsistencyClaim` is evaluated over the same typed
  opening/public/challenge resolvers, and equal-expression consistency must
  hold before the relation's openings are accepted
  each accepted relation records input claim, relation challenges, sumcheck
  point, final claim, expected output claim, and referenced opening claims in
  `Stage1Output` for later opening/packing discharge
  stage1 transcript payload is absorbed and an Fq challenge is returned

stage 2:
  stage2.copy_constraints must be nonempty
  stage2.copy_constraints must equal the canonical active direct-copy stencil
  from `jolt-claims`: GT-local copy edges, G1/G2 scalar-mul shift and boundary
  copy edges, plus the Miller-loop copy edges for line-step, line-evaluation,
  pair-product, accumulator, boundary composition, and quotient-column wiring
  into row-aware GT multiplication rows; singleton-round Dory-reduce transition
  copies bind Dory proof reduce-message artifacts, verifier setup chi/delta
  artifacts, and Fr-derived transcript scalars into GT/G1/G2 transition
  openings, and scalar-fold copies bind `s1_fold_factor` and `s2_fold_factor`
  to transcript scalars
  Dory-reduce public folds evaluate round-indexed proof artifacts, setup
  artifacts, and transcript-scalar vectors at each Dory-reduce relation's
  sumcheck point and compare the folded value with the corresponding witness
  opening
  each witness endpoint must have been recorded as a verified Stage 1 opening
  claim
  each copy endpoint resolves through typed Stage 1 opening/public claims
  each direct copy edge must have equal resolved source and target values
  stage2 transcript payload is absorbed and an Fq challenge is returned

stage 3 / packed opening metadata:
  opening_proof.combined_row must be nonempty
  claims.opening.packed_point must be nonempty
  dense_commitment.rows must be nonempty
  stage3.reduced_openings must be nonempty and must match the canonical order
  of verified Stage 1 opening claims
  stage3.packed_eval must equal claims.opening.packed_eval
  dense_commitment.rows.len() determines row_vars and must be a power of two
  opening_proof.combined_row.len() determines col_vars and must be a power of two
  claims.opening.packed_point.len() must equal row_vars + col_vars
  stage3.packed_eval must equal the equality-polynomial fold of the reduced
  Stage 1 opening claims over claims.opening.packed_point
  stage3 transcript payload is absorbed and an Fq challenge is returned
  the seed-derived Grumpkin Hyrax verifier setup must accept the packed opening

native output:
  public_outputs.pre_final_exponentiation is a raw BN254 Fq12 Miller-loop
  output, not a target-subgroup `Bn254GT`; it is expanded into the 16-slot Fq12
  coefficient layout and must equal the Stage 1 public `MillerLoopOutputGt`
  coefficients
  the verifier will final-exponentiate this raw value natively and compare it
  to the Dory final-check RHS assembled from the reducer final state, final
  Dory artifacts, setup artifacts, and final transcript scalars
```

Stage 1 now binds the declared semantic relation catalog for the full GT,
G1, G2, Miller-loop, and Dory-reduce transition building blocks. It verifies
that each relation carries a well-formed compressed Boolean sumcheck transcript
whose final claim matches the `jolt-claims` output expression. The current
catalog is, in canonical protocol order: `GtExponentiation`,
`GtExponentiationDigitSelector`,
`GtExponentiationBasePower`, `GtExponentiationDigitBitness`,
`GtExponentiationShift`, `GtExponentiationBoundary`, `GtMultiplication`,
`G1ScalarMultiplication`, `G1ScalarMultiplicationShift`,
`G1ScalarMultiplicationBoundary`, `G1Addition`, `G2ScalarMultiplication`,
`G2ScalarMultiplicationShift`, `G2ScalarMultiplicationBoundary`, `G2Addition`,
`MillerLoopLineStep`, `MillerLoopLineEvaluation`, `MillerLoopPairProduct`,
`MillerLoopAccumulator`, `MillerLoopBoundary`, `DoryReduceGtTransition`,
`DoryReduceG1Transition`, `DoryReduceG2Transition`, and
`DoryReduceScalarFold`; for multi-round Dory-reduce dimensions the verifier's
canonical Stage 1 catalog additionally appends `DoryReduceStateChain` and
`DoryReduceBoundary`. The
input and output claims are computed from typed
proof claims, matching the `jolt-verifier` pattern; they are not trusted as
fields inside the stage proof. The relation opening claims are recorded,
deduplicated in canonical Stage 1 order, and bound by Stage 3 to the single
Grumpkin/Pedersen Hyrax dense opening for the currently implemented GT, G1, G2,
Miller-loop, and Dory-reduce blocks. Stage 1 also enforces any
`DoryAssistConsistencyClaim` metadata carried by a semantic relation; the active
Dory catalog currently relies on Stage 2 direct copy constraints for
cross-relation equalities, so this is verifier infrastructure for future
relation-local consistency metadata rather than a replacement for copy edges.
The GT multiplication proof claims are row-aware and component-aware for the
three Miller-loop GT multiplication rows: pair-product multiply, accumulator
square, and accumulator multiply-by-line. Stage 2 now checks the public VMV
C-to-GT-accumulator copy edge, the public VMV E1-to-Miller-loop-G1 copy edges,
the active GT-local direct-copy constraints, the G1/G2 scalar-mul shift and
boundary copy stencils, and the Miller-loop copy stencil:
shifted G2 line state to next line state, line-step coefficients to
line-evaluation coefficients, line-evaluation and pair-product values into
shared GT multiplication rows, accumulator values into shared GT multiplication
rows, GT outputs back into pair-product/accumulator columns, quotient columns
into row-aware GT multiplication quotient slots, accumulator columns into
boundary columns, boundary shifted output into `MillerLoopOutputGt`, and
singleton-round Dory-reduce proof-message, verifier-setup, and transition
transcript-scalar inputs into the GT/G1/G2 reduce transition openings.
For multi-round Dory-reduce shapes, the verifier omits direct row-indexed copy
edges and instead uses public folds for public vectors, `DoryReduceStateChain`
for next-row/current-row chaining, `DoryReduceBoundary` for initial/final
reducer binding, and Stage 3 for the single packed Hyrax opening over all
recorded openings. The verifier currently has end-to-end synthetic fixtures for
this path; prover/runtime tracing must still materialize the same witness
values from a canonical Dory verifier trace.
Stage 3 packs the reduced opening claims for those active Stage 1 relations,
including quotient columns recorded as auxiliary relation openings. The
verifier also binds the public pre-final-exponentiation `Bn254Fq12` output to
the Stage 1 Miller-loop output coefficients. This type boundary is important:
`Bn254GT` remains the post-final-exponentiation target subgroup type, while the
assist SNARK exposes the raw Fq12 Miller-loop value so the native verifier can
perform the cheap final exponentiation and equality check.
`jolt-crypto` exposes this boundary as `Bn254::multi_miller_loop(...) ->
Bn254Fq12` plus `Bn254Fq12::final_exponentiation() -> Option<Bn254GT>`.
`jolt-dory-assist-verifier::native_final` reconstructs the transparent Dory
final equation from the checked opening statement, the Dory proof artifacts,
the verifier setup artifacts, and the replayed transcript scalars:

```text
Pair 1: e(E1_final + d*Gamma10, E2_final + d^{-1}*Gamma20)
Pair 2: e(H1, -gamma * (E2_acc + d^{-1}*s1_acc*Gamma20))
Pair 3: e(-gamma^{-1} * (E1_acc + d*s2_acc*Gamma10), H2)
Pair 4: e(d^2 * E1_init, Gamma20)

RHS =
  C_acc + (s1_acc*s2_acc)*HT + chi_0
  + d*D2_acc + d^{-1}*D1_acc + d^2*D2_init
```

The native transparent final test uses a real Dory proof to assert that final
exponentiating the raw four-term Miller-loop product equals this RHS, matching
the upstream Dory verifier equation while keeping final exponentiation outside
the assist SNARK.

The verifier exposes the reducer state needed by the native final check through
`NativeFinalCheckInput(i)` public claims. The vector currently contains
`C_acc`, `D1_acc`, `D2_acc`, `E1_acc`, `E2_acc`, `E1_init`, `D2_init`,
`s1_acc`, and `s2_acc`, encoded as padded Fq coordinate/coefficient slots with
Fr scalars injected into Fq. The native final verifier decodes these public
claims back into BN254 wrappers, rejects invalid point/GT encodings and
non-canonical Fr injections, and assembles the final RHS from the bound claims.
Native Dory-reducer replay remains only as a fixture/test oracle.

For ZK Dory openings the native-final helper reconstructs the upstream
one-pairing scalar-product equation instead. The reducer state is decoded from
the same `NativeFinalCheckInput` vector, and the scalar-product proof artifacts
plus `sigma_c` define:

```text
Pair 1:
  e(sp.e1 + d*Gamma10, sp.e2 + d^{-1}*Gamma20)

RHS =
  chi_0 + sp.r + sigma_c*sp.q + sigma_c^2*C_acc
  + d*(sp.p2 + sigma_c*D2_acc)
  + d^{-1}*(sp.p1 + sigma_c*D1_acc)
  - (sp.r3 + d*sp.r2 + d^{-1}*sp.r1)*HT
```

The verifier has a native test with a real ZK Dory proof asserting that final
exponentiating the raw one-term Miller-loop product equals this RHS. The
top-level ZK assist verifier now binds the same public raw Miller-loop output
to `MillerLoopOutputGt`, checks its final exponentiation against the
scalar-product RHS, and returns the Dory `y_com` hiding commitment.

The proof fields should remain typed by stage and operation family. Production
code should not route claims through an untyped opening map.

### Error Taxonomy

`jolt-dory-assist-verifier` should use explicit semantic errors rather than
treating all failures as an opaque verifier failure. The current taxonomy is:

```rust
pub enum DoryAssistStage {
    CheckedInputs,
    Stage1,
    Stage2,
    Stage3,
    HyraxOpening,
    NativeOutput,
}

pub enum DoryAssistVerifierError {
    InvalidMode { expected: &'static str, got: &'static str },
    InvalidProofShape { component: &'static str, reason: String },
    CheckedInputMismatch { reason: String },
    StageClaimMismatch { stage: DoryAssistStage, reason: String },
    MissingOpeningClaim { id: DoryAssistOpeningId },
    MissingStageClaimChallenge { id: DoryAssistChallengeId },
    MissingStageClaimPublic { id: DoryAssistPublicId },
    StageSumcheckFailed {
        stage: DoryAssistStage,
        relation: DoryAssistRelationId,
        reason: String,
    },
    StageOutputMismatch { stage: DoryAssistStage, reason: String },
    OpeningClaimMismatch { reason: String },
    HyraxOpeningFailed(jolt_hyrax::HyraxError),
    PublicOutputMismatch { reason: String },
    TranscriptMismatch { reason: String },
}
```

Verifier failures should use the most specific concrete variant so tamper tests
can distinguish semantic rejection from an opaque verifier failure. The active
verifier no longer carries a temporary `Unimplemented` rejection marker;
soundness tests now treat any accepted tampered proof as a failure.

## Dory-Assist Verifier Testing

The verifier crate owns a rigorous, Jolt-style test harness before the prover
side is fully implemented. The goal is to make verifier semantics the testing
oracle for prover development.

The test tree mirrors `jolt-verifier`:

```text
tests/
  generic_boundary.rs
  completeness.rs
  completeness/
    cases.rs
    fixtures.rs
    oracle.rs
  soundness.rs
  soundness/
    fixtures.rs
    tampering/
      manifest.rs
      inputs.rs
      stages.rs
      openings.rs
      public_outputs.rs
  support/
    mod.rs
```

Testing policy:

- Each semantic slice must add at least one valid completeness fixture and one
  targeted soundness/tamper case.
- Active soundness assertions must reject with a concrete verifier error.
- Completeness oracle tests and tamper tests are active for the implemented
  verifier semantics. Any future deferred verifier semantic must be listed in
  the manifest with a concrete coverage note before its active rejection test is
  added.
- As each verifier stage lands, add or tighten the relevant active tamper tests
  and update the manifest.
- When prover-side Dory assist is implemented, synthetic fixture fields should
  be replaced by prover-generated fixtures while preserving the same verifier
  oracle API.
- The `generic_boundary.rs` harness must exercise the concrete `DoryAssist`
  implementation through `jolt_verifier::PcsProofAssist<DoryScheme>` for clear
  and ZK fixtures, including at least one tamper rejection per mode.
- The same boundary harness must include Poseidon-over-`Fr` transcript fixtures:
  fixtures generated against Poseidon must accept through the trait path, while
  fixtures generated against a different transcript backend must reject under
  Poseidon. This keeps the Dory-assist `Fr -> Fq` challenge injection explicit
  and prevents an accidental hidden `Fq` transcript.

Initial tamper coverage targets:

```text
checked inputs:
  clear opening eval
  clear opening point
  ZK opening point
  checked-input digest public claim
  verifier setup digest public claim
  verifier setup artifact public claim
  Dory proof artifact digest public claim
  Dory VMV C GT artifact public claim
  Dory VMV E1 G1 artifact public claim
  Dory ZK artifact public claim
  ZK multi-round Dory e2, y_com, and scalar-product artifact public claims
  Dory reduce-round artifact public claim
  ZK multi-round Dory reduce-round artifact public claim
  Dory-reduce dimension binding for clear and ZK multi-round paths
  Dory final scalar-product artifact public claim
  Jolt commitment digest public claim
  concrete joint commitment GT artifact public claim
  clear Jolt evaluation public claim
  DoryReduceInitialE2 checked public claim
  transcript scalar public claim
  ZK scalar-product sigma_c transcript scalar public claim

stage payloads:
  stage1 payload
  stage1 compressed sumcheck shape
  stage1 relation output expression
  stage1 GT digit-selector, shift, boundary, and multiplication outputs
  stage1 GT shift-kernel and boundary public claims
  stage1 G1/G2 scalar-mul, shift, boundary, and addition outputs
  stage1 G1/G2 scalar-mul boundary public claims
  stage1 Miller-loop line-step, sparse line-evaluation, pair-product,
  accumulator, and boundary outputs
  stage1 Dory-reduce GT/G1/G2 transition outputs
  stage1 Dory-reduce scalar-fold outputs
  stage1 Dory-reduce state-chain and boundary outputs for clear and ZK
  multi-round shapes
  stage2 payload
  stage2 Dory VMV C public artifact to GT accumulator copy edge
  stage2 Dory VMV E1 public artifact to Miller-loop G1 evaluation-point copy
  edges
  stage2 GT direct copy-edge values
  stage2 G1/G2 scalar-mul shift and boundary copy-edge values
  stage2 Miller-loop line, pair-product, accumulator, and boundary copy-edge
  values, including pair-product and accumulator quotient copy edges
  stage2 Dory-reduce proof-message artifact copy-edge values
  stage2 Dory-reduce verifier setup artifact copy-edge values
  stage2 Dory-reduce transition transcript-scalar copy-edge values
  stage2 Dory-reduce scalar-fold transcript-scalar copy-edge values
  stage2 Dory-reduce public-fold values for clear and ZK multi-round shapes
  stage3 payload
  stage3 reduced opening order

packed opening:
  packed opening claim point
  packed opening claim eval
  Hyrax combined row
  Hyrax row-opening scalar
  dense witness commitment

public output:
  pre-final-exponentiation GT output
```

Enable order should follow the implementation order:

```text
checked input binding
  -> enable input tamper tests

stage 1 algebraic relations
  -> active for GT, G1, G2, Miller-loop, and Dory-reduce relation-specific
     tamper tests

stage 2 copy constraints
  -> active for the GT-local direct-copy stencil, row-aware GT multiplication
     rows, and the Miller-loop copy stencil including quotient-column wiring

stage 3 packing / Hyrax opening
  -> active for current Stage 1 reduced claims, packed claim, Hyrax proof, and
     dense-commitment tamper tests

native public output binding
  -> active for pre-final-exponentiation output tamper tests

prover-side implementation
  -> enable completeness oracle tests with prover-generated fixtures
```

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
   - Mirror `jolt-verifier`: `lib.rs` as exports, `proof.rs` for serialized
     proof-owned data, `verifier.rs` for `DoryAssist`, `CheckedInputs`, and
     stage orchestration, and `stages/stage*/{inputs,outputs,verify}.rs` for
     semantic checks.
   - Keep claims and public outputs in `proof.rs` until they become large
     enough to justify a split.
   - Do not add speculative `public_inputs`, `public_outputs`, `transcript`,
     `opening_statement`, `hyrax_opening`, or `final_check` modules before the
     implementation needs them.
   - Review gate: proof-shape and stage-orchestration tests pass, and the
     verifier crate has completeness/soundness registries plus a tamper
     manifest that documents every deferred semantic check.

7. Implement the generic PCS-assist boundary for Dory.
   - Implement `PcsProofAssist<DoryScheme>` for the
     `jolt_dory_assist_verifier::DoryAssist` implementation type.
   - Normalize the generic reduced opening statement and ordinary joint opening
     proof into `CheckedInputs::{Clear,Zk}`.
   - Review gate: `jolt-verifier` can dispatch to Dory-assist clear and ZK
     fixtures without Dory-specific stage modules; accepted tampered assist
     proofs fail the soundness harness.

8. Add Miller-loop verification path.
   - Prove Miller-loop witness constraints in Dory assist.
   - Keep final exponentiation as native Dory-assist verifier work.
   - Current verifier gate: the raw Miller-loop public output is bound to the
     Stage 1 `MillerLoopOutputGt` coefficients; transparent fixtures final
     exponentiate that raw value natively and compare it to the reconstructed
     Dory final RHS. The reducer-state inputs for that RHS are also exposed as
     `NativeFinalCheckInput` public claims; the native verifier decodes those
     claims directly, with reducer replay retained only in fixtures/tests.
   - Remaining review gate: fixtures match the Quang reference branch for
     equivalent inputs, valid prover-generated fixtures are accepted, and
     targeted tamper tests reject for each newly implemented semantic binding.

9. Add wrapper R1CS hooks.
   - Lower Dory-assist stages through claim, sumcheck, packing, and Hyrax R1CS
     helpers.
   - Review gate: wrapper assembly can include configured Dory-assist stages in
     a satisfied R1CS.
