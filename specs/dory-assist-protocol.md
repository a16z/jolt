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
optional PCS-assist payload, e.g. `Option<T>` where `T: PcsProofAssist<PCS>`.
The verifier config decides whether such a payload is required and which assist
implementation is selected. Dory-specific staging does not live in
`jolt-verifier`; it lives in a `dory-assist-verifier` crate that implements the
generic PCS-assist interface for the Dory PCS.

This spec owns the Dory-assist protocol semantics in `jolt-claims`, the
`dory-assist-verifier` crate, and the Hyrax commitment layer used by that proof.
Composition with field inline, wrapper proving, ZK, and proof configuration is
specified in
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
`dory-assist-verifier` crate implementing PcsProofAssist for the Dory PCS
generic PCS-assist payload consumed by jolt-verifier
Hyrax dense-witness opening over Grumpkin-backed Pedersen row commitments
multi-Miller-loop work proven inside the assist proof
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
  Option<T: PcsProofAssist<PCS>> proof payload slot
  proof/config shape validation
  opening snapshot construction
  dispatch to the selected assist implementation

jolt-claims:
  Dory-assist protocol semantics
  relation IDs, public IDs, opening IDs, dimensions, and claim formulas
  operation-family, wiring, and packing constraints

dory-assist-verifier:
  Dory-specific staged verifier organization
  native verifier API mirroring jolt-verifier for Dory-assist semantics
  wrapper-facing R1CS hook for the selected Dory-assist verifier path
  proof payload type implementing PcsProofAssist for the Dory PCS
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
    multi-Miller-loop witnesses
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
Jolt opening snapshot:
  commitments
  opening points
  claimed evaluations
  transcript-derived scalars
Dory-assist transcript challenges
Grumpkin/Pedersen-backed Hyrax dense-witness commitment
Hyrax opening claim
final pairing-check input/output values
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
  multi-Miller-loop rows
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
multi-Miller-loop / pre-final-exponentiation value. Final exponentiation and
pairing equality remain native verifier work in v1.

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

pairing boundary:
  multi-Miller-loop constraints in the assist proof
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

`rho_shift` is not an independent committed object; it is connected to `rho`
by the GT shift sum-check. If the exponentiation zero-check emits the claim
`rho_shift(r_s, r_u, r_c)`, the shift relation proves:

```text
rho_shift(r_s, r_u, r_c)
  = sum_{s,u,c} EqPlusOne(r_s, s) * eq(r_u, u) * eq(r_c, c) * rho(s, u, c)
```

The final shift-round claim is therefore a public shift-kernel value times a
fresh opening of `rho` under the `GtExponentiationShift` relation. Boundary
checks assert `rho(0) = 1_GT` and `rho(n) = h`, where `h` is the public
exponentiation output used by the Dory verifier operation DAG. Base powers for
the digit selector are part of the GT operation-family witness/public-input
interface and are checked with the same polynomial-division semantics.

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

the relation is the random linear combination of the 27 appendix constraints:

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
`dy = y_P - y_T`, the seven local constraints are:

```text
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
shift of `A`.

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
components. This yields 47 constraints: three scalar infinity booleanity
constraints, twelve infinity-encoding constraints, five `P = O` identity
constraints, five `Q = O` identity constraints, two branch booleanity
constraints, two branch-selection constraints for `mu*dx = 1`, four doubling
enforcement constraints, four inverse enforcement constraints, four split
slope constraints, two output-indicator constraints, and four output-coordinate
constraints.

G2 scalar multiplication is the split-coordinate version of the G1 recurrence.
The two denominator-free doubling equations and two conditional-add equations
each produce `c0` and `c1` constraints, followed by the scalar infinity
constraints:

```text
iota_A*(1-iota_T)
iota_T*x_T0
iota_T*x_T1
iota_T*y_T0
iota_T*y_T1
```

The scalar-mul shift relation separately batches the four shifted accumulator
components `(x_A0', x_A1', y_A0', y_A1')` against `(x_A0, x_A1, y_A0, y_A1)`.

The formulas should expose local constraints, virtual-polynomial dependencies,
opening IDs, public IDs, challenge IDs, and relation claim expressions. They
should not expose runtime tracing mechanics.

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
`i`th polynomial. This is the stage-3 claim-reduction step; it is not another
sum-check.

Family packing should be used before prefix packing where the paper does:
witness polynomials of the same type across operation instances share a single
family-packed polynomial whose domain is extended by a family-local
constraint-index suffix.

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
    multi-Miller-loop constraints
    declared wiring and public-input constraints

stage 3:
  prefix packing reduction to one dense polynomial opening

PCS opening:
  Hyrax opening of the dense witness using Grumpkin-backed Pedersen row
  commitments
```

Dory assist proves the multi-Miller-loop work. The Dory-assist verifier receives
the resulting public GT value, computes final exponentiation directly, and
checks the public pairing equality. Final exponentiation is cheap deterministic
verifier work and stays native in v1, but it is still Dory-specific verifier
logic owned by the Dory-assist verifier implementation rather than by
`jolt-verifier`.

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
  WitnessValue { family, row, column, component }
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
Enabled(i)    = 1 for real copy edges, 0 for padding
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

`Src` and `Dst` are virtual polynomials. They must be derived from typed
`ValueRef`s into public inputs, transcript challenges, constants, or
operation-family witness columns. Stage 2 owns the copy zero-check claim; stage
3 and prefix packing bind the resulting witness-column claims to the single
dense Hyrax opening.

The paper's full wiring view indexes both a port domain `x` and a padded
family-local constraint domain `c`:

```text
sum_x sum_c eq(r_x, x) *
  sum_e lambda_e * (
      beta_src(e) * eq(c^src(e), idx_src(e)) * V_src(e)(x)
    - beta_dst(e) * eq(c^dst(e), idx_dst(e)) * V_dst(e)(x)
  ) = 0
```

The `jolt-claims` v1 semantics expose the same equality-edge meaning with
typed virtual `Src`, `Dst`, `Enabled`, tuple-compression, copy-point, and
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

pairing:
  multi-Miller-loop claims and public final-exponentiation check inputs

packing:
  prefix-packing layout and dense-opening claim
```

The Dory assist proof is a multi-stage virtualization protocol in the same
sense that base Jolt has multiple protocol components. The implementation
should mirror the base Jolt claim organization: small top-level modules define
IDs and relation claim types, while `formulas/*` owns dimensions, semantic
claim formulas, openings, and wiring. Stage shape is verifier organization.

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
    pairing.rs
    wiring.rs
    packing.rs
    committed_openings.rs
    claim_reductions/
      mod.rs
      gt.rs
      g1.rs
      g2.rs
      pairing.rs
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
    PairingChallenge, PackingChallenge, WiringChallenge,
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
    GtExponentiationShift,
    GtMultiplication,
    G1ScalarMultiplication,
    G1ScalarMultiplicationShift,
    G1Addition,
    G2ScalarMultiplication,
    G2ScalarMultiplicationShift,
    G2Addition,
    MultiMillerLoop,
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
pub trait PcsProofAssist<PCS: CommitmentScheme>: Sized {
    type Config;
    type Error;

    fn verify<T>(
        &self,
        config: &Self::Config,
        joint_opening_proof: &PCS::Proof,
        opening_snapshot: &PcsOpeningSnapshot<PCS>,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = PCS::Field>;
}
```

The exact trait spelling can move during implementation, but the boundary
should preserve these semantics:

```text
ordinary configured Jolt:
  stages 1-7
  -> build opening snapshot
  -> proof.pcs_assist must be None
  -> verify joint_opening_proof through the ordinary PCS verifier

Dory-assisted configured Jolt:
  stages 1-7
  -> build the same opening snapshot
  -> proof.pcs_assist must be Some(DoryAssistProof)
  -> call DoryAssistProof::verify through the PcsProofAssist boundary
  -> do not also run the expensive ordinary Dory verifier path
```

The configured verifier flow and proof-shape rules live in
[selected-verifier-integration.md](selected-verifier-integration.md).

## `dory-assist-verifier` Layout

The `dory-assist-verifier` crate owns the concrete organization of the semantics
defined in `jolt-claims::protocols::dory_assist`.

Target layout:

```text
crates/jolt-dory-assist-verifier/
  Cargo.toml
  src/
    lib.rs
    config.rs
    proof.rs
    public_inputs.rs
    transcript.rs
    opening_snapshot.rs
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
}

impl PcsProofAssist<DoryCommitmentScheme> for DoryAssistProof<...> {
    type Config = DoryAssistConfig;
    type Error = DoryAssistVerifierError;

    fn verify<T>(...) -> Result<(), Self::Error>
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
   - Add GT, G1, G2, pairing, wiring, and claim-reduction formulas.
   - Encode local constraints, output claims, virtual-polynomial dependencies,
     and wiring formulas.
   - Review gate: formula tests cover each operation family independently.

5. Add prefix-packing formulas.
   - Define dense witness layout, prefix codewords, and dense-opening claim.
   - Review gate: packing tests catch incorrect prefix layout and opening-point
     normalization.

6. Add the `dory-assist-verifier` crate and proof shape.
   - Add typed proof payloads for stage 1, stage 2, stage 3, and Hyrax opening.
   - Organize verification stages around the semantic claims from
     `jolt-claims::protocols::dory_assist`.
   - Review gate: proof-shape validation rejects missing, extra, or misordered
     stage payloads.

7. Implement the generic PCS-assist boundary for Dory.
   - Implement `PcsProofAssist<DoryCommitmentScheme>` for the Dory-assist proof
     payload.
   - Build Dory-assist public inputs from the generic opening snapshot and the
     ordinary joint opening proof.
   - Review gate: `jolt-verifier` can dispatch to Dory-assist
     fixtures without Dory-specific stage modules.

8. Add multi-Miller-loop verification path.
   - Prove multi-Miller-loop witness constraints in Dory assist.
   - Keep final exponentiation as native Dory-assist verifier work.
   - Review gate: fixtures match the Quang reference branch for equivalent
     inputs.

9. Add wrapper R1CS hooks.
   - Lower Dory-assist stages through claim, sumcheck, packing, and Hyrax R1CS
     helpers.
   - Review gate: wrapper assembly can include configured Dory-assist stages in
     a satisfied R1CS.
