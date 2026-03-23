# Protocol Graph: Claim-Level IR for SNARK Structure

## Motivation

Today jolt-ir captures individual sumcheck formulas (`ClaimDefinition`) but not the
structure of the SNARK itself: which claims exist, how they flow between sumcheck
instances, and how they converge to PCS openings. This structure is currently
hand-wired in jolt-zkvm (prover) and independently reimplemented in jolt-verifier,
with no static way to check consistency, completeness, or optimality.

The protocol graph lifts the SNARK's claim dependency structure into a first-class
IR that both prover and verifier derive from. It captures **what needs to be proven**,
not **how to compute it** — the same separation that `Expr` provides for individual
formulas, but at the protocol level.

## What the graph captures

A SNARK proof is a sequence of claim obligations:

1. Polynomials exist (committed or virtual).
2. Claims are made about those polynomials (evaluation at a point, identity holds).
3. Each claim must be **consumed** — either proven by a sumcheck (which produces
   new claims at the sumcheck's challenge point) or discharged by a PCS opening.

The protocol graph encodes this as a directed acyclic graph where:
- **Edges** are claims (a polynomial, an evaluation point, a formula).
- **Vertices** consume claims and produce new claims.
- **Terminal vertices** are PCS openings that discharge claims with no further output.

Soundness reduces to a graph property: **every claim produced must be consumed**.
A dangling claim is an unproven obligation — a potential soundness hole.

## Core types

### Polynomials

```rust
/// A polynomial in the protocol. Committed polynomials have PCS data;
/// virtual polynomials are derived during proving.
struct Polynomial {
    id: PolynomialId,
    kind: PolynomialKind,
    /// Number of variables (determines evaluation table size 2^num_vars).
    num_vars: NumVars,
}

enum PolynomialKind {
    /// Has a PCS commitment. Must be discharged by a terminal Opening vertex.
    /// The `group` determines which other polynomials this one is committed
    /// alongside — affecting proof structure but not claim flow.
    Committed { group: CommitmentGroupId },
    /// Derived from other data (R1CS witness, trace, composition of committed polys).
    /// Claims on virtual polys are proven by sumchecks but don't require PCS openings.
    Virtual,
}

/// Num vars can be a concrete value or symbolic (resolved from config at build time).
enum NumVars {
    Concrete(usize),
    /// `log_T` (cycle dimension), `log_k + log_T` (address + cycle), etc.
    Symbolic(Symbol),
}
```

### Commitment groups

A commitment group is a set of polynomials committed together as a single
prover message. Grouping is a **choice** — it affects proof structure and the
commitment phase transcript, but not the claim dependency graph.

```rust
struct CommitmentGroup {
    id: CommitmentGroupId,
    /// Polynomials in this group, committed as one vector / batch.
    polynomials: Vec<PolynomialId>,
}
```

Commitment groups determine three things:

1. **Proof size**: One commitment object per group in the proof (fewer groups =
   smaller proof if the PCS supports batch commitment).
2. **Transcript binding**: Each group produces one commitment that enters the
   Fiat-Shamir transcript before any sumcheck runs. The ordering of groups
   in the transcript is part of the commitment strategy.
3. **Opening structure**: Polynomials in the same group may share opening
   overhead (e.g., Dory's `combine_hints`). The `OpeningStage` references
   groups to determine how PCS proofs are batched.

The claim graph is agnostic to grouping — a `Committed { group }` polynomial
produces and consumes claims identically regardless of which group it belongs
to. Only the terminal `OpeningStage` and the initial commitment phase care
about groups.

**Current Jolt**: Each committed polynomial is its own group (separate
commitments for `ram_inc`, `rd_inc`, each `instruction_ra[i]`, each
`bytecode_ra[i]`, each `ram_ra[i]`, and the Spartan witness). This is the
simplest strategy — one `CommitmentGroup` per `PolynomialId`.

**Alternative**: Concatenate all RA polynomials into one large table, commit
once, and open via a single proof with internal index selection. This would
change the number of `CommitmentGroup`s and the `OpeningStage` structure but
leave the claim graph (all sumcheck vertices, all claim dependencies) completely
untouched.

### The commitment phase

The commitment phase is the first protocol step: the prover commits to
polynomials and appends the commitments to the transcript. This happens before
any sumcheck vertex executes, and it's what gives the verifier a binding
reference for subsequent PCS openings.

In graph terms, the commitment phase is **not a vertex** — it produces no
claims, consumes no claims, and involves no sumcheck. It's a transcript
operation that establishes the reference data for terminal `Opening` vertices.
Modeling it as a vertex would conflate the claim flow graph with the transcript
protocol, and these are intentionally separate concerns.

Instead, the commitment phase is captured by the **commitment strategy** — a
third choice alongside staging and batching:

```rust
/// How committed polynomials are grouped and ordered for the initial
/// commitment phase.
struct CommitmentStrategy {
    /// Groups of polynomials committed together.
    groups: Vec<CommitmentGroup>,
    /// Transcript ordering: groups are appended to the transcript in this
    /// order before any sumcheck stage executes.
    transcript_order: Vec<CommitmentGroupId>,
}
```

The commitment strategy is a scheduling decision, not a structural one:

| Aspect | Claim graph (invariant) | Staging (choice) | Commitment strategy (choice) |
|---|---|---|---|
| What it captures | Which claims exist and flow between vertices | How vertices are layered into stages | How committed polys are grouped and transcript-bound |
| When it matters | Always — structural truth | Prover/verifier execution order | Proof size, opening batch structure |
| Changing it affects | Nothing (it IS the invariant) | Challenge points, transcript length, point topology | Number of commitment objects, opening proof structure |
| Changing it does NOT affect | — | Claim formulas, polynomial identities | Claim flow, sumcheck structure, point topology |

### Claims

A claim is an assertion that a polynomial evaluates to a specific value at a
specific point. Claims are the **edges** of the protocol graph.

```rust
/// A claim: "polynomial P evaluates to v at point r."
///
/// The point is symbolic — it references the challenge output of a vertex,
/// not a concrete field element vector.
struct Claim {
    id: ClaimId,
    /// Which polynomial this claim is about.
    polynomial: PolynomialId,
    /// Where the polynomial is evaluated.
    point: SymbolicPoint,
    /// How the claimed value relates to polynomial openings (the formula).
    /// `None` for direct opening claims (the value IS the polynomial evaluation).
    formula: Option<ClaimFormula>,
}

/// A symbolic evaluation point — composed from challenge outputs of vertices.
enum SymbolicPoint {
    /// The challenge vector output by a sumcheck vertex.
    Challenges(VertexId),
    /// Concatenation: [point_a || point_b]. Used when claim reductions produce
    /// a multi-dimensional point (e.g., unified = [r_addr || r_cycle]).
    Concat(Vec<SymbolicPoint>),
    /// A sub-range of another point. Used when Spartan's r_y is sliced to
    /// extract r_cycle = r_y[..log_T].
    Slice {
        source: Box<SymbolicPoint>,
        range: VarRange,
    },
}

/// The formula relating the claimed value to polynomial openings.
/// Wraps the existing ClaimDefinition with its Expr + bindings.
struct ClaimFormula {
    definition: ClaimDefinition,
}
```

### Vertices

A vertex is an **atomic proof step**: one sumcheck instance proving one claim,
or one PCS opening discharging one claim. Vertices are the nodes of the claim
flow graph.

```rust
enum Vertex {
    /// A single sumcheck instance: proves one input claim, produces leaf
    /// claims at the stage's challenge point.
    Sumcheck(SumcheckVertex),
    /// Point normalization: transforms claims from a short point to a longer
    /// point via Lagrange zero-selector scaling. Not a proof step — a
    /// mathematical identity that the verifier checks algebraically.
    PointNormalization(PointNormalizationVertex),
    /// PCS opening: terminal vertex that discharges a committed polynomial claim.
    Opening(OpeningVertex),
}
```

#### Sumcheck vertices

A sumcheck vertex is the atomic unit: one sumcheck instance, one claim consumed,
one set of leaf claims produced. It does NOT own a challenge point — that belongs
to the stage (see below).

```rust
struct SumcheckVertex {
    id: VertexId,
    /// The input claim this sumcheck proves.
    consumes: ClaimId,
    /// The leaf claims produced (polynomial evaluations at the stage's
    /// challenge point). Fed to downstream vertices.
    produces: Vec<ClaimId>,
    /// The claim formula (output claim expression).
    formula: ClaimFormula,
    /// Degree of the sumcheck round polynomial.
    degree: usize,
    /// Number of sumcheck variables. May be less than the stage's num_vars
    /// — shorter instances are front-padded with dummy rounds when batched.
    num_vars: NumVars,
}
```

#### Point normalization vertices

```rust
/// Transforms a claim at a short point to an equivalent claim at a longer point.
///
/// When a polynomial has `num_vars = log_T` but must be opened at the unified
/// point `(r_addr, r_cycle)` with `log_k + log_T` variables, the evaluation is
/// scaled by the Lagrange zero-selector:
///
///   eval_at_unified = eval_at_short × ∏(1 − r_extra_i)
///
/// This is NOT a proof step — it's a mathematical identity. But it's a distinct
/// vertex in the graph because it transforms the claim's point, and the verifier
/// must apply the same scaling.
struct PointNormalizationVertex {
    id: VertexId,
    /// Claims at the short point.
    consumes: Vec<ClaimId>,
    /// Claims at the extended point (same polynomial, scaled evaluation).
    produces: Vec<ClaimId>,
    /// The extra dimensions to zero-pad.
    padding_source: SymbolicPoint,
}
```

#### Opening vertices

```rust
/// Terminal vertex: discharges a committed polynomial claim via PCS.
struct OpeningVertex {
    id: VertexId,
    /// The claim discharged by this opening.
    consumes: ClaimId,
}
```

### Stages

A stage is NOT a vertex — it's a **partition** of vertices into groups that
share a single Fiat-Shamir interaction.

A valid staging assigns each vertex to a stage such that:
1. **Acyclicity**: No vertex in stage k depends on another vertex in stage k.
   All dependencies flow from earlier stages to later stages.
2. **Independence**: Vertices within a stage are independent — they can be
   proven simultaneously because they only depend on randomness from prior stages.
3. **Shared challenge point**: All sumcheck vertices in a stage produce their
   leaf claims at the same challenge point (one Fiat-Shamir interaction).

```rust
struct Stage {
    id: StageId,
    /// The sumcheck vertices in this stage.
    vertices: Vec<VertexId>,
    /// The random challenge point produced by this stage's Fiat-Shamir interaction.
    ///
    /// A stage with `num_vars` rounds produces a challenge vector
    /// `(r_0, ..., r_{num_vars-1})` ∈ F^num_vars. All sumcheck vertices in the
    /// stage produce leaf claims evaluated at this point.
    ///
    /// Downstream stages reference this point via `SymbolicPoint::Challenges(self.id)`.
    challenge_point: ChallengePoint,
    /// How the vertices are batched into proof artifacts.
    ///
    /// Multiple vertices can share a batching coefficient α and be combined
    /// into a single sumcheck proof. This is an optimization — it doesn't
    /// change the claim structure, only the proof format.
    batching: Vec<BatchGroup>,
}

/// The random challenge point produced by a stage.
///
/// This is the stage's Fiat-Shamir output: `num_vars` random field elements
/// derived from the transcript after the prover sends round polynomials.
/// The dimension is a structural property — it determines how many rounds
/// the batched sumcheck runs and what evaluation point downstream claims
/// reference.
struct ChallengePoint {
    /// Dimension of the challenge vector.
    num_vars: NumVars,
}

/// A group of vertices batched under a shared α coefficient.
///
/// Vertices within a group are combined via random linear combination into
/// a single sumcheck proof. The group's max degree determines the round
/// polynomial degree.
struct BatchGroup {
    /// Vertices in this batch.
    vertices: Vec<VertexId>,
}

/// Terminal stage: discharges all remaining committed polynomial claims via PCS.
///
/// All consumed claims must be at the same evaluation point. The RLC reduction
/// combines them into a single PCS call.
///
/// The opening stage interacts with the commitment strategy: polynomials in
/// the same `CommitmentGroup` may share opening overhead (e.g., batch hints
/// in Dory). The `opening_groups` field describes how claims are batched for
/// opening, which may differ from commitment groups when the reduction strategy
/// merges claims across groups.
struct OpeningStage {
    /// Opening vertices in this stage.
    vertices: Vec<VertexId>,
    /// The evaluation point shared by all claims.
    point: SymbolicPoint,
    /// The reduction strategy for combining claims.
    reduction: ReductionStrategy,
    /// How opening claims are batched for PCS proofs.
    ///
    /// In the simplest case (current Jolt), all claims at the unified point
    /// are RLC'd into a single proof regardless of commitment group. In more
    /// sophisticated schemes, claims within a commitment group could share
    /// opening hints, producing one proof per group.
    opening_groups: Vec<OpeningGroup>,
}

/// A batch of claims opened together in one PCS proof.
struct OpeningGroup {
    /// The opening vertices in this batch.
    vertices: Vec<VertexId>,
    /// The commitment groups these claims draw from. When all committed polys
    /// are separate (current Jolt), this is one group per vertex. When polys
    /// are committed as a single table, this is one group for all vertices.
    source_groups: Vec<CommitmentGroupId>,
}

enum ReductionStrategy {
    /// Random linear combination — standard approach when all claims
    /// share the same evaluation point.
    Rlc,
}
```

This gives a layered hierarchy of structure vs choice:

```
INVARIANT:
  Vertex graph  = atomic sumcheck / PCS opening / normalization (the claim flow)

CHOICES:
  Staging       = how vertices are layered into Fiat-Shamir interactions
  Batching      = how vertices within a stage are combined into proof artifacts
  Commitment    = how committed polynomials are grouped and transcript-bound
```

The **vertex graph** is the structural truth — the fine-grained claim flow.
The **stage graph** is the quotient — contract each layer into a single node
to get the Fiat-Shamir interaction sequence. The **commitment strategy**
determines the prover's first message and the opening stage's batch structure,
but doesn't affect the vertex graph or staging.

### Graph-theoretic framing: topological layering

A staging is a **topological layering** of the vertex DAG. In standard DAG
terminology (see Coffman-Graham, 1972; Healy & Nikolov, 2013):

Given a DAG G = (V, E), a **layering** is a function L: V → {1, 2, ..., k}
such that for every edge (u, v), L(u) < L(v). Each **layer** L_i = { v | L(v) = i }
is an **antichain** — no two vertices in the same layer have a dependency edge
between them.

Mapping to our protocol:

| DAG concept | Protocol concept |
|---|---|
| Vertex | Atomic sumcheck instance or PCS opening |
| Edge (u, v) | Vertex v consumes a claim produced by vertex u |
| Layer L_i | Stage i — vertices sharing one Fiat-Shamir challenge point |
| Antichain property | Vertices in a stage are independent (can execute simultaneously) |
| Layering validity | Dependencies flow from earlier stages to later stages |
| DAG height (longest path + 1) | Minimum number of stages (sequential FS interactions) |
| Layer width |L_i| | Parallelism available at stage i |

The **minimum number of stages** equals the DAG height — the length of the
longest path in the vertex graph plus one. This is the protocol's inherent
sequential cost; no staging can reduce it.

The choice of staging beyond the minimum is a **constrained DAG scheduling
problem**. Depending on the objective:

- **Minimize layers** (fewest stages = fewest FS interactions = shortest transcript)
  → longest-path layering
- **Minimize width** (balance vertices across stages for uniform per-stage cost)
  → Coffman-Graham scheduling
- **Minimize weighted depth** (accounting for per-stage cost from round polynomial
  degree and batching overhead) → weighted DAG scheduling

In our case the objective is multi-criteria: minimize stages (proof size),
respect degree constraints (per-stage cost from batching), and ensure the
point topology converges to a single PCS opening (structural constraint on
how challenge points compose). This makes staging a constrained DAG layering
with side constraints from the `SymbolicPoint` algebra.

### The protocol graph: invariant vs choice

The protocol graph cleanly separates the **invariant structure** (the vertex
graph — what must be proven) from the **scheduling choice** (the staging —
how to partition vertices into Fiat-Shamir interactions).

```rust
/// The invariant claim dependency structure.
///
/// This is determined entirely by the SNARK's polynomial identities. It does
/// not change when you re-stage, re-batch, or re-order the proof. All
/// structural analyses (claim completeness, dimension consistency) operate
/// on this level.
struct ClaimGraph {
    polynomials: Vec<Polynomial>,
    claims: Vec<Claim>,
    vertices: Vec<Vertex>,
}

/// A particular scheduling of the claim graph into Fiat-Shamir interactions.
///
/// This is a topological layering of the vertex DAG: each stage is a layer
/// (antichain), each layer produces one challenge point. Different stagings
/// of the same claim graph yield different proof structures with different
/// tradeoffs in transcript length, round polynomial degree, and point topology.
///
/// All Fiat-Shamir-dependent analyses (transcript length, critical path,
/// point topology, opening reduction) operate on this level.
struct Staging {
    stages: Vec<Stage>,
    opening: OpeningStage,
}

/// The complete protocol specification: invariant structure + scheduling choices.
struct ProtocolGraph {
    /// The claim dependency DAG — the structural truth.
    claim_graph: ClaimGraph,
    /// A valid topological layering of the claim graph.
    staging: Staging,
    /// How committed polynomials are grouped and transcript-bound.
    commitment: CommitmentStrategy,
}

impl ProtocolGraph {
    /// Validate that the staging is a valid layering of the claim graph.
    ///
    /// Checks:
    /// - Every vertex appears in exactly one stage.
    /// - No vertex in stage k depends on another vertex in stage k (antichain).
    /// - All dependencies flow from earlier stages to later stages.
    /// - Every opening vertex's claim reaches a terminal in the opening stage.
    fn validate_staging(&self) -> Result<(), StagingError>;

    /// Validate the commitment strategy against the claim graph.
    ///
    /// Checks:
    /// - Every committed polynomial belongs to exactly one group.
    /// - No virtual polynomial appears in a commitment group.
    /// - Every opening vertex's polynomial has a commitment group.
    /// - The transcript ordering covers all groups.
    fn validate_commitment(&self) -> Result<(), CommitmentError>;

    /// Recompute the staging while preserving the claim graph and commitment strategy.
    ///
    /// Applies a different layering to the same vertex DAG. The claim graph
    /// is immutable; only the staging changes. Returns a new `ProtocolGraph`
    /// with updated `SymbolicPoint` references (since points reference `StageId`s).
    fn restage(&self, new_staging: Staging) -> Result<ProtocolGraph, StagingError>;

    /// Recompute the commitment strategy while preserving the claim graph and staging.
    ///
    /// Changes how committed polynomials are grouped for commitment and opening.
    /// The claim graph and staging are immutable — only the commitment phase and
    /// opening stage structure change.
    fn recommit(&self, new_strategy: CommitmentStrategy) -> Result<ProtocolGraph, CommitmentError>;
}
```

The `SymbolicPoint` type references `StageId` (not `VertexId`) because challenge
points are produced by stages, not individual vertices:

```rust
enum SymbolicPoint {
    /// The challenge vector output by a stage's Fiat-Shamir interaction.
    Challenges(StageId),
    /// Concatenation: [point_a || point_b].
    Concat(Vec<SymbolicPoint>),
    /// A sub-range of another point.
    Slice { source: Box<SymbolicPoint>, range: VarRange },
}
```

This means re-staging requires updating `SymbolicPoint` references in claims —
if a vertex moves from stage A to stage B, its output claims now reference
`Challenges(B)` instead of `Challenges(A)`. The `restage()` method handles
this consistently.

## Construction

The claim graph and a default staging are built together from a `ProverConfig`
(known at preprocessing time):

```rust
/// Build the claim graph: the invariant structure of the protocol.
///
/// This is determined by the SNARK's polynomial identities and the
/// configuration (trace length, one-hot params, etc.). It does not
/// depend on any scheduling decision.
fn build_claim_graph(config: &ProverConfig) -> ClaimGraph

/// Build the default staging: Jolt's current stage assignment.
///
/// This is the hand-chosen layering that matches the current jolt-core
/// pipeline (S1–S7 + opening). It can be replaced by an optimized
/// staging without changing the claim graph.
fn default_staging(graph: &ClaimGraph) -> Staging

/// Build the default commitment strategy: one group per committed polynomial.
///
/// This matches Jolt's current approach — separate commitments for each
/// polynomial. An alternative strategy might group all RA polynomials
/// into a single commitment.
fn default_commitment_strategy(graph: &ClaimGraph) -> CommitmentStrategy

/// Convenience: build everything together.
fn build_protocol_graph(config: &ProverConfig) -> ProtocolGraph {
    let claim_graph = build_claim_graph(config);
    let staging = default_staging(&claim_graph);
    let commitment = default_commitment_strategy(&claim_graph);
    ProtocolGraph { claim_graph, staging, commitment }
}
```

Both prover and verifier call `build_protocol_graph` and get the same result.
The prover walks the staging forward (executing sumchecks, producing proofs).
The verifier walks the same staging forward (replaying Fiat-Shamir, checking
claims). The claim graph is the shared structural truth; the staging is the
shared scheduling agreement.

## Fiat-Shamir interaction structure

The protocol graph has two views:

1. **Vertex graph** — fine-grained claim flow between atomic proof steps.
2. **Stage graph** — the quotient where each stage is contracted to a single
   node. This IS the Fiat-Shamir interaction sequence.

Each stage represents one round of prover↔verifier interaction:

```
Prover                          Verifier (transcript)
  │                                │
  │  round polynomials (stage 2)   │
  │  (batched from N vertices)     │
  ├───────────────────────────────→│
  │                                │
  │  challenge point r_2           │
  │←───────────────────────────────┤  ← dim = stage_2.challenge_point.num_vars
  │                                │
  │  round polynomials (stage 3)   │
  │  (batched from M vertices)     │
  ├───────────────────────────────→│
  │                                │
  │  challenge point r_3           │
  │←───────────────────────────────┤  ← dim = stage_3.challenge_point.num_vars
  │                                │
  ...
```

The stage ordering must be a valid topological sort of the stage graph: stage
S_j comes after S_i if any vertex in S_j consumes a claim produced by a vertex
in S_i. This is the Fiat-Shamir sequencing constraint.

**Key property**: The dimension of each challenge point is known statically
from the graph. The total transcript length (number of random field elements)
is `Σ stage.challenge_point.num_vars` — computable without running the protocol.

**Staging and randomness**: All vertices within a stage share the same
challenge point. This is the core structural choice:
- Fewer stages → fewer interaction rounds → shorter transcript, but potentially
  higher round polynomial degree from batching more claims together.
- More stages → finer granularity → lower per-stage degree, but more sequential
  interactions and more distinct challenge points (which may complicate the
  point topology for opening reduction).

**Point composition**: The `SymbolicPoint` algebra describes how stage-level
challenge points compose into evaluation points for claims:

- `Challenges(stage_7)` — the raw challenge vector from stage 7 (dim = `log_k_chunk`)
- `Challenges(stage_6)` — the raw challenge vector from stage 6 (dim = `log_T`)
- `Concat(Challenges(stage_7), Challenges(stage_6))` — the unified point (dim = `log_k_chunk + log_T`)
- `Slice(r_y, 0..log_T)` — extracting `r_cycle` from Spartan's full `r_y`

Every evaluation point in the protocol is a composition of stage challenge
points. The graph makes this composition explicit and checkable: the dimension
of a `Concat` is the sum of its children's dimensions, a `Slice` must reference
a valid range, etc.

## Analyses

Analyses partition cleanly between the two levels:

| Level | Analyses | Changes when re-staged? |
|---|---|---|
| **Claim graph** (invariant) | Claim completeness, dimension consistency | No |
| **Staging** (choice) | Transcript length, critical path, point topology, batching, opening reduction | Yes |

### Claim graph analyses (invariant)

#### Dimension consistency

Every claim asserts a polynomial evaluation at a point. The point's dimension
must match the polynomial's `num_vars`:

```
for claim in graph.claims:
    assert!(claim.point.dimension() == graph.polynomial(claim.polynomial).num_vars)
```

Where `SymbolicPoint::dimension()` is:
- `Challenges(v)` → `v.challenge_point.num_vars`
- `Concat(children)` → `Σ child.dimension()`
- `Slice(source, range)` → `range.len()`

This catches mismatches statically: if a committed polynomial has `num_vars = log_k + log_T`
but a claim references it at `Challenges(S6)` with dimension `log_T`, that's an error — a
`PointNormalization` vertex is needed to extend the point.

### Transcript length

Total Fiat-Shamir challenges = `Σ stage.challenge_point.num_vars` for all stages.
This is a structural property of the graph, computable without running the protocol.
Useful for estimating proof size and verifier cost.

### Critical path

The **depth** of the stage graph (longest path from a root stage to the opening
stage) is the minimum number of sequential Fiat-Shamir interactions. This is
the protocol's inherent sequential cost — no amount of parallelism can reduce it.

The **width** at each level (number of independent stages at the same depth) is
the available parallelism. Stages at the same depth can execute concurrently on
a distributed prover.

#### Soundness: claim completeness

Every claim produced by a vertex must be consumed by exactly one downstream vertex:

```
produced = ∪ { v.produces | v ∈ vertices }
consumed = ∪ { v.consumes | v ∈ vertices }

// Every produced claim is consumed (no dangling obligations)
assert!(produced ⊆ consumed);

// No claim is consumed without being produced (except root claims from R1CS)
assert!(consumed \ root_claims ⊆ produced);

// No claim is consumed twice
assert!(all consume lists are disjoint);
```

A dangling claim (produced but not consumed) means the SNARK has an unproven
obligation. A missing claim (consumed but never produced) means the SNARK
assumes something it didn't establish. Both are soundness bugs detectable by
graph inspection.

### Staging analyses (choice-dependent)

#### Opening reduction: point topology

The number of PCS proofs required is determined by the number of distinct
`SymbolicPoint` equivalence classes at terminal `Opening` vertices:

```
point_groups = group_by(opening.consumes, |claim| claim.point)
num_pcs_proofs = |point_groups|
```

If all committed polynomial claims converge to a single unified point (via
`Concat` + `PointNormalization`), one RLC reduction produces one PCS proof.
The graph makes this visible: you can trace the point topology from each
committed polynomial's claims through normalization vertices to the terminal
opening, and verify that all points unify.

**Optimization opportunity**: If the graph shows two distinct terminal points
that could be unified by restructuring the claim reduction chain (e.g., reordering
two independent stages so their challenges concatenate differently), an optimizer
could discover this and suggest a restructuring that eliminates a PCS proof.

#### Batching optimization

Batching operates at two levels:

**Within a stage**: which vertices share a batching coefficient α and are
combined into a single sumcheck proof? Vertices in the same batch group
must have compatible `num_vars` (shorter instances are front-padded). The
batch's round polynomial degree is the maximum degree across its vertices.

**Across stages**: should two stages be merged into one? Two stages at the
same depth in the stage graph (no dependency between them) can be merged,
reducing the number of Fiat-Shamir interactions. But merging means all
vertices from both stages share one challenge point — which changes the
downstream point topology.

Given the vertex graph, optimal staging is a constrained DAG layering problem:

```
minimize:  number of stages (each stage = one Fiat-Shamir interaction)
subject to:
  - valid layering (dependencies flow forward)
  - degree constraints (max degree per batch group within a stage)
  - point topology constraints (staging must allow all committed claims
    to converge to a single opening point, or minimize opening count)
```

The current hand-chosen staging can be validated against this optimum, and
the graph can identify opportunities where merging two stages would reduce
proof size or where splitting a stage would improve the point topology.

#### Consequence analysis

Staging decisions propagate through the graph:

- **Merging two stages** into one means all their vertices share a single
  challenge point. Downstream claims that previously referenced two distinct
  points now reference one — this may enable further merging downstream or
  simplify the opening reduction. But it may also increase the maximum degree
  within the merged stage.

- **Splitting a stage** creates two distinct challenge points. Downstream
  vertices that consumed claims from both halves now need claims at two
  different points, potentially requiring an additional claim reduction vertex
  or point normalization.

- **Moving a vertex** from one stage to another changes which challenge point
  its output claims land at. All downstream references to those claims must
  be updated. The graph makes these dependencies explicit so you can check
  whether a restructuring is valid without running the full prover.

- **Changing the batching** within a stage (which vertices share an α) changes
  the proof artifact structure but NOT the claim graph or point topology. This
  is a pure proof-size / verifier-cost optimization.

The graph enables a "what-if" analysis: propose a staging change, derive the
new stage graph, check that all claims are still consumed and the point
topology still converges to the desired number of PCS openings.

## Relationship to existing jolt-ir types

The protocol graph builds on top of existing jolt-ir infrastructure:

| Existing type | Role in protocol graph |
|---------------|----------------------|
| `ClaimDefinition` | Wrapped by `ClaimFormula` — the symbolic expression for a claim's value |
| `Expr` | The formula inside each `ClaimDefinition` — unchanged |
| `OpeningBinding` | Maps formula variables to `PolynomialId`s in the graph |
| `ChallengeBinding` | Maps formula variables to challenge sources (batching coeff, derived, etc.) |
| `KernelDescriptor` | Not part of the graph — computed by the prover from `ClaimDefinition`s when executing a vertex |
| Tags (`poly::*`, `sumcheck::*`) | Replaced by typed `PolynomialId` / `VertexId` — no more opaque `u64` coordination |

The protocol graph does NOT replace `Expr`, `ClaimDefinition`, or `KernelDescriptor`.
It adds a layer above them that captures how claims compose into a complete proof,
including how committed polynomials are grouped and how openings are batched.

## Relationship to jolt-zkvm and jolt-verifier

**Today**: jolt-zkvm's `prover.rs` is the implicit DAG. Each `prove_stageN` function
is a vertex. Data flow is encoded in function signatures. The verifier independently
reimplements the same DAG in its `verify()` function. Consistency between the two
is checked only by end-to-end tests.

**With protocol graph**: Both crates import the same `ProtocolGraph` from jolt-ir.
The prover and verifier agree on both the claim graph (what to prove) and the
staging (how to schedule the proof). The claim graph is the structural truth;
the staging is the shared scheduling contract.

```
jolt-ir:
  ClaimGraph   (invariant — what must be proven)
  Staging      (choice — how to schedule the proof)
  ProtocolGraph = ClaimGraph + Staging

jolt-zkvm (prover):
  for stage in staging.topological_order():
    for vertex in stage.vertices:
      match vertex:
        Sumcheck(v) → build witness, execute sumcheck
        PointNormalization(v) → scale evaluations by Lagrange factor
    // stage produces one challenge point from transcript
    for vertex in stage.vertices:
      // evaluate produced claims at stage's challenge point
  for opening in staging.opening.vertices:
    // RLC reduce + PCS open

jolt-verifier:
  for stage in staging.topological_order():
    // verify batched sumcheck proof for this stage
    // extract evaluations from proof data
    for vertex in stage.vertices:
      match vertex:
        Sumcheck(v) → check output claim against formula
        PointNormalization(v) → scale evaluations by Lagrange factor
  for opening in staging.opening.vertices:
    // RLC reduce + PCS verify
```

Stage-specific imperative code (building eq tables, materializing virtual polynomials,
computing challenge values) remains in jolt-zkvm. The graph doesn't absorb that — it
captures the **obligation structure**, not the implementation.

**Re-staging without re-implementing**: Because the claim graph is invariant, an
alternative staging (e.g., merging S3 and S4 into one stage) requires no changes
to claim definitions, formulas, or polynomial identities. Only the scheduling
changes — which vertices run together, which challenge point they share, and
how the downstream `SymbolicPoint` references update. The prover and verifier
walk the new staging with the same vertex implementations.

## Parameterization

The graph shape depends on configuration:

- `log_T` (trace length) → `num_vars` for cycle-dimension sumchecks
- `log_k_chunk` (one-hot chunk size) → number of RA polynomials, `num_vars`
  for address-dimension sumchecks
- `instruction_d`, `bytecode_d`, `ram_d` → number of claims in RA virtual /
  booleanity / Hamming weight vertices

These are known at preprocessing time. The `build_protocol_graph(config)` function
instantiates the graph with concrete values, producing a fully resolved structure
that both prover and verifier traverse.

Formulas that are parameterized by arity (e.g., `hamming_weight_claim_reduction`
takes a `poly_tags` slice) are instantiated at graph build time with the concrete
arity from config.

## What the graph does NOT capture

- **Proving implementation**: How eq tables are built, how witnesses are materialized,
  how kernels are compiled. These are jolt-zkvm concerns.
- **Commitment computation**: How commitments are computed (streaming Dory tiers,
  MSM strategy, GPU offload). The graph captures *what* is committed together
  and *when* it enters the transcript, not *how* the commitment is computed.
- **BlindFold / ZK layer**: Committed round polynomials, verifier R1CS, Nova folding.
  This is a layer on top that can be added later as a graph transformation.
- **Multi-phase sumcheck internals**: A prefix-suffix sumcheck is one vertex with
  one set of consumed/produced claims. The two-phase execution is an implementation
  detail of the prover's sumcheck engine.
- **Toom-Cook / kernel shapes**: These are compute backend concerns, not protocol
  structure. The graph says "this vertex proves these claims"; the backend decides
  how to evaluate the round polynomial efficiently.

## Concrete Jolt graph structure

The Jolt protocol graph (at the current pipeline) has the following shape.
Vertex names correspond to current stage names for clarity.

### Commitment phase (before any sumcheck)

Current strategy: one group per committed polynomial.

```
CommitmentGroup 0: ram_inc          → transcript ← commitment_0
CommitmentGroup 1: rd_inc           → transcript ← commitment_1
CommitmentGroup 2..2+D_instr: instruction_ra[0..D_instr]
CommitmentGroup 2+D_instr..2+D_instr+D_bc: bytecode_ra[0..D_bc]
CommitmentGroup 2+D_instr+D_bc..2+D_instr+D_bc+D_ram: ram_ra[0..D_ram]
CommitmentGroup last: spartan_witness → transcript ← commitment_last
```

All commitments enter the transcript before S1 executes, binding the prover
to the committed data. The claim graph is agnostic to this grouping — the
same vertex DAG applies regardless of whether these are 2+D_instr+D_bc+D_ram+1
separate commitments or one big table commitment.

### Claim flow

```
Spartan (S1)
  produces: virtual evals at Slice(r_y, 0..log_T)
    → ram_read_value, ram_write_value, ram_address,
      lookup_output, left/right_operand, left/right_instruction_input,
      rd_write_value, rs1_value, rs2_value

ProductVirtual + RamRW + InstrLookupsCR + RamRafEval + OutputCheck (S2)
  consumes: S1 virtual evals
  produces: ram_val, ram_inc, next_is_noop, left/right_instr_input,
            lookup_output, left/right_operand, ram_raf_eval, ram_val_final
            at Challenges(S2)

Shift + InstrInput + RegistersCR (S3)
  consumes: S1 virtual evals, S2 evals
  produces: rs1_value, rs2_value, rd_write_value
            at Challenges(S3)

RegistersRW + RamValCheck (S4)
  consumes: S2 evals, S3 evals
  produces: ram_inc, rd_inc
            at Challenges(S4)

RegistersValEval (S5)
  consumes: S2 evals, S4 evals
  produces: rd_inc
            at Challenges(S5)

IncCR + HammingBooleanity (S6)
  consumes: S2.ram_inc, S4.ram_inc, S4.rd_inc, S5.rd_inc
  produces: ram_inc_reduced, rd_inc_reduced
            at Challenges(S6) = r_cycle

HammingWeightCR (S7)
  consumes: S6.r_cycle, RA poly claims from S5/S6
  produces: instruction_ra[..], bytecode_ra[..], ram_ra[..]
            at Challenges(S7) = r_addr
  unified_point = Concat(Challenges(S7), Challenges(S6))

PointNormalization
  consumes: S6.ram_inc_reduced at Challenges(S6)
            S6.rd_inc_reduced  at Challenges(S6)
  produces: ram_inc at unified_point (scaled by Lagrange)
            rd_inc  at unified_point (scaled by Lagrange)
  padding_source: Challenges(S7)

Opening (terminal)
  consumes: all committed poly claims at unified_point
    → ram_inc, rd_inc (via PointNormalization)
    → instruction_ra[..], bytecode_ra[..], ram_ra[..] (from S7)
    → spartan_witness at Slice(r_y, ...)
  reduction: Rlc → single PCS proof
  opening_groups: [{all vertices, all commitment groups}]
    (current Jolt: one RLC across all commitment groups → one Dory proof)
```

Note how the commitment strategy and opening structure are orthogonal to
the claim flow above. An alternative commitment strategy (e.g., one big
table for all RA polynomials) would change only the commitment groups and
the `opening_groups` in the terminal — the Spartan vertex, sumcheck
vertices S2–S7, point normalization, and all claim dependencies remain
identical.

## Worked example: jolt-core's commitment strategy

This section maps the real jolt-core prover (the legacy implementation in
`jolt-core/src/zkvm/prover.rs`) onto the `CommitmentStrategy` model, showing
that our system captures what already happens in practice.

### Committed polynomials

jolt-core has two size classes of committed polynomials:

| Class | Polynomials | Length | num_vars |
|---|---|---|---|
| **Dense** | `RdInc`, `RamInc` | `2^log_T` | `log_T` |
| **Sparse (RA)** | `InstructionRa[0..D_instr]`, `BytecodeRa[0..D_bc]`, `RamRa[0..D_ram]` | `2^(log_T + log_k)` | `log_T + log_k` |
| **Spartan witness** | `flat_witness` (interleaved R1CS columns) | `2^log_M` | `log_M` |

There are also advice polynomials (`TrustedAdvice`, `UntrustedAdvice`) in
jolt-core which the refactored pipeline does not yet include; they follow
the same pattern.

Total committed polynomials: `2 + D_instr + D_bc + D_ram + 1` (typically ~20).

### jolt-core's commitment strategy as `CommitmentStrategy`

jolt-core commits each polynomial individually — one `PCS::commit()` call per
polynomial. In our model this is **one `CommitmentGroup` per polynomial**:

```rust
// What jolt-core does, expressed as a CommitmentStrategy.
//
// Reference: jolt-core/src/zkvm/prover.rs lines 659-836
// Reference: jolt-zkvm/src/prover.rs commit_polynomials() lines 165-177
fn jolt_core_commitment_strategy(ohp: &OneHotParams) -> CommitmentStrategy {
    let mut groups = Vec::new();
    let mut order = Vec::new();
    let mut next_id = 0;

    let mut add_group = |poly_id: PolynomialId| -> CommitmentGroupId {
        let gid = CommitmentGroupId(next_id);
        next_id += 1;
        groups.push(CommitmentGroup {
            id: gid,
            polynomials: vec![poly_id],
        });
        order.push(gid);
        gid
    };

    // Dense polynomials — committed first
    // (jolt-core: streaming Dory tier-1 → tier-2 → final commitment)
    add_group(poly_id::RAM_INC);
    add_group(poly_id::RD_INC);

    // RA polynomials — one group each
    // (jolt-core: same streaming pipeline, but length 2^(log_T + log_k))
    for i in 0..ohp.instruction_d {
        add_group(poly_id::instruction_ra(i));
    }
    for i in 0..ohp.bytecode_d {
        add_group(poly_id::bytecode_ra(i));
    }
    for i in 0..ohp.ram_d {
        add_group(poly_id::ram_ra(i));
    }

    // Spartan witness — committed separately
    // (jolt-zkvm: PCS::commit on flat_witness, appended to transcript)
    add_group(poly_id::SPARTAN_WITNESS);

    CommitmentStrategy {
        groups,
        transcript_order: order,
    }
}
```

### Transcript binding

jolt-core appends each commitment to the Fiat-Shamir transcript individually,
in the order above, before any sumcheck stage executes:

```
// jolt-core/src/zkvm/prover.rs lines 773-776
for commitment in &commitments {
    transcript.append_serializable(b"commitment", commitment);
}
// jolt-core/src/zkvm/prover.rs lines 807, 832-835
transcript.append_serializable(b"untrusted_advice", &advice_commitment);
transcript.append_serializable(b"trusted_advice", &trusted_commitment);
```

The `transcript_order` field in `CommitmentStrategy` captures this: it's the
order in which groups enter the transcript. Since each group is one polynomial,
the transcript order is the polynomial commitment order.

### Opening phase

At the end (stage 8), jolt-core converges all committed polynomial claims to
a **unified point** and opens them with a single Dory proof:

```
unified_point = (r_addr || r_cycle)   // Concat(Challenges(S7), Challenges(S6))
                dim = log_k + log_T

Dense polys (RdInc, RamInc):
  - Evaluation at r_cycle from S6 IncCR
  - Scaled by Lagrange zero-selector: eval_unified = eval_short × ∏(1 - r_addr_i)
  - Zero-padded to unified point dimension

RA polys (InstructionRa, BytecodeRa, RamRa):
  - Already at unified point from S7 HammingWeightCR

Spartan witness:
  - At r_y from Spartan (different point — separate PCS proof in jolt-zkvm,
    or folded into the same RLC in jolt-core via point embedding)
```

All claims at the unified point are combined via random linear combination
(`gamma` sampled from transcript after claims are absorbed) into a single
RLC polynomial, and one Dory opening proof is produced.

In our model:

```rust
OpeningStage {
    vertices: vec![/* one OpeningVertex per committed poly */],
    point: SymbolicPoint::Concat(vec![
        SymbolicPoint::Challenges(S7),  // r_addr
        SymbolicPoint::Challenges(S6),  // r_cycle
    ]),
    reduction: ReductionStrategy::Rlc,
    opening_groups: vec![
        // All claims → single RLC → single Dory proof
        OpeningGroup {
            vertices: vec![/* all opening vertices */],
            source_groups: vec![/* all CommitmentGroupIds */],
        },
    ],
}
```

### What an alternative strategy would change

Suppose we grouped all instruction RA polynomials into a single commitment:

```rust
// Alternative: batch InstructionRa[0..D_instr] into one commitment group
CommitmentGroup {
    id: CommitmentGroupId(2),
    polynomials: (0..ohp.instruction_d)
        .map(|i| poly_id::instruction_ra(i))
        .collect(),
}
```

**What changes**:
- Fewer commitment objects in the proof (D_instr commitments → 1 commitment)
- Transcript binding: one commitment enters the transcript instead of D_instr
- Opening: Dory's `combine_hints` can share setup across the group's polynomials
- `OpeningGroup` structure may split if the batch-opening API differs from RLC

**What does NOT change**:
- Claim graph: all sumcheck vertices, all claim formulas, all claim dependencies
- Staging: S1–S7 structure, challenge points, point topology
- PointNormalization: same Lagrange scaling for dense polys
- The unified point construction

This is the key property: the claim graph is the structural invariant, and
commitment grouping is a proof engineering choice that affects only the
boundaries of the protocol (first message + terminal opening).

## Implementation plan

### Phase 1: Claim graph types + construction

Add `ClaimGraph` and its constituent types (`Polynomial`, `Claim`, `Vertex`,
`SymbolicPoint`, etc.) to `jolt-ir::protocol` (new module). Implement
`build_claim_graph(config)` encoding all Jolt sumcheck instances as vertices
and all claim dependencies as edges.

Claim graph validation passes:
- `validate_completeness()` — every produced claim is consumed, no dangling edges
- `validate_acyclicity()` — topological sort exists
- `validate_dimension_consistency()` — point dimensions match polynomial num_vars

### Phase 2: Staging + commitment strategy types

Add `Staging`, `Stage`, `BatchGroup`, `OpeningStage`, `ChallengePoint` types.
Add `CommitmentStrategy`, `CommitmentGroup`, `OpeningGroup` types.
Implement `default_staging(graph)` producing the current Jolt S1–S7 assignment.
Implement `default_commitment_strategy(graph)` producing one-group-per-polynomial.

Staging validation passes:
- `validate_staging()` — valid topological layering (antichain property, all vertices covered)
- `validate_point_convergence()` — all terminal opening claims share a point
- `validate_batching()` — no circular transcript deps within batch groups
- `transcript_length()` — total FS challenges
- `critical_path_depth()` — longest path in the stage graph

Commitment validation passes:
- `validate_commitment()` — every committed poly in exactly one group, no virtual polys in groups
- Opening group consistency — every opening vertex's polynomial has a source group

### Phase 3: Prover/verifier derivation

Refactor jolt-zkvm and jolt-verifier to walk the `ProtocolGraph` instead of
hard-coded stage sequences. The staging's topological order determines execution
order. Vertex implementations (witness building, eq construction) are attached
via a registry keyed by `VertexId`, not embedded in the graph.

### Phase 4: Optimization analysis

Build tools that operate on the three levels:

On the claim graph (invariant):
- Soundness checker: are all claims consumed? Are any formulas under-constrained?
- Dependency analyzer: what's the longest path? Where are the bottlenecks?

On the staging (choice):
- Re-staging: given the claim graph, compute alternative valid layerings
- Batching optimizer: propose merges/splits that reduce proof size
- Point topology analyzer: identify unnecessary PCS proofs from staging choices
- Consequence simulator: `restage()` + re-validate to test "what if we merge S3 and S4?"

On the commitment strategy (choice):
- Group optimizer: identify polynomials that could share a commitment group
  to reduce proof size or enable shared opening hints
- `recommit()` + re-validate to test "what if all RA polys share one commitment?"
- Opening batch analyzer: given the commitment groups and point topology, compute
  the minimum number of PCS proofs
