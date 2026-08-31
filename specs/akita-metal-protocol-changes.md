# Akita Metal protocol changes

## Scope and status

This is the review ledger for every proof-system change used by the combined
Jolt/Akita Metal path. It separates changes to the statement, witness language,
transcript, proof shape, public parameters, or verifier from prover-local
execution changes. A CPU prover and a Metal prover running the same protocol
version must produce the same proof bytes.

The ledger was audited against Jolt `feat/akita-metal` at `f06542b99` and the
Akita fork's quotient-free protocol cut at `71ecb10d8`. It describes branch
state, not an upstream protocol version. The full pre-Metal Akita integration is
specified in [lattice-claims.md](lattice-claims.md); this document is the
authoritative delta for the Metal campaign.

| Change | Classification | Prover | Proof/transcript | Verifier |
|---|---|---|---|---|
| fp128 Akita parameter suite | inherited public configuration | uses fp128 arithmetic and its Akita schedule family | field encoding and schedule are statement-bound | replays the same suite |
| fixed-prefix Akita commitment objects | inherited Akita protocol | commits canonical `OneHotTrace` and auxiliary layouts | binds layout, evaluations, then selector | checks layout and recomputes the selector reduction |
| implicit lane zero and balanced increments | adopted Metal-motivated Jolt protocol change | commits only nonzero lanes and emits centered digits | changes the stage-7 relation, committed data, and layout digest | reconstructs the omitted lane and balanced value |
| eval-oriented K256 schedule | adopted public-parameter change | uses the new canonical root geometry | changes the effective schedule digest and proof geometry | resolves the same row from public shape |
| quotient-free recursive A rows | adopted Akita protocol change | omits A quotient rows and proves a reduced-ring relation | preserves challenge order but changes witness/proof shape and stage-2/3 formulas | checks the reduced relation directly |
| Metal kernels, streaming, hints, overlap, and hybrid routing | execution only | changes where and when identical arithmetic runs | no change | no change |

## Jolt committed-data change

### Implicit public lane zero

For a semantic one-hot column `P(a,t)`, let `A(t)` be its activation and let
`S` be the committed polynomial with lane zero omitted:

```text
S(0,t) = 0
H(t)   = sum_a S(a,t)
P(a,t) = S(a,t) + eq(a,0) * (A(t) - H(t)).
```

At an arbitrary address point `r`, the identity used by stage 7 is

```text
P(r,t) = eq(r,0) * A(t)
       + sum_{a != 0} (eq(r,a) - eq(r,0)) * S(a,t).
```

`A=1` for instruction, bytecode, and increment columns. For RAM it is the
existing RAM-access indicator, whose bound value is supplied by the
`RamHammingBooleanity` claim and is already transcript-bound in stage 6. Stage
6 continues to establish the semantic Booleanity/virtualization claims for
`P`; stage 7 recenters those claims into openings of `S`. The verifier derives
the default-lane equality values and RAM activation itself. There is no
prover-selected default, second commitment, or new Fiat-Shamir challenge.

The canonical layout is domain-separated by
`jolt/akita/one_hot_trace/implicit-zero-balanced-inc/v5`. The verifier checks
this digest before the Akita opening. A selected zero-valued row and an absent
physical coefficient may use different adapter metadata, but that metadata is
only an encoding of the fixed semantic identity above.

### Balanced fused increment

The compatibility names `UnsignedIncChunk` and `UnsignedIncMsb` now denote
centered radix digits and a signed carry. For radix `B=2^b`, the committed
one-hot lanes encode

```text
delta = sum_j B^j * d_j + 2^64 * c,
d_j, c in [-B/2, B/2 - 1].
```

At a Boolean lane `a`, stage 7 decodes the centered value as

```text
balanced(a) = a - 2^b * msb(a).
```

The witness generator emits `c in {-1,0,1}`, but soundness does not trust that
generator restriction. Each digit and carry column is one-hot, and the fused
increment relation checks the reconstruction. Even allowing any centered carry
lane, the represented integer has magnitude below `2^72` for K16 and K256, far
below the fp128 modulus; equality to the 64-bit increment therefore cannot be
satisfied by modular wraparound.

These two representation changes alter Jolt's Akita-only stage-7 relation,
opening claims, committed polynomial meaning, and verifier publics. The Dory
and BlindFold paths are not changed; `akita` and `zk` remain mutually exclusive.

## Eval-oriented canonical schedule

For a single packed K256 polynomial with 38 through 41 variables, Jolt now
generates a canonical schedule with root dimensions

```text
inner = 512, outer = 64, opening = 64, inner_output_rank = 1.
```

The constrained positions per block are:

| Jolt trace | Packed variables | Positions per block |
|---:|---:|---:|
| `T=2^25` | 38 | `2^16` |
| `T=2^26` | 39 | `2^16` |
| `T=2^27` | 40 | `2^17` |
| `T=2^28` | 41 | `2^18` |

Other shapes use the ordinary planner. This is not a Metal-only runtime hint:
the schedule controls commitment and opening geometry, changes proof bytes, and
is covered by Akita's schedule-row/effective-schedule digests. The CPU and Metal
backends use the same row. The row is derived from public shape, validated by
the existing planner and SIS policy, and never selected by the prover inside a
proof.

Quotient-free recursive A rows also change the terminal geometry consumed by
the planner. Every checked-in Jolt schedule family must therefore be generated
against the same Akita protocol revision. This branch regenerates the K16,
K256, and dense catalogs; mixing the current quotient-free prover with an older
K16 or dense catalog is rejected during setup because its terminal geometry no
longer matches the witness.

## Quotient-free recursive A relations

Let `R_D = F[X]/(X^D+1)`. A recursive inner-matrix row must establish, up to the
protocol's sign convention,

```text
A * z = c * t  in R_D.
```

Previously Akita committed an A-row quotient `q` in the next witness and, after
that commitment was absorbed, sampled `alpha` and checked the ordinary
polynomial identity

```text
A(alpha)z(alpha) - c(alpha)t(alpha)
    - (alpha^D + 1)q(alpha) = 0.
```

The new protocol removes `q` for `RelationRowFamily::Inner`. It reduces the
residual modulo `X^D+1` and applies the random linear functional

```text
L_alpha(sum_k p_k X^k) = sum_k p_k alpha^k.
```

`L_alpha` is linear but is not multiplicative in `R_D`. The verifier therefore
must not replace `L_alpha(Az mod (X^D+1))` with `A(alpha)z(alpha)`. For public
`a`, its transposed multiplication weights are

```text
s_j     = L_alpha(a * X^j mod (X^D+1))
s_0     = a(alpha)
s_(j+1) = alpha*s_j - (alpha^D+1)*a_(D-1-j).
```

The same construction handles the sparse challenge product on the `t` side.
The prover may build these weights on Metal, but the verifier derives the
corresponding setup and challenge evaluations independently from public setup,
transcript challenges, and `alpha`.

The transcript dependency is unchanged and load-bearing: the outgoing witness
commitment is absorbed before `alpha` is sampled. There is no new proof message
or prover-supplied scalar. The witness layout changes from one quotient slot per
relation row to row-aligned optional slots; inner/A rows have no slot, while
consistency, outer/B, opening/D, and compression rows retain their quotients.
Consequently the outgoing commitment binds a different, shorter live witness
layout and is not byte-compatible with the old protocol. Power-of-two padding
can leave the serialized proof length unchanged.

Stage 2 now evaluates the reduced A setup and challenge terms at its coefficient
point. The deferred setup check in stage 3 includes that stage-2 coefficient
point in its factor sum. This is a verifier relation change, not merely a device
optimization, even though it adds no round or challenge.

For a nonzero reduced residual of degree below `D`, the local random-functional
test fails to detect it with probability at most `(D-1)/|E|`, where `E` is the
ring-switch challenge field. The protocol-wide soundness statement must account
for every batched row and recursive level; that union-bound ledger should be
written explicitly during cryptographic review rather than inferred from the
kernel parity tests.

## Inherited Akita protocol surface

The following mechanisms predate the Metal backend but are prerequisites of the
combined path and must not be mistaken for prover conveniences:

- `OneHotTrace` is one fixed-capacity prefix-packed polynomial with canonical
  semantic column order, zero-prefix embeddings, logical arities, and a checked
  layout digest.
- The transcript binds the common opening point and ordered semantic
  evaluations before sampling the selector that reduces them to one physical
  Akita opening. The verifier recomputes that reduction.
- Advice, bytecode, and program-image data use separate canonical prefix-packed
  objects and their reconstruction relations. Public/precommitted one-hot
  validity is checked during preprocessing; prover-supplied validity is checked
  in protocol.
- `RdInc` and `RamInc` share the fused increment selected by the bytecode Store
  flag. The consumer relations and stage-7 decode bind that fusion.

The detailed claim-to-code map and stage ordering remain in
[lattice-claims.md](lattice-claims.md).

## What did not change

The following accepted optimizations are proof-equivalent implementations and
must stay outside protocol review unless they later change a transcript boundary:

- selection of CPU, Metal, or a deliberate hybrid route;
- cycle-major packed trace storage, streamed row production, and omission of an
  address-major transpose;
- prepared setup caches, retained outer quotients in private hints, Metal-made
  evaluation indices, compact residency, coefficient packing, and CPU scatter
  specializations;
- streamed root decomposition/folding, live-prefix skipping, fused kernels,
  device/host overlap, static-session preparation overlap, and nonce-local
  speculative work whose rejected state is never absorbed;
- PIOP Metal kernels. Fiat-Shamir remains host-owned: each required round
  polynomial is bound before the next challenge, with unchanged labels, message
  order, challenge distributions, round counts, and verifier equations.

Backend choice is not transcript-bound. A qualified Metal route may fail closed,
but changing route cannot change the statement or accepted proof.

No wider opening basis, altered challenge distribution, relaxed SIS policy,
extra witness commitment, committed-witness degree-nine fusion, or adaptive
default lane was adopted.

## Structural candidates not adopted

The post-S16 T28 feasibility audit considered the following protocol changes. None is
present in the branch. They are recorded here so that prover-side experiments cannot
silently turn into protocol changes.

### Combined Product/Instruction terminal claim

A new protocol version could carry one Fiat--Shamir-weighted combination of the two
remaining lookup-operand evaluations from Stage 2 through InstructionInput,
Instruction Read-RAF, and the final opening accumulator. This preserves the statement
and can use the existing random-linear-combination soundness argument, but it changes
claim types and verifier equations across four stages. Its complete effective ceiling
is only about 2.06 seconds even if the terminal spans and all of Stage 3 disappear, so
it does not change the T28 5x feasibility bound. It was rejected before code.

### Authenticated memory event log

One address-sorted `(address, cycle, pre, post)` event log could replace the RAM and
register read/write/value relations if a permutation argument authenticates it against
the cycle-major access tape and adjacency constraints enforce value continuity. This
would change the memory-consistency argument, add witness columns and challenges,
replace several stage outputs/openings, and require a new soundness proof. It is held
as a separate protocol campaign; it is not a minor Metal backend option.

### Committed address symbols

The T28 follow-up closed both obvious encodings. Raw bit planes would recover a
one-hot opening through

```text
sum_t eq(r_cycle, t) * product_j
    (r_addr[j] * b_j(t) + (1 - r_addr[j]) * (1 - b_j(t))),
```

but sparse commitment needs about 3.194 trillion D512 ring additions on the measured
BTreeMap shape, while a direct post-bind fp128 product frontier exceeds the 90-GiB
limit. Direct bytes require a degree-255 equality polynomial or a new lookup argument;
Akita Stage 1's quadratic range image is not a generic 256-entry lookup.

The remaining candidate uses four balanced radix-4 digits per address byte. Map
`0,1,2,3` to `0,1,-2,-1`, commit four length-`T` digit polynomials per semantic
member, and keep member/digit indices outside the multilinear domain. For a requested
digit `a`, the equality factor is its cubic Lagrange polynomial `L_a(c)`. With
`S = c(c + 1)`, every such cubic can be written

```text
L_a(c) = A_a + B_a c + C_a S + D_a cS.
```

Akita already range-checks the balanced basis-4 digit alphabet and binds `S`; the
smallest proposed extension authenticates the additional virtual `cS` table and uses
a low-degree product tree for the four digit factors. It must not become one
degree-13 sumcheck. A grouped T28 layout has 120 compact digit polynomials, 30 GiB of
i8 source, about 62.9 million D512 root rings, and about 1.96 GiB of root-successor
fields. Sparse root commitment is still worse than the current hot-entry path, so the
candidate requires a dense NTT root.

This remains a candidate, not a branch protocol. It changes the committed witness
layout, one-hot Booleanity/Hamming reduction, mapped opening relation, proof shape,
verifier equations, and schedule family, but not the Jolt statement or memory
argument. Before production code it needs an exact deleted-owner audit, a complete
T28 replacement bound, a <=90-GiB lifetime schedule, a soundness-error delta, and a
new transcript/layout version. The old one-hot config must remain available and
cross-version proofs must fail closed. The active design and admission gates are in
`akita-metal-e2e-structural-5x-goal.md` revision 4.

#### Revision-4 feasibility decision: rejected before protocol code

The bounded revision-4 audit is complete. The accepted trace exposes exactly
23.158291853 seconds of work that the representation can delete: 14.148838542
seconds of one-hot commit, 6.001420542 seconds of one-hot Akita evaluation,
0.514324292 seconds of Stage-6a/6b Booleanity preparation and address rounds,
1.908901644 seconds of causally exposed Stage-6b Booleanity round time, and
0.584806833 seconds of the Hamming-only Stage-7 wrapper. The mixed Stage-6b
credit comes from replaying each accepted host/accelerator join, not from
crediting the member's inclusive span. S15 and unrelated mixed-stage work receive
no credit. With a required 15.079061-second saving, the complete replacement may
cost at most 8.079230853 seconds; the most favorable CPU charge is zero.

Address zero does not require another committed presence bit. The current packed
object is the zero-lane-omitted polynomial
`D_r(t) = eq(r, a(t)) - eq(r, 0)`, and the existing Hamming claim reconstructs
`P_full = D + eq(r, 0) H`. This covers both a valid remapped RAM address zero and
an absent access without changing the memory argument.

The deterministic `(T28, 120 polynomials, bound 2)` planner requires 62,914,560
D512 source rings. Five-prime dense commitment entails about 724.776 billion NTT
butterflies plus 161.061 billion pointwise products. The retained exact D512
operation family sustains about 11.19 billion Montgomery products/s; even granting
an unmeasured 4x improvement puts the root alone at 19.79 seconds. The compulsory
120 `cS` plus 63 selector-product nodes add a 3.003-second ideal fp128 floor. This
22.79-second favorable envelope excludes source extraction, binds, range proof,
recursive opening, inverse/CRT work, synchronization, transcript, verifier, and
CPU effects, yet already exceeds the 8.079-second replacement ceiling by 14.71
seconds. Admission would require the root to run at 174.50 billion modular
products/s, 15.59x the measured exact D512 rate.

The byte-lifetime design itself fits: 30 GiB of compact source retires after the
root; the root successor is about 1.96 GiB; 84 post-prefix address factors use 42
GiB; nine linearly combined increment tables use 4.5 GiB; products are streamed
rather than materialized; and the conservative peak is about 84.1 GiB. Memory is
not the rejecting gate. The planned Akita payload also shrinks from the measured
268,861 bytes to 74,538 bytes, with an estimated 2,928-byte Jolt-side increase.

No measured constant straddles the decision, so the specification's prototype
exception does not apply. No `Radix4AddressV1` layout, transcript separator,
schedule, proof type, prover, verifier, or fallback behavior has been added. A
future revisit would need all of those, cross-version rejection, and a soundness
composition: fp128 currently uses a degree-one challenge field, and the naïve new
sumcheck/RLC delta is roughly `364 / |F|`, insufficient by itself to claim an
unchanged strict 128-bit target. P4c is rejected in Phase 1; OneHotV1 remains the
only branch protocol.

## Compatibility and review gates

The schedule and Jolt layout changes already have canonical digests. The current
Akita fork changes quotient-row semantics without a dedicated quotient-free
protocol/version tag (`AKITA_INSTANCE_DESCRIPTOR_VERSION` is still 1). Before
upstream integration, add an explicit relation-layout/protocol version to the
descriptor or transcript domain separation and test old/new cross-rejection; do
not rely only on proof-shape parsing failure.

Required protocol gates are:

1. CPU/Metal equality of commitments, evaluations, transcript state, proof
   bytes, and verifier result under one fixed protocol version.
2. Verifier tamper tests for layout and schedule digests, omitted-lane publics,
   stage-2 setup/challenge terms, and the outgoing-witness-before-`alpha` order.
3. Direct recurrence tests against negacyclic multiplication for every supported
   A dimension and both base/extension challenge fields.
4. K16 and K256 boundary tests for balanced increment reconstruction, including
   negative extrema and modular-headroom checks.
5. An explicit soundness-error ledger for quotient-free A batching and recursive
   levels, plus confirmation that the planner's SIS bounds are unchanged.
6. Standard and ZK non-regression proofs, since those modes must remain
   byte-identical to their non-Akita baselines.

Protocol ownership should follow the verifier boundary: Jolt committed-data
semantics live in `jolt-claims`/`jolt-verifier`; canonical schedule selection
lives in `jolt-akita`; quotient-free relation layout and checks live in Akita
types/prover/verifier. `akita-metal` implements those definitions but must not be
their only specification.
