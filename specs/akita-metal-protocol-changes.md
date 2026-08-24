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
