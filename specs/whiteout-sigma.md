# Spec: Whiteout Σ-protocol, response statement, and lattice commitments (W4)

| Field     | Value                                                                                   |
| --------- | --------------------------------------------------------------------------------------- |
| Author(s) | @markosg04                                                                              |
| Created   | 2026-10-19                                                                              |
| Status    | design note, awaiting review                                                            |
| Parent    | `specs/shared-primitive-layer.md` (W4, §6, §7), `specs/whiteout-interfaces.md` (§3, §5) |
| Siblings  | `specs/whiteout-gaussian.md` (M5a), `specs/whiteout-masking.md` (M6b/M6c, on `wo/mask`) |

Design only. Paper labels refer to `akita-paper/sections/whiteout/` as read on
2026-10-19. Numbers the text omits come from `scripts/whiteout-costs.py`, at its
headline configuration (T = 2^24, k = 2, variant 1, base 16; the offloading mode
does not change any number used here). Akita references are to
`~/akita-whiteout` at `a0e3087ad`.

## Decision

W4 adds the lattice half of `jolt-zk` and the primitives it needs:

- **`jolt-crypto::lattice`** gains the Ajtai operation as a byte-exact
  definition of what Akita computes today. That covers matrix views cut from
  the seeded stream, balanced gadget digits, one- and two-level layouts, outer
  hiding randomness, and the binary compression chain. Akita keeps its kernels
  as the fast path. A test in `jolt-akita` pins the two to the same bytes.
- **`VectorCommitment`** gains an associated `Blind`. Row openings move to a
  `RowOpening` subtrait that only Pedersen implements. The lattice pad
  commitment implements the base trait. It commits the pads, the
  constant-term mask g, and the randomness r_p, plus the extra row
  **L**r_p that the Σ-protocol needs.
- **`jolt-zk::whiteout`** gets three pieces:
  - the Σ-protocol of Protocol `wo:prot:sigma`. It uses a Gaussian mask,
    joint rejection, internal retries keyed by secret 256-bit nonces, and
    constant-time handling of everything an attempt computes before its
    accept bit is known;
  - the `ResponseStatement` / `ResponseWitness` pair with the
    `ResponseProver` trait that Akita implements in W5;
  - Whiteout's `ZkBackend` impl over M6b's `ResidualRelation`.

Three choices change the interfaces of neighbouring tickets, and §9 lists them:

- product rows become sums of products;
- the residual relation gains a structured term for evaluating w_k;
- Akita's parameter tables, not floating point at runtime, supply the
  Gaussian widths and digit counts.

## 0. Notation and the private vector

`R = F[X]/(X^64 + 1)` with `F = Prime128OffsetA7F7`, `q = 2^128 − 2^32 + 22537`.
R is the challenge ring of §2.1, with d = 64. The private vector **s** of
Def. `wo:def:residual` is held by `jolt-zk::whiteout` as an ordered list of
**segments**. Each segment records:

| Field | Meaning |
|---|---|
| `role` | `Witness` (w_k), `InnerDigits` (t̂_a), `ResidualRandomness` (r_a), `PadDigits` (regular pads and g), `OperandDigits` (operand-pad digit polynomials), `PadRandomness` (r_p) |
| `len` | length in R-elements (multiples of 64 coefficients) |
| `native_dim` | ring dimension of the matrix that consumes it (a multiple of 64) |
| `digit_bound` | honest bound on `‖·‖∞`, which sets the shift bound `‖c‖₁·digit_bound` |
| `mask` | `Gaussian(block)` or `Public` |

A segment of native dimension d′ = 64m is viewed as a vector over R by
restriction of scalars along R ↪ F[Y]/(Y^{d′}+1), X ↦ Y^m (§2.1, "Several ring
dimensions"). The R-coordinates of a native element a = Σ_{k<m} Y^k a_k(Y^m)
are a_0, …, a_{m−1}, so coefficient j = m·i + k of a is coefficient i of a_k
(stride packing). Multiplication by c ∈ R is multiplication by c(Y^m) in the
native ring, so `c·s` never repacks. Repacking happens only where the
constant-term lift needs an R-element view, inside `LiftedFunctional`
construction.

## 1. Ajtai commitments in `jolt-crypto::lattice`

### 1.1 Matrix views

Every Ajtai matrix is an **offset-0 prefix** of the one stream
`derive_public_matrix_prefix(·, &PublicMatrixSeed)` (moved in M5b), viewed
row-major with ring-contiguous entries:

```
entry(row, col) = stream[(row · width + col) · d .. +d]   (coefficients in order)
```

This matches Akita's `FlatMatrix::ring_view_dyn` (`akita-params/src/layout/flat_matrix.rs:96-100`).
Consequences that the layout type must preserve:

- Different roles overlap. A, B, D, K₁ and K₂ all start at offset 0, and
  setup sizing takes the maximum footprint, not the sum
  (`setup_projection.rs:62-90`). Stage-3 setup offloading depends on this.
- A view is a function of `(rows, width, d)`. Changing the width changes every
  row after row 0, so appending randomness columns to an outer map yields a
  different matrix. With zero randomness columns the bytes equal today's.
- The length check is the caller's. `check_public_matrix_prefix` accepts a
  truncated slice, so every `MatrixView` constructor checks
  `rows·width·d ≤ len` and returns `CryptoError::InvalidSetup` otherwise.

```rust
pub struct MatrixShape { pub rows: usize, pub width: usize, pub ring_dim: usize }
/// A borrowed view of an offset-0 prefix. Construction checks the length.
pub struct MatrixView<'a, F> { shape: MatrixShape, data: &'a [F] }
```

### 1.2 Gadget digits

`G⁻¹` is jolt-algebra's balanced power-of-two decomposition
(`ring/cyclotomic/decomposition.rs`): base `2^log_basis` and `levels` digits,
with digit 0 least significant. When `levels·log_basis` equals the field width
it uses the asymmetric threshold `T_k`, and otherwise `q/2`. jolt-crypto adds no
second decomposition. It names the parameters once:

```rust
pub struct GadgetDigits { pub log_basis: u32, pub levels: usize }
```

The **input column order** of an Ajtai level is ring-major, digit-minor:
column `ring · levels + digit` holds digit plane `digit` of input ring `ring`.
This matches Akita's inner commitment (`sources/block_digits.rs:252-273`).

### 1.3 The operation and layout type

```rust
/// One Ajtai map: y = M · (G⁻¹(x) ‖ r) over Z_q[X]/(X^d + 1).
pub struct AjtaiLevel {
    pub matrix: MatrixShape,        // rows = output rank, width = message + randomness columns
    pub digits: GadgetDigits,       // decomposition of the message
    pub message_rings: usize,       // ring elements of x before decomposition
    pub randomness_columns: usize,  // trailing columns acting on r (0 = binding only)
}

/// Akita's two-level commitment (inner A per block, outer B per slice).
pub struct TwoLevelLayout {
    pub blocks: BlockGeometry,      // live rings, rings per block M, live blocks
    pub num_polys: usize,
    pub inner: AjtaiLevel,          // A, randomness_columns = 0
    pub outer: AjtaiLevel,          // [B | B'], applied once per slice
    pub slices: SliceCount,         // 1, 2, 4 or 8; dyadic block split
}
```

The operation semantics are pinned to Akita (paths in `akita-cpu-backend` unless
noted):

- **Inner.** For each block, `t_i = A · G⁻¹(w_i)`, negacyclic in R_{d_A}
  (`arithmetic/commitment.rs:159-237`).
- **Decomposition for the outer level.** Each `t_i` is decomposed into outer
  digits. Each d_A ring splits into d_A/d_B subcolumns of consecutive
  coefficients, in the order `[A row][subcolumn][digit][coefficient]`
  (`kernels/linear/decompose.rs:53-136`).
- **Slices.** Slice inputs are polynomial-major. Each polynomial's segment is
  zero-padded to `max_blocks_per_slice · per_block`
  (`commitment/outer_slices.rs:6-73`).
- **Randomness.** The randomness of slice s occupies the last
  `randomness_columns` columns of that slice's input. This is the layout
  W1 implements. The paper fixes only "appended to the input of its outer
  map" (§4.1).
- **Output.** `u` is the concatenation of the slice outputs.

jolt-crypto exposes three operations:

- `apply(view, input: &[CyclotomicRing<F, d>])`: the linear map on
  arbitrary inputs. The Σ-protocol uses it on the Gaussian mask y.
- `apply_centered(view, input: &[i64])`: the same map on centred
  integers. It never builds field elements from secrets with a branch
  (§2.8).
- `commit(level, x, r)` and `commit_two_level(layout, w, r)`: the
  commitments.

Each operation returns `Result<_, CryptoError>` and checks its input lengths
against the layout in release builds.

`AjtaiLevel` and `TwoLevelLayout` are plain data that Akita's planner fills.
`TwoLevelLayout` has a `From` conversion from Akita's `GroupCommitPhaseParams`
on Akita's side, written in W5 or M8. jolt-crypto never imports Akita (I1).

### 1.4 Hiding randomness

The commit takes an explicit `r`; jolt-crypto does not own `r`'s length, which
is a layout field. It owns the sampler and the entropy rule:

- `sample_box_digits(rng, len, log_basis) -> Vec<i8>` draws uniform digits in
  the balanced alphabet `{−b/2, …, b/2 − 1}`. The draw is a constant-time
  mask of uniform bytes. Distinct vectors differ by a coordinate below
  `q^{1/4}/2`, so their difference is a unit (§2.3).
- `required_randomness_columns(image_coeffs, extra_coeffs, log_basis, slack_bits)`
  returns the smallest column count whose entropy is at least
  `(image_coeffs + extra_coeffs)·⌈log₂ q⌉ + 2λ + slack_bits`. Here
  `extra_coeffs = 64` for the pad commitment's row **L**, and
  `slack_bits = 8` (§6.2). The count is rounded up to whole ring elements.
  With a rank-3 pad matrix over R_256 this gives the script's 26,752 cells.

Each slice draws its own randomness (§4.1).

### 1.5 Compression chain

Every commitment that Whiteout sends travels in compressed form (§2.4). Two of
them, the pad commitment and t_D, are computed by `jolt-zk`, which cannot
depend on Akita (I1). So W4 also copies Akita's compression chain into
`jolt-crypto::lattice`, add-first and unchanged:

- exactly two rank-one maps K₁ and K₂, which are offset-0 views at ring
  dimensions [16, 8] for Q128;
- `PackedNegativeBinary` digits in `{−1, 0}`, bit-major
  (`linear = bit·len + coeff`);
- `width = ⌈in_coeffs·field_bits/d⌉`;
- a 128-byte terminal of 8 field elements;
- an input of at most 8 KiB (`akita-params/src/compression.rs`,
  `compression/chain.rs`).

```rust
pub struct CompressedCommitment<F>([F; 8]);     // CanonicalBytes, 128 bytes
pub struct CompressionWitness<F> { pub xi_1: Vec<i8>, pub xi_2: Vec<i8> }
pub fn compress<F>(seed: &PublicMatrixSeed, image: &[F])
    -> Result<(CompressedCommitment<F>, CompressionWitness<F>), CryptoError>;
```

Akita switching to this copy is a later Akita ticket (§9, A12).

### 1.6 Relation to Akita's kernels

- **Same bytes.** The jolt-crypto functions are the definition, written
  schoolbook or NTT and readable. Akita's kernels (`kernels/linear/*`,
  `arithmetic/exact_i16.rs`, `commitment_stage.rs`, `opaque/compression.rs`)
  are the fast path for the same function.
- **Pinning.** Akita has no commitment-byte golden vectors today. W4 adds
  two kinds of test:
  - frozen digests in jolt-crypto for small layouts: one level, two levels
    with 2 slices and 2 polynomials, randomness columns, and both
    compression maps;
  - an equivalence test in `jolt-akita/tests`. It runs Akita's production
    commit and compression paths through the pinned Akita on each
    production profile of the K16 schedules (root, levels 1–5) and on
    random inputs, and asserts byte equality with jolt-crypto.

  The equivalence test needs Akita's commit entry points to be reachable
  from `jolt-akita`. If they are not, a later Akita ticket adds the same
  test there (W5).
- **Kernel seam.** The prover's per-attempt `N·y` is the commitment of an
  11.8M-coefficient vector (§6.4). jolt-zk takes an implementation of:

  ```rust
  pub trait AjtaiKernel<F> {
      fn apply_centered(&self, view: &MatrixView<'_, F>, input: &[i64], out: &mut [F])
          -> Result<(), CryptoError>;
  }
  ```

  jolt-crypto provides `ReferenceKernel`. Akita may provide a fast one in
  W5, and `WhiteoutSetup` carries the choice. Akita's i8/i16 digit kernels
  do not apply to y, whose coefficients reach about 2^27. A fast kernel must
  satisfy the constant-time contract of §2.8 and pass the equivalence test.

### 1.7 Module structure

The Σ-protocol needs **N** to be R-linear (`wo:lem:sigma-extract` cancels c),
and restriction of scalars provides that only when each multiplication by a
matrix entry acts on an input embedded by the stride map of §0. Akita's
consecutive-coefficient split from d_A to d_B (§1.3) is a relabelling of
coefficients, not a module map. So for the residual commitment and the pad
commitment, `jolt-zk` rejects any layout with `d_A ≠ d_B`, or with a ring
dimension that is not a multiple of 64, with
`ZkError::Setup(NotModuleOverChallengeRing)`. The headline layouts use R_256 for
A, B and the pad matrix, so they pass. Other folds and the proof of the response
are unaffected.

### 1.8 `VectorCommitment::Blind`

This implements `whiteout-interfaces.md` §3.

```rust
pub trait VectorCommitment:
    Commitment<Output: CanonicalBytes + CanonicalDecode + Serialize + DeserializeOwned>
{
    type Field: JoltField;
    type Setup: Clone + Send + Sync;
    /// Pedersen: one scalar. Lattice: randomness digits.
    type Blind: Clone + Send + Sync;
    fn capacity(setup: &Self::Setup) -> usize;
    fn sample_blind<R: RngCore + CryptoRng>(setup: &Self::Setup, rng: &mut R) -> Self::Blind;
    /// `values` are field elements; any gadget decomposition happens inside.
    fn commit(setup: &Self::Setup, values: &[Self::Field], blind: &Self::Blind) -> Self::Output;
    fn verify(setup: &Self::Setup, c: &Self::Output, values: &[Self::Field], blind: &Self::Blind) -> bool;
}

/// Row-combined openings: needs a homomorphic blind of norm independent of
/// the combination, so only Pedersen implements it.
pub trait RowOpening: VectorCommitment<Blind = Self::Field>
where Self::Output: HomomorphicCommitment<Self::Field> + Copy
{ /* open_committed_rows, verify_committed_rows: today's bodies */ }
```

- The `Copy` bound moves from `VectorCommitment::Output` to `RowOpening`
  (`combine_commitments` needs it). Every current caller of
  `open_committed_rows`, `verify_committed_rows` or `HomomorphicCommitment`
  bounds `RowOpening`.
- `Pedersen` and `NoVectorCommitment` set `Blind = F`. The lattice output is
  the 128-byte `CompressedCommitment`, which is fixed size, so
  `CanonicalBytes` holds.
- Interaction with M6c: M6c rewrites the BlindFold code that calls these
  traits. This slice lands after M6c, or rebases onto it, so the change is
  made once.

### 1.9 The lattice pad commitment

`whiteout::LatticePadCommitment` implements `VectorCommitment`, with
`Setup = PadCommitmentSetup` built from M6b's `PadLayout`.

**Committed values** (`commit_pads`, interfaces §5): `values` is the pad vector
p ∈ F^P in pad-index order, then the 64 coefficients of g. The setup knows
which pad indices are operand pads (`PadLayout::operand_pads`). The message
the matrix sees, in R-elements, is:

1. **Regular pads, then g.** Grouped into R-elements of 64 consecutive values
   (the last group zero-padded). g is aligned to its own element. Each
   element is decomposed into 32 balanced radix-16 digit planes (§1.2), so
   `p_i = Σ_j 16^j · digit_{i,j}`, and `g = Σ_j 16^j g_j` is R-linear in s.
2. **Operand pads.** For each operand pad p_x in `operand_pads` order:
   16 constant R-elements p_{x,j}, which are the balanced radix-256 digits of
   p_x, with `P_x = Σ_j 256^j p_{x,j}` (§3.2). An operand pad is also
   committed in group 1 only if it is also a regular slot. M6b's A5 decides
   that. The reconstruction rule is the same either way.
3. **r_p.** The blind: radix-16 box digits with the length from §1.4.

The message is packed into R_{d_p} elements by the stride map (§0) and
committed by one `AjtaiLevel`, `t_p = A_p · (message ‖ r_p)`. `Output` is the
compressed form of t_p.

**The row L.** The paper needs `[A_{p,r} ; L]` to be one uniform matrix
instance (§4.2; H4 hypothesis). The pad level's view therefore has
`rank + 1` rows. Row `rank` is never part of the commitment. Write `a_j` for
its entries on the randomness columns. Then

```
L · r_p := π_0( Σ_j a_j · r_{p,j} ) ∈ R,
```

where π_0 takes the k = 0 stride component (§0). For a uniform `a`, the map
`r ↦ π_0(a·r)` is a uniform R-linear map on the four R-coordinates of r: the
map `a ↦ π_0(a ·)` is an F-linear bijection between spaces of dimension 256.
So the stacked matrix is uniform. This definition is W4's; §9 A8 asks the
authors to confirm it.

## 2. The Σ-protocol

### 2.1 Inputs

`whiteout::prove` receives the following.

- From M6b: the `ResidualRelation` and the `CommittedPads`.
- From the Akita fold-k prover (W3): the residual opening `(w_k, t̂_a, r_a)`
  with its image `u_a`. It also receives the public segments of w_k (§2.7).
- From `WhiteoutSetup`: the layouts, `SigmaParams` (§5) and the
  `ResponseProver`.

It assembles **s** in segment order (§0): w_k, t̂_a, r_a, regular and g digits,
operand digits, r_p.

**Release self-checks** run before any transcript operation. They cost one
application of N and one evaluation of h:

- `N·s = t`. A failure returns `ZkError::ResidualOpeningMismatch`.
- `ct(h) = 0`. A failure returns `ZkError::ResidualUnsatisfied`.

The second check also protects zero knowledge. h is a coordinate of V, V lies
in e, and the proof of the response is not zero knowledge. If `ct(h) ≠ 0`, h
would leak the violated rows, because H6 needs `ct(Q′) = 0`.

### 2.2 Lifting the rows

Each row becomes an R-linear form in `(u, s)` (`wo:lem:ct-lift`), built with
`LiftedFunctional` (moved in M6d).

| Residual element | Lift |
|---|---|
| `ResidualVar::Pad(i)`, regular | `Σ_j 16^j ·` coordinate of digit `(i, j)` |
| `ResidualVar::Pad(x)`, operand | `ct(P_x)`, i.e. `Σ_j 256^j ·` coefficient 0 of `p_{x,j}` (§3.2: "every use of this pad uses ct(P_x)") |
| `ResidualVar::Witness(j)` | coordinate j of w_k |
| structured evaluation term (A4) | `scale · Σ_j eq(r, j) w_{k,j}`, kept symbolic |
| constant `b` | `u · b` |

An `AffineRow` becomes a constant-term row `Λ_j`. Each operand pad adds the 63
**scalarity rows** `ct(X^{−r} P_x) = 0`, r = 1..63. A `ProductRow` becomes

- `X_i(u, s) = u·x̂_i − P_{x_i}(s)`, the constant operand;
- `Y_i`, `Z_i`, the lifts of y and z.

M6b's `ProductRow` has one x. W4 asks for a sum-of-products row (A1), which this
construction already supports. Each term contributes its own `X_{i,t} Y_{i,t}`,
and all terms share the row's label.

### 2.3 Labels and aggregation (Step 1)

Labels are assigned by one public function of the relation, `RowLabels::assign`
(§4.2, "Verifier cost of the zero rows"):

- Operand o (in `operand_pads` order) owns the labels `64·o + r` for
  r ∈ [1, 63]. Label `64·o` is unused, and label 0 is the constant monomial
  reserved for `ct(g)`.
- Product rows follow from label `64·n_ops`, in relation order, and affine
  rows come after them.
- `ℓ = ⌈log₂(max_label + 1)⌉`. At the headline shape the maximum label is
  9,625 (119 operands, 119 products, 1,891 affine rows), so ℓ = 14.

The verifier squeezes ℓ **full** field challenges ρ_1..ρ_ℓ with
`challenge::<F>()`, consecutively and with no intervening prover message.
`challenge_small` is not used, because the error term is ℓ/q. The weights are
`μ_i = Π_{j : bit j of a_i} ρ_j`. Both roles then form

```
Q_μ(u, s) = Σ_i μ_i (X_i Y_i − u Z_i) + u · Λ_μ(u, s) + u · g(s),
Λ_μ = Σ_j μ_{n×+j} Λ_j     (one LiftedFunctional: a combination of lifts is the lift of the combination)
```

The prover computes `h = Q_μ(1, s)` once.

### 2.4 Attempts (Steps 2–4)

The prover first squeezes a 32-byte fork seed σ from the live transcript. The
verifier squeezes it at the same position. σ binds the transcript τ of the
oracle input. Then, for attempt i = 1..T_att:

1. **Mask.** Sample y ← π_0 with `GaussianSampler::fill`, one sampler per
   block, writing y = 0 on Public segments.
2. **First message.**
   - `f = N·y`, through `AjtaiKernel::apply_centered`;
   - `E_0 = L·y + Q_μ(0, y)`;
   - `E_1 = L·s + B_μ((1, s), (0, y))`. Expanded, this is
     `L·s + Σ_i μ_i (X_i(1,s) Y_i(0,y) + X_i(0,y) Y_i(1,s) − Z_i(0,y)) + Λ_μ(0,y) + g(y)`.
3. **Commit to V.** Set `V = (f, E_0, E_1, h)`, take `t_D = A_D · G⁻¹(V)` with
   radix-16 digits (32 per coefficient) and no randomness, and compress it to
   `cm_D` (§1.5).
4. **Nonce and challenge.**
   - Draw `n_i` as 32 uniform bytes from the prover's RNG. It is never a
     counter or a derived value.
   - Take `c_i = SampleChallenge(Fork::attempt(σ, n_i).absorb(cm_D))`.
     This uses the operator-norm-filtered d = 64 family
     (`D64_L2_OP_NORM_PM1/PM2_COUNT`, `OperatorNormRejection::D64_SELECTIVE_L2`)
     through `sample_indexed_challenges_from_seed` on a 32-byte squeeze of
     the fork.
5. **Respond and reject.**
   - Compute `v = c_i·s` as the exact centred-integer negacyclic product,
     never reduced mod q, and set `z = y + v`.
   - Call `JointRejection::accept(z_G, v_G, rng)` on the Gaussian
     coordinates only, block-major.
6. **Outcome.**
   - Accept: send cm_D, send n_i as 32 fixed-length bytes, and leave the
     loop.
   - Reject: discard everything the attempt produced and continue.

`Fork::attempt` is new in `jolt-transcript`. It constructs a fresh sponge with
the tag `jolt-transcript/fork/attempt/v1`, then absorbs σ and the 32-byte nonce.
It is separate from the grinding forks, which are keyed by a `u32` counter, so
the attempt queries live in their own oracle domain (H2 hypothesis: "a separate
domain of the oracle"). `Fork::new(seed, counter)` cannot serve here, because a
counter is predictable and H2 needs a nonce that is uniform and secret until it
is published.

### 2.5 What enters the transcript

| Order | Event | Prover | Verifier |
|---|---|---|---|
| (earlier) | pad commitment `cm(t_p)` | sent by `commit_pads` | received by `receive_pad_commitment` |
| (earlier) | residual commitment `cm(u_a)` | sent by Akita fold k (W3) | received in W3 |
| 1 | ρ_1..ρ_ℓ | `challenge::<F>()` × ℓ | same |
| 2 | σ | `challenge_bytes::<32>()` | same |
| 3 | `cm_D` (128 bytes) | accepted attempt only | `receive` |
| 4 | `n` (32 bytes) | accepted attempt only | `receive_bytes(32)` |
| — | c | not a transcript event: derived from the fork | rebuilt from (σ, n, cm_D) |
| 5… | response proof | `ResponseProver::prove_response` | `verify_response` |

The transcript absorbs `cm_D` and `n` as ordinary messages, so every later
challenge binds them. A rejected attempt leaves no trace: no message, no site
and no telemetry value. Only the attempt count may be recorded. H3 makes it
independent of the witness.

### 2.6 Retry versus abort

| Event | Handling | Why |
|---|---|---|
| accept bit 0 | internal retry, fresh y and n_i | Step 4; Fiat–Shamir with aborts (§2.6 of the paper) |
| all T_att attempts rejected | `ZkError::SigmaAttemptsExhausted`; the caller may rerun the whole proof with fresh randomness | probability ≤ 2^−128; H3 makes the event independent of the witness, so a rerun reveals nothing |
| `SampleError::ShiftOutOfRange` | `ZkError::InvariantViolation`, fatal; never retried, never recorded | only a malformed s produces it (interfaces §5) |
| other `SampleError` / layout errors | `ZkError::Setup(..)`, fatal | public parameters |
| operator-norm challenge budget exhausted (`CryptoError`) | fatal on both sides | a public function of the oracle input, probability about 2^−1580 at 4,096 tries; never a retry trigger, since retrying on it would condition c |
| self-check failure (§2.1) | fatal | invalid witness |
| `ZkError::FoldOverflow` (W2) | whole-proof retry | outside the Σ-protocol |

### 2.7 Public coordinates

Public segments are declared, never inferred. Each one is a whole number of
R-elements (§4.2: "Public coordinates are whole elements of the challenge
ring"), and its ranges come from the residual layout. At k = 2 they are the
6,291,456 setup-group cells of w_2 and the 589,824 inner digits of fully
public blocks. On them:

- y = 0 (§4.2);
- they are excluded from every rejection block, from π_0 and from
  `‖·‖_σ` (§5, standing notation);
- `z = c·s_pub` gets fewer digits: D_pub = 3 at radix 16, since
  `‖c‖₁ · 8 = 424 ≤ 7·(16³−1)/15`.

The response proof checks them with the same equations and no
`z = c·s_pub` rows. Every branch on "public" reads only the segment table.

### 2.8 Constant-time requirements

The ZK theorem's view is the proof. The constant-time requirement adds one
side channel that the paper's model excludes: per-attempt timing. A rejected
attempt's values (y_i, c_i, z_i and V_i) are correlated with the witness
through the rejection event, which is why H2 insists that their oracle answers
are never observed. Timing must not reveal them either. The prover as a whole
is not constant time with respect to s, as no Jolt prover is, and that timing
is the same in every attempt. The rule is therefore:

> Every step computed before the attempt's accept bit is known is constant time
> with respect to the attempt's fresh secrets (y, n, c, and everything derived
> from them). Steps on s alone, and steps after acceptance, inherit the
> prover's ordinary posture.

| Step | Secret inputs | Requirement |
|---|---|---|
| Gaussian `fill` | output y | M5a sampler (isochronous, reviewed) |
| `N·y` | y | `apply_centered`: loop bounds and addresses from the layout; centred-to-field lift by constant-time select; NTT/CRT butterflies on public twiddles |
| `Q_μ(0,y)`, `B_μ`, `L·y`, `g(y)` | y | ring products with public or s-only operands; fixed loops |
| `G⁻¹(V)` | V | the branch-free bias path of `BalancedDecomposePow2Params::biased`; its `match` reads public parameters only |
| `A_D·G⁻¹(V)`, `compress(t_D)` | V, t_D | fixed loops; `PackedNegativeBinary` digit extraction must be branch-free (audit when copying, §1.5) |
| `Fork::attempt`, Keccak | n, cm_D | Keccak-f is constant time |
| challenge sampling | c_i | the accepted candidate's processing must not branch or index on its positions or signs; the discarded-candidate count is independent of the accepted value (rejection-sampling isochrony). `position_sample.rs` must be audited and, if needed, given a constant-time prover variant with identical output (A13) |
| `v = c·s` | c, s | dense 64×64 negacyclic convolution, with c as 64 `i8` coefficients; never a sparse loop over c's support |
| `accept` | z, v | M5a's constant-time rule; only the `Choice` is branched on |
| field arithmetic | all of the above | `Fp128` add/sub/mul reduction must compile without secret-dependent branches; checked by the assembly review |

Further rules:

- Per-attempt buffers are allocated once and overwritten. No tracing span
  carries an attempt value.
- `#[inline(never)]` goes on the attempt body, so the assembly review covers
  the production monomorphization (M5a §5).
- **Platform.** The argument assumes data-independent timing of integer
  multiply, add and compare. W4 records this as an accepted platform
  assumption: neither arm64 DIT nor Intel DOITM is enabled. Enabling them
  needs `unsafe` system-register access, which this crate forbids, so that
  is a follow-up (A11).

## 3. Response statement and `ResponseProver`

### 3.1 The boundary

`jolt-zk` owns everything that defines the Σ-protocol's relation:

- s, N, Q_μ and L;
- the box and the certified alphabets;
- the vector e (layout and contents).

Akita (W5) owns how that relation is proved:

- committing to e;
- the product sumcheck (2a + 3 field elements);
- the row batching `H·e = b` at η;
- the fused degree-3 range-and-relation sumcheck;
- the evaluation of the structured weights A(**r**) and B(**r**);
- setup offloading;
- the final opening.

That split follows §4.3, "Evaluating the public weights": the description must
let the verifier compute A(**r**) and B(**r**) succinctly, and only Akita's
structured-weight machinery does that. The statement therefore carries
**typed, structured descriptions**, never dense matrices.

### 3.2 `ResponseStatement<F>`

The items follow the list in §4.3.

```rust
pub struct ResponseStatement<F> {
    // Σ-protocol transcript values
    pub challenge: SparseChallenge,            // c
    pub mu: Vec<F>,                            // μ, from ρ and the labels
    // the vector e
    pub layout: ResponseLayout,                // segments of e, below
    // Item 2: compression and V-commitment equations
    pub residual_commitment: CompressedCommitment<F>,  // cm(u_a)
    pub pad_commitment: CompressedCommitment<F>,       // cm(t_p)
    pub v_commitment: CompressedCommitment<F>,         // cm(t_D)
    pub v_level: AjtaiLevel,                   // A_D (binding, no randomness)
    pub seed: PublicMatrixSeed,
    // Item 3: N z = f + c t
    pub ring_linear: RingLinearStatement,      // residual TwoLevelLayout, pad AjtaiLevel, segment map of s
    // Item 4: Q_μ(c, z) + L z = E_0 + c E_1 + c² h
    pub quadratic: QuadraticStatement<F>,      // lifted rows (sparse terms + structured eval terms),
                                               // operand list, labels, L (row index of the pad view)
}
```

**`ResponseLayout`** lists e = (G_z⁻¹(z), G⁻¹(V), G⁻¹(θ), ξ_a, ξ_p, ξ_D) as
ordered segments. Each segment has a length, a radix, a digit count and an
alphabet (Item 1).

- **z.** One segment per s-segment, in s order. Within a segment the layout
  is digit-plane-major, which is the output order of jolt-algebra's
  decomposition. Radix 16; Gaussian blocks use D_b digits and public
  segments D_pub. The certified box `𝓑_cert` is the balanced D-digit range.
- **V.** 32 radix-16 digits per coefficient, the same digits that t_D
  commits. So t_D = A_D·G⁻¹(V) is a check on e itself, with no second copy.
- **θ.** 63 coefficients × 32 radix-16 digits (2,016 cells).
- **ξ_a, ξ_p, ξ_D.** Binary `{−1, 0}`, laid out in the bit-major
  `CompressionWitness` order (§1.5).

**`QuadraticStatement`** stores Λ_μ as sparse pad and auxiliary terms plus at
most a few structured evaluation terms over w_k (A4). The scalarity rows are
stored implicitly: one operand list and one label base. Products are stored as
`(x̂, operand, Y, Z)` with sparse lifts. With c, μ and these terms fixed,
`H_e` in Item 4 is affine in e (§4.3, "Reducing the products"), and Akita
derives the tables U_i and W_i from them.

Item 5 (`ct(h) = 0`) is a single coefficient row on the V segment. It needs no
field of its own.

### 3.3 `ResponseWitness<F>`

The witness carries `e: Vec<i8>`, the digit vector in `ResponseLayout` order,
which is the paper's e exactly. It also carries the accepted
`(z, V, θ)` in value form for the prover's product reduction. jolt-zk computes
everything in it:

- `G_z⁻¹(z)`;
- `G⁻¹(V)`, already formed for t_D;
- θ from (4), as the quotient of
  `Σ μ_i [X_i(c,z)][Y_i(c,z)] − [H_e]` by `X^64 + 1`. The degree is below 63
  by §4.3, and a nonzero remainder returns `ZkError::InvariantViolation`;
- ξ_a, from `compress(u_a)` with u_a supplied by W3;
- ξ_p and ξ_D, from `compress(t_p)` and `compress(t_D)`.

At the headline shape, e has 65,450,122 cells (2^25.96), and i8 storage takes
65 MB.

### 3.4 Trait

This is interfaces §5, unchanged:

```rust
pub trait ResponseProver<F> {
    fn prove_response<H: Sponge>(&self, statement: &ResponseStatement<F>,
        witness: &ResponseWitness<F>, transcript: &mut ProverTranscript<H>) -> Result<(), ZkError>;
    fn verify_response<H: Sponge>(&self, statement: &ResponseStatement<F>,
        transcript: &mut VerifierTranscript<'_, H>) -> Result<(), ZkError>;
}
```

Contract for implementors:

- The commitment to e is the first transcript event after n (§4.3: "the
  commitment to e precedes every challenge of this reduction").
- `verify_response` checks Items 1–5 for the whole of e, public z segments
  included.
- Errors map to `ZkError::Response(..)`.

W4 ships no production implementor. Tests use `CheckingResponseProver`, a test
support type that evaluates Items 1–5 directly on the witness and sends
nothing. It is the statement's own semantics, so it is not a second
implementation of an Akita rule.

## 4. Whiteout's `ZkBackend`

```rust
pub struct Whiteout<P>(PhantomData<P>);
impl<P: ResponseProver<F>> ZkBackend<F> for Whiteout<P> {
    type Setup = WhiteoutSetup<F, P>;   // seed, pad/V/residual layouts, SigmaParams, kernel, P
    type Blind = PadBlind;              // r_p digits
    type Witness = ResidualOpening<F>;  // w_k, t̂_a, r_a, u_a, public segment table (W3)
}
```

- **`commit_pads`.**
  1. Sample uniform pads p ∈ F^P and g with `sample_zero_constant_term`
     (moved in M6d).
  2. Draw r_p with `sample_blind`, and commit with `LatticePadCommitment`
     (§1.9).
  3. Send the 128-byte `cm(t_p)` before the first challenge.
  4. Return `CommittedPads { pads, cursor: 0, blind: (g, r_p) }`. g travels
     with the blind because it is committed but is never a pad slot.
- **`receive_pad_commitment`.** Receive `cm(t_p)` and return a view that
  carries it. M6b's A3 (`CommittedPadsView` parameterized by the commitment)
  is the shape W4 needs.
- **`prove`.**
  1. Check `pads.cursor == layout.num_pads`.
  2. Build s and the segment table (§0, §2.1), and run the self-checks.
  3. Lift (§2.2), label and squeeze ρ (§2.3), then run the attempts (§2.4).
  4. Build the statement and witness (§3), and call `prove_response`.
- **`verify`.**
  1. Check the setup's module structure (§1.7) and the relation's shape.
  2. Lift and label (§2.2–2.3). The verifier never applies N or Q_μ; it only
     builds their descriptions.
  3. Squeeze ρ and σ, receive `cm_D` and n, and derive c. A derivation error
     rejects.
  4. Build the statement and call `verify_response`.

  The residual commitment `cm(u_a)` comes from W3's `ResidualRows`. The pad
  commitment comes from the view, which needs the `verify` signature change
  in M6b's A3.

**Consuming `ResidualRelation`** (interfaces §5):

| Part of the relation | Becomes |
|---|---|
| `affine` | constant-term rows Λ_j (§2.2) |
| `products` | product rows (X, Y, Z) |
| `layout.operand_pads` | the operand group of the pad commitment, plus 63 scalarity rows each |
| `layout.num_pads` | must equal P |
| `num_witness` | must equal \|w_k\| |
| `layout.auxiliary_slots` | ordinary pads; their defining equalities arrive as affine rows |

The back end adds the ring-linear rows N·s = t itself, from its `Setup`
layouts and the residual opening:

- the inner rows `A_a w_{k,i} − G t̂_{a,i} = 0`;
- the outer rows `[B_a|B′_a](t̂_a, r_a) = u_a`;
- `A_p(…, r_p) = t_p`.

As the interfaces require, it never sees g as a row variable. g enters only
through Q_μ's `u·g(s)` term.

## 5. Parameters

| Parameter | Headline value | Lives in | Validated by |
|---|---|---|---|
| challenge family | d = 64, 31 × ±1, 11 × ±2, ‖c‖_op ≤ Γ = 18, 2^128.06 elements, κ₁ = 53 | `jolt-crypto::challenge` constants (M5b) | `SparseChallengeConfig::validate_min_entropy` at `SigmaParams::new` |
| α, ε | 16, 2^−137 (so M ≈ 2.37) | `SigmaParams` | `JointRejection::new` (M ≤ 4, bounds) |
| block widths k_b | 16,080,468 (w, t̂, operand digits); 1,005,030 (r_a, pads, r_p) | Akita config table keyed by schedule and k (A10), carried in `WhiteoutSetup` | jolt-zk: exact rational check of `Σ_b 2 ln2 · Γ² γ_b² / k_b² ≤ 1/α²`, with `γ_b² = len_b·64·digit_bound_b²` and ln 2 replaced by a rational upper bound |
| digit counts D_b, D_pub | 7, 3 at radix 16 | same table | jolt-zk: box ⊆ 𝓑_cert, and `D_pub` covers `κ₁ · digit_bound` |
| box L_b | positive end of the D_b-digit range, `7(16^{D_b}−1)/15` (A9) | derived | jolt-zk: `L_b ≥ ⌈τσ_b⌉`, τ for p_𝓑 ≥ 0.999, so p_𝓑 meets §6.2 |
| shift bound | `κ₁ · digit_bound_b` | derived from segments | `JointRejection::accept` (`ShiftOutOfRange`) |
| T_att | 163 | derived | jolt-zk: smallest T with `(1 − (p_𝓑 − ε)/M)^T ≤ 2^−128`, in integer arithmetic |
| ν | 256 | const | — |
| ℓ | 14 | derived from labels | `RowLabels::assign` |
| A_a, [B_a\|B′_a], A_p ranks | 3 × R_256 each (768 rows) | Akita planner and SIS tables (rule 3), in `WhiteoutSetup` | Akita (SIS estimator); jolt-zk: module structure (§1.7), view lengths |
| A_D | 1 × R_64, binding on radix-16 digits | Akita planner | Akita (SIS); jolt-crypto view length |
| randomness | r_a 24,704 and r_p 26,752 cells, radix 16 | lengths in the layouts | jolt-crypto `required_randomness_columns` (§1.4), checked by jolt-zk at setup |
| pad radices | 16 (32 digits), operands 256 (16 digits) | `PadCommitmentSetup` consts | `G⁻¹` parameter checks |
| compression | K₁ at R_16, K₂ at R_8, 128-byte terminal | `jolt-crypto::lattice` (copied) | the chain's own shape checks |

## 6. Soundness and zero knowledge, mapped to the paper

| Implementation choice | Paper item it discharges |
|---|---|
| pad commitment sent before the first challenge; `CommittedPads` is the only constructor of masking channels (M6b) | §4.1 "Pads committed in advance"; `wo:thm:knowledge` (pads common to all leaves) |
| ct(g) = 0 sampler and label 0 reserved | `wo:lem:sigma-complete`; `wo:lem:sigma-aggregate` (extracted g is not assumed scalar-free); H6 |
| distinct nonzero labels, ℓ full-field ρ | `wo:lem:sigma-aggregate`, error ℓ/q |
| c from the filtered family; differences are units (q ≡ 9 mod 16, ‖·‖∞ < q^{1/4}/2) | `wo:lem:sigma-extract`, 2/\|𝓒\| in `wo:thm:knowledge` |
| N is R-linear (§1.7) | §2.1 restriction of scalars; `wo:lem:sigma-extract` |
| 𝓑 ⊆ 𝓑_cert, with binding radius taken from 𝓑_cert | §4.3 "acceptance and certified boxes"; radius 4Γ_Δ B_M in `wo:thm:knowledge` |
| A_D without randomness | `wo:thm:knowledge` (A_D collision on digit vectors); H-sequence needs no hiding for t_D |
| `[A_{p,r}; L]` one offset-0 view, row `rank` | H4 hypothesis ("L is an extra row of the pad commitment's matrix") |
| randomness entropy ≥ image + 2λ + 8 bits, fresh per slice | `wo:lem:stat-hiding`; δ_LHL ≤ 2^−129.2 (§6.2) |
| exact integer v; shift check | `wo:def:joint-rejection` computed exactly; `‖cs‖_σ ≤ 1/α` validated (§5) gives `wo:thm:joint-rejection` |
| y = 0 on declared public segments only | §4.2 "Public coordinates"; standing notation of §5 |
| uniform secret 256-bit n_i; oracle input (σ ⊇ τ, cm_D, n_i) in its own fork domain; only the accepted attempt sent | H2: Bad ≤ (T_att Q_tot + T_att²)/2^256 |
| fresh y per attempt, rejection via M5a | H3: ε/M per attempt; attempt count independent of the witness |
| T_att = 163 | `wo:thm:completeness`; the `T_att ε/M` term of `wo:thm:zk` |
| ct(h) = 0 self-check | H6 precondition, enforced in release |
| sampler precision (M5a) | +2^−149 on `wo:thm:zk`'s 2^−128.8; §6.2 asks implementations to add it |
| ChaCha20 expansion of per-chunk seeds (M5a) | adds a PRG assumption to statistical ZK; stated in the module docs |

Not covered by this mapping:

- whether the decoded transcript is accepting. That is
  `wo:lem:residual-correct`, owned by M6b and W3 through `ResidualRows`;
- prefix soundness of the randomized folds (`wo:lem:randomized-base`, W1
  and W2).

## 7. Test plan

All suites run under `cargo nextest`. The synthetic relations are built in
`jolt-zk` tests from random pads and witness, with row counts and operand
counts scaled down (d = 64 is fixed).

1. **Completeness at the predicted abort rate.**
   - Run 2,000 honest proofs over `CheckingResponseProver`. All verify.
   - The mean attempt count must lie within a binomial 5σ band of
     `M/p_𝓑`, computed from the realized parameters.
   - A second configuration at α = 10 (M ≈ 3.99) moves the rate. That
     shows the test can tell different rates apart.
2. **Tamper rejection.** Each of the following is rejected:
   - flip one byte in each of `cm_D`, n, `cm(t_p)` and `cm(u_a)`;
   - substitute a rejected attempt's (cm_D, n);
   - reorder ρ;
   - a witness violating one affine row, one product row, one scalarity row
     (a non-constant operand pad), or `ct(g) ≠ 0`.

   Each violating witness must be rejected by the prover's self-check. With
   the self-check bypassed in a test-only path, the checking response prover
   must reject it.
3. **Statistics.**
   - Accepted z per Gaussian block: KS and χ² against π_0(·|𝓑).
   - Accepted z is independent of s: a χ² test of homogeneity between two
     different witnesses with equal public shape.
   - Rejection over enumerable toy blocks reuses M5a's harness; W4 does not
     re-test the sampler.
   - The nonce distribution passes a byte-level χ² test.
4. **dudect.** The attempt body joins `examples/sample_dudect.rs`. The test
   uses fixed versus random (y seed, c), with s fixed, and passes at
   |t| < 4.5. The `cargo asm` review of the production monomorphization of
   `attempt`, `apply_centered`, the c·s convolution and the challenge
   sampler's prover path is recorded in the PR (M5a §5).
5. **Ajtai equivalence.**
   - Frozen digests in jolt-crypto (§1.6).
   - The `jolt-akita` test compares jolt-crypto with Akita's production
     kernels and compression on every K16 profile.
   - A property test checks `apply_centered = apply ∘ lift`.
   - Another checks that the view-length rejection fires on truncated
     streams.
6. **Algebra.**
   - The polar-form identity `Q(c,z) + Lz = E_0 + cE_1 + c²h` holds on random
     data (`wo:lem:sigma-complete` as a property).
   - θ satisfies (4) exactly.
   - `R`-linearity of N under the stride map. The test checks
     `N(c·s) = c·N(s)` on random layouts and that a `d_A ≠ d_B` layout
     fails the module check.
7. **Interface.**
   - `VectorCommitment::Blind`: the Pedersen suites are unchanged.
   - The lattice pad commitment satisfies `verify(commit(v, b), v, b)` and
     rejects a changed value or blind.

## 8. Implementation slices (W4 commits, in order)

| # | Slice | Depends on |
|---|---|---|
| 1 | `jolt-crypto::lattice`: `MatrixShape`/`MatrixView`, `GadgetDigits`, `AjtaiLevel`, `TwoLevelLayout`, `apply`/`apply_centered`/`commit`, randomness sampler and entropy rule, `AjtaiKernel` + `ReferenceKernel`, frozen digests | M5b (landed) |
| 2 | `jolt-crypto::lattice`: compression chain copy, digests | 1 |
| 3 | `jolt-akita` equivalence test against Akita kernels | 1, 2; pinned Akita exposes commit entry points |
| 4 | `jolt-transcript`: `Fork::attempt` (32-byte nonce, own tag) | — |
| 5 | `jolt-crypto`: `VectorCommitment::Blind`, `sample_blind`, `RowOpening`; Pedersen and callers | **M6c** (rewrites the callers) |
| 6 | `jolt-zk::whiteout`: segment table, `SigmaParams` (validation of §5), `RowLabels` | M5a, M6d |
| 7 | `jolt-zk::whiteout`: lifting and Q_μ/B_μ evaluation | **M6b** `ResidualRelation` incl. A1 (sum-of-products) and A4 (structured w_k term) |
| 8 | `jolt-zk::whiteout`: `LatticePadCommitment` | 1, 2, 5; **M6b** `PadLayout::operand_pads` (A5) |
| 9 | `jolt-zk::whiteout`: Σ-protocol prover/verifier core, constant-time c·s, statistics tests | 4, 6, 7, 8 |
| 10 | `ResponseStatement`/`ResponseWitness`/`ResponseProver`, θ, e layout, `CheckingResponseProver` (test support), completeness and tamper tests | 9 |
| 11 | `ZkBackend for Whiteout<P>` | 10; **M6b** final `ZkBackend` signatures (A3 view, `verify` arguments) |
| 12 | dudect extension and asm review record | 9 |

Slices 1–4 and 6 do not depend on M6b and can start now. Slices 7, 8 and 11
consume shapes that M6b is still designing. Slice 5 is ordered after M6c to
avoid editing the BlindFold callers twice.

## 9. Ambiguity register

**Open questions from `whiteout-interfaces.md`:**

- **A1. Extension-field products.**
  - Problem: a product of extension elements has `E::DEGREE²` base-field
    cross terms. With M6b's one-operand `ProductRow`, each needs an
    auxiliary slot.
  - Proposal: `ProductRow { terms: Vec<(MaskedOperand, AffineRow)>, z: AffineRow }`,
    i.e. `Σ_t x_t·y_t = z` with one constant operand per term. Q_μ supports
    it at no extra cost (§2.2), and the product sumcheck counts terms.
  - Owner: M6b/W3. Whether W3's fold-k checks contain such products is still
    W3's question.
- **A2. `response_l2_sq` cut side.**
  - If it falls after fold k, it disappears with the suffix and W4 sees
    nothing.
  - If it falls before, W3 must give it a field encoding, and it reaches W4
    as an ordinary masked slot with affine rows.
  - Neither case changes this design.
- **A3. Stage-3 setup-offloading claims.** These are W3's routing question. W4
  needs only one output from it: the **declared public segments** of w_k
  (setup-group data), at whole-R-element granularity in the stride view
  (§2.7). Owner: W3.

**Dependencies on M6b's shapes:**

- **A4. Structured witness terms.**
  - Problem: fold k's final claim `w̃_k(r) = v` touches all 11.8M
    coordinates of w_k. `ResidualVar::Witness(j)` with explicit
    coefficients cannot express it succinctly for the verifier.
  - W4 needs `AffineRow` to admit a term `Eval { point, scale }` meaning
    `scale · w̃_k(point)`, lifted symbolically (§2.2) and passed through
    `QuadraticStatement` to Akita's structured weights.
  - Owner: M6b, with W3 as producer.
- **A5. `PadLayout::operand_pads` producer** (M6b A5). W4 is its first
  consumer.
  - Proposal: derive it from `products[].x` (and from the terms, under A1).
  - It also needs a ruling on whether an operand pad also occupies a
    regular slot (§1.9, group 1).
- **A6. `CommittedPadsView` and `verify` arguments** (M6b A3). W4 needs the
  view to carry `cm(t_p)` and `verify` to receive `cm(u_a)`.

**Paper gaps and decisions that need the authors:**

- **A7. Module structure.** §2.1's restriction of scalars requires the
  stride embedding. Akita's consecutive d_A → d_B split is not a module map.
  W4 requires `d_A = d_B` for the residual and pad commitments (§1.7). The
  paper should state the condition. Confirm that the R_256/R_256 headline is
  intended.
- **A8. Definition of L.** The paper says "one extra row … acts only on r_p",
  with output in R_64 and the matrix over R_256. W4 defines
  `L·r := π_0(row_rank · r)` on the (rank+1)-row view (§1.9). Confirm,
  including that offset-0 overlap of A_p's randomness columns with other
  roles' matrices is acceptable for H4 and H8. The LHL statement allows side
  information that depends on the matrix, so W4 believes it is.
- **A9. Box L_b** (M5a Q2). W4 takes the largest box inside the certified
  alphabet, the positive end of the balanced D_b-digit range. This
  maximizes p_𝓑 and removes a parameter. Confirm.
- **A10. Where widths come from** (M5a Q3).
  - Problem: the script's discrete width optimization uses floating point.
    The verifier needs D_b to build the e layout, so the values must be
    bit-identical on both roles.
  - Decision: k_b, D_b and D_pub are integers in an Akita config table
    generated offline by the script. jolt-zk validates them with exact
    rational arithmetic.
  - Confirm that the paper is satisfied with tabulated widths. The table
    keying (schedule, k) needs an Akita owner.
- **A11. Platform constant time.** DIT/DOITM are not enabled (§2.8). Accept
  this as a documented assumption, or schedule a follow-up with a narrowly
  scoped `unsafe` system-register wrapper outside the verifier closure.
- **A12. Compression ownership.** I1 forces the chain into `jolt-crypto` (§1.5),
  which goes beyond the W4 scope in the parent spec. An Akita ticket that
  switches to it and deletes `akita-params`' copy is needed, analogous to
  M5c. Owner ruling.
- **A13. Constant-time challenge sampling.** `position_sample.rs` and the
  operator-norm loop were written for a public-coin verifier. Whether the
  accepted candidate's processing is constant time is unaudited. If it is
  not, W4 adds a prover-side constant-time variant with byte-identical
  output.
- **A14. Whole-proof rerun after `SigmaAttemptsExhausted`.** It is safe by H3,
  and the paper's simulator outputs ⊥ with the same probability. Confirm that
  outputting ⊥ (an error) is the intended behaviour, rather than an
  unbounded loop.
- **A15. Owner of θ and e.** W4 places them in jolt-zk (§3.3), because they
  are functions of Q_μ and the Σ-protocol layout. W5 could instead own e's
  digit layout to match Akita's witness packing. Then `ResponseWitness`
  would carry values, not digits. Decide before slice 10.
- **A16. Per-attempt cost of N·y.**
  - The cost script charges N·y per attempt (2.37 expected). With y
    centred at ~2^27, Akita's i8/i16 kernels do not apply, and the
    reference NTT path on 11.8M coefficients is unmeasured.
  - A fast `AjtaiKernel` is a W5 option.
  - The prover-time claim (+4.7%) is unverified until measured.

## Not verified by this note

- Akita's commitment semantics are taken from code reading (the explorer
  survey's citations). No byte comparison has been run yet; slice 3 does
  that.
- The headline numbers (labels, ℓ, cell counts) come from the cost script's
  model of the Jolt and Akita schedules, not from a run of M6b or W3.
- The constant-time status of the field reduction, the compression digit
  extraction and the challenge sampler is unaudited (§2.8, A13).
