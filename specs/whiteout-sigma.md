# Spec: Whiteout Σ-protocol, response statement, and lattice commitments (W4)

| Field     | Value                                                                                   |
| --------- | --------------------------------------------------------------------------------------- |
| Author(s) | @markosg04                                                                              |
| Created   | 2026-10-19                                                                              |
| Status    | design note, revised after crypto and architecture review                               |
| Parent    | `specs/shared-primitive-layer.md` (W4, §6, §7), `specs/whiteout-interfaces.md` (§3, §5) |
| Siblings  | `specs/whiteout-gaussian.md` (M5a), `specs/whiteout-masking.md` (M6b/M6c)               |

This note is a design only. Paper labels refer to
`akita-paper/sections/whiteout/` as read on 2026-10-19. Numbers that the text
omits come from `scripts/whiteout-costs.py` at its headline configuration (T =
2^24, k = 2, variant 1, base 16). Akita references are to `~/akita-whiteout` at
`a0e3087ad`.

## Decision

W4 builds the lattice half of `jolt-zk` and the primitives it needs.

- **`jolt-crypto::lattice`** gains the Ajtai map as a plain, constant-time
  function on jolt-algebra's CRT-NTT, with its matrix views and layouts:
  - **Matrix views** are cut from the seeded stream and length-checked in
    release builds.
  - **Gadget digits** are balanced, using one decomposition.
  - **`AjtaiLevel`** is a one-level commitment with hiding randomness
    columns.
  - **`TwoLevelLayout`** describes the residual commitment's part of N. It
    is a description only: Akita computes that commitment.
  - **Box sampling and the randomness entropy rule.**

  The layout defines the hiding columns that W1 adds. With zero randomness
  columns its bytes equal Akita's today.
- **`jolt-zk::whiteout`** gets the following:
  - **The Σ-protocol** of Protocol `wo:prot:sigma`. It uses a Gaussian mask,
    joint rejection, internal retries keyed by secret 256-bit nonces, and a
    constant-time challenge sampler on the prover side.
  - **Typed `ResponseStatement` and `ResponseWitness`**, and the
    `ResponseProver` trait that Akita implements in W5.
  - **`LatticePadCommitment`**, which produces `ZkBackend`'s `PadCommitment`
    and `PadOpening`.

  `ZkBackend for Whiteout` moves to W6, because it needs W3's residual
  opening type.
- **`AttemptFork` in `jolt-zk::whiteout::transport`**: the oracle domain of
  the attempt challenges.

**Out of W4:**

- `VectorCommitment` keeps its scalar blind. The lattice pad commitment is
  not a `VectorCommitment`.
- The compression chain moves to W5, together with the Akita equivalence test
  and any fast kernels. W4 sends t_p and t_D uncompressed (A12).

## 0. Notation and the private vector

`R = F[X]/(X^64 + 1)` with `F = Prime128Offset6A23801F` and `q = 2^128 − 1780711455`. R is the challenge ring of §2.1, with d = 64.

`jolt-zk::whiteout` holds the private vector **s** of Def. `wo:def:residual` as
an ordered list of **segments**:

| Field | Meaning |
|---|---|
| `role` | `Witness` (w_k), `InnerDigits` (t̂_a), `ResidualRandomness` (r_a), `PadDigits` (regular pads and g), `OperandDigits` (operand-pad digit polynomials), `PadRandomness` (r_p) |
| `len` | Length in R-elements, each 64 coefficients. |
| `native_dim` | Ring dimension of the matrix that consumes the segment: a multiple of 64 and a power of two. |
| `alphabet` | Balanced radix `2^log_basis`, i.e. digits in `[−2^{lb−1}, 2^{lb−1} − 1]`. |
| `shape` | `Dense`, or `ConstantPoly`: every R-element is a constant polynomial. |
| `mask` | `Gaussian` or `Public`. |

Each segment has an **honest ℓ₂² bound**. With B = 2^{lb−1}, it is
`γ² = 64·len·B²` for `Dense` and `γ² = len·B²` for `ConstantPoly`. This is the
paper's worst-case honest norm, whose square root `γ_b` sets the widths
(§5). A `ConstantPoly` segment such as the operand digits has a much smaller
ℓ₂ norm than its ∞-norm suggests. That difference is why the widths of §6.2
pass the zero-knowledge precondition.

Coefficients are stored in native order. A native element of dimension
d′ = 64m is viewed as an m-tuple over R by restriction of scalars along
R ↪ F[Y]/(Y^{d′}+1), X ↦ Y^m (§2.1, "Several ring dimensions"):

- `a = Σ_{k<m} Y^k a_k(Y^m)`, so native coefficient j = m·i + k is
  coefficient i of a_k. This is stride packing.
- Multiplying by c ∈ R is multiplying by c(Y^m) in the native ring, so c·s
  never repacks.
- In native order, a constant R-element means that only native coefficients
  0..m of each native element can be nonzero.

**Public ranges** are unions of whole native R_{d′} elements. The segment
table checks this when it is built.

## 1. Ajtai commitments in `jolt-crypto::lattice`

### 1.1 Matrix views

Every Ajtai matrix is an **offset-0 prefix** of the one stream
`derive_public_matrix_prefix(·, &PublicMatrixSeed)` (moved in M5b). The prefix
is viewed row-major, and each ring entry occupies contiguous positions:

```
entry(row, col) = stream[(row · width + col) · d .. +d]   (coefficients in order)
```

This matches Akita's `FlatMatrix::ring_view_dyn`
(`akita-params/src/layout/flat_matrix.rs:96-100`). The layout types preserve
three consequences:

- **Roles overlap.** A, B, D, K₁ and K₂ all start at offset 0, and setup
  sizing takes the maximum footprint (`setup_projection.rs:62-90`). Stage-3
  offloading relies on this. Sizing the pad view includes row `rank`,
  which defines **L** (§1.6).
- **A view is a function of (rows, width, d).** Changing the width changes
  every row after row 0. Appending randomness columns therefore gives a
  different matrix, unless there are zero of them.
- **The caller checks the length.** `check_public_matrix_prefix` accepts a
  truncated slice. `MatrixView::new` checks `rows·width·d ≤ len` and returns
  `CryptoError::InvalidSize` otherwise, in release builds.

```rust
pub struct MatrixShape { pub rows: usize, pub width: usize, pub ring_dim: usize }
/// A borrowed view of an offset-0 prefix. Construction checks the length.
pub struct MatrixView<'a, F> { shape: MatrixShape, data: &'a [F] }
```

### 1.2 Gadget digits

`G⁻¹` is jolt-algebra's balanced power-of-two decomposition
(`ring/cyclotomic/decomposition.rs`): base `2^log_basis`, `levels` digits,
digit 0 least significant. It uses the threshold `T_k` when `levels·log_basis`
equals the field width, and `q/2` otherwise. It is branch-free on the field
path (`biased`). jolt-crypto adds no second decomposition and only names its
parameters:

```rust
pub struct GadgetDigits { pub log_basis: u32, pub levels: usize }
```

An Ajtai level orders its input columns ring-major and digit-minor. Column
`ring · levels + digit` is digit plane `digit` of input ring `ring`, which
matches Akita's inner commitment (`sources/block_digits.rs:252-273`).

### 1.3 The operation

```rust
/// One Ajtai map: y = M · (G⁻¹(x) ‖ r) over Z_q[X]/(X^d + 1).
pub struct AjtaiLevel {
    pub matrix: MatrixShape,        // rows = output rank; width = message + randomness columns
    pub digits: GadgetDigits,       // decomposition of the message
    pub message_rings: usize,       // ring elements of x before decomposition
    pub randomness_columns: usize,  // trailing columns acting on r (0 = binding only)
}
```

A view is prepared once as `PreparedMatrix<F, D>`, which checks
`shape.ring_dim == D`. It offers two operations:

- `apply_centered(x: &[i64], bound) -> Vec<CyclotomicRing<F, D>>`: the map
  `M·x` on centered integer columns with `|x| ≤ bound ≤ 2^62`. This is the
  Σ-protocol's `N·y`, and the core of `commit`. `bound` must be a hard bound,
  such as the sampler's tail cut, never a statistical one: an input above it
  returns an error, which for a secret input would be an abort correlated
  with the secret.
- `commit(level, prepared, x, r)`. It computes
  `apply_centered(G⁻¹(x) ‖ r, 128)`, where `r` is box digits.

A map on arbitrary field columns has no planned caller: every input of N is
a digit or a Gaussian coefficient.

**Implementation, one path for both.** The public matrix is prepared once
in CRT-NTT form. It uses jolt-algebra's `CrtNttParamSet<i32, 6, D>` over the
Q128 primes and `CyclotomicCrtNtt::from_ring`, which may branch because the
matrix is public. Each input column is lifted into Montgomery residues by
`NttPrime::from_canonical` on its centered value. That lift is a branch-free
Montgomery product and is exact for every `i32`. The column is then
transformed by the forward NTT and accumulated pointwise into each output row.

- **Exactness.** Products are reconstructed exactly over Z and reduced mod q.
  A row is chunked whenever `CrtCapacity::supports(width, bound)` fails, and
  the chunk results are added in F.
- **Reconstruction.** It uses a constant-time Garner that W4 adds next to
  the existing one in jolt-algebra. That Garner selects with masks instead of
  bool-to-int products, and lifts each digit into F without
  `F::from_i64`/`from_i128`, which branch on sign
  (`solinas/fp128.rs:1046`).
- **No lookup tables.** No path uses `DigitMontLut`, `CenteredMontLut`, or a
  value-dependent strategy such as `from_centered_coefficients`'s `max_abs`
  switch. The i8 path in Akita's `block_parallel.rs:28-64` indexes
  `DigitMontLut` by digit value, so it is not used for secrets. Neither is any
  other Akita kernel.
- **Input bound.** The bound `|x| ≤ bound` is checked over all inputs as one
  constant-time aggregate, and only the aggregate is branched on.
- **Blind (P7).** Every CRT accumulator starts from the image of a fixed
  public pseudorandom ring element `K`, and `K` is subtracted in F after the
  reconstruction, so the reconstruction never sees the all-zero product of
  an all-zero input or limb (§2.8, Record P7). Chunks keep one column of CRT
  capacity for `K`, since `|K| ≤ q/2`.
- **Limbs (§5.1).** A bound above `i32::MAX` splits each input by shifts
  into balanced 30-bit limbs (`x = Σ_j 2^{30j} x_j`), applies the map to
  each limb, and recombines in F. The limb count depends only on the public
  bound. At the headline the mask's tail cut `14k ≈ 2^39.5` takes two limbs,
  so `N·y` and `L·y` cost two passes.

`q ≡ 9 (mod 16)` gives no NTT over F_q for `X^256 + 1`. Schoolbook `N·y` would
cost seconds per attempt. CRT-NTT is therefore the definition and also the
production path.

### 1.4 `TwoLevelLayout`: description only

```rust
/// Akita's two-level commitment, as one part of N (no commit function).
pub struct TwoLevelLayout {
    pub blocks: BlockGeometry,      // live rings, rings per block M, live blocks
    pub num_polys: usize,
    pub inner: AjtaiLevel,          // A, randomness_columns = 0
    pub outer: AjtaiLevel,          // [B | B'], applied once per slice
    pub slices: SliceCount,         // 1, 2, 4 or 8; dyadic block split
}
```

The layout fixes the column semantics of N on the residual commitment. They
follow Akita, with paths in `akita-cpu-backend`:

- **Inner level.** Each block has `t_i = A·G⁻¹(w_i)`
  (`arithmetic/commitment.rs:159-237`).
- **Decomposition into the outer input.** Each d_A ring splits into
  d_A/d_B subcolumns of consecutive coefficients, ordered
  `[A row][subcolumn][digit][coefficient]`
  (`kernels/linear/decompose.rs:53-136`).
- **Slices.** Slice inputs are polynomial-major. Each polynomial segment is
  zero-padded to `max_blocks_per_slice · per_block`
  (`commitment/outer_slices.rs:6-73`).
- **Randomness.** Slice s's randomness occupies the last
  `randomness_columns` columns of that slice's input.

Akita has no hiding code yet, so jolt-crypto defines W1's hiding layout here.
The layout is byte-exact with today only for zero randomness columns. W4 has no
`commit_two_level`. Akita computes the residual commitment, and W5 tests its
kernels against these semantics (§7, item 5).

### 1.5 Hiding randomness

- `sample_box_digits(rng, len, log_basis) -> Vec<i8>`, which is prover-only
  and gated behind jolt-crypto's `sample` feature, draws uniform digits of
  the balanced alphabet. Each digit is a masked byte, so the draw is constant
  time. Distinct vectors then differ by a unit (§2.3: coordinates below
  `q^{1/4}/2`).
- `required_randomness_columns(image_coeffs, extra_coeffs, ring_dim, log_basis, slack_bits)`
  returns the smallest whole number of ring columns whose entropy is at least
  `(image_coeffs + extra_coeffs)·⌈log₂ q⌉ + 2λ + slack_bits`.

The extra term is `extra_coeffs = 64` for the pad commitment's row **L**
(r_p only, not r_a), and the slack is `slack_bits = 8` (§6.2). The columns
follow the matrices' rank. With the rank-5 matrices over R_256 of §5.1 this gives:

| Randomness | Columns | Cells | (rank 3, before §5.1) |
|---|---|---|---|
| r_p | 169 R_256 | 43,264 | 105, 26,880 |
| r_a | 161 R_256 | 41,216 | 97, 24,832 |

At the production pad commitment's D = 128, radix 16 and 128-bit field,
rank 8 needs `ceil(((8·128 + 64)·128 + 264) / 512) = 273` randomness
columns; rank 9 needs 305. `WhiteoutTail::pad_rank` includes L in each
iteration of the rank/width fixed point. Muldiv has rank 8 and width 489.
The empty layout has width 289, 117 operand pads have width 1225 at rank 8,
and 118 operand pads require rank 9 and width 1265.

Resizing the randomness with the rank keeps every slice at 8 bits of
entropy beyond its image plus 2λ, so δ_LHL stays at 2^−129.2 (§6). Each
slice draws its own randomness (§4.1).

### 1.6 The lattice pad commitment (`jolt-zk::whiteout`)

`LatticePadCommitment` produces `ZkBackend::PadCommitment` (the raw image t_p)
and `ZkBackend::PadOpening` (pads, g and r_p). It is not a `VectorCommitment`.
Its committed vector is mixed-radix, and its opening is proved by the
Σ-protocol, never by a row opening.

**The column vector.** The commitment builds this vector itself. The same
vector is the `PadDigits`, `OperandDigits` and `PadRandomness` segments of s,
in this order, so the e layout of §3.2 follows it.

1. **Regular pads, then g.**
   - The pads go in pad-index order, skipping operand pads, which are
     committed only in group 2 (A5).
   - They are grouped into R-elements of 64 consecutive values, with the last
     element zero-padded. g takes its own element.
   - Each element is decomposed into 32 balanced radix-16 digit planes. Then
     `p_i = Σ_j 16^j · digit_{i,j}` is a field-linear functional of s, and
     `g = Σ_j 16^j g_j` is R-linear.
2. **Operand pads.** For each p_x in `operand_pads` order there are 16
   constant R-elements p_{x,j}, its balanced radix-256 digits, with
   `P_x = Σ_j 256^j p_{x,j}` (§3.2).
3. **r_p**, the box digits of §1.5.

The vector is packed into R_{d_p} elements by the stride map (§0). The
commitment is `t_p = apply_centered(view_p, vector)`.

**The row L.** The paper needs `[A_{p,r} ; L]` to be one uniform matrix
instance (§4.2; H4 hypothesis). The pad view therefore has `rank + 1` rows,
and setup sizing counts row `rank`. That row is not part of the commitment.
With `a_j` its entries on the randomness columns:

```
L · r_p := π_0( Σ_j a_j · r_{p,j} ) ∈ R,
```

where π_0 takes stride component 0 (§0). For uniform `a`, the map
`a ↦ π_0(a ·)` is an F-linear bijection onto R-linear maps on R^{d_p/64}, since
both spaces have dimension d_p. The stacked matrix is therefore uniform.

### 1.7 Module structure

`wo:lem:sigma-extract` cancels c, which needs **N** to be R-linear. Restriction
of scalars gives this only when every multiplication by a matrix entry acts on
stride-embedded inputs (§0). Akita's consecutive-coefficient split from d_A to
d_B (§1.4) only relabels coefficients; it is not a module map.

The segment table therefore rejects a residual or pad layout in either case:

- `d_A ≠ d_B`;
- a ring dimension that is not a multiple of 64.

`SigmaSetup::new` returns `WhiteoutError::InvalidRelation` (reaching
callers as `ZkError::Whiteout`) when a segment's native dimension differs
from the setup's `D`, and `SegmentTable` rejects a native dimension that is
not a power-of-two multiple of 64 with `WhiteoutError::Segment`. The headline
layouts use R_256 throughout and pass.

## 2. The Σ-protocol

### 2.1 Inputs and checks before anything is sent

`whiteout::prove` takes three groups of inputs:

- from M6b: the `ResidualRelation` and the `PadOpening`;
- from W3, through W6: the residual opening `(w_k, t̂_a, r_a)`, its image
  `u_a`, the evaluation points of its claims (A4), and the public segments
  of w_k;
- from the setup: the layouts and `SigmaParams` (§5), plus a
  `ResponseProver`.

It assembles **s** in segment order: w_k, t̂_a, r_a, regular digits and g
digits, operand digits, r_p.

Before anything is sent it runs the following **release checks**. Each
failure is fatal and maps to a typed `ZkError`; the order of precedence is
check 2, then 3, then 1.

1. **Segment alphabets.** Every coefficient lies in its segment's alphabet,
   and every `ConstantPoly` element has zero coefficients outside native
   positions 0..m.
   - This is the zero-knowledge precondition: `‖cs‖_σ ≤ 1/α` (§5) holds
     only for vectors with the declared shapes.
   - M5a's ∞-norm `ShiftOutOfRange` cannot detect a violation. For example,
     a dense operand segment passes the per-coordinate shift bound, but its
     ℓ₂ norm can exceed the constant-polynomial bound by 8×.
   - The check is constant time per coefficient, branching only on the
     aggregate.
2. **Every row of R_res, aggregated.** ρ is squeezed and the prover
   computes `h = Q_μ(1, s)`. `ct(h) = Σ_a μ_a·row_a(s)` is multilinear in ρ
   with one monomial per label (label 0 is `ct(g)`), covering each affine
   row, each product row, each scalarity row, and `ct(g) = 0`. So
   `ct(h) = 0` shows every row vanishes except with probability ℓ/|F|
   (Schwartz–Zippel). On failure the per-row scan (`first_violation`) runs
   only to name the row.
3. **N·s = t,** with t the raw images. They must equal the sent commitments:
   t_p as sent by `commit_pads`, and u_a as W3 sent it.

**Why check 2 runs before anything is sent.** h is a coordinate of V, V
lies in e, and the response proof is not zero knowledge. If ct(h) ≠ 0, then
ct(h) = Σ_i μ_i·violation_i would publish one field element, a ρ-weighted
combination of the violated rows. H6 needs `ct(Q′) = 0`. Squeezing ρ sends
nothing, and the prover aborts on `ct(h) ≠ 0` before σ, so the transcript
never carries such a value. This is a self-check of the honest prover, not a
soundness check: the verifier's guarantees do not depend on it. The
verifier checks the same ρ-aggregated identity, so a cancellation the
prover's check misses can pass verification too; that event is the ℓ/|F|
aggregation term already in the soundness bound (`wo:lem:sigma-aggregate`).

### 2.2 Lifting the rows

Each row becomes an R-linear form in `(u, s)` (`wo:lem:ct-lift`), built in
`whiteout/lift.rs`:

| Residual element | Lift |
|---|---|
| `ResidualVar::Pad(i)`, regular | `Σ_j 16^j ·` coordinate of digit `(i, j)` |
| `ResidualVar::Pad(x)`, operand | `ct(P_x) = Σ_j 256^j ·` coefficient 0 of `p_{x,j}` (§3.2: "every use of this pad uses ct(P_x)") |
| `ResidualVar::Witness(j)` | evaluation claim j of the residual commitment, `w̃_k(r_j)` (A4), kept as the symbolic term `eq(r_j, ·)` over w_k |
| constant `b` | `u · b` |

- **Affine rows.** Each `AffineRow` becomes a constant-term row `Λ_j`.
- **Scalarity rows.** Each operand pad adds 63 of them,
  `ct(X^{−r} P_x) = 0` for r = 1..63.
- **Product rows.** Each `ProductRow` becomes
  `X_i(u, s) = u·x̂_i − P_{x_i}(s)`, with `Y_i, Z_i` the lifts of y, z.

Every `ProductRow.x.pad` must be in `operand_pads`, checked in release (A5).
Extension-field products reach W4 already expanded into single-term rows with
auxiliary slots (A1).

### 2.3 Labels and aggregation (Step 1)

`RowLabels::assign` is one public function of the relation (§4.2, "Verifier
cost of the zero rows"):

- Operand o, in `operand_pads` order, owns labels `64·o + r` for
  r ∈ [1, 63]. Label `64·o` is unused, and label 0 is the constant monomial
  reserved for `ct(g)`.
- Product rows take consecutive labels from `max(64·n_ops, 1)` in relation
  order, then affine rows.
- `ℓ = ⌈log₂(max_label + 1)⌉`. At the headline the largest label is 9,625,
  so ℓ = 14.

The verifier squeezes ℓ **full** field challenges, `challenge::<F>()`
consecutively, with no prover message in between. The error term is ℓ/q,
which is why `challenge_small` is not used. Then
`μ_i = Π_{j : bit j of a_i} ρ_j`, and both sides form

```
Q_μ(u, s) = Σ_i μ_i (X_i Y_i − u Z_i) + u · Λ_μ(u, s) + u · g(s),
Λ_μ = Σ_j μ_{n×+j} Λ_j     (one LiftedFunctional)
```

### 2.4 Attempts (Steps 2–4)

The prover squeezes a 32-byte fork seed σ from the live transcript, and the
verifier squeezes it at the same position. σ binds the oracle input's τ. Then,
for i = 1..T_att:

1. **Mask.** Draw `y ← π_0` with M5a's `GaussianSampler::fill`, one sampler
   for the common width (§5), in parallel (`fill_parallel`, §2.4.1). Public
   segments get `y = 0`.
2. **First message.** Compute:
   - `f = N·y`: `apply_centered` on each part of N;
   - `E_0 = L·y + Q_μ(0, y)`;
   - `E_1 = L·s + Σ_i μ_i (X_i(1,s) Y_i(0,y) + X_i(0,y) Y_i(1,s) − Z_i(0,y)) + Λ_μ(0,y) + g(y)`.

   Every lift of y into F (in Q_μ(0,y), B_μ, L·y, g(y) and Λ_μ(0,y)) uses the
   branch-free centered lift of §1.3.
3. **Commit to V.** Set `V = (f, E_0, E_1, h)` and compute
   `t_D = commit(A_D, G⁻¹(V))`: radix-16 digits, 32 per coefficient, no
   randomness. A_D is 1 × R_64, so t_D has 64 coefficients.
4. **Nonce and challenge.**
   - `n_i` is 32 uniform bytes from the prover's RNG, never a counter.
   - `c_i = SampleChallenge(AttemptFork::new(σ, n_i).absorb(t_D))`, from the
     operator-norm-filtered d = 64 family.
   - The prover uses the constant-time sampler of §2.8. Its output equals
     `sample_indexed_challenges_from_seed(seed, 64, 1, cfg, Some(D64_SELECTIVE_L2))`,
     which the verifier uses.
5. **Response.** `v = c_i·s` is the exact centred-integer negacyclic product,
   never reduced mod q. Set `z = y + v`, then call
   `JointRejection::accept(z_G, v_G, rng)` on the Gaussian coordinates,
   block-major.
6. **Outcome.** On acceptance, send t_D and n_i, and leave the loop. On
   rejection, discard the attempt.

`AttemptFork::new` initializes a fresh sponge with the protocol domain tag
`jolt-zk/whiteout/attempt-fork/v1`, then absorbs σ and the 32-byte nonce.
For an accepted attempt, the verifier rebuilds the fork from σ and the sent
nonce n, absorbs the sent t_D payload, and derives c. The tag separates attempt
queries from the grinding forks, which are keyed by a `u32` counter, giving
attempt queries their own oracle domain (H2: "a separate domain of the
oracle"). A counter would not do: H2 needs a nonce that is uniform and secret
until it is published.

A fork lives outside the transcript. It records no `TranscriptEvent`, and a
rejected attempt leaves nothing in the replay log.

#### 2.4.1 Parallel mask sampling

`GaussianSampler::fill_parallel` fills each Gaussian block. For a block of
N coordinates it:

1. draws `⌈N/2^14⌉` seeds of 32 bytes from the prover's RNG, in chunk
   order, before sampling anything;
2. fills chunk i (coordinates `2^14·i ..`) with M5a's `fill`, driven by
   ChaCha20 keyed by seed i, as one rayon job.

**Distribution.** Each chunk runs the unchanged sampler on its own coin
stream, so each coordinate is a draw from `D_σ`. The streams are keyed by
independent uniform 256-bit seeds (the RNG is a `CryptoRng`). If the
ChaCha20 streams were uniform coins, the coordinates would be exactly
independent, as with `fill` on true randomness. They are not uniform, so
the hybrid that replaces each stream by uniform coins costs the PRG
advantage once per stream. Over at most T_att attempts with
`Σ_b ⌈N_b/2^14⌉` chunks each (405 at the headline, so `n_ch ≤ 4,050`), the
zero-knowledge error gains

  `n_ch · Adv^PRG_ChaCha20(t)`,

where t is the distinguisher's running time. This term is computational:
Thm `wo:thm:zk` stays statistical in all its other terms. Under the usual
conjecture that ChaCha20 with a 256-bit key has no attack better than key
search, `Adv ≤ t/2^256`, so the term is about `t·2^−244`, at most 2^−144
for `t ≤ 2^100`. It does not affect completeness or soundness.

The serial `fill` used the caller's RNG directly, so its coins were as good
as that RNG; a caller passing a seeded `ChaCha20Rng` already had this
assumption. `fill_parallel` makes it unconditional.

**Determinism.** The seeds are drawn sequentially and the chunking depends
only on N, so the output depends only on the RNG's stream. A test fills
`5·2^14 + 123` coordinates on one worker and on eight and compares the
bytes.

**Constant time.** See the `fill` row of §2.8.

### 2.5 What enters the transcript

| Order | Event | Prover | Verifier |
|---|---|---|---|
| (earlier) | pad commitment t_p, 768 coefficients raw | `commit_pads` | `receive_pad_commitment` |
| (earlier) | residual commitment u_a | Akita fold k (W3) | W3; passed to `verify` as an argument |
| 1 | ρ_1..ρ_ℓ | `challenge::<F>()` × ℓ | same |
| 2 | σ | `challenge_bytes::<32>()` | same |
| 3 | t_D, 64 coefficients raw | accepted attempt only | `receive_n` |
| 4 | n, 32 bytes | accepted attempt only | `receive_bytes(32)` |
| — | c | derived from the fork | rebuilt from (σ, n, t_D) |
| 5… | response proof | `ResponseProver::prove_response` | `verify_response` |

Sending t_p and t_D raw costs (768 + 64) × 16 B ≈ 13 KB more than the paper's
two compressed 128-byte forms, until W5 adds compression (A12).

### 2.6 Retry versus abort

| Event | Handling | Why |
|---|---|---|
| accept bit 0 | Internal retry with fresh y and n_i. | Step 4: Fiat–Shamir with aborts. |
| all T_att attempts rejected | `WhiteoutError::AttemptsExhausted`. The caller may rerun the whole proof with fresh randomness. | Probability ≤ 2^−128. H3 makes the event independent of the witness, and the simulator outputs ⊥ at the same rate (A14). |
| `SampleError::ShiftOutOfRange` | `WhiteoutError::InvariantViolation`, fatal. Never retried, never recorded. | Unreachable after check 1 of §2.1. |
| other `SampleError`, layout errors | `WhiteoutError::Sample`, `Crypto` or `InvalidRelation`, fatal. | Public parameters. |
| operator-norm budget exhausted | Fatal on both sides. | A public function of the oracle input, with probability about 2^−1580 at 4,096 tries. Conditioned on the budget not running out, c is uniform on the filtered family. Retrying on exhaustion would change nothing measurable, but it would add an attempt-dependent path, so it is fatal instead. |
| `ZkError::FoldOverflow` (W2) | Whole-proof retry. | Outside the Σ-protocol. |

### 2.7 Public coordinates

The public segments are declared by W3 as unions of whole native elements. At
k = 2 they are the 6,291,456 setup-group cells of w_2 and the 589,824 inner
digits of fully public blocks. On them:

- y = 0 (§4.2), and they are excluded from π_0, from `‖·‖_σ`, and from every
  rejection block;
- `z = c·s_pub` gets D_pub = 3 radix-16 digits, since
  `κ₁ · 8 = 424 ≤ 7·(16³−1)/15`.

The response proof checks the same equations on public coordinates, with no
separate `z = c·s_pub` rows. Every branch on "public" reads only the segment
table.

### 2.8 Constant-time requirements

The ZK theorem's view is the proof. Per-attempt timing is a side channel that
model excludes, and W4 closes it. A rejected attempt's y_i, c_i, z_i, V_i and
t_D are correlated with the witness through the rejection event. H2 insists
that the oracle's answers on them are never observed, and timing must not
reveal them either.

> **Rule.** Every step computed before an attempt's accept bit is known runs in
> time, and with memory addresses, independent of the attempt's fresh secrets
> (y, n, c, and everything derived from them). No secret value is used as a
> branch condition, loop bound, table index, or divisor. Steps on s alone, and
> steps after acceptance, inherit the prover's ordinary posture: Jolt's prover
> is not constant time in its witness, and those steps are identical across
> attempts.

| Step | Secret inputs | Requirement |
|---|---|---|
| Gaussian `fill` | y | M5a sampler (isochronous, reviewed), run as `fill_parallel` (§2.4.1): 2^14-output chunks whose count and boundaries come from the public block length; all chunk seeds drawn from the prover's RNG before any chunk runs; every chunk runs the isochronous loop to completion, with no early exit and no shared state. A chunk's duration depends on its trial accept bits, which are independent of the values it outputs (the HPRR20 argument, per chunk), so rayon's scheduling sees only those. The chunks run on the DIT pool of A11, as rayon jobs of the attempt inside `with_dit`. |
| `N·y`, `A_D·G⁻¹(V)` | y, V | `apply_centered` of §1.3: branch-free Montgomery lift, NTT with public twiddles, constant-time Garner, no LUT. |
| `Q_μ(0,y)`, `B_μ`, `L·y`, `g(y)`, `Λ_μ(0,y)` | y | Branch-free centered lift into F. Ring products with public or s-only operands. |
| `G⁻¹(V)` | V | The bias path of `BalancedDecomposePow2Params::biased`, whose sign select is a 129-bit signed subtraction. A comparison there compiled to branches on x86 (W4 found and fixed this; `cargo asm` on `x86_64-apple-darwin` shows the scalar kernel and the AVX2/AVX-512 remainder loops branch-free). |
| `Fork::attempt`, Keccak | n, t_D | Keccak-f is constant time. |
| challenge sampling | c_i | A prover-side constant-time sampler, byte-identical to the verifier's. It replaces Fisher–Yates swaps at secret indices with full-array masked swaps and drops the sparse hash-table path (d = 64 always takes the dense path). It always runs the operator-norm decision's exact certified evaluation, and never `decide_transposed_fused`'s spectrum-dependent fast path, its `Indeterminate` fallback, or its `freq_row` indexing by secret positions. Discarded candidates go through the same constant-time body. Their number is independent of the accepted value, by rejection-sampling isochrony. |
| sparse-to-dense c | c_i | A constant-time scatter: each of the 64 output coefficients is a masked sum over all support entries. |
| `v = c·s` | c | A dense 64×64 negacyclic convolution, with c as 64 `i8` coefficients. |
| `accept` | z, v | M5a's constant-time rule. Only the `Choice` is branched on. |

**Audit list.** Each item is checked in the assembly record (§7, item 4):

- `Fp128` add/sub `if overflow | carry3` (`solinas/fp128.rs:155-171`) must
  compile to a select;
- `GarnerData::centered_mixed_radix`'s bool-to-int products, which W4
  replaces with masks in its constant-time Garner;
- the value branches at `ntt/ifma52.rs:59,68,82` and
  `crt_ntt_repr/lut.rs:105`, which W4 code must not reach;
- the NEON and AVX butterflies and pointwise kernels;
- the decomposition SIMD paths.

Per-attempt buffers have public sizes. No tracing span carries an attempt
value. The attempt body is `#[inline(never)]`, so the review covers the
production monomorphization.

**Record (W4, 2026-10-20, Apple M4 Max).**

- **dudect.** `crates/jolt-zk/examples/sigma_dudect.rs` (feature
  `whiteout-diagnostics`) times each step against extreme fixed classes
  (masks all 0, all +2^20, all −2^20; V all 0 or −1; c all 0, +2 or −2)
  with dudect's ten percentile crops, at D = 64 and the stride case
  D = 256. Both classes are written into slots allocated once, by one path
  that draws the same randomness for either class. A cloned fixed input,
  one left on the zero page, or inputs allocated afresh per measurement
  reach memory differently and gave false positives (the last one under
  DIT, see below). An A/A run (both classes random) passed everywhere on
  the old harness, and again on the slot harness under DIT (3 · 10⁴
  measurements, seed 0, all 30 slot tests, max |t| = 3.8).

  First record, without DIT and with per-measurement allocation: 10⁵
  measurements per test at D = 64 (2.5 · 10⁴ at D = 256), t_D at 10⁷
  (2.5 · 10⁶), seed 0:

  | Step | D = 64 max \|t\| | D = 256 max \|t\| |
  |---|---|---|
  | linear parts of y (lift, claim sums, dense forms) | ≤ 1.1 | ≤ 1.2 |
  | `Q_μ(0,y)` and `B_μ` | 3.3 | 1.9 |
  | `N·y` with gadget, fixed ±B | ≤ 1.9 | ≤ 5.9 (crop) |
  | `N·y` with gadget, fixed 0 | **17.1** (raw 14.6) | **190** (raw 29) |
  | `L·y` | **10.6–14.4** (raw ≤ 5.3) | **5.9–47.8** (raw ≤ 18.3) |
  | `t_D = A_D G⁻¹(V)` at 10⁷ / 2.5 · 10⁶ | 1.5 | 2.8 |
  | attempt fork seed | 1.4 | 1.0 |
  | `z`/`v` assembly | 1.5 | 2.1 |
  | challenge sampler, per candidate count 1–4 | — | ≤ 3.3 |

  Seed 1 reproduced the D = 64 rows before the run was stopped.

  **Finding: the Ajtai map is not constant time on M4 without DIT.**
  Micro-tests localize it to `CyclotomicCrtNtt::to_ring_ct`: a zero output
  is 0.5% faster at D = 256 (|t| = 13, 2 · 10⁵ samples), while
  `assign_centered_i32_ct`, `add_assign_pointwise_mul`, and `Fp128` mul/add
  pass alone. Every one of those operations is branch-free in the source and
  in the asm audited below, so this is operand-dependent timing in the core.
  With the DIT bit set on every thread, `to_ring_ct` passes (|t| ≤ 1.7).
  The prover now sets DIT for the attempts (A11).

  **Record under DIT (A11, 2026-10-20, Apple M4 Max).** The harness runs
  every step and the sampler on the prover's own DIT pool
  (`diagnostics::with_dit`); 10⁵ measurements, t_D at 10⁵ (2.5 · 10⁴ at
  D = 256), seed 0. The machine was shared and loaded (load average 7–57),
  which widens the noise; the positive controls below ran under the same
  load.

  | Step | D = 64 max \|t\| | D = 256 max \|t\| |
  |---|---|---|
  | linear parts of y | ≤ 1.8 | ≤ 2.1 |
  | `Q_μ(0,y)` and `B_μ` | 2.0 | 2.4 |
  | `N·y` with gadget, fixed 0 / ±B | ≤ 1.5 | ≤ 2.3 |
  | `L·y`, fixed 0 / ±B | ≤ 1.8 | ≤ 3.1 |
  | `t_D = A_D G⁻¹(V)` | 1.9 | 2.3 |
  | attempt fork seed | 0.9 | 1.0 |
  | `z`/`v` assembly | 1.5 | 1.3 |
  | challenge sampler, per candidate count 1–4 | — | ≤ 2.0 |

  **The DIT residual was the harness.** With per-measurement allocation,
  DIT left `L·y` at |t| ≤ 11.2 in crops (raw ≤ 3.2) and `N·y` fixed 0 at
  D = 256 at |t| = 10.7 in crop 1, reproducing the earlier residual. Micro
  tests on `L·y` at D = 64 separate the input's value from its memory:
  - a fixed random mask, built by the random class's own path, passes
    (|t| = 1.4); a copy of a random mask, built by a different allocation
    path, fails at |t| = 16 in crop 7. Both have the same value
    distribution, so the signal is placement, not data;
  - with fixed slots and one write path, `L·y` passes for fixed 0, +B and
    a fixed random mask at 10⁵ on seeds 0 and 1 and at 4 · 10⁵ on seed 2
    (|t| ≤ 2.8), and `N·y` at D = 256 passes for the same classes at
    2.5 · 10⁴ and 10⁵ (|t| ≤ 2.4);
  - positive controls on the slot harness: a delay injected into the fixed
    class is found at 17 ns on `L·y` at 4 · 10⁵ (|t| = 12.9) and at 38 ns
    at 10⁵ (|t| = 27); with DIT cleared on the pool, `N·y` fixed 0 at
    D = 256 fails at 2.5 · 10⁴ (|t| = 9.8) while the same run under DIT
    passes (|t| = 2.0).

  So under DIT no step showed a leak at or above about 17 ns on `L·y` (of
  12 µs) in these runs, and the harness detects the M4 leak DIT removes.
  **Corrected by P7 below: DIT does not remove the zero-output leak.**

  **Record P7 (2026-11-05, Apple M4 Max, under DIT).** A review at load
  4–20 found `N·y` and `L·y` failing for constant non-negative masks at
  D = 256. Reproduced on whiteout/dev b41acbb7b with the slot harness on
  the DIT pool, 2.5 · 10⁴ `N·y` / 10⁵ `L·y` measurements, seeds 0–2, load
  7–130 (max |t| per seed):

  | Step, fixed class | before | after the blind |
  |---|---|---|
  | `L·y`, 0 | 12.0 / 5.2 / 10.3 | 1.2 / 1.8 / 1.1 |
  | `L·y`, +1 | 4.8 / 2.6 / 5.8 | 1.9 / 1.3 / 0.9 |
  | `L·y`, ±B, A/A | ≤ 2.5 | ≤ 2.7 |
  | `N·y`, 0 | 5.9 / 8.5 / 7.6 | 7.4 / 3.1 / 6.7 |
  | `N·y`, +1 | 4.4 / 3.9 / 2.7 | 11.2 / 6.7 / 2.6 |
  | `N·y`, +B | 2.6 / 1.6 / 3.2 | 5.1 / 4.2 / 6.8 |
  | `N·y`, −B, A/A | ≤ 1.9 | ≤ 2.9 |

  - **Root cause (L·y, and the Ajtai map's share of N·y).** The CRT
    reconstruction of an all-zero accumulator. A constant mask in
    `[0, 2^29)` has an all-zero top limb (§1.3, Limbs), and mask 0 has two,
    so a whole limb pass multiplies the matrix by zero and `to_ring_ct`
    reconstructs zero. A probe of `apply_columns` alone (16 columns,
    D = 256, 10⁵ measurements, one row) gives |t| = 36, 8.6 and 17 for
    fixed 0 across three runs. Its two halves pass when timed apart: the
    transforms and pointwise products give ≤ 2.3, and `to_ring_ct` on the
    same accumulators read back from memory gives ≤ 2.4. `to_ring_ct`'s
    stages also pass apart: the inverse NTT, centering, Garner and the
    lift each give ≤ 4.2 at 4 · 10⁵. So the time difference needs a zero
    accumulator that was just computed and is consumed by the fused
    reconstruction. No single instruction was isolated. The source is
    branch-free and DIT is set, so this is a value-dependent effect in the
    core that DIT does not cover. Candidate mechanisms, none verified, are
    the loaded-zero multiply-add effect of the optimization campaign (U1b)
    and load value prediction.
  - **Fix: a public blind.** Every accumulator of `apply_columns` starts
    from the CRT+NTT image of a fixed pseudorandom ring element `K`
    (`PreparedMatrix::blind`, a one-ring public matrix stream), and `K` is
    subtracted in F (§1.3). The reconstruction then sees `M·x + K`, which is
    never the degenerate zero: an input with `M·x = −K` solves an
    inhomogeneous SIS instance. The chunk width keeps one column of CRT
    capacity for `K`. Outputs are unchanged, and the frozen `apply_centered`
    digests still pass.
    An honest mask is unlikely to produce an all-zero limb plane: with
    σ ≈ 2^35.7 that needs about 4096 or more coefficients all below 2^29
    (each about 2^−7). The blind therefore defends against the dudect
    classes, not against an observed leak on honest Σ runs.
  - **Sign-selected multiplies removed (hardening, no measured effect).**
    `lift_ct`, `digits_to_field_ct`, the gadget term of `N·y` and the claim
    sums multiplied by `F(sign)`, which is 0 for every non-negative input.
    They now multiply by `lift_biased(x) = x + 2^63`, which is never zero
    for `x > i64::MIN` while `p > 2^64` (its top bit is still the inverted
    sign, but no operand is zero or a 0/1 selector), and subtract the
    constant bias times the factor.
    Alone, this left `L·y` fixed 0 at |t| 5.7–13.7.
  - **Open: N·y residual.** After the blind, `N·y` still fails for
    constant non-negative masks (0, +1, +B) at |t| 4.5–11 in crops, raw
    ≤ 3.2. The fixed class is about 0.05–0.08% slower (150–350 ns of
    430 µs). Constant −B, a repeated bell-shaped mask, fresh non-negative
    magnitudes, and A/A pass. `L·y`, which is the same Ajtai map on one row
    with no gadget, passes. The cause is not localized. A micro-probe of
    the gadget loop alone passes at 2 · 10⁴. A run with the gadget skipped
    (diagnostic only) still failed, so the residual is not the gadget
    alone.

  The full harness after the fix, with the mask classes 0, +1, ±B, a
  repeated bell-shaped mask and fresh non-negative magnitudes, ran at
  10⁵ / 2.5 · 10⁴ measurements on seeds 0 and 1, load 7–40. Seed 2 was
  stopped when the load reached 260. 101 of 108 tests pass: linear parts
  ≤ 2.4, `Q_μ(0,y)` and `B_μ` ≤ 2.7, `L·y` ≤ 2.7, `t_D` ≤ 3.0, fork seed
  ≤ 2.6, `z`/`v` ≤ 2.8, and the challenge sampler ≤ 2.4. The 7 failures
  are the open `N·y` residual: fixed 0, +1 or +B at |t| 4.5–7.9, at D = 64
  and at D = 256.

  **Cost.** `sigma_timing` at scale 2048 with jolt-algebra `asm`, six
  interleaved pairs of 20 proofs under the perf lock, load 28–90: median
  0.46 s per proof before and 0.43 s after. The difference is within the
  noise, so no slowdown was detected. The blind adds one ring subtraction
  per chunk and row, and the biased lifts each drop one multiplication.

  **Not verified:** x86_64; the
  `asm` build of `Fp128` (the prover's graph enables jolt-algebra `asm`,
  the harness's does not); M1–M3.

- **asm, branch-free.** `Fp128` add, sub and mul have no conditional branch on
  aarch64 or x86_64; add/sub compile to `csel` and `cmov`.
- **asm, public branches only.** These branch only on loop counters and
  public indices:
  - `challenge_times_i64`;
  - the challenge view;
  - the constant-time sampler, apart from two kinds of allowed branch: the
    `next_usize_mod` rejection on discarded bytes, and the per-candidate
    accept bit.
- **asm, decomposition.** The scalar kernel and the AVX2/AVX-512 remainder
  loops of `BalancedDecomposePow2Params::biased` are branch-free on x86_64
  after the W4 fix.
- **Not audited line by line:**
  - the NEON/AVX NTT butterflies and pointwise kernels, which the dudect
    Ajtai test covers only empirically;
  - `Fp128`'s canonical encoding in the attempt fork's absorb of t_D.

  Both are open audit items for the W5 last-PR gate.

**Platform.** The argument assumed that integer multiply, add and compare take
data-independent time without DIT/DOITM. The dudect record above refutes this
on Apple M4 for the CRT reconstruction, so the prover sets DIT for the
attempts (A11). DIT is not sufficient: under DIT the M4 still times the
reconstruction of a just-computed zero accumulator differently (P7), which
the blind of §1.3 avoids, and an `N·y` residual for constant non-negative
masks remains open (P7).

**Memory hygiene.** The prover overwrites its heap secrets with zeros, by
volatile writes through the `zeroize` crate, before freeing them. jolt-zk's
`SecretVec` zeroes a `Vec`'s whole capacity on drop, in parallel chunks,
without needing `Zeroize` on the field type; it holds the large buffers and
every field buffer, and `ResponseWitness` uses the same wipe. jolt-crypto's
buffers use `Zeroizing`. The writes are unconditional and run over public
lengths, so they add no secret-dependent branch. Wiped:

- `s` (`private_vector`), the pad opening's digits (`LatticePadOpening`), and
  the pad commitment's working copies of the pads, g, and their digit planes;
- per attempt: y, z and c·s on the Gaussian coordinates, `N·y`, `L·y`, V,
  the linear parts of s, y and z with their claim terms, and t_D. After a
  rejection the prover zeroes y, z and c·s, the nonce n_i, and the dense
  c_i before the next attempt runs;
- the accepted response's z, c·s per segment, the product polynomial, and
  the response digits e (`ResponseWitness`);
- `fill_parallel`'s chunk seeds and each chunk's ChaCha20 state;
  `sample_box_digits`' bytes and digits; `G⁻¹(V)` in `commit`; and the limb
  scratch of `apply_centered`, which holds the input's low limbs;
- `CommittedPads`' pads and BlindFold's pad-row blinds (M6b).

Not wiped:

- copies the compiler or allocator makes: register contents, stack spills
  and stack locals (the Gaussian trial's values, `Fix192` draws, the sparse
  c_i, the attempt fork's Keccak state, ring values such as E_0, E_1, h and
  `L·s` held by value), and pages the OS swaps out or dumps;
- the NTT-domain accumulators and partial outputs inside the Ajtai kernel
  (`apply_columns`), the parallel partial sums inside the lift's claim
  terms, and `attempt_seed`'s encoding scratch, which is shared with the
  verifier and holds one coefficient of t_D;
- buffers abandoned on a public-shape error before they are wrapped;
- the caller's RNG, the residual opening the caller passes in, and the
  message values `CommittedPads::decode` returns;
- BlindFold's tail, which copies the pads and blinds into its witness rows
  and draws its own random instance;
- `check_witness`, the response statement's definition, which rebuilds z
  and V from e.

Public data and verifier state are not wiped. The one exception is the limb
scratch of `apply_centered`, which the verifier also runs. The test
`tests/whiteout_zeroize.rs` (feature `whiteout-prover`) uses a scanning
global allocator to check that the chunk seeds, the box digits and the pad
copies are freed zeroed. Each of the three fails on the code before this
change, and a positive control in each test shows that the scan finds an
unwiped buffer. No test covers stack values, and none covers the Σ
attempt's buffers, since their contents cannot be planted from outside.

Cost: `sigma_timing` at scale 2,960 (3.0M Gaussian coefficients, 16
threads), before and after binaries interleaved four times at 10 proofs
each, gives 0.261 and 0.266 s per proof (+1.6%). At scale 5,920, two
interleaved pairs on a lightly loaded machine give 0.509/0.513 and
0.516/0.516 s (+1%). The first version wiped through `Zeroizing<Vec<_>>`
on one thread, in two passes with the second byte by byte, and cost about
+18% at scale 2,960 under heavy load. A single sequential pass still cost
+6%. `SecretVec` therefore wipes once, one write per element, in parallel.

## 3. Response statement and `ResponseProver`

### 3.1 The boundary

**`jolt-zk` owns what defines the Σ-protocol's relation:**

- s, N, Q_μ and L;
- the box and the certified alphabets;
- the logical layout and contents of e, as typed segments.

**Akita (W5) owns how that relation is proved:**

- committing to e, the product sumcheck, and the row batch at η;
- the fused range-and-relation sumcheck;
- the structured weights A(**r**) and B(**r**), and setup offloading;
- compression and the ξ digits;
- the final opening.

That split follows §4.3, "Evaluating the public weights": only Akita's
structured-weight machinery evaluates those weights succinctly. So the
statement carries typed, structured descriptions, never dense matrices. W5
states its alignment needs for e's physical packing before slice 8 (A15).

### 3.2 `ResponseStatement<F>`

The statement follows the items of §4.3. The seed and the layouts come from
the setup, and the statement does not duplicate them.

```rust
pub struct ResponseStatement<'a, F: JoltField, const D: usize> {
    setup: &'a SigmaSetup<F, D>,        // N, its targets (u_a, t_p), L, A_D, segment table, params
    relation: &'a LiftedRelation<F>,    // the lifted rows, labels and symbolic claims
    challenge: SparseChallenge,         // c
    rho: Vec<F>,                        // ρ; μ is derived by the relation's labels
    v_image: Vec<ChallengeRing<F>>,     // t_D (raw)
}
```

The fields are crate-private; W5 reads them through `challenge()`, `rho()`,
`v_image()` and `layout()`, and the setup and relation through their own
accessors. The layout, `N`, `L` and the quadratic terms are derived from the
setup and relation, never restated.

- **`ResponseLayout`** lists e = (G_z⁻¹(z), G⁻¹(V), G⁻¹(θ)) as typed
  segments. Each segment carries a length, a radix, a digit count and an
  alphabet (Item 1). The ξ segments are added in W5.
  - **z.** One segment per s-segment, in s order, which is also the pad
    commitment's column order (§1.6). Within a segment, digits are
    plane-major. The radix is 16, with D_b digits for Gaussian segments and
    D_pub for public ones. `𝓑_cert` is the balanced D-digit range.
  - **V.** 32 radix-16 digits per coefficient, the same digits that t_D
    commits.
  - **θ.** 63 coefficients × 32 digits = 2,016 cells.
- **`QuadraticStatement`** lists every term of Item 4:
  - Λ_μ's sparse pad and auxiliary terms, plus its evaluation-claim terms
    `Witness(j) → eq(r_j, ·)`;
  - the scalarity rows, implicitly, as the operand list and the label base;
  - the products as `(x̂, operand, Y, Z)`;
  - the `u·g(s)` term, which uses the g-element of `PadDigits`;
  - the row index of L in the pad view.

  With c and ρ fixed, H_e is affine in e (§4.3).
- **Item 5**, `ct(h) = 0`, is one coefficient row on the V segment.

### 3.3 `ResponseWitness<F>`

The witness is e in digit form only, `Vec<i8>` in `ResponseLayout` order.
jolt-zk computes all of it:

- `G_z⁻¹(z)` and `G⁻¹(V)`, the latter already formed for t_D;
- θ from (4), as the quotient of `Σ μ_i [X_i(c,z)][Y_i(c,z)] − [H_e]` by
  `X^64 + 1`. Its degree is below 63, and a nonzero remainder returns
  `WhiteoutError::InvariantViolation`.

At the headline shape, e has about 65.5M cells (§5), which is 65 MB as `i8`.

### 3.4 Trait

The trait is that of interfaces §5:

```rust
pub trait ResponseProver<F: JoltField, const D: usize> {
    fn prove_response<H: Sponge>(&self, statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness, transcript: &mut ProverTranscript<H>) -> Result<(), ZkError>;
    fn verify_response<H: Sponge>(&self, statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>) -> Result<(), ZkError>;
}
```

(`D` is the setup's native dimension, A17.)

Its contract:

- The commitment to e is the first transcript event after n.
- `verify_response` checks Items 1–5 over all of e, public z segments
  included.
- Errors are `ZkError`. A statement that does not hold is
  `WhiteoutError::ResponseRejected`, carried as `ZkError::Whiteout`.

W4's tests use `CheckingResponseProver`, test support that evaluates Items 1–5
directly on the witness. It is the statement's own semantics, not a second
implementation of an Akita rule.

## 4. Whiteout's `ZkBackend` (W6)

W6 implements the revised trait of interfaces §5 once W3's opening type exists:

| Associated type | Whiteout's choice |
|---|---|
| `PadCommitment` | raw t_p, from `LatticePadCommitment` |
| `PadOpening` | pads, g and r_p |
| `ResidualCommitment` | `cm(u_a)`, the masked prefix's compressed u_a (Akita A47), held as a compressed target of `N` |
| `Witness` | W3's residual opening: w_k, t̂_a, r_a, and the evaluation points |

W4 provides the parts:

- `commit_pads` = sample pads + `sample_zero_constant_term` (M6d) + box r_p +
  `LatticePadCommitment`, then send t_p;
- `prove` = §2.1–§2.4, then §3 and `prove_response`;
- `verify` = lift and label, squeeze ρ and σ, receive t_D and n, derive c,
  then `verify_response`. The verifier never applies N or Q_μ.

**How the backend reads `ResidualRelation`:**

| Field | Consumed as |
|---|---|
| `affine` | constant-term rows (§2.2) |
| `products` | product rows |
| `layout.operand_pads` | the operand group plus 63 scalarity rows each |
| `layout.num_pads` | must equal P |
| `num_witness` | the number of evaluation claims of the residual commitment (A4) |
| `layout.auxiliary_slots` | ordinary pads, with defining rows |

N·s = t is the backend's own. It covers three parts:

- the inner rows `A_a w_{k,i} − G t̂_{a,i} = 0`;
- the outer rows `[B_a|B′_a](t̂_a, r_a) = u_a`;
- `A_p(…) = t_p`.

g enters only through Q_μ's `u·g(s)` term.

## 5. Parameters

| Parameter | Headline value | Lives in | Validated by |
|---|---|---|---|
| challenge family | d = 64, 31 × ±1, 11 × ±2, ‖c‖_op ≤ Γ = 18, 2^128.06 elements, κ₁ = 53 | `jolt-crypto::challenge` (M5b) | `SigmaParams::new`: `OperatorNormRejection::D64_SELECTIVE_L2.validate(64, cfg)` and `validate_min_entropy_for_ring_dim(64, 128)`. The latter bounds the unfiltered family; the filtered family's 2^128.06 is the certificate checked when the table is built (M5b) |
| α, ε | 2^9, 2^−137 (M − 1 ≈ 2.7·10^−2; §5.2) | `SigmaParams` (`ALPHA`, `EPSILON_BITS`) | `JointRejection::new` |
| width k | one for every Gaussian coordinate, derived (below): 229,072,314 ≈ 2^27.8 (headline table at rank 5, ΣN = 6,597,888) | `SigmaParams::derive(segments)` | exact check (below) |
| digits D_b, D_pub | 8 for every Gaussian segment; 3 | derived | `⌊L_b²/k²⌋ ≥ 1 + ⌈log₂(2^21 N)⌉` (below), here 76 ≥ 45; `D_pub` covers `κ₁·B` |
| box L_b | `7(16^{D_b}−1)/15` = 2,004,318,071 ≈ 2^30.9, the positive end of the certified range (A9) | derived | implied by the digit rule: `p_𝓑 ≥ 1 − 2^−21` |
| shift bound | `κ₁ · B_b` | derived | `JointRejection::accept` |
| T_att | 25 | prover-only: `JointRejection::max_attempts`, from the sampler's realized M and `p_𝓑 ≥ 1 − 2^−20` | smallest T with `(1 − (p_𝓑 − ε)/M′)^T ≤ 2^−128`, in 128-bit fixed point with every rounding against acceptance |
| rejection arithmetic | — | derived | `SigmaParams::new` mirrors `JointRejection::new`'s bounds: boxes and shifts ≤ 2^62, the denominator k² < 2^126 (always, as k ≤ 2^58), worst-case exponent numerator < 2^126. A table the prover could not run is rejected by both roles |
| ν | 256 | const | — |
| ℓ | 14 | `RowLabels` | — |
| A_a, [B_a\|B′_a], A_p ranks | 5 × R_256 each (§5.1; 3 before) | Akita planner and SIS tables (rule 3), in the setup | Akita (SIS); W4 checks module structure and view lengths |
| A_D | 1 × R_64, binding on radix-16 digits | Akita planner | Akita; view length |
| randomness | r_a 161 and r_p 169 R_256 columns, radix 16 (§1.5) | layouts | `required_randomness_columns` |
| pad radices | 16 (32 digits), operands 256 (16 digits) | `LatticePadCommitment` consts | `GadgetDigits` checks |

**Width derivation** (A10). The width is derived in jolt-zk with integer
arithmetic, not tabulated. Prover and verifier must agree bit for bit, because
D_b fixes the e layout. The numbers above come from the code's headline
table (`tests/whiteout_params.rs`), in which every segment holds whole R_256
elements, so ΣN = 6,597,888 at rank 5. The test takes the rank from one
constant and the randomness lengths from `required_randomness_columns`.

1. **Closed form.** One width for every Gaussian coordinate (§5.1, "One
   width"): `σ = αΓ·√(Σ_b γ_b²)`, and so
   `k = ⌈√(2 ln 2 · Γ²α² · Σ_b γ_b²)⌉ + 1`.
   - ln 2 is a fixed-point upper bound with 2^−64 error, and the constant
     `2 ln 2·Γ²α²` is rounded up at 2^−32.
   - The final +1 leaves relative slack far above the ceiling errors.
2. **Exact check.** `⌈2^32·2 ln 2·Γ²α²⌉ · Σ_b γ_b² ≤ 2^32 k²`, in integers.
   This is `‖cs‖_σ ≤ 1/α` for every c and every s that passes check 1 of
   §2.1. `SigmaParams::new` runs it on whatever width it is given, so the
   derivation is checked too.
3. **Digits.** `D_b` is the smallest D with
   `⌊L(D)²/k²⌋ ≥ 1 + ⌈log₂(2·2^20·N_tot)⌉`. Then each coefficient leaves the
   box with probability at most `2^{1−L²/k²} ≤ 2^−21/N`, so p_𝓑 ≥ 1 − 2^−21
   needs no separate check. A width that no digit count up to 15 covers is
   rejected (`WhiteoutError::Overflow`).

On the rank-3 table at α = 16, the per-block closed form of the earlier
revision gave widths 6,766,161 and 13,532,321 and seven digits everywhere;
one width gives 6,962,124 and the same digits. At α = 2^17 and rank 5 the
response part of e grows from 64,191,232 (rank 3, α = 16) to 87,802,368
cells, public coordinates included.

### 5.1 Almost-never rejection (review decision, 2026-10-21; superseded by §5.2)

Review (Q. Dao) asked for a rejection probability that almost never
rejects, accepting larger parameters for a predictable prover. This
replaces the planned parallel Gaussian sampler as the answer to the
prover's time budget (A16).

**What sets the rejection probability.** One attempt is rejected with
probability `1 − (p_𝓑 − ε')/M ≈ ln M + (1 − p_𝓑)`, where `ln M = t/α +
1/(2α²)` and `t = √(2 ε_bits ln 2) = 13.78` at `ε = 2^{−137}`. The
probability does not depend on the witness or the shift. Driving it to r
needs `α ≈ t/r` and `1 − p_𝓑 ≪ r`. Every width `σ_b` scales with α, so each
factor of 16 in α costs one radix-16 response digit.

The cost that matters is the digit count D, not α:

- the certified box 𝓑_cert is the D-digit envelope, so the collision radius
  `4Γ_Δ B_M`, and with it the SIS ranks of A, [B|B′] and A_p, depend only
  on D;
- for a fixed D, the largest α whose box still fits the digit rule of §5 is
  free.

So each regime below is D digits with α near the largest that D allows.

**One width.** With the two distinct widths of the closed form, the
rejection rule's worst-case exponent numerator `Σ_b (2L_bV_b + V_b²)·len_b·
Q/k_b²` passes its `2^126` bound at α ≈ 2^9.75, and the denominator
`Q = Π k_b²` passes it at α ≈ 2^12.4 (computed on the rank-3 headline table
with the derivation's integer rules). The widths are therefore one common
width, `k = ⌈√(2 ln 2 · Γ²α² · Σ_b γ_b²)⌉
+ 1` (weights `ω_b = γ_b²/Σγ²`), so `Q = k²`. At the headline this widens
the dense blocks by 2.9% and narrows the operand block by 49%; every block
keeps its digit count at α = 16 (checked with `SigmaParams`' derivation
on the headline table).

**Candidates** (headline T = 2^24, k = 2, variant 1, `mqo+eroot`). Ranks,
proof bytes and verifier scans come from `whiteout-costs.py` with ranks
restricted to R_256 (W4 needs one native dimension, A20) and Akita's
`sis_estimator_once` for the new radii; e counts are the script's envelope
model; T_att makes all-rejected ≤ 2^−128.

| per-attempt rejection r | D | α (M − 1) | \|z\| ≤ | e cells (2^ν) | ranks A / [B\|B′] / A_p (×R_256) | proof Δ | verifier scans Δ | T_att | E[attempts] | sampler bits ℓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.58 (today) | 7 | 16 (1.37) | 2^26.9 | 65.5M (2^26) | 3 / 3 / 3 | +7.28 KB | +0.7% | 163 | 2.37 | 51–53 |
| 2^−9.7 | 9 | 1.15·10⁴ (1.2·10^−3) | 2^34.9 | 80.9M (2^27) | 4 / 5 / 4 | +9.63 KB | +17.2% | 14 | 1.001 | 70 |
| **2^−13.2 (default)** | **10** | **2^17 (1.05·10^−4)** | **2^38.9** | **90.1M (2^27)** | **5 / 5 / 5** | **+9.66 KB** | **+23.0%** | **10** | **1.0001** | **77** |
| 2^−21.2 | 12 | 2^25 (4.1·10^−7) | 2^46.9 | 106.7M (2^27) | 6 / 7 / 6 | +9.94 KB | +39.7% | 7 | 1 + 4·10^−7 | 93 |
| 2^−40 | 17 | ≈ 2^44 | 2^66.9 | — | — | — | — | 4 | 1 | — |

- **2^−40 is infeasible.** z exceeds i64 and the 15-digit cap of the
  rejection arithmetic (box ≤ 2^62), and the collision bound (≈ 2^77)
  is outside Akita's SIS estimator.
- **The first extra digit is the cliff.** Any D ≥ 8 crosses e = 2^26, so
  the response proof's domain doubles to 2^27. Above that, proof bytes grow
  by a few hundred bytes, and verifier work grows with the ranks.
- **Error budget.** With `Q_tot ≤ 2^64`, the zero-knowledge terms of Thm
  `wo:thm:zk` are, for today / default / 2^−21.2:
  - `T_att ε/M`: 2^−130.9 / 2^−133.7 / 2^−134.2;
  - `(T_att Q_tot + T_att²)/2^256`: 2^−184.7 / 2^−188.7 / 2^−189.2;
  - total with δ_LHL = 2^−129.2 and the sampler term: 2^−128.8 / 2^−129.1 /
    2^−129.1;
  - plus, since the parallel fill (§2.4.1), the computational term
    `n_ch · Adv^PRG_ChaCha20(t)` with `n_ch ≤ 405·T_att`: about `t·2^−244`
    at the default, at most 2^−144 for `t ≤ 2^100`.

  ε stays at 2^−137: a smaller T_att would allow a larger ε, but that moves
  α by under 1%.
- **Prover time.** The attempt is dominated by the serial Gaussian `fill`,
  whose cost per sample is about `200 + 6.0·ℓ` ns on one thread, with
  ℓ = ⌈log₂ 27k²⌉ the length of its constant-time exponential product
  (`whiteout-gaussian.md`). Expected Σ time per proof at the headline on an
  idle machine (6.08M samples, plus about 0.1 s of other work per attempt):
  today 2.37 × 3.1 ≈ 7.4 s; 2^−9.7: 1.001 × (6.08M × 0.62 µs) ≈ 3.9 s;
  default: 4.1 s; 2^−21.2: 1 × (6.08M × 0.76 µs) ≈ 4.7 s. Measured before
  and after in A16: 1.8× faster. All are 3–6× over the paper's 1.15 s on
  one thread; the parallel fill (§2.4.1) brings the default to about
  0.8 s per proof at 16 threads (A16).

**Default: D = 10, α = 2^17, box failure ≤ 2^−20.** Per-attempt rejection
is at most 2^−13.2, so a proof takes a second attempt about once in 9,400
proofs, and T_att = 10. Reasons:

- It meets "one attempt almost always" at the smallest verifier cost that
  does so. D = 9 retries once in 830 proofs for 6 points less verifier work.
  D = 12 retries once in 2.4M proofs, but costs 17 more points of verifier
  scans and a longer exponential product in the sampler.
- Two constants set the regime, `ALPHA` and `BOX_FAILURE_INVERSE`, both in
  `params.rs`. Moving to D = 12 is `ALPHA = 1 << 25` and
  `BOX_FAILURE_INVERSE = 1 << 28`, plus the setup's ranks; the rejection
  arithmetic, the sampler, the Ajtai map and the digit layout already admit
  it.

**Rejection stays exact.** Smudging needs `y` wider than `cs` by about
2^λ (statistical distance `‖cs‖/σ`), i.e. about 2^128 instead of 2^17 in
α. That is 28 more digits and is infeasible; exact rejection stays.

**Constant time stays; the orchestration does not change shape.** The rule
of §2.8 binds every step computed before an attempt's accept bit. A rejected
attempt still occurs with probability 2^−13.2 per attempt, and its
(y, c, z) are correlated with the witness through the rejection event.
Within the accepted attempt, `y = z − c·s` with z and c public, so a timing
leak of y is a leak of s. Therefore:

- DIT (A11), the isochronous sampler, the constant-time challenge sampler,
  the constant-time `accept`, and `apply_centered`'s constant-time lift all
  stay;
- the attempt nonce and fork stay: H2 needs a fresh secret nonce for every
  attempt that may be rejected, and T_att > 1;
- the loop stays, bounded by T_att = 10 instead of 163. A single attempt
  with whole-proof rerun on rejection would leave a completeness error of
  2^−13 per proof, and it would move the loop into W6 rather than remove it.

What does change is predictability: the attempt count is 1 except with
probability 2^−13.2, so prover time is one attempt's time.

#### Paper deltas

Each item gives the paper's statement, ours, and why, for the authors to
fold in.

| Item | Paper | Here | Why |
|---|---|---|---|
| α, M | α = 16, M ≈ 2.37 (`02_preliminaries.tex:272`, `06_evaluation.tex:96`) | α = 2^9, M = 1 + 2.7·10^−2 | Occasional rejection (§5.2, the cost optimum at measured prices). α = 2^9 is the largest power of two whose box fits eight digits. |
| Widths | per-block `σ_b = αΓγ_bω_b^{−1/2}`, optimized ω_b (`02_preliminaries.tex:273`, `06_evaluation.tex:97-100`) | one width `σ = αΓ√(Σγ_b²)`, i.e. `ω_b = γ_b²/Σγ²` | Keeps the exact rejection arithmetic's denominator `k²` below 2^126 at large α. It satisfies the same hypothesis `‖cs‖_σ ≤ 1/α`; no block changes digit count at α = 16. |
| Box | `p_𝓑 ≥ 0.999` (`06_evaluation.tex:97`) | `p_𝓑 ≥ 1 − 2^−20` | The box failure must sit well below the rejection target; it costs no digit. |
| Response digits, \|z\| | seven radix-16 digits, \|z\| ≈ 2^27 (`06_evaluation.tex:101-102`) | eight digits, \|z\| ≤ 2^30.9 | Width × 2^5. |
| Per-attempt rejection | 1 − 1/2.37 ≈ 0.58 (implied, `06_evaluation.tex:104`) | ≈ 2^−5.23 | As above. |
| T_att, E[attempts] | 163 and 2.37 (`06_evaluation.tex:103-104`) | 25 and 1.027 | `(2^−5.23)^25 ≤ 2^−128`, the same completeness target (Thm `wo:thm:completeness`, `05_security.tex:128-141`). |
| ZK error | about 2^−128.8 with rejection term 2^−130.9 (`06_evaluation.tex:105-107`) | about 2^−129.0 with rejection term 2^−132.4, plus the computational PRG term `n_ch · Adv^PRG_ChaCha20` (≤ 2^−144 for `t ≤ 2^100`) | Thm `wo:thm:zk` (`05_security.tex:332-339`) unchanged except for the mask coins (row below); only T_att moves. |
| Commitment ranks | collision radii up to 2^36, 704–768 rows, rank 3 over R_256 (`06_evaluation.tex:121-133`) | radius `4Γ_Δ B(8) ≈ 2^39.82` (§5.2; it was `4Γ_Δ B(10) ≈ 2^47.8`, where Akita W5e built `A_a` rank 18–19 and `[B_a \| B′_a]` rank 19 over R_64, randomness r_a 610 R_64 columns); Akita's Σ-binding table is regenerated at the new radius. The pad commitment `A_p` at d 128 drops from rank 9 to 8, and r_p from 305 to 273 columns, including row L's 64 coefficients (pinned by jolt-akita's `whiteout_pad_rank_steps_at_the_sigma_binding_brackets`). The residual's as-built ranks have not been re-read | The radius follows the eight-digit box. Ranks are the setup's (Akita planner), not W4's; the planner's dimension ceiling keeps the cut successor at R_64. This supersedes the W4 estimate of 5 / 5 / 5 over R_256 with r_a 161 (see D29). |
| e, ν, proof, verifier | e ≈ 65.5M (2^26); +7.28 KB; verifier scans +0.7% (`06_evaluation.tex:170-185`, `eval_table.tex`) | e ≈ 84M cells by the §5.2 model (8·\|s\| + 0.4M at sha2-chain 2^24, ν = 27); proof 108,107 B against 108,454 B at α = 2^17 (§5.2) | Consequences of the digits and ranks. The 9.66 KB and +23% scan figures were `whiteout-costs.py` at α = 2^17 (R_256 only) and are not recomputed. |
| Prover-time model | 2.37 attempts × 10 ns per sample (`06_evaluation.tex:250-253`) | 1.027 attempts; per-sample cost grows with ℓ = 61 bits, about `93 + 2.04·ℓ` ns on one thread with the truncated product (§5.2); the mask is sampled in parallel | The paper's 10 ns is a per-sample cost with no thread count; ours is per thread, divided by the threads. A16 measured 655 ns at ℓ = 77 before the truncated product. |
| Mask coins | true randomness, implicitly (Thm `wo:thm:zk` is statistical, `05_security.tex:332-339`) | ChaCha20 streams keyed by 256-bit seeds from the prover's RNG, one per 2^14-coordinate chunk | Parallel sampling with an output independent of the thread count. Adds `n_ch · Adv^PRG_ChaCha20` to the ZK error, so the statement becomes computational in that term only (§2.4.1). |
| Constant time | outside the model (H2 discussion, `05_security.tex:355-362`) | unchanged: §2.8's rule still applies | Rejected attempts still occur, and the accepted attempt's y reveals s. |

### 5.2 Occasional rejection (retune, 2026-11-05)

The owner (Q. Dao) relaxed §5.1's almost-never-rejection requirement: pick
the α that minimizes prover and verifier cost at today's measured prices.
This is O7's P8 (`whiteout-o7-sigma-radius.md` §2):

- α = 2^9 (`ALPHA`); D = 8 response digits (`BINDING_RESPONSE_DIGITS`).
- The radius is `R_Σ = 8κ₁·B(8) = 971,235,270,976 ≈ 2^39.82`. It was
  2^47.82. Akita's Σ-binding table is regenerated at this radius.
- Per-attempt rejection is about 2^−5.23, one attempt in 38. E[attempts]
  is 1.027 and T_att is 25.
- ZK: the rejection term `T_att·ε/M` is 2^−132.4 (it was 2^−133.7). The
  sampler term is at most 2^−154 (`whiteout-gaussian.md`). Thm `wo:thm:zk`
  holds unchanged, with only these numbers moved.

α/16² tracks L(D)/16², so ⌊L(8)²/k²⌋ is 76, as at P10. The headroom in
instance size is therefore the same, about 1.7× the headline's N before a
ninth digit.

**Prices.** These are measured on macpro (M4 Max) at sha2-chain 2^24,
after the Σ engineering of the same date. That engineering has three parts:

- The prover's per-row self-check is replaced by the aggregated `ct(h) = 0`.
  This is Schwartz–Zippel over the ρ already drawn, and the per-row scan
  now runs only to name a failing row.
- The parts at z are derived as `parts_y + c·parts_s`.
- The sampler uses a truncated 192-bit product.

The sampler costs about `93 + 2.04·ℓ` ns per sample on one thread, and an
attempt is about 0.38 s at 16 threads. O7's points, priced this way:

- P9 keeps a ninth digit (about 10.45M e cells, roughly 90 ms of response
  proving and one verifier rank at d = 128) to save 0.025 attempts.
- P7 drops a digit but adds 0.51 attempts, about 0.19 s against about
  0.09 s saved in the response. It reaches the same certified rank as P8.

So P8 is the optimum until an attempt falls below about 0.17 s.
**Measured.** macpro (M4 Max, 16 threads), sha2-chain 2^24, four
interleaved runs per build at 1-minute load 7–8. The entries are the
minimum, then the median, in ms. "Engineering" is the three changes above at
α = 2^17; "P8" adds the retune and Akita's regenerated Σ-binding table.
Σ is `whiteout_prove_tail − whiteout_prove_response`.

| | 1666927cc | Engineering | P8 |
|---|---|---|---|
| prove | 5,948 / 6,039 | 5,885 / 6,191 | 5,395 / 5,603 |
| Σ | 811 / 817 | 714 / 750 | 535 / 537 |
| response proof | 1,862 / 1,876 | 1,858 / 2,067 | 1,300 / 1,340 |
| verify, 16 threads | 65 / 74 | 66 / 69 | 57 / 60 |
| verify, 1 thread | 320 / 321 | 321 / 328 | 252 / 255 |
| proof bytes (median) | 108,454 | 108,500 | 108,107 |
| peak RSS (median) | 5.50 GiB | 5.75 GiB | 4.90 GiB |

The first two columns lock Akita at 71a4871. P8 builds against Akita
523826dcd plus the table, and 523826dcd also carries Akita's own P8
response kernels. A second session separates the two effects. Its control
is the engineering build at α = 2^17 on Akita 523826dcd with that
revision's original table. The control matches the engineering column, so
P8's gains come from the retune:

| | control | P8 |
|---|---|---|
| prove | 6,071 / 6,134 | 5,361 / 5,539 |
| Σ | 733 / 740 | 536 / 561 |
| response proof | 1,942 / 1,959 | 1,318 / 1,368 |
| `whiteout_prove_masked` | 622 / 633 | 671 / 692 |
| verify, 1 thread | 324 / 324 | 254 / 256 |
| peak RSS (median) | 5.97 GiB | 4.91 GiB |

`whiteout_prove_masked` is about 50 ms slower under P8 in both sessions;
the cause is not identified. The trace does not record attempts. The eight
P8 Σ times (535–585 ms) are consistent with one attempt each; a rejection
somewhere in eight runs has probability about 19% at 2^−5.23 per attempt.

## 6. Soundness and zero knowledge, mapped to the paper

| Implementation choice | Paper item it discharges |
|---|---|
| Pad commitment sent before the first challenge (M6b) | §4.1, "Pads committed in advance"; `wo:thm:knowledge` |
| ct(g) = 0 sampler; label 0 reserved | `wo:lem:sigma-complete`, `wo:lem:sigma-aggregate`, H6 |
| Distinct nonzero labels; ℓ full-field ρ | `wo:lem:sigma-aggregate` (ℓ/q) |
| Filtered challenge family; differences are units | `wo:lem:sigma-extract`; 2/\|𝓒\| |
| N is R-linear (§1.7) | §2.1, restriction of scalars; `wo:lem:sigma-extract` |
| 𝓑 ⊆ 𝓑_cert; binding radius from 𝓑_cert | §4.3, "acceptance and certified boxes"; radius 4Γ_Δ B_M |
| A_D has no randomness | `wo:thm:knowledge`: an A_D collision on digit vectors |
| `[A_{p,r}; L]` is one offset-0 view, with L as row `rank` | H4 hypothesis |
| Randomness entropy ≥ image + 2λ + 8 bits, fresh per slice | `wo:lem:stat-hiding`; δ_LHL ≤ 2^−129.2 |
| Alphabet and shape check (§2.1, check 1) plus the derived widths' exact check | the hypothesis `‖cs‖_σ ≤ 1/α` of `wo:thm:joint-rejection` and H3 |
| Exact integer v | `wo:def:joint-rejection`, computed exactly |
| y = 0 only on declared public segments | §4.2, "Public coordinates" |
| Secret uniform 256-bit n_i; oracle input (σ ⊇ τ, t_D, n_i) in its own fork domain; only the accepted attempt is sent | H2: Pr[Bad] ≤ (T_att Q_tot + T_att²)/2^256 |
| Constant-time attempt body (§2.8) | keeps H2's "answers at rejected inputs are never observed" true against a timing adversary (outside the paper's model) |
| Fresh y per attempt; M5a rejection | H3: ε/M per attempt |
| T_att = 25 (§5.2) | `wo:thm:completeness`; the `T_att ε/M` term |
| Aggregated ct(h) = 0 check before anything is sent | H6 precondition; no ρ-weighted leak (§2.1) |
| Raw t_p and t_D | Their compressed forms are functions of them (§2.4), so sending them raw removes nothing from H4 or H8: the hybrids already replace the raw images with uniform ones |
| Sampler precision (M5a, truncated products) | at most +2^−154 at the headline and about +2^−150 at larger traces, on 2^−129.0 (`whiteout-gaussian.md`, 2026-11-05 update) |
| ChaCha20 seed expansion (M5a), now in `fill_parallel` | adds a computational term, `n_ch · Adv^PRG_ChaCha20` with `n_ch ≤ 4,050` (§2.4.1) |

## 7. Test plan

1. **Completeness at the predicted abort rate.**
   - Honest proofs over `CheckingResponseProver` verify at D = 64 and 256.
   - Over 300 deterministic proofs each at α = 16 (the paper's) and α = 10, the
     mean attempt count lies within 4σ of the repetition rate M that α and
     `EPSILON_BITS` predict (M ≈ 2.37 and 3.99). The production α rejects
     about one attempt in 38, too rarely to measure this way; the headline
     test pins its T_att = 25.
2. **Tamper rejection.**
   - Flipping a byte of t_D, of n, of t_p or of u_a is rejected.
   - Reordering ρ is rejected.
   - Each of these witness faults fails the prover's aggregated check,
     which names the row: an
     affine row, a product row, a scalarity row (a non-constant operand
     pad), `ct(g) ≠ 0`, and an alphabet or shape violation. With the check
     bypassed through a test-only path, the checking prover rejects the
     result.
   - Pairing a rejected attempt's (t_D, n) with a different attempt's z is
     rejected (run at α = 16, where rejections occur). A whole rejected attempt, with its own z, is a valid proof
     whenever z ∈ 𝓑_cert, and the test asserts that too.
3. **Statistics.**
   - Accepted z per block passes KS and χ² tests against π_0(·|𝓑).
   - A χ² homogeneity test across two witnesses of equal shape passes.
   - Nonce bytes pass a χ² test.
4. **Constant time.**
   - `examples/sigma_dudect.rs` (feature `whiteout-diagnostics`) times each
     step of the attempt body on extreme fixed classes against random ones,
     at D = 64 and 256, with dudect's percentile crops; the constant-time
     challenge sampler is conditioned on its candidate count. Results are
     in §2.8.
   - The `cargo asm` record covers the audit list of §2.8.
   - Both are a W4 slice, and they land before W5's last PR.
5. **Ajtai.**
   - Frozen digests for `apply_centered` and `commit`: one level, with and
     without randomness columns, at d = 64 and 256.
   - Property tests: agreement with a schoolbook negacyclic oracle in F on
     small shapes; chunking across column chunks; view-length rejection on
     truncated streams.
   - The equivalence test against Akita's kernels is W5's.
6. **Algebra.**
   - The polar identity `Q(c,z) + Lz = E_0 + cE_1 + c²h` holds (as a
     property).
   - θ satisfies (4) exactly.
   - `N(c·s) = c·N(s)` under the stride map, and a `d_A ≠ d_B` layout is
     rejected.
7. **Challenge sampler.** The constant-time prover sampler equals the
   verifier's sampler on 10⁵ random seeds, covering seeds that trigger
   operator-norm rejections and position collisions.
8. **Parameters.**
   - The derived widths pass the exact check.
   - Perturbing k down by 1% fails it.
   - The headline shape reproduces the values of §5.

## 8. Implementation slices

| # | Slice | Depends on | Status |
|---|---|---|---|
| 1 | `jolt-crypto::lattice`: `MatrixShape`/`MatrixView`, `GadgetDigits`, `AjtaiLevel`, constant-time `apply_centered` on CRT-NTT (with jolt-algebra's constant-time Garner and branch-free decomposition), one-level `commit`, `sample_box_digits`, `required_randomness_columns`, frozen digests, `TwoLevelLayout` description | M5b | done |
| 2 | `jolt-zk::whiteout::transport`: `AttemptFork`, redacted `Debug` | — | done |
| 3 | `jolt-zk::whiteout`: segment table, alphabet and shape check, `SigmaParams` with derived widths, the exact check and typed errors; `JointRejection::max_attempts` | M5a, M6d | done |
| 4 | `jolt-crypto::challenge`: constant-time prover-side attempt sampler, byte-identical to the verifier's; `SparseChallenge::to_dense_ct` | M5b | done |
| 5 | `RowLabels`, lifting, Q_μ/B_μ | **M6b** relation shapes | done |
| 6 | `LatticePadCommitment` (raw output), `PadPlacement`, `OperandShapeDomain`, `operand_pads` check | 1, 3; **M6b** | done |
| 7 | Σ core, statistics tests | 2, 4, 5, 6 | done |
| 8 | `ResponseStatement`/`ResponseWitness` (no ξ), trait, θ, `CheckingResponseProver`, completeness and tamper tests | 7; W5 alignment note (A15) | done |
| 9 | dudect and asm record | 1, 4, 7 | done (§2.8 record); remaining audit items before W5's last PR |

Moved out of W4:

- to W5: Akita equivalence tests, compression, and fast kernels;
- to W6: `ZkBackend for Whiteout`;
- deleted: the `VectorCommitment::Blind` slice.

## 9. Ambiguity register

**Resolved by review:**

- **A1 (rejected).** Sum-of-products rows. Extension products expand into
  single-term rows with auxiliary slots. That keeps BlindFold's lowering
  free of extra witness variables.
- **A4 (resolved).** There is no new `ResidualVar`. `Witness(j)` is the j-th
  evaluation claim of the backend's residual commitment, and Whiteout's
  opening type carries the points. W3's test plan gains a dense-evaluation
  test pinning three things:
  - Akita's MLE variable order;
  - power-of-two padding;
  - the stride bit permutation.
- **A5 (resolved).** Operand pads are committed only in the operand group.
  `operand_pads` comes from M6b's shape routine, with a release check that
  every `ProductRow.x.pad ∈ operand_pads`.
- **A6 (resolved).** `verify` takes `CommittedPadsView<PadCommitment>` and
  `ResidualCommitment` as arguments (interfaces §5, revised).
- **A10 (resolved).** Widths are derived (§5). The gap to the script's
  optimum is +111,968 cells of e.

**Still open:**

- **A2.** Which side of the cut `response_l2_sq` falls on. Either answer is
  neutral for W4.
- **A3.** W3 declares the public segments of w_k, as whole native elements.
- **A7, module structure (authors).** §2.1 needs the stride embedding, so W4
  requires `d_A = d_B` (§1.7). The paper should state this condition.
- **A8, L (authors).** Confirm `L·r := π_0(row_rank · r)`, and that
  overlapping other roles at offset 0 is acceptable for H4 and H8. The LHL
  statement allows side information that depends on the matrix.
- **A9, box (authors).** L_b is the full positive digit envelope.
- **A11, platform: resolved, DIT set for the attempts and the pad
  commitment.** The §2.8 record shows `to_ring_ct` leaks a zero output on
  M4 without DIT (|t| up to 190 at D = 256), and DIT removes that leak.
  - **Mechanism.** `crates/jolt-dit` owns the `unsafe` and sits outside
    the verifier closure; only `jolt-zk/whiteout-prover` depends on it, and
    jolt-zk keeps `forbid(unsafe_code)`. `DitGuard` sets PSTATE.DIT on the
    current thread (`msr DIT, #1`) and restores the state it read
    (`mrs DIT`) on drop, so guards nest. `jolt_dit::install` runs a closure
    on a fresh rayon pool of `rayon::current_num_threads()` workers, each
    holding a guard from its start handler for its whole life. DIT is per
    thread, so a guard on the caller alone would leave the rayon workers
    unprotected.
  - **Scope.** Two prover entry points run inside such a pool
    (`whiteout::dit::with_dit`):
    - `prove_sigma`, entirely: building s from the residual and pad
      openings, the attempts, every rayon job they spawn, and the response
      proof;
    - `LatticePadCommitment::commit`: the pads' decomposition and the Ajtai
      map on them.

    Everything upstream is outside DIT: whatever computes the residual
    opening, the pads, g and the commitment randomness before these calls.
    `prove_sigma` now needs `R: Send` and `P: Sync`.
  - **Cost.** One pool per call. Measured on a loaded 16-thread M4 Max
    (load average about 51), 100 calls per sample: `install` of an empty
    closure 0.6–0.8 ms, which returns before every worker has started;
    `install` plus a `rayon::broadcast`, which starts all 16 workers as a
    proof does, 8.5–10.3 ms. A second measurement on a loaded machine gave
    4.0 ms for the latter. So 4–10 ms per call, about 1–2% of a 0.49 s proof
    (A16).
  - **Detection.** FEAT_DIT is detected at run time
    (`is_aarch64_feature_detected!("dit")`). A jolt-dit test asserts it is
    detected on aarch64 macOS, so a broken detection cannot silently turn
    the guard into a no-op there. Without FEAT_DIT, and on every other
    target, the guard is a no-op and no error is raised. Intel DOITM is an
    MSR only the kernel can set, out of scope here.
  - **DMP.** The GoFetch authors report that DIT disables the
    data-memory-dependent prefetcher on M3 but not on M1 or M2
    (gofetch.fail). This note has not checked that against Apple's
    documentation, nor measured it on M4. On M1 and M2, DIT is not a DMP
    mitigation.
  - **Residual.** The A11 runs detected none above about 17 ns on `L·y`
    (seed 2, 4 · 10⁵ measurements) or about 38 ns at 10⁵, the sizes of the
    positive controls the slot harness found (§2.8). P7 (§2.8) later found
    two under DIT: the zero-accumulator reconstruction, fixed by the blind
    of §1.3, and an open `N·y` residual for constant non-negative masks. Not covered: work these entry points
    send to another pool or a `std::thread` (none today), and operations
    the attempt uses that DIT does not cover on a given core.
- **A12, compression (W5, authors).**
  - Akita caps a compression source at 8 KiB, i.e. 512 Fp128 coefficients
    (`akita-params/src/compression.rs:19`). t_p and u_a have 768
    coefficients each.
  - W4 sends t_p and t_D raw, about 13 KB, until W5 provides an
    Akita-owned compression for images above 512 coefficients through the
    setup. W3 faces the same limit for u_a.
  - For the authors: the cost script's `FCHAIN_MAX_IMAGE = 884` disagrees
    with Akita's 512-coefficient cap.
- **A13, challenge sampler.** Resolved by slice 4. The equivalence with the
  verifier's sampler is a test (§7, item 7).
- **A14 (authors).** Return ⊥ (an error) after T_att rejections, with the
  caller free to rerun.
- **A15 (W5).** θ and e's logical layout stay in jolt-zk. W5 states its
  physical alignment needs before slice 8.
- **A16, cost.** Measured on the headline shape: inner A (3 × 1,024 R_256)
  on 45 blocks, [B|B′] (3 × 4,417), A_p (3 × 922), then A_D (1 × 18,144
  R_64). The test machine was an Apple M4 Max, release build, with the
  matrices prepared once in 0.20 s on one thread and 0.03 s on 16.

  | Threads | N·y per attempt | N·y + A_D·G⁻¹(V) |
  |---|---|---|
  | 1 | 0.135 s | 0.143 s |
  | 16 | 0.028 s | 0.030 s |

  Whole attempts, measured by `examples/sigma_timing.rs` (feature
  `whiteout-diagnostics`) on the test fixture over R_256 scaled to
  538,368 and 2,111,232 Gaussian coefficients (800,512 and 3,159,808 in s),
  16 threads, release, response proof (W5) excluded:

  | Gaussian coefficients | per attempt | per proof (mean attempts) | fixed per proof |
  |---|---|---|---|
  | 538,368 | 0.227 s | 0.488 s (2.10) | 0.011 s |
  | 2,111,232 | 0.909 s | 1.984 s (2.15) | 0.029 s |

  One thread gives 0.238 s per attempt at the smaller shape: the attempt is
  serial-bound. Instrumented, 0.221 s of the 0.227 s is M5a's Gaussian
  `fill` (0.41 µs per coefficient, one RNG stream); the lift, `N·y`, `L·y`,
  `Q_μ`, `B_μ`, `t_D`, the challenge, `c·s` and the rejection rule take
  7 ms together (13 ns per coefficient).

  Extrapolated to the headline (ΣN = 6,221,056 Gaussian cells, plus the
  `N·y` above): about 2.6 s per attempt and 6.2 s per proof at
  2.37 attempts, against the paper's +4.7% (1.15 s) budget, which assumes a
  10 ns Gaussian sample. Closing the gap needs a parallel fill (independent
  per-block RNG streams forked from the attempt's randomness) in M5a's
  sampler; the Σ-protocol's own work is about 0.08 s per attempt.

  **After almost-never rejection (§5.1), 2026-10-21.** `sigma_timing` at
  scale 5,920 (6,076,160 Gaussian coefficients, the rank-3 headline's
  count; rank 5 has 6,597,888, 8.6% more), 16
  threads, release, 10 proofs per run, before and after binaries
  interleaved twice. The machine was shared and heavily loaded (load
  average 45–140), so absolute times are about twice the idle ones; the
  ratio is the signal.

  | Build | attempts (mean) | per proof, run 1 / run 2 |
  |---|---|---|
  | before (α = 16, two widths) | 2.20 | 16.2 s / 13.0 s |
  | after (α = 2^17, one width) | 1.00 | 7.1 s / 8.7 s |

  - **Ratio.** About 1.8× faster per proof, as predicted: 2.37 attempts
    become 1, and the per-sample cost rises by 655/502 = 1.30 (ℓ = 51 → 77,
    `whiteout-gaussian.md`).
  - **Idle-machine estimate.** One attempt is `fill` at 6.08M × 655 ns ≈
    4.0 s on one thread, plus about 0.1 s of Σ work (`N·y` and `L·y` now
    take two limb passes). Before: 2.37 × (3.05 + 0.08) ≈ 7.4 s.
  - **Sampler share.** About 97% of the attempt, as before.
  - **Budget.** The paper charges 1.15 s. Almost-never rejection removes
    the 2.37× factor but not the per-sample cost, so the parallel fill is
    still needed. Splitting `fill` into fixed-size chunks, each with a
    32-byte seed drawn from the prover's RNG and expanded by ChaCha20, is
    data-independent: chunk boundaries and seeds' use depend only on public
    lengths, so it stays constant time. At 16 threads that is about
    4.0/16 ≈ 0.25–0.3 s. A shorter exponential product (fewer than ℓ = 77
    192-bit multiplications per trial) needs either a Rényi-divergence
    argument for 64-bit arithmetic (M5a Q1, about 5×) or a different exact
    sampler; vectorizing 192-bit products on NEON gains little. The fixed
    part of a trial (CDT scan, ChaCha draws) is about 200 ns per sample, so
    even ℓ → 0 leaves 30× the paper's 10 ns on one thread: threads are the
    lever.

  **Parallel fill (§2.4.1), 2026-10-21.** Same scale, release, serial-fill
  and parallel-fill binaries interleaved twice; load average 33–64.

  | Threads | serial fill, per proof | parallel fill, per proof |
  |---|---|---|
  | 16 | 5.44 s / 4.78 s | 0.81 s / 0.83 s |
  | 1 | 6.68 s / 8.75 s | 5.22 s / 6.88 s |

  `fill` alone on the 6,076,160 coefficients, at the production width,
  inside `with_dit`, best of three: 4.07 s on one thread (670 ns per
  sample) both serially and through `fill_parallel` (no overhead), and
  0.36 s on 16 threads (60 ns per sample in wall time, 11.2× on a loaded
  machine).

  - **Per attempt** (every proof took one): about 0.82 s at 16 threads, of
    which `fill` is 0.36 s (44%); the rest (`N·y` and `L·y` in two limb
    passes, the lifts, `t_D`, the self-checks before ρ and the witness of
    the accepted attempt) is about 0.45 s. At one thread, `fill` is about
    4.1 s of 5–7 s.
  - **Against the budget.** 0.82 s per proof at 16 threads against the
    paper's 1.15 s for the sampling and `N·y` share, on a loaded machine.
    Sampling is no longer the dominant cost.
  - The one-thread rows differ only by noise (the fill alone is 4.07 s
    either way).

**Found in implementation (W4 slices 5–9):**

- **A17, `ResponseProver` signature.** The trait is
  `ResponseProver<F, const D: usize>`, because the statement references the
  setup's native-dimension `N`. Interfaces §5 names `ResponseProver<F>`.
- **A18, `Witness(j)` variable order.** `w̃(r_j)` is over the `Witness`
  segments in table order and native coefficient order, zero-padded, with the
  first point coordinate as the most significant index bit. This is
  provisional until W3 pins Akita's MLE order and the stride permutation.
- **A19, `operand_pads` at commit time.** Resolved. `PadLayout` carries
  `operand_pads`, and the only production producer is
  `OperandShapeDomain::layout`, the shape routine's own domain
  (`with_operand_pads` is crate-private; the field, like `PadLayout`'s
  others, stays public, so a caller can still build a layout by hand, which
  the verifier's comparison rejects). Its production caller is the verifier's
  `pad_layout` (jolt-verifier `stages/zk/residual`), which derives the layout
  it compares against the committed one. `PadPlacement::new` reads the field,
  and `check_relation` requires the relation's layout to equal the
  placement's.
- **A20, one native dimension.** `SigmaSetup` requires every segment, and so
  every block of `N`, to share one native dimension `D`. That covers the
  headline (R_256 for A, B and A_p). This is the check §1.7 describes; it
  surfaces as `WhiteoutError::InvalidRelation`, not a separate
  `NotModuleOverChallengeRing` variant.
- **A21, per-proof setup cost.**
  - `prove_sigma` rebuilds `JointRejection` and the samplers per proof. Their
    320-bit constants cost about 10–20 ms.
  - W6 should cache them alongside the setup.
- **A22, shift independence is not testable.** At the derived widths,
  `‖c s‖_σ ≤ 1/α` puts the shift's effect on z far below sampling error. The
  statistics test therefore checks z's moments and the abort rate; M5a's
  tests cover the rule on large shifts.
- **Measured abort rate.**
  - The mean attempt count was 2.394 over 10,000 proofs, against
    M = 2.371 (standard error 0.018, so 1.3σ).
  - The relation was synthetic: D = 64, 1 product, 2 affine rows, 1 public
    segment.
  - The permanent test runs 300 deterministic proofs at each of α = 16 and
    α = 10 with a 4σ band.

## Not verified by this note

- Akita's commitment semantics come from reading the code. W5's equivalence
  test checks the bytes.
- The headline counts come from the cost script, not from runs of M6b or W3.
- The constant-time audit (§2.8) leaves the NEON/AVX NTT kernels line by
  line open. The dudect record under DIT is from one seed on a loaded
  machine; x86 runs without DOITM (A11).
- Secret buffers (y, rejected z, V, e) are dropped, not zeroized: `zeroize`
  is not a workspace dependency. Open item for W6.
