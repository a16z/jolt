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
- **`Fork::attempt` in `jolt-transcript`**: the oracle domain of the attempt
  challenges.

**Out of W4:**

- `VectorCommitment` keeps its scalar blind. The lattice pad commitment is
  not a `VectorCommitment`.
- The compression chain moves to W5, together with the Akita equivalence test
  and any fast kernels. W4 sends t_p and t_D uncompressed (A12).

## 0. Notation and the private vector

`R = F[X]/(X^64 + 1)` with `F = Prime128OffsetA7F7` and `q = 2^128 − 2^32 +
22537`. R is the challenge ring of §2.1, with d = 64.

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
`shape.ring_dim == D`. It offers three operations:

- `apply_centered(x: &[i64], bound) -> Vec<CyclotomicRing<F, D>>`: the map
  `M·x` on centered integer columns with `|x| ≤ bound < 2^31`. This is the
  Σ-protocol's `N·y`, and the core of the other two.
- `apply(x: &[CyclotomicRing<F, D>])`: the same map on field columns. It
  splits each column into 16 radix-256 planes, at 16 times the cost.
- `commit(level, prepared, x, r)`. It computes
  `apply_centered(G⁻¹(x) ‖ r, 128)`, where `r` is box digits.

**Implementation, one path for all three.** The public matrix is prepared once
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

- `sample_box_digits(rng, len, log_basis) -> Vec<i8>` draws uniform digits of
  the balanced alphabet. Each digit is a masked byte, so the draw is constant
  time. Distinct vectors then differ by a unit (§2.3: coordinates below
  `q^{1/4}/2`).
- `required_randomness_columns(image_coeffs, extra_coeffs, ring_dim, log_basis, slack_bits)`
  returns the smallest whole number of ring columns whose entropy is at least
  `(image_coeffs + extra_coeffs)·⌈log₂ q⌉ + 2λ + slack_bits`.

The extra term is `extra_coeffs = 64` for the pad commitment's row **L**, and
the slack is `slack_bits = 8` (§6.2). With the rank-3 matrices over R_256 this
gives:

| Randomness | Columns | Cells |
|---|---|---|
| r_p | 105 R_256 | 26,880 |
| r_a | 97 R_256 | 24,832 |

The script's 26,752 and 24,704 round to R_64 elements instead. Each slice
draws its own randomness (§4.1).

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

The error is `ZkError::Setup(NotModuleOverChallengeRing)`. The headline
layouts use R_256 throughout and pass.

## 2. The Σ-protocol

### 2.1 Inputs and checks before ρ

`whiteout::prove` takes three groups of inputs:

- from M6b: the `ResidualRelation` and the `PadOpening`;
- from W3, through W6: the residual opening `(w_k, t̂_a, r_a)`, its image
  `u_a`, the evaluation points of its claims (A4), and the public segments
  of w_k;
- from the setup: the layouts and `SigmaParams` (§5), plus a
  `ResponseProver`.

It assembles **s** in segment order: w_k, t̂_a, r_a, regular digits and g
digits, operand digits, r_p.

Before any transcript operation it runs the following **release checks**.
Each failure is fatal and maps to a typed `ZkError`.

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
2. **Every row of R_res, individually.** This covers each affine row, each
   product row, each scalarity row, and `ct(g) = 0`, evaluated on the
   decoded values.
3. **N·s = t,** with t the raw images. They must equal the sent commitments:
   t_p as sent by `commit_pads`, and u_a as W3 sent it.

After ρ is squeezed and before σ, it computes `h = Q_μ(1, s)` and asserts
`ct(h) = 0`. Given check 2 this cannot fail, so the assertion is a release
invariant check.

**Why check 2 runs before ρ.** h is a coordinate of V, V lies in e, and the
response proof is not zero knowledge. If ct(h) ≠ 0, then ct(h) =
Σ_i μ_i·violation_i would publish one field element, a ρ-weighted combination
of the violated rows. H6 needs `ct(Q′) = 0`. Checking each row before ρ keeps
the transcript free of any such value.

### 2.2 Lifting the rows

Each row becomes an R-linear form in `(u, s)` (`wo:lem:ct-lift`), built with
`LiftedFunctional` (M6d):

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
   per Gaussian segment. Public segments get `y = 0`.
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
   - `c_i = SampleChallenge(Fork::attempt(σ, n_i).absorb(t_D))`, from the
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

`Fork::attempt` is a fresh sponge. It absorbs the tag
`jolt-transcript/fork/attempt/v1`, then σ, then the 32-byte nonce. It is
separate from the grinding forks, which are keyed by a `u32` counter, so the
attempt queries have their own oracle domain (H2: "a separate domain of the
oracle"). A counter would not do: H2 needs a nonce that is uniform and secret
until it is published.

A fork lives outside the transcript. It records no `TranscriptEvent`, and a
rejected attempt leaves nothing in the replay log.

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
| all T_att attempts rejected | `ZkError::SigmaAttemptsExhausted`. The caller may rerun the whole proof with fresh randomness. | Probability ≤ 2^−128. H3 makes the event independent of the witness, and the simulator outputs ⊥ at the same rate (A14). |
| `SampleError::ShiftOutOfRange` | `ZkError::InvariantViolation`, fatal. Never retried, never recorded. | Unreachable after check 1 of §2.1. |
| other `SampleError`, layout errors | `ZkError::Setup(..)`, fatal. | Public parameters. |
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
| Gaussian `fill` | y | M5a sampler (isochronous, reviewed). |
| `N·y`, `A_D·G⁻¹(V)` | y, V | `apply_centered` of §1.3: branch-free Montgomery lift, NTT with public twiddles, constant-time Garner, no LUT. |
| `Q_μ(0,y)`, `B_μ`, `L·y`, `g(y)`, `Λ_μ(0,y)` | y | Branch-free centered lift into F. Ring products with public or s-only operands. |
| `G⁻¹(V)` | V | The bias path of `BalancedDecomposePow2Params::biased`. Its SIMD variants are on the audit list. |
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

Per-attempt buffers are allocated once and overwritten. No tracing span carries
an attempt value. The attempt body is `#[inline(never)]`, so the review covers
the production monomorphization.

**Platform.** The argument assumes that integer multiply, add and compare take
data-independent time. Neither arm64 DIT nor Intel DOITM is enabled, and this
is recorded as an accepted assumption (A11).

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
pub struct ResponseStatement<F> {
    pub challenge: SparseChallenge,            // c
    pub rho: Vec<F>,                           // ρ; μ is derived by RowLabels
    pub layout: ResponseLayout,                // segments of e, below
    pub residual_image: Vec<F>,                // u_a (raw until W5 compresses it)
    pub pad_image: Vec<F>,                     // t_p (raw)
    pub v_image: Vec<F>,                       // t_D (raw)
    pub ring_linear: RingLinearStatement,      // N: residual TwoLevelLayout, pad AjtaiLevel, segment table
    pub quadratic: QuadraticStatement<F>,      // see below
}
```

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
  `ZkError::InvariantViolation`.

At the headline shape, e has about 65.5M cells (§5), which is 65 MB as `i8`.

### 3.4 Trait

The trait is that of interfaces §5:

```rust
pub trait ResponseProver<F> {
    fn prove_response<H: Sponge>(&self, statement: &ResponseStatement<F>,
        witness: &ResponseWitness<F>, transcript: &mut ProverTranscript<H>) -> Result<(), ZkError>;
    fn verify_response<H: Sponge>(&self, statement: &ResponseStatement<F>,
        transcript: &mut VerifierTranscript<'_, H>) -> Result<(), ZkError>;
}
```

Its contract:

- The commitment to e is the first transcript event after n.
- `verify_response` checks Items 1–5 over all of e, public z segments
  included.
- Errors map to `ZkError::Response(..)`.

W4's tests use `CheckingResponseProver`, test support that evaluates Items 1–5
directly on the witness. It is the statement's own semantics, not a second
implementation of an Akita rule.

## 4. Whiteout's `ZkBackend` (W6)

W6 implements the revised trait of interfaces §5 once W3's opening type exists:

| Associated type | Whiteout's choice |
|---|---|
| `PadCommitment` | raw t_p, from `LatticePadCommitment` |
| `PadOpening` | pads, g and r_p |
| `ResidualCommitment` | raw u_a, passed to `verify` as an argument |
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
| challenge family | d = 64, 31 × ±1, 11 × ±2, ‖c‖_op ≤ Γ = 18, 2^128.06 elements, κ₁ = 53 | `jolt-crypto::challenge` (M5b) | `SigmaParams::new` (`validate_min_entropy`) |
| α, ε | 16, 2^−137 (M ≈ 2.37) | `SigmaParams` | `JointRejection::new` |
| widths k_b | derived (below): 6,765,904 for B = 8 dense segments; 13,531,806 for operand digits | `SigmaParams::derive(segments)` | exact rational check (below) |
| digits D_b, D_pub | 7 for every Gaussian segment; 3 | derived | `L_b ≥` the box bound (below); `D_pub` covers `κ₁·B` |
| box L_b | `7(16^{D_b}−1)/15`, the positive end of the certified range (A9) | derived | `Σ_b 64·len_b·2^{1−⌊L_b²/k_b²⌋} ≤ 10^{−3}`, which gives p_𝓑 ≥ 0.999 |
| shift bound | `κ₁ · B_b` | derived | `JointRejection::accept` |
| T_att | 163 | derived | smallest T with `(1 − (p_𝓑 − ε)/M)^T ≤ 2^−128`. It bounds only the prover's retries, so it is computed in `f64`, with the acceptance probability shaded down by 2^−40 |
| ν | 256 | const | — |
| ℓ | 14 | `RowLabels` | — |
| A_a, [B_a\|B′_a], A_p ranks | 3 × R_256 each | Akita planner and SIS tables (rule 3), in the setup | Akita (SIS); W4 checks module structure and view lengths |
| A_D | 1 × R_64, binding on radix-16 digits | Akita planner | Akita; view length |
| randomness | r_a 97 and r_p 105 R_256 columns, radix 16 | layouts | `required_randomness_columns` |
| pad radices | 16 (32 digits), operands 256 (16 digits) | `LatticePadCommitment` consts | `GadgetDigits` checks |

**Width derivation** (A10). Widths are derived in jolt-zk with integer
arithmetic, not tabulated. Prover and verifier must agree bit for bit, because
D_b fixes the e layout.

1. **Closed form.** Take the continuous optimum of §6.2,
   `ω_b = N_b/ΣN` over Gaussian cells. The width is
   `σ_b = αΓγ_b·√(ΣN/N_b)`, and so
   `k_b = ⌈αΓ·√(2 ln 2 · γ_b² · ΣN / N_b)⌉ + 1`.
   - ln 2 is a fixed-point upper bound with 2^−64 error.
   - The final +1 leaves relative slack of about 10^−7, far above the
     ceiling errors.
2. **Exact check.** `Σ_b 2·LN2_HI·Γ²γ_b²α²/k_b² ≤ 1` is evaluated in
   fixed point with ceilings. This is `‖cs‖_σ ≤ 1/α` for every c and every s
   that passes check 1 of §2.1. `SigmaParams::new` runs it on whatever widths
   it is given, so the derivation is checked too.
3. **Digits.** `D_b` is the smallest D with
   `⌊L(D)²/k_b²⌋ ≥ 1 + ⌈log₂(2000·N_tot)⌉`, followed by the exact box check
   above.

**Gap to the script's discrete optimum.**

- The script gives 6 digits to r_a, pads and r_p, and 7 to w, t̂ and the
  operands. It does this by stretching each width to its digit envelope.
- The closed form gives 7 everywhere. That adds 111,968 cells to e
  (65,450,122 → 65,562,090, still below 2^26), so the padded domain and the
  proof size do not change.
- Porting the discrete search in exact arithmetic is possible later and
  would recover those cells. It is not needed now.

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
| T_att = 163 | `wo:thm:completeness`; the `T_att ε/M` term |
| Per-row checks before ρ; ct(h) assertion | H6 precondition; no ρ-weighted leak (§2.1) |
| Raw t_p and t_D | Their compressed forms are functions of them (§2.4), so sending them raw removes nothing from H4 or H8: the hybrids already replace the raw images with uniform ones |
| Sampler precision (M5a) | +2^−149 on 2^−128.8 |
| ChaCha20 seed expansion (M5a) | adds a PRG assumption; stated in the module docs |

## 7. Test plan

1. **Completeness at the predicted abort rate.**
   - 2,000 honest proofs over `CheckingResponseProver` all verify.
   - The mean attempt count lies within a 5σ binomial band of `M/p_𝓑`.
   - A run at α = 10 (M ≈ 3.99) moves the rate measurably.
2. **Tamper rejection.**
   - Flipping a byte of t_D, of n, of t_p or of u_a is rejected.
   - Reordering ρ is rejected.
   - Each of these witness faults fails the prover's per-row check: an
     affine row, a product row, a scalarity row (a non-constant operand
     pad), `ct(g) ≠ 0`, and an alphabet or shape violation. With the check
     bypassed through a test-only path, the checking prover rejects the
     result.
   - Pairing a rejected attempt's (t_D, n) with a different attempt's z is
     rejected. A whole rejected attempt, with its own z, is a valid proof
     whenever z ∈ 𝓑_cert, and the test asserts that too.
3. **Statistics.**
   - Accepted z per block passes KS and χ² tests against π_0(·|𝓑).
   - A χ² homogeneity test across two witnesses of equal shape passes.
   - Nonce bytes pass a χ² test.
4. **Constant time.**
   - `examples/sample_dudect.rs` gains the attempt body (fixed versus random
     y-seed and c, s fixed), `apply_centered`, and the constant-time
     challenge sampler (fixed versus random fork seeds). Each passes
     |t| < 4.5.
   - The `cargo asm` record covers the audit list of §2.8.
   - Both are a W4 slice, and they land before W5's last PR.
5. **Ajtai.**
   - Frozen digests for `apply_centered`, `apply` and `commit`: one level,
     with and without randomness columns, at d = 64 and 256.
   - Property tests: `apply_centered = apply ∘ lift`; agreement with a
     schoolbook negacyclic oracle on small shapes; chunking at the
     `CrtCapacity` boundary; view-length rejection on truncated streams.
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
   - Perturbing any k_b down by 1% fails it.
   - The headline shape reproduces the values of §5.

## 8. Implementation slices

| # | Slice | Depends on | Status |
|---|---|---|---|
| 1 | `jolt-crypto::lattice`: `MatrixShape`/`MatrixView`, `GadgetDigits`, `AjtaiLevel`, constant-time `apply`/`apply_centered` on CRT-NTT (with jolt-algebra's constant-time Garner), one-level `commit`, `sample_box_digits`, `required_randomness_columns`, frozen digests, `TwoLevelLayout` description | M5b | now |
| 2 | `jolt-transcript`: `Fork::attempt` | — | now |
| 3 | `jolt-zk::whiteout`: segment table, alphabet and shape check, `SigmaParams` with derived widths and the exact check | M5a, M6d | now |
| 4 | `jolt-crypto::challenge`: constant-time prover-side attempt sampler, byte-identical to the verifier's | M5b | now |
| 5 | `RowLabels`, lifting, Q_μ/B_μ | **M6b** relation shapes | later |
| 6 | `LatticePadCommitment` (raw output), `operand_pads` check | 1, 3; **M6b** | later |
| 7 | Σ core, statistics tests | 2, 4, 5, 6 | later |
| 8 | `ResponseStatement`/`ResponseWitness` (no ξ), trait, θ, `CheckingResponseProver`, completeness and tamper tests | 7; W5 alignment note (A15) | later |
| 9 | dudect and asm record | 1, 4, 7 | later, before W5's last PR |

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
- **A11, platform.** DIT/DOITM are not enabled. W4 records this as an
  accepted assumption; a follow-up needs a narrowly scoped `unsafe` wrapper
  outside the verifier closure.
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

  At 2.37 expected attempts, that is about 0.07 s per proof on 16 threads.
  This is within the paper's +4.7% (1.15 s) budget, which also assumes a
  10 ns Gaussian sample (M5a measured 0.5 µs on one thread).

## Not verified by this note

- Akita's commitment semantics come from reading the code. W5's equivalence
  test checks the bytes.
- The headline counts come from the cost script, not from runs of M6b or W3.
- The constant-time audit list (§2.8) is pending slice 9.
