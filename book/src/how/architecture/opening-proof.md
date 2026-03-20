# Batched opening proof

The final stage ([stage 8](./architecture.md)) of Jolt is the batched opening proof.
Over the course of the preceding stages, we obtain polynomial evaluation claims that must be proven using the [polynomial commitment scheme](../appendix/pcs.md)'s opening proof.
Instead of proving these openings "just-in-time", we **accumulate** them and defer the opening proof to the last stage (see `ProverOpeningAccumulator` and `VerifierOpeningAccumulator` for how this accumulation is implemented).
By waiting until the end, we can [batch-prove](../optimizations/batched-openings.md) all of the openings, instead of proving them individually.
This is important because the [Dory](../dory.md) opening proof is relatively expensive for the prover, so we want to avoid doing it multiple times.

## Claim reduction sumchecks

Throughout the earlier stages of Jolt, various components generate multiple polynomial evaluation claims that need to be consolidated before the final opening proof. Conceptually, claim reduction sumchecks are instantiation of the ["Multiple polynomials, multiple points"](../optimizations/batched-openings.md#multiple-polynomials-multiple-points) subprotocol.

These claim reduction sumchecks serve two purposes:

1. Reduce the number of claims that need to be virtualized by a subsequent sumcheck. E.g. if the same virtual polynomial $P$ is opened at two different points $r_1$ and $r_2$, a claim reduction can be applied to avoid running two instances of the sumcheck that virtualized $P$. 
2. Reduce the number of claims that need to be proven via PCS opening proof. While we can leverage the homomorphic properties of Dory in the [Multiple polynomials, same point](../optimizations/batched-openings.md#multiple-polynomials-same-point) subprotocol, we must first reduce multiple opening points to a single, unified opening point.

The claim reduction sumchecks can be found in `jolt-core/src/zkvm/claim_reductions/` and include:

- **Instruction lookups** (`instruction_lookups.rs`): Aggregates instruction lookup claims (lookup outputs and operands) from Spartan.
- **Registers** (`registers.rs`): Reduces register read/write claims (rd, rs1, rs2) from Spartan.
- **RAM RA** (`ram_ra.rs`): Consolidates the four RAM read-address (RA) claims from various RAM-related sumchecks (raf evaluation, read-write checking, Val evaluation, Val-final evaluation) into a single claim for the RA virtualization sumcheck.
- **Increments** (`increments.rs`): Reduces claims related to increment checks.
- **Hamming weight** (`hamming_weight.rs`): Reduces hamming weight-related claims.
- **Advice** (`advice.rs`): Reduces claims from advice polynomials.
- **Bytecode** (`bytecode.rs`): Reduces committed bytecode openings into the shared Stage 8 Dory geometry.
- **Program image** (`program_image.rs`): Reduces the committed initial-memory image into the same final opening geometry.

### How claim reduction sumchecks work

A claim reduction sumcheck takes multiple polynomial evaluation claims, potentially at different evaluation points, and consolidates them into a single claim (or fewer claims) using the sumcheck protocol. The general pattern is:

1. **Input**: Multiple polynomial evaluation claims of the form $P_i(\mathbf{r}_i) = v_i$
2. **Batching**: Random challenge $\gamma$ is sampled from the transcript to batch the claims together
3. **Sumcheck**: Prove a sumcheck identity of the form:
   $$\sum_{\mathbf{x}} \text{eq}(\mathbf{r}_1, \mathbf{x}) \cdot P_1(\mathbf{x}) + \gamma \cdot \text{eq}(\mathbf{r}_2, \mathbf{x}) \cdot P_2(\mathbf{x}) + \ldots = v_1 + \gamma \cdot v_2 + \ldots$$
4. **Output**: Polynomial evaluation claims of the form $P_i(\mathbf{r}') = v'_i$ for a **single**, unified point $\mathbf{r}'$ derived from the sumcheck challenges.

## Final reduction

After the claim reduction sumchecks have consolidated related claims, we perform a final reduction to prepare for the Dory opening. 

We apply the [Multiple polynomials, same point](../optimizations/batched-openings.md#multiple-polynomials-same-point) subprotocol to reduce the claims to a single claim, namely the evaluation of an `RLCPolynomial` representing a random linear combination of all the opened polynomials.

On the verifier side, this entails taking a linear combination of commitments.
Since Dory is an additively homomorphic commitment scheme, the verifier is able to do so.

### Precommitted geometry and Dory embedding

Some committed polynomials in Stage 8 do not naturally live in the "main" Dory geometry induced by the trace-domain witness polynomials. Examples include the bytecode chunks, the program image, and trusted or untrusted advice. In the implementation these are called **precommitted** polynomials.

The goal of Stage 8 is still the same: every committed polynomial must be opened at one common Dory point so that a single random linear combination can be opened. The subtlety is that these precommitted polynomials may have a different number of variables from the main trace-domain polynomials.

In this section we write:

- $T$ for the **log** trace length
- $K$ for the **log** main address space size
- $B$ for the number of **extra** variables contributed by the largest precommitted polynomial beyond the main geometry

With that notation, the final Dory opening point has length

$$
D = T + K + B.
$$

Equivalently, Stage 8 works in a joint Dory matrix of size $2^{\nu_D} \times 2^{\sigma_D}$ where

$$
\sigma_D = \left\lceil \frac{D}{2} \right\rceil, \qquad \nu_D = D - \sigma_D.
$$

Here $\nu_D$ is the number of **row variables** and $\sigma_D$ is the number of **column variables**. This matches the implementation in `DoryGlobals::balanced_sigma_nu()` and the split used by `PrecommittedClaimReduction::project_dory_round_permutation_for_poly()`.


Write:

- the main geometry size as $T + K$
- the joint geometry size as $D = T + K + B$
- the joint Dory matrix as $2^{\nu_D} \times 2^{\sigma_D}$ with $\nu_D + \sigma_D = D$

The main design constraint is that we do not want to complicate the existing main sumchecks round scheduling. So Jolt does the following:

- precommitted reductions are forward-loaded
- main reductions are backward-loaded
- Stage 6b always has exactly $T + B$ rounds
- Stage 7 always has exactly $K$ rounds

This way:

- the precommitted reductions see the full challenge set needed for the joint geometry
- the main sumchecks keep their old round scheduling
- Stage 8 only has to normalize already-produced opening points into the final Dory point

If some precommitted polynomial already has $D$ variables, we call it a **dominant precommitted polynomial**. Otherwise there is **no dominant precommitted polynomial**, and the joint point is anchored by the ordinary main openings.

#### How Main Polynomials Sit In The Joint Matrix

The main polynomials are embedded depending on the dory layout. As a concrete example, take $D = 5$. Since Dory uses a balanced split, this means:

$$
\sigma_D = 3, \qquad \nu_D = 2,
$$

so the joint matrix has $2^2 = 4$ rows and $2^3 = 8$ columns, for a total of $2^5 = 32$ slots.

##### `CycleMajor` dense placement

Take a dense polynomial with $T = 3$ variables and coefficients

$$
a_{000}, a_{001}, a_{010}, a_{011}, a_{100}, a_{101}, a_{110}, a_{111}.
$$

In `CycleMajor`, the dense polynomial is written across the top of the matrix, so only the lowest $T$ index bits vary:

```text
Joint 4 x 8 matrix

          col000   col001   col010   col011   col100   col101   col110   col111 
row00   | a_000  | a_001  | a_010  | a_011  | a_100  | a_101  | a_110  |  a_111 |
row01   |    .   |    .   |    .   |    .   |    .   |    .   |    .   |    .   |
row10   |    .   |    .   |    .   |    .   |    .   |    .   |    .   |    .   |
row11   |    .   |    .   |    .   |    .   |    .   |    .   |    .   |    .   |
```

so the first $2$ bits are fixed and only the last $3$ bits vary.

##### `AddressMajor` dense placement

Now take the same joint geometry $D = 5$, but the dense polynomial should now use the highest $T = 3$ bits. Then its coefficients are written into slots whose last $K+B = 2$ bits are zero:

In the same $4 \times 8$ matrix this looks like:

```text
Joint 4 x 8 matrix

          col000   col001   col010   col011   col100   col101   col110   col111
row00   | a_000  |    .   |    .   |    .   | a_001  |    .   |    .   |    .   |
row01   | a_010  |    .   |    .   |    .   | a_011  |    .   |    .   |    .   |
row10   | a_100  |    .   |    .   |    .   | a_101  |    .   |    .   |    .   |
row11   | a_110  |    .   |    .   |    .   | a_111  |    .   |    .   |    .   |
```

The same idea applies to one-hot polynomials:

- in `CycleMajor`, they use the lowest $T+K$ bits
- in `AddressMajor`, they use the highest $T+K$ bits, so the trailing $B$ bits are zero

Therefore, the extra $B$ variables must end up on opposite sides of the final Dory opening point in the two layouts.

#### When Address-Major Dense Stride Exceeds The Row Width

In `AddressMajor`, dense polynomials are embedded with stride $2^{K+B}$. Sometimes that stride is larger than the number of columns of the joint matrix. This is the special branch handled in `dory/wrappers.rs`.

Take a real example:

- joint geometry $D = 7$, so the balanced Dory matrix is $2^3 \times 2^4 = 8 \times 16$
- dense polynomial has $T = 2$ variables, so it has 4 coefficients
- therefore $K+B = 5$, so the stride is $2^5 = 32$

Since the row width is only 16, consecutive coefficients jump by two whole rows:

```text
coeff a_00 -> slot  0 -> row 0, col 0
coeff a_01 -> slot 32 -> row 2, col 0
coeff a_10 -> slot 64 -> row 4, col 0
coeff a_11 -> slot 96 -> row 6, col 0
```

and the matrix picture is:

```text
8 x 16 joint matrix

row0  | a_00  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row1  |  .    .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row2  | a_01  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row3  |  .    .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row4  | a_10  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row5  |  .    .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row6  | a_11  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
row7  |  .    .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
```

So the logical embedding is unchanged, but it is no longer a convenient row-local chunking. That is why the implementation switches to explicit sparse row/column placement in this case. Because polynomial lengths are powers of two, the placement still stays aligned: either the stride is a multiple of the row width, so the polynomial occupies the same column range in every row it touches, or the stride divides the row width, so it stays in a fixed column but appears only in every few rows, as in the example above.

#### Final Dory Opening Point

In summary
- in `CycleMajor`, the main dense / one-hot geometry consumes the low bits of the final Dory point, so any extra precommitted variables must sit on the high side
- in `AddressMajor`, the main geometry consumes the high bits, so any extra precommitted variables must sit on the low side
- each block appears in reverse because we always bind polynomials during claim reduction sumchecks from low to high bits

Now we study two cases:
If there **is** a dominant precommitted polynomial, let the raw Stage 6b challenges be

$$
[x_1, x_2, \dots, x_B, x_{B+1}, \dots, x_{B+T}]
$$

and the raw Stage 7 challenges be

$$
[y_1, y_2, \dots, y_K].
$$

The final big-endian Dory opening point is obtained by normalizing these challenges into Dory order.

For **AddressMajor**:

$$
[x_{B+T}, x_{B+T-1}, \dots, x_{B+1} \;\Vert\; y_K, y_{K-1}, \dots, y_1 \;\Vert\; x_B, x_{B-1}, \dots, x_1]
$$

For **CycleMajor**:

$$
[x_B, x_{B-1}, \dots, x_1 \;\Vert\; y_K, y_{K-1}, \dots, y_1 \;\Vert\; x_{B+T}, x_{B+T-1}, \dots, x_{B+1}]
$$

Each block is reversed, and the extra $B$ variables move to different sides depending on the layout.

If there is **no dominant precommitted polynomial**, then the final point is anchored by the ordinary main openings:

- in this case the joint geometry is just the main geometry, so $B = 0$
- let $r_{\mathrm{inc}}$ be the Stage 6b opening point from `IncClaimReduction`
- let $r_{\mathrm{ham}}$ be the Stage 7 opening point from `HammingWeightClaimReduction`

These are already normalized opening points.

Then:

For **AddressMajor**:

$$
r_{\mathrm{final}} =
\big[
r_{\mathrm{inc}}
\;\Vert\;
r_{\mathrm{ham}}
\big]
$$

For **CycleMajor**:

$$
r_{\mathrm{final}} =
\big[
r_{\mathrm{ham}}
\big].
$$

This is exactly the logic implemented in `stage8_opening_point()` in `prover.rs`.

#### Embedding Precommitted Polynomials

The verifier already has the commitment to the precommitted polynomial. That commitment is computed under the convention that the polynomial occupies the top-left block of its balanced Dory matrix, meaning the earliest rows and earliest columns. So when we embed that polynomial into the larger joint matrix, we must preserve that same top-left placement; otherwise the verifier would be checking the Dory proof against a different geometry from the one encoded in the commitment.

```text
Joint Dory matrix: 2^nu_D rows x 2^sigma_D columns
Smaller precommitted matrix: 2^nu_C rows x 2^sigma_C columns

                    left 2^sigma_C cols         remaining cols
                 +---------------------------+------------------+
top 2^nu_C rows  | smaller precommitted poly | not used by this |
                 |        lives here         |      poly        |
                 +---------------------------+------------------+
remaining rows   | not used by this poly     | not used by this |
                 |                           |      poly        |
                 +---------------------------+------------------+
```

Suppose the smaller precommitted polynomial has

$$
C = \nu_C + \sigma_C
$$

variables, while the joint point has

$$
D = \nu_D + \sigma_D.
$$

Split the joint point as

$$
r_{\mathrm{joint}} =
\big[
r_{\mathrm{row}}^{\mathrm{hi}}
\;\Vert\;
r_{\mathrm{row}}^{\mathrm{lo}}
\;\Vert\;
r_{\mathrm{col}}^{\mathrm{hi}}
\;\Vert\;
r_{\mathrm{col}}^{\mathrm{lo}}
\big]
$$

where:

- $r_{\mathrm{row}}^{\mathrm{hi}}$ has length $\nu_D - \nu_C$
- $r_{\mathrm{row}}^{\mathrm{lo}}$ has length $\nu_C$
- $r_{\mathrm{col}}^{\mathrm{hi}}$ has length $\sigma_D - \sigma_C$
- $r_{\mathrm{col}}^{\mathrm{lo}}$ has length $\sigma_C$

Then the smaller polynomial is evaluated on

$$
r_{\mathrm{small}} =
\big[
r_{\mathrm{row}}^{\mathrm{lo}}
\;\Vert\;
r_{\mathrm{col}}^{\mathrm{lo}}
\big].
$$

The reason is that top-left embedding forces the missing high row bits and high column bits to be zero:

```text
joint row variables : [row_hi | row_lo]
joint col variables : [col_hi | col_lo]

top-left embedding forces:
  row_hi = 0
  col_hi = 0
```

So if $P$ is the smaller polynomial and $P_{\mathrm{emb}}$ is its embedding into the joint matrix, then

$$
P_{\mathrm{emb}}(r_{\mathrm{joint}})
=
\operatorname{eq}\!\left(r_{\mathrm{row}}^{\mathrm{hi}}, 0^{\nu_D - \nu_C}\right)
\cdot
\operatorname{eq}\!\left(r_{\mathrm{col}}^{\mathrm{hi}}, 0^{\sigma_D - \sigma_C}\right)
\cdot
P(r_{\mathrm{small}}).
$$

This selector is exactly why top-left embedding works inside one shared Dory proof.

The same selector appears when a joint `RLCPolynomial` mixes a main polynomial with a smaller precommitted polynomial:

$$
\text{RLC coefficient}
\cdot
P(r_{\mathrm{small}})
\cdot
\operatorname{eq}\!\left(r_{\mathrm{row}}^{\mathrm{hi}}, 0^{\nu_D - \nu_C}\right)
\cdot
\operatorname{eq}\!\left(r_{\mathrm{col}}^{\mathrm{hi}}, 0^{\sigma_D - \sigma_C}\right).
$$

#### Permuting Precommitted Polynomial Variables

The precommitted sumchecks still bind variables low-to-high. But the final Dory point order is determined by the joint geometry, not by the order in which those rounds happen.

So Jolt permutes the variables of each precommitted polynomial before running the sumcheck. This keeps the sumcheck code simple while ensuring the final claim corresponds to the original polynomial at the correct Stage 8 point. This permutation is cheap because it is only a variable-position movement, so on the coefficient table it is just a bit permutation of the $2^n$ Boolean-hypercube evaluations.

Here is a concrete 3-variable example. Suppose the original polynomial is encoded by

```text
point       000  001  010  011  100  101  110  111
P(point)     v0   v1   v2   v3   v4   v5   v6   v7
```

Now suppose the Stage 8 geometry wants the variables in the order $(c,b,a)$ rather than $(a,b,c)$. Define

$$
P'(u,v,w) = P(w,v,u).
$$

Then the new coefficient table becomes

```text
point        000  001  010  011  100  101  110  111
P'(point)     v0   v4   v2   v6   v1   v5   v3   v7
```

because

```text
P'(000) = P(000)
P'(001) = P(100)
P'(010) = P(010)
P'(011) = P(110)
P'(100) = P(001)
P'(101) = P(101)
P'(110) = P(011)
P'(111) = P(111)
```

After the sumcheck finishes, `normalize_opening_point()` converts the collected challenges back into the true opening point of the original, non-permuted polynomial.

### `RLCPolynomial`

Recall that all of the polynomials in Jolt fall into one of two categories: **one-hot** polynomials (the $\widetilde{\textsf{ra}}$ and $\widetilde{\textsf{wa}}$ arising in [Twist/Shout](../twist-shout.md)), and **dense** polynomials (we use this to mean anything that's not one-hot).

We use the `RLCPolynomial` struct to represent a random linear combination (RLC) of multiple polynomials, which may include both dense and one-hot polynomials.
We handle the two types separately:

- We **eagerly** compute the RLC of the dense polynomials. So if there are $N$ dense polynomials, each of length $T$ in the RLC, we compute the linear combination of their coefficients and store the result in the `RLCPolynomial` struct as a single vector of length $T$.
- We **lazily** compute the RLC of the one-hot polynomials. So if there are $N$ one-hot polynomials, we store $N$ (coefficient, reference) pairs in `RLCPolynomial` to represent the RLC. Later in the Dory opening proof when we need to compute a vector-matrix product using `RLCPolynomial`, we do so by computing the vector-matrix product using the individual one-hot polynomials (as well as the dense RLC) and taking the linear combination of the resulting vectors.
