# Jolt Recursion via SNARK Composition

## 1. Motivation & Overview

### 1.1 Why Recursion

Recursion enables two critical capabilities:

1. **Proof Aggregation**: Maintain a single succinct proof verifying entire blockchain state, enabling light clients and efficient synchronization.

2. **On-Chain Verification**: Produce proofs with single-digit millisecond verification on mobile and reasonable EVM gas costs.

### 1.2 The Approach

The Jolt verifier compiled to RISC-V requires ~1.5 billion cycles. Our target is <10 million cycles (~150× reduction).

We decompose the verifier:

$$\mathcal{V}_{\text{Jolt}} = \mathcal{V}_{\text{light}} \circ \mathcal{H}$$

where:
- $\mathcal{H} = \{h_1, \ldots, h_m\}$ are **hints** for expensive operations
- $\mathcal{V}_{\text{light}}$ assumes hints are correct
- A **bespoke SNARK** proves hints are well-formed

The expensive operations (from Dory PCS verification):

| Operation | Description |
|-----------|-------------|
| G1 Scalar Mul | $[k]P$ for $P \in \mathbb{G}_1$ |
| G2 Scalar Mul | $[k]Q$ for $Q \in \mathbb{G}_2$ |
| GT Exponentiation | $a^k$ for $a \in \mathbb{G}_T$ |
| GT Multiplication | $a \cdot b$ for $a, b \in \mathbb{G}_T$ |
| Multi-Pairing | $\prod_i e(P_i, Q_i)$ |

### 1.3 Protocol Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Packed GT Exp Sum-Check                                   │
│  ─────────────────────────────                                      │
│  Packed GT exp → virtual polynomial claims at r_x                   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Batched Constraint Sum-Checks                             │
│  ─────────────────────────────────                                  │
│  Shift + reduction + GT mul + G1 → claims at r_x                    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Direct Evaluation                                         │
│  ────────────────────────────────────                               │
│  Combine all claims into M(s,x) → claim M(r_s, r_x)                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: Jagged Transform Sum-Check                                │
│  ────────────────────────────────────────────                       │
│  Sparse M → Dense q → claim q(r_dense)                              │
└─────────────────────────────────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 5: Jagged Assist Sum-Check                                   │
│  ────────────────────────────────────────────                       │
│  Batch MLE verification for jagged                                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PCS Opening (Hyrax over Grumpkin)                                  │
│  ────────────────────────────────────────────                       │
│  Prove q(r_dense) = v_dense → final verification                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Field Choice

All SNARK arithmetic is over $\mathbb{F}_q$ (BN254 base field), which equals the Grumpkin scalar field. This choice is dictated by:
- GT witnesses produce $\mathbb{F}_q$ elements (from Fq12 representation)
- Hyrax PCS operates over Grumpkin, whose scalar field is $\mathbb{F}_q$

---

## 2. Stage 1: Constraint Sum-Checks

Stage 1 proves correctness of packed GT exponentiation via constraint-specific sum-checks. The output claims feed into Stage 2.

### 2.1 Ring Switching & Quotient Technique

$\mathbb{G}_T$ elements are represented as $\mathbb{F}_{q^{12}} = \mathbb{F}_q[X]/(p(X))$ where $p(X)$ is an irreducible polynomial of degree 12.

**Key insight**: For $a, b, c \in \mathbb{G}_T$, the equation $a \cdot b = c$ holds iff there exists quotient $Q$ such that:
$$a(X) \cdot b(X) = c(X) + Q(X) \cdot p(X)$$

On the Boolean hypercube $\{0,1\}^4$ (viewing elements as 4-variate polynomials):
$$a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

where $g(x)$ is the MLE of $p$ on the hypercube.

This transforms high-degree $\mathbb{F}_{q^{12}}$ operations into low-degree constraints over $\mathbb{F}_q$ by introducing the quotient as auxiliary witness.

### 2.2 GT Exponentiation

Computes $b = a^k$ using a packed base-4 square-and-multiply.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Public Input | $a \in \mathbb{G}_T$ | Base (as Fq12) |
| Public Input | $k \in \mathbb{F}_r$ | Exponent |
| Public Output | $b \in \mathbb{G}_T$ | Result $b = a^k$ |

#### Witness

| Symbol | Description |
|--------|-------------|
| $(u_0, v_0), \ldots, (u_{s-1}, v_{s-1})$ | Base-4 digits of $k$ (lo/hi bits) |
| $\rho_0, \ldots, \rho_s$ | Intermediate values: $\rho_0 = 1$, $\rho_s = a^k$ |
| $Q_0, \ldots, Q_{s-1}$ | Quotient polynomials |

Recurrence (base-4):
$$\rho_{i+1} = \rho_i^4 \cdot a^{d_i}, \quad d_i \in \{0,1,2,3\}$$

#### Packed Witness Structure

We pack all 128 base-4 steps into unified 11-variable MLEs:

| Symbol | Description | Layout |
|--------|-------------|--------|
| $\rho(s, x)$ | Intermediate values | $\rho[x \cdot 128 + s] = \rho_s[x]$ |
| $\rho_{\text{next}}(s, x)$ | Shifted intermediate values | $\rho_{\text{next}}[x \cdot 128 + s] = \rho_{s+1}[x]$ |
| $Q(s, x)$ | Quotient polynomials | $Q[x \cdot 128 + s] = Q_s[x]$ |
| $\text{digit}_{\text{lo}}(s)$ | Low digit bit | Replicated across $x$ |
| $\text{digit}_{\text{hi}}(s)$ | High digit bit | Replicated across $x$ |
| $\text{base}(x), \text{base}^2(x), \text{base}^3(x)$ | Base powers | Replicated across $s$ |

Where:
- $s \in \{0,1\}^7$ indexes the step (0 to 127)
- $x \in \{0,1\}^4$ indexes the field element (0 to 15)
- Layout formula: `index = x * 128 + s` (s in low bits)

#### Unified Constraint

Let $d(s)$ be the base-4 digit from $(u_s, v_s)$ and
$$\text{base}^{d(s)} = w_0 + w_1 \cdot \text{base}(x) + w_2 \cdot \text{base}^2(x) + w_3 \cdot \text{base}^3(x)$$
with $w_0=(1-u)(1-v)$, $w_1=u(1-v)$, $w_2=(1-u)v$, $w_3=uv$.

Then the packed constraint is:
$$C(s, x) = \rho_{\text{next}}(s, x) - \rho(s, x)^4 \cdot \text{base}(x)^{d(s)} - Q(s, x) \cdot g(x) = 0$$

#### Two-Phase Sum-Check

$$0 = \sum_{s \in \{0,1\}^7} \sum_{x \in \{0,1\}^4} \text{eq}(r_s, s) \cdot \text{eq}(r_x, x) \cdot C(s, x)$$

- **Phase 1** (rounds 0-6): Bind step variables $s$
- **Phase 2** (rounds 7-10): Bind element variables $x$
- Total: 11 rounds, degree 7

#### Output Claims

After final challenges $(r_s^*, r_x^*)$:
- `PackedGtExpRho(i)`: $\rho(r_s^*, r_x^*)$
- `PackedGtExpRhoNext(i)`: $\rho_{\text{next}}(r_s^*, r_x^*)$
- `PackedGtExpQuotient(i)`: $Q(r_s^*, r_x^*)$

The base and digit MLE evaluations are derived from public inputs (not committed).
- The two-phase sumcheck maintains the same security as 254 individual sumchecks
- Batching with $\gamma$ preserves zero-knowledge properties

#### Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Polynomials per GT exp | 1,024 | 5 | 204.8× |
| Virtual claims | 1,024 | 5 | 204.8× |
| Proof size contribution | ~32KB | ~160B | 200× |

---

### 2.2.1 Shift Sumcheck Optimization

The packed GT exponentiation can be further optimized by eliminating the `rho_next` polynomial commitment using a shift sumcheck protocol.

#### The Problem

Currently, we commit to three polynomials per packed GT exponentiation:
- `rho(s,x)` - intermediate values at each step
- `rho_next(s,x)` - shifted values where `rho_next(s,x) = rho(s+1,x)`
- `quotient(s,x)` - quotient polynomials

Since `rho_next` is completely determined by `rho` through the shift relationship, committing to it is redundant.

#### The Solution: Shift Sumcheck

Instead of committing to `rho_next`, we prove the shift relationship algebraically. After the constraint sumcheck completes at point `(r_s*, r_x*)`, we run an additional sumcheck to prove:

$$v = \sum_{s \in \{0,1\}^8} \sum_{x \in \{0,1\}^4} \text{EqPlusOne}(r_s^*, s) \cdot \text{eq}(r_x^*, x) \cdot \rho(s,x)$$

Where:
- `EqPlusOne(r_s*, s)` = 1 if `s = r_s* + 1`, 0 otherwise
- The sum evaluates to exactly `rho(r_s*+1, r_x*)` = `rho_next(r_s*, r_x*)`

#### Protocol Flow

```
Stage 1: Packed GT exp sumcheck
  - Outputs: virtual claims for rho, rho_next, quotient at r1
  ↓
Stage 2: Batched constraints
  - Includes shift sumcheck and claim reduction to r2
  ↓
Stage 3: Direct Evaluation Protocol
  - Uses all virtual claims at r2
```

---

### 2.3 GT Multiplication

Proves $c = a \cdot b$ for $a, b, c \in \mathbb{G}_T$.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Public Input | $a, b \in \mathbb{G}_T$ | Operands |
| Public Output | $c \in \mathbb{G}_T$ | Result $c = a \cdot b$ |

#### Witness

| Symbol | Description |
|--------|-------------|
| $Q$ | Quotient polynomial: $Q(x) = \frac{a(x) \cdot b(x) - c(x)}{g(x)}$ |

#### Constraint

$$C(x) = a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

#### Sum-Check

For $m$ multiplication constraints:
$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{m-1} \gamma^i \cdot C_i(x)$$

#### Output Claims

- `RecursionMulLhs(i)`: $\tilde{a}_i(r_x')$
- `RecursionMulRhs(i)`: $\tilde{b}_i(r_x')$
- `RecursionMulResult(i)`: $\tilde{c}_i(r_x')$
- `RecursionMulQuotient(i)`: $\tilde{Q}_i(r_x')$

---

### 2.4 G1 Scalar Multiplication

Proves $Q = [k]P$ using double-and-add.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Public Input | $P \in \mathbb{G}_1$ | Base point $(x_P, y_P)$ |
| Public Input | $k \in \mathbb{F}_r$ | Scalar with bits $(b_0, \ldots, b_{n-1})$ |
| Public Output | $Q \in \mathbb{G}_1$ | Result $Q = [k]P$ |

#### Witness

Execution trace with $n$ rows (padded to $N = 2^\ell$):

| Column | Description |
|--------|-------------|
| $(x_A, y_A)$ | Accumulator at start of iteration |
| $(x_T, y_T)$ | Doubled point: $T = [2]A$ |
| $(x_{A'}, y_{A'})$ | Accumulator after conditional addition |
| $\text{ind}$ | 1 if $T = \mathcal{O}$ (infinity), 0 otherwise |

Each row $i$: $A_{i+1} = T_i + b_i \cdot P$

#### Constraints

**C1 (Doubling x)**: $4y_A^2(x_T + 2x_A) - 9x_A^4 = 0$

**C2 (Doubling y)**: $3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0$

**C3 (Addition x)**: Handles both finite and infinity cases via indicator:
$$\text{ind} \cdot x_{A'} \cdot (x_{A'} - x_P) + (1 - \text{ind}) \cdot [\text{addition formula}] = 0$$

**C4 (Addition y)**: Similarly for y-coordinate.

| Constraint | Degree |
|------------|--------|
| C1 | 4 |
| C2 | 3 |
| C3 | 6 |
| C4 | 6 |

#### Sum-Check

$$0 = \sum_{x \in \{0,1\}^\ell} \text{eq}(r, x) \cdot \sum_{j=1}^{4} \delta^{j-1} \cdot C_j(x)$$

- $\ell = \lceil \log_2 n \rceil$ rounds
- Degree 6 (maximum constraint degree)

#### Output Claims

- `RecursionG1ScalarMulXA(i)`, `RecursionG1ScalarMulYA(i)`
- `RecursionG1ScalarMulXT(i)`, `RecursionG1ScalarMulYT(i)`
- `RecursionG1ScalarMulXANext(i)`, `RecursionG1ScalarMulYANext(i)`
- `RecursionG1ScalarMulIndicator(i)`

---

### 2.5 [Future] G2 Scalar Multiplication

Similar to G1, but over the twist curve $E'(\mathbb{F}_{q^2})$.

*Pending implementation.*

---

### 2.6 [Future] Multi-Pairing

Computes $T = \prod_{i=1}^m e(P_i, Q_i)$ via:
1. Miller loop accumulation
2. Final exponentiation
3. Line evaluation

*Pending implementation.*

---

## 3. Stage 3: Direct Evaluation Protocol

After Stage 2, we have many virtual polynomial claims $(v_0, v_1, \ldots, v_{n-1})$ at point $r_x$. Stage 3 verifies these claims directly without sumcheck.

### 3.1 Why Direct Evaluation

The constraint matrix $M$ has a special structure:
$$M(i, r_x) = v_i \text{ for all } i$$

Therefore:
$$M(r_s, r_x) = \sum_{i \in \{0,1\}^{\log n}} \text{eq}(r_s, i) \cdot M(i, r_x) = \sum_{i} \text{eq}(r_s, i) \cdot v_i$$

This allows direct computation without sumcheck rounds.

### 3.2 Matrix Organization

Define matrix $M : \{0,1\}^s \times \{0,1\}^x \to \mathbb{F}_q$ where:
- Row index $s$ selects polynomial type and constraint index
- Column index $x$ selects evaluation point (11 bits)

Row indexing:
```
row = poly_type × num_constraints_padded + constraint_idx
```

Polynomial types:

| Type | Index | Source |
|------|-------|--------|
| RhoPrev | 0 | Packed GT exp |
| Quotient | 1 | Packed GT exp |
| MulLhs | 2 | GT Mul |
| MulRhs | 3 | GT Mul |
| MulResult | 4 | GT Mul |
| MulQuotient | 5 | GT Mul |
| G1ScalarMulXA | 6 | G1 Scalar Mul |
| G1ScalarMulYA | 7 | G1 Scalar Mul |
| G1ScalarMulXT | 8 | G1 Scalar Mul |
| G1ScalarMulYT | 9 | G1 Scalar Mul |
| G1ScalarMulXANext | 10 | G1 Scalar Mul |
| G1ScalarMulYANext | 11 | G1 Scalar Mul |
| G1ScalarMulIndicator | 12 | G1 Scalar Mul |

**Total**: 13 polynomial types

### 3.3 Direct Evaluation Protocol

**Input**: Virtual claims $\{v_i\}$ from Stage 1 at point $r_x$

**Protocol**:
1. **Sample**: $r_s \leftarrow \mathbb{F}^{\log n}$ from transcript
2. **Prover**:
   - Evaluate $M(r_s, r_x)$ by binding matrix to challenges
   - Send evaluation to verifier
3. **Verifier**:
   - Compute $\text{eq}_{\text{evals}} = \text{EqPolynomial::evals}(r_s)$
   - Compute $v = \sum_i \text{eq}_{\text{evals}}[s_i] \cdot v_i$ where $s_i = \text{matrix\_s\_index}(i)$
   - Verify prover's evaluation matches $v$

**Output**: Opening claim $M(r_s, r_x) = v_{\text{sparse}}$

#### Mathematical Correctness

**Theorem**: Direct evaluation is sound with the same security as sumcheck.

**Proof**: The multilinear extension $\tilde{M}$ is unique. For the boolean hypercube:
$$\tilde{M}(s, x) = M(s, x) \text{ for all } s \in \{0,1\}^{\log n}, x$$

By linearity of multilinear extensions:
$$\tilde{M}(r_s, r_x) = \sum_{i \in \{0,1\}^{\log n}} M(i, r_x) \cdot \text{eq}(r_s, i)$$

Since $M(i, r_x) = v_i$ by construction:
$$\tilde{M}(r_s, r_x) = \sum_i v_i \cdot \text{eq}(r_s, i)$$

The verifier computes exactly this sum, so the protocol is perfectly sound. □

**Security Guarantees**:
- No additional soundness error (deterministic protocol)
- Fiat-Shamir security maintained through transcript inclusion
- Binding: Prover committed to $v_i$ values in Stage 1

**Benefits**:
- Eliminates $\log n$ sumcheck rounds
- Reduces proof size by $3\log n$ field elements
- Verifier work remains $O(n)$

---

## 4. Stage 4: Jagged Transform Sum-Check

The matrix $M$ is sparse: 11-variable polynomials (GT exp/GT mul) and 8-variable polynomials (G1) are zero-padded to a common width. Stage 4 compresses to a dense representation.

### 4.1 Why Jaggedness

| Operation | Native Variables | Native Size | Padded Size |
|-----------|-----------------|-------------|-------------|
| GT Exp/Mul | 11 | 2048 | 2048 (native) |
| G1 Scalar Mul | 8 | 256 | 2048 (zero-padded) |

The jagged transform eliminates redundant zeros, reducing commitment size.

### 4.2 Sparse-to-Dense Bijection

**Sparse**: $M(s, x)$ with row $s$ and column $x \in \{0,1\}^{11}$

**Dense**: $q(i)$ containing only non-redundant entries

Bijection:
- `row(i)` = polynomial index for dense index $i$
- `col(i)` = evaluation index within that polynomial
- Cumulative sizes track where each polynomial's entries begin

### 4.3 Differences from Jagged PCS Paper

| Paper | Our Implementation |
|-------|-------------------|
| Column-wise jaggedness | Row-wise jaggedness |
| Direct bijection to matrix | Three-level mapping |
| Simple row/col | poly_idx → (constraint_idx, poly_type) → matrix_row |

The three-level mapping handles multiple polynomial types per constraint.

### 4.4 The Jagged Indicator Function

$$\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}}) = \sum_{y \in \text{rows}} \text{eq}(r_s, y) \cdot \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$$

where $g(a, b, c, d) = 1$ iff $b < d$ AND $b = a + c$.

### 4.5 Branching Program Optimization

The function $\hat{g}$ can be computed in $O(n)$ time via a width-4 read-once branching program:
- Tracks carry bit (addition check: $b = a + c$)
- Tracks comparison bit ($b < d$)
- Processes bits LSB to MSB

This avoids naive $O(2^{4n})$ computation.

### 4.6 Sum-Check Protocol

**Input**: $M(r_{s,\text{final}}, r_x) = v_{\text{sparse}}$ from Stage 3

**Relation**:
$$v_{\text{sparse}} = \sum_{i \in \{0,1\}^{\ell_{\text{dense}}}} q(i) \cdot \hat{f}_{\text{jagged}}(r_s, r_x, i)$$

**Protocol**:
1. Prover materializes dense $q(i)$ and indicator $\hat{f}_{\text{jagged}}$
2. Sum-check over $\ell_{\text{dense}}$ rounds (degree 2)
3. At challenge $r_{\text{dense}}$: claim $q(r_{\text{dense}}) = v_{\text{dense}}$
4. Verifier computes $\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}})$ via branching program
5. Verify: output = $v_{\text{dense}} \cdot \hat{f}_{\text{jagged}}$

**Output**: $(q, r_{\text{dense}}, v_{\text{dense}})$

### 4.7 Stage 5: Jagged Assist

After Stage 4's sumcheck, the verifier must compute:
$$\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}}) = \sum_{y \in [K]} \text{eq}(r_s, y) \cdot \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$$

where $K$ is the number of polynomials. For large systems, this requires $K \times O(\text{bits})$ field operations.

#### The Jagged Assist Protocol

Instead of computing all $K$ evaluations, we use batch verification:

1. **Prover sends**: $v_y = \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$ for all $y \in [K]$

2. **Verifier samples**: Batching coefficient $r \leftarrow \mathbb{F}$

3. **Batch claim**:
   $$\sum_{y=0}^{K-1} r^y \cdot \hat{g}(x_y) = \sum_{y=0}^{K-1} r^y \cdot v_y$$

4. **Sumcheck**: Over $P(b) = g(b) \cdot \sum_y r^y \cdot \text{eq}(b, x_y)$

5. **Final verification**: At random point $\rho$:
   - Compute $\hat{g}(\rho)$ using branching program
   - Compute $\sum_y r^y \cdot \text{eq}(\rho, x_y)$
   - Verify $P(\rho) = \hat{g}(\rho) \cdot \text{eq\_sum}$

#### Mathematical Correctness

**Theorem (Jagged Assist Soundness)**: If $\exists y^* : v_{y^*} \neq \hat{g}(x_{y^*})$, then the verifier accepts with probability at most $\frac{2m}{|\mathbb{F}|}$.

**Proof Sketch**:
1. Define polynomial $R(Z) = \sum_{y=0}^{K-1} Z^y \cdot (\hat{g}(x_y) - v_y)$
2. If any $v_{y^*}$ is incorrect, then $R(Z) \not\equiv 0$ and $\deg(R) \leq K-1$
3. By Schwartz-Zippel, $\Pr[R(r) = 0] \leq \frac{K-1}{|\mathbb{F}|}$
4. If $R(r) \neq 0$, the sumcheck polynomial $P$ differs from the honest polynomial
5. Sumcheck soundness: $\Pr[\text{accept} | R(r) \neq 0] \leq \frac{2m}{|\mathbb{F}|}$
6. Union bound: $\Pr[\text{accept}] \leq \frac{K-1 + 2m}{|\mathbb{F}|} \approx \frac{2m}{|\mathbb{F}|}$

**Key Insight**: The protocol converts $K$ individual checks into one batched sumcheck with negligible soundness loss.

#### Benefits

| Metric | Without Assist | With Assist | Improvement |
|--------|----------------|-------------|-------------|
| Verifier ops | $K \times 1,300$ | $330K$ | ~64× |
| Extra rounds | 0 | $2m$ | Acceptable |
| Soundness error | 0 | $\frac{2m}{|\mathbb{F}|}$ | Negligible |

The protocol leverages Lemma 4.6 (forward-backward decomposition) for efficient prover computation.

---

## 5. PCS Opening Proof (Hyrax over Grumpkin)

The PCS proves the dense polynomial claim using Hyrax.

### 5.1 Why Hyrax

- **No pairing required**: Works over any curve with efficient MSM
- **Grumpkin-native**: Scalar field = $\mathbb{F}_q$, matching our constraint field
- **Square matrix efficiency**: Optimal for virtualized polynomials

### 5.2 Commitment Structure

For polynomial $q$ with $2^n$ evaluations:
1. Reshape as $2^{n_r} \times 2^{n_c}$ matrix
2. Commit to each row via Pedersen over Grumpkin
3. Commitment: $C = \{C_0, \ldots, C_{2^{n_r}-1}\}$

### 5.3 Opening Protocol

**Input**: Commitment $C$, point $r_{\text{dense}}$, claimed value $v_{\text{dense}}$

**Protocol**:
1. Decompose point into row/column components
2. Sum-check proves tensor product structure
3. Column opening proves consistency
4. Verifier checks against commitment

### 5.4 Final Verification

The verifier accepts iff all checks pass:
- Stage 1-5 sum-check verifications
- Hyrax opening verification

---

## 6. Parameters & Cost Analysis

This section provides analytical formulas for proof sizes, constraint counts, and computational costs.

### 6.1 Sumcheck Degrees and Rounds

| Stage | Protocol | Degree | Rounds | Elements/Round |
|-------|----------|--------|--------|----------------|
| 1 | Packed GT Exponentiation | 7 | 11 | 8 |
| 2 | Batched constraints (shift + reduction + GT mul + G1) | $\leq 6$ | 11 | 8 |
| 3 | Direct Evaluation | - | 0 | 1 |
| 4 | Jagged Transform | 2 | $d$ | 3 |
| 5 | Jagged Assist | 2 | $2m$ | 3 |

Where:
- $\ell = \lceil \log_2 n \rceil$ for $n$-bit scalar
- $d = \lceil \log_2(\text{dense\_size}) \rceil$
- $m = $ number of jagged indicator evaluations (typically $\log K$ where $K$ is polynomial count)

### 6.2 Constraint Counts

**Per-operation constraints**:

| Operation | Constraints | Poly Types | Total Polynomials |
|-----------|-------------|------------|-------------------|
| GT Exp (packed, $t$-bit) | 1 | 2 | 2 |
| GT Mul | 1 | 4 | 4 |
| G1 Scalar Mul ($n$-bit) | 1 | 7 | 7 |

**Typical Dory verification** (256-bit scalars):
- GT exponentiations (packed): 1 unified constraint covering all 128 base-4 steps
- GT multiplications: 1 constraint each
- G1 scalar multiplications: 1 constraint each (but 256 rows in trace)

### 6.3 Matrix Dimensions

**Row count**:
$$\text{num\_rows} = 13 \times c_{\text{pad}}$$

where $c_{\text{pad}} = 2^{\lceil \log_2 c \rceil}$ and $c$ = total constraints.

**Column count**:
- $2^{11} = 2048$ for packed GT exp / GT mul constraints (11-variable MLEs)
- $2^8 = 256$ for G1 operations (8-variable MLEs, zero-padded to 11 vars)

**Example** ($c = 10$ constraints):

| Parameter | Formula | Value |
|-----------|---------|-------|
| $c_{\text{pad}}$ | $2^{\lceil \log_2 10 \rceil}$ | 16 |
| num\_rows | $13 \times 16$ | 208 |
| num\_s\_vars | $\lceil \log_2 208 \rceil$ | 8 |

### 6.4 Dense Size Computation

The jagged transform compresses the sparse matrix by removing zero-padding:

$$\text{dense\_size} = \sum_{\text{poly } p} 2^{\text{num\_vars}(p)}$$

where:
- GT polynomials: $\text{num\_vars} = 11 \Rightarrow 2048$ evaluations
- G1 polynomials: $\text{num\_vars} = 8 \Rightarrow 256$ evaluations

**Example** (3 GT exp + 2 GT mul + 1 G1 scalar mul):

| Source | Polys | Evals/Poly | Total |
|--------|-------|------------|-------|
| GT Exp (3 constraints × 2 types) | 6 | 2048 | 12,288 |
| GT Mul (2 constraints × 4 types) | 8 | 2048 | 16,384 |
| G1 Scalar Mul (1 constraint × 7 types) | 7 | 256 | 1,792 |
| **Total** | 21 | — | **30,464** |

$\Rightarrow d = \lceil \log_2 30464 \rceil = 15$ rounds for Stage 4.

### 6.5 Proof Size Formulas

**Stage 1** (packed GT exp sumcheck):
$$|P_1| = (\text{degree} + 1) \times \text{rounds}$$

| Type | Degree | Rounds | Elements |
|------|--------|--------|----------|
| Packed GT Exp | 7 | 11 | 88 |

**Stage 2** (batched constraint sumcheck):
$$|P_2| = (\text{degree} + 1) \times \text{rounds}$$

| Type | Degree | Rounds | Elements |
|------|--------|--------|----------|
| Batched constraints | $\leq 6$ | 11 | $\leq 77$ |

**Stage 3** (direct evaluation):
$$|P_3| = 1 \text{ element}$$

**Stage 4** (jagged transform):
$$|P_4| = 3d \text{ elements}$$

**Stage 5** (jagged assist):
$$|P_5| = K + 3(2m) \text{ elements}$$
where $K$ is the number of $\hat{g}$ evaluations sent

**PCS opening** (Hyrax):
$$|P_{\text{pcs}}| = O(\sqrt{\text{dense\_size}}) \text{ group elements}$$

**Total proof size** (field elements, excluding PCS):
$$|P| = |P_1| + |P_2| + |P_3| + |P_4| + |P_5| + \text{virtual claims}$$

### 6.6 Concrete Example: Single 256-bit GT Exponentiation (packed)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Constraints ($c$) | 1 | Single unified constraint |
| $c_{\text{pad}}$ | 1 | Already power of 2 |
| Polynomials | 2 | RhoPrev + Quotient |
| num\_s\_vars ($s$) | 4 | $\lceil \log_2(13 \times 1) \rceil$ |
| dense\_size | 4,096 | $2 \times 2048$ (11-var MLEs) |
| num\_dense\_vars ($d$) | 12 | $\lceil \log_2 4096 \rceil$ |
| Virtual claims | 3 | Rho, RhoNext, Quotient |

At 32 bytes per $\mathbb{F}_q$ element:
- Virtual claims: $3 \times 32 = 96$ bytes (excluding PCS proof)

### 6.7 Prover Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Packed GT exp sumcheck | $O(2^{11} \cdot \text{degree})$ |
| Stage 2 | Batched constraints | $O(2^{11} \cdot \text{degree})$ |
| Stage 3 | Direct evaluation | $O(2^s)$ |
| Stage 4 | Jagged transform | $O(\text{dense\_size})$ |
| Stage 5 | Jagged assist | $O(K \cdot \text{bits})$ |
| PCS | Hyrax commitment | $O(\text{dense\_size})$ MSMs |

**Dominant cost**: Stage 1 sumcheck computation scales with constraint count.

**Parallelization**: Stage 2 batches multiple constraint types under a shared transcript.

### 6.8 Verifier Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck verification | $O(\text{rounds} \cdot \text{degree})$ |
| Stage 2 | Sumcheck verification | $O(\text{rounds} \cdot \text{degree})$ |
| Stage 3 | Claim aggregation | $O(c)$ |
| Stage 4 | Branching program | $O(\text{num\_polys} \cdot \text{bits})$ |
| PCS | Hyrax verification | $O(\sqrt{\text{dense\_size}})$ |

**Key efficiency**: Stage 4 verifier uses $O(n)$ branching program instead of naive $O(2^{4n})$ MLE evaluation.

---

## 7. Implementation

This section describes the code architecture and data flow for the recursion implementation.

### 7.1 Module Structure

```
jolt-core/src/zkvm/recursion/
├── mod.rs                    # Re-exports and module docs
├── witness.rs                # Witness types (GTExpWitness, GTMulWitness, G1ScalarMulWitness)
├── constraints_sys.rs        # DoryMatrixBuilder, DoryMultilinearMatrix, ConstraintSystem
├── bijection.rs              # VarCountJaggedBijection, JaggedTransform trait
├── recursion_prover.rs       # RecursionProver orchestrating all stages
├── recursion_verifier.rs     # RecursionVerifier
├── stage1/
│   ├── packed_gt_exp.rs         # Packed GT exponentiation sumcheck
│   ├── shift_rho.rs             # Shift sumcheck for rho(s+1,x)
│   ├── gt_mul.rs                # GT multiplication sumcheck
│   └── g1_scalar_mul.rs         # G1 scalar mul sumcheck
├── stage2/
│   ├── packed_gt_exp_reduction.rs # Reduction from r1 to r2 for rho/quotient
│   └── virtualization.rs        # Direct evaluation protocol
└── stage3/
    ├── jagged.rs                # Jagged transform sumcheck
    ├── jagged_assist.rs         # Jagged assist (batch MLE verification)
    └── branching_program.rs     # O(n) MLE evaluation for g function
```

### 7.2 The Offloading Pattern

The recursion system uses a **hint-based offloading** pattern to reduce verification cost:

```
┌────────────────────────────────────────────────────────────────────┐
│                        PROVER SIDE                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Dory Proof ──→ witness_gen() ──┬──→ WitnessCollection (full)    │
│                                  └──→ HintMap (compact)            │
│                                                                    │
│   WitnessCollection ──→ RecursionProver ──→ RecursionProof        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (send proof + hints)
┌────────────────────────────────────────────────────────────────────┐
│                       VERIFIER SIDE                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   HintMap + RecursionProof ──→ verify_with_hint() ──→ Accept/Reject│
│                                                                    │
│   (Avoids expensive witness regeneration)                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The `RecursionExt` trait defines this interface:

```rust
pub trait RecursionExt<F: JoltField>: CommitmentScheme<Field = F> {
    type Witness;  // Full witness data for proving
    type Hint;     // Compact hints for verification

    /// Generate witness and hints from a Dory proof
    fn witness_gen(&self, proof: &ArkDoryProof, ...) -> (Self::Witness, Self::Hint);

    /// Verify using hints instead of regenerating witnesses
    fn verify_with_hint(&self, proof: &RecursionProof, hint: &Self::Hint, ...) -> Result<()>;
}
```

### 7.3 Witness Types

Each expensive operation has a corresponding witness structure:

**GT Exponentiation** (`GTExpWitness`):
```rust
struct GTExpWitness {
    base: Fq12,                    // Base element a
    exponent: Fr,                  // Scalar k
    result: Fq12,                  // Result a^k
    rho_mles: Vec<Vec<Fq>>,        // Intermediate ρ values as MLEs
    quotient_mles: Vec<Vec<Fq>>,   // Quotient Q_i for each step
    bits: Vec<bool>,               // Binary decomposition of k
}
```

**GT Multiplication** (`GTMulWitness`):
```rust
struct GTMulWitness {
    lhs: Fq12,                     // Left operand a
    rhs: Fq12,                     // Right operand b
    result: Fq12,                  // Result c = a·b
    quotient_mle: Vec<Fq>,         // Quotient polynomial
}
```

**G1 Scalar Multiplication** (`G1ScalarMulWitness`):
```rust
struct G1ScalarMulWitness {
    base: G1Affine,                // Base point P
    scalar: Fr,                    // Scalar k
    result: G1Affine,              // Result [k]P
    x_a_mle: Vec<Fq>,              // Accumulator x-coordinates
    y_a_mle: Vec<Fq>,              // Accumulator y-coordinates
    x_t_mle: Vec<Fq>,              // Doubled point x-coordinates
    y_t_mle: Vec<Fq>,              // Doubled point y-coordinates
    x_a_next_mle: Vec<Fq>,         // Next accumulator x-coordinates
    y_a_next_mle: Vec<Fq>,         // Next accumulator y-coordinates
    indicator_mle: Vec<Fq>,        // Infinity indicators
    bits: Vec<bool>,               // Binary decomposition of k
}
```

**Packed GT Exponentiation**:

The packed representation combines all 128 base-4 steps into unified 11-variable polynomials:

```rust
struct PackedGTExpWitness {
    base: Fq12,                    // Base element a
    exponent: Fr,                  // Scalar k
    result: Fq12,                  // Result a^k

    // Packed polynomials (11 variables each)
    rho_packed: Vec<Fq>,           // ρ(s,x) for all steps
    rho_next_packed: Vec<Fq>,      // ρ_next(s,x) shifted by 1
    quotient_packed: Vec<Fq>,      // Q(s,x) for all steps
    digit_lo_packed: Vec<Fq>,      // low digit bit replicated across x
    digit_hi_packed: Vec<Fq>,      // high digit bit replicated across x
    base_packed: Vec<Fq>,          // base(x) replicated across s
    base2_packed: Vec<Fq>,         // base^2(x) replicated across s
    base3_packed: Vec<Fq>,         // base^3(x) replicated across s
}
```

Layout: For 11-variable MLEs with `s ∈ {0,1}^7` and `x ∈ {0,1}^4`:
- Index formula: `index = x * 128 + s` (s in low bits)
- `rho_packed[x * 128 + s] = ρ_s[x]`
- Each MLE has 2^11 = 2,048 evaluations

### 7.4 Constraint System Construction

The `DoryMatrixBuilder` constructs the constraint matrix from witnesses:

```rust
let mut builder = DoryMatrixBuilder::new();

// Add witnesses (each becomes one or more constraints)
for witness in gt_exp_witnesses {
    builder.add_packed_gt_exp_witness(witness);  // 1 constraint per exponentiation
}
for witness in gt_mul_witnesses {
    builder.add_gt_mul_witness(witness);  // 1 constraint per multiplication
}
for witness in g1_scalar_mul_witnesses {
    builder.add_g1_scalar_mul_witness(witness);  // 1 constraint per scalar mul
}

let constraint_system: ConstraintSystem = builder.build();
```

The resulting `DoryMultilinearMatrix` has shape:

```
M : {0,1}^num_s_vars × {0,1}^num_x_vars → Fq

where:
  num_s_vars = ⌈log₂(15 × num_constraints_padded)⌉
  num_x_vars = 11 (packed constraint vars for GT exp/GT mul/G1)
```

### 7.5 Prover Flow

```rust
impl RecursionProver {
    pub fn prove(
        constraint_system: &ConstraintSystem,
        witnesses: &WitnessCollection,
        transcript: &mut Transcript,
    ) -> RecursionProof {
        // Stage 1: Packed GT exp sumcheck
        let (stage1_proof, r_stage1) = BatchedSumcheck::prove(
            vec![PackedGtExpProver::new(...)],
            &mut accumulator,
            transcript,
        );

        // Stage 2: Batched constraint sumchecks
        // (shift rho + GT exp reduction + GT mul + G1 scalar mul)
        let (stage2_proof, r_stage2) = BatchedSumcheck::prove(
            vec![ShiftRhoProver::new(...),
                 PackedGtExpClaimReductionProver::new(...),
                 GtMulProver::new(...),
                 G1ScalarMulProver::new(...)],
            &mut accumulator,
            transcript,
        );

        // Stage 3: Virtualization (direct evaluation)
        let (stage3_m_eval, r_stage3_s) = DirectEvaluationProver::new(...)
            .prove(transcript, &mut accumulator, r_s);

        // Stage 4: Jagged transform sumcheck
        let (stage4_proof, r_stage4) = BatchedSumcheck::prove(
            vec![JaggedSumcheckProver::new(...)],
            &mut accumulator,
            transcript,
        );

        // Stage 5: Jagged assist
        let stage5_proof = BatchedSumcheck::prove(
            vec![JaggedAssistProver::new(...)],
            &mut accumulator,
            transcript,
        );

        // PCS opening
        let opening_proof = accumulator.prove_openings(pcs, transcript);

        RecursionProof {
            stage1_proof,
            stage2_proof,
            stage3_m_eval,
            stage4_proof,
            stage5_proof,
            opening_proof,
        }
    }
}
```

### 7.6 Verifier Flow

```rust
impl RecursionVerifier {
    pub fn verify(
        proof: &RecursionProof,
        verifier_input: &RecursionVerifierInput,
        transcript: &mut Transcript,
    ) -> Result<()> {
        // Stage 1: Verify packed GT exp sumcheck
        let r_stage1 = BatchedSumcheck::verify(
            vec![PackedGtExpVerifier::new(...)],
            &proof.stage1_proof,
            &mut accumulator,
            transcript,
        )?;

        // Stage 2: Verify batched constraints
        let r_stage2 = BatchedSumcheck::verify(
            vec![ShiftRhoVerifier::new(...),
                 PackedGtExpClaimReductionVerifier::new(...),
                 GtMulVerifier::new(...),
                 G1ScalarMulVerifier::new(...)],
            &proof.stage2_proof,
            &mut accumulator,
            transcript,
        )?;

        // Stage 3: Verify virtualization
        DirectEvaluationVerifier::new(...)
            .verify(transcript, &mut accumulator, proof.stage3_m_eval, r_s)?;

        // Stage 4: Verify jagged transform
        // Verifier computes f_jagged using branching program (O(n) time)
        let r_stage4 = BatchedSumcheck::verify(
            vec![JaggedSumcheckVerifier::new(...)],
            &proof.stage4_proof,
            &mut accumulator,
            transcript,
        )?;

        // Stage 5: Verify jagged assist
        BatchedSumcheck::verify(
            vec![JaggedAssistVerifier::new(...)],
            &proof.stage5_proof.sumcheck_proof,
            &mut accumulator,
            transcript,
        )?;

        // PCS verification
        accumulator.verify_openings(pcs, &proof.opening_proof, transcript)?;

        Ok(())
    }
}
```

### 7.7 Opening Accumulator

The `OpeningAccumulator` tracks virtual polynomial claims across stages:

```rust
// Stage 1 appends virtual claims (rho, rho_next, quotient at r1)
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRhoNext(w),
    SumcheckId::PackedGtExp,
    opening_point,
    claimed_value,
);

// Stage 2 reads Stage 1 claims and writes reduced claims at r2
let (point, value) = accumulator.get_virtual_polynomial_opening(
    VirtualPolynomial::PackedGtExpRhoNext(w),
    SumcheckId::PackedGtExp,
);

// Stage 4 outputs committed polynomial claim (dense matrix)
accumulator.append_committed(
    CommittedPolynomial::DoryDenseMatrix,
    opening_point,
    claimed_value,
);
```

### 7.8 Jagged Bijection

The `VarCountJaggedBijection` maps between sparse and dense indices:

```rust
impl JaggedTransform for VarCountJaggedBijection {
    /// Given dense index i, return which polynomial it belongs to
    fn row(&self, dense_idx: usize) -> usize {
        // Binary search in cumulative_sizes
        self.cumulative_sizes.partition_point(|&s| s <= dense_idx)
    }

    /// Given dense index i, return offset within that polynomial
    fn col(&self, dense_idx: usize) -> usize {
        let poly_idx = self.row(dense_idx);
        dense_idx - self.cumulative_sizes[poly_idx]
    }

    /// Map (poly_idx, eval_idx) to dense index, if valid
    fn sparse_to_dense(&self, poly_idx: usize, eval_idx: usize) -> Option<usize> {
        if eval_idx < self.poly_sizes[poly_idx] {
            Some(self.cumulative_sizes[poly_idx] + eval_idx)
        } else {
            None  // Padded region
        }
    }
}
```

### 7.9 Matrix Row Mapping

The three-level mapping converts polynomial indices to matrix rows:

```rust
// Level 1: Dense index → Polynomial index (via bijection)
let poly_idx = bijection.row(dense_idx);

// Level 2: Polynomial index → (constraint_idx, poly_type)
let (constraint_idx, poly_type) = mapping.decode(poly_idx);

// Level 3: (constraint_idx, poly_type) → Matrix row
let matrix_row = poly_type as usize * num_constraints_padded + constraint_idx;
```

This mapping is precomputed for the verifier:

```rust
let matrix_rows: Vec<usize> = (0..num_polynomials)
    .map(|poly_idx| {
        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        poly_type as usize * num_constraints_padded + constraint_idx
    })
    .collect();
```

### 7.10 Polynomial Type Enumeration

```rust
#[repr(usize)]
pub enum PolyType {
    // Packed GT Exponentiation (11-var MLEs)
    RhoPrev = 0,
    Quotient = 1,

    // GT Multiplication (11-var MLEs)
    MulLhs = 2,
    MulRhs = 3,
    MulResult = 4,
    MulQuotient = 5,

    // G1 Scalar Multiplication (8-var MLEs, zero-padded)
    G1ScalarMulXA = 6,
    G1ScalarMulYA = 7,
    G1ScalarMulXT = 8,
    G1ScalarMulYT = 9,
    G1ScalarMulXANext = 10,
    G1ScalarMulYANext = 11,
    G1ScalarMulIndicator = 12,
}

pub const NUM_POLY_TYPES: usize = 13;
```

### 7.11 Dory Integration: Witness Extraction and Hints

The recursion system integrates with Dory through the `RecursionExt` trait, which enables automatic witness extraction during verification.

#### The TraceContext Mechanism

Dory's `verify_recursive` function accepts a `TraceContext` that captures witnesses during verification:

```rust
// TraceContext modes:
TraceContext::for_witness_gen()  // Collect witnesses during verify
TraceContext::for_hints(hint)    // Use precomputed hints instead
```

When `verify_recursive` executes with a witness-gen context:
1. Each GT exponentiation triggers `WitnessGenerator::generate_gt_exp()`
2. Each GT multiplication triggers `WitnessGenerator::generate_gt_mul()`
3. Each G1 scalar mul triggers `WitnessGenerator::generate_g1_scalar_mul()`
4. Results accumulate in the shared `TraceContext`

#### witness_gen Implementation

```rust
fn witness_gen(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut Transcript,
    point: &[Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
) -> Result<(WitnessCollection, HintMap), ProofVerifyError> {
    // Create context that will collect witnesses
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_witness_gen()
    );

    // Run verification - witnesses collected as side effect
    verify_recursive(
        commitment, evaluation, point, proof, setup,
        transcript, ctx.clone(),
    )?;

    // Extract collected witnesses
    let witnesses = Rc::try_unwrap(ctx)
        .ok().expect("sole ownership")
        .finalize()?;

    // Convert to compact hints
    let hints = witnesses.to_hints::<BN254>();

    Ok((witnesses, hints))
}
```

#### JoltWitnessGenerator

The generator creates witness structures using optimized step computation:

```rust
impl WitnessGenerator<JoltWitness, BN254> for JoltWitnessGenerator {
    fn generate_gt_exp(base: &GT, scalar: &Fr, result: &GT) -> JoltGtExpWitness {
        // ExponentiationSteps computes all intermediate ρ values and quotients
        let steps = ExponentiationSteps::new(base.0, ark_to_jolt(scalar));
        JoltGtExpWitness {
            base: steps.base,
            exponent: steps.exponent,
            result: steps.result,
            rho_mles: steps.rho_mles,          // All ρ_i as MLEs
            quotient_mles: steps.quotient_mles, // All Q_i as MLEs
            bits: steps.bits,
        }
    }

    // Packed GT exponentiation witness generation
    fn generate_packed_gt_exp(base: &GT, scalar: &Fr, result: &GT) -> PackedGtExpWitness {
        let steps = ExponentiationSteps::new(base.0, ark_to_jolt(scalar));

        // Pack 128 base-4 steps into 11-var MLEs
        let rho_packed = pack_sequential_mles(&steps.rho_mles);
        let rho_next_packed = pack_shifted_mles(&steps.rho_mles);
        let quotient_packed = pack_sequential_mles(&steps.quotient_mles);
        let digit_lo_packed = pack_digit_lo_bits(&steps.bits);
        let digit_hi_packed = pack_digit_hi_bits(&steps.bits);
        let base_packed = pack_replicated_base(&steps.base, 128, 16);
        let base2_packed = pack_replicated_base(&steps.base2, 128, 16);
        let base3_packed = pack_replicated_base(&steps.base3, 128, 16);

        PackedGtExpWitness {
            base: steps.base,
            exponent: steps.exponent,
            result: steps.result,
            rho_packed,
            rho_next_packed,
            quotient_packed,
            digit_lo_packed,
            digit_hi_packed,
            base_packed,
            base2_packed,
            base3_packed,
        }
    }

    fn generate_gt_mul(lhs: &GT, rhs: &GT, result: &GT) -> JoltGtMulWitness {
        let steps = MultiplicationSteps::new(lhs.0, rhs.0);
        JoltGtMulWitness {
            lhs: steps.lhs,
            rhs: steps.rhs,
            result: steps.result,
            quotient_mle: steps.quotient_mle,
        }
    }

    fn generate_g1_scalar_mul(point: &G1, scalar: &Fr, result: &G1) -> JoltG1ScalarMulWitness {
        let steps = ScalarMultiplicationSteps::new(point.into(), ark_to_jolt(scalar));
        JoltG1ScalarMulWitness {
            point_base: steps.point_base,
            scalar: steps.scalar,
            result: steps.result,
            x_a_mles: steps.x_a_mles,
            y_a_mles: steps.y_a_mles,
            x_t_mles: steps.x_t_mles,
            y_t_mles: steps.y_t_mles,
            x_a_next_mles: steps.x_a_next_mles,
            y_a_next_mles: steps.y_a_next_mles,
            bits: steps.bits,
        }
    }
}
```

#### HintMap: Compact Verification Data

The `HintMap` contains precomputed values that allow verification to skip expensive operations:

```rust
struct HintMap<C: PairingCurve> {
    num_rounds: usize,
    // Precomputed intermediate values for each operation type
    gt_exp_hints: Vec<GtExpHint>,
    gt_mul_hints: Vec<GtMulHint>,
    g1_scalar_mul_hints: Vec<G1ScalarMulHint>,
}
```

#### verify_with_hint: Fast Verification Path

```rust
fn verify_with_hint(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut Transcript,
    point: &[Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
    hint: &HintMap,  // Precomputed hints
) -> Result<(), ProofVerifyError> {
    // Context initialized with hints instead of empty
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_hints(hint.clone())
    );

    // Same verify_recursive call, but:
    // - Uses precomputed hints instead of regenerating witnesses
    // - Skips ExponentiationSteps, MultiplicationSteps, ScalarMultiplicationSteps
    verify_recursive(
        commitment, evaluation, point, proof, setup,
        transcript, ctx,
    )?;

    Ok(())
}
```

#### Integration with RecursionProver

The recursion prover uses `witness_gen` to bootstrap from a Dory proof:

```rust
impl RecursionProver {
    pub fn new_from_dory_proof(
        dory_proof: &ArkDoryProof,
        verifier_setup: &ArkworksVerifierSetup,
        transcript: &mut Transcript,
        point: &[Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<Self, Error> {
        // Extract witnesses via TraceContext
        let (witness_collection, _hints) = DoryCommitmentScheme::witness_gen(
            dory_proof, verifier_setup, transcript,
            point, evaluation, commitment,
        )?;

        // Convert to recursion witness format
        let recursion_witness = Self::witnesses_to_dory_recursion(&witness_collection)?;

        // Build constraint system from extracted witnesses
        let constraint_system = Self::build_constraint_system(&witness_collection)?;

        Ok(Self { constraint_system, ... })
    }
}
```

#### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DORY PROOF VERIFICATION                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │    witness_gen()    │         │  verify_with_hint() │
        │  (Prover/Setup)     │         │   (Lightweight V)   │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   TraceContext::    │         │   TraceContext::    │
        │   for_witness_gen() │         │   for_hints(hint)   │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │  verify_recursive() │         │  verify_recursive() │
        │  + WitnessGenerator │         │  + Hint lookups     │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
         ┌──────────┴──────────┐                   │
         ▼                     ▼                   │
┌─────────────────┐  ┌─────────────────┐          │
│WitnessCollection│  │    HintMap      │──────────┘
│  (full data)    │  │   (compact)     │
└─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RECURSION SNARK (Stages 1-4)                       │
│  Proves that witness_gen produced correct results for all operations    │
└─────────────────────────────────────────────────────────────────────────┘
```

The key insight is that `witness_gen` runs full verification once to extract witnesses, while `verify_with_hint` uses precomputed hints to achieve ~150× speedup by skipping expensive intermediate computations.

### 7.12 GPU Considerations

The recursion prover has a clear separation between orchestration logic (Rust) and compute-intensive kernels (GPU).

#### GPU-Accelerated Components

| Component | Operation | Why GPU |
|-----------|-----------|---------|
| Stage 1 Sumcheck | Packed GT exp evaluations | Parallel evaluation of $2^n$ points |
| Stage 2 Sumcheck | Batched constraints | Parallel evaluation of $2^n$ points |
| Stage 3 Direct Eval | Direct polynomial evaluation | Efficient dot product computation |
| Stage 4 Sumcheck | Jagged transform sumcheck | Dense polynomial operations |
| Stage 5 Sumcheck | Jagged assist sumcheck | Batched MLE verification |
| Hyrax Commit | Multi-scalar multiplication (MSM) | $O(\sqrt{N})$ group operations |
| Hyrax VMP | Vector-matrix product | Parallelizable linear algebra |

#### Rust-Side Orchestration

The following remain on the CPU/Rust side:

- **Witness generation**: `TraceContext` collection during Dory verify
- **Transcript management**: Fiat-Shamir challenge generation
- **Bijection logic**: `VarCountJaggedBijection` index computations
- **Matrix row mapping**: Three-level decode (poly_idx → constraint_idx → matrix_row)
- **Hint serialization**: `HintMap` construction and transmission
- **Proof assembly**: Collecting stage outputs into `RecursionProof`

#### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RUST (CPU)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  witness_gen() ──→ WitnessCollection ──→ ConstraintSystem               │
│                                              │                          │
│  Transcript ←─────────────────────────────── │ ←── challenges           │
│       │                                      │                          │
│       ▼                                      ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         GPU KERNELS                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Stage 1: sumcheck_prove(PackedGtExp) → round_polys            │   │
│  │  Stage 2: sumcheck_prove(shift+reduction+gtmul+g1) → round_polys│   │
│  │  Stage 3: direct_eval(virtual_claims, eq_evals) → evaluation   │   │
│  │  Stage 4: sumcheck_prove(dense, f_jagged) → round_polys        │   │
│  │  Stage 5: sumcheck_prove(jagged_assist) → round_polys          │   │
│  │  Hyrax:   msm(scalars, bases) → commitments                     │   │
│  │           vmp(matrix, vector) → result                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  RecursionProof { stage1, stage2, stage3, stage4, stage5, opening }    │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Kernel Interface

Each sumcheck stage exposes a similar GPU interface:

```rust
trait SumcheckGpuKernel {
    /// Compute all round polynomials for one sumcheck
    fn prove_rounds(
        &self,
        evaluations: &GpuBuffer<F>,      // Polynomial evals on hypercube
        eq_evals: &GpuBuffer<F>,          // eq(r, x) precomputed
        challenges: &mut impl TranscriptReceiver,
    ) -> Vec<RoundPolynomial>;
}
```

Hyrax operations:

```rust
trait HyraxGpuKernel {
    /// Multi-scalar multiplication for row commitments
    fn msm(&self, scalars: &GpuBuffer<F>, bases: &GpuBuffer<G>) -> Vec<G>;

    /// Vector-matrix product for opening
    fn vmp(&self, matrix: &GpuBuffer<F>, vector: &[F]) -> Vec<F>;
}
```

#### Memory Considerations

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| Stage 1 | $O(c \cdot 2^{11})$ | Packed GT exp polynomials |
| Stage 2 | $O(c \cdot 2^{11})$ | Batched constraint polynomials |
| Stage 3 | $O(2^s)$ | Full virtualized matrix |
| Stage 4 | $O(\text{dense\_size})$ | Compressed representation |
| Hyrax | $O(\sqrt{N})$ bases + scalars | MSM working set |

The jagged transform (Stage 4) reduces GPU memory pressure by compressing sparse rows into dense storage.

---

## References

- [Jolt Paper](https://eprint.iacr.org/2023/1217)
- [Dory Paper](https://eprint.iacr.org/2020/1274)
- [Hyrax Paper](https://eprint.iacr.org/2017/1132)
- [Jagged Polynomial Commitments](https://eprint.iacr.org/2024/504) - Claim 3.2.1 for jagged transform
