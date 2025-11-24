# Hyrax: Polynomial Commitments Without Sum-Check

## 1. Core Idea

**Hyrax avoids sum-check entirely** by exploiting the **multiplicative structure** in polynomial evaluation vectors. Instead of running sum-check (which becomes very slow in the Bulletproofs/IPA setting), Hyrax reduces polynomial evaluation to a **vector-matrix-vector product**.

**Key insight**: The evaluation vector $\vec{r} = (1, r, r^2, ..., r^{N-1})$ has multiplicative structure that can be decomposed into two smaller vectors whose outer product reconstructs $\vec{r}$.

## 2. The Problem

**Given**:
- Polynomial $p(X) = \sum_{i=0}^{N-1} c_i X^i$ with coefficient vector $\vec{c} = (c_0, ..., c_{N-1})$
- Pedersen commitment $C = \langle \vec{c}, \vec{G} \rangle$ where $\vec{G} = (G_1, ..., G_N) \in \mathbb{G}^N$
- Evaluation point $r \in \mathbb{F}$

**Goal**: Prove $p(r) = y$ for claimed value $y$

## 3. The Vector-Matrix-Vector Reduction

### 3.1 Standard Polynomial Evaluation

$$p(r) = \langle \vec{c}, \vec{r} \rangle = \sum_{i=0}^{N-1} c_i \cdot r^i$$

where $\vec{r} = (1, r, r^2, ..., r^{N-1})$ is the evaluation vector.

### 3.2 Hyrax's Key Trick

Assume $N = m^2$ (perfect square). Reshape coefficient vector $\vec{c}$ into an $m \times m$ matrix $M$:
$$M_{i,j} = c_{i \cdot m + j}$$

**Example**: For $N = 9$, $m = 3$:
$$\vec{c} = (c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8) \implies M = \begin{pmatrix} c_0 & c_1 & c_2 \\ c_3 & c_4 & c_5 \\ c_6 & c_7 & c_8 \end{pmatrix}$$

Define two vectors that capture the multiplicative structure:
$$\vec{a} = (1, r, r^2, ..., r^{m-1}) \in \mathbb{F}^m$$
$$\vec{b} = (1, r^m, r^{2m}, ..., r^{(m-1)m}) \in \mathbb{F}^m$$

**What are these?**
- $\vec{a}$: The **first $m$ powers** of $r$ (i.e., $r^0, r^1, ..., r^{m-1}$)
- $\vec{b}$: Powers of $r$ in **steps of $m$** (i.e., $r^0, r^m, r^{2m}, ..., r^{(m-1)m}$)

**Why this decomposition?** The outer product $\vec{b} \otimes \vec{a}$ reconstructs the full evaluation vector:
$$(\vec{b} \otimes \vec{a})_{i \cdot m + j} = b_i \cdot a_j = r^{im} \cdot r^j = r^{im+j}$$

Reading the outer product row-by-row gives: $(r^0, r^1, r^2, ..., r^{N-1})$ - exactly the full evaluation vector!

**Key identity**:
$$p(r) = \vec{b}^T \cdot M \cdot \vec{a}$$

### 3.3 Why This Works

The outer product $\vec{b} \otimes \vec{a}$ reconstructs the full evaluation vector:
$$(\vec{b} \otimes \vec{a})_{i \cdot m + j} = b_i \cdot a_j = r^{im} \cdot r^j = r^{im+j}$$

So:
$$p(r) = \langle \vec{c}, \vec{r} \rangle = \langle \text{vec}(M), \vec{b} \otimes \vec{a} \rangle = \vec{b}^T M \vec{a}$$

## 4. The Hyrax Protocol

### 4.1 Commitment Phase

Instead of committing to the full vector $\vec{c}$, the prover commits to **each column** of $M$ separately.

Let $M^{(j)}$ denote column $j$ of $M$. Using commitment key $\vec{G} = (G_1, ..., G_m) \in \mathbb{G}^m$:
$$C_j = \langle M^{(j)}, \vec{G} \rangle = \sum_{i=1}^{m} M_{i,j} \cdot G_i \quad \text{for } j = 1, ..., m$$

**Commitment**: $(C_1, ..., C_m) \in \mathbb{G}^m$ (vector of $m = \sqrt{N}$ group elements)

### 4.2 Evaluation Phase

To prove $p(r) = y$:

**Step 1**: Prover computes the "partial evaluation vector":
$$\vec{v} = M \cdot \vec{a} \in \mathbb{F}^m$$

This is an $m$-length vector where:
$$v_i = \sum_{j=0}^{m-1} M_{i,j} \cdot a_j = \sum_{j=0}^{m-1} M_{i,j} \cdot r^j$$

**Step 2**: Prover sends $\vec{v}$ to verifier (this is the evaluation proof!)

**Step 3**: Verifier checks:

**Check 1** - Verify $\vec{v}$ is consistent with committed matrix $M$:

**The problem**: Prover sent $\vec{v}$, claims it equals $M \vec{a}$. How can verifier check this without knowing $M$?

**The solution**: Use homomorphic properties of Pedersen commitments!

**Step-by-step**:

1. **What the verifier has**:
   - Commitments $C_1, ..., C_m$ to the columns of $M$ (where $C_j = \langle M^{(j)}, \vec{G} \rangle$)
   - The vector $\vec{a} = (1, r, r^2, ..., r^{m-1})$ (verifier can compute this!)
   - The claimed result $\vec{v}$ (sent by prover)

2. **Homomorphic computation**:

   The verifier computes a **commitment to $M \vec{a}$** without knowing $M$:
   $$C_M = \sum_{j=1}^{m} C_j \cdot r^{j-1}$$

   **Why this works**:
   $$C_M = \sum_{j=1}^{m} \langle M^{(j)}, \vec{G} \rangle \cdot r^{j-1} = \langle \sum_{j=1}^{m} M^{(j)} \cdot r^{j-1}, \vec{G} \rangle = \langle M \vec{a}, \vec{G} \rangle$$

   The last equality holds because:
   $$(M \vec{a})_i = \sum_{j=1}^{m} M_{i,j} \cdot a_j = \sum_{j=1}^{m} M_{i,j} \cdot r^{j-1}$$

   So $M \vec{a} = \sum_{j=1}^{m} M^{(j)} \cdot r^{j-1}$ (linear combination of columns weighted by $\vec{a}$).

3. **The check**:

   Verifier compares:
   $$C_M \stackrel{?}{=} \langle \vec{v}, \vec{G} \rangle$$

   - **Left side**: Commitment to $M \vec{a}$ (computed homomorphically from column commitments)
   - **Right side**: Commitment to $\vec{v}$ (computed from prover's claimed vector)

   If they match, then $\vec{v} = M \vec{a}$ (by binding property of Pedersen commitments).

**Key insight**: The verifier can compute a commitment to $M \vec{a}$ by taking a **linear combination** of the column commitments, weighted by the entries of $\vec{a}$. This is the homomorphic property in action!

**Check 2** - Compute final evaluation:

The verifier computes:
$$y' = \vec{b}^T \cdot \vec{v} = \sum_{i=0}^{m-1} b_i \cdot v_i = \sum_{i=0}^{m-1} r^{im} \cdot v_i$$

and accepts if $y' = y$.

## 5. Why This is Faster Than Bulletproofs/IPA

### 5.1 Bulletproofs Problem

Recall from Sumcheck_IPA_Transformation.md: **Bulletproofs/IPA is sum-check executed "in the exponent"**.

**The problem**:
- In standard sum-check, prover does $O(N)$ **field operations** (very fast)
- In Bulletproofs/IPA, every field operation becomes a **scalar multiplication** (group operation)
- **Scalar multiplication cost**: ~400 group additions = ~3000× slower than field multiplication

**Concrete costs** (Section 5.2 of Sumcheck_IPA_Transformation.md):
- **Prover**: $O(N)$ scalar multiplications to compute round messages
- **Verifier**: $O(N)$ scalar multiplications to compute $H^{\tilde{z}(r_1,...,r_n)}$ (needs to evaluate multilinear Lagrange interpolation)

**Why so slow?**: Because the commitment vector $\vec{z}$ is hidden "in the exponent", all operations on it must be done via expensive group operations instead of cheap field arithmetic.

### 5.2 Hyrax Advantage

**Commitment key size**: $\sqrt{N}$ group elements (not $N$!)

**Evaluation proof**:
- **Prover**: Computes $\vec{v} = M \vec{a}$ via matrix-vector multiplication (pure field arithmetic, no cryptography!)
- **Cost**: $O(N)$ field multiplications (not scalar multiplications!)
- **Proof**: Just send $\vec{v}$ (vector of $\sqrt{N}$ field elements)

**Verifier**:
- Two MSMs of size $\sqrt{N}$ (not size $N$!)
- Cost: $O(\sqrt{N})$ scalar multiplications

### 5.3 Comparison Table

| Property | Bulletproofs/IPA | Hyrax |
|----------|------------------|-------|
| **Commitment** | 1 group element | $\sqrt{N}$ group elements |
| **Commitment key** | $N$ group elements | $\sqrt{N}$ group elements |
| **Evaluation proof** | $2 \log N$ group elements | $\sqrt{N}$ field elements |
| **Prover eval time** | $O(N)$ scalar muls | $O(N)$ field muls |
| **Verifier eval time** | $O(N)$ scalar muls | $O(\sqrt{N})$ scalar muls |

**Key tradeoff**: Hyrax has larger commitments and proofs, but much faster prover and verifier for evaluation.

## 6. Concrete Example

**Setup**: $N = 256$ (so $m = 16$), 256-bit field

### Bulletproofs/IPA
- Commitment: 1 group element (32 bytes)
- Commitment key: 256 group elements
- Proof: $2 \log 256 = 16$ group elements (512 bytes)
- Prover: 256 scalar multiplications ≈ 100k group operations
- Verifier: 256 scalar multiplications ≈ 100k group operations

### Hyrax
- Commitment: 16 group elements (512 bytes)
- Commitment key: 16 group elements
- Proof: 16 field elements (512 bytes)
- Prover: 256 field multiplications (pure arithmetic!)
- Verifier: 2 MSMs of size 16 = 32 scalar multiplications ≈ 13k group operations

**Speedup**: Verifier is ~8× faster, prover is ~300× faster!

## 7. Where Hyrax is Used

From the survey: "The first systems to show that SNARKs can be built this way—by applying a polynomial commitment scheme to a sum-check-based interactive proof—were vSQL and Hyrax in 2017."

**vSQL/Hyrax (2017)**: Applied to GKR protocol for circuit evaluation

**Key limitation**: Hyrax works best for structured circuits where the multiplicative structure of evaluation vectors can be exploited.

## 8. The Path to Dory

Hyrax's main limitation: **large commitments** ($\sqrt{N}$ group elements).

**Dory (2021)** solved this by:
1. Using Hyrax's vector-matrix-vector structure for fast evaluation
2. Compressing the Hyrax commitment (vector of $\sqrt{N}$ group elements) into a single group element using AFGHO commitments and pairings

**Result**: Dory achieves:
- Small commitments (1 group element, like Bulletproofs)
- Fast evaluation (like Hyrax)
- Logarithmic verifier (better than both!)

## 9. Key Insight from Survey

From Section 6.4: "Hyrax avoids sum-check entirely and instead leverages 'multiplicative structure' in the polynomial evaluation vector $\vec{r}$. The evaluation proof is non-interactive, and consists simply of the 'partial evaluation vector' $M \cdot \vec{a}$. Its commitments are sublinear in size, but verifier cost is high."

**Why avoid sum-check here?** Because in the Bulletproofs/IPA setting, sum-check forces scalar multiplications (3000× slower than field operations). By working directly with the evaluation vector structure, Hyrax keeps operations in the field as long as possible.

## 10. Summary

**Hyrax = Vector-Matrix-Vector Polynomial Commitment**

- **No sum-check**: Avoids the Bulletproofs/IPA slowdown
- **Multiplicative structure**: Exploits $\vec{r} = \vec{b} \otimes \vec{a}$
- **Fast prover**: Evaluation proof is pure field arithmetic
- **Sublinear verifier**: $O(\sqrt{N})$ scalar multiplications
- **Tradeoff**: Larger commitments ($\sqrt{N}$ vs 1 group element)

**Evolution**:
- Bulletproofs (2018): Small commitments, slow everything
- Hyrax (2017): Larger commitments, fast evaluation
- Dory (2021): Best of both worlds

---

## References

**Primary source**: Justin Thaler, "Sum-check Is All You Need", Section 6.4 "Hyrax: Avoiding Sum-Check by Leveraging Product Structure"

**Original paper**: Wahby et al., "Doubly-efficient zkSNARKs without trusted setup" (S&P 2018)
