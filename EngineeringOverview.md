# Engineering Overview
This document describes the Surge / Lasso proof system from an engineering perspective. Of course the most accurate and up-to-date view can be found in the code itself.

## Reformulation of Lasso: Figure 9
*The following section paraphrases Lasso Figure 9.*

Params
- $N$: Virtual table size
- $C$: Subtable dimensionality
- $m$: Subtable size / Memory size
- $s$: Sparsity / number of non-sparse table indices

$$
\sum_{i \in \{0,1\}^{log(s)}}{\tilde{eq}(i,r) \cdot T[\text{nz}[i]]}
$$
### 1. Prover commits to $2 \alpha$ $log(m)$-variate multilinear polynomials:
- $E_1,...,E_\alpha$
- $ \text{read_counts}_1, ..., \text{read_counts}_\alpha$

And $\alpha$ different $log(N)/C$-variate multilinear polynomials:
- $ \text{final_counts}_1, ..., \text{final_counts}_\alpha$

*$E_i$ is purported to evaluate to each of the $m$ reads into the corresponding $i$-th subtable.*

### 2. Sumcheck $h(k) = \tilde{eq}(r,k) \cdot g(E_1(k), ..., E_\alpha(k))$
Reduces the check $v = \sum_{k \in \{0,1\}^{log(s)}}{g(E_1(k), ..., E_\alpha(k))}
To: $E_i(r_z) = v_{E_i}$ for $i=1,...,\alpha$ given $r_z \in \mathbb{F}^{log(s)}$

### 3. Verifier checks above equality with $v_{E_i}$ provided by sumcheck and $E_i$ provided by an oracle query to the initially committed polynomial.

### 4. Check $E_i = T'_i[dim_i(j)] \forall j \in \{0,1\}^{log(s)}$
- Verifier provides $\tau, \gamma \in \mathbb{F}$
- Prover and verifier run sumcheck protocol for grand products (Tha13) to reduce the check equality between mutliset hashes: $\mathcal{H}_{\tau, \gamma}(WS) = \mathcall{H}_{\tau, \gamma}(RS) \cdot \mathcal{H}_{\tau, \gamma}(S)$
- Sumcheck reduces the check to (for $r''_i \in \mathbb{F}^\ell; r'''_i \in \mathbb{F}^{log(s)}$):
    - $E_i(r'''_i) = v_{E_i}$
    - $dim_i(r'''_i) = v_i$
    - $\text{read_counts}_i(r'''_i) = v_{\text{read_counts}_i}$
    - $\text{final_counts}_i(r''_i) = v_{\text{final_counts}_i}$

### 5. Check that the equations above hold with the RHS provided by sumcheck and the LHS provided by oracle queries to commitments in **Step 1**.

