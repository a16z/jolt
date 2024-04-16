# GKR
GKR is a SNARK protocol for binary trees of multiplication / addition gates. The standard form allows combinations of both using a wiring predicate $\tilde{V}_i$, and two additional MLEs $\tilde{add}_i$ and $\tilde{mult}_i$. 

$\widetilde{V}_i(j)$ evaluates to the value of he circuit at the $i$-th layer in the $j$-th gate. For example $\tilde{V}_1(0)$ corresponds to the output gate.

$\widetilde{add}_i(j)$ evaluates to 1 if the $j$-th gate of the $i$-th layer is an addition gate.

$\widetilde{mult}_i(j)$ evaluates to 1 if the $j$-th gate of the $i$-th layer is a multiplication gate.

The sumcheck protocol is applied to the following:
$$
\tilde{V}_i(z) = \sum_{(p,\omega_1,\omega_2) \in \{0,1\}^{s_i+2s_{i+1}}} f_{i,z}(p,\omega_1,\omega_2),
$$

where

$$
f_i(z, p, \omega_1, \omega_2) = \beta_{s_i}(z, p) \cdot \tilde{add}_i(p, \omega_1, \omega_2)(\tilde{V}_{i+1}(\omega_1) + \tilde{V}_{i+1}(\omega_2)) + \tilde{mult}_i(p, \omega_1, \omega_2)\tilde{V}_{i+1}(\omega_1) \cdot \tilde{V}_{i+1}(\omega_2)
$$
$$
\beta_{s_i}(z, p) = \prod_{j=1}^{s_i} ((1-z_j)(1-p_j) + z_j p_j).
$$


Lasso and Jolt implement the [Thaler13](https://eprint.iacr.org/2013/351.pdf) version of GKR which is optimized for the far simpler case of a binary tree of multiplication gates. This simplifies each sumcheck to:
$$
\tilde{V}_i(z) = \sum_{p \in \{0,1\}^{s_i}} g^{(i)}_z(p),
$$

where

$$
g^{(i)}_z(p) = \beta_{s_i}(z, p) \cdot \tilde{V}_{i+1}(p,0) \cdot \tilde{V}_{i+1}(p,1)
$$
GKR is utilized in [memory-checking](./memory-checking.html) for the multi-set permutation check.
