# GKR
GKR is a SNARK protocol for small-depth circuits of multiplication and addition gates. The standard form allows combinations of both. The protocol involves three MLEs: $\widetilde{V}_i$ is the MLE of a function capturing the values of the gates at layer $i$, and $\widetilde{\text{add}}_i$ and $\widetilde{\text{mult}}_i$ are MLEs of functions capturing the circuit's wires. 

Specifically, $\widetilde{V}_i(j)$ evaluates to the value of the circuit at the $i$-th layer in the $j$-th gate. For example $\widetilde{V}_1(0)$ corresponds to the output gate.

$\widetilde{\text{add}}_i(j, a, b)$ evaluates to 1 if the $j$-th gate of the $i$-th layer is an addition gate whose inputs are the $a$'th and $b$'th gates at the preceding layer. 

$\widetilde{\text{mult}}_i(j, a, b)$ evaluates to 1 if the $j$-th gate of the $i$-th layer is a multiplication gate whose inputs are the $a$'th and $b$'th gates at the preceding layer. 

The sumcheck protocol is applied to the following:
$$
\widetilde{V}_i(z) = \sum_{(p,\omega_1,\omega_2) \in \{0,1\}^{s_i+2s_{i+1}}} f_{i,z}(p,\omega_1,\omega_2),
$$

where

$$
f_i(z, p, \omega_1, \omega_2) = \widetilde{\text{eq}}_{s_i}(z, p) \cdot ( \widetilde{\text{add}}_i(p, \omega_1, \omega_2)(\widetilde{V}_{i+1}(\omega_1) + \widetilde{V}_{i+1}(\omega_2)) + \widetilde{\text{mult}}_i(p, \omega_1, \omega_2)\widetilde{V}_{i+1}(\omega_1) \cdot \widetilde{V}_{i+1}(\omega_2))
$$
$$
\widetilde{\text{eq}}_{s_i}(z, p) = \prod_{j=1}^{s_i} ((1-z_j)(1-p_j) + z_j p_j).
$$

(See [the eq-polynomial page](https://jolt.a16zcrypto.com/background/eq-polynomial.html) for details.)

Lasso and Jolt currently use the variant of GKR described in [Thaler13](https://eprint.iacr.org/2013/351.pdf), which is optimized for the far simpler case of a binary tree of multiplication gates. This simplifies each sumcheck to:
$$
\widetilde{V}_i(z) = \sum_{p \in \{0,1\}^{s_i}} g^{(i)}_z(p),
$$

where

$$
g^{(i)}_z(p) = \widetilde{\text{eq}}_{s_i}(z, p) \cdot \widetilde{V}_{i+1}(p,0) \cdot \widetilde{V}_{i+1}(p,1)
$$
GKR is utilized in [memory-checking](./memory-checking.md) for the multi-set permutation check.
