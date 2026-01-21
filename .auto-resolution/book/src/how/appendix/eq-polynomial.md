# Eq Extension

The $\widetilde{\text{eq}}$ multilinear extension (MLE) is useful throughout the protocol. It is the MLE of the
function $\text{eq} \colon \{0, 1\}^m \times \{0, 1\}^m \to \mathbb{F}$ defined as follows:

$$
\text{eq}(r,x) = \begin{cases}
1 & \text{if } r = x \\
0 & \text{otherwise}
\end{cases}
$$

$\widetilde{\text{eq}}$ has the following explicit formula:
$$
\widetilde{\text{eq}}(r,x) = \prod_{i=1}^{\log(m)} (r_i \cdot x_i + (1 - r_i) \cdot (1 - x_i)) \quad \text{where } r, x \in \{0,1\}^{\log(m)}
$$
