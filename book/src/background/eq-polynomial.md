# Eq Extension
The $\widetilde{eq}$ MLE is useful throughout the protocol
$$
\widetilde{eq}(r,x) = \begin{cases} 
1 & \text{if } r = x \\
\mathbb{F} & \text{otherwise}
\end{cases}
$$


$$
\widetilde{eq}(r,x) = \prod_{i=1}^{\log(m)} (r_i \cdot x_i + (1 - r_i) \cdot (1 - x_i)) \quad \text{where } r, x \in \{0,1\}^{\log(m)}
$$