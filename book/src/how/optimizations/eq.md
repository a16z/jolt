# EQ polynomial optimizations

Many sumcheck instances in Jolt have summands of the form

$$
g(x_1, \dots, x_n) = \widetilde{\textsf{eq}}(w, x) \cdot p(x)
$$

where $w \in \mathbb{F}^n$ is a verifier challenge vector, $\widetilde{\textsf{eq}}$ is the [equality polynomial](../appendix/eq-polynomial.md), and $p(x)$ is a product of witness polynomials. In round $i$, the prover must compute the *round polynomial*
$$
s_i(X) \;=\; \sum_{x' \in \\{0,1\\}^{n-i}} \widetilde{\textsf{eq}}(w, r_1,\dots,r_{i-1}, X, x') \cdot p(r_1,\dots,r_{i-1}, X, x'),
$$
for $X$ evaluated at a small set of points, and send those evaluations.

A naive implementation materializes $\widetilde{\textsf{eq}}(w, \cdot)$ as a dense table of $2^n$ field elements and [binds](../appendix/multilinear-extensions.md#variable-binding) it alongside witness tables each round.

Jolt uses two *prover-side* optimizations that (a) avoid binding a length-$2^n$ eq table each round and (b) avoid ever materializing a full $2^n$-entry eq table at all, while keeping the **standard sumcheck transcript/verifier unchanged**.

## Dao–Thaler: factor linearization + split-eq tables

The [\[Dao, Thaler, 2024\]](https://eprint.iacr.org/2024/1210.pdf) paper describes two (related) optimizations to sumchecks involving the $\widetilde{\textsf{eq}}$ polynomial that are widely applicable throughout Jolt.

### Current-factor linearization

The equality polynomial factors into univariate terms:
$$
\widetilde{\textsf{eq}}(w, x) = \prod_{j=1}^{n} \bigl(w_j x_j + (1-w_j)(1-x_j)\bigr).
$$

After binding $k$ variables to challenges $r_1,\dots,r_k$, the restricted eq polynomial is
$$
\widetilde{\textsf{eq}}(w, (r_1,\dots,r_k,x_{k+1},\dots,x_n))
= \underbrace{\prod_{j=1}^{k} \bigl(w_j r_j + (1-w_j)(1-r_j)\bigr)}_{s\ \text{(accumulated scalar)}} \cdot
\prod_{j=k+1}^{n} \bigl(w_j x_j + (1-w_j)(1-x_j)\bigr).
$$

In round $i=k+1$, the eq contribution for the *current* variable is the known linear function
$$
\ell_i(X) \;=\; s \cdot \bigl((1-w_i)(1-X) + w_i X\bigr).
$$

So we can write
$$
s_i(X) \;=\; \ell_i(X)\cdot q_i(X),
$$
where
$$
q_i(X) \;=\; \sum_{x' \in \\{0,1\\}^{n-i}}
E_{\textsf{rest}}(x') \cdot p(r_1,\dots,r_{i-1}, X, x'),
\quad
E_{\textsf{rest}}(x')=\prod_{j>i} (w_j x_j + (1-w_j)(1-x_j)).
$$

Rather than "binding" a full $2^n$ eq table each round, we maintain the scalar $s$ (updated in $O(1)$ per round) and treat the current eq factor as a known linear multiplier $\ell_i(X)$. This removes the $O(2^n)$ per-round cost of binding a dense eq table.

### Split-eq tables

The remaining factor $E_{\textsf{rest}}(x')$ still ranges over the $(n-i)$ unbound variables. Materializing it as a dense table would reintroduce $2^{n}$-scale memory. Dao–Thaler avoid this with a **two-layer decomposition**: split the remaining variables into two halves and rewrite the sum as an *iterated sum*.

Concretely, choose a split parameter $m \approx (n-i)/2$. Write the remaining suffix variables as
$$
x' = (x_{\textsf{out}}, x_{\textsf{in}}),
\quad
w' = (w_{\textsf{out}}, w_{\textsf{in}}),
$$
so
$$
E_{\textsf{rest}}(x') \;=\; \widetilde{\textsf{eq}}(w_{\textsf{out}}, x_{\textsf{out}})\cdot
\widetilde{\textsf{eq}}(w_{\textsf{in}}, x_{\textsf{in}}).
$$

Instead of a single $2^{n-i}$-entry table, we keep two smaller tables whose total size is $O(2^{(n-i)/2})$:
- $E_{\textsf{out}}[x_{\textsf{out}}] = \widetilde{\textsf{eq}}(w_{\textsf{out}}, x_{\textsf{out}})$
- $E_{\textsf{in}}[x_{\textsf{in}}] = \widetilde{\textsf{eq}}(w_{\textsf{in}}, x_{\textsf{in}})$

Then the inner polynomial evaluation becomes
$$
q_i(X) \;=\; \sum_{x_{\textsf{out}}} E_{\textsf{out}}[x_{\textsf{out}}]\cdot
\Bigl(\sum_{x_{\textsf{in}}} E_{\textsf{in}}[x_{\textsf{in}}]\cdot
p(r_1,\dots,r_{i-1}, X, x_{\textsf{out}}, x_{\textsf{in}})\Bigr).
$$

This structure:
- reduces prover memory by a square-root factor (two tables of size $O(2^{n/2})$ instead of one of size $O(2^n)$)
- reduces the number of field multiplications performed by the prover by a square root factor (same reason)

## Gruen's optimization: recover one evaluation from the previous round's claim

Even after the above, a degree-$d$ round polynomial typically requires $d+1$ evaluations $s_i(X)$ at chosen points.
Jolt uses an optimization introduced in Section 3.2 of [\[Gruen, 2024\]](https://eprint.iacr.org/2024/108.pdf): recover one needed evaluation from the round-sum constraint, so the expensive inner-sum machinery only runs for the remaining points.

Note that this optimization (as implemented in Jolt) **does not change what is sent to the verifier**.

### Degree-3 case

Assume the inner polynomial has degree 2:
$$
q_i(X)=eX^2 + dX + c.
$$

In Jolt we compute enough to get:
- $q_i(0)=c$ (constant term), and
- the leading coefficient $e$,
and we avoid directly computing $q_i(1)$ via another full inner sum.

Because the sumcheck relation gives the verifier-checked identity
$$
s_i(0)+s_i(1) \;=\; \textsf{claim}_i,
$$
and $s_i(X)=\ell_i(X)q_i(X)$, we can solve for the missing evaluation:
$$
q_i(1)
\;=\;
\frac{\textsf{claim}_i - \ell_i(0)\,q_i(0)}{\ell_i(1)}.
$$
(We only need $\ell_i(1)\neq 0$, which holds with overwhelming probability for random challenges.)

Once $q_i(0)$, $q_i(1)$, and the leading coefficient $e$ are known, we can evaluate $q_i(2)$ and $q_i(3)$ cheaply (no additional inner-hypercube sums), and then compute
$$
s_i(t)=\ell_i(t)\,q_i(t)\quad \text{for }t\in \{ 0,1,2,3 \},
$$
and interpolate the cubic $s_i(X)$ from its four evaluations.

### Degree-2 case

Similarly, when $s_i(X)$ is quadratic (so $q_i$ is linear), we can compute $q_i(0)$, recover $q_i(1)$ from $\textsf{claim}_i=s_i(0)+s_i(1)$, and then evaluate at the remaining interpolation points.

## Implemented in Jolt: `GruenSplitEqPolynomial`

Jolt's implementation combines:
1. **Dao–Thaler current-factor linearization**: maintain `current_scalar = s` and multiply by the current linear term $\ell_i(X)$ rather than binding an eq table each round.
2. **Split-eq iterated-sum tables** (Dao–Thaler decomposition): represent the remaining eq suffix as two half-tables and compute inner sums as a two-layer loop (outer parallel / inner sequential).
3. **No-verifier-change reuse** (Gruen-style honest-prover trick): recover one evaluation per round from the protocol's round-sum constraint, reducing the number of expensive inner evaluations needed.

This combination keeps the verifier logic and transcript format identical to standard sumcheck, while substantially reducing prover field multiplications and avoiding $2^n$-scale memory.

## Cost intuition

- Eliminating dense eq-table binding removes the dominant $O(2^n)$ "bind eq" cost that would otherwise occur in every round.
- Split-eq reduces eq-related memory from $O(2^{n-i})$ to $O(2^{(n-i)/2})$ during the expensive early rounds.
- Reusing the round-sum constraint to recover one evaluation reduces the number of full inner-sum evaluations per round (the same "send one fewer point / derive from the check" theme appears in Gruen's discussion).
