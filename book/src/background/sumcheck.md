# The Sum-check Protocol
*Adapted from the [Thaler13](https://eprint.iacr.org/2013/351.pdf) exposition. For a detailed introduction to sumcheck see Chapter 4.1 of [the textbook](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).*

Suppose we are given a $v$-variate polynomial $g$ defined over a finite field $\mathbb{F}$. The purpose of the sumcheck protocol is to compute the sum:

$$ H := \sum_{b_1 \in \{0,1\}} \sum_{b_2 \in \{0,1\}} \cdots \sum_{b_v \in \{0,1\}} g(b_1, \ldots, b_v). $$

In order to execute the protocol, the verifier needs to be able to evaluate $g(r_1, \ldots, r_v)$ for a randomly chosen vector $(r_1, \ldots, r_v) \in \mathbb{F}^v$. Hence, from the verifier's perspective, the sum-check protocol _reduces_ the task 
of summing $g$'s evaluations over $2^v$ inputs (namely, all inputs in $\{0, 1\}^{v}$) to the task of evaluating $g$
at a _single_ input in $(r_1, \ldots, r_v) \in \mathbb{F}^v$.

The protocol proceeds in $v$ rounds as follows. In the first round, the prover sends a polynomial $g_1(X_1)$, and claims that 

$$ g_1(X_1) = \sum_{x_2, \ldots, x_v \in \{0,1\}^{v-1}} g(X_1, x_2, \ldots, x_v). $$

Observe that if $g_1$ is as claimed, then $H = g_1(0) + g_1(1)$. Also observe that the polynomial $g_1(X_1)$ has degree $\text{deg}_1(g)$, the degree of variable $x_1$ in $g$. Hence $g_1$ can be specified with $\text{deg}_1(g) + 1$ field elements. In our implementation, $P$ will specify $g$ by sending the evaluation of $g_1$ at each point in the set $\{0,1, \ldots, \text{deg}_1(g)\}$. (Actually, the prover does _not_ need to send $g_1(1)$, since the verifier can _infer_ that $g_1(1) = H-g_1(0)$, as if this were not the case, the verifier would reject). 

Then, in round $j > 1$, $V$ chooses a value $r_{j-1}$ uniformly at random from $\mathbb{F}$ and sends $r_{j-1}$ to $P$. We will often refer to this step by saying that variable $j - 1$ gets bound to value $r_{j-1}$. In return, the prover sends a polynomial $g_j(X_j)$, and claims that

$$ g_j(X_j) = \sum_{(x_{j+1}, \ldots, x_v) \in \{0,1\}^{v-j}} g(r_1, \ldots, r_{j-1}, X_j, x_{j+1}, \ldots, x_v). \quad (1) $$

The verifier compares the two most recent polynomials by checking that $g_{j-1}(r_{j-1}) = g_j(0) + g_j(1)$, and rejecting otherwise. The verifier also rejects if the degree of $g_j$ is too high: each $g_j$ should have degree $\text{deg}_j(g)$, the degree of variable $x_j$ in $g$.

In the final round, the prover has sent $g_v(X_v)$ which is claimed to be $g(r_1, \ldots, r_{v-1}, X_v)$. $V$ now checks that $g_v(r_v) = g(r_1, \ldots, r_v)$ (recall that we assumed $V$ can evaluate $g$ at this point). If this test succeeds, and so do all previous tests, then the verifier accepts, and is convinced that $H = g_1(0) + g_1(1)$.
