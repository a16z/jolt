# Multilinear Extensions 
For any $v$-variate polynomial $g(x_1, ... x_v)$ polynomial, it's multilinear extension $f(x_1, ... x_v)$ is the polynomial which agrees over all $2^v$ points $x \in \{0,1\}^v$. By the schwartz zippel lemma, if $g$ and $f$ disagree at even a single input, then $g$ and $f$ must disagree *almost everywhere*.

For more precise details please read **Section 3.5 of [Proofs and Args ZK](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)**. 

## Engineering
In practice, MLE's are stored as the vector of evaluations over the $v$-variate boolean hypercube $\{0,1\}^v$. There are two important algorithms over multilinear extensions: single variable binding, evaluation.

### Single Variable Binding
With a single streaming pass over all $n$ evaluations we can "bind" a single variable of the $v$-variate multilinear extension to a point $r$. This is a critical sub-algorithm in sumcheck. During the binding the number of evaluation points used to represent the MLE gets reduced by a factor of 2:
- **Before:** $\tilde{f}(x_1, ... x_v): \{0,1\}^v \to \mathbb{F}$
- **After:** $\tilde{f}(x_1, ... x_{v-1}): \{0,1\}^{v-1} \to \mathbb{F}$

Algorithm:
- TODO: sragss

Details on reasoning can be found in **Section 3.5 of [Proofs and Args ZK](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)**.

   - MLE 
	- multilinear polynomial definitions
	- Stored as evaluations
	- Can be "bound to a point" in O(n) time where 'n' is the evaluations.len()
	- Evaluation of a derived MLE (product of individual terms)