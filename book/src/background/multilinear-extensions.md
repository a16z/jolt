# Multilinear Extensions 
For any $v$-variate polynomial $g(x_1, ... x_v)$ polynomial, it's multilinear extension $f(x_1, ... x_v)$ is the polynomial which agrees over all $2^v$ points $x \in \{0,1\}^v$: $g(x_1, ... x_v) = \tilde{f}(x_1, ... x_v) \forall x \in \{0,1\}^v$. By the Schwartz-Zippel lemma, if $g$ and $f$ disagree at even a single input, then $g$ and $f$ must disagree *almost everywhere*.

For more precise details please read **Section 3.5 of [Proofs and Args ZK](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)**. 

## Engineering
In practice, MLE's are stored as the vector of evaluations over the $v$-variate boolean hypercube $\{0,1\}^v$. There are two important algorithms over multilinear extensions: single variable binding, and evaluation.

### Single Variable Binding
With a single streaming pass over all $n$ evaluations we can "bind" a single variable of the $v$-variate multilinear extension to a point $r$. This is a critical sub-algorithm in sumcheck. During the binding the number of evaluation points used to represent the MLE gets reduced by a factor of 2:
- **Before:** $\tilde{f}(x_1, ... x_v): \{0,1\}^v \to \mathbb{F}$
- **After:** $\tilde{f'}(x_1, ... x_{v-1})=\tilde{f}(r, x_1, ... x_{v-1}): \{0,1\}^{v-1} \to \mathbb{F}$

Assuming your MLE is represented as a 1-D vector of $2^v$ evaluations $E$ over the $v$-variate boolean hypercube $\{0,1\}^v$, indexed little-endian
- $E[1] = \tilde{f}(0,0,0,1)$
- $E[5] = \tilde{f}(0,1,0,1)$
- $E[8] = \tilde{f}(1,0,0,0)$

Then we can transform the vector $E \to E'$ representing the transformation from $\tilde{f}(x_1, ... x_v) \to \tilde{f'}(r, x_1, ... x_{v-1})$ by "binding" the evaluations vector to a point $r$.

```python
let n = 2^v;
let half = n / 2;
for i in 0..half {
    let low = E[i];
    let high = E[half + i];
    E[i] = low + r * (high - low);
}
```

### Multi Variable Binding
Another common algorithm is to take the MLE $\tilde{f}(x_1, ... x_v)$ and compute its evaluation at a single $v$-variate point outside the boolean hypercube $x \in \mathbb{F}^v$. This algorithm can be performed in $O(n)$ time by performing the single variable binding algorithm $\log(n)$ times. The time spent on $i$'th variable binding is $O(n/2^i)$, so the total time across all $\log n$ bindings is proportional to $\sum_{i=1}^{\log n} n/2^i = O(n)$. 
