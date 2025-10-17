# Instruction execution

One distinguishing feature of Jolt among zkVM architectures is how it handles instruction execution, i.e. proving the correct input/output behavior of every RISC-V instruction in the trace.
This is primarily achieved through the [Shout](../twist-shout.md#shout) lookup argument.

## Large Lookup Tables and Prefix-Suffix Sumcheck

The Shout instance for instruction execution must query a massive lookup table -- effectively of size $2^{128}$, since the lookup query is constructed from two 64-bit operands.
This $K >> T$ parameter regime is discussed in the Twist and Shout paper (Section 7), which proposes the use of sparse-dense sumcheck.
However, upon implementation it became clear that sparse-dense sumcheck did not generalize well to the full RISC-V instruction set.

Instead, Jolt introduces a new algorithm: prefix-suffix sumcheck, described in the appendix of [Proving CPU Executions in Small Space](https://eprint.iacr.org/2025/611).
Like the sparse-dense algorithm, it requires some structure in the lookup table:

- The lookup table must have an MLE that is efficiently evaluable by the verifier. The `JoltLookupTable` trait encapsulates this MLE.
- The lookup index can be split into a prefix and suffix, such that MLEs can be evaluated independently on the two parts and then recombined to obtain the desired lookup entry.
- Every prefix/suffix MLE is efficiently evaluable (constant time) on Boolean inputs

The prefix-suffix sumcheck algorithm can be conceptualized as a careful application of the distributive law to reduce the number of multiplications required for a sumcheck that would otherwise be intractably large.

To unpack this, consider the read-checking sumcheck for Shout, as presented in the paper.

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k = (k_1, \dots, k_d) \in \left(\{0, 1\}^{\log(K) / d}\right)^d, j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \left( \prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j) \right) \cdot \widetilde{\textsf{Val}}(k)
$$

Naively, this would require $\Theta(dKT)$ multiplications, which is far too large given $d = 8$ and $K = 2^{64}$. But suppose $\widetilde{\textsf{Val}}$ has **prefix-suffix structure**. The key intuition of prefix-suffix structure is captured by the following equation:

$$
\widetilde{\textsf{Val}}(k_\text{prefix}, k_\text{suffix}) = \sum_{(\textsf{prefix}, \textsf{suffix}) \in \text{decompose}(\textsf{Val})} \widetilde{\textsf{prefix}}(k_\text{prefix}) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix})
$$

You can think of $k_\text{prefix}$ and $k_\text{suffix}$ as the high-order and low-order "bits" of $k$, respectively, obtained by splitting $k$ at some partition index. The `PrefixSuffixDecomposition` trait specifies which prefix/suffix MLEs to evaluate and how to combine them.

We will split $k$ eight times at eight different indices, and these will induce the eight **phases** of the prefix-suffix sumcheck. Each of the eight phases encompasses 16 rounds of sumcheck (i.e. 16 of the address variables $k$), so together they comprise the first $\log K = 128$ rounds of the read-checking sumcheck.

Given our prefix-suffix decomposition of $\widetilde{\textsf{Val}}$, we can rewrite our read-checking sumcheck as follows:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k_\text{prefix} \in \{0, 1\}^{16}, k_\text{suffix} \in \{0, 1\}^{112}, j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) \cdot \sum_{(\textsf{prefix}, \textsf{suffix})} \widetilde{\textsf{prefix}}(k_\text{prefix}) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix})
$$

Note that we have replaced $\prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j)$ with just $\widetilde{\textsf{ra}}$. Since $\prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j)$ is degree 1 in each $k$ variable, we will treat it as a single multilinear polynomial while we're binding those variables (the first $\log K$ rounds).
The equation as written above depicts the first phase, where $k_\text{prefix}$ is the first 16 variables of $k$, and $k_\text{suffix}$ is the last 112 variables of $k$.

Rearranging the terms in the sum, we have:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k_\text{prefix} \in \{0, 1\}^{16}}
\sum_{(\textsf{prefix}, \textsf{suffix})} \widetilde{\textsf{prefix}}(k_\text{prefix}) \cdot \left(\sum_{k_\text{suffix} \in \{0, 1\}^{112}, j \in \{0, 1\}^{\log T}}\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) \cdot \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix}) \right)
$$

Note that the summand is degree 2 in $k_\text{prefix}$, the variables being bound in the first phase:

1. $k_\text{prefix}$ appears in $\widetilde{\textsf{prefix}}(k_\text{prefix})$
2. $k_\text{prefix}$ also appears in appears in the paranthesized expression in $\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j)$

Written in this way, it becomes clear that we can treat the first 16 rounds as a mini-sumcheck over just the 16 $k_\text{prefix}$ variables, and with just multilinear terms.
If we can efficiently compute the $2^{16}$ coefficients of each mulitlinear term, the rest of this mini-sumcheck is efficient. Each evaluation of $\widetilde{\textsf{prefix}}(k_\text{prefix})$ can be computed in constant time, so that leaves the parenthesized term:

$$
\left(\sum_{k_\text{suffix} \in \{0, 1\}^{112}, j \in \{0, 1\}^{\log T}}\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) \cdot \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix}) \right)
$$

This can be computed in $\Theta(T)$: since $\widetilde{\textsf{ra}}$ is one-hot, we can do a single iteration over $j \in \{0, 1\}^{\log T}$ and only compute the terms of the sum where $\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) = 1$.
We compute a table of $\widetilde{\textsf{eq}}(r_\text{cycle}, j)$ evaluation a priori, and $\widetilde{\textsf{suffix}}(k_\text{suffix})$ can be evaluated in constant time on Boolean inputs.

After the first phase, the high-order 16 variables of $k$ will have been bound.
We will need to use a new sumcheck expression for the next phase:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k_\text{prefix} \in \{0, 1\}^{16}}
\sum_{(\textsf{prefix}, \textsf{suffix})} \widetilde{\textsf{prefix}}(r^{(1)}, k_\text{prefix}) \cdot \left(\sum_{k_\text{suffix} \in \{0, 1\}^{96}, j \in \{0, 1\}^{\log T}}\widetilde{\textsf{ra}}(r^{(1)}, k_\text{prefix}, k_\text{suffix}, j) \cdot \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix}) \right)
$$

Now $r^{(1)} \in \mathbb{F}^{16}$ are random values that the first 16 variables were bound to, and $k_\text{prefix}$ are the _next_ 16 variables of $k$. Meanwhile, $k_\text{suffix}$ now represents the last _96_ variables of $k$.

This complicates things slightly, but the algorithm follows the same blueprint as in phase 1. This is a sumcheck over the 16 $k_\text{prefix}$ variables, and there are two multilinear terms. We can still compute each evaluation of $\widetilde{\textsf{prefix}}(r^{(1)}, k_\text{prefix})$ in constant time, and we can still compute the parenthesized term in $\Theta(T)$ time (observe that there is exactly one non-zero coefficient of $\widetilde{\textsf{ra}}(r^{(1)}, k_\text{prefix}, k_\text{suffix}, j)$ per cycle $j$).

After the first $\log K$ rounds of sumcheck, we are left with:

$$
\sum_{j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{ra}}(r_\text{address}, j) \cdot \widetilde{\textsf{Val}}(r_\text{address})
$$

which we prove using the standard linear-time sumcheck algorithm. Note that $\widetilde{\textsf{ra}}(r_\text{address}, j)$ here is a [virtual](#ra-virtualization) polynomial.


### Prefix and Suffix Implementations

Jolt modularizes prefix/suffix decomposition using two traits:

- `SparseDensePrefix` (under `prefixes/`)
- `SparseDenseSuffix` (under `suffixes/`)

Each prefix/suffix used in a lookup table implements these traits.

### Multiplexing Between Instructions

An execution trace contains many different RISC-V instructions.
Note that there is a many-to-one relationship between instructions and lookup tables -- multiple instructions may share a lookup table (e.g., XOR and XORI).
To manage this, Jolt uses the `InstructionLookupTable` trait, whose `lookup_table` method returns an instruction's associated lookup table, if it has one (some instructions do not require a lookup).

Boolean **lookup table flags** indicate which table is active on a given cycle. At most one flag is set per cycle.
These flags allow us to "multiplex" between all of the lookup tables:

$$
\sum_{\ell} \widetilde{\textsf{flag}}_{\ell}(j) \cdot \widetilde{\textsf{Val}}_\ell(k)
$$

where the sum is over all lookup tables $\ell$.

Only one table's flag $\textsf{flag}}_{\ell}(j)$ is 1 at any given cycle $j$, so only that table's $\widetilde{\textsf{Val}}_\ell(k)$ contributes to the sum.

This term becomes a drop-in replacement for $\widetilde{\textsf{Val}}$ as it appears in the Shout read-checking sumcheck:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k = (k_1, \dots, k_d) \in \left(\{0, 1\}^{\log(K) / d}\right)^d, j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \left( \prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j) \right) \cdot \left( \sum_{\ell} \widetilde{\textsf{flag}}_{\ell}(j) \cdot \widetilde{\textsf{Val}}_\ell(k) \right)
$$

Note that each $\widetilde{\textsf{Val}}_\ell$ here has prefix-suffix structure.

### raf Evaluation

The $\widetilde{\textsf{raf}}$-evaluation sumcheck in Jolt deviates from the description in the Twist/Shout paper.
This is mostly an artifact of the prefix-suffix sumcheck, which imposes some required structure on the lookup index (i.e. the "address" variables $k$).

#### Case 1: Interleaved operands

Consider, for example, the lookup table for `XOR x y`.
Intuitively, the lookup index must be crafted from the bits of `x` and `y`.
A first attempt might be to simply concatenate the bits of `x` and `y`, i.e.:

$$
(k_1, k_2, \dots, k_{128}) = (x_1, x_2, \dots, x_{64}, y_1, y_2, \dots, y_{64})
$$

Unfortunately, there is no apparent way for this formulation to satisfy prefix-suffix structure.
Instead we will *interleave* the bits of `x` and `y`, i.e.

$$
(k_1, k_2, \dots, k_{128}) = (x_1, y_1, x_2, y_2, \dots, x_{64}, y_{64})
$$

With this formulation, the prefix-suffix structure is easily apparent. Suppose the prefix-suffix split index is 16, so:

$$
k_\text{prefix} = (x_1, y_1, x_2, y_2, \dots, x_{8}, y_{8}) \\
k_\text{suffix} = (x_{9}, y_{9}, x_{10}, y_{10}, \dots, x_{64}, y_{64})
$$

Then `XOR x y` has the following prefix-suffix decomposition:

$$
\widetilde{\textsf{Val}}_{\texttt{XOR}}(k_\text{prefix}, k_\text{suffix}) = \widetilde{\textsf{prefix}}_{\texttt{XOR}}(k_\text{prefix}) + \widetilde{\textsf{suffix}}_{\texttt{XOR}}(k_\text{suffix}) \\
\widetilde{\textsf{prefix}}_{\texttt{XOR}}(k_\text{prefix}) = 2^{63} \cdot \left( x_1 + y_1 - 2x_1 y_1 \right) + 2^{62} \cdot \left( x_2 + y_2 - 2x_2 y_2 \right) + \dots + 2^{56} \cdot \left( x_{8} + y_{8} - 2x_{8} y_{8} \right) \\
\widetilde{\textsf{suffix}}_{\texttt{XOR}}(k_\text{suffix}) = 2^{55} \cdot \left( x_{9} + y_{9} - 2x_{9} y_{9} \right) + 2^{54} \cdot \left( x_{10} + y_{10} - 2x_{10} y_{10} \right) + \dots + 2^0 \cdot \left( x_{64} + y_{64} - 2x_{64} y_{64} \right)
$$

By inspection, $\widetilde{\textsf{prefix}}_{\texttt{XOR}}(k_\text{prefix})$ effectively computes the 8-bit `XOR` of the high-order bits of `x` and `y`, while $\widetilde{\textsf{suffix}}_{\texttt{XOR}}$ computes the 56-bit `XOR` of the low-order bits of `x` and `y`.
Then the full result `XOR x y` is obtained by concatenating (adding) the two results.

Now that we've confirmed that we have something with prefix-suffix structure, we can write down the $\widetilde{\textsf{raf}}$-evaluation sumcheck expression.
Instead of a single $\widetilde{\textsf{raf}}$ polynomial, here we have two **lookup operands** `x` and `y`, which are called `LeftLookupOperand` and `RightLookupOperand` in code.
These are the values that appear in Jolt's R1CS constraints.
The point of the $\widetilde{\textsf{raf}}$-evaluation sumcheck is to relate these (non-one-hot) polynomials to their one-hot counterparts.
In the context of instruction execution Shout, this means the $\widetilde{\textsf{ra}}$ polynomial.

Since we have two $\widetilde{\textsf{raf}}$-like polynomials, we have two sumcheck instances:

$$
\widetilde{\textsf{LeftLookupOperand}}(r) = \sum_{k,j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \sum_{\ell = 0}^{\log (K) / 2 - 1} 2^{\ell} \cdot k_{2 \ell} \\
\widetilde{\textsf{RightLookupOperand}}(r) = \sum_{k,j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \sum_{\ell = 0}^{\log (K) / 2 - 1} 2^{\ell} \cdot k_{2 \ell + 1}
$$

This captures the interleaving behavior we described above: `LeftLookupOperand` is the concatenation of the "odd bits" of $k$, while `RightLookupOperand` is the concatenation of the "even bits" of $k$.

#### Case 2: Single operand

Many RISC-V instructions are similar to `XOR`, in that interleaving the operand bits lends itself to prefix-suffix structure.
However, there are some instructions where this is _not_ the case.
The execution of some [arithmetic](./r1cs_constraints.md) instructions, for example, are handled by computing the corresponding operation in the field, and applying a range-check lookup to the result to truncate potential overflow bits.

In this case, the lookup index corresponds to the (single) value `y` being range-checked, so we do not interleave its bits with another operand.
Instead, we just have:

$$
(k_1, k_2, \dots, k_{128}) = (0, 0, \dots, 0, y_1 y_2, \dots, y_{64})
$$

In this case, we have the following sumchecks:

$$
\widetilde{\textsf{LeftLookupOperand}}(r) = 0 \\
\widetilde{\textsf{RightLookupOperand}}(r) = \sum_{k,j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \sum_{\ell = 0}^{\log (K) - 1} 2^{\ell} \cdot k_{\ell}
$$

`LeftLookupOperand` is 0 and `RightLookupOperand` will be the concatenation of _all_ the bits of $k$.

In order to handle Cases 1 and 2 simultaneously, we can use the same "[multiplexing](#multiplexing-between-instructions)" technique as in the read-checking sumcheck.
We use a flag polynomial to indicate which case we're in:

$$
\widetilde{\textsf{InterleaveOperands}}(j) = 1 - \widetilde{\textsf{AddOperands}}(j) - \widetilde{\textsf{SubtractOperands}}(j) - \widetilde{\textsf{MultiplyOperands}}(j)
$$

where $\widetilde{\textsf{AddOperands}}$, $\widetilde{\textsf{SubtractOperands}}$, and $\widetilde{\textsf{MultiplyOperands}}$ are [circuit flags](./r1cs_constraints.md#circuit-flags).

#### Prefix-suffix structure

These sumchecks have similar structure to the read-checking sumcheck.
A side-by-side comparison of the read-checking sumcheck with the sumchecks for Case 1:

$$
\widetilde{\textsf{rv}}(r) = \sum_{k, j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \widetilde{\textsf{Val}}(k) \\
\widetilde{\textsf{LeftLookupOperand}}(r) = \sum_{k,j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \sum_{\ell = 0}^{\log (K) / 2 - 1} 2^{\ell} \cdot k_{2 \ell} \\
\widetilde{\textsf{RightLookupOperand}}(r) = \sum_{k,j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \sum_{\ell = 0}^{\log (K) / 2 - 1} 2^{\ell} \cdot k_{2 \ell + 1}
$$

As it turns out, $\sum_{\ell = 0}^{\log (K) / 2 - 1} 2^{\ell} \cdot k_{2 \ell}$ and $\sum_{\ell = 0}^{\log (K) / 2 - 1} 2^{\ell} \cdot k_{2 \ell + 1}$ have prefix-suffix structure, so we can also use the prefix-suffix algorithm for these sumchecks.
Due to their similarities, we batch the read-checking and these $\widetilde{\textsf{raf}}$-evaluation sumchecks together in a [bespoke](../optimizations/batched-sumcheck.md#bespoke-batching) fashion.

## Other sumchecks

### ra virtualization

The lookup tables used for instruction execution are of size $K = 2^{128}$, so we set $d = 16$ as our decomposition parameter such that $K^{1/d} = 2^8$.

Similar to the [RAM](./ram.md#ra-virtualization) Twist instance, we opt to **virtualize** the $\widetilde{\textsf{ra}}$ polynomial.
In other words, Jolt simply carries out the read checking and $\widetilde{\textsf{raf}}$ evaluation sumchecks *as if* $d = 1$.

At the conclusion of these sumchecks, we are left with claims about the virtual $\widetilde{\textsf{ra}}$ polynomial.
Since the polynomial is uncommitted, Jolt invokes a separate sumcheck that expresses an evaluation $\widetilde{\textsf{ra}}$ in terms of the constituent $\widetilde{\textsf{ra}}_i$ polynomials.
The $\widetilde{\textsf{ra}}_i$ polynomials are, by definition, a tensor decomposition of $\widetilde{\textsf{ra}}$, so the "$\widetilde{\textsf{ra}}$ virtualization" sumcheck is the following:

$$
\widetilde{\textsf{ra}}(r, r') = \sum_{j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r', j) \cdot \left( \prod_{i=1}^d \widetilde{\textsf{ra}}_i(r_i, j) \right)
$$

Since the degree of this sumcheck is $d + 1 = 17$, using the standard linear-time sumcheck prover algorithm would make this sumcheck relatively slow.
However, we employ optimizations specific to high-degree sumchecks, adapting techniques from the [Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm) and [Toom-Cook](https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication) multiplication algorithms.

### One-hot checks

Jolt enforces that the $\widetilde{\textsf{ra}}_i$ polynomials used for instruction execution are [one-hot](../twist-shout.md#one-hot-polynomials), using a Booleanity and Hamming weight sumcheck as described in the paper.
These implementations follow the Twist and Shout paper closely, with no notable deviations.
