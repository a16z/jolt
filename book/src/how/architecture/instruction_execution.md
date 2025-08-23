# Instruction execution

One distinguishing feature of Jolt among zkVM architectures is how it handles instruction execution, i.e. proving the correct input/output behavior of every RISC-V instruction in the trace.
This is primarily achieved through the [Shout](../twist-shout.md#shout) lookup argument.

## Large Lookup Tables and Prefix-Suffix Sumcheck

The Shout instance for instruction execution must query a massive lookup table -- effectively of size $2^{64}$, since the lookup query is constructed from two 32-bit operands.
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

We will split $k$ four times at four different indices, and these will induce the four **phases** of the prefix-suffix sumcheck. Each of the four phases encompasses 16 rounds of sumcheck (i.e. 16 of the address variables $k$), so together they comprise the first $\log K = 64$ rounds of the read-checking sumcheck.

Given our prefix-suffix decomposition of $\widetilde{\textsf{Val}}$, we can rewrite our read-checking sumcheck as follows:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k_\text{prefix} \in \{0, 1\}^{16}, k_\text{suffix} \in \{0, 1\}^{48}, j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) \cdot \sum_{(\textsf{prefix}, \textsf{suffix})} \widetilde{\textsf{prefix}}(k_\text{prefix}) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix})
$$

Note that we have replaced $\prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j)$ with just $\widetilde{\textsf{ra}}$. Since $\prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j)$ is degree 1 in each $k$ variable, we will treat it as a single multilinear polynomial while we're binding those variables (the first $\log K$ rounds).
The equation as written above depicts the first phase, where $k_\text{prefix}$ is the first 16 variables of $k$, and $k_\text{suffix}$ is the last 48 variables of $k$.

Rearranging the terms in the sum, we have:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k_\text{prefix} \in \{0, 1\}^{16}}
\sum_{(\textsf{prefix}, \textsf{suffix})} \widetilde{\textsf{prefix}}(k_\text{prefix}) \cdot \left(\sum_{k_\text{suffix} \in \{0, 1\}^{48}, j \in \{0, 1\}^{\log T}}\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) \cdot \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix}) \right)
$$

Note that the summand is degree 2 in $k_\text{prefix}$, the variables being bound in the first phase:

1. $k_\text{prefix}$ appears in $\widetilde{\textsf{prefix}}(k_\text{prefix})$
2. $k_\text{prefix}$ also appears in appears in the paranthesized expression in $\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j)$

Written in this way, it becomes clear that we can treat the first 16 rounds as a mini-sumcheck over just the 16 $k_\text{prefix}$ variables, and with just multilinear terms.
If we can efficiently compute the $2^{16}$ coefficients of each mulitlinear term, the rest of this mini-sumcheck is efficient. Each evaluation of $\widetilde{\textsf{prefix}}(k_\text{prefix})$ can be computed in constant time, so that leaves the parenthesized term:

$$
\left(\sum_{k_\text{suffix} \in \{0, 1\}^{48}, j \in \{0, 1\}^{\log T}}\widetilde{\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) \cdot \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix}) \right)
$$

This can be computed in $\Theta(T)$: since $\textsf{ra}}$ is one-hot, we can do a single iteration over $j \in \{0, 1\}^{\log T}$ and only compute the terms of the sum where $\textsf{ra}}(k_\text{prefix}, k_\text{suffix}, j) = 1$.
We compute a table of $\widetilde{\textsf{eq}}(r_\text{cycle}, j)$ evaluation a priori, and $\widetilde{\textsf{suffix}}(k_\text{suffix})$ can be evaluated in constant time on Boolean inputs.

After the first phase, the high-order 16 variables of $k$ will have been bound.
We will need to use a new sumcheck expression for the next phase:

$$
\widetilde{\textsf{rv}}(r_\text{cycle}) = \sum_{k_\text{prefix} \in \{0, 1\}^{16}}
\sum_{(\textsf{prefix}, \textsf{suffix})} \widetilde{\textsf{prefix}}(r^{(1)}, k_\text{prefix}) \cdot \left(\sum_{k_\text{suffix} \in \{0, 1\}^{32}, j \in \{0, 1\}^{\log T}}\widetilde{\textsf{ra}}(r^{(1)}, k_\text{prefix}, k_\text{suffix}, j) \cdot \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \widetilde{\textsf{suffix}}(k_\text{suffix}) \right)
$$

Now $r^{(1)} \in \mathbb{F}^{16}$ are random values that the first 16 variables were bound to, and $k_\text{prefix}$ are the _next_ 16 variables of $k$. Meanwhile, $k_\text{suffix}$ now represents the last _32_ variables of $k$.

This complicates things slightly, but the algorithm follows the same blueprint as in phase 1. This is a sumcheck over the 16 $k_\text{prefix}$ variables, and there are two multilinear terms. We can still compute each evaluation of $\widetilde{\textsf{prefix}}(r^{(1)}, k_\text{prefix})$ in constant time, and we can still compute the parenthesized term in $\Theta(T)$ time (observe that there is exactly one non-zero coefficient of $\widetilde{\textsf{ra}}(r^{(1)}, k_\text{prefix}, k_\text{suffix}, j)$ per cycle $j$).


### Prefix and Suffix Implementations

Jolt modularizes prefix/suffix decomposition using two traits:

- `SparseDensePrefix` (under `prefixes/`)
- `SparseDenseSuffix` (under `suffixes/`)

Each prefix/suffix used in a lookup table implements these traits.

### Multiplexing Between Instructions

An execution trace contains many different RISC-V instructions.
Note that there is a many-to-one relationship between instructions and lookup tables -- multiple instructions may share a lookup table (e.g., XOR and XORI).
To manage this, Jolt uses the `InstructionLookupTable` trait, whose `lookup_table` method returns an instruction's associated lookup table, if it has one (some instructions do not require a lookup).

Boolean "lookup table flags" indicate which table is active on a given cycle. At most one flag is set per cycle.
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

🚧 These docs are under construction 🚧

👷If you are urgently interested in this specific page, open a Github issue and we'll try to expedite it.👷

## Other sumchecks

### ra virtualization

The lookup tables used for instruction execution are of size $K = 2^{64}$, so we set $d = 8$ as our decomposition parameter such that $K^{1/d} = 2^8$.

Similar to the [RAM](./ram.md#ra-virtualization) Twist instance, we opt to **virtualize** the $\widetilde{\textsf{ra}}$ polynomial.
In other words, Jolt simply carries out the read checking and $\widetilde{\textsf{raf}}$ evaluation sumchecks *as if* $d = 1$.

At the conclusion of these sumchecks, we are left with claims about the virtual $\widetilde{\textsf{ra}}$ polynomial.
Since the polynomial is uncommitted, Jolt invokes a separate sumcheck that expresses an evaluation $\widetilde{\textsf{ra}}$ in terms of the constituent $\widetilde{\textsf{ra}}_i$ polynomials.
The $\widetilde{\textsf{ra}}_i$ polynomials are, by definition, a tensor decomposition of $\widetilde{\textsf{ra}}$, so the "$\widetilde{\textsf{ra}}$ virtualization" sumcheck is the following:

$$
\widetilde{\textsf{ra}}(r, r') = \sum_{j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r', j) \cdot \left( \prod_{i=1}^d \widetilde{\textsf{ra}}_i(r_i, j) \right)
$$

Since the degree of this sumcheck is $d + 1 = 9$, using the standard linear-time sumcheck prover algorithm would make this sumcheck relatively slow.
However, we employ optimizations specific to high-degree sumchecks, adapting techniques from the [Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm) and [Toom-Cook](https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication) multiplication algorithms.

### One-hot checks

Jolt enforces that the $\widetilde{\textsf{ra}}_i$ polynomials used for instruction execution are [one-hot](../twist-shout.md), using a Booleanity and Hamming weight sumcheck as described in the paper.
These implementations follow the Twist and Shout paper closely, with no notable deviations.
