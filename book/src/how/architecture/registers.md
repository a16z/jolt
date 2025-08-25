# Registers

Jolt proves the correctness of register updates using the [Twist](../twist-shout.md) memory checking algorithm, specifically utilizing the "local" prover algorithm.

In this Twist instance, $K = 64$ because we have 32 RISC-V registers and 32 [virtual](./emulation.md#virtual-instructions-and-sequences) registers.
This is small enough that we can use $d = 1$.

## Deviations from the Twist algorithm as described in the paper

Our implementation of the Twist prover algorithm differs from the description given in the Twist and Shout [paper](https://eprint.iacr.org/2025/105) in a couple of ways. One such deviation is [wv virtualization](../twist-shout.md#wv-virtualization). Other, register-specific deviations are described below.

### Two reads, one write per cycle

The Twist algorithm as described in the paper assumes one read and one write per cycle, with corresponding polynomials $\widetilde{\textsf{ra}}$ (read address) and $\widetilde{\textsf{wa}}$ (write address).
However, in the context of the RV32IM instruction set, a single instruction (specifically, an R-type instruction) can read from two source registers (`rs1` and `rs2`) and write to a destination register (`rd`).

Thus, we have *two* $\widetilde{\textsf{ra}}$ polynomials corresponding to `rs1` and `rs2`, plus a $\widetilde{\textsf{wa}}$) polynomial corresponding to `rd`.
Similarly, there are two $\widetilde{\textsf{rv}}$ polynomials and one $\widetilde{\textsf{wv}}$ polynomial.

As a result, we have two read-checking sumcheck instances and one write-checking sumcheck instance.
In practice, all three are [batched](../optimizations/batched-sumcheck.md#bespoke-batching) into a single sumcheck instance.

### Why we don't need one-hot checks

Normally in Twist and Shout, polynomials like $\widetilde{\textsf{ra}}$ and $\widetilde{\textsf{wa}}$ need to be checked for one-hotness using the Booleanity and Hamming weight sumchecks.

In the context of registers, we do *not* need to perform these one-hot checks for $\widetilde{\textsf{ra}}_\texttt{rs1}$, $\widetilde{\textsf{ra}}_\texttt{rs2}$, and $\widetilde{\textsf{wa}}_\texttt{rd}$.

Observe that the registers are hardcoded in the program bytecode. Thus we can virtualize $\widetilde{\textsf{ra}}_\texttt{rs1}$, $\widetilde{\textsf{ra}}_\texttt{rs2}$, and $\widetilde{\textsf{wa}}_\texttt{rd}$ using the [bytecode](./bytecode.md) read-checking sumcheck.
Since the bytecode is known by the verifier and assumed to be "well-formed", the soundness of the read-checking sumcheck effectively ensures the one-hotness of $\widetilde{\textsf{ra}}_\texttt{rs1}$, $\widetilde{\textsf{ra}}_\texttt{rs2}$, and $\widetilde{\textsf{wa}}_\texttt{rd}$.

A (simplified) sumcheck expression to illustrate how this virtualization works:

$$
\widetilde{\textsf{ra}}_\texttt{rs1}(r_\text{register}, r_\text{cycle}) = \sum_{k = (k_1, \dots, k_d) \in \left(\{0, 1\}^{\log(K) / d}\right)^d, j \in \{0, 1\}^{\log(T)}} \widetilde{\textsf{eq}}(r_\text{cycle}, j) \cdot \left( \prod_{i=1}^d \widetilde{\textsf{ra}}_i(k_i, j) \right) \cdot \widetilde{\textsf{rd}}(k, r_\text{register})
$$

where $\widetilde{\textsf{rd}}(k_\text{bytecode}, k_\text{register}) = 1$ if the instruction at index $k_\text{bytecode}$ in the bytecode has $\texttt{rs1} = k_\text{register}$.
