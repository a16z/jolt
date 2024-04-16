
## Details on using Lasso to handle instruction execution

Lasso requires that each primitive instruction satisfies a decomposability property. 
The property needed is that the input(s) to the instruction can be broken into "chunks" (say, with each chunk
consisting of 16 bits), such that one can obtain the answer to the original instruction by
evaluating a simple function or functions on each chunk and then "collating" the results together.
For example, the bitwise-OR of two 32-bit inputs x and y can be computed by breaking each input up into 8-bit chunks, XORing 
each 8-bit chunk of x with the associated chunk of y, and concatenating the results together.

In Lasso, we call the task of evaluating a simple function on each chunk "subtable lookups" (the relevant lookup table
being the table storing all $2^{16}$ evaluations of the simple function). And the "collating" of 
the results of the subtable lookups into the result of the original lookup (instruction execution on the un-chunked inputs)
is handled via an invocation of the sum-check protocol. We call this the "primary sumcheck" instance in Lasso.

The "primary sumcheck" collation looks as follows for a trace of length $m$ and a VM with $f$ unique instructions.

$\sum_{x \in `\{0,1\}`^{log(m)}} [\widetilde{eq}(r,x) \cdot \sum_{f \in `\{0,1\}`^{log(F)}} {\widetilde{flags_f}(x) \cdot g_f(\text{terms}_f(x))]}$

$\widetilde{flags_f}(x) = 1$ if the $f$-th instruction is used during the $x$-th step of the trace when. $x \in `\{0,1\}`^{log(m)}$ 

$g_f(...)$ is the collation function used by the $f$-th instruction.

$terms_f(x) = [E_1(x), ... E_\alpha(x)]$ where $\alpha$ is the number of independent memories used by an instruction. For simple instructions like the EQ instruction, $\alpha = C$, $terms_f(x) = [E_1(x), ... E_C(x)]$. More complicated instructions such LT might have $terms_f(x) = [E_{eq}(x), E_{lt1}(x), E_{lt2}(x)]$. The exact layout is dependent on the number of subtables required by the decomposition of the instruction. The mappings can be found in the `JoltInstruction::subtable` method implementations.

### Mental Model
For a given $r = x \in `\{0,1\}`^{log(m)}$ (think integer index of the instruction within the trace), $\widetilde{\text{eq}} = 0$ for all but one term of the outer sum. Similarly all $\widetilde{\text{flags}_f}(x) = 0$ for all but one term of the inner sum. Leaving just the collation function of a single instruction, evaluating to the collated lookup output of the single instruction. In reality $r$ is a random point $r \in \mathbb{F}^{log(m)}$ selected by the verifier over the course of the protocol. The evaluation point provides a distance amplified encoding of the entire trace of instructions.


To illustrate more concretely imagine a two-instruction VM for LT and EQ instructions with $C=1$.

$$
\sum_{x \in \{0, 1\}^{\log_2(m)}}{\widetilde{\text{eq}}(r,x) \cdot \left[ \widetilde{\text{flags}}_{\texttt{LT}}(x) \cdot g_{\texttt{LT}}(E_{\texttt{LT}}(x)) + \widetilde{\text{flags}}_{\texttt{EQ}}(x) \cdot g_{\texttt{EQ}}(E_{\texttt{EQ}}(x)) \right]}
$$


## Subtable Flags
`TODO`
- We then use memory checking to determine that each of the memories $E_i$ is well formed
- At a given step of the CPU only a single instruction will be used, that means that only that instruction's subtables will be used. For the rest of the memories we insert a no_op with (a, v) = 0. In order to make the GKR trees cheaper to compute and sumcheck we'll add a single additional layer to the GKR tree. During this layer we'll "toggle" each of the GKR leaves to "1" in the case that it is an unused step of the CPU. This will make the binary tree of multiplication gates cheaper. We'll toggle based on a new flags polynomial called $subtable-flags_f$ which is the sum of all of the $instruction-flags_f$ used in the instruction collation above.
- The function to compute each of the leaves becomes $leaf[i] = \text{subtable-flags}[i] \cdot \text{fingerprint}[i] + (1 - \text{subtable-flags}[i])$


# Lookups
Lasso is a lookup argument (equivalent to a SNARK for reads into a read-only memory). Lookup arguments allow the prover to convince the verifier that a (committed) set of values $Q$ is a subset of a lookup table $T$. Lasso is a special lookup argument with highly desirable asymptotic costs largely correlated to the number of queries $Q$ rather than the length of of the table $T$.

A conversational background on lookups can be found [here](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/). In short: Lookups are great for zkVMs as they allow constant cost / developer complexity for the prover algorithm per VM instruction.

## Lasso
A detailed engineering overview of Lasso can be found [here](https://www.youtube.com/watch?v=iDcXj9Vx3zY).
