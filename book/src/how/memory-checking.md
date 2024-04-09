# Offline Memory Checking
Offline memory checking is a method that enables a prover to demonstrate to a verifier that a read/write memory was used correctly. In such a memory system, values $v$ can be written to addresses $a$ and subsequently retrieved. This technique allows the verifier to efficiently confirm that the prover adhered to the memory's rules (i.e., that the value returned by any read operation is indeed the most recent value that was written to that memory cell). 

Jolt utilizes offline memory checking in the Bytecode prover, Lookup prover (for VM instruction execution), and RAM prover. The RAM prover must support a read-write memory. The Bytecode prover and Lookup prover need only support read-only memories.  In the case of the Bytecode prover, the memory is initialized to contain the Bytecode of the RISC-V program, and the memory is never modified (it's read-only). And the lookup tables used for VM instruction execution are determined entirely by the RISC-V instruction set.

(The term "offline memory checking" refers to techniques that check the correctness of all read operations "all at once", after the reads have all occurred--or in SNARK settings, after the purported values returned by the reads have been committed. Off-line checking techniques do not determine as _a read happens_ whether or not it was correct. They only ascertain, when all the reads are checked at once, whether or not all of the reads were correct. 

This is in contrast to "online memory checking" techniques like Merkle hashing that immediately confirm that a memory read was done correctly by insisting that each read includes an authentication path. Merkle hashing is much more expensive on a per-read basis for SNARK provers, and offline memory checking suffices for SNARK design. This is why Lasso and Jolt use offline memory checking techniques rather than online). 

## Initialization Algorithm
### `TODO`: 
- Initialize four timestamp counters
- Implicitly assume a read operation before each write

## Multiset Check
Define $read$ and $write$ as subsets, and $init$ and $final$ as subsets:
$$
read, write \subseteq \{(a_i, v_i, t_i) \,|\, i \in [0, m]\}
$$
$$
init, final \subseteq \{(a_i, v_i, t_i) \,|\, i \in [0, M]\}
$$
Here, $a_i$, $v_i$, and $t_i$ represent the address, value, and timestamp respectively, with $m$ being the total number of memory operations and $M$ the size of the RAM.

The verifier checks that the combination of $read$ and $final$ matches $write$ and $init$, disregarding the sequence of elements, known as a permutation check:
$$
read \cup final = write \cup init
$$
Jolt conducts this check using a homomorphic hash function applied to each set:
$$
H(read) = \prod_{(a_i, v_i, t_i) \in read} h(a_i, v_i, t_i)
$$
$$
H(write) = \prod_{(a_i, v_i, t_i) \in write} h(a_i, v_i, t_i)
$$
$$
H(init) = \prod_{(a_j, v_j, t_j) \in init} h(a_j, v_j, t_j)
$$
$$
H(final) = \prod_{(a_j, v_j, t_j) \in final} h(a_j, v_j, t_j)
$$
The hash function $h$ is defined as:
$$
h_{\gamma, \tau}(a, v, t) = a \cdot \gamma^2 + v \cdot \gamma + t - \tau
$$

This multiset hashing process is represented by a binary tree of multiplication gates and is computed using an [optimized GKR protocol](https://eprint.iacr.org/2013/351.pdf).

## References
- [Original BEGKN paper on offline memory checking, forming the technical underpinnings of Lasso](https://www.researchgate.net/publication/226386605_Checking_the_correctness_of_memories/link/0c960526fe9ab32634000000/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)
- [Spice Protocol for read-write memories (see Fig. 2)](https://eprint.iacr.org/2018/907.pdf)
- [Spartan Protocol](https://eprint.iacr.org/2019/550.pdf)
- [Lasso Protocol for read-only memories a.k.a. lookups](https://eprint.iacr.org/2023/1216.pdf)
- [Thaler13 Grand Product Protocol (see Prop. 2)](https://eprint.iacr.org/2013/351.pdf)
- [Quarks Grand Product Protocol (see Section 6)](https://eprint.iacr.org/2020/1275.pdf)
