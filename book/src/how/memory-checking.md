# Offline Memory Checking
Offline memory checking is a method that enables a prover to demonstrate to a verifier that a read/write memory was used correctly. In such a memory system, values $v$ can be written to addresses $a$ and subsequently retrieved. This technique allows the verifier to efficiently confirm that the prover adhered to the memory's rules.

Jolt utilizes offline memory checking in the Bytecode prover, Lookup prover, and RAM prover.

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
- [Original Paper on Memory Correctness](https://www.researchgate.net/publication/226386605_Checking_the_correctness_of_memories/link/0c960526fe9ab32634000000/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)
- [Spice Protocol](https://eprint.iacr.org/2018/907.pdf)
- [Spartan Protocol](https://eprint.iacr.org/2019/550.pdf)
- [Lasso Protocol](https://people.cs.georgetown.edu/jthaler/Lasso-paper.pdf)
- [Thaler13 Protocol](https://eprint.iacr.org/2013/351.pdf)
