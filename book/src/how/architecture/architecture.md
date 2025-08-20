# Architecture overview

This section gives a overview of the architecture of the Jolt codebase. As a starting point, we will present two useful mental models of Jolt: Jolt as a CPU, and Jolt as a DAG.


## Jolt as a CPU

One way to understand Jolt is to map the components of the proof system to the components of the virtual machine (VM) whose functionality they prove.

### Jolt's five components

A VM does two things:

- Repeatedly execute the fetch-decode-execute logic of its instruction set architecture.
- Perform reads and writes to memory.

The Jolt paper depicts these two tasks mapped to three components in the Jolt proof system:

![Jolt Alpha](../imgs/figure2.png)

The Jolt codebase is similarly organized, but instead distinguishes read-write memory (comprising registers and RAM) from program code (aka bytecode, which is read-only), for a total of five components:

![fetch-decode-execute](../imgs/fetch_decode_execute.png)

#### R1CS constraints

To handle the "fetch" part of the fetch-decode-execute loop, there is a minimal R1CS instance (about 30 constraints per cycle of the RISC-V VM). These constraints handle program counter (PC) updates and serves as the "glue" enforcing consistency between polynomials used in the components below. Jolt uses [Spartan](https://eprint.iacr.org/2019/550), optimized for the highly-structured nature of the constraint system (i.e. the same small set of constraints are applied to every cycle in the execution trace).

*For more details: [R1CS constraints](./r1cs_constraints.md)*

#### RAM

To handle reads/writes to RAM Jolt uses the Twist memory checking argument.

*For more details: [RAM](./ram.md)*

#### Registers

Similar to RAM, Jolt uses the Twist memory checking argument to handle reads/writes to registers.

*For more details: [Registers](./registers.md)*


#### Instruction execution

To handle the "execute" part of the fetch-decode-execute loop, Jolt invokes the Shout lookup argument. The lookup table, in this case, effectively maps every instruction to its correct output.

*For more details: [Instruction execution](./instruction_execution.md)*

#### Bytecode

To handle the "decode" part of the fetch-decode-execute loop, Jolt uses another instance of Shout. The bytecode of the guest program is "decoded" in preprocessing, and the prover subsequently invokes offline memory-checking on the sequence of reads from this decoded bytecode corresponding to the execution trace being proven.

*For more details: [Bytecode](./bytecode.md)*


## Jolt as a DAG

One useful way to understand Jolt is to view it as a directed acyclic graph (DAG). That might sound surprising -- how can a zkVM be represented as a DAG? The key lies in the structure of its sumchecks, and in particular, the role of virtual polynomials.

### Virtual vs. Committed Polynomials

Virtual polynomials, introduced in the Binius paper (Diamond and Posen, 2023), are used heavily in Jolt.

A virtual polynomial is a part of the witness that is never committed directly. Instead, any claimed evaluation of a virtual polynomial is proven by a subsequent sumcheck.
In contrast, committed polynomials are committed to explicitly and their evaluaions are proven using the opening proof of the PCS.

Virtual polynomials form the conceptual backbone of the nodes in Jolt’s DAG: each sumcheck operates over polynomials -- some committed, some virtual -- and outputs claimed evaluations for those polynomials.

### Sumchecks as Nodes

Each sumcheck in Jolt corresponds to a node in the DAG. You can think of a sumcheck expression as:

$$
\textsf{InputClaim} = \sum_{x \in \{0,1\}^n} f(x)
$$

Here, the left-hand side is the input claim, and the right-hand side is a sum over the Boolean hypercube of a multivariate polynomial $f$.
The sumcheck protocol reduces this to a set of new claims about the constituent multilinear polynomials of f.
These new claims are the output claims of the node.
Input claims can be viewed as in-edges, and output claims as out-edges, thus defining the graph structure.

This DAG isn't just conceptual -- it's manifested in code by the `JoltDag` and the `OpeningsMap` abstractions.

### Managing State and Batching Sumchecks

The `StateManager` is responsible for tracking and managing all the sumcheck stages throughout the Jolt proof. While it’s common to batch multiple sumchecks together by taking random linear combinations, the DAG structure imposes constraints on batching. In particular, if sumcheck B depends on the outputs of sumcheck A (i.e., there's an edge from A to B), they cannot be batched together: A must be proven first.
We’ve found that Jolt’s sumchecks can be grouped into five minimal batches, respecting the DAG’s dependencies. The stages are made explicit by the StateManager and represented in the documentation via a diagram.

### Conceptual Components and Staging

Jolt has six major conceptual components:

- R1CS constraints
- RAM
- Instruction execution
- Registers
- Bytecode
- Opening proof


Each component contains multiple sumchecks, and these sumchecks may fall into different batches depending on their dependencies. Each component registers its sumchecks with the StateManager, indicating which stage each one belongs to. The StateManager then collates all the sumchecks in a given stage and executes a batched sumcheck for that stage.

### Openings Map

Within the `StateManager`, the `OpeningsMap` manages the flow of claimed evaluations –– essentially, the edges of the DAG. It's a mapping from an opening ID to a claimed polynomial evaluation. As sumchecks are proven, their output claims (both for virtual and committed polynomials) are inserted into the map. Later sumchecks can then consume the virtual polynomial openings as input claims.

While virtual polynomial claims are used internally and passed between sumchecks, committed polynomial claims are tracked because they must ultimately be verified via a batched Dory opening proof at the end of the protocol.
