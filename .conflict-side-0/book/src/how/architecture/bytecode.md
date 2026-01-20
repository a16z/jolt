# Bytecode

At each cycle of the RISC-V virtual machine, the current instruction (as indicated by the program counter) is "fetched" from the bytecode and decoded.
In Jolt, this is proven by treating the bytecode as a lookup table, and fetches as lookups.
To prove the correctness of these lookups, we use the [Shout](../twist-shout.md) lookup argument.

One distinguishing feature of the bytecode Shout instance is that we have multiple instances of the read-checking and $\widetilde{\textsf{raf}}$-evaluation sumchecks.
Intuitively, the bytecode serves as the "ground truth" of what's being executed, so one would expect many virtual polynomial claims to eventually lead back to the bytecode.
And this holds in practice –– in the Jolt sumcheck [DAG](./architecture/architecture.md##sumchecks-as-nodes) diagram, we see that there are three colors of edges (corresponding to stages 1, 2, and 3,) pointing to the bytecode read-checking node, and two colors of edges (corresponding to stages 1 and 3) pointing to the bytecode $\widetilde{\textsf{raf}}$-evaluation node.

Each stage has its a unique opening point, so having in-edges of different colors implies that multiple instances of that sumcheck must be run in [parallel](../optimizations/batched-sumcheck.md) to prove the different claims.

## Read-checking

Another distinguishing feature of the bytecode Shout instance is that we treat each entry of lookup table as containing a tuple of values, rather than a single value.
Intuitively, this is because each instruction in the bytecode encodes multiple pieces of information: opcode, operands, etc.

The figure below loosely depicts the relationship between bytecode and $\widetilde{\textsf{Val}}$ polynomial.

![bytecode](../../imgs/bytecode.png)

We start from some ELF file, compiled from the guest program. For each instruction in the ELF (raw bytes), we decode/preprocess the instruction into a structured format containing the individual witness values used in Jolt:

- The instruction operands `rs1`, `rs2`, `rd`, `imm`
- Circuit and lookup table [flags](#flags)
- The instruction [address](#instruction-address)

Then, we compute a [Reed-Solomon fingerprint](https://publish.obsidian.md/matteo/3.+Permanent+notes/Reed-Solomon+Fingerprinting) of some subset of the values in the tuple, depending on what $\widetilde{\textsf{rv}}$ claims are being proven. These fingerprints serve as the coefficients of the $\widetilde{\textsf{Val}}$ polynomial for that read-checking instance.

The figure above depicts the $\widetilde{\textsf{Val}}$ polynomial we would use to prove the read-checking instance for the Stage 2 claims, labeled as "ra/wa claims" in the [DAG]((./architecture/architecture.md##sumchecks-as-nodes)) diagram.
As explained [below](#registers), the $\widetilde{\textsf{ra}}$ and $\widetilde{\textsf{wa}}$ polynomials correspond to the registers (`rs1`, `rs2`, `rd`) in the bytecode.
Each coefficient in $\widetilde{\textsf{Val}}$ is $\texttt{rd} + \gamma \cdot \texttt{rs1} + \gamma^2 \cdot \texttt{rs2}$, for the registers of one particular instruction.

We can write the read-checking sumcheck expression for the Stage 2 claims as the following:

$$
\widetilde{\textsf{rv}}_{\texttt{rd}}(r) + \gamma \cdot \widetilde{\textsf{rv}}_{\texttt{rs1}}(r) + \gamma^2 \cdot \widetilde{\textsf{rv}}_{\texttt{rs2}}(r) = \sum_{k, j} \widetilde{\textsf{eq}}(r, j) \cdot \widetilde{\textsf{ra}}(k, j) \cdot \left(\widetilde{\textsf{Val}}_\texttt{rd}(k) + \gamma \cdot \widetilde{\textsf{Val}}_\texttt{rs1}(k) + \gamma^2 \cdot \widetilde{\textsf{Val}}_\texttt{rs2}(k)\right)
$$

where we treat

$$
\widetilde{\textsf{Val}}(k) = \widetilde{\textsf{Val}}_\texttt{rd}(k) + \gamma \cdot \widetilde{\textsf{Val}}_\texttt{rs1}(k) + \gamma^2 \cdot \widetilde{\textsf{Val}}_\texttt{rs2}(k)
$$

as a single multilinear polynomial.

Observe that this sumcheck satisfies our needs for the Stage 2 claims: $\widetilde{\textsf{rv}}_{\texttt{rd}}(r)$, $\widetilde{\textsf{rv}}_{\texttt{rs1}}(r)$, $\widetilde{\textsf{rv}}_{\texttt{rs2}}(r)$ are the claims output by the registers read/write-checking sumcheck, and we can simply compute Reed-Solomon fingerprint of these claims to obtain the left-hand side (i.e. input claim) of sumcheck expression above.

### Registers

Technically, the explanation above glosses over some details for the sake of exposition.
To be precise, the claims output by registers read/write-checking are actually:

$$
\widetilde{\textsf{wa}}_{\texttt{rd}}(r_\text{address}, r_\text{cycle}) \\
\widetilde{\textsf{ra}}_{\texttt{rs1}}(r_\text{address}, r_\text{cycle}) \\
\widetilde{\textsf{ra}}_{\texttt{rs2}}(r_\text{address}, r_\text{cycle})
$$

where $r_\text{address} \in \mathbb{F}^{\log (\text{\# registers})}$.

So we define:

$$
\widetilde{\textsf{Val}}(k) = \widetilde{\texttt{rd}}(k, r_\text{address}) + \gamma \cdot \widetilde{\texttt{rs1}}(k, r_\text{address}) + \gamma^2 \cdot \widetilde{\texttt{rs2}}(k, r_\text{address})
$$

where $\forall k \in \{0, 1\}^{\log (\text{bytecode size})}, k' \in \{0, 1\}^{\log (\text{\# registers})}$:

- $\widetilde{\texttt{rd}}(k, k') = 1$ if the $k$-th instruction in the bytecode has $rd = k'$
- $\widetilde{\texttt{rd}}(k, k') = 0$ otherwise

and similarly for $\widetilde{\texttt{rs1}}$ and $\widetilde{\texttt{rs2}}$.

Equivalently,

$$
\widetilde{\textsf{Val}}(k) = \widetilde{\textsf{eq}}(\texttt{rd}[k], r_\text{address}) + \gamma \cdot \widetilde{\textsf{eq}}(\texttt{rs1}[k], r_\text{address}) + \gamma^2 \cdot \widetilde{\textsf{eq}}(\texttt{rs2}[k], r_\text{address})
$$

where $\texttt{rd}[k]$ (respectively, $\texttt{rs1}[k]$ and $\texttt{rs2}[k]$) is the bitvector denoting the `rd` (respectively, `rs1` and `rs2`) for the $k$-th instruction in the bytecode.

### Instruction address

Each instruction in the bytecode has two associated "addresses":

- its index in the **expanded** bytecode. "Expanded" bytecode refers to the preprocessed bytecode, after instructions are expanded to their [virtual sequences](./emulation.md#virtual-instructions-and-sequences)
- its memory address as given by ELF. All the instructions in a virtual sequence are assigned the address of the "real" instruction they were expanded from.

The former is used for the $\widetilde{\textsf{raf}}$-evaluation sumcheck, described below.
The latter is used to enforce program counter updates in the [R1CS constraints] (./r1cs_constraints.md), and is treated as a part of the tuple of values in the preprocessed bytecode.

The "outer" and shift sumchecks in Spartan output claims about the virtual `UnexpandedPC` polynomial, which corresponds to the latter. These claims are proven using bytecode read-checking.

### Flags

There are two types of Boolean flags used in Jolt:

- [Circuit flags](./r1cs_constraints.md#circuit-flags), used in R1CS constraints
- [Lookup table flags](./instruction_execution.md#multiplexing-between-instructions), used in the instruction execution Shout

The associated flags for a given instruction in the bytecode can be computed a priori (i.e. in preprocessing), so any claims about these flags arising output by Spartan or instruction execution Shout are also proven using bytecode read-checking.

## raf-evaluation

There are two $\widetilde{\textsf{raf}}$-evaluation instances for bytecode, corresponding to $\widetilde{\textsf{raf}}$ claims arising from stages 1 and 3.

The $\widetilde{\textsf{raf}}$ polynomial, in the context of bytecode, is the program counter (PC). The two sumchecks that output claims about the PC are the "outer" and "shift" sumchecks used to prove Jolt's [R1CS](./r1cs_constraints.md) constraints.

## One-hot checks

Jolt enforces that the $\widetilde{\textsf{ra}}_i$ polynomials used for bytecode Shout are [one-hot](../twist-shout.md#one-hot-polynomials), using a Booleanity and Hamming weight sumcheck as described in the paper.
These implementations follow the Twist and Shout paper closely, with no notable deviations.
