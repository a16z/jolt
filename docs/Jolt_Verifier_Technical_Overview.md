# Jolt Verifier: Technical Overview for Cryptographers

This document explains how Jolt verifies correct execution of RISC-V programs. It targets cryptographers familiar with zero-knowledge proof systems, covering the mathematical structure and verification algorithm without focusing on implementation costs or line-by-line code details.

## What Jolt Proves

Jolt is a zkVM (zero-knowledge virtual machine) that proves correct execution of RISC-V programs. Given a program $P$ (compiled RISC-V bytecode), input $x$, and claimed output $y$, Jolt produces a proof $\pi$ that:

$$P(x) = y$$

More precisely, the proof establishes that executing program $P$ on input $x$ using a RISC-V emulator produces output $y$, or that the program panicked (trapped) in a specific manner consistent with the RISC-V specification.

The verifier checks $\pi$ without re-executing the program. For a program that runs for $T$ cycles, verification time is $O(\log T)$ with proof size $O(\log T)$ (assuming succinct polynomial commitments). The prover time is roughly $O(T)$, making Jolt one of the fastest zkVMs for CPU-based proving.

## High-Level Architecture

Jolt's proof system decomposes VM verification into five core components, each handling a different aspect of CPU behavior:

1. **R1CS Constraints**: Program counter updates, component linking, arithmetic operations
2. **RAM**: Memory read/write correctness
3. **Registers**: Register read/write correctness
4. **Instruction Execution**: Correct input-output behavior for each instruction
5. **Bytecode**: Instruction fetch and decode correctness

Each component is proven using a combination of sum-check protocols, memory-checking arguments (Twist), and lookup arguments (Shout). The entire proof is structured as a directed acyclic graph (DAG) of sum-check instances, organized into 5-6 verification stages depending on whether optimizations are enabled.

### Why This Decomposition?

Traditional zkVMs arithmetize the entire CPU circuit: every instruction becomes a set of constraints over field elements. This approach works but results in large constraint systems ($k$ constraints per instruction, often $k \geq 100$).

Jolt takes a different approach inspired by the "lookup singularity" observation: precomputing entire CPU behavior into lookup tables and proving correct table accesses is more efficient than proving the computation directly. A 64-bit ADD instruction doesn't need 100+ constraints—it needs 16 lookups into a tiny 256-entry table (4-bit ADD), plus proofs that carry bits are handled correctly.

This lookup-centric architecture is enabled by recent advances in lookup arguments (Lasso/Shout) that can handle massive tables ($2^{128}$ entries) efficiently through prefix-suffix decomposition. The result: Jolt uses approximately 30 R1CS constraints per cycle instead of hundreds, with most instructions proven via efficient lookups.

## The Proof DAG Structure

The proof is organized as a directed acyclic graph where nodes are sum-check instances and edges are polynomial evaluation claims. Understanding this DAG structure is key to understanding verification.

### Virtual vs. Committed Polynomials

Polynomials in Jolt come in two flavors:

**Virtual polynomials**: Never committed explicitly. Part of the witness that's used internally during proof generation. When a sum-check instance outputs an evaluation claim for a virtual polynomial $\tilde{P}(r) = v$, this claim must be verified by another sum-check later in the DAG. Virtual polynomials serve as intermediate witnesses connecting different proof components.

**Committed polynomials**: Explicitly committed using Dory polynomial commitment scheme during preprocessing or proving. When a sum-check outputs an evaluation claim for a committed polynomial, the claim is accumulated in an opening accumulator. All accumulated opening claims are verified together in Stage 5 through a batched opening proof.

This distinction matters for soundness: virtual polynomial claims are "passed forward" as input claims to other sum-checks, creating the DAG structure. Committed polynomial claims "bottom out" at the polynomial commitment scheme, which provides the cryptographic binding.

### How the DAG Enforces Order

Each sum-check instance has input claims (polynomial evaluations it assumes correct) and output claims (polynomial evaluations it reduces to). A sum-check cannot execute until all its input claims are available.

Input claims come from:
- Prior sum-checks in the same or earlier stages (virtual polynomials)
- The opening proof in Stage 5 (committed polynomials)

This creates a partial order: Stage 1 sum-checks have no dependencies and can start immediately. Stage 2 sum-checks depend on Stage 1 outputs. And so on. Within a stage, independent sum-checks run in parallel (batched together to reduce rounds of interaction).

Stage 5 is necessarily last among the component stages because it verifies all committed polynomial openings that accumulated during Stages 1-4. Stage 6 (in optimized verification) comes after Stage 5 because it verifies the correctness of Stage 5's hint-based shortcut.

## Verification Stages in Detail

### Stage 1: Initial Sum-Checks

Stage 1 launches verification with sum-checks that have no dependencies on polynomial evaluations from other sum-checks. These are the "root nodes" of the DAG.

#### Spartan Outer Sum-Check

R1CS verification begins with Spartan's outer sum-check. Jolt uses approximately 30 R1CS constraints per execution cycle of the form:

$$\mathbf{Az} \circ \mathbf{Bz} = \mathbf{Cz}$$

where $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{F}^{N \times M}$ are sparse constraint matrices, $\mathbf{z} \in \mathbb{F}^M$ is the witness vector (containing register values, memory contents, instruction operands, etc.), and $\circ$ denotes Hadamard (element-wise) product. With $T$ execution cycles, $N = O(T)$ constraints.

Spartan's key insight is to express this constraint check as a sum over the Boolean hypercube. Define multilinear extensions $\tilde{A}, \tilde{B}, \tilde{C}$ of the constraint matrices and $\tilde{z}$ of the witness. Then the R1CS claim becomes:

$$\sum_{x \in \{0,1\}^{\log N}} \left( \tilde{A}(x, \cdot) \cdot \tilde{z} \right) \left( \tilde{B}(x, \cdot) \cdot \tilde{z} \right) = \sum_{x \in \{0,1\}^{\log N}} \left( \tilde{C}(x, \cdot) \cdot \tilde{z} \right)$$

The outer sum-check verifies this equality. The prover sends univariate polynomials for each round, the verifier checks consistency and samples random challenges from the transcript, and after $\log N$ rounds the sum-check reduces to a claim about evaluations of $\tilde{A}, \tilde{B}, \tilde{C}, \tilde{z}$ at a random point $r_x$.

**Uniformity optimization**: Jolt's constraints are uniform—the same 30 constraints apply to every cycle. This means $\tilde{A}, \tilde{B}, \tilde{C}$ have special structure: they can be written as $\tilde{A}(i, j) = \tilde{A}_{\text{cycle}}(j) \cdot \text{eq}(i, \text{cycle}(j))$ where $\tilde{A}_{\text{cycle}}$ describes one cycle's constraints and the eq term selects which cycle. Spartan exploits this to avoid materializing the full $N \times M$ matrices.

#### Twist Initialization (RAM)

RAM verification uses Twist memory checking to prove that all reads return the most recently written value (or the initial value if no prior write). With memory size $K$ and $T$ memory operations, this is proven by showing that the time-ordered trace (accesses in execution order) is a permutation of the address-ordered trace (accesses sorted by address, then timestamp).

The permutation is proven using grand product equality over fingerprints. Define fingerprint function:

$$\phi(a, t, v; \alpha, \beta, \gamma) = \alpha \cdot a + \beta \cdot t + \gamma \cdot v$$

where $(a, t, v)$ is a memory access tuple (address, timestamp, value) and $\alpha, \beta, \gamma$ are random challenges from Fiat-Shamir. The Twist protocol proves:

$$\prod_{i=1}^T \phi(\text{time-ordered}[i]) = \prod_{i=1}^T \phi(\text{address-ordered}[i])$$

If the two traces are not valid permutations (i.e., a read returned the wrong value), the fingerprints will differ with overwhelming probability (Schwartz-Zippel).

Stage 1 contains the first of three Twist sum-checks for RAM: the **read-checking sum-check**. This verifies the sum of logarithms of fingerprints for the time-ordered trace, which will later be compared against the address-ordered trace's sum in the write-checking sum-check (Stage 2).

The Twist protocol uses the "local" prover algorithm, meaning the prover computes these sums incrementally without needing global knowledge of the entire trace. Products are converted to sums via logarithms (in the exponent of a generator), allowing sum-check to verify them efficiently.

**RAM parameters**: Twist has two parameters:
- $K$: Memory size (number of addressable cells)
- $d$: Chunking parameter for polynomial commitment

Jolt chooses $d$ dynamically such that $K^{1/d} \approx 2^8$, balancing commitment size (which grows with $K^{1/d}$) against sum-check complexity (which grows with $d$). For a guest program using 16MB of memory ($K = 2^{21}$ doublewords), $d = 3$ gives $K^{1/3} = 2^7 \approx 128$.

#### Twist Initialization (Registers)

Register verification uses the same Twist protocol as RAM but with different parameters:
- $K = 64$: 32 architectural RISC-V registers plus 32 virtual registers (scratch space for virtual instruction sequences)
- $d = 1$: No chunking needed for small register file

The key difference is that registers have three operations per cycle instead of one:
- Read from source register 1 (rs1)
- Read from source register 2 (rs2)
- Write to destination register (rd)

This is encoded as three separate Twist polynomials: $ra_{rs1}$, $ra_{rs2}$, $wa_{rd}$. The protocol runs two read-checking sum-checks and one write-checking sum-check, all batched together in Stage 1.

**Virtual registers**: Registers 32-63 don't exist in the RISC-V ISA. They're used exclusively within virtual instruction sequences—multi-instruction expansions of complex operations like division. For example, a division instruction might expand to: (1) virtual "advice" instruction stores quotient and remainder in virtual registers 32-33, (2) multiplication and addition verify quotient and remainder are correct, (3) final instruction stores quotient in architectural destination register. Virtual registers never appear in the architectural state visible to the program.

#### Shout Read-Checking (Instruction Execution)

Instruction execution verification proves that every instruction's input-output behavior matches its specification. For a 64-bit binary operation like ADD or XOR, this conceptually requires checking against a $2^{128}$-entry lookup table (two 64-bit operands). Shout makes this tractable through prefix-suffix decomposition.

The key insight: most CPU operations decompose into independent chunks. A 64-bit XOR is just 16 independent 4-bit XOR operations. A 64-bit ADD is 16 4-bit ADDs with carry chain (the carries must be verified but they're small). This decomposition allows the lookup table to be represented as:

$$\tilde{T}(k_{\text{prefix}}, k_{\text{suffix}}) = f_{\text{prefix}}(k_{\text{prefix}}) \cdot f_{\text{suffix}}(k_{\text{suffix}})$$

where $f_{\text{prefix}}$ and $f_{\text{suffix}}$ are efficiently computable functions (not tabulated). For XOR, $f_{\text{prefix}}$ computes XOR on the high bits and $f_{\text{suffix}}$ computes XOR on the low bits, and they combine multiplicatively.

Stage 1 begins the prefix-suffix sum-check that verifies lookups into this factored table. The sum-check will proceed through 8 phases of 16 rounds each (128 rounds total for $\log 2^{128} = 128$ variables), alternating between binding prefix variables and suffix variables. Stage 1 contains the first phase.

**Why prefix-suffix works**: The alternating structure maintains the factorization throughout the sum-check. After binding some prefix variables and some suffix variables, the remaining sum still factors as $\sum_{\text{prefix}} \sum_{\text{suffix}} (\cdots)$, allowing separate evaluation of prefix and suffix components. This keeps evaluation cost polylogarithmic rather than exponential.

#### Shout Read-Checking (Bytecode)

Bytecode verification proves the instruction fetch phase: that the opcode, register addresses, and immediate values used during execution match the committed program bytecode. During preprocessing, the program bytecode is decoded into constituent fields and committed. During verification, Shout proves that every cycle's instruction fetch reads the correct values from this committed bytecode.

This is essentially offline memory checking: the committed bytecode is "memory" and the execution trace's instruction fetches are "accesses." Shout proves all accesses are consistent with the committed memory.

The bytecode Shout instance reads:
- Opcode (which RISC-V instruction)
- Register addresses: rs1, rs2, rd
- Circuit flags: boolean indicators like is_jump, is_branch, is_load (derived deterministically from opcode)
- Immediate value (constant operand encoded in instruction word)

These read values are used throughout other components. For example, R1CS constraints use circuit flags to enable/disable specific constraints for each instruction type. Register Twist uses rs1/rs2/rd to determine which registers are accessed. But these values are never committed separately—they're virtualized through bytecode read-checking. Their correctness is implied by the correctness of bytecode reads, proven by Shout.

Stage 1 begins the bytecode Shout prefix-suffix sum-check, following the same structure as instruction execution Shout.

#### Sum-Check Batching

All Stage 1 sum-checks are batched together using random linear combination. Instead of running 5+ separate sum-check protocols (Spartan outer, RAM read-check, register read-checks, instruction Shout, bytecode Shout), the verifier generates random coefficients $\gamma_1, \ldots, \gamma_k$ and combines them into a single sum-check over the polynomial:

$$g_{\text{batch}}(x) = \gamma_1 \cdot g_1(x) + \gamma_2 \cdot g_2(x) + \cdots + \gamma_k \cdot g_k(x)$$

The prover sends one univariate polynomial per round (instead of $k$ univariates). The verifier checks the batched consistency relation. After all rounds, the batch sum-check reduces to evaluation claims for the individual polynomials $g_1, \ldots, g_k$ at the random point.

Batching reduces interaction rounds (one set of challenges instead of $k$ sets) and allows parallel processing. Security is preserved because if the prover cheats on any individual sum-check, the random linear combination will fail with overwhelming probability (coefficients are random, so cheating on $g_i$ causes $g_{\text{batch}}$ to be incorrect).

### Stage 2: Dependent Sum-Checks

Stage 2 processes sum-checks that depend on evaluation claims from Stage 1. These sum-checks cannot start until Stage 1 completes and provides the necessary polynomial evaluations.

#### Spartan Product Sum-Check

The Spartan outer sum-check (Stage 1) reduced the R1CS claim to evaluation claims at a random point $r_x$. Stage 2 continues with Spartan's product sum-check, which verifies the inner product structure:

$$\sum_{y \in \{0,1\}^{\log M}} \tilde{A}(r_x, y) \cdot \tilde{z}(y) \cdot \tilde{B}(r_x, y) \cdot \tilde{z}(y) = \sum_{y \in \{0,1\}^{\log M}} \tilde{C}(r_x, y) \cdot \tilde{z}(y)$$

This is a sum over the witness dimension $M$ (rather than constraint dimension $N$). The sum-check proceeds for $\log M$ rounds, reducing the claim to evaluations of $\tilde{A}, \tilde{B}, \tilde{C}$ at $(r_x, r_y)$ and $\tilde{z}$ at $r_y$, where $r_y$ is the random point from this sum-check.

**Sparsity exploitation**: The constraint matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}$ are highly sparse (most entries are zero). Spartan uses SPARK encoding to represent these sparse matrices compactly and evaluate their MLEs efficiently. Instead of summing over all $2^{\log M}$ points, the prover sums only over non-zero entries, achieving $O(|\mathbf{A}| + |\mathbf{B}| + |\mathbf{C}|)$ work where $|\mathbf{A}|$ is the number of non-zero entries in $\mathbf{A}$.

For Jolt, the sparsity is extreme: 30 constraints per cycle with a small number of non-zero entries per constraint. This makes Spartan's product sum-check very efficient despite the large witness dimension.

#### Twist Write-Checking (RAM)

The RAM Twist protocol continues with the write-checking sum-check. While read-checking (Stage 1) verified the time-ordered trace, write-checking verifies the address-ordered trace. The two traces are both sorted by address, then by timestamp within each address, ensuring that for each memory cell, writes and reads appear in chronological order.

The write-checking sum-check computes the sum of logarithms of fingerprints for the address-ordered trace:

$$\sum_{i=1}^T \log \phi(\text{address-ordered}[i])$$

If this sum equals the read-checking sum from Stage 1, the grand products are equal, proving the traces are permutations. If a read returned the wrong value, the fingerprints differ, the sums differ, and verification fails.

**Incremental products**: The local prover algorithm computes these sums incrementally. For address-ordered, the prover iterates through memory cells in address order, accumulating the fingerprint product. This is efficient because the prover can compute fingerprints on-the-fly without storing the entire address-ordered trace.

#### Twist Write-Checking (Registers)

Register Twist has the same structure as RAM Twist but runs three sum-checks: two for read-checking (rs1 and rs2) and one for write-checking (rd). Stage 2 contains the write-checking sum-check, which verifies that the address-ordered write trace (sorted by register index, then timestamp) accumulates the correct fingerprint sum.

Because registers have $K = 64$ (small), there's no chunking ($d = 1$), and the sum-check is straightforward. The key optimization is bytecode virtualization: register addresses don't need one-hot encoding checks because they're hardcoded in the bytecode. If the bytecode Shout protocol verifies correctly, register addresses are guaranteed to be well-formed.

#### Shout Middle Phases

The instruction execution and bytecode Shout protocols continue with middle phases of the prefix-suffix sum-check. Each Shout instance runs 8 phases with 16 rounds each, alternating between prefix and suffix. Stage 2 contains phases 2-4 (varies by exact sumcheck structure).

During these phases, the sum-check progressively binds variables, reducing the dimension of the remaining sum. The alternating prefix-suffix structure is maintained: after binding half the prefix variables and half the suffix variables, the remaining sum is still:

$$\sum_{k_p \in \{0,1\}^{\ell_p}} \sum_{k_s \in \{0,1\}^{\ell_s}} \tilde{P}(k_p; r_p) \cdot \tilde{S}(k_s; r_s)$$

where $r_p$ and $r_s$ are the bound prefix and suffix variables, and $\ell_p$, $\ell_s$ are the remaining dimensions.

**Efficient MLE evaluation**: At each round, the prover must evaluate the multilinear extension of the lookup table at the current bound point. For Jolt's instruction tables, this is efficient because the MLEs have closed-form expressions. For example, the 4-bit XOR table's MLE can be computed as:

$$\tilde{XOR}_4(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4) = \prod_{i=1}^4 (x_i + y_i - 2 x_i y_i)$$

This takes $O(1)$ time to evaluate, not $O(2^8)$ to look up in a table. Every Jolt instruction implements the `JoltLookupTable` trait with an efficient evaluation function.

### Stage 3: Late-Stage Sum-Checks

Stage 3 handles sum-checks that depend on Stage 2 outputs. This includes Spartan's final sum-check, the completion of Twist protocols, and late phases of Shout protocols.

#### Spartan Matrix Evaluation Sum-Check

Spartan's product sum-check (Stage 2) reduced to evaluation claims for $\tilde{A}(r_x, r_y)$, $\tilde{B}(r_x, r_y)$, $\tilde{C}(r_x, r_y)$. These are committed polynomials (the constraint matrices are committed during preprocessing), so one option is to prove the evaluations via opening proof in Stage 5.

However, for sparse matrices, it's more efficient to prove evaluations via sum-check. Spartan's final sum-check verifies:

$$\tilde{A}(r_x, r_y) = \sum_{i,j} A_{ij} \cdot \text{eq}(i, r_x) \cdot \text{eq}(j, r_y)$$

where $A_{ij}$ are the non-zero entries of $\mathbf{A}$ and eq is the multilinear extension of equality. Because $\mathbf{A}$ is sparse, this sum only includes $|\mathbf{A}|$ terms (non-zero entries), making the sum-check efficient.

The same applies to $\tilde{B}$ and $\tilde{C}$. All three are batched into a single sum-check using random linear combination.

**SPARK commitment**: The sparse polynomial commitment (SPARK) binds the prover to the non-zero structure of $\mathbf{A}, \mathbf{B}, \mathbf{C}$. During preprocessing, the prover commits to the positions and values of non-zero entries. During the sum-check, the verifier checks that the sum only includes committed non-zero entries. This prevents the prover from cheating by adding extra terms or changing values.

#### Twist Evaluation Sum-Checks (RAM and Registers)

The final Twist sum-check for both RAM and registers verifies that the final memory state matches the initial state plus all increments. For RAM, this proves the claimed program outputs are correct. For registers, this proves the final register values match the claimed execution result.

The evaluation sum-check verifies:

$$\sum_{a \in \{0,1\}^{\log K}} \tilde{v}_{\text{final}}(a) = \sum_{a \in \{0,1\}^{\log K}} \left( \tilde{v}_{\text{init}}(a) + \Delta(a) \right)$$

where $\tilde{v}_{\text{init}}$ is the initial memory state (committed during preprocessing or derived from program inputs), $\tilde{v}_{\text{final}}$ is the claimed final state, and $\Delta(a)$ is the net change to address $a$ computed from the access trace.

For RAM, the output check is integrated into this sum-check: the verifier computes a sum over the I/O region of memory and checks that it matches the claimed program outputs. If the prover claims output $y$ but the final memory state doesn't contain $y$ in the output region, this sum-check fails.

#### Shout Completion

The final phases of instruction execution and bytecode Shout protocols complete in Stage 3. After 8 phases and 128 rounds, the prefix-suffix sum-check reduces to a small number of base case evaluations: the lookup table MLE evaluated at the final random point.

For instruction execution, this means evaluating the instruction's lookup table at a random $(operand1, operand2)$ pair. For a 64-bit ADD, the MLE evaluation computes the ADD of the random operands (expressed in the prefix-suffix factored form). This is efficient because the ADD operation is simple, and the factorization allows computing prefix and suffix separately.

For bytecode, the MLE evaluation looks up the opcode/registers/flags for a random program counter. Because the bytecode is committed during preprocessing, the verifier has access to this committed representation and can check the evaluation claim via the opening proof in Stage 5.

### Stage 4: Cross-Component Linking

Stage 4 handles sum-checks that enforce consistency between independent components. The main work here is linking constraints in the R1CS system.

Jolt's five components (R1CS, RAM, registers, instruction execution, bytecode) are proven somewhat independently:
- RAM Twist proves memory is consistent
- Register Twist proves registers are consistent
- Instruction Shout proves each instruction's I/O is correct
- Bytecode Shout proves instructions were decoded correctly

But these components must agree with each other. A load instruction involves both RAM (reading memory) and registers (writing the destination register). The value read from RAM must match the value written to the register. This agreement is enforced by R1CS linking constraints.

#### Linking Constraints

Jolt uses approximately 30 R1CS constraints per cycle. Roughly 10 of these are linking constraints:

**Load/store linking**: For load instructions (LB, LH, LW, LD), the R1CS constraint:

$$\text{load\_flag} \cdot (\text{RAM\_read\_value} - \text{rd\_write\_value}) = 0$$

enforces that when load_flag is 1 (this is a load instruction), the value read from RAM equals the value written to destination register rd. Both RAM_read_value (from RAM Twist) and rd_write_value (from register Twist) appear in the witness $\mathbf{z}$, so this is a linear constraint. If they disagree, the constraint is violated, and Spartan's sum-check fails.

Similarly for store instructions:

$$\text{store\_flag} \cdot (\text{rs2\_read\_value} - \text{RAM\_write\_value}) = 0$$

**Instruction linking**: For arithmetic instructions, the R1CS constraints link the instruction Shout output to the register write:

$$\text{add\_flag} \cdot (\text{ADD\_output} - \text{rd\_write\_value}) = 0$$

where ADD_output comes from the instruction execution Shout (proving the ADD lookup was correct) and rd_write_value comes from register Twist.

**PC update constraints**: The program counter (PC) must update correctly:
- Normal instructions: $\text{PC}_{\text{next}} = \text{PC}_{\text{current}} + 4$
- Jump instructions: $\text{PC}_{\text{next}} = \text{jump\_target}$
- Branch instructions: $\text{PC}_{\text{next}} = \text{branch\_taken} \cdot \text{branch\_target} + (1 - \text{branch\_taken}) \cdot (\text{PC}_{\text{current}} + 4)$

These are enforced by R1CS constraints using boolean flags (jump_flag, branch_flag) from bytecode reads.

#### Why Stage 4?

Linking constraints appear in Stage 4 because they depend on values produced by other components' sum-checks. For example, the load linking constraint uses RAM_read_value, which is only available after RAM's Twist sum-checks produce it (Stages 1-3). Similarly, instruction output values come from Shout sum-checks (Stages 1-3).

By placing linking constraints in Stage 4, the verifier ensures all necessary values are available before checking consistency.

### Stage 5: Batched Polynomial Opening

Stage 5 verifies all committed polynomial evaluation claims accumulated during Stages 1-4. Rather than verify each claim individually, Dory's batched opening proof checks all claims together efficiently.

#### Committed Polynomials in Jolt

Committed polynomials include:
- **Witness polynomials**: Register values, memory contents, instruction operands (committed during proving)
- **Bytecode polynomials**: Decoded instruction fields (committed during preprocessing)
- **Constraint matrices**: $\tilde{A}, \tilde{B}, \tilde{C}$ for Spartan (committed during preprocessing)
- **Sparse structure**: Positions and values of non-zero entries (SPARK commitments)

For a program with $T$ cycles and $M$ witness variables, there are approximately 29 committed polynomials (exact number varies by program size and structure). Each polynomial was evaluated at a random point during Stages 1-4, producing 29 opening claims of the form:

$$P_i(r_i) = v_i$$

where $P_i$ is the committed polynomial, $r_i$ is the evaluation point, and $v_i$ is the claimed value.

#### Random Linear Combination (Batching)

The verifier batches these 29 claims using random linear combination. It generates random challenges $\gamma_1, \ldots, \gamma_{29}$ from the Fiat-Shamir transcript and computes:

$$C_{\text{batch}} = \sum_{i=1}^{29} \gamma_i \cdot C_i$$

where $C_i$ is the commitment to $P_i$. Computing this batched commitment requires 29 $\mathbb{G}_T$ exponentiations in BN254's target group:

$$C_{\text{batch}} = C_1^{\gamma_1} \cdot C_2^{\gamma_2} \cdots C_{29}^{\gamma_{29}}$$

The verifier also computes the batched evaluation claim:

$$v_{\text{batch}} = \sum_{i=1}^{29} \gamma_i \cdot v_i$$

and batched evaluation point (handling multi-opening: different polynomials evaluated at different points, but batching is still possible through careful construction).

**Why batching works**: If the prover cheats on even one opening claim (claims $P_i(r_i) = v_i'$ when actually $P_i(r_i) = v_i \neq v_i'$), the batched claim will be incorrect with overwhelming probability. The random $\gamma_i$ ensure that the false claim $v_i'$ contributes to $v_{\text{batch}}$ in a way that doesn't match the true polynomial evaluation. Schwartz-Zippel lemma: the batched polynomial $\sum \gamma_i P_i$ evaluated at the batched point won't equal the claimed $v_{\text{batch}}$.

#### Dory Verification Protocol

Dory is a transparent polynomial commitment scheme (no trusted setup) with logarithmic verification time. It commits to multilinear polynomials by representing them as matrices.

**Matrix representation**: A multilinear polynomial $P$ over $n = 2\nu$ variables is represented as a $\sqrt{N} \times \sqrt{N}$ matrix where $N = 2^n$. The commitment involves structured reference strings (SRS) in groups $\mathbb{G}_1$, $\mathbb{G}_2$, and $\mathbb{G}_T$ (BN254's pairing groups).

**Verification**: To verify an opening claim $P(r) = v$, Dory runs a folding protocol with $\nu$ rounds. For Jolt with polynomial degree $N = 2^{16}$, this is $\nu = 8$ rounds. Each round:

1. Prover sends folding messages (commitments to intermediate folded polynomials)
2. Verifier generates random challenge $\alpha$ from transcript
3. Verifier computes folded commitment using 10 $\mathbb{G}_T$ exponentiations
4. Both parties update the folded polynomial evaluation claim

After $\nu$ rounds, the protocol reduces to a constant-size base case verified using 5 pairing operations.

**Folding formula**: In each round, the verifier updates the commitment as:

$$C_{\text{folded}} = C_L^{\alpha} \cdot C_R \cdot D_1^{f_1(\alpha)} \cdot D_2^{f_2(\alpha)} \cdots$$

where $C_L, C_R$ are left/right halves of the current commitment, $D_i$ are prover messages, and $f_i$ are polynomials in $\alpha$ derived from the protocol structure. Each exponentiation operates in $\mathbb{G}_T$, which is a degree-12 extension field $\mathbb{F}_{q^{12}}$ for BN254.

The total $\mathbb{G}_T$ exponentiations in Stage 5: 29 (from RLC batching) + 80 (from 8 folding rounds × 10 per round) = **109 exponentiations**.

**Final pairing check**: After folding, the verifier performs 5 pairing operations to check the base case:

$$e(G_1, H_2) \stackrel{?}{=} C_{\text{final}} \cdot e(X_1, Y_2)^v$$

where $e$ is the BN254 optimal Ate pairing, $G_1, X_1 \in \mathbb{G}_1$, $H_2, Y_2 \in \mathbb{G}_2$, and $C_{\text{final}} \in \mathbb{G}_T$ is the fully folded commitment.

#### Extension Field Arithmetic

Operations in $\mathbb{G}_T$ are the dominant cost in Stage 5. $\mathbb{G}_T$ is the target of BN254's pairing, specifically the degree-12 extension field $\mathbb{F}_{q^{12}}$ where $q \approx 2^{254}$ is BN254's base field size.

**Tower construction**: The extension is built as a tower:

$$\mathbb{F}_q \to \mathbb{F}_{q^2} \to \mathbb{F}_{q^6} \to \mathbb{F}_{q^{12}}$$

Each level uses irreducible polynomials:
- $\mathbb{F}_{q^2} = \mathbb{F}_q[u]/(u^2 - \xi)$
- $\mathbb{F}_{q^6} = \mathbb{F}_{q^2}[v]/(v^3 - \xi)$
- $\mathbb{F}_{q^{12}} = \mathbb{F}_{q^6}[w]/(w^2 - \gamma)$

where $\xi, \gamma$ are chosen to make the polynomials irreducible.

**Karatsuba multiplication**: At each tower level, multiplication uses Karatsuba's algorithm to reduce operation count. For example, $\mathbb{F}_{q^6}$ multiplication (normally 9 $\mathbb{F}_{q^2}$ multiplications for naive $(a_0 + a_1 v + a_2 v^2)(b_0 + b_1 v + b_2 v^2)$) reduces to 6 $\mathbb{F}_{q^2}$ multiplications with Karatsuba.

**Granger-Scott cyclotomic squaring**: Elements in $\mathbb{G}_T$ belong to the cyclotomic subgroup (elements $x$ satisfying $x^{q^6 - 1} = 1$). This allows compressed representation: instead of 12 $\mathbb{F}_q$ elements, only 6 are needed (the other 6 can be derived). Squaring in the cyclotomic subgroup (Granger-Scott algorithm) costs only 6 $\mathbb{F}_{q^2}$ multiplications instead of 18, a 3× speedup.

**Exponentiation**: A full $\mathbb{G}_T$ exponentiation $g^s$ for 254-bit scalar $s$ uses binary exponentiation: 254 squarings (some can use cyclotomic) and approximately 127 general multiplications (depending on Hamming weight of $s$). Each squaring costs 6 $\mathbb{F}_{q^2}$ multiplications (Granger-Scott), and each general multiplication costs 18 $\mathbb{F}_{q^2}$ multiplications. Karatsuba reduces $\mathbb{F}_{q^2}$ multiplications to 3 base field $\mathbb{F}_q$ multiplications.

Total per exponentiation: roughly $(254 \times 6 + 127 \times 18) \times 3 \approx 11,000$ base field multiplications, explaining the high computational cost.

### Stage 6: Hint Verification (Optimized Path Only)

Stage 6 exists only in the optimized verification approach (PR #975). It replaces the expensive 109 $\mathbb{G}_T$ exponentiations in Stage 5 with a hint-based shortcut: the prover provides the precomputed exponentiation results, and Stage 6 verifies these hints are correct using a cheaper proof system.

#### Why Hints?

Each $\mathbb{G}_T$ exponentiation costs approximately 11 million RISC-V cycles (for Jolt verifying its own verifier, enabling recursion). With 109 exponentiations, Stage 5 alone costs ~1.2 billion cycles, dominating total verification cost (~1.5 billion cycles baseline).

The optimization: instead of computing $h_i = g^{s_i}$ for 109 different exponents $s_i$, the verifier accepts $h_i$ values from the prover (as part of the proof) and uses them directly in Dory's equality checks. Stage 5 then only performs $\mathbb{G}_T$ arithmetic (additions and multiplications using the hints), not exponentiations. This reduces Stage 5 cost from ~1.2B cycles to ~2M cycles.

The catch: the verifier must verify the hints are correct. Otherwise, the prover could provide arbitrary $h_i$ values and break soundness. Stage 6 provides this verification.

#### What Gets Verified

Stage 6 proves 109 equations of the form:

$$g^{s_i} = h_i$$

where:
- $g \in \mathbb{G}_T$ is the generator (public, from Dory SRS)
- $s_i \in \mathbb{F}_q$ are exponents (either Fiat-Shamir challenges or RLC coefficients, public from transcript)
- $h_i \in \mathbb{G}_T$ are the hints provided by the prover

Each $h_i$ is a $\mathbb{G}_T$ element, represented as 12 coefficients in $\mathbb{F}_q$ (base field). All 109 hints together: $109 \times 12 = 1,308$ field elements, about 42 KB of additional proof data.

#### Why Grumpkin?

To verify exponentiation equations, the verifier needs to commit to the hint values $h_i$ and prove relationships between them. This requires a polynomial commitment scheme where the scalars (things being committed) are $\mathbb{F}_q$ elements (BN254's base field, since $\mathbb{G}_T$ coefficients live there).

Grumpkin is chosen because of the **2-cycle property** with BN254:
- BN254's base field = Grumpkin's scalar field (exactly)
- Grumpkin's base field ≈ BN254's scalar field (approximately, orders very close)

This means $\mathbb{G}_T$ coefficients (which are $\mathbb{F}_q$ elements for BN254) are native scalars for Grumpkin commitments. When committing to a $\mathbb{G}_T$ element using Grumpkin, the 12 $\mathbb{F}_q$ coefficients can be used directly as scalars in the commitment—no conversion, no limb decomposition. This is critical for efficiency.

If we used BN254 commitments (e.g., Dory again), we'd need to decompose each $\mathbb{F}_q$ coefficient into limbs in BN254's scalar field (a different field), adding complexity and cost. Grumpkin's field matching eliminates this overhead.

#### Hyrax Commitments

Stage 6 uses Hyrax, a polynomial commitment scheme based on multi-scalar multiplications (MSMs) over elliptic curves. Hyrax commits to multilinear polynomials over 4 variables using a 4×4 matrix structure:

**Commitment structure**: To commit to a 4-variate polynomial $P(x_1, x_2, x_3, x_4)$ (16 coefficients), Hyrax represents it as a $4 \times 4$ matrix and computes 4 row commitments:

$$C[j] = \sum_{k=0}^3 m[j,k] \cdot G_{j,k}$$

for $j \in \{0,1,2,3\}$, where $m[j,k]$ are the polynomial coefficients (arranged as a matrix) and $G_{j,k}$ are Grumpkin curve points from a structured reference string.

The commitment is 4 Grumpkin curve points: $C = (C[0], C[1], C[2], C[3])$.

**Batching commitments**: Stage 6 needs to commit to 109 $\mathbb{G}_T$ elements (the hints), each represented as a 4-variate MLE (the 12 $\mathbb{F}_q$ coefficients are grouped as three 4-variate polynomials, but the exact structure varies). This produces 109 commitments, each consisting of 4 Grumpkin points.

To batch these for efficiency, the verifier computes a random linear combination using challenges from the Fiat-Shamir transcript:

$$C_{\text{batch}}[j] = \sum_{i=1}^{109} \gamma_i \cdot C_i[j]$$

for each row position $j \in \{0,1,2,3\}$. This requires **4 multi-scalar multiplications (MSMs)** over Grumpkin—one for each row.

Each MSM has 109 bases (the individual commitments $C_i[j]$) and 109 scalars (the batching coefficients $\gamma_i$). Using Pippenger's algorithm, a 109-base MSM costs significantly less than 109 individual scalar multiplications (roughly 20-40% of the naive cost, depending on implementation).

#### ExpSumcheck Protocol

After committing to the hints, Stage 6 verifies the exponentiation equations using a specialized sum-check protocol called ExpSumcheck. The protocol exploits the binary structure of exponentiation to verify all 109 equations efficiently.

**Binary expansion**: Each exponent $s_i$ is a 254-bit (or 128-bit, depending on context) scalar. Exponentiation $g^{s_i}$ can be computed via binary method:

$$g^{s_i} = g^{\sum_{k=0}^{253} b_k \cdot 2^k} = \prod_{k=0}^{253} (g^{2^k})^{b_k}$$

where $b_k \in \{0,1\}$ are the bits of $s_i$.

ExpSumcheck verifies this bit-by-bit. It represents the computation as a sequence of doubling and conditional multiplication steps, then uses sum-check to verify the sequence is correct for all 109 instances simultaneously (batched together).

**Batching all instances**: The protocol doesn't verify each of the 109 equations separately. Instead, it uses random linear combination to batch them into a single sum-check over a polynomial that encodes all 109 exponentiation computations. The sum-check runs over:
- Bit positions (254 rounds for 254-bit exponents)
- Instance index (log 109 ≈ 7 rounds for 109 instances)

The total number of rounds depends on the exact structure but is roughly $O(\log(109 \cdot 254)) \approx O(\log 2^{15}) = 15$ rounds (exact details vary by implementation).

**Polynomial evaluations**: During ExpSumcheck, the verifier needs to evaluate polynomials representing the hint values $h_i$ and intermediate doubling results. These evaluations are proven using Hyrax's opening proof (see next section).

#### Hyrax Opening Proof

After ExpSumcheck reduces to specific polynomial evaluation claims, the verifier must check these evaluations against the committed hint polynomials. Hyrax provides an efficient opening proof using 2 MSMs.

**Opening protocol**: To prove $P(r_1, r_2, r_3, r_4) = v$ for a committed 4-variate polynomial $P$ at random point $(r_1, r_2, r_3, r_4)$, Hyrax uses a two-round sum-check-like protocol:

1. Prover sends intermediate messages (partial evaluations)
2. Verifier generates random challenges
3. Verifier computes two MSMs: one for row dimension, one for column dimension
4. Verifier checks final equation involving MSM results and claimed value $v$

The two MSMs each have 4 bases (from the 4×4 matrix structure) and 4 scalars (derived from the evaluation point and challenges). These are much smaller than the batching MSMs (4 bases vs. 109 bases), so they're relatively cheap.

**Total MSMs in Stage 6**: 4 (batching) + 2 (opening) = **6 Grumpkin MSMs**.

#### Stage 6 Cost

The total cost of Stage 6 is dominated by:
- ExpSumcheck: Field arithmetic over $\mathbb{F}_q$ to verify binary exponentiation steps, batched for all 109 instances
- Grumpkin MSMs: 6 MSMs, with the 4 batching MSMs being most expensive (109 bases each)

Grumpkin scalar multiplication (one elliptic curve point scalar mult) costs roughly 500K RISC-V cycles. A 109-base MSM using Pippenger costs approximately 30-40 scalar multiplications' worth of work (Pippenger sublinear scaling), so ~15-20M cycles per MSM. Four batching MSMs: ~60-80M cycles. Two opening MSMs: ~4M cycles. ExpSumcheck: ~240M cycles (field arithmetic for 109-instance batching).

Total Stage 6: ~300-400M cycles. Combined with optimized Stage 5 (~2M cycles), the total is ~300-400M cycles, compared to baseline Stage 5's ~1.2B cycles. Net savings: ~800-900M cycles, reducing total verification from ~1.5B to ~600-700M cycles.

## Cryptographic Primitives: Deep Dive

### Sum-Check Protocol

Sum-check is the core verification primitive in Jolt, used in almost every component. Understanding it in detail is essential.

#### The Basic Protocol

**Claim**: Prover wants to convince verifier that:

$$H = \sum_{x \in \{0,1\}^n} g(x)$$

for some $n$-variate polynomial $g$ over field $\mathbb{F}$.

**Protocol**:
- **Round 1**: Prover sends univariate polynomial $g_1(X_1) = \sum_{x_2, \ldots, x_n \in \{0,1\}^{n-1}} g(X_1, x_2, \ldots, x_n)$. Verifier checks $g_1(0) + g_1(1) = H$. Verifier samples random $r_1 \in \mathbb{F}$ from transcript, sets $H_1 = g_1(r_1)$.
- **Round 2**: Prover sends $g_2(X_2) = \sum_{x_3, \ldots, x_n \in \{0,1\}^{n-2}} g(r_1, X_2, x_3, \ldots, x_n)$. Verifier checks $g_2(0) + g_2(1) = H_1$. Samples $r_2$, sets $H_2 = g_2(r_2)$.
- **Rounds 3 to $n$**: Continue similarly, binding one variable per round.
- **Final check**: After $n$ rounds, verifier has evaluation claim $g(r_1, \ldots, r_n) = H_n$. This claim is either checked directly (if $g$ is simple to evaluate) or passed as input to another sum-check (if $g$ is a virtual polynomial).

**Soundness**: If the prover cheats in round $i$ by sending $g_i' \neq g_i$ (the correct polynomial), the verifier will catch it with high probability. The random challenge $r_i$ is sampled after $g_i'$ is committed (via Fiat-Shamir). Two distinct degree-$d$ univariate polynomials agree on at most $d$ points. If $|\mathbb{F}| \gg d$, the probability $g_i'(r_i) = g_i(r_i)$ is at most $d/|\mathbb{F}|$ (Schwartz-Zippel). For cryptographic fields ($|\mathbb{F}| \approx 2^{254}$) and low-degree polynomials ($d \leq 10$ in most Jolt sum-checks), this probability is negligible.

**Completeness**: If the prover is honest, all checks pass by construction.

#### Batching Sum-Checks

When multiple independent sum-check instances run in the same stage, they are batched to reduce rounds.

**Multiple claims**: Suppose the verifier needs to check:

$$H_1 = \sum_{x \in \{0,1\}^n} g_1(x), \quad H_2 = \sum_{x \in \{0,1\}^n} g_2(x), \quad \ldots, \quad H_k = \sum_{x \in \{0,1\}^n} g_k(x)$$

**Batched protocol**: Verifier samples random $\gamma_1, \ldots, \gamma_k$ from transcript, defines:

$$H_{\text{batch}} = \sum_{i=1}^k \gamma_i H_i, \quad g_{\text{batch}}(x) = \sum_{i=1}^k \gamma_i g_i(x)$$

Then runs a single sum-check for $H_{\text{batch}} = \sum_{x \in \{0,1\}^n} g_{\text{batch}}(x)$.

**Soundness**: If the prover cheats on any individual $g_i$ (claims $H_i'$ when true sum is $H_i \neq H_i'$), the batched claim will be wrong with overwhelming probability. The random linear combination ensures that $\sum \gamma_i H_i' \neq \sum \gamma_i H_i$ except with probability $\approx 1/|\mathbb{F}|$ (if one coefficient $\gamma_i$ is random, the sums differ).

**Efficiency**: Batching reduces from $k \cdot n$ rounds (running $k$ separate sum-checks) to $n$ rounds (one batched sum-check). The prover sends one univariate per round instead of $k$ univariates. Verifier sends one challenge per round instead of $k$ challenges.

#### Sum-Check Over Products

Many Jolt sum-checks involve products of multilinear polynomials:

$$H = \sum_{x \in \{0,1\}^n} f_1(x) \cdot f_2(x) \cdot f_3(x) \cdots f_m(x)$$

**Prover's work**: In round $i$, prover sends:

$$g_i(X_i) = \sum_{x_{i+1}, \ldots, x_n \in \{0,1\}^{n-i}} f_1(r_1, \ldots, r_{i-1}, X_i, x_{i+1}, \ldots, x_n) \cdot f_2(\cdots) \cdots f_m(\cdots)$$

Computing this naively requires summing over $2^{n-i}$ points for each value of $X_i \in \{0,1\}$ (or more, for the full univariate). If $m$ is large or $f_j$ are expensive to evaluate, this is costly.

**Optimization - memoization**: Many Jolt sum-checks exploit structure to avoid redundant computation. For example, if $f_1$ is sparse (most values are zero), the prover only sums over non-zero terms. If $f_2$ has special structure (e.g., eq polynomial), it can be evaluated efficiently.

**Optimization - small field operations**: If some $f_j$ take small values (e.g., boolean flags), multiplications are cheaper than generic field operations.

#### Sum-Check for Multilinear Extensions

Jolt polynomials are multilinear extensions (MLEs) of execution trace data. An MLE $\tilde{f}$ of a function $f: \{0,1\}^n \to \mathbb{F}$ is the unique multilinear polynomial agreeing with $f$ on the Boolean hypercube.

**Lagrange basis representation**: The MLE can be written as:

$$\tilde{f}(x) = \sum_{b \in \{0,1\}^n} f(b) \cdot \text{eq}(x, b)$$

where $\text{eq}(x, b) = \prod_{i=1}^n (x_i b_i + (1 - x_i)(1 - b_i))$ is the multilinear extension of the equality function (1 if $x = b$, 0 otherwise).

**Sum-check evaluation**: When the sum-check reduces to an evaluation claim $\tilde{f}(r) = v$, the verifier can either:
- Check directly if $\tilde{f}$ is simple (e.g., bytecode polynomial committed during preprocessing)
- Verify via another sum-check if $\tilde{f}$ is a virtual polynomial (e.g., intermediate witness values)

**Efficient MLE evaluation**: Many Jolt MLEs have closed-form expressions. For example, the MLE of a register read-address polynomial can be written as:

$$\tilde{ra}(cycle, reg) = \sum_{i=1}^T \text{eq}(cycle, i) \cdot \text{onehot}(reg, rs1[i])$$

where $rs1[i]$ is the source register for cycle $i$ (from bytecode), and onehot is the one-hot encoding. This can be evaluated in $O(T)$ time by summing over cycles, much faster than naively summing over all $2^{\log T + \log 32}$ points.

### Twist Memory Checking Protocol

Twist proves memory consistency: reads return the most recent writes (or initial values). It generalizes to arbitrary read/write patterns with multiple operations per cycle.

#### The Grand Product Argument

**Permutation check**: To prove two sequences $A = (a_1, \ldots, a_n)$ and $B = (b_1, \ldots, b_n)$ are permutations of each other, it suffices to show:

$$\prod_{i=1}^n \phi(a_i) = \prod_{i=1}^n \phi(b_i)$$

for a random fingerprint function $\phi$. If $A$ and $B$ are not permutations (some element appears different number of times), the fingerprints differ with overwhelming probability.

**Fingerprint function**: For memory tuples $(address, timestamp, value)$, Twist uses:

$$\phi(a, t, v) = \alpha \cdot a + \beta \cdot t + \gamma \cdot v$$

where $\alpha, \beta, \gamma$ are random challenges from Fiat-Shamir. This is a linear fingerprint, simple to compute and analyze.

**Why random works**: By Schwartz-Zippel, if two multisets differ, their fingerprint products differ with probability $\geq 1 - 1/|\mathbb{F}|$. For cryptographic fields, this is negligible.

#### Time-Ordered vs. Address-Ordered Traces

**Time-ordered**: Memory accesses in the order they occur during execution:

$$(a_1, t_1, v_1), (a_2, t_2, v_2), \ldots, (a_T, t_T, v_T)$$

where $t_i = i$ (timestamp is just the index).

**Address-ordered**: Same accesses sorted by address, then timestamp:

$$(a'_1, t'_1, v'_1), \ldots, (a'_T, t'_T, v'_T)$$

where $a'_1 \leq a'_2 \leq \cdots \leq a'_T$, and within each address, timestamps are increasing.

**Memory consistency**: For each address $a$, if we look at all accesses to $a$ in the address-ordered trace, they should alternate correctly between writes and reads: a read should return the value from the most recent write (or initial value if no prior write).

Twist proves this by showing time-ordered is a permutation of address-ordered. If a read in time-ordered returns the wrong value, the tuples won't match, breaking the permutation property.

#### Sum-Check Decomposition

Proving $\prod_{i=1}^T \phi(a_i, t_i, v_i) = \prod_{i=1}^T \phi(a'_i, t'_i, v'_i)$ directly is expensive (product of $T$ terms). Twist uses logarithms to convert products to sums:

$$\sum_{i=1}^T \log \phi(a_i, t_i, v_i) = \sum_{i=1}^T \log \phi(a'_i, t'_i, v'_i)$$

Then sum-check verifies each sum. But logarithms are expensive in finite fields. Twist avoids explicit logarithms by working in the exponent:

Define $g = \text{generator of } \mathbb{F}^*$ (a primitive root). Then:

$$\prod_{i=1}^T \phi(a_i, t_i, v_i) = g^{\sum_i \log_g \phi(a_i, t_i, v_i)}$$

The sum-check verifies $\sum_i \log_g \phi(\cdots)$ by checking the exponent, not the logarithm directly. This is done via sum-check over a polynomial that encodes the cumulative product.

**Three sum-checks**:
1. **Read-checking**: Verifies time-ordered trace sum
2. **Write-checking**: Verifies address-ordered trace sum
3. **Evaluation**: Proves final memory state consistent with initial state plus updates

#### Parameters $K$ and $d$

**$K$ (memory size)**: Number of addressable cells. For RAM, this depends on the guest program's memory usage (stack, heap, globals). For registers, $K = 64$ (32 architectural + 32 virtual).

**$d$ (chunking)**: Memory addresses are $\log K$ bits. To keep polynomial sizes bounded, addresses are chunked into $d$ pieces of $(\log K)/d$ bits each. The polynomial commitment represents memory as a $d$-dimensional tensor.

Larger $d$ means smaller polynomials (good for commitment efficiency) but more sum-check rounds (bad for prover/verifier time). Jolt chooses $d$ dynamically: for RAM, $d$ is chosen so $K^{1/d} \approx 2^8$ (256 entries per dimension).

Example: 16MB RAM = $K = 2^{21}$ doublewords (memory is doubleword-addressed). Choose $d = 3$: $K^{1/3} = 2^7 = 128$. Each dimension has 128 entries, commitment is manageable, and sum-check is 3-round instead of 1-round.

For registers, $K = 64 = 2^6$ is small, so $d = 1$ (no chunking) is fine.

### Shout Lookup Argument

Shout proves that $m$ witness lookups all appear in a table of size $N$, where typically $m \ll N$ but $N$ can be enormous ($2^{128}$).

#### Prefix-Suffix Decomposition

The key idea: represent the lookup table's MLE in a factored form that avoids materializing the full table.

**Decomposition theorem**: For a lookup table $T: \{0,1\}^n \to \mathbb{F}$, split the $n$-bit index into prefix (high $n_p$ bits) and suffix (low $n_s$ bits, $n_p + n_s = n$). If the table has structure, its MLE can be written:

$$\tilde{T}(k_p, k_s) = \sum_j \tilde{P}_j(k_p) \cdot \tilde{S}_j(k_s)$$

where $\tilde{P}_j, \tilde{S}_j$ are efficiently computable functions.

Example: 64-bit XOR table. Index is two 64-bit operands: $k = (x[0..63], y[0..63])$ (128 bits). Split each operand into chunks: $x = (x_{\text{hi}}, x_{\text{lo}})$ where $x_{\text{hi}}$ is high 64 bits of first operand... wait, that doesn't make sense. Let me clarify.

Actually, for XOR: the index is $(x, y)$ where $x, y \in \{0,1\}^{64}$. Total index space: 128 bits. XOR output: $z = x \oplus y$ (64 bits). The table $T(x, y) = x \oplus y$.

Decompose: Split $x$ and $y$ into 4-bit chunks. $x = (x_0, x_1, \ldots, x_{15})$ where each $x_i \in \{0,1\}^4$. Similarly $y = (y_0, \ldots, y_{15})$. Then:

$$x \oplus y = (x_0 \oplus y_0, x_1 \oplus y_1, \ldots, x_{15} \oplus y_{15})$$

Each $x_i \oplus y_i$ is independent, a 4-bit XOR (256-entry table). The MLE of the full table factors as:

$$\tilde{T}(x, y) = \sum_{i=0}^{15} 2^{4i} \cdot \tilde{XOR}_4(x_i, y_i)$$

where $\tilde{XOR}_4$ is the MLE of the 4-bit XOR table. This factors into prefix (high chunks) and suffix (low chunks):

$$\tilde{T}(k_p, k_s) = \tilde{P}(k_p) + \tilde{S}(k_s)$$

where $k_p$ represents $(x_{\text{high}}, y_{\text{high}})$ and $k_s$ represents $(x_{\text{low}}, y_{\text{low}})$.

Actually, the exact factorization depends on the table structure. For XOR, it's additive. For other operations (AND, OR), it may be multiplicative or a combination. The key: $\tilde{P}$ and $\tilde{S}$ can be evaluated efficiently (polylog time, not exponential).

#### The Shout Protocol (High-Level)

**Claim**: All $m$ witness lookups $(L_1, \ldots, L_m)$ appear in table $T$ with correct values $(V_1, \ldots, V_m)$, i.e., $T[L_i] = V_i$ for all $i$.

**Read-checking sum-check**: Proves the witness lookups are well-formed. This involves a sum-check over the witness MLE $\tilde{W}$ that encodes the lookup indices and values:

$$\sum_{i \in \{0,1\}^{\log m}} \tilde{W}(i) \cdot (\text{lookup check polynomial})$$

The lookup check polynomial verifies that $\tilde{W}(i)$ claims a lookup at index $L_i$ with value $V_i$, and that $T[L_i] = V_i$.

**Write-checking sum-check**: Proves the table entries are correctly defined. This sum-check verifies:

$$\sum_{k \in \{0,1\}^n} \tilde{T}(k) \cdot (\text{table check polynomial})$$

Together, read-checking and write-checking establish the subset property: witness lookups are a subset of table entries.

**Prefix-suffix handling**: Both sum-checks use prefix-suffix factorization to avoid summing over all $2^n$ table entries. The sum-check proceeds in phases, alternating between binding prefix and suffix variables.

#### Prefix-Suffix Sum-Check (Detailed)

For a sum-check over factored polynomial:

$$H = \sum_{k_p \in \{0,1\}^{n_p}} \sum_{k_s \in \{0,1\}^{n_s}} \tilde{P}(k_p) \cdot \tilde{S}(k_s)$$

**Phase structure**: Alternate between prefix and suffix. For example, with $n_p = n_s = 64$ (128 bits total, split evenly):

- **Phase 1** (bind 16 prefix bits): Reduce to sum over 48 prefix bits and 64 suffix bits
- **Phase 2** (bind 16 suffix bits): Reduce to sum over 48 prefix bits and 48 suffix bits
- **Phase 3** (bind 16 prefix bits): Reduce to sum over 32 prefix bits and 48 suffix bits
- ...
- **Phase 8** (bind last 16 bits): Reduce to evaluation claim

Each phase: 16 rounds, prover sends univariate polynomials, verifier sends challenges.

**Why alternating?**: Maintains the factored structure. If we bound all prefix bits first, the remaining sum $\sum_{k_s} \tilde{S}(k_s, r_p)$ might not have a nice structure (where $r_p$ is the random prefix point). Alternating keeps the factorization throughout.

**Efficiency**: Prover evaluates $\tilde{P}$ and $\tilde{S}$ at random points (one evaluation per round). For Jolt's instruction tables, these evaluations are $O(1)$ time (closed-form formulas), not $O(2^{64})$ table lookups.

#### Efficiently-Evaluable MLEs

The core requirement for Shout: $\tilde{T}$ (or $\tilde{P}, \tilde{S}$ in factored form) must be efficiently evaluable.

**Example - 4-bit ADD table**: Table has $2^8 = 256$ entries: $T[x, y] = (x + y) \mod 16$ for $x, y \in \{0,1\}^4$. The MLE is:

$$\tilde{ADD}_4(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4) = \sum_{a,b \in \{0,1\}^4} ((a + b) \mod 16) \cdot \text{eq}((x_1, \ldots, x_4), a) \cdot \text{eq}((y_1, \ldots, y_4), b)$$

This looks like a sum over 256 terms. But it can be rewritten in closed form using properties of the eq polynomial and modular arithmetic, evaluable in $O(1)$ time.

For 64-bit ADD, decompose into 16 4-bit ADDs with carry chain. The MLE for 64-bit ADD factors into:
- 16 instances of $\tilde{ADD}_4$ for each 4-bit chunk
- Carry propagation polynomial (verifies carries are correct)

The carry polynomial can also be written efficiently. Total evaluation: $O(16) = O(1)$ for 64-bit ADD.

**Example - 4-bit XOR table**: $\tilde{XOR}_4(x_1, \ldots, x_4, y_1, \ldots, y_4) = \prod_{i=1}^4 (x_i + y_i - 2 x_i y_i)$ (XOR is addition mod 2, the multilinear representation). Evaluable in $O(4) = O(1)$ time.

Each Jolt instruction implements `JoltLookupTable` trait with an efficient evaluation function. As long as evaluation is polylog, Shout is tractable even for $2^{128}$-size tables.

### Dory Polynomial Commitment Scheme

Dory is a transparent, logarithmic-time PCS for multilinear polynomials, based on discrete log assumption over bilinear groups.

#### Commitment

**Setup**: Generate structured reference string (SRS) for degree bound $N = 2^n$. SRS contains $O(N)$ group elements in $\mathbb{G}_1$, $\mathbb{G}_2$, and $\mathbb{G}_T$ (BN254 pairing groups).

**Commit**: To commit to multilinear polynomial $P$ over $n$ variables, represent $P$ as a $\sqrt{N} \times \sqrt{N}$ matrix (lexicographic ordering of evaluations over $\{0,1\}^n$). Compute commitment using inner products with SRS elements, resulting in a $\mathbb{G}_T$ element.

#### Opening Proof

**Claim**: Prove $P(r_1, \ldots, r_n) = v$ for committed $P$ at point $r = (r_1, \ldots, r_n)$.

**Folding protocol**: Dory uses $\nu = n/2$ rounds of recursive folding (for $n = 2\nu$ variables). Each round:

1. Prover sends folding messages: commitments to partially-folded polynomials
2. Verifier generates challenge $\alpha$ from transcript
3. Both parties compute folded polynomial: $P_{\text{fold}}(x_1, \ldots, x_{n-2}) = P(x_1, \ldots, x_{n-2}, \alpha, f(\alpha))$ where $f$ is derived from the structure
4. Verifier updates commitment: $C_{\text{fold}} = C_L^{\alpha} \cdot C_R \cdot (\text{prover messages})^{g(\alpha)}$ using $\mathbb{G}_T$ exponentiations

After $\nu$ rounds, the polynomial is fully folded to a constant, and verification reduces to a pairing check.

**Pairing check**: The base case verifies:

$$e(A_1, B_2) = C_{\text{final}} \cdot e(X_1, Y_2)^v$$

where $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ is the pairing, $A_1, X_1 \in \mathbb{G}_1$, $B_2, Y_2 \in \mathbb{G}_2$, $C_{\text{final}} \in \mathbb{G}_T$ (fully folded commitment), and $v$ is the claimed evaluation. This requires 5 pairing computations (some terms may be cached).

#### Batched Opening

**Multiple openings**: To verify $P_1(r_1) = v_1, \ldots, P_k(r_k) = v_k$, use random linear combination:

$$P_{\text{batch}} = \sum_{i=1}^k \gamma_i P_i, \quad C_{\text{batch}} = \prod_{i=1}^k C_i^{\gamma_i}, \quad v_{\text{batch}} = \sum_{i=1}^k \gamma_i v_i$$

Verify single opening $P_{\text{batch}}(r_{\text{batch}}) = v_{\text{batch}}$ (handling multi-point batching via virtual polynomial techniques).

**RLC cost**: Computing $C_{\text{batch}}$ requires $k$ $\mathbb{G}_T$ exponentiations (one for each $C_i^{\gamma_i}$), then $k-1$ $\mathbb{G}_T$ multiplications to combine them.

## Fiat-Shamir Transform

All Jolt protocols are made non-interactive using the Fiat-Shamir heuristic.

#### The Transform

**Interactive protocol**: In an interactive sum-check, verifier sends random challenges after each prover message.

**Non-interactive version**: Prover computes challenges deterministically by hashing the transcript:

$$r_i = H(\text{transcript} \| \text{round number})$$

where $H$ is a cryptographic hash function (Blake2b-256 in Jolt).

**Transcript management**: The transcript is a running state (256-bit hash) updated after each message:
- **Prover message**: Append polynomial coefficients (or commitment) to transcript
- **Verifier challenge**: Hash transcript to derive challenge, append challenge to transcript

This ensures challenges depend on all prior messages, preventing adaptive attacks.

#### Security

**Random oracle model**: Fiat-Shamir is provably secure if $H$ is modeled as a random oracle (outputs uniformly random, independent for different inputs).

**Concrete security**: In practice, using a secure hash function like Blake2b or SHA-256 provides strong security. Attacks require breaking the hash function's collision resistance or preimage resistance (computationally infeasible for 256-bit hashes).

**Soundness preservation**: If the interactive protocol has soundness error $\epsilon$ (probability a cheating prover succeeds), the non-interactive version has soundness error $\approx \epsilon + \text{negligible}$ (with negligible degradation from hash security).

## Putting It All Together: Example Verification Flow

Let's trace a simplified verification for a guest program that computes Fibonacci(50).

### Preprocessing

**Compile and commit**: Host compiles the guest program to RISC-V, decodes bytecode, commits to bytecode using Dory. Generates prover and verifier preprocessing (SRS, constraint matrices, etc.).

### Proving

**Execute and trace**: Prover runs the RISC-V emulator, generating execution trace: register reads/writes, memory accesses, instruction operands, etc. For Fibonacci(50), this might be ~1000 cycles.

**Commit to trace**: Prover commits to witness polynomials (register values, memory contents, instruction operands) using Dory.

**Generate proof**: Prover runs the DAG, computing sum-checks for each component in stages 1-4, generating Dory opening proof in stage 5 (or hint proof in stage 6 if optimized).

### Verification

**Stage 1**: Verifier processes initial sum-checks for R1CS (Spartan outer), RAM (Twist read-check), registers (Twist read-checks), instruction execution (Shout read-check), and bytecode (Shout read-check). All batched together. Verifier sends challenges, prover responds with univariates, verifier checks consistency. End of stage 1: polynomial evaluation claims.

**Stage 2**: Verifier continues with dependent sum-checks: Spartan product, Twist write-checks, Shout middle phases. Uses evaluation claims from stage 1 as inputs. Generates more evaluation claims.

**Stage 3**: Verifier completes Spartan matrix evaluation, Twist evaluation checks, Shout completion. Final evaluation claims accumulated.

**Stage 4**: Verifier checks R1CS linking constraints, ensuring RAM reads match register writes, instruction outputs match register writes, PC updates correctly.

**Stage 5 (baseline)**: Verifier batches all committed polynomial opening claims using RLC (29 $\mathbb{G}_T$ exponentiations), runs Dory verification protocol (8 folding rounds, 80 exponentiations), performs final pairing check (5 pairings). Total: 109 exponentiations + 5 pairings.

**Stage 5 (optimized)**: Verifier accepts hint values for the 109 exponentiations, uses them in Dory checks (cheap $\mathbb{G}_T$ arithmetic), proceeds to stage 6.

**Stage 6 (optimized only)**: Verifier batches Hyrax commitments to hints (4 Grumpkin MSMs), runs ExpSumcheck to verify exponentiation equations (field arithmetic), performs Hyrax opening proof (2 Grumpkin MSMs).

**Acceptance**: If all checks pass, verifier accepts: Fibonacci(50) = [output value] (or program panicked correctly). If any check fails, verifier rejects.

### Security Guarantee

If the verifier accepts, one of the following is true (with overwhelming probability):
1. The claimed output is correct (program executed correctly on given input)
2. The prover broke the discrete log assumption on BN254 (broke Dory binding)
3. The prover found a collision in Blake2b (broke Fiat-Shamir)
4. The prover got lucky with Schwartz-Zippel (probability $\approx 2^{-100}$ for typical parameters)

For well-chosen cryptographic parameters, option 1 is the only computationally feasible outcome. This provides computational soundness.

## Summary

Jolt's verifier combines sum-check protocols, Twist memory checking, Shout lookup arguments, and Dory polynomial commitments into a structured DAG of verification steps. The five core VM components (R1CS, RAM, registers, instruction execution, bytecode) each contribute sum-checks at different stages, with polynomial evaluation claims flowing through the DAG as virtual (verified by later sum-checks) or committed (verified by batched opening proof).

The verification algorithm is:
1. **Stages 1-4**: Process component sum-checks in dependency order, accumulating committed polynomial evaluation claims
2. **Stage 5**: Verify all accumulated claims via batched Dory opening proof
3. **Stage 6** (optional): Verify hints used to optimize stage 5

The soundness relies on:
- Schwartz-Zippel lemma (sum-check soundness)
- Discrete log assumption (Dory binding)
- Fiat-Shamir random oracle model (non-interactive security)

The efficiency comes from:
- Lookup-centric architecture (30 R1CS constraints per cycle vs. 100+)
- Prefix-suffix decomposition (massive tables tractable)
- Structured polynomial handling (uniform constraints, sparse matrices)
- Batching (sum-checks and openings combined to reduce rounds)

This design achieves logarithmic verification time and proof size with linear prover time, making Jolt one of the fastest zkVMs for CPU-based proving.

## Future Improvements

The hint-based optimization introduced in PR #975 (Stage 6 for verifying $\mathbb{G}_T$ exponentiations) represents the first application of a general strategy for reducing verification costs. Several planned improvements extend this approach:

### Extended Hint Mechanism

The same pattern used for $\mathbb{G}_T$ exponentiations will be applied to other expensive verifier operations:

**GT multiplications**: While less expensive than exponentiations, $\mathbb{G}_T$ multiplications in the extension field $\mathbb{F}_{q^{12}}$ still cost roughly 50-100K cycles each. Hundreds of these operations occur in Dory verification. The prover can provide hints for multiplication results, with correctness verified via sum-check over the tower field structure.

**Pairing operations**: The 5 BN254 pairings in Stage 5's final check each cost approximately 20M cycles. Pairing hints would allow the verifier to accept precomputed pairing results, with verification reduced to checking the pairing equation structure using field arithmetic. This requires careful handling of the bilinear property to maintain soundness.

**Other field operations**: Any repeated expensive operation (e.g., $\mathbb{F}_{q^{12}}$ inversions, square roots) becomes a candidate for hints. The general pattern: prover provides result, separate proof verifies correctness using cheaper operations.

**Expected impact**: Extending hints to all expensive operations in Stage 5 could reduce verification from the current ~640M cycles (with only exponentiation hints) to potentially ~200-300M cycles, a further 2-3× improvement.

### Lattice-Based PCS Integration

Dory's reliance on pairing-based cryptography creates inherent verification costs (pairings, extension field arithmetic). Future versions plan to integrate lattice-based polynomial commitment schemes:

**Motivation**:
- **Post-quantum security**: Lattice assumptions resist quantum attacks, unlike discrete log
- **Native arithmetic**: Lattice schemes use simpler arithmetic (polynomial rings over modest fields, not degree-12 extensions)
- **Potential efficiency**: Lattice verification often involves simple polynomial evaluation and inner products, no pairings

**Integration approach**:
- Dory will remain as an option but lattice PCS will be added as an alternative backend
- Sum-check verification of lattice error polynomials and ring operations
- May apply similar hint mechanism: prover provides intermediate lattice samples, verifier checks using sum-check

**Status**: Work in progress. Exact lattice scheme (FRI-based, Brakedown-style, or other) not yet finalized. Implementation details will evolve as the team explores the design space.

### Recursive Composition

The ultimate goal is enabling efficient recursive proof composition: using Jolt to verify its own verifier (or another Jolt proof), creating proof chains or aggregation.

**Current status**: With PR #975 optimizations, Jolt verification costs ~640M cycles, still too expensive for practical recursion (proving 640M cycles would take significant time and memory).

**Recursive target**: Further optimizations (extended hints, lattice PCS) aim to reduce verification to ~200-300M cycles, making recursion feasible. At this level:
- Proof aggregation: Combine multiple program executions into single proof
- Incrementally-verifiable computation: Long computations split across multiple proofs, each verifying the previous
- Blockchain applications: zkVM proofs small enough for on-chain verification or zkRollups

**Technical requirements**:
- Verifier must be expressible as a guest program (already achieved—verifier is pure Rust)
- Verification cost must be low enough that proving it is practical (targeted by optimizations)
- Cycle curves (BN254/Grumpkin 2-cycle) enable efficient arithmetic for cross-curve verification

### Verification Optimization Pattern

The emerging pattern from these improvements:

1. **Identify expensive operation** (GT exponentiation, pairing, etc.)
2. **Replace with hint**: Prover provides precomputed result
3. **Verify hint correctness**: Use cheaper proof system (sum-check, auxiliary SNARK)
4. **Choose verification mechanism**: Grumpkin for BN254 operations, sum-check for field structure

This pattern is universal: any expensive deterministic computation can potentially be hinted and verified separately, trading computation for proof size and verification structure.

**Key insight**: Sum-check acts as the universal verification primitive. Whether verifying error polynomials (lattice), ring operations, or exponentiation equations, sum-check provides a cheap, structured way to verify correctness of hints without recomputing the expensive operation directly.

### Timeline and Experimental Status

**Current state (v0.2.0 with PR #975)**:
- Baseline verification: ~1.5B cycles
- With GT exponentiation hints: ~640M cycles
- Status: Experimental, under active development

**Near-term (extended hints)**:
- Target: ~200-300M cycles with hints for all expensive operations
- Timeline: Active research and development

**Medium-term (lattice PCS)**:
- Target: Post-quantum security + potential further cost reduction
- Timeline: Design exploration phase, implementation details evolving

**Long-term (recursion)**:
- Target: Practical proof composition and aggregation
- Depends on verification costs reaching ~200-300M cycle range

These improvements maintain Jolt's core architecture (lookup-centric, five components, DAG structure) while progressively reducing verification overhead through strategic use of hints and proof system composition.
