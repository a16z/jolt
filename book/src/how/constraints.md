# Constraint System 

The constraint system is used to enforce certain rules of the RISC-V fetch-decode-execute loop 
and to ensure 
consistency between the proofs for the different modules of Jolt (memory-checking 
and instruction lookups). 

## Uniformity

The R1CS constraint system of Jolt is uniform, which means
that the constraint system for an entire program is just repeated copies of the constraint 
system for a single CPU step. 
Each step is conceptually simply and involves just around 60 constraints and about 80 variables. 

## Input Variables and constraints

The inputs required for the constraint system for a single CPU step are: 

#### Pertaining to program code
* Input and output program counters (PCs): this is the only state passed between CPU steps. 
* Program code read address: the address in the program code read at this step. 
* A 5-tuple representation of the instruction: (`opcode_flags_packed`, `rs1`, `rs2`, `rd`, `imm`). 

#### Pertaining to memory
* The (starting) memory address read by the instruction: if the instruction is not a load/store, this is 0.
* The bytes written to or read from memory.

####  Pertaining to the lookup 
* The chunks of the instruction's operands `x` and `y`. 
* The chunks of the lookup query. 
* The lookup output. 

### Circuit and Instruction flags: 
* There are nine circuit flags used to guide the constraints and are dependent only on the opcode of the instruction. These are thus stored are part of the program code in Jolt. 
    1. `operand_x_flag`: 0 if the first operand is the value in `rs1` or the `PC`. 
    2. `operand_y_flag`: 0 if the second operand is the value in `rs2` or the `imm`. 
    3. `is_load_instr`
    4. `is_store_instr`
    5. `is_jump_instr`
    6. `is_branch_instr`
    7. `if_update_rd_with_lookup_output`: 1 if the lookup output is to be stored in `rd` at the end of the step. 
    8. `sign_imm_flag`: used in load/store and branch instructions where the instruction is added as constraints. 
    9. `is_concat`: indicates whether the instruction performs a concat-type lookup. 
* Instruction flags: these are the unary bits used to indicate which lookup subtable is queried by this instruction. There are as many per step as the number of unique subtables in Jolt, which is 19. 

#### Constraint system 

The constraints for a CPU step are detailed in the `get_jolt_matrices()` function in the `r1cs/constraints` module. 

### Reusing commitments 

As with most SNARK backends, Spartan requires computing a commitment to the inputs 
to the constraint system. 
A catch (and an optimization feature) in Jolt is that most of the inputs 
are also used as inputs to proofs in the other modules. For example, 
the address and values pertaining to the bytecode are used in the bytecode memory-checking proof, 
and the lookup chunks, output and flags are used in the instruction lookup proof. 
For Jolt to be sound, it must be ensured that the same inputs are fed to all relevant proofs. 
We do this by re-using the commitments themselves. 
This can be seen in the `format_commitments()` function in the `r1cs/snark` module. 
The proving backend used (Spartan) is adapted to take pre-committed witness variables. 

## Exploiting uniformity 

The uniformity of the constraint system allows us to heavily optimize both the prover and verifier. 
The main changes involved in making this happen are: 
- Spartan is modified to only take in the constraint matrices a single step, and the total number of steps. Using this, the prover and verifier can efficient calculate the multilinear extensions of the full R1CS matrices. 
- The commitment format of the witness values is change to reflect uniformity. All versions of a variable corresponding to each time step is committed together. This affects nearly all variable committed to in Jolt. 
- The inputs and witnesses are provided to the constraint system as segments. 
- Additional constraints are used to enforce consistency of the state transferred between CPU steps. 

These changes and their impact on the code are visible in the `r1cs/spartan` module. 
