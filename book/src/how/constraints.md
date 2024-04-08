# Constraint System 

The constraint system is used to enforce 
consistency between the proofs for the different modules of Jolt, bytecode memory-checking, 
RAM checking, and lookups. 

## Structure

The R1CS structure used is very uniform which effectively means that the full constraint system 
for a program is obtained by repeating the small constraint system defined for a single CPU step 
for each step of the entire program. 
A single step of Jolt involves fewer than 70 constraints and about 80 variables. 

## Input Variables and constraints

The inputs required for the constraint system are: 

Pertaining to program ocde: 
* Input and output program counters (PCs): this is the only state passed between CPU steps
* Program code read address: the address in the program code read at this step 
* A 5-tuple representation of the instruction: (`opcode_flags_packed`, `rs1`, `rs2`, `rd`, `imm`)

Pertaining to memory: 
* The (starting) memory address read by the instruction: if the instruction is not a load/store, this is 0
* The bytes written to or read from memory

Pertaining to the lookup: 
* The chunks of the instruction's operands `x` and `y`
* The chunks of the lookup query 
* The lookup output 

Circuit flags: 
* There are 9 circuit flags just for the R1CS used to guide the operands. They are as follows: 
    - `operand_x_flag`: 0 if the first operand is the value in `rs1` or the `PC` 
    - `operand_y_flag`: 0 if the second operand is the value in `rs2` or the `imm` 
    - `is_load_instr`
    - `is_store_instr`
    - `is_jump_instr`
    - `is_branch_instr`
    - `if_update_rd_with_lookup_output`: 1 if the lookup output is to be stored in `rd` at the end of the step
    - `sign_imm_flag`: used in load/store and branch instructions where the instruction is added as constraints 
    - `is_concat`: indicates whether the instruction performs a concat-type lookup
* Instruction flags: these are the unary bits used to indicate which lookup subtable is queried by this instruction. There are as many per step as the number of unique subtables in Jolt, which is 19. 

#### Constraint system 

The constraints for a CPU step are detailed in the `get_jolt_matrices()` function in the `r1cs/constraints` module. 

### Reusing commitments 

SNARK backends used to prove consistency of constraints generally take their inputs as commitments. 
A catch (and an optimization feature) in Jolt is that most of the inputs to the constraint system 
are also used as inputs to proofs in the other modules: for example, 
the address and values pertaining to the bytecode are used in the bytecode memory-checking proof, 
and the lookup chunks, output and flags are used in the lookup proof. 
For Jolt to be sound, it must be ensured that the same inputs must be fed to both sides of the proofs. 
We do this by re-using the same commitments to both proof systems. 
This can be seen in the `format_commitments()` function in the `r1cs/snark` module. 
The proving backend used (Spartan) is adapted to take pre-committed witness variables. 

## Exploiting uniformity 

The uniformity of the constraint system allows us to heavily optimize both the prover and verifier. 
The main changes involved in making this happen are: 
- Spartan is modified to only take in the constraint matrices a single step, and the total number of steps. Using this, the prover and verifier can efficient calculate the multilinear extensions of the full R1CS matrices. 
- Commitment format of the witness values: all `NUM_STEPS` copies of a variable corresponding to each time step is committed together. This affects nearly all variable in Jolt and the 
- Constraints are used to enforce consistency of the state transferred between CPU steps. 

These changes and their impact on the code are visible in the `r1cs/spartan` module. 
