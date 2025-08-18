## Jolt DAG Sumcheck Specification (initial draft)

This file catalogs the DAG-orchestrated sumchecks with precise statements: claim, integrand, degree, rounds, binding/endian, inputs/outputs, and openings used/produced. Sections mirror DAG stages.

### Stage 1
- Spartan Outer (SumcheckId: SpartanOuter): Σ_x eq(τ, x) · (Az(x)·Bz(x) − Cz(x)) = 0. Degree 2. Rounds = num_rows_bits. Produces virtual Az/Bz/Cz at r, plus committed/non-committed input openings at r_cycle.

### Stage 2
- Spartan Inner: claim_Az + ρ·claim_Bz + ρ^2·claim_Cz = Σ_y (A_small(rx,y)+ρ·B_small+ρ^2·C_small)·z(y). Degree 2. Rounds = log2(|y|).
- Registers Read/Write (SumcheckId: RegistersReadWriteChecking): eq(r',r_cycle) · [ rd_wa·(Inc+Val) + γ·rs1_ra·Val + γ^2·rs2_ra·Val ]. Degree 3. Rounds = log2(K)+log2(T). Produces virtual RegistersVal/Rs1Ra/Rs2Ra/RdWa and dense RdInc at r_cycle.
- RAM Read/Write (SumcheckId: RamReadWriteChecking): eq(r',r_cycle) · ra · (Val + γ·(Val+Inc)). Degree 3. Rounds = log2(K)+log2(T). Produces virtual RamVal/RamRa and dense RamInc at r_cycle.
- RAM RAF Evaluation (SumcheckId: RamRafEvaluation): raf(r_address) = Σ_k ra(k)·unmap(k). Degree 2. Rounds = log2(K). Produces virtual RamRa at (r_address,r_cycle).

### Stage 3
- PC Shift (SumcheckId: SpartanShift): NextUnexpandedPC + γ·NextPC + γ^2·NextIsNoop = Σ_t (UnexpandedPC + γ·PC + γ^2·IsNoop) · EqPlusOne(r_cycle,t). Degree 2. Rounds = log2(T). Produces UnexpandedPC/PC/IsNoop at shifted r.
- Registers Val Evaluation (SumcheckId: RegistersValEvaluation): val(r_address,r_cycle) = Σ_j Inc(j)·Wa(r_address,j)·LT(r',r). Degree 3. Rounds = log2(T). Produces dense RdInc and virtual RdWa.
- Instruction Read-RAF (SumcheckId: InstructionReadRaf): eq(r_cycle',r_cycle) · (∏_i ra_i) · V_eval, where V_eval combines table MLEs and operand prefixes with flags; input claim batches rv and operand terms with γ. Degree D+2. Rounds = LOG_K+log2(T). Produces table flags, InstructionRafFlag, and sparse InstructionRa(i).
- Instruction Booleanity (SumcheckId: InstructionBooleanity): Σ eq · Σ_i γ^i · (ra_i^2 − ra_i). Degree 3. Rounds = LOG_K_CHUNK+log2(T). Produces sparse InstructionRa(i).
- Instruction HammingWeight (SumcheckId: InstructionHammingWeight): linear check over ra_i slices batched by γ powers. Degree 1. Rounds = LOG_K_CHUNK. Produces sparse InstructionRa(i).
- RAM Val Evaluation (SumcheckId: RamValEvaluation): val(r_address,r_cycle) − val_init(r_address) = Σ_j Inc(j)·Wa(r_address,j)·LT(r',r). Degree 3. Rounds = log2(T). Produces dense RamInc and virtual RamRa.
- RAM Hamming Booleanity (SumcheckId: RamHammingBooleanity): Σ_j eq(r',j) · (H(j)^2 − H(j)) with H(j)=1_{addr≠0}. Degree 3. Rounds = log2(T). Produces virtual RamHammingWeight.

### Stage 4
- RAM HammingWeight (SumcheckId: RamHammingWeight): input claim = HammingBooleanity_claim · Σ_i γ^i; verifies Σ_i γ^i · ra_i via sparse committed RamRa(i). Degree 1. Rounds = log2(DTH_ROOT_OF_K).
- RAM RA Virtual (SumcheckId: RamRaVirtual / RASumcheck): RA structural constraints (fill exact integrand). Degree/rounds: per implementation.
- Bytecode Read-RAF / Booleanity / HammingWeight: same as instruction lookups but over PC domain (uses bytecode preprocessing); same degrees/rounds as the corresponding instruction checks.

### Binding/endian notes
- Registers/RAM read-write: cycle bits bound in two segments split at `twist_sumcheck_switch_index`; verifier normalizes by reversing high-order cycle segment. Address bits bound HighToLow after cycle.
- Each instance defines `normalize_opening_point` to match BIG_ENDIAN/LITTLE_ENDIAN MLE expectations.

### Next steps
- Fill exact formulas for RAM OutputSumcheck and RAM RASumcheck once frozen; add file/func pointers for each instance.


