# Part 2:  Execution and Witness Generation Deep Dive

This document expands on Part 2 of [JOLT_CODE_FLOW.md](JOLT_CODE_FLOW.md), connecting the execution and witness generation to both the theory in [Theory/Jolt.md](Theory/Jolt.md) and the actual implementation.

## Overview: From Guest Function Call to Execution Trace

Part 2 is where the **actual computation happens**. When you call `prove_{your_function}(input)`, two major things occur:

1. **RISC-V Emulation**: Execute the guest program instruction-by-instruction, recording everything
2. **Witness Generation**: Transform the execution trace into mathematical objects (polynomials) that can be proven

Think of it as: **Run the program → Record what happened → Convert to math**

---

## Theory Connection: The Execution Trace

From [Theory/Jolt.md:155-169](Theory/Jolt.md#L155):

> **The execution trace** is a complete record of every step the program takes. In Jolt's RISC-V implementation, each cycle is captured in a struct like `RV32IMCycle`, containing:
> - The program counter (PC)
> - The instruction executed
> - The values of the source (`rs1`, `rs2`) and destination (`rd`) registers
> - Any memory addresses and values that were accessed
>
> This trace is the foundation for the entire proof. Jolt transforms this cycle-by-cycle record into a set of witness polynomials (MLEs) that can be verified cryptographically.

From [Theory/Jolt.md:185-195](Theory/Jolt.md#L185):

> Consider a hypothetical 64-bit machine with a simple `ADD` instruction, `ADD rd, rs1, rs2`:
> 1. `LOAD r1, 100`  (Load the value 100 into register r1)
> 2. `LOAD r2, 250`  (Load the value 250 into register r2)
> 3. `ADD r3, r1, r2`   (Add r1 and r2, store in r3)
>
> The execution trace for the third instruction, `ADD r3, r1, r2`, can be viewed as a query to a lookup table. The inputs to the lookup are the operation (`ADD`), the value in `r1` (100), and the value in `r2` (250). The expected output is the result of the addition (350), which will be stored in `r3`.
>
> In Jolt's model, the entire proof of the program's execution is transformed into proving a sequence of such lookups.

**Key insight**: The trace isn't just for debugging—it's the **witness** for the proof!

---

## Part 1: Macro-Generated Prove Function

### Step 1: User Calls `prove_{your_function}()`

From host program:
```rust
let prove_my_function = guest::build_prover_my_function(program, prover_preprocessing);
let (output, proof, program_io) = prove_my_function(input);
```

**What `build_prover_my_function` returns**: A closure that captures the program and preprocessing.

### Step 2: Macro-Generated Prove Function

**File**: [jolt-sdk/macros/src/lib.rs:600+](jolt-sdk/macros/src/lib.rs#L600)

The `#[jolt::provable]` macro generates a `prove_{fn_name}` function:

```rust
pub fn prove_my_function(
    program: jolt::host::Program,
    preprocessing: jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
    // ... input arguments ...
) -> (OutputType, JoltProof, JoltDevice) {
    // 1. Serialize inputs
    let mut input_bytes = vec![];
    input_bytes.append(&mut postcard::to_stdvec(&input).unwrap());

    let mut untrusted_advice_bytes = vec![];
    // ... (if any untrusted advice args)

    let mut trusted_advice_bytes = vec![];
    // ... (if any trusted advice args)

    // 2. Call JoltRV64IMAC::prove
    let (proof, program_io, debug_info, duration) = JoltRV64IMAC::prove(
        &preprocessing,
        &program.elf,
        &input_bytes,
        &untrusted_advice_bytes,
        &trusted_advice_bytes,
        trusted_advice_commitment,
    );

    // 3. Deserialize output
    let output: OutputType = postcard::from_bytes(&program_io.outputs).unwrap();

    (output, proof, program_io)
}
```

**Key steps**:

1. Serialize guest inputs to bytes (using `postcard` - a `no_std` serialization format)
2. Call core prove function (`JoltRV64IMAC::prove`)
3. Deserialize outputs from bytes

---

## Part 2: Core Prove Function Entry

### Step 3: JoltRV64IMAC::prove()

**File**: [jolt-core/src/zkvm/mod.rs:257-296](jolt-core/src/zkvm/mod.rs#L257)

```rust
fn prove(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    trusted_advice_commitment: Option<PCS::Commitment>,
) -> (JoltProof, JoltDevice, Option<ProverDebugInfo>, Duration) {
    // 1. Setup memory configuration
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: preprocessing.shared.memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: preprocessing.shared.memory_layout.max_trusted_advice_size,
        max_input_size: preprocessing.shared.memory_layout.max_input_size,
        max_output_size: preprocessing.shared.memory_layout.max_output_size,
        stack_size: preprocessing.shared.memory_layout.stack_size,
        memory_size: preprocessing.shared.memory_layout.memory_size,
        program_size: Some(preprocessing.shared.memory_layout.program_size),
    };

    // 2. Trace the execution (THE BIG ONE!)
    let (mut trace, final_memory_state, mut program_io) = {
        let _pprof_trace = pprof_scope!("trace");
        guest::program::trace(
            elf_contents,
            None,
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
        )
    };

    // 3. Log trace statistics
    let num_riscv_cycles: usize = trace.par_iter().map(|cycle| {
        // Count real RISC-V cycles vs virtual instruction cycles
        if let Some(inline_sequence_remaining) =
            cycle.instruction().normalize().inline_sequence_remaining
        {
            if inline_sequence_remaining > 0 {
                return 0; // Virtual instruction (part of inline sequence)
            }
        }
        1 // Real RISC-V instruction
    }).sum();

    tracing::info!(
        "{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
        trace.len() - num_riscv_cycles,
        trace.len(),
    );

    // 4. Pad trace to power of 2 (required for MLEs)
    let trace_length = trace.len();
    let padded_trace_length = (trace.len() + 1).next_power_of_two();
    trace.resize(padded_trace_length, Cycle::NoOp);

    // 5. Truncate trailing zeros from outputs
    program_io.outputs.truncate(
        program_io.outputs.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1)
    );

    // 6. Create state manager and generate proof (Part 3!)
    let state_manager = StateManager::new_prover(
        preprocessing,
        trace,
        program_io,
        trusted_advice_commitment,
        final_memory_state,
    );

    let start = Instant::now();
    let (proof, debug_info) = JoltDAG::prove(state_manager)?;
    let duration = start.elapsed();

    (proof, program_io, debug_info, duration)
}
```

**Key outputs**:

- `trace`: Vec<Cycle> - complete execution record
- `final_memory_state`: Memory - RAM state after execution
- `program_io`: JoltDevice - inputs/outputs/panic status

---

## Part 3: RISC-V Emulation (The Tracer)

### Step 4: guest::program::trace()

**File**: [tracer/src/lib.rs:72-91](tracer/src/lib.rs#L72)

```rust
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> (Vec<Cycle>, Memory, JoltDevice) {
    // Create lazy iterator for trace generation
    let mut lazy_trace_iter = LazyTraceIterator::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    ));

    // Collect all cycles (this is where emulation happens!)
    let trace: Vec<Cycle> = lazy_trace_iter.by_ref().collect();

    // Extract final memory state
    let final_memory_state = std::mem::take(lazy_trace_iter.final_memory_state.as_mut().unwrap());

    (trace, final_memory_state, lazy_trace_iter.get_jolt_device())
}
```

**Key components**:

1. **LazyTraceIterator**: Generates cycles on-demand (memory efficient)
2. **setup_emulator_with_backtraces**: Initializes RISC-V CPU emulator
3. **collect()**: Actually runs the emulation, producing `Vec<Cycle>`

### Step 5: The CPU Emulator

**File**: [tracer/src/emulator/cpu.rs:90-116](tracer/src/emulator/cpu.rs#L90)

The `Cpu` struct is a full RISC-V processor emulator:

```rust
pub struct Cpu {
    clock: u64,                              // Cycle counter
    pub xlen: Xlen,                          // 32-bit or 64-bit mode
    pub privilege_mode: PrivilegeMode,       // User/Supervisor/Machine
    pub x: [i64; REGISTER_COUNT],            // 32 general-purpose registers
    f: [f64; 32],                            // 32 floating-point registers (unused in Jolt)
    pub pc: u64,                             // Program counter
    csr: [u64; CSR_CAPACITY],                // Control and Status Registers
    pub mmu: Mmu,                            // Memory Management Unit
    reservation: u64,                        // For atomic operations
    is_reservation_set: bool,
    pub trace_len: usize,                    // Current trace length
    executed_instrs: u64,                    // Real RISC-V cycle count
    active_markers: FnvHashMap<u32, ActiveMarker>, // Cycle tracking markers
    pub vr_allocator: VirtualRegisterAllocator,    // Virtual register allocation
    call_stack: VecDeque<CallFrame>,         // For panic backtraces
}
```

**The emulation loop** (simplified):

```rust
impl Cpu {
    pub fn tick(&mut self, trace: Option<&mut Vec<Cycle>>) {
        // 1. Fetch instruction from memory
        let word = self.mmu.fetch_instruction(self.pc);

        // 2. Decode instruction
        let instruction = Instruction::decode(word, self.pc);

        // 3. Execute and trace
        instruction.trace(&mut self, trace);

        // 4. Update program counter (unless instruction modified it)
        if !instruction.modifies_pc() {
            self.pc += instruction.size();
        }
    }
}
```

### Step 6: Instruction Execution and Tracing

**Two core traits**:

#### RISCVInstruction Trait

**File**: [tracer/src/instruction/mod.rs:321-344](tracer/src/instruction/mod.rs#L321)

```rust
pub trait RISCVInstruction:
    std::fmt::Debug + Sized + Copy + Into<Instruction> + ...
{
    const MASK: u32;   // Bit pattern for instruction matching
    const MATCH: u32;  // Expected bits after masking

    type Format: InstructionFormat;         // R-type, I-type, S-type, etc.
    type RAMAccess: Default + Into<RAMAccess>; // Memory access pattern

    fn operands(&self) -> &Self::Format;
    fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self;

    // Core execution logic (modifies CPU state)
    fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess);
}
```

#### RISCVTrace Trait

**File**: [tracer/src/instruction/mod.rs:346-373](tracer/src/instruction/mod.rs#L346)

```rust
pub trait RISCVTrace: RISCVInstruction
where
    RISCVCycle<Self>: Into<Cycle>,
{
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // 1. Create cycle struct to capture state
        let mut cycle: RISCVCycle<Self> = RISCVCycle {
            instruction: *self,
            register_state: Default::default(),
            ram_access: Default::default(),
        };

        // 2. Capture PRE-execution state (register values BEFORE)
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);

        // 3. Execute the instruction (modifies CPU state)
        self.execute(cpu, &mut cycle.ram_access);

        // 4. Capture POST-execution state (register values AFTER)
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);

        // 5. Add cycle to trace
        if let Some(trace_vec) = trace {
            trace_vec.push(cycle.into());
        }
    }

    // Default: single instruction. Virtual sequences override this.
    fn inline_sequence(&self, ...) -> Vec<Instruction> {
        vec![(*self).into()]
    }
}
```

**Key insight**: `trace()` wraps `execute()`, capturing state before/after!

### Step 7: Example - ADD Instruction

**File**: [tracer/src/instruction/add.rs](tracer/src/instruction/add.rs)

```rust
// Macro defines instruction metadata
declare_riscv_instr!(
    name   = ADD,
    mask   = 0xfe00707f,    // Bits that must match
    match  = 0x00000033,    // Expected pattern for ADD
    format = FormatR,       // R-type: opcode rd, rs1, rs2
    ram    = ()             // No memory access
);

impl ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADD as RISCVInstruction>::RAMAccess) {
        // Read source registers
        let rs1_val = cpu.x[self.operands.rs1 as usize];
        let rs2_val = cpu.x[self.operands.rs2 as usize];

        // Perform addition
        let result = rs1_val.wrapping_add(rs2_val);

        // Write to destination register
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(result);
    }
}

// Use default trace implementation
impl RISCVTrace for ADD {}
```

**What gets captured in the cycle**:

- **Pre-execution**: `rs1` value (e.g., 100), `rs2` value (e.g., 250)
- **Execution**: Compute `100 + 250 = 350`
- **Post-execution**: `rd` value (350)
- **Instruction**: The `ADD` opcode and operand indices

### Step 8: The Cycle Enum

**File**: [tracer/src/instruction/mod.rs:394-399](tracer/src/instruction/mod.rs#L394)

```rust
pub enum Cycle {
    NoOp,                         // Padding cycle
    ADD(RISCVCycle<ADD>),
    SUB(RISCVCycle<SUB>),
    XOR(RISCVCycle<XOR>),
    // ... one variant per instruction ...
}

pub struct RISCVCycle<I: RISCVInstruction> {
    instruction: I,                      // The instruction itself
    register_state: I::Format,           // Register operands (pre + post)
    ram_access: I::RAMAccess,            // Memory read/write (if any)
}
```

**Example cycle for `ADD r3, r1, r2`**:
```rust
Cycle::ADD(RISCVCycle {
    instruction: ADD {
        operands: FormatR { rd: 3, rs1: 1, rs2: 2, ... },
        address: 0x80000100,
    },
    register_state: FormatR {
        rs1_val_pre: 100,    // r1 value before
        rs2_val_pre: 250,    // r2 value before
        rd_val_post: 350,    // r3 value after
    },
    ram_access: (),          // No memory access
})
```

This is **exactly the witness** from the theory: (opcode, inputs, outputs)!

---

## Part 4: Witness Generation (Trace → Polynomials)

After emulation completes, we have `Vec<Cycle>`. Now convert to **multilinear polynomial extensions (MLEs)**.

---

## Mathematical Foundation: From Execution to Polynomials

Before diving into the code, let's understand the **mathematical transformation** we're performing.

### The Witness as a Mathematical Object

**From Theory/Jolt.md (Section 1.3.3)**:

> The execution trace is a complete record of every step the program takes. [...] This trace is the foundation for the entire proof. Jolt transforms this cycle-by-cycle record into a set of witness polynomials (MLEs) that can be verified cryptographically.

**What is a witness?** In the context of interactive proofs:

$$\text{Witness} = \text{Complete information that proves a claim}$$

For Jolt, the claim is: **"Program P executed correctly with input I and produced output O"**

The witness consists of:

1. **Execution trace**: Sequence of CPU states $(s_0, s_1, \ldots, s_T)$
2. **Polynomial encodings**: MLEs that represent this trace cryptographically

### Why Polynomials? The Power of Multilinear Extensions

**Key insight from sumcheck theory**:

Instead of verifying $T$ individual cycles (expensive!), we:

1. Encode the entire trace as a multilinear polynomial $f : \mathbb{F}^{\log T} \to \mathbb{F}$
2. Use sumcheck to verify properties about $\sum_{x \in \{0,1\}^{\log T}} f(x)$
3. Verifier only evaluates $f$ at a **single random point** $r \in \mathbb{F}^{\log T}$

**Why this works** (Schwartz-Zippel lemma):

- If prover cheats and uses wrong polynomial $f'$, then $f' \neq f$
- Probability that $f'(r) = f(r)$ at random $r$ is $\leq \frac{d}{|\mathbb{F}|}$ (negligible for large field)

**Multilinear Extension (MLE)** of vector $v \in \mathbb{F}^{2^n}$:

$$\widetilde{v}(X_1, \ldots, X_n) = \sum_{b \in \{0,1\}^n} v[b] \cdot \text{eq}(X_1, \ldots, X_n; b)$$

Where $\text{eq}(X; b)$ is the multilinear extension of the indicator function:

$$\text{eq}(X; b) = \prod_{i=1}^n (X_i b_i + (1-X_i)(1-b_i))$$

**Property**: $\widetilde{v}$ agrees with $v$ on the Boolean hypercube:
$$\forall b \in \{0,1\}^n: \widetilde{v}(b) = v[b]$$

But extends uniquely to all of $\mathbb{F}^n$ as a multilinear polynomial!

### The Jolt Lookup Architecture

**From Theory/Jolt.md (Section 1.3.4)**:

> Jolt combines all possible operations into a single, giant lookup table, `JOLT_V`:
> $$\text{JOLT\_V}(\text{opcode}, a, b) = (c, \text{flags}) = f_{op}(a, b)$$
>
> A proof of execution for a program trace of length $m$ becomes a sequence of $m$ lookup claims:
> - At step 1: $(c_1, \text{flags}_1) = \text{JOLT\_V}(\text{opcode}_1, a_1, b_1)$
> - At step 2: $(c_2, \text{flags}_2) = \text{JOLT\_V}(\text{opcode}_2, a_2, b_2)$
> - ...

**The witness polynomials encode**:

- **Inputs**: $(a_1, b_1), (a_2, b_2), \ldots, (a_T, b_T)$ → MLEs
- **Outputs**: $(c_1, \text{flags}_1), (c_2, \text{flags}_2), \ldots$ → MLEs
- **Lookup indices**: Decomposed chunks for Shout protocol → One-hot MLEs

### Mathematical Objects in Jolt's Witness

#### 1. Simple Witness Vectors (Direct MLEs)

**Mathematical definition**:

For a witness vector $w = (w_0, w_1, \ldots, w_{T-1}) \in \mathbb{F}^T$ (padded to $T = 2^t$):

$$\widetilde{w}(X_1, \ldots, X_t) : \mathbb{F}^t \to \mathbb{F}$$

**Examples in Jolt**:

**A) Left and Right Instruction Inputs** - The two operands for each instruction:

- `left_instruction_input`: $\widetilde{L} = (a_0, a_1, \ldots, a_{T-1})$ where $a_j$ is the **first operand** at cycle $j$
- `right_instruction_input`: $\widetilde{R} = (b_0, b_1, \ldots, b_{T-1})$ where $b_j$ is the **second operand** at cycle $j$

**What they represent**:
$$\widetilde{L}(j) = a_j \quad \text{(value of left operand at cycle } j\text{)}$$
$$\widetilde{R}(j) = b_j \quad \text{(value of right operand at cycle } j\text{)}$$

Where $j \in \{0,1\}^t$ encodes cycle index in binary.

**Concrete Example - Trace snippet**:

| Cycle | Instruction | rs1 value | rs2 value | Left Input ($a_j$) | Right Input ($b_j$) |
|-------|-------------|-----------|-----------|--------------------|--------------------|
| 0 | `ADD r3, r1, r2` | 100 | 250 | 100 | 250 |
| 1 | `SUB r4, r3, r2` | 350 | 250 | 350 | 250 |
| 2 | `XOR r5, r1, r4` | 100 | 100 | 100 | 100 |
| 3 | `ADDI r6, r5, 42` | 0 | (imm=42) | 0 | 42 |

**The witness vectors**:
$$\widetilde{L} = (100, 350, 100, 0, \ldots) \quad \text{(left operand each cycle)}$$
$$\widetilde{R} = (250, 250, 100, 42, \ldots) \quad \text{(right operand each cycle)}$$

**Key insight**: These are simply the **inputs to the giant JOLT_V lookup table** at each cycle. For R-type instructions, they come from registers `rs1` and `rs2`. For I-type instructions, the right input is the immediate value.

**B) Register Destination Increment** - For Twist memory checking:

- `rd_inc`: $\widetilde{\Delta}_{\text{rd}} = (\Delta_0, \Delta_1, \ldots, \Delta_{T-1})$ where $\Delta_j$ is the **increment written to destination register** at cycle $j$

**Mathematical definition**:
$$\widetilde{\Delta}_{\text{rd}}(j) = \Delta_j = \begin{cases}
\text{rd\_val\_after} - \text{rd\_val\_before} & \text{if cycle } j \text{ writes to a register} \\
0 & \text{otherwise (no write)}
\end{cases}$$

**Why "increment" not "value"?** Twist memory checking uses an **incremental formulation**:

$$\text{final\_register\_state}(r) = \text{initial\_state}(r) + \sum_{j: \text{writes to } r} \Delta_j$$

This proves final state equals initial state plus all writes (memory consistency).

**Concrete Example - Continuing the trace**:

| Cycle | Instruction | rd (dest) | rd value before | rd value after | Increment ($\Delta_j$) |
|-------|-------------|-----------|-----------------|----------------|----------------------|
| 0 | `ADD r3, r1, r2` | r3 | 0 | 350 | **350** |
| 1 | `SUB r4, r3, r2` | r4 | 0 | 100 | **100** |
| 2 | `XOR r5, r1, r4` | r5 | 0 | 0 | **0** |
| 3 | `ADDI r6, r5, 42` | r6 | 0 | 42 | **42** |
| 4 | `LW r7, 0(r3)` | r7 | 0 | 999 | **999** |

**The witness vector**:
$$\widetilde{\Delta}_{\text{rd}} = (350, 100, 0, 42, 999, \ldots)$$

**Verification via Twist sumcheck**:
$$\sum_{j \in \{0,1\}^{\log T}} \widetilde{\Delta}_{\text{rd}}(j) = 350 + 100 + 0 + 42 + 999 + \cdots$$

This sum, combined with initial register state, must equal the final register state (memory consistency check).

**C) RAM Increment** - For Twist memory checking of RAM reads/writes:

- `ram_inc`: $\widetilde{\Delta}_{\text{ram}} = (\Delta_0^{\text{ram}}, \Delta_1^{\text{ram}}, \ldots, \Delta_{T-1}^{\text{ram}})$ where $\Delta_j^{\text{ram}}$ is the **increment written to memory** at cycle $j$

**Mathematical definition**:
$$\widetilde{\Delta}_{\text{ram}}(j) = \Delta_j^{\text{ram}} = \begin{cases}
\text{value\_written} - \text{value\_read} & \text{if cycle } j \text{ writes to RAM (SW/SH/SB)} \\
0 & \text{if cycle } j \text{ only reads RAM (LW/LH/LB)} \\
0 & \text{otherwise (no memory access)}
\end{cases}$$

**Key difference from register increment**:

- **Registers**: Increment = after - before (always has "before" value)
- **RAM**: Increment = written - read (only non-zero for stores, reads don't change memory)

**Why this formulation?** Twist proves memory consistency:
$$\text{final\_memory}(\text{addr}) = \text{initial\_memory}(\text{addr}) + \sum_{j: \text{accesses addr}} \Delta_j^{\text{ram}}$$

For a read-only address: sum of $\Delta$ is zero (no writes) → final = initial 

**Concrete Example - Memory operations**:

Assume initial RAM state: `memory[0x1000] = 0`, `memory[0x1004] = 0`

| Cycle | Instruction | Address | Operation | Value Read | Value Written | Increment ($\Delta_j^{\text{ram}}$) |
|-------|-------------|---------|-----------|------------|---------------|-----------------------------------|
| 0 | `ADD r3, r1, r2` | - | (no memory) | - | - | **0** |
| 1 | `SW r3, 0(r10)` | 0x1000 | **STORE** | 0 (old) | 350 (new) | **350** |
| 2 | `LW r7, 0(r10)` | 0x1000 | **LOAD** | 350 | - | **0** (reads don't increment) |
| 3 | `ADDI r8, r7, 50` | - | (no memory) | - | - | **0** |
| 4 | `SW r8, 4(r10)` | 0x1004 | **STORE** | 0 (old) | 400 (new) | **400** |
| 5 | `SW r3, 0(r10)` | 0x1000 | **STORE** | 350 (old) | 350 (same) | **0** (no change) |
| 6 | `LW r9, 4(r10)` | 0x1004 | **LOAD** | 400 | - | **0** (reads don't increment) |

**The witness vector** (one entry per cycle $j$):
$$\widetilde{\Delta}_{\text{ram}} = (\Delta_0^{\text{ram}}, \Delta_1^{\text{ram}}, \Delta_2^{\text{ram}}, \Delta_3^{\text{ram}}, \Delta_4^{\text{ram}}, \Delta_5^{\text{ram}}, \Delta_6^{\text{ram}}, \ldots)$$
$$= (0, 350, 0, 0, 400, 0, 0, \ldots)$$

 **CRITICAL QUESTION: How does this vector track which address each increment goes to?**

**Answer**: It doesn't on its own! The address information is stored **separately** in $\widetilde{\text{mem}}_i(j,k)$ (the RamRa one-hot polynomials explained in section 2 below).

**The key insight**: Twist uses **TWO separate polynomials** working together:

1. **$\widetilde{\Delta}_{\text{ram}}(j)$**: "What increment happened at cycle $j$?"
   - Domain: $\mathbb{F}^{\log T}$ (indexed by cycle only)
   - $\Delta_1^{\text{ram}} = 350$ (some increment at cycle 1)
   - $\Delta_4^{\text{ram}} = 400$ (some increment at cycle 4)
   - **Missing info**: Which address?

2. **$\widetilde{\text{mem}}_i(j, k)$**: "Which address was accessed at cycle $j$?" (see section 2.C below for full explanation)
   - Domain: $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M}$ (indexed by cycle AND address)
   - $\widetilde{\text{mem}}_i(1, k_1) = 1$ where $k_1$ maps to 0x1000 (cycle 1 accessed 0x1000)
   - $\widetilde{\text{mem}}_i(4, k_2) = 1$ where $k_2$ maps to 0x1004 (cycle 4 accessed 0x1004)
   - **Provides**: Address routing for increments

**How Twist connects them**:

For each address $k$, compute final state using the one-hot $\widetilde{\text{mem}}$ as a **selector**:
$$\text{final}(k) = \text{initial}(k) + \sum_{j=0}^{T-1} \widetilde{\text{mem}}_i(j, k) \cdot \widetilde{\Delta}_{\text{ram}}(j)$$

The one-hot property means $\widetilde{\text{mem}}_i(j, k) = 1$ only when cycle $j$ accessed address $k$, so only those increments get added.

**Concretely for address** $k_1 = 0x1000$:

Cycles 1, 2, 5 accessed 0x1000, so:
$$\text{final}(k_1) = 0 + \sum_{j=0}^{6} \widetilde{\text{mem}}(j, k_1) \cdot \Delta_j^{\text{ram}}$$
$$= 0 + \underbrace{\widetilde{\text{mem}}(1, k_1)}_{=1} \cdot \underbrace{\Delta_1}_{=350} + \underbrace{\widetilde{\text{mem}}(2, k_1)}_{=1} \cdot \underbrace{\Delta_2}_{=0} + \underbrace{\widetilde{\text{mem}}(5, k_1)}_{=1} \cdot \underbrace{\Delta_5}_{=0} + \text{(other terms are 0)}$$
$$= 1 \cdot 350 + 1 \cdot 0 + 1 \cdot 0 = 350 \; \checkmark$$

**For address** $k_2 = 0x1004$:

Cycles 4, 6 accessed 0x1004, so:
$$\text{final}(k_2) = 0 + \sum_{j=0}^{6} \widetilde{\text{mem}}(j, k_2) \cdot \Delta_j^{\text{ram}}$$
$$= 0 + \underbrace{\widetilde{\text{mem}}(4, k_2)}_{=1} \cdot \underbrace{\Delta_4}_{=400} + \underbrace{\widetilde{\text{mem}}(6, k_2)}_{=1} \cdot \underbrace{\Delta_6}_{=0} + \text{(other terms are 0)}$$
$$= 1 \cdot 400 + 1 \cdot 0 = 400 \; \checkmark$$

**Why this design?**

- **Efficiency**: One simple vector $\widetilde{\Delta}_{\text{ram}}$ (size $T$) instead of separate vectors per address (size $M \times T$)
- **Linking**: The one-hot $\widetilde{\text{mem}}$ acts as a **selector** routing each increment to the correct address
- **Separation of concerns**:
  - **Value changes**: $\widetilde{\Delta}_{\text{ram}}(j)$ - "how much changed at cycle $j$?"
  - **Address routing**: $\widetilde{\text{mem}}_i(j,k)$ - "which address was accessed at cycle $j$?"

**Key observations**:

- **Loads (LW) contribute $\Delta = 0$**: They don't change memory, only read it
- **Stores (SW) contribute $\Delta = \text{new} - \text{old}$**: The change in memory
- **Overwriting same value**: Cycle 5 writes 350 to address already containing 350 → $\Delta = 0$
- **Twist proves**: Sum of all increments to each address equals the final state change

#### 2. One-Hot Witness Matrices (For Shout Lookups)

**Mathematical definition**:

For lookups into a table of size $N$, the one-hot polynomial is:

$$\widetilde{\text{ra}}(j, k) : \mathbb{F}^{\log T} \times \mathbb{F}^{\log N} \to \mathbb{F}$$

Such that:
$$\widetilde{\text{ra}}(j, k) = \begin{cases}
1 & \text{if cycle } j \text{ accessed table entry } k \\
0 & \text{otherwise}
\end{cases}$$

**Sumcheck property** (key to Shout):

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log N}} \widetilde{\text{ra}}(j, k) \cdot (\text{value\_read}(k) - \text{value\_written}(k)) = 0$$

This proves: "Every value read from the table matches what was written"

#### **Detailed Explanation of This Formula**

Let's break down what this sumcheck actually verifies:

**Setup**:

- We have $T$ lookups (one per cycle) into a table of size $N$
- Each lookup: cycle $j$ reads entry $k$ from the table
- $\widetilde{\text{ra}}(j, k)$ is the one-hot polynomial encoding these lookups

**The formula has two sums**:

$$\underbrace{\sum_{j \in \{0,1\}^{\log T}}}_{\text{For each cycle}} \underbrace{\sum_{k \in \{0,1\}^{\log N}}}_{\text{For each table entry}} \underbrace{\widetilde{\text{ra}}(j, k)}_{\substack{\text{1 if cycle } j \\ \text{looked up entry } k \\ \text{0 otherwise}}} \cdot \underbrace{(\text{value\_read}(k) - \text{value\_written}(k))}_{\text{Difference between claimed and actual table values}}$$

**What each term means** ( **terminology is confusing here!**):

1. **$\text{value\_read}(k)$**: The value the **prover claims** they got from table entry $k$
   - This comes from the execution trace
   - Example: Prover says "When I looked up entry 42 in the ADD table, I got 100"

2. **$\text{value\_written}(k)$**: The **actual/correct value** at table entry $k$ ( **confusing name!**)
   - **NOT about temporal "writing"** in the instruction lookup case!
   - Better mental model: **"the ground truth value at index $k$"**

   **Two different contexts**:

   **A) For read-only lookup tables (instruction lookups via Shout)**:
   - `value_written(k)` = **Pre-computed table value** (the "definition" of the table)
   - Example: $\text{ADD\_table}(10, 20) = 30$ (computed once during preprocessing, never changes)
   - "written" here means "what's defined in the table spec"
   - The table is **static** - no temporal writes happen!

   **B) For read-write memory (RAM/registers via Twist)**:
   - `value_written(k)` = **Last value written** to address $k$ temporally
   - Example: If address 0x1000 was written with value 42 at cycle 10, then for cycle 15 reading it: `value_written` = 42
   - "written" here genuinely means "most recent temporal write"
   - The memory state **evolves** over time!

3. **$\widetilde{\text{ra}}(j, k)$**: The one-hot polynomial that's 1 only when cycle $j$ accessed entry $k$
   - Acts as a "selector" - only non-zero for actual lookups
   - Example: If cycle 5 looked up entry 42, then $\widetilde{\text{ra}}(5, 42) = 1$ and $\widetilde{\text{ra}}(5, i) = 0$ for all $i \neq 42$

**Why the sum equals zero**:

For each cycle $j$ that looks up entry $k$:

- $\widetilde{\text{ra}}(j, k) = 1$
- If the lookup is **correct**: $\text{value\_read}(k) = \text{value\_written}(k)$
- So: $1 \cdot (\text{value\_read}(k) - \text{value\_written}(k)) = 1 \cdot 0 = 0$

For all other $(j, k)$ pairs:

- $\widetilde{\text{ra}}(j, k) = 0$
- So: $0 \cdot (\ldots) = 0$

**Total sum**: $0 + 0 + \cdots + 0 = 0$

**If prover cheats**:

- Suppose at cycle 5, prover claims they looked up entry 42 and got 999 (but correct value is 100)
- Then: $\widetilde{\text{ra}}(5, 42) \cdot (999 - 100) = 1 \cdot 899 = 899 \neq 0$
- Sum is **non-zero** → Verifier rejects!

**Concrete Example: Instruction Lookup (Read-Only Table)**

For `ADD r3, r1, r2` with `r1=10`, `r2=20` at cycle 5:

- Instruction decomposes into 16 chunks (4 bits each)
- First chunk lookup: 10 (4 bits) + 20 (4 bits) should give 30 (4 bits)
- Table entry: $k = (10, 20)$ in binary encoding
- Pre-computed table: $\text{ADD\_table}(10, 20) = 30$ ← This is `value_written(k)` (the "ground truth")

**The sumcheck verifies**:
$$\widetilde{\text{ra}}_0(5, k) \cdot (\underbrace{\text{prover's claimed result}}_{\text{value\_read}(k)} - \underbrace{30}_{\text{value\_written}(k) = \text{table definition}}) = 0$$

This forces prover to claim the correct result (30), otherwise the sum won't be zero!

**Why "value_written" for static tables?** The term comes from the Shout protocol's unification of read-only lookups with read-write memory checking. In the protocol formalism:

- For static tables: "written" = "what was written into the table definition during preprocessing"
- For dynamic memory: "written" = "what was last written temporally during execution"

The math is the same, but the meaning differs!

**Examples of One-Hot Polynomials in Jolt**:

**A) InstructionRa(i)** - Instruction lookup one-hot:

- `InstructionRa(i)`: Chunk $i$ of instruction lookup (16 chunks total for 64-bit ops)
- Example: $\widetilde{\text{InstructionRa}}_0(j, k) = 1$ if cycle $j$ performed chunk 0 lookup at table index $k$
- See the ADD example above (cycle 5, chunk 0, looking up entry for inputs 10,20)

**B) BytecodeRa(i)** - Bytecode fetch one-hot ( **"Ra" means "read address", same concept as InstructionRa but for bytecode!**):

- `BytecodeRa(i)`: Chunk $i$ of bytecode address lookup
- **What it tracks**: Which bytecode instruction was fetched at each cycle
- **Why one-hot?** Proves that the instruction fetched matches the committed bytecode

**Concrete Example - Bytecode Lookup**:

Assume we have a tiny program with 4 instructions in bytecode:

| Bytecode Address | Instruction | Encoding |
|------------------|-------------|----------|
| 0x0 | `ADD r3, r1, r2` | 0x00208233 |
| 0x4 | `SUB r4, r3, r2` | 0x402182B3 |
| 0x8 | `XOR r5, r1, r4` | 0x004142B3 |
| 0xC | `ADDI r6, r5, 42` | 0x02A28313 |

**Execution trace** (simplified - assume no jumps, sequential execution):

| Cycle | PC (bytecode addr) | Instruction Fetched | BytecodeRa Index $k$ |
|-------|-------------------|---------------------|----------------------|
| 0 | 0x0 | `ADD r3, r1, r2` | 0 |
| 1 | 0x4 | `SUB r4, r3, r2` | 1 |
| 2 | 0x8 | `XOR r5, r1, r4` | 2 |
| 3 | 0xC | `ADDI r6, r5, 42` | 3 |

**The one-hot polynomial** $\widetilde{\text{BytecodeRa}}(j, k)$:

For a non-chunked example (simplified), the matrix would be:

$$\widetilde{\text{BytecodeRa}} = \begin{bmatrix}
j=0: & [1, & 0, & 0, & 0] & \leftarrow \text{Cycle 0 fetched bytecode index 0} \\
j=1: & [0, & 1, & 0, & 0] & \leftarrow \text{Cycle 1 fetched bytecode index 1} \\
j=2: & [0, & 0, & 1, & 0] & \leftarrow \text{Cycle 2 fetched bytecode index 2} \\
j=3: & [0, & 0, & 0, & 1] & \leftarrow \text{Cycle 3 fetched bytecode index 3} \\
\end{bmatrix}$$

**The sumcheck verifies**:
$$\sum_{j=0}^{3} \sum_{k=0}^{3} \widetilde{\text{BytecodeRa}}(j, k) \cdot (\text{instruction\_fetched}(j) - \text{committed\_bytecode}(k)) = 0$$

This proves:

- Cycle 0 fetched the instruction at bytecode index 0 (ADD) 
- Cycle 1 fetched the instruction at bytecode index 1 (SUB) 
- Cycle 2 fetched the instruction at bytecode index 2 (XOR) 
- Cycle 3 fetched the instruction at bytecode index 3 (ADDI) 

**Key insight**: BytecodeRa is exactly analogous to InstructionRa:

- **InstructionRa**: Proves "cycle $j$ looked up the correct entry in the instruction behavior table"
- **BytecodeRa**: Proves "cycle $j$ fetched the correct instruction from the committed bytecode"

Both use the same one-hot polynomial structure and Shout sumcheck!

**C) RamRa(i)** - RAM address one-hot (also written as $\widetilde{\text{mem}}_i$):

- `RamRa(i)`: Chunk $i$ of RAM address lookup
- **Mathematical notation**: $\widetilde{\text{mem}}_i(j, k)$ where $i$ is the chunk index
- **What it tracks**: Which RAM address was accessed at each cycle (chunked for efficiency)
- **Why one-hot?** Proves that memory accesses happened at the claimed addresses

**Concrete Example - RAM Address Lookup**:

Let's revisit our RAM example from section C above, but now focus on the **address lookups**.

**Memory operations** (from earlier example):

| Cycle $j$ | Instruction | Address | RamRa Index $k$ (simplified) |
|-----------|-------------|---------|------------------------------|
| 0 | `ADD r3, r1, r2` | - | - (no memory access) |
| 1 | `SW r3, 0(r10)` | **0x1000** | $k_1$ (maps to 0x1000) |
| 2 | `LW r7, 0(r10)` | **0x1000** | $k_1$ (same address) |
| 3 | `ADDI r8, r7, 50` | - | - (no memory access) |
| 4 | `SW r8, 4(r10)` | **0x1004** | $k_2$ (maps to 0x1004) |
| 5 | `SW r3, 0(r10)` | **0x1000** | $k_1$ (back to 0x1000) |
| 6 | `LW r9, 4(r10)` | **0x1004** | $k_2$ (back to 0x1004) |

**Important**: Addresses are typically chunked (e.g., 64-bit address → 16 chunks of 4 bits). For simplicity, assume 2 possible addresses.

**The one-hot polynomial** $\widetilde{\text{mem}}_0(j, k)$ (chunk 0, simplified):

For cycles that access memory, the one-hot polynomial indicates which address:

$$\widetilde{\text{mem}}_0 = \begin{bmatrix}
j=0: & [0, & 0] & \leftarrow \text{No memory access} \\
j=1: & [1, & 0] & \leftarrow \text{Cycle 1 accessed address } k_1 \text{ (0x1000)} \\
j=2: & [1, & 0] & \leftarrow \text{Cycle 2 accessed address } k_1 \text{ (0x1000)} \\
j=3: & [0, & 0] & \leftarrow \text{No memory access} \\
j=4: & [0, & 1] & \leftarrow \text{Cycle 4 accessed address } k_2 \text{ (0x1004)} \\
j=5: & [1, & 0] & \leftarrow \text{Cycle 5 accessed address } k_1 \text{ (0x1000)} \\
j=6: & [0, & 1] & \leftarrow \text{Cycle 6 accessed address } k_2 \text{ (0x1004)} \\
\end{bmatrix}$$

**Key observations**:

- **Multiple cycles can access same address**: Cycles 1, 2, 5 all access 0x1000
- **No memory access cycles**: Cycles 0, 3 have all zeros (no lookup)
- **Different from increment polynomial**: $\widetilde{\text{mem}}$ tracks **which address**, $\widetilde{\Delta}_{\text{ram}}$ tracks **what value change**

**Connection to Twist Protocol**:

Twist needs to prove:

1. **Which addresses were accessed** ($\widetilde{\text{mem}}_i$ proves this via Shout)
2. **What increments occurred** ($\widetilde{\Delta}_{\text{ram}}$ tracks this)

Together they prove: "For each memory access at cycle $j$ to address $k$, the value changed by $\Delta_j$"

**The Twist sumcheck uses both**:

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log M}} \widetilde{\text{mem}}_i(j, k) \cdot \left( \text{address\_claimed}(j) - \text{address\_actual}(k) \right) = 0$$

This proves: "Every memory access claimed the correct address"

**Then separately**:
$$\text{final\_mem}(k) = \text{initial\_mem}(k) + \sum_{j: \text{accesses } k} \widetilde{\Delta}_{\text{ram}}(j)$$

This proves: "The increments sum to the correct final state"

**Why chunk addresses?** A 64-bit address has $2^{64}$ possible values - too large for one-hot polynomial!

- **Solution**: Split address into 16 chunks of 4 bits each
- Each chunk: $2^4 = 16$ possible values (tractable!)
- Need 16 separate one-hot polynomials: $\widetilde{\text{mem}}_0, \ldots, \widetilde{\text{mem}}_{15}$

**Comparison of the three "Ra" polynomials**:

| Polynomial | What it tracks | Table size | Purpose |
|------------|----------------|------------|---------|
| $\widetilde{\text{ra}}_i(j,k)$ (InstructionRa) | Which instruction behavior entry | $2^{128}$ (chunked to $2^8$) | Prove correct instruction outputs |
| $\widetilde{\text{bc}}_i(j,k)$ (BytecodeRa) | Which bytecode instruction | $K$ (program size) | Prove correct instruction fetch |
| $\widetilde{\text{mem}}_i(j,k)$ (RamRa) | Which RAM address | $M$ (memory size, chunked) | Prove correct memory addressing |

All three use the **same one-hot structure** and **same Shout sumcheck protocol**!

#### 3. Increment Polynomials (For Twist Memory Checking)

**Mathematical definition** (from Twist protocol):

Memory cell with initial value $\text{init}(a)$ and increments $\Delta_1, \Delta_2, \ldots$ at address $a$:

$$\text{final}(a) = \text{init}(a) + \sum_{i : \text{access to } a} \Delta_i$$

**The increment MLE**:

$$\widetilde{\Delta}(j) = \begin{cases}
\text{value\_written} - \text{value\_read} & \text{if cycle } j \text{ writes to memory/register} \\
0 & \text{otherwise}
\end{cases}$$

**Sumcheck verification**:

$$\sum_{a} \text{final}(a) = \sum_{a} \text{init}(a) + \sum_{j} \widetilde{\Delta}(j)$$

**Examples in Jolt**:

- `RdInc`: Register write increments — $\Delta_j = \text{rd\_val\_post} - \text{rd\_val\_pre}$
- `RamInc`: Memory write increments — $\Delta_j = \text{value\_written} - \text{value\_read}$

### Complete Mathematical Picture

For a trace of length $T = 2^t$ cycles:

**Witness = Set of MLEs**:

$$W = \{\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \widetilde{\Delta}_{\text{ram}}, \widetilde{\text{ra}}_0, \ldots, \widetilde{\text{ra}}_{15}, \ldots\}$$

Where each MLE has domain:

- Simple witnesses: $\mathbb{F}^{\log T}$ (just cycle index)
- One-hot witnesses: $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ (cycle index × table index)

**Size comparison**:

| Representation | Size | Committed Size |
|---------------|------|----------------|
| **Raw trace** | $T \times \text{cycle\_size}$ ≈ 10KB-10MB | N/A |
| **MLE evaluations** | $T \times \text{num\_polys}$ ≈ same | N/A |
| **Dory commitments** | $\text{num\_polys} \times 192$ bytes | ~10KB total! |
| **Opening proof** | $O(\log T)$ per polynomial | ~18KB per poly |

**The magic**: Entire trace compressed to ~10KB of commitments, with logarithmic verification!

---

### Step 9: Committed Polynomial Types

**File**: [jolt-core/src/zkvm/witness.rs:47-80](jolt-core/src/zkvm/witness.rs#L47)

```rust
pub enum CommittedPolynomial {
    /* R1CS aux variables */
    LeftInstructionInput,      // Left operand (rs1 or PC)
    RightInstructionInput,     // Right operand (rs2 or immediate)
    WriteLookupOutputToRD,     // Should write result to rd?
    WritePCtoRD,               // Should write PC to rd (JAL, JALR)?
    ShouldBranch,              // Is this a branch instruction?
    ShouldJump,                // Is this a jump instruction?

    /* Twist/Shout witnesses */
    RdInc,                     // Register write increments (Twist)
    RamInc,                    // Memory write increments (Twist)
    InstructionRa(usize),      // Instruction lookup addresses (Shout, d=16)
    BytecodeRa(usize),         // Bytecode lookup addresses (Shout, d varies)
    RamRa(usize),              // RAM addresses (Twist, d varies)
}
```

**Mathematical correspondence**:

| Rust Enum | Math Object | Domain | Purpose |
|-----------|-------------|--------|---------|
| `LeftInstructionInput` | $\widetilde{L}(j)$ | $\mathbb{F}^{\log T}$ | Left operand values |
| `RightInstructionInput` | $\widetilde{R}(j)$ | $\mathbb{F}^{\log T}$ | Right operand values |
| `WriteLookupOutputToRD` | $\widetilde{w}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | Boolean: write to rd? |
| `ShouldBranch` | $\widetilde{b}(j)$ | $\mathbb{F}^{\log T}$ | Boolean: is branch? |
| `RdInc` | $\widetilde{\Delta}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | Register increment |
| `RamInc` | $\widetilde{\Delta}_{\text{ram}}(j)$ | $\mathbb{F}^{\log T}$ | Memory increment |
| `InstructionRa(i)` | $\widetilde{\text{ra}}_i(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | Lookup chunk $i$ |
| `BytecodeRa(i)` | $\widetilde{\text{bc}}_i(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K}$ | Bytecode chunk $i$ |
| `RamRa(i)` | $\widetilde{\text{mem}}_i(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M}$ | RAM addr chunk $i$ |

**Why these specific polynomials?**

- **R1CS variables** ($\widetilde{L}, \widetilde{R}, \widetilde{w}_{\text{rd}}, etc.$): Needed for Spartan's constraint system to prove VM wiring
- **Twist witnesses** ($\widetilde{\Delta}_{\text{rd}}, \widetilde{\Delta}_{\text{ram}}$): Needed for register/RAM memory checking via incremental state updates
- **Shout witnesses** ($\widetilde{\text{ra}}_i, \widetilde{\text{bc}}_i$): Needed for instruction/bytecode lookup arguments (one-hot encodings)

### Step 10: The Witness Generation Pipeline

**File**: [jolt-core/src/zkvm/witness.rs:200+]

**Goal**: Transform execution trace $(s_0, s_1, \ldots, s_{T-1})$ into ~50 committed witness polynomials.

**The pipeline** (three sub-steps):

```
Step 10A: Allocate WitnessData
    ↓
Step 10B: Extract data from each cycle in parallel
    ↓
Step 10C: Convert to MLEs and commit
```

Let's go through each with mathematical detail:

---

#### Step 10A: Allocate WitnessData Structure

**File**: [jolt-core/src/zkvm/witness.rs:84-121](jolt-core/src/zkvm/witness.rs#L84)

**Mathematical setup**: We need to store ~50 vectors, each of length $T$ (trace length, padded to $T = 2^t$).

```rust
struct WitnessData {
    // Simple polynomial coefficients (one per cycle)
    left_instruction_input: Vec<u64>,        // Size T
    right_instruction_input: Vec<i128>,      // Size T
    write_lookup_output_to_rd: Vec<u8>,      // Size T
    write_pc_to_rd: Vec<u8>,                 // Size T
    should_branch: Vec<u8>,                  // Size T
    should_jump: Vec<u8>,                    // Size T
    rd_inc: Vec<i128>,                       // Size T
    ram_inc: Vec<i128>,                      // Size T

    // One-hot polynomial indices (for Twist/Shout)
    instruction_ra: [Vec<Option<u8>>; 16],   // 16 chunks × T entries
    bytecode_ra: Vec<Vec<Option<u8>>>,       // d chunks × T entries
    ram_ra: Vec<Vec<Option<u8>>>,            // d chunks × T entries
}
```

**Memory allocation**:
```rust
let witness_data = WitnessData {
    left_instruction_input: vec![0u64; trace_len],
    right_instruction_input: vec![0i128; trace_len],
    // ... all other fields initialized to size trace_len
};
```

**What we have now**: Empty vectors ready to be filled.

$$\text{WitnessData} = \{\vec{v}_1, \vec{v}_2, \ldots, \vec{v}_{50}\} \text{ where each } \vec{v}_i \in \mathbb{F}^T \text{ (all zeros)}$$

#### Step 10B: Extract Data from Each Cycle (Parallel Processing)

**File**: [jolt-core/src/zkvm/witness.rs:286-375](jolt-core/src/zkvm/witness.rs#L286)

**Mathematical operation**: For each cycle $j \in \{0, 1, \ldots, T-1\}$, extract relevant data and fill vectors.

```rust
trace.par_iter().enumerate().for_each(|(j, cycle)| {
    process_cycle(j, cycle, &witness_data, preprocessing);
});
```

**What `process_cycle` does** (concrete example for `ADD r3, r1, r2` at cycle $j = 5$):

```rust
fn process_cycle(j: usize, cycle: &Cycle, data: &WitnessData, ...) {
    // Extract from cycle state
    let (left_val, right_val) = cycle.instruction_inputs();     // (100, 250)
    let (rd_pre, rd_post) = cycle.rd_values();                  // (0, 350)
    let flags = cycle.circuit_flags();

    // Fill simple vectors
    data.left_instruction_input[j] = left_val;                  // 100
    data.right_instruction_input[j] = right_val;                // 250
    data.rd_inc[j] = rd_post - rd_pre;                         // 350
    data.write_lookup_output_to_rd[j] = flags.write_to_rd;     // 1
    data.should_branch[j] = flags.is_branch;                   // 0

    // Decompose instruction lookup into 16 chunks
    let lookup_index = compute_lookup_index(left_val, right_val, opcode);
    for chunk_i in 0..16 {
        let chunk_value = (lookup_index >> (4 * chunk_i)) & 0xF;  // Extract 4 bits
        data.instruction_ra[chunk_i][j] = Some(chunk_value as u8);
    }

    // Bytecode PC chunking
    let pc = cycle.program_counter();
    for chunk_i in 0..bytecode_d {
        let chunk_value = (pc >> (chunk_size * chunk_i)) & chunk_mask;
        data.bytecode_ra[chunk_i][j] = Some(chunk_value as u8);
    }

    // RAM address (if memory access)
    if let Some(addr) = cycle.ram_address() {
        data.ram_inc[j] = cycle.ram_value_change();            // written - read
        for chunk_i in 0..ram_d {
            let chunk_value = (addr >> (chunk_size * chunk_i)) & chunk_mask;
            data.ram_ra[chunk_i][j] = Some(chunk_value as u8);
        }
    }
}
```

**After parallel processing**, all vectors are filled:

$$\text{left\_instruction\_input} = (100, 250, 350, 0, 42, \ldots) \in \mathbb{F}^T$$
$$\text{rd\_inc} = (350, 100, 0, 42, 999, \ldots) \in \mathbb{F}^T$$
$$\text{instruction\_ra}[0] = (\text{Some}(10), \text{Some}(15), \text{Some}(5), \ldots) \text{ (indices for chunk 0)}$$

---

#### Step 10C: Convert Vectors to MLEs

**File**: [jolt-core/src/zkvm/witness.rs:386-430](jolt-core/src/zkvm/witness.rs#L386)

**Mathematical operation**: Transform raw data into multilinear polynomials.

 **IMPORTANT DISTINCTION**: Two different conversions happen in Step 10C!

---

### **Conversion Type 1: Simple Vectors (Already Dense) → MLE**

**Applies to**: `left_input`, `right_input`, `rd_inc`, `ram_inc`, `should_branch`, etc.

**From Step 10B**: These vectors are **already dense** (one value per cycle):
```rust
data.left_instruction_input = [100, 250, 350, 0, 42, ...]  // Size T, already dense!
data.rd_inc = [350, 100, 0, 42, 999, ...]                 // Size T, already dense!
```

**Step 10C conversion**: Just type conversion $\mathbb{Z} \to \mathbb{F}$ and wrapping in MLE:

```rust
// No sparse→dense conversion needed! Just field conversion
let coeffs: Vec<F> = data.left_instruction_input
    .iter()
    .map(|&x| F::from(x))    // u64 → field element
    .collect();

let mle_L = MultilinearPolynomial::from(coeffs);  // Wrap as MLE
```

**Mathematical definition** of the MLE $\widetilde{L}$:

Given **dense** vector $\vec{L} = (L_0, L_1, \ldots, L_{T-1}) \in \mathbb{F}^T$ where $T = 2^t$:

$$\widetilde{L}(X_1, \ldots, X_t) = \sum_{j \in \{0,1\}^t} L_j \cdot \text{eq}(X_1, \ldots, X_t; j_1, \ldots, j_t)$$

Where $\text{eq}(X; j)$ is the Lagrange basis polynomial:

$$\text{eq}(X_1, \ldots, X_t; j_1, \ldots, j_t) = \prod_{i=1}^{t} (X_i j_i + (1-X_i)(1-j_i))$$

**Properties**:

- **Input**: Dense vector of size $T$ (one value per cycle)
- **Output**: MLE with domain $\mathbb{F}^{\log T}$ (t-variate)
- On Boolean hypercube: $\widetilde{L}(j) = L_j$ for $j \in \{0,1\}^t$
- Extends uniquely to all $\mathbb{F}^t$ as multilinear polynomial

**Concrete example** with $T = 1024 = 2^{10}$ and $\vec{L} = (100, 250, 350, \ldots)$:

$$\widetilde{L}(0,0,0,0,0,0,0,0,0,0) = 100 \quad \text{(cycle 0)}$$
$$\widetilde{L}(1,0,0,0,0,0,0,0,0,0) = 250 \quad \text{(cycle 1)}$$
$$\widetilde{L}(0,1,0,0,0,0,0,0,0,0) = 350 \quad \text{(cycle 2)}$$

But also evaluable at any point, e.g.:
$$\widetilde{L}(0.5, 0.3, 0.7, \ldots) = \text{some field element}$$

**Memory**: Input and output both $O(T)$ - no expansion!

---

### **Conversion Type 2: Sparse One-Hot Indices → Dense Matrix → MLE**

**Applies to**: `instruction_ra`, `bytecode_ra`, `ram_ra`

 **KEY TRANSFORMATION: Sparse → Dense**

**Input from Step 10B** (sparse representation):
```rust
instruction_ra[chunk_i] = [Some(10), Some(15), None, Some(10), ...]  // Size T
```
This is **sparse**: Only stores the index, not the full one-hot vector. Memory: $O(T)$ bytes.

**Output in Step 10C** (dense representation):
```rust
matrix = [
  [0,0,0,...,0,1,0,...,0],  // Row 0: 256 entries, only index 10 is 1
  [0,0,0,...,0,0,1,...,0],  // Row 1: 256 entries, only index 15 is 1
  [0,0,0,...,0,0,0,...,0],  // Row 2: 256 entries, all zeros (no lookup)
  [0,0,0,...,0,1,0,...,0],  // Row 3: 256 entries, only index 10 is 1
]
```
This is **dense storage**: Full $T \times N$ matrix. Memory: $O(T \times N)$ bytes.

 **TERMINOLOGY CLARIFICATION - "Sparse" vs "Dense"**:

The matrix above is:

- **Dense storage representation**: We store ALL entries (all 256 values per row), not just indices
- **Mathematically sparse**: Most entries are zero (only one 1 per row)

**Two different meanings of "sparse"**:

1. **Storage sparsity** (data structure):
   - **Sparse storage**: `[Some(10), Some(15), None, ...]` - store only non-zero indices (what we have in 10B)
   - **Dense storage**: `[[0,...,1,...,0], [0,...,0,1,...,0], ...]` - store all values (what we create in 10C)

2. **Mathematical sparsity** (matrix property):
   - **Mathematically sparse**: Matrix has mostly zeros (true for one-hot matrices)
   - **Mathematically dense**: Matrix has mostly non-zeros

**Why this matters**: When we say "Step 10C converts sparse → dense", we mean:

- Converting from **sparse storage** (indices only) to **dense storage** (full matrix)
- The matrix remains **mathematically sparse** (mostly zeros) in both representations

**Memory impact**:

- 10B: 1 KB (storing only T indices)
- 10C: 256 KB (storing full T×256 matrix with mostly zeros)

**The conversion code**:

```rust
// Step 10B: Sparse (what we have)
let sparse_indices: Vec<Option<u8>> = data.instruction_ra[chunk_i];  // Size T

// Step 10C: Convert sparse → dense
let mut dense_matrix = vec![vec![0u8; TABLE_SIZE]; T];  // Size T × N
for (j, &maybe_k) in sparse_indices.iter().enumerate() {
    if let Some(k) = maybe_k {
        dense_matrix[j][k as usize] = 1;  // Expand: one index → full one-hot row
    }
}

// Convert dense matrix → MLE
let mle_ra = MultilinearPolynomial::from_2d_matrix(dense_matrix);
```

**Why convert sparse → dense?**

1. **Sumcheck requires dense MLE**: During sumcheck, we need to evaluate $\widetilde{\text{ra}}_i(r_1, \ldots, r_{\log T}, r'_1, \ldots, r'_{\log N})$ at arbitrary field points (not just Boolean). This requires the full dense polynomial representation.

2. **Can't evaluate sparse representation at arbitrary points**: The sparse index `Some(10)` only tells us about Boolean hypercube points. To evaluate at $\widetilde{\text{ra}}_i(0.73, 0.42, \ldots)$, we need the full Lagrange interpolation from the dense matrix.

3. **Memory trade-off**:
   - Sparse: $T$ entries (e.g., 1024 bytes)
   - Dense: $T \times N$ entries (e.g., 1024 × 256 = 256 KB)
   - But dense enables efficient sumcheck!

**Memory**: Input $O(T)$, output $O(T \times N)$ - **major expansion!**

**Mathematical definition** of the one-hot MLE $\widetilde{\text{ra}}_i(j, k)$:

Given $T \times N$ matrix $M$ where $M[j][k] \in \{0, 1\}$ and $\sum_k M[j][k] \leq 1$ (at most one "1" per row):

$$\widetilde{\text{ra}}_i(X_1, \ldots, X_{\log T}, Y_1, \ldots, Y_{\log N}) = \sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log N}} M[j][k] \cdot \text{eq}(X; j) \cdot \text{eq}(Y; k)$$

**Properties**:

- Domain: $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ (two-dimensional)
- On hypercube: $\widetilde{\text{ra}}_i(j, k) = 1$ if cycle $j$ accessed index $k$, else $0$
- One-hot property: $\sum_{k=0}^{N-1} \widetilde{\text{ra}}_i(j, k) \leq 1$ for each $j$

**Concrete example** for $T = 4, N = 256$ (chunk table size):

If cycles looked up: cycle 0 → index 10, cycle 1 → index 15, cycle 2 → no lookup, cycle 3 → index 10:

$$\widetilde{\text{ra}}_0(0, 10) = 1, \quad \widetilde{\text{ra}}_0(0, k \neq 10) = 0$$
$$\widetilde{\text{ra}}_0(1, 15) = 1, \quad \widetilde{\text{ra}}_0(1, k \neq 15) = 0$$
$$\widetilde{\text{ra}}_0(2, k) = 0 \quad \forall k \quad \text{(no lookup)}$$
$$\widetilde{\text{ra}}_0(3, 10) = 1, \quad \widetilde{\text{ra}}_0(3, k \neq 10) = 0$$

---

### **Summary: Two Different Conversions in Step 10C**

| Property | Conversion Type 1 | Conversion Type 2 |
|----------|-------------------|-------------------|
| **Applies to** | Simple vectors (left_input, rd_inc, etc.) | One-hot indices (instruction_ra, bytecode_ra, ram_ra) |
| **Step 10B output** | Dense: `[100, 250, 350, ...]` | Sparse: `[Some(10), Some(15), None, ...]` |
| **Conversion needed?** |  No (already dense) |  Yes (sparse → dense) |
| **Step 10C operation** | Type conversion: $\mathbb{Z} \to \mathbb{F}$ | Expand to matrix: $T \to T \times N$ |
| **Memory change** | $O(T) \to O(T)$ (no expansion) | $O(T) \to O(T \times N)$ (major expansion!) |
| **Example size** | 1 KB → 1 KB | 1 KB → 256 KB |
| **Output MLE domain** | $\mathbb{F}^{\log T}$ (univariate in cycle) | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ (bivariate) |

**Key insight**:

- **Type 1**: Already have full information, just wrap as MLE
- **Type 2**: Have compact index, must expand to full one-hot matrix before MLE

---

**Result of Step 10C**: HashMap of MLEs

The complete set of ~50 witness polynomials created from the execution trace:

#### Type 1 MLEs: Simple Witness Vectors (Conversion Type 1)

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `LeftInstructionInput` | $\widetilde{L}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Left operand (rs1 or PC) at cycle $j$ |
| `RightInstructionInput` | $\widetilde{R}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Right operand (rs2 or imm) at cycle $j$ |
| `WriteLookupOutputToRD` | $\widetilde{w}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: should write result to rd? |
| `WritePCtoRD` | $\widetilde{w}_{\text{pc}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: should write PC to rd (JAL/JALR)? |
| `ShouldBranch` | $\widetilde{b}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: is this a branch instruction? |
| `ShouldJump` | $\widetilde{j}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: is this a jump instruction? |
| `RdInc` | $\widetilde{\Delta}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Register write increment at cycle $j$ |
| `RamInc` | $\widetilde{\Delta}_{\text{ram}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | RAM write increment at cycle $j$ |

**Conversion**: Already dense in Step 10B → Just type convert $\mathbb{Z} \to \mathbb{F}$ in Step 10C

**Memory**: $O(T)$ per polynomial (no expansion)

**Example for $T = 1024$**: Each polynomial stores 1024 field elements ≈ 32 KB

---

#### Type 2 MLEs: One-Hot Polynomials (Conversion Type 2)

**A) Instruction Lookup Addresses** (16 chunks for 128-bit lookup index):

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `InstructionRa(0)` | $\widetilde{\text{ra}}_0(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 0 of instruction lookup (bits 0-7) |
| `InstructionRa(1)` | $\widetilde{\text{ra}}_1(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 1 of instruction lookup (bits 8-15) |
| `InstructionRa(2)` | $\widetilde{\text{ra}}_2(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 2 of instruction lookup (bits 16-23) |
| ... | ... | ... | ... | ... |
| `InstructionRa(15)` | $\widetilde{\text{ra}}_{15}(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 15 of instruction lookup (bits 120-127) |

**Table size**: $N = 2^8 = 256$ entries per chunk (8-bit chunks)

**Total**: 16 polynomials for instruction lookups

---

**B) Bytecode Lookup Addresses** ($d$ chunks for program counter):

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `BytecodeRa(0)` | $\widetilde{\text{bc}}_0(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K_c}$ | $T \times K_c$ | Chunk 0 of PC address |
| `BytecodeRa(1)` | $\widetilde{\text{bc}}_1(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K_c}$ | $T \times K_c$ | Chunk 1 of PC address |
| ... | ... | ... | ... | ... |
| `BytecodeRa(d-1)` | $\widetilde{\text{bc}}_{d-1}(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K_c}$ | $T \times K_c$ | Chunk $d-1$ of PC address |

**Chunk table size**: $K_c = K^{1/d}$ where $K$ = bytecode size, $d$ = chunking parameter (dynamically chosen)

**Total**: $d$ polynomials for bytecode lookups (typically 2-4 chunks)

**Example**: If bytecode has 4096 instructions and $d = 2$, then $K_c = \sqrt{4096} = 64$ entries per chunk

---

**C) RAM Address Lookups** ($d$ chunks for memory addresses):

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `RamRa(0)` | $\widetilde{\text{mem}}_0(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M_c}$ | $T \times M_c$ | Chunk 0 of RAM address |
| `RamRa(1)` | $\widetilde{\text{mem}}_1(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M_c}$ | $T \times M_c$ | Chunk 1 of RAM address |
| ... | ... | ... | ... | ... |
| `RamRa(d-1)` | $\widetilde{\text{mem}}_{d-1}(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M_c}$ | $T \times M_c$ | Chunk $d-1$ of RAM address |

**Chunk table size**: $M_c = M^{1/d}$ where $M$ = memory size, $d$ = chunking parameter (dynamically chosen to keep $M_c = 2^8 = 256$)

**Total**: $d$ polynomials for RAM address lookups (typically 8 chunks for 64-bit addresses)

**Example**: If RAM has $M = 2^{64}$ addresses and $d = 8$, then $M_c = 2^8 = 256$ entries per chunk

---

**Conversion**: Sparse indices in Step 10B → Expand to dense $T \times N$ matrix in Step 10C → MLE

**Memory**: $O(T \times N)$ per polynomial (major expansion!)

**Example for $T = 1024, N = 256$**: Each polynomial stores 262,144 field elements ≈ 8 MB

---

#### Complete Summary: All ~50 MLEs from Step 10C

```rust
HashMap<CommittedPolynomial, MultilinearPolynomial<F>> {
    // Type 1: 8 simple witness polynomials (8 × 32 KB = 256 KB total)
    LeftInstructionInput => ˜L(j),                     // t-variate
    RightInstructionInput => ˜R(j),                    // t-variate
    WriteLookupOutputToRD => ˜w_rd(j),                 // t-variate
    WritePCtoRD => ˜w_pc(j),                           // t-variate
    ShouldBranch => ˜b(j),                             // t-variate
    ShouldJump => ˜j(j),                               // t-variate
    RdInc => ˜\Delta_rd(j),                                 // t-variate
    RamInc => ˜\Delta_ram(j),                               // t-variate

    // Type 2: 16 instruction lookup polynomials (16 × 8 MB = 128 MB total)
    InstructionRa(0) => ˜ra_0(j, k),                    // (t+8)-variate
    InstructionRa(1) => ˜ra_1(j, k),                    // (t+8)-variate
    // ... (14 more)
    InstructionRa(15) => ˜ra_1_5(j, k),                  // (t+8)-variate

    // Type 2: ~3 bytecode lookup polynomials (3 × 8 MB = 24 MB total)
    BytecodeRa(0) => ˜bc_0(j, k),                       // (t+log K_c)-variate
    BytecodeRa(1) => ˜bc_1(j, k),                       // (t+log K_c)-variate
    // ... (d-2 more)

    // Type 2: ~8 RAM address polynomials (8 × 8 MB = 64 MB total)
    RamRa(0) => ˜mem_0(j, k),                           // (t+8)-variate
    RamRa(1) => ˜mem_1(j, k),                           // (t+8)-variate
    // ... (6 more)
    RamRa(7) => ˜mem_7(j, k),                           // (t+8)-variate
}

// Total: ~35 polynomials, ~220 MB of MLE data (before commitment)
```

**Memory hierarchy**:

- **Step 10B**: ~2 MB (sparse indices + simple vectors)
- **Step 10C (MLEs)**: ~220 MB (after dense expansion)
- **After Dory commitment**: ~7 KB (35 commitments × 192 bytes each)

**The compression**: 220 MB of witness data → 7 KB of commitments!

**Mathematical summary**:

$$\boxed{\text{Execution Trace } (s_0, \ldots, s_{T-1}) \xrightarrow{\text{Step 10A-C}} \text{~35 MLEs } \{\widetilde{P}_1, \ldots, \widetilde{P}_{35}\}}$$

Where each $\widetilde{P}_i$ is either:

- **Type 1**: Simple MLE $\mathbb{F}^{\log T} \to \mathbb{F}$ (already dense in Step 10B) - 8 polynomials
- **Type 2**: One-hot MLE $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N} \to \mathbb{F}$ (sparse → dense in Step 10C) - 27 polynomials

---

## From MLEs to Commitments: The Final Step

After generating all MLEs, we commit to them using Dory PCS. This is where the polynomials become cryptographic objects.

### Mathematical Object: Dory Commitment

**From Theory/Dory.md (Section 2.2)**:

For a multilinear polynomial $f$ with $N = 2^{\nu}$ evaluations, Dory commits via:

1. **Reshape to matrix**: $N$ coefficients → $\sqrt{N} \times \sqrt{N}$ matrix $M$

2. **Layer 1 (Pedersen to rows)**:
   $$V_i = \sum_{j=1}^{\sqrt{N}} M_{i,j} G_{1,j} + r_i H_1 \in \mathbb{G}_1$$

3. **Layer 2 (AFGHO to vector)**:
   $$C_M = \prod_{i=1}^{\sqrt{N}} e(V_i, G_{2,i}) \cdot e(H_1, H_2)^{r_{fin}} \in \mathbb{G}_T$$

**Result**: Single element $C_M \in \mathbb{G}_T$ (192 bytes) commits to entire polynomial!

### Concrete Example: Committing LeftInstructionInput

**Witness data**:
```
left_instruction_input = [100, 250, 350, 0, ...] (T = 1024 cycles)
MLE: L̃(j) : F^10 → F where L̃(0,0,...,0,0) = 100, L̃(1,0,...,0,0) = 250, etc.
```

**Dory commitment** (T = 1024, K = 16, total size = 16,384):
```
Matrix dimensions: √16,384 × √16,384 = 128 × 128

Step 1: Reshape L̃ evaluations into 128×128 matrix M_L

Step 2 (Layer 1): Create 128 Pedersen commitments (one per row):
  V_0 = M_L[0,0]·G_{1,0} + M_L[0,1]·G_{1,1} + ... + M_L[0,127]·G_{1,127} + r_0·H_1 \in G_1
  V_1 = M_L[1,0]·G_{1,0} + M_L[1,1]·G_{1,1} + ... + M_L[1,127]·G_{1,127} + r_1·H_1 \in G_1
  ...
  V_127 = M_L[127,0]·G_{1,0} + ... + M_L[127,127]·G_{1,127} + r_127·H_1 \in G_1

  Result: 128 commitments (V_0, ..., V_127), each in G_1

Step 3 (Layer 2): AFGHO compresses 128 G_1 elements into single G_T element:
  C_L = e(V_0, G_{2,0}) · e(V_1, G_{2,1}) · ... · e(V_127, G_{2,127}) · e(H_1, H_2)^r_fin \in G_T

  Result: ONE commitment C_L \in G_T ≈ 192 bytes (compressed)
```

**Key insight**:

- **Layer 1** creates 128 intermediate commitments (one per matrix row)
- **Layer 2** (AFGHO) compresses those 128 commitments into 1 final commitment
- Each MLE polynomial gets exactly **ONE final commitment** $C \in \mathbb{G}_T$

---

**For ALL ~35 polynomials**:
```
Per polynomial: 1 G_T element = 192 bytes
Total commitments sent to verifier: 35 × 192 bytes ≈ 7 KB

Compare to:

- Raw MLEs: ~220 MB (dense matrix data)
- Raw trace: 1024 cycles × ~100 bytes/cycle ≈ 100 KB
- Commitments: 7 KB

Compression ratio: ~30,000× compared to MLEs! ~14× compared to raw trace!
```

### What Gets Sent to Verifier?

**Stage 0 (Commitment phase)**:

Prover → Verifier:
$$\{C_L, C_R, C_{\Delta_{rd}}, C_{\Delta_{ram}}, C_{ra_0}, \ldots, C_{ra_{15}}, C_{bc_0}, \ldots\} \subset \mathbb{G}_T^{50}$$

**Size**: ~10 KB of commitments (vs 100 KB raw trace)

**Binding property**: Computationally infeasible to find different polynomial $f' \neq f$ with same commitment $C$

**Hiding property**: Commitment reveals nothing about polynomial values (due to random blinding factors $r_i, r_{fin}$)

### Opening Proofs (Stage 5)

After sumcheck reduces verification to single-point evaluations, verifier needs to check:

$$\text{For random } r \in \mathbb{F}^{\log T}: \quad \widetilde{L}(r) \stackrel{?}{=} y_L$$

**Dory opening proof** (from Theory/Dory.md, Section 4):

1. Prover runs $\log(\sqrt{N})$ rounds of recursive halving (Dory-Reduce)
2. Each round: Prover sends 6 group elements
3. Final round: Scalar-Product protocol (constant size)

**Proof size**: $6 \cdot \log(\sqrt{N})$ group elements ≈ 18 KB per polynomial

**Verification time**: $O(\log N)$ (logarithmic in polynomial size!)

**Batching optimization**: When opening multiple polynomials at the same point, amortized cost drops to $O(1)$ per polynomial!

### Summary: Mathematical Objects at Each Stage

| Stage | Object Type | Size | Security |
|-------|-------------|------|----------|
| **Trace** | $\text{Vec<Cycle>}$ | 10KB-10MB | N/A (plaintext) |
| **MLE Witness** | $\{\widetilde{L}, \widetilde{R}, \ldots\} : \mathbb{F}^{\log T} \to \mathbb{F}$ | Same as trace | N/A (plaintext) |
| **Commitments** | $\{C_L, C_R, \ldots\} \subset \mathbb{G}_T$ | ~10 KB | Binding + Hiding |
| **Opening Proofs** | Dory proofs | ~18 KB each | Soundness $2^{-128}$ |

**The transformation**:
```
Execution (100 KB trace)
   ↓ [Extract witness data]
MLEs (100 KB polynomials)
   ↓ [Dory commit]
Commitments (10 KB in G_T)
   ↓ [Sumcheck reduces to point evaluations]
Opening proofs (18 KB per opening)
   ↓ [Verifier checks]
Accept/Reject (single bit!)
```

**Key property**: Verifier never sees the full trace, only:

1. Commitments ($\mathbb{G}_T$ elements, ~10 KB)
2. Claimed evaluations (field elements, ~1 KB)
3. Opening proofs (logarithmic size, ~18 KB each)

**Total proof size**: ~200-500 KB (vs 100+ KB trace, and verifies in $O(\log T)$ time!)

---

## Complete Execution and Witness Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ User calls: prove_{fn_name}(input)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ Macro-generated prove function                                  │
│  ├─> Serialize inputs to bytes                                  │
│  └─> Call JoltRV64IMAC::prove()                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ JoltRV64IMAC::prove()                                           │
│  ├─> Setup memory configuration                                 │
│  └─> Call guest::program::trace()  ◄─── THE BIG EMULATION      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ guest::program::trace()                                         │
│  ├─> setup_emulator_with_backtraces()                           │
│  │   - Load ELF into memory                                     │
│  │   - Initialize CPU (registers, PC, memory)                   │
│  │   - Create LazyTraceIterator                                 │
│  │                                                               │
│  └─> lazy_trace_iter.collect()                                  │
│      ┌──────────────────────────────────────────────────┐       │
│      │ FOR EACH INSTRUCTION UNTIL HALT:                 │       │
│      │  1. Fetch: Read instruction from PC              │       │
│      │  2. Decode: Determine instruction type           │       │
│      │  3. Execute & Trace:                             │       │
│      │     a. Capture pre-execution state (registers)   │       │
│      │     b. Execute instruction (modify CPU)          │       │
│      │     c. Capture post-execution state              │       │
│      │     d. Create Cycle struct                       │       │
│      │     e. Push to trace Vec                         │       │
│      │  4. Update PC                                     │       │
│      │  5. Check for halt (ECALL or error)              │       │
│      └──────────────────────────────────────────────────┘       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ Output:                                                          │
│  - trace: Vec<Cycle> (e.g., 10,000 cycles)                      │
│  - final_memory_state: Memory                                   │
│  - program_io: JoltDevice (inputs/outputs/panic)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ Back in JoltRV64IMAC::prove()                                   │
│  ├─> Pad trace to power of 2                                    │
│  └─> Create StateManager with trace                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ JoltDAG::prove() (Part 3 starts here!)                          │
│  ├─> generate_and_commit_polynomials()                          │
│  │   ┌──────────────────────────────────────────────────┐       │
│  │   │ WITNESS GENERATION:                              │       │
│  │   │  1. CommittedPolynomial::generate_witness_batch()│       │
│  │   │     - Create WitnessData (vectors)               │       │
│  │   │     - Process each cycle in parallel:            │       │
│  │   │       ├─> Extract instruction operands           │       │
│  │   │       ├─> Extract circuit flags                  │       │
│  │   │       ├─> Extract register increments            │       │
│  │   │       ├─> Decompose lookup queries               │       │
│  │   │       └─> Record memory accesses                 │       │
│  │   │     - Convert vectors → MLEs                     │       │
│  │   │                                                   │       │
│  │   │  2. PCS::batch_commit()                          │       │
│  │   │     - Commit to all MLEs using Dory              │       │
│  │   │     - Generate opening hints                     │       │
│  │   └──────────────────────────────────────────────────┘       │
│  │                                                               │
│  └─> Run 5 stages of sumchecks (Part 3!)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Outputs Summary

### From Emulation (Part 2a):

**Vec<Cycle>**: Complete execution trace
```rust
[
    Cycle::LOAD(RISCVCycle { /* load 100 into r1 */ }),
    Cycle::LOAD(RISCVCycle { /* load 250 into r2 */ }),
    Cycle::ADD(RISCVCycle {
        instruction: ADD { rd: 3, rs1: 1, rs2: 2 },
        register_state: { rs1_val_pre: 100, rs2_val_pre: 250, rd_val_post: 350 },
        ram_access: (),
    }),
    // ... more cycles ...
]
```

### From Witness Generation (Part 2b):

**HashMap<CommittedPolynomial, MLE>**: Polynomials ready for proving
```rust
{
    LeftInstructionInput: MLE([100, 250, ...]),
    RightInstructionInput: MLE([250, ...]),
    WriteLookupOutputToRD: MLE([1, 1, ...]),
    RdInc: MLE([100, 250, 350, ...]),
    InstructionRa(0): OneHotMLE([[0,0,1,0,...], ...]),
    InstructionRa(1): OneHotMLE([[0,1,0,0,...], ...]),
    // ... 16 instruction chunks + d bytecode chunks + d RAM chunks ...
}
```

These MLEs are:

1. **Committed** using Dory PCS (Stage 0 of proving)
2. **Proven** via sumchecks in Stages 1-4
3. **Opened** via batched opening proof in Stage 5

---

## Connection to Theory: Lookup-Centric Architecture

From [Theory/Jolt.md:243-252](Theory/Jolt.md#L243):

> Jolt combines all possible operations into a single, giant lookup table, `JOLT_V`:
> $$ \text{JOLT\_V}(\text{opcode}, a, b) = (c, \text{flags}) = f_{op}(a, b) $$
>
> A proof of execution for a program trace of length $m$ becomes a sequence of $m$ lookup claims:
> - At step 1: $(c_1, \text{flags}_1) = \text{JOLT\_V}(\text{opcode}_1, a_1, b_1)$
> - At step 2: $(c_2, \text{flags}_2) = \text{JOLT\_V}(\text{opcode}_2, a_2, b_2)$
> - ...

**How witness generation enables this**:

For each cycle, we extract:

- **Opcode**: Implicit in `Cycle` enum variant (e.g., `Cycle::ADD`)
- **Inputs** `(a, b)`: Captured in `left_instruction_input` and `right_instruction_input` MLEs
- **Output** `c`: Captured in `rd_val_post` (verified via R1CS wiring)
- **Lookup indices**: Decomposed and stored in `InstructionRa(0)..InstructionRa(15)` MLEs

The witness generation **transforms** execution trace into **lookup queries**!

---

## Performance Considerations

### Why Parallel Processing?

```rust
trace.par_iter().enumerate().for_each(|(cycle_idx, cycle)| {
    process_cycle(cycle_idx, cycle, &shared_data, preprocessing);
});
```

**Rayon** parallelizes witness generation:

- Each cycle processed independently (embarrassingly parallel!)
- Typical trace: 10,000 - 1,000,000 cycles
- Multi-core speedup: 4-8x faster on modern CPUs

### Memory Efficiency

**LazyTraceIterator**: Generates cycles on-demand

- Avoids storing entire program state history
- Only current cycle + final memory kept in RAM
- Critical for large programs (> 100K cycles)

**Option**: `trace_to_file()` for massive traces

- Streams cycles to disk instead of Vec
- Enables proving programs with millions of cycles
- Trade-off: I/O overhead vs memory savings

---

## Debugging and Introspection

### Cycle Tracking

**File**: [jolt-sdk/src/lib.rs] (guest side)

```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn my_function(n: u32) -> u64 {
    start_cycle_tracking("fibonacci_loop");
    let result = fibonacci(n);
    end_cycle_tracking("fibonacci_loop");
    result
}
```

**Output** (during emulation):
```
Cycle tracking report:
  fibonacci_loop: 1,234 real cycles, 1,234 virtual cycles
```

Helps identify bottlenecks in guest programs!

### Panic Backtraces

**File**: [tracer/src/emulator/cpu.rs:114-116](tracer/src/emulator/cpu.rs#L114)

```rust
pub struct Cpu {
    // ...
    call_stack: VecDeque<CallFrame>,  // Circular buffer of depth 32
}
```

When guest panics:

- Call stack unwound from CPU state
- Symbols resolved using ELF file
- Rust-style backtrace printed

Example:
```
Guest program panicked!
Backtrace:
  0: guest::my_function at guest/src/lib.rs:42
  1: guest::helper at guest/src/lib.rs:18
  2: _start at ...
```

---

## Summary: Part 2 Checklist

**Emulation** 

- [x] Load ELF into memory
- [x] Initialize RISC-V CPU (32 registers, PC, CSRs)
- [x] Fetch-decode-execute loop
- [x] Capture pre/post state for each instruction
- [x] Handle memory loads/stores
- [x] Track program I/O (inputs, outputs, panic)
- [x] Generate `Vec<Cycle>` (execution trace)

**Witness Generation** 

- [x] Process trace in parallel
- [x] Extract instruction operands (left/right inputs)
- [x] Extract circuit flags (branch, jump, write)
- [x] Compute register increments (Twist)
- [x] Decompose lookup queries into chunks (Shout)
- [x] Record memory accesses (Twist)
- [x] Convert vectors → MLEs
- [x] Convert one-hot indices → MLEs

**Key Outputs**:

- `Vec<Cycle>`: Complete execution witness (~10KB - 10MB depending on program)
- `HashMap<CommittedPolynomial, MLE>`: 40-60 polynomials ready for proving
- `Memory`: Final RAM state (for Twist verification)
- `JoltDevice`: I/O data (for public inputs)

**Next Step**: Part 3 (Proof Generation) uses these MLEs to construct the proof!

---

## Additional Resources

- [tracer/src/emulator/cpu.rs](tracer/src/emulator/cpu.rs): Full CPU emulator
- [tracer/src/instruction/mod.rs](tracer/src/instruction/mod.rs): Instruction traits and enum
- [tracer/src/instruction/add.rs](tracer/src/instruction/add.rs): Example instruction
- [jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs): Witness generation
- [Theory/Jolt.md:155-260](Theory/Jolt.md#L155): Execution trace theory