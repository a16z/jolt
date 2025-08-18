# Analysis of Sumchecks Contributing to the Final Batched Opening

This document outlines the sumchecks within the Jolt proof system that **contribute opening claims** to the final `reduce_and_prove` procedure. The `ProverOpeningAccumulator` gathers these claims (IOUs) across various stages. The `reduce_and_prove` function, executed after all stages are complete, then proves all these collected claims simultaneously in a single, batched polynomial commitment opening proof.

It is critical to distinguish between the sumcheck instances themselves (e.g., `SpartanOuter`) and the final batched opening. The former generate claims; the latter proves them.

Below is a list of each `SumcheckId` that contributes opening claims, along with the degree of its own internal sumcheck and the formula it proves.

---

### Core R1CS and PC Sumchecks (`SpartanDag`)

These sumchecks form the foundation of the proof, ensuring the main R1CS constraints are met and the program counter behaves correctly.

#### 1. `SumcheckId::SpartanOuter`
- **Source**: Stage 1
- **Degree**: **3**
- **Formula**: `sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0`
- **Purpose**: Proves the main R1CS satisfaction equation.
- **Contributes Opening Claims For**: All committed R1CS input polynomials, and the virtual polynomials `SpartanAz`, `SpartanBz`, `SpartanCz`, and all non-committed R1CS inputs. These claims are generated here and proven in the final batched proof.

#### 2. `SumcheckId::SpartanShift`
- **Source**: Stage 3 (via `PCSumcheck`)
- **Degree**: **2**
- **Formula**: `NextPC(r) + ... = sum_t (PC(t) + ...) * eq_plus_one(r, t)`
- **Purpose**: Proves the correct continuity of the Program Counter.
- **Contributes Opening Claims For**: `VirtualPolynomial::UnexpandedPC`, `VirtualPolynomial::PC`, `VirtualPolynomial::OpFlags(CircuitFlags::IsNoop)`.

---

### Memory and Registers Sumchecks (`RamDag`, `RegistersDag`)

These sumchecks verify the correct behavior of the virtual RAM and CPU registers.

#### 3. `SumcheckId::RamReadWriteChecking`
- **Source**: Stage 2
- **Degree**: **3**
- **Formula**: `sum_{k,j} eq(r', (k,j)) * ra * (val' - (val + inc)) = 0`
- **Purpose**: Enforces memory consistency by checking that `val(t+1) = val(t) + increment`.
- **Contributes Opening Claims For**: `VirtualPolynomial::RamVal`, `VirtualPolynomial::RamRa`, and `CommittedPolynomial::RamInc`.

#### 4. `SumcheckId::RegistersReadWriteChecking`
- **Source**: Stage 2
- **Degree**: **3**
- **Formula**: `sum_{reg,j} eq(r', (reg,j)) * a * (val' - (val + inc)) = 0`
- **Purpose**: Enforces register consistency, analogous to the RAM check.
- **Contributes Opening Claims For**: `VirtualPolynomial::RegistersVal`, `VirtualPolynomial::RegistersA`, and `CommittedPolynomial::RegsInc`.

#### 5. `SumcheckId::RamValEvaluation`
- **Source**: Stage 3
- **Degree**: **4**
- **Formula**: Verifies claimed memory values against read/write timestamps and initial/final memory states.
- **Purpose**: Ensures that the values used in memory lookups are consistent with the chronological history.
- **Contributes Opening Claims For**: `VirtualPolynomial::RamValFinal`.

#### 6. `SumcheckId::RegistersValEvaluation`
- **Source**: Stage 3
- **Degree**: **2**
- **Formula**: `sum_i (v_i - (a_i * v_final_i + (1 - a_i) * v_init_i)) = 0`
- **Purpose**: Verifies that the final values of registers are correctly derived from their initial values and the sequence of writes.
- **Contributes Opening Claims For**: `VirtualPolynomial::RegistersValFinal`.

---

### Low-Level Primitive Sumchecks (`RamDag` - Stage 4)

These Stage 4 sumchecks prove fundamental properties (like booleanity) of helper polynomials.

#### 7. `SumcheckId::RamBooleanity`
- **Source**: Stage 4
- **Degree**: **2**
- **Formula**: `sum_x p(x) * (1 - p(x)) = 0`
- **Purpose**: Proves that the polynomial `p` (in this case, `RamA`) is boolean-valued.
- **Contributes Opening Claims For**: `VirtualPolynomial::RamA`.

#### 8. `SumcheckId::RamHammingWeight`
- **Source**: Stage 4
- **Degree**: **3**
- **Formula**: `sum_x (h(x) - sum_i 2^i * b_i(x)) = 0`
- **Purpose**: Verifies that `h(x)` is the correct Hamming weight for a value decomposed into bits `b_i(x)`.
- **Contributes Opening Claims For**: `VirtualPolynomial::RamH`.

#### 9. `SumcheckId::RamHammingBooleanity`
- **Source**: Stage 4
- **Degree**: **2**
- **Formula**: `sum_x p(x) * (1 - p(x)) = 0`
- **Purpose**: Proves that the bit decomposition polynomials `p` (in this case, `RamB`) are boolean.
- **Contributes Opening Claims For**: `VirtualPolynomial::RamB`.

#### 10. `SumcheckId::RamOutputCheck`
- **Source**: Stage 4
- **Degree**: **2**
- **Formula**: `sum_x (claimed_output(x) - final_memory(x)) * eq(x, output_address) = 0`
- **Purpose**: Verifies that the public output matches the final state of the corresponding memory region.
- **Contributes Opening Claims For**: `VirtualPolynomial::RamOutput`.

---

### Sumchecks With No Opening Contributions

Many sumchecks **do not** contribute opening claims. This includes `SpartanInner` and all sumchecks from `LookupsDag` and `BytecodeDag`. These protocols prove relations between polynomials whose openings are already being proven elsewhere, so they don't need to add redundant claims to the accumulator.
