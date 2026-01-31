- **Handoff reason:** phase complete (optimized `BN254_GT_MUL`; dory-pcs optimization reverted)
- **Summary:** Optimized `BN254_GT_MUL` to use Fq6-layer Karatsuba (`fq12_mul_schoolbook_memmem_to_mem`) by refactoring the builder to stream the LHS from memory, resolving register pressure. This reduced virtual cycles for verification by ~1.8%. Attempted to remove conversion overhead in `dory-pcs` via `unsafe` casting, but measured negligible improvement, so reverted to the safe implementation.
- **Goal and scope:** Reduce zkVM **virtual cycles** (primary) for recursive verifier’s Dory-heavy verification by accelerating GT arithmetic.
- **Current state:** building; tests passing; repo clean except untracked `output/`.

- **Work completed:**
  - **`BN254_GT_MUL` Optimization (Fq6 Karatsuba)**
    - Replaced `fq12_mul_regmem_to_mem` (3x Fq6 Schoolbook, register-heavy) with `fq12_mul_schoolbook_memmem_to_mem` (4x Fq6 Karatsuba, memory-streaming).
    - Reduced underlying Fq2 multiplication count from 27 to 24 per GT mul.
    - Refactored `bn254_gt_mul_sequence_builder` to stream LHS from memory, freeing 48 registers.
    - Optimized register allocation (`Fq2WorkTight`, `fq2_mul_karatsuba_clobber`) to fit within the 128 virtual register limit (using 86 regs).
    - **File:** `jolt-inlines/dory-gt-exp/src/sequence_builder.rs`
  - **Verification**
    - Added `gt_mul_matches_arkworks_mul` test case to `sequence_builder.rs` (lines ~2630).
    - Verified correctness against `arkworks` reference.
  - **Investigation: dory-pcs conversion overhead**
    - Attempted to replace `fq12_to_limbs_mont` copies with `unsafe` pointer casting in `third-party/dory-pcs/src/backends/arkworks/ark_group.rs`.
    - Benchmark showed negligible impact (noise level) because GT operation count (~140 ops) is small relative to the total cost.
    - **Decision:** Reverted to safe implementation.
  - **Docs / perf “stick”**
    - Updated `BN254_GT_INLINES.md` with post-optimization numbers.

- **Files modified:**
  - `jolt-inlines/dory-gt-exp/src/sequence_builder.rs`: Implemented `fq12_mul_schoolbook_memmem_to_mem`, `copy_fq6`, updated `bn254_gt_mul_sequence_builder` to use new path, added tests.
  - `BN254_GT_INLINES.md`: Updated benchmark table.

- **Context files:**
  - `tracer/src/emulator/cpu.rs`: Cycle tracking semantics (RV64IMAC vs virtual).
  - `third-party/dory-pcs/src/backends/arkworks/ark_group.rs`: Dory GT integration.
  - `jolt-core/src/poly/commitment/dory/commitment_scheme.rs`: Guest windowed exponentiation loop.

- **Key decisions and rationale:**
  - **Fq6 Karatsuba via memory streaming:** We accepted the tradeoff of higher memory traffic (loading LHS from memory) to enable the register-intensive Karatsuba algorithm at the Fq6 level. The virtual cycle savings from fewer multiplications outweighed the instruction overhead of loads.
  - **Skip dory-pcs unsafe optimization:** The risk/maintenance burden of `unsafe` pointer casting for layout assumptions was not justified by the negligible performance gain.

- **Blockers / Errors:** None currently.

- **Open questions / Risks:**
  - **Layout assumptions:** If `arkworks` changes `Fq12` internal layout, future unsafe optimizations (if revisited) would break.
  - **Register pressure:** We are close to the limit (86/128 regs) for `GT_MUL`. Future optimizations like `GT_EXP` inline must be careful.

- **Cleanup needed:**
  - `output/` directory contains benchmark artifacts (ignored by git).

- **Tests and commands run:**
  - `cargo test -p jolt-inlines-dory-gt-exp --features host` -> **PASS**
  - `RUST_LOG=info cargo run -p recursion --release trace --example fibonacci --embed` -> **PASS**, used for benchmarks.

- **Next steps:**
  1. **Implement `BN254_GT_INV` (or Conjugation) Inline**:
     - **Goal:** Enable signed-digit window methods (wNAF) for exponentiation.
     - **Strategy:** In the cyclotomic subgroup, inversion is just conjugation ($c_1 \leftarrow -c_1$). This is extremely cheap (just negation of 24 limbs).
     - **Constraint:** Requires verifying inputs are in the subgroup or ensuring the caller guarantees it.
  2. **Implement Signed Window (wNAF) Exponentiation**:
     - **Goal:** Reduce the number of GT multiplications in `dory_gt_exp_ops`.
     - **Strategy:** Update `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` to use wNAF (using the new cheap inversion inline for negative digits). This could reduce `GT_MUL` count by ~10-20% for the same window size.
  3. **ABI Optimization (Explicit Scratch Pointer)**:
     - **Goal:** Allow `GT_MUL` output to alias `rhs` input, and potentially `GT_EXP` output to alias `base`.
     - **Strategy:** Modify the inline instruction format or reuse a register (e.g. `rs2` if unused in SQR, or a new register) to pass an explicit scratch buffer pointer, separating "output" from "scratch". Currently `out_ptr` is used as scratch, preventing aliasing.
  4. **Monolithic `GT_EXP` Inline (Long-term)**:
     - **Goal:** Move the entire exponentiation loop into the inline to save guest control-flow overhead.
     - **Blocker:** Requires changing `virtual_sequence_remaining` from `u16` to `u32` in the tracer/preprocessor to support sequences > 65535 instructions.

- **How to resume:**
  1. Check baseline: `RUST_LOG=info cargo run -p recursion --release trace --example fibonacci --embed`
  2. Start implementing `BN254_GT_INV` (conjugation) inline in `jolt-inlines/dory-gt-exp`.
