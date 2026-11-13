# Spec: Memory and Termination Guards

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @quangvdao                     |
| Created     | 2026-11-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt's tracer and constraints differ from a plain RISC-V machine in four places that a formal model has to carry as premises. Nothing stops a store from changing the program's own code words, although Jolt fetches instructions from committed bytecode rather than from memory. The panic and termination words behave as a device rather than as memory. A witness may stop after any jump, not only at a self-jump. A memory row may select no RAM address at all, and the tracer returns zero for loads below the lowest mapped address. This spec closes all four gaps at the lowest available cost. It reuses the existing RAM read-write sumcheck, the spare uni-skip slot in the outer sumcheck, and public preprocessing, and it adds no new commitment and no new sumcheck round.

## Intent

### Goal

Make every accepted Jolt proof correspond to a run in which code, program input, trusted advice, untrusted advice and the stack canary preserve their initial values, the I/O flag words are ordinary memory, the trace ends only at a repeatable self-jump, and every memory row accesses one mapped RAM word.

The changes are:

1. **Value-preserving protected regions.** RAM read-write checking gains a γ² claim, `0 = Σ_{k,j} eq(r0,j)·C(k)·ra(k,j)·Inc(j)`, where `C` is the public mask of doublewords that hold source instructions, program input, trusted advice, untrusted advice or the stack canary `[stack_end, stack_end + STACK_CANARY_SIZE)`. The mask combines source-code intervals with the public memory layout, rounds extents outward to doublewords and merges overlaps and adjacent intervals in RAM-index space. It is restricted to the proof's RAM domain, including index zero and I/O words below `RAM_START_ADDRESS`. Program preprocessing carries the canonical code intervals and memory layout; the verifier evaluates `C̃` in O(s·log K) for s intervals. The tracer rejects any store that would change a protected word.
2. **Plain flag words.** The panic and termination words are ordinary bytes. A run must end with `(panic, termination) = (0,1)` or `(1,0)`; the public `panic` bit is read from that pair. `ValIo` is unchanged: it already imposes this pair.
3. **Termination at a repeatable self-jump.** A new bytecode circuit flag `CanEnd` marks jumps at which a trace may end (direct JAL; JALR whose target provably repeats). One new R1CS row, `(Jump − ShouldJump)·(LookupOutput − UnexpandedPC + 2^64·(1 − CanEnd)) = 0`, fills the spare slot in the second uni-skip group. The prover sizes the trace to the next power of two of the executed length, so a real row can occupy the last index.
4. **Selected address on every memory row.** RAM read-write checking gains a γ³ claim, `(Load + Store)(r0) = Σ_{k,j} eq(r0,j)·ra(k,j)`, using the Load and Store openings that stage 1 already produces. The tracer rejects loads below `lowest_address`.
5. **Admission checks.** Layout and preprocessing admission check `heap_end ≤ 2^64 − 8`, `lowest_address + 8·K_max ≤ 2^64` for the maximum admitted RAM domain, and even addresses for every decoded instruction.
6. **Unmapped words.** RAM read-write checking gains a γ⁴ claim, `0 = Σ_{k,j} eq(r0,j)·N(k)·ra(k,j)`, where `N` is the public mask of the I/O padding `[io_end, RAM_START_ADDRESS)` and the heap padding `[heap_end, lowest_address + 8·K)`, clipped to the proof's RAM domain. Layout construction rounds `heap_end` up to a doubleword, which matches the linker's 8-aligned `__heap_end`, and admission rejects an unaligned `heap_end`, so both bounds fall on word boundaries. The verifier evaluates `Ñ` like `C̃`. The interpreter and native tracer fault on any access at or above `io_end` below DRAM and at or above `heap_end` in DRAM.

### Invariants

- **Prover/verifier consistency.** The RAM relation and the R1CS row are defined once in `jolt-claims`; the reference kernel, optimized kernel, clear verifier and BlindFold consume them. Reference and optimized kernels produce identical round messages.
- **Soundness of Change 1.** `Inc` and the committed `ra` chunks are fixed before `r0` is drawn. With `ra` zero-or-one-hot on each cycle, `Inc(j)·Σ_k C(k)·ra(k,j)` is a fixed vector; its multilinear extension vanishes at random `r0` only if the vector is zero, except with probability `log T/|F|`.
- **Soundness of Change 3.** The row needs field characteristic greater than 2^65. BN254 `Fr` and Akita's `fp128` qualify; the row's definition asserts it.
- **Soundness of Change 4.** Together with one-hotness, the count claim forces exactly one selected address on memory rows and none on other rows. RAF then places every access at or above `lowest_address`.
- **Soundness of Change 6.** With `ra` zero-or-one-hot on each cycle, `Σ_k N(k)·ra(k,j)` is a fixed 0/1 vector; its multilinear extension vanishes at random `r0` only if no cycle selects an unmapped word, except with probability `log T/|F|`. Together with Change 4 and RAF, every access lands on a mapped word in `[lowest_address, io_end)` or `[RAM_START_ADDRESS, heap_end)`.
- **Completeness.** Honest SDK guests prove as before: their exits write the canonical flag pair and end at `j .`; they never store into text; their loads stay in mapped memory.
- **Tracer agreement.** The interpreter, parallel replay and native x86 tracer apply the same store, load and stopping rules.

### Non-Goals

- Allowing LR/SC on the I/O window. The RAM-only guards stay.
- Proving these properties in Lean. The jolt-qed project will re-prove its theorems against the new constraints separately.

## Evaluation

### Acceptance Criteria

- [ ] A guest that changes a code word and later restores it is rejected by the tracer; a test witness that bypasses the tracer is rejected by the verifier.
- [ ] Interpreter, replay and native tracer stores that change input, trusted advice, untrusted advice or canary words are rejected; same-value stores, including narrow stores and AMOs, are accepted.
- [ ] Failed SC.D and SC.W on canary and code words return rd = 1 and leave memory unchanged; successful same-value SC is accepted. LR/SC retain their RAM-only address guards.
- [ ] Verifier witnesses that change input, trusted advice, untrusted advice or canary words are rejected, including changes followed by restoration.
- [ ] The optimized and reference RAM kernels agree in both binding orders for merged intervals below DRAM, intervals touching index zero, and adjacent or overlapping source and layout intervals.
- [ ] Changing the public memory layout changes the protection mask and preprocessing digest in full and committed program modes.
- [ ] Same-value stores into code words are accepted.
- [ ] A memory row whose `ra` is all zero is rejected by the verifier; a non-memory row with zero `ra` is accepted; a load of the lowest mapped word is accepted.
- [ ] Verifier witnesses that load or store at `io_end`, at `heap_end`, or at the last word of the domain are rejected; an honest guest that uses its whole heap, including a partial final word, proves.
- [ ] Byte, half, word and doubleword loads and stores to every lane of both flag words behave as plain memory; a run ending with a noncanonical flag pair is rejected by the tracer.
- [ ] Truncation after an arbitrary jump, after `jalr x5,0(x5)`, after `c.jalr ra`, or inside an expansion is rejected; SDK exits, stable JALR exits, compressed jumps and complete EBREAK expansions are accepted.
- [ ] A guest whose executed length is exactly a power of two proves in a domain of that size.
- [ ] RV64 has 20 R1CS rows in groups 10+10 (domain 10, degree 27); field-inline has 30 rows in groups 15+15 (domain 15, degree 42).
- [ ] The e2e matrix passes in clear, zk, akita, field-inline and field-inline+zk modes.

### Testing Strategy

The existing e2e matrix, engine twin tests and verifier soundness suites must pass in `host` and `host,zk`. New tests cover each acceptance criterion: kernel tests compare optimized round messages to the reference kernel and to brute-force sums in both binding orders; verifier tampering tests cover each new claim; tracer tests cover the interpreter, replay and native backends.

### Performance

Measured on the optimized backend against `main` with interleaved runs; see the PR description for the numbers. Expected movement: stage 2 (RAM read-write) prover time rises by the extra coefficient work in the existing pass, including one more interval mask for `N`; stage 1 (outer sumcheck) rises by the guard extraction and one folded coefficient; proof size grows by one opening (`CanEnd`); verifier time grows by the two interval-mask evaluations, one more claim in the stage 2 batch, and the extra R1CS row. Traces whose executed length is a power of two get a domain half the size they had.

## Design

### Architecture

| Change | Crates |
|---|---|
| Value-preserving protected regions | `jolt-claims` (RAM relation), `jolt-poly` (interval MLE), `jolt-program` (code intervals), `jolt-kernels` (reference and optimized RAM read-write), `jolt-verifier` (stage 2, BlindFold, preprocessing digest), `tracer`, `jolt-tracer-x86` |
| Plain flag words | `common` (device), `tracer`, `jolt-tracer-x86`, `jolt-witness` |
| Termination | `jolt-riscv` (`CanEnd`), `jolt-claims` (R1CS row), `jolt-kernels` (outer sumcheck), `jolt-witness`, `jolt-prover` (trace sizing), `z3-verifier`, `zklean-extractor` |
| Selected address | `jolt-claims` (RAM relation), kernels, verifier, `tracer`, `jolt-tracer-x86` |
| Unmapped words | `common` (aligned `heap_end`), `jolt-claims` (RAM relation), `jolt-program` (`RamWordMasks`), kernels, verifier, `tracer`, `jolt-tracer-x86` |
| Admission | `common`, `jolt-program`, `jolt-verifier` |

The PR's v3/v2 preprocessing digest domains already bind the code intervals, `CanEnd` and the entire memory layout in full and committed program modes. Deriving the enlarged mask from those fields requires no additional digest input or domain change.

### Alternatives Considered

- **Check only the final code image.** Cheaper, but a program could change code and restore it before exit.
- **A separate code relation or store-flag exclusion.** Adds relation plumbing or a flag opening for no arithmetic saving.
- **A target-only end row, `H·(LookupOutput − UnexpandedPC) = 0`.** Unsound: `jalr x5,0(x5)` passes it and then jumps elsewhere on the next execution.
- **Rejecting self-modifying JALRs statically.** Breaks the standard `auipc ra; jalr ra, lo(ra)` call.
- **Changing an existing PC-row guard instead of adding a row.** Breaks the SDK exit at the padding transition.
- **Two separate rows for target and `CanEnd`.** Grows RV64 to 21 rows (domain 11, degree 30). The packed row fits the spare slot.
- **A required final padding row instead of Change 3b.** Adds a boundary condition and wastes a row at every power of two.

## Documentation

The book's RAM checking, R1CS and memory-layout pages and the specs `proof-trace-row-layout.md` and `field-inline-protocol.md` are updated with the new claims, row counts, flag semantics and admission checks.

## Execution

The optimized RAM kernel folds `C(k)` into the Inc coefficient while address bits are unbound and represents the partially bound mask as full blocks plus at most two boundary blocks per interval. It allocates no K-sized table. The outer kernel stores the new row's B as two limbs within the existing bound.

## References

- jolt-qed (Lean model of Jolt against the Sail RISC-V model): the premises this removes are `no_code_store`, `no_flag_word_load`, `panic_store_nonzero`, `amo_no_flag_word`, `panicStoresOne`, `terminationThenStall`, `claimsNoPanic`, `no_low_load`, `noMaxLengthTrace`, HeapNoWrap, ramRegionFits and RowsAligned.
