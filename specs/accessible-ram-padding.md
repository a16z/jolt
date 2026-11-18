# Spec: Accessible RAM Padding

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Michael Zhu                    |
| Created     | 2026-11-17                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Two parts of the admitted RAM domain `[lowest_address, lowest_address + 8·max_ram_K)` hold no program data: the I/O gap `[io_end, RAM_START_ADDRESS)` between the outputs and the program image, and the heap padding `[heap_end, lowest_address + 8·max_ram_K)` that rounds the domain up to a power of two. Jolt treats both as unmapped: every tracer faults on an access there, and a γ⁴ claim of the RAM read-write sumcheck ([`ram-guards.md`](./ram-guards.md), Change 3) rejects one in a proof. Nothing else in the proof depends on that restriction. Initial RAM is already zero outside the program image, the inputs and the advice, and the RISC-V unprivileged specification leaves it to the execution environment to determine "what portions of the address space are accessible for each kind of memory access". The restriction is a memory-map policy: it costs a claim and a second interval mask in the relation, the kernels, the verifier and BlindFold, plus separate fault paths in every tracer, and it splits the accessible memory into two intervals. This spec makes both regions zero-initialized read/write memory, so a guest may access every address of the admitted domain, subject to the existing write protection, and nothing outside it.

## Intent

### Goal

Make the I/O gap and the heap padding zero-initialized read/write memory in every tracer and in the proof: a plain load or store anywhere in the admitted RAM domain is accepted, except a store that changes a write-protected word, and every access outside the domain is rejected.

- **Admitted RAM domain.** `[lowest_address, lowest_address + 8·max_ram_K)`. `MemoryLayout::max_ram_k`, the existing sizing law, makes `max_ram_K` the smallest power-of-two number of doublewords covering `[lowest_address, heap_end)`. Each proof's domain `[lowest_address, lowest_address + 8K)` lies inside it, because the verifier admits only `K ≤ max_ram_K`.
- **I/O gap.** `[io_end, RAM_START_ADDRESS)`. For every layout `MemoryLayout::try_new` builds, `io_end = output_end ≤ RAM_START_ADDRESS − 8`, so the gap holds at least one doubleword.
- **Heap padding.** `[heap_end, lowest_address + 8·max_ram_K)`, empty when `[lowest_address, heap_end)` already spans a power-of-two number of doublewords.

The changes are:

1. **Relation.** The RAM read-write sumcheck loses its γ⁴ claim `0 = Σ_{k,j} eq(r0,j)·N(k)·ra(k,j)`; the four remaining claims keep γ⁰ to γ³. `RamReadWritePublic::UnmappedEqCycle` becomes the reserved codec slot `Reserved2(RetiredIdentifier)`. `RamWordMasks`, `unmapped_word_intervals` and `PreprocessingError::InvalidUnmappedIntervals` are deleted: the clear verifier, BlindFold and the kernels take the write-protected intervals from `write_protected_word_intervals`, and the optimized kernel binds one interval mask in its default and streaming slots.
2. **Output check.** `PublicIoMemory::new` ends the IoMask at the word of `output_end` instead of `RAM_START_ADDRESS`, so the check binds exactly the inputs and outputs, `[input_start, output_end)`, and `ram_output_check_publics` evaluates an empty mask (no input or output capacity) to zero. The mask now depends on a layout field, so `MemoryLayout::validate_io_segments` requires the I/O segments to follow one another from the lowest address without gaps or overlap, each doubleword aligned and exactly its configured size, each advice capacity zero or a power of two with its segment starting at a multiple of it from the lowest address, and `io_end = output_end ≤ RAM_START_ADDRESS`. Otherwise the layout fails with `MemoryLayoutError::IoSegmentMisplaced` or an advice-size error. `MemoryLayout::max_ram_k` runs it, so construction, verifier-key admission and each proof's input validation do, and so does `PublicIoMemory::new`, which also rejects an input or output longer than its configured size (`MemoryLayoutError::IoTooLong`). The deleted `unmapped_word_intervals` had checked `io_end`'s alignment on every verification.
3. **Layout and device.** `MemoryLayout::ram_size` returns the bytes from `RAM_START_ADDRESS` to the end of the admitted domain and replaces `get_total_memory_size`. `JoltDevice::is_io_gap` classifies gap addresses, and `JoltDevice::load` asserts that its address lies below `io_end`, so a tracer path that fails to route a gap address panics instead of reading zero.
4. **Tracers.** The interpreter's RAM backing and the x86 tracer's plane hold `ram_size` bytes. `jolt-emu` sizes an ELF without `tohost` the same way, attaching the device of the default configuration over the decoded image before it loads the program; riscv-tests ELFs keep the test capacity. `Emulator::setup_program` requires a Jolt guest's device: the fixed 32 MiB backing it used without one is deleted. Both keep the I/O gap in `IoGapMemory` (`tracer/src/emulator/io_gap.rs`), which stores only the gap's nonzero doublewords; lazy checkpoints, replay chunk checkpoints and x86 boundaries carry it. The faults at `io_end` and `heap_end` are deleted; loads and stores below `lowest_address` and any access at or beyond the end of the domain still fault.
5. **Final memory.** `MemoryImage` keys are the absolute guest byte addresses of the nonzero bytes outside the I/O segments, in RAM and in the I/O gap. Both tracers build final images through `final_memory_image`, and the witness's `populate_final_memory_image` reads every key as an absolute address.

### Invariants

- **One owner for the domain bound.** `MemoryLayout::max_ram_k` is the only sizing law for the admitted domain. `MemoryLayout::ram_size` derives the tracers' RAM span from it; the interpreter's backing, the x86 plane and both tracers' upper access bound come from `ram_size`, and verifier admission bounds `K` through `compute_max_ram_k`, which wraps `max_ram_k`. Tests take the bound from these functions or from literals, never from a re-derivation.
- **Tracer agreement.** The interpreter's eager, lazy, two-pass and chunked-replay drivers and the x86 tracer accept the same accesses and produce the same rows, device and final memory. Loads and stores anywhere in the admitted domain succeed, except a store that changes a write-protected word and `LR`/`SC` below `RAM_START_ADDRESS`; loads and stores below `lowest_address` and any access at or beyond the end of the domain fault. Host calls (prints, cycle markers and advice writes) read gap and padding bytes like any other byte a guest load accepts.
- **Zero initial RAM.** Initial RAM is the program image, the inputs and the advice, and zero elsewhere (`PublicInitialRam::new` and the advice openings). The tracers start the gap and the padding at zero, so a load of a word no store has written returns 0 in the trace and in the proof.
- **Output check.** The IoMask is `[word(input_start), word(output_end))`, and ValIo holds the inputs and the claimed outputs, zero elsewhere in that range. Nothing binds the final contents of the I/O gap or the heap padding. No verifier formula reads `io_end`; admission only requires `io_end = output_end ≤ RAM_START_ADDRESS`.
- **Soundness.** No claim treats padding words differently from other unprotected RAM words. The access-count term with RAF evaluation confines every access to the proof's domain `[lowest_address, lowest_address + 8K)`, and verifier admission bounds `K` by `max_ram_K`, so every proved access lies in the admitted domain, where the tracers accept it; the protected-word term still rejects every change to a write-protected word.
- **Retired slot.** `RamReadWritePublic::Reserved2` holds the uninhabited `RetiredIdentifier`: tag 2 fails to decode, and `EqCycle` and `WriteProtectedEqCycle` keep tags 0 and 1.

No `jolt-eval` invariant changes. `tracer_backend_equivalence` already compares final memory across backends, but its corpus never touches the padding, so the focused tracer tests below carry the agreement on padding words.

### Non-Goals

- **LR/SC in the I/O gap.** Memory below `RAM_START_ADDRESS` stays non-reservable: the LR/SC expansions keep their `RAM_START_ADDRESS ≤ addr` assertion, so plain loads, stores and AMOs work in the gap, and LR/SC fault there.
- **Layout changes.** `MemoryLayout::try_new` keeps its `+ 8` reservation after the outputs, so every layout keeps an I/O-gap word and no layout address or program digest changes. Without it, every layout whose I/O word count is a power of two, the default included, would have no gap.
- **Typed tracer faults.** #2488 turns tracer faults into typed errors; the faults that remain keep their current panic and helper-error shapes here.

## Evaluation

### Acceptance Criteria

- [ ] Loads from previously untouched padding return zero; stores persist, and later loads return the stored value.
- [ ] Tests cover the first I/O-gap word, the first heap-padding word and the last word of the admitted RAM domain; accesses beyond that domain remain rejected.
- [ ] Interpreter execution, parallel replay and native x86 execution agree. Proving and verification accept these accesses in clear, ZK and Akita modes for both the ordinary and the field-inline profile.
- [ ] Existing protected-word, access-selection and RAM-consistency checks continue to pass, and the tests that required padding accesses to fail are replaced with focused tests of the new contract.
- [ ] A run that leaves nonzero values in the I/O gap at its halt proves and verifies.
- [ ] Decoding `RamReadWritePublic` tag 2 fails, and the live tags keep their values.

### Testing Strategy

The e2e matrix, the verifier soundness and fixture suites, and the tracer and x86 suites must pass in `host` and `host,zk`, and clippy must pass in CI's feature combinations.

- **e2e.** The memory-ops guest entry `ram_padding` replaces `heap_top` in the ordinary table and joins the field-inline table, so it runs in clear, ZK and Akita modes in both profiles. At the first I/O-gap word, the first heap-padding word and the last domain word, with addresses taken from the production `MemoryLayout`, it loads (expecting 0), stores and reloads a doubleword and then a byte, and leaves the values in place at the halt, which shows that the output check does not reach into the I/O gap; the IoMask unit test below pins its exact end at `output_end`.
- **Layout.** `admission_rejects_misplaced_io_segments` (`common`) gives the outputs a span other than their configured size or a size that is not a multiple of eight, shifts the whole I/O block off a doubleword, overlaps the inputs with the outputs, moves `io_end` off `output_end`, sets a non-power-of-two advice capacity, and puts the smaller advice segment first; `max_ram_k` rejects each.
- **jolt-emu.** `jolt_emu_backs_a_jolt_guest_through_its_ram_domain` runs the binary on a hand-built ELF that loads, stores and reloads the last word of the default layout's domain, in its heap padding.
- **zkLean.** `ram_write_protection_claim_keeps_its_mask` checks that the extracted write-protection claim carries the mask variable.
- **Interpreter.** Contract tests at `io_end`, `RAM_START_ADDRESS − 8`, `heap_end` and the last domain word: zero loads, persisting doubleword and byte stores, faults for a load and a store at the domain end, and host-call readability. A hand-built ELF whose gap and padding stores and later loads fall in different chunks is traced eagerly, lazily, by two-pass replay and by chunked replay; all four agree, and the final memory holds the stored bytes at their absolute addresses.
- **x86.** The same words through `run_machine` and record mode, faults beyond the domain, host calls that read gap values, gap and padding values carried across checkpoints, and an interpreter-vs-x86 parity test on a hand-assembled ELF that compares rows, dimensions, final memory and device.
- **Verifier.** `verifier_accepts_loads_and_stores_on_padding_words` (`crates/jolt-verifier/tests/soundness/ram_guards.rs`) verifies load and store witnesses at the first gap word, the first padding word and the last domain word under the production write-protection mask; the file's other tests keep their assertions. A unit test pins the IoMask from `ram_output_check_publics` at boundary words of a small layout, and checks that it returns zero for a layout with no input or output capacity.
- **Codec.** `retired_identifiers_are_rejected_without_shifting_live_tags` pins `Reserved2`.
- **Fixtures.** The six verifier fuzz bundles are regenerated, and `FIXTURE_MAGIC` moves to `JVCF0019`. The eight program-digest goldens pass unchanged.

### Performance

- **Stage 2.** The RAM read-write instance loses a claim: the prover drops the γ⁴ coefficient and binds one interval mask instead of two, and the verifier evaluates one mask MLE instead of two. Rounds, degree and openings are unchanged, so a clear proof keeps its size.
- **Tracer memory.** The RAM backing grows from `heap_end − RAM_START_ADDRESS` to `ram_size = 8·max_ram_K − (RAM_START_ADDRESS − lowest_address)`, which is at least the padded I/O region and less than twice `heap_end − lowest_address`. For the default layout that is 64 MiB less the 32 KiB I/O region instead of 32 MiB plus the program, canary and stack; a large I/O region grows it further (64 MiB of advice capacity with a 32 KiB heap: about 0.1 MiB to 128 MiB). Execution commits no memory for untouched padding: both backings are zeroed allocations (`vec![0; n]` in the interpreter, a `MAP_NORESERVE` mapping for the x86 plane). The interpreter's final image and its emulator clones read only the prefix below its high-water mark, so they copy no more than before. Paths that copy a whole image grow with the backing: the x86 plane keeps no high-water mark, so its final scan and its dense boundary snapshots do, and so does the interpreter's chunk replay, which clones its boundary image; neither has a production caller, and the PR reports the measured effect. The I/O gap, about 63 MiB in the recursion guests, is stored sparsely, so checkpoints and snapshots copy only its nonzero doublewords.
- **RAM domain.** A run that touches the heap padding can need a larger `K`, up to `max_ram_K`. The e2e `ram_padding` case touches the last domain word, so it proves at `K = max_ram_K = 2^23` under the harness's default 32 MiB heap, the domain `heap_top` proved in the ordinary profile; the field-inline arms prove that domain for the first time.
- **jolt-eval.** The trace-generation objectives (`objective/performance/trace_gen.rs`) and `telemetry:<workload>:peak_rss_gib` may rise with the larger backing and x86 snapshots.

## Design

### Architecture

| Change | Crates |
|---|---|
| Relation and codec | `jolt-claims` (RAM relation, `RamReadWritePublic::Reserved2`), `jolt-program` (`write_protected_word_intervals` alone), `jolt-kernels` (reference and optimized RAM read-write, one interval mask), `jolt-verifier` (stage 2, BlindFold, fuzz target), `jolt-prover` (stage-2 wiring) |
| Output check | `jolt-program` (`PublicIoMemory`), `jolt-verifier` (`ram_output_check_publics`), `common` (`MemoryLayout::validate_io_segments`) |
| Layout and device | `common` (`MemoryLayout::ram_size`, `JoltDevice::is_io_gap`, `JoltDevice::load`) |
| Tracers | `tracer` (`IoGapMemory`, `final_memory_image`, MMU routing and faults, checkpoints and replay, `jolt-emu` sizing), `jolt-tracer-x86` (plane sizing, gap routing in the host helpers, boundaries) |
| Formal model | `zklean-extractor` (claim weights, the `mem_write_protected` ROM in the package template) |
| Final memory | `jolt-program` (`MemoryImage` keys), `jolt-witness` (`populate_final_memory_image`) |
| Tests | the memory-ops guest (`ram_padding`), `jolt-prover` (`e2e_matrix`), verifier fixtures |

### Alternatives Considered

- **End the output check at `RAM_START_ADDRESS` or at `io_end`.** Keeping `RAM_START_ADDRESS` would force every gap word to end the run at zero, so a guest that leaves data in the gap could not be proved, which contradicts read/write memory. `io_end` equals `output_end` for every layout `try_new` builds, and admission rechecks it; ending at `output_end` binds exactly the inputs and outputs, and no verifier formula reads `io_end`.
- **Rebase the flat RAM backings to start at `io_end`.** One dense backing would hold the gap too, but the gap can be about half of the I/O region (about 63 MiB for the recursion guests) and is normally untouched, and every snapshot, checkpoint and replay image would copy it. `IoGapMemory` costs nothing until a guest stores a nonzero value there.
- **Keep `MemoryImage` keys relative to `RAM_START_ADDRESS`.** Gap bytes lie below `RAM_START_ADDRESS`, so they would need negative offsets or a second image, and the witness would keep guessing a key's convention from its value (`address >= dram_start`). Absolute keys give gap and RAM bytes one key space.
- **Bump the program-preprocessing digest domains** (Dory `v5`, Akita `v4`), following [`halt-end-rule.md`](./halt-end-rule.md)'s reason that a key for the old protocol must not match one for the new. `PROGRAM_PREPROCESSING_DIGEST_DOMAIN` (`crates/jolt-verifier/src/preprocessing.rs`) is bumped when the digest input changes, and this change leaves the input unchanged: the layout keeps every address, and the key stores no mask (the verifier derives the protection mask from the key's code region and layout). In `halt-end-rule.md` the layout lost two fields, so its keys changed; here a key describes the same program and layout under both protocols. The eight digest goldens pass unchanged.

## Documentation

The book's RAM page replaces its memory-layout figure with a memory-map table (regions, access, write protection, the end of the admitted domain and what faults), drops mask `N` from read-write checking, states the tracer/constraint agreement on the domain bounds, and ends the output check at `output_end` without the final-memory figure; `memory_layout.png`, `final_memory_state.png` and the unreferenced `initial_memory_state.png` are deleted. The guests page gives foreign toolchains the memory contract, the `LR`/`SC` limit in the I/O gap and the tracers' RAM sizing; the testing-gates page lists `ram_padding` in both profiles. `ram-guards.md` retires its Change 3, and `halt-end-rule.md`, `byte-addressable-memory.md`, `m-mode-trap-csr-semantics.md` and `x86-tracer-backend.md` follow the new contract.

## Execution

The proof side (relation, codec slot, masks, output check, kernels, verifier, BlindFold) and the tracer side (layout accessor, gap storage, MMU and x86 routing, final images, witness) are independent until the e2e case, which needs both. The verifier fuzz target `blindfold_expression_mapping_equivalence` names the retired public, and the fuzz crates form a separate workspace that workspace clippy does not build: check it with `python3 scripts/fuzz.py --workspace jolt-verifier --target-triple x86_64-unknown-linux-gnu build`. Regenerate the fuzz bundles and bump `FIXTURE_MAGIC` once both sides land.

## References

- Issue #2494.
- RISC-V Unprivileged ISA, [Memory](https://docs.riscv.org/reference/isa/v20250508/unpriv/intro.html#_memory): "The execution environment determines what portions of the address space are accessible for each kind of memory access."
- [`ram-guards.md`](./ram-guards.md): Change 3, the unmapped-word claim this spec retires.
- [`halt-end-rule.md`](./halt-end-rule.md): the reserved word after the outputs.
- #2488: typed tracer faults; it currently includes the `io_end` and `heap_end` faults this spec deletes.
- jolt-qed: `zklean-extractor` exports `ReadWriteChecking::unbatched_relation` directly, so the extracted RAM relation loses its γ⁴ claim. The extractor had dropped every claim's public weight, which exported the γ⁴ claim as `RamRa = 0` and the write-protection claim as `RamRa·RamInc = 0` on every cycle, forbidding every memory access. It now lowers each weight: cycle equality states a claim at every cycle, successor-cycle equality links a cycle to the next, and the write-protection claim reads `0 = WriteProtected·(RamRa·RamInc)`. `memory_step` binds `WriteProtected` to the mask C at the cycle's `RamAddress` through a new ROM, `mem_write_protected`, which `jolt_step` and `jolt_next_step` take as a parameter; a weight with no lowering fails extraction.
