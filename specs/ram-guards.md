# Spec: RAM Guards

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @quangvdao                     |
| Created     | 2026-11-10                     |
| Status      | proposed                       |
| PR          | #2408                          |

## Summary

Jolt's constraints differ from a plain RISC-V machine in three places that a formal model of Jolt has to carry as premises. Nothing in them stops a store from changing the program's own code words, although Jolt fetches instructions from committed bytecode rather than from memory; the same holds for program input, advice and the stack canary. They let a memory row whose RAM address is 0 select no RAM word, so a load there reads 0 and a store of 0 there changes nothing, although address 0 is not mapped. And they let a memory row select a word of the proof's RAM domain that no mapped region holds: the padding between `io_end` and `RAM_START_ADDRESS`, or a word at or above `heap_end`. The tracers already refuse all three: they reject a store that changes a protected word, fault on any access below `lowest_address`, and fault on any access at or above `io_end` below DRAM or at or above `heap_end` in DRAM. This spec adds the matching constraints as three more terms of the existing RAM read-write sumcheck over public preprocessing, with no new commitment, opening or sumcheck round. With Booleanity, the access-count term makes the RAM selector one-hot exactly on memory rows, so the RAM Hamming-booleanity sumcheck and the RAM leg of the stage-7 Hamming-weight reduction become redundant, and this spec removes them.

## Intent

### Goal

Make every accepted Jolt proof correspond to a run in which code, program input, trusted advice, untrusted advice and the stack canary keep their initial values, every memory row selects exactly one mapped word of the proof's RAM domain, and every other row selects none.

The changes are:

1. **Value-preserving protected regions.** RAM read-write checking gains a γ² claim, `0 = Σ_{k,j} eq(r0,j)·C(k)·ra(k,j)·Inc(j)`, where `C` is the public mask of doublewords that hold source instructions, program input, trusted advice, untrusted advice or the stack canary `[stack_end, stack_end + STACK_CANARY_SIZE)`. `write_protected_word_intervals` builds the mask from the code intervals and the public memory layout: it rounds extents outward to doublewords, merges overlapping and adjacent intervals in RAM-index space, and restricts the union to the proof's RAM domain, including index zero and I/O words below `RAM_START_ADDRESS`. The verifier evaluates `C̃` in O(s·log K) for s intervals (`interval_mask_mle_msb`).
2. **Selected address on memory rows.** RAM read-write checking gains a γ³ claim, `(Load + Store)(r0) = Σ_{k,j} eq(r0,j)·ra(k,j)`, using the Load and Store openings that stage 1 already produces.
3. **Unmapped words.** RAM read-write checking gains a γ⁴ claim, `0 = Σ_{k,j} eq(r0,j)·N(k)·ra(k,j)`, where `N` is the public mask of the I/O padding `[io_end, RAM_START_ADDRESS)` and the heap padding `[heap_end, lowest_address + 8·K)`, clipped to the proof's RAM domain (`unmapped_word_intervals`). Both bounds fall on word boundaries: layout construction rounds every I/O region size to a multiple of eight, so `io_end` is doubleword aligned, and rounds `heap_end` up to a doubleword, which matches the linker's 8-aligned `__heap_end`; admission rejects an unaligned `heap_end`, and `unmapped_word_intervals` rejects an unaligned bound. `RamWordMasks` carries `C` and `N` together from preprocessing to the relation and rejects masks that overlap. The verifier evaluates `Ñ` like `C̃`.
4. **No RAM Hamming check.** Stage 6b has no RAM Hamming-booleanity sumcheck, and the stage-7 reduction has no RAM Hamming-weight leg in either the base `ClaimReduction` or Akita's `LatticeDigitZeroClaimReduction`. The base reduction keeps a stride of three γ powers per polynomial, with RAM skipping `γ^{3i}`; the Akita reduction has two legs per polynomial, at `γ^{2i}` and `γ^{2i+1}`. The retired serialized identifiers stay as reserved codec slots holding the uninhabited `RetiredIdentifier`: `JoltRelationId::Reserved12`, `JoltVirtualPolynomial::Reserved34` and `JoltDerivedId::Reserved6`.
5. **Code-interval admission.** Program preprocessing carries canonical code intervals: `RAMPreprocessing::code_word_intervals` in full mode and `ProgramMetadata::code_word_intervals` in committed mode. Full preprocessing derives them from the decoded bytecode, and deserializing it checks exact source coverage, canonical shape and RAM bounds. The verifier repeats the check when it builds or deserializes its preprocessing and in each proof's input checks: coverage, shape and bounds in full mode; shape and bounds in committed mode, where the key's coverage is trusted.

### Invariants

- **Prover/verifier consistency.** `ReadWriteChecking::unbatched_relation` in `jolt-claims` defines the three new claims once, and the reference kernel, the clear verifier and BlindFold consume it. The optimized kernel specializes the same batch; kernel parity tests pin its round messages to the reference kernel's.
- **Soundness of the access-count term.** The committed `RamRa` chunks and the bytecode rows each cycle selects, which fix `Load` and `Store`, are bound before `r0` is drawn. The residual `Σ_k ra(k,j) − Load(j) − Store(j)` is therefore a fixed vector, and its multilinear extension vanishes at random `r0` only if the vector is zero, except with probability `log T/|F|`. The selector is `ra = Π_i ra_i`, the product of the committed chunks, so `Σ_k ra(k,j) = Π_i n_i(j)`, where `n_i(j) = Σ_{k_i} ra_i(k_i,j)` sums over the digit values that occur in RAM-domain addresses. Booleanity covers every entry of every committed chunk, so each `n_i(j)` is a nonnegative integer, and the product is far below the field characteristic, so it does not wrap. `Load` and `Store` are Boolean bytecode flags that no instruction sets together (the `load_store_exclusive` test in `jolt-riscv` pins the instruction catalog), and bytecode read-checking ties each cycle's flags to its bytecode row. On a memory row the product is 1, so every chunk's row is one-hot and `ra(·,j)` selects exactly one word. On any other row the product is 0, so some chunk's row is zero and `ra(·,j) ≡ 0`. RAF evaluation then places each access at `8k + lowest_address` for its selected word `k`. No per-chunk weight is checked: on a non-memory row, the chunks other than a zero one may hold several ones.
- **Soundness of the protected-region term.** `Inc` and the committed `RamRa` chunks are fixed before `r0` is drawn, so `Inc(j)·Σ_k C(k)·ra(k,j)` is a fixed vector, and its multilinear extension vanishes at random `r0` only if the vector is zero, except with probability `log T/|F|`. Since `ra` is zero-or-one-hot on each cycle (previous invariant), the vector's entry is `Inc(j)` on a cycle that selects a protected word and 0 on every other cycle, so every access to a protected word has a zero increment. The RAM value check rebuilds each value from initial RAM and the earlier selected increments, so every protected word keeps its initial value for the whole run; a change that a later store undoes is rejected too.
- **Soundness of the unmapped-word term.** The committed `RamRa` chunks are fixed before `r0` is drawn, and `ra` is zero-or-one-hot on each cycle (the access-count invariant: Booleanity of every committed chunk plus the access-count term), so `Σ_k N(k)·ra(k,j)` is a fixed 0/1 vector; its multilinear extension vanishes at random `r0` only if no cycle selects an unmapped word, except with probability `log T/|F|`. Together with the access-count term and RAF evaluation, every access lands on a mapped word in `[lowest_address, io_end)` or `[RAM_START_ADDRESS, heap_end)`.
- **RAM chunks under Akita.** Booleanity and the access-count term make the selector `ra = Π_i ra_i` one-hot on memory rows, where every chunk's row is then one-hot, and zero on other rows. On a non-memory row, the chunks other than a zero one may hold several ones (the access-count invariant). An accepted Akita proof can therefore commit RAM chunk columns outside Akita's `UnitOneHot` source class, which the removed RAM Hamming checks used to exclude. That class is a producer contract: Jolt checks it only on the prover's witness (`validate_witness` in `jolt-akita/src/native_batching.rs`), which honest traces satisfy. It does not bind the verifier, because Akita's verifier enforces its frozen, matrix-certified response caps (`TerminalFoldParams::validate_terminal_linf_cap` and `response_l2_sq_cap` in `akita-params/src/schedule.rs`) whatever the source; `akita-params/src/sis/committed_source.rs` states that a source outside its class is not a soundness break, only an unsupported commitment. Soundness therefore does not depend on per-chunk one-hotness.
- **Completeness.** Honest SDK guests prove as before: they never change a protected word, each memory row of an honest trace selects the word it accesses, and their accesses stay in mapped memory, so no selected word is in `N`.
- **Tracer agreement.** The interpreter, parallel replay and native x86 tracer refuse the same runs: they reject a store that changes a protected word and fault on any access below `lowest_address`, at or above `io_end` below DRAM, or at or above `heap_end` in DRAM.

### Non-Goals

- Allowing LR/SC on the I/O window. The RAM-only guards stay.
- Constraining the rest of the tracers' memory-map policy. The constraints check protected words, address selection and unmapped words only.
- Constraining how a run ends. The constraints accept a trace whose last real row is any jump.
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
- [ ] A memory row that selects two words is rejected by the verifier, and so is a non-memory row that selects a word.
- [ ] Verifier witnesses that load or store at `io_end`, at `heap_end`, or at the last word of the domain are rejected; an honest guest that uses its whole heap, including a partial final word, proves.
- [ ] Stage 6b has no RAM Hamming-booleanity member, and the stage-7 reduction has no RAM Hamming-weight leg in base or Akita mode.
- [ ] The retired identifier slots fail to decode, and the live identifiers after them keep their bincode tags.
- [ ] The e2e matrix passes in clear, ZK and Akita modes for both the ordinary and the field-inline instruction profile.

### Testing Strategy

The existing e2e matrix, engine twin tests and verifier soundness suites must pass in `host` and `host,zk`. New tests cover each acceptance criterion: kernel tests compare optimized round messages to the reference kernel in both binding orders, and the bound address masks to brute-force block sums; a verifier unit test compares the stage-2 output check with the relation's output expression evaluated on brute-force public terms; verifier soundness tests (`crates/jolt-verifier/tests/soundness/ram_guards.rs`) run malicious two-cycle witnesses that bypass the tracer through the stage-2 verifier; relation tests evaluate the base and Akita stage-7 expressions against their expected γ layouts; a codec test pins the reserved slots' bincode tags. The tracer tests for the protected-word, low-load and unmapped-address guards cover the interpreter, replay and native backends.

### Performance

Expected movement against `main`:

- Stage 2 (RAM read-write): the prover does extra coefficient work for the γ², γ³ and γ⁴ terms inside its existing sparse pass, including one more interval mask for `N`, which the binary search that classifies an address for `C` also covers. There is no new trace pass, table, opening or round, and the degree stays 3.
- Stage 6b: one member fewer, so one claim (32 bytes in a clear Dory proof), one batching coefficient and one O(T) prover pass fewer. Rounds and degree are unchanged.
- Stage 7: the RAM chunks lose their Hamming-weight legs.
- Verifier: stage 2 evaluates the cycle equality once and each of the two interval masks once, in O(log T + s·log K) for s mask intervals, and its RAM read-write output check has three more terms; stage 6b has one member fewer to check.
- Stage 1 (the outer sumcheck) is unchanged.

## Design

### Architecture

| Change | Crates |
|---|---|
| Value-preserving protected regions | `jolt-claims` (RAM relation, `CodeEqCycle` public), `jolt-poly` (interval MLE), `jolt-program` (code intervals, protection mask), `jolt-kernels` (reference and optimized RAM read-write), `jolt-verifier` (stage 2, BlindFold, preprocessing digest), `jolt-prover` (interval plumbing) |
| Selected address on memory rows | `jolt-claims` (RAM relation), `jolt-kernels`, `jolt-verifier` |
| Unmapped words | `jolt-claims` (RAM relation, `UnmappedEqCycle` public), `jolt-program` (`RamWordMasks`, `unmapped_word_intervals`), `jolt-kernels` (reference and optimized RAM read-write), `jolt-verifier` (stage 2, BlindFold), `jolt-prover` (mask plumbing) |
| No RAM Hamming check | `jolt-claims` (reserved identifiers, stage-7 relations), `jolt-kernels`, `jolt-witness`, `jolt-verifier` (stages 6b and 7, BlindFold), `jolt-prover` |
| Code-interval admission | `jolt-program` (preprocessing construction and deserialization), `jolt-verifier` (preprocessing and per-proof input checks) |

The tracers' matching guards (`tracer`, `jolt-tracer-x86`), the layout's doubleword-aligned `heap_end` (`common`) and the ELF admission check that rejects a writable section sharing a code word (`jolt-program`) are prerequisites, not part of this change.

The preprocessing digest hashes the whole `ProgramPreprocessing` value, so the code intervals and the memory layout enter it in both full and committed program modes. The interval field changes the digest input, so the domain tags move to `v3` (Dory) and `v2` (Akita), field-inline variants included. The verifier derives both masks from these digest-bound fields and the proof's RAM domain, which needs no further digest input.

### Alternatives Considered

- **Check only the final code image.** Cheaper, but a program could change code and restore it before exit.
- **A separate code relation or store-flag exclusion.** Adds relation plumbing or a flag opening for no arithmetic saving.
- **Keep RAM Hamming booleanity.** Redundant given the access-count term, and it costs a stage-6b claim, a batching coefficient and an O(T) prover pass, plus the RAM legs of the stage-7 reduction.

## Documentation

The book's RAM checking, BlindFold and guest security-model pages and the specs `byte-addressable-memory.md`, `digit-zero-virtualization.md`, `field-inline-protocol.md` and `m-mode-trap-csr-semantics.md` are updated with the new claims, the constraint-enforced protected-word rule, the unmapped-word mask and the RAM one-hotness argument.

## Execution

The optimized RAM kernel folds `C(k)` into the Inc coefficient while address bits are unbound and represents the partially bound mask as full blocks plus at most two boundary blocks per interval. It allocates no K-sized table. It keeps `C` and `N` in one interval list sorted by start, each interval tagged with its mask, so before any address bind one binary search classifies an address for both masks, and it folds `γ⁴·N(k)` into the access-count coefficient. It fuses the access-count term into the same sparse accumulations.

## References

- jolt-qed (Lean model of Jolt against the Sail RISC-V model): the protected-region term removes the premise `no_code_store`, and the access-count term with RAF evaluation removes `no_low_load`. The layout and ELF admission checks of #2415, not this spec, remove HeapNoWrap, ramRegionFits and RowsAligned.
