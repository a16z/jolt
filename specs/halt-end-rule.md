# Spec: Halt End Rule

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Michael Zhu                    |
| Created     | 2026-11-14                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt's constraints let any jump precede padding: `ShouldJump = Jump·(1 − NextIsNoop)` switched off the jump-target check (R1CS row 14) on the last real row, so a proof could end after an arbitrary jump. The run's status lived in two memory words, `MemoryLayout::panic` and `MemoryLayout::termination`, which a device latch, the SDK and ValIo all had to agree on. A formal model of Jolt (jolt-qed) carries three premises only because of this scheme: 5a (`PanicStoresOne`), 5b (`TerminationThenStall`) and 5d (`ClaimsNoPanic`). This spec makes the end of a run a property of control flow. A run ends at a halt instruction, `jolt.success` for success or `EBREAK` for a failure, and only padding follows it. The panic bit equals the kind of that halt, bound by product lane 3, and the panic and termination words are deleted. Premises 5a, 5b and 5d then have nothing left to restrict.

## Intent

### Goal

Make every accepted Jolt proof attest a run from the ELF entry to the first halt it executes, followed only by padding, with the public panic bit equal to 1 exactly when that halt is `EBREAK`.

The changes are:

1. **Halt flags.** `CircuitFlags` gains `Halt` and `HaltPanic`, after `IsLastInSequence` and before the field-inline flags: 16 circuit flags in base builds, 26 with `field-inline`. `JoltSuccess` (the row `jolt.success` lowers to) carries `{Jump, Halt}`; `JoltPanic` (the row `EBREAK` and `c.ebreak` lower to) carries `{Jump, Halt, HaltPanic}`. Both kinds are table-free, so their lookup output is 0.
2. **End rule.** `ShouldJump` is deleted; its serialized slot becomes `JoltVirtualPolynomial::Reserved12`. Row 14 (`NextUnexpPCEqLookupIfJump`) becomes `Jump·(NextUnexpandedPC − LookupOutput) = 0`, so every jump target is checked and a halt jumps to address 0. Product row 21 (`JumpHasSuccessorUnlessHalt`, lane 3) keeps the factors `Jump` and `1 − NextIsNoop`, and its output becomes `Jump − (1 − 2Π)·Halt − 2·HaltPanic`, where `Π` is the claimed panic bit.
3. **Panic claim.** `Π` enters a claim only as the appended derived public `SpartanProductVirtualizationPublic::ClaimedPanic = F::from(public_io.panic)`. `PRODUCT_JUMP_LANE_OUTPUT` in `jolt-claims` owns the lane-3 output; R1CS row 21, the product uni-skip input, the clear verifier, the optimized prover kernel and BlindFold use it, and the reference kernel computes the lane from the trace. The product remainder relation rejects the id.
4. **Memory.** `MemoryLayout` loses `panic` and `termination`. The I/O region pads to `next_pow2(io_words + 1)` words, so an unmapped guard word always follows the outputs: `io_end = output_end ≤ RAM_START_ADDRESS − 8`. The device has no panic latch; the tracers set `JoltDevice.panic` from the halt kind. ValIo holds only the input and output segments, and the witness synthesizes no flag words.
5. **SDK.** The `#[jolt::provable]` epilogue writes the outputs, then executes `jolt.success`. `jolt_panic` is deleted; `__platform_exit` and `__platform_abort` execute `ebreak`.
6. **Admission.** A halt must be a whole source instruction (I2′), and the entry row must be a real instruction (I3′).
7. **Trace rows.** The trace-row metadata word stores the six instruction flags other than `IsNoop`, which the row derives from `bytecode_pc == 0`, and the immediate's sign moves to row-control bit 7, so 26 circuit flags fit.

### Invariants

- **One owner for lane 3.** `PRODUCT_JUMP_LANE_OUTPUT` is the only statement of the lane-3 output. R1CS row 21's C row and the product uni-skip input are derived from it, and `jump_lane_row_matches_product_uniskip_input` pins the two under both values of `Π`. The optimized prover, the verifier and BlindFold lowering consume the symbolic expression, the reference prover brute-forces the lane's left-hand side, and no mode carries its own formula.
- **Per-cycle lane check.** For boolean flags, `Jump·(1 − NextIsNoop) = Jump − (1 − 2Π)·Halt − 2·HaltPanic` holds on a row exactly when the row carries none of the three flags, is an ordinary jump with a real successor, or is a halt followed by padding (or at `T − 1`) whose `HaltPanic` equals `Π`. Excluding the other assignments needs a field characteristic other than 2 and 3; the BN254 scalar field and the Akita prime both qualify. The lane alone rejects the `(Jump, Halt, HaltPanic)` tuples no instruction carries, `(0,1,·)`, `(0,0,1)` and `(1,0,1)`, under both values of `Π`. The residual `g(t) = LHS − RHS` is fixed before the outer challenges are drawn, so the uni-skip and remainder sumchecks force `g ≡ 0` except with probability `log T/|F|`.
- **Only a halt precedes padding.** Padding rows have address 0, and the shifted columns are 0 at `T − 1`. A real row that is not a jump sets `NextUnexpandedPC` from its own address through row 15 or row 16, which cannot give 0, since bytecode addresses are at least `RAM_START_ADDRESS` and branch offsets are 13-bit immediates. So the row before padding is a jump, and lane 3 makes it a halt. A padding row keeps the next address at 0 (row 16), so no real row follows padding, and the real rows form a prefix that ends at exactly one halt.
- **I1: flags sit exactly on the halts.** `Halt` implies `Jump`, `HaltPanic` implies `Halt`, and only `JoltSuccess` and `JoltPanic` carry them; both are table-free. Registry tests pin this (`halt_flags_mark_exactly_the_halt_kinds`, `table_free_jumps_are_exactly_the_halts`). Bytecode read-RAF binds each cycle's flags to its bytecode row; full-mode verification recomputes the rows' flags from their kinds, and committed mode trusts preprocessing, as for every flag.
- **I2′: a halt is a whole source instruction.** Its `pc_map` slot has length 1. `BytecodePCMapper::try_new` rejects a multi-row instruction that contains a halt (`PreprocessingError::HaltInSequence`), and expansion rejects a halt among the final rows of a multi-row sequence, including halts reached through nested or inline expansion (`ExpansionError::HaltInSequence`). Without I2′, a halt as the last row of a longer sequence would end the proof inside an instruction.
- **I3′: the entry row is a real instruction.** Bytecode read-RAF pins cycle 0 to the key's entry row; were that row padding, an all-padding trace would verify under either value of `Π`. `ProgramPreprocessing::validate_entry_row` runs in `validate_inputs` and `validate_inputs_from_parts`: full mode requires `entry_instruction()` and fails with `VerifierError::EntryNotInstruction`; committed mode rejects entry index 0 and trusts the authenticated key for the rest. The Dory full-program prover and the SDK's shared preprocessing refuse a padding entry when they build `JoltSharedPreprocessing`; every prover's stage 0 refuses it through `validate_inputs_from_parts` (`EntryNotInstruction`). Every prover also refuses an empty trace, or one that does not end at a halt (`ProverError::TraceDoesNotEndInHalt`).
- **Guard word.** Mask N of the RAM read-write sumcheck covers `[io_end, RAM_START_ADDRESS)`, which now starts at the guard word, so the constraints reject any access to it, not only the tracers. `MemoryLayout::try_new` reserves the word; verifier-key admission does not recheck it, since the key's layout is trusted preprocessing.
- **Tracer agreement.** Every driver (eager, chunked, two-pass parallel replay, lazy and the native x86 tracer) stops at the first halt and reports `panic` exactly when the halt is `EBREAK`'s. A run that stops anywhere else, such as at a `j .` self-loop, fails with `TraceError::EndedWithoutHalt`.

### Non-Goals

- Public exit codes. `EBREAK` has no operand; an exit code needs another row, a ValIo word or a custom failure instruction, and nothing consumes one yet.
- Running unmodified ELFs whose runs end at `j .`. The success halt is one custom word, `0x0000305B`; foreign toolchains must emit it.
- Proving the end rule in Lean. The jolt-qed project will re-prove its theorems against the new constraints separately.

## Evaluation

### Acceptance Criteria

- [ ] The RV64 case table accepts each halt as the last real row under its own claim and rejects it under the opposite claim (lane 3); rejects an ordinary jump into padding, a jump to 0 at the last row and every flag tuple no instruction carries under both claims (lane 3); and rejects a halt followed by a real row and a final `j .` (row 14).
- [ ] Row 21's C row equals the product uni-skip lane-3 input under both claims.
- [ ] The optimized and reference product uni-skip fronts agree under both claims, with nonzero halt openings, and the optimized front given the opposite claim disagrees; the outer kernels' parity rows, default and streaming, include success and panic halt rows.
- [ ] Engine twins: halts at the domain boundary prove identically on both engines; truncated or misplaced ends (an ordinary jump, `j .`, a mid-expansion end, a halt followed by a real row) are rejected in stage 1; each halt under the wrong claim and a jump to 0 into padding are rejected in stage 2; a forged `HaltPanic` column is rejected by bytecode read-RAF in stage 6a.
- [ ] Every e2e matrix case, in clear, ZK and Akita modes for both instruction profiles, verifies under its panic status and is rejected under the opposite one; `panic_halt` proves `panic = true`.
- [ ] Verification under a key whose entry names a padding row fails with `EntryNotInstruction` in full and committed mode, in clear, ZK, Akita and field-inline builds; the prover rejects a padding entry and an empty trace.
- [ ] Preprocessing and expansion reject a halt inside a multi-row instruction; the expansion golden fixtures are unchanged.
- [ ] A verifier witness that loads or stores the guard word is rejected by the unmapped-word term in stage 2, not only by the tracers.
- [ ] Every layout keeps an unmapped guard word after the outputs; the default layout keeps its input and output addresses.
- [ ] Every tracer driver ends at the halt and reports `panic` from its kind; the public I/O memory is the same for both panic claims.

### Testing Strategy

The e2e matrix (`crates/jolt-prover/tests/e2e_matrix.rs`) runs in clear, ZK and Akita modes for the ordinary and field-inline profiles; after each proof verifies, it verifies the same proof under the flipped panic claim and expects a rejection. That check shows only that the statement binds the claim: `Π` is a statement word, so the flipped bit diverges the transcript before stage 1, whatever lane 3 computes. The R1CS case table (`execution_witness_tests` in `crates/jolt-claims/src/protocols/composed/r1cs/rv64.rs`) checks the end rule per row with the full-trace matrix for both claims. `crates/jolt-prover/tests/engine_twins.rs` (`mod halting`, base build) replays forged stage-1 and stage-2 proofs through the verifier's stages: the honest prover's final-claim self-check refuses to emit a proof for an unsatisfiable witness, so malicious-witness coverage is stage by stage. Its `lane_three_binds_the_claimed_panic_bit` isolates lane 3: it keeps the transcript and flips `Π` only in the product uni-skip input. In the ZK, Akita and field-inline builds, coverage is the shared symbolic lane and its `Π` checks: `lane_three_reads_the_claimed_panic_bit` and `jump_lane_row_matches_product_uniskip_input` (jolt-claims), `uniskip_front_reads_the_claimed_panic_bit` (the optimized product front), `composed_input_claim_matches_five_lane_fold` (the field-inline verifier) and `lowered_uniskip_input_matches_the_clear_claim` (BlindFold's lowering). The honest e2e `panic_halt` and success cases cover the production call sites that pass `Π` to lane 3, since a constant or negated `Π` there would fail one of them, and the entry-admission soundness tests (`crates/jolt-verifier/tests/soundness/entry_admission.rs`) cover I3′. The verifier fixture, tamper and soundness suites, the fuzz bundles and the Fiat–Shamir inventory are regenerated. Both `host` and `host,zk` builds must pass.

### Performance

- Spartan outer: one more opening (two flag openings replace `ShouldJump`): 36 openings in base builds, 51 with `field-inline`, which is 32 bytes more in a clear Dory proof (fibonacci at 2^20: 86,675 to 86,707 bytes). The outer uni-skip keeps 19 equality rows (29 with `field-inline`) and its spare slot. The outer kernel accumulates the circuit-flag inputs in a loop over `CIRCUIT_FLAGS`.
- Product virtualization: one more derived public; rounds, degree and openings are unchanged.
- Committed-bytecode lanes: two more flag lanes; the lane count still pads to 512, so no commitment grows, though every committed bytecode commitment changes.
- Trace rows: `is_noop()` reads `bytecode_pc`, and `instruction_flags()` reassembles the stored bits; both are a few ALU operations on the hot row.
- Guests: the epilogue drops one store. The padded I/O region halves for layouts whose I/O word count is one less than a power of two, no I/O included, and is unchanged otherwise.

## Design

### Architecture

| Change | Crates |
|---|---|
| Halt flags | `jolt-riscv` (flags, registry, trace-row metadata), `jolt-lookup-tables` (tests) |
| End rule and panic claim | `jolt-claims` (R1CS rows 14 and 21, `PRODUCT_JUMP_LANE_OUTPUT`, `ClaimedPanic`, outer openings), `jolt-kernels` (outer and product kernels, default and streaming, `ProductUniskipUpstream`), `jolt-witness`, `jolt-verifier` (stages 1–2, BlindFold), `jolt-prover`, `z3-verifier` and `zklean-extractor` (full-trace matrices per claim) |
| Memory and device | `common` (layout, device), `jolt-program` (ValIo, unmapped-word intervals), `jolt-witness`, `tracer`, `jolt-tracer-x86` |
| SDK | `jolt-sdk` (epilogue, exit paths), the memory-ops guest |
| Admission | `jolt-program` (`HaltInSequence`, `entry_instruction`), `jolt-verifier` (`validate_entry_row`), `jolt-prover` (preprocessing, `TraceDoesNotEndInHalt`) |

The halt instructions themselves (decoding `jolt.success`, `EBREAK`'s `Panic` row, every tracer driver stopping at a halt, the SDK emitting `jolt.success`) and the unmapped-word mask N are prerequisites, not part of this change.

`Π` crosses the optimized `UniskipKernel` seam in `ProductUniskipUpstream { openings, claimed_panic }`, because the optimized front rebuilds the uni-skip input at its base nodes; the reference front brute-forces the lane and ignores it. The full-trace builders take the claim as a parameter (`rv64_trace_constraints(claimed_panic)`), since a public column cannot express the product `Π·Halt`. The program-preprocessing digest domains move to `v4` (Dory) and `v3` (Akita), field-inline variants included: the memory layout loses two fields, and a key for the old protocol must not match one for the new.

### Alternatives Considered

- **Keep the flag words and constrain the end with a `CanEnd` flag** (the earlier #2408 Change 3). It keeps the standard ISA, but panic stays a memory convention whose last write wins, the end rule needs an extra equality row (spending the last spare uni-skip slot), and runs end at replayable self-jumps.
- **`EBREAK` as the success halt.** GCC lowers `__builtin_trap()` to `ebreak`, and LLVM lowers breakpoints to it, so a trap would verify as a clean exit.
- **A custom `PANIC` instruction as well.** `EBREAK` already lowers to the `Panic` row; a custom instruction adds a decoder case and a decode fact, and its only use would be an exit code.
- **V1: each halt stores `1 + kind` to a status word.** No `Π` in claims and no entry check, at the cost of a four-row store recipe per halt, a status word in memory, and the most Lean work.
- **V2: an equality row `Halt·(Imm − Π) = 0`.** One flag and no extra opening, but it spends the spare uni-skip slot and puts `Π` into the linear outer constants.
- **The `+2` encoding `Jump − (1 + 2Π)·Halt + 2·HaltPanic`.** Equivalent, but it gives `Halt` the coefficient −3 when `Π = 1`, which the z3 and zklean coefficient tables reject; the `−2` form keeps every lane-3 coefficient in `{±1, −2}`.
- **An `ECALL` halt selected by a register.** The stop becomes value-dependent, and `ECALL` is already the M-mode trap.

## Documentation

The book's R1CS page gains "Ending a run" (row 14, lane 3 and its case table, the admission checks); the RAM page's mask-N and output-check text describe the guard word and drop the status words; the foreign-toolchain contract in the guest page says how a run ends; the testing-gates page records the wrong-claim check. `field-inline-protocol.md`, `unified-trace-row.md`, `proof-trace-row-layout.md`, `m-mode-trap-csr-semantics.md` and `ram-guards.md` are updated.

## Execution

Fixture and digest changes: `FIXTURE_MAGIC` moves to `JVCF0016`, the program-digest goldens and the six fuzz bundles are regenerated, and the Fiat–Shamir source inventory loses `MemoryLayout.panic` and `MemoryLayout.termination`. The BlindFold shape golden is unchanged: the stage-1 hidden claims grow from 35 to 36 without adding an output-claim row.

## References

- jolt-qed (Lean model of Jolt against the Sail RISC-V model): restrictions 5a `PanicStoresOne`, 5b `TerminationThenStall` and 5d `ClaimsNoPanic`.
- [`ram-guards.md`](./ram-guards.md): the unmapped-word mask N that rejects accesses to the guard word.
- [`m-mode-trap-csr-semantics.md`](./m-mode-trap-csr-semantics.md): `EBREAK` halts the hart.
