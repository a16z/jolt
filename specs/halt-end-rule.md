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
2. **End rule.** `ShouldJump` is deleted; its serialized slot becomes `JoltVirtualPolynomial::Reserved12`. Row 14 (`NextUnexpPCEqLookupIfJump`) becomes `Jump·(NextUnexpandedPC − LookupOutput) = 0`, so every jump target is checked and a halt jumps to address 0. Product row 21 (`NextIsNoopEqIsNoopUnlessHalt`, lane 3) takes the factors `2` and `NextIsNoop − IsNoop`, and its output becomes `(2 − Π)·Halt + HaltPanic`, where `Π` is the claimed panic bit. The product remainder opens `IsNoop` at its point in place of `Jump`, and the stage-2 fold of bytecode read-RAF binds that opening.
3. **Panic claim.** `Π` enters a claim only as the appended derived public `SpartanProductVirtualizationPublic::ClaimedPanic = F::from(public_io.panic)`. `PRODUCT_HALT_LANE_OUTPUT` in `jolt-claims` owns the lane-3 output; R1CS row 21, the product uni-skip input, the clear verifier, the optimized prover kernel and BlindFold use it, and the reference kernel computes the lane from the trace. The product remainder relation rejects the id.
4. **Memory.** `MemoryLayout` loses `panic` and `termination`. The I/O region pads to `next_pow2(io_words + 1)` words, so a reserved word, the first word of the I/O gap, always follows the outputs: `io_end = output_end ≤ RAM_START_ADDRESS − 8`. The device has no panic latch; the tracers set `JoltDevice.panic` from the halt kind. ValIo holds only the input and output segments, and the witness synthesizes no flag words.
5. **SDK.** The `#[jolt::provable]` epilogue writes the outputs, then executes `jolt.success`. `jolt_panic` is deleted; `__platform_exit` and `__platform_abort` execute `ebreak`.
6. **Admission.** A halt must be a whole source instruction (I2′), and the entry row must be a real instruction (I3′).
7. **Trace rows.** The trace-row metadata word stores the six instruction flags other than `IsNoop`, which the row derives from `bytecode_pc == 0`, and the immediate's sign moves to row-control bit 7, so 26 circuit flags fit.

### Invariants

- **One owner for lane 3.** `PRODUCT_HALT_LANE_OUTPUT` is the only statement of the lane-3 output. R1CS row 21's C row and the product uni-skip input are derived from it, and `halt_lane_row_matches_product_uniskip_input` pins the two under both values of `Π`. `PRODUCT_HALT_LANE_LEFT_FACTOR` is the only statement of the lane's constant factor; row 21's A row, the remainder relation and both prover kernels read it. The optimized prover, the verifier and BlindFold lowering consume the symbolic expression, the reference prover brute-forces the lane's left-hand side, and no mode carries its own formula.
- **Per-cycle lane check.** For boolean flags, `2·(NextIsNoop − IsNoop) = (2 − Π)·Halt + HaltPanic` holds exactly when the row is real, carries neither halt flag and has a real successor; or is padding, carries neither halt flag and has a padding successor (or is at `T − 1`); or is a real row with `Halt` whose `HaltPanic` equals `Π`, followed by padding (or at `T − 1`). The left side is −2, 0 or 2. The output is 0 on a row with neither flag, 2 on a halt whose `HaltPanic` equals `Π`, and 1 or 3 on a halt of the other kind or on `HaltPanic` without `Halt`. Excluding every other assignment needs a field characteristic other than 2, 3 and 5, which keeps −2, 0, 1, 2 and 3 distinct; the BN254 scalar field and the Akita prime both qualify. The lane does not read `Jump`: that a halt is a jump is part of I1. `halt_lane_truth_table` checks row 21 over all sixteen assignments of the four cells it reads, under both claims. The residual `g(t) = LHS − RHS` is fixed before the outer challenges are drawn, so the uni-skip and remainder sumchecks force `g ≡ 0` except with probability `log T/|F|`.
- **Only a halt precedes padding, and only padding follows it.** Cycle 0 is the entry row, which is real (I3′). By the lane, a real row that is not a halt has a real successor, a halt of the claimed kind has a padding successor, and a padding row has a padding successor. `NextIsNoop` is 1 at `T − 1`, so the last row is a halt or padding. The real rows therefore form a prefix that ends at exactly one halt. The lane reads no address, so the argument does not depend on where the bytecode is loaded, on branch displacements or on jump targets, and it needs no fact about the bytecode table beyond I1 and I3′.
- **I1: flags sit exactly on the halts.** `Halt` implies `Jump`, `HaltPanic` implies `Halt`, and only `JoltSuccess` and `JoltPanic` carry them; both are table-free. Registry tests pin this (`halt_flags_mark_exactly_the_halt_kinds`, `table_free_jumps_are_exactly_the_halts`). Bytecode read-RAF binds each cycle's flags to its bytecode row; full-mode verification recomputes the rows' flags from their kinds, and committed mode trusts preprocessing, as for every flag.
- **I2′: a halt is a whole source instruction.** Its `pc_map` slot has length 1. `BytecodePCMapper::try_new` rejects a multi-row instruction that contains a halt (`PreprocessingError::HaltInSequence`), and expansion rejects a halt among the final rows of a multi-row sequence, including halts reached through nested or inline expansion (`ExpansionError::HaltInSequence`). Without I2′, a halt as the last row of a longer sequence would end the proof inside an instruction.
- **I3′: the entry row is a real instruction.** Bytecode read-RAF pins cycle 0 to the key's entry row; were that row padding, an all-padding trace would verify under either value of `Π`. `ProgramPreprocessing::validate_entry_row` runs in `validate_inputs` and `validate_inputs_from_parts`: full mode requires `entry_instruction()` and fails with `VerifierError::EntryNotInstruction`; committed mode rejects entry index 0 and trusts the authenticated key for the rest. The Dory full-program prover and the SDK's shared preprocessing refuse a padding entry when they build `JoltSharedPreprocessing`, and the Dory and Akita committed-key builders refuse one before committing (`validate_committed_mode`); every prover's stage 0 refuses it through `validate_inputs_from_parts` (`EntryNotInstruction`). Every prover also refuses an empty trace, or one that does not end at a halt (`ProverError::TraceDoesNotEndInHalt`).
- **The word at `io_end`.** `MemoryLayout::try_new` reserves it, so every layout has a nonempty I/O gap `[io_end, RAM_START_ADDRESS)`. It is the first I/O-gap word, ordinary zero-initialized read/write memory ([`accessible-ram-padding.md`](./accessible-ram-padding.md)); the output check ends at `output_end`, so no value stored there is an output.
- **Tracer agreement.** Every driver (eager, chunked, two-pass parallel replay, lazy and the native x86 tracer) stops at the first halt and reports `panic` exactly when the halt is `EBREAK`'s. A run that stops anywhere else, such as at a `j .` self-loop, fails with `TraceError::EndedWithoutHalt`.

### Non-Goals

- Public exit codes. `EBREAK` has no operand; an exit code needs another row, a ValIo word or a custom failure instruction, and nothing consumes one yet.
- Running unmodified ELFs whose runs end at `j .`. The success halt is one custom word, `0x0000305B`; foreign toolchains must emit it.
- Proving the end rule in Lean. The jolt-qed project will re-prove its theorems against the new constraints separately.

## Evaluation

### Acceptance Criteria

- [ ] The RV64 case table accepts each halt as the last real row under its own claim and rejects it under the opposite claim (lane 3); rejects an ordinary jump into padding and a jump to 0 at the last row under both claims (lane 3); admits, over every boolean `(IsNoop, Halt, HaltPanic, NextIsNoop)`, exactly the end-rule steps (lane 3); and rejects a halt followed by a real row and a final `j .` (row 14).
- [ ] Row 21's C row equals the product uni-skip lane-3 input under both claims.
- [ ] The optimized and reference product uni-skip fronts agree under both claims, with nonzero halt openings; the outer kernels' parity rows, default and streaming, include both halt flag tuples.
- [ ] Both bytecode read-RAF phases, default and streaming kernels, match the reference tier on runs that end at `jolt.success` and at `EBREAK`, with padding after the halt and without, over a bytecode table that holds both halts.
- [ ] Engine twins: halts at the domain boundary prove identically on both engines; truncated or misplaced ends (an ordinary jump, `j .`, a mid-expansion end, a halt followed by a real row) are rejected in stage 1; each halt under the wrong claim, and a jump or a taken branch to address 0 into padding, are rejected in stage 2; a forged `HaltPanic` column is rejected by bytecode read-RAF in stage 6a.
- [ ] Every e2e matrix case, in clear, ZK and Akita modes for both instruction profiles, verifies under its panic status and is rejected under the opposite one; `panic_halt` proves `panic = true`.
- [ ] Verification under a key whose entry names a padding row fails with `EntryNotInstruction` in full and committed mode, in clear, ZK, Akita and field-inline builds; the prover and the committed-key builders reject a padding entry, and the prover an empty trace.
- [ ] Preprocessing and expansion reject a halt inside a multi-row instruction; the expansion golden fixtures are unchanged.
- [ ] Every layout keeps a reserved I/O-gap word after the outputs; the default layout keeps its input and output addresses.
- [ ] Every tracer driver ends at the halt and reports `panic` from its kind; the public I/O memory is the same for both panic claims.

### Testing Strategy

The e2e matrix (`crates/jolt-prover/tests/e2e_matrix.rs`) runs in clear, ZK and Akita modes for the ordinary and field-inline profiles; after each proof verifies, it verifies the same proof under the flipped panic claim and expects a rejection. That check shows only that the statement binds the claim: `Π` is a statement word, so the flipped bit diverges the transcript before stage 1, whatever lane 3 computes. The R1CS case table (`execution_witness_tests` in `crates/jolt-claims/src/protocols/composed/r1cs/rv64.rs`) checks the end rule per row with the full-trace matrix for both claims. `crates/jolt-prover/tests/engine_twins.rs` (`mod halting`, base build) replays forged stage-1 and stage-2 proofs through the verifier's stages: the honest prover's final-claim self-check refuses to emit a proof for an unsatisfiable witness, so malicious-witness coverage is stage by stage. Its `lane_three_binds_the_claimed_panic_bit` isolates lane 3: it keeps the transcript and flips `Π` only in the product uni-skip input. In the ZK, Akita and field-inline builds, coverage is the shared symbolic lane and its `Π` checks: `halt_lane_row_matches_product_uniskip_input` (jolt-claims), `synthetic_parity_with_reference_kernels` (the optimized product front, whose `parity_run` checks both claims), `composed_input_claim_matches_five_lane_fold` (the field-inline verifier) and `lowered_uniskip_input_matches_the_clear_claim` (BlindFold's lowering). The honest e2e `panic_halt` and success cases cover the production call sites that pass `Π` to lane 3, since a constant or negated `Π` there would fail one of them, and the entry-admission soundness tests (`crates/jolt-verifier/tests/soundness/entry_admission.rs`) cover I3′. The verifier fixture, tamper and soundness suites, the fuzz bundles and the Fiat–Shamir inventory are regenerated. Both `host` and `host,zk` builds must pass.

### Performance

- Spartan outer: one more opening (two flag openings replace `ShouldJump`): 36 openings in base builds, 51 with `field-inline`, which is 32 bytes more in a clear Dory proof (fibonacci at 2^20: 86,675 to 86,707 bytes). The outer uni-skip keeps 19 equality rows (29 with `field-inline`) and its spare slot. The outer kernel accumulates the circuit-flag inputs in a loop over `CIRCUIT_FLAGS`.
- Product virtualization: one more derived public; rounds, degree and the opening count are unchanged (`IsNoop` replaces `Jump` at the product point).
- Committed-bytecode lanes: two more flag lanes; the lane count still pads to 512, so no commitment grows, though every committed bytecode commitment changes.
- Trace rows: `is_noop()` reads `bytecode_pc`, and `instruction_flags()` reassembles the stored bits; both are a few ALU operations on the hot row.
- Guests: the epilogue drops one store. The padded I/O region halves for layouts whose I/O word count is one less than a power of two, no I/O included, and is unchanged otherwise.

## Design

### Architecture

| Change | Crates |
|---|---|
| Halt flags | `jolt-riscv` (flags, registry, trace-row metadata), `jolt-lookup-tables` (tests) |
| End rule and panic claim | `jolt-claims` (R1CS rows 14 and 21, `PRODUCT_HALT_LANE_OUTPUT`, `ClaimedPanic`, outer openings, the stage-2 bytecode fold), `jolt-kernels` (outer and product kernels, default and streaming, `ProductUniskipUpstream`), `jolt-witness`, `jolt-verifier` (stages 1–2, BlindFold), `jolt-prover`, `z3-verifier` and `zklean-extractor` (full-trace matrices per claim) |
| Memory and device | `common` (layout, device), `jolt-program` (ValIo), `jolt-witness`, `tracer`, `jolt-tracer-x86` |
| SDK | `jolt-sdk` (epilogue, exit paths), the memory-ops guest |
| Admission | `jolt-program` (`HaltInSequence`, `entry_instruction`), `jolt-verifier` (`validate_entry_row`), `jolt-prover` (preprocessing, `TraceDoesNotEndInHalt`) |

The halt instructions themselves (decoding `jolt.success`, `EBREAK`'s `Panic` row, every tracer driver stopping at a halt, the SDK emitting `jolt.success`) and the unmapped-word mask N, since retired by [`accessible-ram-padding.md`](./accessible-ram-padding.md), were prerequisites, not part of this change.

`Π` crosses the optimized `UniskipKernel` seam in `ProductUniskipUpstream { openings, claimed_panic }`, because the optimized front rebuilds the uni-skip input at its base nodes; the reference front brute-forces the lane and ignores it. The full-trace builders take the claim as a parameter (`rv64_trace_constraints(claimed_panic)`), since a public column cannot express the product `Π·Halt`. The program-preprocessing digest domains move to `v4` (Dory) and `v3` (Akita), field-inline variants included: the memory layout loses two fields, and a key for the old protocol must not match one for the new.

### Alternatives Considered

- **Keep the flag words and constrain the end with a `CanEnd` flag** (the earlier #2408 Change 3). It keeps the standard ISA, but panic stays a memory convention whose last write wins, the end rule needs an extra equality row (spending the last spare uni-skip slot), and runs end at replayable self-jumps.
- **`EBREAK` as the success halt.** GCC lowers `__builtin_trap()` to `ebreak`, and LLVM lowers breakpoints to it, so a trap would verify as a clean exit.
- **A custom `PANIC` instruction as well.** `EBREAK` already lowers to the `Panic` row; a custom instruction adds a decoder case and a decode fact, and its only use would be an exit code.
- **V1: each halt stores `1 + kind` to a status word.** No `Π` in claims and no entry check, at the cost of a four-row store recipe per halt, a status word in memory, and the most Lean work.
- **V2: an equality row `Halt·(Imm − Π) = 0`.** One flag and no extra opening, but it spends the spare uni-skip slot and puts `Π` into the linear outer constants.
- **Lane 3 guarded by `Jump`: `Jump·(1 − NextIsNoop) = Jump − (1 − 2Π)·Halt − 2·HaltPanic`.** It keeps the old lane's factors but constrains only jump rows. A non-jump row before padding is then excluded only through rows 15 and 16 and two facts every bytecode table would have to satisfy: no real row below `RAM_START_ADDRESS`, and no branch whose offset is minus its own address. The chosen lane constrains every row and needs neither.
- **Unit factors: `NextIsNoop − IsNoop = (1 − 2Π)·Halt + 2·HaltPanic`.** A success halt's output under `Π = 1` is −1, which is also the left side of a step out of padding, so this lane holds on a no-op row that carries `Halt` and has a real successor. Its mirror `(1 + 2Π)·Halt − 2·HaltPanic` has the same gap under `Π = 0`. Either would rest on no kind being a no-op and a halt at once. `(1 + 2Π)·Halt + (2 − 4Π)·HaltPanic` has no such row but gives `Halt` the coefficient 3 when `Π = 1`, which the z3 and zklean coefficient tables reject. The chosen lane makes the left side even and every wrong output odd, and keeps every lane-3 coefficient in `{±1, 2}`.
- **An `ECALL` halt selected by a register.** The stop becomes value-dependent, and `ECALL` is already the M-mode trap.

## Documentation

The book's R1CS page gains "Ending a run" (row 14, lane 3 and its case table, the admission checks); the bytecode page's stage-2 fold reads `IsNoop` in place of `Jump`; the RAM page's output-check text drops the status words; the foreign-toolchain contract in the guest page says how a run ends; the testing-gates page records the wrong-claim check. `field-inline-protocol.md`, `unified-trace-row.md`, `proof-trace-row-layout.md`, `m-mode-trap-csr-semantics.md` and `ram-guards.md` are updated.

## Execution

Fixture and digest changes: `FIXTURE_MAGIC` moves to `JVCF0020`, the program-digest goldens and the six fuzz bundles are regenerated, and the Fiat–Shamir source inventory loses `MemoryLayout.panic` and `MemoryLayout.termination`. The BlindFold shape golden is unchanged: the stage-1 hidden claims grow from 35 to 36 without adding an output-claim row.

## References

- jolt-qed (Lean model of Jolt against the Sail RISC-V model): restrictions 5a `PanicStoresOne`, 5b `TerminationThenStall` and 5d `ClaimsNoPanic`.
- [`accessible-ram-padding.md`](./accessible-ram-padding.md): the I/O gap, which starts at the reserved word after the outputs, is zero-initialized read/write memory.
- [`m-mode-trap-csr-semantics.md`](./m-mode-trap-csr-semantics.md): `EBREAK` halts the hart.
