# Spec: Byte-Addressable Memory — Sub-Word and Misaligned Load/Store Support

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @mzhu, Claude                  |
| Created     | 2026-07-30                     |
| Status      | draft (design exploration)     |
| PR          |                                |

## Summary

Jolt's RAM argument is doubleword-addressable: `remap_address` maps guest byte address to witness
index as `(addr - lowest_address) / 8` (`crates/jolt-prover-legacy/src/zkvm/ram/mod.rs:120`), and
every access that is not a naturally-aligned `LD`/`SD` is emulated by an inline sequence over the
containing doubleword. Two problems follow. First, **misaligned accesses are unprovable**: the
sequences carry alignment asserts (`VirtualAssertWordAlignment` etc.), and a doubleword-crossing
access has no expansion at all — a guest that dereferences a misaligned pointer (unsafe Rust, FFI,
packed structs) kills the trace. The Ethereum Foundation's zkVM standard prescribes
**RV64IM + Zicclsm** as the common compilation target, where Zicclsm means misaligned loads and
stores must execute *correctly, if slowly* — an explicitly permitted slow path. Second, the
emulation sequences are expensive: a traced `LB` costs 8 cycles, `LH` 9, `SB` 13, `SW` 14
(the `SLL`s inside them expand further into `VirtualPow2`+`MUL`). On `btreemap`, **41% of all
cycles are memory-op cycles and 22.3% of the entire trace is pure sub-word emulation overhead**.

This spec explores the design space — including modifying the Twist memory argument itself — and
recommends **keeping Twist doubleword-addressable and unchanged**, adding byte semantics at the
instruction layer: (a) fused extract/merge lookup tables, (b) a per-cycle intra-doubleword
`Offset` witness folded into the RAF-evaluation identity, and (c) **dual-expansion dispatch** —
two bytecode expansions per load/store guest PC (non-crossing and crossing), prover-selected per
execution instance, each variant constraint-pinned to its applicability. The result is full
Zicclsm compliance where the aligned path gets *faster* than today (sub-word loads drop from 8–9
cycles to 2, narrow stores from 13–14 to 5) and the doubleword-crossing path costs a bounded 5–11
cycles. Measured on `btreemap`, the trace shrinks ~17% (~21% with the optional fusion tier).

## Background and requirements

### Why now

- **Ethproofs / zkEVM standards v0**: the EF settled on RV64IM+Zicclsm as the prescribed zkVM
  target ISA ([zkEVM standards v0](https://zkevm.ethereum.foundation/blog/zkevm-standards-v0-release)).
  Zicclsm's contract (RVA profiles) is that misaligned loads/stores to main memory are *supported*
  — they may be slow and non-atomic, but they must be correct. The standard frames this as a
  safety net: a compiler-introduced misaligned access must not become a liveness failure.
  With Zicclsm in the target, LLVM freely emits wide loads/stores for possibly-misaligned
  pointers (`memcpy` tails, packed structs, `ptr::read_unaligned`), so the zkVM must execute any
  alignment for any width.
- **Unsafe Rust / FFI**: even on today's strict-align target, code that lies about alignment
  works on real hardware but panics Jolt's tracer.

### What Zicclsm does *not* require

- Misaligned LR/SC/AMOs (the A extension is excluded from the EF target anyway; Jolt's AMO
  expansions keep their alignment asserts).
- Speed: a 5–10× slow path for crossing accesses is compliant. This asymmetry is the single most
  important design lever.

### Current implementation (verified)

Expansions live in `crates/jolt-program/src/expand/memory/` and are shared by the tracer
(`tracer/src/instruction/mod.rs:750` calls `jolt_program::expand::expand_instruction`) and
bytecode preprocessing:

| Source op | Expansion | Traced cycles | RAM cycles |
|---|---|---|---|
| `LD`, `SD` (8B-aligned) | native | 1 | 1 |
| `LB`/`LBU` | load containing dw + extract, **no alignment assert** (`shared.rs:35`) | 8 | 1 |
| `LW` | `VirtualAssertWordAlignment` + extract (`lw.rs:7`) | 8 | 1 |
| `LH`/`LHU`, `LWU` | halfword/word assert + extract (`shared.rs:88`, `lwu.rs`) | 9 | 1 |
| `SB` | RMW via masked-XOR, no assert (`shared.rs:484`) | 13 | 2 |
| `SH` | + halfword assert (`sh.rs:10`) | 14 | 2 |
| `SW` | word assert + RMW, mask built via ORI+SRLI (`sw.rs:8`) | 15 | 2 |

Traced cycles exceed the static recipes because register-amount shifts themselves expand
(`SLL` → `VirtualPow2` + `MUL`, `SRL` → `VirtualShiftRightBitmask` + `VirtualSRL`). Misalignment
today is a **host-process abort, not a trap**: the MMU wrappers assert alignment
(`tracer/src/emulator/mmu.rs:329-470`) and the `TrapType::*AddressMisaligned` variants are never
constructed; at the proof level a fabricated misaligned address fails the alignment-assert
lookups (`AssertLookupOne`, constraint 11) or, for `LD`/`SD` (which carry **no** assert row),
becomes unrepresentable under the RAF-evaluation identity below. Notably the underlying
emulator memory is already byte-capable and misalignment-safe (`tracer/src/emulator/memory.rs`
falls back to byte-by-byte across doubleword boundaries); only the MMU asserts and three latent
trace-recording bugs stand in the way (see Implementation notes).

Key protocol anchors:

- **R1CS (uniform, 22 constraints / 38 wires per cycle)**, `crates/jolt-r1cs/src/constraints/rv64.rs`:
  - `RamAddress = Rs1Value + Imm` when Load/Store (constraint 0) — the wire is the raw **byte**
    effective address; `RamAddress = 0` on idle cycles (constraint 1).
  - Loads: `rv = wv` and `rv = RdWriteValue` (constraints 2–3). Stores: `wv = Rs2Value`
    (constraint 4). Same-cycle read-modify-write is native to the merged-`ra` Twist instance.
  - `NextPC = PC + 1` is *forced* inside virtual sequences (constraint 17, guard
    `VirtualInstruction − IsLastInSequence`), and sequence entry only at heads (constraint 18 via
    `DoNotUpdateUnexpandedPC`). The constraint-17 NOTE documents the NextPC-uniqueness argument.
- **RAF evaluation** ties the byte-address wire to the one-hot doubleword index:
  `RamAddress(r_cycle) = Σ_k ra(k, r_cycle) · unmap(k)` with `unmap(k) = 8k + lowest_address`
  (`UnmapRamAddressPolynomial`, `poly/identity_poly.rs:554`; sumcheck in `zkvm/ram/raf_evaluation.rs`).
  **This affine tie is where doubleword alignment is enforced today**, and where byte addressing
  slots in.
- **Bytecode PC mapping** keys rows by `(guest address, virtual_sequence_remaining)`
  (`BytecodePCMapper`, `crates/jolt-program/src/preprocess/bytecode.rs`), with
  `IsFirstInSequence`/`IsLastInSequence` as per-row circuit flags. `NextUnexpandedPC` is tied to
  the next row's address via the shift sumcheck; nothing except mapper keying assumes one
  expansion per guest PC.
- **Lookups**: one lookup per cycle over a 128-bit interleaved `(x, y)` index, 16 committed
  one-hot `ra` columns at the 8-bit chunk config (`zkvm/config.rs`). Tables that read only low
  bits of `y` are precedented with clean prefix-suffix decompositions (`Pow2Table` reads
  `y % 64`, `tables/pow2.rs`; `VirtualSRL` shifts by a `y` bitmask). Alignment asserts already
  consume `rs1 + imm` via the `AddOperands` flag path.
- **Instruction-input muxes** (`zkvm/spartan/instruction_input.rs:488`):
  `LeftInstructionInput = IsRs1·Rs1Value + IsPC·UnexpandedPC`,
  `RightInstructionInput = IsRs2·Rs2Value + IsImm·Imm`.

### Empirical grounding

Instruction mix (tracer histograms, this branch):

| Workload | Trace | Mem-op cycles | Sub-word loads | Narrow stores | Emulation overhead |
|---|---|---|---|---|---|
| `btreemap` (400 ops, 370k cycles) | 370,269 | 41.1% | 5,272 ops → 45,568 cyc (12.3%) | 3,413 ops → 45,736 cyc (12.4%) | **82,619 cyc = 22.3%** |
| `sha2-chain` (30 iters, 93k cycles) | 92,969 | — | 334 ops → 2,672 cyc (2.9%) | 1,056 ops → 15,168 cyc (16.3%) | ~19% (outside the inline) |
| `fibonacci` (100k, 1.1M cycles) | 1,101,216 | ~0.1% | ~0 | ~0 | ~0% |

Native `LD`+`SD` are 16.5% of `btreemap` cycles (9.4% + 7.0%) — doubleword traffic dominates, as
expected from LLVM codegen. Byte/halfword traffic is common in real code (B-tree key comparisons,
serde, string handling) but *every* such op currently pays ~8–14 cycles.

Prover cost structure (btreemap at 2^24, 52.3s wall / 232 kHz, thread-time sums by subsystem):
commitment work ≈ 272s, Dory opening proof ≈ 111s, instruction-lookup sumchecks ≈ 12.3s,
Spartan ≈ 4.5s, booleanity ≈ 3.8s, **all RAM sumchecks ≈ 2.4s (~0.6%)**, registers ≈ 1.9s.
Commitment cost is dominated by one-hot column count × nonzero density: 16 instruction-`ra`
columns (dense) vs ~2–3 bytecode-`ra` (dense) vs ~3 RAM-`ra` columns (nonzero only on memory
cycles). RAM's read/write-checking prover is built on sparse cycle-major matrices
(`zkvm/ram/read_write_checking.rs`), so idle cycles are already nearly free there.

**Consequences for the design space**: (1) trace length is the master cost driver — a cycle saved
anywhere pays across every subsystem; (2) additional *rarely-active* RAM-side columns or ports are
nearly free; (3) additional *dense* columns (or 8× lane columns active on every memory op) are
expensive; (4) making the RAM sumchecks somewhat more complex is nearly irrelevant at ~0.6%.

## Intent

### Goal

Support all RV64 load/store widths at any address (Zicclsm semantics) while *reducing* trace
length for aligned code, without modifying the Twist memory argument.

### Invariants

- RAM remains doubleword-addressable in Twist: `ra`/`Val`/`Inc`/K semantics unchanged; `rv`/`wv`
  remain full 64-bit doubleword values.
- Every sub-word or misaligned access is realized as 1–2 read-modify-write doubleword accesses,
  totally ordered in the single existing Twist instance (coherence for free).
- `RamAddress = Rs1Value + Imm` stays the guest byte address; the new identity
  `RamAddress − Offset = Σ_k ra(k)·(8k + lowest_address)` with `Offset ∈ [0, 8)` must hold on
  memory cycles and `Offset = 0` on idle cycles.
- Each bytecode expansion variant is applicability-pinned: a prover cannot execute the
  non-crossing variant of an access that crosses a doubleword boundary, or vice versa
  (constraint-enforced, not tracer-enforced).
- Any change to a sumcheck's `input_claim` (RAF evaluation gains the `−Offset` term) carries the
  matching `input_claim_constraint` update (BlindFold), the akita/lattice mirror, and the
  jolt-r1cs/legacy-r1cs pair — the standard two-R1CS sync obligation.
- Guest-visible semantics match the RISC-V spec (byte-exact against the reference emulator),
  including sign extension and the store-merge value on both crossing halves.
- `jolt-eval`'s `source_to_jolt_expansion_equivalence` invariant extends to the new expansions
  and both variants of each dual expansion.

### Non-Goals

- Misaligned LR/SC/AMO support (excluded from Zicclsm; A is not in the EF target). AMO
  expansions keep their alignment asserts.
- Byte-granular `Val`/`Inc` in Twist (see Alternatives).
- Single-cycle *misaligned* accesses (dual-port Twist) — the crossing path is deliberately slow.
- MMIO or device memory semantics.

## Design

The recommendation is tiered so each tier is independently shippable and independently
benchmarkable.

### Tier 0 — fused extract/merge lookup tables (no protocol changes)

Replace the shift/mask arithmetic inside the existing sequences with dedicated lookups. All
operands are register values, so this tier touches only `jolt-lookup-tables`, `jolt-riscv`
(instruction kinds), `jolt-program` expansions, and the tracer:

- `ExtractB/BU/H/HU/W/WU(x = dw, y = ea)` → the selected lane of `x` at byte offset `y & 7`,
  sign- or zero-extended. MLE: `Σ_{o∈[8]} eq(y_{0..3}, o) · lane_o(x)` — same species as
  `Pow2`'s low-bits-of-y dependence; needs a small new prefix/suffix family.
- `ShiftDataB/H/W(x = rs2, y = ea)` → `(x & width_mask) << 8·(y & 7)`.
- `MaskOldB/H/W(x = old_dw, y = ea)` → `x & ~(width_mask << 8·(y & 7))`.

New sequences (aligned-only, asserts kept): loads `[ADDI ea][ANDI dw][LD][Extract]` = 4 cycles;
stores `[ADDI][ANDI][LD][ShiftData][MaskOld][ADD][SD]` = 7 cycles. Traced savings vs today:
LB 8→4, LH 9→4 (halfword assert can fold into the Extract table's definition or stay 1 cycle),
SB 13→7, SW 14→7. On `btreemap` this alone recovers roughly half the 22% overhead.

Within-doubleword misalignment (e.g. `LW` at `ea ≡ 1 mod 8`) becomes *semantically* free here —
the extract tables handle any `ea & 7` — but stays gated behind the asserts until Tier 2 decides
variant dispatch. Tier 0 may relax the asserts from "naturally aligned" to "does not cross a
doubleword boundary" (a new assert table) as an intermediate Zicclsm step: this covers all
misaligned accesses except crossing ones.

**Tier 0 as implemented (PRs #1761, #1762, #1768).** The fused single-lookup extract sketched
above was rejected during implementation: with the offset delivered in *binary* form, the
per-offset lane functionals are linearly independent at every phase-boundary cut, so the table
has intrinsic prefix rank ≥ 8 per width (~16 new prefix/suffix families per width, per registry
side). The implementation instead follows the codebase's own SRL/SRA binary-to-positional
pattern, at one extra cycle per access: a rank-1 mask lookup converts the effective address to
the addressed lane's byte mask (`WindowMask{B,H,W}`, each reading only the offset bits its
alignment class allows, which also keeps every output in u64 range), and a constant-rank
extract pulls the lane through the mask (`PextSigned` = `pext(x,y) + σ·(2^64 − 2^popcount(y))`,
width-independent, full-domain testable; unsigned loads use plain `Pext`). Stores mirror this:
`ANDN` clears the lane and width-specialized `ShiftData{B,H,W}` tables
(`(rs2 mod 2^8w)·2^{8·offset}`, a rank-2 product over disjoint variables) place the store data;
a general `Pdep` table was analyzed and rejected (deposited bit positions depend on the suffix
popcount: prefix rank ~57). Total new machinery: 11 tables, 11 virtual instructions, 5 prefix
and 9 suffix families across both registry sides. Realized traced cycles: LB/LBU 8→5,
LH/LHU 9→6, LW 8→6, LWU 9→6, SB 13→8, SH 14→9, SW 15→9. Measured trace deltas:
`btreemap` −8.9%, `sha2-chain` −7.2%, `fibonacci` ~0%. Measured prover wall-time: parity at
fixed padded trace length (trace savings inside a power-of-two instance are invisible; the two
extra globally-materialized prefixes cost nothing measurable); −27% at a padding-boundary
crossing. The benefit unit is guest cycles per padded instance, not per-input wall time.

### Tier 1 — Offset via the memory cycle's idle lookup slot; floor addressing

Kill the `ADDI`/`ANDI` address-materialization cycles by letting the memory instruction itself
accept an unaligned effective address. The key enabler: **load/store cycles have an idle lookup
slot today** — `impl_lookup_table!(Ld, None)` with inputs `(0, 0)` and output `0`
(`crates/jolt-lookup-tables/src/instructions/riscv/ld.rs`), and every cycle's lookup index is
committed regardless. Repurpose it:

1. Give each load/store kind an `AddressOffset` table riding the existing `AddOperands` path
   (lookup input = `rs1 + imm = ea`, exactly how the alignment-assert tables already consume
   effective addresses). Output: `ea & 7` **when the executing variant applies**, else
   `(ea & 7) + 2^32` — an unsatisfiable marker (see soundness below). The table is a function of
   the low 3 bits of `y` plus a per-width boolean gate, the same MLE species as `Pow2`'s
   low-bits-of-`y` dependence.
2. Materialize `OffsetOrZero = LookupOutput · (Load + Store)` as one R1CS product wire (the
   `ShouldBranch = LookupOutput × Branch` pattern, `NUM_PRODUCT_VIRTUAL` 3→4; a product wire is
   needed because the RAF input claim is an MLE evaluation — a product of two openings is not
   the opening of the product).
3. Change the RAF-evaluation input claim from `RamAddress_claim` to
   `RamAddress_claim − OffsetOrZero_claim` (both opened at the Spartan-outer cycle point).
   `UnmapRamAddressPolynomial` itself is untouched. This is the *entire* Twist-side change;
   BlindFold `input_claim_constraint` supports sum-of-products terms and updates in lockstep,
   as does the akita mirror. Two details verified in code: the input claim carries a
   `mul_pow_2(phase3_cycle_rounds())` renormalization (`raf_evaluation.rs:125-133`) that the
   new term must share, and the non-zk `input_output_claims()` path
   (`raf_evaluation.rs:461-475`) must gain the same `− OffsetOrZero` term alongside the zk
   constraint.
4. Introduce offset-tolerant virtual loads/stores (`VirtualLoadDw`/`VirtualStoreDw`): identical
   to `LD`/`SD` except carrying an `AddressOffset` variant table — the one-hot `ra` encodes
   `floor((ea − lowest)/8)` (which `remap_address` already computes by truncation) and the
   lookup output absorbs `ea mod 8`.

**Soundness of the marker**: if the prover executes a variant whose applicability predicate
fails (e.g. the non-crossing `LW` variant at `ea & 7 = 6`, or plain aligned `LD` at `ea & 7 ≠ 0`),
the table output is offset `+ 2^32`, so RAF-evaluation demands
`RamAddress − offset − 2^32 = 8k + lowest_address` for some one-hot `k ∈ [0, K)` — with guest
addresses ~2^31 and `8K + lowest` bounded far below the wrapped field value, no `k` exists.
The Shout argument binds `LookupOutput` to the committed table, so the prover cannot lie about
the output. Range `[0, 8)` of the offset is guaranteed by the table definition — **no committed
offset columns, no booleanity, and zero extra cycles for variant pinning**: strict `LD`/`SD`
keep their 1 cycle, and applicability enforcement rides the mandatory lookup slot. (An
alternative representation — three committed offset-bit columns with R1CS booleanity and
degree-2 pin constraints — costs ~+2% commit and extra R1CS inputs; it becomes relevant again
only in Tier 3, which repurposes the load's lookup slot for extraction.)

Two facts make this tier cheaper than it looks. First, the one-hot lookup commitment is paid on
every cycle regardless (`LD`/`SD` today commit index 0 with no table flag — there is no
"no-lookup" discount), so claiming the slot costs nothing new. Second, `AddressOffset` rides the
`AddOperands` path, whose lookup index is the raw non-interleaved sum (the `ADD` pattern) — a
single-operand table over `ea`, the simplest possible MLE shape.

Sequences become: sub-word loads `[VirtualLoadDw v ← RAM[floor(ea)]][Extract(v, ea)]` = **2
cycles**; narrow stores `[VirtualLoadDw][ShiftData][MaskOld][ADD][VirtualStoreDw]` = **5
cycles**. The `Extract`/`ShiftData` lookups need `y = ea = rs2 + imm` with `rs2 = base` — this
is **verified free**: the instruction-input identity is a plain sum
(`RightInstructionInput = IsRs2·Rs2Value + IsImm·Imm`, `spartan/instruction_input.rs:479-490`)
with no exclusivity constraint anywhere, so a virtual instruction that sets both flags gets
`Rs2Value + Imm` as its operand; this is sound because the flags are deterministic bytecode
read-values, not prover-chosen.

### Tier 2 — dual-expansion dispatch (full Zicclsm)

The crossing case (`(ea & 7) + width > 8`) needs a second doubleword access. Static
single-variant sequences would tax every access with the worst case; instead, **compile both
variants into the static bytecode** at preprocessing, at distinct expanded PCs:

```
guest PC X (LW):
  variant A (non-crossing): [VirtualLoadDw][ExtractW]                 (2 rows)
  variant B (crossing):     [VirtualLoadDw lo][VirtualLoadDw hi(+8)]
                            [FunnelLo][FunnelHi][OR][ExtractW]        (~6 rows)
```

**The committed bytecode is the union of both variants — the Shout table stays static.** What
varies per dynamic execution instance is only the PC stream: one iteration's LW enters `p_A`
and reads 2 rows, another iteration's LW (crossing this time) enters `p_B` and reads 6. Per-
cycle reads into a static table are precisely what the bytecode lookup argument supports — the
same reason one trace can take both sides of a guest branch across loop iterations. Dual
expansion is an inlined `if/else` at the virtual layer whose "condition" is not evaluated
in-trace; entry-point choice is genuinely prover-free, and the `AddressOffset` marker is what
makes exactly one choice satisfiable per instance. Rows of the un-taken variant are read with
multiplicity zero, like the existing power-of-two padding rows. The tracer's role is only to
*emit the applicable variant's rows into the trace* per instance (`RISCVTrace::trace` →
`inline_sequence` already re-derives the sequence per dynamic execution; its expansion hook
becomes `cpu`-state-dependent), while bytecode-side expansion at preprocessing emits the union
deterministically:

- **Entry**: both variant heads sit at unexpanded address X with `IsFirstInSequence` (variant A
  of `LD`/`SD` stays the plain native row). The disambiguation chain — R1CS pins
  `NextUnexpandedPC`, the shift sumcheck welds `UnexpandedPC(j+1) = NextUnexpandedPC(j)`, the
  bytecode Stage-1 Val pins `UnexpandedPC(j+1) = bytecode[PC(j+1)].address`, and
  `MustStartSequenceFromBeginning` demands *a* sequence head — leaves the prover free to enter
  either variant. Constraint 17 keeps each variant internally straight-line, and cross-variant
  escape is impossible: variant A's terminal row forces `NextUnexpandedPC = X + 4` while
  variant B's interior rows still carry address X. **No R1CS control-flow changes.**
- **Soundness — variant pinning**: prover freedom is made harmless because each variant's
  `AddressOffset` table (Tier 1) emits the unsatisfiable marker exactly when that variant's
  applicability predicate fails — a wrong-variant execution violates RAF-evaluation rather than
  producing a wrong-but-provable value. This costs zero cycles on either path.
- **Mapper**: `BytecodePCMapper` keys `(address, virtual_sequence_remaining)`
  (`crates/jolt-program/src/preprocess/bytecode.rs:97-131`), which collides across variants:
  `validate_indices` (`bytecode.rs:165-193`) requires vsr to descend by exactly 1 per address
  bucket, and `get_pc` returns the first vsr match. Extend the key with a variant discriminator
  stamped by the expansion (`expand/metadata.rs:48-53` is the single writer of
  vsr/`is_first_in_sequence`) and rework validation. Prover-side preprocessing only; no
  constraint-side change.
- **Tracer**: expansion becomes runtime-data-dependent — `InlineSequenceFn` currently takes no
  `Cpu` (`tracer/src/instruction/inline.rs:30-33`), and `cpu.rs:531` counts
  `inline_sequence().len()` for checkpointing, which must count the chosen variant.
- **Crossing sequences** (all use ordinary single-port RAM cycles, two RMWs in program order —
  coherence is inherited from the single Twist instance):
  - loads: `[LoadDw lo][LoadDw hi][FunnelLo = SRL-by-offset(lo, ea)][FunnelHi = SLL-by-(8−offset)(hi, ea)][OR][Extract/sext]` ≈ 5–6 cycles;
  - stores: RMW both doublewords ≈ 9–11 cycles;
  - `LB`/`LBU`/`SB` never cross — single variant.
- **Bounds**: `ea + 7` may exceed the last mapped doubleword; reserve one guard doubleword above
  `heap_end` in `MemoryLayout` so `floor(ea/8) + 1` always remaps.

Bytecode grows (crossing variants are dead rows for aligned executions): roughly +5 rows per
`LD` site, +10 per `SD` site, ±0 for `LW`/`LH` (shorter aligned variant offsets the crossing
one), −5 per `LB` site — call it ~1.5–2.5× rows, i.e. +1/+2 in `log K`. This is cheap in the
right places and has named watch-items:

- Bytecode **commitment is K-independent**: `BytecodeRa(i)` is one-hot over `k_chunk × T`, so K
  reaches commitments only via `bytecode_d = ceil(log K / log_k_chunk)` — one more dense
  T-length column only when `log K` crosses a chunk boundary (at `log_k_chunk = 4`, d=4 covers
  `log K` 13–16; typical programs sit at 13–14 vs T up to 2^24, so K/T ≈ 1/1000).
- Stage-6a address-phase work goes ~1% → 2–4% of prove; proof size +64–128 B.
- The **full-mode verifier** does an O(K) fold (committed-bytecode/ZK modes skip it) — 2–3×
  bytecode is 2–3× that term.
- Committed-program preprocessing materializes `512 × bytecode_len` densely (268 MB at 2^14 →
  1 GB at 2^16); the akita `SparseUnitPolynomial` path avoids this. `MAX_BYTECODE_D = 6` caps
  `log K ≤ 24` — ample. The read-raf `F_s` scratch (`2·S·K·threads`) is the other memory term
  to watch on wide boxes.

A shared "millicode" crossing handler (JAL into a common block) would cap the growth but breaks
the frozen-`UnexpandedPC` invariant inside sequences (constraint 18's `DoNotUpdate` freeze);
rejected for now.

### Tier 3 (optional) — same-cycle fusion

Route the RAM read value into the lookup operands and the lookup output into the store value.
Neither connection exists today — the operand muxes are pinned by the dedicated
`InstructionInputVirtualization` sumcheck (not the R1CS table), and `LookupOutput`'s only
consumers are the assert/rd-write/jump rows — so this tier means: a new `ValueSource` term in
the instruction-input sumcheck (`instruction_input.rs:66-217`, including the hand-enumerated
4-term output-claim constraint at `:181-208`), a new `InstructionFlags` variant (+1 bytecode
lane), rd taking `LookupOutput` (existing flag) with constraint 3 suppressed for the new class
(guard arithmetic: `Load − ExtractLoad`), and a flag-guarded `wv = LookupOutput + Rs2Value`
variant of constraint 4. The `CircuitFlags::Advice` pattern — which already frees
`RightLookupOperand` from the mux for a claim-reduction to bind — is the precedent to imitate.
Non-crossing loads of any width become **1 cycle**; stores become 2 (`[ShiftData][StoreMerge]`).
Note the conflict: this repurposes the memory cycle's lookup slot for extraction, evicting
Tier 1's `AddressOffset` mechanism — Tier 3 swaps the offset representation to the three
committed bit-columns (with booleanity and degree-2 pin constraints) or folds offset validation
into the extract tables. Also mind the LC cap (below): the new flag joins exclusion guards that
are already at capacity. This is the largest lift and is cleanly separable; Tier 2 does not
depend on it. Its marginal gain over Tier 2 is 1 cycle on sub-word loads and 3 on narrow
stores — on btreemap ~4% more trace reduction.

### Per-op cycle costs

| op | today (traced) | Tier 0 | Tier 1+2 non-crossing | Tier 3 non-crossing | crossing (Tier 2) |
|---|---|---|---|---|---|
| `LD`/`SD` aligned | 1 | 1 | 1 | 1 | — |
| `LD` misaligned | ✗ unprovable | ✗ | — | — | ~5–6 |
| `LB`/`LBU` | 8 | 4 | 2 | 1 | n/a (never crosses) |
| `LH`/`LHU`, `LW`/`LWU` | 9 / 8 | 4–5 | 2 | 1 | ~6 |
| `SB` | 13 | 7 | 5 | 2 | n/a |
| `SH`, `SW` | 14 | 7 | 5 | 2 | ~9–11 |
| `SD` misaligned | ✗ | ✗ | — | — | ~9–11 |

Projected trace deltas (from the measured mixes): `btreemap` −13% (Tier 0), −17% (Tier 2),
−21% (Tier 3); `sha2-chain` −10–15%; `fibonacci` ~0%. (Tier 0 as implemented measures
`btreemap` −8.9% and `sha2-chain` −7.2% — the gap to the −13% projection is the one extra
mask-lookup cycle per access; see "Tier 0 as implemented".) Prover time tracks trace length
nearly 1:1 at these scales *only across padded-instance boundaries*; within a fixed padded
size the measured effect is nil, and the benefit is guest cycles per instance.

### Verifier / proof-size impact

Tiers 0–2: **zero new committed columns**; one new product wire (+1 R1CS input, 35→36) and one
product constraint; ~10–18 new lookup tables (verifier cost is one MLE evaluation per table per
read-raf check — negligible); one extra term in the RAF-evaluation input claim; bytecode-K
growth as above (proof size +64–128 B, full-mode verifier's O(K) fold grows proportionally).
No new sumcheck instances. Tier 3 adds its wiring flags and, if chosen, the three offset-bit
columns (+openings).

### Engineering sync obligations

Verified shape of the blast radius:

- **Three hand-duplicated constraint catalogs move in lockstep**: legacy
  (`zkvm/r1cs/{inputs,constraints}.rs` plus the *positionally index-addressed* evaluators in
  `r1cs/evaluation.rs` with hand-picked accumulator widths), `crates/jolt-claims`
  (`geometry/spartan.rs` — `SPARTAN_OUTER_RV64_ROW_COUNT`, `FIRST_GROUP_ROWS`, and the bare
  `OUTER_UNISKIP_DOMAIN_SIZE = 10` literal), and `crates/jolt-r1cs` (`constraints/rv64.rs`
  `V_*` column indices + all rows re-implemented by integer index). Only tests, not types,
  enforce agreement.
- **BlindFold is mostly generic** (the outer output-claim constraint scans constraint groups at
  runtime); the hand-enumerated exceptions are `spartan/product.rs` factor openings,
  `spartan/shift.rs` shifted-poly vectors, and `spartan/instruction_input.rs`'s output
  constraint — exactly where Tier 3's new operand source lands. RAF-evaluation's
  `input_claim_constraint` (Tier 1) is a params-level change.
- **akita is neutral to R1CS/Spartan changes** (no `cfg(akita)` in those trees; both modes share
  the claims path); it matters only for new committed polynomials and the packed witness
  (fused Inc column).
- **Three hard budgets**: the legacy prover's suffix-accumulation hot path caps
  `MAX_SUFFIXES` (`jolt-prover-legacy/src/zkvm/instruction_lookups/read_raf_checking.rs`,
  a stack-scratch partition bound; raised 4→5 for `PextSigned`, the first 5-suffix table —
  any table with more suffixes must bump it, and only the muldiv e2e catches the miss since
  the assert is debug-only). The committed bytecode width is 447 of 512 lanes
  (`bytecode/chunks.rs` — `3·128 + 2 + 14 + 6 + 40 + 1`); every new lookup table and every new
  circuit flag consumes a lane, and crossing 512 doubles the committed width. This design adds
  ~15–22 tables + 0–2 flags — it fits, but economize the table family (parameterize widths
  where the MLE allows). And `LC` caps at 5 variable terms per side
  (`r1cs/ops.rs:45-60`, panics on overflow); `RightLookupEqRightInputOtherwise`'s guard is
  already at 5, so any new flag joining an exclusion guard forces a restructure.
- New virtual instructions touch the known ring: `jolt-riscv` kind macros, tracer exec impls,
  `zkvm/instruction/` + the lookup-tables **hand-maintained mirror** (append-only enum — the
  discriminant is read by unsafe pointer cast), `jolt-program` expansion, and the z3-verifier's
  silently-panicking wildcard.

The `muldiv` e2e in both modes plus the advice e2e tests are the canaries, per `CLAUDE.md`.

## Alternatives Considered

**A. Widen Twist to 8 contiguous byte ports** (the original pre-Twist/Shout design, updated):
each cycle accesses bytes `raf..raf+7` across 8 byte-addressed ports. Every memory op becomes 1
cycle, but: 8 × d one-hot columns whose nonzeros scale with *bytes touched* (8 per `LD`) — on
LD/SD-dominant traffic this multiplies RAM's commit footprint ~4–8× while the trace shrinks only
22% (and each committed column also carries sparsity-*independent* costs: per-chunk generator
re-affinization O(T), tier-2 Miller loops ~√(k_chunk·T), `combine_hints` scalar-muls); the read
value arrives as 8 separate lane claims needing weighted (256^i) recombination and
sign-extension machinery in R1CS; `Val`/`Inc` go byte-granular (8× val-evaluation terms on
stores); Hamming-booleanity is definitionally single-port (`H(j) = bool`, `H² = H`); and
crossing *still* spans two doublewords unless ports are fully independent byte columns. Byte
granularity also adds +3 to `log K` everywhere K-sized state lives, and the K-dense memory
cliffs bite first: `val_check`'s `RaPolynomialRound3::bind` transiently allocates *eight*
length-K tables (≈256 B × K — already ~2 GB at log K 23; ~17 GB byte-granular), plus per-thread
length-K accumulators in RAF evaluation. The RAM sumchecks being ~0.6% of the prover means the
*only* thing this buys over Tier 3 is fusing the extract lookup — at the cost of the largest
protocol change on the table. Rejected on measured numbers.

**B. One byte per cycle + virtual sequences for wider ops**: `LD` becomes ≥8 byte-reads plus
reassembly (~10–12 cycles). `LD`+`SD` are 16.5% of btreemap cycles natively; the trace grows
~+80% on doubleword-dominant code. Strictly dominated. Rejected.

**C. Packed-lane Twist** (specialize Twist for contiguous access): share one doubleword one-hot
per cycle, commit a lane-mask (or a one-hot over the ≤33 valid `(width, offset)` mask patterns,
Shout-checked against a static pattern table), and read `rv` as the 256-weighted lane sum so the
extraction lands *inside* the memory argument. Elegant, and the natural endgame if byte-heavy
workloads ever dominate: it fuses even Tier 3's extract lookup. But it byte-granularizes
`Val`/`Inc` (per-lane increments on stores), needs new booleanity/Hamming arguments for masks,
still needs a second port or slow path for crossing, and delivers ≤1 cycle/op over Tier 3 —
i.e., it competes with a tier that requires no Twist changes at all. Deferred; measure Tier 3
first.

**D. Dual-port doubleword Twist**: a second, almost-always-idle read/write port would make even
crossing accesses 1–2 cycles. Tier-1 commit cost of a sparse port is ~free (O(nonzeros) batch
additions) and read/write checking's sparse matrices price idle entries at ~zero — but the
port-count assumption runs deeper than it looks: registers are *not* a commitment precedent
(their `ra` polynomials are virtual, discharged against bytecode because register indices are
bytecode-hardcoded — data-dependent RAM addresses can't use that trick), so a second port
duplicates the whole committed family (commitment + Hamming-booleanity, which is definitionally
single-port, + ra-virtualization + claim reductions + Dory openings), un-merges `ra`/`wa`
(`witness.rs:37-40`), and touches the fused akita Inc column, the 64-byte `JoltTraceRow`, and
the hardcoded 35-input R1CS layout. All of that only accelerates the rare crossing case that
Zicclsm explicitly permits to be slow. Keep as a future extension if crossing-heavy workloads
materialize (the read-write-matrix machinery itself is port-agnostic, and its u16 lookup-table
sizing has headroom for exactly 2 ports).

**E. Widened 3-operand lookups** (136-bit index: `(lo, hi, offset)` funnel-shift tables): makes
crossing loads 1 lookup, but adds a 17th instruction-`ra` column dense on *every* cycle
(~+4–6% commit globally) for a rare-case win. Rejected.

**F. Trap-and-emulate in the guest**: Zicclsm-compliant on paper (invisible traps), but Jolt has
no CSR/trap machinery and it couples guest images to a runtime. Rejected.

**G. Branchy virtual sequences** (intra-sequence forward skip instead of dual expansion): a
virtual skip whose taken-target is `PC + 1 + Imm_skip` — `imm` is already a committed per-row
R1CS input, and the target is auto-confined: mid-sequence `DoNotUpdate = 1` freezes
`NextUnexpandedPC`, the shift/bytecode chain pins the landing row's address, so skipping past
the sequence end is caught. Precisely priced: +1 `CircuitFlags` (14→15; cannot reuse `Branch`,
whose `ShouldBranch` semantics both move `NextUnexpandedPC` and disable the freeze), +1 product
constraint (`ShouldSkip = LookupOutput · flag`, `NUM_PRODUCT_VIRTUAL` 3→4), +1 R1CS input
(35→36), constraint 17 splits in two (uniskip degree stays 9 for one added constraint; a second
pushes it to 10), plus the same `cpu.rs:531` accounting fix. Soundness is *stronger* than dual
expansion (the prover has no choice), but it burns a dispatch cycle on every access unless
fused, and `validate_indices` would need an explicit consecutive-index assertion for `PC + Imm`
arithmetic. Fallback if dual-expansion mapping proves uglier than expected.

## Evaluation

### Acceptance Criteria

- [ ] All RV64IM loads/stores execute and prove correctly at every `ea mod 8` × width
      combination, including doubleword-crossing, matching the reference emulator byte-exactly
      (differential test over all 8 offsets × {LB,LBU,LH,LHU,LW,LWU,LD,SB,SH,SW,SD}).
- [ ] A guest compiled for rv64im+zicclsm-style codegen (misaligned `memcpy`,
      `ptr::read_unaligned`, packed structs) traces and proves end-to-end.
- [ ] Wrong-variant executions are unsatisfiable: negative tests that force the tracer to emit
      the non-crossing variant for a crossing access (and vice versa) must fail verification.
- [ ] `muldiv` e2e passes in `--features host` and `--features host,zk`; advice e2e passes
      (standard mode `input_claim` reconstruction); akita acceptance suites pass.
- [ ] `source_to_jolt_expansion_equivalence` (jolt-eval) covers the new expansions and both
      variants; new lookup tables get `mle_random` / `prefix_suffix` / `mle_full_hypercube`
      tests and fuzz targets.
- [ ] Guard-doubleword bounds: crossing access at the top of the heap neither panics nor
      escapes `ram_K`.
- [ ] The RISC-V arch-test `misaligned-ldst-01` (ACT4 submodule) passes, and
      `tests/arch-tests/jolt/sail.json` is flipped to `"misaligned": {"supported": true}`
      (plus `load_address_misaligned` / `samo_address_misaligned`) with Zicclsm declared.
      (The test name at `tests/arch-tests/README.md:113` is a sample-output placeholder, not a
      known-failure registry; per the README policy a failing non-privileged test reds CI.)
- [ ] The three latent MMU trace-recording bugs currently masked by alignment asserts are
      fixed: `trace_load` truncates the address to 4 bytes (`(ea >> 2) << 2`) while recording
      an 8-byte value (`tracer/src/emulator/mmu.rs:517-543`); `trace_store_byte` merges an
      8-byte `pre_value` with 32-bit masks under a `% 4` dispatch, zeroing the top half
      (`mmu.rs:548-585`); `trace_store_halfword` has the same truncation and 32-bit-mask
      defect (`mmu.rs:590-629`). Relatedly, `trace_store` (`mmu.rs:632-663`, used by
      `store_word`) records an 8-byte write for a 4-byte store — audit it in the same pass.

### Testing Strategy

Extend `examples/memory-ops` with misaligned asm cases (all offsets, all widths, crossing and
non-crossing, loads and RMW stores, including at region boundaries: input/output region,
heap end). Property test: random (addr, width, value) sequences vs a byte-array oracle.
Both-mode (`host`, `host,zk`) CI lanes as per `CLAUDE.md`; akita lane for the packed mode.

### Performance

- Objective: prover wall-time on `btreemap` at 2^24 improves ≥10% by Tier 2 (projected ~17%
  trace reduction, ~+2% commit overhead); no regression on `fibonacci` (memory-free) beyond
  +2% commit; `sha2-chain` improves ≥8%.
- Crossing-path microbenchmark: a crossing-heavy loop must degrade ≤10× vs aligned (Zicclsm
  "slow but correct" budget), and its existence must cost aligned programs nothing beyond the
  bytecode-size effect.
- Track via jolt-eval performance objectives (add a `bytes-workload` objective if byte-heavy
  guests become Ethproofs-relevant).

## Implementation notes (verified traps)

- **Three copies of the address remapper** must change together: `remap_address`
  (`crates/jolt-prover-legacy/src/zkvm/ram/mod.rs:120`), `remap_word_address`
  (`common/src/jolt_device.rs:444-463` — different name; a grep for `remap_address` misses
  it), and `remap_address` (`crates/jolt-prover/src/config.rs:129-137`) — all truncate by 8
  silently, which is exactly right for floor addressing but means nothing rejects
  misalignment there. They diverge on the below-`lowest_address` case (panic vs
  `Err(AddressBelowLowest)` vs `assert!`), which matters for the Tier 2 guard-doubleword
  bounds.
- **Recipe temp budget is 8** (`expand/grammar.rs:208-218`, hard error) and max sequence length
  is 64 (`materialize.rs:19`). Current narrow stores use 4 temps; crossing-store variants need
  ~5–6 — fits, but audit. Loads are `HAS_SIDE_EFFECTS = true`, so `rd = x0` loads consume a
  temp via the rewrite path.
- **`is_compressed` is stamped on the LAST row only** of an expansion (`expand/metadata.rs:52`)
  — each dual-expansion variant needs its own correctly-stamped terminal row, and
  `IsCompressed ∧ DoNotUpdateUnexpandedPC` never both set (debug-asserted in bytecode
  read-raf).
- **Uniskip degree budget**: `OUTER_UNIVARIATE_SKIP_DEGREE = (NUM_R1CS_CONSTRAINTS − 1)/2`
  auto-resizes on a count change, but *which* constraints sit in the first group is a manual
  editorial call (`FIRST_GROUP_ROWS` in jolt-claims), and one added eq-constraint keeps the
  first-group degree at 9 while a second pushes it to 10. Tier 1+2 as specified adds one
  product constraint and zero eq constraints; Tier 3 and the b-column representation add more —
  track the count.
- **Stale comment trap**: `zkvm/witness.rs:31` claims instruction lookups use d=8; the real
  config is `d = 16` at the 8-bit chunk (`LOG_K = 128`). Don't size anything off that comment.
- **Existing guest-side workarounds** become removable perf wins: `jolt-sdk/src/lib.rs:493-494`
  steps u8→u16→u32 to reach 8-byte alignment; `jolt-inlines/keccak256` carries
  `absorb_unaligned`.
- `expand_byte_load` (`memory/shared.rs:35-83`) already assumes nothing about alignment — it is
  the shape the Tier-1 lowering generalizes.
- The `TrapType::*AddressMisaligned` variants exist but are never constructed; this design
  keeps it that way (no trap machinery).

## Open Questions

1. **Mapper keying**: cleanest variant discriminator for `BytecodePCMapper` — extend
   `virtual_sequence_remaining` numbering, add a variant field to row metadata, or carry the
   expanded PC through the `Cycle`? Audit `get_pc` call sites (incl. debugging/backtrace
   tooling) for one-expansion-per-PC assumptions, and design the `validate_indices`
   replacement.
2. ~~Instruction-input flags~~ **Resolved**: the operand mux identity is a plain sum with no
   exclusivity constraint anywhere; dual-set flags yield `Rs2Value + Imm` for free and soundly
   (flags are bytecode read-values). Add an explicit regression test pinning this behavior.
3. ~~Extract-table decomposition~~ **Resolved** (see "Tier 0 as implemented"): the low-bits-of-y
   dependence itself is harmless (offset bits never straddle a phase boundary), but the fused
   per-offset form has intrinsic prefix rank ≥ 8 per width — a bilinear decomposition restricted
   to any cut with ≥ 32 bound x-bits is a rank decomposition of a matrix whose factor families
   (the 8 lane functionals, the 8 offset indicators) are each linearly independent. The
   implemented mask-form two-step costs one extra cycle and ~6× less machinery; its
   decompositions are total-domain (no `random_lookup_index` restriction). The fused form
   remains an option for Tier 1's 2-cycle loads, ideally after the legacy registry mirror is
   gone. The `AddressOffset` marker-table question (offset + width gate + marker term) remains
   open for Tier 1.
4. **Marker constant**: `2^32` assumes guest addresses < 2^32 − 8K − lowest; pick the marker
   from `MemoryLayout` bounds (or use a field-huge constant) and prove the no-solution range
   argument once, centrally.
5. **Bytecode growth**: measure real `log(bytecode)` movement on Ethproofs-class guests; if a
   chunk boundary is crossed, consider variant-block placement or bytecode chunk-size tuning
   (the F_s scratch and committed-program preprocessing memory are the terms to watch).
6. **Compressed loads/stores**: C-extension forms share expansions (verified byte-identical
   re-encoding) — confirm the dual-expansion metadata survives the 2-byte-PC mapper path
   (`ALIGNMENT_FACTOR_BYTECODE = 2`).
7. **Tier 3 store fusion**: `wv = LookupOutput + Rs2Value` mixes a lookup output into the wv
   wire — confirm no small-value/range assumptions on `wv` elsewhere (witness gen, val-eval
   `Inc = wv − rv` stays 64-bit-bounded ± the merge algebra), and pick the Tier-3 offset
   representation (b-columns vs validation folded into extract tables).

## Documentation

Update `book/src/how/architecture/ram.md` (address remapping section: the offset term and
dual expansion; also fix the stale `registers.md` K=64 → 128), `emulation.md` (inline
sequences: variants), and the lookup-table listing.
