# Spec: Machine-mode trap and CSR semantics

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-11-07                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt runs guests on an M-mode-only RV64IMAC hart with no F, D or V and no interrupt sources.
The inline sequences for `ECALL`, `MRET`, `CSRRW` and `CSRRS` keep the machine-mode CSRs in reserved virtual registers, but they store every CSR write raw and do not update `mstatus` the way the privileged specification requires for that configuration.
A guest can observe the difference: `mstatus` reads 0 at reset instead of `0x1800`, a trap forgets `MIE`, and an `mtvec` write with a reserved MODE makes the next `ECALL` jump into the middle of an instruction.
This spec changes the four sequences so that they compute exactly what the specification prescribes for this hart. The result also matches the Sail RISC-V model configured the same way, except for one WLRL policy noted below (`mcause`).
The change adds rows for CSR legalization and trap return, saves one row per trap entry, uses only instruction kinds that Jolt already proves, and changes no constraint.
It also records why the LR/SC sequences and `EBREAK` stay as they are, and why tracer panics are acceptable as halting failures.

The motivation is formal verification: a proof that Jolt's expansions refine the Sail model needs the two to agree on every reachable state, and these were the system-level places where they disagreed.

All specification references are to [riscv-isa-manual `717aeca3`](https://github.com/riscv/riscv-isa-manual/tree/717aeca370712ae3f3d7b24acba27cda0b94d109) and [riscv-debug-spec `40e5c354`](https://github.com/riscv/riscv-debug-spec/tree/40e5c3546638279f6b7ca189b0bdcc0dcfe1c8b3), cited as `file:line` with the `norm:` tag.
`machine.adoc` means `modules/priv/pages/machine.adoc`.

## Intent

### Goal

For every program, every source instruction among `ECALL`, `MRET` and the Zicsr instructions `CSRRW` and `CSRRS` (on `mstatus`, `mtvec`, `mepc`, `mcause`, `mtval`, `mscratch`) leaves the architectural state the RISC-V privileged specification prescribes for an M-only hart with IALIGN = 16, with the WARL choices listed below.

### The hart configuration and what it implies

Jolt's hart implements M-mode only, no F/D/V, no interrupt sources, little-endian only.
For that configuration the specification fixes almost all of `mstatus`:

| Field | Value | Reason |
|---|---|---|
| MPP (12:11) | read-only `0b11` | `xPP` "can hold only privilege mode _x_ and any implemented privilege mode lower than _x_" (`machine.adoc:430`, `norm:mstatus_xpp_warl`). Only M is implemented. |
| SPP, SIE, SPIE | read-only 0 | no S-mode (`machine.adoc:374`, `:431` `norm:mstatus_xpp_rdonly0`) |
| SXL, UXL | read-only 0 | no S-mode, no U-mode (`machine.adoc:532` `norm:mstatus_sxl_acc_mxlen64`, `:537` `norm:mstatus_uxl_acc_mxlen64`) |
| MPRV, MXR, SUM, TW, TSR, TVM | read-only 0 | no U/S-mode (`machine.adoc:593`, `:603`, `:625`, `:733`, `:758`, `:776`) |
| MBE, SBE, UBE | read-only 0 | little-endian only (`machine.adoc:680`) |
| FS, VS, XS, SD | read-only 0 | no F, no S-mode, no V (`machine.adoc:843` `norm:mstatus_fs_acc2`, `:857`–`:865`, `:882` `norm:mstatus_sd_rdonly0`) |
| MIE (3), MPIE (7) | writable | |

So `mstatus` reads as `0x1800 | (MIE | MPIE bits)`, and reset gives `0x1800`: MIE resets to 0 (`machine.adoc:2773`, `norm:mstatus_mie_mprv_rst`), MPP can only hold M, and MPIE, which the specification leaves unspecified at reset, is chosen to be 0 (so v39 starts at 0 like every register).

The trap and return rules (`machine.adoc:391`, `norm:mstatus_xpie_xie_xpp_trap_op`; `machine.adoc:406`, `norm:mstatus_xret_op`) then reduce to:

- trap into M (here only `ECALL`): `MPIE := MIE`, `MIE := 0`, `MPP := M`; that is `mstatus := 0x1800 | ((mstatus & 0x8) << 4)`;
- `MRET`: `MIE := MPIE`, `MPIE := 1`, `MPP := M` (the least-privileged mode, which is M); that is `mstatus := 0x1880 | ((mstatus >> 4) & 0x8)`.

The other CSRs:

| CSR | Rule | Choice |
|---|---|---|
| `mtvec` | MXLEN-bit WARL (`machine.adoc:1162`, `norm:mtvec_sz_warl_acc`); MODE ≥ 2 is reserved (`machine.adoc:1197`); every synchronous exception goes to BASE in both Direct and Vectored mode (`machine.adoc:1201`–`:1206`) | A write whose MODE is reserved (bit 1 set) keeps the previous MODE and writes BASE. This is a deterministic function of the written value and the old state (`priv-csrs.adoc:669`, `norm:Zicsr_warl_deterministic`), and it is Sail's `legalize_tvec` under `Xtvec_Ignore`. |
| `mepc` | `mepc[0]` is always zero (`machine.adoc:1686`, `norm:mepc_align`); WARL (`machine.adoc:1694`, `norm:mepc_warl`) | Writes clear bit 0. Jolt supports compressed instructions, so IALIGN = 16 and bit 1 stays writable. |
| `mcause`, `mtval`, `mscratch` | `mcause` Exception Code WLRL (`machine.adoc:1732`, `norm:mcause_exccode_wlrl`; `priv-csrs.adoc:653`, `norm:Zicsr_wlrl_deterministic`), `mtval` WARL holding every value Jolt can produce, `mscratch` plain | Stored raw, as before. |

### Invariants

The first three invariants hold at every source-instruction boundary, that is, before and after each complete expansion. The optimized set sequences mask their operands before merging them, so they do not temporarily store an illegal CSR value.

- **Abstraction relation.** The `mstatus` virtual register (v39) holds `mstatus & 0x88`, so `v39 & !0x88 == 0`. The architectural value is `v39 | 0x1800`.
- **`mtvec` MODE is never reserved.** Bit 1 of the `mtvec` virtual register (v34) is 0. Hence `JALR` through v34, which clears bit 0, lands on BASE.
- **`mepc` is even.** Bit 0 of the `mepc` virtual register (v36) is 0. Hence `JALR` through v36 lands on `mepc` exactly.
- The changed sequences use only existing instruction kinds and only the eight instruction temporaries v40–v47 (at most two live at once), so bytecode, R1CS, lookup tables and the virtual-register layout are unchanged.

### Non-Goals

- Turning tracer panics into architectural exceptions (misaligned `LR`/`SC`, `LR`/`SC` below `RAM_START`). See "Tracer panics are halting failures".
- Interrupts, U/S-mode, or any CSR not listed above.
- `CSRRC`, `CSRRWI`, `CSRRSI` and `CSRRCI`, which both decoders still reject.

## Design

### Final sequences

The abstraction relation is unchanged: v39 contains only MIE/MPIE, v34 is architectural `mtvec` with bit 1 clear, and v36 is architectural `mepc` with bit 0 clear. `t`, `d` and `p` denote instruction temporaries. All sequences use at most two live temporaries in v40–v47. `SLLI` and `SRLI` lower to one `VirtualMULI` or `VirtualSRLI` row each. `VirtualShiftRightBitmask rd, rs1` is one row computing `!0 << (rs1 & 63)`.

**`ECALL`** (`ecall.rs`), 6 rows:

```
AUIPC v36, 0             # mepc := source pc
ADDI  v37, x0, 11        # mcause := 11
ADDI  v38, x0, 0         # mtval := 0
ANDI  t, v39, 8
SLLI  v39, t, 4          # MPIE := MIE, MIE := 0
JALR  t, v34, 0          # pc := mtvec.BASE
```

Materialization stamps every inline row with the source address. `AUIPC` uses that address, not a row index or the advanced execution PC, and can write v36 directly. `ECALL` needs one temporary.

**`MRET`** (`mret.rs`), 3 rows:

```
SRLI v39, v39, 4         # MIE := MPIE, old MIE discarded
ORI  v39, v39, 0x80      # MPIE := 1
JALR t, v36, 0           # pc := mepc
```

**CSR read and register swap** (`csr.rs`): `read(rd)` is `ORI rd, v39, 0x1800` for `mstatus` and `ADDI rd, vr, 0` otherwise. `CSRRW` emits no read for `rd = x0`. For `mstatus`, `mepc` and raw CSRs, it reads before writing; if `rd = rs1 ≠ x0`, it first preserves the source with `ADDI t, rs1, 0`.

The register write is:

```
mstatus: ANDI v39, value, 0x88
mepc:    ANDI v36, value, -2
raw:     ADDI vr, value, 0
mtvec with value = x0: ADDI v34, x0, 0
mtvec otherwise:
  XOR                      d, value, v34
  ANDI                     p, value, 2
  VirtualShiftRightBitmask  p, p
  AND                      d, d, p
  read(rd) if rd != x0
  XOR                      v34, v34, d
```

The nonzero-source `mtvec` sequence includes its optional read at the position shown. With `x = value`, its mask `M` is `!0` for MODE 0 or 1 and `!3` for MODE 2 or 3. The result `old ^ ((x ^ old) & M)` is therefore `x` for a legal MODE, or `(x & !3) | (old & 1)` for a reserved MODE. The source is fully consumed before the read can overwrite it, so even an aliased swap needs only the two temporaries and no copy. The writing algorithm costs five rows for every nonzero register source.

**`CSRRS`** (`csr.rs`): when `rs1 = x0`, emit only `read(rd)`, or the canonical `ADDI x0, x0, 0` when `rd = x0`. Otherwise prepare the effective set operand before reading:

```
mstatus: ANDI t, rs1, 0x88
mepc:    ANDI t, rs1, -2
mtvec:
  ANDI                     t, rs1, 2
  VirtualShiftRightBitmask  t, t
  AND                      t, rs1, t
raw:     t denotes rs1 (no row), except an alias needs ADDI t, rs1, 0
then:
  read(rd) if rd != x0
  OR vr, vr, t
```

Setting `mtvec` with a reserved operand MODE contributes only its BASE bits. The old MODE already present in v34 remains unchanged. When operand bit 1 is clear, all its bits contribute. This avoids constructing and then legalizing `old | rs1`. Preparing the operand also consumes an aliased `rs1` before the read overwrites it, so `mstatus`, `mepc` and `mtvec` need no alias copy.

For both `mtvec` algorithms, legal operand MODEs 0 and 1 use `M = !0`; reserved MODEs 2 and 3 use `M = !3`, independently of old MODE 0 or 1 and BASE. An all-ones operand therefore writes `!3 | old_MODE` for `CSRRW` and sets every BASE bit while retaining old MODE for `CSRRS`. With old `mtvec = 5`, operands 3 and 10 produce respectively 1 and 9 for `CSRRW`, and 5 and 13 for `CSRRS`. Neither result depends on whether `rd = x0`, is distinct, or aliases `rs1`; a nonzero `rd` receives 5.

### Cost per expansion

Every entry is **main (`9407f93fb`) → final**, in executed rows per source instruction. Raw means each of `mcause`, `mtval` and `mscratch`. Register cases are disjoint; `rd = rs1` below means a nonzero alias.

| Instruction | Operand case | mstatus | mepc | mtvec | raw |
|---|---|---|---|---|---|
| `ECALL` | all | 7 → 6 | | | |
| `MRET` | all | 1 → 3 | | | |
| `CSRRW` | `rd = x0, rs1 = x0` | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 |
| `CSRRW` | `rd = x0, rs1 ≠ x0` | 1 → 1 | 1 → 1 | 1 → 5 | 1 → 1 |
| `CSRRW` | `rd ≠ x0, rs1 = x0` | 2 → 2 | 2 → 2 | 2 → 2 | 2 → 2 |
| `CSRRW` | `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 → 2 | 2 → 2 | 2 → 6 | 2 → 2 |
| `CSRRW` | `rd = rs1 ≠ x0` | 3 → 3 | 3 → 3 | 3 → 6 | 3 → 3 |
| `CSRRS` | `rd = x0, rs1 = x0` | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 |
| `CSRRS` | `rd = x0, rs1 ≠ x0` | 1 → 2 | 1 → 2 | 1 → 4 | 1 → 1 |
| `CSRRS` | `rd ≠ x0, rs1 = x0` | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 |
| `CSRRS` | `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 → 3 | 2 → 3 | 2 → 5 | 2 → 2 |
| `CSRRS` | `rd = rs1 ≠ x0` | 3 → 3 | 3 → 3 | 3 → 5 | 3 → 3 |

Relative to main, a register `CSRRW` to `mtvec` with a nonzero source costs four extra rows, or three with an alias; a writing `CSRRS mtvec` costs three extra rows, or two with an alias. `CSRRS mstatus` and `CSRRS mepc` cost one extra row except aliases, whose cost equals main. `ECALL` saves one row and `MRET` costs two extra rows. The boot `mtvec` register write therefore costs four extra rows.

### Instruction-count rationale

The tables enumerate each path with **main** and **final** counts. Each raw row applies separately to `mcause`, `mtval` and `mscratch`. Counts are pinned as literals by `system_expansion_counts_match_spec_paths`; the tracer test `system_expansions_refine_architectural_formulas` independently checks the architectural formulas and invariants.

A row has one register output, except a control-flow row also sets the next PC. Distinct required register outputs give a lower bound. The canonical no-op is the minimum one-row representation of an advancing source instruction. The lower bounds below fix the current representation: v39 stores only bits 3 and 7, v34 is architectural `mtvec`, and v36 is canonical architectural `mepc`. Other registers and memory are arbitrary unless an earlier row supplies a required output.

**Aliased two-output paths need at least three rows.** With only two rows, each must write one final output, `rd` or the CSR. Reading into `rd` first destroys the source: different sources with the same old CSR then become indistinguishable although they require different updates. Writing the CSR first destroys the old value needed by `rd`. For a swap, fix a legal source and vary the old CSR: the final CSR is the same. For a set, choose a source with every writable bit set (legal MODE 1 for `mtvec`); every old CSR maps to the same final CSR. The old architectural read cannot be recovered in any of these cases. This information-loss bound applies to the three-row aliased swaps and raw sets in the tables, including the masked sets whose operand preparation preserves the needed source information.

**Masked register sets on `mstatus` and `mepc` need two writing rows.** At old value 0 the result must be `rs1 & mask`, but the same row must also merge arbitrary old writable bits. OR admits forbidden operand bits; AND and ANDN compute intersections or differences instead of the required union. Arithmetic introduces carries, and extract/shift kinds cannot merge old bits with the masked operand. Comparisons and mask generators have the wrong output range; memory, advice and control-flow rows cannot supply this register-only function. Thus no single kind implements the masked merge in this representation. A distinct read adds one row; an alias also has the three-row information-loss bound above.

**The `mtvec` register algorithms are best found; lower bound open.** The five-row write and the three-row `CSRRS` operand transform use the dynamic mask to keep either the whole operand or only its BASE. Bounded searches found nothing shorter:

- *Low-six-bit SMT search, exact for its grammar.* `AND OR XOR ANDN ADD SUB ANDI ORI XORI ADDI MUL VirtualMULI SLLI VirtualShiftRightBitmask[I] VirtualPow2[I]`, with every six-bit immediate and every shift count. The low six bits of these operations depend only on the low six input bits, so unsatisfiability on low-bit samples excludes the full-width sequences. Operands are the source, the old `mtvec`, `x0` and every earlier result. Result: no write in four rows, no operand transform in two rows, and no fused set in three rows.
- *Exact 64-bit enumeration of two-row operand transforms.* This also covers high multiplies, word forms, `VirtualPext`, right shifts, rotates and XOR-rotates, conditional negation, sign moves, extensions, byte reversal, window masks, lane shifters and constants. The immediates are 0–31, −1 to −4, 62, 63, every power of two and every left-shifted all-ones mask. No candidate survived.

These leave open four-row writes and three-row fused sets that use right shifts, extraction, rotates, word forms or lane shifters (a full-width search over that set timed out), and memory, advice or PC rows. A completed search over those kinds would settle the bound.

#### ECALL

| Path | main | final | Rationale |
|---|---|---|---|
| all states | 7 | 6 | mepc, mcause and mtval each need a write; status needs two rows by the exclusion below; next PC needs JALR. AUIPC writes mepc directly. |

#### MRET

| Path | main | final | Rationale |
|---|---|---|---|
| all states | 1 | 3 | Status needs two rows by the exclusion below; JALR writes next PC. |

For the fixed v39 representation, the status functions on inputs `0, 8, 0x80, 0x88` are respectively `0, 0x80, 0, 0x80` for `ECALL` and `0x80, 0x80, 0x88, 0x88` for `MRET`. Neither is a single row of any kind:

| Kind family | Why one status row fails |
|---|---|
| Bitwise/register and immediate kinds, including word extensions | They cannot move bit 3 to bit 7 or bit 7 to bit 3. Constants alone cannot supply the varying output. |
| ADD/ADDI/SUB, including word forms | At zero the required constant is fixed; adding/subtracting the input (or doubling/cancelling it) then fails the other three inputs. |
| Shifts, rotates, byte reversal and lane shifts | A left shift by 4 moves MIE correctly for ECALL but retains old MPIE at bit 11. A right shift by 4 moves MPIE for MRET but cannot insert bit 7 at zero. Rotations/permutations retain the unwanted bit; lane shifts move by multiples of 8. |
| `VirtualMULI`, including word form | ECALL at 8 requires `8c = 128`, which makes `128c = 2048`, not 0, modulo the word width. MRET fails at zero. |
| `MUL` / `MULHU`, including word forms | Repeating the status operand gives 64 / 0 at input 8, not ECALL's 128. A fixed low-product multiplier has the preceding contradiction; a high product at 8 is at most 7. Both fail MRET at zero. An arbitrary second register cannot make the result independent of that register. |
| `VirtualPext` / `VirtualPextSigned` | Extraction packs bits downward, so it cannot move ECALL's bit 3 up to bit 7. It cannot insert MRET's bit 7 at zero. |
| XOR-rotates | With the other input zero they rotate both status bits instead of discarding one. With both inputs the status register, the XOR-then-rotate forms give zero and `VirtualXORROTL1` (`x ^ rol(y, 1)`) gives 24 at input 8, neither the required 0x80. Another register input makes the result depend on unconstrained state. |
| Pow2 and mask kinds | Pow2 always returns a single bit, unlike MRET at `0x80`, and never ECALL's zero. Shift-right masks have a nonzero suffix shape; byte/halfword/word masks have whole-byte windows, neither required status shape. Immediate-only forms are constant. |
| Comparisons | Their 0/1 range excludes the required `0x80` and `0x88`. |
| `VirtualMovsign` / `VirtualNegateIf` | All four inputs have sign zero, so sign extraction is zero and conditional negation cannot select or move the required bits. |
| Loads, stores, assertions, advice and host I/O | Loads/advice depend on unconstrained memory or supplied data; stores/assertions do not produce the required register value. These cannot implement a deterministic status function for arbitrary state. |
| PC constants, jumps, branches and fences | They do not compute the status function; a jump's register result is a PC link. |

This exclusion supplies the two-status-row lower bound for both instructions in the current representation. For ECALL, mepc, mcause and mtval need their own writes. `JALR` writes the link `pc + 4`, which equals none of the required `mepc = pc`, `mcause = 11`, or `mtval = 0` for arbitrary source PC, so its register output cannot save any of those writes. Together with the next-PC row this gives six rows; MRET needs its two status rows and one next-PC row, giving three.

#### CSRRW

| CSR / operand path | main | final | Rationale |
|---|---|---|---|
| mstatus: `rd = x0, rs1 = x0` | 1 | 1 | One legalized CSR write. |
| mstatus: `rd = x0, rs1 ≠ x0` | 1 | 1 | One legalized CSR write. |
| mstatus: `rd ≠ x0, rs1 = x0` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| mstatus: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| mstatus: `rd = rs1 ≠ x0` | 3 | 3 | The three-row alias bound above is met; legalization is folded into the write. |
| mepc: `rd = x0, rs1 = x0` | 1 | 1 | One legalized CSR write. |
| mepc: `rd = x0, rs1 ≠ x0` | 1 | 1 | One legalized CSR write. |
| mepc: `rd ≠ x0, rs1 = x0` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| mepc: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| mepc: `rd = rs1 ≠ x0` | 3 | 3 | The three-row alias bound above is met; legalization is folded into the write. |
| mtvec: `rd = x0, rs1 = x0` | 1 | 1 | One legalized CSR write. |
| mtvec: `rd = x0, rs1 ≠ x0` | 1 | 5 | Five rows mask the source/old difference and apply it. Best found; lower bound open as described above. |
| mtvec: `rd ≠ x0, rs1 = x0` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| mtvec: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 6 | Five writing rows and one read. Best found; lower bound open as described above. |
| mtvec: `rd = rs1 ≠ x0` | 3 | 6 | Five writing rows consume the source before the read, avoiding an alias copy. Best found; lower bound open as described above. |
| raw: `rd = x0, rs1 = x0` | 1 | 1 | One legalized CSR write. |
| raw: `rd = x0, rs1 ≠ x0` | 1 | 1 | One legalized CSR write. |
| raw: `rd ≠ x0, rs1 = x0` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| raw: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 2 | One legalized CSR write and one distinct rd output. |
| raw: `rd = rs1 ≠ x0` | 3 | 3 | The three-row alias bound above is met; legalization is folded into the write. |

#### CSRRS

| CSR / operand path | main | final | Rationale |
|---|---|---|---|
| mstatus: `rd = x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| mstatus: `rd = x0, rs1 ≠ x0` | 1 | 2 | One operand mask and one OR merge meet the two-writing-row bound above. |
| mstatus: `rd ≠ x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| mstatus: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 3 | Two writing rows by the masked-merge bound above, plus one read; operand preparation also satisfies the alias bound. |
| mstatus: `rd = rs1 ≠ x0` | 3 | 3 | Two writing rows by the masked-merge bound above, plus one read; operand preparation also satisfies the alias bound. |
| mepc: `rd = x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| mepc: `rd = x0, rs1 ≠ x0` | 1 | 2 | One operand mask and one OR merge meet the two-writing-row bound above. |
| mepc: `rd ≠ x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| mepc: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 3 | Two writing rows by the masked-merge bound above, plus one read; operand preparation also satisfies the alias bound. |
| mepc: `rd = rs1 ≠ x0` | 3 | 3 | Two writing rows by the masked-merge bound above, plus one read; operand preparation also satisfies the alias bound. |
| mtvec: `rd = x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| mtvec: `rd = x0, rs1 ≠ x0` | 1 | 4 | Three rows transform the operand, then one OR merges with old state. Best found; lower bound open as described above. |
| mtvec: `rd ≠ x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| mtvec: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 5 | Three operand-transform rows, one OR merge and one read. Best found; lower bound open as described above. |
| mtvec: `rd = rs1 ≠ x0` | 3 | 5 | Three operand-transform rows consume the source before the read; one OR merge completes the update. Best found; lower bound open as described above. |
| raw: `rd = x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| raw: `rd = x0, rs1 ≠ x0` | 1 | 1 | One OR update. |
| raw: `rd ≠ x0, rs1 = x0` | 1 | 1 | One read, or one canonical no-op when both registers are x0. |
| raw: `rd ≠ x0, rs1 ≠ x0, rd ≠ rs1` | 2 | 2 | One OR update; one distinct read. |
| raw: `rd = rs1 ≠ x0` | 3 | 3 | One OR update, one read and one preserved input meet the three-row alias bound above. |

Where the sequences save rows, and why:

- `ECALL` needs no PC copy, one row less than on `main`, because `AUIPC` can target v36 directly.
- A nonzero register `mtvec` swap masks the full source/old difference in five writing rows. This consumes the source before the read, so the alias case needs no copy. A zero source needs no masking at all: one write.
- Register `CSRRS mtvec` transforms only the operand with a dynamic mask, without constructing `old | bits` or comparing it with old MODE. This also consumes the source, so its alias case needs no copy. Masking the operand before the read removes the source copy for aliased `CSRRS mstatus` and `CSRRS mepc` too, so their alias cost equals `main`'s.
- For the remaining three-row aliased swaps and raw sets, reordering alone cannot remove preservation: the information-loss argument above excludes either two-row ordering.

### Effect on proving

- Only bytecode changes. Every row uses an instruction kind Jolt already proves (`ORI`, `ANDI`, `AND`, `XOR`, `OR`, `ADDI`, `AUIPC`, `JALR`, `VirtualMULI`, `VirtualSRLI`, `VirtualShiftRightBitmask`), so R1CS, lookup tables, RAM and register checking are unchanged.
- The virtual registers v32–v39 keep their roles; sequences use at most two of the eight instruction temporaries.
- Trace length changes by the per-instance differences above. A trace that crosses a power-of-two padding boundary as a result grows the proof like any longer trace; the proof format does not change.
- The golden expansion fixture (`crates/jolt-program/src/expand/fixtures/main_expand_parity_hashes.json`) differs from main in exactly 24 of its 360 cases: `CSRRW` and `CSRRS` on `mstatus` (768), `mtvec` (773) and `mepc` (833), and `ECALL` and `MRET`, each with suffixes `_compressed_false`, `_compressed_true` and `_rd0`. Only output hashes change; inputs, metadata, raw CSR cases and every other key keep main's values. Regenerate it only with `JOLT_UPDATE_EXPANSION_FIXTURES=1`.
- The full end-to-end proving matrix (`jolt-prover` `e2e_matrix`, clear mode: fibonacci, muldiv, memory-ops, memory-copy, sha2, sha3, sha3-aligned, btreemap, stdlib, advice-consumer) proves and verifies on this branch.

### Alternatives considered

- **Keep the fields inert**, the previous design: no interrupt sources exist, so MIE and MPIE never change control flow, and ZeroOS restores `mstatus` itself before `mret`. But a guest observes the fields through `csrr`, a reserved `mtvec` MODE crashes the tracer, and a refinement proof against the specification has to carry every disagreement as an exception. The correct sequences cost a few rows, so the exceptions are not worth keeping.

- **Store the full architectural `mstatus` in v39** (`0x1800` included). Every write then needs `ANDI` and `ORI` (one more row each), and the reset value needs either an initialization row or a special case. Storing only the writable bits makes reads one `ORI`, writes one `ANDI`, and reset correct for free.
- **Map reserved `mtvec` MODEs to Direct.** The write would keep only the source masked by the dynamic operand mask (`ANDI p, x, 2; VirtualShiftRightBitmask p, p; AND v34, x, p`), consuming the source before the read: register-swap totals 3 / 4 / 4 instead of 5 / 6 / 6 (`rd = x0` / distinct / alias). A zero register source still costs 1 / 2 rows. This disagrees with the required `Xtvec_Ignore` behavior and is not implemented.
- **Clear only `mtvec[1]` on every write.** Another legal WARL choice would map MODE 2 to 0 and MODE 3 to 1. A register write would need one `ANDI`, so swap totals would be 1 / 2 / 3, and register set totals 2 / 3 / 3 (set-only / distinct / alias). Constant swaps would always need one write row. This also violates the fixed MODE-preservation contract and is not implemented.
- **Clear `mepc[0]` on the trap path only.** The specification makes architectural `mepc[0]` always zero, so a read after an odd write must return the even value. Clearing on each write keeps v36 canonical; otherwise every read and state consumer must normalize it, as in the rejected representation below.
- **Leave bit 0 free in a non-canonical v36.** With architectural `mepc = v36 & !1`, reads become `ANDI rd, v36, -2` at the same one-row cost. A non-aliased `CSRRS mepc` with a nonzero source could use OR directly: 1 / 2 rows (`rd = x0` / distinct), instead of 2 / 3. No other path changes. This is rejected because the abstraction becomes many-to-one: v36 no longer equals architectural `mepc`. Every state consumer would need to normalize it, including the Sail differential harness, which compares v36 with Sail's `mepc`, and the formal model's refinement relation. The saving applies only to `csrs mepc, rs` with an odd-capable register, which no compiler or library emits.

## Divergences on `main`, with guest programs

Each test below is in `tracer/src/instruction/system_semantics_tests.rs`. It stores an RV64 program at `DRAM_BASE` and runs it through `Cpu::tick` in trace mode, so every source instruction executes as its expanded rows: what the prover sees. Expected values are the specification's for this hart.

| Test | Program (abridged) | `main` | Spec, this branch |
|---|---|---|---|
| `mstatus_reads_mpp_machine_at_reset` | `csrr a0, mstatus` | `0` | `0x1800` |
| `ecall_stacks_mie_into_mpie_and_mret_restores_it` | `csrw mstatus, 8; ecall` → handler `csrr mstatus; mepc += 4; mret` → `csrr mstatus` | handler `0x1800`, after `0x1800` | handler `0x1880`, after `0x1888` |
| `ecall_with_interrupts_disabled_clears_mpie` | same with `MIE = 0`, `MPIE = 1` | after `0x1800` | handler `0x1800`, after `0x1880` |
| `mret_twice_sets_mpie_then_moves_it_into_mie` | `mret` twice, reading `mstatus` after each | `0`, `0` | `0x1880`, `0x1888` |
| `mstatus_write_keeps_only_mie_and_mpie` | `csrw mstatus, -1`, `0`, `8` | `0xffff_ffff_ffff_ffff`, `0`, `8` | `0x1888`, `0x1800`, `0x1808` |
| `mstatus_set_bits_and_swap_with_rd_equal_rs1` | `csrrs a0, mstatus, a0` with `a0 = 0xffff` | old value `0`, new `0xffff` | old `0x1800`, new `0x1888` |
| `mtvec_reserved_mode_keeps_previous_mode` | `csrw mtvec, HANDLER\|1`, then `csrw mtvec, HANDLER\|2`, then `csrw mtvec, (HANDLER+0x40)\|3`, then `csrw mtvec, 0; csrrs mtvec, HANDLER\|2`, reading `mtvec` after each of the last three | `0x8000_0102`, `0x8000_0143`, `0x8000_0102` | `0x8000_0101`, `0x8000_0141`, `0x8000_0100` |
| `ecall_with_reserved_mtvec_mode_traps_to_base` | `csrw mtvec, HANDLER\|2; ecall` | jumps to `HANDLER + 2`, mid-instruction: the tracer panics with "Failed to decode instruction" | traps to `HANDLER` |
| `mepc_write_clears_bit_zero` | `csrw mepc, CODE+13; csrr` | `0x8000_000d` | `0x8000_000c` |

These tests also pass on `main` and pin behaviour that does not change: `raw_csrs_store_any_value`, `mret_to_odd_mepc_write_lands_on_even_address` (`JALR` already clears bit 0), and the eight LR/SC tests below.

Compatibility: ZeroOS (`LayerZero-Labs/ZeroOS` `c572b88`), the only trap handler Jolt guests use, aligns `_trap_handler` to 4 bytes, writes it to `mtvec` with MODE 0, and saves and restores `mstatus` and `mepc` around the handler before `mret`. It never depends on the raw-store behaviour. The e2e matrix exercises that boot path.

## LR/SC: decisions, no code change

The LR/SC sequences (`crates/jolt-program/src/expand/memory/{lrw,lrd,scw,scd,shared}.rs`) stay as they are. Each question has a short argument.

### 1. A failed SC stores the old value back

The sequences are straight-line (expansions have no branches), so `SC` always issues a store of `old + success · (rs2 − old)`; on failure that is the old value.
The specification says a failed SC "does not write to memory" (`src/unpriv/zalrsc.adoc:14`, `norm:sc-w_failure`).
On Jolt's single hart with no devices in RAM, writing the value already there changes no observable state: no other hart can see the write, no reservation of another hart exists to invalidate, and the next load returns the same value. So the architectural state matches the specification.

What does differ is which permission checks apply. The specification allows a failed SC to be treated as a store for memory protection (`zalrsc.adoc:21`, `norm:sc_failed_as_store`), and Sail runs the store access check on a failed SC as well.
Jolt's only extra check on this store is the tracer's stack-canary guard, which panics on writes into the 128 bytes above the stack. A successful SC there panics too. So the difference only loses completeness (a program that should run cannot be proved), not soundness.
Changing it would need a branch or a conditional-store row kind for no observable gain.

### 2. Misaligned LR raises `E_Load_Addr_Align` or `E_SAMO_Addr_Align`

Neither: on `main`, a misaligned `LR`/`SC` has no Jolt execution at all.
`LR.W` expands to a reservation update plus `LW`, which is a floor-addressed aligned load. A misaligned address makes the tracer panic ("RAM access is not word aligned"), and in the prover the RAM read/write check rejects the leftover offset bits.
The specification requires an exception here ("LR/SC instructions … always raise an exception when misaligned", `machine.adoc:3110`; `zalrsc.adoc:74`, `norm:lr_sc_alignment`). In Sail that is cause 4 (`E_Load_Addr_Align`) for `LR` and cause 6 (`E_SAMO_Addr_Align`) for `SC`.
Jolt has no step from such a state, so it never claims a successor state that differs from the specification: this is again a completeness loss, not a soundness one.
A model of Jolt's expansions that tags `LR`'s load as an AMO access (and so derives cause 6) describes something Jolt does not do; the model should say "no step" instead.

### 3. `addr ≥ 0x8000_0000` encodes reservability

Both `LR` and `SC` assert `RAM_START ≤ addr` before touching the reservation.
The specification lets each physical region be RsrvNone (non-reservable), RsrvNonEventual or RsrvEventual (`machine.adoc:3051`, `norm:pma_rsrv_levels`), and recommends RsrvEventual for main memory while noting that most I/O regions do not support LR/SC (`machine.adoc:3062`).
Jolt's memory map puts I/O (inputs, outputs, advice, termination and panic bits) below `RAM_START` and ordinary RAM at or above it, so the assertion is exactly "the address is reservable".
It also keeps the cleared reservation value 0 from matching an `SC` to address 0.
Below `RAM_START` the specification would raise an access fault; Jolt has no step, which is again only a completeness loss.

### 4. The reservation policy: exact address and width

`LR.D` records its address in both reservation registers (v32 for word, v33 for doubleword); `LR.W` records it in v32 and clears v33. `SC.W` succeeds iff v32 equals its address, `SC.D` iff v33 does, and every `SC` clears both.
This is legal:

- An `SC` must fail if its address is outside the reservation set (`zalrsc.adoc:117`, `norm:sc_addr_not_in_reservation_fail`). Jolt's reservation set is the bytes the `LR` read, and an `SC` succeeds only at the same address with a width that fits inside those bytes.
- Failing an `SC` elsewhere in the set is allowed: SC "may spuriously fail for implementation reasons, provided progress is eventually made" (`zalrsc.adoc:268`).
- The eventual-success guarantee covers only constrained loops whose `SC` is "to the same effective address and of the same data size" as the `LR` (`zalrsc.adoc:204`, `norm:lrsc_same_address_and_size`). Jolt always succeeds there, so the guarantee holds.
- An `SC` pairs only with the most recent `LR` (`zalrsc.adoc:93`, `norm:sc_pairs_latest_lr`), and an intervening `SC` to any address makes it fail (`zalrsc.adoc:123`, `norm:sc_intervening_sc_fail`); Jolt overwrites on `LR` and clears on `SC`.
- `MRET` may but need not clear the reservation (`machine.adoc:2667`, `norm:xret_clr_lr_resv`); Jolt keeps it. Reset leaves no valid reservation (`machine.adoc:2777`, `norm:ld_rsv_rst`); the registers start at 0, which the reservability assertion keeps from matching.

Other legal policies exist: the Sail C++ emulator reserves an 8-byte granule and ignores width, so it lets `SC.W` succeed at the other word of an `LR.D`'s doubleword, and `SC.D` succeed after `LR.W`. Both behaviours are permitted, so a model that is meant to match Jolt must use Jolt's policy.

Guest programs pinning the policy (all pass on `main` and on this branch): `lr_sc_word_succeeds`, `sc_without_reservation_fails_and_leaves_memory`, `second_sc_fails_because_sc_invalidates_the_reservation`, `doubleword_reservation_covers_word_sc_at_same_address`, `sc_elsewhere_in_reservation_set_fails_legally`, `word_reservation_does_not_cover_doubleword_sc`, `sc_pairs_with_most_recent_lr`, `reservation_survives_trap_and_mret`.

## Tracer panics are halting failures

Several states have no Jolt step, and in each the tracer panics. Most are architectural faults, where the specification takes an exception: a misaligned `LR`/`SC`, an `LR`/`SC` below `RAM_START`, an unsupported CSR or instruction word. One is a host guard with no architectural counterpart: a store into the stack canary, where the specification takes an ordinary step.
A panic ends trace generation before the trace is complete. Rows emitted before it may exist, but no complete trace does, so no proof is produced and the verifier is never asked to accept any claim about that execution. This is acceptable: the guest halts with a failure instead of continuing, which is a completeness loss (a program the specification would continue to run cannot be proved), never a soundness loss.
Soundness does not rest on the tracer panicking. Where the specification would take an exception, a trace that continues without one fails the constraints (RAM checking rejects the offset bits of a misaligned address, the `LR`/`SC` assertion `RAM_START ≤ addr` fails below `RAM_START`, and an unsupported word or CSR never reaches the bytecode). Where the panic is the stack-canary guard, a trace that continues describes a step the specification also takes.
For a refinement proof the relation is: from such a state Jolt has no successor, while the specification takes an exception (architectural faults) or an ordinary step (the stack canary). A refinement statement of the form "every Jolt step is a specification step" holds there trivially; it should not be weakened to model the exception.

## Reserved compressed encodings are rejected

The RVC specification reserves `C.LUI` with `nzimm = 0` for every `rd`, `x0` included; only a nonzero `nzimm` with `rd = x0` is a HINT. Both decompressors, `jolt_riscv::uncompress_rv64_instruction` and `tracer::instruction::uncompress_instruction`, used to expand the one reserved `rd = x0` parcel, `0x6001`, to the canonical no-op, where Sail decodes it as `C_ILLEGAL`. They now return the invalid word `0xffff_ffff`, as they already did for every other reserved RV64C parcel: `C.ADDI4SPN` with `nzuimm = 0`, `C.ADDI16SP` with `nzimm = 0` (which is also `C.LUI` with `rd = x2`), `C.ADDIW` with `rd = x0`, `C.LWSP` and `C.LDSP` with `rd = x0`, `C.JR` with `rs1 = x0`, and two opcode blocks that hold the Zcb instructions Jolt does not implement alongside encodings that stay reserved: quadrant 0 `funct3 = 100`, and quadrant 1 `funct6 = 100111` with `funct2 = 10` or `11`.

With `RV64IMAC_JOLT`, a nonzero rejected parcel fails program-image decoding with `MalformedImage("unknown RV64 opcode")`; the image loader skips all-zero halfwords as padding. Executing a rejected parcel in the tracer panics, so it is a halting failure as in the previous section.

The proof system is unaffected: no inline sequence changes, the golden expansion fixture is unchanged, and the change costs only completeness (a guest that executes `0x6001` now halts where the specification raises an illegal-instruction exception). Against the specification it restores step refinement for this parcel: Jolt now has no successor there, which the refinement relation of the previous section allows, whereas before it took a no-op step that is not a specification step.

With D/Zcd enabled, `C.FLDSP` with `rd = f0` is legal, but both decompressors reject it. Jolt has no D extension, so the expanded `fld` would be rejected anyway; this is left as it is.

## `EBREAK` halts the hart

`EBREAK` expands to `JAL t, 0`, a self-jump that the tracer treats as termination.
The unprivileged specification says `EBREAK` "is used to return control to a debugging environment" (`src/unpriv/rv32.adoc:901`, `norm:ebreak_debug_env_op`).
The debug specification makes that precise: with `dcsr.ebreakm = 1`, "`ebreak` instructions in M-mode enter Debug Mode" (`xml/core_registers.xml:158`), and in Debug Mode the hart is halted, so "there is no forward progress in the normal instruction stream" (`Sdext.adoc:22`).
Entering Debug Mode is not a trap, so `mepc`, `mcause`, `mtval` and `mstatus` stay unchanged.

So Jolt implements an M-only hart whose external debugger has set `dcsr.ebreakm`. The end of the trace is the halt, and the final state is the state before `EBREAK`.
A model with `ebreakm = 0` would instead take a breakpoint exception (cause 3, `mepc = mtval = pc`); that is a configuration difference, not a bug, so the model should be configured with `ebreakm = 1`.
No code changes.

## Evaluation

### Acceptance criteria

- [x] The nine divergence tests above pass on this branch and fail on `main`; the ten behaviour-pinning tests pass on both.
- [x] `cargo nextest run -p jolt-program` passes, including the 360-case golden fixture (24 differences from main).
- [x] `cargo nextest run -p tracer` passes.
- [x] `cargo nextest run -p jolt-prover --features prover-fixtures -E 'binary(e2e_matrix)'` proves and verifies all guests.
- [x] `cargo clippy --all --features host -q --all-targets -- -D warnings`, the same with `host,zk`, and `cargo fmt` pass.

### Testing strategy

The existing tracer guests assert concrete values derived from the specification. The independent architectural reference test covers every instruction/CSR/operand class, invariant-satisfying old states, high-bit operands and aliases; it checks registers, the abstraction relation, invariants and trap/return PCs. The literal expansion-count table pins every path, and the golden fixture pins unchanged expansions and format metadata. The e2e matrix's std guests (`stdlib` and `memory_copy`) cover the boot `mtvec` write and the trap path.

### Performance

The per-instance row differences are listed under Cost per expansion; a guest that only writes the boot trap vector pays four extra rows relative to main. No proof-format or prover-structure change.

## Documentation

Module docs in `crates/jolt-program/src/expand/control_flow/{csr,ecall,mret,csrrw,csrrs,ebreak}.rs` and `tracer/src/instruction/{ecall,mret,csrrw,ebreak}.rs` describe the semantics above, and `tracer/src/utils/virtual_registers.rs` documents what v39 holds. The book's emulation page (`book/src/how/architecture/emulation.md`) now describes the new trap and CSR behaviour, and its LR/SC section is corrected to match the code: `LR.D` sets both reservation registers, and SC success is computed from the reservation, not supplied as advice.

## Open decisions

- **`mcause` write policy.** `mcause`'s exception code is WLRL, so any deterministic result of an illegal write is allowed (`priv-csrs.adoc:653`). Jolt stores every write raw. Sail's `legalize_cause` (sail-riscv `model/core/sys_regs.sail`, checked at `8118453`) instead keeps the previous value when the written code is not one the hart supports; a write of all ones tells them apart. Matching Sail would need a membership test over the supported codes in straight-line rows. Software writes only legal codes to WLRL fields, so the cheaper choice is to configure the model's `legalize_cause` as raw storage for Jolt, which is the assumption here.

- **Faults instead of panics.** A misaligned `LR`/`SC` and an `LR`/`SC` below `RAM_START` make the tracer panic where the specification takes an exception; a store into the stack canary panics on a host guard where the specification takes an ordinary step. All are halting failures (see "Tracer panics are halting failures"), so they cost completeness only. Turning the architectural faults into traps would need conditional control flow in expansions; nothing here depends on it.

## References

- RISC-V Privileged and Unprivileged ISA, riscv-isa-manual `717aeca3`.
- RISC-V Debug Specification, riscv-debug-spec `40e5c354`.
- Sail RISC-V model: `legalize_mstatus`, `legalize_tvec` (`Xtvec_Ignore`), `reset_sys`.
