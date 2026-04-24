# BN254 Fr Coprocessor v2 — R1CS + ISA Implementation Spec

Mechanical spec for Phase 2 (new tracer instructions) and Phase 3 (R1CS restructure) of the migration plan in `specs/bn254-fr-coprocessor.md`. Every number, slot index, and row is concrete and source-of-truth for implementation.

Scope: this document covers only the ISA/R1CS/polynomial-id surface. FR Twist structural mirror (Phase 4) and v1 cherry-pick inventory live in sibling sections.

---

## Section 1 — ISA encoding table

All nine instructions share opcode `0x0B` (custom-0), funct7 `0x40`. funct3 selects. R-type layout throughout: `funct7(25..32) | rs2(20..25) | rs1(15..20) | funct3(12..15) | rd(7..12) | opcode(0..7)`. Field-register slot indices use the low 4 bits of the 5-bit register field (0..=15); the high bit must be 0.

Register-type column: `FReg[i]` = slot `i` in `cpu.field_regs` (16 × [u64;4]); `XReg[i]` = integer register `i` in the 32-slot RISC-V register file. Tracer state update column lists the authoritative mutations per cycle.

| funct3 | Mnemonic         | rs1             | rs2             | rd              | Semantics                                                  | Tracer state update                                                                                   |
|--------|------------------|-----------------|-----------------|-----------------|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| 0x02   | `FieldMul`       | FReg[frs1]      | FReg[frs2]      | FReg[frd]       | `FReg[frd] = FReg[frs1] · FReg[frs2]` (mod p)              | Write `FReg[frd]`; emit `FieldRegEvent { slot = frd, new = product_limbs }`                           |
| 0x03   | `FieldAdd`       | FReg[frs1]      | FReg[frs2]      | FReg[frd]       | `FReg[frd] = FReg[frs1] + FReg[frs2]` (mod p)              | Write `FReg[frd]`; emit `FieldRegEvent { slot = frd, new = sum_limbs }`                               |
| 0x04   | `FieldInv`       | FReg[frs1]      | (unused)        | FReg[frd]       | `FReg[frd] = FReg[frs1]⁻¹` (mod p; `0⁻¹` is undefined, guest must avoid) | Write `FReg[frd]`; emit `FieldRegEvent { slot = frd, new = inv_limbs }`                |
| 0x05   | `FieldSub`       | FReg[frs1]      | FReg[frs2]      | FReg[frd]       | `FReg[frd] = FReg[frs1] − FReg[frs2]` (mod p)              | Write `FReg[frd]`; emit `FieldRegEvent { slot = frd, new = diff_limbs }`                              |
| 0x06   | `FieldAssertEq`  | FReg[frs1]      | FReg[frs2]      | (unused)        | `assert FReg[frs1] == FReg[frs2]` (mod p); no write        | No field-register mutation; emit `FieldRegEvent { slot = frs1, new = old }` (no-op write, see §6)     |
| 0x07   | `FieldMov`       | XReg[rs1]       | (unused)        | FReg[frd]       | `FReg[frd] = (XReg[rs1] as Fr)`; integer embeds into Fr    | Write `FReg[frd] = [rs1, 0, 0, 0]`; emit `FieldRegEvent { slot = frd, new }`; XReg read binds Rs1Value |
| 0x08   | `FieldSLL64`     | XReg[rs1]       | (unused)        | FReg[frd]       | `FReg[frd] = (XReg[rs1] as Fr) · 2⁶⁴`                      | Write `FReg[frd] = [0, rs1, 0, 0]`; emit `FieldRegEvent { slot = frd, new }`                          |
| 0x09   | `FieldSLL128`    | XReg[rs1]       | (unused)        | FReg[frd]       | `FReg[frd] = (XReg[rs1] as Fr) · 2¹²⁸`                     | Write `FReg[frd] = [0, 0, rs1, 0]`; emit `FieldRegEvent { slot = frd, new }`                          |
| 0x0A   | `FieldSLL192`    | XReg[rs1]       | (unused)        | FReg[frd]       | `FReg[frd] = (XReg[rs1] as Fr) · 2¹⁹²` (caller must canonicalize < p) | Write `FReg[frd] = [0, 0, 0, rs1]`; emit `FieldRegEvent { slot = frd, new }`                    |

**Register-field mixing pin-down:** the bytecode `rs1`/`rs2`/`rd` bit-fields are the same 5 bits for every instruction. What differs is interpretation. For field-only ops (FMUL/FADD/FSUB/FINV/FieldAssertEq) the low 4 bits index FReg; the bytecode's `Rs1Value`/`Rs2Value`/`RdWriteValue` columns are treated as zero (no XReg read/write occurs). For bridge ops (FieldMov/FieldSLL*) the `rs1` bits index XReg (5 bits, 0..=31) and the `rd` bits index FReg (low 4 bits, 0..=15); `rs2` is zero. `FieldAssertEq` reads two FRegs and writes nothing — its `rd` bits are ignored (zero).

---

## Section 2 — R1CS witness slot layout (post-v2)

```rust
pub const V_CONST: usize = 0;

// Base RV interpreter columns (unchanged from v1)
pub const V_LEFT_INSTRUCTION_INPUT:  usize = 1;
pub const V_RIGHT_INSTRUCTION_INPUT: usize = 2;
pub const V_PRODUCT:                 usize = 3;
pub const V_SHOULD_BRANCH:           usize = 4;
pub const V_PC:                      usize = 5;
pub const V_UNEXPANDED_PC:           usize = 6;
pub const V_IMM:                     usize = 7;
pub const V_RAM_ADDRESS:             usize = 8;
pub const V_RS1_VALUE:               usize = 9;
pub const V_RS2_VALUE:               usize = 10;
pub const V_RD_WRITE_VALUE:          usize = 11;
pub const V_RAM_READ_VALUE:          usize = 12;
pub const V_RAM_WRITE_VALUE:         usize = 13;
pub const V_LEFT_LOOKUP_OPERAND:     usize = 14;
pub const V_RIGHT_LOOKUP_OPERAND:    usize = 15;
pub const V_NEXT_UNEXPANDED_PC:      usize = 16;
pub const V_NEXT_PC:                 usize = 17;
pub const V_NEXT_IS_VIRTUAL:         usize = 18;
pub const V_NEXT_IS_FIRST_IN_SEQUENCE: usize = 19;
pub const V_LOOKUP_OUTPUT:           usize = 20;
pub const V_SHOULD_JUMP:             usize = 21;

// RV base circuit flags (14 flags, unchanged from v1)
pub const V_FLAG_ADD_OPERANDS:              usize = 22;
pub const V_FLAG_SUBTRACT_OPERANDS:         usize = 23;
pub const V_FLAG_MULTIPLY_OPERANDS:         usize = 24;
pub const V_FLAG_LOAD:                      usize = 25;
pub const V_FLAG_STORE:                     usize = 26;
pub const V_FLAG_JUMP:                      usize = 27;
pub const V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD: usize = 28;
pub const V_FLAG_VIRTUAL_INSTRUCTION:       usize = 29;
pub const V_FLAG_ASSERT:                    usize = 30;
pub const V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC: usize = 31;
pub const V_FLAG_ADVICE:                    usize = 32;
pub const V_FLAG_IS_COMPRESSED:             usize = 33;
pub const V_FLAG_IS_FIRST_IN_SEQUENCE:      usize = 34;
pub const V_FLAG_IS_LAST_IN_SEQUENCE:       usize = 35;

// BN254 Fr coprocessor flags — existing 4 (kept), new 5 (added)
pub const V_FLAG_IS_FIELD_MUL:       usize = 36; // KEPT
pub const V_FLAG_IS_FIELD_ADD:       usize = 37; // KEPT
pub const V_FLAG_IS_FIELD_SUB:       usize = 38; // KEPT
pub const V_FLAG_IS_FIELD_INV:       usize = 39; // KEPT
pub const V_FLAG_IS_FIELD_ASSERT_EQ: usize = 40; // NEW
pub const V_FLAG_IS_FIELD_MOV:       usize = 41; // NEW
pub const V_FLAG_IS_FIELD_SLL64:     usize = 42; // NEW
pub const V_FLAG_IS_FIELD_SLL128:    usize = 43; // NEW
pub const V_FLAG_IS_FIELD_SLL192:    usize = 44; // NEW

// Product factors (shift vs v1: v1 had these at 49/50 after 11 FR slots;
// v2 reclaims FR operand + limb + limb-sum slots so these move down)
pub const V_BRANCH:       usize = 45;
pub const V_NEXT_IS_NOOP: usize = 46;

pub const NUM_R1CS_INPUTS:         usize = 44;  // indices 1..=44
pub const NUM_PRODUCT_FACTORS:     usize = 2;
pub const NUM_VARS_PER_CYCLE:      usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS; // = 47
pub const NUM_EQ_CONSTRAINTS:      usize = 29;  // 19 RV base + 10 FR (see §3)
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;   // unchanged (V_PRODUCT, ShouldBranch, ShouldJump)
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // = 32
```

### Classification

| Status            | Slot(s) / Flag(s)                                                                                                                           |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Existing (RV base) | `V_CONST`, `V_LEFT_INSTRUCTION_INPUT`, `V_RIGHT_INSTRUCTION_INPUT`, `V_PRODUCT`, `V_SHOULD_BRANCH`, `V_PC`, `V_UNEXPANDED_PC`, `V_IMM`, `V_RAM_*`, `V_RS1_VALUE`, `V_RS2_VALUE`, `V_RD_WRITE_VALUE`, `V_LEFT_LOOKUP_OPERAND`, `V_RIGHT_LOOKUP_OPERAND`, `V_NEXT_*`, `V_LOOKUP_OUTPUT`, `V_SHOULD_JUMP`, all 14 `V_FLAG_*` base flags, `V_BRANCH`, `V_NEXT_IS_NOOP` |
| Kept from v1      | `V_FLAG_IS_FIELD_MUL`, `V_FLAG_IS_FIELD_ADD`, `V_FLAG_IS_FIELD_SUB`, `V_FLAG_IS_FIELD_INV`. `V_PRODUCT` continues to route FMUL/FINV (see §3). |
| NEW in v2         | `V_FLAG_IS_FIELD_ASSERT_EQ`, `V_FLAG_IS_FIELD_MOV`, `V_FLAG_IS_FIELD_SLL64`, `V_FLAG_IS_FIELD_SLL128`, `V_FLAG_IS_FIELD_SLL192`             |
| DROPPED from v1   | `V_FIELD_OP_A` (42), `V_FIELD_OP_B` (43), `V_FIELD_OP_RESULT` (44), `V_FIELD_REG_READ_LIMB` (45), `V_FIELD_REG_WRITE_LIMB` (46), `V_LIMB_SUM_A` (47), `V_LIMB_SUM_B` (48), `V_FLAG_IS_FMOV_I2F` (40), `V_FLAG_IS_FMOV_F2I` (41) |
| Virtual (not slots) | `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` — not in the R1CS witness vector; their per-cycle values are bound by the FR Twist sumcheck at Stage 2. The Spartan outer sumcheck gets their MLE evaluations from `VirtualPolynomialMaterialize` output at the random cycle point (identical mechanism to `Rs1Value`/`Rs2Value`/`RdWriteValue` for the integer Registers Twist). |

### Slot-count delta vs v1

- v1: `NUM_R1CS_INPUTS = 48`, `NUM_VARS_PER_CYCLE = 51`, `NUM_EQ_CONSTRAINTS = 31`, `NUM_CONSTRAINTS_PER_CYCLE = 34`.
- v2: `NUM_R1CS_INPUTS = 44`, `NUM_VARS_PER_CYCLE = 47`, `NUM_EQ_CONSTRAINTS = 29`, `NUM_CONSTRAINTS_PER_CYCLE = 32`.

Net: 4 fewer witness slots (dropped 7 FR data slots + 2 FMov flags = 9; added 5 new flags = net −4). Two fewer eq rows (dropped rows 27, 28, 29, 30 = 4; added `FieldAssertEq` + `FieldMov` + `FieldSLL64` + `FieldSLL128` + `FieldSLL192` = 5; plus kept `FieldAdd`, `FieldSub`, and the 6 FMUL/FINV binding rows but now binding to *virtual* `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` instead of the dropped `V_FIELD_OP_*` witness columns — row count within FR block drops from 12 to 10).

---

## Section 3 — R1CS row definitions

Format below mirrors v1's `row::<F>(&[(slot, coeff), ...])` syntax. Symbolic slot names are used; substitute the integer constants from §2 when emitting. Entries with coefficient 0 are elided. `SparseRow` entries for the *virtual* columns `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` use their `r1cs_variable_index()` — see §4; here they are written symbolically as `V_FIELD_RS1_VALUE`, `V_FIELD_RS2_VALUE`, `V_FIELD_RD_VALUE` for readability.

### Layout (post-v2)

```
Rows  0..=18   : RV base eq constraints (unchanged from v1)
Rows 19..=20   : FieldAdd, FieldSub gates
Rows 21..=26   : FieldMul operand/result binding (3) + FieldInv operand/result binding (3)
Rows 27..=28   : FieldAssertEq, FieldMov
Rows 29..=31   : FieldSLL64, FieldSLL128, FieldSLL192
Rows 32..=34   : Product constraints (V_PRODUCT, ShouldBranch, ShouldJump) — unchanged
```

### Rows 0..=18 (RV base)

Unchanged from v1 `crates/jolt-r1cs/src/constraints/rv64.rs` rows 0..=18. Copy verbatim.

### Row 19 — FieldAdd gate

Purpose: enforces `FieldRs1Value + FieldRs2Value = FieldRdValue` on FADD cycles.

```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ADD, 1)]));
b_rows.push(row::<F>(&[
    (V_FIELD_RS1_VALUE, 1),
    (V_FIELD_RS2_VALUE, 1),
    (V_FIELD_RD_VALUE, -1),
]));
c_rows.push(empty());
```

### Row 20 — FieldSub gate

```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SUB, 1)]));
b_rows.push(row::<F>(&[
    (V_FIELD_RS1_VALUE, 1),
    (V_FIELD_RS2_VALUE, -1),
    (V_FIELD_RD_VALUE, -1),
]));
c_rows.push(empty());
```

### Rows 21..=23 — FieldMul via `V_PRODUCT` reuse

**Recommendation: reuse `V_PRODUCT` (matches v1).** Adding a new product constraint would bump `NUM_PRODUCT_CONSTRAINTS` from 3 to 4, which ripples into the product virtual sumcheck dimensions and cross-verify compatibility for no gain — V_PRODUCT is otherwise idle on FMUL cycles (RV flags AddOperands/SubtractOperands/MultiplyOperands are all 0, so product constraint 32 is unused by the interpreter on those cycles).

Row 21 — FMUL A binding:
```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
b_rows.push(row::<F>(&[
    (V_LEFT_INSTRUCTION_INPUT, 1),
    (V_FIELD_RS1_VALUE, -1),
]));
c_rows.push(empty());
```

Row 22 — FMUL B binding:
```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
b_rows.push(row::<F>(&[
    (V_RIGHT_INSTRUCTION_INPUT, 1),
    (V_FIELD_RS2_VALUE, -1),
]));
c_rows.push(empty());
```

Row 23 — FMUL product output:
```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_FIELD_RD_VALUE, -1)]));
c_rows.push(empty());
```

### Rows 24..=26 — FieldInv via `V_PRODUCT` reuse

Row 24 — FINV A binding:
```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
b_rows.push(row::<F>(&[
    (V_LEFT_INSTRUCTION_INPUT, 1),
    (V_FIELD_RS1_VALUE, -1),
]));
c_rows.push(empty());
```

Row 25 — FINV result-as-right:
```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
b_rows.push(row::<F>(&[
    (V_RIGHT_INSTRUCTION_INPUT, 1),
    (V_FIELD_RD_VALUE, -1),
]));
c_rows.push(empty());
```

Row 26 — FINV unit product:
```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_CONST, -1)]));
c_rows.push(empty());
```

### Row 27 — FieldAssertEq gate

Purpose: enforces `FieldRs1Value = FieldRs2Value` on assertion cycles. No write. `FieldRdValue` on this cycle is a no-op write (see §6 — event schema).

```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ASSERT_EQ, 1)]));
b_rows.push(row::<F>(&[
    (V_FIELD_RS1_VALUE, 1),
    (V_FIELD_RS2_VALUE, -1),
]));
c_rows.push(empty());
```

### Row 28 — FieldMov gate

Purpose: enforces `Rs1Value (integer) = FieldRdValue (Fr)` on FMov cycles. `V_RS1_VALUE` is bound by the Registers Twist; `V_FIELD_RD_VALUE` is bound by the FR Twist. This row is the cross-domain bridge.

```rust
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MOV, 1)]));
b_rows.push(row::<F>(&[
    (V_RS1_VALUE, 1),
    (V_FIELD_RD_VALUE, -1),
]));
c_rows.push(empty());
```

### Rows 29..=31 — FieldSLL64 / FieldSLL128 / FieldSLL192

Row 29 — FieldSLL64:
```rust
// IsFieldSLL64 · (2^64 · V_RS1_VALUE − V_FIELD_RD_VALUE) = 0
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL64, 1)]));
b_rows.push(row_wide::<F>(&[
    (V_RS1_VALUE, 1i128 << 64),  // careful: i128::MAX = 2^127-1, so 2^64 fits
    (V_FIELD_RD_VALUE, -1),
]));
c_rows.push(empty());
```

Rows 30/31 are structurally identical with coefficients `2^128` and `2^192` respectively. **Implementation note:** coefficients `2^128` and `2^192` exceed `i128` range. Replace `row_wide` with a new helper `row_bigcoeff::<F>(&[(slot, coeff_as_field_element)])` that accepts pre-converted field elements (the coefficient is pre-computed as `F::from_u64(1).square().square()…` or lifted from a u256 literal). Suggested signature:

```rust
fn row_bigcoeff<F: Field>(entries: &[(usize, F)]) -> SparseRow<F> {
    entries.iter().filter(|(_, c)| !c.is_zero()).copied().collect()
}
```

Wire row 30 as:
```rust
let two_pow_128: F = F::from_u64(1u64 << 63) * F::from_u64(2) * F::from_u64(1u64 << 63) * F::from_u64(2);
// or more robustly: F::from_u128(1 << 64).square()
a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL128, 1)]));
b_rows.push(row_bigcoeff::<F>(&[
    (V_RS1_VALUE, two_pow_128),
    (V_FIELD_RD_VALUE, -F::one()),
]));
c_rows.push(empty());
```

Row 31 analogous with `F::from_u128(1 << 64).pow([3])`.

### Rows 32..=34 — Product constraints (unchanged)

```rust
// 32: V_PRODUCT = V_LEFT × V_RIGHT (FMUL/FINV rely on this)
a_rows.push(row::<F>(&[(V_LEFT_INSTRUCTION_INPUT, 1)]));
b_rows.push(row::<F>(&[(V_RIGHT_INSTRUCTION_INPUT, 1)]));
c_rows.push(row::<F>(&[(V_PRODUCT, 1)]));

// 33: ShouldBranch = LookupOutput × Branch
// 34: ShouldJump = Jump × (1 − NextIsNoop)
```

### What's gone vs v1

- **No operand-binding rows** between a committed `V_FIELD_OP_A/B/RESULT` witness column and a virtual value. The FR Twist sumcheck binds `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` directly at the Spartan random cycle point.
- **No limb-sum rows** (v1 rows 29/30). `V_LIMB_SUM_A/B` are gone; the limb↔Fr bridge is no longer an R1CS identity at all. Bridging happens through per-cycle `FieldMov` + `FieldSLL*` + `FieldAdd` arithmetic on 7 cycles (load) or 12 cycles (extract) — each cycle's R1CS row operates purely on one-element facts (`Rs1Value` vs `FieldRdValue`), no four-cycle reconstruction needed.
- **No FMov-I2F/F2I rows.** v1 rows 27/28 bound `V_FIELD_REG_WRITE_LIMB` ↔ `V_RS1_VALUE` / `V_RD_WRITE_VALUE` ↔ `V_FIELD_REG_READ_LIMB`. Both sides of those equalities are obsolete — the limb columns are dropped, and field-to-integer extraction is now an Advice + FieldMov + FieldSLL* + FieldAssertEq chain (§1 table note; SDK sequence in `bn254-fr-coprocessor.md` §Architecture).

---

## Section 4 — PolynomialId additions / removals

File: `crates/jolt-compiler/src/polynomial_id.rs`.

### Additions

Three NEW virtual columns for the FR operand openings. They are *R1CS inputs* (referenced by the A/B coefficient vectors of rows 19..=31 above) but they are NOT backed by a slot in the per-cycle witness vector `z`. Instead, their evaluations at the Spartan random cycle point are supplied by the FR Twist sumcheck output (mirroring how `Rs1Value`, `Rs2Value`, `RdWriteValue` are supplied by the Registers Twist).

```rust
// FR virtual R1CS inputs — proven via FR Twist, not stored in z.
FieldRs1Value,
FieldRs2Value,
FieldRdValue,
```

Descriptor for each:

```rust
PolynomialDescriptor {
    source: PolySource::Virtual,          // materialized via FR Twist sumcheck output
    committed: false,
    storage: StorageHint::Dense,           // T-element virtual MLE
    witness_slot: None,                    // NOT an R1CS witness slot (no z[] index)
}
```

`r1cs_variable_index()` — they DO get variable indices for R1CS matrix lookup purposes, but the constraint evaluator must intercept these three indices and pull values from the FR Twist output rather than `z[var_idx]`:

| Variant           | `r1cs_variable_index()` |
|-------------------|--------------------------|
| `FieldRs1Value`   | `Some(42)`  (reuses dropped `V_FIELD_OP_A` slot index) |
| `FieldRs2Value`   | `Some(43)`  (reuses dropped `V_FIELD_OP_B` slot index) |
| `FieldRdValue`    | `Some(44)`  (reuses dropped `V_FIELD_OP_RESULT` slot index) |

NOTE on slot reuse: v2 repurposes indices 42/43/44 as the *virtual* `FieldRs1/Rs2/RdValue` references. The witness vector `z` no longer has entries at these indices (its length is 47 per §2). The R1CS builder must arrange that when a constraint row has a coefficient at index 42/43/44, the Spartan evaluator reads from the FR Twist output buffer rather than `z`. Implementation option: treat these as "virtual z columns" with a separate overlay buffer, or let the Spartan Az/Bz/Cz materializer dispatch on `PolynomialId` rather than raw index.

Also add five new `CircuitFlags` variants (see §1 / §5) via the `OpFlag(usize)` existing mechanism — no new `PolynomialId` variant needed; `CircuitFlags::IsFieldAssertEq as usize = 19`, etc., map to `OpFlag(19)` through `OpFlag(23)`.

### Removals

Delete the following variants and their descriptor/`r1cs_variable_index` arms:

| Variant                 | Reason                                                                        |
|-------------------------|-------------------------------------------------------------------------------|
| `FieldOpOperandA`       | Replaced by virtual `FieldRs1Value`                                           |
| `FieldOpOperandB`       | Replaced by virtual `FieldRs2Value`                                           |
| `FieldOpResultValue`    | Replaced by virtual `FieldRdValue`                                            |
| `FieldRegReadLimb`      | FMov-F2I deleted                                                              |
| `FieldRegWriteLimb`     | FMov-I2F deleted                                                              |
| `FieldRegReadValue`     | Over-committed; dropped per v2 invariant 1 in `bn254-fr-coprocessor.md`       |
| `FieldRegWriteValue`    | Over-committed; dropped per v2 invariant 1                                    |
| `LimbSumA`              | Plan P rows 29/30 gone                                                        |
| `LimbSumB`              | Plan P rows 29/30 gone                                                        |
| `BridgeValWeightA/B`    | Preprocessed limb-bridge weights — no limb bridge in v2                       |
| `BridgeAnchorA/B`       | Same                                                                          |
| `BridgeValWeight`       | Runtime bridge scratch — gone                                                 |
| `BridgeAnchorWeight`    | Same                                                                          |
| `IsFieldOpAny`          | Derived indicator for bridge — no longer needed                               |
| `FieldOpBGated`         | Same                                                                          |
| `IsFieldOpNoInv`        | Same                                                                          |
| `WeightAOfRd`           | Stage 5 LimbSum reduction gone                                                |
| `WeightBOfRd`           | Same                                                                          |
| `RdIncAtBridge`         | Bridge alias — bridge gone                                                    |
| `RdIncAtBridgeA/B`      | Same                                                                          |

Keep `FieldRegInc`, `FieldRegRa`, `FieldRegVal`, `FieldRegEqCycle` (structural polys for FR Twist — analogs of `RdInc`, `Rs1Ra`, `RegistersVal`, etc.).

### Changes (no rename, just descriptor update)

None — `FieldRegInc`/`FieldRegRa`/`FieldRegVal`/`FieldRegEqCycle` descriptors unchanged.

---

## Section 5 — Tracer instructions

Five new tracer instruction files. Each mirrors `tracer/src/instruction/field_op.rs`'s dispatch pattern (funct7=0x40 family, funct3 selector). Because the bridge instructions have a fundamentally different operand shape (read XReg, write FReg) from FieldOp (read FReg, write FReg), each gets its own struct rather than folding into `FieldOp`. This also simplifies `has_side_effects`, `operands()`, and the `Flags` impl.

### Common structure

All five use `FormatR` for the bytecode operands. All five have `type RAMAccess = ();` (no RAM interaction). All emit exactly one `FieldRegEvent` per `trace()` call (single access per cycle, per v2 invariant 3).

### `tracer/src/instruction/field_mov.rs` (funct3 = 0x07)

```rust
pub const FUNCT3_FIELD_MOV: u8 = 0x07;

impl RISCVInstruction for FieldMov {
    const MASK: u32 = 0xfe00_707f;   // opcode + funct3 + funct7
    const MATCH: u32 = (BN254_FR_FUNCT7 << 25) | ((FUNCT3_FIELD_MOV as u32) << 12) | FIELD_OP_OPCODE;
    type Format = FormatR;
    type RAMAccess = ();

    fn execute(&self, cpu: &mut Cpu, _: &mut ()) {
        let frd  = self.operands.rd  as usize;
        let rs1  = self.operands.rs1 as usize;
        let x    = cpu.x[rs1];                   // u64 integer-register value
        cpu.field_regs[frd] = [x, 0, 0, 0];      // natural-form limbs: low = x
    }
}

impl RISCVTrace for FieldMov {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let frd      = self.operands.rd;
        let frd_pre  = cpu.field_regs[frd as usize];
        // Standard capture/execute/capture dance:
        let mut cycle = RISCVCycle::<Self> { instruction: *self, register_state: Default::default(), ram_access: () };
        self.operands().capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.ram_access);
        self.operands().capture_post_execution_state(&mut cycle.register_state, cpu);
        let frd_post = cpu.field_regs[frd as usize];
        if let Some(tv) = trace {
            cpu.field_reg_events.push(FieldRegEvent {
                cycle: cpu.trace_len + tv.len(),
                slot: frd as usize,
                old: frd_pre,
                new: frd_post,
            });
            tv.push(cycle.into());
        }
    }
}
```

### `field_sll64.rs` / `field_sll128.rs` / `field_sll192.rs` (funct3 = 0x08/0x09/0x0A)

Identical skeleton to `field_mov.rs`; only `execute` differs:

- `FieldSLL64`:  `cpu.field_regs[frd] = [0, x, 0, 0];`
- `FieldSLL128`: `cpu.field_regs[frd] = [0, 0, x, 0];`
- `FieldSLL192`: `cpu.field_regs[frd] = [0, 0, 0, x];`

Event payload identical shape (`slot=frd, old, new`).

### `field_assert_eq.rs` (funct3 = 0x06)

```rust
pub const FUNCT3_FIELD_ASSERT_EQ: u8 = 0x06;

impl RISCVInstruction for FieldAssertEq {
    // MASK/MATCH analogous
    type Format = FormatR;
    type RAMAccess = ();

    fn execute(&self, cpu: &mut Cpu, _: &mut ()) {
        let frs1 = self.operands.rs1 as usize;
        let frs2 = self.operands.rs2 as usize;
        debug_assert_eq!(
            cpu.field_regs[frs1], cpu.field_regs[frs2],
            "FieldAssertEq failed: FReg[{frs1}] != FReg[{frs2}]"
        );
        // No field-register write.
    }
}

impl RISCVTrace for FieldAssertEq {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Emit a no-op write event (slot = frs1, old = new = current value) so the
        // FR Twist sees a FieldRs1Value read at this cycle. The Ra one-hot for
        // rs2 covers FieldRs2Value. See §6 for the rationale.
        let frs1 = self.operands.rs1;
        let val  = cpu.field_regs[frs1 as usize];
        let mut cycle = RISCVCycle::<Self> { instruction: *self, register_state: Default::default(), ram_access: () };
        self.execute(cpu, &mut cycle.ram_access);
        if let Some(tv) = trace {
            cpu.field_reg_events.push(FieldRegEvent {
                cycle: cpu.trace_len + tv.len(),
                slot: frs1 as usize,
                old: val,
                new: val,   // no-op write preserves single-write-per-cycle invariant
            });
            tv.push(cycle.into());
        }
    }
}
```

### FieldOp (existing) — what STAYS and what CHANGES

`tracer/src/instruction/field_op.rs` covers FMUL (0x02), FADD (0x03), FINV (0x04), FSUB (0x05) today. Keep the funct3 dispatch and the `execute` body verbatim — the Fr arithmetic semantics are unchanged.

**CHANGE:** the `trace()` method drops the `FieldOpPayload` construction. The simpler v2 event has no `op: Option<FieldOpPayload>` and no `fmov: Option<FMovPayload>` fields — just `slot`, `old`, `new`, `cycle`. See §6.

```rust
// v2 trace() body (simplified):
fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
    let frd     = self.operands.rd;
    let frd_pre = cpu.field_regs[frd as usize];
    let mut cycle = RISCVCycle::<Self> { instruction: *self, register_state: Default::default(), ram_access: () };
    self.operands().capture_pre_execution_state(&mut cycle.register_state, cpu);
    self.execute(cpu, &mut cycle.ram_access);
    self.operands().capture_post_execution_state(&mut cycle.register_state, cpu);
    let frd_post = cpu.field_regs[frd as usize];
    if let Some(tv) = trace {
        cpu.field_reg_events.push(FieldRegEvent {
            cycle: cpu.trace_len + tv.len(),
            slot: frd as usize,
            old: frd_pre,
            new: frd_post,
        });
        tv.push(cycle.into());
    }
}
```

### Flags impl per instruction

Each of the five new structs implements `Flags` with exactly one circuit flag set:

| Struct          | `circuit_flags()[IsFieldMov]` | `[IsFieldSLL64]` | `[IsFieldSLL128]` | `[IsFieldSLL192]` | `[IsFieldAssertEq]` |
|-----------------|-------------------------------|------------------|-------------------|-------------------|---------------------|
| `FieldMov`      | true                          | false            | false             | false             | false               |
| `FieldSLL64`    | false                         | true             | false             | false             | false               |
| `FieldSLL128`   | false                         | false            | true              | false             | false               |
| `FieldSLL192`   | false                         | false            | false             | true              | false               |
| `FieldAssertEq` | false                         | false            | false             | false             | true                |

`InstructionFlags`: all five set `LeftOperandIsRs1Value = true` for FieldMov/SLL* (so `V_RS1_VALUE` is populated from the integer register). FieldAssertEq sets none (no integer-register interaction).

### `CircuitFlags` additions

In `crates/jolt-instructions/src/flags.rs`:

```rust
pub enum CircuitFlags {
    // ... existing 20 variants 0..=19 (kept) ...
    // NOTE: v1 IsFMovI2F (18) and IsFMovF2I (19) are REMOVED. Variants 14..=17
    // (IsFieldMul/Add/Sub/Inv) stay.

    IsFieldAssertEq,  // = 18
    IsFieldMov,       // = 19
    IsFieldSLL64,     // = 20
    IsFieldSLL128,    // = 21
    IsFieldSLL192,    // = 22
}

pub const NUM_CIRCUIT_FLAGS: usize = 23;
```

---

## Section 6 — FieldRegEvent schema in v2

File: `crates/jolt-witness/src/derived.rs`.

**v1 event:**
```rust
pub struct FieldRegEvent {
    pub cycle: usize,
    pub slot: usize,
    pub old: FrLimbs,
    pub new: FrLimbs,
    pub op: Option<FieldOpPayload>,     // funct3, a, b
    pub fmov: Option<FMovPayload>,      // funct3, limb_idx, limb
}
```

**v2 event:** drop both payloads. The `slot`/`new` pair is the only FR-side fact the R1CS and FR Twist need; all operand information is reconstructed from the bytecode fields (`rs1`, `rs2`, `rd`) and the FR Twist's read side (Ra one-hot polys indexed by `frs1(c)`, `frs2(c)`).

```rust
#[derive(Clone, Copy, Debug)]
pub struct FieldRegEvent {
    pub cycle: usize,
    pub slot: usize,      // = frd(c) for arithmetic/bridge ops;
                          //   = frs1(c) for FieldAssertEq (no-op write)
    pub old: FrLimbs,     // pre-state of field_regs[slot]
    pub new: FrLimbs,     // post-state of field_regs[slot]
}
```

**`FieldOpPayload` struct: DELETED.** `FMovPayload`: DELETED. `FIELD_OP_FUNCT3_*` constants: DELETED (no consumer in v2 — R1CS dispatches on circuit flags, not funct3).

**Read-side binding.** Where v1 attached `(a, b)` to the event, v2 binds reads via the FR Twist's `FieldRs1Ra` / `FieldRs2Ra` one-hot polynomials (constructed from bytecode `rs1`/`rs2` fields — free, already committed as part of `BytecodeRead*`). At Stage 2 the FR Twist sumcheck γ-batches three virtual-value terms:

```
input_claim_FR = gamma^0 · FieldRs1Value(r_cycle)
               + gamma^1 · FieldRs2Value(r_cycle)
               + gamma^2 · FieldRdValue(r_cycle)

where:
  FieldRs1Value(r_cycle) = Σ_s Val_fr(s, r_cycle) · FieldRs1Ra(s, r_cycle)
  FieldRs2Value(r_cycle) = Σ_s Val_fr(s, r_cycle) · FieldRs2Ra(s, r_cycle)
  FieldRdValue(r_cycle)  = Σ_s Val_fr(s, r_cycle) · FieldRdRa(s, r_cycle)
```

This is the exact γ-batching pattern used by the integer `RegistersReadWriteChecking` sumcheck.

**FieldAssertEq rationale for no-op write.** To preserve the "single access per cycle" invariant (every FR cycle emits exactly one event; FR Twist state update is unambiguous), FieldAssertEq emits a `slot = frs1, old = new = current_value` event. The FR Twist's Val sequence observes a zero increment at slot `frs1`, preserving correctness. Alternative: add a "no-event" path, which would require FR Twist to distinguish between "cycle with no FR event" (like every non-FR instruction) and "FR assertion cycle"; the no-op write is simpler.

---

## Section 7 — Migration order within Phases 2 and 3

Goal: keep `cargo check -p <crate>` green after every step where possible. Where not possible, the atomic-landing groups are flagged.

### Phase 2 (tracer + flags)

Phase 2 is standalone — it adds new instruction decoders and flag enum variants but does not yet wire them into R1CS. The branch compiles cleanly at each step.

1. **P2.1** Add `CircuitFlags::IsFieldAssertEq`, `IsFieldMov`, `IsFieldSLL64`, `IsFieldSLL128`, `IsFieldSLL192` to `crates/jolt-instructions/src/flags.rs`. Bump `NUM_CIRCUIT_FLAGS` from 20 to 23 (remove the two v1 FMov flags simultaneously — see §5). Update the `circuit_flags_count_matches_enum` test. *Compiles.*
2. **P2.2** Create `tracer/src/instruction/field_mov.rs`. Add the `FieldMov` struct (mirror `FieldOp` skeleton; single-op execute; simple trace emitting `FieldRegEvent`). Do NOT yet implement `Flags` (depends on enum variants; landed in P2.1 already, so actually this is fine). *Compiles.*
3. **P2.3** Create `field_sll64.rs`, `field_sll128.rs`, `field_sll192.rs`. Each a thin copy of `field_mov.rs` with one `execute` body change. *Compiles.*
4. **P2.4** Create `field_assert_eq.rs`. *Compiles.*
5. **P2.5** Wire all five new structs into `tracer/src/instruction/mod.rs`: `Instruction` enum variants, `Instruction::decode` dispatch on `(funct7=0x40, funct3)`, `RISCVTrace` dispatch. *Compiles; test that each opcode decodes to the right variant.*

**Atomic-landing requirement inside P2.1:** removing `IsFMovI2F`/`IsFMovF2I` in the same commit that adds the five new flags means every `Flags` impl that set those flags (i.e., the v1 FMov instructions) must be updated in the same commit. The v1 FMov instructions themselves aren't deleted until Phase 1 of the migration (per `bn254-fr-coprocessor.md` — Phase 1 is "delete v1 complexity"). If Phase 1 has not yet landed when Phase 2 begins, P2.1 must either (a) wait on Phase 1, or (b) stub the v1 FMov `Flags` impls to return all-false for the removed flags. **Recommend (a):** land Phase 1 before starting Phase 2.

### Phase 3 (R1CS + PolynomialId)

Phase 3 is tightly coupled: adding the new virtual columns and removing the old witness columns must happen atomically, because the `NUM_R1CS_INPUTS` / `NUM_VARS_PER_CYCLE` constants change and downstream consumers (Spartan outer, witness generator, BlindFold constraint tables) snap to those constants.

1. **P3.1** (atomic) In `crates/jolt-compiler/src/polynomial_id.rs`: add `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` variants with `r1cs_variable_index()` = `Some(42/43/44)` and `source: PolySource::Virtual`. Remove all variants listed in §4 under "Removals". This lands as one commit.
2. **P3.2** (atomic with P3.1) In `crates/jolt-r1cs/src/constraints/rv64.rs`: replace the v1 slot constants (§2 "DROPPED") with the new five flag constants (`V_FLAG_IS_FIELD_ASSERT_EQ`, etc.) and virtual-column aliases (`V_FIELD_RS1_VALUE = 42`, etc.). Update `NUM_R1CS_INPUTS = 44`, `NUM_VARS_PER_CYCLE = 47`, `NUM_EQ_CONSTRAINTS = 29`, `NUM_CONSTRAINTS_PER_CYCLE = 32`.
3. **P3.3** (atomic with P3.1/P3.2) Rewrite the FR block of rows in `rv64_constraints()` per §3: keep rows 0..=18 verbatim; replace rows 19..=30 with the new rows 19..=31 (FieldAdd, FieldSub, FMUL×3, FINV×3, FieldAssertEq, FieldMov, FieldSLL64/128/192); product rows at 32..=34.
4. **P3.4** (depends on P3.1/P3.2/P3.3) In the R1CS Az/Bz/Cz materializer (Spartan): handle variable indices 42/43/44 by looking up `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` from the FR Twist output buffer instead of `z[idx]`. Implementation hook: wherever the current code does `z[var_idx]`, add a pre-check for `var_idx in {42, 43, 44}` and dispatch via `PolynomialId::FieldRs{1,2,R[d]}Value` evaluation. (Exact integration point depends on the Spartan implementation — out of scope for this doc.)
5. **P3.5** (depends on P3.1–P3.4) Delete `apply_field_op_events_to_r1cs` and `populate_limb_sum_columns` in `crates/jolt-host/src/r1cs_witness.rs` — v2 does not populate FR columns into the R1CS witness vector at all.
6. **P3.6** (depends on P3.5) Delete the old R1CS-side witness tests in `rv64.rs` that reference `V_FIELD_OP_A/B/RESULT`, `V_LIMB_SUM_A/B`, `V_FIELD_REG_READ_LIMB`, `V_FIELD_REG_WRITE_LIMB`. Add new R1CS-only tests for each new row (FieldAdd/Sub/Mul/Inv/AssertEq/Mov/SLL*) using a test helper that plugs virtual values directly into the witness at slots 42/43/44 (since R1CS unit tests bypass the FR Twist).

**Atomic-landing group for Phase 3:** P3.1 + P3.2 + P3.3 + P3.4 must land together. P3.5 + P3.6 can land in a follow-up commit.

### Cross-phase dependencies

- Phase 2 → Phase 3: the five new circuit flags from P2.1 are referenced by row guards in P3.3. If P3.3 lands first, those flag slot references will be dangling.
- Phase 3 → Phase 4 (FR Twist): the virtual `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` polynomials must be materialized by the FR Twist sumcheck output for the Spartan R1CS to be satisfiable. Until Phase 4 lands, end-to-end proving tests (`muldiv`) will fail — but R1CS unit tests that set these three values directly will pass. This is expected and documents the Phase 3 / Phase 4 boundary.

### Recommended commit sequence

```
feat(phase-1): rip out v1 FR limb infrastructure (per bn254-fr-coprocessor.md Phase 1)
feat(phase-2.1): add new CircuitFlags, remove FMov flags, bump NUM_CIRCUIT_FLAGS
feat(phase-2.2-5): add FieldMov/SLL64/SLL128/SLL192/AssertEq tracer structs + decode dispatch
feat(phase-3): R1CS v2 (new rows + virtual FR columns + slot constants + materializer hook)
refactor(phase-3): delete apply_field_op_events_to_r1cs + populate_limb_sum_columns
test(phase-3): R1CS v2 unit tests for FieldAdd/Sub/Mul/Inv/AssertEq/Mov/SLL*
```

After Phase 3, phase 4 (FR Twist mirror) + phase 5 (SDK) + phase 6 (e2e tests) proceed per `bn254-fr-coprocessor.md`.
