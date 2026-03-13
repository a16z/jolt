# Constantine Backend for jolt-field: Feasibility Analysis & Plan

## Executive Summary

[Constantine](https://github.com/mratsim/constantine) is a high-performance, zero-dependency cryptographic library written in Nim with C/Rust FFI bindings. It claims **2x faster MSM over arkworks**, **60% faster field arithmetic** with x86 assembly (MULX/ADCX/ADOX), and **constant-time** operations by default. This document evaluates using Constantine as a secondary BN254 Fr backend in `jolt-field`.

**Verdict: Feasible but high-friction, with uncertain payoff for field-level arithmetic.**

The primary benefit is in MSM and curve operations (jolt-crypto/jolt-dory territory), not bare field `mul`. For the sumcheck hot loop — which is dominated by `WideAccumulator` fmadd, not individual field muls — Constantine's assembly-optimized field mul may not provide meaningful improvement because Jolt already defers reduction. The build-system complexity (Nim compiler dependency) is the main practical barrier.

---

## 1. What Constantine Provides

### Field Arithmetic (BN254 Fr)

C API exposed via `include/constantine/curves/bn254_snarks.h`:

```c
typedef struct { secret_word limbs[CTT_WORDS_REQUIRED(254)]; } bn254_snarks_fr;

// Arithmetic
void ctt_bn254_snarks_fr_sum(bn254_snarks_fr* r, const bn254_snarks_fr* a, const bn254_snarks_fr* b);
void ctt_bn254_snarks_fr_diff(bn254_snarks_fr* r, ...);
void ctt_bn254_snarks_fr_prod(bn254_snarks_fr* r, ...);
void ctt_bn254_snarks_fr_square(bn254_snarks_fr* r, ...);
void ctt_bn254_snarks_fr_inv(bn254_snarks_fr* r, ...);
void ctt_bn254_snarks_fr_neg(bn254_snarks_fr* r, ...);
void ctt_bn254_snarks_fr_double(bn254_snarks_fr* r, ...);
void ctt_bn254_snarks_fr_div2(bn254_snarks_fr* r, ...);

// In-place variants
void ctt_bn254_snarks_fr_add_in_place(...);
void ctt_bn254_snarks_fr_sub_in_place(...);
void ctt_bn254_snarks_fr_mul_in_place(...);
void ctt_bn254_snarks_fr_square_in_place(...);
void ctt_bn254_snarks_fr_inv_in_place(...);

// Conditional (constant-time)
void ctt_bn254_snarks_fr_ccopy(...);
void ctt_bn254_snarks_fr_cswap(...);
void ctt_bn254_snarks_fr_cneg_in_place(...);

// Serialization
ctt_codec_scalar_status ctt_bn254_snarks_fr_unmarshalBE(bn254_snarks_fr* r, const byte src[], ptrdiff_t len);
ctt_codec_scalar_status ctt_bn254_snarks_fr_marshalBE(byte dst[], ptrdiff_t len, const bn254_snarks_fr* src);

// Comparison
ctt_bool ctt_bn254_snarks_fr_is_eq(...);
ctt_bool ctt_bn254_snarks_fr_is_zero(...);
ctt_bool ctt_bn254_snarks_fr_is_one(...);
```

### Curve Operations (G1/G2/GT, MSM, Pairing)

Constantine also exposes G1/G2 scalar multiplication, MSM (parallel + vartime), and pairing — these live in jolt-crypto/jolt-dory scope, not jolt-field.

### x86 Assembly

Constantine generates x86-64 assembly using MULX, ADCX, ADOX instructions for field multiplication. Benchmark claims:
- 60% speedup over pure C for 6-limb fields (BLS12-381)
- ~2.4x over plain GCC without inline asm
- BN254 is 4-limb, so the asm advantage is smaller (~30-40% estimated)

---

## 2. Integration Surface in jolt-field

### What Needs to Be Implemented

To add a Constantine-backed `Fr`, we need to implement these traits:

| Trait | Methods | Difficulty |
|-------|---------|------------|
| `Field` | 20+ methods (arithmetic, conversion, serialization) | Medium — straightforward FFI mapping |
| `FieldAccumulator` | `fmadd`, `merge`, `reduce` | **Hard — see below** |
| `UnreducedOps` | `mul_unreduced<L>`, `mul_u64_unreduced`, etc. | **Blocker — not exposed by Constantine** |
| `ReductionOps` | `from_montgomery_reduce<L>`, `from_barrett_reduce<L>` | **Blocker — not exposed by Constantine** |
| `WithChallenge` | Challenge type association | Easy |
| `OptimizedMul` | Blanket impl, free | Free |
| `serde::{Serialize, Deserialize}` | Byte serialization | Easy |
| All std::ops traits | Add/Sub/Mul/Div/Neg + Assign variants | Boilerplate FFI wrapping |

### Critical Blockers

#### 2a. WideAccumulator Requires Limb-Level Access

Jolt's `WideAccumulator` (the performance-critical accumulator) works by:
1. Accessing raw Montgomery-form limbs via `Fr::inner_limbs() -> Limbs<4>`
2. Performing 4x4 schoolbook multiply → 8 unreduced limbs
3. Adding unreduced product into a 9-limb accumulator
4. Reducing once at the end via Montgomery REDC

Constantine does **not** expose:
- Raw limb access to the Montgomery representation
- Unreduced multiplication (wide product without reduction)
- Caller-controlled Montgomery/Barrett reduction

Without these, we cannot build a `WideAccumulator` for Constantine-backed `Fr`. We'd be forced to use `NaiveAccumulator`, which reduces after every `fmadd` — this is the **dominant performance factor** in the sumcheck hot loop and would likely **negate any per-multiplication speedup** Constantine provides.

**Options:**
1. **Fork Constantine** and add limb-access + unreduced-mul C exports — high maintenance burden, defeats "use upstream" goal
2. **Use NaiveAccumulator** — easy to implement but likely slower overall than arkworks+WideAccumulator
3. **Propose upstream API** — Constantine is actively developed; `mr_unreduced_mul` and `mr_montgomery_reduce` exports could be proposed. No guarantee of acceptance or timeline.

#### 2b. Montgomery Form Compatibility

Constantine uses Montgomery representation internally but the exact constants (R, R^2) and limb ordering may differ from arkworks. If we ever need to convert between arkworks Fr and Constantine Fr (e.g., for Dory which uses arkworks curves), we'd need to go through canonical (non-Montgomery) bytes, adding serialization overhead at every boundary.

#### 2c. Build System: Nim Compiler Dependency

Constantine's `build.rs` requires:
- **Nim v2.2.0** installed on the build machine
- **nimble** package manager
- **clang** (hardcoded CC)
- Invokes `nimble make_lib_rust` which compiles Nim → C → static library

This is a significant ergonomic burden:
- Every developer, CI runner, and user building from source needs Nim
- Cross-compilation becomes harder (Nim → C for target triple)
- Adds ~10-30s to clean builds
- No crates.io published crate (must use git dependency)

---

## 3. Performance Analysis

### Where Constantine Wins

| Operation | Constantine | Arkworks | Delta |
|-----------|------------|----------|-------|
| Single Fr multiply | ~18ns (asm) | ~25ns | ~1.4x |
| G1 scalar mul | Best-in-class | 2.56x slower | 2.56x |
| G2 scalar mul | Best-in-class | 8.95x slower | 8.95x |
| MSM (large) | Best-in-class | ~2x slower | 2x |
| Pairing | ~0.65ms | ~1.07ms | 1.6x |

### Where It Doesn't Help (Jolt's Hot Path)

The sumcheck inner loop looks like:

```
for each output slot:           // ~millions of iterations
    for each product term:
        acc.fmadd(poly_a, poly_b)   // WideAccumulator: unreduced 4x4 mul + 9-limb add
    acc.reduce()                     // Montgomery REDC once per slot
```

With `WideAccumulator`:
- `fmadd` cost ≈ 25 cycles (schoolbook mul + wide add, no reduction)
- `reduce` cost ≈ 40 cycles (one REDC), amortized over ~100 fmadd calls
- Effective per-fmadd cost ≈ **25.4 cycles**

With `NaiveAccumulator` (what Constantine would use):
- Each `fmadd` does: FFI call + full field mul (~18ns ≈ 55 cycles on 3GHz) + field add
- Effective per-fmadd cost ≈ **55+ cycles** (plus FFI overhead)

**Net: Constantine with NaiveAccumulator would be ~2x slower than arkworks with WideAccumulator in the sumcheck hot loop.**

### Where Constantine *Would* Help

If Constantine exposed unreduced multiplication + limb access:
- We could build `ConstantineWideAccumulator` using Constantine's asm multiply
- The asm 4x4 schoolbook would be ~1.3x faster than Rust's → effective ~20 cycles/fmadd
- **~20% improvement** on the sumcheck inner loop — meaningful but not transformative

---

## 4. Alternative: Constantine for Curve Ops Only

A more practical integration path targets jolt-crypto and jolt-dory, where Constantine's **MSM and pairing** advantages are substantial:

| Component | Current Backend | Constantine Benefit |
|-----------|----------------|-------------------|
| Dory commitment (MSM) | arkworks G1/G2 | 2x MSM speedup |
| Dory pairing verification | arkworks BN254 | 1.6x pairing speedup |
| BlindFold Pedersen commits | arkworks G1 | 2.56x scalar mul |
| Sumcheck field arithmetic | arkworks Fr | Negligible (WideAccumulator dominates) |

This would:
- Keep `jolt-field::Fr` on arkworks (no WideAccumulator issue)
- Replace curve operations in `jolt-crypto` with Constantine FFI
- Require Constantine ↔ arkworks point/scalar conversion at boundaries
- Still require Nim build dependency

---

## 5. Implementation Plan (If Proceeding)

### Phase 1: Benchmark Validation (1-2 days)

Before any integration work, validate Constantine's actual performance advantage on our hardware:

1. Write a standalone benchmark crate that:
   - Links constantine-sys
   - Benchmarks BN254 Fr mul, add, square, inverse
   - Benchmarks BN254 G1 MSM at various sizes (2^10 through 2^20)
   - Compares against equivalent arkworks operations
2. Run on target hardware (M-series Mac + x86-64 CI)
3. **Decision gate**: Proceed only if field mul shows >1.3x improvement AND MSM shows >1.5x

### Phase 2: Feature-Gated Fr Backend (3-5 days)

Add `constantine` feature flag to `jolt-field`:

```toml
[features]
default = ["bn254"]
bn254 = ["ark-bn254", "ark-ff", ...]       # existing
constantine = ["constantine-sys"]            # new
```

Implementation:

```
crates/jolt-field/src/
  constantine/
    mod.rs          # ConstantineFr newtype
    ffi.rs          # Raw FFI bindings to constantine-sys
    ops.rs          # Optimized operation implementations
```

```rust
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ConstantineFr([u64; 4]);  // matches bn254_snarks_fr layout

impl Field for ConstantineFr {
    type Accumulator = NaiveAccumulator<Self>;  // initially
    // ... FFI-delegated implementations
}
```

Key design decisions:
- `ConstantineFr` is a separate type from `Fr` (not a runtime switch)
- Protocol code is generic over `F: Field`, so both work
- `NaiveAccumulator` initially; WideAccumulator if Constantine adds limb APIs
- Feature flags are mutually exclusive: `bn254` XOR `constantine`

### Phase 3: WideAccumulator Parity (Blocked on Upstream)

Propose to Constantine:
```c
// Desired API additions
void ctt_bn254_snarks_fr_mul_unreduced(uint64_t result[8],
    const bn254_snarks_fr* a, const bn254_snarks_fr* b);
void ctt_bn254_snarks_fr_get_limbs(uint64_t dst[4], const bn254_snarks_fr* src);
void ctt_bn254_snarks_fr_from_montgomery_reduce(bn254_snarks_fr* r,
    const uint64_t wide[9]);
```

If accepted upstream:
- Implement `ConstantineWideAccumulator` mirroring the arkworks version
- Expected ~20% sumcheck improvement from asm multiply

### Phase 4: Curve Operations in jolt-crypto (5-8 days)

Replace arkworks G1/G2/GT in `jolt-crypto/src/arkworks/bn254/`:
- G1 affine/projective → Constantine FFI
- G2 affine/projective → Constantine FFI
- MSM → `ctt_bn254_snarks_g1_prj_multi_scalar_mul_fr_coefs_vartime_parallel`
- Pairing → Constantine pairing API
- GLV decomposition → Constantine's built-in endomorphism

Conversion boundary:
```rust
impl From<ConstantineFr> for ArkFr { /* canonical bytes round-trip */ }
impl From<ArkFr> for ConstantineFr { /* canonical bytes round-trip */ }
```

### Phase 5: Native x86 Assembly for WideAccumulator (Alternative Path)

Instead of depending on Constantine for field-level speedups, write custom x86-64 assembly for the `WideAccumulator::fmadd` path:

```asm
; 4x4 schoolbook multiply using MULX/ADCX/ADOX
; Input: rdi = &mut acc (9 limbs), rsi = &a (4 limbs), rdx = &b (4 limbs)
jolt_fmadd_4x4:
    ; ... MULX-based 4x4 multiply + 9-limb accumulate
```

This is orthogonal to Constantine and gives us:
- Same ~30% field mul speedup from asm
- No Nim dependency
- Full control over the accumulation pattern
- Works with existing arkworks Fr (just faster schoolbook kernel)

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Nim build dependency breaks CI/dev workflow | High | Docker with Nim pre-installed; conditional feature |
| NaiveAccumulator negates field mul speedup | High | Don't ship without WideAccumulator parity |
| Montgomery form mismatch at conversion boundaries | Medium | Use canonical byte serialization; benchmark overhead |
| Constantine API changes break builds | Medium | Pin to specific commit; vendor if needed |
| Two Fr types create generic-parameter confusion | Medium | Feature flags make them mutually exclusive |
| Constantine lacks fuzz/audit history for Fr | Medium | Differential testing against arkworks |
| x86 asm only; no ARM (Apple Silicon) benefit | High | Constantine's ARM assembly is "planned" not shipped |

---

## 7. Recommendations

### Short-Term (High ROI, Low Risk)

1. **Custom x86 asm for WideAccumulator fmadd** (Phase 5) — gets the field arithmetic speedup without any external dependency. Estimated 20-30% sumcheck improvement on x86-64.

2. **Benchmark Constantine MSM vs arkworks MSM** (Phase 1) — if MSM is genuinely 2x faster, the Dory commitment stage benefits significantly.

### Medium-Term (High ROI, Medium Risk)

3. **Constantine for MSM/pairing only** (Phase 4) — keep Fr on arkworks, use Constantine for curve ops in jolt-dory. This targets the operations where Constantine's advantage is unambiguous.

### Long-Term (Uncertain ROI, High Risk)

4. **Full Constantine Fr backend** (Phases 2-3) — only worthwhile if Constantine adds unreduced-mul APIs upstream. Without WideAccumulator parity, this is a net performance regression despite faster individual multiplications.

---

## 8. Open Questions

1. **ARM assembly timeline**: Constantine currently only has x86-64 asm. Since the team develops on Apple Silicon, the asm advantage is absent on dev machines. When is ARM asm expected?

2. **Unreduced mul API appetite**: Has anyone proposed wide/unreduced multiplication exports to Constantine? Is mratsim receptive to such APIs?

3. **Cross-language LTO**: Constantine claims Nim→C→Rust LTO is possible. Has anyone validated this actually eliminates FFI overhead on the hot path?

4. **Nim stability**: Nim 2.2.0 is required. How stable is the Nim ecosystem for long-term dependency management?

5. **Halo2 ZAL experience**: Projects using `constantine-halo2-zal` — what's their experience with build reliability and performance?
