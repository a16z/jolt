# Advice Claim Reduction + Dory Batching Explained

This document provides visual explanations of how the two-phase advice claim reduction works and how advice polynomials are batched into the single Stage 8 Dory opening proof.

---

## Summary: What Changed from `better-opening-reduction`

The current branch (`cursor/advice-claim-integration-and-sumcheck-b852`) introduces several key additions not present in `better-opening-reduction`:

### 1. New Two-Phase Advice Claim Reduction (`advice.rs` - entirely new file)
- `better-opening-reduction` had **separate** advice opening proofs (outside the batched pipeline)
- Current branch **folds advice into Stage 6/7 batched sumchecks** and **Stage 8 single Dory opening**

### 2. Batched Sumcheck Enhancement (`sumcheck.rs`)
- Added `round_offset()` support for non-suffix placement of instances
- Added `dummy_after` scaling for instances that have trailing dummy rounds

### 3. Preprocessing-Only Advice Commitment (`dory_globals.rs`, `prover.rs`)
- New `initialize_trusted_advice_matrix()` and `initialize_untrusted_advice_matrix()` in separate Dory contexts
- Advice committed with balanced `(nu_a, sigma_a)` dims independent of trace length

### 4. Stage 7 Embedding Selector (`prover.rs`)
- Computes `row_factor * col_prefix_factor` to scale advice claims
- Makes advice openings compatible with the unified main matrix point

### 5. RLC Streaming with Advice (`rlc_polynomial.rs`)
- Added `add_streaming_advice_vmv()` to include advice in the VMV computation
- Treats advice as a top-left block embedded in the main matrix

---

## Diagram 1: The Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING (TRUSTED ADVICE)                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ commit_trusted_advice() in separate Dory context (TrustedAdvice)        │    │
│  │   • balanced shape: sigma_a = ⌈advice_vars/2⌉, nu_a = advice_vars-sigma_a│   │
│  │   • produces: commitment + opening_hint                                  │    │
│  │   • NO dependency on trace length T                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RUNTIME PROVING                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────┐      ┌────────────────┐      ┌────────────────────────────────┐ │
│  │  Stages    │      │    Stage 6     │      │           Stage 7              │ │
│  │   1–5      │ ──▶  │  (cycle vars)  │ ──▶  │        (address vars)          │ │
│  │            │      │                │      │                                │ │
│  │  (various  │      │ • Booleanity   │      │ • HammingWeightClaimReduction  │ │
│  │ sumchecks) │      │ • IncReduction │      │ • AdvicePhase2 (if nu_a_addr>0)│ │
│  │            │      │ • AdvicePhase1 │◀─┐   │                                │ │
│  └────────────┘      └────────────────┘  │   └────────────────────────────────┘ │
│                              │           │                    │                  │
│                              │           │                    │                  │
│                              ▼           │                    ▼                  │
│                      ┌───────────────────┴─────┐    ┌────────────────────────┐  │
│                      │ intermediate claim C_mid│    │ final advice opening   │  │
│                      │ (if nu_a_addr > 0)      │───▶│ AdviceMLE(advice_point)│  │
│                      └─────────────────────────┘    └────────────────────────┘  │
│                                                              │                   │
│                                                              ▼                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                           STAGE 8 (Dory Opening)                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ DoryOpeningState:                                                   │  │  │
│  │  │   • opening_point = [addr7_be || cycle6_be]                         │  │  │
│  │  │   • claims = [scaled_advice_claim, main_witness_claims...]          │  │  │
│  │  │   • polynomials = [TrustedAdvice, UntrustedAdvice, WitnessPolys...] │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                    │                                       │  │
│  │                                    ▼                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ build_streaming_rlc():                                              │  │  │
│  │  │   • stream main witness from trace                                  │  │  │
│  │  │   • add advice as top-left block via add_streaming_advice_vmv()     │  │  │
│  │  │   • combine with gamma powers: RLC = Σ γⁱ · poly_i                  │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                    │                                       │  │
│  │                                    ▼                                       │  │
│  │                         SINGLE Dory Opening Proof                          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: The Dory Matrix Layout (Main vs Advice)

```
                                    MAIN DORY MATRIX
                               (2^nu_main × 2^sigma_main)
     ◀────────────────────── 2^sigma_main columns ──────────────────────▶
    ┌──────────────────────────────────────────────────────────────────────┐  ▲
    │ ┌───────────────────┐                                                │  │
    │ │                   │                                                │  │
    │ │  ADVICE BLOCK     │        (zeros outside advice block)            │  │
    │ │  2^nu_a × 2^sigma_a                                                │  │  2^nu_main
    │ │                   │                                                │  │
    │ │  Top-left corner  │                                                │  │    rows
    │ └───────────────────┘                                                │  │
    │                                                                      │  │
    │                        MAIN WITNESS POLYNOMIALS                      │  │
    │                   (fill entire 2^nu_main × 2^sigma_main)             │  │
    │                                                                      │  │
    └──────────────────────────────────────────────────────────────────────┘  ▼

    KEY INSIGHT:
    • Main witness occupies the FULL matrix
    • Advice is EMBEDDED as the top-left block
    • The embedding selector (row_factor × col_factor) "picks out" only the top-left
```

---

## Diagram 3: Variable Ordering (The "Magic" Behind Alignment)

```
                     STAGE 6                              STAGE 7
            ◀────── cycle coords ──────▶      ◀───── address coords ─────▶
            
  r6[...] = [...  |  c₀ c₁ c₂ ... c_{log_t-1}]    r7[...] = [b₀ b₁ ... b_{log_k-1}]
                    └───────┬───────────────┘              └─────────┬─────────────┘
                            │                                        │
                            ▼                                        ▼
        ┌─────────────────────────────────────────────────────────────────────────┐
        │           point_dory (Dory's internal LE ordering)                      │
        │                                                                         │
        │   [c₀ c₁ ... c_{log_t-1} | b₀ b₁ ... b_{log_k-1}]                       │
        │    ◀──── cycle₆_le ────▶   ◀───── addr₇_le ─────▶                       │
        └─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────────────────────────┐
        │                 Dory row/column split                      │
        │                                                            │
        │   col_coords = point_dory[0 .. σ_main]                     │
        │   row_coords = point_dory[σ_main .. σ_main + ν_main]       │
        │                                                            │
        │   With σ_main = ⌈(log_k + log_t)/2⌉:                       │
        │                                                            │
        │   col_coords = [ c₀ ... c_{σ-1} ]         (low cycle bits) │
        │   row_coords = [ c_σ...c_{log_t-1} | b₀...b_{log_k-1} ]    │
        │                └─ cycle tail ─┘   └──── address ─────┘     │
        └────────────────────────────────────────────────────────────┘
```

---

## Diagram 4: Two-Phase Advice Claim Reduction (The Core Innovation)

```
                           ADVICE POINT (in Dory order, LE)
                           
    advice_point_le = [ advice_cols  ||  advice_rows  ]
                      [0 .. σ_a]       [0 .. ν_a]
                           │                 │
                           │                 └──── may straddle cycle/addr!
                           │                       (that's why we need 2 phases)
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │   advice_cols = col_coords[0..σ_a] = cycle₆_le[0..σ_a]                   │
    │                 ◀──────── always from Stage 6 ────────▶                  │
    │                                                                          │
    │   advice_rows = row_coords[0..ν_a]                                       │
    │               = [ cycle₆_le[σ_main .. σ_main+ν_a_cycle] ]  ← from Stage 6│
    │               + [ addr₇_le[0 .. ν_a_addr]              ]  ← from Stage 7 │
    │                                                                          │
    │   where: ν_a = ν_a_cycle + ν_a_addr                                      │
    └──────────────────────────────────────────────────────────────────────────┘


    ═══════════════════════════════════════════════════════════════════════════
                        PHASE 1 (Stage 6 Batched Sumcheck)
    ═══════════════════════════════════════════════════════════════════════════

    Input:  RAM subprotocol advice openings at r_val_eval, r_val_final
    Output: Intermediate claim C_mid

    Local rounds:  [0..σ_a) ∪ [σ_a..σ_main) ∪ [σ_main..σ_main+ν_a_cycle)
                   │           │               │
                   │           │               └─ row cycle bits (bind advice)
                   │           │
                   │           └─ DUMMY GAP (traverse but don't bind advice)
                   │              → constant univariates, scale by 2^{-gap}
                   │
                   └─ column bits (bind advice)

    round_offset = booleanity_offset + log_k_chunk  (start at cycle segment)

    ┌────────────────────────────────────────────────────────────────────────┐
    │  round 0        round σ_a      round σ_main         round σ_main+ν_a_cycle
    │    │               │               │                        │
    │    ▼               ▼               ▼                        ▼
    │  ┌─────────────┬───────────────┬─────────────────────────────┐
    │  │  BIND cols  │  DUMMY GAP    │      BIND row_cycle         │
    │  │  (σ_a vars) │  (σ_main-σ_a) │      (ν_a_cycle vars)       │
    │  │             │  scale: ×2^-1 │                             │
    │  │             │  per round    │                             │
    │  └─────────────┴───────────────┴─────────────────────────────┘
    │                                                              │
    │                                                              ▼
    │                                               C_mid = advice(x) · eq(x)
    │                                               where x = bound vars so far
    └────────────────────────────────────────────────────────────────────────┘


    ═══════════════════════════════════════════════════════════════════════════
                        PHASE 2 (Stage 7 Batched Sumcheck)
    ═══════════════════════════════════════════════════════════════════════════
    
    (Only runs if ν_a_addr > 0, i.e., advice rows spill into address bits)

    Input:  C_mid from Phase 1
    Output: Final advice opening: AdviceMLE(advice_point)

    Pre-bind: cycle-derived coords (σ_a col bits + ν_a_cycle row bits)
    Bind:     addr-derived coords (first ν_a_addr bits of addr₇_le)

    round_offset = 0  (start at beginning of Stage 7)

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  round 0                                     round ν_a_addr            │
    │    │                                              │                    │
    │    ▼                                              ▼                    │
    │  ┌────────────────────────────────────────────────┐                    │
    │  │      BIND address-derived row bits             │                    │
    │  │      (ν_a_addr vars from addr₇_le[0..])        │                    │
    │  └────────────────────────────────────────────────┘                    │
    │                                                   │                    │
    │                                                   ▼                    │
    │                              advice_claim = AdviceMLE(full_advice_point)
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 5: Stage 7 Embedding Selector (Scaling the Advice Claim)

```
                       THE PROBLEM
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  advice_claim = AdviceMLE(advice_point)                         │
    │                 ◀─── evaluated at σ_a + ν_a vars ───▶          │
    │                                                                 │
    │  But Stage 8 opens at the FULL main point:                      │
    │  main_point = [addr₇_be || cycle₆_be]                           │
    │               ◀─── σ_main + ν_main vars ───▶                   │
    │                                                                 │
    │  We can't just claim AdviceMLE(main_point) because advice       │
    │  polynomial only has σ_a + ν_a variables!                       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘


                       THE SOLUTION: EMBEDDING SELECTOR
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  View advice as EMBEDDED in the main matrix's top-left block.   │
    │  Define EmbeddedAdvice(main_point) as:                          │
    │                                                                 │
    │    EmbeddedAdvice(r) = row_factor · col_factor · AdviceMLE(...)│
    │                                                                 │
    │  where:                                                         │
    │                                                                 │
    │    row_factor = ∏_{i=ν_a}^{ν_main-1} (1 - r_rows[i])           │
    │                 └─ "high row bits must be 0"                    │
    │                                                                 │
    │    col_factor = ∏_{i=σ_a}^{σ_main-1} (1 - r_cols[i])           │
    │                 └─ "high col bits must be 0"                    │
    │                                                                 │
    │  This is a polynomial identity! The selector forces evaluation  │
    │  to zero unless we're in the top-left block.                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘


                       STAGE 7 APPLIES THE SELECTOR
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  scaled_advice_claim = advice_claim × row_factor × col_factor   │
    │                                                                 │
    │  This scaled claim is added to DoryOpeningState.claims[]        │
    │  alongside all the main witness claims.                         │
    │                                                                 │
    │  Stage 8's RLC batching then produces:                          │
    │                                                                 │
    │    joint_claim = Σ γⁱ · claim_i                                 │
    │                = γ^advice · scaled_advice_claim                 │
    │                + γ^witness₁ · witness₁_claim                    │
    │                + ...                                            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Diagram 6: Stage 8 RLC Streaming (How Advice Gets Into the VMV)

```
                    STREAMING VMV COMPUTATION
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Dory opening proof = VMV (Vector-Matrix-Vector product)             │
    │                                                                      │
    │  Standard approach:                                                  │
    │    result[col] = Σ_{row} left_vec[row] × matrix[row, col]            │
    │                                                                      │
    │  For RLC polynomial, matrix[row,col] = Σ γⁱ · poly_i[row·num_cols+col]
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

                    MAIN WITNESS: Streamed from Trace
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  for each cycle in trace:                                            │
    │      compute witness polynomial values                               │
    │      accumulate into result[col] += left_vec[row] × poly_val         │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

                    ADVICE: Added as Top-Left Block
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  fn add_streaming_advice_vmv(result, left_vec, advice_poly):         │
    │                                                                      │
    │      advice_rows = 2^ν_a                                             │
    │      advice_cols = 2^σ_a                                             │
    │                                                                      │
    │      for row_idx in 0..advice_rows:                                  │
    │          for col_idx in 0..advice_cols:                              │
    │              coeff_idx = row_idx × advice_cols + col_idx             │
    │              result[col_idx] += left_vec[row_idx]                    │
    │                               × γ^advice                             │
    │                               × advice_poly[coeff_idx]               │
    │                                                                      │
    │  NOTE: result[col] for col ≥ advice_cols is NOT touched by advice!   │
    │        result[row] contribution only from row < advice_rows!         │
    │                                                                      │
    │  This is exactly what the embedding selector achieves algebraically. │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 7: Complete Example (log_k=4, log_t=16, σ_main=10, ν_main=10)

```
    SETUP
    ═════
    Main matrix:    2^10 rows × 2^10 cols (total_vars=20)
    Advice:         1024 bytes = 128 words → advice_len = 128 = 2^7
                    advice_vars = 7
                    σ_a = 4, ν_a = 3

    Advice matrix:  2^3 rows × 2^4 cols (8 × 16 = 128 coefficients)


    DORY POINT LAYOUT
    ══════════════════
    
    point_dory = [.... c₀ c₁ ... c₁₅ | b₀ b₁ b₂ b₃]
                  .... ◀── 16 cycle ──▶ ◀─ 4 addr ─▶

    col_coords = point_dory[0..10]  = [c₀ c₁ c₂ c₃ c₄ c₅ c₆ c₇ c₈ c₉]
    row_coords = point_dory[10..20] = [c₁₀ c₁₁ c₁₂ c₁₃ c₁₄ c₁₅ | b₀ b₁ b₂ b₃]
                                       ◀── row_cycle ──────────▶ ◀─ row_addr ─▶


    ADVICE POINT
    ═════════════
    advice_cols = col_coords[0..4] = [c₀ c₁ c₂ c₃]           ← from Stage 6
    advice_rows = row_coords[0..3] = [c₁₀ c₁₁ c₁₂]           ← from Stage 6

    ν_a_cycle = min(3, 6) = 3   (all row bits from cycle)
    ν_a_addr  = 3 - 3 = 0       (no address-derived row bits)

    → Phase 2 NOT needed (all advice coords come from Stage 6)


    PHASE 1 LOCAL ROUNDS
    ═════════════════════
    Round:     0  1  2  3     4  5  6  7  8  9      10 11 12
               ◀─ σ_a=4 ─▶    ◀── gap=6 ──────▶    ◀─ν_a_cycle=3─▶

    Bind:      c₀ c₁ c₂ c₃   (dummy)              c₁₀ c₁₁ c₁₂
               ◀─ col bits─▶  ◀─ traverse ─▶      ◀─ row bits ─▶

    round_offset = (max_rounds - 20) + 4 = start at cycle segment
    
    Scale: accumulates 2^{-6} from the 6 dummy-gap rounds


    EMBEDDING SELECTOR (computed in Stage 7)
    ══════════════════════════════════════════
    row_factor = (1-c₁₃)(1-c₁₄)(1-c₁₅)(1-b₀)(1-b₁)(1-b₂)(1-b₃)  ← 7 terms
                 (ν_main - ν_a = 10 - 3 = 7 terms)

    col_factor = (1-c₄)(1-c₅)(1-c₆)(1-c₇)(1-c₈)(1-c₉)           ← 6 terms
                 (σ_main - σ_a = 10 - 4 = 6 terms)

    scaled_advice_claim = advice_claim × row_factor × col_factor
```

---

## Why It "Feels Like Magic" But Actually Works

### 1. Variable Alignment
Phase 1 is carefully scheduled via `round_offset()` to consume exactly the low cycle bits that map to Dory column coords + the cycle-derived row coords.

### 2. Dummy-Gap Traversal
When `σ_a < σ_main`, Phase 1 must "skip over" intermediate cycle bits with constant univariates, maintaining a `2^{-gap}` scaling factor.

### 3. Two-Phase Split
If advice row bits spill into the address segment, Phase 2 resumes in Stage 7 to bind those remaining coords.

### 4. Embedding Selector
The `row_factor × col_factor` product is the algebraic equivalent of treating advice as zero outside its top-left block. This is what makes the different-dimension advice claim compatible with the full main-point opening.

### 5. Streaming Consistency
The `add_streaming_advice_vmv()` function adds advice contributions only to the top-left block of the VMV, exactly matching the algebraic embedding.

---

## Code Pointers

| Component | File | Key Functions |
|-----------|------|---------------|
| Two-phase reduction | `zkvm/claim_reductions/advice.rs` | `AdviceClaimReductionPhase1Prover`, `AdviceClaimReductionPhase2Prover` |
| Batched sumcheck | `subprotocols/sumcheck.rs` | `round_offset()`, dummy_after scaling |
| Advice contexts | `poly/commitment/dory/dory_globals.rs` | `initialize_trusted_advice_matrix()` |
| Embedding selector | `zkvm/prover.rs` | Stage 7's `row_factor * col_prefix_factor` |
| RLC streaming | `poly/rlc_polynomial.rs` | `add_streaming_advice_vmv()` |
| Prover wiring | `zkvm/prover.rs` | `prove_stage6()`, `prove_stage7()`, `prove_stage8()` |

