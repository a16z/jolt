# Stage 6 Booleanity Sumcheck — Mathematical Specification

## The Claim

```
0 = Σ_{k,j} eq(r_addr, k) · eq(r_cycle, j) · Σ_d γ^{2d} · (ra_d(k,j)² - ra_d(k,j))
```

Where:
- k ∈ [0, K_chunk), K_chunk = 2^log_k_chunk (address chunk index)
- j ∈ [0, T), T = 2^log_t (cycle index)
- d ∈ [0, total_d) across instruction (32), bytecode (4), RAM (4) RA polynomials
- ra_d(k,j) ∈ {0,1} is the d-th one-hot RA value at address k, cycle j
- γ is a Fiat-Shamir challenge; γ^{2d} is the per-polynomial weight

## Two-Phase Sumcheck

### Phase 1: Address (log_k_chunk rounds)

Define G_d(k) = Σ_j eq(r_cycle, j) · ra_d(k, j).

Phase 1 proves:
```
0 = Σ_k eq(r_addr, k) · Σ_d γ^{2d} · G_d(k) · (G_d(k) - 1)
```

Round polynomial at round m (0-indexed), binding variable x_m:
```
s_m(X) = Σ_{k_rest} eq_addr(X, k_rest) · Σ_d γ^{2d} · G_d(X, k_rest) · (G_d(X, k_rest) - 1)
```

Evaluated at X = 0, 1, 2, 3 to get degree-3 UniPoly.

### Phase 1 → Phase 2 Transition

After log_k_chunk rounds of address binding with challenges r'_addr:
- eq_r_r = eq(r_addr, r'_addr) — scalar from split-eq
- H_d(j) = γ^d · eq(r'_addr, k_d(j)) · ra_d(k_d(j), j)
  where k_d(j) is the RA index at cycle j for dimension d
- H is pre-scaled by γ^d (prover gamma_powers[d], NOT γ^{2d})

### Phase 2: Cycle (log_t rounds)

Phase 2 proves (with eq_r_r factored out):
```
claim / eq_r_r = Σ_j eq(r_cycle, j) · Σ_d H_d(j) · (H_d(j) - γ^d)
```

Note: H_d is pre-scaled by γ^d, so H_d·(H_d - γ^d) = γ^d · val · (γ^d · val - γ^d) = γ^{2d} · val · (val - 1).

The round polynomial is multiplied back by eq_r_r before appending to transcript.

## Convention Choices (Core)

### r_address and r_cycle Derivation
- Source: stage 5 opening point for InstructionRa(0), stored in BIG_ENDIAN
- stage5_point.r[..log_k_instruction] → reverse → stage5_addr (LE)
- stage5_point.r[log_k_instruction..] → reverse → r_cycle (LE)
- r_address = stage5_addr[len - log_k_chunk..] (last log_k_chunk of LE = highest address bits)
- **Both r_address and r_cycle are LE** (r[0] = first variable to bind in LowToHigh)

### EqPolynomial::evals Convention
- Takes point in "BE-like" convention: point[0] maps to MSB of index
- BUT core passes LE points directly, so effectively: point_LE[0] → MSB of eq table index
- This means index bit 0 (LSB, first to bind in LowToHigh) corresponds to point[n-1]

### GruenSplitEqPolynomial
- Takes same point as EqPolynomial::evals
- `merge()` produces identical table to `EqPolynomial::evals(point)` (verified by test at split_eq_poly.rs:660)
- For LowToHigh: binds from current_index downward (processes point[n-1] first, then point[n-2], etc.)

### G_d Computation (compute_all_G_impl)
- r_cycle passed as-is (LE) to EqPolynomial::evals for the split-eq tables
- Data indexed by cycle: j = c_hi * 2^lo_bits + c_lo
- G_d[k] = Σ_j eq_table[j] · (1 if ra_d index at cycle j == k, else 0)

### RA Polynomial Data Layout
- Provider returns data in AddressMajor: data[addr * T + cycle]
- For multilinear evaluation: address bits are MSB, cycle bits are LSB

### Gamma Powers
- gamma_powers_square[d] = γ^{2d} (used in Phase 1 formula and output claim)
- gamma_powers[d] = γ^d (used for H pre-scaling in Phase 2)

## Transcript Operations Per Round

For each sumcheck round:
1. Each instance computes UniPoly (degree 3 → 4 evals)
2. Batch: combined = Σ_i batch_coeff_i · poly_i
3. Compress: store coeffs except linear term (3 field elements for degree 3)
4. Append: transcript.append_scalars("sumcheck_poly", &compressed.coeffs_except_linear_term)
5. Squeeze: r_j = transcript.challenge_scalar_optimized()

Before stage 6 sumcheck:
1. Each instance appends input_claim (booleanity: 0)
2. Squeeze batch_coeffs (one per instance)

## Opening Point Normalization

After all rounds, sumcheck challenges [c_0, ..., c_{n-1}] (LE) are split:
- Address: c[0..log_k_chunk] → reverse → BE
- Cycle: c[log_k_chunk..] → reverse → BE

Combined opening point in BE: [addr_BE, cycle_BE]
