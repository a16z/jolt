# Justin's Concatenation Protocol for Virtual Claims

## The Key Idea

Instead of sending all virtual claims, use one round of interaction to reduce them to a single claim about a "claim polynomial".

## Current Protocol

```
Stage 1: p₀(r_x) = v₀, p₁(r_x) = v₁, ..., p_{n-1}(r_x) = v_{n-1}
         ↓
         Send all vᵢ to verifier (n field elements)
         ↓
Stage 2: Prove Σₛ eq(r_s, s) · M(s, r_x) = Σᵢ eq(r_s, i) · vᵢ
```

## Justin's Protocol

```
Stage 1: p₀(r_x) = v₀, p₁(r_x) = v₁, ..., p_{n-1}(r_x) = v_{n-1}
         ↓
         Define "claim polynomial" C where C(i) = vᵢ for i ∈ {0,1,...,n-1}
         ↓
         Verifier sends random challenge α ∈ F
         ↓
         Prover sends C(α) and proves it's correct via sumcheck
         ↓
Stage 2: Modified to use C(α) instead of individual claims
```

## The Concatenation Sumcheck

The claim polynomial C is defined by its evaluations on the boolean hypercube:
- C(0) = v₀ = p₀(r_x)
- C(1) = v₁ = p₁(r_x)
- ...
- C(n-1) = v_{n-1} = p_{n-1}(r_x)

To prove C(α) is correct, we run a sumcheck proving:
```
C(α) = Σᵢ∈{0,1}^{log n} eq(α, i) · pᵢ(r_x)
```

But we can rewrite this using the concatenated polynomial P(i, x) = pᵢ(x):
```
C(α) = Σᵢ∈{0,1}^{log n} eq(α, i) · P(i, r_x)
```

## Protocol Details

1. **After Stage 1**: Prover has values v₀, ..., v_{n-1} where vᵢ = pᵢ(r_x)

2. **Challenge**: Verifier sends α ← F^{log n}

3. **Concatenation Sumcheck**:
   - Claim: `C(α) = c*` (prover sends c*)
   - Prove: `Σᵢ∈{0,1}^{log n} eq(α, i) · P(i, r_x) = c*`
   - This is a sumcheck over the i variables
   - log n rounds, degree 2 (from eq polynomial)
   - Final claim: `P(β, r_x) = v**` for some β

4. **Connection to Stage 2**:
   - Original Stage 2 would verify matrix claim at (r_s, r_x)
   - Now it verifies matrix claim at (β, r_x)
   - The virtual claims are "baked into" the sumcheck

## Why This Reduces Proof Size

- **Without concatenation**: Send n field elements (v₀, ..., v_{n-1})
- **With concatenation**: Send 1 field element (c*) + log n sumcheck rounds

For n = 1024:
- Before: 1024 field elements
- After: 1 field element + 10 rounds × 3 elements/round = 31 elements
- Reduction: ~33×

## The Subtle Point

The key insight is that we're introducing a **new polynomial** C whose evaluations are the virtual claims themselves. This allows us to:
1. Reduce all claims to a single evaluation C(α)
2. Prove this evaluation is correct via sumcheck
3. Connect to the existing matrix structure in Stage 2

## Integration with Current System

This would require:
1. Adding the concatenation sumcheck between Stage 1 and Stage 2
2. Modifying Stage 2 to work with the challenge point α instead of r_s
3. Adjusting how virtual claims flow through the system

The benefit is significant proof size reduction, but at the cost of additional protocol complexity.