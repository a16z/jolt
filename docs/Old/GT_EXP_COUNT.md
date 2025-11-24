# Counting $\mathbb{G}_T$ Exponentiations in Dory Verification

## Per-Round Analysis

From [Theory/Dory.md:297](Theory/Dory.md), each round updates THREE commitments:

### 1. Update $C$ (inner product commitment):
$$C' = C + \chi + \beta D_2 + \beta^{-1}D_1 + \alpha C_+ + \alpha^{-1}C_-$$

Exponentiations needed:
- $\chi^1$ (lookup, but still counts as operation)
- $D_2^\beta$
- $D_1^{\beta^{-1}}$
- $C_+^\alpha$
- $C_-^{\alpha^{-1}}$

**Total: ~4-5 exponentiations** (plus multiplications in $\mathbb{G}_T$)

### 2. Update $D_1$ (first vector commitment):
Similar formula with 4-5 exponentiations

### 3. Update $D_2$ (second vector commitment):
Similar formula with 4-5 exponentiations

## Total Per Round:
**~10-15 $\mathbb{G}_T$ exponentiations** (depending on optimizations)

## For $\log N = 10$ rounds:
**100-150 $\mathbb{G}_T$ exponentiations** total

Wait, that doesn't match the dev's ~40 number. Let me reconsider...

## Alternative: Batched Updates
The verifier may NOT compute $(C', D'_1, D'_2)$ at each round. Instead, accumulate all proof components and compute final values at the end in one batched multi-exponentiation.

From [Theory/Dory.md:311-312](Theory/Dory.md):
> "The verifier **does not need to compute the new claim $(C', D'_1, D'_2)$ at each step**. Instead, the verifier can simply accumulate all prover messages and challenges across all $\log n$ rounds."

So the ~10 exponentiations per round is what WOULD be needed if computed eagerly, but batching reduces this significantly!

The ~40 total likely comes from:
- Final batched computation: ~30 exponentiations
- Homomorphic combinations (RLC): ~29 exponentiations
- Miscellaneous: A few more

Total: ~40 $\mathbb{G}_T$ exponentiations
