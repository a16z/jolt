# GT Exponentiation Packed MLE Optimization

## Problem Statement

The current GT exponentiation arithmetization produces **O(n) polynomial claims per GT exp**, where n is the number of scalar bits (~254 for BN254). This is inefficient:

- **Current**: 4 polynomials × 254 steps = **~1,016 claims per GT exp**
- Each claim requires virtualization, batching, and PCS opening work

## Current Arithmetization

For `result = base^scalar`, we have n=254 constraints (one per scalar bit):

```
C_i(x) = ρ_{i+1}(x) - ρ_i(x)² × base(x)^{b_i} - Q_i(x) × g(x) = 0
```

**Polynomials (each 4-var, padded to 8-var = 256 coefficients):**
- `base_i(x)` - the base element (same for all i, but stored n times!)
- `rho_prev_i(x)` - intermediate result ρ_i
- `rho_curr_i(x)` - next intermediate ρ_{i+1}
- `quotient_i(x)` - quotient polynomial Q_i

**Issues:**
1. `base` is duplicated 254 times
2. `rho_curr_i` = `rho_prev_{i+1}` (same polynomial, stored twice)
3. 1,016 separate virtual polynomial claims

## Proposed Optimization: Packed MLEs

Pack all steps into larger polynomials by adding step-index variables.

### Polynomial Structure

| Polynomial | Variables | Size | Description |
|------------|-----------|------|-------------|
| `rho(s, x)` | 8 + 4 = 12 vars | 4,096 | All ρ values: `rho(i, x) = ρ_i(x)` |
| `rho_next(s, x)` | 8 + 4 = 12 vars | 4,096 | Shifted: `rho_next(i, x) = ρ_{i+1}(x)` |
| `quotient(s, x)` | 8 + 4 = 12 vars | 4,096 | All quotients: `quotient(i, x) = Q_i(x)` |
| `base(x)` | 4 vars | 16 | Single base element (not duplicated) |
| `bit(s)` | 8 vars | 256 | Scalar bits: `bit(i) = b_i` |

Where:
- `s` = 8 variables indexing the step (2^8 = 256 ≥ 254 steps)
- `x` = 4 variables for the Fq12 element representation

### Constraint

The batched constraint over all steps and element indices:

```
Σ_{s ∈ {0,1}^8} Σ_{x ∈ {0,1}^4} eq(r_s, s) × eq(r_x, x) × C(s, x) = 0
```

Where:
```
C(s, x) = rho_next(s, x) - rho(s, x)² × base(x)^{bit(s)} - quotient(s, x) × g(x)
```

### The Shift Relationship

The key insight is that `rho_next` is just `rho` with the step index shifted by 1:

```
rho_next(i, x) = rho(i+1, x)   for i ∈ [0, 253]
```

We store `rho_next` as a **separate polynomial** (not derived on-the-fly). The witness generator populates it with the shifted values. This is the same pattern used in G1 scalar multiplication with `XANext`/`YANext`.

**Boundary conditions:**
- `rho(0, x)` = initial value (identity element or starting ρ)
- `rho(254, x)` = final result
- `rho_next(253, x)` = `rho(254, x)` = final result
- Padding: `rho_next(254..255, x)` = 0 (or constrained separately)

---

## Two-Phase Sumcheck Protocol

The 12-variable sumcheck is split into two phases:

- **Phase 1**: 8 rounds over step variables `s`
- **Phase 2**: 4 rounds over element variables `x`

### Phase 1: Step Variable Sumcheck (8 rounds)

**Goal**: Reduce the sum over step indices `s ∈ {0,1}^8`

**Input claim**:
```
claim_0 = Σ_{s,x} eq(r_s, s) × eq(r_x, x) × C(s, x) = 0
```

**Prover state for Phase 1**:
```rust
struct Phase1State<F> {
    // 12-var packed polynomials (4096 coefficients each)
    rho_poly: MultilinearPolynomial<F>,
    rho_next_poly: MultilinearPolynomial<F>,
    quotient_poly: MultilinearPolynomial<F>,

    // 8-var bit polynomial (256 coefficients)
    bit_poly: MultilinearPolynomial<F>,

    // eq polynomials for batching challenges
    eq_s_poly: MultilinearPolynomial<F>,  // eq(r_s, s) over 8 vars
    eq_x_poly: MultilinearPolynomial<F>,  // eq(r_x, x) over 4 vars

    // Challenges accumulated so far
    sumcheck_challenges: Vec<F::Challenge>,
}
```

**Round computation** (for round j in Phase 1):

In each round, we sum over the "inner" element variables `x` while producing a univariate in the current step variable:

```rust
fn compute_message_phase1(&self, previous_claim: F) -> UniPoly<F> {
    const DEGREE: usize = 4;
    let half = self.rho_poly.len() / 2;  // Half of remaining hypercube

    let evals = (0..half).into_par_iter()
        .map(|i| {
            // For each remaining (s, x) pair, compute constraint contribution
            let eq_s = self.eq_s_poly.sumcheck_evals_array::<DEGREE>(i);
            let rho = self.rho_poly.sumcheck_evals_array::<DEGREE>(i);
            let rho_next = self.rho_next_poly.sumcheck_evals_array::<DEGREE>(i);
            let quotient = self.quotient_poly.sumcheck_evals_array::<DEGREE>(i);
            let bit = self.bit_poly.sumcheck_evals_array::<DEGREE>(i / 16);  // bit only depends on s
            let eq_x = self.eq_x_poly[i % 16];  // eq_x is constant during Phase 1

            // Sum over inner x dimension (implicitly folded into loop structure)
            // C(s,x) = rho_next - rho² × base^bit - quotient × g
            // Accumulate: eq_s × eq_x × C
            // ...
        })
        .reduce(...);

    UniPoly::from_evals_and_hint(previous_claim, &evals)
}
```

**Binding**: After each round, bind all polynomials at challenge `r_j`:
```rust
fn bind_phase1(&mut self, r_j: F::Challenge) {
    self.rho_poly.bind(r_j);
    self.rho_next_poly.bind(r_j);
    self.quotient_poly.bind(r_j);
    self.bit_poly.bind(r_j);
    self.eq_s_poly.bind(r_j);

    self.sumcheck_challenges.push(r_j);
}
```

**Transition condition**: After 8 rounds, step variables are fully bound → transition to Phase 2.

### Phase 2: Element Variable Sumcheck (4 rounds)

**Goal**: Reduce the sum over element indices `x ∈ {0,1}^4`

At the start of Phase 2, the 12-var polynomials have been bound to 4-var polynomials (one value per `x`), and step challenges `r_s* = (r_0, ..., r_7)` have been collected.

After Phase 1 binding, the polynomials are already reduced:
- `rho_poly` is now 4-var (16 coefficients)
- `rho_next_poly` is now 4-var (16 coefficients)
- `quotient_poly` is now 4-var (16 coefficients)
- `bit_poly` is now a scalar `bit_eval` (fully bound after 8 rounds)

**Prover state for Phase 2**:
```rust
struct Phase2State<F> {
    // Reduced 4-var polynomials (16 coefficients each)
    rho_poly: MultilinearPolynomial<F>,
    rho_next_poly: MultilinearPolynomial<F>,
    quotient_poly: MultilinearPolynomial<F>,

    // bit(s) evaluated at r_s* → scalar
    bit_eval: F,

    // eq(r_x, x) polynomial
    eq_x_poly: MultilinearPolynomial<F>,

    // base(x) is always 4-var, no reduction needed
    base_poly: MultilinearPolynomial<F>,

    // g(x) constraint polynomial
    g_poly: MultilinearPolynomial<F>,
}
```

**Round computation** (for round j in Phase 2):
```rust
fn compute_message_phase2(&self, previous_claim: F) -> UniPoly<F> {
    const DEGREE: usize = 4;
    let half = self.rho_poly.len() / 2;

    let evals = (0..half).into_par_iter()
        .map(|i| {
            let eq_x = self.eq_x_poly.sumcheck_evals_array::<DEGREE>(i);
            let rho = self.rho_poly.sumcheck_evals_array::<DEGREE>(i);
            let rho_next = self.rho_next_poly.sumcheck_evals_array::<DEGREE>(i);
            let quotient = self.quotient_poly.sumcheck_evals_array::<DEGREE>(i);
            let base = self.base_poly.sumcheck_evals_array::<DEGREE>(i);
            let g = self.g_poly.sumcheck_evals_array::<DEGREE>(i);

            let mut term_evals = [F::zero(); DEGREE];
            for t in 0..DEGREE {
                // base^{bit_eval}: interpolate between 1 and base
                let base_power = F::one() + (base[t] - F::one()) * self.bit_eval;

                // C(t) = rho_next - rho² × base_power - quotient × g
                let constraint = rho_next[t]
                    - rho[t] * rho[t] * base_power
                    - quotient[t] * g[t];

                term_evals[t] = eq_x[t] * constraint;
            }
            term_evals
        })
        .reduce(|| [F::zero(); DEGREE], |a, b| array::from_fn(|i| a[i] + b[i]));

    UniPoly::from_evals_and_hint(previous_claim, &evals)
}
```

**Binding in Phase 2**: Standard MLE binding on all 4-var polynomials.

### Final Claims (after 12 rounds)

After all 12 rounds, we have challenge point `(r_s*, r_x*)`:
- `r_s* = (r_0, ..., r_7)` - 8 challenges from Phase 1
- `r_x* = (r_8, ..., r_11)` - 4 challenges from Phase 2

**Opening claims**:
```
rho(r_s*, r_x*)       → from rho_poly final binding
rho_next(r_s*, r_x*)  → from rho_next_poly final binding
quotient(r_s*, r_x*)  → from quotient_poly final binding
base(r_x*)            → from base_poly final binding
bit(r_s*)             → already computed at Phase 1→2 transition
```

**Total claims: 5** (vs ~1,016 in current approach)

---

## Verifier Protocol

### Sumcheck Verification

The verifier runs the standard sumcheck verification for 12 rounds:

```rust
fn verify_sumcheck(
    proof: &SumcheckProof,
    transcript: &mut Transcript,
) -> (Vec<Challenge>, F) {
    let mut claim = F::zero();  // Input claim (should be 0)
    let mut challenges = Vec::with_capacity(12);

    for round in 0..12 {
        let round_poly = &proof.round_polys[round];

        // Verify: round_poly(0) + round_poly(1) = claim
        assert_eq!(round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one()), claim);

        // Sample challenge
        transcript.append_poly(round_poly);
        let r_j = transcript.challenge_scalar();
        challenges.push(r_j);

        // Update claim
        claim = round_poly.evaluate(r_j);
    }

    (challenges, claim)  // Final claim at challenge point
}
```

### Expected Output Claim

After sumcheck, verifier computes expected output claim:

```rust
fn expected_output_claim(
    &self,
    accumulator: &VerifierOpeningAccumulator<F>,
    challenges: &[Challenge],  // 12 challenges
) -> F {
    // Split challenges
    let r_s_star = &challenges[0..8];
    let r_x_star = &challenges[8..12];

    // Get polynomial evaluations from accumulator
    let rho_eval = accumulator.get_claim(Poly::Rho);
    let rho_next_eval = accumulator.get_claim(Poly::RhoNext);
    let quotient_eval = accumulator.get_claim(Poly::Quotient);
    let base_eval = accumulator.get_claim(Poly::Base);
    let bit_eval = accumulator.get_claim(Poly::Bit);

    // Compute eq evaluations
    let eq_s = EqPolynomial::evaluate(r_s_star, r_s_star);  // = 1 at boolean points
    let eq_x = EqPolynomial::evaluate(r_x_star, r_x_star);  // = 1 at boolean points

    // Actually, we need eq(r_s, r_s*) × eq(r_x, r_x*) but at the challenge point
    // this simplifies since we're evaluating at the challenge point itself

    // Compute g(r_x*)
    let g_eval = evaluate_g_polynomial(r_x_star);

    // base^{bit_eval} using linear interpolation
    let base_power = F::one() + (base_eval - F::one()) * bit_eval;

    // Constraint at challenge point
    let constraint_eval = rho_next_eval
        - rho_eval * rho_eval * base_power
        - quotient_eval * g_eval;

    // The expected output is the constraint evaluation
    // (should equal the final sumcheck claim)
    constraint_eval
}
```

---

## Witness Generation Changes

### Current Witness Structure

```rust
pub struct JoltGtExpWitness {
    pub base: Fq12,
    pub exponent: Fr,
    pub result: Fq12,
    pub rho_mles: Vec<Vec<Fq>>,      // 255 separate 16-element MLEs
    pub quotient_mles: Vec<Vec<Fq>>, // 254 separate 16-element MLEs
    pub bits: Vec<bool>,             // 254 bits
}
```

### New Witness Structure

```rust
pub struct JoltGtExpWitnessPacked {
    pub base: Fq12,
    pub exponent: Fr,
    pub result: Fq12,

    // Packed polynomials (12-var MLEs, 4096 coefficients each)
    pub rho_packed: Vec<Fq>,         // rho(s, x) for s ∈ [0, 255], x ∈ [0, 15]
    pub rho_next_packed: Vec<Fq>,    // rho_next(s, x) = rho(s+1, x)
    pub quotient_packed: Vec<Fq>,    // quotient(s, x)

    // Scalar bits as MLE (8-var, 256 coefficients)
    pub bits_packed: Vec<Fq>,        // bit(s) for s ∈ [0, 255]
}
```

### Packing Algorithm

```rust
fn pack_gt_exp_witness(old: &JoltGtExpWitness) -> JoltGtExpWitnessPacked {
    let n_steps = old.bits.len();  // 254
    let step_size = 256;           // 2^8 step indices
    let elem_size = 16;            // 2^4 element indices
    let total_size = step_size * elem_size;  // 4096

    // Pack rho: rho_packed[s * 16 + x] = rho_mles[s][x]
    let mut rho_packed = vec![Fq::zero(); total_size];
    for s in 0..=n_steps {  // 0 to 254 inclusive (255 rho values)
        for x in 0..elem_size {
            rho_packed[s * elem_size + x] = old.rho_mles[s][x];
        }
    }

    // Pack rho_next: rho_next_packed[s * 16 + x] = rho_mles[s+1][x]
    let mut rho_next_packed = vec![Fq::zero(); total_size];
    for s in 0..n_steps {  // 0 to 253
        for x in 0..elem_size {
            rho_next_packed[s * elem_size + x] = old.rho_mles[s + 1][x];
        }
    }

    // Pack quotient: quotient_packed[s * 16 + x] = quotient_mles[s][x]
    let mut quotient_packed = vec![Fq::zero(); total_size];
    for s in 0..n_steps {  // 0 to 253
        for x in 0..elem_size {
            quotient_packed[s * elem_size + x] = old.quotient_mles[s][x];
        }
    }

    // Pack bits: bits_packed[s] = bits[s] as field element
    let mut bits_packed = vec![Fq::zero(); step_size];
    for s in 0..n_steps {
        bits_packed[s] = if old.bits[s] { Fq::one() } else { Fq::zero() };
    }

    JoltGtExpWitnessPacked {
        base: old.base,
        exponent: old.exponent,
        result: old.result,
        rho_packed,
        rho_next_packed,
        quotient_packed,
        bits_packed,
    }
}
```

---

## Claim Count Comparison

| | Current | Packed |
|---|---------|--------|
| rho claims | 255 × 2 = 510 | 2 |
| quotient claims | 254 | 1 |
| base claims | 254 | 1 |
| bit claims | (implicit) | 1 |
| **Total per GT exp** | **~1,018** | **5** |

**Reduction: ~200x fewer claims!**

---

## Integration with Existing Stages

### Stage 1 Changes
- Replace `SquareAndMultiplyProver` with `PackedGtExpProver`
- Implement 2-phase sumcheck (8 rounds Phase 1 + 4 rounds Phase 2)
- Single constraint instance instead of 254 batched instances

### Stage 2 (Virtualization)
- Fewer virtual polynomials to virtualize
- Matrix has fewer rows (5 per GT exp instead of ~1,018)

### Stage 3 (Jagged Transform)
- Same structure but with larger polynomials
- Dense polynomial size similar (4,096 vs 254 × 16 = 4,064)

---

## Memory Comparison

| | Current | Packed |
|---|---------|--------|
| rho storage | 255 × 16 = 4,080 Fq | 4,096 Fq |
| rho_next storage | (duplicated in rho_curr) | 4,096 Fq |
| quotient storage | 254 × 16 = 4,064 Fq | 4,096 Fq |
| base storage | 254 × 16 = 4,064 Fq | 16 Fq |
| **Total** | **~16,000 Fq** | **~12,300 Fq** |

Slight memory improvement, but the real win is claim count reduction.

---

## Implementation Plan

1. **Define new witness structure** (`JoltGtExpWitnessPacked`)
2. **Implement packing function** to convert from current witness
3. **Create `PackedGtExpProver`** with 2-phase sumcheck:
   - Phase 1: 8 rounds over step variables `s`
   - Phase 2: 4 rounds over element variables `x`
4. **Create `PackedGtExpVerifier`** with updated constraint check
5. **Update constraint system** to use packed representation
6. **Update Stage 2 virtualization** for new polynomial structure
7. **Test end-to-end** with existing Dory proof verification
