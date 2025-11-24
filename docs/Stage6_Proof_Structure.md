# Stage 6 Proof Structure: What Gets Transmitted

This document details the structure of Stage 6 (SNARK Composition) proofs, focusing on what data the prover sends and what the verifier receives, without emphasis on performance costs.

## Overview

Stage 6 proves the correctness of 109 GT exponentiations from Stage 5 using the SZ-Check protocol over Grumpkin. The proof consists of:

1. **Commitments** to all intermediate exponentiation values (Hyrax commitments over Grumpkin)
2. **Sumcheck proof** for the batched error polynomial (15 rounds)
3. **Opening proofs** for evaluations at random challenge points (Hyrax openings)

## Data Structures

### ExponentiationSteps (Prover's Witness)

For each of the 109 exponentiations, the prover computes:

```rust
pub struct ExponentiationSteps {
    pub base: Fq12,                   // g ∈ GT (12 Fq coefficients)
    pub exponent: Fr,                 // x ∈ Fr (scalar)
    pub bits: Vec<bool>,              // [b_253, ..., b_0] (254 bits of exponent)
    pub rho_mles: Vec<Vec<Fq>>,      // [r_0, r_1, ..., r_254] as MLEs
    pub quotient_mles: Vec<Vec<Fq>>, // Quotient polynomials for constraints
    pub result: Fq12,                 // g^x (final result)
}
```

**Key fields:**

- **`base`**: The GT element being exponentiated (from Dory's pairing operations)
- **`bits`**: Binary representation of the exponent (determines square vs square-and-multiply steps)
- **`rho_mles`**: 255 intermediate accumulator values (r_0 = 1, r_254 = result)
  - Each r_j is an Fq12 element (12 coefficients)
  - Packed into 4-variable MLE: 12 real coefficients + 4 padding = 16 evaluations
  - Stored as vector of 16 Fq elements
- **`quotient_mles`**: Additional polynomials for constraint satisfaction
- **`result`**: Final exponentiation result (stored for verification)

### ExpCommitments (What the Verifier Receives)

Instead of sending the raw witness data, the prover commits to all MLEs using Hyrax:

```rust
pub struct ExpCommitments<const RATIO: usize = 1> {
    pub rho_commitments: Vec<Vec<HyraxCommitment<RATIO, GrumpkinProjective>>>,
    pub quotient_commitments: Vec<Vec<HyraxCommitment<RATIO, GrumpkinProjective>>>,
    pub base_commitments: Vec<HyraxCommitment<RATIO, GrumpkinProjective>>,
    pub num_exponentiations: usize,
    pub bits_per_exponentiation: Vec<Vec<bool>>,
    pub num_constraints_per_exponentiation: Vec<usize>,
}
```

**Structure:**

- **`rho_commitments`**: Nested vector indexed by [exp_idx][step_idx]
  - Outer vector: 109 exponentiations
  - Inner vector: 255 steps per exponentiation (r_0 through r_254)
  - Each commitment is to a 4-variable MLE (one r_j value)

- **`quotient_commitments`**: Additional constraint polynomial commitments
  - Same nested structure as rho_commitments

- **`base_commitments`**: Vector of 109 commitments
  - One Hyrax commitment per base GT element
  - Index: [exp_idx]

- **`bits_per_exponentiation`**: The exponent bits (not committed, sent in clear)
  - Public information needed for constraint evaluation

- **`num_constraints_per_exponentiation`**: Metadata for verification

### HyraxCommitment (The Commitment Format)

Each Hyrax commitment is structured as:

```rust
pub struct HyraxCommitment<const RATIO: usize, G: CurveGroup> {
    pub row_commitments: Vec<G>,  // Vector of elliptic curve points
}
```

**For 4-variable MLEs** (as used in Stage 6):

- 16 coefficients arranged as 4×4 matrix
- **4 row commitments** (one Grumpkin point per row)
- Each row commitment: $C_i = \sum_{j=0}^{3} M_{i,j} \cdot G_j$ (Pedersen commitment)

**Size per commitment:**
- 4 Grumpkin points × 32 bytes (compressed) = **128 bytes**

## Commitment Breakdown

### Total Commitments in Stage 6

#### 1. Base Commitments

**What**: Commitments to the 109 GT base elements being exponentiated

**Structure:**
- 109 HyraxCommitments (one per exponentiation)
- Each is a 4-variable MLE (Fq12 element packed into 16 coefficients)

**Size:**
- 109 commitments × 4 Grumpkin points = **436 Grumpkin points**
- ~14 KB

#### 2. Rho Commitments (Intermediate Accumulator Values)

**What**: Commitments to all intermediate r_j values from square-and-multiply

**Structure:**
- 109 exponentiations
- Each has 255 intermediate values (r_0 through r_254)
- Total: **109 × 255 = 27,795 HyraxCommitments**

**Breakdown:**
- 80 main Dory exponentiations × 255 steps = 20,400 commitments
- 29 RLC exponentiations × 255 steps = 7,395 commitments

**Size:**
- 27,795 commitments × 4 Grumpkin points = **111,180 Grumpkin points**
- ~3.6 MB

**Note**: This is the dominant component of the Stage 6 proof size.

#### 3. Quotient Commitments (Constraint Polynomials)

**What**: Commitments to quotient polynomials used in constraint satisfaction

**Structure:**
- Variable number per exponentiation (depends on constraint complexity)
- Typically fewer than rho commitments

**Size:** Implementation-dependent, typically smaller than rho commitments

## The RecursionProof Structure

The complete Stage 6 proof sent to the verifier:

```rust
pub struct RecursionProof<F: JoltField, ProofTranscript: Transcript, const RATIO: usize> {
    pub commitments: ExpCommitments<RATIO>,
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub r_sumcheck: Vec<F>,
    pub hyrax_proof: HyraxOpeningProof<RATIO, GrumpkinProjective>,
}
```

### Field-by-Field Explanation

#### 1. `commitments: ExpCommitments`

**Contains:**
- All Hyrax commitments described above
- Exponent bits (254 bits per exponentiation)
- Metadata (number of constraints, etc.)

**What verifier gets:**
- ~111,600 Grumpkin points (compressed elliptic curve points)
- 109 × 254 bits = ~3.5 KB of exponent bits
- Small metadata fields

**What verifier does NOT get:**
- Actual Fq12 values (12 coefficients each)
- Intermediate r_j computation details
- MLE coefficient vectors

#### 2. `sumcheck_proof: SumcheckInstanceProof`

**Contains:** The 15-round sumcheck proof for batched error polynomial

**Structure:**
```rust
pub struct SumcheckInstanceProof<F, ProofTranscript> {
    pub poly_A_evals: Vec<F>,     // Evaluations at each round
    pub poly_B_evals: Vec<F>,
    pub poly_C_evals: Vec<F>,
    // ... other round polynomials
}
```

**Details:**
- 15 rounds total:
  - 7 rounds for exponentiation index (log₂(109) ≈ 7)
  - 8 rounds for step index (log₂(256) = 8, padded from 254)
- Each round: prover sends univariate polynomial (typically degree 2-3)
- Verifier sends random challenge per round

**Size:** ~1-2 KB (small compared to commitments)

#### 3. `r_sumcheck: Vec<F>`

**Contains:** The final random challenge point from sumcheck

**Details:**
- Vector of 15 field elements (one per sumcheck variable)
- Split into:
  - `r_exp = [r_0, ..., r_6]` (exponentiation index challenges)
  - `r_step = [r_7, ..., r_14]` (step index challenges)

**Purpose:** Used to evaluate polynomials at the final random point

**Size:** 15 × 32 bytes = 480 bytes

#### 4. `hyrax_proof: HyraxOpeningProof`

**Contains:** Opening proofs for all polynomial evaluations at the random point

**Structure:**
```rust
pub struct HyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    pub vector_matrix_product: Vec<G::ScalarField>,
    // Additional fields for opening verification
}
```

**What it proves:**
- For each of the committed MLEs, the evaluation at `r_sumcheck` is correct
- Uses Hyrax's tensor product structure for efficient verification

**Required openings:**
- Each of 109 exponentiations needs openings at:
  - `r_step` (current step evaluation)
  - `r_step + 1` (next step evaluation, for recurrence relation)
- Total: 109 × 2 openings

**Size:** Dominated by proof vectors for each opening

## Verification Flow

### What the Verifier Does With Each Component

#### 1. Commitments (`ExpCommitments`)

**Verifier actions:**
- Receives and stores all Hyrax commitments (elliptic curve points)
- Does NOT recompute exponentiations
- Does NOT see actual Fq12 values
- Uses commitments as anchors for opening checks

**Purpose:** Binding the prover to specific witness values without revealing them

#### 2. Sumcheck Proof

**Verifier actions:**
- Checks each of 15 rounds:
  1. Verifies univariate polynomial consistency: $s_k(0) + s_k(1) =$ (previous sum)
  2. Sends random challenge $r_k$
  3. Updates claimed sum to $s_k(r_k)$
- Final output: random point $r_{\text{sumcheck}} = (r_0, \ldots, r_{14})$

**Purpose:** Reduces claim about 2^15 evaluations to single point check

#### 3. Random Challenge Point (`r_sumcheck`)

**Verifier actions:**
- Splits into `r_exp` (7 elements) and `r_step` (8 elements)
- Uses to request polynomial openings from prover

**Purpose:** The random point where all polynomials must be opened

#### 4. Hyrax Opening Proof

**Verifier actions:**
- For each of 109 exponentiations:
  1. Request opening at `r_step` and `r_step + 1`
  2. Verify opening using two MSMs per opening:
     - MSM #1: Homomorphic combination of row commitments
     - MSM #2: Product commitment from proof vector
  3. Check two consistency conditions:
     - Commitments match: $C_{\text{derived}} \stackrel{?}{=} C_{\text{product}}$
     - Inner product matches claimed evaluation: $\langle u, R \rangle \stackrel{?}{=} v$
- Evaluates error polynomial using opened values
- Verifies error polynomial evaluation matches sumcheck's final claim

**Purpose:** Proves that committed polynomials actually evaluate to claimed values at random point

## Key Insights

### What Gets Hidden

The prover's witness (actual Fq12 values and all intermediate computations) is **never transmitted**. The verifier only receives:

1. **Commitments** (cryptographic bindings to the witness)
2. **Proof data** (sumcheck round polynomials, opening proof vectors)
3. **Public information** (exponent bits)

### What Gets Proven

Despite not seeing the witness, the verifier is convinced that:

1. **Each intermediate step is correct**: $r_{j+1} = r_j^2 \cdot g^{b_j}$ for all 109 exponentiations
2. **Final results match Stage 5**: The 109th intermediate value equals the claimed result
3. **Computational integrity**: Prover followed the square-and-multiply algorithm correctly

### Why Hyrax Commitments?

**Reason 1: Field matching**
- GT elements have coefficients in Fq (BN254's base field)
- Grumpkin's scalar field = Fq
- Native field arithmetic (no expensive limb decomposition)

**Reason 2: Efficient openings**
- $O(\sqrt{N})$ verifier work per opening
- For 4-variable MLEs: only 8 scalar multiplications per opening
- Tensor product structure exploited for efficiency

**Reason 3: Transparency**
- No trusted setup required
- Generators are deterministically derived
- Compatible with Jolt's transparency goals

## Comparison to Alternative Approaches

### If We Sent Raw Witness Data

**Hypothetical:**
- 109 exponentiations × 255 intermediate values × 12 Fq elements = 333,180 field elements
- Each Fq element: 32 bytes
- **Total: ~10 MB of raw data**

**Problems:**
- No hiding (verifier sees all intermediate values)
- No succinctness (linear in computation size)
- Verifier must recompute to check correctness

### With Hyrax Commitments (Actual Approach)

**Reality:**
- ~111,600 Grumpkin points (commitments)
- ~3.6 MB commitment data
- Plus ~1-2 KB sumcheck proof
- Plus opening proofs

**Benefits:**
- Zero-knowledge (commitments hide values)
- Succinct verification (verifier does not recompute exponentiations)
- Verifier only performs elliptic curve operations (MSMs)

## Summary

**What the prover computes:**
- 109 complete exponentiation witnesses (all 255 intermediate steps each)
- Packs each Fq12 value into 4-variable MLE
- Creates Hyrax commitments for all MLEs

**What the verifier receives:**
- ~111,600 Grumpkin elliptic curve points (Hyrax commitments)
- 15-round sumcheck proof (~1-2 KB)
- Opening proofs for evaluations at random points

**What the verifier does:**
- Verifies sumcheck protocol (15 rounds of cheap field operations)
- Checks Hyrax openings (109 × 2 openings, each requiring two 4-base MSMs)
- Never recomputes exponentiations, never sees Fq12 values

**The key trade-off:**
- Large commitment data (~3.6 MB) in exchange for:
  - Fast verification (no exponentiation recomputation)
  - Zero-knowledge (commitments hide witness)
  - Field compatibility (enables SNARK composition)
