//! Packed GT exponentiation sumcheck with 2-phase protocol (base-4 digits)
//!
//! This packs ~127 base-4 steps into larger MLEs, reducing claims to 3 per GT exp.
//!
//! Polynomial structure (committed by prover):
//! - rho(s, x): 11-var (7 step + 4 element) - intermediate results
//! - quotient(s, x): 11-var - quotient polynomials
//!
//! Public inputs (verifier computes directly, not committed):
//! - digit_lo(s), digit_hi(s): 7-var - scalar digit bits (derived from public scalar exponent)
//! - base(x): 4-var - base element (derived from public Fq12 base)
//! - base2(x), base3(x): 4-var - base powers (derived from public base)
//!
//! Virtual polynomial (verified via shift sumcheck, not committed):
//! - rho_next(s, x): 11-var - shifted: rho_next(i, x) = rho(i+1, x)
//!
//! Constraint: C(s, x) = rho_next(s, x) - rho(s, x)^4 × base(x)^{digit(s)} - quotient(s, x) × g(x)
//!
//! 2-phase sumcheck:
//! - Phase 1: 7 rounds over step variables (s)
//! - Phase 2: 4 rounds over element variables (x)

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        recursion::utils::virtual_polynomial_utils::*,
        witness::VirtualPolynomial,
    },
    virtual_claims,
};
use ark_bn254::{Fq, Fq12};
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::fq12_to_multilinear_evals;
use rayon::prelude::*;

/// Number of step variables (7 for 128 base-4 steps)
pub const NUM_STEP_VARS: usize = 7;

/// Number of element variables (4 for Fq12 = 12 field elements, but we use 16 for power of 2)
pub const NUM_ELEMENT_VARS: usize = 4;

/// Total variables = step + element
pub const NUM_TOTAL_VARS: usize = NUM_STEP_VARS + NUM_ELEMENT_VARS;

/// Public inputs for a single packed GT exponentiation.
///
/// These are known to both prover and verifier, so the verifier can compute
/// the digit and base polynomial evaluations directly without receiving claims.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PackedGtExpPublicInputs {
    /// Base GT element (Fq12) for this exponentiation
    pub base: Fq12,
    /// Scalar bits (MSB-first, no leading zeros)
    pub scalar_bits: Vec<bool>,
}

impl PackedGtExpPublicInputs {
    /// Create new public inputs for a GT exponentiation
    pub fn new(base: Fq12, scalar_bits: Vec<bool>) -> Self {
        Self { base, scalar_bits }
    }

    /// Evaluate digit_lo MLE at challenge point r_s* (7-variable MLE, 128 points).
    pub fn evaluate_digit_lo_mle<F: JoltField>(&self, r_s_star: &[F]) -> F {
        debug_assert_eq!(r_s_star.len(), NUM_STEP_VARS);
        let eq_evals = EqPolynomial::<F>::evals(r_s_star);
        let digits = digits_from_bits_msb(&self.scalar_bits);

        eq_evals
            .iter()
            .enumerate()
            .map(|(s, eq)| {
                if s < digits.len() && digits[s].1 {
                    *eq
                } else {
                    F::zero()
                }
            })
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Evaluate digit_hi MLE at challenge point r_s* (7-variable MLE, 128 points).
    pub fn evaluate_digit_hi_mle<F: JoltField>(&self, r_s_star: &[F]) -> F {
        debug_assert_eq!(r_s_star.len(), NUM_STEP_VARS);
        let eq_evals = EqPolynomial::<F>::evals(r_s_star);
        let digits = digits_from_bits_msb(&self.scalar_bits);

        eq_evals
            .iter()
            .enumerate()
            .map(|(s, eq)| {
                if s < digits.len() && digits[s].0 {
                    *eq
                } else {
                    F::zero()
                }
            })
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Evaluate the base MLE at challenge point r_x* (4-variable MLE, 16 points).
    pub fn evaluate_base_mle<F: JoltField>(&self, r_x_star: &[F]) -> F {
        self.evaluate_fq12_mle(&self.base, r_x_star)
    }

    /// Evaluate base^2 MLE at challenge point r_x* (4-variable MLE, 16 points).
    pub fn evaluate_base2_mle<F: JoltField>(&self, r_x_star: &[F]) -> F {
        let base2 = self.base * self.base;
        self.evaluate_fq12_mle(&base2, r_x_star)
    }

    /// Evaluate base^3 MLE at challenge point r_x* (4-variable MLE, 16 points).
    pub fn evaluate_base3_mle<F: JoltField>(&self, r_x_star: &[F]) -> F {
        let base2 = self.base * self.base;
        let base3 = base2 * self.base;
        self.evaluate_fq12_mle(&base3, r_x_star)
    }

    fn evaluate_fq12_mle<F: JoltField>(&self, fq12: &Fq12, r_x_star: &[F]) -> F {
        use std::any::TypeId;

        debug_assert_eq!(r_x_star.len(), NUM_ELEMENT_VARS);

        // Runtime check that F = Fq for safe transmute
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("evaluate_base_mle requires F = Fq");
        }

        let base_mle_fq = fq12_to_multilinear_evals(fq12); // 16 Fq values
        let base_poly = DensePolynomial::new(base_mle_fq);

        // Convert r_x_star to Fq slice
        // SAFETY: F = Fq verified above, and slice references have same layout
        let r_x_star_fq: &[Fq] = unsafe {
            std::slice::from_raw_parts(r_x_star.as_ptr() as *const Fq, r_x_star.len())
        };
        let result_fq = base_poly.evaluate(r_x_star_fq);
        unsafe { std::mem::transmute_copy(&result_fq) }
    }
}

/// Constraint polynomials for a single packed GT exponentiation
/// Used when extracting from the Dory matrix for the prover
#[derive(Clone)]
pub struct PackedGtExpConstraintPolynomials<F: JoltField> {
    pub rho: Vec<F>,
    pub quotient: Vec<F>,
    pub digit_lo: Vec<F>,
    pub digit_hi: Vec<F>,
    pub base: Vec<F>,
    pub base2: Vec<F>,
    pub base3: Vec<F>,
    pub constraint_index: usize,
}

/// Packed witness for GT exponentiation (used during matrix construction)
///
/// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits)
/// This allows LowToHigh binding to give us:
/// - Phase 1 (rounds 0-6): bind step variables s
/// - Phase 2 (rounds 7-10): bind element variables x
#[derive(Clone)]
pub struct PackedGtExpWitness {
    /// rho(s, x) - all intermediate results packed into 11-var MLE
    /// Layout: rho[x * 128 + s] = ρ_s[x]
    pub rho_packed: Vec<Fq>,

    /// rho_next(s, x) = rho(s+1, x) - shifted intermediate results
    /// Layout: rho_next[x * 128 + s] = ρ_{s+1}[x]
    pub rho_next_packed: Vec<Fq>,

    /// quotient(s, x) - all quotients packed into 11-var MLE
    /// Layout: quotient[x * 128 + s] = Q_s[x]
    pub quotient_packed: Vec<Fq>,

    /// digit_lo(s) - low digit bit replicated across x
    /// Layout: digit_lo[x * 128 + s] = u_s (same for all x)
    pub digit_lo_packed: Vec<Fq>,

    /// digit_hi(s) - high digit bit replicated across x
    /// Layout: digit_hi[x * 128 + s] = v_s (same for all x)
    pub digit_hi_packed: Vec<Fq>,

    /// base(x) - base element replicated across s
    /// Layout: base[x * 128 + s] = base[x] (same for all s)
    pub base_packed: Vec<Fq>,

    /// base2(x) - base^2 element replicated across s
    /// Layout: base2[x * 128 + s] = base2[x] (same for all s)
    pub base2_packed: Vec<Fq>,

    /// base3(x) - base^3 element replicated across s
    /// Layout: base3[x * 128 + s] = base3[x] (same for all s)
    pub base3_packed: Vec<Fq>,

    /// Number of actual steps (<= 127 for BN254 scalar)
    pub num_steps: usize,
}

impl PackedGtExpWitness {
    /// Create packed witness from individual step data
    ///
    /// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits)
    /// This allows LowToHigh binding to naturally give us:
    /// - Phase 1 (rounds 0-6): bind step variables s (low bits)
    /// - Phase 2 (rounds 7-10): bind element variables x (high bits)
    pub fn from_steps(
        rho_mles: &[Vec<Fq>],        // rho_mles[step][x] for step in 0..=num_steps
        quotient_mles: &[Vec<Fq>],   // quotient_mles[step][x] for step in 0..num_steps
        bits: &[bool],               // bits (MSB first)
        base_mle: &[Fq],             // base[x] - 16 values
        base2_mle: &[Fq],            // base^2[x] - 16 values
        base3_mle: &[Fq],            // base^3[x] - 16 values
    ) -> Self {
        let mut digits = digits_from_bits_msb(bits);
        let num_steps = digits.len();
        assert_eq!(rho_mles.len(), num_steps + 1, "Need num_steps + 1 rho MLEs");
        assert_eq!(quotient_mles.len(), num_steps, "Need num_steps quotient MLEs");
        assert_eq!(base_mle.len(), 16, "Base must be 4-var MLE (16 values)");
        assert_eq!(base2_mle.len(), 16, "Base2 must be 4-var MLE (16 values)");
        assert_eq!(base3_mle.len(), 16, "Base3 must be 4-var MLE (16 values)");

        let step_size = 1 << NUM_STEP_VARS;   // 128
        let elem_size = 1 << NUM_ELEMENT_VARS; // 16
        let total_size = 1 << NUM_TOTAL_VARS;  // 2048
        let num_steps_padded = (num_steps + 1).min(step_size);

        let mut quotient_mles_padded = quotient_mles.to_vec();
        if num_steps_padded > num_steps {
            use jolt_optimizations::get_g_mle;
            let g_mle = get_g_mle();
            let rho_prev = &rho_mles[num_steps];
            let mut dummy_q = vec![Fq::zero(); elem_size];
            for j in 0..elem_size {
                let rho4 = rho_prev[j].square().square();
                if !g_mle[j].is_zero() {
                    dummy_q[j] = -rho4 / g_mle[j];
                }
            }
            quotient_mles_padded.push(dummy_q);
            digits.push((false, false));
        }

        // Pack rho: rho_packed[x * 128 + s] = rho_mles[s][x]
        // s in low 7 bits, x in high 4 bits
        // NOTE: Populate through num_steps_padded so rho_next is a pure shift of rho.
        let mut rho_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            for x in 0..elem_size {
                if s < rho_mles.len() && x < rho_mles[s].len() {
                    rho_packed[x * step_size + s] = rho_mles[s][x];
                }
            }
        }

        // Pack rho_next: rho_next_packed[x * 128 + s] = rho_mles[s+1][x]
        let mut rho_next_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            for x in 0..elem_size {
                if s + 1 < rho_mles.len() && x < rho_mles[s + 1].len() {
                    rho_next_packed[x * step_size + s] = rho_mles[s + 1][x];
                }
            }
        }

        // Pack quotient: quotient_packed[x * 128 + s] = quotient_mles[s][x]
        let mut quotient_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            for x in 0..elem_size {
                if s < quotient_mles_padded.len() && x < quotient_mles_padded[s].len() {
                    quotient_packed[x * step_size + s] = quotient_mles_padded[s][x];
                }
            }
        }

        // Pack digit_lo and digit_hi: replicated across x
        let mut digit_lo_packed = vec![Fq::zero(); total_size];
        let mut digit_hi_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            let (digit_hi, digit_lo) = digits[s];
            let lo_val = if digit_lo { Fq::from(1u64) } else { Fq::zero() };
            let hi_val = if digit_hi { Fq::from(1u64) } else { Fq::zero() };
            for x in 0..elem_size {
                digit_lo_packed[x * step_size + s] = lo_val;
                digit_hi_packed[x * step_size + s] = hi_val;
            }
        }

        // Pack base: base_packed[x * 128 + s] = base_mle[x] (replicated across s)
        // base only depends on x, so same value for all s with same x
        let mut base_packed = vec![Fq::zero(); total_size];
        for x in 0..elem_size {
            for s in 0..step_size {
                base_packed[x * step_size + s] = base_mle[x];
            }
        }

        // Pack base2: base2_packed[x * 128 + s] = base2_mle[x] (replicated across s)
        let mut base2_packed = vec![Fq::zero(); total_size];
        for x in 0..elem_size {
            for s in 0..step_size {
                base2_packed[x * step_size + s] = base2_mle[x];
            }
        }

        // Pack base3: base3_packed[x * 128 + s] = base3_mle[x] (replicated across s)
        let mut base3_packed = vec![Fq::zero(); total_size];
        for x in 0..elem_size {
            for s in 0..step_size {
                base3_packed[x * step_size + s] = base3_mle[x];
            }
        }

        let witness = Self {
            rho_packed,
            rho_next_packed,
            quotient_packed,
            digit_lo_packed,
            digit_hi_packed,
            base_packed,
            base2_packed,
            base3_packed,
            num_steps: num_steps_padded,
        };

        // Debug: verify all constraints are zero over the Boolean hypercube
        #[cfg(test)]
        {
            use jolt_optimizations::get_g_mle;

            let g_mle = get_g_mle();
            let mut failed_constraints = Vec::new();

            // Check C_s(x) = 0 for all valid (s, x) pairs
            for s in 0..num_steps {
                let (digit_hi, digit_lo) = digits[s];
                let u = if digit_lo { Fq::from(1u64) } else { Fq::zero() };
                let v = if digit_hi { Fq::from(1u64) } else { Fq::zero() };
                let one = Fq::from(1u64);
                let w0 = (one - u) * (one - v);
                let w1 = u * (one - v);
                let w2 = (one - u) * v;
                let w3 = u * v;

                for x in 0..16 {
                    let rho = rho_mles[s][x];
                    let rho_next = rho_mles[s + 1][x];
                    let quotient = quotient_mles[s][x];
                    let base = base_mle[x];
                    let base2 = base2_mle[x];
                    let base3 = base3_mle[x];
                    let g = g_mle[x];

                    let base_power = w0 + w1 * base + w2 * base2 + w3 * base3;

                    let rho2 = rho * rho;
                    let rho4 = rho2 * rho2;
                    let constraint = rho_next - rho4 * base_power - quotient * g;

                    if !constraint.is_zero() {
                        if failed_constraints.is_empty() {
                            eprintln!(
                                "PackedGtExpWitness debug: num_steps={}, first_fail s={}, x={}",
                                num_steps, s, x
                            );
                            eprintln!(
                                "  digit_hi={}, digit_lo={}",
                                digit_hi, digit_lo
                            );
                            eprintln!(
                                "  rho={:?}",
                                rho
                            );
                            eprintln!(
                                "  rho_next={:?}",
                                rho_next
                            );
                            eprintln!(
                                "  quotient={:?}",
                                quotient
                            );
                            eprintln!(
                                "  base={:?}",
                                base
                            );
                            eprintln!(
                                "  base2={:?}",
                                base2
                            );
                            eprintln!(
                                "  base3={:?}",
                                base3
                            );
                            eprintln!(
                                "  g={:?}",
                                g
                            );
                            eprintln!(
                                "  base_power={:?}",
                                base_power
                            );
                            eprintln!(
                                "  rho4={:?}",
                                rho4
                            );
                        }
                        failed_constraints.push((s, x, constraint));
                    }
                }
            }

            if !failed_constraints.is_empty() {
                eprintln!(
                    "PackedGtExpWitness: {} constraints are non-zero!",
                    failed_constraints.len()
                );
                for (s, x, val) in failed_constraints.iter().take(5) {
                    eprintln!("  step={}, x={}: constraint = {:?}", s, x, val);
                }
                if failed_constraints.len() > 5 {
                    eprintln!("  ... and {} more", failed_constraints.len() - 5);
                }
                panic!(
                    "PackedGtExpWitness: {} constraints are non-zero!",
                    failed_constraints.len()
                );
            }
        }

        witness
    }
}

fn digits_from_bits_msb(bits_msb: &[bool]) -> Vec<(bool, bool)> {
    if bits_msb.is_empty() {
        return vec![];
    }

    let mut padded = bits_msb.to_vec();
    if padded.len() % 2 == 1 {
        padded.insert(0, false);
    }

    let mut digits = Vec::with_capacity(padded.len() / 2);
    for i in (0..padded.len()).step_by(2) {
        let digit_hi = padded[i];
        let digit_lo = padded[i + 1];
        digits.push((digit_hi, digit_lo));
    }

    digits
}

/// Parameters for packed GT exp sumcheck
#[derive(Clone)]
pub struct PackedGtExpParams {
    /// Total number of constraint variables (11 = 7 step + 4 element)
    pub num_constraint_vars: usize,

    /// Number of step variables
    pub num_step_vars: usize,

    /// Number of element variables
    pub num_element_vars: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl PackedGtExpParams {
    pub fn new() -> Self {
        Self {
            num_constraint_vars: NUM_TOTAL_VARS,
            num_step_vars: NUM_STEP_VARS,
            num_element_vars: NUM_ELEMENT_VARS,
            sumcheck_id: SumcheckId::PackedGtExp,
        }
    }
}

impl Default for PackedGtExpParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Prover for packed GT exponentiation sumcheck
///
/// 2-phase sumcheck:
/// - Phase 1 (rounds 0-6): bind step variables s (low bits)
/// - Phase 2 (rounds 7-10): bind element variables x (high bits)
///
/// Note: digit and base power polynomials are used internally during sumcheck computation
/// but are NOT committed as virtual claims. The verifier computes these evaluations
/// directly from public inputs (base Fq12 and scalar bits).
pub struct PackedGtExpProver<F: JoltField> {
    /// Parameters
    pub params: PackedGtExpParams,

    /// Number of witnesses (GT exp instances)
    pub num_witnesses: usize,

    /// Packed rho polynomials (one per witness) - COMMITTED
    pub rho_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed rho_next polynomials (one per witness) - COMMITTED
    pub rho_next_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed quotient polynomials (one per witness) - COMMITTED
    pub quotient_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed digit_lo polynomials (one per witness) - NOT COMMITTED (public input)
    /// Used internally for sumcheck computation only
    pub digit_lo_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed digit_hi polynomials (one per witness) - NOT COMMITTED (public input)
    /// Used internally for sumcheck computation only
    pub digit_hi_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed base polynomials (one per witness) - NOT COMMITTED (public input)
    /// Used internally for sumcheck computation only
    pub base_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed base2 polynomials (one per witness) - NOT COMMITTED (public input)
    pub base2_polys: Vec<MultilinearPolynomial<F>>,

    /// Packed base3 polynomials (one per witness) - NOT COMMITTED (public input)
    pub base3_polys: Vec<MultilinearPolynomial<F>>,

    /// g(x) polynomial (padded to 11-var, shared across all witnesses) - NOT COMMITTED (constant)
    pub g_poly: MultilinearPolynomial<F>,

    /// eq(r_x, x) polynomial for element batching (4-var, bound in Phase 1)
    pub eq_x: MultilinearPolynomial<F>,

    /// eq(r_s, s) polynomial for step batching (7-var, bound in Phase 2)
    pub eq_s: MultilinearPolynomial<F>,

    /// Random challenges for element variables
    pub r_x: Vec<F::Challenge>,

    /// Random challenges for step variables
    pub r_s: Vec<F::Challenge>,

    /// Gamma coefficient for batching GT exp instances
    pub gamma: F,

    /// Current round (0 to 10)
    pub round: usize,

    /// Final claims after all rounds (one per witness) - only for committed polynomials
    pub rho_claims: Vec<F>,
    pub rho_next_claims: Vec<F>,
    pub quotient_claims: Vec<F>,

    #[cfg(test)]
    pub rho_packed_raw: Vec<Vec<F>>,
    #[cfg(test)]
    pub rho_next_packed_raw: Vec<Vec<F>>,
}

impl<F: JoltField> PackedGtExpProver<F> {
    /// Create a new packed GT exp prover with multiple witnesses and gamma batching
    pub fn new<T: Transcript>(
        params: PackedGtExpParams,
        witnesses: &[PackedGtExpWitness],
        g_poly: DensePolynomial<F>,
        transcript: &mut T,
    ) -> Self {
        use std::any::TypeId;

        // Runtime check that F = Fq for safe transmute
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("PackedGtExpProver requires F = Fq");
        }

        // Sample random challenges for element variables (4) - for eq_x polynomial
        let r_x: Vec<F::Challenge> = (0..params.num_element_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        // Sample random challenges for step variables (7) - for eq_s polynomial
        let r_s: Vec<F::Challenge> = (0..params.num_step_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        // Sample gamma for batching across witnesses
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        // Create eq polynomials for 2-phase sumcheck
        // Data layout: index = x * 128 + s (s in low 7 bits)
        // Phase 1 (rounds 0-6): bind s variables, eq_s is sumchecked
        // Phase 2 (rounds 7-10): bind x variables, eq_x is sumchecked
        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));
        let eq_s = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_s));

        // Convert witness to field F (safe because we checked F = Fq above)
        let convert_vec = |v: &[Fq]| -> Vec<F> {
            v.iter()
                .map(|fq| unsafe { std::mem::transmute_copy(fq) })
                .collect()
        };

        let num_witnesses = witnesses.len();

        let mut rho_polys = Vec::with_capacity(num_witnesses);
        let mut rho_next_polys = Vec::with_capacity(num_witnesses);
        let mut quotient_polys = Vec::with_capacity(num_witnesses);
        let mut digit_lo_polys = Vec::with_capacity(num_witnesses);
        let mut digit_hi_polys = Vec::with_capacity(num_witnesses);
        let mut base_polys = Vec::with_capacity(num_witnesses);
        let mut base2_polys = Vec::with_capacity(num_witnesses);
        let mut base3_polys = Vec::with_capacity(num_witnesses);

        #[cfg(test)]
        let mut rho_packed_raw = Vec::with_capacity(num_witnesses);
        #[cfg(test)]
        let mut rho_next_packed_raw = Vec::with_capacity(num_witnesses);

        for witness in witnesses {
            rho_polys.push(MultilinearPolynomial::from(convert_vec(&witness.rho_packed)));
            rho_next_polys.push(MultilinearPolynomial::from(convert_vec(&witness.rho_next_packed)));
            quotient_polys.push(MultilinearPolynomial::from(convert_vec(&witness.quotient_packed)));
            digit_lo_polys.push(MultilinearPolynomial::from(convert_vec(&witness.digit_lo_packed)));
            digit_hi_polys.push(MultilinearPolynomial::from(convert_vec(&witness.digit_hi_packed)));
            base_polys.push(MultilinearPolynomial::from(convert_vec(&witness.base_packed)));
            base2_polys.push(MultilinearPolynomial::from(convert_vec(&witness.base2_packed)));
            base3_polys.push(MultilinearPolynomial::from(convert_vec(&witness.base3_packed)));

            #[cfg(test)]
            {
                rho_packed_raw.push(convert_vec(&witness.rho_packed));
                rho_next_packed_raw.push(convert_vec(&witness.rho_next_packed));
            }
        }

        Self {
            params,
            num_witnesses,
            rho_polys,
            rho_next_polys,
            quotient_polys,
            digit_lo_polys,
            digit_hi_polys,
            base_polys,
            base2_polys,
            base3_polys,
            g_poly: MultilinearPolynomial::LargeScalars(g_poly),
            eq_x,
            eq_s,
            r_x,
            r_s,
            gamma,
            round: 0,
            rho_claims: vec![F::zero(); num_witnesses],
            rho_next_claims: vec![F::zero(); num_witnesses],
            quotient_claims: vec![F::zero(); num_witnesses],
            #[cfg(test)]
            rho_packed_raw,
            #[cfg(test)]
            rho_next_packed_raw,
        }
    }

    /// Check if we're in Phase 1 (step variable rounds 0-6)
    /// Data layout: index = x * 128 + s (s in low 7 bits)
    /// With LowToHigh binding: rounds 0-6 bind s (step), rounds 7-10 bind x (element)
    fn in_step_phase(&self) -> bool {
        self.round < self.params.num_step_vars
    }

    /// Get rho polynomials for shift sumcheck
    pub fn get_rho_polynomials(&self) -> Vec<Vec<F>> {
        self.rho_polys
            .iter()
            .map(|poly| match poly {
                MultilinearPolynomial::LargeScalars(p) => p.Z.to_vec(),
                _ => panic!("Rho polynomials should be LargeScalars"),
            })
            .collect()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for PackedGtExpProver<F> {
    fn degree(&self) -> usize {
        // Degree from constraint: rho^4 × digit selector (degree 6)
        7
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars // 11 rounds total
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero() // The constraint should sum to zero
    }

    #[tracing::instrument(skip_all, name = "PackedGtExp::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 7;

        let half = if self.num_witnesses > 0 {
            self.rho_polys[0].len() / 2
        } else {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(); DEGREE]);
        };

        let gamma = self.gamma;
        let num_witnesses = self.num_witnesses;
        let in_step_phase = self.in_step_phase();

        // Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits)
        // Phase 1 (rounds 0-6): bind s variables, eq_s is sumchecked, eq_x provides constants
        // Phase 2 (rounds 7-10): bind x variables, eq_x is sumchecked, eq_s is constant
        let eq_s_half = self.eq_s.len() / 2;
        let eq_x_len = self.eq_x.len();

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let g = self
                    .g_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                // Compute eq contributions based on phase
                let (eq_s_evals, eq_x_evals) = if in_step_phase {
                    // Phase 1 (rounds 0-6): sumcheck over s, eq_x is constant per x-block
                    // Index i maps to: s_pair_idx = i % eq_s_half, x_idx = i / eq_s_half
                    let s_pair_idx = i % eq_s_half;
                    let x_idx = i / eq_s_half;

                    let eq_s_arr = self
                        .eq_s
                        .sumcheck_evals_array::<DEGREE>(s_pair_idx, BindingOrder::LowToHigh);

                    // eq_x[x_idx] is constant for this s-block
                    let eq_x_val = if x_idx < eq_x_len {
                        self.eq_x.get_bound_coeff(x_idx)
                    } else {
                        F::zero()
                    };
                    let eq_x_arr = [eq_x_val; DEGREE];

                    (eq_s_arr, eq_x_arr)
                } else {
                    // Phase 2 (rounds 7-10): eq_s is fully bound (constant), sumcheck over x
                    let eq_s_val = self.eq_s.get_bound_coeff(0);
                    let eq_s_arr = [eq_s_val; DEGREE];

                    let eq_x_arr = self
                        .eq_x
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    (eq_s_arr, eq_x_arr)
                };

                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = gamma;

                // Sum over all witnesses with gamma batching
                for w in 0..num_witnesses {
                    let rho = self.rho_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let rho_next = self.rho_next_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let quotient = self.quotient_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let digit_lo = self.digit_lo_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let digit_hi = self.digit_hi_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let base = self.base_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let base2 = self.base2_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let base3 = self.base3_polys[w]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        let u = digit_lo[t];
                        let v = digit_hi[t];
                        let w0 = (F::one() - u) * (F::one() - v);
                        let w1 = u * (F::one() - v);
                        let w2 = (F::one() - u) * v;
                        let w3 = u * v;
                        let base_power = w0 + w1 * base[t] + w2 * base2[t] + w3 * base3[t];
                        let rho2 = rho[t] * rho[t];
                        let rho4 = rho2 * rho2;

                        // C(s, x) = rho_next - rho^4 × base_power - quotient × g
                        let constraint = rho_next[t] - rho4 * base_power - quotient[t] * g[t];

                        term_evals[t] += eq_x_evals[t] * eq_s_evals[t] * gamma_power * constraint;
                    }

                    gamma_power *= gamma;
                }
                term_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        // Debug: verify s(0) + s(1) = previous_claim
        #[cfg(test)]
        {
            // Directly compute s(0) and s(1) by summing over all indices
            let mut s_0 = F::zero();
            let mut s_1 = F::zero();

            let full_len = half * 2;
            for i in 0..full_len {
                let g_val = self.g_poly.get_bound_coeff(i);

                // Compute eq contributions based on phase
                let eq_combined = if in_step_phase {
                    let s_idx = i % self.eq_s.len();
                    let x_idx = i / self.eq_s.len();
                    let eq_s_val = self.eq_s.get_bound_coeff(s_idx);
                    let eq_x_val = if x_idx < eq_x_len {
                        self.eq_x.get_bound_coeff(x_idx)
                    } else {
                        F::zero()
                    };
                    eq_s_val * eq_x_val
                } else {
                    let eq_s_val = self.eq_s.get_bound_coeff(0);
                    let eq_x_val = self.eq_x.get_bound_coeff(i);
                    eq_s_val * eq_x_val
                };

                let mut gamma_power = gamma;
                for w in 0..num_witnesses {
                    let rho_val = self.rho_polys[w].get_bound_coeff(i);
                    let rho_next_val = self.rho_next_polys[w].get_bound_coeff(i);
                    let quotient_val = self.quotient_polys[w].get_bound_coeff(i);
                    let digit_lo_val = self.digit_lo_polys[w].get_bound_coeff(i);
                    let digit_hi_val = self.digit_hi_polys[w].get_bound_coeff(i);
                    let base_val = self.base_polys[w].get_bound_coeff(i);
                    let base2_val = self.base2_polys[w].get_bound_coeff(i);
                    let base3_val = self.base3_polys[w].get_bound_coeff(i);

                    let u = digit_lo_val;
                    let v = digit_hi_val;
                    let w0 = (F::one() - u) * (F::one() - v);
                    let w1 = u * (F::one() - v);
                    let w2 = (F::one() - u) * v;
                    let w3 = u * v;
                    let base_power = w0 + w1 * base_val + w2 * base2_val + w3 * base3_val;
                    let rho2 = rho_val * rho_val;
                    let rho4 = rho2 * rho2;
                    let constraint = rho_next_val - rho4 * base_power - quotient_val * g_val;

                    let term = eq_combined * gamma_power * constraint;

                    // Even indices contribute to s(0), odd indices to s(1)
                    if i % 2 == 0 {
                        s_0 += term;
                    } else {
                        s_1 += term;
                    }

                    gamma_power *= gamma;
                }
            }

            let sum = s_0 + s_1;
            if sum != previous_claim {
                eprintln!(
                    "PackedGtExp round {}: s(0) + s(1) != previous_claim!",
                    _round
                );
                eprintln!("  rho_len = {}", self.rho_polys[0].len());
                eprintln!("  eq_s_len = {}", self.eq_s.len());
                eprintln!("  eq_x_len = {}", eq_x_len);
                eprintln!("  half = {}", half);
                eprintln!("  full_len = {}", full_len);
                eprintln!("  evals[0] = {:?}", evals[0]);
                eprintln!("  s(0) = {:?}", s_0);
                eprintln!("  s(1) = {:?}", s_1);
                eprintln!("  s(0) + s(1) = {:?}", sum);
                eprintln!("  previous_claim = {:?}", previous_claim);
                eprintln!("  in_step_phase = {}", in_step_phase);
                panic!("Sumcheck relation violated!");
            }
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "PackedGtExp::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        // Bind committed witness polynomials
        self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        for poly in &mut self.rho_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.rho_next_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.quotient_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        // Bind public input polynomials (used for sumcheck computation, not committed)
        for poly in &mut self.digit_lo_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.digit_hi_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.base_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.base2_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.base3_polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        // Bind eq polynomial based on phase
        // Data layout: index = x * 128 + s (s in low 7 bits)
        // Phase 1 (rounds 0-6): bind eq_s (step variables in low bits)
        // Phase 2 (rounds 7-10): bind eq_x (element variables in high bits)
        if self.in_step_phase() {
            self.eq_s.bind_parallel(r_j, BindingOrder::LowToHigh);
        } else {
            self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        // After all rounds, extract final claims for committed polynomials only.
        if self.round == self.params.num_constraint_vars {
            for w in 0..self.num_witnesses {
                self.rho_claims[w] = self.rho_polys[w].get_bound_coeff(0);
                self.rho_next_claims[w] = self.rho_next_polys[w].get_bound_coeff(0);
                self.quotient_claims[w] = self.quotient_polys[w].get_bound_coeff(0);
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());

        // Cache all polynomial claims including rho_next as virtual
        // rho and quotient are committed polynomials
        // rho_next is virtual (will be verified via shift sumcheck)
        for w in 0..self.num_witnesses {
            let claims = virtual_claims![
                VirtualPolynomial::PackedGtExpRho(w) => self.rho_claims[w],
                VirtualPolynomial::PackedGtExpRhoNext(w) => self.rho_next_claims[w],
                VirtualPolynomial::PackedGtExpQuotient(w) => self.quotient_claims[w],
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point,
                &claims,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for packed GT exponentiation sumcheck
///
/// 2-phase sumcheck verification:
/// - Phase 1 (rounds 0-6): step variables s
/// - Phase 2 (rounds 7-10): element variables x
///
/// The verifier computes digit and base polynomial evaluations directly from
/// public inputs, rather than receiving them as claims from the prover.
pub struct PackedGtExpVerifier<F: JoltField> {
    pub params: PackedGtExpParams,
    pub r_x: Vec<F::Challenge>,
    pub r_s: Vec<F::Challenge>,
    pub gamma: F,
    pub num_witnesses: usize,
    /// Public inputs for each witness (base Fq12 and scalar bits)
    pub public_inputs: Vec<PackedGtExpPublicInputs>,
}

impl<F: JoltField> PackedGtExpVerifier<F> {
    pub fn new<T: Transcript>(
        params: PackedGtExpParams,
        public_inputs: Vec<PackedGtExpPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        let num_witnesses = public_inputs.len();

        // Sample challenges for element variables (4) - must match prover sampling order
        // These form the eq_x polynomial for Phase 2 (rounds 7-10)
        let r_x: Vec<F::Challenge> = (0..params.num_element_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        // Sample challenges for step variables (7) - must match prover sampling order
        // These form the eq_s polynomial for Phase 1 (rounds 0-6)
        let r_s: Vec<F::Challenge> = (0..params.num_step_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        // Sample gamma for batching across witnesses (must match prover)
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        Self {
            params,
            r_x,
            r_s,
            gamma,
            num_witnesses,
            public_inputs,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for PackedGtExpVerifier<F> {
    fn degree(&self) -> usize {
        7
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        use crate::poly::dense_mlpoly::DensePolynomial;
        use jolt_optimizations::get_g_mle;
        use std::any::TypeId;

        // Runtime check that F = Fq
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("PackedGtExp requires F = Fq");
        }

        // Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits)
        // With LowToHigh binding:
        // - Phase 1 (rounds 0-6): bind s variables → challenges[0..7]
        // - Phase 2 (rounds 7-10): bind x variables → challenges[7..11]
        // Each part is reversed to match the sampled challenge order (big-endian convention)
        let r_s_star: Vec<F> = sumcheck_challenges
            .iter()
            .take(self.params.num_step_vars)
            .rev()
            .map(|c| (*c).into())
            .collect();

        let r_x_star: Vec<F> = sumcheck_challenges
            .iter()
            .skip(self.params.num_step_vars)
            .rev()
            .map(|c| (*c).into())
            .collect();

        // Compute eq evaluations for 2-phase
        let r_x_f: Vec<F> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_s_f: Vec<F> = self.r_s.iter().map(|c| (*c).into()).collect();

        let eq_x_eval = EqPolynomial::mle(&r_x_f, &r_x_star);
        let eq_s_eval = EqPolynomial::mle(&r_s_f, &r_s_star);

        // Compute g(r_x*) - g only depends on element variables (4-var)
        let g_eval: F = {
            let g_mle_4var = get_g_mle();
            let g_poly_fq =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_4var));
            let r_x_star_fq: &Vec<Fq> = unsafe { std::mem::transmute(&r_x_star) };
            let g_eval_fq = g_poly_fq.evaluate_dot_product(r_x_star_fq);
            unsafe { std::mem::transmute_copy(&g_eval_fq) }
        };

        // Compute batched constraint value with gamma
        let mut total_constraint = F::zero();
        let mut gamma_power = self.gamma;

        for w in 0..self.num_witnesses {
            // Get all polynomial claims from accumulator (including virtual rho_next)
            let polynomials = vec![
                VirtualPolynomial::PackedGtExpRho(w),
                VirtualPolynomial::PackedGtExpRhoNext(w),
                VirtualPolynomial::PackedGtExpQuotient(w),
            ];
            let claims = get_virtual_claims(accumulator, self.params.sumcheck_id, &polynomials);
            let rho_claim = claims[0];
            let rho_next_claim = claims[1];
            let quotient_claim = claims[2];

            // Compute digit bits and base evaluations directly from public inputs
            let digit_lo = self.public_inputs[w].evaluate_digit_lo_mle(&r_s_star);
            let digit_hi = self.public_inputs[w].evaluate_digit_hi_mle(&r_s_star);
            let base_claim = self.public_inputs[w].evaluate_base_mle(&r_x_star);
            let base2_claim = self.public_inputs[w].evaluate_base2_mle(&r_x_star);
            let base3_claim = self.public_inputs[w].evaluate_base3_mle(&r_x_star);

            let u = digit_lo;
            let v = digit_hi;
            let w0 = (F::one() - u) * (F::one() - v);
            let w1 = u * (F::one() - v);
            let w2 = (F::one() - u) * v;
            let w3 = u * v;
            let base_power = w0 + w1 * base_claim + w2 * base2_claim + w3 * base3_claim;

            // Constraint at challenge point:
            // C(r_s*, r_x*) = rho_next - rho^4 × base_power - quotient × g
            let rho2 = rho_claim * rho_claim;
            let rho4 = rho2 * rho2;
            let constraint_eval = rho_next_claim - rho4 * base_power - quotient_claim * g_eval;

            total_constraint += gamma_power * constraint_eval;
            gamma_power *= self.gamma;
        }

        // Expected output = eq_x(r_x, r_x*) × eq_s(r_s, r_s*) × Σ_w γ^w C_w(r_s*, r_x*)
        eq_x_eval * eq_s_eval * total_constraint
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());

        // Cache openings for all polynomials including virtual rho_next
        // rho and quotient are committed polynomials
        // rho_next is virtual (will be verified via shift sumcheck)
        for w in 0..self.num_witnesses {
            let polynomials = vec![
                VirtualPolynomial::PackedGtExpRho(w),
                VirtualPolynomial::PackedGtExpRhoNext(w),
                VirtualPolynomial::PackedGtExpQuotient(w),
            ];
            append_virtual_openings(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point,
                &polynomials,
            );
        }
    }
}
