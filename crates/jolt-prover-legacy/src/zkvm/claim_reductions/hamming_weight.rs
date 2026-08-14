//! Stage-7 RA address claim reduction — base mode fuses a HammingWeight check
//! into it; lattice mode is the digit-zero claim reduction.
//!
//! ## Background
//!
//! After Stage 6, each ra_i one-hot polynomial has TWO claims at different address points
//! but the SAME cycle point (r_cycle_stage6):
//!
//! 1. **Booleanity claim**: `ra_i(r_addr_bool, r_cycle_stage6)`
//!    - From `BooleanitySumcheck` in Stage 6
//!    - r_addr_bool is shared across all ra_i and across families (instruction/bytecode/ram)
//!
//! 2. **Virtualization claim**: `ra_i(r_addr_virt_i, r_cycle_stage6)`
//!    - For BytecodeRa: from `BytecodeReadRaf` in Stage 6
//!    - For InstructionRa: from `InstructionRaVirtualization` in Stage 6
//!    - For RamRa: from `RamRaVirtualization` in Stage 6
//!    - r_addr_virt_i is DIFFERENT per ra_i (each chunk has its own r_address)
//!
//! All legs operate on the "pushforward" polynomial
//!
//!   `G_i(k) := Σ_j eq(r_cycle, j) · ra_i(k, j)`
//!
//! with different weights, batched by γ into one degree-2 sumcheck over the
//! `log_k_chunk` address variables. After it, each ra_i has a single opening
//! `ra_i(ρ, r_cycle_stage6)` anchoring the stage-8 batched opening.
//!
//! ## Base mode: fused HammingWeight + address reduction (3 legs per ra_i)
//!
//! ```text
//!   Σ_k Σ_i G_i(k) · [
//!       γ^{3i}   · 1                             (HammingWeight)
//!     + γ^{3i+1} · eq(r_addr_bool, k)            (Booleanity reduction)
//!     + γ^{3i+2} · eq(r_addr_virt_i, k)          (Virtualization reduction)
//!   ]
//!   = Σ_i [γ^{3i} · A_i + γ^{3i+1} · claim_bool_i + γ^{3i+2} · claim_virt_i]
//! ```
//!
//! where the activation `A_i` is 1 for instruction/bytecode and the
//! `RamHammingWeight` claim for RAM — the γ^{3i} leg is the anchor tying that
//! claim to the committed column.
//!
//! ## Lattice mode: digit-zero claim reduction (2 legs per ra_i)
//!
//! The commitment omits the digit-zero row, defined as
//! `ra_i(0, t) := M_µ(t) − Σ_{k≠0} ra_i(k, t)` for the family's activation
//! `M_µ` (`specs/digit-zero-virtualization.md`; `M_RAM = Load + Store`, the
//! `RamActivationBooleanity` openings). The weight identity holds by
//! construction, so there is no HammingWeight leg; each remaining leg's
//! digit-zero baseline `w(0)·M̃_µ` folds into the input claim and the sumcheck
//! runs over the committed rows alone with coefficients `w(k) − w(0)`:
//!
//! ```text
//!   Σ_k Σ_i G_i(k) · [ γ^{2i}·(eq(r_addr_bool, k) − eq(r_addr_bool, 0))
//!                    + γ^{2i+1}·(eq(r_addr_virt_i, k) − eq(r_addr_virt_i, 0)) ]
//!   = Σ_i [ γ^{2i}·(claim_bool_i − eq(r_addr_bool, 0)·M̃_µ)
//!         + γ^{2i+1}·(claim_virt_i − eq(r_addr_virt_i, 0)·M̃_µ) ]
//! ```
//!
//! plus one Booleanity leg per balanced-increment column (activation 1) and
//! the fused-increment decode leg, at consecutive γ powers after the RA legs.

use allocative::Allocative;
#[cfg(feature = "prover")]
use rayon::prelude::*;
#[cfg(feature = "prover")]
use tracer::instruction::Cycle;

#[cfg(feature = "prover")]
use crate::curve::JoltCurve;
use crate::field::JoltField;
#[cfg(feature = "prover")]
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
#[cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
#[cfg(feature = "prover")]
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::{
    eq_poly::EqPolynomial,
    opening_proof::{
        AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, SumcheckId,
        BIG_ENDIAN, LITTLE_ENDIAN,
    },
};
#[cfg(feature = "prover")]
use crate::poly::{shared_ra_polys::compute_all_G, unipoly::UniPoly};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
#[cfg(feature = "prover")]
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::{
    config::OneHotParams,
    instruction::CircuitFlags,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

// Degree bound of the sumcheck round polynomials.
// The fused relation includes `G(k) * eq(k)` terms where both are multilinear in k,
// making the round polynomials quadratic (degree 2).
const DEGREE_BOUND: usize = 2;

/// Parameters for the fused HammingWeight + Address Reduction sumcheck.
///
/// This sumcheck handles all three ra_i claim types in a single sumcheck:
/// - HammingWeight: proves Σ_k G_i(k) = H_i
/// - Booleanity: proves Σ_k eq(r_addr_bool, k)·G_i(k) = claim_bool_i
/// - Virtualization: proves Σ_k eq(r_addr_virt_i, k)·G_i(k) = claim_virt_i
///
/// After this sumcheck, each ra_i has a single opening at (ρ, r_cycle_stage6).
#[derive(Allocative, Clone)]
pub struct HammingWeightClaimReductionParams<F: JoltField> {
    /// Batching powers γ^0, γ^1, …. Base layout: 3 per ra polynomial
    /// (γ^{3i} = HW, γ^{3i+1} = Bool, γ^{3i+2} = Virt). Digit-zero (lattice)
    /// layout: 2 per ra polynomial (γ^{2i} = Bool, γ^{2i+1} = Virt), then one
    /// per increment column, then the decode power.
    pub gamma_powers: Vec<F>,
    /// Shared r_cycle from Booleanity (all ra claims share this)
    pub r_cycle: Vec<F::Challenge>,
    /// Shared r_address from Booleanity (all families share this now)
    pub r_addr_bool: Vec<F::Challenge>,
    /// r_address values from Virtualization/ReadRaf sumcheck for each ra_i (N total)
    /// Each ra_i has different r_addr because chunks are bound sequentially
    pub r_addr_virt: Vec<Vec<F::Challenge>>,
    /// Per-family activation `M̃_µ(r_cycle)`: 1 for instruction/bytecode; for
    /// RAM, base mode uses the `RamHammingWeight` claim (also the HammingWeight
    /// leg's expected sum) and digit-zero mode uses `Load + Store` (the
    /// `RamActivationBooleanity` openings).
    pub activations: Vec<F>,
    /// Booleanity claims for each ra_i
    pub claims_bool: Vec<F>,
    /// Virtualization claims for each ra_i
    pub claims_virt: Vec<F>,
    /// log_2(k_chunk) - number of sumcheck rounds
    pub log_k_chunk: usize,
    /// Polynomial labels: InstructionRa(0..d), BytecodeRa(0..d), RamRa(0..d)
    pub polynomial_types: Vec<CommittedPolynomial>,
    /// Lattice-only balanced-digit Booleanity claims, chunks followed by carry.
    pub inc_booleanity_claims: Vec<F>,
    /// Lattice-only fused-increment claim.
    pub fused_inc_claim: Option<F>,
    /// Place weights for the increment columns, chunks followed by `2^64` for carry.
    pub inc_weights: Vec<F>,
    /// Lattice-only: the digit-zero row is omitted from the commitment and
    /// reconstructed from the activation, so each leg's digit-zero baseline
    /// `w(0)·M̃_µ` is folded into the input claim and there is no HammingWeight
    /// leg (`Σ_k ra_i(k,t) = M_µ(t)` holds by construction —
    /// `specs/digit-zero-virtualization.md`). Base mode keeps the plain form.
    pub digit_zero: bool,
    /// Lattice-only `eq(0, r_addr_bool)` — the booleanity legs' `w(0)`.
    pub eq_bool_at_digit_zero: F,
    /// Lattice-only `eq(0, r_addr_virt[i])` per polynomial — the
    /// virtualization legs' `w(0)`.
    pub eq_virt_at_digit_zero: Vec<F>,
}

impl<F: JoltField> HammingWeightClaimReductionParams<F> {
    /// Create base-mode params by fetching claims from Stage 6 and sampling
    /// the batching challenge.
    ///
    /// Fetches:
    /// - the RAM activation (the `RamHammingWeight` claim, also the
    ///   HammingWeight legs' expected sum)
    /// - Booleanity claims (r_addr shared across all families from Booleanity sumcheck)
    /// - Virtualization claims (r_addr different per ra_i)
    pub fn new(
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // RAM activation: now in Stage 6, so shares r_cycle_stage6
        let ram_activation = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .1;
        Self::build(one_hot_params, accumulator, transcript, ram_activation, 3)
    }

    /// Shared constructor body; `powers_per_ra` is 3 in base mode (HW, Bool,
    /// Virt) and 2 in digit-zero mode (Bool, Virt — no HammingWeight leg).
    fn build(
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        ram_activation: F,
        powers_per_ra: usize,
    ) -> Self {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let N = instruction_d + bytecode_d + ram_d;
        let log_k_chunk = one_hot_params.log_k_chunk;

        // Build polynomial types list
        let mut polynomial_types = Vec::with_capacity(N);
        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
        }

        // Sample batching challenge γ and compute powers
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = Vec::with_capacity(powers_per_ra * N);
        let mut power = F::one();
        for _ in 0..(powers_per_ra * N) {
            gamma_powers.push(power);
            power *= gamma;
        }

        // Fetch r_addr_bool and r_cycle from Booleanity opening point.
        // The claims from Booleanity are at (ρ_addr, ρ_cycle) where both are sumcheck challenges.
        //
        // For the reduction's G to satisfy: Σ_k G_i(k) * eq(ρ_addr, k) = claims_bool[i] = ra_i(ρ_addr, ρ_cycle)
        // We need: G_i(k) = Σ_j eq(ρ_cycle, j) * ra_i(k, j)
        //
        // The opening point is stored in BE format (after normalize_opening_point reversed it).
        let (unified_bool_point, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::Booleanity,
        );
        // Keep both segments in BE: this matches the convention expected by `EqPolynomial::evals`
        // and `GruenSplitEqPolynomial` when used with `BindingOrder::LowToHigh` (LSB bound first).
        let r_addr_bool = unified_bool_point.r[..log_k_chunk].to_vec();
        let r_cycle: Vec<F::Challenge> = unified_bool_point.r[log_k_chunk..].to_vec();

        // Fetch claims for each ra_i
        let mut r_addr_virt = Vec::with_capacity(N);
        let mut activations = Vec::with_capacity(N);
        let mut claims_bool = Vec::with_capacity(N);
        let mut claims_virt = Vec::with_capacity(N);

        for poly_type in polynomial_types.iter() {
            // Get virtualization sumcheck ID and activation based on polynomial type
            let (virt_sumcheck_id, activation) = match poly_type {
                CommittedPolynomial::InstructionRa(_) => {
                    (SumcheckId::InstructionRaVirtualization, F::one())
                }
                CommittedPolynomial::BytecodeRa(_) => (SumcheckId::BytecodeReadRaf, F::one()),
                // For Ram: shared across all RAM chunks
                CommittedPolynomial::RamRa(_) => (SumcheckId::RamRaVirtualization, ram_activation),
                _ => unreachable!(),
            };
            activations.push(activation);

            // Booleanity claim (from booleanity sumcheck)
            let (_, bool_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity);
            claims_bool.push(bool_claim);

            // Virtualization claim (with per-polynomial r_addr)
            let (virt_point, virt_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, virt_sumcheck_id);
            r_addr_virt.push(virt_point.r[..log_k_chunk].to_vec());
            claims_virt.push(virt_claim);
        }

        Self {
            gamma_powers,
            r_cycle,
            r_addr_bool,
            r_addr_virt,
            activations,
            claims_bool,
            claims_virt,
            log_k_chunk,
            polynomial_types,
            inc_booleanity_claims: Vec::new(),
            fused_inc_claim: None,
            inc_weights: Vec::new(),
            digit_zero: false,
            eq_bool_at_digit_zero: F::zero(),
            eq_virt_at_digit_zero: Vec::new(),
        }
    }

    pub fn new_lattice(
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // The RAM activation is `M_RAM = Load + Store` — the two flag openings
        // the `RamActivationBooleanity` member produced at the stage-6b cycle
        // point (`specs/digit-zero-virtualization.md`).
        let load = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Load),
                SumcheckId::RamActivationBooleanity,
            )
            .1;
        let store = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Store),
                SumcheckId::RamActivationBooleanity,
            )
            .1;
        let mut params = Self::build(one_hot_params, accumulator, transcript, load + store, 2);
        params.digit_zero = true;
        params.eq_bool_at_digit_zero = EqPolynomial::<F>::evals(&params.r_addr_bool)[0];
        params.eq_virt_at_digit_zero = params
            .r_addr_virt
            .iter()
            .map(|point| EqPolynomial::<F>::evals(point)[0])
            .collect();
        let chunk_count = 64 / one_hot_params.log_k_chunk;
        params.inc_booleanity_claims = (0..chunk_count)
            .map(|index| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BalancedIncDigit(index),
                        SumcheckId::Booleanity,
                    )
                    .1
            })
            .chain(core::iter::once(
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BalancedIncCarry,
                        SumcheckId::Booleanity,
                    )
                    .1,
            ))
            .collect();
        params.fused_inc_claim = Some(
            accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::FusedInc,
                    SumcheckId::BytecodeReadRaf,
                )
                .1,
        );
        params.inc_weights = (0..chunk_count)
            .map(|index| F::from_u128(1u128 << (one_hot_params.log_k_chunk * index)))
            .chain(core::iter::once(F::from_u128(1u128 << 64)))
            .collect();

        // Extend to the digit-zero layout: one Booleanity power per increment
        // column, then the decode power.
        let gamma = params.gamma_powers[1];
        let total_powers =
            2 * params.polynomial_types.len() + params.inc_booleanity_claims.len() + 1;
        let mut power = params.gamma_powers.last().copied().unwrap_or(F::one()) * gamma;
        while params.gamma_powers.len() < total_powers {
            params.gamma_powers.push(power);
            power *= gamma;
        }
        params
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingWeightClaimReductionParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Base: Σ_i (γ^{3i}·A_i + γ^{3i+1}·claim_bool_i + γ^{3i+2}·claim_virt_i).
        // Digit-zero (lattice): no HammingWeight leg — each remaining leg's
        // digit-zero baseline `w(0)·M̃_µ` is folded in:
        // γ^{2i}·(claim_bool_i − eq(0,r_bool)·M̃_µ) +
        // γ^{2i+1}·(claim_virt_i − eq(0,r_virt_i)·M̃_µ).
        let mut claim = F::zero();
        for i in 0..self.polynomial_types.len() {
            if self.digit_zero {
                claim += self.gamma_powers[2 * i]
                    * (self.claims_bool[i] - self.eq_bool_at_digit_zero * self.activations[i]);
                claim += self.gamma_powers[2 * i + 1]
                    * (self.claims_virt[i] - self.eq_virt_at_digit_zero[i] * self.activations[i]);
            } else {
                claim += self.gamma_powers[3 * i] * self.activations[i];
                claim += self.gamma_powers[3 * i + 1] * self.claims_bool[i];
                claim += self.gamma_powers[3 * i + 2] * self.claims_virt[i];
            }
        }
        let offset = 2 * self.polynomial_types.len();
        for (index, booleanity) in self.inc_booleanity_claims.iter().enumerate() {
            // Increment columns are lattice-only, so `digit_zero` holds here;
            // their activation is the constant 1.
            claim += self.gamma_powers[offset + index] * (*booleanity - self.eq_bool_at_digit_zero);
        }
        if let Some(fused_inc) = self.fused_inc_claim {
            claim += self.gamma_powers[offset + self.inc_booleanity_claims.len()] * fused_inc;
        }
        claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Address challenges come from sumcheck (little-endian), convert to big-endian
        // Then concatenate with r_cycle to form full opening point
        let r_addr: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness();
        let full_point = [r_addr.r.as_slice(), self.r_cycle.as_slice()].concat();
        OpeningPoint::<BIG_ENDIAN, F>::new(full_point)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        let n = self.polynomial_types.len();
        let mut terms = Vec::new();
        let mut challenge_idx = 0;

        for i in 0..n {
            let poly_type = self.polynomial_types[i];

            let virt_sumcheck_id = match poly_type {
                CommittedPolynomial::InstructionRa(_) => SumcheckId::InstructionRaVirtualization,
                CommittedPolynomial::BytecodeRa(_) => SumcheckId::BytecodeReadRaf,
                CommittedPolynomial::RamRa(_) => SumcheckId::RamRaVirtualization,
                _ => unreachable!(),
            };

            // HW claim term: γ^{3i} * hw_claim_i
            // For RAM, hw_claim is RamHammingWeight opening; for others, it's F::one()
            match poly_type {
                CommittedPolynomial::RamRa(_) => {
                    let hw_opening = OpeningId::virt(
                        VirtualPolynomial::RamHammingWeight,
                        SumcheckId::RamHammingBooleanity,
                    );
                    terms.push(ProductTerm::scaled(
                        ValueSource::Challenge(challenge_idx),
                        vec![ValueSource::Opening(hw_opening)],
                    ));
                }
                _ => {
                    // For instruction/bytecode, hw_claim = 1, so term is just γ^{3i}
                    terms.push(ProductTerm::single(ValueSource::Challenge(challenge_idx)));
                }
            }
            challenge_idx += 1;

            // Bool claim term: γ^{3i+1} * bool_opening_i
            let bool_opening = OpeningId::committed(poly_type, SumcheckId::Booleanity);
            terms.push(ProductTerm::scaled(
                ValueSource::Challenge(challenge_idx),
                vec![ValueSource::Opening(bool_opening)],
            ));
            challenge_idx += 1;

            // Virt claim term: γ^{3i+2} * virt_opening_i
            let virt_opening = OpeningId::committed(poly_type, virt_sumcheck_id);
            terms.push(ProductTerm::scaled(
                ValueSource::Challenge(challenge_idx),
                vec![ValueSource::Opening(virt_opening)],
            ));
            challenge_idx += 1;
        }

        InputClaimConstraint::sum_of_products(terms)
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        let n = self.polynomial_types.len();
        let mut values = Vec::with_capacity(3 * n);

        for i in 0..n {
            // γ^{3i} for hw term
            values.push(self.gamma_powers[3 * i]);
            // γ^{3i+1} for bool term
            values.push(self.gamma_powers[3 * i + 1]);
            // γ^{3i+2} for virt term
            values.push(self.gamma_powers[3 * i + 2]);
        }

        values
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let N = self.polynomial_types.len();

        let terms: Vec<ProductTerm> = (0..N)
            .map(|i| {
                let opening = OpeningId::committed(
                    self.polynomial_types[i],
                    SumcheckId::HammingWeightClaimReduction,
                );
                ProductTerm::scaled(
                    ValueSource::Challenge(i),
                    vec![ValueSource::Opening(opening)],
                )
            })
            .collect();

        Some(OutputClaimConstraint::sum_of_products(terms))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let N = self.polynomial_types.len();

        let rho_rev: Vec<F::Challenge> = sumcheck_challenges.iter().cloned().rev().collect();
        let eq_bool_eval: F = EqPolynomial::mle(&rho_rev, &self.r_addr_bool);

        (0..N)
            .map(|i| {
                let eq_virt_eval: F = EqPolynomial::mle(&rho_rev, &self.r_addr_virt[i]);

                let gamma_hw = self.gamma_powers[3 * i];
                let gamma_bool = self.gamma_powers[3 * i + 1];
                let gamma_virt = self.gamma_powers[3 * i + 2];

                gamma_hw + gamma_bool * eq_bool_eval + gamma_virt * eq_virt_eval
            })
            .collect()
    }
}

/// Prover for the fused HammingWeight + Address Reduction sumcheck.
///
/// This sumcheck combines all three ra_i claim types (HammingWeight, Booleanity,
/// Virtualization) into a single degree-2 sumcheck over log_k_chunk rounds.
///
/// Memory optimization: eq_bool is shared across all families (1 polynomial, thanks
/// to Booleanity), while eq_virt requires one per ra_i (N polynomials).
#[cfg(feature = "prover")]
#[cfg(feature = "prover")]
#[derive(Allocative)]
pub struct HammingWeightClaimReductionProver<F: JoltField> {
    /// G_i polynomials (pushforward of ra_i over r_cycle)
    /// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
    G: Vec<MultilinearPolynomial<F>>,
    /// eq(r_addr_bool, ·) shared across all families (single polynomial)
    eq_bool: MultilinearPolynomial<F>,
    /// eq(r_addr_virt_i, ·) for each ra polynomial (N total)
    eq_virt: Vec<MultilinearPolynomial<F>>,
    /// Lattice-only decode-weight table: the centered digit value
    /// `balanced_inc_value` over the `k_chunk` digit values. Present iff
    /// `params.digit_zero`.
    inc_value: Option<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    pub params: HammingWeightClaimReductionParams<F>,
}

#[cfg(feature = "prover")]
impl<F: JoltField> HammingWeightClaimReductionProver<F> {
    fn from_G(
        params: HammingWeightClaimReductionParams<F>,
        G: Vec<Vec<F>>,
        inc_value: Option<MultilinearPolynomial<F>>,
    ) -> Self {
        let G = G.into_iter().map(MultilinearPolynomial::from).collect();
        let eq_bool = MultilinearPolynomial::from(EqPolynomial::evals(&params.r_addr_bool));
        let eq_virt = params
            .r_addr_virt
            .iter()
            .map(|point| MultilinearPolynomial::from(EqPolynomial::evals(point)))
            .collect();
        Self {
            G,
            eq_bool,
            eq_virt,
            inc_value,
            params,
        }
    }

    /// Initialize the prover by computing all G_i polynomials.
    /// Returns (prover, ram_hw_claims) where ram_hw_claims contains the computed H_i for RAM polynomials.
    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::initialize")]
    pub fn initialize<C, PCS>(
        params: HammingWeightClaimReductionParams<F>,
        trace: &[Cycle],
        preprocessing: &JoltProverPreprocessing<F, C, PCS>,
        one_hot_params: &OneHotParams,
    ) -> Self
    where
        C: JoltCurve<F = F>,
        PCS: CommitmentScheme<Field = F>,
    {
        // Compute all G_i polynomials via streaming.
        // `params.r_cycle` is in BIG_ENDIAN (OpeningPoint) convention.
        let G_vecs = compute_all_G::<F>(
            trace,
            &preprocessing.materialized_program().bytecode,
            &preprocessing.shared.memory_layout,
            one_hot_params,
            &params.r_cycle,
        );
        Self::from_G(params, G_vecs, None)
    }

    #[cfg(feature = "akita")]
    pub fn initialize_lattice<C, PCS>(
        params: HammingWeightClaimReductionParams<F>,
        trace: &[Cycle],
        preprocessing: &JoltProverPreprocessing<F, C, PCS>,
        one_hot_params: &OneHotParams,
        one_hot_columns: &[std::sync::Arc<Vec<Option<u8>>>],
    ) -> Self
    where
        C: JoltCurve<F = F>,
        PCS: CommitmentScheme<Field = F>,
    {
        // `params.r_cycle` is BIG_ENDIAN, so the head half of the point
        // indexes the high cycle bits: eq(r, j) = e_hi[j >> lo] · e_lo[j & mask].
        let lo_bits = params.r_cycle.len() / 2;
        let hi_bits = params.r_cycle.len() - lo_bits;
        let (r_hi, r_lo) = params.r_cycle.split_at(hi_bits);
        let (e_hi, e_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );
        let increment_g = crate::subprotocols::booleanity::one_hot_pushforwards(
            one_hot_columns,
            &e_hi,
            &e_lo,
            1usize << params.log_k_chunk,
        );
        let mut G = compute_all_G::<F>(
            trace,
            &preprocessing.materialized_program().bytecode,
            &preprocessing.shared.memory_layout,
            one_hot_params,
            &params.r_cycle,
        );
        G.extend(increment_g);
        // The commitment omits the digit-zero row; the sumcheck runs over the
        // committed nonzero-digit rows, so the pushforwards must agree (the
        // digit-zero baselines live in the input claim instead — see
        // `input_claim`).
        for polynomial in &mut G {
            polynomial[0] = F::zero();
        }

        let k_chunk = 1usize << params.log_k_chunk;
        let half = k_chunk / 2;
        let inc_value: Vec<F> = (0..k_chunk)
            .map(|lane| {
                if lane < half {
                    F::from_i128(lane as i128)
                } else {
                    F::from_i128(lane as i128 - k_chunk as i128)
                }
            })
            .collect();
        Self::from_G(params, G, Some(MultilinearPolynomial::from(inc_value)))
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for HammingWeightClaimReductionProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let N = self.params.polynomial_types.len();
        let half_n = self.G[0].len() / 2;

        let mut evals = [F::zero(); DEGREE_BOUND];

        for j in 0..half_n {
            // eq_bool is shared across all polynomials, compute once per j
            let eq_b_evals = self
                .eq_bool
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

            for i in 0..N {
                let g_evals =
                    self.G[i].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_v_evals = self.eq_virt[i]
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                for k in 0..DEGREE_BOUND {
                    // Digit-zero: `w(k) − w(0)` at γ^{2i}/γ^{2i+1} — every
                    // digit-zero baseline is folded into the input claim, so
                    // the summand is purely sparse. Base: γ^{3i}·1 (HW) +
                    // γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt.
                    let coefficient = if self.params.digit_zero {
                        self.params.gamma_powers[2 * i]
                            * (eq_b_evals[k] - self.params.eq_bool_at_digit_zero)
                            + self.params.gamma_powers[2 * i + 1]
                                * (eq_v_evals[k] - self.params.eq_virt_at_digit_zero[i])
                    } else {
                        self.params.gamma_powers[3 * i]
                            + self.params.gamma_powers[3 * i + 1] * eq_b_evals[k]
                            + self.params.gamma_powers[3 * i + 2] * eq_v_evals[k]
                    };
                    evals[k] += g_evals[k] * coefficient;
                }
            }

            if let Some(inc_value) = &self.inc_value {
                let inc_value_evals =
                    inc_value.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let offset = 2 * N;
                let decode =
                    self.params.gamma_powers[offset + self.params.inc_booleanity_claims.len()];
                for index in 0..self.params.inc_booleanity_claims.len() {
                    let g_evals = self.G[N + index]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let booleanity = self.params.gamma_powers[offset + index];
                    let weight = self.params.inc_weights[index];
                    for evaluation in 0..DEGREE_BOUND {
                        evals[evaluation] += g_evals[evaluation]
                            * (booleanity
                                * (eq_b_evals[evaluation] - self.params.eq_bool_at_digit_zero)
                                + decode * weight * inc_value_evals[evaluation]);
                    }
                }
            }
        }

        // `from_evals_and_hint` expects [S(0), S(2), ...] (S(1) is reconstructed from the hint).
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(inc_value) = self.inc_value.as_mut() {
            inc_value.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        // Bind all polynomials in parallel
        rayon::scope(|s| {
            s.spawn(|_| {
                self.G.par_iter_mut().for_each(|g| {
                    g.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
            s.spawn(|_| {
                // Single eq_bool polynomial (shared across all families)
                self.eq_bool.bind_parallel(r_j, BindingOrder::LowToHigh);
            });
            s.spawn(|_| {
                self.eq_virt.par_iter_mut().for_each(|eq| {
                    eq.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
        });
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let N = self.params.polynomial_types.len();

        // Extract r_address portion (just the sumcheck challenges, converted to big-endian)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;

        for i in 0..N {
            // Final claim is G_i(ρ) where ρ is the sumcheck challenges
            let claim = self.G[i].final_sumcheck_claim();

            // All three claim types (HW, Bool, Virt) collapse to this single opening
            accumulator.append_sparse(
                vec![self.params.polynomial_types[i]],
                SumcheckId::HammingWeightClaimReduction,
                r_address.clone(),
                self.params.r_cycle.clone(),
                vec![claim],
            );
        }
        let chunk_count = self.params.inc_booleanity_claims.len().saturating_sub(1);
        for index in 0..chunk_count {
            accumulator.append_sparse(
                vec![CommittedPolynomial::BalancedIncDigit(index)],
                SumcheckId::HammingWeightClaimReduction,
                r_address.clone(),
                self.params.r_cycle.clone(),
                vec![self.G[N + index].final_sumcheck_claim()],
            );
        }
        if !self.params.inc_booleanity_claims.is_empty() {
            accumulator.append_sparse(
                vec![CommittedPolynomial::BalancedIncCarry],
                SumcheckId::HammingWeightClaimReduction,
                r_address,
                self.params.r_cycle.clone(),
                vec![self.G[N + chunk_count].final_sumcheck_claim()],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct HammingWeightClaimReductionVerifier<F: JoltField> {
    params: HammingWeightClaimReductionParams<F>,
}

impl<F: JoltField> HammingWeightClaimReductionVerifier<F> {
    /// Create verifier. r_cycle and r_addr_bool are extracted from Booleanity opening.
    ///
    /// Takes a generic `OpeningAccumulator` to support both real verification
    /// (`VerifierOpeningAccumulator`) and symbolic transpilation (`AstOpeningAccumulator`).
    pub fn new(
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            HammingWeightClaimReductionParams::new(one_hot_params, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for HammingWeightClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let N = self.params.polynomial_types.len();

        // When binding with LowToHigh, challenges[j] binds index bit j which corresponds to
        // r[n-1-j] in EqPolynomial::evals table. So after binding, the result is eq(r, reversed_challenges).
        // To match, compute mle(r, reversed_challenges) or equivalently mle(reversed_challenges, r).
        let rho_rev: Vec<F::Challenge> = sumcheck_challenges.iter().cloned().rev().collect();

        // eq_bool_eval is shared across all polynomials (unified booleanity)
        let eq_bool_eval = EqPolynomial::mle(&rho_rev, &self.params.r_addr_bool);

        let mut output_claim = F::zero();

        for i in 0..N {
            // r_addr values are in BIG_ENDIAN. Compute eq(r_addr, rho) = mle(rho_reversed, r_addr).
            let eq_virt_eval = EqPolynomial::mle(&rho_rev, &self.params.r_addr_virt[i]);

            // Fetch G_i(ρ) from accumulator (prover provided this)
            let (_, g_i_claim) = accumulator.get_committed_polynomial_opening(
                self.params.polynomial_types[i],
                SumcheckId::HammingWeightClaimReduction,
            );

            // γ^{3i} · G_i(ρ) + γ^{3i+1} · eq_bool(ρ) · G_i(ρ) + γ^{3i+2} · eq_virt(ρ) · G_i(ρ)
            let gamma_hw = self.params.gamma_powers[3 * i];
            let gamma_bool = self.params.gamma_powers[3 * i + 1];
            let gamma_virt = self.params.gamma_powers[3 * i + 2];

            // G_i(ρ) · (γ_hw + γ_bool·eq_bool(ρ) + γ_virt·eq_virt(ρ))
            output_claim +=
                g_i_claim * (gamma_hw + gamma_bool * eq_bool_eval + gamma_virt * eq_virt_eval);
        }

        output_claim
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let N = self.params.polynomial_types.len();

        // Compute full opening point (r_address || r_cycle)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;
        let full_point = [r_address.as_slice(), self.params.r_cycle.as_slice()].concat();

        for i in 0..N {
            accumulator.append_sparse(
                vec![self.params.polynomial_types[i]],
                SumcheckId::HammingWeightClaimReduction,
                full_point.clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests comparing compute_all_G output against naive computation
    // TODO: Add tests for sumcheck correctness
}
