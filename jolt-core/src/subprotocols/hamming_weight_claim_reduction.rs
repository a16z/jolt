//! Fused HammingWeight + RA Address Reduction Sumcheck
//!
//! This module implements a **fused** sumcheck combining:
//! 1. **HammingWeight sumcheck**: proves that each ra_i sums to its expected value over the addresses
//!    (1 if there is an access, 0 if not)
//! 2. **Address reduction sumcheck**: aligns the address portion of Booleanity and
//!    Virtualization claims to a common opening point
//!
//! Both sumchecks operate on the same G_i polynomial, just with different weights,
//! enabling fusion with no degree overhead.
//!
//! ## Background
//!
//! After Stage 6, each ra_i one-hot polynomial has TWO claims at different address points
//! but the SAME cycle point (r_cycle_stage6):
//!
//! 1. **Booleanity claim**: `ra_i(r_addr_bool_family, r_cycle_stage6)`
//!    - From `BooleanitySumcheck` in Stage 6
//!    - r_addr_bool is shared across all ra_i within a family (instruction/bytecode/ram)
//!
//! 2. **Virtualization claim**: `ra_i(r_addr_virt_i, r_cycle_stage6)`
//!    - For BytecodeRa: from `BytecodeReadRaf` in Stage 6
//!    - For InstructionRa: from `InstructionRaVirtualization` in Stage 6
//!    - For RamRa: from `RamRaVirtualization` in Stage 6
//!    - r_addr_virt_i is DIFFERENT per ra_i (each chunk has its own r_address)
//!
//! The HammingWeight sumcheck would normally run separately, producing its own
//! address challenges. By fusing it with the address reduction, we ensure all
//! claims collapse to a single opening point.
//!
//! ## Fusion Insight
//!
//! Define the "pushforward" polynomial:
//!
//!   `G_i(k) := Σ_j eq(r_cycle, j) · ra_i(k, j)`
//!
//! All claim types operate on this same G_i, just with different weights:
//! - **HammingWeight**: weight = 1 (constant) → proves Σ_k G_i(k) = H_i
//! - **Booleanity reduction**: weight = eq(r_addr_bool, k) → reduces claim to common point
//! - **Virtualization reduction**: weight = eq(r_addr_virt_i, k) → reduces claim to common point
//!
//! By batching with γ, we fuse HammingWeight with address reduction into one sumcheck!
//!
//! ## Fused Sumcheck Relation
//!
//! Let N = total number of ra polynomials = instruction_d + bytecode_d + ram_d.
//! Let family(i) ∈ {0, 1, 2} denote which family ra_i belongs to (instruction/bytecode/ram).
//!
//! The fused sumcheck proves:
//!
//! ```text
//!   Σ_k Σ_i G_i(k) · [
//!       γ^{3i}   · 1                              (HammingWeight)
//!     + γ^{3i+1} · eq(r_addr_bool_{family(i)}, k)  (Booleanity reduction)
//!     + γ^{3i+2} · eq(r_addr_virt_i, k)           (Virtualization reduction)
//!   ]
//!   = Σ_i [γ^{3i} · H_i + γ^{3i+1} · claim_bool_i + γ^{3i+2} · claim_virt_i]
//! ```
//!
//! ## eq Polynomial Optimization
//!
//! - **eq_bool**: 3 polynomials total (shared per family)
//!   - All ra_i within a family share the same r_addr_bool from booleanity sumcheck
//! - **eq_virt**: N polynomials (one per ra_i)
//!   - Each ra_i has different r_addr_virt from virtualization (chunks bound sequentially)
//!
//! ## Degree Analysis
//!
//! Each round polynomial has degree 2:
//! - G_i(k) contributes degree 1
//! - eq(r_addr, k) contributes degree 1 (or 0 for HammingWeight constant term)
//! - Maximum: 1 + 1 = 2
//!
//! Same degree as address reduction alone - fusion is free!
//!
//! ## After This Sumcheck
//!
//! Let ρ be the challenges from this sumcheck (r_address_stage7). Each ra_i has a SINGLE opening:
//!
//!   `ra_i(ρ, r_cycle_stage6)`
//!
//! The verifier computes expected claims using the single opening G_i(ρ):
//! - HammingWeight: G_i(ρ)
//! - Booleanity: eq(r_addr_bool, ρ) · G_i(ρ)
//! - Virtualization: eq(r_addr_virt_i, ρ) · G_i(ρ)
//!
//! All three claim types collapse to a single committed polynomial opening per ra_i!
//!
//! ## Implementation Details
//!
//! See `OPENING_REDUCTION_REFACTOR.md` for the full implementation plan and status.

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::remap_address;
use common::constants::XLEN;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Maximum number of instruction ra chunks (lookup index splits into at most 32 chunks)
const MAX_INSTRUCTION_D: usize = 32;
/// Maximum number of bytecode ra chunks (PC splits into at most 6 chunks)
const MAX_BYTECODE_D: usize = 6;
/// Maximum number of ram ra chunks (address splits into at most 8 chunks)
const MAX_RAM_D: usize = 8;

/// Stores the chunk indices for all ra polynomials for a single cycle.
/// Uses fixed-size arrays to avoid heap allocation in hot loop.
struct RaIndices {
    /// InstructionRa chunk indices (always present)
    instruction_ra: [u16; MAX_INSTRUCTION_D],
    /// BytecodeRa chunk indices (always present)
    bytecode_ra: [u16; MAX_BYTECODE_D],
    /// RamRa chunk indices (may be None for non-memory cycles)
    ram_ra: [Option<u16>; MAX_RAM_D],
}

impl RaIndices {
    /// Compute all ra chunk indices for a single cycle from trace data.
    /// Only the first `instruction_d`, `bytecode_d`, `ram_d` elements are valid.
    #[inline]
    fn from_cycle<F: JoltField, PCS: CommitmentScheme<Field = F>>(
        cycle: &Cycle,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Assert bounds at runtime (should be checked once at init, but defensive here)
        debug_assert!(
            one_hot_params.instruction_d <= MAX_INSTRUCTION_D,
            "instruction_d {} exceeds MAX_INSTRUCTION_D {}",
            one_hot_params.instruction_d,
            MAX_INSTRUCTION_D
        );
        debug_assert!(
            one_hot_params.bytecode_d <= MAX_BYTECODE_D,
            "bytecode_d {} exceeds MAX_BYTECODE_D {}",
            one_hot_params.bytecode_d,
            MAX_BYTECODE_D
        );
        debug_assert!(
            one_hot_params.ram_d <= MAX_RAM_D,
            "ram_d {} exceeds MAX_RAM_D {}",
            one_hot_params.ram_d,
            MAX_RAM_D
        );

        // 1. InstructionRa: from lookup index
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        let mut instruction_ra = [0u16; MAX_INSTRUCTION_D];
        for i in 0..one_hot_params.instruction_d {
            instruction_ra[i] = one_hot_params.lookup_index_chunk(lookup_index, i);
        }

        // 2. BytecodeRa: from PC
        let pc = preprocessing.bytecode.get_pc(cycle);
        let mut bytecode_ra = [0u16; MAX_BYTECODE_D];
        for i in 0..one_hot_params.bytecode_d {
            bytecode_ra[i] = one_hot_params.bytecode_pc_chunk(pc, i);
        }

        // 3. RamRa: from remapped address (may be None for non-memory cycles)
        let address = remap_address(
            cycle.ram_access().address() as u64,
            &preprocessing.memory_layout,
        );
        let mut ram_ra = [None; MAX_RAM_D];
        for i in 0..one_hot_params.ram_d {
            ram_ra[i] = address.map(|a| one_hot_params.ram_address_chunk(a, i));
        }

        Self {
            instruction_ra,
            bytecode_ra,
            ram_ra,
        }
    }
}

/// Asserts that the one_hot_params dimensions are within bounds.
/// Call this once at the start of compute_all_G to catch issues early.
#[inline]
fn assert_ra_bounds(one_hot_params: &OneHotParams) {
    assert!(
        one_hot_params.instruction_d <= MAX_INSTRUCTION_D,
        "instruction_d {} exceeds MAX_INSTRUCTION_D {}",
        one_hot_params.instruction_d,
        MAX_INSTRUCTION_D
    );
    assert!(
        one_hot_params.bytecode_d <= MAX_BYTECODE_D,
        "bytecode_d {} exceeds MAX_BYTECODE_D {}",
        one_hot_params.bytecode_d,
        MAX_BYTECODE_D
    );
    assert!(
        one_hot_params.ram_d <= MAX_RAM_D,
        "ram_d {} exceeds MAX_RAM_D {}",
        one_hot_params.ram_d,
        MAX_RAM_D
    );
}

/// Computes all G_i polynomials in a single streaming pass over the trace.
///
/// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
///
/// For one-hot ra polynomials:
/// G_i(k) = Σ_{j: addr_chunk_i(j) = k} eq(r_cycle, j)
#[tracing::instrument(skip_all, name = "HammingWeightClaimReduction::compute_all_G")]
pub fn compute_all_G<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    one_hot_params: &OneHotParams,
) -> Vec<Vec<F>> {
    // Verify bounds once at the start
    assert_ra_bounds(one_hot_params);

    let K = one_hot_params.k_chunk;
    let instruction_d = one_hot_params.instruction_d;
    let bytecode_d = one_hot_params.bytecode_d;
    let ram_d = one_hot_params.ram_d;
    let N = instruction_d + bytecode_d + ram_d; // Total number of ra polynomials
    let T = trace.len();

    // Build split-eq polynomial over r_cycle
    // This gives us E_out, E_in tables and the current_w (last challenge)
    let split_eq = GruenSplitEqPolynomial::<F>::new(r_cycle, BindingOrder::LowToHigh);

    let E_in = split_eq.E_in_current();
    let E_out = split_eq.E_out_current();
    let w_current = split_eq.get_current_w();
    let factor_0 = F::one() - w_current;
    let factor_1: F = w_current.into();

    let in_len = E_in.len();
    let x_in_bits = in_len.log_2();

    // Precompute merged inner weights: [E_in[x_in] * (1-w), E_in[x_in] * w] for all x_in
    // This avoids recomputing the product for each (x_out, x_in) pair
    let merged_in_unreduced: Vec<F::Unreduced<9>> = {
        let mut merged: Vec<F::Unreduced<9>> = unsafe_allocate_zero_vec(2 * in_len);
        merged
            .par_chunks_exact_mut(2)
            .zip(E_in.par_iter())
            .for_each(|(chunk, &low)| {
                chunk[0] = low.mul_unreduced::<9>(factor_0);
                chunk[1] = low.mul_unreduced::<9>(factor_1);
            });
        merged
    };

    // Parallel fold over E_out indices
    // Each thread maintains local G arrays for all N polynomials
    let G: Vec<Vec<F>> = E_out
        .par_iter()
        .enumerate()
        .fold(
            || vec![unsafe_allocate_zero_vec::<F>(K); N],
            |mut partial_G, (x_out, &e_out)| {
                // Local unreduced accumulators for this x_out chunk
                let mut local_unreduced: Vec<Vec<F::Unreduced<9>>> =
                    vec![unsafe_allocate_zero_vec(K); N];

                // Track which indices were touched for efficient reduction
                let mut touched_flags: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(K); N];

                let x_out_base = x_out << (x_in_bits + 1);

                // Sequential over x_in
                for x_in in 0..in_len {
                    let j0 = x_out_base + (x_in << 1);
                    let j1 = j0 + 1;
                    let off = 2 * x_in;
                    let add0_unr = merged_in_unreduced[off];
                    let add1_unr = merged_in_unreduced[off + 1];

                    // Process cycle j0 (last_bit = 0)
                    if j0 < T {
                        let ra_indices =
                            RaIndices::from_cycle(&trace[j0], preprocessing, one_hot_params);

                        // InstructionRa contributions
                        for i in 0..instruction_d {
                            let k = ra_indices.instruction_ra[i] as usize;
                            if !touched_flags[i].contains(k) {
                                touched_flags[i].insert(k);
                            }
                            local_unreduced[i][k] += add0_unr;
                        }

                        // BytecodeRa contributions
                        for i in 0..bytecode_d {
                            let poly_idx = instruction_d + i;
                            let k = ra_indices.bytecode_ra[i] as usize;
                            if !touched_flags[poly_idx].contains(k) {
                                touched_flags[poly_idx].insert(k);
                            }
                            local_unreduced[poly_idx][k] += add0_unr;
                        }

                        // RamRa contributions (may be None)
                        for i in 0..ram_d {
                            let poly_idx = instruction_d + bytecode_d + i;
                            if let Some(k) = ra_indices.ram_ra[i] {
                                let k = k as usize;
                                if !touched_flags[poly_idx].contains(k) {
                                    touched_flags[poly_idx].insert(k);
                                }
                                local_unreduced[poly_idx][k] += add0_unr;
                            }
                        }
                    }

                    // Process cycle j1 (last_bit = 1)
                    if j1 < T {
                        let ra_indices =
                            RaIndices::from_cycle(&trace[j1], preprocessing, one_hot_params);

                        // InstructionRa contributions
                        for i in 0..instruction_d {
                            let k = ra_indices.instruction_ra[i] as usize;
                            if !touched_flags[i].contains(k) {
                                touched_flags[i].insert(k);
                            }
                            local_unreduced[i][k] += add1_unr;
                        }

                        // BytecodeRa contributions
                        for i in 0..bytecode_d {
                            let poly_idx = instruction_d + i;
                            let k = ra_indices.bytecode_ra[i] as usize;
                            if !touched_flags[poly_idx].contains(k) {
                                touched_flags[poly_idx].insert(k);
                            }
                            local_unreduced[poly_idx][k] += add1_unr;
                        }

                        // RamRa contributions (may be None)
                        for i in 0..ram_d {
                            let poly_idx = instruction_d + bytecode_d + i;
                            if let Some(k) = ra_indices.ram_ra[i] {
                                let k = k as usize;
                                if !touched_flags[poly_idx].contains(k) {
                                    touched_flags[poly_idx].insert(k);
                                }
                                local_unreduced[poly_idx][k] += add1_unr;
                            }
                        }
                    }
                }

                // Reduce and scale by E_out[x_out]
                for poly_idx in 0..N {
                    for k in touched_flags[poly_idx].ones() {
                        let reduced = F::from_montgomery_reduce::<9>(local_unreduced[poly_idx][k]);
                        partial_G[poly_idx][k] += e_out * reduced;
                    }
                }

                partial_G
            },
        )
        .reduce(
            || vec![unsafe_allocate_zero_vec::<F>(K); N],
            |mut a, b| {
                for poly_idx in 0..N {
                    for (x, y) in a[poly_idx].iter_mut().zip(&b[poly_idx]) {
                        *x += *y;
                    }
                }
                a
            },
        );

    G
}

// ============================================================================
// IMPORTS FOR SUMCHECK
// ============================================================================

use allocative::Allocative;

use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

const DEGREE_BOUND: usize = 2;

// ============================================================================
// PARAMS
// ============================================================================

/// Family indices for the three ra polynomial types.
pub const FAMILY_INSTRUCTION: usize = 0;
pub const FAMILY_BYTECODE: usize = 1;
pub const FAMILY_RAM: usize = 2;
pub const NUM_FAMILIES: usize = 3;

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
    /// γ^0, γ^1, ..., γ^{3N-1} for batching (3 claims per ra polynomial)
    /// Order: γ^{3i} = HW, γ^{3i+1} = Bool, γ^{3i+2} = Virt
    pub gamma_powers: Vec<F>,
    /// Shared r_cycle from Stage 6 (all ra claims share this)
    pub r_cycle: Vec<F::Challenge>,
    /// r_address values from Booleanity sumcheck, shared per family (3 total)
    /// Index: [instruction, bytecode, ram]
    pub r_addr_bool_per_family: [Vec<F::Challenge>; NUM_FAMILIES],
    /// r_address values from Virtualization/ReadRaf sumcheck for each ra_i (N total)
    /// Each ra_i has different r_addr because chunks are bound sequentially
    pub r_addr_virt: Vec<Vec<F::Challenge>>,
    /// HammingWeight claims for each ra_i
    pub claims_hw: Vec<F>,
    /// Booleanity claims for each ra_i
    pub claims_bool: Vec<F>,
    /// Virtualization claims for each ra_i
    pub claims_virt: Vec<F>,
    /// log_2(k_chunk) - number of sumcheck rounds
    pub log_k_chunk: usize,
    /// Polynomial labels: InstructionRa(0..d), BytecodeRa(0..d), RamRa(0..d)
    pub polynomial_types: Vec<CommittedPolynomial>,
    /// Family index for each polynomial (0=instruction, 1=bytecode, 2=ram)
    pub family: Vec<usize>,
}

impl<F: JoltField> HammingWeightClaimReductionParams<F> {
    /// Create params by fetching claims from Stage 6 and sampling batching challenge.
    ///
    /// Fetches:
    /// - HammingWeight claims (from HammingBooleanity virtual polynomial)
    /// - Booleanity claims (r_addr shared per family)
    /// - Virtualization claims (r_addr different per ra_i)
    pub fn new(
        r_cycle: Vec<F::Challenge>,
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let N = instruction_d + bytecode_d + ram_d;
        let log_k_chunk = one_hot_params.log_k_chunk;

        // Build polynomial types list and family mapping
        let mut polynomial_types = Vec::with_capacity(N);
        let mut family = Vec::with_capacity(N);
        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
            family.push(FAMILY_INSTRUCTION);
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
            family.push(FAMILY_BYTECODE);
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
            family.push(FAMILY_RAM);
        }

        // Sample batching challenge γ and compute powers (3 claims per ra_i)
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = Vec::with_capacity(3 * N);
        let mut power = F::one();
        for _ in 0..(3 * N) {
            gamma_powers.push(power);
            power *= gamma;
        }

        // Fetch per-family booleanity r_address (shared within each family)
        // We take the first ra polynomial of each family as representative
        let r_addr_bool_per_family = {
            // Instruction family
            let (instr_bool_point, _) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::InstructionBooleanity,
            );
            // Bytecode family
            let (bc_bool_point, _) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(0),
                SumcheckId::BytecodeBooleanity,
            );
            // Ram family
            let (ram_bool_point, _) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(0),
                SumcheckId::RamBooleanity,
            );

            [
                instr_bool_point.r[..log_k_chunk].to_vec(),
                bc_bool_point.r[..log_k_chunk].to_vec(),
                ram_bool_point.r[..log_k_chunk].to_vec(),
            ]
        };

        // Fetch claims for each ra_i
        let mut r_addr_virt = Vec::with_capacity(N);
        let mut claims_hw = Vec::with_capacity(N);
        let mut claims_bool = Vec::with_capacity(N);
        let mut claims_virt = Vec::with_capacity(N);

        // Get the RAM HammingWeight scaling factor (for RAM, not all cycles access memory)
        let (_, ram_hw_factor) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );

        for (idx, poly_type) in polynomial_types.iter().enumerate() {
            let (bool_sumcheck_id, virt_sumcheck_id) = match poly_type {
                CommittedPolynomial::InstructionRa(_) => (
                    SumcheckId::InstructionBooleanity,
                    SumcheckId::InstructionRaVirtualization,
                ),
                CommittedPolynomial::BytecodeRa(_) => {
                    (SumcheckId::BytecodeBooleanity, SumcheckId::BytecodeReadRaf)
                }
                CommittedPolynomial::RamRa(_) => {
                    (SumcheckId::RamBooleanity, SumcheckId::RamRaVirtualization)
                }
                _ => unreachable!(),
            };

            // HammingWeight claim:
            // - For Instruction/Bytecode: H_i = 1 (one-hot sums to 1)
            // - For Ram: H_i = ram_hw_factor (fraction of cycles accessing RAM)
            let hw_claim = if family[idx] == FAMILY_RAM {
                ram_hw_factor
            } else {
                F::one()
            };
            claims_hw.push(hw_claim);

            // Booleanity claim
            let (_, bool_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, bool_sumcheck_id);
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
            r_addr_bool_per_family,
            r_addr_virt,
            claims_hw,
            claims_bool,
            claims_virt,
            log_k_chunk,
            polynomial_types,
            family,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingWeightClaimReductionParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Σ_i (γ^{3i} · claim_hw_i + γ^{3i+1} · claim_bool_i + γ^{3i+2} · claim_virt_i)
        let mut claim = F::zero();
        for i in 0..self.polynomial_types.len() {
            claim += self.gamma_powers[3 * i] * self.claims_hw[i];
            claim += self.gamma_powers[3 * i + 1] * self.claims_bool[i];
            claim += self.gamma_powers[3 * i + 2] * self.claims_virt[i];
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
}

// ============================================================================
// PROVER
// ============================================================================

/// Prover for the fused HammingWeight + Address Reduction sumcheck.
///
/// This sumcheck combines all three ra_i claim types (HammingWeight, Booleanity,
/// Virtualization) into a single degree-2 sumcheck over log_k_chunk rounds.
///
/// Memory optimization: eq_bool is shared per family (3 polynomials), while
/// eq_virt requires one per ra_i (N polynomials).
#[derive(Allocative)]
pub struct HammingWeightClaimReductionProver<F: JoltField> {
    /// G_i polynomials (pushforward of ra_i over r_cycle)
    /// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
    G: Vec<MultilinearPolynomial<F>>,
    /// eq(r_addr_bool_family, ·) shared per family (3 total: instruction/bytecode/ram)
    eq_bool: [MultilinearPolynomial<F>; NUM_FAMILIES],
    /// eq(r_addr_virt_i, ·) for each ra polynomial (N total)
    eq_virt: Vec<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: HammingWeightClaimReductionParams<F>,
}

impl<F: JoltField> HammingWeightClaimReductionProver<F> {
    /// Initialize the prover by computing all G_i polynomials.
    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::initialize")]
    pub fn initialize<PCS: CommitmentScheme<Field = F>>(
        params: HammingWeightClaimReductionParams<F>,
        trace: &[Cycle],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Compute all G_i polynomials via streaming
        let G_vecs = compute_all_G(trace, &params.r_cycle, preprocessing, one_hot_params);
        let G: Vec<MultilinearPolynomial<F>> = G_vecs
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        // Compute 3 shared eq_bool tables (one per family)
        let eq_bool = std::array::from_fn(|fam| {
            MultilinearPolynomial::from(EqPolynomial::evals(&params.r_addr_bool_per_family[fam]))
        });

        // Compute N eq_virt tables (one per ra polynomial)
        let N = params.polynomial_types.len();
        let eq_virt: Vec<MultilinearPolynomial<F>> = (0..N)
            .map(|i| MultilinearPolynomial::from(EqPolynomial::evals(&params.r_addr_virt[i])))
            .collect();

        Self {
            G,
            eq_bool,
            eq_virt,
            params,
        }
    }
}

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
            for i in 0..N {
                let g_evals =
                    self.G[i].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let family = self.params.family[i];
                let eq_b_evals = self.eq_bool[family]
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_v_evals = self.eq_virt[i]
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                // γ^{3i} · G (HammingWeight)
                // γ^{3i+1} · eq_bool · G (Booleanity)
                // γ^{3i+2} · eq_virt · G (Virtualization)
                let gamma_hw = self.params.gamma_powers[3 * i];
                let gamma_bool = self.params.gamma_powers[3 * i + 1];
                let gamma_virt = self.params.gamma_powers[3 * i + 2];

                for k in 0..DEGREE_BOUND {
                    // Fused: G · (γ_hw + γ_bool·eq_b + γ_virt·eq_v)
                    evals[k] += g_evals[k]
                        * (gamma_hw + gamma_bool * eq_b_evals[k] + gamma_virt * eq_v_evals[k]);
                }
            }
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind all polynomials in parallel
        rayon::scope(|s| {
            s.spawn(|_| {
                self.G.par_iter_mut().for_each(|g| {
                    g.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
            s.spawn(|_| {
                // Only 3 eq_bool polynomials (shared per family)
                self.eq_bool.par_iter_mut().for_each(|eq| {
                    eq.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
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
        transcript: &mut T,
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
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::HammingWeightClaimReduction,
                r_address.clone(),
                self.params.r_cycle.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// ============================================================================
// VERIFIER
// ============================================================================

pub struct HammingWeightClaimReductionVerifier<F: JoltField> {
    params: HammingWeightClaimReductionParams<F>,
}

impl<F: JoltField> HammingWeightClaimReductionVerifier<F> {
    pub fn new(
        r_cycle: Vec<F::Challenge>,
        one_hot_params: &OneHotParams,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = HammingWeightClaimReductionParams::new(
            r_cycle,
            one_hot_params,
            accumulator,
            transcript,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingWeightClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let N = self.params.polynomial_types.len();

        // Compute ρ (final address point)
        let rho: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let rho = rho.r;

        let mut output_claim = F::zero();

        for i in 0..N {
            let family = self.params.family[i];

            // eq evaluations at final point ρ
            let eq_bool_eval = EqPolynomial::mle(&rho, &self.params.r_addr_bool_per_family[family]);
            let eq_virt_eval = EqPolynomial::mle(&rho, &self.params.r_addr_virt[i]);

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

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let N = self.params.polynomial_types.len();

        // Compute full opening point (r_address || r_cycle)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;
        let full_point = [r_address.as_slice(), self.params.r_cycle.as_slice()].concat();

        for i in 0..N {
            accumulator.append_sparse(
                transcript,
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
