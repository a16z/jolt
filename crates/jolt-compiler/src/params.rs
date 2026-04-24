//! Protocol parameters derived from trace metadata.
//!
//! [`ModuleParams`] computes all derived constants (decomposition dimensions,
//! sumcheck round counts, uniskip degrees) from three inputs:
//! - `log_t`: log₂ of the padded trace length
//! - `log_k_bytecode`: log₂ of the bytecode address space
//! - `log_k_ram`: log₂ of the RAM address space
//!
//! The one-hot chunk size switches at `log_t = 25` (the D=16/D=32 threshold).

/// Threshold: traces with `log_t >= 25` use `log_k_chunk = 8` (D=16),
/// smaller traces use `log_k_chunk = 4` (D=32).
pub const ONEHOT_CHUNK_THRESHOLD_LOG_T: usize = 25;

/// Instruction lookup index width (RV64: 64-bit × 2 operands).
pub const LOG_K_INSTRUCTION: usize = 128;

/// Register address space: 128 registers.
pub const LOG_K_REG: usize = 7;

/// Number of distinct lookup table types (LookupTables::COUNT for RV64).
pub const NUM_DISTINCT_LOOKUP_TABLES: usize = 41;

/// Trace length threshold for instruction sumcheck phase count.
/// log_T < 24 → 16 phases; log_T >= 24 → 8 phases.
pub const INSTRUCTION_PHASES_THRESHOLD_LOG_T: usize = 24;

/// R1CS column dimension (padded to next power of 2 for Spartan). After the
/// v2 BN254 Fr coprocessor expansion this must cover `NUM_VARS_PER_CYCLE = 50`
/// in `jolt_r1cs::constraints::rv64`; 50.next_power_of_two() = 64.
pub const NUM_VARS_PADDED: usize = 64;
/// Number of R1CS eq constraints (Spartan outer-sumcheck group-split input
/// count). Matches `jolt_r1cs::constraints::rv64::NUM_EQ_CONSTRAINTS = 32`
/// (19 RV base + 13 BN254 Fr coprocessor rows).
pub const NUM_R1CS_CONSTRAINTS: usize = 32;
/// Number of R1CS input polynomials (slots 1..=47 in `rv64.rs`). Covers the
/// 21 RV-base inputs + 14 RV-base flags + 9 BN254 Fr flags + 3 virtual
/// BN254 Fr operand columns. `V_BRANCH` (48) and `V_NEXT_IS_NOOP` (49) are
/// product factors, not inputs.
pub const NUM_R1CS_INPUTS: usize = 47;

/// Uniskip domain: constraints split into 2 groups, domain = (C-1)/2 + 1.
pub const UNISKIP_DOMAIN_SIZE: usize = (NUM_R1CS_CONSTRAINTS - 1) / 2 + 1; // 16
/// Group 1 has UNISKIP_DOMAIN_SIZE constraints, group 2 has the rest.
pub const NUM_GROUP2_CONSTRAINTS: usize = NUM_R1CS_CONSTRAINTS - UNISKIP_DOMAIN_SIZE; // 16

/// Number of circuit flags (14 RV base + 9 BN254 Fr coprocessor).
pub const NUM_CIRCUIT_FLAGS: usize = 23;
/// Number of instruction flags.
pub const NUM_INSTRUCTION_FLAGS: usize = 6;
/// Number of lookup tables (must match jolt-core's `LookupTables::COUNT` = 41).
pub const NUM_LOOKUP_TABLES: usize = 41;
/// Number of product constraints (shift, instruction input, output check).
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;

/// All derived protocol parameters for a given trace.
#[derive(Debug, Clone)]
pub struct ModuleParams {
    // -- Primary inputs --
    pub log_t: usize,
    pub log_k_bytecode: usize,
    pub log_k_ram: usize,

    // -- One-hot decomposition --
    pub log_k_chunk: usize,
    pub k_chunk: usize,
    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,

    // -- Committed poly count --
    pub num_committed: usize,

    // -- Outer Spartan --
    pub outer_uniskip_degree: usize,
    pub outer_uniskip_domain: usize,
    pub outer_uniskip_num_coeffs: usize,
    pub outer_uniskip_poly_degree: usize,
    pub outer_remaining_degree: usize,
    pub outer_remaining_rounds: usize,
    pub num_tau: usize,
    /// Padded constraint count (next power of two of total R1CS constraints).
    /// This is the stride between cycles in Az/Bz buffers.
    pub num_constraints_padded: usize,
    // -- Product virtual --
    pub product_uniskip_degree: usize,
    pub product_uniskip_domain: usize,
    pub product_uniskip_num_coeffs: usize,
    pub product_uniskip_poly_degree: usize,
    pub product_remainder_degree: usize,
    pub product_remainder_rounds: usize,

    // -- Instruction lookup sumcheck --
    pub instruction_phases: usize,
    pub instruction_chunk_bits: usize,
    pub ra_virtual_log_k_chunk: usize,
    pub n_virtual_ra_polys: usize,

    // -- Stage 2 instances --
    pub rw_checking_degree: usize,
    pub rw_checking_rounds: usize,
    pub instruction_claim_reduction_degree: usize,
    pub instruction_claim_reduction_rounds: usize,
    pub raf_evaluation_degree: usize,
    pub raf_evaluation_rounds: usize,
    pub output_check_degree: usize,
    pub output_check_rounds: usize,
    pub stage2_max_rounds: usize,
    pub stage2_num_instances: usize,
}

impl ModuleParams {
    /// Derive all protocol parameters from trace metadata.
    ///
    /// `log_t` is log₂ of the padded trace length.
    /// `log_k_bytecode` is log₂ of the bytecode address space (program-dependent).
    /// `log_k_ram` is log₂ of the RAM address space (program-dependent).
    pub fn new(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Self {
        let log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let k_chunk = 1 << log_k_chunk;

        let instruction_d = LOG_K_INSTRUCTION / log_k_chunk;
        let bytecode_d = log_k_bytecode.div_ceil(log_k_chunk);
        let ram_d = log_k_ram.div_ceil(log_k_chunk);

        let num_committed = 2 + instruction_d + ram_d + bytecode_d;

        // Outer Spartan — group-split uniskip.
        //
        // 19 constraints split into 2 groups (10 + 9). The uniskip domain
        // covers the larger group. One eq variable selects the group.
        //   L(τ_high, Y) · t1(Y) where t1 has degree 2(K-1), L has degree K-1.
        //   s1(Y) = L × t1 has degree 3(K-1), so 3(K-1)+1 coefficients.
        let outer_uniskip_degree = UNISKIP_DOMAIN_SIZE - 1; // 9
        let outer_uniskip_domain = UNISKIP_DOMAIN_SIZE; // 10
        let outer_uniskip_num_coeffs = 3 * outer_uniskip_degree + 1; // 28
        let outer_uniskip_poly_degree = outer_uniskip_num_coeffs - 1; // 27
        let outer_remaining_degree = 3;
        // 1 streaming round + log_t linear rounds. The streaming round binds
        // the extra streaming variable (Az/Bz are DuplicateInterleaved to 2T).
        let outer_remaining_rounds = log_t + 1;
        // τ = [τ_cycle (log_t) ‖ τ_streaming (1) ‖ τ_high (Lagrange kernel)]
        // jolt-core squeezes num_cycle_vars + 2 = log_t + 2 total.
        let num_tau = log_t + 2;
        let num_constraints_padded = NUM_R1CS_CONSTRAINTS.next_power_of_two();

        // Product virtual
        let product_uniskip_degree = NUM_PRODUCT_CONSTRAINTS - 1;
        let product_uniskip_domain = NUM_PRODUCT_CONSTRAINTS;
        let product_uniskip_num_coeffs = 3 * product_uniskip_degree + 1;
        let product_uniskip_poly_degree = product_uniskip_num_coeffs - 1;
        let product_remainder_degree = 3;
        let product_remainder_rounds = log_t;

        // Instruction lookup sumcheck
        let instruction_phases = if log_t < INSTRUCTION_PHASES_THRESHOLD_LOG_T {
            16
        } else {
            8
        };
        let instruction_chunk_bits = LOG_K_INSTRUCTION / instruction_phases;
        let ra_virtual_log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            LOG_K_INSTRUCTION / 8
        } else {
            LOG_K_INSTRUCTION / 4
        };
        let n_virtual_ra_polys = LOG_K_INSTRUCTION / ra_virtual_log_k_chunk;

        // Stage 2 instances
        let rw_checking_degree = 3;
        let rw_checking_rounds = log_k_ram + log_t;
        let instruction_claim_reduction_degree = 2;
        let instruction_claim_reduction_rounds = log_t;
        let raf_evaluation_degree = 2;
        let raf_evaluation_rounds = log_k_ram;
        let output_check_degree = 3;
        let output_check_rounds = log_k_ram;
        let stage2_max_rounds = rw_checking_rounds;
        let stage2_num_instances = 5;

        Self {
            log_t,
            log_k_bytecode,
            log_k_ram,
            log_k_chunk,
            k_chunk,
            instruction_d,
            bytecode_d,
            ram_d,
            num_committed,
            outer_uniskip_degree,
            outer_uniskip_domain,
            outer_uniskip_num_coeffs,
            outer_uniskip_poly_degree,
            outer_remaining_degree,
            outer_remaining_rounds,
            num_tau,
            num_constraints_padded,
            product_uniskip_degree,
            product_uniskip_domain,
            product_uniskip_num_coeffs,
            product_uniskip_poly_degree,
            product_remainder_degree,
            product_remainder_rounds,
            instruction_phases,
            instruction_chunk_bits,
            ra_virtual_log_k_chunk,
            n_virtual_ra_polys,
            rw_checking_degree,
            rw_checking_rounds,
            instruction_claim_reduction_degree,
            instruction_claim_reduction_rounds,
            raf_evaluation_degree,
            raf_evaluation_rounds,
            output_check_degree,
            output_check_rounds,
            stage2_max_rounds,
            stage2_num_instances,
        }
    }
}
