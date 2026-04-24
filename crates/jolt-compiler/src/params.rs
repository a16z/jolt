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

/// R1CS column dimension (padded to next power of 2 for Spartan).
pub const NUM_VARS_PADDED: usize = 64;
/// Number of R1CS constraints for the BASELINE module (matches jolt-core's
/// 19-constraint layout, preserving cross-verify parity). Modules that need
/// to enforce additional gates (FieldReg's FADD/FSUB/FMUL/FINV at matrix
/// rows 19-26) pass a widened count to
/// [`ModuleParams::new_with_constraints`] instead of using this default.
pub const NUM_R1CS_CONSTRAINTS: usize = 19;
/// Number of R1CS input polynomials.
pub const NUM_R1CS_INPUTS: usize = 35;

/// Baseline uniskip domain (for 19 constraints). FieldReg-extended modules
/// derive their uniskip domain from `num_r1cs_constraints` at runtime — read
/// `params.outer_uniskip_domain` instead of this const in downstream code.
pub const UNISKIP_DOMAIN_SIZE: usize = (NUM_R1CS_CONSTRAINTS - 1) / 2 + 1; // 10

/// Number of circuit flags.
pub const NUM_CIRCUIT_FLAGS: usize = 14;
/// Number of instruction flags.
pub const NUM_INSTRUCTION_FLAGS: usize = 6;
/// Number of lookup tables (must match jolt-core's `LookupTables::COUNT` = 41).
pub const NUM_LOOKUP_TABLES: usize = 41;
/// Number of product constraints (shift, instruction input, output check).
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;

/// Padded stride of the shared `rv64_constraints` R1CS matrix.
///
/// Matrix has 31 eq (including the 2 limb-sum bridge rows added for
/// Phase 3 bridge soundness, task #65) + 3 product = 34 rows, padded
/// to 64. Every module sharing the matrix pads to this value.
///
/// Before the bridge-soundness rows were added, the matrix fit in
/// 32 rows (29+3); the jump to 64 is the cost of the security fix.
pub const TOTAL_MATRIX_CONSTRAINTS_PADDED: usize = 64;

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

    /// Number of active R1CS equality constraints for this module.
    ///
    /// Baseline modules (no FieldOp gates) use [`NUM_R1CS_CONSTRAINTS`] = 19,
    /// preserving cross-verify parity with jolt-core. FieldReg-extended modules
    /// pass `27` via [`ModuleParams::new_with_constraints`], which widens the
    /// uniskip domain so Spartan actually samples rows 19-26 (FADD/FSUB/
    /// FMUL/FINV gates). The R1CS matrices themselves are always 30 rows;
    /// this field only controls how many rows the outer sumcheck enforces.
    pub num_r1cs_constraints: usize,

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
    /// FieldReg Twist (6th stage-2 instance): log_t cycle + log_k_chunk address.
    pub fr_checking_degree: usize,
    pub fr_checking_rounds: usize,
    /// Limb→Fr bridge (7th stage-2 instance): single 1-D cycle sumcheck,
    /// `log_t` rounds, degree 3 (eq × IsFieldOp × LimbSum).
    pub bridge_checking_degree: usize,
    pub bridge_checking_rounds: usize,
    pub stage2_max_rounds: usize,
    pub stage2_num_instances: usize,
}

impl ModuleParams {
    /// Derive all protocol parameters from trace metadata.
    ///
    /// `log_t` is log₂ of the padded trace length.
    /// `log_k_bytecode` is log₂ of the bytecode address space (program-dependent).
    /// `log_k_ram` is log₂ of the RAM address space (program-dependent).
    ///
    /// Uses the baseline [`NUM_R1CS_CONSTRAINTS`] (19) — preserves cross-verify
    /// parity with jolt-core. FieldReg-extended modules should call
    /// [`ModuleParams::new_with_constraints`] with the widened count instead.
    pub fn new(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Self {
        Self::new_with_constraints(log_t, log_k_bytecode, log_k_ram, NUM_R1CS_CONSTRAINTS)
    }

    /// Derive protocol parameters with an explicit R1CS constraint count.
    ///
    /// Modules that extend the baseline constraint set (e.g. the FieldReg-
    /// enabled module authoring the FADD/FSUB/FMUL/FINV gates at matrix rows
    /// 19-26) must pass the widened count here so the outer Spartan uniskip
    /// domain covers the new rows. The R1CS matrices themselves are always
    /// 30 rows (see `jolt-r1cs/src/constraints/rv64.rs`); this parameter
    /// controls only how many rows Spartan samples.
    pub fn new_with_constraints(
        log_t: usize,
        log_k_bytecode: usize,
        log_k_ram: usize,
        num_r1cs_constraints: usize,
    ) -> Self {
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
        // Constraints split into 2 groups. The uniskip domain covers the
        // larger group (ceil(C/2)). One eq variable selects the group.
        //   L(τ_high, Y) · t1(Y) where t1 has degree 2(K-1), L has degree K-1.
        //   s1(Y) = L × t1 has degree 3(K-1), so 3(K-1)+1 coefficients.
        //
        // For 19 constraints: domain = 10, degree = 9, coeffs = 28.
        // For 27 constraints: domain = 14, degree = 13, coeffs = 40.
        let outer_uniskip_domain = (num_r1cs_constraints - 1) / 2 + 1;
        let outer_uniskip_degree = outer_uniskip_domain - 1;
        let outer_uniskip_num_coeffs = 3 * outer_uniskip_degree + 1;
        let outer_uniskip_poly_degree = outer_uniskip_num_coeffs - 1;
        let outer_remaining_degree = 3;
        // 1 streaming round + log_t linear rounds. The streaming round binds
        // the extra streaming variable (Az/Bz are DuplicateInterleaved to 2T).
        let outer_remaining_rounds = log_t + 1;
        // τ = [τ_cycle (log_t) ‖ τ_streaming (1) ‖ τ_high (Lagrange kernel)]
        // jolt-core squeezes num_cycle_vars + 2 = log_t + 2 total.
        let num_tau = log_t + 2;
        // Must match R1csKey.num_constraints_padded, which pads the TOTAL matrix
        // rows (eq + product) for the GLOBAL shared `rv64_constraints` matrix.
        // Matrix has 31 eq + 3 product = 34 rows → pads to 64. Every module shares
        // the same R1csKey stride, regardless of its per-module `num_r1cs_constraints`
        // (which controls only the outer uniskip domain sampling, not the stride).
        let num_constraints_padded = TOTAL_MATRIX_CONSTRAINTS_PADDED;

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
        // FieldReg Twist: log_t cycle + log_k_chunk address (matches standalone).
        let fr_checking_degree = 3;
        let fr_checking_rounds = log_t + log_k_chunk;
        // Limb→Fr bridge: 1-D cycle sumcheck, log_t rounds, degree 3.
        let bridge_checking_degree = 3;
        let bridge_checking_rounds = log_t;
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
            num_r1cs_constraints,
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
            fr_checking_degree,
            fr_checking_rounds,
            bridge_checking_degree,
            bridge_checking_rounds,
            stage2_max_rounds,
            stage2_num_instances,
        }
    }
}
