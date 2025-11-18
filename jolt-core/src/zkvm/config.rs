//! Runtime configuration for the Jolt zkVM.
//!
//! The prover and verifier agree on the following runtime parameters:
//! - how wide one-hot encodings are expanded (default is 256 length),
//! - how instruction lookup addresses are chunked and phased,
//! - how RAM accesses are decomposed into one-hot chunks, and
//! - how bytecode program counters are decomposed into one-hot chunks.
//!
//! These values depend on external inputs (e.g. the padded trace length) and
//! therefore cannot be encoded as compile-time constants.  Instead, the module
//! exposes a single [`JoltParams::initialize`] entry point that materializes [`JoltParams`]
//! exactly once at startup.  Downstream code reads those values through
//! accessor helpers and treats them as immutable session-wide parameters.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::utils::math::Math;
use common::constants::XLEN;

/// Number of rounds in instruction read RAF (two times the word size which is 64-bit).
const LOG_K: usize = XLEN * 2;

/// The default one-hot chunk logarithm (i.e., log2 of the expansion factor)
const DEFAULT_ONE_HOT_CHUNK_LOG: usize = 8;

/// Trace length threshold that selects 16 phases (instead of 8) for the instruction read raf sumcheck.
const TRACE_LENGTH_THRESHOLD_PHASES_16: usize = 1 << 20;

static ONE_HOT_CHUNK_LOG_OVERRIDE: AtomicUsize = AtomicUsize::new(DEFAULT_ONE_HOT_CHUNK_LOG);
static JOLT_PARAMS: OnceLock<JoltParams> = OnceLock::new();

/// One-hot chunk configuration derived at configuration time.
///
/// `log_chunk` is the number of bits used for each chunk in the one-hot
/// expansion and `chunk_size` is the resulting vector length (`1 << log_chunk`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotParams {
    pub log_chunk: usize,
    pub chunk_size: usize,
}

impl OneHotParams {
    #[inline]
    pub fn compute_d_from_log_k(&self, log_k: usize) -> usize {
        log_k.div_ceil(self.log_chunk)
    }

    #[inline]
    pub fn compute_d(&self, k: usize) -> usize {
        self.compute_d_from_log_k(k.log_2())
    }
}

/// Instruction lookup chunk configuration derived at configuration time.
///
/// The fields capture how an XLEN-sized address is split into smaller chunks
/// and how those chunks are distributed across sumcheck phases.  The chunking
/// reuses the one-hot width so that instruction lookups mirror the RAM and
/// bytecode decompositions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionLookupParams {
    pub log_k: usize,
    pub log_k_chunk: usize,
    pub k_chunk: usize,
    pub d: usize,
    pub phases: usize,
    pub log_m: usize,
    pub m: usize,
}

/// RAM chunk configuration derived at configuration time.
///
/// RAM uses the same one-hot chunking as instruction lookups but may evolve to
/// add RAM-specific knobs (e.g. alternative chunk widths for value vs. address
/// lookups).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamParams {
    pub log_chunk: usize,
    pub chunk_size: usize,
}

/// Bytecode chunk configuration derived at configuration time.
///
/// Bytecode program counters currently reuse the one-hot chunk layout. Grouping the
/// knobs under their own struct makes the relationship explicit and leaves room
/// for bytecode-specific widening in the future.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeParams {
    pub log_chunk: usize,
    pub chunk_size: usize,
}

/// Aggregated Jolt zkVM configuration shared across one-hot, instruction, RAM, and bytecode logic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltParams {
    pub one_hot: OneHotParams,
    pub instruction: InstructionLookupParams,
    pub ram: RamParams,
    pub bytecode: BytecodeParams,
}

impl JoltParams {
    /// Initialize the global Jolt zkVM parameters.
    ///
    /// The call materializes a [`JoltParams`] instance using the supplied
    /// `padded_trace_length`, caches it in a [`OnceLock`], and returns a `'static`
    /// reference.  Subsequent calls must supply identical inputs or they will panic
    /// with a helpful assertion.
    ///
    /// Must be called before any witness or sumcheck construction that depends on
    /// runtime instruction lookup values.
    pub fn initialize(padded_trace_length: usize) -> &'static Self {
        let params = build_params(padded_trace_length);
        match JOLT_PARAMS.set(params) {
            Ok(()) => JOLT_PARAMS.get().expect("parameters just set"),
            Err(_) => {
                let existing = JOLT_PARAMS.get().expect("parameters already set");
                assert_eq!(
                    existing,
                    &build_params(padded_trace_length),
                    "Mismatched Jolt zkVM configuration"
                );
                existing
            }
        }
    }
}

/// Returns the configured parameters.
///
/// # Panics
/// Panics if [`JoltParams::initialize`] has not been called yet.
pub fn params() -> &'static JoltParams {
    JOLT_PARAMS
        .get()
        .expect("Jolt zkVM parameters are not configured")
}

/// Override the one-hot chunk logarithm (i.e., log2 of the expansion factor).
///
/// May only be called before [`JoltParams::initialize`]. The value must divide [`LOG_K`] and
/// is validated both here and when configuration happens.
pub fn set_one_hot_chunk_log(log_chunk: usize) {
    validate_log_chunk(log_chunk);
    if let Some(existing) = JOLT_PARAMS.get() {
        assert_eq!(
            existing.one_hot.log_chunk, log_chunk,
            "Cannot change one-hot chunk log after configuration"
        );
        return;
    }
    ONE_HOT_CHUNK_LOG_OVERRIDE.store(log_chunk, Ordering::SeqCst);
}

/// Internal helper that derives [`JoltParams`] from the padded trace length and
/// the current one-hot override.
///
/// The logic enforces:
/// - `log_chunk` divides [`LOG_K`],
/// - the number of phases (8 or 16) divides [`LOG_K`], and
/// - `log_m` remains positive.
fn build_params(padded_trace_length: usize) -> JoltParams {
    let log_chunk = ONE_HOT_CHUNK_LOG_OVERRIDE.load(Ordering::SeqCst);
    validate_log_chunk(log_chunk);
    let chunk_size = 1usize << log_chunk;

    let phases = if padded_trace_length < TRACE_LENGTH_THRESHOLD_PHASES_16 {
        16
    } else {
        8
    };
    assert!(phases == 8 || phases == 16, "phases must be either 8 or 16");
    assert!(LOG_K % phases == 0, "Number of phases must divide LOG_K");

    let log_m = LOG_K / phases;
    let m = 1usize << log_m;
    let d = LOG_K.div_ceil(log_chunk);
    assert!(
        log_m > 0,
        "log_m must be positive (LOG_K={LOG_K}, phases={phases})"
    );

    let one_hot = OneHotParams {
        log_chunk,
        chunk_size,
    };

    // Instruction lookups share the same chunking configuration as the one-hot encoding.
    let instruction = InstructionLookupParams {
        log_k: LOG_K,
        log_k_chunk: one_hot.log_chunk,
        k_chunk: one_hot.chunk_size,
        d,
        phases,
        log_m,
        m,
    };

    let ram = RamParams {
        log_chunk: one_hot.log_chunk,
        chunk_size: one_hot.chunk_size,
    };

    let bytecode = BytecodeParams {
        log_chunk: one_hot.log_chunk,
        chunk_size: one_hot.chunk_size,
    };

    JoltParams {
        one_hot,
        instruction,
        ram,
        bytecode,
    }
}

/// Ensures the requested one-hot chunk size respects the machine constraints.
fn validate_log_chunk(log_chunk: usize) {
    assert!(log_chunk > 0, "one-hot chunk log must be > 0");
    assert!(
        log_chunk <= 32,
        "one-hot chunk log must be <= 32, otherwise RA polys take too much space"
    );
    assert!(
        LOG_K % log_chunk == 0,
        "one-hot chunk log must divide LOG_K ({LOG_K}), got {log_chunk}"
    );
    assert!(
        log_chunk <= u16::BITS as usize,
        "one-hot chunk log {log_chunk} exceeds storage capacity of RaPolynomial"
    );
}
