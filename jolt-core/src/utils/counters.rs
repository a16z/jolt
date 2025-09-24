/// Global counters for field operations in the BN254 scalar field.
///
/// Incremented in `jolt-core::field::tracked_ark::TrackedFr` to enable
/// fine-grained performance accounting. Use the getters/resetters below
/// or `get_field_op_counts`/`reset_all_field_op_counts` for bulk operations.
use std::sync::atomic::{AtomicUsize, Ordering};

// Fine-grained counters
/// Count of field additions `a + b`.
pub static ADD_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of field subtractions `a - b`.
pub static SUB_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of field squarings `a.square()`.
pub static SQUARE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Count of full modular multiplications `a * b`.
pub static MULT_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of field inversions `a.inverse()`.
pub static INVERSE_COUNT: AtomicUsize = AtomicUsize::new(0);

// Conversions (From*)
/// Count of `from_bool` conversions.
pub static FROM_BOOL_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_u8` conversions.
pub static FROM_U8_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_u16` conversions.
pub static FROM_U16_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_u32` conversions.
pub static FROM_U32_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_u64` conversions.
pub static FROM_U64_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_i64` conversions.
pub static FROM_I64_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_i128` conversions.
pub static FROM_I128_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_u128` conversions.
pub static FROM_U128_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of `from_bytes` conversions.
pub static FROM_BYTES_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Count of small-integer multiplications `a * n` (64-bit unsigned).
pub static MUL_U64_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of small-integer multiplications `a * n` (64-bit signed).
pub static MUL_I64_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of small-integer multiplications `a * n` (128-bit unsigned).
pub static MUL_U128_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of small-integer multiplications `a * n` (128-bit signed).
pub static MUL_I128_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Count of unreduced full-width multiplications.
pub static MUL_UNRED_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of unreduced multiplications by u64.
pub static MUL_U64_UNRED_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of unreduced multiplications by u128.
pub static MUL_U128_UNRED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Count of Montgomery reductions.
pub static MONT_REDUCE_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of Barrett reductions.
pub static BARRETT_REDUCE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Reset inverse count.
pub fn reset_inverse_count() {
    INVERSE_COUNT.store(0, Ordering::Relaxed);
}

/// Read inverse count.
pub fn get_inverse_count() -> usize {
    INVERSE_COUNT.load(Ordering::Relaxed)
}

/// Reset modular multiplication count.
pub fn reset_mult_count() {
    MULT_COUNT.store(0, Ordering::Relaxed);
}

/// Read modular multiplication count.
pub fn get_mult_count() -> usize {
    MULT_COUNT.load(Ordering::Relaxed)
}

/// Reset conversion counts.
pub fn reset_from_counts() {
    FROM_BOOL_COUNT.store(0, Ordering::Relaxed);
    FROM_U8_COUNT.store(0, Ordering::Relaxed);
    FROM_U16_COUNT.store(0, Ordering::Relaxed);
    FROM_U32_COUNT.store(0, Ordering::Relaxed);
    FROM_U64_COUNT.store(0, Ordering::Relaxed);
    FROM_I64_COUNT.store(0, Ordering::Relaxed);
    FROM_I128_COUNT.store(0, Ordering::Relaxed);
    FROM_U128_COUNT.store(0, Ordering::Relaxed);
    FROM_BYTES_COUNT.store(0, Ordering::Relaxed);
}

/// Read `from_bool` count.
pub fn get_from_bool_count() -> usize {
    FROM_BOOL_COUNT.load(Ordering::Relaxed)
}
/// Read `from_u8` count.
pub fn get_from_u8_count() -> usize {
    FROM_U8_COUNT.load(Ordering::Relaxed)
}
/// Read `from_u16` count.
pub fn get_from_u16_count() -> usize {
    FROM_U16_COUNT.load(Ordering::Relaxed)
}
/// Read `from_u32` count.
pub fn get_from_u32_count() -> usize {
    FROM_U32_COUNT.load(Ordering::Relaxed)
}
/// Read `from_u64` count.
pub fn get_from_u64_count() -> usize {
    FROM_U64_COUNT.load(Ordering::Relaxed)
}
/// Read `from_i64` count.
pub fn get_from_i64_count() -> usize {
    FROM_I64_COUNT.load(Ordering::Relaxed)
}
/// Read `from_i128` count.
pub fn get_from_i128_count() -> usize {
    FROM_I128_COUNT.load(Ordering::Relaxed)
}
/// Read `from_u128` count.
pub fn get_from_u128_count() -> usize {
    FROM_U128_COUNT.load(Ordering::Relaxed)
}
/// Read `from_bytes` count.
pub fn get_from_bytes_count() -> usize {
    FROM_BYTES_COUNT.load(Ordering::Relaxed)
}

/// Reset addition count.
pub fn reset_add_count() {
    ADD_COUNT.store(0, Ordering::Relaxed);
}

/// Read addition count.
pub fn get_add_count() -> usize {
    ADD_COUNT.load(Ordering::Relaxed)
}

/// Reset subtraction count.
pub fn reset_sub_count() {
    SUB_COUNT.store(0, Ordering::Relaxed);
}

/// Read subtraction count.
pub fn get_sub_count() -> usize {
    SUB_COUNT.load(Ordering::Relaxed)
}

/// Reset square count.
pub fn reset_square_count() {
    SQUARE_COUNT.store(0, Ordering::Relaxed);
}

/// Read square count.
pub fn get_square_count() -> usize {
    SQUARE_COUNT.load(Ordering::Relaxed)
}

/// Reset mul-by-u64 count.
pub fn reset_mul_u64_count() {
    MUL_U64_COUNT.store(0, Ordering::Relaxed);
}

/// Read mul-by-u64 count.
pub fn get_mul_u64_count() -> usize {
    MUL_U64_COUNT.load(Ordering::Relaxed)
}

/// Reset mul-by-i64 count.
pub fn reset_mul_i64_count() {
    MUL_I64_COUNT.store(0, Ordering::Relaxed);
}

/// Read mul-by-i64 count.
pub fn get_mul_i64_count() -> usize {
    MUL_I64_COUNT.load(Ordering::Relaxed)
}

/// Reset mul-by-u128 count.
pub fn reset_mul_u128_count() {
    MUL_U128_COUNT.store(0, Ordering::Relaxed);
}

/// Read mul-by-u128 count.
pub fn get_mul_u128_count() -> usize {
    MUL_U128_COUNT.load(Ordering::Relaxed)
}

/// Reset mul-by-i128 count.
pub fn reset_mul_i128_count() {
    MUL_I128_COUNT.store(0, Ordering::Relaxed);
}

/// Read mul-by-i128 count.
pub fn get_mul_i128_count() -> usize {
    MUL_I128_COUNT.load(Ordering::Relaxed)
}

/// Reset unreduced-mul count.
pub fn reset_mul_unred_count() {
    MUL_UNRED_COUNT.store(0, Ordering::Relaxed);
}

/// Read unreduced-mul count.
pub fn get_mul_unred_count() -> usize {
    MUL_UNRED_COUNT.load(Ordering::Relaxed)
}

/// Reset unreduced u64-mul count.
pub fn reset_mul_u64_unred_count() {
    MUL_U64_UNRED_COUNT.store(0, Ordering::Relaxed);
}

/// Read unreduced u64-mul count.
pub fn get_mul_u64_unred_count() -> usize {
    MUL_U64_UNRED_COUNT.load(Ordering::Relaxed)
}

/// Reset unreduced u128-mul count.
pub fn reset_mul_u128_unred_count() {
    MUL_U128_UNRED_COUNT.store(0, Ordering::Relaxed);
}

/// Read unreduced u128-mul count.
pub fn get_mul_u128_unred_count() -> usize {
    MUL_U128_UNRED_COUNT.load(Ordering::Relaxed)
}

/// Reset Montgomery reduction count.
pub fn reset_mont_reduce_count() {
    MONT_REDUCE_COUNT.store(0, Ordering::Relaxed);
}

/// Read Montgomery reduction count.
pub fn get_mont_reduce_count() -> usize {
    MONT_REDUCE_COUNT.load(Ordering::Relaxed)
}

/// Reset Barrett reduction count.
pub fn reset_barrett_reduce_count() {
    BARRETT_REDUCE_COUNT.store(0, Ordering::Relaxed);
}

/// Read Barrett reduction count.
pub fn get_barrett_reduce_count() -> usize {
    BARRETT_REDUCE_COUNT.load(Ordering::Relaxed)
}

/// Snapshot of all field operation counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldOpCounts {
    pub mod_mults: usize,
    pub inverses: usize,
    pub adds: usize,
    pub subs: usize,
    pub squares: usize,
    // conversions
    pub from_bool: usize,
    pub from_u8: usize,
    pub from_u16: usize,
    pub from_u32: usize,
    pub from_u64: usize,
    pub from_i64: usize,
    pub from_i128: usize,
    pub from_u128: usize,
    pub from_bytes: usize,
    pub mul_u64: usize,
    pub mul_i64: usize,
    pub mul_u128: usize,
    pub mul_i128: usize,
    pub mul_unreduced: usize,
    pub mul_u64_unreduced: usize,
    pub mul_u128_unreduced: usize,
    pub montgomery_reductions: usize,
    pub barrett_reductions: usize,
}

/// Read all counters into a single struct for reporting.
pub fn get_field_op_counts() -> FieldOpCounts {
    FieldOpCounts {
        mod_mults: get_mult_count(),
        inverses: get_inverse_count(),
        adds: get_add_count(),
        subs: get_sub_count(),
        squares: get_square_count(),
        from_bool: get_from_bool_count(),
        from_u8: get_from_u8_count(),
        from_u16: get_from_u16_count(),
        from_u32: get_from_u32_count(),
        from_u64: get_from_u64_count(),
        from_i64: get_from_i64_count(),
        from_i128: get_from_i128_count(),
        from_u128: get_from_u128_count(),
        from_bytes: get_from_bytes_count(),
        mul_u64: get_mul_u64_count(),
        mul_i64: get_mul_i64_count(),
        mul_u128: get_mul_u128_count(),
        mul_i128: get_mul_i128_count(),
        mul_unreduced: get_mul_unred_count(),
        mul_u64_unreduced: get_mul_u64_unred_count(),
        mul_u128_unreduced: get_mul_u128_unred_count(),
        montgomery_reductions: get_mont_reduce_count(),
        barrett_reductions: get_barrett_reduce_count(),
    }
}

/// Reset all field operation counters to zero.
pub fn reset_all_field_op_counts() {
    reset_mult_count();
    reset_inverse_count();
    reset_add_count();
    reset_sub_count();
    reset_square_count();
    reset_from_counts();
    reset_mul_u64_count();
    reset_mul_i64_count();
    reset_mul_u128_count();
    reset_mul_i128_count();
    reset_mul_unred_count();
    reset_mul_u64_unred_count();
    reset_mul_u128_unred_count();
    reset_mont_reduce_count();
    reset_barrett_reduce_count();
}

/// Per-operation weights to convert counts into modular-multiplication equivalents.
pub struct FieldOpWeights {
    pub mod_mult: f64,
    pub inverse: f64,
    pub add: f64,
    pub sub: f64,
    pub square: f64,
    pub mul_u64: f64,
    pub mul_i64: f64,
    pub mul_u128: f64,
    pub mul_i128: f64,
    pub mul_unreduced: f64,
    pub mul_u64_unreduced: f64,
    pub mul_u128_unreduced: f64,
    pub montgomery_reduce: f64,
    pub barrett_reduce: f64,
}

/// Compute the modular-multiplication-equivalent cost given counts and weights.
pub fn compute_weighted_modular_equiv_cost(
    counts: &FieldOpCounts,
    weights: &FieldOpWeights,
) -> f64 {
    counts.mod_mults as f64 * weights.mod_mult
        + counts.inverses as f64 * weights.inverse
        + counts.adds as f64 * weights.add
        + counts.subs as f64 * weights.sub
        + counts.squares as f64 * weights.square
        + counts.mul_u64 as f64 * weights.mul_u64
        + counts.mul_i64 as f64 * weights.mul_i64
        + counts.mul_u128 as f64 * weights.mul_u128
        + counts.mul_i128 as f64 * weights.mul_i128
        + counts.mul_unreduced as f64 * weights.mul_unreduced
        + counts.mul_u64_unreduced as f64 * weights.mul_u64_unreduced
        + counts.mul_u128_unreduced as f64 * weights.mul_u128_unreduced
        + counts.montgomery_reductions as f64 * weights.montgomery_reduce
        + counts.barrett_reductions as f64 * weights.barrett_reduce
}
