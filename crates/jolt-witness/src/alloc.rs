//! Zeroed field-table allocation via `alloc_zeroed`.
//!
//! Witness grids are `K × T` tables that start as all zeros; `vec![F::zero(); n]`
//! clone-fills them element by element, which at grid scale is a full serial
//! memory pass. `alloc_zeroed` gets zero pages from the OS instead (the same
//! strategy as `jolt_poly::thread::unsafe_allocate_zero_vec`, duplicated here
//! because jolt-witness does not depend on jolt-poly).

use jolt_field::Field;

/// Allocates `vec![F::zero(); len]` through `alloc_zeroed`.
///
/// # Safety contract
///
/// `F::zero()` must be represented as all-zero bytes and `F` must have no
/// padding bytes (true for Montgomery-form prime fields, where zero's
/// representative is 0). The all-zero-bytes property is asserted at runtime
/// in ALL builds — one `size_of::<F>()` byte-compare per call, nothing next
/// to the allocation — so a nonzero-repr `Field` implementation fails loudly
/// instead of exposing invalid values in release; the assertion reads the
/// bytes of one `F::zero()` value, so it cannot detect padding.
#[expect(
    clippy::unwrap_used,
    reason = "Layout::array only fails on overflow, which the callers' checked row-count arithmetic already rules out"
)]
pub(crate) fn zero_table<F: Field>(len: usize) -> Vec<F> {
    // `alloc_zeroed` requires a nonzero layout size: use safe construction
    // for empty tables and zero-sized `F` instead of violating that contract.
    if len == 0 || std::mem::size_of::<F>() == 0 {
        return std::iter::repeat_with(F::zero).take(len).collect();
    }

    // SAFETY: reads the zero element's bytes to verify the all-zeros
    // invariant `alloc_zeroed` relies on. Runs in all builds: a release-only
    // wrong-repr instantiation would otherwise construct invalid `F` values.
    unsafe {
        let value = &F::zero();
        let ptr = std::ptr::from_ref::<F>(value).cast::<u8>();
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(
            bytes.iter().all(|&byte| byte == 0),
            "F::zero() is not all-zero bytes — zero_table is invalid for this field"
        );
    }

    // SAFETY: `len` and `size_of::<F>()` are nonzero (checked above), so the
    // layout satisfies `alloc_zeroed`'s nonzero-size requirement, and the
    // assertion above guarantees all-zero bytes are a valid `F` (the zero
    // element).
    unsafe {
        let layout = std::alloc::Layout::array::<F>(len).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout).cast::<F>();
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        #[expect(clippy::same_length_and_capacity)]
        Vec::from_raw_parts(ptr, len, len)
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn zero_table_is_all_zeros() {
        let table: Vec<Fr> = super::zero_table(1 << 10);
        assert_eq!(table, vec![Fr::from_u64(0); 1 << 10]);
        assert!(super::zero_table::<Fr>(0).is_empty());
    }
}
