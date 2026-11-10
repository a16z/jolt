//! Id-free sizing shared by the Akita (lattice) protocol families: the
//! minimum arity of a bounded-dense commitment object and its slot capacity.
//!
//! Like [`crate::twist`], this module carries no protocol ids — each protocol
//! family instantiates it with its own committed-polynomial vocabulary
//! (pinned by the `protocol_modules_are_import_disjoint` boundary test).

/// Minimum physical arity of a dense commitment object (advice, program data,
/// or field-register increments). Akita's bounded-dense DP planner admits no
/// fold schedule below 2^13 coefficients for these single-polynomial groups;
/// one variable of headroom over the current floor absorbs upstream
/// repricing; full-width field increments use the same floor.
/// [`min_dense_slot_capacity`] pads slot capacity — never column
/// arity — up to this bound, so claim reduction is unchanged. Like any unused
/// slot, the padding is unconstrained committed data whose contribution to
/// the single reduced opening is zero w.h.p. under the sampled selector;
/// nothing may assume the padded region is identically zero.
pub const MIN_DENSE_OBJECT_NUM_VARS: usize = 14;

/// Slot capacity of a bounded-dense prefix-packed object: enough slots for
/// the columns, padded up so the physical arity reaches
/// [`MIN_DENSE_OBJECT_NUM_VARS`] when the columns alone fall short.
pub fn min_dense_slot_capacity(column_count: usize, logical_num_vars: usize) -> usize {
    column_count
        .next_power_of_two()
        .max(1usize << MIN_DENSE_OBJECT_NUM_VARS.saturating_sub(logical_num_vars))
}
