//! Heap visitation for scalar tables.
//!
//! Field elements own no heap, so their tables are sized arithmetically from
//! `capacity()`. Going through the derive instead would put `F: Allocative`
//! on every field-generic profiler — a bound the foreign `akita-field`
//! scalar cannot satisfy (both it and `Allocative` are foreign, so no impl
//! can exist here). Reach for these from `#[allocative(visit = ...)]`.
//!
//! TODO(https://github.com/a16z/jolt/issues/1805): the akita-field edge is a
//! pre-cutover bootstrap; once it goes, `JoltField` can require `Allocative`
//! and this module deletes itself along with its ~86 call sites.

use allocative::{Key, Visitor};

/// Bytes the allocator reserved for a flat scalar table.
pub fn visit_scalars<T>(values: &Vec<T>, visitor: &mut Visitor<'_>) {
    visitor.visit_simple(Key::new("elements"), values.capacity() * size_of::<T>());
}

/// [`visit_scalars`] for a table of tables: the outer spine plus every inner
/// reservation. Takes a slice so array-of-table fields visit through the
/// same path; the spine is sized by `len` (a handful of pointers).
pub fn visit_scalar_rows<T>(rows: &[Vec<T>], visitor: &mut Visitor<'_>) {
    visitor.visit_simple(Key::new("spine"), size_of_val(rows));
    visitor.visit_simple(
        Key::new("elements"),
        rows.iter().map(|row| row.capacity() * size_of::<T>()).sum(),
    );
}
