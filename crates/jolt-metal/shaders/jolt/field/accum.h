// jolt::WithAccumulator<F>: the deferred-reduction accumulators of a field,
// mirroring jolt_field::WithAccumulator.
//
// Each field header specializes WithAccumulator with two member types:
// - Accumulator: sums of field elements and of products of two field
//   elements (add, fmadd);
// - SmallScalarAccumulator: sums of field elements times signed or unsigned
//   64-bit scalars (add, fmadd_u64, fmadd_i64, fmadd_signed_u64).
//
// Every accumulator type Acc provides:
// - static constexpr constant ulong CAPACITY: the number of terms, of any
//   size its operations admit, that the representation holds exactly. The
//   bound is derived next to each definition;
// - static Acc zero(): the empty sum. Accumulators have no constructors, so
//   they can live in threadgroup memory;
// - void merge(Acc other): adds another partial sum;
// - F reduce(): the canonical field element equal to the sum;
// - a hidden friend Acc simd_shuffle_xor(Acc, ushort), used by reduce.h.
//
// The representation is not jolt_field's: only reduce() must agree, and it
// is canonical, so results are bit-exact whatever the layout.
//
// Capacity is a kernel-author obligation. Every term counts, including the
// ones add() brings, and merge() adds the counts of its operands. A kernel
// that may exceed CAPACITY reduces first and continues from
// Acc::zero().add(partial), which counts as one term. Kernels in this crate
// check their counts with static_assert where the counts are constant.

#ifndef JOLT_FIELD_ACCUM_H
#define JOLT_FIELD_ACCUM_H

namespace jolt {

template <typename F>
struct WithAccumulator;

template <typename F>
using Accumulator = typename WithAccumulator<F>::Accumulator;

template <typename F>
using SmallScalarAccumulator = typename WithAccumulator<F>::SmallScalarAccumulator;

} // namespace jolt

#endif // JOLT_FIELD_ACCUM_H
