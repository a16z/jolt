//! Shared witness-view and table helpers for the per-relation kernels.

use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
#[cfg(feature = "parallel")]
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::JoltWitnessOracle;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::KernelError;

/// Tables at least this large build in parallel; below it rayon dispatch
/// costs more than the work.
#[cfg(feature = "parallel")]
const PAR_THRESHOLD: usize = 1 << 10;

/// Materialize a dense field-element table of the oracle behind `opening`.
pub(crate) fn dense_view<F: JoltField>(
    witness: &dyn JoltWitnessOracle<F>,
    opening: JoltOpeningId,
) -> Result<Vec<F>, KernelError<F>> {
    Ok(witness.oracle_table(opening.polynomial_id())?)
}

/// `eq(point, ·)` evaluations, big-endian (`point[0]` pairs the index MSB).
pub(crate) fn eq_table<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::evals(point, None)
}

/// Fold the address dimension of an address-major `(K × T)` oracle grid by the
/// eq weights of `point` (big-endian, `K = 2^point.len()`):
/// `out[j] = Σ_k eq(point, k) · grid[(k << log_t) | j]`.
pub(crate) fn address_fold<F: JoltField>(
    witness: &dyn JoltWitnessOracle<F>,
    opening: JoltOpeningId,
    log_t: usize,
    point: &[F],
) -> Result<Vec<F>, KernelError<F>> {
    let grid = dense_view(witness, opening)?;
    let addresses = 1usize << point.len();
    let cycles = 1usize << log_t;
    if grid.len() != addresses << log_t {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{opening:?}"),
            expected: addresses << log_t,
            got: grid.len(),
        });
    }
    let eq_address = eq_table(point);
    let fold = |j: usize| -> F {
        (0..addresses)
            .map(|k| grid[(k << log_t) | j] * eq_address[k])
            .sum()
    };
    #[cfg(feature = "parallel")]
    if grid.len() >= PAR_THRESHOLD {
        return Ok((0..cycles).into_par_iter().map(fold).collect());
    }
    Ok((0..cycles).map(fold).collect())
}

/// Fold the cycle dimension of an address-major `(K × T)` oracle grid by the
/// eq weights of `point` (big-endian, `T = 2^point.len()`):
/// `out[k] = Σ_j eq(point, j) · grid[(k << log_t) | j]`.
pub(crate) fn cycle_fold<F: JoltField>(
    witness: &dyn JoltWitnessOracle<F>,
    opening: JoltOpeningId,
    log_k: usize,
    point: &[F],
) -> Result<Vec<F>, KernelError<F>> {
    let grid = dense_view(witness, opening)?;
    let addresses = 1usize << log_k;
    let cycles = 1usize << point.len();
    if grid.len() != addresses * cycles {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{opening:?}"),
            expected: addresses * cycles,
            got: grid.len(),
        });
    }
    let eq_cycle = eq_table(point);
    let fold = |k: usize| -> F {
        (0..cycles)
            .map(|j| grid[(k * cycles) | j] * eq_cycle[j])
            .sum()
    };
    #[cfg(feature = "parallel")]
    if grid.len() >= PAR_THRESHOLD {
        return Ok((0..addresses).into_par_iter().map(fold).collect());
    }
    Ok((0..addresses).map(fold).collect())
}

/// Tile `base` `copies` times: the `(address ‖ cycle)`-indexed replication of a
/// cycle-indexed table across the address dimension (address bits are the high
/// bits of the joint index).
pub(crate) fn tile<F: JoltField>(base: &[F], copies: usize) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if !base.is_empty() && base.len() * copies >= PAR_THRESHOLD {
        let mut out: Vec<F> = unsafe_allocate_zero_vec(base.len() * copies);
        out.par_chunks_mut(base.len())
            .for_each(|chunk| chunk.copy_from_slice(base));
        return out;
    }
    let mut out = Vec::with_capacity(base.len() * copies);
    for _ in 0..copies {
        out.extend_from_slice(base);
    }
    out
}

/// Replicate a cycle-indexed table across the stream bit at the index LSB
/// (`out[(t << 1) | s] = base[t]`).
pub(crate) fn replicate_stream_lsb<F: JoltField>(base: &[F]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if base.len() >= PAR_THRESHOLD {
        let mut out: Vec<F> = unsafe_allocate_zero_vec(base.len() * 2);
        out.par_chunks_mut(2)
            .zip(base.par_iter())
            .for_each(|(pair, &value)| {
                pair[0] = value;
                pair[1] = value;
            });
        return out;
    }
    let mut out = Vec::with_capacity(base.len() * 2);
    for &value in base {
        out.push(value);
        out.push(value);
    }
    out
}

/// A per-stream constant table over the `(cycle ‖ stream)` domain with the
/// stream bit at the index LSB (`out[(t << 1) | s] = values[s]`).
pub(crate) fn stream_pair_lsb<F: JoltField>(values: [F; 2], cycles: usize) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if cycles >= PAR_THRESHOLD {
        let mut out: Vec<F> = unsafe_allocate_zero_vec(cycles * 2);
        out.par_chunks_mut(2).for_each(|pair| {
            pair[0] = values[0];
            pair[1] = values[1];
        });
        return out;
    }
    let mut out = Vec::with_capacity(cycles * 2);
    for _ in 0..cycles {
        out.push(values[0]);
        out.push(values[1]);
    }
    out
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::{
        JoltOpeningId, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_field::{Fr, Ring};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

    use super::{address_fold, cycle_fold, dense_view};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The stored-column backend replays a `(K x T)` grid into the kernels'
    /// fold helpers — the oracle seam's second implementor, no trace behind
    /// it.
    #[test]
    fn fold_helpers_run_against_a_fixed_backend_grid() {
        let mut backend = FixedBackend::new();
        let id = JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamVal);
        // K = 2, T = 2, address-major: grid[(k << log_t) | j].
        let grid = vec![fr(1), fr(2), fr(3), fr(4)];
        backend
            .insert(id, Shape::new(2, PolynomialEncoding::Dense), grid.clone())
            .unwrap();
        let opening = JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamVal,
            JoltRelationId::RamReadWriteChecking,
        );

        assert_eq!(dense_view::<Fr>(&backend, opening).unwrap(), grid);

        let r = fr(7);
        let folded = address_fold::<Fr>(&backend, opening, 1, &[r]).unwrap();
        // out[j] = (1 - r) * grid[j] + r * grid[2 + j]
        assert_eq!(
            folded,
            vec![
                (Fr::from_u64(1) - r) * fr(1) + r * fr(3),
                (Fr::from_u64(1) - r) * fr(2) + r * fr(4),
            ]
        );

        let folded_cycles = cycle_fold::<Fr>(&backend, opening, 1, &[r]).unwrap();
        // out[k] = (1 - r) * grid[k << 1] + r * grid[(k << 1) | 1]
        assert_eq!(
            folded_cycles,
            vec![
                (Fr::from_u64(1) - r) * fr(1) + r * fr(2),
                (Fr::from_u64(1) - r) * fr(3) + r * fr(4),
            ]
        );
    }
}
