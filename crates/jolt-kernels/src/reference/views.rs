//! Shared witness-view and table helpers for the per-relation kernels.

use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_witness::JoltWitnessOracle;

use crate::KernelError;

/// Materialize a dense field-element table of the oracle behind `opening`.
pub(crate) fn dense_view<F: Field>(
    witness: &dyn JoltWitnessOracle<F>,
    opening: JoltOpeningId,
) -> Result<Vec<F>, KernelError<F>> {
    Ok(witness.oracle_table(opening.polynomial_id())?)
}

/// `eq(point, ·)` evaluations, big-endian (`point[0]` pairs the index MSB).
pub(crate) fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.to_vec()).evaluations()
}

/// Fold the address dimension of an address-major `(K × T)` oracle grid by the
/// eq weights of `point` (big-endian, `K = 2^point.len()`):
/// `out[j] = Σ_k eq(point, k) · grid[(k << log_t) | j]`.
pub(crate) fn address_fold<F: Field>(
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
    Ok((0..cycles)
        .map(|j| {
            (0..addresses)
                .map(|k| grid[(k << log_t) | j] * eq_address[k])
                .sum()
        })
        .collect())
}

/// Fold the cycle dimension of an address-major `(K × T)` oracle grid by the
/// eq weights of `point` (big-endian, `T = 2^point.len()`):
/// `out[k] = Σ_j eq(point, j) · grid[(k << log_t) | j]`.
pub(crate) fn cycle_fold<F: Field>(
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
    Ok((0..addresses)
        .map(|k| {
            (0..cycles)
                .map(|j| grid[(k * cycles) | j] * eq_cycle[j])
                .sum()
        })
        .collect())
}

/// Tile `base` `copies` times: the `(address ‖ cycle)`-indexed replication of a
/// cycle-indexed table across the address dimension (address bits are the high
/// bits of the joint index).
pub(crate) fn tile<F: Field>(base: &[F], copies: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(base.len() * copies);
    for _ in 0..copies {
        out.extend_from_slice(base);
    }
    out
}

/// Replicate a cycle-indexed table across the stream bit at the index LSB
/// (`out[(t << 1) | s] = base[t]`).
pub(crate) fn replicate_stream_lsb<F: Field>(base: &[F]) -> Vec<F> {
    let mut out = Vec::with_capacity(base.len() * 2);
    for &value in base {
        out.push(value);
        out.push(value);
    }
    out
}

/// A per-stream constant table over the `(cycle ‖ stream)` domain with the
/// stream bit at the index LSB (`out[(t << 1) | s] = values[s]`).
pub(crate) fn stream_pair_lsb<F: Field>(values: [F; 2], cycles: usize) -> Vec<F> {
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
    use jolt_field::{Fr, FromPrimitiveInt};
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
