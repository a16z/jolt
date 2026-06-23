use crate::protocol::masking::sample_blinding_digits;
use crate::{AkitaPolyOps, DensePoly, ProverComputeBackend};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::LevelParams;

fn flattened_zk_blinding_digits<const D: usize>(
    digits: &akita_types::FlatDigitBlocks<D>,
) -> Vec<i8> {
    let mut out = Vec::with_capacity(digits.flat_digits().len() * D);
    for plane in digits.flat_digits() {
        out.extend_from_slice(plane);
    }
    out
}

pub(crate) fn commit_zk_hiding_witness<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    root_params: &LevelParams,
    hiding_witness: &[F],
) -> Result<(Vec<F>, Vec<i8>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: ProverComputeBackend<F>,
{
    let num_ring = hiding_witness.len().div_ceil(D).max(1).next_power_of_two();
    let eval_len = num_ring
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness length overflow".to_string()))?;
    let mut evals = vec![F::zero(); eval_len];
    evals[..hiding_witness.len()].copy_from_slice(hiding_witness);
    let num_vars = eval_len.trailing_zeros() as usize;
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals)?;
    let hiding_params = root_params.with_decomp(
        num_ring.trailing_zeros() as usize,
        0,
        root_params.num_digits_commit,
        root_params.num_digits_open,
        num_ring,
    )?;
    let inner = poly.commit_inner(
        backend,
        prepared,
        hiding_params.a_key.row_len(),
        hiding_params.block_len,
        hiding_params.num_blocks,
        hiding_params.num_digits_commit,
        hiding_params.num_digits_open,
        hiding_params.log_basis,
    )?;
    let b_input_digits = inner.decomposed_inner_rows.flat_digits().to_vec();
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(hiding_params.b_key.row_len(), hiding_params.log_basis)?;
    let mut u_blind_rings: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
        prepared,
        hiding_params.b_key.row_len(),
        &b_input_digits,
        hiding_params.log_basis,
    )?;
    let blinding_rows = backend.zk_b_digit_rows::<D>(
        prepared,
        hiding_params.b_key.row_len(),
        b_blinding_digits.flat_digits().len(),
        b_blinding_digits.flat_digits(),
    )?;
    for (row, blinding) in u_blind_rings.iter_mut().zip(blinding_rows) {
        *row += blinding;
    }
    if u_blind_rings.len() != hiding_params.b_key.row_len() {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} ZK hiding rows, expected {}",
            u_blind_rings.len(),
            hiding_params.b_key.row_len()
        )));
    }
    let u_blind = u_blind_rings
        .iter()
        .flat_map(|ring| ring.coeffs.iter().copied())
        .collect();
    Ok((u_blind, flattened_zk_blinding_digits(&b_blinding_digits)))
}
