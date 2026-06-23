//! Preprocessing helpers for setup-prefix commitment artifacts (slice 02B).

use crate::api::commitment::{
    commit_inner_block_digit_count, commit_inner_flat_digit_count,
    validate_commit_outer_input_nonempty,
};
use crate::compute::{CommitmentComputeBackend, DenseCommitInput, DenseCommitRowsPlan};
use crate::kernels::linear::decompose_rows_i8_into;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use akita_algebra::CyclotomicRing;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{
    digest_level_params, setup_prefix_slot_id, AkitaCommitmentHint, AkitaExpandedSetup,
    FlatDigitBlocks, LevelParams, RingCommitment, SetupPrefixSlot,
};

/// Commit one padded flat prefix of the shared setup matrix.
///
/// The witness is the coefficient form of `S^flat[0..natural_len]`,
/// zero-padded to `n_prefix`. The caller must supply `level_params` whose inner
/// witness shape satisfies `num_blocks * block_len == n_prefix / D`.
///
/// # Errors
///
/// Returns an error if shapes overflow, the prefix does not fit the setup matrix,
/// or backend commitment fails.
pub fn commit_setup_prefix<F, const D: usize, B>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    level_params: &LevelParams,
    setup_seed_digest: [u8; 32],
    n_prefix: usize,
    natural_len: usize,
) -> Result<SetupPrefixSlot<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    if natural_len == 0 || natural_len > n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup prefix natural length must be in 1..=n_prefix".to_string(),
        ));
    }
    if !n_prefix.is_multiple_of(D) || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a power-of-two multiple of D".to_string(),
        ));
    }
    let padded_ring_slots = n_prefix / D;
    let witness_ring_slots = level_params
        .num_blocks
        .checked_mul(level_params.block_len)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup prefix witness shape overflow".to_string())
        })?;
    if witness_ring_slots != padded_ring_slots {
        return Err(AkitaError::InvalidSetup(format!(
            "level params witness shape {witness_ring_slots} ring slots does not match padded prefix {padded_ring_slots}"
        )));
    }

    let available_field_len = expanded
        .shared_matrix()
        .total_ring_elements_at::<D>()?
        .checked_mul(D)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
        })?;
    if n_prefix > available_field_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length exceeds shared matrix capacity".to_string(),
        ));
    }

    let ring_elems =
        extract_setup_prefix_ring_elems::<F, D>(expanded, padded_ring_slots, natural_len)?;
    let block_slices =
        setup_prefix_block_slices(&ring_elems, level_params.num_blocks, level_params.block_len)?;

    let recomposed_inner_rows = backend.dense_commit_rows(
        prepared,
        DenseCommitRowsPlan {
            n_a: level_params.a_key.row_len(),
            input: DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_commit: level_params.num_digits_commit,
                log_basis: level_params.log_basis,
            },
        },
    )?;

    let block_sizes = recomposed_inner_rows
        .iter()
        .map(|_| {
            commit_inner_block_digit_count(
                level_params.a_key.row_len(),
                level_params.num_digits_open,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut decomposed_inner_rows = FlatDigitBlocks::zeroed(block_sizes)?;
    let dst_blocks = decomposed_inner_rows.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(recomposed_inner_rows))
        .try_for_each(|(dst, rows)| -> Result<(), AkitaError> {
            decompose_rows_i8_into(
                rows,
                dst,
                level_params.num_digits_open,
                level_params.log_basis,
            );
            Ok(())
        })?;
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(recomposed_inner_rows.iter())
        .try_for_each(|(dst, rows)| -> Result<(), AkitaError> {
            decompose_rows_i8_into(
                rows,
                dst,
                level_params.num_digits_open,
                level_params.log_basis,
            );
            Ok(())
        })?;

    let b_input_len = commit_inner_flat_digit_count(
        level_params.num_blocks,
        level_params.a_key.row_len(),
        level_params.num_digits_open,
    )?;
    validate_commit_outer_input_nonempty(b_input_len)?;
    let mut b_input_digits = vec![[0i8; D]; b_input_len];
    b_input_digits.copy_from_slice(decomposed_inner_rows.flat_digits());
    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(level_params.b_key.row_len(), level_params.log_basis)?;
    #[cfg(feature = "zk")]
    let mut u = backend.digit_rows::<D>(
        prepared,
        level_params.b_key.row_len(),
        &b_input_digits,
        level_params.log_basis,
    )?;
    #[cfg(not(feature = "zk"))]
    let u = backend.digit_rows::<D>(
        prepared,
        level_params.b_key.row_len(),
        &b_input_digits,
        level_params.log_basis,
    )?;
    #[cfg(feature = "zk")]
    {
        let blinding_rows = backend.zk_b_digit_rows::<D>(
            prepared,
            level_params.b_key.row_len(),
            b_blinding_digits.flat_digits().len(),
            b_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
    }
    if u.len() != level_params.b_key.row_len() {
        return Err(AkitaError::InvalidSetup(format!(
            "setup prefix commit returned {} B rows, expected {}",
            u.len(),
            level_params.b_key.row_len()
        )));
    }

    let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
        #[cfg(feature = "zk")]
        b_blinding_digits,
    );
    let id = setup_prefix_slot_id(
        setup_seed_digest,
        D,
        natural_len,
        n_prefix,
        digest_level_params(std::slice::from_ref(level_params)),
    );
    Ok(SetupPrefixSlot {
        id,
        natural_len,
        padded_len: n_prefix,
        commitment: RingCommitment { u },
        hint,
    })
}

fn extract_setup_prefix_ring_elems<F, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    padded_ring_slots: usize,
    natural_len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    let fields = expanded.shared_matrix().as_field_slice();
    let padded_field_len = padded_ring_slots.checked_mul(D).ok_or_else(|| {
        AkitaError::InvalidSetup("setup prefix padded field length overflow".to_string())
    })?;
    if natural_len > padded_field_len || padded_field_len > fields.len() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length exceeds shared matrix capacity".to_string(),
        ));
    }

    let mut ring_elems = vec![CyclotomicRing::zero(); padded_ring_slots];
    for (ring, coeffs) in ring_elems.iter_mut().zip(fields[..natural_len].chunks(D)) {
        ring.coefficients_mut()[..coeffs.len()].copy_from_slice(coeffs);
    }
    Ok(ring_elems)
}

fn setup_prefix_block_slices<F, const D: usize>(
    ring_elems: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
) -> Result<Vec<&[CyclotomicRing<F, D>]>, AkitaError>
where
    F: FieldCore,
{
    if num_blocks
        .checked_mul(block_len)
        .is_none_or(|witness| witness != ring_elems.len())
    {
        return Err(AkitaError::InvalidSetup(
            "setup prefix ring elements do not match witness block layout".to_string(),
        ));
    }
    Ok((0..num_blocks)
        .map(|block_idx| {
            let start = block_idx
                .checked_mul(block_len)
                .expect("block index fits after witness length check");
            &ring_elems[start..start + block_len]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use crate::AkitaProverSetup;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128Offset275 as F;
    use akita_types::{
        active_setup_field_len, setup_seed_digest, OpeningBatch, SetupMatrixEnvelope,
        SisModulusFamily,
    };

    fn prefix_level_params(ring_dimension: usize) -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            ring_dimension,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(2, 3, 2, 2, 3)
        .expect("level params")
    }

    fn setup_capacity_for(level_params: &LevelParams, n_prefix: usize) -> usize {
        n_prefix.max(
            level_params
                .b_key
                .row_len()
                .checked_mul(
                    level_params
                        .num_blocks
                        .checked_mul(level_params.a_key.row_len())
                        .and_then(|n| n.checked_mul(level_params.num_digits_open))
                        .expect("b input shape"),
                )
                .expect("setup capacity"),
        )
    }

    fn test_setup<const D: usize>(
        level_params: &LevelParams,
        n_prefix: usize,
    ) -> AkitaProverSetup<F, D> {
        AkitaProverSetup::<F, D>::generate_with_capacity(
            8,
            1,
            SetupMatrixEnvelope {
                max_setup_len: setup_capacity_for(level_params, n_prefix).max(1),
                #[cfg(feature = "zk")]
                max_zk_b_len: level_params
                    .b_key
                    .row_len()
                    .checked_mul(akita_types::zk::blinding_digit_plane_count::<F>(
                        level_params.b_key.row_len(),
                        D,
                        level_params.log_basis,
                    ))
                    .expect("ZK B setup capacity"),
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .expect("setup")
    }

    #[test]
    fn setup_prefix_extraction_zero_pads_after_natural_len() {
        let level_params = prefix_level_params(64);
        let natural_len = 65usize;
        let padded_ring_slots = 2usize;
        let setup = test_setup::<64>(&level_params, padded_ring_slots * 64);
        let fields = setup.expanded.shared_matrix().as_field_slice();

        let ring_elems = extract_setup_prefix_ring_elems::<F, 64>(
            &setup.expanded,
            padded_ring_slots,
            natural_len,
        )
        .expect("extract setup prefix");

        assert_eq!(ring_elems.len(), padded_ring_slots);
        assert_eq!(ring_elems[0].coefficients(), &fields[..64]);
        assert_eq!(ring_elems[1].coefficients()[0], fields[64]);
        assert!(
            ring_elems[1].coefficients()[1..]
                .iter()
                .all(|coeff| coeff.is_zero()),
            "coefficients after natural_len must be zero padded"
        );
    }

    fn assert_commit_setup_prefix_populates_singleton_slot<const D: usize>() {
        let level_params = prefix_level_params(D);
        let opening_batch = OpeningBatch::same_point(4, 1).expect("opening_batch");
        let witness_ring_slots = level_params
            .num_blocks
            .checked_mul(level_params.block_len)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(D).expect("prefix length");
        let natural_len = active_setup_field_len(&level_params, &opening_batch, D)
            .expect("natural len")
            .min(n_prefix);
        let mut setup = test_setup::<D>(&level_params, n_prefix);
        let backend = CpuBackend;
        let prepared = backend.prepare_setup::<D>(&setup).expect("prepared setup");
        let seed_digest = setup_seed_digest(setup.expanded.seed()).expect("digest");
        let slot = commit_setup_prefix::<F, D, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &level_params,
            seed_digest,
            n_prefix,
            natural_len,
        )
        .expect("commit prefix");
        assert_eq!(slot.natural_len, natural_len);
        assert_eq!(slot.padded_len, n_prefix);
        setup.prefix_slots.insert(slot).expect("insert");
        assert_eq!(setup.prefix_slots.len(), 1);
    }

    #[test]
    fn commit_setup_prefix_populates_d32_singleton_slot() {
        assert_commit_setup_prefix_populates_singleton_slot::<32>();
    }

    #[test]
    fn commit_setup_prefix_populates_d64_singleton_slot() {
        assert_commit_setup_prefix_populates_singleton_slot::<64>();
    }
}
