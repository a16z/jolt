use super::*;

#[cfg(feature = "zk")]
#[cfg(test)]
use crate::protocol::masking::zk_matrix_cyclic_digit_rows;

#[cfg(feature = "zk")]
pub(super) fn add_zk_d_blinding_cyclic_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    n_d: usize,
    blinding: &FlatDigitBlocks<D>,
    rows: &mut [CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: CyclicRowsComputeBackend<F>,
{
    add_zk_blinding_cyclic_rows(
        backend,
        prepared,
        ZkBlindingRole::D,
        n_d,
        blinding.flat_digits().len(),
        blinding,
        rows,
    )
}

#[cfg(feature = "zk")]
pub(super) fn add_zk_b_blinding_cyclic_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    n_b: usize,
    row_width: usize,
    blinding: &FlatDigitBlocks<D>,
    rows: &mut [CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: CyclicRowsComputeBackend<F>,
{
    add_zk_blinding_cyclic_rows(
        backend,
        prepared,
        ZkBlindingRole::B,
        n_b,
        row_width,
        blinding,
        rows,
    )
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy)]
enum ZkBlindingRole {
    B,
    D,
}

#[cfg(feature = "zk")]
fn add_zk_blinding_cyclic_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    role: ZkBlindingRole,
    row_len: usize,
    row_width: usize,
    blinding: &FlatDigitBlocks<D>,
    rows: &mut [CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: CyclicRowsComputeBackend<F>,
{
    if blinding.is_empty() {
        return Ok(());
    }
    if rows.len() != row_len {
        return Err(AkitaError::InvalidProof);
    }
    let blinding_rows = match role {
        ZkBlindingRole::B => backend.zk_b_cyclic_digit_rows::<D>(
            prepared,
            row_len,
            row_width,
            blinding.flat_digits(),
        )?,
        ZkBlindingRole::D => backend.zk_d_cyclic_digit_rows::<D>(
            prepared,
            row_len,
            row_width,
            blinding.flat_digits(),
        )?,
    };
    if blinding_rows.len() != row_len {
        return Err(AkitaError::InvalidProof);
    }
    for (row, b_blinding_row) in rows.iter_mut().zip(blinding_rows) {
        *row += b_blinding_row;
    }
    Ok(())
}

fn checked_digit_plane_sum(sizes: &[usize]) -> Result<usize, AkitaError> {
    sizes.iter().try_fold(0usize, |acc, &size| {
        acc.checked_add(size).ok_or(AkitaError::InvalidProof)
    })
}

fn repeated_b_planes_per_claim(
    block_sizes: &[usize],
    blocks_per_claim: usize,
) -> Result<usize, AkitaError> {
    if blocks_per_claim == 0
        || block_sizes.is_empty()
        || !block_sizes.len().is_multiple_of(blocks_per_claim)
    {
        return Err(AkitaError::InvalidProof);
    }
    let block_width = block_sizes[0];
    if block_width == 0 {
        return Err(AkitaError::InvalidProof);
    }
    for &size in block_sizes {
        if size != block_width {
            return Err(AkitaError::InvalidProof);
        }
    }
    block_width
        .checked_mul(blocks_per_claim)
        .ok_or(AkitaError::InvalidProof)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn repeated_b_commitment_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    n_b: usize,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    num_polys_per_commitment_group: &[usize],
    blocks_per_claim: usize,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: CyclicRowsComputeBackend<F>,
{
    if num_polys_per_commitment_group.is_empty() || blocks_per_claim == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let mut num_group_polys = 0usize;
    let mut max_group_poly_count = 0usize;
    for &group_poly_count in num_polys_per_commitment_group {
        if group_poly_count == 0 {
            return Err(AkitaError::InvalidProof);
        }
        num_group_polys = num_group_polys
            .checked_add(group_poly_count)
            .ok_or(AkitaError::InvalidProof)?;
        max_group_poly_count = max_group_poly_count.max(group_poly_count);
    }
    let expected_block_count = num_group_polys
        .checked_mul(blocks_per_claim)
        .ok_or(AkitaError::InvalidProof)?;
    if t_hat.block_count() != expected_block_count {
        return Err(AkitaError::InvalidProof);
    }
    let planes_per_claim = repeated_b_planes_per_claim(t_hat.block_sizes(), blocks_per_claim)?;
    let row_width = max_group_poly_count
        .checked_mul(planes_per_claim)
        .ok_or(AkitaError::InvalidProof)?;
    #[cfg(not(feature = "zk"))]
    let b_blinding_digits =
        vec![FlatDigitBlocks::<D>::empty(); num_polys_per_commitment_group.len()];
    if b_blinding_digits.len() != num_polys_per_commitment_group.len() {
        return Err(AkitaError::InvalidProof);
    }

    let mut groups = Vec::with_capacity(num_polys_per_commitment_group.len());
    let mut block_offset = 0usize;
    let mut plane_offset = 0usize;
    for (&group_poly_count, blinding) in num_polys_per_commitment_group
        .iter()
        .zip(b_blinding_digits.iter())
    {
        let group_block_count = group_poly_count
            .checked_mul(blocks_per_claim)
            .ok_or(AkitaError::InvalidProof)?;
        let next_block_offset = block_offset
            .checked_add(group_block_count)
            .ok_or(AkitaError::InvalidProof)?;
        let group_block_sizes = t_hat
            .block_sizes()
            .get(block_offset..next_block_offset)
            .ok_or(AkitaError::InvalidProof)?;
        let group_planes = checked_digit_plane_sum(group_block_sizes)?;
        let expected_group_planes = group_poly_count
            .checked_mul(planes_per_claim)
            .ok_or(AkitaError::InvalidProof)?;
        if group_planes != expected_group_planes {
            return Err(AkitaError::InvalidProof);
        }
        let next_plane_offset = plane_offset
            .checked_add(group_planes)
            .ok_or(AkitaError::InvalidProof)?;
        t_hat
            .flat_digits()
            .get(plane_offset..next_plane_offset)
            .ok_or(AkitaError::InvalidProof)?;
        groups.push((plane_offset, next_plane_offset, blinding));
        block_offset = next_block_offset;
        plane_offset = next_plane_offset;
    }
    if block_offset != t_hat.block_count() || plane_offset != t_hat.flat_digits().len() {
        return Err(AkitaError::InvalidProof);
    }

    let mut rows = Vec::with_capacity(num_polys_per_commitment_group.len() * n_b);
    for (start, end, blinding) in groups {
        #[cfg(not(feature = "zk"))]
        let _ = blinding;
        let group_digits = t_hat
            .flat_digits()
            .get(start..end)
            .ok_or(AkitaError::InvalidProof)?;
        let group_rows = if group_digits.len() == row_width {
            backend.cyclic_digit_rows::<D>(prepared, n_b, group_digits, log_basis)?
        } else {
            let mut padded = vec![[0i8; D]; row_width];
            padded[..group_digits.len()].copy_from_slice(group_digits);
            backend.cyclic_digit_rows::<D>(prepared, n_b, &padded, log_basis)?
        };
        if group_rows.len() != n_b {
            return Err(AkitaError::InvalidProof);
        }
        rows.extend(group_rows);
        #[cfg(feature = "zk")]
        {
            let row_start = rows.len() - n_b;
            add_zk_b_blinding_cyclic_rows(
                backend,
                prepared,
                n_b,
                blinding.flat_digits().len(),
                blinding,
                &mut rows[row_start..row_start + n_b],
            )?;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::ComputeBackendSetup;
    use crate::kernels::crt_ntt::build_ntt_slot;
    use crate::kernels::linear::mat_vec_mul_ntt_single_i8_cyclic;
    use crate::{AkitaProverSetup, CpuBackend};
    use akita_field::Fp64;
    use akita_types::{AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix};

    #[test]
    fn nonuniform_groups_use_max_group_row_width() {
        type F = Fp64<4294967197>;
        const D: usize = 32;
        let n_b = 2;
        let num_polys_per_commitment_group = [2usize, 1usize];
        let blocks_per_claim = 1;
        let block_width = 3;
        let log_basis = 3;
        let row_width = num_polys_per_commitment_group
            .iter()
            .copied()
            .max()
            .unwrap()
            * block_width;
        let setup_rows: Vec<CyclotomicRing<F, D>> = (0..(n_b * row_width))
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_i64(((idx * 17 + k * 5) % 43) as i64 - 21)
                }))
            })
            .collect();
        let expanded = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 8,
                max_num_batched_polys: 3,
                gen_ring_dim: D,
                max_setup_len: setup_rows.len(),
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
                public_matrix_seed: [7u8; 32],
            },
            FlatMatrix::from_ring_slice::<D>(&setup_rows),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![F::zero(); D], D),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![F::zero(); D], D),
        );
        let setup = AkitaProverSetup::from_seed_validated_expanded(expanded).expect("valid setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let blocks: Vec<Vec<[i8; D]>> = (0..3)
            .map(|block| {
                (0..block_width)
                    .map(|plane| std::array::from_fn(|k| ((block * 3 + plane + k) % 5) as i8 - 2))
                    .collect()
            })
            .collect();
        let t_hat = FlatDigitBlocks::from_blocks(blocks);
        #[cfg(feature = "zk")]
        let b_blinding = vec![FlatDigitBlocks::<D>::empty(); num_polys_per_commitment_group.len()];
        let got = repeated_b_commitment_rows::<F, _, D>(
            &CpuBackend,
            &prepared,
            n_b,
            &t_hat,
            #[cfg(feature = "zk")]
            &b_blinding,
            &num_polys_per_commitment_group,
            blocks_per_claim,
            log_basis,
        )
        .expect("commitment rows");

        let slot = build_ntt_slot(
            setup
                .expanded
                .shared_matrix()
                .ring_view::<D>(n_b, row_width)
                .expect("valid B view"),
        )
        .expect("valid NTT slot");
        let first = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
            &slot,
            n_b,
            row_width,
            &t_hat.flat_digits()[..row_width],
            log_basis,
        )
        .expect("first expected rows");
        let mut second_digits = vec![[0i8; D]; row_width];
        second_digits[..block_width].copy_from_slice(&t_hat.flat_digits()[row_width..]);
        let second = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
            &slot,
            n_b,
            row_width,
            &second_digits,
            log_basis,
        )
        .expect("second expected rows");
        let expected = first.into_iter().chain(second).collect::<Vec<_>>();
        assert_eq!(got, expected);

        let nonuniform_blocks: Vec<Vec<[i8; D]>> = [block_width, block_width - 1, block_width]
            .into_iter()
            .enumerate()
            .map(|(block, width)| {
                (0..width)
                    .map(|plane| std::array::from_fn(|k| ((block * 7 + plane + k) % 5) as i8 - 2))
                    .collect()
            })
            .collect();
        let nonuniform_t_hat = FlatDigitBlocks::from_blocks(nonuniform_blocks);
        assert!(repeated_b_commitment_rows::<F, _, D>(
            &CpuBackend,
            &prepared,
            n_b,
            &nonuniform_t_hat,
            #[cfg(feature = "zk")]
            &b_blinding,
            &num_polys_per_commitment_group,
            blocks_per_claim,
            log_basis,
        )
        .is_err());

        let same_total_nonuniform_blocks: Vec<Vec<[i8; D]>> =
            [block_width, block_width + 1, block_width + 1, block_width]
                .into_iter()
                .enumerate()
                .map(|(block, width)| {
                    (0..width)
                        .map(|plane| {
                            std::array::from_fn(|k| ((block * 11 + plane + k) % 5) as i8 - 2)
                        })
                        .collect()
                })
                .collect();
        let same_total_nonuniform_t_hat =
            FlatDigitBlocks::from_blocks(same_total_nonuniform_blocks);
        assert!(repeated_b_commitment_rows::<F, _, D>(
            &CpuBackend,
            &prepared,
            n_b,
            &same_total_nonuniform_t_hat,
            #[cfg(feature = "zk")]
            &b_blinding,
            &[1usize, 1usize],
            2,
            log_basis,
        )
        .is_err());
    }

    #[cfg(feature = "zk")]
    #[test]
    fn nonuniform_groups_insert_group_local_blinding() {
        type F = Fp64<4294967197>;
        const D: usize = 32;
        let n_b = 2;
        let num_polys_per_commitment_group = [2usize, 1usize];
        let blocks_per_claim = 1;
        let block_width = 3;
        let blinding_width = 2;
        let log_basis = 3;
        let row_width = num_polys_per_commitment_group
            .iter()
            .map(|&count| count * block_width)
            .max()
            .unwrap();
        let setup_rows: Vec<CyclotomicRing<F, D>> = (0..(n_b * row_width))
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_i64(((idx * 19 + k * 7) % 47) as i64 - 23)
                }))
            })
            .collect();
        let zk_b_rows: Vec<CyclotomicRing<F, D>> = (0..(n_b * blinding_width))
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_i64(((idx * 23 + k * 11) % 53) as i64 - 26)
                }))
            })
            .collect();
        let expanded = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 8,
                max_num_batched_polys: 3,
                gen_ring_dim: D,
                max_setup_len: setup_rows.len(),
                #[cfg(feature = "zk")]
                max_zk_b_len: zk_b_rows.len(),
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
                public_matrix_seed: [9u8; 32],
            },
            FlatMatrix::from_ring_slice::<D>(&setup_rows),
            #[cfg(feature = "zk")]
            FlatMatrix::from_ring_slice::<D>(&zk_b_rows),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![F::zero(); D], D),
        );
        let setup = AkitaProverSetup::from_seed_validated_expanded(expanded).expect("valid setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let blocks: Vec<Vec<[i8; D]>> = (0..3)
            .map(|block| {
                (0..block_width)
                    .map(|plane| std::array::from_fn(|k| ((block * 5 + plane + k) % 7) as i8 - 3))
                    .collect()
            })
            .collect();
        let t_hat = FlatDigitBlocks::from_blocks(blocks);
        let b_blinding: Vec<FlatDigitBlocks<D>> = (0..num_polys_per_commitment_group.len())
            .map(|group| {
                FlatDigitBlocks::new(
                    (0..blinding_width)
                        .map(|plane| {
                            std::array::from_fn(|k| ((group * 11 + plane * 3 + k) % 5) as i8 - 2)
                        })
                        .collect(),
                    vec![blinding_width],
                )
                .expect("valid blinding block")
            })
            .collect();
        let got = repeated_b_commitment_rows::<F, _, D>(
            &CpuBackend,
            &prepared,
            n_b,
            &t_hat,
            &b_blinding,
            &num_polys_per_commitment_group,
            blocks_per_claim,
            log_basis,
        )
        .expect("commitment rows");

        let slot = build_ntt_slot(
            setup
                .expanded
                .shared_matrix()
                .ring_view::<D>(n_b, row_width)
                .expect("valid B view"),
        )
        .expect("valid NTT slot");
        let mut first_digits = vec![[0i8; D]; row_width];
        first_digits[..(2 * block_width)]
            .copy_from_slice(&t_hat.flat_digits()[..(2 * block_width)]);
        let mut first = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
            &slot,
            n_b,
            row_width,
            &first_digits,
            log_basis,
        )
        .expect("first expected rows");
        let first_blinding = zk_matrix_cyclic_digit_rows(
            setup.expanded.zk_b_matrix(),
            n_b,
            0,
            blinding_width,
            b_blinding[0].flat_digits(),
        )
        .unwrap();
        for (row, blinding) in first.iter_mut().zip(first_blinding) {
            *row += blinding;
        }

        let mut second_digits = vec![[0i8; D]; row_width];
        second_digits[..block_width]
            .copy_from_slice(&t_hat.flat_digits()[(2 * block_width)..(3 * block_width)]);
        let mut second = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
            &slot,
            n_b,
            row_width,
            &second_digits,
            log_basis,
        )
        .expect("second expected rows");
        let second_blinding = zk_matrix_cyclic_digit_rows(
            setup.expanded.zk_b_matrix(),
            n_b,
            0,
            blinding_width,
            b_blinding[1].flat_digits(),
        )
        .unwrap();
        for (row, blinding) in second.iter_mut().zip(second_blinding) {
            *row += blinding;
        }

        let expected = first.into_iter().chain(second).collect::<Vec<_>>();
        assert_eq!(got, expected);
    }
}
