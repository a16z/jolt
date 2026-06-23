use super::fold::{fold_onehot_block, fold_onehot_block_ring};
use super::*;
use akita_field::MulBaseUnreduced;

/// Inner (low) coordinate count for the factorized one-hot column-partials
/// fast path. The high opening coordinates split into `inner_bits` low bits
/// (a small, reusable, cache-resident `low_eq` table) and the remaining high
/// bits (the parallelizable outer blocks). This is a pure performance knob:
/// the computed partials are independent of its value.
const ONEHOT_TENSOR_PARTIALS_INNER_BITS: usize = 12;

impl<F, const D: usize, I: OneHotIndex> AkitaPolyOps<F, D> for OneHotPoly<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
{
    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.onehot_k)
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::fold_blocks: invalid block_len for this polynomial");
        let num_blocks = blocks.num_blocks();
        match blocks {
            OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_onehot_block(flat.block(i), scalars, block_len))
                .collect(),
            OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_onehot_block(flat.block(i), scalars, block_len))
                .collect(),
        }
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::fold_blocks_ring: invalid block_len for this polynomial");
        let num_blocks = blocks.num_blocks();
        match blocks {
            OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_onehot_block_ring(flat.block(i), scalars, block_len))
                .collect(),
            OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_onehot_block_ring(flat.block(i), scalars, block_len))
                .collect(),
        }
    }

    fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        if point.len() != self.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let low_vars = self.onehot_k.trailing_zeros() as usize;
        if low_vars > point.len() {
            return Err(AkitaError::InvalidPointDimension {
                expected: low_vars,
                actual: point.len(),
            });
        }
        let low_weights =
            akita_types::basis_weights(&point[..low_vars], akita_types::BasisMode::Lagrange)?;
        let high_weights =
            akita_types::basis_weights(&point[low_vars..], akita_types::BasisMode::Lagrange)?;
        Ok(self
            .indices
            .iter()
            .enumerate()
            .filter_map(|(chunk_idx, hot_idx)| {
                hot_idx.map(|hot_idx| high_weights[chunk_idx] * low_weights[hot_idx.as_usize()])
            })
            .fold(E::zero(), |acc, weight| acc + weight))
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        if logical_point.len() != self.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.num_vars,
                actual: logical_point.len(),
            });
        }
        let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        let low_vars = self.onehot_k.trailing_zeros() as usize;
        if low_vars > logical_point.len() {
            return Err(AkitaError::InvalidPointDimension {
                expected: low_vars,
                actual: logical_point.len(),
            });
        }
        if split_bits <= low_vars {
            let head_mask = width - 1;
            let low_tail_weights = akita_types::basis_weights(
                &logical_point[split_bits..low_vars],
                akita_types::BasisMode::Lagrange,
            )?;
            let high_weights = akita_types::basis_weights(
                &logical_point[low_vars..],
                akita_types::BasisMode::Lagrange,
            )?;
            let mut partials = vec![E::zero(); width];
            for (chunk_idx, hot_idx) in self.indices.iter().copied().enumerate() {
                let Some(raw) = hot_idx else {
                    continue;
                };
                let raw = raw.as_usize();
                let head = raw & head_mask;
                let low_tail = raw >> split_bits;
                partials[head] += high_weights[chunk_idx] * low_tail_weights[low_tail];
            }
            return Ok(partials);
        }

        let mut point = logical_point.to_vec();
        let mut partials = Vec::with_capacity(width);
        for head in 0..width {
            for (bit, coord) in point.iter_mut().enumerate().take(split_bits) {
                *coord = if ((head >> bit) & 1) == 0 {
                    E::zero()
                } else {
                    E::one()
                };
            }
            partials.push(self.evaluate_extension::<E>(&point)?);
        }
        Ok(partials)
    }

    fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(Vec::new());
        };
        if logical_point.len() != first.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: first.num_vars,
                actual: logical_point.len(),
            });
        }
        let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
        if split_bits > first.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        let _span = tracing::info_span!(
            "onehot_tensor_extension_column_partials_batch",
            num_polys = polys.len(),
            num_vars = first.num_vars,
            onehot_k = first.onehot_k,
            split_bits,
            width
        )
        .entered();
        let low_vars = first.onehot_k.trailing_zeros() as usize;
        let can_share_weights = split_bits <= low_vars
            && low_vars <= logical_point.len()
            && polys
                .iter()
                .all(|poly| poly.num_vars == first.num_vars && poly.onehot_k == first.onehot_k);
        if !can_share_weights {
            return polys
                .iter()
                .map(|poly| poly.tensor_extension_column_partials(logical_point))
                .collect();
        }

        // Non-power-of-two `onehot_k` would let `raw >> split_bits` escape
        // `low_tail_weights`; the sparse fast path only covers the (production)
        // power-of-two case, so fall back to the per-poly reference otherwise.
        if !first.onehot_k.is_power_of_two() {
            return polys
                .iter()
                .map(|poly| poly.tensor_extension_column_partials(logical_point))
                .collect();
        }

        // Factor the high `hi_vars = num_vars - low_vars` opening coordinates
        // into an inner/outer tensor split so the chunk weight
        // `high_weights[(j << inner_bits) | i] == high_eq[j] * low_eq[i]`. This
        // avoids materializing the full `2^hi_vars` weight table and lets the
        // sparse kernel turn the per-chunk multiply into a per-chunk add.
        let hi_vars = first.num_vars - low_vars;
        let inner_bits = hi_vars.min(ONEHOT_TENSOR_PARTIALS_INNER_BITS);
        let low_tail_weights = akita_types::basis_weights(
            &logical_point[split_bits..low_vars],
            akita_types::BasisMode::Lagrange,
        )?;
        let low_eq = akita_types::basis_weights(
            &logical_point[low_vars..low_vars + inner_bits],
            akita_types::BasisMode::Lagrange,
        )?;
        let high_eq = akita_types::basis_weights(
            &logical_point[low_vars + inner_bits..],
            akita_types::BasisMode::Lagrange,
        )?;
        let out = polys
            .iter()
            .map(|poly| {
                poly.tensor_column_partials_from_shared_eq::<E>(
                    split_bits,
                    width,
                    inner_bits,
                    &low_eq,
                    &high_eq,
                    &low_tail_weights,
                )
            })
            .collect::<Vec<_>>();
        Ok(out)
    }

    fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        Ok(Some(self.tensor_packed_sparse_witness::<E>()?))
    }

    fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        let _span = tracing::info_span!(
            "OneHotPoly::tensor_packed_sparse_witness_linear_combination",
            num_polys = polys.len()
        )
        .entered();
        if polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: polys.len(),
                actual: coeffs.len(),
            });
        }
        let first = polys.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "onehot sparse witness linear combination requires at least one polynomial"
                    .to_string(),
            )
        })?;
        let (width, total_evals) = first.tensor_packing_shape::<E>()?;
        let table_len = total_evals / width;
        let basis = (0..width)
            .map(|head| {
                let mut coords = vec![F::zero(); width];
                coords[head] = F::one();
                E::from_base_slice(&coords)
            })
            .collect::<Vec<_>>();
        let weighted_basis = coeffs
            .iter()
            .copied()
            .map(|coeff| {
                basis
                    .iter()
                    .copied()
                    .map(|basis_elem| basis_elem * coeff)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let capacity = polys.iter().map(|poly| poly.indices.len()).sum();
        let same_chunk_layout = polys.iter().all(|poly| {
            poly.onehot_k == first.onehot_k && poly.indices.len() == first.indices.len()
        });
        if same_chunk_layout && first.onehot_k >= width && first.onehot_k.is_multiple_of(width) {
            let tails_per_chunk = first.onehot_k / width;
            #[cfg(feature = "parallel")]
            let target_ranges = rayon::current_num_threads().max(1) * 4;
            #[cfg(not(feature = "parallel"))]
            let target_ranges = 1usize;
            let range_len = (first.indices.len() / target_ranges).max(1 << 12);
            let ranges = (0..first.indices.len())
                .step_by(range_len)
                .map(|start| (start, (start + range_len).min(first.indices.len())))
                .collect::<Vec<_>>();
            let chunks = cfg_into_iter!(ranges)
                .map(|(start, end)| {
                    let mut entries = Vec::with_capacity((end - start) * polys.len());
                    let mut local = Vec::with_capacity(polys.len());
                    for chunk_idx in start..end {
                        local.clear();
                        for (poly_idx, poly) in polys.iter().enumerate() {
                            if coeffs[poly_idx] == E::zero() {
                                continue;
                            }
                            let Some(raw) = poly.indices[chunk_idx] else {
                                continue;
                            };
                            let field_pos = poly.hot_field_position(
                                chunk_idx,
                                raw,
                                "tensor-packed witness batch",
                            )?;
                            let tail = field_pos / width;
                            let head = field_pos % width;
                            debug_assert_eq!(tail / tails_per_chunk, chunk_idx);
                            local.push((tail % tails_per_chunk, weighted_basis[poly_idx][head]));
                        }
                        local.sort_unstable_by_key(|(local_tail, _)| *local_tail);
                        for &(local_tail, value) in &local {
                            let tail = chunk_idx * tails_per_chunk + local_tail;
                            if let Some((last_tail, last_value)) = entries.last_mut() {
                                if *last_tail == tail {
                                    *last_value += value;
                                    if *last_value == E::zero() {
                                        entries.pop();
                                    }
                                    continue;
                                }
                            }
                            if value != E::zero() {
                                entries.push((tail, value));
                            }
                        }
                    }
                    Ok(entries)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let mut entries = Vec::with_capacity(capacity);
            for chunk in chunks {
                entries.extend(chunk);
            }
            return Ok(Some(
                SparseExtensionOpeningWitness::from_sorted_unique_entries(table_len, entries)?,
            ));
        }

        let mut cursors = vec![0usize; polys.len()];
        let mut next_entries = Vec::with_capacity(polys.len());
        for (poly_idx, (poly, &coeff)) in polys.iter().zip(coeffs).enumerate() {
            let (poly_width, poly_total_evals) = poly.tensor_packing_shape::<E>()?;
            if poly_width != width || poly_total_evals != total_evals {
                return Err(AkitaError::InvalidSize {
                    expected: total_evals,
                    actual: poly_total_evals,
                });
            }
            if coeff == E::zero() {
                next_entries.push(None);
                continue;
            }
            next_entries
                .push(poly.next_tensor_packed_sparse_position(&mut cursors[poly_idx], width)?);
        }

        let mut entries = Vec::with_capacity(capacity);
        while let Some(tail) = next_entries
            .iter()
            .filter_map(|entry| entry.map(|(tail, _)| tail))
            .min()
        {
            let mut value = E::zero();
            for (poly_idx, poly) in polys.iter().enumerate() {
                while matches!(next_entries[poly_idx], Some((entry_tail, _)) if entry_tail == tail)
                {
                    let (_, head) = next_entries[poly_idx].expect("entry checked above");
                    value += weighted_basis[poly_idx][head];
                    next_entries[poly_idx] =
                        poly.next_tensor_packed_sparse_position(&mut cursors[poly_idx], width)?;
                }
            }
            entries.push((tail, value));
        }
        Ok(Some(SparseExtensionOpeningWitness::from_sorted_entries(
            table_len, entries,
        )?))
    }

    fn tensor_packed_extension_fold_input<E>(
        &self,
    ) -> Result<FoldInputPoly<'_, F, Self, D>, AkitaError>
    where
        Self: Sized,
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        Ok(FoldInputPoly::ProjectedSparse(
            self.tensor_packed_sparse_ring_poly::<E>()?,
        ))
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::decompose_fold: invalid block_len for this polynomial");
        match blocks {
            OneHotBlocks::SingleChunk(blocks) => self.decompose_fold_onehot::<SingleChunkEntry>(
                blocks, challenges, block_len, num_digits,
            ),
            OneHotBlocks::MultiChunk(blocks) => self.decompose_fold_onehot::<MultiChunkEntry>(
                blocks, challenges, block_len, num_digits,
            ),
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_batched")]
    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        // Materialize per-poly block caches up front so every poly agrees on
        // `block_len` before we touch the batched kernels.
        for poly in polys {
            poly.blocks_for(block_len).expect(
                "OneHotPoly::decompose_fold_batched: invalid block_len for one of the polynomials",
            );
        }
        let first = polys.first()?;
        let (_, first_blocks) = first
            .block_cache
            .get()
            .expect("block cache was just built above");
        match first_blocks {
            OneHotBlocks::SingleChunk(_) => Self::decompose_fold_batched_single_chunk_onehot(
                polys, challenges, block_len, num_digits,
            ),
            OneHotBlocks::MultiChunk(_) => Self::decompose_fold_batched_multi_chunk_onehot(
                polys, challenges, block_len, num_digits,
            ),
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_tensor_batched")]
    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        Self::decompose_fold_batched_tensor_onehot(polys, tensor, block_len, num_digits)
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
    fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        _num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let blocks = self.blocks_for(block_len)?;
        let zero_block_len = n_a.checked_mul(num_digits_open).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "one-hot inner commitment digit block count overflow".to_string(),
            )
        })?;
        let t = backend.onehot_commit_rows::<D>(
            prepared,
            OneHotCommitRowsPlan {
                n_a,
                block_len,
                num_digits_commit,
                blocks: blocks.commit_plan_blocks(),
            },
        )?;

        let mut t_hat = FlatDigitBlocks::zeroed(vec![zero_block_len; t.len()])?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t))
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks.into_iter().zip(t.iter()).for_each(|(dst, t_i)| {
            if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
            }
        });

        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows: t_hat,
        })
    }

    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut evals = vec![F::zero(); total_evals];
        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = chunk_idx
                .checked_mul(self.onehot_k)
                .and_then(|base| base.checked_add(raw.as_usize()))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("onehot direct witness index overflow".to_string())
                })?;
            if field_pos >= evals.len() {
                return Err(AkitaError::InvalidInput(format!(
                    "onehot direct witness index {field_pos} out of range for {} evals",
                    evals.len()
                )));
            }
            evals[field_pos] = F::one();
        }
        Ok(CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(evals),
        ))
    }
}
