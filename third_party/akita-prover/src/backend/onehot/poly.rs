use super::*;

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// The polynomial is stored layout-agnostically as the flat list of hot
/// indices supplied at construction. Each op takes `block_len` at call time
/// and the per-block bucketing is materialized lazily on the first call and
/// cached for subsequent calls (as a `(block_len, OneHotBlocks)` pair inside
/// a `OnceLock`). That mirrors how [`DensePoly`](crate::DensePoly) accepts `block_len` per op,
/// and keeps `OneHotPoly` free of the commit-layout parameters it used to
/// bake in at construction.
///
/// Generic over `I`: the index type accepted and stored per chunk. Use `u8`
/// when `onehot_k <= 256` to reduce index storage footprint.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    pub(crate) num_vars: usize,
    pub(crate) onehot_k: usize,
    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    pub(crate) indices: Vec<Option<I>>,
    pub(crate) total_ring_elems: usize,
    pub(crate) block_cache: OnceLock<(usize, OneHotBlocks)>,
    pub(crate) tensor_root_cache: OnceLock<(usize, Arc<SparseRingPoly<F, D>>)>,
    pub(crate) _marker: PhantomData<(F, I)>,
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// The commit-layout split (how blocks are tiled within the polynomial)
    /// is no longer baked in at construction. Each op receives `block_len`
    /// from the caller and the per-block representation is materialized on
    /// demand.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent, any index is out of
    /// range, or `onehot_k` and `D` are not nicely matched.
    pub fn new(onehot_k: usize, indices: Vec<Option<I>>) -> Result<Self, AkitaError> {
        if onehot_k == 0 {
            return Err(AkitaError::InvalidInput(
                "onehot_k must be nonzero".to_string(),
            ));
        }
        if !(onehot_k.is_multiple_of(D) || D.is_multiple_of(onehot_k)) {
            return Err(AkitaError::InvalidInput(format!(
                "onehot_k={onehot_k} and D={D} must be nicely matched (one divides the other)"
            )));
        }
        let total_field_elems = indices.len().checked_mul(onehot_k).ok_or_else(|| {
            AkitaError::InvalidInput("onehot total field element count overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "onehot total field elements {total_field_elems} is not a power of two"
            )));
        }
        if !total_field_elems.is_multiple_of(D) {
            return Err(AkitaError::InvalidInput(format!(
                "total field elements {total_field_elems} is not divisible by D={D}"
            )));
        }
        let total_ring_elems = total_field_elems / D;
        for (chunk_idx, opt) in indices.iter().copied().enumerate() {
            if let Some(raw) = opt {
                let idx = raw.as_usize();
                if idx >= onehot_k {
                    return Err(AkitaError::InvalidInput(format!(
                        "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
                    )));
                }
            }
        }
        Ok(Self {
            num_vars: total_field_elems.trailing_zeros() as usize,
            onehot_k,
            indices,
            total_ring_elems,
            block_cache: OnceLock::new(),
            tensor_root_cache: OnceLock::new(),
            _marker: PhantomData,
        })
    }

    /// Number of field-evaluation slots in each compact one-hot chunk.
    #[inline]
    pub fn onehot_k(&self) -> usize {
        self.onehot_k
    }

    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    #[inline]
    pub fn indices(&self) -> &[Option<I>] {
        &self.indices
    }

    /// Return cached per-block storage, building it on first call for
    /// `block_len`.
    ///
    /// Subsequent calls must pass the same `block_len`; differing `block_len`
    /// is rejected rather than silently rebuilt because it indicates a
    /// layout mismatch between ops on the same polynomial.
    pub(super) fn blocks_for(&self, block_len: usize) -> Result<&OneHotBlocks, AkitaError> {
        // Fast path: cache already built for this `block_len`.
        if let Some((cached_len, blocks)) = self.block_cache.get() {
            if *cached_len == block_len {
                return Ok(blocks);
            }
            return Err(AkitaError::InvalidInput(format!(
                "OneHotPoly was first used with block_len={cached_len} but is now being \
                 used with block_len={block_len}; all ops on the same \
                 polynomial must share a single layout"
            )));
        }
        // Slow path: build blocks and install them. Validate `block_len`
        // *before* building so the error path is cheap.
        if block_len == 0 || !block_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} must be a nonzero power of two"
            )));
        }
        if !self.total_ring_elems.is_multiple_of(block_len) {
            return Err(AkitaError::InvalidSize {
                expected: self.total_ring_elems,
                actual: block_len,
            });
        }
        let (cached_len, blocks) = {
            let _span = tracing::debug_span!("OneHotPoly::build_blocks", block_len).entered();
            self.block_cache.get_or_init(|| {
                let blocks = self
                    .build_blocks_inner(block_len)
                    .expect("block_len validated above");
                (block_len, blocks)
            })
        };
        if *cached_len != block_len {
            // A concurrent caller installed a different `block_len` before
            // our closure ran. Report the mismatch instead of silently
            // accepting the mismatched cache.
            return Err(AkitaError::InvalidInput(format!(
                "OneHotPoly was first used with block_len={cached_len} but is now being \
                 used with block_len={block_len}; all ops on the same \
                 polynomial must share a single layout"
            )));
        }
        Ok(blocks)
    }

    /// Sparse fast path for [`AkitaPolyOps::tensor_extension_column_partials_batch`]
    /// (the `split_bits <= low_vars`, power-of-two `onehot_k`, shared-shape
    /// case). Byte-identical to the dense column partials but exploits the
    /// one-hot structure to replace the per-chunk extension *multiply* of the
    /// dense path with a per-chunk extension *add*.
    ///
    /// The caller supplies the opening point already split into Lagrange
    /// factor tables:
    /// * `low_tail_weights = eq(point[split_bits..low_vars])`
    /// * the high `hi_vars = num_vars - low_vars` coordinates factored as
    ///   `low_eq = eq(point[low_vars..low_vars + inner_bits])` and
    ///   `high_eq = eq(point[low_vars + inner_bits..])`, so that the high
    ///   Lagrange weight of chunk `c = (j << inner_bits) | i` is exactly
    ///   `high_eq[j] * low_eq[i]` (the standard little-endian tensor split of
    ///   the `eq` table). We therefore never materialize the full
    ///   `2^hi_vars`-entry weight table.
    ///
    /// Each chunk carries a single hot position `raw in 0..onehot_k`. We:
    /// 1. scatter `low_eq[i]` into a `raw`-indexed scratch table using *adds
    ///    only* (one add per nonzero chunk),
    /// 2. fold the scratch into a running per-`raw` bucket with one multiply by
    ///    `high_eq[j]` per touched `raw` (cheap: at most `onehot_k` per outer
    ///    block), and
    /// 3. collapse the `onehot_k` buckets into the `width` column partials via
    ///    `partials[raw & (width - 1)] += bucket[raw] * low_tail_weights[raw >> split_bits]`.
    ///
    /// Field addition/multiplication are exactly associative, commutative, and
    /// distributive, so the bucket regrouping and the parallel block split both
    /// yield the identical field element the dense path produces.
    pub(super) fn tensor_column_partials_from_shared_eq<E>(
        &self,
        split_bits: usize,
        width: usize,
        inner_bits: usize,
        low_eq: &[E],
        high_eq: &[E],
        low_tail_weights: &[E],
    ) -> Vec<E>
    where
        E: ExtField<F>,
    {
        let onehot_k = self.onehot_k;
        let head_mask = width - 1;
        let inner_len = low_eq.len();
        let num_blocks = high_eq.len();
        let zero = E::zero();
        debug_assert_eq!(inner_len, 1usize << inner_bits);
        debug_assert_eq!(self.indices.len(), num_blocks * inner_len);

        // Partition the outer blocks into contiguous ranges so the heavy
        // scatter is parallel; each range accumulates an independent per-`raw`
        // bucket which we then reduce (addition is associative, so the result
        // is independent of the range split).
        #[cfg(feature = "parallel")]
        let target_ranges = rayon::current_num_threads().max(1) * 4;
        #[cfg(not(feature = "parallel"))]
        let target_ranges = 1usize;
        let range_len = num_blocks.div_ceil(target_ranges.max(1)).max(1);
        let ranges = (0..num_blocks)
            .step_by(range_len)
            .map(|start| (start, (start + range_len).min(num_blocks)))
            .collect::<Vec<_>>();

        let partial_buckets = cfg_into_iter!(ranges)
            .map(|(jstart, jend)| {
                let mut bucket = vec![zero; onehot_k];
                let mut scratch = vec![zero; onehot_k];
                let mut touched = vec![false; onehot_k];
                let mut touched_raws = Vec::with_capacity(inner_len.min(onehot_k));
                for (jrel, &hj) in high_eq[jstart..jend].iter().enumerate() {
                    let base = (jstart + jrel) << inner_bits;
                    let block = &self.indices[base..base + inner_len];
                    for (hot, &le) in block.iter().copied().zip(low_eq.iter()) {
                        if let Some(raw) = hot {
                            let raw = raw.as_usize();
                            if !touched[raw] {
                                touched[raw] = true;
                                touched_raws.push(raw);
                            }
                            scratch[raw] += le;
                        }
                    }
                    for raw in touched_raws.drain(..) {
                        let slot = &mut scratch[raw];
                        bucket[raw] += hj * *slot;
                        *slot = zero;
                        touched[raw] = false;
                    }
                }
                bucket
            })
            .collect::<Vec<_>>();

        let mut bucket = vec![zero; onehot_k];
        for partial in &partial_buckets {
            for (acc, part) in bucket.iter_mut().zip(partial.iter()) {
                *acc += *part;
            }
        }

        let mut partials = vec![zero; width];
        for (raw, &value) in bucket.iter().enumerate() {
            if value != zero {
                partials[raw & head_mask] += value * low_tail_weights[raw >> split_bits];
            }
        }
        partials
    }

    pub(super) fn tensor_packed_sparse_witness<E>(
        &self,
    ) -> Result<SparseExtensionOpeningWitness<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (width, total_evals) = self.tensor_packing_shape::<E>()?;
        let table_len = total_evals / width;
        let _span = tracing::info_span!(
            "OneHotPoly::tensor_packed_sparse_witness",
            width,
            table_len,
            chunks = self.indices.len()
        )
        .entered();
        let mut entries = Vec::with_capacity(self.indices.len());
        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = self.hot_field_position(chunk_idx, raw, "tensor-packed witness")?;
            let tail = field_pos / width;
            let head = field_pos % width;
            let mut coords = vec![F::zero(); width];
            coords[head] = F::one();
            entries.push((tail, E::from_base_slice(&coords)));
        }
        SparseExtensionOpeningWitness::new(table_len, entries)
    }

    pub(super) fn tensor_packed_sparse_ring_poly<E>(
        &self,
    ) -> Result<Arc<SparseRingPoly<F, D>>, AkitaError>
    where
        F: FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        let (width, total_evals) = self.tensor_packing_shape::<E>()?;
        let _span = tracing::info_span!(
            "OneHotPoly::tensor_packed_sparse_ring_poly",
            width,
            total_evals,
            chunks = self.indices.len()
        )
        .entered();
        if !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(
                "tensor width must divide root ring dimension".to_string(),
            ));
        }
        let double_width = width.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidInput(
                "tensor width is too large for root ring projection".to_string(),
            )
        })?;
        if D < double_width {
            return Err(AkitaError::InvalidInput(
                "root ring dimension must be at least twice the tensor width".to_string(),
            ));
        }
        let packed_len = D / width;
        let half = D / double_width;
        let step = D / double_width;
        let total_ring_elems = total_evals / D;
        if let Some((cached_width, poly)) = self.tensor_root_cache.get() {
            if *cached_width == width {
                return Ok(Arc::clone(poly));
            }
        }
        let mut coeffs = Vec::with_capacity(self.indices.len() * width.min(2));

        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = self.hot_field_position(chunk_idx, raw, "tensor-projected ring")?;
            let tail = field_pos / width;
            let coord = field_pos % width;
            let ring_idx = tail / packed_len;
            let slot_idx = tail % packed_len;
            if slot_idx < half {
                let shift = slot_idx;
                if coord == 0 {
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift, 1)?);
                } else {
                    let pos_offset = coord * step;
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift + pos_offset, 1)?);
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift + D - pos_offset, -1)?);
                }
            } else {
                let shift = slot_idx - half + D / 2;
                if coord == 0 {
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift, 1)?);
                } else {
                    let pos_offset = coord * step;
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift - pos_offset, 1)?);
                    coeffs.push(SparseRingCoeff::new(ring_idx, shift + pos_offset, 1)?);
                }
            }
        }

        let poly = if self.onehot_k >= D {
            SparseRingPoly::<F, D>::from_sorted_packed_coeffs(
                self.num_vars,
                total_ring_elems,
                coeffs,
            )
        } else {
            SparseRingPoly::<F, D>::from_packed_coeffs(self.num_vars, total_ring_elems, coeffs)
        }?;
        let poly = Arc::new(poly);
        let _ = self.tensor_root_cache.set((width, Arc::clone(&poly)));
        if let Some((cached_width, cached_poly)) = self.tensor_root_cache.get() {
            if *cached_width == width {
                return Ok(Arc::clone(cached_poly));
            }
        }
        Ok(poly)
    }

    pub(super) fn tensor_packing_shape<E>(&self) -> Result<(usize, usize), AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        Ok((width, total_evals))
    }

    pub(super) fn hot_field_position(
        &self,
        chunk_idx: usize,
        raw: I,
        context: &'static str,
    ) -> Result<usize, AkitaError> {
        chunk_idx
            .checked_mul(self.onehot_k)
            .and_then(|base| base.checked_add(raw.as_usize()))
            .ok_or_else(|| AkitaError::InvalidInput(format!("onehot {context} index overflow")))
    }

    pub(super) fn next_tensor_packed_sparse_position(
        &self,
        cursor: &mut usize,
        width: usize,
    ) -> Result<Option<(usize, usize)>, AkitaError> {
        while *cursor < self.indices.len() {
            let chunk_idx = *cursor;
            *cursor += 1;
            let Some(raw) = self.indices[chunk_idx] else {
                continue;
            };
            let field_pos =
                self.hot_field_position(chunk_idx, raw, "tensor-packed witness batch")?;
            return Ok(Some((field_pos / width, field_pos % width)));
        }
        Ok(None)
    }

    pub(super) fn build_blocks_inner(&self, block_len: usize) -> Result<OneHotBlocks, AkitaError> {
        // `blocks_for` has already validated that `block_len` is a nonzero
        // power of two and that `total_ring_elems % block_len == 0`, and
        // `OneHotPoly::new` has validated that K, D, and every per-chunk
        // index are in range. Here we only need to compute `num_blocks`
        // for the flat-layout offsets array and check that `block_len`
        // and `D` fit in the packed entry field widths.
        if u32::try_from(block_len).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} exceeds u32::MAX and cannot be packed into an entry"
            )));
        }
        // Coefficient indices inside a ring element are `< D` and get
        // packed as `u16` in the entry types below (see
        // `SingleChunkEntry::coeff_idx` and `MultiChunkEntry::nonzero_coeffs`).
        // Reject out-of-range `D` here rather than silently truncating below.
        if D > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "D={D} exceeds 65536 and cannot be packed into SingleChunkEntry::coeff_idx / MultiChunkEntry::nonzero_coeffs (both `u16`)"
            )));
        }
        let num_blocks = self.total_ring_elems / block_len;

        // The single-chunk (one-hot-chunk-per-ring-element) layout
        // applies when K >= D && D | K; otherwise fall back to the
        // multi-chunk layout.
        if self.onehot_k >= D && self.onehot_k.is_multiple_of(D) {
            Ok(OneHotBlocks::SingleChunk(
                FlatBlocks::<SingleChunkEntry>::from_indices(
                    self.onehot_k,
                    &self.indices,
                    block_len,
                    D,
                    num_blocks,
                )?,
            ))
        } else {
            Ok(OneHotBlocks::MultiChunk(
                FlatBlocks::<MultiChunkEntry>::from_indices(
                    self.onehot_k,
                    &self.indices,
                    block_len,
                    D,
                    num_blocks,
                )?,
            ))
        }
    }
}
