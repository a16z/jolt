//! Runtime-only caches for recursive Akita prove levels.
//!
//! These structures sit between the recursive `w` witness and the verifier-
//! facing proof wire. They preserve the commitment-side prover caches that the
//! next recursive level needs, without forcing the prover to round-trip through
//! the proof-oriented flat adapters each time.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};
use akita_types::{AkitaCommitmentHint, FlatDigitBlocks};

/// D-erased prover cache for a recursive commitment hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveCommitmentHintCache<F: FieldCore> {
    decomposed_inner_rows: Vec<i8>,
    decomposed_inner_row_block_sizes: Vec<usize>,
    recomposed_inner_row_coeffs: Vec<F>,
    recomposed_inner_row_block_sizes: Vec<usize>,
    #[cfg(feature = "zk")]
    blinding_digits: Vec<i8>,
    #[cfg(feature = "zk")]
    blinding_block_sizes: Vec<usize>,
    ring_dim: usize,
}

impl<F: FieldCore> RecursiveCommitmentHintCache<F> {
    /// Flatten a typed prover hint into a runtime cache that preserves both
    /// decomposed digit planes and recomposed inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed hint does not carry recomposed inner rows.
    pub fn from_typed<const D: usize>(hint: AkitaCommitmentHint<F, D>) -> Result<Self, AkitaError> {
        #[cfg(feature = "zk")]
        let (flat_hint_digits, recomposed_inner_rows, mut blinding_by_group) =
            hint.into_flat_parts();
        #[cfg(not(feature = "zk"))]
        let (flat_hint_digits, recomposed_inner_rows) = hint.into_flat_parts();
        #[cfg(feature = "zk")]
        let blinding = {
            if blinding_by_group.len() != 1 {
                return Err(AkitaError::InvalidInput(
                    "recursive commitment hint must carry exactly one blinding group".to_string(),
                ));
            }
            blinding_by_group
                .pop()
                .ok_or_else(|| AkitaError::InvalidInput("missing recursive blinding".to_string()))?
        };
        let decomposed_inner_row_block_sizes = flat_hint_digits.block_sizes().to_vec();
        let total_digit_planes: usize = flat_hint_digits.flat_digits().len();
        let mut decomposed_inner_rows = Vec::with_capacity(total_digit_planes * D);
        for plane in flat_hint_digits.flat_digits() {
            decomposed_inner_rows.extend_from_slice(plane);
        }

        let recomposed_inner_rows = recomposed_inner_rows.ok_or_else(|| {
            AkitaError::InvalidInput(
                "missing recomposed inner rows in recursive commitment hint".to_string(),
            )
        })?;
        let recomposed_inner_row_block_sizes: Vec<usize> =
            recomposed_inner_rows.iter().map(Vec::len).collect();
        let total_recomposed_inner_rows: usize = recomposed_inner_row_block_sizes.iter().sum();
        let mut recomposed_inner_row_coeffs = Vec::with_capacity(total_recomposed_inner_rows * D);
        for block in &recomposed_inner_rows {
            for ring in block {
                recomposed_inner_row_coeffs.extend_from_slice(ring.coefficients());
            }
        }
        #[cfg(feature = "zk")]
        let blinding_block_sizes = blinding.block_sizes().to_vec();
        #[cfg(feature = "zk")]
        let total_blinding_planes = blinding.flat_digits().len();
        #[cfg(feature = "zk")]
        let mut blinding_digits = Vec::with_capacity(total_blinding_planes * D);
        #[cfg(feature = "zk")]
        for plane in blinding.flat_digits() {
            blinding_digits.extend_from_slice(plane);
        }

        Ok(Self {
            decomposed_inner_rows,
            decomposed_inner_row_block_sizes,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
            #[cfg(feature = "zk")]
            blinding_digits,
            #[cfg(feature = "zk")]
            blinding_block_sizes,
            ring_dim: D,
        })
    }

    /// Reconstruct the typed prover hint without recomputing inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match the
    /// cache, or if the flattened block metadata is inconsistent.
    pub fn to_typed<const D: usize>(&self) -> Result<AkitaCommitmentHint<F, D>, AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "recursive hint cache D mismatch: cache={}, requested={D}",
                self.ring_dim
            )));
        }
        if self.decomposed_inner_row_block_sizes.len()
            != self.recomposed_inner_row_block_sizes.len()
        {
            return Err(AkitaError::InvalidInput(
                "recursive hint cache block metadata mismatch".to_string(),
            ));
        }

        let (flat_digits, digit_remainder) = self.decomposed_inner_rows.as_chunks::<D>();
        if !digit_remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.decomposed_inner_rows.len(),
            });
        }
        let (flat_recomposed_rows, recomposed_remainder) =
            self.recomposed_inner_row_coeffs.as_chunks::<D>();
        if !recomposed_remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.recomposed_inner_row_coeffs.len(),
            });
        }
        #[cfg(feature = "zk")]
        let (flat_blinding, blinding_remainder) = self.blinding_digits.as_chunks::<D>();
        #[cfg(feature = "zk")]
        if !blinding_remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.blinding_digits.len(),
            });
        }

        let mut digit_offset = 0usize;
        let mut recomposed_offset = 0usize;
        let mut decomposed_inner_rows = Vec::with_capacity(flat_digits.len());
        let mut recomposed_inner_rows =
            Vec::with_capacity(self.recomposed_inner_row_block_sizes.len());

        for (&digit_block_size, &recomposed_block_size) in self
            .decomposed_inner_row_block_sizes
            .iter()
            .zip(self.recomposed_inner_row_block_sizes.iter())
        {
            let digit_end = digit_offset + digit_block_size;
            let recomposed_end = recomposed_offset + recomposed_block_size;
            if digit_end > flat_digits.len() || recomposed_end > flat_recomposed_rows.len() {
                return Err(AkitaError::InvalidInput(
                    "recursive hint cache block data is truncated".to_string(),
                ));
            }

            decomposed_inner_rows.extend_from_slice(&flat_digits[digit_offset..digit_end]);
            recomposed_inner_rows.push(
                flat_recomposed_rows[recomposed_offset..recomposed_end]
                    .iter()
                    .map(|coeffs| CyclotomicRing::from_coefficients(*coeffs))
                    .collect(),
            );
            digit_offset = digit_end;
            recomposed_offset = recomposed_end;
        }

        if digit_offset != flat_digits.len() || recomposed_offset != flat_recomposed_rows.len() {
            return Err(AkitaError::InvalidInput(
                "recursive hint cache has trailing block data".to_string(),
            ));
        }

        let decomposed_inner_rows = FlatDigitBlocks::new(
            decomposed_inner_rows,
            self.decomposed_inner_row_block_sizes.clone(),
        )?;
        #[cfg(feature = "zk")]
        {
            let b_blinding_digits =
                FlatDigitBlocks::new(flat_blinding.to_vec(), self.blinding_block_sizes.clone())?;
            Ok(AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                decomposed_inner_rows,
                recomposed_inner_rows,
                b_blinding_digits,
            ))
        }
        #[cfg(not(feature = "zk"))]
        {
            Ok(AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                decomposed_inner_rows,
                recomposed_inner_rows,
            ))
        }
    }
}
