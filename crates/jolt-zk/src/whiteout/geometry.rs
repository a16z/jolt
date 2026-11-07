use jolt_algebra::JoltField;
use jolt_crypto::lattice::{AjtaiLevel, GadgetDigits, MatrixShape, CHALLENGE_RING_DIM};

use super::response::FIELD_LEVELS;
use super::sigma::response_num_vars;
use super::{
    ImageCompression, PadGeometry, ResponseLayout, ResponseSegment, Segment, SegmentRole,
    SegmentTable, SigmaParams, WhiteoutError, RESPONSE_LOG_BASIS,
};

/// Shape-only sizing shared by preparation and Sigma construction.
pub struct SigmaGeometry {
    pub(crate) table: SegmentTable,
    pub(crate) params: SigmaParams,
    pub(crate) outputs: usize,
    message_rings: usize,
    compressed_images: Vec<usize>,
}

impl SigmaGeometry {
    pub fn new<const D: usize>(
        residual: Vec<Segment>,
        blocks: impl IntoIterator<Item = (MatrixShape, bool)>,
        pads: &PadGeometry<D>,
    ) -> Result<Self, WhiteoutError> {
        if residual.iter().any(|segment| {
            matches!(
                segment.role,
                SegmentRole::PadDigits | SegmentRole::OperandDigits | SegmentRole::PadRandomness
            )
        }) {
            return Err(WhiteoutError::InvalidRelation(
                "residual segments may not hold pads",
            ));
        }
        let mut segments = residual;
        segments.extend(pads.segments());
        let table = SegmentTable::new(segments)?;
        if !D.is_multiple_of(CHALLENGE_RING_DIM)
            || table.segments().any(|segment| segment.native_dim != D)
        {
            return Err(WhiteoutError::InvalidRelation(
                "every segment must have the setup's native ring dimension",
            ));
        }
        let blocks: Vec<_> = blocks.into_iter().collect();
        if blocks.iter().any(|(shape, _)| shape.ring_dim != D) {
            return Err(WhiteoutError::InvalidRelation(
                "residual ring dimension differs from pads",
            ));
        }
        let outputs =
            Self::output_rows(blocks.iter().map(|(shape, _)| *shape), pads.matrix_shape())?;
        let mut compressed_images = blocks
            .iter()
            .filter(|(_, compressed)| *compressed)
            .map(|(shape, _)| shape.rows.checked_mul(D).ok_or(WhiteoutError::Overflow))
            .collect::<Result<Vec<_>, _>>()?;
        compressed_images.push(
            pads.matrix_shape()
                .rows
                .checked_mul(D)
                .ok_or(WhiteoutError::Overflow)?,
        );
        Ok(Self {
            params: SigmaParams::derive(&table)?,
            outputs,
            table,
            message_rings: Self::v_message_rings(outputs, D)?,
            compressed_images,
        })
    }

    pub(crate) fn output_rows(
        blocks: impl IntoIterator<Item = MatrixShape>,
        pads: MatrixShape,
    ) -> Result<usize, WhiteoutError> {
        blocks.into_iter().try_fold(pads.rows, |rows, shape| {
            rows.checked_add(shape.rows).ok_or(WhiteoutError::Overflow)
        })
    }

    pub(crate) fn v_message_rings(outputs: usize, ring_dim: usize) -> Result<usize, WhiteoutError> {
        outputs
            .checked_mul(ring_dim)
            .and_then(|coefficients| (coefficients / CHALLENGE_RING_DIM).checked_add(3))
            .ok_or(WhiteoutError::Overflow)
    }

    pub(crate) fn v_level_of(
        message_rings: usize,
        rank: usize,
    ) -> Result<AjtaiLevel, WhiteoutError> {
        Ok(AjtaiLevel {
            matrix: MatrixShape {
                rows: rank,
                width: message_rings
                    .checked_mul(FIELD_LEVELS)
                    .ok_or(WhiteoutError::Overflow)?,
                ring_dim: CHALLENGE_RING_DIM,
            },
            digits: GadgetDigits {
                log_basis: RESPONSE_LOG_BASIS,
                levels: FIELD_LEVELS,
            },
            message_rings,
            randomness_columns: 0,
        })
    }

    pub(crate) fn v_level(&self, rank: usize) -> Result<AjtaiLevel, WhiteoutError> {
        Self::v_level_of(self.message_rings, rank)
    }

    pub fn v_shape(&self, rank: usize) -> Result<MatrixShape, WhiteoutError> {
        Ok(self.v_level(rank)?.matrix)
    }

    pub fn response<F: JoltField>(
        &self,
        rank: usize,
        compression: &dyn ImageCompression<F>,
    ) -> Result<ResponseLayout, WhiteoutError> {
        Self::response_layout(
            &self.table,
            &self.params,
            self.v_level(rank)?,
            self.compressed_images.iter().copied(),
            compression,
        )
    }

    pub(crate) fn response_layout<F: JoltField>(
        table: &SegmentTable,
        params: &SigmaParams,
        level: AjtaiLevel,
        images: impl Iterator<Item = usize>,
        compression: &dyn ImageCompression<F>,
    ) -> Result<ResponseLayout, WhiteoutError> {
        let mut segments: Vec<ResponseSegment> = table
            .segments()
            .zip(params.response_digits())
            .enumerate()
            .map(|(segment, (entry, &digits))| ResponseSegment::Response {
                segment,
                coefficients: entry.num_coefficients(),
                digits,
            })
            .collect();
        segments.push(ResponseSegment::V {
            coefficients: level
                .message_rings
                .checked_mul(CHALLENGE_RING_DIM)
                .ok_or(WhiteoutError::Overflow)?,
        });
        segments.push(ResponseSegment::Theta);
        let v_image = level
            .matrix
            .rows
            .checked_mul(CHALLENGE_RING_DIM)
            .ok_or(WhiteoutError::Overflow)?;
        for (image, coefficients) in images.chain(std::iter::once(v_image)).enumerate() {
            segments.push(ResponseSegment::Compression {
                image,
                coefficients,
                digits: compression.witness_len(coefficients)?,
            });
        }
        let layout = ResponseLayout { segments };
        let _ = response_num_vars(layout.checked_len()?)?;
        Ok(layout)
    }
}
