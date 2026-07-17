//! The canonical native-Akita `OneHotTrace` commitment layout. Every semantic
//! column returned by
//! [`one_hot_trace_columns`](super::packing::one_hot_trace_columns) is a
//! strict `K x T` one-hot polynomial, and the columns open together at one
//! common `(cycle || address)` point.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::OpeningsError;

use super::super::JoltCommittedPolynomial;
use super::packing::{one_hot_trace_columns, OneHotTraceShape};

/// `OneHotTrace` is committed as one native Akita group of one-hot columns.
pub const ONE_HOT_TRACE_LAYOUT: OneHotTraceLayout = OneHotTraceLayout;

/// The one protocol layout for the per-proof `OneHotTrace` commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotTraceLayout;

/// The canonical column order and uniform arity for one proof.
pub struct OneHotTraceLayoutPlan {
    pub columns: Vec<JoltCommittedPolynomial>,
    pub column_arity: usize,
}

/// The commitment-object setup shape the layout requires.
pub struct OneHotTraceSetupShape {
    pub num_vars: usize,
    pub num_polys: usize,
}

impl OneHotTraceLayout {
    /// The canonical object layout for `shape`.
    pub fn plan(&self, shape: &OneHotTraceShape) -> Result<OneHotTraceLayoutPlan, OpeningsError> {
        let columns = one_hot_trace_columns(shape)
            .map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        Ok(OneHotTraceLayoutPlan {
            columns,
            column_arity: shape.log_k_chunk + shape.log_t,
        })
    }

    /// The commitment-object setup shape.
    pub fn setup_shape(
        &self,
        shape: &OneHotTraceShape,
    ) -> Result<OneHotTraceSetupShape, OpeningsError> {
        let plan = self.plan(shape)?;
        Ok(OneHotTraceSetupShape {
            num_vars: plan.column_arity,
            num_polys: plan.columns.len(),
        })
    }

    /// A protocol-owned digest of the exact native batch layout. This binds
    /// the commitment and verifier setup to the ordered column identities,
    /// dimensions, and layout version; it is never supplied by the proof.
    pub fn layout_digest(&self, shape: &OneHotTraceShape) -> Result<[u8; 32], OpeningsError> {
        let OneHotTraceLayoutPlan {
            columns,
            column_arity,
        } = self.plan(shape)?;
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(b"jolt/akita/one_hot_trace/native-one-hot-columns/v2");
        append_usize(&mut hasher, column_arity);
        append_usize(&mut hasher, columns.len());
        append_usize(&mut hasher, shape.log_t);
        append_usize(&mut hasher, shape.log_k_chunk);
        append_usize(&mut hasher, shape.ra_layout.instruction());
        append_usize(&mut hasher, shape.ra_layout.bytecode());
        append_usize(&mut hasher, shape.ra_layout.ram());
        for column in columns {
            match column {
                JoltCommittedPolynomial::InstructionRa(index) => {
                    hasher.update([0]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::BytecodeRa(index) => {
                    hasher.update([1]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::RamRa(index) => {
                    hasher.update([2]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::UnsignedIncChunk(index) => {
                    hasher.update([3]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::UnsignedIncMsb => hasher.update([4]),
                other => {
                    return Err(OpeningsError::InvalidBatch(format!(
                        "non-OneHotTrace polynomial {other:?} in native one-hot layout"
                    )));
                }
            }
        }
        Ok(hasher.finalize().into())
    }

    /// Maps a column's leaf-claim point from `(address || cycle)` to the
    /// row-major committed order `(cycle || address)`.
    pub fn column_point<F: Field>(
        &self,
        polynomial: JoltCommittedPolynomial,
        chunk_width: usize,
        leaf_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let invalid = |message: String| OpeningsError::InvalidBatch(message);
        if !matches!(
            polynomial,
            JoltCommittedPolynomial::InstructionRa(_)
                | JoltCommittedPolynomial::BytecodeRa(_)
                | JoltCommittedPolynomial::RamRa(_)
                | JoltCommittedPolynomial::UnsignedIncChunk(_)
                | JoltCommittedPolynomial::UnsignedIncMsb
        ) {
            return Err(invalid(format!(
                "polynomial {polynomial:?} is not a OneHotTrace column"
            )));
        }
        if leaf_point.len() < chunk_width {
            return Err(invalid(format!(
                "OneHotTrace leaf point has {} variables, below its \
                 {chunk_width}-variable address block",
                leaf_point.len()
            )));
        }
        let (address, cycle) = leaf_point.split_at(chunk_width);
        let mut point = Vec::with_capacity(leaf_point.len());
        point.extend_from_slice(cycle);
        point.extend_from_slice(address);
        Ok(point)
    }
}

fn append_usize(hasher: &mut Blake2b<U32>, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn shape(log_t: usize) -> OneHotTraceShape {
        OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t,
            log_k_chunk: 8,
        }
    }

    #[test]
    fn native_layout_is_uniform_and_digest_bound() {
        let OneHotTraceLayoutPlan {
            columns,
            column_arity,
        } = ONE_HOT_TRACE_LAYOUT.plan(&shape(5)).unwrap();
        assert_eq!(column_arity, 13);
        assert_eq!(
            columns.last(),
            Some(&JoltCommittedPolynomial::UnsignedIncMsb)
        );

        let digest = ONE_HOT_TRACE_LAYOUT.layout_digest(&shape(5)).unwrap();
        assert_ne!(digest, [0; 32]);
        assert_ne!(
            digest,
            ONE_HOT_TRACE_LAYOUT.layout_digest(&shape(6)).unwrap()
        );
    }

    #[test]
    fn every_column_uses_the_same_point_permutation() {
        let leaf = (0..5).map(Fr::from_u64).collect::<Vec<_>>();
        let expected = [2, 3, 4, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        for polynomial in [
            JoltCommittedPolynomial::InstructionRa(0),
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::RamRa(0),
            JoltCommittedPolynomial::UnsignedIncChunk(0),
            JoltCommittedPolynomial::UnsignedIncMsb,
        ] {
            assert_eq!(
                ONE_HOT_TRACE_LAYOUT
                    .column_point(polynomial, 2, &leaf)
                    .unwrap(),
                expected
            );
        }
    }
}
