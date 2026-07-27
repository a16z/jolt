//! The canonical native-Akita `OneHotTrace` commitment layout. Its semantic
//! columns are prefix-packed into one strict `K x (capacity · T)` one-hot
//! polynomial and reduced to one opening at a random selector point.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::OpeningsError;
use jolt_poly::eq_index_msb;
use std::ops::Range;

use super::super::JoltCommittedPolynomial;
use super::packing::{one_hot_trace_column_capacity, one_hot_trace_columns, OneHotTraceShape};

/// `OneHotTrace` is committed as one prefix-packed Akita one-hot polynomial.
pub const ONE_HOT_TRACE_LAYOUT: OneHotTraceLayout = OneHotTraceLayout;

/// The one protocol layout for the per-proof `OneHotTrace` commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotTraceLayout;

/// Semantic column ranges in the packed selector domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotTraceColumnRanges {
    pub instruction: Range<usize>,
    pub bytecode: Range<usize>,
    pub ram: Range<usize>,
    pub unsigned_inc: Range<usize>,
    pub unsigned_inc_msb: usize,
}

/// The canonical column order and packed geometry for one proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotTraceLayoutPlan {
    pub columns: Vec<JoltCommittedPolynomial>,
    pub column_arity: usize,
    pub packed_arity: usize,
    pub column_capacity: usize,
    pub selector_bits: usize,
    pub ranges: OneHotTraceColumnRanges,
}

/// The commitment-object setup shape the layout requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotTraceSetupShape {
    pub num_vars: usize,
    pub num_polys: usize,
}

impl OneHotTraceLayout {
    /// The canonical object layout for `shape`.
    pub fn plan(&self, shape: &OneHotTraceShape) -> Result<OneHotTraceLayoutPlan, OpeningsError> {
        let columns = one_hot_trace_columns(shape)
            .map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        let column_capacity = one_hot_trace_column_capacity(shape.log_k_chunk)
            .map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        if columns.len() > column_capacity {
            return Err(OpeningsError::InvalidBatch(
                super::geometry::LatticeGeometryError::TooManyOneHotTraceColumns {
                    chunk_width: shape.log_k_chunk,
                    actual: columns.len(),
                    capacity: column_capacity,
                }
                .to_string(),
            ));
        }
        let instruction_end = shape.ra_layout.instruction();
        let unsigned_inc_end =
            instruction_end + super::geometry::UNSIGNED_INC_BITS / shape.log_k_chunk;
        let unsigned_inc_msb = unsigned_inc_end;
        let bytecode_start = unsigned_inc_msb + 1;
        let bytecode_end = bytecode_start + shape.ra_layout.bytecode();
        let ram_end = bytecode_end + shape.ra_layout.ram();
        debug_assert_eq!(ram_end, columns.len());
        let selector_bits = column_capacity.ilog2() as usize;
        let column_arity = shape.log_k_chunk + shape.log_t;
        Ok(OneHotTraceLayoutPlan {
            columns,
            column_arity,
            packed_arity: column_arity + selector_bits,
            column_capacity,
            selector_bits,
            ranges: OneHotTraceColumnRanges {
                instruction: 0..instruction_end,
                bytecode: bytecode_start..bytecode_end,
                ram: bytecode_end..ram_end,
                unsigned_inc: instruction_end..unsigned_inc_end,
                unsigned_inc_msb,
            },
        })
    }

    /// The commitment-object setup shape.
    pub fn setup_shape(
        &self,
        shape: &OneHotTraceShape,
    ) -> Result<OneHotTraceSetupShape, OpeningsError> {
        let plan = self.plan(shape)?;
        Ok(OneHotTraceSetupShape {
            num_vars: plan.packed_arity,
            num_polys: 1,
        })
    }

    /// A protocol-owned digest of the exact packed layout. This binds
    /// the commitment and verifier setup to the ordered column identities,
    /// dimensions, and layout version; it is never supplied by the proof.
    pub fn layout_digest(&self, shape: &OneHotTraceShape) -> Result<[u8; 32], OpeningsError> {
        let OneHotTraceLayoutPlan {
            columns,
            column_arity,
            packed_arity,
            column_capacity,
            selector_bits,
            ranges: _,
        } = self.plan(shape)?;
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(b"jolt/akita/one_hot_trace/prefix-packed/v4");
        append_usize(&mut hasher, column_arity);
        append_usize(&mut hasher, packed_arity);
        append_usize(&mut hasher, column_capacity);
        append_usize(&mut hasher, selector_bits);
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
                        "non-OneHotTrace polynomial {other:?} in packed one-hot layout"
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

impl OneHotTraceLayoutPlan {
    /// Forms the physical packed point `(selector || cycle || address)`.
    pub fn packed_point<F: Field>(
        &self,
        selector_point: &[F],
        column_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        if selector_point.len() != self.selector_bits {
            return Err(OpeningsError::InvalidBatch(format!(
                "OneHotTrace selector point has {} variables, expected {}",
                selector_point.len(),
                self.selector_bits
            )));
        }
        if column_point.len() != self.column_arity {
            return Err(OpeningsError::InvalidBatch(format!(
                "OneHotTrace column point has {} variables, expected {}",
                column_point.len(),
                self.column_arity
            )));
        }
        let mut point = Vec::with_capacity(self.packed_arity);
        point.extend_from_slice(selector_point);
        point.extend_from_slice(column_point);
        Ok(point)
    }

    /// Evaluates the selector MLE of the semantic claims, with unused packed
    /// slots fixed to zero.
    pub fn packed_evaluation<F: Field>(
        &self,
        selector_point: &[F],
        evaluations: &[F],
    ) -> Result<F, OpeningsError> {
        if selector_point.len() != self.selector_bits {
            return Err(OpeningsError::InvalidBatch(format!(
                "OneHotTrace selector point has {} variables, expected {}",
                selector_point.len(),
                self.selector_bits
            )));
        }
        if evaluations.len() != self.columns.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "OneHotTrace has {} evaluations, expected {}",
                evaluations.len(),
                self.columns.len()
            )));
        }
        Ok(evaluations
            .iter()
            .enumerate()
            .fold(F::zero(), |sum, (column, evaluation)| {
                sum + *evaluation * eq_index_msb(selector_point, column as u128)
            }))
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
    use jolt_poly::{MultilinearPoly, OneHotPolynomial};

    fn shape(log_t: usize) -> OneHotTraceShape {
        OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 1, 1).unwrap(),
            log_t,
            log_k_chunk: 8,
        }
    }

    #[test]
    fn packed_layout_has_fixed_selector_capacity_and_is_digest_bound() {
        let OneHotTraceLayoutPlan {
            columns,
            column_arity,
            packed_arity,
            column_capacity,
            selector_bits,
            ranges,
        } = ONE_HOT_TRACE_LAYOUT.plan(&shape(5)).unwrap();
        assert_eq!(column_arity, 13);
        assert_eq!(packed_arity, 18);
        assert_eq!(column_capacity, 32);
        assert_eq!(selector_bits, 5);
        assert_eq!(ranges.instruction, 0..16);
        assert_eq!(ranges.unsigned_inc, 16..24);
        assert_eq!(ranges.unsigned_inc_msb, 24);
        assert_eq!(ranges.bytecode, 25..26);
        assert_eq!(ranges.ram, 26..27);
        assert_eq!(columns.last(), Some(&JoltCommittedPolynomial::RamRa(0)));
        assert_eq!(
            ONE_HOT_TRACE_LAYOUT.setup_shape(&shape(5)).unwrap(),
            OneHotTraceSetupShape {
                num_vars: 18,
                num_polys: 1
            }
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

    #[test]
    fn selector_reduction_includes_zero_padding_slots() {
        let plan = ONE_HOT_TRACE_LAYOUT.plan(&shape(5)).unwrap();
        let evaluations = (1..=plan.columns.len())
            .map(|value| Fr::from_u64(value as u64))
            .collect::<Vec<_>>();
        assert_eq!(
            plan.packed_evaluation(&vec![Fr::from_u64(0); plan.selector_bits], &evaluations)
                .unwrap(),
            evaluations[0]
        );
        assert_eq!(
            plan.packed_evaluation(&vec![Fr::from_u64(1); plan.selector_bits], &evaluations)
                .unwrap(),
            Fr::from_u64(0)
        );
    }

    #[test]
    fn selector_reduction_matches_the_physical_packed_polynomial() {
        let plan = ONE_HOT_TRACE_LAYOUT.plan(&shape(5)).unwrap();
        let k = 1usize << shape(5).log_k_chunk;
        let rows = 1usize << shape(5).log_t;
        let semantic = (0..plan.columns.len())
            .map(|column| {
                OneHotPolynomial::new(
                    k,
                    (0..rows)
                        .map(|row| Some(((row * (column + 1) + column) % k) as u8))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let packed = OneHotPolynomial::new(
            k,
            semantic
                .iter()
                .flat_map(|polynomial| polynomial.indices().iter().copied())
                .chain(std::iter::repeat_n(
                    None,
                    (plan.column_capacity - semantic.len()) * rows,
                ))
                .collect(),
        );
        let selector = (2..2 + plan.selector_bits as u64)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        let column_point = (11..11 + plan.column_arity as u64)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        let evaluations = semantic
            .iter()
            .map(|polynomial| polynomial.evaluate(&column_point))
            .collect::<Vec<_>>();
        assert_eq!(
            packed.evaluate(&plan.packed_point(&selector, &column_point).unwrap()),
            plan.packed_evaluation(&selector, &evaluations).unwrap()
        );
    }

    #[test]
    fn layout_rejects_semantic_columns_beyond_the_fixed_capacity() {
        let oversized = OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 4, 4).unwrap(),
            ..shape(5)
        };
        assert!(matches!(
            ONE_HOT_TRACE_LAYOUT.plan(&oversized),
            Err(OpeningsError::InvalidBatch(message))
                if message.contains("exceeding the K=2^8 packed capacity 32")
        ));
    }

    #[test]
    fn ranges_keep_fixed_and_variable_families_separate() {
        let variable_ra_shape = OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 2, 3).unwrap(),
            ..shape(5)
        };
        let plan = ONE_HOT_TRACE_LAYOUT.plan(&variable_ra_shape).unwrap();
        assert_eq!(plan.ranges.instruction, 0..16);
        assert_eq!(plan.ranges.unsigned_inc, 16..24);
        assert_eq!(plan.ranges.unsigned_inc_msb, 24);
        assert_eq!(plan.ranges.bytecode, 25..27);
        assert_eq!(plan.ranges.ram, 27..30);
        assert_eq!(plan.column_capacity, 32);
    }

    #[test]
    fn layout_rejects_noncanonical_instruction_width() {
        let invalid = OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(15, 1, 1).unwrap(),
            ..shape(5)
        };
        assert!(matches!(
            ONE_HOT_TRACE_LAYOUT.plan(&invalid),
            Err(OpeningsError::InvalidBatch(message))
                if message.contains("requires 16 instruction columns")
        ));
    }
}
