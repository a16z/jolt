//! Shared ABI and reference implementation for product uni-skip.

use std::mem::{align_of, size_of};

use jolt_field::signed::{S128, S192, S256};
#[cfg(any(test, feature = "test-utils"))]
use jolt_field::Field;
use jolt_field::Zero as _;
use jolt_field::{Accumulator as _, Prime128OffsetA7F7 as AkitaField, WithAccumulator};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;

pub use super::product_remainder::ProductRemainderRow;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const PRODUCT_UNISKIP_EXTENDED_NODES: usize = 2;
pub const PRODUCT_UNISKIP_SIMD_WIDTH: usize = 32;
pub const PRODUCT_UNISKIP_NODE_ORDER: [i64; PRODUCT_UNISKIP_EXTENDED_NODES] = [-2, 2];
pub const PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS: [[i64; 3]; 2] = [[3, -3, 1], [1, -3, 3]];

pub(crate) const BLOCKS_PIPELINE: &str = "solinas_product_uniskip_extended_blocks2";
pub(crate) const STAGE1_BLOCKS_PIPELINE: &str = "solinas_product_uniskip_stage1_extended_blocks2";

const _: [(); 40] = [(); size_of::<ProductRemainderRow>()];
const _: [(); 8] = [(); align_of::<ProductRemainderRow>()];

fn cpu_extended_product(row: ProductRemainderRow, coefficients: &[i64; 3]) -> S256 {
    let left = i128::from(coefficients[0]) * i128::from(row.left_instruction_input())
        + i128::from(coefficients[1]) * i128::from(row.lookup_output())
        + i128::from(coefficients[2]) * i128::from(u8::from(row.jump()));
    let right_wide = S192::from_i128(row.right_instruction_input());
    let mut right = S192::from_i64(coefficients[0]).mul_trunc::<3, 3>(&right_wide);
    right += S192::from_i64(
        coefficients[1] * i64::from(u8::from(row.branch()))
            + coefficients[2] * i64::from(u8::from(!row.next_is_noop())),
    );
    S128::from_i128(left).mul_trunc::<3, 4>(&right)
}

pub fn evaluate_product_uniskip_extensions_cpu(
    rows: &[ProductRemainderRow],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> Result<ProductUniskipExtendedNodes<AkitaField>, ProductUniskipShapeError> {
    let _ = ProductUniskipBlockParams::new(rows.len(), e_in.len(), e_out.len())?;
    let block = |x_out: usize| {
        let mut accumulators: [<AkitaField as WithAccumulator>::SignedProductAccumulator;
            PRODUCT_UNISKIP_EXTENDED_NODES] = Default::default();
        for (x_in, &weight) in e_in.iter().enumerate() {
            let row = rows[x_out * e_in.len() + x_in];
            for (accumulator, coefficients) in accumulators
                .iter_mut()
                .zip(&PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS)
            {
                accumulator.fmadd_s256(weight, &cpu_extended_product(row, coefficients));
            }
        }
        std::array::from_fn(|node| e_out[x_out] * accumulators[node].reduce())
    };
    let merge = |mut left: [AkitaField; PRODUCT_UNISKIP_EXTENDED_NODES], right| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(feature = "parallel")]
    let values = (0..e_out.len()).into_par_iter().map(block).reduce(
        || [AkitaField::zero(); PRODUCT_UNISKIP_EXTENDED_NODES],
        merge,
    );
    #[cfg(not(feature = "parallel"))]
    let values = (0..e_out.len())
        .map(block)
        .fold([AkitaField::zero(); PRODUCT_UNISKIP_EXTENDED_NODES], merge);
    Ok(ProductUniskipExtendedNodes {
        minus_two: values[0],
        plus_two: values[1],
    })
}

/// The two evaluations returned by Metal, ordered as `[-2, 2]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductUniskipExtendedNodes<F> {
    pub minus_two: F,
    pub plus_two: F,
}

impl<F: Copy> ProductUniskipExtendedNodes<F> {
    pub const fn as_array(self) -> [F; PRODUCT_UNISKIP_EXTENDED_NODES] {
        [self.minus_two, self.plus_two]
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ProductUniskipShapeError {
    #[error("product uni-skip needs a nonzero power-of-two row count, got {0}")]
    InvalidRows(usize),
    #[error(
        "product uni-skip {phase} weights have e_in={e_in}, e_out={e_out}; expected product {expected}"
    )]
    WeightShape {
        phase: &'static str,
        expected: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("product uni-skip {name} element count exceeds its 32-bit shader index")]
    ShaderIndexOverflow { name: &'static str },
    #[error("product uni-skip {name} byte length overflows host indexing")]
    ByteLengthOverflow { name: &'static str },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProductUniskipBlockParams {
    pub(crate) rows: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) _reserved: u32,
}

const _: [(); 16] = [(); size_of::<ProductUniskipBlockParams>()];

impl ProductUniskipBlockParams {
    pub(crate) fn new(
        rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, ProductUniskipShapeError> {
        validate_rows(rows)?;
        validate_weight_shape("blocks", rows, e_in_length, e_out_length)?;
        validate_partial_index(PRODUCT_UNISKIP_EXTENDED_NODES, e_out_length)?;
        Ok(Self {
            rows: shader_count("rows", rows)?,
            e_in_length: shader_count("e_in", e_in_length)?,
            e_out_length: shader_count("e_out", e_out_length)?,
            _reserved: 0,
        })
    }
}

fn validate_rows(rows: usize) -> Result<(), ProductUniskipShapeError> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(ProductUniskipShapeError::InvalidRows(rows));
    }
    let _ = shader_count("rows", rows)?;
    Ok(())
}

fn validate_weight_shape(
    phase: &'static str,
    expected: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), ProductUniskipShapeError> {
    let covered = e_in.checked_mul(e_out);
    if e_in == 0 || e_out == 0 || covered != Some(expected) {
        return Err(ProductUniskipShapeError::WeightShape {
            phase,
            expected,
            e_in,
            e_out,
        });
    }
    Ok(())
}

fn validate_partial_index(
    columns: usize,
    fields_per_column: usize,
) -> Result<(), ProductUniskipShapeError> {
    let fields = checked_product("partial buffer", columns, fields_per_column)?;
    let _ = shader_count("partial buffer", fields)?;
    Ok(())
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, ProductUniskipShapeError> {
    u32::try_from(value).map_err(|_| ProductUniskipShapeError::ShaderIndexOverflow { name })
}

fn checked_product(
    name: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, ProductUniskipShapeError> {
    lhs.checked_mul(rhs)
        .ok_or(ProductUniskipShapeError::ByteLengthOverflow { name })
}

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod reference {
    use super::*;

    pub fn extended_node_values<F: Field>(
        rows: &[ProductRemainderRow],
        e_in: &[F],
        e_out: &[F],
    ) -> Result<ProductUniskipExtendedNodes<F>, ProductUniskipShapeError> {
        let _ = ProductUniskipBlockParams::new(rows.len(), e_in.len(), e_out.len())?;
        let coefficients = PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS.map(|row| {
            row.map(|coefficient| {
                if coefficient < 0 {
                    -F::from_u64(coefficient.unsigned_abs())
                } else {
                    F::from_u64(coefficient as u64)
                }
            })
        });
        let mut endpoints = [F::zero(); PRODUCT_UNISKIP_EXTENDED_NODES];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = [F::zero(); PRODUCT_UNISKIP_EXTENDED_NODES];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let row = rows[x_out * e_in.len() + x_in];
                for (sum, weights) in inner.iter_mut().zip(&coefficients) {
                    let (left, right) = row.relation_values(weights);
                    *sum += inner_weight * left * right;
                }
            }
            for (endpoint, inner) in endpoints.iter_mut().zip(inner) {
                *endpoint += outer_weight * inner;
            }
        }
        Ok(ProductUniskipExtendedNodes {
            minus_two: endpoints[0],
            plus_two: endpoints[1],
        })
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use fixed valid shapes")]
mod tests {
    use jolt_field::Prime128OffsetA7F7 as AkitaField;
    use jolt_field::{One as _, Ring as _};

    use super::*;

    fn edge_rows() -> Vec<ProductRemainderRow> {
        vec![
            ProductRemainderRow::new(
                u64::MAX,
                i128::MIN,
                true,
                true,
                u64::MAX - 1,
                true,
                false,
                true,
            ),
            ProductRemainderRow::new(0, -1, false, false, 1, false, true, false),
            ProductRemainderRow::new(17, 0, true, false, 23, false, false, true),
            ProductRemainderRow::new(
                u64::MAX - 2,
                i128::MAX,
                false,
                true,
                u64::MAX,
                true,
                true,
                false,
            ),
        ]
    }

    fn direct_node(
        rows: &[ProductRemainderRow],
        weights: [AkitaField; 3],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> AkitaField {
        let mut result = AkitaField::zero();
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = AkitaField::zero();
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let row = rows[x_out * e_in.len() + x_in];
                let (left, right) = row.relation_values(&weights);
                inner += inner_weight * left * right;
            }
            result += outer_weight * inner;
        }
        result
    }

    #[test]
    fn product_row_abi_is_shared_without_translation() {
        assert_eq!(size_of::<ProductRemainderRow>(), 40);
        assert_eq!(align_of::<ProductRemainderRow>(), 8);
    }

    #[test]
    fn extended_nodes_match_direct_edge_evaluations() {
        let rows = edge_rows();
        let e_in = [AkitaField::from_u64(2), AkitaField::from_u64(3)];
        let e_out = [AkitaField::from_u64(5), AkitaField::from_u64(7)];
        let got = reference::extended_node_values(&rows, &e_in, &e_out)
            .expect("the fixed shape is valid");
        let three = AkitaField::from_u64(3);
        let expected = ProductUniskipExtendedNodes {
            minus_two: direct_node(&rows, [three, -three, AkitaField::one()], &e_in, &e_out),
            plus_two: direct_node(&rows, [AkitaField::one(), -three, three], &e_in, &e_out),
        };
        assert_eq!(got, expected);
    }

    #[test]
    fn block_shape_requires_exact_eq_factorization() {
        assert_eq!(
            ProductUniskipBlockParams::new(0, 0, 0),
            Err(ProductUniskipShapeError::InvalidRows(0))
        );
        assert_eq!(
            ProductUniskipBlockParams::new(6, 2, 3),
            Err(ProductUniskipShapeError::InvalidRows(6))
        );
        assert_eq!(
            ProductUniskipBlockParams::new(8, 2, 2),
            Err(ProductUniskipShapeError::WeightShape {
                phase: "blocks",
                expected: 8,
                e_in: 2,
                e_out: 2,
            })
        );
        assert!(ProductUniskipBlockParams::new(8, 2, 4).is_ok());
    }
}
