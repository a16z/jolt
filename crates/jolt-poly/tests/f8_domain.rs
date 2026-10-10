#![cfg(feature = "binary")]

use jolt_field::{Field, F128, F192, F64, F8};
use jolt_poly::lagrange::{f8_domain_nodes, F8DomainSizeError, F8_DOMAIN_MAX_SIZE};

fn nodes_follow_raw_order<F: Field + From<F8>>() -> Result<(), F8DomainSizeError> {
    assert_eq!(F8_DOMAIN_MAX_SIZE, 256);
    for size in [8, 256] {
        let nodes = f8_domain_nodes::<F>(size)?;
        assert_eq!(nodes.len(), size);
        for (index, &node) in nodes.iter().enumerate() {
            assert_eq!(node, F::from(F8::from_raw(index as u8)));
        }
    }
    for size in [1, 2, 10, 64, 256] {
        let nodes = f8_domain_nodes::<F>(size)?;
        assert_eq!(nodes.len(), size);
        for (index, node) in nodes.iter().enumerate() {
            assert!(!nodes.iter().skip(index + 1).any(|other| node == other));
        }
    }
    let prefix = f8_domain_nodes::<F>(64)?;
    let extension = f8_domain_nodes::<F>(128)?;
    assert!(prefix.iter().eq(extension.iter().take(64)));
    for size in [0, 257, usize::MAX] {
        assert_eq!(f8_domain_nodes::<F>(size), Err(F8DomainSizeError { size }));
    }
    Ok(())
}

fn next_block_is_ordered_coset<F: Field + From<F8>>() -> Result<(), F8DomainSizeError> {
    for exponent in [2, 7] {
        let size = 1 << exponent;
        let nodes = f8_domain_nodes::<F>(2 * size)?;
        let offset = F::from(F8::from_raw(size as u8));
        for (&point, &coset_point) in nodes.iter().take(size).zip(nodes.iter().skip(size)) {
            assert_eq!(coset_point, point + offset);
        }
    }
    Ok(())
}

macro_rules! field_tests {
    ($module:ident, $field:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn raw_order_distinctness_prefix_and_size_errors() -> Result<(), F8DomainSizeError> {
                nodes_follow_raw_order::<$field>()
            }

            #[test]
            fn coset_order() -> Result<(), F8DomainSizeError> {
                next_block_is_ordered_coset::<$field>()
            }
        }
    };
}

field_tests!(f64, F64);
field_tests!(f128, F128);
field_tests!(f192, F192);

#[test]
fn point_two_is_the_specified_embedding_root() -> Result<(), F8DomainSizeError> {
    assert_eq!(
        f8_domain_nodes::<F64>(3)?.get(2),
        Some(&F64::from_raw(0x033c_e8be_ddc8_a656))
    );
    assert_eq!(
        f8_domain_nodes::<F128>(3)?.get(2),
        Some(&F128::from_raw(0x053d_8555_a997_9a1c_a13f_e8ac_5560_ce0d))
    );
    Ok(())
}
