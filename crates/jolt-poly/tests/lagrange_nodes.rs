use jolt_field::{Field, Fr, Ring};
use jolt_poly::lagrange::{
    interpolate_nodes_to_coeffs, interpolate_to_coeffs, lagrange_evals_at_nodes, LagrangeNodesError,
};

fn horner<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |value, &coefficient| value * point + coefficient)
}

fn basis_and_interpolation<F: Field>(
    nodes: &[F],
    point: F,
    coefficients: &[F],
) -> Result<(), LagrangeNodesError> {
    assert!(!nodes.contains(&point));
    for (node_index, &node) in nodes.iter().enumerate() {
        let basis = lagrange_evals_at_nodes(nodes, node)?;
        assert_eq!(basis.len(), nodes.len());
        for (basis_index, &value) in basis.iter().enumerate() {
            assert_eq!(
                value,
                if basis_index == node_index {
                    F::one()
                } else {
                    F::zero()
                }
            );
        }
    }
    let values: Vec<_> = nodes
        .iter()
        .map(|&node| horner(coefficients, node))
        .collect();
    let reproduced: F = lagrange_evals_at_nodes(nodes, point)?
        .iter()
        .zip(&values)
        .map(|(&basis, &value)| basis * value)
        .sum();
    assert_eq!(reproduced, horner(coefficients, point));
    assert_eq!(interpolate_nodes_to_coeffs(nodes, &values)?, coefficients);
    Ok(())
}

fn low_degree_interpolation<F: Field>(
    nodes: &[F],
    coefficients: &[F],
) -> Result<(), LagrangeNodesError> {
    let values: Vec<_> = nodes
        .iter()
        .map(|&node| horner(coefficients, node))
        .collect();
    assert_eq!(interpolate_nodes_to_coeffs(nodes, &values)?, coefficients);
    Ok(())
}

fn node_errors<F: Field>(a: F, b: F, non_node: F) {
    assert_eq!(
        lagrange_evals_at_nodes::<F>(&[], non_node),
        Err(LagrangeNodesError::EmptyNodes)
    );
    assert_eq!(
        interpolate_nodes_to_coeffs::<F>(&[], &[a]),
        Err(LagrangeNodesError::EmptyNodes)
    );
    let nodes = [a, b, b, a, a];
    let repeated = LagrangeNodesError::RepeatedNode {
        first: 0,
        second: 3,
    };
    for point in [non_node, a] {
        assert_eq!(lagrange_evals_at_nodes(&nodes, point), Err(repeated));
    }
    for values in [&[a; 5][..], &[a][..]] {
        assert_eq!(interpolate_nodes_to_coeffs(&nodes, values), Err(repeated));
    }
    assert_eq!(
        interpolate_nodes_to_coeffs(&[a, b], &[a]),
        Err(LagrangeNodesError::LengthMismatch {
            nodes: 2,
            values: 1
        })
    );
}

#[test]
fn prime_field_unsorted_nodes_and_singleton() -> Result<(), LagrangeNodesError> {
    let nodes = [7, 2, 11, 4, 9].map(Fr::from_u64);
    let coefficients = [3, 5, 7, 11, 13].map(Fr::from_u64);
    basis_and_interpolation(&nodes, Fr::from_u64(15), &coefficients)?;
    low_degree_interpolation(&nodes, &[3, 5, 7, 0, 0].map(Fr::from_u64))?;
    basis_and_interpolation(&[Fr::from_u64(7)], Fr::from_u64(15), &[Fr::from_u64(3)])
}

#[test]
fn prime_field_node_validation_order() {
    node_errors(Fr::from_u64(7), Fr::from_u64(2), Fr::from_u64(15));
}

#[test]
fn integer_node_interpolators_recover_fixed_coefficients() -> Result<(), LagrangeNodesError> {
    let coefficients = [3, 5, 7, 11, 13].map(Fr::from_u64);
    for start in [-2, 3] {
        let nodes: Vec<_> = (0..5).map(|offset| Fr::from_i64(start + offset)).collect();
        let values: Vec<_> = nodes
            .iter()
            .map(|&node| horner(&coefficients, node))
            .collect();
        assert_eq!(interpolate_nodes_to_coeffs(&nodes, &values)?, coefficients);
        assert_eq!(interpolate_to_coeffs(start, &values), coefficients);
    }
    Ok(())
}

#[cfg(feature = "binary")]
mod binary {
    use super::*;
    use jolt_field::F64;

    #[test]
    fn binary_field_unsorted_nodes_and_singleton() -> Result<(), LagrangeNodesError> {
        let nodes = [7, 2, 11, 4, 9].map(F64::from_raw);
        let coefficients = [3, 5, 7, 11, 13].map(F64::from_raw);
        basis_and_interpolation(&nodes, F64::from_raw(15), &coefficients)?;
        low_degree_interpolation(&nodes, &[3, 5, 7, 0, 0].map(F64::from_raw))?;
        basis_and_interpolation(&[F64::from_raw(7)], F64::from_raw(15), &[F64::from_raw(3)])
    }

    #[test]
    fn binary_field_node_validation_order() {
        node_errors(F64::from_raw(7), F64::from_raw(2), F64::from_raw(15));
    }
}
