//! Dense oracle for the paired stage-2 service.

use jolt_field::Field;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductNativeRow {
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub lookup_output: u64,
    pub jump: bool,
    pub write_lookup_output_to_rd: bool,
    pub branch: bool,
    pub next_is_noop: bool,
    pub virtual_instruction: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LookupCompanionRow {
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductState<F> {
    pub left: Vec<F>,
    pub right: Vec<F>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointState<F> {
    pub product: ProductState<F>,
    pub instruction_combined: Vec<F>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointMessage<F> {
    /// ProductRemainder returns `q(0)` and the quadratic coefficient.
    pub product: [F; 2],
    /// InstructionClaimReduction returns `q(0)` and `q(2)`.
    pub instruction: [F; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointOpenings<F> {
    pub product: [F; 8],
    pub instruction_unique: [F; 2],
}

fn gamma_powers<F: Field>(gamma: F) -> [F; 5] {
    let gamma_squared = gamma * gamma;
    [
        F::one(),
        gamma,
        gamma_squared,
        gamma_squared * gamma,
        gamma_squared * gamma_squared,
    ]
}

fn instruction_combined<F: Field>(
    product: ProductNativeRow,
    lookup: LookupCompanionRow,
    powers: [F; 5],
) -> F {
    F::from_u64(product.lookup_output)
        + powers[1] * F::from_u64(lookup.left_lookup_operand)
        + powers[2] * F::from_u128(lookup.right_lookup_operand)
        + powers[3] * F::from_u64(product.left_instruction_input)
        + powers[4] * F::from_i128(product.right_instruction_input)
}

fn product_factors<F: Field>(row: ProductNativeRow, lagrange: [F; 3]) -> (F, F) {
    let left = lagrange[0] * F::from_u64(row.left_instruction_input)
        + lagrange[1] * F::from_u64(row.lookup_output)
        + lagrange[2] * F::from_bool(row.jump);
    let right = lagrange[0] * F::from_i128(row.right_instruction_input)
        + lagrange[1] * F::from_bool(row.branch)
        + lagrange[2] * F::from_bool(!row.next_is_noop);
    (left, right)
}

pub fn materialize<F: Field>(
    product_rows: &[ProductNativeRow],
    lookup_rows: &[LookupCompanionRow],
    lagrange: [F; 3],
    gamma: F,
) -> JointState<F> {
    assert_eq!(product_rows.len(), lookup_rows.len());
    let powers = gamma_powers(gamma);
    let mut left = Vec::with_capacity(product_rows.len());
    let mut right = Vec::with_capacity(product_rows.len());
    let mut instruction_combined_state = Vec::with_capacity(product_rows.len());
    for (&product, &lookup) in product_rows.iter().zip(lookup_rows) {
        let (left_value, right_value) = product_factors(product, lagrange);
        left.push(left_value);
        right.push(right_value);
        instruction_combined_state.push(instruction_combined(product, lookup, powers));
    }
    JointState {
        product: ProductState { left, right },
        instruction_combined: instruction_combined_state,
    }
}

pub fn message<F: Field>(state: &JointState<F>, e_in: &[F], e_out: &[F]) -> JointMessage<F> {
    assert_eq!(state.product.left.len(), state.product.right.len());
    assert_eq!(state.product.left.len(), state.instruction_combined.len());
    assert_eq!(2 * e_in.len() * e_out.len(), state.product.left.len());

    let two = F::one() + F::one();
    let mut product = [F::zero(); 2];
    let mut instruction = [F::zero(); 2];
    for (x_out, &outer) in e_out.iter().enumerate() {
        let mut product_inner = [F::zero(); 2];
        let mut instruction_inner = [F::zero(); 2];
        for (x_in, &inner) in e_in.iter().enumerate() {
            let low = 2 * (x_out * e_in.len() + x_in);
            let high = low + 1;
            let left_low = state.product.left[low];
            let left_high = state.product.left[high];
            let right_low = state.product.right[low];
            let right_high = state.product.right[high];
            let combined_low = state.instruction_combined[low];
            let combined_high = state.instruction_combined[high];
            product_inner[0] += inner * left_low * right_low;
            product_inner[1] += inner * (left_high - left_low) * (right_high - right_low);
            instruction_inner[0] += inner * combined_low;
            instruction_inner[1] += inner * (two * combined_high - combined_low);
        }
        for column in 0..2 {
            product[column] += outer * product_inner[column];
            instruction[column] += outer * instruction_inner[column];
        }
    }
    JointMessage {
        product,
        instruction,
    }
}

pub fn bind_low_to_high<F: Field>(table: &[F], challenge: F) -> Vec<F> {
    assert_eq!(table.len() % 2, 0);
    table
        .chunks_exact(2)
        .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
        .collect()
}

pub fn bind_joint<F: Field>(state: &JointState<F>, challenge: F) -> JointState<F> {
    JointState {
        product: ProductState {
            left: bind_low_to_high(&state.product.left, challenge),
            right: bind_low_to_high(&state.product.right, challenge),
        },
        instruction_combined: bind_low_to_high(&state.instruction_combined, challenge),
    }
}

pub fn evaluate_low_to_high<F: Field>(mut table: Vec<F>, challenges: &[F]) -> F {
    for &challenge in challenges {
        table = bind_low_to_high(&table, challenge);
    }
    assert_eq!(table.len(), 1);
    table[0]
}

pub fn openings<F: Field>(
    product_rows: &[ProductNativeRow],
    lookup_rows: &[LookupCompanionRow],
    challenges: &[F],
) -> JointOpenings<F> {
    assert_eq!(product_rows.len(), lookup_rows.len());
    assert_eq!(product_rows.len(), 1usize << challenges.len());

    let product_tables = [
        product_rows
            .iter()
            .map(|row| F::from_u64(row.left_instruction_input))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_i128(row.right_instruction_input))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_bool(row.jump))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_bool(row.write_lookup_output_to_rd))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_u64(row.lookup_output))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_bool(row.branch))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_bool(row.next_is_noop))
            .collect(),
        product_rows
            .iter()
            .map(|row| F::from_bool(row.virtual_instruction))
            .collect(),
    ];
    let product = product_tables.map(|table| evaluate_low_to_high(table, challenges));
    let instruction_unique = [
        evaluate_low_to_high(
            lookup_rows
                .iter()
                .map(|row| F::from_u64(row.left_lookup_operand))
                .collect(),
            challenges,
        ),
        evaluate_low_to_high(
            lookup_rows
                .iter()
                .map(|row| F::from_u128(row.right_lookup_operand))
                .collect(),
            challenges,
        ),
    ];
    JointOpenings {
        product,
        instruction_unique,
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;

    fn rows() -> (Vec<ProductNativeRow>, Vec<LookupCompanionRow>) {
        let mut product = Vec::new();
        let mut lookup = Vec::new();
        for index in 0..16u64 {
            product.push(ProductNativeRow {
                left_instruction_input: 3 * index + 1,
                right_instruction_input: if index == 7 {
                    i128::MIN
                } else {
                    i128::from(index) - 9
                },
                lookup_output: 5 * index + 2,
                jump: index & 1 != 0,
                write_lookup_output_to_rd: index & 2 != 0,
                branch: index & 4 != 0,
                next_is_noop: index & 8 != 0,
                virtual_instruction: index % 3 == 0,
            });
            lookup.push(LookupCompanionRow {
                left_lookup_operand: 7 * index + 4,
                right_lookup_operand: if index == 9 {
                    u128::MAX
                } else {
                    u128::from(11 * index + 6)
                },
            });
        }
        (product, lookup)
    }

    #[test]
    fn message_is_the_direct_pair_sum() {
        let (product, lookup) = rows();
        let lagrange = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let gamma = Fr::from_u64(13);
        let state = materialize(&product, &lookup, lagrange, gamma);
        let e_in = [Fr::from_u64(17), Fr::from_u64(19)];
        let e_out = [
            Fr::from_u64(23),
            Fr::from_u64(29),
            Fr::from_u64(31),
            Fr::from_u64(37),
        ];
        let got = message(&state, &e_in, &e_out);

        let mut expected_instruction_zero = Fr::from_u64(0);
        for (x_out, outer) in e_out.into_iter().enumerate() {
            for (x_in, inner) in e_in.into_iter().enumerate() {
                let low = 2 * (x_out * e_in.len() + x_in);
                expected_instruction_zero += outer * inner * state.instruction_combined[low];
            }
        }
        assert_eq!(got.instruction[0], expected_instruction_zero);
    }

    #[test]
    fn joint_bind_matches_three_independent_dense_binds() {
        let (product, lookup) = rows();
        let state = materialize(
            &product,
            &lookup,
            [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)],
            Fr::from_u64(7),
        );
        let challenge = Fr::from_u64(11);
        let bound = bind_joint(&state, challenge);
        assert_eq!(
            bound.product.left,
            bind_low_to_high(&state.product.left, challenge)
        );
        assert_eq!(
            bound.product.right,
            bind_low_to_high(&state.product.right, challenge)
        );
        assert_eq!(
            bound.instruction_combined,
            bind_low_to_high(&state.instruction_combined, challenge)
        );
    }

    #[test]
    fn output_aliases_reconstruct_the_bound_combined_table() {
        let (product, lookup) = rows();
        let gamma = Fr::from_u64(41);
        let powers = gamma_powers(gamma);
        let state = materialize(
            &product,
            &lookup,
            [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)],
            gamma,
        );
        let challenges = [
            Fr::from_u64(43),
            Fr::from_u64(47),
            Fr::from_u64(53),
            Fr::from_u64(59),
        ];
        let output = openings(&product, &lookup, &challenges);
        let reconstructed = output.product[4]
            + powers[1] * output.instruction_unique[0]
            + powers[2] * output.instruction_unique[1]
            + powers[3] * output.product[0]
            + powers[4] * output.product[1];
        assert_eq!(
            reconstructed,
            evaluate_low_to_high(state.instruction_combined, &challenges)
        );
    }

    #[test]
    fn gamma_zero_leaves_unique_openings_independently_checked() {
        let (product, lookup) = rows();
        let output = openings::<Fr>(
            &product,
            &lookup,
            &[
                Fr::from_u64(3),
                Fr::from_u64(5),
                Fr::from_u64(7),
                Fr::from_u64(11),
            ],
        );
        assert_ne!(output.instruction_unique[0], Fr::from_u64(0));
        assert_ne!(output.instruction_unique[1], Fr::from_u64(0));
    }
}
