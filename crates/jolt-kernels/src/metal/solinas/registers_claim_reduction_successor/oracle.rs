use jolt_field::Field;

pub fn eq_evals<F: Field>(point: &[F]) -> Vec<F> {
    let mut evaluations = vec![F::one()];
    for &coordinate in point {
        let mut next = Vec::with_capacity(2 * evaluations.len());
        for value in evaluations {
            next.push(value * (F::one() - coordinate));
            next.push(value * coordinate);
        }
        evaluations = next;
    }
    evaluations
}

pub fn dot<F: Field>(left: &[F], right: &[F]) -> F {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(F::zero(), |sum, (&x, &y)| sum + x * y)
}

pub fn bind_low<F: Field>(table: &[F], challenges: &[F]) -> Vec<F> {
    let mut table = table.to_vec();
    for &challenge in challenges {
        assert!(table.len() >= 2 && table.len().is_multiple_of(2));
        let half = table.len() / 2;
        for index in 0..half {
            let low = table[2 * index];
            table[index] = low + challenge * (table[2 * index + 1] - low);
        }
        table.truncate(half);
    }
    table
}

pub fn q_components<F: Field>(rows: &[[u64; 3]], tau: &[F]) -> [Vec<F>; 3] {
    assert_eq!(rows.len(), 1usize << tau.len());
    let suffix_vars = tau.len() / 2;
    let prefix_vars = tau.len() - suffix_vars;
    let prefix_elements = 1usize << prefix_vars;
    let suffix_elements = 1usize << suffix_vars;
    let e_out = eq_evals(&tau[..suffix_vars]);
    let mut components = core::array::from_fn(|_| vec![F::zero(); prefix_elements]);
    for x_hi in 0..suffix_elements {
        for x_lo in 0..prefix_elements {
            let row = rows[x_hi * prefix_elements + x_lo];
            for column in 0..3 {
                components[column][x_lo] += e_out[x_hi] * F::from_u64(row[column]);
            }
        }
    }
    components
}

pub fn midpoint_columns<F: Field>(rows: &[[u64; 3]], prefix_challenges: &[F]) -> [Vec<F>; 3] {
    let prefix_elements = 1usize << prefix_challenges.len();
    assert!(rows.len().is_multiple_of(prefix_elements));
    let weights = eq_evals(&prefix_challenges.iter().rev().copied().collect::<Vec<_>>());
    let suffix_elements = rows.len() / prefix_elements;
    let mut columns = core::array::from_fn(|_| vec![F::zero(); suffix_elements]);
    for x_hi in 0..suffix_elements {
        let chunk = &rows[x_hi * prefix_elements..(x_hi + 1) * prefix_elements];
        for column in 0..3 {
            columns[column][x_hi] = chunk
                .iter()
                .zip(&weights)
                .fold(F::zero(), |sum, (row, &weight)| {
                    sum + weight * F::from_u64(row[column])
                });
        }
    }
    columns
}

#[cfg(test)]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn rows(log_t: usize) -> Vec<[u64; 3]> {
        (0..1u64 << log_t)
            .map(|row| {
                [
                    17 * row + 3,
                    row.wrapping_mul(row).wrapping_add(9),
                    5 * row + 11,
                ]
            })
            .collect()
    }

    fn check_geometry(log_t: usize) {
        let rows = rows(log_t);
        let tau = (0..log_t)
            .map(|index| field(3 + 2 * index as u64))
            .collect::<Vec<_>>();
        let components = q_components(&rows, &tau);
        let e_in = eq_evals(&tau[log_t / 2..]);
        let dense_weights = eq_evals(&tau);
        for column in 0..3 {
            let dense = rows
                .iter()
                .map(|row| field(row[column]))
                .collect::<Vec<_>>();
            assert_eq!(dot(&e_in, &components[column]), dot(&dense_weights, &dense));
        }

        let gamma = field(19);
        let combined = (0..e_in.len())
            .map(|index| {
                components[0][index]
                    + gamma * components[1][index]
                    + gamma * gamma * components[2][index]
            })
            .collect::<Vec<_>>();
        let dense_combined = rows
            .iter()
            .map(|row| field(row[0]) + gamma * field(row[1]) + gamma * gamma * field(row[2]))
            .collect::<Vec<_>>();
        assert_eq!(dot(&e_in, &combined), dot(&dense_weights, &dense_combined));

        let prefix_vars = log_t - log_t / 2;
        let prefix = (0..prefix_vars)
            .map(|index| field(17 + 6 * index as u64))
            .collect::<Vec<_>>();
        let columns = midpoint_columns(&rows, &prefix);
        for column in 0..3 {
            let table = rows
                .iter()
                .map(|row| field(row[column]))
                .collect::<Vec<_>>();
            assert_eq!(columns[column], bind_low(&table, &prefix));
        }

        let suffix = (0..log_t / 2)
            .map(|index| field(31 + 6 * index as u64))
            .collect::<Vec<_>>();
        let suffix_weights = eq_evals(&suffix.iter().rev().copied().collect::<Vec<_>>());
        let full_point = suffix
            .iter()
            .rev()
            .chain(prefix.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        let full_weights = eq_evals(&full_point);
        for column in 0..3 {
            let dense = rows
                .iter()
                .map(|row| field(row[column]))
                .collect::<Vec<_>>();
            assert_eq!(
                dot(&suffix_weights, &columns[column]),
                dot(&full_weights, &dense)
            );
        }
    }

    #[test]
    fn split_oracle_matches_dense_at_odd_geometry() {
        check_geometry(5);
    }

    #[test]
    fn split_oracle_matches_dense_at_even_geometry() {
        check_geometry(4);
    }
}
