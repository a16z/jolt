use jolt_field::{Fr, One, Ring, Zero};

use super::*;
use crate::optimized::parity::synthetic_point;

#[derive(Clone, Copy)]
struct Row {
    pc: usize,
    weight: Fr,
}

#[test]
fn membership_is_independent_of_zero_contributions() {
    let log_t = 4;
    let addresses = 4;
    let mut points: [Vec<Fr>; 5] =
        std::array::from_fn(|stage| synthetic_point(log_t, 101 + stage as u64));
    points[0][2] = Fr::one();

    let pcs = [2, 0, 2, 1, 3, 3, 0, 2, 1, 1, 1, 1, 0, 3, 2, 0];
    let rows: Vec<Row> = pcs
        .into_iter()
        .enumerate()
        .map(|(index, pc)| Row {
            pc,
            weight: if index.is_multiple_of(3) {
                Fr::zero()
            } else {
                Fr::from_u64(index as u64 + 1)
            },
        })
        .collect();

    let expected = |weighted: bool| {
        points
            .iter()
            .map(|point| {
                let eq = eq_table(point);
                let mut values = vec![Fr::zero(); addresses];
                for (index, row) in rows.iter().enumerate() {
                    values[row.pc] += eq[index] * if weighted { row.weight } else { Fr::one() };
                }
                values
            })
            .collect::<Vec<_>>()
    };

    let pushforwards = |base: &[Vec<Fr>], weighted: &[Vec<Fr>]| {
        stage_pushforwards(
            base,
            weighted,
            |index| Ok(rows[index]),
            rows.len() - 1,
            addresses,
            |row| row.pc,
            |row| row.weight,
        )
        .unwrap()
    };
    let unweighted = pushforwards(&points, &[]);
    let weighted = pushforwards(&[], &points);
    let combined = pushforwards(&points, &points);
    assert_eq!(unweighted, expected(false));
    assert_eq!(weighted, expected(true));
    assert_eq!(
        combined,
        unweighted
            .iter()
            .chain(&weighted)
            .cloned()
            .collect::<Vec<_>>()
    );
}

fn tiled_weighted_pushforwards<F: JoltField>(log_t: usize, base_stages: usize, addresses: usize) {
    let weighted_stages = if base_stages == 7 { 4 } else { 1 };
    let points = (0..base_stages + weighted_stages)
        .map(|stage| {
            (0..log_t)
                .map(|bit| {
                    if bit == 0 {
                        F::one()
                    } else {
                        F::from_u64((stage * 31 + bit * 7 + 3) as u64)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rows = (0..1usize << log_t)
        .map(|index| {
            let address = if index % 5 == 0 {
                1023
            } else {
                (index * 19 + index / 7) % addresses
            };
            let weight = F::from_u64((index % 11) as u64) - F::from_u64(5);
            (address, weight)
        })
        .collect::<Vec<_>>();
    for prefix in [0, 1, 9, rows.len() / 2 + 1, rows.len()] {
        let mut dense = rows.clone();
        dense[prefix..].fill((addresses - 1, F::from_u64(7)));
        // A read past the prefix and its tail row is out of bounds.
        let served = &dense[..(prefix + 1).min(dense.len())];
        let actual = stage_pushforwards(
            &points[..base_stages],
            &points[base_stages..],
            |index| Ok(served[index]),
            served.len() - 1,
            addresses,
            |row| row.0,
            |row| row.1,
        )
        .unwrap();
        for (stage, point) in points.iter().enumerate() {
            let eq = eq_table(point);
            let mut expected = vec![F::zero(); addresses];
            for ((address, weight), value) in dense.iter().zip(eq) {
                expected[*address] += if stage < base_stages {
                    value
                } else {
                    value * *weight
                };
            }
            assert_eq!(actual[stage], expected);
        }
    }
}

#[test]
fn weighted_tiles_and_fragments_match_the_cycle_eq_mle() {
    tiled_weighted_pushforwards::<Fr>(15, 7, 1 << 17);
}

#[test]
fn bounded_direct_domain_matches_the_cycle_eq_mle() {
    tiled_weighted_pushforwards::<Fr>(19, 1, 1 << 16);
}

#[cfg(feature = "akita")]
#[test]
fn fp128_weighted_tiles_and_batches_match_the_cycle_eq_mle() {
    use jolt_field::Prime128Offset6A23801F;
    tiled_weighted_pushforwards::<Prime128Offset6A23801F>(19, 7, 1 << 17);
}
