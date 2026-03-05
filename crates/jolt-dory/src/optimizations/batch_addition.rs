use ark_bn254::G1Affine;
use rayon::prelude::*;

/// Performs batch addition of G1 affine points using Montgomery's inversion trick.
pub fn batch_g1_additions(bases: &[G1Affine], indices: &[usize]) -> G1Affine {
    if indices.is_empty() {
        return G1Affine::identity();
    }

    if indices.len() == 1 {
        return bases[indices[0]];
    }

    let mut points: Vec<G1Affine> = Vec::with_capacity(indices.len());
    points.extend(indices.iter().map(|&i| bases[i]));

    while points.len() > 1 {
        let current_len = points.len();
        let pairs_count = current_len / 2;
        let has_odd = current_len % 2 == 1;

        let denominators: Vec<_> = (0..pairs_count)
            .into_par_iter()
            .map(|i| {
                let p1 = points[i * 2];
                let p2 = points[i * 2 + 1];
                p2.x - p1.x
            })
            .collect();

        let mut inverses = denominators;
        ark_ff::fields::batch_inversion(&mut inverses);

        let mut new_points: Vec<G1Affine> = (0..pairs_count)
            .into_par_iter()
            .zip(inverses.par_iter())
            .map(|(i, inv)| {
                let p1 = points[i * 2];
                let p2 = points[i * 2 + 1];
                let lambda = (p2.y - p1.y) * inv;
                let x3 = lambda * lambda - p1.x - p2.x;
                let y3 = lambda * (p1.x - x3) - p1.y;
                G1Affine::new_unchecked(x3, y3)
            })
            .collect();

        if has_odd {
            new_points.push(points[current_len - 1]);
        }

        points = new_points;
    }

    points[0]
}

/// Performs multiple batch additions of G1 affine points in parallel,
/// sharing a single batch inversion across all sets per round.
pub fn batch_g1_additions_multi(bases: &[G1Affine], indices_sets: &[Vec<usize>]) -> Vec<G1Affine> {
    if indices_sets.is_empty() {
        return vec![];
    }

    let mut working_sets: Vec<Vec<G1Affine>> = indices_sets
        .par_iter()
        .map(|indices| {
            if indices.is_empty() {
                vec![G1Affine::identity()]
            } else if indices.len() == 1 {
                vec![bases[indices[0]]]
            } else {
                indices.iter().map(|&i| bases[i]).collect()
            }
        })
        .collect();

    loop {
        let total_pairs: usize = working_sets.iter().map(|set| set.len() / 2).sum();

        if total_pairs == 0 {
            break;
        }

        let mut all_denominators = Vec::with_capacity(total_pairs);
        let mut pair_info = Vec::with_capacity(total_pairs);

        for (set_idx, set) in working_sets.iter().enumerate() {
            let pairs_in_set = set.len() / 2;
            for pair_idx in 0..pairs_in_set {
                let p1 = set[pair_idx * 2];
                let p2 = set[pair_idx * 2 + 1];
                all_denominators.push(p2.x - p1.x);
                pair_info.push((set_idx, pair_idx));
            }
        }

        let mut inverses = all_denominators;
        ark_ff::fields::batch_inversion(&mut inverses);

        let mut new_working_sets: Vec<Vec<G1Affine>> = working_sets
            .iter()
            .map(|set| Vec::with_capacity(set.len().div_ceil(2)))
            .collect();

        for ((set_idx, pair_idx), inv) in pair_info.iter().zip(inverses.iter()) {
            let set = &working_sets[*set_idx];
            let p1 = set[*pair_idx * 2];
            let p2 = set[*pair_idx * 2 + 1];
            let lambda = (p2.y - p1.y) * inv;
            let x3 = lambda * lambda - p1.x - p2.x;
            let y3 = lambda * (p1.x - x3) - p1.y;
            new_working_sets[*set_idx].push(G1Affine::new_unchecked(x3, y3));
        }

        for (set_idx, set) in working_sets.iter().enumerate() {
            if set.len() % 2 == 1 {
                new_working_sets[set_idx].push(set[set.len() - 1]);
            }
        }

        working_sets = new_working_sets;
    }

    working_sets.into_iter().map(|set| set[0]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_std::rand::RngCore;
    use ark_std::UniformRand;

    #[test]
    fn test_batch_addition_correctness() {
        let mut rng = ark_std::test_rng();
        let bases: Vec<G1Affine> = (0..10).map(|_| G1Affine::rand(&mut rng)).collect();
        let indices = vec![2, 3, 4, 5, 6, 7];

        let batch_result = batch_g1_additions(&bases, &indices);

        let mut expected = G1Affine::identity();
        for &idx in &indices {
            expected = (expected + bases[idx]).into();
        }
        assert_eq!(batch_result, expected);
    }

    #[test]
    fn test_empty_indices() {
        let bases: Vec<G1Affine> = vec![G1Affine::generator(); 5];
        let result = batch_g1_additions(&bases, &[]);
        assert_eq!(result, G1Affine::identity());
    }

    #[test]
    fn test_single_index() {
        let mut rng = ark_std::test_rng();
        let bases: Vec<G1Affine> = (0..5).map(|_| G1Affine::rand(&mut rng)).collect();
        let result = batch_g1_additions(&bases, &[2]);
        assert_eq!(result, bases[2]);
    }

    #[test]
    fn test_batch_additions_multi() {
        let mut rng = ark_std::test_rng();
        let base_size = 10000;
        let num_batches = 50;

        let bases: Vec<G1Affine> = (0..base_size).map(|_| G1Affine::rand(&mut rng)).collect();
        let indices_sets: Vec<Vec<usize>> = (0..num_batches)
            .map(|_| {
                let size = (rng.next_u64() as usize) % 100 + 1;
                (0..size)
                    .map(|_| (rng.next_u64() as usize) % base_size)
                    .collect()
            })
            .collect();

        let batch_results = batch_g1_additions_multi(&bases, &indices_sets);

        for (i, (result, indices)) in batch_results.iter().zip(indices_sets.iter()).enumerate() {
            let single_result = batch_g1_additions(&bases, indices);
            assert_eq!(
                *result, single_result,
                "Multi vs single mismatch at batch {}",
                i
            );
        }
    }
}
