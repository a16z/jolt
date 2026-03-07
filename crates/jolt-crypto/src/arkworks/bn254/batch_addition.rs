//! Batch affine addition for BN254 G1 using Montgomery's inversion trick.

use ark_bn254::G1Affine;
use ark_ec::CurveGroup;
use rayon::prelude::*;

use super::Bn254G1;

/// Performs batch addition of G1 affine points using Montgomery's inversion trick.
#[cfg(test)]
fn batch_g1_additions_affine(bases: &[G1Affine], indices: &[usize]) -> G1Affine {
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

/// Performs multiple batch additions of G1 points in parallel,
/// sharing a single batch inversion across all sets per round.
///
/// Takes `Bn254G1` bases (converted to affine internally) and index sets.
/// Returns one `Bn254G1` per index set — the sum of the selected points.
pub fn batch_g1_additions_multi(bases: &[Bn254G1], indices_sets: &[Vec<usize>]) -> Vec<Bn254G1> {
    if indices_sets.is_empty() {
        return vec![];
    }

    // SAFETY: Bn254G1 is #[repr(transparent)] over G1Projective — identical layout.
    let projective: &[ark_bn254::G1Projective] = unsafe {
        std::slice::from_raw_parts(
            bases.as_ptr().cast::<ark_bn254::G1Projective>(),
            bases.len(),
        )
    };
    let affines = ark_bn254::G1Projective::normalize_batch(projective);

    batch_g1_additions_multi_affine_inner(&affines, indices_sets)
        .into_iter()
        .map(|a| {
            let proj: ark_bn254::G1Projective = a.into();
            Bn254G1::from(proj)
        })
        .collect()
}

/// Same as [`batch_g1_additions_multi`] but operates directly on affine points.
///
/// Avoids the projective → affine normalization when callers already have affine bases.
/// Returns one `G1Affine` per index set.
pub fn batch_g1_additions_multi_affine(
    bases: &[G1Affine],
    indices_sets: &[Vec<usize>],
) -> Vec<G1Affine> {
    batch_g1_additions_multi_affine_inner(bases, indices_sets)
}

fn batch_g1_additions_multi_affine_inner(
    affines: &[G1Affine],
    indices_sets: &[Vec<usize>],
) -> Vec<G1Affine> {
    if indices_sets.is_empty() {
        return vec![];
    }

    let mut working_sets: Vec<Vec<G1Affine>> = indices_sets
        .par_iter()
        .map(|indices| {
            if indices.is_empty() {
                vec![G1Affine::identity()]
            } else if indices.len() == 1 {
                vec![affines[indices[0]]]
            } else {
                indices.iter().map(|&i| affines[i]).collect()
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

        let batch_result = batch_g1_additions_affine(&bases, &indices);

        let mut expected = G1Affine::identity();
        for &idx in &indices {
            expected = (expected + bases[idx]).into();
        }
        assert_eq!(batch_result, expected);
    }

    #[test]
    fn test_empty_indices() {
        let bases: Vec<G1Affine> = vec![G1Affine::generator(); 5];
        let result = batch_g1_additions_affine(&bases, &[]);
        assert_eq!(result, G1Affine::identity());
    }

    #[test]
    fn test_single_index() {
        let mut rng = ark_std::test_rng();
        let bases: Vec<G1Affine> = (0..5).map(|_| G1Affine::rand(&mut rng)).collect();
        let result = batch_g1_additions_affine(&bases, &[2]);
        assert_eq!(result, bases[2]);
    }

    #[test]
    fn test_batch_additions_multi() {
        let mut rng = ark_std::test_rng();
        let base_size = 1000;
        let num_batches = 10;

        let projectiles: Vec<ark_bn254::G1Projective> = (0..base_size)
            .map(|_| ark_bn254::G1Projective::rand(&mut rng))
            .collect();
        let jolt_bases: Vec<Bn254G1> = projectiles.into_iter().map(Bn254G1::from).collect();

        let indices_sets: Vec<Vec<usize>> = (0..num_batches)
            .map(|_| {
                let size = (rng.next_u64() as usize) % 50 + 1;
                (0..size)
                    .map(|_| (rng.next_u64() as usize) % base_size)
                    .collect()
            })
            .collect();

        let results = batch_g1_additions_multi(&jolt_bases, &indices_sets);
        assert_eq!(results.len(), num_batches);
    }
}
