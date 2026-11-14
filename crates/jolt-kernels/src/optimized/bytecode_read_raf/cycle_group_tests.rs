use jolt_field::Fr;
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use super::*;

const GROUPS: usize = 4;

fn random_poly<F: JoltField>(rng: &mut ChaCha20Rng) -> Polynomial<F> {
    Polynomial::new((0..2 * GROUPS).map(|_| F::random(rng)).collect())
}

fn at<F: JoltField>(poly: &Polynomial<F>, y: usize, t: F) -> F {
    let (lo, hi) = pair(poly, y);
    lo + t * (hi - lo)
}

fn check_group_samples<F: JoltField>() {
    let mut rng = ChaCha20Rng::seed_from_u64(1966);
    for num_ra in 1..=8 {
        let ra: Vec<Polynomial<F>> = (0..num_ra).map(|_| random_poly(&mut rng)).collect();
        let combined = random_poly(&mut rng);
        #[cfg(feature = "akita")]
        let (fused_inc, fused_combined) = (random_poly(&mut rng), random_poly(&mut rng));
        // One sample per degree: the RA product times the coefficient,
        // which is quadratic once the fused term is present.
        let degree = num_ra + if cfg!(feature = "akita") { 2 } else { 1 };
        let kernel = CycleKernel {
            progress: RoundProgress::new(GROUPS.ilog2() as usize + 1),
            degree,
            ra: LazyFoldedRa::Dense(ra.clone()),
            combined: EqPrefix::from_evals(combined.evals().to_vec()),
            #[cfg(feature = "akita")]
            fused_inc: LazyFusedInc::Prefix {
                values: fused_inc.evals().to_vec(),
                tail: F::zero(),
            },
            #[cfg(feature = "akita")]
            fused_combined: EqPrefix::from_evals(fused_combined.evals().to_vec()),
            output_openings: Vec::new(),
        };
        if num_ra == 4 {
            let expected: Vec<F> = (0..degree)
                .map(|slot| {
                    let t = F::from_u64(if slot == 0 { 0 } else { slot as u64 + 1 });
                    (0..GROUPS)
                        .map(|y| {
                            let coefficient = at(&combined, y, t);
                            #[cfg(feature = "akita")]
                            let coefficient =
                                coefficient + at(&fused_inc, y, t) * at(&fused_combined, y, t);
                            ra.iter().fold(coefficient, |p, poly| p * at(poly, y, t))
                        })
                        .sum()
                })
                .collect();
            assert_eq!(kernel.four_ra_evals(GROUPS), Some(expected));
        }
        let mut pairs = vec![(F::zero(), F::zero()); num_ra];
        for y in 0..GROUPS {
            let start: Vec<F> = (0..degree).map(|_| F::random(&mut rng)).collect();
            let mut acc = start.clone();
            kernel.accumulate_group(y, &mut acc, &mut pairs);
            for (slot, (acc, start)) in acc.iter().zip(&start).enumerate() {
                let t = F::from_u64(if slot == 0 { 0 } else { slot as u64 + 1 });
                let coefficient = at(&combined, y, t);
                #[cfg(feature = "akita")]
                let coefficient = coefficient + at(&fused_inc, y, t) * at(&fused_combined, y, t);
                let expected = ra
                    .iter()
                    .fold(coefficient, |product, poly| product * at(poly, y, t));
                assert_eq!(*acc - *start, expected, "{num_ra} factors, slot {slot}");
            }
        }
    }
}
#[test]
fn stepped_group_samples_match_recomputed_samples() {
    check_group_samples::<Fr>();
}

#[cfg(feature = "akita")]
#[test]
fn stepped_group_samples_match_recomputed_samples_fp128() {
    check_group_samples::<Prime128Offset6A23801F>();
}
