use ark_bn254::Fr;
use ark_ff::UniformRand;
use criterion::Criterion;
use jolt_core::{
    field::JoltField,
    poly::unipoly::UniPoly,
    subprotocols::{
        karatsuba::{coeff_kara_16, coeff_kara_32, coeff_kara_4, coeff_kara_8, coeff_naive},
        toom::{eval_toom16, eval_toom4, eval_toom8, FieldMulSmall},
    },
};
use num::Integer;
use rand_core::SeedableRng;

fn toom_branch<F: FieldMulSmall, const D: usize>(polys: &[(F, F); D]) {
    if D == 4 {
        eval_toom4(polys[..4].try_into().unwrap());
    } else if D == 8 {
        eval_toom8(polys[..8].try_into().unwrap());
    } else if D == 16 {
        eval_toom16(polys[..16].try_into().unwrap());
    } else {
        panic!("D must be 4, 8, or 16");
    }
}

fn karatsuba_branch<F: JoltField, const D: usize>(left: &[F; D], right: &[F; D]) {
    if D == 4 {
        coeff_kara_4(
            left[..4].try_into().unwrap(),
            right[..4].try_into().unwrap(),
        );
    }
    if D == 8 {
        coeff_kara_8(
            left[..8].try_into().unwrap(),
            right[..8].try_into().unwrap(),
        );
    }
    if D == 16 {
        coeff_kara_16(
            left[..16].try_into().unwrap(),
            right[..16].try_into().unwrap(),
        );
    }
    if D == 32 {
        coeff_kara_32(
            left[..32].try_into().unwrap(),
            right[..32].try_into().unwrap(),
        );
    }
}

fn benchmark_naive<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
    c.bench_function(&format!("naive_{}", D), |b| {
        b.iter_with_setup(
            || {
                (0..D)
                    .map(|_| vec![Fr::rand(&mut rng), Fr::rand(&mut rng)])
                    .collect::<Vec<_>>()
            },
            |polys| {
                criterion::black_box(coeff_naive(&polys));
            },
        );
    });
}

fn toom_to_univariate<F: FieldMulSmall, const D: usize>(polys: &[(F, F); D]) -> UniPoly<F> {
    let univariate_poly = match D {
        4 => eval_toom4(polys[..4].try_into().unwrap()).to_vec(),
        8 => eval_toom8(polys[..8].try_into().unwrap()).to_vec(),
        16 => eval_toom16(polys[..16].try_into().unwrap()).to_vec(),
        _ => panic!("D must be 4, 8, or 16"),
    };
    UniPoly::from_evals(&univariate_poly)
}

fn benchmark_toom_to_univariate_poly<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
    c.bench_function(&format!("toom_to_univariate_{}", D), |b| {
        b.iter_with_setup(
            || {
                let polys: [(Fr, Fr); D] = (0..D)
                    .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                polys
            },
            |polys| {
                criterion::black_box(toom_to_univariate(&polys));
            },
        );
    });
}

fn benchmark_toom<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
    c.bench_function(&format!("toom_{}", D), |b| {
        b.iter_with_setup(
            || {
                let polys: [(Fr, Fr); D] = (0..D)
                    .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                polys
            },
            |polys| {
                criterion::black_box(toom_branch(&polys));
            },
        );
    });
}

fn karatsuba_to_univariate<F: JoltField, const D: usize>(
    left: &[F; D],
    right: &[F; D],
) -> UniPoly<F> {
    let left: [F; D] = core::array::from_fn(|i| {
        // Optimization
        if i.is_even() {
            left[i]
        } else {
            left[i] - left[i - 1]
        }
    });

    let right: [F; D] = core::array::from_fn(|i| {
        // Optimization
        if i.is_even() {
            right[i]
        } else {
            right[i] - right[i - 1]
        }
    });

    let univariate_poly = match D {
        4 => coeff_kara_4(
            left[..4].try_into().unwrap(),
            right[..4].try_into().unwrap(),
        )
        .to_vec(),
        8 => coeff_kara_8(
            left[..8].try_into().unwrap(),
            right[..8].try_into().unwrap(),
        )
        .to_vec(),
        16 => coeff_kara_16(
            left[..16].try_into().unwrap(),
            right[..16].try_into().unwrap(),
        )
        .to_vec(),
        32 => coeff_kara_32(
            left[..32].try_into().unwrap(),
            right[..32].try_into().unwrap(),
        )
        .to_vec(),
        _ => panic!("D must be 4, 8, 16, or 32"),
    };
    UniPoly::from_evals(&univariate_poly)
}

fn benchmark_karatsuba_to_univariate_poly<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
    c.bench_function(&format!("karatsuba_to_univariate_{}", D), |b| {
        b.iter_with_setup(
            || {
                let polys: Vec<Vec<Fr>> = (0..D)
                    .map(|_| vec![Fr::rand(&mut rng), Fr::rand(&mut rng)])
                    .collect();
                let left: [Fr; D] = polys[..D / 2]
                    .iter()
                    .flat_map(|p| [p[0], p[1]])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let right: [Fr; D] = polys[D / 2..]
                    .iter()
                    .flat_map(|p| [p[0], p[1]])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                (left, right)
            },
            |(left, right)| {
                criterion::black_box(karatsuba_to_univariate(&left, &right));
            },
        );
    });
}

fn benchmark_karatsuba<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
    c.bench_function(&format!("karatsuba_{}", D), |b| {
        b.iter_with_setup(
            || {
                let polys: Vec<Vec<Fr>> = (0..D)
                    .map(|_| vec![Fr::rand(&mut rng), Fr::rand(&mut rng)])
                    .collect();
                let left: [Fr; D] = polys[..D / 2]
                    .iter()
                    .flat_map(|p| [p[0], p[1]])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let right: [Fr; D] = polys[D / 2..]
                    .iter()
                    .flat_map(|p| [p[0], p[1]])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                (left, right)
            },
            |(left, right)| {
                criterion::black_box(karatsuba_branch::<Fr, D>(&left, &right));
            },
        );
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(10));

    benchmark_naive::<Fr, 4>(&mut criterion);
    benchmark_karatsuba::<Fr, 4>(&mut criterion);
    benchmark_toom::<Fr, 4>(&mut criterion);
    benchmark_toom_to_univariate_poly::<Fr, 4>(&mut criterion);
    benchmark_karatsuba_to_univariate_poly::<Fr, 4>(&mut criterion);

    benchmark_naive::<Fr, 8>(&mut criterion);
    benchmark_karatsuba::<Fr, 8>(&mut criterion);
    benchmark_toom::<Fr, 8>(&mut criterion);
    benchmark_toom_to_univariate_poly::<Fr, 8>(&mut criterion);
    benchmark_karatsuba_to_univariate_poly::<Fr, 8>(&mut criterion);

    // benchmark_naive::<Fr, 16>(&mut criterion);
    benchmark_karatsuba::<Fr, 16>(&mut criterion);
    benchmark_toom::<Fr, 16>(&mut criterion);
    benchmark_toom_to_univariate_poly::<Fr, 16>(&mut criterion);
    benchmark_karatsuba_to_univariate_poly::<Fr, 16>(&mut criterion);

    benchmark_naive::<Fr, 32>(&mut criterion);
    benchmark_karatsuba::<Fr, 32>(&mut criterion);
}
