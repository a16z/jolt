use ark_bn254::Fr;
use ark_ff::UniformRand;
use criterion::Criterion;
use jolt_core::{
    field::JoltField,
    subprotocols::{
        karatsuba::{coeff_kara_16, coeff_kara_32, coeff_kara_8, coeff_naive},
        quang_optimization::{prod16, prod8, FieldMulSmall},
    },
};
use rand_core::SeedableRng;

fn quang_optimization_branch<F: FieldMulSmall, const D: usize>(polys: &[(F, F); D]) {
    match D {
        16 => {
            prod16(polys[..16].try_into().unwrap());
        }
        8 => {
            prod8(polys[..8].try_into().unwrap());
        }
        _ => panic!("D must be 8 or 16"),
    }
}

fn karatsuba_branch<F: JoltField, const D: usize>(left: &[F; D], right: &[F; D]) {
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
    let polys: Vec<Vec<Fr>> = (0..D)
        .map(|_| vec![Fr::rand(&mut rng), Fr::rand(&mut rng)])
        .collect();

    c.bench_function(&format!("naive_{}", D), |b| {
        b.iter_with_setup(
            || (polys.clone(), polys.clone()),
            |(left, right)| {
                criterion::black_box(coeff_naive(&polys));
            },
        );
    });
}

fn benchmark_karatsuba<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
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

    c.bench_function(&format!("karatsuba_{}", D), |b| {
        b.iter_with_setup(
            || (left.clone(), right.clone()),
            |(left, right)| {
                criterion::black_box(karatsuba_branch::<Fr, D>(&left, &right));
            },
        );
    });
}

fn quang_optimization<F: JoltField, const D: usize>(c: &mut Criterion) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
    let polys: [(Fr, Fr); D] = core::array::from_fn(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)));

    c.bench_function(&format!("quang_optimization_{}", D), |b| {
        b.iter_with_setup(
            || polys.clone(),
            |polys| {
                criterion::black_box(quang_optimization_branch::<Fr, D>(&polys));
            },
        );
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(10));

    benchmark_naive::<Fr, 8>(&mut criterion);
    benchmark_karatsuba::<Fr, 8>(&mut criterion);

    benchmark_naive::<Fr, 16>(&mut criterion);
    benchmark_karatsuba::<Fr, 16>(&mut criterion);

    benchmark_naive::<Fr, 32>(&mut criterion);
    benchmark_karatsuba::<Fr, 32>(&mut criterion);

    benchmark_karatsuba::<Fr, 8>(&mut criterion);
    quang_optimization::<Fr, 8>(&mut criterion);
}
