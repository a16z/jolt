//! The row program reproduces the deferred check bit for bit on a real
//! opening, and its fixed shape at the fibonacci-2^18 profile fits 2^18 rows.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "tests report shapes and fail loudly"
)]

mod common;

use std::time::Instant;

use ark_bn254::{Fq12, Fr as ArkFr, G1Affine, G2Affine};
use ark_ff::UniformRand;
use jolt_wrapper::limb_table::dory::{
    DoryChallenges, DorySetupInputs, DoryStatement, FlattenedCheck, NativeCheck,
};
use jolt_wrapper::limb_table::program::Program;
use jolt_wrapper::limb_table::schedule::{build, Layout};
use jolt_wrapper::limb_table::tower::fq12_from_coords;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn cell(values: &[ark_bn254::Fq], lins: &[Vec<(u32, i32)>; 12]) -> Fq12 {
    fq12_from_coords(&std::array::from_fn(|c| {
        Program::lin_value(values, &lins[c])
    }))
}

fn print_sections(layout: &Layout) {
    let program = &layout.program;
    for section in &program.sections {
        println!(
            "section {:<14} rows {:>7}  [{}..{})",
            section.name,
            section.rows.len(),
            section.rows.start,
            section.rows.end
        );
    }
    println!(
        "rows {} (2^{:.3})  max_slots {}  max_kappa_sum {}  pins {}  inputs {}",
        program.len(),
        (program.len() as f64).log2(),
        program.max_slots(),
        program.max_kappa_sum(),
        program.pinned_rows().count(),
        program.input_rows.len()
    );
}

#[test]
fn program_reproduces_the_deferred_check_on_a_real_opening() {
    let opening = common::synthetic_opening(8, 5, 0xD0);
    let sigma = opening.witness.sigma();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(&opening.statement, sigma, n);
    let native = NativeCheck::evaluate(&check, &opening.setup, &opening.witness);
    assert!(native.holds(), "flattened deferred check holds natively");

    let start = Instant::now();
    let layout = build(&check, &opening.setup, sigma, n);
    println!("build {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    print_sections(&layout);
    let start = Instant::now();
    let values = layout
        .program
        .evaluate(&opening.witness.coordinates())
        .expect("no exceptional case");
    println!("evaluate {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    assert_eq!(
        cell(&values, &layout.rhs),
        native.rhs,
        "RHS multi-exponentiation"
    );
    assert_eq!(cell(&values, &layout.miller), native.miller, "Miller loop");
    assert_eq!(
        cell(&values, &layout.lhs),
        native.lhs,
        "final exponentiation"
    );
    layout.program.check_pins(&values).expect("pins hold");
}

#[test]
fn fibonacci_profile_fits_2_18_rows() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xF1B);
    let (sigma, n) = (11, 42);
    let random = |rng: &mut ChaCha20Rng| ArkFr::rand(rng);
    let statement = DoryStatement {
        rho: random(&mut rng),
        point: (0..2 * sigma).map(|_| random(&mut rng)).collect(),
        evaluation: random(&mut rng),
        challenges: DoryChallenges {
            beta: (0..sigma).map(|_| random(&mut rng)).collect(),
            alpha: (0..sigma).map(|_| random(&mut rng)).collect(),
            gamma: random(&mut rng),
            d: random(&mut rng),
        },
    };
    let gt = |rng: &mut ChaCha20Rng| Fq12::rand(rng);
    let setup = DorySetupInputs {
        chi: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        delta_1r: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        delta_2r: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        ht: gt(&mut rng),
        g1_0: G1Affine::rand(&mut rng),
        g2_0: G2Affine::rand(&mut rng),
        h1: G1Affine::rand(&mut rng),
        h2: G2Affine::rand(&mut rng),
    };
    let check = FlattenedCheck::derive(&statement, sigma, n);
    assert_eq!(check.gt.bases.len(), 9 * sigma + n + 4);
    let start = Instant::now();
    let layout = build(&check, &setup, sigma, n);
    println!("build {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    print_sections(&layout);
    assert!(
        layout.program.len() <= 1 << 18,
        "{} rows",
        layout.program.len()
    );
}
