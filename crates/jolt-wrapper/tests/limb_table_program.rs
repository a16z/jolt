//! The fixed-layout program reproduces the deferred check bit for bit on a
//! real opening, and its shape at the fibonacci-2^18 profile fits 2^18 rows.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "tests report shapes and fail loudly"
)]

mod common;

use std::time::Instant;

use ark_bn254::Fq12;
use ark_ff::Zero;
use jolt_wrapper::limb_table::dory::{FlattenedCheck, NativeCheck, WireValues};
use jolt_wrapper::limb_table::layout::ROWS;
use jolt_wrapper::limb_table::schedule::{build, Layout};
use jolt_wrapper::limb_table::tower::fq12_from_coords;

fn gt(values: &[ark_bn254::Fq], rows: &[u32; 12]) -> Fq12 {
    fq12_from_coords(&std::array::from_fn(|c| values[rows[c] as usize]))
}

fn print_shape(layout: &Layout) {
    let mut total_rows = 0;
    let mut merged: Vec<(&str, usize, usize, usize, usize, usize)> = Vec::new();
    for family in &layout.families {
        total_rows += family.rows;
        match merged.iter_mut().find(|m| m.0 == family.name) {
            Some(m) => {
                m.1 += family.ops;
                m.2 += family.rows;
                m.3 += family.fixed_pieces;
                m.4 += family.selected_pieces;
                m.5 += 1;
            }
            None => merged.push((
                family.name,
                family.ops,
                family.rows,
                family.fixed_pieces,
                family.selected_pieces,
                1,
            )),
        }
    }
    for (name, ops, rows, fixed, selected, families) in &merged {
        println!(
            "{name:<22} ops {ops:>6} rows {rows:>7} pieces {fixed:>4} selected {selected:>3} families {families:>2}"
        );
    }
    let program = &layout.program;
    println!(
        "used rows {} of {} (families {}, leaves/constants {})  max_slots {}  max_kappa_sum {}  pins {}  inputs {}  fixed pieces {}  selected pieces {}  digits {}",
        layout.used_rows(),
        ROWS,
        total_rows,
        layout.used_rows() - total_rows,
        program.max_slots(),
        program.max_kappa_sum(),
        program.pinned_rows().count(),
        program.input_rows.len(),
        layout.pieces.len(),
        layout.selected.len(),
        layout.digits.len()
    );
    let cost: usize = layout.pieces.iter().map(|p| p.kernel.cost()).sum();
    println!("fixed-kernel verifier cost ≈ {cost} field multiplications");
}

#[test]
fn program_reproduces_the_deferred_check_on_a_real_opening() {
    let opening = common::synthetic_opening(8, 5, 0xD0);
    let sigma = opening.witness.sigma();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(sigma, n);
    let values = WireValues::derive(&opening.statement, sigma, n);
    let native = NativeCheck::evaluate(&check, &values, &opening.setup, &opening.witness);
    assert!(native.holds(), "flattened deferred check holds natively");

    let start = Instant::now();
    let layout = build(&check, &values, &opening.setup);
    println!("build {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    print_shape(&layout);
    let start = Instant::now();
    let coords = opening.witness.coordinates_in(&layout.input_order);
    let row_values = layout
        .program
        .evaluate(&coords)
        .expect("no exceptional case");
    println!("evaluate {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    assert_eq!(
        gt(&row_values, &layout.rhs),
        native.rhs,
        "RHS multi-exponentiation"
    );
    assert_eq!(
        gt(&row_values, &layout.miller),
        native.miller,
        "Miller loop"
    );
    assert_eq!(
        gt(&row_values, &layout.lhs),
        native.lhs,
        "final exponentiation"
    );
    layout
        .program
        .check_pins(&row_values)
        .expect("every pinned row holds");
    for row in layout.final_check {
        assert!(row_values[row as usize].is_zero());
    }
}

/// Random wire values for every named scalar the check reads (the shape
/// does not depend on them).
fn random_values(check: &FlattenedCheck, seed: u64) -> WireValues {
    use ark_ff::UniformRand;
    use jolt_wrapper::limb_table::dory::Wire;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut pairs = Vec::new();
    let mut push = |wire: &Wire| {
        if let Wire::Named(name) = wire {
            pairs.push((name.clone(), ark_bn254::Fr::rand(&mut rng)));
        }
    };
    for (_, wire) in &check.gt.bases {
        push(wire);
    }
    for msm in check.g1_chains() {
        for (_, wire) in &msm.bases {
            push(wire);
        }
    }
    for msm in check.g2_chains() {
        for (_, wire) in &msm.bases {
            push(wire);
        }
    }
    WireValues::from_wires(pairs)
}

#[test]
fn fibonacci_profile_fits_2_18_rows() {
    // σ = 11 (2^22 → 2^11 × 2^11 matrix), n = 42 committed polynomials.
    let (sigma, n) = (11, 42);
    let check = FlattenedCheck::derive(sigma, n);
    let values = random_values(&check, 0xF1);
    let setup = common::random_setup(sigma, 0xF2);
    let start = Instant::now();
    let layout = build(&check, &values, &setup);
    println!("build {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    print_shape(&layout);
    assert_eq!(layout.program.len(), ROWS);
    assert!(layout.used_rows() <= ROWS);
}
