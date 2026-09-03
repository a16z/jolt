//! The Miller loop cells reproduce arkworks' `multi_miller_loop` step by step
//! (independent oracle: `G2Prepared` lines and `mul_by_034`).

#![expect(clippy::expect_used, clippy::print_stdout)]
mod common;

use ark_bn254::{Config as Bn254Config, Fq, Fq12, Fq2, Fq6};
use ark_ec::bn::{BnConfig, G2Prepared};
use ark_ff::{Field, One};
use jolt_wrapper::limb_table::dory::{FlattenedCheck, NativeCheck, WireValues};
use jolt_wrapper::limb_table::schedule::{build, cells};
use jolt_wrapper::limb_table::tower::{fq12_coords, fq12_from_coords};

fn gt_cell(values: &[Fq], cell: u32) -> Fq12 {
    fq12_from_coords(&std::array::from_fn(|c| values[(cell * 16) as usize + c]))
}

fn fq2_at(values: &[Fq], row: u32) -> Fq2 {
    Fq2::new(values[row as usize], values[row as usize + 1])
}

#[test]
fn miller_cells_match_arkworks_step_by_step() {
    let opening = common::synthetic_opening(8, 5, 0xD0);
    let sigma = opening.witness.sigma();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(sigma, n);
    let values = WireValues::derive(&opening.statement, sigma, n);
    let native = NativeCheck::evaluate(&check, &values, &opening.setup, &opening.witness);
    let layout = build(&check, &values, &opening.setup, &check.wires());
    let coords = opening.witness.coordinates_in(&layout.input_order);
    let v = layout.program.evaluate(&coords).expect("eval");

    let g1_at = |cell: u32| {
        ark_bn254::G1Affine::new_unchecked(
            v[(cell * 16 + 14) as usize],
            v[(cell * 16 + 15) as usize],
        )
    };
    let g2_at = |half: u32, off: u32| {
        ark_bn254::G2Affine::new_unchecked(
            fq2_at(&v, half * 8 + off),
            fq2_at(&v, half * 8 + off + 2),
        )
    };
    println!(
        "E1_acc ok: {}",
        g1_at(layout.g1_outputs[0]) == native.e1_acc
    );
    println!(
        "A3 ok: {}",
        g1_at(layout.g1_outputs[1]) == native.pairs[2].0
    );
    println!(
        "A1 ok: {}",
        g1_at(layout.g1_outputs[2]) == native.pairs[0].0
    );
    println!(
        "A4 ok: {}",
        g1_at(layout.g1_outputs[3]) == native.pairs[3].0
    );
    println!(
        "E2_acc ok: {}",
        g2_at(layout.g2_outputs[0], 4) == native.e2_acc
    );
    println!(
        "B2 ok: {}",
        g2_at(layout.g2_outputs[1], 4) == native.pairs[1].1
    );
    println!(
        "Q0 ok: {}",
        g2_at(layout.q_halves[0], 0) == native.pairs[0].1
    );
    println!(
        "Q1 ok: {}",
        g2_at(layout.q_halves[1], 0) == native.pairs[1].1
    );
    // Pair 0: Q = E2_fin, P = A1.
    let (p0, q0) = native.pairs[0];
    let prepared = G2Prepared::<Bn254Config>::from(q0);
    let coeffs = &prepared.ell_coeffs;
    // My first doubling-step lines for p = 0.
    let group = cells::MILLER_DBL_LINES * 16;
    let h = fq2_at(&v, group + 12);
    let j = fq2_at(&v, group + 14);
    let i = fq2_at(&v, group + 18);
    println!(
        "line0 mine: c0={:?}\n c1={:?}\n c2={:?}",
        -h,
        j * Fq2::from(3u64),
        i
    );
    println!(
        "line0 ark : c0={:?}\n c1={:?}\n c2={:?}",
        coeffs[0].0, coeffs[0].1, coeffs[0].2
    );
    assert_eq!((-h, j * Fq2::from(3u64), i), coeffs[0]);
    // Replay arkworks' multi_miller_loop step by step against my cells.
    let pairs = native.pairs;
    let prepared_all: Vec<_> = pairs
        .iter()
        .map(|(_, q)| G2Prepared::<Bn254Config>::from(*q))
        .collect();
    let mut cursor = vec![0usize; 4];
    let mut f = Fq12::one();
    let ell = |f: &mut Fq12, pi: usize, cursor: &mut Vec<usize>| {
        let (p, _) = pairs[pi];
        let c = prepared_all[pi].ell_coeffs[cursor[pi]];
        cursor[pi] += 1;
        let mut c0 = c.0;
        c0.mul_assign_by_fp(&p.y);
        let mut c1 = c.1;
        c1.mul_assign_by_fp(&p.x);
        f.mul_by_034(&c0, &c1, &c.2);
    };
    let ate = Bn254Config::ATE_LOOP_COUNT;
    let mut a = 0u32;
    for t in 0..64u32 {
        if t > 0 {
            let _ = f.square_in_place();
            let mine = gt_cell(&v, cells::MILLER_DBL_GT + 8 * t);
            assert_eq!(mine, f, "sq at t={t}");
        }
        for pi in 0..4 {
            ell(&mut f, pi, &mut cursor);
            let cell = cells::MILLER_DBL_GT + 8 * t + 1 + pi as u32;
            let mine = gt_cell(&v, cell);
            if mine != f {
                let (p, _) = pairs[pi];
                let c = prepared_all[pi].ell_coeffs[cursor[pi] - 1];
                let mut c0 = c.0;
                c0.mul_assign_by_fp(&p.y);
                let mut c1 = c.1;
                c1.mul_assign_by_fp(&p.x);
                println!(
                    "f_prev (slot {}) = {:?}",
                    pi,
                    fq12_coords(&gt_cell(&v, cell - 1))
                );
                println!(
                    "scaled rows 12..16: {:?} {:?} {:?} {:?}",
                    v[(cell * 16 + 12) as usize],
                    v[(cell * 16 + 13) as usize],
                    v[(cell * 16 + 14) as usize],
                    v[(cell * 16 + 15) as usize]
                );
                println!("expected c0·Py = {:?}, c1·Px = {:?}", c0, c1);
                println!("mine {:?}", fq12_coords(&mine));
                println!("ark  {:?}", fq12_coords(&f));
                let spec = &layout.program.rows[(cell * 16) as usize];
                println!("row 0 slots: {:?}", spec.slots);
            }
            assert_eq!(mine, f, "dbl ell t={t} p={pi}");
        }
        if ate[63 - t as usize] != 0 {
            for pi in 0..4 {
                ell(&mut f, pi, &mut cursor);
                let mine = gt_cell(&v, cells::MILLER_ADD_GT + 4 * a + pi as u32);
                assert_eq!(mine, f, "add ell a={a} p={pi}");
            }
            a += 1;
        }
    }
    for extra in 0..2u32 {
        for pi in 0..4 {
            ell(&mut f, pi, &mut cursor);
            let mine = gt_cell(&v, cells::MILLER_ADD_GT + 4 * (a + extra) + pi as u32);
            assert_eq!(mine, f, "final ell {extra} p={pi}");
        }
    }
    assert_eq!(f, native.miller);
    let _ = (p0, Fq6::one());
}
