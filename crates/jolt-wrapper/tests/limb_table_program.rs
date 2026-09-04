//! The fixed-layout program reproduces the deferred check bit for bit on a
//! real opening, and its shape at the fibonacci-2^18 profile fits 2^18 rows.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "tests report shapes and fail loudly"
)]

#[expect(
    dead_code,
    reason = "fixtures shared with the other limb-table test binaries"
)]
mod common;

use std::cmp::Reverse;
use std::time::Instant;

use ark_bn254::{Fq, Fq12, Fr as ArkFr};
use ark_ff::Zero;
use jolt_poly::EqPolynomial;
use jolt_wrapper::limb_table::dory::{DorySetupInputs, FlattenedCheck, NativeCheck, WireValues};
use jolt_wrapper::limb_table::layout::{Factor, ROWS};
use jolt_wrapper::limb_table::lookup::PublicColumns;
use jolt_wrapper::limb_table::program::Source;
use jolt_wrapper::limb_table::schedule::{build, Layout};
use jolt_wrapper::limb_table::tower::fq12_from_coords;

fn gt(values: &[Fq], rows: &[u32; 12]) -> Fq12 {
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
        layout.pieces().len(),
        layout.selected.len(),
        layout.digit_ops.len()
    );
    // Free cells (no row written in the cell) and their largest contiguous run.
    let used: Vec<bool> = (0..ROWS / 16)
        .map(|cell| {
            (0..16).any(|c| {
                program.rows[cell * 16 + c].source != Source::Compute
                    || !program.rows[cell * 16 + c].slots.is_empty()
                    || program.rows[cell * 16 + c].pin.is_some()
            })
        })
        .collect();
    let free = used.iter().filter(|u| !**u).count();
    let (mut best, mut best_start, mut run, mut run_start) = (0usize, 0usize, 0usize, 0usize);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (cell, u) in used.iter().enumerate() {
        if !*u {
            if run == 0 {
                run_start = cell;
            }
            run += 1;
            if run > best {
                best = run;
                best_start = run_start;
            }
        } else if run > 0 {
            runs.push((run_start, run));
            run = 0;
        }
    }
    if run > 0 {
        runs.push((run_start, run));
    }
    runs.sort_by_key(|(_, len)| Reverse(*len));
    println!(
        "free cells {free}, largest run {best} at {best_start}; runs ≥ 32: {:?}",
        runs.iter().filter(|(_, l)| *l >= 32).collect::<Vec<_>>()
    );
}

#[test]
fn program_reproduces_the_deferred_check_on_a_real_opening() {
    let opening = common::synthetic_opening(8, 5, 0xD0);
    let sigma = opening.witness.sigma();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(sigma, n);
    let values = WireValues::derive(&opening.statement, sigma, n, common::offset_challenge());
    let native = NativeCheck::evaluate(&check, &values, &opening.setup, &opening.witness);
    assert!(native.holds(), "flattened deferred check holds natively");

    let start = Instant::now();
    let layout = build(&check, &values, &opening.setup, &check.wires());
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

/// Random verifier-key constants of the right shape (for layout-shape tests).
fn random_setup(sigma: usize, seed: u64) -> DorySetupInputs {
    use ark_bn254::{G1Affine, G2Affine};
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let gt = |rng: &mut ChaCha20Rng| Fq12::rand(rng);
    DorySetupInputs {
        chi: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        delta_1r: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        delta_2r: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        ht: gt(&mut rng),
        g1_0: G1Affine::rand(&mut rng),
        g2_0: G2Affine::rand(&mut rng),
        h1: G1Affine::rand(&mut rng),
        h2: G2Affine::rand(&mut rng),
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
            pairs.push((name.clone(), ArkFr::rand(&mut rng)));
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
    WireValues::from_wires(pairs, ArkFr::rand(&mut rng))
}

#[test]
fn fibonacci_profile_fits_2_18_rows() {
    // σ = 11 (2^22 → 2^11 × 2^11 matrix), n = 42 committed polynomials.
    let (sigma, n) = (11, 42);
    let check = FlattenedCheck::derive(sigma, n);
    let values = random_values(&check, 0xF1);
    let setup = random_setup(sigma, 0xF2);
    let start = Instant::now();
    let layout = build(&check, &values, &setup, &check.wires());
    println!("build {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    print_shape(&layout);
    println!("used rows {} of {ROWS}", layout.used_rows());
    assert_eq!(layout.program.len(), ROWS);
    assert!(layout.used_rows() <= ROWS);
}

/// Every occurrence of the fibonacci profile owns one complete radix-16
/// string and one `Source::Window` row; the spare rows cannot hide an
/// omitted occurrence.
#[test]
fn fibonacci_profile_binds_every_link_occurrence_to_one_window_row() {
    use jolt_wrapper::limb_table::digits::{WINDOWS, WINDOW_ROWS};
    use jolt_wrapper::limb_table::schedule::WINDOW_ROW_BASE;

    let check = FlattenedCheck::derive(11, 42);
    let values = random_values(&check, 0x5E05E);
    let setup = random_setup(11, 0x5E05F);
    let layout = build(&check, &values, &setup, &check.wires());
    assert_eq!(check.wires().len(), 173);
    assert_eq!(layout.digit_bases, 175);
    assert_eq!(layout.link_occurrences, 230);
    let mut windows = vec![[0u8; WINDOWS]; layout.link_occurrences as usize];
    for op in &layout.digit_ops {
        windows[op.link as usize][op.w as usize] += 1;
    }
    for (occurrence, counts) in windows.iter().enumerate() {
        assert!(
            counts.iter().all(|&count| count == 1),
            "occurrence {occurrence} owns every window exactly once"
        );
    }
    for occurrence in 0..WINDOW_ROWS {
        assert!(matches!(
            layout.program.rows[WINDOW_ROW_BASE as usize + occurrence].source,
            Source::Window(_)
        ));
    }
}

/// The memoized evaluator agrees with the brute-force kernel MLEs on every
/// kernel of the layout (copies, fingerprints, selected-family domains), and
/// the kernel MLE agrees with its edge list on a sample.
#[test]
fn evaluator_matches_kernel_mles() {
    use ark_ff::UniformRand;
    use jolt_field::{Fr, Ring, Zero};
    use jolt_wrapper::limb_table::layout::{Kernel, LOG_ROWS};
    use jolt_wrapper::limb_table::verifier::Evaluator;
    use jolt_wrapper::stream::VerifierCost;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    let check = FlattenedCheck::derive(8, 5);
    let setup = random_setup(8, 7);
    let values = random_values(&check, 5);
    let layout = build(&check, &values, &setup, &check.wires());
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let point = |rng: &mut ChaCha20Rng| -> Vec<Fr> {
        (0..LOG_ROWS).map(|_| Fr::from(ArkFr::rand(rng))).collect()
    };
    let (r, rs) = (point(&mut rng), point(&mut rng));
    let mut kernels: Vec<Kernel> = layout.pieces().into_iter().map(|p| p.kernel).collect();
    for group in &layout.fingerprints {
        for (_, map) in group.maps.iter().chain(&group.conj_maps) {
            let mut factors = group.cell.clone();
            factors.push(map.clone());
            kernels.push(Kernel::new(factors));
        }
    }
    let mut cost = VerifierCost::default();
    let mut ev = Evaluator::new(&r, &rs, &mut cost);
    let eq_r = EqPolynomial::<Fr>::evals(&r.iter().rev().copied().collect::<Vec<_>>(), None);
    let eq_s = EqPolynomial::<Fr>::evals(&rs.iter().rev().copied().collect::<Vec<_>>(), None);
    let (mut bad, mut bad_edges) = (0, 0);
    for kernel in &kernels {
        let expected = kernel.mle(&r, &rs);
        let got = ev.kernel(&kernel.factors);
        if expected != got {
            bad += 1;
            if bad <= 3 {
                println!("evaluator mismatch: {kernel:?}");
            }
        }
        let edges = kernel
            .edges()
            .into_iter()
            .fold(Fr::zero(), |acc, (row, src, w)| {
                acc + eq_r[row as usize] * eq_s[src as usize] * Fr::from_i64(i64::from(w))
            });
        if edges != expected {
            bad_edges += 1;
            if bad_edges <= 3 {
                println!("edges mismatch: {kernel:?}");
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} kernels mismatch", kernels.len());
    assert_eq!(
        bad_edges,
        0,
        "{bad_edges} of {} kernel MLEs differ from their edges",
        kernels.len()
    );
    // Shared-cell group evaluation equals the sum of its kernels.
    let mut bad_groups = 0;
    for group in &layout.copies {
        let maps: Vec<(usize, &Factor)> = group.maps.iter().map(|(_, _, map)| (0, map)).collect();
        let mut bucket = [Fr::zero()];
        ev.group_into(&group.cell, &maps, &mut bucket);
        let via_group = bucket[0];
        let direct = group.maps.iter().fold(Fr::zero(), |acc, (_, _, map)| {
            let mut factors = group.cell.clone();
            factors.push(map.clone());
            acc + Kernel::new(factors).mle(&r, &rs)
        });
        if via_group != direct {
            bad_groups += 1;
            if bad_groups <= 3 {
                println!(
                    "group mismatch: cell {:?} maps {}",
                    group.cell,
                    group.maps.len()
                );
            }
        }
    }
    assert_eq!(
        bad_groups,
        0,
        "{bad_groups} of {} groups mismatch",
        layout.copies.len()
    );
    println!("{} kernels, {} fr_mul", kernels.len(), cost.fr_mul);
}

/// The copy kernels' edge lists equal the program's fixed operand reads, and
/// the fingerprint kernels' edges equal the table reads.
#[test]
fn kernels_match_program_rows() {
    use jolt_wrapper::limb_table::layout::{Kernel, Side};
    use jolt_wrapper::limb_table::relation::SLOTS;
    use std::collections::BTreeMap;
    let check = FlattenedCheck::derive(8, 5);
    let setup = random_setup(8, 7);
    let values = random_values(&check, 5);
    let layout = build(&check, &values, &setup, &check.wires());
    let public = PublicColumns::new(&layout);
    // (row, slot, side) -> (src, weight) from the program.
    let mut from_rows: BTreeMap<(u32, u8, u8), (u32, i32)> = BTreeMap::new();
    for (row, spec) in layout.program.rows.iter().enumerate() {
        let skip = public.kinds[row].fp_slots();
        for (s, slot) in spec.slots.iter().enumerate() {
            if slot.kappa != 0 {
                let _ = from_rows.insert((row as u32, s as u8, 0), (slot.x, slot.kappa));
            }
            if s >= skip {
                let _ = from_rows.insert((row as u32, s as u8, 1), (slot.y, 1));
            }
        }
    }
    let mut from_kernels: BTreeMap<(u32, u8, u8), (u32, i32)> = BTreeMap::new();
    let mut owner: BTreeMap<(u32, u8, u8), &str> = BTreeMap::new();
    let mut duplicates = 0;
    // Copies are recorded per family in order; attribute them by piece counts.
    let mut groups = layout.copies.iter();
    for family in &layout.families {
        let mut remaining = family.fixed_pieces;
        while remaining > 0 {
            let group = groups.next().expect("group");
            remaining -= group.maps.len();
            for piece in group.pieces() {
                let side = match piece.side {
                    Side::X => 0,
                    Side::Y => 1,
                };
                for (row, src, w) in piece.kernel.edges() {
                    if let Some(previous) = from_kernels.insert((row, piece.slot, side), (src, w)) {
                        duplicates += 1;
                        if duplicates <= 5 {
                            println!(
                                "duplicate edge row {row} slot {} side {side}: {} {:?} vs {} {:?}",
                                piece.slot,
                                family.name,
                                (src, w),
                                owner[&(row, piece.slot, side)],
                                previous
                            );
                        }
                    }
                    let _ = owner.insert((row, piece.slot, side), family.name);
                }
            }
        }
    }
    assert!(groups.next().is_none(), "unattributed copy groups");
    let mut shown = 0;
    for (key, value) in &from_rows {
        if from_kernels.get(key) != Some(value) && shown < 8 {
            println!(
                "row {} slot {} side {}: program {:?} kernels {:?} ({:?})",
                key.0,
                key.1,
                key.2,
                value,
                from_kernels.get(key),
                owner.get(key)
            );
            shown += 1;
        }
    }
    for (key, value) in &from_kernels {
        if from_rows.get(key) != Some(value) && shown < 16 {
            println!(
                "row {} slot {} side {}: kernels {:?} ({}) program {:?}",
                key.0,
                key.1,
                key.2,
                value,
                owner[key],
                from_rows.get(key)
            );
            shown += 1;
        }
    }
    assert_eq!(duplicates, 0, "duplicate kernel edges");
    assert_eq!(from_rows, from_kernels, "copy kernels vs program slots");
    // Fingerprints.
    let mut reads: BTreeMap<(u32, u8, bool), u32> = BTreeMap::new();
    for read in &layout.table_reads {
        let _ = reads.insert((read.row, read.slot, false), read.src);
        let _ = reads.insert((read.row, read.slot, true), read.src);
    }
    let mut fp_edges: BTreeMap<(u32, u8, bool), (u32, i32)> = BTreeMap::new();
    for group in &layout.fingerprints {
        for (conj, maps) in [(false, &group.maps), (true, &group.conj_maps)] {
            for (slot, map) in maps {
                let mut factors = group.cell.clone();
                factors.push(map.clone());
                for (row, src, w) in Kernel::new(factors).edges() {
                    let _ = fp_edges.insert((row, *slot, conj), (src, w));
                }
            }
        }
    }
    let fp_rows: BTreeMap<(u32, u8, bool), u32> =
        fp_edges.iter().map(|(k, (src, _))| (*k, *src)).collect();
    assert_eq!(reads, fp_rows, "fingerprint kernels vs table reads");
    let _ = SLOTS;
}

/// Verifier arithmetic of the row and link members at the fibonacci profile
/// (σ = 11, N = 42): every field multiplication of the exporter's derivation
/// (relation, public evaluations and `ω̃`, terms, link batching) is
/// observed, and the count stays within the 10k `Fr` multiplication budget.
#[test]
fn verifier_arithmetic_within_budget_at_fibonacci_profile() {
    use ark_ff::UniformRand;
    use jolt_field::Fr;
    use jolt_wrapper::limb_table::lookup::public_and_link_evals;
    use jolt_wrapper::limb_table::relation::{Col, LookupConstants, RowRelation};
    use jolt_wrapper::limb_table::stream::{StreamTermExporter, T2Challenges};
    use jolt_wrapper::stream::{ColumnId, TermContext, TermExporter, TermObserver, VerifierCost};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    let check = FlattenedCheck::derive(11, 42);
    let values = random_values(&check, 1);
    let setup = random_setup(11, 2);
    let layout = build(&check, &values, &setup, &check.wires());
    let mut rng = ChaCha20Rng::seed_from_u64(0xB0D6);
    let mut fr = || Fr::from(ArkFr::rand(&mut rng));
    // θ, the per-phase challenges, ρ; the stage point; two batching coefficients.
    let challenges: Vec<Fr> = (0..T2Challenges::count() + 2).map(|_| fr()).collect();
    let point: Vec<Fr> = (0..18).map(|_| fr()).collect();
    let batching = [fr(), fr()];
    let ids: Vec<ColumnId> = (0..Col::CLAIMED)
        .map(|i| ColumnId {
            group: i / 4,
            slot: i % 4,
        })
        .collect();
    let exporter = StreamTermExporter {
        layout: &layout,
        challenge_offset: 1,
        theta_offset: 0,
        rho_offset: 1 + T2Challenges::count(),
        columns: &ids,
        row_member: 0,
        link_member: 1,
    };
    let mut cost = VerifierCost::default();
    let terms = exporter.terms_observed(
        &TermContext {
            row_point: &point,
            batching_coefficients: &batching,
            challenges: &challenges,
        },
        &mut cost,
    );
    let max_degree = terms.iter().map(|t| t.factors.len()).max().expect("terms");

    // The same derivation component by component; the parts sum to the whole.
    let t2 = exporter.challenges(&challenges);
    let mut relation_cost = VerifierCost::default();
    let relation = RowRelation::new_with(
        t2.row.clone(),
        LookupConstants {
            one_row: layout.one_cell * 16,
        },
        &mut |a, b| relation_cost.fr_mul(a, b),
    );
    let tau_le = t2.tau_le();
    let r_le: Vec<Fr> = point.iter().rev().copied().collect();
    let mut public_cost = VerifierCost::default();
    let (public, _link) =
        public_and_link_evals(&layout, &relation, &tau_le, &r_le, t2.rho, &mut public_cost);
    let mut terms_cost = VerifierCost::default();
    let _ = relation.batched_terms(&public, batching[0], &mut |a, b| terms_cost.fr_mul(a, b));
    let link_batching = 3;
    assert_eq!(
        relation_cost.fr_mul + public_cost.fr_mul + terms_cost.fr_mul + link_batching,
        cost.fr_mul,
        "component counts sum to the exporter's"
    );
    println!(
        "fibonacci profile: execution-derived verifier {} fr_mul = relation {} + public \
         evaluations and link weights {} + terms {} + link batching {}; {} terms, max degree \
         {}, {} link occurrences over {} digit bases",
        cost.fr_mul,
        relation_cost.fr_mul,
        public_cost.fr_mul,
        terms_cost.fr_mul,
        link_batching,
        terms.len(),
        max_degree,
        layout.link_occurrences,
        layout.digit_bases
    );
    assert!(cost.fr_mul <= 10_000, "{} fr_mul", cost.fr_mul);
    // The measured count at this profile; every field constant is a literal,
    // so a cold process observes the same number (update deliberately).
    assert_eq!(cost.fr_mul, 9_963, "fr_mul at σ = 11, N = 42");
}
