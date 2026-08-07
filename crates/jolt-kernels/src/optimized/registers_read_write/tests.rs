use jolt_claims::protocols::jolt::geometry::dimensions::{
    ReadWriteDimensions, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::Polynomial;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_witness::JoltWitnessOracle;

use super::test_support::{
    assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture, TraceFixture,
};
use super::OptimizedRegistersReadWrite;

fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
    fixture.with_plane(log_t, |backend| {
        let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
            log_t,
            REGISTER_ADDRESS_BITS,
            log_t,
            0,
        ));
        let r_cycle = challenge_sequence(log_t, seed ^ 0xA5A5);
        let evaluate = |polynomial: JoltVirtualPolynomial| {
            let table = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(polynomial),
            )
            .unwrap();
            Polynomial::new(table).evaluate(&r_cycle)
        };
        let gamma = Fr::from_u64(0x5EED_1234_5678_9ABC);
        let claims = RegistersReadWriteInputClaims {
            rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
            rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
            rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
        };
        let points = RegistersReadWriteInputClaims {
            rd_write_value: r_cycle.clone(),
            rs1_value: r_cycle.clone(),
            rs2_value: r_cycle,
        };
        let input_claim =
            claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
        assert_nontrivial(input_claim);
        let round_challenges = challenge_sequence(log_t + REGISTER_ADDRESS_BITS, seed);
        assert_kernel_parity(
            &OptimizedRegistersReadWrite,
            backend,
            &relation,
            &claims,
            &points,
            &RegistersReadWriteChallenges { gamma },
            input_claim,
            &round_challenges,
        );
    });
}

#[test]
fn parity_structured_odd_log_t() {
    run_parity(structured_fixture(8), 3, 17);
}

#[test]
fn parity_structured_even_log_t() {
    run_parity(structured_fixture(16), 4, 23);
}

#[test]
fn parity_past_lut_saturation() {
    // log_t = 6 runs three LUT-mode binds, the deref at the fourth, and
    // two more cycle binds on direct field coefficients.
    run_parity(structured_fixture(60), 6, 29);
}

#[test]
fn parity_minimal_padded_trace() {
    // Three real cycles padded to four: exercises the padding rows and
    // registers that are never touched.
    let mut fixture = TraceFixture::new();
    fixture.op(Some(3), Some(1), Some(2));
    fixture.op(Some(3), Some(3), None);
    fixture.op(None, Some(3), Some(3));
    run_parity(fixture, 2, 31);
}

#[test]
fn parity_single_cycle_round() {
    let mut fixture = TraceFixture::new();
    fixture.op(Some(9), Some(9), Some(9));
    fixture.op(Some(9), None, Some(9));
    run_parity(fixture, 1, 41);
}

/// The in-place bind must reproduce the out-of-place merge exactly —
/// same entries, same order — including across pair-aligned block
/// boundaries and the left-compaction of shrunken blocks (the entry set
/// is large enough for several blocks).
#[test]
fn in_place_bind_matches_out_of_place() {
    use super::sparse::{
        bind_sparse_entries, bind_sparse_entries_in_place, MatrixEntry, SparseEntry,
    };
    let mut state = 0xC0FF_EE00_1234_5678u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let mut entries: Vec<SparseEntry<Fr, Fr>> = Vec::new();
    for row in 0..100_000usize {
        let count = (next() % 4) as usize;
        let mut cols: Vec<u8> = (0..count).map(|_| (next() % 64) as u8).collect();
        cols.sort_unstable();
        cols.dedup();
        for col in cols {
            entries.push(SparseEntry {
                val: Fr::from_u64(next()),
                prev_val: next(),
                next_val: next(),
                row,
                ra: Fr::from_u64(next()),
                wa: Fr::from_u64(next()),
                col,
            });
        }
    }
    let unused = super::sparse::SparseEntries::<Fr>::unused_lut();
    let r = Fr::from_u64(0x1234_5678_9ABC_DEF1);
    let bind = |even: Option<&SparseEntry<Fr, Fr>>, odd: Option<&SparseEntry<Fr, Fr>>| {
        <SparseEntry<Fr, Fr> as MatrixEntry<Fr>>::bind(even, odd, r, &unused, &unused)
    };
    let expected = bind_sparse_entries(&entries, bind);
    let mut actual = entries;
    bind_sparse_entries_in_place(&mut actual, bind);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            (actual.row, actual.col),
            (expected.row, expected.col),
            "entry {index} position"
        );
        assert_eq!(actual.val, expected.val, "entry {index} val");
        assert_eq!(actual.ra, expected.ra, "entry {index} ra");
        assert_eq!(actual.wa, expected.wa, "entry {index} wa");
        assert_eq!(actual.prev_val, expected.prev_val, "entry {index} prev");
        assert_eq!(actual.next_val, expected.next_val, "entry {index} next");
    }
}

/// The SoA machinery (seed transition, in-place bind, quadratic walk,
/// direct transition) must reproduce the AoS machinery exactly — same
/// entries, same order, same summands — across every layout step.
#[test]
fn soa_paths_match_aos_machinery() {
    use super::sparse::{
        bind_indexed_in_place_soa, bind_indexed_to_direct, bind_seed_entries_soa,
        bind_sparse_entries, load_indexed, sparse_quadratic, sparse_quadratic_soa, CoeffLut,
        LutIndex, MatrixEntry, OneHotCoeff, SeedEntry, SparseEntries, SparseEntry,
    };
    let mut state = 0xFEED_5EED_1234_5678u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let mut seed: Vec<SeedEntry> = Vec::new();
    for row in 0..4096u32 {
        let count = (next() % 4) as usize;
        let mut cols: Vec<u8> = (0..count)
            .map(|_| (next() % (1 << REGISTER_ADDRESS_BITS)) as u8)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for col in cols {
            let prev_val = next();
            seed.push(SeedEntry {
                prev_val,
                next_val: if next() % 2 == 0 { prev_val } else { next() },
                row,
                ra: (next() % 4) as u8,
                wa: (next() % 2) as u8,
                col,
            });
        }
    }
    let gamma = Fr::from_u64(0x0D15_EA5E_0BAD_F00D);
    let mut ra_lut = CoeffLut::new(vec![
        Fr::from_u64(0),
        gamma,
        gamma * gamma,
        gamma + gamma * gamma,
    ]);
    let mut wa_lut = CoeffLut::new(vec![Fr::from_u64(0), Fr::from_u64(1)]);
    let (r1, r2, r3) = (
        Fr::from_u64(next() | 1),
        Fr::from_u64(next() | 1),
        Fr::from_u64(next() | 1),
    );

    // Step 1: seed transition.
    let (mut vals, mut metas) = bind_seed_entries_soa(&seed, &ra_lut, &wa_lut, r1);
    let mut aos: Vec<SparseEntry<Fr, LutIndex>> = bind_sparse_entries(&seed, |even, odd| {
        <SeedEntry as MatrixEntry<Fr>>::bind(even, odd, r1, &ra_lut, &wa_lut)
    });
    ra_lut.bind(r1);
    wa_lut.bind(r1);
    let assert_soa_matches =
        |vals: &[Fr], metas: &[super::sparse::IndexedMeta], aos: &[SparseEntry<Fr, LutIndex>]| {
            assert_eq!(vals.len(), aos.len());
            assert_eq!(metas.len(), aos.len());
            for (index, expected) in aos.iter().enumerate() {
                let actual = load_indexed((vals, metas), index);
                assert_eq!(
                    (actual.row, actual.col),
                    (expected.row, expected.col),
                    "entry {index} position"
                );
                assert_eq!(actual.val, expected.val, "entry {index} val");
                assert_eq!(actual.ra.0, expected.ra.0, "entry {index} ra");
                assert_eq!(actual.wa.0, expected.wa.0, "entry {index} wa");
                assert_eq!(actual.prev_val, expected.prev_val, "entry {index} prev");
                assert_eq!(actual.next_val, expected.next_val, "entry {index} next");
            }
        };
    assert_soa_matches(&vals, &metas, &aos);

    // Step 2: round message over the indexed state (z < 1024, in_bits 5).
    let e_in: Vec<Fr> = (0..32).map(|_| Fr::from_u64(next() | 1)).collect();
    let e_out: Vec<Fr> = (0..32).map(|_| Fr::from_u64(next() | 1)).collect();
    let inc_at = |z: usize| {
        [
            Fr::from_u64((z as u64) | 1),
            Fr::from_u64((z as u64).wrapping_mul(0x9E37_79B9) | 1),
        ]
    };
    assert_eq!(
        sparse_quadratic(&aos, &ra_lut, &wa_lut, &e_in, &e_out, inc_at),
        sparse_quadratic_soa(&vals, &metas, &ra_lut, &wa_lut, &e_in, &e_out, inc_at),
        "quadratic mismatch on the indexed state"
    );

    // Step 3: same-layout in-place bind.
    bind_indexed_in_place_soa(&mut vals, &mut metas, &ra_lut, &wa_lut, r2);
    aos = bind_sparse_entries(&aos, |even, odd| {
        <SparseEntry<Fr, LutIndex> as MatrixEntry<Fr>>::bind(even, odd, r2, &ra_lut, &wa_lut)
    });
    ra_lut.bind(r2);
    wa_lut.bind(r2);
    assert_soa_matches(&vals, &metas, &aos);

    // Step 4: direct transition (deref during the merge).
    let direct = bind_indexed_to_direct(&vals, &metas, &ra_lut, &wa_lut, r3);
    let unused = SparseEntries::<Fr>::unused_lut();
    let deref = |entry: &SparseEntry<Fr, LutIndex>| SparseEntry::<Fr, Fr> {
        val: entry.val,
        prev_val: entry.prev_val,
        next_val: entry.next_val,
        row: entry.row,
        ra: entry.ra.value(&ra_lut),
        wa: entry.wa.value(&wa_lut),
        col: entry.col,
    };
    let expected_direct = bind_sparse_entries(&aos, |even, odd| {
        <SparseEntry<Fr, Fr> as MatrixEntry<Fr>>::bind(
            even.map(deref).as_ref(),
            odd.map(deref).as_ref(),
            r3,
            &unused,
            &unused,
        )
    });
    assert_eq!(direct.len(), expected_direct.len());
    for (index, (actual, expected)) in direct.iter().zip(&expected_direct).enumerate() {
        assert_eq!(
            (actual.row, actual.col),
            (expected.row, expected.col),
            "direct entry {index} position"
        );
        assert_eq!(actual.val, expected.val, "direct entry {index} val");
        assert_eq!(actual.ra, expected.ra, "direct entry {index} ra");
        assert_eq!(actual.wa, expected.wa, "direct entry {index} wa");
    }
}

/// Job-2 parity pin: the fused round-1 quadratic and the fused
/// Seed → Indexed transition must reproduce the sequential two-bind path
/// exactly — same round message, same entries, same order, same LUT
/// indices — on a random workload with empty rows, empty half-groups and
/// column collisions.
#[test]
fn fused_first_two_binds_match_sequential() {
    use super::sparse::{
        bind_indexed_in_place_soa, bind_seed_entries_fused, bind_seed_entries_soa,
        sparse_quadratic_fused, sparse_quadratic_soa, CoeffLut, SeedEntry,
    };
    let mut state = 0xB1BD_F00D_9876_5432u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let mut seed: Vec<SeedEntry> = Vec::new();
    for row in 0..4096u32 {
        let count = (next() % 4) as usize;
        let mut cols: Vec<u8> = (0..count)
            .map(|_| (next() % (1 << REGISTER_ADDRESS_BITS)) as u8)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for col in cols {
            let prev_val = next();
            seed.push(SeedEntry {
                prev_val,
                next_val: if next() % 2 == 0 { prev_val } else { next() },
                row,
                ra: (next() % 4) as u8,
                wa: (next() % 2) as u8,
                col,
            });
        }
    }
    let gamma = Fr::from_u64(0xFACE_FEED_0123_4567);
    let seed_ra_lut = CoeffLut::new(vec![
        Fr::from_u64(0),
        gamma,
        gamma * gamma,
        gamma + gamma * gamma,
    ]);
    let seed_wa_lut = CoeffLut::new(vec![Fr::from_u64(0), Fr::from_u64(1)]);
    let (r1, r2) = (Fr::from_u64(next() | 1), Fr::from_u64(next() | 1));

    // Sequential: materialize the T/2 generation, square, message, bind
    // in place, square again.
    let (mut seq_vals, mut seq_metas) =
        bind_seed_entries_soa(&seed, &seed_ra_lut, &seed_wa_lut, r1);
    let mut ra_lut = CoeffLut::new(seed_ra_lut.values.clone());
    let mut wa_lut = CoeffLut::new(seed_wa_lut.values.clone());
    ra_lut.bind(r1);
    wa_lut.bind(r1);
    let e_in: Vec<Fr> = (0..32).map(|_| Fr::from_u64(next() | 1)).collect();
    let e_out: Vec<Fr> = (0..32).map(|_| Fr::from_u64(next() | 1)).collect();
    let inc_at = |z: usize| {
        [
            Fr::from_u64((z as u64) | 1),
            Fr::from_u64((z as u64).wrapping_mul(0x9E37_79B9) | 1),
        ]
    };
    let sequential_quadratic = sparse_quadratic_soa(
        &seq_vals, &seq_metas, &ra_lut, &wa_lut, &e_in, &e_out, inc_at,
    );
    let fused_quadratic = sparse_quadratic_fused(
        &seed,
        &seed_ra_lut,
        &seed_wa_lut,
        &ra_lut,
        &wa_lut,
        r1,
        &e_in,
        &e_out,
        inc_at,
    );
    assert_eq!(
        sequential_quadratic, fused_quadratic,
        "round-1 quadratic mismatch"
    );

    let (fused_vals, fused_metas) =
        bind_seed_entries_fused(&seed, &seed_ra_lut, &seed_wa_lut, &ra_lut, &wa_lut, r1, r2);
    bind_indexed_in_place_soa(&mut seq_vals, &mut seq_metas, &ra_lut, &wa_lut, r2);
    assert_eq!(fused_vals.len(), seq_vals.len());
    for index in 0..seq_vals.len() {
        let (expected, actual) = (seq_metas[index], fused_metas[index]);
        // Tuples of copies: packed fields cannot be referenced directly.
        assert_eq!(
            (
                actual.row,
                actual.col,
                actual.ra,
                actual.wa,
                actual.prev_val,
                actual.next_val
            ),
            (
                expected.row,
                expected.col,
                expected.ra,
                expected.wa,
                expected.prev_val,
                expected.next_val
            ),
            "fused entry {index} meta"
        );
        assert_eq!(
            fused_vals[index], seq_vals[index],
            "fused entry {index} val"
        );
    }
}
