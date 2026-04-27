//! End-to-end: a synthetic 3-cycle FR trace feeds `FieldRegConfig` into
//! `DerivedSource`, and the FR Twist materializers produce:
//! - `FieldRegRaRs1` / `FieldRegRaRs2`: K_FR×T one-hots matching the
//!   per-cycle frs1/frs2 indices.
//! - `FieldRegWa`: K_FR×T one-hot at each write cycle's frd slot.
//! - `FieldRegVal`: K_FR×T running state, each element an Fr.
//! - `FrdGatherIndex`: T-element per-cycle write slot (or sentinel).
//!
//! Without a config, everything materializes to zero — preserving the
//! Phase-4d behavior for traces with no FR cycles.
#![allow(non_snake_case)]

use jolt_field::Fr;
use jolt_witness::{
    FieldRegConfig, FieldRegEvent, FrCycleBytecode, PolynomialConfig, PolynomialId,
};
use num_traits::{One, Zero};

const K_FR: usize = 16;

/// Build a minimal DerivedSource for testing.  We borrow a trivial
/// witness buffer (all zero) since the FR materializers don't consult
/// `witness[…]` — they read from the attached `FieldRegConfig`.
fn make_derived(
    witness: &[Fr],
    num_cycles: usize,
    vars_padded: usize,
    cfg: Option<FieldRegConfig>,
) -> jolt_witness::derived::DerivedSource<'_, Fr> {
    let mut d = jolt_witness::derived::DerivedSource::<Fr>::new(witness, num_cycles, vars_padded);
    if let Some(cfg) = cfg {
        d = d.with_field_reg(cfg);
    }
    d
}

#[test]
fn field_reg_ra_and_wa_one_hots_match_events() {
    // 3-cycle trace: FieldMov writes f1=10, FieldMov writes f2=20,
    // FieldAdd reads f1+f2, writes f3=30.
    let bytecode = vec![
        // FieldMov f1: writes frd=1
        FrCycleBytecode {
            frs1: 0,
            frs2: 0,
            frd: 1,
            reads_frs1: false,
            reads_frs2: false,
            writes_frd: true,
        },
        // FieldMov f2: writes frd=2
        FrCycleBytecode {
            frs1: 0,
            frs2: 0,
            frd: 2,
            reads_frs1: false,
            reads_frs2: false,
            writes_frd: true,
        },
        // FieldAdd: reads frs1=1, frs2=2, writes frd=3
        FrCycleBytecode {
            frs1: 1,
            frs2: 2,
            frd: 3,
            reads_frs1: true,
            reads_frs2: true,
            writes_frd: true,
        },
    ];
    let events = vec![
        FieldRegEvent {
            cycle: 0,
            slot: 1,
            old: [0; 4],
            new: [10, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 1,
            slot: 2,
            old: [0; 4],
            new: [20, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 2,
            slot: 3,
            old: [0; 4],
            new: [30, 0, 0, 0],
        },
    ];
    let t = 3usize;
    let witness = vec![Fr::zero(); t * 64]; // vars_padded = 64
    let d = make_derived(
        &witness,
        t,
        64,
        Some(FieldRegConfig { bytecode, events }),
    );

    // FieldRegRaRs1 — only cycle 2 reads frs1=1, so one-hot at (slot 1,
    // cycle 2) and zero everywhere else.
    let rs1_ra = d.compute(PolynomialId::FieldRegRaRs1);
    for (k, slot_chunk) in rs1_ra.chunks(t).enumerate() {
        for (c, &v) in slot_chunk.iter().enumerate() {
            let expected = if k == 1 && c == 2 {
                Fr::one()
            } else {
                Fr::zero()
            };
            assert_eq!(v, expected, "rs1_ra mismatch at (slot={k}, cycle={c})");
        }
    }

    // FieldRegRaRs2 — only cycle 2 reads frs2=2.
    let rs2_ra = d.compute(PolynomialId::FieldRegRaRs2);
    for (k, slot_chunk) in rs2_ra.chunks(t).enumerate() {
        for (c, &v) in slot_chunk.iter().enumerate() {
            let expected = if k == 2 && c == 2 {
                Fr::one()
            } else {
                Fr::zero()
            };
            assert_eq!(v, expected, "rs2_ra mismatch at (slot={k}, cycle={c})");
        }
    }

    // FieldRegWa — write slots: cycle 0 slot 1, cycle 1 slot 2, cycle 2 slot 3.
    let wa = d.compute(PolynomialId::FieldRegWa);
    let expected_writes = [(1usize, 0usize), (2, 1), (3, 2)];
    for (k, slot_chunk) in wa.chunks(t).enumerate() {
        for (c, &v) in slot_chunk.iter().enumerate() {
            let is_written = expected_writes.iter().any(|(ek, ec)| *ek == k && *ec == c);
            let expected = if is_written { Fr::one() } else { Fr::zero() };
            assert_eq!(v, expected, "wa mismatch at (slot={k}, cycle={c})");
        }
    }
}

#[test]
fn field_reg_val_tracks_running_state() {
    let bytecode = vec![FrCycleBytecode::default(); 3];
    let events = vec![
        FieldRegEvent {
            cycle: 0,
            slot: 5,
            old: [0; 4],
            new: [42, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 2,
            slot: 5,
            old: [42, 0, 0, 0],
            new: [100, 0, 0, 0],
        },
    ];
    let t = 3usize;
    let witness = vec![Fr::zero(); t * 64];
    let d = make_derived(
        &witness,
        t,
        64,
        Some(FieldRegConfig { bytecode, events }),
    );
    let val = d.compute(PolynomialId::FieldRegVal);

    // Slot 5 across 3 cycles: pre-execution state is 0 at cycle 0,
    // then 42 at cycles 1 and 2 (the cycle-2 event hasn't applied yet).
    let slot5 = &val[5 * t..5 * t + t];
    assert_eq!(slot5[0], Fr::zero(), "slot 5, cycle 0 pre-state is 0");
    assert_eq!(slot5[1], Fr::from(42u64), "slot 5, cycle 1 pre-state is 42");
    assert_eq!(slot5[2], Fr::from(42u64), "slot 5, cycle 2 pre-state is 42");

    // All other slots stay at zero across all cycles.
    for k in 0..K_FR {
        if k == 5 {
            continue;
        }
        for c in 0..t {
            assert_eq!(val[k * t + c], Fr::zero(), "slot {k}, cycle {c} should be 0");
        }
    }
}

#[test]
fn frd_gather_index_marks_writes() {
    // `frd_gather_index` is sourced from BYTECODE (not events) so the FR
    // write-slot indicator inherits a committed-bytecode anchor. Set
    // writes_frd/frd in the bytecode entries; events provide the values
    // (used by other materializers but not by frd_gather_index).
    let bytecode = vec![
        FrCycleBytecode::default(),
        FrCycleBytecode {
            frd: 7,
            writes_frd: true,
            ..Default::default()
        },
        FrCycleBytecode::default(),
        FrCycleBytecode {
            frd: 3,
            writes_frd: true,
            ..Default::default()
        },
    ];
    let events = vec![
        FieldRegEvent {
            cycle: 1,
            slot: 7,
            old: [0; 4],
            new: [1, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 3,
            slot: 3,
            old: [0; 4],
            new: [2, 0, 0, 0],
        },
    ];
    let t = 4usize;
    let witness = vec![Fr::zero(); t * 64];
    let d = make_derived(
        &witness,
        t,
        64,
        Some(FieldRegConfig { bytecode, events }),
    );
    let gather = d.compute(PolynomialId::FrdGatherIndex);

    let sentinel = Fr::from(u64::MAX);
    assert_eq!(gather[0], sentinel, "cycle 0 no write → sentinel");
    assert_eq!(gather[1], Fr::from(7u64), "cycle 1 bytecode.frd = 7");
    assert_eq!(gather[2], sentinel, "cycle 2 no write → sentinel");
    assert_eq!(gather[3], Fr::from(3u64), "cycle 3 bytecode.frd = 3");
}

#[test]
fn no_config_yields_zero_materializers() {
    let t = 4usize;
    let witness = vec![Fr::zero(); t * 64];
    let d = make_derived(&witness, t, 64, None);

    for id in [
        PolynomialId::FieldRegRaRs1,
        PolynomialId::FieldRegRaRs2,
        PolynomialId::FieldRegWa,
        PolynomialId::FieldRegVal,
    ] {
        let poly = d.compute(id);
        assert_eq!(poly.len(), K_FR * t, "{id:?} length");
        assert!(
            poly.iter().all(|v| v.is_zero()),
            "{id:?} should be all-zero without FieldRegConfig"
        );
    }

    // FrdGatherIndex is T-length sentinels.
    let gather = d.compute(PolynomialId::FrdGatherIndex);
    let sentinel = Fr::from(u64::MAX);
    assert_eq!(gather.len(), t);
    assert!(
        gather.iter().all(|v| *v == sentinel),
        "FrdGatherIndex should be all-sentinel without FieldRegConfig"
    );
}

/// FieldRegRa(d) is not committed; `field_reg_d == 0` is the steady-state
/// expectation since the FR write-slot indicator polys are materialized
/// from bytecode at proof time.
#[test]
fn polynomial_config_has_field_reg_chunk_count() {
    let cfg = PolynomialConfig::new(4, 128, 16, 24);
    assert_eq!(cfg.field_reg_d, 0);
}
