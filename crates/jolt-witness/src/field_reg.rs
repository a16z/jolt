//! Per-cycle BN254 Fr coprocessor state materialization.
//!
//! Replays a [`FieldRegEvent`] stream against the bytecode's per-cycle
//! `rs1`/`rs2`/`rd` fields to produce [`FrCycleData`] — the three FR operand
//! values (`rs1_val`, `rs2_val`, `rd_val`) that populate R1CS slots
//! `V_FIELD_RS1/RS2/RD_VALUE`, plus the inc delta that populates the
//! committed `FieldRegInc` polynomial, plus the write-slot index for
//! `FieldRegRa` one-hot derivation.
//!
//! The replay runs in one linear pass over the trace, maintaining the
//! 16 × 256-bit `field_regs` state internally. On non-FR cycles the snapshot
//! is all-zero (including `write_slot = None` → FR one-hot is the zero
//! vector).

/// Natural-form BN254 Fr limb representation: little-endian `u64` limbs.
///
/// Mirrors `tracer::emulator::cpu::FieldRegEvent::old`/`new`. Kept as a type
/// alias rather than a newtype so arithmetic and array conversions stay
/// zero-cost at the witness layer.
pub type FrLimbs = [u64; 4];

/// Per-cycle BN254 Fr coprocessor access snapshot.
///
/// Aligned 1:1 with the trace: `snapshots[t]` is the FR access on cycle `t`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrCycleData {
    /// `field_regs[frs1(t)]` at the start of cycle `t`.
    pub rs1_val: FrLimbs,
    /// `field_regs[frs2(t)]` at the start of cycle `t`. Zero when the
    /// instruction doesn't read `frs2` (FieldMov / FieldSLL*).
    pub rs2_val: FrLimbs,
    /// `field_regs[frd(t)]` at the END of cycle `t`, i.e. the written
    /// value. For `FieldAssertEq` (which emits a no-op event at `frs1`)
    /// this equals `rs1_val` so the bridge rows see the correct value.
    pub rd_val: FrLimbs,
    /// Write delta `new − old` at `write_slot`. Zero limbs when the cycle
    /// is not a writing FR op (i.e. `write_slot = None`).
    pub inc: FrLimbs,
    /// Slot written on this cycle (0..=15). `None` on non-FR cycles.
    pub write_slot: Option<u8>,
}

/// Input snapshot of a single cycle's bytecode fields as consumed by the
/// FR replay. The caller populates this from the bytecode preprocessing
/// or directly from the tracer's per-cycle instruction decoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrCycleBytecode {
    /// Low 4 bits of the 5-bit `rs1` field (0..=15). Ignored when
    /// `reads_frs1` is false.
    pub frs1: u8,
    /// Low 4 bits of `rs2`. Ignored when `reads_frs2` is false.
    pub frs2: u8,
    /// True if the cycle's instruction reads `frs1` (i.e. the 2-input FR
    /// ops FMUL/FADD/FSUB/FAssertEq, plus FINV's single-input read).
    /// False for non-FR cycles and the bridge ops (FieldMov/FieldSLL*
    /// which read the integer register file, not FReg).
    pub reads_frs1: bool,
    /// True if the cycle reads `frs2` (FMUL/FADD/FSUB/FAssertEq).
    /// False for FieldInv, all bridge ops, and non-FR cycles.
    pub reads_frs2: bool,
}

/// A `FieldRegEvent` as emitted by the tracer. Duplicated here rather than
/// imported directly to avoid pulling the full `tracer` crate into
/// `jolt-witness`; the fields are identical.
#[derive(Clone, Copy, Debug)]
pub struct FieldRegEvent {
    pub cycle: usize,
    pub slot: u8,
    pub old: FrLimbs,
    pub new: FrLimbs,
}

/// Replays the FieldRegEvent stream to produce per-cycle snapshots.
///
/// Assumes `events` are sorted by `cycle` (the tracer emits them in order)
/// and that each cycle has at most one event. Non-FR cycles get the all-zero
/// default snapshot.
///
/// `bytecode` supplies the per-cycle `frs1`/`frs2`/`reads_frs2` metadata;
/// events drive the write-side state evolution.
pub fn replay_field_regs(
    trace_len: usize,
    bytecode: &[FrCycleBytecode],
    events: &[FieldRegEvent],
) -> Vec<FrCycleData> {
    assert_eq!(
        bytecode.len(),
        trace_len,
        "bytecode snapshot length must match trace length"
    );

    let mut state: [FrLimbs; 16] = [[0u64; 4]; 16];
    let mut snapshots: Vec<FrCycleData> = vec![FrCycleData::default(); trace_len];
    let mut event_iter = events.iter().peekable();

    for (t, bc) in bytecode.iter().enumerate() {
        let event_for_cycle = match event_iter.peek() {
            Some(e) if e.cycle == t => Some(*event_iter.next().unwrap()),
            _ => None,
        };

        let rs1_val = if bc.reads_frs1 {
            state[bc.frs1 as usize & 0xF]
        } else {
            [0u64; 4]
        };
        let rs2_val = if bc.reads_frs2 {
            state[bc.frs2 as usize & 0xF]
        } else {
            [0u64; 4]
        };

        if let Some(ev) = event_for_cycle {
            let slot = ev.slot as usize & 0xF;
            debug_assert_eq!(
                state[slot], ev.old,
                "FieldRegEvent.old disagrees with replayed state at cycle {t}, slot {slot}"
            );
            state[slot] = ev.new;

            let inc = sub_limbs(ev.new, ev.old);
            // `rd_val` is the value AT slot after the write. For FieldOp
            // writes this equals `ev.new`. For FieldAssertEq (which emits
            // a no-op write with `slot = frs1, new = old`), this still
            // equals the slot's (unchanged) value — consistent with what
            // the R1CS row expects.
            let rd_val = ev.new;

            snapshots[t] = FrCycleData {
                rs1_val,
                rs2_val,
                rd_val,
                inc,
                write_slot: Some(ev.slot & 0xF),
            };
        } else {
            snapshots[t] = FrCycleData {
                rs1_val,
                rs2_val,
                rd_val: [0u64; 4],
                inc: [0u64; 4],
                write_slot: None,
            };
        }
    }

    snapshots
}

/// Signed subtraction of two BN254 Fr naturally-represented limb arrays. The
/// prover needs `new - old` to commit into `FieldRegInc`; since values are
/// already reduced mod p, the caller interprets negative results modulo p
/// at field-materialization time. Here we return the bitwise
/// little-endian borrow-propagated difference (which is the correct value
/// mod 2^256; reducing mod p is a downstream concern handled by the field
/// element `from_natural_limbs` conversion).
fn sub_limbs(a: FrLimbs, b: FrLimbs) -> FrLimbs {
    let mut result = [0u64; 4];
    let mut borrow: u128 = 0;
    for i in 0..4 {
        let a_i = a[i] as u128;
        let b_i = b[i] as u128 + borrow;
        if a_i >= b_i {
            result[i] = (a_i - b_i) as u64;
            borrow = 0;
        } else {
            result[i] = ((1u128 << 64) + a_i - b_i) as u64;
            borrow = 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_noop_cycles_produce_zero_snapshots() {
        let bytecode = vec![FrCycleBytecode::default(); 4];
        let events: Vec<FieldRegEvent> = vec![];
        let snaps = replay_field_regs(4, &bytecode, &events);
        for snap in snaps {
            assert_eq!(snap, FrCycleData::default());
        }
    }

    #[test]
    fn field_mov_updates_state_and_populates_rd_val() {
        // Cycle 0: FieldMov x5 → f3. frs1 is irrelevant for bridge ops; frd=3.
        let bytecode = vec![FrCycleBytecode {
            frs1: 0,
            frs2: 0,
            reads_frs1: false,
            reads_frs2: false,
        }];
        let events = vec![FieldRegEvent {
            cycle: 0,
            slot: 3,
            old: [0, 0, 0, 0],
            new: [42, 0, 0, 0],
        }];
        let snaps = replay_field_regs(1, &bytecode, &events);
        assert_eq!(snaps[0].rd_val, [42, 0, 0, 0]);
        assert_eq!(snaps[0].inc, [42, 0, 0, 0]);
        assert_eq!(snaps[0].write_slot, Some(3));
    }

    #[test]
    fn field_add_reads_both_operands_from_running_state() {
        // Cycle 0: set f1 = 10. Cycle 1: set f2 = 20. Cycle 2: FieldAdd f1+f2 → f3.
        let bytecode = vec![
            FrCycleBytecode::default(), // FieldMov writes f1
            FrCycleBytecode::default(), // FieldMov writes f2
            FrCycleBytecode {
                frs1: 1,
                frs2: 2,
                reads_frs1: true,
                reads_frs2: true,
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
        let snaps = replay_field_regs(3, &bytecode, &events);

        // Cycle 2: FieldAdd sees f1 and f2 pre-execution.
        assert_eq!(snaps[2].rs1_val, [10, 0, 0, 0]);
        assert_eq!(snaps[2].rs2_val, [20, 0, 0, 0]);
        assert_eq!(snaps[2].rd_val, [30, 0, 0, 0]);
        assert_eq!(snaps[2].write_slot, Some(3));
    }

    #[test]
    fn field_assert_eq_noop_write_preserves_state() {
        // Cycle 0: FieldMov writes f1 = 7. Cycle 1: FieldAssertEq f1 == f1.
        // FieldAssertEq emits slot=frs1, old=new=current value.
        let bytecode = vec![
            FrCycleBytecode::default(),
            FrCycleBytecode {
                frs1: 1,
                frs2: 1,
                reads_frs1: true,
                reads_frs2: true,
            },
        ];
        let events = vec![
            FieldRegEvent {
                cycle: 0,
                slot: 1,
                old: [0; 4],
                new: [7, 0, 0, 0],
            },
            FieldRegEvent {
                cycle: 1,
                slot: 1,
                old: [7, 0, 0, 0],
                new: [7, 0, 0, 0],
            },
        ];
        let snaps = replay_field_regs(2, &bytecode, &events);
        assert_eq!(snaps[1].rs1_val, [7, 0, 0, 0]);
        assert_eq!(snaps[1].rs2_val, [7, 0, 0, 0]);
        assert_eq!(snaps[1].rd_val, [7, 0, 0, 0]);
        assert_eq!(snaps[1].inc, [0; 4]); // no-op write
    }

    #[test]
    #[should_panic(expected = "FieldRegEvent.old disagrees")]
    fn stale_event_old_state_is_rejected() {
        let bytecode = vec![FrCycleBytecode::default()];
        let events = vec![FieldRegEvent {
            cycle: 0,
            slot: 0,
            // Replayed state is [0;4]; this event claims old=[1;4].
            old: [1, 1, 1, 1],
            new: [2, 2, 2, 2],
        }];
        let _ = replay_field_regs(1, &bytecode, &events);
    }
}
