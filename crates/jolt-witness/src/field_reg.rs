//! BN254 Fr coprocessor witness types and per-cycle replay.
//!
//! The FR Twist sumcheck operates on a 16-entry field-register file that
//! the tracer emits as a stream of [`FieldRegEvent`]s. This module owns:
//!
//! - [`FrLimbs`]: natural-form 256-bit Fr value (4 × u64 little-endian limbs)
//! - [`FrCycleData`]: per-cycle FR operand snapshot (rs1_pre, rs2_pre, rd_post)
//! - [`replay_field_regs`]: walks the tracer event stream and produces a
//!   per-cycle data table sized to the (padded) trace length
//! - [`sub_limbs`]: borrow-aware 256-bit subtraction used by FSUB events
//!
//! Phase 3 scope: types + replay scaffolding. Until guest programs actually
//! execute FR instructions, the event stream is empty and replay produces
//! all-zero data — keeping the new R1CS rows (Phase 2) trivially satisfied.
//! Phase 4 (FR Twist sumchecks) consumes the populated `FrCycleData`.

/// Number of 256-bit Fr registers (matches `tracer::emulator::cpu::FIELD_REG_COUNT`).
pub const FIELD_REG_COUNT: usize = 16;

/// Log2 of [`FIELD_REG_COUNT`]. Used for the FR Twist one-hot polynomial domain.
pub const LOG_K_FR: usize = 4;

/// Natural-form 256-bit value stored as 4 little-endian u64 limbs.
///
/// Identical wire shape to `tracer::emulator::cpu::FieldRegEvent::value`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FrLimbs(pub [u64; 4]);

impl FrLimbs {
    pub const ZERO: Self = Self([0; 4]);

    #[inline]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    #[inline]
    pub const fn into_limbs(self) -> [u64; 4] {
        self.0
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == [0u64; 4]
    }
}

impl From<[u64; 4]> for FrLimbs {
    fn from(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }
}

impl From<FrLimbs> for [u64; 4] {
    fn from(limbs: FrLimbs) -> Self {
        limbs.0
    }
}

/// One FR cycle's operand snapshot: pre-values of the two read registers and
/// the post-value of the write register. Phase 4's FieldRegRW sumcheck binds
/// the R1CS `V_FIELD_RS1/RS2/RD_WRITE_VALUE` slots to these values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrCycleData {
    pub rs1_pre: FrLimbs,
    pub rs2_pre: FrLimbs,
    pub rd_post: FrLimbs,
    /// Index of the destination FR register in `0..FIELD_REG_COUNT`. The
    /// FR Twist's `Wa` one-hot polynomial reads this value.
    pub rd_index: u8,
    /// Whether this cycle wrote a field register (every FR op except
    /// `FieldAssertEq`). Drives the `IsFieldWrite` mask in Phase 4.
    pub rd_written: bool,
}

/// Borrow-aware 256-bit subtraction over `FrLimbs`. Wraps modulo 2^256;
/// callers reduce the result against the BN254 Fr prime themselves.
#[inline]
pub fn sub_limbs(a: FrLimbs, b: FrLimbs) -> FrLimbs {
    let mut out = [0u64; 4];
    let mut borrow: u64 = 0;
    for ((a_i, b_i), out_i) in a.0.iter().zip(b.0.iter()).zip(out.iter_mut()) {
        let (d1, b1) = a_i.overflowing_sub(*b_i);
        let (d2, b2) = d1.overflowing_sub(borrow);
        *out_i = d2;
        borrow = u64::from(b1 || b2);
    }
    FrLimbs(out)
}

/// A single FR-coprocessor cycle event, mirroring `tracer::emulator::cpu::FieldRegEvent`
/// without dragging the tracer dep into this crate. The host-side replay
/// constructs these from the tracer's event stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldRegEvent {
    pub cycle: u64,
    pub frs1: u8,
    pub frs2: u8,
    pub frd: u8,
    pub rs1_pre: FrLimbs,
    pub rs2_pre: FrLimbs,
    pub rd_post: FrLimbs,
    pub rd_written: bool,
}

/// Replays the tracer's FR event stream into a per-cycle data table sized
/// to `trace_len`. Non-FR cycles are filled with [`FrCycleData::default()`]
/// (all-zero, `rd_index = 0`, `rd_written = false`), which keeps the
/// Phase-2 R1CS rows trivially satisfied.
///
/// Panics if any event has a cycle index ≥ `trace_len`, since that would
/// corrupt the Fr Twist sumcheck domain.
pub fn replay_field_regs(events: Vec<FieldRegEvent>, trace_len: usize) -> Vec<FrCycleData> {
    let mut table = vec![FrCycleData::default(); trace_len];
    for event in events {
        // 32-bit hosts only — `event.cycle` is bounded by `trace_len: usize`,
        // so the conversion never fails on the supported targets.
        let cycle = event.cycle as usize;
        assert!(
            cycle < trace_len,
            "FR event at cycle {cycle} exceeds trace_len {trace_len}"
        );
        table[cycle] = FrCycleData {
            rs1_pre: event.rs1_pre,
            rs2_pre: event.rs2_pre,
            rd_post: event.rd_post,
            rd_index: event.frd,
            rd_written: event.rd_written,
        };
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fr_limbs_default_is_zero() {
        assert_eq!(FrLimbs::default(), FrLimbs::ZERO);
        assert!(FrLimbs::default().is_zero());
    }

    #[test]
    fn fr_limbs_roundtrip_through_u64_array() {
        let limbs = [1u64, 2, 3, 4];
        let v = FrLimbs::from(limbs);
        assert_eq!(v.into_limbs(), limbs);
        let back: [u64; 4] = v.into();
        assert_eq!(back, limbs);
    }

    #[test]
    fn sub_limbs_borrows_across_words() {
        // 2^64 − 1 = (u64::MAX, 0, 0, 0)
        let a = FrLimbs([0, 1, 0, 0]);
        let b = FrLimbs([1, 0, 0, 0]);
        let diff = sub_limbs(a, b);
        assert_eq!(diff, FrLimbs([u64::MAX, 0, 0, 0]));
    }

    #[test]
    fn sub_limbs_zero_minus_zero_is_zero() {
        assert_eq!(sub_limbs(FrLimbs::ZERO, FrLimbs::ZERO), FrLimbs::ZERO);
    }

    #[test]
    fn replay_zero_events_yields_all_default_rows() {
        let table = replay_field_regs(Vec::new(), 8);
        assert_eq!(table.len(), 8);
        assert!(table.iter().all(|c| *c == FrCycleData::default()));
    }

    #[test]
    fn replay_writes_event_to_indexed_cycle() {
        let event = FieldRegEvent {
            cycle: 3,
            frs1: 1,
            frs2: 2,
            frd: 5,
            rs1_pre: FrLimbs([10, 0, 0, 0]),
            rs2_pre: FrLimbs([20, 0, 0, 0]),
            rd_post: FrLimbs([30, 0, 0, 0]),
            rd_written: true,
        };
        let table = replay_field_regs(vec![event], 8);
        assert_eq!(table[3].rs1_pre, FrLimbs([10, 0, 0, 0]));
        assert_eq!(table[3].rs2_pre, FrLimbs([20, 0, 0, 0]));
        assert_eq!(table[3].rd_post, FrLimbs([30, 0, 0, 0]));
        assert_eq!(table[3].rd_index, 5);
        assert!(table[3].rd_written);
        // All other cycles stay default.
        for (idx, cycle) in table.iter().enumerate() {
            if idx != 3 {
                assert_eq!(*cycle, FrCycleData::default());
            }
        }
    }
}
