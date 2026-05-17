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

/// Per-cycle bytecode metadata consumed by the FR Twist materializers.
///
/// Mirrors the source-branch `FrCycleBytecode`: the low 4 bits of the
/// instruction's `rs1`/`rs2`/`rd` fields plus boolean flags marking which
/// reads are FR-register reads (as opposed to integer-register reads on
/// FieldMov/FieldSll* bridges).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrCycleBytecode {
    pub frs1: u8,
    pub frs2: u8,
    pub frd: u8,
    pub reads_frs1: bool,
    pub reads_frs2: bool,
}

/// Replay context: bytecode metadata + tracer event stream for one trace.
///
/// `bytecode.len()` must equal `num_cycles`. `events` are sorted by cycle and
/// each cycle has at most one FR event. When `events` is empty (no FR
/// instructions executed), every materializer falls back to the zero shape —
/// the same behavior the Stage 4/5 sumchecks consume for inert programs like
/// muldiv.
#[derive(Clone, Debug)]
pub struct FieldRegReplay {
    pub num_cycles: usize,
    pub bytecode: Vec<FrCycleBytecode>,
    pub events: Vec<FieldRegEvent>,
}

impl FieldRegReplay {
    /// Inert replay: T cycles, zero bytecode, zero events. Materializers
    /// return all-zero buffers. Used as the default for FR-inactive traces.
    pub fn empty(num_cycles: usize) -> Self {
        Self {
            num_cycles,
            bytecode: vec![FrCycleBytecode::default(); num_cycles],
            events: Vec::new(),
        }
    }

    fn k_t(&self) -> usize {
        FIELD_REG_COUNT * self.num_cycles
    }

    /// `K_FR × T` register-file state, row-major by slot then by cycle.
    /// `val(k, t)` = Fr-encoded value of slot `k` at the START of cycle `t`.
    /// Slots start at zero and update to `event.rd_post` after each event.
    pub fn materialize_field_reg_val<F: jolt_field::Field>(&self) -> Vec<F> {
        if self.events.is_empty() {
            return vec![F::zero(); self.k_t()];
        }
        let t = self.num_cycles;
        let mut out = vec![F::zero(); self.k_t()];
        let mut current: [F; FIELD_REG_COUNT] = [F::zero(); FIELD_REG_COUNT];
        let mut events = self.events.iter().peekable();

        for c in 0..t {
            for (k, val) in current.iter().enumerate() {
                out[k * t + c] = *val;
            }
            if let Some(ev) = events.next_if(|ev| ev.cycle as usize == c) {
                if ev.rd_written {
                    let slot = (ev.frd as usize) & 0xF;
                    current[slot] = limbs_to_field::<F>(ev.rd_post.into_limbs());
                }
            }
        }
        out
    }

    /// `K_FR × T` one-hot at `(frs1(t) & 0xF, t)` when bytecode marks the
    /// cycle as reading FR slot `frs1` (i.e., FMUL/FADD/FSUB/FAssertEq/FINV).
    pub fn materialize_frs1_ra<F: jolt_field::Field>(&self) -> Vec<F> {
        if self.events.is_empty() {
            return vec![F::zero(); self.k_t()];
        }
        let t = self.num_cycles;
        let mut out = vec![F::zero(); self.k_t()];
        for (c, bc) in self.bytecode.iter().enumerate().take(t) {
            if bc.reads_frs1 {
                let slot = (bc.frs1 as usize) & 0xF;
                out[slot * t + c] = F::one();
            }
        }
        out
    }

    /// `K_FR × T` one-hot at `(frs2(t) & 0xF, t)` when bytecode marks the
    /// cycle as reading FR slot `frs2` (FMUL/FADD/FSUB/FAssertEq).
    pub fn materialize_frs2_ra<F: jolt_field::Field>(&self) -> Vec<F> {
        if self.events.is_empty() {
            return vec![F::zero(); self.k_t()];
        }
        let t = self.num_cycles;
        let mut out = vec![F::zero(); self.k_t()];
        for (c, bc) in self.bytecode.iter().enumerate().take(t) {
            if bc.reads_frs2 {
                let slot = (bc.frs2 as usize) & 0xF;
                out[slot * t + c] = F::one();
            }
        }
        out
    }

    /// `K_FR × T` one-hot at `(event.frd & 0xF, event.cycle)` for cycles
    /// with a writing FR event. Drawn from the event stream rather than
    /// bytecode so the shape matches what `FrdInc` commits.
    pub fn materialize_frd_wa<F: jolt_field::Field>(&self) -> Vec<F> {
        if self.events.is_empty() {
            return vec![F::zero(); self.k_t()];
        }
        let t = self.num_cycles;
        let mut out = vec![F::zero(); self.k_t()];
        for ev in &self.events {
            if ev.rd_written {
                let slot = (ev.frd as usize) & 0xF;
                out[slot * t + (ev.cycle as usize)] = F::one();
            }
        }
        out
    }

    /// T-element FR write delta: `inc(t) = limbs_to_field(rd_post) - val_pre`
    /// at the write slot. Zero on non-writing cycles (including those with no
    /// event). The running pre-state is computed from the event stream.
    pub fn materialize_frd_inc<F: jolt_field::Field>(&self) -> Vec<F> {
        let t = self.num_cycles;
        if self.events.is_empty() {
            return vec![F::zero(); t];
        }
        let mut out = vec![F::zero(); t];
        let mut current: [F; FIELD_REG_COUNT] = [F::zero(); FIELD_REG_COUNT];
        let mut events = self.events.iter().peekable();
        for (c, slot_out) in out.iter_mut().enumerate().take(t) {
            if let Some(ev) = events.next_if(|ev| ev.cycle as usize == c) {
                if ev.rd_written {
                    let slot = (ev.frd as usize) & 0xF;
                    let post = limbs_to_field::<F>(ev.rd_post.into_limbs());
                    *slot_out = post - current[slot];
                    current[slot] = post;
                }
            }
        }
        out
    }
}

/// Convert a natural-form `[u64; 4]` limb array to an Fr field element:
/// `a[0] + a[1]·2⁶⁴ + a[2]·2¹²⁸ + a[3]·2¹⁹²`. Used by the FR materializers
/// to encode the running FR register-file state.
pub fn limbs_to_field<F: jolt_field::Field>(limbs: [u64; 4]) -> F {
    let lo = F::from_u128((limbs[0] as u128) | ((limbs[1] as u128) << 64));
    let hi = F::from_u128((limbs[2] as u128) | ((limbs[3] as u128) << 64));
    lo + hi.mul_pow_2(128)
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
    fn empty_replay_yields_zero_materializers() {
        use jolt_field::Fr;
        let replay = FieldRegReplay::empty(8);
        let val: Vec<Fr> = replay.materialize_field_reg_val();
        assert_eq!(val.len(), FIELD_REG_COUNT * 8);
        assert!(val.iter().all(|v| *v == Fr::from_u64(0)));
        let inc: Vec<Fr> = replay.materialize_frd_inc();
        assert_eq!(inc.len(), 8);
        assert!(inc.iter().all(|v| *v == Fr::from_u64(0)));
    }

    #[test]
    fn frs1_ra_one_hot_at_read_cycles() {
        use jolt_field::Fr;
        let t = 4;
        let event = FieldRegEvent {
            cycle: 0,
            frs1: 3,
            frs2: 0,
            frd: 7,
            rs1_pre: FrLimbs::ZERO,
            rs2_pre: FrLimbs::ZERO,
            rd_post: FrLimbs::ZERO,
            rd_written: true,
        };
        let bytecode = vec![
            FrCycleBytecode { frs1: 3, frs2: 0, frd: 7, reads_frs1: true, reads_frs2: false },
            FrCycleBytecode::default(),
            FrCycleBytecode::default(),
            FrCycleBytecode::default(),
        ];
        let replay = FieldRegReplay { num_cycles: t, bytecode, events: vec![event] };
        let ra: Vec<Fr> = replay.materialize_frs1_ra();
        assert_eq!(ra.len(), FIELD_REG_COUNT * t);
        // Slot 3, cycle 0 should be 1.
        assert_eq!(ra[3 * t], Fr::from_u64(1));
        // Everything else zero.
        for (i, v) in ra.iter().enumerate() {
            if i == 3 * t {
                continue;
            }
            assert_eq!(*v, Fr::from_u64(0), "non-write slot {i} should be zero");
        }
    }

    #[test]
    fn frd_wa_marks_write_slot_at_event_cycle() {
        use jolt_field::Fr;
        let t = 4;
        let event = FieldRegEvent {
            cycle: 2,
            frs1: 0,
            frs2: 0,
            frd: 9,
            rs1_pre: FrLimbs::ZERO,
            rs2_pre: FrLimbs::ZERO,
            rd_post: FrLimbs::ZERO,
            rd_written: true,
        };
        let replay = FieldRegReplay {
            num_cycles: t,
            bytecode: vec![FrCycleBytecode::default(); t],
            events: vec![event],
        };
        let wa: Vec<Fr> = replay.materialize_frd_wa();
        // Slot 9, cycle 2.
        assert_eq!(wa[9 * t + 2], Fr::from_u64(1));
        for (i, v) in wa.iter().enumerate() {
            if i == 9 * t + 2 {
                continue;
            }
            assert_eq!(*v, Fr::from_u64(0), "non-write slot {i} should be zero");
        }
    }

    #[test]
    fn field_reg_val_tracks_running_state() {
        use jolt_field::Fr;
        let t = 4;
        // Two events: cycle 1 writes slot 5 = 42; cycle 2 writes slot 5 = 99.
        let events = vec![
            FieldRegEvent {
                cycle: 1, frs1: 0, frs2: 0, frd: 5,
                rs1_pre: FrLimbs::ZERO, rs2_pre: FrLimbs::ZERO,
                rd_post: FrLimbs([42, 0, 0, 0]),
                rd_written: true,
            },
            FieldRegEvent {
                cycle: 2, frs1: 0, frs2: 0, frd: 5,
                rs1_pre: FrLimbs::ZERO, rs2_pre: FrLimbs::ZERO,
                rd_post: FrLimbs([99, 0, 0, 0]),
                rd_written: true,
            },
        ];
        let replay = FieldRegReplay {
            num_cycles: t,
            bytecode: vec![FrCycleBytecode::default(); t],
            events,
        };
        let val: Vec<Fr> = replay.materialize_field_reg_val();
        // Slot 5 timeline (pre-cycle values): [0, 0, 42, 99]
        assert_eq!(val[5 * t], Fr::from_u64(0));
        assert_eq!(val[5 * t + 1], Fr::from_u64(0));
        assert_eq!(val[5 * t + 2], Fr::from_u64(42));
        assert_eq!(val[5 * t + 3], Fr::from_u64(99));
        // Other slots remain zero throughout.
        for k in 0..FIELD_REG_COUNT {
            if k == 5 {
                continue;
            }
            for c in 0..t {
                assert_eq!(val[k * t + c], Fr::from_u64(0), "slot {k} cycle {c}");
            }
        }
    }

    #[test]
    fn frd_inc_is_post_minus_pre() {
        use jolt_field::Fr;
        let t = 4;
        let events = vec![
            FieldRegEvent {
                cycle: 1, frs1: 0, frs2: 0, frd: 5,
                rs1_pre: FrLimbs::ZERO, rs2_pre: FrLimbs::ZERO,
                rd_post: FrLimbs([42, 0, 0, 0]),
                rd_written: true,
            },
            FieldRegEvent {
                cycle: 2, frs1: 0, frs2: 0, frd: 5,
                rs1_pre: FrLimbs::ZERO, rs2_pre: FrLimbs::ZERO,
                rd_post: FrLimbs([99, 0, 0, 0]),
                rd_written: true,
            },
        ];
        let replay = FieldRegReplay {
            num_cycles: t,
            bytecode: vec![FrCycleBytecode::default(); t],
            events,
        };
        let inc: Vec<Fr> = replay.materialize_frd_inc();
        // inc[1] = 42 - 0 = 42; inc[2] = 99 - 42 = 57.
        assert_eq!(inc[0], Fr::from_u64(0));
        assert_eq!(inc[1], Fr::from_u64(42));
        assert_eq!(inc[2], Fr::from_u64(57));
        assert_eq!(inc[3], Fr::from_u64(0));
    }

    #[test]
    fn limbs_to_field_assembles_4_limbs_little_endian() {
        use jolt_field::Fr;
        // limbs = [1, 0, 0, 0] → 1
        assert_eq!(limbs_to_field::<Fr>([1, 0, 0, 0]), Fr::from_u64(1));
        // limbs = [0, 1, 0, 0] → 2^64
        let two_64 = Fr::from_u128(1u128 << 64);
        assert_eq!(limbs_to_field::<Fr>([0, 1, 0, 0]), two_64);
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
