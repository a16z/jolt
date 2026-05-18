//! BN254 Fr coprocessor witness types and per-cycle replay.
//!
//! The FR Twist sumcheck operates on a 16-entry field-register file that
//! the tracer emits as a stream of [`FieldRegEvent`]s. This module owns:
//!
//! - [`FrLimbs`]: natural-form 256-bit Fr value (4 × u64 little-endian limbs)
//! - [`FieldRegEvent`]: a single FR-active cycle's read/write payload
//! - [`FrCycleBytecode`]: per-cycle FR read-flag metadata (sized to T)
//! - [`FieldRegReplay`]: bytecode + event stream consumed by Stage 4/5
//!   sumchecks
//! - [`sub_limbs`]: borrow-aware 256-bit subtraction used by FSUB events
//!
//! Stage 4's `SparseFieldRegState` walks the replay events directly and
//! emits sparse cycle-major entries; only [`FieldRegReplay::materialize_frd_inc`]
//! (T-length) is still materialized as a dense vector.

/// Number of 256-bit Fr registers (matches `tracer::emulator::cpu::FIELD_REG_COUNT`).
pub const FIELD_REG_COUNT: usize = 16;

/// Log2 of [`FIELD_REG_COUNT`]. Used for the FR Twist one-hot polynomial domain.
pub const LOG_K_FR: usize = 4;

/// Address-bit mask for FR register indices: `FIELD_REG_COUNT - 1 = 0xF`.
///
/// Centralizes the canonical low-4-bit slot mask that callers apply to
/// raw operand fields before indexing into the FR register file. Stage 4
/// and Stage 5 derive their masks from the runtime `field_reg_count`
/// parameter; this constant is for the call sites that work with the
/// fixed-size representation directly (host materializers, witness
/// helpers).
pub const FIELD_REG_ADDR_MASK: usize = FIELD_REG_COUNT - 1;

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

/// Per-cycle bytecode metadata consumed by the FR Twist materializers.
///
/// Mirrors the source-branch `FrCycleBytecode`: the low 4 bits of the
/// instruction's `rs1`/`rs2`/`rd` fields plus boolean flags marking which
/// reads are FR-register reads (as opposed to integer-register reads on
/// FieldMov/FieldSll* bridges) and whether the instruction writes back to
/// `frd` (false for FieldAssertEq, true for all other FR-active kinds).
///
/// `writes_frd` + `frd` together are the *cryptographic anchor* for the FR
/// write-slot indicator: FR Twist materializers source the slot from
/// `bc.frd` (committed via bytecode preprocessing), not from `event.frd`
/// (uncommitted prover input).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrCycleBytecode {
    pub frs1: u8,
    pub frs2: u8,
    pub frd: u8,
    pub reads_frs1: bool,
    pub reads_frs2: bool,
    pub writes_frd: bool,
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
            let bc = self.bytecode.get(c).copied().unwrap_or_default();
            if let Some(ev) = events.next_if(|ev| ev.cycle as usize == c) {
                // Slot comes from bytecode (committed via preprocessing), not
                // event.frd (uncommitted). Gate on bc.writes_frd so a malicious
                // event claiming rd_written on a non-writing kind is dropped.
                if ev.rd_written && bc.writes_frd {
                    let slot = (bc.frd as usize) & FIELD_REG_ADDR_MASK;
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
    fn empty_replay_yields_zero_frd_inc() {
        use jolt_field::Fr;
        let replay = FieldRegReplay::empty(8);
        let inc: Vec<Fr> = replay.materialize_frd_inc();
        assert_eq!(inc.len(), 8);
        assert!(inc.iter().all(|v| *v == Fr::from_u64(0)));
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
        // Post-C7: the write slot is sourced from `bc.frd`, not `event.frd`.
        // For inc to land at slot 5 on cycles 1 and 2, the bytecode for those
        // cycles must say `writes_frd=true, frd=5`.
        let mut bytecode = vec![FrCycleBytecode::default(); t];
        for cycle in [1usize, 2] {
            bytecode[cycle] = FrCycleBytecode {
                frd: 5,
                writes_frd: true,
                ..FrCycleBytecode::default()
            };
        }
        let replay = FieldRegReplay { num_cycles: t, bytecode, events };
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

}
