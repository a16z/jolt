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
///
/// **Invariant:** all index fields (`frs1`, `frs2`, `frd`) MUST be in
/// `0..=15`. Producers mask once at construction; consumers may rely
/// on the value being in range. The modular pipeline's producer is
/// `<Cycle as CycleRow>::fr_meta` (in jolt-host).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrCycleBytecode {
    /// FR slot index for `frs1` (0..=15). Ignored when `reads_frs1`
    /// is false.
    pub frs1: u8,
    /// FR slot index for `frs2` (0..=15). Ignored when `reads_frs2`
    /// is false.
    pub frs2: u8,
    /// FR slot index for `frd` (0..=15). Ignored when `writes_frd` is
    /// false. Sourced from the cycle's instruction (and thus from
    /// committed bytecode via Spartan), so the FR write-slot indicator
    /// polys inherit a cryptographic anchor.
    pub frd: u8,
    /// True if the cycle's instruction reads `frs1` (i.e. the 2-input FR
    /// ops FMUL/FADD/FSUB/FAssertEq, plus FINV's single-input read).
    /// False for non-FR cycles and the bridge ops (FieldMov/FieldSLL*
    /// which read the integer register file, not FReg).
    pub reads_frs1: bool,
    /// True if the cycle reads `frs2` (FMUL/FADD/FSUB/FAssertEq).
    /// False for FieldInv, all bridge ops, and non-FR cycles.
    pub reads_frs2: bool,
    /// True if the cycle writes `frd`. FMUL/FADD/FSUB/FINV/FMov/FSLL64/128/192
    /// all write; FAssertEq does not (it asserts equality, no register write).
    pub writes_frd: bool,
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
/// Validates the event stream up-front (sorted strictly by cycle, no
/// duplicates, all in-range), then walks the trace applying each event in
/// order. Non-FR cycles get the all-zero default snapshot.
///
/// `bytecode` supplies the per-cycle `frs1`/`frs2`/`reads_frs2` metadata;
/// events drive the write-side state evolution. Stage 5 FieldRegValEvaluation
/// cryptographically enforces `events.slot == bytecode.frd` and a consistent
/// state evolution (any malformed stream causes sumcheck rejection
/// downstream) — but we still validate at the host layer so callers get a
/// clear, immediate error rather than a cryptic deep-prover failure.
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

    // Stream-shape validation. Out-of-order, duplicate, and out-of-range
    // events would otherwise be silently dropped by the per-cycle
    // peekable iteration below, masking host-side bugs as cryptic
    // downstream sumcheck failures.
    for window in events.windows(2) {
        assert!(
            window[0].cycle < window[1].cycle,
            "FieldRegEvent stream not strictly sorted: cycle {} >= cycle {} (no duplicates allowed)",
            window[0].cycle,
            window[1].cycle,
        );
    }
    if let Some(last) = events.last() {
        assert!(
            last.cycle < trace_len,
            "FieldRegEvent at cycle {} is past trace_len {}",
            last.cycle,
            trace_len,
        );
    }

    let mut state: [FrLimbs; 16] = [[0u64; 4]; 16];
    let mut snapshots: Vec<FrCycleData> = vec![FrCycleData::default(); trace_len];
    let mut event_iter = events.iter().peekable();

    for (t, bc) in bytecode.iter().enumerate() {
        let event_for_cycle = event_iter.next_if(|e| e.cycle == t).copied();

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
            // Runtime assert (not `debug_assert!`): a host-side state
            // tracking bug would otherwise produce an inconsistent witness
            // that fails far from the bug site as a cryptic sumcheck error.
            // The witness IS caught downstream by Stage 5 ValEvaluation
            // either way, but failing loud here gives a clear error near
            // the source.
            assert_eq!(
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

/// Number of slots in the BN254 Fr coprocessor register file. Must match
/// `tracer::emulator::cpu::FIELD_REG_COUNT` and the R1CS LOG_K_FR contract.
pub const FIELD_REG_COUNT: usize = 16;

/// Per-cycle BN254 Fr operand values in field form, ready to feed into
/// Stage 3 FieldRegClaimReduction.
///
/// Two layouts:
/// - **Dense** (`sparse_only == false`): `rd_value`/`rs1_value`/`rs2_value`
///   are length `trace_len` (= 1<<log_T).
/// - **Sparse** (`sparse_only == true`): the per-cycle operand values are
///   carried in `sparse_accesses` (one entry per FR-active cycle, sorted
///   strictly by cycle). The dense vectors above are empty. Saves
///   `3 × trace_len × |F|` bytes on FR-less traces and on traces with
///   sparse FR activity (Rule 1 in
///   `specs/bn254-fr-coprocessor.md`).
///
/// Both the Stage 3 ClaimReduction kernel and the Stage 4 ReadWriteChecking
/// kernel honour the same flag (set centrally by
/// [`zero_field_registers_witness`]).
#[derive(Clone, Debug)]
pub struct FieldRegisterColumns<F: jolt_field::Field> {
    pub trace_len: usize,
    pub rd_value: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
    /// Per-cycle FR access entries for the Stage 3 sparse path. Empty when
    /// the dense layout is in use; possibly empty (length 0) for an
    /// FR-less trace under the sparse layout.
    pub sparse_accesses: Vec<Stage3FieldRegisterAccessOwned<F>>,
    /// When `true`, the kernel MUST take the sparse path.
    pub sparse_only: bool,
}

/// Owned mirror of the kernel-side `Stage3FieldRegisterAccess`. The host
/// builds these once per proof; the prover wiring borrows them as
/// `&[stage3::Stage3FieldRegisterAccess]` into the kernel input.
#[derive(Clone, Copy, Debug)]
pub struct Stage3FieldRegisterAccessOwned<F: jolt_field::Field> {
    pub cycle: usize,
    pub rd_value: F,
    pub rs1_value: F,
    pub rs2_value: F,
}

/// Full FR witness for Stages 4 + 5 over a `field_register_count`-slot file.
///
/// Layout: dense polynomials in **address-major** order, i.e.
/// `slice[slot * trace_len + cycle]`. This matches the address-major
/// convention used by the integer registers Twist in the modular stack.
///
/// - `field_registers_val[slot * T + t]` = value at `slot` at the START of
///   cycle `t` (pre-write state for that cycle)
/// - `field_rs1_ra[slot * T + t]` = 1 iff cycle `t` reads `slot` as frs1
///   (only when `bytecode[t].reads_frs1` is true)
/// - `field_rs2_ra[slot * T + t]` = 1 iff cycle `t` reads `slot` as frs2
/// - `field_rd_wa[slot * T + t]` = 1 iff cycle `t` writes `slot` as frd
/// - `field_rd_inc[t]` = (new − old) mod p when cycle `t` writes,
///   else 0 — the COMMITTED delta polynomial (audit C11)
///
/// When `sparse_only` is `true`, the dense `field_registers_val` /
/// `field_rs1_ra` / `field_rs2_ra` / `field_rd_wa` vectors are guaranteed
/// to be empty — the Stage 4 sparse FR path consumes `field_rd_inc` and
/// the prover-side access list directly, so the dense `K · T` allocations
/// are skipped entirely (Rule 1 in `specs/bn254-fr-coprocessor.md`).
#[derive(Clone, Debug)]
pub struct FieldRegistersWitness<F: jolt_field::Field> {
    pub field_register_count: usize,
    pub trace_len: usize,
    pub field_registers_val: Vec<F>,
    pub field_rs1_ra: Vec<F>,
    pub field_rs2_ra: Vec<F>,
    pub field_rd_wa: Vec<F>,
    pub field_rd_inc: Vec<F>,
    /// Per-cycle FR write address (`Some(slot)` when the cycle writes
    /// `field_regs[slot]`, `None` otherwise). Always populated, length
    /// `trace_len`. The Stage 5 FR ValEvaluation sparse path consumes this
    /// instead of the `K · T` `field_rd_wa` dense vector — see Rule 1 in
    /// `specs/bn254-fr-coprocessor.md`.
    pub field_rd_write_addresses: Vec<Option<usize>>,
    /// When `true`, the dense Twist vectors above are empty by construction
    /// and the kernel MUST take the sparse path. Set by
    /// [`zero_field_registers_witness`] and by sparse builders that
    /// produce only the per-cycle delta poly.
    pub sparse_only: bool,
}

/// Construct a low-allocation FR witness pair for a trace with no FR
/// events. Used by `prove_jolt_with_witness_inputs` when the caller didn't
/// provide FR data — Bolt's Stage 3/4/5 FR plans always reference the
/// relations, so the kernel needs *some* witness even on FR-less traces.
///
/// Rule 1: the Stage 4 Twist tables for Val/Rs1Ra/Rs2Ra (3 × 16 × T) and
/// the Stage 5 dense `field_rd_wa` (16 × T) are *not* allocated — the FR
/// kernels consume the per-cycle access list / write-slot indices and the
/// `field_rd_inc` trace-length delta poly directly. Net savings at
/// `T = 2^16`: ~134 MB (the full 4 × 16 × T dense set).
pub fn zero_field_registers_witness<F: jolt_field::Field>(
    trace_len: usize,
) -> (FieldRegisterColumns<F>, FieldRegistersWitness<F>) {
    let n = FIELD_REG_COUNT;
    (
        FieldRegisterColumns {
            trace_len,
            rd_value: Vec::new(),
            rs1_value: Vec::new(),
            rs2_value: Vec::new(),
            sparse_accesses: Vec::new(),
            sparse_only: true,
        },
        FieldRegistersWitness {
            field_register_count: n,
            trace_len,
            field_registers_val: Vec::new(),
            field_rs1_ra: Vec::new(),
            field_rs2_ra: Vec::new(),
            field_rd_wa: Vec::new(),
            field_rd_inc: vec![F::zero(); trace_len],
            field_rd_write_addresses: vec![None; trace_len],
            sparse_only: true,
        },
    )
}

/// Materialize the full FR witness for Stages 3-5 from a replayed FR event
/// stream. The caller has already invoked `replay_field_regs` to produce the
/// per-cycle `FrCycleData` snapshots; this expands them into the dense
/// polynomials the FR Twist kernels consume.
///
/// The single linear pass over snapshots is O(trace_len) and fills the
/// 16 × T tables address-major. For traces with sparse FR activity
/// (typical), most slots stay zero.
pub fn field_registers_witness<F: jolt_field::Field>(
    bytecode: &[FrCycleBytecode],
    snapshots: &[FrCycleData],
    events: &[FieldRegEvent],
) -> (FieldRegisterColumns<F>, FieldRegistersWitness<F>) {
    assert_eq!(
        bytecode.len(),
        snapshots.len(),
        "bytecode/snapshots length mismatch"
    );
    let trace_len = snapshots.len();
    let n = FIELD_REG_COUNT;
    let total = n * trace_len;

    // Stage 3 inputs: per-cycle field-element values (rd, rs1, rs2). These
    // come from the snapshot's limb values converted to field elements.
    let mut rd_value = vec![F::zero(); trace_len];
    let mut rs1_value = vec![F::zero(); trace_len];
    let mut rs2_value = vec![F::zero(); trace_len];

    // Stage 4 polys: 16xT address-major.
    let mut field_registers_val = vec![F::zero(); total];
    let mut field_rs1_ra = vec![F::zero(); total];
    let mut field_rs2_ra = vec![F::zero(); total];
    let mut field_rd_wa = vec![F::zero(); total];
    let mut field_rd_write_addresses: Vec<Option<usize>> = vec![None; trace_len];

    // Stage 5 input: trace-length committed delta poly via the helper that
    // computes (new - old) as a FIELD subtraction (mod p), not limb-wise.
    let field_rd_inc = field_reg_inc_polynomial::<F>(events, trace_len);

    // Track running state for the per-cycle Val column. The R1CS contract
    // wants Val[slot, t] = state at slot at the START of cycle t (i.e. the
    // value cycle t reads).
    let mut state: [FrLimbs; FIELD_REG_COUNT] = [[0u64; 4]; FIELD_REG_COUNT];

    for (t, (bc, snap)) in bytecode.iter().zip(snapshots.iter()).enumerate() {
        // 1. Per-cycle Stage 3 operand values (field elements from limbs).
        rd_value[t] = limbs_to_field::<F>(snap.rd_val);
        rs1_value[t] = limbs_to_field::<F>(snap.rs1_val);
        rs2_value[t] = limbs_to_field::<F>(snap.rs2_val);

        // 2. Stage 4 Val poly: snapshot of pre-cycle state across all slots.
        for slot in 0..n {
            field_registers_val[slot * trace_len + t] = limbs_to_field::<F>(state[slot]);
        }

        // 3. Stage 4 one-hots from bytecode (audit C7: bytecode-anchored).
        if bc.reads_frs1 {
            let slot = (bc.frs1 as usize) & 0xF;
            field_rs1_ra[slot * trace_len + t] = F::one();
        }
        if bc.reads_frs2 {
            let slot = (bc.frs2 as usize) & 0xF;
            field_rs2_ra[slot * trace_len + t] = F::one();
        }
        if bc.writes_frd {
            let slot = (bc.frd as usize) & 0xF;
            field_rd_wa[slot * trace_len + t] = F::one();
            field_rd_write_addresses[t] = Some(slot);
        }

        // 4. Apply this cycle's write to the running state (for the NEXT
        //    cycle's Val column). FieldAssertEq's no-op write (new == old)
        //    keeps state unchanged.
        if let Some(write_slot) = snap.write_slot {
            state[(write_slot as usize) & 0xF] = snap.rd_val;
        }
    }

    (
        FieldRegisterColumns {
            trace_len,
            rd_value,
            rs1_value,
            rs2_value,
            sparse_accesses: Vec::new(),
            sparse_only: false,
        },
        FieldRegistersWitness {
            field_register_count: n,
            trace_len,
            field_registers_val,
            field_rs1_ra,
            field_rs2_ra,
            field_rd_wa,
            field_rd_inc,
            field_rd_write_addresses,
            sparse_only: false,
        },
    )
}

/// Materialize the per-cycle `FieldRegInc` dense polynomial directly from a
/// `FieldRegEvent` stream. Each event contributes `new - old` (as field
/// elements, mod p) at its cycle index; non-event cycles stay zero.
///
/// Required for any caller that builds a `Polynomials<F>` for a program with
/// non-empty FR events: the default `Polynomials::push` path leaves
/// `FieldRegInc` dense at all-zero (because the per-cycle delta is 256-bit
/// and doesn't fit `CycleInput::dense`'s `i128`). Prover-side callers must
/// call `polys.insert(PolynomialId::FieldRegInc, field_reg_inc_polynomial(events, T))`
/// after `polys.finish()`. Forgetting this for an FR-active program silently
/// commits all-zero deltas while `FieldRegRa` is non-zero — soundness gap
/// (FR Twist Stage 5 ValEvaluation only satisfies `0 = 0`).
///
/// Computes `new - old` as a field subtraction (mod p), NOT the limb-level
/// `sub_limbs` (which is mod 2^256 and would alias on underflow).
pub fn field_reg_inc_polynomial<F: jolt_field::Field>(
    events: &[FieldRegEvent],
    trace_length: usize,
) -> Vec<F> {
    let mut out = vec![F::zero(); trace_length];
    for ev in events {
        if ev.cycle < trace_length {
            out[ev.cycle] = limbs_to_field::<F>(ev.new) - limbs_to_field::<F>(ev.old);
        }
    }
    out
}

/// Convert a natural-form `[u64; 4]` limb array to an Fr field element:
/// `a[0] + a[1]·2⁶⁴ + a[2]·2¹²⁸ + a[3]·2¹⁹²`. Mirrors the private helper
/// in `derived.rs`; kept here so `field_reg_inc_polynomial` is self-contained.
fn limbs_to_field<F: jolt_field::Field>(limbs: FrLimbs) -> F {
    let lo = F::from_u128((limbs[0] as u128) | ((limbs[1] as u128) << 64));
    let hi = F::from_u128((limbs[2] as u128) | ((limbs[3] as u128) << 64));
    lo + hi * F::one().mul_pow_2(128)
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
            frd: 3,
            reads_frs1: false,
            reads_frs2: false,
            writes_frd: true,
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
            FrCycleBytecode {
                frd: 1,
                writes_frd: true,
                ..Default::default()
            }, // FieldMov writes f1
            FrCycleBytecode {
                frd: 2,
                writes_frd: true,
                ..Default::default()
            }, // FieldMov writes f2
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
            FrCycleBytecode {
                frd: 1,
                writes_frd: true,
                ..Default::default()
            }, // FieldMov writes f1
            // FieldAssertEq: reads frs1+frs2, no write.
            FrCycleBytecode {
                frs1: 1,
                frs2: 1,
                frd: 0,
                reads_frs1: true,
                reads_frs2: true,
                writes_frd: false,
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

    #[test]
    #[should_panic(expected = "not strictly sorted")]
    fn out_of_order_events_are_rejected() {
        let bytecode = vec![FrCycleBytecode::default(); 4];
        // cycle 2 before cycle 1 → out of order.
        let events = vec![
            FieldRegEvent {
                cycle: 2,
                slot: 0,
                old: [0; 4],
                new: [1, 0, 0, 0],
            },
            FieldRegEvent {
                cycle: 1,
                slot: 0,
                old: [0; 4],
                new: [2, 0, 0, 0],
            },
        ];
        let _ = replay_field_regs(4, &bytecode, &events);
    }

    #[test]
    #[should_panic(expected = "not strictly sorted")]
    fn duplicate_cycle_events_are_rejected() {
        let bytecode = vec![FrCycleBytecode::default(); 2];
        let events = vec![
            FieldRegEvent {
                cycle: 0,
                slot: 0,
                old: [0; 4],
                new: [1, 0, 0, 0],
            },
            FieldRegEvent {
                cycle: 0, // duplicate
                slot: 0,
                old: [0; 4],
                new: [2, 0, 0, 0],
            },
        ];
        let _ = replay_field_regs(2, &bytecode, &events);
    }

    #[test]
    fn field_registers_witness_address_major_layout() {
        use jolt_field::{Field, Fr};
        use num_traits::{One, Zero};
        // 3-cycle trace:
        //   t=0: FieldMov writes f1 = 10
        //   t=1: FieldMov writes f2 = 20
        //   t=2: FieldAdd f1 + f2 → f3 (= 30)
        let bytecode = vec![
            FrCycleBytecode {
                frd: 1,
                writes_frd: true,
                ..Default::default()
            },
            FrCycleBytecode {
                frd: 2,
                writes_frd: true,
                ..Default::default()
            },
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
        let snaps = replay_field_regs(3, &bytecode, &events);
        let (columns, witness) = field_registers_witness::<Fr>(&bytecode, &snaps, &events);

        // Stage 3 columns track per-cycle limb→field values.
        assert_eq!(columns.rs1_value[2], Fr::from_u64(10));
        assert_eq!(columns.rs2_value[2], Fr::from_u64(20));
        assert_eq!(columns.rd_value[2], Fr::from_u64(30));

        // FieldRdWa is one-hot at the write slot per cycle.
        let t = witness.trace_len;
        assert!(witness.field_rd_wa[1 * t + 0].is_one(), "f1 written at t=0");
        assert!(witness.field_rd_wa[2 * t + 1].is_one(), "f2 written at t=1");
        assert!(witness.field_rd_wa[3 * t + 2].is_one(), "f3 written at t=2");
        // No other writes.
        let total_writes: usize = witness.field_rd_wa.iter().filter(|v| !v.is_zero()).count();
        assert_eq!(total_writes, 3);

        // FieldRs1Ra: t=2 reads f1 as frs1; nobody else reads frs1.
        assert!(witness.field_rs1_ra[1 * t + 2].is_one());
        assert_eq!(
            witness.field_rs1_ra.iter().filter(|v| !v.is_zero()).count(),
            1,
        );

        // FieldRs2Ra: t=2 reads f2 as frs2.
        assert!(witness.field_rs2_ra[2 * t + 2].is_one());

        // FieldRegistersVal: at t=2, slot 1 should hold the pre-cycle value
        // 10 (written at t=0) and slot 2 should hold 20 (written at t=1).
        // Slot 3 should be 0 at t=2 (it's the write target this cycle, but
        // Val captures the START-of-cycle state).
        assert_eq!(witness.field_registers_val[1 * t + 2], Fr::from_u64(10));
        assert_eq!(witness.field_registers_val[2 * t + 2], Fr::from_u64(20));
        assert_eq!(witness.field_registers_val[3 * t + 2], Fr::zero());
        // At t=1, slot 1 should already hold 10.
        assert_eq!(witness.field_registers_val[1 * t + 1], Fr::from_u64(10));

        // FieldRdInc: per-cycle field-element delta (new - old) mod p.
        assert_eq!(witness.field_rd_inc[0], Fr::from_u64(10));
        assert_eq!(witness.field_rd_inc[1], Fr::from_u64(20));
        assert_eq!(witness.field_rd_inc[2], Fr::from_u64(30));
    }

    #[test]
    #[should_panic(expected = "past trace_len")]
    fn out_of_range_event_is_rejected() {
        let bytecode = vec![FrCycleBytecode::default(); 2];
        let events = vec![FieldRegEvent {
            cycle: 5, // > trace_len = 2
            slot: 0,
            old: [0; 4],
            new: [1, 0, 0, 0],
        }];
        let _ = replay_field_regs(2, &bytecode, &events);
    }
}
