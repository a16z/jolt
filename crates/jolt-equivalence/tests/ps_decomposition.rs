//! Incremental equivalence tests for prefix/suffix data decomposition.
//!
//! Each test compares one method of the PrefixSuffixEvaluator against a
//! data-driven replacement using values from InstanceConfig. Tests are
//! designed so that as each decomposition step lands, the corresponding
//! test starts passing.
//!
//! Validation loop:
//!   cargo nextest run -p jolt-equivalence -E 'binary(ps_decomposition)' --cargo-quiet
#![allow(non_snake_case, clippy::print_stderr)]

use jolt_compiler::module::{
    BilinearExpr, CheckpointAction, CheckpointRule, CombineEntry, Comparison, DefaultVal, IntBitOp,
    PrefixMleFormula, PrefixMleRule, RemainingTest, RoundGuard, WeightFn,
};
use jolt_host::prefix_suffix_evaluator::JoltPrefixSuffixEvaluator;
use jolt_instructions::tables::prefixes::{Prefixes, ALL_PREFIXES};
use jolt_instructions::tables::suffixes::Suffixes;
use jolt_instructions::{LookupBits, LookupTableKind, LookupTables};
use jolt_zkvm::checkpoint_eval::{compute_read_checking_from_lowered, eval_checkpoint_rule};
use jolt_zkvm::runtime::prefix_suffix::LookupTraceData;

use jolt_field::Field;
use num_traits::{One, Zero};
type F = jolt_field::Fr;

const XLEN: usize = 64;

// ─── SuffixOp: data-driven suffix evaluation ────────────────────

/// Describes a suffix computation as a pure data operation.
/// These will eventually live in InstanceConfig, populated by the compiler.
#[derive(Clone, Copy, Debug)]
enum SuffixOp {
    One,
    And,
    Or,
    Xor,
    Andn,
    Eq,
    Lt,
    Gt,
    RightOperand,
    RightOperandW,
    XIsZero,
    YIsZero,
    Lsb,
    TwoLsbZero,
    DivByZero,
    ChangeDivisor,
    ChangeDivisorW,
    LeftShift,
    LeftShiftW,
    LeftShiftWHelper,
    RightShift,
    RightShiftW,
    RightShiftHelper,
    RightShiftWHelper,
    SignExtension,
    SignExtensionUpperHalf,
    SignExtensionRightOperand,
    XorRotate {
        rotation: u32,
        word_bits: u32,
    },
    /// 1 << (lower `split_bits` of input). Returns 1 on empty.
    Pow2 {
        split_bits: usize,
    },
    /// 1 << (XLEN - 1 - shift), where shift = lower log2(XLEN) bits. Returns 1 on empty.
    RightShiftPaddingMask,
    UpperWord,
    LowerWord,
    LowerHalfWord,
    ByteReverseW,
    OverflowBitsZero,
}

/// Minimum bit width required for eval_suffix_op to not panic on this op.
/// Split-based ops need at least `split_bits` when the input is non-empty.
fn min_bits(op: SuffixOp) -> usize {
    match op {
        SuffixOp::Pow2 { split_bits } => split_bits,
        SuffixOp::RightShiftPaddingMask => XLEN.trailing_zeros() as usize,
        _ => 0,
    }
}

/// Map a Suffixes enum variant to its SuffixOp representation.
fn suffix_to_op(suffix: &Suffixes) -> SuffixOp {
    match suffix {
        Suffixes::One => SuffixOp::One,
        Suffixes::And => SuffixOp::And,
        Suffixes::NotAnd => SuffixOp::Andn,
        Suffixes::Or => SuffixOp::Or,
        Suffixes::Xor => SuffixOp::Xor,
        Suffixes::Eq => SuffixOp::Eq,
        Suffixes::LessThan => SuffixOp::Lt,
        Suffixes::GreaterThan => SuffixOp::Gt,
        Suffixes::RightOperand => SuffixOp::RightOperand,
        Suffixes::RightOperandW => SuffixOp::RightOperandW,
        Suffixes::LeftOperandIsZero => SuffixOp::XIsZero,
        Suffixes::RightOperandIsZero => SuffixOp::YIsZero,
        Suffixes::Lsb => SuffixOp::Lsb,
        Suffixes::TwoLsb => SuffixOp::TwoLsbZero,
        Suffixes::DivByZero => SuffixOp::DivByZero,
        Suffixes::ChangeDivisor => SuffixOp::ChangeDivisor,
        Suffixes::ChangeDivisorW => SuffixOp::ChangeDivisorW,
        Suffixes::LeftShift => SuffixOp::LeftShift,
        Suffixes::LeftShiftW => SuffixOp::LeftShiftW,
        Suffixes::LeftShiftWHelper => SuffixOp::LeftShiftWHelper,
        Suffixes::RightShift => SuffixOp::RightShift,
        Suffixes::RightShiftW => SuffixOp::RightShiftW,
        Suffixes::RightShiftHelper => SuffixOp::RightShiftHelper,
        Suffixes::RightShiftWHelper => SuffixOp::RightShiftWHelper,
        Suffixes::SignExtension => SuffixOp::SignExtension,
        Suffixes::SignExtensionUpperHalf => SuffixOp::SignExtensionUpperHalf,
        Suffixes::SignExtensionRightOperand => SuffixOp::SignExtensionRightOperand,
        Suffixes::Pow2 => SuffixOp::Pow2 {
            split_bits: XLEN.trailing_zeros() as usize,
        },
        Suffixes::Pow2W => SuffixOp::Pow2 { split_bits: 5 },
        Suffixes::RightShiftPadding => SuffixOp::RightShiftPaddingMask,
        Suffixes::UpperWord => SuffixOp::UpperWord,
        Suffixes::LowerWord => SuffixOp::LowerWord,
        Suffixes::LowerHalfWord => SuffixOp::LowerHalfWord,
        Suffixes::Rev8W => SuffixOp::ByteReverseW,
        Suffixes::OverflowBitsZero => SuffixOp::OverflowBitsZero,
        Suffixes::XorRot16 => SuffixOp::XorRotate {
            rotation: 16,
            word_bits: 64,
        },
        Suffixes::XorRot24 => SuffixOp::XorRotate {
            rotation: 24,
            word_bits: 64,
        },
        Suffixes::XorRot32 => SuffixOp::XorRotate {
            rotation: 32,
            word_bits: 64,
        },
        Suffixes::XorRot63 => SuffixOp::XorRotate {
            rotation: 63,
            word_bits: 64,
        },
        Suffixes::XorRotW7 => SuffixOp::XorRotate {
            rotation: 7,
            word_bits: 32,
        },
        Suffixes::XorRotW8 => SuffixOp::XorRotate {
            rotation: 8,
            word_bits: 32,
        },
        Suffixes::XorRotW12 => SuffixOp::XorRotate {
            rotation: 12,
            word_bits: 32,
        },
        Suffixes::XorRotW16 => SuffixOp::XorRotate {
            rotation: 16,
            word_bits: 32,
        },
    }
}

/// Evaluate a suffix op on raw bits — no jolt-instructions dependency needed.
/// Mirrors the exact logic of each suffix_mle implementation.
fn eval_suffix_op(op: SuffixOp, bits: LookupBits) -> u64 {
    match op {
        // ── Full-bits operations (no uninterleave) ──
        SuffixOp::Pow2 { split_bits } => {
            if bits.is_empty() {
                return 1;
            }
            let (_, shift) = bits.split(split_bits);
            1u64 << u64::from(shift)
        }
        SuffixOp::RightShiftPaddingMask => {
            if bits.is_empty() {
                return 1;
            }
            let log_xlen = XLEN.trailing_zeros() as usize;
            let (_, shift) = bits.split(log_xlen);
            1u64 << (XLEN - 1 - u64::from(shift) as usize)
        }
        SuffixOp::UpperWord => (u128::from(bits) >> XLEN) as u64,
        SuffixOp::LowerWord => (u128::from(bits) % (1u128 << XLEN)) as u64,
        SuffixOp::LowerHalfWord => {
            let h = XLEN / 2;
            if h == 64 {
                u128::from(bits) as u64
            } else {
                (u128::from(bits) % (1u128 << h)) as u64
            }
        }
        SuffixOp::ByteReverseW => (u128::from(bits) as u32).swap_bytes() as u64,
        SuffixOp::OverflowBitsZero => ((u128::from(bits) >> XLEN) == 0) as u64,
        SuffixOp::Lsb => {
            if bits.is_empty() {
                1
            } else {
                (u128::from(bits) & 1) as u64
            }
        }
        SuffixOp::TwoLsbZero => (bits.is_empty() || u128::from(bits).trailing_zeros() >= 2) as u64,
        SuffixOp::SignExtensionUpperHalf => {
            let half = XLEN / 2;
            if bits.len() >= half {
                let b = u128::from(bits);
                let sign = (b >> (half - 1)) & 1;
                if sign == 1 {
                    ((1u64 << half) - 1) << half
                } else {
                    0
                }
            } else {
                1
            }
        }
        SuffixOp::SignExtensionRightOperand => {
            if bits.len() >= XLEN {
                let b = u128::from(bits);
                let sign = (b >> (XLEN - 2)) & 1;
                if sign == 1 {
                    ((1u128 << XLEN) - (1u128 << (XLEN / 2))) as u64
                } else {
                    0
                }
            } else {
                1
            }
        }

        // ── Uninterleave-based operations ──
        _ => {
            let (x, y) = bits.uninterleave();
            let xv = u64::from(x);
            let yv = u64::from(y);
            match op {
                SuffixOp::One => 1,
                SuffixOp::And => xv & yv,
                SuffixOp::Or => xv | yv,
                SuffixOp::Xor => xv ^ yv,
                SuffixOp::Andn => xv & !yv,
                SuffixOp::Eq => (xv == yv) as u64,
                SuffixOp::Lt => (xv < yv) as u64,
                SuffixOp::Gt => (xv > yv) as u64,
                SuffixOp::RightOperand => yv,
                SuffixOp::RightOperandW => yv as u32 as u64,
                SuffixOp::XIsZero => (xv == 0) as u64,
                SuffixOp::YIsZero => (yv == 0) as u64,
                SuffixOp::DivByZero => {
                    let div_zero = xv == 0;
                    let quot_ones = yv == (1u64 << y.len()) - 1;
                    (div_zero && quot_ones) as u64
                }
                SuffixOp::ChangeDivisor => ((1u64 << y.len()) - 1 == yv && xv == 0) as u64,
                SuffixOp::ChangeDivisorW => {
                    let y_len = y.len().min(XLEN / 2);
                    let xw = xv as u32 as u64;
                    let yw = yv as u32 as u64;
                    ((1u64 << y_len) - 1 == yw && xw == 0) as u64
                }
                SuffixOp::LeftShift => (xv & !yv).unbounded_shl(y.leading_ones()),
                SuffixOp::LeftShiftW => {
                    let yc = LookupBits::new(u128::from(y), y.len().min(XLEN / 2));
                    let xw = xv as u32;
                    let yw = u32::from(yc);
                    (xw & !yw).unbounded_shl(yc.leading_ones()) as u64
                }
                SuffixOp::LeftShiftWHelper => (1u32 << y.leading_ones()) as u64,
                SuffixOp::RightShift => xv.unbounded_shr(y.trailing_zeros()),
                SuffixOp::RightShiftW => {
                    (xv as u32).unbounded_shr(y.trailing_zeros().min(XLEN as u32 / 2)) as u64
                }
                SuffixOp::RightShiftHelper => 1u64 << y.leading_ones(),
                SuffixOp::RightShiftWHelper => {
                    let yc = LookupBits::new(u128::from(y), y.len().min(XLEN / 2));
                    1u64 << yc.leading_ones()
                }
                SuffixOp::SignExtension => {
                    let padding = std::cmp::min(yv.trailing_zeros() as usize, y.len());
                    ((1u128 << XLEN) - (1u128 << (XLEN - padding))) as u64
                }
                SuffixOp::XorRotate {
                    rotation,
                    word_bits,
                } => {
                    if word_bits == 64 {
                        (xv ^ yv).rotate_right(rotation)
                    } else {
                        ((xv as u32) ^ (yv as u32)).rotate_right(rotation) as u64
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

fn all_table_kinds() -> [LookupTableKind; LookupTableKind::COUNT] {
    [
        LookupTableKind::RangeCheck,
        LookupTableKind::RangeCheckAligned,
        LookupTableKind::And,
        LookupTableKind::Andn,
        LookupTableKind::Or,
        LookupTableKind::Xor,
        LookupTableKind::Equal,
        LookupTableKind::NotEqual,
        LookupTableKind::SignedLessThan,
        LookupTableKind::UnsignedLessThan,
        LookupTableKind::SignedGreaterThanEqual,
        LookupTableKind::UnsignedGreaterThanEqual,
        LookupTableKind::UnsignedLessThanEqual,
        LookupTableKind::UpperWord,
        LookupTableKind::LowerHalfWord,
        LookupTableKind::SignExtendHalfWord,
        LookupTableKind::Movsign,
        LookupTableKind::Pow2,
        LookupTableKind::Pow2W,
        LookupTableKind::ShiftRightBitmask,
        LookupTableKind::VirtualSRL,
        LookupTableKind::VirtualSRA,
        LookupTableKind::VirtualROTR,
        LookupTableKind::VirtualROTRW,
        LookupTableKind::ValidDiv0,
        LookupTableKind::ValidUnsignedRemainder,
        LookupTableKind::ValidSignedRemainder,
        LookupTableKind::VirtualChangeDivisor,
        LookupTableKind::VirtualChangeDivisorW,
        LookupTableKind::HalfwordAlignment,
        LookupTableKind::WordAlignment,
        LookupTableKind::MulUNoOverflow,
        LookupTableKind::VirtualRev8W,
        LookupTableKind::VirtualXORROT32,
        LookupTableKind::VirtualXORROT24,
        LookupTableKind::VirtualXORROT16,
        LookupTableKind::VirtualXORROT63,
        LookupTableKind::VirtualXORROTW16,
        LookupTableKind::VirtualXORROTW12,
        LookupTableKind::VirtualXORROTW8,
        LookupTableKind::VirtualXORROTW7,
    ]
}

/// Extract SuffixOp for each suffix of each table via direct enum mapping.
fn extract_suffix_ops() -> Vec<Vec<SuffixOp>> {
    all_table_kinds()
        .iter()
        .map(|&kind| {
            let table = LookupTables::<XLEN>::from(kind);
            table.suffixes().iter().map(suffix_to_op).collect()
        })
        .collect()
}

fn synthetic_trace(trace_length: usize) -> LookupTraceData {
    let mut lookup_keys = vec![0u128; trace_length];
    let mut table_kind_indices = vec![None; trace_length];
    let mut is_interleaved = vec![true; trace_length];

    if trace_length > 0 {
        lookup_keys[0] = 0x0000_00FF_0000_00AA_0000_0055_0000_0033;
        table_kind_indices[0] = Some(2); // And
        is_interleaved[0] = true;
    }
    if trace_length > 1 {
        lookup_keys[1] = 0x0000_0012_0000_0034_0000_0056_0000_0078;
        table_kind_indices[1] = Some(5); // Xor
        is_interleaved[1] = true;
    }
    if trace_length > 2 {
        lookup_keys[2] = 0x0000_0000_0000_0000_0000_0000_0000_0042;
        table_kind_indices[2] = Some(0); // RangeCheck
        is_interleaved[2] = false;
    }
    if trace_length > 3 {
        is_interleaved[3] = true;
    }

    LookupTraceData {
        lookup_keys,
        table_kind_indices,
        is_interleaved,
    }
}

// ═════════════════════════════════════════════════════════════════
// Step 1: Suffix evaluation
// ═════════════════════════════════════════════════════════════════

/// Validates that SuffixOp evaluation matches suffix_mle for all suffixes
/// across multiple bit widths.
#[test]
fn suffix_ops_match_suffix_mle() {
    let suffix_ops = extract_suffix_ops();
    let all_kinds = all_table_kinds();

    for suffix_bits in (2..=16).step_by(2) {
        let m = 1usize << suffix_bits;
        let step = if m > 256 { m / 256 } else { 1 };

        for (t_idx, &kind) in all_kinds.iter().enumerate() {
            let table = LookupTables::<XLEN>::from(kind);
            let suffixes = table.suffixes();
            for (s_idx, suffix) in suffixes.iter().enumerate() {
                let op = suffix_ops[t_idx][s_idx];
                // Skip widths too small for split-based ops
                if suffix_bits < min_bits(op) {
                    continue;
                }
                for bits in (0..m).step_by(step) {
                    let b = LookupBits::new(bits as u128, suffix_bits);
                    let expected = suffix.suffix_mle::<XLEN>(b);
                    let got = eval_suffix_op(op, b);
                    assert_eq!(
                        expected, got,
                        "suffix mismatch: table={t_idx} suffix={s_idx} bits={bits:#x} \
                         width={suffix_bits} op={op:?}",
                    );
                }
            }
        }
        eprintln!(
            "suffix_ops_match: width={suffix_bits} OK ({} tables)",
            all_kinds.len(),
        );
    }
}

/// Validates that scatter using SuffixOp produces the same suffix_polys
/// as the evaluator's init_phase_buffers.
#[test]
fn suffix_scatter_matches_evaluator() {
    let chunk_bits = 2usize;
    let num_phases = 8usize;
    let trace_length = 8;
    let trace = synthetic_trace(trace_length);
    let evaluator = JoltPrefixSuffixEvaluator::new();
    let suffix_ops = extract_suffix_ops();

    let u_evals: Vec<F> = (0..trace_length)
        .map(|i| F::from_u64(i as u64 + 1))
        .collect();

    let num_tables = evaluator.num_tables();
    let mut lookup_indices_by_table = vec![Vec::new(); num_tables];
    for (j, kind) in trace.table_kind_indices.iter().enumerate() {
        if let Some(idx) = kind {
            lookup_indices_by_table[*idx].push(j);
        }
    }

    let registry_cps: [Option<F>; 3] = [None, None, None];

    for phase in 0..num_phases {
        let ref_buffers = evaluator.init_phase_buffers(
            phase,
            &trace.lookup_keys,
            &u_evals,
            &trace.table_kind_indices,
            &trace.is_interleaved,
            &lookup_indices_by_table,
            &registry_cps,
            chunk_bits,
            num_phases,
        );

        let m = 1usize << chunk_bits;
        let m_mask = m - 1;
        let suffix_len = (num_phases - 1 - phase) * chunk_bits;

        let mut new_suffix_polys: Vec<Vec<Vec<F>>> = Vec::with_capacity(num_tables);
        for t_idx in 0..num_tables {
            let num_suffixes = suffix_ops[t_idx].len();
            let mut table_polys = vec![vec![F::zero(); m]; num_suffixes];

            for &j in &lookup_indices_by_table[t_idx] {
                let k = LookupBits::new(trace.lookup_keys[j], 128);
                let (prefix, suffix) = k.split(suffix_len);
                let idx: usize = prefix & m_mask;
                let u = u_evals[j];

                for s_idx in 0..num_suffixes {
                    let t = eval_suffix_op(suffix_ops[t_idx][s_idx], suffix);
                    if t == 0 {
                        continue;
                    }
                    if t == 1 {
                        table_polys[s_idx][idx] += u;
                    } else {
                        table_polys[s_idx][idx] += u.mul_u64(t);
                    }
                }
            }
            new_suffix_polys.push(table_polys);
        }

        for (t_idx, (ref_table, new_table)) in ref_buffers
            .suffix_polys
            .iter()
            .zip(new_suffix_polys.iter())
            .enumerate()
        {
            for (s_idx, (ref_poly, new_poly)) in ref_table.iter().zip(new_table.iter()).enumerate()
            {
                assert_eq!(
                    ref_poly, new_poly,
                    "suffix_poly mismatch at phase={phase} table={t_idx} suffix={s_idx}"
                );
            }
        }
    }
    eprintln!("suffix_scatter_matches_evaluator: all {num_phases} phases OK");
}

// ═════════════════════════════════════════════════════════════════
// Step 2: Checkpoint update rules
// ═════════════════════════════════════════════════════════════════

/// Validates that data-driven checkpoint updates match the evaluator
/// for all 46 prefixes across many rounds.
#[test]
fn checkpoint_updates_match_evaluator() {
    let evaluator = JoltPrefixSuffixEvaluator::new();
    let num_prefixes = evaluator.num_prefixes();
    let rules: Vec<CheckpointRule> = ALL_PREFIXES.iter().map(prefix_to_rule::<XLEN>).collect();
    assert_eq!(rules.len(), num_prefixes);

    let chunk_bits = 2usize;
    let total_bits = 128usize;

    let mut checkpoints_ref: Vec<Option<F>> = vec![None; num_prefixes];
    let mut checkpoints_new: Vec<Option<F>> = vec![None; num_prefixes];

    let mut rounds_checked = 0usize;
    for global_round in (1..total_bits).step_by(2) {
        let r_x = F::from_u64(global_round as u64 * 7 + 3);
        let r_y = F::from_u64(global_round as u64 * 13 + 5);
        let phase = global_round / chunk_bits;
        let suffix_len = total_bits.saturating_sub((phase + 1) * chunk_bits);
        let round = global_round;

        evaluator.update_checkpoints(&mut checkpoints_ref, r_x, r_y, round, suffix_len);
        let prev: Vec<Option<F>> = checkpoints_new.clone();
        for (i, rule) in rules.iter().enumerate() {
            checkpoints_new[i] = eval_checkpoint_rule(rule, i, &prev, r_x, r_y, round, suffix_len);
        }

        for (i, (a, b)) in checkpoints_ref
            .iter()
            .zip(checkpoints_new.iter())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "checkpoint mismatch at prefix={i} ({:?}) after round={round} suffix_len={suffix_len}",
                ALL_PREFIXES[i],
            );
        }
        rounds_checked += 1;
    }
    eprintln!(
        "checkpoint_updates_match_evaluator: {rounds_checked} rounds, {num_prefixes} prefixes OK"
    );
}

fn prefix_to_rule<const XLEN: usize>(prefix: &Prefixes) -> CheckpointRule {
    use BilinearExpr::{
        AntiXY, AntiYX, EqBit, NorBit, OneMinusX, OneMinusY, OnePlusY, OrBit, Product, XorBit, X, Y,
    };
    use CheckpointAction::{
        AddTwoTerm, AddWeighted, Const, DepAdd, DepAddWeighted, Hybrid, Mul, Null, Passthrough,
        Pow2DoubleMul, Pow2Init, Rev8WAdd, Set, SetScaled, SignExtAccum,
    };
    use DefaultVal::{Custom, One, Zero};
    use RoundGuard::{JEq, JGe, JGt, JLt, SuffixLenNonZero};
    use WeightFn::{LinearJ, LinearJMinusOne, Positional};

    let pos = |rot: u32, wb: u32, off: usize| Positional {
        rotation: rot,
        word_bits: wb,
        j_offset: off,
    };

    match prefix {
        Prefixes::Eq => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(EqBit),
        },
        Prefixes::LeftOperandIsZero => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(OneMinusX),
        },
        Prefixes::RightOperandIsZero => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(OneMinusY),
        },
        Prefixes::DivByZero => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(AntiXY),
        },
        Prefixes::LeftShiftHelper => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(OnePlusY),
        },
        Prefixes::And => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: Product,
            },
        },
        Prefixes::Andn => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: AntiYX,
            },
        },
        Prefixes::Or => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: OrBit,
            },
        },
        Prefixes::Xor => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::RightOperand => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: Y,
            },
        },
        Prefixes::XorRot16 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(16, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRot24 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(24, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRot32 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(32, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRot63 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(63, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW7 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(7, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW8 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(8, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW12 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(12, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW16 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(16, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::UpperWord => CheckpointRule {
            default: Zero,
            cases: vec![(JGe(XLEN), Passthrough)],
            fallback: AddTwoTerm {
                x_weight: LinearJ { base: XLEN },
                y_weight: LinearJMinusOne { base: XLEN },
            },
        },
        Prefixes::LowerWord => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Null)],
            fallback: AddTwoTerm {
                x_weight: LinearJ { base: 2 * XLEN },
                y_weight: LinearJMinusOne { base: 2 * XLEN },
            },
        },
        Prefixes::LowerHalfWord => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN + XLEN / 2), Null)],
            fallback: AddTwoTerm {
                x_weight: LinearJ { base: 2 * XLEN },
                y_weight: LinearJMinusOne { base: 2 * XLEN },
            },
        },
        Prefixes::LessThan => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: DepAdd {
                dep: Prefixes::Eq as usize,
                dep_default: One,
                expr: AntiXY,
            },
        },
        Prefixes::LeftShift => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: DepAddWeighted {
                dep: Prefixes::LeftShiftHelper as usize,
                dep_default: One,
                weight: pos(0, XLEN as u32, 0),
                expr: AntiYX,
            },
        },
        Prefixes::RightShift => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: Hybrid {
                mul: OnePlusY,
                add: Product,
            },
        },
        Prefixes::LeftOperandMsb => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(X))],
            fallback: Passthrough,
        },
        Prefixes::RightOperandMsb => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(Y))],
            fallback: Passthrough,
        },
        Prefixes::TwoLsb => CheckpointRule {
            default: One,
            cases: vec![(JEq(2 * XLEN - 1), Set(NorBit))],
            fallback: Passthrough,
        },
        Prefixes::SignExtensionUpperHalf => CheckpointRule {
            default: Zero,
            cases: vec![(
                JEq(XLEN + XLEN / 2 + 1),
                SetScaled {
                    coeff: (((1i128 << (XLEN / 2)) - 1) << (XLEN / 2)),
                    expr: X,
                },
            )],
            fallback: Passthrough,
        },
        Prefixes::SignExtensionRightOperand => CheckpointRule {
            default: Zero,
            cases: vec![(
                JEq(XLEN + 1),
                SetScaled {
                    coeff: (1i128 << XLEN) - (1i128 << (XLEN / 2)),
                    expr: Y,
                },
            )],
            fallback: Passthrough,
        },
        Prefixes::Lsb => CheckpointRule {
            default: One,
            cases: vec![(JEq(2 * XLEN - 1), Set(Y))],
            fallback: Const(One),
        },
        Prefixes::PositiveRemainderEqualsDivisor => CheckpointRule {
            default: One,
            cases: vec![(JEq(1), Set(NorBit))],
            fallback: Mul(EqBit),
        },
        Prefixes::NegativeDivisorEqualsRemainder => CheckpointRule {
            default: One,
            cases: vec![(JEq(1), Set(Product))],
            fallback: Mul(EqBit),
        },
        Prefixes::NegativeDivisorZeroRemainder => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(AntiXY))],
            fallback: Mul(OneMinusX),
        },
        Prefixes::PositiveRemainderLessThanDivisor => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(NorBit)), (JEq(3), Mul(AntiXY))],
            fallback: DepAdd {
                dep: Prefixes::PositiveRemainderEqualsDivisor as usize,
                dep_default: One,
                expr: AntiXY,
            },
        },
        Prefixes::NegativeDivisorGreaterThanRemainder => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(Product)), (JEq(3), Mul(AntiYX))],
            fallback: DepAdd {
                dep: Prefixes::NegativeDivisorEqualsRemainder as usize,
                dep_default: One,
                expr: AntiYX,
            },
        },
        Prefixes::ChangeDivisor => CheckpointRule {
            default: Custom(2 - (1i128 << XLEN)),
            cases: vec![(JEq(1), Mul(Product))],
            fallback: Mul(AntiXY),
        },
        Prefixes::ChangeDivisorW => CheckpointRule {
            default: Zero,
            cases: vec![
                (JLt(XLEN), Const(Zero)),
                (
                    JEq(XLEN + 1),
                    SetScaled {
                        coeff: 2 - (1i128 << XLEN),
                        expr: Product,
                    },
                ),
            ],
            fallback: Mul(AntiXY),
        },
        Prefixes::SignExtension => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Null)],
            fallback: SignExtAccum {
                dep: Prefixes::LeftOperandMsb as usize,
                final_j: 2 * XLEN - 1,
            },
        },
        Prefixes::Pow2 => {
            let trail = XLEN.trailing_zeros() as usize;
            CheckpointRule {
                default: One,
                cases: vec![
                    (SuffixLenNonZero, Const(One)),
                    (
                        JEq(2 * XLEN - trail),
                        Pow2Init {
                            half_pow: (XLEN / 2) as u32,
                        },
                    ),
                    (JGt(2 * XLEN - trail), Pow2DoubleMul { xlen: XLEN }),
                ],
                fallback: Const(One),
            }
        }
        Prefixes::Pow2W => CheckpointRule {
            default: One,
            cases: vec![
                (SuffixLenNonZero, Const(One)),
                (JEq(2 * XLEN - 5), Pow2Init { half_pow: 16 }),
                (JGt(2 * XLEN - 5), Pow2DoubleMul { xlen: XLEN }),
            ],
            fallback: Const(One),
        },
        Prefixes::Rev8W => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: Rev8WAdd { xlen: 64 },
        },
        Prefixes::RightShiftW => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: Hybrid {
                mul: OnePlusY,
                add: Product,
            },
        },
        Prefixes::LeftShiftWHelper => CheckpointRule {
            default: One,
            cases: vec![(JLt(XLEN), Const(One))],
            fallback: Mul(OnePlusY),
        },
        Prefixes::LeftShiftW => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: DepAddWeighted {
                dep: Prefixes::LeftShiftWHelper as usize,
                dep_default: One,
                weight: pos(0, XLEN as u32, 0),
                expr: AntiYX,
            },
        },
        Prefixes::RightOperandW => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN + 1), Passthrough)],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: Y,
            },
        },
        Prefixes::OverflowBitsZero => CheckpointRule {
            default: One,
            cases: vec![(JGe(128 - XLEN), Passthrough)],
            fallback: Mul(NorBit),
        },
    }
}

// ═════════════════════════════════════════════════════════════════
// Step 3: Prefix materialization (read-checking reduce)
// ═════════════════════════════════════════════════════════════════

fn prefix_to_mle_rule<const XLEN: usize>(prefix: &Prefixes) -> PrefixMleRule {
    use BilinearExpr::*;

    let total_bits = 2 * XLEN;
    let idx = *prefix as usize;

    let rule = |default, formula| PrefixMleRule {
        checkpoint_idx: idx,
        default,
        formula,
    };

    match prefix {
        // ── Multiplicative family ──
        Prefixes::Eq => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: EqBit,
                remaining: RemainingTest::Equality,
            },
        ),
        Prefixes::LeftOperandIsZero => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: OneMinusX,
                remaining: RemainingTest::LeftZero,
            },
        ),
        Prefixes::RightOperandIsZero => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: OneMinusY,
                remaining: RemainingTest::RightZero,
            },
        ),
        Prefixes::DivByZero => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: AntiXY,
                remaining: RemainingTest::LeftZeroRightAllOnes,
            },
        ),

        // ── BitwiseAdditive family ──
        Prefixes::And => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: Product,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::And,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::Andn => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: AntiYX,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::AndNot,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::Or => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: OrBit,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Or,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::Xor => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::RightOperand => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightOperandExtract {
                word_bits: XLEN,
                start_round: 0,
                total_bits,
                suffix_guard: total_bits,
            },
        ),
        Prefixes::XorRot16 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 16,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 16,
            },
        ),
        Prefixes::XorRot24 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 24,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 24,
            },
        ),
        Prefixes::XorRot32 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 32,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 32,
            },
        ),
        Prefixes::XorRot63 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 63,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 63,
            },
        ),
        Prefixes::XorRotW7 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 7,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 7,
            },
        ),
        Prefixes::XorRotW8 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 8,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 8,
            },
        ),
        Prefixes::XorRotW12 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 12,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 12,
            },
        ),
        Prefixes::XorRotW16 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 16,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 16,
            },
        ),

        // ── OperandExtract family ──
        Prefixes::UpperWord => rule(
            DefaultVal::Zero,
            PrefixMleFormula::OperandExtract {
                base_shift: XLEN,
                has_x: true,
                has_y: true,
                word_bits: XLEN,
                total_bits,
                active_when: Some(RoundGuard::JLt(XLEN)),
                passthrough_when_inactive: true,
                is_upper: true,
            },
        ),
        Prefixes::LowerWord => rule(
            DefaultVal::Zero,
            PrefixMleFormula::OperandExtract {
                base_shift: 2 * XLEN,
                has_x: true,
                has_y: true,
                word_bits: XLEN,
                total_bits,
                active_when: Some(RoundGuard::JGe(XLEN)),
                passthrough_when_inactive: false,
                is_upper: false,
            },
        ),
        Prefixes::LowerHalfWord => rule(
            DefaultVal::Zero,
            PrefixMleFormula::OperandExtract {
                base_shift: 2 * XLEN,
                has_x: true,
                has_y: true,
                word_bits: XLEN / 2,
                total_bits,
                active_when: Some(RoundGuard::JGe(XLEN + XLEN / 2)),
                passthrough_when_inactive: false,
                is_upper: false,
            },
        ),
        Prefixes::RightOperandW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightOperandExtract {
                word_bits: XLEN,
                start_round: XLEN,
                total_bits,
                suffix_guard: XLEN,
            },
        ),

        // ── DependentComparison family ──
        Prefixes::LessThan => rule(
            DefaultVal::Zero,
            PrefixMleFormula::DependentComparison {
                eq_idx: Prefixes::Eq as usize,
                eq_default: DefaultVal::One,
                cmp: Comparison::LessThan,
            },
        ),

        // ── ComplexComparison family ──
        Prefixes::PositiveRemainderLessThanDivisor => rule(
            DefaultVal::Zero,
            PrefixMleFormula::ComplexComparison {
                eq_idx: Prefixes::PositiveRemainderEqualsDivisor as usize,
                cmp: Comparison::LessThan,
                sign_pair_j0: NorBit,
                init_cmp: Comparison::LessThan,
                mul_anti: AntiXY,
                start_round: 0,
            },
        ),
        Prefixes::NegativeDivisorGreaterThanRemainder => rule(
            DefaultVal::Zero,
            PrefixMleFormula::ComplexComparison {
                eq_idx: Prefixes::NegativeDivisorEqualsRemainder as usize,
                cmp: Comparison::GreaterThan,
                sign_pair_j0: Product,
                init_cmp: Comparison::GreaterThan,
                mul_anti: AntiYX,
                start_round: 0,
            },
        ),

        // ── SignGatedMultiplicative family ──
        Prefixes::PositiveRemainderEqualsDivisor => rule(
            DefaultVal::One,
            PrefixMleFormula::SignGatedMultiplicative {
                sign_pair: NorBit,
                base_pair: EqBit,
                remaining: RemainingTest::Equality,
                start_round: 0,
            },
        ),
        Prefixes::NegativeDivisorEqualsRemainder => rule(
            DefaultVal::One,
            PrefixMleFormula::SignGatedMultiplicative {
                sign_pair: Product,
                base_pair: EqBit,
                remaining: RemainingTest::Equality,
                start_round: 0,
            },
        ),

        // ── LeftShift family ──
        Prefixes::LeftShift => rule(
            DefaultVal::Zero,
            PrefixMleFormula::LeftShift {
                helper_idx: Prefixes::LeftShiftHelper as usize,
                helper_default: DefaultVal::One,
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::LeftShiftHelper => rule(
            DefaultVal::One,
            PrefixMleFormula::LeftShiftHelper {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::LeftShiftW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::LeftShift {
                helper_idx: Prefixes::LeftShiftWHelper as usize,
                helper_default: DefaultVal::One,
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),
        Prefixes::LeftShiftWHelper => rule(
            DefaultVal::One,
            PrefixMleFormula::LeftShiftHelper {
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),

        // ── RightShift family ──
        Prefixes::RightShift => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightShift {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::RightShiftW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightShift {
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),

        // ── MSB family ──
        Prefixes::LeftOperandMsb => rule(
            DefaultVal::Zero,
            PrefixMleFormula::Msb {
                msb_idx: Prefixes::LeftOperandMsb as usize,
                start_round: 0,
                is_left: true,
            },
        ),
        Prefixes::RightOperandMsb => rule(
            DefaultVal::Zero,
            PrefixMleFormula::Msb {
                msb_idx: Prefixes::RightOperandMsb as usize,
                start_round: 0,
                is_left: false,
            },
        ),

        // ── Lsb / TwoLsb family ──
        Prefixes::Lsb => rule(DefaultVal::One, PrefixMleFormula::Lsb { total_bits }),
        Prefixes::TwoLsb => rule(DefaultVal::One, PrefixMleFormula::TwoLsb { total_bits }),

        // ── SignExtension family ──
        Prefixes::SignExtension => rule(
            DefaultVal::Zero,
            PrefixMleFormula::SignExtension {
                msb_idx: Prefixes::LeftOperandMsb as usize,
                word_bits: XLEN,
            },
        ),
        Prefixes::SignExtensionUpperHalf => rule(
            DefaultVal::Zero,
            PrefixMleFormula::SignExtUpperHalf { word_bits: XLEN },
        ),
        Prefixes::SignExtensionRightOperand => rule(
            DefaultVal::Zero,
            PrefixMleFormula::SignExtRightOp { word_bits: XLEN },
        ),

        // ── Pow2 family ──
        Prefixes::Pow2 => rule(
            DefaultVal::One,
            PrefixMleFormula::Pow2 {
                word_mask: XLEN - 1,
                log_word_bits: XLEN.trailing_zeros(),
                total_bits,
            },
        ),
        Prefixes::Pow2W => rule(
            DefaultVal::One,
            PrefixMleFormula::Pow2 {
                word_mask: 31,
                log_word_bits: 5,
                total_bits,
            },
        ),

        // ── Rev8W ──
        Prefixes::Rev8W => rule(DefaultVal::Zero, PrefixMleFormula::Rev8W { xlen: 64 }),

        // ── ChangeDivisor family ──
        Prefixes::ChangeDivisor => rule(
            DefaultVal::Custom(2 - (1i128 << XLEN)),
            PrefixMleFormula::ChangeDivisor {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::ChangeDivisorW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::ChangeDivisor {
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),

        // ── NegDivZeroRem ──
        Prefixes::NegativeDivisorZeroRemainder => rule(
            DefaultVal::Zero,
            PrefixMleFormula::NegDivZeroRem {
                word_bits: XLEN,
                start_round: 0,
            },
        ),

        // ── OverflowBitsZero ──
        Prefixes::OverflowBitsZero => rule(
            DefaultVal::One,
            PrefixMleFormula::OverflowBitsZero {
                pair: NorBit,
                word_bits: XLEN,
                total_bits,
            },
        ),
    }
}

/// Validates that data-driven prefix buffer materialization + inner product
/// matches evaluator.compute_read_checking across multiple phases.
#[test]
fn read_checking_matches_evaluator() {
    let chunk_bits = 2usize;
    let num_phases = 8usize;
    let trace_length = 8;
    let trace = synthetic_trace(trace_length);
    let evaluator = JoltPrefixSuffixEvaluator::new();

    let u_evals: Vec<F> = (0..trace_length)
        .map(|i| F::from_u64(i as u64 + 1))
        .collect();

    let num_tables = evaluator.num_tables();
    let mut lookup_indices_by_table = vec![Vec::new(); num_tables];
    for (j, kind) in trace.table_kind_indices.iter().enumerate() {
        if let Some(idx) = kind {
            lookup_indices_by_table[*idx].push(j);
        }
    }

    let registry_cps: [Option<F>; 3] = [None, None, None];
    let num_prefixes = evaluator.num_prefixes();
    let mut checkpoints: Vec<Option<F>> = vec![None; num_prefixes];

    // Build data-driven prefix MLE rules and combine entries
    let mle_rules: Vec<PrefixMleRule> = ALL_PREFIXES
        .iter()
        .map(prefix_to_mle_rule::<XLEN>)
        .collect();

    let all_kinds = all_table_kinds();
    let combine_entries: Vec<CombineEntry> = all_kinds
        .iter()
        .enumerate()
        .flat_map(|(t_idx, &kind)| {
            let table = LookupTables::<XLEN>::from(kind);
            table
                .combine_entries()
                .into_iter()
                .map(move |entry| CombineEntry {
                    table_idx: t_idx,
                    prefix_idx: entry.prefix.map(|p| p as usize),
                    suffix_local_idx: entry.suffix_idx,
                    coefficient: entry.coefficient,
                })
        })
        .collect();

    let total_bits = 2 * XLEN;
    let mut rounds_checked = 0;

    for phase in 0..num_phases {
        let buffers = evaluator.init_phase_buffers(
            phase,
            &trace.lookup_keys,
            &u_evals,
            &trace.table_kind_indices,
            &trace.is_interleaved,
            &lookup_indices_by_table,
            &registry_cps,
            chunk_bits,
            num_phases,
        );

        let mut suffix_polys = buffers.suffix_polys;
        let global_round_base = phase * chunk_bits;

        for local_round in 0..chunk_bits {
            let round = global_round_base + local_round;
            let r_x: Option<F> = if round % 2 == 1 {
                Some(F::from_u64(round as u64 * 11 + 7))
            } else {
                None
            };

            let ref_result =
                evaluator.compute_read_checking(round, &suffix_polys, &checkpoints, r_x);
            let new_result = compute_read_checking_from_lowered(
                &mle_rules,
                &combine_entries,
                round,
                &suffix_polys,
                &checkpoints,
                r_x,
                total_bits,
            );

            assert_eq!(
                ref_result[0], new_result[0],
                "read_checking eval_0 mismatch at phase={phase} round={round}"
            );
            assert_eq!(
                ref_result[1], new_result[1],
                "read_checking eval_2 mismatch at phase={phase} round={round}"
            );
            rounds_checked += 1;

            let challenge = F::from_u64(round as u64 * 17 + 2);
            for table_polys in &mut suffix_polys {
                for poly in table_polys.iter_mut() {
                    let half = poly.len() / 2;
                    if half > 0 {
                        let mut bound = Vec::with_capacity(half);
                        for i in 0..half {
                            bound.push(poly[i] + challenge * (poly[i + half] - poly[i]));
                        }
                        *poly = bound;
                    }
                }
            }

            if round >= 1 && round % 2 == 1 {
                let r_x_val = r_x.unwrap();
                let suffix_len = total_bits - (round + 1);
                evaluator.update_checkpoints(
                    &mut checkpoints,
                    r_x_val,
                    challenge,
                    round,
                    suffix_len,
                );
            }
        }
    }
    eprintln!(
        "read_checking_matches_evaluator: {rounds_checked} rounds across {num_phases} phases OK"
    );
}

// ═════════════════════════════════════════════════════════════════
// Step 4: Combined val
// ═════════════════════════════════════════════════════════════════

/// Validates that data-driven combined_val computation matches evaluator.
/// Already fully data-driven — uses combine_matrix + suffix_at_empty.
#[test]
fn combined_val_matches_evaluator() {
    let evaluator = JoltPrefixSuffixEvaluator::new();
    let num_prefixes = evaluator.num_prefixes();
    let trace_length = 8;
    let trace = synthetic_trace(trace_length);

    let checkpoints: Vec<Option<F>> = (0..num_prefixes)
        .map(|i| Some(F::from_u64(i as u64 * 37 + 1)))
        .collect();

    let gamma = F::from_u64(999);
    let gamma_sqr = gamma * gamma;

    let registry_cps: [Option<F>; 3] = [
        Some(F::from_u64(100)),
        Some(F::from_u64(200)),
        Some(F::from_u64(300)),
    ];

    let ref_result = evaluator.compute_combined_val(
        &checkpoints,
        gamma,
        gamma_sqr,
        &trace.table_kind_indices,
        &trace.is_interleaved,
        &registry_cps,
        trace_length,
    );

    let all_kinds = all_table_kinds();
    let num_tables = all_kinds.len();

    let mut combine_entries = Vec::new();
    let mut suffix_at_empty = Vec::with_capacity(num_tables);
    for (t_idx, &kind) in all_kinds.iter().enumerate() {
        let table = LookupTables::<XLEN>::from(kind);
        let suffixes = table.suffixes();
        let empty = LookupBits::new(0, 0);
        suffix_at_empty.push(
            suffixes
                .iter()
                .map(|s| s.suffix_mle::<XLEN>(empty))
                .collect::<Vec<_>>(),
        );
        for entry in table.combine_entries() {
            combine_entries.push((
                t_idx,
                entry.prefix.map(|p| p as usize),
                entry.suffix_idx,
                entry.coefficient,
            ));
        }
    }

    let left_prefix = registry_cps[1].unwrap_or(F::zero());
    let right_prefix = registry_cps[0].unwrap_or(F::zero());
    let identity_prefix = registry_cps[2].unwrap_or(F::zero());
    let raf_interleaved = gamma * left_prefix + gamma_sqr * right_prefix;
    let raf_identity = gamma_sqr * identity_prefix;

    let prefix_vals: Vec<F> = checkpoints.iter().map(|v| v.unwrap()).collect();

    let mut table_values_at_r_addr = vec![F::zero(); num_tables];
    for &(t_idx, prefix_idx, suffix_local_idx, coefficient) in &combine_entries {
        let p_val = match prefix_idx {
            Some(p) => prefix_vals[p],
            None => F::one(),
        };
        let s_val = F::from_u64(suffix_at_empty[t_idx][suffix_local_idx]);
        let coeff = F::from_i128(coefficient);
        table_values_at_r_addr[t_idx] += coeff * p_val * s_val;
    }

    let new_result: Vec<F> = (0..trace_length)
        .map(|j| {
            let mut val = F::zero();
            if let Some(t_idx) = trace.table_kind_indices[j] {
                val += table_values_at_r_addr[t_idx];
            }
            if trace.is_interleaved[j] {
                val += raf_interleaved;
            } else {
                val += raf_identity;
            }
            val
        })
        .collect();

    for (j, (r, n)) in ref_result.iter().zip(new_result.iter()).enumerate() {
        assert_eq!(r, n, "combined_val mismatch at cycle={j}");
    }
    eprintln!("combined_val_matches_evaluator: {trace_length} cycles OK");
}

// ═════════════════════════════════════════════════════════════════
// Integration: full scatter-reduce-update loop
// ═════════════════════════════════════════════════════════════════

/// End-to-end test: runs through all phases and all rounds of a synthetic
/// trace, comparing data-driven outputs against the evaluator at every point.
///
/// NOTE: Placeholder — both sides use the evaluator until all steps land.
#[test]
fn full_ps_loop_matches_evaluator() {
    let chunk_bits = 2usize;
    let num_phases = 4usize;
    let trace_length = 8;
    let trace = synthetic_trace(trace_length);
    let evaluator = JoltPrefixSuffixEvaluator::new();

    let num_tables = evaluator.num_tables();
    let num_prefixes = evaluator.num_prefixes();
    let mut checkpoints: Vec<Option<F>> = vec![None; num_prefixes];
    let registry_cps: [Option<F>; 3] = [None, None, None];

    let mut u_evals: Vec<F> = (0..trace_length)
        .map(|i| F::from_u64(i as u64 + 1))
        .collect();

    let mut lookup_indices_by_table = vec![Vec::new(); num_tables];
    for (j, kind) in trace.table_kind_indices.iter().enumerate() {
        if let Some(idx) = kind {
            lookup_indices_by_table[*idx].push(j);
        }
    }

    let mut global_round = 0usize;
    for phase in 0..num_phases {
        let buffers = evaluator.init_phase_buffers(
            phase,
            &trace.lookup_keys,
            &u_evals,
            &trace.table_kind_indices,
            &trace.is_interleaved,
            &lookup_indices_by_table,
            &registry_cps,
            chunk_bits,
            num_phases,
        );

        let mut suffix_polys = buffers.suffix_polys;

        for local_round in 0..chunk_bits {
            let r_x: Option<F> = if local_round % 2 == 1 {
                Some(F::from_u64(global_round as u64 * 11 + 7))
            } else {
                None
            };

            let ref_result =
                evaluator.compute_read_checking(global_round, &suffix_polys, &checkpoints, r_x);
            // TODO: replace with data-driven compute
            let new_result =
                evaluator.compute_read_checking(global_round, &suffix_polys, &checkpoints, r_x);

            assert_eq!(
                ref_result[0], new_result[0],
                "eval_0 @ phase={phase} round={local_round}"
            );
            assert_eq!(
                ref_result[1], new_result[1],
                "eval_2 @ phase={phase} round={local_round}"
            );

            let challenge = F::from_u64(global_round as u64 * 17 + 2);
            for table_polys in &mut suffix_polys {
                for poly in table_polys.iter_mut() {
                    let half = poly.len() / 2;
                    if half > 0 {
                        let mut bound = Vec::with_capacity(half);
                        for i in 0..half {
                            bound.push(poly[i] + challenge * (poly[i + half] - poly[i]));
                        }
                        *poly = bound;
                    }
                }
            }

            if local_round >= 1 && local_round % 2 == 1 {
                let r_x_val = r_x.unwrap();
                let suffix_len =
                    (num_phases - phase - 1) * chunk_bits + (chunk_bits - local_round - 1);
                evaluator.update_checkpoints(
                    &mut checkpoints,
                    r_x_val,
                    challenge,
                    global_round,
                    suffix_len,
                );
            }

            global_round += 1;
        }

        if phase + 1 < num_phases {
            let m = 1usize << chunk_bits;
            let m_mask = m - 1;
            let suffix_len = (num_phases - phase - 1) * chunk_bits;
            for (j, &key) in trace.lookup_keys.iter().enumerate() {
                let prefix = key >> suffix_len;
                let k_bound = (prefix as usize) & m_mask;
                u_evals[j] *= F::from_u64(k_bound as u64 + 1);
            }
        }
    }
    eprintln!(
        "full_ps_loop: {num_phases} phases × {chunk_bits} rounds = {} total rounds OK",
        num_phases * chunk_bits
    );
}
