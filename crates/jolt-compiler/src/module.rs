//! Compiler module types consumed by the prover runtime and verifier.

use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::formula::BindingOrder;
use crate::ir::PolyKind;
use crate::kernel_spec::KernelSpec;
use crate::polynomial_id::PolynomialId;

/// Index into `VerifierSchedule::stages`.
///
/// Distinguishes verifier stage indices from staging loop counters.
/// Staging stages that produce no sumcheck (e.g. eval-only) may be
/// skipped, so the staging index and verifier index can diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerifierStageIndex(pub usize);

/// Index into `Module::challenges` / `RuntimeState::challenges`.
///
/// Prevents confusion with polynomial indices, batch indices, and other usize values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChallengeIdx(pub usize);

/// Index into `Schedule::batched_sumchecks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchIdx(pub usize);

/// Index into `BatchedSumcheckDef::instances`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstanceIdx(pub usize);

/// Complete output of the compilation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub polys: Vec<PolyDecl>,
    pub challenges: Vec<ChallengeDecl>,
    pub prover: Schedule,
    pub verifier: VerifierSchedule,
}

impl Module {
    /// Serialize to a compact binary format (`.jolt` protocol binary).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("module serialization should not fail")
    }

    /// Deserialize from a `.jolt` protocol binary.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let (module, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .expect("invalid protocol.jolt binary");
        module
    }
}

/// Polynomial declaration in the compiled module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyDecl {
    pub name: String,
    pub kind: PolyKind,
    /// Total number of field elements (2^num_vars).
    pub num_elements: usize,
    /// PCS-level number of variables for the opening proof.
    /// When `Some(n)`, the polynomial is zero-padded to `2^n` elements
    /// before PCS opening (e.g. dense cycle-only polys padded to K*T).
    /// When `None`, `num_elements` is used as-is.
    pub committed_num_vars: Option<usize>,
}

/// Challenge declaration in the compiled module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeDecl {
    pub name: String,
    pub source: ChallengeSource,
}

/// How a challenge value is determined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeSource {
    /// Squeezed from transcript after a stage completes.
    FiatShamir { after_stage: usize },
    /// Squeezed within a sumcheck stage after a round polynomial is appended.
    SumcheckRound {
        stage: VerifierStageIndex,
        round: usize,
    },
    /// Power of another challenge: `challenges[base]^exponent`.
    Power { base: ChallengeIdx, exponent: usize },
    /// From outside the protocol (preprocessing, public input).
    External,
}

/// Prover execution schedule: a flat sequence of ops with compiled kernel defs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub ops: Vec<Op>,
    pub kernels: Vec<KernelDef>,
    /// Batched sumcheck definitions (indexed by batch index in granular ops).
    pub batched_sumchecks: Vec<BatchedSumcheckDef>,
}

/// A batched sumcheck stage grouping heterogeneous instances.
///
/// Each instance has its own kernel (formula + inputs) and runs for a
/// subset of the total rounds. Shorter instances are front-loaded: inactive
/// in early rounds, contributing `claim/2` per round until they activate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedSumcheckDef {
    pub instances: Vec<BatchedInstance>,
    /// Per-instance input claim formulas (evaluated by the runtime before
    /// the first round, absorbed into transcript for Fiat-Shamir).
    pub input_claims: Vec<ClaimFormula>,
    pub max_rounds: usize,
    pub max_degree: usize,
}

/// A single instance within a [`BatchedSumcheckDef`].
///
/// An instance may span multiple *phases*, each with its own compiled kernel.
/// For most instances a single phase suffices. Multi-phase instances (e.g.
/// RamReadWriteChecking) transition between kernels mid-sumcheck — the runtime
/// resolves fresh inputs at each phase boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedInstance {
    /// Kernel phases executed sequentially. The sum of all phase `num_rounds`
    /// equals the total round count for this instance.
    pub phases: Vec<InstancePhase>,
    /// Challenge index for this instance's batching coefficient.
    pub batch_coeff: ChallengeIdx,
    /// First round where this instance is active (`max_rounds - num_rounds`).
    pub first_active_round: usize,
}

impl BatchedInstance {
    /// Total number of rounds across all phases.
    pub fn num_rounds(&self) -> usize {
        self.phases.iter().map(|p| p.num_rounds).sum()
    }

    /// Find the phase and its start offset for a given instance-local round.
    ///
    /// Returns `(phase_index, phase_start_round)` where `phase_start_round`
    /// is the first instance-local round of that phase.
    pub fn phase_for_round(&self, instance_round: usize) -> (usize, usize) {
        let mut cumulative = 0;
        for (i, phase) in self.phases.iter().enumerate() {
            if instance_round < cumulative + phase.num_rounds {
                return (i, cumulative);
            }
            cumulative += phase.num_rounds;
        }
        panic!(
            "instance_round {instance_round} exceeds total rounds {}",
            self.num_rounds()
        );
    }
}

/// One phase of a [`BatchedInstance`].
///
/// Each phase has its own compiled kernel. The runtime resolves the phase's
/// kernel inputs when the phase begins and binds them each subsequent round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstancePhase {
    /// Index into `Schedule.kernels`.
    pub kernel: usize,
    /// Number of sumcheck rounds in this phase.
    pub num_rounds: usize,
    /// Scalar captures: at the start of this phase, the runtime reads the
    /// scalar value from each listed polynomial's (now fully-bound) device
    /// buffer and stores it in the corresponding challenge slot.
    ///
    /// This bridges phase boundaries: intermediate values from a prior phase
    /// become challenge constants for the next phase's formula.
    pub scalar_captures: Vec<ScalarCapture>,
    /// When present, this phase uses segmented reduce for mixed-size inputs.
    pub segmented: Option<SegmentedConfig>,
    /// Extra buffers to bind alongside kernel inputs at each round.
    ///
    /// Carry bindings are materialized at phase start and bound with the
    /// same round challenge as the kernel inputs.  They do NOT participate
    /// in kernel evaluation — only in binding.  This bridges multi-phase
    /// instances where a later phase needs the bound-down version of a
    /// buffer that was not part of the earlier phase's formula.
    pub carry_bindings: Vec<InputBinding>,
    /// Ops emitted at phase activation, before kernel input materialization.
    /// Used for data-driven buffer construction (e.g., WeightedSum for
    /// BytecodeVal decomposition). Empty for most phases.
    pub pre_activation_ops: Vec<Op>,
}

/// Captures a scalar value from a bound buffer into a challenge slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarCapture {
    /// Polynomial whose device buffer holds a single scalar (1-element).
    pub poly: PolynomialId,
    /// Challenge index where the scalar value is stored.
    pub challenge: ChallengeIdx,
}

/// Configuration for segmented reduce in a multi-dimensional sumcheck phase.
///
/// When a phase has mixed-size inputs (e.g. T-element cycle-only polynomials
/// alongside T×K-element cycle×address polynomials), the runtime performs a
/// segmented reduce: iterating over outer positions, extracting inner columns
/// from mixed inputs, running the Dense kernel on inner-sized slices, and
/// accumulating with outer eq weights.
///
/// The kernel itself is compiled as Dense over the inner dimension. The
/// segmented structure is runtime-level orchestration, not a backend concern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentedConfig {
    /// Log₂ of the inner segment size (bound in this phase).
    pub inner_num_vars: usize,
    /// Log₂ of the outer segment size (bound in a later phase).
    pub outer_num_vars: usize,
    /// Per kernel input: `true` = inner-only (`2^inner_num_vars` elements),
    /// `false` = full inner×outer (`2^(inner + outer)` elements).
    pub inner_only: Vec<bool>,
    /// Challenge indices for the outer eq table (built once at phase start).
    pub outer_eq_challenges: Vec<ChallengeIdx>,
}

/// One entry of the combine matrix for read-checking.
///
/// Encodes `coefficient × prefix[prefix_idx] × suffix[table_idx][suffix_local_idx]`.
/// When `prefix_idx` is `None`, the prefix factor is 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombineEntry {
    pub table_idx: usize,
    pub prefix_idx: Option<usize>,
    pub suffix_local_idx: usize,
    pub coefficient: i128,
}

/// Data-driven suffix operation.
///
/// Replaces jolt-instructions `suffix_mle` calls at runtime.
/// The compiler maps each table's suffixes to these ops at module build time;
/// the runtime evaluates them without importing jolt-instructions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SuffixOp {
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
    LeftShiftW { half_xlen: usize },
    LeftShiftWHelper,
    RightShift,
    RightShiftW { half_xlen: usize },
    RightShiftHelper,
    RightShiftWHelper { half_xlen: usize },
    SignExtension { xlen: usize },
    SignExtensionUpperHalf { half_xlen: usize },
    SignExtensionRightOperand { xlen: usize },
    XorRotate { rotation: u32, word_bits: u32 },
    Pow2 { split_bits: usize },
    RightShiftPaddingMask { xlen: usize },
    UpperWord { xlen: usize },
    LowerWord { xlen: usize },
    LowerHalfWord { half_xlen: usize },
    ByteReverseW,
    OverflowBitsZero { xlen: usize },
}

impl SuffixOp {
    /// Evaluate this suffix op on a bitvector of width `len`.
    ///
    /// `bits` contains the low `len` bits of the suffix bitvector.
    /// Returns the same value as the corresponding `suffix_mle`.
    pub fn eval(self, bits: u128, len: usize) -> u64 {
        match self {
            // ── Full-bits operations (no uninterleave) ──
            SuffixOp::Pow2 { split_bits } => {
                if len == 0 {
                    return 1;
                }
                let shift = bits & ((1u128 << split_bits) - 1);
                1u64 << shift
            }
            SuffixOp::RightShiftPaddingMask { xlen } => {
                if len == 0 {
                    return 1;
                }
                let log_xlen = xlen.trailing_zeros() as usize;
                let shift = (bits & ((1u128 << log_xlen) - 1)) as usize;
                1u64 << (xlen - 1 - shift)
            }
            SuffixOp::UpperWord { xlen } => (bits >> xlen) as u64,
            SuffixOp::LowerWord { xlen } => {
                if xlen >= 128 {
                    bits as u64
                } else {
                    (bits % (1u128 << xlen)) as u64
                }
            }
            SuffixOp::LowerHalfWord { half_xlen } => {
                if half_xlen >= 64 {
                    bits as u64
                } else {
                    (bits % (1u128 << half_xlen)) as u64
                }
            }
            SuffixOp::ByteReverseW => (bits as u32).swap_bytes() as u64,
            SuffixOp::OverflowBitsZero { xlen } => ((bits >> xlen) == 0) as u64,
            SuffixOp::Lsb => {
                if len == 0 {
                    1
                } else {
                    (bits & 1) as u64
                }
            }
            SuffixOp::TwoLsbZero => (len == 0 || bits.trailing_zeros() >= 2) as u64,
            SuffixOp::SignExtensionUpperHalf { half_xlen } => {
                if len >= half_xlen {
                    let sign = (bits >> (half_xlen - 1)) & 1;
                    if sign == 1 {
                        ((1u64 << half_xlen) - 1) << half_xlen
                    } else {
                        0
                    }
                } else {
                    1
                }
            }
            SuffixOp::SignExtensionRightOperand { xlen } => {
                if len >= xlen {
                    let sign = (bits >> (xlen - 2)) & 1;
                    if sign == 1 {
                        ((1u128 << xlen) - (1u128 << (xlen / 2))) as u64
                    } else {
                        0
                    }
                } else {
                    1
                }
            }

            // ── Uninterleave-based operations ──
            _ => {
                let (xv, yv, y_len) = uninterleave_suffix(bits, len);
                match self {
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
                        let quot_ones = yv == (1u64 << y_len) - 1;
                        (div_zero && quot_ones) as u64
                    }
                    SuffixOp::ChangeDivisor => ((1u64 << y_len) - 1 == yv && xv == 0) as u64,
                    SuffixOp::ChangeDivisorW => {
                        let yl = y_len.min(32);
                        let xw = xv as u32 as u64;
                        let yw = yv as u32 as u64;
                        ((1u64 << yl) - 1 == yw && xw == 0) as u64
                    }
                    SuffixOp::LeftShift => (xv & !yv).unbounded_shl(leading_ones_u64(yv, y_len)),
                    SuffixOp::LeftShiftW { half_xlen } => {
                        let yl = y_len.min(half_xlen);
                        let yw = yv as u32;
                        let lo = leading_ones_u64(yv & ((1u64 << yl) - 1), yl);
                        ((xv as u32) & !yw).unbounded_shl(lo) as u64
                    }
                    SuffixOp::LeftShiftWHelper => (1u32 << leading_ones_u64(yv, y_len)) as u64,
                    SuffixOp::RightShift => xv.unbounded_shr(trailing_zeros_u64(yv, y_len)),
                    SuffixOp::RightShiftW { half_xlen } => {
                        let tz = trailing_zeros_u64(yv, y_len).min(half_xlen as u32);
                        (xv as u32).unbounded_shr(tz) as u64
                    }
                    SuffixOp::RightShiftHelper => 1u64 << leading_ones_u64(yv, y_len),
                    SuffixOp::RightShiftWHelper { half_xlen } => {
                        let yl = y_len.min(half_xlen);
                        let lo = leading_ones_u64(yv & ((1u64 << yl) - 1), yl);
                        1u64 << lo
                    }
                    SuffixOp::SignExtension { xlen } => {
                        let padding = std::cmp::min(yv.trailing_zeros() as usize, y_len);
                        ((1u128 << xlen) - (1u128 << (xlen - padding))) as u64
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
}

/// Separate interleaved x/y bits into (x_val, y_val, y_len).
///
/// Odd bit positions (1, 3, 5, …) → x; even positions (0, 2, 4, …) → y.
fn uninterleave_suffix(bits: u128, len: usize) -> (u64, u64, usize) {
    let mut x: u64 = 0;
    let mut y: u64 = 0;
    let half = len / 2;
    for i in 0..half.max(len - half) {
        if 2 * i < len {
            y |= (((bits >> (2 * i)) & 1) as u64) << i;
        }
        if 2 * i + 1 < len {
            x |= (((bits >> (2 * i + 1)) & 1) as u64) << i;
        }
    }
    let y_len = len - half;
    (x, y, y_len)
}

/// Count leading 1-bits in a `val` of width `len`.
fn leading_ones_u64(val: u64, len: usize) -> u32 {
    if len == 0 {
        return 0;
    }
    (val as u128)
        .wrapping_shl((128 - len) as u32)
        .leading_ones()
}

/// Count trailing 0-bits in `val`, bounded by `len`.
fn trailing_zeros_u64(val: u64, len: usize) -> u32 {
    std::cmp::min(val.trailing_zeros(), len as u32)
}

/// Byte-reversal for 64-bit words (two 32-bit halves swapped independently).
/// Duplicated from jolt-instructions so the runtime doesn't need that crate.
pub fn rev8w(v: u64) -> u64 {
    let lo = (v as u32).swap_bytes();
    let hi = ((v >> 32) as u32).swap_bytes();
    lo as u64 + ((hi as u64) << 32)
}

/// Default value when a prefix checkpoint is `None`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DefaultVal {
    Zero,
    One,
    /// Signed field literal (e.g. `2 - 2^XLEN` for ChangeDivisor).
    Custom(i128),
}

/// Two-variable expression evaluated from `(r_x, r_y)`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BilinearExpr {
    /// r_x × r_y
    Product,
    /// (1 − r_x) × r_y
    AntiXY,
    /// r_x × (1 − r_y)
    AntiYX,
    /// (1 − r_x) × (1 − r_y)
    NorBit,
    /// r_x×r_y + (1−r_x)×(1−r_y)
    EqBit,
    /// (1−r_x)×r_y + r_x×(1−r_y)
    XorBit,
    /// r_x + r_y − r_x×r_y
    OrBit,
    /// 1 − r_x
    OneMinusX,
    /// 1 − r_y
    OneMinusY,
    /// 1 + r_y
    OnePlusY,
    /// r_x
    X,
    /// r_y
    Y,
}

/// Round-dependent weight function producing a u64 scalar.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum WeightFn {
    /// `2^(word_bits − 1 − ((j − j_offset)/2 + rotation) % word_bits)`.
    /// Covers And/Or/Xor/Andn (`rotation=0`), XorRot/XorRotW (with rotation),
    /// RightOperand, LeftShift.
    Positional {
        rotation: u32,
        word_bits: u32,
        j_offset: usize,
    },
    /// `2^(base − j)` — for the x-term in UpperWord/LowerWord/LowerHalfWord.
    LinearJ { base: usize },
    /// `2^(base − j − 1)` — for the y-term.
    LinearJMinusOne { base: usize },
    /// `2^(j / 2)` — for SignExtension.
    HalfJ,
}

/// Condition on the round index `j` (or `suffix_len`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RoundGuard {
    JEq(usize),
    JLt(usize),
    JGe(usize),
    JGt(usize),
    SuffixLenNonZero,
}

/// Action to produce the updated checkpoint value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CheckpointAction {
    /// `cp *= expr(r_x, r_y)`
    Mul(BilinearExpr),
    /// `cp += weight(j) × expr(r_x, r_y)`
    AddWeighted {
        weight: WeightFn,
        expr: BilinearExpr,
    },
    /// `cp += weight_x(j)×r_x + weight_y(j)×r_y`
    AddTwoTerm {
        x_weight: WeightFn,
        y_weight: WeightFn,
    },
    /// `cp += checkpoints[dep] × expr(r_x, r_y)`
    DepAdd {
        dep: usize,
        dep_default: DefaultVal,
        expr: BilinearExpr,
    },
    /// `cp += checkpoints[dep] × weight(j) × expr(r_x, r_y)`
    DepAddWeighted {
        dep: usize,
        dep_default: DefaultVal,
        weight: WeightFn,
        expr: BilinearExpr,
    },
    /// `cp = cp × mul_expr + add_expr`
    Hybrid {
        mul: BilinearExpr,
        add: BilinearExpr,
    },
    /// SignExtension: `cp += 2^(j/2) × (1−r_y)`;
    /// then if `j == final_j`: `cp *= checkpoints[dep]`
    SignExtAccum { dep: usize, final_j: usize },
    /// Rev8W: `cp += r_x × rev8w(1 << r_x_bit) + r_y × rev8w(1 << r_y_bit)`
    Rev8WAdd { xlen: usize },
    /// Pow2 double-multiply:
    /// `cp *= (1 + (2^(1<<(2×xlen−j)) − 1)×r_x) × (1 + (2^(1<<(2×xlen−j−1)) − 1)×r_y)`
    Pow2DoubleMul { xlen: usize },
    /// Pow2 init: `return 1 + (2^half_pow − 1) × r_y`
    Pow2Init { half_pow: u32 },
    /// `return expr(r_x, r_y)` (ignores current checkpoint)
    Set(BilinearExpr),
    /// `return coeff × expr(r_x, r_y)` (signed coefficient)
    SetScaled { coeff: i128, expr: BilinearExpr },
    /// `return default`
    Const(DefaultVal),
    /// Return checkpoint unchanged.
    Passthrough,
    /// Return `None` (checkpoint becomes unset).
    Null,
}

/// Precomputed checkpoint update rule for one prefix.
///
/// The runtime evaluates `cases` in order; the first matching guard wins.
/// If no guard matches, `fallback` is used.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointRule {
    pub default: DefaultVal,
    pub cases: Vec<(RoundGuard, CheckpointAction)>,
    pub fallback: CheckpointAction,
}

/// One factor in a monomial for a `ScalarExpr`.
///
/// Leaf values pulled from runtime state at evaluation time. Runtime
/// computes each factor as a field element and multiplies them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ValueSource {
    /// `2^k` as a field element.
    Pow2(u32),
    /// `challenges[idx]`.
    Challenge(ChallengeIdx),
    /// `1 − challenges[idx]`.
    OneMinusChallenge(ChallengeIdx),
    /// Checkpoint snapshot `checkpoints[idx].unwrap_or(default)`.
    ///
    /// Reads from the pre-batch snapshot so updates within one
    /// `CheckpointEvalBatch` don't see each other's writes.
    Checkpoint { idx: usize, default: DefaultVal },
    /// `buffers[poly][index]` — read the current evaluation index from a
    /// named buffer. Used when evaluating per-`b` expressions such as
    /// prefix MLEs against precomputed mask polynomials.
    IndexedPoly(PolynomialId),
    /// `values[buffers[index_poly][index]]` — gather a compile-time
    /// constant by using the current-index entry of `index_poly` as an
    /// offset into `values`. Treats `values[k]` as an `i128` coefficient
    /// promoted to the field.
    SelectByIndex {
        index_poly: PolynomialId,
        values: Vec<i128>,
    },
}

/// Signed monomial `coeff × Π factors`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Monomial {
    pub coeff: i128,
    pub factors: Vec<ValueSource>,
}

/// Sum of monomials, evaluated against challenges + checkpoint snapshot.
pub type ScalarExpr = Vec<Monomial>;

/// Update action for a single checkpoint slot in `Op::CheckpointEvalBatch`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CheckpointEvalAction {
    /// Write `expr` evaluated against the pre-batch snapshot.
    Set(ScalarExpr),
    /// Clear the checkpoint (becomes `None`).
    Clear,
}

/// Condition on the remaining (uninterleaved) bits after the current pair.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemainingTest {
    /// I(x == y)
    Equality,
    /// I(x == 0)
    LeftZero,
    /// I(y == 0)
    RightZero,
    /// I(x == 0 && y == all ones)
    LeftZeroRightAllOnes,
    /// Always 1
    Always,
}

/// Integer bitwise operation on the remaining (uninterleaved) operand halves.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntBitOp {
    And,
    AndNot,
    Or,
    Xor,
}

/// Comparison direction for dependent-comparison prefixes.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comparison {
    LessThan,
    GreaterThan,
}

/// How to evaluate `prefix_mle(checkpoints, r_x, c, b, j)` for one prefix.
///
/// The runtime evaluates this without importing jolt-instructions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixMleRule {
    /// Primary checkpoint index.
    pub checkpoint_idx: usize,
    /// Default if primary checkpoint is `None`.
    pub default: DefaultVal,
    /// The evaluation formula.
    pub formula: PrefixMleFormula,
}

/// Evaluation formula for a single prefix MLE.
///
/// Each variant encodes a distinct algebraic family found across the 46
/// prefix types. The evaluator in `checkpoint_eval.rs` interprets these.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PrefixMleFormula {
    /// `cp × pair(x, y) × I(remaining_test)`
    Multiplicative {
        pair: BilinearExpr,
        remaining: RemainingTest,
    },

    /// `cp + weight(j) × pair(x, y) + bitop(rem_x, rem_y) << suffix_shift`
    BitwiseAdditive {
        pair: BilinearExpr,
        weight: WeightFn,
        op: IntBitOp,
        total_bits: usize,
        word_bits: usize,
        /// Non-zero for rotated XOR variants (XorRot16, etc.).
        rotation: usize,
    },

    /// Operand extraction: `cp + weight(position) × x + weight(position+1) × y + remaining_in_word(b)`.
    OperandExtract {
        /// Base for weight computation: `weight = 2^(base_shift - 1 - position)`.
        /// XLEN for upper-word, 2×XLEN for lower-word.
        base_shift: usize,
        has_x: bool,
        has_y: bool,
        word_bits: usize,
        total_bits: usize,
        active_when: Option<RoundGuard>,
        /// When inactive and true, return checkpoint; when false, return zero.
        passthrough_when_inactive: bool,
        /// Upper-word extracts high bits of `b`; lower-word shifts full `b`.
        is_upper: bool,
    },

    /// `lt_cp + eq_cp × anti_pair + eq_cp × eq_pair × I(rem_x cmp rem_y)`
    DependentComparison {
        eq_idx: usize,
        eq_default: DefaultVal,
        cmp: Comparison,
    },

    /// Complex comparison (NegDivGtRem, PosRemLtDiv).
    /// Phases: j < start_round → 0; j = start_round..start_round+4 → init;
    /// j ≥ start_round+4 → standard dependent comparison.
    ComplexComparison {
        eq_idx: usize,
        cmp: Comparison,
        sign_pair_j0: BilinearExpr,
        init_cmp: Comparison,
        mul_anti: BilinearExpr,
        start_round: usize,
    },

    /// Eq-like with sign gates (PosRemEqDiv, NegDivEqRem).
    /// j < start_round → 0; j = start_round,start_round+1 → sign_pair with equality test;
    /// j ≥ start_round+2 → cp × base_pair × I(equality).
    SignGatedMultiplicative {
        sign_pair: BilinearExpr,
        base_pair: BilinearExpr,
        remaining: RemainingTest,
        start_round: usize,
    },

    /// LeftShift family.
    LeftShift {
        helper_idx: usize,
        helper_default: DefaultVal,
        word_bits: usize,
        start_round: usize,
    },

    /// LeftShiftHelper: `cp × (1 + y) × 2^(leading_ones of remaining y)`.
    LeftShiftHelper {
        word_bits: usize,
        start_round: usize,
    },

    /// RightShift family.
    RightShift {
        word_bits: usize,
        start_round: usize,
    },

    /// MSB extraction: c or pop_msb at j=start, r_x or c at j=start+1, then checkpoint.
    Msb {
        msb_idx: usize,
        start_round: usize,
        /// true = left operand (x-variable), false = right operand (y-variable).
        is_left: bool,
    },

    /// LSB extraction.
    Lsb { total_bits: usize },

    /// Two-LSB extraction.
    TwoLsb { total_bits: usize },

    /// Sign extension.
    SignExtension { msb_idx: usize, word_bits: usize },

    /// Sign extension upper half.
    SignExtUpperHalf { word_bits: usize },

    /// Sign extension right operand.
    SignExtRightOp { word_bits: usize },

    /// Power of 2.
    Pow2 {
        word_mask: usize,
        log_word_bits: u32,
        total_bits: usize,
    },

    /// Rev8W: byte-reversal additive.
    Rev8W { xlen: usize },

    /// ChangeDivisor (sparse multiplicative).
    ChangeDivisor {
        word_bits: usize,
        start_round: usize,
    },

    /// NegativeDivisorZeroRemainder.
    NegDivZeroRem {
        word_bits: usize,
        start_round: usize,
    },

    /// RightOperand(W): `cp + [j>start && j odd] c×2^(word_bits-1-j/2) + uninterleaved_y_suffix`.
    /// `suffix_guard`: only add y-suffix when suffix_len < this value. Use total_bits for "always".
    RightOperandExtract {
        word_bits: usize,
        start_round: usize,
        total_bits: usize,
        suffix_guard: usize,
    },

    /// Overflow bits zero (multiplicative, active in specific round range).
    OverflowBitsZero {
        pair: BilinearExpr,
        word_bits: usize,
        total_bits: usize,
    },
}

/// Configuration for a stateful sumcheck instance.
///
/// Contains structural parameters that the runtime needs to initialize and
/// drive a multi-phase address-decomposition state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceConfig {
    pub kernel: usize,
    pub total_address_bits: usize,
    /// Bits per sub-phase (LOG_K / num_phases).
    pub chunk_bits: usize,
    /// Number of sub-phases in the address decomposition (8 or 16).
    pub num_phases: usize,
    /// Log₂ of the virtual RA polynomial chunk size.
    pub ra_virtual_log_k_chunk: usize,
    /// Challenge index for γ (instruction read-RAF batching).
    pub gamma: ChallengeIdx,
    /// Challenge indices for r_reduction (log_T entries, BIG_ENDIAN).
    pub r_reduction: Vec<ChallengeIdx>,
    /// PolynomialIds for materialized RA polys at address→cycle transition.
    pub output_ra_polys: Vec<PolynomialId>,
    /// PolynomialId for the combined_val polynomial.
    pub output_combined_val: PolynomialId,
    /// Number of instruction tables.
    pub num_tables: usize,
    /// Number of suffix polynomials per table.
    pub suffixes_per_table: Vec<usize>,
    /// Combine matrix entries for read-checking.
    pub combine_entries: Vec<CombineEntry>,
    /// Precomputed suffix evaluations at the empty point (suffix_len=0).
    /// `suffix_at_empty[table][suffix]` = suffix_mle(LookupBits::new(0, 0)).
    pub suffix_at_empty: Vec<Vec<u64>>,
    /// Data-driven suffix operations: `suffix_ops[table][suffix]`.
    /// Replaces runtime calls to `suffix_mle`.
    pub suffix_ops: Vec<Vec<SuffixOp>>,
    /// Number of prefix checkpoints.
    pub num_prefixes: usize,
    /// Data-driven checkpoint update rules: `checkpoint_rules[prefix]`.
    /// Replaces runtime calls to `update_prefix_checkpoint`.
    pub checkpoint_rules: Vec<CheckpointRule>,
    /// Data-driven prefix MLE evaluation rules: `prefix_mle_rules[prefix]`.
    /// Replaces runtime calls to `prefix_mle` in `compute_read_checking`.
    pub prefix_mle_rules: Vec<PrefixMleRule>,
    /// Per-round pre-lowered prefix MLE expressions.
    ///
    /// `prefix_lowered[round][rule]` holds the c=0/c=1 `ScalarExpr` pair plus
    /// the mask-buffer bindings for that (round, rule). Round geometry
    /// (j, b_len, r_x ChallengeIdx) is compile-time constant; baking the
    /// lowering here keeps `lower_prefix_mle` out of the runtime hot path.
    pub prefix_lowered: Vec<PrefixLoweredRound>,
    /// Challenge indices for the 3 RAF registry checkpoints:
    /// \[0\] = p_right, \[1\] = p_left, \[2\] = p_identity.
    pub registry_checkpoint_slots: [ChallengeIdx; 3],
}

/// Pre-lowered prefix MLE expressions for a single sumcheck round.
///
/// Emitted by the compiler at InstanceConfig-build time. Each entry holds the
/// `ScalarExpr` for both c-sides and the `(MaskRole, PolynomialId)` bindings
/// needed to evaluate it. The runtime evaluates these expressions directly —
/// no protocol-specific match arms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixLoweredRound {
    /// `b_len` at this round (suffix poly has `2^(b_len+1)` entries).
    pub b_len: usize,
    /// Per-rule c=0 lowering.
    pub c0: Vec<ScalarExpr>,
    /// Per-rule c=1 lowering.
    pub c1: Vec<ScalarExpr>,
    /// Per-rule mask bindings used by c0. Roles are evaluated via
    /// `compute_mask_value(role, b, b_len)` at runtime.
    pub masks_c0: Vec<Vec<(PolynomialId, crate::prefix_mle_lowering::MaskRole)>>,
    /// Per-rule mask bindings used by c1.
    pub masks_c1: Vec<Vec<(PolynomialId, crate::prefix_mle_lowering::MaskRole)>>,
    /// ChallengeIdx carrying `r_x` on odd rounds (`None` on even rounds).
    pub r_x: Option<ChallengeIdx>,
}

impl InstanceConfig {
    /// All device buffer IDs that must be bound each round.
    pub fn bindable_polys(&self) -> Vec<PolynomialId> {
        let mut polys = Vec::new();
        for t in 0..self.num_tables {
            for s in 0..self.suffixes_per_table[t] {
                polys.push(PolynomialId::InstanceSuffix(t, s));
            }
        }
        for c in 0..3 {
            polys.push(PolynomialId::InstanceQ(c, 0));
            polys.push(PolynomialId::InstanceQ(c, 1));
            // P only has one buffer per category (no second half).
            polys.push(PolynomialId::InstanceP(c, 0));
        }
        polys
    }
}

impl Schedule {
    pub fn compute_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_compute()).count()
    }

    pub fn pcs_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_pcs()).count()
    }

    pub fn orchestration_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_orchestration()).count()
    }
}

/// Definition of a single sumcheck kernel (compiled by the backend at link time).
///
/// Combines a [`KernelSpec`] (what the backend compiles) with runtime context
/// (where to get inputs, how many rounds). The spec captures the algorithmic
/// decisions; the rest is orchestration metadata for the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDef {
    /// Backend compilation target: formula, iteration pattern, eval grid, binding order.
    pub spec: KernelSpec,
    /// Data provenance for each kernel input. `inputs[i]` describes where
    /// the data for the i-th input comes from at runtime. The first
    /// `spec.formula.num_inputs` entries are formula value columns; any
    /// remaining entries are extra inputs required by the iteration pattern
    /// (e.g., tensor eq buffers, sparse key column).
    pub inputs: Vec<InputBinding>,
    /// Total sumcheck rounds for this kernel.
    pub num_rounds: usize,
    /// Instance-type config for subprotocol kernels (address decomposition, booleanity,
    /// HwReduction). `None` for standard compute kernels (Dense/Sparse/Domain).
    pub instance_config: Option<InstanceConfig>,
}

/// Data provenance for a kernel input.
///
/// Each variant describes where the runtime obtains buffer data:
/// - [`Provided`](InputBinding::Provided) — loaded from the [`BufferProvider`]
///   (witness data, preprocessed tables). The poly index references `Module.polys`.
/// - Table variants — built on-device from challenge values. The runtime calls
///   the appropriate backend primitive (e.g., `eq_table`) using the
///   challenge values at the given indices. The poly index is the storage slot
///   in `Module.polys` for lifecycle tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputBinding {
    /// Buffer loaded from the provider (witness / preprocessed).
    Provided { poly: PolynomialId },
    /// Eq table built on-device: `eq(r, x) = Π(rᵢxᵢ + (1−rᵢ)(1−xᵢ))`.
    /// Challenge indices whose values form the evaluation point.
    EqTable {
        poly: PolynomialId,
        challenges: Vec<ChallengeIdx>,
    },
    /// Eq-plus-one table: `eq(r, x) · (1 + r_{n-1})`.
    EqPlusOneTable {
        poly: PolynomialId,
        challenges: Vec<ChallengeIdx>,
    },
    /// Less-than table from challenge points.
    LtTable {
        poly: PolynomialId,
        challenges: Vec<ChallengeIdx>,
    },
    /// Project a T×K source polynomial onto K elements via cycle eq weighting.
    ///
    /// Computes `result[k] = Σ_t eq(r_cycle, t) · source[t * outer_size + k]`
    /// where `r_cycle` comes from the challenge slots at runtime.
    ///
    /// Used when a downstream sumcheck instance needs the cycle-bound
    /// projection of a larger polynomial (e.g., RAM RAF RA from RamCombinedRa).
    EqProject {
        /// Polynomial ID for the projected result (storage/lifecycle).
        poly: PolynomialId,
        /// Source T×K polynomial to project from.
        source: PolynomialId,
        /// Challenge indices forming the cycle eq point.
        challenges: Vec<ChallengeIdx>,
        /// Size of the inner (cycle) dimension.
        inner_size: usize,
        /// Size of the outer (address) dimension.
        outer_size: usize,
    },
    /// Eq-gather: build eq table from challenges, then gather per-element
    /// values using integer indices from a source polynomial.
    ///
    /// Computes `result[j] = eq(r, index[j])` where `r` is the challenge
    /// point and `index[j]` are per-cycle lookup indices from the provider.
    ///
    /// Used for register write-address indicators (eq(r_address, rd[j]))
    /// and RAM access indicators (eq(r_address, addr[j])).
    EqGather {
        /// Polynomial ID for the gathered result (T elements).
        poly: PolynomialId,
        /// Challenge indices forming the eq point (log₂K entries).
        eq_challenges: Vec<ChallengeIdx>,
        /// Source of per-cycle integer indices (T entries, each in 0..K-1).
        /// The provider materializes this from trace data.
        indices: PolynomialId,
    },
    /// Pushforward of eq polynomial through an index mapping.
    ///
    /// Computes `result[k] = Σ_j eq(r, j) × 1{indices[j] == k}` where
    /// `r` is the challenge point and `indices[j]` maps cycle `j` to an
    /// address-space bin `k`. The result has `output_size` elements.
    ///
    /// Used for BytecodeReadRaf F[stage] tables: each F[s][k] accumulates
    /// the eq weight of all cycles whose PC maps to bytecode index k.
    EqPushforward {
        poly: PolynomialId,
        eq_challenges: Vec<ChallengeIdx>,
        indices: PolynomialId,
        output_size: usize,
    },
    /// Multiply a source polynomial element-wise by a challenge value.
    ///
    /// Computes `result[k] = challenges[challenge]^power × source[k]`.
    /// Used for gamma-weighting preprocessed polynomials (e.g., entry_gamma × f_expected).
    ScaleByChallenge {
        poly: PolynomialId,
        source: PolynomialId,
        challenge: ChallengeIdx,
        power: u8,
    },
    /// Transpose a source polynomial from row-major to column-major layout.
    ///
    /// Source layout: `src[row * cols + col]` (rows × cols elements).
    /// Result layout: `dst[col * rows + row]` (cols × rows elements).
    ///
    /// Used by Booleanity to rearrange RA polynomials from address-major
    /// `[k * T + j]` to cycle-major `[j * K + k]` so that LowToHigh binding
    /// binds address variables first (matching jolt-core's Phase 1 → Phase 2).
    Transpose {
        poly: PolynomialId,
        source: PolynomialId,
        rows: usize,
        cols: usize,
    },
}

impl InputBinding {
    /// The poly slot this binding writes to / reads from in the buffer table.
    pub fn poly(&self) -> PolynomialId {
        match self {
            InputBinding::Provided { poly }
            | InputBinding::EqTable { poly, .. }
            | InputBinding::EqPlusOneTable { poly, .. }
            | InputBinding::LtTable { poly, .. }
            | InputBinding::EqProject { poly, .. }
            | InputBinding::EqGather { poly, .. }
            | InputBinding::EqPushforward { poly, .. }
            | InputBinding::ScaleByChallenge { poly, .. }
            | InputBinding::Transpose { poly, .. } => *poly,
        }
    }
}

/// Fiat-Shamir domain-separation tag for transcript operations.
///
/// Each variant maps to a concrete byte string that the runtime absorbs
/// before the payload. Using an enum (not raw strings) ensures the Module
/// is self-contained and tag mismatches are caught at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainSeparator {
    /// Polynomial commitment: `b"commitment"`.
    Commitment,
    /// Untrusted advice commitment: `b"untrusted_advice"`.
    UntrustedAdvice,
    /// Trusted advice commitment: `b"trusted_advice"`.
    TrustedAdvice,
    /// Full univariate-skip polynomial: `b"uniskip_poly"`.
    UniskipPoly,
    /// Compressed sumcheck round polynomial: `b"sumcheck_poly"`.
    SumcheckPoly,
    /// Batched sumcheck input claim: `b"sumcheck_claim"`.
    SumcheckClaim,
    /// Polynomial opening evaluation: `b"opening_claim"`.
    OpeningClaim,
    /// RAM value check batching gamma: `b"ram_val_check_gamma"`.
    RamValCheckGamma,
}

impl DomainSeparator {
    /// The concrete byte string for Fiat-Shamir domain separation.
    pub fn as_bytes(&self) -> &'static [u8] {
        match self {
            Self::Commitment => b"commitment",
            Self::UntrustedAdvice => b"untrusted_advice",
            Self::TrustedAdvice => b"trusted_advice",
            Self::UniskipPoly => b"uniskip_poly",
            Self::SumcheckPoly => b"sumcheck_poly",
            Self::SumcheckClaim => b"sumcheck_claim",
            Self::OpeningClaim => b"opening_claim",
            Self::RamValCheckGamma => b"ram_val_check_gamma",
        }
    }
}

/// A single prover operation in the schedule.
///
/// Three categories:
/// - **Compute** — dispatched to [`ComputeBackend`] via compiled kernels.
/// - **PCS** — dispatched to [`CommitmentScheme`] (commit, reduce, open).
/// - **Orchestration** — zero-cost host bookkeeping (transcript, lifecycle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    /// Compute one sumcheck round polynomial.
    ///
    /// Round 0 (`bind_challenge: None`): reduce only.
    /// Rounds 1+ (`bind_challenge: Some(ch)`): fused bind-at-challenge + reduce.
    SumcheckRound {
        kernel: usize,
        round: usize,
        bind_challenge: Option<ChallengeIdx>,
    },
    /// Initialize a batched sumcheck round: zero the combined accumulator and
    /// update per-instance claims from the previous round's evaluations.
    BatchRoundBegin {
        batch: BatchIdx,
        round: usize,
        max_evals: usize,
        bind_challenge: Option<ChallengeIdx>,
    },
    /// Inactive instance contribution: add `coeff * (claim / 2)` to all eval
    /// slots in the combined accumulator, then halve the stored claim.
    BatchInactiveContribution {
        batch: BatchIdx,
        instance: InstanceIdx,
    },
    /// Materialize a single kernel input buffer.
    ///
    /// Unconditionally builds/uploads the buffer described by `binding`.
    /// Emitted by the compiler at the exact schedule position where the
    /// buffer is needed — no runtime skip logic.
    Materialize { binding: InputBinding },
    /// Materialize a kernel input, but skip if a buffer of the expected
    /// size already exists (e.g., produced by a prior compute op like
    /// `MaterializeRA` / `MaterializeCombinedVal`).
    ///
    /// Only used for `Provided` bindings at instance activation where
    /// a cross-instance compute op may have already produced the buffer.
    MaterializeUnlessFresh {
        binding: InputBinding,
        expected_size: usize,
    },
    /// Materialize a kernel input, but only if no buffer exists for this
    /// poly. Used at phase transitions where bound-down buffers from the
    /// previous phase (or other instances) should be preserved.
    MaterializeIfAbsent { binding: InputBinding },
    /// Build the outer eq table for a segmented phase and store it in
    /// the runtime's per-instance segmented state.
    MaterializeSegmentedOuterEq {
        batch: BatchIdx,
        instance: InstanceIdx,
        segmented: SegmentedConfig,
    },
    /// Bind the previous phase's kernel inputs at a challenge.
    /// Emitted at phase transitions (before resolving the new phase's inputs).
    InstanceBindPreviousPhase {
        batch: BatchIdx,
        instance: InstanceIdx,
        kernel: usize,
        challenge: ChallengeIdx,
    },
    /// Capture a scalar from a fully-bound 1-element device buffer into a
    /// challenge slot. Bridges phase boundaries: an intermediate value computed
    /// in one phase becomes a challenge constant for the next phase's formula.
    CaptureScalar {
        poly: PolynomialId,
        challenge: ChallengeIdx,
    },
    /// Standard dense reduce for one instance within a batched round.
    /// Stores the per-instance evaluations for later accumulation.
    InstanceReduce {
        batch: BatchIdx,
        instance: InstanceIdx,
        kernel: usize,
    },
    /// Segmented reduce for one instance (mixed-dimensional inputs).
    /// Uses the outer eq table to weight inner-dimension kernel evaluations.
    InstanceSegmentedReduce {
        batch: BatchIdx,
        instance: InstanceIdx,
        kernel: usize,
        round_within_phase: usize,
        segmented: SegmentedConfig,
    },
    /// Bind kernel inputs for an active instance within a round.
    /// Emitted for rounds after the first within a phase.
    InstanceBind {
        batch: BatchIdx,
        instance: InstanceIdx,
        kernel: usize,
        challenge: ChallengeIdx,
    },
    /// Extrapolate lower-degree instance evals to `max_evals` via interpolation,
    /// then accumulate `coeff * evals[i]` into the combined polynomial.
    BatchAccumulateInstance {
        batch: BatchIdx,
        instance: InstanceIdx,
        max_evals: usize,
        num_evals: usize,
    },
    /// Finalize a batched round: store the combined evaluations as
    /// `last_round_coeffs` for subsequent `AbsorbRoundPoly`.
    BatchRoundFinalize { batch: BatchIdx },

    /// Initialize instance weights from the eq polynomial at the r_reduction point.
    /// Only emitted for phase 0 of address-decomposition instances.
    InitInstanceWeights {
        r_reduction: Vec<ChallengeIdx>,
        num_prefixes: usize,
    },
    /// Multiply instance weights by expanding-table lookups at a phase boundary.
    /// Emitted for phase > 0 of address-decomposition instances.
    UpdateInstanceWeights {
        expanding_table: PolynomialId,
        chunk_bits: usize,
        num_phases: usize,
        phase: usize,
    },
    /// Scatter weighted suffix evaluations into per-table polynomial buffers.
    SuffixScatter { kernel: usize, phase: usize },
    /// Compute RAF quotient (Q) buffers from lookup keys and instance weights.
    QBufferScatter { kernel: usize, phase: usize },
    /// Materialize RAF product (P) buffers from registry checkpoint scalars.
    MaterializePBuffers { kernel: usize },
    /// Initialize an expanding eq table with a single 1.0 entry.
    InitExpandingTable { table: PolynomialId, size: usize },
    /// Read-checking reduce: evaluate prefix MLEs × suffix polys via combine matrix.
    /// Stores partial result in `instance_read_checking_evals`.
    ReadCheckingReduce {
        kernel: usize,
        round: usize,
        r_x_challenge: Option<ChallengeIdx>,
    },
    /// RAF reduce: compute p × q contribution weighted by gamma.
    /// Combines with read-checking result and stores final round evaluations.
    RafReduce {
        batch: BatchIdx,
        instance: InstanceIdx,
        kernel: usize,
    },
    /// Materialize RA polynomials from expanding tables and lookup keys.
    MaterializeRA { kernel: usize },
    /// Materialize combined_val polynomial from checkpoints + combine matrix.
    MaterializeCombinedVal { kernel: usize },

    /// Weighted linear combination of provider/device buffers with challenge scalars.
    ///
    /// `result[i] = Σ_j (challenge_j^power_j × source_j[i]) + identity_scale × i`
    /// optionally scaled by `overall_scale`. Sources are resolved from device_buffers
    /// if present, else from the provider. Generic — no protocol knowledge.
    WeightedSum {
        output: PolynomialId,
        terms: Vec<(PolynomialId, ChallengeIdx, u8)>,
        identity_term: Option<(ChallengeIdx, u8)>,
        overall_scale: Option<(ChallengeIdx, u8)>,
    },

    /// Update an expanding eq table by consuming a challenge value.
    ///
    /// Doubles the active portion of the buffer: for each entry `v[i]` in
    /// `0..current_len`, produces `v[2i] = v[i] − r·v[i]` and `v[2i+1] = r·v[i]`.
    /// After the update, the active length is `2 × current_len`.
    ExpandingTableUpdate {
        table: PolynomialId,
        challenge: ChallengeIdx,
        /// Active length before this update (compiler knows from round geometry).
        current_len: usize,
    },
    /// Update a batch of instance checkpoints using compiled scalar expressions.
    ///
    /// Atomic: reads all inputs from a snapshot of `instance_checkpoints`
    /// taken before any writes, so `DepAdd`-style rules that read one
    /// checkpoint while updating another observe the pre-batch state.
    CheckpointEvalBatch {
        updates: Vec<(usize, CheckpointEvalAction)>,
    },

    /// Extract polynomial evaluation.
    Evaluate { poly: PolynomialId, mode: EvalMode },
    /// Bind polynomial buffers at a challenge value (post-sumcheck survivors).
    Bind {
        polys: Vec<PolynomialId>,
        challenge: ChallengeIdx,
        order: BindingOrder,
    },
    /// Project polynomial buffers by evaluating Lagrange basis at a challenge,
    /// collapsing the constraint dimension after a univariate skip round.
    ///
    /// Transforms each buffer from `num_cycles × stride` entries to
    /// `num_cycles × num_groups` entries:
    ///
    /// ```text
    /// result[c * G + g] = scale · Σ_{k=0}^{D-1} L_k(r) · buf[c * stride + offsets[g] + k]
    /// ```
    ///
    /// where `D` = `domain_size`, `G` = `group_offsets.len()`,
    /// `L_k` are Lagrange basis polynomials over the symmetric domain
    /// `{domain_start, …, domain_start + D - 1}`, `r = challenges[challenge]`,
    /// and `scale = L_kernel(challenges[kernel_tau], r)` if `kernel_tau` is set (1 otherwise).
    LagrangeProject {
        polys: Vec<PolynomialId>,
        challenge: ChallengeIdx,
        domain_size: usize,
        domain_start: i64,
        stride: usize,
        group_offsets: Vec<usize>,
        /// When set, all projected values are multiplied by the Lagrange kernel
        /// `L(challenges[kernel_tau], challenges[challenge])` over the projection domain.
        /// This folds the uniskip kernel factor into the projected buffers.
        kernel_tau: Option<ChallengeIdx>,
    },
    /// Interleave-duplicate polynomial buffers: `buf'[2i] = buf'[2i+1] = buf[i]`.
    ///
    /// Extends a polynomial that does not depend on a new low-order variable
    /// (e.g. the streaming variable in the outer Spartan remaining sumcheck).
    /// The resulting buffer is twice as large and ready for standard dense
    /// sumcheck rounds that bind the new variable first (LowToHigh order).
    DuplicateInterleave { polys: Vec<PolynomialId> },
    /// Regroup constraint buffers for the group-split uniskip.
    ///
    /// Transforms Az/Bz from flat layout `[cycle * old_stride + constraint]`
    /// into interleaved layout `[(2 * cycle + group) * new_stride + k]`
    /// where `k` is the constraint index within the group.
    ///
    /// `group_indices[0]` constraints form group 0, `group_indices[1]` form group 1.
    /// Groups are zero-padded to `new_stride`. The group dimension is
    /// INTERLEAVED (group bit at the LOW end / LSB) so that the eq table's
    /// group-selection variable is the first variable bound in LowToHigh order,
    /// matching jolt-core's GruenSplitEqPolynomial layout.
    RegroupConstraints {
        polys: Vec<PolynomialId>,
        /// Indices of original constraints in each group (within `old_stride`).
        group_indices: Vec<Vec<usize>>,
        old_stride: usize,
        new_stride: usize,
        num_cycles: usize,
    },

    /// Commit polynomials, absorb commitments into transcript,
    /// capture raw data and hints for later opening proofs.
    Commit {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
        /// Total multilinear variables for the PCS grid.
        /// Polynomials shorter than `2^num_vars` are zero-padded.
        /// Determines Dory matrix dimensions via balanced (sigma, nu) split.
        num_vars: usize,
    },
    /// Commit polynomials via streaming (chunked) PCS.
    ///
    /// Uses `StreamingCommitment::begin/feed/finish` instead of
    /// `CommitmentScheme::commit` — enables large polynomials to be
    /// committed without holding the full evaluation table in a single
    /// contiguous allocation on the PCS side.
    CommitStreaming {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
        /// Chunk size (evaluations per feed call).
        chunk_size: usize,
        /// Total multilinear variables for the PCS grid.
        num_vars: usize,
    },
    /// RLC-reduce all accumulated opening claims via transcript challenges.
    ReduceOpenings,
    /// Generate PCS opening proofs for all reduced claims.
    Open,

    /// Absorb public instance data into the transcript.
    Preamble,
    /// Begin a new verifier stage (for incremental proof assembly).
    BeginStage { index: usize },
    /// Interpolate round evals → monomial coefficients, absorb into transcript.
    AbsorbRoundPoly {
        num_coeffs: usize,
        tag: DomainSeparator,
        encoding: RoundPolyEncoding,
    },
    /// Record polynomial evaluations in the stage proof for the verifier.
    ///
    /// Pushes values to `stage.evals` so the verifier can read them via
    /// `VerifierOp::RecordEvals`. Does not touch the transcript.
    RecordEvals { polys: Vec<PolynomialId> },
    /// Absorb polynomial evaluations into the Fiat-Shamir transcript.
    ///
    /// Transcript-only — does not record in the stage proof. Pair with
    /// `RecordEvals` when the verifier also needs the values.
    AbsorbEvals {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
    },
    /// Evaluate a [`ClaimFormula`] against current evaluations/challenges and
    /// absorb the resulting scalar into the Fiat-Shamir transcript.
    AbsorbInputClaim {
        formula: ClaimFormula,
        tag: DomainSeparator,
        /// Batch index and instance index within the batch.
        /// Used to initialize the runtime's per-instance claim for
        /// inactive-round constant contributions.
        batch: BatchIdx,
        instance: InstanceIdx,
        /// Pre-computed scale: `val * 2^inactive_scale_bits`. The compiler
        /// knows `max_rounds - inst.num_rounds()` at emit time.
        inactive_scale_bits: usize,
    },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: ChallengeIdx },
    /// Compute a derived challenge: `challenges[target] = challenges[base]^exponent`.
    ComputePower {
        target: ChallengeIdx,
        base: ChallengeIdx,
        exponent: u64,
    },
    /// Append a domain separator label (empty payload) to the transcript.
    AppendDomainSeparator { tag: DomainSeparator },
    /// Accumulate a PCS opening claim: (poly data, eval point from stage).
    CollectOpeningClaim {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// Scale an evaluation by `∏(1 − ch[i])` (Lagrange zero selector).
    /// Used for dense (cycle-only) polynomials whose Dory matrix embedding
    /// includes a `eq(r_addr, 0)` factor.
    ScaleEval {
        poly: PolynomialId,
        factor_challenges: Vec<ChallengeIdx>,
    },
    /// Accumulate a PCS opening claim with an explicit challenge-index point.
    /// Unlike `CollectOpeningClaim`, the point spans multiple stages
    /// (e.g. `[r_address_stage7, r_cycle_stage6]`).
    CollectOpeningClaimAt {
        poly: PolynomialId,
        point_challenges: Vec<ChallengeIdx>,
        /// When set, the polynomial's evaluation table is zero-padded to
        /// `2^committed_num_vars` elements for the RLC combination.
        committed_num_vars: Option<usize>,
    },
    /// Post-proof transcript binding: absorb opening point + joint eval.
    /// Calls `PCS::bind_opening_inputs(transcript, point, eval)`.
    BindOpeningInputs { point_challenges: Vec<ChallengeIdx> },
    /// Evaluate a preprocessed polynomial's MLE at a challenge-derived point.
    ///
    /// Materializes the polynomial from the provider, evaluates the MLE at
    /// `[challenges[i] for i in at_challenges]`, and stores the result in
    /// `state.evaluations[store_as]`. Used for init_eval in RamValCheck.
    EvaluatePreprocessed {
        source: PolynomialId,
        at_challenges: Vec<ChallengeIdx>,
        store_as: PolynomialId,
    },
    /// Release a device buffer (GPU memory).
    ReleaseDevice { poly: PolynomialId },
    /// Release host-side polynomial data (provider memory).
    /// Emitted after `ReduceOpenings` when evaluation tables are no longer needed.
    ReleaseHost { polys: Vec<PolynomialId> },
    /// Alias one evaluation under another polynomial ID.
    /// Runtime: `state.evaluations[to] = state.evaluations[from]`.
    AliasEval {
        from: PolynomialId,
        to: PolynomialId,
    },
    /// Bind carry buffers for a phase.  These are extra polynomial buffers
    /// that are not kernel inputs but must be bound at the same cadence
    /// so they are the right size when the next phase begins.
    BindCarryBuffers {
        polys: Vec<PolynomialId>,
        challenge: ChallengeIdx,
        order: BindingOrder,
    },
}

/// How to compute and encode the round polynomial for transcript absorption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoundPolyEncoding {
    /// Standard sumcheck: interpolate evaluations at {0, 1, ..., num_coeffs-1}
    /// to monomial coefficients, send compressed (skip c1).
    Compressed,
    /// Univariate skip: convolve composition evaluations with the Lagrange
    /// kernel polynomial, send all coefficients (no compression).
    Uniskip {
        domain_size: usize,
        domain_start: i64,
        tau_challenge: ChallengeIdx,
        zero_base: bool,
    },
}

/// How the runtime should extract a polynomial evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalMode {
    /// Buffer is fully bound (1 element). Direct scalar read.
    FullyBound,
    /// Buffer has 2 elements after n-1 bind rounds. Interpolate at last
    /// squeezed challenge: `buf[0] + r * (buf[1] - buf[0])`.
    FinalBind,
    /// No buffer — evaluate the last round polynomial at the last squeezed challenge.
    RoundPoly,
}

impl Op {
    pub fn is_compute(&self) -> bool {
        matches!(
            self,
            Op::SumcheckRound { .. }
                | Op::InstanceReduce { .. }
                | Op::InstanceSegmentedReduce { .. }
                | Op::InstanceBind { .. }
                | Op::ReadCheckingReduce { .. }
                | Op::RafReduce { .. }
                | Op::Evaluate { .. }
                | Op::Bind { .. }
                | Op::LagrangeProject { .. }
                | Op::DuplicateInterleave { .. }
                | Op::RegroupConstraints { .. }
        )
    }

    pub fn is_pcs(&self) -> bool {
        matches!(
            self,
            Op::Commit { .. }
                | Op::CommitStreaming { .. }
                | Op::ReduceOpenings
                | Op::Open
                | Op::BindOpeningInputs { .. }
        )
    }

    pub fn is_orchestration(&self) -> bool {
        matches!(
            self,
            Op::Preamble
                | Op::BeginStage { .. }
                | Op::AbsorbRoundPoly { .. }
                | Op::RecordEvals { .. }
                | Op::AbsorbEvals { .. }
                | Op::AbsorbInputClaim { .. }
                | Op::Squeeze { .. }
                | Op::ComputePower { .. }
                | Op::AppendDomainSeparator { .. }
                | Op::CollectOpeningClaim { .. }
                | Op::ScaleEval { .. }
                | Op::CollectOpeningClaimAt { .. }
                | Op::EvaluatePreprocessed { .. }
                | Op::ReleaseDevice { .. }
                | Op::ReleaseHost { .. }
                | Op::AliasEval { .. }
                | Op::BatchRoundBegin { .. }
                | Op::BatchInactiveContribution { .. }
                | Op::Materialize { .. }
                | Op::MaterializeUnlessFresh { .. }
                | Op::MaterializeIfAbsent { .. }
                | Op::MaterializeSegmentedOuterEq { .. }
                | Op::InstanceBindPreviousPhase { .. }
                | Op::CaptureScalar { .. }
                | Op::BatchAccumulateInstance { .. }
                | Op::BatchRoundFinalize { .. }
                | Op::ExpandingTableUpdate { .. }
                | Op::InitInstanceWeights { .. }
                | Op::UpdateInstanceWeights { .. }
                | Op::SuffixScatter { .. }
                | Op::QBufferScatter { .. }
                | Op::MaterializePBuffers { .. }
                | Op::InitExpandingTable { .. }
                | Op::MaterializeRA { .. }
                | Op::MaterializeCombinedVal { .. }
                | Op::WeightedSum { .. }
        )
    }
}

/// Verifier execution schedule: a flat sequence of ops for Fiat-Shamir replay
/// and claim checking.
///
/// The verifier is a generic interpreter: it walks `ops` in order, one match
/// arm per variant, mirroring the prover's flat `Vec<Op>` execution model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSchedule {
    pub ops: Vec<VerifierOp>,
    /// Total number of challenge slots.
    pub num_challenges: usize,
    /// Total number of polynomial slots (for evaluation tracking).
    pub num_polys: usize,
    /// Total number of sumcheck stages (for preallocating point/eval vectors).
    pub num_stages: usize,
}

/// A single verifier operation in the schedule.
///
/// Mirrors the prover's [`Op`] enum: the verifier walks a flat `Vec<VerifierOp>`
/// in a single match loop. The compiler places each op at the exact position
/// where its data dependencies are satisfied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifierOp {
    /// Absorb prover config into transcript (matches prover's `Preamble`).
    Preamble,
    /// Advance stage proof cursor; subsequent `VerifySumcheck` and `RecordEvals`
    /// read from this stage proof.
    BeginStage,
    /// Absorb next commitment from proof, store in commitment map.
    AbsorbCommitment {
        poly: PolynomialId,
        tag: DomainSeparator,
    },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: ChallengeIdx },
    /// Append a domain separator label (empty payload) to the transcript.
    AppendDomainSeparator { tag: DomainSeparator },
    /// Absorb a round polynomial from the current stage proof into transcript.
    ///
    /// Reads the next round polynomial (at `round_poly_cursor`) from the stage
    /// proof, absorbs its coefficients into the Fiat-Shamir transcript, and
    /// advances the cursor. Used for uniskip rounds that are not part of the
    /// batched sumcheck verified by [`VerifySumcheck`].
    AbsorbRoundPoly {
        num_coeffs: usize,
        tag: DomainSeparator,
    },
    /// Verify batched sumcheck from current stage proof.
    ///
    /// Computes combined claim from instance `input_claim` formulas, verifies
    /// sumcheck rounds, stores `final_eval` and challenge point for the stage.
    ///
    /// When `batch_challenges` is non-empty, the handler:
    /// 1. Evaluates each instance's `input_claim` formula
    /// 2. Absorbs each claim into transcript with `claim_tag`
    /// 3. Squeezes batch coefficients into `challenges[batch_challenges[i]]`
    /// 4. Combines: `Σ batch_coeff[i] * claim[i] * 2^(max_rounds - num_rounds[i])`
    VerifySumcheck {
        instances: Vec<SumcheckInstance>,
        stage: usize,
        /// Challenge indices for per-instance batching coefficients.
        /// Empty for unbatched stages (scaling uses `2^offset` only).
        batch_challenges: Vec<ChallengeIdx>,
        /// Transcript tag for absorbing input claims before squeezing
        /// batch coefficients. Required when `batch_challenges` is non-empty.
        claim_tag: Option<DomainSeparator>,
        /// Challenge slots to populate with sumcheck round challenges.
        /// After `SumcheckVerifier::verify` returns its Vec<F> of round
        /// challenges, slot `i` receives round-`i` challenge. Used by
        /// later ops that reference those challenges by `ChallengeIdx`.
        sumcheck_challenge_slots: Vec<ChallengeIdx>,
    },
    /// Read polynomial evaluations from current stage proof into the global table.
    RecordEvals { evals: Vec<Evaluation> },
    /// Absorb polynomial evaluations into transcript.
    AbsorbEvals {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
    },
    /// Verify output: composition formula must equal stored `final_eval`.
    ///
    /// The compiler places this at the exact position where all referenced
    /// evaluations are available, eliminating deferred checks.
    CheckOutput {
        instances: Vec<SumcheckInstance>,
        stage: usize,
        /// When non-empty, each instance's output is multiplied by its batch
        /// coefficient (stored at the given challenge index). Empty for
        /// unbatched sumchecks.
        batch_challenges: Vec<ChallengeIdx>,
    },
    /// Accumulate a PCS opening claim for a committed polynomial.
    CollectOpeningClaim {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// Accumulate a PCS opening claim at an explicit challenge point (not a
    /// stage's sumcheck point). Mirrors the prover's `Op::CollectOpeningClaimAt`.
    ///
    /// Used when a committed polynomial is opened at a point shorter than, or
    /// unrelated to, any stage's full sumcheck point — e.g. a standalone Twist
    /// that publishes ReadValue / WriteValue evaluations at the cycle-only
    /// challenge vector before running its main address × cycle sumcheck.
    CollectOpeningClaimAt {
        poly: PolynomialId,
        point_challenges: Vec<ChallengeIdx>,
    },
    /// Scale a recorded evaluation by `∏(1 − challenges[ci])`. Mirrors the
    /// prover's `Op::ScaleEval`. Used for dense cycle-only polys committed
    /// zero-padded to the full K×T grid: the prover's raw FinalBind scalar is
    /// the cycle-only MLE; the PCS opens at the padded grid's MLE which equals
    /// `∏(1 − r_addr_i) · MLE_cycle(r_cycle)`. Both sides pre-scale the claim
    /// by the `∏(1 − r_addr_i)` factor so the opening matches.
    ScaleEval {
        poly: PolynomialId,
        factor_challenges: Vec<ChallengeIdx>,
    },
    /// RLC-reduce all collected claims and verify PCS opening proofs.
    VerifyOpenings,
}

/// A single sumcheck instance within a batched stage.
///
/// The verifier evaluates `input_claim` before the sumcheck to compute
/// this instance's contribution to the combined claimed sum. After the
/// sumcheck, it evaluates `output_check` at the instance's challenge
/// slice to verify the composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SumcheckInstance {
    /// Symbolic formula for this instance's input claim.
    pub input_claim: ClaimFormula,
    /// Composition formula for the output claim check.
    /// Evaluated at the instance's challenge slice (offset by
    /// `max_rounds − num_rounds` into the stage's sumcheck challenges).
    pub output_check: ClaimFormula,
    /// Number of sumcheck rounds for this instance.
    pub num_rounds: usize,
    /// Composition degree (determines round polynomial size).
    pub degree: usize,
    /// How to convert raw sumcheck challenges to the canonical opening point
    /// used when evaluating `output_check`. Applied by the verifier before
    /// formula evaluation. `None` means raw challenges are used as-is.
    pub normalize: Option<PointNormalization>,
}

/// How raw sumcheck challenges are converted to the canonical opening point.
///
/// Sumcheck rounds produce challenges in binding order (LowToHigh = LSB first).
/// The opening point used by output check formulas is typically in big-endian
/// (MSB first) order. This enum specifies the transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PointNormalization {
    /// Reverse the full challenge sequence (LowToHigh → big-endian).
    Reverse,
    /// Multi-segment: split raw challenges into contiguous segments,
    /// reverse within each segment, then concatenate in the specified order.
    ///
    /// Example: RamRW with phase1=25 cycle vars, phase2=20 address vars:
    ///   `sizes = [25, 20], output_order = [1, 0]`
    ///   Result: `[reversed(raw[25..45]) ∥ reversed(raw[0..25])]`
    ///   = `[big-endian address ∥ big-endian cycle]`
    Segments {
        sizes: Vec<usize>,
        output_order: Vec<usize>,
    },
}

/// Univariate-skip first-round verification parameters.
///
/// The uniskip sends a full (uncompressed) polynomial. The verifier absorbs
/// it, squeezes a challenge, evaluates the polynomial at that challenge, and
/// records the result as an opening claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniskipVerify {
    /// Number of full polynomial coefficients (`degree + 1`).
    pub num_coeffs: usize,
    /// Poly index for the output evaluation (stored after verification).
    pub eval_poly: PolynomialId,
}

/// Symbolic sum-of-products for computing a verifier claim from
/// upstream evaluation values and challenges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimFormula {
    pub terms: Vec<ClaimTerm>,
}

impl ClaimFormula {
    pub fn zero() -> Self {
        Self { terms: vec![] }
    }
}

/// A single term in a claim formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimTerm {
    pub coeff: i128,
    pub factors: Vec<ClaimFactor>,
}

/// Factor in a claim formula term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClaimFactor {
    /// Value of evaluation `evals[poly_index]` (accumulated across stages).
    Eval(PolynomialId),
    /// Value of challenge `challenges[i]`.
    Challenge(ChallengeIdx),
    /// Single-variable eq between two challenge values:
    /// `eq(challenges[a], challenges[b]) = a*b + (1-a)(1-b)`.
    EqChallengePair { a: ChallengeIdx, b: ChallengeIdx },
    /// Multilinear eq polynomial evaluated at two points:
    /// `eq(r, s) = ∏ᵢ (rᵢ·sᵢ + (1−rᵢ)(1−sᵢ))`
    /// where `r` is formed from challenges at the given indices and
    /// `s` is the sumcheck challenge point from the given stage.
    EqEval {
        challenges: Vec<ChallengeIdx>,
        at_stage: VerifierStageIndex,
    },
    /// Lagrange kernel evaluation `L(τ, r)` over the uniform R1CS constraint
    /// domain. The runtime computes this from the domain size in the R1CS key.
    ///
    /// `L(τ, r) = Σ_k eq(τ, k) × L_k(r)` where `k` ranges over the
    /// constraint domain and `L_k` is the k-th Lagrange basis polynomial.
    LagrangeKernel {
        /// Challenge index for the τ value (e.g. τ_high).
        tau_challenge: ChallengeIdx,
        /// Challenge index for the evaluation point (e.g. uniskip r0).
        at_challenge: ChallengeIdx,
    },
    /// Uniform R1CS matrix–evaluation inner product at a Lagrange point.
    ///
    /// The runtime computes `Σ_k L_k(r0) × (Σ_j M[k][j] × z_j)` where:
    /// - `M` is the A or B matrix from the preprocessed `UniformSpartanKey`
    /// - `L_k(r0)` is the Lagrange basis for constraint `k` at `r0`
    /// - `z_j` are the prover-provided evaluation values at `eval_polys[j]`
    ///
    /// The Module never embeds matrix coefficients — the runtime resolves
    /// them from the R1CS key at verification time.
    UniformR1CSEval {
        matrix: R1CSMatrix,
        /// Poly identifiers whose evaluations form the z-vector.
        eval_polys: Vec<PolynomialId>,
        /// Challenge index for the Lagrange interpolation point (r0).
        at_challenge: ChallengeIdx,
        /// Number of constraints to evaluate (may be less than the full R1CS).
        num_constraints: usize,
        /// First integer in the Lagrange domain (symmetric convention: -(N-1)/2).
        domain_start: i64,
    },
    /// Eq evaluation between challenge values and a contiguous **slice** of a
    /// stage's (normalized) sumcheck point.
    ///
    /// Computes `eq(r, s[offset..offset+len])` where `len = challenges.len()`.
    /// This is needed when an output check uses only a portion of the opening
    /// point (e.g. the cycle portion of a combined address×cycle point).
    EqEvalSlice {
        challenges: Vec<ChallengeIdx>,
        at_stage: VerifierStageIndex,
        /// Starting index within the (normalized) sumcheck point.
        offset: usize,
    },
    /// Lagrange kernel `L(τ, r) = Σ_{k=0}^{N-1} L_k(τ) × L_k(r)` over an
    /// explicit domain `{0, 1, ..., domain_size-1}`.
    ///
    /// Generalizes [`LagrangeKernel`] to arbitrary domain sizes (e.g. size 3
    /// for product virtualization, size 10 for R1CS outer).
    LagrangeKernelDomain {
        tau_challenge: ChallengeIdx,
        at_challenge: ChallengeIdx,
        domain_size: usize,
        domain_start: i64,
    },
    /// Single Lagrange basis polynomial `L_k(r)` at a challenge value `r`,
    /// over the domain `{domain_start, ..., domain_start + domain_size - 1}`.
    LagrangeWeight {
        challenge: ChallengeIdx,
        domain_size: usize,
        domain_start: i64,
        basis_index: usize,
    },
    /// Evaluation of a public/preprocessed polynomial at the current stage's
    /// (normalized) sumcheck point. The runtime resolves the polynomial from
    /// the verifying key by its module poly index.
    PreprocessedPolyEval {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// Two-group R1CS inner product with Lagrange interpolation at `r0` and
    /// linear interpolation at `r_group` between two disjoint row-index sets.
    ///
    /// Computes:
    /// ```text
    /// (1 − r_group) · Σ_k L_k(r0) · (M[group0_indices[k]] · z)
    ///  + r_group   · Σ_k L_k(r0) · (M[group1_indices[k]] · z)
    /// ```
    ///
    /// Matches jolt-core's outer-Spartan group-split evaluation used after the
    /// univariate-skip round folds the constraint dimension. `group1_indices`
    /// may be shorter than `domain_size` (zero-padded to match `group0_indices`).
    GroupSplitR1CSEval {
        matrix: R1CSMatrix,
        /// Poly identifiers whose evaluations form the z-vector (z[0] = 1).
        eval_polys: Vec<PolynomialId>,
        /// Challenge index for the Lagrange interpolation point `r0`.
        at_r0: ChallengeIdx,
        /// Challenge index for the group-bit interpolation point `r_group`
        /// (usually the first sumcheck round challenge).
        at_r_group: ChallengeIdx,
        /// Constraint row indices for group 0.
        group0_indices: Vec<usize>,
        /// Constraint row indices for group 1 (may be shorter than group 0).
        group1_indices: Vec<usize>,
        /// Domain size for Lagrange basis at `r0`.
        domain_size: usize,
        /// First integer in the Lagrange domain.
        domain_start: i64,
    },
    /// Evaluation from the current stage's prover-provided evaluation list,
    /// at the given position. Used in output_check formulas when the same
    /// polynomial is opened at multiple points by different instances within
    /// a single batched stage. Position indexes into the stage's evaluation list.
    StageEval(usize),
    /// Evaluation of a polynomial at a specific prover stage.
    /// Resolved from `staged_evals[(poly, stage)]` — the value the polynomial
    /// had when it was evaluated during that stage. Prover-only: the verifier
    /// never sees this variant (it uses StageEval or Eval instead).
    StagedEval { poly: PolynomialId, stage: usize },
}

/// Which matrix of the R1CS relation `Az ∘ Bz = Cz`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum R1CSMatrix {
    A,
    B,
}

/// A polynomial evaluation at a specific point in the verifier schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    /// Poly identifier in the module's poly table.
    pub poly: PolynomialId,
    /// Verifier stage whose sumcheck challenge point is the evaluation point.
    pub at_stage: VerifierStageIndex,
}
