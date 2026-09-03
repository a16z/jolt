//! Optimized stage-1 Spartan outer kernels: the legacy prover's algorithms
//! behind the reference kernels' exact wire behavior.
//!
//! Techniques ported from `jolt-prover-legacy`'s `zkvm/spartan/outer.rs` +
//! `r1cs/evaluation.rs`:
//!
//! - **Typed small-scalar row evaluation**: the 19 eq-conditional constraint
//!   rows are evaluated per cycle as integers (`i64` guards, `S192`
//!   magnitudes) straight off a typed witness bundle — the 35 R1CS input
//!   tables are never materialized as field vectors
//!   (`R1CSEval::{eval_az,eval_bz}_*_group`).
//! - **Univariate skip over the centered integer domain**: the first-round
//!   polynomial needs only the 9 extended-node evaluations (in-domain nodes
//!   vanish); each is an integer Lagrange extension of the row values
//!   (`COEFFS_PER_J` / `extended_azbz_product_*`), so the whole pass costs 9
//!   integer dot products and one field fmadd per `(cycle, stream)` instead
//!   of per-row field multiplies.
//! - **Unreduced accumulation**: field × wide-integer products accumulate
//!   through `jolt-field`'s `SignedProductAccumulator` /
//!   `SmallScalarAccumulator` and reduce once per block
//!   (`FullAccumS`/`SmallAccumU`/`WideAccumS` + `barrett_reduce`).
//! - **Split-eq (Gruen/Dao-Thaler) factoring**: `eq(τ_low, ·)` is held as an
//!   `E_out ⊗ E_in` tensor and a per-round linear factor
//!   ([`GruenSplitEqPolynomial`]); round polynomials come from the two
//!   endpoints `q(0)`, `q(∞)` and the running claim (`gruen_poly_deg_3`),
//!   never from four full-domain evaluation sweeps.
//! - **Fused round-0 materialization**: the bound `Az`/`Bz` tables over the
//!   joint `(cycle ‖ stream)` domain and the first round's endpoints are
//!   produced by one pass over the typed rows
//!   (`OuterLinearStage::fused_materialise_polynomials_round_zero`).
//! - **In-place binding**: `Az`/`Bz` bind low-to-high in place with a reused
//!   scratch buffer; the 35 input tables are never bound at all.
//! - **Post-hoc opening evaluation**: the 35 produced opening claims come
//!   from one final eq-weighted walk over the typed rows
//!   (`R1CSEval::compute_claimed_inputs`), not from binding 35 polynomials
//!   through every round.
//!
//! Byte parity with the reference kernels holds because every step computes
//! the same field values by exact integer/field algebra (the integer Lagrange
//! extension coefficients equal the field Lagrange evaluations at integer
//! nodes; ring homomorphism does the rest), and the wire assembly reuses the
//! reference's own `jolt-poly` interpolation path.

use std::collections::BTreeMap;
#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
use std::time::{Duration, Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{outer_opening, SpartanOuterDimensions};
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltOpeningId, SpartanOuterPublic};
#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
use jolt_claims::NoChallenges;
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::signed::{S192, S256, S64};
#[cfg(all(feature = "metal", target_os = "macos"))]
use jolt_field::AkitaField;
use jolt_field::{
    Field, SignedProductAccumulator as _, SignedScalarAccumulator as _,
    WithSignedProductAccumulator, WithSmallScalarAccumulator,
};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
#[cfg(all(feature = "metal", target_os = "macos"))]
use jolt_riscv::InterleavedBitsMarker;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::witnesses::SpartanOuterRow;
#[cfg(all(feature = "metal", target_os = "macos"))]
use jolt_witness::witnesses::{
    Extract, FusedInc, InstructionRafFlag, LookupIndex, MappedPc, RamHammingWeight, RamInc,
    RamReadValue as Stage1RamReadValue, RamWriteValue as Stage1RamWriteValue, RemappedRamAddress,
    TableIndex, WitnessEnv,
};
#[cfg(test)]
use jolt_witness::witnesses::{
    Imm, InstructionFlag, LeftInstructionInput, LeftLookupOperand, LookupOutput,
    NextIsFirstInSequence, NextIsNoop, NextIsVirtual, NextPc, NextUnexpandedPc, OpFlag, Pc,
    Product, RamAddress, RamReadValue, RamWriteValue, RdWriteValue, RightInstructionInput,
    RightLookupOperand, Rs1Value, Rs2Value, ShouldBranch, ShouldJump, UnexpandedPc,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use jolt_witness::WitnessBundle;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_input::prepare_instruction_input_rows;
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::ram_trace::{
    RamAccessBundle, RamAccessCollection, RamAccessCollectionChunkWriter,
    RamAccessCollectionStorage, RamReadWriteRecordCollection,
    RamReadWriteRecordCollectionChunkWriter, RamReadWriteRecordCollectionStorage,
};
use super::support::collect_rows;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::solinas::bytecode_read_raf_address::{
    BytecodeAddressStage1TopologyChunkWriter, BytecodeAddressStage1TopologyOwner,
    BytecodeAddressStage1TopologyScratch, BytecodeAddressStage1TopologyStorage,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::solinas::spartan_shift::{
    SpartanShiftFlagWord, SpartanShiftResidentRows, SPARTAN_SHIFT_FLAG_ROWS_PER_WORD,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::solinas::{
    instruction_input_row_bytes, instruction_read_raf_claim_and_count_rank,
    spartan_outer_uniskip_successor_row_bytes, BooleanityRow, InstructionInputRow,
    InstructionInputRows, InstructionReadRafStage1ChunkWriter, InstructionReadRafStage1Owner,
    InstructionReadRafStage1Storage, MetalError, RegistersReadWriteStage1ChunkWriter,
    RegistersReadWriteStage1Source, RegistersReadWriteStage1Storage,
    RegistersValInstructionSourceRequest, SolinasMetal, SpartanOuterUniskipColdRow,
    SpartanOuterUniskipConfig, SpartanOuterUniskipRow, SpartanOuterUniskipRows,
    SpartanOuterUniskipSuccessorRow, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
    RAM_READ_WRITE_CYCLE_TILE_LOG2,
};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = OUTER_UNISKIP_DOMAIN_SIZE;
const SECOND_GROUP_LEN: usize = DOMAIN - 1;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const EXTENDED_NODE_COUNT: usize = DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);
/// One cycle's integer values of the 19 eq-conditional rows, split into the
/// two uni-skip stream groups (A-side guards as `i64`, B-side magnitudes as
/// `S192` — wide enough for the `RightLookupOperand`-bearing rows, whose
/// values reach ±2^130).
struct RowGroupValues {
    a_first: [i64; DOMAIN],
    a_second: [i64; SECOND_GROUP_LEN],
    b_first: [S192; DOMAIN],
    b_second: [S192; SECOND_GROUP_LEN],
}

/// Evaluate the 19 constraint rows at one cycle with exact integer
/// arithmetic. Formulas transcribe `jolt-r1cs`'s `rv64_eq_constraint_rows`
/// verbatim (matrix semantics, not satisfied-witness shortcuts), grouped as
/// `SPARTAN_OUTER_{FIRST,SECOND}_GROUP_ROWS` orders them.
fn row_group_values(row: &SpartanOuterRow) -> RowGroupValues {
    let flag = |value: bool| i64::from(value);
    let load = flag(row.load.0);
    let store = flag(row.store.0);
    let add = flag(row.add_operands.0);
    let sub = flag(row.subtract_operands.0);
    let mul = flag(row.multiply_operands.0);
    let jump = flag(row.jump.0);
    let should_branch = flag(row.should_branch.0);

    // Rows 1, 2, 3, 4, 5, 6, 11, 14, 17, 18.
    let a_first = [
        1 - load - store,
        load,
        load,
        store,
        add + sub + mul,
        1 - add - sub - mul,
        flag(row.assert_flag.0),
        flag(row.should_jump.0),
        flag(row.virtual_instruction.0) - flag(row.is_last_in_sequence.0),
        flag(row.next_is_virtual.0) - flag(row.next_is_first_in_sequence.0),
    ];
    // Rows 0, 7, 8, 9, 10, 12, 13, 15, 16.
    let a_second = [
        load + store,
        add,
        sub,
        mul,
        1 - add - sub - mul - flag(row.advice.0),
        flag(row.write_lookup_output_to_rd.0),
        jump,
        should_branch,
        1 - should_branch - jump,
    ];

    let diff = |a: u64, b: u64| S192::from_i128(i128::from(a) - i128::from(b));
    let b_first = [
        S192::from_u64(row.ram_address.0),
        diff(row.ram_read_value.0, row.ram_write_value.0),
        diff(row.ram_read_value.0, row.rd_write_value.0),
        diff(row.rs2_value.0, row.ram_write_value.0),
        S192::from_u64(row.left_lookup_operand.0),
        diff(row.left_lookup_operand.0, row.left_instruction_input.0),
        S192::from_i128(i128::from(row.lookup_output.0) - 1),
        diff(row.next_unexpanded_pc.0, row.lookup_output.0),
        S192::from_i128(i128::from(row.next_pc.0) - i128::from(row.pc.0) - 1),
        S192::from_i64(1 - flag(row.do_not_update_unexpanded_pc.0)),
    ];

    let flag_i128 = |value: bool| i128::from(value);
    let right_lookup = S192::from_u128(row.right_lookup_operand.0);
    let right_input = S192::from_i128(row.right_instruction_input.0);
    let left_input = S192::from_u64(row.left_instruction_input.0);
    let imm = S192::from_i128(row.imm.0);
    let product_limbs = row.product.0.magnitude_limbs();
    let product = S192::new(
        [product_limbs[0], product_limbs[1], 0],
        row.product.0.is_positive,
    );
    let two_pow_64 = S192::new([0, 1, 0], true);
    let b_second = [
        S192::from_i128(i128::from(row.ram_address.0) - i128::from(row.rs1_value.0)) - imm,
        right_lookup - left_input - right_input,
        right_lookup - left_input + right_input - two_pow_64,
        right_lookup - product,
        right_lookup - right_input,
        S192::from_i128(i128::from(row.rd_write_value.0) - i128::from(row.lookup_output.0)),
        S192::from_i128(
            i128::from(row.rd_write_value.0) - i128::from(row.unexpanded_pc.0) - 4
                + 2 * flag_i128(row.is_compressed.0),
        ),
        S192::from_i128(i128::from(row.next_unexpanded_pc.0) - i128::from(row.unexpanded_pc.0))
            - imm,
        S192::from_i128(
            i128::from(row.next_unexpanded_pc.0) - i128::from(row.unexpanded_pc.0) - 4
                + 4 * flag_i128(row.do_not_update_unexpanded_pc.0)
                + 2 * flag_i128(row.is_compressed.0),
        ),
    ];

    RowGroupValues {
        a_first,
        a_second,
        b_first,
        b_second,
    }
}

/// The exact integer Lagrange extension coefficients from the 10-node base
/// window to each out-of-domain extended node: `coeffs[i] = L_i(node)`.
/// Consecutive-integer domains make these integers (legacy's `COEFFS_PER_J`);
/// their field images equal `centered_lagrange_evals` at the node, which is
/// what ties the integer pipeline to the reference's field pipeline.
fn extension_coefficients() -> [(usize, [i64; DOMAIN]); EXTENDED_NODE_COUNT] {
    let mut out = [(0usize, [0i64; DOMAIN]); EXTENDED_NODE_COUNT];
    let mut slot = 0;
    for position in 0..EXTENDED_SIZE {
        let node = EXTENDED_START + position as i64;
        if node >= DOMAIN_START && node < DOMAIN_START + DOMAIN as i64 {
            continue;
        }
        let mut coefficients = [0i64; DOMAIN];
        for (i, coefficient) in coefficients.iter_mut().enumerate() {
            let mut numerator: i128 = 1;
            let mut denominator: i128 = 1;
            for j in 0..DOMAIN {
                if j == i {
                    continue;
                }
                numerator *= i128::from(node - (DOMAIN_START + j as i64));
                denominator *= i128::from(i as i64 - j as i64);
            }
            debug_assert_eq!(numerator % denominator, 0);
            *coefficient = (numerator / denominator) as i64;
        }
        out[slot] = (position, coefficients);
        slot += 1;
    }
    debug_assert_eq!(slot, EXTENDED_NODE_COUNT);
    out
}

/// `Az·Bz` at every extended node for one cycle, per stream: integer Lagrange
/// extension of the group row values, then one wide integer product. Ranges:
/// `|az| < 2^22`, `|bz| < 2^152`, product `< 2^174` — inside `S256`.
fn extended_products(
    values: &RowGroupValues,
    coefficients: &[(usize, [i64; DOMAIN]); EXTENDED_NODE_COUNT],
) -> [(S256, S256); EXTENDED_NODE_COUNT] {
    let mut out = [(S256::zero(), S256::zero()); EXTENDED_NODE_COUNT];
    for (slot, (_, coefficients)) in coefficients.iter().enumerate() {
        let mut az_first: i64 = 0;
        let mut az_second: i64 = 0;
        let mut bz_first = S192::zero();
        let mut bz_second = S192::zero();
        for (i, &c) in coefficients.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let a_first = values.a_first[i];
            if a_first != 0 {
                az_first += c * a_first;
            }
            let b_first = &values.b_first[i];
            if b_first.magnitude_limbs() != [0; 3] {
                S64::from_i64(c).fmadd_trunc::<3, 3>(b_first, &mut bz_first);
            }
            if i < SECOND_GROUP_LEN {
                let a_second = values.a_second[i];
                if a_second != 0 {
                    az_second += c * a_second;
                }
                let b_second = &values.b_second[i];
                if b_second.magnitude_limbs() != [0; 3] {
                    S64::from_i64(c).fmadd_trunc::<3, 3>(b_second, &mut bz_second);
                }
            }
        }
        out[slot] = (
            S64::from_i64(az_first).mul_trunc::<3, 4>(&bz_first),
            S64::from_i64(az_second).mul_trunc::<3, 4>(&bz_second),
        );
    }
    out
}

fn widen(value: &S192) -> S256 {
    let limbs = value.magnitude_limbs();
    S256::new([limbs[0], limbs[1], limbs[2], 0], value.is_positive)
}

/// Fold group row values with the uni-skip challenge's Lagrange weights into
/// the bound `Az`/`Bz` values for one `(cycle, stream)` cell, through the
/// unreduced accumulators.
fn fold_group<F: Field>(weights: &[F], guards: &[i64], magnitudes: &[S192]) -> (F, F) {
    let mut az = <F as WithSmallScalarAccumulator>::SmallScalarAccumulator::default();
    let mut bz = <F as WithSignedProductAccumulator>::SignedProductAccumulator::default();
    for ((&weight, &guard), magnitude) in weights.iter().zip(guards).zip(magnitudes) {
        az.fmadd_i64(weight, guard);
        let limbs = magnitude.magnitude_limbs();
        if limbs[1] == 0 && limbs[2] == 0 {
            bz.fmadd_signed_u64(weight, limbs[0], magnitude.is_positive);
        } else {
            bz.fmadd_s256(weight, &widen(magnitude));
        }
    }
    (az.reduce(), bz.reduce())
}

/// The uni-skip carry: everything the uni-skip front computes that the
/// remainder slot reclaims — the typed-row store (reused for
/// materialization and the final opening walk), the stage challenge vector,
/// and the extended-node evaluations of `t1`.
struct SpartanOuterCarry<F: Field> {
    log_t: usize,
    tau: Vec<F>,
    rows: RowsStore,
    /// All `2·DOMAIN − 1` node values of `t1`; in-domain nodes stay zero (a
    /// satisfying witness vanishes there), matching the reference layout.
    t1_values: Vec<F>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for SpartanOuterCarry<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("tau"), vec_heap_bytes(&self.tau));
        visitor.visit_simple(allocative::Key::new("rows"), self.rows.heap_bytes());
        visitor.visit_simple(
            allocative::Key::new("t1_values"),
            vec_heap_bytes(&self.t1_values),
        );
        visitor.exit();
    }
}

/// Where the kernel's typed rows live. A slice-backed witness serves an
/// owning handle and every pass re-extracts its windows on the fly — the
/// materialized row vector (~176 B × T, the prover's peak allocation at
/// large scale) never exists. Re-emulating sources retain the collected
/// rows as before.
enum RowsStore {
    Owned(jolt_witness::OwnedRows),
    Retained(Vec<SpartanOuterRow>),
}

impl RowsStore {
    /// Resolve for a witness plane: the owning handle when the source is
    /// slice-backed (and covers the cycle domain), a materialized collect
    /// otherwise.
    fn resolve<F: Field>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        match witness.owned_rows() {
            Some(owned) if cycles <= owned.cycles() => Ok(Self::Owned(owned)),
            _ => Ok(Self::Retained(collect_rows(witness, cycles)?)),
        }
    }

    fn access(&self) -> RowsAccess<'_> {
        match self {
            Self::Owned(owned) => RowsAccess::View(owned.view()),
            Self::Retained(rows) => RowsAccess::Retained(rows),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn production_source_kind(&self) -> &'static str {
        match self {
            Self::Owned(_) => "owned_random_access",
            Self::Retained(_) => "retained_host_repack",
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn host_repack_rows(&self) -> usize {
        match self {
            Self::Owned(_) => 0,
            Self::Retained(rows) => rows.len(),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn explicit_rows(&self) -> usize {
        match self {
            Self::Owned(rows) => rows.physical_rows().min(rows.cycles()),
            Self::Retained(rows) => rows.len(),
        }
    }
}

#[cfg(feature = "allocative")]
impl RowsStore {
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Owned(_) => 0,
            Self::Retained(rows) => crate::backend::vec_heap_bytes(rows),
        }
    }
}

/// One pass's borrowed row provider.
enum RowsAccess<'a> {
    View(jolt_witness::RandomAccessRows<'a>),
    Retained(&'a [SpartanOuterRow]),
}

impl RowsAccess<'_> {
    /// The typed row at cycle `t` — an extraction window over a slice-backed
    /// source, an indexed copy from a retained vector. Pure per index.
    #[inline]
    fn row(&self, t: usize) -> Result<SpartanOuterRow, WitnessError> {
        match self {
            Self::View(view) => view.window(t),
            Self::Retained(rows) => Ok(rows[t]),
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct Stage1InstructionFacts {
    lookup_index: LookupIndex,
    table_index: TableIndex,
    raf_flag: InstructionRafFlag,
    mapped_pc: MappedPc,
    remapped_ram_address: RemappedRamAddress,
    fused_inc: FusedInc,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Copy, Debug)]
struct Stage1ProjectionRow {
    outer: SpartanOuterRow,
    instruction: Stage1InstructionFacts,
    ram_access: RamAccessBundle,
    register_indices: [Option<u8>; 2],
    register_write: Option<(u8, u64, u64)>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl WitnessBundle for Stage1ProjectionRow {
    fn from_row(
        row: &jolt_riscv::JoltTraceRow,
        next: Option<&jolt_riscv::JoltTraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let outer = SpartanOuterRow::from_row(row, next, env)?;
        let raf = !row.circuit_flags().is_interleaved_operands();
        let lookup_index = if raf {
            outer.right_lookup_operand.0
        } else {
            jolt_lookup_tables::interleave_bits(
                outer.left_lookup_operand.0,
                outer.right_lookup_operand.0 as u64,
            )
        };
        let remapped_ram_address = RemappedRamAddress::extract(row, next, env)?;
        let ram_inc = RamInc::extract(row, next, env)?;
        let fused_inc = if row.is_store() {
            ram_inc.0
        } else if row.rd_index().is_some() {
            i128::from(row.rd_write_value()) - i128::from(row.rd_pre_value())
        } else {
            0
        };
        let register_write = row
            .rd_index()
            .map(|register| (register, row.rd_pre_value(), row.rd_write_value()));
        Ok(Self {
            outer,
            instruction: Stage1InstructionFacts {
                lookup_index: LookupIndex(lookup_index),
                table_index: TableIndex::extract(row, next, env)?,
                raf_flag: InstructionRafFlag(raf),
                mapped_pc: MappedPc(Some(row.pc() as usize)),
                remapped_ram_address,
                fused_inc: FusedInc(fused_inc),
            },
            ram_access: RamAccessBundle {
                address: remapped_ram_address,
                pre_value: Stage1RamReadValue(row.ram_read_value()),
                post_value: Stage1RamWriteValue(row.ram_write_value()),
                ram_inc,
                ram_hamming_weight: RamHammingWeight::extract(row, next, env)?,
            },
            register_indices: [row.rs1_index(), row.rs2_index()],
            register_write,
        })
    }

    fn annotated_ids() -> Vec<jolt_claims::protocols::jolt::JoltPolynomialId> {
        SpartanOuterRow::annotated_ids()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn pack_stage1_instruction_source(
    facts: Stage1InstructionFacts,
) -> Result<(BooleanityRow, u8, bool), MetalError> {
    let mapped_pc = facts.mapped_pc.0.map(|pc| pc as u64);
    let row = BooleanityRow::new(
        facts.lookup_index.0,
        mapped_pc,
        facts.remapped_ram_address.0,
        facts.fused_inc.0,
    )?;
    let table_plus_one = facts
        .table_index
        .0
        .map_or(Some(0), |table| table.checked_add(1))
        .and_then(|table| u8::try_from(table).ok())
        .ok_or(MetalError::InvalidInstructionReadRafGrouped(
            "lookup table index cannot be encoded by the Stage-1 owner".to_owned(),
        ))?;
    let _ = instruction_read_raf_claim_and_count_rank(table_plus_one, facts.raf_flag.0).ok_or(
        MetalError::InvalidInstructionReadRafGrouped(
            "lookup table index exceeds the InstructionReadRAF table domain".to_owned(),
        ),
    )?;
    Ok((row, table_plus_one, facts.raf_flag.0))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Copy)]
struct PackedStage1PaddingRow {
    instruction_input: InstructionInputRow,
    successor: SpartanOuterUniskipSuccessorRow,
    cold: SpartanOuterUniskipColdRow,
    instruction_source: BooleanityRow,
    table_plus_one: u8,
    raf: bool,
    unexpanded_pc: u64,
    pc: u64,
    shift_flags: SpartanShiftFlagWord,
    ram_access: RamAccessBundle,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Copy)]
struct Stage1PaddingRows {
    regular: Option<PackedStage1PaddingRow>,
    terminal: Option<PackedStage1PaddingRow>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Stage1PaddingRows {
    fn new(
        access: &jolt_witness::RandomAccessRows<'_>,
        explicit_rows: usize,
        cycles: usize,
    ) -> Result<Self, MetalError> {
        let regular = (explicit_rows + 1 < cycles)
            .then(|| pack_stage1_padding_row(access, explicit_rows))
            .transpose()?;
        let terminal = (explicit_rows < cycles)
            .then(|| pack_stage1_padding_row(access, cycles - 1))
            .transpose()?;
        Ok(Self { regular, terminal })
    }

    const fn source_window_count(self, explicit_rows: usize) -> usize {
        explicit_rows + self.regular.is_some() as usize + self.terminal.is_some() as usize
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Copy)]
struct Stage1ChunkParts {
    physical: usize,
    regular_padding: usize,
    terminal_padding: usize,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn stage1_chunk_parts(
    chunk_start: usize,
    chunk_len: usize,
    explicit_rows: usize,
    cycles: usize,
) -> Stage1ChunkParts {
    let physical = explicit_rows.saturating_sub(chunk_start).min(chunk_len);
    let terminal_padding = usize::from(
        explicit_rows < cycles && chunk_start + chunk_len == cycles && physical < chunk_len,
    );
    Stage1ChunkParts {
        physical,
        regular_padding: chunk_len - physical - terminal_padding,
        terminal_padding,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn pack_stage1_padding_row(
    access: &jolt_witness::RandomAccessRows<'_>,
    row_index: usize,
) -> Result<PackedStage1PaddingRow, MetalError> {
    let projected: Stage1ProjectionRow =
        access
            .window(row_index)
            .map_err(|error| MetalError::SpartanOuterRowExtraction {
                row: row_index,
                message: error.to_string(),
            })?;
    let packed = SpartanOuterUniskipRow::from_spartan_outer(&projected.outer);
    let (instruction_input, residual) = packed.split();
    let (successor, cold) = residual.partition();
    let (instruction_source, table_plus_one, raf) =
        pack_stage1_instruction_source(projected.instruction)?;
    let full_mask = |value: bool| if value { u32::MAX } else { 0 };
    Ok(PackedStage1PaddingRow {
        instruction_input,
        successor,
        cold,
        instruction_source,
        table_plus_one,
        raf,
        unexpanded_pc: projected.outer.unexpanded_pc.0,
        pc: projected.outer.pc.0,
        shift_flags: SpartanShiftFlagWord {
            is_virtual: full_mask(projected.outer.virtual_instruction.0),
            is_first_in_sequence: full_mask(projected.outer.is_first_in_sequence.0),
            is_noop: full_mask(projected.outer.is_noop.0),
        },
        ram_access: projected.ram_access,
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn fill_stage1_outer_padding(
    instruction_input: &mut [InstructionInputRow],
    successor: &mut [SpartanOuterUniskipSuccessorRow],
    cold: &mut [SpartanOuterUniskipColdRow],
    start: usize,
    count: usize,
    padding: &PackedStage1PaddingRow,
) {
    let end = start + count;
    instruction_input[start..end].fill(padding.instruction_input);
    successor[start..end].fill(padding.successor);
    cold[start..end].fill(padding.cold);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn fill_stage1_shift_padding(
    unexpanded_pc: &mut [u64],
    pc: &mut [u64],
    flags: &mut [SpartanShiftFlagWord],
    start: usize,
    count: usize,
    padding: &PackedStage1PaddingRow,
) {
    let end = start + count;
    unexpanded_pc[start..end].fill(padding.unexpanded_pc);
    pc[start..end].fill(padding.pc);
    for (word, flag_word) in flags
        .iter_mut()
        .enumerate()
        .take(end.div_ceil(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
        .skip(start / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD)
    {
        let low = start.saturating_sub(word * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        let high = end
            .saturating_sub(word * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD)
            .min(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        let low_mask = u32::MAX.checked_shl(low as u32).unwrap_or(0);
        let high_mask = u32::MAX
            .checked_shr((SPARTAN_SHIFT_FLAG_ROWS_PER_WORD - high) as u32)
            .unwrap_or(0);
        let mask = low_mask & high_mask;
        let merge = |current: u32, value: u32| (current & !mask) | (value & mask);
        flag_word.is_virtual = merge(flag_word.is_virtual, padding.shift_flags.is_virtual);
        flag_word.is_first_in_sequence = merge(
            flag_word.is_first_in_sequence,
            padding.shift_flags.is_first_in_sequence,
        );
        flag_word.is_noop = merge(flag_word.is_noop, padding.shift_flags.is_noop);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
struct Stage1OwnerChunkWriters<'borrow, 'instruction, 'bytecode, 'ram, 'ram_records, 'registers> {
    instruction: &'borrow mut InstructionReadRafStage1ChunkWriter<'instruction>,
    bytecode: Option<&'borrow mut BytecodeAddressStage1TopologyChunkWriter<'bytecode>>,
    ram_access: Option<&'borrow mut RamAccessCollectionChunkWriter<'ram>>,
    ram_records: Option<&'borrow mut RamReadWriteRecordCollectionChunkWriter<'ram_records>>,
    registers: Option<&'borrow mut RegistersReadWriteStage1ChunkWriter<'registers>>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Stage1OwnerChunkWriters<'_, '_, '_, '_, '_, '_> {
    fn len(&self) -> usize {
        self.instruction.len()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one decoded row feeds several independently optional Stage-1 owners"
    )]
    fn push(
        &mut self,
        row_index: usize,
        explicit_rows: usize,
        instruction: Stage1InstructionFacts,
        ram_access: RamAccessBundle,
        register_indices: [Option<u8>; 2],
        register_write: Option<(u8, u64, u64)>,
        bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<(), MetalError> {
        self.instruction.record_ram_remap_compatibility(
            ram_access.ram_hamming_weight.0 == instruction.remapped_ram_address.0.is_some(),
        );
        let (row, table_plus_one, raf) = pack_stage1_instruction_source(instruction)?;
        if let Some(topology) = self.bytecode.as_mut() {
            let rank = if row_index < explicit_rows {
                topology.record(bytecode_scratch, instruction.mapped_pc.0.unwrap_or(0))?
            } else {
                0
            };
            self.instruction.push_with_register_write(
                row,
                table_plus_one,
                raf,
                rank,
                register_write,
            )?;
        } else {
            self.instruction.push_with_register_write(
                row,
                table_plus_one,
                raf,
                0,
                register_write,
            )?;
        }
        if let Some(writer) = self.ram_access.as_mut() {
            writer.push(ram_access).map_err(|error| {
                MetalError::InvalidRamAccessCollection(error.reason().to_owned())
            })?;
        }
        if let Some(writer) = self.ram_records.as_mut() {
            writer.push(ram_access).map_err(|error| {
                MetalError::InvalidRamAccessCollection(error.reason().to_owned())
            })?;
        }
        if let Some(writer) = self.registers.as_mut() {
            writer.push(register_indices, register_write)?;
        }
        Ok(())
    }

    fn fill_padding(
        &mut self,
        padding: &PackedStage1PaddingRow,
        count: usize,
    ) -> Result<(), MetalError> {
        self.instruction.record_ram_remap_compatibility(
            padding.ram_access.ram_hamming_weight.0 == padding.ram_access.address.0.is_some(),
        );
        self.instruction.fill_repeated_with_register_write(
            padding.instruction_source,
            padding.table_plus_one,
            padding.raf,
            0,
            None,
            count,
        )?;
        if let Some(writer) = self.ram_access.as_mut() {
            writer
                .fill_repeated(padding.ram_access, count)
                .map_err(|error| {
                    MetalError::InvalidRamAccessCollection(error.reason().to_owned())
                })?;
        }
        if let Some(writer) = self.ram_records.as_mut() {
            writer
                .fill_repeated(padding.ram_access, count)
                .map_err(|error| {
                    MetalError::InvalidRamAccessCollection(error.reason().to_owned())
                })?;
        }
        if let Some(writer) = self.registers.as_mut() {
            writer.fill_empty(count)?;
        }
        Ok(())
    }

    fn finish(
        &mut self,
        bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<(), MetalError> {
        if let Some(topology) = self.bytecode.as_mut() {
            topology.finish(bytecode_scratch)?;
        }
        if let Some(writer) = self.ram_access.as_mut() {
            writer.finish().map_err(|error| {
                MetalError::InvalidRamAccessCollection(error.reason().to_owned())
            })?;
        }
        if let Some(writer) = self.ram_records.as_mut() {
            writer.finish().map_err(|error| {
                MetalError::InvalidRamAccessCollection(error.reason().to_owned())
            })?;
        }
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_stage1_owner_chunks<'instruction, 'bytecode, 'ram, 'ram_records, 'registers, R>(
    instruction: &mut [InstructionReadRafStage1ChunkWriter<'instruction>],
    bytecode: Option<&mut [BytecodeAddressStage1TopologyChunkWriter<'bytecode>]>,
    ram_access: Option<&mut [RamAccessCollectionChunkWriter<'ram>]>,
    ram_records: Option<&mut [RamReadWriteRecordCollectionChunkWriter<'ram_records>]>,
    registers: Option<&mut [RegistersReadWriteStage1ChunkWriter<'registers>]>,
    fill: impl FnOnce(
        &mut [Stage1OwnerChunkWriters<
            '_,
            'instruction,
            'bytecode,
            'ram,
            'ram_records,
            'registers,
        >],
    ) -> Result<R, MetalError>,
) -> Result<R, MetalError> {
    if bytecode
        .as_ref()
        .is_some_and(|writers| writers.len() != instruction.len())
    {
        return Err(MetalError::InvalidInstructionReadRafGrouped(
            "Stage-1 owner chunk counts disagree".to_owned(),
        ));
    }
    if ram_access
        .as_ref()
        .is_some_and(|writers| writers.len() != instruction.len())
    {
        return Err(MetalError::InvalidRamAccessCollection(
            "Stage-1 RAM chunk counts disagree".to_owned(),
        ));
    }
    if ram_records
        .as_ref()
        .is_some_and(|writers| writers.len() != instruction.len())
    {
        return Err(MetalError::InvalidRamAccessCollection(
            "Stage-1 RAM record chunk counts disagree".to_owned(),
        ));
    }
    if registers
        .as_ref()
        .is_some_and(|writers| writers.len() != instruction.len())
    {
        return Err(MetalError::InvalidRegistersReadWriteState(
            "Stage-1 register chunk counts disagree",
        ));
    }
    let mut chunks: Vec<
        Stage1OwnerChunkWriters<'_, 'instruction, 'bytecode, 'ram, 'ram_records, 'registers>,
    > = match bytecode {
        Some(bytecode) => instruction
            .iter_mut()
            .zip(bytecode)
            .map(|(instruction, bytecode)| Stage1OwnerChunkWriters {
                instruction,
                bytecode: Some(bytecode),
                ram_access: None,
                ram_records: None,
                registers: None,
            })
            .collect(),
        None => instruction
            .iter_mut()
            .map(|instruction| Stage1OwnerChunkWriters {
                instruction,
                bytecode: None,
                ram_access: None,
                ram_records: None,
                registers: None,
            })
            .collect(),
    };
    if let Some(ram_access) = ram_access {
        for (chunk, ram_access) in chunks.iter_mut().zip(ram_access) {
            chunk.ram_access = Some(ram_access);
        }
    }
    if let Some(ram_records) = ram_records {
        for (chunk, ram_records) in chunks.iter_mut().zip(ram_records) {
            chunk.ram_records = Some(ram_records);
        }
    }
    if let Some(registers) = registers {
        for (chunk, registers) in chunks.iter_mut().zip(registers) {
            chunk.registers = Some(registers);
        }
    }
    fill(&mut chunks)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn with_stage1_owner_chunks<R>(
    instruction: &mut InstructionReadRafStage1Storage,
    bytecode: Option<&mut BytecodeAddressStage1TopologyStorage>,
    ram_access: Option<&mut RamAccessCollectionStorage>,
    ram_records: Option<&mut RamReadWriteRecordCollectionStorage>,
    registers: Option<&mut RegistersReadWriteStage1Storage>,
    fill: impl FnOnce(&mut [Stage1OwnerChunkWriters<'_, '_, '_, '_, '_, '_>]) -> Result<R, MetalError>,
) -> Result<R, MetalError> {
    instruction.with_chunk_writers(|instruction| match ram_records {
        Some(ram_records) => ram_records.with_chunk_writers(|ram_records| match registers {
            Some(registers) => registers.with_chunk_writers(|registers| match ram_access {
                Some(ram_access) => ram_access.with_chunk_writers(|ram_access| match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            Some(ram_access),
                            Some(ram_records),
                            Some(registers),
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        Some(ram_access),
                        Some(ram_records),
                        Some(registers),
                        fill,
                    ),
                }),
                None => match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            None,
                            Some(ram_records),
                            Some(registers),
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        None,
                        Some(ram_records),
                        Some(registers),
                        fill,
                    ),
                },
            }),
            None => match ram_access {
                Some(ram_access) => ram_access.with_chunk_writers(|ram_access| match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            Some(ram_access),
                            Some(ram_records),
                            None,
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        Some(ram_access),
                        Some(ram_records),
                        None,
                        fill,
                    ),
                }),
                None => match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            None,
                            Some(ram_records),
                            None,
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        None,
                        Some(ram_records),
                        None,
                        fill,
                    ),
                },
            },
        }),
        None => match registers {
            Some(registers) => registers.with_chunk_writers(|registers| match ram_access {
                Some(ram_access) => ram_access.with_chunk_writers(|ram_access| match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            Some(ram_access),
                            None,
                            Some(registers),
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        Some(ram_access),
                        None,
                        Some(registers),
                        fill,
                    ),
                }),
                None => match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            None,
                            None,
                            Some(registers),
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        None,
                        None,
                        Some(registers),
                        fill,
                    ),
                },
            }),
            None => match ram_access {
                Some(ram_access) => ram_access.with_chunk_writers(|ram_access| match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(
                            instruction,
                            Some(bytecode),
                            Some(ram_access),
                            None,
                            None,
                            fill,
                        )
                    }),
                    None => run_stage1_owner_chunks(
                        instruction,
                        None,
                        Some(ram_access),
                        None,
                        None,
                        fill,
                    ),
                }),
                None => match bytecode {
                    Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                        run_stage1_owner_chunks(instruction, Some(bytecode), None, None, None, fill)
                    }),
                    None => run_stage1_owner_chunks(instruction, None, None, None, None, fill),
                },
            },
        },
    })
}

/// Extended-node evaluations of
/// `t1(Y) = Σ_{t,s} eq(τ_low, (t,s)) · Az(Y,s,t) · Bz(Y,s,t)`, with the eq
/// table factored as `E_out ⊗ E_in` and the per-cycle products from the
/// integer extension pipeline.
fn extended_t1_values<F: Field>(
    rows: &RowsAccess<'_>,
    tau_low: &[F],
) -> Result<Vec<F>, WitnessError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<F>::evals(out_point, None);
    let e_in = EqPolynomial::<F>::evals(in_point, None);
    // `in_point` always covers the stream bit (τ_low's last entry), so every
    // (cycle, stream) pair sits inside one `x_out` block.
    let pairs_per_block = e_in.len() / 2;
    let coefficients = extension_coefficients();

    let block = |x_out: usize| -> Result<Vec<F>, WitnessError> {
        let mut accumulators: Vec<<F as WithSignedProductAccumulator>::SignedProductAccumulator> =
            vec![Default::default(); EXTENDED_NODE_COUNT];
        for pair in 0..pairs_per_block {
            let t = x_out * pairs_per_block + pair;
            let row = rows.row(t)?;
            let values = row_group_values(&row);
            let products = extended_products(&values, &coefficients);
            for (accumulator, (first, second)) in accumulators.iter_mut().zip(&products) {
                accumulator.fmadd_s256(e_in[2 * pair], first);
                accumulator.fmadd_s256(e_in[2 * pair + 1], second);
            }
        }
        Ok(accumulators
            .into_iter()
            .map(|accumulator| e_out[x_out] * accumulator.reduce())
            .collect())
    };
    let merge = |mut left: Vec<F>, right: Vec<F>| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };

    #[cfg(feature = "parallel")]
    let extended = (0..e_out.len()).into_par_iter().map(block).try_reduce(
        || vec![F::zero(); EXTENDED_NODE_COUNT],
        |left, right| Ok(merge(left, right)),
    )?;
    #[cfg(not(feature = "parallel"))]
    let extended = {
        let mut folded = vec![F::zero(); EXTENDED_NODE_COUNT];
        for x_out in 0..e_out.len() {
            folded = merge(folded, block(x_out)?);
        }
        folded
    };

    let mut t1_values = vec![F::zero(); EXTENDED_SIZE];
    for ((position, _), value) in extension_coefficients().iter().zip(extended) {
        t1_values[*position] = value;
    }
    Ok(t1_values)
}

/// The stage-1 uni-skip front: typed-row collection, the extended-node
/// evaluation pass, and the first-round polynomial assembly.
pub struct OptimizedOuterUniskip;

impl OptimizedOuterUniskip {
    /// The post-collection half of [`UniskipKernel::prepare`], for the
    /// in-module parity tests (which construct rows directly).
    #[cfg(test)]
    fn prepare_from_rows<F: Field>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: Vec<SpartanOuterRow>,
    ) -> Result<(), KernelError<F>> {
        if rows.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer row count disagrees with log_t",
            });
        }
        Self::prepare_from_store(session, log_t, tau, RowsStore::Retained(rows))
    }

    /// The store-generic half of `prepare`.
    fn prepare_from_store<F: Field>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: RowsStore,
    ) -> Result<(), KernelError<F>> {
        if tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer tau must carry log_t + 2 challenges",
            });
        }
        let (tau_low, _) = tau.split_at(log_t + 1);
        let t1_values = extended_t1_values(&rows.access(), tau_low)?;
        session.park(SpartanOuterCarry {
            log_t,
            tau: tau.to_vec(),
            rows,
            t1_values,
        });
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_spartan_outer_uniskip(
    context: &SolinasMetal,
    config: SpartanOuterUniskipConfig,
    session: &mut ProofSession,
    log_t: usize,
    tau: &[AkitaField],
    witness: &dyn JoltWitnessPlane<AkitaField>,
) -> Result<(), KernelError<AkitaField>> {
    if tau.len() != log_t + 2 {
        return Err(KernelError::InvariantViolation {
            reason: "Spartan outer tau must carry log_t + 2 challenges",
        });
    }
    let cycles = 1usize << log_t;
    let rows = RowsStore::resolve(witness, cycles)?;
    let (tau_low, _) = tau.split_at(log_t + 1);
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<AkitaField>::evals(out_point, None);
    let e_in = EqPolynomial::<AkitaField>::evals(in_point, None);
    let (extended, resident) = {
        let explicit_rows = rows.explicit_rows();
        let resident = {
            let _span = tracing::info_span!("MetalSpartanOuterUniskip::row_handoff").entered();
            match session.take::<SpartanOuterUniskipRows>() {
                Some(resident)
                    if resident.len() == cycles && resident.explicit_rows() == explicit_rows =>
                {
                    resident
                }
                _ => prepare_metal_spartan_outer_rows(context, &rows, cycles)?,
            }
        };
        let compact_rows_storage_id = resident.instruction_input_allocation_identity();
        let residual_rows_storage_id = resident.allocation_identity();
        let _handoff = tracing::info_span!(
            "MetalInstructionInput::compact_rows_stage1_handoff",
            compact_rows_storage_id,
            residual_rows_storage_id,
            resident_rows = cycles,
            explicit_rows,
            compact_row_bytes = 48,
            residual_row_bytes = 112,
            residual_allocations = 1,
            full_domain_copy_bytes = 0,
            full_domain_copy_dispatches = 0,
            host_repack_rows = 0,
        )
        .entered();
        let invocation = context
            .prepare_spartan_outer_uniskip_with_rows(&resident, &e_in, &e_out, config)
            .map_err(metal_outer_error)?;
        {
            let dispatch_span = tracing::info_span!(
                "MetalSpartanOuterUniskip::dispatch",
                gpu_active_ns = tracing::field::Empty,
            );
            let _dispatch = dispatch_span.enter();
            let gpu_active = invocation.execute_timed().map_err(metal_outer_error)?;
            let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
            let _ = dispatch_span.record("gpu_active_ns", gpu_active_ns);
        }
        let output = invocation.read_output().map_err(metal_outer_error)?;
        drop(invocation);
        (output, resident)
    };
    session.park(resident);
    let mut t1_values = vec![AkitaField::zero(); EXTENDED_SIZE];
    for ((position, _), value) in extension_coefficients().iter().zip(extended) {
        t1_values[*position] = value;
    }
    session.park(SpartanOuterCarry {
        log_t,
        tau: tau.to_vec(),
        rows,
        t1_values,
    });
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn take_metal_spartan_outer_tau(
    session: &mut ProofSession,
    expected_log_t: usize,
) -> Result<Vec<AkitaField>, KernelError<AkitaField>> {
    let carry =
        session
            .take::<SpartanOuterCarry<AkitaField>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "Metal outer remainder found no uni-skip carry",
            })?;
    if carry.log_t != expected_log_t {
        return Err(KernelError::InvariantViolation {
            reason: "Metal outer remainder carry disagrees with relation geometry",
        });
    }
    Ok(carry.tau)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_spartan_outer_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<SpartanOuterUniskipRows, KernelError<AkitaField>> {
    let rows = RowsStore::resolve(witness, cycles)?;
    prepare_metal_spartan_outer_rows(context, &rows, cycles)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug)]
pub(crate) enum MetalSpartanDenseRowsError {
    Kernel(KernelError<AkitaField>),
    Metal(MetalError),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalSpartanDenseRowsError {
    pub(crate) fn is_capacity_error(&self) -> bool {
        matches!(self, Self::Metal(error) if error.is_capacity_error())
    }

    pub(crate) fn into_kernel_error(self) -> KernelError<AkitaField> {
        match self {
            Self::Kernel(error) => error,
            Self::Metal(error) => metal_outer_error(error),
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn stage1_owner_rows_span(
    cycles: usize,
    explicit_rows: usize,
    witness_row_extractions: usize,
) -> tracing::Span {
    tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind = "owned_random_access",
        witness_row_extractions,
        padding_rows_bulk_filled = cycles - explicit_rows,
        residual_rows_written = cycles,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 112,
        compact_allocations = 1,
        residual_allocations = 1,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows = 0,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = tracing::field::Empty,
        resident_rows = cycles,
        explicit_rows,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bytecode_stage1_topology_span(enabled: bool, physical_rows: usize) -> tracing::Span {
    tracing::info_span!(
        "MetalBytecodeReadRafAddress::fused_topology_prepare",
        enabled,
        physical_rows,
        chunk_rows = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bytecode_stage1_topology_publish_span(enabled: bool, physical_rows: usize) -> tracing::Span {
    tracing::info_span!(
        "MetalBytecodeReadRafAddress::fused_topology_publish",
        enabled,
        physical_rows,
        chunk_rows = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
        chunks = tracing::field::Empty,
        descriptors = tracing::field::Empty,
        descriptor_elements = tracing::field::Empty,
        descriptor_bytes = tracing::field::Empty,
        descriptor_storage_id = tracing::field::Empty,
        pivots = tracing::field::Empty,
        pivot_elements = tracing::field::Empty,
        pivot_bytes = tracing::field::Empty,
        pivot_storage_id = tracing::field::Empty,
        chunk_offset_elements = tracing::field::Empty,
        chunk_offset_bytes = tracing::field::Empty,
        chunk_offset_storage_id = tracing::field::Empty,
        work_items = tracing::field::Empty,
        work_item_elements = tracing::field::Empty,
        work_item_bytes = tracing::field::Empty,
        work_item_storage_id = tracing::field::Empty,
        address_offset_elements = tracing::field::Empty,
        address_offset_bytes = tracing::field::Empty,
        address_offset_storage_id = tracing::field::Empty,
        max_descriptors_per_chunk = tracing::field::Empty,
        max_pivots_per_chunk = tracing::field::Empty,
        first_push_pc = tracing::field::Empty,
        source_generation = tracing::field::Empty,
        source_completion_serial = tracing::field::Empty,
        source_rows_storage_id = tracing::field::Empty,
        source_claim_storage_id = tracing::field::Empty,
        topology_completion_serial = tracing::field::Empty,
        shared_source_row_scans = tracing::field::Empty,
        additional_source_row_scans = tracing::field::Empty,
        extra_source_scans = tracing::field::Empty,
        source_windows = tracing::field::Empty,
        member_upload_bytes = tracing::field::Empty,
        complete_overwrite = tracing::field::Empty,
        covered_rows = tracing::field::Empty,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn record_bytecode_stage1_topology_span(
    source: &InstructionReadRafStage1Owner,
    topology: Option<&BytecodeAddressStage1TopologyOwner>,
    physical_rows: usize,
) {
    let span = bytecode_stage1_topology_publish_span(topology.is_some(), physical_rows);
    let source = source.receipt();
    let _ = span.record("source_generation", source.source_generation());
    let _ = span.record("source_completion_serial", source.completion_serial());
    let _ = span.record("source_rows_storage_id", source.row_allocation_identity());
    let _ = span.record(
        "source_claim_storage_id",
        source.claim_allocation_identity(),
    );
    let _ = span.record("source_windows", source.rows());
    let _ = span.record("shared_source_row_scans", 1usize);
    let _ = span.record("additional_source_row_scans", 0usize);
    let _ = span.record("extra_source_scans", 0usize);
    let _ = span.record("member_upload_bytes", 0usize);
    let Some(topology) = topology else {
        for field in [
            "chunks",
            "descriptors",
            "descriptor_elements",
            "descriptor_bytes",
            "descriptor_storage_id",
            "pivots",
            "pivot_elements",
            "pivot_bytes",
            "pivot_storage_id",
            "chunk_offset_elements",
            "chunk_offset_bytes",
            "chunk_offset_storage_id",
            "work_items",
            "work_item_elements",
            "work_item_bytes",
            "work_item_storage_id",
            "address_offset_elements",
            "address_offset_bytes",
            "address_offset_storage_id",
            "max_descriptors_per_chunk",
            "max_pivots_per_chunk",
            "first_push_pc",
            "topology_completion_serial",
            "covered_rows",
        ] {
            let _ = span.record(field, 0usize);
        }
        let _ = span.record("complete_overwrite", false);
        let _entered = span.enter();
        return;
    };
    let receipt = topology.receipt();
    let values = [
        ("chunks", receipt.chunks()),
        ("descriptors", receipt.descriptors()),
        ("descriptor_elements", receipt.descriptor_elements()),
        ("descriptor_bytes", receipt.descriptor_bytes()),
        (
            "descriptor_storage_id",
            receipt.descriptor_allocation_identity(),
        ),
        ("pivots", receipt.pivots()),
        ("pivot_elements", receipt.pivot_elements()),
        ("pivot_bytes", receipt.pivot_bytes()),
        ("pivot_storage_id", receipt.pivot_allocation_identity()),
        ("chunk_offset_elements", receipt.chunk_offset_elements()),
        ("chunk_offset_bytes", receipt.chunk_offset_bytes()),
        (
            "chunk_offset_storage_id",
            receipt.chunk_offset_allocation_identity(),
        ),
        ("work_items", receipt.work_items()),
        ("work_item_elements", receipt.work_items()),
        ("work_item_bytes", receipt.work_item_bytes()),
        (
            "work_item_storage_id",
            receipt.work_item_allocation_identity(),
        ),
        ("address_offset_elements", receipt.address_offset_elements()),
        ("address_offset_bytes", receipt.address_offset_bytes()),
        (
            "address_offset_storage_id",
            receipt.address_offset_allocation_identity(),
        ),
        (
            "max_descriptors_per_chunk",
            receipt.max_descriptors_per_chunk(),
        ),
        ("max_pivots_per_chunk", receipt.max_pivots_per_chunk()),
        ("first_push_pc", receipt.first_push_pc()),
        (
            "topology_completion_serial",
            receipt.completion_serial() as usize,
        ),
        ("covered_rows", receipt.covered_rows()),
    ];
    for (field, value) in values {
        let _ = span.record(field, value);
    }
    let _ = span.record("complete_overwrite", receipt.complete_overwrite());
    let _entered = span.enter();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct InstructionReadRafStage1Ready {
    pub(crate) owner: InstructionReadRafStage1Owner,
    pub(crate) bytecode_topology: Option<BytecodeAddressStage1TopologyOwner>,
    pub(crate) registers_read_write: Option<RegistersReadWriteStage1Source>,
    pub(crate) registers_val: Option<RegistersValInstructionSourceRequest>,
    pub(crate) ram_access: Option<RamAccessCollection>,
    pub(crate) ram_read_write_records: Option<RamReadWriteRecordCollection>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
type Stage1OwnerPreparedRows = (SpartanOuterUniskipRows, InstructionReadRafStage1Ready);

#[cfg(all(feature = "metal", target_os = "macos"))]
type ShiftStage1OwnerPreparedRows = (
    SpartanOuterUniskipRows,
    SpartanShiftResidentRows,
    InstructionReadRafStage1Ready,
);

#[cfg(all(feature = "metal", target_os = "macos"))]
#[expect(
    clippy::too_many_arguments,
    reason = "Stage-1 admission selects each optional resident owner independently"
)]
pub(crate) fn prepare_metal_spartan_outer_stage1_owner_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
    prepare_bytecode_carrier: bool,
    prepare_registers_read_write: bool,
    prepare_registers_val: bool,
    prepare_ram_access: bool,
    prepare_ram_read_write_records: bool,
) -> Result<Stage1OwnerPreparedRows, MetalSpartanDenseRowsError> {
    let owned = witness
        .owned_rows()
        .filter(|rows| cycles <= rows.cycles())
        .ok_or(MetalSpartanDenseRowsError::Kernel(
            KernelError::InvariantViolation {
                reason: "InstructionReadRAF Stage-1 ownership requires a random-access witness",
            },
        ))?;
    let explicit_rows = owned.physical_rows().min(cycles);
    let access = owned.view();
    let padding = Stage1PaddingRows::new(&access, explicit_rows, cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let span = stage1_owner_rows_span(
        cycles,
        explicit_rows,
        padding.source_window_count(explicit_rows),
    );
    let _entered = span.enter();
    let topology_span = bytecode_stage1_topology_span(prepare_bytecode_carrier, explicit_rows);
    let _topology_entered = topology_span.enter();
    let mut source = context
        .prepare_instruction_read_raf_stage1_storage(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut bytecode_topology = prepare_bytecode_carrier
        .then(|| context.prepare_bytecode_address_stage1_topology_storage(cycles, explicit_rows))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut ram_access = prepare_ram_access
        .then(|| RamAccessCollectionStorage::new(cycles, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let mut ram_read_write_records = prepare_ram_read_write_records
        .then(|| {
            RamReadWriteRecordCollectionStorage::new(
                cycles,
                INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
                (cycles.ilog2() as usize).min(RAM_READ_WRITE_CYCLE_TILE_LOG2),
            )
        })
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let mut registers_read_write = prepare_registers_read_write
        .then(|| context.prepare_registers_read_write_stage1_storage(cycles))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let outer_rows = context
        .prepare_spartan_outer_uniskip_rows_with_fill(
            cycles,
            |instruction_input, successor, cold| {
                with_stage1_owner_chunks(
                    &mut source,
                    bytecode_topology.as_mut(),
                    ram_access.as_mut(),
                    ram_read_write_records.as_mut(),
                    registers_read_write.as_mut(),
                    |owner_chunks| {
                        let fill_chunk =
                            |chunk: usize,
                             instruction_input: &mut [InstructionInputRow],
                             successor: &mut [SpartanOuterUniskipSuccessorRow],
                             cold: &mut [SpartanOuterUniskipColdRow],
                             owner: &mut Stage1OwnerChunkWriters<'_, '_, '_, '_, '_, '_>,
                             bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch|
                             -> Result<(), MetalError> {
                                if instruction_input.len() != owner.len()
                                    || successor.len() != owner.len()
                                    || cold.len() != owner.len()
                                {
                                    return Err(MetalError::InvalidInstructionReadRafGrouped(
                                        "Stage-1 owner chunks disagree on row count".to_owned(),
                                    ));
                                }
                                let chunk_start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                                let parts = stage1_chunk_parts(
                                    chunk_start,
                                    owner.len(),
                                    explicit_rows,
                                    cycles,
                                );
                                for offset in 0..parts.physical {
                                    let row_index = chunk_start + offset;
                                    let projected: Stage1ProjectionRow =
                                        access.window(row_index).map_err(|error| {
                                            MetalError::SpartanOuterRowExtraction {
                                                row: row_index,
                                                message: error.to_string(),
                                            }
                                        })?;
                                    let (input, residual_row) =
                                        SpartanOuterUniskipRow::from_spartan_outer(
                                            &projected.outer,
                                        )
                                        .split();
                                    let (successor_row, cold_row) = residual_row.partition();
                                    instruction_input[offset] = input.with_register_indices(
                                        projected.register_indices[0],
                                        projected.register_indices[1],
                                        projected.register_write.map(|(index, _, _)| index),
                                    )?;
                                    successor[offset] = successor_row;
                                    cold[offset] = cold_row;
                                    owner.push(
                                        row_index,
                                        explicit_rows,
                                        projected.instruction,
                                        projected.ram_access,
                                        projected.register_indices,
                                        projected.register_write,
                                        bytecode_scratch,
                                    )?;
                                }
                                let mut padding_start = parts.physical;
                                if parts.regular_padding != 0 {
                                    let regular = padding.regular.ok_or_else(|| {
                                        MetalError::InvalidInstructionReadRafGrouped(
                                            "regular Stage-1 padding template is missing"
                                                .to_owned(),
                                        )
                                    })?;
                                    fill_stage1_outer_padding(
                                        instruction_input,
                                        successor,
                                        cold,
                                        padding_start,
                                        parts.regular_padding,
                                        &regular,
                                    );
                                    owner.fill_padding(&regular, parts.regular_padding)?;
                                    padding_start += parts.regular_padding;
                                }
                                if parts.terminal_padding != 0 {
                                    let terminal = padding.terminal.ok_or_else(|| {
                                        MetalError::InvalidInstructionReadRafGrouped(
                                            "terminal Stage-1 padding template is missing"
                                                .to_owned(),
                                        )
                                    })?;
                                    fill_stage1_outer_padding(
                                        instruction_input,
                                        successor,
                                        cold,
                                        padding_start,
                                        parts.terminal_padding,
                                        &terminal,
                                    );
                                    owner.fill_padding(&terminal, parts.terminal_padding)?;
                                }
                                owner.finish(bytecode_scratch)
                            };
                        #[cfg(feature = "parallel")]
                    instruction_input
                        .par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                        .zip(successor.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                        .zip(cold.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                        .zip(owner_chunks.par_iter_mut())
                        .enumerate()
                        .try_for_each_init(
                            BytecodeAddressStage1TopologyScratch::new,
                            |scratch,
                             (chunk, (((instruction_input, successor), cold), owner))| {
                                fill_chunk(
                                    chunk,
                                    instruction_input,
                                    successor,
                                    cold,
                                    owner,
                                    scratch,
                                )
                            },
                        )?;
                        #[cfg(not(feature = "parallel"))]
                        {
                            let mut scratch = BytecodeAddressStage1TopologyScratch::new();
                            for (chunk, (((instruction_input, successor), cold), owner)) in
                                instruction_input
                                    .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                                    .zip(
                                        successor
                                            .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS),
                                    )
                                    .zip(cold.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                                    .zip(owner_chunks.iter_mut())
                                    .enumerate()
                            {
                                fill_chunk(
                                    chunk,
                                    instruction_input,
                                    successor,
                                    cold,
                                    owner,
                                    &mut scratch,
                                )?;
                            }
                        }
                        Ok(())
                    },
                )
            },
        )
        .map_err(MetalSpartanDenseRowsError::Metal)?
        .with_explicit_rows(explicit_rows)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let owner = source.seal().map_err(MetalSpartanDenseRowsError::Metal)?;
    let registers_read_write = registers_read_write
        .map(|storage| {
            storage.seal(
                outer_rows.clone_instruction_input_rows(),
                &owner,
                explicit_rows,
            )
        })
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let source_storage_ids = [
        outer_rows.instruction_input_allocation_identity(),
        outer_rows.allocation_identity(),
    ];
    let source_storage_bytes = [
        instruction_input_row_bytes(cycles).map_err(MetalSpartanDenseRowsError::Metal)?,
        spartan_outer_uniskip_successor_row_bytes(cycles)
            .map_err(MetalSpartanDenseRowsError::Metal)?,
    ];
    let bytecode_topology = bytecode_topology
        .map(|topology| topology.seal(&owner))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let ram_access = ram_access
        .map(RamAccessCollectionStorage::seal)
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let ram_read_write_records = ram_read_write_records
        .map(RamReadWriteRecordCollectionStorage::seal)
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let registers_val = prepare_registers_val
        .then(|| {
            context.prepare_registers_val_instruction_source_request(
                cycles,
                explicit_rows,
                source_storage_ids[0],
                source_storage_bytes[0],
                source_storage_ids[1],
                source_storage_bytes[1],
                owner.receipt(),
            )
        })
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    record_bytecode_stage1_topology_span(&owner, bytecode_topology.as_ref(), explicit_rows);
    let prepared = InstructionReadRafStage1Ready {
        owner,
        bytecode_topology,
        registers_read_write,
        registers_val,
        ram_access,
        ram_read_write_records,
    };
    let _ = span.record(
        "compact_rows_storage_id",
        outer_rows.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", outer_rows.allocation_identity());
    Ok((outer_rows, prepared))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_spartan_outer_shift_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<(SpartanOuterUniskipRows, SpartanShiftResidentRows), MetalSpartanDenseRowsError> {
    context
        .validate_spartan_outer_uniskip_shift_rows_capacity(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let rows = RowsStore::resolve(witness, cycles).map_err(MetalSpartanDenseRowsError::Kernel)?;
    let access = rows.access();
    let explicit_rows = rows.explicit_rows();
    let source_kind = rows.production_source_kind();
    let host_repack_rows = rows.host_repack_rows();
    let span = tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind,
        witness_row_extractions = cycles,
        residual_rows_written = cycles,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 112,
        compact_allocations = 1,
        residual_allocations = 1,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = tracing::field::Empty,
        resident_rows = cycles,
        explicit_rows,
    );
    let _entered = span.enter();
    let (outer_rows, shift_rows) = context
        .prepare_spartan_outer_uniskip_rows_with_shift_fill(
            cycles,
            |instruction_input, successor, cold, unexpanded_pc, pc, flags| {
                #[cfg(feature = "parallel")]
                {
                    instruction_input
                        .par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD)
                        .zip(successor.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(cold.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(unexpanded_pc.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(pc.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(flags.par_iter_mut())
                        .enumerate()
                        .try_for_each(
                            |(
                                word_index,
                                (
                                    ((((instruction_input, successor), cold), unexpanded_pc), pc),
                                    flags,
                                ),
                            )|
                             -> Result<(), MetalError> {
                                let mut packed_flags = SpartanShiftFlagWord::default();
                                for offset in 0..instruction_input.len() {
                                    let row_index =
                                        word_index * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD + offset;
                                    let row = access.row(row_index).map_err(|error| {
                                        MetalError::SpartanOuterRowExtraction {
                                            row: row_index,
                                            message: error.to_string(),
                                        }
                                    })?;
                                    let (input, residual) =
                                        SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                                    let (successor_row, cold_row) = residual.partition();
                                    instruction_input[offset] = input;
                                    successor[offset] = successor_row;
                                    cold[offset] = cold_row;
                                    write_metal_spartan_shift_row(
                                        &row,
                                        offset,
                                        &mut unexpanded_pc[offset],
                                        &mut pc[offset],
                                        &mut packed_flags,
                                    );
                                }
                                *flags = packed_flags;
                                Ok(())
                            },
                        )?;
                }
                #[cfg(not(feature = "parallel"))]
                {
                    flags.fill(SpartanShiftFlagWord::default());
                    for row_index in 0..cycles {
                        let row = access.row(row_index).map_err(|error| {
                            MetalError::SpartanOuterRowExtraction {
                                row: row_index,
                                message: error.to_string(),
                            }
                        })?;
                        let (input, residual) =
                            SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                        let (successor_row, cold_row) = residual.partition();
                        instruction_input[row_index] = input;
                        successor[row_index] = successor_row;
                        cold[row_index] = cold_row;
                        write_metal_spartan_shift_row(
                            &row,
                            row_index % SPARTAN_SHIFT_FLAG_ROWS_PER_WORD,
                            &mut unexpanded_pc[row_index],
                            &mut pc[row_index],
                            &mut flags[row_index / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD],
                        );
                    }
                }
                Ok(())
            },
        )
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let prepared = (
        outer_rows
            .with_explicit_rows(explicit_rows)
            .map_err(MetalSpartanDenseRowsError::Metal)?,
        shift_rows,
    );
    let _ = span.record(
        "compact_rows_storage_id",
        prepared.0.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", prepared.0.allocation_identity());
    Ok(prepared)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[expect(
    clippy::too_many_arguments,
    reason = "Stage-1 admission selects each optional resident owner independently"
)]
pub(crate) fn prepare_metal_spartan_outer_shift_stage1_owner_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
    prepare_bytecode_carrier: bool,
    prepare_registers_read_write: bool,
    prepare_registers_val: bool,
    prepare_ram_access: bool,
    prepare_ram_read_write_records: bool,
) -> Result<ShiftStage1OwnerPreparedRows, MetalSpartanDenseRowsError> {
    context
        .validate_spartan_outer_uniskip_shift_rows_capacity(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let owned = witness
        .owned_rows()
        .filter(|rows| cycles <= rows.cycles())
        .ok_or(MetalSpartanDenseRowsError::Kernel(
            KernelError::InvariantViolation {
                reason: "InstructionReadRAF Stage-1 ownership requires a random-access witness",
            },
        ))?;
    let explicit_rows = owned.physical_rows().min(cycles);
    let access = owned.view();
    let padding = Stage1PaddingRows::new(&access, explicit_rows, cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let span = stage1_owner_rows_span(
        cycles,
        explicit_rows,
        padding.source_window_count(explicit_rows),
    );
    let _entered = span.enter();
    let topology_span = bytecode_stage1_topology_span(prepare_bytecode_carrier, explicit_rows);
    let _topology_entered = topology_span.enter();
    let mut source = context
        .prepare_instruction_read_raf_stage1_storage(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut bytecode_topology = prepare_bytecode_carrier
        .then(|| context.prepare_bytecode_address_stage1_topology_storage(cycles, explicit_rows))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut ram_access = prepare_ram_access
        .then(|| RamAccessCollectionStorage::new(cycles, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let mut ram_read_write_records = prepare_ram_read_write_records
        .then(|| {
            RamReadWriteRecordCollectionStorage::new(
                cycles,
                INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
                (cycles.ilog2() as usize).min(RAM_READ_WRITE_CYCLE_TILE_LOG2),
            )
        })
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let mut registers_read_write = prepare_registers_read_write
        .then(|| context.prepare_registers_read_write_stage1_storage(cycles))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let (outer_rows, shift_rows) = context
        .prepare_spartan_outer_uniskip_rows_with_shift_fill(
            cycles,
            |instruction_input, successor, cold, unexpanded_pc, pc, flags| {
                with_stage1_owner_chunks(
                    &mut source,
                    bytecode_topology.as_mut(),
                    ram_access.as_mut(),
                    ram_read_write_records.as_mut(),
                    registers_read_write.as_mut(),
                    |owner_chunks| {
                        let fill_chunk =
                            |chunk: usize,
                             instruction_input: &mut [InstructionInputRow],
                             successor: &mut [SpartanOuterUniskipSuccessorRow],
                             cold: &mut [SpartanOuterUniskipColdRow],
                             unexpanded_pc: &mut [u64],
                             pc: &mut [u64],
                             flags: &mut [SpartanShiftFlagWord],
                             owner: &mut Stage1OwnerChunkWriters<'_, '_, '_, '_, '_, '_>,
                             bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch|
                             -> Result<(), MetalError> {
                                if instruction_input.len() != owner.len()
                                    || successor.len() != owner.len()
                                    || cold.len() != owner.len()
                                    || unexpanded_pc.len() != owner.len()
                                    || pc.len() != owner.len()
                                    || flags.len() != owner.len() / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD
                                {
                                    return Err(MetalError::InvalidInstructionReadRafGrouped(
                                        "Stage-1 owner/Shift chunks disagree on row count"
                                            .to_owned(),
                                    ));
                                }
                                flags.fill(SpartanShiftFlagWord::default());
                                let chunk_start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                                let parts = stage1_chunk_parts(
                                    chunk_start,
                                    owner.len(),
                                    explicit_rows,
                                    cycles,
                                );
                                for offset in 0..parts.physical {
                                    let row_index = chunk_start + offset;
                                    let projected: Stage1ProjectionRow =
                                        access.window(row_index).map_err(|error| {
                                            MetalError::SpartanOuterRowExtraction {
                                                row: row_index,
                                                message: error.to_string(),
                                            }
                                        })?;
                                    let (input, residual_row) =
                                        SpartanOuterUniskipRow::from_spartan_outer(
                                            &projected.outer,
                                        )
                                        .split();
                                    let (successor_row, cold_row) = residual_row.partition();
                                    instruction_input[offset] = input.with_register_indices(
                                        projected.register_indices[0],
                                        projected.register_indices[1],
                                        projected.register_write.map(|(index, _, _)| index),
                                    )?;
                                    successor[offset] = successor_row;
                                    cold[offset] = cold_row;
                                    write_metal_spartan_shift_row(
                                        &projected.outer,
                                        offset % SPARTAN_SHIFT_FLAG_ROWS_PER_WORD,
                                        &mut unexpanded_pc[offset],
                                        &mut pc[offset],
                                        &mut flags[offset / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD],
                                    );
                                    owner.push(
                                        row_index,
                                        explicit_rows,
                                        projected.instruction,
                                        projected.ram_access,
                                        projected.register_indices,
                                        projected.register_write,
                                        bytecode_scratch,
                                    )?;
                                }
                                let mut padding_start = parts.physical;
                                if parts.regular_padding != 0 {
                                    let regular = padding.regular.ok_or_else(|| {
                                        MetalError::InvalidInstructionReadRafGrouped(
                                            "regular Stage-1 padding template is missing"
                                                .to_owned(),
                                        )
                                    })?;
                                    fill_stage1_outer_padding(
                                        instruction_input,
                                        successor,
                                        cold,
                                        padding_start,
                                        parts.regular_padding,
                                        &regular,
                                    );
                                    fill_stage1_shift_padding(
                                        unexpanded_pc,
                                        pc,
                                        flags,
                                        padding_start,
                                        parts.regular_padding,
                                        &regular,
                                    );
                                    owner.fill_padding(&regular, parts.regular_padding)?;
                                    padding_start += parts.regular_padding;
                                }
                                if parts.terminal_padding != 0 {
                                    let terminal = padding.terminal.ok_or_else(|| {
                                        MetalError::InvalidInstructionReadRafGrouped(
                                            "terminal Stage-1 padding template is missing"
                                                .to_owned(),
                                        )
                                    })?;
                                    fill_stage1_outer_padding(
                                        instruction_input,
                                        successor,
                                        cold,
                                        padding_start,
                                        parts.terminal_padding,
                                        &terminal,
                                    );
                                    fill_stage1_shift_padding(
                                        unexpanded_pc,
                                        pc,
                                        flags,
                                        padding_start,
                                        parts.terminal_padding,
                                        &terminal,
                                    );
                                    owner.fill_padding(&terminal, parts.terminal_padding)?;
                                }
                                owner.finish(bytecode_scratch)
                            };
                        let flags_per_chunk = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS
                            / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD;
                        #[cfg(feature = "parallel")]
                        instruction_input
                            .par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                            .zip(successor.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                            .zip(cold.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                            .zip(
                                unexpanded_pc
                                    .par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS),
                            )
                            .zip(pc.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                            .zip(flags.par_chunks_mut(flags_per_chunk))
                            .zip(owner_chunks.par_iter_mut())
                            .enumerate()
                            .try_for_each_init(
                                BytecodeAddressStage1TopologyScratch::new,
                                |bytecode_scratch,
                                 (
                                    chunk,
                                    (
                                        (
                                            (
                                                (
                                                    ((instruction_input, successor), cold),
                                                    unexpanded_pc,
                                                ),
                                                pc,
                                            ),
                                            flags,
                                        ),
                                        owner,
                                    ),
                                )| {
                                    fill_chunk(
                                        chunk,
                                        instruction_input,
                                        successor,
                                        cold,
                                        unexpanded_pc,
                                        pc,
                                        flags,
                                        owner,
                                        bytecode_scratch,
                                    )
                                },
                            )?;
                        #[cfg(not(feature = "parallel"))]
                        {
                            let mut bytecode_scratch = BytecodeAddressStage1TopologyScratch::new();
                            for (
                                chunk,
                                (
                                    (
                                        (
                                            (((instruction_input, successor), cold), unexpanded_pc),
                                            pc,
                                        ),
                                        flags,
                                    ),
                                    owner,
                                ),
                            ) in instruction_input
                                .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                                .zip(successor.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                                .zip(cold.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                                .zip(
                                    unexpanded_pc
                                        .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS),
                                )
                                .zip(pc.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                                .zip(flags.chunks_mut(flags_per_chunk))
                                .zip(owner_chunks.iter_mut())
                                .enumerate()
                            {
                                fill_chunk(
                                    chunk,
                                    instruction_input,
                                    successor,
                                    cold,
                                    unexpanded_pc,
                                    pc,
                                    flags,
                                    owner,
                                    &mut bytecode_scratch,
                                )?;
                            }
                        }
                        Ok(())
                    },
                )
            },
        )
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let outer_rows = outer_rows
        .with_explicit_rows(explicit_rows)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let owner = source.seal().map_err(MetalSpartanDenseRowsError::Metal)?;
    let registers_read_write = registers_read_write
        .map(|storage| {
            storage.seal(
                outer_rows.clone_instruction_input_rows(),
                &owner,
                explicit_rows,
            )
        })
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let source_storage_ids = [
        outer_rows.instruction_input_allocation_identity(),
        outer_rows.allocation_identity(),
    ];
    let source_storage_bytes = [
        instruction_input_row_bytes(cycles).map_err(MetalSpartanDenseRowsError::Metal)?,
        spartan_outer_uniskip_successor_row_bytes(cycles)
            .map_err(MetalSpartanDenseRowsError::Metal)?,
    ];
    let bytecode_topology = bytecode_topology
        .map(|topology| topology.seal(&owner))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let ram_access = ram_access
        .map(RamAccessCollectionStorage::seal)
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let ram_read_write_records = ram_read_write_records
        .map(RamReadWriteRecordCollectionStorage::seal)
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let registers_val = prepare_registers_val
        .then(|| {
            context.prepare_registers_val_instruction_source_request(
                cycles,
                explicit_rows,
                source_storage_ids[0],
                source_storage_bytes[0],
                source_storage_ids[1],
                source_storage_bytes[1],
                owner.receipt(),
            )
        })
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    record_bytecode_stage1_topology_span(&owner, bytecode_topology.as_ref(), explicit_rows);
    let prepared = InstructionReadRafStage1Ready {
        owner,
        bytecode_topology,
        registers_read_write,
        registers_val,
        ram_access,
        ram_read_write_records,
    };
    let _ = span.record(
        "compact_rows_storage_id",
        outer_rows.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", outer_rows.allocation_identity());
    Ok((outer_rows, shift_rows, prepared))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn write_metal_spartan_shift_row(
    row: &SpartanOuterRow,
    bit: usize,
    unexpanded_pc: &mut u64,
    pc: &mut u64,
    flags: &mut SpartanShiftFlagWord,
) {
    *unexpanded_pc = row.unexpanded_pc.0;
    *pc = row.pc.0;
    let mask = 1u32 << bit;
    flags.is_virtual |= u32::from(row.virtual_instruction.0) * mask;
    flags.is_first_in_sequence |= u32::from(row.is_first_in_sequence.0) * mask;
    flags.is_noop |= u32::from(row.is_noop.0) * mask;
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_instruction_input_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<InstructionInputRows, KernelError<AkitaField>> {
    let rows = RowsStore::resolve(witness, cycles)?;
    let access = rows.access();
    let explicit_rows = rows.explicit_rows();
    let source_kind = rows.production_source_kind();
    let host_repack_rows = rows.host_repack_rows();
    let span = tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind,
        witness_row_extractions = cycles,
        residual_rows_written = 0,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 0,
        compact_allocations = 1,
        residual_allocations = 0,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = 0,
        resident_rows = cycles,
        explicit_rows,
    );
    let _entered = span.enter();
    let prepared = context
        .prepare_instruction_input_rows_with_fill(cycles, |destination| {
            #[cfg(feature = "parallel")]
            {
                destination.par_iter_mut().enumerate().try_for_each(
                    |(row_index, destination)| -> Result<(), MetalError> {
                        let row = access.row(row_index).map_err(|error| {
                            MetalError::SpartanOuterRowExtraction {
                                row: row_index,
                                message: error.to_string(),
                            }
                        })?;
                        *destination = InstructionInputRow::from_spartan_outer(&row);
                        Ok(())
                    },
                )?;
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (row_index, destination) in destination.iter_mut().enumerate() {
                    let row = access.row(row_index).map_err(|error| {
                        MetalError::SpartanOuterRowExtraction {
                            row: row_index,
                            message: error.to_string(),
                        }
                    })?;
                    *destination = InstructionInputRow::from_spartan_outer(&row);
                }
            }
            Ok(())
        })
        .map_err(metal_outer_error)?;
    let _ = span.record("compact_rows_storage_id", prepared.allocation_identity());
    Ok(prepared)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn prepare_metal_spartan_outer_rows(
    context: &SolinasMetal,
    rows: &RowsStore,
    cycles: usize,
) -> Result<SpartanOuterUniskipRows, KernelError<AkitaField>> {
    let access = rows.access();
    let explicit_rows = rows.explicit_rows();
    let source_kind = rows.production_source_kind();
    let host_repack_rows = rows.host_repack_rows();
    let span = tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind,
        witness_row_extractions = cycles,
        residual_rows_written = cycles,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 112,
        compact_allocations = 1,
        residual_allocations = 1,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = tracing::field::Empty,
        resident_rows = cycles,
        explicit_rows,
    );
    let _entered = span.enter();
    let prepared = context
        .prepare_spartan_outer_uniskip_rows_with_fill(cycles, |instruction_input, successor, cold| {
            #[cfg(feature = "parallel")]
            {
                instruction_input
                    .par_iter_mut()
                    .zip(successor.par_iter_mut())
                    .zip(cold.par_iter_mut())
                    .enumerate()
                    .try_for_each(
                        |(row_index, ((instruction_input, successor), cold))| -> Result<(), MetalError> {
                            let row = access.row(row_index).map_err(|error| {
                                MetalError::SpartanOuterRowExtraction {
                                    row: row_index,
                                    message: error.to_string(),
                                }
                            })?;
                            let (input, residual) =
                                SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                            let (successor_row, cold_row) = residual.partition();
                            *instruction_input = input;
                            *successor = successor_row;
                            *cold = cold_row;
                            Ok(())
                        },
                    )?;
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (row_index, ((instruction_input, successor), cold)) in instruction_input
                    .iter_mut()
                    .zip(successor)
                    .zip(cold)
                    .enumerate()
                {
                    let row = access.row(row_index).map_err(|error| {
                        MetalError::SpartanOuterRowExtraction {
                            row: row_index,
                            message: error.to_string(),
                        }
                    })?;
                    let (input, residual) =
                        SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                    let (successor_row, cold_row) = residual.partition();
                    *instruction_input = input;
                    *successor = successor_row;
                    *cold = cold_row;
                }
            }
            Ok(())
        })
        .map_err(metal_outer_error)?
        .with_explicit_rows(explicit_rows)
        .map_err(metal_outer_error)?;
    let _ = span.record(
        "compact_rows_storage_id",
        prepared.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", prepared.allocation_identity());
    Ok(prepared)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_outer_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

impl<F: Field> UniskipKernel<F, OuterRemainder<F>> for OptimizedOuterUniskip {
    fn prepare_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        prepare_instruction_input_rows(session, witness, 1usize << log_t)
    }

    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        let rows = RowsStore::resolve(witness, 1usize << log_t)?;
        Self::prepare_from_store(session, log_t, tau, rows)
    }

    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        _late_tau: &[F],
        _known_values: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let carry =
            session
                .state::<SpartanOuterCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason:
                        "the outer uni-skip slot parked no carry for the first-round polynomial",
                })?;
        // The reference's exact assembly path, fed the same t1 node values.
        let tau_high = carry.tau[carry.log_t + 1];
        let kernel_values = centered_lagrange_evals::<F>(DOMAIN, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &carry.t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }
}

/// The stage-1 remainder slot: reclaims the uni-skip carry and builds the
/// linear-time round kernel.
pub struct OptimizedOuterRemainder;

impl<F: Field> PrepareKernel<F, OuterRemainder<F>> for OptimizedOuterRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        // A mixed Metal-uni-skip/CPU-remainder registry must not retain the
        // now-unused residual allocation through the rest of the proof.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        drop(session.take::<SpartanOuterUniskipRows>());
        let carry =
            session
                .take::<SpartanOuterCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "the outer uni-skip slot parked no carry for the remainder member",
                })?;
        Ok(Box::new(OuterRemainderKernel::prepare(carry, &inputs)?))
    }
}

/// The `Az`/`Bz` linear forms folded at both stream values — the closed forms
/// of the relation's derived leaves after the stream bind, kept for
/// [`SumcheckKernel::validate_derived_tables`].
struct DerivedWeights<F> {
    az_weights: [Vec<F>; 2],
    bz_weights: [Vec<F>; 2],
    az_constant: [F; 2],
    bz_constant: [F; 2],
}

/// The linear-time outer remainder rounds over the joint `(cycle ‖ stream)`
/// domain (stream = index LSB, bound `LowToHigh`).
struct OuterRemainderKernel<F: Field> {
    rounds: usize,
    az: Polynomial<F>,
    bz: Polynomial<F>,
    scratch: Vec<F>,
    split_eq: GruenSplitEqPolynomial<F>,
    /// Round-0 endpoints, fused into the materialization pass.
    pending_endpoints: Option<(F, F)>,
    challenges: Vec<F>,
    rows: RowsStore,
    opening_ids: Vec<JoltOpeningId>,
    derived: DerivedWeights<F>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for OuterRemainderKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{gruen_heap_bytes, poly_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("az"), poly_heap_bytes(&self.az));
        visitor.visit_simple(allocative::Key::new("bz"), poly_heap_bytes(&self.bz));
        visitor.visit_simple(
            allocative::Key::new("scratch"),
            vec_heap_bytes(&self.scratch),
        );
        visitor.visit_simple(
            allocative::Key::new("split_eq"),
            gruen_heap_bytes(&self.split_eq),
        );
        visitor.visit_simple(
            allocative::Key::new("challenges"),
            vec_heap_bytes(&self.challenges),
        );
        visitor.visit_simple(allocative::Key::new("rows"), self.rows.heap_bytes());
        visitor.visit_simple(
            allocative::Key::new("opening_ids"),
            vec_heap_bytes(&self.opening_ids),
        );
        visitor.visit_simple(
            allocative::Key::new("derived"),
            self.derived
                .az_weights
                .iter()
                .chain(&self.derived.bz_weights)
                .map(vec_heap_bytes)
                .sum::<usize>(),
        );
        visitor.exit();
    }
}

impl<F: Field> OuterRemainderKernel<F> {
    fn prepare(
        carry: SpartanOuterCarry<F>,
        inputs: &ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Self, KernelError<F>> {
        let SpartanOuterCarry {
            log_t, tau, rows, ..
        } = carry;
        let rounds = inputs.relation.rounds();
        if rounds != log_t + 1 {
            return Err(KernelError::InvariantViolation {
                reason: "outer remainder rounds disagree with the uni-skip carry's log_t",
            });
        }
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let tau_high = tau[log_t + 1];
        let tau_low = &tau[..=log_t];
        let lagrange_r0 = centered_lagrange_evals::<F>(DOMAIN, uniskip_challenge)?;
        let kernel = centered_lagrange_kernel::<F>(DOMAIN, tau_high, uniskip_challenge)?;
        let split_eq = GruenSplitEqPolynomial::new_with_scaling(
            tau_low,
            BindingOrder::LowToHigh,
            Some(kernel),
        );

        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let opening_ids: Vec<JoltOpeningId> = dimensions
            .variables()
            .iter()
            .map(|&variable| outer_opening(variable))
            .collect();
        let derived = Self::derived_weights(uniskip_challenge, opening_ids.len())?;

        // Fused round-0 materialization: one pass over the typed rows writes
        // the bound Az/Bz tables and accumulates the first round's endpoints
        // q(0) = Σ_t E(t)·az₀·bz₀ and q(∞) = Σ_t E(t)·(az₁−az₀)(bz₁−bz₀).
        let cycles = 1usize << log_t;
        let mut az: Vec<F> = unsafe_allocate_zero_vec(2 * cycles);
        let mut bz: Vec<F> = unsafe_allocate_zero_vec(2 * cycles);
        let e_out = split_eq.e_out_current();
        let e_in = split_eq.e_in_current();
        let in_len = e_in.len();
        let width = 2 * in_len;
        let access = rows.access();
        let lagrange = &lagrange_r0;
        let block = |x_out: usize,
                     az_chunk: &mut [F],
                     bz_chunk: &mut [F]|
         -> Result<(F, F), WitnessError> {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            for x_in in 0..in_len {
                let t = x_out * in_len + x_in;
                let row = access.row(t)?;
                let values = row_group_values(&row);
                let (az_zero, bz_zero) = fold_group(lagrange, &values.a_first, &values.b_first);
                let (az_one, bz_one) = fold_group(
                    &lagrange[..SECOND_GROUP_LEN],
                    &values.a_second,
                    &values.b_second,
                );
                az_chunk[2 * x_in] = az_zero;
                az_chunk[2 * x_in + 1] = az_one;
                bz_chunk[2 * x_in] = bz_zero;
                bz_chunk[2 * x_in + 1] = bz_one;
                let e = e_in[x_in];
                inner_zero += e * (az_zero * bz_zero);
                inner_infinity += e * ((az_one - az_zero) * (bz_one - bz_zero));
            }
            Ok((e_out[x_out] * inner_zero, e_out[x_out] * inner_infinity))
        };
        let add = |left: (F, F), right: (F, F)| (left.0 + right.0, left.1 + right.1);

        #[cfg(feature = "parallel")]
        let endpoints = az
            .par_chunks_mut(width)
            .zip(bz.par_chunks_mut(width))
            .enumerate()
            .map(|(x_out, (az_chunk, bz_chunk))| block(x_out, az_chunk, bz_chunk))
            .try_reduce(
                || (F::zero(), F::zero()),
                |left, right| Ok(add(left, right)),
            )?;
        #[cfg(not(feature = "parallel"))]
        let endpoints = {
            let mut folded = (F::zero(), F::zero());
            for (x_out, (az_chunk, bz_chunk)) in
                az.chunks_mut(width).zip(bz.chunks_mut(width)).enumerate()
            {
                folded = add(folded, block(x_out, az_chunk, bz_chunk)?);
            }
            folded
        };
        Ok(Self {
            rounds,
            az: Polynomial::new(az),
            bz: Polynomial::new(bz),
            scratch: Vec::new(),
            split_eq,
            pending_endpoints: Some(endpoints),
            challenges: Vec::with_capacity(rounds),
            rows,
            opening_ids,
            derived,
        })
    }

    /// Az/Bz column weights at both stream values, from the same `jolt-r1cs`
    /// sources the verifier's coefficient build uses.
    fn derived_weights(
        uniskip_challenge: F,
        variable_count: usize,
    ) -> Result<DerivedWeights<F>, KernelError<F>> {
        let matrices = spartan_outer_constraints::<F>();
        let columns: Vec<usize> = (1..=variable_count).collect();
        let mut az_weights = [Vec::new(), Vec::new()];
        let mut bz_weights = [Vec::new(), Vec::new()];
        let mut az_constant = [F::zero(); 2];
        let mut bz_constant = [F::zero(); 2];
        for (index, stream) in [F::zero(), F::one()].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
            let weighted = matrices.weighted_columns(&weights, &columns)?;
            az_weights[index] = weighted.a;
            bz_weights[index] = weighted.b;
            let constants = matrices.public_column_contributions(&weights, 0, F::one())?;
            az_constant[index] = constants.a;
            bz_constant[index] = constants.b;
        }
        Ok(DerivedWeights {
            az_weights,
            bz_weights,
            az_constant,
            bz_constant,
        })
    }

    /// The current round's endpoints `q(0)`, `q(∞)` over the remaining
    /// `(lo, hi)` pairs, eq-weighted by the split tensor.
    fn endpoints(&self) -> (F, F) {
        let az = self.az.evals();
        let bz = self.bz.evals();
        let e_out = self.split_eq.e_out_current();
        let e_in = self.split_eq.e_in_current();
        let in_len = e_in.len();
        debug_assert_eq!(e_out.len() * in_len * 2, az.len());

        let block = |x_out: usize| -> (F, F) {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            for (x_in, &e) in e_in.iter().enumerate() {
                let pair = 2 * (x_out * in_len + x_in);
                let az_low = az[pair];
                let az_high = az[pair + 1];
                let bz_low = bz[pair];
                let bz_high = bz[pair + 1];
                inner_zero += e * (az_low * bz_low);
                inner_infinity += e * ((az_high - az_low) * (bz_high - bz_low));
            }
            (e_out[x_out] * inner_zero, e_out[x_out] * inner_infinity)
        };
        let add = |left: (F, F), right: (F, F)| (left.0 + right.0, left.1 + right.1);

        #[cfg(feature = "parallel")]
        {
            (0..e_out.len())
                .into_par_iter()
                .map(block)
                .reduce(|| (F::zero(), F::zero()), add)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..e_out.len())
                .map(block)
                .fold((F::zero(), F::zero()), add)
        }
    }

    fn bind(&mut self, challenge: F) {
        self.az
            .bind_low_to_high_reusing_scratch(challenge, &mut self.scratch);
        self.bz
            .bind_low_to_high_reusing_scratch(challenge, &mut self.scratch);
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
    }

    /// The 35 produced opening values at the bound cycle point: one
    /// eq-weighted walk over the typed rows (`compute_claimed_inputs`),
    /// mixed-width accumulators per input.
    #[tracing::instrument(skip_all, name = "SpartanOuter::claimed_inputs")]
    fn claimed_inputs(&self) -> Result<Vec<F>, WitnessError> {
        let reversed: Vec<F> = self.challenges[1..].iter().rev().copied().collect();
        let weights = {
            let _span = tracing::info_span!("SpartanOuter::claimed_input_weights").entered();
            EqPolynomial::<F>::evals(&reversed, None)
        };
        let cycles = weights.len();
        let access = self.rows.access();

        let block_size = 1usize << 12;
        let blocks = cycles.div_ceil(block_size);
        let block = |index: usize| -> Result<Vec<F>, WitnessError> {
            let start = index * block_size;
            let end = (start + block_size).min(cycles);
            let mut accumulator = ClaimAccumulator::<F>::default();
            for (t, &weight) in (start..end).zip(&weights[start..end]) {
                let row = access.row(t)?;
                accumulator.add_row(weight, &row);
            }
            Ok(accumulator.finish())
        };
        let merge = |mut left: Vec<F>, right: Vec<F>| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        };

        let claimed = {
            let _span = tracing::info_span!("SpartanOuter::claimed_input_walk").entered();
            #[cfg(feature = "parallel")]
            {
                (0..blocks).into_par_iter().map(block).try_reduce(
                    || vec![F::zero(); VARIABLE_COUNT],
                    |left, right| Ok(merge(left, right)),
                )
            }

            #[cfg(not(feature = "parallel"))]
            {
                let mut folded = vec![F::zero(); VARIABLE_COUNT];
                for index in 0..blocks {
                    folded = merge(folded, block(index)?);
                }
                Ok(folded)
            }
        };
        claimed
    }
}

const VARIABLE_COUNT: usize = 35;

/// Which canonical inputs are boolean-valued: those stay on the small-scalar
/// accumulator, whose 5-limb (320-bit) window only has headroom when the
/// scalar sum stays tiny (Σ ≤ block size for 0/1 scalars). Word-valued
/// columns would overflow it — a full-range `u64` scalar puts a single term
/// at ~2^318 — so they go through the signed-product path instead.
const BOOLEAN_INPUT: [bool; VARIABLE_COUNT] = {
    let mut mask = [false; VARIABLE_COUNT];
    mask[3] = true; // ShouldBranch
    mask[17] = true; // NextIsVirtual
    mask[18] = true; // NextIsFirstInSequence
    mask[20] = true; // ShouldJump
    let mut flag = 21; // the 14 circuit flags
    while flag < VARIABLE_COUNT {
        mask[flag] = true;
        flag += 1;
    }
    mask
};

/// Mixed-width claim accumulators for the final opening walk: boolean inputs
/// through the small-scalar path, word/wide inputs through the signed-product
/// path.
struct ClaimAccumulator<F: Field> {
    small: Vec<<F as WithSmallScalarAccumulator>::SmallScalarAccumulator>,
    wide: Vec<<F as WithSignedProductAccumulator>::SignedProductAccumulator>,
}

impl<F: Field> Default for ClaimAccumulator<F> {
    fn default() -> Self {
        Self {
            small: vec![Default::default(); VARIABLE_COUNT],
            wide: vec![Default::default(); VARIABLE_COUNT],
        }
    }
}

impl<F: Field> ClaimAccumulator<F> {
    fn add_row(&mut self, weight: F, row: &SpartanOuterRow) {
        let mut flag = |index: usize, value: bool| {
            self.small[index].fmadd_u64(weight, u64::from(value));
        };
        flag(3, row.should_branch.0);
        flag(17, row.next_is_virtual.0);
        flag(18, row.next_is_first_in_sequence.0);
        flag(20, row.should_jump.0);
        flag(21, row.add_operands.0);
        flag(22, row.subtract_operands.0);
        flag(23, row.multiply_operands.0);
        flag(24, row.load.0);
        flag(25, row.store.0);
        flag(26, row.jump.0);
        flag(27, row.write_lookup_output_to_rd.0);
        flag(28, row.virtual_instruction.0);
        flag(29, row.assert_flag.0);
        flag(30, row.do_not_update_unexpanded_pc.0);
        flag(31, row.advice.0);
        flag(32, row.is_compressed.0);
        flag(33, row.is_first_in_sequence.0);
        flag(34, row.is_last_in_sequence.0);

        let mut word = |index: usize, magnitude: u128, is_positive: bool| {
            if let Ok(magnitude) = u64::try_from(magnitude) {
                self.wide[index].fmadd_signed_u64(weight, magnitude, is_positive);
            } else {
                self.wide[index].fmadd_s256(
                    weight,
                    &S256::new(
                        [magnitude as u64, (magnitude >> 64) as u64, 0, 0],
                        is_positive,
                    ),
                );
            }
        };
        word(0, u128::from(row.left_instruction_input.0), true);
        word(4, u128::from(row.pc.0), true);
        word(5, u128::from(row.unexpanded_pc.0), true);
        word(7, u128::from(row.ram_address.0), true);
        word(8, u128::from(row.rs1_value.0), true);
        word(9, u128::from(row.rs2_value.0), true);
        word(10, u128::from(row.rd_write_value.0), true);
        word(11, u128::from(row.ram_read_value.0), true);
        word(12, u128::from(row.ram_write_value.0), true);
        word(13, u128::from(row.left_lookup_operand.0), true);
        word(15, u128::from(row.next_unexpanded_pc.0), true);
        word(16, u128::from(row.next_pc.0), true);
        word(19, u128::from(row.lookup_output.0), true);

        let product_limbs = row.product.0.magnitude_limbs();
        word(
            1,
            row.right_instruction_input.0.unsigned_abs(),
            row.right_instruction_input.0 >= 0,
        );
        word(
            2,
            (u128::from(product_limbs[1]) << 64) | u128::from(product_limbs[0]),
            row.product.0.is_positive,
        );
        word(6, row.imm.0.unsigned_abs(), row.imm.0 >= 0);
        word(14, row.right_lookup_operand.0, true);
    }

    fn finish(self) -> Vec<F> {
        self.small
            .into_iter()
            .zip(self.wide)
            .zip(BOOLEAN_INPUT)
            .map(|((small, wide), boolean)| {
                if boolean {
                    small.reduce()
                } else {
                    wide.reduce()
                }
            })
            .collect()
    }
}

impl<F: Field> ProveRounds<F> for OuterRemainderKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let (q_zero, q_infinity) = match self.pending_endpoints.take() {
            Some(endpoints) => endpoints,
            None => self.endpoints(),
        };
        Ok(self
            .split_eq
            .gruen_poly_deg_3(q_zero, q_infinity, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for OuterRemainderKernel<F> {
    type Relation = OuterRemainder<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        let remaining = self.rounds - self.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let claimed =
            self.claimed_inputs()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "outer opening walk re-extraction failed after the rounds",
                })?;
        let claims: BTreeMap<JoltOpeningId, F> =
            self.opening_ids.iter().copied().zip(claimed).collect();
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id| {
            claims.get(id).copied().or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let remaining = self.rounds - self.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        // The stream challenge binds the per-stream weight pairs; the split-eq
        // scalar is the fully bound TauKernel — both from the kernel's own
        // state, cross-checked against the verifier's coefficient build.
        let stream = self.challenges[0];
        let blend = |pair: [&F; 2]| *pair[0] + stream * (*pair[1] - *pair[0]);
        let variable_count = self.opening_ids.len();
        let ids = std::iter::once(SpartanOuterPublic::TauKernel)
            .chain((0..variable_count).map(SpartanOuterPublic::AzWeight))
            .chain((0..variable_count).map(SpartanOuterPublic::BzWeight))
            .chain([
                SpartanOuterPublic::AzConstant,
                SpartanOuterPublic::BzConstant,
            ]);
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let expected =
                match relation.derive_output_term(&id, input_points, output_points, challenges) {
                    Ok(value) => value,
                    Err(VerifierError::MissingStageClaimDerived { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let got = match public_id {
                SpartanOuterPublic::TauKernel => self.split_eq.current_scalar(),
                SpartanOuterPublic::AzWeight(index) => blend([
                    &self.derived.az_weights[0][index],
                    &self.derived.az_weights[1][index],
                ]),
                SpartanOuterPublic::BzWeight(index) => blend([
                    &self.derived.bz_weights[0][index],
                    &self.derived.bz_weights[1][index],
                ]),
                SpartanOuterPublic::AzConstant => {
                    blend([&self.derived.az_constant[0], &self.derived.az_constant[1]])
                }
                SpartanOuterPublic::BzConstant => {
                    blend([&self.derived.bz_constant[0], &self.derived.bz_constant[1]])
                }
            };
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OptimizedOuterEvalResult {
    pub(crate) round_polynomials: Vec<Vec<AkitaField>>,
    pub(crate) final_claim: AkitaField,
    pub(crate) output_claims: Vec<AkitaField>,
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
#[derive(Clone, Debug)]
pub(crate) struct OptimizedOuterEvalSample {
    pub(crate) result: OptimizedOuterEvalResult,
    pub(crate) member_wall: Duration,
    pub(crate) prepare_wall: Duration,
    pub(crate) rounds_wall: Duration,
    pub(crate) finish_wall: Duration,
    pub(crate) output_wall: Duration,
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
pub(crate) fn compute_optimized_outer_eval_input_claim(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    log_t: usize,
    tau: &[AkitaField],
    uniskip_challenge: AkitaField,
) -> Result<AkitaField, String> {
    if tau.len() != log_t + 2 {
        return Err("Outer evaluator tau geometry is invalid".to_owned());
    }
    let cycles = 1usize
        .checked_shl(u32::try_from(log_t).map_err(|error| error.to_string())?)
        .ok_or_else(|| "Outer evaluator trace domain overflows usize".to_owned())?;
    let rows = RowsStore::resolve(witness, cycles).map_err(|error| error.to_string())?;
    let tau_low = &tau[..=log_t];
    let lagrange = centered_lagrange_evals::<AkitaField>(DOMAIN, uniskip_challenge)
        .map_err(|error| error.to_string())?;
    let scaling = centered_lagrange_kernel(DOMAIN, tau[log_t + 1], uniskip_challenge)
        .map_err(|error| error.to_string())?;
    let split_eq =
        GruenSplitEqPolynomial::new_with_scaling(tau_low, BindingOrder::LowToHigh, Some(scaling));
    let e_out = split_eq.e_out_current();
    let e_in = split_eq.e_in_current();
    let access = rows.access();
    let block = |x_out: usize| -> Result<(AkitaField, AkitaField), WitnessError> {
        let mut q_zero = AkitaField::zero();
        let mut q_one = AkitaField::zero();
        for (x_in, &weight) in e_in.iter().enumerate() {
            let row = access.row(x_out * e_in.len() + x_in)?;
            let values = row_group_values(&row);
            let (az_zero, bz_zero) = fold_group(&lagrange, &values.a_first, &values.b_first);
            let (az_one, bz_one) = fold_group(
                &lagrange[..SECOND_GROUP_LEN],
                &values.a_second,
                &values.b_second,
            );
            q_zero += weight * az_zero * bz_zero;
            q_one += weight * az_one * bz_one;
        }
        Ok((e_out[x_out] * q_zero, e_out[x_out] * q_one))
    };
    let add = |left: (AkitaField, AkitaField), right: (AkitaField, AkitaField)| {
        (left.0 + right.0, left.1 + right.1)
    };
    #[cfg(feature = "parallel")]
    let (q_zero, q_one) = (0..e_out.len())
        .into_par_iter()
        .map(block)
        .try_reduce(
            || (AkitaField::zero(), AkitaField::zero()),
            |left, right| Ok(add(left, right)),
        )
        .map_err(|error| error.to_string())?;
    #[cfg(not(feature = "parallel"))]
    let (q_zero, q_one) = {
        let mut total = (AkitaField::zero(), AkitaField::zero());
        for x_out in 0..e_out.len() {
            total = add(total, block(x_out).map_err(|error| error.to_string())?);
        }
        total
    };
    let eq_one = split_eq.current_scalar() * tau_low[log_t];
    let eq_zero = split_eq.current_scalar() - eq_one;
    Ok(eq_zero * q_zero + eq_one * q_one)
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
pub(crate) fn run_optimized_outer_eval(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    log_t: usize,
    tau: &[AkitaField],
    uniskip_challenge: AkitaField,
    input_claim: AkitaField,
    challenges: &[AkitaField],
) -> Result<OptimizedOuterEvalSample, String> {
    let cycles = 1usize
        .checked_shl(u32::try_from(log_t).map_err(|error| error.to_string())?)
        .ok_or_else(|| "Outer evaluator trace domain overflows usize".to_owned())?;
    if tau.len() != log_t + 2 || challenges.len() != log_t + 1 {
        return Err("Outer evaluator challenge geometry is invalid".to_owned());
    }
    let rows = RowsStore::resolve(witness, cycles).map_err(|error| error.to_string())?;
    let relation = OuterRemainder::new(
        SpartanOuterDimensions::rv64(log_t),
        tau.to_vec(),
        uniskip_challenge,
    );
    let claims =
        jolt_verifier::stages::stage1::outer_remainder::outer_remainder_input_values_from_uniskip_output(
            input_claim,
        );
    let points = jolt_verifier::stages::stage1::outer_remainder::OuterRemainderInputClaims::<
        Vec<AkitaField>,
    >::default();
    let no_challenges = NoChallenges::<AkitaField>::default();
    let carry = SpartanOuterCarry {
        log_t,
        tau: tau.to_vec(),
        rows,
        t1_values: Vec::new(),
    };

    let member_started = Instant::now();
    let prepare_started = Instant::now();
    let mut kernel = OuterRemainderKernel::prepare(
        carry,
        &ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &no_challenges,
        },
    )
    .map_err(|error| error.to_string())?;
    let prepare_wall = prepare_started.elapsed();

    let rounds_started = Instant::now();
    let mut bind = None;
    let mut previous_claim = input_claim;
    let mut round_polynomials = Vec::with_capacity(challenges.len());
    for (round, &challenge) in challenges.iter().enumerate() {
        let polynomial = kernel
            .prove_round(bind, round, previous_claim)
            .map_err(|error| error.to_string())?;
        previous_claim = polynomial.evaluate(challenge);
        round_polynomials.push(polynomial.coefficients().to_vec());
        bind = Some(challenge);
    }
    let rounds_wall = rounds_started.elapsed();

    let finish_started = Instant::now();
    let final_challenge = challenges
        .last()
        .copied()
        .ok_or_else(|| "Outer evaluator has no terminal challenge".to_owned())?;
    kernel
        .finish_rounds(final_challenge)
        .map_err(|error| error.to_string())?;
    let finish_wall = finish_started.elapsed();

    let output_started = Instant::now();
    let output_points = relation
        .derive_opening_points(challenges, &points)
        .map_err(|error| error.to_string())?;
    let output_claims = kernel
        .output_claims(&claims)
        .map_err(|error| error.to_string())?;
    kernel
        .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
        .map_err(|error| error.to_string())?;
    let output_claims = output_claims.opening_values();
    let output_wall = output_started.elapsed();

    Ok(OptimizedOuterEvalSample {
        result: OptimizedOuterEvalResult {
            round_polynomials,
            final_claim: previous_claim,
            output_claims,
        },
        member_wall: member_started.elapsed(),
        prepare_wall,
        rounds_wall,
        finish_wall,
        output_wall,
    })
}

/// Byte parity against the reference kernels: identical uni-skip first-round
/// polynomials, identical remainder round polynomials at every round,
/// identical typed output claims — from identical `ProverInputs`, over
/// synthetic structured witnesses (both groups' wide integer paths exercised)
/// and over the real sample trace through the full trait path.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::jolt::JoltPolynomialId;
    use jolt_claims::NoChallenges;
    use jolt_field::signed::S128;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    use jolt_witness::{BundleSource, FixedBackend, JoltWitnessOracle, PolynomialEncoding, Shape};

    use super::*;
    use crate::optimized::instruction_input::PreparedInstructionInputRows;
    use crate::reference::spartan_outer::{ReferenceOuterRemainder, SpartanOuterKernel};
    use crate::ReferenceBackend;

    /// The `ToField` image of one canonical R1CS input, straight off the
    /// typed row — the single conversion source for backend columns and
    /// consistency checks.
    fn variable_field_value(row: &SpartanOuterRow, index: usize) -> Fr {
        match index {
            0 => row.left_instruction_input.to_field(),
            1 => row.right_instruction_input.to_field(),
            2 => row.product.to_field(),
            3 => row.should_branch.to_field(),
            4 => row.pc.to_field(),
            5 => row.unexpanded_pc.to_field(),
            6 => row.imm.to_field(),
            7 => row.ram_address.to_field(),
            8 => row.rs1_value.to_field(),
            9 => row.rs2_value.to_field(),
            10 => row.rd_write_value.to_field(),
            11 => row.ram_read_value.to_field(),
            12 => row.ram_write_value.to_field(),
            13 => row.left_lookup_operand.to_field(),
            14 => row.right_lookup_operand.to_field(),
            15 => row.next_unexpanded_pc.to_field(),
            16 => row.next_pc.to_field(),
            17 => row.next_is_virtual.to_field(),
            18 => row.next_is_first_in_sequence.to_field(),
            19 => row.lookup_output.to_field(),
            20 => row.should_jump.to_field(),
            21 => row.add_operands.to_field(),
            22 => row.subtract_operands.to_field(),
            23 => row.multiply_operands.to_field(),
            24 => row.load.to_field(),
            25 => row.store.to_field(),
            26 => row.jump.to_field(),
            27 => row.write_lookup_output_to_rd.to_field(),
            28 => row.virtual_instruction.to_field(),
            29 => row.assert_flag.to_field(),
            30 => row.do_not_update_unexpanded_pc.to_field(),
            31 => row.advice.to_field(),
            32 => row.is_compressed.to_field(),
            33 => row.is_first_in_sequence.to_field(),
            34 => row.is_last_in_sequence.to_field(),
            _ => unreachable!("35 canonical R1CS inputs"),
        }
    }

    /// Structured pseudo-random rows: full-range `u64`s, mixed-sign `i128`s,
    /// two-limb `u128`/`S128` values (both wide B-row paths), diverse flags.
    /// No satisfying-witness structure — parity must hold pointwise on any
    /// witness.
    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<SpartanOuterRow> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..1usize << log_t)
            .map(|_| {
                let mut bit = {
                    let value = next();
                    let mut position = 0;
                    move || {
                        position += 1;
                        (value >> position) & 1 == 1
                    }
                };
                let wide = |low: u64, high: u64| (u128::from(high) << 64) | u128::from(low);
                let signed = |value: u64| {
                    let magnitude = i128::from(value >> 1) << 33;
                    if value & 1 == 1 {
                        -magnitude
                    } else {
                        magnitude
                    }
                };
                SpartanOuterRow {
                    left_instruction_input: LeftInstructionInput(next()),
                    right_instruction_input: RightInstructionInput(signed(next())),
                    product: Product(S128::new([next() | 1, next()], next() & 1 == 1)),
                    should_branch: ShouldBranch(bit()),
                    pc: Pc(next() >> 20),
                    unexpanded_pc: UnexpandedPc(next()),
                    imm: Imm(signed(next()) >> 40),
                    ram_address: RamAddress(next()),
                    rs1_value: Rs1Value(next()),
                    rs2_value: Rs2Value(next()),
                    rd_write_value: RdWriteValue(next()),
                    ram_read_value: RamReadValue(next()),
                    ram_write_value: RamWriteValue(next()),
                    left_lookup_operand: LeftLookupOperand(next()),
                    right_lookup_operand: RightLookupOperand(wide(next(), next())),
                    next_unexpanded_pc: NextUnexpandedPc(next()),
                    next_pc: NextPc(next() >> 20),
                    next_is_virtual: NextIsVirtual(bit()),
                    next_is_first_in_sequence: NextIsFirstInSequence(bit()),
                    lookup_output: LookupOutput(next()),
                    should_jump: ShouldJump(bit()),
                    branch_flag: InstructionFlag(bit()),
                    is_noop: InstructionFlag(bit()),
                    next_is_noop: NextIsNoop(bit()),
                    add_operands: OpFlag(bit()),
                    subtract_operands: OpFlag(bit()),
                    multiply_operands: OpFlag(bit()),
                    load: OpFlag(bit()),
                    store: OpFlag(bit()),
                    jump: OpFlag(bit()),
                    write_lookup_output_to_rd: OpFlag(bit()),
                    virtual_instruction: OpFlag(bit()),
                    assert_flag: OpFlag(bit()),
                    do_not_update_unexpanded_pc: OpFlag(bit()),
                    advice: OpFlag(bit()),
                    is_compressed: OpFlag(bit()),
                    is_first_in_sequence: OpFlag(bit()),
                    is_last_in_sequence: OpFlag(bit()),
                    left_operand_is_rs1: InstructionFlag(bit()),
                    left_operand_is_pc: InstructionFlag(bit()),
                    right_operand_is_rs2: InstructionFlag(bit()),
                    right_operand_is_imm: InstructionFlag(bit()),
                }
            })
            .collect()
    }

    fn fixed_backend_from_rows(log_t: usize, rows: &[SpartanOuterRow]) -> FixedBackend<Fr> {
        let mut backend = FixedBackend::new();
        for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
            let values: Vec<Fr> = rows
                .iter()
                .map(|row| variable_field_value(row, index))
                .collect();
            backend
                .insert(
                    JoltPolynomialId::Virtual(*variable),
                    Shape::new(log_t, PolynomialEncoding::Dense),
                    values,
                )
                .unwrap();
        }
        backend
    }

    /// The remainder's true input claim
    /// `Σ_{t,s} kernel · eq(τ_low, (t,s)) · Az(t,s) · Bz(t,s)`, computed
    /// through the public `jolt-r1cs` column-weight path (independent of both
    /// kernels' row-value pipelines).
    fn true_input_claim(rows: &[SpartanOuterRow], tau: &[Fr], r0: Fr, log_t: usize) -> Fr {
        let tau_low = &tau[..=log_t];
        let tau_high = tau[log_t + 1];
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let kernel = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, r0).unwrap();
        let matrices = spartan_outer_constraints::<Fr>();
        let columns: Vec<usize> = (1..=VARIABLE_COUNT).collect();
        let mut total = Fr::from_u64(0);
        for (s, stream) in [Fr::from_u64(0), Fr::from_u64(1)].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(r0, stream).unwrap();
            let weighted = matrices.weighted_columns(&weights, &columns).unwrap();
            let constants = matrices
                .public_column_contributions(&weights, 0, Fr::from_u64(1))
                .unwrap();
            for (t, row) in rows.iter().enumerate() {
                let mut az = constants.a;
                let mut bz = constants.b;
                for (index, (&a, &b)) in weighted.a.iter().zip(&weighted.b).enumerate() {
                    let value = variable_field_value(row, index);
                    az += a * value;
                    bz += b * value;
                }
                total += eq[(t << 1) | s] * az * bz;
            }
        }
        kernel * total
    }

    /// One full parity case: uni-skip polynomial, every remainder round
    /// polynomial, typed output claims, and both kernels' derived-table
    /// validation — reference and optimized fed identical `ProverInputs`.
    fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
        let rows = synthetic_rows(log_t, seed);
        let tau: Vec<Fr> = (0..log_t + 2)
            .map(|i| Fr::from_u64(3 + seed + 7 * i as u64))
            .collect();
        let backend = fixed_backend_from_rows(log_t, &rows);

        let mut reference_session = ProofSession::default();
        reference_session.park(SpartanOuterKernel::<Fr>::prepare(log_t, &tau, &backend).unwrap());
        let reference_uniskip =
            <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &ReferenceBackend,
                &mut reference_session,
                &[],
                &[],
            )
            .unwrap();

        let mut optimized_session = ProofSession::default();
        OptimizedOuterUniskip::prepare_from_rows(&mut optimized_session, log_t, &tau, rows.clone())
            .unwrap();
        let optimized_uniskip =
            <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &OptimizedOuterUniskip,
                &mut optimized_session,
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(
            optimized_uniskip, reference_uniskip,
            "uni-skip first-round polynomial, log_t = {log_t}"
        );

        let r0 = Fr::from_u64(40961 + seed);
        let input_claim = true_input_claim(&rows, &tau, r0, log_t);
        let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
        let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
        let points = OuterRemainderInputClaims::<Vec<Fr>>::default();
        let no_challenges = NoChallenges::<Fr>::default();

        let mut reference_kernel = ReferenceOuterRemainder
            .prepare(
                &mut reference_session,
                dummy_plane,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();
        let mut optimized_kernel = OptimizedOuterRemainder
            .prepare(
                &mut optimized_session,
                dummy_plane,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();

        let rounds = log_t + 1;
        let challenges: Vec<Fr> = (0..rounds)
            .map(|i| Fr::from_u64(7919 + seed + 31 * i as u64))
            .collect();
        let mut bind = None;
        let mut previous = input_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
            let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
            assert_eq!(
                optimized_round, reference_round,
                "round {round} polynomial, log_t = {log_t}"
            );
            previous = reference_round.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference_kernel.finish_rounds(last).unwrap();
        optimized_kernel.finish_rounds(last).unwrap();

        let reference_outputs = reference_kernel.output_claims(&claims).unwrap();
        let optimized_outputs = optimized_kernel.output_claims(&claims).unwrap();
        assert_eq!(
            optimized_outputs, reference_outputs,
            "typed output claims, log_t = {log_t}"
        );

        let output_points = relation
            .derive_opening_points(&challenges, &points)
            .unwrap();
        reference_kernel
            .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
            .unwrap();
        optimized_kernel
            .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
            .unwrap();
    }

    /// Synthetic parity across sizes spanning the uni-skip boundary and
    /// degenerate small domains. The sample backend only supplies the (never
    /// read) witness-plane argument of the remainder `prepare` calls.
    #[test]
    fn synthetic_parity_with_reference_kernels() {
        with_sample_backend(|dummy| {
            for (log_t, seed) in [(1usize, 111u64), (2, 222), (3, 333), (4, 444)] {
                parity_case(dummy, log_t, seed);
            }
        });
    }

    #[test]
    fn prepare_witness_parks_instruction_input_rows() {
        with_sample_backend(|backend| {
            let log_t = 2;
            let mut session = ProofSession::default();
            <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare_witness(
                &OptimizedOuterUniskip,
                &mut session,
                log_t,
                backend,
            )
            .unwrap();
            assert_eq!(
                session
                    .state::<PreparedInstructionInputRows>()
                    .unwrap()
                    .len(),
                1 << log_t
            );
        });
    }

    /// Full trait-path parity on the real sample trace: the optimized bundle
    /// walk against the reference's oracle tables, with the genuine uni-skip
    /// output claim feeding the remainder (a satisfying witness, so the
    /// uni-skip reduction and the joint-domain sum agree).
    #[test]
    fn sample_trace_parity_through_the_trait_path() {
        with_sample_backend(|backend| {
            let log_t = 2usize;
            let tau: Vec<Fr> = (0..log_t + 2)
                .map(|i| Fr::from_u64(29 + 13 * i as u64))
                .collect();

            let mut reference_session = ProofSession::default();
            <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                log_t,
                &tau,
                backend,
            )
            .unwrap();
            let reference_uniskip =
                <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                    &ReferenceBackend,
                    &mut reference_session,
                    &[],
                    &[],
                )
                .unwrap();

            let mut optimized_session = ProofSession::default();
            <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare(
                &OptimizedOuterUniskip,
                &mut optimized_session,
                log_t,
                &tau,
                backend,
            )
            .unwrap();
            let optimized_uniskip = <OptimizedOuterUniskip as UniskipKernel<
                Fr,
                OuterRemainder<Fr>,
            >>::first_round_poly(
                &OptimizedOuterUniskip, &mut optimized_session, &[], &[]
            )
            .unwrap();
            assert_eq!(optimized_uniskip, reference_uniskip);

            // The sample fixture is a witness-extraction fixture, not a
            // constraint-satisfying trace (its second row RAM-writes without
            // the Store flag), so the uni-skip reduction at r0 need not equal
            // the joint-domain sum here; the remainder is driven by the true
            // sum, which is what the naive reference self-checks against.
            let r0 = Fr::from_u64(9173);
            let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
            let input_claim = true_input_claim(&rows, &tau, r0, log_t);

            let relation =
                OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
            let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
            let points = OuterRemainderInputClaims::<Vec<Fr>>::default();
            let no_challenges = NoChallenges::<Fr>::default();
            let mut reference_kernel = ReferenceOuterRemainder
                .prepare(
                    &mut reference_session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &no_challenges,
                    },
                )
                .unwrap();
            let mut optimized_kernel = OptimizedOuterRemainder
                .prepare(
                    &mut optimized_session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &no_challenges,
                    },
                )
                .unwrap();

            let challenges: Vec<Fr> = (0..=log_t)
                .map(|i| Fr::from_u64(523 + 17 * i as u64))
                .collect();
            let mut bind = None;
            let mut previous = input_claim;
            for (round, &challenge) in challenges.iter().enumerate() {
                let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
                let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
                assert_eq!(optimized_round, reference_round, "round {round}");
                previous = reference_round.evaluate(challenge);
                bind = Some(challenge);
            }
            let last = bind.unwrap();
            reference_kernel.finish_rounds(last).unwrap();
            optimized_kernel.finish_rounds(last).unwrap();
            assert_eq!(
                optimized_kernel.output_claims(&claims).unwrap(),
                reference_kernel.output_claims(&claims).unwrap()
            );
        });
    }

    /// The integer extension coefficients are exactly the field Lagrange
    /// basis evaluations at the extended nodes — the fact that ties the
    /// integer pipeline to the reference's field pipeline.
    #[test]
    fn extension_coefficients_match_field_lagrange() {
        for (position, coefficients) in extension_coefficients() {
            let node = EXTENDED_START + position as i64;
            let expected = centered_lagrange_evals::<Fr>(DOMAIN, Fr::from_i64(node)).unwrap();
            for (i, &coefficient) in coefficients.iter().enumerate() {
                assert_eq!(
                    Fr::from_i64(coefficient),
                    expected[i],
                    "node {node}, basis {i}"
                );
            }
        }
    }

    /// The typed bundle's columns equal the oracle tables the reference
    /// kernel materializes — the two witness paths meeting at the shared
    /// `Extract` impls, for all 35 R1CS inputs.
    #[test]
    fn bundle_columns_match_oracle_tables() {
        with_sample_backend(|backend| {
            let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
            for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
                let table: Vec<Fr> = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(*variable),
                )
                .unwrap();
                let column: Vec<Fr> = rows
                    .iter()
                    .map(|row| variable_field_value(row, index))
                    .collect();
                assert_eq!(column, table, "{variable:?}");
            }
        });
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn instruction_source_lookup_and_increment_are_reconstructible() {
        use jolt_lookup_tables::interleave_bits;

        with_sample_backend(|backend| {
            let rows: Vec<Stage1ProjectionRow> = backend.bundles().unwrap();
            let independently_extracted: Vec<Stage1InstructionFacts> = backend.bundles().unwrap();
            for (cycle, (row, expected)) in
                rows.into_iter().zip(independently_extracted).enumerate()
            {
                assert_eq!(row.instruction.lookup_index.0, expected.lookup_index.0);
                assert_eq!(row.instruction.table_index.0, expected.table_index.0);
                assert_eq!(row.instruction.raf_flag.0, expected.raf_flag.0);
                assert_eq!(row.instruction.mapped_pc.0, expected.mapped_pc.0);
                assert_eq!(
                    row.instruction.remapped_ram_address.0,
                    expected.remapped_ram_address.0
                );
                assert_eq!(row.instruction.fused_inc.0, expected.fused_inc.0);
                let right = row.outer.right_lookup_operand.0;
                let reconstructed_lookup = if row.instruction.raf_flag.0 {
                    right
                } else {
                    assert_eq!(right >> 64, 0, "cycle {cycle}");
                    interleave_bits(row.outer.left_lookup_operand.0, right as u64)
                };
                assert_eq!(
                    reconstructed_lookup, row.instruction.lookup_index.0,
                    "lookup index at cycle {cycle}"
                );

                let reconstructed_increment = if row.outer.store.0 {
                    assert!(row.register_write.is_none(), "store cycle {cycle}");
                    i128::from(row.outer.ram_write_value.0) - i128::from(row.outer.ram_read_value.0)
                } else {
                    row.register_write
                        .map_or(0, |(_, pre, post)| i128::from(post) - i128::from(pre))
                };
                assert_eq!(
                    reconstructed_increment, row.instruction.fused_inc.0,
                    "fused increment at cycle {cycle}"
                );
            }
        });
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn stage1_projection_register_source_matches_register_cycle_rows() {
        use crate::optimized::registers_read_write::RegisterCycleRow;

        with_sample_backend(|backend| {
            let projected: Vec<Stage1ProjectionRow> = backend.bundles().unwrap();
            let expected: Vec<RegisterCycleRow> = backend.bundles().unwrap();
            assert_eq!(projected.len(), expected.len());
            for (cycle, (projected, expected)) in projected.into_iter().zip(expected).enumerate() {
                assert_eq!(
                    projected.register_indices[0],
                    expected.rs1.map(|(index, _)| index),
                    "rs1 index at cycle {cycle}"
                );
                assert_eq!(
                    projected.register_indices[1],
                    expected.rs2.map(|(index, _)| index),
                    "rs2 index at cycle {cycle}"
                );
                assert_eq!(
                    projected.register_write, expected.rd,
                    "rd row at cycle {cycle}"
                );
            }
        });
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_co_produced_projection_matches_independent_producers() {
        use jolt_claims::protocols::jolt::JoltOneHotConfig;
        use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
        use jolt_program::preprocess::{
            BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
        };
        use jolt_riscv::{
            JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT,
        };
        use jolt_witness::{
            JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend,
        };

        use crate::metal::solinas::spartan_shift::{
            SpartanShiftGeometry, SpartanShiftKernelConfig,
        };
        use crate::metal::solinas::SolinasMetal;
        use crate::optimized::spartan_shift::prepare_metal_spartan_shift_witness_rows;

        fn instruction(
            address: usize,
            virtual_sequence_remaining: Option<u16>,
            first: bool,
        ) -> JoltInstructionRow {
            JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADDI,
                address,
                operands: NormalizedOperands {
                    rd: Some(1),
                    rs1: Some(2),
                    rs2: None,
                    imm: 3,
                },
                virtual_sequence_remaining,
                is_first_in_sequence: first,
                is_compressed: false,
            }
        }

        let log_t = 4usize;
        let cycles = 1usize << log_t;
        let plain_a = instruction(0x8000_0000, None, false);
        let virtual_first = instruction(0x8000_0004, Some(1), true);
        let virtual_last = instruction(0x8000_0004, Some(0), false);
        let plain_b = instruction(0x8000_0008, None, false);
        let noop = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::NoOp,
            ..plain_a
        };
        let bytecode = vec![plain_a, virtual_first, virtual_last, plain_b];
        let rows: Vec<TraceRow> = [plain_a, virtual_first, virtual_last, noop, plain_b, plain_a]
            .into_iter()
            .map(|instruction| TraceRow {
                instruction,
                ..TraceRow::default()
            })
            .collect();
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                bytecode,
                plain_a.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: cycles,
        };
        let program = JoltProgram::default();
        let config = JoltVmWitnessConfig::new(
            log_t,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs);
        let witness = &backend as &dyn JoltWitnessPlane<jolt_field::AkitaField>;
        let projected: Vec<SpartanOuterRow> = backend.bundles().unwrap();
        assert!(projected.iter().any(|row| row.virtual_instruction.0));
        assert!(projected.iter().any(|row| row.is_first_in_sequence.0));
        assert!(projected.iter().any(|row| row.is_noop.0));

        let context = SolinasMetal::for_akita().unwrap();
        let independent_outer =
            prepare_metal_spartan_outer_witness_rows(&context, witness, cycles).unwrap();
        let independent_shift =
            prepare_metal_spartan_shift_witness_rows(&context, witness, cycles).unwrap();
        let (combined_outer, combined_shift) =
            prepare_metal_spartan_outer_shift_witness_rows(&context, witness, cycles).unwrap();

        let e_out = EqPolynomial::<jolt_field::AkitaField>::evals(
            &[
                jolt_field::AkitaField::from_u64(3),
                jolt_field::AkitaField::from_u64(5),
            ],
            None,
        );
        let e_in = EqPolynomial::<jolt_field::AkitaField>::evals(
            &[
                jolt_field::AkitaField::from_u64(7),
                jolt_field::AkitaField::from_u64(11),
                jolt_field::AkitaField::from_u64(13),
            ],
            None,
        );
        let outer_config = SpartanOuterUniskipConfig {
            threads_per_threadgroup: Some(32),
        };
        let independent_outer_invocation = context
            .prepare_spartan_outer_uniskip_with_rows(
                &independent_outer,
                &e_in,
                &e_out,
                outer_config,
            )
            .unwrap();
        independent_outer_invocation.execute().unwrap();
        let independent_outer_output = independent_outer_invocation.read_output().unwrap();
        let combined_outer_invocation = context
            .prepare_spartan_outer_uniskip_with_rows(&combined_outer, &e_in, &e_out, outer_config)
            .unwrap();
        combined_outer_invocation.execute().unwrap();
        assert_eq!(
            combined_outer_invocation.read_output().unwrap(),
            independent_outer_output
        );

        let geometry = SpartanShiftGeometry::new(cycles).unwrap();
        let point = |seed: u64| {
            (0..log_t)
                .scan(seed, |state, _| {
                    *state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    Some(jolt_field::AkitaField::from_u64(*state | 1))
                })
                .collect::<Vec<_>>()
        };
        let r_outer = point(0xA11C_E001);
        let r_product = point(0xB22D_F002);
        let gamma = jolt_field::AkitaField::from_u64(0xC33E_1003);
        let shift_config = SpartanShiftKernelConfig {
            build_threads_per_threadgroup: 32,
            high_tile_elements: geometry.suffix_elements(),
            fold_threads_per_threadgroup: 32,
        };
        let independent_shift_output = context
            .prepare_spartan_shift_prefix(
                &independent_shift,
                &r_outer,
                &r_product,
                gamma,
                shift_config,
            )
            .unwrap()
            .execute()
            .unwrap();
        let combined_shift_output = context
            .prepare_spartan_shift_prefix(
                &combined_shift,
                &r_outer,
                &r_product,
                gamma,
                shift_config,
            )
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(combined_shift_output.q, independent_shift_output.q);
    }
}
