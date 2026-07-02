use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{CircuitFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use super::super::claim_reductions::bytecode as bytecode_reduction;
use super::super::dimensions::{
    JoltFormulaPointError, TracePolynomialOrder, REGISTER_ADDRESS_BITS,
};
use super::families::JoltPackingFamilyId;
use super::openings::unsigned_inc_chunking;
use super::types::{PackingFamilyId, PackingViewFormula, PackingViewTerm};

pub fn bytecode_store_flag_lattice_view_formula<F: Field>(chunk: usize) -> PackingViewFormula<F> {
    PackingViewFormula::direct(
        JoltPackingFamilyId::BytecodeCircuitFlag {
            chunk,
            flag: CircuitFlags::Store as usize,
        }
        .into(),
        0,
        1,
    )
}

pub fn bytecode_rd_present_lattice_view_formula<F: Field>(chunk: usize) -> PackingViewFormula<F> {
    PackingViewFormula::linear_decoded(weighted_symbol_terms(
        JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 }.into(),
        0,
        [F::one(); 1 << REGISTER_ADDRESS_BITS],
    ))
}

pub fn unsigned_inc_lower_value_lattice_view_formula<F: Field>(
    log_k_chunk: usize,
) -> Option<PackingViewFormula<F>> {
    Some(PackingViewFormula::linear_decoded(
        unsigned_inc_lower_value_terms(log_k_chunk)?,
    ))
}

pub fn unsigned_inc_lower_value_terms<F: Field>(
    log_k_chunk: usize,
) -> Option<Vec<PackingViewTerm<F>>> {
    let chunking = unsigned_inc_chunking(log_k_chunk)?;
    let mut terms = Vec::with_capacity(chunking.chunk_count * chunking.alphabet_size);
    let mut place = F::one();
    for index in 0..chunking.chunk_count {
        terms.extend(weighted_symbol_terms(
            JoltPackingFamilyId::UnsignedIncChunk { index }.into(),
            0,
            (0..chunking.alphabet_size).map(|symbol| place * F::from_u64(symbol as u64)),
        ));
        place *= F::from_u64(chunking.radix);
    }
    Some(terms)
}

pub fn unsigned_inc_msb_lattice_view_formula<F: Field>() -> PackingViewFormula<F> {
    PackingViewFormula::direct(JoltPackingFamilyId::UnsignedIncMsb.into(), 0, 1)
}

pub fn byte_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb: usize,
) -> Vec<PackingViewTerm<F>> {
    weighted_byte_decode_terms(family, [(limb, F::one())])
}

pub fn symbol_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb: usize,
    alphabet_size: usize,
) -> Vec<PackingViewTerm<F>> {
    weighted_symbol_terms(
        family,
        limb,
        (0..alphabet_size).map(|symbol| F::from_u64(symbol as u64)),
    )
}

pub fn weighted_symbol_terms<F>(
    family: PackingFamilyId,
    limb: usize,
    weights: impl IntoIterator<Item = F>,
) -> Vec<PackingViewTerm<F>> {
    weights
        .into_iter()
        .enumerate()
        .map(|(symbol, coefficient)| PackingViewTerm::new(coefficient, family, limb, symbol))
        .collect()
}

pub fn weighted_byte_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb_weights: impl IntoIterator<Item = (usize, F)>,
) -> Vec<PackingViewTerm<F>> {
    limb_weights
        .into_iter()
        .flat_map(|(limb, limb_weight)| {
            (0..256).map(move |symbol| {
                PackingViewTerm::new(
                    limb_weight * F::from_u64(symbol as u64),
                    family,
                    limb,
                    symbol,
                )
            })
        })
        .collect()
}

pub fn little_endian_byte_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb_count: usize,
) -> Vec<PackingViewTerm<F>> {
    let mut limb_weights = Vec::with_capacity(limb_count);
    let mut place = F::one();
    for limb in 0..limb_count {
        limb_weights.push((limb, place));
        place *= F::from_u64(256);
    }
    weighted_byte_decode_terms(family, limb_weights)
}

pub fn bytecode_chunk_lattice_view_formula<F: Field>(
    chunk: usize,
    opening_point: &[F],
    trace_order: TracePolynomialOrder,
    log_bytecode: usize,
    field_byte_width: usize,
) -> Result<PackingViewFormula<F>, JoltFormulaPointError> {
    let lane_vars = bytecode_reduction::committed_lane_vars();
    let expected = lane_vars + log_bytecode;
    if opening_point.len() != expected {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected,
            got: opening_point.len(),
        });
    }
    let lane_point = match trace_order {
        TracePolynomialOrder::CycleMajor => &opening_point[..lane_vars],
        TracePolynomialOrder::AddressMajor => &opening_point[log_bytecode..],
    };
    let lane_weights = EqPolynomial::<F>::evals(lane_point, None);
    let lane_layout = bytecode_reduction::BYTECODE_LANE_LAYOUT;
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let mut terms = Vec::new();

    for selector in 0..3 {
        let start = match selector {
            0 => lane_layout.rs1_start,
            1 => lane_layout.rs2_start,
            _ => lane_layout.rd_start,
        };
        terms.extend(weighted_symbol_terms(
            JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector }.into(),
            0,
            lane_weights[start..start + register_count].iter().copied(),
        ));
    }
    terms.extend(weighted_byte_decode_terms(
        JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk }.into(),
        byte_limb_weights(lane_weights[lane_layout.unexp_pc_idx], 8),
    ));
    terms.extend(weighted_byte_decode_terms(
        JoltPackingFamilyId::BytecodeImmBytes { chunk }.into(),
        byte_limb_weights(lane_weights[lane_layout.imm_idx], field_byte_width),
    ));
    for flag in 0..NUM_CIRCUIT_FLAGS {
        terms.push(PackingViewTerm::new(
            lane_weights[lane_layout.circuit_start + flag],
            JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag }.into(),
            0,
            1,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        terms.push(PackingViewTerm::new(
            lane_weights[lane_layout.instr_start + flag],
            JoltPackingFamilyId::BytecodeInstructionFlag { chunk, flag }.into(),
            0,
            1,
        ));
    }
    terms.extend(weighted_symbol_terms(
        JoltPackingFamilyId::BytecodeLookupSelector { chunk }.into(),
        0,
        lane_weights
            [lane_layout.lookup_start..lane_layout.lookup_start + LookupTableKind::<XLEN>::COUNT]
            .iter()
            .copied(),
    ));
    terms.push(PackingViewTerm::new(
        lane_weights[lane_layout.raf_flag_idx],
        JoltPackingFamilyId::BytecodeRafFlag { chunk }.into(),
        0,
        1,
    ));

    Ok(PackingViewFormula::linear_decoded(terms))
}

fn byte_limb_weights<F: Field>(lane_weight: F, limb_count: usize) -> Vec<(usize, F)> {
    let mut weights = Vec::with_capacity(limb_count);
    let mut place = F::one();
    for limb in 0..limb_count {
        weights.push((limb, lane_weight * place));
        place *= F::from_u64(256);
    }
    weights
}
