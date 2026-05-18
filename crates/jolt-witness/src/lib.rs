//! Primitive oracle construction kernels for Bolt-generated Jolt code.
//!
//! This crate is intentionally not a runtime/provider abstraction. Generated
//! code calls these kernels after the Bolt lowering pipeline has made oracle
//! generation explicit in IR.

use jolt_field::Field;
use jolt_poly::EqPolynomial;

pub mod field_reg;

pub const NUM_DENSE_TRACE_COLUMNS: usize = 3;
pub const NUM_ONE_HOT_TRACE_SOURCES: usize = 4;

/// Dense column slots inside [`CycleInput::dense`].
pub const DENSE_RD_INC: usize = 0;
pub const DENSE_RAM_INC: usize = 1;
pub const DENSE_FIELD_REG_INC: usize = 2;

/// One-hot source slots inside [`CycleInput::one_hot`].
pub const ONE_HOT_INSTRUCTION_KEYS: usize = 0;
pub const ONE_HOT_BYTECODE_INDICES: usize = 1;
pub const ONE_HOT_RAM_ADDRESSES: usize = 2;
pub const ONE_HOT_FIELD_REG_INDICES: usize = 3;

/// Per-cycle primitive inputs consumed by Bolt oracle generation.
#[derive(Clone, Copy, Debug)]
pub struct CycleInput {
    pub dense: [i128; NUM_DENSE_TRACE_COLUMNS],
    pub one_hot: [Option<u128>; NUM_ONE_HOT_TRACE_SOURCES],
}

impl CycleInput {
    pub const PADDING: Self = Self {
        dense: [0; NUM_DENSE_TRACE_COLUMNS],
        // FieldReg access defaults to register 0 on non-FR cycles (matches
        // the all-zero initial FR register file; FieldReg flags are also
        // zero, so the FR R1CS rows are trivially satisfied).
        one_hot: [Some(0), Some(0), None, Some(0)],
    };
}

impl Default for CycleInput {
    fn default() -> Self {
        Self::PADDING
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommitmentTraceSources {
    pub rd_inc: Vec<i128>,
    pub ram_inc: Vec<i128>,
    pub field_reg_inc: Vec<i128>,
    pub instruction_keys: Vec<Option<u128>>,
    pub ram_addresses: Vec<Option<u128>>,
    pub bytecode_indices: Vec<Option<u128>>,
    pub field_reg_indices: Vec<Option<u128>>,
}

impl CommitmentTraceSources {
    pub fn from_cycle_inputs(cycle_inputs: &[CycleInput]) -> Self {
        Self {
            rd_inc: cycle_inputs.iter().map(|c| c.dense[DENSE_RD_INC]).collect(),
            ram_inc: cycle_inputs.iter().map(|c| c.dense[DENSE_RAM_INC]).collect(),
            field_reg_inc: cycle_inputs
                .iter()
                .map(|c| c.dense[DENSE_FIELD_REG_INC])
                .collect(),
            instruction_keys: one_hot_cycle_column(cycle_inputs, ONE_HOT_INSTRUCTION_KEYS),
            ram_addresses: one_hot_cycle_column(cycle_inputs, ONE_HOT_RAM_ADDRESSES),
            bytecode_indices: one_hot_cycle_column(cycle_inputs, ONE_HOT_BYTECODE_INDICES),
            field_reg_indices: one_hot_cycle_column(cycle_inputs, ONE_HOT_FIELD_REG_INDICES),
        }
    }
}

pub fn commitment_trace_sources(cycle_inputs: &[CycleInput]) -> CommitmentTraceSources {
    CommitmentTraceSources::from_cycle_inputs(cycle_inputs)
}

/// Returns a dense trace source by its generated oracle source name.
pub fn dense_cycle_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<i128> {
    let slot = match source {
        "trace.rd_inc" => DENSE_RD_INC,
        "trace.ram_inc" => DENSE_RAM_INC,
        "trace.field_reg_inc" => DENSE_FIELD_REG_INC,
        _ => unreachable!("unsupported dense source `{source}`"),
    };
    cycle_inputs.iter().map(|cycle| cycle.dense[slot]).collect()
}

/// Returns a one-hot trace source by its generated oracle source name.
pub fn one_hot_cycle_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<Option<u128>> {
    let slot = match source {
        "trace.instruction_keys" => ONE_HOT_INSTRUCTION_KEYS,
        "trace.bytecode_indices" => ONE_HOT_BYTECODE_INDICES,
        "trace.ram_addresses" => ONE_HOT_RAM_ADDRESSES,
        "trace.field_reg_indices" => ONE_HOT_FIELD_REG_INDICES,
        _ => unreachable!("unsupported one-hot source `{source}`"),
    };
    one_hot_cycle_column(cycle_inputs, slot)
}

/// Maps generated one-hot padding policy names to the corresponding value.
pub fn one_hot_padding_value(padding: &str) -> Option<u128> {
    match padding {
        "zero" => Some(0),
        "none" => None,
        _ => unreachable!("unsupported padding `{padding}`"),
    }
}

/// Converts an i128 trace column to field elements and pads it to `target_len`.
///
/// The input is normally trace-length data; commitment domains can be larger
/// than the trace domain, so generated code asks for the final committed length.
pub fn dense_i128_column_to_field<F: Field>(values: &[i128], target_len: usize) -> Vec<F> {
    assert!(
        values.len() <= target_len,
        "dense trace column has {} values, target length is {target_len}",
        values.len()
    );
    let mut output: Vec<F> = values.iter().map(|&value| F::from_i128(value)).collect();
    output.resize(target_len, F::zero());
    output
}

/// Pads an optional field-valued oracle to `target_len`.
///
/// `None` stays `None`; zero-skipping policy is deliberately left to the
/// generated commitment code because skip semantics are protocol metadata.
pub fn optional_field_oracle<F: Field>(values: Option<&[F]>, target_len: usize) -> Option<Vec<F>> {
    values.map(|values| pad_field_oracle(values, target_len))
}

/// Pads a field-valued oracle to `target_len`.
pub fn pad_field_oracle<F: Field>(values: &[F], target_len: usize) -> Vec<F> {
    assert!(
        values.len() <= target_len,
        "field oracle has {} values, target length is {target_len}",
        values.len()
    );
    let mut output = values.to_vec();
    output.resize(target_len, F::zero());
    output
}

/// Deterministic placeholder data for optional advice oracles in synthetic tests.
pub fn deterministic_oracle_data<F: Field>(oracle: &str, num_vars: usize) -> Vec<F> {
    let seed = oracle.bytes().fold(17u64, |state, byte| {
        state.wrapping_mul(131).wrapping_add(byte as u64)
    });
    (0..(1usize << num_vars))
        .map(|index| F::from_u64(seed.wrapping_add(index as u64 + 1)))
        .collect()
}

/// Returns synthetic data for non-advice oracles and `None` for optional advice.
pub fn optional_oracle_data<F: Field>(oracle: &str, num_vars: usize) -> Option<Vec<F>> {
    match oracle {
        "UntrustedAdvice" | "TrustedAdvice" => None,
        _ => Some(deterministic_oracle_data(oracle, num_vars)),
    }
}

/// Builds sparse per-cycle one-hot chunk indices.
///
/// The returned vector has one entry per trace cycle. `None` means no one-hot
/// entry is active for that cycle. Chunk `0` is the most significant chunk,
/// matching jolt-core's committed RA decomposition.
pub fn one_hot_chunk_indices(
    values: &[Option<u128>],
    chunk: usize,
    num_chunks: usize,
    chunk_bits: usize,
    trace_len: usize,
    padding_value: Option<u128>,
) -> Vec<Option<u8>> {
    assert!(
        values.len() <= trace_len,
        "one-hot source has {} values, trace length is {trace_len}",
        values.len()
    );
    assert!(
        chunk < num_chunks,
        "chunk index {chunk} out of bounds for {num_chunks} chunks"
    );
    assert!(
        chunk_bits <= u8::BITS as usize,
        "chunk_bits must fit in one byte"
    );
    assert!(
        chunk_bits * num_chunks <= u128::BITS as usize,
        "one-hot chunks must fit in u128 source values"
    );

    let chunk_domain = 1usize << chunk_bits;
    let shift = chunk_bits * (num_chunks - 1 - chunk);
    let mask = (chunk_domain - 1) as u128;
    let mut output = Vec::with_capacity(trace_len);

    for cycle in 0..trace_len {
        let value = values.get(cycle).copied().flatten().or(padding_value);
        output.push(value.map(|value| ((value >> shift) & mask) as u8));
    }

    output
}

/// Builds one address-major one-hot chunk polynomial.
///
/// Layout is `output[chunk_value * trace_len + cycle]`. Chunk `0` is the most
/// significant chunk, matching jolt-core's committed RA decomposition.
pub fn one_hot_chunk_address_major<F: Field>(
    values: &[Option<u128>],
    chunk: usize,
    num_chunks: usize,
    chunk_bits: usize,
    trace_len: usize,
    padding_value: Option<u128>,
) -> Vec<F> {
    let indices = one_hot_chunk_indices(
        values,
        chunk,
        num_chunks,
        chunk_bits,
        trace_len,
        padding_value,
    );
    one_hot_address_major_from_indices(&indices, chunk_bits)
}

/// Builds one address-major one-hot chunk polynomial from sparse per-cycle indices.
///
/// Layout is `output[chunk_value * trace_len + cycle]`.
pub fn one_hot_address_major_from_indices<F: Field>(
    indices: &[Option<u8>],
    chunk_bits: usize,
) -> Vec<F> {
    assert!(
        chunk_bits < usize::BITS as usize,
        "chunk_bits must fit in usize"
    );

    let chunk_domain = 1usize << chunk_bits;
    let mut output = vec![F::zero(); chunk_domain * indices.len()];

    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            assert!(
                index < chunk_domain,
                "one-hot index {index} exceeds chunk domain {chunk_domain}"
            );
            output[index * indices.len() + cycle] = F::one();
        }
    }

    output
}

/// Builds one cycle-major one-hot chunk polynomial from sparse per-cycle indices.
///
/// Layout is `output[cycle * chunk_domain + chunk_value]`.
pub fn one_hot_cycle_major_from_indices<F: Field>(
    indices: &[Option<u8>],
    chunk_bits: usize,
) -> Vec<F> {
    assert!(
        chunk_bits < usize::BITS as usize,
        "chunk_bits must fit in usize"
    );

    let chunk_domain = 1usize << chunk_bits;
    let mut output = vec![F::zero(); chunk_domain * indices.len()];

    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            assert!(
                index < chunk_domain,
                "one-hot index {index} exceeds chunk domain {chunk_domain}"
            );
            output[cycle * chunk_domain + index] = F::one();
        }
    }

    output
}

/// Evaluates one-hot per-cycle indices at an address-chunk point.
///
/// The returned vector has one field element per cycle. Skipped entries
/// evaluate to zero.
pub fn one_hot_evals_at_chunk_point<F: Field>(indices: &[Option<u8>], point: &[F]) -> Vec<F> {
    let eq_table = EqPolynomial::<F>::evals(point, None);
    indices
        .iter()
        .map(|index| {
            index.map_or(F::zero(), |index| {
                let index = usize::from(index);
                assert!(
                    index < eq_table.len(),
                    "one-hot index {index} exceeds chunk point domain {}",
                    eq_table.len()
                );
                eq_table[index]
            })
        })
        .collect()
}

/// Returns most-significant-first chunk widths for a bitstring split by `chunk_bits`.
///
/// If the high chunk is partial, it appears first. The result is padded with
/// full-width chunks until it reaches `chunk_count`.
pub fn msb_chunk_bit_widths(
    total_bits: usize,
    chunk_bits: usize,
    chunk_count: usize,
) -> Vec<usize> {
    assert!(chunk_bits > 0, "chunk_bits must be nonzero");
    let first_chunk_bits = total_bits % chunk_bits;
    let mut widths = Vec::with_capacity(chunk_count);
    if first_chunk_bits != 0 {
        widths.push(first_chunk_bits);
    }
    while widths.len() < chunk_count {
        widths.push(chunk_bits);
    }
    widths
}

/// Splits a most-significant-first point into fixed-width chunks.
///
/// The high chunk is left-padded with zero challenges if the point length is
/// not a multiple of `chunk_bits`.
pub fn msb_point_chunks<F: Field>(point: &[F], chunk_bits: usize) -> Vec<Vec<F>> {
    assert!(chunk_bits > 0, "chunk_bits must be nonzero");
    let mut padded = Vec::new();
    let remainder = point.len() % chunk_bits;
    if remainder != 0 {
        padded.resize(chunk_bits - remainder, F::zero());
    }
    padded.extend_from_slice(point);
    padded
        .chunks(chunk_bits)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Computes `post - pre` in the field for a `u64` value transition.
pub fn u64_increment<F: Field>(pre: u64, post: u64) -> F {
    if post >= pre {
        F::from_u64(post - pre)
    } else {
        -F::from_u64(pre - post)
    }
}

/// Computes a field increment column from `(pre, post)` `u64` transitions.
pub fn u64_increment_column<F: Field>(transitions: impl IntoIterator<Item = (u64, u64)>) -> Vec<F> {
    transitions
        .into_iter()
        .map(|(pre, post)| u64_increment(pre, post))
        .collect()
}

/// Computes a field increment column where a missing write contributes zero.
pub fn optional_u64_increment_column<F: Field>(
    transitions: impl IntoIterator<Item = Option<(u64, u64)>>,
) -> Vec<F> {
    transitions
        .into_iter()
        .map(|transition| transition.map_or_else(F::zero, |(pre, post)| u64_increment(pre, post)))
        .collect()
}

/// Materializes an optional `usize` source column.
pub fn optional_usize_column(
    values: impl IntoIterator<Item = Option<usize>>,
) -> Vec<Option<usize>> {
    values.into_iter().collect()
}

/// Stage 4/5 sparse trace witness.
///
/// FR Twist witness data is stored as a sparse [`field_reg::FieldRegReplay`]
/// (bytecode + event stream) rather than as 5 pre-materialized `K_FR × T`
/// dense vectors. Stage 4/5 kernels call `replay.materialize_*` to expand
/// only the polys they need, kernel-scoped — host RSS stays ~10 MB
/// regardless of trace length instead of paying the ~500 MB the dense
/// materialization would cost. Empty `replay` (no FR events) is the inert
/// default — kernels short-circuit to zero-claim states.
#[derive(Clone, Debug)]
pub struct Stage45SparseTraceWitness<F: Field> {
    pub rd_inc: Vec<F>,
    pub ram_addresses: Vec<Option<usize>>,
    pub ram_inc: Vec<F>,
    pub rd_write_addresses: Vec<Option<usize>>,
    /// Sparse FR replay: bytecode metadata + FR event stream. Empty `events`
    /// is the inert default; `with_field_reg_replay` overlays a real replay.
    pub fr_replay: field_reg::FieldRegReplay,
    /// Cached `FrdInc(j) = limbs_to_field(rd_post[j]) − running_pre[bc.frd[j]]`
    /// for every cycle. Materialized once in `with_field_reg_replay` so the
    /// Stage 4 RW and Stage 5 ValEvaluation kernels can each clone-from-slice
    /// into their own mutable working buffers instead of paying the O(T) +
    /// per-event Fr conversion twice. Empty Vec when no FR replay attached.
    pub fr_frd_inc: Vec<F>,
}

impl<F: Field> Stage45SparseTraceWitness<F> {
    /// Attach the FR Twist replay and eagerly cache `FrdInc` (T-sized).
    /// The Stage 4/5 kernels read `fr_frd_inc` directly — avoids a
    /// duplicate materialize pass (saves ~one O(T)+event scan on FR-active
    /// proofs; significant at production trace lengths). Empty
    /// `replay.events` leaves the witness in the inert shape.
    pub fn with_field_reg_replay(mut self, replay: field_reg::FieldRegReplay) -> Self {
        assert_eq!(
            replay.num_cycles,
            self.rd_inc.len(),
            "FieldRegReplay.num_cycles must match the trace length"
        );
        self.fr_frd_inc = replay.materialize_frd_inc::<F>();
        self.fr_replay = replay;
        self
    }
}

pub fn stage4_5_sparse_trace_witness<F: Field>(
    register_writes: impl IntoIterator<Item = Option<(usize, u64, u64)>>,
    ram_accesses: impl IntoIterator<Item = (Option<usize>, u64, u64)>,
) -> Stage45SparseTraceWitness<F> {
    let mut rd_inc = Vec::new();
    let mut rd_write_addresses = Vec::new();
    for write in register_writes {
        if let Some((address, pre_value, post_value)) = write {
            rd_inc.push(u64_increment(pre_value, post_value));
            rd_write_addresses.push(Some(address));
        } else {
            rd_inc.push(F::zero());
            rd_write_addresses.push(None);
        }
    }

    let mut ram_addresses = Vec::new();
    let mut ram_inc = Vec::new();
    for (address, read_value, write_value) in ram_accesses {
        ram_addresses.push(address);
        ram_inc.push(u64_increment(read_value, write_value));
    }

    let trace_len = rd_inc.len();

    Stage45SparseTraceWitness {
        rd_inc,
        ram_addresses,
        ram_inc,
        rd_write_addresses,
        fr_replay: field_reg::FieldRegReplay::empty(trace_len),
        fr_frd_inc: Vec::new(),
    }
}

/// Evaluates a `u64`-valued multilinear extension at `point`.
pub fn mle_eval_u64<F: Field>(values: &[u64], point: &[F]) -> F {
    EqPolynomial::<F>::evals(point, None)
        .iter()
        .zip(values)
        .map(|(&weight, &value)| weight * F::from_u64(value))
        .sum()
}

/// Builds the Stage 4 `RamValInit` opening from the initial RAM image.
///
/// Stage 4 consumes this at the same address point as `RamValFinal`.
pub fn stage4_ram_val_init_opening<F: Field>(
    initial_ram_state: &[u64],
    ram_val_final_point: &[F],
) -> (Vec<F>, F) {
    (
        ram_val_final_point.to_vec(),
        mle_eval_u64(initial_ram_state, ram_val_final_point),
    )
}

/// Reverses a challenge point.
pub fn reverse_point<F: Copy>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// Returns the last `len` point coordinates in reverse order.
pub fn reversed_suffix<F: Copy>(point: &[F], len: usize) -> Vec<F> {
    let Some(start) = point.len().checked_sub(len) else {
        unreachable!("point is shorter than suffix length {len}");
    };
    point[start..].iter().rev().copied().collect()
}

/// Normalizes Stage 4 register read/write points to address-major order.
pub fn normalized_stage4_registers_rw_point<F: Copy>(
    log_t: usize,
    register_log_k: usize,
    point: &[F],
) -> Vec<F> {
    let expected = log_t + register_log_k;
    assert_eq!(
        point.len(),
        expected,
        "Stage 4 registers point length mismatch"
    );
    let (cycle, address) = point.split_at(log_t);
    address
        .iter()
        .rev()
        .copied()
        .chain(cycle.iter().rev().copied())
        .collect()
}

/// Extracts the Stage 5 instruction read-RAF cycle point.
pub fn stage5_instruction_cycle_point<F: Copy>(
    stage5_point: &[F],
    instruction_ra_virtual_d: usize,
    ra_virtual_log_k_chunk: usize,
    log_t: usize,
) -> Vec<F> {
    let address_len = instruction_ra_virtual_d * ra_virtual_log_k_chunk;
    let end = address_len + log_t;
    assert!(
        end <= stage5_point.len(),
        "Stage 5 point is shorter than instruction address plus cycle arity"
    );
    reverse_point(&stage5_point[address_len..end])
}

/// Builds a Stage 5 instruction RA opening point for one virtual address chunk.
pub fn stage5_instruction_ra_point<F: Copy>(
    stage5_point: &[F],
    instruction_ra_virtual_d: usize,
    ra_virtual_log_k_chunk: usize,
    log_t: usize,
    index: usize,
) -> Vec<F> {
    let start = index * ra_virtual_log_k_chunk;
    let end = start + ra_virtual_log_k_chunk;
    assert!(
        end <= stage5_point.len(),
        "Stage 5 point is shorter than instruction RA chunk {index}"
    );
    let mut point = stage5_point[start..end].to_vec();
    point.extend(stage5_instruction_cycle_point(
        stage5_point,
        instruction_ra_virtual_d,
        ra_virtual_log_k_chunk,
        log_t,
    ));
    point
}

/// Builds the Stage 5 RAM RA opening point from its input address and cycle point.
pub fn stage5_ram_ra_point<F: Copy>(
    stage5_input_point: &[F],
    stage5_point: &[F],
    log_k_ram: usize,
    log_t: usize,
) -> Vec<F> {
    assert!(
        stage5_input_point.len() >= log_k_ram,
        "Stage 5 RAM RA input point is shorter than RAM address arity"
    );
    let mut point = stage5_input_point[..log_k_ram].to_vec();
    point.extend(reversed_suffix(stage5_point, log_t));
    point
}

/// Builds the Stage 5 RegistersVal opening point from address and cycle points.
pub fn stage5_registers_val_point<F: Copy>(
    stage5_input_point: &[F],
    stage5_point: &[F],
    register_log_k: usize,
    log_t: usize,
) -> Vec<F> {
    assert!(
        stage5_input_point.len() >= register_log_k,
        "Stage 5 RegistersVal input point is shorter than register address arity"
    );
    let mut point = stage5_input_point[..register_log_k].to_vec();
    point.extend(reversed_suffix(stage5_point, log_t));
    point
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6WitnessParams {
    pub trace_len: usize,
    pub log_k_chunk: usize,
    pub log_k_bytecode: usize,
    pub log_k_ram: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    pub instruction_d: usize,
    pub instruction_ra_virtual_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeEntry<F: Field> {
    pub address: F,
    pub imm: F,
    pub circuit_flags: [bool; 23],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
    // FR-coprocessor slots: the bytecode-RAF entries expose these so the
    // Stage 6 bytecode-RAF binding can prove FR Twist's `FrRs1Ra / FrRs2Ra
    // / FrdWa` openings agree with per-cycle bytecode-derived one-hots —
    // preventing a malicious prover from smuggling an FR-active R1CS row
    // with zero operands by dropping the FR event entirely.
    pub frd: Option<usize>,
    pub frs1: Option<usize>,
    pub frs2: Option<usize>,
    pub reads_frs1: bool,
    pub reads_frs2: bool,
    pub writes_frd: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6OpeningInputRef<'a, F: Field> {
    pub symbol: &'a str,
    pub point: &'a [F],
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6WitnessInputs<'a, F: Field> {
    pub params: Stage6WitnessParams,
    pub cycle_inputs: &'a [CycleInput],
    pub opening_inputs: &'a [Stage6OpeningInputRef<'a, F>],
}

#[derive(Clone, Debug)]
pub struct Stage6WitnessPolynomials<F: Field> {
    pub instruction_ra_indices: Vec<Vec<Option<u8>>>,
    pub bytecode_ra_indices: Vec<Vec<Option<u8>>>,
    pub ram_ra_indices: Vec<Vec<Option<u8>>>,
    pub instruction_ra_booleanity: Vec<Vec<F>>,
    pub bytecode_ra_booleanity: Vec<Vec<F>>,
    pub ram_ra_booleanity: Vec<Vec<F>>,
    pub bytecode_ra_read_raf: Vec<Vec<F>>,
    pub bytecode_ra_read_raf_chunk_lens: Vec<usize>,
    pub instruction_ra_virtual: Vec<Vec<F>>,
    pub ram_ra_virtual: Vec<Vec<F>>,
    pub hamming_weight: Vec<F>,
    pub ram_inc: Vec<F>,
    pub rd_inc: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage6WitnessSlices<'a, F: Field> {
    pub booleanity_chunks: Vec<&'a [F]>,
    pub booleanity_index_chunks: Vec<&'a [Option<u8>]>,
    pub bytecode_ra_read_raf_chunks: Vec<&'a [F]>,
    pub bytecode_ra_read_raf_chunk_lens: Vec<usize>,
    pub ram_ra_virtual_chunks: Vec<&'a [F]>,
    pub instruction_ra_virtual_chunks: Vec<&'a [F]>,
    pub instruction_ra_index_chunks: Vec<&'a [Option<u8>]>,
    pub bytecode_ra_index_chunks: Vec<&'a [Option<u8>]>,
    pub ram_ra_index_chunks: Vec<&'a [Option<u8>]>,
}

impl<F: Field> Stage6WitnessPolynomials<F> {
    /// Returns borrowed slices in the order expected by the generated Stage 6/7 kernels.
    pub fn slices(&self) -> Stage6WitnessSlices<'_, F> {
        let mut booleanity_chunks = field_slices(&self.instruction_ra_booleanity);
        booleanity_chunks.extend(field_slices(&self.bytecode_ra_booleanity));
        booleanity_chunks.extend(field_slices(&self.ram_ra_booleanity));

        let mut booleanity_index_chunks = index_slices(&self.instruction_ra_indices);
        booleanity_index_chunks.extend(index_slices(&self.bytecode_ra_indices));
        booleanity_index_chunks.extend(index_slices(&self.ram_ra_indices));

        Stage6WitnessSlices {
            booleanity_chunks,
            booleanity_index_chunks,
            bytecode_ra_read_raf_chunks: field_slices(&self.bytecode_ra_read_raf),
            bytecode_ra_read_raf_chunk_lens: self.bytecode_ra_read_raf_chunk_lens.clone(),
            ram_ra_virtual_chunks: field_slices(&self.ram_ra_virtual),
            instruction_ra_virtual_chunks: field_slices(&self.instruction_ra_virtual),
            instruction_ra_index_chunks: index_slices(&self.instruction_ra_indices),
            bytecode_ra_index_chunks: index_slices(&self.bytecode_ra_indices),
            ram_ra_index_chunks: index_slices(&self.ram_ra_indices),
        }
    }
}

pub fn stage6_witness_polynomials<F: Field>(
    inputs: Stage6WitnessInputs<'_, F>,
) -> Stage6WitnessPolynomials<F> {
    let params = inputs.params;
    let trace_len = params.trace_len;
    assert!(
        inputs.cycle_inputs.len() <= trace_len,
        "cycle input length {} exceeds trace length {trace_len}",
        inputs.cycle_inputs.len()
    );

    let instruction_keys = one_hot_cycle_column(inputs.cycle_inputs, 0);
    let bytecode_indices_source = one_hot_cycle_column(inputs.cycle_inputs, 1);
    let ram_addresses = one_hot_cycle_column(inputs.cycle_inputs, 2);

    let instruction_indices = (0..params.instruction_d)
        .map(|index| {
            one_hot_chunk_indices(
                &instruction_keys,
                index,
                params.instruction_d,
                params.log_k_chunk,
                trace_len,
                Some(0),
            )
        })
        .collect::<Vec<_>>();
    let bytecode_indices = (0..params.bytecode_d)
        .map(|index| {
            one_hot_chunk_indices(
                &bytecode_indices_source,
                index,
                params.bytecode_d,
                params.log_k_chunk,
                trace_len,
                Some(0),
            )
        })
        .collect::<Vec<_>>();
    let ram_indices = (0..params.ram_d)
        .map(|index| {
            one_hot_chunk_indices(
                &ram_addresses,
                index,
                params.ram_d,
                params.log_k_chunk,
                trace_len,
                None,
            )
        })
        .collect::<Vec<_>>();

    let bytecode_ra_read_raf_chunk_lens =
        msb_chunk_bit_widths(params.log_k_bytecode, params.log_k_chunk, params.bytecode_d);

    let ram_address_chunks = stage6_ram_virtual_address_chunks(params, inputs.opening_inputs);
    assert_eq!(
        ram_address_chunks.len(),
        params.ram_d,
        "RAM Stage 6 address chunk count mismatch"
    );
    let ram_ra_virtual = ram_indices
        .iter()
        .zip(&ram_address_chunks)
        .map(|(indices, point)| one_hot_evals_at_chunk_point(indices, point))
        .collect::<Vec<_>>();

    let instruction_address_chunks =
        stage6_instruction_virtual_address_chunks(params, inputs.opening_inputs);
    assert_eq!(
        instruction_address_chunks.len(),
        params.instruction_d,
        "instruction Stage 6 address chunk count mismatch"
    );
    let instruction_ra_virtual = instruction_indices
        .iter()
        .zip(&instruction_address_chunks)
        .map(|(indices, point)| one_hot_evals_at_chunk_point(indices, point))
        .collect::<Vec<_>>();

    Stage6WitnessPolynomials {
        instruction_ra_indices: instruction_indices,
        bytecode_ra_indices: bytecode_indices,
        ram_ra_indices: ram_indices,
        instruction_ra_booleanity: Vec::new(),
        bytecode_ra_booleanity: Vec::new(),
        ram_ra_booleanity: Vec::new(),
        bytecode_ra_read_raf: Vec::new(),
        bytecode_ra_read_raf_chunk_lens,
        instruction_ra_virtual,
        ram_ra_virtual,
        hamming_weight: hamming_weight_from_cycle_inputs(inputs.cycle_inputs, trace_len),
        ram_inc: dense_cycle_column_to_field(inputs.cycle_inputs, 1, trace_len),
        rd_inc: dense_cycle_column_to_field(inputs.cycle_inputs, 0, trace_len),
    }
}

fn field_slices<F: Field>(values: &[Vec<F>]) -> Vec<&[F]> {
    values.iter().map(Vec::as_slice).collect()
}

fn index_slices(values: &[Vec<Option<u8>>]) -> Vec<&[Option<u8>]> {
    values.iter().map(Vec::as_slice).collect()
}

fn one_hot_cycle_column(cycle_inputs: &[CycleInput], slot: usize) -> Vec<Option<u128>> {
    cycle_inputs
        .iter()
        .map(|cycle| cycle.one_hot[slot])
        .collect()
}

fn dense_cycle_column_to_field<F: Field>(
    cycle_inputs: &[CycleInput],
    slot: usize,
    trace_len: usize,
) -> Vec<F> {
    assert!(
        cycle_inputs.len() <= trace_len,
        "cycle input length {} exceeds trace length {trace_len}",
        cycle_inputs.len()
    );
    let mut output = cycle_inputs
        .iter()
        .map(|cycle| F::from_i128(cycle.dense[slot]))
        .collect::<Vec<_>>();
    output.resize(trace_len, F::zero());
    output
}

fn hamming_weight_from_cycle_inputs<F: Field>(
    cycle_inputs: &[CycleInput],
    trace_len: usize,
) -> Vec<F> {
    assert!(
        cycle_inputs.len() <= trace_len,
        "cycle input length {} exceeds trace length {trace_len}",
        cycle_inputs.len()
    );
    let mut output = cycle_inputs
        .iter()
        .map(|cycle| {
            if cycle.one_hot[2].is_some() {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect::<Vec<_>>();
    output.resize(trace_len, F::zero());
    output
}

fn stage6_ram_virtual_address_chunks<F: Field>(
    params: Stage6WitnessParams,
    opening_inputs: &[Stage6OpeningInputRef<'_, F>],
) -> Vec<Vec<F>> {
    let point = stage6_opening_point(
        opening_inputs,
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    );
    assert!(
        point.len() >= params.log_k_ram,
        "RAM RA opening point is shorter than the RAM address arity"
    );
    msb_point_chunks(&point[..params.log_k_ram], params.log_k_chunk)
}

fn stage6_instruction_virtual_address_chunks<F: Field>(
    params: Stage6WitnessParams,
    opening_inputs: &[Stage6OpeningInputRef<'_, F>],
) -> Vec<Vec<F>> {
    let mut address = Vec::with_capacity(params.instruction_d * params.log_k_chunk);
    for index in 0..params.instruction_ra_virtual_d {
        let symbol = format!("stage6.input.stage5.instruction_read_raf.InstructionRa_{index}");
        let point = stage6_opening_point(opening_inputs, &symbol);
        assert!(
            point.len() >= params.lookups_ra_virtual_log_k_chunk,
            "instruction RA opening point is shorter than the virtual address chunk arity"
        );
        address.extend_from_slice(&point[..params.lookups_ra_virtual_log_k_chunk]);
    }
    msb_point_chunks(&address, params.log_k_chunk)
}

fn stage6_opening_point<'a, F: Field>(
    opening_inputs: &'a [Stage6OpeningInputRef<'_, F>],
    symbol: &str,
) -> &'a [F] {
    let Some(input) = opening_inputs.iter().find(|input| input.symbol == symbol) else {
        unreachable!("missing Stage 6 opening input `{symbol}`");
    };
    input.point
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[test]
    fn dense_column_converts_and_pads() {
        let output = dense_i128_column_to_field::<Fr>(&[5, -3], 4);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0], Fr::from_i128(5));
        assert_eq!(output[1], Fr::from_i128(-3));
        assert_eq!(output[2], Fr::from_u64(0));
        assert_eq!(output[3], Fr::from_u64(0));
    }

    #[test]
    fn cycle_sources_select_generated_trace_columns() {
        let cycle_inputs = [
            CycleInput {
                dense: [3, -2, 100],
                one_hot: [Some(7), Some(5), None, Some(2)],
            },
            CycleInput {
                dense: [8, 11, 200],
                one_hot: [Some(1), Some(4), Some(9), Some(6)],
            },
        ];
        let sources = commitment_trace_sources(&cycle_inputs);
        assert_eq!(sources.rd_inc, vec![3, 8]);
        assert_eq!(sources.ram_inc, vec![-2, 11]);
        assert_eq!(sources.field_reg_inc, vec![100, 200]);
        assert_eq!(sources.instruction_keys, vec![Some(7), Some(1)]);
        assert_eq!(sources.ram_addresses, vec![None, Some(9)]);
        assert_eq!(sources.bytecode_indices, vec![Some(5), Some(4)]);
        assert_eq!(sources.field_reg_indices, vec![Some(2), Some(6)]);
        assert_eq!(
            dense_cycle_source(&cycle_inputs, "trace.rd_inc"),
            vec![3, 8]
        );
        assert_eq!(
            dense_cycle_source(&cycle_inputs, "trace.ram_inc"),
            vec![-2, 11]
        );
        assert_eq!(
            dense_cycle_source(&cycle_inputs, "trace.field_reg_inc"),
            vec![100, 200]
        );
        assert_eq!(
            one_hot_cycle_source(&cycle_inputs, "trace.instruction_keys"),
            vec![Some(7), Some(1)]
        );
        assert_eq!(
            one_hot_cycle_source(&cycle_inputs, "trace.bytecode_indices"),
            vec![Some(5), Some(4)]
        );
        assert_eq!(
            one_hot_cycle_source(&cycle_inputs, "trace.ram_addresses"),
            vec![None, Some(9)]
        );
        assert_eq!(
            one_hot_cycle_source(&cycle_inputs, "trace.field_reg_indices"),
            vec![Some(2), Some(6)]
        );
    }

    #[test]
    fn increment_columns_compute_signed_field_deltas() {
        assert_eq!(u64_increment::<Fr>(2, 9), Fr::from_u64(7));
        assert_eq!(u64_increment::<Fr>(9, 2), -Fr::from_u64(7));
        assert_eq!(
            u64_increment_column::<Fr>([(5, 8), (8, 3)]),
            vec![Fr::from_u64(3), -Fr::from_u64(5)]
        );
        assert_eq!(
            optional_u64_increment_column::<Fr>([Some((5, 8)), None, Some((8, 3))]),
            vec![Fr::from_u64(3), Fr::from_u64(0), -Fr::from_u64(5)]
        );
    }

    #[test]
    fn stage4_5_sparse_trace_witness_groups_sparse_columns() {
        let witness = stage4_5_sparse_trace_witness::<Fr>(
            [Some((2, 5, 8)), None, Some((3, 9, 4))],
            [(Some(7), 10, 12), (None, 3, 3), (Some(8), 1, 0)],
        );

        assert_eq!(witness.rd_inc, vec![fr(3), fr(0), -fr(5)]);
        assert_eq!(witness.rd_write_addresses, vec![Some(2), None, Some(3)]);
        assert_eq!(witness.ram_addresses, vec![Some(7), None, Some(8)]);
        assert_eq!(witness.ram_inc, vec![fr(2), fr(0), -fr(1)]);
    }

    #[test]
    fn mle_eval_u64_matches_boolean_hypercube_points() {
        let values = [10, 20, 30, 40];
        let point = [Fr::from_u64(1), Fr::from_u64(0)];
        assert_eq!(mle_eval_u64(&values, &point), Fr::from_u64(30));
    }

    #[test]
    fn stage4_ram_val_init_opening_uses_final_ram_point() {
        let values = [10, 20, 30, 40];
        let point = [Fr::from_u64(1), Fr::from_u64(0)];
        let (opening_point, eval) = stage4_ram_val_init_opening(&values, &point);

        assert_eq!(opening_point, point);
        assert_eq!(eval, Fr::from_u64(30));
    }

    #[test]
    fn point_helpers_normalize_stage_points() {
        let point = [fr(1), fr(2), fr(3), fr(4), fr(5)];
        assert_eq!(
            reverse_point(&point),
            vec![fr(5), fr(4), fr(3), fr(2), fr(1)]
        );
        assert_eq!(reversed_suffix(&point, 3), vec![fr(5), fr(4), fr(3)]);
        assert_eq!(
            normalized_stage4_registers_rw_point(2, 3, &point),
            vec![fr(5), fr(4), fr(3), fr(2), fr(1)]
        );
    }

    #[test]
    fn stage5_point_helpers_compose_address_and_cycle_points() {
        let stage5_point = [fr(10), fr(11), fr(12), fr(13), fr(14), fr(15)];
        let input_point = [fr(20), fr(21), fr(22), fr(23)];
        assert_eq!(
            stage5_instruction_cycle_point(&stage5_point, 2, 2, 2),
            vec![fr(15), fr(14)]
        );
        assert_eq!(
            stage5_instruction_ra_point(&stage5_point, 2, 2, 2, 1),
            vec![fr(12), fr(13), fr(15), fr(14)]
        );
        assert_eq!(
            stage5_ram_ra_point(&input_point, &stage5_point, 3, 2),
            vec![fr(20), fr(21), fr(22), fr(15), fr(14)]
        );
        assert_eq!(
            stage5_registers_val_point(&input_point, &stage5_point, 2, 2),
            vec![fr(20), fr(21), fr(15), fr(14)]
        );
    }

    #[test]
    fn one_hot_chunks_are_address_major_and_msb_first() {
        let values = [Some(0xABu128), Some(0x12), None];
        let output = one_hot_chunk_address_major::<Fr>(&values, 0, 2, 4, 4, Some(0));

        assert_eq!(output.len(), 16 * 4);
        assert_eq!(output[0xA * 4], Fr::from_u64(1));
        assert_eq!(output[5], Fr::from_u64(1));
        assert_eq!(output[2], Fr::from_u64(1));
        assert_eq!(output[3], Fr::from_u64(1));
    }

    #[test]
    fn one_hot_address_major_from_indices_skips_none_entries() {
        let output = one_hot_address_major_from_indices::<Fr>(&[Some(2), None, Some(1)], 2);

        assert_eq!(output.len(), 12);
        assert_eq!(output[2 * 3], Fr::from_u64(1));
        assert_eq!(output[5], Fr::from_u64(1));
        assert_eq!(
            output
                .iter()
                .enumerate()
                .filter(|(_, value)| **value == Fr::from_u64(1))
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![5, 6]
        );
    }

    #[test]
    fn one_hot_cycle_major_from_indices_skips_none_entries() {
        let output = one_hot_cycle_major_from_indices::<Fr>(&[Some(2), None, Some(1)], 2);

        assert_eq!(output.len(), 12);
        assert_eq!(output[2], Fr::from_u64(1));
        assert_eq!(output[2 * 4 + 1], Fr::from_u64(1));
        assert_eq!(
            output
                .iter()
                .enumerate()
                .filter(|(_, value)| **value == Fr::from_u64(1))
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![2, 9]
        );
    }

    #[test]
    fn one_hot_evals_at_chunk_point_evaluates_sparse_indices() {
        let point = [Fr::from_u64(5), Fr::from_u64(7)];
        let eq = EqPolynomial::<Fr>::evals(&point, None);
        let output = one_hot_evals_at_chunk_point(&[Some(0), Some(3), None], &point);

        assert_eq!(output, vec![eq[0], eq[3], Fr::from_u64(0)]);
    }

    #[test]
    fn stage6_witness_slices_preserve_kernel_order() {
        let witness = Stage6WitnessPolynomials {
            instruction_ra_indices: vec![vec![Some(1)]],
            bytecode_ra_indices: vec![vec![Some(2)]],
            ram_ra_indices: vec![vec![None]],
            instruction_ra_booleanity: vec![vec![fr(10)]],
            bytecode_ra_booleanity: vec![vec![fr(20)]],
            ram_ra_booleanity: vec![vec![fr(30)]],
            bytecode_ra_read_raf: vec![vec![fr(40)]],
            bytecode_ra_read_raf_chunk_lens: vec![1],
            instruction_ra_virtual: vec![vec![fr(50)]],
            ram_ra_virtual: vec![vec![fr(60)]],
            hamming_weight: vec![fr(70)],
            ram_inc: vec![fr(80)],
            rd_inc: vec![fr(90)],
        };

        let slices = witness.slices();
        assert_eq!(
            slices.booleanity_chunks,
            vec![
                witness.instruction_ra_booleanity[0].as_slice(),
                witness.bytecode_ra_booleanity[0].as_slice(),
                witness.ram_ra_booleanity[0].as_slice(),
            ]
        );
        assert_eq!(
            slices.booleanity_index_chunks,
            vec![
                witness.instruction_ra_indices[0].as_slice(),
                witness.bytecode_ra_indices[0].as_slice(),
                witness.ram_ra_indices[0].as_slice(),
            ]
        );
        assert_eq!(
            slices.bytecode_ra_read_raf_chunks,
            vec![witness.bytecode_ra_read_raf[0].as_slice()]
        );
        assert_eq!(slices.bytecode_ra_read_raf_chunk_lens, vec![1]);
        assert_eq!(
            slices.instruction_ra_index_chunks,
            vec![witness.instruction_ra_indices[0].as_slice()]
        );
        assert_eq!(
            slices.bytecode_ra_index_chunks,
            vec![witness.bytecode_ra_indices[0].as_slice()]
        );
        assert_eq!(
            slices.ram_ra_index_chunks,
            vec![witness.ram_ra_indices[0].as_slice()]
        );
    }

    #[test]
    fn msb_chunk_bit_widths_puts_partial_high_chunk_first() {
        assert_eq!(msb_chunk_bit_widths(10, 4, 3), vec![2, 4, 4]);
        assert_eq!(msb_chunk_bit_widths(12, 4, 3), vec![4, 4, 4]);
    }

    #[test]
    fn msb_point_chunks_left_pads_partial_high_chunk() {
        let point = [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let chunks = msb_point_chunks(&point, 2);

        assert_eq!(
            chunks,
            vec![
                vec![Fr::from_u64(0), Fr::from_u64(1)],
                vec![Fr::from_u64(2), Fr::from_u64(3)]
            ]
        );
    }

    #[test]
    fn one_hot_chunk_indices_are_msb_first_and_padded() {
        let values = [Some(0xABu128), Some(0x12), None];
        let output = one_hot_chunk_indices(&values, 0, 2, 4, 4, Some(0));

        assert_eq!(output, vec![Some(0xA), Some(0x1), Some(0), Some(0)]);
    }

    #[test]
    fn one_hot_chunk_indices_preserve_skipped_entries() {
        let values = [Some(3u128), None];
        let output = one_hot_chunk_indices(&values, 0, 1, 2, 3, None);

        assert_eq!(output, vec![Some(3), None, None]);
    }

    #[test]
    fn one_hot_none_padding_skips_entries() {
        let values = [Some(3u128), None];
        let output = one_hot_chunk_address_major::<Fr>(&values, 0, 1, 2, 3, None);

        assert_eq!(output[3 * 3], Fr::from_u64(1));
        assert!(output
            .iter()
            .enumerate()
            .all(|(index, value)| index == 9 || *value == Fr::from_u64(0)));
    }
}
