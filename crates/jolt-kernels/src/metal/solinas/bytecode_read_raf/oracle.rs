use jolt_field::{CanonicalBytes, Field, FixedByteSize};

use super::{
    abi::shader_count, BytecodeReadRafError, BytecodeReadRafRowWords, BytecodeReadRafRun,
    BytecodeReadRafShape, BYTECODE_ADDRESS_BASE_STAGES, BYTECODE_ADDRESS_INNER_LOG2,
    BYTECODE_ADDRESS_ROUNDS, BYTECODE_ADDRESS_STAGES, BYTECODE_ADDRESS_VALUE_TABLES,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafTopology {
    pub occurrences: Vec<u32>,
    pub short_runs: Vec<BytecodeReadRafRun>,
    pub long_runs: Vec<BytecodeReadRafRun>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafTopologyStats {
    pub short_runs: usize,
    pub long_runs: usize,
    pub short_occurrences: usize,
    pub long_occurrences: usize,
    pub maximum_run: usize,
}

impl BytecodeReadRafTopology {
    pub fn stats(&self) -> BytecodeReadRafTopologyStats {
        let mut stats = BytecodeReadRafTopologyStats {
            short_runs: self.short_runs.len(),
            long_runs: self.long_runs.len(),
            ..BytecodeReadRafTopologyStats::default()
        };
        for run in &self.short_runs {
            let count = run.count as usize;
            stats.short_occurrences += count;
            stats.maximum_run = stats.maximum_run.max(count);
        }
        for run in &self.long_runs {
            let count = run.count as usize;
            stats.long_occurrences += count;
            stats.maximum_run = stats.maximum_run.max(count);
        }
        stats
    }

    pub fn run_count(&self) -> usize {
        self.short_runs.len() + self.long_runs.len()
    }
}

/// Builds the same occurrence partition without relying on GPU atomics or run order.
pub fn build_topology(
    rows: &[BytecodeReadRafRowWords],
    shape: BytecodeReadRafShape,
    short_threshold: usize,
) -> Result<BytecodeReadRafTopology, BytecodeReadRafError> {
    if rows.len() != shape.rows {
        return Err(BytecodeReadRafError::RowCount {
            expected: shape.rows,
            got: rows.len(),
        });
    }
    if short_threshold == 0 || short_threshold > shape.inner_length {
        return Err(BytecodeReadRafError::InvalidShortThreshold(short_threshold));
    }

    let mut occurrences = vec![u32::MAX; shape.rows];
    let mut short_runs = Vec::new();
    let mut long_runs = Vec::new();
    let mut counts = vec![0usize; shape.addresses];
    let mut cursors = vec![0usize; shape.addresses];

    for outer in 0..shape.outer_length {
        counts.fill(0);
        let row_start = outer * shape.inner_length;
        let row_end = row_start + shape.inner_length;
        for (row_index, row) in rows[row_start..row_end].iter().copied().enumerate() {
            let pc = checked_push_pc(row_start + row_index, row, shape.addresses)?;
            counts[pc] += 1;
        }

        let mut local_start = 0usize;
        for address in 0..shape.addresses {
            cursors[address] = local_start;
            let count = counts[address];
            if count != 0 {
                let run = BytecodeReadRafRun::new(row_start + local_start, count, outer, address)?;
                if count <= short_threshold {
                    short_runs.push(run);
                } else {
                    long_runs.push(run);
                }
            }
            local_start = local_start
                .checked_add(count)
                .ok_or(BytecodeReadRafError::SizeOverflow("local prefix"))?;
        }
        if local_start != shape.inner_length {
            return Err(BytecodeReadRafError::TopologyInvariant);
        }

        for (row_index, row) in rows[row_start..row_end].iter().copied().enumerate() {
            let pc = checked_push_pc(row_start + row_index, row, shape.addresses)?;
            let destination = row_start + cursors[pc];
            occurrences[destination] = shader_count("cycle index", row_start + row_index)?;
            cursors[pc] += 1;
        }
    }

    if occurrences.contains(&u32::MAX) || short_runs.len() + long_runs.len() > shape.run_capacity {
        return Err(BytecodeReadRafError::TopologyInvariant);
    }
    Ok(BytecodeReadRafTopology {
        occurrences,
        short_runs,
        long_runs,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafSplitEqTables<F> {
    pub e_lo: Vec<Vec<F>>,
    pub e_hi: Vec<Vec<F>>,
}

/// Builds big-endian equality tables without using the optimized CPU helper.
pub fn split_stage_eq_tables<F: Field>(
    stage_cycle_points: &[Vec<F>],
    shape: BytecodeReadRafShape,
) -> Result<BytecodeReadRafSplitEqTables<F>, BytecodeReadRafError> {
    if stage_cycle_points.len() != BYTECODE_ADDRESS_STAGES {
        return Err(BytecodeReadRafError::InvalidStageCount);
    }
    let log_rows = shape.rows.trailing_zeros() as usize;
    let hi_bits = shape.outer_length.trailing_zeros() as usize;
    if hi_bits + BYTECODE_ADDRESS_INNER_LOG2 != log_rows {
        return Err(BytecodeReadRafError::TopologyInvariant);
    }
    let mut e_lo = Vec::with_capacity(BYTECODE_ADDRESS_STAGES);
    let mut e_hi = Vec::with_capacity(BYTECODE_ADDRESS_STAGES);
    for (stage, point) in stage_cycle_points.iter().enumerate() {
        if point.len() != log_rows {
            return Err(BytecodeReadRafError::InvalidPointLength {
                stage,
                expected: log_rows,
                got: point.len(),
            });
        }
        e_hi.push(eq_table_oracle(&point[..hi_bits])?);
        e_lo.push(eq_table_oracle(&point[hi_bits..])?);
    }
    Ok(BytecodeReadRafSplitEqTables { e_lo, e_hi })
}

fn eq_table_oracle<F: Field>(point: &[F]) -> Result<Vec<F>, BytecodeReadRafError> {
    let length = 1usize
        .checked_shl(point.len() as u32)
        .ok_or(BytecodeReadRafError::SizeOverflow("equality table"))?;
    let mut table = Vec::with_capacity(length);
    table.push(F::one());
    for challenge in point {
        let one_minus = F::one() - *challenge;
        let mut next = Vec::with_capacity(table.len() * 2);
        for value in table {
            next.push(value * one_minus);
            next.push(value * *challenge);
        }
        table = next;
    }
    Ok(table)
}

/// Direct definition of all nine pushforwards, independent of the CSR algorithm.
pub fn direct_pushforward_oracle<F: Field>(
    rows: &[BytecodeReadRafRowWords],
    e_lo: &[Vec<F>],
    e_hi: &[Vec<F>],
    shape: BytecodeReadRafShape,
) -> Result<Vec<F>, BytecodeReadRafError> {
    validate_pushforward_inputs(rows, e_lo, e_hi, shape)?;
    let mut output = vec![F::zero(); BYTECODE_ADDRESS_STAGES * shape.addresses];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let address = checked_push_pc(row_index, row, shape.addresses)?;
        let inner = row_index & (shape.inner_length - 1);
        let outer = row_index / shape.inner_length;
        let increment = F::from_i128(row.fused_inc());
        for stage in 0..BYTECODE_ADDRESS_STAGES {
            let mut weight = e_hi[stage][outer] * e_lo[stage][inner];
            if stage >= BYTECODE_ADDRESS_BASE_STAGES {
                weight *= increment;
            }
            output[stage * shape.addresses + address] += weight;
        }
    }
    Ok(output)
}

/// CSR-form oracle used to check run partitioning separately from the direct definition.
pub fn topology_pushforward_oracle<F: Field>(
    rows: &[BytecodeReadRafRowWords],
    topology: &BytecodeReadRafTopology,
    e_lo: &[Vec<F>],
    e_hi: &[Vec<F>],
    shape: BytecodeReadRafShape,
) -> Result<Vec<F>, BytecodeReadRafError> {
    validate_pushforward_inputs(rows, e_lo, e_hi, shape)?;
    let stats = topology.stats();
    if topology.occurrences.len() != shape.rows
        || topology.run_count() > shape.run_capacity
        || stats.short_occurrences + stats.long_occurrences != shape.rows
    {
        return Err(BytecodeReadRafError::TopologyInvariant);
    }

    let mut output = vec![F::zero(); BYTECODE_ADDRESS_STAGES * shape.addresses];
    for run in topology.short_runs.iter().chain(&topology.long_runs) {
        let start = run.start as usize;
        let end = start
            .checked_add(run.count as usize)
            .ok_or(BytecodeReadRafError::TopologyInvariant)?;
        if end > topology.occurrences.len()
            || run.outer as usize >= shape.outer_length
            || run.address as usize >= shape.addresses
        {
            return Err(BytecodeReadRafError::TopologyInvariant);
        }
        let mut sums = [F::zero(); BYTECODE_ADDRESS_STAGES];
        for &row_index in &topology.occurrences[start..end] {
            let row_index = row_index as usize;
            let row = *rows
                .get(row_index)
                .ok_or(BytecodeReadRafError::TopologyInvariant)?;
            if row_index / shape.inner_length != run.outer as usize
                || checked_push_pc(row_index, row, shape.addresses)? != run.address as usize
            {
                return Err(BytecodeReadRafError::TopologyInvariant);
            }
            let inner = row_index & (shape.inner_length - 1);
            let increment = F::from_i128(row.fused_inc());
            for stage in 0..BYTECODE_ADDRESS_STAGES {
                let mut value = e_lo[stage][inner];
                if stage >= BYTECODE_ADDRESS_BASE_STAGES {
                    value *= increment;
                }
                sums[stage] += value;
            }
        }
        for stage in 0..BYTECODE_ADDRESS_STAGES {
            output[stage * shape.addresses + run.address as usize] +=
                sums[stage] * e_hi[stage][run.outer as usize];
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeStageValueSource {
    Table(usize),
    Complement(usize),
}

pub const BYTECODE_ADDRESS_STAGE_VALUES: [BytecodeStageValueSource; BYTECODE_ADDRESS_STAGES] = [
    BytecodeStageValueSource::Table(0),
    BytecodeStageValueSource::Table(1),
    BytecodeStageValueSource::Table(2),
    BytecodeStageValueSource::Table(3),
    BytecodeStageValueSource::Table(4),
    BytecodeStageValueSource::Table(5),
    BytecodeStageValueSource::Table(5),
    BytecodeStageValueSource::Complement(5),
    BytecodeStageValueSource::Complement(5),
];

pub struct BytecodeAddressMessageInputs<'a, F> {
    pub pushforwards: &'a [Vec<F>],
    pub values: &'a [Vec<F>],
    pub stage_values: &'a [BytecodeStageValueSource],
    pub stage_weights: &'a [F],
    pub raf_weights: &'a [F],
    pub int_table: &'a [F],
    pub entry_trace: &'a [F],
    pub entry_expected: &'a [F],
    pub entry_weight: F,
}

/// Returns the address summand at `t = 0` and `t = 2` for one host round.
pub fn address_message_oracle<F: Field>(
    inputs: BytecodeAddressMessageInputs<'_, F>,
) -> Result<[F; 2], BytecodeReadRafError> {
    if inputs.pushforwards.len() != BYTECODE_ADDRESS_STAGES
        || inputs.stage_values.len() != BYTECODE_ADDRESS_STAGES
        || inputs.stage_weights.len() != BYTECODE_ADDRESS_STAGES
        || inputs.raf_weights.len() != BYTECODE_ADDRESS_STAGES
    {
        return Err(BytecodeReadRafError::InvalidStageCount);
    }
    if inputs.values.len() != BYTECODE_ADDRESS_VALUE_TABLES {
        return Err(BytecodeReadRafError::InvalidValueTableCount {
            expected: BYTECODE_ADDRESS_VALUE_TABLES,
            got: inputs.values.len(),
        });
    }
    let elements = inputs.int_table.len();
    if elements < 2 || !elements.is_power_of_two() {
        return Err(BytecodeReadRafError::InvalidAddressTableLength(elements));
    }
    for (name, tables) in [
        ("pushforward", inputs.pushforwards),
        ("value", inputs.values),
    ] {
        for (index, table) in tables.iter().enumerate() {
            if table.len() != elements {
                return Err(BytecodeReadRafError::InvalidTableShape {
                    name,
                    index,
                    expected: elements,
                    got: table.len(),
                });
            }
        }
    }
    for (name, table) in [
        ("entry trace", inputs.entry_trace),
        ("entry expected", inputs.entry_expected),
    ] {
        if table.len() != elements {
            return Err(BytecodeReadRafError::InvalidTableShape {
                name,
                index: 0,
                expected: elements,
                got: table.len(),
            });
        }
    }

    let mut result = [F::zero(); 2];
    for pair_index in 0..elements / 2 {
        let int_lo = inputs.int_table[2 * pair_index];
        let int_hi = inputs.int_table[2 * pair_index + 1];
        let int_at_two = int_hi + int_hi - int_lo;
        for stage in 0..BYTECODE_ADDRESS_STAGES {
            let f_lo = inputs.pushforwards[stage][2 * pair_index];
            let f_hi = inputs.pushforwards[stage][2 * pair_index + 1];
            let (v_lo, v_hi) =
                stage_value_pair(inputs.values, inputs.stage_values[stage], pair_index)?;
            let val_at_zero = v_lo + inputs.raf_weights[stage] * int_lo;
            let val_at_two = v_hi + v_hi - v_lo + inputs.raf_weights[stage] * int_at_two;
            result[0] += inputs.stage_weights[stage] * f_lo * val_at_zero;
            result[1] += inputs.stage_weights[stage] * (f_hi + f_hi - f_lo) * val_at_two;
        }
        let trace_lo = inputs.entry_trace[2 * pair_index];
        let trace_hi = inputs.entry_trace[2 * pair_index + 1];
        let expected_lo = inputs.entry_expected[2 * pair_index];
        let expected_hi = inputs.entry_expected[2 * pair_index + 1];
        result[0] += inputs.entry_weight * trace_lo * expected_lo;
        result[1] += inputs.entry_weight
            * (trace_hi + trace_hi - trace_lo)
            * (expected_hi + expected_hi - expected_lo);
    }
    Ok(result)
}

pub struct BytecodeAddressOutputInputs<'a, F> {
    pub pushforwards: &'a [F],
    pub values: &'a [F],
    pub stage_values: &'a [BytecodeStageValueSource],
    pub stage_weights: &'a [F],
    pub raf_weights: &'a [F],
    pub int_value: F,
    pub entry_trace: F,
    pub entry_expected: F,
    pub entry_weight: F,
    pub committed_program: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytecodeAddressOutputClaims<F> {
    pub intermediate: F,
    pub val_stages: Vec<F>,
}

/// Evaluates the fully bound address relation and preserves raw value claims.
pub fn address_output_claims_oracle<F: Field>(
    inputs: BytecodeAddressOutputInputs<'_, F>,
) -> Result<BytecodeAddressOutputClaims<F>, BytecodeReadRafError> {
    if inputs.pushforwards.len() != BYTECODE_ADDRESS_STAGES
        || inputs.stage_values.len() != BYTECODE_ADDRESS_STAGES
        || inputs.stage_weights.len() != BYTECODE_ADDRESS_STAGES
        || inputs.raf_weights.len() != BYTECODE_ADDRESS_STAGES
    {
        return Err(BytecodeReadRafError::InvalidStageCount);
    }
    if inputs.values.len() != BYTECODE_ADDRESS_VALUE_TABLES {
        return Err(BytecodeReadRafError::InvalidValueTableCount {
            expected: BYTECODE_ADDRESS_VALUE_TABLES,
            got: inputs.values.len(),
        });
    }

    let mut intermediate = inputs.entry_weight * inputs.entry_trace * inputs.entry_expected;
    for stage in 0..BYTECODE_ADDRESS_STAGES {
        let value = bound_stage_value(inputs.values, inputs.stage_values[stage])?;
        intermediate += inputs.stage_weights[stage]
            * inputs.pushforwards[stage]
            * (value + inputs.raf_weights[stage] * inputs.int_value);
    }
    let val_stages = if inputs.committed_program {
        inputs.values.to_vec()
    } else {
        Vec::new()
    };
    Ok(BytecodeAddressOutputClaims {
        intermediate,
        val_stages,
    })
}

pub fn bind_multilinear_table<F: Field>(
    values: &[F],
    challenge: F,
) -> Result<Vec<F>, BytecodeReadRafError> {
    if values.len() < 2 || !values.len().is_power_of_two() {
        return Err(BytecodeReadRafError::InvalidAddressTableLength(
            values.len(),
        ));
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
        .collect())
}

pub fn address_opening_point<F: Copy>(
    sumcheck_challenges: &[F],
) -> Result<Vec<F>, BytecodeReadRafError> {
    if sumcheck_challenges.len() != BYTECODE_ADDRESS_ROUNDS {
        return Err(BytecodeReadRafError::InvalidAddressChallengeCount {
            expected: BYTECODE_ADDRESS_ROUNDS,
            got: sumcheck_challenges.len(),
        });
    }
    Ok(sumcheck_challenges.iter().rev().copied().collect())
}

pub fn canonical_field_checksum<F>(values: &[F]) -> u64
where
    F: CanonicalBytes,
{
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut checksum = FNV_OFFSET;
    let mut bytes = vec![0u8; <F as FixedByteSize>::NUM_BYTES];
    for value in values {
        value.to_bytes_le(&mut bytes);
        for byte in &bytes {
            checksum ^= u64::from(*byte);
            checksum = checksum.wrapping_mul(FNV_PRIME);
        }
    }
    checksum
}

fn bound_stage_value<F: Field>(
    values: &[F],
    source: BytecodeStageValueSource,
) -> Result<F, BytecodeReadRafError> {
    let (index, complement) = match source {
        BytecodeStageValueSource::Table(index) => (index, false),
        BytecodeStageValueSource::Complement(index) => (index, true),
    };
    let value = *values
        .get(index)
        .ok_or(BytecodeReadRafError::InvalidStageValueSource(index))?;
    if complement {
        Ok(F::one() - value)
    } else {
        Ok(value)
    }
}

fn stage_value_pair<F: Field>(
    values: &[Vec<F>],
    source: BytecodeStageValueSource,
    pair_index: usize,
) -> Result<(F, F), BytecodeReadRafError> {
    let (index, complement) = match source {
        BytecodeStageValueSource::Table(index) => (index, false),
        BytecodeStageValueSource::Complement(index) => (index, true),
    };
    let table = values
        .get(index)
        .ok_or(BytecodeReadRafError::InvalidStageValueSource(index))?;
    let lo = table[2 * pair_index];
    let hi = table[2 * pair_index + 1];
    if complement {
        Ok((F::one() - lo, F::one() - hi))
    } else {
        Ok((lo, hi))
    }
}

fn validate_pushforward_inputs<F>(
    rows: &[BytecodeReadRafRowWords],
    e_lo: &[Vec<F>],
    e_hi: &[Vec<F>],
    shape: BytecodeReadRafShape,
) -> Result<(), BytecodeReadRafError> {
    if rows.len() != shape.rows {
        return Err(BytecodeReadRafError::RowCount {
            expected: shape.rows,
            got: rows.len(),
        });
    }
    if e_lo.len() != BYTECODE_ADDRESS_STAGES || e_hi.len() != BYTECODE_ADDRESS_STAGES {
        return Err(BytecodeReadRafError::InvalidStageCount);
    }
    for stage in 0..BYTECODE_ADDRESS_STAGES {
        for (name, got, expected) in [
            ("E_lo", e_lo[stage].len(), shape.inner_length),
            ("E_hi", e_hi[stage].len(), shape.outer_length),
        ] {
            if got != expected {
                return Err(BytecodeReadRafError::InvalidTableShape {
                    name,
                    index: stage,
                    expected,
                    got,
                });
            }
        }
    }
    Ok(())
}

fn checked_push_pc(
    row_index: usize,
    row: BytecodeReadRafRowWords,
    addresses: usize,
) -> Result<usize, BytecodeReadRafError> {
    let pc = row.push_pc() as usize;
    if pc >= addresses {
        Err(BytecodeReadRafError::MappedPcOutsideDomain {
            row: row_index,
            pc,
            addresses,
        })
    } else {
        Ok(pc)
    }
}
