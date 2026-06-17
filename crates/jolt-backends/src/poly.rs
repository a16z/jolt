#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    AdviceClaimReductionLayout, JoltCommittedPolynomial,
};
use jolt_field::{Field, RingAccumulator};
use jolt_poly::{eq_index_msb, EqPolynomial, MultilinearPoly};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JoltVmStage6Row, JoltVmStage6Rows},
    PolynomialChunk, WitnessProvider,
};

use crate::BackendError;

pub use crate::cpu::poly::{
    stage8_streaming_rlc_vector_matrix_product, Stage8StreamingRlcVectorMatrixProductInput,
};

const BACKEND: &str = "cpu";
const TASK: &str = "stage8 joint RLC polynomial";

#[derive(Clone, Copy, Debug)]
pub struct Stage8JointRlcConfig<'a> {
    pub log_t: usize,
    pub committed_chunk_bits: usize,
    pub layout: JoltRaPolynomialLayout,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
}

pub struct Stage8JointRlcSource<'a, F: Field, W> {
    config: Stage8JointRlcConfig<'a>,
    witness: &'a W,
    gamma_powers: Vec<F>,
    rows: Vec<JoltVmStage6Row>,
    field_rd_inc: Option<Vec<F>>,
}

impl<'a, F, W> Stage8JointRlcSource<'a, F, W>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
{
    pub fn new(
        config: Stage8JointRlcConfig<'a>,
        witness: &'a W,
        gamma_powers: Vec<F>,
        field_rd_inc: Option<Vec<F>>,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            config,
            witness,
            gamma_powers,
            rows: witness.stage6_rows()?,
            field_rd_inc,
        })
    }
}

impl<F, W> MultilinearPoly<F> for Stage8JointRlcSource<'_, F, W>
where
    F: Field,
    <F as jolt_field::WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
{
    fn num_vars(&self) -> usize {
        self.config.committed_chunk_bits + self.config.log_t
    }

    fn evaluate(&self, point: &[F]) -> F {
        backend_or_panic(evaluate_stage8_joint_polynomial_at_point(
            &self.config,
            self.witness,
            &self.gamma_powers,
            self.field_rd_inc.as_deref(),
            point,
        ))
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let evals = backend_or_panic(build_stage8_joint_polynomial_evals_with_field(
            &self.config,
            self.witness,
            &self.gamma_powers,
            self.field_rd_inc.as_deref(),
        ));
        let num_cols = 1usize << sigma;
        for (row, values) in evals.chunks(num_cols).enumerate() {
            f(row, values);
        }
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let expected_rows = 1usize << (self.num_vars() - sigma);
        assert_eq!(
            left.len(),
            expected_rows,
            "Stage 8 RLC fold left-vector length mismatch"
        );

        let ra_start = 2 + usize::from(self.field_rd_inc.is_some());
        let (instruction_coefficients, rest) =
            self.gamma_powers[ra_start..].split_at(self.config.layout.instruction());
        let (bytecode_coefficients, rest) = rest.split_at(self.config.layout.bytecode());
        let (ram_coefficients, _) = rest.split_at(self.config.layout.ram());
        let mut result = stage8_streaming_rlc_vector_matrix_product(
            Stage8StreamingRlcVectorMatrixProductInput {
                rows: &self.rows,
                field_rd_inc: self.field_rd_inc.as_deref(),
                log_t: self.config.log_t,
                committed_chunk_bits: self.config.committed_chunk_bits,
                trace_polynomial_order: self.config.trace_polynomial_order,
                ram_inc_coefficient: self.gamma_powers[0],
                rd_inc_coefficient: self.gamma_powers[1],
                field_rd_inc_coefficient: self.field_rd_inc.as_ref().map(|_| self.gamma_powers[2]),
                instruction_coefficients,
                bytecode_coefficients,
                ram_coefficients,
                left_vec: left,
                num_columns: num_cols,
            },
        );
        let mut batch_index = ra_start + self.config.layout.total();
        for (polynomial, layout) in [
            (
                JoltCommittedPolynomial::TrustedAdvice,
                self.config.trusted_advice_layout,
            ),
            (
                JoltCommittedPolynomial::UntrustedAdvice,
                self.config.untrusted_advice_layout,
            ),
        ] {
            let Some(layout) = layout else {
                continue;
            };
            let coefficient = self.gamma_powers[batch_index];
            batch_index += 1;
            backend_or_panic(fold_stage8_advice_rows(
                self.witness,
                polynomial,
                layout,
                coefficient,
                left,
                &mut result,
            ));
        }
        result
    }
}

#[expect(
    clippy::panic,
    reason = "MultilinearPoly cannot return Result; construction validates before PCS calls this adapter."
)]
fn backend_or_panic<T>(result: Result<T, BackendError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("Stage 8 streaming RLC source failed: {error}"),
    }
}

fn build_stage8_joint_polynomial_evals_with_field<F, W>(
    config: &Stage8JointRlcConfig<'_>,
    witness: &W,
    gamma_powers: &[F],
    field_rd_inc: Option<&[F]>,
) -> Result<Vec<F>, BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let num_rows = 1usize << config.log_t;
    let full = 1usize << (config.committed_chunk_bits + config.log_t);
    let mut joint = vec![F::zero(); full];

    for (batch_index, polynomial) in [
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::RdInc,
    ]
    .into_iter()
    .enumerate()
    {
        add_stage8_dense_constituent_to_joint(
            polynomial,
            config,
            witness,
            gamma_powers[batch_index],
            &mut joint,
        )?;
    }

    let mut batch_index = 2;
    if let Some(field_rd_inc) = field_rd_inc {
        if field_rd_inc.len() != num_rows {
            return invalid(format!(
                "field-inline rd_inc stream produced {} rows, expected {num_rows}",
                field_rd_inc.len()
            ));
        }
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        let addresses = 1usize << config.committed_chunk_bits;
        for (cycle, value) in field_rd_inc.iter().enumerate() {
            let flat = config
                .trace_polynomial_order
                .address_cycle_to_index(0, cycle, addresses, num_rows);
            joint[flat] += coefficient * *value;
        }
    }

    for polynomial in (0..config.layout.instruction())
        .map(JoltCommittedPolynomial::InstructionRa)
        .chain((0..config.layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa))
        .chain((0..config.layout.ram()).map(JoltCommittedPolynomial::RamRa))
    {
        let indices = collect_one_hot_ra_indices::<F, W>(polynomial, num_rows, witness)?;
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        let addresses = 1usize << config.committed_chunk_bits;
        for (cycle, opt_col) in indices.into_iter().enumerate() {
            if let Some(col) = opt_col {
                let flat = config.trace_polynomial_order.address_cycle_to_index(
                    usize::from(col),
                    cycle,
                    addresses,
                    num_rows,
                );
                joint[flat] += coefficient;
            }
        }
    }

    let mut batch_index = 2 + usize::from(field_rd_inc.is_some()) + config.layout.total();
    for (polynomial, layout) in [
        (
            JoltCommittedPolynomial::TrustedAdvice,
            config.trusted_advice_layout,
        ),
        (
            JoltCommittedPolynomial::UntrustedAdvice,
            config.untrusted_advice_layout,
        ),
    ] {
        let Some(layout) = layout else {
            continue;
        };
        add_stage8_advice_to_joint(
            polynomial,
            layout,
            witness,
            gamma_powers[batch_index],
            &mut joint,
        )?;
        batch_index += 1;
    }

    Ok(joint)
}

fn evaluate_stage8_joint_polynomial_at_point<F, W>(
    config: &Stage8JointRlcConfig<'_>,
    witness: &W,
    gamma_powers: &[F],
    field_rd_inc: Option<&[F]>,
    point: &[F],
) -> Result<F, BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let expected_vars = config
        .committed_chunk_bits
        .checked_add(config.log_t)
        .ok_or_else(|| invalid_error("joint evaluation point length overflow".to_owned()))?;
    if point.len() != expected_vars {
        return invalid(format!(
            "joint evaluation point has {} variables, expected {expected_vars}",
            point.len()
        ));
    }

    let num_rows = 1usize << config.log_t;
    let trace_eq = Stage8TraceEqTables::new(config, point);
    let mut result = F::zero();

    for (batch_index, polynomial) in [
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::RdInc,
    ]
    .into_iter()
    .enumerate()
    {
        add_stage8_dense_stream_evaluation(
            polynomial,
            config,
            witness,
            gamma_powers[batch_index],
            &trace_eq,
            &mut result,
        )?;
    }

    let mut batch_index = 2;
    if let Some(field_rd_inc) = field_rd_inc {
        if field_rd_inc.len() != num_rows {
            return invalid(format!(
                "field-inline rd_inc stream produced {} rows, expected {num_rows}",
                field_rd_inc.len()
            ));
        }
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        for (cycle, value) in field_rd_inc.iter().copied().enumerate() {
            if value.is_zero() {
                continue;
            }
            result += coefficient * value * trace_eq.weight(0, cycle);
        }
    }

    for polynomial in (0..config.layout.instruction())
        .map(JoltCommittedPolynomial::InstructionRa)
        .chain((0..config.layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa))
        .chain((0..config.layout.ram()).map(JoltCommittedPolynomial::RamRa))
    {
        let indices = collect_one_hot_ra_indices::<F, W>(polynomial, num_rows, witness)?;
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        for (cycle, opt_col) in indices.into_iter().enumerate() {
            let Some(col) = opt_col else {
                continue;
            };
            result += coefficient * trace_eq.weight(usize::from(col), cycle);
        }
    }

    let mut advice_batch_index = 2 + usize::from(field_rd_inc.is_some()) + config.layout.total();
    for (polynomial, layout) in [
        (
            JoltCommittedPolynomial::TrustedAdvice,
            config.trusted_advice_layout,
        ),
        (
            JoltCommittedPolynomial::UntrustedAdvice,
            config.untrusted_advice_layout,
        ),
    ] {
        let Some(layout) = layout else {
            continue;
        };
        add_stage8_advice_evaluation(
            polynomial,
            layout,
            witness,
            gamma_powers[advice_batch_index],
            point,
            &mut result,
        )?;
        advice_batch_index += 1;
    }

    Ok(result)
}

struct Stage8TraceEqTables<F> {
    address: Vec<F>,
    cycle: Vec<F>,
}

impl<F: Field> Stage8TraceEqTables<F> {
    fn new(config: &Stage8JointRlcConfig<'_>, point: &[F]) -> Self {
        let (address_point, cycle_point) = match config.trace_polynomial_order {
            TracePolynomialOrder::CycleMajor => {
                let (address, cycle) = point.split_at(config.committed_chunk_bits);
                (address, cycle)
            }
            TracePolynomialOrder::AddressMajor => {
                let (cycle, address) = point.split_at(config.log_t);
                (address, cycle)
            }
        };
        Self {
            address: EqPolynomial::<F>::evals(address_point, None),
            cycle: EqPolynomial::<F>::evals(cycle_point, None),
        }
    }

    #[inline]
    fn weight(&self, address: usize, cycle: usize) -> F {
        self.address[address] * self.cycle[cycle]
    }
}

fn add_stage8_dense_constituent_to_joint<F, W>(
    polynomial: JoltCommittedPolynomial,
    config: &Stage8JointRlcConfig<'_>,
    witness: &W,
    coefficient: F,
    joint: &mut [F],
) -> Result<(), BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let rows = 1usize << config.log_t;
    let addresses = 1usize << config.committed_chunk_bits;
    let mut stream = witness.committed_stream(polynomial, rows.max(1))?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::I128(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    let flat = config
                        .trace_polynomial_order
                        .address_cycle_to_index(0, index, addresses, rows);
                    joint[flat] += coefficient * F::from_i128(value);
                    index += 1;
                }
            }
            PolynomialChunk::Dense(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    let flat = config
                        .trace_polynomial_order
                        .address_cycle_to_index(0, index, addresses, rows);
                    joint[flat] += coefficient * value;
                    index += 1;
                }
            }
            PolynomialChunk::Zeros(count) => {
                index = index
                    .checked_add(count)
                    .ok_or_else(|| too_many_dense_rows(polynomial, rows))?;
                if index > rows {
                    return Err(too_many_dense_rows(polynomial, rows));
                }
            }
            _ => {
                return invalid(format!(
                    "expected a dense increment stream for {polynomial:?}"
                ));
            }
        }
    }
    require_exact_stream_len(polynomial, index, rows)
}

fn add_stage8_dense_stream_evaluation<F, W>(
    polynomial: JoltCommittedPolynomial,
    config: &Stage8JointRlcConfig<'_>,
    witness: &W,
    coefficient: F,
    trace_eq: &Stage8TraceEqTables<F>,
    result: &mut F,
) -> Result<(), BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let rows = 1usize << config.log_t;
    let mut stream = witness.committed_stream(polynomial, rows.max(1))?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::I128(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    *result += coefficient * F::from_i128(value) * trace_eq.weight(0, index);
                    index += 1;
                }
            }
            PolynomialChunk::Dense(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    *result += coefficient * value * trace_eq.weight(0, index);
                    index += 1;
                }
            }
            PolynomialChunk::Zeros(count) => {
                index = index
                    .checked_add(count)
                    .ok_or_else(|| too_many_dense_rows(polynomial, rows))?;
                if index > rows {
                    return Err(too_many_dense_rows(polynomial, rows));
                }
            }
            _ => {
                return invalid(format!(
                    "expected a dense increment stream for {polynomial:?}"
                ));
            }
        }
    }
    require_exact_stream_len(polynomial, index, rows)
}

fn add_stage8_advice_to_joint<F, W>(
    polynomial: JoltCommittedPolynomial,
    layout: &AdviceClaimReductionLayout,
    witness: &W,
    coefficient: F,
    joint: &mut [F],
) -> Result<(), BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let num_vars = layout.main_shape().total_vars();
    let num_cols = 1usize << num_vars.div_ceil(2);
    let advice_cols = 1usize << layout.advice_shape().column_vars();
    let advice_rows = 1usize << layout.advice_shape().row_vars();
    let mut stream = witness.committed_stream(polynomial, 1024)?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::U64(values) = chunk else {
            return invalid(format!("expected U64 advice stream for {polynomial:?}"));
        };
        for value in values {
            let row = index / advice_cols;
            let col = index % advice_cols;
            let flat = row * num_cols + col;
            joint[flat] += coefficient * F::from_u64(value);
            index += 1;
        }
    }
    let expected = advice_rows * advice_cols;
    require_exact_stream_len(polynomial, index, expected)
}

fn add_stage8_advice_evaluation<F, W>(
    polynomial: JoltCommittedPolynomial,
    layout: &AdviceClaimReductionLayout,
    witness: &W,
    coefficient: F,
    point: &[F],
    result: &mut F,
) -> Result<(), BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let num_vars = layout.main_shape().total_vars();
    if point.len() != num_vars {
        return invalid(format!(
            "advice evaluation point has {} variables, expected {num_vars}",
            point.len()
        ));
    }
    let num_cols = 1usize << num_vars.div_ceil(2);
    let advice_cols = 1usize << layout.advice_shape().column_vars();
    let advice_rows = 1usize << layout.advice_shape().row_vars();
    let mut stream = witness.committed_stream(polynomial, 1024)?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::U64(values) = chunk else {
            return invalid(format!("expected U64 advice stream for {polynomial:?}"));
        };
        for value in values {
            let row = index / advice_cols;
            let col = index % advice_cols;
            let flat = row * num_cols + col;
            if value != 0 {
                *result += coefficient * F::from_u64(value) * eq_index_msb(point, flat);
            }
            index += 1;
        }
    }
    let expected = advice_rows * advice_cols;
    require_exact_stream_len(polynomial, index, expected)
}

fn fold_stage8_advice_rows<F, W>(
    witness: &W,
    polynomial: JoltCommittedPolynomial,
    layout: &AdviceClaimReductionLayout,
    coefficient: F,
    left: &[F],
    result: &mut [F],
) -> Result<(), BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if coefficient.is_zero() {
        return Ok(());
    }
    let advice_cols = 1usize << layout.advice_shape().column_vars();
    let advice_rows = 1usize << layout.advice_shape().row_vars();
    if advice_cols > result.len() || advice_rows > left.len() {
        return invalid(format!(
            "advice block for {polynomial:?} does not fit the main Dory matrix"
        ));
    }
    let mut stream = witness.committed_stream(polynomial, 1024)?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::U64(values) = chunk else {
            return invalid(format!("expected U64 advice stream for {polynomial:?}"));
        };
        for value in values {
            let row = index / advice_cols;
            let col = index % advice_cols;
            result[col] += left[row] * coefficient * F::from_u64(value);
            index += 1;
        }
    }
    let expected = advice_rows * advice_cols;
    require_exact_stream_len(polynomial, index, expected)
}

#[cfg(feature = "field-inline")]
pub fn collect_stage8_field_rd_inc_rows<F, FI>(
    witness: &FI,
    rows: usize,
) -> Result<Vec<F>, BackendError>
where
    F: Field,
    FI: WitnessProvider<F, FieldInlineNamespace> + ?Sized,
{
    let mut stream = witness.committed_stream(FieldInlineCommittedPolynomial::FieldRdInc, 1024)?;
    let mut values = Vec::with_capacity(rows);
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::Dense(chunk) => values.extend(chunk),
            PolynomialChunk::Zeros(count) => {
                values.extend(std::iter::repeat_n(F::zero(), count));
            }
            _ => {
                return invalid("expected dense field-inline rd_inc stream".to_owned());
            }
        }
    }
    if values.len() != rows {
        return invalid(format!(
            "field-inline rd_inc stream produced {} rows, expected {rows}",
            values.len()
        ));
    }
    Ok(values)
}

fn collect_one_hot_ra_indices<F, W>(
    polynomial: JoltCommittedPolynomial,
    rows: usize,
    witness: &W,
) -> Result<Vec<Option<u8>>, BackendError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let mut stream = witness.committed_stream(polynomial, rows.max(1))?;
    let mut indices = Vec::with_capacity(rows);
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::OneHot(values) = chunk else {
            return invalid(format!("expected a one-hot stream for {polynomial:?}"));
        };
        for value in values {
            let index = match value {
                Some(index) => Some(u8::try_from(index).map_err(|_| {
                    invalid_error(format!(
                        "RA chunk index {index} for {polynomial:?} exceeds u8 range"
                    ))
                })?),
                None => None,
            };
            indices.push(index);
        }
    }
    if indices.len() != rows {
        return invalid(format!(
            "RA stream for {polynomial:?} produced {} rows, expected {rows}",
            indices.len()
        ));
    }
    Ok(indices)
}

fn too_many_dense_rows(polynomial: JoltCommittedPolynomial, rows: usize) -> BackendError {
    invalid_error(format!(
        "dense stream for {polynomial:?} exceeded {rows} rows"
    ))
}

fn require_exact_stream_len(
    polynomial: JoltCommittedPolynomial,
    got: usize,
    expected: usize,
) -> Result<(), BackendError> {
    if got == expected {
        return Ok(());
    }
    invalid(format!(
        "stream for {polynomial:?} produced {got} rows, expected {expected}"
    ))
}

fn invalid<T>(reason: String) -> Result<T, BackendError> {
    Err(invalid_error(reason))
}

fn invalid_error(reason: String) -> BackendError {
    BackendError::InvalidRequest {
        backend: BACKEND,
        task: TASK,
        reason,
    }
}
