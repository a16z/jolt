use std::sync::Arc;

use jolt_field::Field;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};

use jolt_field::Fr;

use super::context::{context_for, CudaKernelContext};
use super::device::{require_fr_slice, DeviceFrVec};
use super::devices::{fan_out, DeviceTask};
use super::error::CudaError;
use super::half_fold::{half_fold, half_fold_into, FoldColumn, NarrowColumn, SummedHalf};
use super::sum_of_products::{DeviceSumOfProducts, SumOfProducts};

#[derive(Clone, Debug)]
pub struct PrefixSuffixPair<F: Field> {
    pub prefix: Vec<F>,
    pub suffix: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct PrefixSuffixGroup<F: Field> {
    pub pairs: Vec<PrefixSuffixPair<F>>,
    pub columns: Vec<(usize, F)>,
    pub constant: F,
}

pub const fn prefix_rounds_ceil(log_t: usize) -> Option<usize> {
    if log_t < 2 {
        return None;
    }
    Some(log_t - log_t / 2)
}

pub const fn prefix_rounds_floor(log_t: usize) -> Option<usize> {
    if log_t < 2 {
        return None;
    }
    Some(log_t / 2)
}

pub fn eq_pair<F: Field>(
    point: &[F],
    prefix_rounds: usize,
) -> Result<PrefixSuffixPair<F>, CudaError> {
    let (r_hi, r_lo) = split_point(point, prefix_rounds)?;
    Ok(PrefixSuffixPair {
        prefix: EqPolynomial::<F>::evals(r_lo, None),
        suffix: EqPolynomial::<F>::evals(r_hi, None),
    })
}

pub fn eq_plus_one_pairs<F: Field>(
    point: &[F],
    prefix_rounds: usize,
) -> Result<Vec<PrefixSuffixPair<F>>, CudaError> {
    let (r_hi, r_lo) = split_point(point, prefix_rounds)?;
    let eq_lo = EqPolynomial::<F>::evals(r_lo, None);
    let eq_hi = EqPolynomial::<F>::evals(r_hi, None);
    let mut is_max = vec![F::zero(); eq_lo.len()];
    is_max[0] = eq_lo[eq_lo.len() - 1];
    Ok(vec![
        PrefixSuffixPair {
            prefix: shifted(&eq_lo),
            suffix: eq_hi.clone(),
        },
        PrefixSuffixPair {
            prefix: is_max,
            suffix: shifted(&eq_hi),
        },
    ])
}

fn shifted<F: Field>(table: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); table.len()];
    out[1..].copy_from_slice(&table[..table.len() - 1]);
    out
}

fn split_point<F: Field>(point: &[F], prefix_rounds: usize) -> Result<(&[F], &[F]), CudaError> {
    if prefix_rounds == 0 || prefix_rounds >= point.len() {
        return Err(CudaError::InvariantViolation {
            reason: "a prefix-suffix split needs at least one variable on each side",
        });
    }
    Ok(point.split_at(point.len() - prefix_rounds))
}

fn window_accumulators<F: Field>(
    context: &CudaKernelContext,
    window: &PrefixSuffixWindow,
    groups: &[PrefixSuffixGroup<F>],
    prefix_len: usize,
) -> Result<Vec<Vec<Fr>>, CudaError> {
    let mut out = Vec::new();
    for group in groups {
        for pair in &group.pairs {
            let slice = pair
                .suffix
                .get(window.suffix_offset..window.suffix_offset + window.suffix_len)
                .ok_or(CudaError::InvariantViolation {
                    reason: "a prefix-suffix window lies outside the suffix table",
                })?;
            let suffix = context.upload(require_fr_slice(slice)?)?;
            let mut suffix_sum = F::zero();
            for value in slice {
                suffix_sum += *value;
            }
            let bias = group.constant * suffix_sum;
            let mut accumulator = context.alloc(prefix_len)?;
            for (index, &(column, coefficient)) in group.columns.iter().enumerate() {
                half_fold_into(
                    context,
                    window.columns.column(column)?,
                    &suffix,
                    &mut accumulator,
                    SummedHalf::High,
                    coefficient,
                    bias,
                    index > 0,
                )?;
            }
            out.push(accumulator.to_host()?);
        }
    }
    Ok(out)
}

struct PhaseOne {
    tables: Vec<DeviceFrVec>,
    form: DeviceSumOfProducts,
}

struct PhaseTwo {
    tables: Vec<DeviceFrVec>,
    column_offset: usize,
    column_count: usize,
    form: DeviceSumOfProducts,
}

enum Phase {
    One(PhaseOne),
    Two(PhaseTwo),
}

pub trait NarrowColumns: Send + Sync {
    fn count(&self) -> usize;

    fn entries(&self) -> usize;

    fn column(&self, index: usize) -> Option<NarrowColumn<'_>>;
}

pub enum ColumnSet {
    Field(Vec<DeviceFrVec>),
    Narrow(Arc<dyn NarrowColumns>),
}

impl ColumnSet {
    pub fn count(&self) -> usize {
        match self {
            Self::Field(columns) => columns.len(),
            Self::Narrow(columns) => columns.count(),
        }
    }

    pub fn entries(&self) -> usize {
        match self {
            Self::Field(columns) => columns.first().map_or(0, DeviceFrVec::len),
            Self::Narrow(columns) => columns.entries(),
        }
    }

    pub fn column(&self, index: usize) -> Result<FoldColumn<'_>, CudaError> {
        let absent = || CudaError::InvariantViolation {
            reason: "a prefix-suffix group names a column outside the claimed set",
        };
        match self {
            Self::Field(columns) => columns.get(index).map(FoldColumn::Field).ok_or_else(absent),
            Self::Narrow(columns) => columns
                .column(index)
                .map(FoldColumn::Narrow)
                .ok_or_else(absent),
        }
    }

    fn uniform(&self) -> bool {
        match self {
            Self::Field(columns) => {
                let entries = self.entries();
                columns.iter().all(|column| column.len() == entries)
            }
            Self::Narrow(_) => true,
        }
    }

    fn release(&mut self) {
        *self = Self::Field(Vec::new());
    }

    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        match self {
            Self::Field(columns) => columns.iter().map(DeviceFrVec::device_bytes).sum(),
            Self::Narrow(_) => 0,
        }
    }
}

pub struct PrefixSuffixWindow {
    pub ordinal: usize,
    pub columns: ColumnSet,
    pub suffix_offset: usize,
    pub suffix_len: usize,
}

pub struct PrefixSuffixRounds<F: Field> {
    context: &'static CudaKernelContext,
    groups: Vec<PrefixSuffixGroup<F>>,
    columns: ColumnSet,
    windows: Vec<PrefixSuffixWindow>,
    phase: Phase,
    challenges: Vec<F>,
    log_t: usize,
    prefix_rounds: usize,
    len: usize,
}

impl<F: Field> PrefixSuffixRounds<F> {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        self.columns.device_bytes()
            + self
                .windows
                .iter()
                .map(|window| window.columns.device_bytes())
                .sum::<usize>()
    }

    pub fn new(
        context: &'static CudaKernelContext,
        columns: ColumnSet,
        groups: Vec<PrefixSuffixGroup<F>>,
        prefix_rounds: usize,
    ) -> Result<Self, CudaError> {
        let log_t = Self::validate(&columns, &groups, prefix_rounds)?;
        let prefix_len = 1usize << prefix_rounds;
        let mut tables =
            Vec::with_capacity(2 * groups.iter().map(|g| g.pairs.len()).sum::<usize>());
        let mut form = SumOfProducts::<F>::new();
        for group in &groups {
            for pair in &group.pairs {
                let prefix = context.upload(require_fr_slice(&pair.prefix)?)?;
                let suffix = context.upload(require_fr_slice(&pair.suffix)?)?;
                let mut suffix_sum = F::zero();
                for value in &pair.suffix {
                    suffix_sum += *value;
                }
                let bias = group.constant * suffix_sum;
                let mut accumulator = context.alloc(prefix_len)?;
                for (index, &(column, coefficient)) in group.columns.iter().enumerate() {
                    half_fold_into(
                        context,
                        columns.column(column)?,
                        &suffix,
                        &mut accumulator,
                        SummedHalf::High,
                        coefficient,
                        bias,
                        index > 0,
                    )?;
                }
                let base = tables.len();
                tables.push(prefix);
                tables.push(accumulator);
                form.push(F::one(), &[base, base + 1])?;
            }
        }
        let form = form.upload(context)?;
        Ok(Self {
            context,
            groups,
            columns,
            windows: Vec::new(),
            phase: Phase::One(PhaseOne { tables, form }),
            challenges: Vec::with_capacity(log_t),
            log_t,
            prefix_rounds,
            len: prefix_len,
        })
    }

    pub fn new_windowed(
        context: &'static CudaKernelContext,
        windows: Vec<PrefixSuffixWindow>,
        groups: Vec<PrefixSuffixGroup<F>>,
        prefix_rounds: usize,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        if windows.len() == 1 {
            let window = windows
                .into_iter()
                .next()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a single-window prefix-suffix driver lost its columns",
                })?;
            return Self::new(context, window.columns, groups, prefix_rounds);
        }
        if prefix_rounds == 0 || prefix_rounds >= log_t {
            return Err(CudaError::InvariantViolation {
                reason: "a prefix-suffix split needs at least one round in each phase",
            });
        }
        let prefix_len = 1usize << prefix_rounds;
        let suffix_len = 1usize << (log_t - prefix_rounds);
        let column_count = windows.first().map(|window| window.columns.count()).ok_or(
            CudaError::InvariantViolation {
                reason: "a windowed prefix-suffix driver needs at least one window",
            },
        )?;
        let mut covered = 0;
        for window in &windows {
            if window.suffix_offset != covered || window.columns.count() != column_count {
                return Err(CudaError::InvariantViolation {
                    reason: "prefix-suffix windows must tile the suffix domain in order with the \
                             same column set",
                });
            }
            if window.columns.entries() != prefix_len * window.suffix_len
                || !window.columns.uniform()
            {
                return Err(CudaError::LengthMismatch {
                    expected: prefix_len * window.suffix_len,
                    got: window.columns.entries(),
                });
            }
            covered += window.suffix_len;
        }
        if covered != suffix_len {
            return Err(CudaError::LengthMismatch {
                expected: suffix_len,
                got: covered,
            });
        }
        for group in &groups {
            if group.columns.is_empty() || group.pairs.is_empty() {
                return Err(CudaError::InvariantViolation {
                    reason: "a prefix-suffix group needs at least one pair and one column term",
                });
            }
            if group
                .columns
                .iter()
                .any(|&(column, _)| column >= column_count)
            {
                return Err(CudaError::InvariantViolation {
                    reason: "a prefix-suffix group names a column outside the claimed set",
                });
            }
            for pair in &group.pairs {
                if pair.prefix.len() != prefix_len || pair.suffix.len() != suffix_len {
                    return Err(CudaError::LengthMismatch {
                        expected: prefix_len,
                        got: pair.prefix.len(),
                    });
                }
            }
        }

        let tasks: Vec<DeviceTask<'_, Vec<Vec<Fr>>, CudaError>> = windows
            .iter()
            .map(|window| {
                let groups = &groups;
                let task: DeviceTask<'_, Vec<Vec<Fr>>, CudaError> = Box::new(move || {
                    let device =
                        context_for(window.ordinal).ok_or(CudaError::InvariantViolation {
                            reason: "a prefix-suffix window names an absent device",
                        })?;
                    window_accumulators(device, window, groups, prefix_len)
                });
                task
            })
            .collect();
        let mut summed: Vec<Vec<Fr>> = Vec::new();
        for part in fan_out(tasks)? {
            if summed.is_empty() {
                summed = part;
                continue;
            }
            if part.len() != summed.len() {
                return Err(CudaError::LengthMismatch {
                    expected: summed.len(),
                    got: part.len(),
                });
            }
            for (total, addend) in summed.iter_mut().zip(&part) {
                if addend.len() != total.len() {
                    return Err(CudaError::LengthMismatch {
                        expected: total.len(),
                        got: addend.len(),
                    });
                }
                for (slot, value) in total.iter_mut().zip(addend) {
                    *slot += *value;
                }
            }
        }

        let mut tables = Vec::with_capacity(2 * summed.len());
        let mut form = SumOfProducts::<F>::new();
        let mut accumulators = summed.into_iter();
        for group in &groups {
            for pair in &group.pairs {
                let accumulator = accumulators.next().ok_or(CudaError::InvariantViolation {
                    reason: "a windowed prefix-suffix accumulator is missing for a pair",
                })?;
                let base = tables.len();
                tables.push(context.upload(require_fr_slice(&pair.prefix)?)?);
                tables.push(context.upload(&accumulator)?);
                form.push(F::one(), &[base, base + 1])?;
            }
        }
        let form = form.upload(context)?;
        Ok(Self {
            context,
            groups,
            columns: ColumnSet::Field(Vec::new()),
            windows,
            phase: Phase::One(PhaseOne { tables, form }),
            challenges: Vec::with_capacity(log_t),
            log_t,
            prefix_rounds,
            len: prefix_len,
        })
    }

    fn validate(
        columns: &ColumnSet,
        groups: &[PrefixSuffixGroup<F>],
        prefix_rounds: usize,
    ) -> Result<usize, CudaError> {
        if columns.count() == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "a prefix-suffix sumcheck needs at least one claimed column",
            });
        }
        let length = columns.entries();
        if length < 2 || !length.is_power_of_two() {
            return Err(CudaError::LengthMismatch {
                expected: length.next_power_of_two().max(2),
                got: length,
            });
        }
        if !columns.uniform() {
            return Err(CudaError::InvariantViolation {
                reason: "every prefix-suffix column must span the same number of cycles",
            });
        }
        let log_t = length.ilog2() as usize;
        if prefix_rounds == 0 || prefix_rounds >= log_t {
            return Err(CudaError::InvariantViolation {
                reason: "a prefix-suffix split needs at least one round in each phase",
            });
        }
        let prefix_len = 1usize << prefix_rounds;
        let suffix_len = length / prefix_len;
        if groups.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a prefix-suffix sumcheck needs at least one weight group",
            });
        }
        for group in groups {
            if group.pairs.is_empty() {
                return Err(CudaError::InvariantViolation {
                    reason: "a prefix-suffix group needs at least one (prefix, suffix) pair",
                });
            }
            if group.columns.is_empty() {
                return Err(CudaError::InvariantViolation {
                    reason: "a prefix-suffix group needs at least one column term; a \
                             constant-only group has nothing to fold",
                });
            }
            if group
                .columns
                .iter()
                .any(|&(column, _)| column >= columns.count())
            {
                return Err(CudaError::InvariantViolation {
                    reason: "a prefix-suffix group names a column outside the claimed set",
                });
            }
            for pair in &group.pairs {
                if pair.prefix.len() != prefix_len {
                    return Err(CudaError::LengthMismatch {
                        expected: prefix_len,
                        got: pair.prefix.len(),
                    });
                }
                if pair.suffix.len() != suffix_len {
                    return Err(CudaError::LengthMismatch {
                        expected: suffix_len,
                        got: pair.suffix.len(),
                    });
                }
            }
        }
        Ok(log_t)
    }

    pub fn column_claims(&self) -> Result<Vec<F>, CudaError> {
        let Phase::Two(phase) = &self.phase else {
            return Err(CudaError::InvariantViolation {
                reason: "prefix-suffix column claims are only readable after phase 2",
            });
        };
        if self.len != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len,
            });
        }
        phase.tables[phase.column_offset..phase.column_offset + phase.column_count]
            .iter()
            .map(|table| {
                super::device::fr_into(table.first()?).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    fn message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, CudaError> {
        let half = self.len / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let (tables, form) = match &self.phase {
            Phase::One(phase) => (&phase.tables, &phase.form),
            Phase::Two(phase) => (&phase.tables, &phase.form),
        };
        let handles: Vec<&DeviceFrVec> = tables.iter().collect();
        let lanes: Vec<F> = form.round_lanes(self.context, &handles, half, 1, true, 2)?;
        let [at_one, at_infinity] = lanes.as_slice() else {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: lanes.len(),
            });
        };
        let at_zero = previous_claim - *at_one;
        Ok(UnivariatePoly::from_evals_toom(&[
            at_zero,
            *at_one,
            *at_infinity,
        ]))
    }

    fn bind(&mut self, challenge: F) -> Result<(), CudaError> {
        let challenge = super::device::require_fr(challenge)?;
        let tables = match &mut self.phase {
            Phase::One(phase) => &mut phase.tables,
            Phase::Two(phase) => &mut phase.tables,
        };
        for table in tables.iter_mut() {
            *table = self.context.bind_rows(table, table.len(), challenge)?;
        }
        self.len /= 2;
        Ok(())
    }

    fn transition(&mut self) -> Result<(), CudaError> {
        let mut point: Vec<F> = self.challenges[..self.prefix_rounds].to_vec();
        point.reverse();
        let eq_lo = EqPolynomial::<F>::evals(&point, None);
        let device_eq_lo = self.context.upload(require_fr_slice(&eq_lo)?)?;
        let suffix_len = 1usize << (self.log_t - self.prefix_rounds);

        let mut tables = Vec::with_capacity(self.groups.len() + self.columns.count() + 1);
        for group in &self.groups {
            let mut weight = vec![F::zero(); suffix_len];
            for pair in &group.pairs {
                let mut prefix_eval = F::zero();
                for (value, eq) in pair.prefix.iter().zip(&eq_lo) {
                    prefix_eval += *value * *eq;
                }
                for (slot, value) in weight.iter_mut().zip(&pair.suffix) {
                    *slot += prefix_eval * *value;
                }
            }
            tables.push(self.context.upload(require_fr_slice(&weight)?)?);
        }
        let column_offset = tables.len();
        if self.windows.is_empty() {
            for index in 0..self.columns.count() {
                tables.push(half_fold(
                    self.context,
                    self.columns.column(index)?,
                    &device_eq_lo,
                    SummedHalf::Low,
                    F::one(),
                )?);
            }
        } else {
            for folded in self.windowed_column_folds(&eq_lo)? {
                tables.push(self.context.upload(&folded)?);
            }
        }
        let column_count = tables.len() - column_offset;

        let ones = if self.groups.iter().any(|group| group.constant != F::zero()) {
            let slot = tables.len();
            tables.push(
                self.context
                    .upload(require_fr_slice(&vec![F::one(); suffix_len])?)?,
            );
            Some(slot)
        } else {
            None
        };

        let mut form = SumOfProducts::<F>::new();
        for (index, group) in self.groups.iter().enumerate() {
            for &(column, coefficient) in &group.columns {
                form.push(coefficient, &[index, column_offset + column])?;
            }
            if let Some(ones) = ones.filter(|_| group.constant != F::zero()) {
                form.push(group.constant, &[index, ones])?;
            }
        }
        let form = form.upload(self.context)?;

        self.columns.release();
        self.windows = Vec::new();
        self.len = suffix_len;
        self.phase = Phase::Two(PhaseTwo {
            tables,
            column_offset,
            column_count,
            form,
        });
        Ok(())
    }

    fn windowed_column_folds(&self, eq_lo: &[F]) -> Result<Vec<Vec<Fr>>, CudaError> {
        let column_count = self
            .windows
            .first()
            .map(|window| window.columns.count())
            .ok_or(CudaError::InvariantViolation {
                reason: "a windowed prefix-suffix transition lost its windows",
            })?;
        let tasks: Vec<DeviceTask<'_, Vec<Vec<Fr>>, CudaError>> = self
            .windows
            .iter()
            .map(|window| {
                let task: DeviceTask<'_, Vec<Vec<Fr>>, CudaError> = Box::new(move || {
                    let device =
                        context_for(window.ordinal).ok_or(CudaError::InvariantViolation {
                            reason: "a prefix-suffix window names an absent device",
                        })?;
                    let weights = device.upload(require_fr_slice(eq_lo)?)?;
                    let mut folded = Vec::with_capacity(window.columns.count());
                    for index in 0..window.columns.count() {
                        folded.push(
                            half_fold(
                                device,
                                window.columns.column(index)?,
                                &weights,
                                SummedHalf::Low,
                                F::one(),
                            )?
                            .to_host()?,
                        );
                    }
                    Ok(folded)
                });
                task
            })
            .collect();
        let mut out: Vec<Vec<Fr>> = vec![Vec::new(); column_count];
        for part in fan_out(tasks)? {
            if part.len() != column_count {
                return Err(CudaError::LengthMismatch {
                    expected: column_count,
                    got: part.len(),
                });
            }
            for (whole, piece) in out.iter_mut().zip(part) {
                whole.extend(piece);
            }
        }
        Ok(out)
    }

    fn ingest(&mut self, challenge: F) -> Result<(), CudaError> {
        self.challenges.push(challenge);
        if self.challenges.len() == self.prefix_rounds {
            self.transition()
        } else {
            self.bind(challenge)
        }
    }
}

impl<F: Field> ProveRounds<F> for PrefixSuffixRounds<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.ingest(challenge)
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda prefix-suffix bind",
                })?;
        }
        self.message(previous_claim)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda prefix-suffix round",
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.ingest(bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda prefix-suffix bind",
            })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{EqPlusOnePolynomial, EqPlusOnePrefixSuffix, EqPolynomial, UnivariatePoly};
    use jolt_sumcheck::ProveRounds;
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{
        eq_pair, eq_plus_one_pairs, ColumnSet, PrefixSuffixGroup, PrefixSuffixPair,
        PrefixSuffixRounds, PrefixSuffixWindow,
    };

    fn zero() -> Fr {
        Fr::from_u64(0)
    }

    fn one() -> Fr {
        Fr::from_u64(1)
    }

    fn dense_weight(
        pairs: &[PrefixSuffixPair<Fr>],
        prefix_len: usize,
        suffix_len: usize,
    ) -> Vec<Fr> {
        let mut dense = vec![zero(); prefix_len * suffix_len];
        for pair in pairs {
            for hi in 0..suffix_len {
                for lo in 0..prefix_len {
                    dense[lo + hi * prefix_len] += pair.prefix[lo] * pair.suffix[hi];
                }
            }
        }
        dense
    }

    fn bind_low_to_high(table: &[Fr], challenge: Fr) -> Vec<Fr> {
        (0..table.len() / 2)
            .map(|i| table[2 * i] + challenge * (table[2 * i + 1] - table[2 * i]))
            .collect()
    }

    struct Oracle {
        weights: Vec<Vec<Fr>>,
        columns: Vec<Vec<Fr>>,
        terms: Vec<Vec<(usize, Fr)>>,
        constants: Vec<Fr>,
    }

    impl Oracle {
        fn claim(&self) -> Fr {
            let mut total = zero();
            for index in 0..self.columns[0].len() {
                for (group, weight) in self.weights.iter().enumerate() {
                    let mut linear = self.constants[group];
                    for &(column, coefficient) in &self.terms[group] {
                        linear += coefficient * self.columns[column][index];
                    }
                    total += weight[index] * linear;
                }
            }
            total
        }

        fn round_poly(&self) -> UnivariatePoly<Fr> {
            let half = self.columns[0].len() / 2;
            let mut evals = [zero(); 3];
            for (slot, eval) in evals.iter_mut().enumerate() {
                for y in 0..half {
                    for (group, weight) in self.weights.iter().enumerate() {
                        let mut linear = if slot == 2 {
                            zero()
                        } else {
                            self.constants[group]
                        };
                        for &(column, coefficient) in &self.terms[group] {
                            linear += coefficient * Self::lane(&self.columns[column], y, slot);
                        }
                        *eval += Self::lane(weight, y, slot) * linear;
                    }
                }
            }
            UnivariatePoly::from_evals_toom(&evals)
        }

        fn lane(table: &[Fr], y: usize, slot: usize) -> Fr {
            let lo = table[2 * y];
            let hi = table[2 * y + 1];
            match slot {
                0 => lo,
                1 => hi,
                _ => hi - lo,
            }
        }

        fn bind(&mut self, challenge: Fr) {
            for weight in &mut self.weights {
                *weight = bind_low_to_high(weight, challenge);
            }
            for column in &mut self.columns {
                *column = bind_low_to_high(column, challenge);
            }
        }
    }

    #[test]
    fn eq_plus_one_table_is_the_shifted_eq_table() {
        for log_t in 1usize..8 {
            let point: Vec<Fr> = (0..log_t).map(|i| fr(i as u64 * 31 + 7)).collect();
            let (eq, eq_plus_one) = EqPlusOnePolynomial::<Fr>::evals(&point, None);
            assert_eq!(eq_plus_one[0], zero(), "eq+1 must vanish at index 0");
            for index in 1..eq.len() {
                assert_eq!(
                    eq_plus_one[index],
                    eq[index - 1],
                    "eq+1[{index}] must equal eq[{}] at log_t {log_t}",
                    index - 1,
                );
            }
            assert_eq!(eq, EqPolynomial::<Fr>::evals(&point, None));
        }
    }

    #[test]
    fn eq_plus_one_pairs_match_the_host_decomposition() {
        for log_t in 2usize..9 {
            let point: Vec<Fr> = (0..log_t).map(|i| fr(i as u64 * 17 + 3)).collect();
            let expected = EqPlusOnePrefixSuffix::<Fr>::new(&point);
            let got = eq_plus_one_pairs(&point, log_t - log_t / 2).expect("eq+1 pairs");
            assert_eq!(got[0].prefix, expected.prefix_0);
            assert_eq!(got[0].suffix, expected.suffix_0);
            assert_eq!(got[1].prefix, expected.prefix_1);
            assert_eq!(got[1].suffix, expected.suffix_1);
        }
    }

    #[test]
    fn eq_plus_one_pairs_reproduce_the_dense_table_at_any_split() {
        for log_t in 2usize..8 {
            let point: Vec<Fr> = (0..log_t).map(|i| fr(i as u64 * 11 + 5)).collect();
            let (_, expected) = EqPlusOnePolynomial::<Fr>::evals(&point, None);
            for prefix_rounds in 1..log_t {
                let pairs = eq_plus_one_pairs(&point, prefix_rounds).expect("eq+1 pairs");
                let prefix_len = 1usize << prefix_rounds;
                let got = dense_weight(&pairs, prefix_len, 1usize << (log_t - prefix_rounds));
                assert_eq!(got, expected, "eq+1 split at {prefix_rounds} of {log_t}");
            }
        }
    }

    #[test]
    fn eq_pair_reproduces_the_dense_table_at_any_split() {
        for log_t in 2usize..8 {
            let point: Vec<Fr> = (0..log_t).map(|i| fr(i as u64 * 23 + 9)).collect();
            let expected = EqPolynomial::<Fr>::evals(&point, None);
            for prefix_rounds in 1..log_t {
                let pair = eq_pair(&point, prefix_rounds).expect("eq pair");
                let prefix_len = 1usize << prefix_rounds;
                let got = dense_weight(
                    std::slice::from_ref(&pair),
                    prefix_len,
                    1usize << (log_t - prefix_rounds),
                );
                assert_eq!(got, expected, "eq split at {prefix_rounds} of {log_t}");
            }
        }
    }

    fn drive(
        log_t: usize,
        prefix_rounds: usize,
        groups: Vec<PrefixSuffixGroup<Fr>>,
        columns: Vec<Vec<Fr>>,
        seed: u64,
    ) -> Result<(), TestCaseError> {
        let Some(context) = shared_context() else {
            return Ok(());
        };
        let prefix_len = 1usize << prefix_rounds;
        let suffix_len = 1usize << (log_t - prefix_rounds);
        let mut oracle = Oracle {
            weights: groups
                .iter()
                .map(|group| dense_weight(&group.pairs, prefix_len, suffix_len))
                .collect(),
            columns: columns.clone(),
            terms: groups.iter().map(|group| group.columns.clone()).collect(),
            constants: groups.iter().map(|group| group.constant).collect(),
        };
        let uploaded: Vec<_> = columns
            .iter()
            .map(|column| context.upload(column).expect("upload column"))
            .collect();
        let mut got = PrefixSuffixRounds::<Fr>::new(
            context,
            ColumnSet::Field(uploaded),
            groups,
            prefix_rounds,
        )
        .expect("device prefix-suffix driver");

        prop_assert_eq!(got.num_rounds(), log_t);
        let mut claim = oracle.claim();
        let mut bind = None;
        for round in 0..log_t {
            let expected = oracle.round_poly();
            let message = got
                .prove_round(bind, round, claim)
                .expect("device round polynomial");
            prop_assert_eq!(
                message.coefficients().to_vec(),
                expected.coefficients().to_vec(),
                "round polynomial diverged at round {}",
                round
            );
            let challenge = fr(seed ^ (round as u64 * 977 + 13));
            claim = expected.evaluate(challenge);
            oracle.bind(challenge);
            bind = Some(challenge);
        }
        got.finish_rounds(bind.expect("a final challenge"))
            .expect("device finish");

        let expected: Vec<Fr> = oracle.columns.iter().map(|column| column[0]).collect();
        prop_assert_eq!(
            got.column_claims().expect("device column claims"),
            expected,
            "column claims diverged"
        );
        Ok(())
    }

    fn drive_windowed(
        log_t: usize,
        prefix_rounds: usize,
        shards: usize,
        groups: Vec<PrefixSuffixGroup<Fr>>,
        columns: Vec<Vec<Fr>>,
        seed: u64,
    ) -> Result<(), TestCaseError> {
        let Some(context) = shared_context() else {
            return Ok(());
        };
        let prefix_len = 1usize << prefix_rounds;
        let suffix_len = 1usize << (log_t - prefix_rounds);
        let mut oracle = Oracle {
            weights: groups
                .iter()
                .map(|group| dense_weight(&group.pairs, prefix_len, suffix_len))
                .collect(),
            columns: columns.clone(),
            terms: groups.iter().map(|group| group.columns.clone()).collect(),
            constants: groups.iter().map(|group| group.constant).collect(),
        };
        let window_suffix = suffix_len / shards;
        let windows: Vec<PrefixSuffixWindow> = (0..shards)
            .map(|shard| {
                let base = shard * window_suffix * prefix_len;
                PrefixSuffixWindow {
                    ordinal: 0,
                    columns: ColumnSet::Field(
                        columns
                            .iter()
                            .map(|column| {
                                context
                                    .upload(&column[base..base + window_suffix * prefix_len])
                                    .expect("upload window column")
                            })
                            .collect(),
                    ),
                    suffix_offset: shard * window_suffix,
                    suffix_len: window_suffix,
                }
            })
            .collect();
        let mut got =
            PrefixSuffixRounds::<Fr>::new_windowed(context, windows, groups, prefix_rounds, log_t)
                .expect("windowed prefix-suffix driver");

        prop_assert_eq!(got.num_rounds(), log_t);
        let mut claim = oracle.claim();
        let mut bind = None;
        for round in 0..log_t {
            let expected = oracle.round_poly();
            let message = got
                .prove_round(bind, round, claim)
                .expect("windowed round polynomial");
            prop_assert_eq!(
                message.coefficients().to_vec(),
                expected.coefficients().to_vec(),
                "shards {} round {}: the windowed round polynomial diverged",
                shards,
                round
            );
            let challenge = fr(seed ^ (round as u64 * 977 + 13));
            claim = expected.evaluate(challenge);
            oracle.bind(challenge);
            bind = Some(challenge);
        }
        got.finish_rounds(bind.expect("a final challenge"))
            .expect("windowed finish");

        let expected: Vec<Fr> = oracle.columns.iter().map(|column| column[0]).collect();
        prop_assert_eq!(
            got.column_claims().expect("windowed column claims"),
            expected,
            "shards {}: the windowed column claims diverged",
            shards
        );
        Ok(())
    }

    #[test]
    fn windowed_prefix_suffix_matches_the_whole_domain_round_for_round() {
        for log_t in 4usize..8 {
            let prefix_rounds = log_t - log_t / 2;
            let suffix_rounds = log_t - prefix_rounds;
            let seed = 0xA11Cu64;
            let columns = random_columns(5, 1usize << log_t, seed);
            let outer: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 53 + 7))).collect();
            let product: Vec<Fr> = (0..log_t)
                .map(|i| fr(seed ^ (i as u64 * 97 + 19)))
                .collect();
            let gamma = fr(seed ^ 0xbeef);
            let mut powers = vec![one()];
            for index in 1..5 {
                powers.push(powers[index - 1] * gamma);
            }
            for shards in [2usize, 4] {
                if shards > 1usize << suffix_rounds {
                    continue;
                }
                let groups = vec![
                    PrefixSuffixGroup {
                        pairs: eq_plus_one_pairs(&outer, prefix_rounds).expect("eq+1 outer"),
                        columns: (0..4).map(|c| (c, powers[c])).collect(),
                        constant: zero(),
                    },
                    PrefixSuffixGroup {
                        pairs: eq_plus_one_pairs(&product, prefix_rounds).expect("eq+1 product"),
                        columns: vec![(4, -powers[4])],
                        constant: powers[4],
                    },
                ];
                drive_windowed(log_t, prefix_rounds, shards, groups, columns.clone(), seed)
                    .expect("windowed drive");
            }
        }
    }

    fn random_columns(count: usize, len: usize, seed: u64) -> Vec<Vec<Fr>> {
        (0..count)
            .map(|c| {
                (0..len)
                    .map(|i| fr(seed ^ ((c as u64) << 40) ^ (i as u64 * 31 + 7)))
                    .collect()
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn one_pair_prefix_suffix_matches_cpu_round_for_round(
            log_t in 2usize..7,
            split in 1usize..6,
            column_count in 1usize..4,
            seed in any::<u64>(),
        ) {
            let prefix_rounds = 1 + split % (log_t - 1);
            let columns = random_columns(column_count, 1usize << log_t, seed);
            let point: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 101 + 3))).collect();
            let group = PrefixSuffixGroup {
                pairs: vec![eq_pair(&point, prefix_rounds).expect("eq pair")],
                columns: (0..column_count)
                    .map(|c| (c, fr(seed ^ (c as u64 * 613 + 29))))
                    .collect(),
                constant: zero(),
            };
            drive(log_t, prefix_rounds, vec![group], columns, seed)?;
        }

        #[test]
        fn two_pair_prefix_suffix_matches_cpu_round_for_round(
            log_t in 2usize..7,
            split in 1usize..6,
            column_count in 1usize..4,
            seed in any::<u64>(),
        ) {
            let prefix_rounds = 1 + split % (log_t - 1);
            let columns = random_columns(column_count, 1usize << log_t, seed);
            let point: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 71 + 11))).collect();
            let group = PrefixSuffixGroup {
                pairs: eq_plus_one_pairs(&point, prefix_rounds).expect("eq+1 pairs"),
                columns: (0..column_count)
                    .map(|c| (c, fr(seed ^ (c as u64 * 409 + 17))))
                    .collect(),
                constant: fr(seed ^ 0xc0de),
            };
            drive(log_t, prefix_rounds, vec![group], columns, seed)?;
        }

        #[test]
        fn two_group_prefix_suffix_matches_cpu_round_for_round(
            log_t in 3usize..7,
            split in 1usize..6,
            seed in any::<u64>(),
        ) {
            let prefix_rounds = 1 + split % (log_t - 1);
            let columns = random_columns(5, 1usize << log_t, seed);
            let outer: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 53 + 7))).collect();
            let product: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 97 + 19))).collect();
            let gamma = fr(seed ^ 0xbeef);
            let mut powers = vec![one()];
            for index in 1..5 {
                powers.push(powers[index - 1] * gamma);
            }
            let groups = vec![
                PrefixSuffixGroup {
                    pairs: eq_plus_one_pairs(&outer, prefix_rounds).expect("eq+1 outer"),
                    columns: (0..4).map(|c| (c, powers[c])).collect(),
                    constant: zero(),
                },
                PrefixSuffixGroup {
                    pairs: eq_plus_one_pairs(&product, prefix_rounds).expect("eq+1 product"),
                    columns: vec![(4, -powers[4])],
                    constant: powers[4],
                },
            ];
            drive(log_t, prefix_rounds, groups, columns, seed)?;
        }
    }
}
