#![allow(dead_code)]

use std::borrow::Cow;

use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_field::{Field, FieldAccumulator, Fr};
use jolt_openings::CommitmentScheme as _;
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};
use jolt_witness::{dense_i128_column_to_field, one_hot_chunk_address_major, one_hot_chunk_indices, optional_field_oracle, CommitmentTraceSources};
use rayon::prelude::*;

pub type DefaultCommitmentTranscript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OraclePlan {
    pub oracle: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentBatchPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle_family: &'static str,
    pub label: &'static str,
    pub oracles: &'static [&'static str],
    pub count: usize,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionalSkipPolicy {
    MissingOrZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionalCommitmentPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
    pub skip_policy: OptionalSkipPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptStep {
    pub label: &'static str,
    pub source: &'static str,
    pub optional: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentProverProgramPlan {
    pub params: CommitmentParams,
    pub oracle_plans: &'static [OraclePlan],
    pub batch_plans: &'static [CommitmentBatchPlan],
    pub optional_plans: &'static [OptionalCommitmentPlan],
    pub transcript_steps: &'static [TranscriptStep],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRecord {
    pub artifact: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Debug)]
pub struct OracleOpeningHint {
    pub oracle: &'static str,
    pub hint: DoryHint,
}

#[derive(Clone, Debug)]
pub struct CommittedOracle {
    pub commitment: Option<DoryCommitment>,
    pub record: CommitmentRecord,
    pub hint: Option<OracleOpeningHint>,
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentArtifacts {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
    pub hints: Vec<OracleOpeningHint>,
}

pub trait CommitmentInputProvider {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>>;

    fn materialize_with_num_vars(
        &mut self,
        oracle: &'static str,
        _num_vars: usize,
    ) -> Option<Cow<'_, [Fr]>> {
        self.materialize(oracle)
    }

    fn commit_batch(
        &mut self,
        _program: &CommitmentProverProgramPlan,
        _plan: &CommitmentBatchPlan,
        _prover_setup: &DoryProverSetup,
    ) -> Option<Result<Vec<CommittedOracle>, CommitmentPhaseError>> {
        None
    }

    fn add_scaled_to_joint(
        &mut self,
        _oracle: &'static str,
        _joint: &mut [Fr],
        _num_vars: usize,
        _limit: usize,
        _scalar: Fr,
    ) -> bool {
        false
    }

    fn add_scaled_to_joint_batch(
        &mut self,
        requests: &[JointContribution],
        joint: &mut [Fr],
    ) -> Vec<bool> {
        requests
            .iter()
            .map(|request| {
                self.add_scaled_to_joint(
                    request.oracle,
                    joint,
                    request.num_vars,
                    request.limit,
                    request.scalar,
                )
            })
            .collect()
    }

    fn open_joint_polynomial<T>(
        &mut self,
        _requests: &[JointContribution],
        _opening_point: &[Fr],
        _joint_claim: Fr,
        _prover_setup: &DoryProverSetup,
        hint: DoryHint,
        _transcript: &mut T,
    ) -> Result<jolt_dory::DoryProof, DoryHint>
    where
        T: Transcript<Challenge = Fr>,
    {
        Err(hint)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JointContribution {
    pub oracle: &'static str,
    pub num_vars: usize,
    pub limit: usize,
    pub scalar: Fr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingOracle { oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    OracleTooLarge { oracle: &'static str, len: usize, target_len: usize },
    TargetSizeOverflow { num_vars: usize },
}

#[derive(Clone, Copy)]
pub struct CommitmentOracleInputs<'a> {
    pub rd_inc: &'a [i128],
    pub ram_inc: &'a [i128],
    pub instruction_keys: &'a [Option<u128>],
    pub ram_addresses: &'a [Option<u128>],
    pub bytecode_indices: &'a [Option<u128>],
    pub untrusted_advice: Option<&'a [Fr]>,
    pub trusted_advice: Option<&'a [Fr]>,
}

impl<'a> CommitmentOracleInputs<'a> {
    pub fn from_trace_sources(
        sources: &'a CommitmentTraceSources,
        untrusted_advice: Option<&'a [Fr]>,
        trusted_advice: Option<&'a [Fr]>,
    ) -> Self {
        Self {
            rd_inc: &sources.rd_inc,
            ram_inc: &sources.ram_inc,
            instruction_keys: &sources.instruction_keys,
            ram_addresses: &sources.ram_addresses,
            bytecode_indices: &sources.bytecode_indices,
            untrusted_advice,
            trusted_advice,
        }
    }
}


struct AddressMajorOneHotPolynomial {
    trace_len: usize,
    chunk_domain: usize,
    indices: Vec<Option<u8>>,
    num_vars: usize,
}

impl AddressMajorOneHotPolynomial {
    fn new(
        trace_len: usize,
        chunk_domain: usize,
        indices: Vec<Option<u8>>,
        num_vars: usize,
    ) -> Result<Self, CommitmentPhaseError> {
        let active_len = trace_len
            .checked_mul(chunk_domain)
            .ok_or(CommitmentPhaseError::TargetSizeOverflow { num_vars })?;
        let target_len = target_len(num_vars)?;
        if active_len > target_len {
            return Err(CommitmentPhaseError::OracleTooLarge {
                oracle: "one_hot",
                len: active_len,
                target_len,
            });
        }
        Ok(Self {
            trace_len,
            chunk_domain,
            indices,
            num_vars,
        })
    }

    fn nonzero_flat_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices
            .iter()
            .enumerate()
            .filter_map(|(cycle, &index)| {
                index.map(|index| {
                    let index = index as usize;
                    assert!(
                        index < self.chunk_domain,
                        "one-hot index {index} exceeds domain {}",
                        self.chunk_domain
                    );
                    index * self.trace_len + cycle
                })
            })
    }
}

impl MultilinearPoly<Fr> for AddressMajorOneHotPolynomial {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        assert_eq!(point.len(), self.num_vars);
        let eq_evals = EqPolynomial::new(point.to_vec()).evaluations();
        self.nonzero_flat_indices()
            .fold(Fr::from_u64(0), |acc, flat| acc + eq_evals[flat])
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[Fr])) {
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << (self.num_vars - sigma);
        let mut entries = Vec::with_capacity(self.indices.len());
        for flat in self.nonzero_flat_indices() {
            entries.push((flat / num_cols, flat % num_cols));
        }
        entries.sort_unstable_by_key(|(row, _)| *row);

        let mut cursor = 0;
        let mut row = vec![Fr::from_u64(0); num_cols];
        for row_index in 0..num_rows {
            row.fill(Fr::from_u64(0));
            while cursor < entries.len() && entries[cursor].0 == row_index {
                row[entries[cursor].1] = Fr::from_u64(1);
                cursor += 1;
            }
            f(row_index, &row);
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << (self.num_vars - sigma);
        assert_eq!(left.len(), num_rows);
        let mut result = vec![Fr::from_u64(0); num_cols];
        for flat in self.nonzero_flat_indices() {
            result[flat % num_cols] += left[flat / num_cols];
        }
        result
    }

    fn is_sparse(&self) -> bool {
        true
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(usize, Fr)) {
        for flat in self.nonzero_flat_indices() {
            f(flat, Fr::from_u64(1));
        }
    }
}

pub struct SparseCommitmentInputs<'a> {
    pub inputs: CommitmentOracleInputs<'a>,
    cache: std::collections::BTreeMap<(&'static str, usize), Option<Vec<Fr>>>,
    chunk_counts: OneHotChunkCounts,
}

impl<'a> SparseCommitmentInputs<'a> {
    pub fn new(inputs: CommitmentOracleInputs<'a>) -> Self {
        Self {
            inputs,
            cache: std::collections::BTreeMap::new(),
            chunk_counts: OneHotChunkCounts::default(),
        }
    }

    fn update_chunk_counts(&mut self, program: &CommitmentProverProgramPlan) {
        let mut counts = OneHotChunkCounts::default();
        let mut instruction = 0;
        let mut ram = 0;
        let mut bytecode = 0;
        for plan in program.oracle_plans {
            if plan.oracle.strip_prefix("InstructionRa_").is_some() {
                instruction += 1;
            } else if plan.oracle.strip_prefix("RamRa_").is_some() {
                ram += 1;
            } else if plan.oracle.strip_prefix("BytecodeRa_").is_some() {
                bytecode += 1;
            }
        }
        if instruction > 0 {
            counts.instruction = instruction;
        }
        if ram > 0 {
            counts.ram = ram;
        }
        if bytecode > 0 {
            counts.bytecode = bytecode;
        }
        self.chunk_counts = counts;
    }

    fn one_hot_spec(&self, oracle: &'static str) -> Option<OneHotSpec> {
        let (prefix, num_chunks, values, padding) =
            if let Some(suffix) = oracle.strip_prefix("InstructionRa_") {
                (
                    suffix,
                    self.chunk_counts.instruction,
                    OneHotSource::InstructionKeys,
                    Some(0),
                )
            } else if let Some(suffix) = oracle.strip_prefix("RamRa_") {
                (
                    suffix,
                    self.chunk_counts.ram,
                    OneHotSource::RamAddresses,
                    None,
                )
            } else if let Some(suffix) = oracle.strip_prefix("BytecodeRa_") {
                (
                    suffix,
                    self.chunk_counts.bytecode,
                    OneHotSource::BytecodeIndices,
                    Some(0),
                )
            } else {
                return None;
            };
        let chunk = prefix.parse::<usize>().ok()?;
        if chunk >= num_chunks {
            return None;
        }
        Some(OneHotSpec {
            source: values,
            chunk,
            num_chunks,
            chunk_bits: 4,
            padding,
        })
    }

    fn source_values(&self, source: OneHotSource) -> &'a [Option<u128>] {
        match source {
            OneHotSource::InstructionKeys => self.inputs.instruction_keys,
            OneHotSource::RamAddresses => self.inputs.ram_addresses,
            OneHotSource::BytecodeIndices => self.inputs.bytecode_indices,
        }
    }

    fn one_hot_indices(
        &self,
        oracle: &'static str,
        trace_len: usize,
    ) -> Option<Vec<Option<u8>>> {
        let spec = self.one_hot_spec(oracle)?;
        let values = self.source_values(spec.source);
        Some(one_hot_chunk_indices(
            values,
            spec.chunk,
            spec.num_chunks,
            spec.chunk_bits,
            trace_len,
            spec.padding,
        ))
    }

    fn add_one_hot_batch(
        &self,
        requests: &[JointContribution],
        joint: &mut [Fr],
        handled: &mut [bool],
        source: OneHotSource,
    ) {
        let mut entries = Vec::new();
        for (request_index, request) in requests.iter().enumerate() {
            if handled[request_index] {
                continue;
            }
            let Some(spec) = self.one_hot_spec(request.oracle) else {
                continue;
            };
            if spec.source != source {
                continue;
            }
            let Some(trace_num_vars) = request.num_vars.checked_sub(spec.chunk_bits) else {
                continue;
            };
            let Ok(trace_len) = target_len(trace_num_vars) else {
                continue;
            };
            let Ok(chunk_domain) = target_len(spec.chunk_bits) else {
                continue;
            };
            let Some(active_len) = trace_len.checked_mul(chunk_domain) else {
                continue;
            };
            let values = self.source_values(source);
            if values.len() > trace_len || spec.chunk_bits > u8::BITS as usize {
                continue;
            }
            let shift = spec.chunk_bits * (spec.num_chunks - 1 - spec.chunk);
            if shift >= u128::BITS as usize || spec.chunk_bits * spec.num_chunks > u128::BITS as usize
            {
                continue;
            }
            entries.push((
                shift,
                (chunk_domain - 1) as u128,
                trace_len,
                request.limit.min(joint.len()).min(active_len),
                spec.padding,
                request.scalar,
            ));
            handled[request_index] = true;
        }
        if entries.is_empty() {
            return;
        }

        let values = self.source_values(source);
        let max_trace_len = entries
            .iter()
            .map(|(_, _, trace_len, _, _, _)| *trace_len)
            .max()
            .unwrap_or(0);
        for cycle in 0..max_trace_len {
            for (shift, mask, trace_len, max_flat, padding, scalar) in &entries {
                if cycle >= *trace_len {
                    continue;
                }
                let value = values.get(cycle).copied().flatten().or(*padding);
                let Some(value) = value else {
                    continue;
                };
                let flat = (((value >> *shift) & *mask) as usize) * *trace_len + cycle;
                if flat < *max_flat {
                    joint[flat] += *scalar;
                }
            }
        }
    }

    #[expect(
        clippy::option_option,
        reason = "distinguishes missing oracle from present optional oracle"
    )]
    fn materialize_oracle(
        &self,
        oracle: &'static str,
        num_vars: usize,
    ) -> Option<Option<Vec<Fr>>> {
        let materialized = match oracle {
            "RdInc" => Some(dense_i128_column_to_field(
                self.inputs.rd_inc,
                target_len(num_vars).ok()?,
            )),
            "RamInc" => Some(dense_i128_column_to_field(
                self.inputs.ram_inc,
                target_len(num_vars).ok()?,
            )),
            "UntrustedAdvice" => optional_field_oracle(
                self.inputs.untrusted_advice,
                target_len(num_vars).ok()?,
            ),
            "TrustedAdvice" => {
                optional_field_oracle(self.inputs.trusted_advice, target_len(num_vars).ok()?)
            }
            _ => {
                let spec = self.one_hot_spec(oracle)?;
                let trace_len = target_len(num_vars.checked_sub(spec.chunk_bits)?).ok()?;
                let values = self.source_values(spec.source);
                Some(one_hot_chunk_address_major(
                    values,
                    spec.chunk,
                    spec.num_chunks,
                    spec.chunk_bits,
                    trace_len,
                    spec.padding,
                ))
            }
        };
        Some(materialized)
    }

    fn commit_oracle(
        &self,
        program: &CommitmentProverProgramPlan,
        oracle: &'static str,
        layout_num_vars: usize,
        prover_setup: &DoryProverSetup,
    ) -> Result<(DoryCommitment, DoryHint), CommitmentPhaseError> {
        let oracle_num_vars = oracle_num_vars(program, oracle, layout_num_vars);
        if let Some(spec) = self.one_hot_spec(oracle) {
            let trace_len = target_len(oracle_num_vars - spec.chunk_bits)?;
            let chunk_domain = target_len(spec.chunk_bits)?;
            let indices = self
                .one_hot_indices(oracle, trace_len)
                .ok_or(CommitmentPhaseError::MissingOracle { oracle })?;
            let poly = AddressMajorOneHotPolynomial::new(
                trace_len,
                chunk_domain,
                indices,
                layout_num_vars,
            )?;
            let _dory_commit_span = tracing::info_span!("bolt.commitment.dory_commit").entered();
            Ok(DoryScheme::commit(&poly, prover_setup))
        } else {
            let data = self
                .materialize_oracle(oracle, oracle_num_vars)
                .flatten()
                .ok_or(CommitmentPhaseError::MissingOracle { oracle })?;
            let data = into_padded_oracle(oracle, oracle_num_vars, Cow::Owned(data))?;
            commit_with_layout(&data, layout_num_vars, prover_setup)
        }
    }
}

impl CommitmentInputProvider for SparseCommitmentInputs<'_> {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        let num_vars = match oracle {
            "RdInc" | "RamInc" | "UntrustedAdvice" | "TrustedAdvice" => 16,
            _ if self.one_hot_spec(oracle).is_some() => 20,
            _ => return None,
        };
        self.materialize_with_num_vars(oracle, num_vars)
    }

    fn materialize_with_num_vars(
        &mut self,
        oracle: &'static str,
        num_vars: usize,
    ) -> Option<Cow<'_, [Fr]>> {
        if !self.cache.contains_key(&(oracle, num_vars)) {
            let materialized = self.materialize_oracle(oracle, num_vars).flatten();
            let _ = self.cache.insert((oracle, num_vars), materialized);
        }
        self.cache
            .get(&(oracle, num_vars))
            .and_then(|values| values.as_ref())
            .map(|values| Cow::Borrowed(values.as_slice()))
    }

    fn commit_batch(
        &mut self,
        program: &CommitmentProverProgramPlan,
        plan: &CommitmentBatchPlan,
        prover_setup: &DoryProverSetup,
    ) -> Option<Result<Vec<CommittedOracle>, CommitmentPhaseError>> {
        self.update_chunk_counts(program);
        Some(
            plan.oracles
                .par_iter()
                .map(|&oracle| {
                    let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
                    let (commitment, hint) =
                        self.commit_oracle(program, oracle, plan.num_vars, prover_setup)?;
                    Ok(CommittedOracle {
                        commitment: Some(commitment),
                        record: CommitmentRecord {
                            artifact: plan.artifact,
                            oracle,
                            label: plan.label,
                            num_vars: oracle_num_vars,
                        },
                        hint: Some(OracleOpeningHint { oracle, hint }),
                    })
                })
                .collect(),
        )
    }

    fn add_scaled_to_joint(
        &mut self,
        oracle: &'static str,
        joint: &mut [Fr],
        num_vars: usize,
        limit: usize,
        scalar: Fr,
    ) -> bool {
        let dense = match oracle {
            "RdInc" => Some(self.inputs.rd_inc),
            "RamInc" => Some(self.inputs.ram_inc),
            _ => None,
        };
        if let Some(values) = dense {
            let Ok(target_len) = target_len(num_vars) else {
                return false;
            };
            let len = limit.min(joint.len()).min(values.len()).min(target_len);
            for (dst, &value) in joint.iter_mut().take(len).zip(values.iter()) {
                if value != 0 {
                    *dst += Fr::from_i128(value) * scalar;
                }
            }
            return true;
        }

        let Some(spec) = self.one_hot_spec(oracle) else {
            return false;
        };
        let Some(trace_num_vars) = num_vars.checked_sub(spec.chunk_bits) else {
            return false;
        };
        let Ok(trace_len) = target_len(trace_num_vars) else {
            return false;
        };
        let Ok(chunk_domain) = target_len(spec.chunk_bits) else {
            return false;
        };
        let Some(active_len) = trace_len.checked_mul(chunk_domain) else {
            return false;
        };
        let max_flat = limit.min(joint.len()).min(active_len);
        let Some(indices) = self.one_hot_indices(oracle, trace_len) else {
            return false;
        };
        for (cycle, index) in indices.into_iter().enumerate() {
            let Some(index) = index else {
                continue;
            };
            let flat = index as usize * trace_len + cycle;
            if flat < max_flat {
                joint[flat] += scalar;
            }
        }
        true
    }

    fn add_scaled_to_joint_batch(
        &mut self,
        requests: &[JointContribution],
        joint: &mut [Fr],
    ) -> Vec<bool> {
        let mut handled = vec![false; requests.len()];
        for (index, request) in requests.iter().enumerate() {
            let dense = match request.oracle {
                "RdInc" => Some(self.inputs.rd_inc),
                "RamInc" => Some(self.inputs.ram_inc),
                _ => None,
            };
            if let Some(values) = dense {
                let Ok(target_len) = target_len(request.num_vars) else {
                    continue;
                };
                let len = request.limit.min(joint.len()).min(values.len()).min(target_len);
                for (dst, &value) in joint.iter_mut().take(len).zip(values.iter()) {
                    if value != 0 {
                        *dst += Fr::from_i128(value) * request.scalar;
                    }
                }
                handled[index] = true;
            }
        }

        self.add_one_hot_batch(requests, joint, &mut handled, OneHotSource::InstructionKeys);
        self.add_one_hot_batch(requests, joint, &mut handled, OneHotSource::RamAddresses);
        self.add_one_hot_batch(requests, joint, &mut handled, OneHotSource::BytecodeIndices);
        handled
    }

    fn open_joint_polynomial<T>(
        &mut self,
        requests: &[JointContribution],
        opening_point: &[Fr],
        joint_claim: Fr,
        prover_setup: &DoryProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> Result<jolt_dory::DoryProof, DoryHint>
    where
        T: Transcript<Challenge = Fr>,
    {
        let poly = SparseJointPolynomial {
            inputs: self.inputs,
            chunk_counts: self.chunk_counts,
            requests: requests.to_vec(),
            num_vars: opening_point.len(),
            claimed_eval: joint_claim,
        };
        Ok(DoryScheme::open_poly(
            &poly,
            opening_point,
            joint_claim,
            prover_setup,
            Some(hint),
            transcript,
        ))
    }
}

struct SparseJointPolynomial<'a> {
    inputs: CommitmentOracleInputs<'a>,
    chunk_counts: OneHotChunkCounts,
    requests: Vec<JointContribution>,
    num_vars: usize,
    claimed_eval: Fr,
}

impl SparseJointPolynomial<'_> {
    fn one_hot_spec(&self, oracle: &'static str) -> Option<OneHotSpec> {
        let (prefix, num_chunks, values, padding) =
            if let Some(suffix) = oracle.strip_prefix("InstructionRa_") {
                (
                    suffix,
                    self.chunk_counts.instruction,
                    OneHotSource::InstructionKeys,
                    Some(0),
                )
            } else if let Some(suffix) = oracle.strip_prefix("RamRa_") {
                (
                    suffix,
                    self.chunk_counts.ram,
                    OneHotSource::RamAddresses,
                    None,
                )
            } else if let Some(suffix) = oracle.strip_prefix("BytecodeRa_") {
                (
                    suffix,
                    self.chunk_counts.bytecode,
                    OneHotSource::BytecodeIndices,
                    Some(0),
                )
            } else {
                return None;
            };
        let chunk = prefix.parse::<usize>().ok()?;
        if chunk >= num_chunks {
            return None;
        }
        Some(OneHotSpec {
            source: values,
            chunk,
            num_chunks,
            chunk_bits: 4,
            padding,
        })
    }

    fn source_values(&self, source: OneHotSource) -> &[Option<u128>] {
        match source {
            OneHotSource::InstructionKeys => self.inputs.instruction_keys,
            OneHotSource::RamAddresses => self.inputs.ram_addresses,
            OneHotSource::BytecodeIndices => self.inputs.bytecode_indices,
        }
    }

    fn dense_values(&self, oracle: &'static str) -> Option<&[i128]> {
        match oracle {
            "RdInc" => Some(self.inputs.rd_inc),
            "RamInc" => Some(self.inputs.ram_inc),
            _ => None,
        }
    }

    fn fold_rows_slow(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << self.num_vars.saturating_sub(sigma);
        let mut result_accs = vec![<Fr as Field>::Accumulator::default(); num_cols];
        for request in &self.requests {
            if let Some(values) = self.dense_values(request.oracle) {
                let len = request.limit.min(values.len()).min(num_rows * num_cols);
                for (flat, value) in values.iter().copied().enumerate().take(len) {
                    if value == 0 {
                        continue;
                    }
                    let row = flat / num_cols;
                    let col = flat % num_cols;
                    result_accs[col].fmadd(left[row] * request.scalar, Fr::from_i128(value));
                }
                continue;
            }
            let Some(spec) = self.one_hot_spec(request.oracle) else {
                continue;
            };
            let Some(trace_num_vars) = request.num_vars.checked_sub(spec.chunk_bits) else {
                continue;
            };
            let Ok(trace_len) = target_len(trace_num_vars) else {
                continue;
            };
            let Ok(chunk_domain) = target_len(spec.chunk_bits) else {
                continue;
            };
            let Some(active_len) = trace_len.checked_mul(chunk_domain) else {
                continue;
            };
            let max_flat = request.limit.min(active_len).min(num_rows * num_cols);
            let values = self.source_values(spec.source);
            let shift = spec.chunk_bits * (spec.num_chunks - 1 - spec.chunk);
            let mask = (chunk_domain - 1) as u128;
            for (cycle, value) in values.iter().take(trace_len).enumerate() {
                let value = (*value).or(spec.padding);
                let Some(value) = value else {
                    continue;
                };
                let flat = (((value >> shift) & mask) as usize) * trace_len + cycle;
                if flat >= max_flat {
                    continue;
                }
                let row = flat / num_cols;
                let col = flat % num_cols;
                result_accs[col].fmadd(left[row], request.scalar);
            }
        }
        result_accs.into_iter().map(FieldAccumulator::reduce).collect()
    }
}

impl MultilinearPoly<Fr> for SparseJointPolynomial<'_> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, _point: &[Fr]) -> Fr {
        self.claimed_eval
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[Fr])) {
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << self.num_vars.saturating_sub(sigma);
        let mut row = vec![Fr::from_u64(0); num_cols];
        for row_index in 0..num_rows {
            row.fill(Fr::from_u64(0));
            let base = row_index * num_cols;
            for request in &self.requests {
                if let Some(values) = self.dense_values(request.oracle) {
                    let end = request.limit.min(values.len()).min(base + num_cols);
                    for flat in base..end {
                        let value = values[flat];
                        if value != 0 {
                            row[flat - base] += Fr::from_i128(value) * request.scalar;
                        }
                    }
                    continue;
                }
                let Some(spec) = self.one_hot_spec(request.oracle) else {
                    continue;
                };
                let Some(trace_num_vars) = request.num_vars.checked_sub(spec.chunk_bits) else {
                    continue;
                };
                let Ok(trace_len) = target_len(trace_num_vars) else {
                    continue;
                };
                let Ok(chunk_domain) = target_len(spec.chunk_bits) else {
                    continue;
                };
                let Some(active_len) = trace_len.checked_mul(chunk_domain) else {
                    continue;
                };
                let max_flat = request.limit.min(active_len);
                let values = self.source_values(spec.source);
                let shift = spec.chunk_bits * (spec.num_chunks - 1 - spec.chunk);
                let mask = (chunk_domain - 1) as u128;
                for (cycle, value) in values.iter().take(trace_len).enumerate() {
                    let value = (*value).or(spec.padding);
                    let Some(value) = value else {
                        continue;
                    };
                    let flat = (((value >> shift) & mask) as usize) * trace_len + cycle;
                    if flat >= max_flat || flat < base || flat >= base + num_cols {
                        continue;
                    }
                    row[flat - base] += request.scalar;
                }
            }
            f(row_index, &row);
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        struct DenseEntry<'a> {
            values: &'a [i128],
            limit: usize,
            scalar: Fr,
        }

        struct OneHotEntry {
            shift: usize,
            mask: u128,
            table: Vec<Fr>,
        }

        struct OneHotGroup<'a> {
            values: &'a [Option<u128>],
            padding: Option<u128>,
            entries: Vec<OneHotEntry>,
            cache: std::collections::HashMap<u128, Fr>,
        }

        impl<'a> OneHotGroup<'a> {
            fn new(values: &'a [Option<u128>], padding: Option<u128>) -> Self {
                Self {
                    values,
                    padding,
                    entries: Vec::new(),
                    cache: std::collections::HashMap::new(),
                }
            }

            fn contribution(&mut self, value: u128) -> Fr {
                if self.entries.len() <= 16 {
                    let mut contribution = Fr::from_u64(0);
                    for entry in &self.entries {
                        let k = ((value >> entry.shift) & entry.mask) as usize;
                        if k < entry.table.len() {
                            contribution += entry.table[k];
                        }
                    }
                    return contribution;
                }
                if let Some(contribution) = self.cache.get(&value) {
                    return *contribution;
                }
                let mut contribution = Fr::from_u64(0);
                for entry in &self.entries {
                    let k = ((value >> entry.shift) & entry.mask) as usize;
                    if k < entry.table.len() {
                        contribution += entry.table[k];
                    }
                }
                let _ = self.cache.insert(value, contribution);
                contribution
            }
        }

        let _span = tracing::info_span!("SparseJointPolynomial::fold_rows").entered();
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << self.num_vars.saturating_sub(sigma);
        let mut trace_len = None;
        let mut dense_entries = Vec::new();
        let mut one_hot_specs = Vec::new();
        for request in &self.requests {
            if let Some(values) = self.dense_values(request.oracle) {
                if trace_len
                    .replace(request.limit.min(values.len()))
                    .is_some_and(|previous| previous != request.limit.min(values.len()))
                {
                    return self.fold_rows_slow(left, sigma);
                }
                dense_entries.push(DenseEntry {
                    values,
                    limit: request.limit,
                    scalar: request.scalar,
                });
                continue;
            }
            let Some(spec) = self.one_hot_spec(request.oracle) else {
                continue;
            };
            let Some(trace_num_vars) = request.num_vars.checked_sub(spec.chunk_bits) else {
                continue;
            };
            let Ok(len) = target_len(trace_num_vars) else {
                continue;
            };
            if trace_len
                .replace(len)
                .is_some_and(|previous| previous != len)
            {
                return self.fold_rows_slow(left, sigma);
            }
            one_hot_specs.push((request, spec));
        }
        let Some(trace_len) = trace_len else {
            return vec![Fr::from_u64(0); num_cols];
        };
        if trace_len == 0 || trace_len % num_cols != 0 {
            return self.fold_rows_slow(left, sigma);
        }
        let rows_per_k = trace_len / num_cols;
        if rows_per_k == 0 || !num_rows.is_multiple_of(rows_per_k) {
            return self.fold_rows_slow(left, sigma);
        }
        let k_domain = num_rows / rows_per_k;

        let mut row_factors = vec![Fr::from_u64(0); rows_per_k];
        let mut eq_k = vec![Fr::from_u64(0); k_domain];
        for (k, eq) in eq_k.iter_mut().enumerate() {
            let base = k * rows_per_k;
            for row in 0..rows_per_k {
                let value = left[base + row];
                row_factors[row] += value;
                *eq += value;
            }
        }

        let mut instruction =
            OneHotGroup::new(self.source_values(OneHotSource::InstructionKeys), Some(0));
        let mut bytecode =
            OneHotGroup::new(self.source_values(OneHotSource::BytecodeIndices), Some(0));
        let mut ram = OneHotGroup::new(self.source_values(OneHotSource::RamAddresses), None);
        for (request, spec) in one_hot_specs {
            let Ok(chunk_domain) = target_len(spec.chunk_bits) else {
                continue;
            };
            let shift = spec.chunk_bits * (spec.num_chunks - 1 - spec.chunk);
            if shift >= u128::BITS as usize
                || spec.chunk_bits * spec.num_chunks > u128::BITS as usize
            {
                continue;
            }
            let mut table = vec![Fr::from_u64(0); k_domain];
            for k in 0..k_domain.min(chunk_domain) {
                table[k] = request.scalar * eq_k[k];
            }
            let entry = OneHotEntry {
                shift,
                mask: (chunk_domain - 1) as u128,
                table,
            };
            match spec.source {
                OneHotSource::InstructionKeys => instruction.entries.push(entry),
                OneHotSource::RamAddresses => ram.entries.push(entry),
                OneHotSource::BytecodeIndices => bytecode.entries.push(entry),
            }
        }

        let mut result_accs = vec![<Fr as Field>::Accumulator::default(); num_cols];
        for cycle in 0..trace_len {
            let row = cycle / num_cols;
            let col = cycle % num_cols;
            let dense_weight = left[row];
            for entry in &dense_entries {
                if cycle >= entry.limit || cycle >= entry.values.len() {
                    continue;
                }
                let value = entry.values[cycle];
                if value != 0 {
                    result_accs[col].fmadd(dense_weight * entry.scalar, Fr::from_i128(value));
                }
            }

            let row_factor = row_factors[row];
            if row_factor == Fr::from_u64(0) {
                continue;
            }
            let mut inner = Fr::from_u64(0);
            if !instruction.entries.is_empty() {
                if let Some(value) = instruction
                    .values
                    .get(cycle)
                    .copied()
                    .flatten()
                    .or(instruction.padding)
                {
                    inner += instruction.contribution(value);
                }
            }
            if !bytecode.entries.is_empty() {
                if let Some(value) = bytecode
                    .values
                    .get(cycle)
                    .copied()
                    .flatten()
                    .or(bytecode.padding)
                {
                    inner += bytecode.contribution(value);
                }
            }
            if !ram.entries.is_empty() {
                if let Some(value) = ram.values.get(cycle).copied().flatten().or(ram.padding) {
                    inner += ram.contribution(value);
                }
            }
            if inner != Fr::from_u64(0) {
                result_accs[col].fmadd(row_factor, inner);
            }
        }
        result_accs.into_iter().map(FieldAccumulator::reduce).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OneHotSource {
    InstructionKeys,
    RamAddresses,
    BytecodeIndices,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OneHotSpec {
    source: OneHotSource,
    chunk: usize,
    num_chunks: usize,
    chunk_bits: usize,
    padding: Option<u128>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OneHotChunkCounts {
    instruction: usize,
    ram: usize,
    bytecode: usize,
}

impl Default for OneHotChunkCounts {
    fn default() -> Self {
        Self {
            instruction: 32,
            ram: 4,
            bytecode: 3,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentOracles {
    pub rd_inc: Vec<Fr>,
    pub ram_inc: Vec<Fr>,
    pub instruction_ra_0: Vec<Fr>,
    pub instruction_ra_1: Vec<Fr>,
    pub instruction_ra_2: Vec<Fr>,
    pub instruction_ra_3: Vec<Fr>,
    pub instruction_ra_4: Vec<Fr>,
    pub instruction_ra_5: Vec<Fr>,
    pub instruction_ra_6: Vec<Fr>,
    pub instruction_ra_7: Vec<Fr>,
    pub instruction_ra_8: Vec<Fr>,
    pub instruction_ra_9: Vec<Fr>,
    pub instruction_ra_10: Vec<Fr>,
    pub instruction_ra_11: Vec<Fr>,
    pub instruction_ra_12: Vec<Fr>,
    pub instruction_ra_13: Vec<Fr>,
    pub instruction_ra_14: Vec<Fr>,
    pub instruction_ra_15: Vec<Fr>,
    pub instruction_ra_16: Vec<Fr>,
    pub instruction_ra_17: Vec<Fr>,
    pub instruction_ra_18: Vec<Fr>,
    pub instruction_ra_19: Vec<Fr>,
    pub instruction_ra_20: Vec<Fr>,
    pub instruction_ra_21: Vec<Fr>,
    pub instruction_ra_22: Vec<Fr>,
    pub instruction_ra_23: Vec<Fr>,
    pub instruction_ra_24: Vec<Fr>,
    pub instruction_ra_25: Vec<Fr>,
    pub instruction_ra_26: Vec<Fr>,
    pub instruction_ra_27: Vec<Fr>,
    pub instruction_ra_28: Vec<Fr>,
    pub instruction_ra_29: Vec<Fr>,
    pub instruction_ra_30: Vec<Fr>,
    pub instruction_ra_31: Vec<Fr>,
    pub ram_ra_0: Vec<Fr>,
    pub ram_ra_1: Vec<Fr>,
    pub ram_ra_2: Vec<Fr>,
    pub ram_ra_3: Vec<Fr>,
    pub bytecode_ra_0: Vec<Fr>,
    pub bytecode_ra_1: Vec<Fr>,
    pub bytecode_ra_2: Vec<Fr>,
    pub untrusted_advice: Option<Vec<Fr>>,
    pub trusted_advice: Option<Vec<Fr>>,
}

impl CommitmentInputProvider for CommitmentOracles {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        match oracle {
            "RdInc" => Some(Cow::Borrowed(&self.rd_inc)),
            "RamInc" => Some(Cow::Borrowed(&self.ram_inc)),
            "InstructionRa_0" => Some(Cow::Borrowed(&self.instruction_ra_0)),
            "InstructionRa_1" => Some(Cow::Borrowed(&self.instruction_ra_1)),
            "InstructionRa_2" => Some(Cow::Borrowed(&self.instruction_ra_2)),
            "InstructionRa_3" => Some(Cow::Borrowed(&self.instruction_ra_3)),
            "InstructionRa_4" => Some(Cow::Borrowed(&self.instruction_ra_4)),
            "InstructionRa_5" => Some(Cow::Borrowed(&self.instruction_ra_5)),
            "InstructionRa_6" => Some(Cow::Borrowed(&self.instruction_ra_6)),
            "InstructionRa_7" => Some(Cow::Borrowed(&self.instruction_ra_7)),
            "InstructionRa_8" => Some(Cow::Borrowed(&self.instruction_ra_8)),
            "InstructionRa_9" => Some(Cow::Borrowed(&self.instruction_ra_9)),
            "InstructionRa_10" => Some(Cow::Borrowed(&self.instruction_ra_10)),
            "InstructionRa_11" => Some(Cow::Borrowed(&self.instruction_ra_11)),
            "InstructionRa_12" => Some(Cow::Borrowed(&self.instruction_ra_12)),
            "InstructionRa_13" => Some(Cow::Borrowed(&self.instruction_ra_13)),
            "InstructionRa_14" => Some(Cow::Borrowed(&self.instruction_ra_14)),
            "InstructionRa_15" => Some(Cow::Borrowed(&self.instruction_ra_15)),
            "InstructionRa_16" => Some(Cow::Borrowed(&self.instruction_ra_16)),
            "InstructionRa_17" => Some(Cow::Borrowed(&self.instruction_ra_17)),
            "InstructionRa_18" => Some(Cow::Borrowed(&self.instruction_ra_18)),
            "InstructionRa_19" => Some(Cow::Borrowed(&self.instruction_ra_19)),
            "InstructionRa_20" => Some(Cow::Borrowed(&self.instruction_ra_20)),
            "InstructionRa_21" => Some(Cow::Borrowed(&self.instruction_ra_21)),
            "InstructionRa_22" => Some(Cow::Borrowed(&self.instruction_ra_22)),
            "InstructionRa_23" => Some(Cow::Borrowed(&self.instruction_ra_23)),
            "InstructionRa_24" => Some(Cow::Borrowed(&self.instruction_ra_24)),
            "InstructionRa_25" => Some(Cow::Borrowed(&self.instruction_ra_25)),
            "InstructionRa_26" => Some(Cow::Borrowed(&self.instruction_ra_26)),
            "InstructionRa_27" => Some(Cow::Borrowed(&self.instruction_ra_27)),
            "InstructionRa_28" => Some(Cow::Borrowed(&self.instruction_ra_28)),
            "InstructionRa_29" => Some(Cow::Borrowed(&self.instruction_ra_29)),
            "InstructionRa_30" => Some(Cow::Borrowed(&self.instruction_ra_30)),
            "InstructionRa_31" => Some(Cow::Borrowed(&self.instruction_ra_31)),
            "RamRa_0" => Some(Cow::Borrowed(&self.ram_ra_0)),
            "RamRa_1" => Some(Cow::Borrowed(&self.ram_ra_1)),
            "RamRa_2" => Some(Cow::Borrowed(&self.ram_ra_2)),
            "RamRa_3" => Some(Cow::Borrowed(&self.ram_ra_3)),
            "BytecodeRa_0" => Some(Cow::Borrowed(&self.bytecode_ra_0)),
            "BytecodeRa_1" => Some(Cow::Borrowed(&self.bytecode_ra_1)),
            "BytecodeRa_2" => Some(Cow::Borrowed(&self.bytecode_ra_2)),
            "UntrustedAdvice" => self.untrusted_advice.as_deref().map(Cow::Borrowed),
            "TrustedAdvice" => self.trusted_advice.as_deref().map(Cow::Borrowed),
            _ => None,
        }
    }
}

pub fn build_commitment_oracles(
    inputs: &CommitmentOracleInputs<'_>,
) -> Result<CommitmentOracles, CommitmentPhaseError> {
    Ok(CommitmentOracles {
        rd_inc: dense_i128_column_to_field(inputs.rd_inc, target_len(16)?),
        ram_inc: dense_i128_column_to_field(inputs.ram_inc, target_len(16)?),
        instruction_ra_0: one_hot_chunk_address_major(inputs.instruction_keys, 0, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_1: one_hot_chunk_address_major(inputs.instruction_keys, 1, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_2: one_hot_chunk_address_major(inputs.instruction_keys, 2, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_3: one_hot_chunk_address_major(inputs.instruction_keys, 3, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_4: one_hot_chunk_address_major(inputs.instruction_keys, 4, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_5: one_hot_chunk_address_major(inputs.instruction_keys, 5, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_6: one_hot_chunk_address_major(inputs.instruction_keys, 6, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_7: one_hot_chunk_address_major(inputs.instruction_keys, 7, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_8: one_hot_chunk_address_major(inputs.instruction_keys, 8, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_9: one_hot_chunk_address_major(inputs.instruction_keys, 9, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_10: one_hot_chunk_address_major(inputs.instruction_keys, 10, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_11: one_hot_chunk_address_major(inputs.instruction_keys, 11, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_12: one_hot_chunk_address_major(inputs.instruction_keys, 12, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_13: one_hot_chunk_address_major(inputs.instruction_keys, 13, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_14: one_hot_chunk_address_major(inputs.instruction_keys, 14, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_15: one_hot_chunk_address_major(inputs.instruction_keys, 15, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_16: one_hot_chunk_address_major(inputs.instruction_keys, 16, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_17: one_hot_chunk_address_major(inputs.instruction_keys, 17, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_18: one_hot_chunk_address_major(inputs.instruction_keys, 18, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_19: one_hot_chunk_address_major(inputs.instruction_keys, 19, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_20: one_hot_chunk_address_major(inputs.instruction_keys, 20, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_21: one_hot_chunk_address_major(inputs.instruction_keys, 21, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_22: one_hot_chunk_address_major(inputs.instruction_keys, 22, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_23: one_hot_chunk_address_major(inputs.instruction_keys, 23, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_24: one_hot_chunk_address_major(inputs.instruction_keys, 24, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_25: one_hot_chunk_address_major(inputs.instruction_keys, 25, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_26: one_hot_chunk_address_major(inputs.instruction_keys, 26, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_27: one_hot_chunk_address_major(inputs.instruction_keys, 27, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_28: one_hot_chunk_address_major(inputs.instruction_keys, 28, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_29: one_hot_chunk_address_major(inputs.instruction_keys, 29, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_30: one_hot_chunk_address_major(inputs.instruction_keys, 30, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_31: one_hot_chunk_address_major(inputs.instruction_keys, 31, 32, 4, target_len(16)?, Some(0)),
        ram_ra_0: one_hot_chunk_address_major(inputs.ram_addresses, 0, 4, 4, target_len(16)?, None),
        ram_ra_1: one_hot_chunk_address_major(inputs.ram_addresses, 1, 4, 4, target_len(16)?, None),
        ram_ra_2: one_hot_chunk_address_major(inputs.ram_addresses, 2, 4, 4, target_len(16)?, None),
        ram_ra_3: one_hot_chunk_address_major(inputs.ram_addresses, 3, 4, 4, target_len(16)?, None),
        bytecode_ra_0: one_hot_chunk_address_major(inputs.bytecode_indices, 0, 3, 4, target_len(16)?, Some(0)),
        bytecode_ra_1: one_hot_chunk_address_major(inputs.bytecode_indices, 1, 3, 4, target_len(16)?, Some(0)),
        bytecode_ra_2: one_hot_chunk_address_major(inputs.bytecode_indices, 2, 3, 4, target_len(16)?, Some(0)),
        untrusted_advice: optional_field_oracle(inputs.untrusted_advice, target_len(16)?),
        trusted_advice: optional_field_oracle(inputs.trusted_advice, target_len(16)?),
    })
}

pub const COMMITMENT_PARAMS: CommitmentParams = CommitmentParams {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const ORACLE_PLANS: &[OraclePlan] = &[
    OraclePlan { oracle: "RdInc", domain: "jolt.trace_domain", num_vars: 16 },
    OraclePlan { oracle: "RamInc", domain: "jolt.trace_domain", num_vars: 16 },
    OraclePlan { oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "UntrustedAdvice", domain: "jolt.trace_domain", num_vars: 16 },
    OraclePlan { oracle: "TrustedAdvice", domain: "jolt.trace_domain", num_vars: 16 },
];
pub const COMMITMENT_BATCH_0_ORACLES: &[&str] = &[
    "RdInc",
    "RamInc",
    "InstructionRa_0",
    "InstructionRa_1",
    "InstructionRa_2",
    "InstructionRa_3",
    "InstructionRa_4",
    "InstructionRa_5",
    "InstructionRa_6",
    "InstructionRa_7",
    "InstructionRa_8",
    "InstructionRa_9",
    "InstructionRa_10",
    "InstructionRa_11",
    "InstructionRa_12",
    "InstructionRa_13",
    "InstructionRa_14",
    "InstructionRa_15",
    "InstructionRa_16",
    "InstructionRa_17",
    "InstructionRa_18",
    "InstructionRa_19",
    "InstructionRa_20",
    "InstructionRa_21",
    "InstructionRa_22",
    "InstructionRa_23",
    "InstructionRa_24",
    "InstructionRa_25",
    "InstructionRa_26",
    "InstructionRa_27",
    "InstructionRa_28",
    "InstructionRa_29",
    "InstructionRa_30",
    "InstructionRa_31",
    "RamRa_0",
    "RamRa_1",
    "RamRa_2",
    "RamRa_3",
    "BytecodeRa_0",
    "BytecodeRa_1",
    "BytecodeRa_2",
];
pub const COMMITMENT_BATCH_PLANS: &[CommitmentBatchPlan] = &[
    CommitmentBatchPlan { artifact: "jolt.main_witness_commitments", pcs: "dory", oracle_family: "jolt.main_witness_polys", label: "commitment", oracles: COMMITMENT_BATCH_0_ORACLES, count: 41, domain: "jolt.main_witness_commit_domain", num_vars: 20 },
];
pub const OPTIONAL_COMMITMENT_PLANS: &[OptionalCommitmentPlan] = &[
    OptionalCommitmentPlan { artifact: "jolt.untrusted_advice_commitment", pcs: "dory", oracle: "UntrustedAdvice", label: "untrusted_advice", domain: "jolt.trace_domain", num_vars: 16, skip_policy: OptionalSkipPolicy::MissingOrZero },
    OptionalCommitmentPlan { artifact: "jolt.trusted_advice_commitment", pcs: "dory", oracle: "TrustedAdvice", label: "trusted_advice", domain: "jolt.trace_domain", num_vars: 16, skip_policy: OptionalSkipPolicy::MissingOrZero },
];
pub const TRANSCRIPT_PLAN: &[TranscriptStep] = &[
    TranscriptStep { label: "commitment", source: "jolt.main_witness_commitments", optional: false },
    TranscriptStep { label: "untrusted_advice", source: "jolt.untrusted_advice_commitment", optional: true },
    TranscriptStep { label: "trusted_advice", source: "jolt.trusted_advice_commitment", optional: true },
];
pub const COMMITMENT_PROGRAM: CommitmentProverProgramPlan = CommitmentProverProgramPlan {
    params: COMMITMENT_PARAMS,
    oracle_plans: ORACLE_PLANS,
    batch_plans: COMMITMENT_BATCH_PLANS,
    optional_plans: OPTIONAL_COMMITMENT_PLANS,
    transcript_steps: TRANSCRIPT_PLAN,
};

pub fn prove_commitment_phase<I, T>(
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    I: CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    prove_commitment_phase_with_program(&COMMITMENT_PROGRAM, inputs, prover_setup, transcript)
}

pub fn prove_commitment_phase_with_program<I, T>(
    program: &'static CommitmentProverProgramPlan,
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    I: CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    for plan in program.batch_plans {
        let _batch_span = tracing::info_span!("bolt.commitment.batch").entered();
        commit_batch(program, inputs, prover_setup, &mut artifacts, plan)?;
    }
    for plan in program.optional_plans {
        let _optional_span = tracing::info_span!("bolt.commitment.optional").entered();
        commit_optional(program, inputs, prover_setup, &mut artifacts, plan)?;
    }
    absorb_transcript(program, &artifacts, transcript)?;
    Ok(artifacts)
}

fn commit_batch<I>(
    program: &CommitmentProverProgramPlan,
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &CommitmentBatchPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    if plan.count != plan.oracles.len() {
        return Err(CommitmentPhaseError::PlanCountMismatch {
            artifact: plan.artifact,
            expected: plan.count,
            actual: plan.oracles.len(),
        });
    }
    if let Some(committed) = inputs.commit_batch(program, plan, prover_setup) {
        for committed in committed? {
            artifacts.records.push(committed.record);
            artifacts.commitments.push(committed.commitment);
            if let Some(hint) = committed.hint {
                artifacts.hints.push(hint);
            }
        }
        return Ok(());
    }
    for &oracle in plan.oracles {
        let data = inputs
            .materialize_with_num_vars(oracle, oracle_num_vars(program, oracle, plan.num_vars))
            .ok_or(CommitmentPhaseError::MissingOracle { oracle })?;
        let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
        let data = into_padded_oracle(oracle, oracle_num_vars, data)?;
        let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
        artifacts.records.push(CommitmentRecord {
            artifact: plan.artifact,
            oracle,
            label: plan.label,
            num_vars: oracle_num_vars,
        });
        artifacts.commitments.push(Some(commitment));
        artifacts.hints.push(OracleOpeningHint { oracle, hint });
    }
    Ok(())
}

fn commit_optional<I>(
    program: &CommitmentProverProgramPlan,
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    let Some(data) = inputs.materialize_with_num_vars(plan.oracle, plan.num_vars) else {
        return push_skipped_optional(program, artifacts, plan);
    };
    if should_skip_optional(plan.skip_policy, data.as_ref()) {
        return push_skipped_optional(program, artifacts, plan);
    }
    let data = into_padded_oracle(plan.oracle, plan.num_vars, data)?;
    let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(program, plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(Some(commitment));
    artifacts.hints.push(OracleOpeningHint {
        oracle: plan.oracle,
        hint,
    });
    Ok(())
}

fn push_skipped_optional(
    program: &CommitmentProverProgramPlan,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(program, plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(None);
    Ok(())
}

fn should_skip_optional(policy: OptionalSkipPolicy, data: &[Fr]) -> bool {
    match policy {
        OptionalSkipPolicy::MissingOrZero => data.iter().all(|value| *value == Fr::from_u64(0)),
    }
}

fn into_padded_oracle(
    oracle: &'static str,
    num_vars: usize,
    data: Cow<'_, [Fr]>,
) -> Result<Vec<Fr>, CommitmentPhaseError> {
    let target_len = target_len(num_vars)?;
    if data.len() > target_len {
        return Err(CommitmentPhaseError::OracleTooLarge {
            oracle,
            len: data.len(),
            target_len,
        });
    }
    let mut data = data.into_owned();
    data.resize(target_len, Fr::from_u64(0));
    Ok(data)
}

fn oracle_num_vars(
    program: &CommitmentProverProgramPlan,
    oracle: &'static str,
    fallback: usize,
) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn commit_with_layout(
    data: &[Fr],
    layout_num_vars: usize,
    prover_setup: &DoryProverSetup,
) -> Result<(DoryCommitment, DoryHint), CommitmentPhaseError> {
    let row_len = target_len(layout_num_vars.div_ceil(2))?;
    let _dory_commit_span = tracing::info_span!("bolt.commitment.dory_commit").entered();
    Ok(DoryScheme::commit_evaluations_with_row_len(
        data,
        row_len,
        prover_setup,
    ))
}

fn target_len(num_vars: usize) -> Result<usize, CommitmentPhaseError> {
    if num_vars >= usize::BITS as usize {
        return Err(CommitmentPhaseError::TargetSizeOverflow { num_vars });
    }
    Ok(1usize << num_vars)
}

fn absorb_transcript<T>(
    program: &CommitmentProverProgramPlan,
    artifacts: &CommitmentArtifacts,
    transcript: &mut T,
) -> Result<(), CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    for step in program.transcript_steps {
        let mut appended = false;
        for (record, commitment) in artifacts.records.iter().zip(&artifacts.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(step.label.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        if !step.optional && !appended {
            return Err(CommitmentPhaseError::MissingTranscriptSource {
                source: step.source,
            });
        }
    }
    Ok(())
}
