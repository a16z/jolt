mod constants;
mod parse;
mod source;

use super::checks::require_supported_symbol_for_emitter;
use super::source::{role_filename, role_module_source, rust_str};
use crate::ir::{BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

use crate::emit::rust::{EmitError, RustSourceFile};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentCpuProgram {
    pub role: Role,
    pub params: CommitmentParams,
    pub oracle_plans: Vec<OraclePlan>,
    pub batch_plans: Vec<CommitmentBatchPlan>,
    pub optional_plans: Vec<OptionalCommitmentPlan>,
    pub transcript_steps: Vec<TranscriptStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OraclePlan {
    pub oracle: String,
    pub source: String,
    pub domain: String,
    pub num_vars: usize,
    pub generation: OracleGeneration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleGeneration {
    Reference,
    DenseTrace {
        padding: String,
    },
    OneHotChunk {
        trace_num_vars: usize,
        chunk: usize,
        num_chunks: usize,
        chunk_bits: usize,
        padding: String,
        layout: String,
    },
    OptionalAdvice {
        skip_policy: OptionalSkipPolicy,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentBatchPlan {
    pub artifact: String,
    pub pcs: String,
    pub oracle_family: String,
    pub label: String,
    pub oracles: Vec<String>,
    pub count: usize,
    pub domain: String,
    pub num_vars: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionalCommitmentPlan {
    pub artifact: String,
    pub pcs: String,
    pub oracle: String,
    pub label: String,
    pub domain: String,
    pub num_vars: usize,
    pub skip_policy: OptionalSkipPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionalSkipPolicy {
    MissingOrZero,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptStep {
    pub label: String,
    pub source: String,
    pub optional: bool,
}

pub fn emit_commitment_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = commitment_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

pub fn commitment_cpu_program(
    module: &BoltModule<'_, Cpu>,
) -> Result<CommitmentCpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = CommitmentCpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

impl CommitmentCpuProgram {
    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol_for_emitter(
            "commitment",
            "field",
            &self.params.field,
            "bn254_fr",
        )?;
        require_supported_symbol_for_emitter("commitment", "pcs", &self.params.pcs, "dory")?;
        require_supported_symbol_for_emitter(
            "commitment",
            "transcript",
            &self.params.transcript,
            "blake2b_transcript",
        )?;
        for plan in &self.batch_plans {
            require_supported_symbol_for_emitter("commitment", "batch pcs", &plan.pcs, "dory")?;
        }
        for plan in &self.optional_plans {
            require_supported_symbol_for_emitter("commitment", "optional pcs", &plan.pcs, "dory")?;
        }
        Ok(())
    }

    fn emit_source(&self) -> Result<String, EmitError> {
        let types = self.emit_types()?;
        let constants = self.emit_constants();
        Ok(role_module_source(
            self.emit_imports(),
            &types,
            &constants,
            self.emit_entrypoint(),
        ))
    }

    fn filename(&self) -> &'static str {
        role_filename(
            &self.role,
            "prove_commitment_phase.rs",
            "verify_commitment_phase.rs",
        )
    }

    fn emit_imports(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "use std::borrow::Cow;\n\
                 \n\
                 use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};\n\
                 use jolt_field::{Field, Fr};\n\
                 use jolt_openings::CommitmentScheme as _;\n\
                 use jolt_poly::{EqPolynomial, MultilinearPoly};\n\
                 use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};\n\
                 use jolt_witness::{dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle};\n\
                 use rayon::prelude::*;"
            }
            Role::Verifier => {
                "use jolt_dory::DoryCommitment;\n\
                 use jolt_field::Fr;\n\
                 use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};"
            }
        }
    }

    fn emit_types(&self) -> Result<String, EmitError> {
        match self.role {
            Role::Prover => {
                let mut types = Self::emit_prover_types().to_owned();
                types.push('\n');
                types.push_str(&self.emit_oracle_store_types()?);
                Ok(types)
            }
            Role::Verifier => Ok(Self::emit_verifier_types().to_owned()),
        }
    }

    fn emit_prover_types() -> &'static str {
        r"pub type DefaultCommitmentTranscript = Blake2bTranscript<Fr>;

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingOracle { oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    OracleTooLarge { oracle: &'static str, len: usize, target_len: usize },
    TargetSizeOverflow { num_vars: usize },
}"
    }

    fn emit_oracle_store_types(&self) -> Result<String, EmitError> {
        let input_type = r"
pub struct CommitmentOracleInputs<'a> {
    pub rd_inc: &'a [i128],
    pub ram_inc: &'a [i128],
    pub instruction_keys: &'a [Option<u128>],
    pub ram_addresses: &'a [Option<u128>],
    pub bytecode_indices: &'a [Option<u128>],
    pub untrusted_advice: Option<&'a [Fr]>,
    pub trusted_advice: Option<&'a [Fr]>,
}
";
        let sparse_provider = r#"
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
        let chunk_domain = 1usize << spec.chunk_bits;
        let shift = spec.chunk_bits * (spec.num_chunks - 1 - spec.chunk);
        let mask = (chunk_domain - 1) as u128;
        let mut indices = Vec::with_capacity(trace_len);
        for cycle in 0..trace_len {
            let value = values.get(cycle).copied().flatten().or(spec.padding);
            indices.push(value.map(|value| ((value >> shift) & mask) as u8));
        }
        Some(indices)
    }

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
"#;
        let mut fields = Vec::new();
        let mut provider_arms = Vec::new();
        let mut initializers = Vec::new();
        for plan in &self.oracle_plans {
            match &plan.generation {
                OracleGeneration::Reference => {}
                OracleGeneration::OptionalAdvice { .. } => {
                    let field = rust_field_name(&plan.oracle);
                    fields.push(format!("    pub {field}: Option<Vec<Fr>>,"));
                    provider_arms.push(format!(
                        "            {} => self.{field}.as_deref().map(Cow::Borrowed),",
                        rust_str(&plan.oracle)
                    ));
                    initializers.push(format!(
                        "        {field}: {},",
                        Self::oracle_initializer(plan)?
                    ));
                }
                _ => {
                    let field = rust_field_name(&plan.oracle);
                    fields.push(format!("    pub {field}: Vec<Fr>,"));
                    provider_arms.push(format!(
                        "            {} => Some(Cow::Borrowed(&self.{field})),",
                        rust_str(&plan.oracle)
                    ));
                    initializers.push(format!(
                        "        {field}: {},",
                        Self::oracle_initializer(plan)?
                    ));
                }
            }
        }
        let fields = fields.join("\n");
        let provider_arms = provider_arms.join("\n");
        let initializers = initializers.join("\n");

        Ok(format!(
            "{input_type}
{sparse_provider}
#[derive(Clone, Debug, Default)]
pub struct CommitmentOracles {{
{fields}
}}

impl CommitmentInputProvider for CommitmentOracles {{
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {{
        match oracle {{
{provider_arms}
            _ => None,
        }}
    }}
}}

pub fn build_commitment_oracles(
    inputs: &CommitmentOracleInputs<'_>,
) -> Result<CommitmentOracles, CommitmentPhaseError> {{
    Ok(CommitmentOracles {{
{initializers}
    }})
}}
"
        ))
    }

    fn oracle_initializer(plan: &OraclePlan) -> Result<String, EmitError> {
        match &plan.generation {
            OracleGeneration::Reference => Err(EmitError::new(format!(
                "reference oracle @{} has no prover initializer",
                plan.oracle
            ))),
            OracleGeneration::DenseTrace { .. } => Ok(format!(
                "dense_i128_column_to_field(inputs.{}, target_len({})?)",
                rust_input_field(&plan.source)?,
                plan.num_vars
            )),
            OracleGeneration::OneHotChunk {
                trace_num_vars,
                chunk,
                num_chunks,
                chunk_bits,
                padding,
                ..
            } => Ok(format!(
                "one_hot_chunk_address_major(inputs.{}, {chunk}, {num_chunks}, {chunk_bits}, target_len({trace_num_vars})?, {})",
                rust_input_field(&plan.source)?,
                rust_padding_value(padding)?
            )),
            OracleGeneration::OptionalAdvice { .. } => Ok(format!(
                "optional_field_oracle(inputs.{}, target_len({})?)",
                rust_input_field(&plan.source)?,
                plan.num_vars
            )),
        }
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultCommitmentTranscript = Blake2bTranscript<Fr>;

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
pub struct CommitmentVerifierProgramPlan {
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

#[derive(Clone, Debug, Default)]
pub struct CommitmentArtifacts {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingProofCommitment { oracle: &'static str },
    MissingProofCommitmentSlot { artifact: &'static str, oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    ProofCommitmentCountMismatch { expected: usize, actual: usize },
}"
    }
}

fn rust_field_name(value: &str) -> String {
    let mut output = String::new();
    let mut previous_was_separator = false;
    for (index, character) in value.chars().enumerate() {
        if character == '_' {
            output.push('_');
            previous_was_separator = true;
            continue;
        }
        if character.is_ascii_uppercase() {
            if index != 0 && !previous_was_separator {
                output.push('_');
            }
            output.push(character.to_ascii_lowercase());
        } else {
            output.push(character);
        }
        previous_was_separator = false;
    }
    output
}

fn rust_input_field(source: &str) -> Result<&'static str, EmitError> {
    match source {
        "trace.rd_inc" => Ok("rd_inc"),
        "trace.ram_inc" => Ok("ram_inc"),
        "trace.instruction_keys" => Ok("instruction_keys"),
        "trace.ram_addresses" => Ok("ram_addresses"),
        "trace.bytecode_indices" => Ok("bytecode_indices"),
        "advice.untrusted" => Ok("untrusted_advice"),
        "advice.trusted" => Ok("trusted_advice"),
        _ => Err(EmitError::new(format!(
            "unsupported oracle source `{source}`"
        ))),
    }
}

fn rust_padding_value(padding: &str) -> Result<&'static str, EmitError> {
    match padding {
        "zero" => Ok("Some(0)"),
        "none" => Ok("None"),
        _ => Err(EmitError::new(format!(
            "unsupported oracle padding `{padding}`"
        ))),
    }
}
