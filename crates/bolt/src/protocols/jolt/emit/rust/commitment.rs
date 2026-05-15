use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

use crate::emit::rust::{push_format, EmitError, RustSourceFile};

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
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut oracle_plans = Vec::new();
        let mut batch_plans = Vec::new();
        let mut optional_plans = Vec::new();
        let mut transcript_steps = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(CommitmentParams {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.oracle_dense_trace" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: symbol_attr(op, "source")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::DenseTrace {
                            padding: string_attr(op, "padding")?,
                        },
                    });
                }
                "cpu.oracle_one_hot_chunk" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: symbol_attr(op, "source")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::OneHotChunk {
                            trace_num_vars: int_attr(op, "trace_num_vars")?,
                            chunk: int_attr(op, "chunk")?,
                            num_chunks: int_attr(op, "num_chunks")?,
                            chunk_bits: int_attr(op, "chunk_bits")?,
                            padding: string_attr(op, "padding")?,
                            layout: string_attr(op, "layout")?,
                        },
                    });
                }
                "cpu.oracle_optional_advice" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: symbol_attr(op, "source")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::OptionalAdvice {
                            skip_policy: skip_policy_attr(op, "skip_policy")?,
                        },
                    });
                }
                "cpu.oracle_ref" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: String::new(),
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::Reference,
                    });
                }
                "cpu.pcs_commit_batch" | "cpu.pcs_receive_batch" => {
                    batch_plans.push(CommitmentBatchPlan {
                        artifact: symbol_attr(op, "artifact")?,
                        pcs: symbol_attr(op, "pcs")?,
                        oracle_family: symbol_attr(op, "oracle_family")?,
                        label: string_attr(op, "label")?,
                        oracles: symbol_array_attr(op, "ordered_oracles")?,
                        count: int_attr(op, "count")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                    });
                }
                "cpu.pcs_commit_optional" | "cpu.pcs_receive_optional" => {
                    optional_plans.push(OptionalCommitmentPlan {
                        artifact: symbol_attr(op, "artifact")?,
                        pcs: symbol_attr(op, "pcs")?,
                        oracle: symbol_attr(op, "oracle")?,
                        label: string_attr(op, "label")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        skip_policy: skip_policy_attr(op, "skip_policy")?,
                    });
                }
                "cpu.transcript_absorb" => {
                    transcript_steps.push(TranscriptStep {
                        label: string_attr(op, "label")?,
                        source: transcript_artifact_source(op)?,
                        optional: bool_attr(op, "optional")?,
                    });
                }
                _ => {}
            }
        }

        Ok(Self {
            params: params.ok_or_else(|| EmitError::new("missing cpu.params"))?,
            role: module
                .role()
                .ok_or_else(|| EmitError::new("missing cpu party role"))?,
            oracle_plans,
            batch_plans,
            optional_plans,
            transcript_steps,
        })
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol("field", &self.params.field, "bn254_fr")?;
        require_supported_symbol("pcs", &self.params.pcs, "dory")?;
        require_supported_symbol("transcript", &self.params.transcript, "blake2b_transcript")?;
        for plan in &self.batch_plans {
            require_supported_symbol("batch pcs", &plan.pcs, "dory")?;
        }
        for plan in &self.optional_plans {
            require_supported_symbol("optional pcs", &plan.pcs, "dory")?;
        }
        Ok(())
    }

    fn emit_source(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        source.push_str(self.emit_imports());
        source.push_str("\n\n");
        source.push_str(&self.emit_types()?);
        source.push('\n');
        source.push_str(&self.emit_constants());
        source.push('\n');
        source.push_str(self.emit_entrypoint());
        Ok(source)
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_commitment_phase.rs",
            Role::Verifier => "verify_commitment_phase.rs",
        }
    }

    fn emit_imports(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "use std::borrow::Cow;\n\
                 \n\
                 use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};\n\
                 use jolt_field::Fr;\n\
                 use jolt_openings::CommitmentScheme as _;\n\
                 use jolt_poly::{EqPolynomial, MultilinearPoly};\n\
                 use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};\n\
                 use jolt_witness::{dense_i128_column_to_field, one_hot_chunk_address_major, one_hot_chunk_indices, optional_field_oracle, CommitmentTraceSources};\n\
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

    fn is_one_hot(&self) -> bool {
        true
    }

    fn for_each_one(&self, f: &mut dyn FnMut(usize)) {
        for flat in self.nonzero_flat_indices() {
            f(flat);
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
            // Pad to layout_num_vars (not oracle_num_vars) so the row-chunked
            // commitment has uniform row-count across all oracles in the batch.
            // Required for `joint_opening_hint`'s `combine_hints` to produce a
            // valid aggregate commitment.
            let data = into_padded_oracle(oracle, layout_num_vars, Cow::Owned(data))?;
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
        let input_binding = if initializers
            .iter()
            .any(|initializer| initializer.contains("inputs."))
        {
            "inputs"
        } else {
            "_inputs"
        };
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
    {input_binding}: &CommitmentOracleInputs<'_>,
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

    fn emit_constants(&self) -> String {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const COMMITMENT_PARAMS: CommitmentParams = CommitmentParams {{\n\
             \x20   field: {},\n\
             \x20   pcs: {},\n\
             \x20   transcript: {},\n\
             }};\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript)
            ),
        );

        let oracle_plans = self
            .oracle_plans
            .iter()
            .map(|plan| {
                format!(
                    "    OraclePlan {{ oracle: {}, domain: {}, num_vars: {} }},",
                    rust_str(&plan.oracle),
                    rust_str(&plan.domain),
                    plan.num_vars
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!("pub const ORACLE_PLANS: &[OraclePlan] = &[\n{oracle_plans}\n];\n"),
        );

        for (index, plan) in self.batch_plans.iter().enumerate() {
            let oracles = plan
                .oracles
                .iter()
                .map(|oracle| format!("    {},", rust_str(oracle)))
                .collect::<Vec<_>>()
                .join("\n");
            push_format(
                &mut source,
                format_args!(
                    "pub const COMMITMENT_BATCH_{index}_ORACLES: &[&str] = &[\n{oracles}\n];\n"
                ),
            );
        }

        let batch_plans = self
            .batch_plans
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                format!(
                    "    CommitmentBatchPlan {{ artifact: {}, pcs: {}, oracle_family: {}, label: {}, oracles: COMMITMENT_BATCH_{index}_ORACLES, count: {}, domain: {}, num_vars: {} }},",
                    rust_str(&plan.artifact),
                    rust_str(&plan.pcs),
                    rust_str(&plan.oracle_family),
                    rust_str(&plan.label),
                    plan.count,
                    rust_str(&plan.domain),
                    plan.num_vars
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const COMMITMENT_BATCH_PLANS: &[CommitmentBatchPlan] = &[\n{batch_plans}\n];\n"
            ),
        );

        let optional_plans = self
            .optional_plans
            .iter()
            .map(|plan| {
                format!(
                    "    OptionalCommitmentPlan {{ artifact: {}, pcs: {}, oracle: {}, label: {}, domain: {}, num_vars: {}, skip_policy: {} }},",
                    rust_str(&plan.artifact),
                    rust_str(&plan.pcs),
                    rust_str(&plan.oracle),
                    rust_str(&plan.label),
                    rust_str(&plan.domain),
                    plan.num_vars,
                    plan.skip_policy.rust_variant()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const OPTIONAL_COMMITMENT_PLANS: &[OptionalCommitmentPlan] = &[\n{optional_plans}\n];\n"
            ),
        );

        let steps = self
            .transcript_steps
            .iter()
            .map(|step| {
                format!(
                    "    TranscriptStep {{ label: {}, source: {}, optional: {} }},",
                    rust_str(&step.label),
                    rust_str(&step.source),
                    step.optional
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!("pub const TRANSCRIPT_PLAN: &[TranscriptStep] = &[\n{steps}\n];"),
        );
        source.push('\n');
        let program_type = match self.role {
            Role::Prover => "CommitmentProverProgramPlan",
            Role::Verifier => "CommitmentVerifierProgramPlan",
        };
        push_format(
            &mut source,
            format_args!(
                "pub const COMMITMENT_PROGRAM: {program_type} = {program_type} {{\n\
                 \x20   params: COMMITMENT_PARAMS,\n\
                 \x20   oracle_plans: ORACLE_PLANS,\n\
                 \x20   batch_plans: COMMITMENT_BATCH_PLANS,\n\
                 \x20   optional_plans: OPTIONAL_COMMITMENT_PLANS,\n\
                 \x20   transcript_steps: TRANSCRIPT_PLAN,\n\
                 }};\n"
            ),
        );

        source
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => Self::emit_prover_entrypoint(),
            Role::Verifier => Self::emit_verifier_entrypoint(),
        }
    }

    fn emit_prover_entrypoint() -> &'static str {
        r#"pub fn prove_commitment_phase<I, T>(
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
        let data = into_padded_oracle(oracle, plan.num_vars, data)?;
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
"#
    }

    fn emit_verifier_entrypoint() -> &'static str {
        r"pub fn verify_commitment_phase<T>(
    proof_commitments: &[Option<DoryCommitment>],
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    verify_commitment_phase_with_program(&COMMITMENT_PROGRAM, proof_commitments, transcript)
}

pub fn verify_commitment_phase_with_program<T>(
    program: &'static CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    let mut cursor = 0usize;
    for plan in program.batch_plans {
        receive_batch(program, proof_commitments, &mut cursor, &mut artifacts, plan)?;
    }
    for plan in program.optional_plans {
        receive_optional(program, proof_commitments, &mut cursor, &mut artifacts, plan)?;
    }
    if cursor != proof_commitments.len() {
        return Err(CommitmentPhaseError::ProofCommitmentCountMismatch {
            expected: cursor,
            actual: proof_commitments.len(),
        });
    }
    absorb_transcript(program, &artifacts, transcript)?;
    Ok(artifacts)
}

fn receive_batch(
    program: &'static CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
    cursor: &mut usize,
    artifacts: &mut CommitmentArtifacts,
    plan: &CommitmentBatchPlan,
) -> Result<(), CommitmentPhaseError> {
    if plan.count != plan.oracles.len() {
        return Err(CommitmentPhaseError::PlanCountMismatch {
            artifact: plan.artifact,
            expected: plan.count,
            actual: plan.oracles.len(),
        });
    }
    for &oracle in plan.oracles {
        let commitment = proof_commitments
            .get(*cursor)
            .ok_or(CommitmentPhaseError::MissingProofCommitmentSlot {
                artifact: plan.artifact,
                oracle,
            })?
            .as_ref()
            .ok_or(CommitmentPhaseError::MissingProofCommitment { oracle })?
            .clone();
        *cursor += 1;
        let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
        artifacts.records.push(CommitmentRecord {
            artifact: plan.artifact,
            oracle,
            label: plan.label,
            num_vars: oracle_num_vars,
        });
        artifacts.commitments.push(Some(commitment));
    }
    Ok(())
}

fn receive_optional(
    program: &'static CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
    cursor: &mut usize,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    let commitment = proof_commitments
        .get(*cursor)
        .ok_or(CommitmentPhaseError::MissingProofCommitmentSlot {
            artifact: plan.artifact,
            oracle: plan.oracle,
        })?
        .clone();
    *cursor += 1;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(program, plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(commitment);
    Ok(())
}

pub fn commitment_verifier_program() -> &'static CommitmentVerifierProgramPlan {
    &COMMITMENT_PROGRAM
}

fn oracle_num_vars(
    program: &'static CommitmentVerifierProgramPlan,
    oracle: &'static str,
    fallback: usize,
) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn absorb_transcript<T>(
    program: &'static CommitmentVerifierProgramPlan,
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
"
    }
}

impl OptionalSkipPolicy {
    fn parse(value: &str) -> Result<Self, EmitError> {
        match value {
            "missing_or_zero" => Ok(Self::MissingOrZero),
            _ => Err(EmitError::new(format!(
                "unsupported optional commitment skip policy `{value}`"
            ))),
        }
    }

    fn rust_variant(&self) -> &'static str {
        match self {
            Self::MissingOrZero => "OptionalSkipPolicy::MissingOrZero",
        }
    }
}

fn string_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

fn skip_policy_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<OptionalSkipPolicy, EmitError> {
    OptionalSkipPolicy::parse(&string_attr(operation, attr)?)
}

fn symbol_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol"))
}

fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&attribute).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

fn parse_symbol_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().strip_prefix('@').map(ToOwned::to_owned))
        .collect()
}

fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn bool_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<bool, EmitError> {
    operation
        .attribute(attr)
        .map(|attribute| match attribute.to_string().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        })
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "bool"))
}

fn transcript_artifact_source(operation: OperationRef<'_, '_>) -> Result<String, EmitError> {
    let artifact = operation
        .operand(1)
        .map_err(|_| attr_error(operation, "artifact operand", "value"))?;
    let owner = OperationResult::try_from(artifact)
        .map_err(|_| EmitError::new("cpu.transcript_absorb artifact operand must be op result"))?
        .owner();
    symbol_attr(owner, "artifact")
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> EmitError {
    EmitError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}

fn operation_name(operation: OperationRef<'_, '_>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
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

fn require_supported_symbol(kind: &str, actual: &str, expected: &str) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; Rust commitment emitter currently supports @{expected}"
        )))
    }
}
