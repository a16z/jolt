//! Commitment-phase transcript bridge between Bolt IR and jolt-core.
//!
//! This keeps the first oracle intentionally narrow: Bolt owns the commitment
//! ordering through its CPU IR, while jolt-core owns the reference
//! `append_serializable` transcript semantics for the same Dory commitments.

use std::collections::BTreeMap;

use ark_serialize::CanonicalSerialize;
use jolt_compiler_v2::{
    build_commitment_protocol, commitment_cpu_program, lower_commitment_to_compute,
    lower_compute_to_cpu, lower_piop_and_fiat_shamir, project_prover_party, project_verifier_party,
    CommitmentCpuProgram, JoltProtocolParams, MeliorContext, OptionalSkipPolicy, OracleGeneration,
    Role, TranscriptStep,
};
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme as CoreCommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _};
use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_equivalence::checkpoint::{
    assert_transcripts_match, CheckpointTranscript, TranscriptEvent,
};
use jolt_equivalence::cross_verifier::conversion::commitment_to_ark;
use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, Program};
use jolt_openings::StreamingCommitment;
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};
use jolt_verifier::TRANSCRIPT_LABEL;
use jolt_witness::CycleInput;
use jolt_witness_v2::{
    dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle,
};

type CoreFr = ark_bn254::Fr;
type CoreCommitment = <DoryCommitmentScheme as CoreCommitmentScheme>::Commitment;
type CoreProver<'a> =
    JoltCpuProver<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreProof = CoreJoltProof<CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;

#[test]
fn bolt_commitment_transcript_matches_jolt_core_append_serializable() {
    let (prover_program, verifier_program) = bolt_commitment_programs();
    let prover_trace = run_bolt_commitment_prover(&prover_program);
    let verifier_trace = run_bolt_commitment_verifier(&verifier_program, &prover_trace.commitments);
    let core_log = core_commitment_log(&prover_trace, &prover_program.transcript_steps);

    assert_eq!(prover_trace.commitments, verifier_trace.commitments);
    assert_transcripts_match(&prover_trace.log, &verifier_trace.log);
    assert_transcripts_match(&core_log, &prover_trace.log);
}

#[test]
fn bolt_commitment_real_muldiv_trace_matches_jolt_core() {
    let fixture = core_muldiv_commitment_fixture();
    let (prover_program, verifier_program) = bolt_commitment_programs_with_params(&fixture.params);
    let oracle_data = real_muldiv_oracle_data(&prover_program, &fixture.cycle_inputs);

    let prover_trace = run_bolt_commitment_prover_with(
        &prover_program,
        &fixture.pcs_setup,
        |oracle, _num_vars| oracle_data.get(oracle).cloned().flatten(),
    );
    let verifier_trace = run_bolt_commitment_verifier(&verifier_program, &prover_trace.commitments);

    let bolt_core_commitments = prover_trace
        .commitments
        .iter()
        .filter_map(|commitment| commitment.as_ref())
        .take(fixture.commitments.len())
        .map(commitment_to_ark)
        .collect::<Vec<_>>();
    assert_eq!(bolt_core_commitments, fixture.commitments);

    let core_log =
        core_commitments_transcript_log(&fixture.commitments, &prover_program.transcript_steps);
    assert_eq!(prover_trace.commitments, verifier_trace.commitments);
    assert_transcripts_match(&prover_trace.log, &verifier_trace.log);
    assert_transcripts_match(&core_log, &prover_trace.log);
}

#[derive(Clone, Debug)]
struct CommitmentRecord {
    artifact: String,
}

#[derive(Clone, Debug)]
struct BoltCommitmentTrace {
    commitments: Vec<Option<DoryCommitment>>,
    records: Vec<CommitmentRecord>,
    log: Vec<TranscriptEvent>,
}

#[derive(Clone)]
struct CoreMuldivCommitmentFixture {
    params: JoltProtocolParams,
    pcs_setup: DoryProverSetup,
    cycle_inputs: Vec<CycleInput>,
    commitments: Vec<CoreCommitment>,
}

fn bolt_commitment_programs() -> (CommitmentCpuProgram, CommitmentCpuProgram) {
    let params = JoltProtocolParams::new(0, 0, 0);
    bolt_commitment_programs_with_params(&params)
}

fn bolt_commitment_programs_with_params(
    params: &JoltProtocolParams,
) -> (CommitmentCpuProgram, CommitmentCpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_commitment_protocol(&context, params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_commitment_to_compute(&context, &prover_party).expect("lower prover compute");
    let verifier_compute =
        lower_commitment_to_compute(&context, &verifier_party).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = commitment_cpu_program(&prover_cpu).expect("extract prover CPU program");
    let verifier_program =
        commitment_cpu_program(&verifier_cpu).expect("extract verifier CPU program");
    (prover_program, verifier_program)
}

fn core_muldiv_commitment_fixture() -> CoreMuldivCommitmentFixture {
    DoryGlobals::reset();

    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("muldiv inputs");
    let mut core_program = host::Program::new("muldiv-guest");
    let (core_bytecode, init_memory_state, _, entry_address) = core_program.decode();
    let (_, _, _, core_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        core_io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        entry_address,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = core_program.get_elf_contents().expect("muldiv elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let (proof, _debug): (CoreProof, _) = prover.prove();

    let mut host_program = Program::new("muldiv-guest");
    let (bytecode_raw, _, _, host_entry_address) = host_program.decode();
    let (_, trace, _, host_io_device) = host_program.trace(&inputs, &[], &[]);
    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, host_entry_address);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), proof.trace_length);
    let (cycle_inputs, _, _) = extract_trace::<_, Fr>(
        &trace,
        proof.trace_length,
        &bytecode,
        &host_io_device.memory_layout,
        r1cs_key.num_vars_padded,
    );

    let log_t = proof.trace_length.trailing_zeros() as usize;
    let log_k_bytecode = prover_preprocessing
        .shared
        .bytecode
        .code_size
        .trailing_zeros() as usize;
    let log_k_ram = proof.ram_K.trailing_zeros() as usize;
    let params = JoltProtocolParams::new(log_t, log_k_bytecode, log_k_ram);

    CoreMuldivCommitmentFixture {
        params,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
        cycle_inputs,
        commitments: proof.commitments,
    }
}

fn run_bolt_commitment_prover(program: &CommitmentCpuProgram) -> BoltCommitmentTrace {
    assert_eq!(program.role, Role::Prover);
    let setup = DoryScheme::setup_prover(max_num_vars(program));
    run_bolt_commitment_prover_with(program, &setup, |oracle, num_vars| {
        optional_oracle_data(oracle, num_vars)
    })
}

fn run_bolt_commitment_prover_with<F>(
    program: &CommitmentCpuProgram,
    setup: &DoryProverSetup,
    mut materialize: F,
) -> BoltCommitmentTrace
where
    F: FnMut(&str, usize) -> Option<Vec<Fr>>,
{
    assert_eq!(program.role, Role::Prover);
    let mut commitments = Vec::new();
    let mut records = Vec::new();

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for oracle in &plan.oracles {
            let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
            let data = materialize(oracle, oracle_num_vars)
                .unwrap_or_else(|| panic!("missing batch oracle `{oracle}`"));
            let data = into_padded_oracle(data, oracle_num_vars);
            let (commitment, _) = commit_with_layout(&data, plan.num_vars, setup);
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            commitments.push(Some(commitment));
        }
    }
    for plan in &program.optional_plans {
        let oracle_num_vars = oracle_num_vars(program, &plan.oracle, plan.num_vars);
        let commitment = materialize(&plan.oracle, oracle_num_vars)
            .filter(|data| !should_skip_optional(&plan.skip_policy, data))
            .map(|data| {
                let data = into_padded_oracle(data, oracle_num_vars);
                commit_with_layout(&data, plan.num_vars, setup).0
            });
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }

    let log = bolt_transcript_log(&records, &commitments, &program.transcript_steps);
    BoltCommitmentTrace {
        commitments,
        records,
        log,
    }
}

fn real_muldiv_oracle_data(
    program: &CommitmentCpuProgram,
    cycle_inputs: &[CycleInput],
) -> BTreeMap<String, Option<Vec<Fr>>> {
    let mut data = BTreeMap::new();
    for plan in &program.oracle_plans {
        let materialized = match &plan.generation {
            OracleGeneration::Reference => continue,
            OracleGeneration::DenseTrace { .. } => {
                let values = dense_source(cycle_inputs, &plan.source);
                Some(dense_i128_column_to_field(
                    &values,
                    target_len(plan.num_vars),
                ))
            }
            OracleGeneration::OneHotChunk {
                trace_num_vars,
                chunk,
                num_chunks,
                chunk_bits,
                padding,
                ..
            } => {
                let values = one_hot_source(cycle_inputs, &plan.source);
                Some(one_hot_chunk_address_major(
                    &values,
                    *chunk,
                    *num_chunks,
                    *chunk_bits,
                    target_len(*trace_num_vars),
                    padding_value(padding),
                ))
            }
            OracleGeneration::OptionalAdvice { .. } => {
                optional_field_oracle::<Fr>(None, target_len(plan.num_vars))
            }
        };
        let _ = data.insert(plan.oracle.clone(), materialized);
    }
    data
}

fn run_bolt_commitment_verifier(
    program: &CommitmentCpuProgram,
    proof_commitments: &[Option<DoryCommitment>],
) -> BoltCommitmentTrace {
    assert_eq!(program.role, Role::Verifier);
    let mut commitments = Vec::new();
    let mut records = Vec::new();
    let mut cursor = 0;

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for _ in &plan.oracles {
            let commitment = proof_commitments
                .get(cursor)
                .expect("proof commitment slot")
                .clone();
            assert!(commitment.is_some(), "batch commitments cannot be skipped");
            cursor += 1;
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            commitments.push(commitment);
        }
    }
    for plan in &program.optional_plans {
        let commitment = proof_commitments
            .get(cursor)
            .expect("optional proof commitment slot")
            .clone();
        cursor += 1;
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }
    assert_eq!(cursor, proof_commitments.len());

    let log = bolt_transcript_log(&records, &commitments, &program.transcript_steps);
    BoltCommitmentTrace {
        commitments,
        records,
        log,
    }
}

fn bolt_transcript_log(
    records: &[CommitmentRecord],
    commitments: &[Option<DoryCommitment>],
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);

    for step in transcript_steps {
        let mut appended = false;
        for (record, commitment) in records.iter().zip(commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(
                    static_transcript_label(&step.label),
                    commitment.serialized_len(),
                ));
                commitment.append_to_transcript(&mut transcript);
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing transcript source");
    }

    transcript.into_log()
}

fn core_commitment_log(
    bolt_trace: &BoltCommitmentTrace,
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript = CoreBlake2bTranscript::new(TRANSCRIPT_LABEL);
    let mut events = Vec::new();

    for step in transcript_steps {
        let mut appended = false;
        for (record, commitment) in bolt_trace.records.iter().zip(&bolt_trace.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                let core_commitment = commitment_to_ark(commitment);
                let label = static_transcript_label(&step.label);
                let bytes = core_append_serializable_bytes(label, &core_commitment);
                let state_count_before = transcript.state_history.len();
                transcript.append_serializable(label, &core_commitment);
                let new_states = &transcript.state_history[state_count_before..];
                assert_eq!(new_states.len(), bytes.len());
                for (bytes, state_after) in bytes.into_iter().zip(new_states) {
                    events.push(TranscriptEvent::Append {
                        bytes,
                        state_after: *state_after,
                    });
                }
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing core transcript source");
    }

    events
}

fn core_commitments_transcript_log(
    commitments: &[CoreCommitment],
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript = CoreBlake2bTranscript::new(TRANSCRIPT_LABEL);
    let mut events = Vec::new();

    for step in transcript_steps {
        if step.source != "jolt.main_witness_commitments" {
            assert!(step.optional, "unexpected non-main commitment source");
            continue;
        }
        for commitment in commitments {
            let label = static_transcript_label(&step.label);
            let bytes = core_append_serializable_bytes(label, commitment);
            let state_count_before = transcript.state_history.len();
            transcript.append_serializable(label, commitment);
            let new_states = &transcript.state_history[state_count_before..];
            assert_eq!(new_states.len(), bytes.len());
            for (bytes, state_after) in bytes.into_iter().zip(new_states) {
                events.push(TranscriptEvent::Append {
                    bytes,
                    state_after: *state_after,
                });
            }
        }
    }

    events
}

fn core_append_serializable_bytes<T: CanonicalSerialize>(
    label: &'static [u8],
    data: &T,
) -> Vec<Vec<u8>> {
    let mut payload = Vec::new();
    data.serialize_uncompressed(&mut payload)
        .expect("core commitment serialization");
    let mut header = [0u8; 32];
    header[..label.len()].copy_from_slice(label);
    header[24..].copy_from_slice(&(payload.len() as u64).to_be_bytes());
    payload.reverse();
    vec![header.to_vec(), payload]
}

fn deterministic_oracle_data(oracle: &str, num_vars: usize) -> Vec<Fr> {
    let seed = oracle.bytes().fold(17u64, |state, byte| {
        state.wrapping_mul(131).wrapping_add(byte as u64)
    });
    (0..target_len(num_vars))
        .map(|index| Fr::from_u64(seed.wrapping_add(index as u64 + 1)))
        .collect()
}

fn dense_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<i128> {
    let slot = match source {
        "trace.rd_inc" => 0,
        "trace.ram_inc" => 1,
        _ => panic!("unsupported dense source `{source}`"),
    };
    cycle_inputs.iter().map(|cycle| cycle.dense[slot]).collect()
}

fn one_hot_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<Option<u128>> {
    let slot = match source {
        "trace.instruction_keys" => 0,
        "trace.bytecode_indices" => 1,
        "trace.ram_addresses" => 2,
        _ => panic!("unsupported one-hot source `{source}`"),
    };
    cycle_inputs
        .iter()
        .map(|cycle| cycle.one_hot[slot])
        .collect()
}

fn padding_value(padding: &str) -> Option<u128> {
    match padding {
        "zero" => Some(0),
        "none" => None,
        _ => panic!("unsupported padding `{padding}`"),
    }
}

fn optional_oracle_data(oracle: &str, num_vars: usize) -> Option<Vec<Fr>> {
    match oracle {
        "UntrustedAdvice" | "TrustedAdvice" => None,
        _ => Some(deterministic_oracle_data(oracle, num_vars)),
    }
}

fn should_skip_optional(policy: &OptionalSkipPolicy, data: &[Fr]) -> bool {
    match policy {
        OptionalSkipPolicy::MissingOrZero => data.iter().all(|value| *value == Fr::from_u64(0)),
    }
}

fn oracle_num_vars(program: &CommitmentCpuProgram, oracle: &str, fallback: usize) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn into_padded_oracle(mut data: Vec<Fr>, num_vars: usize) -> Vec<Fr> {
    let target_len = target_len(num_vars);
    assert!(
        data.len() <= target_len,
        "oracle has {} values, target length is {target_len}",
        data.len()
    );
    data.resize(target_len, Fr::from_u64(0));
    data
}

fn commit_with_layout(
    data: &[Fr],
    layout_num_vars: usize,
    setup: &DoryProverSetup,
) -> (DoryCommitment, DoryHint) {
    let row_len = target_len(layout_num_vars.div_ceil(2));
    let mut partial = DoryScheme::begin(setup);
    for row in data.chunks(row_len) {
        DoryScheme::feed(&mut partial, row, setup);
    }
    let hint = DoryHint(partial.row_commitments.clone());
    let commitment = DoryScheme::finish(partial, setup);
    (commitment, hint)
}

fn max_num_vars(program: &CommitmentCpuProgram) -> usize {
    program
        .batch_plans
        .iter()
        .map(|plan| plan.num_vars)
        .chain(program.optional_plans.iter().map(|plan| plan.num_vars))
        .max()
        .unwrap_or(0)
}

fn target_len(num_vars: usize) -> usize {
    1usize << num_vars
}

fn static_transcript_label(label: &str) -> &'static [u8] {
    match label {
        "commitment" => b"commitment",
        "untrusted_advice" => b"untrusted_advice",
        "trusted_advice" => b"trusted_advice",
        _ => panic!("unsupported transcript label `{label}`"),
    }
}
