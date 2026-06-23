//! Akita prover-side artifact construction.
//!
//! This module is intentionally feature-gated and keeps Akita-specific setup
//! native to the modular verifier/opening crates. The legacy prover still owns
//! trace execution and sumcheck proving; Akita artifacts are derived from those
//! prover-native inputs without going through deleted core/compat paths.

use common::constants::XLEN;
use jolt_akita::{AkitaCommitment, AkitaField, AkitaProverHint, AkitaScheme, AkitaSetupParams};
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    lattice_packed_validity_digest,
};
use jolt_field::FromPrimitiveInt;
use jolt_openings::{
    CommitmentScheme, PackingProverSetup, PackingSetupParams, PackingVerifierSetup,
};
use jolt_poly::Polynomial;
use jolt_program::preprocess::ProgramMetadata as VerifierProgramMetadata;
use jolt_verifier::{
    akita::{
        commit_akita_packing_jolt_witness, AkitaCommittedPackedJoltWitness,
        AkitaPackingJoltWitnessInput, AkitaPackingProverSetup, AkitaPackingVerifierSetup,
        AkitaPrecommittedOpeningInput, AkitaVerifierPreprocessing,
    },
    config::{IncrementCommitmentMode, JoltProtocolConfig, PcsFamily, ProgramMode},
    stages::{
        stage8::{
            derive_lattice_packed_validity_requirements, derive_lattice_packed_witness_layout,
        },
        CommittedProgramSchedule, PrecommittedSchedule,
    },
    CommittedProgramPreprocessing as VerifierCommittedProgramPreprocessing,
    ProgramPreprocessing as VerifierProgramPreprocessing, VerifierError,
};
use tracer::{build_trace_rows, instruction::Cycle};

use crate::{
    curve::JoltCurve,
    field::{akita::JoltAkitaField, JoltField},
    poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment},
    transcripts::Transcript,
    zkvm::{
        bytecode::chunks::{
            committed_bytecode_chunk_cycle_len, committed_lanes, for_each_active_lane_value,
            ActiveLaneValue,
        },
        instruction::LookupQuery,
        preprocessing::JoltSharedPreprocessing,
        program::{build_program_image_words_padded, ProgramPreprocessing},
        prover::JoltCpuProver,
    },
};

#[derive(Clone, Debug)]
pub struct AkitaPackedWitnessProverData {
    pub protocol: JoltProtocolConfig,
    pub precommitted: PrecommittedSchedule,
    pub prover_setup: AkitaPackingProverSetup,
    pub verifier_setup: AkitaPackingVerifierSetup,
    pub committed: AkitaCommittedPackedJoltWitness,
}

#[derive(Clone)]
pub struct AkitaPrecommittedProgramProverData {
    pub preprocessing: AkitaVerifierPreprocessing,
    pub opening_inputs: Vec<AkitaOwnedPrecommittedOpening>,
}

impl AkitaPrecommittedProgramProverData {
    pub fn opening_inputs(&self) -> Vec<AkitaPrecommittedOpeningInput<'_>> {
        self.opening_inputs
            .iter()
            .map(AkitaOwnedPrecommittedOpening::as_input)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct AkitaOwnedPrecommittedOpening {
    pub commitment: AkitaCommitment,
    pub polynomial: Polynomial<AkitaField>,
    pub hint: AkitaProverHint,
}

impl AkitaOwnedPrecommittedOpening {
    pub fn as_input(&self) -> AkitaPrecommittedOpeningInput<'_> {
        AkitaPrecommittedOpeningInput {
            polynomial: &self.polynomial,
            hint: &self.hint,
        }
    }
}

impl<F, C, PCS, ProofTranscript> JoltCpuProver<'_, F, C, PCS, ProofTranscript>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
    ProofTranscript: Transcript,
{
    pub fn commit_akita_packed_witness(
        &self,
    ) -> Result<AkitaPackedWitnessProverData, VerifierError> {
        if !self.program_io.trusted_advice.is_empty()
            || self.advice.trusted_advice_commitment.is_some()
        {
            return Err(invalid_akita_prover_config(
                "Akita packed witness construction does not support trusted advice yet",
            ));
        }

        let precommitted = derive_precommitted_schedule(
            &self.preprocessing.shared,
            self.trace.len().ilog2() as usize,
            self.one_hot_params.log_k_chunk,
            !self.program_io.untrusted_advice.is_empty(),
        )?;
        let layout = derive_akita_packed_layout(
            &self.preprocessing.shared,
            self.trace.len().ilog2() as usize,
            self.one_hot_params.log_k_chunk,
            self.one_hot_params.instruction_d,
            self.one_hot_params.bytecode_d,
            self.one_hot_params.ram_d,
            &precommitted,
        )?;
        let max_num_vars = layout
            .dimension
            .max(akita_precommitted_program_max_num_vars(
                &self.preprocessing.shared,
            )?);
        let (prover_setup, verifier_setup) =
            akita_packing_setup_with_max_num_vars(&layout, max_num_vars, 1);
        let trace_rows = build_trace_rows(
            &self.trace,
            &self.preprocessing.materialized_program().bytecode,
        )
        .map_err(|error| {
            invalid_akita_prover_config(format!("failed to build Akita trace rows: {error}"))
        })?;
        let instruction_lookup_indices = instruction_lookup_indices(&self.trace);
        let committed = commit_akita_packing_jolt_witness(
            &prover_setup,
            AkitaPackingJoltWitnessInput {
                layout,
                trace_rows: &trace_rows,
                log_k_chunk: self.one_hot_params.log_k_chunk,
                instruction_lookup_indices: &instruction_lookup_indices,
                untrusted_advice: (!self.program_io.untrusted_advice.is_empty())
                    .then_some(self.program_io.untrusted_advice.as_slice()),
            },
        )?;

        Ok(AkitaPackedWitnessProverData {
            protocol: committed.artifacts.protocol,
            precommitted,
            prover_setup,
            verifier_setup,
            committed,
        })
    }

    pub fn commit_akita_precommitted_program(
        &self,
        packed_witness: &AkitaPackedWitnessProverData,
    ) -> Result<AkitaPrecommittedProgramProverData, VerifierError>
    where
        PCS::Commitment: ark_serialize::CanonicalSerialize,
    {
        if packed_witness.precommitted.bytecode.is_none()
            || packed_witness.precommitted.program_image.is_none()
        {
            return Err(invalid_akita_prover_config(
                "Akita committed-program openings require bytecode and program-image schedules",
            ));
        }

        let shared = &self.preprocessing.shared;
        let committed = require_committed_program(&shared.program)?;
        let program = self.preprocessing.materialized_program();
        let bytecode_chunk_count = committed.bytecode_commitments.bytecode_chunk_count;
        let mut bytecode_chunk_commitments = Vec::with_capacity(bytecode_chunk_count);
        let mut opening_inputs = Vec::with_capacity(bytecode_chunk_count + 1);
        for polynomial in
            build_akita_bytecode_chunk_polynomials(&program.bytecode.bytecode, bytecode_chunk_count)
        {
            let opening =
                commit_akita_precommitted_polynomial(polynomial, &packed_witness.prover_setup.pcs);
            bytecode_chunk_commitments.push(opening.commitment.clone());
            opening_inputs.push(opening);
        }

        let program_image_polynomial = build_akita_program_image_polynomial(
            program,
            committed.program_commitments.program_image_num_words,
        );
        let program_image_opening = commit_akita_precommitted_polynomial(
            program_image_polynomial,
            &packed_witness.prover_setup.pcs,
        );
        let program_image_commitment = program_image_opening.commitment.clone();
        opening_inputs.push(program_image_opening);

        let verifier_program =
            VerifierProgramPreprocessing::Committed(VerifierCommittedProgramPreprocessing {
                meta: verifier_program_metadata(&committed.meta),
                memory_layout: shared.memory_layout.clone(),
                max_padded_trace_length: shared.max_padded_trace_length,
                bytecode_chunk_commitments,
                program_image_commitment,
            });
        let preprocessing = AkitaVerifierPreprocessing::new(
            verifier_program,
            shared.digest(),
            packed_witness.verifier_setup.clone(),
            None,
        );

        Ok(AkitaPrecommittedProgramProverData {
            preprocessing,
            opening_inputs,
        })
    }
}

pub fn derive_akita_packed_layout<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
    log_t: usize,
    log_k_chunk: usize,
    instruction_ra_count: usize,
    bytecode_ra_count: usize,
    ram_ra_count: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<jolt_openings::PackingWitnessLayout, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    require_committed_program(&shared.program)?;
    let config = lattice_layout_derivation_config(log_k_chunk, precommitted)?;
    let ra_layout =
        JoltRaPolynomialLayout::new(instruction_ra_count, bytecode_ra_count, ram_ra_count)
            .map_err(|error| invalid_akita_prover_config(error.to_string()))?;
    derive_lattice_packed_witness_layout(&config, log_t, log_k_chunk, ra_layout, precommitted)
}

pub fn derive_precommitted_schedule<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
    log_t: usize,
    log_k_chunk: usize,
    include_untrusted_advice: bool,
) -> Result<PrecommittedSchedule, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    let committed = committed_program_schedule(shared)?;
    PrecommittedSchedule::new(
        TracePolynomialOrder::CycleMajor,
        log_t,
        log_k_chunk,
        None,
        include_untrusted_advice.then_some(shared.memory_layout.max_untrusted_advice_size as usize),
        Some(committed),
    )
    .map_err(|error| VerifierError::InvalidPrecommittedSchedule {
        reason: error.to_string(),
    })
}

pub fn akita_packing_setup(
    layout: &jolt_openings::PackingWitnessLayout,
    max_num_polys_per_commitment_group: usize,
) -> (AkitaPackingProverSetup, AkitaPackingVerifierSetup) {
    akita_packing_setup_with_max_num_vars(
        layout,
        layout.dimension,
        max_num_polys_per_commitment_group,
    )
}

pub fn akita_packing_setup_with_max_num_vars(
    layout: &jolt_openings::PackingWitnessLayout,
    max_num_vars: usize,
    max_num_polys_per_commitment_group: usize,
) -> (AkitaPackingProverSetup, AkitaPackingVerifierSetup) {
    assert!(
        max_num_vars >= layout.dimension,
        "Akita setup max_num_vars ({max_num_vars}) must cover packed layout dimension ({})",
        layout.dimension
    );
    let params = PackingSetupParams {
        pcs: AkitaSetupParams::new(
            max_num_vars,
            max_num_polys_per_commitment_group,
            layout.digest,
        ),
        layout: layout.clone(),
    };
    let (pcs, verifier_pcs) = AkitaScheme::setup(params.pcs);
    (
        PackingProverSetup {
            pcs,
            layout: params.layout.clone(),
        },
        PackingVerifierSetup {
            pcs: verifier_pcs,
            layout: params.layout,
        },
    )
}

fn akita_precommitted_program_max_num_vars<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
) -> Result<usize, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    let committed = require_committed_program(&shared.program)?;
    let chunk_len = committed_lanes()
        * committed_bytecode_chunk_cycle_len(
            committed.meta.bytecode_len,
            committed.bytecode_commitments.bytecode_chunk_count,
        );
    let program_image_len = committed.program_commitments.program_image_num_words;
    Ok(akita_polynomial_num_vars(chunk_len)?.max(akita_polynomial_num_vars(program_image_len)?))
}

fn akita_polynomial_num_vars(len: usize) -> Result<usize, VerifierError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(invalid_akita_prover_config(format!(
            "Akita precommitted polynomial length must be a non-zero power of two, got {len}"
        )));
    }
    Ok(len.ilog2() as usize)
}

fn committed_program_schedule<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
) -> Result<CommittedProgramSchedule, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    let committed = require_committed_program(&shared.program)?;
    let program_image_start_index = shared
        .memory_layout
        .remapped_word_address(committed.meta.min_bytecode_address)
        .map_err(|error| VerifierError::InvalidCommittedProgram {
            reason: error.to_string(),
        })?;

    Ok(CommittedProgramSchedule {
        bytecode_len: committed.meta.bytecode_len,
        bytecode_chunk_count: committed.bytecode_commitments.bytecode_chunk_count,
        program_image_len_words: committed.meta.program_image_len_words,
        program_image_start_index: program_image_start_index as usize,
    })
}

fn require_committed_program<PCS>(
    program: &ProgramPreprocessing<PCS>,
) -> Result<&crate::zkvm::program::CommittedProgramPreprocessing<PCS>, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    match program {
        ProgramPreprocessing::Committed(committed) => Ok(committed),
        ProgramPreprocessing::Full(_) => Err(invalid_akita_prover_config(
            "Akita lattice mode requires committed program preprocessing",
        )),
    }
}

fn lattice_layout_derivation_config(
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<JoltProtocolConfig, VerifierError> {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice.program_mode = ProgramMode::Committed;
    config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
    config.lattice.advice.untrusted = precommitted.untrusted_advice.is_some();
    // Layout derivation only needs the lattice feature switches, but verifier
    // config validation also requires concrete packed-witness bindings.
    config.lattice.packed_witness.layout_digest = Some([0; 32]);
    config.lattice.packed_witness.d_pack = Some(0);
    config.lattice.packed_witness.validity_digest = Some([0; 32]);
    config.lattice.packed_witness.validity_digest = Some(lattice_packed_validity_digest(
        &derive_lattice_packed_validity_requirements(&config, log_k_chunk, precommitted)?,
    ));
    Ok(config)
}

fn instruction_lookup_indices(trace: &[Cycle]) -> Vec<u128> {
    trace
        .iter()
        .map(LookupQuery::<XLEN>::to_lookup_index)
        .collect()
}

fn build_akita_bytecode_chunk_polynomials(
    instructions: &[jolt_riscv::JoltInstructionRow],
    chunk_count: usize,
) -> Vec<Polynomial<AkitaField>> {
    let bytecode_len = instructions.len();
    let chunk_cycle_len = committed_bytecode_chunk_cycle_len(bytecode_len, chunk_count);
    let lane_capacity = committed_lanes();
    let mut chunks = (0..chunk_count)
        .map(|_| vec![AkitaField::default(); lane_capacity * chunk_cycle_len])
        .collect::<Vec<_>>();

    for (cycle, instruction) in instructions.iter().enumerate() {
        let chunk_index = cycle / chunk_cycle_len;
        let chunk_cycle = cycle % chunk_cycle_len;
        let coeffs = &mut chunks[chunk_index];
        for_each_active_lane_value::<JoltAkitaField>(instruction, |global_lane, lane_value| {
            let index = TracePolynomialOrder::CycleMajor.address_cycle_to_index(
                global_lane,
                chunk_cycle,
                lane_capacity,
                chunk_cycle_len,
            );
            let value = match lane_value {
                ActiveLaneValue::One => <AkitaField as FromPrimitiveInt>::from_u64(1),
                ActiveLaneValue::Scalar(value) => value.into_akita(),
            };
            coeffs[index] += value;
        });
    }

    chunks.into_iter().map(Polynomial::new).collect()
}

fn build_akita_program_image_polynomial(
    program: &crate::zkvm::program::FullProgramPreprocessing,
    padded_len: usize,
) -> Polynomial<AkitaField> {
    Polynomial::new(
        build_program_image_words_padded(program, padded_len)
            .into_iter()
            .map(<AkitaField as FromPrimitiveInt>::from_u64)
            .collect(),
    )
}

fn commit_akita_precommitted_polynomial(
    polynomial: Polynomial<AkitaField>,
    setup: &jolt_akita::AkitaProverSetup,
) -> AkitaOwnedPrecommittedOpening {
    let (commitment, hint) = AkitaScheme::commit(&polynomial, setup);
    AkitaOwnedPrecommittedOpening {
        commitment,
        polynomial,
        hint,
    }
}

fn verifier_program_metadata(
    meta: &crate::zkvm::program::ProgramMetadata,
) -> VerifierProgramMetadata {
    VerifierProgramMetadata {
        entry_address: meta.entry_address,
        min_bytecode_address: meta.min_bytecode_address,
        entry_bytecode_index: meta.entry_bytecode_index,
        program_image_len_words: meta.program_image_len_words,
        bytecode_len: meta.bytecode_len,
    }
}

fn invalid_akita_prover_config(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidProtocolConfig {
        reason: reason.into(),
    }
}

#[cfg(all(test, feature = "host"))]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::unwrap_used,
        reason = "tests assert successful prover artifact construction"
    )]

    use serial_test::serial;

    use super::*;
    use crate::{
        host,
        poly::commitment::dory::DoryGlobals,
        zkvm::{
            preprocessing::JoltSharedPreprocessing, program::ProgramPreprocessing,
            prover::JoltProverPreprocessing, RV64IMACProver,
        },
    };

    #[test]
    #[serial]
    fn muldiv_committed_program_builds_akita_packed_witness() {
        DoryGlobals::reset();
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
        let program_preprocessing =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address).unwrap();
        let (shared, committed_program_prover_data, generators) =
            JoltSharedPreprocessing::new_committed(
                program_preprocessing,
                io_device.memory_layout,
                1 << 16,
                1,
            );
        let prover_preprocessing = JoltProverPreprocessing::new_committed(
            shared,
            committed_program_prover_data,
            generators,
        );
        let elf_contents = program
            .get_elf_contents()
            .expect("test program should provide ELF contents");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );

        let data = prover
            .commit_akita_packed_witness()
            .expect("Akita packed witness should commit");
        let payload = data
            .committed
            .artifacts
            .payload()
            .expect("Akita artifacts should carry a lattice payload");

        assert_eq!(data.protocol.pcs, PcsFamily::Lattice);
        assert_eq!(
            payload.layout_digest,
            data.committed.artifacts.layout.digest
        );
        assert_eq!(payload.d_pack, data.committed.artifacts.layout.dimension);
        assert!(data.precommitted.bytecode.is_some());
        assert!(data.precommitted.program_image.is_some());

        let precommitted = prover
            .commit_akita_precommitted_program(&data)
            .expect("Akita precommitted program should commit");
        let verifier_committed = precommitted
            .preprocessing
            .program
            .committed()
            .expect("Akita preprocessing should use committed program mode");

        assert_eq!(
            precommitted.preprocessing.pcs_setup.layout.dimension,
            payload.d_pack
        );
        assert!(
            precommitted.preprocessing.pcs_setup.pcs.max_num_vars >= payload.d_pack,
            "Akita setup should cover packed and precommitted opening dimensions"
        );
        assert_eq!(
            verifier_committed.bytecode_chunk_commitments.len(),
            prover.preprocessing.shared.bytecode_chunk_count
        );
        assert_eq!(
            precommitted.opening_inputs.len(),
            prover.preprocessing.shared.bytecode_chunk_count + 1
        );
        for (opening, commitment) in precommitted
            .opening_inputs
            .iter()
            .zip(verifier_committed.bytecode_chunk_commitments.iter())
        {
            assert_eq!(&opening.commitment, commitment);
            assert!(opening.hint.matches_commitment(commitment));
        }
        let program_image_opening = precommitted
            .opening_inputs
            .last()
            .expect("program image opening should follow bytecode chunks");
        assert_eq!(
            program_image_opening.commitment,
            verifier_committed.program_image_commitment
        );
        assert!(program_image_opening
            .hint
            .matches_commitment(&verifier_committed.program_image_commitment));
        let borrowed_inputs = precommitted.opening_inputs();
        assert_eq!(borrowed_inputs.len(), precommitted.opening_inputs.len());
    }
}
