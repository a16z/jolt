//! Akita prover-side artifact construction.
//!
//! This module is intentionally feature-gated and keeps Akita-specific setup
//! native to the modular verifier/opening crates. The legacy prover still owns
//! trace execution and sumcheck proving; Akita artifacts are derived from those
//! prover-native inputs without going through deleted core/compat paths.

use common::constants::XLEN;
use jolt_akita::{AkitaScheme, AkitaSetupParams};
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    lattice_packed_validity_digest,
};
use jolt_openings::{
    CommitmentScheme, PackingProverSetup, PackingSetupParams, PackingVerifierSetup,
};
use jolt_verifier::{
    akita::{
        commit_akita_packing_jolt_witness, AkitaCommittedPackedJoltWitness,
        AkitaPackingJoltWitnessInput, AkitaPackingProverSetup, AkitaPackingVerifierSetup,
    },
    config::{IncrementCommitmentMode, JoltProtocolConfig, PcsFamily, ProgramMode},
    stages::{
        stage8::{
            derive_lattice_packed_validity_requirements, derive_lattice_packed_witness_layout,
        },
        CommittedProgramSchedule, PrecommittedSchedule,
    },
    VerifierError,
};
use tracer::{build_trace_rows, instruction::Cycle};

use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment},
    transcripts::Transcript,
    zkvm::{
        instruction::LookupQuery, preprocessing::JoltSharedPreprocessing,
        program::ProgramPreprocessing, prover::JoltCpuProver,
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
        let (prover_setup, verifier_setup) = akita_packing_setup(&layout, 1);
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
    let params = PackingSetupParams {
        pcs: AkitaSetupParams::new(
            layout.dimension,
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
    }
}
