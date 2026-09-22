use std::{mem::size_of, path::PathBuf, sync::Arc, time::Instant};

use common::constants::RAM_START_ADDRESS;
use jolt_akita::schedule_registry::provision_precommitted_for_k;
use jolt_openings::CommitmentScheme;
use jolt_program::{
    build_jolt_program_with_inline_provider,
    execution::{FieldInlineTraceData, OwnedTrace, TraceInputs, TraceOutput, TraceRow},
};
use jolt_prover::{
    akita::{self, JoltAkitaBackend},
    JoltProverPreprocessing, ProverConfig,
};
use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;
use jolt_sdk::{
    jolt_prover_legacy::{
        field::akita::AkitaFp128,
        zkvm::{
            packed::{
                akita_verifier_preprocessing, field_inc_limb_schedule_params,
                field_inline_one_hot_trace_setup_params, AkitaField, AkitaNoCurve,
                AkitaPackedScheme, AkitaScheduleArtifacts, AkitaScheme, AkitaTranscript, AkitaVc,
            },
            preprocessing::JoltSharedPreprocessing,
            program::ProgramPreprocessing,
            prover::JoltProverPreprocessing as LegacyPreprocessing,
        },
    },
    jolt_verifier, MemoryConfig,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use serde_json::json;
use tracer::{execution_backend::TracerBackend, TracerInlineExpansionProvider};
use tracing::info;

/// The ELF and input are already built artifacts. This command never rewrites guest sources.
#[derive(clap::Args)]
pub(super) struct Args {
    #[arg(long)]
    elf: PathBuf,
    /// Exact guest entry-point bytes, already framed by the guest's caller.
    #[arg(long)]
    input: PathBuf,
    /// New output directory; an existing directory is rejected.
    #[arg(long)]
    workdir: PathBuf,
    /// Derive geometry and provision schedule rows, then stop before PCS setup.
    #[arg(long)]
    preflight: bool,
    /// Row storage reserved up front and the maximum padded proof geometry.
    #[arg(long)]
    max_trace_length: usize,
    #[arg(long, default_value_t = 16_000_000)]
    max_input_size: u64,
    #[arg(long, default_value_t = 4096)]
    max_output_size: u64,
    /// Reserved guest I/O capacity, even when no advice is supplied. Must match the ELF.
    #[arg(long)]
    max_untrusted_advice_size: u64,
    /// Reserved guest I/O capacity, even when no advice is supplied. Must match the ELF.
    #[arg(long)]
    max_trusted_advice_size: u64,
    #[arg(long, default_value_t = 134_217_728)]
    heap_size: u64,
    #[arg(long, default_value_t = 33_554_432)]
    stack_size: u64,
}

impl Args {
    pub(super) fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        for capacity in [self.max_untrusted_advice_size, self.max_trusted_advice_size] {
            if capacity != 0 && !capacity.is_power_of_two() {
                return Err("reserved advice capacities must be zero or powers of two".into());
            }
        }
        if let Some(parent) = self.workdir.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::create_dir(&self.workdir)?;
        let started = Instant::now();
        let mut program = build_jolt_program_with_inline_provider(
            &std::fs::read(&self.elf)?,
            &mut TracerInlineExpansionProvider::new(),
            RV64IMAC_JOLT_FIELD_INLINE,
        )?;
        let memory_config = MemoryConfig {
            max_input_size: self.max_input_size,
            max_output_size: self.max_output_size,
            heap_size: self.heap_size,
            stack_size: self.stack_size,
            program_size: Some(program.program_end - RAM_START_ADDRESS),
            max_untrusted_advice_size: self.max_untrusted_advice_size,
            max_trusted_advice_size: self.max_trusted_advice_size,
        };
        let input = std::fs::read(&self.input)?;
        if input.len() > self.max_input_size as usize {
            return Err("guest input exceeds configured maximum".into());
        }
        info!("Streaming pinned ELF into bounded modular row storage");
        let trace_output = TracerBackend::with_elf_path(self.elf.clone()).trace_streaming(
            &program,
            TraceInputs::new(input, Vec::new(), Vec::new(), memory_config),
            self.max_trace_length,
        )?;
        if trace_output.device.panic {
            return Err("guest panicked".into());
        }
        let preprocessed = ProgramPreprocessing::<AkitaPackedScheme>::preprocess_with_profile(
            std::mem::take(&mut program.expanded_bytecode),
            std::mem::take(&mut program.memory_init),
            program.entry_address,
            program.profile,
        )?;
        let program = Arc::new(program);
        let shared = JoltSharedPreprocessing::new(
            preprocessed,
            trace_output.device.memory_layout.clone(),
            self.max_trace_length,
        );
        let legacy: LegacyPreprocessing<AkitaFp128, AkitaNoCurve, AkitaPackedScheme> =
            LegacyPreprocessing::new(shared);
        let config = ProverConfig::derive::<AkitaField>(
            trace_output.trace.rows(),
            &trace_output.device.memory_layout,
            legacy.shared.program_meta.min_bytecode_address,
            legacy.shared.program.program_image_len_words(),
            self.max_trace_length,
        )?;
        // Ownership is transferred, so OwnedTrace::into_rows does not clone the trace.
        let mut rows = trace_output.trace.into_rows();
        let (fr_rows, virtual_sequence_rows) =
            rows.iter()
                .fold((0usize, 0usize), |(fr, virtual_rows), row| {
                    (
                        fr + usize::from(row.field_inline.is_some()),
                        virtual_rows
                            + usize::from(row.instruction().virtual_sequence_remaining.is_some()),
                    )
                });
        let (shape, digest, one_hot_k) =
            akita::one_hot_trace_setup_shape(&config, legacy.shared.bytecode_size())?;
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let params = field_inline_one_hot_trace_setup_params(
            shape.num_vars,
            shape.num_polys,
            digest,
            one_hot_k,
            artifacts.clone(),
        )?;
        let admission = provision_precommitted_for_k(
            &artifacts.dense_catalog()?,
            &artifacts.one_hot_catalog(one_hot_k)?,
            None,
            None,
            &[],
            Some(field_inc_limb_schedule_params(one_hot_k)?),
            one_hot_k,
            shape.num_vars,
        );
        let report = json!({
            "elf": self.elf, "preflight_only": self.preflight,
            "memory_config": format!("{memory_config:?}"),
            "max_untrusted_advice_size": self.max_untrusted_advice_size,
            "max_trusted_advice_size": self.max_trusted_advice_size,
            "trace_rows": rows.len(), "padded_rows": config.trace_length,
            "ram_k": config.ram_K, "config": format!("{config:?}"),
            "fr_rows": fr_rows, "virtual_sequence_rows": virtual_sequence_rows,
            "row_size_bytes": size_of::<TraceRow>(),
            "trace_collection": "serial lazy cycles converted directly into pre-reserved modular rows",
            "row_limit": self.max_trace_length,
            "full_cycle_vector_materialized": false,
            "row_capacity": rows.capacity(),
            "row_storage_bytes": rows.capacity() * size_of::<TraceRow>(),
            "padded_row_storage_bytes": config.trace_length * size_of::<TraceRow>(),
            "fr_payload_size_bytes": size_of::<FieldInlineTraceData>(),
            "fr_payload_storage_upper_estimate_bytes": fr_rows * (size_of::<FieldInlineTraceData>() + 2 * size_of::<usize>()),
            "storage_estimate_excludes": "not a peak RSS estimate: excludes emulator/decode state, per-tick cycle scratch, final-memory extraction overlap, allocator overhead, preprocessing, PCS and witness/sumcheck buffers; FR estimate counts each occupied row as a separate Arc allocation",
            "one_hot_k": one_hot_k, "setup_num_vars": shape.num_vars,
            "setup_num_polys": shape.num_polys,
            "catalog_provisioning": match &admission { Ok(rows) => format!("accepted: {} grouped rows", rows.rows().len()), Err(error) => format!("rejected: {error}") },
            "pcs_setup_performed": false,
            "guest_output_bytes": trace_output.device.outputs,
            "elapsed_before_setup_s": started.elapsed().as_secs_f64(),
        });
        std::fs::write(
            self.workdir.join("preflight.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        info!("Preflight: {report}");
        admission?;
        if self.preflight {
            return Ok(());
        }
        info!("Entering PCS setup");
        let (pcs_setup, verifier_setup) = AkitaScheme::setup(params)?;
        let verifier = akita_verifier_preprocessing(&legacy, verifier_setup, None);
        drop(legacy);
        let program_preprocessing = verifier
            .program
            .as_full_arc()
            .ok_or("full program required")?;
        let public_io = trace_output.device.clone();
        rows.reserve_exact(config.trace_length - rows.len());
        rows.resize(config.trace_length, TraceRow::default());
        let padded = TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device,
            trace_output.final_memory,
            trace_output.advice_tape,
        );
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&program, &program_preprocessing, padded),
        )
        .with_field_inline()?;
        let preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier,
            pcs_setup,
            committed_program: None,
        };
        info!("Proving with optimized modular q128 Akita/FR backend");
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(),
            &preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )?;
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &preprocessing.verifier,
            &public_io,
            &proof,
            None,
        )?;
        let staging = self.workdir.join(".proof-incomplete");
        std::fs::create_dir(&staging)?;
        std::fs::write(
            staging.join("outer-proof.bin"),
            bincode::serde::encode_to_vec(&proof, bincode::config::standard())?,
        )?;
        std::fs::write(
            staging.join("outer-device.bin"),
            bincode::serde::encode_to_vec(&public_io, bincode::config::standard())?,
        )?;
        std::fs::rename(&staging, self.workdir.join("accepted-proof"))?;
        info!(
            "Full outer verification accepted; elapsed {} s",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }
}
