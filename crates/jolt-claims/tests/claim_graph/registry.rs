//! The vertex set: every relation's `ProtocolVertex` impl and the
//! `claim_graph_vertices!` manifest, sectioned by PIOP membership.
//!
//! `registration` is a wildcard-free match over `JoltRelationId`, so adding a
//! relation variant fails compilation here until it is classified; the
//! exhaustiveness test then requires it to be `Registered` (or documented).
//!
//! Shape construction mirrors `jolt-verifier`'s stage modules (cited per
//! helper/impl) using only jolt-claims' public API.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::claim_reductions::advice as advice_geometry;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode as bytecode_reduction_geometry;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::geometry::claim_reductions::program_image as program_image_geometry;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::ram as ram_geometry;
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::{
    SpartanOuterDimensions, SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::lattice::relations::advice_reconstruction::{
    AdviceReconstructionDimensions, TrustedAdviceReconstruction, UntrustedAdviceReconstruction,
};
use jolt_claims::protocols::jolt::lattice::relations::booleanity::{
    LatticeBooleanity, LatticeBooleanityCyclePhase, LatticeBooleanityDimensions,
};
use jolt_claims::protocols::jolt::lattice::relations::bytecode_reconstruction::{
    BytecodeChunkReconstruction, BytecodeReconstructionDimensions,
};
use jolt_claims::protocols::jolt::lattice::relations::hamming_weight::{
    LatticeHammingWeightClaimReduction, LatticeHammingWeightClaimReductionDimensions,
};
use jolt_claims::protocols::jolt::lattice::relations::program_image_reconstruction::ProgramImageReconstruction;
use jolt_claims::protocols::jolt::lattice::relations::read_raf::{
    LatticeReadRafAddressPhase, LatticeReadRafCyclePhase, LatticeReadRafCyclePhaseCommitted,
};
use jolt_claims::protocols::jolt::relations::booleanity::{
    Booleanity, BooleanityAddressPhase, BooleanityCyclePhase,
};
use jolt_claims::protocols::jolt::relations::bytecode::{
    ReadRaf as BytecodeReadRaf, ReadRafAddressPhase as BytecodeReadRafAddressPhase,
    ReadRafCyclePhase as BytecodeReadRafCyclePhase,
    ReadRafCyclePhaseCommitted as BytecodeReadRafCyclePhaseCommitted,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
    TrustedAddressPhase as TrustedAdviceAddressPhase, TrustedCyclePhase as TrustedAdviceCyclePhase,
    UntrustedAddressPhase as UntrustedAdviceAddressPhase,
    UntrustedCyclePhase as UntrustedAdviceCyclePhase,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
    AddressPhase as BytecodeReductionAddressPhase, CyclePhase as BytecodeReductionCyclePhase,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::ClaimReduction as HammingWeightClaimReduction;
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::ClaimReduction as IncClaimReduction;
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::ClaimReduction as InstructionClaimReduction;
use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
    AddressPhase as ProgramImageReductionAddressPhase,
    CyclePhase as ProgramImageReductionCyclePhase,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::registers::ClaimReduction as RegistersClaimReduction;
use jolt_claims::protocols::jolt::relations::instruction::{
    InputVirtualization as InstructionInputVirtualization,
    RaVirtualization as InstructionRaVirtualization, ReadRaf as InstructionReadRaf,
};
use jolt_claims::protocols::jolt::relations::ram::{
    HammingBooleanity as RamHammingBooleanity, OutputCheck as RamOutputCheck,
    RaClaimReduction as RamRaClaimReduction, RaVirtualization as RamRaVirtualization,
    RafEvaluation as RamRafEvaluation, RamValCheck, RamValCheckShape, RamValContribution,
    ReadWriteChecking as RamReadWriteChecking,
};
use jolt_claims::protocols::jolt::relations::registers::{
    ReadWriteChecking as RegistersReadWriteChecking, ValEvaluation as RegistersValEvaluation,
};
use jolt_claims::protocols::jolt::relations::spartan::{
    OuterRemainder, OuterUniskip, ProductRemainder, ProductUniskip, Shift,
};
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, CommitmentMatrixShape,
    JoltAdviceKind, JoltRelationId, PrecommittedClaimReduction, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout, RamValCheckPublic, ReadWriteDimensions, TraceDimensions,
};
use jolt_field::{FixedByteSize, Fr};

use super::{Piop, ProtocolConfig, ProtocolVertex, VertexRecord};

fn spartan_outer_dimensions(config: &ProtocolConfig) -> SpartanOuterDimensions {
    SpartanOuterDimensions::rv64(config.log_t)
}

/// `stage6a/verify.rs` / `stage6b/batch.rs`: `BooleanityDimensions::new(
/// formula.ra_layout, log_t, one_hot_config.committed_chunk_bits())`.
fn booleanity_dimensions(config: &ProtocolConfig) -> BooleanityDimensions {
    BooleanityDimensions::new(config.formula.ra_layout, config.log_t, config.log_k_chunk)
}

/// `stage2/verify.rs`: `proof.rw_config.ram_dimensions(log_t, ram_K.ilog2())`.
fn ram_read_write_dimensions(config: &ProtocolConfig) -> ReadWriteDimensions {
    config
        .read_write
        .ram_dimensions(config.log_t, config.ram_log_k)
}

/// `stage4/verify.rs`: `proof.rw_config.register_dimensions(log_t,
/// REGISTER_ADDRESS_BITS)`.
fn register_read_write_dimensions(config: &ProtocolConfig) -> ReadWriteDimensions {
    config
        .read_write
        .register_dimensions(config.log_t, REGISTER_ADDRESS_BITS)
}

/// Committed bytecode is padded to a power of two, so its length is
/// recoverable from the formula's bytecode `log_k`.
fn committed_bytecode_len(config: &ProtocolConfig) -> usize {
    1usize << config.formula.bytecode_read_raf.log_k()
}

/// Per-polynomial precommitted claim-reduction layouts over the shared
/// scheduling reference — mirrors `jolt-verifier`'s
/// `PrecommittedSchedule::new` (`stages/mod.rs`), candidate order included
/// (trusted advice, untrusted advice, committed bytecode, program image).
struct PrecommittedLayouts {
    trusted_advice: Option<AdviceClaimReductionLayout>,
    untrusted_advice: Option<AdviceClaimReductionLayout>,
    bytecode: Option<BytecodeClaimReductionLayout>,
    program_image: Option<ProgramImageClaimReductionLayout>,
}

fn precommitted_layouts(config: &ProtocolConfig) -> PrecommittedLayouts {
    let trusted_bytes = config.trusted_advice.then_some(config.advice_max_bytes);
    let untrusted_bytes = config.untrusted_advice.then_some(config.advice_max_bytes);
    let mut candidates = advice_geometry::candidate_total_vars(trusted_bytes, untrusted_bytes);
    let bytecode_len = committed_bytecode_len(config);
    if let Some(chunk_count) = config.committed_program_chunks {
        candidates.push(
            bytecode_reduction_geometry::precommitted_candidate(bytecode_len, chunk_count)
                .expect("committed bytecode chunking is valid"),
        );
        candidates.push(program_image_geometry::precommitted_candidate(
            config.program_image_len_words,
        ));
    }
    let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
        config.log_t + config.log_k_chunk,
        &candidates,
        config.log_k_chunk,
    );
    let advice_layout = |max_bytes: Option<usize>| {
        max_bytes.map(|max_bytes| {
            AdviceClaimReductionLayout::balanced(
                config.trace_order,
                config.log_t,
                scheduling_reference,
                max_bytes,
            )
            .expect("advice reduction layout builds")
        })
    };
    PrecommittedLayouts {
        trusted_advice: advice_layout(trusted_bytes),
        untrusted_advice: advice_layout(untrusted_bytes),
        bytecode: config.committed_program_chunks.map(|chunk_count| {
            BytecodeClaimReductionLayout::balanced(
                config.trace_order,
                config.log_t,
                scheduling_reference,
                bytecode_len,
                chunk_count,
            )
            .expect("committed bytecode reduction layout builds")
        }),
        // `start_index` positions the image inside RAM; the reduction
        // dimensions do not depend on it, so 0 suffices for shape expansion.
        program_image: config.committed_program_chunks.map(|_| {
            ProgramImageClaimReductionLayout::balanced(
                config.trace_order,
                config.log_t,
                scheduling_reference,
                config.program_image_len_words,
                0,
            )
            .expect("program image reduction layout builds")
        }),
    }
}

impl ProtocolVertex for OuterUniskip {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![spartan_outer_dimensions(config)]
    }
}

impl ProtocolVertex for OuterRemainder {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![spartan_outer_dimensions(config)]
    }
}

impl ProtocolVertex for ProductUniskip {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![SpartanProductDimensions::new(config.log_t)]
    }
}

impl ProtocolVertex for ProductRemainder {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![SpartanProductDimensions::new(config.log_t)]
    }
}

impl ProtocolVertex for Shift {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![TraceDimensions::new(config.log_t)]
    }
}

/// `stage2/verify.rs`: `InstructionClaimReduction::new(trace_dimensions, ..)`.
impl ProtocolVertex for InstructionClaimReduction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage3/verify.rs`: `InstructionInput::new(dimensions, ..)`.
impl ProtocolVertex for InstructionInputVirtualization {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage5/verify.rs`: `InstructionReadRaf::new(formula.instruction_read_raf)`.
impl ProtocolVertex for InstructionReadRaf {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.instruction_read_raf]
    }
}

/// `stage6b/batch.rs`: `InstructionRaVirtualization::new(
/// formula.instruction_ra_virtualization, ..)`.
impl ProtocolVertex for InstructionRaVirtualization {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.instruction_ra_virtualization]
    }
}

/// `stage2/verify.rs`: `RamReadWriteChecking::new(read_write_dimensions, ..)`.
impl ProtocolVertex for RamReadWriteChecking {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![ram_read_write_dimensions(config)]
    }
}

/// `stage2/verify.rs`: `RamRafEvaluationDimensions::try_from(read_write_dimensions)`.
impl ProtocolVertex for RamRafEvaluation {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![
            RamRafEvaluationDimensions::try_from(ram_read_write_dimensions(config))
                .expect("RAM RAF evaluation dimensions are valid"),
        ]
    }
}

/// `stage2/verify.rs`: `RamOutputCheck::new(read_write_dimensions, ..)`.
impl ProtocolVertex for RamOutputCheck {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![ram_read_write_dimensions(config)]
    }
}

/// `stage4/ram_val_check.rs`: trace dimensions plus the `Val_init`
/// contributions in `ram_val_check_init_structure` order — program image
/// first, then untrusted, then trusted advice.
impl ProtocolVertex for RamValCheck {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        let mut contributions = Vec::new();
        if config.committed_program_chunks.is_some() {
            contributions.push(RamValContribution {
                selector: RamValCheckPublic::InitSelectorProgramImage,
                opening: program_image_geometry::ram_val_check_contribution_opening(),
            });
        }
        if config.untrusted_advice {
            contributions.push(RamValContribution {
                selector: RamValCheckPublic::InitSelector(JoltAdviceKind::Untrusted),
                opening: ram_geometry::val_check_advice_opening(JoltAdviceKind::Untrusted),
            });
        }
        if config.trusted_advice {
            contributions.push(RamValContribution {
                selector: RamValCheckPublic::InitSelector(JoltAdviceKind::Trusted),
                opening: ram_geometry::val_check_advice_opening(JoltAdviceKind::Trusted),
            });
        }
        vec![RamValCheckShape {
            dimensions: config.formula.trace,
            contributions,
        }]
    }
}

/// `stage5/verify.rs`: `RamRaClaimReduction::new(trace_dimensions, ..)`.
impl ProtocolVertex for RamRaClaimReduction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage6b/ram_hamming_booleanity.rs`: `HammingBooleanity::new(trace_dimensions)`.
impl ProtocolVertex for RamHammingBooleanity {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage6b/batch.rs`: `RamRaVirtualization::new(formula.ram_ra_virtualization, ..)`.
impl ProtocolVertex for RamRaVirtualization {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.ram_ra_virtualization]
    }
}

/// `stage3/verify.rs`: `RegistersClaimReduction::new(dimensions, ..)`.
impl ProtocolVertex for RegistersClaimReduction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage4/verify.rs`: `RegistersReadWriteChecking::new(register_dimensions)`.
impl ProtocolVertex for RegistersReadWriteChecking {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![register_read_write_dimensions(config)]
    }
}

/// `stage5/verify.rs`: `RegistersValEvaluation::new(trace_dimensions)`.
impl ProtocolVertex for RegistersValEvaluation {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage6a/verify.rs`: `BooleanityAddressPhase::new(booleanity_dimensions)`.
/// The address phase is column-agnostic, so it serves both PIOPs (see
/// `lattice/relations/booleanity.rs`).
impl ProtocolVertex for BooleanityAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![booleanity_dimensions(config)]
    }
}

/// The monolithic booleanity relation is never instantiated by the verifier
/// (it always runs the address/cycle phase split); registering it with
/// instances would double-produce the cycle phase's openings.
impl ProtocolVertex for Booleanity {
    fn instances(_config: &ProtocolConfig) -> Vec<Self::Shape> {
        Vec::new()
    }
}

/// `stage6b/booleanity.rs` (base build): `BooleanityCyclePhase::new(
/// booleanity_dimensions)`.
impl ProtocolVertex for BooleanityCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![booleanity_dimensions(config)]
    }
}

/// The monolithic bytecode read-RAF is instantiated by the verifier only as
/// an expression-evaluation helper inside the cycle phase's expected output
/// (`stage6b/bytecode_read_raf.rs`); the graph vertices are the phase splits.
impl ProtocolVertex for BytecodeReadRaf {
    fn instances(_config: &ProtocolConfig) -> Vec<Self::Shape> {
        Vec::new()
    }
}

/// `stage6a/bytecode_read_raf.rs` (base build): `ReadRafAddressPhase::new(
/// formula.bytecode_read_raf)`.
impl ProtocolVertex for BytecodeReadRafAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.bytecode_read_raf]
    }
}

/// `stage6b/bytecode_read_raf.rs` (base build, full-program dispatch):
/// `ReadRafCyclePhase::new((dimensions, NUM_BYTECODE_VAL_STAGES))`.
impl ProtocolVertex for BytecodeReadRafCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if config.committed_program_chunks.is_some() {
            return Vec::new();
        }
        vec![(config.formula.bytecode_read_raf, NUM_BYTECODE_VAL_STAGES)]
    }
}

/// `stage6b/bytecode_read_raf.rs` (base build, committed-program dispatch):
/// `ReadRafCyclePhaseCommitted::new((dimensions, NUM_BYTECODE_VAL_STAGES))`.
impl ProtocolVertex for BytecodeReadRafCyclePhaseCommitted {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if config.committed_program_chunks.is_none() {
            return Vec::new();
        }
        vec![(config.formula.bytecode_read_raf, NUM_BYTECODE_VAL_STAGES)]
    }
}

/// `stage6b/inc_claim_reduction.rs`: `IncClaimReduction::new(trace_dimensions, ..)`.
impl ProtocolVertex for IncClaimReduction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.trace]
    }
}

/// `stage7/verify.rs` (base build): `HammingWeightClaimReductionDimensions::new(
/// formula.ra_layout, committed_chunk_bits)`.
impl ProtocolVertex for HammingWeightClaimReduction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![HammingWeightClaimReductionDimensions::new(
            config.formula.ra_layout,
            config.log_k_chunk,
        )]
    }
}

/// `stage6b/committed_reduction_cycle_phase.rs`: `TrustedCyclePhase::new(
/// layout.dimensions())`, present when the trusted advice layout exists.
impl ProtocolVertex for TrustedAdviceCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .trusted_advice
            .map(|layout| layout.dimensions())
            .into_iter()
            .collect()
    }
}

impl ProtocolVertex for UntrustedAdviceCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .untrusted_advice
            .map(|layout| layout.dimensions())
            .into_iter()
            .collect()
    }
}

/// `stage7/verify.rs`: address-phase members additionally require active
/// address rounds (`layout.dimensions().has_address_phase()`).
impl ProtocolVertex for TrustedAdviceAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .trusted_advice
            .map(|layout| layout.dimensions())
            .filter(|dimensions| dimensions.has_address_phase())
            .into_iter()
            .collect()
    }
}

impl ProtocolVertex for UntrustedAdviceAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .untrusted_advice
            .map(|layout| layout.dimensions())
            .filter(|dimensions| dimensions.has_address_phase())
            .into_iter()
            .collect()
    }
}

/// `stage6b/committed_reduction_cycle_phase.rs`: `bytecode::CyclePhase::new((
/// layout.dimensions(), layout.chunk_count()))`, committed-program only.
impl ProtocolVertex for BytecodeReductionCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .bytecode
            .map(|layout| (layout.dimensions(), layout.chunk_count()))
            .into_iter()
            .collect()
    }
}

/// `stage7/committed_reduction_address_phase.rs`: same shape, gated on
/// `has_address_phase()`.
impl ProtocolVertex for BytecodeReductionAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .bytecode
            .map(|layout| (layout.dimensions(), layout.chunk_count()))
            .filter(|(dimensions, _)| dimensions.has_address_phase())
            .into_iter()
            .collect()
    }
}

/// `stage6b/committed_reduction_cycle_phase.rs`: `program_image::CyclePhase::new(
/// layout.dimensions())`, committed-program only.
impl ProtocolVertex for ProgramImageReductionCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .program_image
            .map(|layout| layout.dimensions())
            .into_iter()
            .collect()
    }
}

impl ProtocolVertex for ProgramImageReductionAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        precommitted_layouts(config)
            .program_image
            .map(|layout| layout.dimensions())
            .filter(|dimensions| dimensions.has_address_phase())
            .into_iter()
            .collect()
    }
}

/// Like the base monolith, the lattice full booleanity is not instantiated by
/// the verifier (the akita build runs `BooleanityAddressPhase` +
/// `LatticeBooleanityCyclePhase`).
impl ProtocolVertex for LatticeBooleanity {
    fn instances(_config: &ProtocolConfig) -> Vec<Self::Shape> {
        Vec::new()
    }
}

/// `stage6b/batch.rs` (akita build): `LatticeBooleanityDimensions::new(
/// booleanity_dimensions)`.
impl ProtocolVertex for LatticeBooleanityCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![
            LatticeBooleanityDimensions::new(booleanity_dimensions(config))
                .expect("lattice booleanity chunking is valid"),
        ]
    }
}

/// `stage6a/bytecode_read_raf.rs` (akita build): same
/// `formula.bytecode_read_raf` shape as the base address phase.
impl ProtocolVertex for LatticeReadRafAddressPhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![config.formula.bytecode_read_raf]
    }
}

/// `stage6b/bytecode_read_raf.rs` (akita build, full-program dispatch).
impl ProtocolVertex for LatticeReadRafCyclePhase {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if config.committed_program_chunks.is_some() {
            return Vec::new();
        }
        vec![config.formula.bytecode_read_raf]
    }
}

/// `stage6b/bytecode_read_raf.rs` (akita build, committed-program dispatch).
impl ProtocolVertex for LatticeReadRafCyclePhaseCommitted {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if config.committed_program_chunks.is_none() {
            return Vec::new();
        }
        vec![config.formula.bytecode_read_raf]
    }
}

/// `stage7/verify.rs` (akita build):
/// `LatticeHammingWeightClaimReductionDimensions::new(layout, log_k_chunk)`.
impl ProtocolVertex for LatticeHammingWeightClaimReduction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        vec![LatticeHammingWeightClaimReductionDimensions::new(
            config.formula.ra_layout,
            config.log_k_chunk,
        )
        .expect("lattice hamming chunking is valid")]
    }
}

/// `stage8/reconstruction.rs`: `AdviceReconstructionDimensions { word_vars:
/// layout.advice_shape().total_vars() }`, present with untrusted advice.
impl ProtocolVertex for UntrustedAdviceReconstruction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if !config.untrusted_advice {
            return Vec::new();
        }
        vec![AdviceReconstructionDimensions {
            word_vars: CommitmentMatrixShape::advice_from_max_bytes(config.advice_max_bytes)
                .total_vars(),
        }]
    }
}

/// `stage8/reconstruction.rs`: `TrustedSymbolic::new(())`, present with
/// trusted advice.
impl ProtocolVertex for TrustedAdviceReconstruction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if !config.trusted_advice {
            return Vec::new();
        }
        vec![()]
    }
}

/// `stage8/reconstruction.rs`: present in committed-program mode.
impl ProtocolVertex for ProgramImageReconstruction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        if config.committed_program_chunks.is_none() {
            return Vec::new();
        }
        vec![()]
    }
}

/// `stage8/reconstruction.rs`: `BytecodeReconstructionDimensions { chunks,
/// imm_byte_width: <F as FixedByteSize>::NUM_BYTES }`, committed-program only.
impl ProtocolVertex for BytecodeChunkReconstruction {
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape> {
        config
            .committed_program_chunks
            .map(|chunks| BytecodeReconstructionDimensions {
                chunks,
                imm_byte_width: <Fr as FixedByteSize>::NUM_BYTES,
            })
            .into_iter()
            .collect()
    }
}

/// One manifest type's records: its instances under the configuration, or —
/// when the configuration yields none (advice / committed-program
/// conditionals, and the never-instantiated monoliths) — a single
/// declaration-only record so the exhaustiveness backstop still sees the
/// manifest entry.
macro_rules! manifest_entry {
    ($records:ident, $config:ident, $ty:ty, $graphs:expr) => {{
        let batch = super::vertices::<$ty>($config, $graphs);
        if batch.is_empty() {
            $records.push(super::declaration_only::<$ty>());
        } else {
            $records.extend(batch);
        }
    }};
}

macro_rules! claim_graph_vertices {
    ($config:expr, {
        shared: [$($shared:ty),* $(,)?],
        dory: [$($dory:ty),* $(,)?],
        akita: [$($akita:ty),* $(,)?] $(,)?
    }) => {{
        let config: &ProtocolConfig = $config;
        let mut records: Vec<VertexRecord> = Vec::new();
        $(manifest_entry!(records, config, $shared, &[Piop::Dory, Piop::Akita]);)*
        $(manifest_entry!(records, config, $dory, &[Piop::Dory]);)*
        $(manifest_entry!(records, config, $akita, &[Piop::Akita]);)*
        records
    }};
}

/// The full vertex set for one configuration.
///
/// Section reasoning:
/// - `shared`: relations the verifier instantiates in both builds — the
///   Spartan stack, instruction/RAM/registers relations, the booleanity
///   address phase (column-agnostic, reused by the akita build), and the base
///   precommitted claim reductions (advice / committed bytecode / program
///   image), whose termini the akita reconstructions consume
///   (`specs/lattice-claims.md`, "Packed-Column Reconstructions").
/// - `dory`: the base booleanity cycle sumchecks, base bytecode read-RAF
///   phases, the standalone increment claim reduction (fused into the lattice
///   read-RAF under akita), and the base hamming-weight claim reduction
///   (extended/replaced by the lattice variant) — the akita substitution set
///   of `specs/lattice-claims.md` ("The Dory build instantiates the original
///   Booleanity, increment claim reduction, HammingWeightClaimReduction").
/// - `akita`: everything under `protocols/jolt/lattice/` — the lattice-mode
///   booleanity/read-RAF/hamming variants and the packed-column
///   reconstruction relations.
pub fn all_vertices(config: &ProtocolConfig) -> Vec<VertexRecord> {
    claim_graph_vertices!(config, {
        shared: [
            OuterUniskip,
            OuterRemainder,
            ProductUniskip,
            ProductRemainder,
            Shift,
            InstructionClaimReduction,
            InstructionInputVirtualization,
            InstructionReadRaf,
            InstructionRaVirtualization,
            RamReadWriteChecking,
            RamRafEvaluation,
            RamOutputCheck,
            RamValCheck,
            RamRaClaimReduction,
            RamHammingBooleanity,
            RamRaVirtualization,
            RegistersClaimReduction,
            RegistersReadWriteChecking,
            RegistersValEvaluation,
            BooleanityAddressPhase,
            TrustedAdviceCyclePhase,
            UntrustedAdviceCyclePhase,
            TrustedAdviceAddressPhase,
            UntrustedAdviceAddressPhase,
            BytecodeReductionCyclePhase,
            BytecodeReductionAddressPhase,
            ProgramImageReductionCyclePhase,
            ProgramImageReductionAddressPhase,
        ],
        dory: [
            Booleanity,
            BooleanityCyclePhase,
            BytecodeReadRaf,
            BytecodeReadRafAddressPhase,
            BytecodeReadRafCyclePhase,
            BytecodeReadRafCyclePhaseCommitted,
            IncClaimReduction,
            HammingWeightClaimReduction,
        ],
        akita: [
            LatticeBooleanity,
            LatticeBooleanityCyclePhase,
            LatticeReadRafAddressPhase,
            LatticeReadRafCyclePhase,
            LatticeReadRafCyclePhaseCommitted,
            LatticeHammingWeightClaimReduction,
            UntrustedAdviceReconstruction,
            TrustedAdviceReconstruction,
            ProgramImageReconstruction,
            BytecodeChunkReconstruction,
        ],
    })
}

/// How a relation id is accounted for in the vertex set.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Registration {
    /// At least one `ProtocolVertex` in the manifest carries this relation id.
    Registered,
    /// Not yet added to the manifest (fails the exhaustiveness test).
    #[expect(
        dead_code,
        reason = "kept as the classification vocabulary for future relations; \
                  every current relation is Registered"
    )]
    Pending,
}

/// Wildcard-free: a new `JoltRelationId` variant fails compilation here until
/// classified.
// NOTE: variants are written `JoltRelationId::X` (no glob import) because
// several relation-type aliases in this file share the variant names.
pub fn registration(id: JoltRelationId) -> Registration {
    use Registration::*;
    match id {
        JoltRelationId::SpartanOuter => Registered,
        JoltRelationId::SpartanProductVirtualization => Registered,
        JoltRelationId::SpartanShift => Registered,
        JoltRelationId::InstructionClaimReduction => Registered,
        JoltRelationId::InstructionInputVirtualization => Registered,
        JoltRelationId::InstructionReadRaf => Registered,
        JoltRelationId::InstructionRaVirtualization => Registered,
        JoltRelationId::RamReadWriteChecking => Registered,
        JoltRelationId::RamRafEvaluation => Registered,
        JoltRelationId::RamOutputCheck => Registered,
        JoltRelationId::RamValCheck => Registered,
        JoltRelationId::RamRaClaimReduction => Registered,
        JoltRelationId::RamHammingBooleanity => Registered,
        JoltRelationId::RamRaVirtualization => Registered,
        JoltRelationId::RegistersClaimReduction => Registered,
        JoltRelationId::RegistersReadWriteChecking => Registered,
        JoltRelationId::RegistersValEvaluation => Registered,
        JoltRelationId::BytecodeReadRaf => Registered,
        JoltRelationId::Booleanity => Registered,
        JoltRelationId::AdviceClaimReductionCyclePhase => Registered,
        JoltRelationId::AdviceClaimReduction => Registered,
        JoltRelationId::BytecodeClaimReductionCyclePhase => Registered,
        JoltRelationId::BytecodeClaimReduction => Registered,
        JoltRelationId::ProgramImageClaimReductionCyclePhase => Registered,
        JoltRelationId::ProgramImageClaimReduction => Registered,
        JoltRelationId::IncClaimReduction => Registered,
        JoltRelationId::HammingWeightClaimReduction => Registered,
        JoltRelationId::UntrustedAdviceReconstruction => Registered,
        JoltRelationId::TrustedAdviceReconstruction => Registered,
        JoltRelationId::ProgramImageReconstruction => Registered,
        JoltRelationId::BytecodeChunkReconstruction => Registered,
    }
}

/// Every relation id, for the exhaustiveness test. Kept adjacent to
/// `registration` so both change together.
pub fn all_relation_ids() -> Vec<JoltRelationId> {
    vec![
        JoltRelationId::SpartanOuter,
        JoltRelationId::SpartanProductVirtualization,
        JoltRelationId::SpartanShift,
        JoltRelationId::InstructionClaimReduction,
        JoltRelationId::InstructionInputVirtualization,
        JoltRelationId::InstructionReadRaf,
        JoltRelationId::InstructionRaVirtualization,
        JoltRelationId::RamReadWriteChecking,
        JoltRelationId::RamRafEvaluation,
        JoltRelationId::RamOutputCheck,
        JoltRelationId::RamValCheck,
        JoltRelationId::RamRaClaimReduction,
        JoltRelationId::RamHammingBooleanity,
        JoltRelationId::RamRaVirtualization,
        JoltRelationId::RegistersClaimReduction,
        JoltRelationId::RegistersReadWriteChecking,
        JoltRelationId::RegistersValEvaluation,
        JoltRelationId::BytecodeReadRaf,
        JoltRelationId::Booleanity,
        JoltRelationId::AdviceClaimReductionCyclePhase,
        JoltRelationId::AdviceClaimReduction,
        JoltRelationId::BytecodeClaimReductionCyclePhase,
        JoltRelationId::BytecodeClaimReduction,
        JoltRelationId::ProgramImageClaimReductionCyclePhase,
        JoltRelationId::ProgramImageClaimReduction,
        JoltRelationId::IncClaimReduction,
        JoltRelationId::HammingWeightClaimReduction,
        JoltRelationId::UntrustedAdviceReconstruction,
        JoltRelationId::TrustedAdviceReconstruction,
        JoltRelationId::ProgramImageReconstruction,
        JoltRelationId::BytecodeChunkReconstruction,
    ]
}

/// Every relation id is carried by a registered vertex, and every registered
/// vertex's relation id is classified `Registered`. Lives in the shared
/// module so the backstop runs in every test target that builds the graph
/// (the Dory and Akita targets compile under different feature sets).
#[test]
fn vertex_set_is_exhaustive() {
    let config = ProtocolConfig::small();
    let records = all_vertices(&config);
    let covered: std::collections::BTreeSet<_> =
        records.iter().map(|record| record.relation).collect();
    let mut missing = Vec::new();
    for id in all_relation_ids() {
        match registration(id) {
            Registration::Registered => {
                assert!(
                    covered.contains(&id),
                    "{id:?} is classified Registered but no manifest vertex carries it"
                );
            }
            Registration::Pending => missing.push(id),
        }
    }
    assert!(
        missing.is_empty(),
        "unregistered relations (add ProtocolVertex impls and manifest entries): {missing:?}"
    );
}

/// The wire-copy alias pairs `(aliased, source)`, assembled from the same
/// jolt-claims geometry functions the verifier's `aliased_output_openings`
/// declarations use (stage2/instruction_claim_reduction.rs,
/// stage3/instruction_input.rs, stage3/registers_claim_reduction.rs).
pub fn all_aliases() -> Vec<(
    jolt_claims::protocols::jolt::JoltOpeningId,
    jolt_claims::protocols::jolt::JoltOpeningId,
)> {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::registers as registers_reduction_geometry;
    use jolt_claims::protocols::jolt::geometry::{bytecode, instruction};

    let mut aliases = Vec::new();
    // Instruction claim reduction: lookup output + left/right inputs alias the
    // product-remainder openings.
    aliases.extend(instruction::read_raf_consistency_openings());
    aliases.extend(instruction::input_virtualization_consistency_openings());
    // Instruction input virtualization: unexpanded PC aliases the shift's
    // (the geometry pair is (shift, instruction-input): swap to (aliased, source)).
    let [(shift_unexpanded_pc, instruction_unexpanded_pc)] =
        bytecode::read_raf_consistency_openings();
    aliases.push((instruction_unexpanded_pc, shift_unexpanded_pc));
    // Registers claim reduction: reduced rs1/rs2 values alias the
    // instruction-input virtualization outputs.
    aliases.push((
        registers_reduction_geometry::rs1_value_reduced(),
        instruction::rs1_value(),
    ));
    aliases.push((
        registers_reduction_geometry::rs2_value_reduced(),
        instruction::rs2_value(),
    ));
    aliases
}
