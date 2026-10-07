//! Akita advice and direct committed-program objects.

use std::collections::HashMap;

use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, committed_program_packing_plan, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::{JoltField, Ring};
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::Polynomial;
use jolt_program::preprocess::JoltProgramPreprocessing;

use crate::ProverError;

/// One advice-word commitment object: one field coefficient per
/// canonical little-endian `u64`, embedded in slot zero when Akita's dense
/// schedule floor exceeds the logical word arity.
pub struct AdviceObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Builds the canonical zero-padded advice-word commitment. The setup
/// is derived from the application-owned immutable setup context plus the
/// public advice shape (the setup is transparent).
pub fn commit_advice<PCS>(
    setup_context: &PCS::SetupContext,
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
    max_advice_bytes: usize,
) -> Result<AdviceObject<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    let words = common::advice::canonical_advice_words(advice_bytes, max_advice_bytes)
        .map_err(commit_failed)?;
    let word_vars = words.len().ilog2() as usize;
    let plan = advice_packing_plan(kind, word_vars).map_err(commit_failed)?;
    let physical_vars = plan.packing().packed_num_vars();
    let (setup, _) =
        PCS::transparent_object_setup(setup_context, physical_vars, plan.layout_digest())
            .map_err(commit_failed)?;
    let mut evaluations = vec![PCS::Field::default(); 1usize << physical_vars];
    for (evaluation, word) in evaluations.iter_mut().zip(words) {
        *evaluation = PCS::Field::from_u64(word);
    }
    let polynomial = Polynomial::new(evaluations);
    let (commitment, hint) = PCS::commit(&polynomial, &setup).map_err(commit_failed)?;
    Ok(AdviceObject {
        plan,
        commitment,
        hint,
    })
}

fn commit_failed<F: JoltField>(error: impl ToString) -> ProverError<F> {
    ProverError::Verifier(
        jolt_verifier::VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        },
    )
}

/// One direct bounded-dense committed-program object.
#[derive(Clone)]
pub struct DirectProgramObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// The precommitted direct program objects in canonical order: indexed
/// bytecode chunks followed by the program image. Built once at preprocessing
/// time and retained in
/// [`crate::CommittedProgramProverData`], so proving consumes the objects
/// directly.
#[derive(Clone)]
pub struct DirectProgramObjects<PCS: CommitmentScheme> {
    pub objects: Vec<DirectProgramObject<PCS>>,
}

/// Assembles and commits the direct bytecode chunks and program-image object.
pub fn commit_direct_program<PCS>(
    setup_context: &PCS::SetupContext,
    program: &JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
    trace_order: TracePolynomialOrder,
) -> Result<DirectProgramObjects<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    let bytecode_len = program.bytecode.bytecode.len();
    let image_words =
        jolt_kernels::committed_program::program_image_words_padded(&program.ram.bytecode_words);
    let plan = committed_program_packing_plan(
        bytecode_len,
        bytecode_chunk_count,
        program.ram.bytecode_words.len(),
        trace_order,
    )
    .map_err(commit_failed)?;
    let mut chunk_coeffs = jolt_kernels::committed_program::build_committed_bytecode_chunk_coeffs(
        &program.bytecode.bytecode,
        bytecode_chunk_count,
        trace_order,
    )
    .map_err(commit_failed)?
    .into_iter();
    let mut setups = HashMap::<usize, PCS::ProverSetup>::new();
    let objects = plan
        .objects()
        .map(|object_plan| {
            let id = object_plan.packing().ids()[0];
            let mut evaluations = match id {
                JoltCommittedPolynomial::BytecodeChunk(_) => {
                    chunk_coeffs.next().ok_or(ProverError::InvariantViolation {
                        reason: "missing direct bytecode chunk witness",
                    })?
                }
                JoltCommittedPolynomial::ProgramImageInit => image_words
                    .iter()
                    .map(|word| PCS::Field::from_u64(*word))
                    .collect(),
                _ => {
                    return Err(ProverError::InvariantViolation {
                        reason: "unexpected direct committed-program object",
                    })
                }
            };
            evaluations.resize(
                1usize << object_plan.packing().packed_num_vars(),
                PCS::Field::default(),
            );
            let witness = Polynomial::new(evaluations);
            let physical_vars = object_plan.packing().packed_num_vars();
            let setup = if let Some(setup) = setups.get(&physical_vars) {
                PCS::retag_transparent_object_setup(setup, object_plan.layout_digest())
                    .map_err(commit_failed)?
                    .0
            } else {
                PCS::transparent_object_setup(
                    setup_context,
                    physical_vars,
                    object_plan.layout_digest(),
                )
                .map_err(commit_failed)?
                .0
            };
            let (commitment, hint) = PCS::commit(&witness, &setup).map_err(commit_failed)?;
            let _ = setups.entry(physical_vars).or_insert_with(|| setup.clone());
            Ok(DirectProgramObject {
                plan: object_plan.clone(),
                commitment,
                hint,
            })
        })
        .collect::<Result<Vec<_>, ProverError<PCS::Field>>>()?;
    Ok(DirectProgramObjects { objects })
}
