//! Akita advice and direct committed-program objects.

#[cfg(feature = "zk")]
use jolt_akita::TraceOneHotCommitment;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, PrecommittedPackingPlan, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, TracePolynomialOrder};
#[cfg(feature = "zk")]
use jolt_crypto::Commitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::Polynomial;
use jolt_program::preprocess::JoltProgramPreprocessing;
#[cfg(feature = "zk")]
use jolt_zk::whiteout::{DenseSource, PrecommittedShape};

use crate::ProverError;

/// One advice-word commitment object: one field coefficient per
/// canonical little-endian `u64`, embedded in slot zero when Akita's dense
/// schedule floor exceeds the logical word arity.
pub struct AdviceObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// The canonical zero-padded advice-word polynomial and its plan.
fn advice_polynomial<F: JoltField>(
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
    max_advice_bytes: usize,
) -> Result<(PrefixPackedObjectPlan, Polynomial<F>), ProverError<F>> {
    let words = common::advice::canonical_advice_words(advice_bytes, max_advice_bytes)
        .map_err(commit_failed)?;
    let word_vars = words.len().ilog2() as usize;
    let plan = advice_packing_plan(kind, word_vars).map_err(commit_failed)?;
    let mut evaluations = vec![F::default(); 1usize << plan.packing().packed_num_vars()];
    for (evaluation, word) in evaluations.iter_mut().zip(words) {
        *evaluation = F::from_u64(word);
    }
    Ok((plan, Polynomial::new(evaluations)))
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
    let (plan, polynomial) = advice_polynomial(kind, advice_bytes, max_advice_bytes)?;
    let (setup, _) = PCS::transparent_object_setup(
        setup_context,
        plan.packing().packed_num_vars(),
        plan.layout_digest(),
    )
    .map_err(commit_failed)?;
    let (commitment, hint) = PCS::commit(&polynomial, &setup).map_err(commit_failed)?;
    Ok(AdviceObject {
        plan,
        commitment,
        hint,
    })
}

/// Whiteout: commits the advice-word object hiding, with fresh randomness,
/// as a precommitted group on the grouped trace `setup`'s Whiteout backend,
/// the one that opens it.
#[cfg(feature = "zk")]
pub fn commit_advice_hiding<PCS>(
    setup: &PCS::ProverSetup,
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
    max_advice_bytes: usize,
) -> Result<AdviceObject<PCS>, ProverError<PCS::Field>>
where
    PCS: TraceOneHotCommitment,
{
    let (plan, polynomial) = advice_polynomial(kind, advice_bytes, max_advice_bytes)?;
    let (commitment, hint) = commit_bounded_hiding::<PCS>(setup, &plan, &polynomial)?;
    Ok(AdviceObject {
        plan,
        commitment,
        hint,
    })
}

/// A commitment and the hint that opens it.
#[cfg(feature = "zk")]
type Committed<PCS> = (
    <PCS as Commitment>::Output,
    <PCS as CommitmentScheme>::OpeningHint,
);

/// Whiteout: commits one bounded prefix-packed object hiding.
#[cfg(feature = "zk")]
fn commit_bounded_hiding<PCS>(
    setup: &PCS::ProverSetup,
    plan: &PrefixPackedObjectPlan,
    polynomial: &Polynomial<PCS::Field>,
) -> Result<Committed<PCS>, ProverError<PCS::Field>>
where
    PCS: TraceOneHotCommitment,
{
    let shape = PrecommittedShape {
        num_vars: plan.packing().packed_num_vars(),
        source: DenseSource::Bounded,
    };
    PCS::whiteout_commit_precommitted(setup, shape, polynomial, plan.layout_digest())
        .map_err(commit_failed)
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

/// The precommitted direct program objects in canonical order: whole bytecode
/// followed by the program image. Built once at preprocessing
/// time and retained in
/// [`crate::CommittedProgramProverData`], so proving consumes the objects
/// directly.
#[derive(Clone)]
pub struct DirectProgramObjects<PCS: CommitmentScheme> {
    pub bytecode: DirectProgramObject<PCS>,
    pub program_image: DirectProgramObject<PCS>,
}

impl<PCS: CommitmentScheme> DirectProgramObjects<PCS> {
    pub fn objects(&self) -> impl Iterator<Item = &DirectProgramObject<PCS>> {
        [&self.bytecode, &self.program_image].into_iter()
    }
}

/// The whole-bytecode and program-image polynomials, each zero-padded to
/// its plan's physical arity.
fn direct_program_polynomials<F: JoltField>(
    program: &JoltProgramPreprocessing,
    trace_order: TracePolynomialOrder,
    plan: &PrecommittedPackingPlan,
) -> Result<[Polynomial<F>; 2], ProverError<F>> {
    let padded = |object_plan: &PrefixPackedObjectPlan, mut evaluations: Vec<F>| {
        evaluations.resize(
            1usize << object_plan.packing().packed_num_vars(),
            F::default(),
        );
        Polynomial::new(evaluations)
    };
    let bytecode_coeffs = jolt_kernels::committed_program::build_committed_bytecode_coeffs(
        &program.bytecode.bytecode,
        trace_order,
    )
    .map_err(commit_failed)?;
    let image_words =
        jolt_kernels::committed_program::program_image_words_padded(&program.ram.bytecode_words);
    Ok([
        padded(&plan.bytecode, bytecode_coeffs),
        padded(
            &plan.program_image,
            image_words.into_iter().map(F::from_u64).collect(),
        ),
    ])
}

/// Assembles and commits the whole bytecode and program-image object.
#[cfg(not(feature = "zk"))]
pub(super) fn commit_direct_program<PCS>(
    setup_context: &PCS::SetupContext,
    program: &JoltProgramPreprocessing,
    trace_order: TracePolynomialOrder,
    plan: &PrecommittedPackingPlan,
) -> Result<DirectProgramObjects<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    let bytecode_vars = plan.bytecode.packing().packed_num_vars();
    let image_vars = plan.program_image.packing().packed_num_vars();
    let bytecode_setup =
        PCS::transparent_object_setup(setup_context, bytecode_vars, plan.bytecode.layout_digest())
            .map_err(commit_failed)?
            .0;
    let image_setup = if image_vars == bytecode_vars {
        PCS::retag_transparent_object_setup(&bytecode_setup, plan.program_image.layout_digest())
    } else {
        PCS::transparent_object_setup(
            setup_context,
            image_vars,
            plan.program_image.layout_digest(),
        )
    }
    .map_err(commit_failed)?
    .0;
    let [bytecode, program_image] = direct_program_polynomials(program, trace_order, plan)?;
    let commit = |object_plan: &PrefixPackedObjectPlan,
                  setup: &PCS::ProverSetup,
                  witness: &Polynomial<PCS::Field>|
     -> Result<DirectProgramObject<PCS>, ProverError<PCS::Field>> {
        let (commitment, hint) = PCS::commit(witness, setup).map_err(commit_failed)?;
        Ok(DirectProgramObject {
            plan: object_plan.clone(),
            commitment,
            hint,
        })
    };
    Ok(DirectProgramObjects {
        bytecode: commit(&plan.bytecode, &bytecode_setup, &bytecode)?,
        program_image: commit(&plan.program_image, &image_setup, &program_image)?,
    })
}

/// Whiteout: assembles the whole bytecode and program-image objects and
/// commits them hiding, with fresh randomness, as precommitted groups on the
/// grouped trace `setup`'s Whiteout backend (`specs/whiteout-jolt-integration.md` §5).
#[cfg(feature = "zk")]
pub(super) fn commit_direct_program_hiding<PCS>(
    setup: &PCS::ProverSetup,
    program: &JoltProgramPreprocessing,
    trace_order: TracePolynomialOrder,
    plan: &PrecommittedPackingPlan,
) -> Result<DirectProgramObjects<PCS>, ProverError<PCS::Field>>
where
    PCS: TraceOneHotCommitment,
{
    let [bytecode, program_image] = direct_program_polynomials(program, trace_order, plan)?;
    let commit = |object_plan: &PrefixPackedObjectPlan,
                  witness: &Polynomial<PCS::Field>|
     -> Result<DirectProgramObject<PCS>, ProverError<PCS::Field>> {
        let (commitment, hint) = commit_bounded_hiding::<PCS>(setup, object_plan, witness)?;
        Ok(DirectProgramObject {
            plan: object_plan.clone(),
            commitment,
            hint,
        })
    };
    Ok(DirectProgramObjects {
        bytecode: commit(&plan.bytecode, &bytecode)?,
        program_image: commit(&plan.program_image, &program_image)?,
    })
}
