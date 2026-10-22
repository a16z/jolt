//! Akita advice and direct committed-program objects.

use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, PrecommittedPackingPlan, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, TracePolynomialOrder};
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
    let _commit = tracing::info_span!("akita_advice_commit").entered();
    let preparation = tracing::info_span!("akita_advice_host_prepare").entered();
    if advice_bytes.len() > max_advice_bytes {
        return Err(commit_failed(
            common::advice::AdviceWordsError::AdviceTooLong {
                actual: advice_bytes.len(),
                max: max_advice_bytes,
            },
        ));
    }
    let word_vars = common::advice::canonical_advice_word_capacity(max_advice_bytes)
        .map_err(commit_failed)?
        .ilog2() as usize;
    let plan = advice_packing_plan(kind, word_vars).map_err(commit_failed)?;
    let physical_vars = plan.packing().packed_num_vars();
    let (setup, _) =
        PCS::transparent_object_setup(setup_context, physical_vars, plan.layout_digest())
            .map_err(commit_failed)?;
    drop(preparation);
    let (commitment, hint) =
        PCS::commit_packed_words(&setup, advice_bytes, max_advice_bytes, physical_vars)
            .map_err(commit_failed)?;
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

/// Assembles and commits the whole bytecode and program-image object.
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
    let commit = |object_plan: &PrefixPackedObjectPlan,
                  setup: &PCS::ProverSetup,
                  mut evaluations: Vec<PCS::Field>|
     -> Result<DirectProgramObject<PCS>, ProverError<PCS::Field>> {
        evaluations.resize(
            1usize << object_plan.packing().packed_num_vars(),
            PCS::Field::default(),
        );
        let witness = Polynomial::new(evaluations);
        let (commitment, hint) = PCS::commit(&witness, setup).map_err(commit_failed)?;
        Ok(DirectProgramObject {
            plan: object_plan.clone(),
            commitment,
            hint,
        })
    };
    let bytecode_coeffs = jolt_kernels::committed_program::build_committed_bytecode_coeffs(
        &program.bytecode.bytecode,
        trace_order,
    )
    .map_err(commit_failed)?;
    let bytecode = commit(&plan.bytecode, &bytecode_setup, bytecode_coeffs)?;
    let image_words =
        jolt_kernels::committed_program::program_image_words_padded(&program.ram.bytecode_words);
    let program_image = commit(
        &plan.program_image,
        &image_setup,
        image_words.into_iter().map(PCS::Field::from_u64).collect(),
    )?;
    Ok(DirectProgramObjects {
        bytecode,
        program_image,
    })
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "oracle tests fail loudly on invalid commitments"
)]
mod packed_word_tests {
    use super::*;
    use jolt_akita::{AkitaChunkProfile, AkitaField, AkitaScheduleArtifacts, AkitaScheme};

    #[test]
    fn advice_hook_matches_canonical_cpu_polynomial_and_preserves_roles() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(1, 0));
        for kind in [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted] {
            for length in [32, 9, 8, 7, 1, 0] {
                let bytes: Vec<u8> = (0..length).map(|i| (i * 37 + 255) as u8).collect();
                let object = commit_advice::<AkitaScheme>(&artifacts, kind, &bytes, 32).unwrap();
                let n = object.plan.packing().packed_num_vars();
                let (setup, _) = AkitaScheme::transparent_object_setup(
                    &artifacts,
                    n,
                    object.plan.layout_digest(),
                )
                .unwrap();
                let words = common::advice::canonical_advice_words(&bytes, 32).unwrap();
                let mut table = vec![AkitaField::default(); 1 << n];
                for (value, word) in table.iter_mut().zip(words) {
                    *value = AkitaField::from_u64(word);
                }
                let (expected, _) = AkitaScheme::commit(&Polynomial::new(table), &setup).unwrap();
                assert_eq!(object.commitment, expected);
            }
        }
        assert!(
            commit_advice::<AkitaScheme>(&artifacts, JoltAdviceKind::Untrusted, &[0; 33], 32)
                .is_err()
        );
    }
}
