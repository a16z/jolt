//! The prover-owned generated stage driver.
//!
//! [`StageProver`] is the driver trait, implemented for each stage batch
//! struct by [`impl_stage_prover!`] — expanded from the batch's
//! derive-emitted member-list callback macro (`jolt-verifier`'s
//! `<snake_case_struct>_members!`), so no stage's member list, order, or
//! presence is ever restated on the prove side. One recorder-generic
//! [`prove`](StageProver::prove) runs the whole body: `begin_batch` (the
//! generated shared head) → per-member kernel `prepare` in declaration order
//! (through [`KernelSource`], the per-stage `HasKernel` bound collector) →
//! the engine round loop → derived opening points → per-member
//! `validate_derived_tables` → typed extraction into the stage's
//! `OutputClaims` aggregate → [`curate_opening_values`](StageProver::curate_opening_values)
//! (default: the derive-generated canonical absorb order; curated stages
//! override at the invocation site) → shape validation → the
//! `expected_final_claim` hard self-check → `recorder.finish`. The recorder
//! is the clear/ZK seam, exactly as in `begin_batch`: no `prove_clear`/
//! `prove_zk` split exists to drift.
//!
//! See `specs/prover-stage-drivers.md`.

use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_kernels::{HasKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError};
use jolt_sumcheck::{RecordedSumcheck, SumcheckRecorder};
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::VerifierError;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::ProverError;

/// The generated per-stage driver: implemented for each batch struct by
/// [`impl_stage_prover!`], never by hand. The associated types are the
/// derive-generated aggregate projections of the batch declaration;
/// [`Kernels`](Self::Kernels) is the typed kernel bundle — one boxed
/// [`SumcheckKernel`] per member, `Option`-wrapped for a conditional member,
/// in declaration order — that [`KernelSource::prepare_members`] mints.
pub trait StageProver<F: Field>: Sized {
    type InputClaims;
    type InputPoints;
    type Challenges;
    type OutputClaims;
    type OutputPoints;
    type Kernels;

    /// Prove this stage's batch on `transcript`, recorder-generically: the
    /// recorder type decides clear vs. committed recording, never a runtime
    /// flag. Returns the [`Proved`] carrier (recorded proof, typed output
    /// claims, derived points, hard-checked final claim).
    #[expect(
        clippy::too_many_arguments,
        reason = "the driver's fixed protocol signature: upstream carriers in, recorded proof out"
    )]
    fn prove<B, Rec, T>(
        &self,
        kernels: &B,
        session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: &Self::InputClaims,
        input_points: &Self::InputPoints,
        challenges: &Self::Challenges,
        recorder: Rec,
        transcript: &mut T,
    ) -> Result<Proved<F, Self, Rec::Commitment>, ProverError<F>>
    where
        B: KernelSource<F, Self> + ?Sized,
        Rec: SumcheckRecorder<F>,
        T: Transcript<Challenge = F>;

    /// The stage's absorbed opening scalars, with any wire-claim mutation the
    /// stage curates (stage 4 has none to make — its staged advice openings
    /// ride in from the kernel — but stage 6b's runtime point dedup reorders).
    /// The default emitted by [`impl_stage_prover!`] returns the
    /// derive-generated canonical order (`opening_values`); curated stages
    /// supply an override block at the macro invocation site.
    fn curate_opening_values(
        &self,
        claims: &mut Self::OutputClaims,
        points: &Self::OutputPoints,
    ) -> Result<Vec<F>, ProverError<F>>;
}

/// The per-stage kernel-source bound collector: [`impl_stage_prover!`] emits
/// one blanket impl per stage — over any `B` carrying `HasKernel<F, R>` for
/// exactly that stage's member relations — so [`StageProver::prove`]'s `B`
/// bound stays uniform while each stage demands exactly its members' slots.
/// `prepare_members` mints the typed kernel bundle in declaration order
/// (`Option` members gated on presence, mismatched presence attributed to the
/// member's relation id).
pub trait KernelSource<F: Field, S: StageProver<F>> {
    fn prepare_members(
        &self,
        batch: &S,
        session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: &S::InputClaims,
        input_points: &S::InputPoints,
        challenges: &S::Challenges,
    ) -> Result<S::Kernels, ProverError<F>>;
}

/// A proved stage batch, as assembled by the generated
/// [`prove`](StageProver::prove): the recorded wire proof (plus retained
/// witness for a committed recorder), the typed output claims and derived
/// opening points, and the batch's final running claim (already hard-checked
/// against the generated `expected_final_claim`).
pub struct Proved<F: Field, S: StageProver<F>, C> {
    pub recorded: RecordedSumcheck<F, C>,
    pub output_claims: S::OutputClaims,
    pub output_points: S::OutputPoints,
    pub final_claim: F,
}

/// One batch member's boxed kernel, as minted through its [`HasKernel`] slot.
pub type MemberKernel<F, R> = Box<dyn SumcheckKernel<F, Relation = R>>;

/// Mint one required member's kernel through the source's [`HasKernel`] slot.
pub fn prepare_required<F, R, B>(
    kernels: &B,
    relation: &R,
    session: &mut ProofSession,
    witness: &dyn JoltVmWitnessPlane<F>,
    claims: &SumcheckInputClaims<F, R>,
    points: &SumcheckInputPoints<F, R>,
    challenges: &ConcreteSumcheckChallenges<F, R>,
) -> Result<MemberKernel<F, R>, ProverError<F>>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    B: HasKernel<F, R> + ?Sized,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    Ok(kernels.kernel().prepare(
        session,
        witness,
        ProverInputs {
            relation,
            claims,
            points,
            challenges,
        },
    )?)
}

/// Mint one `Option` member's kernel when present. A present instance with an
/// absent input, point, or challenge cell is a wiring bug, attributed to the
/// member's relation id — silently skipping it would desynchronize the batch.
pub fn prepare_optional<F, R, B>(
    kernels: &B,
    relation: Option<&R>,
    session: &mut ProofSession,
    witness: &dyn JoltVmWitnessPlane<F>,
    claims: Option<&SumcheckInputClaims<F, R>>,
    points: Option<&SumcheckInputPoints<F, R>>,
    challenges: Option<&ConcreteSumcheckChallenges<F, R>>,
) -> Result<Option<MemberKernel<F, R>>, ProverError<F>>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    B: HasKernel<F, R> + ?Sized,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    match (relation, claims, points, challenges) {
        (Some(relation), Some(claims), Some(points), Some(challenges)) => {
            Ok(Some(prepare_required(
                kernels, relation, session, witness, claims, points, challenges,
            )?))
        }
        (None, _, _, _) => Ok(None),
        (Some(relation), _, _, _) => Err(VerifierError::StageClaimSumcheckFailed {
            stage: relation.id(),
            reason: "present instance is missing an input, point, or challenge cell for prepare"
                .to_string(),
        }
        .into()),
    }
}

/// Cross-check one `Option` member's hand-materialized derived tables when
/// every cell is present (the presence invariants were enforced at prepare).
pub fn validate_optional_tables<F, R>(
    kernel: Option<&dyn SumcheckKernel<F, Relation = R>>,
    relation: Option<&R>,
    input_points: Option<&SumcheckInputPoints<F, R>>,
    output_points: Option<&SumcheckOutputPoints<F, R>>,
    challenges: Option<&ConcreteSumcheckChallenges<F, R>>,
) -> Result<(), SumcheckKernelError<F>>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    if let (
        Some(kernel),
        Some(relation),
        Some(input_points),
        Some(output_points),
        Some(challenges),
    ) = (kernel, relation, input_points, output_points, challenges)
    {
        kernel.validate_derived_tables(relation, input_points, output_points, challenges)?;
    }
    Ok(())
}

/// Extract one `Option` member's typed output claims when present.
pub fn extract_optional<F, R>(
    kernel: Option<&mut (dyn SumcheckKernel<F, Relation = R> + '_)>,
) -> Result<Option<SumcheckOutputClaims<F, R>>, SumcheckKernelError<F>>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    kernel.map(|kernel| kernel.output_claims()).transpose()
}

/// One member slot of the generated driver body, dispatched on the member's
/// presence: the kernel-bundle tuple element type, the prepare expression,
/// the round-loop push, the derived-table cross-check, and the typed
/// extraction. Internal to [`impl_stage_prover!`]; every local it touches is
/// passed in by name (macro hygiene).
macro_rules! __stage_member {
    (kernel_ty required $relation:ident) => {
        ::std::boxed::Box<dyn ::jolt_kernels::SumcheckKernel<F, Relation = $relation<F>>>
    };
    (kernel_ty optional $relation:ident) => {
        ::core::option::Option<
            ::std::boxed::Box<dyn ::jolt_kernels::SumcheckKernel<F, Relation = $relation<F>>>,
        >
    };
    (prepare required $member:ident, $relation:ident, $source:expr, $batch:expr, $session:expr, $witness:expr, $inputs:expr, $points:expr, $challenges:expr) => {
        $crate::driver::prepare_required::<F, $relation<F>, _>(
            $source,
            &$batch.$member,
            $session,
            $witness,
            &$inputs.$member,
            &$points.$member,
            &$challenges.$member,
        )?
    };
    (prepare optional $member:ident, $relation:ident, $source:expr, $batch:expr, $session:expr, $witness:expr, $inputs:expr, $points:expr, $challenges:expr) => {
        $crate::driver::prepare_optional::<F, $relation<F>, _>(
            $source,
            $batch.$member.as_ref(),
            $session,
            $witness,
            $inputs.$member.as_ref(),
            $points.$member.as_ref(),
            $challenges.$member.as_ref(),
        )?
    };
    (round_slot required $member:ident) => {
        ::core::option::Option::Some(&mut *$member as &mut dyn ::jolt_sumcheck::ProveRounds<F>)
    };
    (round_slot optional $member:ident) => {
        $member
            .as_mut()
            .map(|__kernel| &mut **__kernel as &mut dyn ::jolt_sumcheck::ProveRounds<F>)
    };
    (validate required $member:ident, $self:expr, $input_points:expr, $output_points:expr, $challenges:expr) => {
        $member.validate_derived_tables(
            &$self.$member,
            &$input_points.$member,
            &$output_points.$member,
            &$challenges.$member,
        )?;
    };
    (validate optional $member:ident, $self:expr, $input_points:expr, $output_points:expr, $challenges:expr) => {
        $crate::driver::validate_optional_tables(
            $member.as_deref(),
            $self.$member.as_ref(),
            $input_points.$member.as_ref(),
            $output_points.$member.as_ref(),
            $challenges.$member.as_ref(),
        )?;
    };
    (extract required $member:ident) => {
        $member.output_claims()?
    };
    (extract optional $member:ident) => {
        $crate::driver::extract_optional($member.as_deref_mut())?
    };
}

/// The output-shape leg of the generated driver, keyed by the derive-emitted
/// flag: `checked` runs the generated `validate_output_claims`; `unchecked`
/// (a `no_output_shape` stage, whose wire shape is runtime-curated) has no
/// validator to run.
macro_rules! __stage_shape_check {
    (checked, $self:expr, $claims:expr) => {
        $self.validate_output_claims(&$claims)?;
    };
    (unchecked, $self:expr, $claims:expr) => {};
}

/// Expand one stage's [`StageProver`] and [`KernelSource`] impls from its
/// derive-emitted member-list macro. Invoke as the member-list macro's
/// callback, at a site where the batch's relation and aggregate names
/// resolve:
///
/// ```ignore
/// jolt_verifier::stage3_sumchecks_members!(impl_stage_prover);
/// // a curated stage passes its absorb-order override ahead of the list
/// // (the first closure parameter binds the batch, i.e. `&self`):
/// jolt_verifier::stage6b_sumchecks_members!(impl_stage_prover curate = |batch, claims, points| { .. },);
/// ```
macro_rules! impl_stage_prover {
    // Uncurated stage: default to the derive-generated canonical absorb order.
    // The batch receiver is an explicit closure parameter (not `self`) because
    // macro hygiene separates a `self` written in this arm from the `&self`
    // the main arm's method signature introduces.
    (batch = $($rest:tt)*) => {
        $crate::driver::impl_stage_prover! {
            curate = |__batch, __claims, __points| {
                ::core::result::Result::Ok(__batch.opening_values(__claims))
            },
            batch = $($rest)*
        }
    };
    (
        curate = |$curate_batch:ident, $curate_claims:ident, $curate_points:ident| $curate_body:block,
        batch = $batch:ident,
        aggregates = {
            input_claims = $input_claims:ident,
            input_points = $input_points:ident,
            output_claims = $output_claims:ident,
            output_points = $output_points:ident,
            challenges = $challenges_ty:ident,
        },
        shape = $shape:ident,
        members = [
            $({ name: $member:ident, relation: $relation:ident, presence: $presence:ident },)+
        ]
    ) => {
        impl<F: ::jolt_field::Field> $crate::driver::StageProver<F> for $batch<F> {
            type InputClaims = $input_claims<F>;
            type InputPoints = $input_points<F>;
            type Challenges = $challenges_ty<F>;
            type OutputClaims = $output_claims<F>;
            type OutputPoints = $output_points<F>;
            type Kernels = ( $($crate::driver::__stage_member!(kernel_ty $presence $relation),)+ );

            fn prove<B, Rec, T>(
                &self,
                kernels: &B,
                session: &mut ::jolt_kernels::ProofSession,
                witness: &dyn ::jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane<F>,
                inputs: &Self::InputClaims,
                input_points: &Self::InputPoints,
                challenges: &Self::Challenges,
                mut recorder: Rec,
                transcript: &mut T,
            ) -> ::core::result::Result<
                $crate::driver::Proved<F, Self, Rec::Commitment>,
                $crate::ProverError<F>,
            >
            where
                B: $crate::driver::KernelSource<F, Self> + ?Sized,
                Rec: ::jolt_sumcheck::SumcheckRecorder<F>,
                T: ::jolt_transcript::Transcript<Challenge = F>,
            {
                let (__batch, __coefficients) =
                    self.begin_batch(inputs, challenges, &mut recorder, transcript)?;

                let ($(mut $member,)+) = $crate::driver::KernelSource::prepare_members(
                    kernels,
                    self,
                    session,
                    witness,
                    inputs,
                    input_points,
                    challenges,
                )?;

                let mut __rounds: ::std::vec::Vec<&mut dyn ::jolt_sumcheck::ProveRounds<F>> =
                    [$($crate::driver::__stage_member!(round_slot $presence $member),)+]
                        .into_iter()
                        .flatten()
                        .collect();
                let __proved = ::jolt_sumcheck::prove_batch(
                    &__batch,
                    &mut __rounds,
                    &mut recorder,
                    transcript,
                )?;

                let __output_points =
                    self.derive_opening_points(&__proved.challenges, input_points)?;
                $($crate::driver::__stage_member!(
                    validate $presence $member, self, input_points, __output_points, challenges
                );)+

                let mut __output_claims = $output_claims {
                    $($member: $crate::driver::__stage_member!(extract $presence $member),)+
                };

                let __opening_values =
                    self.curate_opening_values(&mut __output_claims, &__output_points)?;
                $crate::driver::__stage_shape_check!($shape, self, __output_claims);

                let __expected = self.expected_final_claim(
                    &__coefficients,
                    input_points,
                    &__output_claims,
                    &__output_points,
                    challenges,
                )?;
                if __expected != __proved.final_claim {
                    let __stage_id = [$(
                        <<$relation<F> as ::jolt_verifier::stages::relations::ConcreteSumcheck<F>>
                            ::Symbolic as ::jolt_claims::SymbolicSumcheck>::id(),
                    )+][0];
                    return ::core::result::Result::Err(
                        ::jolt_verifier::VerifierError::StageClaimSumcheckFailed {
                            stage: __stage_id,
                            reason: ::std::format!(
                                "prover final claim {:?} disagrees with the expected output fold {:?}",
                                __proved.final_claim,
                                __expected,
                            ),
                        }
                        .into(),
                    );
                }

                let __recorded = recorder.finish(&__opening_values, transcript)?;
                ::core::result::Result::Ok($crate::driver::Proved {
                    recorded: __recorded,
                    output_claims: __output_claims,
                    output_points: __output_points,
                    final_claim: __proved.final_claim,
                })
            }

            fn curate_opening_values(
                &self,
                $curate_claims: &mut Self::OutputClaims,
                $curate_points: &Self::OutputPoints,
            ) -> ::core::result::Result<::std::vec::Vec<F>, $crate::ProverError<F>> {
                let $curate_batch = self;
                let _ = ($curate_batch, $curate_points);
                $curate_body
            }
        }

        impl<F: ::jolt_field::Field, B> $crate::driver::KernelSource<F, $batch<F>> for B
        where
            B: ?Sized $(+ ::jolt_kernels::HasKernel<F, $relation<F>>)+,
        {
            fn prepare_members(
                &self,
                batch: &$batch<F>,
                session: &mut ::jolt_kernels::ProofSession,
                witness: &dyn ::jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane<F>,
                inputs: &$input_claims<F>,
                input_points: &$input_points<F>,
                challenges: &$challenges_ty<F>,
            ) -> ::core::result::Result<
                <$batch<F> as $crate::driver::StageProver<F>>::Kernels,
                $crate::ProverError<F>,
            > {
                ::core::result::Result::Ok(($(
                    $crate::driver::__stage_member!(
                        prepare $presence $member, $relation, self, batch, session, witness,
                        inputs, input_points, challenges
                    ),
                )+))
            }
        }
    };
}

pub(crate) use {__stage_member, __stage_shape_check, impl_stage_prover};
