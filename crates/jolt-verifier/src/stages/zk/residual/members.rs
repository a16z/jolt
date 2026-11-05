//! Whiteout's stages 5–7: each batch member lowered by the clear verifier's
//! own formula owners ([`MaskedMember`]), since the Akita build changes these
//! stages' relations (canonical instruction addresses at stage 5; the
//! lattice read-RAF, booleanity and digit-zero relations; no increment
//! reduction; advice reduced at stage 8). Stages 1–4 have the same relations
//! in both builds and keep the homomorphic build's lowering.

use super::*;

/// A batch member as a masked proof's residual relation lowers it: its
/// symbolic claim formulas, and the value of every public source they
/// reference, resolved by the member's own
/// [`derive_input_term`](crate::stages::relations::ConcreteSumcheck::derive_input_term),
/// [`derive_output_term`](crate::stages::relations::ConcreteSumcheck::derive_output_term)
/// and drawn challenges: the clear verifier's owners of each formula.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedMember<F: JoltField> {
    claim: MemberClaim<F>,
    publics: Vec<(VerifierPublicId, F)>,
}

impl<F: JoltField> MaskedMember<F> {
    /// Lowers `relation`, resolving each derived source of its input formula
    /// by `input_term`, of its output formula by `output_term`, and each
    /// challenge by `challenge`.
    ///
    /// # Errors
    ///
    /// The first resolver error, or a source resolving to two values.
    pub fn new<S>(
        relation: &S,
        mut input_term: impl FnMut(&S::DerivedId) -> Result<F, VerifierError>,
        mut output_term: impl FnMut(&S::DerivedId) -> Result<F, VerifierError>,
        mut challenge: impl FnMut(&S::ChallengeId) -> Result<F, VerifierError>,
    ) -> Result<Self, VerifierError>
    where
        S: SymbolicSumcheck,
        S::OpeningId: Into<ComposedOpeningId>,
        S::DerivedId: Copy + Into<VerifierPublicId>,
        S::ChallengeId: Copy + Into<VerifierPublicId>,
    {
        let mut publics = Vec::new();
        collect_publics(
            &relation.input_factored::<F>(),
            &mut input_term,
            &mut challenge,
            &mut publics,
        )?;
        collect_publics(
            &relation.output_factored::<F>(),
            &mut output_term,
            &mut challenge,
            &mut publics,
        )?;
        Ok(Self {
            claim: relation_claim(relation),
            publics,
        })
    }
}

/// Resolves every public source of `formula` into `publics`.
fn collect_publics<F, O, P, C>(
    formula: &FactoredExpr<F, O, P, C>,
    derived: &mut impl FnMut(&P) -> Result<F, VerifierError>,
    challenge: &mut impl FnMut(&C) -> Result<F, VerifierError>,
    publics: &mut Vec<(VerifierPublicId, F)>,
) -> Result<(), VerifierError>
where
    F: JoltField,
    P: Copy + Into<VerifierPublicId>,
    C: Copy + Into<VerifierPublicId>,
{
    for source in formula
        .terms
        .iter()
        .flat_map(|term| &term.factors)
        .flat_map(|factor| &factor.terms)
        .flat_map(|term| &term.factors)
    {
        let (id, value) = match source {
            Source::Opening(_) => continue,
            Source::Derived(id) => ((*id).into(), derived(id)?),
            Source::Challenge(id) => ((*id).into(), challenge(id)?),
        };
        match publics.iter().find(|(known, _)| *known == id) {
            Some((_, known)) if *known != value => {
                return Err(residual_error(format!(
                    "public source {id:?} resolves to two values"
                )))
            }
            Some(_) => {}
            None => publics.push((id, value)),
        }
    }
    Ok(())
}

pub(super) fn add_stage5<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    builder: ResidualBuilder<PCS::Field>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<ResidualBuilder<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    add_member_stage(
        builder,
        "stage5.batch",
        &input.stage5.members,
        &input.stage5.batch_rounds,
        &input.stage5.batch_output_claims,
        values,
    )
}

pub(super) fn add_stage6a<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    builder: ResidualBuilder<PCS::Field>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<ResidualBuilder<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    add_member_stage(
        builder,
        "stage6.address_phase",
        &input.stage6a.members,
        &input.stage6a.rounds,
        &input.stage6a.output_claims,
        values,
    )
}

pub(super) fn add_stage6b<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    builder: ResidualBuilder<PCS::Field>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<ResidualBuilder<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    add_member_stage(
        builder,
        "stage6.cycle_phase",
        &input.stage6b.members,
        &input.stage6b.batch_rounds,
        &input.stage6b.batch_output_claims,
        values,
    )
}

pub(super) fn add_stage7<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    builder: ResidualBuilder<PCS::Field>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<ResidualBuilder<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    add_member_stage(
        builder,
        "stage7.batch",
        &input.stage7.members,
        &input.stage7.batch_rounds,
        &input.stage7.batch_output_claims,
        values,
    )
}

/// Registers a batched stage whose members carry their own lowering: each
/// member's public values, then its claim formulas.
fn add_member_stage<F: JoltField>(
    builder: ResidualBuilder<F>,
    name: &'static str,
    members: &[MaskedMember<F>],
    rounds: &BatchedMaskedRounds<F>,
    output_claims: &MaskedClaims<F>,
    values: &mut SourceValues<F>,
) -> Result<ResidualBuilder<F>, VerifierError> {
    let claims = members
        .iter()
        .map(|member| {
            for &(id, value) in &member.publics {
                values.public(id, value)?;
            }
            Ok(member.claim.clone())
        })
        .collect::<Result<Vec<_>, VerifierError>>()?;
    // Batched rounds are compressed whatever the members' domain.
    add_batched_stage(
        builder,
        name,
        SumcheckDomain::BooleanHypercube,
        &claims,
        rounds,
        output_claims,
        values,
    )
}
