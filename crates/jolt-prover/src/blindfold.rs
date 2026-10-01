//! The ZK proof tail: BlindFold over the committed stage proofs.
//!
//! The prover does not mirror the verifier's protocol lowering — it *runs*
//! it, strictly through `jolt-verifier`'s public verification surface. After
//! stage 8 it replays its own argument string through the verifier's
//! `verify_stages` — the seeding messages, `stage1::verify` … `stage8::verify`,
//! and the lowering with `stages::zk::blindfold::build` — on a verifier
//! transcript that must land exactly on the prover's own. The `BlindFoldProtocol`
//! the prover proves against is therefore the same code path the verifier
//! executes — a claim-formula change that updates the verifier's lowering is
//! picked up here automatically — and the replay doubles as a full
//! self-check of the assembled proof.
//!
//! The witness rows come from the recorder-retained per-stage secrets via
//! [`BlindFoldProtocol::assign_witness`], which needs only the protocol's
//! public parts plus the stage domains — protocol constants this crate's own
//! stage recipes prove over.

use common::jolt_device::JoltDevice;
use jolt_blindfold::BlindFoldWitness;
use jolt_claims::protocols::composed::geometry::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Accumulator, CanonicalDecode, JoltField, WithAccumulator};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_r1cs::constraints::jolt::SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_sumcheck::{CommittedSumcheckWitness, SumcheckDomainSpec};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use jolt_verifier::{jolt_protocol_id, verify_stages, VerifiedStages, VerifierError, JOLT_SESSION};

use crate::{JoltProverPreprocessing, ProverError};

/// The recorder-retained committed sumcheck witnesses, one per BlindFold
/// stage, named to pin the protocol stage order (`blindfold::build` inserts
/// each stage's uni-skip before its remainder batch).
pub(crate) struct ZkStageWitnesses<F> {
    pub stage1_uniskip: CommittedSumcheckWitness<F>,
    pub stage1: CommittedSumcheckWitness<F>,
    pub stage2_uniskip: CommittedSumcheckWitness<F>,
    pub stage2: CommittedSumcheckWitness<F>,
    pub stage3: CommittedSumcheckWitness<F>,
    pub stage4: CommittedSumcheckWitness<F>,
    pub stage5: CommittedSumcheckWitness<F>,
    pub stage6a: CommittedSumcheckWitness<F>,
    pub stage6b: CommittedSumcheckWitness<F>,
    pub stage7: CommittedSumcheckWitness<F>,
}

impl<F> ZkStageWitnesses<F> {
    fn in_protocol_order(&self) -> [&CommittedSumcheckWitness<F>; 10] {
        [
            &self.stage1_uniskip,
            &self.stage1,
            &self.stage2_uniskip,
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
            &self.stage6a,
            &self.stage6b,
            &self.stage7,
        ]
    }
}

/// The BlindFold stage domains in the same order: the two uni-skips run over
/// their centered integer domains, every batch over the Boolean hypercube —
/// the constants the stage recipes themselves prove over. The uni-skip sizes
/// are the COMPOSED jolt-r1cs constants (feature-aware): identical to the
/// jolt-claims RV64-only constants without field-inline, and the field-inline-extended row/lane
/// domains under `field-inline` — the domains the verifier's lowering
/// (`stages::zk::blindfold`) builds its round constraints over.
const STAGE_DOMAINS: [SumcheckDomainSpec; 10] = [
    SumcheckDomainSpec::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE),
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
];

/// The stage-8 hiding-opening secrets: the joint evaluation committed inside
/// the PCS's hiding evaluation commitment and its blind.
pub(crate) struct ZkFinalOpening<F> {
    pub joint_evaluation: F,
    pub evaluation_blind: F,
}

/// Prove the BlindFold tail onto `transcript`, which holds the argument string
/// through stage 8. The verifier's stage spine is replayed over that prefix to
/// obtain the BlindFold protocol, and the replay must land on the forward
/// transcript's exact state.
pub(crate) fn prove_blindfold<F, PCS, VC, H>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    witnesses: &ZkStageWitnesses<F>,
    final_opening: &ZkFinalOpening<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalDecode,
    H: Sponge,
    <F as WithAccumulator>::Accumulator: Accumulator<Element = F>,
{
    let mut replay =
        VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, transcript.narg());
    let VerifiedStages::Zk(protocol) = verify_stages::<F, PCS, VC, H>(
        &preprocessing.verifier,
        public_io,
        trusted_advice_commitment,
        &mut replay,
    )?
    else {
        return Err(ProverError::InvariantViolation {
            reason: "the verifier replay of a ZK proof produced no BlindFold protocol",
        });
    };
    // Hard error (not debug-only) so release provers diagnose drift here
    // rather than as a downstream BlindFold verification failure.
    if replay.remaining() != 0
        || replay.preview().squeeze::<32>() != transcript.preview().squeeze::<32>()
    {
        return Err(ProverError::InvariantViolation {
            reason: "the verifier replay diverged from the prover's forward transcript",
        });
    }

    let assigned = protocol.assign_witness(
        &STAGE_DOMAINS,
        &witnesses.in_protocol_order(),
        &[final_opening.joint_evaluation],
        &[final_opening.evaluation_blind],
        &mut rand_core::OsRng,
    )?;

    let vc_setup = preprocessing
        .verifier
        .vc_setup
        .as_ref()
        .ok_or(ProverError::Verifier(
            VerifierError::MissingVectorCommitmentSetup,
        ))?;
    jolt_blindfold::prove::<F, VC, H, _>(
        vc_setup,
        &protocol,
        transcript,
        BlindFoldWitness {
            rows: &assigned.rows,
            blindings: &assigned.blindings,
            eval_outputs: &[final_opening.joint_evaluation],
            eval_blindings: &[final_opening.evaluation_blind],
        },
        &mut rand_core::OsRng,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage table follows the composed geometry, including field-inline lanes when enabled.
    #[test]
    fn stage_domains_use_the_composed_uniskip_constants() {
        use jolt_claims::protocols::jolt::geometry::dimensions::{
            OUTER_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_DOMAIN_SIZE,
        };

        assert_eq!(
            STAGE_DOMAINS.first().copied(),
            Some(SumcheckDomainSpec::centered_integer(
                SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE
            )),
        );
        assert_eq!(
            STAGE_DOMAINS.get(2).copied(),
            Some(SumcheckDomainSpec::centered_integer(
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE
            )),
        );
        for (index, domain) in STAGE_DOMAINS.iter().enumerate() {
            if index != 0 && index != 2 {
                assert_eq!(*domain, SumcheckDomainSpec::BooleanHypercube);
            }
        }

        #[cfg(not(feature = "field-inline"))]
        {
            assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_DOMAIN_SIZE);
            assert_eq!(
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                PRODUCT_UNISKIP_DOMAIN_SIZE
            );
        }
        #[cfg(feature = "field-inline")]
        {
            use jolt_claims::protocols::composed::geometry::{
                SPARTAN_PRODUCT_BASE_LANES, SPARTAN_PRODUCT_FIELD_INLINE_LANES,
            };
            assert_eq!(
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                SPARTAN_PRODUCT_BASE_LANES + SPARTAN_PRODUCT_FIELD_INLINE_LANES
            );
            assert_ne!(
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                PRODUCT_UNISKIP_DOMAIN_SIZE
            );
            assert_ne!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_DOMAIN_SIZE);
        }
    }
}
