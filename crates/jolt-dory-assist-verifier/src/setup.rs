//! Dory-assist Hyrax setup derivation.

use jolt_crypto::{GrumpkinPedersenSetupSeed, GrumpkinPoint, Pedersen};
use jolt_hyrax::{HyraxDimensions, HyraxError, HyraxProverSetup, HyraxScheme, HyraxVerifierSetup};

pub type DoryAssistHyrax = HyraxScheme<Pedersen<GrumpkinPoint>>;
pub type DoryAssistHyraxProverSetup = HyraxProverSetup<Pedersen<GrumpkinPoint>>;
pub type DoryAssistHyraxVerifierSetup = HyraxVerifierSetup<Pedersen<GrumpkinPoint>>;

pub const DORY_ASSIST_HYRAX_GRUMPKIN_DOMAIN: &[u8] = b"JoltDoryAssistHyraxGrumpkin";
pub const DORY_ASSIST_HYRAX_GRUMPKIN_SEED: &[u8] = b"v1";
pub const DORY_ASSIST_HYRAX_GRUMPKIN_SETUP_SEED: GrumpkinPedersenSetupSeed<'static> =
    GrumpkinPedersenSetupSeed::new(
        DORY_ASSIST_HYRAX_GRUMPKIN_DOMAIN,
        DORY_ASSIST_HYRAX_GRUMPKIN_SEED,
    );

pub fn derive_hyrax_prover_setup(
    dimensions: HyraxDimensions,
) -> Result<DoryAssistHyraxProverSetup, HyraxError> {
    DoryAssistHyraxProverSetup::derive_from(&DORY_ASSIST_HYRAX_GRUMPKIN_SETUP_SEED, dimensions)
}

pub fn derive_hyrax_verifier_setup(
    dimensions: HyraxDimensions,
) -> Result<DoryAssistHyraxVerifierSetup, HyraxError> {
    DoryAssistHyraxVerifierSetup::derive_from(&DORY_ASSIST_HYRAX_GRUMPKIN_SETUP_SEED, dimensions)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_field::{Fq, FromPrimitiveInt};
    use jolt_openings::CommitmentScheme;
    use jolt_poly::Polynomial;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use super::*;

    fn dimensions() -> HyraxDimensions {
        HyraxDimensions::new(3, 1, 2).expect("valid dimensions")
    }

    fn polynomial() -> Polynomial<Fq> {
        Polynomial::from(
            (0..8)
                .map(|index| Fq::from_u64(index + 3))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn dory_assist_seed_derives_matching_hyrax_setups() {
        let dimensions = dimensions();
        let prover_setup =
            derive_hyrax_prover_setup(dimensions).expect("derive prover setup from seed");
        let verifier_setup =
            derive_hyrax_verifier_setup(dimensions).expect("derive verifier setup from seed");

        assert_eq!(prover_setup.dimensions, verifier_setup.dimensions);
        assert_eq!(prover_setup.vc_setup, verifier_setup.vc_setup);
        assert_eq!(prover_setup.vc_setup.message_generators.len(), 4);
    }

    #[test]
    fn seed_derived_setup_verifies_hyrax_opening() {
        let dimensions = dimensions();
        let prover_setup =
            derive_hyrax_prover_setup(dimensions).expect("derive prover setup from seed");
        let verifier_setup =
            derive_hyrax_verifier_setup(dimensions).expect("derive verifier setup from seed");
        let poly = polynomial();
        let point = vec![Fq::from_u64(2), Fq::from_u64(3), Fq::from_u64(5)];
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryAssistHyrax::commit(&poly, &prover_setup);

        let mut prover_transcript = Blake2bTranscript::new(b"dory-assist-hyrax-seed");
        let proof = DoryAssistHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"dory-assist-hyrax-seed");
        DoryAssistHyrax::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect("seed-derived Hyrax opening verifies");
    }
}
