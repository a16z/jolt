//! Complete transparent stage1 remainder, conditional on shared upstream handles.
use jolt_akita::{AkitaField, AkitaScheme};
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_claims::protocols::jolt::r1cs::fp128::{
    OuterRemainderR1csError, SpartanOuterRemainderR1cs, SpartanOuterRemainderVars,
};
use jolt_claims::protocols::jolt::r1cs::SPARTAN_OUTER_REMAINDER_DEGREE;
use jolt_claims::protocols::jolt::relations::spartan::OuterRemainderOutputClaims;
use jolt_crypto::VectorCommitment;
use jolt_field::Fr;
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::R1csBuilder;
use jolt_sumcheck::r1cs::fp128::{Fp128CompressedRoundShape, Fp128SumcheckError};
use jolt_sumcheck::{
    ClearProof, SumcheckProof, OPENING_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_CLAIM_TRANSCRIPT_LABEL,
};
use jolt_transcript::r1cs::{Blake2bR1csError, Fp128TranscriptError};
use thiserror::Error;

use super::AkitaStage1BoundaryVars;
use crate::{JoltProof, VerifierError};

#[derive(Debug, Error)]
pub enum Stage1RemainderError {
    #[error("unsupported fixed stage1 remainder shape")]
    Shape,
    #[error(transparent)]
    Native(#[from] VerifierError),
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Round(#[from] Fp128SumcheckError),
    #[error(transparent)]
    Formula(#[from] OuterRemainderR1csError),
    #[error(transparent)]
    Transcript(#[from] Fp128TranscriptError),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
}

pub struct Stage1RemainderWitness {
    rounds: [[u128; 3]; 13],
    openings: OuterRemainderOutputClaims<u128>,
}
/// Typed virtual openings remain obligations for later stages and final PCS discharge.
pub struct Stage1RemainderVars {
    pub openings: OuterRemainderOutputClaims<Fp128Var>,
    pub point: Vec<Fp128Var>,
    pub rho: Fp128Var,
    pub running_claim: Fp128Var,
    pub formula: SpartanOuterRemainderVars,
}
impl Stage1RemainderVars {
    pub fn opening_point(&self) -> impl Iterator<Item = &Fp128Var> {
        self.point.iter().rev()
    }
    pub fn cycle_binding(&self) -> impl Iterator<Item = &Fp128Var> {
        self.point.iter().skip(1)
    }
}

pub struct Stage1RemainderShape {
    round: Fp128CompressedRoundShape,
    formula: SpartanOuterRemainderR1cs,
}
impl Stage1RemainderShape {
    pub fn new() -> Result<Self, Stage1RemainderError> {
        if SpartanOuterDimensions::rv64(12).remainder_rounds() != 13
            || SPARTAN_OUTER_REMAINDER_DEGREE != 3
        {
            return Err(Stage1RemainderError::Shape);
        }
        Ok(Self {
            round: Fp128CompressedRoundShape::new(3, SPARTAN_OUTER_REMAINDER_DEGREE)?,
            formula: SpartanOuterRemainderR1cs::new(12)?,
        })
    }
    /// Fixed profile descriptor appended to the application-owned key identity.
    pub fn profile_descriptor(&self) -> [u8; 2] {
        [13, 3]
    }
    pub fn witness<VC, Zk>(
        &self,
        proof: &JoltProof<AkitaScheme, VC, Zk>,
    ) -> Result<Stage1RemainderWitness, Stage1RemainderError>
    where
        VC: VectorCommitment<Field = AkitaField>,
    {
        let SumcheckProof::Clear(ClearProof::Compressed(compressed)) =
            &proof.stages.stage1_sumcheck_proof
        else {
            return Err(Stage1RemainderError::Shape);
        };
        if compressed.round_polynomials.len() != 13
            || compressed
                .round_polynomials
                .iter()
                .any(|round| round.coeffs_except_linear_term().len() != 3)
        {
            return Err(Stage1RemainderError::Shape);
        }
        let mut rounds = [[0; 3]; 13];
        for (destination, source) in rounds.iter_mut().zip(&compressed.round_polynomials) {
            for (destination, source) in destination
                .iter_mut()
                .zip(source.coeffs_except_linear_term())
            {
                *destination = source.to_canonical_u128();
            }
        }
        let openings = proof
            .clear_claims()?
            .stage1
            .outer
            .outer_remainder
            .try_map_cells(|_, value| Ok::<_, Stage1RemainderError>(value.to_canonical_u128()))?;
        Ok(Stage1RemainderWitness { rounds, openings })
    }
    /// Continue the same builder/transcript; no new public inputs or checkpoint witness.
    pub fn constrain(
        &self,
        builder: &mut R1csBuilder<Fr>,
        upstream: &mut AkitaStage1BoundaryVars,
        witness: Option<&Stage1RemainderWitness>,
    ) -> Result<Stage1RemainderVars, Stage1RemainderError> {
        if upstream.uniskip.tau.len() != 14 {
            return Err(Stage1RemainderError::Shape);
        }
        for value in upstream
            .uniskip
            .tau
            .iter()
            .chain([&upstream.uniskip.challenge, &upstream.uniskip.claim])
        {
            value.validate_indices(builder)?;
        }
        upstream.transcript.check_schedule(138)?;
        let unknown = OuterRemainderOutputClaims::<u128>::default();
        let source = witness.map_or(&unknown, |w| &w.openings);
        let openings = source
            .try_map_cells(|_, value| Fp128Var::allocate(builder, witness.map(|_| *value)))?;
        upstream
            .transcript
            .append_label(builder, SUMCHECK_CLAIM_TRANSCRIPT_LABEL)?;
        upstream
            .transcript
            .append_fp128(builder, &upstream.uniskip.claim)?;
        let rho = upstream.transcript.challenge_scalar_fp128(builder)?;
        let mut running_claim = rho.multiply(builder, &upstream.uniskip.claim)?;
        let mut point = Vec::with_capacity(13);
        for index in 0..13 {
            let row = witness.and_then(|w| w.rounds.get(index));
            let coefficients = (0..3)
                .map(|column| Fp128Var::allocate(builder, row.and_then(|r| r.get(column)).copied()))
                .collect::<Result<Vec<_>, _>>()?;
            let reduction = self.round.constrain(
                builder,
                &mut upstream.transcript,
                &coefficients,
                &running_claim,
            )?;
            point.push(reduction.challenge);
            running_claim = reduction.claim;
        }
        let formula = self.formula.constrain(
            builder,
            &upstream.uniskip.tau,
            &upstream.uniskip.challenge,
            &point,
            &openings,
        )?;
        let expected = rho.multiply(builder, &formula.claim)?;
        builder.assert_equal(running_claim.variable(), expected.variable());
        let _ = openings.try_map_cells(|_, value| {
            upstream
                .transcript
                .append_label(builder, OPENING_CLAIM_TRANSCRIPT_LABEL)?;
            upstream.transcript.append_fp128(builder, value)?;
            Ok::<(), Stage1RemainderError>(())
        })?;
        Ok(Stage1RemainderVars {
            openings,
            point,
            rho,
            running_claim,
            formula,
        })
    }
}

#[cfg(all(test, feature = "prover-fixtures"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "recorded proof and exact private-cell parity tests"
)]
mod tests {
    use super::super::{boundary::tests::fingerprint, AkitaStage1BoundaryShape};
    use super::*;
    use crate::JoltVerifierPreprocessing;
    use common::jolt_device::JoltDevice;
    use jolt_field::{CanonicalEncoding, Ring};
    use jolt_prover_legacy::zkvm::packed::AkitaVc;
    use serde::de::DeserializeOwned;
    use serde_json::Value;
    use std::fmt::Write;
    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> T {
        let (value, used) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
        assert_eq!(used, bytes.len());
        value
    }
    fn fixture() -> (
        JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        JoltDevice,
        JoltProof<AkitaScheme, AkitaVc>,
    ) {
        (
            decode(include_bytes!(
                "../../tests/fixtures/akita-boundary/preprocessing.bin"
            )),
            decode(include_bytes!(
                "../../tests/fixtures/akita-boundary/public-io.bin"
            )),
            decode(include_bytes!(
                "../../tests/fixtures/akita-boundary/proof.bin"
            )),
        )
    }
    fn scalar(v: &Value) -> Fr {
        Fr::from_u128(
            u128::from_str_radix(v.as_str().unwrap().strip_prefix("0x").unwrap(), 16).unwrap(),
        )
    }
    #[test]
    fn stage1_complete_recorded_parity_and_frozen_mutations() {
        let (pre, io, proof) = fixture();
        let boundary = AkitaStage1BoundaryShape::new(&pre, &io, &proof).unwrap();
        let shape = Stage1RemainderShape::new().unwrap();
        let mut builder = R1csBuilder::new();
        let mut upstream = boundary
            .constrain(&mut builder, Some(&boundary.witness(&io, &proof).unwrap()))
            .unwrap();
        let output = shape
            .constrain(
                &mut builder,
                &mut upstream,
                Some(&shape.witness(&proof).unwrap()),
            )
            .unwrap();
        let expected: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/akita-boundary/stage1-vectors.json"
        ))
        .unwrap();
        let mut assignment = builder.witness().unwrap();
        for (value, name) in [
            (&output.rho, "rho"),
            (&output.formula.kernel, "tau_kernel"),
            (&output.formula.az, "az"),
            (&output.formula.bz, "bz"),
            (&output.running_claim, "running_final"),
        ] {
            assert_eq!(
                assignment[value.variable().index()],
                scalar(&expected[name])
            );
        }
        let native_point = expected["remainder_point"].as_array().unwrap();
        assert_eq!(native_point.len(), 13);
        assert_eq!(output.point.len(), native_point.len());
        assert_eq!(output.opening_point().count(), 13);
        for (value, native) in output.opening_point().zip(native_point.iter().rev()) {
            assert_eq!(assignment[value.variable().index()], scalar(native));
        }
        assert_eq!(output.cycle_binding().count(), 12);
        for (value, native) in output.cycle_binding().zip(native_point.iter().skip(1)) {
            assert_eq!(assignment[value.variable().index()], scalar(native));
        }
        for (value, native) in output.point.iter().zip(native_point) {
            assert_eq!(assignment[value.variable().index()], scalar(native));
        }
        let native_openings = expected["opening_values"].as_array().unwrap();
        assert_eq!(native_openings.len(), 35);
        let mut openings = native_openings.iter();
        let _ = output
            .openings
            .try_map_cells(|_, value| {
                assert_eq!(
                    assignment[value.variable().index()],
                    scalar(openings.next().unwrap())
                );
                Ok::<(), ()>(())
            })
            .unwrap();
        assert!(openings.next().is_none());
        let mut state = String::new();
        for byte in upstream.transcript.state() {
            write!(
                &mut state,
                "{:02x}",
                builder
                    .evaluate(&byte.expression())
                    .unwrap()
                    .to_u64_checked()
                    .unwrap()
            )
            .unwrap();
        }
        assert_eq!(state, expected["state_after_stage1"].as_str().unwrap());
        let matrices = builder.into_matrices();
        matrices.check_witness(&assignment).unwrap();
        for variable in [
            upstream.public,
            upstream.coefficients[0].variable(),
            output.rho.variable(),
            output.point[0].variable(),
            output.openings.product.variable(),
        ] {
            assignment[variable.index()] += Fr::from_u64(1);
            assert!(matrices.check_witness(&assignment).is_err());
            assignment[variable.index()] -= Fr::from_u64(1);
        }
    }
    #[test]
    fn stage1_complete_unknown_shape_matches_recorded() {
        let (pre, io, proof) = fixture();
        let boundary = AkitaStage1BoundaryShape::new(&pre, &io, &proof).unwrap();
        let shape = Stage1RemainderShape::new().unwrap();
        let known = {
            let mut builder = R1csBuilder::new();
            let mut upstream = boundary
                .constrain(&mut builder, Some(&boundary.witness(&io, &proof).unwrap()))
                .unwrap();
            let _ = shape
                .constrain(
                    &mut builder,
                    &mut upstream,
                    Some(&shape.witness(&proof).unwrap()),
                )
                .unwrap();
            fingerprint(&builder.into_matrices())
        };
        let mut builder = R1csBuilder::new();
        let mut upstream = boundary.constrain(&mut builder, None).unwrap();
        let _ = shape.constrain(&mut builder, &mut upstream, None).unwrap();
        assert_eq!(known, fingerprint(&builder.into_matrices()));
    }
    #[test]
    fn stage1_complete_changed_opening_with_regenerated_auxiliaries_rejects() {
        let (pre, io, proof) = fixture();
        let boundary = AkitaStage1BoundaryShape::new(&pre, &io, &proof).unwrap();
        let shape = Stage1RemainderShape::new().unwrap();
        let mut witness = shape.witness(&proof).unwrap();
        witness.openings.product ^= 1;
        let mut builder = R1csBuilder::new();
        let mut upstream = boundary
            .constrain(&mut builder, Some(&boundary.witness(&io, &proof).unwrap()))
            .unwrap();
        let _ = shape
            .constrain(&mut builder, &mut upstream, Some(&witness))
            .unwrap();
        let assignment = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&assignment).is_err());
    }
}

#[cfg(all(test, feature = "prover-fixtures"))]
#[expect(
    clippy::unwrap_used,
    reason = "recorded fixed-profile shape rejection before synthesis"
)]
mod profile_tests {
    use super::*;
    use jolt_poly::CompressedPoly;
    use jolt_prover_legacy::zkvm::packed::AkitaVc;
    #[test]
    fn stage1_remainder_rejects_native_shorter_encoding_before_allocation() {
        let (mut proof, used): (JoltProof<AkitaScheme, AkitaVc>, usize) =
            bincode::serde::decode_from_slice(
                include_bytes!("../../tests/fixtures/akita-boundary/proof.bin"),
                bincode::config::standard(),
            )
            .unwrap();
        assert_eq!(
            used,
            include_bytes!("../../tests/fixtures/akita-boundary/proof.bin").len()
        );
        let shape = Stage1RemainderShape::new().unwrap();
        let valid = proof.stages.stage1_sumcheck_proof.clone();
        let _ = shape.witness(&proof).unwrap();
        if let SumcheckProof::Clear(ClearProof::Compressed(rounds)) =
            &mut proof.stages.stage1_sumcheck_proof
        {
            *rounds.round_polynomials.first_mut().unwrap() = CompressedPoly::new(Vec::new());
        }
        assert!(matches!(
            shape.witness(&proof),
            Err(Stage1RemainderError::Shape)
        ));
        if let SumcheckProof::Clear(ClearProof::Compressed(rounds)) =
            &mut proof.stages.stage1_sumcheck_proof
        {
            *rounds.round_polynomials.first_mut().unwrap() =
                CompressedPoly::new(vec![AkitaField::default()]);
        }
        assert!(matches!(
            shape.witness(&proof),
            Err(Stage1RemainderError::Shape)
        ));
        proof.stages.stage1_sumcheck_proof = valid;
        let _ = shape.witness(&proof).unwrap();
        if let SumcheckProof::Clear(ClearProof::Compressed(rounds)) =
            &mut proof.stages.stage1_sumcheck_proof
        {
            let _ = rounds.round_polynomials.pop();
        }
        assert!(matches!(
            shape.witness(&proof),
            Err(Stage1RemainderError::Shape)
        ));
    }
}
