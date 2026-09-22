use super::PreprocessedMatrices;
use crate::{
    prove_rounds,
    rounds::{InnerRounds, OuterRounds},
};
use jolt_field::{Fr, One, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme};
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_spartan_verifier::preprocessed::sparse::SparseQuery;
use jolt_spartan_verifier::preprocessed::{ComputationKey, MatrixError, PreprocessedProof};
use jolt_spartan_verifier::{SpartanError, SpartanKey, INNER_DEGREE, OUTER_DEGREE};
use jolt_transcript::{Bn254WideBlake2bTranscript, Transcript};

impl PreprocessedMatrices {
    /// Proves the actual fixed relation using all four matrix PCS openings and
    /// the original witness opening. Clear, conditional correctness prototype.
    pub fn prove(
        &self,
        public_inputs: &[Fr],
        witness: &[Fr],
        setup: &HyperKZGProverSetup,
    ) -> Result<PreprocessedProof, MatrixError> {
        let direct = &self.direct;
        direct.validate_public_inputs(public_inputs)?;
        if witness.len() != direct.witness_len() {
            return Err(SpartanError::WitnessLength.into());
        }
        let assignment = std::iter::once(Fr::one())
            .chain(public_inputs.iter().copied())
            .chain(witness.iter().copied())
            .collect::<Vec<_>>();
        let mut witness = witness.to_vec();
        witness.resize(direct.padded_witness_len(), Fr::zero());
        let polynomial = Polynomial::new(witness.clone());
        let (witness_commitment, hint) = HyperKZGScheme::commit(&polynomial, setup)?;
        let mut transcript = Bn254WideBlake2bTranscript::new(b"spartan-preprocessed-clear-v2");
        let tau = self.key.begin(
            &self.key.id(),
            &HyperKZGScheme::verifier_setup(setup),
            public_inputs,
            &witness_commitment,
            &mut transcript,
        )?;
        let mut outer_rounds = OuterRounds::new(direct, &assignment, &tau)?;
        let (outer, rx, claim) =
            prove_rounds(&mut outer_rounds, OUTER_DEGREE, Fr::zero(), &mut transcript)?;
        let outer_evaluations = outer_rounds.evaluations()?;
        direct.check_outer(&tau, &rx, claim, outer_evaluations)?;
        let weights = ComputationKey::outer_weights(&outer_evaluations, &mut transcript);
        let public = self.prove_public(&rx, setup, &mut transcript)?;
        let row_weights = EqPolynomial::new(rx.clone()).evaluations();
        let public_assignment = std::iter::once(Fr::one())
            .chain(public_inputs.iter().copied())
            .collect::<Vec<_>>();
        let contribution = direct
            .matrices()
            .linear_form_bilinear_eval(
                &row_weights,
                &public_assignment,
                0,
                direct.public_columns(),
                weights,
            )
            .map_err(SpartanError::from)?;
        let claim =
            ComputationKey::inner_claim(outer_evaluations, weights, contribution, &mut transcript);
        let mut linear = direct
            .matrices()
            .project_column_range(
                &row_weights,
                direct.public_columns(),
                direct.witness_len(),
                weights,
            )
            .map_err(SpartanError::from)?;
        linear.resize(direct.padded_witness_len(), Fr::zero());
        let mut inner_rounds = InnerRounds::new(linear, witness, direct.witness_vars());
        let (inner, ry, claim) =
            prove_rounds(&mut inner_rounds, INNER_DEGREE, claim, &mut transcript)?;
        let [linear, witness_evaluation] = inner_rounds.evaluations()?;
        if claim != linear * witness_evaluation {
            return Err(MatrixError::Relation("honest inner product"));
        }
        let column_weights = EqPolynomial::new(ry.clone()).evaluations();
        let column_weights = column_weights
            .get(..direct.witness_len())
            .ok_or(MatrixError::Shape)?;
        let mut private_values = [Fr::zero(); 3];
        for (matrix, value) in private_values.iter_mut().enumerate() {
            let mut selector = [Fr::zero(); 3];
            *selector.get_mut(matrix).ok_or(MatrixError::Shape)? = Fr::one();
            *value = direct
                .matrices()
                .linear_form_bilinear_eval(
                    &row_weights,
                    column_weights,
                    direct.public_columns(),
                    direct.witness_len(),
                    selector,
                )
                .map_err(SpartanError::from)?;
        }
        let sparse = self.prove_sparse(
            SparseQuery {
                rows: &rx,
                columns: &ry,
                values: private_values,
            },
            setup,
            &mut transcript,
        )?;
        SpartanKey::append_witness_evaluation(witness_evaluation, &mut transcript);
        let witness_opening = HyperKZGScheme::open(
            &polynomial,
            &ry,
            witness_evaluation,
            setup,
            Some(hint),
            &mut transcript,
        )?;
        Ok(PreprocessedProof {
            witness_commitment,
            outer,
            outer_evaluations,
            public,
            inner,
            private_values,
            sparse,
            witness_evaluation,
            witness_opening,
        })
    }
}
