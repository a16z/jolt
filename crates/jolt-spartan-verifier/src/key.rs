use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use jolt_r1cs::ConstraintMatrices;
use jolt_transcript::{AppendToTranscript, Transcript, U64Word};

use crate::SpartanError;

/// Checked immutable relation. Columns are `[1, public inputs, witness]`.
///
/// `policy_id` identifies the application's authenticated PCS/transcript policy.
/// Matrix contents and shape are additionally bound verbatim in every proof.
/// The constructor validates shape; it does not authenticate an application.
#[derive(Clone, Debug)]
pub struct SpartanKey<F: JoltField> {
    matrices: ConstraintMatrices<F>,
    public_columns: usize,
    padded_rows: usize,
    padded_witness: usize,
    policy_id: [u8; 32],
}

impl<F: JoltField> SpartanKey<F> {
    /// Checks matrix dimensions, partition, and power-of-two padding sizes.
    /// Requires nonempty constraints/private witness and characteristic > 3.
    pub fn new(
        matrices: ConstraintMatrices<F>,
        public_input_count: usize,
        policy_id: [u8; 32],
    ) -> Result<Self, SpartanError<F>> {
        let public_columns = public_input_count
            .checked_add(1)
            .ok_or(SpartanError::InvalidShape)?;
        let witness_len = matrices
            .num_vars
            .checked_sub(public_columns)
            .filter(|len| *len > 0)
            .ok_or(SpartanError::InvalidShape)?;
        if F::from_u64(2).is_zero() || F::from_u64(3).is_zero() {
            return Err(SpartanError::UnsupportedField);
        }
        if matrices.num_constraints == 0 {
            return Err(SpartanError::InvalidShape);
        }
        for matrix in [&matrices.a, &matrices.b, &matrices.c] {
            if matrix.len() != matrices.num_constraints
                || matrix
                    .iter()
                    .flatten()
                    .any(|(column, _)| *column >= matrices.num_vars)
            {
                return Err(SpartanError::InvalidShape);
            }
        }
        let padded_rows = matrices
            .num_constraints
            .max(2)
            .checked_next_power_of_two()
            .ok_or(SpartanError::InvalidShape)?;
        let padded_witness = witness_len
            .max(2)
            .checked_next_power_of_two()
            .ok_or(SpartanError::InvalidShape)?;
        Ok(Self {
            matrices,
            public_columns,
            padded_rows,
            padded_witness,
            policy_id,
        })
    }

    /// Original unpadded matrices, available immutably to the prover.
    pub fn matrices(&self) -> &ConstraintMatrices<F> {
        &self.matrices
    }
    /// Number of public columns, including the constant-one column.
    pub fn public_columns(&self) -> usize {
        self.public_columns
    }
    /// Number of application-supplied private coordinates, before padding.
    pub fn witness_len(&self) -> usize {
        self.matrices.num_vars - self.public_columns
    }
    /// Boolean-hypercube size of the outer sumcheck, including zero rows.
    pub fn padded_rows(&self) -> usize {
        self.padded_rows
    }
    /// Committed witness table size, including unused private columns.
    pub fn padded_witness_len(&self) -> usize {
        self.padded_witness
    }
    /// Number of outer sumcheck rounds.
    pub fn row_vars(&self) -> usize {
        self.padded_rows.trailing_zeros() as usize
    }
    /// Number of inner sumcheck rounds and witness opening coordinates.
    pub fn witness_vars(&self) -> usize {
        self.padded_witness.trailing_zeros() as usize
    }

    /// Checks the public vector length against this key's fixed partition.
    pub fn validate_public_inputs(&self, public_inputs: &[F]) -> Result<(), SpartanError<F>> {
        if public_inputs.len() != self.public_columns - 1 {
            return Err(SpartanError::PublicInputs);
        }
        Ok(())
    }

    /// Check the outer sumcheck's final relation; shared with the prover.
    pub fn check_outer(
        &self,
        tau: &[F],
        rx: &[F],
        claim: F,
        evaluations: [F; 3],
    ) -> Result<(), SpartanError<F>> {
        check_outer_relation(self.row_vars(), tau, rx, claim, evaluations)
    }
}

impl<F: JoltField + AppendToTranscript> SpartanKey<F> {
    /// Bind the authenticated relation, public input, and commitment before tau.
    pub fn begin<C: AppendToTranscript>(
        &self,
        public_inputs: &[F],
        commitment: &C,
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Result<Vec<F>, SpartanError<F>> {
        self.validate_public_inputs(public_inputs)?;
        transcript.append_labeled(
            b"spartan-clear-v1",
            &U64Word(self.matrices.num_constraints as u64),
        );
        transcript.append_bytes(&self.policy_id);
        transcript.append(&U64Word(self.matrices.num_vars as u64));
        transcript.append(&U64Word(self.public_columns as u64));
        for matrix in [&self.matrices.a, &self.matrices.b, &self.matrices.c] {
            for row in matrix {
                transcript.append(&U64Word(row.len() as u64));
                for (column, coefficient) in row {
                    transcript.append(&U64Word(*column as u64));
                    transcript.append(coefficient);
                }
            }
        }
        transcript.append_values(b"public-inputs", public_inputs);
        transcript.append_labeled(b"witness-commitment", commitment);
        let tau = transcript.challenge_vector(self.row_vars());
        transcript.append_labeled(b"spartan-outer", &F::zero());
        Ok(tau)
    }

    /// Absorb outer claims, draw matrix weights, and bind the inner input claim.
    pub fn begin_inner(
        &self,
        row_weights: &[F],
        public_inputs: &[F],
        evaluations: [F; 3],
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Result<([F; 3], F), SpartanError<F>> {
        self.validate_public_inputs(public_inputs)?;
        transcript.append_values(b"outer-evaluations", &evaluations);
        let weights = [
            transcript.challenge(),
            transcript.challenge(),
            transcript.challenge(),
        ];
        let public = std::iter::once(F::one())
            .chain(public_inputs.iter().copied())
            .collect::<Vec<_>>();
        let contribution = self.matrices.linear_form_bilinear_eval(
            row_weights,
            &public,
            0,
            self.public_columns,
            weights,
        )?;
        let claim = weights
            .iter()
            .zip(evaluations)
            .map(|(weight, evaluation)| *weight * evaluation)
            .sum::<F>()
            - contribution;
        transcript.append_labeled(b"spartan-inner", &claim);
        Ok((weights, claim))
    }

    /// Binds the terminal witness claim before the PCS opening transcript.
    pub fn append_witness_evaluation(
        evaluation: F,
        transcript: &mut impl Transcript<Challenge = F>,
    ) {
        transcript.append_labeled(b"witness-evaluation", &evaluation);
    }
}

/// The outer terminal identity is shared by v1 and preprocessed v2.
pub(crate) fn check_outer_relation<F: JoltField>(
    row_vars: usize,
    tau: &[F],
    rx: &[F],
    claim: F,
    evaluations: [F; 3],
) -> Result<(), SpartanError<F>> {
    if tau.len() != row_vars || rx.len() != row_vars {
        return Err(SpartanError::InternalShape);
    }
    let [a, b, c] = evaluations;
    if claim != EqPolynomial::new(tau.to_vec()).evaluate(rx) * (a * b - c) {
        return Err(SpartanError::OuterClaim);
    }
    Ok(())
}
