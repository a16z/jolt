use jolt_claims::protocols::wrapper_spartan_hyperkzg::{
    SpartanRelationDimensions, WrapperPublicInputLayout, WrapperRelationDimensions,
    WrapperSpartanHyperKzgStatementFacts, WRAPPER_PROTOCOL_ID_TRANSCRIPT_LABEL,
    WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_LABEL, WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_PREFIX,
    WRAPPER_PUBLIC_INPUT_LAYOUT_TRANSCRIPT_LABEL, WRAPPER_RELATION_DIMS_TRANSCRIPT_LABEL,
    WRAPPER_RELATION_MATRICES_TRANSCRIPT_LABEL, WRAPPER_RELATION_MATRIX_A_TRANSCRIPT_LABEL,
    WRAPPER_RELATION_MATRIX_B_TRANSCRIPT_LABEL, WRAPPER_RELATION_MATRIX_C_TRANSCRIPT_LABEL,
    WRAPPER_RELATION_MATRIX_ROW_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_DIMS_TRANSCRIPT_LABEL,
    WRAPPER_STATEMENT_TRANSCRIPT_LABEL,
};
use jolt_field::Field;
use jolt_r1cs::{ConstraintMatrices, SparseRow};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use crate::{stages::r1cs_relation::inputs::R1csRelationInputs, WrapperError};

use super::outputs::R1csRelationOutput;

pub fn verify<'a, F, T>(
    inputs: R1csRelationInputs<'a, F>,
    transcript: &mut T,
) -> Result<R1csRelationOutput<'a, F>, WrapperError>
where
    F: Field + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let relation = WrapperRelationDimensions::new(
        inputs.relation.num_vars,
        inputs.relation.num_constraints,
        inputs.public_inputs.len(),
    );
    if inputs.proof_relation.dimensions != relation {
        return Err(WrapperError::R1csRelationMismatch {
            expected: relation,
            actual: inputs.proof_relation.dimensions,
        });
    }

    let statement_facts = WrapperSpartanHyperKzgStatementFacts::from_relation_dimensions(relation)
        .map_err(|error| WrapperError::InvalidR1csRelationFacts {
            reason: error.to_string(),
        })?;

    absorb_statement_facts(transcript, &statement_facts)?;
    absorb_relation_matrices(transcript, inputs.relation)?;
    absorb_public_inputs(transcript, inputs.public_inputs)?;

    Ok(R1csRelationOutput {
        statement_facts,
        relation: inputs.relation,
        public_inputs: inputs.public_inputs.to_vec(),
    })
}

fn absorb_statement_facts<T>(
    transcript: &mut T,
    facts: &WrapperSpartanHyperKzgStatementFacts,
) -> Result<(), WrapperError>
where
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_STATEMENT_TRANSCRIPT_LABEL));
    transcript.append(&LabelWithCount(
        WRAPPER_PROTOCOL_ID_TRANSCRIPT_LABEL,
        usize_to_u64("protocol_id_len", facts.protocol_id.len())?,
    ));
    transcript.append_bytes(facts.protocol_id);

    absorb_relation_dimensions(transcript, facts.relation)?;
    absorb_spartan_dimensions(transcript, facts.spartan)?;
    absorb_public_input_layout(transcript, facts.spartan.public_inputs_layout())?;

    Ok(())
}

fn absorb_relation_dimensions<T>(
    transcript: &mut T,
    relation: WrapperRelationDimensions,
) -> Result<(), WrapperError>
where
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_RELATION_DIMS_TRANSCRIPT_LABEL));
    transcript.append(&U64Word(usize_to_u64(
        "relation.variables",
        relation.variables,
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "relation.constraints",
        relation.constraints,
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "relation.public_inputs",
        relation.public_inputs,
    )?));
    Ok(())
}

fn absorb_spartan_dimensions<T>(
    transcript: &mut T,
    spartan: SpartanRelationDimensions,
) -> Result<(), WrapperError>
where
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_SPARTAN_DIMS_TRANSCRIPT_LABEL));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.variables.raw",
        spartan.num_vars(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.variables.padded",
        spartan.num_vars_padded(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.variables.log_padded",
        spartan.num_var_rounds(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.constraints.raw",
        spartan.num_constraints(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.constraints.padded",
        spartan.num_constraints_padded(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.constraints.log_padded",
        spartan.num_constraint_rounds(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "spartan.public_inputs",
        spartan.num_public_inputs(),
    )?));
    Ok(())
}

fn absorb_public_input_layout<T>(
    transcript: &mut T,
    layout: WrapperPublicInputLayout,
) -> Result<(), WrapperError>
where
    T: Transcript,
{
    let wrapper_inputs = layout.wrapper_inputs();
    transcript.append(&Label(WRAPPER_PUBLIC_INPUT_LAYOUT_TRANSCRIPT_LABEL));
    transcript.append(&U64Word(usize_to_u64(
        "public_inputs.start",
        wrapper_inputs.start(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "public_inputs.len",
        wrapper_inputs.len(),
    )?));
    transcript.append(&U64Word(usize_to_u64(
        "public_inputs.total",
        layout.total(),
    )?));
    Ok(())
}

fn absorb_relation_matrices<F, T>(
    transcript: &mut T,
    relation: &ConstraintMatrices<F>,
) -> Result<(), WrapperError>
where
    F: AppendToTranscript + Field,
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_RELATION_MATRICES_TRANSCRIPT_LABEL));
    absorb_matrix(
        transcript,
        WRAPPER_RELATION_MATRIX_A_TRANSCRIPT_LABEL,
        &relation.a,
    )?;
    absorb_matrix(
        transcript,
        WRAPPER_RELATION_MATRIX_B_TRANSCRIPT_LABEL,
        &relation.b,
    )?;
    absorb_matrix(
        transcript,
        WRAPPER_RELATION_MATRIX_C_TRANSCRIPT_LABEL,
        &relation.c,
    )?;
    Ok(())
}

fn absorb_matrix<F, T>(
    transcript: &mut T,
    label: &'static [u8],
    rows: &[SparseRow<F>],
) -> Result<(), WrapperError>
where
    F: AppendToTranscript,
    T: Transcript,
{
    transcript.append(&LabelWithCount(
        label,
        usize_to_u64("matrix.rows", rows.len())?,
    ));
    for row in rows {
        transcript.append(&LabelWithCount(
            WRAPPER_RELATION_MATRIX_ROW_TRANSCRIPT_LABEL,
            usize_to_u64("matrix.row.len", row.len())?,
        ));
        for &(column, ref coefficient) in row {
            transcript.append(&U64Word(usize_to_u64("matrix.column", column)?));
            transcript.append(coefficient);
        }
    }
    Ok(())
}

fn absorb_public_inputs<F, T>(transcript: &mut T, public_inputs: &[F]) -> Result<(), WrapperError>
where
    F: AppendToTranscript,
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_PREFIX));
    transcript.append(&LabelWithCount(
        WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_LABEL,
        usize_to_u64("public_inputs.len", public_inputs.len())?,
    ));
    for public_input in public_inputs {
        transcript.append(public_input);
    }
    Ok(())
}

fn usize_to_u64(dimension: &'static str, value: usize) -> Result<u64, WrapperError> {
    u64::try_from(value)
        .map_err(|_| WrapperError::R1csRelationDimensionTooLarge { dimension, value })
}
