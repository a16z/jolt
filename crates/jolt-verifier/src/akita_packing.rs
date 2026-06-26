use std::collections::BTreeSet;

use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint, AkitaProverSetup, AkitaScheme,
};
use jolt_openings::{
    has_packing_view, packing_witness_source_polynomial, prove_sparse_packing_reduction,
    validate_packing_source_dimension, validate_packing_source_layout, validate_packing_statement,
    BatchOpeningScheme, BatchOpeningStatement, OpeningsError, PackingBatch, PackingBatchProof,
    PackingProverSetup, PackingSource, PackingWitnessLayout, PackingWitnessSource,
};
use jolt_transcript::Transcript;

pub(crate) type AkitaPackingScheme = PackingBatch<AkitaScheme, PackingWitnessLayout>;
pub(crate) type AkitaPackingBatchProof = PackingBatchProof<AkitaBatchProof>;
pub(crate) type AkitaPackingProverSetup =
    PackingProverSetup<AkitaProverSetup, PackingWitnessLayout>;

pub(crate) fn commit_packing_source<S>(
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError>
where
    S: PackingWitnessSource<AkitaField>,
{
    validate_packing_source_dimension(setup.pcs.max_num_vars(), source.layout())?;
    if let Some(unit_indices) = packed_source_unit_indices(source)? {
        return AkitaScheme::commit_unit_sparse_indices(
            &setup.pcs,
            source.layout().digest,
            source.layout().dimension,
            unit_indices,
        );
    }
    let polynomial = packing_witness_source_polynomial(source)?;
    AkitaScheme::commit_group(&setup.pcs, source.layout().digest, &[polynomial])
}

pub(crate) fn prove_packing_source_batch<T, OpeningId, RelationId, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    source: &S,
    hint: AkitaProverHint,
) -> Result<AkitaPackingBatchProof, OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    validate_packing_source_dimension(setup.pcs.max_num_vars(), source.layout())?;
    if let Some(unit_indices) = packed_source_unit_indices(source)? {
        if !has_packing_view(statement) {
            let native = AkitaScheme::prove_unit_sparse_batch(
                &setup.pcs,
                transcript,
                statement,
                source.layout().dimension,
                unit_indices,
                hint,
            )?;
            return Ok(PackingBatchProof::direct(native));
        }

        let shape = validate_packed_source_prover_inputs(setup, statement, source, &hint)?;
        let source = AkitaPackingSource(source);
        let reduction =
            prove_sparse_packing_reduction(shape.layout, statement, &source, transcript)?;
        let native_statement = singleton_statement(
            shape.commitment.clone(),
            &reduction.opening_point,
            reduction.opening_eval,
        );
        let native = AkitaScheme::prove_unit_sparse_batch(
            &setup.pcs,
            transcript,
            &native_statement,
            source.layout().dimension,
            unit_indices,
            hint,
        )?;
        return Ok(PackingBatchProof::packed(reduction.proof, native));
    }

    let polynomial = packing_witness_source_polynomial(source)?;
    <AkitaPackingScheme as BatchOpeningScheme>::prove_batch(
        setup,
        transcript,
        statement,
        std::slice::from_ref(&polynomial),
        vec![hint],
    )
}

struct AkitaPackingSource<'a, S>(&'a S);

impl<S> PackingSource<AkitaField> for AkitaPackingSource<'_, S>
where
    S: PackingWitnessSource<AkitaField>,
{
    type Layout = PackingWitnessLayout;

    fn layout(&self) -> &Self::Layout {
        self.0.layout()
    }

    fn for_each_nonzero(&self, f: impl FnMut(usize, AkitaField)) {
        self.0.for_each_nonzero(f);
    }
}

struct PackingBatchShape<'a> {
    layout: &'a PackingWitnessLayout,
    commitment: AkitaCommitment,
}

fn validate_packed_source_prover_inputs<'a, OpeningId, RelationId, S>(
    setup: &'a AkitaPackingProverSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    source: &S,
    hint: &AkitaProverHint,
) -> Result<PackingBatchShape<'a>, OpeningsError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let shape = validate_packed_adapter_statement(&setup.pcs, &setup.layout, statement)?;
    validate_packing_source_layout(shape.layout, source)?;
    if !hint.matches_commitment(&shape.commitment) {
        return Err(invalid_batch(
            "Akita packing proof requires one hint matching the packed witness commitment",
        ));
    }
    Ok(shape)
}

fn validate_packed_adapter_statement<'a, OpeningId, RelationId>(
    setup: &AkitaProverSetup,
    layout: &'a PackingWitnessLayout,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<PackingBatchShape<'a>, OpeningsError> {
    let commitment = validate_packing_statement(layout, statement)?;
    validate_packed_setup_shape(
        setup.max_num_vars(),
        setup.default_layout_digest(),
        layout,
        &commitment,
    )?;
    Ok(PackingBatchShape { layout, commitment })
}

fn validate_packed_setup_shape(
    max_num_vars: usize,
    default_layout_digest: [u8; 32],
    layout: &PackingWitnessLayout,
    commitment: &AkitaCommitment,
) -> Result<(), OpeningsError> {
    if commitment.num_vars() > max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: commitment.num_vars(),
            setup_max: max_num_vars,
        });
    }
    if commitment.num_vars() != max_num_vars {
        return Err(invalid_batch(format!(
            "Akita packing commitment dimension {} does not match exact setup dimension {}",
            commitment.num_vars(),
            max_num_vars
        )));
    }
    if commitment.layout_digest() != default_layout_digest {
        return Err(invalid_batch(
            "Akita packing commitment layout digest does not match setup",
        ));
    }
    if commitment.layout_digest() != layout.digest {
        return Err(invalid_batch(
            "Akita packing commitment layout digest does not match setup layout",
        ));
    }
    if commitment.num_vars() != layout.dimension {
        return Err(invalid_batch(format!(
            "Akita packing commitment dimension {} does not match layout dimension {}",
            commitment.num_vars(),
            layout.dimension
        )));
    }
    if commitment.poly_count() != 1 {
        return Err(invalid_batch(format!(
            "Akita packing witness commitment must contain one polynomial, got {}",
            commitment.poly_count()
        )));
    }
    Ok(())
}

fn packed_source_unit_indices<S>(source: &S) -> Result<Option<Vec<usize>>, OpeningsError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let layout = source.layout();
    if layout.cells == 0 {
        return Err(invalid_batch(
            "Akita packing witness layout must contain at least one cell",
        ));
    }
    if layout.dimension >= usize::BITS as usize {
        return Err(invalid_batch(format!(
            "Akita packing witness dimension {} exceeds usize bit width",
            layout.dimension
        )));
    }
    if !AkitaScheme::supports_unit_sparse_dimension(layout.dimension) {
        return Ok(None);
    }
    let domain_size = 1usize << layout.dimension;
    if layout.cells > domain_size {
        return Err(invalid_batch(format!(
            "Akita packing witness has {} cells but dimension {} supports {domain_size}",
            layout.cells, layout.dimension
        )));
    }

    let mut seen = BTreeSet::new();
    let mut ranks = Vec::new();
    let mut result = Ok(());
    source.for_each_nonzero(|rank, value| {
        if result.is_err() {
            return;
        }
        if rank >= layout.cells {
            result = Err(invalid_batch(format!(
                "Akita packing witness source emitted rank {rank} outside {} real cells",
                layout.cells
            )));
            return;
        }
        if !seen.insert(rank) {
            result = Err(invalid_batch(format!(
                "Akita packing witness source emitted rank {rank} more than once"
            )));
            return;
        }
        if value != AkitaField::one() {
            result = Err(invalid_batch(format!(
                "Akita sparse packed witness source emitted non-unit value at rank {rank}"
            )));
            return;
        }
        ranks.push(rank);
    });
    result?;

    Ok(Some(ranks))
}

fn singleton_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: commitment.layout_digest(),
        claims: vec![jolt_openings::BatchOpeningClaim {
            id: (),
            relation: (),
            commitment,
            claim: eval,
            view: jolt_openings::PhysicalView::Direct,
            scale: AkitaField::one(),
        }],
    }
}

fn invalid_batch(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}
