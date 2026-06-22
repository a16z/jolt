pub(crate) use jolt_akita::{
    AkitaCommitInput, AkitaCommitment, AkitaField, AkitaPackedScheme, AkitaScheme,
    AkitaSetupParams, AKITA_FIELD_MODULUS,
};
pub(crate) use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PackedAlphabet, PackedCellAddress, PackedCombine, PackedFactDomain, PackedFamilyId,
    PackedFamilySpec, PackedLayoutError, PackedLinearTerm, PackedWitnessLayout,
    PackedWitnessSource, PhysicalView, SparsePackedWitness, ZkOpeningScheme,
};
pub(crate) use jolt_poly::{EqPolynomial, Polynomial};
pub(crate) use jolt_transcript::{Blake2bTranscript, Transcript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelationId {
    Packed,
}

pub(crate) type PackedAkita = PackedCombine<AkitaScheme>;

pub(crate) fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

pub(crate) fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

pub(crate) fn polynomial(offset: u64) -> Polynomial<AkitaField> {
    Polynomial::new((0..16).map(|value| f(value + offset)).collect())
}

pub(crate) fn setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::new(4, 2, layout(7)))
}

pub(crate) fn packed_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        },
        PackedFactDomain::TraceRows { log_t: 1 },
        1,
        PackedAlphabet::Fixed { size: 3 },
    )])
    .expect("packed layout should be valid")
}

pub(crate) fn ring_sized_packed_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        },
        PackedFactDomain::TraceRows { log_t: 4 },
        1,
        PackedAlphabet::Fixed { size: 4 },
    )])
    .expect("ring-sized packed layout should be valid")
}

pub(crate) fn packed_reduction_family() -> PackedFamilyId {
    PackedFamilyId::Custom {
        namespace: 2,
        index: 0,
    }
}

pub(crate) fn packed_reduction_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        packed_reduction_family(),
        PackedFactDomain::TraceRows { log_t: 2 },
        2,
        PackedAlphabet::Bit,
    )])
    .expect("packed reduction layout should be valid")
}

pub(crate) fn packed_address(row: usize, symbol: usize) -> PackedCellAddress {
    PackedCellAddress {
        family: PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        },
        row,
        limb: 0,
        symbol,
    }
}

pub(crate) fn ring_sized_packed_address(row: usize, symbol: usize) -> PackedCellAddress {
    PackedCellAddress {
        family: PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        },
        row,
        limb: 0,
        symbol,
    }
}

pub(crate) fn packed_reduction_address(
    row: usize,
    limb: usize,
    symbol: usize,
) -> PackedCellAddress {
    PackedCellAddress {
        family: packed_reduction_family(),
        row,
        limb,
        symbol,
    }
}

pub(crate) fn packed_term(coefficient: AkitaField) -> PackedLinearTerm<AkitaField> {
    packed_term_at(coefficient, 0)
}

pub(crate) fn packed_term_at(
    coefficient: AkitaField,
    symbol: usize,
) -> PackedLinearTerm<AkitaField> {
    PackedLinearTerm::new(
        coefficient,
        (PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        })
        .physical_ref(),
        0,
        symbol,
    )
}

pub(crate) fn packed_reduction_term(
    coefficient: AkitaField,
    limb: usize,
    symbol: usize,
    row_point: &[AkitaField],
) -> PackedLinearTerm<AkitaField> {
    PackedLinearTerm::new(
        coefficient,
        packed_reduction_family().physical_ref(),
        limb,
        symbol,
    )
    .with_row_point(row_point.to_vec())
}

pub(crate) fn ring_sized_packed_term(
    coefficient: AkitaField,
    symbol: usize,
    row_point: &[AkitaField],
) -> PackedLinearTerm<AkitaField> {
    PackedLinearTerm::new(
        coefficient,
        (PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        })
        .physical_ref(),
        0,
        symbol,
    )
    .with_row_point(row_point.to_vec())
}

pub(crate) fn packed_polynomial(
    layout: &PackedWitnessLayout,
    entries: &[(usize, AkitaField)],
) -> Polynomial<AkitaField> {
    let mut evals = vec![AkitaField::zero(); 1usize << layout.dimension];
    for &(rank, value) in entries {
        evals[rank] = value;
    }
    Polynomial::new(evals)
}

pub(crate) fn packed_view_eval(
    layout: &PackedWitnessLayout,
    witness: &SparsePackedWitness<AkitaField>,
    terms: &[PackedLinearTerm<AkitaField>],
) -> AkitaField {
    terms.iter().fold(AkitaField::zero(), |acc, term| {
        let family = layout
            .families
            .iter()
            .find(|family| family.id.physical_ref() == term.family)
            .expect("term family should exist");
        let row_weights = EqPolynomial::new(term.row_point.clone()).evaluations();
        let contribution = row_weights.iter().copied().enumerate().fold(
            AkitaField::zero(),
            |acc, (row, row_weight)| {
                let value = witness
                    .eval_direct_fact(&PackedCellAddress {
                        family: family.id.clone(),
                        row,
                        limb: term.limb,
                        symbol: term.symbol,
                    })
                    .expect("packed address should be valid");
                acc + row_weight * value
            },
        );
        acc + term.coefficient * contribution
    })
}

pub(crate) struct EmittingPackedSource {
    pub(crate) layout: PackedWitnessLayout,
    pub(crate) entries: Vec<(usize, AkitaField)>,
}

impl PackedWitnessSource<AkitaField> for EmittingPackedSource {
    fn layout(&self) -> &PackedWitnessLayout {
        &self.layout
    }

    fn for_each_nonzero(&self, mut f: impl FnMut(usize, AkitaField)) {
        for &(rank, value) in &self.entries {
            f(rank, value);
        }
    }

    fn eval_direct_fact(
        &self,
        address: &PackedCellAddress,
    ) -> Result<AkitaField, PackedLayoutError> {
        let rank = self.layout.rank(address)?;
        Ok(self
            .entries
            .iter()
            .find(|(entry_rank, _)| *entry_rank == rank)
            .map_or_else(AkitaField::zero, |(_, value)| *value))
    }
}

pub(crate) fn direct_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: layout(7),
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::Direct,
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval_b,
                view: PhysicalView::Direct,
                scale: f(5),
            },
        ],
    }
}

pub(crate) fn unit_packed_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: layout(7),
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout(7),
                    terms: vec![packed_term(f(1))],
                },
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval_b,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout(7),
                    terms: vec![packed_term(f(1))],
                },
                scale: f(5),
            },
        ],
    }
}

pub(crate) fn packed_reduction_statement(
    layout: &PackedWitnessLayout,
    commitment: AkitaCommitment,
    row_point: &[AkitaField],
    terms_a: Vec<PackedLinearTerm<AkitaField>>,
    claim_a: AkitaField,
    terms_b: Vec<PackedLinearTerm<AkitaField>>,
    claim_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: row_point.to_vec(),
        pcs_point: row_point.to_vec(),
        layout_digest: layout.digest,
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: claim_a,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: terms_a,
                },
                scale: f(3),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: claim_b,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: terms_b,
                },
                scale: f(7),
            },
        ],
    }
}

pub(crate) fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(test)
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked");
}
