use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheme, AkitaSetupParams};
use jolt_openings::{BatchOpeningClaim, BatchOpeningStatement, CommitmentScheme, PhysicalView};
use jolt_poly::Polynomial;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationId {
    NativeBatch,
}

pub fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

pub fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

pub fn polynomial(offset: u64) -> Polynomial<AkitaField> {
    Polynomial::new((0..16).map(|value| f(value + offset)).collect())
}

pub fn setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::new(4, 2, layout(7)))
}

pub fn direct_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: commitment.layout_digest,
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::NativeBatch,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::Direct,
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::NativeBatch,
                commitment,
                claim: eval_b,
                view: PhysicalView::Direct,
                scale: f(5),
            },
        ],
    }
}

pub fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("test thread should spawn")
        .join()
        .expect("test thread should complete");
}
