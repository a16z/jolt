use jolt_akita::{AkitaField, AkitaScheme, AkitaSetupParams};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningStatement, CommitmentScheme, OpeningsError, ZkBatchOpeningScheme,
};
use jolt_transcript::{Blake2bTranscript, Transcript};

fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

#[test]
fn akita_batch_zk_is_explicitly_unsupported() {
    let (prover_setup, _) = AkitaScheme::setup(AkitaSetupParams::new(4, 2, layout(7)));
    let statement = BatchOpeningStatement {
        logical_point: vec![f(1)],
        pcs_point: vec![f(1)],
        layout_digest: layout(7),
        claims: Vec::<BatchOpeningClaim<_, _, (), (), ()>>::new(),
    };
    let mut transcript = Blake2bTranscript::new(b"akita-zk");
    let result = <AkitaScheme as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        &mut transcript,
        &statement,
        &[],
        &[],
        vec![],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
