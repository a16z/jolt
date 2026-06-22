#![expect(clippy::expect_used, reason = "tests assert serialization shape")]

use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaField, AkitaPackedBatchProof, AkitaPackedReductionProof,
};
use jolt_field::FixedByteSize;

fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

#[test]
fn akita_proof_payloads_reject_unknown_serialized_fields() {
    let proof = AkitaPackedBatchProof {
        reduction: Some(AkitaPackedReductionProof {
            rounds: Vec::new(),
            opening_eval: vec![0; AkitaField::NUM_BYTES],
        }),
        native: AkitaBatchProof {
            commitment: AkitaCommitment {
                layout_digest: layout(7),
                num_vars: 4,
                poly_count: 1,
                native: vec![1, 2, 3],
            },
            statement_bridge: vec![4],
            proof_shape: vec![5],
            proof: vec![6],
        },
    };

    let mut root = serde_json::to_value(&proof).expect("proof should serialize");
    let _ = root
        .as_object_mut()
        .expect("proof should serialize as an object")
        .insert("extra_payload".to_string(), serde_json::Value::Bool(true));
    assert!(
        serde_json::from_value::<AkitaPackedBatchProof>(root).is_err(),
        "Akita packed proof must reject unknown root fields"
    );

    let mut native = serde_json::to_value(&proof).expect("proof should serialize");
    let _ = native
        .as_object_mut()
        .expect("proof should serialize as an object")
        .get_mut("native")
        .expect("native proof should be present")
        .as_object_mut()
        .expect("native proof should serialize as an object")
        .insert("extra_native".to_string(), serde_json::Value::Bool(true));
    assert!(
        serde_json::from_value::<AkitaPackedBatchProof>(native).is_err(),
        "Akita native proof must reject unknown nested fields"
    );

    let mut commitment = serde_json::to_value(&proof).expect("proof should serialize");
    let _ = commitment
        .as_object_mut()
        .expect("proof should serialize as an object")
        .get_mut("native")
        .expect("native proof should be present")
        .as_object_mut()
        .expect("native proof should serialize as an object")
        .get_mut("commitment")
        .expect("commitment should be present")
        .as_object_mut()
        .expect("commitment should serialize as an object")
        .insert(
            "extra_commitment".to_string(),
            serde_json::Value::Bool(true),
        );
    assert!(
        serde_json::from_value::<AkitaPackedBatchProof>(commitment).is_err(),
        "Akita commitment payload must reject unknown nested fields"
    );
}
