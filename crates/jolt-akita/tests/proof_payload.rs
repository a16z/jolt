#![expect(clippy::expect_used, reason = "tests assert serialization shape")]

use jolt_akita::{AkitaBatchProof, AkitaCommitment};

fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

#[test]
fn akita_proof_payloads_reject_unknown_serialized_fields() {
    let proof = AkitaBatchProof::serialized(
        AkitaCommitment {
            layout_digest: layout(7),
            num_vars: 4,
            poly_count: 1,
            native: vec![1, 2, 3],
        },
        vec![4],
        vec![5],
        vec![6],
    );

    let mut root = serde_json::to_value(&proof).expect("proof should serialize");
    let _ = root
        .as_object_mut()
        .expect("proof should serialize as an object")
        .insert("extra_payload".to_string(), serde_json::Value::Bool(true));
    assert!(
        serde_json::from_value::<AkitaBatchProof>(root).is_err(),
        "Akita native proof must reject unknown root fields"
    );

    let mut commitment = serde_json::to_value(&proof).expect("proof should serialize");
    let _ = commitment
        .as_object_mut()
        .expect("proof should serialize as an object")
        .get_mut("commitment")
        .expect("commitment should be present")
        .as_object_mut()
        .expect("commitment should serialize as an object")
        .insert(
            "extra_commitment".to_string(),
            serde_json::Value::Bool(true),
        );
    assert!(
        serde_json::from_value::<AkitaBatchProof>(commitment).is_err(),
        "Akita commitment payload must reject unknown nested fields"
    );
}
