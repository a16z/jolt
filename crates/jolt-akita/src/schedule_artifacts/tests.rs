use super::*;

#[test]
fn collection_normalizes_order_and_preserves_transport_identity() {
    let entry = |k, chunks, levels| AkitaOneHotScheduleArtifact {
        k,
        profile: AkitaChunkProfile::new(chunks, levels),
        bytes: vec![1],
    };
    let entries = vec![entry(256, 8, 2), entry(16, 4, 2), entry(16, 2, 3)];
    let make = |entries| {
        AkitaScheduleArtifacts::new(AkitaChunkProfile::default(), vec![], vec![], entries).unwrap()
    };
    let bundle = make(entries.clone());
    let reversed = make(entries.into_iter().rev().collect());
    assert_eq!(bundle, reversed);
    assert_eq!(
        bundle
            .one_hot_artifacts()
            .iter()
            .map(|entry| (entry.k, entry.profile.to_string()))
            .collect::<Vec<_>>(),
        [
            (16, "w2r3".to_owned()),
            (16, "w4r2".to_owned()),
            (256, "w8r2".to_owned())
        ]
    );
    let bytes = bincode::serde::encode_to_vec(&bundle, bincode::config::standard()).unwrap();
    assert_eq!(
        bytes,
        bincode::serde::encode_to_vec(&reversed, bincode::config::standard()).unwrap()
    );
    let (decoded, used): (AkitaScheduleArtifacts, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(used, bytes.len());
}

#[test]
fn construction_and_deserialization_reject_duplicate_or_invalid_keys() {
    let entry = AkitaOneHotScheduleArtifact {
        k: 16,
        profile: AkitaChunkProfile::default(),
        bytes: vec![],
    };
    for entries in [
        vec![entry.clone(), entry.clone()],
        vec![AkitaOneHotScheduleArtifact {
            k: 64,
            ..entry.clone()
        }],
        vec![AkitaOneHotScheduleArtifact {
            profile: AkitaChunkProfile::new(3, 2),
            ..entry.clone()
        }],
    ] {
        assert!(AkitaScheduleArtifacts::new(
            AkitaChunkProfile::default(),
            vec![],
            vec![],
            entries.clone()
        )
        .is_err());
        let input = serde_json::json!({
            "dense_profile": AkitaChunkProfile::default(),
            "dense": [], "full_dense": [], "one_hot": entries,
        });
        assert!(serde_json::from_value::<AkitaScheduleArtifacts>(input).is_err());
        let bytes = bincode::serde::encode_to_vec(
            (
                AkitaChunkProfile::default(),
                Vec::<u8>::new(),
                Vec::<u8>::new(),
                entries,
            ),
            bincode::config::standard(),
        )
        .unwrap();
        assert!(
            bincode::serde::decode_from_slice::<AkitaScheduleArtifacts, _>(
                &bytes,
                bincode::config::standard()
            )
            .is_err()
        );
    }
    assert!(
        AkitaScheduleArtifacts::new(AkitaChunkProfile::new(2, 0), vec![], vec![], vec![]).is_err()
    );
}
