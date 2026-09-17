mod external_example {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/clear_custom_transcript.rs"
    ));

    #[test]
    fn clear_sumcheck_roundtrip() {
        main().unwrap();
    }
}
