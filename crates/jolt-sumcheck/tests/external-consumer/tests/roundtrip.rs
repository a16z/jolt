mod external_example {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/clear_sumcheck.rs"
    ));

    #[test]
    fn clear_sumcheck_roundtrip() {
        main().unwrap();
    }
}
