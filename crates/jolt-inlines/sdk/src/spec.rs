pub trait InlineReference {
    type Input;
    type Output: PartialEq + core::fmt::Debug;

    fn reference(input: &Self::Input) -> Self::Output;
}

#[cfg(feature = "test-utils")]
pub trait InlineSpec: InlineReference + crate::host::InlineOp {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input>;
    fn random(rng: &mut impl rand::RngCore) -> Self::Input;
    fn harness() -> jolt_tracer::utils::inline_test_harness::InlineTestHarness;
    fn load(
        harness: &mut jolt_tracer::utils::inline_test_harness::InlineTestHarness,
        input: &Self::Input,
    );
    fn read(harness: &mut jolt_tracer::utils::inline_test_harness::InlineTestHarness) -> Self::Output;

    fn instruction() -> jolt_tracer::instruction::inline::INLINE {
        jolt_tracer::utils::inline_test_harness::InlineTestHarness::create_default_instruction(
            Self::OPCODE,
            Self::FUNCT3,
            Self::FUNCT7,
        )
    }
}

#[cfg(feature = "test-utils")]
pub fn assert_reference_matches_harness<S: InlineSpec>(input: &S::Input) {
    let expected = S::reference(input);
    let mut harness = S::harness();
    harness.setup_registers();
    S::load(&mut harness, input);
    harness.execute_inline(S::instruction());
    let actual = S::read(&mut harness);

    assert_eq!(actual, expected, "{} inline output mismatch", S::NAME);
}

#[cfg(feature = "test-utils")]
pub fn assert_edge_cases_match_reference<S: InlineSpec>() {
    for input in S::edge_cases() {
        assert_reference_matches_harness::<S>(&input);
    }
}

#[cfg(feature = "test-utils")]
pub fn assert_random_cases_match_reference<S: InlineSpec>(seed: u64, cases: usize) {
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for _ in 0..cases {
        let input = S::random(&mut rng);
        assert_reference_matches_harness::<S>(&input);
    }
}
