use crate::host::InlineOp;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub use rand;
pub use tracer::emulator::cpu::Xlen;
pub use tracer::instruction::inline::INLINE;
pub use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

/// Formal specification of an inline's behavior.
/// Connects the mathematical reference implementation to the test harness.
/// Implemented on the same struct as `InlineOp`.
pub trait InlineSpec: InlineOp {
    type Input;
    type Output: PartialEq + core::fmt::Debug;

    /// Pure reference implementation — the formal verification target.
    fn reference(input: &Self::Input) -> Self::Output;

    /// Generate a random input for property testing.
    fn random_input(rng: &mut StdRng) -> Self::Input;

    fn create_harness() -> InlineTestHarness;

    fn instruction() -> INLINE;

    /// Load typed input into harness memory and set up registers.
    fn load(harness: &mut InlineTestHarness, input: &Self::Input);

    /// Read typed output from harness memory after execution.
    fn read(harness: &mut InlineTestHarness) -> Self::Output;
}

/// Verify sequence builder matches reference for a given input.
pub fn verify<S: InlineSpec>(input: &S::Input) {
    let mut harness = S::create_harness();
    S::load(&mut harness, input);
    harness.execute_inline(S::instruction());
    let actual = S::read(&mut harness);
    let expected = S::reference(input);
    assert_eq!(actual, expected);
}

/// Verify sequence builder matches reference over `count` random inputs.
pub fn proptest<S: InlineSpec>(count: usize) {
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..count {
        verify::<S>(&S::random_input(&mut rng));
    }
}
