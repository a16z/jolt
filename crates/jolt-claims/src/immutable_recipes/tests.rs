#![expect(clippy::unwrap_used, reason = "literal recipe fixtures must validate")]

#[test]
fn full_shape_key_distinguishes_order_payloads_and_affine_flag() {
    use super::{KeySink, KeyVisit};
    use crate::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
    use crate::protocols::jolt::JoltVirtualPolynomial;
    use jolt_riscv::CircuitFlags;
    use std::collections::BTreeSet;

    struct Words(Vec<u64>);
    impl KeySink for Words {
        fn word(&mut self, value: u64) -> bool {
            self.0.push(value);
            true
        }
    }
    let address = JoltVirtualPolynomial::InstructionRa(7);
    let load = JoltVirtualPolynomial::OpFlags(CircuitFlags::Load);
    let shapes = [
        (12, vec![address, load], false),
        (13, vec![address, load], false),
        (12, vec![address], false),
        (12, vec![load, address], false),
        (
            12,
            vec![JoltVirtualPolynomial::InstructionRa(8), load],
            false,
        ),
        (
            12,
            vec![address, JoltVirtualPolynomial::OpFlags(CircuitFlags::Store)],
            false,
        ),
        (12, vec![address, load], true),
    ];
    let mut keys = BTreeSet::new();
    for (index, (log_t, variables, affine)) in shapes.into_iter().enumerate() {
        let shape = SpartanOuterDimensions::new(log_t, variables, affine).unwrap();
        let mut words = Words(Vec::new());
        assert!(shape.visit(&mut words));
        if index == 0 {
            assert_eq!(words.0, [12, 2, 25, 7, 36, 3, 0]);
        }
        assert!(keys.insert(words.0));
    }
    assert_eq!(keys.len(), 7);
}
