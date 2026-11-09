#![expect(clippy::unwrap_used, reason = "focused reference parity test")]
use super::*;

#[test]
fn native_input_packed_and_dense_rounds_match_reference() {
    let mut gpu = MetalInstructionInput::new().unwrap();
    let config = Arc::get_mut(&mut gpu.0).unwrap();
    config.min_prefix = 1;
    config.cpu_tail = 2;
    config.tile_rows = 32;
    // Multiple equality-aligned source tiles, odd prefix and nonzero tail; all four packed widths, two dense device
    // binds, then a small CPU tail. Retained rows include MIN/MAX immediates
    // and independent flags. Round2 needs q(1), while non-Boolean binds keep
    // the accumulated equality scalar nonzero.
    super::super::super::tests::assert_parity_field::<F>(
        7,
        717,
        67,
        false,
        0xDADA_CAFE,
        Some(Arc::new(gpu)),
        Some(2),
    );
}
