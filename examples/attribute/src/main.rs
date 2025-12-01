pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";

    // Test 1: allow(arithmetic_overflow)
    let mut program = guest::compile_test_allow_overflow(target_dir);
    let preprocessing = guest::preprocess_prover_test_allow_overflow(&mut program);
    let prove = guest::build_prover_test_allow_overflow(program, preprocessing);
    let (output, _, _) = prove();
    assert_eq!(output, 0u8); // 255 + 1 = 0 (wrapped)
    println!("✓ test_allow_overflow passed");

    // Test 2: allow(unused_variables)
    let mut program = guest::compile_test_allow_unused(target_dir);
    let preprocessing = guest::preprocess_prover_test_allow_unused(&mut program);
    let prove = guest::build_prover_test_allow_unused(program, preprocessing);
    let (output, _, _) = prove();
    assert_eq!(output, 100);
    println!("✓ test_allow_unused passed");

    // Test 3: inline
    let mut program = guest::compile_test_inline(target_dir);
    let preprocessing = guest::preprocess_prover_test_inline(&mut program);
    let prove = guest::build_prover_test_inline(program, preprocessing);
    let (output, _, _) = prove();
    assert_eq!(output, 10);
    println!("✓ test_inline passed");

    // Test 4: multiple attributes
    let mut program = guest::compile_test_multiple_attrs(target_dir);
    let preprocessing = guest::preprocess_prover_test_multiple_attrs(&mut program);
    let prove = guest::build_prover_test_multiple_attrs(program, preprocessing);
    let (output, _, _) = prove();
    assert_eq!(output, 254u8); // 255 + 255 = 254 (wrapped)
    println!("✓ test_multiple_attrs passed");

    println!("\nAll attribute propagation tests passed! ✓");
}
