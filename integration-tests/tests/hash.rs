use compiler::compile_example;

#[test]
fn hash_e2e() {
    // Run the compiler
    compile_example("hash");

    // Run the integration test
    unimplemented!("we built, but did not test");
}