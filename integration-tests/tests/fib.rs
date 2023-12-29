use compiler::compile_example;

#[test]
fn fib_e2e() {
    // Run the compiler
    compile_example("fibonacci");

    // Run the integration test
    unimplemented!("we built, but did not test");
}