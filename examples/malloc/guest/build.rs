fn main() {
    // Tell cargo to invalidate the built crate whenever the C source changes
    println!("cargo:rerun-if-changed=src/malloc.c");

    // Build the C source file
    cc::Build::new().file("src/malloc.c").compile("malloc");
}
