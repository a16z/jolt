fn main() {
    println!("cargo:rustc-link-arg-bin=fibonacci=-Texamples/fibonacci/linker.ld");
}
