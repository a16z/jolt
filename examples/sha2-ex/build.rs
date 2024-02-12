fn main() {
    println!("cargo:rustc-link-arg-bin=sha2-ex=-Texamples/sha2-ex/linker.ld");
}
