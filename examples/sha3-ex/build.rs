fn main() {
    println!("cargo:rustc-link-arg-bin=sha3-ex=-Texamples/sha3-ex/linker.ld");
}
