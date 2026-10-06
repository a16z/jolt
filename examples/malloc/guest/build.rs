fn main() {
    println!("cargo:rerun-if-changed=src/malloc.c");

    cc::Build::new().file("src/malloc.c").compile("malloc");
}
