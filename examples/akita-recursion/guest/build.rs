use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=JOLT_AKITA_RECURSION_MATERIAL");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("build output directory"));
    let code = if let Some(directory) = env::var_os("JOLT_AKITA_RECURSION_MATERIAL") {
        let directory = PathBuf::from(directory);
        let mut code =
            String::from("#[repr(align(16))] struct Aligned<const N: usize>([u8; N]);\n");
        for (name, file) in [("KEY", "key.bin"), ("TERMINAL", "terminal.bin")] {
            let path = directory
                .join(file)
                .canonicalize()
                .expect("prepared material");
            println!("cargo:rerun-if-changed={}", path.display());
            let len = fs::metadata(&path)
                .expect("prepared material metadata")
                .len();
            code.push_str(&format!(
                "static {name}: Aligned<{len}> = Aligned(*include_bytes!({path:?}));\n"
            ));
        }
        code
    } else {
        "#[repr(align(16))] struct Aligned<const N: usize>([u8; N]);\nstatic KEY: Aligned<0> = Aligned([]);\nstatic TERMINAL: Aligned<0> = Aligned([]);\n".to_owned()
    };
    fs::write(output.join("material.rs"), code).expect("write embedded material");
}
