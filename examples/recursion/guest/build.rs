use std::{fs, path::Path};

fn main() {
    let embedded_bytes_path = Path::new("src/embedded_bytes.rs");

    if !embedded_bytes_path.exists() {
        let fallback_content = r#"pub static EMBEDDED_BYTES: &[u8] = &[];"#;
        fs::write(embedded_bytes_path, fallback_content).unwrap();
        println!("cargo:warning=Created fallback embedded_bytes.rs with empty bytes");
    } else {
        println!("cargo:rerun-if-changed=src/embedded_bytes.rs");
    }

    let provable_macro_path = Path::new("src/provable_macro.rs");

    if !provable_macro_path.exists() {
        let fallback_content = r#"macro_rules! provable_with_config {
    ($item: item) => {
        #[jolt::provable(
            max_input_size = 4096,
            max_output_size = 4096,
            memory_size = 33554432,
            stack_size = 4096,
            max_trace_length = 16777216
        )]
        $item
    };
}"#;
        fs::write(provable_macro_path, fallback_content).unwrap();
        println!("cargo:warning=Created fallback provable_macro.rs with default macro");
    } else {
        println!("cargo:rerun-if-changed=src/provable_macro.rs");
    }
}
