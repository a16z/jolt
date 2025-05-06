use std::env;
use std::fs;
use std::path::Path;

// Provides `read_fs_tree_recursively` helper function. Also imports `build_fs_tree::serde_yaml`.
include!("src/modules/util.rs");

const TEMPLATE_DIR: &'static str = "package-template";

fn main() {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set in environment");
    let template_dir = Path::new(&manifest_dir).join(TEMPLATE_DIR);
    if ! template_dir.is_dir() {
        panic!("template directory {template_dir:?} is not a directory");
    }
    let template = read_fs_tree_recursively(&template_dir.into()).unwrap();
    let serialized_template = serde_yaml::to_string(&template).unwrap();

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let serialized_template_path = Path::new(&out_dir).join(format!("{TEMPLATE_DIR}.yaml"));
    fs::write(&serialized_template_path, serialized_template).unwrap();
    println!("cargo::rustc-env=TEMPLATE_YAML_PATH={}", serialized_template_path.as_os_str().to_string_lossy());
}
