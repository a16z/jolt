use std::path::PathBuf;
use build_fs_tree::{dir, file, serde_yaml};

mod util;
use util::{read_fs_tree_recursively, FSTree, FSResult};
pub use util::FSError;

const DEFAULT_TEMPLATE_YAML: &'static str = include_str!(env!("TEMPLATE_YAML_PATH"));

/// A module to write to the ZkLean Jolt package
pub struct Module {
    /// The name of the module. The filename will become `src/Jolt/{name}.lean`.
    pub name: String,
    /// A list of modules to import
    pub imports: Vec<String>,
    /// The contents of the module formatted as Lean4 code
    pub contents: Vec<u8>,
}

/// Trait for objects that can be converted to `Module`.
// NOTE: We cannot simply use `Into<Module>` here, because `Into` does not support dynamic
// dispatch.
pub trait AsModule {
    fn as_module(&self) -> std::io::Result<Module>;
}

/// Write each module, along with its imports, to `src/Jolt/{name}.lean` in a template directory.
/// If a `template_dir` is provided, read the template directory from that path. Otherwise, use the
/// default template read at compile time.
///
/// NB: Any files in the template that collide with `src/Jolt/{name}.lean` for any module will be
/// clobbered.
pub fn make_jolt_zk_lean_package(
    template_dir: &Option<PathBuf>,
    modules: Vec<Box<dyn AsModule>>,
) -> FSResult<FSTree> {
    let mut builder: util::FSTree = template_dir
        .as_ref()
        .map_or(
            serde_yaml::from_str(DEFAULT_TEMPLATE_YAML).map_err(FSError::from),
            |dir| read_fs_tree_recursively(dir)
        )?;

    let src_jolt_dir = builder
        .dir_content_mut().ok_or(FSError::TemplateError(format!("{template_dir:?} is not a directory")))?
        .entry(String::from("src")).or_insert(dir! {})
        .dir_content_mut().ok_or(FSError::TemplateError(format!("{template_dir:?}/src is not a directory")))?
        .entry(String::from("Jolt")).or_insert(dir! {})
        .dir_content_mut().ok_or(FSError::TemplateError(format!("{template_dir:?}/src/Jolt is not a directory")))?;

    for module in modules {
        let module = module.as_module()?;
        let contents_with_imports: Vec<u8> = module.imports
            .into_iter()
            .map(|i| format!("import {i}\n").bytes().collect::<Vec<u8>>())
            .flatten()
            .chain(vec![b'\n'])
            .chain(module.contents)
            .collect();
        let _ = src_jolt_dir.insert(format!("{}.lean", module.name), file!(contents_with_imports));
    }

    Ok(builder)
}
