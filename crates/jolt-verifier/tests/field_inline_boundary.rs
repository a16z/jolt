//! Structural guards for the verifier composition boundary. Protocol-family
//! import disjointness is checked in jolt-claims; composed expression coverage
//! is checked by stages::composed's typed contract test.

#![expect(clippy::expect_used, reason = "test-only source-tree inspection")]

use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("source directory is readable") {
            let path = entry.expect("directory entry is readable").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn composed_relations_use_symbolic_claim_evaluation() {
    use syn::{ImplItem, Item};
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stages");
    for file in [
        "stage1/outer_remainder.rs",
        "stage2/product_uniskip.rs",
        "stage2/product_remainder.rs",
        "stage6a/bytecode_read_raf.rs",
    ] {
        let parsed = syn::parse_file(&fs::read_to_string(src.join(file)).expect("read relation"))
            .expect("valid Rust source");
        for item in parsed.items {
            if let Item::Impl(item) = item {
                if !item.trait_.as_ref().is_some_and(|(path, _)| {
                    path.segments
                        .last()
                        .is_some_and(|segment| segment.ident == "ConcreteSumcheck")
                }) {
                    continue;
                }
                for method in item.items {
                    if let ImplItem::Fn(method) = method {
                        assert!(
                            method.sig.ident != "input_claim"
                                && method.sig.ident != "expected_output",
                            "{file} bypasses its symbolic claim contract with {}",
                            method.sig.ident
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn runtime_sources_do_not_import_prover_implementation_crates() {
    use syn::visit::{self, Visit};
    use syn::{ItemMod, Path as SyntaxPath, UsePath};

    #[derive(Default)]
    struct Imports {
        forbidden: Vec<String>,
    }
    impl Imports {
        fn check(&mut self, name: String) {
            if [
                "jolt_prover",
                "jolt_prover_legacy",
                "jolt_kernels",
                "jolt_witness",
                "tracer",
            ]
            .contains(&name.as_str())
            {
                self.forbidden.push(name);
            }
        }
    }
    impl<'ast> Visit<'ast> for Imports {
        fn visit_item_mod(&mut self, module: &'ast ItemMod) {
            if module.attrs.iter().any(|attr| {
                attr.path().is_ident("cfg")
                    && attr
                        .parse_args::<SyntaxPath>()
                        .is_ok_and(|path| path.is_ident("test"))
            }) {
                return;
            }
            visit::visit_item_mod(self, module);
        }
        fn visit_use_path(&mut self, path: &'ast UsePath) {
            self.check(path.ident.to_string());
            visit::visit_use_path(self, path);
        }
        fn visit_path(&mut self, path: &'ast SyntaxPath) {
            if let Some(first) = path.segments.first() {
                self.check(first.ident.to_string());
            }
            visit::visit_path(self, path);
        }
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for file in rust_sources(&src) {
        let source = fs::read_to_string(&file).expect("read source");
        let parsed = syn::parse_file(&source).expect("valid Rust source");
        let mut imports = Imports::default();
        imports.visit_file(&parsed);
        assert!(
            imports.forbidden.is_empty(),
            "{} imports prover implementation crates: {:?}",
            file.display(),
            imports.forbidden
        );
    }
}
