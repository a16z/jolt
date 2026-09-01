#![cfg(feature = "fs-audit")]
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the source census must fail loudly when Cargo metadata or Rust syntax is malformed"
)]

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use quote::ToTokens;
use serde_json::Value;
use syn::{
    visit::{self, Visit},
    Attribute, ExprCall, ExprMethodCall, ImplItemFn, ItemEnum, ItemFn, ItemImpl, ItemMod,
    ItemStruct, ItemTrait, Macro, TraitItemFn,
};

const ABSORB_INVENTORY: &str = "tests/fs_inventory/absorb-sites.inventory";
const CHALLENGE_INVENTORY: &str = "tests/fs_inventory/challenge-sites.inventory";
const SCOPE_INVENTORY: &str = "tests/fs_inventory/scope-sites.inventory";
const SOURCE_INVENTORY: &str = "tests/fs_inventory/source-schema.inventory";

#[derive(Clone)]
struct PackageSource {
    name: String,
    src: PathBuf,
}

#[test]
fn verifier_fs_inventory_is_complete() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("jolt-verifier must be in <workspace>/crates");
    let packages = production_sources(workspace);
    let mut absorb_sites = BTreeSet::new();
    let mut challenge_sites = BTreeSet::new();
    let mut scope_sites = BTreeSet::new();
    let mut source_fields = BTreeSet::new();

    for package in packages {
        for path in rust_sources(&package.src) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let syntax = syn::parse_file(&source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
            let relative = path
                .strip_prefix(&package.src)
                .expect("source path left its package src directory");
            let file_id = format!(
                "{}::{}",
                package.name,
                relative.to_string_lossy().replace('\\', "/")
            );

            let mut visitor = InventoryVisitor::new(file_id);
            visitor.visit_file(&syntax);
            absorb_sites.extend(visitor.absorb_sites);
            challenge_sites.extend(visitor.challenge_sites);
            scope_sites.extend(visitor.scope_sites);
            source_fields.extend(visitor.source_fields);
        }
    }

    check_inventory(
        &manifest_dir.join(ABSORB_INVENTORY),
        "Fiat-Shamir absorption sites",
        &absorb_sites,
    );
    check_inventory(
        &manifest_dir.join(CHALLENGE_INVENTORY),
        "Fiat-Shamir challenge sites",
        &challenge_sites,
    );
    check_inventory(
        &manifest_dir.join(SCOPE_INVENTORY),
        "Fiat-Shamir scope annotations",
        &scope_sites,
    );
    check_inventory(
        &manifest_dir.join(SOURCE_INVENTORY),
        "verifier source schema",
        &source_fields,
    );
}

fn production_sources(workspace: &Path) -> Vec<PackageSource> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(workspace)
        .output()
        .expect("failed to execute cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata returned invalid JSON");
    let packages = metadata["packages"]
        .as_array()
        .expect("cargo metadata packages are missing");
    let nodes = metadata["resolve"]["nodes"]
        .as_array()
        .expect("cargo metadata resolve nodes are missing");

    let package_by_id = packages
        .iter()
        .map(|package| {
            (
                package["id"].as_str().expect("package id is missing"),
                package,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let node_by_id = nodes
        .iter()
        .map(|node| (node["id"].as_str().expect("node id is missing"), node))
        .collect::<BTreeMap<_, _>>();

    let mut pending = packages
        .iter()
        .filter(|package| {
            matches!(
                package["name"].as_str(),
                Some("jolt-verifier" | "jolt-dory" | "jolt-akita")
            )
        })
        .map(|package| package["id"].as_str().expect("seed package id is missing"))
        .collect::<Vec<_>>();
    let mut closure = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !closure.insert(id) {
            continue;
        }
        let Some(node) = node_by_id.get(id) else {
            continue;
        };
        for dependency in node["deps"]
            .as_array()
            .expect("node dependencies are missing")
        {
            let is_production = dependency["dep_kinds"]
                .as_array()
                .expect("dependency kinds are missing")
                .iter()
                .any(|kind| kind["kind"].is_null() || kind["kind"] == "build");
            if is_production {
                pending.push(
                    dependency["pkg"]
                        .as_str()
                        .expect("dependency package id is missing"),
                );
            }
        }
    }

    let mut sources = closure
        .into_iter()
        .filter_map(|id| package_by_id.get(id))
        .filter(|package| package["source"].is_null())
        .filter(|package| package["name"] != "jolt-prover-legacy")
        .filter_map(|package| {
            let manifest = PathBuf::from(
                package["manifest_path"]
                    .as_str()
                    .expect("package manifest path is missing"),
            );
            let src = manifest.parent()?.join("src");
            src.is_dir().then(|| PackageSource {
                name: package["name"]
                    .as_str()
                    .expect("package name is missing")
                    .to_owned(),
                src,
            })
        })
        .collect::<Vec<_>>();
    sources.sort_by(|left, right| left.name.cmp(&right.name));
    sources
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    fn collect(path: &Path, output: &mut Vec<PathBuf>) {
        let mut entries = fs::read_dir(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
            .map(|entry| entry.expect("failed to read source directory entry").path())
            .collect::<Vec<_>>();
        entries.sort();
        for entry in entries {
            if entry.is_dir() {
                collect(&entry, output);
            } else if entry.extension().is_some_and(|extension| extension == "rs")
                && entry
                    .file_stem()
                    .is_none_or(|stem| !matches!(stem.to_str(), Some("prove" | "prover" | "tests")))
            {
                output.push(entry);
            }
        }
    }

    let mut output = Vec::new();
    collect(root, &mut output);
    output
}

fn check_inventory(path: &Path, label: &str, actual: &BTreeSet<String>) {
    if env::var_os("JOLT_FS_BLESS").is_some() {
        let body = format!(
            "# Generated by `JOLT_FS_BLESS=1 cargo nextest run -p jolt-verifier \\\n#   --test fs_obligations --features fs-audit`.\n# Review changes; this inventories identities, not transcript values.\n{}\n",
            actual.iter().cloned().collect::<Vec<_>>().join("\n")
        );
        fs::create_dir_all(path.parent().expect("inventory path has no parent"))
            .expect("failed to create inventory directory");
        fs::write(path, body)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        return;
    }

    let expected = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let added = actual.difference(&expected).collect::<Vec<_>>();
    let removed = expected.difference(actual).collect::<Vec<_>>();
    assert!(
        added.is_empty() && removed.is_empty(),
        "{label} changed.\nUnclassified: {added:#?}\nNo longer present: {removed:#?}\n\
         Run the blessed command and review the inventory diff."
    );
}

struct InventoryVisitor {
    file_id: String,
    context: Vec<String>,
    /// One counter per context, shared by absorb and challenge records, so
    /// an identity encodes its position in the combined absorb/challenge
    /// sequence. Reordering an absorb against a squeeze — the canonical
    /// weak-FS bug — renumbers both sites and trips the inventory even when
    /// each per-kind subsequence is unchanged.
    call_ordinals: BTreeMap<String, usize>,
    absorb_sites: BTreeSet<String>,
    challenge_sites: BTreeSet<String>,
    scope_sites: BTreeSet<String>,
    source_fields: BTreeSet<String>,
}

impl InventoryVisitor {
    fn new(file_id: String) -> Self {
        Self {
            file_id,
            context: Vec::new(),
            call_ordinals: BTreeMap::new(),
            absorb_sites: BTreeSet::new(),
            challenge_sites: BTreeSet::new(),
            scope_sites: BTreeSet::new(),
            source_fields: BTreeSet::new(),
        }
    }

    fn with_context(&mut self, name: String, visit: impl FnOnce(&mut Self)) {
        self.context.push(name);
        visit(self);
        let _ = self.context.pop();
    }

    fn record_absorb(&mut self, kind: &str, expression: &impl ToTokens) {
        let context = self.context();
        let ordinal = self.call_ordinals.entry(context.clone()).or_default();
        let expression = expression.to_token_stream().to_string().replace(' ', "");
        let _ = self.absorb_sites.insert(format!(
            "{}::{}::{}#{}::{expression}",
            self.file_id, context, kind, *ordinal
        ));
        *ordinal += 1;
    }

    fn record_challenge(&mut self, kind: &str) {
        let context = self.context();
        let ordinal = self.call_ordinals.entry(context.clone()).or_default();
        let _ = self.challenge_sites.insert(format!(
            "{}::{}::{}#{}",
            self.file_id, context, kind, *ordinal
        ));
        *ordinal += 1;
    }

    fn context(&self) -> String {
        if self.context.is_empty() {
            "<module>".to_owned()
        } else {
            self.context.join("::")
        }
    }

    fn record_struct(&mut self, item: &ItemStruct) {
        if has_derive(&item.attrs, "SumcheckBatch") {
            let _ = self.challenge_sites.insert(format!(
                "{}::{}::<generated>::batching_coefficient[*]",
                self.file_id, item.ident
            ));
        }
        if !is_source_type(&item.ident.to_string()) {
            return;
        }
        for (index, field) in item.fields.iter().enumerate() {
            let field = field
                .ident
                .as_ref()
                .map_or_else(|| index.to_string(), ToString::to_string);
            let _ = self
                .source_fields
                .insert(format!("{}::{}.{}", self.file_id, item.ident, field));
        }
    }

    fn record_enum(&mut self, item: &ItemEnum) {
        if !is_source_type(&item.ident.to_string()) {
            return;
        }
        for variant in &item.variants {
            if variant.fields.is_empty() {
                let _ = self.source_fields.insert(format!(
                    "{}::{}::{}",
                    self.file_id, item.ident, variant.ident
                ));
            }
            for (index, field) in variant.fields.iter().enumerate() {
                let field = field
                    .ident
                    .as_ref()
                    .map_or_else(|| index.to_string(), ToString::to_string);
                let _ = self.source_fields.insert(format!(
                    "{}::{}::{}.{}",
                    self.file_id, item.ident, variant.ident, field
                ));
            }
        }
    }
}

impl<'ast> Visit<'ast> for InventoryVisitor {
    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        if cfg_test(&item.attrs) || prover_only_name(&item.sig.ident.to_string()) {
            return;
        }
        for attribute in &item.attrs {
            if attribute
                .path()
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "fs_scope")
            {
                let _ = self.scope_sites.insert(format!(
                    "{}::{}::{}",
                    self.file_id,
                    item.sig.ident,
                    attribute
                        .meta
                        .to_token_stream()
                        .to_string()
                        .replace(' ', "")
                ));
            }
        }
        self.with_context(item.sig.ident.to_string(), |visitor| {
            visit::visit_item_fn(visitor, item);
        });
    }

    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        if cfg_test(&item.attrs) {
            return;
        }
        let name = item.self_ty.to_token_stream().to_string().replace(' ', "");
        self.with_context(name, |visitor| visit::visit_item_impl(visitor, item));
    }

    fn visit_impl_item_fn(&mut self, item: &'ast ImplItemFn) {
        if cfg_test(&item.attrs) || prover_only_name(&item.sig.ident.to_string()) {
            return;
        }
        self.with_context(item.sig.ident.to_string(), |visitor| {
            visit::visit_impl_item_fn(visitor, item);
        });
    }

    fn visit_item_mod(&mut self, item: &'ast ItemMod) {
        if cfg_test(&item.attrs) {
            return;
        }
        self.with_context(item.ident.to_string(), |visitor| {
            visit::visit_item_mod(visitor, item);
        });
    }

    fn visit_item_trait(&mut self, item: &'ast ItemTrait) {
        if cfg_test(&item.attrs) {
            return;
        }
        self.with_context(item.ident.to_string(), |visitor| {
            visit::visit_item_trait(visitor, item);
        });
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        let method = expression.method.to_string();
        if is_challenge_call(&method) {
            self.record_challenge(&method);
        }
        let receiver = expression
            .receiver
            .to_token_stream()
            .to_string()
            .replace(' ', "");
        let arguments = expression
            .args
            .to_token_stream()
            .to_string()
            .replace(' ', "");
        let receives_transcript = arguments.to_ascii_lowercase().contains("transcript");
        let is_named_absorb =
            method.starts_with("absorb") || (method.starts_with("bind_") && receives_transcript);
        // A bare `self.append(...)` is a transcript absorb only when `self`
        // is a transcript: inside the `jolt-transcript` package or inside a
        // `*Transcript*` impl/trait scope. Without the scope requirement,
        // unrelated builder methods (e.g. `BlindFoldStatement::build`'s
        // `self.append(builder, ...)`) enroll as false positives.
        let self_is_transcript = self.file_id.starts_with("jolt-transcript::")
            || self
                .context
                .iter()
                .any(|scope| scope.contains("Transcript"));
        let is_transcript_method = is_absorb_method(&method)
            && (method == "append_to_transcript"
                || receiver.to_ascii_lowercase().contains("transcript")
                || (receiver == "self" && self_is_transcript));
        if is_named_absorb || is_transcript_method {
            self.record_absorb(&method, expression);
        }
        visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_call(&mut self, expression: &'ast ExprCall) {
        let function = expression
            .func
            .to_token_stream()
            .to_string()
            .replace(' ', "");
        let name = function.rsplit("::").next().unwrap_or(&function);
        let arguments = expression
            .args
            .to_token_stream()
            .to_string()
            .replace(' ', "");
        if name == "append_to_transcript"
            || name.starts_with("absorb")
            || ((name.starts_with("append_") || name.starts_with("bind_"))
                && arguments.to_ascii_lowercase().contains("transcript"))
        {
            self.record_absorb(name, expression);
        }
        visit::visit_expr_call(self, expression);
    }

    fn visit_macro(&mut self, invocation: &'ast Macro) {
        if invocation
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "fs_scope_guard")
        {
            let context = if self.context.is_empty() {
                "<module>".to_owned()
            } else {
                self.context.join("::")
            };
            let _ = self.scope_sites.insert(format!(
                "{}::{}::fs_scope_guard({})",
                self.file_id,
                context,
                invocation.tokens.to_string().replace(' ', "")
            ));
        }
        visit::visit_macro(self, invocation);
    }

    fn visit_trait_item_fn(&mut self, item: &'ast TraitItemFn) {
        if cfg_test(&item.attrs) || prover_only_name(&item.sig.ident.to_string()) {
            return;
        }
        self.with_context(item.sig.ident.to_string(), |visitor| {
            visit::visit_trait_item_fn(visitor, item);
        });
    }

    fn visit_item_struct(&mut self, item: &'ast ItemStruct) {
        if !cfg_test(&item.attrs) {
            self.record_struct(item);
            visit::visit_item_struct(self, item);
        }
    }

    fn visit_item_enum(&mut self, item: &'ast ItemEnum) {
        if !cfg_test(&item.attrs) {
            self.record_enum(item);
            visit::visit_item_enum(self, item);
        }
    }
}

fn cfg_test(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute
                .meta
                .to_token_stream()
                .to_string()
                .contains("test")
    })
}

fn has_derive(attributes: &[Attribute], derive: &str) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("derive")
            && attribute
                .meta
                .to_token_stream()
                .to_string()
                .split(|character: char| !character.is_alphanumeric() && character != '_')
                .any(|name| name == derive)
    })
}

fn prover_only_name(name: &str) -> bool {
    name.starts_with("prove") || name.starts_with("commit_round")
}

fn is_challenge_call(name: &str) -> bool {
    matches!(
        name,
        "challenge" | "challenge_scalar" | "challenge_vector" | "challenge_scalar_powers"
    )
}

fn is_absorb_method(name: &str) -> bool {
    matches!(name, "append" | "append_bytes" | "append_to_transcript")
}

fn is_source_type(name: &str) -> bool {
    matches!(
        name,
        "AkitaJointOpeningProof"
            | "ClearProofClaims"
            | "CommittedProgramPreprocessing"
            | "JoltCommitments"
            | "JoltDevice"
            | "JoltProgramPreprocessing"
            | "JoltProof"
            | "JoltProofClaims"
            | "JoltProtocolConfig"
            | "JoltStageProofs"
            | "JoltVerifierPreprocessing"
            | "MemoryLayout"
            | "ProgramMetadata"
            | "ProgramPreprocessing"
            | "ZkConfig"
    )
}
