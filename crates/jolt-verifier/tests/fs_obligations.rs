//! Source census of every place the Jolt protocol touches its Fiat-Shamir
//! transcript, frozen into `tests/fs_inventory/*.inventory`.
//!
//! The census walks the production module tree of the verifier's runtime
//! closure (seeded at `jolt-verifier`, `jolt-dory`, `jolt-akita`) plus
//! `jolt-prover`, so both halves of every prover message are inventoried. It is
//! a review gate, not a soundness check: adding, removing, or reordering a
//! transcript operation changes an inventory line, and the diff must be
//! reviewed and blessed. Run-time agreement between the prover's and
//! verifier's transcripts is checked separately from the `logging` event
//! streams.
//!
//! `jolt-transcript` itself is out of scope: its sponge, framing, and
//! grinding behaviour is pinned by its per-sponge known-answer tests.
//!
//! Only receivers the census can see are transcripts are recorded: names or
//! fields containing `transcript`/`channel`, parameters and `let` bindings
//! typed (or constructed) as a transcript or a `Channel`/`Transcript`-bounded
//! generic, and `self` inside a `Channel`/`*Transcript*` impl. An untyped
//! closure parameter (`|t| t.challenge()`) is invisible to it.
//!
//! Regenerate with
//! `JOLT_FS_BLESS=1 cargo nextest run -p jolt-verifier --test fs_obligations`.

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
    punctuated::Punctuated,
    visit::{self, Visit},
    Attribute, Expr, ExprCall, ExprClosure, ExprMethodCall, FnArg, Generics, ImplItemFn, Item,
    ItemConst, ItemEnum, ItemFn, ItemImpl, ItemMod, ItemStruct, ItemTrait, Local, Macro, Pat,
    Signature, Token, TraitItemFn, WherePredicate,
};

const ABSORB_INVENTORY: &str = "tests/fs_inventory/absorb-sites.inventory";
const MESSAGE_INVENTORY: &str = "tests/fs_inventory/message-sites.inventory";
const CHALLENGE_INVENTORY: &str = "tests/fs_inventory/challenge-sites.inventory";
const DELEGATION_INVENTORY: &str = "tests/fs_inventory/delegation-sites.inventory";
const SITE_INVENTORY: &str = "tests/fs_inventory/site-tags.inventory";
const SOURCE_INVENTORY: &str = "tests/fs_inventory/source-schema.inventory";

/// Public-input absorbs on a [`jolt_transcript::Channel`].
const ABSORB_METHODS: &[&str] = &["public", "public_all", "public_bytes"];
/// Prover messages: written by `ProverTranscript`, read by `VerifierTranscript`,
/// or exchanged role-generically through `Channel`.
const MESSAGE_METHODS: &[&str] = &[
    "send",
    "send_all",
    "send_bytes",
    "send_bounded_bytes",
    "send_nonce",
    "grind",
    "receive",
    "receive_n",
    "receive_bytes",
    "receive_bounded_bytes",
    "receive_nonce",
    "check_grind",
    "exchange",
    "exchange_all",
];
const CHALLENGE_METHODS: &[&str] = &[
    "challenge",
    "challenge_small",
    "challenges_small",
    "challenge_powers",
    "challenge_bytes",
];
/// `CommitmentScheme`'s transcript hooks, recognised by name at any call shape
/// (`PCS::send_commitment(..)`, `<PCS as CommitmentScheme>::..`).
const PCS_ABSORBS: &[&str] = &["absorb_commitment"];
const PCS_MESSAGES: &[&str] = &["send_commitment", "receive_commitment"];

/// Types whose shape is part of the proof or of the public statement. The
/// header and commitments are the proof's leading prover messages; the
/// preprocessing, device, and configuration types are what the preamble must
/// bind. A field added to any of them needs a matching send/receive or public
/// absorb, so the schema diff is reviewed next to the site inventories.
const SOURCE_TYPES: &[&str] = &[
    "CommittedProgramPreprocessing",
    "FieldInlineCommitments",
    "FieldRegistersCommitments",
    "JoltCommitments",
    "JoltDevice",
    "JoltOneHotConfig",
    "JoltProgramPreprocessing",
    "JoltProof",
    "JoltProtocolConfig",
    "JoltReadWriteConfig",
    "JoltVerifierPreprocessing",
    "MemoryLayout",
    "ProgramMetadata",
    "ProgramPreprocessing",
    "ProofCommitments",
    "ProofHeader",
    "TracePolynomialOrder",
    "ZkConfig",
];

#[derive(Clone)]
struct PackageSource {
    name: String,
    src: PathBuf,
}

#[derive(Default)]
struct Census {
    absorb: BTreeSet<String>,
    message: BTreeSet<String>,
    challenge: BTreeSet<String>,
    delegation: BTreeSet<String>,
    site: BTreeSet<String>,
    source: BTreeSet<String>,
}

#[test]
fn fs_inventory_is_complete() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("jolt-verifier must be in <workspace>/crates");
    let mut census = Census::default();

    for package in production_sources(workspace) {
        for path in module_tree(&package.src) {
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
            InventoryVisitor::new(file_id, &mut census).visit_file(&syntax);
        }
    }

    for (file, label, actual) in [
        (ABSORB_INVENTORY, "public absorb sites", &census.absorb),
        (MESSAGE_INVENTORY, "prover message sites", &census.message),
        (CHALLENGE_INVENTORY, "challenge sites", &census.challenge),
        (
            DELEGATION_INVENTORY,
            "transcript delegation sites",
            &census.delegation,
        ),
        (SITE_INVENTORY, "transcript site tags", &census.site),
        (
            SOURCE_INVENTORY,
            "proof and statement schema",
            &census.source,
        ),
    ] {
        check_inventory(&manifest_dir.join(file), label, actual);
    }
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

    // Path dependencies outside the workspace (a local Akita checkout) are
    // upstream code with their own transcript tests.
    let workspace_members = metadata["workspace_members"]
        .as_array()
        .expect("cargo metadata workspace members are missing")
        .iter()
        .map(|id| id.as_str().expect("workspace member id is missing"))
        .collect::<BTreeSet<_>>();
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
        .map(package_id)
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
    // The prover writes the messages the verifier closure reads; its own
    // dependency closure (tracer, witness, kernels) never touches a transcript.
    closure.extend(
        packages
            .iter()
            .filter(|package| package["name"] == "jolt-prover")
            .map(package_id),
    );

    let mut sources = closure
        .into_iter()
        .filter_map(|id| package_by_id.get(id))
        .filter(|package| {
            workspace_members.contains(package_id(package)) && package["name"] != "jolt-transcript"
        })
        .filter_map(|package| {
            let manifest = PathBuf::from(
                package["manifest_path"]
                    .as_str()
                    .expect("package manifest path is missing"),
            );
            let src = manifest.parent()?.join("src");
            src.join("lib.rs").is_file().then(|| PackageSource {
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

/// The library's source files reachable from `lib.rs` through `mod` items that
/// are not test-gated. Feature-gated modules are included: every build's
/// transcript sites are inventoried.
fn module_tree(src: &Path) -> Vec<PathBuf> {
    fn declared_modules(items: &[Item], dir: &Path, output: &mut Vec<PathBuf>) {
        for item in items {
            let Item::Mod(module) = item else {
                continue;
            };
            if cfg_test(&module.attrs) {
                continue;
            }
            let name = module.ident.to_string();
            if let Some((_, items)) = &module.content {
                declared_modules(items, &dir.join(&name), output);
                continue;
            }
            let file = dir.join(format!("{name}.rs"));
            let nested = dir.join(&name).join("mod.rs");
            let path = if file.is_file() { file } else { nested };
            assert!(
                path.is_file(),
                "module `{name}` under {} has no source file",
                dir.display()
            );
            visit_file(&path, output);
        }
    }

    fn visit_file(path: &Path, output: &mut Vec<PathBuf>) {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let syntax = syn::parse_file(&source)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
        let parent = path.parent().expect("source file has no parent");
        let dir = match path.file_stem().and_then(|stem| stem.to_str()) {
            Some("lib" | "mod") => parent.to_path_buf(),
            Some(stem) => parent.join(stem),
            None => panic!("source file {} has no stem", path.display()),
        };
        output.push(path.to_path_buf());
        declared_modules(&syntax.items, &dir, output);
    }

    let mut output = Vec::new();
    visit_file(&src.join("lib.rs"), &mut output);
    output.sort();
    output
}

fn check_inventory(path: &Path, label: &str, actual: &BTreeSet<String>) {
    if env::var_os("JOLT_FS_BLESS").is_some() {
        let body = format!(
            "# Generated by `JOLT_FS_BLESS=1 cargo nextest run -p jolt-verifier \\\n#   --test fs_obligations`.\n# Review changes; this inventories identities, not transcript values.\n{}\n",
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
        "Fiat-Shamir {label} changed.\nUnclassified: {added:#?}\nNo longer present: {removed:#?}\n\
         Rerun with JOLT_FS_BLESS=1 and review the inventory diff."
    );
}

/// Names visible in one item scope that denote a transcript.
#[derive(Clone, Default)]
struct TranscriptScope {
    /// Generic parameters bounded by `Channel` or a `*Transcript*` trait.
    generics: BTreeSet<String>,
    /// Bindings typed or constructed as a transcript.
    bindings: BTreeSet<String>,
    /// `self` is itself a transcript.
    self_is_transcript: bool,
}

struct InventoryVisitor<'a> {
    file_id: String,
    context: Vec<String>,
    scopes: Vec<TranscriptScope>,
    /// One counter per context, shared by absorb, message, challenge, and
    /// delegation records, so an identity encodes its position in the
    /// function's combined transcript sequence. Reordering a message against
    /// a squeeze — the canonical weak-FS bug — renumbers both sites and trips
    /// the inventory even when each per-kind subsequence is unchanged.
    call_ordinals: BTreeMap<String, usize>,
    census: &'a mut Census,
}

impl<'a> InventoryVisitor<'a> {
    fn new(file_id: String, census: &'a mut Census) -> Self {
        Self {
            file_id,
            context: Vec::new(),
            scopes: vec![TranscriptScope::default()],
            call_ordinals: BTreeMap::new(),
            census,
        }
    }

    fn scope(&mut self) -> &mut TranscriptScope {
        self.scopes.last_mut().expect("scope stack is never empty")
    }

    fn with_context(
        &mut self,
        name: String,
        scope: TranscriptScope,
        visit: impl FnOnce(&mut Self),
    ) {
        self.context.push(name);
        self.scopes.push(scope);
        visit(self);
        let _ = self.scopes.pop();
        let _ = self.context.pop();
    }

    /// The scope for a new item: inherits enclosing generics and `self`, not
    /// bindings, and adds `generics`' transcript-bounded parameters.
    fn child_scope(&self, generics: &Generics) -> TranscriptScope {
        let parent = self.scopes.last().expect("scope stack is never empty");
        let mut scope = TranscriptScope {
            generics: parent.generics.clone(),
            bindings: BTreeSet::new(),
            self_is_transcript: parent.self_is_transcript,
        };
        for parameter in generics.type_params() {
            if names_transcript_type(&tokens(&parameter.bounds)) {
                let _ = scope.generics.insert(parameter.ident.to_string());
            }
        }
        for predicate in generics.where_clause.iter().flat_map(|w| &w.predicates) {
            if let WherePredicate::Type(predicate) = predicate {
                if names_transcript_type(&tokens(&predicate.bounds)) {
                    let _ = scope.generics.insert(tokens(&predicate.bounded_ty));
                }
            }
        }
        scope
    }

    fn fn_scope(&self, signature: &Signature) -> TranscriptScope {
        let mut scope = self.child_scope(&signature.generics);
        for input in &signature.inputs {
            if let FnArg::Typed(argument) = input {
                if is_transcript_type(&scope, &tokens(&argument.ty)) {
                    if let Some(name) = binding_name(&argument.pat) {
                        let _ = scope.bindings.insert(name);
                    }
                }
            }
        }
        scope
    }

    fn is_transcript_expr(&self, expression: &str) -> bool {
        let expression = expression
            .trim_start_matches(['&', '*'])
            .trim_start_matches("mut");
        let is_place = expression
            .chars()
            .all(|character| character.is_alphanumeric() || matches!(character, '_' | '.'));
        if !is_place {
            return false;
        }
        let lowercase = expression.to_ascii_lowercase();
        if lowercase.contains("transcript") || lowercase.contains("channel") {
            return true;
        }
        let root = expression.split('.').next().unwrap_or_default();
        let scope = self.scopes.last().expect("scope stack is never empty");
        scope.bindings.contains(root) || (expression == "self" && scope.self_is_transcript)
    }

    fn next_ordinal(&mut self) -> (String, usize) {
        let context = self.context();
        let ordinal = self.call_ordinals.entry(context.clone()).or_default();
        let current = *ordinal;
        *ordinal += 1;
        (context, current)
    }

    fn record_with_expression(&mut self, kind: Kind, name: &str, expression: &impl ToTokens) {
        let (context, ordinal) = self.next_ordinal();
        let entry = format!(
            "{}::{context}::{name}#{ordinal}::{}",
            self.file_id,
            tokens(expression)
        );
        let _ = match kind {
            Kind::Absorb => self.census.absorb.insert(entry),
            Kind::Message => self.census.message.insert(entry),
        };
    }

    fn record_challenge(&mut self, name: &str) {
        let (context, ordinal) = self.next_ordinal();
        let _ = self
            .census
            .challenge
            .insert(format!("{}::{context}::{name}#{ordinal}", self.file_id));
    }

    fn record_delegation(&mut self, callee: &str) {
        let (context, ordinal) = self.next_ordinal();
        let _ = self
            .census
            .delegation
            .insert(format!("{}::{context}::{callee}#{ordinal}", self.file_id));
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
            let _ = self.census.challenge.insert(format!(
                "{}::{}::<generated>::batching_coefficient[*]",
                self.file_id, item.ident
            ));
        }
        if !SOURCE_TYPES.contains(&item.ident.to_string().as_str()) {
            return;
        }
        for (index, field) in item.fields.iter().enumerate() {
            let name = field
                .ident
                .as_ref()
                .map_or_else(|| index.to_string(), ToString::to_string);
            let _ = self.census.source.insert(format!(
                "{}::{}.{name}{}",
                self.file_id,
                item.ident,
                cfg_suffix(&field.attrs)
            ));
        }
    }

    fn record_enum(&mut self, item: &ItemEnum) {
        if !SOURCE_TYPES.contains(&item.ident.to_string().as_str()) {
            return;
        }
        for variant in &item.variants {
            let variant_id = format!("{}::{}::{}", self.file_id, item.ident, variant.ident);
            if variant.fields.is_empty() {
                let _ = self
                    .census
                    .source
                    .insert(format!("{variant_id}{}", cfg_suffix(&variant.attrs)));
            }
            for (index, field) in variant.fields.iter().enumerate() {
                let name = field
                    .ident
                    .as_ref()
                    .map_or_else(|| index.to_string(), ToString::to_string);
                let _ = self
                    .census
                    .source
                    .insert(format!("{variant_id}.{name}{}", cfg_suffix(&field.attrs)));
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Absorb,
    Message,
}

impl<'ast> Visit<'ast> for InventoryVisitor<'_> {
    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        if cfg_test(&item.attrs) {
            return;
        }
        let scope = self.fn_scope(&item.sig);
        self.with_context(item.sig.ident.to_string(), scope, |visitor| {
            visit::visit_item_fn(visitor, item);
        });
    }

    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        if cfg_test(&item.attrs) {
            return;
        }
        let name = tokens(&item.self_ty);
        let mut scope = self.child_scope(&item.generics);
        let implements_channel = item
            .trait_
            .as_ref()
            .and_then(|(path, ..)| path.segments.last())
            .is_some_and(|segment| segment.ident == "Channel");
        scope.self_is_transcript = implements_channel || name.contains("Transcript");
        self.with_context(name, scope, |visitor| visit::visit_item_impl(visitor, item));
    }

    fn visit_impl_item_fn(&mut self, item: &'ast ImplItemFn) {
        if cfg_test(&item.attrs) {
            return;
        }
        let scope = self.fn_scope(&item.sig);
        self.with_context(item.sig.ident.to_string(), scope, |visitor| {
            visit::visit_impl_item_fn(visitor, item);
        });
    }

    fn visit_trait_item_fn(&mut self, item: &'ast TraitItemFn) {
        if cfg_test(&item.attrs) {
            return;
        }
        let scope = self.fn_scope(&item.sig);
        self.with_context(item.sig.ident.to_string(), scope, |visitor| {
            visit::visit_trait_item_fn(visitor, item);
        });
    }

    fn visit_item_mod(&mut self, item: &'ast ItemMod) {
        if cfg_test(&item.attrs) {
            return;
        }
        let scope = TranscriptScope::default();
        self.with_context(item.ident.to_string(), scope, |visitor| {
            visit::visit_item_mod(visitor, item);
        });
    }

    fn visit_item_trait(&mut self, item: &'ast ItemTrait) {
        if cfg_test(&item.attrs) {
            return;
        }
        let mut scope = self.child_scope(&item.generics);
        scope.self_is_transcript =
            item.ident == "Channel" || item.ident.to_string().contains("Transcript");
        self.with_context(item.ident.to_string(), scope, |visitor| {
            visit::visit_item_trait(visitor, item);
        });
    }

    fn visit_item_const(&mut self, item: &'ast ItemConst) {
        if !cfg_test(&item.attrs) && tokens(&item.ty) == "SiteId" {
            let _ = self.census.site.insert(format!(
                "{}::const {}={}",
                self.file_id,
                item.ident,
                tokens(&item.expr)
            ));
        }
        visit::visit_item_const(self, item);
    }

    fn visit_local(&mut self, local: &'ast Local) {
        let (pattern, ty) = match &local.pat {
            Pat::Type(typed) => (&*typed.pat, Some(tokens(&typed.ty))),
            pattern => (pattern, None),
        };
        let constructs_transcript = local
            .init
            .as_ref()
            .is_some_and(|init| tokens(&init.expr).contains("Transcript::new"));
        let typed_transcript = ty.is_some_and(|ty| {
            let scope = self.scopes.last().expect("scope stack is never empty");
            is_transcript_type(scope, &ty)
        });
        if constructs_transcript || typed_transcript {
            if let Some(name) = binding_name(pattern) {
                let _ = self.scope().bindings.insert(name);
            }
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_closure(&mut self, closure: &'ast ExprClosure) {
        for input in &closure.inputs {
            if let Pat::Type(typed) = input {
                let scope = self.scopes.last().expect("scope stack is never empty");
                if is_transcript_type(scope, &tokens(&typed.ty)) {
                    if let Some(name) = binding_name(&typed.pat) {
                        let _ = self.scope().bindings.insert(name);
                    }
                }
            }
        }
        visit::visit_expr_closure(self, closure);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        let method = expression.method.to_string();
        let on_transcript = self.is_transcript_expr(&tokens(&expression.receiver));
        if PCS_ABSORBS.contains(&method.as_str()) {
            self.record_with_expression(Kind::Absorb, &method, expression);
        } else if PCS_MESSAGES.contains(&method.as_str()) {
            self.record_with_expression(Kind::Message, &method, expression);
        } else if on_transcript && ABSORB_METHODS.contains(&method.as_str()) {
            self.record_with_expression(Kind::Absorb, &method, expression);
        } else if on_transcript && MESSAGE_METHODS.contains(&method.as_str()) {
            self.record_with_expression(Kind::Message, &method, expression);
        } else if on_transcript && CHALLENGE_METHODS.contains(&method.as_str()) {
            self.record_challenge(&method);
        } else if on_transcript && method == "site" {
            let _ = self.census.site.insert(format!(
                "{}::{}::site({})",
                self.file_id,
                self.context(),
                tokens(&expression.args)
            ));
        } else if on_transcript
            || expression
                .args
                .iter()
                .any(|argument| self.is_transcript_expr(&tokens(argument)))
        {
            self.record_delegation(&format!(".{method}"));
        }
        visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_call(&mut self, expression: &'ast ExprCall) {
        let function = tokens(&expression.func);
        let name = function
            .rsplit("::")
            .find(|segment| !segment.starts_with('<'))
            .unwrap_or(&function)
            .to_owned();
        if PCS_ABSORBS.contains(&name.as_str()) {
            self.record_with_expression(Kind::Absorb, &name, expression);
        } else if PCS_MESSAGES.contains(&name.as_str()) {
            self.record_with_expression(Kind::Message, &name, expression);
        } else if expression
            .args
            .iter()
            .any(|argument| self.is_transcript_expr(&tokens(argument)))
        {
            self.record_delegation(&function);
        }
        visit::visit_expr_call(self, expression);
    }

    /// Expression-list macro bodies (`vec!`, `assert_eq!`, `format!`) are
    /// visited as expressions; other bodies are opaque to the census.
    fn visit_macro(&mut self, invocation: &'ast Macro) {
        if let Ok(arguments) =
            invocation.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
        {
            for argument in &arguments {
                self.visit_expr(argument);
            }
        }
        visit::visit_macro(self, invocation);
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

fn package_id(package: &Value) -> &str {
    package["id"].as_str().expect("package id is missing")
}

fn tokens(node: &impl ToTokens) -> String {
    node.to_token_stream().to_string().replace(' ', "")
}

fn names_transcript_type(bounds: &str) -> bool {
    bounds.contains("Channel") || bounds.contains("Transcript")
}

fn is_transcript_type(scope: &TranscriptScope, ty: &str) -> bool {
    names_transcript_type(ty)
        || ty
            .split(|character: char| !character.is_alphanumeric() && character != '_')
            .any(|token| scope.generics.contains(token))
}

fn binding_name(pattern: &Pat) -> Option<String> {
    match pattern {
        Pat::Ident(binding) => Some(binding.ident.to_string()),
        Pat::Type(typed) => binding_name(&typed.pat),
        _ => None,
    }
}

fn cfg_test(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("cfg") && tokens(&attribute.meta).contains("test")
    })
}

/// ` @cfg(...)` for a cfg-gated field, so per-build proof shapes stay distinct.
fn cfg_suffix(attributes: &[Attribute]) -> String {
    let mut suffix = String::new();
    for attribute in attributes
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
    {
        suffix.push_str(" @");
        suffix.push_str(&tokens(&attribute.meta));
    }
    suffix
}

fn has_derive(attributes: &[Attribute], derive: &str) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("derive")
            && tokens(&attribute.meta)
                .split(|character: char| !character.is_alphanumeric() && character != '_')
                .any(|name| name == derive)
    })
}
