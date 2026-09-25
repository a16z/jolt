use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::Arc;

use crate::error::MetalError;
use crate::runtime::device::Device;
use crate::runtime::sys;

/// A Rust type with an MSL counterpart that kernel templates can be
/// instantiated over.
pub trait MslType {
    /// The MSL spelling of the type, e.g. `uint`.
    const MSL_NAME: &'static str;
    /// Suffix appended to a template's name to form the instance's host
    /// name, e.g. `u32` in `jolt_vec_add_u32`. Unique per type.
    const HOST_SUFFIX: &'static str;
}

impl MslType for u32 {
    const MSL_NAME: &'static str = "uint";
    const HOST_SUFFIX: &'static str = "u32";
}

impl MslType for u64 {
    const MSL_NAME: &'static str = "ulong";
    const HOST_SUFFIX: &'static str = "u64";
}

/// The host name of `template` instantiated over `T`: `{template}_{suffix}`.
pub fn host_name<T: MslType>(template: &str) -> String {
    format!("{template}_{}", T::HOST_SUFFIX)
}

#[derive(Clone, Debug)]
enum KernelDecl {
    Plain(String),
    Instance {
        template: String,
        msl_type: &'static str,
        host_name: String,
    },
}

impl KernelDecl {
    fn host_name(&self) -> &str {
        match self {
            Self::Plain(name)
            | Self::Instance {
                host_name: name, ..
            } => name,
        }
    }
}

/// The source and the kernels of a [`ShaderLibrary`].
///
/// Sources are concatenated in order, each preceded by a `#line` directive
/// naming it so compiler diagnostics point at the right unit. Template
/// instances are appended as explicit instantiations:
///
/// ```metal
/// template [[host_name("jolt_vec_add_u32")]] [[kernel]]
/// decltype(jolt_vec_add<uint>) jolt_vec_add<uint>;
/// ```
#[derive(Clone, Debug, Default)]
pub struct LibrarySpec {
    sources: Vec<(String, String)>,
    kernels: Vec<KernelDecl>,
}

impl LibrarySpec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a source unit. `name` appears in compiler diagnostics.
    pub fn source(mut self, name: &str, text: &str) -> Self {
        self.sources.push((name.to_owned(), text.to_owned()));
        self
    }

    /// Declares a non-template kernel function.
    pub fn kernel(mut self, name: &str) -> Self {
        self.kernels.push(KernelDecl::Plain(name.to_owned()));
        self
    }

    /// Declares `template<T>`, reachable as [`host_name::<T>(template)`](host_name).
    pub fn instantiate<T: MslType>(mut self, template: &str) -> Self {
        self.kernels.push(KernelDecl::Instance {
            template: template.to_owned(),
            msl_type: T::MSL_NAME,
            host_name: host_name::<T>(template),
        });
        self
    }

    /// Validates the declaration and produces the complete MSL source.
    fn assemble(&self) -> Result<String, MetalError> {
        let invalid = |reason: String| MetalError::InvalidLibrary { reason };
        if self.kernels.is_empty() {
            return Err(invalid("no kernels declared".to_owned()));
        }
        let mut source = String::new();
        for (name, text) in &self.sources {
            if name.contains(['"', '\\', '\n', '\r']) {
                return Err(invalid(format!(
                    "source name {name:?} is not a #line file name"
                )));
            }
            // Writing to a `String` cannot fail.
            let _ = writeln!(source, "#line 1 \"{name}\"\n{text}");
        }
        let _ = writeln!(source, "#line 1 \"jolt-metal instantiations\"");
        let mut seen = BTreeSet::new();
        for decl in &self.kernels {
            let identifier = match decl {
                KernelDecl::Plain(name) => name,
                KernelDecl::Instance { template, .. } => template,
            };
            if !is_identifier(identifier) || !is_identifier(decl.host_name()) {
                return Err(invalid(format!("{identifier:?} is not an MSL identifier")));
            }
            if !seen.insert(decl.host_name()) {
                return Err(invalid(format!(
                    "kernel `{}` declared twice",
                    decl.host_name()
                )));
            }
            if let KernelDecl::Instance {
                template,
                msl_type,
                host_name,
            } = decl
            {
                let _ = writeln!(
                    source,
                    "template [[host_name(\"{host_name}\")]] [[kernel]] \
                     decltype({template}<{msl_type}>) {template}<{msl_type}>;"
                );
            }
        }
        Ok(source)
    }
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A buffer argument of a kernel, from pipeline reflection.
#[derive(Clone, Debug)]
pub(crate) struct ArgumentSlot {
    pub(crate) name: String,
    pub(crate) index: usize,
    /// Size of the pointee (`device T*`) or referent (`constant T&`).
    pub(crate) data_size: usize,
}

/// What the platform backend reads from a newly created pipeline.
pub(crate) struct PipelineInfo {
    pub(crate) max_total_threads_per_threadgroup: usize,
    pub(crate) thread_execution_width: usize,
    /// Buffer arguments in declaration order.
    pub(crate) slots: Vec<ArgumentSlot>,
}

/// A compiled compute kernel.
pub struct Pipeline {
    pub(crate) sys: sys::Pipeline,
    pub(crate) name: Arc<str>,
    pub(crate) device_id: u64,
    pub(crate) info: PipelineInfo,
}

impl Pipeline {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The largest threadgroup this pipeline can be dispatched with.
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.info.max_total_threads_per_threadgroup
    }

    /// SIMD-group width; threadgroup sizes should be a multiple of it.
    pub fn thread_execution_width(&self) -> usize {
        self.info.thread_execution_width
    }
}

/// A compiled library with a pipeline for every declared kernel.
///
/// All pipelines are created in [`ShaderLibrary::compile`], so shader
/// compilation and pipeline creation fail at setup, never mid-proof.
pub struct ShaderLibrary {
    pipelines: BTreeMap<String, Pipeline>,
}

impl ShaderLibrary {
    pub fn compile(device: &Device, spec: &LibrarySpec) -> Result<Self, MetalError> {
        let library = device.sys.compile(&spec.assemble()?)?;
        let mut pipelines = BTreeMap::new();
        for decl in &spec.kernels {
            let kernel = decl.host_name();
            let (sys, mut info) = library.pipeline(&device.sys, kernel)?;
            info.slots.sort_by_key(|slot| slot.index);
            // Bindings are positional, so buffer arguments must occupy
            // indices 0..n with no gaps.
            if let Some((position, slot)) = info
                .slots
                .iter()
                .enumerate()
                .find(|(position, slot)| slot.index != *position)
            {
                return Err(MetalError::Pipeline {
                    kernel: kernel.to_owned(),
                    reason: format!(
                        "buffer argument `{}` is at index {} but position {position} is unbound; \
                         use contiguous [[buffer(i)]] indices from 0",
                        slot.name, slot.index
                    ),
                });
            }
            let pipeline = Pipeline {
                sys,
                name: Arc::from(kernel),
                device_id: device.registry_id,
                info,
            };
            let _ = pipelines.insert(kernel.to_owned(), pipeline);
        }
        Ok(Self { pipelines })
    }

    pub fn pipeline(&self, name: &str) -> Result<&Pipeline, MetalError> {
        self.pipelines
            .get(name)
            .ok_or_else(|| MetalError::UnknownPipeline {
                name: name.to_owned(),
            })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;

    #[test]
    fn instances_become_explicit_instantiations() {
        let source = LibrarySpec::new()
            .source("kernels.metal", "KERNELS")
            .kernel("plain")
            .instantiate::<u32>("vec_add")
            .instantiate::<u64>("vec_add")
            .assemble()
            .unwrap();
        assert_eq!(
            source,
            "#line 1 \"kernels.metal\"\nKERNELS\n\
             #line 1 \"jolt-metal instantiations\"\n\
             template [[host_name(\"vec_add_u32\")]] [[kernel]] decltype(vec_add<uint>) vec_add<uint>;\n\
             template [[host_name(\"vec_add_u64\")]] [[kernel]] decltype(vec_add<ulong>) vec_add<ulong>;\n"
        );
    }

    #[test]
    fn malformed_declarations_are_setup_errors() {
        let rejected = [
            LibrarySpec::new(),
            LibrarySpec::new().kernel("a").kernel("a"),
            LibrarySpec::new().instantiate::<u32>("a").kernel("a_u32"),
            LibrarySpec::new().kernel("1a"),
            LibrarySpec::new().kernel("a(b)"),
            LibrarySpec::new().instantiate::<u32>("f<int>"),
            LibrarySpec::new().source("a\"b", "").kernel("k"),
        ];
        for spec in rejected {
            let error = spec.assemble().unwrap_err();
            assert!(
                matches!(error, MetalError::InvalidLibrary { .. }),
                "{error}"
            );
        }
    }
}
