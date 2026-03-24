//! Shader compilation utilities for the Metal backend.
//!
//! Provides shared infrastructure for compiling MSL source from generated
//! field arithmetic and embedded shader files, and pre-compiled pipeline
//! bundles for element-wise and interpolation operations.

use metal::{CompileOptions, ComputePipelineState, Device, Library};

use crate::field_config::MslFieldParams;

pub(crate) const SHADER_COMMON: &str = include_str!("shaders/common.metal");
pub(crate) const SHADER_ELEMENTWISE: &str = include_str!("shaders/elementwise.metal");
pub(crate) const SHADER_INTERPOLATION: &str = include_str!("shaders/interpolation.metal");

/// Pre-compiled pipelines for element-wise Fr operations on Metal.
pub(crate) struct ElementwiseKernels {
    pub scale: ComputePipelineState,
    pub add: ComputePipelineState,
    pub sub: ComputePipelineState,
    pub accumulate: ComputePipelineState,
    pub sum: ComputePipelineState,
    pub dot_product: ComputePipelineState,
}

impl ElementwiseKernels {
    pub fn compile(device: &Device, field_config: &MslFieldParams) -> Self {
        let source = build_source_with_preamble(&field_config.msl_preamble, &[SHADER_ELEMENTWISE]);
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .expect("elementwise MSL compilation failed");

        Self {
            scale: make_pipeline(device, &library, "fr_scale_kernel"),
            add: make_pipeline(device, &library, "fr_add_buf_kernel"),
            sub: make_pipeline(device, &library, "fr_sub_buf_kernel"),
            accumulate: make_pipeline(device, &library, "fr_accumulate_kernel"),
            sum: make_pipeline(device, &library, "fr_sum_kernel"),
            dot_product: make_pipeline(device, &library, "fr_dot_product_kernel"),
        }
    }
}

/// Pre-compiled pipelines for interpolation and product table operations.
pub(crate) struct InterpolationKernels {
    pub interpolate_low: ComputePipelineState,
    pub interpolate_inplace_high: ComputePipelineState,
    pub product_table_round: ComputePipelineState,
}

impl InterpolationKernels {
    pub fn compile(device: &Device, field_config: &MslFieldParams) -> Self {
        let source =
            build_source_with_preamble(&field_config.msl_preamble, &[SHADER_INTERPOLATION]);
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .expect("interpolation MSL compilation failed");

        Self {
            interpolate_low: make_pipeline(device, &library, "fr_interpolate_low_kernel"),
            interpolate_inplace_high: make_pipeline(
                device,
                &library,
                "fr_interpolate_inplace_high_kernel",
            ),
            product_table_round: make_pipeline(device, &library, "fr_product_table_round_kernel"),
        }
    }
}

/// Build MSL source from a generated preamble and additional shader fragments.
///
/// The preamble contains the complete field arithmetic (Fr, constants, ops, WideAcc).
/// Additional shaders use Fr/WideAcc by name — they are field-agnostic.
///
/// `#include` lines in the shader fragments are stripped (already inlined via
/// the preamble).
pub fn build_source_with_preamble(preamble: &str, shaders: &[&str]) -> String {
    let total: usize =
        SHADER_COMMON.len() + preamble.len() + shaders.iter().map(|s| s.len()).sum::<usize>();
    let mut src = String::with_capacity(total + 256);

    src.push_str(SHADER_COMMON);
    src.push('\n');

    // FR_NOINLINE define if present in preamble is already baked in
    src.push_str(preamble);
    src.push('\n');

    for shader in shaders {
        for line in shader.lines() {
            if !line.starts_with("#include") {
                src.push_str(line);
                src.push('\n');
            }
        }
    }

    src
}

/// Build MSL source with noinline mode (for fast compilation).
pub fn build_source_with_preamble_noinline(preamble: &str, shaders: &[&str]) -> String {
    let mut src = String::with_capacity(preamble.len() + 4096);
    src.push_str("#define FR_NOINLINE 1\n");
    src.push_str(&build_source_with_preamble(preamble, shaders));
    src
}

/// Create a compute pipeline from a named kernel function.
pub fn make_pipeline(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let func = library
        .get_function(name, None)
        .unwrap_or_else(|e| panic!("kernel function '{name}' not found: {e}"));
    device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap_or_else(|e| panic!("pipeline creation failed for '{name}': {e}"))
}
