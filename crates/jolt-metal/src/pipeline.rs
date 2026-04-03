//! MSL source assembly and pipeline compilation utilities.

use metal::{ComputePipelineState, Device, Library};

pub(crate) const SHADER_COMMON: &str = include_str!("shaders/common.metal");
pub(crate) const SHADER_INTERPOLATION: &str = include_str!("shaders/interpolation.metal");

/// Build MSL source from a generated preamble and additional shader fragments.
///
/// The preamble contains the complete field arithmetic (Fr, constants, ops, WideAcc).
/// Additional shaders use Fr/WideAcc by name — they are field-agnostic.
///
/// `#include` lines in the shader fragments are stripped (already inlined via
/// the preamble).
pub fn build_source_with_preamble(preamble: &str, shaders: &[&str], noinline: bool) -> String {
    let total: usize =
        SHADER_COMMON.len() + preamble.len() + shaders.iter().map(|s| s.len()).sum::<usize>();
    let mut src = String::with_capacity(total + 256);

    if noinline {
        src.push_str("#define FR_NOINLINE 1\n");
    }

    src.push_str(SHADER_COMMON);
    src.push('\n');

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

/// Create a compute pipeline from a named kernel function.
pub fn make_pipeline(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let func = library
        .get_function(name, None)
        .unwrap_or_else(|e| panic!("kernel function '{name}' not found: {e}"));
    device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap_or_else(|e| panic!("pipeline creation failed for '{name}': {e}"))
}
