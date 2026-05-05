mod body;
mod role;

use crate::ir::{BoltModule, Phase};

use body::push_body_text;
pub(super) use role::PhaseCopyRole;

pub(super) fn phase_copy_source<P: Phase>(
    module: &BoltModule<'_, P>,
    target_phase: &str,
    role: PhaseCopyRole<'_>,
    prefix_ops: &[String],
) -> String {
    let mut source = format!(
        "module @{} attributes {{bolt.phase = \"{target_phase}\"",
        module.name()
    );
    role.append_attr(&mut source);
    source.push_str("} {\n");
    for op in prefix_ops {
        source.push_str(op);
        source.push('\n');
    }
    push_body_text(&mut source, module);
    source.push_str("}\n");
    source
}
