use melior::ir::Module;
use melior::utility::load_irdl_dialects;
use melior::Context;

pub const BOLT_IRDL: &str = concat!(
    "module {\n",
    include_str!("../irdl/field.mlir"),
    "\n",
    include_str!("../irdl/poly.mlir"),
    "\n",
    include_str!("../irdl/hash.mlir"),
    "\n",
    include_str!("../irdl/transcript.mlir"),
    "\n",
    include_str!("../irdl/commit.mlir"),
    "\n",
    include_str!("../irdl/pcs.mlir"),
    "\n",
    include_str!("../irdl/protocol.mlir"),
    "\n",
    include_str!("../irdl/piop.mlir"),
    "\n",
    include_str!("../irdl/party.mlir"),
    "\n",
    include_str!("../irdl/compute.mlir"),
    "\n",
    include_str!("../irdl/cpu.mlir"),
    "\n}\n"
);

pub fn load_bolt_dialects(context: &Context) -> Result<(), String> {
    let module = Module::parse(context, BOLT_IRDL)
        .ok_or_else(|| "failed to parse Bolt IRDL dialect definitions".to_owned())?;
    if load_irdl_dialects(&module) {
        Ok(())
    } else {
        Err("failed to load Bolt IRDL dialect definitions".to_owned())
    }
}
