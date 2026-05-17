#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct JoltLocalScalarEmitPlan {
    pub(crate) symbol: String,
    pub(crate) kind: JoltLocalScalarMleKind,
}

impl JoltLocalScalarEmitPlan {
    pub(crate) fn is_lookup_table(&self) -> bool {
        matches!(self.kind, JoltLocalScalarMleKind::LookupTable { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum JoltLocalScalarMleKind {
    LookupTable { index: usize },
    LeftOperand,
    RightOperand,
    Identity,
}

pub(crate) fn emit_jolt_local_scalar_constants(
    const_name: &str,
    values: &[JoltLocalScalarEmitPlan],
) -> String {
    let values = values
        .iter()
        .map(|value| {
            format!(
                "    JoltLocalScalarPlan {{ symbol: {}, kind: {} }},",
                rust_str(&value.symbol),
                local_scalar_kind_expr(&value.kind),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {const_name}: &[JoltLocalScalarPlan] = &[\n{values}\n];\n\n")
}

fn local_scalar_kind_expr(kind: &JoltLocalScalarMleKind) -> String {
    match kind {
        JoltLocalScalarMleKind::LookupTable { index } => {
            format!("JoltLocalScalarMleKind::LookupTable {{ index: {index} }}")
        }
        JoltLocalScalarMleKind::LeftOperand => "JoltLocalScalarMleKind::LeftOperand".to_owned(),
        JoltLocalScalarMleKind::RightOperand => "JoltLocalScalarMleKind::RightOperand".to_owned(),
        JoltLocalScalarMleKind::Identity => "JoltLocalScalarMleKind::Identity".to_owned(),
    }
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}
