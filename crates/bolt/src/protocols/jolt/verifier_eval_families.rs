use crate::emit::rust::{push_format, EmitError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IndexedEvalFamilyPlan {
    pub(crate) symbol: String,
    pub(crate) evals: Vec<String>,
}

impl IndexedEvalFamilyPlan {
    pub(crate) fn from_parts(
        symbol: String,
        evals: Vec<String>,
        count: usize,
    ) -> Result<Self, EmitError> {
        if count != evals.len() {
            return Err(EmitError::new(format!(
                "indexed eval family @{symbol} count mismatch: expected {count}, got {}",
                evals.len()
            )));
        }
        Ok(Self { symbol, evals })
    }

    pub(crate) fn find<'a>(
        families: &'a [Self],
        symbol: &str,
    ) -> Result<&'a IndexedEvalFamilyPlan, EmitError> {
        Self::find_with_index(families, symbol).map(|(_, family)| family)
    }

    pub(crate) fn find_with_index<'a>(
        families: &'a [Self],
        symbol: &str,
    ) -> Result<(usize, &'a IndexedEvalFamilyPlan), EmitError> {
        let mut matching_families = families
            .iter()
            .enumerate()
            .filter(|(_, family)| family.symbol == symbol);
        let (index, family) = matching_families
            .next()
            .ok_or_else(|| EmitError::new(format!("missing eval family `{symbol}`")))?;
        if matching_families.next().is_some() {
            return Err(EmitError::new(format!("duplicate eval family `{symbol}`")));
        }
        Ok((index, family))
    }
}

pub(crate) fn emit_runtime_slice_constant(
    families: &[IndexedEvalFamilyPlan],
    visibility: &str,
    names_prefix: &str,
    families_const: &str,
    family_type: &str,
) -> String {
    let mut source = String::new();
    let mut family_rows = Vec::with_capacity(families.len());
    for (index, family) in families.iter().enumerate() {
        let names_const = format!("{names_prefix}_{index}_NAMES");
        let names_source = family
            .evals
            .iter()
            .map(|name| rust_str(name))
            .collect::<Vec<_>>()
            .join(", ");
        push_format(
            &mut source,
            format_args!("#[rustfmt::skip]\nconst {names_const}: &[&str] = &[{names_source}];\n"),
        );
        family_rows.push(format!(
            "    {family_type} {{ symbol: {}, evals: {names_const} }},",
            rust_str(&family.symbol),
        ));
    }
    push_format(
        &mut source,
        format_args!(
            "{visibility}const {families_const}: &[{family_type}] = &[\n{}\n];\n\n",
            family_rows.join("\n"),
        ),
    );
    source
}

pub(crate) fn emit_named_runtime_slice_constant(
    families: &[IndexedEvalFamilyPlan],
    visibility: &str,
    names_prefix: &str,
    families_const: &str,
) -> String {
    emit_runtime_slice_constant(
        families,
        visibility,
        names_prefix,
        families_const,
        "bolt_verifier_runtime::NamedEvalFamilyPlan",
    )
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use super::{
        emit_named_runtime_slice_constant, emit_runtime_slice_constant, IndexedEvalFamilyPlan,
    };

    #[test]
    fn find_rejects_missing_families() {
        let error = IndexedEvalFamilyPlan::find(&[], "stage.eval.LookupTableFlag")
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert!(error.contains("missing eval family `stage.eval.LookupTableFlag`"));
    }

    #[test]
    fn from_parts_rejects_count_mismatch() {
        let error = IndexedEvalFamilyPlan::from_parts(
            "stage.eval.BytecodeRa".to_owned(),
            vec!["stage.eval.BytecodeRa_0".to_owned()],
            2,
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();

        assert!(error.contains(
            "indexed eval family @stage.eval.BytecodeRa count mismatch: expected 2, got 1"
        ));
    }

    #[test]
    fn find_rejects_duplicate_families() {
        let families = [
            IndexedEvalFamilyPlan {
                symbol: "stage.eval.BytecodeRa".to_owned(),
                evals: vec!["stage.eval.BytecodeRa_0".to_owned()],
            },
            IndexedEvalFamilyPlan {
                symbol: "stage.eval.BytecodeRa".to_owned(),
                evals: vec!["stage.eval.BytecodeRa_1".to_owned()],
            },
        ];

        let error = IndexedEvalFamilyPlan::find(&families, "stage.eval.BytecodeRa")
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert!(error.contains("duplicate eval family `stage.eval.BytecodeRa`"));
    }

    #[test]
    fn find_with_index_preserves_family_row_position() -> Result<(), crate::emit::rust::EmitError> {
        let families = [
            IndexedEvalFamilyPlan {
                symbol: "stage.eval.LookupTableFlag".to_owned(),
                evals: vec!["stage.eval.LookupTableFlag_0".to_owned()],
            },
            IndexedEvalFamilyPlan {
                symbol: "stage.eval.BytecodeRa".to_owned(),
                evals: vec!["stage.eval.BytecodeRa_0".to_owned()],
            },
        ];

        let (index, family) =
            IndexedEvalFamilyPlan::find_with_index(&families, "stage.eval.BytecodeRa")?;

        assert_eq!(index, 1);
        assert_eq!(family.symbol, "stage.eval.BytecodeRa");
        Ok(())
    }

    #[test]
    fn emit_runtime_slice_constant_groups_eval_families() {
        let source = emit_runtime_slice_constant(
            &[IndexedEvalFamilyPlan {
                symbol: "stage.eval.BytecodeRa".to_owned(),
                evals: vec![
                    "stage.eval.BytecodeRa_0".to_owned(),
                    "stage.eval.BytecodeRa_1".to_owned(),
                ],
            }],
            "pub ",
            "STAGE_EVAL_FAMILY",
            "STAGE_EVAL_FAMILIES",
            "bolt_verifier_runtime::NamedEvalFamilyPlan",
        );

        assert!(source.contains("pub const STAGE_EVAL_FAMILIES"));
        assert!(source.contains("STAGE_EVAL_FAMILY_0_NAMES"));
        assert!(source.contains("stage.eval.BytecodeRa_1"));
    }

    #[test]
    fn emit_named_runtime_slice_constant_uses_runtime_family_plan() {
        let source = emit_named_runtime_slice_constant(
            &[IndexedEvalFamilyPlan {
                symbol: "stage.eval.BytecodeRa".to_owned(),
                evals: vec!["stage.eval.BytecodeRa_0".to_owned()],
            }],
            "pub ",
            "STAGE_EVAL_FAMILY",
            "STAGE_EVAL_FAMILIES",
        );

        assert!(source.contains("bolt_verifier_runtime::NamedEvalFamilyPlan"));
    }
}
