use crate::emit::rust::{push_format, EmitError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IndexedEvalFamilyPlan {
    pub(crate) symbol: String,
    pub(crate) evals: Vec<String>,
}

impl IndexedEvalFamilyPlan {
    pub(crate) fn find<'a>(
        families: &'a [Self],
        symbol: &str,
    ) -> Result<&'a IndexedEvalFamilyPlan, EmitError> {
        let mut matching_families = families.iter().filter(|family| family.symbol == symbol);
        let family = matching_families
            .next()
            .ok_or_else(|| EmitError::new(format!("missing eval family `{symbol}`")))?;
        if matching_families.next().is_some() {
            return Err(EmitError::new(format!("duplicate eval family `{symbol}`")));
        }
        Ok(family)
    }

    pub(crate) fn emit_runtime_constant(
        &self,
        visibility: &str,
        names_const: &str,
        family_const: &str,
        family_type: &str,
    ) -> String {
        let names_source = self
            .evals
            .iter()
            .map(|name| rust_str(name))
            .collect::<Vec<_>>()
            .join(", ");
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "#[rustfmt::skip]\n{visibility}const {names_const}: &[&str] = &[{names_source}];\n"
            ),
        );
        push_format(
            &mut source,
            format_args!(
                "{visibility}const {family_const}: {family_type} = {family_type} {{ symbol: {}, evals: {names_const} }};\n\n",
                rust_str(&self.symbol),
            ),
        );
        source
    }
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use super::IndexedEvalFamilyPlan;

    #[test]
    fn find_rejects_missing_families() {
        let error = IndexedEvalFamilyPlan::find(&[], "stage.eval.LookupTableFlag")
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert!(error.contains("missing eval family `stage.eval.LookupTableFlag`"));
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
}
