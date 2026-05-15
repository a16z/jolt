use std::collections::BTreeMap;

use crate::emit::rust::{push_format, EmitError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IndexedEvalFamilyPlan {
    pub(crate) symbol: String,
    pub(crate) evals: Vec<String>,
}

impl IndexedEvalFamilyPlan {
    pub(crate) fn from_indexed_oracles<'a>(
        symbol: &str,
        oracle_prefix: &str,
        evals: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, EmitError> {
        let evals = indexed_values(
            symbol,
            "eval oracle",
            oracle_prefix,
            evals
                .into_iter()
                .map(|(oracle, name)| (oracle, name.to_owned())),
        )?;
        Ok(Self {
            symbol: symbol.to_owned(),
            evals,
        })
    }

    pub(crate) fn from_indexed_names<'a>(
        symbol: &str,
        name_prefix: &str,
        eval_names: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, EmitError> {
        let evals = indexed_values(
            symbol,
            "eval name",
            name_prefix,
            eval_names
                .into_iter()
                .map(|eval_name| (eval_name, eval_name.to_owned())),
        )?;
        Ok(Self {
            symbol: symbol.to_owned(),
            evals,
        })
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

fn indexed_values<'a>(
    symbol: &str,
    indexed_value_kind: &str,
    prefix: &str,
    values: impl IntoIterator<Item = (&'a str, String)>,
) -> Result<Vec<String>, EmitError> {
    let mut indexed_values = BTreeMap::new();
    for (indexed_name, value) in values {
        let Some(suffix) = indexed_name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            EmitError::new(format!(
                "invalid indexed {indexed_value_kind} `{indexed_name}` for family `{symbol}`"
            ))
        })?;
        if indexed_values.insert(index, value).is_some() {
            return Err(EmitError::new(format!(
                "duplicate indexed {indexed_value_kind} `{indexed_name}` for family `{symbol}`"
            )));
        }
    }
    if indexed_values.is_empty() {
        return Err(EmitError::new(format!("missing eval family `{symbol}`")));
    }

    let mut evals = Vec::with_capacity(indexed_values.len());
    for (expected, (actual, value)) in indexed_values.into_iter().enumerate() {
        if expected != actual {
            return Err(EmitError::new(format!(
                "non-contiguous eval family `{symbol}`: expected index {expected}, got {actual}"
            )));
        }
        evals.push(value);
    }
    Ok(evals)
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use crate::emit::rust::EmitError;

    use super::IndexedEvalFamilyPlan;

    #[test]
    fn named_eval_family_sorts_indexed_oracles() -> Result<(), EmitError> {
        let family = IndexedEvalFamilyPlan::from_indexed_oracles(
            "stage.eval.LookupTableFlag",
            "LookupTableFlag_",
            [
                ("LookupTableFlag_1", "stage.eval.LookupTableFlag_1"),
                ("Other_0", "stage.eval.Other_0"),
                ("LookupTableFlag_0", "stage.eval.LookupTableFlag_0"),
            ],
        )?;

        assert_eq!(
            family.evals,
            vec![
                "stage.eval.LookupTableFlag_0".to_owned(),
                "stage.eval.LookupTableFlag_1".to_owned(),
            ]
        );
        Ok(())
    }

    #[test]
    fn named_eval_family_rejects_gaps() {
        let error = IndexedEvalFamilyPlan::from_indexed_names(
            "stage.eval.BytecodeRa",
            "stage.eval.BytecodeRa_",
            ["stage.eval.BytecodeRa_0", "stage.eval.BytecodeRa_2"],
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();

        assert!(error.contains("non-contiguous eval family `stage.eval.BytecodeRa`"));
    }
}
