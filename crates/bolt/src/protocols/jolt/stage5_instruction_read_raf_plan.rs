use crate::emit::rust::{push_format, EmitError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5InstructionReadRafEmitPlan {
    pub(crate) point: String,
    pub(crate) lookup_output_point: String,
    pub(crate) table_flag_evals: Stage5NamedEvalFamilyEmitPlan,
    pub(crate) instruction_ra_evals: Stage5NamedEvalFamilyEmitPlan,
    pub(crate) raf_flag_eval: String,
    pub(crate) gamma: String,
    pub(crate) log_k: usize,
}

impl Stage5InstructionReadRafEmitPlan {
    pub(crate) fn from_evals<'a>(
        evals: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, EmitError> {
        let evals = evals.into_iter().collect::<Vec<_>>();
        Ok(Self {
            point: "stage5.instruction_read_raf.point".to_owned(),
            lookup_output_point: "stage5.input.stage2.instruction.LookupOutput".to_owned(),
            table_flag_evals: Stage5NamedEvalFamilyEmitPlan::from_indexed_oracles(
                "stage5.instruction_read_raf.eval.LookupTableFlag",
                "LookupTableFlag_",
                evals.iter().copied(),
            )?,
            instruction_ra_evals: Stage5NamedEvalFamilyEmitPlan::from_indexed_oracles(
                "stage5.instruction_read_raf.eval.InstructionRa",
                "InstructionRa_",
                evals.iter().copied(),
            )?,
            raf_flag_eval: "stage5.instruction_read_raf.eval.InstructionRafFlag".to_owned(),
            gamma: "stage5.instruction_read_raf.gamma".to_owned(),
            log_k: 128,
        })
    }

    pub(crate) fn emit_runtime_constants(&self) -> String {
        let families = [
            (
                "STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVAL_NAMES",
                "STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS",
                &self.table_flag_evals,
            ),
            (
                "STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVAL_NAMES",
                "STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS",
                &self.instruction_ra_evals,
            ),
        ];

        let mut source = String::new();
        for (names_const, family_const, family) in families {
            let names_source = family
                .evals
                .iter()
                .map(|name| rust_str(name))
                .collect::<Vec<_>>()
                .join(", ");
            push_format(
                &mut source,
                format_args!(
                    "#[rustfmt::skip]\npub const {names_const}: &[&str] = &[{names_source}];\n"
                ),
            );
            push_format(
                &mut source,
                format_args!(
                    "pub const {family_const}: NamedEvalFamilyPlan = NamedEvalFamilyPlan {{ symbol: {}, evals: {names_const} }};\n\n",
                    rust_str(&family.symbol),
                ),
            );
        }
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_INSTRUCTION_READ_RAF_PLAN: Stage5InstructionReadRafPlan = Stage5InstructionReadRafPlan {{\n\
                 \x20   point: {},\n\
                 \x20   lookup_output_point: {},\n\
                 \x20   table_flag_evals: &STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS,\n\
                 \x20   instruction_ra_evals: &STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS,\n\
                 \x20   raf_flag_eval: {},\n\
                 \x20   gamma: {},\n\
                 \x20   log_k: {},\n\
                 }};\n\n",
                rust_str(&self.point),
                rust_str(&self.lookup_output_point),
                rust_str(&self.raf_flag_eval),
                rust_str(&self.gamma),
                self.log_k,
            ),
        );
        source
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5NamedEvalFamilyEmitPlan {
    pub(crate) symbol: String,
    pub(crate) evals: Vec<String>,
}

impl Stage5NamedEvalFamilyEmitPlan {
    fn from_indexed_oracles<'a>(
        symbol: &str,
        oracle_prefix: &str,
        evals: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, EmitError> {
        let mut indexed_names = Vec::new();
        for (oracle, name) in evals {
            let Some(suffix) = oracle.strip_prefix(oracle_prefix) else {
                continue;
            };
            let index = suffix.parse::<usize>().map_err(|_| {
                EmitError::new(format!(
                    "invalid indexed eval oracle `{oracle}` for family `{symbol}`"
                ))
            })?;
            indexed_names.push((index, name.to_owned()));
        }
        if indexed_names.is_empty() {
            return Err(EmitError::new(format!("missing eval family `{symbol}`")));
        }
        indexed_names.sort_by_key(|(index, _)| *index);
        for (expected, (actual, _)) in indexed_names.iter().enumerate() {
            if expected != *actual {
                return Err(EmitError::new(format!(
                    "non-contiguous eval family `{symbol}` at index {actual}"
                )));
            }
        }
        Ok(Self {
            symbol: symbol.to_owned(),
            evals: indexed_names
                .into_iter()
                .map(|(_, name)| name)
                .collect::<Vec<_>>(),
        })
    }
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use crate::emit::rust::EmitError;

    use super::Stage5InstructionReadRafEmitPlan;

    #[test]
    fn instruction_read_raf_plan_groups_indexed_eval_families() -> Result<(), EmitError> {
        let plan = Stage5InstructionReadRafEmitPlan::from_evals([
            (
                "LookupTableFlag_1",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1",
            ),
            (
                "InstructionRafFlag",
                "stage5.instruction_read_raf.eval.InstructionRafFlag",
            ),
            (
                "InstructionRa_0",
                "stage5.instruction_read_raf.eval.InstructionRa_0",
            ),
            (
                "LookupTableFlag_0",
                "stage5.instruction_read_raf.eval.LookupTableFlag_0",
            ),
        ])?;

        assert_eq!(
            plan.table_flag_evals.evals,
            vec![
                "stage5.instruction_read_raf.eval.LookupTableFlag_0",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1"
            ]
        );
        assert_eq!(
            plan.instruction_ra_evals.evals,
            vec!["stage5.instruction_read_raf.eval.InstructionRa_0"]
        );
        Ok(())
    }

    #[test]
    fn instruction_read_raf_plan_rejects_non_contiguous_eval_families() -> Result<(), EmitError> {
        let error = match Stage5InstructionReadRafEmitPlan::from_evals([
            (
                "LookupTableFlag_1",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1",
            ),
            (
                "InstructionRa_0",
                "stage5.instruction_read_raf.eval.InstructionRa_0",
            ),
        ]) {
            Ok(_) => {
                return Err(EmitError::new(
                    "non-contiguous table flag family should fail planning",
                ));
            }
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("non-contiguous eval family `stage5.instruction_read_raf.eval.LookupTableFlag` at index 1")
        );
        Ok(())
    }
}
