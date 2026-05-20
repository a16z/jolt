use crate::ir::{BoltModule, Protocol};
use crate::mlir::{MeliorContext, MlirError};

use super::params::JoltProtocolParams;

pub const FIELD_SYMBOL: &str = "bn254_fr";
pub const HASH_SYMBOL: &str = "blake2b";
pub const TRANSCRIPT_SYMBOL: &str = "blake2b_transcript";
pub const PCS_SYMBOL: &str = "dory";
pub const TRACE_DOMAIN_SYMBOL: &str = "jolt.trace_domain";
pub const MAIN_WITNESS_COMMIT_DOMAIN_SYMBOL: &str = "jolt.main_witness_commit_domain";
pub const MAIN_WITNESS_FAMILY_SYMBOL: &str = "jolt.main_witness_polys";
pub const ADVICE_FAMILY_SYMBOL: &str = "jolt.advice_polys";

pub fn main_witness_oracles(params: &JoltProtocolParams) -> Vec<String> {
    let mut oracles = vec!["RdInc".to_owned(), "RamInc".to_owned()];
    if params.field_reg_d > 0 {
        oracles.push("FieldRegInc".to_owned());
    }
    oracles.extend((0..params.instruction_d).map(|index| format!("InstructionRa_{index}")));
    oracles.extend((0..params.ram_d).map(|index| format!("RamRa_{index}")));
    oracles.extend((0..params.bytecode_d).map(|index| format!("BytecodeRa_{index}")));
    oracles.extend((0..params.field_reg_d).map(|index| format!("FieldRegRa_{index}")));
    oracles
}

pub fn main_witness_oracle_attr(params: &JoltProtocolParams) -> String {
    symbol_array_attr(&main_witness_oracles(params))
}

pub fn append_foundation_ops<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "field.define",
        Some(FIELD_SYMBOL),
        &[("modulus_bits", "254 : i64"), ("role", r#""scalar""#)],
    )?;
    context.append_op(
        module,
        "hash.function",
        Some(HASH_SYMBOL),
        &[("algorithm", r#""blake2b""#)],
    )?;
    context.append_op(
        module,
        "transcript.scheme",
        Some(TRANSCRIPT_SYMBOL),
        &[("hash", "@blake2b")],
    )?;
    context.append_op(
        module,
        "pcs.scheme",
        Some(PCS_SYMBOL),
        &[("field", "@bn254_fr")],
    )?;
    context.append_op(
        module,
        "poly.domain",
        Some(TRACE_DOMAIN_SYMBOL),
        &[
            ("field", "@bn254_fr"),
            ("log_size", &format!("{} : i64", params.log_t)),
        ],
    )?;
    context.append_op(
        module,
        "poly.domain",
        Some(MAIN_WITNESS_COMMIT_DOMAIN_SYMBOL),
        &[
            ("field", "@bn254_fr"),
            (
                "log_size",
                &format!("{} : i64", params.log_t + params.log_k_chunk),
            ),
        ],
    )?;
    Ok(())
}

pub fn append_committed_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_oracle(
        context,
        module,
        OracleSpec {
            symbol: "RdInc".to_owned(),
            domain: "@jolt.trace_domain",
            commit_domain: "@jolt.main_witness_commit_domain",
            layout: "dense_trace",
            visibility: "committed",
            extra_attrs: Vec::new(),
        },
    )?;
    append_oracle(
        context,
        module,
        OracleSpec {
            symbol: "RamInc".to_owned(),
            domain: "@jolt.trace_domain",
            commit_domain: "@jolt.main_witness_commit_domain",
            layout: "dense_trace",
            visibility: "committed",
            extra_attrs: Vec::new(),
        },
    )?;
    if params.field_reg_d > 0 {
        append_oracle(
            context,
            module,
            OracleSpec {
                symbol: "FieldRegInc".to_owned(),
                domain: "@jolt.trace_domain",
                commit_domain: "@jolt.main_witness_commit_domain",
                layout: "dense_trace",
                visibility: "committed",
                extra_attrs: Vec::new(),
            },
        )?;
    }
    for index in 0..params.instruction_d {
        append_indexed_oracle(
            context,
            module,
            "InstructionRa",
            index,
            "@jolt.main_witness_commit_domain",
        )?;
    }
    for index in 0..params.ram_d {
        append_indexed_oracle(
            context,
            module,
            "RamRa",
            index,
            "@jolt.main_witness_commit_domain",
        )?;
    }
    for index in 0..params.bytecode_d {
        append_indexed_oracle(
            context,
            module,
            "BytecodeRa",
            index,
            "@jolt.main_witness_commit_domain",
        )?;
    }
    for index in 0..params.field_reg_d {
        append_indexed_oracle(
            context,
            module,
            "FieldRegRa",
            index,
            "@jolt.main_witness_commit_domain",
        )?;
    }
    append_oracle(
        context,
        module,
        OracleSpec {
            symbol: "UntrustedAdvice".to_owned(),
            domain: "@jolt.trace_domain",
            commit_domain: "@jolt.trace_domain",
            layout: "dense_trace",
            visibility: "optional_committed",
            extra_attrs: Vec::new(),
        },
    )?;
    append_oracle(
        context,
        module,
        OracleSpec {
            symbol: "TrustedAdvice".to_owned(),
            domain: "@jolt.trace_domain",
            commit_domain: "@jolt.trace_domain",
            layout: "dense_trace",
            visibility: "optional_committed",
            extra_attrs: Vec::new(),
        },
    )?;
    Ok(())
}

fn append_indexed_oracle<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    family: &str,
    index: usize,
    domain: &str,
) -> Result<(), MlirError> {
    append_oracle(
        context,
        module,
        OracleSpec {
            symbol: format!("{family}_{index}"),
            domain,
            commit_domain: "@jolt.main_witness_commit_domain",
            layout: "onehot_expanded",
            visibility: "committed",
            extra_attrs: vec![
                ("family", format!("@{family}")),
                ("index", format!("{index} : i64")),
            ],
        },
    )
}

struct OracleSpec<'a> {
    symbol: String,
    domain: &'a str,
    commit_domain: &'a str,
    layout: &'a str,
    visibility: &'a str,
    extra_attrs: Vec<(&'a str, String)>,
}

fn append_oracle<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    spec: OracleSpec<'_>,
) -> Result<(), MlirError> {
    let mut attrs = vec![
        ("field", "@bn254_fr".to_owned()),
        ("domain", spec.domain.to_owned()),
        ("commit_domain", spec.commit_domain.to_owned()),
        ("visibility", format!("\"{}\"", spec.visibility)),
        ("layout", format!("\"{}\"", spec.layout)),
    ];
    attrs.extend(spec.extra_attrs);
    context.append_op_with_owned_attrs(
        module,
        "piop.oracle",
        Some(&spec.symbol),
        &attrs
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect::<Vec<_>>(),
    )
}

fn symbol_array_attr(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}
