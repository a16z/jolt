//! Native timing and compatibility audit against an immutable saved outer bundle.
use akita_types::setup_claims::SetupBatchStatement;
use anyhow::{anyhow, ensure, Result};
use jolt_akita::deferred::SetupMode;
use jolt_akita::{AkitaField, AkitaScheme, AkitaVerifierResources};
use jolt_akita_external::pair_api::{ProofPair, Vc};
use jolt_akita_external::{ExternalInput, OuterAdvice, PairAdvice, Tape};
use jolt_field::Ring;
use jolt_transcript::{Blake2b512, VerifierState};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 4 && args[1] == "--import-json" {
        let input: ExternalInput = serde_json::from_slice(&fs::read(&args[2])?)?;
        let _resources =
            AkitaVerifierResources::prepare([Ok(input.preprocessing.pcs_setup.clone())])?;
        let checked = input
            .replay(SetupMode::Check, &mut Tape(input.answers.iter()), |_| {})
            .map_err(|e| anyhow!(e))?;
        ensure!(checked.claims().is_empty(), "unresolved imported claims");
        fs::write(
            &args[3],
            bincode::serde::encode_to_vec(&input, bincode::config::standard())?,
        )?;
        return Ok(());
    }
    if args.len() == 4 && args[1] == "--prepare-pair" {
        let pair =
            jolt_akita_external::decode_exact(&fs::read(&args[2])?).map_err(|e| anyhow!(e))?;
        prepare_pair(pair, &PathBuf::from(&args[3]))?;
        return Ok(());
    }
    if args.len() == 4 && args[1] == "--prepare" {
        let input = jolt_akita_external::decode_exact::<ExternalInput>(&fs::read(&args[2])?)
            .map_err(|e| anyhow!(e))?;
        prepare(input, &PathBuf::from(&args[3]))?;
        return Ok(());
    }
    ensure!(
        args.len() == 4,
        "usage: outer-verifier-audit EXTERNAL_INPUT EXPECTED_GUEST_OUTPUT REPORT"
    );
    let path = PathBuf::from(&args[1]);
    let mut input: ExternalInput =
        jolt_akita_external::decode_exact::<ExternalInput>(&fs::read(path)?)
            .map_err(|e| anyhow!(e))?;
    let output = fs::read(PathBuf::from(&args[2]))?;
    let (expected, tail): (Vec<u8>, &[u8]) = postcard::take_from_bytes(&output)?;
    ensure!(tail.iter().all(|b| *b == 0), "nonzero output padding");
    let start = Instant::now();
    let _resources = AkitaVerifierResources::prepare([Ok(input.preprocessing.pcs_setup.clone())])?;
    let preparation = start.elapsed().as_secs_f64();
    ensure!(
        jolt_verifier::verify::<AkitaField, AkitaScheme, Vc, VerifierState<Blake2b512>>(
            &input.preprocessing,
            &input.outer.io,
            &input.outer.proof,
            None,
        )
        .is_err(),
        "default Check accepted an accumulated proof without its inherited statement"
    );
    // Empty inherited statements remain valid transport; attaching a nonempty
    // batch for proof verification is a separate policy boundary.
    let empty = SetupBatchStatement::<AkitaField>::new(
        input
            .inherited()
            .map_err(|e| anyhow!(e))?
            .polynomial()
            .clone(),
        2,
        Vec::new(),
    )?;
    let encoded = postcard::to_stdvec(&empty.descriptor_bytes::<AkitaField>()?)?;
    let original = std::mem::replace(&mut input.outer.io.outputs, encoded);
    ensure!(
        input
            .inherited()
            .map_err(|e| anyhow!(e))?
            .claims()
            .is_empty(),
        "empty inherited statement rejected"
    );
    input.outer.io.outputs = original;
    let mut full = Vec::new();
    let mut deferred = Vec::new();
    for _ in 0..10 {
        let mut tape = Tape(input.answers.iter());
        let t = Instant::now();
        let checked = input
            .replay(SetupMode::Check, &mut tape, |_| {})
            .map_err(|e| anyhow!(e))?;
        full.push(t.elapsed().as_secs_f64());
        ensure!(checked.claims().is_empty(), "unresolved claims in Check");
        let mut tape = Tape(input.answers.iter());
        let t = Instant::now();
        let pending = input
            .replay(SetupMode::Defer, &mut tape, |_| {})
            .map_err(|e| anyhow!(e))?;
        deferred.push(t.elapsed().as_secs_f64());
        ensure!(tape.0.next().is_none(), "unused advice");
        ensure!(
            pending.claims().len() == expected_pending(&input)?,
            "changed remaining count"
        );
        ensure!(
            pending.descriptor_bytes::<AkitaField>()? == expected,
            "changed canonical pending statement"
        );
    }
    let bytes = bincode::serde::encode_to_vec(&input, bincode::config::standard())?;
    ensure!(
        !input.answers.is_empty(),
        "audit fixture must have direct-fold answers"
    );
    let mut wrong =
        jolt_akita_external::decode_exact::<ExternalInput>(&bytes).map_err(|e| anyhow!(e))?;
    wrong.answers[0] += AkitaField::from_u64(1);
    ensure!(
        wrong
            .replay(SetupMode::Defer, &mut Tape(wrong.answers.iter()), |_| {})
            .is_err(),
        "false candidate accepted"
    );
    let mut wrong_io =
        jolt_akita_external::decode_exact::<ExternalInput>(&bytes).map_err(|e| anyhow!(e))?;
    wrong_io.outer.io.outputs[0] ^= 1;
    ensure!(
        wrong_io
            .replay(SetupMode::Check, &mut Tape(wrong_io.answers.iter()), |_| {})
            .is_err(),
        "altered public output accepted"
    );
    let median = |values: &[f64]| {
        let mut v = values[2..].to_vec();
        v.sort_by(f64::total_cmp);
        (v[3] + v[4]) / 2.
    };
    let result = serde_json::json!({"prepare_seconds":preparation,"warmup":2,"measured":8,"full_seconds":full,"deferred_seconds":deferred,"full_median_seconds":median(&full),"deferred_median_seconds":median(&deferred),"pending_claims":expected_pending(&input)?,"full_unresolved":0,"empty_inherited_statement_decodes":true,"default_check_rejects_missing_inherited_batch":true,"canonical_output_identical":true,"corrupt_advice_rejected":true,"altered_public_output_rejected":true});
    fs::write(PathBuf::from(&args[3]), serde_json::to_vec_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}

fn expected_pending(input: &ExternalInput) -> Result<usize> {
    let setup = &input.preprocessing.pcs_setup;
    let catalog = setup.one_hot_catalog()?;
    let row = catalog.resolve_selection(input.outer.proof.joint_opening_proof.selection())?;
    let nonterminal = row
        .schedule()
        .recursive_folds
        .len()
        .checked_add(1)
        .ok_or_else(|| anyhow!("fold count overflow"))?;
    let offloads = setup.setup_offloads(&input.outer.proof.joint_opening_proof)?;
    nonterminal
        .checked_sub(offloads.len())
        .ok_or_else(|| anyhow!("offloading count exceeds schedule"))
}
fn prepare(input: ExternalInput, dir: &Path) -> Result<()> {
    ensure!(!dir.exists(), "choose a fresh preparation directory");
    let _resources = AkitaVerifierResources::prepare([Ok(input.preprocessing.pcs_setup.clone())])?;
    let mut tape = Tape(input.answers.iter());
    let statement = input
        .replay(SetupMode::Defer, &mut tape, |_| {})
        .map_err(|e| anyhow!(e))?;
    let count = expected_pending(&input)?;
    ensure!(
        statement.claims().len() == count && tape.0.next().is_none(),
        "schedule/accumulator mismatch"
    );
    let (key, terminal) = input
        .preprocessing
        .pcs_setup
        .prepare_recursion_material(&input.outer.proof.joint_opening_proof)?;
    fs::create_dir_all(dir)?;
    fs::write(dir.join("key.bin"), key)?;
    fs::write(dir.join("terminal.bin"), terminal)?;
    fs::write(
        dir.join("verifier.bin"),
        bincode::serde::encode_to_vec(&input.preprocessing, bincode::config::standard())?,
    )?;
    fs::write(
        dir.join("config.rs"),
        format!(
            "const INHERITED_FIELDS:usize={};\nconst EXPECTED_PENDING:usize={count};\n",
            input.inherited_setup_fields
        ),
    )?;
    fs::write(
        dir.join("expected-output.bin"),
        postcard::to_stdvec(&statement.descriptor_bytes::<AkitaField>()?)?,
    )?;
    let schedule = input.preprocessing.pcs_setup.one_hot_catalog()?;
    let row = schedule.resolve_selection(input.outer.proof.joint_opening_proof.selection())?;
    fs::write(
        dir.join("schedule.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"selected_row_digest":input.outer.proof.joint_opening_proof.selection().row_digest.as_bytes(),"nonterminal_folds":1+row.schedule().recursive_folds.len(),"direct_folds":count,"inherited_claims":input.inherited().map_err(|e|anyhow!(e))?.claims().len()}),
        )?,
    )?;
    let outer = OuterAdvice {
        outer: input.outer,
        output_setup_fields: input.output_setup_fields,
        answers: input.answers.clone(),
    };
    let advice = bincode::serde::encode_to_vec(&outer, bincode::config::standard())?;
    fs::write(dir.join("advice.bin"), postcard::to_stdvec(&advice)?)?;
    fs::write(dir.join("input.bin"), [])?;
    Ok(())
}

fn prepare_pair(pair: ProofPair, dir: &Path) -> Result<()> {
    use jolt_akita_external::pair_api::AnswerTape;
    ensure!(!dir.exists(), "choose a fresh preparation directory");
    let _resources = AkitaVerifierResources::prepare([Ok(pair.preprocessing.pcs_setup.clone())])?;
    let mut tape = AnswerTape(pair.answers.iter());
    let statement = pair
        .replay(SetupMode::Defer, &mut tape, |_| {})
        .map_err(|e| anyhow!(e))?;
    ensure!(tape.0.next().is_none(), "unused pair advice");
    let catalog = pair.preprocessing.pcs_setup.one_hot_catalog()?;
    let mut expected = 0usize;
    for inner in &pair.inner {
        let row = catalog.resolve_selection(inner.proof.joint_opening_proof.selection())?;
        let nonterminal = row
            .schedule()
            .recursive_folds
            .len()
            .checked_add(1)
            .ok_or_else(|| anyhow!("fold count overflow"))?;
        let offloaded = pair
            .preprocessing
            .pcs_setup
            .setup_offloads(&inner.proof.joint_opening_proof)?
            .len();
        expected = expected
            .checked_add(
                nonterminal
                    .checked_sub(offloaded)
                    .ok_or_else(|| anyhow!("offload count"))?,
            )
            .ok_or_else(|| anyhow!("pending count overflow"))?;
    }
    ensure!(
        statement.claims().len() == expected,
        "pair schedule/accumulator mismatch"
    );
    let (key, terminal) = pair
        .preprocessing
        .pcs_setup
        .prepare_recursion_material(&pair.inner[0].proof.joint_opening_proof)?;
    fs::create_dir_all(dir)?;
    fs::write(dir.join("key.bin"), key)?;
    fs::write(dir.join("terminal.bin"), terminal)?;
    fs::write(
        dir.join("verifier.bin"),
        bincode::serde::encode_to_vec(&pair.preprocessing, bincode::config::standard())?,
    )?;
    fs::write(
        dir.join("config.rs"),
        format!("const INHERITED_FIELDS:usize=0;\nconst EXPECTED_PENDING:usize={expected};\n"),
    )?;
    fs::write(
        dir.join("expected-output.bin"),
        postcard::to_stdvec(&statement.descriptor_bytes::<AkitaField>()?)?,
    )?;
    let advice = PairAdvice {
        inner: pair.inner,
        output_setup_fields: pair.setup_field_len,
        answers: pair.answers,
    };
    fs::write(
        dir.join("advice.bin"),
        postcard::to_stdvec(&bincode::serde::encode_to_vec(
            &advice,
            bincode::config::standard(),
        )?)?,
    )?;
    fs::write(dir.join("input.bin"), [])?;
    Ok(())
}
