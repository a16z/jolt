use spinners::{Spinner, Spinners};
use tracing::info;

macro_rules! step {
    ($msg:expr, $action:expr) => {{
        let mut sp = Spinner::new(Spinners::Dots9, $msg.to_string());
        let result = $action;
        sp.stop_with_message(format!("✓ {}", $msg));
        result
    }};
}

pub fn btreemap() {
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = step!("Compiling guest code", {
        guest::compile_btreemap(target_dir)
    });

    let prover_preprocessing = step!("Preprocessing prover", {
        guest::preprocess_btreemap(&mut program)
    });

    let verifier_preprocessing = step!("Preprocessing verifier", {
        guest::verifier_preprocessing_from_prover_btreemap(&prover_preprocessing)
    });

    let prove = step!("Building prover", {
        guest::build_prover_btreemap(program, prover_preprocessing)
    });

    let verify = step!("Building verifier", {
        guest::build_verifier_btreemap(verifier_preprocessing)
    });

    let n = 50;
    let (output, proof, io_device) = step!("Proving", { prove(n) });
    assert!(output >= 1);

    let is_valid = step!("Verifying", { verify(n, output, io_device.panic, proof) });
    assert!(is_valid);
}

fn main() {
    tracing_subscriber::fmt::init();

    info!("BTreeMap");
    btreemap();
}
