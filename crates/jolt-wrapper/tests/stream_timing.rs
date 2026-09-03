#![expect(
    clippy::expect_used,
    reason = "the ignored benchmark fails immediately when its subprocess cannot run"
)]

use std::path::Path;
use std::process::Command;

#[test]
#[ignore = "2^17 benchmark: about 5 seconds after the release binary is built"]
fn n3_g_shape_timing() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "run",
            "--release",
            "-q",
            "--message-format=short",
            "-p",
            "jolt-wrapper-bench",
            "--",
            "batched",
            "s=3",
            "k=8",
            "col",
            "17",
        ])
        .output()
        .expect("run N3 benchmark");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("= 11936;"), "{stdout}");
}
