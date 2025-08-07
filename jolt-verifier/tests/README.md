# Test Fixtures

These tests rely on fixtures that may break if the prover or verifier implementation changes.  
If you encounter failing tests due to fixture mismatches, you can regenerate the fixtures:

1. Run `cargo run --release -p fibonacci -- --save` in the root dir to generate new fixtures in the `/tmp` directory.
2. Copy the new fixtures from `/tmp` and overwrite the existing ones in this directory.

This will update the test data to match the latest prover/verifier output.