# Test Fixtures

These tests rely on fixtures that may break if the prover or verifier implementation changes.  
If you encounter failing tests due to fixture mismatches, you can regenerate the fixtures:

Run the update script from the repository root:
```bash
bash jolt-sdk/tests/gen-fixtures.sh
```