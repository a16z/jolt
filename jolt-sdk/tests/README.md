# Test Fixtures

These tests rely on fixtures that may break if the prover or verifier implementation changes.  
If you encounter failing tests due to fixture mismatches, you can regenerate the fixtures:

Run the update script from the jolt-verifier directory:
```bash
./gen-fixtures.sh
```

# Guest function cache regression

After installing the current Jolt CLI, run:

```sh
python3 jolt-sdk/tests/guest_function_cache.py
```

This builds the existing multi-function guest repeatedly in one Cargo target
directory. It checks switching, reselecting and unsetting `JOLT_FUNC_NAME`
without changing the source or cleaning cached artifacts. The SDK CI job runs
the same check.
