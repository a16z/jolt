# Recorded Akita first-stage fixture

Existing complete muldiv proof: trace4096, RAM8192, input bytes[9,5,3], output[15].
These artifacts support native replay and the constrained preamble plus complete
first-stage partial relation. Later Jolt stages and PCS authentication are absent
from that circuit; these files are not a BN254 wrapper proof.

The fixture was produced by Jolt `80180a38ad5b137ab46a3b764bd67bab1e8ef357`
plus an archived diagnostic export patch, with Akita
`28fc72021c120e7bc4e0102e07844f25299051eb`. The export patch is not included in
this fixture directory; the hashes below identify the retained artifacts rather
than promise a reproducible producer command. Later native replay accepted the
same files with Akita `f5f75335eae18241681fd24ca0a60fca8f0512af`.

| File | SHA256 |
|---|---|
| proof.bin | ea5ade69da3b297ae3fa6b9756236f6cf8fcf4d041d9c2d3ab74b7064eaa9ad4 |
| preprocessing.bin | a2b786212156fa59982a8bf7eec13762459b1ee53b1bbfba5121c9f372bafc15 |
| public-io.bin | 5211e8ce9ddb7d3a620996a942ebc7161f6a6c9a253adb06b67d0f7b690b1810 |

Private advice is not serialized in public-io; no-advice is separately checked
through proof options and the fixed profile. The profile uses ordinary q128 and
LegacyBlake2b, with no field-inline dependency.

Run the recorded regressions serially:

```sh
cargo nextest run --locked -p jolt-verifier --features r1cs,akita,prover-fixtures \
  -E 'test(r1cs::boundary::) | test(r1cs::remainder::)' --test-threads 1 --cargo-quiet
cargo run --locked -p jolt-spartan-prover --features akita-boundary \
  --example akita_boundary_relation -- crates/jolt-verifier/tests/fixtures/akita-boundary
```

Complete first-stage tests synthesize230 Blake compressions; the separate boundary
uses92. Recorded local full-stage controls sampled up to8.8 GB process-session RSS.
The application checks the public partition and matrix satisfaction; it does not
produce an outer SNARK. The accepted integrated source measured12,054,769 rows,
11,961,755 variables,65,964,357 nonzeros and2 public columns. The focused PR
projection requires its own checks; source equality alone does not transfer those
runtime observations.

`stage1-vectors.json` contains native values frozen by a Python source/proof census,
then matched by a native observer in all17 exported fields and228 transcript
events. Observer binary SHA256:
`23970ed113c3e563ed65660ec47e19f776ff583ebeb5daa32b7cbacabed30090`.
The JSON omits the verbose event list. Its values are reference data, not by
themselves evidence that the constrained relation accepts.
