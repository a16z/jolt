# Recorded Akita boundary fixture

Complete existing muldiv proof at trace4096/RAM8192, inputs[9,5,3], output[15].
These artifacts support typed native replay and the partial boundary regression;
they are not a wrapper proof. Exact provenance and SHA256 hashes are in
`specs/akita-wrapper/upstream-boundary-constraints.md`.

Producer Jolt80180a38ad5b137ab46a3b764bd67bab1e8ef357 plus archived export patch,
Akita28fc72021c120e7bc4e0102e07844f25299051eb. Previously accepted public-e24 replay
and current-f5 native replay are recorded in the boundary evidence packet.
Private advice is not serialized in public-io; no-advice is separately checked
through proof options and the fixed profile.

Run the regression with `cargo nextest run -p jolt-verifier --features
r1cs,akita,prover-fixtures boundary_recorded --test-threads 1`. Full circuit tests
need a separately approved resource envelope; they synthesize roughly92 Blake
compression calls. Tests do not create an outer SNARK or claim full constrained
Jolt acceptance.

`stage1-vectors.json` extends this fixture through the complete stage1 remainder.
Values were frozen by an independent Python source/proof census before the current-f5
native observer ran; all17 exported fields and228 transcript events then matched.
Observer binary SHA256:23970ed113c3e563ed65660ec47e19f776ff583ebeb5daa32b7cbacabed30090.
The JSON omits the verbose event list; its remaining fields preserve those frozen values.
These are native reference values, not a constrained stage1 acceptance result.
