# jolt-akita

The Akita adapter uses Solinas fields. Its normal, build, and test dependencies
do not select BN254 or Arkworks. Tests use Blake2b with `AkitaField`.

Check the dependency boundary with `scripts/check-akita-dependencies.sh`.
Cargo features are additive: another crate in a larger application can still
select BN254 on a shared dependency.

The Akita/Dory comparison benchmark lives in `jolt-dory`, which owns the BN254
baseline. Run it with:

```sh
cargo bench -p jolt-dory --bench akita_paths
```

For its profiling configuration, add `--features profiling`.
