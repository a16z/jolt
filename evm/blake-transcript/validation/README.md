# Complete clear verifier validation

These frozen results cover the canonical preprocessed Spartan relation and its five HyperKZG openings. They do not prove the whole Akita verifier circuit, operational ROM extraction, ZK, or a deployment-secure setup. Fixtures use a known beta=7 setup secret.

The tested implementation is the accepted counter-word revision e32387d96bac9dde343ef0087bde2e346ffe64cd. This PR preserves its production contracts, native fixture generator, tests and fixture bytes. `source-hashes.json` and `unchanged-fixtures.json` identify the tested inputs. Full diagnostic history remains retained in that implementation's evidence packets; these files retain the final observations needed to assess this change.

Pinned tools: Solidity0.8.30, optimizer200, Prague, EthereumJS10.1.0. Complete verification uses viaIR; primitive parity records its own settings. Each EVM case starts with the documented fresh state and precompile warmness. The full harness awaits journal cleanup and asserts cold module accounts. Earlier incomplete warmness measurements are not evidence for the results here.

Passed controls:123 Blake/native parity,79canonical decoding,5MODEXP failures,33public-opening/precompile cases,36full-verifier cases. The three accepted complete-verifier fixtures match their native transcript receipts. The remaining cases check mutations and diagnostic errors; changing the public input or the relevant proof claims rejects. The norm fixture also has two rejecting controls. The files contain exact outcome, return-byte and tool/configuration records, not a substitute for rerunning the tests.

The norm proof is54,547wire bytes and55,812calldata bytes. Execution is32,635,986 gas; 902,640 intrinsic gives 33,538,626 estimated transaction gas. The calldata floor is nonbinding. The initial30M call still exhausts gas; the authorized200M diagnostic accepts. Deployment is excluded; no live-chain compatibility claim follows. All five runtime contracts satisfyEIP170. Timings include compilation and are not verifier/prover performance estimates.

Run from this directory's parent:

```sh
npm ci --ignore-scripts
npm test
npm run test:spartan
npm run test:spartan-failures
npm run test:spartan-public-opening
npm run compile:spartan-modules
npm run test:spartan-complete
npm run measure:spartan-norm
```

`compile:spartan-modules` checks the frozen policy. Regeneration is an explicit reviewed action, not a test repair. The native fixture example and CI regenerate toy/empty/empty-public/zero-products references and compare their complete bytes. The norm fixture has its own committed provenance and hash guards.

The counter-word optimization reduces product execution by 23.91% and complete execution by 21.92% against the decoder-word baseline, with all 36 complete receipts unchanged. Its 123 primitive controls include 71 uint64 encoding vectors and four Blake2F failure/return-length controls. The original comparison script failed on an omitted optional exception field, and norm ran before that comparator status was checked. The corrected comparison and independent review passed; the ordering error remains recorded in the full evidence packet.
