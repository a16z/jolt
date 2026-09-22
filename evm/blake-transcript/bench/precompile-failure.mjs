import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { Common, Mainnet, Hardfork } from '@ethereumjs/common';
import { createEVM, EVMError } from '@ethereumjs/evm';

// Supplemental fault injection, separate from the frozen native parity corpus.
const baseline = fs.readFileSync(process.argv[2]);
const candidate = Buffer.from(fs.readFileSync('evidence/generated/runtime.hex', 'utf8').trim(), 'hex');
const address = '0000000000000000000000000000000000000009';
const records = [];
for (const [name, size, failure] of [['empty-return', 0, false], ['short-return', 63, false], ['long-return', 65, false], ['call-failure', 64, true]]) {
  const pair = [];
  for (const [implementation, code] of [['baseline', baseline], ['candidate', candidate]]) {
    const evm = await createEVM({ common: new Common({ chain: Mainnet, hardfork: Hardfork.Prague }) });
    evm.journal.addAlwaysWarmAddress(address);
    let calls = 0;
    evm.precompiles.set(address, input => {
      calls++;
      assert.equal(input.data.length, 213);
      assert.equal(Buffer.from(input.data.subarray(0, 4)).toString('hex'), '0000000c');
      assert.equal(input.data[212], 1);
      return { executionGasUsed: 12n, returnValue: new Uint8Array(size), ...(failure ? { exceptionError: new EVMError(EVMError.errorMessages.REVERT) } : {}) };
    });
    const result = await evm.runCode({ code, data: Buffer.from([0, 97, 98, 99]), gasLimit: 30_000_000n });
    assert.equal(calls, 1);
    assert.equal(result.exceptionError?.error, 'revert');
    const data = Buffer.from(result.returnValue).toString('hex');
    assert.equal(data.length, 8, 'CompressionFailed custom error selector only');
    pair.push(data);
    records.push({ name, implementation, exception: result.exceptionError.error, returnHex: data });
  }
  assert.equal(pair[0], pair[1], 'failure behavior must remain exact');
}
console.log(JSON.stringify({ note: 'fault injection only; these synthetic callees are not gas measurements or cryptographic oracles', baselineRuntimeSha256: crypto.createHash('sha256').update(baseline).digest('hex'), candidateRuntimeSha256: crypto.createHash('sha256').update(candidate).digest('hex'), records }, null, 2));
