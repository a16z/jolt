import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import solc from 'solc';
import { Common, Hardfork, Mainnet } from '@ethereumjs/common';
import { createEVM, EVMError } from '@ethereumjs/evm';

const source = fs.readFileSync('contracts/BlakeTranscript.sol', 'utf8');
const settings = { optimizer: { enabled: true, runs: 200 }, evmVersion: 'prague', outputSelection: { '*': { '*': ['evm.deployedBytecode.object'] } } };
const helperSource = `pragma solidity 0.8.30;
import {reverse64} from "./BlakeTranscript.sol";
contract U64Harness {
    fallback(bytes calldata input) external returns (bytes memory) {
        require(input.length == 8);
        return abi.encodePacked(bytes8(reverse64(uint64(bytes8(input)))));
    }
}`;
const compiled = JSON.parse(solc.compile(JSON.stringify({ language: 'Solidity', sources: { 'BlakeTranscript.sol': { content: source }, 'U64Harness.sol': { content: helperSource } }, settings })));
for (const error of compiled.errors ?? []) if (error.severity === 'error') throw new Error(error.formattedMessage);
const runtime = Buffer.from(compiled.contracts['BlakeTranscript.sol'].BlakeTranscriptHarness.evm.deployedBytecode.object, 'hex');
const vectors = JSON.parse(fs.readFileSync('test/native-vectors.json'));
const records = [];
const gasLimit = 30_000_000n;
const u32 = n => { const out = Buffer.alloc(4); out.writeUInt32BE(n); return out; };
const opcode = { absorb: 1, squeeze: 2, ratchet: 3, peek: 4, challenge: 5 };
function encode(v) {
  const label = Buffer.from(v.label, 'hex');
  const chunks = v.mode === 'wide' ? [Buffer.from([2, label.length]), label] : [Buffer.from([1])];
  for (const op of v.ops) {
    chunks.push(Buffer.from([opcode[op.kind]]));
    if (op.kind === 'absorb') { const bytes = Buffer.from(op.data, 'hex'); chunks.push(u32(bytes.length), bytes); }
    if (op.kind === 'squeeze') chunks.push(u32(op.length));
  }
  return Buffer.concat(chunks);
}
async function run(name, data, expected, rejection = false, { code = runtime, fault = null } = {}) {
  const evm = await createEVM({ common: new Common({ chain: Mainnet, hardfork: Hardfork.Prague }) });
  // EIP-2929 warms precompiles at transaction start; runCode omits that wrapper.
  evm.journal.addAlwaysWarmAddress('0000000000000000000000000000000000000009');
  let faultCalls = 0;
  if (fault !== null) evm.precompiles.set('0000000000000000000000000000000000000009', input => {
    ++faultCalls;
    assert.equal(input.data.length, 213);
    return { executionGasUsed: 0n, returnValue: new Uint8Array(fault.length), ...(fault.fail ? { exceptionError: new EVMError(EVMError.errorMessages.REVERT) } : {}) };
  });
  const result = await evm.runCode({ code, data, gasLimit });
  if (fault !== null) assert.equal(faultCalls, 1, name + ' must exercise the production Blake2F call');
  if (rejection) assert.equal(result.exceptionError?.error, 'revert', name + ' must explicitly revert');
  else { assert.equal(result.exceptionError, undefined, name + ': ' + result.exceptionError); assert.equal(Buffer.from(result.returnValue).toString('hex'), expected, name); }
  records.push({ name, accepted: !result.exceptionError, executionGas: result.executionGasUsed.toString(), calldataBytes: data.length, returnHex: Buffer.from(result.returnValue).toString('hex'), exception: result.exceptionError?.error ?? null });
}
const helperRuntime = Buffer.from(compiled.contracts['U64Harness.sol'].U64Harness.evm.deployedBytecode.object, 'hex');
const u64Values = new Set([0n, (1n << 64n)-1n, 127n, 128n, 129n, 255n, 256n, (1n << 32n)-1n, 1n << 32n, (1n << 32n)+1n]);
for (let bit = 0n; bit < 64n; ++bit) u64Values.add(1n << bit);
for (const value of u64Values) {
  const input = Buffer.alloc(8); input.writeBigUInt64BE(value);
  const expected = Buffer.alloc(8); expected.writeBigUInt64LE(value);
  await run('u64-' + value.toString(16), input, expected.toString('hex'), false, { code: helperRuntime });
}
for (const [name, length, fail] of [['failure', 64, true], ['empty', 0, false], ['short', 63, false], ['oversized', 65, false]]) {
  await run('blake2f-' + name, Buffer.from([0, 1]), null, true, { fault: { length, fail } });
}
// Independent RFC/EIP ground truth before native transcript comparisons.
await run('eip152-abc', Buffer.concat([Buffer.from([0]), Buffer.from('abc')]), 'ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923');
const scalarModulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
for (const value of [0n, 1n, (1n << 384n)-1n, scalarModulus, scalarModulus+1n, (1n << 256n)+(2n << 128n)+3n]) {
  await run('reduce384-'+value.toString(16), Buffer.concat([Buffer.from([3]), Buffer.from(value.toString(16).padStart(96,'0'),'hex')]), (value % scalarModulus).toString(16).padStart(64,'0'));
}
for (const v of vectors.hashes) await run('hash-' + v.data.length / 2, Buffer.concat([Buffer.from([0]), Buffer.from(v.data, 'hex')]), v.expected);
for (const v of vectors.cases) await run(v.name, encode(v), v.ops.filter(op => op.expected !== null).map(op => op.expected).join(''));
for (const [name, data] of [
  ['missing-mode', []], ['unknown-mode', [9]], ['truncated-absorb-length', [1,1,0]],
  ['truncated-absorb-body', [1,1,0,0,0,2,7]], ['unknown-opcode', [1,255]],
  ['label-too-long', [2,33,...Array(33).fill(1)]], ['truncated-label', [2,3,1]],
  ['raw-field-challenge', [1,5]], ['short-reduction', [3,...Array(47).fill(0)]], ['long-reduction', [3,...Array(49).fill(0)]], ['wide-raw-squeeze', [2,0,2,0,0,0,1]],
]) await run(name, Buffer.from(data), null, true);
const byName = Object.fromEntries(vectors.cases.map(v => [v.name,v]));
assert.notEqual(byName['framing-split'].ops.at(-1).expected, byName['framing-joined'].ops.at(-1).expected);
const wideCases = vectors.cases.filter(v => v.name.startsWith('wide-'));
assert.equal(new Set(wideCases.map(v => v.ops.at(-1).expected)).size, wideCases.length, 'application labels must separate challenges');
const report = { boundary: 'Prague bytecode call execution with EIP-2929-warm Blake2F precompile and fresh context per case; excludes transaction intrinsic gas, deployment, and complete Spartan verification', node: process.version, solc: solc.version(), ethereumjs: '10.1.0', hardfork: 'Prague', chain: 'Mainnet', gasLimit: gasLimit.toString(), prewarmedAddresses: ['0x0000000000000000000000000000000000000009'], context: 'fresh EVM per case', evmVersion: settings.evmVersion, optimizer: settings.optimizer, runtimeBytes: runtime.length, runtimeSha256: crypto.createHash('sha256').update(runtime).digest('hex'), sourceSha256: crypto.createHash('sha256').update(source).digest('hex'), helperSourceSha256: crypto.createHash('sha256').update(helperSource).digest('hex'), helperRuntimeSha256: crypto.createHash('sha256').update(helperRuntime).digest('hex'), nativeVectorsSha256: crypto.createHash('sha256').update(fs.readFileSync('test/native-vectors.json')).digest('hex'), records };
fs.mkdirSync('evidence/generated', { recursive: true });
fs.writeFileSync('evidence/generated/runtime.hex', runtime.toString('hex')+'\n');
fs.writeFileSync('evidence/generated/evm-results.json', JSON.stringify(report, null, 2)+'\n');
console.log(JSON.stringify(report, null, 2));
