import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import solc from 'solc';
import { Common, Mainnet, Hardfork } from '@ethereumjs/common';
import { createEVM } from '@ethereumjs/evm';
const source = fs.readFileSync(process.argv[2] ?? 'contracts/BlakeTranscript.sol', 'utf8');
const settings = { optimizer: { enabled: true, runs: 200 }, evmVersion: 'prague', outputSelection: { '*': { '*': ['evm.deployedBytecode.object', 'evm.deployedBytecode.sourceMap'] } } };
const compiled = JSON.parse(solc.compile(JSON.stringify({ language: 'Solidity', sources: { 'BlakeTranscript.sol': { content: source } }, settings })));
for (const error of compiled.errors ?? []) if (error.severity === 'error') throw new Error(error.formattedMessage);
const artifact = compiled.contracts['BlakeTranscript.sol'].BlakeTranscriptHarness.evm.deployedBytecode;
const runtime = Buffer.from(artifact.object, 'hex');
const mappings = artifact.sourceMap.split(';');
const locations = new Map();
let previous = ['0', '0', '-1'];
for (let pc = 0, index = 0; pc < runtime.length && index < mappings.length; index++) {
  const parts = mappings[index].split(':');
  previous = previous.map((value, i) => parts[i] === undefined || parts[i] === '' ? value : parts[i]);
  const [start, length, file] = previous.map(Number);
  locations.set(pc, { start, length, file, line: file === 0 ? source.slice(0, start).split('\n').length : 0 });
  const opcode = runtime[pc];
  pc += 1 + (opcode >= 0x60 && opcode <= 0x7f ? opcode - 0x5f : 0);
}
const vectors = JSON.parse(fs.readFileSync('test/native-vectors.json'));
const u32 = n => { const out = Buffer.alloc(4); out.writeUInt32BE(n); return out; };
function encode(v) {
  const label = Buffer.from(v.label, 'hex');
  const chunks = [Buffer.from([2, label.length]), label];
  for (const op of v.ops) {
    chunks.push(Buffer.from([{ absorb: 1, peek: 4, challenge: 5 }[op.kind]]));
    if (op.kind === 'absorb') { const data = Buffer.from(op.data, 'hex'); chunks.push(u32(data.length), data); }
  }
  return Buffer.concat(chunks);
}
const cases = [128, 1024].map(n => { const v = vectors.hashes.find(v => v.data.length === n * 2); return { name: 'hash-' + n, data: Buffer.concat([Buffer.from([0]), Buffer.from(v.data, 'hex')]), expected: v.expected }; });
for (const name of ['framing-joined', 'wide-' + Buffer.from('spartan-preprocessed-clear-v2').toString('hex')]) {
  const v = vectors.cases.find(v => v.name === name);
  cases.push({ name, data: encode(v), expected: v.ops.filter(op => op.expected !== null).map(op => op.expected).join('') });
}
const records = [];
for (const v of cases) {
  const evm = await createEVM({ common: new Common({ chain: Mainnet, hardfork: Hardfork.Prague }) });
  evm.journal.addAlwaysWarmAddress('0000000000000000000000000000000000000009');
  const opcodes = {}, lines = {};
  let last, total = 0n;
  function charge(nextGas) {
    if (!last) return;
    const gas = last.gasLeft - nextGas;
    assert(gas >= 0n, 'unexpected gas increase');
    total += gas;
    for (const [map, key] of [[opcodes, last.opcode.name], [lines, locations.get(last.pc)?.line ?? 0]]) {
      const bucket = map[key] ??= { count: 0, gas: 0n };
      bucket.count++;
      bucket.gas += gas;
    }
  }
  evm.events.on('step', step => { assert.equal(step.depth, 0, 'only precompile callees expected'); charge(step.gasLeft); last = step; });
  const gasLimit = 30_000_000n;
  const result = await evm.runCode({ code: runtime, data: v.data, gasLimit });
  charge(gasLimit - result.executionGasUsed);
  assert.equal(total, result.executionGasUsed, 'complete gas attribution');
  assert.equal(result.exceptionError, undefined);
  assert.equal(Buffer.from(result.returnValue).toString('hex'), v.expected);
  const calls = opcodes.STATICCALL?.count ?? 0;
  records.push({ name: v.name, executionGas: total, compressionCalls: calls, blake2fInternalGas: calls * 12, opcodes, sourceLines: lines });
}
console.log(JSON.stringify({ sourceSha256: crypto.createHash('sha256').update(source).digest('hex'), runtimeSha256: crypto.createHash('sha256').update(runtime).digest('hex'), runtimeBytes: runtime.length, settings, hardfork: 'Prague', boundary: 'fresh EVM, warm0x09, 30M gas; exact same bytecode execution boundary as parity runner', sourceLineNote: 'compiler source mapping, shared generated guards may map to broad expressions; opcode attribution is exact', records }, (_, value) => typeof value === 'bigint' ? value.toString() : value, 2));
