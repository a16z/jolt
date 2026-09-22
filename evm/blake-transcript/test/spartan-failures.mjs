// Fault injection against the frozen compiled runtime; synthetic callees are not gas oracles.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {blake2b} from '@noble/hashes/blake2.js';
import {Common,Mainnet,Hardfork} from '@ethereumjs/common';
import {createEVM,EVMError} from '@ethereumjs/evm';
const hash=b=>Buffer.from(blake2b(b,{dkLen:32}));
const dir='test/spartan/toy/';const key=fs.readFileSync(dir+'key.bin'),setup=fs.readFileSync(dir+'setup.bin');
const code=Buffer.from(fs.readFileSync('evidence/spartan/runtime.hex','utf8').trim(),'hex');
const {refs,names}=JSON.parse(fs.readFileSync('evidence/spartan/immutable-references.json'));
for(const [id,locations]of Object.entries(refs))for(const p of locations)(names[id]==='expectedKey'?hash(key):hash(setup)).copy(code,p.start);
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const args=[key,setup,fs.readFileSync(dir+'inputs.bin'),fs.readFileSync(dir+'proof.bin')];
let at=128;const head=[],tail=[];for(const b of args){head.push(word(at));const x=Buffer.concat([word(b.length),b,Buffer.alloc((32-b.length%32)%32)]);tail.push(x);at+=x.length;}
// Solidity selector recorded from solc in the normal runner, not a protocol hash.
const selector=Buffer.from(JSON.parse(fs.readFileSync('evidence/spartan/selector.json')).selector,'hex');
const data=Buffer.concat([selector,...head,...tail]);const records=[];
for(const [name,length,failure]of [['empty',0,false],['short',31,false],['long',33,false],['failure',32,true],['wrong-root',32,false]]){
 const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});
 for(const addr of [5,9])evm.journal.addAlwaysWarmAddress(addr.toString(16).padStart(40,'0'));
 let calls=0;evm.precompiles.set('0000000000000000000000000000000000000005',input=>{++calls;assert.equal(input.data.length,192);return {executionGasUsed:0n,returnValue:new Uint8Array(length),...(failure?{exceptionError:new EVMError(EVMError.errorMessages.REVERT)}:{})};});
 const result=await evm.runCode({code,data,gasLimit:30_000_000n});assert.equal(result.exceptionError?.error,'revert');assert.equal(calls,1);assert.equal(result.returnValue.length,4);
 records.push({name,calls,exception:result.exceptionError.error,returnHex:Buffer.from(result.returnValue).toString('hex')});
}
assert.equal(new Set(records.slice(0,4).map(r=>r.returnHex)).size,1);assert.notEqual(records[4].returnHex,records[0].returnHex);
console.log(JSON.stringify({scope:'MODEXP fault injection, not cryptographic or gas oracle',records},null,2));
