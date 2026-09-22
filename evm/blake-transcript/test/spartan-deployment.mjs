// Intentional deployment-only measurement; requires compile:spartan-modules artifacts.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {keccak_256} from '@noble/hashes/sha3.js';
import {blake2b} from '@noble/hashes/blake2.js';
import {Common,Hardfork,Mainnet} from '@ethereumjs/common';
import {createEVM} from '@ethereumjs/evm';
import {resetWarmth} from './evm-warmth.mjs';
import {transactionCost} from './transaction-cost.mjs';
const build=JSON.parse(fs.readFileSync('evidence/sparse/compiled.json'));
for(const [name,source]of Object.entries(build.sources))assert.equal(fs.readFileSync('contracts/'+name,'utf8'),source.content,'exact compiled source '+name);
const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});
const word=b=>Buffer.concat([Buffer.alloc(32-b.length),b]);
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const deployments=[],addresses=[];
for(const name of [...build.names,'SpartanVerifier']){
 const artifact=build.complete.contracts[name+'.sol'][name].evm;
 const args=name==='SpartanVerifier'?Buffer.concat([digest(fs.readFileSync('test/spartan/norm/key.bin')),digest(fs.readFileSync('test/spartan/norm/setup.bin')),...addresses.map(a=>word(a.bytes))]):Buffer.alloc(0);
 const initcode=Buffer.concat([Buffer.from(artifact.bytecode.object,'hex'),args]);
 await resetWarmth(evm,addresses,{precompiles:false});
 const result=await evm.runCall({data:initcode,gasLimit:30_000_000n});
 assert.equal(result.execResult.exceptionError,undefined,name);
 const runtime=await evm.stateManager.getCode(result.createdAddress);
 assert.equal(runtime.length,artifact.deployedBytecode.object.length/2);
 assert(runtime.length<=24576);
 const runtimeKeccak=Buffer.from(keccak_256(runtime)).toString('hex');
 if(name!=='SpartanVerifier')assert.equal(runtimeKeccak,build.hashes[addresses.length]);
 deployments.push({name,address:result.createdAddress.toString(),runtimeBytes:runtime.length,runtimeKeccak,executionGas:result.execResult.executionGasUsed.toString(),transaction:transactionCost(initcode,result.execResult.executionGasUsed,true)});
 addresses.push(result.createdAddress);
}
fs.mkdirSync('evidence/deployment',{recursive:true});
const result={scope:'standalone cold-module CREATE/code-deposit execution; transaction charges calculated separately',warmness:'journal cleared before every CREATE; module addresses cold at coordinator constructor',node:process.version,settings:build.settings,sourceHashes:Object.fromEntries(Object.entries(build.sources).map(([name,v])=>[name,crypto.createHash('sha256').update(v.content).digest('hex')])),deployments};
fs.writeFileSync('evidence/deployment/results.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
