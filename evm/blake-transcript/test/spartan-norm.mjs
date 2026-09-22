import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {blake2b} from '@noble/hashes/blake2.js';
import {keccak_256} from '@noble/hashes/sha3.js';
import {Common,Hardfork,Mainnet} from '@ethereumjs/common';
import {createEVM} from '@ethereumjs/evm';
import {compileModules} from './compile-modules.mjs';
import {resetWarmth} from './evm-warmth.mjs';
import {transactionCost} from './transaction-cost.mjs';
const build=compileModules();
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const at=(b,n)=>BigInt('0x'+b.subarray(n,n+32).toString('hex'));
const artifact=name=>build.complete.contracts[name+'.sol'][name];
const coordinator=artifact('SpartanVerifier');
const selector=Buffer.from(coordinator.evm.methodIdentifiers['verify(bytes,bytes,bytes,bytes,uint256[4][2])'],'hex');
function fixture(name){const base='test/spartan/'+name;return Object.fromEntries(['key','setup','inputs','proof','g2-affine'].map(k=>[k,fs.readFileSync(base+'/'+k+'.bin')]));}
function calldata(f){let offset=384;const heads=[],tails=[];for(const arg of [f.key,f.setup,f.inputs,f.proof]){heads.push(word(offset));const tail=Buffer.concat([word(arg.length),arg,Buffer.alloc((32-arg.length%32)%32)]);tails.push(tail);offset+=tail.length;}return Buffer.concat([selector,...heads,f['g2-affine'],...tails]);}
const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});
let gasLimit=30_000_000n;const deployments=[];const modules=[];
async function deploy(name,args=Buffer.alloc(0)){const initcode=Buffer.concat([Buffer.from(artifact(name).evm.bytecode.object,'hex'),args]);const result=await evm.runCall({data:initcode,gasLimit});deployments.push({name,executionGas:result.execResult.executionGasUsed.toString(),transaction:transactionCost(initcode,result.execResult.executionGasUsed,true),exception:result.execResult.exceptionError?.error??null,address:result.createdAddress?.toString()});assert.equal(result.execResult.exceptionError,undefined,name+' deployment');const code=await evm.stateManager.getCode(result.createdAddress);deployments.at(-1).actualRuntimeKeccak=Buffer.from(keccak_256(code)).toString('hex');assert.equal(Buffer.from(code).length,artifact(name).evm.deployedBytecode.object.length/2);return result.createdAddress;}
for(const name of build.names)modules.push(await deploy(name));
const messageStack=[];let moduleCalls=[];let messageFailures=[];
evm.events.on('beforeMessage',m=>{messageStack.push({address:m.to?.toString(),inputBytes:m.data.length,gasForwarded:m.gasLimit.toString()});});
evm.events.on('afterMessage',result=>{const m=messageStack.pop();if(m&&result.execResult.exceptionError)messageFailures.push({...m,error:result.execResult.exceptionError.error});if(m && modules.some(a=>a.toString()===m.address))moduleCalls.push({...m,executionGas:result.execResult.executionGasUsed.toString(),outputBytes:result.execResult.returnValue.length});});
const addressWord=a=>Buffer.concat([Buffer.alloc(12),a.bytes]);
const configs=new Map();
async function configured(f){const key=digest(f.key).toString('hex');if(!configs.has(key))configs.set(key,await deploy('SpartanVerifier',Buffer.concat([digest(f.key),digest(f.setup),...modules.map(addressWord)])));return configs.get(key);}
const records=[];
async function run(name,f,expected='accept'){
 const to=await configured(f);await resetWarmth(evm,modules);
 const data=calldata(f);
 moduleCalls=[];messageFailures=[];const result=await evm.runCall({to,data,gasLimit});
 const record={name,moduleCalls,messageFailures,coordinatorExecutionGas:(result.execResult.executionGasUsed-moduleCalls.reduce((sum,c)=>sum+BigInt(c.executionGas),0n)).toString(),accepted:!result.execResult.exceptionError,executionGas:result.execResult.executionGasUsed.toString(),exception:result.execResult.exceptionError?.error??null,returnHex:Buffer.from(result.execResult.returnValue).toString('hex'),calldataBytes:data.length,transaction:transactionCost(data,result.execResult.executionGasUsed)};records.push(record);
 if(expected==='measure'){}else if(expected==='reject')assert.equal(record.exception,'revert',name);else {assert.equal(record.exception,null,name);assert.deepEqual(moduleCalls.map(c=>c.address),modules.map(a=>a.toString()),name+' all four same-call stages');}
 return Buffer.from(result.execResult.returnValue);
}

const f=fixture('norm');const provenance=JSON.parse(fs.readFileSync('test/spartan/norm/provenance.json'));
for(const [name,value] of Object.entries(f))assert.equal(hash(value),provenance.sha256[name+'.bin']);
await run('norm-30m',f,'measure');
if(!records.at(-1).accepted){assert(records.at(-1).messageFailures.some(x=>x.error==='out of gas'),'30M failure must be resource-related before diagnostic retry');gasLimit=200_000_000n;await run('norm-200m-diagnostic',f);for(let j=0;j<2;j++)assert.equal(records[0].moduleCalls[j].executionGas,records[1].moduleCalls[j].executionGas,'completed stage cost independent of preceding failed call');}
const accepted=records.at(-1);assert.equal(at(Buffer.from(accepted.returnHex,'hex'),0),1n);
const bad=Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.from(v)]));bad.inputs[0]^=1;await run('norm-changed-public-input',bad,'reject');
const badProof=Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.from(v)]));badProof.proof[85]^=1;await run('norm-changed-outer-proof',badProof,'reject');
const transaction=transactionCost(calldata(f),accepted.executionGas);
fs.mkdirSync('evidence/norm',{recursive:true});fs.writeFileSync('evidence/norm/results.json',JSON.stringify({scope:provenance.scope,settings:build.settings,node:process.version,sourceHashes:Object.fromEntries(Object.entries(build.sources).map(([name,v])=>[name,hash(Buffer.from(v.content))])),codeHashes:build.hashes,runtimeBytes:Object.fromEntries([...build.names,'SpartanVerifier'].map(n=>[n,artifact(n).evm.deployedBytecode.object.length/2])),deployments,records,transaction,warmness:'Prague, precompiles05/06/07/08/09 warm; module accounts cold then warmed by EXTCODEHASH; execution measured by runCall excludes tx processing',provenance},null,2)+'\n');console.log(JSON.stringify({deployments,records:records.map(({returnHex,...r})=>r),transaction},null,2));
