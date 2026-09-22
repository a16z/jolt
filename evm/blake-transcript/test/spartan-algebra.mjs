import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import solc from 'solc';
import { blake2b } from '@noble/hashes/blake2.js';
import { Common, Hardfork, Mainnet } from '@ethereumjs/common';
import { createEVM } from '@ethereumjs/evm';
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const paths=['BlakeTranscript.sol','SpartanInputs.sol','SpartanAlgebra.sol'];
const sources=Object.fromEntries(paths.map(p=>[p,{content:fs.readFileSync('contracts/'+p,'utf8')}]));
const settings={viaIR:true,optimizer:{enabled:true,runs:200},evmVersion:'prague',outputSelection:{'*':{'*':['evm.deployedBytecode','evm.methodIdentifiers'],'':['ast']}}};
const compiled=JSON.parse(solc.compile(JSON.stringify({language:'Solidity',sources,settings})));
for(const e of compiled.errors??[])if(e.severity==='error')throw Error(e.formattedMessage);
const contract=compiled.contracts['SpartanAlgebra.sol'].SpartanIncompleteAlgebraBoundary;
const unconfigured=Buffer.from(contract.evm.deployedBytecode.object,'hex');
assert(unconfigured.length<=24576,'EIP-170 deployed code-size limit');
const refs=contract.evm.deployedBytecode.immutableReferences;
const ast=compiled.sources['SpartanAlgebra.sol'].ast.nodes.find(n=>n.nodeType==='ContractDefinition'&&n.name==='SpartanIncompleteAlgebraBoundary');
const names=Object.fromEntries(ast.nodes.filter(n=>n.nodeType==='VariableDeclaration').map(n=>[n.id,n.name]));
assert.deepEqual(Object.keys(refs).map(x=>names[x]).sort(),['expectedKey','expectedSetup']);
function code(keyId,setupId){const b=Buffer.from(unconfigured);for(const [id,positions]of Object.entries(refs))for(const p of positions){assert.equal(p.length,32);(names[id]==='expectedKey'?keyId:setupId).copy(b,p.start);}return b;}
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const le=(n,len=32)=>Buffer.from(BigInt(n).toString(16).padStart(len*2,'0'),'hex').reverse();
const selector=Buffer.from(contract.evm.methodIdentifiers['checkIncompleteAlgebra(bytes,bytes,bytes,bytes)'],'hex');
function calldata(args){let offset=args.length*32;const heads=[],tails=[];for(const arg of args){heads.push(word(offset));const tail=Buffer.concat([word(arg.length),arg,Buffer.alloc((32-arg.length%32)%32)]);tails.push(tail);offset+=tail.length;}return Buffer.concat([selector,...heads,...tails]);}
const records=[];const gasLimit=30_000_000n;
async function execute(name,runtime,data,reject=false){const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});for(const a of [5,9])evm.journal.addAlwaysWarmAddress(a.toString(16).padStart(40,'0'));const res=await evm.runCode({code:runtime,data,gasLimit});records.push({name,accepted:!res.exceptionError,executionGas:res.executionGasUsed.toString(),calldataBytes:data.length,returnHex:Buffer.from(res.returnValue).toString('hex'),exception:res.exceptionError?.error??null});if(reject)assert.equal(res.exceptionError?.error,'revert',name);else assert.equal(res.exceptionError,undefined,name+': '+res.exceptionError);return Buffer.from(res.returnValue);}

function fixture(name){const dir='test/spartan/'+name;return Object.fromEntries(['key','setup','inputs','proof'].map(k=>[k,fs.readFileSync(dir+'/'+k+'.bin')]));}
const wordAt=(b,at)=>BigInt('0x'+b.subarray(at,at+32).toString('hex'));
function vector(b,at){return Array.from({length:Number(wordAt(b,at))},(_,j)=>wordAt(b,at+32*(j+1)));}
function decode(b){const root=Number(wordAt(b,0));const get=j=>wordAt(b,root+32*j);const outer=root+Number(get(0));const inner=root+Number(get(6));const pending=root+Number(get(7));return {
 values:[...vector(b,outer+Number(wordAt(b,outer))),wordAt(b,outer+32),get(1),get(2),get(3),...vector(b,pending+Number(wordAt(b,pending+32))),wordAt(b,pending+64),get(4),get(5),...vector(b,inner+Number(wordAt(b,inner))),wordAt(b,inner+32)],
 state:b.subarray(root+256,root+288).toString('hex'),pendingOffset:wordAt(b,pending),publicState:b.subarray(root+288,root+320).toString('hex'),
 };}
const clone=f=>Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.from(v)]));
async function run(name,f,reject=false){return execute(name,code(digest(f.key),digest(f.setup)),calldata([f.key,f.setup,f.inputs,f.proof]),reject);}
for(const name of ['toy','empty','empty-public']){
 const f=fixture(name);const decoded=decode(await run(name,f));
 const file='test/spartan/'+name+'/algebra.json';
 if(!fs.existsSync(file))throw Error('Native checkpoint required: '+file);
 const native=JSON.parse(fs.readFileSync(file));
 assert.deepEqual(decoded.values,native.values_le.map(x=>BigInt('0x'+Buffer.from(x,'hex').reverse().toString('hex'))),name+' all native claims/challenges');
 assert.equal(decoded.state,native.post_inner_state,name+' native post-inner state');
 assert.equal(decoded.publicState,native.post_public_state,name+' native post-public state');
}
const toy=fixture('toy');
{const f=fixture('empty');le(1).copy(f.proof,1398);await run('public-pairing-pending-native-rejects',f);}
for(const [name,offset]of [['outer-coefficient',85],['outer-terminal-evaluation',278],['public-column',374],['public-fold-evaluation',822]]){
 const f=clone(toy);f.proof[offset]^=1;await run(name,f,true);
}
// Inner terminal and pairing obligations have deliberately not been discharged.
for(const [name,offset]of [['inner-coefficient-pending-terminal',1495],['private-sparse-unchecked',1689]]){
 const f=clone(toy);f.proof[offset]^=1;await run(name,f);
}
fs.mkdirSync('evidence/algebra',{recursive:true});fs.writeFileSync('evidence/algebra/results.json',JSON.stringify({scope:'incomplete outer/public/inner algebra; public KZG pairing and sparse/witness relations pending',settings,node:process.version,solc:solc.version(),sourceHashes:Object.fromEntries(paths.map(p=>[p,sha(Buffer.from(sources[p].content))])),runtimeSha256:sha(unconfigured),runtimeBytes:unconfigured.length,immutableReferences:{refs,names},selector:selector.toString('hex'),fixtureHashes:Object.fromEntries(['toy','empty','empty-public'].map(n=>[n,Object.fromEntries(Object.entries(fixture(n)).map(([k,v])=>[k,sha(v)]))])),prewarmedAddresses:['0x05 MODEXP','0x09 Blake2F'],gasLimit:gasLimit.toString(),context:'fresh Prague EVM; immutables patched to constructor-equivalent policy; no deployment; excludes intrinsic/calldata transaction gas',records},null,2)+'\n');
fs.writeFileSync('evidence/algebra/runtime.hex',unconfigured.toString('hex')+'\n');
console.log(JSON.stringify(records,null,2));
