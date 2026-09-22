import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import solc from 'solc';
import { blake2b } from '@noble/hashes/blake2.js';
import { Common, Hardfork, Mainnet } from '@ethereumjs/common';
import { createEVM, EVMError } from '@ethereumjs/evm';
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const paths=['BlakeTranscript.sol','SpartanInputs.sol','SpartanAlgebra.sol','SpartanPublicOpening.sol'];
const sources=Object.fromEntries(paths.map(p=>[p,{content:fs.readFileSync('contracts/'+p,'utf8')}]));
const settings={viaIR:true,optimizer:{enabled:true,runs:200},evmVersion:'prague',outputSelection:{'*':{'*':['evm.deployedBytecode','evm.methodIdentifiers'],'':['ast']}}};
const compiled=JSON.parse(solc.compile(JSON.stringify({language:'Solidity',sources,settings})));
for(const e of compiled.errors??[])if(e.severity==='error')throw Error(e.formattedMessage);
const contract=compiled.contracts['SpartanPublicOpening.sol'].SpartanPublicOpeningBoundary;
const unconfigured=Buffer.from(contract.evm.deployedBytecode.object,'hex');
assert(unconfigured.length<=24576,'EIP-170 deployed code-size limit');
const refs=contract.evm.deployedBytecode.immutableReferences;
const ast=compiled.sources['SpartanPublicOpening.sol'].ast.nodes.find(n=>n.nodeType==='ContractDefinition'&&n.name==='SpartanPublicOpeningBoundary');
const names=Object.fromEntries(ast.nodes.filter(n=>n.nodeType==='VariableDeclaration').map(n=>[n.id,n.name]));
assert.deepEqual(Object.keys(refs).map(x=>names[x]).sort(),['expectedKey','expectedSetup']);
function code(keyId,setupId){const b=Buffer.from(unconfigured);for(const [id,positions]of Object.entries(refs))for(const p of positions){assert.equal(p.length,32);(names[id]==='expectedKey'?keyId:setupId).copy(b,p.start);}return b;}
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const le=(n,len=32)=>Buffer.from(BigInt(n).toString(16).padStart(len*2,'0'),'hex').reverse();
const selector=Buffer.from(contract.evm.methodIdentifiers['checkIncompleteWithPublicOpening(bytes,bytes,bytes,bytes,uint256[4][2])'],'hex');
function calldata(args,g2){let offset=args.length*32+256;const heads=[],tails=[];for(const arg of args){heads.push(word(offset));const tail=Buffer.concat([word(arg.length),arg,Buffer.alloc((32-arg.length%32)%32)]);tails.push(tail);offset+=tail.length;}return Buffer.concat([selector,...heads,g2,...tails]);}
let fault=null;const records=[];const gasLimit=30_000_000n;
async function execute(name,runtime,data,reject=false){const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});for(const a of [5,6,7,8,9])evm.journal.addAlwaysWarmAddress(a.toString(16).padStart(40,'0'));const calls={};for(const a of [6,7,8]){const key=a.toString(16).padStart(40,'0');const original=evm.precompiles.get(key);assert(original);evm.precompiles.set(key,input=>{calls[a]=(calls[a]??0)+1;if(fault?.address===a)return {executionGasUsed:0n,returnValue:fault.bytes,...(fault.fail?{exceptionError:new EVMError(EVMError.errorMessages.REVERT)}:{})};return original(input);});}const res=await evm.runCode({code:runtime,data,gasLimit});records.push({name,calls,accepted:!res.exceptionError,executionGas:res.executionGasUsed.toString(),calldataBytes:data.length,returnHex:Buffer.from(res.returnValue).toString('hex'),exception:res.exceptionError?.error??null});if(reject)assert.equal(res.exceptionError?.error,'revert',name);else assert.equal(res.exceptionError,undefined,name+': '+res.exceptionError);return Buffer.from(res.returnValue);}

function fixture(name){const dir='test/spartan/'+name;return Object.fromEntries(['key','setup','inputs','proof','g2-affine'].map(k=>[k,fs.readFileSync(dir+'/'+k+'.bin')]));}
const wordAt=(b,at)=>BigInt('0x'+b.subarray(at,at+32).toString('hex'));
function vector(b,at){return Array.from({length:Number(wordAt(b,at))},(_,j)=>wordAt(b,at+32*(j+1)));}
function decode(b){const root=Number(wordAt(b,0));const get=j=>wordAt(b,root+32*j);const outer=root+Number(get(0));const inner=root+Number(get(6));return {
 values:[...vector(b,outer+Number(wordAt(b,outer))),wordAt(b,outer+32),get(1),get(2),get(3),get(4),get(5),...vector(b,inner+Number(wordAt(b,inner))),wordAt(b,inner+32)],
 state:b.subarray(root+224,root+256).toString('hex'),publicState:b.subarray(root+256,root+288).toString('hex'),
 };}
const clone=f=>Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.from(v)]));
async function run(name,f,reject=false){return execute(name,code(digest(f.key),digest(f.setup)),calldata([f.key,f.setup,f.inputs,f.proof],f['g2-affine']),reject);}
for(const name of ['toy','empty','empty-public']){
 const f=fixture(name);const decoded=decode(await run(name,f));
 const file='test/spartan/'+name+'/algebra.json';
 if(!fs.existsSync(file))throw Error('Native checkpoint required: '+file);
 const native=JSON.parse(fs.readFileSync(file));
 assert.deepEqual(decoded.values,native.values_le.filter((_,i)=>i<6 || i>=13).map(x=>BigInt('0x'+Buffer.from(x,'hex').reverse().toString('hex'))),name+' all native claims/challenges');
 assert.equal(decoded.state,native.post_inner_state,name+' native post-inner state');
 assert.equal(decoded.publicState,native.post_public_state,name+' native post-public state');
}
const toy=fixture('toy');
{const f=fixture('empty');le(1).copy(f.proof,1398);await run('public-pairing-now-rejected',f,true);}
for(const [name,offset]of [['outer-coefficient',85],['outer-terminal-evaluation',278],['public-column',374],['public-fold-evaluation',822]]){
 const f=clone(toy);f.proof[offset]^=1;await run(name,f,true);
}
// Inner terminal and pairing obligations have deliberately not been discharged.
for(const [name,offset]of [['inner-coefficient-pending-terminal',1495],['private-sparse-unchecked',1689]]){
 const f=clone(toy);f.proof[offset]^=1;await run(name,f);
}

const Q=21888242871839275222246405745257275088696311157297823662689037894645226208583n;
for(const name of ['swap-x','swap-y','negative-sign','coordinate-alias','infinity','swap-setup-points']){
 const f=clone(toy);const g=f['g2-affine'];
 if(name==='swap-x' || name==='swap-y'){const at=name==='swap-x'?0:64;const a=Buffer.from(g.subarray(at,at+32));g.copy(g,at,at+32,at+64);a.copy(g,at+32);}
 if(name==='negative-sign')for(const at of [64,96]){const y=wordAt(g,at);word(y===0n?0n:Q-y).copy(g,at);}
 if(name==='coordinate-alias')word(Q).copy(g,0);
 if(name==='infinity')g.fill(0,0,128);
 if(name==='swap-setup-points'){const a=Buffer.from(g.subarray(0,128));g.copy(g,0,128,256);a.copy(g,128);}
 await run('g2-'+name,f,true);
}
// Even identity G1 pairing terms must validate the actual supplied G2 point.
{const f=fixture('empty');const g=f['g2-affine'];word(wordAt(g,96)+1n).copy(g,96);await run('g2-offcurve-with-identity-g1',f,true);assert.equal(records.at(-1).calls[8],1);}
{const f=fixture('empty');fs.readFileSync('test/spartan/empty/non-subgroup-compressed.bin').copy(f.setup,96);fs.readFileSync('test/spartan/empty/non-subgroup-affine.bin').copy(f['g2-affine'],0);digest(f.setup).copy(f.key,348);digest(f.key).copy(f.proof,20);await run('authenticated-non-subgroup-setup',f,true);assert.equal(records.at(-1).calls[8],1);}
for(const address of [6,7,8])for(const kind of ['failed','empty','short','long','malformed']){
 const size=address===8?32:64;
 fault={address,fail:kind==='failed',bytes:Buffer.alloc(kind==='empty'?0:kind==='short'?size-1:kind==='long'?size+1:size)};
 if(kind==='malformed')word(2).copy(fault.bytes,0);
 await run('precompile-'+address+'-'+kind,toy,true);
 assert(records.at(-1).calls[address]>0);
 fault=null;
}
fs.mkdirSync('evidence/public-opening',{recursive:true});fs.writeFileSync('evidence/public-opening/results.json',JSON.stringify({scope:'public HyperKZG checked, sparse/GKR/witness and remaining4PCS pending',settings,node:process.version,solc:solc.version(),sourceHashes:Object.fromEntries(paths.map(p=>[p,sha(Buffer.from(sources[p].content))])),runtimeSha256:sha(unconfigured),runtimeBytes:unconfigured.length,immutableReferences:{refs,names},selector:selector.toString('hex'),prewarmedAddresses:['0x05 MODEXP','0x06 ADD','0x07 MUL','0x08 PAIRING','0x09 Blake2F'],gasLimit:gasLimit.toString(),records},null,2)+'\n');fs.writeFileSync('evidence/public-opening/runtime.hex',unconfigured.toString('hex')+'\n');console.log(JSON.stringify(records,null,2));
