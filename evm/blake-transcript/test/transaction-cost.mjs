// Prague transaction charges, EIP-7623 and EIP-3860; runCall measures execution separately.
// https://eips.ethereum.org/EIPS/eip-7623
export function transactionCost(data,executionGas,creation=false){
 const zeroBytes=[...data].filter(x=>x===0).length;
 const nonzeroBytes=data.length-zeroBytes;
 const tokens=zeroBytes+4*nonzeroBytes;
 const intrinsicGas=21000+4*tokens+(creation?32000+2*Math.ceil(data.length/32):0);
 const floorGas=21000+10*tokens;
 const executionAndIntrinsic=BigInt(executionGas)+BigInt(intrinsicGas);
 return {bytes:data.length,zeroBytes,nonzeroBytes,intrinsicGas,floorGas,
  estimatedTransactionGas:(executionAndIntrinsic>BigInt(floorGas)?executionAndIntrinsic:BigInt(floorGas)).toString()};
}
