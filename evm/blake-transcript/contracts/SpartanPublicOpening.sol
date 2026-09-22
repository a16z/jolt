// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {SpartanAlgebra} from "./SpartanAlgebra.sol";

/// @dev Native HyperKZG verify_batch, with authenticated affine G2 auxiliaries.
library SpartanKzgPairing {
    uint256 constant R = SpartanInputs.R;
    uint256 constant Q = SpartanInputs.Q;
    error G2Encoding();
    error CurveCall(uint256 precompile);
    error PairingEquation();

    function pointResult(SpartanInputs.G1 memory p) private pure {
        if(p.x>=Q || p.y>=Q)revert CurveCall(0);
        if(p.x==0 && p.y==0)return;
        if(mulmod(p.y,p.y,Q)!=addmod(mulmod(mulmod(p.x,p.x,Q),p.x,Q),3,Q))revert CurveCall(0);
    }
    function add(SpartanInputs.G1 memory a,SpartanInputs.G1 memory b) private view returns(SpartanInputs.G1 memory c) {
        uint256[4] memory input=[a.x,a.y,b.x,b.y];
        bool ok;uint256 size;
        assembly ("memory-safe") {ok:=staticcall(gas(),6,input,128,c,64) size:=returndatasize()}
        if(!ok || size!=64)revert CurveCall(6);
        pointResult(c);
    }
    function mul(SpartanInputs.G1 memory a,uint256 scalar) private view returns(SpartanInputs.G1 memory c) {
        uint256[3] memory input=[a.x,a.y,scalar];
        bool ok;uint256 size;
        assembly ("memory-safe") {ok:=staticcall(gas(),7,input,96,c,64) size:=returndatasize()}
        if(!ok || size!=64)revert CurveCall(7);
        pointResult(c);
    }
    function neg(SpartanInputs.G1 memory a) private pure returns(SpartanInputs.G1 memory) {
        return SpartanInputs.G1(a.x,a.y==0?0:Q-a.y);
    }
    function leWord(bytes calldata b,uint256 at) private pure returns(uint256 v) {
        for(uint256 j;j<32;++j)v|=uint256(uint8(b[at+j]))<<(8*j);
    }
    // Auxiliary order is EIP-197: x.c1,x.c0,y.c1,y.c0. Compression is arkworks LE c0,c1.
    function bindG2(bytes calldata setup,uint256 at,uint256[4] calldata p) private pure {
        for(uint256 j;j<4;++j)if(p[j]>=Q)revert G2Encoding();
        uint256 encoded=leWord(setup,at+32);
        uint256 flags=encoded>>254;
        if(flags!=0 && flags!=2)revert G2Encoding();
        if(p[1]!=leWord(setup,at) || p[0]!=(encoded&((uint256(1)<<254)-1)))revert G2Encoding();
        if(p[0]==0 && p[1]==0 && p[2]==0 && p[3]==0)revert G2Encoding();
        uint256 negativeImag=p[2]==0?0:Q-p[2];
        uint256 negativeReal=p[3]==0?0:Q-p[3];
        bool larger=p[2]>negativeImag || (p[2]==negativeImag && p[3]>negativeReal);
        if(larger!=(flags==2))revert G2Encoding();
    }
    function check(SpartanInputs.Decoded memory d,SpartanAlgebra.PendingKzg memory query,
        bytes calldata setup,uint256[4][2] calldata g2) internal view {
        bindG2(setup,96,g2[0]);bindG2(setup,160,g2[1]);
        uint256 n=query.point.length;
        uint256 evalStart=query.scalarOffset;
        uint256 power=1;
        SpartanInputs.G1 memory combined;
        for(uint256 j;j<n;++j) {
            combined=add(combined,mul(j==0?query.commitment:d.points[query.pointOffset+j-1],power));
            power=mulmod(power,query.polynomialBatchChallenge,R);
        }
        uint256 rho=query.openingBatchChallenge;
        uint256[3] memory weights=[uint256(1),rho,mulmod(rho,rho,R)];
        uint256 r=query.foldChallenge;
        uint256[3] memory points=[r,R-r,mulmod(r,r,R)];
        uint256 scale=addmod(addmod(weights[0],weights[1],R),weights[2],R);
        SpartanInputs.G1 memory lhs=mul(combined,scale);
        SpartanInputs.G1 memory rhs;
        uint256 evaluation;
        for(uint256 row;row<3;++row) {
            power=1;uint256 rowEvaluation;
            for(uint256 j;j<n;++j) {
                rowEvaluation=addmod(rowEvaluation,mulmod(d.scalars[evalStart+row*n+j],power,R),R);
                power=mulmod(power,query.polynomialBatchChallenge,R);
            }
            evaluation=addmod(evaluation,mulmod(weights[row],rowEvaluation,R),R);
            SpartanInputs.G1 memory witness=d.points[query.pointOffset+n-1+row];
            lhs=add(lhs,mul(witness,mulmod(points[row],weights[row],R)));
            rhs=add(rhs,mul(witness,weights[row]));
        }
        lhs=add(lhs,neg(mul(d.setupG1,evaluation)));rhs=neg(rhs);
        uint256[12] memory input=[lhs.x,lhs.y,g2[0][0],g2[0][1],g2[0][2],g2[0][3],rhs.x,rhs.y,g2[1][0],g2[1][1],g2[1][2],g2[1][3]];
        uint256[1] memory output;bool ok;uint256 size;
        assembly ("memory-safe") {ok:=staticcall(gas(),8,input,384,output,32) size:=returndatasize()}
        if(!ok || size!=32)revert CurveCall(8);
        if(output[0]!=1)revert PairingEquation();
    }
}

/// @notice Public PCS checked; sparse/GKR/witness obligations remain. NOT proof acceptance.
contract SpartanPublicOpeningBoundary {
    bytes32 public immutable expectedKey;
    bytes32 public immutable expectedSetup;
    struct IncompleteCheckpoint {
        SpartanAlgebra.Sumcheck outer;
        uint256[3] weights;
        uint256 publicContribution;
        uint256 innerInitialClaim;
        SpartanAlgebra.Sumcheck inner;
        bytes32 transcriptState;
        bytes32 postPublicState;
    }
    constructor(bytes32 keyId,bytes32 setupDigest){expectedKey=keyId;expectedSetup=setupDigest;}
    function checkIncompleteWithPublicOpening(bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof,uint256[4][2] calldata g2)
        external view returns(IncompleteCheckpoint memory) {
        SpartanInputs.Decoded memory d=SpartanInputs.decodeAuthenticated(expectedKey,expectedSetup,key,setup,inputs,proof);
        SpartanAlgebra.IncompleteCheckpoint memory c=SpartanAlgebra.checkIncomplete(d,key,setup,proof);
        SpartanKzgPairing.check(d,c.pendingPublicKzg,setup,g2);
        return IncompleteCheckpoint(c.outer,c.weights,c.publicContribution,c.innerInitialClaim,c.inner,c.transcriptState,c.postPublicState);
    }
}
