// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {BlakeHashSponge, Bn254WideBlake} from "./BlakeTranscript.sol";

/// @notice Partial algebra only. Pending openings and sparse relations are NOT verified.
library SpartanAlgebra {
    uint256 constant R = SpartanInputs.R;
    error OuterRelation();
    error PublicFold();
    error DegenerateChallenge();
    struct Sumcheck { uint256[] point; uint256 value; }
    struct PendingKzg {
        uint256 proofOffset;
        uint256[] point;
        uint256 evaluation;
        uint256 foldChallenge;
        uint256 polynomialBatchChallenge;
        uint256 openingBatchChallenge;
        SpartanInputs.G1 commitment;
        uint256 scalarOffset;
        uint256 pointOffset;
    }
    struct IncompleteCheckpoint {
        Sumcheck outer;
        uint256[3] weights;
        uint256 publicContribution;
        uint256 innerInitialClaim;
        Sumcheck inner;
        PendingKzg pendingPublicKzg;
        bytes32 transcriptState;
        bytes32 postPublicState;
    }
    function sub(uint256 a,uint256 b) internal pure returns(uint256) {return addmod(a,R-b,R);}
    function append(BlakeHashSponge.State memory t,uint256 value) internal view {
        Bn254WideBlake.append(t,abi.encodePacked(value));
    }
    function label(BlakeHashSponge.State memory t,bytes32 value) internal view {
        Bn254WideBlake.append(t,abi.encodePacked(value));
    }
    function values(BlakeHashSponge.State memory t,bytes32 name,uint256[] memory v) internal view {
        label(t,name|bytes32(v.length));
        for(uint256 j;j<v.length;++j)append(t,v[j]);
    }
    // Width determines native transcript length, not allocation or the degree bound.
    function sumcheck(SpartanInputs.Decoded memory d,uint256 start,uint256 tag,uint256 rounds,uint256 degree,uint256 claim)
        internal view returns(Sumcheck memory result) {
        result.point=new uint256[](rounds);
        for(uint256 j;j<rounds;++j) {
            uint256 at=start+j*degree;
            uint256 width=d.widths[tag+j];
            label(d.transcript,bytes32("sumcheck_poly")|bytes32(width));
            uint256 linear=sub(claim,addmod(d.scalars[at],d.scalars[at],R));
            for(uint256 k;k<width;++k) {
                append(d.transcript,d.scalars[at+k]);
                if(k>0)linear=sub(linear,d.scalars[at+k]);
            }
            uint256 x=Bn254WideBlake.challenge(d.transcript);
            uint256 value;
            for(uint256 k=degree;k>1;--k)value=addmod(mulmod(value,x,R),d.scalars[at+k-1],R);
            value=addmod(mulmod(value,x,R),linear,R);
            claim=addmod(mulmod(value,x,R),d.scalars[at],R);
            result.point[j]=x;
        }
        result.value=claim;
    }
    function eq(uint256[] memory a,uint256[] memory b) internal pure returns(uint256 v) {
        v=1;
        for(uint256 j;j<a.length;++j)
            v=mulmod(v,addmod(mulmod(a[j],b[j],R),mulmod(sub(1,a[j]),sub(1,b[j]),R),R),R);
    }
    // EqPolynomial uses the first coordinate as the most significant index bit.
    function eqIndex(uint256[] memory point,uint256 index) internal pure returns(uint256 v) {
        v=1;
        for(uint256 j;j<point.length;++j) {
            uint256 x=point[j];
            v=mulmod(v,((index>>(point.length-1-j))&1)==1?x:sub(1,x),R);
        }
    }
    function little64(bytes calldata b,uint256 at) private pure returns(uint256 v) {
        for(uint256 j;j<8;++j)v|=uint256(uint8(b[at+j]))<<(8*j);
    }
    // Replays the production public opening transcript and checks binary folding.
    // The three-point batched KZG equation remains a typed pending obligation.
    function publicOpening(SpartanInputs.Decoded memory d,bytes calldata key,bytes calldata setup,bytes calldata proof,
        uint256[] memory rx,uint256 evalStart) private view returns(PendingKzg memory pending) {
        uint256 count=3*d.shape.p;
        uint256[] memory evaluations=new uint256[](count);
        for(uint256 j;j<count;++j)evaluations[j]=d.scalars[evalStart+j];
        label(d.transcript,bytes32("matrix-public-v2"));
        values(d.transcript,bytes32("public-columns"),evaluations);
        label(d.transcript,bytes32("public-selector"));
        uint256[] memory selector=new uint256[](d.shape.t);
        uint256 arity=d.shape.t+d.shape.r;
        pending.point=new uint256[](arity);
        for(uint256 j;j<selector.length;++j) {
            selector[j]=Bn254WideBlake.challenge(d.transcript);
            pending.point[j]=selector[j];
        }
        for(uint256 j;j<rx.length;++j)pending.point[selector.length+j]=rx[j];
        for(uint256 j;j<count;++j)pending.evaluation=addmod(pending.evaluation,mulmod(eqIndex(selector,j),evaluations[j],R),R);
        pending=opening(d,setup,proof,key[380:412],d.commitments[0],pending.point,pending.evaluation,
            SpartanInputs.OpeningCursor(d.publicOpeningOffset,evalStart+count,1));
    }
    // Shared production HyperKZG transcript/folding owner for all five commitments.
    function opening(SpartanInputs.Decoded memory d,bytes calldata setup,bytes calldata proof,
        bytes calldata commitment,SpartanInputs.G1 memory decodedCommitment,uint256[] memory point,uint256 evaluation,
        SpartanInputs.OpeningCursor memory cursor) internal view returns(PendingKzg memory pending) {
        pending.point=point;pending.evaluation=evaluation;pending.commitment=decodedCommitment;
        uint256 arity=point.length;
        label(d.transcript,bytes32("hyperkzg-binary-bn254-v1"));
        append(d.transcript,little64(setup,48));
        Bn254WideBlake.append(d.transcript,setup[16:48]);
        append(d.transcript,little64(setup,56));
        Bn254WideBlake.append(d.transcript,setup[64:96]);
        Bn254WideBlake.append(d.transcript,setup[96:160]);
        Bn254WideBlake.append(d.transcript,setup[160:224]);
        Bn254WideBlake.append(d.transcript,commitment);
        values(d.transcript,bytes32("opening-point"),pending.point);
        append(d.transcript,pending.evaluation);
        uint256 at=cursor.wire;
        pending.proofOffset=at;pending.scalarOffset=cursor.scalar;pending.pointOffset=cursor.point;
        for(uint256 j;j<arity-1;++j) {Bn254WideBlake.append(d.transcript,proof[at:at+32]);at+=32;}
        uint256 r=Bn254WideBlake.challenge(d.transcript);
        if(r==0)revert DegenerateChallenge();
        pending.foldChallenge=r;
        uint256 vStart=cursor.scalar;
        for(uint256 j;j<arity;++j) {
            uint256 positive=d.scalars[vStart+j];
            uint256 negative=d.scalars[vStart+arity+j];
            uint256 next=j+1==arity?pending.evaluation:d.scalars[vStart+2*arity+j+1];
            uint256 coordinate=pending.point[arity-1-j];
            uint256 rhs=addmod(mulmod(mulmod(r,sub(1,coordinate),R),addmod(positive,negative,R),R),mulmod(coordinate,sub(positive,negative),R),R);
            if(mulmod(addmod(r,r,R),next,R)!=rhs)revert PublicFold();
        }
        for(uint256 j;j<3*arity;++j)append(d.transcript,d.scalars[vStart+j]);
        at+=96*arity;
        pending.polynomialBatchChallenge=Bn254WideBlake.challenge(d.transcript);
        for(uint256 j;j<3;++j){Bn254WideBlake.append(d.transcript,proof[at:at+32]);at+=32;}
        pending.openingBatchChallenge=Bn254WideBlake.challenge(d.transcript);
    }
    function checkIncomplete(SpartanInputs.Decoded memory d,bytes calldata key,bytes calldata setup,bytes calldata proof)
        internal view returns(IncompleteCheckpoint memory c) {
        c.outer=sumcheck(d,0,0,d.shape.r,3,0);
        uint256 at=3*d.shape.r;
        uint256 a=d.scalars[at];uint256 b=d.scalars[at+1];uint256 z=d.scalars[at+2];
        if(c.outer.value!=mulmod(eq(d.tau,c.outer.point),sub(mulmod(a,b,R),z),R))revert OuterRelation();
        label(d.transcript,bytes32("outer-evaluations")|bytes32(uint256(3)));
        append(d.transcript,a);append(d.transcript,b);append(d.transcript,z);
        for(uint256 j;j<3;++j)c.weights[j]=Bn254WideBlake.challenge(d.transcript);
        at+=3;
        c.pendingPublicKzg=publicOpening(d,key,setup,proof,c.outer.point,at);
        bytes memory publicState=BlakeHashSponge.peek(d.transcript);
        bytes32 publicCheckpoint;assembly ("memory-safe"){publicCheckpoint:=mload(add(publicState,32))}
        c.postPublicState=publicCheckpoint;
        for(uint256 matrix;matrix<3;++matrix) {
            uint256 v=d.scalars[at+matrix*d.shape.p];
            for(uint256 j=1;j<d.shape.p;++j)v=addmod(v,mulmod(d.scalars[at+matrix*d.shape.p+j],d.publicInputs[j-1],R),R);
            c.publicContribution=addmod(c.publicContribution,mulmod(c.weights[matrix],v,R),R);
        }
        c.innerInitialClaim=sub(addmod(addmod(mulmod(a,c.weights[0],R),mulmod(b,c.weights[1],R),R),mulmod(z,c.weights[2],R),R),c.publicContribution);
        label(d.transcript,bytes32("spartan-inner"));append(d.transcript,c.innerInitialClaim);
        c.inner=sumcheck(d,d.innerScalarOffset,d.shape.r,d.shape.w,2,c.innerInitialClaim);
        bytes memory state=BlakeHashSponge.peek(d.transcript);
        bytes32 checkpoint;assembly ("memory-safe"){checkpoint:=mload(add(state,32))}
        c.transcriptState=checkpoint;
    }
}

/// @notice Checks only a protocol prefix. NOT a proof acceptance API.
contract SpartanIncompleteAlgebraBoundary {
    bytes32 public immutable expectedKey;
    bytes32 public immutable expectedSetup;
    constructor(bytes32 keyId,bytes32 setupDigest){expectedKey=keyId;expectedSetup=setupDigest;}
    function checkIncompleteAlgebra(bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof)
        external view returns(SpartanAlgebra.IncompleteCheckpoint memory) {
        SpartanInputs.Decoded memory d=SpartanInputs.decodeAuthenticated(expectedKey,expectedSetup,key,setup,inputs,proof);
        return SpartanAlgebra.checkIncomplete(d,key,setup,proof);
    }
}
