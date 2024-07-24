// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {Transcript, FiatShamirTranscript} from "./FiatShamirTranscript.sol";
import {MODULUS, Fr, FrLib} from "../reference/Fr.sol";
import {SumcheckVerifier, SumcheckInstanceProof} from "./SumcheckVerifier.sol";
import {HyperKZG, HyperKZGProof} from "./HyperKZG.sol";

struct SpartanProof {
    SumcheckInstanceProof outer;
    uint256 outerClaim0;
    uint256 outerClaim1;
    uint256 outerClaim2;
    SumcheckInstanceProof inner;
    uint256[] claimedEvals;
    HyperKZGProof proof;
}

contract SpartanVerifier is HyperKZG {
    using FiatShamirTranscript for Transcript;
    using FrLib for Fr;
    using SumcheckVerifier for SumcheckInstanceProof;

    function verifySpartanR1CS(SpartanProof memory proof) internal pure {}
}
