// SPDX-License-Identifier: MIT

pragma solidity ^0.8.21;

import {TestBase} from "./TestBase.sol";
import {Jolt} from "../../src/reference/JoltTypes.sol";
import {GrandProductArgument} from "../../src/reference/JoltVerifier.sol";
import {Transcript, FiatShamirTranscript} from "../../src/subprotocols/FiatShamirTranscript.sol";
import {Fr, FrLib, MODULUS} from "../../src/reference/Fr.sol";
import {UniPoly, UniPolyLib} from "../../src/reference/UniPoly.sol";
import {GrandProductArgumentGasWrapper} from "../GrandProductArgumentGasWrapper.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

contract TestGrandProduct is TestBase {
    using FiatShamirTranscript for Transcript;


    function testValidGrandProductProof() public {
        // Inits the transcript with the same string label as the rust code
        Transcript memory transcript = FiatShamirTranscript.new_transcript("test_transcript", 4);

        (Jolt.BatchedGrandProductProof memory proof, uint256[] memory claims, uint256[] memory r) = getProofData();

        Fr[] memory claims_fr;
        assembly ("memory-safe") {
            claims_fr := claims
        }

        Fr[] memory verifierRGrandProduct = GrandProductArgument.verify(proof, claims_fr, transcript);

        for (uint256 i = 0; i < verifierRGrandProduct.length; i++) {
            assertTrue(r[i] == Fr.unwrap(verifierRGrandProduct[i]));
        }
    }

    function testGasGrandProductVerify() public {
        Transcript memory transcript = FiatShamirTranscript.new_transcript("test_transcript", 4);

        (Jolt.BatchedGrandProductProof memory proof, uint256[] memory claims_fr,) = getProofData();

        Fr[] memory claims;
        assembly ("memory-safe") {
            claims := claims_fr
        }

        GrandProductArgumentGasWrapper wrapper = new GrandProductArgumentGasWrapper();

        wrapper.verify(proof, claims, transcript);

    }
}
