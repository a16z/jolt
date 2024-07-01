pragma solidity ^0.8.21;

import {TestBase} from "./base/TestBase.sol";
import {FiatShamirTranscript, Transcript} from "../src/subprotocols/FiatShamirTranscript.sol";

import "forge-std/console.sol";

contract TestTranscript is TestBase {
    using FiatShamirTranscript for Transcript;

    uint256 constant PRIME = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001;

    function testTranscriptIntegration() public {
        // The values are loaded from a rust run which runs the various function analogues
        TranscriptExampleValues memory vals = getTranscriptExample();

        // We run a series of tests, in sequence two appends per type then pulling a scalar and vector
        // While this isn't a combinatorial test, the sequence gives us good assurances that the hashes work out
        bytes32 start_string = "test_transcript";
        Transcript memory transcript = FiatShamirTranscript.new_transcript(start_string, 3);

        // First the u64 test
        for (uint256 i = 0; i < vals.usizes.length; i++) {
            transcript.append_u64(vals.usizes[i]);
        }
        assertEq(vals.expectedScalarResponses[0], transcript.challenge_scalar(PRIME));
        array_eq(vals.expectedVectorResponses[0], transcript.challenge_scalars(4, PRIME));

        // Next check the scalar append
        for (uint256 i = 0; i < vals.scalars.length; i++) {
            transcript.append_scalar(vals.scalars[i]);
        }
        assertEq(vals.expectedScalarResponses[1], transcript.challenge_scalar(PRIME));
        array_eq(vals.expectedVectorResponses[1], transcript.challenge_scalars(4, PRIME));

        // Next check the vector append
        for (uint256 i = 0; i < vals.scalarArrays.length; i++) {
            transcript.append_vector(vals.scalarArrays[i]);
        }
        assertEq(vals.expectedScalarResponses[2], transcript.challenge_scalar(PRIME));
        array_eq(vals.expectedVectorResponses[2], transcript.challenge_scalars(4, PRIME));

        // Next check the points append
        for (uint256 i = 0; i < vals.points.length; i += 2) {
            FiatShamirTranscript.append_point(transcript, vals.points[i], vals.points[i + 1]);
        }
        assertEq(vals.expectedScalarResponses[3], transcript.challenge_scalar(PRIME));
        array_eq(vals.expectedVectorResponses[3], transcript.challenge_scalars(4, PRIME));

        // Next check the point vector append
        for (uint256 i = 0; i < vals.pointArrays.length; i++) {
            FiatShamirTranscript.append_points(transcript, vals.pointArrays[i]);
        }
        assertEq(vals.expectedScalarResponses[4], transcript.challenge_scalar(PRIME));
        array_eq(vals.expectedVectorResponses[4], transcript.challenge_scalars(4, PRIME));

        // Next check the bytes vector append
        for (uint256 i = 0; i < vals.bytesExamples.length; i++) {
            FiatShamirTranscript.append_bytes(transcript, vals.bytesExamples[i]);
        }
        assertEq(vals.expectedScalarResponses[5], transcript.challenge_scalar(PRIME));
        array_eq(vals.expectedVectorResponses[5], transcript.challenge_scalars(4, PRIME));
    }
}
